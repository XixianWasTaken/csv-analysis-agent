"""Render computed fields using public answer requirements, without scoring data."""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

AnswerKind = Literal[
    "auto", "number", "integer", "number_or_string", "string", "boolean", "list", "dict"
]

_PUBLIC_NUMBERS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
]


def _public_precision(text: str) -> set[int]:
    values = re.findall(
        r"\b(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten) decimal places?\b",
        text.casefold(),
    )
    return {
        int(value) if value.isdigit() else _PUBLIC_NUMBERS.index(value)
        for value in values
    }


class AnswerSubmissionError(ValueError):
    """A result can be repaired using its public field location and requirement."""


class AnswerField(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: AnswerKind = "auto"
    decimals: int | None = Field(default=None, ge=0, le=20)
    preserve_precision: bool = False
    separator: str = Field(default=",", min_length=1, max_length=16)
    quoted: bool = False
    allow_null: bool = False
    items: list[AnswerField] | None = Field(default=None, min_length=1, max_length=100)
    item: AnswerField | None = None
    properties: dict[str, AnswerField] | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def validate_precision(self) -> AnswerField:
        if self.preserve_precision and self.decimals is not None:
            raise ValueError("Preserved precision cannot specify decimal places")
        return self

    def component(self, index: int) -> AnswerField:
        if self.items is not None and index < len(self.items):
            return self.items[index]
        return self.item or AnswerField(
            decimals=self.decimals,
            preserve_precision=self.preserve_precision,
            allow_null=self.allow_null,
        )

    def member(self, name: str) -> AnswerField:
        if self.properties is not None:
            member = self.properties[name]
            if (
                member.decimals is None
                and not member.preserve_precision
                and self.decimals is not None
            ):
                return member.model_copy(update={"decimals": self.decimals})
            return member
        return AnswerField(
            decimals=self.decimals,
            preserve_precision=self.preserve_precision,
            allow_null=self.allow_null,
        )


class AnswerFormat(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    fields: dict[str, AnswerField] = Field(min_length=1, max_length=100)
    alternatives: list[list[str]] | None = Field(
        default=None, min_length=2, max_length=10
    )

    @model_validator(mode="after")
    def validate_declarations(self) -> AnswerFormat:
        if any(not re.fullmatch(r"\w{1,128}", name) for name in self.fields):
            raise ValueError("Answer fields require names of 1–128 word characters")
        if self.alternatives is not None:
            if any(
                not names or len(names) != len(set(names))
                for names in self.alternatives
            ):
                raise ValueError(
                    "Each alternative requires unique nonempty field names"
                )
            if {name for names in self.alternatives for name in names} != set(
                self.fields
            ):
                raise ValueError(
                    "Alternative declarations must cover exactly the public fields"
                )
        return self

    def required_names(self, submitted: list[str], path: str) -> list[str]:
        if self.alternatives is None:
            return list(self.fields)
        if not any(set(submitted) == set(branch) for branch in self.alternatives):
            choices = " or ".join(", ".join(branch) for branch in self.alternatives)
            raise AnswerSubmissionError(
                f"{path}: submit one complete public alternative: {choices}"
            )
        return [name for name in self.fields if name in submitted]

    @classmethod
    def from_public_format(cls, text: str) -> AnswerFormat:
        if not text.strip() or len(text) > 65536:
            raise ValueError("Public answer format requires 1–65536 characters")
        pattern = r"@(\w+)\[([^\]\r\n]*)\]?"
        tags = re.findall(pattern, text)
        # Examples repeat declarations; the first template owns the field rule.
        templates: dict[str, str] = {}
        for name, placeholder in tags:
            templates.setdefault(name, placeholder)
        if not templates:
            raise ValueError("Public answer format requires @field[value] declarations")
        prose = re.sub(pattern, "", text)
        clauses = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n|\\n|;\s*|\s+while\s+", prose):
            start = 0
            for boundary in re.finditer(
                r"(?:,\s*(?:and\s+)?|\s+and\s+)(?=(?:where\s+)?[\"']?[\w.]+[\"']?\s+(?:is|are|should|can)\b)",
                sentence,
                flags=re.IGNORECASE,
            ):
                # Keep grouped subjects ("mean", "median", and "std" are ...)
                # together; split only after an earlier definition has begun.
                if re.search(
                    r"\b(is|are|should|can|represent)\b",
                    sentence[start : boundary.start()],
                    re.IGNORECASE,
                ):
                    clauses.append(sentence[start : boundary.start()])
                    start = boundary.end()
            clauses.append(sentence[start:])
        combined_clauses: list[str] = []
        for clause in clauses:
            if not clause.strip():
                continue
            if combined_clauses and re.match(
                r"\s*(?:both|they|these|each value|the value|the number|round(?:ed)?(?:\s+up)?)\b",
                clause,
                re.IGNORECASE,
            ):
                combined_clauses[-1] += " " + clause
            else:
                combined_clauses.append(clause)
        clauses = combined_clauses
        global_precision = " ".join(
            clause
            for clause in clauses
            if re.search(r"\b(all|each answer|each output)\b", clause, re.IGNORECASE)
            and "decimal place" in clause.lower()
        )
        local_prose: dict[str, str] = {}
        matches = list(re.finditer(pattern, text))
        for index, match in enumerate(matches):
            after = text[
                match.end() : matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            ]
            if after.lstrip().startswith(":"):
                local_prose.setdefault(match[1], after)

        def rule_for(
            placeholder: str,
            name: str = "",
            *,
            prefix: bool = False,
            default_rule: AnswerField | None = None,
        ) -> AnswerField:
            alias = placeholder.strip().strip("\"'")
            raw_parts = [part.strip() for part in placeholder.split(",")]
            composite = (
                not alias.startswith("{")
                and len(raw_parts) > 1
                and all(
                    re.fullmatch(r"\w+|[\"']\w+[\"']|\.{3}|…", part)
                    for part in raw_parts
                )
            )
            terms = tuple(
                term for term in (alias, alias.replace("_", " "), name) if term
            )
            relevant = [
                clause
                for clause in clauses
                if any(
                    re.search(
                        r"(?<!\w)" + re.escape(term) + ("" if prefix else r"(?!\w)"),
                        clause,
                        re.IGNORECASE,
                    )
                    for term in terms
                )
            ]
            description = " ".join(relevant).lower()
            description += " " + local_prose.get(name, "").lower()
            if "decimal place" in alias.lower() or re.search(
                r"\b(number|integer|float|boolean|list|dictionary|dict|string|str)\b",
                alias,
                re.IGNORECASE,
            ):
                description += " " + alias.lower()
            kind: AnswerKind = "auto"
            if alias.startswith("{"):
                kind = "dict"
            elif (
                re.search(r"\blist\b", description)
                or "separated by commas" in description
            ):
                kind = "list"
            elif "dictionary" in description or re.search(r"\bdict\b", description):
                kind = "dict"
            elif "boolean" in description:
                kind = "boolean"
            elif "integer" in description:
                kind = "integer"
            elif re.search(r"\bor\s+(?:(?:the|a)\s+)?string\b", description):
                kind = "number_or_string"
            elif any(
                re.search(
                    r"(?<!\w)"
                    + re.escape(term)
                    + r"(?!\w)[\"']?\s+(?:is|are|should be)\s+(?:a\s+)?string\b",
                    description,
                    re.IGNORECASE,
                )
                for term in terms
            ):
                kind = "string"
            elif re.search(
                r"\b(numbers?|floats?|floating-point|decimal|real)\b", description
            ):
                kind = "number"
            elif re.search(r"\b(string|str)\b", description):
                kind = "string"
            if kind == "auto" and default_rule is not None:
                kind = default_rule.kind
            preserve_precision = bool(
                re.search(
                    r"\b(?:without rounding|no rounding|unrounded|do not round|not rounded)\b",
                    description,
                )
            )
            decimals = set() if preserve_precision else _public_precision(description)
            if not decimals and not preserve_precision:
                if default_rule is None:
                    decimals = _public_precision(global_precision)
                else:
                    preserve_precision = default_rule.preserve_precision
                    if default_rule.decimals is not None:
                        decimals = {default_rule.decimals}
            decimal_places = next(iter(decimals)) if len(decimals) == 1 else None
            item_rule = None
            fixed_items = None
            properties = None
            if kind == "dict" and alias.startswith("{"):
                try:
                    template = ast.parse(alias, mode="eval").body
                except (SyntaxError, ValueError):
                    template = None
                if isinstance(template, ast.Dict) and all(
                    isinstance(key, ast.Constant) and isinstance(key.value, str)
                    for key in template.keys
                ):
                    properties = {}
                    member_default = AnswerField(
                        decimals=decimal_places,
                        preserve_precision=preserve_precision,
                    )
                    for key, expression in zip(template.keys, template.values):
                        assert isinstance(key, ast.Constant) and isinstance(
                            key.value, str
                        )
                        if isinstance(expression, ast.Name):
                            member_rule = rule_for(
                                expression.id, default_rule=member_default
                            )
                        elif isinstance(expression, ast.Dict):
                            member_rule = rule_for(
                                ast.unparse(expression),
                                default_rule=member_default,
                            )
                        elif isinstance(expression, ast.Constant) and isinstance(
                            expression.value, str
                        ):
                            member_rule = rule_for(
                                json.dumps(expression.value),
                                default_rule=member_default,
                            )
                        else:
                            member_rule = member_default
                        properties[key.value] = member_rule
            if kind == "list":
                item_kind: AnswerKind = "auto"
                if re.search(r"\bstrings?\b", description):
                    item_kind = "string"
                elif "integer" in description:
                    item_kind = "integer"
                elif re.search(r"\b(numbers?|floats?)\b", description):
                    item_kind = "number"
                item_rule = AnswerField(
                    kind=item_kind,
                    decimals=decimal_places,
                    preserve_precision=preserve_precision,
                )
                cardinality = re.search(
                    r"\blist of (?:exactly )?(\d+|one|two|three|four|five|six|seven|eight|nine|ten) (?:strings?|numbers?|integers?|items?|elements?|values?)\b",
                    description,
                )
                if cardinality:
                    count = (
                        int(cardinality[1])
                        if cardinality[1].isdigit()
                        else _PUBLIC_NUMBERS.index(cardinality[1])
                    )
                    if count > 100:
                        raise ValueError(
                            "Public fixed lists support at most 100 components"
                        )
                    fixed_items = [item_rule] * count
            if composite:
                component_default = AnswerField(
                    kind=item_rule.kind if item_rule is not None else kind,
                    decimals=decimal_places,
                    preserve_precision=preserve_precision,
                )
                kind = "list"
                if any("..." in part or "…" in part for part in raw_parts):
                    stem = re.sub(r"\d+$", "", raw_parts[0].strip("\"'"))
                    item_rule = rule_for(
                        stem, prefix=True, default_rule=component_default
                    )
                    if item_rule.kind == "list":
                        item_rule = item_rule.component(0)
                    if raw_parts[0].startswith(('"', "'")):
                        item_rule = item_rule.model_copy(update={"quoted": True})
                else:
                    item_rule = None
                    fixed_items = [
                        rule_for(part, default_rule=component_default)
                        for part in raw_parts
                    ]
            return AnswerField(
                kind=kind,
                decimals=decimal_places,
                preserve_precision=preserve_precision,
                quoted=not composite and placeholder.strip().startswith(('"', "'")),
                separator=";" if "separated by semicolons" in description else ",",
                item=item_rule,
                items=fixed_items,
                properties=properties,
            )

        alternatives = None
        conditional = re.search(
            r"\bif\b[^@]*?\b(?:(?:simply|only|instead)\s+)?output\s+",
            text,
            re.IGNORECASE,
        )
        if conditional and (
            not re.search(r"\belse\b", text[conditional.end() :], re.IGNORECASE)
            or re.search(r"\b(simply|only|instead)\b", conditional[0], re.IGNORECASE)
        ):
            primary = list(dict(re.findall(pattern, text[: conditional.start()])))
            alternative = list(dict(re.findall(pattern, text[conditional.end() :])))
            if primary and alternative:
                alternatives = [primary, alternative]
        return cls(
            fields={
                name: rule_for(placeholder, name)
                for name, placeholder in templates.items()
            },
            alternatives=alternatives,
        )


def _validate_value(value: object, path: str, rule: AnswerField) -> None:
    if value is None and not rule.allow_null:
        raise AnswerSubmissionError(
            f"{path}: null is not permitted; compute a value or report insufficient input"
        )
    if isinstance(value, str) and not value.strip():
        raise AnswerSubmissionError(f"{path}: return a nonempty value")
    if isinstance(value, float) and not math.isfinite(value):
        raise AnswerSubmissionError(
            f"{path}: non-finite value; correct or explicitly explain the undefined calculation"
        )
    if isinstance(value, Decimal) and not value.is_finite():
        raise AnswerSubmissionError(f"{path}: non-finite numeric value")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_value(item, f"{path}[{index}]", rule.component(index))
    elif isinstance(value, dict):
        if rule.properties is not None:
            _check_fields(list(value), list(rule.properties), path)
        for name, item in value.items():
            _validate_value(item, f"{path}.{name}", rule.member(name))


def _render_json_value(value: object, rule: AnswerField, path: str) -> str:
    if isinstance(value, dict):
        if rule.kind not in {"auto", "dict"}:
            raise AnswerSubmissionError(
                f"{path}: expected {rule.kind}, got a dictionary"
            )
        text = (
            "{"
            + ",".join(
                json.dumps(name, ensure_ascii=False)
                + ":"
                + _render_json_value(value[name], rule.member(name), f"{path}.{name}")
                for name in (rule.properties if rule.properties is not None else value)
            )
            + "}"
        )
        return json.dumps(text, ensure_ascii=False) if rule.quoted else text
    if isinstance(value, list):
        if rule.kind not in {"auto", "list"}:
            raise AnswerSubmissionError(f"{path}: expected {rule.kind}, got a list")
        text = (
            "["
            + ",".join(
                _render_json_value(item, rule.component(index), f"{path}[{index}]")
                for index, item in enumerate(value)
            )
            + "]"
        )
        return json.dumps(text, ensure_ascii=False) if rule.quoted else text
    if isinstance(value, str) and rule.kind in {"auto", "string"}:
        return json.dumps(value, ensure_ascii=False)
    rendered = _render_value(value, rule, path)
    if rule.quoted:
        return rendered
    if isinstance(value, bool) or value is None:
        return json.dumps(value)
    if isinstance(value, str) and rule.kind == "number_or_string":
        try:
            Decimal(rendered)
        except InvalidOperation:
            return json.dumps(value, ensure_ascii=False)
    # JSON numbers cannot retain spellings such as +1, .5, or 01.
    return rendered if rule.decimals is not None else str(Decimal(rendered))


def _render_value(value: object, rule: AnswerField, path: str) -> str:
    if value is None:
        text = "null"
    elif isinstance(value, list):
        if rule.kind not in {"auto", "list"}:
            raise AnswerSubmissionError(f"{path}: expected {rule.kind}, got a list")
        if rule.items is not None and len(value) != len(rule.items):
            raise AnswerSubmissionError(
                f"{path}: expected {len(rule.items)} list components, got {len(value)}"
            )
        parts = []
        for index, item in enumerate(value):
            component = rule.component(index)
            if isinstance(item, list):
                raise AnswerSubmissionError(
                    f"{path}[{index}]: nested lists cannot be flattened into public tags; return an unambiguous representation permitted by the question"
                )
            if (
                isinstance(item, str)
                and rule.separator in item
                and not component.quoted
            ):
                raise AnswerSubmissionError(
                    f"{path}[{index}]: string contains the list separator; use quoting or a different separator only when permitted by the public format"
                )
            parts.append(_render_value(item, component, f"{path}[{index}]"))
        text = rule.separator.join(parts)
    elif isinstance(value, dict):
        if rule.kind not in {"auto", "dict"}:
            raise AnswerSubmissionError(
                f"{path}: expected {rule.kind}, got a dictionary"
            )
        text = _render_json_value(
            value, rule.model_copy(update={"quoted": False}), path
        )
    elif rule.kind in {"number", "integer"} or type(value) in {int, float, Decimal}:
        if isinstance(value, bool) or rule.kind not in {
            "auto",
            "number",
            "integer",
            "number_or_string",
        }:
            raise AnswerSubmissionError(f"{path}: expected {rule.kind}")
        try:
            numeric = Decimal(str(value))
        except InvalidOperation:
            raise AnswerSubmissionError(f"{path}: expected a numeric value") from None
        if not numeric.is_finite():
            raise AnswerSubmissionError(f"{path}: non-finite numeric value")
        if (
            rule.kind == "integer" or rule.decimals is not None
        ) and numeric.adjusted() + (rule.decimals or 0) > 65536:
            raise AnswerSubmissionError(
                f"{path}: formatted value exceeds the answer size limit"
            )
        if rule.kind == "integer":
            if numeric != numeric.to_integral_value():
                raise AnswerSubmissionError(
                    f"{path}: expected an integer, do not round a fractional count"
                )
            text = format(numeric, ".0f")
        elif rule.decimals is not None:
            text = format(numeric, f".{rule.decimals}f")
        else:
            text = str(value)
    elif isinstance(value, bool):
        if rule.kind not in {"auto", "boolean"}:
            raise AnswerSubmissionError(f"{path}: expected {rule.kind}")
        text = str(value)
    elif isinstance(value, str):
        if rule.kind not in {"auto", "string", "number_or_string"}:
            raise AnswerSubmissionError(f"{path}: expected {rule.kind}")
        if rule.kind in {"auto", "number_or_string"} and not rule.quoted:
            try:
                numeric_string = Decimal(value)
            except InvalidOperation:
                pass
            else:
                if not numeric_string.is_finite():
                    raise AnswerSubmissionError(
                        f"{path}: non-finite numeric value; declare literal categories as string fields"
                    )
                if rule.kind == "number_or_string" or rule.decimals is not None:
                    return _render_value(numeric_string, rule, path)
        text = value
    else:
        raise AnswerSubmissionError(f"{path}: expected a JSON value")
    return json.dumps(text, ensure_ascii=False) if rule.quoted else text


def _check_fields(names: list[str], required: list[str], path: str) -> None:
    if len(names) != len(set(names)):
        raise AnswerSubmissionError(f"{path}: duplicate fields; submit each field once")
    missing = set(required) - set(names)
    if missing:
        raise AnswerSubmissionError(
            f"{path}: missing required fields {', '.join(sorted(missing))}"
        )
    extra = set(names) - set(required)
    if extra:
        raise AnswerSubmissionError(
            f"{path}: unexpected fields {', '.join(sorted(extra))}; put optional detail in the table"
        )


def _split_list(text: str, separator: str, path: str) -> list[str]:
    parts = []
    start = index = depth = 0
    quoted = escaped = False
    while index < len(text):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif depth == 0 and text.startswith(separator, index):
            parts.append(text[start:index])
            index += len(separator)
            start = index
            continue
        index += 1
    if quoted:
        raise AnswerSubmissionError(f"{path}: unterminated JSON-quoted list item")
    return [*parts, text[start:]] if text else []


def _parse_legacy_value(text: str, rule: AnswerField, path: str) -> object:
    if rule.quoted:
        try:
            return json.loads(text)
        except ValueError:
            raise AnswerSubmissionError(
                f"{path}: expected a JSON-quoted value"
            ) from None
    if rule.kind == "list":
        return (
            [
                _parse_legacy_value(
                    part.strip(),
                    rule.component(index),
                    f"{path}[{index}]",
                )
                for index, part in enumerate(_split_list(text, rule.separator, path))
            ]
            if text
            else []
        )
    if rule.kind == "string":
        return text
    if text in {"True", "False"}:
        return text == "True"
    try:
        return Decimal(text)
    except InvalidOperation:
        try:
            return json.loads(text)
        except ValueError:
            return text


def _validate_legacy(answer: str, contract: AnswerFormat) -> None:
    pattern = r"@(\w+)\[([^\[\]\r\n]*)\]"
    tags = re.findall(pattern, answer)
    names = [name for name, _ in tags]
    _check_fields(names, contract.required_names(names, "answer"), "answer")
    if re.sub(pattern, "", answer).strip(" \t\r\n,"):
        raise AnswerSubmissionError(
            "answer: invalid presentation; submit only the required @field[value] entries"
        )
    for name, text in tags:
        rule = contract.fields[name]
        path = f"answer.{name}"
        parsed = _parse_legacy_value(text.strip(), rule, path)
        _validate_value(parsed, path, rule)
        rendered = _render_value(parsed, rule, path)
        if (
            isinstance(parsed, Decimal)
            and rule.decimals is None
            and rule.kind != "integer"
        ):
            matches = parsed == Decimal(rendered)
        else:
            matches = rendered == text.strip()
        if not matches:
            raise AnswerSubmissionError(
                f"{path}: invalid precision or presentation; use answer_fields for deterministic formatting"
            )


def prepare_answer(value: object, contract: AnswerFormat | None) -> object:
    if not isinstance(value, dict) or "answer_fields" not in value:
        if (
            contract
            and isinstance(value, dict)
            and isinstance(value.get("answer"), str)
        ):
            _validate_legacy(value["answer"], contract)
        return value
    fields = value["answer_fields"]
    if not isinstance(fields, dict) or not fields:
        raise AnswerSubmissionError(
            "answer_fields: return a nonempty JSON field mapping"
        )
    names = (
        contract.required_names(list(fields), "answer_fields")
        if contract
        else list(fields)
    )
    _check_fields(list(fields), names, "answer_fields")
    for name in names:
        rule = contract.fields[name] if contract else AnswerField()
        _validate_value(fields[name], f"answer_fields.{name}", rule)
    rendered = {
        name: json.dumps(
            fields[name], ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        if not contract and isinstance(fields[name], (list, dict))
        else _render_value(
            fields[name],
            contract.fields[name] if contract else AnswerField(),
            f"answer_fields.{name}",
        )
        for name in names
    }
    if contract:
        for name, text in rendered.items():
            if any(char in text for char in "[]\r\n"):
                raise AnswerSubmissionError(
                    f"answer_fields.{name}: value cannot be represented unambiguously inside @field[value]; follow the public representation without truncation"
                )
    answer = "\n".join(
        f"@{name}[{text}]" if contract else f"{name}: {text}"
        for name, text in rendered.items()
    )
    if "answer" in value and value["answer"] != answer:
        raise AnswerSubmissionError(
            "answer: conflicts with computed answer_fields; omit answer and let the public format render it"
        )
    return {**value, "answer": answer}


def comparison_fields(
    answer: str, fields: Mapping[str, object] | None, contract: AnswerFormat | None
) -> dict[str, str]:
    """Compare complete fields at the public precision, without scoring data.

    Unstructured legacy prose is compared as one answer; public tagged answers
    use the same parser and rendering rules as submission validation.
    """
    if fields is not None:
        return {
            name: json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
            if not contract and isinstance(value, (str, list, dict))
            else _comparison_text(
                value, contract.fields[name] if contract else AnswerField(), name
            )
            for name, value in fields.items()
        }
    if contract is not None:
        return {
            name: _comparison_text(
                _parse_legacy_value(text.strip(), contract.fields[name], name),
                contract.fields[name],
                name,
            )
            for name, text in re.findall(r"@(\w+)\[([^\[\]\r\n]*)\]", answer)
        }
    return {"answer": answer}


def _comparison_text(value: object, rule: AnswerField, path: str) -> str:
    if isinstance(value, (list, dict)):
        return _render_json_value(
            value, rule.model_copy(update={"quoted": False}), path
        )
    rendered = _render_value(value, rule, path)
    if rule.kind == "string" and isinstance(value, str) and not rule.quoted:
        return json.dumps(value, ensure_ascii=False)
    return rendered
