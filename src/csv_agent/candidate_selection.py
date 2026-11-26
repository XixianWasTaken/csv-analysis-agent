"""Field comparisons and evidence gates for independent solution selection."""

import json
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel

from csv_agent.model import SelectCandidate


class FieldComparison(BaseModel):
    first: str | None
    second: str | None
    equal: bool


class CandidateComparison(BaseModel):
    first: int
    second: int
    fields: dict[str, FieldComparison]


class CandidateSelection(SelectCandidate):
    number: int


def _parse_comparison_value(text: str) -> object:
    try:
        return Decimal(text)
    except InvalidOperation:
        try:
            return json.loads(text, parse_int=Decimal, parse_float=Decimal)
        except ValueError:
            return text


def _same_value(first: object, second: object) -> bool:
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(
            _same_value(value, second[key]) for key, value in first.items()
        )
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            _same_value(a, b) for a, b in zip(first, second)
        )
    return type(first) is type(second) and first == second


def compare_fields(
    first: int,
    first_fields: dict[str, str],
    second: int,
    second_fields: dict[str, str],
) -> CandidateComparison:
    return CandidateComparison(
        first=first,
        second=second,
        fields={
            name: FieldComparison(
                first=first_fields.get(name),
                second=second_fields.get(name),
                equal=name in first_fields
                and name in second_fields
                and _same_value(
                    _parse_comparison_value(first_fields[name]),
                    _parse_comparison_value(second_fields[name]),
                ),
            )
            for name in dict.fromkeys([*first_fields, *second_fields])
        },
    )


def first_field_disagreements(comparisons: list[CandidateComparison]) -> dict[str, int]:
    """Request where each differing field first needs a subsequent check."""
    requests: dict[str, int] = {}
    for comparison in comparisons:
        for name, field in comparison.fields.items():
            if not field.equal:
                requests[name] = min(
                    requests.get(name, comparison.second), comparison.second
                )
    return requests


def validate_selection(
    instruction: SelectCandidate,
    comparisons: list[CandidateComparison],
    *,
    available_evidence: set[int],
    checking_evidence: set[int],
) -> None:
    if not set(instruction.evidence) <= available_evidence:
        raise ValueError(
            "Selection evidence must reference successful candidate or inspection requests"
        )
    if instruction.candidate not in instruction.evidence:
        raise ValueError("Selection must reference the selected candidate's execution")
    disagreements = first_field_disagreements(comparisons)
    if set(instruction.resolutions) != set(disagreements):
        raise ValueError(
            "Resolve every disagreement field with a factual finding and further executed evidence: "
            + ", ".join(sorted(disagreements))
        )
    if not disagreements:
        return
    for name, resolution in instruction.resolutions.items():
        evidence = set(resolution.evidence)
        fresh_checks = {
            number for number in checking_evidence if number > disagreements[name]
        }
        if not evidence <= set(instruction.evidence) or not evidence & fresh_checks:
            raise ValueError(
                f"Unresolved disagreement in {name}: inspect data or independently recompute after the disagreement, then cite that execution"
            )
