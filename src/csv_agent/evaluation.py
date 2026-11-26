"""Batch analysis with host-only labels, official scoring, and fixed denominators."""

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from csv_agent._vendor import dabench
from csv_agent.analysis import STRATEGY_POLICIES, AnalysisResult, Strategy, analyze
from csv_agent.benchmark import (
    Manifest,
    Question,
    Split,
    load_questions,
    sha256,
    verified_bytes,
)
from csv_agent.config import ModelConfig, RunLimits, configuration_error_message
from csv_agent.evaluation_scope import EvaluationSelection, Purpose, select_samples
from csv_agent.evidence import verify_analysis_evidence
from csv_agent.model import SYSTEM, TOOLS, Model, Usage
from csv_agent.runtime import capture_runtime, inspect_image

ANSWER_CONTRACT = (
    "\nCompute every requested answer directly. Prefer result.json's answer_fields "
    "mapping, keyed by the public @name declarations, with computed JSON values. "
    "The collector renders the required @name[value] format. A complete legacy "
    "answer string is also accepted. Do not use an additional model to reformat."
)


def comparison_prompt_config() -> dict[str, object]:
    """The prompt and action contracts shared by comparisons and frozen batches."""
    return {
        "system": SYSTEM,
        "answer_contract": ANSWER_CONTRACT,
        "tools": {
            name: {"description": description, "schema": schema.model_json_schema()}
            for name, (description, schema) in TOOLS.items()
        },
        "strategy_policies": STRATEGY_POLICIES,
    }


def analysis_prompt(question: Question) -> str:
    """Public question text sent to analysis, with no host scoring labels."""
    return (
        question.question
        + "\nConstraints:\n"
        + question.constraints
        + "\nRequired answer format:\n"
        + question.format
        + ANSWER_CONTRACT
    )


def _save_metadata(directory: Path, metadata: Mapping[str, object]) -> None:
    with (directory / "run.json").open("x") as output:
        output.write(json.dumps(dict(metadata), indent=2) + "\n")
        output.flush()
        os.fsync(output.fileno())


class EvaluationRecord(BaseModel):
    id: int
    file_name: str
    analysis: AnalysisResult
    response: str | None = None
    correctness: dict[str, bool] = Field(default_factory=dict)
    failure: str | None = None


class RequestUsageSummary(BaseModel):
    """Saved requests only, keyed by analysis run ID and attempt number."""

    request_keys: list[tuple[str, int]]
    unique_requests: int
    duplicate_requests: int
    usage: Usage
    known_usage: Usage
    unknown_usage_requests: dict[str, int]


def summarize_request_usage(records: Iterable[EvaluationRecord]) -> RequestUsageSummary:
    requests: dict[tuple[str, int], Usage | None] = {}
    fingerprints: dict[tuple[str, int], str] = {}
    duplicates = 0
    for record in records:
        for attempt in record.analysis.attempts:
            if attempt.action == "replay":
                continue
            key = (record.analysis.run_id, attempt.number)
            fingerprint = attempt.model_dump_json()
            if key in requests:
                if fingerprints[key] != fingerprint:
                    raise ValueError("Conflicting evidence for the same saved request")
                duplicates += 1
                continue
            requests[key] = attempt.usage
            fingerprints[key] = fingerprint
    known = {}
    cumulative = {}
    unknown = {}
    for field in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
        values = [
            getattr(usage, field) if usage else None for usage in requests.values()
        ]
        reported = [v for v in values if v is not None]
        unknown[field] = len(values) - len(reported)
        known[field] = sum(reported) if reported else None
        cumulative[field] = known[field] if not unknown[field] else None
    return RequestUsageSummary(
        request_keys=list(requests),
        unique_requests=len(requests),
        duplicate_requests=duplicates,
        usage=Usage(**cumulative),
        known_usage=Usage(**known),
        unknown_usage_requests=unknown,
    )


class EvaluationSummary(BaseModel):
    total: int
    correct_questions: int
    total_subquestions: int
    correct_subquestions: int
    question_accuracy: float
    subquestion_accuracy: float
    proportional_subquestion_accuracy: float
    execution_successes: int
    execution_success_rate: float
    failures: dict[str, int]
    timeout_rate: float
    failure_rate: float
    task_seconds: float
    model_seconds: float
    model_calls: int
    execution_attempts: int
    usage: Usage
    known_usage: Usage
    usage_reporting_tasks: dict[str, int]
    request_usage: RequestUsageSummary | None = None


class EvaluationRun(BaseModel):
    strict_strategy: bool = False
    run_id: str
    started_at: str
    mode: Literal["live", "controlled"]
    split: Split
    manifest_sha256: str
    manifest: Manifest
    config: ModelConfig
    limits: RunLimits
    strategy: Strategy = "langgraph"
    selection: EvaluationSelection | None = None
    wall_seconds: float
    records: list[EvaluationRecord]


class EvaluationReport(EvaluationRun):
    summary: EvaluationSummary


class StrategyBudget(BaseModel):
    limits: RunLimits
    max_model_calls: int


class PairedRecord(BaseModel):
    id: int
    file_name: str
    results: dict[Strategy, EvaluationRecord]


class ComparisonDelta(BaseModel):
    """Candidate minus reference, preserving negative results and unknown usage."""

    reference: Strategy
    candidate: Strategy
    wins: int
    losses: int
    ties: int
    question_accuracy: float
    subquestion_accuracy: float
    execution_success_rate: float
    failure_rate: float
    timeout_rate: float
    task_seconds: float
    model_seconds: float
    model_calls: int
    execution_attempts: int
    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None


class ComparisonReport(BaseModel):
    freeze_sha256: str | None = None
    strict_strategy: bool = False
    workers: int = Field(default=1, ge=1, le=8)
    run_id: str
    started_at: str
    mode: Literal["live", "controlled"]
    model_version: str
    config: ModelConfig
    requested_limits: RunLimits
    limits: RunLimits
    split: Split
    manifest: Manifest
    manifest_sha256: str
    strategy_order: list[Strategy]
    selection: EvaluationSelection | None = None
    prompt_config: dict[str, object]
    source_sha256: dict[str, str]
    environment: dict[str, str] = Field(default_factory=dict)
    budgets: dict[Strategy, StrategyBudget]
    wall_seconds: float
    runs: dict[Strategy, EvaluationReport]
    paired_records: list[PairedRecord]
    deltas: dict[str, ComparisonDelta]
    request_usage: RequestUsageSummary | None = None


class Label(BaseModel):
    model_config = ConfigDict(strict=True)
    id: int
    common_answers: list[tuple[str, str]] = Field(min_length=1)


def score_records(
    root: Path,
    manifest: Manifest,
    records: list[EvaluationRecord],
    *,
    include_request_usage: bool = False,
) -> EvaluationSummary:
    """Only scoring reads labels, after all Agent calls have finished."""
    source = manifest.dataset
    data = verified_bytes(root / source.labels_file, source.labels_sha256)
    labels = [
        Label.model_validate_json(line) for line in data.splitlines() if line.strip()
    ]
    by_id = {label.id: label for label in labels}
    if len(by_id) != len(labels) or set(by_id) != {s.id for s in manifest.samples}:
        raise ValueError("Labels do not match the locked sample manifest")
    official_results = []
    for record in records:
        label = by_id[record.id]
        analysis = record.analysis
        record.correctness = {name: False for name, _ in label.common_answers}
        if analysis.status != "completed":
            record.failure = record.failure or analysis.diagnostic or analysis.status
        elif not analysis.answer or not analysis.answer.strip():
            record.failure = "empty_answer"
        else:
            record.response = analysis.answer
            results = dabench.evaluate_responses(
                [label.model_dump()], [{"id": record.id, "response": record.response}]
            )
            extracted = results[0]["predicted_answers"]
            if not extracted:
                record.failure = "unparseable_answer"
            else:
                record.correctness = results[0]["correctness"]
                if set(record.correctness) - set(extracted):
                    record.failure = "missing_answer_fields"
                elif not all(record.correctness.values()):
                    record.failure = "incorrect_answer"
        # Fill official omissions, preserving one row and all expected subquestions.
        official_results.append({"id": record.id, "correctness": record.correctness})
    total = len(records)
    # A successful task produces a valid computed result; the trusted runner's
    # exit code alone is insufficient because it also envelopes Python failures.
    execution_successes = sum(r.analysis.status == "completed" for r in records)
    fields = ("input_tokens", "output_tokens", "cache_read_input_tokens")
    reporting: dict[str, int] = {}
    cumulative: dict[str, int | None] = {}
    for field in fields:
        values = [
            getattr(r.analysis.usage, field) if r.analysis.usage else None
            for r in records
        ]
        reporting[field] = sum(v is not None for v in values)
        cumulative[field] = (
            sum(v for v in values if v is not None)
            if reporting[field] == total
            else None
        )
    timeouts = sum(r.analysis.status == "timeout" for r in records)
    request_usage = summarize_request_usage(records)
    return EvaluationSummary(
        total=total,
        correct_questions=sum(all(r.correctness.values()) for r in records),
        total_subquestions=sum(len(r.correctness) for r in records),
        correct_subquestions=sum(sum(r.correctness.values()) for r in records),
        question_accuracy=dabench.evaluate_accuracy_by_question(official_results),
        subquestion_accuracy=dabench.evaluate_accuracy_by_sub_question(
            official_results
        ),
        proportional_subquestion_accuracy=dabench.evaluate_accuracy_proportional_by_sub_question_adjusted(
            official_results
        ),
        execution_successes=execution_successes,
        execution_success_rate=execution_successes / total,
        failures=dict(Counter(r.failure for r in records if r.failure)),
        timeout_rate=timeouts / total,
        failure_rate=sum(r.analysis.status != "completed" for r in records) / total,
        task_seconds=sum(r.analysis.seconds for r in records),
        model_seconds=sum(
            a.model_seconds or 0 for r in records for a in r.analysis.attempts
        ),
        model_calls=request_usage.unique_requests,
        execution_attempts=sum(r.analysis.execution_attempts for r in records),
        usage=Usage(**cumulative),
        known_usage=request_usage.known_usage,
        usage_reporting_tasks=reporting,
        request_usage=request_usage if include_request_usage else None,
    )


def _run_batch(
    root: Path,
    manifest_path: Path,
    *,
    split: Split,
    output_dir: Path,
    config: ModelConfig,
    limits: RunLimits,
    model: Model | None = None,
    strategy: Strategy = "langgraph",
    selection: EvaluationSelection | None = None,
    workers: int = 1,
    strict_strategy: bool = False,
    saved_records: list[EvaluationRecord] | None = None,
) -> EvaluationRun:
    """Run every selected sample through analyze; injected models are labelled controlled."""
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    manifest_data = manifest_path.read_bytes()
    manifest = Manifest.model_validate_json(manifest_data)
    if sha256(Path(dabench.__file__).read_bytes()) != manifest.scorer.vendored_sha256:
        raise ValueError("Installed scorer does not match the locked version")
    questions = load_questions(root, manifest)
    selected = selection.sample_ids if selection else manifest.splits[split]
    if not selected:
        raise ValueError("Cannot evaluate an empty split")
    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    metadata = {
        "run_id": uuid4().hex,
        "started_at": started_at,
        "mode": "controlled" if model is not None else "live",
        "split": split,
        "manifest_sha256": sha256(manifest_data),
        "manifest": manifest.model_dump(),
        "config": config.model_dump(),
        "limits": limits.model_dump(),
        "strategy": strategy,
        "strict_strategy": strict_strategy,
        "selection": selection.model_dump() if selection else None,
    }
    _save_metadata(output_dir, metadata)
    saved = {record.id: record for record in saved_records or []}

    def run_sample(sample_id: int) -> EvaluationRecord:
        if sample_id in saved:
            return saved[sample_id].model_copy(deep=True)
        question = questions[sample_id]
        text = analysis_prompt(question)
        record = EvaluationRecord(
            id=sample_id,
            file_name=question.file_name,
            analysis=AnalysisResult(run_id=uuid4().hex, status="failed"),
        )
        task_started = time.monotonic()
        try:
            table_path = root / "da-dev-tables" / question.file_name
            table_source = manifest.tables[question.file_name]
            # Bound reads before passing even damaged local inputs to analyze.
            with table_path.open("rb") as table:
                data = table.read(
                    min(table_source.size_bytes, limits.input_max_bytes) + 1
                )
            if len(data) > limits.input_max_bytes:
                raise ValueError("Benchmark CSV exceeds configured input limit")
            if (
                len(data) != table_source.size_bytes
                or sha256(data) != table_source.sha256
            ):
                raise ValueError("Benchmark CSV checksum mismatch")
            record.analysis = analyze(
                data,
                text,
                config=config,
                limits=limits,
                model=model,
                strategy=strategy,
                answer_format=question.format,
                strict_strategy=strict_strategy,
            )
        except (OSError, ValueError):
            record.failure = "data_error"
            record.analysis.error = (
                "Benchmark CSV missing, damaged, or exceeds the input limit"
            )
            record.analysis.seconds = time.monotonic() - task_started
        return record

    # map preserves the selected order; only the parent writes evidence.
    with (
        (output_dir / "analysis.jsonl").open("w") as output,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        for record in executor.map(run_sample, selected):
            records.append(record)
            output.write(record.model_dump_json() + "\n")
            output.flush()
            os.fsync(output.fileno())
    return EvaluationRun.model_validate(
        {
            **metadata,
            "wall_seconds": time.monotonic() - started,
            "records": records,
        }
    )


def _score_and_save(
    root: Path, run: EvaluationRun, output_dir: Path
) -> EvaluationReport:
    started = time.monotonic()
    summary = score_records(root, run.manifest, run.records, include_request_usage=True)
    report = EvaluationReport(**run.model_dump(), summary=summary)
    report.wall_seconds += time.monotonic() - started
    (output_dir / "results.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in report.records)
    )
    (output_dir / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    return report


def _comparison_markdown(report: ComparisonReport) -> str:
    lines = [
        "# Paired DABench comparison",
        "",
        f"Concurrent tasks per strategy: {report.workers}. Fixed action scope: {report.strict_strategy}.",
        f"Mode: `{report.mode}`. Model version declaration: `{report.model_version}`.",
        f"Split: `{report.split}`. Manifest SHA-256: `{report.manifest_sha256}`.",
        "Controlled runs verify contracts; their scores do not measure real model capability.",
    ]
    if report.selection:
        selection = report.selection
        lines.extend(
            [
                f"Purpose: `{selection.purpose}`. Fixed question denominator per strategy: {selection.total}.",
                f"Selected IDs (execution order): `{selection.sample_ids}`.",
                "Optimization scores describe repeatedly used samples. Holdout selections require complete, isolated final file groups and exclude the known optimization files.",
                "",
                "| File | Selected IDs | All file IDs | Complete group | SHA-256 |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for name, group in selection.file_groups.items():
            lines.append(
                f"| {name} | {group.selected_ids} | {group.available_ids} | {group.complete} | {group.source.sha256} |"
            )
    lines.extend(
        [
            "",
            "| Strategy | Question accuracy | Subquestion accuracy | Execution success | Task failure | Timeout | Task seconds | Model calls | Executions | Input tokens | Output tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy in report.strategy_order:
        s = report.runs[strategy].summary
        lines.append(
            f"| {strategy} | {s.question_accuracy:.4f} | {s.subquestion_accuracy:.4f} | "
            f"{s.execution_success_rate:.4f} | {s.failure_rate:.4f} | {s.timeout_rate:.4f} | "
            f"{s.task_seconds:.3f} | {s.model_calls} | {s.execution_attempts} | "
            f"{s.usage.input_tokens} | {s.usage.output_tokens} |"
        )
    lines.extend(
        [
            "",
            "| Strategy | Max task seconds | Max calls | Max executions | Format repairs | Transport retries |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy, budget in report.budgets.items():
        limits = budget.limits
        lines.append(
            f"| {strategy} | {limits.task_timeout_seconds} | {budget.max_model_calls} | "
            f"{limits.max_executions} | {limits.max_format_repairs} | {limits.max_model_retries} |"
        )
    lines.extend(
        [
            "",
            "| Strategy | Question denominator | Subquestion denominator | Unique requests | Known input | Known output | Known cache reads | Unknown input requests | Unknown output requests | Unknown cache requests |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy, run in report.runs.items():
        accounting = run.summary.request_usage
        if accounting:
            known, unknown = accounting.known_usage, accounting.unknown_usage_requests
            lines.append(
                f"| {strategy} | {run.summary.total} | {run.summary.total_subquestions} | {accounting.unique_requests} | {known.input_tokens} | {known.output_tokens} | {known.cache_read_input_tokens} | {unknown['input_tokens']} | {unknown['output_tokens']} | {unknown['cache_read_input_tokens']} |"
            )
    lines.extend(
        [
            "",
            "| Candidate minus reference | Accuracy delta | Wins | Losses | Ties | Task seconds delta | Calls delta | Input tokens delta | Output tokens delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, d in report.deltas.items():
        lines.append(
            f"| {name} | {d.question_accuracy:+.4f} | {d.wins} | {d.losses} | {d.ties} | "
            f"{d.task_seconds:+.3f} | {d.model_calls:+d} | {d.input_tokens} | {d.output_tokens} |"
        )
    lines.extend(
        [
            "",
            "`None` denotes unknown usage; reported portions and reporting counts are in report.json.",
            "Saved requests are deduplicated by analysis run_id + attempt.number; conflicting copies are rejected. Reused evidence counts once; an actual rerun gets a new run_id and counts again. Interrupted unsaved requests and other clients are outside these totals.",
            "Local input/output/cache counters are separate from the account dashboard. Cache billing may overlap input billing; these counters do not establish account balance or remaining credits.",
            "Single generation uses less computation. Extra computation and history length can affect results; an accuracy difference alone does not isolate orchestration benefits.",
            "langgraph_no_repair keeps the same maximum budgets but disables instruction/code feedback repairs; transport retries remain. Actual computation can be lower.",
            "All strategies share CSV inspection, isolated execution, artifact validation and the official scorer. Only feedback repair is ablated.",
            "Batches run in the saved fixed strategy order; provider load, stochastic responses and mutable model aliases can affect comparison. No minimum improvement is required.",
            "These are pinned public validation samples, not official hidden test results.",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(
    root: Path,
    manifest_path: Path,
    *,
    split: Split,
    output_dir: Path,
    config: ModelConfig,
    limits: RunLimits,
    model: Model | None = None,
) -> EvaluationReport:
    """Existing single-path evaluation; UI and evaluation share analyze."""
    run = _run_batch(
        root,
        manifest_path,
        split=split,
        output_dir=output_dir,
        config=config,
        limits=limits,
        model=model,
    )
    return _score_and_save(root, run, output_dir)


def _compare(
    runs: dict[Strategy, EvaluationReport], reference: Strategy, candidate: Strategy
) -> ComparisonDelta:
    before, after = runs[reference].summary, runs[candidate].summary
    pairs = list(zip(runs[reference].records, runs[candidate].records, strict=True))
    outcomes = [
        int(all(b.correctness.values())) - int(all(a.correctness.values()))
        for a, b in pairs
    ]
    values = {
        field: getattr(after, field) - getattr(before, field)
        for field in (
            "question_accuracy",
            "subquestion_accuracy",
            "execution_success_rate",
            "failure_rate",
            "timeout_rate",
            "task_seconds",
            "model_seconds",
            "model_calls",
            "execution_attempts",
        )
    }
    for field in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
        a, b = getattr(before.usage, field), getattr(after.usage, field)
        values[field] = b - a if a is not None and b is not None else None
    return ComparisonDelta(
        reference=reference,
        candidate=candidate,
        wins=outcomes.count(1),
        losses=outcomes.count(-1),
        ties=outcomes.count(0),
        **values,
    )


def run_comparison(
    root: Path,
    manifest_path: Path,
    *,
    split: Split,
    output_dir: Path,
    config: ModelConfig,
    limits: RunLimits,
    model_version: str,
    models: dict[Strategy, Model] | None = None,
    include_ablation: bool = False,
    sample_ids: list[int] | None = None,
    strategies: list[Strategy] | None = None,
    purpose: Purpose = "optimization",
    workers: int = 1,
    strict_strategy: bool = False,
    freeze_sha256: str | None = None,
    saved_records: dict[Strategy, list[EvaluationRecord]] | None = None,
) -> ComparisonReport:
    """Run fixed-order paired batches, then use the same host-only official scorer.

    Controlled callers supply a separate model per strategy; no mixed live runs.
    The image ID and explicit temperature freeze local settings. model_version is
    an operator declaration, not proof that a provider alias is immutable.
    """
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Comparison workers must be an integer from 1 to 8")
    started = time.monotonic()
    order: list[Strategy] = (
        list(strategies)
        if strategies is not None
        else ["single", "tool_loop", "langgraph"]
    )
    if (
        not order
        or len(set(order)) != len(order)
        or any(s not in STRATEGY_POLICIES for s in order)
    ):
        raise ValueError("Select one or more distinct implemented strategies")
    if include_ablation and "langgraph_no_repair" not in order:
        order.append("langgraph_no_repair")
    if models is not None and set(models) != set(order):
        raise ValueError("Controlled comparison must supply every selected strategy")
    if not model_version.strip() or config.temperature is None:
        raise ValueError("Comparison requires a model version and explicit temperature")
    manifest_data = manifest_path.read_bytes()
    manifest = Manifest.model_validate_json(manifest_data)
    if sha256(Path(dabench.__file__).read_bytes()) != manifest.scorer.vendored_sha256:
        raise ValueError("Installed scorer does not match the locked version")
    questions = load_questions(root, manifest)
    selection = select_samples(manifest, split, sample_ids, purpose)
    if saved_records is not None:
        if models is not None or set(saved_records) - set(order):
            raise ValueError("Saved analyses require the same live strategies")
        for strategy, records in saved_records.items():
            if [r.id for r in records] != selection.sample_ids[: len(records)]:
                raise ValueError("Saved analyses must be a selected complete prefix")
            if any(
                r.analysis.status == "running"
                or r.response is not None
                or r.correctness
                or r.failure not in {None, "data_error"}
                for r in records
            ):
                raise ValueError("Saved analyses must be complete and unscored")
            for record in records:
                question = questions[record.id]
                if record.file_name != question.file_name or (
                    record.analysis.attempts
                    and record.analysis.profile.get("source_sha256")
                    != manifest.tables[question.file_name].sha256
                ):
                    raise ValueError(
                        "Saved analysis data source differs from the manifest"
                    )
                verify_analysis_evidence(
                    record.analysis,
                    config=config,
                    limits=limits.for_single_generation()
                    if strategy == "single"
                    else limits,
                    strategy=strategy,
                    answer_format=question.format,
                    strict_strategy=strict_strategy,
                    question=analysis_prompt(question),
                )
    image = inspect_image(limits.image)
    runtime = capture_runtime()
    pinned_limits = limits.model_copy(update={"image": image})
    single_limits = pinned_limits.for_single_generation()
    budgets = {
        strategy: StrategyBudget(
            limits=single_limits if strategy == "single" else pinned_limits,
            max_model_calls=(
                single_limits if strategy == "single" else pinned_limits
            ).model_call_budget,
        )
        for strategy in order
    }
    metadata = {
        "freeze_sha256": freeze_sha256,
        "run_id": uuid4().hex,
        "started_at": datetime.now(UTC).isoformat(),
        "mode": "controlled" if models is not None else "live",
        "model_version": model_version,
        "config": config.model_dump(),
        "workers": workers,
        "strict_strategy": strict_strategy,
        "requested_limits": limits.model_dump(),
        "limits": pinned_limits.model_dump(),
        "split": split,
        "manifest": manifest.model_dump(),
        "manifest_sha256": sha256(manifest_data),
        "strategy_order": order,
        "selection": selection.model_dump(),
        "prompt_config": comparison_prompt_config(),
        "source_sha256": runtime.source_sha256,
        "environment": runtime.environment,
        "budgets": {name: budget.model_dump() for name, budget in budgets.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _save_metadata(output_dir, metadata)
    batches = {
        strategy: _run_batch(
            root,
            manifest_path,
            split=split,
            output_dir=output_dir / strategy,
            config=config,
            limits=budgets[strategy].limits,
            model=models[strategy] if models is not None else None,
            strategy=strategy,
            selection=selection,
            workers=workers,
            strict_strategy=strict_strategy,
            saved_records=saved_records.get(strategy) if saved_records else None,
        )
        for strategy in order
    }
    if any(run.manifest_sha256 != sha256(manifest_data) for run in batches.values()):
        raise ValueError("Manifest changed during comparison; raw evidence retained")
    if capture_runtime() != runtime or inspect_image(image) != image:
        raise ValueError(
            "Source or environment changed during comparison; raw evidence retained"
        )
    # Labels are parsed only after all selected strategies have finished.
    runs = {
        strategy: _score_and_save(root, run, output_dir / strategy)
        for strategy, run in batches.items()
    }
    paired = [
        PairedRecord(
            id=sample_id,
            file_name=runs[order[0]].records[index].file_name,
            results={strategy: runs[strategy].records[index] for strategy in order},
        )
        for index, sample_id in enumerate(selection.sample_ids)
    ]
    comparisons: list[tuple[Strategy, Strategy]] = [
        ("single", "tool_loop"),
        ("single", "langgraph"),
        ("tool_loop", "langgraph"),
    ]
    if include_ablation:
        comparisons.append(("langgraph_no_repair", "langgraph"))
    if strategies is not None:
        comparisons = [
            (reference, candidate)
            for index, reference in enumerate(order)
            for candidate in order[index + 1 :]
        ]
    comparisons = [(a, b) for a, b in comparisons if a in runs and b in runs]
    report = ComparisonReport.model_validate(
        {
            **metadata,
            "wall_seconds": time.monotonic() - started,
            "runs": runs,
            "request_usage": summarize_request_usage(
                record for run in runs.values() for record in run.records
            ),
            "paired_records": paired,
            "deltas": {
                f"{candidate}_vs_{reference}": _compare(runs, reference, candidate)
                for reference, candidate in comparisons
            },
        }
    )
    (output_dir / "paired.jsonl").write_text(
        "".join(pair.model_dump_json() + "\n" for pair in paired)
    )
    (output_dir / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    (output_dir / "summary.md").write_text(_comparison_markdown(report))
    return report


def main() -> None:
    from csv_agent.benchmark import prepare_dataset

    parser = argparse.ArgumentParser(
        description="Pinned public DABench preparation and batch evaluation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "compare", "replay"):
        command = commands.add_parser(name)
        command.add_argument(
            "--manifest", type=Path, default=Path("benchmarks/dabench/manifest.json")
        )
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--split", choices=("development", "final"), required=True)
        if name in {"run", "compare", "replay"}:
            command.add_argument(
                "--output",
                type=Path,
                required=True,
                help="New directory for evidence; replay makes no model requests"
                if name == "replay"
                else "New directory for evidence; run consumes configured model credits",
            )
        if name in {"compare", "replay"}:
            selection_args = command.add_mutually_exclusive_group()
            selection_args.add_argument(
                "--sample-ids",
                type=int,
                nargs="+",
                help="Sample IDs from the locked split, in execution order",
            )
            selection_args.add_argument(
                "--sample-list",
                type=Path,
                help="UTF-8 JSON array of sample IDs from the locked split",
            )
            command.add_argument(
                "--strategies",
                nargs="+",
                choices=tuple(STRATEGY_POLICIES),
                help="Explicit historical source strategies in replay order"
                if name == "replay"
                else "Implemented strategies in execution order; defaults to the existing three",
            )
            if name == "compare":
                command.add_argument(
                    "--model-version",
                    required=True,
                    help="Operator-verified provider version; a mutable alias is not proof of an immutable version",
                )
                command.add_argument("--include-ablation", action="store_true")
                command.add_argument(
                    "--strict-strategy",
                    action="store_true",
                    help="Keep each ablation strategy within its declared action scope",
                )
                command.add_argument(
                    "--workers", type=int, default=1, choices=range(1, 9)
                )
                command.add_argument(
                    "--purpose",
                    choices=("optimization", "holdout"),
                    default="optimization",
                    help="Data use; holdout requires untouched complete final file groups",
                )
            else:
                command.add_argument(
                    "--source",
                    type=Path,
                    required=True,
                    help="Historical comparison directory; replay makes no model requests",
                )
    verification = commands.add_parser(
        "verify-replay",
        help="Verify copied source evidence and repeat official scoring offline",
    )
    verification.add_argument("--data", type=Path, required=True)
    verification.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_dataset(args.data, args.manifest, split=args.split)
            print(json.dumps({"status": "prepared", "split": args.split}))
        elif args.command == "verify-replay":
            from csv_agent.replay import verify_replay

            replay = verify_replay(args.data, args.output)
            print(
                json.dumps(
                    {"status": "historical_replay_verified", "run_id": replay.run_id}
                )
            )
        elif args.command == "replay":
            from csv_agent.replay import run_replay

            load_dotenv()
            sample_ids = args.sample_ids
            if args.sample_list:
                sample_ids = json.loads(args.sample_list.read_text(encoding="utf-8"))
            if not isinstance(sample_ids, list) or not args.strategies:
                raise ValueError(
                    "Replay requires explicit sample IDs and source strategies"
                )
            replay = run_replay(
                args.data,
                args.manifest,
                split=args.split,
                source_dir=args.source,
                output_dir=args.output,
                sample_ids=sample_ids,
                strategies=args.strategies,
                limits=RunLimits.from_env(),
            )
            print(
                json.dumps(
                    {
                        "evidence_kind": replay.evidence_kind,
                        "summaries": {
                            s: run.summary.model_dump()
                            for s, run in replay.runs.items()
                        },
                    },
                    indent=2,
                )
            )
        elif args.command == "compare":
            load_dotenv()
            sample_ids = args.sample_ids
            if args.sample_list:
                sample_ids = json.loads(args.sample_list.read_text(encoding="utf-8"))
                if not isinstance(sample_ids, list):
                    raise ValueError("Sample list must be a JSON array of integer IDs")
            comparison = run_comparison(
                args.data,
                args.manifest,
                split=args.split,
                output_dir=args.output,
                config=ModelConfig.from_env(),
                limits=RunLimits.from_env(
                    exploration=bool(
                        {
                            "langgraph_explore",
                            "langgraph_verify",
                            "langgraph_resolve",
                        }.intersection(args.strategies or [])
                    )
                ),
                model_version=args.model_version,
                include_ablation=args.include_ablation,
                sample_ids=sample_ids,
                strategies=args.strategies,
                purpose=args.purpose,
                workers=args.workers,
                strict_strategy=args.strict_strategy,
            )
            print(
                json.dumps(
                    {
                        "mode": comparison.mode,
                        "selection": comparison.selection.model_dump()
                        if comparison.selection
                        else None,
                        "request_usage": comparison.request_usage.model_dump()
                        if comparison.request_usage
                        else None,
                        "summaries": {
                            s: run.summary.model_dump()
                            for s, run in comparison.runs.items()
                        },
                        "deltas": {
                            name: delta.model_dump()
                            for name, delta in comparison.deltas.items()
                        },
                    },
                    indent=2,
                )
            )
        else:
            load_dotenv()
            report = run_evaluation(
                args.data,
                args.manifest,
                split=args.split,
                output_dir=args.output,
                config=ModelConfig.from_env(),
                limits=RunLimits.from_env(),
            )
            print(report.summary.model_dump_json(indent=2))
    except ValidationError as exc:
        parser.exit(2, configuration_error_message(exc) + "\n")
    except (OSError, ValueError, httpx.HTTPError):
        parser.exit(
            2,
            "Evaluation setup failed: check selected IDs/strategies/purpose, configuration, explicit comparison temperature/model version, prepared Docker image, pinned files and a new output directory\n",
        )


if __name__ == "__main__":
    main()
