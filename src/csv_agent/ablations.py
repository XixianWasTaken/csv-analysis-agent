"""Rescore representative batches and preserve paired gains, losses and settings."""

import argparse
import json
from pathlib import Path
from typing import Any

from csv_agent._vendor import dabench
from csv_agent.analysis import Strategy
from csv_agent.benchmark import sha256
from csv_agent.evaluation import (
    ComparisonReport,
    EvaluationRecord,
    EvaluationReport,
    score_records,
    summarize_request_usage,
)


def _verified_batch(root: Path, directory: Path) -> ComparisonReport:
    report = ComparisonReport.model_validate_json(
        (directory / "report.json").read_bytes()
    )
    if report.selection is None or report.selection.purpose != "optimization":
        raise ValueError(
            "Representative ablations require an explicit optimization selection"
        )
    if (
        sha256(Path(dabench.__file__).read_bytes())
        != report.manifest.scorer.vendored_sha256
    ):
        raise ValueError("Installed official scorer differs from the batch")
    if set(report.strategy_order) != set(report.runs) or len(
        set(report.strategy_order)
    ) != len(report.strategy_order):
        raise ValueError("Batch strategy order does not match its runs")
    metadata = json.loads((directory / "run.json").read_bytes())
    dumped = report.model_dump(mode="json")
    if any(dumped.get(field) != value for field, value in metadata.items()):
        raise ValueError("Batch settings disagree with original run metadata")
    for strategy, run in report.runs.items():
        budget = report.budgets[strategy]
        if (
            run.strategy != strategy
            or run.config != report.config
            or run.limits != budget.limits
            or budget.max_model_calls != budget.limits.model_call_budget
            or run.selection != report.selection
            or run.manifest != report.manifest
            or run.manifest_sha256 != report.manifest_sha256
            or run.mode != report.mode
            or run.strict_strategy != report.strict_strategy
            or run.split != report.split
        ):
            raise ValueError("Per-strategy settings disagree with the declared batch")
        records = [
            EvaluationRecord.model_validate_json(line)
            for line in (directory / strategy / "analysis.jsonl")
            .read_bytes()
            .splitlines()
            if line.strip()
        ]
        if [r.id for r in records] != report.selection.sample_ids:
            raise ValueError(
                "Batch records do not match the complete selected denominator"
            )
        files = {s.id: s.file_name for s in report.selection.samples}
        if any(record.file_name != files[record.id] for record in records):
            raise ValueError("Selected records disagree with their file groups")
        summary = score_records(
            root, report.manifest, records, include_request_usage=True
        )
        if records != run.records or summary != run.summary:
            raise ValueError(
                "Saved predictions or scores disagree with independent official scoring"
            )
    accounting = summarize_request_usage(
        r for run in report.runs.values() for r in run.records
    )
    if accounting != report.request_usage:
        raise ValueError("Batch request accounting disagrees with saved evidence")
    return report


def _paired(
    reference: tuple[str, EvaluationReport],
    candidate: tuple[str, EvaluationReport],
    kind: str,
) -> dict[str, Any]:
    before_name, before = reference
    after_name, after = candidate
    wins, losses, ties = [], [], []
    for a, b in zip(before.records, after.records, strict=True):
        correct_a, correct_b = all(a.correctness.values()), all(b.correctness.values())
        if correct_b and not correct_a:
            wins.append(a.id)
        elif correct_a and not correct_b:
            losses.append(a.id)
        else:
            ties.append(a.id)
    delta: dict[str, Any] = {
        "kind": kind,
        "reference": before_name,
        "candidate": after_name,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }
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
    ):
        delta[field + "_delta"] = getattr(after.summary, field) - getattr(
            before.summary, field
        )
    for field in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
        before_tokens = (
            getattr(before.summary.request_usage.usage, field)
            if before.summary.request_usage
            else None
        )
        after_tokens = (
            getattr(after.summary.request_usage.usage, field)
            if after.summary.request_usage
            else None
        )
        delta[field + "_delta"] = (
            after_tokens - before_tokens
            if before_tokens is not None and after_tokens is not None
            else None
        )
    return delta


def summarize_ablations(
    root: Path,
    batches: dict[str, Path],
    *,
    output_dir: Path,
    selected: tuple[str, Strategy] | None = None,
) -> dict[str, Any]:
    """Offline delivery; selection is an operator decision, never a hindsight oracle."""
    if not batches or any(not label.strip() for label in batches):
        raise ValueError("Supply distinct named completed batches")
    reports = {
        label: _verified_batch(root, directory) for label, directory in batches.items()
    }
    first = next(iter(reports.values()))
    for report in reports.values():
        for field in (
            "mode",
            "model_version",
            "manifest",
            "manifest_sha256",
            "selection",
            "source_sha256",
            "environment",
            "prompt_config",
            "limits",
            "workers",
            "strict_strategy",
        ):
            if getattr(report, field) != getattr(first, field):
                raise ValueError(
                    f"Ablation batches differ in {field}; compare fixed settings"
                )
        for field in type(first.config).model_fields:
            if field not in {
                "thinking",
                "disable_thinking",
                "instruction_mode",
            } and getattr(report.config, field) != getattr(first.config, field):
                raise ValueError(f"Model mode comparison also changed {field}")
    if selected is not None:
        label, strategy = selected
        if label not in reports or strategy not in reports[label].runs:
            raise ValueError(
                "Selected configuration must reference a completed batch strategy"
            )
    comparisons = []
    if len({report.run_id for report in reports.values()}) != len(reports):
        raise ValueError("Repeated batches must contain distinct run identities")
    previous: dict[Strategy, list[tuple[str, ComparisonReport]]] = {}
    for label, report in reports.items():
        for a, b in zip(report.strategy_order, report.strategy_order[1:]):
            comparisons.append(
                _paired(
                    (f"{label}/{a}", report.runs[a]),
                    (f"{label}/{b}", report.runs[b]),
                    "workflow",
                )
            )
        for strategy in report.strategy_order:
            for old_label, old in previous.get(strategy, []):
                kind = "repeat" if old.config == report.config else "model_mode"
                comparisons.append(
                    _paired(
                        (f"{old_label}/{strategy}", old.runs[strategy]),
                        (f"{label}/{strategy}", report.runs[strategy]),
                        kind,
                    )
                )
            previous.setdefault(strategy, []).append((label, report))
    delivery: dict[str, Any] = {
        "purpose": "optimization",
        "selection": first.selection.model_dump() if first.selection else None,
        "batches": {label: report.model_dump() for label, report in reports.items()},
        "comparisons": comparisons,
        "request_usage": summarize_request_usage(
            record
            for report in reports.values()
            for run in report.runs.values()
            for record in run.records
        ).model_dump(),
        "evidence_sha256": {
            label: sha256((directory / "report.json").read_bytes())
            for label, directory in batches.items()
        },
        "selected": list(selected) if selected else None,
        "limitations": [
            "Repeated optimization samples do not measure independent generalization.",
            "Single-run gains can fluctuate; paired repeats retain every regression.",
            "Workflow stages change their action contracts and prompts; differences are not isolated causal effects.",
            "Known usage covers saved requests; missing provider usage remains unknown.",
            "Controlled scores verify functionality and cannot select a live model configuration.",
        ],
    }
    lines = [
        "# Representative ablations",
        "",
        "Purpose: optimization. Fixed denominator per run: "
        + str(first.selection.total if first.selection else 0),
        "",
        "| Batch / strategy | Correct questions | Correct fields | Valid computations | Task seconds | Requests | Executions | Failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, report in reports.items():
        for strategy in report.strategy_order:
            s = report.runs[strategy].summary
            lines.append(
                f"| {label}/{strategy} | {s.correct_questions}/{s.total} | {s.correct_subquestions}/{s.total_subquestions} | {s.execution_successes}/{s.total} | {s.task_seconds:.2f} | {s.model_calls} | {s.execution_attempts} | {s.failures} |"
            )
    lines += [
        "",
        "| Batch / strategy | Known input tokens | Known output tokens | Known cache-read tokens | Unknown input/output/cache requests |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for label, report in reports.items():
        for strategy in report.strategy_order:
            usage = report.runs[strategy].summary.request_usage
            if usage is not None:
                lines.append(
                    f"| {label}/{strategy} | {usage.known_usage.input_tokens} | {usage.known_usage.output_tokens} | {usage.known_usage.cache_read_input_tokens} | {usage.unknown_usage_requests} |"
                )
    lines += [
        "",
        "| Comparison | Kind | Wins | Losses | Ties | Accuracy delta |",
        "| --- | --- | --- | --- | --- | ---: |",
    ]
    for pair in comparisons:
        lines.append(
            f"| {pair['candidate']} vs {pair['reference']} | {pair['kind']} | {pair['wins']} | {pair['losses']} | {len(pair['ties'])} | {pair['question_accuracy_delta']:.4f} |"
        )
    lines += ["", *delivery["limitations"], ""]
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(delivery, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "summary.md").write_text("\n".join(lines))
    if selected is not None:
        label, strategy = selected
        report = reports[label]
        config = {
            "mode": report.mode,
            "strategy": strategy,
            "config": report.config.model_dump(),
            "model_version": report.model_version,
            "limits": report.budgets[strategy].limits.model_dump(),
            "workers": report.workers,
            "strict_strategy": report.strict_strategy,
            "source_sha256": report.source_sha256,
            "environment": report.environment,
            "prompt_config": report.prompt_config,
            "selection": report.selection.model_dump() if report.selection else None,
            "manifest_sha256": report.manifest_sha256,
            "full_evaluation_split": report.split,
            "full_evaluation_sample_ids": report.manifest.splits[report.split],
            "full_evaluation_purpose": "optimization",
            "evidence_batch": label,
            "evidence_sha256": delivery["evidence_sha256"][label],
        }
        (output_dir / "selected-config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n"
        )
    return delivery


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore and compare completed representative optimization batches offline"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--batch",
        nargs=2,
        action="append",
        metavar=("LABEL", "DIRECTORY"),
        required=True,
    )
    parser.add_argument("--select", nargs=2, metavar=("LABEL", "STRATEGY"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        batches = {label: Path(directory) for label, directory in args.batch}
        if len(batches) != len(args.batch):
            raise ValueError("Batch labels must be distinct")
        summarize_ablations(
            args.data,
            batches,
            output_dir=args.output,
            selected=tuple(args.select) if args.select else None,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps({"status": "ablations_rescored", "output": str(args.output)}))


if __name__ == "__main__":
    main()
