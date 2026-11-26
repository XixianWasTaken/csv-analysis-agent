"""Development evidence and frozen final comparison, separate from controlled demos."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from csv_agent._vendor import dabench
from csv_agent.analysis import STRATEGY_POLICIES, Strategy
from csv_agent.benchmark import Manifest, Split, load_questions, sha256
from csv_agent.config import ModelConfig, RunLimits
from csv_agent.evaluation import (
    ComparisonReport,
    EvaluationRecord,
    analysis_prompt,
    comparison_prompt_config,
    run_comparison,
    score_records,
    summarize_request_usage,
)
from csv_agent.evaluation_scope import EvaluationSelection, Purpose, select_samples
from csv_agent.evidence import verify_analysis_evidence
from csv_agent.runtime import capture_runtime, inspect_image


class FrozenExperiment(BaseModel):
    schema_version: Literal[1, 2] = 1
    frozen_at: str
    development_dir: str
    development_sha256: dict[str, str]
    config: ModelConfig
    limits: RunLimits
    model_version: str
    manifest_sha256: str
    manifest: Manifest
    include_ablation: bool
    source_sha256: dict[str, str]
    environment: dict[str, str]
    split: Split = "final"
    strategy_order: list[Strategy] | None = None
    selection: EvaluationSelection | None = None
    prompt_config: dict[str, object] | None = None
    workers: int = 1
    strict_strategy: bool = False

    @property
    def strategies(self) -> list[Strategy]:
        if self.schema_version == 2:
            if (
                not self.strategy_order
                or self.selection is None
                or self.prompt_config is None
            ):
                raise ValueError("Incomplete new-flow freeze")
            return self.strategy_order
        order: list[Strategy] = ["single", "tool_loop", "langgraph"]
        if self.include_ablation:
            order.append("langgraph_no_repair")
        return order

    @property
    def exploration(self) -> bool:
        return self.schema_version == 2 and any(
            s in {"langgraph_explore", "langgraph_verify", "langgraph_resolve"}
            for s in self.strategies
        )


class CompletionReceipt(BaseModel):
    freeze_sha256: str
    artifacts_sha256: dict[str, str]


def _artifacts(report: ComparisonReport) -> list[str]:
    return ["run.json", "report.json", "paired.jsonl", "summary.md"] + [
        f"{strategy}/{name}"
        for strategy in report.strategy_order
        for name in ("run.json", "report.json", "analysis.jsonl", "results.jsonl")
    ]


def _complete(report: ComparisonReport) -> None:
    ids = (
        report.selection.sample_ids
        if report.selection
        else report.manifest.splits[report.split]
    )
    if (
        not report.strategy_order
        or len(set(report.strategy_order)) != len(report.strategy_order)
        or any(s not in STRATEGY_POLICIES for s in report.strategy_order)
        or set(report.budgets) != set(report.strategy_order)
    ):
        raise ValueError("Incomplete comparison strategies")
    if report.selection and report.selection != select_samples(
        report.manifest, report.split, ids, report.selection.purpose
    ):
        raise ValueError("Incomplete comparison selection")
    if (
        set(report.runs) != set(report.strategy_order)
        or [r.id for r in report.paired_records] != ids
    ):
        raise ValueError("Incomplete paired comparison")
    identities = [
        record.analysis.run_id for run in report.runs.values() for record in run.records
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Different analysis tasks share a saved request identity")
    for strategy, run in report.runs.items():
        if (
            [r.id for r in run.records] != ids
            or run.summary.total != len(ids)
            or run.mode != report.mode
            or run.split != report.split
            or run.config != report.config
            or run.strategy != strategy
            or run.selection != report.selection
            or run.strict_strategy != report.strict_strategy
            or run.manifest != report.manifest
            or run.limits != report.budgets[strategy].limits
            or report.budgets[strategy].limits
            != (
                report.limits.for_single_generation()
                if strategy == "single"
                else report.limits
            )
            or report.budgets[strategy].max_model_calls != run.limits.model_call_budget
            or run.manifest_sha256 != report.manifest_sha256
        ):
            raise ValueError("Incomplete comparison denominator or settings")


def _check_saved_artifacts(report: ComparisonReport, directory: Path) -> None:
    """Compare the independent raw, strategy and paired copies of the evidence."""
    if json.loads((directory / "run.json").read_bytes()) != report.model_dump(
        mode="json",
        exclude={"wall_seconds", "runs", "paired_records", "deltas", "request_usage"},
    ):
        raise ValueError("Saved comparison metadata disagrees with report evidence")
    pairs = [
        json.loads(line)
        for line in (directory / "paired.jsonl").read_bytes().splitlines()
    ]
    if pairs != [pair.model_dump(mode="json") for pair in report.paired_records]:
        raise ValueError("Paired file disagrees with report evidence")
    for strategy, run in report.runs.items():
        path = directory / strategy
        if json.loads((path / "run.json").read_bytes()) != run.model_dump(
            mode="json", exclude={"wall_seconds", "records", "summary"}
        ):
            raise ValueError("Saved strategy metadata disagrees with report evidence")
        if json.loads((path / "report.json").read_bytes()) != run.model_dump(
            mode="json"
        ):
            raise ValueError("Strategy report disagrees with comparison evidence")
        raw = [
            EvaluationRecord.model_validate_json(line)
            for line in (path / "analysis.jsonl").read_bytes().splitlines()
        ]
        scored = [
            EvaluationRecord.model_validate_json(line)
            for line in (path / "results.jsonl").read_bytes().splitlines()
        ]
        if (
            scored != run.records
            or [r.id for r in raw] != [r.id for r in run.records]
            or any(
                a.analysis != b.analysis
                or a.file_name != b.file_name
                or a.response is not None
                or a.correctness
                or a.failure not in {None, "data_error"}
                for a, b in zip(raw, run.records)
            )
        ):
            raise ValueError("Raw analysis evidence disagrees with report")


def freeze_development(development_dir: Path, output: Path) -> None:
    report = ComparisonReport.model_validate_json(
        (development_dir / "report.json").read_bytes()
    )
    if report.mode != "live" or report.split != "development":
        raise ValueError("Freeze requires a complete live development comparison")
    if report.strategy_order not in (
        ["single", "tool_loop", "langgraph"],
        ["single", "tool_loop", "langgraph", "langgraph_no_repair"],
    ):
        raise ValueError("Use freeze-batch for selected new-flow strategies")
    _complete(report)
    if any(run.summary.execution_successes == 0 for run in report.runs.values()):
        raise ValueError("Development must demonstrate computation in every strategy")
    runtime = capture_runtime()
    if (
        runtime.source_sha256 != report.source_sha256
        or runtime.environment != report.environment
    ):
        raise ValueError(
            "Source or environment changed since development; start a new experiment"
        )
    image = inspect_image(report.requested_limits.image)
    if image != report.limits.image:
        raise ValueError("Docker image changed since development")
    frozen = FrozenExperiment(
        frozen_at=datetime.now(UTC).isoformat(),
        development_dir=os.path.relpath(
            development_dir.resolve(), output.resolve().parent
        ),
        development_sha256={
            name: sha256((development_dir / name).read_bytes())
            for name in _artifacts(report)
        },
        config=report.config,
        limits=report.limits,
        model_version=report.model_version,
        manifest_sha256=report.manifest_sha256,
        manifest=report.manifest,
        include_ablation="langgraph_no_repair" in report.strategy_order,
        source_sha256=report.source_sha256,
        environment=report.environment,
    )
    with output.open("x") as handle:
        handle.write(frozen.model_dump_json(indent=2) + "\n")


def freeze_batch(
    evidence_dir: Path,
    output: Path,
    *,
    split: Split = "final",
    sample_ids: list[int] | None = None,
    purpose: Purpose = "optimization",
) -> None:
    """Freeze an explicitly selected next batch from complete live flow evidence."""
    report = ComparisonReport.model_validate_json(
        (evidence_dir / "report.json").read_bytes()
    )
    _complete(report)
    _check_saved_artifacts(report, evidence_dir)
    if report.prompt_config != comparison_prompt_config():
        raise ValueError("Evidence prompt differs from current source")
    if report.mode != "live" or any(
        run.summary.execution_successes == 0 for run in report.runs.values()
    ):
        raise ValueError("Freeze requires complete live computation evidence")
    runtime = capture_runtime()
    if (
        runtime.source_sha256 != report.source_sha256
        or runtime.environment != report.environment
        or inspect_image(report.requested_limits.image) != report.limits.image
    ):
        raise ValueError("Source or environment changed; start a new evidence batch")
    frozen = FrozenExperiment(
        schema_version=2,
        frozen_at=datetime.now(UTC).isoformat(),
        development_dir=os.path.relpath(
            evidence_dir.resolve(), output.resolve().parent
        ),
        development_sha256={
            name: sha256((evidence_dir / name).read_bytes())
            for name in _artifacts(report)
        },
        config=report.config,
        limits=report.limits,
        model_version=report.model_version,
        manifest_sha256=report.manifest_sha256,
        manifest=report.manifest,
        include_ablation="langgraph_no_repair" in report.strategy_order,
        source_sha256=report.source_sha256,
        environment=report.environment,
        split=split,
        strategy_order=report.strategy_order,
        selection=select_samples(report.manifest, split, sample_ids, purpose),
        prompt_config=report.prompt_config,
        workers=report.workers,
        strict_strategy=report.strict_strategy,
    )
    with output.open("x") as handle:
        handle.write(frozen.model_dump_json(indent=2) + "\n")


def run_final(
    freeze_path: Path,
    root: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    saved_records: dict[Strategy, list[EvaluationRecord]] | None = None,
) -> ComparisonReport:
    freeze_data = freeze_path.read_bytes()
    frozen = FrozenExperiment.model_validate_json(freeze_data)
    if frozen.schema_version == 2 and (
        frozen.prompt_config != comparison_prompt_config()
        or frozen.selection is None
        or frozen.selection
        != select_samples(
            frozen.manifest,
            frozen.split,
            frozen.selection.sample_ids,
            frozen.selection.purpose,
        )
    ):
        raise ValueError("Frozen prompt or selection changed; start a new batch")
    development = freeze_path.resolve().parent / frozen.development_dir
    for name, digest in frozen.development_sha256.items():
        if sha256((development / name).read_bytes()) != digest:
            raise ValueError("Development evidence changed; start a new experiment")
    current_limits = RunLimits.from_env(exploration=frozen.exploration)
    image = inspect_image(current_limits.image)
    runtime = capture_runtime()
    if (
        ModelConfig.from_env() != frozen.config
        or current_limits.model_copy(update={"image": image}) != frozen.limits
        or sha256(manifest_path.read_bytes()) != frozen.manifest_sha256
        or runtime.source_sha256 != frozen.source_sha256
        or runtime.environment != frozen.environment
    ):
        raise ValueError("Frozen settings changed; start a new development experiment")
    if saved_records is not None:
        questions = load_questions(root, frozen.manifest)
        for strategy, records in saved_records.items():
            if strategy not in frozen.strategies:
                raise ValueError("Saved strategy is outside the frozen batch")
            for record in records:
                question = questions[record.id]
                verify_analysis_evidence(
                    record.analysis,
                    config=frozen.config,
                    limits=frozen.limits.for_single_generation()
                    if strategy == "single"
                    else frozen.limits,
                    strategy=strategy,
                    answer_format=question.format,
                    strict_strategy=frozen.strict_strategy,
                    question=analysis_prompt(question),
                )
    report = run_comparison(
        root,
        manifest_path,
        split=frozen.split,
        output_dir=output_dir,
        config=frozen.config,
        limits=frozen.limits,
        model_version=frozen.model_version,
        include_ablation=frozen.include_ablation,
        strategies=frozen.strategy_order if frozen.schema_version == 2 else None,
        sample_ids=frozen.selection.sample_ids if frozen.selection else None,
        purpose=frozen.selection.purpose if frozen.selection else "optimization",
        workers=frozen.workers,
        strict_strategy=frozen.strict_strategy,
        freeze_sha256=sha256(freeze_data) if frozen.schema_version == 2 else None,
        saved_records=saved_records,
    )
    runtime = capture_runtime()
    if (
        runtime.source_sha256 != frozen.source_sha256
        or runtime.environment != frozen.environment
        or freeze_path.read_bytes() != freeze_data
    ):
        raise ValueError(
            "Frozen settings changed during evaluation; raw evidence retained"
        )
    (output_dir / "frozen.json").write_bytes(freeze_data)
    receipt = CompletionReceipt(
        freeze_sha256=sha256(freeze_data),
        artifacts_sha256={
            name: sha256((output_dir / name).read_bytes())
            for name in _artifacts(report) + ["frozen.json"]
        },
    )
    (output_dir / "receipt.json").write_text(receipt.model_dump_json(indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze development, run final comparison, or verify saved evidence"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze", help="No model calls; requires completed live development"
    )
    freeze.add_argument("--development", type=Path, required=True)
    freeze.add_argument(
        "--output", type=Path, required=True, help="New freeze JSON file"
    )
    batch = commands.add_parser(
        "freeze-batch",
        help="Freeze selected next-batch settings from complete live flow evidence; no model calls",
    )
    batch.add_argument("--evidence", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--split", choices=("development", "final"), default="final")
    batch.add_argument(
        "--purpose", choices=("optimization", "holdout"), default="optimization"
    )
    batch.add_argument(
        "--sample-list",
        type=Path,
        help="JSON integer array; omitted means the complete split",
    )
    for name in ("final", "verify"):
        command = commands.add_parser(
            name,
            help="Consumes model credits"
            if name == "final"
            else "No model calls; repeats official scoring",
        )
        command.add_argument("--freeze", type=Path, required=True)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "final":
            command.add_argument(
                "--manifest",
                type=Path,
                default=Path("benchmarks/dabench/manifest.json"),
            )
    args = parser.parse_args()
    try:
        if args.command in {"freeze", "freeze-batch"}:
            if args.command == "freeze":
                freeze_development(args.development, args.output)
            else:
                freeze_batch(
                    args.evidence,
                    args.output,
                    split=args.split,
                    purpose=args.purpose,
                    sample_ids=json.loads(args.sample_list.read_bytes())
                    if args.sample_list
                    else None,
                )
            print(json.dumps({"status": "frozen"}))
        else:
            if args.command == "final":
                load_dotenv()
                report = run_final(args.freeze, args.data, args.manifest, args.output)
            else:
                report = verify_final(args.freeze, args.data, args.output)
            print(
                json.dumps(
                    {
                        "status": "completed"
                        if args.command == "final"
                        else "verified",
                        "run_id": report.run_id,
                        "summaries": {
                            s: r.summary.model_dump() for s, r in report.runs.items()
                        },
                    },
                    indent=2,
                )
            )
    except (OSError, ValueError):
        parser.exit(
            2,
            "Frozen experiment failed: check complete live evidence, unchanged settings and environment, pinned inputs, and new output paths\n",
        )


def verify_final(freeze_path: Path, root: Path, output_dir: Path) -> ComparisonReport:
    """Verify complete evidence and independently repeat host-only official scoring."""
    freeze_data = freeze_path.read_bytes()
    frozen = FrozenExperiment.model_validate_json(freeze_data)
    receipt = CompletionReceipt.model_validate_json(
        (output_dir / "receipt.json").read_bytes()
    )
    if sha256(freeze_data) != receipt.freeze_sha256:
        raise ValueError("Freeze changed since final evaluation")
    report = ComparisonReport.model_validate_json(
        (output_dir / "report.json").read_bytes()
    )
    names = _artifacts(report) + ["frozen.json"]
    if set(receipt.artifacts_sha256) != set(names):
        raise ValueError("Incomplete final evidence receipt")
    for name in names:
        if sha256((output_dir / name).read_bytes()) != receipt.artifacts_sha256[name]:
            raise ValueError("Final evidence changed or is incomplete")
    _complete(report)
    if (
        report.mode != "live"
        or report.split != frozen.split
        or report.config != frozen.config
        or report.limits != frozen.limits
        or report.manifest_sha256 != frozen.manifest_sha256
        or report.manifest != frozen.manifest
        or report.model_version != frozen.model_version
        or report.source_sha256 != frozen.source_sha256
        or report.environment != frozen.environment
        or ("langgraph_no_repair" in report.strategy_order) != frozen.include_ablation
        or report.strategy_order != frozen.strategies
        or (
            frozen.schema_version == 2
            and (
                report.selection != frozen.selection
                or report.prompt_config != frozen.prompt_config
                or report.workers != frozen.workers
                or report.strict_strategy != frozen.strict_strategy
                or report.freeze_sha256 != sha256(freeze_data)
            )
        )
    ):
        raise ValueError("Final comparison does not match frozen settings")
    if (
        sha256(Path(dabench.__file__).read_bytes())
        != report.manifest.scorer.vendored_sha256
    ):
        raise ValueError("Installed scorer changed")
    if frozen.schema_version == 2:
        questions = load_questions(root, frozen.manifest)
        _check_saved_artifacts(report, output_dir)
        for strategy, run in report.runs.items():
            for record in run.records:
                question = questions[record.id]
                public_prompt = analysis_prompt(question)
                if record.file_name != question.file_name or (
                    record.analysis.attempts
                    and record.analysis.profile.get("source_sha256")
                    != frozen.manifest.tables[question.file_name].sha256
                ):
                    raise ValueError(
                        "Saved analysis data source differs from the manifest"
                    )
                verify_analysis_evidence(
                    record.analysis,
                    config=frozen.config,
                    limits=run.limits,
                    strategy=strategy,
                    answer_format=questions[record.id].format,
                    strict_strategy=frozen.strict_strategy,
                    question=public_prompt,
                )
    for run in report.runs.values():
        records = [r.model_copy(deep=True) for r in run.records]
        if (
            score_records(
                root,
                report.manifest,
                records,
                include_request_usage=run.summary.request_usage is not None,
            )
            != run.summary
            or records != run.records
        ):
            raise ValueError("Official rescoring disagrees with saved results")
    if (
        report.request_usage is not None
        and report.request_usage
        != summarize_request_usage(
            record for run in report.runs.values() for record in run.records
        )
    ):
        raise ValueError("Request usage disagrees with saved analysis evidence")
    for index, pair in enumerate(report.paired_records):
        if pair.results != {
            s: report.runs[s].records[index] for s in report.strategy_order
        }:
            raise ValueError("Paired evidence disagrees with strategy results")
    return report


if __name__ == "__main__":
    main()
