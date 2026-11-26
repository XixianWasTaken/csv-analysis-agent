"""Continue a frozen live comparison by replaying its recorded real analyses.

The frozen evaluator still creates and scores the entire comparison. This storage
adapter only returns saved analyses for an exact, unscored prefix; every remaining
sample calls the original analyze function. Original evidence is never modified.
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from csv_agent import evaluation
from csv_agent.analysis import AnalysisResult, Strategy, analyze
from csv_agent.benchmark import load_questions, sha256
from csv_agent.config import ModelConfig, RunLimits
from csv_agent.evaluation import (
    EvaluationRecord,
    analysis_prompt,
    comparison_prompt_config,
)
from csv_agent.evidence import verify_analysis_evidence
from csv_agent.reproducibility import FrozenExperiment, run_final, verify_final
from csv_agent.runtime import capture_runtime, inspect_image

MAX_MEMORY_REPORT_DRIFT_BYTES = 1024 * 1024


def environment_differences(
    frozen: dict[str, str], current: dict[str, str]
) -> dict[str, Any]:
    """Permit reboot memory-report noise, preserving every execution setting."""
    if frozen == current:
        return {}
    error = "Continuation must use the original frozen runtime"
    if set(frozen) != set(current) or any(
        frozen[name] != current[name] for name in frozen if name != "docker"
    ):
        raise ValueError(error)
    before_value = frozen.get("docker", "")
    after_value = current.get("docker", "")
    if not isinstance(before_value, str) or not isinstance(after_value, str):
        raise TypeError(error)
    before = before_value.split("|")
    after = after_value.split("|")
    if (
        len(before) != 5
        or len(after) != 5
        or before[:4] != after[:4]
        or not before[4].isdigit()
        or not after[4].isdigit()
        or min(int(before[4]), int(after[4])) <= 0
        or abs(int(after[4]) - int(before[4])) > MAX_MEMORY_REPORT_DRIFT_BYTES
    ):
        raise ValueError(error)
    return {
        "docker_memory_bytes": {
            "frozen": int(before[4]),
            "current": int(after[4]),
            "delta": int(after[4]) - int(before[4]),
        }
    }


def resume(
    freeze_path: Path,
    data_root: Path,
    manifest_path: Path,
    source: Path,
    destination: Path,
    *,
    check_only: bool = False,
) -> dict[str, Any]:
    if destination.exists() or source.resolve() == destination.resolve():
        raise ValueError("Continuation requires a new output directory")
    if (source / "report.json").exists() or (source / "receipt.json").exists():
        raise ValueError(
            "Completed comparisons must be independently verified, not resumed"
        )
    frozen = FrozenExperiment.model_validate_json(freeze_path.read_bytes())
    runtime = capture_runtime()
    current_limits = RunLimits.from_env(exploration=frozen.exploration)
    if (
        runtime.source_sha256 != frozen.source_sha256
        or ModelConfig.from_env() != frozen.config
        or current_limits.model_copy(
            update={"image": inspect_image(current_limits.image)}
        )
        != frozen.limits
        or sha256(manifest_path.read_bytes()) != frozen.manifest_sha256
    ):
        raise ValueError("Continuation must use the original frozen runtime")
    differences = environment_differences(frozen.environment, runtime.environment)
    development = freeze_path.resolve().parent / frozen.development_dir
    for name, digest in frozen.development_sha256.items():
        if sha256((development / name).read_bytes()) != digest:
            raise ValueError("Development evidence changed")
    original = json.loads((source / "run.json").read_bytes())
    source_differences = environment_differences(
        frozen.environment, original.get("environment", {})
    )
    order = frozen.strategies
    limits = {
        strategy: frozen.limits.for_single_generation()
        if strategy == "single"
        else frozen.limits
        for strategy in order
    }
    expected = {
        "mode": "live",
        "split": frozen.split,
        "model_version": frozen.model_version,
        "config": frozen.config.model_dump(),
        "limits": frozen.limits.model_dump(),
        "requested_limits": frozen.limits.model_dump(),
        "manifest_sha256": frozen.manifest_sha256,
        "manifest": frozen.manifest.model_dump(),
        "source_sha256": frozen.source_sha256,
        "environment": original["environment"],
        "strategy_order": order,
        "budgets": {
            s: {
                "limits": limit.model_dump(),
                "max_model_calls": limit.model_call_budget,
            }
            for s, limit in limits.items()
        },
        "prompt_config": comparison_prompt_config(),
    }
    if frozen.schema_version == 2:
        if frozen.prompt_config != comparison_prompt_config():
            raise ValueError("Frozen prompt changed; start a new batch")
        expected.update(
            selection=frozen.selection.model_dump() if frozen.selection else None,
            prompt_config=frozen.prompt_config,
            workers=frozen.workers,
            strict_strategy=frozen.strict_strategy,
        )
        previous_path = source.with_name(source.name + "-resumption.json")
        source_freeze_hash = sha256(freeze_path.read_bytes())
        if previous_path.exists():
            previous = json.loads(previous_path.read_bytes())
            if previous.get("freeze_sha256") != source_freeze_hash:
                raise ValueError("Previous continuation belongs to another freeze")
            source_freeze_hash = previous["verification_freeze_sha256"]
        expected["freeze_sha256"] = source_freeze_hash
    if any(original.get(name) != value for name, value in expected.items()):
        raise ValueError("Original comparison does not match the frozen experiment")
    questions = load_questions(data_root, frozen.manifest)
    selected = (
        frozen.selection.sample_ids
        if frozen.selection
        else frozen.manifest.splits[frozen.split]
    )
    sample_keys = {}
    for sample_id in selected:
        question = questions[sample_id]
        prompt = analysis_prompt(question)
        key = (frozen.manifest.tables[question.file_name].sha256, prompt)
        if key in sample_keys:
            raise ValueError("Ambiguous final sample; cannot replay saved analyses")
        sample_keys[key] = sample_id

    prefixes: dict[Strategy, list[EvaluationRecord]] = {}
    saved: dict[tuple[str, int], EvaluationRecord] = {}
    evidence_hashes = {"run.json": sha256((source / "run.json").read_bytes())}
    discarded_partial_records: dict[str, dict[str, Any]] = {}
    discarded_incomplete_records: dict[str, dict[str, Any]] = {}
    incomplete_seen = False
    for strategy in order:
        directory = source / strategy
        if not directory.exists():
            prefixes[strategy] = []
            incomplete_seen = True
            continue
        if incomplete_seen:
            raise ValueError("Original strategies are not a fixed-order prefix")
        metadata = json.loads((directory / "run.json").read_bytes())
        for name in ("mode", "split", "config", "manifest", "manifest_sha256"):
            if metadata.get(name) != expected[name]:
                raise ValueError("Original strategy settings disagree with freeze")
        if (
            metadata.get("strategy") != strategy
            or metadata.get("limits") != limits[strategy].model_dump()
        ):
            raise ValueError("Original strategy budget disagrees with freeze")
        if frozen.schema_version == 2 and (
            metadata.get("selection") != expected["selection"]
            or metadata.get("strict_strategy") != frozen.strict_strategy
        ):
            raise ValueError("Original strategy selection or action contract changed")
        raw = (directory / "analysis.jsonl").read_bytes()
        lines = raw.splitlines()
        if raw and not raw.endswith(b"\n"):
            partial = lines.pop()
            discarded_partial_records[strategy] = {
                "bytes": len(partial),
                "sha256": sha256(partial),
            }
        records = [EvaluationRecord.model_validate_json(line) for line in lines]
        if records and records[-1].analysis.status == "running":
            record = records[-1]
            if len(records) > len(selected) or record.id != selected[len(records) - 1]:
                raise ValueError("Incomplete tail is not the next selected task")
            discarded_incomplete_records[strategy] = {
                "id": record.id,
                "bytes": len(lines[-1]),
                "sha256": sha256(lines[-1]),
            }
            records.pop()
        if strategy in discarded_partial_records and len(records) >= len(selected):
            raise ValueError("Unexpected partial record after a completed strategy")
        if [r.id for r in records] != selected[: len(records)]:
            raise ValueError("Only an exact, complete-record prefix can be resumed")
        for record in records:
            if (
                record.file_name != questions[record.id].file_name
                or record.response is not None
                or record.correctness
                or record.failure not in {None, "data_error"}
                or record.analysis.status == "running"
                or len(record.analysis.attempts) > limits[strategy].model_call_budget
                or record.analysis.execution_attempts > limits[strategy].max_executions
                or (
                    frozen.schema_version == 1
                    and any(
                        attempt.request is not None and attempt.request != frozen.config
                        for attempt in record.analysis.attempts
                    )
                )
            ):
                raise ValueError("Saved analysis is not an unscored frozen live record")
            if frozen.schema_version == 2:
                if (
                    record.analysis.attempts
                    and record.analysis.profile.get("source_sha256")
                    != frozen.manifest.tables[record.file_name].sha256
                ):
                    raise ValueError(
                        "Saved analysis data source differs from the manifest"
                    )
                verify_analysis_evidence(
                    record.analysis,
                    config=frozen.config,
                    limits=limits[strategy],
                    strategy=strategy,
                    answer_format=questions[record.id].format,
                    strict_strategy=frozen.strict_strategy,
                    question=next(
                        prompt
                        for (_, prompt), sample_id in sample_keys.items()
                        if sample_id == record.id
                    ),
                )
            saved[strategy, record.id] = record
        prefixes[strategy] = records
        incomplete_seen = len(records) < len(selected)
        for filename in ("run.json", "analysis.jsonl"):
            name = f"{strategy}/{filename}"
            evidence_hashes[name] = sha256((source / name).read_bytes())
    details: dict[str, Any] = {
        "status": "checked" if check_only else "running",
        "source": str(source),
        "destination": str(destination),
        "original_comparison_run_id": original["run_id"],
        "original_started_at": original["started_at"],
        "resumed_at": datetime.now(UTC).isoformat(),
        "freeze_sha256": sha256(freeze_path.read_bytes()),
        "runtime_differences": differences,
        "source_runtime_differences": source_differences,
        "source_environment": original["environment"],
        "resumed_environment": runtime.environment,
        "memory_report_drift_limit_bytes": MAX_MEMORY_REPORT_DRIFT_BYTES,
        "runner_sha256": sha256(Path(__file__).read_bytes()),
        "source_evidence_sha256": evidence_hashes,
        "discarded_partial_records": discarded_partial_records,
        "discarded_incomplete_records": discarded_incomplete_records,
        "reused_tasks": [
            {
                "strategy": strategy,
                "id": record.id,
                "analysis_run_id": record.analysis.run_id,
                "source": str(source / strategy / "analysis.jsonl"),
                "line": index + 1,
            }
            for strategy, records in prefixes.items()
            for index, record in enumerate(records)
        ],
        "reused_records": {s: len(records) for s, records in prefixes.items()},
        "remaining_records": {s: len(selected) - len(prefixes[s]) for s in order},
        "resource_scope": (
            "Task times, requests, executions and usage count persisted analyses only. "
            "An interrupted in-flight task may have consumed unrecorded resources. "
            "The new report wall_seconds measures this continuation, including replay; "
            "it excludes the original segment and the interruption gap."
        ),
    }
    verification_freeze = freeze_path
    verification_data = freeze_path.read_bytes()
    if differences:
        verification_freeze = destination.with_name(destination.name + "-freeze.json")
        segment_freeze = frozen.model_copy(
            update={
                "frozen_at": details["resumed_at"],
                "environment": runtime.environment,
                "development_dir": os.path.relpath(
                    development, verification_freeze.resolve().parent
                ),
            }
        )
        verification_data = (segment_freeze.model_dump_json(indent=2) + "\n").encode()
    details["verification_freeze"] = str(verification_freeze)
    details["verification_freeze_sha256"] = sha256(verification_data)
    ledger = destination.with_name(destination.name + "-resumption.json")
    if ledger.exists() or (differences and verification_freeze.exists()):
        raise ValueError("Continuation requires new evidence paths")
    previous_ledger = source.with_name(source.name + "-resumption.json")
    if previous_ledger.exists():
        previous_data = previous_ledger.read_bytes()
        details["previous_resumption"] = {
            "path": str(previous_ledger),
            "sha256": sha256(previous_data),
            "record": json.loads(previous_data),
        }
    if check_only:
        return details
    with ledger.open("x") as handle:
        handle.write(json.dumps(details, indent=2) + "\n")
    if differences:
        with verification_freeze.open("xb") as freeze_handle:
            freeze_handle.write(verification_data)
    original_analyze = analyze
    replayed: set[tuple[str, int]] = set()

    def analyze_or_replay(data: bytes, question: str, **kwargs: Any) -> AnalysisResult:
        strategy = kwargs["strategy"]
        if (
            kwargs["config"] != frozen.config
            or kwargs["limits"] != limits[strategy]
            or kwargs.get("model") is not None
        ):
            raise ValueError("Continuation changed the analysis interface")
        sample_id = sample_keys[sha256(data), question]
        key = strategy, sample_id
        if key in saved:
            if key in replayed:
                raise ValueError("Saved sample was replayed more than once")
            replayed.add(key)
            return saved[key].analysis.model_copy(deep=True)
        print(
            json.dumps({"event": "start", "strategy": strategy, "id": sample_id}),
            flush=True,
        )
        result = original_analyze(data, question, **kwargs)
        print(
            json.dumps(
                {
                    "event": "finish",
                    "strategy": strategy,
                    "id": sample_id,
                    "status": result.status,
                }
            ),
            flush=True,
        )
        return result

    # The evaluator privately imports analyze; only this fresh CLI process replaces it.
    evaluation.analyze = analyze_or_replay  # type: ignore[attr-defined]
    try:
        report = run_final(
            verification_freeze,
            data_root,
            manifest_path,
            destination,
            saved_records=prefixes if frozen.schema_version == 2 else None,
        )
    finally:
        evaluation.analyze = original_analyze  # type: ignore[attr-defined]
    if frozen.schema_version == 2:
        replayed = {
            (strategy, record.id)
            for strategy, run in report.runs.items()
            for record in run.records
            if (strategy, record.id) in saved
            and record.analysis == saved[strategy, record.id].analysis
        }
    if replayed != set(saved):
        raise ValueError("Not every recorded real analysis was reused exactly once")
    for name, digest in evidence_hashes.items():
        if sha256((source / name).read_bytes()) != digest:
            raise ValueError("Original evidence changed during continuation")
    if sha256(freeze_path.read_bytes()) != details["freeze_sha256"]:
        raise ValueError("Original freeze changed during continuation")
    for strategy, records in prefixes.items():
        continued = [
            EvaluationRecord.model_validate_json(line)
            for line in (destination / strategy / "analysis.jsonl")
            .read_bytes()
            .splitlines()
        ]
        if continued[: len(records)] != records:
            raise ValueError("Replayed raw records differ from original real evidence")
    verify_final(verification_freeze, data_root, destination)
    finished = datetime.now(UTC)
    details.update(
        status="verified",
        final_run_id=report.run_id,
        finished_at=finished.isoformat(),
        resumed_wall_seconds=report.wall_seconds,
        elapsed_since_original_start_seconds=(
            finished - datetime.fromisoformat(original["started_at"])
        ).total_seconds(),
    )
    temporary = ledger.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(details, indent=2) + "\n")
    temporary.replace(ledger)
    return details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/dabench/manifest.json")
    )
    parser.add_argument(
        "--check", action="store_true", help="Validate only; no model requests"
    )
    args = parser.parse_args()
    load_dotenv()
    result = resume(
        args.freeze,
        args.data,
        args.manifest,
        args.source,
        args.output,
        check_only=args.check,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
