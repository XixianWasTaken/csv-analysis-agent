"""Offline historical first-code replay with immutable source evidence."""

import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from csv_agent._vendor import dabench
from csv_agent.analysis import (
    AnalysisResult,
    Strategy,
    analyze,
    collect_computed_result,
)
from csv_agent.answers import AnswerFormat
from csv_agent.benchmark import Manifest, Split, load_questions, sha256, verified_bytes
from csv_agent.config import RunLimits
from csv_agent.csv_input import inspect_csv
from csv_agent.evaluation import (
    ComparisonReport,
    EvaluationRecord,
    EvaluationReport,
    EvaluationSummary,
    score_records,
)
from csv_agent.evaluation_scope import EvaluationSelection, select_samples
from csv_agent.model import Usage
from csv_agent.runtime import capture_runtime, inspect_image


class ReplayOrigin(BaseModel):
    id: int
    file_name: str
    source_run_id: str
    attempt_number: int | None
    code_sha256: str | None
    disposition: Literal["replayed", "not_replayed_no_code", "input_failed"]


class ReplayRun(BaseModel):
    origins: list[ReplayOrigin]
    records: list[EvaluationRecord]
    summary: EvaluationSummary
    historical_summary: EvaluationSummary
    first_code_summary: EvaluationSummary
    execution_statuses: dict[str, int]
    result_contract_statuses: dict[str, int]
    wins: int
    losses: int
    ties: int


class ReplayReport(BaseModel):
    evidence_kind: Literal["historical_code_replay"] = "historical_code_replay"
    run_id: str
    started_at: str
    source_dir: str
    source_run_id: str
    source_mode: Literal["live", "controlled"]
    source_artifacts_sha256: dict[str, str]
    source_sha256: dict[str, str]
    environment: dict[str, str]
    split: Split
    manifest: Manifest
    manifest_sha256: str
    selection: EvaluationSelection
    strategy_order: list[Strategy]
    limits: RunLimits
    wall_seconds: float
    runs: dict[Strategy, ReplayRun] = Field(default_factory=dict)


def _read_source(directory: Path) -> tuple[ComparisonReport, dict[str, bytes]]:
    report_data = (directory / "report.json").read_bytes()
    report = ComparisonReport.model_validate_json(report_data)
    if len(set(report.strategy_order)) != len(report.strategy_order) or set(
        report.strategy_order
    ) != set(report.runs):
        raise ValueError("Invalid historical strategy inventory")
    names = ["run.json", "report.json", "paired.jsonl", "summary.md"]
    for strategy in report.strategy_order:
        names.extend(
            f"{strategy}/{name}"
            for name in ("run.json", "report.json", "analysis.jsonl", "results.jsonl")
        )
    names.extend(
        name for name in ("frozen.json", "receipt.json") if (directory / name).exists()
    )
    files = {name: (directory / name).read_bytes() for name in names}
    if files["report.json"] != report_data:
        raise ValueError("Historical evidence changed while reading")
    if "receipt.json" in files:
        receipt = json.loads(files["receipt.json"])
        hashes = {
            name: sha256(data) for name, data in files.items() if name != "receipt.json"
        }
        if receipt["artifacts_sha256"] != hashes or receipt[
            "freeze_sha256"
        ] != hashes.get("frozen.json"):
            raise ValueError("Historical frozen receipt disagrees with source files")
    samples = {s.id: s.file_name for s in report.manifest.samples}
    for strategy, run in report.runs.items():
        selected = (
            run.selection.sample_ids
            if run.selection
            else run.manifest.splits[run.split]
        )
        raw = [
            EvaluationRecord.model_validate_json(line)
            for line in files[f"{strategy}/analysis.jsonl"].splitlines()
            if line.strip()
        ]
        scored = [
            EvaluationRecord.model_validate_json(line)
            for line in files[f"{strategy}/results.jsonl"].splitlines()
            if line.strip()
        ]
        group = EvaluationReport.model_validate_json(files[f"{strategy}/report.json"])
        if (
            [r.id for r in raw] != selected
            or [r.id for r in run.records] != selected
            or run != group
            or scored != run.records
            or run.manifest != report.manifest
            or run.manifest_sha256 != report.manifest_sha256
            or run.split != report.split
            or any(
                a.analysis != b.analysis or a.id != b.id or a.file_name != b.file_name
                for a, b in zip(raw, run.records)
            )
            or any(r.file_name != samples.get(r.id) for r in raw)
        ):
            raise ValueError("Historical analysis records disagree with source report")
    return report, files


def _first_code(record: EvaluationRecord) -> tuple[int | None, str | None]:
    for attempt in record.analysis.attempts:
        if attempt.code.strip():
            return attempt.number, attempt.code
    if record.analysis.code.strip():
        raise ValueError("Historical code has no originating attempt")
    return None, None


def _selected_history(
    source: ComparisonReport, selection: EvaluationSelection, strategies: list[Strategy]
) -> dict[Strategy, list[EvaluationRecord]]:
    if (
        not strategies
        or len(strategies) != len(set(strategies))
        or set(strategies) - set(source.runs)
    ):
        raise ValueError("Select unique historical strategies available in the source")
    result = {}
    for strategy in strategies:
        by_id = {r.id: r for r in source.runs[strategy].records}
        if set(selection.sample_ids) - set(by_id):
            raise ValueError("Selected task has no historical record")
        result[strategy] = [
            by_id[i].model_copy(deep=True) for i in selection.sample_ids
        ]
        for record in result[strategy]:
            _first_code(record)
    return result


def _first_code_records(records: list[EvaluationRecord]) -> list[EvaluationRecord]:
    """Score first-attempt artifacts separately from the final repaired history."""
    result = []
    for record in records:
        number, code = _first_code(record)
        analysis = AnalysisResult(run_id=record.analysis.run_id, status="failed")
        if code is not None:
            attempt = next(a for a in record.analysis.attempts if a.number == number)
            execution = attempt.execution
            analysis.execution = execution
            if execution:
                analysis.status = execution.status
                try:
                    envelope = json.loads(execution.output)
                    if envelope.get("error"):
                        analysis.status = "failed"
                    else:
                        # The historical collector's final answer is authoritative
                        # only when this was the attempt it actually published.
                        if (
                            record.analysis.status == "completed"
                            and record.analysis.code == code
                            and record.analysis.execution == execution
                        ):
                            analysis.answer = record.analysis.answer
                        else:
                            analysis.status = "failed"
                except (ValueError, AttributeError):
                    analysis.status = "failed"
        result.append(
            EvaluationRecord(
                id=record.id, file_name=record.file_name, analysis=analysis
            )
        )
    return result


def _summarize(
    root: Path, report: ReplayReport, histories: dict[Strategy, list[EvaluationRecord]]
) -> None:
    for strategy in report.strategy_order:
        run = report.runs[strategy]
        historical = histories[strategy]
        saved_history = [r.model_copy(deep=True) for r in historical]
        run.historical_summary = score_records(
            root, report.manifest, historical, include_request_usage=True
        )
        if saved_history != historical:
            raise ValueError("Historical official scores disagree with saved source")
        run.first_code_summary = score_records(
            root,
            report.manifest,
            _first_code_records(historical),
            include_request_usage=True,
        )
        run.summary = score_records(
            root, report.manifest, run.records, include_request_usage=True
        )
        run.execution_statuses = dict(
            Counter(
                r.analysis.execution.status if r.analysis.execution else "not_executed"
                for r in run.records
            )
        )
        run.result_contract_statuses = dict(
            Counter(
                "valid"
                if r.analysis.status == "completed"
                else "invalid"
                if r.analysis.execution and r.analysis.execution.status == "completed"
                else "not_collected"
                for r in run.records
            )
        )
        run.wins = sum(
            all(r.correctness.values()) and not all(h.correctness.values())
            for r, h in zip(run.records, historical)
        )
        run.losses = sum(
            all(h.correctness.values()) and not all(r.correctness.values())
            for r, h in zip(run.records, historical)
        )
        run.ties = report.selection.total - run.wins - run.losses


def _markdown(report: ReplayReport) -> str:
    lines = [
        "# Isolated replay of historical first code",
        "",
        f"Source batch：`{report.source_run_id}`（{report.source_mode}）；Purpose：optimization。",
        f"Fixed question count：{report.selection.total}；ID：{report.selection.sample_ids}。",
        f"New image：`{report.limits.image}`；Wall time {report.wall_seconds:.2f} seconds；New model calls: 0。",
        "",
        "This is diagnostic evidence from the original first code, not a new model score.",
        "Final answers may include later repairs; first-code results reflect accepted original outputs.",
        "Execution status, result validity, and official answer scoring are recorded separately.",
        "",
        "| Strategy | Total | No code | Executed | Valid | Original final correct | Original first correct | Replay correct | Fields correct | wins/losses/ties |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for strategy, run in report.runs.items():
        missing = sum(o.disposition == "not_replayed_no_code" for o in run.origins)
        lines.append(
            f"| {strategy} | {run.summary.total} | {missing} | {run.execution_statuses.get('completed', 0)} | {run.summary.execution_successes} | {run.historical_summary.correct_questions} | {run.first_code_summary.correct_questions} | {run.summary.correct_questions} | {run.summary.correct_subquestions}/{run.summary.total_subquestions} | {run.wins}/{run.losses}/{run.ties} |"
        )
    lines.extend(
        [
            "",
            "Failures, timeouts, rejected inputs and missing code remain in the denominator. Replay does not repair code; differences cannot be attributed to a single change.",
            "",
        ]
    )
    return "\n".join(lines)


def run_replay(
    root: Path,
    manifest_path: Path,
    *,
    source_dir: Path,
    output_dir: Path,
    split: Split,
    sample_ids: list[int],
    strategies: list[Strategy],
    limits: RunLimits,
) -> ReplayReport:
    started = time.monotonic()
    manifest_data = manifest_path.read_bytes()
    manifest = Manifest.model_validate_json(manifest_data)
    selection = select_samples(manifest, split, sample_ids, "optimization")
    if sha256(Path(dabench.__file__).read_bytes()) != manifest.scorer.vendored_sha256:
        raise ValueError("Installed scorer differs from locked version")
    source, files = _read_source(source_dir)
    if (
        source.manifest != manifest
        or source.manifest_sha256 != sha256(manifest_data)
        or source.split != split
    ):
        raise ValueError("Historical dataset or split differs from selected manifest")
    histories = _selected_history(source, selection, strategies)
    questions = load_questions(root, manifest)
    if (
        source_dir.resolve() in output_dir.resolve().parents
        or output_dir.resolve() in source_dir.resolve().parents
        or output_dir.resolve() == source_dir.resolve()
        or root.resolve() == output_dir.resolve()
        or root.resolve() in output_dir.resolve().parents
    ):
        raise ValueError("Replay output must be separate from source and dataset")
    runtime = capture_runtime()
    limits = limits.for_single_generation().model_copy(
        update={"image": inspect_image(limits.image)}
    )
    report = ReplayReport(
        run_id=uuid4().hex,
        started_at=datetime.now(UTC).isoformat(),
        source_dir=str(source_dir.resolve()),
        source_run_id=source.run_id,
        source_mode=source.mode,
        source_artifacts_sha256={name: sha256(data) for name, data in files.items()},
        source_sha256=runtime.source_sha256,
        environment=runtime.environment,
        split=split,
        manifest=manifest,
        manifest_sha256=sha256(manifest_data),
        selection=selection,
        strategy_order=strategies,
        limits=limits,
        wall_seconds=0,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        path = output_dir / "history" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output_dir / "manifest.json").write_bytes(manifest_data)
    (output_dir / "run.json").write_text(
        report.model_dump_json(exclude={"runs", "wall_seconds"}, indent=2) + "\n"
    )
    zero = Usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=0)
    for strategy in strategies:
        records = []
        origins = []
        directory = output_dir / strategy
        directory.mkdir()
        with (directory / "analysis.jsonl").open("w") as output:
            for historical in histories[strategy]:
                number, code = _first_code(historical)
                analysis = AnalysisResult(
                    run_id=uuid4().hex, status="failed", usage=zero
                )
                origin = ReplayOrigin(
                    id=historical.id,
                    file_name=historical.file_name,
                    source_run_id=historical.analysis.run_id,
                    attempt_number=number,
                    code_sha256=sha256(code.encode()) if code is not None else None,
                    disposition="replayed"
                    if code is not None
                    else "not_replayed_no_code",
                )
                question = questions[historical.id]
                task_started = time.monotonic()
                try:
                    table = manifest.tables[historical.file_name]
                    with (root / "da-dev-tables" / historical.file_name).open(
                        "rb"
                    ) as handle:
                        data = handle.read(
                            min(limits.input_max_bytes, table.size_bytes) + 1
                        )
                    if (
                        len(data) != table.size_bytes
                        or len(data) > limits.input_max_bytes
                        or sha256(data) != table.sha256
                    ):
                        raise ValueError(
                            "Benchmark CSV checksum or input limit mismatch"
                        )
                    if code is None:
                        # Record input recovery without inventing an answer.
                        analysis.profile = inspect_csv(data)
                        analysis.error = analysis.stop_reason = (
                            "Not replayed: no historical code"
                        )
                    else:
                        analysis = analyze(
                            data,
                            question.question + "\n" + question.constraints,
                            limits=limits,
                            strategy="single",
                            answer_format=question.format,
                            historical_code=code,
                        )
                except (OSError, ValueError):
                    analysis.error = "Replay input is missing, changed or invalid"
                    if code is not None:
                        origin.disposition = "input_failed"
                analysis.seconds = time.monotonic() - task_started
                record = EvaluationRecord(
                    id=historical.id,
                    file_name=historical.file_name,
                    analysis=analysis,
                    failure="not_replayed_no_code" if code is None else None,
                )
                records.append(record)
                origins.append(origin)
                output.write(record.model_dump_json() + "\n")
                output.flush()
        placeholder = source.runs[strategy].summary
        report.runs[strategy] = ReplayRun(
            origins=origins,
            records=records,
            summary=placeholder,
            historical_summary=placeholder,
            first_code_summary=placeholder,
            execution_statuses={},
            result_contract_statuses={},
            wins=0,
            losses=0,
            ties=0,
        )
    # Labels are read only after every replay has ended.
    _summarize(root, report, histories)
    report.wall_seconds = time.monotonic() - started
    for strategy, run in report.runs.items():
        (output_dir / strategy / "results.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in run.records)
        )
    (output_dir / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    (output_dir / "summary.md").write_text(_markdown(report))
    artifacts = ["run.json", "report.json", "summary.md", "manifest.json"]
    artifacts += [
        f"{s}/{name}"
        for s in strategies
        for name in ("analysis.jsonl", "results.jsonl")
    ]
    artifacts += [f"history/{name}" for name in files]
    (output_dir / "receipt.json").write_text(
        json.dumps(
            {
                "artifacts_sha256": {
                    name: sha256((output_dir / name).read_bytes()) for name in artifacts
                }
            },
            indent=2,
        )
        + "\n"
    )
    return report


def verify_replay(root: Path, output_dir: Path) -> ReplayReport:
    """No Docker, model, original directory or current runtime is needed."""
    report = ReplayReport.model_validate_json((output_dir / "report.json").read_bytes())
    receipt = json.loads((output_dir / "receipt.json").read_bytes())["artifacts_sha256"]
    names = {"run.json", "report.json", "summary.md", "manifest.json"}
    names.update(
        f"{s}/{name}"
        for s in report.strategy_order
        for name in ("analysis.jsonl", "results.jsonl")
    )
    names.update(f"history/{name}" for name in report.source_artifacts_sha256)
    if set(receipt) != names:
        raise ValueError("Incomplete replay receipt")
    for name in names:
        verified_bytes(output_dir / name, receipt[name])
    source, files = _read_source(output_dir / "history")
    if (
        {n: sha256(d) for n, d in files.items()} != report.source_artifacts_sha256
        or source.run_id != report.source_run_id
        or source.mode != report.source_mode
        or source.manifest != report.manifest
        or source.split != report.split
    ):
        raise ValueError("Replay historical source identity changed")
    manifest_data = verified_bytes(output_dir / "manifest.json", report.manifest_sha256)
    if (
        Manifest.model_validate_json(manifest_data) != report.manifest
        or source.manifest_sha256 != report.manifest_sha256
        or sha256(Path(dabench.__file__).read_bytes())
        != report.manifest.scorer.vendored_sha256
    ):
        raise ValueError("Replay manifest or scorer changed")
    if report.selection != select_samples(
        report.manifest, report.split, report.selection.sample_ids, "optimization"
    ) or set(report.runs) != set(report.strategy_order):
        raise ValueError("Replay selection changed")
    metadata = json.loads((output_dir / "run.json").read_bytes())
    if metadata != report.model_dump(mode="json", exclude={"runs", "wall_seconds"}):
        raise ValueError("Replay run metadata differs from report")
    histories = _selected_history(source, report.selection, report.strategy_order)
    questions = load_questions(root, report.manifest)
    for strategy, run in report.runs.items():
        records = [
            EvaluationRecord.model_validate_json(line)
            for line in (output_dir / strategy / "analysis.jsonl")
            .read_bytes()
            .splitlines()
            if line.strip()
        ]
        results = [
            EvaluationRecord.model_validate_json(line)
            for line in (output_dir / strategy / "results.jsonl")
            .read_bytes()
            .splitlines()
            if line.strip()
        ]
        if (
            [r.id for r in records] != report.selection.sample_ids
            or len(run.origins) != len(records)
            or results != run.records
            or any(
                a.analysis != b.analysis or a.id != b.id or a.file_name != b.file_name
                for a, b in zip(records, run.records)
            )
        ):
            raise ValueError("Replay records differ from saved analysis or denominator")
        for record, origin, historical in zip(
            records, run.origins, histories[strategy]
        ):
            number, code = _first_code(historical)
            if (
                origin.id != historical.id
                or record.file_name != historical.file_name
                or origin.file_name != historical.file_name
                or origin.source_run_id != historical.analysis.run_id
                or origin.attempt_number != number
                or origin.code_sha256
                != (sha256(code.encode()) if code is not None else None)
            ):
                raise ValueError("Replay code origin changed")
            analysis = record.analysis
            if (
                any(
                    a.action != "replay"
                    or a.request is not None
                    or a.response is not None
                    or a.prompt is not None
                    or a.usage is not None
                    or a.code != code
                    for a in analysis.attempts
                )
                or len(analysis.attempts) > 1
                or analysis.execution_attempts > 1
                or analysis.usage
                != Usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=0)
            ):
                raise ValueError(
                    "Replay contains model requests or repaired computation"
                )
            if code is None:
                if (
                    origin.disposition != "not_replayed_no_code"
                    or analysis.status == "completed"
                    or analysis.execution_attempts
                    or analysis.code
                    or analysis.attempts
                ):
                    raise ValueError("No-code task was falsely replayed")
            elif origin.disposition == "not_replayed_no_code" or (
                analysis.execution_attempts and analysis.code != code
            ):
                raise ValueError("Replay did not execute original first code")
            if analysis.status == "completed":
                if (
                    not analysis.execution
                    or analysis.execution.status != "completed"
                    or analysis.execution_attempts != 1
                ):
                    raise ValueError("Completed replay has no successful execution")
                computed, artifacts = collect_computed_result(
                    analysis.execution.output,
                    report.limits,
                    AnswerFormat.from_public_format(questions[record.id].format),
                )
                if (
                    analysis.answer != computed.answer
                    or analysis.answer_fields != computed.answer_fields
                    or analysis.table != computed.table
                    or analysis.charts != artifacts
                ):
                    raise ValueError(
                        "Replay answer or artifacts differ from execution output"
                    )
    expected = report.model_copy(deep=True)
    _summarize(root, expected, histories)
    if expected != report or (output_dir / "summary.md").read_text() != _markdown(
        report
    ):
        raise ValueError("Official replay scoring or summaries disagree")
    return report
