"""New-flow freezing and recovery through batch/HTTP boundaries and real Docker."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from test_candidate_verification import RECOMPUTED_TOTAL, WRONG_TOTAL, review
from test_data_exploration import CSV
from test_evaluation import CONFIG, dataset
from test_solution_disagreements import CHECK_SCOPE, recompute, select

from csv_agent import evaluation, reproducibility
from csv_agent.analysis import analyze
from csv_agent.benchmark import sha256
from csv_agent.config import RunLimits
from csv_agent.evaluation import summarize_request_usage

HTTP_CLIENT = httpx.AsyncClient


@pytest.fixture
def new_flow_batch(tmp_path, monkeypatch, request):
    mode = getattr(request, "param", "tools")
    root, manifest = dataset(tmp_path, count=4, csv_data=CSV)
    lock = json.loads(manifest.read_text())
    final_table = root / "da-dev-tables/final_sales.csv"
    final_table.write_bytes(CSV)
    if mode == "input_failure":
        final_table.write_bytes(CSV * 10)
    questions = [
        json.loads(line) for line in (root / "questions.jsonl").read_text().splitlines()
    ]
    for question in questions[1:]:
        question["file_name"] = "final_sales.csv"
        question["question"] += f" Sample {question['id']}."
        lock["samples"][question["id"]]["file_name"] = "final_sales.csv"
    (root / "questions.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in questions)
    )
    lock["dataset"]["questions_sha256"] = sha256(
        (root / "questions.jsonl").read_bytes()
    )
    lock["tables"]["final_sales.csv"] = lock["tables"]["sales.csv"]
    if mode == "input_failure":
        lock["tables"]["final_sales.csv"] = {
            "sha256": sha256(final_table.read_bytes()),
            "size_bytes": final_table.stat().st_size,
        }
    lock["splits"] = {"development": [0], "final": [1, 2, 3]}
    manifest.write_text(json.dumps(lock))
    monkeypatch.setenv("MODEL_ENDPOINT", CONFIG.endpoint)
    monkeypatch.setenv("MODEL_NAME", CONFIG.name)
    monkeypatch.setenv("MODEL_API_KEY_ENV", "NEW_FLOW_TEST_KEY")
    monkeypatch.setenv("NEW_FLOW_TEST_KEY", "private-new-flow-credential")
    monkeypatch.setenv("MODEL_TEMPERATURE", "1")
    monkeypatch.setenv("MODEL_THINKING", "enabled")
    monkeypatch.setenv("MODEL_DISABLE_THINKING", "false")
    monkeypatch.setenv("MODEL_INSTRUCTION_MODE", "tools")
    for key in ("MAX_MODEL_CALLS", "MAX_EXECUTIONS", "TASK_TIMEOUT_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("INPUT_MAX_BYTES", raising=False)
    if mode == "input_failure":
        monkeypatch.setenv("INPUT_MAX_BYTES", "128")
    if mode == "structured":
        monkeypatch.setenv("MODEL_INSTRUCTION_MODE", "structured")
        monkeypatch.setenv("MODEL_DISABLE_THINKING", "true")
        monkeypatch.delenv("MODEL_THINKING")
    if mode == "short_deadline":
        monkeypatch.setenv("TASK_TIMEOUT_SECONDS", "20")
    from csv_agent.config import ModelConfig

    config = ModelConfig.from_env()
    limits = RunLimits.from_env(exploration=True)
    checks = {
        "field_meaning": "sales holds amounts",
        "filters_and_groups": "All regions contribute",
        "missing_values": "Exclude empty amounts",
        "statistical_definition": "Sum, not mean",
        "algorithm_parameters": "Exact addition",
        "precision": "Integer amounts",
    }
    first = WRONG_TOTAL.replace("df['sales'].sum()", "df['sales'].mean()").replace(
        "{'answer': str(total)}", "{'answer_fields': {'total': total}}"
    )
    independent = RECOMPUTED_TOTAL.replace(
        "if row['region'] == 'east' and row['sales']", "if row['sales']"
    ).replace(
        "{'answer': str(total), 'table': [{'east_sales': total}]}",
        "{'answer_fields': {'total': total}}",
    )
    inspection = CHECK_SCOPE.replace(
        "if row['region'] == 'east' and row['sales']", "if row['sales']"
    )
    actions = [
        ("inspect_data", {"action": "inspect", "code": inspection}),
        ("compute_candidate", {"action": "candidate", "code": first}),
        ("recompute_candidate", recompute(2, independent)),
        ("review_candidate", review(3, checks=checks)),
        ("inspect_data", {"action": "inspect", "code": inspection}),
        (
            "select_candidate",
            select(
                3,
                evidence=[3, 5],
                resolutions={
                    "total": {
                        "finding": "45 is the source sum; 15 was the mean",
                        "evidence": [5],
                    }
                },
            ),
        ),
    ]
    requests = []
    original_http = HTTP_CLIENT

    def provider(request):
        payload = json.loads(request.content)
        requests.append(payload)
        name, arguments = actions[(len(requests) - 1) % len(actions)]
        response = {
            "model": "controlled-http",
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "private thinking is not persisted"},
                {"type": "tool_use", "id": "call", "name": name, "input": arguments},
            ],
        }
        if config.instruction_mode == "structured":
            response["content"] = [{"type": "text", "text": json.dumps(arguments)}]
            response["stop_reason"] = "end_turn"
        if name != "review_candidate":
            response["usage"] = {"input_tokens": 12, "output_tokens": 20}
        return httpx.Response(200, json=response)

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_http(
            transport=httpx.MockTransport(provider), **kwargs
        ),
    )
    evidence = tmp_path / "development"
    evaluation.run_comparison(
        root,
        manifest,
        split="development",
        strategies=["langgraph_resolve"],
        output_dir=evidence,
        config=config,
        limits=limits,
        model_version="HTTP-fixture-not-a-real-score",
        strict_strategy=True,
    )
    return root, manifest, evidence, requests


@pytest.mark.docker
@pytest.mark.parametrize(
    "new_flow_batch", ["tools", "structured", "short_deadline"], indirect=True
)
def test_new_flow_freeze_preserves_selection_modes_and_candidate_evidence(
    tmp_path, new_flow_batch
):
    root, manifest, evidence, requests = new_flow_batch
    freeze = tmp_path / "freeze.json"
    reproducibility.freeze_batch(evidence, freeze, sample_ids=[1, 2])
    frozen = json.loads(freeze.read_text())
    assert frozen["schema_version"] == 2
    assert frozen["strategy_order"] == ["langgraph_resolve"]
    assert frozen["selection"]["sample_ids"] == [1, 2]
    assert frozen["selection"]["purpose"] == "optimization"
    assert frozen["config"]["instruction_mode"] in {"tools", "structured"}
    assert frozen["config"]["thinking"] == (
        "enabled" if frozen["config"]["instruction_mode"] == "tools" else None
    )
    assert frozen["limits"]["max_model_calls"] == 16
    assert frozen["limits"]["max_executions"] == 12
    assert "select_candidate" in frozen["prompt_config"]["tools"]
    output = tmp_path / "final"
    report = reproducibility.run_final(freeze, root, manifest, output)
    assert reproducibility.verify_final(freeze, root, output) == report
    run = report.runs["langgraph_resolve"]
    assert run.summary.total == 2 and run.summary.correct_questions == 2
    assert [r.id for r in run.records] == [1, 2]
    assert len(requests) == 18
    assert run.summary.request_usage.unique_requests == 12
    assert run.summary.request_usage.unknown_usage_requests["input_tokens"] == 2
    assert run.summary.request_usage.usage.input_tokens is None
    for record in run.records:
        result = record.analysis
        assert result.answer == "@total[45]"
        assert result.candidates[0].result.answer == "@total[15]"
        assert result.candidates[1].recomputed_from == 2
        assert result.reviews[0].candidate == 3
        assert result.selection.evidence == [3, 5]
        assert result.observations[1].values["east_total"] == 45
        assert all(
            a.response.thinking_present == (frozen["config"]["thinking"] == "enabled")
            for a in result.attempts
        )
        assert all(
            a.request.timeout_seconds <= frozen["config"]["timeout_seconds"]
            for a in result.attempts
        )
    assert "private-new-flow-credential" not in freeze.read_text()
    assert "private thinking" not in (output / "report.json").read_text()
    assert all(
        "common_answers" not in json.dumps(p["messages"])
        and "correctness" not in json.dumps(p["messages"])
        for p in requests
    )


def resumption_adapter():
    spec = importlib.util.spec_from_file_location(
        "new_flow_resumption", Path("scripts/resume_frozen_experiment.py")
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


@pytest.mark.docker
@pytest.mark.parametrize("tail", ["in_flight", "partial", "running"])
def test_new_flow_resumes_only_durable_tasks_after_interruption(
    tmp_path, monkeypatch, new_flow_batch, tail
):
    root, manifest, evidence, requests = new_flow_batch
    adapter = resumption_adapter()
    freeze = tmp_path / "freeze.json"
    reproducibility.freeze_batch(evidence, freeze, sample_ids=[1, 2])
    source = tmp_path / "interrupted"
    tasks = 0

    def interrupted(data, question, **kwargs):
        nonlocal tasks
        tasks += 1
        if tasks == 2:
            raise RuntimeError("process stopped before persisting task")
        return evaluation_analyze(data, question, **kwargs)

    evaluation_analyze = evaluation.analyze
    with monkeypatch.context() as stop:
        stop.setattr(evaluation, "analyze", interrupted)
        with pytest.raises(RuntimeError, match="process stopped"):
            reproducibility.run_final(freeze, root, manifest, source)
    assert len(requests) == 12
    raw_path = source / "langgraph_resolve/analysis.jsonl"
    if tail == "partial":
        raw_path.write_bytes(raw_path.read_bytes() + b'{"id":2,"analysis":')
    elif tail == "running":
        unfinished = json.loads(raw_path.read_text())
        unfinished["id"] = 2
        unfinished["analysis"]["status"] = "running"
        raw_path.write_text(raw_path.read_text() + json.dumps(unfinished) + "\n")
    originals = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    destination = tmp_path / "continued"
    checked = adapter.resume(
        freeze, root, manifest, source, destination, check_only=True
    )
    assert checked["reused_records"] == {"langgraph_resolve": 1}
    assert checked["remaining_records"] == {"langgraph_resolve": 1}
    assert len(requests) == 12 and not destination.exists()
    details = adapter.resume(freeze, root, manifest, source, destination)
    report = reproducibility.verify_final(
        Path(details["verification_freeze"]), root, destination
    )
    assert details["status"] == "verified"
    assert len(requests) == 18
    run = report.runs["langgraph_resolve"]
    assert run.summary.total == 2 and run.summary.correct_questions == 2
    assert run.summary.request_usage.unique_requests == 12
    assert run.summary.request_usage.known_usage.input_tokens == 120
    assert run.summary.request_usage.usage.input_tokens is None
    assert (destination / "langgraph_resolve/analysis.jsonl").read_bytes().splitlines()[
        0
    ] == originals[Path("langgraph_resolve/analysis.jsonl")].splitlines()[0]
    assert originals == {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    if tail == "partial":
        assert details["discarded_partial_records"]["langgraph_resolve"]["bytes"] == 19
    if tail == "running":
        assert details["discarded_incomplete_records"]["langgraph_resolve"]["id"] == 2
    assert "private-new-flow-credential" not in freeze.read_text()
    assert "private thinking" not in (destination / "report.json").read_text()
    assert all(
        "common_answers" not in json.dumps(p["messages"])
        and "correctness" not in json.dumps(p["messages"])
        for p in requests
    )


@pytest.mark.docker
def test_independent_verification_rejects_candidate_not_supported_by_execution(
    tmp_path, new_flow_batch
):
    root, manifest, evidence, requests = new_flow_batch
    freeze = tmp_path / "freeze.json"
    reproducibility.freeze_batch(evidence, freeze, sample_ids=[1])
    output = tmp_path / "final"
    reproducibility.run_final(freeze, root, manifest, output)
    # A new checksum cannot make a contradictory computed candidate valid.
    report_path = output / "report.json"
    report = json.loads(report_path.read_text())
    for record in [
        report["runs"]["langgraph_resolve"]["records"][0],
        report["paired_records"][0]["results"]["langgraph_resolve"],
    ]:
        record["analysis"]["candidates"][1]["result"]["answer_fields"]["total"] = 999
    report_path.write_text(json.dumps(report))
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts_sha256"]["report.json"] = sha256(report_path.read_bytes())
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="evidence|Candidate|candidate"):
        reproducibility.verify_final(freeze, root, output)
    assert len(requests) == 12


@pytest.mark.docker
def test_consistently_rehashed_records_cannot_bypass_analysis_evidence_checks(
    tmp_path, new_flow_batch
):
    root, manifest, evidence, requests = new_flow_batch
    freeze = tmp_path / "freeze.json"
    reproducibility.freeze_batch(evidence, freeze, sample_ids=[1])
    output = tmp_path / "final"
    reproducibility.run_final(freeze, root, manifest, output)
    originals = {
        p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
    }

    def change_records(value, mutation):
        if isinstance(value, dict):
            if "analysis" in value:
                analysis = value["analysis"]
                if mutation == "candidate":
                    analysis["candidates"][1]["result"]["answer_fields"]["total"] = 999
                elif mutation == "observation":
                    analysis["observations"][1]["values"]["east_total"] = 999
                elif mutation == "review":
                    analysis["reviews"][0]["outcome"] = "needs_recompute"
                elif mutation == "selection":
                    analysis["selection"]["resolutions"] = {}
                elif mutation == "comparison":
                    analysis["comparisons"][0]["fields"]["total"]["equal"] = True
                elif mutation == "final_answer":
                    analysis["answer"] = "@total[999]"
                elif mutation == "executions":
                    analysis["execution_attempts"] = 1
                elif mutation == "usage":
                    analysis["usage"] = {"input_tokens": 60, "output_tokens": 100}
                elif mutation == "wrong_question":
                    prompt = json.loads(analysis["attempts"][0]["prompt"])
                    prompt["question"] = "A different public task"
                    analysis["attempts"][0]["prompt"] = json.dumps(prompt)
            else:
                for child in value.values():
                    change_records(child, mutation)
        elif isinstance(value, list):
            for child in value:
                change_records(child, mutation)

    for mutation in (
        "candidate",
        "observation",
        "review",
        "selection",
        "comparison",
        "final_answer",
        "executions",
        "usage",
        "wrong_question",
    ):
        for name, data in originals.items():
            path = output / name
            if path.suffix == ".json" and name != Path("receipt.json"):
                value = json.loads(data)
                change_records(value, mutation)
                path.write_text(json.dumps(value))
            elif path.suffix == ".jsonl":
                rows = [json.loads(line) for line in data.splitlines()]
                change_records(rows, mutation)
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        receipt = json.loads(originals[Path("receipt.json")])
        receipt["artifacts_sha256"] = {
            name: sha256((output / name).read_bytes())
            for name in receipt["artifacts_sha256"]
        }
        (output / "receipt.json").write_text(json.dumps(receipt))
        with pytest.raises(ValueError):
            reproducibility.verify_final(freeze, root, output)
        for name, data in originals.items():
            (output / name).write_bytes(data)
    assert (
        reproducibility.verify_final(freeze, root, output)
        .runs["langgraph_resolve"]
        .summary.correct_questions
        == 1
    )
    assert len(requests) == 12


@pytest.mark.docker
def test_new_flow_repeated_reboot_continuations_deduplicate_saved_requests(
    tmp_path, monkeypatch, new_flow_batch
):
    root, manifest, evidence, requests = new_flow_batch
    freeze = tmp_path / "freeze.json"
    reproducibility.freeze_batch(evidence, freeze)
    adapter = resumption_adapter()
    source = tmp_path / "first"
    original_analyze = evaluation.analyze

    def after_one():
        calls = 0

        def run(data, question, **kwargs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("shutdown")
            return original_analyze(data, question, **kwargs)

        return run

    with monkeypatch.context() as interrupted:
        interrupted.setattr(evaluation, "analyze", after_one())
        with pytest.raises(RuntimeError, match="shutdown"):
            reproducibility.run_final(freeze, root, manifest, source)
    runtime = adapter.capture_runtime()
    docker = runtime.environment["docker"].split("|")
    docker[-1] = str(int(docker[-1]) + 8192)
    rebooted = runtime.model_copy(
        update={"environment": {**runtime.environment, "docker": "|".join(docker)}}
    )
    for boundary in (adapter, evaluation, reproducibility):
        monkeypatch.setattr(boundary, "capture_runtime", lambda: rebooted)
    middle = tmp_path / "second"
    with monkeypatch.context() as interrupted:
        interrupted.setattr(adapter, "analyze", after_one())
        with pytest.raises(RuntimeError, match="shutdown"):
            adapter.resume(freeze, root, manifest, source, middle)
    output = tmp_path / "third"
    details = adapter.resume(freeze, root, manifest, middle, output)
    report = reproducibility.verify_final(
        Path(details["verification_freeze"]), root, output
    )
    assert details["reused_records"] == {"langgraph_resolve": 2}
    assert details["previous_resumption"]["record"]["reused_records"] == {
        "langgraph_resolve": 1
    }
    assert details["runtime_differences"]["docker_memory_bytes"]["delta"] == 8192
    assert [r["id"] for r in details["reused_tasks"]] == [1, 2]
    records = [
        evaluation.EvaluationRecord.model_validate_json(line)
        for directory in (source, middle, output)
        for line in (directory / "langgraph_resolve/analysis.jsonl")
        .read_bytes()
        .splitlines()
    ]
    usage = summarize_request_usage(records)
    assert usage.unique_requests == 18
    assert usage.duplicate_requests == 18
    assert usage.known_usage.input_tokens == 180
    assert usage.unknown_usage_requests["input_tokens"] == 3
    assert usage.usage.input_tokens is None
    assert report.runs["langgraph_resolve"].summary.total == 3
    assert len(requests) == 24


@pytest.mark.docker
@pytest.mark.parametrize("new_flow_batch", ["input_failure"], indirect=True)
def test_saved_input_failures_are_reused_without_changing_the_denominator(
    tmp_path, new_flow_batch
):
    root, manifest, evidence, requests = new_flow_batch
    freeze = tmp_path / "freeze.json"
    reproducibility.freeze_batch(evidence, freeze, sample_ids=[1, 2])
    completed = tmp_path / "completed"
    report = reproducibility.run_final(freeze, root, manifest, completed)
    assert report.runs["langgraph_resolve"].summary.failures == {"data_error": 2}
    source = tmp_path / "interrupted"
    (source / "langgraph_resolve").mkdir(parents=True)
    for name in ("run.json", "langgraph_resolve/run.json"):
        (source / name).write_bytes((completed / name).read_bytes())
    raw = (completed / "langgraph_resolve/analysis.jsonl").read_bytes().splitlines()[
        0
    ] + b"\n"
    (source / "langgraph_resolve/analysis.jsonl").write_bytes(raw)
    output = tmp_path / "resumed"
    details = resumption_adapter().resume(freeze, root, manifest, source, output)
    resumed = reproducibility.verify_final(
        Path(details["verification_freeze"]), root, output
    )
    assert details["reused_records"] == {"langgraph_resolve": 1}
    assert resumed.runs["langgraph_resolve"].summary.total == 2
    assert resumed.runs["langgraph_resolve"].summary.failures == {"data_error": 2}
    assert (
        len(requests) == 6
    )  # Only the successful development task requested the model.
    assert (output / "langgraph_resolve/analysis.jsonl").read_bytes().splitlines()[
        0
    ] == raw.rstrip(b"\n")


@pytest.mark.docker
def test_changed_freeze_or_source_settings_are_rejected_before_any_new_requests(
    tmp_path, monkeypatch, new_flow_batch
):
    root, manifest, evidence, requests = new_flow_batch
    freeze = tmp_path / "freeze.json"
    reproducibility.freeze_batch(evidence, freeze, sample_ids=[1, 2])
    source = tmp_path / "interrupted"
    original_analyze = evaluation.analyze
    calls = 0

    def interrupted(data, question, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("shutdown")
        return original_analyze(data, question, **kwargs)

    with monkeypatch.context() as stop:
        stop.setattr(evaluation, "analyze", interrupted)
        with pytest.raises(RuntimeError, match="shutdown"):
            reproducibility.run_final(freeze, root, manifest, source)
    adapter = resumption_adapter()
    for key, value in (
        ("MODEL_THINKING", "adaptive"),
        ("MODEL_INSTRUCTION_MODE", "structured"),
        ("MAX_MODEL_CALLS", "15"),
    ):
        with monkeypatch.context() as changed:
            changed.setenv(key, value)
            with pytest.raises(ValueError):
                adapter.resume(freeze, root, manifest, source, tmp_path / "rejected")
    original = (source / "run.json").read_bytes()
    for key, value in (
        ("selection", {}),
        ("strategy_order", ["langgraph_explore"]),
        ("prompt_config", {}),
        ("freeze_sha256", "0" * 64),
        ("workers", 2),
        ("strict_strategy", False),
    ):
        metadata = json.loads(original)
        metadata[key] = value
        (source / "run.json").write_text(json.dumps(metadata))
        with pytest.raises(ValueError):
            adapter.resume(freeze, root, manifest, source, tmp_path / "rejected")
        (source / "run.json").write_bytes(original)
    assert not (tmp_path / "rejected").exists()
    assert len(requests) == 12


@pytest.mark.docker
def test_fresh_process_resumes_after_terminating_an_in_flight_model_request(
    tmp_path, new_flow_batch
):
    root, manifest, evidence, requests = new_flow_batch
    freeze = tmp_path / "freeze.json"
    reproducibility.freeze_batch(evidence, freeze, sample_ids=[1, 2])
    worker = tmp_path / "worker.py"
    worker.write_text("""
import asyncio, importlib.util, json, os, sys
from pathlib import Path
import httpx
from csv_agent.reproducibility import run_final
freeze, root, manifest, evidence, source, output, log, marker, mode = map(Path, sys.argv[1:])
analysis = json.loads((evidence / 'report.json').read_text())['runs']['langgraph_resolve']['records'][0]['analysis']
candidates, observations, reviews = analysis['candidates'], analysis['observations'], analysis['reviews']
actions = [
    ('inspect_data', {'action': 'inspect', 'code': observations[0]['code']}),
    ('compute_candidate', {'action': 'candidate', 'code': candidates[0]['code']}),
    ('recompute_candidate', {'action': 'recompute', 'candidate': 2, 'code': candidates[1]['code'], 'reason': candidates[1]['reason'], 'method': candidates[1]['method']}),
    ('review_candidate', {'action': 'review', **{key: value for key, value in reviews[0].items() if key != 'number'}}),
    ('inspect_data', {'action': 'inspect', 'code': observations[1]['code']}),
    ('select_candidate', {key: value for key, value in analysis['selection'].items() if key != 'number'}),
]
calls = 0
original_http = httpx.AsyncClient
async def provider(request):
    global calls
    payload = json.loads(request.content)
    calls += 1
    with log.open('a') as handle:
        handle.write(json.dumps({'mode': str(mode), 'request': calls}) + '\\n')
        handle.flush()
        os.fsync(handle.fileno())
    if str(mode) == 'run' and 'Sample 2.' in payload['messages'][-1]['content']:
        marker.write_text('in-flight request')
        await asyncio.Event().wait()
    name, arguments = actions[(calls - 1) % 6]
    response = {'content': [{'type': 'tool_use', 'id': 'call', 'name': name, 'input': arguments}]}
    if name != 'review_candidate':
        response['usage'] = {'input_tokens': 12, 'output_tokens': 20}
    return httpx.Response(200, json=response)
httpx.AsyncClient = lambda **kwargs: original_http(transport=httpx.MockTransport(provider), **kwargs)
if str(mode) == 'run':
    run_final(freeze, root, manifest, source)
else:
    spec = importlib.util.spec_from_file_location('resume', Path.cwd() / 'scripts/resume_frozen_experiment.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    details = adapter.resume(freeze, root, manifest, source, output)
    (output.parent / 'process-details.json').write_text(json.dumps(details))
""")
    source, output = tmp_path / "terminated", tmp_path / "resumed"
    log, marker = tmp_path / "requests.jsonl", tmp_path / "in-flight"
    command = [
        sys.executable,
        str(worker),
        str(freeze),
        str(root),
        str(manifest),
        str(evidence),
        str(source),
        str(output),
        str(log),
        str(marker),
    ]
    process = subprocess.Popen(
        [*command, "run"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    try:
        deadline = time.monotonic() + 30
        while (
            not marker.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert marker.exists(), (
            process.stderr.read()
            if process.poll() is not None
            else "Worker did not reach the in-flight request"
        )
        process.terminate()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    assert not (source / "receipt.json").exists()
    original = (source / "langgraph_resolve/analysis.jsonl").read_bytes()
    assert len(original.splitlines()) == 1
    resumed = subprocess.run(
        [*command, "resume"], capture_output=True, text=True, timeout=40, check=False
    )
    assert resumed.returncode == 0, resumed.stderr
    details = json.loads((tmp_path / "process-details.json").read_text())
    report = reproducibility.verify_final(
        Path(details["verification_freeze"]), root, output
    )
    assert details["reused_records"] == {"langgraph_resolve": 1}
    assert report.runs["langgraph_resolve"].summary.correct_questions == 2
    assert report.request_usage.unique_requests == 12
    assert report.request_usage.unknown_usage_requests["input_tokens"] == 2
    # One request was actually in flight and could not be persisted as a task.
    assert len(log.read_text().splitlines()) == 13
    assert (source / "langgraph_resolve/analysis.jsonl").read_bytes() == original
    assert len(requests) == 6  # The parent only ran the development fixture.


@pytest.mark.docker
def test_batch_rejects_inconsistent_reused_model_and_csv_before_starting(
    tmp_path, new_flow_batch
):
    root, manifest, evidence, requests = new_flow_batch
    original = evaluation.EvaluationRecord.model_validate_json(
        (evidence / "langgraph_resolve/analysis.jsonl").read_bytes()
    )
    report = evaluation.ComparisonReport.model_validate_json(
        (evidence / "report.json").read_bytes()
    )
    for mutation in ("model", "data"):
        record = original.model_copy(deep=True)
        if mutation == "model":
            record.analysis.attempts[0].request = record.analysis.attempts[
                0
            ].request.model_copy(update={"name": "another-model"})
        else:
            record.analysis.profile["source_sha256"] = "0" * 64
        output = tmp_path / f"rejected-{mutation}"
        with pytest.raises(ValueError):
            evaluation.run_comparison(
                root,
                manifest,
                split="development",
                strategies=["langgraph_resolve"],
                output_dir=output,
                config=report.config,
                limits=report.limits,
                model_version=report.model_version,
                strict_strategy=True,
                saved_records={"langgraph_resolve": [record]},
            )
        assert not output.exists()
    assert len(requests) == 6


@pytest.mark.docker
def test_frozen_basic_answer_must_match_its_saved_execution_attempt(
    tmp_path, monkeypatch, new_flow_batch
):
    from test_comparison import SUCCESS

    from csv_agent.config import ModelConfig

    root, manifest, _, _ = new_flow_batch
    monkeypatch.setenv("MODEL_INSTRUCTION_MODE", "structured")
    config = ModelConfig.from_env()

    def provider(request):
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": json.dumps({"code": SUCCESS})}]},
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: HTTP_CLIENT(transport=httpx.MockTransport(provider), **kwargs),
    )
    evidence = tmp_path / "basic"
    evaluation.run_comparison(
        root,
        manifest,
        split="development",
        strategies=["langgraph"],
        output_dir=evidence,
        config=config,
        limits=RunLimits.from_env(),
        model_version="HTTP-fixture-basic",
        strict_strategy=True,
    )
    freeze = tmp_path / "basic-freeze.json"
    reproducibility.freeze_batch(evidence, freeze, sample_ids=[1])
    output = tmp_path / "basic-final"
    reproducibility.run_final(freeze, root, manifest, output)
    paths = [
        p
        for p in output.rglob("*")
        if p.suffix in {".json", ".jsonl"} and p.name != "receipt.json"
    ]

    def mutate(value):
        if isinstance(value, dict):
            if "analysis" in value:
                value["analysis"]["attempts"][0]["execution"]["output"] = value[
                    "analysis"
                ]["attempts"][0]["execution"]["output"].replace("45", "15")
            else:
                for child in value.values():
                    mutate(child)
        elif isinstance(value, list):
            for child in value:
                mutate(child)

    for path in paths:
        if path.suffix == ".json":
            value = json.loads(path.read_text())
            mutate(value)
            path.write_text(json.dumps(value))
        else:
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            mutate(rows)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts_sha256"] = {
        name: sha256((output / name).read_bytes())
        for name in receipt["artifacts_sha256"]
    }
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        reproducibility.verify_final(freeze, root, output)


@pytest.mark.docker
def test_deadline_failure_retains_selection_evidence_and_remains_resumable(
    tmp_path, monkeypatch, new_flow_batch
):
    root, manifest, evidence, requests = new_flow_batch
    freeze = tmp_path / "freeze.json"
    reproducibility.freeze_batch(evidence, freeze, sample_ids=[1, 2])
    source = tmp_path / "deadline"
    clock = [time.monotonic()]
    calls = 0

    def interrupted(data, question, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("shutdown")

        def step(message):
            if message.startswith("Final submission"):
                clock[0] += kwargs["limits"].task_timeout_seconds + 1

        return analyze(data, question, on_step=step, **kwargs)

    with monkeypatch.context() as expired:
        expired.setattr(time, "monotonic", lambda: clock[0])
        expired.setattr(evaluation, "analyze", interrupted)
        with pytest.raises(RuntimeError, match="shutdown"):
            reproducibility.run_final(freeze, root, manifest, source)
    record = json.loads((source / "langgraph_resolve/analysis.jsonl").read_text())
    assert record["analysis"]["status"] == "timeout"
    assert record["analysis"]["answer"] is None
    assert record["analysis"]["selection"] is not None
    output = tmp_path / "resumed"
    details = resumption_adapter().resume(freeze, root, manifest, source, output)
    report = reproducibility.verify_final(
        Path(details["verification_freeze"]), root, output
    )
    run = report.runs["langgraph_resolve"]
    assert details["reused_records"] == {"langgraph_resolve": 1}
    assert run.summary.total == 2 and run.summary.correct_questions == 1
    assert run.summary.failures == {"timeout": 1}
    assert len(requests) == 18
