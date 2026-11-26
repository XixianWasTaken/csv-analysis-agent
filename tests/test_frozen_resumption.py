"""Interrupted batch evidence at the evaluation boundary; HTTP fixture, no credits."""

import importlib.util
import json
import shutil
from pathlib import Path

import httpx
import pytest
from test_comparison import SUCCESS
from test_evaluation import CONFIG, dataset

from csv_agent import evaluation, reproducibility
from csv_agent.analysis import analyze
from csv_agent.benchmark import sha256
from csv_agent.config import RunLimits
from csv_agent.evaluation import run_comparison
from csv_agent.reproducibility import freeze_development, run_final, verify_final


@pytest.mark.docker
@pytest.mark.parametrize(
    ("memory_report_delta", "incomplete_write"),
    [
        pytest.param(0, False, id="same_runtime"),
        pytest.param(8192, False, id="after_reboot_8_kib"),
        pytest.param(-8192, False, id="after_reboot_minus_8_kib"),
        pytest.param(0, True, id="interrupted_write"),
    ],
)
def test_frozen_continuation_preserves_evidence_and_only_requests_missing_tasks(
    tmp_path, monkeypatch, memory_report_delta, incomplete_write
):
    spec = importlib.util.spec_from_file_location(
        "frozen_resumption", Path("scripts/resume_frozen_experiment.py")
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    root, manifest = dataset(tmp_path, count=3)
    lock = json.loads(manifest.read_text())
    final_table = root / "da-dev-tables/final_sales.csv"
    final_table.write_bytes((root / "da-dev-tables/sales.csv").read_bytes())
    questions = [
        json.loads(line) for line in (root / "questions.jsonl").read_text().splitlines()
    ]
    for question in questions[1:]:
        question["file_name"] = "final_sales.csv"
        question["question"] += f" Sample {question['id']}."
        lock["samples"][question["id"]]["file_name"] = "final_sales.csv"
    (root / "questions.jsonl").write_text(
        "".join(json.dumps(question) + "\n" for question in questions)
    )
    lock["dataset"]["questions_sha256"] = sha256(
        (root / "questions.jsonl").read_bytes()
    )
    lock["tables"]["final_sales.csv"] = lock["tables"]["sales.csv"]
    lock["splits"] = {"development": [0], "final": [1, 2]}
    manifest.write_text(json.dumps(lock))

    config = CONFIG.model_copy(
        update={"temperature": 1, "api_key_env": "RESUMPTION_TEST_KEY"}
    )
    monkeypatch.setenv("RESUMPTION_TEST_KEY", "resumption-fixture-secret")
    monkeypatch.setenv("MODEL_ENDPOINT", config.endpoint)
    monkeypatch.setenv("MODEL_NAME", config.name)
    monkeypatch.setenv("MODEL_API_KEY_ENV", config.api_key_env)
    monkeypatch.setenv("MODEL_TEMPERATURE", "1")
    for key in (
        "MODEL_PROTOCOL",
        "MODEL_TIMEOUT_SECONDS",
        "MODEL_MAX_TOKENS",
        "MODEL_DISABLE_THINKING",
        "MODEL_INSTRUCTION_MODE",
    ):
        monkeypatch.delenv(key, raising=False)
    original_http = httpx.AsyncClient
    requests = []

    def provider(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": json.dumps({"code": SUCCESS})}],
                "usage": {"input_tokens": 12, "output_tokens": 20},
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_http(
            transport=httpx.MockTransport(provider), **kwargs
        ),
    )
    run_comparison(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "development",
        config=config,
        limits=RunLimits.from_env(),
        model_version="HTTP-fixture-not-a-real-score",
    )
    freeze = tmp_path / "freeze.json"
    freeze_development(tmp_path / "development", freeze)
    source = tmp_path / "interrupted"
    started_tasks = 0

    def interrupt_after_one(data, question, **kwargs):
        nonlocal started_tasks
        started_tasks += 1
        if started_tasks > 1:
            raise RuntimeError("simulated process interruption")
        return analyze(data, question, **kwargs)

    with monkeypatch.context() as interrupted:
        interrupted.setattr(evaluation, "analyze", interrupt_after_one)
        with pytest.raises(RuntimeError, match="simulated process interruption"):
            run_final(freeze, root, manifest, source)
    assert len(requests) == 4
    assert not (source / "report.json").exists()
    original_files = {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "continued"
    checked = adapter.resume(
        freeze, root, manifest, source, output, check_only=True
    )
    assert checked["reused_records"] == {
        "single": 1,
        "tool_loop": 0,
        "langgraph": 0,
    }
    assert not output.exists()
    assert len(requests) == 4

    for change in ("out_of_order", "scored", "request_drift", "budget_overrun"):
        changed = tmp_path / change
        shutil.copytree(source, changed)
        evidence = changed / "single/analysis.jsonl"
        record = json.loads(evidence.read_text())
        if change == "out_of_order":
            record["id"] = 2
        elif change == "scored":
            record["correctness"] = {"total": True}
        elif change == "request_drift":
            record["analysis"]["attempts"][0]["request"]["temperature"] = 0.5
        else:
            record["analysis"]["attempts"] *= 2
        evidence.write_text(json.dumps(record) + "\n")
        rejected = tmp_path / f"{change}-output"
        with pytest.raises(ValueError):
            adapter.resume(freeze, root, manifest, changed, rejected)
        assert not rejected.exists()
        assert not rejected.with_name(rejected.name + "-resumption.json").exists()
        assert len(requests) == 4

    with monkeypatch.context() as changed_runtime:
        changed_runtime.setenv("MODEL_TEMPERATURE", "0.5")
        with pytest.raises(ValueError, match="original frozen runtime"):
            adapter.resume(freeze, root, manifest, source, output)
    assert len(requests) == 4
    if memory_report_delta == 8192:
        baseline = adapter.capture_runtime()
        docker = baseline.environment["docker"].split("|")
        hardware_upgrade = docker.copy()
        hardware_upgrade[-1] = str(int(docker[-1]) + 1073741824)
        cpu_change = docker.copy()
        cpu_change[-2] = str(int(docker[-2]) + 1)
        for changed in (
            {"docker": "|".join(hardware_upgrade)},
            {"docker": "|".join(cpu_change)},
            {"python": "changed-python"},
            {"packages": "changed-packages"},
        ):
            changed_runtime = baseline.model_copy(
                update={"environment": {**baseline.environment, **changed}}
            )
            with monkeypatch.context() as incompatible:
                incompatible.setattr(
                    adapter, "capture_runtime", lambda snapshot=changed_runtime: snapshot
                )
                with pytest.raises(ValueError, match="original frozen runtime"):
                    adapter.resume(
                        freeze, root, manifest, source, output, check_only=True
                    )
            assert len(requests) == 4
    partial = b'{"id":2,"analysis":'
    if incomplete_write:
        evidence = source / "single/analysis.jsonl"
        evidence.write_bytes(evidence.read_bytes() + partial)
        original_files[evidence.relative_to(source)] = evidence.read_bytes()
    if memory_report_delta:
        runtime = adapter.capture_runtime()
        docker = runtime.environment["docker"].split("|")
        docker[-1] = str(int(docker[-1]) + memory_report_delta)
        rebooted = runtime.model_copy(
            update={"environment": {**runtime.environment, "docker": "|".join(docker)}}
        )
        for boundary in (adapter, evaluation, reproducibility):
            monkeypatch.setattr(boundary, "capture_runtime", lambda: rebooted)
    if memory_report_delta or incomplete_write:
        checked = adapter.resume(
            freeze, root, manifest, source, output, check_only=True
        )
        assert checked["reused_records"]["single"] == 1
        assert len(requests) == 4
        assert not output.exists()
    if memory_report_delta == 8192:
        continued_tasks = 0

        def stop_continuation_after_one(data, question, **kwargs):
            nonlocal continued_tasks
            continued_tasks += 1
            if continued_tasks > 1:
                raise RuntimeError("second shutdown")
            return analyze(data, question, **kwargs)

        with monkeypatch.context() as shutdown_again:
            shutdown_again.setattr(adapter, "analyze", stop_continuation_after_one)
            with pytest.raises(RuntimeError, match="second shutdown"):
                adapter.resume(freeze, root, manifest, source, output)
        assert len(requests) == 5
        second_source = output
        output = tmp_path / "continued-again"
        details = adapter.resume(freeze, root, manifest, second_source, output)
        assert details["reused_records"]["single"] == 2
        assert "previous_resumption" in details
    else:
        details = adapter.resume(freeze, root, manifest, source, output)
    verification_freeze = Path(details.get("verification_freeze", freeze))
    report = verify_final(verification_freeze, root, output)
    if memory_report_delta:
        assert (
            details["runtime_differences"]["docker_memory_bytes"]["delta"]
            == memory_report_delta
        )
        assert verification_freeze != freeze
        with pytest.raises(ValueError, match="Freeze changed"):
            verify_final(freeze, root, output)
    if incomplete_write:
        assert details["discarded_partial_records"]["single"] == {
            "bytes": len(partial), "sha256": sha256(partial)
        }
    assert details["status"] == "verified"
    assert details["final_run_id"] == report.run_id
    assert len(requests) == 9  # Three development tasks, six final tasks, no repeats.
    assert all(
        run.summary.total == 2
        and run.summary.correct_questions == 2
        and run.summary.model_calls == 2
        and run.summary.execution_attempts == 2
        for run in report.runs.values()
    )
    assert original_files == {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    assert (output / "single/analysis.jsonl").read_bytes().splitlines()[0] == (
        original_files[Path("single/analysis.jsonl")].splitlines()[0]
    )
    assert "resumption-fixture-secret" not in json.dumps(details)
