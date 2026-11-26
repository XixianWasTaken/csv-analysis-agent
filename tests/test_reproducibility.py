"""Freeze workflow at the spec's batch evaluation boundary; no real API credits."""

import json

import httpx
import pytest
from test_comparison import SUCCESS, SequenceModel
from test_evaluation import CONFIG, dataset

from csv_agent.config import RunLimits
from csv_agent.evaluation import run_comparison


@pytest.mark.docker
def test_controlled_development_cannot_be_frozen_as_real_model_evidence(tmp_path):
    import json

    from csv_agent.reproducibility import freeze_development

    root, manifest = dataset(tmp_path)
    run_comparison(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "development",
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(),
        model_version="controlled-fixture",
        models={
            s: SequenceModel([json.dumps({"code": SUCCESS})])
            for s in ("single", "tool_loop", "langgraph")
        },
    )
    with pytest.raises(ValueError, match="live development"):
        freeze_development(tmp_path / "development", tmp_path / "freeze.json")
    assert not (tmp_path / "freeze.json").exists()


@pytest.mark.docker
@pytest.mark.parametrize("legacy_summary", [False, True])
def test_frozen_final_rejects_configuration_drift_before_calling_model(
    tmp_path, monkeypatch, legacy_summary
):
    from csv_agent.reproducibility import freeze_development, run_final, verify_final

    root, manifest = dataset(tmp_path, count=2)
    lock = json.loads(manifest.read_text())
    final_table = root / "da-dev-tables/final_sales.csv"
    final_table.write_bytes((root / "da-dev-tables/sales.csv").read_bytes())
    questions = [
        json.loads(line) for line in (root / "questions.jsonl").read_text().splitlines()
    ]
    questions[1]["file_name"] = "final_sales.csv"
    (root / "questions.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in questions)
    )
    from csv_agent.benchmark import sha256

    lock["dataset"]["questions_sha256"] = sha256(
        (root / "questions.jsonl").read_bytes()
    )
    lock["tables"]["final_sales.csv"] = lock["tables"]["sales.csv"]
    lock["samples"][1]["file_name"] = "final_sales.csv"
    lock["splits"] = {"development": [0], "final": [1]}
    manifest.write_text(json.dumps(lock))
    config = CONFIG.model_copy(
        update={"temperature": 1, "api_key_env": "FREEZE_TEST_KEY"}
    )
    monkeypatch.setenv("FREEZE_TEST_KEY", "test-secret")
    monkeypatch.setenv("MODEL_ENDPOINT", config.endpoint)
    monkeypatch.setenv("MODEL_NAME", config.name)
    monkeypatch.setenv("MODEL_API_KEY_ENV", config.api_key_env)
    monkeypatch.setenv("MODEL_TEMPERATURE", "1")
    for key in ("MODEL_PROTOCOL", "MODEL_DISABLE_THINKING", "MODEL_INSTRUCTION_MODE"):
        monkeypatch.delenv(key, raising=False)
    original = httpx.AsyncClient
    calls = []

    def provider(request):
        calls.append(request)
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
        lambda **kw: original(transport=httpx.MockTransport(provider), **kw),
    )
    run_comparison(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "development",
        config=config,
        limits=RunLimits(),
        model_version="HTTP-fixture-not-a-real-score",
    )
    import platform

    with monkeypatch.context() as changed_environment:
        changed_environment.setattr(platform, "python_version", lambda: "0.0.0")
        with pytest.raises(ValueError, match="changed"):
            freeze_development(
                tmp_path / "development", tmp_path / "drifted-freeze.json"
            )
    assert not (tmp_path / "drifted-freeze.json").exists()
    freeze_development(tmp_path / "development", tmp_path / "freeze.json")
    assert "test-secret" not in (tmp_path / "freeze.json").read_text()
    monkeypatch.setenv("MODEL_TEMPERATURE", "0.5")
    with pytest.raises(ValueError, match="changed"):
        run_final(tmp_path / "freeze.json", root, manifest, tmp_path / "final")
    assert len(calls) == 3
    assert not (tmp_path / "final").exists()
    monkeypatch.setenv("MODEL_TEMPERATURE", "1")
    report = run_final(tmp_path / "freeze.json", root, manifest, tmp_path / "final")
    assert (
        report.mode == "live"
    )  # Mocked HTTP boundary, explicitly named fixture version.
    assert all(
        run.summary.total == 1 and run.summary.correct_questions == 1
        for run in report.runs.values()
    )
    if legacy_summary:
        # Model a signed report written before request-level accounting existed.
        report_path = tmp_path / "final/report.json"
        historical = json.loads(report_path.read_text())
        historical.pop("request_usage")
        for run in historical["runs"].values():
            run["summary"].pop("request_usage")
        report_path.write_text(json.dumps(historical))
        receipt_path = tmp_path / "final/receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["artifacts_sha256"]["report.json"] = sha256(report_path.read_bytes())
        receipt_path.write_text(json.dumps(receipt))
        from csv_agent.evaluation import ComparisonReport

        report = ComparisonReport.model_validate(historical)
    assert verify_final(tmp_path / "freeze.json", root, tmp_path / "final") == report
    if not legacy_summary:
        report_path = tmp_path / "final/report.json"
        receipt_path = tmp_path / "final/receipt.json"
        original_report, original_receipt = (
            report_path.read_bytes(),
            receipt_path.read_bytes(),
        )
        for location in ("aggregate", "strategy"):
            changed = json.loads(original_report)
            if location == "aggregate":
                changed["request_usage"]["unique_requests"] += 1
                reason = "Request usage"
            else:
                changed["runs"]["single"]["summary"]["request_usage"]["known_usage"][
                    "input_tokens"
                ] += 1
                reason = "Official rescoring"
            report_path.write_text(json.dumps(changed))
            receipt = json.loads(original_receipt)
            receipt["artifacts_sha256"]["report.json"] = sha256(
                report_path.read_bytes()
            )
            receipt_path.write_text(json.dumps(receipt))
            with pytest.raises(ValueError, match=reason):
                verify_final(tmp_path / "freeze.json", root, tmp_path / "final")
        report_path.write_bytes(original_report)
        receipt_path.write_bytes(original_receipt)
    assert "test-secret" not in (tmp_path / "final/receipt.json").read_text()
    (tmp_path / "final/langgraph/analysis.jsonl").write_text("")
    with pytest.raises(ValueError, match="changed"):
        verify_final(tmp_path / "freeze.json", root, tmp_path / "final")
