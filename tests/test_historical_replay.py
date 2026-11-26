import hashlib
import json
import os
import shutil
import subprocess
import sys

import pytest
from test_evaluation import CONFIG, AnswerModel, dataset

from csv_agent.config import RunLimits
from csv_agent.evaluation import run_comparison
from csv_agent.model import ModelReply


@pytest.mark.docker
def test_replay_uses_first_code_without_model_requests_and_scores_offline(tmp_path):
    from csv_agent.replay import run_replay, verify_replay

    root, manifest = dataset(tmp_path)
    history = tmp_path / "history"
    run_comparison(
        root,
        manifest,
        split="development",
        output_dir=history,
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(),
        model_version="fixture",
        models={"single": AnswerModel("@total[45]")},
        strategies=["single"],
    )
    original = {
        p.relative_to(history): p.read_bytes()
        for p in history.rglob("*")
        if p.is_file()
    }
    output = tmp_path / "replay"
    report = run_replay(
        root,
        manifest,
        source_dir=history,
        output_dir=output,
        split="development",
        sample_ids=[0],
        strategies=["single"],
        limits=RunLimits(),
    )
    assert report.evidence_kind == "historical_code_replay"
    assert report.runs["single"].summary.correct_questions == 1
    assert report.runs["single"].summary.model_calls == 0
    assert report.runs["single"].summary.execution_attempts == 1
    assert report.runs["single"].records[0].analysis.table == [{"total": 45}]
    assert {
        p.relative_to(history): p.read_bytes()
        for p in history.rglob("*")
        if p.is_file()
    } == original
    assert verify_replay(root, output) == report

    altered = tmp_path / "altered-artifacts"
    shutil.copytree(output, altered)
    forged = json.loads((altered / "report.json").read_text())
    analysis = forged["runs"]["single"]["records"][0]["analysis"]
    envelope = json.loads(analysis["execution"]["output"])
    envelope["result"]["charts"] = [
        {"path": "absent.png", "caption": "Missing artifact"}
    ]
    analysis["execution"]["output"] = json.dumps(envelope)
    analysis["attempts"][0]["execution"]["output"] = json.dumps(envelope)
    (altered / "report.json").write_text(json.dumps(forged))
    for name in ("analysis.jsonl", "results.jsonl"):
        record_path = altered / "single" / name
        record = json.loads(record_path.read_text())
        record["analysis"] = analysis
        record_path.write_text(json.dumps(record) + "\n")
    receipt = json.loads((altered / "receipt.json").read_text())
    for name in ("report.json", "single/analysis.jsonl", "single/results.jsonl"):
        receipt["artifacts_sha256"][name] = hashlib.sha256(
            (altered / name).read_bytes()
        ).hexdigest()
    (altered / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_replay(root, altered)

    (root / "da-dev-tables/sales.csv").write_bytes(b"sales\n99\n")
    damaged = run_replay(
        root,
        manifest,
        source_dir=history,
        output_dir=tmp_path / "damaged",
        split="development",
        sample_ids=[0],
        strategies=["single"],
        limits=RunLimits(),
    )
    assert damaged.runs["single"].summary.total == 1
    assert damaged.runs["single"].summary.correct_questions == 0
    assert damaged.runs["single"].summary.execution_attempts == 0
    assert damaged.runs["single"].origins[0].disposition == "input_failed"
    assert verify_replay(root, tmp_path / "damaged") == damaged


@pytest.mark.docker
def test_cli_replays_science_and_normalized_tables_without_model_configuration(
    tmp_path, monkeypatch
):
    from csv_agent.replay import verify_replay

    root, manifest = dataset(tmp_path)
    labels = root / "labels.jsonl"
    code = f"""
import json, os, socket
from pathlib import Path
import pandas as pd
from scipy.stats import norm
assert not Path({str(labels)!r}).exists()
assert 'REPLAY_HOST_SECRET' not in os.environ
assert sorted(p.name for p in Path('/input').iterdir()) == ['analysis.py', 'data.csv']
try:
    socket.create_connection(('1.1.1.1', 80), timeout=0.2)
except OSError:
    pass
else:
    raise AssertionError('network reachable')
total = int(pd.read_csv('/input/data.csv')['sales'].sum())
json.dump({{'answer': f'@total[{{total}}]', 'table': [{{'p': float(norm.cdf(0)), 'missing': float('nan'), 'values': [1, None]}}]}}, open('/output/result.json', 'w'))
"""

    class ScientificModel:
        def generate(self, prompt, *, timeout_seconds):
            return ModelReply(json.dumps({"code": code}), None)

    monkeypatch.setenv("REPLAY_HOST_SECRET", "host-only")
    history = tmp_path / "history"
    run_comparison(
        root,
        manifest,
        split="development",
        output_dir=history,
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(),
        model_version="fixture",
        models={"single": ScientificModel()},
        strategies=["single"],
    )
    output = tmp_path / "replay"
    ids = tmp_path / "ids.json"
    ids.write_text("[0]")
    environment = dict(
        os.environ,
        MODEL_ENDPOINT="invalid-model-config",
        MODEL_API_KEY="",
        SPARK_API_KEY="",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "csv_agent.evaluation",
            "replay",
            "--data",
            str(root),
            "--manifest",
            str(manifest),
            "--split",
            "development",
            "--source",
            str(history),
            "--output",
            str(output),
            "--sample-list",
            str(ids),
            "--strategies",
            "single",
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = verify_replay(root, output)
    assert report.runs["single"].records[0].analysis.table == [
        {"p": 0.5, "missing": None, "values": "[1,null]"}
    ]
    assert report.runs["single"].summary.model_calls == 0
    assert report.runs["single"].summary.correct_questions == 1

    # Hashes alone are not the scoring authority: the offline verifier must
    # still reject an edited aggregate even if a new receipt was supplied.
    saved = json.loads((output / "report.json").read_text())
    saved["runs"]["single"]["summary"]["correct_questions"] = 3
    (output / "report.json").write_text(json.dumps(saved))
    receipt = json.loads((output / "receipt.json").read_text())
    receipt["artifacts_sha256"]["report.json"] = hashlib.sha256(
        (output / "report.json").read_bytes()
    ).hexdigest()
    (output / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify_replay(root, output)
    saved = json.loads((output / "report.json").read_text())
    assert saved["runs"]["single"]["summary"]["request_usage"]["unique_requests"] == 0


@pytest.mark.docker
def test_input_recovery_without_code_and_later_repairs_do_not_inflate_replay(tmp_path):
    from csv_agent.replay import run_replay, verify_replay

    root, manifest = dataset(tmp_path, count=3)
    large = root / "da-dev-tables/unnamed.csv"
    large.write_bytes(b",sales\n" + b"0,1\n" * 20)
    questions_path = root / "questions.jsonl"
    questions = [json.loads(line) for line in questions_path.read_text().splitlines()]
    questions[0]["file_name"] = "unnamed.csv"
    questions_path.write_text("".join(json.dumps(q) + "\n" for q in questions))
    lock = json.loads(manifest.read_text())
    lock["dataset"]["questions_sha256"] = hashlib.sha256(
        questions_path.read_bytes()
    ).hexdigest()
    lock["samples"][0]["file_name"] = "unnamed.csv"
    lock["tables"]["unnamed.csv"] = {
        "sha256": hashlib.sha256(large.read_bytes()).hexdigest(),
        "size_bytes": large.stat().st_size,
    }
    manifest.write_text(json.dumps(lock))

    class RepairModel:
        calls = 0

        def generate(self, prompt, *, timeout_seconds):
            self.calls += 1
            if self.calls == 1:
                return ModelReply(
                    json.dumps({"code": "raise ValueError('original failure')"}), None
                )
            return AnswerModel("@total[45]").generate(
                prompt, timeout_seconds=timeout_seconds
            )

    history = tmp_path / "history"
    original = run_comparison(
        root,
        manifest,
        split="development",
        output_dir=history,
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(input_max_bytes=40),
        model_version="fixture",
        models={"langgraph": RepairModel()},
        strategies=["langgraph"],
    )
    assert original.runs["langgraph"].summary.correct_questions == 2
    output = tmp_path / "replay"
    report = run_replay(
        root,
        manifest,
        source_dir=history,
        output_dir=output,
        split="development",
        sample_ids=[0, 1, 2],
        strategies=["langgraph"],
        limits=RunLimits(),
    )
    run = report.runs["langgraph"]
    assert run.summary.total == 3
    assert run.summary.correct_questions == 1
    assert run.first_code_summary.correct_questions == 1
    assert run.historical_summary.correct_questions == 2
    assert run.summary.model_calls == 0
    assert run.summary.execution_attempts == 2
    assert (run.wins, run.losses, run.ties) == (0, 1, 2)
    assert run.origins[0].disposition == "not_replayed_no_code"
    assert run.records[0].analysis.profile["columns"] == ["Unnamed: 0", "sales"]
    assert run.records[0].analysis.answer is None
    assert run.records[1].analysis.code == "raise ValueError('original failure')"
    assert run.records[1].analysis.status == "failed"
    assert verify_replay(root, output) == report
