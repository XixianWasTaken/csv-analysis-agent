import json
import subprocess
import sys

import pytest
from test_evaluation import dataset

from csv_agent.benchmark import sha256


def publication(tmp_path):
    data, manifest = dataset(tmp_path, count=2)
    locked = json.loads(manifest.read_text())
    locked["splits"] = {"development": [], "final": [0, 1]}
    manifest.write_text(json.dumps(locked))
    strategies = ["single", "tool_loop", "langgraph"]
    unknown_usage = {
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_input_tokens": None,
    }
    summary = {
        "total": 2,
        "correct_questions": 1,
        "total_subquestions": 2,
        "correct_subquestions": 1,
        "question_accuracy": 0.5,
        "subquestion_accuracy": 0.5,
        "proportional_subquestion_accuracy": 0.5,
        "execution_successes": 1,
        "execution_success_rate": 0.5,
        "failures": {"failed": 1},
        "timeout_rate": 0,
        "failure_rate": 0.5,
        "task_seconds": 0,
        "model_seconds": 0,
        "model_calls": 0,
        "execution_attempts": 0,
        "usage": unknown_usage,
        "known_usage": unknown_usage,
        "usage_reporting_tasks": {field: 0 for field in unknown_usage},
    }
    output = tmp_path / "publication"
    output.mkdir()
    metadata = {
        "manifest_sha256": sha256(manifest.read_bytes()),
        "dataset": locked["dataset"],
        "scorer": locked["scorer"],
        "splits": locked["splits"],
        "strategy_order": strategies,
        "final_summaries": {s: summary for s in strategies},
    }
    (output / "metadata.json").write_text(json.dumps(metadata))
    rows = []
    for number in range(2):
        result = {
            "status": "completed" if number == 0 else "failed",
            "answer": "@total[45]" if number == 0 else None,
            "correctness": {"total": number == 0},
            "failure": None if number == 0 else "failed",
            "seconds": 0,
            "model_calls": 0,
            "execution_attempts": 0,
            "usage": None,
            "attempts": [],
        }
        rows.append(
            {
                "id": number,
                "file_name": "sales.csv",
                "results": {s: result.copy() for s in strategies},
            }
        )
    paired = output / "paired-results.jsonl"
    paired.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return data, manifest, output, rows


def audit(data, manifest, output):
    return subprocess.run(
        [
            sys.executable,
            "scripts/verify_published_experiment.py",
            "--data",
            str(data),
            "--manifest",
            str(manifest),
            "--report",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_reader_can_rescore_published_predictions_without_models_or_docker(tmp_path):
    data, manifest, output, _ = publication(tmp_path)
    result = audit(data, manifest, output)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["scores"]["single"]["question_accuracy"] == 0.5


@pytest.mark.parametrize("tampering", ["prediction", "missing_id", "summary"])
def test_public_audit_rejects_changed_predictions_denominators_and_totals(
    tmp_path, tampering
):
    data, manifest, output, rows = publication(tmp_path)
    if tampering == "prediction":
        rows[0]["results"]["single"]["answer"] = "@total[99]"
    elif tampering == "missing_id":
        rows.pop()
    else:
        metadata_path = output / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["final_summaries"]["single"]["total"] = 1
        metadata_path.write_text(json.dumps(metadata))
    (output / "paired-results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    result = audit(data, manifest, output)
    assert result.returncode == 2
    assert "Public audit failed" in result.stderr
