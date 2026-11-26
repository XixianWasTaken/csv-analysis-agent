import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_input import ColumnTotalsModel

from csv_agent.config import ModelConfig, RunLimits
from csv_agent.evaluation import run_evaluation
from csv_agent.model import ModelError, ModelReply, Usage

CONFIG = ModelConfig(endpoint="https://fixture.example/v1/messages", name="controlled")


def dataset(
    tmp_path: Path, count: int = 1, csv_data: bytes = b"sales\n10\n20\n15\n"
) -> tuple[Path, Path]:
    root = tmp_path / "data"
    (root / "da-dev-tables").mkdir(parents=True)
    questions = [
        {
            "id": i,
            "file_name": "sales.csv",
            "question": "What is total sales?",
            "constraints": "Sum all rows; keep decimal precision.",
            "format": "@total[value]",
        }
        for i in range(count)
    ]
    labels = [{"id": i, "common_answers": [["total", "45"]]} for i in range(count)]
    for name, rows in [("questions.jsonl", questions), ("labels.jsonl", labels)]:
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    table = root / "da-dev-tables/sales.csv"
    table.write_bytes(csv_data)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "dataset": {
            "repository": "https://huggingface.co/datasets/infiagent/DABench",
            "revision": "fixture",
            "questions_file": "questions.jsonl",
            "questions_sha256": digest(root / "questions.jsonl"),
            "labels_file": "labels.jsonl",
            "labels_sha256": digest(root / "labels.jsonl"),
            "scope": "controlled fixture; not a real model score",
        },
        "scorer": json.loads(Path("benchmarks/dabench/manifest.json").read_text())[
            "scorer"
        ],
        "tables": {
            "sales.csv": {"sha256": digest(table), "size_bytes": table.stat().st_size}
        },
        "samples": [{"id": i, "file_name": "sales.csv"} for i in range(count)],
        "splits": {"development": list(range(count)), "final": []},
    }
    lock = tmp_path / "manifest.json"
    lock.write_text(json.dumps(manifest))
    return root, lock


class AnswerModel:
    def __init__(self, answer: str):
        self.answer = answer

    def generate(self, prompt: str, *, timeout_seconds: float) -> ModelReply:
        code = f"""
import json
import pandas as pd
total = int(pd.read_csv('/input/data.csv')['sales'].sum())
assert total == 45
json.dump({{'answer': {self.answer!r}, 'table': [{{'total': total}}]}}, open('/output/result.json','w'))
"""
        return ModelReply(
            text=json.dumps({"code": code}),
            usage=Usage(input_tokens=12, output_tokens=20, cache_read_input_tokens=0),
            request=CONFIG,
        )


@pytest.mark.docker
def test_batch_computes_unnamed_data_with_the_same_csv_mapping(tmp_path):
    data = b",,sales\n10,100,1\n20,200,2\n15,300,3\n"
    root, manifest = dataset(tmp_path, csv_data=data)
    model = ColumnTotalsModel(answer_format="@total[{total}]")
    report = run_evaluation(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "report",
        config=CONFIG,
        limits=RunLimits(),
        model=model,
    )
    result = report.records[0].analysis
    assert result.status == "completed", result.error
    assert result.answer == "@total[45]"
    assert result.table == [{"Unnamed: 0": 45, "Unnamed: 1": 600, "sales": 6}]
    assert model.profile == result.profile
    assert result.profile["source_sha256"] == hashlib.sha256(data).hexdigest()
    assert (root / "da-dev-tables/sales.csv").read_bytes() == data
    assert report.summary.total == 1
    assert report.records[0].correctness == {"total": True}
    saved = json.loads((tmp_path / "report/analysis.jsonl").read_text())
    assert saved["analysis"]["profile"] == result.profile


@pytest.mark.docker
def test_batch_uses_official_numeric_tolerance_and_records_configuration(tmp_path):
    root, manifest = dataset(tmp_path)
    report = run_evaluation(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "report",
        config=CONFIG,
        limits=RunLimits(),
        model=AnswerModel("@total[45.0000005]"),
    )
    assert report.summary.question_accuracy == 1
    assert report.summary.subquestion_accuracy == 1
    assert report.summary.execution_success_rate == 1
    assert report.summary.total == 1
    assert report.summary.usage.input_tokens == 12
    assert report.config == CONFIG
    assert report.limits.max_executions == 3
    assert report.mode == "controlled"
    assert report.records[0].correctness == {"total": True}
    saved = json.loads((tmp_path / "report/report.json").read_text())
    assert saved["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert len((tmp_path / "report/analysis.jsonl").read_text().splitlines()) == 1


@pytest.mark.docker
def test_code_failure_and_invalid_submission_stay_in_fixed_denominator(tmp_path):
    root, manifest = dataset(tmp_path, count=3)

    class MixedModel:
        def __init__(self):
            self.index = 0

        def generate(self, prompt, *, timeout_seconds):
            self.index += 1
            if self.index == 2:
                return ModelReply(
                    text=json.dumps(
                        {"code": "raise RuntimeError('broken calculation')"}
                    ),
                    usage=None,
                )
            answer = "@total[45]" if self.index == 1 else "The total is 45"
            return AnswerModel(answer).generate(prompt, timeout_seconds=timeout_seconds)

    report = run_evaluation(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "report",
        config=CONFIG,
        limits=RunLimits(max_executions=1),
        model=MixedModel(),
    )
    assert report.summary.total == 3
    assert report.summary.question_accuracy == 0.3333
    assert report.summary.execution_successes == 1
    assert report.summary.execution_success_rate == 1 / 3
    assert report.summary.failures == {"failed": 2}
    assert report.records[2].analysis.status == "failed"
    assert report.records[2].analysis.answer is None
    assert report.records[1].correctness == {"total": False}
    assert report.summary.usage.input_tokens is None
    assert report.summary.known_usage.input_tokens == 24


@pytest.mark.docker
def test_standard_answers_are_absent_from_prompts_and_execution_mounts(tmp_path):
    root, manifest = dataset(tmp_path)
    labels = root / "labels.jsonl"
    labels.write_text(
        json.dumps({"id": 0, "common_answers": [["total", "label-only-sentinel"]]})
        + "\n"
    )
    lock = json.loads(manifest.read_text())
    lock["dataset"]["labels_sha256"] = hashlib.sha256(labels.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(lock))

    class IsolationModel:
        def generate(self, prompt, *, timeout_seconds):
            assert "label-only-sentinel" not in prompt
            assert "common_answers" not in prompt
            assert str(root) not in prompt
            assert "Sum all rows" in prompt and "@total[value]" in prompt
            reply = AnswerModel("@total[45]").generate(
                prompt, timeout_seconds=timeout_seconds
            )
            code = json.loads(reply.text)["code"]
            code = (
                "from pathlib import Path\n"
                "assert sorted(p.name for p in Path('/input').iterdir()) == ['analysis.py', 'data.csv']\n"
                f"assert not Path({str(labels)!r}).exists()\n" + code
            )
            return ModelReply(text=json.dumps({"code": code}), usage=None)

    report = run_evaluation(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "report",
        config=CONFIG,
        limits=RunLimits(),
        model=IsolationModel(),
    )
    assert report.records[0].analysis.status == "completed"
    assert report.summary.question_accuracy == 0
    assert "label-only-sentinel" not in (tmp_path / "report/analysis.jsonl").read_text()


@pytest.mark.docker
@pytest.mark.parametrize(
    "outcome, category",
    [
        ("timeout", "timeout"),
        ("exception", "failed"),
        ("empty", "response_format"),
        ("insufficient", "insufficient_input"),
        ("blank_result", "failed"),
        ("wrong", "incorrect_answer"),
    ],
)
def test_unsuccessful_samples_count_zero_without_real_api_calls(
    tmp_path, outcome, category
):
    root, manifest = dataset(tmp_path)

    class FailureModel:
        def generate(self, prompt, *, timeout_seconds):
            if outcome == "timeout":
                raise ModelError("Timed out", timed_out=True, diagnostic="timeout")
            if outcome == "exception":
                raise RuntimeError("untrusted-secret-exception")
            if outcome == "empty":
                return ModelReply(text="", usage=None)
            if outcome == "insufficient":
                return ModelReply(
                    text=json.dumps(
                        {
                            "status": "insufficient_input",
                            "reason": "Missing required column",
                        }
                    ),
                    usage=None,
                )
            return AnswerModel(
                " " if outcome == "blank_result" else "@total[45.000002]"
            ).generate(prompt, timeout_seconds=timeout_seconds)

    report = run_evaluation(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "report",
        config=CONFIG,
        limits=RunLimits(max_executions=1, max_format_repairs=0),
        model=FailureModel(),
    )
    assert report.summary.total == 1
    assert report.summary.question_accuracy == 0
    assert report.summary.subquestion_accuracy == 0
    assert report.summary.failures == {category: 1}
    assert report.summary.timeout_rate == (1 if outcome == "timeout" else 0)
    assert "untrusted-secret-exception" not in report.model_dump_json()
    if outcome in {"timeout", "exception", "empty", "insufficient"}:
        assert report.summary.known_usage.input_tokens is None


@pytest.mark.docker
def test_question_subquestion_and_proportional_metrics_have_explicit_denominators(
    tmp_path,
):
    root, manifest = dataset(tmp_path, count=2)
    labels = root / "labels.jsonl"
    labels.write_text(
        json.dumps({"id": 0, "common_answers": [["total", "45"]]})
        + "\n"
        + json.dumps({"id": 1, "common_answers": [["total", "45"], ["count", "3"]]})
        + "\n"
    )
    lock = json.loads(manifest.read_text())
    lock["dataset"]["labels_sha256"] = hashlib.sha256(labels.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(lock))
    report = run_evaluation(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "report",
        config=CONFIG,
        limits=RunLimits(),
        model=AnswerModel("@total[45]"),
    )
    assert report.summary.question_accuracy == 0.5
    assert report.summary.total_subquestions == 3
    assert report.summary.subquestion_accuracy == 0.6667
    assert report.summary.proportional_subquestion_accuracy == 0.75
    assert report.records[1].failure == "missing_answer_fields"
    assert report.records[1].correctness == {"total": True, "count": False}


def test_batch_cli_saves_failed_live_run_without_contacting_a_provider(tmp_path):
    root, manifest = dataset(tmp_path)
    environment = {
        **os.environ,
        "MODEL_ENDPOINT": CONFIG.endpoint,
        "MODEL_NAME": "missing-key-fixture",
        "MODEL_API_KEY_ENV": "ABSENT_EVALUATION_TEST_KEY",
    }
    environment.pop("ABSENT_EVALUATION_TEST_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "csv_agent.evaluation",
            "run",
            "--data",
            str(root),
            "--manifest",
            str(manifest),
            "--split",
            "development",
            "--output",
            str(tmp_path / "report"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["mode"] == "live"
    assert report["summary"]["total"] == 1
    assert report["summary"]["failures"] == {"authentication": 1}
    assert json.loads(result.stdout)["question_accuracy"] == 0


@pytest.mark.parametrize("damage", ["missing", "changed", "too_large"])
def test_bad_csv_keeps_its_sample_in_the_denominator(tmp_path, damage):
    root, manifest = dataset(tmp_path)
    csv = root / "da-dev-tables/sales.csv"
    if damage == "missing":
        csv.unlink()
    elif damage == "changed":
        csv.write_bytes(b"sales\n99\n20\n15\n")

    class UnreachableModel:
        def generate(self, prompt, *, timeout_seconds):
            raise AssertionError("Damaged data must not reach the model")

    report = run_evaluation(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "report",
        config=CONFIG,
        limits=RunLimits(input_max_bytes=5 if damage == "too_large" else 100),
        model=UnreachableModel(),
    )
    assert report.summary.total == 1
    assert report.summary.question_accuracy == 0
    assert report.summary.failures == {"data_error": 1}
    assert report.summary.model_calls == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "shared_csv",
        "missing_id",
        "unsafe_path",
        "scorer_drift",
        "questions_drift",
        "empty_split",
    ],
)
def test_invalid_or_changed_manifests_are_rejected_before_starting(tmp_path, mutation):
    root, manifest = dataset(tmp_path, count=2)
    lock = json.loads(manifest.read_text())
    split = "development"
    if mutation == "duplicate_id":
        lock["samples"][1]["id"] = 0
    elif mutation == "shared_csv":
        lock["splits"] = {"development": [0], "final": [1]}
    elif mutation == "missing_id":
        lock["splits"]["development"] = [0]
    elif mutation == "unsafe_path":
        lock["dataset"]["labels_file"] = "../labels.jsonl"
    elif mutation == "scorer_drift":
        lock["scorer"]["vendored_sha256"] = "0" * 64
    elif mutation == "questions_drift":
        (root / "questions.jsonl").write_text("{}\n")
    else:
        split = "final"
    manifest.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        run_evaluation(
            root,
            manifest,
            split=split,
            output_dir=tmp_path / "report",
            config=CONFIG,
            limits=RunLimits(),
            model=AnswerModel("@total[45]"),
        )
    assert not (tmp_path / "report").exists()


def test_data_preparation_checks_downloads_and_reuses_verified_files(
    tmp_path, monkeypatch
):
    import httpx

    from csv_agent.benchmark import prepare_dataset

    source, manifest = dataset(tmp_path)
    lock = json.loads(manifest.read_text())
    lock["dataset"]["revision"] = "a" * 40
    manifest.write_text(json.dumps(lock))
    original_client = httpx.Client
    seen = []

    def provider(request):
        name = request.url.path.split("/" + "a" * 40 + "/")[1]
        seen.append(name)
        return httpx.Response(200, content=(source / name).read_bytes())

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(provider), **kwargs
        ),
    )
    root = tmp_path / "download"
    prepare_dataset(root, manifest, split="development")
    prepare_dataset(root, manifest, split="development")
    assert sorted(seen) == [
        "da-dev-tables/sales.csv",
        "labels.jsonl",
        "questions.jsonl",
    ]
    (root / "da-dev-tables/sales.csv").write_text("tampered")
    prepare_dataset(root, manifest, split="development")
    assert (root / "da-dev-tables/sales.csv").read_bytes() == b"sales\n10\n20\n15\n"


@pytest.mark.docker
def test_duplicate_official_label_fields_use_last_value_and_one_scoring_field(tmp_path):
    root, manifest = dataset(tmp_path)
    labels = root / "labels.jsonl"
    labels.write_text(
        json.dumps({"id": 0, "common_answers": [["total", "44"], ["total", "45"]]})
        + "\n"
    )
    lock = json.loads(manifest.read_text())
    lock["dataset"]["labels_sha256"] = hashlib.sha256(labels.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(lock))
    report = run_evaluation(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "report",
        config=CONFIG,
        limits=RunLimits(),
        model=AnswerModel("@total[45]"),
    )
    assert report.summary.question_accuracy == 1
    assert report.summary.total_subquestions == 1
    assert report.records[0].correctness == {"total": True}
