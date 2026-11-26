"""Ticket 07 at the agreed analysis and batch evaluation boundaries."""

import hashlib
import json
import os
import subprocess
import sys
import time

import pytest
from test_evaluation import CONFIG, dataset

from csv_agent.analysis import analyze
from csv_agent.config import RunLimits
from csv_agent.model import ModelError, ModelReply, Usage

CSV = b"sales\n10\n20\n15\n"
SUCCESS = """
import json, pandas as pd
total = int(pd.read_csv('/input/data.csv')['sales'].sum())
json.dump({'answer': f'@total[{total}]'}, open('/output/result.json', 'w'))
"""


class SequenceModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def generate(self, prompt, *, timeout_seconds):
        self.prompts.append(json.loads(prompt))
        return ModelReply(next(self.replies), Usage(input_tokens=10, output_tokens=5))


@pytest.mark.docker
def test_single_generation_stops_after_one_failed_execution():
    model = SequenceModel(
        [
            json.dumps({"code": "raise ValueError('repair-me')"}),
            json.dumps({"code": SUCCESS}),
        ]
    )
    result = analyze(CSV, "total sales", model=model, strategy="single")
    assert result.status == "failed"
    assert result.answer is None
    assert result.execution_attempts == 1
    assert len(result.attempts) == 1
    assert "repair-me" in result.error


@pytest.mark.docker
@pytest.mark.parametrize("strategy", ["tool_loop", "langgraph", "langgraph_no_repair"])
def test_feedback_recovery_and_ablation_share_the_analysis_contract(strategy):
    model = SequenceModel(
        [
            json.dumps({"code": "raise ValueError('first-feedback')"}),
            json.dumps({"code": "raise ValueError('second-feedback')"}),
            json.dumps({"code": SUCCESS}),
        ]
    )
    result = analyze(CSV, "total sales", model=model, strategy=strategy)
    if strategy == "langgraph_no_repair":
        assert result.status == "failed"
        assert result.execution_attempts == 1
        assert len(result.attempts) == 1
        assert "disabled" in result.stop_reason
    else:
        assert result.status == "completed", result.error
        assert result.answer == "@total[45]"
        assert result.execution_attempts == 3
        assert result.usage.input_tokens == 30
        assert "second-feedback" in json.dumps(model.prompts[-1])
        if strategy == "tool_loop":
            assert "first-feedback" in json.dumps(model.prompts[-1])


@pytest.mark.docker
@pytest.mark.parametrize("strategy", ["tool_loop", "langgraph"])
def test_model_call_budget_stops_repairs_before_execution_budget(strategy):
    model = SequenceModel(
        [
            json.dumps({"code": "raise ValueError('broken')"}),
            json.dumps({"code": SUCCESS}),
        ]
    )
    result = analyze(
        CSV,
        "sum",
        model=model,
        strategy=strategy,
        limits=RunLimits(max_model_calls=1, max_executions=3),
    )
    assert result.status == "failed"
    assert result.execution_attempts == 1
    assert len(result.attempts) == 1
    assert "Model call budget" in result.stop_reason


@pytest.mark.docker
def test_paired_comparison_scores_all_strategies_and_discloses_compute(tmp_path):
    from csv_agent.evaluation import run_comparison

    root, manifest = dataset(tmp_path)
    strategies = ["single", "tool_loop", "langgraph", "langgraph_no_repair"]
    models = {
        name: SequenceModel(
            [
                json.dumps({"code": "raise ValueError('repair-me')"}),
                json.dumps({"code": SUCCESS}),
            ]
        )
        for name in strategies
    }
    report = run_comparison(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "comparison",
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(),
        model_version="controlled-fixture",
        models=models,
        include_ablation=True,
    )
    assert report.mode == "controlled"
    assert len(report.paired_records) == 1
    pair = report.paired_records[0]
    assert pair.id == 0
    assert set(pair.results) == set(strategies)
    assert pair.results["single"].correctness == {"total": False}
    assert pair.results["tool_loop"].correctness == {"total": True}
    assert pair.results["langgraph"].correctness == {"total": True}
    assert pair.results["langgraph_no_repair"].correctness == {"total": False}
    assert report.runs["single"].summary.question_accuracy == 0
    assert report.runs["langgraph"].summary.question_accuracy == 1
    assert report.runs["single"].summary.model_calls == 1
    assert report.runs["langgraph"].summary.model_calls == 2
    assert report.budgets["single"].max_model_calls == 1
    assert report.budgets["single"].limits.max_executions == 1
    assert report.budgets["tool_loop"] == report.budgets["langgraph"]
    assert report.budgets["langgraph_no_repair"] == report.budgets["langgraph"]
    delta = report.deltas["langgraph_vs_single"]
    assert delta.question_accuracy == 1
    assert delta.model_calls == 1
    assert delta.input_tokens == 10
    assert report.config.temperature == 1
    assert report.limits.image.startswith("sha256:")
    saved = json.loads((tmp_path / "comparison/report.json").read_text())
    assert saved["model_version"] == "controlled-fixture"
    assert len((tmp_path / "comparison/paired.jsonl").read_text().splitlines()) == 1
    assert "repair-me" in pair.results["langgraph"].analysis.attempts[1].prompt


@pytest.mark.docker
def test_comparison_keeps_failures_and_can_report_regression(tmp_path):
    from csv_agent.evaluation import run_comparison

    root, manifest = dataset(tmp_path, count=4)
    questions_path = root / "questions.jsonl"
    questions = [json.loads(line) for line in questions_path.read_text().splitlines()]
    for index, question in enumerate(questions):
        question["question"] = f"case-{index}: total sales?"
    questions_path.write_text("".join(json.dumps(q) + "\n" for q in questions))
    lock = json.loads(manifest.read_text())
    lock["dataset"]["questions_sha256"] = hashlib.sha256(
        questions_path.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(lock))

    class Cases:
        def __init__(self, regress=False):
            self.regress = regress

        def generate(self, prompt, *, timeout_seconds):
            question = json.loads(prompt)["question"]
            if question.startswith("case-0"):
                code = SUCCESS.replace("f'@total[{total}]'", "'@total[45.0000005]'")
                if self.regress:
                    code = SUCCESS.replace("f'@total[{total}]'", "'@total[44]'")
                return ModelReply(json.dumps({"code": code}), None)
            if question.startswith("case-1"):
                raise ModelError(
                    "Model request timed out", timed_out=True, diagnostic="timeout"
                )
            if question.startswith("case-2"):
                return ModelReply("", None)
            return ModelReply(
                json.dumps(
                    {"code": SUCCESS.replace("f'@total[{total}]'", "'unparseable'")}
                ),
                None,
            )

    report = run_comparison(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "comparison",
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(),
        model_version="fixture",
        models={
            "single": Cases(),
            "tool_loop": Cases(),
            "langgraph": Cases(regress=True),
        },
    )
    assert [p.id for p in report.paired_records] == [0, 1, 2, 3]
    for strategy, run in report.runs.items():
        assert run.summary.total == 4
        assert run.summary.total_subquestions == 4
        assert run.summary.timeout_rate == 0.25
        assert run.summary.failure_rate == 0.75
        assert run.summary.execution_successes == 1
        assert run.records[1].correctness == {"total": False}
        assert run.records[2].correctness == {"total": False}
        assert run.records[3].failure == "failed"
        assert run.records[3].analysis.status == "failed"
        assert "missing required fields total" in run.records[3].analysis.error
        assert run.summary.usage.input_tokens is None
        assert (
            len(
                (tmp_path / f"comparison/{strategy}/results.jsonl")
                .read_text()
                .splitlines()
            )
            == 4
        )
    delta = report.deltas["langgraph_vs_tool_loop"]
    assert delta.question_accuracy == -0.25
    assert delta.wins == 0
    assert delta.losses == 1
    assert delta.ties == 3
    assert delta.input_tokens is None
    assert "controlled" in (tmp_path / "comparison/summary.md").read_text()


@pytest.mark.docker
def test_compare_cli_saves_all_failed_strategies_without_model_credentials(tmp_path):
    root, manifest = dataset(tmp_path)
    environment = {
        **os.environ,
        "MODEL_ENDPOINT": CONFIG.endpoint,
        "MODEL_NAME": "fixture",
        "MODEL_TEMPERATURE": "1",
        "MODEL_API_KEY_ENV": "ABSENT_COMPARISON_TEST_KEY",
    }
    environment.pop("ABSENT_COMPARISON_TEST_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "csv_agent.evaluation",
            "compare",
            "--data",
            str(root),
            "--manifest",
            str(manifest),
            "--split",
            "development",
            "--output",
            str(tmp_path / "comparison"),
            "--model-version",
            "fixture",
            "--include-ablation",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "comparison/report.json").read_text())
    assert report["mode"] == "live"
    assert len(report["runs"]) == 4
    for run in report["runs"].values():
        assert run["summary"]["total"] == 1
        assert run["summary"]["question_accuracy"] == 0
        assert run["summary"]["failures"] == {"authentication": 1}


@pytest.mark.parametrize(
    "strategy", ["single", "tool_loop", "langgraph", "langgraph_no_repair"]
)
def test_all_strategies_stop_late_model_results_at_the_same_deadline(strategy):
    class LateModel:
        def generate(self, prompt, *, timeout_seconds):
            time.sleep(timeout_seconds + 0.02)
            return ModelReply(json.dumps({"code": SUCCESS}), None)

    result = analyze(
        CSV,
        "sum",
        model=LateModel(),
        strategy=strategy,
        limits=RunLimits(task_timeout_seconds=0.05),
    )
    assert result.status == "timeout"
    assert result.answer is None
    assert result.execution_attempts == 0
    assert len(result.attempts) <= 1


@pytest.mark.parametrize("strategy", ["single", "langgraph_no_repair"])
def test_single_and_ablation_do_not_repair_invalid_instructions(strategy):
    result = analyze(
        CSV,
        "sum",
        model=SequenceModel(["invalid", json.dumps({"code": SUCCESS})]),
        strategy=strategy,
    )
    assert result.status == "failed"
    assert len(result.attempts) == 1
    assert result.execution_attempts == 0


@pytest.mark.docker
def test_labels_are_not_read_until_all_strategy_batches_finish(tmp_path):
    from csv_agent.evaluation import run_comparison

    root, manifest = dataset(tmp_path)
    labels = root / "labels.jsonl"
    label_bytes = labels.read_bytes()
    labels.unlink()
    calls = []

    class WithoutLabels:
        def __init__(self, strategy):
            self.strategy = strategy

        def generate(self, prompt, *, timeout_seconds):
            calls.append(self.strategy)
            assert "common_answers" not in prompt
            if len(calls) == 3:
                labels.write_bytes(label_bytes)
            return ModelReply(
                '{"status":"insufficient_input","reason":"controlled stop"}', None
            )

    report = run_comparison(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "comparison",
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(),
        model_version="fixture",
        models={s: WithoutLabels(s) for s in ["single", "tool_loop", "langgraph"]},
    )
    assert calls == ["single", "tool_loop", "langgraph"]
    assert all(run.summary.question_accuracy == 0 for run in report.runs.values())


@pytest.mark.docker
@pytest.mark.parametrize("damage", ["missing", "changed", "too_large"])
def test_bad_data_is_paired_as_failure_for_every_strategy(tmp_path, damage):
    from csv_agent.evaluation import run_comparison

    root, manifest = dataset(tmp_path)
    table = root / "da-dev-tables/sales.csv"
    if damage == "missing":
        table.unlink()
    elif damage == "changed":
        table.write_bytes(b"sales\n99\n20\n15\n")

    class UnreachableModel:
        def generate(self, prompt, *, timeout_seconds):
            raise AssertionError("Damaged inputs must not reach the model")

    report = run_comparison(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "comparison",
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(input_max_bytes=5 if damage == "too_large" else 100),
        model_version="fixture",
        models={s: UnreachableModel() for s in ["single", "tool_loop", "langgraph"]},
    )
    assert len(report.paired_records) == 1
    for run in report.runs.values():
        assert run.summary.total == 1
        assert run.summary.failures == {"data_error": 1}
        assert run.summary.model_calls == 0
        assert run.summary.question_accuracy == 0


@pytest.mark.parametrize("invalid", ["models", "temperature", "version", "manifest"])
def test_invalid_comparison_setup_does_not_create_output(tmp_path, invalid):
    from csv_agent.evaluation import run_comparison

    root, manifest = dataset(tmp_path)
    config = CONFIG.model_copy(update={"temperature": 1})
    version = "fixture"
    models = {s: SequenceModel([]) for s in ["single", "tool_loop", "langgraph"]}
    if invalid == "models":
        models.pop("single")
    elif invalid == "temperature":
        config = CONFIG
    elif invalid == "version":
        version = " "
    else:
        manifest.write_text("{}")
    with pytest.raises(ValueError):
        run_comparison(
            root,
            manifest,
            split="development",
            output_dir=tmp_path / "comparison",
            config=config,
            limits=RunLimits(),
            model_version=version,
            models=models,
        )
    assert not (tmp_path / "comparison").exists()
