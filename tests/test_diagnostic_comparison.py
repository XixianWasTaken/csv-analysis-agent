"""Selected diagnostic batches at the established comparison boundary."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from threading import Barrier

import pytest
from test_evaluation import CONFIG, AnswerModel, dataset

from csv_agent.analysis import AnalysisAttempt, AnalysisResult
from csv_agent.benchmark import load_manifest
from csv_agent.config import RunLimits
from csv_agent.evaluation import EvaluationRecord, run_comparison, score_records
from csv_agent.model import ModelError, ModelReply, Usage


@pytest.mark.docker
def test_fixed_strategy_ablation_cannot_promote_the_base_group_to_inspection(tmp_path):
    root, manifest = dataset(tmp_path)

    class EscalatingModel:
        def generate(self, prompt, *, timeout_seconds):
            return ModelReply(
                json.dumps(
                    {
                        "action": "inspect",
                        "code": "raise AssertionError('must not execute')",
                    }
                ),
                None,
            )

    report = compare(
        root,
        manifest,
        tmp_path / "comparison",
        strategies=["langgraph"],
        models={"langgraph": EscalatingModel()},
        strict_strategy=True,
    )
    result = report.runs["langgraph"].records[0].analysis
    assert result.status == "failed"
    assert result.execution_attempts == 0 and result.observations == []
    assert "fixed strategy" in result.error
    assert report.strict_strategy is True
    assert report.runs["langgraph"].summary.total == 1


@pytest.mark.docker
def test_parallel_selected_tasks_keep_order_failures_and_independent_budgets(tmp_path):
    root, manifest = dataset(tmp_path, count=2)
    rendezvous = Barrier(2, timeout=10)

    class ConcurrentModel:
        def generate(self, prompt, *, timeout_seconds):
            # Both tasks must reach the external model boundary concurrently.
            slot = rendezvous.wait()
            if slot == 0:
                raise ModelError(
                    "request timed out", timed_out=True, diagnostic="timeout"
                )
            return AnswerModel("@total[45]").generate(
                prompt, timeout_seconds=timeout_seconds
            )

    report = compare(
        root,
        manifest,
        tmp_path / "comparison",
        sample_ids=[1, 0],
        strategies=["single"],
        models={"single": ConcurrentModel()},
        workers=2,
    )
    summary = report.runs["single"].summary
    assert report.workers == 2
    assert [r.id for r in report.runs["single"].records] == [1, 0]
    assert [r.id for r in report.paired_records] == [1, 0]
    assert summary.total == 2 and summary.correct_questions == 1
    assert summary.failures == {"timeout": 1}
    assert summary.model_calls == 2 and summary.execution_attempts == 1
    assert summary.request_usage.unknown_usage_requests["input_tokens"] == 1
    saved = json.loads((tmp_path / "comparison/run.json").read_text())
    assert saved["workers"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "comparison/single/analysis.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [r["id"] for r in rows] == [1, 0]


def compare(root, manifest, output, **options):
    return run_comparison(
        root,
        manifest,
        split=options.pop("split", "development"),
        output_dir=output,
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(),
        model_version="controlled-fixture",
        **options,
    )


@pytest.mark.docker
def test_selected_questions_and_strategies_have_their_own_fixed_denominator(tmp_path):
    root, manifest = dataset(tmp_path, count=4)
    report = run_comparison(
        root,
        manifest,
        split="development",
        sample_ids=[3, 1],
        strategies=["tool_loop", "langgraph"],
        output_dir=tmp_path / "comparison",
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(),
        model_version="controlled-fixture",
        models={
            "tool_loop": AnswerModel("@total[44]"),
            "langgraph": AnswerModel("@total[45]"),
        },
    )
    assert report.strategy_order == ["tool_loop", "langgraph"]
    assert [record.id for record in report.paired_records] == [3, 1]
    assert report.selection.sample_ids == [3, 1]
    assert report.selection.purpose == "optimization"
    assert report.selection.total == 2
    group = report.selection.file_groups["sales.csv"]
    assert group.selected_ids == [3, 1]
    assert group.available_ids == [0, 1, 2, 3]
    assert not group.complete
    for run in report.runs.values():
        assert run.summary.total == 2
        assert run.summary.total_subquestions == 2
        assert run.selection == report.selection
    assert report.deltas["langgraph_vs_tool_loop"].question_accuracy == 1
    assert report.deltas["langgraph_vs_tool_loop"].wins == 2
    saved = json.loads((tmp_path / "comparison/run.json").read_text())
    assert saved["selection"] == report.selection.model_dump()
    assert not (tmp_path / "comparison/single").exists()


@pytest.mark.parametrize("ids", [[], [0, 0], [99], [True], ["0"]])
def test_invalid_selection_does_not_start_a_batch(tmp_path, ids):
    root, manifest = dataset(tmp_path)
    with pytest.raises(ValueError):
        compare(root, manifest, tmp_path / "comparison", sample_ids=ids)
    assert not (tmp_path / "comparison").exists()


@pytest.mark.parametrize(
    "strategies", [[], ["langgraph", "langgraph"], ["future_solver"]]
)
def test_unimplemented_or_duplicate_strategies_do_not_start_a_batch(
    tmp_path, strategies
):
    root, manifest = dataset(tmp_path)
    with pytest.raises(ValueError):
        compare(root, manifest, tmp_path / "comparison", strategies=strategies)
    assert not (tmp_path / "comparison").exists()


def test_holdout_requires_complete_files_in_the_final_split(tmp_path):
    root, manifest = dataset(tmp_path, count=2)
    with pytest.raises(ValueError, match="final"):
        compare(root, manifest, tmp_path / "comparison", purpose="holdout")
    lock = json.loads(manifest.read_text())
    lock["splits"] = {"development": [], "final": [0, 1]}
    manifest.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="complete file"):
        compare(
            root,
            manifest,
            tmp_path / "comparison",
            split="final",
            sample_ids=[0],
            purpose="holdout",
        )
    assert not (tmp_path / "comparison").exists()


def test_original_optimized_files_cannot_be_relabelled_as_holdout(tmp_path):
    root, manifest = dataset(tmp_path)
    lock = json.loads(manifest.read_text())
    original = json.loads(Path("benchmarks/dabench/manifest.json").read_text())
    sample = next(
        s for s in original["samples"] if s["id"] in original["splits"]["final"]
    )
    lock["tables"]["sales.csv"] = original["tables"][sample["file_name"]]
    lock["splits"] = {"development": [], "final": [0]}
    manifest.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="optimization set"):
        compare(
            root, manifest, tmp_path / "comparison", split="final", purpose="holdout"
        )
    assert not (tmp_path / "comparison").exists()


@pytest.mark.docker
def test_selected_failures_stay_zero_and_usage_is_counted_per_request(tmp_path):
    root, manifest = dataset(tmp_path, count=4)

    class MixedModel:
        def __init__(self):
            self.index = 0

        def generate(self, prompt, *, timeout_seconds):
            self.index += 1
            if self.index == 1:
                return AnswerModel("@total[45]").generate(
                    prompt, timeout_seconds=timeout_seconds
                )
            if self.index == 2:
                raise ModelError(
                    "Model request timed out",
                    timed_out=True,
                    diagnostic="timeout",
                    usage=Usage(input_tokens=7),
                )
            if self.index == 3:
                return ModelReply("", None)
            return ModelReply(
                '{"code":"raise ValueError(\'broken\')"}',
                Usage(input_tokens=0, output_tokens=2, cache_read_input_tokens=0),
            )

    report = compare(
        root,
        manifest,
        tmp_path / "comparison",
        strategies=["single"],
        models={"single": MixedModel()},
    )
    summary = report.runs["single"].summary
    assert summary.total == 4
    assert summary.total_subquestions == 4
    assert summary.correct_questions == 1
    assert summary.question_accuracy == 0.25
    assert summary.failures == {"timeout": 1, "response_format": 1, "failed": 1}
    assert [r.correctness for r in report.runs["single"].records] == [
        {"total": True},
        {"total": False},
        {"total": False},
        {"total": False},
    ]
    accounting = summary.request_usage
    assert accounting.unique_requests == 4
    assert accounting.duplicate_requests == 0
    assert accounting.known_usage == Usage(
        input_tokens=19, output_tokens=22, cache_read_input_tokens=0
    )
    assert accounting.unknown_usage_requests == {
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_read_input_tokens": 2,
    }
    assert accounting.usage == Usage()
    assert report.request_usage == accounting
    assert not report.deltas


def test_saved_request_copies_are_counted_once_and_conflicts_are_rejected(tmp_path):
    root, manifest = dataset(tmp_path)
    attempt = AnalysisAttempt(
        number=1,
        usage=Usage(input_tokens=12, output_tokens=3, cache_read_input_tokens=0),
    )
    record = EvaluationRecord(
        id=0,
        file_name="sales.csv",
        analysis=AnalysisResult(
            run_id="resumed-task",
            status="failed",
            attempts=[attempt, attempt.model_copy(deep=True)],
        ),
    )
    summary = score_records(
        root, load_manifest(manifest), [record], include_request_usage=True
    )
    assert summary.model_calls == 1
    assert summary.known_usage == Usage(
        input_tokens=12, output_tokens=3, cache_read_input_tokens=0
    )
    assert summary.request_usage.duplicate_requests == 1
    assert summary.request_usage.request_keys == [("resumed-task", 1)]
    record.analysis.attempts[1].usage.input_tokens = 99
    with pytest.raises(ValueError, match="Conflicting evidence"):
        score_records(root, load_manifest(manifest), [record])


@pytest.mark.docker
def test_selected_batches_keep_labels_host_only_until_every_strategy_finishes(tmp_path):
    root, manifest = dataset(tmp_path, count=2)
    labels = root / "labels.jsonl"
    labels.write_text(
        "".join(
            json.dumps({"id": i, "common_answers": [["total", "host-only-sentinel"]]})
            + "\n"
            for i in range(2)
        )
    )
    lock = json.loads(manifest.read_text())
    lock["dataset"]["labels_sha256"] = hashlib.sha256(labels.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(lock))
    label_bytes = labels.read_bytes()
    labels.unlink()
    finished = set()

    class IsolatedModel:
        def __init__(self, name):
            self.name = name

        def generate(self, prompt, *, timeout_seconds):
            assert not labels.exists()
            assert "host-only-sentinel" not in prompt
            assert "common_answers" not in prompt
            finished.add(self.name)
            if finished == {"tool_loop", "langgraph"}:
                labels.write_bytes(label_bytes)
            code = json.loads(
                AnswerModel("@total[45]")
                .generate(prompt, timeout_seconds=timeout_seconds)
                .text
            )["code"]
            code = (
                "from pathlib import Path\nassert sorted(p.name for p in Path('/input').iterdir()) == ['analysis.py', 'data.csv']\n"
                + f"assert not Path({str(labels)!r}).exists()\n"
                + code
            )
            return ModelReply(json.dumps({"code": code}), None)

    report = compare(
        root,
        manifest,
        tmp_path / "comparison",
        sample_ids=[1],
        strategies=["tool_loop", "langgraph"],
        models={s: IsolatedModel(s) for s in ["tool_loop", "langgraph"]},
    )
    for run in report.runs.values():
        assert run.summary.execution_successes == 1
        assert run.summary.correct_questions == 0
    assert (
        "host-only-sentinel" not in (tmp_path / "comparison/paired.jsonl").read_text()
    )


@pytest.mark.docker
@pytest.mark.parametrize("selection_arg", ["--sample-ids", "--sample-list"])
def test_cli_selects_one_implemented_strategy_and_saves_failures(
    tmp_path, selection_arg
):
    root, manifest = dataset(tmp_path, count=3)
    ids_file = tmp_path / "ids.json"
    ids_file.write_text("[2, 0]")
    environment = {
        **os.environ,
        "MODEL_ENDPOINT": CONFIG.endpoint,
        "MODEL_NAME": "fixture",
        "MODEL_TEMPERATURE": "1",
        "MODEL_API_KEY_ENV": "ABSENT_DIAGNOSTIC_KEY",
    }
    environment.pop("ABSENT_DIAGNOSTIC_KEY", None)
    args = ["2", "0"] if selection_arg == "--sample-ids" else [str(ids_file)]
    completed = subprocess.run(
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
            "--strategies",
            "langgraph",
            "--purpose",
            "optimization",
            selection_arg,
            *args,
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((tmp_path / "comparison/report.json").read_text())
    assert report["mode"] == "live"
    assert report["selection"]["sample_ids"] == [2, 0]
    assert report["strategy_order"] == ["langgraph"]
    assert report["runs"]["langgraph"]["summary"]["failures"] == {"authentication": 2}
    assert report["runs"]["langgraph"]["summary"]["total"] == 2
    assert report["deltas"] == {}
    summary = (tmp_path / "comparison/summary.md").read_text()
    assert "optimization" in summary and "Unknown input requests" in summary
    assert "account dashboard" in summary


@pytest.mark.docker
def test_untouched_complete_final_file_can_be_reported_as_holdout(tmp_path):
    root, manifest = dataset(tmp_path)
    lock = json.loads(manifest.read_text())
    lock["splits"] = {"development": [], "final": [0]}
    manifest.write_text(json.dumps(lock))
    report = compare(
        root,
        manifest,
        tmp_path / "comparison",
        split="final",
        purpose="holdout",
        strategies=["single"],
        models={"single": AnswerModel("@total[45]")},
    )
    assert report.selection.purpose == "holdout"
    assert report.selection.file_groups["sales.csv"].complete
    assert report.runs["single"].summary.correct_questions == 1


def test_selected_ids_must_still_belong_to_the_declared_split(tmp_path):
    root, manifest = dataset(tmp_path)
    lock = json.loads(manifest.read_text())
    lock["splits"] = {"development": [], "final": [0]}
    manifest.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="locked split"):
        compare(root, manifest, tmp_path / "comparison", sample_ids=[0])
    assert not (tmp_path / "comparison").exists()


@pytest.mark.docker
def test_selected_damaged_files_are_failures_for_every_chosen_strategy(tmp_path):
    root, manifest = dataset(tmp_path, count=3)
    (root / "da-dev-tables/sales.csv").write_bytes(b"sales\n99\n")
    report = compare(
        root,
        manifest,
        tmp_path / "comparison",
        sample_ids=[2],
        strategies=["tool_loop", "langgraph"],
        models={s: AnswerModel("@total[45]") for s in ["tool_loop", "langgraph"]},
    )
    for run in report.runs.values():
        assert run.summary.total == 1
        assert run.summary.failures == {"data_error": 1}
        assert run.summary.model_calls == 0
        assert run.records[0].correctness == {"total": False}
    assert report.request_usage.unique_requests == 0
