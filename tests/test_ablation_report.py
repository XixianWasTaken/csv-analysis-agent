"""Offline ablation delivery at the agreed batch evaluation boundary."""

import json

import pytest
from test_evaluation import CONFIG, AnswerModel, dataset

from csv_agent.config import RunLimits
from csv_agent.evaluation import run_comparison


@pytest.mark.docker
def test_ablation_delivery_rescores_pairs_and_exports_the_actual_selected_config(
    tmp_path,
):
    from csv_agent.ablations import summarize_ablations

    root, manifest = dataset(tmp_path, count=2)
    batches = {}
    for label, answer in [("initial", "@total[44]"), ("repeat", "@total[45]")]:
        batches[label] = tmp_path / label
        run_comparison(
            root,
            manifest,
            split="development",
            output_dir=batches[label],
            config=CONFIG.model_copy(update={"temperature": 1}),
            limits=RunLimits(),
            model_version="controlled-fixture",
            sample_ids=[1, 0],
            strategies=["langgraph"],
            models={"langgraph": AnswerModel(answer)},
        )
    delivery = summarize_ablations(
        root,
        batches,
        output_dir=tmp_path / "delivery",
        selected=("repeat", "langgraph"),
    )
    pair = delivery["comparisons"][0]
    assert pair["kind"] == "repeat"
    assert pair["wins"] == [1, 0] and pair["losses"] == []
    assert pair["question_accuracy_delta"] == 1
    assert delivery["batches"]["initial"]["runs"]["langgraph"]["summary"][
        "failures"
    ] == {"incorrect_answer": 2}
    selected = json.loads((tmp_path / "delivery/selected-config.json").read_text())
    assert selected["selection"]["sample_ids"] == [1, 0]
    assert selected["strategy"] == "langgraph"
    assert selected["limits"]["image"].startswith("sha256:")
    assert selected["mode"] == "controlled"
    assert "optimization" in (tmp_path / "delivery/summary.md").read_text()
    assert (
        "independent generalization" in (tmp_path / "delivery/summary.md").read_text()
    )


@pytest.mark.docker
def test_ablation_delivery_rejects_an_unreported_per_strategy_budget_change(tmp_path):
    from csv_agent.ablations import summarize_ablations

    root, manifest = dataset(tmp_path)
    output = tmp_path / "batch"
    run_comparison(
        root,
        manifest,
        split="development",
        output_dir=output,
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(),
        model_version="controlled-fixture",
        strategies=["langgraph"],
        models={"langgraph": AnswerModel("@total[45]")},
    )
    path = output / "report.json"
    report = json.loads(path.read_text())
    report["runs"]["langgraph"]["limits"]["max_model_calls"] = 40
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="settings"):
        summarize_ablations(root, {"batch": output}, output_dir=tmp_path / "delivery")
    assert not (tmp_path / "delivery").exists()
