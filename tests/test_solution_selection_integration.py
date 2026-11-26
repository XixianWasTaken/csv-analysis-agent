"""Independent solution selection uses the real HTTP and batch boundaries."""

import json

import httpx
import pytest
from test_candidate_verification import CHECKS, RECOMPUTED_TOTAL, WRONG_TOTAL, review
from test_data_exploration import CSV, SequenceModel
from test_evaluation import CONFIG, dataset
from test_model_portability import connect_fixture
from test_solution_disagreements import (
    CHECK_SCOPE,
    FORMAT,
    WRONG_FIELDS,
    recompute,
    select,
)

from csv_agent.analysis import analyze
from csv_agent.config import ModelConfig, RunLimits
from csv_agent.evaluation import run_comparison


@pytest.mark.docker
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("mode", ["structured", "tools"])
def test_http_modes_compare_and_select_with_executed_field_evidence(
    monkeypatch, protocol, mode
):
    actions = [
        ("compute_candidate", {"action": "candidate", "code": WRONG_FIELDS}),
        ("recompute_candidate", recompute(1)),
        ("review_candidate", review(2)),
        ("inspect_data", {"action": "inspect", "code": CHECK_SCOPE}),
        (
            "select_candidate",
            select(
                2,
                evidence=[2, 4],
                resolutions={
                    "total": {
                        "finding": "The source check confirms east total 10, excluding other regions.",
                        "evidence": [4],
                    }
                },
            ),
        ),
    ]
    requests = []

    def provider(request):
        payload = json.loads(request.content)
        name, arguments = actions[len(requests)]
        requests.append(payload)
        if protocol == "anthropic":
            content = (
                {"type": "tool_use", "id": "call", "name": name, "input": arguments}
                if mode == "tools"
                else {"type": "text", "text": json.dumps(arguments)}
            )
            return httpx.Response(200, json={"content": [content]})
        message = (
            {"content": json.dumps(arguments)}
            if mode == "structured"
            else {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }
        )
        return httpx.Response(200, json={"choices": [{"message": message}]})

    connect_fixture(monkeypatch, provider)
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        limits=RunLimits(max_model_calls=5, max_executions=3),
        config=ModelConfig(
            protocol=protocol,
            instruction_mode=mode,
            endpoint="https://fixture.example/messages",
            name="controlled-http",
            api_key_env="FIXTURE_KEY",
        ),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@total[10]\n@count[1]"
    assert result.selection.evidence == [2, 4]
    assert "private-fixture-credential" not in result.model_dump_json()
    final_prompt = json.loads(requests[-1]["messages"][-1]["content"])
    assert final_prompt["comparisons"][0]["fields"]["total"]["equal"] is False
    assert final_prompt["remaining_budget"]["executions"] == 0
    assert final_prompt["remaining_solutions"] == 1


@pytest.mark.docker
def test_batch_saves_selection_and_field_differences_without_labels_in_requests(
    tmp_path,
):
    root, manifest = dataset(tmp_path, csv_data=CSV)
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
    checks = {
        **CHECKS,
        "filters_and_groups": "All rows with present sales contribute; total 45.",
    }
    model = SequenceModel(
        [
            {"action": "candidate", "code": first},
            recompute(1, independent),
            review(2, checks=checks),
            {"action": "inspect", "code": inspection},
            select(
                2,
                evidence=[2, 4],
                resolutions={
                    "total": {
                        "finding": "The data check sums 10, 20 and 15 to 45; 15 is the mean, not total.",
                        "evidence": [4],
                    }
                },
            ),
        ]
    )
    output = tmp_path / "comparison"
    report = run_comparison(
        root,
        manifest,
        split="development",
        sample_ids=[0],
        strategies=["langgraph_resolve"],
        output_dir=output,
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(
            max_model_calls=16, max_executions=12, task_timeout_seconds=1200
        ),
        model_version="controlled-fixture",
        models={"langgraph_resolve": model},
    )
    run = report.runs["langgraph_resolve"]
    assert run.summary.total == 1
    assert run.records[0].correctness == {"total": True}
    assert run.summary.model_calls == 5
    assert run.summary.execution_attempts == 3
    saved = json.loads((output / "langgraph_resolve/analysis.jsonl").read_text())[
        "analysis"
    ]
    assert saved["answer"] == "@total[45]"
    assert saved["comparisons"][0]["fields"]["total"] == {
        "first": "15",
        "second": "45",
        "equal": False,
    }
    assert saved["selection"]["resolutions"]["total"]["evidence"] == [4]
    assert saved["candidates"][1]["recomputed_from"] == 1
    assert saved["candidates"][1]["code"] == independent
    assert saved["observations"][0]["values"]["east_total"] == 45
    assert all(
        "common_answers" not in json.dumps(prompt)
        and "correctness" not in json.dumps(prompt)
        and "labels.jsonl" not in json.dumps(prompt)
        for prompt in model.prompts
    )
