"""Review actions use the existing provider and batch evaluation boundaries."""

import json

import httpx
import pytest
from test_candidate_verification import CHECKS, RECOMPUTED_TOTAL, WRONG_TOTAL, review
from test_data_exploration import CSV, SequenceModel
from test_evaluation import CONFIG, dataset
from test_model_portability import connect_fixture

from csv_agent.analysis import analyze
from csv_agent.config import ModelConfig, RunLimits
from csv_agent.evaluation import run_comparison


@pytest.mark.docker
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("mode", ["structured", "tools"])
def test_http_modes_support_review_and_independent_recomputation(
    monkeypatch, protocol, mode
):
    actions = [
        ("compute_candidate", {"action": "candidate", "code": WRONG_TOTAL}),
        (
            "review_candidate",
            review(
                1,
                "needs_revision",
                {
                    **CHECKS,
                    "filters_and_groups": "All regions were included; east is required",
                },
            ),
        ),
        (
            "recompute_candidate",
            {"action": "recompute", "candidate": 1, "code": RECOMPUTED_TOTAL},
        ),
        ("review_candidate", review(3)),
        ("submit_answer", {"action": "submit", "candidate": 3}),
    ]
    requests = []

    def provider(request):
        payload = json.loads(request.content)
        name, arguments = actions[len(requests)]
        requests.append(payload)
        if mode == "tools":
            definitions = payload["tools"]
            names = {
                tool["name"] if protocol == "anthropic" else tool["function"]["name"]
                for tool in definitions
            }
            assert name in names
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
        "east sales ignoring missing values",
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=5, max_executions=2),
        config=ModelConfig(
            protocol=protocol,
            instruction_mode=mode,
            endpoint="https://fixture.example/messages",
            name="controlled-http",
            api_key_env="FIXTURE_KEY",
        ),
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert result.execution_attempts == 2
    assert result.usage is None
    assert len(requests) == 5
    assert "private-fixture-credential" not in result.model_dump_json()
    recompute_prompt = json.loads(requests[2]["messages"][-1]["content"])
    assert recompute_prompt["reviews"][0]["outcome"] == "needs_revision"
    assert recompute_prompt["candidates"][0]["result"]["answer"] == "45"
    submit_prompt = json.loads(requests[-1]["messages"][-1]["content"])
    assert submit_prompt["reviews"][1]["checks"] == CHECKS
    assert submit_prompt["remaining_budget"]["executions"] == 0


@pytest.mark.docker
def test_batch_saves_review_and_recomputed_evidence_before_host_scoring(tmp_path):
    root, manifest = dataset(tmp_path, csv_data=CSV)
    first = WRONG_TOTAL.replace(
        "total = int(df['sales'].sum())", "total = int(df['sales'].mean())"
    ).replace("{'answer': str(total)}", "{'answer_fields': {'total': total}}")
    independent = RECOMPUTED_TOTAL.replace(
        "if row['region'] == 'east' and row['sales']", "if row['sales']"
    ).replace(
        "{'answer': str(total), 'table': [{'east_sales': total}]}",
        "{'answer_fields': {'total': total}, 'table': [{'total': total}]}",
    )
    checks = {
        **CHECKS,
        "filters_and_groups": "All regions contribute; three present sales values total 45",
        "statistical_definition": "Sum the amounts, not the mean of 15",
    }
    model = SequenceModel(
        [
            {"action": "candidate", "code": first},
            review(1, "needs_revision", checks),
            {"action": "recompute", "candidate": 1, "code": independent},
            review(3, checks=checks),
            {"action": "submit", "candidate": 3},
        ]
    )
    output = tmp_path / "comparison"
    report = run_comparison(
        root,
        manifest,
        split="development",
        sample_ids=[0],
        strategies=["langgraph_verify"],
        output_dir=output,
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(max_model_calls=5, max_executions=2),
        model_version="controlled-fixture",
        models={"langgraph_verify": model},
    )
    run = report.runs["langgraph_verify"]
    assert run.summary.total == 1
    assert run.records[0].analysis.answer == "@total[45]"
    assert run.records[0].correctness == {"total": True}
    assert run.summary.model_calls == 5
    assert run.summary.execution_attempts == 2
    saved = json.loads((output / "langgraph_verify/analysis.jsonl").read_text())[
        "analysis"
    ]
    assert saved["candidates"][0]["result"]["answer"] == "@total[15]"
    assert saved["candidates"][1]["recomputed_from"] == 1
    assert saved["reviews"][0]["outcome"] == "needs_revision"
    assert (
        saved["reviews"][1]["checks"]["statistical_definition"]
        == checks["statistical_definition"]
    )
    assert saved["selected_candidate"] == 3
    for prompt in model.prompts:
        serialized = json.dumps(prompt)
        assert "common_answers" not in serialized
        assert "correctness" not in serialized
        assert "labels.jsonl" not in serialized


@pytest.mark.docker
def test_encoded_credentials_in_nested_review_checks_are_rejected(monkeypatch):
    requests = []
    secret = "private-fixture-credential"

    def provider(request):
        requests.append(request)
        arguments = (
            {"action": "candidate", "code": RECOMPUTED_TOTAL}
            if len(requests) == 1
            else review(1, checks={**CHECKS, "field_meaning": secret})
        )
        encoded = json.dumps(arguments).replace(
            secret, "".join(f"\\u{ord(char):04x}" for char in secret)
        )
        return httpx.Response(
            200, json={"content": [{"type": "text", "text": encoded}]}
        )

    connect_fixture(monkeypatch, provider)
    result = analyze(
        CSV,
        "east sales",
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=2, max_executions=1),
        config=ModelConfig(
            endpoint="https://fixture.example/messages",
            name="controlled-http",
            api_key_env="FIXTURE_KEY",
        ),
    )
    assert result.status == "failed"
    assert result.diagnostic == "credential_material"
    assert not result.reviews
    assert secret not in result.model_dump_json()
