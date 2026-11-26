"""Exploration uses existing HTTP and selected batch boundaries."""

import json

import httpx
import pytest
from test_data_exploration import CANDIDATE, CSV, INSPECT, SequenceModel
from test_evaluation import CONFIG, dataset
from test_model_portability import connect_fixture

from csv_agent.analysis import analyze
from csv_agent.config import ModelConfig, RunLimits
from csv_agent.evaluation import run_comparison


@pytest.mark.docker
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
@pytest.mark.parametrize("mode", ["structured", "tools"])
def test_http_protocols_share_validated_inspection_candidate_and_submission(
    monkeypatch, protocol, mode
):
    actions = [
        ("inspect_data", {"action": "inspect", "code": INSPECT}),
        ("compute_candidate", {"action": "candidate", "code": CANDIDATE}),
        ("submit_answer", {"action": "submit", "candidate": 2}),
    ]
    requests = []

    def provider(request):
        payload = json.loads(request.content)
        requests.append(payload)
        name, arguments = actions[len(requests) - 1]
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
                        "function": {
                            "name": name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                ],
            }
        )
        return httpx.Response(200, json={"choices": [{"message": message}]})

    connect_fixture(monkeypatch, provider)
    result = analyze(
        CSV,
        "east sales",
        strategy="langgraph_explore",
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
    assert len(requests) == 3
    assert "private-fixture-credential" not in result.model_dump_json()
    submitted_prompt = json.loads(requests[-1]["messages"][-1]["content"])
    assert submitted_prompt["observations"][0]["values"]["missing_sales"] == 1
    assert submitted_prompt["candidates"][0]["number"] == 2


@pytest.mark.docker
def test_selected_batch_exports_phases_and_scores_only_after_submission(tmp_path):
    root, manifest = dataset(tmp_path, csv_data=CSV)
    code = """
import json, pandas as pd
df = pd.read_csv('/input/data.csv')
total = int(df['sales'].sum())
json.dump({'answer': f'@total[{total}]'}, open('/output/result.json', 'w'))
"""
    model = SequenceModel(
        [
            {"action": "inspect", "code": INSPECT},
            {"action": "candidate", "code": code},
            {"action": "submit", "candidate": 2},
        ]
    )
    output = tmp_path / "comparison"
    report = run_comparison(
        root,
        manifest,
        split="development",
        sample_ids=[0],
        strategies=["langgraph_explore"],
        output_dir=output,
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits.from_env(exploration=True),
        model_version="controlled-fixture",
        models={"langgraph_explore": model},
    )
    run = report.runs["langgraph_explore"]
    assert run.summary.total == 1
    assert run.records[0].analysis.answer == "@total[45]"
    assert run.records[0].correctness == {"total": True}
    assert run.summary.model_calls == 3
    assert run.summary.execution_attempts == 2
    saved = json.loads((output / "langgraph_explore/analysis.jsonl").read_text())
    assert saved["analysis"]["observations"][0]["values"]["missing_sales"] == 1
    assert saved["analysis"]["candidates"][0]["number"] == 2
    assert saved["analysis"]["attempts"][-1]["action"] == "submit"
    for prompt in model.prompts:
        serialized = json.dumps(prompt)
        assert "common_answers" not in serialized
        assert "correctness" not in serialized
        assert "labels.jsonl" not in serialized
