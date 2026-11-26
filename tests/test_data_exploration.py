"""Active data inspection through the agreed public analysis boundary."""

import json

import pytest

from csv_agent.analysis import analyze
from csv_agent.config import RunLimits
from csv_agent.model import ModelReply, Usage

CSV = b"region,sales\neast,10\nwest,20\neast,\nnorth,15\n"
INSPECT = """
import json, pandas as pd
df = pd.read_csv('/input/data.csv')
observations = {
    'sales_dtype': str(df['sales'].dtype),
    'missing_sales': int(df['sales'].isna().sum()),
    'regions': sorted(df['region'].unique().tolist()),
    'east_rows': int(df['region'].eq('east').sum()),
}
json.dump({'observations': observations}, open('/output/result.json', 'w'))
"""
CANDIDATE = """
import json, pandas as pd
df = pd.read_csv('/input/data.csv')
total = int(df.loc[df['region'].eq('east'), 'sales'].sum())
json.dump({'answer': str(total), 'table': [{'east_sales': total}]},
          open('/output/result.json', 'w'))
"""


class SequenceModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def generate(self, prompt, *, timeout_seconds):
        self.prompts.append(json.loads(prompt))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, ModelReply):
            return reply
        return ModelReply(json.dumps(reply), Usage(input_tokens=10, output_tokens=5))


@pytest.mark.docker
@pytest.mark.parametrize("strategy", ["langgraph_explore", "langgraph", "tool_loop"])
def test_inspection_and_candidate_continue_until_evidence_is_submitted(strategy):
    class ObservingModel(SequenceModel):
        def generate(self, prompt, *, timeout_seconds):
            payload = json.loads(prompt)
            if len(self.prompts) == 1:
                assert payload["observations"][0]["values"] == {
                    "sales_dtype": "float64",
                    "missing_sales": 1,
                    "regions": ["east", "north", "west"],
                    "east_rows": 2,
                }
            if len(self.prompts) == 2:
                assert payload["candidates"][0]["result"]["answer"] == "10"
            return super().generate(prompt, timeout_seconds=timeout_seconds)

    model = ObservingModel(
        [
            {"action": "inspect", "code": INSPECT},
            {"action": "candidate", "code": CANDIDATE},
            {"action": "submit", "candidate": 2},
        ]
    )
    steps = []
    result = analyze(
        CSV,
        "east \u7684Total sales，\u5ffd\u7565\u7f3a\u5931\u503c",
        model=model,
        strategy=strategy,
        on_step=steps.append,
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert result.table == [{"east_sales": 10}]
    assert result.execution_attempts == 2
    assert [attempt.action for attempt in result.attempts] == [
        "inspect",
        "candidate",
        "submit",
    ]
    assert result.attempts[-1].execution is None
    assert result.observations[0].values["missing_sales"] == 1
    assert result.candidates[0].number == 2
    assert result.code == CANDIDATE
    assert result.usage.input_tokens == 30
    assert steps == result.steps


@pytest.mark.docker
def test_submission_and_format_repair_share_requests_without_spending_executions():
    model = SequenceModel(
        [
            {"action": "candidate", "code": CANDIDATE},
            "invalid instruction",
            {"action": "submit", "candidate": 1},
        ]
    )
    result = analyze(
        CSV,
        "east sales",
        model=model,
        strategy="langgraph_explore",
        limits=RunLimits(max_executions=1, max_model_calls=3),
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert result.execution_attempts == 1
    assert len(result.attempts) == 3
    assert result.attempts[1].diagnostic == "response_format"
    assert [p["remaining_budget"]["model_calls"] for p in model.prompts] == [2, 1, 0]
    assert [p["remaining_budget"]["executions"] for p in model.prompts] == [1, 0, 0]


def test_exploration_defaults_to_the_confirmed_single_task_budget(monkeypatch):
    for name in ("MAX_MODEL_CALLS", "MAX_EXECUTIONS", "TASK_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    model = SequenceModel(
        [
            {"status": "insufficient_input", "reason": "Missing cost field"},
        ]
    )
    result = analyze(CSV, "profit", model=model, strategy="langgraph_explore")
    assert result.status == "insufficient_input"
    remaining = model.prompts[0]["remaining_budget"]
    assert remaining["model_calls"] == 15
    assert remaining["executions"] == 12
    assert 1190 < remaining["seconds"] <= 1200
    assert result.execution_attempts == 0


def test_submission_without_computed_evidence_is_an_explicit_failure():
    model = SequenceModel([{"action": "submit", "candidate": 99}])
    result = analyze(
        CSV,
        "east sales",
        model=model,
        strategy="langgraph_explore",
        limits=RunLimits(max_format_repairs=0),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert result.execution_attempts == 0
    assert "successfully computed candidate" in result.error


@pytest.mark.docker
@pytest.mark.parametrize(
    "action,code", [("inspect", INSPECT), ("candidate", CANDIDATE)]
)
def test_request_exhaustion_keeps_evidence_without_publishing_an_answer(action, code):
    model = SequenceModel([{"action": action, "code": code}])
    result = analyze(
        CSV,
        "east sales",
        model=model,
        strategy="langgraph_explore",
        limits=RunLimits(max_model_calls=1, max_executions=1),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert result.stop_reason == "Model call budget exhausted"
    assert result.execution_attempts == 1
    assert bool(result.observations if action == "inspect" else result.candidates)


@pytest.mark.docker
def test_execution_exhaustion_does_not_reset_after_inspection():
    model = SequenceModel(
        [
            {"action": "inspect", "code": INSPECT},
            {"action": "candidate", "code": CANDIDATE},
        ]
    )
    result = analyze(
        CSV,
        "east sales",
        model=model,
        strategy="langgraph_explore",
        limits=RunLimits(max_model_calls=3, max_executions=1),
    )
    assert result.status == "failed"
    assert result.execution_attempts == 1
    assert result.answer is None
    assert len(result.observations) == 1
    assert not result.candidates
    assert result.attempts[-1].execution is None
    assert result.stop_reason == "Analysis code execution limit reached"


@pytest.mark.docker
def test_transport_retry_keeps_observations_and_unknown_usage_within_budget():
    from csv_agent.model import ModelError

    model = SequenceModel(
        [
            {"action": "inspect", "code": INSPECT},
            ModelError(
                "Model connection failed", retryable=True, diagnostic="connection"
            ),
            {"action": "candidate", "code": CANDIDATE},
            {"action": "submit", "candidate": 3},
        ]
    )
    result = analyze(
        CSV,
        "east sales",
        model=model,
        strategy="langgraph_explore",
        limits=RunLimits(max_model_calls=4, max_executions=2, max_model_retries=1),
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert result.execution_attempts == 2
    assert len(result.attempts) == 4
    assert model.prompts[1]["observations"] == model.prompts[2]["observations"]
    assert model.prompts[1]["feedback"] == model.prompts[2]["feedback"]
    assert [prompt["remaining_budget"]["model_calls"] for prompt in model.prompts] == [
        3,
        2,
        1,
        0,
    ]
    assert model.prompts[2]["observations"][0]["values"]["east_rows"] == 2
    assert result.usage is None
    assert result.attempts[1].usage is None
    assert result.attempts[2].usage.input_tokens == 10


@pytest.mark.docker
def test_isolated_phases_rebuild_from_data_and_cannot_read_previous_variables():
    code = CANDIDATE.replace(
        "import json, pandas as pd",
        "import json, pandas as pd\nassert 'df' not in globals()",
    )
    model = SequenceModel(
        [
            {"action": "inspect", "code": INSPECT},
            {"action": "candidate", "code": code},
            {"action": "submit", "candidate": 2},
        ]
    )
    result = analyze(CSV, "east sales", model=model, strategy="langgraph_explore")
    assert result.status == "completed", result.error
    assert result.answer == "10"


@pytest.mark.docker
def test_inspection_timeout_stops_immediately_and_removes_its_container():
    import subprocess

    model = SequenceModel([{"action": "inspect", "code": "while True: pass"}])
    result = analyze(
        CSV,
        "east sales",
        model=model,
        strategy="langgraph_explore",
        limits=RunLimits(task_timeout_seconds=1, timeout_seconds=20),
    )
    assert result.status == "timeout"
    assert result.stop_reason == "Task time budget exhausted"
    assert result.execution_attempts == 1
    assert result.answer is None
    remaining = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=csv-agent-{result.run_id}"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert not remaining.stdout.strip()


@pytest.mark.docker
def test_missing_fields_found_during_inspection_end_with_a_limit_explanation():
    model = SequenceModel(
        [
            {"action": "inspect", "code": INSPECT},
            {
                "status": "insufficient_input",
                "reason": "Missing cost field; profit cannot be computed",
            },
        ]
    )
    result = analyze(CSV, "profit", model=model, strategy="langgraph_explore")
    assert result.status == "insufficient_input"
    assert result.execution_attempts == 1
    assert result.answer is None
    assert len(result.observations) == 1
    assert result.error == "Missing cost field; profit cannot be computed"


@pytest.mark.docker
def test_oversized_observation_returns_a_specific_repairable_limit():
    code = "import json; json.dump({'observations': {'values': 'x' * 65536}}, open('/output/result.json', 'w'))"
    model = SequenceModel([{"action": "inspect", "code": code}])
    result = analyze(
        CSV,
        "inspect",
        model=model,
        strategy="langgraph_explore",
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed"
    assert not result.observations
    assert result.answer is None
    assert "Inspection observations exceed 64 KiB" in result.error


@pytest.mark.docker
@pytest.mark.parametrize("failed_action", ["inspect", "candidate"])
def test_saved_candidate_can_be_submitted_after_the_last_repairable_execution_fails(
    failed_action,
):
    model = SequenceModel(
        [
            {"action": "candidate", "code": CANDIDATE},
            {
                "action": failed_action,
                "code": "raise ValueError('further calculation failed')",
            },
            {"action": "submit", "candidate": 1},
        ]
    )
    result = analyze(
        CSV,
        "east sales",
        model=model,
        strategy="langgraph_explore",
        limits=RunLimits(max_executions=2, max_model_calls=3),
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert result.execution_attempts == 2
    assert len(result.attempts) == 3
    assert "further calculation failed" in result.attempts[1].error
    assert result.execution == result.attempts[0].execution
    assert result.code == CANDIDATE
