"""Ticket 03 through the agreed analyze boundary, using real isolated execution."""

import json

import pytest

from csv_agent.analysis import analyze
from csv_agent.model import ModelReply, Usage


class SequenceModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def generate(self, prompt, *, timeout_seconds):
        self.prompts.append(json.loads(prompt))
        return ModelReply(next(self.replies), Usage(input_tokens=10, output_tokens=5))


SUCCESS = """
import json, pandas as pd
value = int(pd.read_csv('/input/data.csv')['sales'].sum())
json.dump({'answer': str(value)}, open('/output/result.json', 'w'))
"""


@pytest.mark.docker
def test_execution_failure_is_repaired_with_actual_feedback():
    model = SequenceModel(
        [
            json.dumps({"code": "raise ValueError('repair-me')"}),
            json.dumps({"code": SUCCESS}),
        ]
    )
    result = analyze(b"sales\n10\n20\n15\n", "sum", model=model)
    assert result.status == "completed", result.error
    assert result.answer == "45"
    assert result.execution_attempts == 2
    assert len(result.attempts) == 2
    assert "repair-me" in result.attempts[0].error
    assert "repair-me" in json.dumps(model.prompts[1])
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 10
    assert result.usage.cache_read_input_tokens is None
    assert result.seconds >= sum(attempt.seconds for attempt in result.attempts)


@pytest.mark.docker
@pytest.mark.parametrize("maximum", [1, 3])
def test_repeated_code_errors_stop_at_execution_budget(maximum):
    from csv_agent.config import RunLimits

    model = SequenceModel(
        [json.dumps({"code": "raise ValueError('still broken')"})] * maximum
    )
    result = analyze(
        b"sales\n10\n", "sum", model=model, limits=RunLimits(max_executions=maximum)
    )
    assert result.status == "failed"
    assert result.answer is None
    assert result.execution_attempts == maximum
    assert len(result.attempts) == maximum
    assert "execution limit" in result.stop_reason
    assert "still broken" in result.error


@pytest.mark.docker
def test_invalid_instruction_is_repaired_before_any_execution():
    model = SequenceModel(["not json", json.dumps({"code": SUCCESS})])
    result = analyze(b"sales\n10\n20\n15\n", "sum", model=model)
    assert result.status == "completed", result.error
    assert result.answer == "45"
    assert result.execution_attempts == 1
    assert result.attempts[0].execution is None
    assert "valid JSON" in result.attempts[0].error


def test_format_repair_budget_is_task_wide():
    model = SequenceModel(["not json", "still not json"])
    result = analyze(b"sales\n10\n", "sum", model=model)
    assert result.status == "failed"
    assert len(result.attempts) == 2
    assert result.execution_attempts == 0
    assert "instruction repair limit" in result.stop_reason


@pytest.mark.docker
def test_task_deadline_stops_execution_and_cleans_container():
    import subprocess

    from csv_agent.config import RunLimits

    model = SequenceModel([json.dumps({"code": "while True: pass"})])
    result = analyze(
        b"sales\n10\n",
        "sum",
        model=model,
        limits=RunLimits(task_timeout_seconds=1, timeout_seconds=20),
    )
    assert result.status == "timeout"
    assert result.error == "Analysis task timed out"
    assert result.execution_attempts == 1
    assert result.answer is None
    assert result.seconds < 5
    remaining = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=csv-agent-{result.run_id}"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert not remaining.stdout.strip()


def test_model_retry_budget_and_unknown_usage(monkeypatch):
    import httpx

    from csv_agent.config import ModelConfig, RunLimits

    monkeypatch.setenv("TEST_KEY", "test-secret")
    original = httpx.AsyncClient
    requests = []

    def provider(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.ReadTimeout("raw private diagnostic")
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": '{"status":"insufficient_input","reason":"missing cost"}',
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(provider), **kwargs),
    )
    result = analyze(
        b"sales\n10\n",
        "profit",
        config=ModelConfig(
            endpoint="https://model.example/messages",
            name="test",
            api_key_env="TEST_KEY",
        ),
        limits=RunLimits(max_model_retries=1),
    )
    assert result.status == "insufficient_input"
    assert len(result.attempts) == 2
    assert result.attempts[0].error == "Model request timed out"
    assert result.usage is None
    assert result.attempts[1].usage.input_tokens == 10
    assert "raw private diagnostic" not in result.model_dump_json()


def test_repeated_model_timeouts_stop_at_request_retry_budget(monkeypatch):
    import httpx

    from csv_agent.config import ModelConfig, RunLimits

    monkeypatch.setenv("TEST_KEY", "test-secret")
    original = httpx.AsyncClient

    def provider(request):
        raise httpx.ReadTimeout("private")

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(provider), **kwargs),
    )
    result = analyze(
        b"sales\n10\n",
        "sum",
        config=ModelConfig(
            endpoint="https://model.example/messages",
            name="test",
            api_key_env="TEST_KEY",
        ),
        limits=RunLimits(max_model_retries=1),
    )
    assert result.status == "timeout"
    assert len(result.attempts) == 2
    assert result.execution_attempts == 0
    assert "retry limit" in result.stop_reason


@pytest.mark.parametrize("injected", [False, True])
def test_task_deadline_cancels_slow_streaming_model(monkeypatch, injected):
    import asyncio
    import time

    import httpx

    from csv_agent.config import ModelConfig, RunLimits

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(100):
                await asyncio.sleep(0.02)
                yield b" "

    monkeypatch.setenv("TEST_KEY", "test-secret")
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=SlowStream())
            ),
            **kwargs,
        ),
    )
    from csv_agent.model import HttpModel

    config = ModelConfig(
        endpoint="https://model.example/messages", name="test", api_key_env="TEST_KEY"
    )
    started = time.monotonic()
    result = analyze(
        b"sales\n10\n",
        "sum",
        config=config,
        model=HttpModel(config) if injected else None,
        limits=RunLimits(task_timeout_seconds=0.1),
    )
    assert result.status == "timeout"
    assert result.error == "Analysis task timed out"
    assert time.monotonic() - started < 1
    assert result.execution_attempts == 0


@pytest.mark.docker
def test_transient_retry_preserves_code_failure_feedback():
    from csv_agent.config import RunLimits
    from csv_agent.model import ModelError

    class RetryModel:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, *, timeout_seconds):
            self.prompts.append(json.loads(prompt))
            if len(self.prompts) == 1:
                return ModelReply(
                    json.dumps({"code": "raise ValueError('keep-this-feedback')"}), None
                )
            if len(self.prompts) == 2:
                raise ModelError(
                    "Model request timed out", retryable=True, timed_out=True
                )
            return ModelReply(json.dumps({"code": SUCCESS}), None)

    model = RetryModel()
    result = analyze(
        b"sales\n10\n20\n15\n",
        "sum",
        model=model,
        limits=RunLimits(max_model_retries=1),
    )
    assert result.status == "completed", result.error
    assert model.prompts[1] == model.prompts[2]
    assert "keep-this-feedback" in json.dumps(model.prompts[2])
