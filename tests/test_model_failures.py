"""Observe provider failures through the agreed analysis-task interface."""

import httpx
import pytest

from csv_agent.analysis import analyze
from csv_agent.config import ModelConfig


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(401, text="do-not-leak-secret"), "HTTP 401"),
        (httpx.Response(200, json={"content": []}), "empty answer"),
        (
            httpx.Response(
                200, json={"content": [{"type": "text", "text": "do-not-leak-secret"}]}
            ),
            "credential material",
        ),
        (httpx.Response(200, json={"choices": []}), "Invalid model response"),
        (
            httpx.Response(200, text="<html>upstream error</html>"),
            "Invalid model response",
        ),
        (
            httpx.Response(
                200, json={"content": [{"type": "text", "text": "not json"}]}
            ),
            "valid JSON",
        ),
    ],
)
def test_provider_errors_are_safe_task_failures(monkeypatch, response, expected):
    monkeypatch.setenv("TEST_MODEL_KEY", "do-not-leak-secret")
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(lambda request: response), **kwargs
        ),
    )
    config = ModelConfig(
        endpoint="https://model.example/v1/messages",
        name="test",
        api_key_env="TEST_MODEL_KEY",
    )
    result = analyze(b"sales\n10\n", "sum", config=config)
    assert result.status == "failed"
    assert expected in result.error
    assert result.execution is None
    assert "do-not-leak-secret" not in result.model_dump_json()


def test_model_timeout_is_reported_as_timeout(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_KEY", "unused-test-key")
    original_client = httpx.AsyncClient

    def timeout(request):
        raise httpx.ReadTimeout("sensitive raw error")

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(timeout), **kwargs
        ),
    )
    config = ModelConfig(
        endpoint="https://model.example/v1/messages",
        name="test",
        api_key_env="TEST_MODEL_KEY",
    )
    result = analyze(b"sales\n10\n", "sum", config=config)
    assert result.status == "timeout"
    assert result.error == "Model request timed out"
    assert result.execution is None


def test_missing_key_does_not_start_execution(monkeypatch):
    monkeypatch.delenv("TEST_MODEL_KEY", raising=False)
    config = ModelConfig(
        endpoint="https://model.example/v1/messages",
        name="test",
        api_key_env="TEST_MODEL_KEY",
    )
    result = analyze(b"sales\n10\n", "sum", config=config)
    assert result.status == "failed"
    assert result.error == "Missing API key environment variable: TEST_MODEL_KEY"
    assert result.execution is None
