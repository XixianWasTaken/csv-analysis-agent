"""Ticket 05 at the agreed analyze seam; HTTP fixtures, real Docker compute."""

import json

import httpx
import pytest

from csv_agent.analysis import analyze
from csv_agent.config import ModelConfig

SUM = """
import json, pandas as pd
total = int(pd.read_csv('/input/data.csv')['sales'].sum())
with open('/output/result.json', 'w') as output:
    json.dump({'answer': str(total), 'table': [{'total': total}]}, output)
"""


def connect_fixture(monkeypatch, provider):
    monkeypatch.setenv("FIXTURE_KEY", "private-fixture-credential")
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(provider), **kwargs),
    )


@pytest.mark.docker
@pytest.mark.parametrize(
    ("protocol", "mode"),
    [
        ("openai", "tools"),
        ("anthropic", "tools"),
        ("openai", "structured"),
        ("anthropic", "structured"),
    ],
)
def test_service_configuration_completes_the_same_analysis(monkeypatch, protocol, mode):
    requests = []

    def provider(request):
        requests.append(request)
        payload = json.loads(request.content)
        if ("tools" in payload) != (mode == "tools"):
            return httpx.Response(400)
        if protocol == "anthropic":
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"type": "thinking", "thinking": "private reasoning"},
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "execute_analysis",
                            "input": {"code": SUM},
                        }
                        if mode == "tools"
                        else {"type": "text", "text": json.dumps({"code": SUM})},
                    ],
                    # This service omits usage entirely.
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None
                            if mode == "tools"
                            else json.dumps({"code": SUM}),
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "execute_analysis",
                                        "arguments": json.dumps({"code": SUM}),
                                    },
                                }
                            ]
                            if mode == "tools"
                            else [],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 34,
                    "prompt_tokens_details": {"cached_tokens": 5},
                },
            },
        )

    connect_fixture(monkeypatch, provider)
    config = ModelConfig(
        protocol=protocol,
        endpoint="https://other.example/v1/chat/completions"
        if protocol == "openai"
        else "https://first.example/v1/messages",
        name="other-model" if protocol == "openai" else "first-model",
        api_key_env="FIXTURE_KEY",
        instruction_mode=mode,
        temperature=0.2,
        max_tokens=2048,
    )
    result = analyze(b"sales\n10\n20\n15\n", "sum", config=config)
    assert result.status == "completed", result.error
    assert result.answer == "45"
    assert result.table == [{"total": 45}]
    if protocol == "openai":
        assert result.usage.model_dump() == {
            "input_tokens": 12,
            "output_tokens": 34,
            "cache_read_input_tokens": 5,
        }
    else:
        assert result.usage is None
        assert result.attempts[0].usage is None
    attempt = result.attempts[0]
    assert attempt.request.name == config.name
    assert attempt.request.protocol == protocol
    assert attempt.request.instruction_mode == mode
    assert attempt.request.temperature == 0.2
    assert attempt.request.max_tokens == 2048
    assert 0 < attempt.model_seconds <= attempt.seconds
    assert str(requests[0].url) == config.endpoint
    if protocol == "openai":
        assert (
            requests[0].headers["Authorization"] == "Bearer private-fixture-credential"
        )
    else:
        assert requests[0].headers["x-api-key"] == "private-fixture-credential"
    assert "private-fixture-credential" not in result.model_dump_json()
    assert "private reasoning" not in result.model_dump_json()


@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_native_input_limitation_ends_without_execution(monkeypatch, protocol):
    limitation = {"status": "insufficient_input", "reason": "Missing cost field"}
    body = (
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "report_insufficient_input",
                                    "arguments": json.dumps(limitation),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        if protocol == "openai"
        else {
            "content": [
                {
                    "type": "tool_use",
                    "name": "report_insufficient_input",
                    "input": limitation,
                }
            ]
        }
    )
    connect_fixture(monkeypatch, lambda request: httpx.Response(200, json=body))
    result = analyze(
        b"sales\n10\n",
        "profit",
        config=ModelConfig(
            protocol=protocol,
            endpoint="https://service.example/request",
            name="configured-model",
            api_key_env="FIXTURE_KEY",
            instruction_mode="tools",
        ),
    )
    assert result.status == "insufficient_input", result.error
    assert result.error == "Missing cost field"
    assert result.execution_attempts == 0
    assert result.usage is None


def test_unrequested_native_tool_does_not_execute_text_in_structured_mode(monkeypatch):
    connect_fixture(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"code": SUM}),
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "unexpected",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        ),
    )
    result = analyze(
        b"sales\n10\n",
        "sum",
        config=ModelConfig(
            protocol="openai",
            endpoint="https://service.example/request",
            name="configured-model",
            api_key_env="FIXTURE_KEY",
        ),
    )
    assert result.status == "failed"
    assert result.diagnostic == "response_format"
    assert result.execution_attempts == 0
    assert result.attempts[0].usage.input_tokens == 2


@pytest.mark.parametrize(
    ("status", "category", "hint"),
    [
        (401, "authentication", "API key"),
        (403, "authentication", "permission"),
        (404, "endpoint_or_model", "request URL"),
        (429, "rate_limit", "rate limit"),
        (400, "capability", "structured"),
        (503, "service", "unavailable"),
    ],
)
def test_provider_failures_have_actionable_safe_diagnostics(
    monkeypatch, status, category, hint
):
    requests = []

    def provider(request):
        requests.append(request)
        return httpx.Response(
            status, text="private-fixture-credential raw provider detail"
        )

    connect_fixture(monkeypatch, provider)
    config = ModelConfig(
        protocol="openai",
        endpoint="https://other.example/chat/completions",
        name="other-model",
        api_key_env="FIXTURE_KEY",
        instruction_mode="tools",
    )
    result = analyze(b"sales\n10\n", "sum", config=config)
    assert result.status == "failed"
    assert result.diagnostic == category
    assert hint in result.error
    assert result.attempts[0].diagnostic == category
    assert result.attempts[0].request.name == "other-model"
    assert result.attempts[0].model_seconds > 0
    assert result.attempts[0].model_seconds <= result.attempts[0].seconds
    assert result.execution_attempts == 0
    assert len(requests) == 1
    assert result.usage is None
    assert "private-fixture-credential" not in result.model_dump_json()
    assert "raw provider detail" not in result.model_dump_json()


@pytest.mark.parametrize(
    "bad_call",
    [
        {"name": "execute_analysis", "arguments": '{"code": 12}'},
        {"name": "execute_analysis", "arguments": '{"code":"pass","extra":true}'},
        {"name": "unconfigured_function", "arguments": '{"code":"pass"}'},
    ],
)
def test_invalid_native_tool_preserves_usage_without_executing(monkeypatch, bad_call):
    connect_fixture(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"type": "function", "function": bad_call}],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 18, "completion_tokens": 7},
            },
        ),
    )
    result = analyze(
        b"sales\n10\n",
        "sum",
        config=ModelConfig(
            protocol="openai",
            endpoint="https://other.example/chat/completions",
            name="other-model",
            api_key_env="FIXTURE_KEY",
            instruction_mode="tools",
        ),
    )
    assert result.status == "failed"
    assert result.diagnostic == "response_format"
    assert result.execution_attempts == 0
    assert result.code == ""
    assert result.usage.input_tokens == 18
    assert result.attempts[0].usage.output_tokens == 7


def test_bad_model_name_is_distinct_from_tools_and_retains_reported_usage(monkeypatch):
    connect_fixture(
        monkeypatch,
        lambda request: httpx.Response(
            400,
            json={
                "error": {
                    "code": "model_not_found",
                    "message": "private-fixture-credential",
                },
                "usage": {"prompt_tokens": 9, "completion_tokens": 0},
            },
        ),
    )
    result = analyze(
        b"sales\n10\n",
        "sum",
        config=ModelConfig(
            protocol="openai",
            endpoint="https://other.example/chat/completions",
            name="wrong-model",
            api_key_env="FIXTURE_KEY",
            instruction_mode="tools",
        ),
    )
    assert result.diagnostic == "endpoint_or_model"
    assert "model name" in result.error
    assert result.attempts[0].request.name == "wrong-model"
    assert result.usage.input_tokens == 9
    assert result.usage.output_tokens == 0  # An explicitly reported zero is valid.
    assert "private-fixture-credential" not in result.model_dump_json()


def test_invalid_environment_limits_return_a_safe_configuration_failure(monkeypatch):
    monkeypatch.setenv("MAX_EXECUTIONS", "private-fixture-credential")
    result = analyze(b"sales\n10\n", "sum")
    assert result.status == "failed"
    assert result.diagnostic == "configuration"
    assert result.attempts == []
    assert "private-fixture-credential" not in result.model_dump_json()


def test_config_switch_from_environment_uses_structured_path_without_native_tools(
    monkeypatch,
):
    observed = []
    limitation = {"status": "insufficient_input", "reason": "Missing cost field"}

    def provider(request):
        observed.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(limitation)}}],
                "usage": {"prompt_tokens": True, "completion_tokens": -1},
            },
        )

    connect_fixture(monkeypatch, provider)
    for name, value in {
        "MODEL_PROTOCOL": "openai",
        "MODEL_ENDPOINT": "https://text.example/chat/completions",
        "MODEL_NAME": "text-only-model",
        "MODEL_API_KEY_ENV": "FIXTURE_KEY",
        "MODEL_INSTRUCTION_MODE": "structured",
        "MODEL_TEMPERATURE": "",
        "MODEL_DISABLE_THINKING": "false",
    }.items():
        monkeypatch.setenv(name, value)
    result = analyze(b"sales\n10\n", "profit")
    assert result.status == "insufficient_input", result.error
    assert result.attempts[0].request.name == "text-only-model"
    assert result.usage.input_tokens is None
    assert result.usage.output_tokens is None
    assert "tools" not in json.loads(observed[0].content)
    assert "temperature" not in json.loads(observed[0].content)


def test_invalid_model_address_from_environment_has_a_safe_address_diagnostic(
    monkeypatch,
):
    connect_fixture(monkeypatch, lambda request: httpx.Response(200))
    for name, value in {
        "MODEL_ENDPOINT": "https://service.example:private-port-detail/request",
        "MODEL_NAME": "configured-model",
        "MODEL_API_KEY_ENV": "FIXTURE_KEY",
    }.items():
        monkeypatch.setenv(name, value)
    result = analyze(b"sales\n10\n", "sum")
    assert result.status == "failed"
    assert result.diagnostic in {"configuration", "endpoint_or_model"}
    assert "Docker" not in result.error
    assert result.execution_attempts == 0
    assert "private-port-detail" not in result.model_dump_json()


def test_http_client_invalid_url_is_a_safe_recorded_address_failure(monkeypatch):
    def invalid_url(**kwargs):
        raise httpx.InvalidURL("private-fixture-credential malformed address detail")

    monkeypatch.setenv("FIXTURE_KEY", "private-fixture-credential")
    monkeypatch.setattr(httpx, "AsyncClient", invalid_url)
    result = analyze(
        b"sales\n10\n",
        "sum",
        config=ModelConfig(
            endpoint="https://service.example/messages",
            name="configured-model",
            api_key_env="FIXTURE_KEY",
        ),
    )
    assert result.status == "failed"
    assert result.diagnostic == "endpoint_or_model"
    assert "request URL" in result.error
    assert result.attempts[0].request.name == "configured-model"
    assert result.execution_attempts == 0
    assert "private-fixture-credential" not in result.model_dump_json()


def test_escaped_credential_in_structured_instruction_cannot_reach_execution(
    monkeypatch,
):
    # The first character is JSON-escaped; decoding must not expose the credential.
    instruction = '{"code":"raise ValueError(\\"\\u0070rivate-fixture-credential\\")"}'
    connect_fixture(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": instruction}],
                "usage": {"input_tokens": 3, "output_tokens": 8},
            },
        ),
    )
    result = analyze(
        b"sales\n10\n",
        "sum",
        config=ModelConfig(
            endpoint="https://service.example/messages",
            name="configured-model",
            api_key_env="FIXTURE_KEY",
        ),
    )
    assert result.status == "failed"
    assert result.diagnostic == "credential_material"
    assert result.execution_attempts == 0
    assert result.usage.output_tokens == 8
    assert "private-fixture-credential" not in result.model_dump_json()
