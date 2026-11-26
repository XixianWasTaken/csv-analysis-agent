"""Spark mode capabilities at the existing HTTP and analysis boundaries."""

import json
from pathlib import Path

import httpx
import pytest
from test_model_portability import connect_fixture

from csv_agent.analysis import analyze
from csv_agent.config import ModelConfig, RunLimits


def test_explicit_thinking_from_environment_is_sent_and_recorded(monkeypatch):
    requests = []

    def provider(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "spark-x2.6",
                "content": [
                    {"type": "thinking", "thinking": "private reasoning"},
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"status": "insufficient_input", "reason": "Missing cost"}
                        ),
                    },
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 7, "output_tokens": 11},
            },
        )

    connect_fixture(monkeypatch, provider)
    for name, value in {
        "MODEL_PROTOCOL": "anthropic",
        "MODEL_ENDPOINT": "https://fixture.example/messages",
        "MODEL_NAME": "spark-x2.6",
        "MODEL_API_KEY_ENV": "FIXTURE_KEY",
        "MODEL_DISABLE_THINKING": "false",
        "MODEL_THINKING": "enabled",
        "MODEL_MAX_TOKENS": "8192",
        "MODEL_TIMEOUT_SECONDS": "120",
    }.items():
        monkeypatch.setenv(name, value)
    result = analyze(b"sales\n10\n", "profit")
    assert result.status == "insufficient_input", result.error
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert requests[0]["max_tokens"] == 8192
    attempt = result.attempts[0]
    assert attempt.request.thinking == "enabled"
    assert attempt.request.timeout_seconds == 120
    assert attempt.response.model == "spark-x2.6"
    assert attempt.response.thinking_present is True
    assert attempt.response.stop_reason == "end_turn"
    assert attempt.usage.output_tokens == 11
    assert "private reasoning" not in result.model_dump_json()


@pytest.mark.parametrize(
    "settings",
    [
        {"disable_thinking": True, "thinking": "enabled"},
        {"protocol": "openai", "thinking": "enabled"},
        {"protocol": "openai", "disable_thinking": True},
    ],
)
def test_conflicting_or_incompatible_thinking_is_rejected_before_request(settings):
    with pytest.raises(ValueError, match="thinking"):
        ModelConfig(
            endpoint="https://fixture.example/messages", name="spark-x2.6", **settings
        )


@pytest.mark.parametrize("mode", ["structured", "tools"])
def test_rejected_thinking_preserves_capability_diagnostic_and_usage(monkeypatch, mode):
    requests = []

    def provider(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "error": {"code": "unsupported_parameter", "message": "raw detail"},
                "usage": {"input_tokens": 9, "output_tokens": 0},
            },
        )

    connect_fixture(monkeypatch, provider)
    result = analyze(
        b"sales\n10\n",
        "sum",
        config=ModelConfig(
            endpoint="https://fixture.example/messages",
            name="spark-x2.6",
            api_key_env="FIXTURE_KEY",
            thinking="enabled",
            instruction_mode=mode,
        ),
    )
    assert result.diagnostic == "capability"
    assert result.execution_attempts == 0
    assert len(requests) == 1
    assert result.attempts[0].request.thinking == "enabled"
    assert result.attempts[0].response.http_status == 400
    assert result.usage.output_tokens == 0
    assert "raw detail" not in result.model_dump_json()


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, None),
        ({"disable_thinking": True}, {"type": "disabled"}),
        ({"thinking": "adaptive"}, {"type": "adaptive"}),
    ],
)
def test_omission_disabled_and_adaptive_have_distinct_request_evidence(
    monkeypatch, settings, expected
):
    requests = []

    def provider(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": '{"status":"insufficient_input","reason":"Missing cost"}',
                    }
                ]
            },
        )

    connect_fixture(monkeypatch, provider)
    result = analyze(
        b"sales\n10\n",
        "profit",
        config=ModelConfig(
            endpoint="https://fixture.example/messages",
            name="spark-x2.6",
            api_key_env="FIXTURE_KEY",
            **settings,
        ),
    )
    assert result.status == "insufficient_input", result.error
    assert requests[0].get("thinking") == expected
    assert result.attempts[0].response.thinking_present is False
    assert result.attempts[0].response.model is None
    assert result.usage is None


@pytest.mark.parametrize("mode", ["structured", "tools"])
@pytest.mark.docker
def test_explicit_thinking_completes_inspection_candidate_and_submission(
    monkeypatch, mode
):
    from test_data_exploration import CANDIDATE, CSV, INSPECT

    actions = [
        ("inspect_data", {"action": "inspect", "code": INSPECT}),
        ("compute_candidate", {"action": "candidate", "code": CANDIDATE}),
        ("submit_answer", {"action": "submit", "candidate": 2}),
    ]
    requests = []

    def provider(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["thinking"] == {"type": "enabled"}
        if len(requests) > 1:
            prompt = json.loads(payload["messages"][0]["content"])
            assert prompt["observations"][0]["values"]["missing_sales"] == 1
        name, arguments = actions[len(requests) - 1]
        block = {"type": "text", "text": json.dumps(arguments)}
        if mode == "tools":
            assert name in {tool["name"] for tool in payload["tools"]}
            block = {"type": "tool_use", "name": name, "input": arguments}
        return httpx.Response(
            200,
            json={
                "model": "spark-x2.6",
                "stop_reason": "tool_use" if mode == "tools" else "end_turn",
                "content": [
                    {"type": "thinking", "thinking": "private reasoning"},
                    block,
                ],
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 11,
                    "cache_read_input_tokens": -1,
                },
            },
        )

    connect_fixture(monkeypatch, provider)
    result = analyze(
        CSV,
        "east sales ignoring missing values",
        strategy="langgraph_explore",
        config=ModelConfig(
            endpoint="https://fixture.example/messages",
            name="spark-x2.6",
            api_key_env="FIXTURE_KEY",
            thinking="enabled",
            instruction_mode=mode,
        ),
        limits=RunLimits(max_model_calls=3, max_executions=2),
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert result.execution_attempts == 2
    assert result.usage.input_tokens == 21
    assert result.usage.cache_read_input_tokens is None
    assert [a.response.native_tool for a in result.attempts] == (
        ["inspect_data", "compute_candidate", "submit_answer"]
        if mode == "tools"
        else [None] * 3
    )
    assert all(a.response.thinking_present for a in result.attempts)
    assert "private reasoning" not in result.model_dump_json()


def test_output_limit_keeps_usage_and_does_not_execute_partial_code(monkeypatch):
    connect_fixture(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": '{"code":"pass"}'}],
                "stop_reason": "max_tokens",
                "usage": {"output_tokens": 128},
            },
        ),
    )
    result = analyze(
        b"sales\n10\n",
        "sum",
        config=ModelConfig(
            endpoint="https://fixture.example/messages",
            name="spark-x2.6",
            api_key_env="FIXTURE_KEY",
            max_tokens=128,
            thinking="enabled",
        ),
    )
    assert result.diagnostic == "output_limit"
    assert result.execution_attempts == 0
    assert result.attempts[0].response.stop_reason == "max_tokens"
    assert result.usage.output_tokens == 128


def test_thinking_request_timeout_preserves_actual_budget_and_unknown_usage(
    monkeypatch,
):
    def provider(request):
        raise httpx.ReadTimeout("raw private-fixture-credential")

    connect_fixture(monkeypatch, provider)
    result = analyze(
        b"sales\n10\n",
        "sum",
        config=ModelConfig(
            endpoint="https://fixture.example/messages",
            name="spark-x2.6",
            api_key_env="FIXTURE_KEY",
            thinking="enabled",
            timeout_seconds=120,
        ),
        limits=RunLimits(task_timeout_seconds=0.5),
    )
    assert result.status == "timeout"
    assert result.attempts[0].request.thinking == "enabled"
    assert 0 < result.attempts[0].request.timeout_seconds <= 0.5
    assert result.attempts[0].response is None
    assert result.usage is None
    assert result.execution_attempts == 0
    assert "raw" not in result.model_dump_json()


def test_known_spark_service_does_not_accept_ineffective_temperature():
    with pytest.raises(ValueError, match="temperature"):
        ModelConfig(
            endpoint="https://maas-api.cn-huabei-1.xf-yun.com/anthropic/v1/messages",
            name="spark-x2.6",
            temperature=0.2,
        )


@pytest.mark.parametrize(
    ("settings", "hint"),
    [
        ({"MODEL_TEMPERATURE": "0.2"}, "temperature=1"),
        (
            {"MODEL_DISABLE_THINKING": "true", "MODEL_THINKING": "enabled"},
            "MODEL_DISABLE_THINKING=false",
        ),
        (
            {"MODEL_PROTOCOL": "openai", "MODEL_THINKING": "enabled"},
            "Anthropic Messages",
        ),
    ],
)
def test_unsupported_environment_options_have_safe_hints_without_requests(
    monkeypatch, settings, hint
):
    connect_fixture(
        monkeypatch,
        lambda request: pytest.fail("Rejected configuration must not send HTTP"),
    )
    for name, value in {
        "MODEL_PROTOCOL": "anthropic",
        "MODEL_ENDPOINT": "https://maas-api.cn-huabei-1.xf-yun.com/anthropic/v1/messages",
        "MODEL_NAME": "spark-x2.6",
        "MODEL_API_KEY_ENV": "FIXTURE_KEY",
        "MODEL_TEMPERATURE": "1",
        "MODEL_THINKING": "",
        "MODEL_DISABLE_THINKING": "false",
        **settings,
    }.items():
        monkeypatch.setenv(name, value)
    result = analyze(b"sales\n10\n", "sum")
    assert result.status == "failed"
    assert result.diagnostic == "configuration"
    assert hint in result.error
    assert result.attempts == []
    assert result.execution_attempts == 0


def test_smoke_cli_reports_safe_compatible_thinking_setting(monkeypatch, capsys):
    from csv_agent import __main__ as cli

    connect_fixture(
        monkeypatch,
        lambda request: pytest.fail("Rejected smoke configuration must not send HTTP"),
    )
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setenv("MODEL_ENDPOINT", "https://fixture.example/messages")
    monkeypatch.setenv("MODEL_NAME", "spark-x2.6")
    monkeypatch.setenv("MODEL_API_KEY_ENV", "FIXTURE_KEY")
    monkeypatch.setenv("MODEL_PROTOCOL", "anthropic")
    monkeypatch.setenv("MODEL_DISABLE_THINKING", "true")
    monkeypatch.setenv("MODEL_THINKING", "enabled")
    monkeypatch.setattr("sys.argv", ["csv_agent", "--smoke"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "MODEL_DISABLE_THINKING=false" in capsys.readouterr().err


def test_app_reports_the_compatible_spark_temperature(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    for name, value in {
        "CSV_AGENT_DEMO": "0",
        "MODEL_PROTOCOL": "anthropic",
        "MODEL_ENDPOINT": "https://maas-api.cn-huabei-1.xf-yun.com/anthropic/v1/messages",
        "MODEL_NAME": "spark-x2.6",
        "MODEL_TEMPERATURE": "0.2",
        "MODEL_THINKING": "",
        "MODEL_DISABLE_THINKING": "false",
    }.items():
        monkeypatch.setenv(name, value)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    assert not app.exception
    assert "temperature=1" in app.error[0].value
