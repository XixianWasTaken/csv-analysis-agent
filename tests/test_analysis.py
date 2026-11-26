import json

import pytest

from csv_agent.analysis import analyze
from csv_agent.config import ModelConfig, RunLimits
from csv_agent.model import ModelReply


class FixedModel:
    def __init__(self, code: str):
        self.code = code

    def generate(self, prompt: str, *, timeout_seconds: float) -> ModelReply:
        return ModelReply(text=json.dumps({"code": self.code}), usage=None)


AGGREGATE = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
total = int(df['sales'].sum())
with open('/output/result.json', 'w') as f:
    json.dump({'answer': f'Total sales {total}', 'table': [{'total': total}]}, f)
"""


@pytest.mark.docker
def test_known_aggregate_is_computed_in_isolation():
    result = analyze(
        b"region,sales\neast,10\nwest,20\neast,15\n",
        "What are total sales?",
        model=FixedModel(AGGREGATE),
    )
    assert result.status == "completed", result.error
    assert result.answer == "Total sales 45"
    assert result.table == [{"total": 45}]
    assert result.code == AGGREGATE
    assert result.execution is not None and result.execution.exit_code == 0
    assert result.usage is None


@pytest.mark.docker
def test_generated_code_cannot_read_host_secrets_or_write_input_or_access_network(
    monkeypatch, tmp_path
):
    secret = tmp_path / "host-secret.txt"
    secret.write_text("host-only-sentinel")
    monkeypatch.setenv("SPARK_API_KEY", "host-only-credential")
    code = """
import os, json, socket
import scipy, sklearn, statsmodels
from pathlib import Path
checks = {}
checks['credential_absent'] = 'SPARK_API_KEY' not in os.environ
checks['socket_absent'] = not Path('/var/run/docker.sock').exists()
checks['unrelated_file_absent'] = not Path(HOST_SECRET).exists()
checks['non_root'] = os.getuid() != 0
checks['capabilities_dropped'] = 'CapEff:\\t0000000000000000' in Path('/proc/self/status').read_text()
try:
    Path('/input/data.csv').write_text('tampered')
    checks['input_read_only'] = False
except OSError:
    checks['input_read_only'] = True
try:
    socket.create_connection(('1.1.1.1', 443), timeout=0.5).close()
    checks['network_denied'] = False
except OSError:
    checks['network_denied'] = True
json.dump({'answer': 'isolation checked', 'table': [checks]}, open('/output/result.json', 'w'))
""".replace("HOST_SECRET", repr(str(secret)))
    result = analyze(b"sales\n10\n", "Check isolation", model=FixedModel(code))
    assert result.status == "completed", result.error
    assert result.table == [
        {
            "credential_absent": True,
            "socket_absent": True,
            "unrelated_file_absent": True,
            "non_root": True,
            "capabilities_dropped": True,
            "input_read_only": True,
            "network_denied": True,
        }
    ]
    assert "host-only-credential" not in result.model_dump_json()


@pytest.mark.docker
def test_execution_timeout_is_reported_and_container_is_removed():
    import subprocess

    result = analyze(
        b"sales\n10\n",
        "Run slowly",
        model=FixedModel("import scipy, sklearn, statsmodels\nwhile True: pass"),
        limits=RunLimits(timeout_seconds=2),
    )
    assert result.status == "timeout", result.error
    remaining = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=csv-agent-{result.run_id}"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert remaining.stdout.strip() == ""
    assert result.answer is None


@pytest.mark.docker
def test_output_disk_and_stdout_floods_are_bounded():
    result = analyze(
        b"sales\n10\n",
        "Produce too much output",
        model=FixedModel("import os\nwhile True: os.write(1, b'x' * 8192)"),
        limits=RunLimits(output_bytes=16384, timeout_seconds=5),
    )
    assert result.status == "failed", result.error
    assert result.error == "Execution output limit exceeded"
    assert result.answer is None


@pytest.mark.parametrize(
    "csv_data", [b"", b"\xff\xfe", b"a,b\n1,2,3", b"a,a\n1,2", b"a,b\n"]
)
def test_invalid_csv_never_becomes_a_success(csv_data):
    result = analyze(csv_data, "sum", model=FixedModel(AGGREGATE))
    assert result.status == "failed"
    assert result.execution is None
    assert result.answer is None


def test_unstructured_model_text_is_rejected_before_execution():
    class InvalidModel:
        def generate(self, prompt, *, timeout_seconds):
            return ModelReply(text="```python\nprint(45)\n```", usage=None)

    result = analyze(b"sales\n10\n", "sum", model=InvalidModel())
    assert result.status == "failed"
    assert result.execution is None
    assert (
        result.error
        == "Model instruction must be valid JSON with one nonempty Python code field"
    )


@pytest.mark.docker
@pytest.mark.parametrize(
    "code",
    [
        'raise RuntimeError("computation failed")',
        "pass",
        "import json; json.dump({'answer':'','table':[]}, open('/output/result.json','w'))",
        "import json; json.dump({'answer':'fabricated','table':[{'x':float('inf')}]}, open('/output/result.json','w'))",
    ],
)
def test_failed_or_missing_computation_does_not_produce_an_answer(code):
    result = analyze(b"sales\n10\n", "sum", model=FixedModel(code))
    assert result.status == "failed", result.model_dump_json()
    assert result.answer is None


def test_input_limit_is_enforced_before_model_or_execution():
    result = analyze(
        b"sales\n10\n",
        "sum",
        model=FixedModel(AGGREGATE),
        limits=RunLimits(input_max_bytes=4),
    )
    assert result.status == "failed"
    assert result.execution is None


@pytest.mark.docker
def test_anthropic_analysis_can_disable_thinking_and_preserve_usage(monkeypatch):
    import httpx

    monkeypatch.setenv("TEST_MODEL_KEY", "test-provider-key")
    original_client = httpx.AsyncClient

    def provider(request):
        payload = json.loads(request.content)
        if (
            payload.get("thinking") != {"type": "disabled"}
            or request.headers.get("x-api-key") != "test-provider-key"
        ):
            return httpx.Response(400)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "thinking", "thinking": "not a public answer"},
                    {"type": "text", "text": json.dumps({"code": AGGREGATE})},
                ],
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 34,
                    "cache_read_input_tokens": -1,
                },
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(provider), **kwargs
        ),
    )
    config = ModelConfig(
        endpoint="https://model.example/v1/messages",
        name="test",
        api_key_env="TEST_MODEL_KEY",
        disable_thinking=True,
    )
    result = analyze(b"region,sales\neast,10\nwest,20\neast,15\n", "sum", config=config)
    assert result.status == "completed", result.error
    assert result.answer == "Total sales 45"
    assert result.usage.model_dump() == {
        "input_tokens": 12,
        "output_tokens": 34,
        "cache_read_input_tokens": None,
    }
    assert "not a public answer" not in result.model_dump_json()


@pytest.mark.docker
def test_lost_docker_create_response_does_not_leave_a_container(monkeypatch):
    import subprocess

    original_run = subprocess.run

    def lost_response(command, **kwargs):
        response = original_run(command, **kwargs)
        if command[:2] == ["docker", "create"] and response.returncode == 0:
            raise subprocess.TimeoutExpired(command, 15)
        return response

    monkeypatch.setattr(subprocess, "run", lost_response)
    result = analyze(b"sales\n10\n", "sum", model=FixedModel(AGGREGATE))
    name = f"csv-agent-{result.run_id}"
    try:
        assert result.status == "timeout"
        remaining = original_run(
            ["docker", "ps", "-aq", "--filter", f"name={name}"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert remaining.stdout.strip() == ""
    finally:
        original_run(
            ["docker", "rm", "--force", name], capture_output=True, check=False
        )
