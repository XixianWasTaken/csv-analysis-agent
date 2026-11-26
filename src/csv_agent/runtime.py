"""Record the source and local execution environment before an experiment starts."""

import json
import platform
import re
import subprocess
from importlib.metadata import distributions
from pathlib import Path

from pydantic import BaseModel

from csv_agent.benchmark import sha256


class RuntimeSnapshot(BaseModel):
    source_sha256: dict[str, str]
    environment: dict[str, str]


def _docker(*args: str) -> str:
    try:
        return subprocess.run(
            ["docker", *args], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise ValueError(
            "Experiment requires the prepared local Docker environment"
        ) from None


def inspect_image(image: str) -> str:
    digest = _docker("image", "inspect", image, "--format", "{{.Id}}")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
        raise ValueError("Docker did not return an immutable image ID")
    return digest


def capture_runtime() -> RuntimeSnapshot:
    repo = Path(__file__).resolve().parents[2]
    files = list((repo / "src/csv_agent").rglob("*.py"))
    files += list((repo / "scripts").glob("*.py"))
    files += [
        repo / name
        for name in (
            "app.py",
            "uv.lock",
            "pyproject.toml",
            "docker/Dockerfile",
            "docker/requirements.in",
            "docker/requirements.lock",
            "docker/runner.py",
        )
    ]
    return RuntimeSnapshot(
        source_sha256={
            str(path.relative_to(repo)): sha256(path.read_bytes())
            for path in sorted(files)
        },
        environment={
            "python": platform.python_version(),
            "os": platform.platform(),
            "docker": _docker(
                "info",
                "--format",
                "{{.ServerVersion}}|{{.OSType}}|{{.Architecture}}|{{.NCPU}}|{{.MemTotal}}",
            ),
            "packages": json.dumps(
                {d.metadata["Name"]: d.version for d in distributions()}, sort_keys=True
            ),
        },
    )
