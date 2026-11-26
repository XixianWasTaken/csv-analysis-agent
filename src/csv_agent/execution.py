import os
import selectors
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from csv_agent.config import RunLimits


class Execution(BaseModel):
    status: Literal["completed", "failed", "timeout"]
    exit_code: int | None = None
    output: str = ""
    error: str | None = None
    seconds: float = 0


def execute(input_dir: Path, run_id: str, limits: RunLimits) -> Execution:
    """Only this task's input is mounted. Output lives in a bounded container tmpfs."""
    name = f"csv-agent-{run_id}"
    started = time.monotonic()
    process = None
    creation_attempted = False
    result = Execution(status="failed", error="Container execution did not start")

    def remaining() -> float:
        seconds = limits.timeout_seconds - (time.monotonic() - started)
        if seconds <= 0:
            raise subprocess.TimeoutExpired("Docker execution", limits.timeout_seconds)
        return seconds

    try:
        command = [
            "docker",
            "create",
            "--name",
            name,
            "--pull=never",
            "--network=none",
            "--read-only",
            "--user=65534:65534",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--memory",
            f"{limits.memory_mb}m",
            "--memory-swap",
            f"{limits.memory_mb}m",
            "--cpus",
            str(limits.cpus),
            "--pids-limit",
            str(limits.pids),
            "--ulimit",
            "nofile=128:128",
            "--ulimit",
            "core=0:0",
            "--log-driver=none",
            "--mount",
            f"type=bind,src={input_dir.resolve()},dst=/input,readonly",
            "--tmpfs",
            f"/output:rw,noexec,nosuid,nodev,size={limits.output_bytes},uid=65534,gid=65534,mode=0700",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=16777216,mode=1777",
            limits.image,
        ]
        # The daemon may create the container before the CLI acknowledges it.
        creation_attempted = True
        subprocess.run(
            command, capture_output=True, timeout=min(15, remaining()), check=True
        )
        remaining()
        process = subprocess.Popen(
            ["docker", "start", "--attach", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        output = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                seconds_left = limits.timeout_seconds - (time.monotonic() - started)
                if seconds_left <= 0:
                    result = Execution(
                        status="timeout", error="Python execution timed out"
                    )
                    break
                for key, _ in selector.select(min(seconds_left, 0.1)):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    output.extend(chunk)
                    if len(output) > limits.output_bytes:
                        result = Execution(
                            status="failed", error="Execution output limit exceeded"
                        )
                        break
                else:
                    continue
                break
            else:
                process.wait(timeout=remaining())
                inspected = subprocess.run(
                    ["docker", "inspect", "--format", "{{.State.ExitCode}}", name],
                    capture_output=True,
                    timeout=min(10, remaining()),
                    text=True,
                    check=False,
                )
                code = (
                    int(inspected.stdout.strip()) if inspected.returncode == 0 else None
                )
                result = Execution(
                    status="completed"
                    if code == 0 and process.returncode == 0
                    else "failed",
                    exit_code=code,
                    output=output.decode(errors="replace"),
                    error=None
                    if code == 0
                    else "Python container exited unsuccessfully",
                )
    except FileNotFoundError:
        creation_attempted = False
        result = Execution(status="failed", error="Docker CLI not installed")
    except subprocess.CalledProcessError:
        result = Execution(
            status="failed",
            error="Docker create failed; check daemon permission and prepared execution image",
        )
    except subprocess.TimeoutExpired:
        result = Execution(status="timeout", error="Docker operation timed out")
    except (OSError, ValueError):
        result = Execution(status="failed", error="Docker execution failed")
    finally:
        # Disconnect the attach client before removing a flooding container.
        # Otherwise daemon output copying can block removal on a full pipe.
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdout:
                process.stdout.close()
        if creation_attempted:
            try:
                cleanup = subprocess.run(
                    ["docker", "rm", "--force", name],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                if cleanup.returncode and b"No such container" not in cleanup.stderr:
                    result = Execution(
                        status="failed",
                        error=f"Container cleanup failed: {name}; remove it manually",
                    )
            except (OSError, subprocess.TimeoutExpired):
                result = Execution(
                    status="failed",
                    error=f"Container cleanup failed: {name}; remove it manually",
                )
    result.seconds = time.monotonic() - started
    return result
