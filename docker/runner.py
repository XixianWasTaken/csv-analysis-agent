"""Trusted envelope: generated code writes /output/result.json, never host files."""

import base64
import json
import os
import stat
import subprocess
import sys


def read_output(name, limit):
    if (
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        or name in (".", "..")
    ):
        raise ValueError("Output must be a filename in /output")
    descriptor = os.open("/output/" + name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Output must be a bounded regular file")
        data = handle.read(limit + 1)
        if len(data) > limit:
            raise ValueError("Output size limit exceeded")
        return data


def main():
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import contextlib
import runpy
import sys
import traceback
with open('/output/stdout.txt', 'w') as out, open('/output/stderr.txt', 'w') as err:
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            runpy.run_path('/input/analysis.py', run_name='__main__')
        except BaseException:
            traceback.print_exc()
            sys.exit(1)
""",
        ],
        check=False,
    )
    logs = {
        name: read_output(name + ".txt", 1048576).decode(errors="replace")[:8192]
        for name in ("stdout", "stderr")
    }
    error = (
        (logs["stderr"] or "Python analysis exited unsuccessfully")
        if child.returncode
        else None
    )
    result = None
    artifacts = []
    if error is None:
        try:
            result = json.loads(read_output("result.json", 524288))
            charts = result.get("charts", [])
            if not isinstance(charts, list) or len(charts) > 4:
                raise ValueError("At most four charts are supported")
            total = 0
            names = set()
            for chart in charts:
                name = chart["path"]
                if (
                    not isinstance(name, str)
                    or not name.endswith(".png")
                    or name in names
                ):
                    raise ValueError("Charts require unique PNG filenames")
                names.add(name)
                data = read_output(name, 524288 - total)
                total += len(data)
                artifacts.append(
                    {
                        "path": name,
                        "caption": chart["caption"],
                        "data_base64": base64.b64encode(data).decode("ascii"),
                    }
                )
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            artifacts = []
            error = f"Invalid result.json or chart artifact: {type(exc).__name__}"
    print(
        json.dumps(
            {"result": result, "artifacts": artifacts, "error": error, **logs},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
