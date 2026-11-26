import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import ValidationError

from csv_agent.analysis import STRATEGY_POLICIES, analyze
from csv_agent.config import ModelConfig, RunLimits, configuration_error_message
from csv_agent.model import HttpModel, Instruction, ModelError


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="CSV analysis task and explicit live model smoke check"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Call the configured model once; consumes credits",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--question", default="What are total sales?")
    parser.add_argument(
        "--strategy", choices=tuple(STRATEGY_POLICIES), default="langgraph_resolve"
    )
    args = parser.parse_args()
    try:
        config = ModelConfig.from_env()
        limits = RunLimits.from_env(
            exploration=args.strategy
            in {"langgraph_explore", "langgraph_verify", "langgraph_resolve"}
        )
    except ValidationError as exc:
        parser.exit(2, configuration_error_message(exc) + "\n")
    except ValueError:
        parser.exit(2, "Invalid model configuration or run limits\n")
    if args.smoke:
        started = time.monotonic()
        record: dict[str, object] = {
            "status": "ok",
            "request": config.model_dump(),
            "usage": None,
            "response": None,
        }
        try:
            reply = HttpModel(config).generate(
                'Return code which writes {"answer":"smoke ok","table":[]} to /output/result.json; do not read CSV.'
            )
            record["request"] = reply.request.model_dump() if reply.request else None
            record["usage"] = reply.usage.model_dump() if reply.usage else None
            record["response"] = reply.response.model_dump() if reply.response else None
            Instruction.model_validate_json(reply.text)
        except (ModelError, ValueError) as exc:
            record["status"] = "failed"
            record["error"] = (
                str(exc)
                if isinstance(exc, ModelError)
                else "Invalid execution instruction"
            )
            record["diagnostic"] = (
                exc.diagnostic if isinstance(exc, ModelError) else "response_format"
            )
            if isinstance(exc, ModelError):
                record["request"] = exc.request.model_dump() if exc.request else None
                record["usage"] = exc.usage.model_dump() if exc.usage else None
                record["response"] = exc.response.model_dump() if exc.response else None
        record["model"] = config.name
        record["model_seconds"] = time.monotonic() - started
        print(json.dumps(record))
        if record["status"] != "ok":
            raise SystemExit(1)
    elif args.csv:
        with args.csv.open("rb") as handle:
            data = handle.read(limits.input_max_bytes + 1)
        result = analyze(
            data, args.question, config=config, limits=limits, strategy=args.strategy
        )
        print(result.model_dump_json(indent=2))
        if result.status != "completed":
            raise SystemExit(1)
    else:
        parser.error("choose --smoke or --csv PATH")


if __name__ == "__main__":
    main()
