"""Audit public predictions and arithmetic; no model requests or Docker needed."""

import argparse
import json
from pathlib import Path

from csv_agent._vendor import dabench
from csv_agent.analysis import AnalysisResult
from csv_agent.benchmark import load_manifest, sha256
from csv_agent.evaluation import EvaluationRecord, EvaluationSummary, score_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("benchmarks/dabench/manifest.json")
    )
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest)
        metadata = json.loads((args.report / "metadata.json").read_bytes())
        if (
            metadata["manifest_sha256"] != sha256(args.manifest.read_bytes())
            or metadata["dataset"] != manifest.dataset.model_dump()
            or metadata["scorer"] != manifest.scorer.model_dump()
            or metadata["splits"] != manifest.splits
            or sha256(Path(dabench.__file__).read_bytes())
            != manifest.scorer.vendored_sha256
        ):
            raise ValueError("Publication or scorer does not match locked manifest")
        rows = [
            json.loads(line)
            for line in (args.report / "paired-results.jsonl").read_bytes().splitlines()
            if line.strip()
        ]
        if [row["id"] for row in rows] != manifest.splits["final"]:
            raise ValueError("Publication must retain every final ID exactly once")
        files = {sample.id: sample.file_name for sample in manifest.samples}
        strategies = metadata["strategy_order"]
        if strategies != ["single", "tool_loop", "langgraph"]:
            raise ValueError("Publication must contain the three fixed strategies")
        if set(metadata["final_summaries"]) != set(strategies):
            raise ValueError("Incomplete strategy summaries")
        scores = {}
        for strategy in strategies:
            records = []
            for row in rows:
                if row["file_name"] != files[row["id"]] or set(row["results"]) != set(
                    strategies
                ):
                    raise ValueError(
                        "Paired files or strategies disagree with manifest"
                    )
                result = row["results"][strategy]
                if result["model_calls"] != len(result["attempts"]):
                    raise ValueError(
                        "Published model call count disagrees with attempts"
                    )
                analysis = AnalysisResult.model_validate(
                    {**result, "run_id": f"public-{row['id']}"}
                )
                records.append(
                    EvaluationRecord(
                        id=row["id"],
                        file_name=row["file_name"],
                        analysis=analysis,
                        response=result["answer"],
                        correctness=result["correctness"],
                        failure=result["failure"],
                    )
                )
            saved = [record.model_copy(deep=True) for record in records]
            summary = score_records(args.data, manifest, records)
            if records != saved or summary != EvaluationSummary.model_validate(
                metadata["final_summaries"][strategy]
            ):
                raise ValueError(
                    "Official score or resource sums disagree with publication"
                )
            scores[strategy] = {
                "total": summary.total,
                "correct_questions": summary.correct_questions,
                "total_subquestions": summary.total_subquestions,
                "correct_subquestions": summary.correct_subquestions,
                "question_accuracy": summary.question_accuracy,
                "subquestion_accuracy": summary.subquestion_accuracy,
            }
        print(json.dumps({"status": "public_scores_verified", "scores": scores}))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Public audit failed: {error}\n")


if __name__ == "__main__":
    main()
