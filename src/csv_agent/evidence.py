"""Audit persisted analysis evidence without executing code or calling a model."""

import ast
import json

from csv_agent.analysis import (
    AnalysisResult,
    InspectionResult,
    Strategy,
    collect_computed_result,
)
from csv_agent.answers import AnswerFormat, comparison_fields
from csv_agent.candidate_selection import compare_fields, validate_selection
from csv_agent.config import ModelConfig, RunLimits
from csv_agent.model import Usage


def verify_analysis_evidence(
    analysis: AnalysisResult,
    *,
    config: ModelConfig,
    limits: RunLimits,
    strategy: Strategy,
    answer_format: str,
    strict_strategy: bool = False,
    question: str | None = None,
) -> None:
    """Verify source links, calculations, review gates and saved request accounting.

    This checks consistency of saved evidence, not provider authenticity or the
    factual truth of a model's review. Correctness is independently host-scored.
    """
    error = "Saved analysis evidence is inconsistent"
    attempts = analysis.attempts
    executions = [a for a in attempts if a.execution is not None]
    if (
        analysis.status == "running"
        or not analysis.run_id
        or [a.number for a in attempts] != list(range(1, len(attempts) + 1))
        or len(attempts) > limits.model_call_budget
        or analysis.execution_attempts != len(executions)
        or len(executions) > limits.max_executions
        or any(
            a.request is None
            or a.request.timeout_seconds > config.timeout_seconds
            or a.request.model_copy(update={"timeout_seconds": config.timeout_seconds})
            != config
            or a.action == "replay"
            for a in attempts
        )
        or analysis.seconds < 0
    ):
        raise ValueError(error)
    for saved_request in attempts:
        if question is not None and (
            saved_request.prompt is None
            or json.loads(saved_request.prompt).get("question") != question
        ):
            raise ValueError("Saved request belongs to a different public question")
    usage = None
    if attempts and all(a.usage is not None for a in attempts):
        totals = {}
        for field in Usage.model_fields:
            values = [getattr(a.usage, field) for a in attempts]
            totals[field] = sum(values) if all(v is not None for v in values) else None
        usage = Usage(**totals)
    if analysis.usage != usage:
        raise ValueError("Saved request usage disagrees with task usage")
    if strict_strategy:
        allowed = {"legacy", "insufficient_input"}
        if strategy in {"langgraph_explore", "langgraph_verify", "langgraph_resolve"}:
            allowed |= {
                "inspect",
                "candidate",
                "unresolved",
                "select" if strategy == "langgraph_resolve" else "submit",
            }
        if strategy in {"langgraph_verify", "langgraph_resolve"}:
            allowed |= {"review", "recompute"}
        if any(a.error is None and a.action not in allowed for a in attempts):
            raise ValueError("Saved action violates the frozen strategy")
    by_number = {a.number: a for a in attempts}
    candidates = {c.number: c for c in analysis.candidates}
    observations = {o.number: o for o in analysis.observations}
    if (
        len(candidates) != len(analysis.candidates)
        or len(observations) != len(analysis.observations)
        or candidates.keys() & observations.keys()
    ):
        raise ValueError(error)
    contract = AnswerFormat.from_public_format(answer_format)
    for candidate in analysis.candidates:
        attempt = by_number.get(candidate.number)
        if (
            attempt is None
            or attempt.action not in {"candidate", "recompute"}
            or attempt.code != candidate.code
            or attempt.execution is None
            or attempt.execution.status != "completed"
            or attempt.error is not None
        ):
            raise ValueError("Candidate does not reference a successful calculation")
        computed, charts = collect_computed_result(
            attempt.execution.output, limits, contract
        )
        if candidate.result != computed or candidate.charts != charts:
            raise ValueError("Candidate differs from saved execution evidence")
        if (
            candidate.reason != attempt.reason
            or candidate.method != attempt.method
            or candidate.recomputed_from
            != (attempt.candidate if attempt.action == "recompute" else None)
        ):
            raise ValueError("Candidate recomputation source disagrees with request")
        if candidate.recomputed_from is not None:
            previous = candidates.get(candidate.recomputed_from)
            if (
                previous is None
                or previous.number >= candidate.number
                or ast.dump(ast.parse(previous.code))
                == ast.dump(ast.parse(candidate.code))
            ):
                raise ValueError("Candidate is not an independent recomputation")
    for observation in analysis.observations:
        attempt = by_number.get(observation.number)
        if (
            attempt is None
            or attempt.action != "inspect"
            or attempt.code != observation.code
            or attempt.execution is None
            or attempt.execution.status != "completed"
            or attempt.error is not None
        ):
            raise ValueError("Observation does not reference a successful inspection")
        envelope = json.loads(attempt.execution.output)
        computed_observation = InspectionResult.model_validate(envelope["result"])
        if (
            envelope.get("error")
            or envelope.get("artifacts")
            or observation.values != computed_observation.observations
        ):
            raise ValueError("Observation differs from saved execution evidence")
    for attempt in executions:
        if (
            attempt.error is None
            and attempt.action in {"candidate", "recompute", "inspect"}
            and attempt.number not in candidates
            and attempt.number not in observations
        ):
            raise ValueError("Successful execution is missing its saved evidence")
    reviews = set()
    flagged = set()
    for review in analysis.reviews:
        attempt = by_number.get(review.number)
        if (
            review.number in reviews
            or attempt is None
            or attempt.action != "review"
            or attempt.candidate != review.candidate
            or attempt.error is not None
            or review.candidate not in candidates
            or review.candidate >= review.number
        ):
            raise ValueError("Review source disagrees with saved evidence")
        if review.outcome == "passed" and review.candidate in flagged:
            raise ValueError("A flagged candidate needs a reviewed replacement")
        if review.outcome != "passed":
            flagged.add(review.candidate)
        reviews.add(review.number)
    if strategy == "langgraph_resolve":
        if (
            len(
                [
                    a
                    for a in executions
                    if a.action in {"candidate", "recompute", "legacy"}
                ]
            )
            > 3
        ):
            raise ValueError("Saved analysis exceeds three solution paths")
        comparisons = [
            compare_fields(
                first.number,
                comparison_fields(
                    first.result.answer, first.result.answer_fields, contract
                ),
                second.number,
                comparison_fields(
                    second.result.answer, second.result.answer_fields, contract
                ),
            )
            for index, second in enumerate(analysis.candidates)
            for first in analysis.candidates[:index]
        ]
        if analysis.comparisons != comparisons:
            raise ValueError("Candidate comparisons disagree with computed fields")
    if analysis.status != "completed":
        if (
            analysis.answer is not None
            or analysis.answer_fields is not None
            or analysis.table
            or analysis.charts
        ):
            raise ValueError("An unfinished analysis cannot publish a candidate")
        if analysis.selected_candidate is None:
            if analysis.selection is not None:
                raise ValueError("Selection is missing its candidate source")
            return
        # A deadline can expire immediately after submission. Keep the already
        # saved choice as evidence, while the timeout publishes no prediction.
        if analysis.status != "timeout" or analysis.diagnostic != "timeout":
            raise ValueError("Unfinished selection is not a terminal deadline failure")
    if not candidates:
        if strategy in {"langgraph_explore", "langgraph_verify", "langgraph_resolve"}:
            raise ValueError("Completed workflow is missing a selected candidate")
        if analysis.execution is None or analysis.execution.status != "completed":
            raise ValueError("Completed analysis is missing its calculation")
        if (
            not executions
            or analysis.execution != executions[-1].execution
            or analysis.code != executions[-1].code
            or executions[-1].error is not None
        ):
            raise ValueError(
                "Final calculation differs from its saved execution attempt"
            )
        computed, charts = collect_computed_result(
            analysis.execution.output, limits, contract
        )
    else:
        selected = candidates.get(analysis.selected_candidate or -1)
        if selected is None:
            raise ValueError("Final answer is missing its selected candidate")
        computed, charts = selected.result, selected.charts
        if (
            analysis.code != selected.code
            or analysis.execution != by_number[selected.number].execution
        ):
            raise ValueError("Final calculation differs from selected candidate")
        final = attempts[-1]
        if (
            final.action not in {"submit", "select"}
            or final.candidate != selected.number
            or (
                final.error is not None
                and (analysis.status != "timeout" or final.error != analysis.error)
            )
        ):
            raise ValueError("Final answer is missing a successful submission")
        needs_review = strategy in {"langgraph_verify", "langgraph_resolve"} or bool(
            analysis.reviews
        )
        candidate_reviews = [
            r for r in analysis.reviews if r.candidate == selected.number
        ]
        if needs_review and (
            not candidate_reviews
            or candidate_reviews[-1].outcome != "passed"
            or candidate_reviews[-1].number >= final.number
        ):
            raise ValueError("Final candidate has not passed review")
        selection = analysis.selection
        if strategy == "langgraph_resolve" and selection is None:
            raise ValueError("Final candidate is missing its selection evidence")
        if selection is not None:
            if (
                selection.number != final.number
                or selection.candidate != selected.number
                or final.action != "select"
            ):
                raise ValueError("Final selection disagrees with submission")
            validate_selection(
                selection,
                analysis.comparisons,
                available_evidence=set(candidates) | set(observations),
                checking_evidence=set(observations)
                | {
                    c.number
                    for c in analysis.candidates
                    if c.recomputed_from is not None
                },
            )
    if analysis.status != "completed":
        return
    if (
        analysis.answer != computed.answer
        or analysis.answer_fields != computed.answer_fields
        or analysis.table != computed.table
        or analysis.charts != charts
    ):
        raise ValueError("Final prediction differs from computed candidate evidence")
