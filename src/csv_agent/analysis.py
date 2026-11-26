import ast
import base64
import binascii
import csv
import io
import json
import math
import re
import tempfile
import time
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from csv_agent.answers import (
    AnswerFormat,
    AnswerSubmissionError,
    comparison_fields,
    prepare_answer,
)
from csv_agent.candidate_selection import (
    CandidateComparison,
    CandidateSelection,
    compare_fields,
    first_field_disagreements,
    validate_selection,
)
from csv_agent.config import ModelConfig, RunLimits, configuration_error_message
from csv_agent.csv_input import inspect_csv
from csv_agent.execution import Execution, execute
from csv_agent.model import (
    CandidateChecks,
    ComputeCandidate,
    Diagnostic,
    HttpModel,
    InspectData,
    InsufficientInput,
    Model,
    ModelAction,
    ModelError,
    RecomputeCandidate,
    ResponseEvidence,
    ReviewCandidate,
    SelectCandidate,
    SubmitAnswer,
    UnresolvedAnalysis,
    Usage,
)
from csv_agent.task_budget import TaskBudget


class ChartReference(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(
        min_length=5, max_length=128, pattern=r"^[A-Za-z0-9_-][A-Za-z0-9_.-]*\.png$"
    )
    caption: str = Field(min_length=1, max_length=1000)


class ChartArtifact(ChartReference):
    data_base64: str = Field(max_length=699052)


def _normalize_table_value(value: object, path: str) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError(
                f"{path}: non-finite value; correct the calculation or explain "
                "why it is undefined in a string; do not substitute zero"
            )
    if isinstance(value, list):
        return [
            _normalize_table_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _normalize_table_value(item, f"{path}.{key}")
            for key, item in value.items()
        }
    return value


_ANSWER_TAG = re.compile(r"@[^@\[\]\r\n]+\[([^\[\]\r\n]*)\]")
_ANSWER_LABEL = re.compile(r"(?:^|[;\n])([^:=;\n]+)[:=]\s*([^;\n]+)")


def _has_answer_explanation(value: str) -> bool:
    for part in re.split(r"[;\n]", value):
        prose = _ANSWER_TAG.sub("", part).strip()
        label = _ANSWER_LABEL.fullmatch(prose)
        if label:
            if label[1].strip().casefold() not in {
                "reason",
                "explanation",
                "\u539f\u56e0",
                "\u8bf4\u660e",
            }:
                continue
            prose = label[2].strip()
        if not prose:
            continue
        try:
            Decimal(prose)
        except InvalidOperation:
            return True
    return False


class ComputedResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    answer: str = Field(min_length=1, max_length=65536)
    answer_fields: dict[str, JsonValue] | None = None
    table: list[dict[str, str | int | float | bool | None]] = Field(
        default_factory=list, max_length=1000
    )
    charts: list[ChartReference] = Field(default_factory=list, max_length=4)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError("Return a nonempty computed answer")
        if info.context and info.context.get("validated_answer_fields"):
            return value
        # Inspect scalar values in existing labels and evaluation tags; keep
        # explanations and explicitly quoted category strings readable.
        positions = [
            value,
            *_ANSWER_TAG.findall(value),
            *(match[2] for match in _ANSWER_LABEL.finditer(value)),
        ]
        for position in positions:
            try:
                numeric = Decimal(position)
            except InvalidOperation:
                continue
            if not numeric.is_finite():
                if not _has_answer_explanation(value):
                    raise ValueError(
                        "non-finite answer; correct the calculation or explain why it is undefined"
                    )
                break
        return value

    @field_validator("table", mode="before")
    @classmethod
    def normalize_table(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [
            {
                column: json.dumps(
                    _normalize_table_value(cell, f"table[{index}].{column}"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                if isinstance(cell, (list, dict))
                else _normalize_table_value(cell, f"table[{index}].{column}")
                for column, cell in row.items()
            }
            if isinstance(row, dict)
            else row
            for index, row in enumerate(value)
        ]


TaskStatus = Literal["running", "completed", "failed", "timeout", "insufficient_input"]
Strategy = Literal[
    "single",
    "tool_loop",
    "langgraph",
    "langgraph_no_repair",
    "langgraph_explore",
    "langgraph_verify",
    "langgraph_resolve",
]
STRATEGY_POLICIES = {
    "single": "One model request and at most one isolated code execution; no retries.",
    "tool_loop": "Plain sequential tool loop with the full request/tool feedback history; stop on a valid computed result or a shared budget/runtime limit.",
    "langgraph": "LangGraph conditional repair using the latest failed instruction/code and execution feedback.",
    "langgraph_no_repair": "Same LangGraph path and maximum budgets, with instruction/code feedback repairs disabled; transient transport retries remain enabled.",
    "langgraph_explore": "Inspect actual data and compute candidates in isolation, then explicitly submit a computed candidate; all phases share independent request/execution budgets and one deadline.",
    "langgraph_verify": "Review successful candidates against the question and computation evidence, independently recompute important or uncertain results, and submit only a reviewed candidate within the shared task budget.",
    "langgraph_resolve": "Use at most three independent solutions based on the public question and execution evidence; compare all answer fields, investigate disagreements, and select a reviewed candidate with recorded calculation evidence within one shared budget.",
}


class InspectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    observations: dict[str, JsonValue] = Field(min_length=1)


class DataObservation(BaseModel):
    number: int
    code: str
    values: dict[str, JsonValue]


class ComputedCandidate(BaseModel):
    number: int
    code: str
    result: ComputedResult
    charts: list[ChartArtifact] = Field(default_factory=list)
    recomputed_from: int | None = None
    reason: str | None = None
    method: str | None = None


class CandidateReview(BaseModel):
    number: int
    candidate: int
    outcome: Literal["passed", "needs_revision", "needs_recompute"]
    checks: CandidateChecks


class AnalysisAttempt(BaseModel):
    number: int
    code: str = ""
    execution: Execution | None = None
    usage: Usage | None = None
    error: str | None = None
    seconds: float = 0
    request: ModelConfig | None = None
    response: ResponseEvidence | None = None
    model_seconds: float | None = None
    diagnostic: Diagnostic | None = None
    prompt: str | None = None
    action: (
        Literal[
            "legacy",
            "replay",
            "inspect",
            "candidate",
            "review",
            "recompute",
            "submit",
            "select",
            "unresolved",
            "insufficient_input",
        ]
        | None
    ) = None
    candidate: int | None = None
    reason: str | None = None
    method: str | None = None

    def feedback(self) -> dict[str, object]:
        return {
            "code": self.code,
            "error": self.error,
            "execution": self.execution.model_dump() if self.execution else None,
        }


class AnalysisResult(BaseModel):
    run_id: str
    status: TaskStatus = "failed"
    answer: str | None = None
    answer_fields: dict[str, JsonValue] | None = None
    charts: list[ChartArtifact] = Field(default_factory=list)
    table: list[dict[str, str | int | float | bool | None]] = Field(
        default_factory=list
    )
    profile: dict[str, object] = Field(default_factory=dict)
    code: str = ""
    execution: Execution | None = None
    usage: Usage | None = None
    steps: list[str] = Field(default_factory=list)
    error: str | None = None
    seconds: float = 0
    attempts: list[AnalysisAttempt] = Field(default_factory=list)
    execution_attempts: int = 0
    stop_reason: str | None = None
    diagnostic: Diagnostic | None = None
    observations: list[DataObservation] = Field(default_factory=list)
    candidates: list[ComputedCandidate] = Field(default_factory=list)
    reviews: list[CandidateReview] = Field(default_factory=list)
    selected_candidate: int | None = None
    selection: CandidateSelection | None = None
    comparisons: list[CandidateComparison] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ResultCollectionError(ValueError):
    """A collected result violates the execution envelope's contract."""


def collect_computed_result(
    output: str, limits: RunLimits, answer_contract: AnswerFormat | None
) -> tuple[ComputedResult, list[ChartArtifact]]:
    """Apply the same result and chart contract during execution and offline audit."""
    envelope = json.loads(output)
    if envelope.get("error"):
        raise ResultCollectionError(str(envelope["error"])[:8192])
    computed = ComputedResult.model_validate(
        prepare_answer(envelope["result"], answer_contract),
        context={
            "validated_answer_fields": bool(answer_contract)
            or (
                isinstance(envelope["result"], dict)
                and "answer_fields" in envelope["result"]
            )
        },
    )
    normalized_bytes = len(computed.model_dump_json().encode("utf-8"))
    envelope_bytes = len(
        json.dumps(
            {**envelope, "result": computed.model_dump()},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    if normalized_bytes > 524288 or envelope_bytes > limits.output_bytes:
        raise ResultCollectionError(
            "Normalized result.json or execution envelope size limit exceeded; "
            "return a compact table while preserving the complete answer"
        )
    artifacts = [
        ChartArtifact.model_validate(item) for item in envelope.get("artifacts", [])
    ]
    if [
        ChartReference(path=item.path, caption=item.caption) for item in artifacts
    ] != computed.charts:
        raise ValueError("Chart references do not match collected artifacts")
    if len({item.path for item in artifacts}) != len(artifacts):
        raise ValueError("Duplicate chart references")
    total_bytes = 0
    for artifact in artifacts:
        image_data = base64.b64decode(artifact.data_base64, validate=True)
        total_bytes += len(image_data)
        if total_bytes > 524288:
            raise ValueError("Chart artifact size limit exceeded")
        with Image.open(io.BytesIO(image_data)) as image:
            if image.format != "PNG" or image.width > 4096 or image.height > 4096:
                raise ValueError("Charts must be PNG images up to 4096 by 4096")
            image.verify()
        with Image.open(io.BytesIO(image_data)) as image:
            image.load()
    return computed, artifacts


class TaskState(TypedDict, total=False):
    profile: dict[str, object]
    code: str
    execution: Execution
    action: str


def analyze(
    data: bytes,
    question: str,
    *,
    config: ModelConfig | None = None,
    limits: RunLimits | None = None,
    model: Model | None = None,
    on_step: Callable[[str], None] | None = None,
    on_status: Callable[[TaskStatus], None] | None = None,
    strategy: Strategy = "langgraph",
    answer_format: str | AnswerFormat | None = None,
    historical_code: str | None = None,
    strict_strategy: bool = False,
) -> AnalysisResult:
    """Shared business boundary for UI, CLI and later batch evaluation.

    Bounded feedback repairs share a task deadline. Insufficient input ends
    without execution. No fallback
    ever executes generated Python on the host. Workspaces are task-local.
    """
    started = time.monotonic()
    result = AnalysisResult(run_id=uuid4().hex, status="running")
    if historical_code is not None:
        result.usage = Usage(input_tokens=0, output_tokens=0, cache_read_input_tokens=0)
    budget: TaskBudget
    format_repairs = 0
    model_retries = 0
    retry = False
    retry_prompt: str | None = None
    attempt_started = started
    resolution = strategy == "langgraph_resolve"
    exploration = strategy in {
        "langgraph_explore",
        "langgraph_verify",
        "langgraph_resolve",
    }
    verification = strategy in {"langgraph_verify", "langgraph_resolve"}

    def remaining() -> float:
        return budget.remaining_seconds()

    def can_generate() -> bool:
        if not budget.can_generate:
            result.stop_reason = "Model call budget exhausted"
            if result.status == "running":
                result.status = "failed"
                result.error = result.stop_reason
            return False
        return True

    def finish_attempt() -> None:
        result.attempts[-1].seconds = time.monotonic() - attempt_started
        result.attempts[-1].error = result.error
        result.attempts[-1].diagnostic = result.diagnostic

    def add_usage(usage: Usage | None) -> None:
        if len(result.attempts) == 1:
            result.usage = usage
        elif result.usage is None or usage is None:
            result.usage = None
        else:
            result.usage = Usage(
                **{
                    key: a + b if a is not None and b is not None else None
                    for key, a in result.usage.model_dump().items()
                    for b in [getattr(usage, key)]
                }
            )

    def step(message: str) -> None:
        result.steps.append(message)
        if on_step:
            on_step(message)

    def publish(computed: ComputedResult, charts: list[ChartArtifact]) -> None:
        result.answer = computed.answer
        result.answer_fields = computed.answer_fields
        result.table = computed.table
        result.charts = charts

    def solution_attempts() -> list[AnalysisAttempt]:
        return [
            attempt
            for attempt in result.attempts
            if attempt.execution is not None
            and attempt.action in {"candidate", "recompute", "legacy"}
        ]

    try:
        if on_status:
            on_status("running")
        run_limits = limits or RunLimits.from_env(exploration=exploration)
        if historical_code is not None and strategy != "single":
            raise ValueError("Historical replay requires single execution")
        if strategy not in STRATEGY_POLICIES:
            raise ValueError("Unknown analysis strategy")
        feedback_repairs = strategy != "langgraph_no_repair"
        if strategy == "single":
            run_limits = run_limits.for_single_generation()
        budget = TaskBudget(run_limits, started)
        if not question.strip() or len(question) > 8000:
            raise ValueError("Question must contain 1–8000 characters")
        if not data or len(data) > run_limits.input_max_bytes:
            raise ValueError("CSV is empty or exceeds the configured input limit")
        answer_contract = (
            AnswerFormat.from_public_format(answer_format)
            if isinstance(answer_format, str)
            else answer_format
        )
        with tempfile.TemporaryDirectory(
            prefix=f"csv-agent-{result.run_id}-"
        ) as workspace:
            input_dir = Path(workspace)
            input_dir.chmod(0o755)
            (input_dir / "data.csv").write_bytes(data)
            (input_dir / "data.csv").chmod(0o444)

            def inspect_data(state: TaskState) -> TaskState:
                remaining()
                step("Inspect CSV data")
                result.profile = inspect_csv(data)
                return {"profile": result.profile}

            def generate(state: TaskState) -> TaskState:
                nonlocal \
                    format_repairs, \
                    model_retries, \
                    retry, \
                    retry_prompt, \
                    attempt_started, \
                    exploration, \
                    verification
                remaining()
                feedback = None
                if result.attempts:
                    feedback = result.attempts[-1].feedback()
                retry = False
                result.status = "running"
                result.error = None
                result.diagnostic = None
                actual_model = model or HttpModel(config or ModelConfig.from_env())
                attempt_started = time.monotonic()
                attempt = AnalysisAttempt(number=len(result.attempts) + 1)
                budget.model_calls += 1
                result.attempts.append(attempt)
                step(
                    f"Model request {attempt.number}："
                    + ("Choose next analysis action" if exploration else "Generate or repair analysis code")
                )
                payload_prompt: dict[str, object] = {
                    "question": question,
                    "csv_profile": state["profile"],
                }
                if strict_strategy:
                    allowed_actions = ["code", "insufficient_input"]
                    if exploration:
                        allowed_actions += ["inspect", "candidate", "unresolved"]
                        allowed_actions.append("select" if resolution else "submit")
                    if verification:
                        allowed_actions += ["review", "recompute"]
                    payload_prompt["fixed_strategy"] = strategy
                    payload_prompt["allowed_actions"] = allowed_actions
                if exploration:
                    payload_prompt.update(
                        {
                            "workflow": "resolution"
                            if resolution
                            else "verification"
                            if verification
                            else "exploration",
                            "observations": [
                                item.model_dump() for item in result.observations
                            ],
                            "candidates": [
                                item.model_dump(exclude={"charts"})
                                for item in result.candidates
                            ],
                            "remaining_budget": budget.remaining(),
                            "reviews": [item.model_dump() for item in result.reviews],
                            "comparisons": [
                                item.model_dump() for item in result.comparisons
                            ],
                            "solution_limit": 3 if resolution else None,
                            "remaining_solutions": max(0, 3 - len(solution_attempts()))
                            if resolution
                            else None,
                        }
                    )
                if answer_contract:
                    payload_prompt["answer_format"] = answer_contract.model_dump()
                    if isinstance(answer_format, str):
                        payload_prompt["public_answer_format"] = answer_format
                if feedback and strategy == "tool_loop":
                    payload_prompt["tool_history"] = [
                        previous.feedback() for previous in result.attempts[:-1]
                    ]
                elif feedback:
                    payload_prompt["feedback"] = feedback
                if retry_prompt and exploration:
                    payload_prompt = json.loads(retry_prompt)
                    payload_prompt["remaining_budget"] = budget.remaining()
                    prompt = json.dumps(payload_prompt, ensure_ascii=False)
                else:
                    prompt = retry_prompt or json.dumps(
                        payload_prompt, ensure_ascii=False
                    )
                attempt.prompt = prompt
                retry_prompt = None
                try:
                    try:
                        reply = actual_model.generate(
                            prompt, timeout_seconds=remaining()
                        )
                    finally:
                        attempt.model_seconds = time.monotonic() - attempt_started
                except ModelError as exc:
                    attempt.request = exc.request
                    attempt.response = exc.response
                    attempt.usage = exc.usage
                    attempt.diagnostic = exc.diagnostic
                    result.diagnostic = exc.diagnostic
                    result.status = "timeout" if exc.timed_out else "failed"
                    result.error = str(exc)
                    add_usage(exc.usage)
                    if exc.retryable and model_retries < run_limits.max_model_retries:
                        retry_prompt = prompt
                        model_retries += 1
                        retry = True
                    else:
                        result.stop_reason = (
                            "Model request retry limit reached or non-retryable error"
                        )
                    finish_attempt()
                    remaining()
                    return {}
                attempt.request = reply.request
                attempt.response = reply.response
                attempt.usage = reply.usage
                add_usage(reply.usage)
                remaining()
                try:
                    if not reply.text.strip():
                        raise ValueError("Model returned an empty answer")
                    payload = json.loads(reply.text)
                    instruction = ModelAction.model_validate(payload).root
                    if strict_strategy:
                        action_name = getattr(instruction, "action", "code")
                        if isinstance(instruction, InsufficientInput):
                            action_name = "insufficient_input"
                        if action_name not in allowed_actions:
                            result.error = f"Action {action_name} is disabled by the fixed strategy {strategy}"
                            raise ValueError(result.error)
                    if isinstance(
                        instruction,
                        (
                            InspectData,
                            ComputeCandidate,
                            SubmitAnswer,
                            ReviewCandidate,
                            RecomputeCandidate,
                            SelectCandidate,
                        ),
                    ):
                        exploration = True
                    if isinstance(instruction, (ReviewCandidate, RecomputeCandidate)):
                        verification = True
                    if isinstance(instruction, InsufficientInput):
                        attempt.action = "insufficient_input"
                        result.status = "insufficient_input"
                        result.error = instruction.reason
                        step("Insufficient input; analysis ended")
                        finish_attempt()
                        return {}
                    if isinstance(instruction, UnresolvedAnalysis):
                        attempt.action = "unresolved"
                        result.status = "failed"
                        result.error = instruction.reason
                        result.stop_reason = "Unresolved solution disagreements"
                        result.limitations.append(instruction.reason)
                        step("Unresolved disagreement; preserve limitations and stop")
                        finish_attempt()
                        return {}
                    if isinstance(
                        instruction,
                        (
                            SubmitAnswer,
                            SelectCandidate,
                            ReviewCandidate,
                            RecomputeCandidate,
                        ),
                    ):
                        attempt.action = instruction.action
                        attempt.candidate = instruction.candidate
                        candidate = next(
                            (
                                item
                                for item in result.candidates
                                if item.number == instruction.candidate
                            ),
                            None,
                        )
                        if candidate is None:
                            result.error = "Reference an existing, successfully computed candidate by request number"
                            raise ValueError(result.error)
                        if isinstance(instruction, ReviewCandidate):
                            if instruction.outcome == "passed" and any(
                                item.candidate == instruction.candidate
                                and item.outcome != "passed"
                                for item in result.reviews
                            ):
                                result.error = "A flagged candidate requires a computed and reviewed replacement"
                                raise ValueError(result.error)
                            result.reviews.append(
                                CandidateReview(
                                    number=attempt.number,
                                    candidate=instruction.candidate,
                                    outcome=instruction.outcome,
                                    checks=instruction.checks,
                                )
                            )
                            step(
                                f"Review candidate {instruction.candidate}：{instruction.outcome}"
                            )
                            finish_attempt()
                            return {"action": "review"}
                        if (
                            isinstance(instruction, RecomputeCandidate)
                            and instruction.code.strip() == candidate.code.strip()
                        ):
                            result.error = "Independent recomputation requires a different calculation path, not identical code"
                            raise ValueError(result.error)
                        if isinstance(instruction, (SubmitAnswer, SelectCandidate)):
                            if not resolution and isinstance(
                                instruction, SelectCandidate
                            ):
                                result.error = (
                                    "select requires the langgraph_resolve workflow"
                                )
                                raise ValueError(result.error)
                            if resolution and isinstance(instruction, SubmitAnswer):
                                result.error = "Use select with a factual reason and executed evidence"
                                raise ValueError(result.error)
                            reviews = [
                                item
                                for item in result.reviews
                                if item.candidate == instruction.candidate
                            ]
                            if verification and (
                                not reviews or reviews[-1].outcome != "passed"
                            ):
                                result.error = "Submit only a candidate that has passed calculation review"
                                raise ValueError(result.error)
                            if isinstance(instruction, SelectCandidate):
                                available = {
                                    item.number for item in result.candidates
                                } | {item.number for item in result.observations}
                                try:
                                    validate_selection(
                                        instruction,
                                        result.comparisons,
                                        available_evidence=available,
                                        checking_evidence={
                                            item.number for item in result.observations
                                        }
                                        | {
                                            item.number
                                            for item in result.candidates
                                            if item.recomputed_from is not None
                                        },
                                    )
                                except ValueError as exc:
                                    result.error = str(exc)
                                    raise
                                result.selection = CandidateSelection(
                                    number=attempt.number, **instruction.model_dump()
                                )
                            publish(candidate.result, candidate.charts)
                            result.selected_candidate = candidate.number
                            result.code = candidate.code
                            result.execution = result.attempts[
                                candidate.number - 1
                            ].execution
                            result.status = "completed"
                            result.stop_reason = (
                                f"Selected reviewed candidate {candidate.number} using recorded calculation evidence"
                                if resolution
                                else f"Submitted computed candidate {candidate.number}"
                            )
                            step(f"Final submission: candidate {candidate.number}")
                            finish_attempt()
                            return {}
                    compile(instruction.code, "analysis.py", "exec")
                    if resolution and not isinstance(instruction, InspectData):
                        previous_solutions = solution_attempts()
                        if len(previous_solutions) >= 3:
                            result.error = "At most three solution paths may execute, including failed paths; inspect or select existing evidence instead"
                            raise ValueError(result.error)
                        if result.candidates:
                            if not isinstance(instruction, RecomputeCandidate):
                                result.error = "Additional solutions must use recompute and reference an existing candidate"
                                raise ValueError(result.error)
                            if not instruction.reason or not instruction.reason.strip():
                                result.error = "Additional solutions require a public question or execution evidence reason"
                                raise ValueError(result.error)
                            if not instruction.method or not instruction.method.strip():
                                result.error = "Additional solutions require a different calculation method description"
                                raise ValueError(result.error)
                        signature = ast.dump(ast.parse(instruction.code))
                        if any(
                            ast.dump(ast.parse(previous.code)) == signature
                            for previous in previous_solutions
                        ):
                            result.error = "Independent solutions require different calculation paths; comments or whitespace do not change the source computation"
                            raise ValueError(result.error)
                except (ValidationError, SyntaxError, ValueError):
                    result.status = "failed"
                    result.error = (
                        "Model returned an empty answer"
                        if not reply.text.strip()
                        else result.error
                        or (
                            "Model instruction must be valid JSON: inspect/candidate/recompute with Python code, review with all six factual checks, select with an existing candidate, reason, evidence and field resolutions, submit with a candidate, or unresolved/insufficient_input with a reason"
                            if exploration
                            else "Model instruction must be valid JSON with one nonempty Python code field"
                        )
                    )
                    result.diagnostic = attempt.diagnostic = "response_format"
                    if (
                        feedback_repairs
                        and format_repairs < run_limits.max_format_repairs
                    ):
                        format_repairs += 1
                        retry = True
                    else:
                        result.stop_reason = (
                            "Feedback repairs disabled"
                            if not feedback_repairs
                            else "Model instruction repair limit reached"
                        )
                    finish_attempt()
                    return {}
                result.code = instruction.code
                attempt.code = instruction.code
                if isinstance(instruction, RecomputeCandidate):
                    attempt.reason = instruction.reason
                    attempt.method = instruction.method
                attempt.action = (
                    "inspect"
                    if isinstance(instruction, InspectData)
                    else "recompute"
                    if isinstance(instruction, RecomputeCandidate)
                    else "candidate"
                    if isinstance(instruction, ComputeCandidate) or exploration
                    else "legacy"
                )
                return {"code": instruction.code, "action": attempt.action}

            def run(state: TaskState) -> TaskState:
                remaining()
                budget.executions += 1
                result.execution_attempts = budget.executions
                step(
                    f"Execution attempt {result.execution_attempts}/{run_limits.max_executions}: Docker without network"
                )
                code_path = input_dir / "analysis.py"
                if code_path.exists():
                    code_path.chmod(0o644)
                code_path.write_text(state["code"], encoding="utf-8")
                code_path.chmod(0o444)
                result.execution = execute(
                    input_dir,
                    result.run_id,
                    run_limits.model_copy(
                        update={
                            "timeout_seconds": min(
                                run_limits.timeout_seconds, remaining()
                            )
                        }
                    ),
                )
                result.attempts[-1].execution = result.execution
                return {"execution": result.execution}

            def collect(state: TaskState) -> TaskState:
                step("Collect results and execution evidence")
                execution = state["execution"]
                result.status = execution.status
                if execution.status != "completed":
                    result.error = execution.error
                    return {}
                try:
                    envelope = json.loads(execution.output)
                    if envelope.get("error"):
                        result.status = "failed"
                        result.error = str(envelope["error"])[:8192]
                        return {}
                    if state["action"] == "inspect":
                        observation = InspectionResult.model_validate(
                            envelope["result"]
                        )
                        if len(observation.model_dump_json().encode("utf-8")) > 65536:
                            result.error = "Inspection observations exceed 64 KiB; return compact factual statistics"
                            raise ValueError(result.error)
                        if envelope.get("artifacts"):
                            result.error = "Inspection cannot submit chart artifacts; compute charts in a candidate"
                            raise ValueError(result.error)
                        result.observations.append(
                            DataObservation(
                                number=result.attempts[-1].number,
                                code=state["code"],
                                values=observation.observations,
                            )
                        )
                        result.status = "running"
                        step(f"Data exploration {result.attempts[-1].number} recorded; continue solving")
                        return {}
                    computed, artifacts = collect_computed_result(
                        execution.output, run_limits, answer_contract
                    )
                except ResultCollectionError as exc:
                    result.status = "failed"
                    result.error = str(exc)
                    return {}
                except AnswerSubmissionError as exc:
                    result.status = "failed"
                    result.error = f"Invalid result.json at {exc}"[:8192]
                    return {}
                except ValidationError as exc:
                    issue = exc.errors(include_input=False, include_url=False)[0]
                    location = ".".join(str(part) for part in issue["loc"])
                    result.status = "failed"
                    result.error = (
                        f"Invalid result.json at {location}: {issue['msg']}"
                    )[:8192]
                    return {}
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    AttributeError,
                    OSError,
                    binascii.Error,
                    Image.DecompressionBombError,
                ):
                    result.status = "failed"
                    result.error = (
                        result.error
                        or "Execution did not produce a valid result.json with a nonempty answer or readable chart artifacts"
                    )
                    return {}
                if state["action"] in {"candidate", "recompute"}:
                    if resolution:
                        for previous in result.candidates:
                            result.comparisons.append(
                                compare_fields(
                                    previous.number,
                                    comparison_fields(
                                        previous.result.answer,
                                        previous.result.answer_fields,
                                        answer_contract,
                                    ),
                                    result.attempts[-1].number,
                                    comparison_fields(
                                        computed.answer,
                                        computed.answer_fields,
                                        answer_contract,
                                    ),
                                )
                            )
                    result.candidates.append(
                        ComputedCandidate(
                            number=result.attempts[-1].number,
                            code=state["code"],
                            result=computed,
                            charts=artifacts,
                            recomputed_from=(
                                result.attempts[-1].candidate
                                if state["action"] == "recompute"
                                else None
                            ),
                            reason=result.attempts[-1].reason,
                            method=result.attempts[-1].method,
                        )
                    )
                    result.status = "running"
                    step(f"Candidate {result.attempts[-1].number} recorded; awaiting final submission")
                    return {}
                publish(computed, artifacts)
                return {}

            def after_generate(state: TaskState) -> str:
                remaining()
                if retry:
                    return "generate" if can_generate() else END
                if result.status == "running" and state.get("action") == "review":
                    return "generate" if can_generate() else END
                if result.status == "running" and not budget.can_execute:
                    result.status = "failed"
                    result.stop_reason = result.error = (
                        "Analysis code execution limit reached"
                    )
                    finish_attempt()
                    return END
                return "execute" if result.status == "running" else END

            def after_collect(state: TaskState) -> str:
                finish_attempt()
                remaining()
                if result.status == "completed":
                    return END
                if result.status == "running":
                    return "generate" if can_generate() else END
                # Runtime/cleanup failures and execution timeouts stop immediately.
                # Only code failures with a successfully collected envelope are repaired.
                if state["execution"].status != "completed":
                    result.stop_reason = (
                        "Execution stopped; no further execution retries"
                    )
                    return END
                if not feedback_repairs:
                    result.stop_reason = "Feedback repairs disabled"
                    return END
                if not budget.can_execute:
                    if exploration and result.candidates:
                        return "generate" if can_generate() else END
                    result.stop_reason = "Analysis code execution limit reached"
                    return END
                return "generate" if can_generate() else END

            if historical_code is not None:
                # The same input, sandbox and collector as a fresh analysis;
                # no generation, feedback repair or provider configuration.
                state = inspect_data({})
                if not historical_code.strip():
                    raise ValueError("Historical code must be nonempty")
                result.code = historical_code
                compile(historical_code, "analysis.py", "exec")
                attempt_started = time.monotonic()
                result.attempts.append(
                    AnalysisAttempt(number=1, code=historical_code, action="replay")
                )
                state.update({"code": historical_code, "action": "replay"})
                step("Replay original first code without model calls or repairs")
                state.update(run(state))
                collect(state)
                finish_attempt()
                remaining()
                result.stop_reason = "Historical first-code replay finished"
                return result

            if strategy in {"single", "tool_loop"}:
                state = inspect_data({})
                while True:
                    state.update(generate(state))
                    next_action = after_generate(state)
                    if next_action == "generate":
                        continue
                    if next_action == END:
                        break
                    state.update(run(state))
                    collect(state)
                    if after_collect(state) == END:
                        break
                return result

            graph = StateGraph(TaskState)
            graph.add_node("inspect", inspect_data)
            graph.add_node("generate", generate)
            graph.add_node("execute", run)
            graph.add_node("collect", collect)
            graph.add_edge(START, "inspect")
            graph.add_edge("inspect", "generate")
            graph.add_conditional_edges(
                "generate",
                after_generate,
                {END: END, "execute": "execute", "generate": "generate"},
            )
            graph.add_edge("execute", "collect")
            graph.add_conditional_edges(
                "collect", after_collect, {END: END, "generate": "generate"}
            )
            graph.compile().invoke(
                {},
                {
                    "recursion_limit": 2 * run_limits.model_call_budget
                    + 2 * run_limits.max_executions
                    + 10
                },
            )
    except TimeoutError:
        result.status = "timeout"
        result.answer = None
        result.answer_fields = None
        result.table = []
        result.charts = []
        result.error = "Analysis task timed out"
        result.stop_reason = "Task time budget exhausted"
        result.diagnostic = "timeout"
    except ModelError as exc:
        result.status = "timeout" if exc.timed_out else "failed"
        result.error = str(exc)
        result.diagnostic = exc.diagnostic
    except ValidationError as exc:
        result.status = "failed"
        result.diagnostic = "configuration"
        result.error = configuration_error_message(exc)
    except UnicodeDecodeError:
        result.status = "failed"
        result.error = "CSV must be encoded as UTF-8 (BOM is supported)"
    except csv.Error as exc:
        result.status = "failed"
        result.error = f"Invalid CSV: {exc}"
    except (ValueError, StopIteration) as exc:
        result.status = "failed"
        result.error = str(exc) or "Invalid CSV data"
    except Exception:  # noqa: BLE001 - public task boundary must return a safe failure, never a raw traceback.
        result.status = "failed"
        result.error = (
            "Analysis task failed unexpectedly; check local runtime and Docker access"
        )
    finally:
        if resolution and result.status != "completed":
            differing_fields = sorted(first_field_disagreements(result.comparisons))
            if differing_fields:
                result.limitations.append(
                    "Unresolved candidate field disagreements: "
                    + ", ".join(differing_fields)
                )
            if result.candidates:
                result.limitations.append(
                    "No candidate was finally selected with sufficient review and calculation evidence"
                )
        if result.attempts and result.attempts[-1].seconds == 0:
            finish_attempt()
        result.seconds = time.monotonic() - started
        if on_status:
            on_status(result.status)
    return result
