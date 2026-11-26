"""Independent solutions are selected through the shared analysis boundary."""

import pytest
from test_candidate_verification import RECOMPUTED_TOTAL, WRONG_TOTAL, review
from test_data_exploration import CSV, SequenceModel

from csv_agent.analysis import analyze
from csv_agent.answers import AnswerField, AnswerFormat
from csv_agent.config import RunLimits
from csv_agent.model import ModelError

WRONG_FIELDS = WRONG_TOTAL.replace(
    "{'answer': str(total)}", "{'answer_fields': {'total': total, 'count': 1}}"
)
CORRECT_FIELDS = RECOMPUTED_TOTAL.replace(
    "{'answer': str(total), 'table': [{'east_sales': total}]}",
    "{'answer_fields': {'count': 1, 'total': total}}",
)
PANDAS_FIELDS = WRONG_FIELDS.replace(
    "df['sales'].sum()", "df.loc[df['region'] == 'east', 'sales'].sum()"
)
READER_FIELDS = """
import csv, json
with open('/input/data.csv') as source:
    records = list(csv.reader(source))[1:]
values = [int(amount) for region, amount in records if region == 'east' and amount]
total = 0
for amount in values:
    total += amount
json.dump({'answer_fields': {'total': total, 'count': len(values)}},
          open('/output/result.json', 'w'))
"""
FORMAT = "@total[number] @count[integer]"
CHECK_SCOPE = """
import csv, json
with open('/input/data.csv') as source:
    rows = list(csv.DictReader(source))
values = [int(row['sales']) for row in rows if row['region'] == 'east' and row['sales']]
json.dump({'observations': {'east_present_values': values, 'east_total': sum(values),
                          'east_present_count': len(values)}},
          open('/output/result.json', 'w'))
"""


def recompute(candidate, code=CORRECT_FIELDS):
    return {
        "action": "recompute",
        "candidate": candidate,
        "code": code,
        "reason": "The question requires a filter and missing-value handling; verify the scope from source rows.",
        "method": "Use csv.DictReader and explicit integer addition instead of pandas aggregation.",
    }


def select(candidate, *, evidence=None, resolutions=None):
    return {
        "action": "select",
        "candidate": candidate,
        "reason": "The reviewed calculation reads the authorized CSV and follows the requested east filter.",
        "evidence": evidence or [candidate],
        "resolutions": resolutions or {},
    }


@pytest.mark.docker
def test_simple_reviewed_solution_finishes_early_with_saved_selection_evidence():
    model = SequenceModel(
        [
            {"action": "candidate", "code": RECOMPUTED_TOTAL},
            review(1),
            select(1),
        ]
    )
    result = analyze(
        CSV,
        "Sum east sales, ignoring missing values",
        strategy="langgraph_resolve",
        model=model,
        limits=RunLimits(max_model_calls=16, max_executions=1),
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert result.selected_candidate == 1
    assert result.selection.candidate == 1
    assert result.selection.evidence == [1]
    assert result.selection.reason == select(1)["reason"]
    assert result.execution_attempts == 1
    assert len(result.attempts) == 3
    assert model.prompts[0]["workflow"] == "resolution"


@pytest.mark.docker
def test_disagreement_compares_every_field_and_requires_further_computation():
    model = SequenceModel(
        [
            {"action": "candidate", "code": WRONG_FIELDS},
            recompute(1),
            review(2),
            select(2),
        ]
    )
    result = analyze(
        CSV,
        "Sum east sales and count present east sales, ignoring missing values",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=model,
        limits=RunLimits(max_model_calls=4, max_executions=3, max_format_repairs=0),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert result.selection is None
    assert "disagreement" in result.error.lower()
    comparison = result.comparisons[0]
    assert comparison.first == 1 and comparison.second == 2
    assert comparison.fields["total"].first == "45"
    assert comparison.fields["total"].second == "10"
    assert not comparison.fields["total"].equal
    assert comparison.fields["count"].equal
    assert model.prompts[-1]["comparisons"][0]["fields"]["total"]["equal"] is False


@pytest.mark.docker
def test_further_data_check_resolves_fields_and_preserves_final_choice():
    decision = select(
        2,
        evidence=[2, 4],
        resolutions={
            "total": {
                "finding": "Only one present east sale (10) contributes; 45 incorrectly includes other regions.",
                "evidence": [4],
            }
        },
    )
    result = analyze(
        CSV,
        "Sum east sales and count present east sales, ignoring missing values",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=SequenceModel(
            [
                {"action": "candidate", "code": WRONG_FIELDS},
                recompute(1),
                review(2),
                {"action": "inspect", "code": CHECK_SCOPE},
                decision,
            ]
        ),
        limits=RunLimits(max_model_calls=16, max_executions=12),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@total[10]\n@count[1]"
    assert result.selection.number == 5
    assert result.selection.resolutions["total"].evidence == [4]
    assert result.observations[0].values == {
        "east_present_values": [10],
        "east_total": 10,
        "east_present_count": 1,
    }
    assert result.candidates[1].reason == recompute(1)["reason"]
    assert result.candidates[1].method == recompute(1)["method"]
    assert result.candidates[1].recomputed_from == 1
    assert result.execution_attempts == 3


@pytest.mark.docker
@pytest.mark.parametrize(
    "extra,error",
    [
        ({"action": "candidate", "code": CORRECT_FIELDS}, "recompute"),
        ({"code": CORRECT_FIELDS}, "recompute"),
        ({"action": "recompute", "candidate": 1, "code": CORRECT_FIELDS}, "reason"),
        ({**recompute(1), "method": " "}, "method"),
        (recompute(1, PANDAS_FIELDS + "\n# same calculation"), "different"),
    ],
)
def test_additional_solutions_require_public_reason_and_different_source_path(
    extra, error
):
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=SequenceModel([{"action": "candidate", "code": PANDAS_FIELDS}, extra]),
        limits=RunLimits(max_model_calls=2, max_executions=12, max_format_repairs=0),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert result.execution_attempts == 1
    assert error in result.error.lower()


@pytest.mark.docker
@pytest.mark.parametrize("initial_failure", [False, True])
def test_at_most_three_solution_executions_including_failed_paths(initial_failure):
    actions = (
        [
            {"action": "candidate", "code": "raise ValueError('bad initial path')"},
            {"action": "candidate", "code": PANDAS_FIELDS},
            recompute(2),
        ]
        if initial_failure
        else [
            {"action": "candidate", "code": PANDAS_FIELDS},
            recompute(1),
            recompute(1, READER_FIELDS),
        ]
    )
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=SequenceModel(
            [
                *actions,
                recompute(3, READER_FIELDS + "\nprint('fourth path')"),
                review(3),
                select(3),
            ]
        ),
        limits=RunLimits(max_model_calls=16, max_executions=12),
    )
    assert result.status == "completed", result.error
    assert result.execution_attempts == 3
    assert len(result.candidates) == (2 if initial_failure else 3)
    assert "three" in result.attempts[3].error.lower()
    assert result.attempts[3].execution is None
    assert result.answer == "@total[10]\n@count[1]"


@pytest.mark.docker
@pytest.mark.parametrize("stop", ["budget", "explicit"])
def test_unresolved_disagreement_retains_limitations_and_real_failure(stop):
    actions = [{"action": "candidate", "code": WRONG_FIELDS}, recompute(1)]
    if stop == "explicit":
        actions.append(
            {
                "action": "unresolved",
                "reason": "The total differs and source scope has not yet been checked.",
            }
        )
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=SequenceModel(actions),
        limits=RunLimits(max_model_calls=len(actions), max_executions=12),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert result.selection is None
    assert len(result.candidates) == 2
    assert any("total" in limitation for limitation in result.limitations)
    assert result.stop_reason == (
        "Model call budget exhausted"
        if stop == "budget"
        else "Unresolved solution disagreements"
    )


@pytest.mark.docker
@pytest.mark.parametrize("with_check", [False, True])
def test_shared_wrong_majority_cannot_replace_data_supported_minority(with_check):
    wrong_reader = READER_FIELDS.replace(
        "if region == 'east' and amount", "if amount"
    ).replace("'count': len(values)", "'count': 1")
    actions = [
        {"action": "candidate", "code": WRONG_FIELDS},
        recompute(1),
        recompute(2, wrong_reader),
        review(2),
    ]
    if with_check:
        actions.append({"action": "inspect", "code": CHECK_SCOPE})
        actions.append(
            select(
                2,
                evidence=[2, 5],
                resolutions={
                    "total": {
                        "finding": "The source check finds only east amount 10; both totals 45 share an all-regions filter error.",
                        "evidence": [5],
                    }
                },
            )
        )
    else:
        actions.append(select(2))
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=SequenceModel(actions),
        limits=RunLimits(
            max_model_calls=len(actions), max_executions=12, max_format_repairs=0
        ),
    )
    assert [item.result.answer_fields["total"] for item in result.candidates] == [
        45,
        10,
        45,
    ]
    assert len(result.comparisons) == 3
    assert result.comparisons[1].fields["total"].equal
    if with_check:
        assert result.status == "completed", result.error
        assert result.answer == "@total[10]\n@count[1]"
        assert result.selected_candidate == 2
        assert result.selection.resolutions["total"].evidence == [5]
    else:
        assert result.status == "failed"
        assert result.answer is None
        assert "disagreement" in result.error.lower()


@pytest.mark.docker
@pytest.mark.parametrize("evidence", [[2], [1], [3], [99]])
def test_field_resolution_cannot_cite_old_review_or_fabricated_execution(evidence):
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=SequenceModel(
            [
                {"action": "candidate", "code": WRONG_FIELDS},
                recompute(1),
                review(2),
                select(
                    2,
                    evidence=[2, *evidence],
                    resolutions={
                        "total": {
                            "finding": "A claim without further source computation",
                            "evidence": evidence,
                        }
                    },
                ),
            ]
        ),
        limits=RunLimits(max_model_calls=4, max_executions=12, max_format_repairs=0),
    )
    assert result.status == "failed"
    assert result.selection is None
    assert result.answer is None
    assert result.execution_attempts == 2


@pytest.mark.docker
def test_consensus_still_requires_review_before_selection():
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=SequenceModel(
            [
                {"action": "candidate", "code": PANDAS_FIELDS},
                recompute(1),
                select(2),
            ]
        ),
        limits=RunLimits(max_model_calls=3, max_executions=12, max_format_repairs=0),
    )
    assert result.status == "failed"
    assert result.comparisons[0].fields["total"].equal
    assert result.answer is None
    assert "passed calculation review" in result.error


@pytest.mark.docker
def test_comparison_uses_public_precision_and_preserves_literal_categories():
    first = PANDAS_FIELDS.replace(
        "'total': total, 'count': 1", "'value': 0.1244, 'category': '01'"
    )
    second = CORRECT_FIELDS.replace(
        "'count': 1, 'total': total", "'category': '1', 'value': 0.12449"
    )
    result = analyze(
        CSV,
        "Report value rounded to two decimal places and the exact category",
        strategy="langgraph_resolve",
        answer_format=AnswerFormat(
            fields={
                "value": AnswerField(kind="number", decimals=2),
                "category": AnswerField(kind="string"),
            }
        ),
        model=SequenceModel(
            [{"action": "candidate", "code": first}, recompute(1, second)]
        ),
        limits=RunLimits(max_model_calls=2, max_executions=12),
    )
    assert result.comparisons[0].fields["value"].equal
    assert not result.comparisons[0].fields["category"].equal
    assert result.candidates[0].result.answer_fields["value"] == 0.1244


@pytest.mark.docker
def test_numerically_equivalent_legacy_field_and_mapping_do_not_create_disagreement():
    second = CORRECT_FIELDS.replace(
        "{'answer_fields': {'count': 1, 'total': total}}",
        "{'answer': '@total[10.0]\\n@count[1]'}",
    )
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=SequenceModel(
            [{"action": "candidate", "code": PANDAS_FIELDS}, recompute(1, second)]
        ),
        limits=RunLimits(max_model_calls=2, max_executions=12),
    )
    assert len(result.candidates) == 2, result.error
    assert result.comparisons[0].fields["total"].equal
    assert result.comparisons[0].fields["count"].equal


@pytest.mark.docker
@pytest.mark.parametrize("budget", ["executions", "timeout"])
def test_further_check_exhaustion_never_publishes_an_unresolved_candidate(budget):
    limits = RunLimits(
        max_model_calls=16,
        max_executions=2 if budget == "executions" else 12,
        timeout_seconds=1 if budget == "timeout" else 20,
    )
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        limits=limits,
        model=SequenceModel(
            [
                {"action": "candidate", "code": WRONG_FIELDS},
                recompute(1),
                {
                    "action": "inspect",
                    "code": "while True: pass" if budget == "timeout" else CHECK_SCOPE,
                },
            ]
        ),
    )
    assert result.status == ("failed" if budget == "executions" else "timeout")
    assert result.answer is None and result.selection is None
    assert len(result.candidates) == 2
    assert result.observations == []
    assert any("total" in text for text in result.limitations)
    assert result.execution_attempts == (2 if budget == "executions" else 3)


@pytest.mark.docker
def test_transport_retry_preserves_disagreements_and_decrements_shared_budget():
    model = SequenceModel(
        [
            {"action": "candidate", "code": WRONG_FIELDS},
            recompute(1),
            ModelError("temporary network failure", retryable=True),
            {"action": "inspect", "code": CHECK_SCOPE},
            review(2),
            select(
                2,
                evidence=[2, 4],
                resolutions={
                    "total": {
                        "finding": "Post-retry CSV check confirms east total 10.",
                        "evidence": [4],
                    }
                },
            ),
        ]
    )
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=model,
        limits=RunLimits(max_model_calls=6, max_executions=3, max_model_retries=1),
    )
    assert result.status == "completed", result.error
    assert result.selected_candidate == 2
    assert result.execution_attempts == 3
    assert len(result.attempts) == 6
    assert result.usage is None
    assert model.prompts[2]["comparisons"] == model.prompts[3]["comparisons"]
    assert model.prompts[3]["remaining_budget"]["model_calls"] == 2
    assert (
        model.prompts[3]["remaining_budget"]["seconds"]
        < model.prompts[2]["remaining_budget"]["seconds"]
    )


@pytest.mark.docker
def test_resolution_defaults_use_the_declared_shared_task_budget(monkeypatch):
    for name in ("MAX_MODEL_CALLS", "MAX_EXECUTIONS", "TASK_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    model = SequenceModel(
        [
            {
                "action": "unresolved",
                "reason": "Additional source evidence is required before a choice.",
            }
        ]
    )
    result = analyze(CSV, "Sum east sales", strategy="langgraph_resolve", model=model)
    assert result.status == "failed"
    assert model.prompts[0]["remaining_budget"]["model_calls"] == 15
    assert model.prompts[0]["remaining_budget"]["executions"] == 12
    assert 1190 < model.prompts[0]["remaining_budget"]["seconds"] <= 1200
    assert model.prompts[0]["remaining_solutions"] == 3


@pytest.mark.docker
def test_numeric_list_components_compare_by_value_at_public_precision():
    first = PANDAS_FIELDS.replace(
        "'total': total, 'count': 1", "'values': [10, 0.1244]"
    )
    second = CORRECT_FIELDS.replace(
        "'count': 1, 'total': total", "'values': [10.0, 0.12449]"
    )
    result = analyze(
        CSV,
        "Report the numeric pair; round only the second item to two decimals",
        strategy="langgraph_resolve",
        answer_format=AnswerFormat(
            fields={
                "values": AnswerField(
                    kind="list",
                    items=[
                        AnswerField(kind="number"),
                        AnswerField(kind="number", decimals=2),
                    ],
                )
            }
        ),
        model=SequenceModel(
            [{"action": "candidate", "code": first}, recompute(1, second)]
        ),
        limits=RunLimits(max_model_calls=2, max_executions=12),
    )
    assert len(result.candidates) == 2, result.error
    assert result.comparisons[0].fields["values"].equal


@pytest.mark.docker
def test_new_field_disagreement_requires_evidence_after_that_field_difference():
    new_count_difference = READER_FIELDS.replace("'count': len(values)", "'count': 2")
    result = analyze(
        CSV,
        "Sum east sales and count present east sales",
        strategy="langgraph_resolve",
        answer_format=FORMAT,
        model=SequenceModel(
            [
                {"action": "candidate", "code": WRONG_FIELDS},
                recompute(1),
                recompute(2, new_count_difference),
                review(2),
                select(
                    2,
                    evidence=[2, 3],
                    resolutions={
                        "total": {
                            "finding": "The third path confirms the east total 10.",
                            "evidence": [3],
                        },
                        "count": {
                            "finding": "The third path first introduces count 2; its correctness is unverified.",
                            "evidence": [3],
                        },
                    },
                ),
            ]
        ),
        limits=RunLimits(max_model_calls=5, max_executions=12, max_format_repairs=0),
    )
    assert result.status == "failed"
    assert result.answer is None and result.selection is None
    assert "count" in result.error
