"""Successful but incorrect calculations are reviewed at the analysis boundary."""

from itertools import pairwise

import pytest
from test_data_exploration import CSV, SequenceModel

from csv_agent.analysis import analyze
from csv_agent.config import RunLimits

WRONG_TOTAL = """
import json, pandas as pd
df = pd.read_csv('/input/data.csv')
total = int(df['sales'].sum())
json.dump({'answer': str(total)}, open('/output/result.json', 'w'))
"""
RECOMPUTED_TOTAL = """
import csv, json
with open('/input/data.csv') as source:
    rows = list(csv.DictReader(source))
total = sum(int(row['sales']) for row in rows
            if row['region'] == 'east' and row['sales'])
json.dump({'answer': str(total), 'table': [{'east_sales': total}]},
          open('/output/result.json', 'w'))
"""
CHECKS = {
    "field_meaning": "sales is the sales amount; region identifies the requested group",
    "filters_and_groups": "Only east rows contribute; west and north are excluded",
    "missing_values": "The empty east sales value is excluded as requested",
    "statistical_definition": "Sum of sales, not a row count or average",
    "algorithm_parameters": "Exact integer addition; no fitted model is needed",
    "precision": "Integer values are retained without rounding",
}


def review(candidate, outcome="passed", checks=None):
    return {
        "action": "review",
        "candidate": candidate,
        "outcome": outcome,
        "checks": checks or CHECKS,
    }


@pytest.mark.docker
def test_successful_wrong_filter_is_reviewed_and_recomputed_from_source():
    model = SequenceModel(
        [
            {"action": "candidate", "code": WRONG_TOTAL},
            review(
                1,
                "needs_revision",
                {
                    **CHECKS,
                    "filters_and_groups": "The code sums all regions (45); the question asks only for east",
                },
            ),
            {"action": "recompute", "candidate": 1, "code": RECOMPUTED_TOTAL},
            review(3),
            {"action": "submit", "candidate": 3},
        ]
    )
    result = analyze(
        CSV,
        "east \u7684Total sales，\u5ffd\u7565\u7f3a\u5931\u503c",
        model=model,
        strategy="langgraph_explore",
        limits=RunLimits(max_model_calls=5, max_executions=2),
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert result.table == [{"east_sales": 10}]
    assert result.candidates[0].result.answer == "45"
    assert result.candidates[1].recomputed_from == 1
    assert result.reviews[0].outcome == "needs_revision"
    assert result.reviews[1].candidate == 3
    assert result.reviews[1].checks.filters_and_groups == CHECKS["filters_and_groups"]
    assert result.selected_candidate == 3
    assert result.execution_attempts == 2
    assert result.attempts[1].execution is None
    assert result.attempts[3].execution is None
    assert model.prompts[2]["reviews"][0]["outcome"] == "needs_revision"
    assert model.prompts[3]["candidates"][1]["recomputed_from"] == 1
    assert model.prompts[4]["reviews"][1]["outcome"] == "passed"
    assert result.code == RECOMPUTED_TOTAL
    assert result.usage.input_tokens == 50


@pytest.mark.docker
def test_verification_strategy_does_not_submit_an_unreviewed_result():
    result = analyze(
        CSV,
        "east sales",
        strategy="langgraph_verify",
        model=SequenceModel(
            [
                {"code": RECOMPUTED_TOTAL},
                {"action": "submit", "candidate": 1},
            ]
        ),
        limits=RunLimits(max_model_calls=2, max_executions=1, max_format_repairs=0),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert "passed calculation review" in result.error
    assert result.candidates[0].result.answer == "10"
    assert result.execution_attempts == 1


@pytest.mark.docker
def test_simple_correct_result_can_be_reviewed_after_execution_budget_is_spent():
    model = SequenceModel(
        [
            {"action": "candidate", "code": RECOMPUTED_TOTAL},
            review(1),
            {"action": "submit", "candidate": 1},
        ]
    )
    result = analyze(
        CSV,
        "east sales ignoring missing values",
        model=model,
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=16, max_executions=1),
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert len(result.attempts) == 3
    assert result.execution_attempts == 1
    assert len(result.candidates) == 1
    assert model.prompts[0]["workflow"] == "verification"
    assert [p["remaining_budget"]["executions"] for p in model.prompts] == [1, 0, 0]


@pytest.mark.docker
def test_many_nonexecuting_reviews_share_the_request_budget_and_one_deadline():
    model = SequenceModel(
        [
            {"action": "candidate", "code": RECOMPUTED_TOTAL},
            *[review(1) for _ in range(19)],
            {"action": "submit", "candidate": 1},
        ]
    )
    result = analyze(
        CSV,
        "east sales",
        model=model,
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=21, max_executions=1),
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert len(result.reviews) == 19
    assert result.execution_attempts == 1
    assert [p["remaining_budget"]["model_calls"] for p in model.prompts] == list(
        range(20, -1, -1)
    )
    deadlines = [p["remaining_budget"]["seconds"] for p in model.prompts]
    assert all(later < earlier for earlier, later in pairwise(deadlines))


@pytest.mark.docker
@pytest.mark.parametrize(
    "alternative,outcome",
    [
        ("greater", "needs_revision"),
        ("two-sided", "needs_recompute"),
    ],
)
def test_important_t_test_is_independently_checked_with_statsmodels(
    alternative, outcome
):
    first = f"""
import json, pandas as pd
from scipy.stats import ttest_1samp
values = pd.read_csv('/input/data.csv')['value']
test = ttest_1samp(values, 0, alternative={alternative!r})
json.dump({{'answer_fields': {{'t': float(test.statistic), 'p': float(test.pvalue)}}}},
          open('/output/result.json', 'w'))
"""
    independent = """
import json, pandas as pd
from statsmodels.stats.weightstats import DescrStatsW
values = pd.read_csv('/input/data.csv')['value']
t, p, degrees = DescrStatsW(values).ttest_mean(0, alternative='two-sided')
json.dump({'answer_fields': {'t': float(t), 'p': float(p)},
           'table': [{'degrees_of_freedom': float(degrees)}]},
          open('/output/result.json', 'w'))
"""
    checks = {
        "field_meaning": "value is the sample being tested against mean zero",
        "filters_and_groups": "All five sample rows are included in one test",
        "missing_values": "All five observations are finite and present",
        "statistical_definition": "Two-sided one-sample t test; sample variance uses n-1; df=4",
        "algorithm_parameters": "statsmodels ttest_mean(value=0, alternative=two-sided) independently checks scipy",
        "precision": "Raw t statistic and p value are preserved without rounding",
    }
    model = SequenceModel(
        [
            {"action": "candidate", "code": first},
            review(
                1,
                outcome,
                {
                    **checks,
                    "statistical_definition": "Check sidedness and independently verify the important p value",
                },
            ),
            {"action": "recompute", "candidate": 1, "code": independent},
            review(3, checks=checks),
            {"action": "submit", "candidate": 3},
        ]
    )
    result = analyze(
        b"value\n1\n2\n3\n4\n5\n",
        "Two-sided one-sample t-test against mean zero; return unrounded t and p",
        model=model,
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=5, max_executions=2),
    )
    assert result.status == "completed", result.error
    assert result.answer_fields == pytest.approx(
        {"t": 4.242640687119285, "p": 0.013235599563682695}, abs=1e-12
    )
    assert result.table == [{"degrees_of_freedom": 4.0}]
    assert result.candidates[0].result.answer_fields["p"] == pytest.approx(
        0.0066177997818413475 if alternative == "greater" else 0.013235599563682695,
        abs=1e-12,
    )
    assert result.candidates[1].recomputed_from == 1


@pytest.mark.docker
def test_machine_learning_recomputation_corrects_the_intercept_default():
    first = """
import json, numpy as np, pandas as pd
df = pd.read_csv('/input/data.csv')
slope = np.linalg.lstsq(df[['x']].to_numpy(), df['y'].to_numpy(), rcond=None)[0][0]
json.dump({'answer_fields': {'slope': float(slope), 'intercept': 0.0}},
          open('/output/result.json', 'w'))
"""
    independent = """
import json, pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv('/input/data.csv')
fit = LinearRegression().fit(df[['x']], df['y'])
json.dump({'answer_fields': {'slope': float(fit.coef_[0]), 'intercept': float(fit.intercept_)}},
          open('/output/result.json', 'w'))
"""
    checks = {
        "field_meaning": "x is the feature and y the target",
        "filters_and_groups": "Fit all five supplied observations",
        "missing_values": "Both columns are complete; no imputation needed",
        "statistical_definition": "Ordinary least squares with an intercept",
        "algorithm_parameters": "LinearRegression() uses fit_intercept=True; no split or stochastic preprocessing requested",
        "precision": "Keep fitted coefficients at their computed precision",
    }
    result = analyze(
        b"x,y\n1,3\n2,5\n3,7\n4,9\n5,11\n",
        "Fit sklearn LinearRegression() with defaults on all rows; return slope and intercept",
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=5, max_executions=2),
        model=SequenceModel(
            [
                {"action": "candidate", "code": first},
                review(
                    1,
                    "needs_revision",
                    {
                        **checks,
                        "algorithm_parameters": "The initial least-squares design omits the required default intercept",
                    },
                ),
                {"action": "recompute", "candidate": 1, "code": independent},
                review(3, checks=checks),
                {"action": "submit", "candidate": 3},
            ]
        ),
    )
    assert result.status == "completed", result.error
    assert result.candidates[0].result.answer_fields == pytest.approx(
        {"slope": 2.2727272727272725, "intercept": 0}
    )
    assert result.answer_fields == pytest.approx(
        {"slope": 2, "intercept": 1}, abs=1e-12
    )
    assert (
        result.reviews[1].checks.algorithm_parameters == checks["algorithm_parameters"]
    )


@pytest.mark.docker
@pytest.mark.parametrize("outcome", ["needs_revision", "needs_recompute"])
@pytest.mark.parametrize("next_action", ["submit", "clear"])
def test_flagged_candidate_cannot_be_submitted_or_cleared_by_assertion(
    outcome, next_action
):
    last = (
        {"action": "submit", "candidate": 1} if next_action == "submit" else review(1)
    )
    result = analyze(
        CSV,
        "east sales",
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=3, max_executions=1, max_format_repairs=0),
        model=SequenceModel(
            [
                {"action": "candidate", "code": WRONG_TOTAL},
                review(1, outcome),
                last,
            ]
        ),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert len(result.reviews) == 1
    assert result.reviews[0].outcome == outcome
    assert (
        "passed calculation review"
        if next_action == "submit"
        else "reviewed replacement"
    ) in result.error


@pytest.mark.docker
@pytest.mark.parametrize(
    "checks",
    [
        {k: v for k, v in CHECKS.items() if k != "missing_values"},
        {**CHECKS, "statistical_definition": "  "},
    ],
)
def test_incomplete_or_empty_review_checks_cannot_approve_a_candidate(checks):
    result = analyze(
        CSV,
        "east sales",
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=2, max_executions=1, max_format_repairs=0),
        model=SequenceModel(
            [
                {"action": "candidate", "code": RECOMPUTED_TOTAL},
                {
                    "action": "review",
                    "candidate": 1,
                    "outcome": "passed",
                    "checks": checks,
                },
            ]
        ),
    )
    assert result.status == "failed"
    assert result.diagnostic == "response_format"
    assert not result.reviews
    assert result.answer is None


@pytest.mark.docker
@pytest.mark.parametrize(
    "candidate,code,message",
    [
        (99, RECOMPUTED_TOTAL, "successfully computed candidate"),
        (1, WRONG_TOTAL, "different calculation path"),
    ],
)
def test_recomputation_requires_existing_evidence_and_a_distinct_path(
    candidate, code, message
):
    result = analyze(
        CSV,
        "east sales",
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=2, max_executions=2, max_format_repairs=0),
        model=SequenceModel(
            [
                {"action": "candidate", "code": WRONG_TOTAL},
                {"action": "recompute", "candidate": candidate, "code": code},
            ]
        ),
    )
    assert result.status == "failed"
    assert message in result.error
    assert result.execution_attempts == 1
    assert len(result.candidates) == 1


@pytest.mark.docker
@pytest.mark.parametrize(
    "model_calls,executions,stop_reason",
    [
        (2, 2, "Model call budget exhausted"),
        (3, 1, "Analysis code execution limit reached"),
    ],
)
def test_budget_exhaustion_keeps_unresolved_review_without_publishing_an_answer(
    model_calls, executions, stop_reason
):
    result = analyze(
        CSV,
        "east sales",
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=model_calls, max_executions=executions),
        model=SequenceModel(
            [
                {"action": "candidate", "code": WRONG_TOTAL},
                review(1, "needs_recompute"),
                {"action": "recompute", "candidate": 1, "code": RECOMPUTED_TOTAL},
            ]
        ),
    )
    assert result.status == "failed"
    assert result.stop_reason == stop_reason
    assert result.answer is None
    assert result.reviews[0].outcome == "needs_recompute"
    assert result.execution_attempts == 1


@pytest.mark.docker
def test_review_transport_retry_preserves_evidence_budget_and_unknown_usage():
    from csv_agent.model import ModelError

    model = SequenceModel(
        [
            {"action": "candidate", "code": RECOMPUTED_TOTAL},
            ModelError(
                "Transient model failure", retryable=True, diagnostic="connection"
            ),
            review(1),
            {"action": "submit", "candidate": 1},
        ]
    )
    result = analyze(
        CSV,
        "east sales",
        strategy="langgraph_verify",
        model=model,
        limits=RunLimits(max_model_calls=4, max_executions=1, max_model_retries=1),
    )
    assert result.status == "completed", result.error
    assert result.answer == "10"
    assert result.execution_attempts == 1
    assert result.usage is None
    assert result.attempts[1].diagnostic == "connection"
    assert model.prompts[1]["candidates"] == model.prompts[2]["candidates"]
    assert model.prompts[1]["reviews"] == model.prompts[2]["reviews"]
    assert [p["remaining_budget"]["model_calls"] for p in model.prompts] == [3, 2, 1, 0]


@pytest.mark.docker
def test_recomputation_timeout_retains_review_and_does_not_submit_the_wrong_answer():
    result = analyze(
        CSV,
        "east sales",
        strategy="langgraph_verify",
        limits=RunLimits(max_model_calls=5, max_executions=2, timeout_seconds=1),
        model=SequenceModel(
            [
                {"action": "candidate", "code": WRONG_TOTAL},
                review(1, "needs_recompute"),
                {"action": "recompute", "candidate": 1, "code": "while True: pass"},
            ]
        ),
    )
    assert result.status == "timeout", result.error
    assert result.answer is None
    assert result.execution_attempts == 2
    assert len(result.candidates) == 1
    assert result.reviews[0].outcome == "needs_recompute"
    assert result.stop_reason == "Execution stopped; no further execution retries"
