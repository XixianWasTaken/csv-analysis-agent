"""Task 02 at the agreed public analysis and batch seams; real offline Docker."""

import hashlib
import json

import httpx
import pytest
from test_analysis import FixedModel
from test_evaluation import CONFIG, dataset

from csv_agent.analysis import analyze
from csv_agent.config import ModelConfig, RunLimits
from csv_agent.evaluation import run_comparison

T_TEST = """
import json
import pandas as pd
from scipy import stats
values = pd.read_csv('/input/data.csv')['value']
test = stats.ttest_1samp(values, popmean=0, alternative='two-sided', nan_policy='raise')
row = {'statistic': float(test.statistic), 'p_value': float(test.pvalue)}
json.dump({'answer': str(row['p_value']), 'table': [row]}, open('/output/result.json', 'w'))
"""


@pytest.mark.docker
def test_known_one_sample_t_test_is_computed_without_rounding():
    result = analyze(
        b"value\n1\n2\n3\n4\n5\n",
        "Two-sided one-sample t-test against mean zero using scipy; reject missing values and keep full precision.",
        model=FixedModel(T_TEST),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "completed", result.error
    assert result.table == [
        {
            "statistic": pytest.approx(4.242640687119285, rel=0, abs=1e-12),
            "p_value": pytest.approx(0.013235599563682695, rel=0, abs=1e-12),
        }
    ]
    assert float(result.answer) == pytest.approx(0.013235599563682695, rel=0, abs=1e-12)


@pytest.mark.docker
def test_seeded_split_and_training_only_scaling_preserve_regression_predictions():
    code = """
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('/input/data.csv')
train, test = train_test_split(df, test_size=0.25, random_state=42)
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(train[['x']], train['y'])
predictions = model.predict(test[['x']])
mse = float(mean_squared_error(test['y'], predictions))
rows = [
    {'test_row': int(index + 1), 'prediction': float(prediction),
     'training_mean': float(model[0].mean_[0])}
    for index, prediction in zip(test.index, predictions)
]
json.dump({'answer': str(mse), 'table': rows}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"x,y\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13\n7,15\n8,17\n",
        "Use LinearRegression defaults in a StandardScaler pipeline; split 25% test data with random_state=42 and fit preprocessing on training data only. Report test predictions and MSE without rounding.",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "completed", result.error
    assert result.table == [
        {
            "test_row": 2,
            "prediction": pytest.approx(5, rel=0, abs=1e-12),
            "training_mean": pytest.approx(4.666666666666667, rel=0, abs=1e-12),
        },
        {
            "test_row": 6,
            "prediction": pytest.approx(13, rel=0, abs=1e-12),
            "training_mean": pytest.approx(4.666666666666667, rel=0, abs=1e-12),
        },
    ]
    assert float(result.answer) == pytest.approx(0, rel=0, abs=1e-24)


@pytest.mark.docker
def test_statsmodels_ols_preserves_known_coefficients_and_goodness_of_fit():
    code = """
import json
import pandas as pd
import statsmodels.api as sm
df = pd.read_csv('/input/data.csv')
model = sm.OLS(df['y'], sm.add_constant(df[['x']]), missing='raise').fit()
row = {'intercept': float(model.params['const']), 'slope': float(model.params['x']),
       'r_squared': float(model.rsquared), 'residual_mse': float(model.mse_resid)}
json.dump({'answer': str(row['r_squared']), 'table': [row]}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"x,y\n1,3\n2,5\n3,8\n4,9\n5,11\n",
        "Fit statsmodels OLS with an intercept, reject missing data, and report coefficients, R squared, and residual MSE at full precision.",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "completed", result.error
    assert result.table == [
        {
            "intercept": pytest.approx(1.2, rel=0, abs=1e-12),
            "slope": pytest.approx(2, rel=0, abs=1e-12),
            "r_squared": pytest.approx(0.9803921568627451, rel=0, abs=1e-12),
            "residual_mse": pytest.approx(0.2666666666666667, rel=0, abs=1e-12),
        }
    ]
    assert float(result.answer) == pytest.approx(0.9803921568627451, rel=0, abs=1e-12)


@pytest.mark.docker
def test_batch_strategies_and_ordinary_analysis_share_scientific_computation(tmp_path):
    data = b"value\n1\n2\n3\n4\n5\n"
    root, manifest = dataset(tmp_path, csv_data=data)
    question = (
        "Two-sided scipy one-sample t-test against mean zero; reject missing data."
    )
    public = {
        "id": 0,
        "file_name": "sales.csv",
        "question": question,
        "constraints": "Keep full precision.",
        "format": "@p_value[value]",
    }
    labels = {"id": 0, "common_answers": [["p_value", "0.013235599563682695"]]}
    lock = json.loads(manifest.read_text())
    for name, row, checksum in (
        ("questions.jsonl", public, "questions_sha256"),
        ("labels.jsonl", labels, "labels_sha256"),
    ):
        path = root / name
        path.write_text(json.dumps(row) + "\n")
        lock["dataset"][checksum] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(lock))
    code = T_TEST.replace("str(row['p_value'])", "f\"@p_value[{row['p_value']}]\"")
    report = run_comparison(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "scientific-comparison",
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(max_executions=1),
        model_version="controlled-scientific-fixture",
        models={
            strategy: FixedModel(code)
            for strategy in ("single", "tool_loop", "langgraph")
        },
    )
    ordinary = analyze(data, question, model=FixedModel(code), limits=report.limits)
    assert report.mode == "controlled"
    assert report.limits.image.startswith("sha256:")
    assert ordinary.status == "completed", ordinary.error
    for batch in report.runs.values():
        record = batch.records[0]
        assert record.analysis.status == "completed", record.analysis.error
        assert record.analysis.answer == ordinary.answer
        assert record.analysis.table == ordinary.table
        assert record.correctness == {"p_value": True}
        assert batch.limits.image == report.limits.image
        assert batch.summary.total == 1


@pytest.mark.parametrize(
    ("protocol", "mode"),
    [("anthropic", "structured"), ("openai", "tools")],
)
def test_scientific_capabilities_and_question_constraints_reach_the_model(
    monkeypatch, protocol, mode
):
    question = (
        "Fit RandomForestClassifier with library defaults except random_state=42; "
        "split with test_size=0.25 and random_state=42, fit preprocessing only on "
        "training data, reject missing values, and keep full output precision."
    )
    requests = []

    def provider(request):
        payload = json.loads(request.content)
        requests.append(payload)
        limitation = {"status": "insufficient_input", "reason": "Missing target column"}
        return httpx.Response(
            200,
            json={"content": [{"type": "text", "text": json.dumps(limitation)}]}
            if protocol == "anthropic"
            else {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "report_insufficient_input",
                                        "arguments": json.dumps(limitation),
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        )

    monkeypatch.setenv("SCIENTIFIC_TEST_KEY", "controlled-fixture-key")
    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(provider), **kwargs),
    )
    result = analyze(
        b"x\n1\n2\n",
        question,
        config=ModelConfig(
            protocol=protocol,
            instruction_mode=mode,
            endpoint="https://fixture.example/messages",
            name="controlled",
            api_key_env="SCIENTIFIC_TEST_KEY",
        ),
    )
    assert requests
    assert result.status == "insufficient_input"
    assert result.execution is None
    payload = requests[0]
    system = (
        payload["system"]
        if protocol == "anthropic"
        else payload["messages"][0]["content"]
    )
    assert all(
        library in system
        for library in ("numpy", "scipy", "scikit-learn", "statsmodels")
    )
    assert json.loads(payload["messages"][-1]["content"])["question"] == question
