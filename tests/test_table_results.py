"""Task 03 at the agreed analyze seam, using real isolated execution."""

import json

import pytest
from test_analysis import FixedModel
from test_charts import COMPARISON
from test_evaluation import CONFIG, dataset
from test_repair import SequenceModel

from csv_agent.analysis import AnalysisResult, analyze
from csv_agent.config import RunLimits
from csv_agent.evaluation import run_comparison


@pytest.mark.docker
def test_missing_table_values_return_null_without_losing_the_computed_answer():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
summary = df[['sales', 'empty']].mean().to_frame('mean').reset_index(names='column')
summary['nullable'] = pd.Series([1, pd.NA], dtype='Int64')
summary['time'] = pd.Series([pd.Timestamp('2026-10-31'), pd.NaT])
rows = json.loads(summary.to_json(orient='records', date_format='iso'))
rows[1]['mean'] = float('nan')
json.dump({'answer': 'Mean sales: 15', 'table': rows}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales,empty\n10,\n20,\n",
        "Report mean sales and the empty column; retain missing values.",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "completed", result.error
    assert result.answer == "Mean sales: 15"
    assert result.table == [
        {
            "column": "sales",
            "mean": 15.0,
            "nullable": 1,
            "time": "2026-10-31T00:00:00.000",
        },
        {"column": "empty", "mean": None, "nullable": None, "time": None},
    ]
    serialized = json.dumps(result.model_dump(), ensure_ascii=False, allow_nan=False)
    restored = AnalysisResult.model_validate_json(serialized)
    assert restored.answer == result.answer
    assert restored.table == result.table


@pytest.mark.docker
def test_nested_table_cells_preserve_values_answer_and_chart_when_serialized():
    code = COMPARISON.replace(
        "'sales_cny': int(total)",
        "'sales_cny': int(total), 'details': {'values': [int(total), None, float('nan')], 'label': '\u533a\u57df'}, 'categories': [region, '\u533a\u57df']",
    )
    result = analyze(
        b"region,sales\neast,10\nwest,20\neast,15\n",
        "Compare total sales by region in CNY with a chart and nested details.",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "completed", result.error
    assert result.answer == "east: 25 CNY; west: 20 CNY"
    assert result.table == [
        {
            "region": "east",
            "sales_cny": 25,
            "details": '{"values":[25,null,null],"label":"\u533a\u57df"}',
            "categories": '["east","\u533a\u57df"]',
        },
        {
            "region": "west",
            "sales_cny": 20,
            "details": '{"values":[20,null,null],"label":"\u533a\u57df"}',
            "categories": '["west","\u533a\u57df"]',
        },
    ]
    restored = AnalysisResult.model_validate_json(result.model_dump_json())
    assert restored.answer == result.answer
    assert restored.table == result.table
    assert restored.charts == result.charts
    assert restored.charts[0].path == "comparison.png"


@pytest.mark.docker
def test_uninterpretable_table_value_gives_specific_feedback_and_can_be_repaired():
    invalid = """
import json
json.dump({'answer': '45', 'table': [{'total': 45, 'details': [float('inf')]}]}, open('/output/result.json', 'w'))
"""
    valid = invalid.replace("[float('inf')]", "['Undefined: no denominator']")
    model = SequenceModel([json.dumps({"code": invalid}), json.dumps({"code": valid})])
    result = analyze(b"sales\n10\n20\n15\n", "sum", model=model)
    assert result.status == "completed", result.error
    assert result.answer == "45"
    assert result.table == [{"total": 45, "details": '["Undefined: no denominator"]'}]
    feedback = model.prompts[1]["feedback"]["error"]
    assert "table[0].details[0]" in feedback
    assert "non-finite" in feedback
    assert "explain" in feedback
    assert result.attempts[0].error == feedback


@pytest.mark.docker
@pytest.mark.parametrize(
    "answer",
    [
        "NaN",
        "Infinity",
        "-inf",
        "sNaN",
        "Mean sales: NaN",
        "Mean sales: Infinity",
        "@mean[NaN]",
        "@total[45] @mean[-Infinity]",
        "Total: 45; mean=nan",
    ],
)
def test_unexplained_nonfinite_numeric_answers_are_not_published(answer):
    code = f"""
import json
json.dump({{'answer': {answer!r}, 'table': [{{'total': 45}}]}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "sum",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert result.table == []
    assert "answer" in result.error
    assert "non-finite" in result.error


@pytest.mark.docker
@pytest.mark.parametrize(
    "answer",
    [
        "Mean sales: undefined because all observations are missing",
        "Mean sales: NaN; the mean is undefined because all observations are missing",
        "Mean sales: NaN\nThe mean is undefined because all observations are missing",
        "@mean[NaN]; the mean is undefined because all observations are missing",
        "Mean sales: NaN; Reason: no observations",
        'Most common category: "NaN"',
        '@category["Infinity"]',
    ],
)
def test_explained_missing_answer_and_quoted_category_labels_remain_readable(answer):
    code = f"""
import json
json.dump({{'answer': {answer!r}, 'table': [{{'mean': None}}]}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b'sales\n""\n',
        "Explain missing results or report the requested category label.",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "completed", result.error
    assert result.answer == answer
    assert result.table == [{"mean": None}]


@pytest.mark.docker
def test_large_finite_answer_is_preserved_without_float_conversion():
    code = """
import json
json.dump({'answer': '1e999', 'table': [{'zero': 0, 'enabled': True, 'note': 'NaN is a category label'}]}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n",
        "Return the computed answer with its original numeric precision.",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "completed", result.error
    assert result.answer == "1e999"
    assert result.table == [
        {"zero": 0, "enabled": True, "note": "NaN is a category label"}
    ]


@pytest.mark.docker
@pytest.mark.parametrize(
    ("repetitions", "output_bytes"), [(140000, 1048576), (5000, 16384)]
)
def test_normalized_table_cannot_exceed_size_limits_or_truncate_the_answer(
    repetitions, output_bytes
):
    code = r"""
import json
json.dump({'answer': '45', 'table': [{'details': ['\\' * 140000]}]}, open('/output/result.json', 'w'))
""".replace("140000", str(repetitions))
    result = analyze(
        b"sales\n10\n20\n15\n",
        "sum",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1, output_bytes=output_bytes),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert result.table == []
    assert "size limit" in result.error
    assert "compact" in result.error


@pytest.mark.docker
def test_batch_strategies_score_and_persist_answers_with_normalized_tables(tmp_path):
    root, manifest = dataset(tmp_path)
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
total = int(df['sales'].sum())
json.dump({'answer': f'@total[{total}]', 'table': [{'total': total, 'missing': float('nan'), 'values': df['sales'].tolist()}]}, open('/output/result.json', 'w'))
"""
    report = run_comparison(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "table-comparison",
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(max_executions=1),
        model_version="controlled-table-fixture",
        models={
            name: FixedModel(code) for name in ("single", "tool_loop", "langgraph")
        },
    )
    for batch in report.runs.values():
        record = batch.records[0]
        assert record.analysis.status == "completed", record.analysis.error
        assert record.analysis.answer == "@total[45]"
        assert record.analysis.table == [
            {"total": 45, "missing": None, "values": "[10,20,15]"}
        ]
        assert record.correctness == {"total": True}
        assert batch.summary.total == 1
    saved = json.loads((tmp_path / "table-comparison" / "report.json").read_text())
    for batch in saved["runs"].values():
        assert batch["records"][0]["analysis"]["answer"] == "@total[45]"
        assert batch["records"][0]["analysis"]["table"] == [
            {"total": 45, "missing": None, "values": "[10,20,15]"}
        ]


@pytest.mark.docker
@pytest.mark.parametrize(
    "expression",
    [
        "{'answer': ' '}",
        "{'answer': 'x' * 65537}",
        "{'answer': '45', 'table': [{}] * 1001}",
        "{'answer': '45', 'table': [None]}",
        "{'answer': '45', 'table': {'total': 45}}",
        "{'answer': float('inf')}",
        "{'answer': '45', 'table': [{'total': float('-inf')}]}",
    ],
    ids=[
        "blank-answer",
        "long-answer",
        "row-limit",
        "invalid-row",
        "invalid-table",
        "numeric-answer",
        "infinite-cell",
    ],
)
def test_invalid_results_still_fail_without_publishing_partial_answers(expression):
    code = f"""
import json
json.dump({expression}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "sum",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert result.table == []
    assert result.charts == []
    assert "Invalid result.json" in result.error
