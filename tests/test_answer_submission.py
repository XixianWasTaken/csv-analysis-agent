"""Complete answer submission at the agreed analyze and batch boundaries."""

import json

import pytest
from test_analysis import FixedModel
from test_evaluation import CONFIG, dataset
from test_repair import SequenceModel

from csv_agent.analysis import analyze
from csv_agent.answers import AnswerField, AnswerFormat
from csv_agent.config import RunLimits
from csv_agent.evaluation import run_comparison


@pytest.mark.docker
def test_computed_fields_submit_every_public_field_in_declared_order():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
json.dump({'answer_fields': {'mean': float(df.sales.mean()), 'total': int(df.sales.sum()), 'count': len(df)}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report total, count, and mean sales.",
        answer_format="@total[value] @count[value] @mean[value]",
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@total[45]\n@count[3]\n@mean[15.0]"
    assert result.answer_fields == {"total": 45, "count": 3, "mean": 15.0}
    assert result.execution is not None and result.execution.exit_code == 0


@pytest.mark.docker
def test_missing_field_feedback_repairs_the_complete_computation():
    incomplete = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
json.dump({'answer_fields': {'total': int(df.sales.sum())}}, open('/output/result.json', 'w'))
"""
    complete = incomplete.replace(
        "'total': int(df.sales.sum())",
        "'total': int(df.sales.sum()), 'count': len(df)",
    )
    model = SequenceModel(
        [json.dumps({"code": code}) for code in (incomplete, complete)]
    )
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report total and row count.",
        answer_format="@total[value] @count[value]",
        model=model,
    )
    assert result.status == "completed", result.error
    assert result.answer == "@total[45]\n@count[3]"
    assert "missing required fields count" in result.attempts[0].error
    assert model.prompts[1]["feedback"]["error"] == result.attempts[0].error


@pytest.mark.docker
@pytest.mark.parametrize(
    "fields, location",
    [
        ("{'total': 45, 'count': 3, 'extra': 1}", "extra"),
        ("{'total': None, 'count': 3}", "total"),
        ("{'total': '', 'count': 3}", "total"),
        ("{'total': float('nan'), 'count': 3}", "total"),
        ("{'total': {'value': [float('inf')]}, 'count': 3}", "total.value[0]"),
    ],
)
def test_invalid_fields_do_not_publish_an_answer_or_optional_table(fields, location):
    from csv_agent.config import RunLimits

    code = f"""
import json
json.dump({{'answer_fields': {fields}, 'table': [{{'total': 45}}]}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report total and count.",
        answer_format="@total[value] @count[value]",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed"
    assert result.answer is None and result.answer_fields is None and result.table == []
    assert location in result.error


@pytest.mark.docker
def test_public_precision_and_compound_values_are_rendered_without_prose():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
total = int(df.sales.sum())
json.dump({'answer_fields': {'ratio': total / 7, 'small': total / 700, 'category': '\u4e1c', 'rows': df.sales.tolist(), 'summary': {'total': total, '\u533a\u57df': '\u4e1c'}, 'flag': True}}, open('/output/result.json', 'w'), ensure_ascii=False)
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report sales ratios, rows, summary, and category.",
        answer_format=(
            "@ratio[r_value] @small[p_value] @category[label] @rows[row1,row2,...] "
            "@summary[summary_value] @flag[flag_value]\n"
            '"r_value" is a number rounded to two decimal places. '
            '"p_value" is a number rounded to four decimal places. '
            '"label" is a string. "summary_value" is a JSON dictionary. '
            '"flag_value" is a boolean.'
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == (
        "@ratio[6.43]\n@small[0.0643]\n@category[\u4e1c]\n@rows[10,20,15]\n"
        '@summary[{"total":45,"\u533a\u57df":"\u4e1c"}]\n@flag[True]'
    )
    assert result.answer_fields["ratio"] == 6.428571428571429


@pytest.mark.docker
def test_explicit_public_rules_preserve_precision_quotes_separators_and_null():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
json.dump({'answer_fields': {'sum': float(df.sales.sum()), 'exact': '1.2345678901234567890123456789', 'category': '\u4e1c', 'rows': df.sales.tolist(), 'empty': None}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n0.1\n0.2\n",
        'Keep full precision; quote the category, separate rows by "; ", and allow null for empty.',
        answer_format=AnswerFormat(
            fields={
                "sum": AnswerField(kind="number"),
                "exact": AnswerField(kind="number"),
                "category": AnswerField(kind="string", quoted=True),
                "rows": AnswerField(kind="list", separator="; "),
                "empty": AnswerField(allow_null=True),
            }
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == (
        "@sum[0.30000000000000004]\n@exact[1.2345678901234567890123456789]\n"
        '@category["\u4e1c"]\n@rows[0.1; 0.2]\n@empty[null]'
    )


@pytest.mark.docker
def test_public_tuple_applies_precision_to_each_component():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
values = [int(df.sales.min()), int(df.sales.max()), float(df.sales.mean())]
json.dump({'answer_fields': {'summary': values}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report minimum, maximum, and mean sales.",
        answer_format=(
            "@summary[min_value,max_value,mean_value]\n"
            '"min_value" and "max_value" are integers. '
            '"mean_value" is a number rounded to two decimal places.'
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@summary[10,20,15.00]"


@pytest.mark.docker
@pytest.mark.parametrize(
    "answer, reason",
    [
        ("@total[45]", "count"),
        ("@total[45] @count[3] @count[3]", "duplicate"),
        ("@total[45] @count[3] Explanation: complete", "presentation"),
        ("@total[45] @count[3] @extra[1]", "extra"),
        ("@total[45] @count[null]", "count"),
        ("@total[45] @count[3.5]", "integer"),
    ],
)
def test_legacy_strings_follow_public_completeness_and_presentation(answer, reason):
    from csv_agent.config import RunLimits

    code = f"""
import json
json.dump({{'answer': {answer!r}}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report total and row count.",
        answer_format='@total[total_value] @count[count_value] where "count_value" is an integer.',
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert reason in result.error


@pytest.mark.docker
def test_valid_legacy_string_retains_its_original_presentation():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
json.dump({'answer': f'@total[{int(df.sales.sum())}], @count[{len(df)}]'}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "total and count",
        answer_format='@total[total_value] @count[count_value] where "count_value" is an integer.',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@total[45], @count[3]"


@pytest.mark.docker
def test_conflicting_legacy_text_cannot_override_computed_fields():
    from csv_agent.config import RunLimits

    code = """
import json
json.dump({'answer': '@total[0]', 'answer_fields': {'total': 45}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "sum",
        answer_format="@total[value]",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed"
    assert "conflicts" in result.error
    assert result.answer is None


@pytest.mark.docker
def test_all_batch_strategies_submit_and_score_fields_using_only_public_requirements(
    tmp_path,
):
    import hashlib

    root, manifest = dataset(tmp_path)
    public_format = (
        "@total[total_value] @count[count_value] @mean[mean_value] @rows[row1,row2,...] "
        '@summary[summary_value] @category["category_value"]\n'
        '"count_value" is an integer. "mean_value" is a number rounded to two decimal places. '
        '"summary_value" is a JSON dictionary. "category_value" is a string.'
    )
    question_path = root / "questions.jsonl"
    question = json.loads(question_path.read_text())
    question.update(
        question="Report total, count, mean, sales rows, summary, and category.",
        format=public_format,
    )
    question_path.write_text(json.dumps(question) + "\n")
    labels_path = root / "labels.jsonl"
    labels_path.write_text(
        json.dumps(
            {
                "id": 0,
                "common_answers": [
                    ["total", "45"],
                    ["count", "3"],
                    ["mean", "15.00"],
                    ["rows", "10,20,15"],
                    ["summary", '{"total":45}'],
                    ["category", '"\u4e1c"'],
                ],
                "label_only_metadata": "host-only-label-sentinel",
            }
        )
        + "\n"
    )
    lock = json.loads(manifest.read_text())
    lock["dataset"]["questions_sha256"] = hashlib.sha256(
        question_path.read_bytes()
    ).hexdigest()
    lock["dataset"]["labels_sha256"] = hashlib.sha256(
        labels_path.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(lock))

    class FieldsModel:
        def generate(self, prompt, *, timeout_seconds):
            from csv_agent.model import ModelReply

            payload = json.loads(prompt)
            assert "host-only-label-sentinel" not in prompt
            assert "common_answers" not in prompt
            assert set(payload["answer_format"]["fields"]) == {
                "total",
                "count",
                "mean",
                "rows",
                "summary",
                "category",
            }
            code = """
import json
from pathlib import Path
import pandas as pd
assert sorted(p.name for p in Path('/input').iterdir()) == ['analysis.py', 'data.csv']
df = pd.read_csv('/input/data.csv')
total = int(df.sales.sum())
json.dump({'answer_fields': {'total': total, 'count': len(df), 'mean': float(df.sales.mean()), 'rows': df.sales.tolist(), 'summary': {'total': total}, 'category': '\u4e1c'}}, open('/output/result.json', 'w'))
"""
            return ModelReply(json.dumps({"code": code}), usage=None)

    report = run_comparison(
        root,
        manifest,
        split="development",
        output_dir=tmp_path / "comparison",
        config=CONFIG.model_copy(update={"temperature": 1}),
        limits=RunLimits(max_executions=1),
        model_version="controlled-fields",
        models={name: FieldsModel() for name in ("single", "tool_loop", "langgraph")},
    )
    expected = '@total[45]\n@count[3]\n@mean[15.00]\n@rows[10,20,15]\n@summary[{"total":45}]\n@category["\u4e1c"]'
    for run in report.runs.values():
        record = run.records[0]
        assert record.analysis.status == "completed", record.analysis.error
        assert record.analysis.answer == expected
        assert record.correctness == {
            name: True
            for name in ("total", "count", "mean", "rows", "summary", "category")
        }
        assert run.summary.total == 1 and run.summary.total_subquestions == 6
    saved = json.loads((tmp_path / "comparison/report.json").read_text())
    assert saved["runs"]["langgraph"]["records"][0]["analysis"]["answer"] == expected


@pytest.mark.docker
def test_comma_linked_public_rules_do_not_mix_numeric_precision_or_string_types():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
total = int(df.sales.sum())
json.dump({'answer_fields': {'ratio': total / 7, 'small': total / 700, 'feature': 'sales'}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report ratios and the feature name.",
        answer_format=(
            '@ratio[r_value] @small[p_value] @feature[feature_name] where "r_value" is a number '
            'rounded to two decimal places, and "p_value" is a number rounded to four decimal places. '
            '"feature_name" is the name of the feature, listed by importance.'
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@ratio[6.43]\n@small[0.0643]\n@feature[sales]"


@pytest.mark.docker
@pytest.mark.parametrize(
    "value", ["'[broken]'", "'line1\\nline2'", "'@extra[1]'", "'1e1000000'"]
)
def test_unrepresentable_or_oversized_fields_fail_with_field_feedback(value):
    code = f"""
import json
json.dump({{'answer_fields': {{'value': {value}}}}}, open('/output/result.json', 'w'))
"""
    contract = (
        AnswerFormat(fields={"value": AnswerField(kind="number", decimals=2)})
        if value == "'1e1000000'"
        else "@value[value]"
    )
    result = analyze(
        b"sales\n10\n",
        "Report the value without truncation.",
        answer_format=contract,
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed"
    assert result.answer is None
    assert "value" in result.error


@pytest.mark.docker
def test_readable_field_mapping_keeps_nested_data_in_normal_analysis():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
json.dump({'answer_fields': {'total': int(df.sales.sum()), 'category': '\u4e1c', 'details': [[10,20], {'rows': 2}]}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n", "Report sales with details.", model=FixedModel(code)
    )
    assert result.status == "completed", result.error
    assert result.answer == 'total: 30\ncategory: \u4e1c\ndetails: [[10,20],{"rows":2}]'


@pytest.mark.docker
def test_positional_public_rules_allow_null_only_in_the_declared_component():
    code = """
import json
json.dump({'answer_fields': {'values': [45, None]}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Return total and null for the absent optional value.",
        answer_format=AnswerFormat(
            fields={
                "values": AnswerField(
                    kind="list",
                    items=[
                        AnswerField(kind="integer"),
                        AnswerField(allow_null=True),
                    ],
                )
            }
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@values[45,null]"


@pytest.mark.docker
def test_explicit_string_category_named_nan_retains_public_spelling():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv', keep_default_na=False)
json.dump({'answer_fields': {'category': df.category.iloc[0]}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"category\nNaN\n",
        "Report the literal category string.",
        answer_format='@category[category_value] where "category_value" is a string.',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@category[NaN]"


@pytest.mark.docker
def test_invalid_json_is_rejected_before_publishing_any_fields():
    code = "open('/output/result.json', 'w').write('{\"answer_fields\": {\"total\": 45,}}')"
    result = analyze(
        b"sales\n10\n20\n15\n",
        "sum",
        answer_format="@total[value]",
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed" and result.answer is None
    assert "Invalid result.json" in result.error


@pytest.mark.docker
@pytest.mark.parametrize(
    "fields, public_format, expected",
    [
        (
            "{'total': 45}",
            '@total[value] where "value" is a number rounded to two decimal places. Example: @total[23].',
            "@total[45.00]",
        ),
        (
            "{'summary': {'total': 45}}",
            "@summary[{'total':value",
            '@summary[{"total":45}]',
        ),
        (
            "{'total': 45, 'mean': 15}",
            '@total[t] @mean[m] where "t", and "m" are numbers. Both should be rounded to two decimal places.',
            "@total[45.00]\n@mean[15.00]",
        ),
        (
            "{'total': 45}",
            "@total[number, rounded to 2 decimal places]",
            "@total[45.00]",
        ),
        (
            "{'rows': [10,20,15]}",
            '@rows[items] where "items" is a list of numbers separated by semicolons.',
            "@rows[10;20;15]",
        ),
    ],
)
def test_public_examples_grouped_precision_and_inline_descriptions(
    fields, public_format, expected
):
    code = f"""
import json
import pandas as pd
assert int(pd.read_csv('/input/data.csv').sales.sum()) == 45
json.dump({{'answer_fields': {fields}}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report computed sales statistics in the required format.",
        answer_format=public_format,
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == expected


@pytest.mark.docker
@pytest.mark.parametrize("group_kind", ["list", "tuple"])
def test_fixed_numeric_components_reject_a_string_with_field_feedback(group_kind):
    code = """
import json
import pandas as pd
json.dump({'answer_fields': {'summary': pd.read_csv('/input/data.csv').value.tolist()}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"value\n1.234567\nbanana\n",
        "Report the numeric components to two decimal places.",
        answer_format=f'@summary[first_value,second_value] where "summary" is a {group_kind} of numbers rounded to two decimal places.',
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed" and result.answer is None
    assert "summary[1]: expected a numeric value" in result.error


@pytest.mark.docker
@pytest.mark.parametrize("use_alternative", [False, True])
def test_public_conditional_submission_requires_one_complete_declared_branch(
    use_alternative,
):
    fields = (
        "{'status': 'No significant correlation'}"
        if use_alternative
        else "{'correlation': 0.75, 'p_value': 0.01}"
    )
    code = f"""
import json
json.dump({{'answer_fields': {fields}}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report correlation or the specified status when not significant.",
        answer_format=(
            '@correlation[r] @p_value[p] where "r" is a number rounded to two decimal places; '
            '"p" is a number rounded to four decimal places. '
            'If there is no significant correlation, please simply output @status["No significant correlation"]'
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == (
        '@status["No significant correlation"]'
        if use_alternative
        else "@correlation[0.75]\n@p_value[0.0100]"
    )


@pytest.mark.docker
def test_followup_precision_applies_only_to_its_public_field_group():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
json.dump({'answer_fields': {'median': float(df.sales.median()), 'range': float(df.sales.max() - df.sales.min())}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n0.123456\n0.654321\n",
        "Report median rounded to two decimals and the exact range.",
        answer_format=(
            "@median[median_value] @range[range_value]\n"
            'where "median_value" is the median sales. They are numbers rounded to two decimal places.\n'
            'where "range_value" is the range. It is a number without rounding.'
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@median[0.39]\n@range[0.530865]"


@pytest.mark.docker
@pytest.mark.parametrize(
    "public_format",
    [
        "@total[The sum of sales, rounded to two decimal places]",
        "@total[value]: The total should be a real number rounded to two decimal places.",
        '@total[value] where "value" is a number. Round up to two decimal places.',
    ],
)
def test_public_inline_local_and_followup_precision(public_format):
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
json.dump({'answer_fields': {'total': float(df.sales.sum())}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report total to two decimal places.",
        answer_format=public_format,
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@total[45.00]"


@pytest.mark.docker
@pytest.mark.parametrize("mean", ["15", "15.000", "15.001"])
def test_legacy_numeric_presentation_has_the_requested_decimal_places(mean):
    code = f"""
import json
import pandas as pd
assert float(pd.read_csv('/input/data.csv').sales.mean()) == 15.0
json.dump({{'answer': '@mean[{mean}]'}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report mean sales to two decimal places.",
        answer_format='@mean[value] where "value" is a number rounded to two decimal places.',
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed" and result.answer is None
    assert "precision" in result.error


@pytest.mark.docker
def test_capitalized_global_precision_is_applied_to_computed_numeric_fields():
    code = """
import json
import pandas as pd
total = int(pd.read_csv('/input/data.csv').sales.sum())
json.dump({'answer_fields': {'ratio': total / 7}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report sales divided by seven.",
        answer_format="@ratio[value] All outputs should be rounded to Two decimal places.",
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@ratio[6.43]"


@pytest.mark.docker
def test_and_linked_definitions_do_not_cast_status_strings_to_numbers():
    code = """
import json
import pandas as pd
from scipy import stats
p = float(stats.shapiro(pd.read_csv('/input/data.csv').sales).pvalue)
json.dump({'answer_fields': {'p_value': p, 'normal_distribution': 'yes' if p > 0.05 else 'no'}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Test normality and return the public status.",
        answer_format='@p_value[p] @normal_distribution[yes/no] where "p" is a number rounded to four decimal places and "normal_distribution" is a string that can be "yes" if p > 0.05 or "no" otherwise.',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@p_value[1.0000]\n@normal_distribution[yes]"


@pytest.mark.docker
@pytest.mark.parametrize("has_previous", [True, False])
def test_public_date_and_numeric_or_string_missing_data_result(has_previous):
    data = (
        b"date,sales\n2017-11,10\n2017-12,20\n"
        if has_previous
        else b"date,sales\n2017-12,20\n"
    )
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
date = df.loc[df.sales.idxmax(), 'date']
change = float((df.sales.iloc[-1] - df.sales.iloc[-2]) / df.sales.iloc[-2] * 100) if len(df) > 1 else "Previous day data doesn't exist"
json.dump({'answer_fields': {'date': date, 'change': change}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        data,
        "Report maximum sales date and the change from the previous row.",
        answer_format='@date[date_value] @change[percentage_change] where "date_value" is a string in YYYY-MM format and "percentage_change" is a number rounded to two decimal places or the string "Previous day data doesn\'t exist".',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == (
        "@date[2017-12]\n@change[100.00]"
        if has_previous
        else "@date[2017-12]\n@change[Previous day data doesn't exist]"
    )


@pytest.mark.docker
def test_plain_public_conditional_can_submit_its_no_outliers_status():
    code = """
import json
import pandas as pd
count = int((pd.read_csv('/input/data.csv').sales > 100).sum())
fields = {'count': count} if count else {'status': 'No Outliers Detected'}
json.dump({'answer_fields': fields}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report sales above 100 or the no outliers status.",
        answer_format='@count[count_value] where "count_value" is an integer. If there are no outliers, output @status["No Outliers Detected"]',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@status["No Outliers Detected"]'


@pytest.mark.docker
def test_if_else_describes_a_required_field_alongside_the_numeric_answer():
    code = """
import json
import pandas as pd
total = int(pd.read_csv('/input/data.csv').sales.sum())
json.dump({'answer_fields': {'total': total, 'positive': total > 0}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report total and whether it is positive.",
        answer_format='@total[value] where "value" is a number. If total is above zero, output @positive[True], else output @positive[False].',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@total[45]\n@positive[True]"


@pytest.mark.docker
def test_ellipsis_list_uses_its_public_component_precision():
    code = """
import json
import pandas as pd
values = pd.read_csv('/input/data.csv').damage.tolist()
json.dump({'answer_fields': {'average_damage': values}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"year,damage\n2000,1.234567\n2001,2.345678\n2010,3.456789\n",
        "Report the average damage for each year.",
        answer_format='@average_damage[average_damage_from_2000, average_damage_from_2001, ..., average_damage_from_2010] where "average_damage_from_year" is a mean damage rounded to two decimal places.',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@average_damage[1.23,2.35,3.46]"


@pytest.mark.docker
def test_nested_list_groups_are_rejected_without_silently_flattening():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
groups = [group.sales.tolist() for _, group in df.groupby('region')]
json.dump({'answer_fields': {'pairs': groups}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"region,sales\na,1\na,2\nb,3\nb,4\n",
        "Report grouped pairs without flattening them.",
        answer_format=AnswerFormat(
            fields={
                "pairs": AnswerField(
                    kind="list",
                    items=[
                        AnswerField(kind="list"),
                        AnswerField(kind="list"),
                    ],
                )
            }
        ),
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed" and result.answer is None
    assert "pairs" in result.error and "nested" in result.error


@pytest.mark.docker
def test_strings_containing_list_separator_require_an_unambiguous_public_format():
    code = """
import json
import pandas as pd
json.dump({'answer_fields': {'locations': pd.read_csv('/input/data.csv').city.tolist()}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b'city\n"New York, NY"\nBoston\n',
        "Report locations separated by commas.",
        answer_format='@locations[values] where "values" is a list of strings separated by commas.',
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    assert result.status == "failed" and result.answer is None
    assert "locations" in result.error and "separator" in result.error


@pytest.mark.docker
def test_quoted_public_list_items_preserve_separator_characters_in_strings():
    code = """
import json
import pandas as pd
json.dump({'answer_fields': {'locations': pd.read_csv('/input/data.csv').city.tolist()}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b'city\n"New York, NY"\nBoston\n',
        "Report JSON-quoted locations separated by commas.",
        answer_format=AnswerFormat(
            fields={
                "locations": AnswerField(
                    kind="list", item=AnswerField(kind="string", quoted=True)
                )
            }
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@locations["New York, NY","Boston"]'


@pytest.mark.docker
def test_legacy_quoted_list_keeps_separator_characters_and_its_presentation():
    code = """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
answer = '@locations[' + ','.join(json.dumps(city) for city in df.city) + ']'
json.dump({'answer': answer}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b'city\n"New York, NY"\nBoston\n',
        "Report JSON-quoted locations separated by commas.",
        answer_format=AnswerFormat(
            fields={
                "locations": AnswerField(
                    kind="list", item=AnswerField(kind="string", quoted=True)
                )
            }
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@locations["New York, NY","Boston"]'


@pytest.mark.docker
def test_public_string_preserves_email_category_characters():
    code = """
import json
import pandas as pd
json.dump({'answer_fields': {'category': pd.read_csv('/input/data.csv').category.iloc[0]}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"category\nuser@example.com\n",
        "Report the literal category string.",
        answer_format='@category[value] where "value" is a string.',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@category[user@example.com]"


@pytest.mark.docker
def test_dictionary_numeric_values_obey_their_explicit_public_precision():
    code = """
import json
import pandas as pd
mean = float(pd.read_csv('/input/data.csv').sales.mean())
json.dump({'answer_fields': {'statistics': {'mean': mean, 'label': '\u4e1c'}}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Return mean sales to two decimal places in a JSON dictionary.",
        answer_format=AnswerFormat(
            fields={"statistics": AnswerField(kind="dict", decimals=2)}
        ),
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@statistics[{"mean":15.00,"label":"\u4e1c"}]'


@pytest.mark.docker
@pytest.mark.parametrize(
    "public_format",
    [
        '@statistics[{"mean":value}] All outputs should be rounded to Two decimal places.',
        '@statistics[{"mean":value}] where "value" is a number rounded to two decimal places.',
    ],
)
def test_public_dictionary_template_uses_its_numeric_precision(public_format):
    code = """
import json
import pandas as pd
mean = float(pd.read_csv('/input/data.csv').sales.mean()) / 7
json.dump({'answer_fields': {'statistics': {'mean': mean}}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report mean sales divided by seven to two decimal places.",
        answer_format=public_format,
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@statistics[{"mean":2.14}]'


@pytest.mark.docker
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize(
    "public_format", ['@locations["city1","city2"]', '@locations["city1","city2",...]']
)
def test_public_quoted_placeholders_quote_each_component(legacy, public_format):
    code = """
import json
import pandas as pd
cities = pd.read_csv('/input/data.csv').city.tolist()
result = {'answer': '@locations[' + ','.join(json.dumps(city) for city in cities) + ']'} if LEGACY else {'answer_fields': {'locations': cities}}
json.dump(result, open('/output/result.json', 'w'))
""".replace("LEGACY", repr(legacy))
    result = analyze(
        b'city\n"New York, NY"\nBoston\n',
        "Report JSON-quoted locations in the required format.",
        answer_format=public_format,
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@locations["New York, NY","Boston"]'


@pytest.mark.docker
@pytest.mark.parametrize("column_count", [2, 3])
def test_public_list_of_two_strings_validates_the_pair_cardinality(column_count):
    code = """
import json
import pandas as pd
json.dump({'answer_fields': {'pair': pd.read_csv('/input/data.csv').columns.tolist()}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"age,fare\n10,20\n" if column_count == 2 else b"age,fare,Parch\n10,20,1\n",
        "Report exactly two column names as a pair.",
        answer_format='@pair[pair_value] where "pair_value" is a list of two strings representing column names.',
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    if column_count == 2:
        assert result.status == "completed", result.error
        assert result.answer == "@pair[age,fare]"
    else:
        assert result.status == "failed" and result.answer is None
        assert "expected 2 list components" in result.error


@pytest.mark.docker
@pytest.mark.parametrize("missing_component", [False, True])
def test_dictionary_components_have_independent_public_precision_and_required_keys(
    missing_component,
):
    code = """
import json
import pandas as pd
total = int(pd.read_csv('/input/data.csv').sales.sum())
statistics = {'r': total / 700}
if not MISSING:
    statistics['p'] = total / 7000
json.dump({'answer_fields': {'statistics': statistics}}, open('/output/result.json', 'w'))
""".replace("MISSING", repr(missing_component))
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report the two sales ratios using different precisions.",
        answer_format='@statistics[{"r": r_value, "p": p_value}] where "r_value" is a number rounded to two decimal places. "p_value" is a number rounded to four decimal places.',
        model=FixedModel(code),
        limits=RunLimits(max_executions=1),
    )
    if missing_component:
        assert result.status == "failed" and result.answer is None
        assert (
            "statistics" in result.error and "missing required fields p" in result.error
        )
    else:
        assert result.status == "completed", result.error
        assert result.answer == '@statistics[{"r":0.06,"p":0.0064}]'


@pytest.mark.docker
def test_dictionary_properties_inherit_the_public_dictionary_precision():
    code = """
import json
import pandas as pd
mean = float(pd.read_csv('/input/data.csv').sales.mean()) / 7
json.dump({'answer_fields': {'statistics': {'mean': mean}}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report mean sales divided by seven to two decimal places.",
        answer_format='@statistics[{"mean": value}] where "statistics" is a JSON dictionary with numeric values rounded to two decimal places.',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@statistics[{"mean":2.14}]'


@pytest.mark.docker
@pytest.mark.parametrize(
    ("numeric_string", "expected"), [("+1", "1"), (".5", "0.5"), ("01", "1")]
)
def test_dictionary_numeric_strings_submit_valid_json(numeric_string, expected):
    code = """
import json
import pandas as pd
value = pd.read_csv('/input/data.csv', dtype=str).value.iloc[0]
json.dump({'answer_fields': {'statistics': {'value': value}}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        f"value\n{numeric_string}\n".encode(),
        "Report the numeric value in a JSON dictionary.",
        answer_format='@statistics[{"value": numeric_value}] where "numeric_value" is a number.',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@statistics[{"value":' + expected + "}]"
    assert json.loads(result.answer[len("@statistics[") : -1]) == {
        "value": json.loads(expected)
    }


@pytest.mark.docker
def test_dictionary_default_precision_and_member_override_are_independent():
    code = """
import json
import pandas as pd
mean = float(pd.read_csv('/input/data.csv').sales.mean())
json.dump({'answer_fields': {'statistics': {'r': mean / 7, 'p': mean / 700}}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"sales\n10\n20\n15\n",
        "Report sales ratios using the dictionary default and a member override.",
        answer_format='@statistics[{"r": r_value, "p": p_value}] where "statistics" is a JSON dictionary with numeric values rounded to two decimal places by default. "r_value" is a number. "p_value" is a number rounded to four decimal places.',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@statistics[{"r":2.14,"p":0.0214}]'


@pytest.mark.docker
@pytest.mark.parametrize(
    "public_format",
    [
        '@statistics[{"rounded": rounded_value, "exact": exact_value}] where "statistics" is a JSON dictionary with numeric values rounded to two decimal places by default. "rounded_value" is a number. "exact_value" is a number without rounding.',
        '@statistics[{"rounded": rounded_value, "exact": exact_value}] All outputs should be rounded to two decimal places. "rounded_value" is a number. "exact_value" is a number without rounding.',
        '@statistics[{"rounded": rounded_value, "exact": exact_value}] where "rounded_value" is a number rounded to two decimal places. "exact_value" is a number without rounding.',
    ],
)
def test_dictionary_member_without_rounding_keeps_the_computed_precision(public_format):
    code = """
import json
import pandas as pd
values = pd.read_csv('/input/data.csv').value.tolist()
json.dump({'answer_fields': {'statistics': {'rounded': values[0], 'exact': values[1]}}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"value\n1.23456\n0.123456789\n",
        "Report one rounded value and one value without rounding.",
        answer_format=public_format,
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@statistics[{"rounded":1.23,"exact":0.123456789}]'


@pytest.mark.docker
def test_unrounded_dictionary_overrides_global_defaults_but_keeps_member_precision():
    code = """
import json
import pandas as pd
values = pd.read_csv('/input/data.csv').value.tolist()
json.dump({'answer_fields': {'statistics': {'mean': values[0], 'ratio': values[1]}}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"value\n0.123456789\n0.0123456789\n",
        "Report an unrounded dictionary with one explicitly rounded member.",
        answer_format='@statistics[{"mean": value, "ratio": ratio_value}] All outputs should be rounded to two decimal places by default. "statistics" is a JSON dictionary with numeric values without rounding. "value" is a number. "ratio_value" is a number rounded to four decimal places.',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == '@statistics[{"mean":0.123456789,"ratio":0.0123}]'


@pytest.mark.docker
@pytest.mark.parametrize(
    ("public_format", "expected"),
    [
        (
            '@summary[first_value,second_value] where "summary" is a tuple of numbers rounded to four decimal places.',
            "@summary[0.1235,0.0123]",
        ),
        (
            '@summary[first_value,second_value] All outputs should be rounded to two decimal places by default. "summary" is a tuple of numbers without rounding.',
            "@summary[0.123456789,0.0123456789]",
        ),
    ],
)
def test_tuple_components_follow_their_field_precision(public_format, expected):
    code = """
import json
import pandas as pd
json.dump({'answer_fields': {'summary': pd.read_csv('/input/data.csv').value.tolist()}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"value\n0.123456789\n0.0123456789\n",
        "Report both values using the tuple's public precision.",
        answer_format=public_format,
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == expected


@pytest.mark.docker
def test_variable_components_keep_their_precision_override():
    code = """
import json
import pandas as pd
json.dump({'answer_fields': {'summary': pd.read_csv('/input/data.csv').value.tolist()}}, open('/output/result.json', 'w'))
"""
    result = analyze(
        b"value\n0.123456789\n0.0123456789\n",
        "Report each component using its explicit precision.",
        answer_format='@summary[entry1,entry2,...] where "summary" is a tuple of numbers rounded to four decimal places by default. "entry" is a number rounded to two decimal places.',
        model=FixedModel(code),
    )
    assert result.status == "completed", result.error
    assert result.answer == "@summary[0.12,0.01]"
