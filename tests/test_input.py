"""Ticket 02 behavior through the agreed analysis-task boundary."""

import hashlib
import json

import pytest

from csv_agent.analysis import analyze
from csv_agent.model import ModelReply


class ReplyModel:
    def __init__(self, text):
        self.text = text
        self.prompt = None

    def generate(self, prompt, *, timeout_seconds):
        self.prompt = json.loads(prompt)
        return ModelReply(self.text, None)


class ColumnTotalsModel:
    """Controlled model using the CSV reading contract supplied to real models."""

    def __init__(self, answer_format="{total}"):
        self.answer_format = answer_format
        self.profile = None

    def generate(self, prompt, *, timeout_seconds):
        self.profile = json.loads(prompt)["csv_profile"]
        code = f"""
import hashlib
import json
from pathlib import Path
import pandas as pd
source = Path('/input/data.csv').read_bytes()
assert hashlib.sha256(source).hexdigest() == {self.profile["source_sha256"]!r}
df = pd.read_csv('/input/data.csv', **{self.profile["read_csv_options"]!r})
assert list(df.columns) == {self.profile["columns"]!r}
totals = {{column: int(df[column].sum()) for column in df.columns}}
total = totals[df.columns[0]]
json.dump({{'answer': {self.answer_format!r}.format(total=total), 'table': [totals]}}, open('/output/result.json', 'w'))
"""
        return ModelReply(json.dumps({"code": code}), None)


@pytest.mark.docker
def test_unnamed_column_keeps_its_data_and_computes_in_isolation():
    data = b",sales\n10,1\n20,2\n15,3\n"
    model = ColumnTotalsModel()
    result = analyze(data, "Sum the first unnamed column.", model=model)

    assert result.status == "completed", result.error
    assert result.answer == "45"
    assert result.table == [{"Unnamed: 0": 45, "sales": 6}]
    assert result.profile["column_mapping"] == [
        {"position": 1, "original": "", "name": "Unnamed: 0"},
        {"position": 2, "original": "sales", "name": "sales"},
    ]
    assert result.profile["source_sha256"] == hashlib.sha256(data).hexdigest()
    assert model.profile == result.profile
    assert result.execution is not None and result.execution.exit_code == 0


@pytest.mark.docker
@pytest.mark.parametrize(
    "data", [b"\n10\n20\n", b"\r\n10\r\n20\r\n", b"\xef\xbb\xbf\n10\n20\n"]
)
def test_single_empty_header_record_keeps_first_data_row(data):
    result = analyze(data, "Sum the unnamed column.", model=ColumnTotalsModel())
    assert result.status == "completed", result.error
    assert result.answer == "30"
    assert result.profile["rows"] == 2
    assert result.profile["column_mapping"] == [
        {"position": 1, "original": "", "name": "Unnamed: 0"}
    ]
    assert result.table == [{"Unnamed: 0": 30}]


@pytest.mark.docker
@pytest.mark.parametrize(
    "data,names,totals",
    [
        (
            b",,Unnamed: 0,Unnamed: 1,Unnamed: 0.1\n1,2,3,4,5\n10,20,30,40,50\n",
            [
                "Unnamed: 0.2",
                "Unnamed: 1.1",
                "Unnamed: 0",
                "Unnamed: 1",
                "Unnamed: 0.1",
            ],
            [11, 22, 33, 44, 55],
        ),
        (
            b'\xef\xbb\xbf"",,\r\n1,2,3\r\n4,5,6\r\n',
            ["Unnamed: 0", "Unnamed: 1", "Unnamed: 2"],
            [5, 7, 9],
        ),
        (
            b' ,\t,""\n1,2,3\n4,5,6\n',
            ["Unnamed: 0", "Unnamed: 1", "Unnamed: 2"],
            [5, 7, 9],
        ),
        (b"   \n10\n20\n", ["Unnamed: 0"], [30]),
        (b'"a\nb",\n10,1\n20,2\n', ["a\nb", "Unnamed: 1"], [30, 3]),
    ],
)
def test_unnamed_identifiers_avoid_collisions_and_keep_all_rows(data, names, totals):
    result = analyze(data, "Sum each column.", model=ColumnTotalsModel())
    assert result.status == "completed", result.error
    assert result.profile["columns"] == names
    assert result.profile["rows"] == 2
    assert result.table == [dict(zip(names, totals))]
    assert [column["name"] for column in result.profile["column_mapping"]] == names
    assert result.profile["source_sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize(
    "strategy", ["single", "tool_loop", "langgraph", "langgraph_no_repair"]
)
def test_unnamed_profile_includes_types_missing_values_and_bounded_observations(
    strategy,
):
    model = ReplyModel(
        '{"status":"insufficient_input","reason":"Missing required sales field"}'
    )
    result = analyze(
        b",active\n10,True\n,False\n20,True\n",
        "sales total",
        model=model,
        strategy=strategy,
    )
    assert result.status == "insufficient_input"
    assert result.profile["types"] == {"Unnamed: 0": "float64", "active": "bool"}
    assert result.profile["missing"] == {"Unnamed: 0": 1, "active": 0}
    assert result.profile["sample"] == [
        ["10.0", "True"],
        ["", "False"],
        ["20.0", "True"],
    ]
    assert model.prompt["csv_profile"] == result.profile
    assert result.execution is None
    assert result.answer is None


@pytest.mark.parametrize(
    "data,diagnostic",
    [
        (b",a,a\n1,2,3\n", "unique"),
        (b",a\n1,2,3\n", "field counts"),
        (b",a\n1\n", "field counts"),
        (b"\n1,2\n", "field counts"),
        (b",a\n\xff,2\n", "UTF-8"),
        (b',a\n"unterminated,2\n', "Invalid CSV"),
    ],
)
def test_unnamed_headers_do_not_hide_corrupt_or_ambiguous_input(data, diagnostic):
    model = ColumnTotalsModel()
    result = analyze(data, "Sum each column.", model=model)
    assert result.status == "failed"
    assert diagnostic in result.error
    assert model.profile is None
    assert result.answer is None
    assert result.execution is None


def test_profile_describes_full_data_with_only_a_bounded_sample():
    model = ReplyModel("invalid")
    result = analyze(
        b"region,sales,active\neast,10,True\nwest,,False\neast,15,True\nwest,20,False\neast,30,True\nwest,40,False\n",
        "sum",
        model=model,
    )
    assert result.profile["rows"] == 6
    assert result.profile["columns"] == ["region", "sales", "active"]
    assert result.profile["types"] == {
        "region": "str",
        "sales": "float64",
        "active": "bool",
    }
    assert result.profile["missing"] == {"region": 0, "sales": 1, "active": 0}
    assert len(result.profile["sample"]) == 5
    assert model.prompt["csv_profile"] == result.profile
    assert "40" not in json.dumps(model.prompt)


def test_missing_field_or_business_definition_ends_with_explanation():
    for question, reason in [
        ("\u5229\u6da6\u5408\u8ba1\u662f\u591a\u5c11？", "Missing cost field; cannot calculate profit."),
        ("\u9ad8\u4ef7\u503c\u5ba2\u6237\u6709\u591a\u5c11？", "\u7f3a\u5c11\u9ad8\u4ef7\u503c\u5ba2\u6237\u7684\u5224\u5b9a\u9608\u503c。"),
    ]:
        model = ReplyModel(
            json.dumps({"status": "insufficient_input", "reason": reason})
        )
        result = analyze(b"sales\n10\n", question, model=model)
        assert result.status == "insufficient_input"
        assert result.error == reason
        assert result.answer is None
        assert result.execution is None
        assert result.code == ""
        assert result.profile["rows"] == 1


def test_invalid_input_is_diagnosed_before_model_configuration(monkeypatch):
    monkeypatch.delenv("MODEL_ENDPOINT", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    for data, diagnostic in [
        (b"\xff\xfe", "UTF-8"),
        (b"a,b\n1,2,3\n", "field counts"),
        (b"a,b\n", "no data rows"),
        (b"\n", "no data rows"),
        (b'a\n"unterminated', "Invalid CSV"),
    ]:
        result = analyze(data, "sum")
        assert result.status == "failed"
        assert diagnostic in result.error
        assert result.execution is None
        assert result.answer is None


def test_task_reports_running_then_terminal_status():
    statuses = []
    result = analyze(
        b"sales\n10\n", "sum", model=ReplyModel(""), on_status=statuses.append
    )
    assert statuses == ["running", "failed"]
    assert result.status == "failed"
    assert "empty" in result.error.lower()
    assert result.execution is None


def test_large_sample_cells_are_truncated():
    result = analyze(
        ("name\n" + "x" * 1000 + "\n").encode(), "count", model=ReplyModel("invalid")
    )
    assert result.profile["sample"] == [["x" * 200]]
    assert "truncated" in result.profile["sample_note"]


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "   ",
        "The answer is 45",
        "{}",
        '{"code": 123}',
        '{"code": "print("}',
        '{"status": "insufficient_input", "reason": " "}',
        '{"status": "insufficient_input", "reason": "missing", "code": "pass"}',
    ],
)
def test_invalid_model_instructions_keep_failure_reason_and_no_answer(reply):
    result = analyze(b"sales\n10\n", "sum", model=ReplyModel(reply))
    assert result.status == "failed"
    assert result.error
    assert result.answer is None
    assert result.execution is None


@pytest.mark.docker
def test_known_filter_is_computed_and_reports_terminal_status():
    statuses = []
    model = ReplyModel(
        json.dumps(
            {
                "code": """
import json
import pandas as pd
df = pd.read_csv('/input/data.csv')
selected = df.loc[df['region'] == 'east', ['sales']]
json.dump({'answer': str(int(selected['sales'].sum())), 'table': selected.to_dict('records')}, open('/output/result.json', 'w'))
"""
            }
        )
    )
    result = analyze(
        b"region,sales\neast,10\nwest,20\neast,15\n",
        "east sales",
        model=model,
        on_status=statuses.append,
    )
    assert result.status == "completed", result.error
    assert result.answer == "25"
    assert result.table == [{"sales": 10}, {"sales": 15}]
    assert statuses == ["running", "completed"]


def test_profile_matches_rows_actually_read_by_pandas():
    result = analyze(b"sales\n10\n   \n20\n", "sum", model=ReplyModel("invalid"))
    assert result.profile["rows"] == 2
    assert result.profile["sample"] == [["10"], ["20"]]


def test_whitespace_only_data_is_rejected_before_model_generation():
    model = ReplyModel('{"status":"insufficient_input","reason":"wrong path"}')
    result = analyze(b"sales\n   \n", "sum", model=model)
    assert result.status == "failed"
    assert result.error == "CSV contains no data rows"
    assert model.prompt is None
    assert result.execution is None


def test_null_bytes_are_rejected_instead_of_silently_truncating_data():
    model = ReplyModel("invalid")
    result = analyze(b"sales\n10\x00corrupted\n", "sum", model=model)
    assert result.status == "failed"
    assert "NUL" in result.error
    assert model.prompt is None
    assert result.profile == {}
