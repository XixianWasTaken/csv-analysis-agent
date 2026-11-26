from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from csv_agent.analysis import AnalysisResult, ChartArtifact


@pytest.fixture
def demo_app(monkeypatch):
    monkeypatch.setenv("CSV_AGENT_DEMO", "1")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not app.exception
    return app


def submit_demo(app, case):
    app.selectbox[0].select(case)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    return app.session_state["result"]


@pytest.mark.docker
def test_demo_submission_computes_sales_in_the_real_application(demo_app):
    app = demo_app
    assert "Demo" in app.warning[0].value
    result = submit_demo(app, "Aggregation: total sales")
    assert result.status == "completed", result.error
    assert result.table == [{"total_sales_cny": 45}]
    assert result.execution_attempts == 1
    assert result.usage is None
    assert result.attempts[0].request is None


@pytest.mark.docker
def test_demo_recovery_shows_the_failed_attempt_and_computed_answer(demo_app):
    app = demo_app
    result = submit_demo(app, "Repair: recover from initial failure")
    assert result.status == "completed", result.error
    assert result.table == [{"total_sales_cny": 45}]
    assert result.execution_attempts == 2
    assert "demo-first-attempt" in result.attempts[0].error
    assert any("demo-first-attempt" in item.value for item in app.json)


@pytest.mark.docker
def test_demo_limit_is_a_visible_failure_and_the_next_task_has_no_stale_answer(
    demo_app,
):
    app = demo_app
    submit_demo(app, "Aggregation: total sales")
    result = submit_demo(app, "Stop: execution limit reached")
    assert result.status == "failed"
    assert result.execution_attempts == 3
    assert result.answer is None
    assert "execution limit" in result.stop_reason
    assert not app.dataframe
    assert not app.get("image")
    assert any("demo-always-fails" in item.value for item in app.error)


def test_demo_insufficient_input_is_explicit_and_never_executes(demo_app):
    app = demo_app
    result = submit_demo(app, "Insufficient input: missing cost")
    assert result.status == "insufficient_input"
    assert result.execution_attempts == 0
    assert any("Missing cost" in item.value for item in app.warning)


@pytest.mark.docker
def test_demo_filter_uses_only_east_rows(demo_app):
    result = submit_demo(demo_app, "Filter: east sales")
    assert result.status == "completed", result.error
    assert result.table == [{"region": "east", "sales_cny": 25}]


@pytest.mark.docker
def test_demo_chart_is_collected_and_rendered_from_real_computation(demo_app):
    app = demo_app
    result = submit_demo(app, "Chart: sales by region")
    assert result.status == "completed", result.error
    assert result.table == [
        {"region": "east", "sales_cny": 25},
        {"region": "west", "sales_cny": 20},
    ]
    assert len(result.charts) == 1
    assert app.get("image")[0].proto.imgs[0].caption == "Total sales by region (CNY)"


def test_ui_opens_and_renders_success_and_failure(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://model.example/v1/messages")
    monkeypatch.setenv("MODEL_NAME", "test")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not app.exception
    assert app.button[0].label == "Start analysis"

    app.session_state["result"] = AnalysisResult(
        run_id="ui-example",
        status="completed",
        answer="Total sales 45",
        table=[{"total": 45, "missing": None, "values": "[10,20,15]"}],
        code="total = df.sales.sum()",
    )
    app.run()
    assert not app.exception
    assert any("Total sales 45" in item.value for item in app.markdown)
    assert app.dataframe[0].value.to_dict("records") == [
        {"total": 45, "missing": None, "values": "[10,20,15]"}
    ]
    assert app.code[0].value == "total = df.sales.sum()"

    app.session_state["result"] = AnalysisResult(
        run_id="ui-failed",
        status="timeout",
        error="Python execution timed out",
    )
    app.run()
    assert not app.exception
    assert app.error[0].value == "Python execution timed out"
    assert not app.dataframe


def test_ui_shows_input_limitation_and_profile(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://model.example/v1/messages")
    monkeypatch.setenv("MODEL_NAME", "test")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    app.session_state["result"] = AnalysisResult(
        run_id="insufficient",
        status="insufficient_input",
        error="Missing cost field; cannot calculate profit.",
        profile={
            "columns": ["sales"],
            "rows": 1,
            "types": {"sales": "int64"},
            "missing": {"sales": 0},
            "sample": [["10"]],
        },
    )
    app.run()
    assert not app.exception
    assert app.warning[0].value == "Missing cost field; cannot calculate profit."
    assert any("Insufficient input" in item.value for item in app.caption)
    assert any(
        '"sales"' in item.value and '"missing"' in item.value for item in app.json
    )
    assert not app.dataframe


def test_ui_explains_unnamed_columns_alongside_computed_result(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://model.example/v1/messages")
    monkeypatch.setenv("MODEL_NAME", "test")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    app.session_state["result"] = AnalysisResult(
        run_id="unnamed",
        status="completed",
        answer="First column total 45",
        table=[{"Unnamed: 0": 45}],
        profile={
            "column_mapping": [
                {"position": 1, "original": "", "name": "Unnamed: 0"},
                {"position": 2, "original": "sales", "name": "sales"},
            ]
        },
    )
    app.run()
    assert not app.exception
    assert any(
        "Column 1" in item.value and "Unnamed: 0" in item.value for item in app.info
    )
    assert any("Unnamed column data is retained" in item.value for item in app.info)
    assert app.dataframe[0].value.to_dict("records") == [{"Unnamed: 0": 45}]


def test_ui_exposes_attempt_feedback_and_stop_reason(monkeypatch):
    from csv_agent.analysis import AnalysisAttempt
    from csv_agent.execution import Execution

    monkeypatch.setenv("MODEL_ENDPOINT", "https://model.example/v1/messages")
    monkeypatch.setenv("MODEL_NAME", "test")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    app.session_state["result"] = AnalysisResult(
        run_id="limited",
        status="failed",
        error="bad column",
        execution_attempts=1,
        stop_reason="Analysis code execution limit reached",
        attempts=[
            AnalysisAttempt(
                number=1,
                code="raise ValueError('bad column')",
                error="bad column",
                seconds=0.5,
                execution=Execution(status="completed", output="traceback evidence"),
            )
        ],
    )
    app.run()
    assert not app.exception
    assert any("Executions: 1" in item.value for item in app.caption)
    assert any("execution limit" in item.value for item in app.info)
    assert any("raise ValueError" in item.value for item in app.code)
    assert any("traceback evidence" in item.value for item in app.json)


def test_ui_displays_chart_from_shared_result_and_clears_it_on_next_task(monkeypatch):
    monkeypatch.setenv("MODEL_ENDPOINT", "https://model.example/v1/messages")
    monkeypatch.setenv("MODEL_NAME", "test")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    app.session_state["result"] = AnalysisResult(
        run_id="chart-task",
        status="completed",
        answer="east: 25 CNY; west: 20 CNY",
        table=[
            {"region": "east", "sales_cny": 25},
            {"region": "west", "sales_cny": 20},
        ],
        charts=[
            ChartArtifact(
                path="comparison.png",
                caption="Total sales by region (CNY)",
                data_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
            )
        ],
        code='totals = df.groupby("region")["sales"].sum()',
    )
    app.run()
    assert not app.exception
    assert app.dataframe[0].value.to_dict("records") == [
        {"region": "east", "sales_cny": 25},
        {"region": "west", "sales_cny": 20},
    ]
    assert app.get("image")[0].proto.imgs[0].caption == "Total sales by region (CNY)"
    assert app.code[0].value == 'totals = df.groupby("region")["sales"].sum()'
    app.session_state["result"] = AnalysisResult(
        run_id="next-task", status="failed", error="failed"
    )
    app.run()
    assert not app.exception
    assert not app.get("image")
