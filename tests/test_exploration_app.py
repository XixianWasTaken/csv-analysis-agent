"""The application exposes inspection progress and its computed result."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_app import submit_demo


@pytest.mark.docker
def test_application_shows_inspection_candidate_and_submission(monkeypatch):
    monkeypatch.setenv("CSV_AGENT_DEMO", "1")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    assert not app.exception
    result = submit_demo(app, "Explore data and submit")
    assert result.status == "completed", result.error
    assert result.table == [{"total_sales_cny": 45}]
    assert len(result.observations) == 1
    assert len(result.candidates) == 1
    assert result.execution_attempts == 2
    assert any("Data exploration" in item.value for item in app.markdown)
    assert any("Final submission" in item.value for item in app.markdown)
    assert any("sales_missing" in item.value for item in app.json)
