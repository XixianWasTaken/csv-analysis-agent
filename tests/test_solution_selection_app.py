"""The normal application makes disagreement and selection evidence reviewable."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_app import submit_demo


@pytest.mark.docker
def test_application_shows_differing_fields_and_data_supported_selection(monkeypatch):
    monkeypatch.setenv("CSV_AGENT_DEMO", "1")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    result = submit_demo(app, "Disagreement: investigate and select")
    assert result.status == "completed", result.error
    assert result.answer_fields == {"total": 25, "count": 2}
    assert result.candidates[0].result.answer_fields == {"total": 45, "count": 3}
    assert result.selection.candidate == 2
    assert result.selection.evidence == [2, 4]
    assert result.execution_attempts == 3
    assert any(
        "comparisons" in item.value
        and "resolutions" in item.value
        and "selection" in item.value
        for item in app.json
    )
