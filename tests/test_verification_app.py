"""The normal application shows review evidence and the corrected answer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from test_app import submit_demo


@pytest.mark.docker
def test_application_shows_successful_wrong_calculation_and_reviewed_correction(
    monkeypatch,
):
    monkeypatch.setenv("CSV_AGENT_DEMO", "1")
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
    result = submit_demo(app, "Review: recompute an incorrect result")
    assert result.status == "completed", result.error
    assert result.table == [{"region": "east", "sales_cny": 25}]
    assert result.candidates[0].result.table == [{"total_sales_cny": 45}]
    assert result.selected_candidate == 3
    assert result.reviews[0].outcome == "needs_revision"
    assert any("Review candidate" in item.value for item in app.markdown)
    assert any(
        "filters_and_groups" in item.value and "needs_revision" in item.value
        for item in app.json
    )
    assert any(
        "selected_candidate" in item.value and "recomputed_from" in item.value
        for item in app.json
    )
