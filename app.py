import base64
import os

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from csv_agent.analysis import analyze
from csv_agent.config import ModelConfig, RunLimits, configuration_error_message
from csv_agent.demo import CASES, CSV, DemoModel

STATUS_LABELS = {
    "running": "Running",
    "completed": "Completed",
    "failed": "Failed",
    "timeout": "Timed out",
    "insufficient_input": "Insufficient input",
}

load_dotenv()
demo_mode = os.environ.get("CSV_AGENT_DEMO") == "1"
st.set_page_config(page_title="CSV Data Analysis Agent", page_icon="📊")
st.title("CSV Data Analysis Agent")
st.caption("Upload a CSV and submit a question to view results computed in isolated Python.")

try:
    model_config = (
        ModelConfig(
            endpoint="https://controlled.example/messages", name="controlled-demo"
        )
        if demo_mode
        else ModelConfig.from_env()
    )
    limits = RunLimits.from_env(exploration=not demo_mode)
    if demo_mode and limits.max_model_calls is None:
        limits = limits.model_copy(update={"max_model_calls": 5})
except ValidationError as exc:
    st.error("Invalid configuration: " + configuration_error_message(exc))
    st.stop()
except ValueError:
    st.error("Invalid configuration. Check model parameters and execution limits in .env.")
    st.stop()

if demo_mode:
    st.warning(
        "Demo: preset model instructions and sample sales data run in isolated Python. No model service is called; this does not measure model performance or usage."
    )

with st.expander("Current configuration"):
    st.json({"model": model_config.model_dump(), "limits": limits.model_dump()})

with st.form("analysis"):
    if demo_mode:
        demo_case = st.selectbox("Demo scenario", list(CASES))
        question = CASES[demo_case]
        st.write(question)
        st.code(CSV.decode(), language="text")
        upload = None
    else:
        upload = st.file_uploader("CSV file (UTF-8)", type=["csv"])
        question = st.text_area("Analysis question", placeholder="What are total sales?")
    submitted = st.form_submit_button("Start analysis", type="primary")

if submitted:
    if not demo_mode and (upload is None or not question.strip()):
        st.warning("Select a CSV and enter a question.")
    elif upload is not None and upload.size > limits.input_max_bytes:
        st.error("The file exceeds the input size limit.")
    else:
        if demo_mode:
            data = CSV
        else:
            assert upload is not None
            data = upload.getvalue()
        with st.status("Analysis in progress", expanded=True) as progress:
            result = analyze(
                data,
                question,
                config=model_config,
                limits=limits,
                model=DemoModel(demo_case) if demo_mode else None,
                on_step=lambda message: st.write(message),
                strategy="langgraph_resolve"
                if not demo_mode or demo_case == "Disagreement: investigate and select"
                else "langgraph_verify"
                if demo_case == "Review: recompute an incorrect result"
                else "langgraph_explore"
                if demo_case == "Explore data and submit"
                else "langgraph",
            )
            progress.update(
                label=f"Analysis: {STATUS_LABELS[result.status]}",
                state="complete" if result.status == "completed" else "error",
            )
        st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    st.caption(
        f"Task {result.run_id} · {STATUS_LABELS[result.status]} ({result.status}) · {result.seconds:.2f} seconds · Executions: {result.execution_attempts} · Model requests: {len(result.attempts)}"
    )
    if result.status == "completed":
        st.write(result.answer)
        if result.table:
            st.dataframe(result.table, use_container_width=True)
        for chart in result.charts:
            st.image(base64.b64decode(chart.data_base64), caption=chart.caption)
        st.caption("Completed means the code produced a valid result; the answer was not scored against a reference.")
    elif result.status == "insufficient_input":
        st.warning(result.error or "Insufficient input")
    elif result.status == "running":
        st.info("Analysis in progress")
    else:
        st.error(result.error or "Analysis failed")
    if result.diagnostic:
        st.caption(f"Diagnostic: {result.diagnostic}")
    if result.stop_reason:
        st.info(result.stop_reason)
    column_mapping = result.profile.get("column_mapping")
    if isinstance(column_mapping, list):
        renamed_columns = [
            f"Column {column['position']} → {column['name']}"
            for column in column_mapping
            if isinstance(column, dict) and column.get("original") != column.get("name")
        ]
        if renamed_columns:
            st.info(
                "Empty column names were mapped by position: "
                + "；".join(renamed_columns)
                + ". Unnamed column data is retained and is not treated as an index."
            )
    with st.expander("Data profile, code and execution evidence", expanded=True):
        st.json(result.profile)
        st.code(result.code, language="python")
        if result.execution:
            st.json(result.execution.model_dump())
        st.write("Model usage")
        st.json(result.usage.model_dump() if result.usage else {"usage": "Not reported by provider"})

    if result.attempts:
        if result.observations or result.candidates:
            with st.expander("Data exploration and candidates", expanded=True):
                st.json(
                    {
                        "observations": [
                            item.model_dump() for item in result.observations
                        ],
                        "candidates": [
                            item.model_dump(exclude={"charts"})
                            for item in result.candidates
                        ],
                        "reviews": [item.model_dump() for item in result.reviews],
                        "selected_candidate": result.selected_candidate,
                        "comparisons": [
                            item.model_dump() for item in result.comparisons
                        ],
                        "selection": result.selection.model_dump()
                        if result.selection
                        else None,
                        "limitations": result.limitations,
                    }
                )
        with st.expander("Request history", expanded=True):
            for attempt in result.attempts:
                st.write(f"Model request {attempt.number} · {attempt.seconds:.2f} seconds")
                if attempt.action:
                    st.write(f"Stage: {attempt.action}")
                if attempt.model_seconds is not None:
                    st.write(f"Model request time {attempt.model_seconds:.2f} seconds")
                if attempt.request:
                    st.json(attempt.request.model_dump())
                if attempt.diagnostic:
                    st.write(f"Diagnostic: {attempt.diagnostic}")
                if attempt.code:
                    st.code(attempt.code, language="python")
                if attempt.error:
                    st.write(attempt.error)
                if attempt.execution:
                    st.json(attempt.execution.model_dump())
                st.json(
                    attempt.usage.model_dump()
                    if attempt.usage
                    else {"usage": "Not reported by provider"}
                )
