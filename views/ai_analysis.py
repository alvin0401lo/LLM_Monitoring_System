import re

import streamlit as st

from common import SOURCE_CATALOG, display_reading, get_dataset, get_predictor, initialize_state, inject_theme
from llm import local_explain, machine_context, ollama_settings
from rag import prepare_context, retrieve
from source_view import show_sources

inject_theme()
dataset = get_dataset()
predictor = get_predictor()
initialize_state(dataset, predictor)
result = st.session_state.result
impact_rows = result["feature_impact"]
labels = {
    "Air temperature [K]": "Air temperature",
    "Process temperature [K]": "Process temperature",
    "Rotational speed [rpm]": "Rotational speed",
    "Torque [Nm]": "Torque",
    "Tool wear [min]": "Tool wear",
}
top_driver = impact_rows[0]
top_label = labels[top_driver["feature"]]
status_color = "#ff9b66" if result["status"] == "HIGH RISK" else "#f5c46b" if result["status"] == "MEDIUM RISK" else "#9de6b4"

st.markdown('<div class="kicker">OPERATOR SUMMARY / CURRENT READING</div><h1>AI analysis</h1><p class="subtle">Only the information needed for the next maintenance decision.</p>', unsafe_allow_html=True)

st.markdown('<div class="section-label">DECISION SNAPSHOT</div>', unsafe_allow_html=True)
key_columns = st.columns(3)
key_columns[0].markdown(f'<div class="kpi accent"><div class="kpi-label">FAILURE PROBABILITY</div><div class="kpi-value">{result["failure_probability"]:.0%}</div></div>', unsafe_allow_html=True)
key_columns[1].markdown(f'<div class="kpi"><div class="kpi-label">RISK STATUS</div><div class="kpi-value" style="color:{status_color}">{result["status"]}</div></div>', unsafe_allow_html=True)
key_columns[2].markdown(f'<div class="kpi"><div class="kpi-label">MAIN DRIVER</div><div class="kpi-value" style="font-size:1.6rem">{top_label}</div></div>', unsafe_allow_html=True)

def render_rule_summary() -> None:
    """Keep secondary rule-based guidance separate from the LLM analysis."""
    with st.expander("Rule-based guidance and model sensitivity"):
        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("**Rule-based next step**")
            if result["status"] == "HIGH RISK":
                action = "Inspect the tool condition and torque transmission before continuing operation."
            elif result["status"] == "MEDIUM RISK":
                action = "Review the next readings and inspect the tool if the risk continues rising."
            else:
                action = "Continue monitoring and compare the next readings with the normal range."
            st.markdown(action)
            st.caption("General recommendation only. Confirm actions against qualified site procedures.")
        with right:
            st.markdown(f"**Main contributing factor:** {top_label}")
            other_factors = ", ".join(labels[row["feature"]] for row in impact_rows[1:3])
            st.markdown(f"**Other contributing readings:** {other_factors}")
            st.caption("Model sensitivity is not a confirmed physical cause.")


def display_analysis(answer: str) -> None:
    """Promote standalone section labels without changing the stored answer."""
    sections = r"Observed condition|Possible explanation|Suggested checks|Condition|Possible factors"
    st.markdown(re.sub(rf"(?m)^\*\*({sections})\*\*[ \t]*$", r"### \1", answer))

@st.fragment(run_every="5s")
def render_llm_analysis() -> None:
    st.markdown('<div class="section-label">LOCAL LLM ANALYSIS</div>', unsafe_allow_html=True)
    settings = ollama_settings()
    record = st.session_state.get("analysis_records", {}).get(st.session_state.machine_id, {})
    job = st.session_state.get("analysis_job")
    if job and job["machine_id"] == st.session_state.machine_id:
        st.info("Generating an Ollama analysis for the captured readings.")
    if record.get("error"):
        st.warning(record["error"])
    if record.get("content"):
        snapshot = record["context"]
        st.caption(f"Ollama / {record['model']} | Machine {snapshot['machine_id']} | "
                   f"Generated {record['generated_at']}")
        st.caption(f"Data snapshot: {snapshot['data_time']} | Snapshot risk: "
                   f"{snapshot['failure_probability']:.0%} | {snapshot['history']['sample_count']} replay observations")
        display_analysis(record["content"])
    else:
        if not settings["host"] or not settings["model"]:
            st.info("Ollama is not configured. Showing local rule-based analysis.")
        elif not job and not record.get("error"):
            st.info("Waiting for the first Ollama analysis.")
        st.caption("Local rule-based fallback")
        display_analysis(local_explain(st.session_state.result))
    with st.expander("Analysis evidence", expanded=False):
        st.markdown("- **AI4I synthetic replay**: readings and available recent replay summary")
        st.markdown("- **Random Forest**: estimated failure probability and feature sensitivity")
        if record.get("content"):
            show_sources(record["context"].get("retrieved_documents", []), record.get("citations"))
        else:
            preview = prepare_context(machine_context(st.session_state.result, st.session_state.history, st.session_state.machine_id))
            st.caption("Retrieval preview only: these excerpts have not been analysed by Ollama.")
            show_sources(preview["retrieved_documents"])


render_llm_analysis()
render_rule_summary()

with st.expander("Search the synthetic reference library", expanded=False):
    with st.form("reference_search"):
        query = st.text_input("Reference question", placeholder="Temperature rise and cooling checks")
        searched = st.form_submit_button("Search references", icon=":material/search:")
    if searched and query.strip():
        snapshot = machine_context(st.session_state.result, st.session_state.history, st.session_state.machine_id)
        show_sources(retrieve(query, snapshot["machine_id"], snapshot["data_time"]))

with st.expander("Available source files", expanded=False):
    for name, state, detail in SOURCE_CATALOG:
        st.markdown(f"- **{name}** · `{state}`")
        st.caption(detail)

with st.expander("Details and limitations", expanded=False):
    st.markdown(f"**Current driver:** {top_label}")
    for row in impact_rows[:3]:
        direction = "raises" if row["probability_change"] >= 0 else "reduces"
        st.markdown(f"- {labels[row['feature']]} {direction} estimated risk by {abs(row['probability_change']):.0%}; global weight {row['importance']:.0%}.")
    st.markdown(st.session_state.explanation)
    st.caption("Rule-based output above does not use documents. Ollama receives only the retrieved excerpts shown with its captured snapshot.")

