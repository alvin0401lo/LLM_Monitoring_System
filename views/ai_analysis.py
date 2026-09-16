import streamlit as st

from common import SOURCE_CATALOG, display_reading, get_dataset, get_predictor, initialize_state, inject_theme

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
key_columns[0].markdown(f'<div class="kpi"><div class="kpi-label">RISK STATUS</div><div class="kpi-value" style="color:{status_color}">{result["status"]}</div></div>', unsafe_allow_html=True)
key_columns[1].markdown(f'<div class="kpi accent"><div class="kpi-label">FAILURE PROBABILITY</div><div class="kpi-value">{result["failure_probability"]:.0%}</div></div>', unsafe_allow_html=True)
key_columns[2].markdown(f'<div class="kpi"><div class="kpi-label">MAIN DRIVER</div><div class="kpi-value" style="font-size:1.25rem">{top_label}</div></div>', unsafe_allow_html=True)

left, right = st.columns([1.2, 1])
with left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Recommended next step</div>', unsafe_allow_html=True)
        if result["status"] == "HIGH RISK":
            action = "Inspect the tool condition and torque transmission before continuing operation."
        elif result["status"] == "MEDIUM RISK":
            action = "Review the next readings and inspect the tool if the risk continues rising."
        else:
            action = "Continue monitoring and compare the next readings with the normal range."
        st.markdown(f'<div class="action-title">{action}</div>', unsafe_allow_html=True)
        st.caption("General recommendation only. Confirm actions against qualified site procedures.")
with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Why this result?</div>', unsafe_allow_html=True)
        st.markdown(f"**Main contributing factor:** {top_label}")
        other_factors = ", ".join(labels[row["feature"]] for row in impact_rows[1:3])
        st.markdown(f"**Other contributing readings:** {other_factors}")
        st.caption(f"Overall predicted failure risk is {result['failure_probability']:.0%}. Model sensitivity is not a confirmed physical cause.")

with st.expander("References used for this analysis", expanded=False):
    st.caption("Only these three inputs produced the current output.")
    st.markdown("- **AI4I dataset** · current sensor row and training distribution")
    st.markdown("- **Random Forest model** · failure probability and risk driver")
    st.markdown("- **Local analysis rules** · general inspection suggestion")

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
    st.caption("Maintenance logs and equipment manuals are not connected. This output does not cite them.")

