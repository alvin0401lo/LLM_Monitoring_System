import streamlit as st

from common import FEATURES, SENSOR_GUIDE, get_dataset, get_predictor, initialize_state, inject_theme, render_sensors

inject_theme()
dataset = get_dataset()
predictor = get_predictor()
initialize_state(dataset, predictor)
result = st.session_state.result
impact_rows = result["feature_impact"]
labels = {feature: SENSOR_GUIDE[feature]["label"] for feature in FEATURES}

stream_status = "REPLAY ACTIVE" if st.session_state.monitoring_enabled else "REPLAY PAUSED"
st.markdown(f'<div class="topbar"><div><div class="kicker">INDUSTRIAL AI MONITORING</div><h1>Overview</h1><p class="subtle">Machine {st.session_state.machine_id} · Updated {st.session_state.last_update}</p></div><div class="live-badge"><span></span> {stream_status}</div></div>', unsafe_allow_html=True)

tone = "high" if result["status"] == "HIGH RISK" else "medium" if result["status"] == "MEDIUM RISK" else "low"
st.markdown(f'<div class="status-hero {tone}"><div><div class="status-hero-label">CURRENT MACHINE STATUS</div><div class="status-hero-value" style="color:{"#ff9b66" if tone == "high" else "#f5c46b" if tone == "medium" else "#9de6b4"}">{result["status"]}</div></div><div><div class="status-hero-label">FAILURE PROBABILITY</div><div class="status-hero-risk">{result["failure_probability"]:.0%}</div></div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">KEY SENSOR SUMMARY</div>', unsafe_allow_html=True)
render_sensors(result)

st.markdown('<div class="section-label">CURRENT ISSUE</div>', unsafe_allow_html=True)
issue_left, issue_right = st.columns([1.15, 1])
with issue_left:
    with st.container():
        st.markdown('<div class="panel-title">Primary signal to watch</div>', unsafe_allow_html=True)
        top_factors = " + ".join(labels[row["feature"]] for row in impact_rows[:2])
        st.markdown(f'<div class="action-title">{top_factors}</div>', unsafe_allow_html=True)
        st.caption("Most influential features in the current prediction. This is not a confirmed physical cause.")
with issue_right:
    with st.container():
        st.markdown('<div class="panel-title">Recommended next step</div>', unsafe_allow_html=True)
        if result["status"] == "HIGH RISK":
            suggestion = "Inspect tool condition and mechanical load before continuing operation."
        elif result["status"] == "MEDIUM RISK":
            suggestion = "Review the next readings and check whether risk continues to rise."
        else:
            suggestion = "No immediate action required. Continue routine monitoring."
        st.markdown(f'<div class="action-title">{suggestion}</div>', unsafe_allow_html=True)
        st.caption("Deterministic local analysis · general recommendation only")
