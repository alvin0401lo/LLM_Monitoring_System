import streamlit as st

from common import FEATURES, SENSOR_GUIDE, display_reading, get_dataset, get_predictor, initialize_state, inject_theme

inject_theme()
dataset = get_dataset()
predictor = get_predictor()
initialize_state(dataset, predictor)
result = st.session_state.result
impact_rows = result["feature_impact"]
labels = {feature: SENSOR_GUIDE[feature]["label"] for feature in FEATURES}

st.sidebar.markdown("## NORTHSTAR")
st.sidebar.caption("Industrial AI Monitoring")
st.sidebar.divider()
st.session_state.machine_id = st.sidebar.selectbox("Machine", ["M001", "M002", "M003", "M004"], index=["M001", "M002", "M003", "M004"].index(st.session_state.machine_id))
st.sidebar.divider()
st.sidebar.markdown("**System**")
st.sidebar.markdown(":material/check_circle: ML Model Online")
st.sidebar.markdown(":material/check_circle: Local Analysis Online")
st.sidebar.caption("Maintenance logs and equipment manuals are not connected.")

st.markdown(f'<div class="topbar"><div><div class="kicker">INDUSTRIAL AI MONITORING</div><h1>Overview</h1><p class="subtle">Machine {st.session_state.machine_id} · latest machine summary</p></div><div class="live-badge"><span></span> MONITORING</div></div>', unsafe_allow_html=True)

st.markdown(f'<div class="status-hero"><div><div class="status-hero-label">CURRENT MACHINE STATUS</div><div class="status-hero-value" style="color:{"#ff9b66" if result["status"] == "HIGH RISK" else "#f5c46b" if result["status"] == "MEDIUM RISK" else "#9de6b4"}">{result["status"]}</div></div><div><div class="status-hero-label">FAILURE RISK</div><div class="status-hero-risk">{result["failure_probability"]:.0%}</div></div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">SYSTEM PULSE</div>', unsafe_allow_html=True)
kpis = [("LAST UPDATE", st.session_state.get("last_update", "--:--:--"), "neutral"), ("MACHINE", st.session_state.machine_id, "neutral")]
for column, (label, value, tone) in zip(st.columns(2), kpis):
    column.markdown(f'<div class="kpi {tone}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">KEY SENSOR SUMMARY</div>', unsafe_allow_html=True)
sensor_columns = st.columns(5)
for column, feature in zip(sensor_columns, FEATURES):
    value, unit = display_reading(feature, result["readings"][feature])
    with column.container(border=True):
        st.markdown(f":material/{SENSOR_GUIDE[feature]['icon']}: **{SENSOR_GUIDE[feature]['label']}**")
        st.markdown(f'<div class="sensor-value">{value:.1f} <span>{unit}</span></div>', unsafe_allow_html=True)
        st.caption("Current reading")

st.markdown('<div class="section-label">CURRENT ISSUE</div>', unsafe_allow_html=True)
issue_left, issue_right = st.columns([1.15, 1])
with issue_left:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Primary signal to watch</div>', unsafe_allow_html=True)
        top_factors = " + ".join(labels[row["feature"]] for row in impact_rows[:2])
        st.markdown(f'<div class="action-title">{top_factors}</div>', unsafe_allow_html=True)
        st.caption("Most influential features in the current prediction. This is not a confirmed physical cause.")
with issue_right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">Latest AI suggestion</div>', unsafe_allow_html=True)
        if result["status"] == "HIGH RISK":
            suggestion = "Inspect tool condition and mechanical load before continuing operation."
        elif result["status"] == "MEDIUM RISK":
            suggestion = "Review the next readings and check whether risk continues to rise."
        else:
            suggestion = "No immediate action required. Continue routine monitoring."
        st.markdown(suggestion)
        st.caption("Deterministic local analysis · general recommendation only")
