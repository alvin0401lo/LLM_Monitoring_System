import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    FEATURES,
    SENSOR_GUIDE,
    advance_replay,
    alert_state,
    alert_threshold,
    display_reading,
    get_dataset,
    get_predictor,
    initialize_state,
    inject_theme,
    reset_replay,
    sensor_state,
    sync_replay,
)

inject_theme()
dataset = get_dataset()
predictor = get_predictor()
initialize_state(dataset, predictor)

st.markdown('<div class="kicker">LIVE STREAM / LINE 01</div><h1>Live monitoring</h1><p class="subtle">Watch the simulated sensor stream and risk trend.</p>', unsafe_allow_html=True)

with st.expander("Demo controls", expanded=False):
    control_a, control_b, control_c, control_d = st.columns([1, 1, 1.1, 1.3])
    with control_a:
        if st.button("Next sample", type="primary", use_container_width=True):
            advance_replay(dataset, predictor)
            st.rerun()
    with control_b:
        if st.button("Reset stream", use_container_width=True):
            reset_replay(dataset, predictor)
            st.rerun()
    with control_c:
        monitoring = st.toggle("Auto monitoring", value=st.session_state.monitoring_enabled, key="live_monitoring_toggle")
        st.session_state.monitoring_enabled = monitoring
    with control_d:
        interval = st.select_slider("Update interval", options=[1, 2, 3, 5], value=st.session_state.replay_interval, format_func=lambda value: f"Every {value}s")
        st.session_state.replay_interval = interval

if "monitoring" not in locals():
    monitoring = st.session_state.monitoring_enabled
    interval = st.session_state.replay_interval

st.info("Sensor alert checks one reading against a threshold. ML risk combines all five readings into one failure probability.")


@st.fragment(run_every=interval if monitoring else None)
def render_live() -> None:
    if monitoring:
        sync_replay(dataset, predictor)
    result = st.session_state.result
    st.caption(f"Machine {st.session_state.machine_id} · last update {st.session_state.get('last_update', '--:--:--')}")

    st.markdown('<div class="section-label">CURRENT SENSORS</div>', unsafe_allow_html=True)
    sensor_columns = st.columns(5)
    for column, feature in zip(sensor_columns, FEATURES):
        guide = SENSOR_GUIDE[feature]
        value = result["readings"][feature]
        shown_value, unit = display_reading(feature, value)
        alarm = alert_state(dataset, feature, value)
        threshold, direction = alert_threshold(dataset, feature)
        threshold_value, threshold_unit = display_reading(feature, threshold)
        with column.container(border=True):
            st.markdown(f":material/{guide['icon']}: **{guide['label']}**")
            st.markdown(f'<div class="sensor-value">{shown_value:.1f} <span>{unit}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alert-state {"alert" if alarm == "ALERT" else "normal"}">{alarm}</div>', unsafe_allow_html=True)
            st.caption(f"Alert {direction} {threshold_value:.1f} {threshold_unit}")

    history = pd.DataFrame(st.session_state.history)
    risk_values = [predictor.predict({feature: row[feature] for feature in FEATURES})["failure_probability"] for _, row in history.iterrows()]
    sensor_left, sensor_right = st.columns([1.35, 1])
    with sensor_left:
        st.markdown('<div class="section-label">KEY SENSOR TREND</div>', unsafe_allow_html=True)
        st.caption("Selected signals shown for comparison only.")
        sensor_chart = go.Figure()
        for feature, label in [(FEATURES[0], "Air temp"), (FEATURES[2], "RPM"), (FEATURES[3], "Torque")]:
            values = history[feature]
            normalized = (values - values.min()) / (values.max() - values.min() or 1)
            sensor_chart.add_trace(go.Scatter(x=history["sample"], y=normalized, mode="lines+markers", name=label))
        sensor_chart.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#10243a", plot_bgcolor="#10243a", font=dict(color="#91a7b9"), legend=dict(orientation="h", y=1.12), xaxis_title="Time", yaxis_title="Normalized level")
        st.plotly_chart(sensor_chart, use_container_width=True, config={"displayModeBar": False})
    with sensor_right:
        st.markdown('<div class="section-label">FAILURE RISK TREND</div>', unsafe_allow_html=True)
        risk_chart = go.Figure(go.Scatter(x=history["sample"], y=risk_values, mode="lines+markers", line=dict(color="#ff9b66"), name="Risk"))
        risk_chart.add_hline(y=0.7, line_dash="dash", line_color="#ff9b66", annotation_text="High risk")
        risk_chart.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#10243a", plot_bgcolor="#10243a", font=dict(color="#91a7b9"), yaxis=dict(range=[0, 1], tickformat=".0%"), xaxis_title="Sample", yaxis_title="Failure probability")
        st.plotly_chart(risk_chart, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="section-label">CURRENT PREDICTION</div>', unsafe_allow_html=True)
    prediction_a, prediction_b = st.columns(2)
    prediction_a.metric("Failure risk", f"{result['failure_probability']:.0%}")
    prediction_b.metric("Status", result["status"])


render_live()
