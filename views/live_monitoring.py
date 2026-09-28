import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    FEATURES,
    advance_replay,
    ensure_history_predictions,
    get_dataset,
    get_predictor,
    initialize_state,
    inject_theme,
    reset_replay,
    render_sensors,
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
        if st.button("Next sample", type="primary", width="stretch"):
            advance_replay(dataset, predictor)
            st.rerun()
    with control_b:
        if st.button("Reset stream", width="stretch"):
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

with st.expander("Alert thresholds and model interpretation"):
    st.caption("Sensor alerts use dataset quantiles, not approved safety limits. ML risk combines all five readings into one classification estimate.")


@st.fragment(run_every=interval if monitoring else None)
def render_live() -> None:
    if monitoring:
        sync_replay(dataset, predictor)
    result = st.session_state.result
    st.caption(f"Machine {st.session_state.machine_id} · last update {st.session_state.get('last_update', '--:--:--')}")

    prediction_a, prediction_b = st.columns(2)
    prediction_a.metric("Failure probability", f"{result['failure_probability']:.0%}", border=True)
    prediction_b.metric("Risk status", result["status"], border=True)
    st.markdown('<div class="section-label">CURRENT SENSORS</div>', unsafe_allow_html=True)
    render_sensors(result, dataset)

    ensure_history_predictions(predictor)
    history = pd.DataFrame(st.session_state.history).tail(900)
    risk_values = history["failure_probability"]
    sensor_left, sensor_right = st.columns([1.35, 1])
    with sensor_left:
        st.markdown('<div class="section-label">KEY SENSOR TREND</div>', unsafe_allow_html=True)
        st.caption("Selected signals shown for comparison only.")
        sensor_chart = go.Figure()
        for feature, label in [(FEATURES[0], "Air temp"), (FEATURES[2], "RPM"), (FEATURES[3], "Torque")]:
            values = history[feature]
            normalized = (values - values.min()) / (values.max() - values.min() or 1)
            sensor_chart.add_trace(go.Scatter(x=history["sample"], y=normalized, mode="lines+markers", name=label))
        sensor_chart.update_layout(height=290, margin=dict(l=12, r=12, t=45, b=35), paper_bgcolor="#111517", plot_bgcolor="#111517", font=dict(color="#b0bdc4"), legend=dict(orientation="h", y=1.15), xaxis_title="Replay sample", yaxis_title="Normalized level")
        st.plotly_chart(sensor_chart, width="stretch", config={"displayModeBar": False})
    with sensor_right:
        st.markdown('<div class="section-label">FAILURE RISK TREND</div>', unsafe_allow_html=True)
        st.caption("Recorded model estimates.")
        risk_chart = go.Figure(go.Scatter(x=history["sample"], y=risk_values, mode="lines+markers", line=dict(color="#ff9b66"), name="Risk"))
        risk_chart.add_hline(y=0.7, line_dash="dash", line_color="#ff9b66", annotation_text="High risk")
        risk_chart.update_layout(height=290, margin=dict(l=12, r=12, t=45, b=35), paper_bgcolor="#111517", plot_bgcolor="#111517", font=dict(color="#b0bdc4"), yaxis=dict(range=[0, 1], tickformat=".0%"), xaxis_title="Replay sample", yaxis_title="Failure probability")
        st.plotly_chart(risk_chart, width="stretch", config={"displayModeBar": False})



render_live()
