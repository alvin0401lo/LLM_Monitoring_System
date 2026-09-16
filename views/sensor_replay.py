import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import advance_replay, alert_state, alert_threshold, apply_fault_scenario, display_reading, FEATURES, get_dataset, get_predictor, initialize_state, inject_theme, reset_replay, SENSOR_GUIDE, sensor_state, sync_replay

inject_theme()
dataset = get_dataset()
predictor = get_predictor()
initialize_state(dataset, predictor)
st.markdown('<div class="kicker">DATA STREAM / LINE 01</div><h1>Sensor replay</h1><p class="subtle">Replay dataset rows as simulated machine readings, manually or automatically.</p>', unsafe_allow_html=True)
with st.expander("How do I read these sensors?", expanded=True):
    st.markdown("**TYPICAL** means the value is inside the middle 80% of this dataset. **HIGH** and **LOW** describe the sensor distribution. **ALERT** means the value crossed the displayed demo threshold. The Random Forest combines all five readings to estimate failure risk.")

st.markdown('<div class="section-label">FAULT DEMO</div>', unsafe_allow_html=True)
fault_column, fault_action = st.columns([2.2, 1])
with fault_column:
    scenario = st.selectbox("Inject a controlled scenario", ["High torque / mechanical load", "Excessive tool wear", "Overheating"], label_visibility="collapsed")
with fault_action:
    if st.button("Inject fault", type="secondary", use_container_width=True):
        apply_fault_scenario(predictor, scenario)
        st.rerun()
if st.session_state.get("fault_scenario"):
    st.warning(f"Injected demo scenario: {st.session_state.fault_scenario}. This is simulated data, not a real sensor event.")

control_a, control_b, control_c, control_d = st.columns([1, 1, 1.05, 1.4])
with control_a:
    if st.button("Replay next sample", type="primary", use_container_width=True):
        advance_replay(dataset, predictor)
        st.rerun()
with control_b:
    if st.button("Reset replay", use_container_width=True):
        reset_replay(dataset, predictor)
        st.rerun()
with control_c:
    auto_play = st.toggle("Auto monitoring", value=st.session_state.monitoring_enabled, key="auto_replay", help="Automatically process the next dataset row at the selected interval.")
    st.session_state.monitoring_enabled = auto_play
with control_d:
    interval = st.select_slider("Interval", options=[1, 2, 3, 5], value=2, format_func=lambda value: f"Every {value}s")
    st.session_state.replay_interval = interval


@st.fragment(run_every=interval if auto_play else None)
def render_replay() -> None:
    if auto_play:
        sync_replay(dataset, predictor)
    st.caption(f"Dataset row {st.session_state.sample_index + 1} of {len(dataset)} · {'playing automatically' if auto_play else 'paused'}")

    result = st.session_state.result
    st.markdown('<div class="section-label">CURRENT SENSORS</div>', unsafe_allow_html=True)
    sensor_columns = st.columns(5)
    for column, feature in zip(sensor_columns, FEATURES):
        guide = SENSOR_GUIDE[feature]
        value = result["readings"][feature]
        state = sensor_state(dataset, feature, value)
        shown_value, shown_unit = display_reading(feature, value)
        threshold, direction = alert_threshold(dataset, feature)
        threshold_value, threshold_unit = display_reading(feature, threshold)
        alarm = alert_state(dataset, feature, value)
        with column.container(border=True):
            st.markdown(f":material/{guide['icon']}: **{guide['label']}**")
            st.markdown(f'<div class="sensor-value">{shown_value:.1f} <span>{shown_unit}</span></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="sensor-state {state.lower()}">{state}</div>', unsafe_allow_html=True)
            st.caption(guide["meaning"])
            st.markdown(f'<div class="alert-state {"alert" if alarm == "ALERT" else "normal"}">{alarm}</div><div class="threshold">Alert {direction} {threshold_value:.1f} {threshold_unit}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">SIGNAL HISTORY</div>', unsafe_allow_html=True)
    chart_data = pd.DataFrame(st.session_state.history)
    fig = go.Figure()
    for feature, label in [(FEATURES[0], "Air temp"), (FEATURES[2], "RPM"), (FEATURES[3], "Torque")]:
        values = chart_data[feature]
        normalized = (values - values.min()) / (values.max() - values.min() or 1)
        fig.add_trace(go.Scatter(x=chart_data["sample"], y=normalized, mode="lines+markers", name=label))
    fig.update_layout(height=235, margin=dict(l=0, r=0, t=15, b=0), paper_bgcolor="#10243a", plot_bgcolor="#10243a", font=dict(color="#91a7b9"), legend=dict(orientation="h", y=1.12), xaxis_title="Replay sample", yaxis_title="Relative level (normalized)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


render_replay()
