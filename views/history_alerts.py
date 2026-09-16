from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import FEATURES, display_reading, get_dataset, get_predictor, initialize_state, inject_theme

inject_theme()
dataset = get_dataset()
predictor = get_predictor()
initialize_state(dataset, predictor)

st.markdown('<div class="kicker">REVIEW / ACTION RECORD</div><h1>History & Alerts</h1><p class="subtle">Review detected risk and record the engineer response.</p>', unsafe_allow_html=True)

history = pd.DataFrame(st.session_state.history)
if history.empty:
    st.info("No monitoring events recorded yet.")
else:
    prediction_rows = []
    for _, row in history.iterrows():
        readings = {feature: row[feature] for feature in FEATURES}
        prediction = predictor.predict(readings)
        driver = prediction["feature_impact"][0]["feature"].replace(" [K]", "").replace(" [rpm]", "").replace(" [Nm]", "").replace(" [min]", "")
        prediction_rows.append({"timestamp": row.get("timestamp", "-"), "machine": row.get("machine_id", st.session_state.machine_id), "risk": prediction["failure_probability"], "status": prediction["status"], "trigger": driver})
    events = pd.DataFrame(prediction_rows)
    alerts = events[events["status"] != "LOW RISK"].copy()

    st.markdown('<div class="section-label">ALERT HISTORY</div>', unsafe_allow_html=True)
    if alerts.empty:
        st.success("No medium- or high-risk events in the recorded replay.")
    else:
        display_alerts = alerts.rename(columns={"timestamp": "Time", "machine": "Machine", "status": "Status", "risk": "Risk", "trigger": "Main trigger"})
        display_alerts["Risk"] = display_alerts["Risk"].map(lambda value: f"{value:.0%}")
        display_alerts["Action"] = "Pending"
        st.dataframe(display_alerts[["Time", "Machine", "Status", "Risk", "Main trigger", "Action"]], use_container_width=True, hide_index=True)

    st.markdown('<div class="section-label">HISTORICAL RISK TREND</div>', unsafe_allow_html=True)
    period = st.selectbox("Time range", ["All events", "Last 30 events", "Last 10 events"], label_visibility="collapsed")
    window = {"All events": len(events), "Last 30 events": 30, "Last 10 events": 10}[period]
    trend = events.tail(window)
    x_values = trend["timestamp"] if "timestamp" in trend.columns else list(range(1, len(trend) + 1))
    figure = go.Figure(go.Scatter(x=x_values, y=trend["risk"], mode="lines+markers", line=dict(color="#ff9b66"), name="Risk"))
    figure.add_hline(y=.7, line_dash="dash", line_color="#ff9b66", annotation_text="High risk")
    figure.update_layout(height=270, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="#10243a", plot_bgcolor="#10243a", font=dict(color="#91a7b9"), yaxis=dict(range=[0, 1], tickformat=".0%"), xaxis_title="Time", yaxis_title="Failure probability")
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="section-label">ACTION STATUS</div>', unsafe_allow_html=True)
action_columns = st.columns(4)
for column, status, icon in zip(action_columns, ["Pending", "Acknowledged", "Checked", "Resolved"], ["schedule", "visibility", "fact_check", "task_alt"]):
    with column.container(border=True):
        st.markdown(f":material/{icon}: **{status}**")
        count = len(alerts) if status == "Pending" and "alerts" in locals() else 0
        st.markdown(f'<div class="sensor-value">{count}</div>', unsafe_allow_html=True)

st.caption(f"Machine {st.session_state.machine_id} · last update {st.session_state.get('last_update', datetime.now().strftime('%H:%M:%S'))}")
