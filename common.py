import pandas as pd
import streamlit as st
import time
from datetime import datetime

from predictor import MachinePredictor
from train import FEATURES, ensure_dataset


def inject_theme() -> None:
    st.set_page_config(page_title="Northstar Monitoring", page_icon="N", layout="wide")
    st.markdown("""
    <style>
    :root { --panel:#1b2226; --line:#354047; --text:#edf2f4; --muted:#b0bdc4;
            --cyan:#79ddce; --green:#9de6b4; --orange:#ff9b66; }
    .block-container { max-width:1440px; padding:2.8rem 2rem 3rem; }
    .stMain h1 { font-size:2rem; line-height:1.25; margin:0 0 .4rem; color:var(--text); }
    .stMain h2 { font-size:1.6rem; } .stMain h3 { font-size:1.3rem; }
    .topbar { display:flex; justify-content:space-between; align-items:center; gap:1rem;
              border-bottom:1px solid var(--line); padding-bottom:1rem; margin-bottom:.4rem; }
    .kicker, .section-label { color:var(--cyan); font-size:1rem; font-weight:700; letter-spacing:0; }
    .kicker { margin-bottom:.35rem; } .section-label { margin:1rem 0 .3rem; }
    .subtle, .microcopy { color:var(--muted); font-size:.9rem; line-height:1.5; margin:0; }
    .live-badge { border:1px solid var(--line); color:var(--cyan); padding:.4rem .7rem;
                  font-size:.75rem; border-radius:4px; white-space:nowrap; }
    .live-badge span { display:inline-block; width:6px; height:6px; background:var(--green);
                       border-radius:50%; margin-right:.4rem; }
    .kpi { background:var(--panel); border:1px solid var(--line); border-radius:6px;
           min-height:110px; box-sizing:border-box; padding:1rem; }
    .kpi-label, .panel-title { color:var(--muted); font-size:.95rem; font-weight:650; letter-spacing:0; }
    .kpi-value { color:var(--text); font-size:2.1rem; font-weight:700; margin-top:.5rem;
                 line-height:1.25; overflow-wrap:anywhere; }
    .kpi.accent { border-top:3px solid var(--cyan); } .kpi.accent .kpi-value { color:var(--cyan); }
    .panel-title { margin-bottom:.65rem; }
    .sensor-value { color:var(--text); font-size:2.1rem; font-weight:700; line-height:1.35;
                    display:flex; flex-wrap:wrap; align-items:baseline; gap:.35rem; }
    .sensor-value span { color:var(--muted); font-size:.85rem; font-weight:400; }
    .alert-state { font-size:.75rem; font-weight:650; margin-top:.15rem; }
    .alert-state.normal { color:var(--green); } .alert-state.alert { color:var(--orange); }
    .action-title { font-size:1.35rem; line-height:1.5; font-weight:700; color:var(--text); overflow-wrap:anywhere; }
    .status-hero { background:var(--panel); border:1px solid var(--line); border-left:4px solid var(--green);
                   border-radius:6px; padding:1.2rem 1.5rem; display:flex; justify-content:space-between;
                   align-items:center; flex-wrap:wrap; gap:1rem; }
    .status-hero.high { border-left-color:var(--orange); } .status-hero.medium { border-left-color:#f5c46b; }
    .status-hero-label { color:var(--muted); font-size:.8rem; }
    .status-hero-value { font-size:2.6rem; font-weight:700; line-height:1.2; margin-top:.4rem; }
    .status-hero-risk { color:var(--cyan); font-size:3.2rem; font-weight:700; line-height:1.2; }
    .st-key-sensor_strip > [data-testid="stLayoutWrapper"] {
        flex:1 1 160px; min-width:0; max-width:100%; }
    .st-key-sensor_strip > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:first-child p {
        min-height:2.8rem; margin-bottom:0; font-size:.95rem; }
    .stMain [data-testid="stText"] { white-space:pre-wrap; overflow-wrap:anywhere; }
    .stMain [data-testid="stMetricValue"] { font-size:2.2rem; white-space:normal; overflow-wrap:anywhere; }
    @media (max-width:1000px) {
        .stMain [data-testid="stColumn"] { min-width:min(100%, 260px); flex:1 1 260px; }
        .stMain [data-testid="stHorizontalBlock"] { flex-wrap:wrap; gap:1rem; }
    }
    @media (max-width:640px) {
        .block-container { padding:2.8rem 1rem 2rem; }
        .topbar { align-items:flex-start; flex-direction:column; }
        .status-hero { padding:1rem; } .status-hero-value { font-size:1.95rem; }
        .status-hero-risk { font-size:2.8rem; }
        .kpi { min-height:90px; } .st-key-sensor_strip > [data-testid="stLayoutWrapper"] { flex-basis:145px; }
    }
    </style>
    """, unsafe_allow_html=True)


def render_sensors(result: dict, dataset: pd.DataFrame | None = None) -> None:
    """Display wrapping sensor tiles with optional dataset-derived alert limits."""
    with st.container(horizontal=True, key="sensor_strip"):
        for feature in FEATURES:
            guide = SENSOR_GUIDE[feature]
            value = result["readings"][feature]
            shown_value, unit = display_reading(feature, value)
            with st.container(width=180, border=True):
                st.markdown(f":material/{guide['icon']}: **{guide['label']}**")
                st.markdown(f'<div class="sensor-value">{shown_value:.1f} <span>{unit}</span></div>', unsafe_allow_html=True)
                if dataset is None:
                    st.caption("Current reading")
                else:
                    alarm = alert_state(dataset, feature, value)
                    threshold, direction = alert_threshold(dataset, feature)
                    threshold_value, threshold_unit = display_reading(feature, threshold)
                    st.markdown(f'<div class="alert-state {"alert" if alarm == "ALERT" else "normal"}">{alarm}</div>', unsafe_allow_html=True)
                    st.caption(f"Alert {direction} {threshold_value:.1f} {threshold_unit}")


@st.cache_resource
def get_predictor() -> MachinePredictor:
    return MachinePredictor()


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return ensure_dataset()


def latest_row(dataset: pd.DataFrame, index: int) -> dict[str, float]:
    row = dataset.iloc[index]
    return {feature: float(row[feature]) for feature in FEATURES}


SENSOR_GUIDE = {
    "Air temperature [K]": {"icon": "thermostat", "label": "Air temperature", "unit": "K", "meaning": "Ambient temperature around the machine.", "why": "A sudden change can indicate environmental or cooling conditions."},
    "Process temperature [K]": {"icon": "device_thermostat", "label": "Process temperature", "unit": "K", "meaning": "Temperature measured during the manufacturing process.", "why": "Higher process heat can indicate increased load or poor cooling."},
    "Rotational speed [rpm]": {"icon": "speed", "label": "Rotational speed", "unit": "RPM", "meaning": "How fast the machine shaft is rotating.", "why": "Speed and torque together describe mechanical operating load."},
    "Torque [Nm]": {"icon": "build", "label": "Torque", "unit": "Nm", "meaning": "Twisting force required to rotate the tool.", "why": "High torque can mean the machine is working against more resistance."},
    "Tool wear [min]": {"icon": "construction", "label": "Tool wear", "unit": "min", "meaning": "Accumulated tool operating time since replacement.", "why": "More wear can reduce tool quality and increase mechanical stress."},
}

SOURCE_CATALOG = [
    ("Real-time sensor data", "AVAILABLE", "data/sources/real_time_sensor_data.csv · replay input."),
    ("Historical records", "AVAILABLE", "data/sources/historical_records.csv · long-term trend context."),
    ("Maintenance logs", "AVAILABLE", "data/sources/maintenance_logs.csv · service and replacement history."),
    ("Failure records", "AVAILABLE", "data/sources/failure_records.csv · symptoms and resolutions."),
    ("Equipment manuals", "AVAILABLE", "data/sources/equipment_manuals.md · synthetic procedure reference."),
]


def display_reading(feature: str, value: float) -> tuple[float, str]:
    if feature in {"Air temperature [K]", "Process temperature [K]"}:
        return value - 273.15, "°C"
    return value, SENSOR_GUIDE[feature]["unit"]


def alert_threshold(dataset: pd.DataFrame, feature: str) -> tuple[float, str]:
    """Return a transparent demo threshold derived from the current dataset."""
    if feature == "Rotational speed [rpm]":
        return float(dataset[feature].quantile(0.1)), "below"
    return float(dataset[feature].quantile(0.9)), "above"


def alert_state(dataset: pd.DataFrame, feature: str, value: float) -> str:
    threshold, direction = alert_threshold(dataset, feature)
    if direction == "below":
        return "ALERT" if value < threshold else "NORMAL"
    return "ALERT" if value > threshold else "NORMAL"


def sensor_state(dataset: pd.DataFrame, feature: str, value: float) -> str:
    low = float(dataset[feature].quantile(0.1))
    high = float(dataset[feature].quantile(0.9))
    if value < low:
        return "LOW"
    if value > high:
        return "HIGH"
    return "TYPICAL"


def record_history(readings: dict[str, float], sample: int) -> None:
    result = st.session_state.result
    result["data_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.history.append({
        "timestamp": result["data_time"],
        "recorded_at": time.time(),
        "machine_id": st.session_state.get("machine_id", "M001"),
        "sample": sample,
        **readings,
        "failure_probability": result["failure_probability"],
        "status": result["status"],
        "main_driver": result["feature_impact"][0]["feature"],
    })


def ensure_history_predictions(predictor: MachinePredictor) -> None:
    """Upgrade old in-memory history once after the app hot-reloads."""
    for row in st.session_state.history:
        if all(key in row for key in ("failure_probability", "status", "main_driver")):
            continue
        result = predictor.predict({feature: row[feature] for feature in FEATURES})
        row.update(failure_probability=result["failure_probability"], status=result["status"],
                   main_driver=result["feature_impact"][0]["feature"])


def advance_replay(dataset: pd.DataFrame, predictor: MachinePredictor) -> None:
    from llm import explain

    st.session_state.sample_index = (st.session_state.sample_index + 1) % len(dataset)
    readings = latest_row(dataset, st.session_state.sample_index)
    st.session_state.result = predictor.predict(readings)
    record_history(readings, len(st.session_state.history) + 1)
    st.session_state.explanation, st.session_state.ollama_used = explain(st.session_state.result)
    st.session_state.replay_started_at = time.time()
    st.session_state.replay_start_index = st.session_state.sample_index
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")


def sync_replay(dataset: pd.DataFrame, predictor: MachinePredictor) -> None:
    """Move the simulated stream to the row implied by elapsed wall-clock time."""
    if not st.session_state.get("monitoring_enabled", True):
        return
    started_at = st.session_state.get("replay_started_at", time.time())
    interval = st.session_state.get("replay_interval", 2)
    start_index = st.session_state.get("replay_start_index", 0)
    target_index = (start_index + int((time.time() - started_at) // interval)) % len(dataset)
    if target_index == st.session_state.sample_index:
        return
    from llm import explain

    st.session_state.sample_index = target_index
    readings = latest_row(dataset, target_index)
    st.session_state.result = predictor.predict(readings)
    record_history(readings, len(st.session_state.history) + 1)
    st.session_state.explanation, st.session_state.ollama_used = explain(st.session_state.result)
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")


def reset_replay(dataset: pd.DataFrame, predictor: MachinePredictor) -> None:
    from llm import explain

    st.session_state.sample_index = 0
    st.session_state.history = []
    readings = latest_row(dataset, 0)
    st.session_state.result = predictor.predict(readings)
    record_history(readings, 1)
    st.session_state.explanation, st.session_state.ollama_used = explain(st.session_state.result)
    st.session_state.replay_started_at = time.time()
    st.session_state.replay_start_index = 0
    st.session_state.monitoring_enabled = True
    st.session_state.fault_scenario = ""
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")

def apply_fault_scenario(predictor: MachinePredictor, scenario: str) -> None:
    from llm import explain

    scenarios = {
        "High torque / mechanical load": {
            "Air temperature [K]": 304.0,
            "Process temperature [K]": 320.0,
            "Rotational speed [rpm]": 1050.0,
            "Torque [Nm]": 75.0,
            "Tool wear [min]": 220.0,
        },
        "Excessive tool wear": {
            "Air temperature [K]": 300.5,
            "Process temperature [K]": 313.5,
            "Rotational speed [rpm]": 1280.0,
            "Torque [Nm]": 61.0,
            "Tool wear [min]": 245.0,
        },
        "Overheating": {
            "Air temperature [K]": 310.0,
            "Process temperature [K]": 335.0,
            "Rotational speed [rpm]": 1200.0,
            "Torque [Nm]": 70.0,
            "Tool wear [min]": 210.0,
        },
    }
    readings = scenarios[scenario]
    st.session_state.monitoring_enabled = False
    st.session_state.result = predictor.predict(readings)
    record_history(readings, len(st.session_state.history) + 1)
    st.session_state.explanation, st.session_state.ollama_used = explain(st.session_state.result)
    st.session_state.fault_scenario = scenario
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")


def initialize_state(dataset: pd.DataFrame, predictor: MachinePredictor) -> None:
    if "machine_id" not in st.session_state:
        st.session_state.machine_id = "M001"
    if "monitoring_enabled" not in st.session_state:
        st.session_state.monitoring_enabled = True
    if "fault_scenario" not in st.session_state:
        st.session_state.fault_scenario = ""
    if "replay_interval" not in st.session_state:
        st.session_state.replay_interval = 2
    if "replay_started_at" not in st.session_state:
        st.session_state.replay_started_at = time.time()
    if "replay_start_index" not in st.session_state:
        st.session_state.replay_start_index = 0
    if "sample_index" not in st.session_state:
        st.session_state.sample_index = 0
    if "history" not in st.session_state:
        st.session_state.history = []
    if "result" not in st.session_state:
        st.session_state.result = predictor.predict(latest_row(dataset, 0))
        record_history(st.session_state.result["readings"], 1)
    if "explanation" not in st.session_state:
        from llm import explain
        st.session_state.explanation, st.session_state.ollama_used = explain(st.session_state.result)
    if "last_update" not in st.session_state:
        st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
    sync_replay(dataset, predictor)
