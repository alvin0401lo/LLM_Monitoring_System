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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --navy:#07111f; --panel:#10243a; --line:#1d3850; --text:#e7eef5; --muted:#91a7b9; --cyan:#67d8e8; --green:#9de6b4; --orange:#ff9b66; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; color:var(--text); font-size:16px; }
    .stApp { background:var(--navy); } [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#091827; border-right:1px solid var(--line); } [data-testid="stSidebar"] * { color:var(--text); }
    .block-container { max-width:1320px; padding:2.6rem 3rem 3.5rem; } [data-testid="stHorizontalBlock"] { align-items:stretch; gap:1rem; } [data-testid="column"] { display:flex; flex-direction:column; } [data-testid="column"] > div { width:100%; }
    h1, h2, h3 { font-family:'Space Grotesk', sans-serif; letter-spacing:0; color:var(--text); }
    h1 { font-size:clamp(2.8rem, 5vw, 4.6rem); line-height:1; margin:.35rem 0 .65rem; } h2 { font-size:2rem; } h3 { margin:.15rem 0 .55rem; font-size:1.35rem; }
    .topbar { display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid var(--line); padding-bottom:1.35rem; margin-bottom:1.7rem; }
    .kicker, .section-label { color:var(--cyan); font-size:.82rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; } .section-label { margin:2rem 0 .85rem; }
    .subtle, .microcopy { color:var(--muted); font-size:1rem; } .subtle { margin:0; }
    .live-badge { border:1px solid #2b5869; background:#0d2939; color:var(--cyan); padding:.5rem .75rem; font-size:.72rem; font-weight:700; letter-spacing:.08em; } .live-badge span { display:inline-block; width:7px; height:7px; background:var(--green); border-radius:50%; margin-right:.45rem; }
    .kpi, .panel, .next-step { background:var(--panel); border:1px solid var(--line); } .kpi { min-height:108px; height:100%; box-sizing:border-box; padding:1.2rem 1.3rem; }
    .kpi-label, .panel-title { color:var(--muted); font-size:.78rem; font-weight:700; letter-spacing:.1em; } .kpi-value { font-family:'Space Grotesk'; font-size:2.25rem; font-weight:700; margin-top:.7rem; }
    .kpi.high-risk .kpi-value { color:var(--orange); } .kpi.medium-risk .kpi-value { color:#f5c46b; } .kpi.low-risk .kpi-value { color:var(--green); } .kpi.accent .kpi-value { color:var(--cyan); }
    .panel-title { margin-bottom:1rem; font-size:.9rem; } [data-testid="stVerticalBlockBorderWrapper"] { background:var(--panel); border:1px solid var(--line); border-radius:0; padding:1.35rem 1.5rem; height:100%; box-sizing:border-box; }
    .reading-row { display:flex; justify-content:space-between; border-bottom:1px solid var(--line); padding:.78rem 0; color:var(--muted); font-size:1rem; } .reading-row:last-child { border-bottom:0; } .reading-row strong { color:var(--text); font-size:1.05rem; }
    .sensor-value { color:var(--text); font-family:'Space Grotesk'; font-size:2.25rem; font-weight:700; margin-top:.7rem; white-space:nowrap; } .sensor-value span { color:var(--muted); font-family:'DM Sans'; font-size:.95rem; font-weight:500; }
    .sensor-state { font-size:.68rem; font-weight:700; letter-spacing:.08em; margin-top:.3rem; } .sensor-state.typical { color:var(--green); } .sensor-state.high { color:var(--orange); } .sensor-state.low { color:var(--cyan); }
    .alert-state { font-size:.8rem; font-weight:700; letter-spacing:.08em; margin-top:.45rem; } .alert-state.normal { color:var(--green); } .alert-state.alert { color:var(--orange); } .threshold { color:var(--muted); font-size:.8rem; margin-top:.25rem; }
    .signal-number { color:var(--cyan); font-family:'Space Grotesk'; font-size:5.5rem; font-weight:700; line-height:1; margin:1.4rem 0 .7rem; } .progress-track { background:#1d354b; height:9px; margin:1.2rem 0 .7rem; } .progress-fill { background:var(--orange); height:9px; }
    .microcopy { font-size:.9rem; } .next-step { display:flex; justify-content:space-between; align-items:center; margin-top:1.5rem; padding:1.35rem 1.5rem; } .next-step h3 { font-size:1.2rem; } .next-arrow { color:var(--cyan); font-size:2.4rem; }
    .action-title { font-family:'Space Grotesk'; font-size:1.55rem; line-height:1.2; font-weight:700; color:var(--text); }
    .status-hero { background:linear-gradient(110deg, #102f43, #10243a); border:1px solid #2b596d; border-left:7px solid var(--orange); padding:1.6rem 1.8rem; display:flex; justify-content:space-between; align-items:end; gap:1rem; } .status-hero-label { color:var(--muted); font-size:.8rem; font-weight:700; letter-spacing:.12em; } .status-hero-value { font-family:'Space Grotesk'; font-size:3.5rem; font-weight:700; line-height:1; margin-top:.5rem; } .status-hero-risk { color:var(--cyan); font-family:'Space Grotesk'; font-size:3.8rem; font-weight:700; line-height:1; text-align:right; }
    .stButton > button { background:#173750; border:1px solid #2b596d; color:var(--text); border-radius:2px; font-size:1rem; min-height:2.8rem; } .stButton > button:hover { border-color:var(--cyan); color:var(--cyan); } [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:1.2rem; } [data-testid="stMetricLabel"] { font-size:.9rem; } [data-testid="stMetricValue"] { font-size:2rem; }
    .stCaption, [data-testid="stCaptionContainer"] { font-size:.9rem; }
    @media (max-width:800px) { .block-container { padding:1.2rem 1rem 2rem; } .topbar { align-items:flex-start; gap:1rem; flex-direction:column; } }
    </style>
    """, unsafe_allow_html=True)


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
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "machine_id": st.session_state.get("machine_id", "M001"),
        "sample": sample,
        **readings,
    })


def advance_replay(dataset: pd.DataFrame, predictor: MachinePredictor) -> None:
    from llm import explain

    st.session_state.sample_index = (st.session_state.sample_index + 1) % len(dataset)
    readings = latest_row(dataset, st.session_state.sample_index)
    st.session_state.result = predictor.predict(readings)
    record_history(readings, len(st.session_state.history) + 1)
    st.session_state.explanation, st.session_state.ollama_used = explain(st.session_state.result)
    st.session_state.replay_started_at = time.time()
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")


def sync_replay(dataset: pd.DataFrame, predictor: MachinePredictor) -> None:
    """Move the simulated stream to the row implied by elapsed wall-clock time."""
    if not st.session_state.get("monitoring_enabled", True):
        return
    started_at = st.session_state.get("replay_started_at", time.time())
    interval = st.session_state.get("replay_interval", 2)
    target_index = int((time.time() - started_at) // interval) % len(dataset)
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
    st.session_state.last_update = datetime.now().strftime("%H:%M:%S")
    sync_replay(dataset, predictor)
