from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "data" / "synthetic"
SOURCE_DIR = ROOT / "data" / "sources"
ROWS = 2000


def timestamps(rows: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=rows, freq="min")


def machine_ids(rows: int) -> np.ndarray:
    return np.array([f"SYN-{index % 4 + 1:02d}" for index in range(rows)])


def save(frame: pd.DataFrame, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    frame.to_csv(path, index=False)
    print(f"Created {path} ({len(frame)} rows)")


def save_source(frame: pd.DataFrame, name: str) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / name
    frame.to_csv(path, index=False)
    print(f"Created {path} ({len(frame)} rows)")


def generate_temperature(rng: np.random.Generator) -> None:
    time = timestamps(ROWS)
    phase = np.linspace(0, 18 * np.pi, ROWS)
    ambient = 23 + 2.5 * np.sin(phase / 8) + rng.normal(0, 0.7, ROWS)
    process = ambient + 31 + 1.8 * np.sin(phase) + rng.normal(0, 0.8, ROWS)
    humidity = 48 + 8 * np.cos(phase / 5) + rng.normal(0, 2, ROWS)
    stress = np.maximum(process - 58, 0) + np.maximum(humidity - 68, 0) * 0.3
    anomaly = (stress > 1.8).astype(int)
    save(pd.DataFrame({
        "timestamp": time,
        "machine_id": machine_ids(ROWS),
        "air_temperature_C": ambient.round(2),
        "process_temperature_C": process.round(2),
        "humidity_pct": humidity.clip(20, 90).round(2),
        "temperature_anomaly": anomaly,
        "failure": (stress > 2.2).astype(int),
    }), "temperature_synthetic.csv")


def generate_vibration(rng: np.random.Generator) -> None:
    time = timestamps(ROWS)
    wear = np.linspace(0, 1, ROWS)
    vibration_x = 0.18 + 0.08 * wear + rng.normal(0, 0.025, ROWS)
    vibration_y = 0.22 + 0.12 * wear + rng.normal(0, 0.03, ROWS)
    vibration_z = 0.15 + 0.07 * wear + rng.normal(0, 0.02, ROWS)
    shock = rng.random(ROWS) < 0.045
    vibration_x[shock] += rng.uniform(0.25, 0.65, shock.sum())
    vibration_y[shock] += rng.uniform(0.3, 0.8, shock.sum())
    vibration_z[shock] += rng.uniform(0.2, 0.55, shock.sum())
    rms = np.sqrt((vibration_x**2 + vibration_y**2 + vibration_z**2) / 3)
    kurtosis = 3 + np.maximum(rms - 0.35, 0) * 18 + rng.normal(0, 0.15, ROWS)
    anomaly = ((rms > 0.4) | (kurtosis > 5)).astype(int)
    save(pd.DataFrame({
        "timestamp": time,
        "machine_id": machine_ids(ROWS),
        "vibration_x_g": vibration_x.round(4),
        "vibration_y_g": vibration_y.round(4),
        "vibration_z_g": vibration_z.round(4),
        "vibration_rms_g": rms.round(4),
        "kurtosis": kurtosis.round(3),
        "vibration_anomaly": anomaly,
        "failure": (anomaly & (rms > 0.48)).astype(int),
    }), "vibration_imu_synthetic.csv")


def generate_air_quality(rng: np.random.Generator) -> None:
    time = timestamps(ROWS)
    ventilation = 1 + 0.15 * np.sin(np.linspace(0, 12 * np.pi, ROWS))
    co = 2.5 / ventilation + rng.normal(0, 0.35, ROWS)
    voc = 85 / ventilation + rng.normal(0, 8, ROWS)
    methane = 1.2 / ventilation + rng.normal(0, 0.12, ROWS)
    smoke_event = rng.random(ROWS) < 0.035
    co[smoke_event] += rng.uniform(3, 10, smoke_event.sum())
    voc[smoke_event] += rng.uniform(80, 240, smoke_event.sum())
    methane[smoke_event] += rng.uniform(1, 3, smoke_event.sum())
    anomaly = ((co > 5) | (voc > 180) | (methane > 2.2)).astype(int)
    save(pd.DataFrame({
        "timestamp": time,
        "machine_id": machine_ids(ROWS),
        "co_ppm": co.clip(0, None).round(3),
        "voc_ppm": voc.clip(0, None).round(2),
        "methane_ppm": methane.clip(0, None).round(3),
        "ventilation_index": ventilation.round(3),
        "gas_anomaly": anomaly,
        "failure": (smoke_event & (voc > 220)).astype(int),
    }), "air_quality_gas_synthetic.csv")


def generate_expandable(rng: np.random.Generator) -> None:
    time = timestamps(ROWS)
    pressure = 4.2 + rng.normal(0, 0.12, ROWS)
    flow = 62 + rng.normal(0, 2.5, ROWS)
    sound = 61 + rng.normal(0, 1.8, ROWS)
    door_open = (rng.random(ROWS) < 0.025).astype(int)
    pressure[door_open == 1] -= rng.uniform(0.8, 1.8, (door_open == 1).sum())
    flow[door_open == 1] += rng.uniform(8, 18, (door_open == 1).sum())
    sound[door_open == 1] += rng.uniform(5, 13, (door_open == 1).sum())
    anomaly = ((pressure < 3.3) | (flow > 78) | (sound > 72)).astype(int)
    save(pd.DataFrame({
        "timestamp": time,
        "machine_id": machine_ids(ROWS),
        "pressure_bar": pressure.round(3),
        "flow_lpm": flow.round(2),
        "sound_db": sound.round(2),
        "door_open": door_open,
        "sensor_anomaly": anomaly,
        "failure": ((pressure < 3.0) | (flow > 82)).astype(int),
    }), "expandable_sensors_synthetic.csv")


def generate_source_files(rng: np.random.Generator) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    ai4i_path = ROOT / "data" / "ai4i2020.csv"
    ai4i = pd.read_csv(ai4i_path)
    replay = ai4i.head(240).copy()
    replay.insert(0, "timestamp", pd.date_range("2026-09-15 08:00", periods=len(replay), freq="min"))
    replay.insert(1, "machine_id", [f"M{index % 4 + 1:03d}" for index in range(len(replay))])
    replay.rename(columns={"Machine failure": "failure"}, inplace=True)
    replay.to_csv(SOURCE_DIR / "real_time_sensor_data.csv", index=False)
    print(f"Created {SOURCE_DIR / 'real_time_sensor_data.csv'} ({len(replay)} rows)")

    history_dates = pd.date_range("2026-06-01", periods=90, freq="D")
    history = pd.DataFrame({
        "date": history_dates,
        "machine_id": [f"M{index % 4 + 1:03d}" for index in range(len(history_dates))],
        "avg_failure_risk": np.clip(0.18 + np.linspace(0, 0.25, len(history_dates)) + rng.normal(0, 0.04, len(history_dates)), 0, 1).round(3),
        "runtime_hours": rng.integers(6, 24, len(history_dates)),
        "production_cycles": rng.integers(400, 1800, len(history_dates)),
        "performance_trend": ["stable" if index < 45 else "degrading" for index in range(len(history_dates))],
    })
    save_source(history, "historical_records.csv")

    generate_knowledge_sources()


def generate_knowledge_sources() -> None:
    """Create clearly labelled reference cases without regenerating telemetry."""
    save_source(pd.DataFrame({"machine_id": [f"M{i:03d}" for i in range(1, 5)],
                              "equipment_model": ["SYN-MILL-01"] * 4,
                              "synthetic": [True] * 4}), "equipment_registry.csv")

    maintenance = pd.DataFrame({
        "log_id": [f"ML-{index:04d}" for index in range(1, 13)],
        "date": pd.date_range("2026-06-10", periods=12, freq="7D"),
        "machine_id": [f"M{index % 4 + 1:03d}" for index in range(12)],
        "maintenance_type": ["Inspection", "Tool replacement", "Lubrication", "Bearing inspection"] * 3,
        "component": ["Tool", "Tool", "Spindle", "Bearing"] * 3,
        "technician_note": [
            "No abnormality found", "Tool replaced after wear threshold", "Lubricant level restored", "Minor vibration noted; monitor next run"
        ] * 3,
        "action_status": ["Closed", "Closed", "Closed", "Follow-up"] * 3,
    })
    maintenance["equipment_model"] = "SYN-MILL-01"
    maintenance["synthetic"] = True
    maintenance["symptom"] = ["Routine tool condition review", "Tool wear and surface quality drop",
                               "Lubrication level low", "Minor bearing vibration"] * 3
    maintenance["action"] = ["Inspected tooling", "Replaced worn tooling", "Restored lubricant level",
                              "Checked bearing and requested follow-up"] * 3
    maintenance["outcome"] = ["No abnormality observed", "Surface quality improved in simulated follow-up",
                               "Lubricant level restored; cause not established", "Vibration unresolved; follow-up required"] * 3
    save_source(maintenance, "maintenance_logs.csv")

    failures = pd.DataFrame({
        "failure_id": [f"FR-{index:04d}" for index in range(1, 9)],
        "date": pd.date_range("2026-06-18", periods=8, freq="11D"),
        "machine_id": [f"M{index % 4 + 1:03d}" for index in range(8)],
        "failure_type": ["Tool wear", "Heat dissipation", "Power failure", "Overstrain"] * 2,
        "symptom": ["Surface quality drop", "Process temperature rise", "Unexpected stop", "High torque and low RPM"] * 2,
        "root_cause_status": ["Confirmed", "Under review", "Confirmed", "Confirmed"] * 2,
        "resolution": ["Tool replaced", "Cooling system cleaned", "Power module replaced", "Load reduced and bearing checked"] * 2,
    })
    failures["equipment_model"] = "SYN-MILL-01"
    failures["synthetic"] = True
    failures["outcome"] = ["Tooling replaced in simulated case", "Cooling restored; cause still under review",
                            "Simulated restart completed", "Load reduced; bearing follow-up required"] * 2
    save_source(failures, "failure_records.csv")

    manual = SOURCE_DIR / "equipment_manuals.md"
    manual.write_text(
        """# Synthetic Equipment Manual
Equipment model: SYN-MILL-01
Version: demo-1
This is synthetic documentation, not an OEM manual or an approved safety procedure.

## SYN-MAN-001 | Torque and rotational speed
Higher torque with lower rotational speed may indicate mechanical resistance or changing load.
Qualified personnel should check tooling, shaft, bearing condition, and load path against approved site procedures.
This pattern alone does not establish a root cause.

## SYN-MAN-002 | Temperature and cooling
Rising process temperature can warrant a cooling and load review against the approved operating range.
Check cooling performance and compare subsequent readings. No numerical safety limit is provided by this demo manual.
Model alerts do not authorise bypassing safety controls.

## SYN-MAN-003 | Tool wear and lubrication
Review tool wear, lubrication records, and tooling condition against the approved maintenance procedure.
Replacement and lubrication decisions require qualified review. A past repair does not prove the present fault has the same cause.
""",
        encoding="utf-8",
    )
    print(f"Created {manual}")


def main() -> None:
    rng = np.random.default_rng(20260914)
    generate_temperature(rng)
    generate_vibration(rng)
    generate_air_quality(rng)
    generate_expandable(rng)
    generate_source_files(rng)


if __name__ == "__main__":
    main()
