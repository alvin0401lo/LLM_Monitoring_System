# Machine Monitoring Prototype

A minimal monitoring demo using the AI4I 2020 sensor dataset, a Random Forest model, deterministic local analysis, and Streamlit.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Open the local URL printed by Streamlit. The project currently uses `data/ai4i2020.csv`, containing the real AI4I 2020 dataset. It must contain the five sensor columns used by `train.py` and `Machine failure`. If the CSV changes, the predictor detects the new file signature and retrains the model automatically.

The explanation page currently uses deterministic local rules and does not call an LLM. This keeps the prototype self-contained while the real dataset and ML workflow are being validated. A local LLM can be added later behind the same `explain()` interface. The explanation layer never determines failure status; it only interprets the ML result and gives general inspection suggestions.

## Synthetic sensor datasets

To create repeatable datasets for the proposed multi-sensor workflow:

```powershell
C:/Python/Python313/python.exe generate_synthetic_data.py
```

The files are written to `data/synthetic/`:

- `temperature_synthetic.csv`: ambient/process temperature and humidity
- `vibration_imu_synthetic.csv`: three-axis vibration, RMS and kurtosis
- `air_quality_gas_synthetic.csv`: CO, VOC, methane and ventilation
- `expandable_sensors_synthetic.csv`: pressure, flow, sound and door state

Each file contains `timestamp`, `machine_id`, sensor measurements, an anomaly label, and a failure label. These files are demonstration inputs for the future Edge Gateway / Data Processing stages; the current Random Forest predictor still trains on the AI4I dataset.

The requested operational source files are also generated under `data/sources/`:

- `real_time_sensor_data.csv`: timestamped replay sensor stream
- `historical_records.csv`: long-term performance and degradation context
- `maintenance_logs.csv`: service and part replacement history
- `failure_records.csv`: symptoms, failure types, and resolutions
- `equipment_manuals.md`: synthetic equipment procedure extracts for demo reference

These are clearly labeled as demo/synthetic sources. The current prediction output uses the AI4I sensor data, Random Forest model, and local analysis rules; the additional files are available source inputs for the next multi-source integration step.
