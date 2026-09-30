# Machine Monitoring Prototype

A monitoring demo using the public synthetic AI4I 2020 dataset, a Random Forest model, local Ollama analysis, and Streamlit.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Open the local URL printed by Streamlit. The project uses `data/ai4i2020.csv`, containing the public synthetic AI4I 2020 dataset. It must contain the five sensor columns used by `train.py` and `Machine failure`. If the CSV changes, the predictor detects the new file signature and retrains the model automatically.

The Random Forest determines risk status. Its probability is a classification estimate, not a calibrated prediction of a failure within the next 30 minutes. Ollama explains the supplied result, recent replay summary and retrieved synthetic references; it does not determine failure status.

## Connect to Ollama on Ubuntu

1. On Ubuntu, install Ollama, download the selected model (`gemma4:12b`), and confirm it answers locally. Model feasibility depends on available RAM, VRAM and context size.
2. Configure the Ollama service to accept requests from the website PC. In `sudo systemctl edit ollama.service`, add:

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
```

Then run `sudo systemctl daemon-reload` and `sudo systemctl restart ollama`. Restrict port 11434 to the website PC using the Ubuntu firewall; the native API does not require authentication. Do not expose it directly to the public internet.

3. From the website PC, confirm `curl.exe http://<UBUNTU_IP>:11434/api/tags` returns the installed model list.
4. Create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`, then set the Ubuntu address and exact model name:

```toml
[ollama]
host = "http://<UBUNTU_IP>:11434"
model = "gemma4:12b"
```

Alternatively set `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in the website process environment. These override the file. A blank host or model disables remote calls. Never use `localhost` for a model running on another PC.

An initial analysis is submitted when a configured session opens. Subsequent automatic attempts occur at most once every 30 minutes per selected machine in that session. The sidebar polls background jobs every five seconds and provides **Refresh analysis** for an immediate request. Failed requests retain the last successful analysis and are not repeatedly retried on every poll. The AI Analysis page shows the model, generation time and sensor snapshot time. Chat questions call Ollama immediately using the current snapshot and recent conversation.

Scheduling and sensor replay depend on an active browser session. Closing the browser or restarting the website stops this schedule; it is not a persistent background monitoring service. Each browser session owns its analysis history. Sensor replay and ML predictions remain independent of LLM requests, and recorded risk is reused by charts rather than re-predicting all past samples.

Storing model files on an aiDAPTIV SSD does not prove its inference cache or memory extension is in use. Verify the installed Phison middleware and supported inference integration separately.

The client requests `think: false` for Gemma 4 models: their default thinking can otherwise exhaust the 600-token generation budget before producing a final answer. Other models retain their default thinking controls. Only `message.content` is displayed; thinking traces are never substituted for answers. An empty response identifies thinking-only or output-limit failures when the API provides that information.

## Checks

Run `python -m unittest discover -s tests`. The checks use a simulated HTTP API to verify client behaviour and a controlled clock to verify the 30-minute schedule. They do not prove real model output, Ubuntu connectivity or aiDAPTIV acceleration.

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

These are demo/synthetic sources, not real factory records. The current prediction still uses the AI4I sensor data and Random Forest model. Historical-record and sensor-source CSVs are not used by RAG; the recent replay summary comes from the browser session.

## Synthetic reference retrieval

The local reference library contains 12 maintenance cases, eight failure cases and three manual sections. `equipment_registry.csv` maps the four demo machines to the synthetic equipment model. TF-IDF keyword retrieval uses the existing scikit-learn dependency: no vector database, embedding server or fine-tuning is required. Logs are restricted to the selected machine and snapshot date; shared manuals must match the equipment model. At most three excerpts above a 0.1 cosine-similarity threshold enter the LLM context. This score is keyword similarity, not confidence or failure probability.

AI Analysis includes an offline reference search and a retrieval preview when Ollama is unavailable. Generated analyses and chat messages retain their exact supplied excerpts, with IDs such as `[SYN-MAN-002]` and `[ML-0002]`. The interface flags unknown identifiers and absent reference citations; valid identifiers alone do not establish that a claim is supported. Synthetic cases cannot validate real diagnosis, forecasting or maintenance safety. English keyword retrieval supports a few Chinese domain aliases, not general multilingual semantic search.

In the sidebar, **Reference files** accepts UTF-8 TXT, Markdown and CSV files up to 200 KB (CSV: 1-200 rows). Upload one or more files, then choose one for the current machine. Only the selected file is eligible for AI Analysis, reference search and AI Chat; relevant excerpts compete with the built-in references for the three RAG slots. An unrelated selected file will not be sent to Ollama. CSV rows with a different `machine_id` or a future `date` are excluded. Uploaded files are session-only, are not saved to the project, and do not change the Random Forest model or its failure probability. Select a file separately for each machine. Uploaded content is unverified and may be sent to the configured Ollama endpoint; do not use sensitive records through a public tunnel. PDF files are not supported by this lightweight prototype.

To regenerate only the reference corpus without changing sensor datasets:

```powershell
.\.venv\Scripts\python.exe -c "from generate_synthetic_data import generate_knowledge_sources; generate_knowledge_sources()"
```

Reference files are reread for each request so edits apply to the next request. Updating real-world manuals requires approved equipment/version metadata and appropriate access controls; this demo does not implement production authorisation or an audit database.
