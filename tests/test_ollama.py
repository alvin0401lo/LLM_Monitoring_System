import json
import os
import threading
import time
import unittest
from types import SimpleNamespace
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import requests
import pandas as pd
from streamlit.testing.v1 import AppTest

from llm import OllamaError, analysis_due, machine_context, request_ollama, update_analysis


class FakeOllamaHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.received = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(self.server.response_status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.server.response_body).encode())

    def log_message(self, *args):
        pass


class ControlledExecutor:
    def __init__(self):
        self.futures = []

    def submit(self, *args):
        future = Future()
        self.futures.append(future)
        return future


class OllamaIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllamaHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.server.response_status = 200
        self.server.response_body = {"message": {"role": "assistant", "content": "Simulated API response"}}
        self.settings = {"host": f"http://127.0.0.1:{self.server.server_port}", "model": "gemma4:12b"}
        self.result = {
            "readings": {"Air temperature [K]": 300.0, "Process temperature [K]": 310.0,
                         "Rotational speed [rpm]": 1500.0, "Torque [Nm]": 40.0, "Tool wear [min]": 100.0},
            "failure_probability": 0.6, "status": "MEDIUM RISK",
            "feature_impact": [{"feature": "Air temperature [K]", "probability_change": 0.1}],
            "data_time": "2026-09-28 12:00:00",
        }
        self.history = [{"machine_id": "M001", "timestamp": self.result["data_time"],
                         "recorded_at": time.time(), "Air temperature [K]": 300.0}]

    def test_native_api_payload_and_answer(self):
        answer = request_ollama(self.settings, [{"role": "user", "content": "Explain this snapshot"}])
        self.assertEqual(answer, "Simulated API response")
        self.assertEqual(self.server.received["model"], "gemma4:12b")
        self.assertIs(self.server.received["stream"], False)
        self.assertIs(self.server.received["think"], False)

    def test_thinking_is_not_used_as_final_answer(self):
        self.server.response_body = {"message": {"content": "", "thinking": "Internal trace"},
                                     "done_reason": "length"}
        with self.assertRaisesRegex(OllamaError, "thinking but no final answer"):
            request_ollama(self.settings, [])

    def test_final_answer_excludes_thinking(self):
        self.server.response_body = {"message": {"content": "Final answer", "thinking": "Internal trace"}}
        self.assertEqual(request_ollama(self.settings, []), "Final answer")

    def test_other_models_keep_default_thinking_controls(self):
        request_ollama({**self.settings, "model": "gpt-oss:20b"}, [])
        self.assertNotIn("think", self.server.received)

    def test_empty_output_limit_has_specific_error(self):
        self.server.response_body = {"message": {"content": ""}, "done_reason": "length"}
        with self.assertRaisesRegex(OllamaError, "reached the output limit"):
            request_ollama(self.settings, [])

    def test_model_not_found_and_empty_answer(self):
        self.server.response_status = 404
        with self.assertRaisesRegex(OllamaError, "404"):
            request_ollama(self.settings, [])
        self.server.response_status = 200
        self.server.response_body = {"message": {"content": ""}}
        with self.assertRaisesRegex(OllamaError, "empty answer"):
            request_ollama(self.settings, [])

    def test_timeout_is_an_operator_error(self):
        with patch("llm.requests.post", side_effect=requests.ReadTimeout()):
            with self.assertRaisesRegex(OllamaError, "timed out"):
                request_ollama(self.settings, [])

    def test_context_filters_other_machines_and_old_observations(self):
        history = self.history + [
            {**self.history[0], "machine_id": "M002", "Air temperature [K]": 1000},
            {**self.history[0], "recorded_at": time.time() - 1801, "Air temperature [K]": 500},
        ]
        context = machine_context(self.result, history, "M001")
        self.assertEqual(context["history"]["sample_count"], 1)
        self.assertEqual(context["history"]["sensor_summary"]["Air temperature [K]"]["mean"], 300)
        self.assertEqual(context["data_time"], self.result["data_time"])
        json.dumps(context, allow_nan=False)

    def test_30minute_boundary_and_configuration_change(self):
        record = {"last_attempt": 0, "configuration": (self.settings["host"], self.settings["model"])}
        self.assertFalse(analysis_due(record, self.settings, 1799))
        self.assertTrue(analysis_due(record, self.settings, 1800))
        self.assertTrue(analysis_due(record, self.settings, 10, force=True))
        self.assertTrue(analysis_due(record, {**self.settings, "model": "other-model"}, 10))

    def test_background_request_failure_retains_success_and_manual_retry(self):
        executor = ControlledExecutor()
        state = {}
        with patch("llm.analysis_executor", return_value=executor):
            update_analysis(state, self.result, self.history, "M001", self.settings, now=0)
            self.assertEqual(len(executor.futures), 1)
            executor.futures[0].set_result("First successful analysis")
            record = update_analysis(state, self.result, self.history, "M001", self.settings, now=5)
            self.assertEqual(record["content"], "First successful analysis")
            update_analysis(state, self.result, self.history, "M001", self.settings, now=1799)
            self.assertEqual(len(executor.futures), 1)
            update_analysis(state, self.result, self.history, "M001", self.settings, now=1800)
            self.assertEqual(len(executor.futures), 2)
            executor.futures[1].set_exception(OllamaError("Unavailable"))
            record = update_analysis(state, self.result, self.history, "M001", self.settings, now=1805)
            self.assertEqual(record["content"], "First successful analysis")
            self.assertEqual(record["error"], "Unavailable")
            update_analysis(state, self.result, self.history, "M001", self.settings, now=1810)
            self.assertEqual(len(executor.futures), 2)
            update_analysis(state, self.result, self.history, "M001", self.settings, force=True, now=1810)
            self.assertEqual(len(executor.futures), 3)

    def test_pending_analysis_stays_with_its_original_machine(self):
        executor = ControlledExecutor()
        state = {}
        with patch("llm.analysis_executor", return_value=executor):
            update_analysis(state, self.result, self.history, "M001", self.settings, now=0)
            update_analysis(state, self.result, self.history, "M002", self.settings, now=1)
            self.assertEqual(len(executor.futures), 1)
            executor.futures[0].set_result("Machine one")
            record = update_analysis(state, self.result, self.history, "M002", self.settings, now=2)
            self.assertNotIn("content", record)
            self.assertEqual(state["analysis_records"]["M001"]["content"], "Machine one")

    def test_unconfigured_host_never_submits(self):
        with patch("llm.analysis_executor") as executor:
            update_analysis({}, self.result, self.history, "M001", {"host": "", "model": "gemma4:12b"})
            executor.assert_not_called()

    def test_streamlit_pages_offline_reuse_recorded_predictions(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "", "OLLAMA_MODEL": "gemma4:12b"}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
            self.assertEqual(list(app.exception), [])
            app.session_state.monitoring_enabled = False
            with patch("predictor.MachinePredictor.predict", side_effect=AssertionError("History re-predicted")):
                for page in ("views/live_monitoring.py", "views/history_alerts.py", "views/ai_analysis.py", "views/ai_chat.py"):
                    app.switch_page(page).run(timeout=30)
                    self.assertEqual(list(app.exception), [], page)
            self.assertTrue(app.chat_input[0].disabled)
            self.assertIn("failure_probability", app.session_state.history[0])

    def test_manual_replay_step_survives_background_poll(self):
        from common import advance_replay, sync_replay

        class State(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        state = State(sample_index=3, replay_interval=2, replay_started_at=1000,
                      replay_start_index=0, monitoring_enabled=True, machine_id="M001", history=[])
        dataset = pd.DataFrame([self.result["readings"] for _ in range(10)])
        predictor = Mock()
        predictor.predict.side_effect = lambda readings: {**self.result, "readings": readings.copy()}
        with patch("common.st", SimpleNamespace(session_state=state)), patch("common.time.time", return_value=1000):
            advance_replay(dataset, predictor)
            sync_replay(dataset, predictor)
            self.assertEqual(state.sample_index, 4)
            self.assertEqual(predictor.predict.call_count, 1)
        with patch("common.st", SimpleNamespace(session_state=state)), patch("common.time.time", return_value=1002):
            sync_replay(dataset, predictor)
            self.assertEqual(state.sample_index, 5)

    def test_streamlit_analysis_and_chat_use_simulated_http_api(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": self.settings["host"], "OLLAMA_MODEL": self.settings["model"]}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
            self.assertEqual(list(app.exception), [])
            app.session_state.analysis_job["future"].result(timeout=10)
            app.run(timeout=30)
            self.assertEqual(app.session_state.analysis_records["M001"]["content"], "Simulated API response")
            app.switch_page("views/ai_analysis.py").run(timeout=30)
            self.assertEqual(list(app.exception), [])
            self.assertTrue(any(element.value == "Simulated API response" for element in app.markdown))
            app.switch_page("views/ai_chat.py").run(timeout=30)
            app.chat_input[0].set_value("Why is the risk high?").run(timeout=30)
            self.assertEqual(list(app.exception), [])
            self.assertEqual(app.session_state.chat_messages[-1]["content"], "Simulated API response")
            self.assertEqual(app.session_state.chat_messages[-1]["model"], "gemma4:12b")
            self.assertIs(app.session_state.chat_messages[-1]["error"], False)
            saved = app.session_state.chat_messages[-1]["retrieved_documents"]
            self.assertTrue(saved)
            evidence = json.loads(self.server.received["messages"][1]["content"].removeprefix("Current evidence: "))
            self.assertEqual(evidence["retrieved_documents"], saved)
            self.assertTrue(app.session_state.chat_messages[-1]["citations"]["uncited"])

    def test_offline_reference_search(self):
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "", "OLLAMA_MODEL": "gemma4:12b"}):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
            app.switch_page("views/ai_analysis.py").run(timeout=30)
            app.text_input[0].set_value("temperature cooling")
            next(button for button in app.button if button.label == "Search references").click().run(timeout=30)
            self.assertEqual(list(app.exception), [])
            self.assertTrue(any("SYN-MAN-002" in element.label for element in app.expander))


if __name__ == "__main__":
    unittest.main()
