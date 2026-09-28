import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"OLLAMA_BASE_URL": "", "OLLAMA_MODEL": "gemma4:12b"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
        self.app.session_state.monitoring_enabled = False

    def test_machine_filter_on_all_pages(self):
        for page in ("views/overview.py", "views/live_monitoring.py", "views/ai_analysis.py",
                     "views/ai_chat.py", "views/history_alerts.py"):
            self.app.switch_page(page).run(timeout=30)
            self.assertEqual(list(self.app.exception), [], page)
            self.assertTrue(any(widget.label == "Machine" for widget in self.app.selectbox), page)

    def test_high_risk_and_long_driver_render_without_errors(self):
        self.app.session_state.result = {**self.app.session_state.result,
                                        "failure_probability": 0.95, "status": "HIGH RISK"}
        self.app.switch_page("views/ai_analysis.py").run(timeout=30)
        self.assertEqual(list(self.app.exception), [])
        self.assertTrue(any("95%" in element.value for element in self.app.markdown))
        headings = [element.value for element in self.app.markdown]
        self.assertTrue(any("### Suggested checks" in heading for heading in headings))

    def test_chat_uses_full_width_messages(self):
        self.app.session_state.chat_messages = [
            {"role": "assistant", "content": "Simulated answer with a longer inspection recommendation.",
             "machine_id": "M001", "model": "gemma4:12b", "data_time": "2026-09-28",
             "sources": [], "retrieved_documents": [], "error": False}]
        self.app.switch_page("views/ai_chat.py").run(timeout=30)
        self.assertEqual(list(self.app.exception), [])
        self.assertEqual(len(self.app.chat_message), 1)
        self.assertTrue(self.app.chat_input[0].disabled)
