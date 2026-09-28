import shutil
import tempfile
import unittest
from pathlib import Path

from rag import SOURCE_DIR, citation_report, prepare_context, retrieve


class ReferenceRetrievalTests(unittest.TestCase):
    def test_machine_scope_and_synthetic_metadata(self):
        documents = retrieve("High torque low RPM load bearing", "M004", "2026-09-28")
        self.assertTrue(documents)
        self.assertTrue(any(doc["source_id"].startswith("FR-") for doc in documents))
        self.assertTrue(all(doc["machine_id"] in {"M004", "shared model reference"} for doc in documents))
        self.assertTrue(all(doc["synthetic"] and doc["equipment_model"] == "SYN-MILL-01" for doc in documents))
        self.assertLessEqual(len(documents), 3)

    def test_no_match_and_unknown_machine(self):
        self.assertEqual(retrieve("lunar geology", "M001", "2026-09-28"), [])
        self.assertEqual(retrieve("temperature", "UNKNOWN", "2026-09-28"), [])

    def test_future_logs_excluded(self):
        documents = retrieve("Tool wear", "M001", "2026-06-01")
        self.assertTrue(documents)
        self.assertTrue(all(doc["source_id"].startswith("SYN-MAN-") for doc in documents))

    def test_chinese_domain_alias(self):
        documents = retrieve("温度冷却", "M002", "2026-09-28")
        self.assertTrue(any(doc["source_id"] == "SYN-MAN-002" for doc in documents))

    def test_registry_blocks_wrong_equipment_manual(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(SOURCE_DIR, root, dirs_exist_ok=True)
            registry = root / "equipment_registry.csv"
            registry.write_text("machine_id,equipment_model,synthetic\nM001,OTHER,True\n", encoding="utf-8")
            self.assertEqual(retrieve("temperature", "M001", "2026-09-28", root), [])

    def test_files_are_refreshed_and_missing_corpus_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(retrieve("temperature", "M001", "2026-09-28", root), [])
            shutil.copytree(SOURCE_DIR, root, dirs_exist_ok=True)
            self.assertTrue(retrieve("temperature", "M001", "2026-09-28", root))
            (root / "equipment_manuals.md").unlink()
            self.assertEqual(retrieve("temperature", "M001", "2026-09-28", root), [])

    def test_snapshot_preserves_classifier_output(self):
        context = {"machine_id": "M001", "data_time": "2026-09-28", "failure_probability": 0.7,
                   "feature_sensitivity": [{"feature": "Tool wear [min]"}], "retrieved_documents": []}
        prepared = prepare_context(context)
        self.assertEqual(prepared["failure_probability"], 0.7)
        self.assertTrue(prepared["retrieved_documents"])
        self.assertEqual(context["retrieved_documents"], [])
        self.assertEqual(prepare_context(context, "lunar geology")["retrieved_documents"], [])

    def test_citation_identifier_checks(self):
        docs = [{"source_id": "SYN-MAN-002"}]
        report = citation_report("Cooling review [SYN-MAN-002]. [FR-9999]", docs)
        self.assertEqual(report["cited"], ["SYN-MAN-002"])
        self.assertEqual(report["invalid"], ["FR-9999"])
        self.assertTrue(citation_report("Uncited reference claim", docs)["uncited"])
        self.assertFalse(citation_report("No reference evidence", [])["uncited"])

