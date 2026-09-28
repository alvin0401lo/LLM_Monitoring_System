"""Small, machine-scoped lexical retrieval for the synthetic demo corpus."""

import csv
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SOURCE_DIR = Path(__file__).parent / "data" / "sources"
ALIASES = {"温度": "temperature", "扭矩": "torque", "转速": "RPM speed",
           "轴承": "bearing", "润滑": "lubrication", "磨损": "tool wear",
           "维修": "maintenance", "冷却": "cooling", "振动": "vibration"}


def retrieve(query: str, machine_id: str, data_time: str, source_dir: Path = SOURCE_DIR) -> list[dict]:
    """Rank eligible reference excerpts, rejecting unrelated and future records."""
    registry_path = source_dir / "equipment_registry.csv"
    if not registry_path.exists():
        return []
    with registry_path.open(encoding="utf-8", newline="") as file:
        registry = {row["machine_id"]: row["equipment_model"] for row in csv.DictReader(file)}
    model = registry.get(machine_id)
    if not model:
        return []
    documents = []
    for filename, id_field in [("maintenance_logs.csv", "log_id"), ("failure_records.csv", "failure_id")]:
        if not (source_dir / filename).exists():
            continue
        with (source_dir / filename).open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if row["machine_id"] != machine_id or row["date"] > data_time[:10]:
                    continue
                if row.get("equipment_model") != model:
                    continue
                body = "\n".join(f"{key}: {value}" for key, value in row.items()
                                 if key not in {id_field, "machine_id", "date", "equipment_model", "synthetic"})
                documents.append({"source_id": row[id_field], "file": filename,
                                  "machine_id": machine_id, "equipment_model": model,
                                  "date": row["date"], "synthetic": True, "text": body})
    manual_path = source_dir / "equipment_manuals.md"
    manual = manual_path.read_text(encoding="utf-8") if manual_path.exists() else ""
    if f"Equipment model: {model}\n" in manual:
        for heading, body in re.findall(r"^## (SYN-MAN-\d+[^\n]*)\n(.*?)(?=^## |\Z)", manual, re.M | re.S):
            documents.append({"source_id": heading.split(" | ")[0], "file": "equipment_manuals.md",
                              "machine_id": "shared model reference", "equipment_model": model,
                              "synthetic": True, "text": heading + "\n" + body.strip()})
    if not documents:
        return []
    for word, english in ALIASES.items():
        if word in query:
            query += " " + english
    query = re.sub(r"\bM\d{3}\b", "", query, flags=re.I)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([document["text"] for document in documents])
    scores = cosine_similarity(vectorizer.transform([query]), matrix)[0]
    indices = sorted(range(len(scores)), key=lambda index: (-scores[index], documents[index]["source_id"]))
    return [dict(documents[index], score=round(float(scores[index]), 3))
            for index in indices[:3] if scores[index] >= 0.1]


def prepare_context(context: dict, question: str = "") -> dict:
    """Attach retrieved evidence without changing the classifier's predictions."""
    query = question
    if not query or re.search(r"risk|inspect|condition|worse|风险|检查|恶化", query, re.I):
        features = context.get("feature_sensitivity", [])[:2]
        query += " " + " ".join(feature["feature"] for feature in features)
    return dict(context, retrieved_documents=retrieve(query, context["machine_id"], context["data_time"]))


def citation_report(answer: str, documents: list[dict]) -> dict:
    """Check source identifiers, not the factual validity of an LLM answer."""
    cited = set(re.findall(r"\[((?:SYN-MAN|ML|FR)-\d+)\]", answer))
    supplied = {document["source_id"] for document in documents}
    return {"cited": sorted(cited & supplied), "invalid": sorted(cited - supplied),
            "uncited": bool(documents) and not bool(cited & supplied)}
