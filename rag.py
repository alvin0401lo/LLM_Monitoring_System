"""Small, machine-scoped lexical retrieval for the synthetic demo corpus."""

import csv
import hashlib
import io
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SOURCE_DIR = Path(__file__).parent / "data" / "sources"
ALIASES = {"温度": "temperature", "扭矩": "torque", "转速": "RPM speed",
           "轴承": "bearing", "润滑": "lubrication", "磨损": "tool wear",
           "维修": "maintenance", "冷却": "cooling", "振动": "vibration"}
MAX_UPLOAD_BYTES = 200_000


def parse_uploaded_reference(filename: str, content: bytes) -> dict:
    """Validate a session-only reference file before it can enter a prompt."""
    name = Path(filename).name
    extension = Path(name).suffix.lower()
    if extension not in {".txt", ".md", ".csv"}:
        raise ValueError("Upload a TXT, Markdown or CSV reference file.")
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Reference files must be non-empty and no larger than 200 KB.")
    try:
        body = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("The reference file must use UTF-8 text encoding.") from error
    if not body.strip():
        raise ValueError("The reference file contains no searchable text.")
    if extension == ".csv":
        reader = csv.DictReader(io.StringIO(body))
        if not reader.fieldnames or not any(field.strip() for field in reader.fieldnames):
            raise ValueError("The CSV file needs a header row.")
        rows = list(reader)
        if not rows or len(rows) > 200:
            raise ValueError("The CSV file must contain 1 to 200 data rows.")
        chunks = [{"text": "\n".join(f"{key}: {value}" for key, value in row.items() if key and value),
                   "machine_id": row.get("machine_id", ""), "date": row.get("date", "")}
                  for row in rows]
    else:
        chunks = [{"text": body[index:index + 1400], "machine_id": "", "date": ""}
                  for index in range(0, len(body), 1400)]
    if not any(chunk["text"].strip() for chunk in chunks):
        raise ValueError("The reference file contains no searchable records.")
    return {"name": name, "digest": hashlib.sha256(content).hexdigest()[:8].upper(), "chunks": chunks}


def uploaded_excerpts(uploaded: dict | None, machine_id: str, data_time: str, model: str) -> list[dict]:
    if not uploaded:
        return []
    documents = []
    for index, chunk in enumerate(uploaded["chunks"], 1):
        if chunk["machine_id"] and chunk["machine_id"] != machine_id:
            continue
        if chunk["date"] and chunk["date"][:10] > data_time[:10]:
            continue
        if chunk["text"].strip():
            documents.append({"source_id": f"UP-{uploaded['digest']}-{index:03d}",
                              "file": uploaded["name"], "machine_id": machine_id,
                              "equipment_model": model, "date": chunk["date"] or "uploaded",
                              "synthetic": False, "text": chunk["text"]})
    return documents


def retrieve(query: str, machine_id: str, data_time: str, source_dir: Path = SOURCE_DIR,
             uploaded: dict | None = None) -> list[dict]:
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
    documents.extend(uploaded_excerpts(uploaded, machine_id, data_time, model))
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


def prepare_context(context: dict, question: str = "", uploaded: dict | None = None) -> dict:
    """Attach retrieved evidence without changing the classifier's predictions."""
    query = question
    if not query or re.search(r"risk|inspect|condition|worse|风险|检查|恶化", query, re.I):
        features = context.get("feature_sensitivity", [])[:2]
        query += " " + " ".join(feature["feature"] for feature in features)
    return dict(context, retrieved_documents=retrieve(query, context["machine_id"], context["data_time"],
                                                      uploaded=uploaded))


def citation_report(answer: str, documents: list[dict]) -> dict:
    """Check source identifiers, not the factual validity of an LLM answer."""
    cited = set(re.findall(r"\[((?:(?:SYN-MAN|ML|FR)-\d+)|(?:UP-[A-F0-9]{8}-\d+))\]", answer))
    supplied = {document["source_id"] for document in documents}
    return {"cited": sorted(cited & supplied), "invalid": sorted(cited - supplied),
            "uncited": bool(documents) and not bool(cited & supplied)}
