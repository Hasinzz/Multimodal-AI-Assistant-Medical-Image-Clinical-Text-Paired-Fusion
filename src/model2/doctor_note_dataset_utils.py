from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEARCH_ROOTS = [PROJECT_ROOT, PROJECT_ROOT / "data", PROJECT_ROOT / "datasets"]
EXPECTED_TEXT_COLUMNS = ["description", "medical_specialty", "sample_name", "transcription", "keywords"]

SYMPTOM_TERMS = [
    "headache",
    "fever",
    "cough",
    "pain",
    "nausea",
    "vomiting",
    "dizziness",
    "weakness",
    "shortness of breath",
    "sob",
    "chest pain",
    "fatigue",
    "swelling",
    "blurred vision",
    "numbness",
    "seizure",
    "infection",
    "rash",
    "abdominal pain",
    "back pain",
]

HISTORY_TERMS = [
    "hypertension",
    "diabetes",
    "asthma",
    "copd",
    "smoking",
    "surgery",
    "pregnant",
    "pregnancy",
    "cancer",
    "stroke",
    "allergy",
    "prior",
    "history",
    "family history",
    "medication",
]

MEDICATION_TERMS = [
    "paracetamol",
    "acetaminophen",
    "ibuprofen",
    "aspirin",
    "amoxicillin",
    "metformin",
    "insulin",
    "lisinopril",
    "amlodipine",
    "prednisone",
    "antibiotic",
    "medication",
    "tablet",
    "capsule",
    "dose",
]

ALLERGY_TERMS = [
    "allergy",
    "allergic",
    "penicillin",
    "latex",
    "food allergy",
    "drug allergy",
]

TEST_TERMS = [
    "xray",
    "x-ray",
    "mri",
    "ct",
    "ultrasound",
    "cbc",
    "cmp",
    "blood test",
    "lab",
    "ecg",
    "ekg",
    "urine",
    "biopsy",
    "culture",
]

CONCERN_TERMS = [
    "concern",
    "suspect",
    "suspected",
    "rule out",
    "follow up",
    "evaluate",
    "assessment",
    "diagnosis",
    "infection",
    "fracture",
    "tumor",
    "pneumonia",
    "stroke",
    "anemia",
    "allergy",
    "medication reaction",
]

URGENCY_TERMS = [
    "urgent",
    "stat",
    "asap",
    "emergency",
    "immediate",
    "same day",
    "high priority",
    "critical",
    "severe",
]


@dataclass(frozen=True)
class MTSamplesLocation:
    source_path: Path
    source_type: str
    archive_member: Optional[str]
    size_bytes: int


def _normalize_whitespace(text: Any) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\t\f\v]+", " ", value)
    value = re.sub(r"[ ]{2,}", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def clean_doctor_text(text: Any) -> str:
    value = _normalize_whitespace(text)
    value = re.sub(r"\s*:\s*", ": ", value)
    value = re.sub(r"[•\u2022]+", "-", value)
    return _normalize_whitespace(value)


def normalize_column_name(name: str) -> str:
    value = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return re.sub(r"_{2,}", "_", value).strip("_")


def normalize_mtsamples_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: normalize_column_name(column) for column in frame.columns}
    frame = frame.rename(columns=renamed).copy()

    for expected in EXPECTED_TEXT_COLUMNS:
        if expected not in frame.columns:
            frame[expected] = ""

    return frame


def _candidate_paths(search_roots: Optional[Iterable[Path]] = None) -> List[Path]:
    roots = list(search_roots or DEFAULT_SEARCH_ROOTS)
    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            if name == "mtsamples.csv" or (path.suffix.lower() == ".zip" and name == "archive.zip"):
                candidates.append(path)
    return sorted(dict.fromkeys(candidates))


def discover_mtsamples_location(search_roots: Optional[Iterable[Path]] = None) -> Optional[MTSamplesLocation]:
    for path in _candidate_paths(search_roots):
        if path.suffix.lower() == ".csv":
            return MTSamplesLocation(
                source_path=path,
                source_type="csv",
                archive_member=None,
                size_bytes=path.stat().st_size,
            )

        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if member.lower().endswith("mtsamples.csv"):
                        return MTSamplesLocation(
                            source_path=path,
                            source_type="zip",
                            archive_member=member,
                            size_bytes=path.stat().st_size,
                        )

    return None


def load_mtsamples_frame(location: MTSamplesLocation) -> pd.DataFrame:
    if location.source_type == "csv":
        frame = pd.read_csv(location.source_path, encoding="utf-8", low_memory=False)
        return normalize_mtsamples_columns(frame)

    with zipfile.ZipFile(location.source_path) as archive:
        member = location.archive_member
        if member is None:
            for candidate in archive.namelist():
                if candidate.lower().endswith("mtsamples.csv"):
                    member = candidate
                    break

        if member is None:
            raise FileNotFoundError(f"No mtsamples.csv member found inside {location.source_path}")

        with archive.open(member) as raw_handle:
            text_handle = TextIOWrapper(raw_handle, encoding="utf-8", newline="")
            frame = pd.read_csv(text_handle, low_memory=False)

    return normalize_mtsamples_columns(frame)


def summarize_missing_values(frame: pd.DataFrame) -> Dict[str, int]:
    return {column: int(frame[column].isna().sum()) for column in frame.columns}


def _find_terms(text: str, terms: Iterable[str]) -> List[str]:
    lowered = _normalize_whitespace(text).lower()
    found: List[str] = []
    for term in terms:
        pattern = re.escape(term.lower()).replace(r"\ ", r"\s+")
        if re.search(rf"\b{pattern}\b", lowered):
            found.append(term)
    return sorted(dict.fromkeys(found))


def extract_weak_entity_lists(text: str, keywords: Optional[str] = None) -> Dict[str, List[str]]:
    combined_text = _normalize_whitespace(text)
    keyword_text = _normalize_whitespace(keywords)
    searchable = "\n".join(value for value in [combined_text, keyword_text] if value)

    urgency_matches = _find_terms(searchable, URGENCY_TERMS)
    duration_matches = re.findall(
        r"\b(?:for\s+)?(?:\d+\s*(?:day|days|week|weeks|month|months|year|years|hour|hours)|"
        r"(?:since\s+)?(?:yesterday|today|last\s+night|this\s+morning|this\s+week|last\s+week))\b",
        searchable,
        flags=re.IGNORECASE,
    )

    entities = {
        "SYMPTOM": _find_terms(searchable, SYMPTOM_TERMS),
        "HISTORY": _find_terms(searchable, HISTORY_TERMS),
        "MEDICATION": _find_terms(searchable, MEDICATION_TERMS),
        "ALLERGY": _find_terms(searchable, ALLERGY_TERMS),
        "TEST": _find_terms(searchable, TEST_TERMS),
        "DIAGNOSIS_OR_CONCERN": _find_terms(searchable, CONCERN_TERMS),
        "URGENCY": urgency_matches,
    }

    if duration_matches:
        entities["HISTORY"] = sorted(dict.fromkeys(entities["HISTORY"] + [match.strip() for match in duration_matches if match.strip()]))

    if keyword_text:
        keyword_terms = re.split(r"[,;/|]+", keyword_text)
        for term in keyword_terms:
            cleaned = term.strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if any(keyword in lowered for keyword in ["pain", "fever", "cough", "nausea", "dizziness", "weakness", "rash", "fatigue"]):
                entities["SYMPTOM"].append(cleaned)
            if any(keyword in lowered for keyword in ["history", "hypertension", "diabetes", "asthma", "stroke", "cancer"]):
                entities["HISTORY"].append(cleaned)
            if any(keyword in lowered for keyword in ["tablet", "capsule", "mg", "medication", "dose", "aspirin", "ibuprofen", "insulin"]):
                entities["MEDICATION"].append(cleaned)
            if any(keyword in lowered for keyword in ["allergy", "allergic", "penicillin", "latex"]):
                entities["ALLERGY"].append(cleaned)
            if any(keyword in lowered for keyword in ["xray", "mri", "ct", "ecg", "ekg", "lab", "test", "biopsy", "culture"]):
                entities["TEST"].append(cleaned)
            if any(keyword in lowered for keyword in ["concern", "rule out", "diagnosis", "infection", "fracture", "pneumonia", "anemia"]):
                entities["DIAGNOSIS_OR_CONCERN"].append(cleaned)
            if any(keyword in lowered for keyword in ["urgent", "stat", "asap", "emergency", "immediate", "critical"]):
                entities["URGENCY"].append(cleaned)

    return {label: sorted(dict.fromkeys(values)) for label, values in entities.items()}


def extract_weak_entity_spans(text: str) -> List[Dict[str, Any]]:
    cleaned_text = _normalize_whitespace(text)
    lowered_text = cleaned_text.lower()
    spans: List[Dict[str, Any]] = []

    category_terms = {
        "SYMPTOM": SYMPTOM_TERMS,
        "HISTORY": HISTORY_TERMS,
        "MEDICATION": MEDICATION_TERMS,
        "ALLERGY": ALLERGY_TERMS,
        "TEST": TEST_TERMS,
        "DIAGNOSIS_OR_CONCERN": CONCERN_TERMS,
        "URGENCY": URGENCY_TERMS,
    }

    seen = set()

    for label, terms in category_terms.items():
        for term in terms:
            pattern = re.escape(term.lower()).replace(r"\ ", r"\s+")
            for match in re.finditer(rf"\b{pattern}\b", lowered_text):
                start, end = match.span()
                key = (start, end, label)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(
                    {
                        "label": label,
                        "text": cleaned_text[start:end],
                        "start": start,
                        "end": end,
                    }
                )

    spans.sort(key=lambda item: (item["start"], item["end"], item["label"]))
    return spans


def count_total_weak_entities(weak_entities: Dict[str, List[str]]) -> int:
    return sum(len(values) for values in weak_entities.values())


def weak_entity_category_counts(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts = {label: 0 for label in ["SYMPTOM", "HISTORY", "MEDICATION", "ALLERGY", "TEST", "DIAGNOSIS_OR_CONCERN", "URGENCY"]}
    for record in records:
        entities = record.get("weak_entities") or {}
        for label, values in entities.items():
            if label in counts:
                counts[label] += len(values or [])
    return counts


def safe_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)
