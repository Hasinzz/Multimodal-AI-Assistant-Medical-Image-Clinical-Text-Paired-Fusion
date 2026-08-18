from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib

from src.model2.doctor_note_dataset_utils import clean_doctor_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL2_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "model2"
DOCTOR_NOTE_CLASSIFIER_PATH = MODEL2_CHECKPOINT_DIR / "doctor_note_classifier_tfidf.joblib"
DOCTOR_NOTE_WEAK_NER_DIR = MODEL2_CHECKPOINT_DIR / "doctor_note_weak_ner"


_SYMPTOM_TERMS = [
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

_CONCERN_TERMS = [
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

_URGENCY_TERMS = [
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

_TEST_TERMS = [
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


def _normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_field(value: Optional[str]) -> str:
    return _normalize_whitespace(value or "")


def build_doctor_note_text(
    chief_complaint: Optional[str] = None,
    doctor_note: Optional[str] = None,
    relevant_history: Optional[str] = None,
    current_medication_allergy: Optional[str] = None,
    report_related_issue: Optional[str] = None,
    urgency_level: Optional[str] = None,
) -> str:
    sections = []

    field_map = [
        ("Chief complaint / symptoms", chief_complaint),
        ("Doctor note", doctor_note),
        ("Relevant history", relevant_history),
        ("Current medication / allergy", current_medication_allergy),
        ("Report-related issue / concern", report_related_issue),
        ("Urgency level", urgency_level),
    ]

    for label, value in field_map:
        cleaned = _clean_field(value)
        if cleaned:
            sections.append(f"{label}: {cleaned}")

    return _normalize_whitespace("\n".join(sections))


def _find_terms(text: str, terms: Sequence[str]) -> List[str]:
    lowered = text.lower()
    found: List[str] = []
    for term in terms:
        pattern = re.escape(term.lower()).replace(r"\ ", r"\s+")
        if re.search(rf"\b{pattern}\b", lowered):
            found.append(term)
    return sorted(dict.fromkeys(found))


def _extract_duration_mentions(text: str) -> List[str]:
    pattern = re.compile(
        r"\b(?:for\s+)?(?:\d+\s*(?:day|days|week|weeks|month|months|year|years|hour|hours)|"
        r"(?:since\s+)?(?:yesterday|today|last\s+night|this\s+morning|this\s+week|last\s+week))\b",
        re.IGNORECASE,
    )
    matches = [match.group(0).strip() for match in pattern.finditer(text)]
    return sorted(dict.fromkeys(matches))


def _extract_history_mentions(text: str) -> List[str]:
    history_terms = [
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
    return _find_terms(text, history_terms)


def _extract_medications(text: str) -> List[str]:
    medication_terms = [
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
    return _find_terms(text, medication_terms)


def _extract_allergies(text: str) -> List[str]:
    allergy_terms = [
        "allergy",
        "allergic",
        "penicillin",
        "latex",
        "food allergy",
        "drug allergy",
    ]
    return _find_terms(text, allergy_terms)


def _extract_symptoms(text: str) -> List[str]:
    return _find_terms(text, _SYMPTOM_TERMS)


def _extract_tests(text: str) -> List[str]:
    return _find_terms(text, _TEST_TERMS)


def _extract_concerns(text: str) -> List[str]:
    concerns = _find_terms(text, _CONCERN_TERMS)
    if not concerns:
        return []

    lowered = text.lower()
    extra_matches = []
    for sentence in re.split(r"[\n\.\;]+", lowered):
        if any(term in sentence for term in concerns):
            extra_matches.append(sentence.strip())
    return sorted(dict.fromkeys([item for item in extra_matches if item])) or concerns


def _extract_urgency(text: str, urgency_level: Optional[str]) -> List[str]:
    urgency_terms = _find_terms(text, _URGENCY_TERMS)
    manual_level = _clean_field(urgency_level)
    if manual_level:
        urgency_terms.append(manual_level)
    return sorted(dict.fromkeys(urgency_terms))


def _join_phrase(items: Iterable[str]) -> str:
    values = [item for item in items if item]
    if not values:
        return "no explicit clinical details were detected"
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"


def _load_classifier_bundle() -> Optional[Dict[str, object]]:
    if not DOCTOR_NOTE_CLASSIFIER_PATH.exists():
        return None
    try:
        return joblib.load(DOCTOR_NOTE_CLASSIFIER_PATH)
    except Exception:
        return None


def _predict_specialty(clean_text: str, classifier_bundle: Optional[Dict[str, object]]) -> Tuple[str, float]:
    if not classifier_bundle:
        return "unknown", 0.0

    vectorizer = classifier_bundle.get("vectorizer")
    classifier = classifier_bundle.get("classifier")
    if vectorizer is None or classifier is None:
        return "unknown", 0.0

    try:
        feature_matrix = vectorizer.transform([clean_text])
        predicted_specialty = classifier.predict(feature_matrix)[0]
        confidence = 0.0
        if hasattr(classifier, "predict_proba"):
            probabilities = classifier.predict_proba(feature_matrix)[0]
            confidence = float(max(probabilities)) if len(probabilities) else 0.0
        return str(predicted_specialty), confidence
    except Exception:
        return "unknown", 0.0


def _load_ner_bundle() -> Optional[Tuple[object, object, Dict[int, str]]]:
    label_path = DOCTOR_NOTE_WEAK_NER_DIR / "label_list.json"
    if not DOCTOR_NOTE_WEAK_NER_DIR.exists() or not label_path.exists():
        return None

    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        import torch
    except Exception:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(DOCTOR_NOTE_WEAK_NER_DIR), local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(str(DOCTOR_NOTE_WEAK_NER_DIR), local_files_only=True)
        label_list = json.loads(label_path.read_text(encoding="utf-8"))
        id2label = {index: label for index, label in enumerate(label_list)}
        _ = torch  # keep import checked
        return tokenizer, model, id2label
    except Exception:
        return None


def _extract_entities_with_ner(clean_text: str, ner_bundle: Optional[Tuple[object, object, Dict[int, str]]]) -> Optional[Dict[str, List[str]]]:
    if not ner_bundle:
        return None

    tokenizer, model, id2label = ner_bundle
    try:
        import torch
    except Exception:
        return None

    try:
        encoded = tokenizer(
            clean_text,
            truncation=True,
            max_length=256,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        predicted_ids = logits.argmax(dim=-1).tolist()

        entities = {
            "symptoms": [],
            "history": [],
            "medications": [],
            "allergies": [],
            "tests": [],
            "diagnosis_or_concern": [],
            "urgency_terms": [],
        }

        active_label = None
        active_start = None
        active_end = None

        def flush_entity() -> None:
            nonlocal active_label, active_start, active_end
            if active_label is None or active_start is None or active_end is None:
                return
            text = clean_text[active_start:active_end].strip()
            if text:
                key = {
                    "SYMPTOM": "symptoms",
                    "HISTORY": "history",
                    "MEDICATION": "medications",
                    "ALLERGY": "allergies",
                    "TEST": "tests",
                    "DIAGNOSIS_OR_CONCERN": "diagnosis_or_concern",
                    "URGENCY": "urgency_terms",
                }.get(active_label)
                if key and text not in entities[key]:
                    entities[key].append(text)
            active_label = None
            active_start = None
            active_end = None

        for index, predicted_id in enumerate(predicted_ids):
            start, end = offsets[index]
            if start == end:
                continue
            label = id2label.get(predicted_id, "O")
            if label == "O":
                flush_entity()
                continue
            prefix, entity_label = label.split("-", 1)
            if prefix == "B" or entity_label != active_label or active_start is None:
                flush_entity()
                active_label = entity_label
                active_start = start
                active_end = end
            else:
                active_end = end
        flush_entity()
        return entities
    except Exception:
        return None


def _rule_based_entities(clean_text: str, urgency_level: Optional[str]) -> Dict[str, List[str]]:
    return {
        "symptoms": _extract_symptoms(clean_text),
        "duration": _extract_duration_mentions(clean_text),
        "history": _extract_history_mentions(clean_text),
        "medications": _extract_medications(clean_text),
        "allergies": _extract_allergies(clean_text),
        "tests": _extract_tests(clean_text),
        "diagnosis_or_concern": _extract_concerns(clean_text),
        "urgency_terms": _extract_urgency(clean_text, urgency_level),
    }


def build_patient_summary_text(entities: Dict[str, List[str]]) -> str:
    symptoms = _join_phrase(entities.get("symptoms", []))
    history = _join_phrase(entities.get("history", []))
    medications = _join_phrase(entities.get("medications", []))
    allergies = _join_phrase(entities.get("allergies", []))
    tests = _join_phrase(entities.get("tests", []))
    concern = _join_phrase(entities.get("diagnosis_or_concern", []))
    urgency = _join_phrase(entities.get("urgency_terms", []))

    parts = [
        f"Symptoms/chief complaint: {symptoms}.",
        f"Relevant history: {history}.",
        f"Current medication: {medications}.",
        f"Allergies: {allergies}.",
        f"Test mentions: {tests}.",
        f"Concern or diagnosis cues: {concern}.",
        f"Urgency cues: {urgency}.",
    ]
    return " ".join(parts)


def run_doctor_note_pipeline(
    *,
    chief_complaint: Optional[str] = None,
    doctor_note: Optional[str] = None,
    relevant_history: Optional[str] = None,
    current_medication_allergy: Optional[str] = None,
    report_related_issue: Optional[str] = None,
    urgency_level: Optional[str] = None,
    case_id: str = "case_001",
    mode: str = "auto",
) -> Dict[str, object]:
    raw_text = build_doctor_note_text(
        chief_complaint=chief_complaint,
        doctor_note=doctor_note,
        relevant_history=relevant_history,
        current_medication_allergy=current_medication_allergy,
        report_related_issue=report_related_issue,
        urgency_level=urgency_level,
    )
    clean_text = clean_doctor_text(raw_text)

    classifier_bundle = _load_classifier_bundle() if mode in {"auto", "trained"} else None
    ner_bundle = _load_ner_bundle() if mode in {"auto", "trained"} else None

    trained_components = []
    if classifier_bundle:
        trained_components.append("classifier")
    if ner_bundle:
        trained_components.append("ner")

    use_trained = mode == "trained" or (mode == "auto" and trained_components)

    predicted_specialty, specialty_confidence = _predict_specialty(clean_text, classifier_bundle if use_trained else None)

    entities = _extract_entities_with_ner(clean_text, ner_bundle if use_trained else None)
    entity_mode = "trained" if entities is not None else "rule_based"
    if entities is None:
        entities = _rule_based_entities(clean_text, urgency_level)
    else:
        entities.setdefault("duration", _extract_duration_mentions(clean_text))

    return {
        "case_id": case_id,
        "input_type": "doctor_note",
        "mode": "trained" if use_trained else "rule_based",
        "trained_components": trained_components,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "predicted_specialty": predicted_specialty,
        "specialty_confidence": specialty_confidence,
        "entity_mode": entity_mode,
        "entities": entities,
        "patient_summary_text": build_patient_summary_text(entities),
        "doctor_note_available": bool(clean_text),
    }
