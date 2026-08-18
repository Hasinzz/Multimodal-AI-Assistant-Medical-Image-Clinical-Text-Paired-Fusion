from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.config import DEFAULT_BRAIN_CHECKPOINT, DEFAULT_XRAY_CHECKPOINT, DEFAULT_XRAY_THRESHOLDS, PROJECT_ROOT
from src.evaluation.common import (
    EVALUATION_OUTPUT_DIR,
    THESIS_FIGURES_DIR,
    average,
    ensure_output_dirs,
    safe_ratio,
    save_bar_chart,
    save_csv,
    save_json,
    write_markdown,
)
from src.model1.infer import predict_image
from src.model2.doctor_note_pipeline import run_doctor_note_pipeline
from src.model2.pipeline import run_document_pipeline
from src.model3.pipeline import run_fusion_pipeline


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DOCUMENT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt"}


def _first_file(paths: List[Path], extensions: set[str]) -> Optional[Path]:
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                return path
    return None


def _discover_sample_inputs() -> Dict[str, Path]:
    image_root_candidates = [PROJECT_ROOT / "data" / "images" / "xray", PROJECT_ROOT / "data" / "images" / "brain_mri"]
    document_root_candidates = [PROJECT_ROOT / "data" / "documents" / "prescriptions", PROJECT_ROOT / "data" / "documents" / "lab_reports"]

    image_path = _first_file(image_root_candidates, IMAGE_EXTENSIONS)
    document_path = _first_file(document_root_candidates, DOCUMENT_EXTENSIONS)

    if image_path is None:
        raise FileNotFoundError("No sample image was found under data/images.")
    if document_path is None:
        raise FileNotFoundError("No sample document was found under data/documents.")

    return {"image": image_path, "document": document_path}


def _build_scenarios(sample_image: Path, sample_document: Path) -> List[Dict[str, Any]]:
    synthetic_notes = [
        {
            "chief_complaint": "Headache and dizziness for 2 days.",
            "doctor_note": "Patient reports worsening pain, no known allergy, urgent review requested.",
            "relevant_history": "History of migraine and hypertension.",
            "current_medication_allergy": "Taking ibuprofen. Allergy: penicillin.",
            "report_related_issue": "Concern about abnormal imaging report.",
            "urgency_level": "urgent",
        },
        {
            "chief_complaint": "Chest pain with shortness of breath since yesterday.",
            "doctor_note": "Please correlate with ECG and lab test results.",
            "relevant_history": "History of asthma and smoking.",
            "current_medication_allergy": "Uses inhaler, allergic to latex.",
            "report_related_issue": "Need follow-up on report impression.",
            "urgency_level": "same day",
        },
    ]

    return [
        {"name": "image_only", "image": sample_image, "document": None, "doctor_note": None},
        {"name": "document_only", "image": None, "document": sample_document, "doctor_note": None},
        {"name": "doctor_note_only", "image": None, "document": None, "doctor_note": synthetic_notes[0]},
        {"name": "image_document", "image": sample_image, "document": sample_document, "doctor_note": None},
        {"name": "image_doctor_note", "image": sample_image, "document": None, "doctor_note": synthetic_notes[0]},
        {"name": "document_doctor_note", "image": None, "document": sample_document, "doctor_note": synthetic_notes[1]},
        {"name": "image_document_doctor_note", "image": sample_image, "document": sample_document, "doctor_note": synthetic_notes[1]},
    ]


def _run_case(case: Dict[str, Any]) -> Dict[str, Any]:
    case_id = f"ablation_{case['name']}"
    model1_output = None
    model2_output = None
    doctor_note_output = None

    if case["image"] is not None:
        image_path = str(case["image"])
        modality = "xray"
        checkpoint_path = str(DEFAULT_XRAY_CHECKPOINT)
        thresholds_path = str(DEFAULT_XRAY_THRESHOLDS)
        original_cuda_check = torch.cuda.is_available
        try:
            torch.cuda.is_available = lambda: False  # type: ignore[assignment]
            model1_output = predict_image(
                image_path=image_path,
                modality=modality,
                checkpoint_path=checkpoint_path,
                case_id=case_id,
                thresholds_path=thresholds_path,
            )
        finally:
            torch.cuda.is_available = original_cuda_check  # type: ignore[assignment]

    if case["document"] is not None:
        model2_output = run_document_pipeline(document_path=str(case["document"]), case_id=case_id)

    if case["doctor_note"] is not None:
        doctor_note_output = run_doctor_note_pipeline(case_id=case_id, **case["doctor_note"])

    fusion_output = run_fusion_pipeline(
        case_id=case_id,
        model1_output=model1_output,
        model2_output=model2_output,
        doctor_note_output=doctor_note_output,
    )

    return {
        "case_id": case_id,
        "modality_combination": case["name"],
        "available_modalities": fusion_output.get("available_modalities", []),
        "retrieved_evidence_count": len(fusion_output.get("retrieved_evidence", [])),
        "summary_length": len(str(fusion_output.get("final_summary", ""))),
        "feedback_length": len(str(fusion_output.get("doctor_feedback", fusion_output.get("doctor_oriented_feedback", "")))),
        "missing_information_count": len(fusion_output.get("missing_information_notes", [])),
        "json_completion": int(bool(fusion_output)),
        "model1_present": int(model1_output is not None),
        "model2_present": int(model2_output is not None),
        "doctor_note_present": int(doctor_note_output is not None),
        "fusion_output": fusion_output,
    }


def run_multimodal_ablation() -> Dict[str, Any]:
    ensure_output_dirs()
    sample_inputs = _discover_sample_inputs()
    scenarios = _build_scenarios(sample_inputs["image"], sample_inputs["document"])
    rows = [_run_case(case) for case in scenarios]

    completion_scores = []
    for row in rows:
        modality_score = len(row["available_modalities"])
        evidence_score = row["retrieved_evidence_count"]
        missing_penalty = row["missing_information_count"]
        completion_score = modality_score + evidence_score - missing_penalty
        completion_scores.append(completion_score)
        row["completion_score"] = completion_score

    summary = {
        "scenario_count": len(rows),
        "average_available_modalities": average([float(len(row["available_modalities"])) for row in rows]),
        "average_retrieved_evidence_count": average([float(row["retrieved_evidence_count"]) for row in rows]),
        "average_completion_score": average(completion_scores),
        "average_summary_length": average([float(row["summary_length"]) for row in rows]),
        "average_feedback_length": average([float(row["feedback_length"]) for row in rows]),
        "json_completion_rate": safe_ratio(sum(row["json_completion"] for row in rows), len(rows)),
        "note": "This ablation measures output completeness, retrieval count, and technical execution across modality combinations; it does not claim clinical improvement.",
    }

    save_json(EVALUATION_OUTPUT_DIR / "multimodal_ablation_summary.json", {"summary": summary, "rows": rows})
    save_csv(
        EVALUATION_OUTPUT_DIR / "multimodal_ablation_summary.csv",
        rows,
        [
            "case_id",
            "modality_combination",
            "available_modalities",
            "retrieved_evidence_count",
            "summary_length",
            "feedback_length",
            "missing_information_count",
            "json_completion",
            "completion_score",
            "model1_present",
            "model2_present",
            "doctor_note_present",
        ],
    )
    write_markdown(
        EVALUATION_OUTPUT_DIR / "multimodal_ablation_report.md",
        [
            "# Multimodal Ablation Report",
            "",
            f"Scenario count: {len(rows)}",
            "",
            "## Summary",
            f"- Average available modalities: {summary['average_available_modalities']:.2f}",
            f"- Average retrieved evidence count: {summary['average_retrieved_evidence_count']:.2f}",
            f"- Average completion score: {summary['average_completion_score']:.2f}",
            f"- Average summary length: {summary['average_summary_length']:.1f}",
            f"- Average feedback length: {summary['average_feedback_length']:.1f}",
            f"- JSON completion rate: {summary['json_completion_rate']:.3f}",
            "",
            "This ablation measures output completeness, retrieval count, and technical execution across modality combinations; it does not claim clinical improvement.",
        ],
    )

    save_bar_chart(
        THESIS_FIGURES_DIR / "multimodal_ablation_summary.png",
        "Multimodal Ablation Summary",
        [row["modality_combination"] for row in rows],
        [float(row["completion_score"]) for row in rows],
        "Completion score",
    )

    return summary


def main() -> None:
    summary = run_multimodal_ablation()
    print(summary["note"])
    print(f"Wrote {EVALUATION_OUTPUT_DIR / 'multimodal_ablation_summary.json'}")


if __name__ == "__main__":
    main()
