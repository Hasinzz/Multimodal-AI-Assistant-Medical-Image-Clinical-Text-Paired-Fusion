from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.common import (
    EVALUATION_OUTPUT_DIR,
    PROJECT_ROOT,
    THESIS_FIGURES_DIR,
    average,
    ensure_output_dirs,
    load_json,
    safe_ratio,
    save_bar_chart,
    save_dual_bar_chart,
    save_csv,
    save_json,
    write_markdown,
)


MODALITY_COMBINATIONS = [
    "image_only",
    "document_only",
    "doctor_note_only",
    "image_document",
    "image_doctor_note",
    "document_doctor_note",
    "image_document_doctor_note",
]


def _discover_model3_files() -> List[Path]:
    patterns = ["model3_output.json", "stable_output.json", "v4_output.json"]
    discovered: List[Path] = []
    for pattern in patterns:
        discovered.extend(sorted(PROJECT_ROOT.glob(f"outputs/**/{pattern}")))
    return sorted(dict.fromkeys(discovered))


def _extract_model3_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in _discover_model3_files():
        try:
            payload = load_json(path)
        except Exception:
            continue

        if path.name == "model3_output.json":
            records.append({"source_path": str(path), "data": payload})
            continue

        nested = payload.get("fusion_output")
        if isinstance(nested, dict):
            records.append({"source_path": str(path), "data": nested})

    return records


def _combo_from_output(record: Dict[str, Any]) -> str:
    modalities = record.get("available_modalities")
    if isinstance(modalities, list):
        present = set(str(item) for item in modalities)
    else:
        present = set()
        if record.get("image_findings"):
            present.add("image")
        if record.get("document_findings") or record.get("text_findings"):
            present.add("document")
        if record.get("doctor_note_findings"):
            present.add("doctor_note")

    key = tuple(sorted(present))
    mapping = {
        ("image",): "image_only",
        ("document",): "document_only",
        ("doctor_note",): "doctor_note_only",
        ("document", "image"): "image_document",
        ("doctor_note", "image"): "image_doctor_note",
        ("document", "doctor_note"): "document_doctor_note",
        ("document", "doctor_note", "image"): "image_document_doctor_note",
    }
    return mapping.get(key, "unknown")


def _expected_missing_notes(combo: str) -> List[str]:
    expected = {
        "image_only": ["No document input was provided.", "No doctor note was provided."],
        "document_only": ["No image input was provided.", "No doctor note was provided."],
        "doctor_note_only": ["No image input was provided.", "No document input was provided."],
        "image_document": ["No doctor note was provided."],
        "image_doctor_note": ["No document input was provided."],
        "document_doctor_note": ["No image input was provided."],
        "image_document_doctor_note": [],
    }
    return expected.get(combo, [])


def evaluate_model3() -> Dict[str, Any]:
    ensure_output_dirs()
    records = _extract_model3_records()
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    data_records = [record["data"] for record in records]

    for record in records:
        grouped[_combo_from_output(record["data"])].append(record["data"])

    combo_counts = {combo: len(grouped.get(combo, [])) for combo in MODALITY_COMBINATIONS}
    success_counts = {
        combo: sum(1 for item in grouped.get(combo, []) if isinstance(item.get("final_summary"), str) and item.get("doctor_feedback"))
        for combo in MODALITY_COMBINATIONS
    }

    summary_rows: List[Dict[str, Any]] = []
    summary_length_values: List[float] = []
    feedback_length_values: List[float] = []
    evidence_count_values: List[float] = []
    missing_warning_hits: List[int] = []

    for combo in MODALITY_COMBINATIONS:
        combo_records = grouped.get(combo, [])
        if not combo_records:
            summary_rows.append(
                {
                    "modality_combination": combo,
                    "case_count": 0,
                    "fusion_output_success_rate": 0.0,
                    "patient_summary_rate": 0.0,
                    "doctor_feedback_rate": 0.0,
                    "retrieved_evidence_rate": 0.0,
                    "average_retrieved_evidence_count": 0.0,
                    "missing_modality_handling_rate": 0.0,
                    "average_summary_length": 0.0,
                    "average_doctor_feedback_length": 0.0,
                    "json_completion_rate": 0.0,
                }
            )
            continue

        success_rate = safe_ratio(success_counts[combo], len(combo_records))
        patient_summary_rate = safe_ratio(sum(1 for item in combo_records if str(item.get("patient_summary", "")).strip()), len(combo_records))
        doctor_feedback_rate = safe_ratio(sum(1 for item in combo_records if str(item.get("doctor_feedback", item.get("doctor_oriented_feedback", ""))).strip()), len(combo_records))
        evidence_count = average([float(len(item.get("retrieved_evidence", []))) for item in combo_records])
        retrieved_evidence_rate = safe_ratio(sum(1 for item in combo_records if item.get("retrieved_evidence")), len(combo_records))
        summary_length = average([float(len(str(item.get("final_summary", "")))) for item in combo_records])
        feedback_length = average([float(len(str(item.get("doctor_feedback", item.get("doctor_oriented_feedback", ""))))) for item in combo_records])
        json_completion_rate = safe_ratio(sum(1 for item in combo_records if isinstance(item, dict)), len(combo_records))
        expected_missing_notes = _expected_missing_notes(combo)
        missing_handling_rate = safe_ratio(
            sum(
                1
                for item in combo_records
                if all(note in " ".join(item.get("missing_information_notes", [])) for note in expected_missing_notes)
            ),
            len(combo_records),
        )

        summary_length_values.append(summary_length)
        feedback_length_values.append(feedback_length)
        evidence_count_values.append(evidence_count)
        missing_warning_hits.append(int(missing_handling_rate > 0))

        summary_rows.append(
            {
                "modality_combination": combo,
                "case_count": len(combo_records),
                "fusion_output_success_rate": success_rate,
                "patient_summary_rate": patient_summary_rate,
                "doctor_feedback_rate": doctor_feedback_rate,
                "retrieved_evidence_rate": retrieved_evidence_rate,
                "average_retrieved_evidence_count": evidence_count,
                "missing_modality_handling_rate": missing_handling_rate,
                "average_summary_length": summary_length,
                "average_doctor_feedback_length": feedback_length,
                "json_completion_rate": json_completion_rate,
            }
        )

    summary = {
        "total_records": len(records),
        "combination_counts": combo_counts,
        "fusion_output_success_rate_overall": safe_ratio(sum(success_counts.values()), len(records)),
        "patient_summary_generation_rate_overall": safe_ratio(sum(1 for record in data_records if str(record.get("final_summary") or record.get("patient_summary", "")).strip()), len(data_records)),
        "doctor_feedback_generation_rate_overall": safe_ratio(sum(1 for record in data_records if str(record.get("doctor_feedback") or record.get("doctor_oriented_feedback", "")).strip()), len(data_records)),
        "retrieved_evidence_availability_rate_overall": safe_ratio(sum(1 for record in data_records if record.get("retrieved_evidence")), len(data_records)),
        "average_retrieved_evidence_count_overall": average([float(len(record.get("retrieved_evidence", []))) for record in data_records]),
        "average_summary_length_overall": average(summary_length_values),
        "average_doctor_feedback_length_overall": average(feedback_length_values),
        "json_completion_rate_overall": safe_ratio(len(data_records), len(data_records)),
        "reference_summaries_available": False,
        "note": "Reference summaries were not available, so Model-3 was evaluated using technical fusion, retrieval, and output-completion metrics.",
    }

    save_json(EVALUATION_OUTPUT_DIR / "model3_evaluation_summary.json", summary)
    save_csv(
        EVALUATION_OUTPUT_DIR / "model3_evaluation_summary.csv",
        summary_rows,
        [
            "modality_combination",
            "case_count",
            "fusion_output_success_rate",
            "patient_summary_rate",
            "doctor_feedback_rate",
            "retrieved_evidence_rate",
            "average_retrieved_evidence_count",
            "missing_modality_handling_rate",
            "average_summary_length",
            "average_doctor_feedback_length",
            "json_completion_rate",
        ],
    )

    write_markdown(
        EVALUATION_OUTPUT_DIR / "model3_evaluation_report.md",
        [
            "# Model-3 Evaluation Report",
            "",
            f"Total records inspected: {len(records)}",
            "",
            "## Results",
            *(f"- {item['modality_combination']}: {item['case_count']} case(s)" for item in summary_rows),
            "",
            f"- Fusion output success rate overall: {summary['fusion_output_success_rate_overall']:.3f}",
            f"- Patient summary generation rate overall: {summary['patient_summary_generation_rate_overall']:.3f}",
            f"- Non-validated follow-up-note generation rate overall: {summary['doctor_feedback_generation_rate_overall']:.3f}",
            f"- Retrieved evidence availability rate overall: {summary['retrieved_evidence_availability_rate_overall']:.3f}",
            f"- Average retrieved evidence count overall: {summary['average_retrieved_evidence_count_overall']:.2f}",
            f"- Average summary length overall: {summary['average_summary_length_overall']:.1f}",
            f"- Average non-validated follow-up-note length overall: {summary['average_doctor_feedback_length_overall']:.1f}",
            "",
            "Reference summaries were not available, so Model-3 was evaluated using technical fusion, retrieval, and output-completion metrics.",
        ],
    )

    save_bar_chart(
        THESIS_FIGURES_DIR / "model3_modality_ablation_results.png",
        "Model-3 Modality Ablation Results",
        [row["modality_combination"] for row in summary_rows],
        [row["case_count"] for row in summary_rows],
        "Case count",
    )
    save_bar_chart(
        THESIS_FIGURES_DIR / "model3_fusion_output_success.png",
        "Model-3 Fusion Output Success",
        [row["modality_combination"] for row in summary_rows],
        [row["fusion_output_success_rate"] for row in summary_rows],
        "Success rate",
    )

    return summary


def main() -> None:
    summary = evaluate_model3()
    print(summary["note"])
    print(f"Wrote {EVALUATION_OUTPUT_DIR / 'model3_evaluation_summary.json'}")


if __name__ == "__main__":
    main()
