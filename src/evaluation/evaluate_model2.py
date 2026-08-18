from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.evaluation.common import (
    EVALUATION_OUTPUT_DIR,
    PROJECT_ROOT,
    THESIS_FIGURES_DIR,
    average,
    ensure_output_dirs,
    load_json,
    safe_ratio,
    save_bar_chart,
    save_csv,
    save_json,
    write_markdown,
)


def _discover_candidate_files() -> List[Path]:
    patterns = ["model2_output.json", "stable_output.json", "v4_output.json", "doctor_note_inference_outputs.jsonl"]
    discovered: List[Path] = []
    for pattern in patterns:
        discovered.extend(sorted(PROJECT_ROOT.glob(f"outputs/**/{pattern}")))
    return sorted(dict.fromkeys(discovered))


def _extract_model2_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in _discover_candidate_files():
        try:
            if path.suffix.lower() == ".jsonl":
                payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            else:
                payload = load_json(path)
        except Exception:
            continue

        if path.name == "model2_output.json":
            records.append(
                {
                    "source_path": str(path),
                    "record_type": "direct",
                    "data": payload,
                }
            )
            continue

        if path.suffix.lower() == ".jsonl":
            for index, item in enumerate(payload):
                nested = item.get("doctor_note_output") if isinstance(item, dict) else None
                if isinstance(nested, dict):
                    records.append(
                        {
                            "source_path": str(path),
                            "record_type": "doctor_note_inference",
                            "data": nested,
                            "record_index": index,
                            "branch": "doctor_note",
                        }
                    )
            continue

        nested = payload.get("model2_output")
        if isinstance(nested, dict):
            records.append(
                {
                    "source_path": str(path),
                    "record_type": "nested",
                    "data": nested,
                }
            )

    return records


def _entity_count(record: Dict[str, Any]) -> int:
    entities = record.get("extracted_entities") or record.get("entities") or []
    if isinstance(entities, list):
        return len(entities)
    if isinstance(entities, dict):
        return sum(len(value) for value in entities.values() if isinstance(value, list))
    return 0


def _structured_json_complete(record: Dict[str, Any]) -> bool:
    structured_json = record.get("structured_json")
    return isinstance(structured_json, dict)


def _record_field_completion(record: Dict[str, Any]) -> int:
    required_fields = ["raw_text", "raw_text_preview", "entities", "structured_json", "patient_summary"]
    return sum(1 for field in required_fields if field in record and record.get(field) not in (None, "", [], {}))


def _doctor_note_record_metrics(record: Dict[str, Any]) -> Dict[str, Any]:
    entities = record.get("entities") or {}
    return {
        "predicted_specialty": record.get("predicted_specialty", ""),
        "specialty_confidence": record.get("specialty_confidence", 0.0),
        "entity_count": _entity_count(record),
        "summary_length": len(str(record.get("patient_summary_text", ""))),
        "structured_complete": int(bool(record.get("doctor_note_available")) and bool(record.get("patient_summary_text"))),
        "has_specialty": int(bool(str(record.get("predicted_specialty", "")).strip()) and str(record.get("predicted_specialty", "")).lower() != "unknown"),
        "symptom_count": len(entities.get("symptoms", []) if isinstance(entities, dict) else []),
    }


def evaluate_model2() -> Dict[str, Any]:
    ensure_output_dirs()
    records = _extract_model2_records()

    document_records = [record for record in records if record["data"].get("input_type") != "doctor_note" and record.get("record_type") != "doctor_note_inference"]
    doctor_note_records = [record for record in records if record["data"].get("input_type") == "doctor_note" or record.get("record_type") == "doctor_note_inference"]

    total_records = len(records)
    ocr_success_count = sum(1 for record in document_records if str(record["data"].get("raw_text", "")).strip())
    empty_ocr_count = sum(1 for record in document_records if not str(record["data"].get("raw_text", "")).strip())
    document_entity_success_count = sum(1 for record in document_records if _entity_count(record["data"]) > 0)
    document_structured_complete_count = sum(1 for record in document_records if _structured_json_complete(record["data"]))
    document_field_completion_scores = [_record_field_completion(record["data"]) / 5.0 for record in document_records if record["data"]]
    document_average_entities = average([float(_entity_count(record["data"])) for record in document_records])
    overall_entity_success_count = sum(1 for record in records if _entity_count(record["data"]) > 0)
    overall_structured_complete_count = sum(1 for record in records if _structured_json_complete(record["data"]))
    overall_field_completion_scores = [_record_field_completion(record["data"]) / 5.0 for record in records if record["data"]]
    overall_average_entities = average([float(_entity_count(record["data"])) for record in records])
    average_ocr_length = average([float(len(str(record["data"].get("raw_text", "")))) for record in document_records])
    doctor_note_success_count = sum(1 for record in doctor_note_records if bool(record["data"].get("doctor_note_available")))
    doctor_note_average_entities = average([float(_entity_count(record["data"])) for record in doctor_note_records])
    doctor_note_average_summary_length = average([float(len(str(record["data"].get("patient_summary_text", "")))) for record in doctor_note_records])
    doctor_note_specialty_rate = safe_ratio(
        sum(1 for record in doctor_note_records if str(record["data"].get("predicted_specialty", "")).strip() and str(record["data"].get("predicted_specialty", "")).lower() != "unknown"),
        len(doctor_note_records),
    )
    doctor_note_structured_completion = safe_ratio(
        sum(1 for record in doctor_note_records if bool(record["data"].get("doctor_note_available")) and bool(record["data"].get("patient_summary_text"))),
        len(doctor_note_records),
    )

    ground_truth_available = any(
        key in record["data"] for record in records for key in ["ground_truth", "reference_text", "reference_entities"]
    )

    weak_ner_metrics_path = PROJECT_ROOT / "outputs" / "doctor_note_training" / "weak_ner_metrics.json"

    summary = {
        "total_records": total_records,
        "document_records": len(document_records),
        "doctor_note_records": len(doctor_note_records),
        "ocr_success_rate": safe_ratio(ocr_success_count, len(document_records)),
        "average_ocr_text_length": average_ocr_length,
        "empty_ocr_output_count": empty_ocr_count,
        "entity_extraction_success_rate": safe_ratio(document_entity_success_count, len(document_records)),
        "average_entities_per_record": document_average_entities,
        "structured_json_completion_rate": safe_ratio(document_structured_complete_count, len(document_records)),
        "average_field_completion_rate": average(document_field_completion_scores),
        "document_entity_extraction_success_rate": safe_ratio(document_entity_success_count, len(document_records)),
        "document_average_entities_per_record": document_average_entities,
        "document_structured_json_completion_rate": safe_ratio(document_structured_complete_count, len(document_records)),
        "document_average_field_completion_rate": average(document_field_completion_scores),
        "overall_entity_extraction_success_rate": safe_ratio(overall_entity_success_count, total_records),
        "overall_average_entities_per_record": overall_average_entities,
        "overall_structured_json_completion_rate": safe_ratio(overall_structured_complete_count, total_records),
        "overall_average_field_completion_rate": average(overall_field_completion_scores),
        "doctor_note_extraction_success_rate": safe_ratio(doctor_note_success_count, len(doctor_note_records)),
        "doctor_note_average_entities_per_record": doctor_note_average_entities,
        "doctor_note_average_summary_length": doctor_note_average_summary_length,
        "doctor_note_predicted_specialty_availability_rate": doctor_note_specialty_rate,
        "doctor_note_structured_completion_rate": doctor_note_structured_completion,
        "ground_truth_ocr_entity_labels_available": ground_truth_available,
        "weak_label_ner_metrics_available": weak_ner_metrics_path.exists(),
        "note": "Ground-truth OCR/entity labels were not available, so Model-2 was evaluated using technical extraction and structured-output metrics.",
    }

    by_source_rows: List[Dict[str, Any]] = []
    for record in records:
        data = record["data"]
        by_source_rows.append(
            {
                "source_path": record["source_path"],
                "record_type": record["record_type"],
                "input_type": data.get("input_type", "document"),
                "raw_text_length": len(str(data.get("raw_text", ""))),
                "entity_count": _entity_count(data),
                "structured_json_complete": int(_structured_json_complete(data)),
                "field_completion_count": _record_field_completion(data),
                "branch": record.get("branch", "document" if data.get("input_type") != "doctor_note" else "doctor_note"),
            }
        )

    save_json(EVALUATION_OUTPUT_DIR / "model2_evaluation_summary.json", summary)
    save_csv(
        EVALUATION_OUTPUT_DIR / "model2_evaluation_summary.csv",
        by_source_rows,
        ["source_path", "record_type", "input_type", "branch", "raw_text_length", "entity_count", "structured_json_complete", "field_completion_count"],
    )

    write_markdown(
        EVALUATION_OUTPUT_DIR / "model2_evaluation_report.md",
        [
            "# Model-2 Evaluation Report",
            "",
            f"Total records inspected: {total_records}",
            f"Document records: {len(document_records)}",
            f"Doctor-note records: {len(doctor_note_records)}",
            "",
            "## A. Document/OCR Branch Results",
            f"- Document records: {len(document_records)}",
            f"- OCR success rate: {summary['ocr_success_rate']:.3f}",
            f"- Average OCR text length: {summary['average_ocr_text_length']:.1f}",
            f"- Empty OCR output count: {summary['empty_ocr_output_count']}",
            f"- Entity extraction success rate: {summary['entity_extraction_success_rate']:.3f}",
            f"- Average entities per record: {summary['average_entities_per_record']:.2f}",
            f"- Structured JSON completion rate: {summary['structured_json_completion_rate']:.3f}",
            f"- Average field completion rate: {summary['average_field_completion_rate']:.3f}",
            "",
            "## B. Doctor-Note Branch Results",
            f"- Doctor-note records: {len(doctor_note_records)}",
            f"- Doctor-note extraction success rate: {summary['doctor_note_extraction_success_rate']:.3f}",
            f"- Doctor-note average entities per record: {summary['doctor_note_average_entities_per_record']:.2f}",
            f"- Doctor-note average summary length: {summary['doctor_note_average_summary_length']:.1f}",
            f"- Doctor-note predicted specialty availability rate: {summary['doctor_note_predicted_specialty_availability_rate']:.3f}",
            f"- Doctor-note structured completion rate: {summary['doctor_note_structured_completion_rate']:.3f}",
            f"- Weak-label NER metrics file available: {summary['weak_label_ner_metrics_available']}",
            "",
            "## C. Overall Structured-Output Completion",
            f"- Overall entity extraction success rate: {summary['overall_entity_extraction_success_rate']:.3f}",
            f"- Overall average entities per record: {summary['overall_average_entities_per_record']:.2f}",
            f"- Overall structured JSON completion rate: {summary['overall_structured_json_completion_rate']:.3f}",
            f"- Overall average field completion rate: {summary['overall_average_field_completion_rate']:.3f}",
            "",
            "Ground-truth OCR/entity labels were not available, so Model-2 was evaluated using technical extraction and structured-output metrics.",
            "",
            "The doctor-note branch now includes newly generated MTSamples inference outputs, so the doctor-note metrics reflect a dedicated text-modality run.",
            "",
            "Weak-label NER metrics are read from the dedicated doctor-note weak-label training output, not from the OCR evaluator.",
        ],
    )

    document_counts = Counter(
        "doctor_note" if record["data"].get("input_type") == "doctor_note" else "document"
        for record in records
    )
    save_bar_chart(
        THESIS_FIGURES_DIR / "model2_document_processing_results.png",
        "Model-2 Document Processing Results",
        list(document_counts.keys()) or ["document"],
        [float(document_counts[key]) for key in (list(document_counts.keys()) or ["document"])],
        "Record count",
    )

    doctor_note_labels = ["with note", "without note"]
    doctor_note_values = [len(doctor_note_records), max(total_records - len(doctor_note_records), 0)]
    save_bar_chart(
        THESIS_FIGURES_DIR / "model2_doctor_note_entity_results.png",
        "Model-2 Doctor Note Entity Results",
        doctor_note_labels,
        [float(value) for value in doctor_note_values],
        "Record count",
    )

    save_bar_chart(
        THESIS_FIGURES_DIR / "model2_document_and_doctor_note_results.png",
        "Model-2 Document and Doctor-Note Results",
        ["document_ocr_success", "doctor_note_success", "structured_completion"],
        [
            float(summary["ocr_success_rate"]),
            float(summary["doctor_note_extraction_success_rate"]),
            float(summary["structured_json_completion_rate"]),
        ],
        "Score",
    )

    return summary


def main() -> None:
    summary = evaluate_model2()
    print(summary["note"])
    print(f"Wrote {EVALUATION_OUTPUT_DIR / 'model2_evaluation_summary.json'}")


if __name__ == "__main__":
    main()
