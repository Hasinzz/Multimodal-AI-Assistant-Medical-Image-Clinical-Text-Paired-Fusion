from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.common import (  # noqa: E402
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
from src.model3.pipeline import run_fusion_pipeline  # noqa: E402


IMAGE_ONLY_SAMPLE_SIZE = 20
DOCUMENT_ONLY_SAMPLE_SIZE = 20
DOCTOR_NOTE_ONLY_SAMPLE_SIZE = 20
COMBO_SAMPLE_SIZE = 10


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _discover_case_outputs(case_type_prefix: str, required_files: Sequence[str]) -> List[Dict[str, Any]]:
    case_dirs = []
    for directory in sorted((PROJECT_ROOT / "outputs" / "final_run_100_tuned_v2").iterdir()):
        if directory.is_dir() and directory.name.startswith(case_type_prefix):
            case_dirs.append(directory)

    records: List[Dict[str, Any]] = []
    for case_dir in case_dirs:
        payload: Dict[str, Any] = {"source_case_id": case_dir.name}
        loaded = True
        for file_name in required_files:
            file_path = case_dir / file_name
            if not file_path.exists():
                loaded = False
                break
            payload[file_name] = _read_json(file_path)
        if loaded:
            records.append(payload)
    return records


def _load_image_pool() -> List[Dict[str, Any]]:
    pool = []
    pool.extend(_discover_case_outputs("main_brain_", ["model1_output.json"]))
    pool.extend(_discover_case_outputs("main_xray_", ["model1_output.json"]))
    return pool


def _load_document_pool() -> List[Dict[str, Any]]:
    pool = []
    pool.extend(_discover_case_outputs("main_prescription_", ["model2_output.json"]))
    pool.extend(_discover_case_outputs("main_lab_", ["model2_output.json"]))
    return pool


def _load_doctor_note_pool() -> List[Dict[str, Any]]:
    jsonl_path = PROJECT_ROOT / "outputs" / "doctor_note_training" / "doctor_note_inference_outputs.jsonl"
    pool: List[Dict[str, Any]] = []
    for record in _read_jsonl(jsonl_path):
        nested = record.get("doctor_note_output")
        if isinstance(nested, dict):
            pool.append(
                {
                    "source_case_id": record.get("note_id", "doctor_note"),
                    "doctor_note_output": nested,
                }
            )
    return pool


def _count_image_findings(model1_output: Optional[Dict[str, Any]]) -> int:
    if not model1_output:
        return 0
    positive_labels = model1_output.get("xray_positive_labels")
    if isinstance(positive_labels, list) and positive_labels:
        return len(positive_labels)
    top_predictions = model1_output.get("top_predictions")
    if isinstance(top_predictions, list) and top_predictions:
        return min(len(top_predictions), 5)
    return int(bool(model1_output.get("patient_summary_text")))


def _count_document_findings(model2_output: Optional[Dict[str, Any]]) -> int:
    if not model2_output:
        return 0
    entities = model2_output.get("entities")
    if isinstance(entities, list):
        return len(entities)
    if isinstance(entities, dict):
        return sum(len(values) for values in entities.values() if isinstance(values, list))
    structured_json = model2_output.get("structured_json")
    if isinstance(structured_json, dict):
        return sum(len(values) for values in structured_json.values() if isinstance(values, list))
    return int(bool(model2_output.get("patient_summary")))


def _count_doctor_note_findings(doctor_note_output: Optional[Dict[str, Any]]) -> int:
    if not doctor_note_output:
        return 0
    entities = doctor_note_output.get("entities")
    if isinstance(entities, dict):
        return sum(len(values) for values in entities.values() if isinstance(values, list))
    return int(bool(doctor_note_output.get("patient_summary_text")))


def _build_case(case_id: str, model1_output: Optional[Dict[str, Any]], model2_output: Optional[Dict[str, Any]], doctor_note_output: Optional[Dict[str, Any]], synthetic_pairing: bool) -> Dict[str, Any]:
    fusion_output = run_fusion_pipeline(
        case_id=case_id,
        model1_output=model1_output,
        model2_output=model2_output,
        doctor_note_output=doctor_note_output,
    )

    retrieved_evidence = fusion_output.get("retrieved_evidence", []) or []
    available_modalities = fusion_output.get("available_modalities", []) or []
    missing_information_notes = fusion_output.get("missing_information_notes", []) or []
    final_summary = str(fusion_output.get("final_summary", ""))
    doctor_feedback = str(fusion_output.get("doctor_feedback", fusion_output.get("doctor_oriented_feedback", "")))

    extracted_finding_count = _count_image_findings(model1_output) + _count_document_findings(model2_output) + _count_doctor_note_findings(doctor_note_output)
    technical_completion_score = (
        len(available_modalities)
        + len(retrieved_evidence)
        + int(bool(fusion_output.get("patient_summary")))
        + int(bool(doctor_feedback))
        + int(bool(final_summary))
        - len(missing_information_notes)
    )

    return {
        "case_id": case_id,
        "synthetic_unpaired_validation": int(synthetic_pairing),
        "source_case_ids": "; ".join(
            [item for item in [
                model1_output.get("case_id") if isinstance(model1_output, dict) else None,
                model2_output.get("case_id") if isinstance(model2_output, dict) else None,
                doctor_note_output.get("case_id") if isinstance(doctor_note_output, dict) else None,
            ] if item]
        ),
        "available_modalities": ", ".join(available_modalities),
        "available_modality_count": len(available_modalities),
        "missing_modality_warnings": len(missing_information_notes),
        "fusion_output_success": int(bool(final_summary and doctor_feedback)),
        "patient_summary_generation": int(bool(fusion_output.get("patient_summary"))),
        "doctor_feedback_generation": int(bool(doctor_feedback)),
        "retrieved_evidence_availability": int(bool(retrieved_evidence)),
        "average_retrieved_evidence_count": float(len(retrieved_evidence)),
        "extracted_finding_count": int(extracted_finding_count),
        "summary_length": len(final_summary),
        "feedback_length": len(doctor_feedback),
        "json_completion": int(bool(fusion_output)),
        "technical_completion_score": int(technical_completion_score),
    }


def _sample_cycle(items: Sequence[Any], sample_size: int) -> List[Any]:
    if not items or sample_size <= 0:
        return []
    return list(islice(cycle(items), sample_size))


def _scenario_rows() -> List[Dict[str, Any]]:
    image_pool = _load_image_pool()
    document_pool = _load_document_pool()
    doctor_note_pool = _load_doctor_note_pool()

    if not image_pool:
        raise FileNotFoundError("No archived image outputs were found under outputs/final_run_100_tuned_v2.")
    if not document_pool:
        raise FileNotFoundError("No archived document outputs were found under outputs/final_run_100_tuned_v2.")
    if not doctor_note_pool:
        raise FileNotFoundError("No doctor-note inference outputs were found under outputs/doctor_note_training.")

    scenarios = {
        "image_only": [
            _build_case(
                f"cross_modal_image_only_{index + 1:03d}",
                source.get("model1_output.json"),
                None,
                None,
                synthetic_pairing=False,
            )
            for index, source in enumerate(_sample_cycle(image_pool, IMAGE_ONLY_SAMPLE_SIZE))
        ],
        "document_only": [
            _build_case(
                f"cross_modal_document_only_{index + 1:03d}",
                None,
                source.get("model2_output.json"),
                None,
                synthetic_pairing=False,
            )
            for index, source in enumerate(_sample_cycle(document_pool, DOCUMENT_ONLY_SAMPLE_SIZE))
        ],
        "doctor_note_only": [
            _build_case(
                f"cross_modal_doctor_note_only_{index + 1:03d}",
                None,
                None,
                source.get("doctor_note_output"),
                synthetic_pairing=False,
            )
            for index, source in enumerate(_sample_cycle(doctor_note_pool, DOCTOR_NOTE_ONLY_SAMPLE_SIZE))
        ],
    }

    combo_sources = list(zip(
        _sample_cycle(image_pool, COMBO_SAMPLE_SIZE),
        _sample_cycle(document_pool, COMBO_SAMPLE_SIZE),
        _sample_cycle(doctor_note_pool, COMBO_SAMPLE_SIZE),
    ))

    scenarios["image_document"] = [
        _build_case(
            f"cross_modal_image_document_{index + 1:03d}",
            image_source.get("model1_output.json"),
            document_source.get("model2_output.json"),
            None,
            synthetic_pairing=True,
        )
        for index, (image_source, document_source, _) in enumerate(combo_sources)
    ]

    scenarios["image_doctor_note"] = [
        _build_case(
            f"cross_modal_image_doctor_note_{index + 1:03d}",
            image_source.get("model1_output.json"),
            None,
            doctor_source.get("doctor_note_output"),
            synthetic_pairing=True,
        )
        for index, (image_source, _, doctor_source) in enumerate(combo_sources)
    ]

    scenarios["document_doctor_note"] = [
        _build_case(
            f"cross_modal_document_doctor_note_{index + 1:03d}",
            None,
            document_source.get("model2_output.json"),
            doctor_source.get("doctor_note_output"),
            synthetic_pairing=True,
        )
        for index, (_, document_source, doctor_source) in enumerate(combo_sources)
    ]

    scenarios["image_document_doctor_note"] = [
        _build_case(
            f"cross_modal_image_document_doctor_note_{index + 1:03d}",
            image_source.get("model1_output.json"),
            document_source.get("model2_output.json"),
            doctor_source.get("doctor_note_output"),
            synthetic_pairing=True,
        )
        for index, (image_source, document_source, doctor_source) in enumerate(combo_sources)
    ]

    ordered_rows: List[Dict[str, Any]] = []
    for scenario_name in [
        "image_only",
        "document_only",
        "doctor_note_only",
        "image_document",
        "image_doctor_note",
        "document_doctor_note",
        "image_document_doctor_note",
    ]:
        ordered_rows.extend([{**row, "scenario": scenario_name} for row in scenarios[scenario_name]])

    return ordered_rows


def run_cross_modal_validation() -> Dict[str, Any]:
    ensure_output_dirs()
    rows = _scenario_rows()

    summary_rows: List[Dict[str, Any]] = []
    for scenario_name in [
        "image_only",
        "document_only",
        "doctor_note_only",
        "image_document",
        "image_doctor_note",
        "document_doctor_note",
        "image_document_doctor_note",
    ]:
        scenario_rows = [row for row in rows if row["scenario"] == scenario_name]
        summary_rows.append(
            {
                "scenario": scenario_name,
                "case_count": len(scenario_rows),
                "synthetic_unpaired_validation": int(any(row["synthetic_unpaired_validation"] for row in scenario_rows)),
                "average_available_modalities": average([float(row["available_modality_count"]) for row in scenario_rows]),
                "average_missing_modality_warnings": average([float(row["missing_modality_warnings"]) for row in scenario_rows]),
                "fusion_output_success_rate": safe_ratio(sum(row["fusion_output_success"] for row in scenario_rows), len(scenario_rows)),
                "patient_summary_generation_rate": safe_ratio(sum(row["patient_summary_generation"] for row in scenario_rows), len(scenario_rows)),
                "doctor_feedback_generation_rate": safe_ratio(sum(row["doctor_feedback_generation"] for row in scenario_rows), len(scenario_rows)),
                "retrieved_evidence_availability_rate": safe_ratio(sum(row["retrieved_evidence_availability"] for row in scenario_rows), len(scenario_rows)),
                "average_retrieved_evidence_count": average([float(row["average_retrieved_evidence_count"]) for row in scenario_rows]),
                "average_extracted_finding_count": average([float(row["extracted_finding_count"]) for row in scenario_rows]),
                "average_summary_length": average([float(row["summary_length"]) for row in scenario_rows]),
                "average_feedback_length": average([float(row["feedback_length"]) for row in scenario_rows]),
                "json_completion_rate": safe_ratio(sum(row["json_completion"] for row in scenario_rows), len(scenario_rows)),
                "average_technical_completion_score": average([float(row["technical_completion_score"]) for row in scenario_rows]),
            }
        )

    summary = {
        "scenario_count": len(summary_rows),
        "synthetic_unpaired_validation_used": any(row["synthetic_unpaired_validation"] for row in summary_rows),
        "summary_rows": summary_rows,
        "note": (
            "This validates cross-modal pipeline behavior and output completeness. "
            "It does not prove clinical correctness. It does not prove true paired-patient multimodal learning unless paired data exists. "
            "Synthetic/unpaired modality pairing was used for the combination scenarios because the repository does not contain a real paired multimodal dataset."
        ),
    }

    save_json(EVALUATION_OUTPUT_DIR / "cross_modal_validation_summary.json", summary)
    save_csv(
        EVALUATION_OUTPUT_DIR / "cross_modal_validation_summary.csv",
        summary_rows,
        [
            "scenario",
            "case_count",
            "synthetic_unpaired_validation",
            "average_available_modalities",
            "average_missing_modality_warnings",
            "fusion_output_success_rate",
            "patient_summary_generation_rate",
            "doctor_feedback_generation_rate",
            "retrieved_evidence_availability_rate",
            "average_retrieved_evidence_count",
            "average_extracted_finding_count",
            "average_summary_length",
            "average_feedback_length",
            "json_completion_rate",
            "average_technical_completion_score",
        ],
    )

    write_markdown(
        EVALUATION_OUTPUT_DIR / "cross_modal_validation_report.md",
        [
            "# Cross-Modal Validation Report",
            "",
            f"Scenario count: {summary['scenario_count']}",
            "",
            "This validates cross-modal pipeline behavior and output completeness.",
            "This does not prove clinical correctness.",
            "This does not prove true paired-patient multimodal learning unless paired data exists.",
            "Synthetic/unpaired modality pairing was used for the combination scenarios because the repository does not contain a real paired multimodal dataset.",
            "",
            "## Scenario Results",
            *(
                f"- {row['scenario']}: cases={row['case_count']}, success={row['fusion_output_success_rate']:.3f}, "
                f"summary={row['patient_summary_generation_rate']:.3f}, feedback={row['doctor_feedback_generation_rate']:.3f}, "
                f"retrieval={row['retrieved_evidence_availability_rate']:.3f}, avg evidence={row['average_retrieved_evidence_count']:.2f}, "
                f"avg findings={row['average_extracted_finding_count']:.2f}, avg score={row['average_technical_completion_score']:.2f}"
                for row in summary_rows
            ),
            "",
            f"- Overall JSON completion rate: {safe_ratio(sum(row['json_completion_rate'] * row['case_count'] for row in summary_rows), sum(row['case_count'] for row in summary_rows)):.3f}",
        ],
    )

    save_bar_chart(
        THESIS_FIGURES_DIR / "cross_modal_validation_results.png",
        "Cross-Modal Validation Results",
        [row["scenario"] for row in summary_rows],
        [float(row["average_technical_completion_score"]) for row in summary_rows],
        "Technical completion score",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run honest cross-modal validation across archived outputs.")
    parser.parse_args()
    summary = run_cross_modal_validation()
    print(summary["note"])
    print(f"Wrote {EVALUATION_OUTPUT_DIR / 'cross_modal_validation_summary.json'}")


if __name__ == "__main__":
    main()
