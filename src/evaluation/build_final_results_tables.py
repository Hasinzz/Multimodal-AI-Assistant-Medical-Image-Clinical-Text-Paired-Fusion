from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.common import average, ensure_output_dirs, safe_ratio, save_json, write_markdown  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_revision"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_train_row(history_path: Path, epoch: int) -> Optional[Dict[str, Any]]:
    rows = _read_csv(history_path)
    for row in rows:
        if int(float(row.get("epoch", 0))) == epoch:
            return row
    return None


def _format_number(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _model1_tables() -> Dict[str, List[Dict[str, Any]]]:
    brain_final = _read_json(PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_final_v2" / "brain_metrics.json")
    brain_retrain = _read_json(PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_retrain_v3" / "brain_metrics.json")
    xray_large = _read_json(PROJECT_ROOT / "outputs" / "training" / "xray_gpu_large_v2" / "xray_metrics.json")
    xray_retrain = _read_json(PROJECT_ROOT / "outputs" / "training" / "xray_gpu_retrain_v3" / "xray_metrics.json")

    return {
        "model1_rows": [
            {
                "component": "Model-1A Brain MRI final_v2",
                "best_validation_metric": brain_final.get("best_val_accuracy"),
                "train_metric": _find_train_row(PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_final_v2" / "brain_training_history.csv", int(brain_final.get("best_epoch", 0))).get("train_accuracy") if brain_final.get("best_epoch") else None,
                "validation_metric": brain_final.get("best_val_accuracy"),
                "test_metric": brain_final.get("metrics", {}).get("accuracy"),
                "note": "Single-label MRI classification; trained and evaluated on the Brain MRI split.",
            },
            {
                "component": "Model-1A Brain MRI retrain_v3",
                "best_validation_metric": brain_retrain.get("best_val_accuracy"),
                "train_metric": _find_train_row(PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_retrain_v3" / "brain_training_history.csv", int(brain_retrain.get("best_epoch", 0))).get("train_accuracy") if brain_retrain.get("best_epoch") else None,
                "validation_metric": brain_retrain.get("best_val_accuracy"),
                "test_metric": brain_retrain.get("metrics", {}).get("accuracy"),
                "note": "Retrain version; same task, slightly different checkpoint and history.",
            },
            {
                "component": "Model-1B Chest X-ray large_v2",
                "best_validation_metric": xray_large.get("best_metrics", {}).get("macro_auroc"),
                "train_metric": _find_train_row(PROJECT_ROOT / "outputs" / "training" / "xray_gpu_large_v2" / "xray_training_history.csv", int(xray_large.get("best_epoch", 0))).get("train_loss") if xray_large.get("best_epoch") else None,
                "validation_metric": xray_large.get("best_metrics", {}).get("macro_auroc"),
                "test_metric": xray_large.get("best_metrics", {}).get("micro_auroc"),
                "note": "Multi-label X-ray classification; validation metric is macro AUROC.",
            },
            {
                "component": "Model-1B Chest X-ray retrain_v3",
                "best_validation_metric": xray_retrain.get("best_metrics", {}).get("macro_auroc"),
                "train_metric": _find_train_row(PROJECT_ROOT / "outputs" / "training" / "xray_gpu_retrain_v3" / "xray_training_history.csv", int(xray_retrain.get("best_epoch", 0))).get("train_loss") if xray_retrain.get("best_epoch") else None,
                "validation_metric": xray_retrain.get("best_metrics", {}).get("macro_auroc"),
                "test_metric": xray_retrain.get("best_metrics", {}).get("micro_auroc"),
                "note": "Retrain version; same task, slightly different checkpoint and history.",
            },
        ]
    }


def _model2a_table() -> Dict[str, List[Dict[str, Any]]]:
    model2_summary = _read_json(PROJECT_ROOT / "outputs" / "evaluation" / "model2_evaluation_summary.json")
    return {
        "model2a_rows": [
            {
                "component": "Model-2A document OCR/extraction",
                "cases": model2_summary.get("document_records"),
                "ocr_success_rate": model2_summary.get("ocr_success_rate"),
                "entity_extraction_success_rate": model2_summary.get("document_entity_extraction_success_rate", model2_summary.get("entity_extraction_success_rate")),
                "average_entities": model2_summary.get("document_average_entities_per_record", model2_summary.get("average_entities_per_record")),
                "json_completion_rate": model2_summary.get("document_structured_json_completion_rate", model2_summary.get("structured_json_completion_rate")),
                "field_completion_rate": model2_summary.get("document_average_field_completion_rate", model2_summary.get("average_field_completion_rate")),
                "note": "OCR and extraction are pipeline-based technical metrics from archived document outputs.",
            },
        ]
    }


def _model2b_table() -> Dict[str, List[Dict[str, Any]]]:
    classifier = _read_json(PROJECT_ROOT / "outputs" / "doctor_note_training" / "classifier_metrics.json")
    weak_ner = _read_json(PROJECT_ROOT / "outputs" / "doctor_note_training" / "weak_ner_metrics.json")
    inference = _read_json(PROJECT_ROOT / "outputs" / "doctor_note_training" / "doctor_note_inference_summary.json")
    return {
        "model2b_rows": [
            {
                "component": "Doctor-note specialty classifier",
                "metric_1": classifier.get("test_metrics", {}).get("accuracy"),
                "metric_2": classifier.get("test_metrics", {}).get("macro_f1"),
                "metric_3": classifier.get("kept_class_count"),
                "metric_4": classifier.get("usable_rows"),
                "note": classifier.get("note"),
            },
            {
                "component": "Weak-label NER",
                "metric_1": weak_ner.get("mode"),
                "metric_2": weak_ner.get("entity_f1"),
                "metric_3": weak_ner.get("token_accuracy"),
                "metric_4": weak_ner.get("fallback_reason") or "trained",
                "note": weak_ner.get("note"),
            },
            {
                "component": "Doctor-note inference run",
                "metric_1": inference.get("success_rate"),
                "metric_2": inference.get("average_extracted_entities"),
                "metric_3": inference.get("predicted_specialty_availability_rate"),
                "metric_4": inference.get("structured_json_completion_rate"),
                "note": inference.get("note"),
            },
        ]
    }


def _model3_table() -> Dict[str, List[Dict[str, Any]]]:
    model3 = _read_json(PROJECT_ROOT / "outputs" / "evaluation" / "model3_evaluation_summary.json")
    return {
        "model3_rows": [
            {
                "component": "Model-3 fusion/RAG",
                "fusion_output_success_rate": model3.get("fusion_output_success_rate_overall"),
                "summary_generation_rate": model3.get("patient_summary_generation_rate_overall"),
                "feedback_generation_rate": model3.get("doctor_feedback_generation_rate_overall"),
                "retrieved_evidence_availability_rate": model3.get("retrieved_evidence_availability_rate_overall"),
                "average_retrieved_evidence_count": model3.get("average_retrieved_evidence_count_overall"),
                "average_summary_length": model3.get("average_summary_length_overall"),
                "average_feedback_length": model3.get("average_doctor_feedback_length_overall"),
                "note": model3.get("note"),
            }
        ]
    }


def _cross_modal_table() -> Dict[str, List[Dict[str, Any]]]:
    cross = _read_json(PROJECT_ROOT / "outputs" / "evaluation" / "cross_modal_validation_summary.json")
    return {
        "cross_modal_rows": [
            {
                "scenario": row["scenario"],
                "case_count": row["case_count"],
                "synthetic_unpaired_validation": row["synthetic_unpaired_validation"],
                "fusion_output_success_rate": row["fusion_output_success_rate"],
                "patient_summary_generation_rate": row["patient_summary_generation_rate"],
                "doctor_feedback_generation_rate": row["doctor_feedback_generation_rate"],
                "retrieved_evidence_availability_rate": row["retrieved_evidence_availability_rate"],
                "average_retrieved_evidence_count": row["average_retrieved_evidence_count"],
                "average_extracted_finding_count": row["average_extracted_finding_count"],
                "average_technical_completion_score": row["average_technical_completion_score"],
            }
            for row in cross.get("summary_rows", [])
        ]
    }


def _validation_100_case_table() -> Dict[str, List[Dict[str, Any]]]:
    final_run = _read_csv(PROJECT_ROOT / "outputs" / "final_run_100_tuned_v2" / "main_run_summary.csv")
    total_cases = len(final_run)
    completed_cases = sum(1 for row in final_run if str(row.get("status", "")).strip().lower() == "success")
    failed_cases = total_cases - completed_cases
    success_rate = safe_ratio(completed_cases, total_cases) * 100.0
    return {
        "validation_100_rows": [
            {
                "component": "100-case technical validation",
                "total_cases": total_cases,
                "completed_cases": completed_cases,
                "failed_cases": failed_cases,
                "technical_execution_success": success_rate,
                "note": "Inference-only validation; not a clinical accuracy claim.",
            }
        ]
    }


def _v4_table() -> Dict[str, List[Dict[str, Any]]]:
    stable_output = _read_json(PROJECT_ROOT / "outputs" / "v4_advanced_improvement" / "comparison" / "stable_output.json")
    v4_output = _read_json(PROJECT_ROOT / "outputs" / "v4_advanced_improvement" / "comparison" / "v4_output.json")
    stable_model2 = stable_output.get("model2_output", stable_output)
    v4_model2 = v4_output.get("model2_output", v4_output)
    stable_fusion = stable_output.get("fusion_output", {})
    v4_fusion = v4_output.get("fusion_output", {})
    return {
        "v4_rows": [
            {
                "component": "V4 experimental enhancement layer",
                "stable_ocr_length": len(str(stable_model2.get("raw_text", ""))),
                "stable_entity_count": len(stable_model2.get("entities", [])) if isinstance(stable_model2.get("entities", []), list) else 0,
                "v4_ocr_length": len(str(v4_model2.get("raw_text", ""))),
                "v4_entity_count": len(v4_model2.get("entities", [])) if isinstance(v4_model2.get("entities", []), list) else 0,
                "stable_retrieved_evidence_count": len(stable_fusion.get("retrieved_evidence", [])),
                "v4_retrieved_evidence_count": len(v4_fusion.get("retrieved_evidence", [])),
                "note": "V4 is an optional experimental enhancement. The cross-attention checkpoint folder is empty, so no trained cross-attention result is safely claimable.",
            }
        ]
    }


def _safe_table_row_count(rows: List[Dict[str, Any]]) -> int:
    return len(rows)


def _latex_escape(value: Any) -> str:
    if value is None:
        return "n/a"
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _render_table(rows: List[Dict[str, Any]], columns: List[str], caption: str, label: str) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "|".join(["l"] * len(columns)) + "}",
        "\\hline",
        " & ".join(_latex_escape(column) for column in columns) + " \\\\ \\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row.get(column, "")) for column in columns) + " \\\\ \\hline")
    lines.extend(["\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def build_final_results_tables() -> Dict[str, Any]:
    ensure_output_dirs()

    model1 = _model1_tables()["model1_rows"]
    model2a = _model2a_table()["model2a_rows"]
    model2b = _model2b_table()["model2b_rows"]
    model3 = _model3_table()["model3_rows"]
    cross = _cross_modal_table()["cross_modal_rows"]
    validation_100 = _validation_100_case_table()["validation_100_rows"]
    v4 = _v4_table()["v4_rows"]

    summary = {
        "model1_image_results": model1,
        "model2a_document_results": model2a,
        "model2b_doctor_note_results": model2b,
        "model3_fusion_results": model3,
        "cross_modal_validation_results": cross,
        "technical_100_case_validation": validation_100,
        "v4_experimental_result": v4,
        "note": "All results are taken from existing summaries and archived outputs. No new metrics were invented.",
    }

    save_json(OUTPUT_DIR / "final_model_results_summary.json", summary)

    markdown_lines = [
        "# Final Model Results Summary",
        "",
        "## Model-1 Image Results",
        *(
            f"- {row['component']}: validation={_format_number(row['validation_metric'], 4)}, "
            f"train={_format_number(_safe_float(row['train_metric']), 4)}, test={_format_number(_safe_float(row['test_metric']), 4)}."
            for row in model1
        ),
        "",
        "## Model-2A Document OCR/Extraction Results",
        *(
            f"- {row['component']}: cases={row['cases']}, OCR={_format_number(_safe_float(row['ocr_success_rate']), 3)}, "
            f"entity success={_format_number(_safe_float(row['entity_extraction_success_rate']), 3)}, avg entities={_format_number(_safe_float(row['average_entities']), 2)}, "
            f"JSON completion={_format_number(_safe_float(row['json_completion_rate']), 3)}, field completion={_format_number(_safe_float(row['field_completion_rate']), 3)}."
            for row in model2a
        ),
        "",
        "## Model-2B Doctor-Note Text Results",
        *(
            f"- {row['component']}: metric1={row['metric_1']}, metric2={row['metric_2']}, metric3={row['metric_3']}, metric4={row['metric_4']}."
            for row in model2b
        ),
        "",
        "## Model-3 Fusion/RAG Results",
        *(
            f"- {row['component']}: fusion={_format_number(_safe_float(row['fusion_output_success_rate']), 3)}, summary={_format_number(_safe_float(row['summary_generation_rate']), 3)}, feedback={_format_number(_safe_float(row['feedback_generation_rate']), 3)}, evidence={_format_number(_safe_float(row['retrieved_evidence_availability_rate']), 3)}, avg evidence={_format_number(_safe_float(row['average_retrieved_evidence_count']), 2)}."
            for row in model3
        ),
        "",
        "## Cross-Modal Validation Results",
        *(
            f"- {row['scenario']}: cases={row['case_count']}, synthetic={row['synthetic_unpaired_validation']}, completion={_format_number(_safe_float(row['average_technical_completion_score']), 2)}."
            for row in cross
        ),
        "",
        "## 100-Case Technical Validation",
        *(
            f"- {row['component']}: completed={row['completed_cases']}/{row['total_cases']}, technical success={row['technical_execution_success']}%."
            for row in validation_100
        ),
        "",
        "## V4 Experimental Result",
        *(
            f"- {row['component']}: stable OCR length={row['stable_ocr_length']}, stable entities={row['stable_entity_count']}, V4 OCR length={row['v4_ocr_length']}, V4 entities={row['v4_entity_count']}."
            for row in v4
        ),
        "",
        "All results are technical or pipeline-completeness results. None of the tables should be rewritten as clinical validation evidence.",
    ]
    write_markdown(OUTPUT_DIR / "final_model_results_summary.md", markdown_lines)

    tex_parts = [
        _render_table(
            [
                {
                    "Component": row["component"],
                    "Best validation": _format_number(_safe_float(row["validation_metric"]), 4),
                    "Train": _format_number(_safe_float(row["train_metric"]), 4),
                    "Test": _format_number(_safe_float(row["test_metric"]), 4),
                }
                for row in model1
            ],
            ["Component", "Best validation", "Train", "Test"],
            "Model-1 image results",
            "tab:model1-image-results",
        ),
        _render_table(
            [
                {
                    "Component": row["component"],
                    "Cases": row["cases"],
                    "OCR": _format_number(_safe_float(row["ocr_success_rate"]), 3),
                    "Entity success": _format_number(_safe_float(row["entity_extraction_success_rate"]), 3),
                    "Avg entities": _format_number(_safe_float(row["average_entities"]), 2),
                    "JSON completion": _format_number(_safe_float(row["json_completion_rate"]), 3),
                    "Field completion": _format_number(_safe_float(row["field_completion_rate"]), 3),
                }
                for row in model2a
            ],
            ["Component", "Cases", "OCR", "Entity success", "Avg entities", "JSON completion", "Field completion"],
            "Model-2A document OCR/extraction results",
            "tab:model2a-document-results",
        ),
        _render_table(
            [
                {
                    "Component": row["component"],
                    "Metric 1": row["metric_1"],
                    "Metric 2": _format_number(_safe_float(row["metric_2"]), 3) if isinstance(row["metric_2"], (int, float)) or _safe_float(row["metric_2"]) is not None else row["metric_2"],
                    "Metric 3": row["metric_3"],
                    "Metric 4": row["metric_4"],
                }
                for row in model2b
            ],
            ["Component", "Metric 1", "Metric 2", "Metric 3", "Metric 4"],
            "Model-2B doctor-note text results",
            "tab:model2b-doctor-note-results",
        ),
        _render_table(
            [
                {
                    "Scenario": row["scenario"],
                    "Cases": row["case_count"],
                    "Synthetic": row["synthetic_unpaired_validation"],
                    "Fusion": _format_number(_safe_float(row["fusion_output_success_rate"]), 3),
                    "Summary": _format_number(_safe_float(row["patient_summary_generation_rate"]), 3),
                    "Feedback": _format_number(_safe_float(row["doctor_feedback_generation_rate"]), 3),
                    "Evidence": _format_number(_safe_float(row["retrieved_evidence_availability_rate"]), 3),
                    "Avg evidence": _format_number(_safe_float(row["average_retrieved_evidence_count"]), 2),
                    "Avg findings": _format_number(_safe_float(row["average_extracted_finding_count"]), 2),
                    "Score": _format_number(_safe_float(row["average_technical_completion_score"]), 2),
                }
                for row in cross
            ],
            ["Scenario", "Cases", "Synthetic", "Fusion", "Summary", "Feedback", "Evidence", "Avg evidence", "Avg findings", "Score"],
            "Cross-modal validation results",
            "tab:cross-modal-validation-results",
        ),
        _render_table(
            [
                {
                    "Component": row["component"],
                    "Total": row["total_cases"],
                    "Completed": row["completed_cases"],
                    "Failed": row["failed_cases"],
                    "Technical success": row["technical_execution_success"],
                }
                for row in validation_100
            ],
            ["Component", "Total", "Completed", "Failed", "Technical success"],
            "100-case technical validation",
            "tab:100-case-technical-validation",
        ),
        _render_table(
            [
                {
                    "Component": row["component"],
                    "Stable OCR": row["stable_ocr_length"],
                    "Stable entities": row["stable_entity_count"],
                    "V4 OCR": row["v4_ocr_length"],
                    "V4 entities": row["v4_entity_count"],
                    "Stable evidence": row["stable_retrieved_evidence_count"],
                    "V4 evidence": row["v4_retrieved_evidence_count"],
                }
                for row in v4
            ],
            ["Component", "Stable OCR", "Stable entities", "V4 OCR", "V4 entities", "Stable evidence", "V4 evidence"],
            "V4 experimental result",
            "tab:v4-experimental-result",
        ),
    ]
    (OUTPUT_DIR / "final_model_results_tables.tex").write_text("\n".join(tex_parts).rstrip() + "\n", encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final thesis result tables from existing outputs.")
    parser.parse_args()
    summary = build_final_results_tables()
    print(summary["note"])
    print(f"Wrote {OUTPUT_DIR / 'final_model_results_summary.json'}")


if __name__ == "__main__":
    main()
