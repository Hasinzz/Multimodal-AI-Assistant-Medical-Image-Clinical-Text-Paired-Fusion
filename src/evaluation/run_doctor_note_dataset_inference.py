from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.common import EVALUATION_OUTPUT_DIR, THESIS_FIGURES_DIR, average, ensure_output_dirs, safe_ratio, save_bar_chart, save_csv, save_json, write_markdown  # noqa: E402
from src.model2.doctor_note_pipeline import run_doctor_note_pipeline  # noqa: E402


INPUT_JSONL = PROJECT_ROOT / "data" / "text" / "doctor_notes" / "mtsamples" / "processed" / "mtsamples_doctor_notes.jsonl"
OUTPUT_JSONL = PROJECT_ROOT / "outputs" / "doctor_note_training" / "doctor_note_inference_outputs.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "doctor_note_training"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def run_inference(input_jsonl: Path = INPUT_JSONL, output_jsonl: Path = OUTPUT_JSONL, limit: int | None = None) -> Dict[str, Any]:
    ensure_output_dirs()
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Prepared MTSamples JSONL not found: {input_jsonl}")

    source_records = _read_jsonl(input_jsonl)
    if limit is not None:
        source_records = source_records[:limit]

    output_records: List[Dict[str, Any]] = []
    success_count = 0
    specialty_available_count = 0
    structured_complete_count = 0
    entity_counts: List[int] = []
    summary_lengths: List[int] = []
    failed_records: List[Dict[str, Any]] = []

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8", newline="") as handle:
        for record in source_records:
            note_id = str(record.get("note_id", "unknown_note"))
            try:
                pipeline_output = run_doctor_note_pipeline(
                    doctor_note=record.get("clean_text") or record.get("raw_text") or "",
                    case_id=note_id,
                    mode="auto",
                )
                success_count += 1
                predicted_specialty = str(pipeline_output.get("predicted_specialty", "")).strip()
                if predicted_specialty and predicted_specialty.lower() != "unknown":
                    specialty_available_count += 1
                entities = pipeline_output.get("entities", {}) or {}
                entity_counts.append(sum(len(values or []) for values in entities.values()))
                summary_lengths.append(len(str(pipeline_output.get("patient_summary_text", ""))))
                structured_complete_count += int(bool(pipeline_output.get("input_type") == "doctor_note" and pipeline_output.get("doctor_note_available")))

                output_record = {
                    "note_id": note_id,
                    "source": record.get("source", "MTSamples"),
                    "input_type": record.get("input_type", "doctor_note_text"),
                    "medical_specialty": record.get("medical_specialty", ""),
                    "description": record.get("description", ""),
                    "sample_name": record.get("sample_name", ""),
                    "keywords": record.get("keywords", ""),
                    "source_text": record.get("raw_text", ""),
                    "doctor_note_output": pipeline_output,
                    "status": "success",
                }
                handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                output_records.append(output_record)
            except Exception as error:  # pragma: no cover - operational safeguard
                failed_record = {
                    "note_id": note_id,
                    "source": record.get("source", "MTSamples"),
                    "status": "failed",
                    "error": str(error),
                }
                handle.write(json.dumps(failed_record, ensure_ascii=False) + "\n")
                failed_records.append(failed_record)

    summary = {
        "total_notes_run": len(source_records),
        "success_rate": safe_ratio(success_count, len(source_records)),
        "average_extracted_entities": average([float(value) for value in entity_counts]),
        "predicted_specialty_availability_rate": safe_ratio(specialty_available_count, len(output_records) or 1),
        "structured_json_completion_rate": safe_ratio(structured_complete_count, len(output_records) or 1),
        "average_summary_length": average([float(value) for value in summary_lengths]),
        "failed_records": len(failed_records),
        "output_jsonl": str(output_jsonl),
        "note": "The inference run evaluates the doctor-note text branch on MTSamples-style clinical notes and produces structured technical outputs.",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "doctor_note_inference_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_markdown(
        OUTPUT_DIR / "doctor_note_inference_report.md",
        [
            "# Doctor-Note Inference Report",
            "",
            f"- Total notes run: {summary['total_notes_run']}",
            f"- Success rate: {summary['success_rate']:.3f}",
            f"- Average extracted entities: {summary['average_extracted_entities']:.2f}",
            f"- Predicted specialty availability rate: {summary['predicted_specialty_availability_rate']:.3f}",
            f"- Structured JSON completion rate: {summary['structured_json_completion_rate']:.3f}",
            f"- Average summary length: {summary['average_summary_length']:.1f}",
            "",
            "This run evaluates the doctor-note text branch on MTSamples-style clinical notes and produces structured technical outputs.",
        ],
    )

    save_csv(
        OUTPUT_DIR / "doctor_note_inference_summary.csv",
        [
            {
                "note_id": item["note_id"],
                "status": item.get("status", "success"),
                "medical_specialty": item.get("medical_specialty", ""),
                "predicted_specialty": item.get("doctor_note_output", {}).get("predicted_specialty", "") if item.get("status") == "success" else "",
                "summary_length": len(str(item.get("doctor_note_output", {}).get("patient_summary_text", ""))) if item.get("status") == "success" else 0,
            }
            for item in output_records + failed_records
        ],
        ["note_id", "status", "medical_specialty", "predicted_specialty", "summary_length"],
    )

    save_bar_chart(
        THESIS_FIGURES_DIR / "doctor_note_inference_results.png",
        "Doctor-Note Inference Results",
        ["success", "specialty", "structured"],
        [summary["success_rate"], summary["predicted_specialty_availability_rate"], summary["structured_json_completion_rate"]],
        "Rate",
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run doctor-note inference across MTSamples notes.")
    parser.add_argument("--input-jsonl", type=Path, default=INPUT_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=OUTPUT_JSONL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    summary = run_inference(args.input_jsonl, args.output_jsonl, args.limit)
    print(summary["note"])
    print(f"Wrote {summary['output_jsonl']}")


if __name__ == "__main__":
    main()
