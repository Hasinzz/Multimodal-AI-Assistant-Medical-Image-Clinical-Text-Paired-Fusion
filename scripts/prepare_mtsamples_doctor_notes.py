from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model2.doctor_note_dataset_utils import (
    DEFAULT_SEARCH_ROOTS,
    clean_doctor_text,
    discover_mtsamples_location,
    load_mtsamples_frame,
    summarize_missing_values,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "doctor_note_training"
PROCESSED_DIR = PROJECT_ROOT / "data" / "text" / "doctor_notes" / "mtsamples" / "processed"
JSONL_PATH = PROCESSED_DIR / "mtsamples_doctor_notes.jsonl"


def _to_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def prepare_dataset() -> Dict[str, Any]:
    location = discover_mtsamples_location(DEFAULT_SEARCH_ROOTS)
    if location is None:
        raise FileNotFoundError("Could not locate mtsamples.csv or archive.zip containing mtsamples.csv under the workspace roots.")

    frame = load_mtsamples_frame(location)
    normalized_columns = {column: column.lower().strip() for column in frame.columns}
    frame = frame.rename(columns=normalized_columns).copy()

    if "transcription" not in frame.columns:
        raise KeyError("The MTSamples dataset does not contain a transcription column after normalization.")

    frame["transcription"] = frame["transcription"].apply(_to_text)
    usable_frame = frame[frame["transcription"].str.strip() != ""].copy()
    usable_frame["clean_text"] = usable_frame["transcription"].apply(clean_doctor_text)

    def _clean_optional(column: str) -> List[str]:
        if column not in usable_frame.columns:
            return [""] * len(usable_frame)
        return [_to_text(value) for value in usable_frame[column].tolist()]

    description_values = _clean_optional("description")
    specialty_values = _clean_optional("medical_specialty")
    sample_name_values = _clean_optional("sample_name")
    keywords_values = _clean_optional("keywords")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    with JSONL_PATH.open("w", encoding="utf-8", newline="") as handle:
        for index, (_, row) in enumerate(usable_frame.iterrows(), start=1):
            record = {
                "note_id": f"mtsamples_{index:06d}",
                "source": "MTSamples",
                "input_type": "doctor_note_text",
                "medical_specialty": specialty_values[index - 1],
                "description": description_values[index - 1],
                "sample_name": sample_name_values[index - 1],
                "keywords": keywords_values[index - 1],
                "raw_text": _to_text(row["transcription"]),
                "clean_text": _to_text(row["clean_text"]),
                "label_source": "dataset_metadata",
                "task_use": [
                    "doctor_note_classification",
                    "weak_entity_extraction",
                    "text_modality_fusion",
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    missing_values = summarize_missing_values(frame)
    summary = {
        "source_path": str(location.source_path),
        "source_type": location.source_type,
        "archive_member": location.archive_member,
        "raw_rows": int(frame.shape[0]),
        "usable_notes": int(len(records)),
        "dropped_rows": int(frame.shape[0] - len(records)),
        "output_jsonl": str(JSONL_PATH),
        "columns_present": list(frame.columns),
        "missing_values": missing_values,
        "medical_specialty_count": int(frame["medical_specialty"].fillna("").astype(str).str.strip().ne("").sum()) if "medical_specialty" in frame.columns else 0,
        "description_count": int(frame["description"].fillna("").astype(str).str.strip().ne("").sum()) if "description" in frame.columns else 0,
        "sample_name_count": int(frame["sample_name"].fillna("").astype(str).str.strip().ne("").sum()) if "sample_name" in frame.columns else 0,
        "keywords_count": int(frame["keywords"].fillna("").astype(str).str.strip().ne("").sum()) if "keywords" in frame.columns else 0,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "mtsamples_preparation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# MTSamples Preparation Report",
        "",
        f"- Source path: {summary['source_path']}",
        f"- Source type: {summary['source_type']}",
        f"- Raw rows: {summary['raw_rows']}",
        f"- Usable notes: {summary['usable_notes']}",
        f"- Dropped rows: {summary['dropped_rows']}",
        f"- Output JSONL: {summary['output_jsonl']}",
        "",
        "## Columns",
        f"- {', '.join(summary['columns_present'])}",
        "",
        "## Missing Values",
    ]

    for key, value in summary["missing_values"].items():
        report_lines.append(f"- {key}: {value}")

    report_lines.extend([
        "",
        "## Metadata Coverage",
        f"- medical_specialty present in {summary['medical_specialty_count']} row(s)",
        f"- description present in {summary['description_count']} row(s)",
        f"- sample_name present in {summary['sample_name_count']} row(s)",
        f"- keywords present in {summary['keywords_count']} row(s)",
        "",
        "Each JSONL record preserves dataset metadata and uses cleaned transcription text for downstream doctor-note training and evaluation.",
    ])

    (OUTPUT_DIR / "mtsamples_preparation_report.md").write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MTSamples doctor-note records.")
    parser.parse_args()
    summary = prepare_dataset()
    print(f"Prepared {summary['usable_notes']} notes from {summary['source_path']}")
    print(f"Wrote {summary['output_jsonl']}")


if __name__ == "__main__":
    main()
