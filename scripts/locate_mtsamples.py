from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model2.doctor_note_dataset_utils import (
    DEFAULT_SEARCH_ROOTS,
    discover_mtsamples_location,
    load_mtsamples_frame,
    summarize_missing_values,
)


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "doctor_note_training"


def _value_counts(frame, column: str, limit: int = 20) -> Dict[str, int]:
    if column not in frame.columns:
        return {}
    series = frame[column].fillna("").astype(str).str.strip()
    series = series[series != ""]
    return {key: int(value) for key, value in series.value_counts().head(limit).items()}


def locate_dataset() -> Dict[str, Any]:
    location = discover_mtsamples_location(DEFAULT_SEARCH_ROOTS)
    if location is None:
        raise FileNotFoundError("Could not locate mtsamples.csv or archive.zip containing mtsamples.csv under the workspace roots.")

    frame = load_mtsamples_frame(location)
    transcription_count = int(frame["transcription"].fillna("").astype(str).str.strip().ne("").sum()) if "transcription" in frame.columns else 0

    audit = {
        "source_path": str(location.source_path),
        "source_type": location.source_type,
        "archive_member": location.archive_member,
        "file_size_bytes": location.size_bytes,
        "csv_shape": [int(frame.shape[0]), int(frame.shape[1])],
        "column_names": list(frame.columns),
        "non_empty_transcription_count": transcription_count,
        "medical_specialty_value_counts": _value_counts(frame, "medical_specialty"),
        "missing_values": summarize_missing_values(frame),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "mtsamples_location_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# MTSamples Location Audit",
        "",
        f"- Exact path: {audit['source_path']}",
        f"- Source type: {audit['source_type']}",
        f"- Archive member: {audit['archive_member'] or 'n/a'}",
        f"- File size (bytes): {audit['file_size_bytes']}",
        f"- CSV shape: {audit['csv_shape'][0]} rows x {audit['csv_shape'][1]} columns",
        f"- Column names: {', '.join(audit['column_names'])}",
        f"- Non-empty transcription count: {audit['non_empty_transcription_count']}",
        "",
        "## Medical Specialty Value Counts",
    ]

    if audit["medical_specialty_value_counts"]:
        for key, value in audit["medical_specialty_value_counts"].items():
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- medical_specialty column not found or empty")

    lines.extend([
        "",
        "## Missing Values",
    ])

    for key, value in audit["missing_values"].items():
        lines.append(f"- {key}: {value}")

    (OUTPUT_DIR / "mtsamples_location_audit.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate the MTSamples dataset in the workspace.")
    parser.parse_args()
    audit = locate_dataset()
    print(f"Exact path: {audit['source_path']}")
    print(f"File size: {audit['file_size_bytes']} bytes")
    print(f"CSV shape: {audit['csv_shape'][0]} x {audit['csv_shape'][1]}")
    print(f"Columns: {', '.join(audit['column_names'])}")
    print(f"Non-empty transcription count: {audit['non_empty_transcription_count']}")


if __name__ == "__main__":
    main()
