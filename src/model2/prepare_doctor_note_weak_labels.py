from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model2.doctor_note_dataset_utils import (  # noqa: E402
    count_total_weak_entities,
    extract_weak_entity_lists,
    extract_weak_entity_spans,
)


INPUT_JSONL = PROJECT_ROOT / "data" / "text" / "doctor_notes" / "mtsamples" / "processed" / "mtsamples_doctor_notes.jsonl"
OUTPUT_JSONL = PROJECT_ROOT / "data" / "text" / "doctor_notes" / "mtsamples" / "processed" / "mtsamples_weak_entities.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "doctor_note_training"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_weak_labels(input_path: Path = INPUT_JSONL, output_path: Path = OUTPUT_JSONL) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Prepared MTSamples JSONL not found: {input_path}")

    input_records = _read_jsonl(input_path)
    output_records: List[Dict[str, Any]] = []
    notes_with_entities = 0
    entity_totals = Counter()

    for index, record in enumerate(input_records, start=1):
        raw_text = str(record.get("raw_text", ""))
        clean_text = str(record.get("clean_text", ""))
        keywords = str(record.get("keywords", ""))
        weak_entities = extract_weak_entity_lists(clean_text, keywords=keywords)
        weak_entity_spans = extract_weak_entity_spans(clean_text)
        weak_entity_count = count_total_weak_entities(weak_entities)

        if weak_entity_count > 0:
            notes_with_entities += 1

        for label, values in weak_entities.items():
            entity_totals[label] += len(values)

        output_record = {
            "note_id": record.get("note_id", f"mtsamples_{index:06d}"),
            "source": record.get("source", "MTSamples"),
            "input_type": record.get("input_type", "doctor_note_text"),
            "medical_specialty": record.get("medical_specialty", ""),
            "description": record.get("description", ""),
            "sample_name": record.get("sample_name", ""),
            "keywords": keywords,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "weak_entities": weak_entities,
            "weak_entity_spans": weak_entity_spans,
            "weak_entity_count": weak_entity_count,
            "label_source": "rules_and_keywords",
            "task_use": record.get(
                "task_use",
                ["doctor_note_classification", "weak_entity_extraction", "text_modality_fusion"],
            ),
        }
        output_records.append(output_record)

    _write_jsonl(output_path, output_records)

    summary = {
        "notes_processed": len(output_records),
        "notes_with_at_least_one_weak_entity": notes_with_entities,
        "total_weak_entities": int(sum(entity_totals.values())),
        "average_weak_entities_per_note": (sum(entity_totals.values()) / len(output_records)) if output_records else 0.0,
        "entity_count_by_category": dict(entity_totals),
        "input_jsonl": str(input_path),
        "output_jsonl": str(output_path),
        "note": "These are weak labels generated from rules and keywords. They are not expert clinical annotations.",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "weak_label_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Weak Doctor-Note Label Report",
        "",
        f"- Notes processed: {summary['notes_processed']}",
        f"- Notes with at least one weak entity: {summary['notes_with_at_least_one_weak_entity']}",
        f"- Total weak entities: {summary['total_weak_entities']}",
        f"- Average weak entities per note: {summary['average_weak_entities_per_note']:.2f}",
        "",
        "## Entity Count by Category",
    ]

    for label, value in summary["entity_count_by_category"].items():
        report_lines.append(f"- {label}: {value}")

    report_lines.extend([
        "",
        "## Limitation",
        "These are weak labels generated from rules and keywords. They are not expert clinical annotations.",
    ])

    (OUTPUT_DIR / "weak_label_report.md").write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate weak labels for MTSamples doctor notes.")
    parser.add_argument("--input-jsonl", type=Path, default=INPUT_JSONL)
    parser.add_argument("--output-jsonl", type=Path, default=OUTPUT_JSONL)
    args = parser.parse_args()
    summary = generate_weak_labels(args.input_jsonl, args.output_jsonl)
    print(summary["note"])
    print(f"Wrote {summary['output_jsonl']}")


if __name__ == "__main__":
    main()
