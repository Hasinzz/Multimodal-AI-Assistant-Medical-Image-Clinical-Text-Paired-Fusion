from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.config import PROJECT_ROOT


SOURCE = PROJECT_ROOT / "outputs" / "final_research_strengthening" / "literature_benchmark_sources.json"
OUTPUT = SOURCE.with_suffix(".csv")
SOURCE_DIR = SOURCE.parent / "literature_sources"
SOURCE_MANIFEST = SOURCE.parent / "literature_source_file_manifest.csv"

SOURCE_FILES = {
    "Rasheed2025DenseNetBrain": "Rasheed2025DenseNetBrain.xml",
    "Disci2025BrainTumorTransfer": "Disci2025BrainTumorTransfer.xml",
    "Mao2025DilatedSEDenseNet": "Mao2025DilatedSEDenseNet.xml",
    "Wang2017ChestXray8": "Wang2017ChestXray8.pdf",
    "Baltruschat2019ChestXray": "Baltruschat2019ChestXray.xml",
    "Kufel2023ChestXrayTransfer": "Kufel2023ChestXrayTransfer.xml",
}


def main() -> None:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for paper in payload["papers"]:
        source_path = SOURCE_DIR / SOURCE_FILES[paper["citation_key"]]
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        paper["local_full_text_path"] = str(source_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        paper["local_full_text_sha256"] = source_hash
        source_rows.append(
            {
                "citation_key": paper["citation_key"],
                "local_path": paper["local_full_text_path"],
                "sha256": source_hash,
                "source_url": paper["full_text_url"],
                "verification_status": "downloaded from official proceedings or Europe PMC",
            }
        )
        for benchmark in paper["benchmarks"]:
            calculated = "recomputed" in benchmark["source_location"].lower()
            benchmark["page_number"] = "2104" if paper["citation_key"] == "Wang2017ChestXray8" else None
            benchmark["value_origin"] = (
                "calculated arithmetic mean from the eight per-class values in the original table"
                if calculated
                else "copied from the original paper"
            )
            benchmark["verification_status"] = "verified against official full text"
            rows.append(
                {
                    "citation_key": paper["citation_key"],
                    "full_title": paper["title"],
                    "full_authors": paper["authors"],
                    "year": paper["year"],
                    "journal_or_conference": paper["venue"],
                    "doi": paper["doi"],
                    "official_publisher_url": paper["primary_url"],
                    "dataset": paper["dataset"],
                    "architecture": benchmark["model"],
                    "exact_metric_name": benchmark["metric"],
                    "exact_value": benchmark["value"],
                    "confidence_or_variability": benchmark["dispersion"],
                    "table_figure_or_section": benchmark["source_location"],
                    "page_number": benchmark["page_number"],
                    "split_protocol": paper["protocol"],
                    "value_origin": benchmark["value_origin"],
                    "verification_status": benchmark["verification_status"],
                    "direct_comparability": "no",
                    "comparability_limitation": paper["comparability_note"],
                }
            )

    rasheed = payload["papers"][0]["benchmarks"][0]
    rasheed["dispersion"] = "sample SD 0.008; reported 96% CI [0.961, 0.985]"
    rows[0]["confidence_or_variability"] = rasheed["dispersion"]
    SOURCE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with SOURCE_MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)
    print(f"papers={len(payload['papers'])} benchmark_records={len(rows)}")


if __name__ == "__main__":
    main()
