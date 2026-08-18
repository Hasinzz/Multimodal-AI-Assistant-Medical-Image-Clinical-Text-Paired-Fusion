from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.run_cross_modal_validation_v2 import OUTPUT_DIR, PROJECT_ROOT, SCENARIO_SPECS, _aggregate_scenario


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_cross_modal_validation_v2() -> Dict[str, Any]:
    manifest_path = OUTPUT_DIR / "provenance_manifest.json"
    summary_path = OUTPUT_DIR / "cross_modal_validation_summary.json"
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    if not isinstance(manifest, list) or not isinstance(summary, dict):
        raise ValueError("Manifest and summary must be a JSON list and object, respectively.")

    expected_total = sum(spec[1] for spec in SCENARIO_SPECS)
    if len(manifest) != expected_total:
        raise AssertionError(f"Expected {expected_total} manifest rows, found {len(manifest)}.")

    rows_by_scenario: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    source_ids = {"image": set(), "document": set(), "doctor_note": set()}
    for provenance in manifest:
        case_path = PROJECT_ROOT / provenance["output_path"]
        if not case_path.is_file():
            raise FileNotFoundError(case_path)
        payload = _load_json(case_path)
        if payload.get("case_id") != provenance["case_id"]:
            raise AssertionError(f"Case ID mismatch in {case_path}")
        if payload.get("model3_function_called") != "src.model3.pipeline.run_fusion_pipeline":
            raise AssertionError(f"Unexpected Model-3 function in {case_path}")

        scenario = provenance["scenario"]
        expected_image = scenario.startswith("image_") or scenario == "image_only"
        expected_document = "scanned_document" in scenario
        expected_doctor_note = "doctor_note" in scenario
        for expected, field in (
            (expected_image, "image_source_id"),
            (expected_document, "document_source_id"),
            (expected_doctor_note, "doctor_note_source_id"),
        ):
            if bool(provenance.get(field)) != expected:
                raise AssertionError(f"Unexpected source presence for {field} in {provenance['case_id']}")

        expected_pairing = (
            "single_source_technical_case"
            if sum((expected_image, expected_document, expected_doctor_note)) == 1
            else "synthetic_unpaired_technical_combination"
        )
        if provenance.get("pairing_type") != expected_pairing:
            raise AssertionError(f"Pairing label mismatch in {provenance['case_id']}")
        if int(provenance.get("execution_success", 0)) != 1:
            raise AssertionError(f"Case did not execute successfully: {provenance['case_id']}")

        for source_name, field in (
            ("image", "image_source_id"),
            ("document", "document_source_id"),
            ("doctor_note", "doctor_note_source_id"),
        ):
            if provenance.get(field):
                source_ids[source_name].add(provenance[field])

        metrics = payload.get("derived_metrics")
        if not isinstance(metrics, dict):
            raise AssertionError(f"No persisted metrics in {case_path}")
        rows_by_scenario[scenario].append({"provenance": provenance, "metrics": metrics})

    recomputed_rows = [
        _aggregate_scenario(scenario, rows_by_scenario[scenario], requested)
        for scenario, requested, _, _, _ in SCENARIO_SPECS
    ]
    recorded_rows = summary.get("scenario_results")
    if recomputed_rows != recorded_rows:
        raise AssertionError("Summary rows do not equal metrics recomputed from saved case files.")
    if summary.get("requested_cases") != expected_total or summary.get("completed_cases") != expected_total:
        raise AssertionError("Summary requested/completed totals are inconsistent.")
    if summary.get("failed_cases") != 0:
        raise AssertionError("Summary contains failed cases.")

    result = {
        "manifest_rows": len(manifest),
        "case_files": len(list((OUTPUT_DIR / "cases").glob("*.json"))),
        "requested_cases": expected_total,
        "completed_cases": summary["completed_cases"],
        "failed_cases": summary["failed_cases"],
        "unique_image_sources": len(source_ids["image"]),
        "unique_document_sources": len(source_ids["document"]),
        "unique_doctor_note_sources": len(source_ids["doctor_note"]),
        "persisted_summary_match": True,
    }
    return result


def main() -> None:
    result = verify_cross_modal_validation_v2()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
