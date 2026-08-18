from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.common import average, safe_ratio, save_csv, save_json, write_markdown
from src.model3.pipeline import run_fusion_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "cross_modal_validation_v2"
CASES_DIR = OUTPUT_DIR / "cases"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "thesis_figures" / "cross_modal_input_combination_validation.png"
RANDOM_SEED = 42
MODEL3_FUNCTION = "src.model3.pipeline.run_fusion_pipeline"

SCENARIO_SPECS = [
    ("image_only", 20, True, False, False),
    ("scanned_document_only", 20, False, True, False),
    ("doctor_note_only", 20, False, False, True),
    ("image_scanned_document", 10, True, True, False),
    ("image_doctor_note", 10, True, False, True),
    ("scanned_document_doctor_note", 10, False, True, True),
    ("image_scanned_document_doctor_note", 10, True, True, True),
]

MANIFEST_FIELDS = [
    "case_id",
    "scenario",
    "pairing_type",
    "image_source_id",
    "image_source_path",
    "image_modality",
    "document_source_id",
    "document_source_path",
    "document_type",
    "doctor_note_source_id",
    "doctor_note_source_path",
    "random_seed",
    "model3_function_called",
    "output_path",
    "execution_success",
    "execution_time_seconds",
    "error_message",
]

SUMMARY_FIELDS = [
    "scenario",
    "pairing_type",
    "requested_cases",
    "completed_cases",
    "failed_cases",
    "pipeline_output_success_rate",
    "patient_summary_generation_rate",
    "non_validated_follow_up_note_generation_rate",
    "retrieved_evidence_availability_rate",
    "average_retrieved_evidence_count",
    "structured_json_completion_rate",
    "average_completed_field_count",
    "average_missing_input_warning_count",
    "average_extracted_finding_entity_count",
    "average_summary_length",
    "average_follow_up_note_length",
    "average_execution_time_seconds",
]


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {path}:{line_number}")
        records.append(payload)
    return records


def _load_image_pool() -> Dict[str, List[Dict[str, Any]]]:
    run_root = PROJECT_ROOT / "outputs" / "final_run_100_tuned_v2"
    groups: Dict[str, List[Dict[str, Any]]] = {"brain_mri": [], "chest_xray": []}
    prefixes = {"brain_mri": "main_brain_", "chest_xray": "main_xray_"}
    for modality, prefix in prefixes.items():
        for case_dir in sorted(run_root.glob(f"{prefix}*")):
            output_path = case_dir / "model1_output.json"
            if not output_path.is_file():
                continue
            output = _read_json(output_path)
            if not output.get("patient_summary_text"):
                continue
            groups[modality].append(
                {
                    "source_id": case_dir.name,
                    "source_path": _relative(output_path),
                    "source_type": modality,
                    "output": output,
                }
            )
    return groups


def _load_document_pool() -> Dict[str, List[Dict[str, Any]]]:
    run_root = PROJECT_ROOT / "outputs" / "final_run_100_tuned_v2"
    groups: Dict[str, List[Dict[str, Any]]] = {"prescription": [], "lab_report": []}
    prefixes = {"prescription": "main_prescription_", "lab_report": "main_lab_"}
    for document_type, prefix in prefixes.items():
        for case_dir in sorted(run_root.glob(f"{prefix}*")):
            output_path = case_dir / "model2_output.json"
            if not output_path.is_file():
                continue
            output = _read_json(output_path)
            if not (output.get("patient_summary") or output.get("raw_text_preview")):
                continue
            groups[document_type].append(
                {
                    "source_id": case_dir.name,
                    "source_path": _relative(output_path),
                    "source_type": document_type,
                    "output": output,
                }
            )
    return groups


def _load_doctor_note_pool() -> List[Dict[str, Any]]:
    output_path = PROJECT_ROOT / "outputs" / "doctor_note_training" / "doctor_note_inference_outputs.jsonl"
    pool: List[Dict[str, Any]] = []
    for record in _read_jsonl(output_path):
        output = record.get("doctor_note_output")
        source_id = str(record.get("note_id", "")).strip()
        if record.get("status") != "success" or not source_id or not isinstance(output, dict):
            continue
        if not output.get("patient_summary_text"):
            continue
        pool.append(
            {
                "source_id": source_id,
                "source_path": _relative(output_path),
                "source_type": "doctor_note_text",
                "output": output,
            }
        )
    return pool


class BalancedSelector:
    def __init__(self, groups: Mapping[str, Sequence[Dict[str, Any]]], rng: random.Random) -> None:
        self.group_names = sorted(groups)
        self.groups: Dict[str, List[Dict[str, Any]]] = {}
        self.positions: Dict[str, int] = {}
        for group_name in self.group_names:
            items = list(groups[group_name])
            if not items:
                raise ValueError(f"No valid records are available for source group {group_name}.")
            rng.shuffle(items)
            self.groups[group_name] = items
            self.positions[group_name] = 0

    def take(self, count: int) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for index in range(count):
            group_name = self.group_names[index % len(self.group_names)]
            items = self.groups[group_name]
            position = self.positions[group_name]
            selected.append(items[position % len(items)])
            self.positions[group_name] = position + 1
        return selected


class SequentialSelector:
    def __init__(self, items: Sequence[Dict[str, Any]], rng: random.Random) -> None:
        self.items = list(items)
        if not self.items:
            raise ValueError("No valid doctor-note records are available.")
        rng.shuffle(self.items)
        self.position = 0

    def take(self, count: int) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for _ in range(count):
            selected.append(self.items[self.position % len(self.items)])
            self.position += 1
        return selected


def _follow_up_note(output: Mapping[str, Any]) -> str:
    for key in ("follow_up_note", "generated_follow_up_note", "doctor_feedback", "doctor_oriented_feedback"):
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _count_collection(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return sum(_count_collection(nested) for nested in value.values())
    return 0


def _count_upstream_findings(
    image_output: Optional[Mapping[str, Any]],
    document_output: Optional[Mapping[str, Any]],
    doctor_note_output: Optional[Mapping[str, Any]],
) -> int:
    count = 0
    if image_output:
        positives = image_output.get("xray_positive_labels")
        predictions = image_output.get("top_predictions")
        if isinstance(positives, list) and positives:
            count += len(positives)
        elif isinstance(predictions, list) and predictions:
            count += min(len(predictions), 5)
        else:
            count += int(bool(image_output.get("patient_summary_text")))
    if document_output:
        entities = document_output.get("entities")
        structured = document_output.get("structured_json")
        entity_count = _count_collection(entities)
        count += entity_count if entity_count else _count_collection(structured)
        if not entity_count and not _count_collection(structured):
            count += int(bool(document_output.get("patient_summary")))
    if doctor_note_output:
        entity_count = _count_collection(doctor_note_output.get("entities"))
        count += entity_count if entity_count else int(bool(doctor_note_output.get("patient_summary_text")))
    return count


def _derive_metrics(output: Mapping[str, Any], expected_modalities: Sequence[str], execution_time: float) -> Dict[str, Any]:
    patient_summary = output.get("patient_summary")
    final_summary = output.get("final_summary")
    follow_up_note = _follow_up_note(output)
    evidence = output.get("retrieved_evidence")
    modalities = output.get("available_modalities")
    warnings = output.get("missing_information_notes")

    completed_checks = {
        "patient_summary": isinstance(patient_summary, str) and bool(patient_summary.strip()),
        "final_summary": isinstance(final_summary, str) and bool(final_summary.strip()),
        "non_validated_follow_up_note": bool(follow_up_note),
        "retrieved_evidence_field": isinstance(evidence, list),
        "available_modalities": isinstance(modalities, list) and set(modalities) == set(expected_modalities),
        "missing_information_notes": isinstance(warnings, list),
        "fused_query": isinstance(output.get("fused_query"), str) and bool(output.get("fused_query", "").strip()),
        "kb_used": isinstance(output.get("kb_used"), str) and bool(output.get("kb_used", "").strip()),
    }

    output_success = (
        isinstance(output, dict)
        and completed_checks["patient_summary"]
        and completed_checks["final_summary"]
        and completed_checks["non_validated_follow_up_note"]
    )

    return {
        "pipeline_output_success": int(output_success),
        "patient_summary_generation": int(completed_checks["patient_summary"]),
        "non_validated_follow_up_note_generation": int(completed_checks["non_validated_follow_up_note"]),
        "retrieved_evidence_availability": int(isinstance(evidence, list) and bool(evidence)),
        "retrieved_evidence_count": len(evidence) if isinstance(evidence, list) else 0,
        "structured_json_completion": int(all(completed_checks.values())),
        "completed_field_count": sum(int(value) for value in completed_checks.values()),
        "completed_field_checks": completed_checks,
        "missing_input_warning_count": len(warnings) if isinstance(warnings, list) else 0,
        "summary_length": len(patient_summary.strip()) if isinstance(patient_summary, str) else 0,
        "follow_up_note_length": len(follow_up_note),
        "execution_time_seconds": execution_time,
    }


def _pairing_type(has_image: bool, has_document: bool, has_doctor_note: bool) -> str:
    modality_count = sum((has_image, has_document, has_doctor_note))
    if modality_count == 1:
        return "single_source_technical_case"
    return "synthetic_unpaired_technical_combination"


def _build_provenance(
    case_id: str,
    scenario: str,
    image_source: Optional[Mapping[str, Any]],
    document_source: Optional[Mapping[str, Any]],
    doctor_note_source: Optional[Mapping[str, Any]],
    output_path: Path,
) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "scenario": scenario,
        "pairing_type": _pairing_type(bool(image_source), bool(document_source), bool(doctor_note_source)),
        "image_source_id": image_source.get("source_id", "") if image_source else "",
        "image_source_path": image_source.get("source_path", "") if image_source else "",
        "image_modality": image_source.get("source_type", "") if image_source else "",
        "document_source_id": document_source.get("source_id", "") if document_source else "",
        "document_source_path": document_source.get("source_path", "") if document_source else "",
        "document_type": document_source.get("source_type", "") if document_source else "",
        "doctor_note_source_id": doctor_note_source.get("source_id", "") if doctor_note_source else "",
        "doctor_note_source_path": doctor_note_source.get("source_path", "") if doctor_note_source else "",
        "random_seed": RANDOM_SEED,
        "model3_function_called": MODEL3_FUNCTION,
        "output_path": _relative(output_path),
        "execution_success": 0,
        "execution_time_seconds": 0.0,
        "error_message": "",
    }


def _run_case(
    case_id: str,
    scenario: str,
    image_source: Optional[Mapping[str, Any]],
    document_source: Optional[Mapping[str, Any]],
    doctor_note_source: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    output_path = CASES_DIR / f"{case_id}.json"
    provenance = _build_provenance(case_id, scenario, image_source, document_source, doctor_note_source, output_path)
    image_output = image_source.get("output") if image_source else None
    document_output = document_source.get("output") if document_source else None
    doctor_note_output = doctor_note_source.get("output") if doctor_note_source else None
    expected_modalities = [
        name
        for name, source in (("image", image_source), ("document", document_source), ("doctor_note", doctor_note_source))
        if source
    ]

    start = time.perf_counter()
    try:
        model3_output = run_fusion_pipeline(
            case_id=case_id,
            model1_output=image_output,
            model2_output=document_output,
            doctor_note_output=doctor_note_output,
        )
        elapsed = time.perf_counter() - start
        if not isinstance(model3_output, dict) or not model3_output:
            raise RuntimeError("Model-3 returned an empty or non-object output.")
        metrics = _derive_metrics(model3_output, expected_modalities, elapsed)
        metrics["extracted_finding_entity_count"] = _count_upstream_findings(
            image_output, document_output, doctor_note_output
        )
        provenance["execution_success"] = 1
        provenance["execution_time_seconds"] = elapsed
        save_json(
            output_path,
            {
                "case_id": case_id,
                "scenario": scenario,
                "pairing_type": provenance["pairing_type"],
                "random_seed": RANDOM_SEED,
                "model3_function_called": MODEL3_FUNCTION,
                "source_provenance": {
                    key: provenance[key]
                    for key in (
                        "image_source_id",
                        "image_source_path",
                        "image_modality",
                        "document_source_id",
                        "document_source_path",
                        "document_type",
                        "doctor_note_source_id",
                        "doctor_note_source_path",
                    )
                },
                "metric_definitions_version": 2,
                "derived_metrics": metrics,
                "model3_output": model3_output,
            },
        )
        return {"provenance": provenance, "metrics": metrics}
    except Exception as error:
        elapsed = time.perf_counter() - start
        provenance["execution_time_seconds"] = elapsed
        provenance["error_message"] = f"{type(error).__name__}: {error}"
        save_json(
            output_path,
            {
                "case_id": case_id,
                "scenario": scenario,
                "pairing_type": provenance["pairing_type"],
                "random_seed": RANDOM_SEED,
                "model3_function_called": MODEL3_FUNCTION,
                "source_provenance": provenance,
                "execution_success": False,
                "error_message": provenance["error_message"],
            },
        )
        return {"provenance": provenance, "metrics": None}


def _aggregate_scenario(scenario: str, rows: Sequence[Dict[str, Any]], requested_cases: int) -> Dict[str, Any]:
    completed_rows = [row for row in rows if row["provenance"]["execution_success"] and row["metrics"]]
    metrics = [row["metrics"] for row in completed_rows]
    completed = len(completed_rows)
    pairing_type = rows[0]["provenance"]["pairing_type"] if rows else ""
    return {
        "scenario": scenario,
        "pairing_type": pairing_type,
        "requested_cases": requested_cases,
        "completed_cases": completed,
        "failed_cases": requested_cases - completed,
        "pipeline_output_success_rate": safe_ratio(sum(row["pipeline_output_success"] for row in metrics), requested_cases),
        "patient_summary_generation_rate": safe_ratio(sum(row["patient_summary_generation"] for row in metrics), requested_cases),
        "non_validated_follow_up_note_generation_rate": safe_ratio(
            sum(row["non_validated_follow_up_note_generation"] for row in metrics), requested_cases
        ),
        "retrieved_evidence_availability_rate": safe_ratio(
            sum(row["retrieved_evidence_availability"] for row in metrics), requested_cases
        ),
        "average_retrieved_evidence_count": average([float(row["retrieved_evidence_count"]) for row in metrics]),
        "structured_json_completion_rate": safe_ratio(
            sum(row["structured_json_completion"] for row in metrics), requested_cases
        ),
        "average_completed_field_count": average([float(row["completed_field_count"]) for row in metrics]),
        "average_missing_input_warning_count": average(
            [float(row["missing_input_warning_count"]) for row in metrics]
        ),
        "average_extracted_finding_entity_count": average(
            [float(row["extracted_finding_entity_count"]) for row in metrics]
        ),
        "average_summary_length": average([float(row["summary_length"]) for row in metrics]),
        "average_follow_up_note_length": average([float(row["follow_up_note_length"]) for row in metrics]),
        "average_execution_time_seconds": average([float(row["execution_time_seconds"]) for row in metrics]),
    }


def _reload_saved_rows(manifest_rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    scenario_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for provenance in manifest_rows:
        output_path = PROJECT_ROOT / str(provenance["output_path"])
        if not output_path.is_file():
            raise FileNotFoundError(f"Manifest output does not exist: {output_path}")
        payload = _read_json(output_path)
        metrics = payload.get("derived_metrics") if provenance["execution_success"] else None
        if provenance["execution_success"] and not isinstance(metrics, dict):
            raise ValueError(f"Successful case has no persisted derived metrics: {output_path}")
        scenario_rows[str(provenance["scenario"])].append(
            {"provenance": dict(provenance), "metrics": metrics}
        )
    return scenario_rows


def _write_figure(summary_rows: Sequence[Mapping[str, Any]]) -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "Image",
        "Document",
        "Doctor note",
        "Image + doc",
        "Image + note",
        "Doc + note",
        "All three",
    ]
    x_positions = list(range(len(summary_rows)))
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1.25]})

    completed = [float(row["completed_cases"]) for row in summary_rows]
    requested = [float(row["requested_cases"]) for row in summary_rows]
    axes[0].bar(x_positions, requested, color="#d8dde3", label="Requested")
    axes[0].bar(x_positions, completed, color="#287271", label="Completed")
    axes[0].set_ylabel("Cases")
    axes[0].set_title("Cross-Modal Input-Combination Technical Validation")
    axes[0].set_xticks(x_positions, labels)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    width = 0.26
    evidence = [float(row["average_retrieved_evidence_count"]) for row in summary_rows]
    completed_fields = [float(row["average_completed_field_count"]) for row in summary_rows]
    warnings = [float(row["average_missing_input_warning_count"]) for row in summary_rows]
    axes[1].bar([x - width for x in x_positions], evidence, width, color="#3b6ea8", label="Avg. evidence")
    axes[1].bar(x_positions, completed_fields, width, color="#d98b2b", label="Avg. completed fields")
    axes[1].bar([x + width for x in x_positions], warnings, width, color="#a33f55", label="Avg. missing-input warnings")
    axes[1].set_ylabel("Count")
    axes[1].set_xticks(x_positions, labels, rotation=20, ha="right")
    axes[1].legend(ncol=3)
    axes[1].grid(axis="y", alpha=0.25)

    figure.text(
        0.5,
        0.01,
        "Multi-input cases are synthetic/unpaired technical combinations; this is not clinical validation.",
        ha="center",
        color="#8a2635",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 1))
    figure.savefig(FIGURE_PATH, dpi=220)
    plt.close(figure)


def _report_lines(summary: Mapping[str, Any]) -> List[str]:
    lines = [
        "# Cross-Modal Input-Combination Validation V2",
        "",
        f"- Random seed: {summary['random_seed']}",
        f"- Model-3 function called: `{summary['model3_function_called']}`",
        f"- Requested/completed/failed: {summary['requested_cases']}/{summary['completed_cases']}/{summary['failed_cases']}",
        f"- Provenance CSV: `{summary['provenance_manifest_csv']}`",
        f"- Provenance JSON: `{summary['provenance_manifest_json']}`",
        "",
        "## Metric Definitions",
        "",
        "- **Pipeline output success:** Model-3 returned a non-empty object containing non-empty patient-summary, final-summary, and legacy-compatible follow-up-note values, and the per-case record was saved.",
        "- **Evidence availability:** `retrieved_evidence` is a non-empty list.",
        "- **Completed field:** one of eight expected output checks is satisfied: patient summary, final summary, non-validated follow-up note, evidence-list field, exact available-modality list, missing-information list, fused query, and knowledge-base path.",
        "- **Missing-input warning:** one item in Model-3 `missing_information_notes`.",
        "- **Structured JSON completion:** all eight expected field checks are satisfied in the saved per-case record.",
        "- **Summary length:** character count of the generated patient-summary string.",
        "- **Follow-up-note length:** character count of the non-validated follow-up-note value read from the legacy-compatible Model-3 output.",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Pairing type | Requested | Completed | Failed | Model-3 output success | Evidence availability | Avg. evidence | JSON completion | Avg. completed fields | Avg. warnings |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scenario_results"]:
        lines.append(
            f"| {row['scenario']} | {row['pairing_type']} | {row['requested_cases']} | "
            f"{row['completed_cases']} | {row['failed_cases']} | {row['pipeline_output_success_rate']:.4f} | "
            f"{row['retrieved_evidence_availability_rate']:.4f} | {row['average_retrieved_evidence_count']:.2f} | "
            f"{row['structured_json_completion_rate']:.4f} | {row['average_completed_field_count']:.2f} | "
            f"{row['average_missing_input_warning_count']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This validation measures technical input handling, output generation, retrieval availability, structured completion, and missing-modality handling. Combined cases use synthetic or unpaired upstream outputs because a verified same-patient paired multimodal dataset was unavailable. The experiment does not establish clinical correctness, patient-level multimodal reasoning, or superiority over another model.",
            "",
            "The internal legacy field name is retained for compatibility. Report-facing text describes the value as a non-validated retrieval-supported follow-up note.",
        ]
    )
    return lines


def run_cross_modal_validation_v2() -> Dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    image_groups = _load_image_pool()
    document_groups = _load_document_pool()
    doctor_note_pool = _load_doctor_note_pool()
    image_selector = BalancedSelector(image_groups, rng)
    document_selector = BalancedSelector(document_groups, rng)
    doctor_note_selector = SequentialSelector(doctor_note_pool, rng)

    all_rows: List[Dict[str, Any]] = []
    for scenario, count, has_image, has_document, has_doctor_note in SCENARIO_SPECS:
        image_sources = image_selector.take(count) if has_image else [None] * count
        document_sources = document_selector.take(count) if has_document else [None] * count
        doctor_note_sources = doctor_note_selector.take(count) if has_doctor_note else [None] * count
        for index, (image_source, document_source, doctor_note_source) in enumerate(
            zip(image_sources, document_sources, doctor_note_sources), start=1
        ):
            case_id = f"cmv2_{scenario}_{index:03d}"
            row = _run_case(case_id, scenario, image_source, document_source, doctor_note_source)
            all_rows.append(row)

    manifest_rows = [row["provenance"] for row in all_rows]
    errors = [row for row in manifest_rows if not row["execution_success"]]
    manifest_csv_path = OUTPUT_DIR / "provenance_manifest.csv"
    manifest_json_path = OUTPUT_DIR / "provenance_manifest.json"
    save_csv(manifest_csv_path, manifest_rows, MANIFEST_FIELDS)
    save_json(manifest_json_path, manifest_rows)
    save_csv(OUTPUT_DIR / "cross_modal_case_errors.csv", errors, MANIFEST_FIELDS)

    scenario_rows = _reload_saved_rows(manifest_rows)
    summary_rows = [
        _aggregate_scenario(scenario, scenario_rows[scenario], requested)
        for scenario, requested, _, _, _ in SCENARIO_SPECS
    ]
    requested_cases = sum(row["requested_cases"] for row in summary_rows)
    completed_cases = sum(row["completed_cases"] for row in summary_rows)
    failed_cases = sum(row["failed_cases"] for row in summary_rows)
    summary = {
        "validation_version": 2,
        "random_seed": RANDOM_SEED,
        "model3_function_called": MODEL3_FUNCTION,
        "script_path": _relative(Path(__file__)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aggregation_source": "persisted per-case JSON files listed in the provenance manifest",
        "requested_cases": requested_cases,
        "completed_cases": completed_cases,
        "failed_cases": failed_cases,
        "source_pool_counts": {
            "brain_mri": len(image_groups["brain_mri"]),
            "chest_xray": len(image_groups["chest_xray"]),
            "prescription": len(document_groups["prescription"]),
            "lab_report": len(document_groups["lab_report"]),
            "doctor_note": len(doctor_note_pool),
        },
        "selection": {
            "image": "seeded shuffled pools with round-robin Brain MRI/Chest X-ray balance",
            "scanned_document": "seeded shuffled pools with round-robin prescription/lab-report balance",
            "doctor_note": "seeded shuffled unique sequence until pool exhaustion",
            "multi_input_pairing": "synthetic_unpaired_technical_combination",
        },
        "metric_definitions": {
            "pipeline_output_success": "saved Model-3 object with non-empty patient summary, final summary, and legacy-compatible follow-up-note value",
            "evidence_availability": "retrieved_evidence is a non-empty list",
            "completed_field": "one satisfied check among the eight documented expected output checks",
            "missing_input_warning": "one item in missing_information_notes",
            "structured_json_completion": "all eight expected output checks are satisfied",
            "summary_length": "character count of patient_summary",
            "follow_up_note_length": "character count of the non-validated follow-up-note value",
        },
        "provenance_manifest_csv": _relative(manifest_csv_path),
        "provenance_manifest_json": _relative(manifest_json_path),
        "cases_directory": _relative(CASES_DIR),
        "errors_csv": _relative(OUTPUT_DIR / "cross_modal_case_errors.csv"),
        "figure_path": _relative(FIGURE_PATH),
        "scenario_results": summary_rows,
        "claim_boundary": (
            "This validation measures technical input handling, output generation, retrieval availability, structured completion, "
            "and missing-modality handling. Combined cases use synthetic or unpaired upstream outputs because a verified "
            "same-patient paired multimodal dataset was unavailable. The experiment does not establish clinical correctness, "
            "patient-level multimodal reasoning, or superiority over another model."
        ),
        "legacy_field_note": (
            "The internal legacy field name is retained for compatibility. Report-facing text describes the value as a "
            "non-validated retrieval-supported follow-up note."
        ),
    }
    save_json(OUTPUT_DIR / "cross_modal_validation_summary.json", summary)
    save_csv(OUTPUT_DIR / "cross_modal_validation_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_markdown(OUTPUT_DIR / "cross_modal_validation_report.md", _report_lines(summary))
    _write_figure(summary_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic seven-scenario technical validation through the stable Model-3 pipeline."
    )
    return parser.parse_args()


def main() -> None:
    parse_args()
    summary = run_cross_modal_validation_v2()
    print(f"Command module: src.evaluation.run_cross_modal_validation_v2")
    print(f"Requested/completed/failed: {summary['requested_cases']}/{summary['completed_cases']}/{summary['failed_cases']}")
    print(f"Provenance: {summary['provenance_manifest_csv']}")


if __name__ == "__main__":
    main()
