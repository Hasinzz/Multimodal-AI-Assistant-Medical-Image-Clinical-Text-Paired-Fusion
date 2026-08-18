from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.model3.pipeline import run_fusion_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "cross_modal_validation_v2"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "model3_ablation"
CASES_DIR = OUTPUT_DIR / "cases"
MANIFEST_PATH = SOURCE_DIR / "provenance_manifest.json"
DOCTOR_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "doctor_note_training" / "doctor_note_inference_outputs.jsonl"
SEED = 42
VARIANTS = (("late_fusion_no_rag", False), ("stable_late_fusion_with_rag", True))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty results: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_doctor_outputs() -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for line in DOCTOR_OUTPUT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "success" and isinstance(record.get("doctor_note_output"), dict):
            outputs[str(record.get("note_id"))] = record["doctor_note_output"]
    return outputs


def load_optional_json(relative_path: str) -> Optional[dict[str, Any]]:
    if not relative_path:
        return None
    payload = read_json(PROJECT_ROOT / relative_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {relative_path}")
    return payload


def normalize(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def recursive_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        if normalize(value):
            strings.append(value)
    elif isinstance(value, Mapping):
        for nested in value.values():
            strings.extend(recursive_strings(nested))
    elif isinstance(value, list):
        for nested in value:
            strings.extend(recursive_strings(nested))
    return strings


def source_entities(
    image_output: Optional[Mapping[str, Any]],
    document_output: Optional[Mapping[str, Any]],
    doctor_output: Optional[Mapping[str, Any]],
) -> list[str]:
    entities: list[str] = []
    if image_output:
        positives = image_output.get("xray_positive_labels")
        if isinstance(positives, list):
            entities.extend(str(item) for item in positives)
        predictions = image_output.get("top_predictions")
        if isinstance(predictions, list) and predictions:
            label = predictions[0].get("label") if isinstance(predictions[0], dict) else None
            if label:
                entities.append(str(label))
    if document_output:
        for entity in document_output.get("entities", []) if isinstance(document_output.get("entities"), list) else []:
            if isinstance(entity, dict) and entity.get("text"):
                entities.append(str(entity["text"]))
            elif isinstance(entity, str):
                entities.append(entity)
    if doctor_output:
        entities.extend(recursive_strings(doctor_output.get("entities", {})))
    deduplicated = {normalize(item): str(item) for item in entities if len(normalize(item)) >= 2}
    return list(deduplicated.values())


def source_findings(
    image_output: Optional[Mapping[str, Any]],
    document_output: Optional[Mapping[str, Any]],
    doctor_output: Optional[Mapping[str, Any]],
) -> list[str]:
    values = [
        image_output.get("patient_summary_text") if image_output else None,
        document_output.get("patient_summary") if document_output else None,
        doctor_output.get("patient_summary_text") if doctor_output else None,
    ]
    return [str(value) for value in values if isinstance(value, str) and normalize(value)]


def retention(items: Sequence[str], output_text: str) -> Optional[float]:
    if not items:
        return None
    normalized_output = normalize(output_text)
    represented = sum(normalize(item) in normalized_output for item in items)
    return represented / len(items)


def derive_metrics(
    output: Mapping[str, Any],
    image_output: Optional[Mapping[str, Any]],
    document_output: Optional[Mapping[str, Any]],
    doctor_output: Optional[Mapping[str, Any]],
    runtime_seconds: float,
) -> dict[str, object]:
    entities = source_entities(image_output, document_output, doctor_output)
    findings = source_findings(image_output, document_output, doctor_output)
    summary_fields = [
        output.get("image_findings"),
        output.get("document_findings"),
        output.get("doctor_note_findings"),
        output.get("patient_summary"),
        output.get("final_summary"),
    ]
    output_text = " ".join(str(value) for value in summary_fields if isinstance(value, str))
    evidence = output.get("retrieved_evidence") if isinstance(output.get("retrieved_evidence"), list) else []
    evidence_text = " ".join(str(item.get("text", "")) for item in evidence if isinstance(item, dict))
    structured_findings = [
        str(output[key])
        for key in ("image_findings", "document_findings", "doctor_note_findings")
        if isinstance(output.get(key), str) and normalize(output[key])
    ]
    support_text = normalize(" ".join(findings + [evidence_text]))
    unsupported_count = sum(normalize(item) not in support_text for item in structured_findings)
    required_fields = [
        "case_id",
        "available_modalities",
        "retrieved_evidence",
        "patient_summary",
        "missing_information_notes",
        "final_summary",
        "fused_query",
        "kb_used",
        "rag_enabled",
    ]
    completed_fields = sum(key in output and output[key] is not None for key in required_fields)
    return {
        "output_success": 1,
        "summary_generation": int(bool(str(output.get("patient_summary", "")).strip())),
        "evidence_availability": int(bool(evidence)),
        "evidence_count": len(evidence),
        "source_entity_count": len(entities),
        "source_entity_retention": retention(entities, output_text),
        "source_finding_count": len(findings),
        "source_finding_retention": retention(findings, output_text),
        "structured_finding_count": len(structured_findings),
        "unsupported_entity_proxy_count": unsupported_count,
        "unsupported_entity_proxy_rate": unsupported_count / len(structured_findings) if structured_findings else None,
        "json_completed_fields": completed_fields,
        "json_required_fields": len(required_fields),
        "json_completion": int(completed_fields == len(required_fields)),
        "summary_length_characters": len(str(output.get("final_summary", ""))),
        "runtime_seconds": runtime_seconds,
        "missing_modality_warning_count": len(output.get("missing_information_notes", [])),
    }


def mean_non_null(rows: Sequence[dict[str, object]], key: str) -> Optional[float]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def paired_bootstrap(differences: np.ndarray, rng: np.random.Generator, iterations: int = 2000):
    if differences.size == 0:
        return None
    samples = rng.choice(differences, size=(iterations, differences.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def main() -> None:
    manifest = read_json(MANIFEST_PATH)
    if not isinstance(manifest, list) or len(manifest) != 100:
        raise ValueError(f"Expected exactly 100 source cases, found {len(manifest) if isinstance(manifest, list) else 'invalid'}")
    doctor_outputs = load_doctor_outputs()
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for case in manifest:
        case_id = str(case["case_id"])
        image_output = load_optional_json(str(case.get("image_source_path", "")))
        document_output = load_optional_json(str(case.get("document_source_path", "")))
        doctor_id = str(case.get("doctor_note_source_id", ""))
        doctor_output = doctor_outputs.get(doctor_id) if doctor_id else None
        for variant, use_rag in VARIANTS:
            started = time.perf_counter()
            try:
                output = run_fusion_pipeline(
                    case_id=case_id,
                    model1_output=image_output,
                    model2_output=document_output,
                    doctor_note_output=doctor_output,
                    use_rag=use_rag,
                )
                elapsed = time.perf_counter() - started
                metrics = derive_metrics(
                    output, image_output, document_output, doctor_output, elapsed
                )
                output_path = CASES_DIR / f"{case_id}_{variant}.json"
                write_json(
                    output_path,
                    {
                        "case_id": case_id,
                        "scenario": case["scenario"],
                        "pairing_type": case["pairing_type"],
                        "variant": variant,
                        "use_rag": use_rag,
                        "source_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
                        "source_provenance": case,
                        "automated_metrics": metrics,
                        "model3_output": output,
                    },
                )
                row = {
                    "case_id": case_id,
                    "scenario": case["scenario"],
                    "pairing_type": case["pairing_type"],
                    "variant": variant,
                    "use_rag": int(use_rag),
                    "output_path": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    **metrics,
                    "error_message": "",
                }
            except Exception as exc:
                elapsed = time.perf_counter() - started
                row = {
                    "case_id": case_id,
                    "scenario": case["scenario"],
                    "pairing_type": case["pairing_type"],
                    "variant": variant,
                    "use_rag": int(use_rag),
                    "output_path": "",
                    "output_success": 0,
                    "summary_generation": 0,
                    "evidence_availability": 0,
                    "evidence_count": 0,
                    "source_entity_count": None,
                    "source_entity_retention": None,
                    "source_finding_count": None,
                    "source_finding_retention": None,
                    "structured_finding_count": None,
                    "unsupported_entity_proxy_count": None,
                    "unsupported_entity_proxy_rate": None,
                    "json_completed_fields": 0,
                    "json_required_fields": 9,
                    "json_completion": 0,
                    "summary_length_characters": 0,
                    "runtime_seconds": elapsed,
                    "missing_modality_warning_count": None,
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
                failures.append(row)
            rows.append(row)
        print(f"[ablation] completed paired case {case_id}", flush=True)

    write_csv(OUTPUT_DIR / "model3_ablation_case_results.csv", rows)
    if failures:
        write_csv(OUTPUT_DIR / "model3_ablation_failures.csv", failures)
    else:
        (OUTPUT_DIR / "model3_ablation_failures.csv").write_text(
            "case_id,variant,error_message\n", encoding="utf-8"
        )

    metric_names = [
        "output_success",
        "summary_generation",
        "evidence_availability",
        "evidence_count",
        "source_entity_retention",
        "source_finding_retention",
        "unsupported_entity_proxy_rate",
        "json_completion",
        "summary_length_characters",
        "runtime_seconds",
        "missing_modality_warning_count",
    ]
    by_variant = {variant: [row for row in rows if row["variant"] == variant] for variant, _ in VARIANTS}
    summary_rows = []
    variant_summary: dict[str, Any] = {}
    for variant, _ in VARIANTS:
        variant_rows = by_variant[variant]
        metrics = {key: mean_non_null(variant_rows, key) for key in metric_names}
        variant_summary[variant] = {
            "requested_cases": 100,
            "completed_cases": sum(int(row["output_success"]) for row in variant_rows),
            "failed_cases": sum(1 - int(row["output_success"]) for row in variant_rows),
            "mean_metrics": metrics,
        }
        summary_rows.append(
            {
                "variant": variant,
                "requested_cases": 100,
                "completed_cases": variant_summary[variant]["completed_cases"],
                "failed_cases": variant_summary[variant]["failed_cases"],
                **metrics,
            }
        )

    rng = np.random.default_rng(SEED)
    paired: dict[str, Any] = {}
    no_rag_lookup = {row["case_id"]: row for row in by_variant["late_fusion_no_rag"]}
    rag_lookup = {row["case_id"]: row for row in by_variant["stable_late_fusion_with_rag"]}
    for metric in metric_names:
        differences = []
        for case_id in sorted(no_rag_lookup):
            left = no_rag_lookup[case_id].get(metric)
            right = rag_lookup[case_id].get(metric)
            if left is not None and right is not None:
                differences.append(float(right) - float(left))
        difference_array = np.asarray(differences, dtype=np.float64)
        paired[metric] = {
            "paired_cases": len(differences),
            "mean_difference_rag_minus_no_rag": float(difference_array.mean()) if difference_array.size else None,
            "paired_bootstrap_95_ci": paired_bootstrap(difference_array, rng),
        }

    summary = {
        "generated": "2026-08-02",
        "random_seed": SEED,
        "source_case_manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_case_count": len(manifest),
        "paired_case_identity_verified": sorted(no_rag_lookup) == sorted(rag_lookup),
        "variants": variant_summary,
        "paired_comparisons": paired,
        "metric_scope": "Automated technical proxies; not expert factuality, clinical quality, or clinical validation.",
    }
    write_json(OUTPUT_DIR / "model3_ablation_summary.json", summary)
    write_csv(OUTPUT_DIR / "model3_ablation_summary.csv", summary_rows)

    report_lines = [
        "# Model-3 Controlled RAG Ablation",
        "",
        "## Protocol",
        "",
        "The same 100 cross-modal v2 source cases were executed twice: late fusion with retrieval disabled and the stable late-fusion pipeline with retrieval enabled. Source IDs, modality combinations, and upstream outputs were held fixed. Combined cases remain synthetic unpaired technical combinations.",
        "",
        "## Metric Definitions",
        "",
        "- Source-entity retention: proportion of upstream labels/extracted entity strings represented in the final structured summary fields.",
        "- Source-finding retention: proportion of upstream modality summary strings represented in the final structured summary fields.",
        "- Unsupported-entity proxy: proportion of structured modality-finding phrases absent from both upstream finding text and retrieved evidence.",
        "- JSON completion: presence of all nine required stable output fields.",
        "- These are automated technical proxies, not expert factuality metrics, clinical-quality metrics, or clinical validation.",
        "",
        "## Aggregate Results",
        "",
        "| Metric | No RAG | Stable RAG | Paired difference (RAG - no RAG) |",
        "|---|---:|---:|---:|",
    ]
    for metric in metric_names:
        no_rag = variant_summary["late_fusion_no_rag"]["mean_metrics"][metric]
        rag = variant_summary["stable_late_fusion_with_rag"]["mean_metrics"][metric]
        difference = paired[metric]["mean_difference_rag_minus_no_rag"]
        report_lines.append(
            f"| {metric} | {'n/a' if no_rag is None else f'{no_rag:.4f}'} | "
            f"{'n/a' if rag is None else f'{rag:.4f}'} | "
            f"{'n/a' if difference is None else f'{difference:.4f}'} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This ablation tests whether retrieval changes technical output properties on fixed inputs. It does not assess correctness against clinician-authored reference summaries, and it cannot establish clinical benefit.",
        ]
    )
    (OUTPUT_DIR / "model3_ablation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    labels = ["Evidence available", "Entity retention", "Finding retention", "JSON complete"]
    keys = ["evidence_availability", "source_entity_retention", "source_finding_retention", "json_completion"]
    no_rag_values = [variant_summary["late_fusion_no_rag"]["mean_metrics"][key] or 0.0 for key in keys]
    rag_values = [variant_summary["stable_late_fusion_with_rag"]["mean_metrics"][key] or 0.0 for key in keys]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.bar(x - width / 2, no_rag_values, width, label="Late fusion, no RAG", color="#557a95")
    ax.bar(x + width / 2, rag_values, width, label="Stable RAG", color="#c26d3a")
    ax.set_ylabel("Mean proportion")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model3_rag_ablation.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
