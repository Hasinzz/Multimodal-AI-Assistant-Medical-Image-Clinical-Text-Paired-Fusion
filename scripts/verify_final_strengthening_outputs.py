from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_research_strengthening"
BRAIN = ROOT / "outputs" / "evaluation" / "model1_cross_validation" / "brain_mri"
XRAY = ROOT / "outputs" / "evaluation" / "model1_cross_validation" / "chest_xray"
XRAY_FOLDS = XRAY / "five_fold"
CROSS_MODAL = ROOT / "outputs" / "evaluation" / "cross_modal_validation_v2"
ABLATION = ROOT / "outputs" / "evaluation" / "model3_ablation"
COMPARISON = ROOT / "outputs" / "evaluation" / "model1_model_comparison"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def non_finite_paths(value: Any, prefix: str = "root") -> list[str]:
    hits: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        hits.append(prefix)
    elif isinstance(value, dict):
        for key, child in value.items():
            hits.extend(non_finite_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(non_finite_paths(child, f"{prefix}[{index}]"))
    return hits


def check_file(path: Path, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"missing file: {path.relative_to(ROOT)}")
    elif path.stat().st_size == 0:
        failures.append(f"empty file: {path.relative_to(ROOT)}")


def verify_brain(failures: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"backbones": {}}
    for backbone in ("densenet121", "resnet50"):
        fold_status = []
        outer_group_folds: dict[str, set[int]] = {}
        outer_image_folds: dict[str, set[int]] = {}
        for fold in range(1, 6):
            fold_dir = BRAIN / backbone / f"fold_{fold}"
            checkpoint = ROOT / "checkpoints" / "model1" / "cross_validation" / "brain_mri" / backbone / f"fold_{fold}_best.pt"
            required = [
                fold_dir / "fold_metrics.json",
                fold_dir / "training_history.csv",
                fold_dir / "train_manifest.csv",
                fold_dir / "validation_manifest.csv",
                fold_dir / "test_manifest.csv",
                fold_dir / "test_predictions.csv",
                fold_dir / "class_distribution.csv",
                fold_dir / "confusion_matrix.csv",
                fold_dir / "confusion_matrix.png",
                fold_dir / "error_log.txt",
                checkpoint,
            ]
            for path in required:
                check_file(path, failures)
            if any(not path.exists() or path.stat().st_size == 0 for path in required):
                fold_status.append({"fold": fold, "complete": False})
                continue
            manifests = {
                "train": read_csv(fold_dir / "train_manifest.csv"),
                "validation": read_csv(fold_dir / "validation_manifest.csv"),
                "test": read_csv(fold_dir / "test_manifest.csv"),
            }
            groups = {name: {row["duplicate_group"] for row in rows} for name, rows in manifests.items()}
            overlap = {
                "train_validation": len(groups["train"] & groups["validation"]),
                "train_test": len(groups["train"] & groups["test"]),
                "validation_test": len(groups["validation"] & groups["test"]),
            }
            if any(overlap.values()):
                failures.append(f"Brain {backbone} fold {fold} duplicate-group overlap: {overlap}")
            for group in groups["test"]:
                outer_group_folds.setdefault(group, set()).add(fold)
            for row in manifests["test"]:
                outer_image_folds.setdefault(row["path"], set()).add(fold)
            fold_status.append({"fold": fold, "complete": True, "group_overlap": overlap})
        cross_outer = {group: sorted(folds) for group, folds in outer_group_folds.items() if len(folds) > 1}
        if cross_outer:
            failures.append(f"Brain {backbone} duplicate groups assigned to multiple outer folds: {len(cross_outer)}")
        repeated_outer_images = {path: sorted(folds) for path, folds in outer_image_folds.items() if len(folds) > 1}
        if len(outer_image_folds) != 7200 or repeated_outer_images:
            failures.append(
                f"Brain {backbone} outer coverage: unique_images={len(outer_image_folds)}, "
                f"repeated_images={len(repeated_outer_images)}"
            )
        oof_path = BRAIN / f"{backbone}_out_of_fold_predictions.csv"
        check_file(oof_path, failures)
        oof_rows = read_csv(oof_path) if oof_path.exists() and oof_path.stat().st_size else []
        if len(oof_rows) != 7200:
            failures.append(f"Brain {backbone} OOF rows: expected 7200, found {len(oof_rows)}")
        result["backbones"][backbone] = {
            "folds": fold_status,
            "folds_complete": sum(item["complete"] for item in fold_status),
            "oof_rows": len(oof_rows),
            "duplicate_groups_in_multiple_outer_folds": len(cross_outer),
            "unique_outer_images": len(outer_image_folds),
            "outer_images_in_multiple_folds": len(repeated_outer_images),
        }
    summary_path = BRAIN / "brain_mri_5fold_summary.json"
    check_file(summary_path, failures)
    if summary_path.exists():
        finite = non_finite_paths(read_json(summary_path))
        if finite:
            failures.append(f"Brain non-finite summary values: {finite}")
        result["non_finite_summary_values"] = finite
    return result


def verify_xray(failures: list[str]) -> dict[str, Any]:
    fold_status = []
    all_outer_patients: dict[str, set[int]] = {}
    all_outer_images: dict[str, set[int]] = {}
    for fold in range(1, 6):
        fold_dir = XRAY_FOLDS / f"fold_{fold}"
        checkpoint = ROOT / "checkpoints" / "model1" / "cross_validation" / "chest_xray" / "five_fold" / f"fold_{fold}_best.pt"
        required = [
            fold_dir / "fold_metrics.json",
            fold_dir / "training_history.csv",
            fold_dir / "train_manifest.csv",
            fold_dir / "validation_manifest.csv",
            fold_dir / "test_manifest.csv",
            fold_dir / "test_predictions.csv",
            fold_dir / "inner_validation_thresholds.csv",
            fold_dir / "label_distribution.csv",
            fold_dir / "failure_log.txt",
            checkpoint,
        ]
        for path in required:
            check_file(path, failures)
        if any(not path.exists() or path.stat().st_size == 0 for path in required):
            fold_status.append({"fold": fold, "complete": False})
            continue
        manifests = {
            "train": read_csv(fold_dir / "train_manifest.csv"),
            "validation": read_csv(fold_dir / "validation_manifest.csv"),
            "test": read_csv(fold_dir / "test_manifest.csv"),
        }
        patients = {name: {row["patient_id"] for row in rows} for name, rows in manifests.items()}
        overlap = {
            "train_validation": len(patients["train"] & patients["validation"]),
            "train_test": len(patients["train"] & patients["test"]),
            "validation_test": len(patients["validation"] & patients["test"]),
        }
        if any(overlap.values()):
            failures.append(f"X-ray fold {fold} patient overlap: {overlap}")
        for patient in patients["test"]:
            all_outer_patients.setdefault(patient, set()).add(fold)
        for row in manifests["test"]:
            all_outer_images.setdefault(row["image_name"], set()).add(fold)
        metrics = read_json(fold_dir / "fold_metrics.json")
        finite = non_finite_paths(metrics)
        if finite:
            failures.append(f"X-ray fold {fold} non-finite metric values: {finite}")
        distributions = read_csv(fold_dir / "label_distribution.csv")
        outer_support = {
            row["label"]: int(row["positive_images"])
            for row in distributions
            if row["split"] == "outer_test"
        }
        auc_rows = metrics["outer_test_inner_tuned"]["per_class_table"]
        invalid_fold_auc = [
            row["class_name"]
            for row in auc_rows
            if outer_support.get(row["class_name"], 0) > 0 and row["auroc"] is None
        ]
        if invalid_fold_auc:
            failures.append(f"X-ray fold {fold} missing AUROC with positive support: {invalid_fold_auc}")
        failure_text = (fold_dir / "failure_log.txt").read_text(encoding="utf-8")
        if "status=completed" not in failure_text or "error=none" not in failure_text:
            failures.append(f"X-ray fold {fold} failure log does not record clean completion")
        fold_status.append({"fold": fold, "complete": True, "patient_overlap": overlap})
    cross_outer = {patient: sorted(folds) for patient, folds in all_outer_patients.items() if len(folds) > 1}
    if cross_outer:
        failures.append(f"X-ray patients assigned to multiple outer folds: {len(cross_outer)}")
    repeated_outer_images = {image: sorted(folds) for image, folds in all_outer_images.items() if len(folds) > 1}
    if len(all_outer_patients) != 30805 or len(all_outer_images) != 112120 or repeated_outer_images:
        failures.append(
            "X-ray outer coverage mismatch: "
            f"patients={len(all_outer_patients)}, images={len(all_outer_images)}, "
            f"repeated_images={len(repeated_outer_images)}"
        )
    oof_path = XRAY / "xray_out_of_fold_predictions.csv"
    check_file(oof_path, failures)
    oof_rows = read_csv(oof_path) if oof_path.exists() and oof_path.stat().st_size else []
    if len(oof_rows) != 112120:
        failures.append(f"X-ray OOF rows: expected 112120, found {len(oof_rows)}")
    if oof_rows and len({row["image_name"] for row in oof_rows}) != 112120:
        failures.append("X-ray OOF image identifiers are not unique")
    per_label_path = XRAY / "xray_per_label_out_of_fold_metrics.csv"
    check_file(per_label_path, failures)
    invalid_auroc = []
    if per_label_path.exists() and per_label_path.stat().st_size:
        for row in read_csv(per_label_path):
            support = int(row["positive_support"])
            auroc = row["pooled_auroc"].strip()
            if support == 0 and auroc:
                invalid_auroc.append(row["label"])
            if support > 0 and not auroc:
                invalid_auroc.append(row["label"])
    if invalid_auroc:
        failures.append(f"X-ray invalid support/AUROC rows: {invalid_auroc}")
    summary_path = XRAY / "xray_cross_validation_summary.json"
    check_file(summary_path, failures)
    finite = non_finite_paths(read_json(summary_path)) if summary_path.exists() else []
    if finite:
        failures.append(f"X-ray non-finite summary values: {finite}")
    return {
        "folds": fold_status,
        "folds_complete": sum(item["complete"] for item in fold_status),
        "oof_rows": len(oof_rows),
        "patients_in_multiple_outer_folds": len(cross_outer),
        "unique_outer_patients": len(all_outer_patients),
        "unique_outer_images": len(all_outer_images),
        "outer_images_in_multiple_folds": len(repeated_outer_images),
        "invalid_support_auroc_rows": invalid_auroc,
        "non_finite_summary_values": finite,
    }


def verify_cross_modal(failures: list[str]) -> dict[str, Any]:
    summary_path = CROSS_MODAL / "cross_modal_validation_summary.json"
    manifest_path = CROSS_MODAL / "provenance_manifest.csv"
    errors_path = CROSS_MODAL / "cross_modal_case_errors.csv"
    for path in (summary_path, manifest_path, errors_path):
        check_file(path, failures)
    summary = read_json(summary_path) if summary_path.exists() else {}
    manifest = read_csv(manifest_path) if manifest_path.exists() else []
    errors = read_csv(errors_path) if errors_path.exists() else []
    case_files = list((CROSS_MODAL / "cases").glob("*.json"))
    counts = (summary.get("requested_cases"), summary.get("completed_cases"), summary.get("failed_cases"))
    if counts != (100, 100, 0) or len(manifest) != 100 or len(case_files) != 100 or errors:
        failures.append(
            f"Cross-modal completeness mismatch: summary={counts}, manifest={len(manifest)}, cases={len(case_files)}, errors={len(errors)}"
        )
    return {"summary_counts": counts, "manifest_rows": len(manifest), "case_files": len(case_files), "error_rows": len(errors)}


def verify_ablation(failures: list[str]) -> dict[str, Any]:
    summary_path = ABLATION / "model3_ablation_summary.json"
    case_path = ABLATION / "model3_ablation_case_results.csv"
    check_file(summary_path, failures)
    check_file(case_path, failures)
    summary = read_json(summary_path) if summary_path.exists() else {}
    rows = read_csv(case_path) if case_path.exists() else []
    variants = summary.get("variants", {})
    expected = {"late_fusion_no_rag", "stable_late_fusion_with_rag"}
    if set(variants) != expected or len(rows) != 200:
        failures.append(f"Model-3 ablation mismatch: variants={sorted(variants)}, case rows={len(rows)}")
    for name in expected:
        item = variants.get(name, {})
        if (item.get("requested_cases"), item.get("completed_cases"), item.get("failed_cases")) != (100, 100, 0):
            failures.append(f"Model-3 ablation incomplete for {name}: {item}")
    finite = non_finite_paths(summary)
    if finite:
        failures.append(f"Model-3 non-finite summary values: {finite}")
    return {"variants": sorted(variants), "case_rows": len(rows), "non_finite_summary_values": finite}


def verify_comparison_and_literature(failures: list[str]) -> dict[str, Any]:
    prediction_dir = COMPARISON / "model1_model_comparison_predictions"
    comparison_counts: dict[str, int] = {}
    for model in ("densenet121", "resnet50"):
        path = prediction_dir / f"brain_mri_{model}_out_of_fold_predictions.csv"
        check_file(path, failures)
        rows = read_csv(path) if path.exists() and path.stat().st_size else []
        comparison_counts[model] = len(rows)
        if len(rows) != 7200:
            failures.append(f"Comparison {model} paired prediction rows: expected 7200, found {len(rows)}")
    literature_json = OUT / "literature_benchmark_sources.json"
    literature_csv = OUT / "literature_benchmark_sources.csv"
    source_manifest = OUT / "literature_source_file_manifest.csv"
    for path in (literature_json, literature_csv, source_manifest):
        check_file(path, failures)
    literature = read_json(literature_json) if literature_json.exists() else {"papers": []}
    records = read_csv(literature_csv) if literature_csv.exists() else []
    sources = read_csv(source_manifest) if source_manifest.exists() else []
    expected_keys = {
        "Rasheed2025DenseNetBrain",
        "Disci2025BrainTumorTransfer",
        "Mao2025DilatedSEDenseNet",
        "Wang2017ChestXray8",
        "Baltruschat2019ChestXray",
        "Kufel2023ChestXrayTransfer",
    }
    keys = {paper["citation_key"] for paper in literature.get("papers", [])}
    if keys != expected_keys or len(records) != 13 or len(sources) != 6:
        failures.append(
            f"Literature provenance mismatch: keys={sorted(keys)}, records={len(records)}, sources={len(sources)}"
        )
    return {
        "paired_prediction_rows": comparison_counts,
        "required_paper_keys": sorted(keys),
        "benchmark_records": len(records),
        "local_source_records": len(sources),
    }


def main() -> None:
    failures: list[str] = []
    payload = {
        "brain_mri": verify_brain(failures),
        "chest_xray": verify_xray(failures),
        "cross_modal": verify_cross_modal(failures),
        "model3_ablation": verify_ablation(failures),
        "comparison_and_literature": verify_comparison_and_literature(failures),
    }
    payload["failures"] = failures
    payload["pass"] = not failures
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "final_technical_output_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Final Technical Output Audit",
        "",
        f"- Overall pass: `{not failures}`",
        f"- Brain MRI folds: {payload['brain_mri']['backbones']['densenet121']['folds_complete']}/5 DenseNet-121 and {payload['brain_mri']['backbones']['resnet50']['folds_complete']}/5 ResNet-50",
        f"- Chest X-ray folds: {payload['chest_xray']['folds_complete']}/5",
        f"- Cross-modal requested/completed/failed: {'/'.join(str(value) for value in payload['cross_modal']['summary_counts'])}",
        f"- Model-3 ablation case rows: {payload['model3_ablation']['case_rows']}",
        "",
        "## Failures",
        "",
        *(f"- {failure}" for failure in failures),
    ]
    if not failures:
        lines.append("- None")
    (OUT / "final_technical_output_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
