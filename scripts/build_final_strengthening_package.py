from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DESTINATION = OUTPUTS / "final_submission_package"
STRENGTHENING = OUTPUTS / "final_research_strengthening"
BRAIN = OUTPUTS / "evaluation" / "model1_cross_validation" / "brain_mri"
XRAY = OUTPUTS / "evaluation" / "model1_cross_validation" / "chest_xray"
COMPARISON = OUTPUTS / "evaluation" / "model1_model_comparison"
CROSS_MODAL = OUTPUTS / "evaluation" / "cross_modal_validation_v2"
ABLATION = OUTPUTS / "evaluation" / "model3_ablation"


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Required package file is missing or empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Required package directory is missing: {source}")
    shutil.copytree(source, destination, dirs_exist_ok=True)


def copy_fold_evidence(source: Path, destination: Path, filenames: tuple[str, ...]) -> None:
    for fold_dir in sorted(source.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        for filename in filenames:
            copy_file(fold_dir / filename, destination / fold_dir.name / filename)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_readme() -> str:
    brain = read_json(BRAIN / "brain_mri_5fold_summary.json")
    xray = read_json(XRAY / "xray_cross_validation_summary.json")
    cross = read_json(CROSS_MODAL / "cross_modal_validation_summary.json")
    ablation = read_json(ABLATION / "model3_ablation_summary.json")
    brain_accuracy = brain["aggregate"]["accuracy"]
    brain_f1 = brain["aggregate"]["macro_f1"]
    xray_auc = xray["aggregate"]["inner_tuned"]["macro_auroc"]
    xray_f1 = xray["aggregate"]["inner_tuned"]["macro_f1"]
    rag = ablation["variants"]["stable_late_fusion_with_rag"]["mean_metrics"]
    return f"""# Final Thesis Submission Package

Generated from the verified repository evidence on 2026-08-03.

## Thesis-ready files

- `report_source/`: complete updated LaTeX source and figures.
- `final_compiled_thesis.pdf`: MiKTeX/Biber-compiled thesis.
- `FINAL_RESEARCH_STRENGTHENING_REPORT.md`: final evidence and readiness report.
- `model1_cross_validation/`: fold summaries, manifests, thresholds, diagnostics, and pooled out-of-fold records.
- `model1_model_comparison/`: matched Brain MRI DenseNet-121/ResNet-50 comparison.
- `cross_modal_validation_v2/`: technical execution report and source provenance.
- `model3_ablation/`: paired no-RAG versus stable-RAG technical ablation.
- `audits/`: leakage, literature, citation, language, administrative, runtime, and final technical checks.
- `logs/`: retained training, compilation, and smoke-test logs.

## Safe claims

- Brain MRI DenseNet-121 completed 5/5 duplicate-grouped outer folds: mean accuracy {brain_accuracy['mean']:.4f} (SD {brain_accuracy['std_sample']:.4f}) and mean macro F1 {brain_f1['mean']:.4f} (SD {brain_f1['std_sample']:.4f}).
- Chest X-ray DenseNet-121 completed 5/5 patient-wise outer folds on all 112,120 images: mean macro AUROC {xray_auc['mean']:.4f} (SD {xray_auc['std_sample']:.4f}) and inner-validation-tuned macro F1 {xray_f1['mean']:.4f} (SD {xray_f1['std_sample']:.4f}).
- The cross-modal v2 run requested/completed/failed {cross['requested_cases']}/{cross['completed_cases']}/{cross['failed_cases']} technical executions, with combined inputs labeled synthetic and unpaired.
- Stable RAG supplied evidence in {rag['evidence_availability']:.4f} of cases with a mean evidence count of {rag['evidence_count']:.2f}; ablation metrics are automated technical proxies.

## Claims to avoid

- Do not call execution success diagnostic accuracy or clinical validation.
- Do not claim doctor, clinician, or expert validation of generated text.
- Do not describe synthetic/unpaired combinations as same-patient multimodal validation.
- Do not claim trained supervised cross-modal attention; no paired training cohort or non-empty trained checkpoint exists.
- Do not claim superiority over published studies because protocols and metric definitions differ.

## Experiments completed

- Brain MRI duplicate audit, five-fold grouped DenseNet-121 validation, and matched five-fold ResNet-50 baseline.
- Chest X-ray patient-overlap audit and five-fold patient-wise DenseNet-121 validation with inner-validation threshold selection.
- Six-paper peer-reviewed contextual benchmark extraction.
- Seven-scenario, 100-case Model-3 pipeline execution with per-case provenance.
- Paired late-fusion no-RAG versus stable-RAG technical ablation on the same 100 cases.

## Experiments not completed

- External hospital or scanner validation.
- Clinician-rated summary or follow-up-note quality evaluation.
- Same-patient paired multimodal validation or supervised cross-modal-attention training.
- A fair Chest X-ray architecture baseline under the full five-fold patient-wise protocol.

## Manual confirmations

The title-page date, approval semester, degree wording, registered title, approval wording, and committee details require supervisor or department confirmation. See `audits/ADMINISTRATIVE_CONFIRMATION_REQUIRED.md`.
"""


def main() -> None:
    expected = (OUTPUTS / "final_submission_package").resolve()
    if DESTINATION.resolve() != expected or ROOT.resolve() not in DESTINATION.resolve().parents:
        raise RuntimeError(f"Unsafe package destination: {DESTINATION}")
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    DESTINATION.mkdir(parents=True)

    copy_tree(ROOT / "report_source", DESTINATION / "report_source")
    copy_file(ROOT / "report_source" / "main.pdf", DESTINATION / "final_compiled_thesis.pdf")
    copy_file(
        STRENGTHENING / "FINAL_RESEARCH_STRENGTHENING_REPORT.md",
        DESTINATION / "FINAL_RESEARCH_STRENGTHENING_REPORT.md",
    )

    audit_destination = DESTINATION / "audits"
    for path in sorted(STRENGTHENING.iterdir()):
        if path.is_file():
            copy_file(path, audit_destination / path.name)
    for name in ("data_audit", "literature_sources", "xray_runtime_pilot"):
        copy_tree(STRENGTHENING / name, audit_destination / name)

    brain_destination = DESTINATION / "model1_cross_validation" / "brain_mri"
    for filename in (
        "brain_mri_5fold_summary.json",
        "brain_mri_5fold_summary.csv",
        "brain_mri_5fold_report.md",
        "brain_mri_out_of_fold_predictions.csv",
        "brain_mri_pooled_confusion_matrix.png",
    ):
        copy_file(BRAIN / filename, brain_destination / filename)
    brain_fold_files = (
        "fold_metrics.json",
        "training_history.csv",
        "train_manifest.csv",
        "validation_manifest.csv",
        "test_manifest.csv",
        "class_distribution.csv",
        "confusion_matrix.csv",
        "confusion_matrix.png",
        "error_log.txt",
    )
    for backbone in ("densenet121", "resnet50"):
        copy_fold_evidence(BRAIN / backbone, brain_destination / backbone, brain_fold_files)

    xray_destination = DESTINATION / "model1_cross_validation" / "chest_xray"
    for filename in (
        "xray_cross_validation_summary.json",
        "xray_cross_validation_summary.csv",
        "xray_cross_validation_report.md",
        "xray_out_of_fold_predictions.csv",
        "xray_per_label_out_of_fold_metrics.csv",
    ):
        copy_file(XRAY / filename, xray_destination / filename)
    copy_fold_evidence(
        XRAY / "five_fold",
        xray_destination / "five_fold",
        (
            "fold_metrics.json",
            "training_history.csv",
            "train_manifest.csv",
            "validation_manifest.csv",
            "test_manifest.csv",
            "inner_validation_thresholds.csv",
            "label_distribution.csv",
            "failure_log.txt",
        ),
    )

    copy_tree(COMPARISON, DESTINATION / "model1_model_comparison")
    for filename in (
        "cross_modal_validation_report.md",
        "cross_modal_validation_summary.json",
        "cross_modal_validation_summary.csv",
        "provenance_manifest.csv",
        "provenance_manifest.json",
        "cross_modal_case_errors.csv",
    ):
        copy_file(CROSS_MODAL / filename, DESTINATION / "cross_modal_validation_v2" / filename)
    copy_tree(ABLATION, DESTINATION / "model3_ablation")
    copy_tree(STRENGTHENING / "logs", DESTINATION / "logs")

    figure_destination = DESTINATION / "strengthening_figures"
    for filename in (
        "brain_mri_fold_results.png",
        "brain_mri_pooled_confusion_matrix.png",
        "brain_mri_architecture_comparison.png",
        "xray_patientwise_fold_auroc.png",
        "xray_per_label_auroc_intervals.png",
        "xray_label_correlation_matrix.png",
        "model1_error_analysis.png",
        "literature_benchmark_context.png",
        "cross_modal_input_combination_validation.png",
    ):
        copy_file(OUTPUTS / "thesis_figures" / filename, figure_destination / filename)
    copy_file(ABLATION / "model3_rag_ablation.png", figure_destination / "model3_rag_ablation.png")

    (DESTINATION / "README.md").write_text(build_readme(), encoding="utf-8")
    print(DESTINATION)


if __name__ == "__main__":
    main()
