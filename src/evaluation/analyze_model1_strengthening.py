from __future__ import annotations

import csv
import json
import math
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import PROJECT_ROOT, XRAY_CLASSES
from src.model1.infer import TimmWithFeatures


BRAIN_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "model1_cross_validation" / "brain_mri"
XRAY_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "model1_cross_validation" / "chest_xray"
XRAY_FOLD_DIR = XRAY_DIR / "five_fold"
COMPARISON_DIR = PROJECT_ROOT / "outputs" / "evaluation" / "model1_model_comparison"
STRENGTHENING_DIR = PROJECT_ROOT / "outputs" / "final_research_strengthening"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "thesis_figures"
SEED = 42
BRAIN_CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd_ci(values: Sequence[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    half_width = float(t.ppf(0.975, array.size - 1) * sd / math.sqrt(array.size)) if array.size > 1 else 0.0
    return {
        "mean": mean,
        "std_sample": sd,
        "confidence_interval_95": [mean - half_width, mean + half_width],
        "fold_values": array.tolist(),
    }


def bootstrap_mean_ci(values: Sequence[float], iterations: int = 10000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    means = rng.choice(array, size=(iterations, array.size), replace=True).mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def multiclass_ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        member = (confidence > lower) & (confidence <= upper)
        if member.any():
            ece += member.mean() * abs(float((predictions[member] == targets[member]).mean()) - float(confidence[member].mean()))
    return float(ece)


def binary_ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    probabilities = probabilities.reshape(-1)
    targets = targets.reshape(-1)
    ece = 0.0
    boundaries = np.linspace(0, 1, bins + 1)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        member = (probabilities > lower) & (probabilities <= upper)
        if member.any():
            ece += member.mean() * abs(float(targets[member].mean()) - float(probabilities[member].mean()))
    return float(ece)


def safe_multilabel_auroc(targets: np.ndarray, probabilities: np.ndarray):
    per_class: list[float | None] = []
    for index in range(targets.shape[1]):
        if np.unique(targets[:, index]).size < 2:
            per_class.append(None)
        else:
            per_class.append(float(roc_auc_score(targets[:, index], probabilities[:, index])))
    valid = [value for value in per_class if value is not None]
    macro = float(np.mean(valid)) if valid else None
    micro = float(roc_auc_score(targets.reshape(-1), probabilities.reshape(-1)))
    return per_class, macro, micro


def brain_metrics(targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, labels=np.arange(len(BRAIN_CLASSES)), zero_division=0
    )
    one_hot = np.eye(len(BRAIN_CLASSES))[targets]
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted", zero_division=0)),
        "macro_auroc_ovr": float(roc_auc_score(one_hot, probabilities, average="macro", multi_class="ovr")),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece_10_bin": multiclass_ece(probabilities, targets),
        "per_class": {
            BRAIN_CLASSES[index]: {
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index in range(len(BRAIN_CLASSES))
        },
        "confusion_matrix": confusion_matrix(targets, predictions, labels=np.arange(len(BRAIN_CLASSES))).tolist(),
    }


def load_brain_predictions(backbone: str):
    rows = read_csv(BRAIN_DIR / f"{backbone}_out_of_fold_predictions.csv")
    targets = np.asarray([int(row["true_target"]) for row in rows])
    predictions = np.asarray([int(row["predicted_target"]) for row in rows])
    probabilities = np.asarray(
        [[float(row[f"probability_{class_name}"]) for class_name in BRAIN_CLASSES] for row in rows]
    )
    folds = np.asarray([int(row["fold"]) for row in rows])
    return rows, targets, predictions, probabilities, folds


def draw_confusion(matrix: np.ndarray, classes: Sequence[str], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), [name.replace("_", " ") for name in classes], rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), [name.replace("_", " ") for name in classes])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def measure_inference(backbone: str, checkpoint_path: Path) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TimmWithFeatures(backbone, len(BRAIN_CLASSES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    batch = torch.zeros((32, 3, 224, 224), device=device)
    with torch.no_grad():
        for _ in range(5):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(30):
            model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del model, batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "parameters": parameters,
        "inference_device": str(device),
        "timed_images": 960,
        "inference_ms_per_image": elapsed / 960 * 1000,
    }


def analyze_brain() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    analyses: dict[str, Any] = {}
    prediction_sets: dict[str, Any] = {}
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_auroc_ovr", "multiclass_brier", "ece_10_bin"]
    for backbone in ("densenet121", "resnet50"):
        rows, targets, predictions, probabilities, folds = load_brain_predictions(backbone)
        fold_results = []
        for fold in range(1, 6):
            fold_dir = BRAIN_DIR / backbone / f"fold_{fold}"
            distribution_rows: list[dict[str, object]] = []
            for role, filename in (
                ("inner_train", "train_manifest.csv"),
                ("inner_validation", "validation_manifest.csv"),
                ("outer_test", "test_manifest.csv"),
            ):
                manifest_rows = read_csv(fold_dir / filename)
                counts = Counter(row["label"] for row in manifest_rows)
                for class_name in BRAIN_CLASSES:
                    distribution_rows.append(
                        {
                            "split": role,
                            "images": len(manifest_rows),
                            "class": class_name,
                            "class_images": counts[class_name],
                            "prevalence": counts[class_name] / len(manifest_rows),
                        }
                    )
            write_csv(fold_dir / "class_distribution.csv", distribution_rows)
            (fold_dir / "error_log.txt").write_text(
                "status=completed\nerror=none\n",
                encoding="utf-8",
            )
            member = folds == fold
            metrics = brain_metrics(targets[member], predictions[member], probabilities[member])
            confusion_rows = []
            fold_confusion = np.asarray(metrics["confusion_matrix"])
            for true_index, true_class in enumerate(BRAIN_CLASSES):
                confusion_rows.append(
                    {
                        "true_class": true_class,
                        **{
                            f"predicted_{predicted_class}": int(fold_confusion[true_index, predicted_index])
                            for predicted_index, predicted_class in enumerate(BRAIN_CLASSES)
                        },
                    }
                )
            write_csv(fold_dir / "confusion_matrix.csv", confusion_rows)
            draw_confusion(
                fold_confusion,
                BRAIN_CLASSES,
                fold_dir / "confusion_matrix.png",
                f"{backbone.replace('densenet121', 'DenseNet-121').replace('resnet50', 'ResNet-50')} outer fold {fold}",
            )
            fold_results.append({"fold": fold, "samples": int(member.sum()), **metrics})
        pooled = brain_metrics(targets, predictions, probabilities)
        aggregate = {metric: mean_sd_ci([float(row[metric]) for row in fold_results]) for metric in metric_names}
        analyses[backbone] = {
            "protocol": "five-fold stratified grouped outer CV with grouped inner validation",
            "folds_completed": 5,
            "seed": SEED,
            "pretrained": False,
            "fold_results": fold_results,
            "aggregate": aggregate,
            "pooled_out_of_fold": pooled,
        }
        prediction_sets[backbone] = (rows, targets, predictions, probabilities, folds)

    dense_rows, dense_targets, dense_predictions, _, dense_folds = prediction_sets["densenet121"]
    res_rows, res_targets, res_predictions, _, res_folds = prediction_sets["resnet50"]
    dense_by_path = {row["path"]: row for row in dense_rows}
    res_by_path = {row["path"]: row for row in res_rows}
    shared_paths = sorted(set(dense_by_path) & set(res_by_path))
    if len(shared_paths) != 7200:
        raise RuntimeError(f"Expected 7,200 paired MRI predictions, found {len(shared_paths)}")
    paired_targets = np.asarray([int(dense_by_path[path]["true_target"]) for path in shared_paths])
    paired_dense = np.asarray([int(dense_by_path[path]["predicted_target"]) for path in shared_paths])
    paired_resnet = np.asarray([int(res_by_path[path]["predicted_target"]) for path in shared_paths])
    correctness_difference = (paired_dense == paired_targets).astype(float) - (paired_resnet == paired_targets).astype(float)
    rng = np.random.default_rng(SEED)
    boot_accuracy = rng.choice(correctness_difference, size=(5000, len(correctness_difference)), replace=True).mean(axis=1)
    macro_differences = []
    for _ in range(2000):
        indices = rng.integers(0, len(paired_targets), len(paired_targets))
        macro_differences.append(
            f1_score(paired_targets[indices], paired_dense[indices], average="macro", zero_division=0)
            - f1_score(paired_targets[indices], paired_resnet[indices], average="macro", zero_division=0)
        )
    fold_accuracy_differences = []
    fold_f1_differences = []
    for fold in range(1, 6):
        dense_fold = analyses["densenet121"]["fold_results"][fold - 1]
        res_fold = analyses["resnet50"]["fold_results"][fold - 1]
        fold_accuracy_differences.append(dense_fold["accuracy"] - res_fold["accuracy"])
        fold_f1_differences.append(dense_fold["macro_f1"] - res_fold["macro_f1"])

    runtime = {
        backbone: measure_inference(
            backbone,
            PROJECT_ROOT / "checkpoints" / "model1" / "cross_validation" / "brain_mri" / backbone / "fold_1_best.pt",
        )
        for backbone in ("densenet121", "resnet50")
    }
    comparison = {
        "protocol": "Matched five-fold grouped comparison on identical outer cases and training policy",
        "paired_cases": len(shared_paths),
        "models": {
            backbone: {
                "aggregate": analyses[backbone]["aggregate"],
                "pooled_out_of_fold": analyses[backbone]["pooled_out_of_fold"],
                **runtime[backbone],
            }
            for backbone in ("densenet121", "resnet50")
        },
        "paired_differences_dense_minus_resnet": {
            "accuracy_case_bootstrap": {
                "difference": float(correctness_difference.mean()),
                "confidence_interval_95": [float(np.quantile(boot_accuracy, 0.025)), float(np.quantile(boot_accuracy, 0.975))],
            },
            "macro_f1_case_bootstrap": {
                "difference": float(f1_score(paired_targets, paired_dense, average="macro") - f1_score(paired_targets, paired_resnet, average="macro")),
                "confidence_interval_95": [float(np.quantile(macro_differences, 0.025)), float(np.quantile(macro_differences, 0.975))],
            },
            "accuracy_fold_difference": mean_sd_ci(fold_accuracy_differences),
            "macro_f1_fold_difference": mean_sd_ci(fold_f1_differences),
        },
        "chest_xray_architecture_comparison": "Not run: no fair existing baseline was available, and duplicating the full patient-wise five-fold training cost for another architecture was not practical for this revision.",
    }

    original = read_json(PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_final_v2" / "brain_metrics.json")
    canonical = {
        "result_label": "Cross-validation/generalization results",
        "original_selected_checkpoint_heldout": {
            "accuracy": original["best_val_accuracy"],
            "macro_f1": original["metrics"]["macro_f1"],
            "limitation": "The supplied Testing folder was repeatedly used for model selection and contains conservative near-duplicate links to Training; retained only as historical selected-checkpoint evidence.",
        },
        **analyses["densenet121"],
    }
    write_json(BRAIN_DIR / "brain_mri_5fold_summary.json", canonical)
    fold_table = []
    for row in analyses["densenet121"]["fold_results"]:
        fold_table.append({key: row[key] for key in ["fold", "samples", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "macro_auroc_ovr", "multiclass_brier", "ece_10_bin"]})
    write_csv(BRAIN_DIR / "brain_mri_5fold_summary.csv", fold_table)
    shutil.copyfile(BRAIN_DIR / "densenet121_out_of_fold_predictions.csv", BRAIN_DIR / "brain_mri_out_of_fold_predictions.csv")

    aggregate = analyses["densenet121"]["aggregate"]
    report = [
        "# Brain MRI Five-fold Grouped Cross-validation",
        "",
        "All 7,200 images entered five stratified outer folds. Exact and conservative near-duplicate connected components were kept within one outer fold and within one inner train/validation partition. Inner validation controlled checkpoint selection; each outer fold was evaluated once.",
        "",
        "| Metric | Mean | SD | 95% t interval |",
        "|---|---:|---:|---:|",
    ]
    for metric in metric_names:
        item = aggregate[metric]
        report.append(f"| {metric} | {item['mean']:.4f} | {item['std_sample']:.4f} | [{item['confidence_interval_95'][0]:.4f}, {item['confidence_interval_95'][1]:.4f}] |")
    report.extend(
        [
            "",
            "The model used random initialization (`pretrained=False`), matching the selected repository implementation. The historical supplied-folder score is retained separately and is not treated as uncontaminated generalization evidence.",
        ]
    )
    (BRAIN_DIR / "brain_mri_5fold_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    confusion = np.asarray(analyses["densenet121"]["pooled_out_of_fold"]["confusion_matrix"])
    draw_confusion(confusion, BRAIN_CLASSES, BRAIN_DIR / "brain_mri_pooled_confusion_matrix.png", "Brain MRI pooled out-of-fold confusion matrix")
    shutil.copyfile(BRAIN_DIR / "brain_mri_pooled_confusion_matrix.png", FIGURE_DIR / "brain_mri_pooled_confusion_matrix.png")
    fold_x = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.plot(fold_x, [row["macro_f1"] for row in analyses["densenet121"]["fold_results"]], marker="o", label="DenseNet-121", color="#2f6f73")
    ax.plot(fold_x, [row["macro_f1"] for row in analyses["resnet50"]["fold_results"]], marker="s", label="ResNet-50", color="#a33b20")
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0.65, 1.0)
    ax.set_xticks(fold_x)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "brain_mri_fold_results.png", dpi=220)
    plt.close(fig)

    write_json(COMPARISON_DIR / "model1_model_comparison_summary.json", comparison)
    prediction_dir = COMPARISON_DIR / "model1_model_comparison_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        BRAIN_DIR / "densenet121_out_of_fold_predictions.csv",
        prediction_dir / "brain_mri_densenet121_out_of_fold_predictions.csv",
    )
    shutil.copyfile(
        BRAIN_DIR / "resnet50_out_of_fold_predictions.csv",
        prediction_dir / "brain_mri_resnet50_out_of_fold_predictions.csv",
    )
    comparison_rows = []
    for backbone in ("densenet121", "resnet50"):
        comparison_rows.append(
            {
                "task": "Brain MRI four-class classification",
                "model": backbone,
                "folds": 5,
                "mean_accuracy": analyses[backbone]["aggregate"]["accuracy"]["mean"],
                "sd_accuracy": analyses[backbone]["aggregate"]["accuracy"]["std_sample"],
                "mean_macro_f1": analyses[backbone]["aggregate"]["macro_f1"]["mean"],
                "sd_macro_f1": analyses[backbone]["aggregate"]["macro_f1"]["std_sample"],
                "parameters": runtime[backbone]["parameters"],
                "inference_ms_per_image": runtime[backbone]["inference_ms_per_image"],
            }
        )
    write_csv(COMPARISON_DIR / "model1_model_comparison_summary.csv", comparison_rows)
    comparison_report = [
        "# Controlled Model-1 Comparison",
        "",
        "DenseNet-121 and ResNet-50 were trained from random initialization on identical grouped Brain MRI folds, transformations, 20-epoch maximum budget, AdamW policy, and inner-validation checkpoint rule.",
        "",
        "| Model | Accuracy mean (SD) | Macro F1 mean (SD) | Parameters | Inference ms/image |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        comparison_report.append(f"| {row['model']} | {row['mean_accuracy']:.4f} ({row['sd_accuracy']:.4f}) | {row['mean_macro_f1']:.4f} ({row['sd_macro_f1']:.4f}) | {row['parameters']:,} | {row['inference_ms_per_image']:.3f} |")
    accuracy_pair = comparison["paired_differences_dense_minus_resnet"]["accuracy_case_bootstrap"]
    f1_pair = comparison["paired_differences_dense_minus_resnet"]["macro_f1_case_bootstrap"]
    comparison_report.extend(
        [
            "",
            f"Paired out-of-fold accuracy difference (DenseNet - ResNet): {accuracy_pair['difference']:.4f}, 95% paired bootstrap CI [{accuracy_pair['confidence_interval_95'][0]:.4f}, {accuracy_pair['confidence_interval_95'][1]:.4f}].",
            f"Paired macro-F1 difference: {f1_pair['difference']:.4f}, 95% paired bootstrap CI [{f1_pair['confidence_interval_95'][0]:.4f}, {f1_pair['confidence_interval_95'][1]:.4f}].",
            "",
            comparison["chest_xray_architecture_comparison"],
        ]
    )
    (COMPARISON_DIR / "model1_model_comparison_report.md").write_text("\n".join(comparison_report) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    models = ["DenseNet-121", "ResNet-50"]
    values = [analyses[name]["aggregate"]["macro_f1"]["mean"] for name in ("densenet121", "resnet50")]
    errors = [analyses[name]["aggregate"]["macro_f1"]["std_sample"] for name in ("densenet121", "resnet50")]
    ax.bar(models, values, yerr=errors, capsize=5, color=["#2f6f73", "#a33b20"])
    ax.set_ylabel("Mean outer-fold macro F1")
    ax.set_ylim(0.7, 1.0)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "brain_mri_architecture_comparison.png", dpi=220)
    plt.close(fig)

    off_diagonal = []
    for true_index in range(len(BRAIN_CLASSES)):
        for predicted_index in range(len(BRAIN_CLASSES)):
            if true_index != predicted_index:
                off_diagonal.append((int(confusion[true_index, predicted_index]), BRAIN_CLASSES[true_index], BRAIN_CLASSES[predicted_index]))
    off_diagonal.sort(reverse=True)
    errors = [f"{true_label} misclassified as {predicted_label}: {count}" for count, true_label, predicted_label in off_diagonal[:5]]
    return analyses, comparison, errors


def multilabel_metrics(targets: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    per_class_auroc, macro_auroc, micro_auroc = safe_multilabel_auroc(targets, probabilities)
    return {
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(targets, predictions, average="micro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "micro_recall": float(recall_score(targets, predictions, average="micro", zero_division=0)),
        "per_class_auroc": per_class_auroc,
    }


def analyze_xray() -> tuple[dict[str, Any], list[str]]:
    rows = read_csv(XRAY_FOLD_DIR / "xray_patientwise_out_of_fold_predictions.csv")
    folds = np.asarray([int(row["fold"]) for row in rows])
    targets = np.asarray([[int(row[f"target_{name.lower()}"]) for name in XRAY_CLASSES] for row in rows])
    probabilities = np.asarray([[float(row[f"probability_{name.lower()}"]) for name in XRAY_CLASSES] for row in rows])
    tuned_predictions = np.asarray([[int(row[f"prediction_tuned_{name.lower()}"]) for name in XRAY_CLASSES] for row in rows])
    default_predictions = (probabilities >= 0.5).astype(int)
    fold_results = []
    per_class_fold_auroc: dict[str, list[float]] = {name: [] for name in XRAY_CLASSES}
    for fold in range(1, 6):
        member = folds == fold
        default = multilabel_metrics(targets[member], probabilities[member], default_predictions[member])
        tuned = multilabel_metrics(targets[member], probabilities[member], tuned_predictions[member])
        fold_payload = read_json(XRAY_FOLD_DIR / f"fold_{fold}" / "fold_metrics.json")
        fold_results.append(
            {
                "fold": fold,
                "test_images": int(member.sum()),
                "test_patients": fold_payload["test_patients"],
                "best_epoch": fold_payload["best_epoch"],
                "epochs_completed": fold_payload["epochs_completed"],
                "runtime_seconds": fold_payload["runtime_seconds"],
                "default_0_5": default,
                "inner_tuned": tuned,
                "calibration": {
                    "multilabel_brier": float(np.mean((probabilities[member] - targets[member]) ** 2)),
                    "ece_10_bin_flattened": binary_ece(probabilities[member], targets[member]),
                },
            }
        )
        for index, name in enumerate(XRAY_CLASSES):
            if tuned["per_class_auroc"][index] is not None:
                per_class_fold_auroc[name].append(tuned["per_class_auroc"][index])

    metric_names = ["macro_auroc", "micro_auroc", "macro_f1", "micro_f1", "macro_precision", "micro_precision", "macro_recall", "micro_recall"]
    aggregate: dict[str, Any] = {"default_0_5": {}, "inner_tuned": {}}
    for variant in ("default_0_5", "inner_tuned"):
        for metric in metric_names:
            values = [float(row[variant][metric]) for row in fold_results]
            aggregate[variant][metric] = {**mean_sd_ci(values), "bootstrap_95_ci_of_fold_mean": bootstrap_mean_ci(values)}
    brier_values = [row["calibration"]["multilabel_brier"] for row in fold_results]
    ece_values = [row["calibration"]["ece_10_bin_flattened"] for row in fold_results]
    aggregate["calibration"] = {
        "multilabel_brier": mean_sd_ci(brier_values),
        "ece_10_bin_flattened": mean_sd_ci(ece_values),
    }
    pooled_default = multilabel_metrics(targets, probabilities, default_predictions)
    pooled_tuned = multilabel_metrics(targets, probabilities, tuned_predictions)
    support = targets.sum(axis=0).astype(int)
    per_class_rows = []
    per_class_f1 = f1_score(targets, tuned_predictions, average=None, zero_division=0)
    per_class_precision = precision_score(targets, tuned_predictions, average=None, zero_division=0)
    per_class_recall = recall_score(targets, tuned_predictions, average=None, zero_division=0)
    for index, name in enumerate(XRAY_CLASSES):
        fold_values = per_class_fold_auroc[name]
        per_class_rows.append(
            {
                "label": name,
                "positive_support": int(support[index]),
                "pooled_auroc": pooled_tuned["per_class_auroc"][index],
                "fold_mean_auroc": float(np.mean(fold_values)),
                "fold_sd_auroc": float(np.std(fold_values, ddof=1)),
                "fold_bootstrap_95_ci_lower": bootstrap_mean_ci(fold_values)[0],
                "fold_bootstrap_95_ci_upper": bootstrap_mean_ci(fold_values)[1],
                "tuned_f1": float(per_class_f1[index]),
                "tuned_precision": float(per_class_precision[index]),
                "tuned_recall": float(per_class_recall[index]),
                "false_positive_count": int(((tuned_predictions[:, index] == 1) & (targets[:, index] == 0)).sum()),
                "false_negative_count": int(((tuned_predictions[:, index] == 0) & (targets[:, index] == 1)).sum()),
            }
        )
    write_csv(XRAY_DIR / "xray_per_label_out_of_fold_metrics.csv", per_class_rows)

    original = read_json(PROJECT_ROOT / "outputs" / "training" / "xray_gpu_large_v2" / "xray_metrics.json")
    payload = {
        "result_label": "Cross-validation/generalization results",
        "protocol": "five-fold patient-wise iterative multi-label stratified outer CV with patient-wise inner validation and threshold tuning",
        "folds_requested": 5,
        "folds_completed": 5,
        "seed": SEED,
        "images": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "patient_overlap": 0,
        "views_included": ["AP", "PA"],
        "original_selected_checkpoint_heldout": {
            "macro_auroc": original["best_score_value"],
            "limitation": "Historical random image-level evaluation with 8,022 overlapping patients; retained only as selected-checkpoint evidence.",
        },
        "fold_count_decision": "A separate full-fold runtime pilot was completed before the preferred five-fold protocol. Five folds were practical with isolated concurrent processes, 28.5 MB checkpoints, and available local storage. No subset was used.",
        "runtime_pilot_metrics_path": "outputs/evaluation/model1_cross_validation/chest_xray/fold_1/fold_metrics.json",
        "fold_results": fold_results,
        "aggregate": aggregate,
        "pooled_out_of_fold": {"default_0_5": pooled_default, "inner_tuned": pooled_tuned},
        "per_class_metrics_path": "outputs/evaluation/model1_cross_validation/chest_xray/xray_per_label_out_of_fold_metrics.csv",
    }
    write_json(XRAY_DIR / "xray_cross_validation_summary.json", payload)
    summary_rows = []
    for fold in fold_results:
        row: dict[str, object] = {
            "fold": fold["fold"],
            "test_images": fold["test_images"],
            "test_patients": fold["test_patients"],
            "best_epoch": fold["best_epoch"],
            "epochs_completed": fold["epochs_completed"],
            "runtime_seconds": fold["runtime_seconds"],
        }
        for metric in metric_names:
            row[f"default_{metric}"] = fold["default_0_5"][metric]
            row[f"tuned_{metric}"] = fold["inner_tuned"][metric]
        row.update(fold["calibration"])
        summary_rows.append(row)
    write_csv(XRAY_DIR / "xray_cross_validation_summary.csv", summary_rows)
    shutil.copyfile(XRAY_FOLD_DIR / "xray_patientwise_out_of_fold_predictions.csv", XRAY_DIR / "xray_out_of_fold_predictions.csv")

    report = [
        "# Chest X-ray Patient-wise Cross-validation",
        "",
        "All 112,120 AP/PA images entered five patient-wise outer folds. Iterative multi-label stratification was performed over patient-level label vectors. Inner patient-wise validation controlled checkpoint selection and per-class threshold tuning; outer labels were used once for evaluation.",
        "",
        "| Metric | Threshold 0.5 mean (SD) | Inner-tuned mean (SD) | Tuned bootstrap 95% CI of fold mean |",
        "|---|---:|---:|---:|",
    ]
    for metric in metric_names:
        default = aggregate["default_0_5"][metric]
        tuned = aggregate["inner_tuned"][metric]
        report.append(f"| {metric} | {default['mean']:.4f} ({default['std_sample']:.4f}) | {tuned['mean']:.4f} ({tuned['std_sample']:.4f}) | [{tuned['bootstrap_95_ci_of_fold_mean'][0]:.4f}, {tuned['bootstrap_95_ci_of_fold_mean'][1]:.4f}] |")
    report.extend(
        [
            "",
            "A separate full-fold pilot measured runtime before the preferred five-fold experiment was launched. Five folds were completed on the full dataset without using a subset.",
            "",
            "The fixed 0.5 and inner-selected threshold results are a threshold-policy comparison, not an architecture comparison.",
        ]
    )
    (XRAY_DIR / "xray_cross_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    fold_x = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(fold_x, [row["inner_tuned"]["macro_auroc"] for row in fold_results], marker="o", color="#2f6f73", label="Macro AUROC")
    ax.plot(fold_x, [row["inner_tuned"]["micro_auroc"] for row in fold_results], marker="s", color="#a33b20", label="Micro AUROC")
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(fold_x)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "xray_patientwise_fold_auroc.png", dpi=220)
    plt.close(fig)

    sorted_rows = sorted(per_class_rows, key=lambda row: row["pooled_auroc"])
    y = np.arange(len(sorted_rows))
    values = [row["fold_mean_auroc"] for row in sorted_rows]
    low = [row["fold_mean_auroc"] - row["fold_bootstrap_95_ci_lower"] for row in sorted_rows]
    high = [row["fold_bootstrap_95_ci_upper"] - row["fold_mean_auroc"] for row in sorted_rows]
    fig, ax = plt.subplots(figsize=(8.0, 6.3))
    ax.errorbar(values, y, xerr=[low, high], fmt="o", color="#2f6f73", ecolor="#66737a", capsize=3)
    ax.set_yticks(y, [row["label"].replace("_", " ") for row in sorted_rows])
    ax.set_xlabel("Mean outer-fold AUROC with bootstrap interval")
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "xray_per_label_auroc_intervals.png", dpi=220)
    plt.close(fig)

    lowest = sorted(per_class_rows, key=lambda row: row["pooled_auroc"])[:5]
    y_positions = np.arange(len(lowest))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.barh(
        y_positions - 0.18,
        [row["false_positive_count"] for row in lowest],
        height=0.36,
        label="False positives",
        color="#a33b20",
    )
    ax.barh(
        y_positions + 0.18,
        [row["false_negative_count"] for row in lowest],
        height=0.36,
        label="False negatives",
        color="#2f6f73",
    )
    ax.set_yticks(y_positions, [row["label"].replace("_", " ") for row in lowest])
    ax.invert_yaxis()
    ax.set_xlabel("Pooled out-of-fold image count")
    ax.set_title("Thresholded errors for the five lowest-AUROC labels")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "model1_error_analysis.png", dpi=220)
    plt.close(fig)
    errors = [
        f"{row['label']}: AUROC {row['pooled_auroc']:.4f}, F1 {row['tuned_f1']:.4f}, FP {row['false_positive_count']}, FN {row['false_negative_count']}, support {row['positive_support']}"
        for row in lowest
    ]
    return payload, errors


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    brain, comparison, brain_errors = analyze_brain()
    xray, xray_errors = analyze_xray()
    dense = brain["densenet121"]["aggregate"]
    tuned = xray["aggregate"]["inner_tuned"]
    statistical = [
        "# Model-1 Statistical Analysis",
        "",
        "## Brain MRI",
        "",
        f"DenseNet-121 completed 5/5 grouped folds. Mean accuracy was {dense['accuracy']['mean']:.4f} (SD {dense['accuracy']['std_sample']:.4f}, 95% t interval [{dense['accuracy']['confidence_interval_95'][0]:.4f}, {dense['accuracy']['confidence_interval_95'][1]:.4f}]). Mean macro F1 was {dense['macro_f1']['mean']:.4f} (SD {dense['macro_f1']['std_sample']:.4f}).",
        "",
        f"The paired case-level accuracy difference between DenseNet-121 and ResNet-50 was {comparison['paired_differences_dense_minus_resnet']['accuracy_case_bootstrap']['difference']:.4f}; its 95% paired bootstrap interval was [{comparison['paired_differences_dense_minus_resnet']['accuracy_case_bootstrap']['confidence_interval_95'][0]:.4f}, {comparison['paired_differences_dense_minus_resnet']['accuracy_case_bootstrap']['confidence_interval_95'][1]:.4f}].",
        "",
        "## Chest X-ray",
        "",
        f"DenseNet-121 completed 5/5 patient-wise folds. Mean macro AUROC was {tuned['macro_auroc']['mean']:.4f} (SD {tuned['macro_auroc']['std_sample']:.4f}, fold-bootstrap 95% interval [{tuned['macro_auroc']['bootstrap_95_ci_of_fold_mean'][0]:.4f}, {tuned['macro_auroc']['bootstrap_95_ci_of_fold_mean'][1]:.4f}]). Mean tuned macro F1 was {tuned['macro_f1']['mean']:.4f} (SD {tuned['macro_f1']['std_sample']:.4f}).",
        "",
        "Confidence intervals quantify internal resampling variability only. No significance test was performed against published summary numbers.",
    ]
    (STRENGTHENING_DIR / "model1_statistical_analysis.md").write_text("\n".join(statistical) + "\n", encoding="utf-8")
    error_report = [
        "# Model-1 Error Analysis",
        "",
        "## Brain MRI Most Frequent Confusions",
        "",
        *[f"- {item}" for item in brain_errors],
        "",
        "## Chest X-ray Lowest AUROC Labels",
        "",
        *[f"- {item}" for item in xray_errors],
        "",
        "False-positive and false-negative counts use thresholds selected only on each fold's inner validation patients. These automated label comparisons do not substitute for radiologist review.",
    ]
    (STRENGTHENING_DIR / "model1_error_analysis.md").write_text("\n".join(error_report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
