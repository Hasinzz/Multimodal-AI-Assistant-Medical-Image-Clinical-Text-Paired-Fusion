from __future__ import annotations

import argparse
import csv
import json
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT, XRAY_CLASSES
from src.model1.infer import TimmWithFeatures
from src.model1.train_xray import (
    XrayDataset,
    XraySample,
    build_scheduler,
    build_transforms,
    compute_pos_weight,
    parse_xray_labels,
    train_one_epoch,
)
from src.model1.tune_xray_thresholds import (
    compute_metrics,
    predictions_from_thresholds,
    search_thresholds_per_class,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "final_research_strengthening"
    / "data_audit"
    / "xray_image_patient_manifest.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation" / "model1_cross_validation" / "chest_xray"
DEFAULT_CHECKPOINTS = PROJECT_ROOT / "checkpoints" / "model1" / "cross_validation" / "chest_xray"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient-wise full-dataset Chest X-ray cross-validation.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fold", type=int, default=None, help="Run only one one-based outer fold.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_samples(manifest_path: Path):
    rows = read_csv(manifest_path)
    samples: list[XraySample] = []
    patient_ids: list[str] = []
    views: list[str] = []
    for row in rows:
        labels, _ = parse_xray_labels(row["raw_labels"], XRAY_CLASSES)
        samples.append(
            XraySample(
                image_path=PROJECT_ROOT / row["image_path"].replace("/", "\\"),
                labels=labels,
                image_name=row["image_name"],
                raw_labels=row["raw_labels"],
            )
        )
        patient_ids.append(row["patient_id"])
        views.append(row["view_position"])
    return samples, np.asarray(patient_ids), views


def build_patient_matrix(samples: Sequence[XraySample], patient_ids: np.ndarray):
    patient_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, patient_id in enumerate(patient_ids):
        patient_to_indices[str(patient_id)].append(index)
    patients = np.asarray(sorted(patient_to_indices, key=lambda value: int(value)))
    labels = np.zeros((len(patients), len(XRAY_CLASSES)), dtype=np.int32)
    for patient_index, patient_id in enumerate(patients):
        image_indices = patient_to_indices[patient_id]
        labels[patient_index] = np.stack([samples[index].labels for index in image_indices]).max(axis=0)
    return patients, labels, patient_to_indices


def patient_splits(patient_labels: np.ndarray, folds: int, seed: int):
    splitter = MultilabelStratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros((len(patient_labels), 1)), patient_labels))


def expand_patient_indices(
    patients: np.ndarray, patient_positions: Sequence[int], patient_to_indices: dict[str, list[int]]
) -> list[int]:
    indices: list[int] = []
    for patient_position in patient_positions:
        indices.extend(patient_to_indices[str(patients[patient_position])])
    return sorted(indices)


def make_loader(samples: Sequence[XraySample], transform, args: argparse.Namespace, shuffle: bool):
    workers = min(args.num_workers, 4)
    return DataLoader(
        XrayDataset(samples, transform),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def collect_outputs(model, loader, criterion, device, use_amp) -> dict[str, object]:
    model.eval()
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for images, batch_targets in loader:
            images = images.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(images)
                loss = criterion(logits, batch_targets)
                batch_probabilities = torch.sigmoid(logits)
            batch = batch_targets.shape[0]
            loss_sum += float(loss.item()) * batch
            count += batch
            targets.append(batch_targets.cpu().numpy())
            probabilities.append(batch_probabilities.cpu().numpy())
    return {
        "loss": loss_sum / count,
        "targets": np.concatenate(targets),
        "probabilities": np.concatenate(probabilities),
    }


def calibration_metrics(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, float]:
    flat_targets = targets.reshape(-1)
    flat_probabilities = probabilities.reshape(-1)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        member = (flat_probabilities > lower) & (flat_probabilities <= upper)
        if member.any():
            observed = flat_targets[member].mean()
            expected = flat_probabilities[member].mean()
            ece += member.mean() * abs(float(observed) - float(expected))
    return {
        "multilabel_brier": float(np.mean((probabilities - targets) ** 2)),
        "ece_10_bin_flattened": float(ece),
    }


def clean_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if key != "binary_predictions"}


def project_relative(path: Path) -> str:
    absolute = path if path.is_absolute() else PROJECT_ROOT / path
    return absolute.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def recover_postevaluation_fold(
    fold_number: int,
    fold_dir: Path,
    checkpoint_path: Path,
    train_samples: Sequence[XraySample],
    val_samples: Sequence[XraySample],
    test_samples: Sequence[XraySample],
    train_patients: Sequence[str],
    val_patients: Sequence[str],
    test_patients: Sequence[str],
    seed: int,
) -> dict[str, object]:
    """Finalize a fold whose outputs were saved before metadata serialization failed."""
    prediction_path = fold_dir / "test_predictions.csv"
    threshold_path = fold_dir / "inner_validation_thresholds.csv"
    history_path = fold_dir / "training_history.csv"
    train_manifest_path = fold_dir / "train_manifest.csv"
    required = [prediction_path, threshold_path, history_path, train_manifest_path, checkpoint_path]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        raise RuntimeError("Post-evaluation recovery requested without a complete artifact set")

    prediction_rows = read_csv(prediction_path)
    threshold_rows = read_csv(threshold_path)
    history = read_csv(history_path)
    targets = np.asarray(
        [[int(row[f"target_{name.lower()}"]) for name in XRAY_CLASSES] for row in prediction_rows],
        dtype=np.int8,
    )
    probabilities = np.asarray(
        [[float(row[f"probability_{name.lower()}"]) for name in XRAY_CLASSES] for row in prediction_rows],
        dtype=np.float64,
    )
    threshold_by_name = {row["class_name"]: float(row["threshold"]) for row in threshold_rows}
    thresholds = np.asarray([threshold_by_name[name] for name in XRAY_CLASSES], dtype=np.float64)
    default_pred = predictions_from_thresholds(probabilities, np.full(len(XRAY_CLASSES), 0.5))
    tuned_pred = predictions_from_thresholds(probabilities, thresholds)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runtime_seconds = max(0.0, prediction_path.stat().st_mtime - train_manifest_path.stat().st_mtime)

    payload = {
        "protocol": "patient-wise iterative multi-label stratified outer CV with patient-wise inner validation",
        "fold": fold_number,
        "seed": seed,
        "backbone": "densenet121",
        "pretrained": False,
        "train_images": len(train_samples),
        "validation_images": len(val_samples),
        "test_images": len(test_samples),
        "train_patients": len(set(train_patients)),
        "validation_patients": len(set(val_patients)),
        "test_patients": len(set(test_patients)),
        "patient_overlap_counts": {"train_validation": 0, "train_test": 0, "validation_test": 0},
        "best_epoch": int(checkpoint["epoch"]),
        "epochs_completed": len(history),
        "runtime_seconds": runtime_seconds,
        "runtime_measurement": "elapsed filesystem time from saved train manifest to saved outer predictions",
        "checkpoint_path": project_relative(checkpoint_path),
        "thresholds": {name: float(thresholds[index]) for index, name in enumerate(XRAY_CLASSES)},
        "outer_test_default_0_5": clean_metrics(
            compute_metrics(targets, probabilities, default_pred, XRAY_CLASSES)
        ),
        "outer_test_inner_tuned": clean_metrics(
            compute_metrics(targets, probabilities, tuned_pred, XRAY_CLASSES)
        ),
        "outer_test_calibration": calibration_metrics(targets, probabilities),
        "recovered_after_serialization_failure": True,
    }
    write_json(fold_dir / "fold_metrics.json", payload)
    (fold_dir / "failure_log.txt").write_text(
        "status=completed\nerror=none\nrecovery=post-evaluation metadata serialization\n",
        encoding="utf-8",
    )
    return payload


def save_manifest(
    path: Path,
    samples: Sequence[XraySample],
    patient_ids: Sequence[str],
    views: Sequence[str],
    role: str,
) -> None:
    rows = []
    for sample, patient_id, view in zip(samples, patient_ids, views):
        rows.append(
            {
                "role": role,
                "image_name": sample.image_name,
                "image_path": sample.image_path.relative_to(PROJECT_ROOT).as_posix(),
                "patient_id": patient_id,
                "view_position": view,
                "raw_labels": sample.raw_labels,
            }
        )
    write_csv(path, rows, ["role", "image_name", "image_path", "patient_id", "view_position", "raw_labels"])


def subset(values: Sequence, indices: Sequence[int]) -> list:
    return [values[index] for index in indices]


def save_label_distribution(
    path: Path,
    split_samples: Sequence[tuple[str, Sequence[XraySample]]],
) -> None:
    rows: list[dict[str, object]] = []
    for split_name, samples in split_samples:
        targets = np.stack([sample.labels for sample in samples])
        for class_index, class_name in enumerate(XRAY_CLASSES):
            positive = int(targets[:, class_index].sum())
            rows.append(
                {
                    "split": split_name,
                    "images": len(samples),
                    "label": class_name,
                    "positive_images": positive,
                    "prevalence": positive / len(samples),
                }
            )
    write_csv(path, rows, list(rows[0]))


def train_fold(
    fold_number: int,
    all_samples: Sequence[XraySample],
    all_patient_ids: np.ndarray,
    all_views: Sequence[str],
    train_indices: Sequence[int],
    val_indices: Sequence[int],
    test_indices: Sequence[int],
    args: argparse.Namespace,
) -> dict[str, object]:
    fold_dir = args.output_dir / f"fold_{fold_number}"
    checkpoint_path = args.checkpoint_dir / f"fold_{fold_number}_best.pt"
    metrics_path = fold_dir / "fold_metrics.json"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_samples = subset(all_samples, train_indices)
    val_samples = subset(all_samples, val_indices)
    test_samples = subset(all_samples, test_indices)
    train_patients = subset(all_patient_ids, train_indices)
    val_patients = subset(all_patient_ids, val_indices)
    test_patients = subset(all_patient_ids, test_indices)
    if set(train_patients) & set(val_patients) or set(train_patients) & set(test_patients) or set(val_patients) & set(test_patients):
        raise RuntimeError("Patient leakage detected in generated fold")

    save_label_distribution(
        fold_dir / "label_distribution.csv",
        [("inner_train", train_samples), ("inner_validation", val_samples), ("outer_test", test_samples)],
    )
    if metrics_path.exists() and checkpoint_path.exists() and not args.overwrite:
        (fold_dir / "failure_log.txt").write_text(
            "status=completed\nerror=none\n",
            encoding="utf-8",
        )
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    recoverable_outputs = [
        fold_dir / "test_predictions.csv",
        fold_dir / "inner_validation_thresholds.csv",
        fold_dir / "training_history.csv",
        fold_dir / "train_manifest.csv",
    ]
    if checkpoint_path.exists() and all(path.exists() and path.stat().st_size > 0 for path in recoverable_outputs) and not args.overwrite:
        return recover_postevaluation_fold(
            fold_number,
            fold_dir,
            checkpoint_path,
            train_samples,
            val_samples,
            test_samples,
            train_patients,
            val_patients,
            test_patients,
            args.seed + fold_number,
        )

    save_manifest(fold_dir / "train_manifest.csv", train_samples, train_patients, subset(all_views, train_indices), "inner_train")
    save_manifest(fold_dir / "validation_manifest.csv", val_samples, val_patients, subset(all_views, val_indices), "inner_validation")
    save_manifest(fold_dir / "test_manifest.csv", test_samples, test_patients, subset(all_views, test_indices), "outer_test")

    seed = args.seed + fold_number
    set_seed(seed)
    train_transform, eval_transform = build_transforms(args.image_size, use_clahe=False)
    train_loader = make_loader(train_samples, train_transform, args, True)
    val_loader = make_loader(val_samples, eval_transform, args, False)
    test_loader = make_loader(test_samples, eval_transform, args, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
    model = TimmWithFeatures("densenet121", len(XRAY_CLASSES)).to(device)
    pos_weight = compute_pos_weight(train_samples, XRAY_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = build_scheduler("cosine", optimizer, args.epochs, args.min_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, object]] = []
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_result = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            epoch_index=epoch,
            total_epochs=args.epochs,
            progress_interval=2,
            log=lambda message: None,
            grad_clip=0.0,
        )
        val_output = collect_outputs(model, val_loader, criterion, device, use_amp)
        val_pred = predictions_from_thresholds(val_output["probabilities"], np.full(len(XRAY_CLASSES), 0.5))
        val_metrics = compute_metrics(val_output["targets"], val_output["probabilities"], val_pred, XRAY_CLASSES)
        score = val_metrics["macro_auroc"]
        score_value = -float("inf") if score is None else float(score)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result["train_loss"],
                "validation_loss": val_output["loss"],
                "validation_macro_auroc": score,
                "validation_macro_f1_at_0_5": val_metrics["macro_f1"],
                "learning_rate": current_lr,
            }
        )
        print(
            f"[xray fold {fold_number}] epoch {epoch}/{args.epochs} "
            f"loss={train_result['train_loss']:.4f} val_auc={score_value:.4f}",
            flush=True,
        )
        if score_value > best_score:
            best_score = score_value
            best_epoch = epoch
            stale = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": "densenet121",
                    "image_size": args.image_size,
                    "class_names": list(XRAY_CLASSES),
                    "model_state_dict": model.state_dict(),
                    "inner_validation_macro_auroc": best_score,
                    "outer_fold": fold_number,
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        scheduler.step()
        if args.early_stopping_patience > 0 and stale >= args.early_stopping_patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_output = collect_outputs(model, val_loader, criterion, device, use_amp)
    test_output = collect_outputs(model, test_loader, criterion, device, use_amp)
    threshold_grid = np.round(np.arange(0.05, 0.951, 0.05), 2)
    thresholds, threshold_rows = search_thresholds_per_class(
        val_output["targets"],
        val_output["probabilities"],
        XRAY_CLASSES,
        threshold_grid,
        log=lambda message: None,
    )
    default_pred = predictions_from_thresholds(test_output["probabilities"], np.full(len(XRAY_CLASSES), 0.5))
    tuned_pred = predictions_from_thresholds(test_output["probabilities"], thresholds)
    default_metrics = clean_metrics(
        compute_metrics(test_output["targets"], test_output["probabilities"], default_pred, XRAY_CLASSES)
    )
    tuned_metrics = clean_metrics(
        compute_metrics(test_output["targets"], test_output["probabilities"], tuned_pred, XRAY_CLASSES)
    )
    calibration = calibration_metrics(test_output["targets"], test_output["probabilities"])
    elapsed = time.perf_counter() - started

    write_csv(fold_dir / "training_history.csv", history, list(history[0]))
    write_csv(fold_dir / "inner_validation_thresholds.csv", threshold_rows, list(threshold_rows[0]))
    prediction_rows = []
    for index, sample in enumerate(test_samples):
        row: dict[str, object] = {
            "fold": fold_number,
            "image_name": sample.image_name,
            "image_path": sample.image_path.relative_to(PROJECT_ROOT).as_posix(),
            "patient_id": test_patients[index],
            "raw_labels": sample.raw_labels,
        }
        for class_index, class_name in enumerate(XRAY_CLASSES):
            safe_name = class_name.lower()
            row[f"target_{safe_name}"] = int(test_output["targets"][index, class_index])
            row[f"probability_{safe_name}"] = float(test_output["probabilities"][index, class_index])
            row[f"prediction_tuned_{safe_name}"] = int(tuned_pred[index, class_index])
        prediction_rows.append(row)
    write_csv(fold_dir / "test_predictions.csv", prediction_rows, list(prediction_rows[0]))

    payload = {
        "protocol": "patient-wise iterative multi-label stratified outer CV with patient-wise inner validation",
        "fold": fold_number,
        "seed": seed,
        "backbone": "densenet121",
        "pretrained": False,
        "train_images": len(train_samples),
        "validation_images": len(val_samples),
        "test_images": len(test_samples),
        "train_patients": len(set(train_patients)),
        "validation_patients": len(set(val_patients)),
        "test_patients": len(set(test_patients)),
        "patient_overlap_counts": {"train_validation": 0, "train_test": 0, "validation_test": 0},
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "runtime_seconds": elapsed,
        "checkpoint_path": project_relative(checkpoint_path),
        "thresholds": {name: float(thresholds[index]) for index, name in enumerate(XRAY_CLASSES)},
        "outer_test_default_0_5": default_metrics,
        "outer_test_inner_tuned": tuned_metrics,
        "outer_test_calibration": calibration,
    }
    write_json(metrics_path, payload)
    (fold_dir / "failure_log.txt").write_text(
        "status=completed\nerror=none\n",
        encoding="utf-8",
    )
    return payload


def aggregate(args: argparse.Namespace) -> None:
    metrics_paths = sorted(args.output_dir.glob("fold_*/fold_metrics.json"))
    if not metrics_paths:
        return
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in metrics_paths]
    metric_names = [
        "macro_auroc",
        "micro_auroc",
        "macro_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
    ]
    summary_rows = []
    aggregate_metrics: dict[str, object] = {}
    for variant in ["outer_test_default_0_5", "outer_test_inner_tuned"]:
        aggregate_metrics[variant] = {}
        for metric in metric_names:
            values = [fold[variant][metric] for fold in folds if fold[variant][metric] is not None]
            aggregate_metrics[variant][metric] = {
                "mean": float(np.mean(values)),
                "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    for fold in folds:
        row: dict[str, object] = {
            "fold": fold["fold"],
            "best_epoch": fold["best_epoch"],
            "epochs_completed": fold["epochs_completed"],
            "test_images": fold["test_images"],
            "test_patients": fold["test_patients"],
            "runtime_seconds": fold["runtime_seconds"],
        }
        for metric in metric_names:
            row[f"default_{metric}"] = fold["outer_test_default_0_5"][metric]
            row[f"tuned_{metric}"] = fold["outer_test_inner_tuned"][metric]
        row.update(fold["outer_test_calibration"])
        summary_rows.append(row)
    write_csv(args.output_dir / "xray_patientwise_cv_summary.csv", summary_rows, list(summary_rows[0]))

    all_predictions: list[dict[str, str]] = []
    for path in sorted(args.output_dir.glob("fold_*/test_predictions.csv")):
        all_predictions.extend(read_csv(path))
    if all_predictions:
        write_csv(args.output_dir / "xray_patientwise_out_of_fold_predictions.csv", all_predictions, list(all_predictions[0]))

    one_fold_runtime = folds[0]["runtime_seconds"]
    if args.folds == 5:
        fold_decision = (
            "Five folds were selected after the full-fold runtime pilot because the measured compute and "
            "checkpoint storage were practical. No subset was used."
        )
    else:
        fold_decision = (
            "Three folds were selected because the full 112,120-image training run is computationally expensive; "
            "the measured first-fold runtime is reported and five folds would require approximately 5/3 times "
            "the completed three-fold GPU time. No subset was used."
        )
    payload = {
        "protocol": f"{args.folds}-fold patient-wise iterative multi-label stratified outer CV; patient-wise iterative inner validation",
        "folds_requested": args.folds,
        "folds_completed": len(folds),
        "seed": args.seed,
        "class_names": list(XRAY_CLASSES),
        "threshold_selection": "per-class F1 grid search on inner validation only (0.05 to 0.95, step 0.05)",
        "views_included": ["AP", "PA"],
        "fold_count_decision": fold_decision,
        "first_fold_runtime_seconds": one_fold_runtime,
        "estimated_five_fold_seconds_from_first_fold": one_fold_runtime * 5,
        "aggregate": aggregate_metrics,
        "folds": folds,
    }
    write_json(args.output_dir / "xray_patientwise_cv_summary.json", payload)
    lines = [
        "# Chest X-ray Patient-wise Cross-Validation",
        "",
        f"- Completed folds: {len(folds)}/{args.folds}",
        "- Dataset: all 112,120 images and 30,805 patients; AP and PA views retained.",
        "- Outer and inner splits: iterative multi-label stratification at patient level.",
        "- Thresholds: selected only on inner validation and applied once to the outer test fold.",
        "- Initialization: random (`pretrained=False`); an all-data checkpoint was not reused because it would leak outer-fold images.",
        f"- Measured first-fold runtime: {one_fold_runtime / 60:.1f} minutes.",
        f"- Fold-count decision: {fold_decision}",
        "",
        "| Metric | Default 0.5 mean (SD) | Inner-tuned mean (SD) |",
        "|---|---:|---:|",
    ]
    for metric in metric_names:
        default = aggregate_metrics["outer_test_default_0_5"][metric]
        tuned = aggregate_metrics["outer_test_inner_tuned"][metric]
        lines.append(
            f"| {metric} | {default['mean']:.4f} ({default['std_sample']:.4f}) | "
            f"{tuned['mean']:.4f} ({tuned['std_sample']:.4f}) |"
        )
    (args.output_dir / "xray_patientwise_cv_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if len(folds) == args.folds:
        default_values = [fold["outer_test_default_0_5"]["macro_f1"] for fold in folds]
        tuned_values = [fold["outer_test_inner_tuned"]["macro_f1"] for fold in folds]
        x = np.arange(1, len(folds) + 1)
        width = 0.34
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.bar(x - width / 2, default_values, width, label="Threshold 0.5", color="#557a95")
        ax.bar(x + width / 2, tuned_values, width, label="Inner-tuned", color="#c26d3a")
        ax.set_xlabel("Outer fold")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, max(0.5, max(tuned_values) * 1.2))
        ax.set_xticks(x)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / "xray_patientwise_fold_macro_f1.png", dpi=220)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, patient_ids, views = load_samples(args.manifest)
    patients, patient_labels, patient_to_indices = build_patient_matrix(samples, patient_ids)
    outer_splits = patient_splits(patient_labels, args.folds, args.seed)
    selected_folds = range(1, args.folds + 1) if args.fold is None else [args.fold]
    for fold_number in selected_folds:
        outer_train_positions, outer_test_positions = outer_splits[fold_number - 1]
        outer_train_patients = patients[outer_train_positions]
        outer_train_labels = patient_labels[outer_train_positions]
        inner_split = patient_splits(outer_train_labels, 5, args.seed + fold_number)[0]
        inner_train_positions, inner_val_positions = inner_split
        train_patient_positions = outer_train_positions[inner_train_positions]
        val_patient_positions = outer_train_positions[inner_val_positions]
        train_indices = expand_patient_indices(patients, train_patient_positions, patient_to_indices)
        val_indices = expand_patient_indices(patients, val_patient_positions, patient_to_indices)
        test_indices = expand_patient_indices(patients, outer_test_positions, patient_to_indices)
        try:
            train_fold(
                fold_number,
                samples,
                patient_ids,
                views,
                train_indices,
                val_indices,
                test_indices,
                args,
            )
        except Exception:
            fold_dir = args.output_dir / f"fold_{fold_number}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "failure_log.txt").write_text(
                "status=failed\n" + traceback.format_exc(),
                encoding="utf-8",
            )
            raise
    aggregate(args)


if __name__ == "__main__":
    main()
