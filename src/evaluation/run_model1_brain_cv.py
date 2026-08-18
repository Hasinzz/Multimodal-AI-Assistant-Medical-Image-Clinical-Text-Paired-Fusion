from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

from src.config import PROJECT_ROOT
from src.model1.infer import TimmWithFeatures
from src.model1.train_brain_mri import CLASS_FOLDER_TO_LABEL, build_transforms


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "final_research_strengthening"
    / "data_audit"
    / "brain_mri_image_manifest.csv"
)
DEFAULT_GROUPS = (
    PROJECT_ROOT
    / "outputs"
    / "final_research_strengthening"
    / "data_audit"
    / "brain_mri_duplicate_groups.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation" / "model1_cross_validation" / "brain_mri"
DEFAULT_CHECKPOINTS = PROJECT_ROOT / "checkpoints" / "model1" / "cross_validation" / "brain_mri"


@dataclass(frozen=True)
class BrainRecord:
    path: Path
    source_split: str
    label: str
    target: int
    duplicate_group: str


class BrainDataset(Dataset):
    def __init__(self, records: Sequence[BrainRecord], transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, record.target, str(record.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped five-fold Brain MRI evaluation.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--duplicate-groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--backbones", nargs="+", default=["densenet121", "resnet50"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=None, help="Run only one one-based outer fold.")
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


def load_records(manifest_path: Path, groups_path: Path) -> tuple[list[BrainRecord], list[str]]:
    duplicate_lookup = {
        row["path"].replace("\\", "/"): row["duplicate_group_id"] for row in read_csv(groups_path)
    }
    rows = read_csv(manifest_path)
    raw_labels = sorted(CLASS_FOLDER_TO_LABEL, key=lambda value: CLASS_FOLDER_TO_LABEL[value])
    class_names = [CLASS_FOLDER_TO_LABEL[label] for label in raw_labels]
    label_to_target = {label: index for index, label in enumerate(raw_labels)}
    records: list[BrainRecord] = []
    for row in rows:
        rel_path = row["path"].replace("\\", "/")
        path = PROJECT_ROOT / rel_path
        group = duplicate_lookup.get(rel_path, f"singleton_{row['record_index']}")
        records.append(
            BrainRecord(
                path=path,
                source_split=row["split"],
                label=CLASS_FOLDER_TO_LABEL[row["label"]],
                target=label_to_target[row["label"]],
                duplicate_group=group,
            )
        )
    return records, class_names


def make_loader(records: Sequence[BrainRecord], transform, batch_size: int, workers: int, shuffle: bool):
    return DataLoader(
        BrainDataset(records, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, _ = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch = targets.shape[0]
        loss_sum += float(loss.item()) * batch
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        count += batch
    return loss_sum / count, correct / count


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        member = (confidence > lower) & (confidence <= upper)
        if member.any():
            bin_accuracy = (predicted[member] == targets[member]).mean()
            ece += member.mean() * abs(float(bin_accuracy) - float(confidence[member].mean()))
    return float(ece)


def evaluate(model, loader, criterion, device, use_amp, class_count: int) -> dict[str, object]:
    model.eval()
    loss_sum = 0.0
    count = 0
    targets_all: list[int] = []
    probabilities_all: list[list[float]] = []
    paths_all: list[str] = []
    with torch.no_grad():
        for images, targets, paths in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, _ = model(images)
                loss = criterion(logits, targets)
            probabilities = torch.softmax(logits, dim=1)
            batch = targets.shape[0]
            loss_sum += float(loss.item()) * batch
            count += batch
            targets_all.extend(targets.cpu().tolist())
            probabilities_all.extend(probabilities.cpu().tolist())
            paths_all.extend(paths)

    targets_np = np.asarray(targets_all, dtype=np.int64)
    probabilities_np = np.asarray(probabilities_all, dtype=np.float64)
    predictions_np = probabilities_np.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, predictions_np, labels=np.arange(class_count), zero_division=0
    )
    one_hot = np.eye(class_count, dtype=np.float64)[targets_np]
    try:
        macro_auroc = float(roc_auc_score(one_hot, probabilities_np, average="macro", multi_class="ovr"))
    except ValueError:
        macro_auroc = None
    return {
        "loss": loss_sum / count,
        "accuracy": float(accuracy_score(targets_np, predictions_np)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "macro_auroc_ovr": macro_auroc,
        "multiclass_brier": float(np.mean(np.sum((probabilities_np - one_hot) ** 2, axis=1))),
        "ece_10_bin": expected_calibration_error(probabilities_np, targets_np),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": confusion_matrix(targets_np, predictions_np, labels=np.arange(class_count)).tolist(),
        "targets": targets_np.tolist(),
        "predictions": predictions_np.tolist(),
        "probabilities": probabilities_np.tolist(),
        "paths": paths_all,
    }


def split_indices(records: Sequence[BrainRecord], folds: int, seed: int):
    y = np.asarray([record.target for record in records])
    groups = np.asarray([record.duplicate_group for record in records])
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(records)), y, groups))


def assert_disjoint_groups(left: Sequence[BrainRecord], right: Sequence[BrainRecord], label: str) -> None:
    overlap = {record.duplicate_group for record in left} & {record.duplicate_group for record in right}
    if overlap:
        raise RuntimeError(f"Duplicate-group leakage in {label}: {len(overlap)} overlapping groups")


def save_manifest(path: Path, records: Sequence[BrainRecord], role: str) -> None:
    rows = [
        {
            "role": role,
            "path": record.path.relative_to(PROJECT_ROOT).as_posix(),
            "source_split": record.source_split,
            "label": record.label,
            "target": record.target,
            "duplicate_group": record.duplicate_group,
        }
        for record in records
    ]
    write_csv(path, rows, ["role", "path", "source_split", "label", "target", "duplicate_group"])


def strip_arrays(result: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in result.items() if key not in {"targets", "predictions", "probabilities", "paths"}}


def train_fold(
    backbone: str,
    fold_number: int,
    train_records: Sequence[BrainRecord],
    val_records: Sequence[BrainRecord],
    test_records: Sequence[BrainRecord],
    class_names: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    seed = args.seed + fold_number
    set_seed(seed)
    fold_dir = args.output_dir / backbone / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / backbone / f"fold_{fold_number}_best.pt"
    if (fold_dir / "fold_metrics.json").exists() and checkpoint_path.exists() and not args.overwrite:
        return json.loads((fold_dir / "fold_metrics.json").read_text(encoding="utf-8"))

    save_manifest(fold_dir / "train_manifest.csv", train_records, "inner_train")
    save_manifest(fold_dir / "validation_manifest.csv", val_records, "inner_validation")
    save_manifest(fold_dir / "test_manifest.csv", test_records, "outer_test")
    assert_disjoint_groups(train_records, val_records, "inner train/validation")
    assert_disjoint_groups(train_records, test_records, "inner train/outer test")
    assert_disjoint_groups(val_records, test_records, "inner validation/outer test")

    train_transform, eval_transform = build_transforms(args.image_size, use_n4=False)
    workers = min(args.num_workers, 4)
    train_loader = make_loader(train_records, train_transform, args.batch_size, workers, True)
    val_loader = make_loader(val_records, eval_transform, args.batch_size, workers, False)
    test_loader = make_loader(test_records, eval_transform, args.batch_size, workers, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
    model = TimmWithFeatures(backbone_name=backbone, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, object]] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        validation = evaluate(model, val_loader, criterion, device, use_amp, len(class_names))
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
        }
        history.append(row)
        print(
            f"[{backbone} fold {fold_number}] epoch {epoch}/{args.epochs} "
            f"train_acc={train_accuracy:.4f} val_acc={validation['accuracy']:.4f} "
            f"val_f1={validation['macro_f1']:.4f}",
            flush=True,
        )
        improved = validation["accuracy"] > best_accuracy or (
            np.isclose(validation["accuracy"], best_accuracy) and validation["loss"] < best_loss
        )
        if improved:
            best_accuracy = float(validation["accuracy"])
            best_loss = float(validation["loss"])
            best_epoch = epoch
            stale_epochs = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "backbone": backbone,
                    "image_size": args.image_size,
                    "class_names": list(class_names),
                    "model_state_dict": model.state_dict(),
                    "inner_validation_accuracy": best_accuracy,
                    "inner_validation_loss": best_loss,
                    "outer_fold": fold_number,
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation = evaluate(model, val_loader, criterion, device, use_amp, len(class_names))
    test = evaluate(model, test_loader, criterion, device, use_amp, len(class_names))
    elapsed = time.perf_counter() - started

    history_fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "validation_macro_f1",
    ]
    write_csv(fold_dir / "training_history.csv", history, history_fields)
    prediction_rows = []
    for index, path in enumerate(test["paths"]):
        row: dict[str, object] = {
            "fold": fold_number,
            "backbone": backbone,
            "path": Path(path).relative_to(PROJECT_ROOT).as_posix(),
            "true_target": test["targets"][index],
            "true_label": class_names[test["targets"][index]],
            "predicted_target": test["predictions"][index],
            "predicted_label": class_names[test["predictions"][index]],
            "correct": int(test["targets"][index] == test["predictions"][index]),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"probability_{class_name}"] = test["probabilities"][index][class_index]
        prediction_rows.append(row)
    prediction_fields = list(prediction_rows[0])
    write_csv(fold_dir / "test_predictions.csv", prediction_rows, prediction_fields)

    payload = {
        "protocol": "five-fold stratified grouped outer CV with grouped inner validation",
        "backbone": backbone,
        "pretrained": False,
        "fold": fold_number,
        "seed": seed,
        "train_samples": len(train_records),
        "validation_samples": len(val_records),
        "test_samples": len(test_records),
        "train_groups": len({record.duplicate_group for record in train_records}),
        "validation_groups": len({record.duplicate_group for record in val_records}),
        "test_groups": len({record.duplicate_group for record in test_records}),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "runtime_seconds": elapsed,
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "inner_validation": strip_arrays(validation),
        "outer_test": strip_arrays(test),
    }
    write_json(fold_dir / "fold_metrics.json", payload)
    return payload


def aggregate(backbone: str, args: argparse.Namespace, class_names: Sequence[str]) -> None:
    metrics_paths = sorted((args.output_dir / backbone).glob("fold_*/fold_metrics.json"))
    prediction_paths = sorted((args.output_dir / backbone).glob("fold_*/test_predictions.csv"))
    if not metrics_paths:
        return
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in metrics_paths]
    metric_names = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_auroc_ovr",
        "multiclass_brier",
        "ece_10_bin",
    ]
    summary_rows = []
    aggregate_metrics: dict[str, object] = {}
    for metric in metric_names:
        values = [fold["outer_test"][metric] for fold in folds if fold["outer_test"][metric] is not None]
        aggregate_metrics[metric] = {
            "mean": float(np.mean(values)),
            "std_sample": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
    for fold in folds:
        row = {
            "backbone": backbone,
            "fold": fold["fold"],
            "best_epoch": fold["best_epoch"],
            "epochs_completed": fold["epochs_completed"],
            "test_samples": fold["test_samples"],
            "runtime_seconds": fold["runtime_seconds"],
        }
        row.update({metric: fold["outer_test"][metric] for metric in metric_names})
        summary_rows.append(row)
    write_csv(args.output_dir / f"{backbone}_fold_summary.csv", summary_rows, list(summary_rows[0]))

    all_predictions: list[dict[str, str]] = []
    for path in prediction_paths:
        all_predictions.extend(read_csv(path))
    if all_predictions:
        write_csv(args.output_dir / f"{backbone}_out_of_fold_predictions.csv", all_predictions, list(all_predictions[0]))

    payload = {
        "protocol": "five-fold stratified grouped outer CV; first grouped fifth of each outer-training set used for inner validation",
        "backbone": backbone,
        "pretrained": False,
        "folds_requested": args.folds,
        "folds_completed": len(folds),
        "seed": args.seed,
        "class_names": list(class_names),
        "aggregate": aggregate_metrics,
        "folds": folds,
    }
    write_json(args.output_dir / f"{backbone}_5fold_summary.json", payload)
    lines = [
        f"# Brain MRI {backbone} Grouped Cross-Validation",
        "",
        f"- Completed folds: {len(folds)}/{args.folds}",
        "- Initialization: random (`pretrained=False`), matching the selected project implementation.",
        "- Outer split: stratified grouped five-fold cross-validation over all 7,200 images.",
        "- Group rule: every exact/near-duplicate connected component is confined to one fold.",
        "- Model selection: inner grouped validation only; outer folds are evaluated once.",
        "",
        "| Metric | Mean | SD |",
        "|---|---:|---:|",
    ]
    for metric in metric_names:
        values = aggregate_metrics[metric]
        lines.append(f"| {metric} | {values['mean']:.4f} | {values['std_sample']:.4f} |")
    (args.output_dir / f"{backbone}_5fold_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if len(folds) == args.folds:
        values = [fold["outer_test"]["macro_f1"] for fold in folds]
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        ax.bar(np.arange(1, len(values) + 1), values, color="#2f6f73")
        ax.axhline(np.mean(values), color="#a33b20", linestyle="--", label=f"Mean {np.mean(values):.3f}")
        ax.set_xlabel("Outer fold")
        ax.set_ylabel("Macro F1")
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(1, len(values) + 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output_dir / f"{backbone}_fold_macro_f1.png", dpi=220)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, class_names = load_records(args.manifest, args.duplicate_groups)
    outer_splits = split_indices(records, args.folds, args.seed)
    selected_folds = range(1, args.folds + 1) if args.fold is None else [args.fold]
    for backbone in args.backbones:
        for fold_number in selected_folds:
            outer_train_indices, test_indices = outer_splits[fold_number - 1]
            outer_train_records = [records[index] for index in outer_train_indices]
            test_records = [records[index] for index in test_indices]
            inner_splits = split_indices(outer_train_records, 5, args.seed + fold_number)
            train_indices, val_indices = inner_splits[0]
            train_records = [outer_train_records[index] for index in train_indices]
            val_records = [outer_train_records[index] for index in val_indices]
            train_fold(
                backbone,
                fold_number,
                train_records,
                val_records,
                test_records,
                class_names,
                args,
            )
        aggregate(backbone, args, class_names)


if __name__ == "__main__":
    main()
