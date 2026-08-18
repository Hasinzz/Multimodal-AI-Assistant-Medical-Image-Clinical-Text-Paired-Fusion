from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from scipy.fft import dctn

from src.config import PROJECT_ROOT, XRAY_CLASSES


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_research_strengthening" / "data_audit"
BRAIN_ROOT = PROJECT_ROOT / "data" / "images" / "brain_mri"
XRAY_ROOT = PROJECT_ROOT / "data" / "images" / "xray"
XRAY_METADATA = PROJECT_ROOT / "data" / "structured" / "Data_Entry_2017.csv"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NEAR_DUPLICATE_DISTANCE = 4
SEED = 42


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def perceptual_hash(image: Image.Image) -> int:
    grayscale = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    coefficients = dctn(np.asarray(grayscale, dtype=np.float32), norm="ortho")[:8, :8]
    flattened = coefficients.flatten()
    threshold = float(np.median(flattened[1:]))
    bits = flattened > threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def audit_brain_mri() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    corrupt_rows: list[dict[str, Any]] = []

    for path in sorted(BRAIN_ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts = path.relative_to(BRAIN_ROOT).parts
        split = parts[0] if len(parts) > 0 else "unknown"
        label = parts[1] if len(parts) > 1 else "unknown"
        try:
            raw = path.read_bytes()
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                phash = perceptual_hash(image)
            records.append(
                {
                    "record_index": len(records),
                    "path": relative(path),
                    "split": split,
                    "label": label,
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "perceptual_hash": f"{phash:016x}",
                    "perceptual_hash_int": phash,
                }
            )
        except Exception as exc:  # pragma: no cover - records real corrupt-file failures
            corrupt_rows.append(
                {
                    "path": relative(path),
                    "split": split,
                    "label": label,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_csv(
        OUTPUT_DIR / "brain_mri_image_manifest.csv",
        ["record_index", "path", "split", "label", "width", "height", "mode", "sha256", "perceptual_hash"],
        records,
    )
    write_csv(
        OUTPUT_DIR / "brain_mri_corrupt_files.csv",
        ["path", "split", "label", "error"],
        corrupt_rows,
    )

    union_find = UnionFind(len(records))
    by_sha: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_sha[record["sha256"]].append(index)

    exact_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    exact_group_number = 0
    exact_cross_split_groups = 0
    for sha256, indices in sorted(by_sha.items()):
        if len(indices) < 2:
            continue
        exact_group_number += 1
        for other in indices[1:]:
            union_find.union(indices[0], other)
        splits = {records[index]["split"] for index in indices}
        labels = {records[index]["label"] for index in indices}
        cross_split = len(splits) > 1
        if cross_split:
            exact_cross_split_groups += 1
        for index in indices:
            record = records[index]
            row = {
                "exact_group_id": f"exact_{exact_group_number:04d}",
                "sha256": sha256,
                "group_size": len(indices),
                "cross_split": int(cross_split),
                "path": record["path"],
                "split": record["split"],
                "label": record["label"],
                "width": record["width"],
                "height": record["height"],
            }
            exact_rows.append(row)
            if len(labels) > 1:
                conflict_rows.append(row | {"labels_in_group": "|".join(sorted(labels))})

    write_csv(
        OUTPUT_DIR / "brain_mri_exact_duplicates.csv",
        ["exact_group_id", "sha256", "group_size", "cross_split", "path", "split", "label", "width", "height"],
        exact_rows,
    )
    write_csv(
        OUTPUT_DIR / "brain_mri_label_conflicts.csv",
        [
            "exact_group_id",
            "sha256",
            "group_size",
            "cross_split",
            "path",
            "split",
            "label",
            "labels_in_group",
        ],
        conflict_rows,
    )

    hashes = np.asarray([record["perceptual_hash_int"] for record in records], dtype=np.uint64)
    near_rows: list[dict[str, Any]] = []
    near_cross_split_pairs = 0
    pair_number = 0
    for left in range(len(records) - 1):
        distances = np.bitwise_count(np.bitwise_xor(hashes[left], hashes[left + 1 :]))
        candidate_offsets = np.flatnonzero(distances <= NEAR_DUPLICATE_DISTANCE)
        for offset in candidate_offsets.tolist():
            right = left + 1 + offset
            if records[left]["sha256"] == records[right]["sha256"]:
                continue
            distance = int(distances[offset])
            union_find.union(left, right)
            pair_number += 1
            cross_split = records[left]["split"] != records[right]["split"]
            if cross_split:
                near_cross_split_pairs += 1
            near_rows.append(
                {
                    "near_pair_id": f"near_{pair_number:06d}",
                    "phash_distance": distance,
                    "cross_split": int(cross_split),
                    "path_a": records[left]["path"],
                    "split_a": records[left]["split"],
                    "label_a": records[left]["label"],
                    "path_b": records[right]["path"],
                    "split_b": records[right]["split"],
                    "label_b": records[right]["label"],
                }
            )

    write_csv(
        OUTPUT_DIR / "brain_mri_near_duplicates.csv",
        [
            "near_pair_id",
            "phash_distance",
            "cross_split",
            "path_a",
            "split_a",
            "label_a",
            "path_b",
            "split_b",
            "label_b",
        ],
        near_rows,
    )

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[union_find.find(index)].append(index)

    duplicate_groups = [indices for indices in grouped.values() if len(indices) > 1]
    duplicate_groups.sort(key=lambda values: min(records[index]["path"] for index in values))
    group_id_by_index: dict[int, str] = {}
    group_rows: list[dict[str, Any]] = []
    for group_number, indices in enumerate(duplicate_groups, start=1):
        group_id = f"dupgrp_{group_number:04d}"
        splits = sorted({records[index]["split"] for index in indices})
        labels = sorted({records[index]["label"] for index in indices})
        for index in indices:
            group_id_by_index[index] = group_id
            record = records[index]
            group_rows.append(
                {
                    "duplicate_group_id": group_id,
                    "group_size": len(indices),
                    "cross_split": int(len(splits) > 1),
                    "labels_in_group": "|".join(labels),
                    "path": record["path"],
                    "split": record["split"],
                    "label": record["label"],
                    "sha256": record["sha256"],
                    "perceptual_hash": record["perceptual_hash"],
                }
            )

    write_csv(
        OUTPUT_DIR / "brain_mri_duplicate_groups.csv",
        [
            "duplicate_group_id",
            "group_size",
            "cross_split",
            "labels_in_group",
            "path",
            "split",
            "label",
            "sha256",
            "perceptual_hash",
        ],
        group_rows,
    )

    split_counts = Counter(record["split"] for record in records)
    class_counts = Counter((record["split"], record["label"]) for record in records)
    cross_split_group_count = sum(
        1 for indices in duplicate_groups if len({records[index]["split"] for index in indices}) > 1
    )
    grouped_record_count = len(group_rows)

    report_lines = [
        "# Brain MRI Split and Leakage Audit",
        "",
        "Generated: 2026-08-02",
        "",
        "## Dataset",
        "",
        f"- Readable images: {len(records):,}",
        f"- Training folder: {split_counts.get('Training', 0):,}",
        f"- Testing folder: {split_counts.get('Testing', 0):,}",
        f"- Corrupt images: {len(corrupt_rows):,}",
        "- Perceptual hash: 64-bit pHash from a 32x32 grayscale DCT.",
        f"- Conservative near-duplicate threshold: Hamming distance <= {NEAR_DUPLICATE_DISTANCE}.",
        "",
        "## Findings",
        "",
        f"- Exact duplicate groups: {exact_group_number:,}",
        f"- Exact duplicate groups crossing the provided Training/Testing folders: {exact_cross_split_groups:,}",
        f"- Non-identical near-duplicate pairs: {len(near_rows):,}",
        f"- Non-identical near-duplicate pairs crossing the provided split: {near_cross_split_pairs:,}",
        f"- Combined duplicate/near-duplicate groups: {len(duplicate_groups):,}",
        f"- Combined groups crossing the provided split: {cross_split_group_count:,}",
        f"- Images assigned to a duplicate group: {grouped_record_count:,}",
        f"- Exact duplicate groups with conflicting class labels: {len({row['exact_group_id'] for row in conflict_rows}):,}",
        "",
        "## Class Counts",
        "",
        "| Split | Class | Images |",
        "|---|---|---:|",
    ]
    for (split, label), count in sorted(class_counts.items()):
        report_lines.append(f"| {split} | {label} | {count:,} |")
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            "The provided Training/Testing folders must not be used as uncontaminated generalization evidence when any duplicate group crosses the boundary. New cross-validation folds must assign every combined exact/near-duplicate group to a single outer fold. Near-duplicate pairs are conservative automated candidates and should not be described as proven identical acquisitions without manual review.",
        ]
    )
    (OUTPUT_DIR / "brain_mri_split_audit.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "readable_images": len(records),
        "corrupt_images": len(corrupt_rows),
        "split_counts": dict(split_counts),
        "exact_duplicate_groups": exact_group_number,
        "exact_cross_split_groups": exact_cross_split_groups,
        "near_duplicate_pairs": len(near_rows),
        "near_cross_split_pairs": near_cross_split_pairs,
        "combined_duplicate_groups": len(duplicate_groups),
        "combined_cross_split_groups": cross_split_group_count,
        "grouped_record_count": grouped_record_count,
        "exact_label_conflict_groups": len({row["exact_group_id"] for row in conflict_rows}),
        "near_duplicate_distance": NEAR_DUPLICATE_DISTANCE,
    }


def parse_xray_labels(raw_labels: str) -> tuple[list[int], list[str]]:
    target = [0] * len(XRAY_CLASSES)
    if not raw_labels or raw_labels == "No Finding":
        return target, []
    unknown: list[str] = []
    index = {label: position for position, label in enumerate(XRAY_CLASSES)}
    for label in (part.strip() for part in raw_labels.split("|")):
        if label in index:
            target[index[label]] = 1
        elif label:
            unknown.append(label)
    return target, unknown


def audit_xray() -> dict[str, Any]:
    image_paths_by_name: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(XRAY_ROOT.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths_by_name[path.name.lower()].append(path)

    rows: list[dict[str, Any]] = []
    metadata_occurrences: dict[str, list[int]] = defaultdict(list)
    unknown_label_counts: Counter[str] = Counter()
    missing_label_rows = 0
    missing_images = 0
    view_counts: Counter[str] = Counter()

    with XRAY_METADATA.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            image_name = (row.get("Image Index") or "").strip()
            raw_labels = (row.get("Finding Labels") or "").strip()
            patient_id = (row.get("Patient ID") or "").strip()
            view_position = (row.get("View Position") or "").strip()
            if not raw_labels:
                missing_label_rows += 1
            target, unknown = parse_xray_labels(raw_labels)
            unknown_label_counts.update(unknown)
            view_counts[view_position or "missing"] += 1
            metadata_occurrences[image_name.lower()].append(row_number)
            paths = image_paths_by_name.get(image_name.lower(), [])
            if not paths:
                missing_images += 1
                continue
            rows.append(
                {
                    "image_name": image_name,
                    "image_path": relative(paths[0]),
                    "patient_id": patient_id,
                    "view_position": view_position,
                    "raw_labels": raw_labels,
                    "target": target,
                    "row_number": row_number,
                }
            )

    total = len(rows)
    evaluation_size = max(1, int(round(total * 0.2)))
    training_size = total - evaluation_size
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(total, generator=generator).tolist()
    training_indices = set(order[:training_size])

    patient_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    manifest_rows: list[dict[str, Any]] = []
    distribution: dict[str, np.ndarray] = {
        "all": np.zeros(len(XRAY_CLASSES), dtype=np.int64),
        "original_random_train": np.zeros(len(XRAY_CLASSES), dtype=np.int64),
        "original_random_evaluation": np.zeros(len(XRAY_CLASSES), dtype=np.int64),
    }
    split_sizes = Counter()
    no_finding_counts = Counter()
    for index, row in enumerate(rows):
        split = "original_random_train" if index in training_indices else "original_random_evaluation"
        patient_split_counts[row["patient_id"]][split] += 1
        split_sizes[split] += 1
        target = np.asarray(row["target"], dtype=np.int64)
        distribution["all"] += target
        distribution[split] += target
        if row["raw_labels"] == "No Finding":
            no_finding_counts["all"] += 1
            no_finding_counts[split] += 1
        manifest_rows.append(
            {
                "image_name": row["image_name"],
                "image_path": row["image_path"],
                "patient_id": row["patient_id"],
                "view_position": row["view_position"],
                "raw_labels": row["raw_labels"],
                "original_seed42_split": split,
            }
        )

    overlap_rows = []
    for patient_id, counts in sorted(patient_split_counts.items(), key=lambda item: item[0]):
        train_count = counts["original_random_train"]
        evaluation_count = counts["original_random_evaluation"]
        if train_count and evaluation_count:
            overlap_rows.append(
                {
                    "patient_id": patient_id,
                    "training_images": train_count,
                    "evaluation_images": evaluation_count,
                    "total_images": train_count + evaluation_count,
                }
            )

    write_csv(
        OUTPUT_DIR / "xray_patient_overlap.csv",
        ["patient_id", "training_images", "evaluation_images", "total_images"],
        overlap_rows,
    )
    write_csv(
        OUTPUT_DIR / "xray_image_patient_manifest.csv",
        ["image_name", "image_path", "patient_id", "view_position", "raw_labels", "original_seed42_split"],
        manifest_rows,
    )

    distribution_rows: list[dict[str, Any]] = []
    for split in ["all", "original_random_train", "original_random_evaluation"]:
        denominator = total if split == "all" else split_sizes[split]
        for label_index, label in enumerate(XRAY_CLASSES):
            positive = int(distribution[split][label_index])
            distribution_rows.append(
                {
                    "split": split,
                    "label": label,
                    "images": denominator,
                    "positive_count": positive,
                    "positive_prevalence": positive / denominator if denominator else 0.0,
                    "negative_count": denominator - positive,
                }
            )
        distribution_rows.append(
            {
                "split": split,
                "label": "No Finding",
                "images": denominator,
                "positive_count": no_finding_counts[split],
                "positive_prevalence": no_finding_counts[split] / denominator if denominator else 0.0,
                "negative_count": denominator - no_finding_counts[split],
            }
        )

    write_csv(
        OUTPUT_DIR / "xray_split_label_distribution.csv",
        ["split", "label", "images", "positive_count", "positive_prevalence", "negative_count"],
        distribution_rows,
    )

    duplicate_rows: list[dict[str, Any]] = []
    for image_name, row_numbers in sorted(metadata_occurrences.items()):
        if image_name and len(row_numbers) > 1:
            duplicate_rows.append(
                {
                    "source": "metadata",
                    "image_id": image_name,
                    "occurrence_count": len(row_numbers),
                    "locations": "|".join(str(number) for number in row_numbers),
                }
            )
    for image_name, paths in sorted(image_paths_by_name.items()):
        if len(paths) > 1:
            duplicate_rows.append(
                {
                    "source": "filesystem",
                    "image_id": image_name,
                    "occurrence_count": len(paths),
                    "locations": "|".join(relative(path) for path in paths),
                }
            )
    write_csv(
        OUTPUT_DIR / "xray_duplicate_images.csv",
        ["source", "image_id", "occurrence_count", "locations"],
        duplicate_rows,
    )

    evaluation_patients = {
        row["patient_id"] for row in rows if row["patient_id"] and patient_split_counts[row["patient_id"]]["original_random_evaluation"]
    }
    overlap_patient_ids = {row["patient_id"] for row in overlap_rows}
    overlap_rate = len(overlap_patient_ids) / len(evaluation_patients) if evaluation_patients else 0.0

    report_lines = [
        "# Chest X-ray Patient Split Audit",
        "",
        "Generated: 2026-08-02",
        "",
        "## Dataset",
        "",
        f"- Metadata rows: {sum(len(values) for values in metadata_occurrences.values()):,}",
        f"- Matched image records: {total:,}",
        f"- Unique patients: {len(patient_split_counts):,}",
        f"- Missing image matches: {missing_images:,}",
        f"- Empty label rows: {missing_label_rows:,}",
        f"- Unknown label tokens: {sum(unknown_label_counts.values()):,}",
        f"- Duplicate image identifiers/files: {len(duplicate_rows):,}",
        "",
        "## Original Seed-42 Split Reconstruction",
        "",
        f"- Training images: {split_sizes['original_random_train']:,}",
        f"- Evaluation images: {split_sizes['original_random_evaluation']:,}",
        f"- Patients appearing in both subsets: {len(overlap_rows):,}",
        f"- Evaluation patients also present in training: {overlap_rate:.2%}",
        "- Split implementation: seeded random image-level 80/20 split in `src/model1/train_xray.py`.",
        "- Threshold tuning: performed on the same reconstructed evaluation subset in the original workflow.",
        "",
        "## View Positions",
        "",
        "| View position | Images |",
        "|---|---:|",
    ]
    for view, count in sorted(view_counts.items()):
        report_lines.append(f"| {view} | {count:,} |")
    report_lines.extend(
        [
            "",
            "## Decision",
            "",
            "The original random image-level split is not acceptable as patient-independent generalization evidence when patient overlap is non-zero. New outer folds must assign each patient to exactly one fold. Threshold selection and early stopping must use only an inner validation subset derived from the outer-fold training patients. The current trainer applies no frontal-view filter; PA and AP records are retained and must be described accordingly.",
        ]
    )
    (OUTPUT_DIR / "xray_patient_split_audit.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "metadata_rows": sum(len(values) for values in metadata_occurrences.values()),
        "matched_images": total,
        "unique_patients": len(patient_split_counts),
        "missing_images": missing_images,
        "missing_label_rows": missing_label_rows,
        "unknown_label_counts": dict(unknown_label_counts),
        "duplicate_image_records": len(duplicate_rows),
        "original_split": {
            "seed": SEED,
            "training_images": split_sizes["original_random_train"],
            "evaluation_images": split_sizes["original_random_evaluation"],
            "overlap_patients": len(overlap_rows),
            "evaluation_patient_overlap_rate": overlap_rate,
        },
        "view_counts": dict(view_counts),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    brain_summary = audit_brain_mri()
    xray_summary = audit_xray()
    summary = {
        "generated": "2026-08-02",
        "brain_mri": brain_summary,
        "chest_xray": xray_summary,
    }
    (OUTPUT_DIR / "model1_data_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
