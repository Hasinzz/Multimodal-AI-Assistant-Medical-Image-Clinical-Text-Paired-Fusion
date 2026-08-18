from __future__ import annotations

import csv
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src.config import PROJECT_ROOT, XRAY_CLASSES


METADATA = PROJECT_ROOT / "data" / "structured" / "Data_Entry_2017.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final_research_strengthening"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "thesis_figures"
REPORT_FIGURE_DIR = PROJECT_ROOT / "report_source" / "figures"


def load_targets() -> tuple[np.ndarray, list[int]]:
    rows: list[list[float]] = []
    with METADATA.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            findings = set(record["Finding Labels"].split("|"))
            rows.append([float(label in findings) for label in XRAY_CLASSES])
    targets = np.asarray(rows, dtype=np.float64)
    support = targets.sum(axis=0).astype(int).tolist()
    return targets, support


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", *XRAY_CLASSES])
        for label, values in zip(XRAY_CLASSES, matrix, strict=True):
            writer.writerow([label, *[f"{value:.8f}" for value in values]])


def strongest_pairs(matrix: np.ndarray) -> list[tuple[str, str, float]]:
    pairs = [
        (XRAY_CLASSES[i], XRAY_CLASSES[j], float(matrix[i, j]))
        for i in range(len(XRAY_CLASSES))
        for j in range(i + 1, len(XRAY_CLASSES))
    ]
    return sorted(pairs, key=lambda item: abs(item[2]), reverse=True)


def plot_matrix(matrix: np.ndarray, output_path: Path) -> float:
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    limit = max(0.05, math.ceil(float(np.max(np.abs(off_diagonal))) * 20) / 20)
    display = matrix.copy()
    np.fill_diagonal(display, np.nan)

    labels = [label.replace("_", " ") for label in XRAY_CLASSES]
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e7e9ec")

    fig, ax = plt.subplots(figsize=(13.6, 11.2))
    fig.subplots_adjust(left=0.18, right=0.88, bottom=0.20, top=0.86)
    image = ax.imshow(
        display,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=48, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.tick_params(axis="both", labelsize=10, length=0)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if i == j:
                color = "#30343b"
            else:
                color = "white" if abs(value) >= limit * 0.52 else "#20242a"
            displayed_value = 0.0 if abs(value) < 0.005 else value
            ax.text(j, i, f"{displayed_value:.2f}", ha="center", va="center", fontsize=7.6, color=color)

    fig.suptitle(
        "ChestXray14 pathology label correlation matrix",
        fontsize=17,
        fontweight="bold",
        y=0.965,
        color="#20242a",
    )
    fig.text(
        0.5,
        0.925,
        (
            f"Phi coefficient across 112,120 images; No Finding excluded; "
            f"off-diagonal color scale: {-limit:.2f} to {limit:.2f}"
        ),
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4b5159",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label("Phi correlation", fontsize=11)
    colorbar.ax.tick_params(labelsize=9)
    fig.text(
        0.5,
        0.045,
        "Correlation describes label co-occurrence in the released annotations; it does not imply causality or clinical dependence.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#4b5159",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return limit


def write_report(
    path: Path,
    support: list[int],
    matrix: np.ndarray,
    color_limit: float,
) -> None:
    pairs = strongest_pairs(matrix)
    positive = sorted(pairs, key=lambda item: item[2], reverse=True)[:5]
    negative = sorted(pairs, key=lambda item: item[2])[:5]
    lines = [
        "# Chest X-Ray Label Correlation Matrix",
        "",
        "- Source: `data/structured/Data_Entry_2017.csv`.",
        "- Metadata rows: 112,120.",
        "- Targets: the 14 pathology labels used by Model-1B; `No Finding` excluded.",
        "- Statistic: pairwise Phi coefficient, equivalent to Pearson correlation on binary label indicators.",
        f"- Figure off-diagonal color range: {-color_limit:.2f} to {color_limit:.2f}; diagonal values are 1.00.",
        "- Interpretation: annotation co-occurrence only; not causality, diagnostic dependence, or clinical validation.",
        "",
        "## Label Support",
        "",
        "| Label | Positive images |",
        "|---|---:|",
    ]
    lines.extend(f"| {label.replace('_', ' ')} | {count:,} |" for label, count in zip(XRAY_CLASSES, support, strict=True))
    lines.extend(["", "## Strongest Positive Pairs", "", "| Label pair | Phi |", "|---|---:|"])
    lines.extend(f"| {a.replace('_', ' ')} / {b.replace('_', ' ')} | {value:.4f} |" for a, b, value in positive)
    lines.extend(["", "## Most Negative Pairs", "", "| Label pair | Phi |", "|---|---:|"])
    lines.extend(f"| {a.replace('_', ' ')} / {b.replace('_', ' ')} | {value:.4f} |" for a, b, value in negative)
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `outputs/final_research_strengthening/xray_label_correlation_matrix.csv`",
            "- `outputs/thesis_figures/xray_label_correlation_matrix.png`",
            "- `report_source/figures/xray_label_correlation_matrix.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    targets, support = load_targets()
    if targets.shape != (112_120, len(XRAY_CLASSES)):
        raise RuntimeError(f"Unexpected target matrix shape: {targets.shape}")
    matrix = np.corrcoef(targets, rowvar=False)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Correlation matrix contains non-finite values")

    matrix_path = OUTPUT_DIR / "xray_label_correlation_matrix.csv"
    figure_path = FIGURE_DIR / "xray_label_correlation_matrix.png"
    write_matrix(matrix_path, matrix)
    color_limit = plot_matrix(matrix, figure_path)
    REPORT_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(figure_path, REPORT_FIGURE_DIR / figure_path.name)
    write_report(OUTPUT_DIR / "xray_label_correlation_matrix.md", support, matrix, color_limit)

    print(f"rows={targets.shape[0]}")
    print(f"labels={targets.shape[1]}")
    print(f"matrix={matrix_path}")
    print(f"figure={figure_path}")
    print(f"report_figure={REPORT_FIGURE_DIR / figure_path.name}")


if __name__ == "__main__":
    main()
