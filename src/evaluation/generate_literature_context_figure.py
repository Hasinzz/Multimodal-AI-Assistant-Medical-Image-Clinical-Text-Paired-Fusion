from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT


BRAIN_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model1_cross_validation"
    / "brain_mri"
    / "brain_mri_5fold_summary.json"
)
XRAY_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "model1_cross_validation"
    / "chest_xray"
    / "xray_cross_validation_summary.json"
)
OUTPUT = PROJECT_ROOT / "outputs" / "thesis_figures" / "literature_benchmark_context.png"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_panel(ax, labels, values, errors, colors, title, xlabel, lower):
    positions = list(range(len(labels)))
    ax.barh(positions, values, xerr=errors, color=colors, alpha=0.9, capsize=3)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(lower, 1.0)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.22)
    for position, value in zip(positions, values):
        ax.text(min(value + 0.008, 0.985), position, f"{value:.3f}", va="center", fontsize=8)


def main() -> None:
    brain = read_json(BRAIN_SUMMARY)
    xray = read_json(XRAY_SUMMARY)
    brain_mean = brain["aggregate"]["accuracy"]["mean"]
    brain_sd = brain["aggregate"]["accuracy"]["std_sample"]
    resnet = read_json(BRAIN_SUMMARY.parent / "resnet50_5fold_summary.json")
    resnet_mean = resnet["aggregate"]["accuracy"]["mean"]
    resnet_sd = resnet["aggregate"]["accuracy"]["std_sample"]
    xray_mean = xray["aggregate"]["inner_tuned"]["macro_auroc"]["mean"]
    xray_sd = xray["aggregate"]["inner_tuned"]["macro_auroc"]["std_sample"]

    thesis_color = "#2f6f73"
    published_color = "#b45f34"
    historical_color = "#68747c"
    brain_labels = [
        "Thesis DenseNet-121\ngrouped 5-fold",
        "Thesis ResNet-50\ngrouped 5-fold",
        "Rasheed 2025\nstated 5-fold",
        "Disci 2025 Xception\nsupplied-folder test",
        "Disci 2025 DenseNet\nsupplied-folder test",
        "Disci 2025 ResNet\nsupplied-folder test",
        "Mao 2025 enhanced\n10-fold + ten-crop",
        "Mao 2025 original\n10-fold + ten-crop",
    ]
    brain_values = [brain_mean, resnet_mean, 0.973, 0.9527, 0.9285, 0.9062, 0.962, 0.499]
    brain_errors = [brain_sd, resnet_sd, 0.008, 0, 0, 0, 0, 0]
    brain_colors = [thesis_color, thesis_color] + [published_color] * 6

    xray_labels = [
        "Thesis DenseNet-121\npatient-wise 5-fold",
        "Thesis historical\nrandom image split",
        "Wang 2017\n8-label image split",
        "Baltruschat 2019\npatient resampling",
        "Baltruschat 2019\nofficial split",
        "Kufel 2023\ncustom patient split",
    ]
    xray_values = [xray_mean, 0.8133, 0.7212, 0.821, 0.806, 0.843]
    xray_errors = [xray_sd, 0, 0, 0.012, 0, 0]
    xray_colors = [thesis_color, historical_color] + [published_color] * 4

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2))
    plot_panel(axes[0], brain_labels, brain_values, brain_errors, brain_colors, "Brain MRI protocol context", "Reported accuracy", 0.45)
    plot_panel(axes[1], xray_labels, xray_values, xray_errors, xray_colors, "Chest X-ray protocol context", "Reported mean/macro AUROC", 0.68)
    fig.suptitle("Published results provide context, not a directly comparable leaderboard", fontsize=13)
    fig.text(
        0.5,
        0.012,
        "Metrics, partitions, duplicate/patient controls, initialization, augmentation, labels, and model-selection rules differ.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220)
    plt.close(fig)
    print(OUTPUT.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
