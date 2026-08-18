from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "thesis_figures"

INK = "#24323f"
MUTED = "#66717c"
BLUE = "#3b6ea8"
TEAL = "#2a9d8f"
GREEN = "#4f8f46"
ORANGE = "#d9912b"
RED = "#c65f5f"
GRAY = "#eef2f5"
LIGHT_BLUE = "#e8f0f8"
LIGHT_TEAL = "#e6f4f1"
LIGHT_ORANGE = "#fff2dc"
LIGHT_RED = "#fae9e9"
WHITE = "#ffffff"


def wrap(text: str, width: int = 24) -> str:
    wrapped_lines: list[str] = []
    for line in str(text).splitlines():
        if not line.strip():
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(line, width=width, break_long_words=False))
    return "\n".join(wrapped_lines)


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fc: str = WHITE,
    ec: str = INK,
    fontsize: int = 10,
    weight: str = "normal",
    wrap_width: int = 24,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.4,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        wrap(text, wrap_width),
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        fontweight=weight,
    )
    return patch


def arrow(ax, start, end, color: str = MUTED, lw: float = 1.8, rad: float = 0.0):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        shrinkA=6,
        shrinkB=6,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    return arr


def new_canvas(width: float = 13, height: float = 7, title: str | None = None):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    if title:
        ax.text(6, 6.75, title, ha="center", va="top", fontsize=16, fontweight="bold", color=INK)
    return fig, ax


def save(fig, name: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_system_architecture():
    fig, ax = new_canvas(width=14, height=7.5, title="Complete Multimodal AI Assistant Architecture")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    draw_box(ax, 0.25, 5.35, 2.05, 0.85, "Brain MRI /\nChest X-ray", LIGHT_BLUE, BLUE)
    draw_box(ax, 0.25, 3.75, 2.05, 0.95, "Scanned prescription /\nlab report", LIGHT_TEAL, TEAL)
    draw_box(ax, 0.25, 2.15, 2.05, 0.95, "Typed doctor note /\nclinical text", LIGHT_TEAL, TEAL)

    draw_box(ax, 2.85, 5.2, 2.45, 1.15, "Model-1\nImage Classification", WHITE, BLUE, weight="bold", wrap_width=25)
    draw_box(ax, 2.85, 3.55, 2.45, 1.15, "Model-2A\nOCR + Extraction", WHITE, TEAL, weight="bold", wrap_width=25)
    draw_box(ax, 2.85, 1.9, 2.45, 1.15, "Model-2B\nClinical Text Processing", WHITE, TEAL, weight="bold", wrap_width=25)
    draw_box(ax, 6.15, 3.3, 2.55, 1.6, "Model-3\nLate Fusion + RAG\nRetrieval-supported\nsummarization", WHITE, GREEN, weight="bold", wrap_width=28)
    draw_box(ax, 9.65, 4.65, 2.75, 0.85, "Patient summary", LIGHT_ORANGE, ORANGE)
    draw_box(ax, 9.65, 3.4, 2.75, 0.85, "Retrieval-supported\nfollow-up note", LIGHT_ORANGE, ORANGE)
    draw_box(ax, 9.65, 2.15, 2.75, 0.85, "Structured JSON output", LIGHT_ORANGE, ORANGE)

    draw_box(ax, 6.15, 0.75, 2.55, 0.85, "Streamlit UI\nStable + V4 modes", GRAY, MUTED)
    draw_box(ax, 2.85, 0.7, 2.45, 0.95, "Optional V4 layer\nYOLO ROI + weak-label BERT NER", LIGHT_RED, RED, wrap_width=28)

    arrow(ax, (2.3, 5.78), (2.85, 5.78), color=BLUE)
    arrow(ax, (2.3, 4.22), (2.85, 4.12), color=TEAL)
    arrow(ax, (2.3, 2.62), (2.85, 2.47), color=TEAL)
    arrow(ax, (5.3, 5.78), (6.15, 4.55), color=GREEN)
    arrow(ax, (5.3, 4.12), (6.15, 4.2), color=GREEN)
    arrow(ax, (5.3, 2.47), (6.15, 3.65), color=GREEN)
    arrow(ax, (8.7, 4.5), (9.65, 5.08), color=ORANGE)
    arrow(ax, (8.7, 4.1), (9.65, 3.82), color=ORANGE)
    arrow(ax, (8.7, 3.7), (9.65, 2.58), color=ORANGE)
    arrow(ax, (4.05, 1.65), (4.05, 1.9), color=RED)
    arrow(ax, (7.42, 3.3), (7.42, 1.6), color=MUTED)

    ax.text(6.5, 0.25, "Stable system is the final thesis baseline; V4 is an experimental comparison layer.", ha="center", fontsize=10, color=RED)
    save(fig, "fig01_complete_system_architecture")


def figure_model1_pipeline():
    fig, ax = new_canvas(width=14, height=7, title="Model-1 Image Analysis Pipeline")
    ax.set_xlim(0, 13)
    steps = [
        ("Medical image\nBrain MRI or X-ray", LIGHT_BLUE, BLUE),
        ("Preprocess\nResize 224x224\nNormalize", WHITE, BLUE),
        ("Optional (default off)\nCLAHE for X-ray\nN4 for MRI", GRAY, MUTED),
        ("DenseNet-121\nfeature extractor", WHITE, BLUE),
        ("Linear classifier\n4 MRI classes or\n14 X-ray labels", WHITE, BLUE),
        ("Output\nprobabilities,\nembedding, summary", LIGHT_ORANGE, ORANGE),
    ]
    xs = [0.35, 2.45, 4.55, 6.65, 8.75, 10.85]
    for x, (text, fc, ec) in zip(xs, steps):
        draw_box(ax, x, 3.2, 1.65, 1.35, text, fc, ec, fontsize=9, wrap_width=20)
    for x in xs[:-1]:
        arrow(ax, (x + 1.65, 3.88), (x + 2.1, 3.88), color=BLUE)

    draw_box(ax, 2.2, 1.25, 3.0, 0.9, "Brain MRI final_v2\naccuracy 0.9375, macro F1 0.9359", LIGHT_TEAL, TEAL, fontsize=10)
    draw_box(ax, 6.8, 1.25, 3.1, 0.9, "X-ray large_v2 + tuned thresholds\nmacro AUROC 0.8133, micro AUROC 0.8377", LIGHT_TEAL, TEAL, fontsize=10)
    arrow(ax, (4.0, 3.2), (3.7, 2.15), color=TEAL)
    arrow(ax, (8.85, 3.2), (8.35, 2.15), color=TEAL)
    save(fig, "fig02_model1_image_pipeline")


def figure_brain_architecture():
    fig, ax = new_canvas(title="Brain MRI Classifier Architecture")
    draw_box(ax, 0.8, 4.0, 1.8, 1.0, "Input MRI\n224x224 RGB", LIGHT_BLUE, BLUE)
    draw_box(ax, 3.2, 4.0, 2.0, 1.0, "DenseNet-121\nbackbone", WHITE, BLUE, weight="bold")
    draw_box(ax, 5.9, 4.0, 1.8, 1.0, "Global average\nfeatures", WHITE, BLUE)
    draw_box(ax, 8.3, 4.0, 1.8, 1.0, "Linear layer\n4 classes", WHITE, BLUE)
    draw_box(ax, 10.3, 4.0, 1.2, 1.0, "Softmax\nlabel", LIGHT_ORANGE, ORANGE)
    for start, end in [((2.6, 4.5), (3.2, 4.5)), ((5.2, 4.5), (5.9, 4.5)), ((7.7, 4.5), (8.3, 4.5)), ((10.1, 4.5), (10.3, 4.5))]:
        arrow(ax, start, end, color=BLUE)
    draw_box(ax, 1.2, 1.55, 3.1, 1.25, "Classes\nglioma, meningioma,\nno_tumor, pituitary", LIGHT_TEAL, TEAL)
    draw_box(ax, 4.65, 1.55, 3.1, 1.25, "Selected checkpoint\nbrain_best_model_gpu_final_v2.pt", GRAY, MUTED)
    draw_box(ax, 8.1, 1.55, 2.7, 1.25, "Validation\naccuracy 0.9375\nmacro F1 0.9359", LIGHT_TEAL, TEAL)
    save(fig, "fig03_brain_mri_classifier_architecture")


def figure_xray_architecture():
    fig, ax = new_canvas(title="Chest X-ray Multi-label Classifier Architecture")
    draw_box(ax, 0.65, 4.0, 1.65, 1.0, "Input X-ray\n224x224 RGB", LIGHT_BLUE, BLUE)
    draw_box(ax, 2.8, 4.0, 1.85, 1.0, "DenseNet-121\nbackbone", WHITE, BLUE, weight="bold")
    draw_box(ax, 5.1, 4.0, 1.75, 1.0, "Feature vector", WHITE, BLUE)
    draw_box(ax, 7.25, 4.0, 1.9, 1.0, "Linear layer\n14 labels", WHITE, BLUE)
    draw_box(ax, 9.6, 4.0, 1.75, 1.0, "Sigmoid +\ntuned thresholds", LIGHT_ORANGE, ORANGE)
    for start, end in [((2.3, 4.5), (2.8, 4.5)), ((4.65, 4.5), (5.1, 4.5)), ((6.85, 4.5), (7.25, 4.5)), ((9.15, 4.5), (9.6, 4.5))]:
        arrow(ax, start, end, color=BLUE)
    draw_box(ax, 0.9, 1.45, 3.0, 1.35, "14 NIH labels\nAtelectasis, Effusion,\nInfiltration, etc.", LIGHT_TEAL, TEAL)
    draw_box(ax, 4.5, 1.45, 3.1, 1.35, "Selected checkpoint\nxray_best_model_gpu_large_v2.pt\nthreshold JSON used", GRAY, MUTED)
    draw_box(ax, 8.1, 1.45, 3.0, 1.35, "Final metrics\nmacro AUROC 0.8133\nmicro AUROC 0.8377", LIGHT_TEAL, TEAL)
    save(fig, "fig04_xray_classifier_architecture")


def figure_model2_pipeline():
    fig, ax = new_canvas(width=14, height=8, title="Model-2 Clinical Text and Document Understanding Pipeline")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.text(0.35, 6.35, "Model-2A: Scanned Document OCR/Extraction", fontsize=11, fontweight="bold", color=TEAL)
    draw_box(ax, 0.35, 5.0, 1.9, 0.9, "Prescription /\nlab report", LIGHT_TEAL, TEAL)
    draw_box(ax, 2.65, 5.0, 1.35, 0.9, "OCR", WHITE, TEAL)
    draw_box(ax, 4.4, 5.0, 1.55, 0.9, "Text cleaning", WHITE, TEAL)
    draw_box(ax, 6.35, 5.0, 1.9, 0.9, "Rule-based\nentity extraction", WHITE, TEAL)
    draw_box(ax, 8.65, 5.0, 1.75, 0.9, "Structured JSON", LIGHT_ORANGE, ORANGE)
    for start, end in [((2.25, 5.45), (2.65, 5.45)), ((4.0, 5.45), (4.4, 5.45)), ((5.95, 5.45), (6.35, 5.45)), ((8.25, 5.45), (8.65, 5.45))]:
        arrow(ax, start, end, color=TEAL)

    ax.text(0.35, 3.75, "Model-2B: Doctor-Note Clinical Text Understanding", fontsize=11, fontweight="bold", color=BLUE)
    draw_box(ax, 0.35, 2.4, 1.9, 0.9, "Typed doctor note /\nMTSamples-style text", LIGHT_BLUE, BLUE, fontsize=9)
    draw_box(ax, 2.65, 2.4, 1.35, 0.9, "Text cleaning", WHITE, BLUE, fontsize=9)
    draw_box(ax, 4.4, 2.4, 1.55, 0.9, "Specialty\nclassifier", WHITE, BLUE, fontsize=9)
    draw_box(ax, 6.35, 2.4, 1.9, 0.9, "Weak-label NER", WHITE, BLUE, fontsize=9)
    draw_box(ax, 8.65, 2.4, 1.75, 0.9, "Structured JSON", LIGHT_ORANGE, ORANGE)
    for start, end in [((2.25, 2.85), (2.65, 2.85)), ((4.0, 2.85), (4.4, 2.85)), ((5.95, 2.85), (6.35, 2.85)), ((8.25, 2.85), (8.65, 2.85))]:
        arrow(ax, start, end, color=BLUE)

    draw_box(ax, 11.0, 3.65, 1.7, 1.35, "Model-3\nLate fusion + RAG\nsummarization", LIGHT_ORANGE, ORANGE, weight="bold", fontsize=9)
    arrow(ax, (10.4, 5.45), (11.0, 4.65), color=GREEN)
    arrow(ax, (10.4, 2.85), (11.0, 4.0), color=GREEN)
    ax.text(6.5, 0.85, "Weak-label NER output is not expert clinical annotation.", ha="center", fontsize=10, color=RED)
    save(fig, "fig05_model2_document_pipeline")


def figure_model3_pipeline():
    fig, ax = new_canvas(width=14, height=7, title="Model-3 Fusion and RAG Pipeline")
    ax.set_xlim(0, 13)
    draw_box(ax, 0.45, 4.9, 2.15, 0.85, "Model-1 output\nimage findings", LIGHT_BLUE, BLUE)
    draw_box(ax, 0.45, 3.5, 2.15, 0.85, "Model-2A output\nOCR + entities", LIGHT_TEAL, TEAL)
    draw_box(ax, 0.45, 2.1, 2.15, 0.85, "Model-2B output\ndoctor-note text", LIGHT_TEAL, TEAL)
    draw_box(ax, 3.6, 3.65, 2.1, 1.0, "Build fused query", WHITE, GREEN, weight="bold")
    draw_box(ax, 6.45, 3.65, 2.0, 1.0, "Local TF-IDF\nretriever", WHITE, GREEN)
    draw_box(ax, 6.45, 1.75, 2.0, 0.9, "Knowledge base\ndata/kb", GRAY, MUTED)
    draw_box(ax, 9.25, 3.55, 2.65, 1.2, "Generate patient summary\n+ retrieval-supported\nfollow-up note", LIGHT_ORANGE, ORANGE, wrap_width=28)
    draw_box(ax, 9.25, 1.75, 2.45, 0.9, "Evidence snippets\nwith source + score", LIGHT_ORANGE, ORANGE, wrap_width=28)
    arrow(ax, (2.6, 5.32), (3.5, 4.35), color=GREEN)
    arrow(ax, (2.6, 3.92), (3.5, 4.15), color=GREEN)
    arrow(ax, (2.6, 2.52), (3.5, 3.95), color=GREEN)
    arrow(ax, (5.7, 4.15), (6.45, 4.15), color=GREEN)
    arrow(ax, (7.45, 2.65), (7.45, 3.65), color=MUTED)
    arrow(ax, (8.45, 4.15), (9.25, 4.15), color=GREEN)
    arrow(ax, (10.45, 3.65), (10.45, 2.65), color=ORANGE)
    ax.text(6.5, 0.75, "Stable late fusion with retrieved knowledge-base context; trained cross-modal attention remains future work.", ha="center", fontsize=10, color=RED)
    save(fig, "fig06_model3_rag_fusion_pipeline")


def figure_v4_layer():
    fig, ax = new_canvas(title="V4 Advanced Improvement Layer")
    draw_box(ax, 0.6, 4.6, 2.2, 1.0, "Stable final system\nunchanged baseline", LIGHT_TEAL, TEAL, weight="bold")
    draw_box(ax, 3.4, 5.0, 2.4, 0.9, "YOLOv8n ROI\npseudo-labeled pages", LIGHT_RED, RED)
    draw_box(ax, 3.4, 3.75, 2.4, 0.9, "BERT NER\nweak labels", LIGHT_RED, RED)
    draw_box(ax, 3.4, 2.5, 2.4, 0.9, "RAG KB V4\nexpanded text KB", LIGHT_RED, RED)
    draw_box(ax, 6.6, 4.65, 2.0, 0.9, "Optional V4 mode\ncomparison only", WHITE, RED, weight="bold")
    draw_box(ax, 9.1, 5.0, 2.2, 0.9, "Claim: pseudo-label\nROI experiment", LIGHT_ORANGE, ORANGE)
    draw_box(ax, 9.1, 3.75, 2.2, 0.9, "Claim: weak-label\nBERT NER", LIGHT_ORANGE, ORANGE)
    draw_box(ax, 9.1, 2.5, 2.2, 0.9, "Do not claim\nclinical validation", LIGHT_ORANGE, ORANGE)
    draw_box(ax, 0.6, 1.35, 2.2, 0.9, "Cross-modal attention\nscaffold only", GRAY, MUTED)
    draw_box(ax, 3.4, 1.35, 2.4, 0.9, "No true paired\nimage-text-label dataset", GRAY, MUTED)
    arrow(ax, (2.8, 5.1), (3.4, 5.45), color=RED)
    arrow(ax, (2.8, 5.1), (3.4, 4.2), color=RED)
    arrow(ax, (2.8, 5.1), (3.4, 2.95), color=RED)
    arrow(ax, (5.8, 5.45), (6.6, 5.1), color=RED)
    arrow(ax, (5.8, 4.2), (6.6, 5.0), color=RED)
    arrow(ax, (5.8, 2.95), (6.6, 4.9), color=RED)
    arrow(ax, (8.6, 5.1), (9.1, 5.45), color=ORANGE)
    arrow(ax, (8.6, 5.0), (9.1, 4.2), color=ORANGE)
    arrow(ax, (8.6, 4.85), (9.1, 2.95), color=ORANGE)
    arrow(ax, (2.8, 1.8), (3.4, 1.8), color=MUTED)
    ax.text(6, 0.55, "Use V4 as an experimental improvement layer, not as a replacement final model.", ha="center", fontsize=10, color=RED)
    save(fig, "fig07_v4_advanced_layer")


def figure_streamlit_workflow():
    fig, ax = new_canvas(title="Streamlit Demonstration Workflow")
    draw_box(ax, 0.65, 4.2, 1.7, 1.0, "Open local UI\nlocalhost:8501", GRAY, MUTED)
    draw_box(ax, 2.9, 4.2, 1.7, 1.0, "Select mode\nStable or V4", WHITE, BLUE)
    draw_box(ax, 5.15, 4.2, 1.7, 1.0, "Image/document\nor note text", WHITE, BLUE)
    draw_box(ax, 7.4, 4.2, 1.7, 1.0, "Run analysis", WHITE, GREEN)
    draw_box(ax, 9.65, 4.2, 1.7, 1.0, "View outputs\nand download JSON", LIGHT_ORANGE, ORANGE)
    for x in [2.35, 4.6, 6.85, 9.1]:
        arrow(ax, (x, 4.7), (x + 0.55, 4.7), color=BLUE)
    draw_box(ax, 1.2, 1.75, 2.1, 0.9, "Image prediction", LIGHT_BLUE, BLUE)
    draw_box(ax, 3.7, 1.75, 2.1, 0.9, "OCR preview\nand entities", LIGHT_TEAL, TEAL)
    draw_box(ax, 6.2, 1.75, 2.1, 0.9, "Retrieved evidence", GRAY, MUTED)
    draw_box(ax, 8.7, 1.75, 2.1, 0.9, "Patient summary +\nfollow-up note", LIGHT_ORANGE, ORANGE)
    arrow(ax, (8.25, 4.2), (2.25, 2.65), color=MUTED, rad=0.12)
    arrow(ax, (8.25, 4.2), (4.75, 2.65), color=MUTED, rad=0.08)
    arrow(ax, (8.25, 4.2), (7.25, 2.65), color=MUTED, rad=0.02)
    arrow(ax, (8.25, 4.2), (9.75, 2.65), color=MUTED, rad=-0.08)
    save(fig, "fig08_streamlit_demo_workflow")


def figure_brain_confusion_matrix():
    path = PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_final_v2" / "brain_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cm = np.array(data["metrics"]["confusion_matrix"])
    labels = ["Glioma", "Meningioma", "No tumor", "Pituitary"]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Brain MRI Final_v2 Confusion Matrix", fontsize=15, fontweight="bold", color=INK)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=WHITE if cm[i, j] > threshold else INK, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count")
    fig.tight_layout()
    save(fig, "fig09_brain_mri_confusion_matrix")


def figure_xray_auroc():
    path = PROJECT_ROOT / "outputs" / "training" / "xray_gpu_large_v2" / "xray_metrics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    table = data["best_metrics"]["per_class_table"]
    labels = [row["class_name"] for row in table]
    values = [row["auroc"] for row in table]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    bars = ax.bar(range(len(labels)), values, color=BLUE, edgecolor=INK, linewidth=0.7)
    ax.axhline(data["best_metrics"]["macro_auroc"], color=RED, linestyle="--", linewidth=1.6, label=f"Macro AUROC {data['best_metrics']['macro_auroc']:.3f}")
    ax.set_title("Chest X-ray Large_v2 Per-class AUROC", fontsize=15, fontweight="bold", color=INK)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.55, 1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 0.008, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save(fig, "fig10_xray_per_class_auroc")


def figure_case_distribution():
    report = PROJECT_ROOT / "outputs" / "final_run_100_tuned_v2" / "main_run_report.txt"
    counts = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- ") and ":" in line:
            label, value = line[2:].split(":", 1)
            counts[label.strip()] = int(value.strip())

    order = ["brain_image_only", "xray_image_only", "lab_doc_only", "prescription_doc_only", "brain_fusion", "xray_fusion"]
    labels = ["Brain image", "X-ray image", "Lab doc", "Prescription doc", "Brain fusion", "X-ray fusion"]
    values = [counts[key] for key in order]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = [BLUE, BLUE, TEAL, TEAL, ORANGE, ORANGE]
    bars = ax.bar(labels, values, color=colors, edgecolor=INK, linewidth=0.8)
    ax.set_title("100-case Technical Pipeline Validation Breakdown", fontsize=15, fontweight="bold", color=INK)
    ax.set_ylabel("Number of cases")
    ax.set_ylim(0, max(values) + 7)
    ax.grid(axis="y", alpha=0.25)
    for rect, value in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 0.7, str(value), ha="center", va="bottom", fontsize=11)
    ax.text(0.5, -0.22, "Final run: 100 requested, 100 completed, 0 failed. This is technical validation, not clinical validation.", transform=ax.transAxes, ha="center", fontsize=10, color=RED)
    fig.tight_layout()
    save(fig, "fig11_technical_validation_case_distribution")


def figure_stable_v4_comparison():
    path = PROJECT_ROOT / "outputs" / "v4_advanced_improvement" / "comparison" / "comparison_summary.md"
    text = path.read_text(encoding="utf-8")
    stable = {"OCR text length": 1345, "Entities": 2, "Runtime seconds": 0.48}
    v4 = {"OCR text length": 1249, "Entities": 41, "Runtime seconds": 6.15}
    current = None
    for line in text.splitlines():
        line = line.strip()
        if "Stable Pipeline" in line:
            current = stable
        elif "V4 Advanced Pipeline" in line:
            current = v4
        elif current is not None and line.startswith("- OCR text length:"):
            current["OCR text length"] = float(line.split(":", 1)[1].strip())
        elif current is not None and line.startswith("- Number of entities extracted:"):
            current["Entities"] = float(line.split(":", 1)[1].strip())
        elif current is not None and line.startswith("- Runtime seconds:"):
            current["Runtime seconds"] = float(line.split(":", 1)[1].strip())

    metrics = ["OCR text length", "Entities", "Runtime seconds"]
    ylabels = ["Characters", "Count", "Seconds"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        vals = [stable[metric], v4[metric]]
        bars = ax.bar(["Stable", "V4"], vals, color=[BLUE, RED], edgecolor=INK, linewidth=0.8)
        ax.set_title(metric, fontsize=12, fontweight="bold", color=INK)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for rect, value in zip(bars, vals):
            text_value = f"{value:.2f}" if metric == "Runtime seconds" else f"{int(value)}"
            ax.text(rect.get_x() + rect.get_width() / 2, value + max(vals) * 0.035, text_value, ha="center", va="bottom", fontsize=10)
    fig.suptitle("Stable vs V4 Document Pipeline Comparison", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.5, -0.02, "V4 extracts more weak-label entities but should not be interpreted as clinically better without manual review.", ha="center", fontsize=10, color=RED)
    fig.tight_layout()
    save(fig, "fig12_stable_vs_v4_document_comparison")


def figure_report_roadmap():
    fig, ax = new_canvas(title="Recommended Report Storyline")
    chapters = [
        ("1. Introduction\nproblem, motivation, objectives", LIGHT_BLUE, BLUE),
        ("2. Related Work\nimage AI, OCR/NLP, RAG", LIGHT_BLUE, BLUE),
        ("3. Requirements\nconstraints, ethics, risks", GRAY, MUTED),
        ("4. System Design\nModel-1, Model-2, Model-3, UI", LIGHT_TEAL, TEAL),
        ("5. Data + Preprocessing\ndatasets and splits", LIGHT_TEAL, TEAL),
        ("6. Evaluation\nmetrics, 100-case validation", LIGHT_ORANGE, ORANGE),
        ("7. Discussion\nlimits and valid claims", LIGHT_RED, RED),
        ("8. Conclusion\nfuture work", LIGHT_RED, RED),
    ]
    positions = [(0.5, 4.95), (3.4, 4.95), (6.3, 4.95), (9.2, 4.95), (0.5, 2.45), (3.4, 2.45), (6.3, 2.45), (9.2, 2.45)]
    for (x, y), (text, fc, ec) in zip(positions, chapters):
        draw_box(ax, x, y, 2.3, 1.0, text, fc, ec, fontsize=9, wrap_width=22)
    for i in range(3):
        arrow(ax, (positions[i][0] + 2.3, positions[i][1] + 0.5), (positions[i + 1][0], positions[i + 1][1] + 0.5), color=MUTED)
    arrow(ax, (10.35, 4.95), (10.35, 3.45), color=MUTED, rad=0.1)
    for i in range(4, 7):
        arrow(ax, (positions[i][0] + 2.3, positions[i][1] + 0.5), (positions[i + 1][0], positions[i + 1][1] + 0.5), color=MUTED)
    save(fig, "fig13_report_storyline_roadmap")


def figure_component_output_success():
    summary_path = PROJECT_ROOT / "outputs" / "final_run_100_tuned_v2" / "main_run_summary.csv"
    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    components = [
        ("Model-1\nimage analysis", "model1_output_path", BLUE),
        ("Model-2\ndocument pipeline", "model2_output_path", TEAL),
        ("Model-3\nfusion + RAG", "model3_output_path", GREEN),
    ]
    applicable = []
    success = []
    for _, column, _ in components:
        relevant = [row for row in rows if row.get(column)]
        ok = [
            row
            for row in relevant
            if row.get("status") == "success" and Path(row.get(column, "")).exists()
        ]
        applicable.append(len(relevant))
        success.append(len(ok))

    labels = [item[0] for item in components]
    colors = [item[2] for item in components]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.bar(x, success, color=colors, edgecolor=INK, linewidth=0.8)
    ax.set_title("Model-1 vs Model-2 vs Model-3 Technical Output Success", fontsize=15, fontweight="bold", color=INK)
    ax.set_ylabel("Successful output files")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(applicable) + 12)
    ax.grid(axis="y", alpha=0.25)
    for rect, ok, total in zip(bars, success, applicable):
        ax.text(rect.get_x() + rect.get_width() / 2, ok + 2, f"{ok}/{total}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(
        0.5,
        -0.2,
        "Counts are from the 100-case final technical validation.\nThis compares execution coverage, not shared diagnostic accuracy.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        color=RED,
    )
    fig.tight_layout()
    save(fig, "fig14_model_component_output_success")


def figure_best_version_summary():
    brain = json.loads((PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_final_v2" / "brain_metrics.json").read_text(encoding="utf-8"))
    xray = json.loads((PROJECT_ROOT / "outputs" / "training" / "xray_gpu_large_v2_threshold_tuning" / "xray_threshold_tuning_metrics.json").read_text(encoding="utf-8"))
    yolo_rows = list(csv.DictReader((PROJECT_ROOT / "outputs" / "v4_advanced_improvement" / "yolo_roi" / "yolov8n_roi_v4_pseudolabel_full" / "results.csv").open("r", encoding="utf-8")))
    yolo_last = yolo_rows[-1]
    ner = json.loads((PROJECT_ROOT / "outputs" / "v4_advanced_improvement" / "biobert_ner" / "ner_metrics_v4.json").read_text(encoding="utf-8"))
    report = (PROJECT_ROOT / "outputs" / "final_run_100_tuned_v2" / "main_run_report.txt").read_text(encoding="utf-8")

    cards = [
        (
            "Brain MRI\nfinal_v2",
            f"Accuracy {brain['metrics']['accuracy']:.4f}\nMacro F1 {brain['metrics']['macro_f1']:.4f}\nStable final",
            LIGHT_TEAL,
            TEAL,
        ),
        (
            "Chest X-ray\nlarge_v2",
            f"Macro AUROC {xray['default_metrics']['macro_auroc']:.4f}\nTuned macro F1 {xray['tuned_metrics']['macro_f1']:.4f}\nStable final",
            LIGHT_TEAL,
            TEAL,
        ),
        (
            "Model-2\nstable OCR/rules",
            "OCR + rule extraction\nStructured JSON\nStable final",
            LIGHT_BLUE,
            BLUE,
        ),
        (
            "Model-3\nRAG fusion",
            "TF-IDF retrieval\nSummary + follow-up note\nStable final",
            LIGHT_BLUE,
            BLUE,
        ),
        (
            "YOLO ROI V4",
            f"mAP50 {float(yolo_last['metrics/mAP50(B)']):.4f}\nPseudo-label eval\nExperimental",
            LIGHT_RED,
            RED,
        ),
        (
            "BERT NER V4",
            f"Weak-label Entity-F1 {ner['test_entity_f1']:.5f}\nExperimental",
            LIGHT_RED,
            RED,
        ),
    ]

    fig, ax = new_canvas(width=14, height=7.4, title="Best Available Versions and Claim Status")
    ax.set_xlim(0, 13)
    positions = [(0.55, 4.55), (3.65, 4.55), (6.75, 4.55), (9.85, 4.55), (2.15, 2.05), (7.75, 2.05)]
    for (x, y), (title, body, fc, ec) in zip(positions, cards):
        draw_box(ax, x, y, 2.55, 1.35, f"{title}\n\n{body}", fc, ec, fontsize=9, weight="bold", wrap_width=26)
    ax.text(
        6.5,
        0.85,
        "Stable final system: Model-1 + Model-2 + Model-3. V4 components are optional experimental evidence.",
        ha="center",
        fontsize=10,
        color=RED,
    )
    save(fig, "fig15_best_version_claim_status_summary")


def _load_brain_runs():
    names = [
        ("Initial retrained", "brain_mri"),
        ("GPU final", "brain_mri_gpu_final"),
        ("final_v2\nselected", "brain_mri_gpu_final_v2"),
        ("retrain_v3", "brain_mri_gpu_retrain_v3"),
    ]
    rows = []
    for label, folder in names:
        p = PROJECT_ROOT / "outputs" / "training" / folder / "brain_metrics.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        metrics = data["metrics"]
        rows.append(
            {
                "label": label,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "selected": "selected" in label,
            }
        )
    return rows


def figure_brain_run_comparison():
    rows = _load_brain_runs()
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.bar(x - width / 2, [row["accuracy"] for row in rows], width, label="Accuracy", color=BLUE, edgecolor=INK, linewidth=0.7)
    ax.bar(x + width / 2, [row["macro_f1"] for row in rows], width, label="Macro F1", color=TEAL, edgecolor=INK, linewidth=0.7)
    ax.set_title("Brain MRI Training Run Comparison", fontsize=15, fontweight="bold", color=INK)
    ax.set_ylabel("Score")
    ax.set_ylim(0.88, 0.95)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    for idx, row in enumerate(rows):
        if row["selected"]:
            ax.text(idx, 0.948, "selected", ha="center", va="top", fontsize=9, color=RED, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig16_brain_mri_run_comparison")


def _load_xray_tuned_runs():
    names = [
        ("10k subset", "xray_gpu_10k", None),
        ("Full", "xray_gpu_full", "xray_gpu_full_threshold_tuning"),
        ("large_v2\nselected", "xray_gpu_large_v2", "xray_gpu_large_v2_threshold_tuning"),
        ("retrain_v3", "xray_gpu_retrain_v3", "xray_gpu_retrain_v3_threshold_tuning"),
    ]
    rows = []
    for label, metrics_folder, tuning_folder in names:
        metrics_path = PROJECT_ROOT / "outputs" / "training" / metrics_folder / "xray_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["best_metrics"]
        tuned_macro_f1 = metrics["macro_f1"]
        tuned_micro_f1 = metrics["micro_f1"]
        if tuning_folder:
            tuning_path = PROJECT_ROOT / "outputs" / "training" / tuning_folder / "xray_threshold_tuning_metrics.json"
            if tuning_path.exists():
                tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
                tuned_macro_f1 = tuning["tuned_metrics"]["macro_f1"]
                tuned_micro_f1 = tuning["tuned_metrics"]["micro_f1"]
        rows.append(
            {
                "label": label,
                "macro_auroc": metrics["macro_auroc"],
                "micro_auroc": metrics["micro_auroc"],
                "tuned_macro_f1": tuned_macro_f1,
                "tuned_micro_f1": tuned_micro_f1,
                "selected": "selected" in label,
            }
        )
    return rows


def figure_xray_version_comparison():
    rows = _load_xray_tuned_runs()
    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.bar(x - width, [row["macro_auroc"] for row in rows], width, label="Macro AUROC", color=BLUE, edgecolor=INK, linewidth=0.7)
    ax.bar(x, [row["micro_auroc"] for row in rows], width, label="Micro AUROC", color=TEAL, edgecolor=INK, linewidth=0.7)
    ax.bar(x + width, [row["tuned_macro_f1"] for row in rows], width, label="Tuned macro F1", color=ORANGE, edgecolor=INK, linewidth=0.7)
    ax.set_title("Chest X-ray Version Comparison", fontsize=15, fontweight="bold", color=INK)
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left")
    for idx, row in enumerate(rows):
        if row["selected"]:
            ax.text(idx, 0.875, "selected", ha="center", va="top", fontsize=9, color=RED, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig17_xray_version_comparison")


def figure_early_architecture_baseline_comparison():
    checkpoint_names = [
        ("Small CNN", "best_small_elasticnet_cnn.pt", ORANGE),
        ("ResNet50", "best_resnet50.pt", BLUE),
        ("DenseNet121", "best_densenet121.pt", TEAL),
    ]
    labels = []
    values = []
    colors = []
    epochs = []
    for label, filename, color in checkpoint_names:
        ckpt = torch.load(PROJECT_ROOT / "checkpoints" / "model1" / filename, map_location="cpu")
        labels.append(label)
        values.append(float(ckpt.get("best_score")))
        epochs.append(int(ckpt.get("epoch", 0)))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8.7, 5.4))
    bars = ax.bar(labels, values, color=colors, edgecolor=INK, linewidth=0.8)
    ax.set_title("Early Brain MRI Architecture Baseline Comparison", fontsize=15, fontweight="bold", color=INK)
    ax.set_ylabel("Stored checkpoint best_score")
    ax.set_ylim(0.65, 0.97)
    ax.grid(axis="y", alpha=0.25)
    for rect, value, epoch in zip(bars, values, epochs):
        ax.text(rect.get_x() + rect.get_width() / 2, value + 0.008, f"{value:.3f}\nepoch {epoch}", ha="center", va="bottom", fontsize=10)
    ax.text(
        0.5,
        -0.22,
        "This figure uses the stored best_score field in early checkpoints.\nFinal Model-1 uses later GPU final_v2/large_v2 runs.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color=RED,
    )
    fig.tight_layout()
    save(fig, "fig18_early_architecture_baseline_comparison")


def figure_v4_experimental_metrics():
    yolo_rows = list(csv.DictReader((PROJECT_ROOT / "outputs" / "v4_advanced_improvement" / "yolo_roi" / "yolov8n_roi_v4_pseudolabel_full" / "results.csv").open("r", encoding="utf-8")))
    yolo_last = yolo_rows[-1]
    yolo_metrics = {
        "Precision": float(yolo_last["metrics/precision(B)"]),
        "Recall": float(yolo_last["metrics/recall(B)"]),
        "mAP50": float(yolo_last["metrics/mAP50(B)"]),
        "mAP50-95": float(yolo_last["metrics/mAP50-95(B)"]),
    }
    ner = json.loads((PROJECT_ROOT / "outputs" / "v4_advanced_improvement" / "biobert_ner" / "ner_metrics_v4.json").read_text(encoding="utf-8"))
    ner_report = ner["entity_metrics"]["classification_report"]
    ner_metrics = {
        "Precision": float(ner["test_precision"]),
        "Recall": float(ner["test_recall"]),
        "Entity F1": float(ner["test_entity_f1"]),
        "UNIT F1\n(support=1)": float(ner_report["UNIT"]["f1-score"]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, title, metrics, color in [
        (axes[0], "YOLO ROI V4\npseudo-label validation", yolo_metrics, RED),
        (axes[1], "BERT NER V4\nweak-label validation", ner_metrics, TEAL),
    ]:
        names = list(metrics.keys())
        values = list(metrics.values())
        bars = ax.bar(names, values, color=color, edgecolor=INK, linewidth=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold", color=INK)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.25)
        for rect, value in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("V4 Experimental Metrics", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.5, -0.01, "Pseudo/weak-label metrics only; UNIT F1 is 0.000 with one supported test entity.", ha="center", fontsize=10, color=RED)
    fig.tight_layout()
    save(fig, "fig19_v4_experimental_metrics")


def figure_model1_final_vs_retrain():
    brain_final = json.loads((PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_final_v2" / "brain_metrics.json").read_text(encoding="utf-8"))["metrics"]
    brain_v3 = json.loads((PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_retrain_v3" / "brain_metrics.json").read_text(encoding="utf-8"))["metrics"]
    xray_final = json.loads((PROJECT_ROOT / "outputs" / "training" / "xray_gpu_large_v2_threshold_tuning" / "xray_threshold_tuning_metrics.json").read_text(encoding="utf-8"))
    xray_v3 = json.loads((PROJECT_ROOT / "outputs" / "training" / "xray_gpu_retrain_v3_threshold_tuning" / "xray_threshold_tuning_metrics.json").read_text(encoding="utf-8"))

    comparisons = [
        ("Brain MRI\nmacro F1", brain_final["macro_f1"], brain_v3["macro_f1"]),
        ("X-ray\nmacro AUROC", xray_final["default_metrics"]["macro_auroc"], xray_v3["default_metrics"]["macro_auroc"]),
        ("X-ray tuned\nmacro F1", xray_final["tuned_metrics"]["macro_f1"], xray_v3["tuned_metrics"]["macro_f1"]),
    ]
    labels = [item[0] for item in comparisons]
    final_values = [item[1] for item in comparisons]
    v3_values = [item[2] for item in comparisons]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    ax.bar(x - width / 2, final_values, width, label="Selected stable", color=TEAL, edgecolor=INK, linewidth=0.8)
    ax.bar(x + width / 2, v3_values, width, label="retrain_v3", color=ORANGE, edgecolor=INK, linewidth=0.8)
    ax.set_title("Selected Stable Model-1 vs retrain_v3", fontsize=15, fontweight="bold", color=INK)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    for xpos, stable_value, v3_value in zip(x, final_values, v3_values):
        ax.text(xpos - width / 2, stable_value + 0.025, f"{stable_value:.3f}", ha="center", fontsize=9)
        ax.text(xpos + width / 2, v3_value + 0.025, f"{v3_value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    save(fig, "fig20_model1_selected_vs_retrain_v3")


def write_catalog():
    items = [
        ("fig01_complete_system_architecture", "Complete Multimodal AI Assistant Architecture", "Chapter 4/System Design or defense overview slide"),
        ("fig02_model1_image_pipeline", "Model-1 Image Analysis Pipeline", "Model-1 methodology section"),
        ("fig03_brain_mri_classifier_architecture", "Brain MRI Classifier Architecture", "Brain MRI model subsection"),
        ("fig04_xray_classifier_architecture", "Chest X-ray Multi-label Classifier Architecture", "X-ray model subsection"),
        ("fig05_model2_document_pipeline", "Model-2 Document Understanding Pipeline", "Document processing subsection"),
        ("fig06_model3_rag_fusion_pipeline", "Model-3 Fusion and RAG Pipeline", "Fusion/RAG subsection"),
        ("fig07_v4_advanced_layer", "V4 Advanced Improvement Layer", "Advanced experiments or limitations section"),
        ("fig08_streamlit_demo_workflow", "Streamlit Demonstration Workflow", "Implementation/UI subsection or defense demo slide"),
        ("fig09_brain_mri_confusion_matrix", "Brain MRI Final_v2 Confusion Matrix", "Evaluation chapter"),
        ("fig10_xray_per_class_auroc", "Chest X-ray Large_v2 Per-class AUROC", "Evaluation chapter"),
        ("fig11_technical_validation_case_distribution", "100-case Technical Pipeline Validation Breakdown", "System validation subsection"),
        ("fig12_stable_vs_v4_document_comparison", "Stable vs V4 Document Pipeline Comparison", "Advanced experiment results"),
        ("fig13_report_storyline_roadmap", "Recommended Report Storyline", "Planning only; do not include unless useful"),
        ("fig14_model_component_output_success", "Model-1 vs Model-2 vs Model-3 Technical Output Success", "Evaluation/system validation subsection"),
        ("fig15_best_version_claim_status_summary", "Best Available Versions and Claim Status", "Results summary or defense claim slide"),
        ("fig16_brain_mri_run_comparison", "Brain MRI Training Run Comparison", "Model-1 evaluation subsection"),
        ("fig17_xray_version_comparison", "Chest X-ray Version Comparison", "Model-1 evaluation subsection"),
        ("fig18_early_architecture_baseline_comparison", "Early Brain MRI Architecture Baseline Comparison", "Architecture comparison subsection, with caveat"),
        ("fig19_v4_experimental_metrics", "V4 Experimental Metrics", "Advanced V4 experiment results"),
        ("fig20_model1_selected_vs_retrain_v3", "Selected Stable Model-1 vs retrain_v3", "Final model selection discussion"),
    ]
    lines = ["# Thesis Figure Catalog", ""]
    for i, (base, title, placement) in enumerate(items, start=1):
        lines.append(f"{i}. {title}")
        lines.append(f"   - PNG: `{base}.png`")
        lines.append(f"   - SVG: `{base}.svg`")
        lines.append(f"   - Suggested placement: {placement}")
        lines.append("")
    (OUTPUT_DIR / "FIGURE_CATALOG.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_system_architecture()
    figure_model1_pipeline()
    figure_brain_architecture()
    figure_xray_architecture()
    figure_model2_pipeline()
    figure_model3_pipeline()
    figure_v4_layer()
    figure_streamlit_workflow()
    figure_brain_confusion_matrix()
    figure_xray_auroc()
    figure_case_distribution()
    figure_stable_v4_comparison()
    figure_report_roadmap()
    figure_component_output_success()
    figure_best_version_summary()
    figure_brain_run_comparison()
    figure_xray_version_comparison()
    figure_early_architecture_baseline_comparison()
    figure_v4_experimental_metrics()
    figure_model1_final_vs_retrain()
    write_catalog()
    print(f"Wrote thesis figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
