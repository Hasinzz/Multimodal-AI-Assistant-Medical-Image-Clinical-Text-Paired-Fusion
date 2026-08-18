from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EVALUATION_OUTPUT_DIR = OUTPUT_ROOT / "evaluation"
THESIS_FIGURES_DIR = OUTPUT_ROOT / "thesis_figures"


def ensure_output_dirs() -> None:
    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def save_bar_chart(path: Path, title: str, labels: Sequence[str], values: Sequence[float], ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, values, color="#2a6f97")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_dual_bar_chart(
    path: Path,
    title: str,
    labels: Sequence[str],
    series_a: Sequence[float],
    series_b: Sequence[float],
    label_a: str,
    label_b: str,
    ylabel: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    x_positions = range(len(labels))
    width = 0.38
    axis.bar([pos - width / 2 for pos in x_positions], series_a, width=width, label=label_a, color="#2a6f97")
    axis.bar([pos + width / 2 for pos in x_positions], series_b, width=width, label=label_b, color="#e76f51")
    axis.set_xticks(list(x_positions))
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
