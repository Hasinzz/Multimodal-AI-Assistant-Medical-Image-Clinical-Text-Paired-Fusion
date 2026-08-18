from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


INPUT_JSONL = PROJECT_ROOT / "data" / "text" / "doctor_notes" / "mtsamples" / "processed" / "mtsamples_doctor_notes.jsonl"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "model2" / "doctor_note_classifier_tfidf.joblib"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "doctor_note_training"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "thesis_figures" / "doctor_note_specialty_classifier_results.png"
REPORT_CSV_PATH = OUTPUT_DIR / "classifier_classification_report.csv"
CONFUSION_CSV_PATH = OUTPUT_DIR / "classifier_confusion_matrix.csv"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _build_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prepared doctor-note JSONL not found: {path}")
    frame = pd.DataFrame(_read_jsonl(path))
    if frame.empty:
        raise ValueError("No doctor-note records were found in the prepared JSONL.")
    frame["clean_text"] = frame.get("clean_text", "").fillna("").astype(str)
    frame["medical_specialty"] = frame.get("medical_specialty", "").fillna("").astype(str)
    frame = frame[(frame["clean_text"].str.strip() != "") & (frame["medical_specialty"].str.strip() != "")].copy()
    return frame


def _split_data(frame: pd.DataFrame, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=0.4,
        random_state=random_state,
        stratify=frame["medical_specialty"],
    )
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_frame["medical_specialty"],
    )
    return train_frame, val_frame, test_frame


def _metrics(y_true, y_pred) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
    }


def train_classifier(input_jsonl: Path = INPUT_JSONL, min_class_samples: int = 20) -> Dict[str, Any]:
    frame = _build_frame(input_jsonl)
    specialty_counts = frame["medical_specialty"].value_counts()
    kept_classes = specialty_counts[specialty_counts >= min_class_samples].index.tolist()
    filtered = frame[frame["medical_specialty"].isin(kept_classes)].copy()

    if filtered.empty:
        raise ValueError("No MTSamples specialties met the minimum sample threshold for classification.")

    train_frame, val_frame, test_frame = _split_data(filtered)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
    )
    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")

    train_matrix = vectorizer.fit_transform(train_frame["clean_text"].tolist())
    classifier.fit(train_matrix, train_frame["medical_specialty"].tolist())

    val_predictions = classifier.predict(vectorizer.transform(val_frame["clean_text"].tolist()))
    test_predictions = classifier.predict(vectorizer.transform(test_frame["clean_text"].tolist()))

    val_metrics = _metrics(val_frame["medical_specialty"].tolist(), val_predictions)
    test_metrics = _metrics(test_frame["medical_specialty"].tolist(), test_predictions)

    class_names = sorted(filtered["medical_specialty"].unique().tolist())
    confusion = confusion_matrix(test_frame["medical_specialty"].tolist(), test_predictions, labels=class_names)
    report = classification_report(
        test_frame["medical_specialty"].tolist(),
        test_predictions,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
            report_rows.append(row)
        else:
            report_rows.append({"label": label, "value": values})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(REPORT_CSV_PATH, index=False)
    pd.DataFrame(confusion, index=class_names, columns=class_names).to_csv(CONFUSION_CSV_PATH)

    checkpoint_payload = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "classes": class_names,
        "kept_classes": kept_classes,
        "min_class_samples": min_class_samples,
        "class_counts": specialty_counts[specialty_counts >= min_class_samples].to_dict(),
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(checkpoint_payload, CHECKPOINT_PATH)

    metrics = {
        "input_jsonl": str(input_jsonl),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "usable_rows": int(len(filtered)),
        "kept_class_count": int(len(kept_classes)),
        "kept_classes": kept_classes,
        "train_rows": int(len(train_frame)),
        "validation_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "confusion_matrix": confusion.tolist(),
        "classification_report_labels": class_names,
        "note": "This classifier evaluates whether the doctor-note text branch can learn clinical text patterns from MTSamples metadata. It does not diagnose MRI or X-ray diseases.",
    }

    (OUTPUT_DIR / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Doctor-Note Specialty Classifier Report",
        "",
        f"- Usable rows: {metrics['usable_rows']}",
        f"- Kept classes: {metrics['kept_class_count']}",
        f"- Train rows: {metrics['train_rows']}",
        f"- Validation rows: {metrics['validation_rows']}",
        f"- Test rows: {metrics['test_rows']}",
        "",
        "## Test Metrics",
        f"- Accuracy: {metrics['test_metrics']['accuracy']:.3f}",
        f"- Macro precision: {metrics['test_metrics']['macro_precision']:.3f}",
        f"- Macro recall: {metrics['test_metrics']['macro_recall']:.3f}",
        f"- Macro F1: {metrics['test_metrics']['macro_f1']:.3f}",
        f"- Weighted F1: {metrics['test_metrics']['weighted_f1']:.3f}",
        "",
        "This classifier evaluates whether the doctor-note text branch can learn clinical text patterns from MTSamples metadata. It does not diagnose MRI or X-ray diseases.",
    ]
    (OUTPUT_DIR / "classifier_report.md").write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(9, 5))
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    metric_values = [metrics["test_metrics"][name] for name in metric_names]
    axis.bar(metric_names, metric_values, color="#2a6f97")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Doctor-Note Specialty Classifier Results")
    axis.set_ylabel("Score")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=200)
    plt.close(figure)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a doctor-note specialty classifier on MTSamples.")
    parser.add_argument("--input-jsonl", type=Path, default=INPUT_JSONL)
    parser.add_argument("--min-class-samples", type=int, default=20)
    args = parser.parse_args()
    metrics = train_classifier(args.input_jsonl, args.min_class_samples)
    print(metrics["note"])
    print(f"Wrote {metrics['checkpoint_path']}")


if __name__ == "__main__":
    main()
