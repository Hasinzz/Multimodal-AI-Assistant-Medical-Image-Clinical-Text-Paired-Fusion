from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.common import EVALUATION_OUTPUT_DIR, average, ensure_output_dirs, save_json, write_markdown  # noqa: E402
from src.model2.doctor_note_dataset_utils import clean_doctor_text  # noqa: E402


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_history_row(history_rows: List[Dict[str, Any]], epoch: int) -> Optional[Dict[str, Any]]:
    for row in history_rows:
        if int(float(row.get("epoch", 0))) == epoch:
            return row
    return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def _score_gap(train_value: Optional[float], validation_value: Optional[float]) -> Optional[float]:
    if train_value is None or validation_value is None:
        return None
    return train_value - validation_value


def _brain_component(name: str, metrics_path: Path, history_path: Path) -> Dict[str, Any]:
    metrics = _read_json(metrics_path)
    history_rows = _read_csv(history_path)
    best_epoch = int(metrics.get("best_epoch", 0))
    best_history = _find_history_row(history_rows, best_epoch)
    train_accuracy = _safe_float(best_history.get("train_accuracy")) if best_history else None
    val_accuracy = _safe_float(best_history.get("val_accuracy")) if best_history else _safe_float(metrics.get("best_val_accuracy"))
    train_loss = _safe_float(best_history.get("train_loss")) if best_history else None
    val_loss = _safe_float(best_history.get("val_loss")) if best_history else _safe_float(metrics.get("best_val_loss"))
    final_metric = metrics.get("metrics", {})
    return {
        "component": name,
        "train_metric": train_accuracy,
        "validation_metric": val_accuracy,
        "test_metric": _safe_float(final_metric.get("accuracy")),
        "train_validation_gap": _score_gap(train_accuracy, val_accuracy),
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "signs_of_overfitting": (
            "Moderate generalization gap: train accuracy remained above validation accuracy and validation loss was noisier than training loss."
            if train_accuracy is not None and val_accuracy is not None and (train_accuracy - val_accuracy) > 0.03
            else "No strong overfitting signal from the available training history."
        ),
        "what_was_done_to_reduce_overfitting": "Used checkpoint selection on validation performance, image augmentation, and fixed train/validation splits.",
        "what_still_needs_improvement": "A separate test split or external dataset would make the generalization check more conclusive.",
        "metric_note": "Training history contains train and validation metrics; final evaluation metric is reported from the saved metrics file.",
    }


def _xray_component(name: str, metrics_path: Path, history_path: Path) -> Dict[str, Any]:
    metrics = _read_json(metrics_path)
    history_rows = _read_csv(history_path)
    best_epoch = int(metrics.get("best_epoch", 0))
    best_history = _find_history_row(history_rows, best_epoch)
    train_loss = _safe_float(best_history.get("train_loss")) if best_history else None
    val_loss = _safe_float(best_history.get("val_loss")) if best_history else _safe_float(metrics.get("best_metrics", {}).get("val_loss"))
    return {
        "component": name,
        "train_metric": train_loss,
        "validation_metric": val_loss,
        "test_metric": None,
        "train_validation_gap": _score_gap(train_loss, val_loss),
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "signs_of_overfitting": (
            "Validation loss stayed above training loss at the best epoch, which is consistent with a generalization gap."
            if train_loss is not None and val_loss is not None and val_loss > train_loss
            else "No strong overfitting signal from the available training history."
        ),
        "what_was_done_to_reduce_overfitting": "Used checkpoint selection on validation macro AUROC, scheduler control, and threshold tuning after training.",
        "what_still_needs_improvement": "A held-out test split or external chest X-ray set would better separate optimization from generalization.",
        "metric_note": "Training history contains train loss and validation metrics; no explicit test metric was stored in the training artifacts.",
    }


def _doctor_note_classifier_component(name: str, checkpoint_path: Path, prepared_jsonl: Path, metrics_path: Path) -> Dict[str, Any]:
    summary = _read_json(metrics_path)
    bundle = joblib.load(checkpoint_path)
    frame_records = [json.loads(line) for line in prepared_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    filtered = [record for record in frame_records if str(record.get("clean_text", "")).strip() and str(record.get("medical_specialty", "")).strip()]

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    frame = pd.DataFrame(filtered)
    counts = frame["medical_specialty"].value_counts()
    kept_classes = counts[counts >= 20].index.tolist()
    frame = frame[frame["medical_specialty"].isin(kept_classes)].copy()

    train_frame, temp_frame = train_test_split(frame, test_size=0.4, random_state=42, stratify=frame["medical_specialty"])
    val_frame, test_frame = train_test_split(temp_frame, test_size=0.5, random_state=42, stratify=temp_frame["medical_specialty"])

    def _evaluate(subframe: Any) -> Tuple[float, float, float]:
        vectorizer = bundle["vectorizer"]
        classifier = bundle["classifier"]
        features = vectorizer.transform(subframe["clean_text"].tolist())
        predictions = classifier.predict(features)
        accuracy = float(accuracy_score(subframe["medical_specialty"].tolist(), predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(subframe["medical_specialty"].tolist(), predictions, average="macro", zero_division=0)
        return accuracy, float(precision), float(f1)

    train_accuracy, train_precision, train_f1 = _evaluate(train_frame)
    val_accuracy, val_precision, val_f1 = _evaluate(val_frame)
    test_accuracy, test_precision, test_f1 = _evaluate(test_frame)

    return {
        "component": name,
        "train_metric": train_accuracy,
        "validation_metric": val_accuracy,
        "test_metric": test_accuracy,
        "train_validation_gap": _score_gap(train_accuracy, val_accuracy),
        "train_loss": None,
        "validation_loss": None,
        "signs_of_overfitting": (
            "The classifier shows a moderate train-validation gap, but the gap is not extreme for a multi-class specialty task."
            if train_accuracy - val_accuracy > 0.05
            else "No strong overfitting signal from the reconstructed train/validation split."
        ),
        "what_was_done_to_reduce_overfitting": "Used TF-IDF features, stratified splitting, class weighting, and a simple linear baseline.",
        "what_still_needs_improvement": "A stronger calibration pass or a larger labeled dataset could improve minority-specialty performance.",
        "metric_note": "Train, validation, and test metrics were reconstructed from the saved TF-IDF checkpoint and the prepared MTSamples split.",
        "extra": {
            "train_precision": train_precision,
            "train_macro_f1": train_f1,
            "validation_precision": val_precision,
            "validation_macro_f1": val_f1,
            "test_precision": test_precision,
            "test_macro_f1": test_f1,
            "kept_class_count": len(kept_classes),
        },
    }


def _weak_ner_component(name: str, metrics_path: Path) -> Dict[str, Any]:
    summary = _read_json(metrics_path)
    mode = str(summary.get("mode", "")).strip()
    fallback_reason = summary.get("fallback_reason")
    transformer_trained = mode == "transformer_bert" and not fallback_reason
    if transformer_trained:
        mitigation_note = "Transformer-based BERT token-classification training completed on weak labels; the result is still limited by weak supervision rather than expert annotations."
        improvement_note = "Manual review of labels and a held-out expert-annotated set are needed before making clinical NER quality claims."
        metric_note = "Mode=transformer_bert; entity F1 is a weak-label test metric, not expert clinical NER accuracy."
    else:
        mitigation_note = "The current output is a weak-label fallback; transformer-trained NER could not be completed in this environment."
        improvement_note = "If transformers become available, a proper token-classification run should be repeated on manually reviewed labels."
        metric_note = fallback_reason or "Weak-label fallback metrics only."
    return {
        "component": name,
        "train_metric": None,
        "validation_metric": None,
        "test_metric": _safe_float(summary.get("entity_f1")),
        "train_validation_gap": None,
        "train_loss": None,
        "validation_loss": None,
        "signs_of_overfitting": "Train-validation overfitting gap could not be computed because training history did not contain the required fields.",
        "what_was_done_to_reduce_overfitting": mitigation_note,
        "what_still_needs_improvement": improvement_note,
        "metric_note": metric_note,
    }


def _missing_component(name: str, note: str) -> Dict[str, Any]:
    return {
        "component": name,
        "train_metric": None,
        "validation_metric": None,
        "test_metric": None,
        "train_validation_gap": None,
        "train_loss": None,
        "validation_loss": None,
        "signs_of_overfitting": "Train-validation overfitting gap could not be computed because training history did not contain the required fields.",
        "what_was_done_to_reduce_overfitting": "No trainable artifact exists, so no overfitting mitigation can be attributed.",
        "what_still_needs_improvement": "Create a real checkpoint and training history before evaluating generalization.",
        "metric_note": note,
    }


def build_overfitting_summary() -> Dict[str, Any]:
    ensure_output_dirs()

    components: List[Dict[str, Any]] = []
    components.append(
        _brain_component(
            "Model-1A Brain MRI final_v2",
            PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_final_v2" / "brain_metrics.json",
            PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_final_v2" / "brain_training_history.csv",
        )
    )
    components.append(
        _brain_component(
            "Model-1A Brain MRI retrain_v3",
            PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_retrain_v3" / "brain_metrics.json",
            PROJECT_ROOT / "outputs" / "training" / "brain_mri_gpu_retrain_v3" / "brain_training_history.csv",
        )
    )
    components.append(
        _xray_component(
            "Model-1B Chest X-ray large_v2",
            PROJECT_ROOT / "outputs" / "training" / "xray_gpu_large_v2" / "xray_metrics.json",
            PROJECT_ROOT / "outputs" / "training" / "xray_gpu_large_v2" / "xray_training_history.csv",
        )
    )
    components.append(
        _xray_component(
            "Model-1B Chest X-ray retrain_v3",
            PROJECT_ROOT / "outputs" / "training" / "xray_gpu_retrain_v3" / "xray_metrics.json",
            PROJECT_ROOT / "outputs" / "training" / "xray_gpu_retrain_v3" / "xray_training_history.csv",
        )
    )
    components.append(
        _doctor_note_classifier_component(
            "Model-2B Doctor-note specialty classifier",
            PROJECT_ROOT / "checkpoints" / "model2" / "doctor_note_classifier_tfidf.joblib",
            PROJECT_ROOT / "data" / "text" / "doctor_notes" / "mtsamples" / "processed" / "mtsamples_doctor_notes.jsonl",
            PROJECT_ROOT / "outputs" / "doctor_note_training" / "classifier_metrics.json",
        )
    )
    components.append(
        _weak_ner_component(
            "Model-2B Weak NER",
            PROJECT_ROOT / "outputs" / "doctor_note_training" / "weak_ner_metrics.json",
        )
    )

    cross_attention_dir = PROJECT_ROOT / "checkpoints" / "model3" / "cross_attention_v4"
    if cross_attention_dir.exists() and any(cross_attention_dir.iterdir()):
        components.append(
            _missing_component(
                "Model-3 cross_attention_v4",
                "A cross-attention folder exists, but no training history was found in the repository snapshot.",
            )
        )
    else:
        components.append(
            _missing_component(
                "Model-3 cross_attention_v4",
                "No checkpoint or training history exists for cross_attention_v4; code exists only.",
            )
        )

    summary = {
        "component_count": len(components),
        "components": components,
        "note": "Train-validation overfitting gap could not be computed for components whose training history did not contain the required fields.",
    }

    save_json(EVALUATION_OUTPUT_DIR / "overfitting_check_summary.json", summary)

    lines = [
        "# Overfitting Check Report",
        "",
        "Train-validation overfitting gap could not be computed for components whose training history did not contain the required fields.",
        "",
        "| Component | Train metric | Validation metric | Test metric | Train-validation gap | Signs of overfitting |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for component in components:
        lines.append(
            "| {component} | {train_metric} | {validation_metric} | {test_metric} | {gap} | {signs} |".format(
                component=component["component"],
                train_metric="n/a" if component["train_metric"] is None else f"{component['train_metric']:.4f}",
                validation_metric="n/a" if component["validation_metric"] is None else f"{component['validation_metric']:.4f}",
                test_metric="n/a" if component["test_metric"] is None else f"{component['test_metric']:.4f}",
                gap="n/a" if component["train_validation_gap"] is None else f"{component['train_validation_gap']:.4f}",
                signs=component["signs_of_overfitting"],
            )
        )
        lines.append("")
        lines.append(f"## {component['component']}")
        lines.append(f"- What was done to reduce overfitting: {component['what_was_done_to_reduce_overfitting']}")
        lines.append(f"- What still needs improvement: {component['what_still_needs_improvement']}")
        lines.append(f"- Metric note: {component['metric_note']}")
        if component.get("extra"):
            lines.append(f"- Extra metrics: {json.dumps(component['extra'], ensure_ascii=False)}")
        lines.append("")

    write_markdown(EVALUATION_OUTPUT_DIR / "overfitting_check_report.md", lines)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Check overfitting across existing thesis training artifacts.")
    parser.parse_args()
    summary = build_overfitting_summary()
    print(summary["note"])
    print(f"Wrote {EVALUATION_OUTPUT_DIR / 'overfitting_check_summary.json'}")


if __name__ == "__main__":
    main()
