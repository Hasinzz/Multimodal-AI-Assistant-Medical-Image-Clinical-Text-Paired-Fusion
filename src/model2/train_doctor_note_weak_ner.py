from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model2.doctor_note_dataset_utils import extract_weak_entity_spans  # noqa: E402


INPUT_JSONL = PROJECT_ROOT / "data" / "text" / "doctor_notes" / "mtsamples" / "processed" / "mtsamples_weak_entities.jsonl"
FALLBACK_INPUT_JSONL = PROJECT_ROOT / "data" / "text" / "doctor_notes" / "mtsamples" / "processed" / "mtsamples_doctor_notes.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "model2" / "doctor_note_weak_ner"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "doctor_note_training"
FIGURE_PATH = PROJECT_ROOT / "outputs" / "thesis_figures" / "doctor_note_weak_ner_results.png"

LABEL_CATEGORIES = ["SYMPTOM", "HISTORY", "MEDICATION", "ALLERGY", "TEST", "DIAGNOSIS_OR_CONCERN", "URGENCY"]
BIO_LABELS = ["O"] + [f"B-{category}" for category in LABEL_CATEGORIES] + [f"I-{category}" for category in LABEL_CATEGORIES]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _load_records(path: Path = INPUT_JSONL) -> List[Dict[str, Any]]:
    if path.exists():
        return _read_jsonl(path)

    fallback_records: List[Dict[str, Any]] = []
    if FALLBACK_INPUT_JSONL.exists():
        for record in _read_jsonl(FALLBACK_INPUT_JSONL):
            weak_entity_spans = extract_weak_entity_spans(str(record.get("clean_text", "")))
            weak_entities = defaultdict(list)
            for span in weak_entity_spans:
                weak_entities[span["label"]].append(span["text"])
            fallback_records.append(
                {
                    **record,
                    "weak_entities": dict(weak_entities),
                    "weak_entity_spans": weak_entity_spans,
                }
            )
    return fallback_records


def _category_from_label(label: str) -> Optional[str]:
    if label == "O":
        return None
    return label.split("-", 1)[1] if "-" in label else None


def _assign_word_labels(text: str, spans: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[str, int, int]]]:
    import re

    words = [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]
    labels = ["O"] * len(words)

    span_items = []
    for span in spans:
        category = span.get("label")
        start = int(span.get("start", 0))
        end = int(span.get("end", 0))
        if category not in LABEL_CATEGORIES:
            continue
        span_items.append((start, end, category))

    for start, end, category in sorted(span_items, key=lambda item: (item[0], item[1])):
        overlapping = [index for index, (_, word_start, word_end) in enumerate(words) if not (word_end <= start or word_start >= end)]
        if not overlapping:
            continue
        first = overlapping[0]
        if labels[first] == "O":
            labels[first] = f"B-{category}"
        for index in overlapping[1:]:
            if labels[index] == "O":
                labels[index] = f"I-{category}"

    return labels, words


def _entity_spans_from_labels(words: Sequence[Tuple[str, int, int]], labels: Sequence[str]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    current_category: Optional[str] = None
    current_start: Optional[int] = None
    current_end: Optional[int] = None

    def _flush() -> None:
        nonlocal current_category, current_start, current_end
        if current_category is None or current_start is None or current_end is None:
            return
        spans.append({"label": current_category, "start": current_start, "end": current_end})
        current_category = None
        current_start = None
        current_end = None

    for (word, start, end), label in zip(words, labels):
        category = _category_from_label(label)
        if category is None:
            _flush()
            continue

        if label.startswith("B-") or category != current_category or current_start is None:
            _flush()
            current_category = category
            current_start = start
            current_end = end
        else:
            current_end = end

    _flush()
    return spans


def _token_metrics(true_labels: Sequence[str], pred_labels: Sequence[str]) -> Dict[str, float]:
    total = len(true_labels)
    correct = sum(1 for gold, pred in zip(true_labels, pred_labels) if gold == pred)
    return {"token_accuracy": float(correct / total) if total else 0.0}


def _span_metrics(true_spans: Sequence[Sequence[Dict[str, Any]]], pred_spans: Sequence[Sequence[Dict[str, Any]]]) -> Tuple[float, float, float, Dict[str, float]]:
    true_total = 0
    pred_total = 0
    correct_total = 0
    true_by_label = Counter()
    pred_by_label = Counter()
    correct_by_label = Counter()

    for gold, pred in zip(true_spans, pred_spans):
        gold_set = {(item["label"], int(item["start"]), int(item["end"])) for item in gold}
        pred_set = {(item["label"], int(item["start"]), int(item["end"])) for item in pred}
        true_total += len(gold_set)
        pred_total += len(pred_set)
        correct_total += len(gold_set & pred_set)
        for label, _, _ in gold_set:
            true_by_label[label] += 1
        for label, _, _ in pred_set:
            pred_by_label[label] += 1
        for label, _, _ in gold_set & pred_set:
            correct_by_label[label] += 1

    precision = correct_total / pred_total if pred_total else 0.0
    recall = correct_total / true_total if true_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    per_entity_f1 = {}
    for label in LABEL_CATEGORIES:
        tp = correct_by_label[label]
        fp = pred_by_label[label] - tp
        fn = true_by_label[label] - tp
        label_precision = tp / (tp + fp) if (tp + fp) else 0.0
        label_recall = tp / (tp + fn) if (tp + fn) else 0.0
        per_entity_f1[label] = (2 * label_precision * label_recall / (label_precision + label_recall)) if (label_precision + label_recall) else 0.0
    return precision, recall, f1, per_entity_f1


def _build_examples(records: Sequence[Dict[str, Any]], max_examples: int) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for record in records[:max_examples]:
        text = str(record.get("clean_text", ""))
        spans = record.get("weak_entity_spans") or extract_weak_entity_spans(text)
        labels, words = _assign_word_labels(text, spans)
        if not words:
            continue
        examples.append({"text": text, "words": words, "labels": labels, "spans": spans})
    return examples


def _prepare_for_transformers(examples: Sequence[Dict[str, Any]], tokenizer, label_to_id: Dict[str, int], max_length: int = 256):
    features = []
    for example in examples:
        words = [word for word, _, _ in example["words"]]
        word_labels = example["labels"]
        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        word_ids = tokenized.word_ids()
        labels = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
                continue
            label = word_labels[word_id]
            if word_id != previous_word_id:
                labels.append(label_to_id[label])
            else:
                if label == "O":
                    labels.append(label_to_id["O"])
                else:
                    category = label.split("-", 1)[1]
                    labels.append(label_to_id.get(f"I-{category}", label_to_id[label]))
            previous_word_id = word_id
        tokenized["labels"] = labels
        features.append(tokenized)
    return features


def _predict_with_model(model, tokenizer, examples: Sequence[Dict[str, Any]]) -> Tuple[List[List[str]], List[List[Dict[str, Any]]]]:
    import torch

    predicted_labels: List[List[str]] = []
    predicted_spans: List[List[Dict[str, Any]]] = []
    device = next(model.parameters()).device
    model.eval()

    for example in examples:
        word_tuples = example["words"]
        words = [word for word, _, _ in word_tuples]
        encoded = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        word_ids = encoded.word_ids()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0]
        token_predictions = logits.argmax(dim=-1).tolist()

        word_level_labels = ["O"] * len(words)
        assigned_words = set()
        for token_index, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            predicted_label = BIO_LABELS[token_predictions[token_index]]
            if word_id not in assigned_words:
                word_level_labels[word_id] = predicted_label
                assigned_words.add(word_id)
            elif word_level_labels[word_id] == "O" and predicted_label != "O":
                word_level_labels[word_id] = predicted_label

        predicted_labels.append(word_level_labels)
        predicted_spans.append(_entity_spans_from_labels(word_tuples, word_level_labels))

    return predicted_labels, predicted_spans


def _predict_rule_based(examples: Sequence[Dict[str, Any]]) -> Tuple[List[List[str]], List[List[Dict[str, Any]]]]:
    pred_labels: List[List[str]] = []
    pred_spans: List[List[Dict[str, Any]]] = []
    for example in examples:
        spans = extract_weak_entity_spans(example["text"])
        labels, words = _assign_word_labels(example["text"], spans)
        pred_labels.append(labels)
        pred_spans.append(_entity_spans_from_labels(words, labels))
    return pred_labels, pred_spans


def train_weak_ner(input_jsonl: Path = INPUT_JSONL, max_examples: int = 300) -> Dict[str, Any]:
    records = _load_records(input_jsonl)
    if not records:
        raise FileNotFoundError("No weak-label doctor-note records were found.")

    examples = _build_examples(records, max_examples=max_examples)
    if not examples:
        raise ValueError("No usable examples were available for weak NER training.")

    train_cut = max(1, int(len(examples) * 0.7))
    val_cut = max(train_cut + 1, int(len(examples) * 0.85))
    train_examples = examples[:train_cut]
    validation_examples = examples[train_cut:val_cut]
    test_examples = examples[val_cut:] or examples[-max(1, len(examples) // 5):]

    transformer_error = None
    trained_mode = False
    token_accuracy = 0.0
    entity_precision = 0.0
    entity_recall = 0.0
    entity_f1 = 0.0
    per_entity_f1: Dict[str, float] = {label: 0.0 for label in LABEL_CATEGORIES}

    try:
        import torch
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
            Trainer,
            TrainingArguments,
        )

        model_name = "bert-base-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(BIO_LABELS),
            id2label={index: label for index, label in enumerate(BIO_LABELS)},
            label2id={label: index for index, label in enumerate(BIO_LABELS)},
            local_files_only=True,
        )

        label_to_id = {label: index for index, label in enumerate(BIO_LABELS)}
        train_features = _prepare_for_transformers(train_examples, tokenizer, label_to_id)
        validation_features = _prepare_for_transformers(validation_examples, tokenizer, label_to_id)
        test_features = _prepare_for_transformers(test_examples, tokenizer, label_to_id)

        class TokenDataset:
            def __init__(self, features):
                self.features = features

            def __len__(self):
                return len(self.features)

            def __getitem__(self, index):
                item = self.features[index]
                return {key: value for key, value in item.items()}

        training_kwargs = dict(
            output_dir=str(CHECKPOINT_DIR),
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-5,
            weight_decay=0.01,
            logging_dir=str(OUTPUT_DIR / "logs"),
            logging_steps=50,
            save_strategy="epoch",
            report_to="none",
            load_best_model_at_end=False,
        )
        training_signature = inspect.signature(TrainingArguments.__init__)
        if "eval_strategy" in training_signature.parameters:
            training_kwargs["eval_strategy"] = "epoch"
        elif "evaluation_strategy" in training_signature.parameters:
            training_kwargs["evaluation_strategy"] = "epoch"

        if "use_cpu" in training_signature.parameters:
            training_kwargs["use_cpu"] = not torch.cuda.is_available()
        elif "no_cuda" in training_signature.parameters:
            training_kwargs["no_cuda"] = not torch.cuda.is_available()

        training_args = TrainingArguments(**training_kwargs)

        trainer_kwargs = dict(
            model=model,
            args=training_args,
            train_dataset=TokenDataset(train_features),
            eval_dataset=TokenDataset(validation_features),
            data_collator=DataCollatorForTokenClassification(tokenizer),
        )
        trainer_signature = inspect.signature(Trainer.__init__)
        if "processing_class" in trainer_signature.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in trainer_signature.parameters:
            trainer_kwargs["tokenizer"] = tokenizer

        trainer = Trainer(**trainer_kwargs)

        trainer.train()
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(CHECKPOINT_DIR))
        tokenizer.save_pretrained(str(CHECKPOINT_DIR))
        (CHECKPOINT_DIR / "label_list.json").write_text(json.dumps(BIO_LABELS, indent=2), encoding="utf-8")

        pred_label_sequences, pred_span_sequences = _predict_with_model(model, tokenizer, test_examples)
        true_label_sequences = [example["labels"] for example in test_examples]
        true_span_sequences = [example["spans"] for example in test_examples]
        flat_true = [label for sequence in true_label_sequences for label in sequence]
        flat_pred = [label for sequence in pred_label_sequences for label in sequence]
        token_accuracy = float(sum(1 for gold, pred in zip(flat_true, flat_pred) if gold == pred) / len(flat_true)) if flat_true else 0.0
        entity_precision, entity_recall, entity_f1, per_entity_f1 = _span_metrics(true_span_sequences, pred_span_sequences)
        trained_mode = True
    except Exception as error:  # pragma: no cover - depends on local HF availability
        transformer_error = f"Transformer training unavailable: {error}"
        true_label_sequences = [example["labels"] for example in test_examples]
        pred_label_sequences, pred_span_sequences = _predict_rule_based(test_examples)
        true_span_sequences = [example["spans"] for example in test_examples]
        flat_true = [label for sequence in true_label_sequences for label in sequence]
        flat_pred = [label for sequence in pred_label_sequences for label in sequence]
        token_accuracy = float(sum(1 for gold, pred in zip(flat_true, flat_pred) if gold == pred) / len(flat_true)) if flat_true else 0.0
        entity_precision, entity_recall, entity_f1, per_entity_f1 = _span_metrics(true_span_sequences, pred_span_sequences)

    metrics = {
        "mode": "transformer_bert" if trained_mode else "fallback_rule_based",
        "fallback_reason": transformer_error,
        "notes_processed": int(len(records)),
        "examples_used": int(len(examples)),
        "train_examples": int(len(train_examples)),
        "validation_examples": int(len(validation_examples)),
        "test_examples": int(len(test_examples)),
        "token_accuracy": token_accuracy,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "per_entity_f1": per_entity_f1,
        "label_list": BIO_LABELS,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "note": "The NER model is trained and evaluated against weak labels, not expert annotations. Therefore, the result measures weak-label learning, not clinical NER quality.",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "weak_ner_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Weak Doctor-Note NER Report",
        "",
        f"- Mode: {metrics['mode']}",
        f"- Notes processed: {metrics['notes_processed']}",
        f"- Examples used: {metrics['examples_used']}",
        f"- Train examples: {metrics['train_examples']}",
        f"- Validation examples: {metrics['validation_examples']}",
        f"- Test examples: {metrics['test_examples']}",
        f"- Token accuracy: {metrics['token_accuracy']:.3f}",
        f"- Entity precision: {metrics['entity_precision']:.3f}",
        f"- Entity recall: {metrics['entity_recall']:.3f}",
        f"- Entity F1: {metrics['entity_f1']:.3f}",
        "",
        "## Per-Entity F1",
    ]

    for label, score in metrics["per_entity_f1"].items():
        report_lines.append(f"- {label}: {score:.3f}")

    report_lines.extend([
        "",
        "## Limitation",
        "The NER model is trained and evaluated against weak labels, not expert annotations. Therefore, the result measures weak-label learning, not clinical NER quality.",
    ])
    if transformer_error:
        report_lines.extend([
            "",
            "## Fallback",
            transformer_error,
        ])

    (OUTPUT_DIR / "weak_ner_report.md").write_text("\n".join(report_lines).rstrip() + "\n", encoding="utf-8")

    figure, axis = plt.subplots(figsize=(9, 5))
    metric_names = ["token_accuracy", "entity_precision", "entity_recall", "entity_f1"]
    metric_values = [metrics[name] for name in metric_names]
    axis.bar(metric_names, metric_values, color="#5b8c5a")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Doctor-Note Weak NER Results")
    axis.set_ylabel("Score")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_PATH, dpi=200)
    plt.close(figure)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weak-label NER for MTSamples doctor notes.")
    parser.add_argument("--input-jsonl", type=Path, default=INPUT_JSONL)
    parser.add_argument("--max-examples", type=int, default=300)
    args = parser.parse_args()
    metrics = train_weak_ner(args.input_jsonl, args.max_examples)
    print(metrics["note"])
    print(f"Wrote {metrics['checkpoint_dir']}")


if __name__ == "__main__":
    main()
