#!/usr/bin/env python
"""Phase 3A - validation evaluation of a saved CNN (analysis only).

Loads the trained model from data/models/<run-name>.keras (--run-name is
required), predicts the locked Phase 2B validation split
(data/processed/features_val.npy + labels_val.npy, cross-checked against
feature_stats.json), and reports per-class precision, recall, F1 and support
plus the 17x17 confusion matrix. Results are written to
scripts/results/phase3a_validation/<run-name>/ (validation_report.md,
per_class_metrics.csv, confusion_matrix.csv); if that directory already exists
the script aborts, so previous run results are never overwritten.

Read-only analysis: the model, preprocessing, configuration, dataset and the
Phase 3A training script are not modified and nothing is retrained. The test
split is never loaded (Phase 3B evaluation only).

Usage:
  python scripts/run_phase3a_validation_eval.py --run-name baseline10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src import config

MODEL_DIR = config.DATA_DIR / "models"
RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "phase3a_validation"
TOP_CONFUSIONS = 15
PREDICT_BATCH = 512


def _validate_run_name(raw: str) -> str:
    """Return a safe bare run name (no directories) or raise ValueError."""
    name = raw.strip()
    if not name:
        raise ValueError("--run-name must be a non-empty name")
    if name in {".", ".."}:
        raise ValueError("--run-name must not be '.' or '..'")
    bad = [ch for ch in '/\\<>:"|?*' if ch in name]
    if bad:
        raise ValueError(
            "--run-name must not contain path separators or Windows-unsafe "
            f"characters (found {bad!r})"
        )
    return name


def _load_validation() -> tuple[np.ndarray, np.ndarray]:
    """Load the locked validation features and integer labels."""
    features = np.load(config.PROCESSED_DATA_DIR / "features_val.npy")
    labels = np.load(config.PROCESSED_DATA_DIR / "labels_val.npy").astype(np.int32)
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{features.shape[0]} feature rows vs {labels.shape[0]} labels"
        )
    return features, labels


def _verify(features: np.ndarray, labels: np.ndarray) -> None:
    """Fail loudly if the validation arrays disagree with the frozen record."""
    with config.FEATURE_STATS_PATH.open(encoding="utf-8") as handle:
        stats = json.load(handle)
    recorded = stats["splits"]["val"]
    if int(recorded["windows"]) != features.shape[0]:
        raise ValueError(
            f"{features.shape[0]} rows vs {recorded['windows']} windows "
            "in feature_stats.json"
        )
    if list(features.shape[1:]) != list(config.SPEC_SHAPE):
        raise ValueError(f"feature shape {features.shape[1:]} != {config.SPEC_SHAPE}")
    if features.dtype != np.float32:
        raise ValueError(f"dtype {features.dtype} != float32")
    if int(labels.min()) < 0 or int(labels.max()) >= config.NUM_TRIGGER_CLASSES:
        raise ValueError(f"labels outside [0, {config.NUM_TRIGGER_CLASSES})")
    if int(stats["sample_rate"]) != config.SAMPLE_RATE:
        raise ValueError("feature_stats.json sample_rate disagrees with config")


def _per_class_rows(
    report: dict, class_names: list[str]
) -> list[dict]:
    """Extract per-class precision/recall/F1/support from the sklearn report."""
    rows = []
    for class_id, name in enumerate(class_names):
        entry = report[name]
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "support": int(entry["support"]),
                "precision": float(entry["precision"]),
                "recall": float(entry["recall"]),
                "f1": float(entry["f1-score"]),
            }
        )
    return rows


def _top_confusions(cm: np.ndarray, class_names: list[str]) -> list[dict]:
    """Rank the largest off-diagonal confusion pairs by count."""
    entries = []
    for i in range(len(class_names)):
        row_total = int(cm[i].sum())
        for j in range(len(class_names)):
            count = int(cm[i, j])
            if i != j and count > 0:
                entries.append(
                    {
                        "actual": class_names[i],
                        "predicted": class_names[j],
                        "count": count,
                        "rate": count / row_total if row_total else 0.0,
                    }
                )
    entries.sort(key=lambda item: (-item["count"], item["actual"], item["predicted"]))
    return entries[:TOP_CONFUSIONS]


def _write_csvs(
    rows: list[dict],
    cm: np.ndarray,
    class_names: list[str],
    metrics_path: Path,
    cm_path: Path,
) -> None:
    """Write per-class metrics and the confusion matrix as CSV files."""
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "support", "precision", "recall", "f1"])
        for row in rows:
            writer.writerow(
                [
                    row["class_id"],
                    row["class_name"],
                    row["support"],
                    f"{row['precision']:.6f}",
                    f"{row['recall']:.6f}",
                    f"{row['f1']:.6f}",
                ]
            )
    with cm_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for i, name in enumerate(class_names):
            writer.writerow([name, *[int(value) for value in cm[i]]])


def _markdown_cm(cm: np.ndarray, class_names: list[str]) -> list[str]:
    """Render the confusion matrix as markdown table lines."""
    lines = ["| actual \\ predicted | " + " | ".join(class_names) + " |"]
    lines.append("|---" * (len(class_names) + 1) + "|")
    for i, name in enumerate(class_names):
        cells = []
        for j in range(len(class_names)):
            value = int(cm[i, j])
            cells.append(f"**{value}**" if i == j else str(value))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return lines


def _write_report(
    rows: list[dict],
    cm: np.ndarray,
    class_names: list[str],
    summary: dict,
    confusions: list[dict],
    model_label: str,
    model_bytes: int,
    evaluate_seconds: float,
    predict_seconds: float,
    report_path: Path,
) -> None:
    """Write the full validation analysis report as Markdown."""
    lines = [
        "# Phase 3A Validation Evaluation (analysis only)",
        "",
        f"Model: `{model_label}` "
        f"({model_bytes:,} bytes, {summary['parameters']:,} parameters). "
        "Locked Phase 2B validation split: "
        f"{summary['windows']} windows, {summary['classes']} classes. "
        "No retraining; model, preprocessing, configuration and dataset "
        "unchanged.",
        "",
        "## Overall metrics",
        "",
        f"- val loss: **{summary['loss']:.4f}**",
        f"- val accuracy: **{summary['accuracy']:.4f}** "
        f"({summary['correct']}/{summary['windows']})",
        f"- macro precision / recall / F1: {summary['macro_precision']:.4f} / "
        f"{summary['macro_recall']:.4f} / {summary['macro_f1']:.4f}",
        f"- weighted precision / recall / F1: {summary['weighted_precision']:.4f} / "
        f"{summary['weighted_recall']:.4f} / {summary['weighted_f1']:.4f}",
        f"- model.evaluate: {evaluate_seconds:.1f} s, predict: {predict_seconds:.1f} s",
        "",
        "## Per-class metrics",
        "",
        "| id | class | support | precision | recall | F1 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_id']} | {row['class_name']} | {row['support']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
        )
    lines += [
        "",
        "## Confusion matrix (rows = actual, columns = predicted)",
        "",
    ]
    lines += _markdown_cm(cm, class_names)
    lines += [
        "",
        f"## Top confusions (off-diagonal, top {TOP_CONFUSIONS})",
        "",
        "| actual | predicted | windows | rate of actual class |",
        "|---|---|---:|---:|",
    ]
    for entry in confusions:
        lines.append(
            f"| {entry['actual']} | {entry['predicted']} | {entry['count']} | "
            f"{entry['rate']:.3f} |"
        )
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 3A validation-set analysis of a saved CNN (read-only)"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        metavar="NAME",
        help="run name; reads data/models/NAME.keras and writes "
        "scripts/results/phase3a_validation/NAME/ (aborts if that directory "
        "already exists)",
    )
    args = parser.parse_args()
    try:
        run_name = _validate_run_name(args.run_name)
    except ValueError as exc:
        parser.error(str(exc))

    started = time.perf_counter()
    model_path = MODEL_DIR / f"{run_name}.keras"
    results_dir = RESULTS_ROOT / run_name
    if results_dir.exists():
        raise SystemExit(
            f"results directory already exists: {results_dir} - pick a "
            "different --run-name; existing results are never overwritten"
        )
    if not model_path.exists():
        raise SystemExit(f"saved model not found: {model_path}")

    class_names = list(config.TRIGGER_CLASS_NAMES)
    print(f"run: {run_name}", flush=True)
    print(f"model: {model_path}", flush=True)
    model = tf.keras.models.load_model(str(model_path))
    parameters = int(model.count_params())
    print(f"parameters: {parameters}", flush=True)

    features, labels = _load_validation()
    _verify(features, labels)
    print(f"validation: {features.shape[0]} windows", flush=True)

    evaluate_started = time.perf_counter()
    scores = model.evaluate(
        features,
        labels,
        batch_size=config.BATCH_SIZE,
        verbose=0,
        return_dict=True,
    )
    evaluate_seconds = time.perf_counter() - evaluate_started

    predict_started = time.perf_counter()
    probabilities = model.predict(features, batch_size=PREDICT_BATCH, verbose=0)
    predict_seconds = time.perf_counter() - predict_started
    predictions = probabilities.argmax(axis=1).astype(np.int32)

    accuracy = float(accuracy_score(labels, predictions))
    report = classification_report(
        labels,
        predictions,
        labels=list(range(config.NUM_TRIGGER_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(
        labels, predictions, labels=list(range(config.NUM_TRIGGER_CLASSES))
    )
    rows = _per_class_rows(report, class_names)
    confusions = _top_confusions(cm, class_names)

    summary = {
        "parameters": parameters,
        "windows": int(features.shape[0]),
        "classes": len(class_names),
        "loss": float(scores["loss"]),
        "accuracy": accuracy,
        "correct": int((predictions == labels).sum()),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }

    report_path = results_dir / "validation_report.md"
    metrics_path = results_dir / "per_class_metrics.csv"
    cm_path = results_dir / "confusion_matrix.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_csvs(rows, cm, class_names, metrics_path, cm_path)
    _write_report(
        rows,
        cm,
        class_names,
        summary,
        confusions,
        f"data/models/{run_name}.keras",
        model_path.stat().st_size,
        evaluate_seconds,
        predict_seconds,
        report_path,
    )

    print(f"val loss: {summary['loss']:.4f}  val accuracy: {summary['accuracy']:.4f}", flush=True)
    print(
        f"macro F1: {summary['macro_f1']:.4f}  weighted F1: {summary['weighted_f1']:.4f}",
        flush=True,
    )
    print("", flush=True)
    print(
        f"{'class':<28} {'supp':>6} {'prec':>7} {'rec':>7} {'f1':>7}",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['class_name']:<28} {row['support']:>6} "
            f"{row['precision']:>7.4f} {row['recall']:>7.4f} {row['f1']:>7.4f}",
            flush=True,
        )
    print("", flush=True)
    print(f"top confusions (top {TOP_CONFUSIONS}):", flush=True)
    for entry in confusions:
        print(
            f"  {entry['actual']} -> {entry['predicted']}: "
            f"{entry['count']} ({entry['rate']:.3f})",
            flush=True,
        )
    print("", flush=True)
    print(f"report: {report_path}", flush=True)
    print(f"csv:    {metrics_path}", flush=True)
    print(f"csv:    {cm_path}", flush=True)
    print(f"total analysis time: {time.perf_counter() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
