#!/usr/bin/env python
"""Phase 3A - clip-level evaluation of any saved CNN (analysis only, Step 3).

Loads any .keras model via --model PATH, predicts the validation split of
the selected analysis preset (--preset, default A = the locked Phase 2B
baseline at data/processed/features_val.npy + labels_val.npy, cross-checked
against feature_stats.json), and reports window-level and clip-level
metrics side by side:

  - window level: one prediction per window (200 ms for preset A).
  - clip_mean_softmax: softmax probabilities averaged over a clip's
    windows, then argmax.
  - clip_topk_mean: per class, the mean softmax probability across that
    class's --top-k highest-probability windows in the clip (k = min(
    --top-k, windows in the clip)), then argmax over classes.
  - clip_majority_vote: majority vote of per-window argmax; ties are
    broken by the higher mean softmax over the clip.

For each level the report gives accuracy, macro and weighted precision /
recall / F1, per-class precision / recall / F1 / support, and a
confusion matrix. Clips are grouped through window_index_val.csv
(clip_id per window); the set of validation clips comes from
clip_splits.csv (split == val). Any validation clip with zero windows is
counted and listed. With --restrict-clips-to-preset R, evaluation is
limited to the validation clips that have at least one window in the
reference preset R's window_index_val.csv, so models trained on
different presets can be compared on an identical clip set. Results are
written to
scripts/results/phase3a_clip_eval/<run-name>/; if that directory already
exists the script aborts, so previous run results are never overwritten.

Read-only analysis: model, preprocessing, configuration and dataset are
not modified, nothing is retrained, and the test split is never loaded
(Phase 3B evaluation only).

Usage:
  .venv/Scripts/python scripts/run_phase3a_clip_eval.py \
      --model data/models/archive/weighted_epoch1.keras --run-name epoch1
  .venv/Scripts/python scripts/run_phase3a_clip_eval.py \
      --model data/models/lightweight_cnn.keras --run-name baseline --top-k 3
  .venv/Scripts/python scripts/run_phase3a_clip_eval.py \
      --preset D --model data/models/presetD_freqpool.keras \
      --run-name presetD_freqpool
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
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src import config

RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "phase3a_clip_eval"
PREDICT_BATCH = 512
DEFAULT_TOP_K = 5
LEVELS = [
    ("window", "window"),
    ("clip_mean_softmax", "clip mean-softmax"),
    ("clip_topk_mean", "clip top-k mean"),
    ("clip_majority_vote", "clip majority vote"),
]


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


def _load_validation(preset: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the validation features and integer labels for a preset."""
    spec = config.PRESETS[preset]
    features = np.load(spec["processed_dir"] / "features_val.npy")
    labels = np.load(spec["processed_dir"] / "labels_val.npy").astype(np.int32)
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{features.shape[0]} feature rows vs {labels.shape[0]} labels"
        )
    with spec["feature_stats_path"].open(encoding="utf-8") as handle:
        stats = json.load(handle)
    recorded = stats["splits"]["val"]
    if int(recorded["windows"]) != features.shape[0]:
        raise ValueError(
            f"{features.shape[0]} rows vs {recorded['windows']} windows "
            "in feature_stats.json"
        )
    if list(features.shape[1:]) != list(spec["spec_shape"]):
        raise ValueError(
            f"feature shape {features.shape[1:]} != {spec['spec_shape']}"
        )
    if features.dtype != np.float32:
        raise ValueError(f"dtype {features.dtype} != float32")
    if int(labels.min()) < 0 or int(labels.max()) >= config.NUM_TRIGGER_CLASSES:
        raise ValueError(f"labels outside [0, {config.NUM_TRIGGER_CLASSES})")
    if int(stats["sample_rate"]) != config.SAMPLE_RATE:
        raise ValueError("feature_stats.json sample_rate disagrees with config")
    return features, labels


def _load_val_clips() -> pd.DataFrame:
    """Return the validation clip truth (clip_id, class_id, project_class)."""
    frame = pd.read_csv(
        config.SPLIT_DATA_DIR / "clip_splits.csv",
        dtype={"clip_id": str, "class_id": "int32"},
    )
    frame = frame.loc[frame["split"] == "val"]
    frame = frame[["clip_id", "class_id", "project_class"]]
    if frame["clip_id"].duplicated().any():
        raise RuntimeError("duplicate clip_id in clip_splits.csv (split=val)")
    return frame


def _verify_window_labels(
    labels: np.ndarray, window_index: pd.DataFrame, clip_labels: pd.Series
) -> None:
    """Confirm window rows agree with labels, each clip carrying one label."""
    if len(window_index) != labels.shape[0]:
        raise ValueError(
            f"window_index rows {len(window_index)} != labels {labels.shape[0]}"
        )
    index_classes = window_index["class_id"].to_numpy(dtype=np.int32)
    if not np.array_equal(index_classes, labels):
        raise RuntimeError(
            "window_index class_id disagrees with labels_val.npy; clip "
            "grouping cannot be trusted"
        )
    per_clip = window_index.groupby("clip_id")["class_id"].nunique()
    if int(per_clip.max()) != 1:
        raise RuntimeError(
            "window_index_val.csv maps some clips to more than one class_id"
        )
    clip_class = window_index.groupby("clip_id")["class_id"].first()
    mismatched = [
        clip
        for clip in clip_class.index
        if clip in clip_labels.index
        and int(clip_class[clip]) != int(clip_labels[clip])
    ]
    if mismatched:
        raise RuntimeError(
            "window_index class_id disagrees with clip_splits for "
            f"{len(mismatched)} clip(s), e.g. {', '.join(mismatched[:5])}"
        )


def _aggregate_clips(
    probabilities: np.ndarray, window_index: pd.DataFrame, top_k: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return one prediction per clip for each clip-level aggregation."""
    clip_ids = [str(value) for value in window_index["clip_id"].tolist()]
    row_ids = np.arange(len(probabilities))
    layout = pd.DataFrame({"clip_id": clip_ids, "row": row_ids})
    order: list[str] = []
    n_windows: list[int] = []
    pred_mean: list[int] = []
    pred_topk: list[int] = []
    pred_vote: list[int] = []
    for clip_id, group in layout.groupby("clip_id", sort=False):
        indices = group["row"].to_numpy()
        block = probabilities[indices]
        order.append(clip_id)
        n_windows.append(int(indices.size))
        pred_mean.append(int(block.mean(axis=0).argmax()))
        k_eff = min(top_k, int(indices.size))
        sorted_block = np.sort(block, axis=0)[::-1]
        pred_topk.append(int(sorted_block[:k_eff].mean(axis=0).argmax()))
        votes = block.argmax(axis=1)
        counts = np.bincount(votes, minlength=block.shape[1])
        top_votes = int(counts.max())
        tied = np.flatnonzero(counts == top_votes)
        if tied.size == 1:
            pred_vote.append(int(tied[0]))
        else:
            means = block.mean(axis=0)
            pred_vote.append(int(tied[np.argmax(means[tied])]))
    result = pd.DataFrame(
        {
            "clip_id": order,
            "n_windows": n_windows,
            "pred_mean_softmax": pred_mean,
            "pred_topk_mean": pred_topk,
            "pred_majority_vote": pred_vote,
        }
    )
    return result, np.asarray(n_windows, dtype=np.int64)


def _assess(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]
) -> tuple[dict, list[dict]]:
    """Compute accuracy, macro/weighted F1 and per-class rows for one level."""
    correct = int(np.count_nonzero(y_true == y_pred))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "samples": int(y_true.size),
        "correct": correct,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_precision": float(report["weighted avg"]["precision"]),
        "weighted_recall": float(report["weighted avg"]["recall"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
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
    return summary, rows


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


def _write_confusion_csv(
    path: Path, cm: np.ndarray, class_names: list[str]
) -> None:
    """Write a confusion matrix as CSV (rows actual, columns predicted)."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *class_names])
        for i, name in enumerate(class_names):
            writer.writerow([name, *[int(value) for value in cm[i]]])


def _top_confusions(
    cm: np.ndarray, class_names: list[str], limit: int
) -> list[tuple[str, str, int]]:
    """Return the largest off-diagonal confusion cells as (true, pred, n)."""
    cells = [
        (class_names[i], class_names[j], int(cm[i, j]))
        for i in range(cm.shape[0])
        for j in range(cm.shape[1])
        if i != j and int(cm[i, j]) > 0
    ]
    cells.sort(key=lambda cell: cell[2], reverse=True)
    return cells[:limit]


def _load_ledger() -> pd.DataFrame:
    """Return kept ledger clips with dataset, sample-rate and class id."""
    frame = pd.read_csv(
        config.PROVENANCE_DIR / "phase2a_clip_ledger.csv",
        dtype={"clip_id": str},
    )
    frame = frame.loc[frame["status"] == "kept"]
    return frame[["clip_id", "class_id", "source_dataset", "sample_rate"]]


def _analyze_dataset(
    clips: pd.DataFrame,
    val_clips: pd.DataFrame,
    class_names: list[str],
) -> dict:
    """Compute sample-rate and source-dataset analytics from the ledger.

    Joins the clip ledger on clip_id, then returns:

      - baby_rows: Baby_Crying clip recall (mean-softmax) split by ledger
        sample_rate group (below 16000 Hz vs 16000 Hz and above), with
        total/assessed clip counts per group.
      - dominant: per-class source_dataset with the most clips of that
        class in the ledger (None if the class has no clipped dataset).
      - within/cross: for misclassified clips (mean-softmax), the count of
        errors whose predicted class has the same dominant source_dataset
        as the true class (within) versus a different one (cross).
      - pair_rows: per (true class, predicted class) within/cross summaries.
      - dataset_rows: per-class clip recall split by source_dataset for
        classes with more than one dataset in the validation split.
      - baby_no_pred: Baby_Crying clips without a window prediction.
    """
    ledger = _load_ledger()
    merged = val_clips.merge(
        ledger[["clip_id", "source_dataset", "sample_rate"]],
        on="clip_id",
        how="left",
    )
    clip_pred = clips.set_index("clip_id")["pred_mean_softmax"]
    merged["amp_pred"] = merged["clip_id"].map(clip_pred)
    predicted = merged["amp_pred"].notna()

    baby = merged.loc[merged["class_id"] == class_names.index("Baby_Crying")]
    baby_rows: list[tuple[str, int, int, int, float]] = []
    for label, mask in (
        ("below 16000", baby["sample_rate"] < 16000.0),
        ("16000 and above", baby["sample_rate"] >= 16000.0),
    ):
        group = baby.loc[mask]
        total = int(group.shape[0])
        assessed = group.loc[predicted.reindex(group.index, fill_value=False)]
        n = int(assessed.shape[0])
        correct = int(np.count_nonzero(assessed["amp_pred"] == assessed["class_id"]))
        recall = correct / n if n else 0.0
        baby_rows.append((label, total, n, correct, recall))
    baby_no_pred = int(baby.loc[~predicted.reindex(baby.index, fill_value=False)].shape[0])
    baby_no_rate = int(baby.loc[baby["sample_rate"].isna()].shape[0])

    dominant: dict[str, str | None] = {}
    for class_id, name in enumerate(class_names):
        class_ledger = ledger.loc[ledger["class_id"] == class_id]
        counts = class_ledger["source_dataset"].value_counts(dropna=True)
        dominant[name] = str(counts.index[0]) if counts.size else None

    mis = merged.loc[predicted & (merged["amp_pred"] != merged["class_id"])]
    within = 0
    cross = 0
    unclass = 0
    pair_counts: dict[tuple[str, str], int] = {}
    for row in mis.itertuples():
        true_name = class_names[int(row.class_id)]
        pred_name = class_names[int(row.amp_pred)]
        if dominant[true_name] is None or dominant[pred_name] is None:
            unclass += 1
        elif dominant[true_name] == dominant[pred_name]:
            within += 1
        else:
            cross += 1
        pair = (true_name, pred_name)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    n_mis = int(mis.shape[0])
    pair_rows: list[tuple[str, str, str, int]] = []
    for (true_name, pred_name), count in sorted(
        pair_counts.items(), key=lambda item: item[1], reverse=True
    ):
        if dominant[true_name] is None or dominant[pred_name] is None:
            kind = "unclassifiable"
        elif dominant[true_name] == dominant[pred_name]:
            kind = "within"
        else:
            kind = "cross"
        pair_rows.append((true_name, pred_name, kind, count))

    dataset_rows: list[tuple[str, str, int, int, float]] = []
    for class_id, name in enumerate(class_names):
        sub = merged.loc[
            (merged["class_id"] == class_id)
            & predicted
            & merged["source_dataset"].notna()
        ]
        if sub["source_dataset"].nunique() <= 1:
            continue
        for dataset, group in sub.groupby("source_dataset"):
            n = int(group.shape[0])
            correct = int(np.count_nonzero(group["amp_pred"] == class_id))
            dataset_rows.append((name, str(dataset), n, correct, correct / n if n else 0.0))

    return {
        "baby_rows": baby_rows,
        "baby_no_pred": baby_no_pred,
        "baby_no_rate": baby_no_rate,
        "dominant": dominant,
        "n_mis": n_mis,
        "within": within,
        "cross": cross,
        "unclass": unclass,
        "pair_rows": pair_rows,
        "dataset_rows": dataset_rows,
    }


def _write_report(
    report_path: Path,
    preset: str,
    model_label: str,
    model_bytes: int,
    parameters: int,
    top_k: int,
    class_names: list[str],
    window_summary: dict,
    level_summaries: list[tuple[str, dict]],
    per_class: dict[str, list[dict]],
    confusion: dict[str, np.ndarray],
    n_val_clips: int,
    zero_window: pd.DataFrame,
    predict_seconds: float,
    top_confusions: list[tuple[str, str, int]],
    dataset: dict,
    restriction_preset: str | None,
) -> None:
    """Write the full clip-level analysis report as Markdown."""
    restriction_bullet = (
        f"- evaluated on the validation clips that have windows in preset "
        f"`{restriction_preset}` (identical clip set across compared runs)"
        if restriction_preset is not None
        else None
    )
    lines = [
        "# Phase 3A Clip-Level Evaluation (analysis only)",
        "",
        f"Model: `{model_label}` ({model_bytes:,} bytes, {parameters:,} "
        f"parameters). Analysis preset `{preset}`: "
        f"{config.PRESETS[preset]['description']}. Frozen Phase 2B "
        "clip-level split, window and clip levels reported side by side. "
        "No retraining; model, preprocessing, configuration and dataset "
        "unchanged. The test split is never opened.",
        "",
        f"- validation windows: **{window_summary['samples']:,}**",
        f"- validation clips: **{n_val_clips:,}**",
        f"- clips with zero windows: **{len(zero_window):,}**",
        f"- top-k size for clip_topk_mean: **{top_k}**",
    ]
    if restriction_bullet is not None:
        lines.append(restriction_bullet)
    lines += [
        "",
        "## Aggregation definitions",
        "",
        "- window: one prediction per window (no aggregation).",
        "- clip_mean_softmax: mean over the clip's softmax probability "
        "vectors, then argmax.",
        "- clip_topk_mean: per class, the mean softmax probability across "
        "that class's --top-k highest-probability windows in the clip "
        "(k = min(--top-k, windows in the clip)), then argmax over classes.",
        "- clip_majority_vote: majority vote of per-window argmax; ties "
        "are broken by the higher mean softmax over the clip.",
        "",
        "## Overall metrics (window and clip levels)",
        "",
        "| metric | " + " | ".join(name for _, name in LEVELS) + " |",
        "|---" + "|---" * len(LEVELS) + "|",
    ]
    metric_rows = [
        ("samples", lambda s: f"{s['samples']:,}"),
        ("correct", lambda s: f"{s['correct']:,}"),
        ("accuracy", lambda s: f"{s['accuracy']:.4f}"),
        ("macro precision", lambda s: f"{s['macro_precision']:.4f}"),
        ("macro recall", lambda s: f"{s['macro_recall']:.4f}"),
        ("macro F1", lambda s: f"{s['macro_f1']:.4f}"),
        ("weighted precision", lambda s: f"{s['weighted_precision']:.4f}"),
        ("weighted recall", lambda s: f"{s['weighted_recall']:.4f}"),
        ("weighted F1", lambda s: f"{s['weighted_f1']:.4f}"),
    ]
    for label, extract in metric_rows:
        cells = [extract(window_summary)]
        for _level, summary in level_summaries:
            cells.append(extract(summary))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    for level, _name in LEVELS:
        lines += [
            "",
            f"## Per-class metrics - {level}",
            "",
            "| id | class | support | precision | recall | F1 |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for row in per_class[level]:
            lines.append(
                f"| {row['class_id']} | {row['class_name']} | {row['support']} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} |"
            )
    for level, _name in LEVELS:
        lines += [
            "",
            f"## Confusion matrix - {level} (rows = actual, columns = predicted)",
            "",
        ]
        lines += _markdown_cm(confusion[level], class_names)
    lines += [
        "",
        "## Top 15 clip-level confusions (mean-softmax)",
        "",
        "largest off-diagonal cells of the clip_mean_softmax matrix.",
        "",
        "| true | predicted | clips |",
        "|---|---|---:|",
    ]
    for true_name, pred_name, count in top_confusions:
        lines.append(f"| {true_name} | {pred_name} | {count} |")
    lines += [
        "",
        "## Baby_Crying clip recall by sample-rate group (mean-softmax)",
        "",
        "ledger sample_rate is the original clip sample rate; clips with no "
        "prediction (zero-window clips) are excluded.",
        "",
        "| sample-rate group | clips (total) | clips (assessed) | correct | recall |",
        "|---:|---:|---:|---:|---:|",
    ]
    for label, total, n, correct, recall in dataset["baby_rows"]:
        lines.append(
            f"| {label} | {total} | {n} | {correct} | {recall:.4f} |"
        )
    if dataset["baby_no_pred"]:
        lines.append(
            f"Baby_Crying clips with no window prediction: "
            f"**{dataset['baby_no_pred']}**"
        )
    if dataset["baby_no_rate"]:
        lines.append(
            f"Baby_Crying clips with no ledger sample_rate row: "
            f"**{dataset['baby_no_rate']}**"
        )
    lines += [
        "",
        "## Misclassified clips - within vs cross dominant source_dataset "
        "(mean-softmax)",
        "",
        "A class's dominant source_dataset is the dataset with the most "
        "clips of that class in the ledger. An error is within-dataset "
        "when the predicted class has the same dominant source_dataset as "
        "the true class, and cross-dataset otherwise.",
        "",
        f"- misclassified clips: **{dataset['n_mis']}**",
        f"- within-dataset errors: **{dataset['within']}**",
        f"- cross-dataset errors: **{dataset['cross']}**",
        f"- errors with no dominant dataset for either class: "
        f"**{dataset['unclass']}**",
        "",
        "| true class | predicted class | kind | clips |",
        "|---|---|---:|---:|",
    ]
    for true_name, pred_name, kind, count in dataset["pair_rows"]:
        lines.append(f"| {true_name} | {pred_name} | {kind} | {count} |")
    lines += [
        "",
        "## Per-class clip recall by source_dataset (mean-softmax)",
        "",
        "classes with more than one source_dataset in the validation split.",
        "",
        "| class | source_dataset | clips | correct | recall |",
        "|---|---|---:|---:|---:|",
    ]
    for name, dataset_label, n, correct, recall in dataset["dataset_rows"]:
        lines.append(
            f"| {name} | {dataset_label} | {n} | {correct} | {recall:.4f} |"
        )
    lines += [
        "",
        "## Clips with zero windows",
        "",
    ]
    if zero_window.empty:
        lines.append("No validation clip has zero windows.")
    else:
        lines.append("| class_id | class_name | clip_id |")
        lines.append("|---:|---|---|")
        for row in zero_window.itertuples():
            lines.append(
                f"| {int(row.class_id)} | {row.project_class} | {row.clip_id} |"
            )
    lines.append("")
    lines.append(f"model.predict: {predict_seconds:.1f} s")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 3A clip-level validation analysis of any saved .keras "
            "model (read-only, validation split only)"
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="path to any .keras model (e.g. data/models/archive/weighted_epoch1.keras)",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        metavar="NAME",
        help="run name; writes scripts/results/phase3a_clip_eval/NAME/ "
        "(aborts if that directory already exists)",
    )
    parser.add_argument(
        "--preset",
        default=config.PRESET_A,
        metavar="PRESET",
        help=(
            "analysis preset (default: A, the locked Phase 2B baseline); "
            f"known presets: {', '.join(config.PRESETS)}"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="K",
        help=f"windows per clip for clip_topk_mean (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--restrict-clips-to-preset",
        default=None,
        metavar="PRESET",
        help=(
            "limit evaluation to the validation clips that have at least one "
            "window in the reference preset's window_index_val.csv; used to "
            "compare presets on an identical clip set"
        ),
    )
    args = parser.parse_args()
    try:
        run_name = _validate_run_name(args.run_name)
    except ValueError as exc:
        parser.error(str(exc))
    if args.preset not in config.PRESETS:
        parser.error(
            f"unknown preset: {args.preset} (known: {', '.join(config.PRESETS)})"
        )
    if (
        args.restrict_clips_to_preset is not None
        and args.restrict_clips_to_preset not in config.PRESETS
    ):
        parser.error(
            f"unknown reference preset: {args.restrict_clips_to_preset} "
            f"(known: {', '.join(config.PRESETS)})"
        )
    model_path = Path(args.model)
    if model_path.suffix != ".keras":
        parser.error(f"--model must point to a .keras file: {model_path}")
    if not model_path.exists():
        parser.error(f"saved model not found: {model_path}")
    if args.top_k < 1:
        parser.error(f"--top-k must be at least 1 (got {args.top_k})")
    results_dir = RESULTS_ROOT / run_name
    if results_dir.exists():
        raise SystemExit(
            f"results directory already exists: {results_dir} - pick a "
            "different --run-name; existing results are never overwritten"
        )

    started = time.perf_counter()
    class_names = list(config.TRIGGER_CLASS_NAMES)
    print(f"run: {run_name}", flush=True)
    print(f"preset: {args.preset}", flush=True)
    print(f"model: {model_path}", flush=True)
    model = tf.keras.models.load_model(str(model_path))
    parameters = int(model.count_params())
    print(f"parameters: {parameters}", flush=True)

    preset_spec = config.PRESETS[args.preset]
    features, labels = _load_validation(args.preset)
    window_index = pd.read_csv(
        preset_spec["processed_dir"] / "window_index_val.csv",
        usecols=["clip_id", "class_id"],
        dtype={"clip_id": str, "class_id": "int32"},
    )
    val_clips = _load_val_clips()
    if args.restrict_clips_to_preset is not None:
        reference_spec = config.PRESETS[args.restrict_clips_to_preset]
        reference_index = pd.read_csv(
            reference_spec["processed_dir"] / "window_index_val.csv",
            usecols=["clip_id", "class_id"],
            dtype={"clip_id": str, "class_id": "int32"},
        )
        restrict_ids = set(reference_index["clip_id"].tolist())
        keep_mask = window_index["clip_id"].isin(restrict_ids).to_numpy()
        window_index = window_index.loc[keep_mask].reset_index(drop=True)
        features = features[keep_mask]
        labels = labels[keep_mask]
        total_counts = val_clips["class_id"].value_counts()
        kept_counts = (
            reference_index.drop_duplicates("clip_id")["class_id"].value_counts()
        )
        for class_id, name in enumerate(class_names):
            print(
                f"kept clips for {name}: "
                f"{int(kept_counts.get(class_id, 0))}/"
                f"{int(total_counts.get(class_id, 0))}",
                flush=True,
            )
        val_clips = val_clips.loc[val_clips["clip_id"].isin(restrict_ids)]
        print(
            f"restricted validation: {features.shape[0]} windows "
            f"({window_index['clip_id'].nunique()} clips)",
            flush=True,
        )
    _verify_window_labels(
        labels, window_index, val_clips.set_index("clip_id")["class_id"]
    )
    print(f"validation: {features.shape[0]} windows", flush=True)

    predict_started = time.perf_counter()
    probabilities = model.predict(features, batch_size=PREDICT_BATCH, verbose=0)
    predict_seconds = time.perf_counter() - predict_started
    window_predictions = probabilities.argmax(axis=1).astype(np.int32)

    clip_truth = val_clips.set_index("clip_id")["class_id"]
    clips, _n_windows = _aggregate_clips(probabilities, window_index, args.top_k)
    zero_window = val_clips.loc[~val_clips["clip_id"].isin(clips["clip_id"])]
    if zero_window.empty:
        print("clips with zero windows: none", flush=True)
    else:
        print(
            f"clips with zero windows: {len(zero_window)} "
            "(reported, excluded from clip-level metrics)",
            flush=True,
        )

    clip_true = clip_truth.reindex(clips["clip_id"]).to_numpy(dtype=np.int32)
    level_data = {
        "window": (labels, window_predictions),
        "clip_mean_softmax": (
            clip_true,
            clips["pred_mean_softmax"].to_numpy(dtype=np.int32),
        ),
        "clip_topk_mean": (
            clip_true,
            clips["pred_topk_mean"].to_numpy(dtype=np.int32),
        ),
        "clip_majority_vote": (
            clip_true,
            clips["pred_majority_vote"].to_numpy(dtype=np.int32),
        ),
    }

    per_class: dict[str, list[dict]] = {}
    confusion: dict[str, np.ndarray] = {}
    level_summaries: list[tuple[str, dict]] = []
    window_summary: dict | None = None
    for level, _name in LEVELS:
        y_true, y_pred = level_data[level]
        summary, rows = _assess(y_true, y_pred, class_names)
        per_class[level] = rows
        confusion[level] = confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        )
        level_summaries.append((level, summary))
        if level == "window":
            window_summary = summary

    results_dir.mkdir(parents=True, exist_ok=True)
    top_confusions = _top_confusions(
        confusion["clip_mean_softmax"], class_names, 15
    )
    dataset = _analyze_dataset(clips, val_clips, class_names)
    _write_report(
        results_dir / "clip_eval_report.md",
        args.preset,
        str(model_path),
        model_path.stat().st_size,
        parameters,
        args.top_k,
        class_names,
        window_summary,
        level_summaries,
        per_class,
        confusion,
        len(val_clips),
        zero_window,
        predict_seconds,
        top_confusions,
        dataset,
        args.restrict_clips_to_preset,
    )
    _write_confusion_csv(
        results_dir / "confusion_matrix_window.csv",
        confusion["window"],
        class_names,
    )
    for clip_level in ("clip_mean_softmax", "clip_topk_mean", "clip_majority_vote"):
        _write_confusion_csv(
            results_dir / f"confusion_matrix_{clip_level}.csv",
            confusion[clip_level],
            class_names,
        )
    with (results_dir / "per_class_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["level", "class_id", "class_name", "support", "precision", "recall", "f1"]
        )
        for level, _name in LEVELS:
            for row in per_class[level]:
                writer.writerow(
                    [
                        level,
                        row["class_id"],
                        row["class_name"],
                        row["support"],
                        f"{row['precision']:.6f}",
                        f"{row['recall']:.6f}",
                        f"{row['f1']:.6f}",
                    ]
                )
    with (results_dir / "summary_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "level",
                "samples",
                "correct",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_precision",
                "weighted_recall",
                "weighted_f1",
            ]
        )
        for level, _name in LEVELS:
            summary = dict(level_summaries)[level]
            writer.writerow(
                [
                    level,
                    summary["samples"],
                    summary["correct"],
                    f"{summary['accuracy']:.6f}",
                    f"{summary['macro_precision']:.6f}",
                    f"{summary['macro_recall']:.6f}",
                    f"{summary['macro_f1']:.6f}",
                    f"{summary['weighted_precision']:.6f}",
                    f"{summary['weighted_recall']:.6f}",
                    f"{summary['weighted_f1']:.6f}",
                ]
            )
    clips.merge(val_clips[["clip_id", "class_id", "project_class"]], on="clip_id")[
        ["clip_id", "class_id", "project_class", "n_windows",
         "pred_mean_softmax", "pred_topk_mean", "pred_majority_vote"]
    ].to_csv(results_dir / "clip_predictions.csv", index=False)
    zero_window.to_csv(results_dir / "zero_window_clips.csv", index=False)
    with (results_dir / "within_cross_dataset_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["true_class", "predicted_class", "error_kind", "clips"]
        )
        for true_name, pred_name, kind, count in dataset["pair_rows"]:
            writer.writerow([true_name, pred_name, kind, count])

    print("", flush=True)
    print(
        f"{'level':<22} {'acc':>7} {'macroF1':>9} {'wF1':>9} {'samples':>9}",
        flush=True,
    )
    for level, _name in LEVELS:
        summary = dict(level_summaries)[level]
        print(
            f"{level:<22} {summary['accuracy']:>7.4f} "
            f"{summary['macro_f1']:>9.4f} {summary['weighted_f1']:>9.4f} "
            f"{summary['samples']:>9,}",
            flush=True,
        )
    print("", flush=True)
    print("Baby_Crying clip recall by sample-rate group (mean-softmax):", flush=True)
    for label, total, n, correct, recall in dataset["baby_rows"]:
        print(
            f"  {label}: {recall:.4f} ({correct}/{n} clips; {total} total)",
            flush=True,
        )
    print(
        f"  misclassified clips: {dataset['n_mis']} "
        f"(within-domain {dataset['within']}, cross-domain {dataset['cross']})",
        flush=True,
    )
    if dataset["dataset_rows"]:
        print("per-class clip recall by source_dataset (mean-softmax):", flush=True)
        for name, ds, n, correct, recall in dataset["dataset_rows"]:
            print(
                f"  {name} / {ds}: {recall:.4f} ({correct}/{n})",
                flush=True,
            )
    print("", flush=True)
    print("top 15 clip-level confusions (mean-softmax):", flush=True)
    for true_name, pred_name, count in top_confusions:
        print(f"  {true_name} -> {pred_name}: {count}", flush=True)
    print("", flush=True)
    print(f"report: {results_dir / 'clip_eval_report.md'}", flush=True)
    print(f"report: {results_dir / 'confusion_matrix_window.csv'}", flush=True)
    print(f"report: {results_dir / 'per_class_metrics.csv'}", flush=True)
    print(f"report: {results_dir / 'summary_metrics.csv'}", flush=True)
    print(f"report: {results_dir / 'clip_predictions.csv'}", flush=True)
    print(f"time: {time.perf_counter() - started:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())