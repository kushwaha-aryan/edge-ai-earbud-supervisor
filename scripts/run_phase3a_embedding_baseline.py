#!/usr/bin/env python
"""Phase 3A - YAMNet frozen-embedding baseline over the same clean audio.

Two independent stages over the same kept clips (train + val only; the test
split is never loaded):

  --stage extract --run-name NAME
    For every kept clip in train + val, apply the exact Phase 2A audio
    cleaning used by the CNN baseline (soundfile read, mean-mono downmix,
    soxr_hq resample to 16 kHz, truncate to the first 15 s - the same
    load_clean() the mel-spectrogram pipeline uses), then run the frozen
    YAMNet SavedModel on the whole cleaned waveform once and record, per
    clip:
      - embed_mean : mean over frames of the 1024-d frame embeddings
      - embed_std  : population std (ddof=0) over frames of the 1024-d
                     frame embeddings
      - score_mean : mean over frames of the 521-d output class scores
      - n_frames   : number of YAMNet frames for the clip
    Outputs are written to data/embeddings/yamnet/<run-name>/:
      yamnet_embeddings_<run-name>.npz   embed_mean/embed_std/score_mean
      yamnet_embeddings_<run-name>.csv   clip_id,split,class_id,class_name,
                                         n_frames (row order matches the npz)
      yamnet_class_names.csv             the model's 521 display names
    If the output directory already exists the script aborts, so previous
    extracts are never overwritten. On any clip failure nothing is written:
    every failing clip is printed and the script exits nonzero (no silent
    skip). YAMNet is never fine-tuned here (frozen SavedModel).

  --stage fit --run-name NAME                                           [NOT RUN]
    Written for the later controlled comparison, not executed in the
    preparation pass. Reads a completed extract directory, standardizes the
    per-clip mean embeddings on train, fits a logistic-regression head on
    train (seed config.SPLIT_SEED), and reports clip-level accuracy, macro /
    weighted precision-recall-F1, per-class metrics, confusion matrices, the
    Baby_Crying recall split by source sample rate (below 16000 vs 16000 and
    above), the within vs cross dominant-source_dataset error share (same
    definition as run_phase3a_clip_eval.py), and a zero-shot variant that
    scores each clip by its mean YAMNet class scores summed over a
    hand-mapped set of defensible YAMNet classes per project class. Classes
    with no defensible YAMNet class are listed and left unmapped (they can
    never be predicted). Results land in
    scripts/results/phase3a_embedding/<run-name>/.

    Optional --save-model persists the fitted StandardScaler and logistic
    head with joblib (plus the ordered per-class display names and a pipeline
    description) to data/models/yamnet_logistic/<run-name>/ so the saved
    model can be reloaded and verified or driven by the demo script. The
    model directory must not already exist.

Usage:
  # extract (this is the stage executed during preparation)
  .venv/Scripts/python scripts/run_phase3a_embedding_baseline.py \
      --stage extract --run-name yamnet_extract
  # fit without saving (deferred; documented here so the command is not
  # forgotten)
  .venv/Scripts/python scripts/run_phase3a_embedding_baseline.py \
      --stage fit --run-name yamnet_lr
  # fit and persist the scaler + classifier for the demo / verification
  .venv/Scripts/python scripts/run_phase3a_embedding_baseline.py \
      --stage fit --run-name yamnet_lr --save-model
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.preprocessing.features import load_clean

RESULTS_ROOT = Path(__file__).resolve().parent / "results" / "phase3a_embedding"
CLIP_EVAL_ROOT = Path(__file__).resolve().parent / "results" / "phase3a_clip_eval"
EMBED_ROOT = Path(__file__).resolve().parent.parent / "data" / "embeddings" / "yamnet"
MODEL_ROOT = Path(__file__).resolve().parent.parent / "data" / "models" / "yamnet_logistic"
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_SCORE_KEYS = ("output_0", "output_1", "output_2")
SCORES_KEY = "output_0"
EMBED_KEY = "output_1"

ZERO_SHOT_MAP: dict[str, tuple[str, ...]] = {
    "Alarm": ("Fire alarm",),
    "Aircraft": ("Aircraft", "Aircraft engine", "Jet engine", "Helicopter"),
    "Baby_Crying": ("Baby cry, infant cry", "Crying, sobbing"),
    "Car_Engine": ("Engine",),
    "Dog_Bark": ("Bark", "Bow-wow", "Yip", "Growling", "Howl"),
    "Doorbell": ("Doorbell",),
    "Drilling": ("Drill", "Power tool"),
    "Footsteps": ("Walk, footsteps",),
    "Glass_Breaking": ("Glass", "Shatter"),
    "Gunshot": ("Gunshot, gunfire",),
    "Help_Shouting": ("Shout", "Screaming", "Yell"),
    "Jackhammer": ("Jackhammer",),
    "Knocking": ("Knock",),
    "Motorcycle": ("Motorcycle",),
    "Siren": ("Siren", "Civil defense siren"),
    "Train": ("Train", "Train wheels squealing"),
    "Vehicle_Horn": ("Vehicle horn, car horn, honking", "Air horn, truck horn"),
}

DEFAULT_EXTRACT_DIR = EMBED_ROOT / "yamnet_extract"
DEFAULT_COMPARE_DIR = (
    Path(__file__).resolve().parent
    / "results"
    / "phase3a_clip_eval"
    / "weighted_epoch1_fixed"
)


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


def _read_kept_samples() -> pd.DataFrame:
    """Return kept train+val clips with class names, paths and sample rates."""
    ledger = pd.read_csv(
        ROOT / "data" / "provenance" / "phase2a_clip_ledger.csv"
    )
    kept = ledger.loc[ledger["status"] == "kept", [
        "clip_id", "class_id", "local_path", "source_dataset", "sample_rate"
    ]]
    splits = pd.read_csv(ROOT / "data" / "splits" / "clip_splits.csv")
    parts = splits.loc[splits["split"].isin(("train", "val")), ["clip_id", "split"]]
    merged = kept.merge(parts, on="clip_id", how="inner", validate="one_to_one")
    merged = merged.sort_values(
        ["split", "clip_id"], kind="stable"
    ).reset_index(drop=True)
    names = {i: n for i, n in enumerate(config.TRIGGER_CLASS_NAMES)}
    merged["class_name"] = merged["class_id"].map(names)
    return merged


def _load_yamnet() -> tuple[object, object, pd.DataFrame]:
    """Load the frozen YAMNet SavedModel and its 521-name class map."""
    model = hub.load(YAMNET_URL)
    signature = model.signatures["serving_default"]
    missing = [k for k in YAMNET_SCORE_KEYS if k not in signature.structured_outputs]
    if missing:
        raise RuntimeError(f"YAMNet signature missing outputs {missing}")
    class_path = Path(str(model.class_map_path().numpy().decode("utf-8")))
    class_map = pd.read_csv(class_path)
    return model, signature, class_map


def _dominant_source(kept_ledger: pd.DataFrame) -> dict[str, str | None]:
    """Per class name, the ledger source_dataset with the most clips."""
    names = {i: n for i, n in enumerate(config.TRIGGER_CLASS_NAMES)}
    dominant: dict[str, str | None] = {}
    for class_id, group in kept_ledger.groupby("class_id"):
        counts = group["source_dataset"].value_counts()
        dominant[names[class_id]] = str(counts.index[0]) if counts.size else None
    return dominant


def _run_extract(args) -> int:
    """Embed kept train+val clips with frozen YAMNet into fresh files."""
    run_name = _validate_run_name(args.run_name)
    out_dir = EMBED_ROOT / run_name
    if out_dir.exists():
        raise SystemExit(
            f"output directory already exists: {out_dir} - pick a different "
            "--run-name; previous extracts are never overwritten"
        )

    samples = _read_kept_samples()
    model, signature, class_map = _load_yamnet()
    sig_out = signature.structured_outputs
    print(
        "YAMNet signature: waveform float32 -> "
        f"scores {tuple(sig_out[SCORES_KEY].shape)}, "
        f"embeddings {tuple(sig_out[EMBED_KEY].shape)}"
    )
    print(f"extract: {len(samples)} clips (train+val only)")

    class_names: list[str] = []
    index_rows: list[dict[str, object]] = []
    embed_means: list[np.ndarray] = []
    embed_stds: list[np.ndarray] = []
    score_means: list[np.ndarray] = []
    failures: list[tuple[str, str, str, str]] = []

    started = time.time()
    for row in samples.itertuples(index=False):
        try:
            y = load_clean(row.local_path)
            results = signature(tf.constant(y, dtype=tf.float32))
            scores = results[SCORES_KEY].numpy()
            embeddings = results[EMBED_KEY].numpy()
            frames = embeddings.shape[0]
            if scores.shape[0] != frames:
                raise RuntimeError(
                    f"frame mismatch: scores {scores.shape[0]} vs "
                    f"embeddings {frames}"
                )
            if frames == 0 or embeddings.shape[1] != 1024 or scores.shape[1] != 521:
                raise RuntimeError(
                    f"unexpected shapes scores {scores.shape}, "
                    f"embeddings {embeddings.shape}"
                )
            embed_means.append(embeddings.mean(axis=0, dtype=np.float32))
            embed_stds.append(embeddings.std(axis=0, dtype=np.float32))
            score_means.append(scores.mean(axis=0, dtype=np.float32))
            class_names.append(str(row.class_name))
            index_rows.append({
                "clip_id": row.clip_id,
                "split": row.split,
                "class_id": int(row.class_id),
                "class_name": str(row.class_name),
                "sample_rate": int(row.sample_rate),
                "n_frames": int(frames),
            })
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((str(row.clip_id), str(row.local_path),
                             type(exc).__name__, str(exc)))

    elapsed = time.time() - started

    if failures:
        for clip_id, path, kind, message in failures:
            print(f"FAILED {clip_id} | {path} | {kind}: {message}")
        print(f"embedding failed for {len(failures)} clips; nothing was "
              "written (rerun with the same --run-name after fixing the input)")
        return 1

    out_dir.mkdir(parents=True, exist_ok=False)
    np.savez(
        out_dir / f"yamnet_embeddings_{run_name}.npz",
        embed_mean=np.asarray(embed_means, dtype=np.float32),
        embed_std=np.asarray(embed_stds, dtype=np.float32),
        score_mean=np.asarray(score_means, dtype=np.float32),
    )
    pd.DataFrame(index_rows).to_csv(
        out_dir / f"yamnet_embeddings_{run_name}.csv", index=False
    )
    class_map[["index", "display_name"]].to_csv(
        out_dir / "yamnet_class_names.csv", index=False
    )

    counts = pd.Series([r["split"] for r in index_rows]).value_counts()
    frames = int(sum(r["n_frames"] for r in index_rows))
    print(f"embedded: train {counts.get('train', 0)}, "
          f"val {counts.get('val', 0)} clips; {frames} frames total")
    print(f"failures: 0 | time: {elapsed:.1f} s")
    print(f"outputs: {out_dir}")
    return 0


def _per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> list[dict[str, float]]:
    """Per-class precision/recall/F1/support from the confusion matrix."""
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    rows = []
    for c in range(n_classes):
        tp = int(confusion[c, c])
        fp = int(confusion[:, c].sum() - tp)
        fn = int(confusion[c, :].sum() - tp)
        support = int(confusion[c, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "class": config.TRIGGER_CLASS_NAMES[c],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        })
    return rows


def _top_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, top_n: int = 15
) -> list[tuple[str, str, int]]:
    """Largest off-diagonal (true, predicted) class-name counts."""
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    pairs = []
    for t in range(n_classes):
        for p in range(n_classes):
            if t != p and confusion[t, p]:
                pairs.append((
                    config.TRIGGER_CLASS_NAMES[t],
                    config.TRIGGER_CLASS_NAMES[p],
                    int(confusion[t, p]),
                ))
    pairs.sort(key=lambda row: row[2], reverse=True)
    return pairs[:top_n]


def _fit_summary(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Return accuracy / macro / weighted precision-recall-F1 as floats."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro",
                                           zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro",
                                     zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred,
                                              average="weighted",
                                              zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted",
                                        zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                zero_division=0),
    }


def _run_fit(args) -> int:
    """Fit a logistic head on standardized embeddings and compare (deferred)."""
    run_name = _validate_run_name(args.run_name)
    out_dir = RESULTS_ROOT / run_name
    if out_dir.exists():
        raise SystemExit(
            f"results directory already exists: {out_dir} - pick a different "
            "--run-name; existing results are never overwritten"
        )
    model_dir = MODEL_ROOT / run_name
    if args.save_model and model_dir.exists():
        raise SystemExit(
            f"model directory already exists: {model_dir} - pick a different "
            "--run-name or drop --save-model; saved models are never "
            "overwritten"
        )
    extract_dir = Path(args.extract_dir)
    npz = np.load(extract_dir / f"yamnet_embeddings_{extract_dir.name}.npz")
    index = pd.read_csv(extract_dir / f"yamnet_embeddings_{extract_dir.name}.csv")
    class_names = config.TRIGGER_CLASS_NAMES
    n_classes = len(class_names)
    class_names_df = pd.read_csv(extract_dir / "yamnet_class_names.csv")

    embed_mean = np.asarray(npz["embed_mean"], dtype=np.float64)
    score_mean = np.asarray(npz["score_mean"], dtype=np.float64)
    y = index["class_id"].to_numpy(dtype=np.int64)
    is_train = (index["split"] == "train").to_numpy(bool)
    is_val = ~is_train
    y_val = y[is_val]

    scaler = StandardScaler().fit(embed_mean[is_train])
    print(f"scikit-learn {sklearn.__version__} | logistic head C={args.c}, "
          f"max_iter={args.max_iter}, lbfgs (multinomial)")
    clf = LogisticRegression(
        C=args.c,
        max_iter=args.max_iter,
        solver="lbfgs",
        random_state=config.SPLIT_SEED,
    ).fit(scaler.transform(embed_mean[is_train]), y[is_train])
    pred_lr = clf.predict(scaler.transform(embed_mean[is_val]))

    name_to_ix = dict(
        zip(class_names_df["display_name"], class_names_df["index"])
    )
    missing = sorted({
        name
        for our in ZERO_SHOT_MAP.values()
        for name in our
        if name not in name_to_ix
    })
    if missing:
        raise RuntimeError(f"YAMNet class map is missing names {missing}")
    mask = np.zeros((len(class_names_df), n_classes), dtype=np.float64)
    mapped: list[str] = []
    unmapped: list[str] = []
    for c, our in enumerate(class_names):
        picks = ZERO_SHOT_MAP.get(our, ())
        if picks:
            mapped.append(our)
            for name in picks:
                mask[int(name_to_ix[name]), c] = 1.0
        else:
            unmapped.append(our)
    zs_scores = score_mean[is_val] @ mask
    pred_zs = np.argmax(zs_scores, axis=1).astype(np.int64)

    ledger = pd.read_csv(
        ROOT / "data" / "provenance" / "phase2a_clip_ledger.csv"
    )
    dominant = _dominant_source(ledger.loc[ledger["status"] == "kept"])
    val_meta = index.loc[is_val].reset_index(drop=True).copy()
    val_meta["sample_rate"] = val_meta["clip_id"].map(
        ledger.set_index("clip_id")["sample_rate"]
    )

    def within_cross(pred: np.ndarray) -> dict[str, int]:
        """Within/cross dominant source_dataset errors over misclassified
        clips only (predicted != true), matching run_phase3a_clip_eval.py."""
        within = 0
        cross = 0
        excluded = 0
        n_mis = 0
        for true_id, pred_id in zip(y_val, pred):
            if true_id == pred_id:
                continue
            n_mis += 1
            true_name = class_names[true_id]
            pred_name = class_names[pred_id]
            if (dominant.get(true_name) is None
                    or dominant.get(pred_name) is None):
                excluded += 1
            elif dominant[true_name] == dominant[pred_name]:
                within += 1
            else:
                cross += 1
        return {"within": within, "cross": cross, "excluded": excluded,
                "misclassified": n_mis}

    def baby_recall(pred: np.ndarray) -> dict[str, dict[str, int]]:
        baby = val_meta["class_id"].to_numpy() == class_names.index("Baby_Crying")
        out: dict[str, dict[str, int]] = {}
        for key, cond in (
            ("below_16000", val_meta["sample_rate"].to_numpy() < 16000),
            ("at_least_16000", val_meta["sample_rate"].to_numpy() >= 16000),
        ):
            sel = baby & cond
            out[key] = {
                "correct": int((pred == class_names.index("Baby_Crying"))[sel].sum()),
                "total": int(sel.sum()),
            }
        return out

    variants = (("logistic", pred_lr), ("zero_shot", pred_zs))
    summary = [
        {"variant": name, **_fit_summary(y_val, pred)}
        for name, pred in variants
    ]
    per_class = {
        name: _per_class_metrics(y_val, pred, n_classes)
        for name, pred in variants
    }
    top_conf = {
        name: _top_confusions(y_val, pred, n_classes)
        for name, pred in variants
    }
    wc = {name: within_cross(pred) for name, pred in variants}
    baby = {name: baby_recall(pred) for name, pred in variants}

    compare_dir = Path(args.compare_clip_eval)
    compare_skipped = True
    cnn_row: dict[str, object] | None = None
    if not compare_dir.is_dir():
        print(f"comparison skipped: no compare directory {compare_dir}")
    elif not (compare_dir / "summary_metrics.csv").exists():
        print(f"comparison skipped: no summary file "
              f"{compare_dir / 'summary_metrics.csv'}")
    else:
        compare_skipped = False
        cnn = pd.read_csv(compare_dir / "summary_metrics.csv")
        row = cnn.loc[cnn["level"] == "clip_mean_softmax"]
        if not row.empty:
            r = row.iloc[0]
            cnn_row = {
                "variant": "clip_mean_softmax (CNN)",
                "accuracy": round(float(r["accuracy"]), 4),
                "macro_precision": round(float(r["macro_precision"]), 4),
                "macro_recall": round(float(r["macro_recall"]), 4),
                "macro_f1": round(float(r["macro_f1"]), 4),
                "weighted_precision": round(float(r["weighted_precision"]), 4),
                "weighted_recall": round(float(r["weighted_recall"]), 4),
                "weighted_f1": round(float(r["weighted_f1"]), 4),
            }

    out_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summary).round(4).to_csv(
        out_dir / "summary_metrics.csv", index=False
    )
    for name, rows in per_class.items():
        pd.DataFrame(rows).to_csv(
            out_dir / f"per_class_{name}.csv", index=False
        )
    for name, rows in top_conf.items():
        pd.DataFrame(rows, columns=["true_class", "predicted_class", "count"]).to_csv(
            out_dir / f"top_confusions_{name}.csv", index=False
        )
    pd.DataFrame(
        [{"variant": name, **values} for name, values in wc.items()]
    ).to_csv(out_dir / "within_cross_dataset_errors.csv", index=False)
    pd.DataFrame(
        [
            {"variant": name, "group": group, **values}
            for name, groups in baby.items()
            for group, values in groups.items()
        ]
    ).to_csv(out_dir / "baby_crying_recall_by_sample_rate.csv", index=False)
    for name, pred in variants:
        confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        for t, p in zip(y_val, pred):
            confusion[t, p] += 1
        pd.DataFrame(
            confusion,
            index=class_names,
            columns=class_names,
        ).to_csv(out_dir / f"confusion_matrix_{name}.csv")

    _write_fit_report(
        out_dir=out_dir,
        run_name=run_name,
        extract_dir=extract_dir,
        compare_dir=compare_dir,
        compare_skipped=compare_skipped,
        cnn_row=cnn_row,
        summary=summary,
        per_class=per_class,
        top_conf=top_conf,
        wc=wc,
        baby=baby,
        mapped=mapped,
        unmapped=unmapped,
        name_to_ix=name_to_ix,
        train_count=int(is_train.sum()),
        val_count=int(is_val.sum()),
    )

    if args.save_model:
        saved_dir = _save_fit_pipeline(
            model_dir=model_dir,
            run_name=run_name,
            extract_dir=extract_dir,
            scaler=scaler,
            clf=clf,
            class_names=class_names,
            summary=summary,
            c=args.c,
            max_iter=args.max_iter,
            train_count=int(is_train.sum()),
            val_count=int(is_val.sum()),
            sklearn_version=sklearn.__version__,
        )
        print(f"saved model: {saved_dir}")

    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print("zero-shot mapping (project class -> YAMNet labels, indices):")
    for our in sorted(mapped):
        picks = ZERO_SHOT_MAP[our]
        refs = ", ".join(f"{label} -> {int(name_to_ix[label])}" for label in picks)
        print(f"  {our}: {refs}")
    print(f"zero-shot classes with no defensible YAMNet label "
          f"(never predicted): {', '.join(unmapped) if unmapped else 'none'}")
    for variant, values in wc.items():
        within = values["within"]
        cross = values["cross"]
        print(f"within/cross ({variant}): within {within} + cross {cross} "
              f"= {within + cross} misclassified clips {values['misclassified']}"
              f" (excluded {values['excluded']})")
    print(f"report: {out_dir}")
    return 0


def _write_fit_report(
    out_dir: Path,
    run_name: str,
    extract_dir: Path,
    compare_dir: Path,
    compare_skipped: bool,
    cnn_row: dict[str, object] | None,
    summary: list[dict[str, float]],
    per_class: dict[str, list[dict[str, float]]],
    top_conf: dict[str, list[tuple[str, str, int]]],
    wc: dict[str, dict[str, int]],
    baby: dict[str, dict[str, dict[str, int]]],
    mapped: list[str],
    unmapped: list[str],
    name_to_ix: dict[str, int],
    train_count: int,
    val_count: int,
) -> None:
    """Write the deferred fit-stage markdown report (ASCII only)."""
    lines: list[str] = []
    add = lines.append
    add(f"# YAMNet embedding baseline - fit report ({run_name})")
    add("")
    add(f"- extract: {extract_dir} (train {train_count} / val {val_count} clips)")
    if compare_skipped:
        add(f"- CNN comparator: skipped (no {compare_dir / 'summary_metrics.csv'})")
    else:
        add(f"- CNN comparator: {compare_dir} (clip_mean_softmax level)")
    add(f"- YAMNet: frozen, {YAMNET_URL}")
    add(f"- logistic head: StandardScaler fit on train, "
        f"multinomial logistic regression on standardized per-clip mean "
        f"embeddings, seed {config.SPLIT_SEED}")
    add(f"- test split: never loaded")
    add("")

    add("## Side-by-side (validation clip level)")
    add("")
    add("| variant | accuracy | macro F1 | weighted F1 |")
    add("|---|---|---|---|")
    for row in summary:
        add(f"| {row['variant']} | {row['accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['weighted_f1']:.4f} |")
    if cnn_row is not None:
        add(f"| {cnn_row['variant']} | {cnn_row['accuracy']:.4f} | "
            f"{cnn_row['macro_f1']:.4f} | {cnn_row['weighted_f1']:.4f} |")
    add("")
    add("Logistic uses mean frame embeddings; zero-shot uses mean YAMNet "
        "class scores summed over a hand-mapped subset of the 521 classes.")
    add("")

    for name in ("logistic", "zero_shot"):
        add(f"## {name}")
        add("")
        rows = per_class[name]
        add("| class | precision | recall | f1 | support |")
        add("|---|---|---|---|---|")
        for r in rows:
            add(f"| {r['class']} | {r['precision']} | {r['recall']} | "
                f"{r['f1']} | {r['support']} |")
        add("")
        if name == "zero_shot":
            add("Zero-shot mapping (project class -> resolved YAMNet classes):")
            add("")
            add("| project class | mapped YAMNet classes |")
            add("|---|---|")
            for our in sorted(mapped):
                picks = ZERO_SHOT_MAP[our]
                refs = ", ".join(
                    f"{z} (index {name_to_ix[z]})" for z in picks
                )
                add(f"| {our} | {refs} |")
            add("")
            add(f"Unmapped classes (never predicted): {', '.join(unmapped)}")
            add("")
        add("Top clip-level confusions:")
        add("")
        add("| true class | predicted class | count |")
        add("|---|---|---|")
        for true_name, pred_name, count in top_conf[name]:
            add(f"| {true_name} | {pred_name} | {count} |")
        add("")
        add(f"Within vs cross dominant source_dataset errors (over "
            f"misclassified clips only): within {wc[name]['within']}, "
            f"cross {wc[name]['cross']}, excluded {wc[name]['excluded']} "
            f"of {wc[name]['misclassified']} misclassified clips (same "
            f"definition as run_phase3a_clip_eval.py).")
        add("")
        groups = baby[name]
        add("Baby_Crying clip recall by source sample-rate group:")
        add("")
        add("| group | correct | total |")
        add("|---|---|---|")
        for group, value in groups.items():
            add(f"| {group} | {value['correct']} | {value['total']} |")
        add("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _pipeline_description(
    run_name: str,
    extract_dir: Path,
    class_names: tuple[str, ...],
    c: float,
    max_iter: int,
    train_count: int,
    val_count: int,
    val_accuracy: float,
    val_macro_f1: float,
    val_weighted_f1: float,
    sklearn_version: str,
) -> str:
    """Return an ASCII, human-readable description of the saved pipeline."""
    return (
        "Saved pipeline description - YAMNet frozen embeddings + logistic head\n"
        "\n"
        f"run-name: {run_name}\n"
        f"saved at (UTC): {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n"
        f"extract dir: {extract_dir} (train {train_count} / val {val_count} "
        "clips)\n"
        f"feature extractor: frozen YAMNet SavedModel ({YAMNET_URL}); per-clip "
        "mean of the 1024-d frame embeddings\n"
        "classifier head: sklearn StandardScaler fit on train, then scikit-learn "
        "multinomial LogisticRegression\n"
        f"logistic params: C={c}, max_iter={max_iter}, solver=lbfgs, "
        f"random_state={config.SPLIT_SEED}\n"
        f"scikit-learn: {sklearn_version}\n"
        f"validation clip-level metrics: accuracy {val_accuracy:.4f}, macro F1 "
        f"{val_macro_f1:.4f}, weighted F1 {val_weighted_f1:.4f}\n"
        "test split: never loaded\n"
        f"class names: saved in index order as class_names.txt "
        f"({len(class_names)} classes)\n"
        "\n"
        "scaler.joblib and classifier.joblib are joblib-serialized; reload with "
        "joblib.load. Feeding the same standardized per-clip mean YAMNet "
        "embeddings reproduces the reported validation metrics."
    )


def _save_fit_pipeline(
    model_dir: Path,
    run_name: str,
    extract_dir: Path,
    scaler: StandardScaler,
    clf: LogisticRegression,
    class_names: tuple[str, ...],
    summary: list[dict[str, float]],
    c: float,
    max_iter: int,
    train_count: int,
    val_count: int,
    sklearn_version: str,
) -> Path:
    """Persist the fitted scaler, logistic head, order names and description."""
    model_dir.mkdir(parents=True, exist_ok=False)
    joblib.dump(scaler, model_dir / "scaler.joblib")
    joblib.dump(clf, model_dir / "classifier.joblib")
    (model_dir / "class_names.txt").write_text(
        "\n".join(class_names) + "\n", encoding="utf-8"
    )
    logistic = next(
        row for row in summary if row["variant"] == "logistic"
    )
    (model_dir / "pipeline_description.txt").write_text(
        _pipeline_description(
            run_name=run_name,
            extract_dir=extract_dir,
            class_names=class_names,
            c=c,
            max_iter=max_iter,
            train_count=train_count,
            val_count=val_count,
            val_accuracy=float(logistic["accuracy"]),
            val_macro_f1=float(logistic["macro_f1"]),
            val_weighted_f1=float(logistic["weighted_f1"]),
            sklearn_version=sklearn_version,
        ),
        encoding="utf-8",
    )
    return model_dir


def main() -> int:
    """Parse arguments and dispatch the requested stage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stage", choices=("extract", "fit"), required=True,
        help="stage to run (extract is the preparation-time stage; fit is "
             "written but deferred to the controlled comparison)",
    )
    parser.add_argument(
        "--run-name", required=True,
        help="bare name for the output directory (extract: data/embeddings/"
             "yamnet/<name>; fit: scripts/results/phase3a_embedding/<name>)",
    )
    parser.add_argument(
        "--extract-dir", default=str(DEFAULT_EXTRACT_DIR),
        help="extract directory used by --stage fit "
             f"(default {DEFAULT_EXTRACT_DIR})",
    )
    parser.add_argument(
        "--c", type=float, default=1.0,
        help="logistic regression C (fit only, default 1.0)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=2000,
        help="logistic regression max_iter (fit only, default 2000)",
    )
    parser.add_argument(
        "--save-model", action="store_true",
        help="persist the fitted StandardScaler and logistic head with joblib "
             "to data/models/yamnet_logistic/<run-name>/ plus class_names.txt "
             "and pipeline_description.txt (fit only; the directory must not "
             "already exist)",
    )
    parser.add_argument(
        "--compare-clip-eval", default=str(CLIP_EVAL_ROOT / "baseline_A"),
        help="CNN comparator directory written by run_phase3a_clip_eval.py; "
             "the fit report adds a clip_mean_softmax row read from its "
             "summary_metrics.csv. Missing directory or file only prints a "
             "one-line skip notice; it never fails the run (fit only, default "
             "scripts/results/phase3a_clip_eval/baseline_A)",
    )
    args = parser.parse_args()
    if args.stage == "extract":
        if args.save_model:
            raise SystemExit("--save-model applies to --stage fit only")
        return _run_extract(args)
    return _run_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())