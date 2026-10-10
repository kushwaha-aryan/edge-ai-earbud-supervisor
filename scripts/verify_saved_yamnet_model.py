#!/usr/bin/env python
"""Verify a saved YAMNet + logistic pipeline against held-out validation data.

Reads the fitted scaler.joblib, classifier.joblib and class_names.txt written
by run_phase3a_embedding_baseline.py --stage fit --save-model, standardizes
the held-out (validation) per-clip mean YAMNet embeddings the same way the
fit stage did, predicts with the saved logistic head, and reports clip-level
accuracy and macro F1.

The expected numbers default to the verified reference run (accuracy 0.7769,
macro F1 0.7426); override with --expected-accuracy / --expected-macro-f1.
Anything off by more than the tolerance (1e-4) prints a MISMATCH line and
exits nonzero, so a broken save fails the caller. ASCII only; never writes
any output file.

Usage:
  .venv/Scripts/python scripts/verify_saved_yamnet_model.py \
      --model-dir data/models/yamnet_logistic/yamnet_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_MODEL_DIR = ROOT / "data" / "models" / "yamnet_logistic"
DEFAULT_EXTRACT_DIR = ROOT / "data" / "embeddings" / "yamnet" / "yamnet_extract"
EXPECTED_ACCURACY = 0.7769
EXPECTED_MACRO_F1 = 0.7426
TOLERANCE = 1e-4


def _load_saved(model_dir: Path) -> tuple[object, object, list[str]]:
    """Return (scaler, classifier, class names) or abort on a missing file."""
    scaler_path = model_dir / "scaler.joblib"
    clf_path = model_dir / "classifier.joblib"
    names_path = model_dir / "class_names.txt"
    missing = [
        str(p.relative_to(ROOT))
        for p in (scaler_path, clf_path, names_path)
        if not p.is_file()
    ]
    if missing:
        raise SystemExit(
            f"missing artifacts in {model_dir}: {', '.join(missing)} - run the "
            "fit stage with --save-model first"
        )
    scaler = joblib.load(scaler_path)
    clf = joblib.load(clf_path)
    names = [
        line.strip()
        for line in names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise SystemExit(f"class_names.txt in {model_dir} is empty")
    return scaler, clf, names


def _load_extract(extract_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (val embed_mean float64, val class ids, val split mask)."""
    npz_path = extract_dir / f"yamnet_embeddings_{extract_dir.name}.npz"
    csv_path = extract_dir / f"yamnet_embeddings_{extract_dir.name}.csv"
    missing = [str(p.relative_to(ROOT)) for p in (npz_path, csv_path)
               if not p.is_file()]
    if missing:
        raise SystemExit(
            f"missing extract files: {', '.join(missing)} - run the extract "
            "stage first"
        )
    npz = np.load(npz_path)
    index = pd.read_csv(csv_path)
    embed_mean = np.asarray(npz["embed_mean"], dtype=np.float64)
    if embed_mean.shape[0] != len(index):
        raise SystemExit(
            f"extract mismatch: npz has {embed_mean.shape[0]} rows but the "
            f"csv index has {len(index)} rows"
        )
    y = index["class_id"].to_numpy(dtype=np.int64)
    is_val = (index["split"] != "train").to_numpy(bool)
    if is_val.sum() == 0:
        raise SystemExit(f"no validation clips found in {extract_dir}")
    return embed_mean, y, is_val


def main() -> int:
    """Parse arguments, reload the saved pipeline and check the metrics."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir", default=str(DEFAULT_MODEL_DIR),
        help=f"directory with scaler.joblib / classifier.joblib / "
             f"class_names.txt (default {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--extract-dir", default=str(DEFAULT_EXTRACT_DIR),
        help=f"extract directory whose validation clips are scored "
             f"(default {DEFAULT_EXTRACT_DIR})",
    )
    parser.add_argument(
        "--expected-accuracy", type=float, default=EXPECTED_ACCURACY,
        help=f"expected validation accuracy (default {EXPECTED_ACCURACY})",
    )
    parser.add_argument(
        "--expected-macro-f1", type=float, default=EXPECTED_MACRO_F1,
        help=f"expected validation macro F1 (default {EXPECTED_MACRO_F1})",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    extract_dir = Path(args.extract_dir)
    scaler, clf, names = _load_saved(model_dir)
    embed_mean, y, is_val = _load_extract(extract_dir)
    if y[is_val].max() >= len(names):
        raise SystemExit(
            f"validation data references class {y[is_val].max()} but the "
            f"saved model knows {len(names)} classes"
        )

    x_val = scaler.transform(embed_mean[is_val])
    y_val = y[is_val]
    pred = np.asarray(clf.predict(x_val), dtype=np.int64)
    accuracy = float(accuracy_score(y_val, pred))
    macro_f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))

    acc_ok = abs(accuracy - args.expected_accuracy) <= TOLERANCE
    f1_ok = abs(macro_f1 - args.expected_macro_f1) <= TOLERANCE
    ok = acc_ok and f1_ok

    print(f"model dir: {model_dir}")
    print(f"extract dir: {extract_dir}")
    print(f"validation clips: {int(is_val.sum())}")
    print(f"classes in saved model: {len(names)}")
    print(f"expected accuracy {args.expected_accuracy:.4f}, macro F1 "
          f"{args.expected_macro_f1:.4f} (tolerance {TOLERANCE})")
    print(f"accuracy    = {accuracy:.4f}  {'match' if acc_ok else 'DIFFERS'}")
    print(f"macro F1    = {macro_f1:.4f}  {'match' if f1_ok else 'DIFFERS'}")
    if ok:
        print("RESULT: MATCH - the saved pipeline reproduces the reported "
              "validation metrics")
        return 0
    print("RESULT: MISMATCH - the saved pipeline does NOT reproduce the "
          "reported validation metrics")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())