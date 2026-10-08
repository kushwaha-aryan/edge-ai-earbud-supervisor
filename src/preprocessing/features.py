"""Stage 3 of the Phase 2A preprocessing pipeline: audio cleaning, windowing
and mel-spectrogram feature extraction.

Per kept clip (blueprint Change Log 2026-10-08, decisions D2/D5/D6/OA1):
  read with soundfile -> mean-mono downmix -> soxr_hq resample to 16 kHz
  -> truncate to the first 15.0 s -> windows of 3,200 samples every 1,600
  samples -> windows with RMS <= -60 dBFS are dropped -> batched
  librosa.melspectrogram (n_fft 1024, hop 512, 64 mels, center=False)
  giving exactly (N, 64, 5) -> power_to_db (ref 1.0, amin 1e-10,
  top_db=None, i.e. absolute dB with no per-clip clipping) -> scaled to
  [0, 1] using dB bounds frozen from the train split and recorded in
  data/processed/feature_stats.json.

The train split is processed twice: pass A collects the dB bounds and the
per-clip survivor counts, pass B fills the preallocated array; val and test
are processed once with the frozen bounds.

Outputs per split:
  data/processed/features_<split>.npy   float32 (N, 64, 5, 1)
  data/processed/labels_<split>.npy     int64   (N,)
  data/processed/window_index_<split>.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from src import config
from src.preprocessing.clips import resolve_path

SPLIT_ORDER = ("train", "val", "test")


def load_clean(local_path: str) -> np.ndarray:
    """Read one clip, downmix to mono, resample to 16 kHz, truncate to 15 s."""
    y, sr = sf.read(resolve_path(local_path), dtype="float32", always_2d=True)
    y = y.mean(axis=1, dtype=np.float32)
    if int(sr) != config.SAMPLE_RATE:
        y = librosa.resample(
            y,
            orig_sr=int(sr),
            target_sr=config.SAMPLE_RATE,
            res_type="soxr_hq",
        )
    n = int(config.TRUNCATE_SECONDS * config.SAMPLE_RATE)
    y = y[:n]
    if y.shape[0] < config.WINDOW_SAMPLES:
        raise ValueError(
            f"kept clip shorter than one window: {local_path} "
            f"({y.shape[0]} samples after cleaning)"
        )
    return np.ascontiguousarray(y, dtype=np.float32)


def clip_db(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Window one clip and return (db features, kept starts, n_raw, n_dropped)."""
    n = config.WINDOW_SAMPLES
    step = config.WINDOW_STEP
    starts = np.arange(0, y.shape[0] - n + 1, step)
    if starts.size == 0:
        raise ValueError("clip produced no windows")
    windows = np.stack([y[s : s + n] for s in starts])
    rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))
    dbfs = 20.0 * np.log10(rms + 1e-12)
    keep = dbfs > config.EXCLUDE_SILENT_DBFS
    kept_windows = windows[keep]
    kept_starts = starts[keep]
    if kept_windows.shape[0] == 0:
        raise ValueError("all windows of a kept clip fell below the silence floor")
    S = librosa.feature.melspectrogram(
        y=kept_windows,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        center=False,
        power=2.0,
    )
    db = librosa.power_to_db(
        S, ref=config.DB_POWER_REF, amin=1e-10, top_db=None
    )
    return db, kept_starts, int(starts.size), int((~keep).sum())


def _scaled(db: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = (db - lo) / (hi - lo)
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)[..., np.newaxis]


def _iter_split(df: pd.DataFrame, label: str):
    total = len(df)
    for i, row in enumerate(df.itertuples(index=False)):
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"[features] {label}: {i + 1}/{total} clips", flush=True)
        yield row


def extract_features(
    ledger: pd.DataFrame,
    clip_splits: pd.DataFrame,
    limit: int | None = None,
) -> dict:
    """Extract windowed mel-spectrograms for all splits; return a summary."""
    out_dir = config.PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = ledger.loc[
        ledger["status"] == "kept",
        ["clip_id", "local_path", "project_class", "class_id"],
    ]
    merged = kept.merge(
        clip_splits[["clip_id", "split"]], on="clip_id", how="inner", validate="one_to_one"
    )
    if len(merged) != len(kept):
        raise ValueError("clip_splits does not cover every kept clip")

    by_split = {
        s: merged.loc[merged["split"] == s].sort_values("clip_id").reset_index(drop=True)
        for s in SPLIT_ORDER
    }
    if limit is not None:
        by_split = {s: df.head(limit) for s, df in by_split.items()}
        print(f"[features] SMOKE TEST: first {limit} clips per split only", flush=True)

    t0 = time.time()
    db_lo = np.inf
    db_hi = -np.inf
    train_counts: list[tuple[int, int]] = []
    for row in _iter_split(by_split["train"], "pass A train"):
        db, _starts, n_raw, n_drop = clip_db(load_clean(row.local_path))
        db_lo = min(db_lo, float(db.min()))
        db_hi = max(db_hi, float(db.max()))
        train_counts.append((int(db.shape[0]), n_drop))
    if not np.isfinite(db_lo) or not np.isfinite(db_hi) or db_hi <= db_lo:
        raise ValueError(f"degenerate train dB range: [{db_lo}, {db_hi}]")

    n_windows = sum(k for k, _ in train_counts)
    X_train = np.empty(
        (n_windows, config.N_MELS, config.SPEC_FRAMES, 1), dtype=np.float32
    )
    y_train = np.empty(n_windows, dtype=np.int64)
    index_rows: dict[str, list[dict]] = {s: [] for s in SPLIT_ORDER}
    summary: dict = {
        "db_min": db_lo,
        "db_max": db_hi,
        "splits": {},
    }

    cursor = 0
    for row, (k, _drop) in zip(_iter_split(by_split["train"], "pass B train"), train_counts):
        db, starts, _n_raw, _n_drop = clip_db(load_clean(row.local_path))
        if db.shape[0] != k:
            raise RuntimeError(f"nondeterministic window count for {row.clip_id}")
        X_train[cursor : cursor + k] = _scaled(db, db_lo, db_hi)
        y_train[cursor : cursor + k] = int(row.class_id)
        for j, s in enumerate(starts):
            index_rows["train"].append(
                {
                    "window_index": cursor + j,
                    "clip_id": row.clip_id,
                    "project_class": row.project_class,
                    "class_id": int(row.class_id),
                    "start_sample": int(s),
                }
            )
        cursor += k
    np.save(out_dir / "features_train.npy", X_train)
    np.save(out_dir / "labels_train.npy", y_train)
    summary["splits"]["train"] = _split_summary(
        by_split["train"], len(train_counts), X_train.shape[0],
        sum(d for _k, d in train_counts),
    )

    for split in ("val", "test"):
        parts_X: list[np.ndarray] = []
        parts_y: list[np.ndarray] = []
        n_clips = 0
        n_drop_total = 0
        n_raw_total = 0
        cursor = 0
        for row in _iter_split(by_split[split], split):
            db, starts, n_raw, n_drop = clip_db(load_clean(row.local_path))
            parts_X.append(_scaled(db, db_lo, db_hi))
            parts_y.append(np.full(db.shape[0], int(row.class_id), dtype=np.int64))
            for j, s in enumerate(starts):
                index_rows[split].append(
                    {
                        "window_index": cursor + j,
                        "clip_id": row.clip_id,
                        "project_class": row.project_class,
                        "class_id": int(row.class_id),
                        "start_sample": int(s),
                    }
                )
            cursor += db.shape[0]
            n_clips += 1
            n_drop_total += n_drop
            n_raw_total += n_raw
        if not parts_X:
            raise ValueError(f"split {split} is empty")
        X = np.concatenate(parts_X, axis=0)
        y = np.concatenate(parts_y, axis=0)
        np.save(out_dir / f"features_{split}.npy", X)
        np.save(out_dir / f"labels_{split}.npy", y)
        summary["splits"][split] = {
            "clips": n_clips,
            "windows": int(X.shape[0]),
            "raw_windows": n_raw_total,
            "dropped_silent": n_drop_total,
        }

    for split in SPLIT_ORDER:
        pd.DataFrame(index_rows[split]).to_csv(
            out_dir / f"window_index_{split}.csv", index=False
        )

    stats = {
        "db_min": db_lo,
        "db_max": db_hi,
        "db_ref": config.DB_POWER_REF,
        "db_amin": 1e-10,
        "db_top_db": None,
        "sample_rate": config.SAMPLE_RATE,
        "n_fft": config.N_FFT,
        "hop_length": config.HOP_LENGTH,
        "n_mels": config.N_MELS,
        "spec_frames": config.SPEC_FRAMES,
        "spec_shape": list(config.SPEC_SHAPE),
        "window_samples": config.WINDOW_SAMPLES,
        "window_step": config.WINDOW_STEP,
        "truncate_seconds": config.TRUNCATE_SECONDS,
        "silence_dbfs": config.EXCLUDE_SILENT_DBFS,
        "split_seed": config.SPLIT_SEED,
        "ratios": [config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO],
        "splits": summary["splits"],
        "elapsed_seconds": round(time.time() - t0, 1),
        "limit": limit,
    }
    with open(out_dir / "feature_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    summary["stats"] = stats
    print(
        f"[features] done in {stats['elapsed_seconds']}s; "
        f"dB range [{db_lo:.2f}, {db_hi:.2f}]",
        flush=True,
    )
    return summary


def _split_summary(df: pd.DataFrame, n_clips: int, windows: int, dropped: int) -> dict:
    return {
        "clips": n_clips,
        "windows": int(windows),
        "raw_windows": int(windows + dropped),
        "dropped_silent": int(dropped),
    }


def compute_clip_covariates() -> dict:
    """Derive per-clip feature covariates for the D9 covariate-balanced split.

    Reads window_index_<split>.csv and features_<split>.npy for every split,
    computes each clip's mean scaled feature value over its surviving windows
    (identical to the Phase 2B clip_feature_mean), and writes
    data/provenance/clip_covariates.csv (clip_id, n_windows, feat_mean).

    Because scaling is an affine map of the frozen dB bounds, these means are
    affine-equivalent to raw mel-dB clip means: balancing them keeps its
    meaning even if the bounds change in a later extraction pass.
    """
    out_dir = config.PROCESSED_DATA_DIR
    parts: list[pd.DataFrame] = []
    for split in SPLIT_ORDER:
        wi = pd.read_csv(out_dir / f"window_index_{split}.csv")
        x = np.load(out_dir / f"features_{split}.npy", mmap_mode="r")
        if len(wi) != len(x):
            raise RuntimeError(
                f"covariates {split}: window_index rows {len(wi)} != features {len(x)}"
            )
        wmean = np.asarray(x.mean(axis=(1, 2, 3)), dtype=np.float64)
        df = pd.DataFrame({"clip_id": wi["clip_id"].to_numpy(), "wmean": wmean})
        parts.append(
            df.groupby("clip_id")["wmean"]
            .agg(feat_mean="mean", n_windows="count")
            .reset_index()
        )
    cov = pd.concat(parts, ignore_index=True)
    if cov["clip_id"].duplicated().any():
        raise RuntimeError("clip covariates contain duplicate clip_ids")
    cov = cov.sort_values("clip_id").reset_index(drop=True)
    path = config.PROVENANCE_DIR / "clip_covariates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    cov.to_csv(path, index=False)
    print(
        f"[covariates] {len(cov)} clips, {int(cov['n_windows'].sum())} windows "
        f"-> {path}",
        flush=True,
    )
    return {"clips": int(len(cov)), "windows": int(cov["n_windows"].sum())}
