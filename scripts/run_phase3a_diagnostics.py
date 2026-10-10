"""Phase 3A controlled-experiment diagnostics (Step 2).

Four read-only reports over the Phase 2A artifacts, written to
scripts/results/diagnostics/ as CSV + Markdown:

  1. Normalization audit -- determines from src/preprocessing/features.py
     and data/processed/feature_stats.json whether feature scaling is
     global or per-window, prints the exact formula with file/line
     references, and reports per class x split the fraction of values at
     the -100 dB floor (normalized 0.0), mean, std, and fraction >= 0.95.
   2. Class x dataset confounding -- per class, the share of clips and
      windows contributed by each source_dataset; a pair is flagged when
      either share exceeds 80%. Also reports kept clips with zero windows
      per class and split, plus a class x original sample_rate table with
      per-class share of clips below 16 kHz and a flag when one sample
      rate contributes more than 80% of a class's clips.
   3. Window label-support proxy -- nearest-centroid margin: centroids are
      computed from the train features; per class x split the fraction of
      windows closer to another class centroid than to their own
      (margin <= 0) marks centroid-confusable windows (a separability
      proxy, not a noise estimate).
  4. Architecture time-axis trace -- static shape and parameter walk of
     build_cnn() showing how the 64x5 time axis collapses to a single
     frame before flatten.

The test split is never opened (src loaders reject any split other than
train/val). Existing artifacts are only read; the script refuses to write
into a non-empty output directory.

Usage:
  .venv/Scripts/python scripts/run_phase3a_diagnostics.py
  .venv/Scripts/python scripts/run_phase3a_diagnostics.py --splits train
  .venv/Scripts/python scripts/run_phase3a_diagnostics.py --out-dir scripts/results/diagnostics_run2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config

ALLOWED_SPLITS = ("train", "val")
LEDGER_PATH = config.PROVENANCE_DIR / "phase2a_clip_ledger.csv"
FEATURES_SRC = config.PROJECT_ROOT / "src" / "preprocessing" / "features.py"
TRAINING_SRC = config.PROJECT_ROOT / "scripts" / "run_phase3a_training.py"
DEFAULT_OUT_DIR = config.PROJECT_ROOT / "scripts" / "results" / "diagnostics"
FLOOR_VALUE = 0.0
SATURATION_THRESHOLD = 0.95
CONFOUND_THRESHOLD = 0.80
REQUIRED_MELS = 64
LOW_BINS_END = 48
BELOW_RATE_HZ = 16000
REQUIRED_STATS_KEYS = ("db_min", "db_max", "db_ref", "db_amin", "db_top_db")
EXPECTED_PARAMS = 56657


def _find_line(text: str, needle: str) -> int | None:
    """Return the 1-based line number of the first line containing needle."""
    for number, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return number
    return None


def _fmt(value: float, digits: int) -> str:
    """Format a float for tables; render non-finite values as '-'."""
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table from preformatted cell strings."""
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def _write_md(path: Path, title: str, sections: list[str]) -> None:
    """Write a Markdown report with one top-level title and given sections."""
    text = "\n\n".join([f"# {title}", *sections]) + "\n"
    path.write_text(text, encoding="utf-8")


def _load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Memory-map features and labels for one allowed split and sanity-check them."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split not allowed (test is never opened): {split}")
    x = np.load(config.PROCESSED_DATA_DIR / f"features_{split}.npy", mmap_mode="r")
    y = np.load(config.PROCESSED_DATA_DIR / f"labels_{split}.npy")
    y = np.asarray(y)
    if x.shape[0] != y.shape[0]:
        raise RuntimeError(f"{split}: features {x.shape[0]} != labels {y.shape[0]}")
    if tuple(x.shape[1:]) != tuple(config.SPEC_SHAPE):
        raise RuntimeError(
            f"{split}: feature shape {tuple(x.shape[1:])} != {tuple(config.SPEC_SHAPE)}"
        )
    if y.size and (int(y.min()) < 0 or int(y.max()) >= config.NUM_TRIGGER_CLASSES):
        raise RuntimeError(f"{split}: labels outside [0, {config.NUM_TRIGGER_CLASSES})")
    index_path = config.PROCESSED_DATA_DIR / f"window_index_{split}.csv"
    n_index = len(pd.read_csv(index_path, usecols=["window_index"]))
    if n_index != x.shape[0]:
        raise RuntimeError(f"{split}: window_index rows {n_index} != features {x.shape[0]}")
    return x, y


def audit_normalization(out_dir: Path, splits: tuple[str, ...]) -> dict:
    """Section 1: global-vs-per-window verdict plus per-class floor/saturation stats."""
    print("[diagnostics] 1/4 normalization audit", flush=True)
    with open(config.FEATURE_STATS_PATH, encoding="utf-8") as handle:
        stats = json.load(handle)
    print(
        f"[diagnostics]   feature_stats.json keys: {', '.join(sorted(stats))}",
        flush=True,
    )
    missing_keys = [key for key in REQUIRED_STATS_KEYS if key not in stats]
    if missing_keys:
        raise SystemExit(
            f"feature_stats.json is missing required key(s): "
            f"{', '.join(missing_keys)} (path: {config.FEATURE_STATS_PATH})"
        )
    if int(config.N_MELS) < REQUIRED_MELS:
        raise SystemExit(
            f"config.N_MELS={config.N_MELS} < {REQUIRED_MELS}: "
            "the bins 48-63 audit needs at least 64 mel bins"
        )
    lo = float(stats["db_min"])
    hi = float(stats["db_max"])
    ref = float(stats["db_ref"])
    amin = float(stats["db_amin"])
    top_db = stats["db_top_db"]
    global_bounds = "db_min" in stats and "db_max" in stats
    per_split_bounds = any(f"db_min_{s}" in stats for s in ("train", "val", "test"))
    verdict = (
        "GLOBAL: one fixed (db_min, db_max) pair frozen from the train split"
        if global_bounds and not per_split_bounds
        else "NOT GLOBAL: per-split or per-window bounds found in feature_stats.json"
    )
    source = FEATURES_SRC.read_text(encoding="utf-8")
    markers = [
        ("_scaled definition", "def _scaled"),
        ("scaling formula", "(db - lo) / (hi - lo)"),
        ("clip to [0, 1]", "np.clip(out, 0.0, 1.0"),
        ("dB conversion (top_db=None)", "top_db=None"),
        ("bounds frozen from train pass A", "db_lo = min(db_lo"),
        ("train scaling applies frozen bounds", "X_train[cursor : cursor + k] = _scaled"),
        ("val/test scaling applies frozen bounds", "parts_X.append(_scaled"),
    ]
    found = [(label, _find_line(source, needle)) for label, needle in markers]
    rel_src = "src/preprocessing/features.py"
    formula_lines = [
        f"    db = librosa.power_to_db(S, ref={ref}, amin={amin}, top_db={top_db})",
        f"    normalized = clip((db - ({lo})) / (({hi}) - ({lo})), 0.0, 1.0)",
        "    [source: " + rel_src + " lines " + str(_fmt_line_range(found)) + "]",
    ]
    print(
        f"[diagnostics]   {verdict} bounds [{lo}, {hi}] applied to every window",
        flush=True,
    )
    rows: list[dict] = []
    for split in splits:
        x, y = _load_split(split)
        totals = {"n_values": 0, "n_floor": 0, "n_ge": 0, "sum": 0.0, "sumsq": 0.0}
        for class_id in range(config.NUM_TRIGGER_CLASSES):
            idx = np.flatnonzero(y == class_id)
            n_windows = int(idx.size)
            name = config.TRIGGER_CLASS_NAMES[class_id]
            if n_windows == 0:
                rows.append(
                    {
                        "split": split,
                        "class_id": class_id,
                        "class_name": name,
                        "n_windows": 0,
                        "n_values": 0,
                        "n_at_floor": 0,
                        "frac_at_floor": np.nan,
                        "mean_norm": np.nan,
                        "std_norm": np.nan,
                        "mean_bins_0_47": np.nan,
                        "mean_bins_48_63": np.nan,
                        "frac_at_floor_bins_48_63": np.nan,
                        "n_ge_095": 0,
                        "frac_ge_095": np.nan,
                    }
                )
                continue
            chunk = np.asarray(x[idx])
            n_values = int(chunk.size)
            n_floor = int(np.count_nonzero(chunk == FLOOR_VALUE))
            n_ge = int(np.count_nonzero(chunk >= SATURATION_THRESHOLD))
            mean = float(chunk.mean(dtype=np.float64))
            std = float(chunk.std(dtype=np.float64))
            low_bins = chunk[:, :LOW_BINS_END]
            high_bins = chunk[:, LOW_BINS_END:REQUIRED_MELS]
            mean_bins_0_47 = float(low_bins.mean(dtype=np.float64))
            mean_bins_48_63 = float(high_bins.mean(dtype=np.float64))
            frac_floor_bins_48_63 = float(
                np.count_nonzero(high_bins == FLOOR_VALUE) / high_bins.size
            )
            squares = chunk.astype(np.float64) ** 2
            totals["n_values"] += n_values
            totals["n_floor"] += n_floor
            totals["n_ge"] += n_ge
            totals["sum"] += mean * n_values
            totals["sumsq"] += float(squares.sum())
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": name,
                    "n_windows": n_windows,
                    "n_values": n_values,
                    "n_at_floor": n_floor,
                    "frac_at_floor": n_floor / n_values,
                    "mean_norm": mean,
                    "std_norm": std,
                    "mean_bins_0_47": mean_bins_0_47,
                    "mean_bins_48_63": mean_bins_48_63,
                    "frac_at_floor_bins_48_63": frac_floor_bins_48_63,
                    "n_ge_095": n_ge,
                    "frac_ge_095": n_ge / n_values,
                }
            )
        overall_mean = totals["sum"] / totals["n_values"]
        overall_var = totals["sumsq"] / totals["n_values"] - overall_mean**2
        print(
            f"[diagnostics]   {split}: {int(totals['n_values']):,} values, "
            f"overall floor frac {totals['n_floor'] / totals['n_values']:.4f}, "
            f"overall mean {overall_mean:.4f}, std {np.sqrt(max(overall_var, 0.0)):.4f}, "
            f"frac >= {SATURATION_THRESHOLD}: {totals['n_ge'] / totals['n_values']:.4f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    csv_path = out_dir / "normalization_per_class.csv"
    frame.to_csv(csv_path, index=False)
    ref_rows = []
    for label, line in found:
        location = f"{rel_src}:{line}" if line is not None else "NOT FOUND"
        ref_rows.append([label, location])
    ref_table = _md_table(["code reference", "file:line"], ref_rows)
    class_rows = []
    for row in rows:
        class_rows.append(
            [
                row["split"],
                str(row["class_id"]),
                row["class_name"],
                str(row["n_windows"]),
                str(row["n_at_floor"]),
                _fmt(row["frac_at_floor"], 6),
                _fmt(row["mean_norm"], 4),
                _fmt(row["std_norm"], 4),
                _fmt(row["mean_bins_0_47"], 4),
                _fmt(row["mean_bins_48_63"], 4),
                _fmt(row["frac_at_floor_bins_48_63"], 6),
                str(row["n_ge_095"]),
                _fmt(row["frac_ge_095"], 6),
            ]
        )
    class_table = _md_table(
        [
            "split",
            "class_id",
            "class_name",
            "n_windows",
            "n_at_floor",
            "frac_at_floor",
            "mean_norm",
            "std_norm",
            "mean_bins_0_47",
            "mean_bins_48_63",
            "frac_at_floor_bins_48_63",
            "n_ge_095",
            "frac_ge_095",
        ],
        class_rows,
    )
    scored = frame.dropna(subset=["frac_at_floor"])
    worst_floor = scored.loc[scored["frac_at_floor"].idxmax()] if len(scored) else None
    worst_sat = scored.loc[scored["frac_ge_095"].idxmax()] if len(scored) else None
    absent = frame.loc[frame["n_windows"] == 0, ["split", "class_name"]]
    absent_note = (
        "none"
        if absent.empty
        else ", ".join(f"{r.split}/{r.class_name}" for r in absent.itertuples())
    )
    _write_md(
        out_dir / "normalization_audit.md",
        "Normalization audit",
        [
            "\n".join(
                [
                    "## Verdict",
                    "",
                    f"- feature_stats.json contains one global db_min/db_max pair: "
                    f"**{verdict}**",
                    "- exact formula:",
                    "",
                    "```",
                    *formula_lines,
                    "```",
                    "",
                    f"- normalization is **global (fixed train-frozen dB bounds), "
                    f"NOT per-window**: a single (db_min, db_max) pair, collected once "
                    f"from the train split pass A and recorded in "
                    f"data/processed/feature_stats.json, is applied to every window of "
                    f"every split ({', '.join(splits)}).",
                    f"- floor: db_min={lo} maps to normalized {FLOOR_VALUE}; any value "
                    f"at {FLOOR_VALUE} is dB <= {lo} (the -100 dB floor).",
                    f"- saturation check: fraction of values >= {SATURATION_THRESHOLD}.",
                ]
            ),
            "\n".join(["## Code references", "", ref_table]),
            "\n".join(
                [
                    "## Per-class value statistics",
                    "",
                    "Test split not examined. Non-finite stats mean the class has no "
                    "windows in that split.",
                    "",
                    "Bin columns: mean_bins_0_47 is the mean normalized value over "
                    "mel bins 0-47 (below about 4 kHz), mean_bins_48_63 over bins "
                    "48-63 (about 4 kHz and above), and frac_at_floor_bins_48_63 "
                    "is the fraction of values exactly at 0.0 (-100 dB floor) "
                    "within bins 48-63 only.",
                    "",
                    class_table,
                    "",
                    f"- highest floor fraction: "
                    f"{_describe_scored(worst_floor, 'frac_at_floor')}",
                    f"- highest saturation fraction: "
                    f"{_describe_scored(worst_sat, 'frac_ge_095')}",
                    f"- classes absent from the reported splits: {absent_note}",
                ]
            ),
            "\n".join(
                [
                    "## Artifacts",
                    "",
                    f"- {csv_path.name}",
                    f"- source of bounds: {config.FEATURE_STATS_PATH}",
                ]
            ),
        ],
    )
    return {
        "verdict": verdict,
        "lo": lo,
        "hi": hi,
        "worst_floor": _describe_scored(worst_floor, "frac_at_floor"),
        "worst_sat": _describe_scored(worst_sat, "frac_ge_095"),
        "absent": absent_note,
    }


def _fmt_line_range(found: list[tuple[str, int | None]]) -> str:
    """Compact line list for the formula block, preserving marker order."""
    lines = [str(line) for _label, line in found if line is not None]
    return ", ".join(lines) if lines else "none"


def _describe_scored(row: pd.Series | None, column: str) -> str:
    """One-line description of an extreme row, or '-' when unavailable."""
    if row is None:
        return "-"
    return (
        f"{row['split']}/{row['class_name']} = {row[column]:.6f} "
        f"({int(row['n_windows'])} windows)"
    )


def confounding_report(out_dir: Path, splits: tuple[str, ...]) -> dict:
    """Section 2: per class, source_dataset shares with >80% flags."""
    print("[diagnostics] 2/4 class x dataset confounding", flush=True)
    ledger = pd.read_csv(
        LEDGER_PATH,
        usecols=[
            "clip_id",
            "source_dataset",
            "project_class",
            "class_id",
            "sample_rate",
            "status",
        ],
    )
    ledger = ledger.loc[ledger["status"] == "kept"].drop_duplicates(subset=["clip_id"])
    clip_splits = pd.read_csv(
        config.SPLIT_DATA_DIR / "clip_splits.csv", usecols=["clip_id", "split"]
    )
    clip_splits = clip_splits.loc[clip_splits["split"].isin(splits)]
    clips = ledger.merge(
        clip_splits, on="clip_id", how="inner", validate="one_to_one"
    )
    count_frames = []
    for split in splits:
        window_index = pd.read_csv(
            config.PROCESSED_DATA_DIR / f"window_index_{split}.csv",
            usecols=["clip_id"],
        )
        counts = window_index["clip_id"].value_counts()
        count_frames.append(
            pd.DataFrame(
                {
                    "clip_id": counts.index.to_numpy(),
                    "split": split,
                    "n_windows": counts.to_numpy(),
                }
            )
        )
    clips = clips.merge(
        pd.concat(count_frames, ignore_index=True),
        on=["clip_id", "split"],
        how="left",
        validate="one_to_one",
    )
    clips["n_windows"] = clips["n_windows"].fillna(0).astype(np.int64)
    zero_mask = clips["n_windows"] == 0
    zero_by_split = (
        clips.loc[zero_mask]
        .groupby(["split", "class_id", "project_class"], as_index=False)
        .agg(n_zero_window=("clip_id", "count"))
    )
    n_zero_total = int(zero_mask.sum())
    zero_note = (
        "none"
        if zero_by_split.empty
        else "; ".join(
            f"{r.split}/{r.project_class}: {int(r.n_zero_window)}"
            for r in zero_by_split.itertuples()
        )
    )
    print(
        f"[diagnostics]   kept clips with zero windows: {n_zero_total} "
        f"({zero_note})",
        flush=True,
    )
    agg = (
        clips.groupby(["class_id", "project_class", "source_dataset"], as_index=False)
        .agg(n_clips=("clip_id", "count"), n_windows=("n_windows", "sum"))
    )
    clip_total = agg.groupby("class_id")["n_clips"].transform("sum")
    window_total = agg.groupby("class_id")["n_windows"].transform("sum")
    agg["clip_share"] = agg["n_clips"] / clip_total
    agg["window_share"] = agg["n_windows"] / window_total
    agg["flag_over_80"] = (agg["clip_share"] > CONFOUND_THRESHOLD) | (
        agg["window_share"] > CONFOUND_THRESHOLD
    )
    agg = agg.sort_values(["class_id", "n_clips"], ascending=[True, False])
    csv_path = out_dir / "class_dataset_confusion.csv"
    agg.to_csv(csv_path, index=False)
    flagged = agg.loc[agg["flag_over_80"]]
    present = set(int(v) for v in agg["class_id"].unique())
    absent = [
        config.TRIGGER_CLASS_NAMES[cid]
        for cid in range(config.NUM_TRIGGER_CLASSES)
        if cid not in present
    ]

    rate_missing = int(clips["sample_rate"].isna().sum())
    rate_agg = (
        clips.groupby(["class_id", "project_class", "sample_rate"], dropna=False)
        .agg(n_clips=("clip_id", "count"))
        .reset_index()
    )
    rate_total = rate_agg.groupby("class_id")["n_clips"].transform("sum")
    rate_agg["clip_share"] = rate_agg["n_clips"] / rate_total
    rate_agg["sample_rate_missing"] = rate_agg["sample_rate"].isna()
    rate_agg["sample_rate"] = rate_agg["sample_rate"].fillna(-1).astype(np.int64)
    rate_agg["flag_single_rate"] = rate_agg["clip_share"] > CONFOUND_THRESHOLD
    rate_agg = rate_agg.sort_values(
        ["class_id", "n_clips"], ascending=[True, False]
    )
    rate_csv_path = out_dir / "class_sample_rate.csv"
    rate_agg.to_csv(rate_csv_path, index=False)
    zero_csv_path = out_dir / "zero_window_clips.csv"
    zero_by_split.to_csv(zero_csv_path, index=False)
    rate_flagged = rate_agg.loc[rate_agg["flag_single_rate"]]

    below_16k = (
        clips.loc[clips["sample_rate"].notna() & (clips["sample_rate"] < 16000)]
        .groupby("class_id")
        .size()
    )
    rate_class_totals = clips.groupby("class_id").size()
    below_lines = []
    for class_id in range(config.NUM_TRIGGER_CLASSES):
        name = config.TRIGGER_CLASS_NAMES[class_id]
        total = int(rate_class_totals.get(class_id, 0))
        below = int(below_16k.get(class_id, 0))
        share = below / total if total else float("nan")
        below_lines.append(
            f"{name}: {below}/{total} ({share:.4f})"
        )
    print(
        "[diagnostics]   share of clips below 16000 Hz per class: "
        + "; ".join(below_lines),
        flush=True,
    )

    def _rows(frame: pd.DataFrame) -> list[list[str]]:
        return [
            [
                str(int(row.class_id)),
                row.project_class,
                row.source_dataset,
                str(int(row.n_clips)),
                str(int(row.n_windows)),
                f"{row.clip_share:.4f}",
                f"{row.window_share:.4f}",
            ]
            for row in frame.itertuples()
        ]

    headers = [
        "class_id",
        "class_name",
        "source_dataset",
        "n_clips",
        "n_windows",
        "clip_share",
        "window_share",
    ]
    flag_rows = _rows(flagged)

    def _rate_rows(frame: pd.DataFrame) -> list[list[str]]:
        out = []
        for row in frame.itertuples():
            rate = "missing" if row.sample_rate_missing else str(int(row.sample_rate))
            out.append(
                [
                    str(int(row.class_id)),
                    row.project_class,
                    rate,
                    str(int(row.n_clips)),
                    f"{row.clip_share:.4f}",
                    "yes" if row.flag_single_rate else "no",
                ]
            )
        return out

    rate_headers = [
        "class_id",
        "class_name",
        "sample_rate",
        "n_clips",
        "clip_share",
        "flag_single_rate",
    ]

    def _zero_rows(frame: pd.DataFrame) -> list[list[str]]:
        return [
            [
                str(int(row.class_id)),
                row.project_class,
                row.split,
                str(int(row.n_zero_window)),
            ]
            for row in frame.itertuples()
        ]

    zero_headers = ["class_id", "class_name", "split", "n_zero_window"]

    below_table = _md_table(
        ["class_name: clips below 16000 Hz (share)"],
        [[line] for line in below_lines],
    )

    _write_md(
        out_dir / "class_dataset_confusion.md",
        f"Class x dataset confounding ({', '.join(splits)})",
        [
            "\n".join(
                [
                    "## Method",
                    "",
                    f"Per class, each source_dataset's share of that class's clips and "
                    f"windows over splits ({', '.join(splits)}). A pair is flagged when "
                    f"either share exceeds {CONFOUND_THRESHOLD:.0%}. The test split is "
                    f"excluded; test-split metadata is never read.",
                ]
            ),
            "\n".join(
                [
                    f"## Flagged pairs (share > {CONFOUND_THRESHOLD:.0%})",
                    "",
                    (
                        _md_table(headers, flag_rows)
                        if flag_rows
                        else "No class x dataset pair exceeds the threshold."
                    ),
                ]
            ),
            "\n".join(
                [
                    "## All pairs",
                    "",
                    _md_table(headers, _rows(agg)),
                ]
            ),
            "\n".join(
                [
                    "## Zero-window clips (kept clips with no extracted windows)",
                    "",
                    f"- total kept clips with zero windows: **{n_zero_total}** "
                    f"(these clips contribute {n_zero_total} labels but no windows; "
                    f"n_windows is filled with 0, not an error)",
                    "",
                    (
                        _md_table(zero_headers, _zero_rows(zero_by_split))
                        if not zero_by_split.empty
                        else "No kept clip in these splits has zero windows."
                    ),
                ]
            ),
            "\n".join(
                [
                    "## Class x original sample rate",
                    "",
                    f"A class is flagged when one sample_rate contributes more than "
                    f"{CONFOUND_THRESHOLD:.0%} of its kept clips "
                    f"(missing sample_rate rows are reported as 'missing').",
                    "",
                    _md_table(rate_headers, _rate_rows(rate_agg)),
                    "",
                    f"- classes with a single dominant sample rate: "
                    f"{len(rate_flagged)}",
                    f"- ledger rows with missing sample_rate: {rate_missing}",
                ]
            ),
            "\n".join(
                [
                    "## Share of clips below 16000 Hz per class",
                    "",
                    below_table,
                ]
            ),
            "\n".join(
                [
                    "## Notes",
                    "",
                    f"- classes with no clips in these splits: "
                    f"{', '.join(absent) if absent else 'none'}",
                    f"- flagged pairs: {len(flag_rows)}",
                    f"- source: {LEDGER_PATH} (status=kept) joined to clip_splits.csv "
                    f"and window_index_*.csv on clip_id",
                ]
            ),
        ],
    )
    print(
        f"[diagnostics]   {len(agg)} class x dataset pairs, "
        f"{len(flag_rows)} flagged above {CONFOUND_THRESHOLD:.0%}",
        flush=True,
    )
    return {
        "n_pairs": int(len(agg)),
        "n_flagged": len(flag_rows),
        "flagged": [
            f"{r.project_class}/{r.source_dataset} "
            f"(clips {r.clip_share:.3f}, windows {r.window_share:.3f})"
            for r in flagged.itertuples()
        ],
        "absent": absent,
        "n_zero_total": n_zero_total,
        "zero_note": zero_note,
        "n_rate_flagged": int(len(rate_flagged)),
        "rate_flagged": [
            f"{r.project_class} ({'missing' if r.sample_rate_missing else int(r.sample_rate)} Hz: {r.clip_share:.3f})"
            for r in rate_flagged.itertuples()
        ],
        "rate_missing": rate_missing,
    }


def _compute_centroids(
    x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean feature vector per class from train; NaN row and count 0 if absent."""
    flat_dim = int(np.prod(config.SPEC_SHAPE))
    centroids = np.full((config.NUM_TRIGGER_CLASSES, flat_dim), np.nan)
    counts = np.zeros(config.NUM_TRIGGER_CLASSES, dtype=np.int64)
    for class_id in range(config.NUM_TRIGGER_CLASSES):
        idx = np.flatnonzero(y == class_id)
        counts[class_id] = idx.size
        if idx.size == 0:
            continue
        chunk = np.asarray(x[idx]).reshape(idx.size, flat_dim)
        centroids[class_id] = chunk.mean(axis=0, dtype=np.float64)
    return centroids, counts


def label_support_report(out_dir: Path, splits: tuple[str, ...]) -> dict:
    """Section 3: nearest-centroid margin support per class and split."""
    print("[diagnostics] 3/4 window label-support proxy", flush=True)
    train_x, train_y = _load_split("train")
    centroids, train_counts = _compute_centroids(train_x, train_y)
    centroid_sq = np.einsum("ij,ij->i", centroids, centroids)
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {"train": (train_x, train_y)}
    rows: list[dict] = []
    for split in splits:
        if split not in loaded:
            loaded[split] = _load_split(split)
        x, y = loaded[split]
        for class_id in range(config.NUM_TRIGGER_CLASSES):
            name = config.TRIGGER_CLASS_NAMES[class_id]
            idx = np.flatnonzero(y == class_id)
            if idx.size == 0 or not np.isfinite(centroids[class_id]).all():
                rows.append(
                    {
                        "split": split,
                        "class_id": class_id,
                        "class_name": name,
                        "n_windows": int(idx.size),
                        "frac_centroid_confusable": np.nan,
                        "mean_margin": np.nan,
                        "median_margin": np.nan,
                    }
                )
                continue
            chunk = np.asarray(x[idx]).reshape(idx.size, -1).astype(np.float64)
            own_sq = np.einsum("ij,ij->i", chunk, chunk)
            cross = chunk @ centroids.T
            distances = own_sq[:, None] - 2.0 * cross + centroid_sq[None, :]
            own = distances[:, class_id].copy()
            distances[:, ~np.isfinite(centroid_sq)] = np.inf
            distances[:, class_id] = np.inf
            other = distances.min(axis=1)
            margin = other - own
            if not np.isfinite(margin).all():
                raise RuntimeError(f"{split}/{name}: non-finite margin")
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": name,
                    "n_windows": int(idx.size),
                    "frac_centroid_confusable": float(np.mean(margin <= 0.0)),
                    "mean_margin": float(margin.mean()),
                    "median_margin": float(np.median(margin)),
                }
            )
        print(f"[diagnostics]   {split}: margins computed", flush=True)
    frame = pd.DataFrame(rows)
    csv_path = out_dir / "label_support.csv"
    frame.to_csv(csv_path, index=False)
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["split"],
                str(row["class_id"]),
                row["class_name"],
                str(row["n_windows"]),
                _fmt(row["frac_centroid_confusable"], 4),
                _fmt(row["mean_margin"], 4),
                _fmt(row["median_margin"], 4),
                str(int(train_counts[row["class_id"]])),
            ]
        )
    table = _md_table(
        [
            "split",
            "class_id",
            "class_name",
            "n_windows",
            "frac_centroid_confusable",
            "mean_margin",
            "median_margin",
            "train_windows",
        ],
        table_rows,
    )
    scored = frame.dropna(subset=["frac_centroid_confusable"])
    worst = []
    if len(scored):
        order = scored.sort_values("frac_centroid_confusable", ascending=False)
        for row in order.head(5).itertuples():
            worst.append(
                f"{row.split}/{row.class_name}: {row.frac_centroid_confusable:.4f} "
                f"centroid-confusable ({int(row.n_windows)} windows)"
            )
    _write_md(
        out_dir / "label_support.md",
        "Window label-support proxy (nearest-centroid margin)",
        [
            "\n".join(
                [
                    "## Definition",
                    "",
                    "Centroids mu_c are mean train features per class (320 values = "
                    "64 x 5 x 1). For each window x with clip label c:",
                    "",
                    "```",
                    "margin(x) = min_{j != c} ||x - mu_j||^2  -  ||x - mu_c||^2",
                    "centroid-confusable window  <=>  margin(x) <= 0",
                    "```",
                    "",
                    "A positive margin means the window is closer to its own class "
                    "centroid than to every other class centroid. The proxy is content "
                    "based: all windows inherit the clip label, this measures whether "
                    "the window's spectrogram supports that label.",
                    "This is a separability proxy, not a noise estimate.",
                    "Centroids always come from the train split, even when only val "
                    "is reported.",
                ]
            ),
            "\n".join(["## Per class and split", "", table]),
            "\n".join(
                [
                    "## Highest centroid-confusable fractions",
                    "",
                    *([f"- {item}" for item in worst] if worst else ["- none"]),
                ]
            ),
            "\n".join(["## Artifacts", "", f"- {csv_path.name}"]),
        ],
    )
    return {"worst": worst, "n_rows": len(rows)}


def architecture_trace(out_dir: Path) -> dict:
    """Section 4: static time-axis and parameter trace of build_cnn()."""
    print("[diagnostics] 4/4 architecture time-axis trace", flush=True)
    source = TRAINING_SRC.read_text(encoding="utf-8")
    build_line = _find_line(source, "def build_cnn")
    pool_lines = [
        number
        for number, line in enumerate(source.splitlines(), start=1)
        if "MaxPooling2D" in line
    ]
    height = int(config.N_MELS)
    frames = int(config.SPEC_FRAMES)
    c1 = int(config.CONV1_FILTERS)
    c2 = int(config.CONV2_FILTERS)
    c3 = int(config.CONV3_FILTERS)
    dense = int(config.DENSE_UNITS)
    n_classes = int(config.NUM_TRIGGER_CLASSES)
    entries: list[tuple[str, str, int, int]] = []

    def record(layer: str, shape: tuple[int, ...], time_axis: int, params: int) -> None:
        entries.append((layer, str(shape), time_axis, params))

    record("input mel_window", (height, frames, 1), frames, 0)
    record(
        f"Conv2D {c1} 3x3 same relu",
        (height, frames, c1),
        frames,
        3 * 3 * 1 * c1 + c1,
    )
    height, frames = (height - 2) // 2 + 1, (frames - 2) // 2 + 1
    record("MaxPooling2D 2x2", (height, frames, c1), frames, 0)
    record(
        f"Conv2D {c2} 3x3 same relu",
        (height, frames, c2),
        frames,
        3 * 3 * c1 * c2 + c2,
    )
    height, frames = (height - 2) // 2 + 1, (frames - 2) // 2 + 1
    record("MaxPooling2D 2x2", (height, frames, c2), frames, 0)
    record(
        f"Conv2D {c3} 3x3 same relu",
        (height, frames, c3),
        frames,
        3 * 3 * c2 * c3 + c3,
    )
    flat = height * frames * c3
    record("Flatten", (flat,), frames, 0)
    record(f"Dense {dense} relu", (dense,), frames, flat * dense + dense)
    record(f"Dropout {config.DROPOUT_RATE}", (dense,), frames, 0)
    record(
        f"Dense {n_classes} softmax",
        (n_classes,),
        frames,
        dense * n_classes + n_classes,
    )
    total_params = sum(entry[3] for entry in entries)
    time_chain = " -> ".join(str(entry[2]) for entry in entries)
    table_rows = [
        [layer, shape, str(time_axis), f"{params:,}"]
        for layer, shape, time_axis, params in entries
    ]
    table = _md_table(
        ["layer", "output shape", "time frames", "params"], table_rows
    )
    refs = [
        ["build_cnn definition", f"scripts/run_phase3a_training.py:{build_line}"],
        [
            "pool layers",
            ", ".join(
                f"scripts/run_phase3a_training.py:{number}" for number in pool_lines
            ),
        ],
    ]
    ref_table = _md_table(["code reference", "file:line"], refs)
    _write_md(
        out_dir / "architecture_trace.md",
        "Architecture time-axis trace (build_cnn)",
        [
            "\n".join(["## Source references", "", ref_table]),
            "\n".join(["## Layer trace", "", table]),
            "\n".join(
                [
                    "## Time-axis summary",
                    "",
                    f"- frames through the network: {time_chain}",
                    f"- the 5-frame time axis collapses to {entries[4][2]} frame at the "
                    f"second pool (64x5 -> 32x2 -> 16x1); flatten sees a single frame "
                    f"per window ({flat} values), so temporal evolution inside the "
                    f"200 ms window is discarded before the dense layers.",
                    f"- total parameters: {total_params:,} "
                    f"({'under' if total_params < 100_000 else 'OVER'} the 100,000 "
                    f"parameter ceiling)",
                ]
                + (
                    [
                        f"- PARAM MISMATCH: trace={total_params:,} vs expected "
                        f"{EXPECTED_PARAMS:,}"
                    ]
                    if total_params != EXPECTED_PARAMS
                    else []
                )
                + [
                    f"- input from config.SPEC_SHAPE = {tuple(config.SPEC_SHAPE)}, "
                    f"filters {c1}/{c2}/{c3}, dense {dense}, classes {n_classes}",
                ]
            ),
        ],
    )
    print(
        f"[diagnostics]   time frames {time_chain}; total params {total_params:,}",
        flush=True,
    )
    if total_params != EXPECTED_PARAMS:
        print(
            f"[diagnostics]   PARAM MISMATCH: trace={total_params:,} vs "
            f"expected {EXPECTED_PARAMS:,}",
            flush=True,
        )
    return {"total_params": total_params, "time_chain": time_chain}


def _write_summary(out_dir: Path, splits: tuple[str, ...], info: list[dict]) -> None:
    """Combine the four section verdicts into summary.md."""
    norm, conf, label, arch = info
    flagged_lines = conf["flagged"] if conf["flagged"] else ["- none"]
    worst_lines = label["worst"] if label["worst"] else ["- none"]
    _write_md(
        out_dir / "summary.md",
        "Phase 3A diagnostics summary",
        [
            f"Generated by scripts/run_phase3a_diagnostics.py "
            f"(splits: {', '.join(splits)}; test never opened).",
            "\n".join(
                [
                    "## 1. Normalization",
                    "",
                    f"- {norm['verdict']}",
                    f"- bounds: [{norm['lo']}, {norm['hi']}] (frozen from train)",
                    f"- worst floor fraction: {norm['worst_floor']}",
                    f"- worst saturation fraction: {norm['worst_sat']}",
                    f"- absent classes: {norm['absent']}",
                ]
            ),
            "\n".join(
                [
                    "## 2. Class x dataset confounding",
                    "",
                    f"- pairs: {conf['n_pairs']}, flagged (>80%): {conf['n_flagged']}",
                    *(f"- {item}" for item in flagged_lines),
                    f"- zero-window kept clips: {conf['n_zero_total']} "
                    f"({conf['zero_note']})",
                    f"- classes with a single dominant sample rate (>80%): "
                    f"{conf['n_rate_flagged']} "
                    f"({', '.join(conf['rate_flagged']) if conf['rate_flagged'] else 'none'})",
                    f"- ledger rows with missing sample_rate: {conf['rate_missing']}",
                ]
            ),
            "\n".join(
                [
                    "## 3. Window label-support proxy",
                    "",
                    "highest centroid-confusable fractions (margin <= 0):",
                    *(f"- {item}" for item in worst_lines),
                ]
            ),
            "\n".join(
                [
                    "## 4. Architecture time-axis trace",
                    "",
                    f"- time frames: {arch['time_chain']}",
                    f"- total parameters: {arch['total_params']:,}",
                ]
            ),
            "\n".join(
                [
                    "## Reports",
                    "",
                    "- normalization_audit.md / normalization_per_class.csv",
                    "- class_dataset_confusion.md / class_dataset_confusion.csv",
                    "- class_sample_rate.csv / zero_window_clips.csv",
                    "- label_support.md / label_support.csv",
                    "- architecture_trace.md",
                ]
            ),
        ],
    )


def main() -> int:
    """Parse arguments, run the four sections, write reports, return exit code."""
    parser = argparse.ArgumentParser(
        description="Phase 3A diagnostics: normalization, confounding, "
        "label support, architecture trace (train/val only, read-only)"
    )
    parser.add_argument(
        "--splits",
        default="train,val",
        help="comma-separated splits to analyze; only train and val are allowed",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: scripts/results/diagnostics)",
    )
    args = parser.parse_args()
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    if not splits:
        parser.error("no splits given")
    for split in splits:
        if split not in ALLOWED_SPLITS:
            parser.error(
                f"split '{split}' is not allowed; the test split is never opened"
            )
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error(
            f"output directory is not empty: {out_dir} "
            f"(choose another --out-dir or remove it yourself)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[diagnostics] splits: {', '.join(splits)}; output: {out_dir}",
        flush=True,
    )
    info_norm = audit_normalization(out_dir, splits)
    info_conf = confounding_report(out_dir, splits)
    info_label = label_support_report(out_dir, splits)
    info_arch = architecture_trace(out_dir)
    _write_summary(out_dir, splits, [info_norm, info_conf, info_label, info_arch])
    print(f"[diagnostics] reports written to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
