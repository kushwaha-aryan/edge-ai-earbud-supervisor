"""D11 leakage gates for the Phase 2A outputs (approved 2026-10-08).

Run before any training may start (blueprint section 6). Checks:

  G1  every clip_id appears in exactly one split, and the split covers all
      kept ledger clips
  G2  every source group maps to exactly one split, and group_splits.csv
      agrees with clip_splits.csv
  G3  SHA-256 content hashes are disjoint across splits
  G4  per-class and overall clip-level split proportions lie within
      PROPORTION_TOLERANCE (5 pp) of the 70/15/15 targets
  G5  feature artifacts: shapes match SPEC_SHAPE, dtype float32, values
      finite and in [0, 1], labels valid, window-index row counts and
      feature_stats.json counts consistent

``run_gates`` returns ``(problems, details)``; an empty problem list means
PASS. Details carry the proportion table for the report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

PROPORTION_TOLERANCE = 0.05
SPLITS = ("train", "val", "test")


def _gate_split_membership(ledger, clip_splits, problems):
    if clip_splits["clip_id"].duplicated().any():
        n = int(clip_splits["clip_id"].duplicated().sum())
        problems.append(f"G1: {n} clip_ids appear in more than one split row")
    kept_ids = set(ledger.loc[ledger["status"] == "kept", "clip_id"])
    split_ids = set(clip_splits["clip_id"])
    missing = kept_ids - split_ids
    extra = split_ids - kept_ids
    if missing:
        problems.append(f"G1: {len(missing)} kept clips missing from splits "
                        f"(e.g. {sorted(missing)[:3]})")
    if extra:
        problems.append(f"G1: {len(extra)} split clips not kept in ledger "
                        f"(e.g. {sorted(extra)[:3]})")


def _gate_group_atomicity(clip_splits, group_splits, problems):
    per_group = clip_splits.groupby("group_key")["split"].nunique()
    bad = per_group[per_group > 1]
    for g in bad.index[:10]:
        rows = sorted(clip_splits.loc[clip_splits["group_key"] == g, "split"].unique())
        problems.append(f"G2: group {g} spans multiple splits {rows}")
    if len(bad) > 10:
        problems.append(f"G2: ... {len(bad) - 10} more multi-split groups")
    gs = group_splits.set_index("group_key")["split"]
    cs = clip_splits.groupby("group_key")["split"].first()
    common = gs.index.intersection(cs.index)
    mismatch = common[gs.loc[common] != cs.loc[common]]
    for g in mismatch[:10]:
        problems.append(
            f"G2: group_splits says {gs[g]} but clip_splits says {cs[g]} for {g}"
        )
    only_cs = cs.index.difference(gs.index)
    only_gs = gs.index.difference(cs.index)
    if len(only_cs):
        problems.append(f"G2: {len(only_cs)} groups missing from group_splits.csv")
    if len(only_gs):
        problems.append(f"G2: {len(only_gs)} groups in group_splits.csv unknown to clip_splits")


def _gate_hash_disjointness(ledger, clip_splits, problems):
    kept = ledger.loc[
        ledger["status"] == "kept", ["clip_id", "sha256"]
    ].merge(clip_splits[["clip_id", "split"]], on="clip_id", how="inner")
    per_sha = kept.groupby("sha256")["split"].nunique()
    bad = per_sha[per_sha > 1]
    for h in bad.index[:10]:
        rows = sorted(kept.loc[kept["sha256"] == h, "split"].unique())
        problems.append(f"G3: sha256 {h[:16]}... appears in splits {rows}")
    if len(bad) > 10:
        problems.append(f"G3: ... {len(bad) - 10} more shared hashes across splits")
    return {"shared_hashes_across_splits": int(len(bad))}


def _gate_proportions(clip_splits, problems):
    ratios = {"train": config.TRAIN_RATIO, "val": config.VAL_RATIO, "test": config.TEST_RATIO}
    details: dict[str, dict] = {}
    counts = clip_splits.groupby(["project_class", "split"]).size().unstack(fill_value=0)
    for s in SPLITS:
        if s not in counts.columns:
            counts[s] = 0
    max_dev = 0.0
    for cls, row in counts.iterrows():
        n = int(row.sum())
        entry = {}
        for s in SPLITS:
            frac = float(row[s]) / n if n else 0.0
            dev = abs(frac - ratios[s])
            max_dev = max(max_dev, dev)
            entry[s] = round(frac, 4)
            if dev > PROPORTION_TOLERANCE:
                problems.append(
                    f"G4: class {cls} split {s} proportion {frac:.3f} vs target "
                    f"{ratios[s]:.2f} (dev {dev:.3f} > {PROPORTION_TOLERANCE})"
                )
        entry["n_clips"] = n
        details[cls] = entry
    total = len(clip_splits)
    overall = {}
    for s in SPLITS:
        frac = float((clip_splits["split"] == s).sum()) / total
        dev = abs(frac - ratios[s])
        max_dev = max(max_dev, dev)
        overall[s] = round(frac, 4)
        if dev > PROPORTION_TOLERANCE:
            problems.append(
                f"G4: overall split {s} proportion {frac:.3f} vs target "
                f"{ratios[s]:.2f} (dev {dev:.3f} > {PROPORTION_TOLERANCE})"
            )
    overall["n_clips"] = total
    details["ALL"] = overall
    return {"proportions": details, "max_abs_deviation": round(max_dev, 4)}


def _gate_artifacts(split, problems):
    out = config.PROCESSED_DATA_DIR
    fx = out / f"features_{split}.npy"
    ly = out / f"labels_{split}.npy"
    wi = out / f"window_index_{split}.csv"
    for p in (fx, ly, wi):
        if not p.exists():
            problems.append(f"G5: missing artifact {p.name}")
            return None
    X = np.load(fx, mmap_mode="r")
    y = np.load(ly)
    idx = pd.read_csv(wi)
    if tuple(X.shape[1:]) != tuple(config.SPEC_SHAPE):
        problems.append(
            f"G5: {split} feature shape {tuple(X.shape[1:])} != SPEC_SHAPE "
            f"{tuple(config.SPEC_SHAPE)}"
        )
    if X.dtype != np.float32:
        problems.append(f"G5: {split} feature dtype {X.dtype} != float32")
    xmin = xmax = None
    if not np.isfinite(np.asarray(X)).all():
        problems.append(f"G5: {split} features contain non-finite values")
    else:
        xmin, xmax = float(np.asarray(X).min()), float(np.asarray(X).max())
        if xmin < 0.0 or xmax > 1.0:
            problems.append(
                f"G5: {split} features outside [0,1]: [{xmin:.4f}, {xmax:.4f}]"
            )
    if y.shape[0] != X.shape[0]:
        problems.append(
            f"G5: {split} labels {y.shape[0]} != features {X.shape[0]}"
        )
    if y.size and (int(y.min()) < 0 or int(y.max()) >= config.NUM_TRIGGER_CLASSES):
        problems.append(
            f"G5: {split} labels outside [0, {config.NUM_TRIGGER_CLASSES}): "
            f"[{int(y.min())}, {int(y.max())}]"
        )
    if len(idx) != X.shape[0]:
        problems.append(
            f"G5: {split} window_index rows {len(idx)} != features {X.shape[0]}"
        )
    elif not np.array_equal(idx["class_id"].to_numpy(), y):
        problems.append(f"G5: {split} window_index class_id != labels")
    return {
        "windows": int(X.shape[0]),
        "clips": int(idx["clip_id"].nunique()),
        "min": xmin,
        "max": xmax,
    }


def run_gates(
    ledger: pd.DataFrame,
    clip_splits: pd.DataFrame,
    group_splits: pd.DataFrame,
    stats_path: Path | None = None,
) -> tuple[list[str], dict]:
    """Run G1-G5; return (problems, details). Empty problems means PASS."""
    stats_path = Path(stats_path or (config.PROCESSED_DATA_DIR / "feature_stats.json"))
    problems: list[str] = []
    details: dict = {}

    _gate_split_membership(ledger, clip_splits, problems)
    _gate_group_atomicity(clip_splits, group_splits, problems)
    details.update(_gate_hash_disjointness(ledger, clip_splits, problems))
    details.update(_gate_proportions(clip_splits, problems))

    artifact_details = {}
    for split in SPLITS:
        d = _gate_artifacts(split, problems)
        if d:
            artifact_details[split] = d
    details["artifacts"] = artifact_details

    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        for split in SPLITS:
            expected = stats.get("splits", {}).get(split, {})
            got = artifact_details.get(split, {})
            if expected and got and expected.get("windows") != got.get("windows"):
                problems.append(
                    f"G5: feature_stats.json says {split} windows="
                    f"{expected.get('windows')} but artifacts have {got.get('windows')}"
                )
        if not stats.get("db_max", 0) > stats.get("db_min", 0):
            problems.append("G5: feature_stats.json dB range is not increasing")
        details["db_range"] = [stats.get("db_min"), stats.get("db_max")]
    else:
        problems.append("G5: feature_stats.json missing")

    details["passed"] = not problems
    details["problems"] = problems
    return problems, details
