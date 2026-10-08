"""Phase 1.5 — Dataset Sanity & Validation (read-only over raw audio).

Scans every file referenced by data/manifests/master_audio_manifest.csv,
collects health metrics, and writes:
  scripts/results/dataset_validation/file_health.csv
  scripts/results/dataset_validation/validation_data.json
  scripts/results/dataset_validation_summary.csv

Never modifies audio or the master manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data" / "manifests" / "master_audio_manifest.csv"
OUTDIR = ROOT / "scripts" / "results"
VALDIR = OUTDIR / "dataset_validation"

# ---- thresholds (reported in the validation report) ----
T_EXTREME_SHORT = 0.5   # s
T_SHORT = 1.0           # s
T_LONG = 15.0           # s
T_VERY_LONG = 30.0      # s
T_SILENT_DB = -60.0     # dBFS RMS
T_LOW_ENERGY_DB = -45.0 # dBFS RMS
T_CLIP_PEAK = 0.999     # |sample| at/above this counts as full scale
T_CLIP_FRAC = 0.001     # >=0.1% of samples at full scale => flagged clipped


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_file(rel_path: str) -> dict:
    row = {
        "exists": False, "size_bytes": 0, "readable": False, "format": None,
        "subtype": None, "duration": math.nan, "sample_rate": math.nan,
        "channels": math.nan, "rms_db": math.nan, "peak": math.nan,
        "clip_frac": math.nan, "sha256": None, "error": None,
    }
    path = ROOT / rel_path
    if not path.exists():
        row["error"] = "missing"
        return row
    row["exists"] = True
    row["size_bytes"] = path.stat().st_size
    if row["size_bytes"] == 0:
        row["error"] = "zero_byte"
        return row
    try:
        info = sf.info(str(path))
        row["format"] = info.format
        row["subtype"] = info.subtype
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.size == 0:
            row["error"] = "zero_frames"
            return row
        row["readable"] = True
        row["duration"] = data.shape[0] / sr
        row["sample_rate"] = sr
        row["channels"] = 1 if data.ndim == 1 else data.shape[1]
        flat = data.ravel().astype(np.float64)
        rms = float(np.sqrt(np.mean(flat * flat)))
        row["rms_db"] = 20.0 * math.log10(rms) if rms > 0 else -200.0
        row["peak"] = float(np.max(np.abs(flat)))
        row["clip_frac"] = float(np.mean(np.abs(flat) >= T_CLIP_PEAK))
        row["sha256"] = sha256_file(path)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"unreadable:{type(exc).__name__}"
    return row


def main() -> int:
    VALDIR.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MASTER)
    print(f"master rows: {len(m)}")

    # ---------- physical extra/missing files ----------
    raw_root = ROOT / "data" / "raw"
    manifest_paths = {str(p).replace("\\", "/") for p in m["local_path"]}
    disk_audio = {
        str(p.relative_to(ROOT)).replace("\\", "/")
        for p in raw_root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
    }
    extra_on_disk = sorted(disk_audio - manifest_paths)

    # ---------- per-file scan ----------
    records = []
    for i, rp in enumerate(m["local_path"], 1):
        records.append(scan_file(rp))
        if i % 1000 == 0:
            print(f"  scanned {i}/{len(m)}", flush=True)
    health = pd.DataFrame(records)
    health.insert(0, "manifest_row", range(len(m)))
    health["sha_match"] = [
        None if (a is None or not isinstance(b, str))
        else (a == b)
        for a, b in zip(health["sha256"], m["v_sha256"])
    ]

    d = health["duration"]
    rms = health["rms_db"]
    flags = {
        "missing": ~health["exists"],
        "zero_length": health["error"].isin(["zero_byte", "zero_frames"]),
        "unreadable": health["error"].fillna("").str.startswith("unreadable"),
        "extreme_short": d < T_EXTREME_SHORT,
        "short": (d >= T_EXTREME_SHORT) & (d < T_SHORT),
        "long": (d > T_LONG) & (d <= T_VERY_LONG),
        "very_long": d > T_VERY_LONG,
        "silent": rms <= T_SILENT_DB,
        "low_energy": (rms > T_SILENT_DB) & (rms <= T_LOW_ENERGY_DB),
        "clipped": health["clip_frac"] >= T_CLIP_FRAC,
        "sha_mismatch": health["sha_match"] == False,  # noqa: E712
    }
    flag_df = pd.DataFrame(flags)
    health["flags"] = [
        ",".join(flag_df.columns[rowvals]) for rowvals in flag_df.to_numpy()
    ]
    health["n_flags"] = flag_df.sum(axis=1).to_numpy()

    out = health.copy()
    out.insert(1, "local_path", m["local_path"].to_numpy())
    out.insert(2, "source_dataset", m["source_dataset"].to_numpy())
    out.insert(3, "project_class", m["project_class"].to_numpy())
    out.to_csv(VALDIR / "file_health.csv", index=False)
    print("wrote file_health.csv")

    # ---------- assembled frame for analysis ----------
    df = m.copy()
    for c in ["duration", "sample_rate", "channels", "rms_db", "peak",
              "clip_frac", "sha256", "n_flags", "flags"]:
        df[c] = health[c].to_numpy()
    for c in flag_df.columns:
        df[f"flag_{c}"] = flag_df[c].to_numpy()

    # ---------- duplicates ----------
    sha_counts = df["sha256"].value_counts()
    dup_hashes = sha_counts[sha_counts > 1]
    dup_groups = []
    for h in dup_hashes.index:
        g = df[df["sha256"] == h]
        dup_groups.append({
            "sha256": h, "n": int(len(g)),
            "sources": sorted(g["source_dataset"].unique()),
            "classes": sorted(g["project_class"].unique()),
            "paths": g["local_path"].tolist(),
        })
    cross_source_dups = [g for g in dup_groups if len(g["sources"]) > 1]
    cross_class_dups = [g for g in dup_groups if len(g["classes"]) > 1]

    # ---------- source-level leakage ----------
    df["group_key"] = df["source_dataset"].astype(str) + "::" + df["source_id"].astype(str)
    grp = df.groupby("group_key")
    source_groups = grp.agg(
        n_clips=("local_path", "size"),
        n_classes=("project_class", "nunique"),
        classes=("project_class", lambda s: sorted(set(s))),
        datasets=("source_dataset", lambda s: sorted(set(s))),
    )
    multi_clip_groups = source_groups[source_groups["n_clips"] > 1]
    cross_class_groups = source_groups[source_groups["n_classes"] > 1]

    # deeper: babycry DonateACry subject/session uuid grouping
    babycry = df[df["source_dataset"] == "owlgebra_babycry"].copy()
    babycry["subject"] = np.where(
        babycry["source_id"].str.startswith("donateacry:"),
        babycry["source_id"].str.extract(
            r"donateacry:([0-9a-fA-F-]+?)-\d{13}-", expand=False),
        babycry["source_id"],
    )
    bb_subjects = babycry.groupby("subject").size() if len(babycry) else pd.Series(dtype=int)

    # ---------- label sanity ----------
    label_tables = {}
    for sd, g in df.groupby("source_dataset"):
        ct = (g.groupby(["original_label", "project_class"]).size()
              .reset_index(name="n").sort_values("n", ascending=False))
        label_tables[sd] = ct.to_dict(orient="records")

    # FSD50K ambiguity: co-occurring unrelated major labels
    CFG_AMBIG = {
        "Help_Shouting": r"music|guitar|applause|cheering|crowd|chant|piano|singing",
        "Alarm": r"bicycle_bell|clock|fireworks|music|door",
        "Knocking": r"music|alarm|engine",
        "Footsteps": r"music|applause|dance",
        "Train": r"music|alarm",
        "Aircraft": r"music",
        "Motorcycle": r"music|engine",
        "Vehicle_Horn": r"music|alarm|engine",
        "Gunshot": r"music|explosion|fireworks",
        "Glass_Breaking": r"music|applause",
        "Doorbell": r"music|alarm|bell,",
    }
    fsd = df[df["source_dataset"] == "FSD50K"].copy()
    ambig = {}
    for cls, pat in CFG_AMBIG.items():
        sub = fsd[fsd["project_class"] == cls]
        if len(sub) == 0:
            continue
        mask = sub["original_label"].str.lower().str.contains(pat, regex=True, na=False)
        ambig[cls] = {
            "n": int(len(sub)), "ambiguous_n": int(mask.sum()),
            "examples": sub.loc[mask, "original_label"].head(3).tolist(),
        }

    # ---------- duration / sr / channel stats ----------
    def dur_stats(g):
        dd = g["duration"]
        return {
            "min": round(float(dd.min()), 3), "median": round(float(dd.median()), 3),
            "mean": round(float(dd.mean()), 3), "max": round(float(dd.max()), 3),
            "std": round(float(dd.std()), 3),
            "total_hours": round(float(dd.sum()) / 3600.0, 2),
        }

    per_class_dur = {c: dur_stats(g) for c, g in df.groupby("project_class")}
    overall_dur = dur_stats(df)

    sr_dist = df["v_sample_rate"].value_counts().sort_index()
    ch_dist = df["v_channels"].value_counts().sort_index()
    sr_by_class = {
        c: {int(k): int(v) for k, v in g["v_sample_rate"].value_counts().items()}
        for c, g in df.groupby("project_class")
    }
    ch_by_class = {
        c: {int(k): int(v) for k, v in g["v_channels"].value_counts().items()}
        for c, g in df.groupby("project_class")
    }

    # ---------- split feasibility ----------
    split = []
    for c, g in df.groupby("project_class"):
        gg = g.groupby("group_key").size()
        n_groups = len(gg)
        max_g = int(gg.max()) if n_groups else 0
        share = max_g / len(g) if len(g) else 0
        if n_groups >= 20 and share <= 0.20:
            verdict = "FEASIBLE"
        elif n_groups >= 7:
            verdict = "MARGINAL"
        else:
            verdict = "INFEASIBLE"
        split.append({
            "project_class": c, "n_clips": len(g), "n_source_groups": n_groups,
            "multi_clip_groups": int((gg > 1).sum()),
            "max_group_size": max_g, "max_group_share": round(share, 3),
            "groups_needed_min": 3, "verdict": verdict,
        })
    split_df = pd.DataFrame(split).sort_values("n_source_groups")

    # ---------- suspicious rows table ----------
    susp_cols = ["local_path", "source_dataset", "project_class", "duration",
                 "rms_db", "peak", "clip_frac", "n_flags", "flags"]
    suspicious = df[df["n_flags"] > 0][susp_cols].sort_values("n_flags", ascending=False)

    # ---------- per-class summary csv ----------
    summary_rows = []
    for c, g in df.groupby("project_class"):
        gg = g.groupby("group_key").size()
        srs = ",".join(f"{int(k)}Hz:{v}" for k, v in sorted(g["v_sample_rate"].value_counts().items()))
        chs = ",".join(f"{int(k)}ch:{v}" for k, v in sorted(g["v_channels"].value_counts().items()))
        summary_rows.append({
            "project_class": c,
            "total_clips": len(g),
            "source_dataset_count": g["source_dataset"].nunique(),
            "sources": "|".join(sorted(g["source_dataset"].unique())),
            "source_group_count": len(gg),
            "multi_clip_groups": int((gg > 1).sum()),
            "max_group_size": int(gg.max()),
            "min_duration": round(float(g["duration"].min()), 3),
            "median_duration": round(float(g["duration"].median()), 3),
            "mean_duration": round(float(g["duration"].mean()), 3),
            "max_duration": round(float(g["duration"].max()), 3),
            "total_hours": round(float(g["duration"].sum()) / 3600.0, 3),
            "sample_rate_summary": srs,
            "channel_summary": chs,
            "suspicious_count": int((g["n_flags"] > 0).sum()),
            "silent_count": int(g["flag_silent"].sum()),
            "low_energy_count": int(g["flag_low_energy"].sum()),
            "short_count_lt1s": int((g["duration"] < T_SHORT).sum()),
            "long_count_gt15s": int((g["duration"] > T_LONG).sum()),
            "clipped_count": int(g["flag_clipped"].sum()),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("total_clips", ascending=False)
    summary_df.to_csv(OUTDIR / "dataset_validation_summary.csv", index=False)

    # ---------- assemble json ----------
    inv = {
        "manifest_rows": int(len(m)),
        "unique_files": int(m["local_path"].nunique()),
        "files_present": int(health["exists"].sum()),
        "files_missing": int((~health["exists"]).sum()),
        "readable": int(health["readable"].sum()),
        "unreadable": int(flags["unreadable"].sum()),
        "zero_length": int(flags["zero_length"].sum()),
        "extra_audio_on_disk_not_in_manifest": extra_on_disk,
        "sha_mismatch_vs_manifest": int(flags["sha_mismatch"].sum()),
        "manifest_sha_duplicates": int(len(dup_hashes)),
    }
    data = {
        "thresholds": {
            "extreme_short_s": T_EXTREME_SHORT, "short_s": T_SHORT,
            "long_s": T_LONG, "very_long_s": T_VERY_LONG,
            "silent_rms_dbfs": T_SILENT_DB, "low_energy_rms_dbfs": T_LOW_ENERGY_DB,
            "clip_peak": T_CLIP_PEAK, "clip_fraction": T_CLIP_FRAC,
        },
        "inventory": inv,
        "counts": {
            "per_class": df["project_class"].value_counts().to_dict(),
            "per_source": df["source_dataset"].value_counts().to_dict(),
            "per_source_class": {
                f"{sd}::{c}": int(n)
                for (sd, c), n in df.groupby(["source_dataset", "project_class"]).size().items()
            },
        },
        "flag_counts": {k: int(v.sum()) for k, v in flags.items()},
        "duplicates": {
            "n_groups": int(len(dup_groups)), "groups": dup_groups,
            "cross_source": cross_source_dups, "cross_class": cross_class_dups,
        },
        "leakage": {
            "n_source_groups": int(len(source_groups)),
            "n_multi_clip_groups": int(len(multi_clip_groups)),
            "largest_groups": [
                {"group": k, "n_clips": int(r.n_clips), "classes": r.classes,
                 "datasets": r.datasets}
                for k, r in multi_clip_groups.sort_values(
                    "n_clips", ascending=False).head(25).iterrows()
            ],
            "cross_class_groups": [
                {"group": k, "n_clips": int(r.n_clips), "classes": r.classes}
                for k, r in cross_class_groups.iterrows()
            ],
            "babycry_subject_groups": int(len(bb_subjects)),
            "babycry_multi_clip_subjects": int((bb_subjects > 1).sum()),
            "babycry_max_subject_clips": int(bb_subjects.max()) if len(bb_subjects) else 0,
        },
        "duration": {"overall": overall_dur, "per_class": per_class_dur},
        "sample_rates": {str(int(k)): int(v) for k, v in sr_dist.items()},
        "channels": {str(int(k)): int(v) for k, v in ch_dist.items()},
        "sample_rate_by_class": sr_by_class,
        "channels_by_class": ch_by_class,
        "label_tables": label_tables,
        "fsd50k_ambiguity": ambig,
        "split_feasibility": split_df.to_dict(orient="records"),
        "suspicious_rows": suspicious.head(100).round(4).to_dict(orient="records"),
        "n_suspicious_total": int((df["n_flags"] > 0).sum()),
    }
    (VALDIR / "validation_data.json").write_text(
        json.dumps(data, indent=1, default=str), encoding="utf-8")
    print("wrote validation_data.json + dataset_validation_summary.csv")

    print("\n=== FLAG COUNTS ===")
    for k, v in flags.items():
        print(f"  {k:16s} {int(v.sum())}")
    print("\n=== SPLIT FEASIBILITY ===")
    print(split_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
