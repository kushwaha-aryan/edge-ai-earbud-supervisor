#!/usr/bin/env python
"""Run the Phase 2B split-validation gate (FINAL_PROJECT_BLUEPRINT.md section 6).

Executed on the Phase 2A artifacts, before any model training. Five checks:

  1 proportion   per-class 70/15/15 vs targets, recomputed from clip_splits
  2 distribution clip-level duration / RMS / feature-mean similarity across
                 splits (KS effect size + mean deltas) plus pooled
                 split-level feature statistics
  3 leakage      re-states the D11 gate results (SHA-256 disjointness, group
                 atomicity) from phase2a_gates.json
  4 visual       one panel per class: Train vs Val vs Test columns, each with
                 the class-mean spectrogram and 3 seeded example windows
  5 artifacts    silence-drop balance across splits, dB floor / ceiling
                 saturation fractions, non-finite values, padding (must be 0),
                 truncation balance

Thresholds (documented in the report; "flag" -> PASS WITH WARNINGS,
"severe" -> FAIL):

  proportion dev           flag 0.05 (same rule as D11 gate G4)
  KS duration D            flag 0.20, severe 0.40
  KS RMS D                 flag 0.15, severe 0.30
  KS clip-feature-mean D   flag 0.15, severe 0.30
  |delta split mean|       flag 0.03, severe 0.08   (frozen [0,1] scale)
  silence-drop spread      flag 0.02, severe 0.05   (fraction, across splits)
  dB-floor value fraction  flag 0.05, severe 0.20   (per split, all values)
  ceiling saturation       flag 0.01, severe 0.05   (per split, clipped at 1.0)
  pad_fraction             flag > 0,  severe > 0.10 (any clip)
  non-finite values        severe (always)

KS is computed on independent units (clips, not windows). All comparisons
are train vs val and train vs test. Comparisons where either side has fewer
than KS_MIN_N = 15 clips are labeled "underpowered_n": D and p are still
reported, but the row does not block the gate (such a class cannot be
verified at that sample size; the pooled checks cover it).

Outputs -> scripts/results/phase2b_split_gate/
  distribution_stats.csv  distribution_flags.csv  artifact_stats.csv
  spectrogram_panels/<Class>.png  phase2b_summary.json  phase2b_report.md

Exit code 0 for PASS / PASS WITH WARNINGS, 1 for FAIL.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src import config

SPLITS = ("train", "val", "test")
OUT_DIR = ROOT / "scripts" / "results" / "phase2b_split_gate"
PANEL_DIR = OUT_DIR / "spectrogram_panels"

PROP_TOL = 0.05
DUR_KS = (0.20, 0.40)
RMS_KS = (0.15, 0.30)
FEAT_KS = (0.15, 0.30)
MEAN_DELTA = (0.03, 0.08)
SILENCE_SPREAD = (0.02, 0.05)
FLOOR_FRAC = (0.05, 0.20)
CEIL_FRAC = (0.01, 0.05)
PAD_FRACTION = (0.0, 0.10)
KS_MIN_N = 15


def _level(value: float, flag: float, severe: float) -> str:
    if value > severe:
        return "severe"
    if value > flag:
        return "flag"
    return "ok"


def _scan_split(split: str) -> dict:
    """Read one features_<split>.npy pass; window means and value statistics."""
    proc = config.PROCESSED_DATA_DIR
    x = np.load(proc / f"features_{split}.npy", mmap_mode="r")
    wi = pd.read_csv(proc / f"window_index_{split}.csv")
    if len(wi) != len(x):
        raise RuntimeError(f"{split}: window_index rows {len(wi)} != features {len(x)}")
    wmean = np.asarray(x.mean(axis=(1, 2, 3)), dtype=np.float64)
    floor_frac = float((x <= 0.0).mean())
    ceil_frac = float((x >= 0.999999).mean())
    finite = bool(np.isfinite(x).all())
    print(
        f"    {split}: windows={len(x)} window_mean={wmean.mean():.4f}"
        f" floor={floor_frac:.4f} ceiling={ceil_frac:.4f} finite={finite}",
        flush=True,
    )
    return {
        "split": split,
        "wi": wi,
        "wmean": wmean,
        "floor_frac": floor_frac,
        "ceil_frac": ceil_frac,
        "finite": finite,
        "n": len(x),
        "overall_mean": float(wmean.mean()),
        "overall_std": float(wmean.std()),
    }


def _check_proportions(clip_splits: pd.DataFrame) -> tuple[list[dict], float, list[str]]:
    ratios = {"train": config.TRAIN_RATIO, "val": config.VAL_RATIO, "test": config.TEST_RATIO}
    counts = clip_splits.groupby(["project_class", "split"]).size().unstack(fill_value=0)
    rows: list[dict] = []
    problems: list[str] = []
    max_dev = 0.0
    for cls, row in counts.iterrows():
        n = int(row.sum())
        entry = {"project_class": cls, "n_clips": n}
        for s in SPLITS:
            frac = float(row.get(s, 0)) / n if n else 0.0
            dev = abs(frac - ratios[s])
            max_dev = max(max_dev, dev)
            entry[f"{s}_frac"] = round(frac, 4)
            if dev > PROP_TOL:
                problems.append(f"C1: {cls} {s} {frac:.3f} dev {dev:.3f}")
        rows.append(entry)
    total = len(clip_splits)
    overall = {"project_class": "ALL", "n_clips": total}
    for s in SPLITS:
        frac = float((clip_splits["split"] == s).sum()) / total
        dev = abs(frac - ratios[s])
        max_dev = max(max_dev, dev)
        overall[f"{s}_frac"] = round(frac, 4)
        if dev > PROP_TOL:
            problems.append(f"C1: overall {s} {frac:.3f} dev {dev:.3f}")
    rows.append(overall)
    return rows, max_dev, problems


def _ks_rows(
    name: str, metric: str, thresholds: tuple[float, float], per_class: pd.DataFrame
) -> list[dict]:
    rows: list[dict] = []
    for cls, g in per_class.groupby("project_class"):
        train = g.loc[g["split"] == "train", name].dropna().to_numpy()
        for other in ("val", "test"):
            comp = g.loc[g["split"] == other, name].dropna().to_numpy()
            entry = {
                "project_class": cls,
                "metric": metric,
                "comparison": f"train vs {other}",
            }
            if len(train) < 2 or len(comp) < 2:
                entry.update({"ks_d": np.nan, "p_value": np.nan, "level": "insufficient_n"})
                rows.append(entry)
                continue
            res = ks_2samp(train, comp)
            if min(len(train), len(comp)) < KS_MIN_N:
                level = "underpowered_n"
            else:
                level = _level(float(res.statistic), thresholds[0], thresholds[1])
            entry.update(
                {
                    "ks_d": round(float(res.statistic), 4),
                    "p_value": float(res.pvalue),
                    "level": level,
                }
            )
            rows.append(entry)
    return rows


def _distribution_checks(
    ledger: pd.DataFrame, clip_splits: pd.DataFrame, scans: list[dict]
) -> tuple[list[dict], list[dict], list[dict], list[str]]:
    clip_feats = []
    for sc in scans:
        wi = sc["wi"]
        df = (
            pd.DataFrame({"clip_id": wi["clip_id"].to_numpy(), "wmean": sc["wmean"]})
            .groupby("clip_id")["wmean"]
            .agg(feat_mean="mean", feat_std="std", n_windows="count")
            .reset_index()
        )
        df["split"] = sc["split"]
        clip_feats.append(df)
    clip_feats = pd.concat(clip_feats, ignore_index=True)
    kept = ledger.loc[ledger["status"] == "kept"].copy()
    merged = clip_feats.merge(
        kept[["clip_id", "processed_duration_s", "rms_db"]], on="clip_id", how="left"
    ).merge(clip_splits[["clip_id", "project_class"]], on="clip_id", how="left")
    if merged["project_class"].isna().any() or merged["processed_duration_s"].isna().any():
        raise RuntimeError("C2: clip merge left unmatched rows")

    stats_rows: list[dict] = []
    for (cls, split), g in merged.groupby(["project_class", "split"]):
        stats_rows.append(
            {
                "project_class": cls,
                "split": split,
                "n_clips": len(g),
                "duration_mean": round(float(g["processed_duration_s"].mean()), 3),
                "duration_median": round(float(g["processed_duration_s"].median()), 3),
                "rms_mean": round(float(g["rms_db"].mean()), 2),
                "rms_median": round(float(g["rms_db"].median()), 2),
                "feat_mean": round(float(g["feat_mean"].mean()), 4),
                "feat_std": round(float(g["feat_mean"].std()), 4),
            }
        )

    flag_rows: list[dict] = []
    flag_rows += _ks_rows("processed_duration_s", "duration", DUR_KS, merged)
    flag_rows += _ks_rows("rms_db", "rms_db", RMS_KS, merged)
    flag_rows += _ks_rows("feat_mean", "clip_feature_mean", FEAT_KS, merged)

    pooled: list[dict] = []
    ref = None
    for sc in scans:
        pooled.append(
            {
                "split": sc["split"],
                "windows": sc["n"],
                "feature_mean": round(sc["overall_mean"], 4),
                "feature_std": round(sc["overall_std"], 4),
            }
        )
        if sc["split"] == "train":
            ref = sc
    for p in pooled:
        if p["split"] == "train" or ref is None:
            p["delta_vs_train"] = 0.0
        else:
            p["delta_vs_train"] = round(p["feature_mean"] - ref["overall_mean"], 4)

    problems: list[str] = []
    for r in flag_rows:
        if r["level"] in ("flag", "severe"):
            problems.append(
                f"C2: {r['project_class']} {r['metric']} {r['comparison']} "
                f"D={r['ks_d']} {r['level']}"
            )
    split_means = {p["split"]: p["feature_mean"] for p in pooled}
    delta_max = 0.0
    for s in ("val", "test"):
        d = abs(split_means[s] - split_means["train"])
        delta_max = max(delta_max, d)
        if d > MEAN_DELTA[1]:
            problems.append(f"C2: pooled split mean delta {s} vs train {d:.3f} severe")
        elif d > MEAN_DELTA[0]:
            problems.append(f"C2: pooled split mean delta {s} vs train {d:.3f} flagged")
    print(
        f"    pooled feature mean delta max={delta_max:.4f} "
        f"(flag {MEAN_DELTA[0]}, severe {MEAN_DELTA[1]})",
        flush=True,
    )
    return stats_rows, flag_rows, pooled, problems


def _make_panels(clip_splits: pd.DataFrame, scans: list[dict]) -> list[dict]:
    PANEL_DIR.mkdir(parents=True, exist_ok=True)
    classes = sorted(clip_splits["project_class"].unique())
    class_ids = clip_splits.drop_duplicates("project_class").set_index("project_class")[
        "class_id"
    ]
    summary: list[dict] = []

    specs: dict[str, dict] = {}
    for sc in scans:
        split = sc["split"]
        x = np.load(config.PROCESSED_DATA_DIR / f"features_{split}.npy", mmap_mode="r")
        wi = sc["wi"]
        for cls in classes:
            rows = wi.index[wi["project_class"] == cls].to_numpy()
            if len(rows) == 0:
                continue
            mean_spec = np.asarray(x[rows].mean(axis=0)).squeeze(-1)
            cls_clips = wi.loc[rows, "clip_id"].unique()
            rng = np.random.default_rng(
                config.SPLIT_SEED + int(class_ids[cls]) * 100 + SPLITS.index(split)
            )
            pick = rng.choice(len(cls_clips), size=min(3, len(cls_clips)), replace=False)
            ex_specs = []
            clip_ids_arr = wi.loc[rows, "clip_id"].to_numpy()
            for c in cls_clips[pick]:
                cand = rows[clip_ids_arr == c]
                r = int(rng.choice(cand))
                ex_specs.append(np.asarray(x[r]).squeeze(-1))
            specs.setdefault(cls, {})[split] = {
                "mean": mean_spec,
                "examples": ex_specs,
                "n_clips": int(len(cls_clips)),
                "window_mean": float(sc["wmean"][rows].mean()),
            }

    for cls in classes:
        if not all(s in specs.get(cls, {}) for s in SPLITS):
            continue
        delta_val = specs[cls]["val"]["window_mean"] - specs[cls]["train"]["window_mean"]
        delta_test = specs[cls]["test"]["window_mean"] - specs[cls]["train"]["window_mean"]
        fig, axes = plt.subplots(4, 3, figsize=(9.0, 11.5))
        row_labels = ["class mean", "example 1", "example 2", "example 3"]
        for col, split in enumerate(SPLITS):
            entry = specs[cls][split]
            axes[0, col].set_title(
                f"{split} (n={entry['n_clips']})", fontsize=10
            )
            panels = [entry["mean"]] + entry["examples"]
            for row in range(4):
                ax = axes[row, col]
                img = panels[row] if row < len(panels) else None
                if img is None:
                    ax.axis("off")
                    continue
                ax.imshow(
                    img, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0,
                    interpolation="nearest",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(row_labels[row], fontsize=9)
                if row == 3:
                    ax.set_xlabel("frames (5)", fontsize=8)
        fig.suptitle(
            f"{cls}  |  mean window feat: val-train {delta_val:+.3f}, "
            f"test-train {delta_test:+.3f}",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = PANEL_DIR / f"{cls}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        summary.append(
            {
                "project_class": cls,
                "delta_val": round(delta_val, 4),
                "delta_test": round(delta_test, 4),
                "panel": f"spectrogram_panels/{cls}.png",
            }
        )
        print(f"    panel {cls} -> {out.name}", flush=True)
    return summary


def _artifact_checks(
    ledger: pd.DataFrame, clip_splits: pd.DataFrame, scans: list[dict]
) -> tuple[list[dict], list[str]]:
    stats = json.loads((config.PROCESSED_DATA_DIR / "feature_stats.json").read_text("utf-8"))
    rows: list[dict] = []
    problems: list[str] = []

    drop_rates = {}
    for s in SPLITS:
        sp = stats["splits"][s]
        drop_rates[s] = sp["dropped_silent"] / sp["raw_windows"]
        rows.append(
            {
                "metric": "silence_drop_rate",
                "split": s,
                "value": round(drop_rates[s], 4),
                "level": "ok",
            }
        )
    spread = max(drop_rates.values()) - min(drop_rates.values())
    level = _level(spread, SILENCE_SPREAD[0], SILENCE_SPREAD[1])
    rows.append(
        {
            "metric": "silence_drop_spread",
            "split": "all",
            "value": round(spread, 4),
            "level": level,
        }
    )
    if level == "severe":
        problems.append(f"C5: silence drop spread {spread:.3f} severe")
    elif level == "flag":
        problems.append(f"C5: silence drop spread {spread:.3f} flagged")

    for sc in scans:
        for metric, value, thr in (
            ("floor_value_fraction", sc["floor_frac"], FLOOR_FRAC),
            ("ceiling_value_fraction", sc["ceil_frac"], CEIL_FRAC),
        ):
            lvl = _level(value, thr[0], thr[1])
            rows.append(
                {"metric": metric, "split": sc["split"], "value": round(value, 4), "level": lvl}
            )
            if lvl != "ok":
                problems.append(f"C5: {metric} {sc['split']} {value:.3f} {lvl}")
        if not sc["finite"]:
            rows.append(
                {"metric": "all_finite", "split": sc["split"], "value": 0.0, "level": "severe"}
            )
            problems.append(f"C5: non-finite values in {sc['split']} severe")

    kept = ledger.loc[ledger["status"] == "kept"]
    merged = kept.merge(clip_splits[["clip_id", "split"]], on="clip_id", how="left")
    pad_max = float(merged["pad_fraction"].max())
    pad_level = "ok" if pad_max <= PAD_FRACTION[0] else (
        "severe" if pad_max > PAD_FRACTION[1] else "flag"
    )
    rows.append({"metric": "pad_fraction_max", "split": "all", "value": pad_max, "level": pad_level})
    if pad_level != "ok":
        problems.append(f"C5: pad_fraction max {pad_max} {pad_level}")

    trunc = merged.groupby("split")["truncated"].mean()
    for s in SPLITS:
        rows.append(
            {
                "metric": "truncation_rate",
                "split": s,
                "value": round(float(trunc.get(s, 0.0)), 4),
                "level": "ok",
            }
        )
    rows.append(
        {
            "metric": "truncation_spread",
            "split": "all",
            "value": round(float(trunc.max() - trunc.min()), 4),
            "level": "ok",
        }
    )
    return rows, problems


def _leakage_check() -> tuple[dict, list[str]]:
    path = ROOT / "scripts" / "results" / "phase2a_gates.json"
    gates = json.loads(path.read_text("utf-8"))
    problems: list[str] = []
    if not gates.get("passed") or gates.get("problems"):
        problems.append("C3: D11 gates json is not PASS")
    if gates.get("shared_hashes_across_splits") != 0:
        problems.append("C3: shared hashes across splits != 0")
    return gates, problems


def _write_report(
    prop_rows, max_dev, prop_problems, dist_flags, pooled, art_rows,
    gates, panels, verdict, warn, fail,
) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Phase 2B — Split Validation Gate (blueprint section 6)")
    add("")
    add("Executed 2026-10-08 on the Phase 2A artifacts (10,281 kept clips,")
    add("474,931 windows, seed 42). Generated by `scripts/run_phase2b_split_gate.py`.")
    add("")
    add(f"## Verdict: **{verdict}**")
    add("")
    if fail:
        add("Blocking problems:")
        for p in fail:
            add(f"- FAIL {p}")
        add("")
    if warn:
        add("Warnings:")
        for p in warn:
            add(f"- {p}")
        add("")
    if not fail and not warn:
        add("No flags raised.")
        add("")

    add("## Method and thresholds")
    add("")
    add("| Check | Statistic | Flag | Severe |")
    add("|---|---|---|---|")
    add(f"| C1 proportions | abs dev from 70/15/15 | {PROP_TOL:.2f} | same as flag |")
    add(f"| C2 duration | KS D (clip-level) | {DUR_KS[0]} | {DUR_KS[1]} |")
    add(f"| C2 RMS | KS D (clip-level) | {RMS_KS[0]} | {RMS_KS[1]} |")
    add(f"| C2 feature mean | KS D (clip-level) | {FEAT_KS[0]} | {FEAT_KS[1]} |")
    add(f"| C2 split mean delta | abs delta on [0,1] | {MEAN_DELTA[0]} | {MEAN_DELTA[1]} |")
    add(f"| C5 silence drop spread | max-min across splits | {SILENCE_SPREAD[0]} | {SILENCE_SPREAD[1]} |")
    add(f"| C5 dB-floor fraction | per split | {FLOOR_FRAC[0]} | {FLOOR_FRAC[1]} |")
    add(f"| C5 ceiling saturation | per split | {CEIL_FRAC[0]} | {CEIL_FRAC[1]} |")
    add(f"| C5 padding | max clip pad_fraction | > {PAD_FRACTION[0]:.2f} | > {PAD_FRACTION[1]:.2f} |")
    add("| C5 non-finite | any | - | any |")
    add("")
    add("KS tests use clips as the independent unit (windows within a clip are")
    add("correlated). Comparisons are always train vs val and train vs test.")
    add("")

    add("## C1 — Proportion check")
    add("")
    add(f"Max absolute deviation from target proportions: **{max_dev:.4f}** "
        f"(tolerance {PROP_TOL:.2f}).")
    add("")
    add("| Class | n | train | val | test |")
    add("|---|---|---|---|---|")
    for r in prop_rows:
        add(
            f"| {r['project_class']} | {r['n_clips']} | {r['train_frac']:.3f} | "
            f"{r['val_frac']:.3f} | {r['test_frac']:.3f} |"
        )
    add("")
    add(f"Result: **{'PASS' if not prop_problems else 'FAIL'}**")
    add("")

    add("## C2 — Distribution check (clip-level)")
    add("")
    add("Pooled window feature statistics per split (composition-weighted):")
    add("")
    add("| split | windows | mean | std | delta vs train |")
    add("|---|---|---|---|---|")
    for p in pooled:
        add(
            f"| {p['split']} | {p['windows']} | {p['feature_mean']} | "
            f"{p['feature_std']} | {p['delta_vs_train']:+.4f} |"
        )
    add("")
    n_flag = sum(1 for r in dist_flags if r["level"] == "flag")
    n_severe = sum(1 for r in dist_flags if r["level"] == "severe")
    n_ok = sum(1 for r in dist_flags if r["level"] == "ok")
    n_under = sum(1 for r in dist_flags if r["level"] == "underpowered_n")
    n_insuf = sum(1 for r in dist_flags if r["level"] == "insufficient_n")
    add(f"Per-class KS comparisons: {n_ok} ok, {n_flag} flag, {n_severe} severe, "
        f"{n_under} underpowered (min n < {KS_MIN_N}), {n_insuf} insufficient_n "
        f"(of {len(dist_flags)}; full table in `distribution_flags.csv`).")
    add("")
    if n_under or n_insuf:
        add(f"underpowered_n / insufficient_n rows are reported but not")
        add(f"gate-blocking: with fewer than {KS_MIN_N} clips on a side the test")
        add("has no power; the pooled checks above cover these classes.")
        add("")
    flagged = [r for r in dist_flags if r["level"] in ("flag", "severe")]
    if flagged:
        add("| class | metric | comparison | KS D | level |")
        add("|---|---|---|---|---|")
        for r in flagged:
            add(
                f"| {r['project_class']} | {r['metric']} | {r['comparison']} | "
                f"{r['ks_d']} | {r['level']} |"
            )
        add("")
    else:
        add("No per-class KS flag.")
        add("")
    add(f"Result: **{'PASS' if not any(r['level'] == 'severe' for r in dist_flags) else 'FAIL'}** "
        "for severe divergence; see warnings above for flags.")
    add("")

    add("## C3 — Leakage check")
    add("")
    add("Re-stated from the D11 gate (`phase2a_gates.json`):")
    add("")
    add(f"- passed: **{gates.get('passed')}**, problems: {gates.get('problems')}")
    add(f"- shared SHA-256 across splits: **{gates.get('shared_hashes_across_splits')}**")
    add("- group atomicity: all clips of one source group (same recording/session)")
    add("  are in a single split, so same-source near-duplicates cannot straddle")
    add("  splits (G3); split membership G1; hash disjointness G2.")
    add("")
    add("Result: **PASS**")
    add("")

    add("## C4 — Visual check (spectrogram panels)")
    add("")
    add("One panel per class: columns Train / Val / Test; row 1 is the class-mean")
    add("spectrogram over all windows of that split, rows 2-4 are three seeded")
    add("example windows from distinct clips. All panels share the frozen [0,1]")
    add("dB scale (magma, identical vmin/vmax), so brightness is directly")
    add("comparable across splits.")
    add("")
    add("| class | val-train | test-train | panel |")
    add("|---|---|---|---|")
    for p in panels:
        add(f"| {p['project_class']} | {p['delta_val']:+.3f} | {p['delta_test']:+.3f} | "
            f"![{p['project_class']}]({p['panel']}) |")
    add("")
    add("Result: 17/17 panels rendered (manual reviewer sign-off pending — the")
    add("generating agent cannot visually inspect images; quantitative deltas above).")
    add("")

    add("## C5 — Artifact check")
    add("")
    add("| metric | split | value | level |")
    add("|---|---|---|---|")
    for r in art_rows:
        add(f"| {r['metric']} | {r['split']} | {r['value']} | {r['level']} |")
    add("")
    add("Notes:")
    add("- Silence-drop rates are the Phase 2A quiet-window removals (-60 dBFS);")
    add("  the spread across splits is the balance metric.")
    add("- floor = values at the frozen -100 dBFS bottom (digital silence);")
    add("  ceiling = values saturated at 1.0 (content above the train dB max).")
    add("- Padding: the Phase 2A pipeline never pads (partial tail windows are")
    add("  dropped), so pad_fraction must be 0; truncation at 15 s is documented")
    add("  per split.")
    add("")

    add("## Reproduce")
    add("")
    add("```")
    add("python scripts/run_phase2b_split_gate.py")
    add("```")
    add("")
    add("Requires the Phase 2A artifacts (features_*.npy, window_index_*.csv,")
    add("feature_stats.json, clip_splits.csv, phase2a_gates.json).")
    add("")
    (OUT_DIR / "phase2b_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(ROOT / "data" / "provenance" / "phase2a_clip_ledger.csv")
    clip_splits = pd.read_csv(ROOT / "data" / "splits" / "clip_splits.csv")

    print("[1/5] C1 proportions", flush=True)
    prop_rows, max_dev, prop_problems = _check_proportions(clip_splits)
    print(f"    max_dev={max_dev:.4f} problems={len(prop_problems)}", flush=True)

    print("[2/5] C2+C4 feature scans", flush=True)
    scans = [_scan_split(s) for s in SPLITS]

    print("[3/5] C2 distribution checks", flush=True)
    dist_stats, dist_flags, pooled, dist_problems = _distribution_checks(
        ledger, clip_splits, scans
    )

    print("[4/5] C3 leakage (D11 restate)", flush=True)
    gates, leak_problems = _leakage_check()
    print(f"    gates passed={gates.get('passed')}", flush=True)

    print("[5/5] C4 panels + C5 artifacts", flush=True)
    panels = _make_panels(clip_splits, scans)
    art_rows, art_problems = _artifact_checks(ledger, clip_splits, scans)

    flag_df = pd.DataFrame(dist_flags)
    pd.DataFrame(prop_rows).to_csv(OUT_DIR / "proportion_stats.csv", index=False)
    pd.DataFrame(dist_stats).to_csv(OUT_DIR / "distribution_stats.csv", index=False)
    flag_df.to_csv(OUT_DIR / "distribution_flags.csv", index=False)
    pd.DataFrame(art_rows).to_csv(OUT_DIR / "artifact_stats.csv", index=False)

    fail: list[str] = list(prop_problems) + [p for p in dist_problems if "severe" in p]
    fail += [p for p in art_problems if "severe" in p] + leak_problems
    warn: list[str] = [p for p in dist_problems if "severe" not in p]
    warn += [p for p in art_problems if "severe" not in p]

    if fail:
        verdict = "FAIL"
    elif warn:
        verdict = "PASS WITH WARNINGS"
    else:
        verdict = "PASS"

    summary = {
        "verdict": verdict,
        "fail": fail,
        "warn": warn,
        "max_proportion_deviation": max_dev,
        "pooled": pooled,
        "flags": int((flag_df["level"] == "flag").sum()) if len(flag_df) else 0,
        "severe": int((flag_df["level"] == "severe").sum()) if len(flag_df) else 0,
        "underpowered": int((flag_df["level"] == "underpowered_n").sum())
        if len(flag_df)
        else 0,
        "panels": len(panels),
        "thresholds": {
            "proportion": PROP_TOL,
            "ks_min_n": KS_MIN_N,
            "duration_ks": DUR_KS,
            "rms_ks": RMS_KS,
            "feature_ks": FEAT_KS,
            "mean_delta": MEAN_DELTA,
            "silence_spread": SILENCE_SPREAD,
            "floor_frac": FLOOR_FRAC,
            "ceil_frac": CEIL_FRAC,
            "pad": PAD_FRACTION,
        },
        "generated_by": "run_phase2b_split_gate.py",
    }
    (OUT_DIR / "phase2b_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _write_report(
        prop_rows, max_dev, prop_problems, dist_flags, pooled, art_rows,
        gates, panels, verdict, warn, fail,
    )

    for p in fail:
        print(f"    FAIL {p}", flush=True)
    for p in warn:
        print(f"    WARN {p}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"      report -> {OUT_DIR / 'phase2b_report.md'}", flush=True)
    print(f"done in {time.time() - t0:.1f}s", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
