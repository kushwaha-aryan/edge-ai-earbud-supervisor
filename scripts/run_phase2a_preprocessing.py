#!/usr/bin/env python
"""Run the Phase 2A preprocessing pipeline.

Stages (FINAL_PROJECT_BLUEPRINT.md Change Log 2026-10-08, decisions D1-D12
plus OA1-OA4; D9 amended after the Phase 2B gate FAIL):

  ledger     clip exclusions/truncation decisions + provenance ledger
             -> data/provenance/phase2a_clip_ledger.csv
  split      seeded 70/15/15 source-group split (D9): count greedy, plus the
             covariate hill-climb when data/provenance/clip_covariates.csv
             exists (D9 amendment: balances per-class duration / RMS / clip
             feature-mean across splits)
             -> data/splits/clip_splits.csv, data/splits/group_splits.csv
  covariates per-clip mean feature value from the existing feature arrays
             (input to the balanced split)
             -> data/provenance/clip_covariates.csv
  features   cleaned audio -> windowed mel-spectrograms -> [0,1] using
             train-frozen dB bounds
             -> data/processed/features_<split>.npy, labels_<split>.npy,
                window_index_<split>.csv, feature_stats.json
  gates      D11 leakage gates G1-G5 (exit code 1 on failure)
             -> scripts/results/phase2a_gates.json

The balanced split needs covariates, which need features, which need a
split; on a cold start run the pipeline twice:

  python scripts/run_phase2a_preprocessing.py                    # v1 split
  python scripts/run_phase2a_preprocessing.py --stage covariates  # derive
  python scripts/run_phase2a_preprocessing.py                    # v2 + gates

Re-running with an unchanged covariates file is idempotent (same seed, same
inputs -> same split). If you refresh covariates, re-run split + features +
gates together afterwards.

Usage:
  python scripts/run_phase2a_preprocessing.py
  python scripts/run_phase2a_preprocessing.py --stage gates
  python scripts/run_phase2a_preprocessing.py --stage features --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src import config
from src.preprocessing import clips, features, gates, splits

STAGE_CHOICES = ("all", "ledger", "split", "covariates", "features", "gates")


def _print_split_summary(clip_splits: pd.DataFrame) -> None:
    counts = clip_splits.groupby("split").size()
    total = int(counts.sum())
    for name in ("train", "val", "test"):
        n = int(counts.get(name, 0))
        print(f"      {name:<6} {n:>6} clips ({n / total:6.2%})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=STAGE_CHOICES, default="all")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke test: only the first N clips per split in the feature stage",
    )
    args = parser.parse_args()
    t0 = time.time()

    print("[1/5] ledger: exclusions, truncation, provenance", flush=True)
    ledger = clips.build_ledger()
    n_kept = int((ledger["status"] == "kept").sum())
    n_excluded = len(ledger) - n_kept
    reasons = ledger.loc[ledger["status"] == "excluded", "reason"].value_counts()
    print(f"      rows={len(ledger)} kept={n_kept} excluded={n_excluded}", flush=True)
    for r, c in reasons.items():
        print(f"        {r}: {c}", flush=True)

    print("[2/5] split: seeded source-group 70/15/15 (D9)", flush=True)
    clip_splits, group_splits = splits.build_splits(ledger)
    _print_split_summary(clip_splits)
    print(f"      groups={len(group_splits)} (seed={config.SPLIT_SEED})", flush=True)

    if args.stage in ("all", "features"):
        print("[3/5] features: clean -> window -> mel-spectrogram -> [0,1]", flush=True)
        features.extract_features(ledger, clip_splits, limit=args.limit)
    else:
        print("[3/5] features: skipped (stage=%s)" % args.stage, flush=True)

    if args.stage == "covariates":
        print("[4/5] covariates: per-clip mean feature value", flush=True)
        features.compute_clip_covariates()
    else:
        print("[4/5] covariates: skipped (stage=%s)" % args.stage, flush=True)

    exit_code = 0
    if args.stage in ("all", "gates"):
        print("[5/5] gates: D11 leakage checks", flush=True)
        problems, details = gates.run_gates(ledger, clip_splits, group_splits)
        out = ROOT / "scripts" / "results" / "phase2a_gates.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        details["generated_by"] = "run_phase2a_preprocessing.py"
        details["limit"] = args.limit
        out.write_text(json.dumps(details, indent=2), encoding="utf-8")
        if problems:
            for p in problems:
                print(f"      FAIL {p}", flush=True)
            print(f"GATES: FAIL ({len(problems)} problems)", flush=True)
            exit_code = 1
        else:
            dev = details.get("max_abs_deviation", "?")
            print(
                f"GATES: PASS (max proportion deviation {dev}, "
                f"shared hashes {details.get('shared_hashes_across_splits')})",
                flush=True,
            )
            print(f"      details -> {out}", flush=True)
    else:
        print("[5/5] gates: skipped (stage=%s)" % args.stage, flush=True)

    print(f"done in {time.time() - t0:.1f}s", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
