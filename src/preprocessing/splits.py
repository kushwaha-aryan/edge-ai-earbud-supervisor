"""Stage 2 of the Phase 2A preprocessing pipeline: deterministic seeded
source-group split.

Implements D9 (approved 2026-10-08): every kept clip belongs to exactly one
source group (ledger ``group_key``), and groups are assigned atomically to
train/val/test by a seeded greedy procedure that, for each group, picks the
split with the smallest summed relative per-class deficit against the
70/15/15 targets. Groups may span several classes; all of them move
together, which keeps recordings (subjects, sessions, fsIDs) from leaking
across splits.

Writes:
  data/splits/clip_splits.csv    clip_id, group_key, project_class, class_id, split
  data/splits/group_splits.csv   group_key, split, n_clips, classes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config

SPLITS = ("train", "val", "test")


def _ratios() -> np.ndarray:
    ratios = np.array(
        [config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO], dtype=np.float64
    )
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"split ratios must sum to 1, got {ratios.sum()}")
    return ratios


def build_splits(
    ledger: pd.DataFrame,
    seed: int | None = None,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign every kept clip to train/val/test and persist both CSV files."""
    seed = config.SPLIT_SEED if seed is None else seed
    out_dir = Path(out_dir or config.SPLIT_DATA_DIR)

    kept = ledger.loc[
        ledger["status"] == "kept",
        ["clip_id", "group_key", "project_class", "class_id"],
    ].copy()
    if kept.empty:
        raise ValueError("ledger contains no kept clips")

    ratios = _ratios()
    class_ids = sorted(int(c) for c in kept["class_id"].unique())
    class_pos = {cid: i for i, cid in enumerate(class_ids)}
    targets = np.zeros((len(class_ids), len(SPLITS)), dtype=np.float64)
    for cid, n in kept["class_id"].value_counts().items():
        targets[class_pos[int(cid)], :] = float(n) * ratios

    group_records: list[tuple[str, pd.Series, int, list[str]]] = []
    for gkey, gdf in kept.groupby("group_key", sort=True):
        group_records.append(
            (
                str(gkey),
                gdf["class_id"].value_counts(),
                int(len(gdf)),
                sorted(gdf["project_class"].unique()),
            )
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(group_records))

    current = np.zeros_like(targets)
    assignment: dict[str, int] = {}
    for idx in order:
        gkey, class_counts, _n, _classes = group_records[int(idx)]
        deficit = np.zeros(len(SPLITS), dtype=np.float64)
        for cid, n in class_counts.items():
            row = targets[class_pos[int(cid)]]
            deficit += np.maximum(0.0, row - current[class_pos[int(cid)]]) / np.maximum(
                row, 1.0
            )
        split_idx = int(np.argmax(deficit))
        assignment[gkey] = split_idx
        for cid, n in class_counts.items():
            current[class_pos[int(cid)], split_idx] += float(n)

    kept["split"] = kept["group_key"].map(lambda g: SPLITS[assignment[str(g)]])
    if kept["split"].isna().any():
        raise RuntimeError("some kept clips received no split assignment")

    clip_splits = (
        kept.sort_values("clip_id").reset_index(drop=True)[
            ["clip_id", "group_key", "project_class", "class_id", "split"]
        ]
    )
    group_splits = pd.DataFrame(
        [
            {
                "group_key": gkey,
                "split": SPLITS[assignment[gkey]],
                "n_clips": n,
                "classes": ";".join(classes),
            }
            for gkey, _vc, n, classes in group_records
        ]
    ).sort_values("group_key").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    clip_splits.to_csv(out_dir / "clip_splits.csv", index=False)
    group_splits.to_csv(out_dir / "group_splits.csv", index=False)
    return clip_splits, group_splits
