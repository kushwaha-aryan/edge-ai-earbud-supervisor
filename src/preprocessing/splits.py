"""Stage 2 of the Phase 2A preprocessing pipeline: deterministic seeded
source-group split (D9, amended 2026-10-08 after the Phase 2B gate FAIL).

Every kept clip belongs to exactly one source group (ledger ``group_key``),
and groups are assigned atomically to train/val/test. Assignment has three
phases, all deterministic for a given seed (config.SPLIT_SEED = 42):

1. count greedy (original D9): groups in seeded random order go to the split
   with the smallest summed relative per-class deficit against the 70/15/15
   targets. This centres the class counts on their targets.
2. covariate hill-climb (D9 amendment, Phase 2B resolution): seeded single-
   group moves that reduce a per-class covariate imbalance objective over
   duration / RMS dB / clip feature mean. Each (class, split) cell is scored
   on its count deviation, covariate means (class-sd units) and spreads, plus
   a binned-quantile shape term: each class's clips are rank-binned into
   ``_HIST_BINS`` equal-count quantile bins per covariate and each split's
   bin histogram is matched to the class histogram (a binned ECDF, so tail
   and mid-distribution mismatches are visible to the objective — KS-style
   shape mismatches such as the v2 gate's Vehicle_Horn RMS residual). Moves
   are subject to a per-class count band of 3% of the class size (gate G4's
   tolerance is 5%).
3. exact-KS repair: seeded single-group moves with a lexicographic accept
   rule over (number of cells whose exact two-sample KS statistic crosses
   the gate C2 flag threshold, then the phase-2 objective plus squared KS
   statistics scaled by those thresholds). Reducing the flag count always
   wins, so the repair can never trade one threshold crossing for gains
   elsewhere; cells with fewer than ``_KS_MIN_N`` clips per side are left
   alone rather than over-fitting sampling noise.

Phases 2-3 are skipped with a notice when
data/provenance/clip_covariates.csv is absent (cold start: run
``--stage covariates`` after the first feature extraction, then re-run
``--stage all``).

Covariates per clip: ledger ``processed_duration_s``, ledger ``rms_db``,
and ``feat_mean`` from the covariates CSV (mean scaled feature value over
the clip's surviving windows; affine-equivalent to the raw mel-dB clip
mean under the frozen dB bounds).

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

COVS = ("processed_duration_s", "rms_db", "feat_mean")

_W_CNT = 1.0
_W_MEAN = 1.0
_W_SD = 0.5
_W_HIST = 1.0
_W_KS = 1.0
_KS_FLAG = (0.20, 0.15, 0.15)  # gate C2 flag thresholds, COVS order
_KS_BARRIER = 3.0  # extra weight once a cell's D crosses its flag threshold
_KS_MIN_N = 15  # gate C2 underpowered rule
_BAND_FRAC = 0.03
_MAX_PASSES = 20
_HIST_BINS = 10
_SEED_OFFSET = 104729
_EPS_SD = 1e-6


def _ks_d(a: np.ndarray, b: np.ndarray) -> float:
    """Exact two-sample KS statistic (matches scipy.stats.ks_2samp)."""
    sa = np.sort(a)
    sb = np.sort(b)
    u = np.concatenate((sa, sb))
    fa = np.searchsorted(sa, u, side="right") / sa.size
    fb = np.searchsorted(sb, u, side="right") / sb.size
    return float(np.max(np.abs(fa - fb)))


def _ratios() -> np.ndarray:
    ratios = np.array(
        [config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO], dtype=np.float64
    )
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(f"split ratios must sum to 1, got {ratios.sum()}")
    return ratios


def _merge_covariates(kept: pd.DataFrame) -> pd.DataFrame | None:
    path = config.PROVENANCE_DIR / "clip_covariates.csv"
    if not path.exists():
        return None
    cov = pd.read_csv(path)
    if cov["clip_id"].duplicated().any() or cov["feat_mean"].isna().any():
        raise ValueError(f"invalid covariates file: {path}")
    merged = kept.merge(
        cov[["clip_id", "feat_mean"]], on="clip_id", how="left", validate="one_to_one"
    )
    missing = int(merged["feat_mean"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} kept clips missing from {path}; "
            "re-run --stage covariates against the current feature files"
        )
    return merged


def _class_stats(
    kept: pd.DataFrame, class_pos: dict[int, int], n_cls: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per class: mean, std and activity flag for each covariate (clip level)."""
    n_cov = len(COVS)
    mu = np.zeros((n_cov, n_cls), dtype=np.float64)
    sd = np.ones((n_cov, n_cls), dtype=np.float64)
    active = np.zeros((n_cov, n_cls), dtype=np.float64)
    for cid, c in class_pos.items():
        rows = kept.loc[kept["class_id"] == cid, list(COVS)].to_numpy(np.float64)
        if len(rows) == 0:
            raise ValueError(f"class {cid} has no kept clips")
        mu[:, c] = rows.mean(axis=0)
        s = rows.std(axis=0)
        for xi in range(n_cov):
            if s[xi] > _EPS_SD:
                sd[xi, c] = s[xi]
                active[xi, c] = 1.0
    return mu, sd, active


def _hill_climb(
    group_records: list[dict],
    class_pos: dict[int, int],
    targets: np.ndarray,
    class_n: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    active: np.ndarray,
    kept: pd.DataFrame,
    assignment: dict[str, int],
    seed: int,
) -> None:
    """Move whole groups between splits to reduce covariate imbalance.

    Two hill-climb passes (see module docstring): phase 2 minimizes the
    binned-quantile objective, phase 3 minimizes that objective plus squared
    exact KS statistics for cells with power to block the gate. Both are
    deterministic for a given seed and never leave the per-class count band
    ``_BAND_FRAC * class_size`` (stricter than gate G4's 5%).
    """
    n_cls, n_splits = targets.shape
    n_cov = len(COVS)
    b_max = _HIST_BINS

    cls_arr = kept["class_id"].to_numpy()
    gkey_arr = kept["group_key"].to_numpy()
    n_clips = len(kept)
    bid = np.zeros((n_cov, n_clips), dtype=np.int64)
    bin_p = np.zeros((n_cov, n_cls, b_max), dtype=np.float64)
    for cid, c in class_pos.items():
        pos = np.flatnonzero(cls_arr == cid)
        n_c = int(pos.size)
        bc = min(b_max, n_c)
        for xi in range(n_cov):
            vals = kept[COVS[xi]].to_numpy(np.float64)[pos]
            order = np.argsort(vals, kind="stable")
            rank = np.empty(n_c, dtype=np.int64)
            rank[order] = np.arange(n_c, dtype=np.int64)
            bid[xi, pos] = (rank * bc) // n_c
            counts = np.bincount(bid[xi, pos], minlength=b_max)
            bin_p[xi, c, :bc] = counts[:bc] / n_c

    pos_map: dict[tuple[str, int], list[int]] = {}
    for i in range(n_clips):
        pos_map.setdefault((str(gkey_arr[i]), int(cls_arr[i])), []).append(i)

    Entry = tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    items: list[list[Entry]] = []
    for rec in group_records:
        per: list[Entry] = []
        for cid, pc in rec["per_class"].items():
            idxs = np.asarray(pos_map[(rec["key"], int(cid))], dtype=np.int64)
            bvec = np.stack(
                [np.bincount(bid[xi, idxs], minlength=b_max) for xi in range(n_cov)]
            ).astype(np.float64)
            per.append(
                (
                    class_pos[cid],
                    float(pc["k"]),
                    np.array([pc["sums"][x] for x in COVS], dtype=np.float64),
                    np.array([pc["sumsq"][x] for x in COVS], dtype=np.float64),
                    bvec,
                    idxs,
                )
            )
        items.append(per)

    n_state = np.zeros((n_cls, n_splits), dtype=np.float64)
    sums = np.zeros((n_cov, n_cls, n_splits), dtype=np.float64)
    sq = np.zeros((n_cov, n_cls, n_splits), dtype=np.float64)
    hist = np.zeros((n_cov, n_cls, n_splits, b_max), dtype=np.float64)
    gsplit = np.array([assignment[rec["key"]] for rec in group_records], dtype=np.int64)
    for gi, per in enumerate(items):
        s = int(gsplit[gi])
        for c, k, svec, qvec, bvec, _ix in per:
            n_state[c, s] += k
            sums[:, c, s] += svec
            sq[:, c, s] += qvec
            hist[:, c, s, :] += bvec

    row_vals = {xi: kept[COVS[xi]].to_numpy(np.float64) for xi in range(n_cov)}
    cur_ids: list[list[np.ndarray]] = [
        [np.empty(0, dtype=np.int64) for _ in range(n_splits)] for _ in range(n_cls)
    ]
    for gi, per in enumerate(items):
        s = int(gsplit[gi])
        for c, _k, _sv, _qv, _bv, ix in per:
            cur_ids[c][s] = np.concatenate((cur_ids[c][s], ix))

    bands = np.maximum(
        _BAND_FRAC * class_n, np.max(np.abs(n_state - targets), axis=1) + 1e-9
    )

    def cell(
        c: int,
        s: int,
        nn: float,
        svec: np.ndarray,
        qvec: np.ndarray,
        hvec: np.ndarray,
    ) -> float:
        t = targets[c, s]
        v = _W_CNT * ((nn - t) / max(t, 1.0)) ** 2
        if nn >= 1.0:
            m = svec / nn
            d = (m - mu[:, c]) / sd[:, c]
            v += _W_MEAN * float((active[:, c] * d * d).sum())
            diff = hvec / nn - bin_p[:, c]
            v += _W_HIST * float((diff * diff).sum())
            if nn >= 2.0:
                var = np.maximum(qvec / nn - m * m, 0.0)
                r = np.sqrt(var) / sd[:, c] - 1.0
                v += _W_SD * float((active[:, c] * r * r).sum())
        return v

    def objective() -> float:
        total = 0.0
        for c in range(n_cls):
            for s in range(n_splits):
                total += cell(
                    c, s, n_state[c, s], sums[:, c, s], sq[:, c, s], hist[:, c, s]
                )
        return total

    def ks_energy(c: int) -> float:
        e = 0.0
        for xi in range(n_cov):
            ids_tr = cur_ids[c][0]
            if ids_tr.size < _KS_MIN_N:
                continue
            tr = row_vals[xi][ids_tr]
            thr = _KS_FLAG[xi]
            for s in (1, 2):
                ids_o = cur_ids[c][s]
                if ids_o.size < _KS_MIN_N:
                    continue
                d = _ks_d(tr, row_vals[xi][ids_o])
                w = _KS_BARRIER if d > thr else 1.0
                e += w * (d / thr) ** 2
        return e

    def ks_stat(c: int) -> tuple[int, float]:
        """(cells above flag threshold, weighted KS energy) for one class."""
        over = 0
        e = 0.0
        for xi in range(n_cov):
            ids_tr = cur_ids[c][0]
            if ids_tr.size < _KS_MIN_N:
                continue
            tr = row_vals[xi][ids_tr]
            thr = _KS_FLAG[xi]
            for s in (1, 2):
                ids_o = cur_ids[c][s]
                if ids_o.size < _KS_MIN_N:
                    continue
                d = _ks_d(tr, row_vals[xi][ids_o])
                if d > thr:
                    over += 1
                w = _KS_BARRIER if d > thr else 1.0
                e += w * (d / thr) ** 2
        return over, e

    def class_stat(c: int) -> tuple[int, float]:
        """(flag-threshold crossings, full cell + weighted KS energy)."""
        over, e_ks = ks_stat(c)
        e_cell = 0.0
        for s in range(n_splits):
            e_cell += cell(
                c, s, n_state[c, s], sums[:, c, s], sq[:, c, s], hist[:, c, s]
            )
        return over, e_cell + _W_KS * e_ks

    j0 = objective()
    rng = np.random.default_rng(seed + _SEED_OFFSET)
    total_accepted = 0
    passes = 0
    for _pass in range(_MAX_PASSES):
        passes = _pass + 1
        accepted = 0
        for gi in rng.permutation(len(items)):
            a = int(gsplit[gi])
            b = int(rng.integers(n_splits - 1))
            if b >= a:
                b += 1
            per = items[gi]
            ok = True
            for c, k, _sv, _qv, _bv, _ix in per:
                if abs(n_state[c, a] - k - targets[c, a]) > bands[c]:
                    ok = False
                    break
                if abs(n_state[c, b] + k - targets[c, b]) > bands[c]:
                    ok = False
                    break
            if not ok:
                continue
            dj = 0.0
            for c, k, svec, qvec, bvec, _ix in per:
                dj += cell(
                    c,
                    a,
                    n_state[c, a] - k,
                    sums[:, c, a] - svec,
                    sq[:, c, a] - qvec,
                    hist[:, c, a] - bvec,
                )
                dj -= cell(
                    c, a, n_state[c, a], sums[:, c, a], sq[:, c, a], hist[:, c, a]
                )
                dj += cell(
                    c,
                    b,
                    n_state[c, b] + k,
                    sums[:, c, b] + svec,
                    sq[:, c, b] + qvec,
                    hist[:, c, b] + bvec,
                )
                dj -= cell(
                    c, b, n_state[c, b], sums[:, c, b], sq[:, c, b], hist[:, c, b]
                )
            if dj < -1e-12:
                for c, k, svec, qvec, bvec, _ix in per:
                    n_state[c, a] -= k
                    n_state[c, b] += k
                    sums[:, c, a] -= svec
                    sums[:, c, b] += svec
                    sq[:, c, a] -= qvec
                    sq[:, c, b] += qvec
                    hist[:, c, a, :] -= bvec
                    hist[:, c, b, :] += bvec
                gsplit[gi] = b
                accepted += 1
                total_accepted += 1
        if accepted == 0:
            break

    # phase 1 reassigns groups without touching cur_ids; rebuild so phase 3
    # hill-climbs on the actual post-phase-1 membership
    cur_ids = [
        [np.empty(0, dtype=np.int64) for _ in range(n_splits)] for _ in range(n_cls)
    ]
    for gi, per in enumerate(items):
        s = int(gsplit[gi])
        for c, _k, _sv, _qv, _bv, ix in per:
            cur_ids[c][s] = np.concatenate((cur_ids[c][s], ix))

    over0 = sum(ks_stat(c)[0] for c in range(n_cls))
    ks0 = sum(ks_energy(c) for c in range(n_cls))
    ks_accepted = 0
    ks_passes = 0
    for _pass in range(_MAX_PASSES):
        ks_passes = _pass + 1
        accepted = 0
        for gi in rng.permutation(len(items)):
            a = int(gsplit[gi])
            b = int(rng.integers(n_splits - 1))
            if b >= a:
                b += 1
            per = items[gi]
            ok = True
            for c, k, _sv, _qv, _bv, _ix in per:
                if abs(n_state[c, a] - k - targets[c, a]) > bands[c]:
                    ok = False
                    break
                if abs(n_state[c, b] + k - targets[c, b]) > bands[c]:
                    ok = False
                    break
            if not ok:
                continue
            affected = sorted({c for c, *_ in per})
            o_before = 0
            e_before = 0.0
            for c in affected:
                o_c, e_c = class_stat(c)
                o_before += o_c
                e_before += e_c
            saved: list[tuple[int, np.ndarray, np.ndarray]] = []
            for c, k, svec, qvec, bvec, ix in per:
                n_state[c, a] -= k
                n_state[c, b] += k
                sums[:, c, a] -= svec
                sums[:, c, b] += svec
                sq[:, c, a] -= qvec
                sq[:, c, b] += qvec
                hist[:, c, a, :] -= bvec
                hist[:, c, b, :] += bvec
                saved.append((c, cur_ids[c][a], cur_ids[c][b]))
                keep = ~np.isin(cur_ids[c][a], ix)
                cur_ids[c][a] = cur_ids[c][a][keep]
                cur_ids[c][b] = np.concatenate((cur_ids[c][b], ix))
            o_after = 0
            e_after = 0.0
            for c in affected:
                o_c, e_c = class_stat(c)
                o_after += o_c
                e_after += e_c
            # lexicographic: fewer flag-threshold crossings always wins;
            # on ties minimize the full cell + weighted-KS energy
            if o_after < o_before or (
                o_after == o_before and e_after < e_before - 1e-12
            ):
                gsplit[gi] = b
                accepted += 1
                ks_accepted += 1
            else:
                for c, old_a, old_b in saved:
                    cur_ids[c][a] = old_a
                    cur_ids[c][b] = old_b
                for c, k, svec, qvec, bvec, _ix in per:
                    n_state[c, a] += k
                    n_state[c, b] -= k
                    sums[:, c, a] += svec
                    sums[:, c, b] -= svec
                    sq[:, c, a] += qvec
                    sq[:, c, b] -= qvec
                    hist[:, c, a, :] += bvec
                    hist[:, c, b, :] -= bvec
        if accepted == 0:
            break

    over1 = sum(ks_stat(c)[0] for c in range(n_cls))
    ks1 = sum(ks_energy(c) for c in range(n_cls))
    j1 = objective()
    for gi, rec in enumerate(group_records):
        assignment[rec["key"]] = int(gsplit[gi])
    print(
        f"[split] covariate balance: objective {j0:.4f} -> {j1:.4f} "
        f"({total_accepted} moves in {passes} passes)",
        flush=True,
    )
    print(
        f"[split] exact-KS repair: crossings {over0} -> {over1}, "
        f"energy {ks0:.4f} -> {ks1:.4f} "
        f"({ks_accepted} moves in {ks_passes} passes)",
        flush=True,
    )


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
        ["clip_id", "group_key", "project_class", "class_id", *COVS[:2]],
    ].copy()
    if kept.empty:
        raise ValueError("ledger contains no kept clips")

    cov_kept = _merge_covariates(kept)
    balanced = cov_kept is not None
    if balanced:
        kept = cov_kept
        print(
            f"[split] covariate-balanced mode (D9 v2; {len(kept)} clips)",
            flush=True,
        )
    else:
        print(
            "[split] covariates not found -> counts-only split (D9 v1); "
            "run --stage covariates after features, then --stage all again",
            flush=True,
        )

    ratios = _ratios()
    class_ids = sorted(int(c) for c in kept["class_id"].unique())
    class_pos = {cid: i for i, cid in enumerate(class_ids)}
    n_cls = len(class_ids)
    targets = np.zeros((n_cls, len(SPLITS)), dtype=np.float64)
    class_n = np.zeros(n_cls, dtype=np.float64)
    for cid, n in kept["class_id"].value_counts().items():
        i = class_pos[int(cid)]
        class_n[i] = float(n)
        targets[i, :] = float(n) * ratios

    group_records: list[dict] = []
    cov_cols = [x for x in COVS if x in kept.columns]
    for gkey, gdf in kept.groupby("group_key", sort=True):
        per_class: dict[int, dict] = {}
        for cid, sub in gdf.groupby("class_id"):
            per_class[int(cid)] = {
                "k": int(len(sub)),
                "sums": {x: float(sub[x].sum()) for x in cov_cols},
                "sumsq": {
                    x: float((sub[x].astype(np.float64) ** 2).sum()) for x in cov_cols
                },
            }
        group_records.append(
            {
                "key": str(gkey),
                "n": int(len(gdf)),
                "classes": sorted(gdf["project_class"].unique()),
                "per_class": per_class,
            }
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(group_records))

    current = np.zeros_like(targets)
    assignment: dict[str, int] = {}
    for idx in order:
        rec = group_records[int(idx)]
        deficit = np.zeros(len(SPLITS), dtype=np.float64)
        for cid, pc in rec["per_class"].items():
            row = targets[class_pos[cid]]
            deficit += np.maximum(0.0, row - current[class_pos[cid]]) / np.maximum(
                row, 1.0
            )
        split_idx = int(np.argmax(deficit))
        assignment[rec["key"]] = split_idx
        for cid, pc in rec["per_class"].items():
            current[class_pos[cid], split_idx] += float(pc["k"])

    if balanced:
        mu, sd, active = _class_stats(kept, class_pos, n_cls)
        _hill_climb(
            group_records,
            class_pos,
            targets,
            class_n,
            mu,
            sd,
            active,
            kept,
            assignment,
            seed,
        )

    n_state = np.zeros((n_cls, len(SPLITS)), dtype=np.float64)
    for rec in group_records:
        s = assignment[rec["key"]]
        for cid, pc in rec["per_class"].items():
            n_state[class_pos[cid], s] += float(pc["k"])

    prop_dev = float(np.max(np.abs(n_state / class_n[:, None] - ratios[None, :])))
    if prop_dev > 0.05:
        raise RuntimeError(
            f"G4 violated after split: proportion deviation {prop_dev:.4f}"
        )

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
                "group_key": rec["key"],
                "split": SPLITS[assignment[rec["key"]]],
                "n_clips": rec["n"],
                "classes": ";".join(rec["classes"]),
            }
            for rec in group_records
        ]
    ).sort_values("group_key").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    clip_splits.to_csv(out_dir / "clip_splits.csv", index=False)
    group_splits.to_csv(out_dir / "group_splits.csv", index=False)
    return clip_splits, group_splits
