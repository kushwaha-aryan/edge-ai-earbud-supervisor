# Phase 2A Preprocessing — Execution Report

**Date:** 2026-10-08
**Policy:** FINAL_PROJECT_BLUEPRINT.md Change Log 2026-10-08 (decisions D1–D12 + amendments OA1–OA4)
**Driver:** `scripts/run_phase2a_preprocessing.py` (stages: ledger → split → features → gates)
**Verdict: PASS — all D11 leakage gates (G1–G5) passed; artifacts ready for Phase 2B/Phase 3.**

Runtime: 515 s total (506 s feature extraction) on the project venv (Python 3.13.14, librosa 1.0.0, soxr 1.1.0, numpy 2.5.3, pandas 3.0.6).

---

## 1. Stage 1 — clip ledger (D1, D4, D2)

Input: `data/manifests/master_audio_manifest.csv` (10,565 rows, immutable) ×
`scripts/results/dataset_validation/file_health.csv` (row-aligned check passed).

| Outcome | Clips |
|---|---|
| Kept | **10,281** |
| Excluded `extreme_short(<0.5s)` (D1) | 254 |
| Excluded `silent(RMS<=-60dBFS)` (D4) | 30 |
| Excluded total (no overlap) | 284 |

Kept per class (n=17): Car_Engine 1000, Jackhammer 999, Drilling 993, Dog_Bark 950,
Siren 904, Gunshot 748, Alarm 747, Footsteps 562, Baby_Crying 533, Help_Shouting 493,
Glass_Breaking 492, Train 467, Vehicle_Horn 386, Knocking 380, Aircraft 338,
Motorcycle 238, **Doorbell 51** (smallest class; Smoke_Fire_Alarm 0 — UNRESOLVED).

Ledger: `data/provenance/phase2a_clip_ledger.csv` (10,565 rows, 27 columns) — one row
per manifest clip with source dataset/release, original label, sample id + grouping id
(OA4 keys), duration/sample rate/energy, accepted class, status/reason, and every
applied parameter (truncate 15.0 s, resample 16 kHz, mono, window 3,200/step 1,600,
pad fraction 0.0). Covers blueprint §5.4.

## 2. Stage 2 — seeded source-group split (D9)

Greedy per-class deficit assignment, `np.random.default_rng(42)`, groups atomic.

| Split | Clips | Share |
|---|---|---|
| train | 7,110 | 69.16% |
| val | 1,577 | 15.34% |
| test | 1,594 | 15.50% |

- **5,341 source groups** exist in the ledger; **5,201** contain at least one kept clip
  (140 groups lost entirely to exclusions); split rows: train 3,611 / val 817 / test 773.
- Group keys (OA4): `urbansound8k/<fsID>`, `fsd50k/<source_id>`, `esc-50/<src_file>`,
  `babycry/donateacry:<uuid>` (170 subjects), `babycry/recanvo:<YYMMDD_HHMM>` (14 sessions).
- **4 multi-class groups** — the W6 US8K fsIDs (180937, 77751, 106905, 176638) moved
  atomically as required.
- Outputs: `data/splits/clip_splits.csv`, `data/splits/group_splits.csv`.

### Per-class proportions vs 70/15/15 target (gate tolerance ±5 pp)

| Class | train | val | test | n |
|---|---|---|---|---|
| Aircraft | 0.698 | 0.151 | 0.151 | 338 |
| Alarm | 0.700 | 0.150 | 0.150 | 747 |
| Baby_Crying | 0.702 | 0.148 | 0.150 | 533 |
| Car_Engine | 0.673 | 0.160 | 0.167 | 1000 |
| Dog_Bark | 0.701 | 0.149 | 0.149 | 950 |
| Doorbell | 0.686 | 0.157 | 0.157 | 51 |
| Drilling | 0.691 | 0.162 | 0.147 | 993 |
| Footsteps | 0.698 | 0.151 | 0.151 | 562 |
| Glass_Breaking | 0.699 | 0.150 | 0.150 | 492 |
| Gunshot | 0.693 | 0.156 | 0.151 | 748 |
| Help_Shouting | 0.700 | 0.150 | 0.150 | 493 |
| Jackhammer | 0.682 | 0.142 | 0.176 | 999 |
| Knocking | 0.700 | 0.150 | 0.150 | 380 |
| Motorcycle | 0.698 | 0.151 | 0.151 | 238 |
| Siren | 0.673 | 0.167 | 0.160 | 904 |
| Train | 0.700 | 0.150 | 0.150 | 467 |
| Vehicle_Horn | 0.700 | 0.150 | 0.150 | 386 |
| **ALL** | **0.692** | **0.153** | **0.155** | **10,281** |

**Max absolute deviation: 2.74 pp** (Car_Engine/Siren train 0.673; Jackhammer test
0.176). Well inside the ±5 pp tolerance; the loose end comes from large atomic groups
(e.g. one US8K fsID contributes 100 Siren clips).

## 3. Stage 3 — features (D2, D5, D6, D3/D7, OA1–OA3)

Pipeline per kept clip: soundfile read → mean-mono → soxr_hq 16 kHz → first 15.0 s →
windows of 3,200 samples every 1,600 → drop windows ≤ −60 dBFS → batched
`librosa.melspectrogram` (n_fft 1024, hop 512, 64 mels, center=False) → exactly 5
frames → `power_to_db(ref=1.0, amin=1e-10, top_db=None)` → scale to [0,1] with
**train-frozen dB bounds**.

| Split | Clips | Raw windows | Dropped (≤ −60 dBFS) | Kept windows |
|---|---|---|---|---|
| train | 7,110 | 351,405 | 24,855 | **326,550** |
| val | 1,577 | 79,802 | 5,446 | **74,356** |
| test | 1,594 | 79,129 | 5,104 | **74,025** |
| **total** | 10,281 | 510,336 | 35,405 (6.9%) | **474,931** |

- Frozen dB range: **[−100.00, +34.16]** (floor is the `amin=1e-10` limit of absolute
  `power_to_db`; `top_db=None` so no per-clip clipping), recorded with all parameters in
  `data/processed/feature_stats.json`.
- Artifacts (float32, shape (N, 64, 5, 1); labels int64):
  `features_train.npy` 398.6 MB, `features_val.npy` 90.8 MB, `features_test.npy`
  90.4 MB + `labels_*.npy` + `window_index_*.csv` (clip_id, class, start_sample for
  every window).
- Observed value ranges: train [0.0, 1.0], val [0.0, 0.9898], test [0.0, 0.9993].

## 4. Stage 4 — D11 leakage gates

Details: `scripts/results/phase2a_gates.json`. **All passed:**

| Gate | Check | Result |
|---|---|---|
| G1 | clip_id in exactly one split; split covers all 10,281 kept clips | PASS |
| G2 | 5,201 groups each map to exactly one split; group_splits ≡ clip_splits | PASS |
| G3 | SHA-256 disjoint across splits | PASS — **0 shared hashes** |
| G4 | per-class + overall proportions within ±5 pp | PASS — max dev 2.74 pp |
| G5 | shapes (·,64,5,1), float32, finite ∈ [0,1], labels ∈ [0,17), window-index and stats counts consistent | PASS |

## 5. Notes, deviations, open items

- **OA4** (discovered during implementation, recorded in the approval entry): babycry
  group keys use correct UUID/session extraction (170 DonateACry subjects, 14 ReCANVo
  sessions), replacing the Phase 1.5 regex that undercounted subjects.
- **Silent-window policy:** D4 was specified for clip exclusion via file_health flags;
  the same −60 dBFS floor is applied at window level (35,405 interior quiet windows
  dropped, 6.9% of raw windows). Documented here as the operational reading of D4.
- **Tolerance:** G4 uses ±5 pp (documented as PROPORTION_TOLERANCE in
  `src/preprocessing/gates.py`) because atomic groups up to 100 clips make tighter
  per-class guarantees impossible without breaking group integrity (D9 > exact ratios).
- **W1 resolved (2026-10-08):** the 5,732 UrbanSound8K clips were copied from the
  Kaggle cache into `data/raw/urbansound8k/fold*/` with SHA-256 verified against the
  ledger, and manifest `local_path` rewritten project-relative — the repo is now
  self-contained (1.4 GB, `data/raw/*` stays gitignored).
- `Smoke_Fire_Alarm` still has 0 clips (W2, UNRESOLVED); 17-class taxonomy is used.
- §6 full gate (per-class distribution similarity + visual spectrogram panels) is not
  part of D11 — proposed as Phase 2B before training.

## 6. Reproduce

```powershell
& ".\.venv\Scripts\python.exe" scripts\run_phase2a_preprocessing.py            # all stages, gates must PASS
& ".\.venv\Scripts\python.exe" scripts\run_phase2a_preprocessing.py --stage gates
& ".\.venv\Scripts\python.exe" scripts\run_phase2a_preprocessing.py --stage features --limit 50   # smoke
```

Exit code is non-zero if any gate fails.
