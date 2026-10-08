# Phase 2A Preprocessing — Execution Report

**Date:** 2026-10-08 (re-executed after the D9 v2 split amendment — see §2)
**Policy:** FINAL_PROJECT_BLUEPRINT.md Change Log 2026-10-08 (decisions D1–D12 + amendments OA1–OA4, D9 amended after the Phase 2B gate FAIL)
**Driver:** `scripts/run_phase2a_preprocessing.py` (stages [1/5] ledger → [2/5] split → [3/5] features → [4/5] covariates → [5/5] gates)
**Verdict: PASS — all D11 leakage gates (G1–G5) passed; Phase 2B §6 gate verdict PASS WITH WARNINGS (`scripts/results/phase2b_split_gate/phase2b_report.md`); artifacts ready for Phase 3 training.**

Runtime: 239 s total (192 s feature extraction) on the project venv (Python 3.13.14, librosa 1.0.0, soxr 1.1.0, numpy 2.5.3, pandas 3.0.6).

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

## 2. Stage 2 — seeded source-group split (D9, amended to D9 v2)

Three deterministic phases, `np.random.default_rng(42)`, groups atomic
(`src/preprocessing/splits.py`):

1. **count greedy (original D9)** — seeded group order, smallest summed relative
   per-class deficit against the 70/15/15 targets.
2. **covariate hill-climb** — single-group moves over a per-class objective:
   count deviation, means/spreads (class-sd units) of `processed_duration_s` /
   `rms_db` / `clip_feature_mean`, plus a 10-bin binned-quantile histogram match.
3. **exact-KS repair** — single-group moves with lexicographic accept over
   (cells crossing the gate C2 flag thresholds, then energy); cells with < 15
   clips per side are never optimized (underpowered rule).

Phases 2–3 are skipped with a notice when `data/provenance/clip_covariates.csv`
is absent (stage [4/5]; cold start = `--stage all` → `--stage covariates` →
`--stage all`). Per-class count band: max(3% of class size, initial deviation).

| Split | Clips | Share | Groups |
|---|---|---|---|
| train | 7,021 | 68.29% | 3,492 |
| val | 1,618 | 15.74% | 857 |
| test | 1,642 | 15.97% | 852 |

- **5,341 source groups** exist in the ledger; **5,201** contain at least one kept
  clip (140 groups lost entirely to exclusions).
- Group keys (OA4): `urbansound8k/<fsID>`, `fsd50k/<source_id>`, `esc-50/<src_file>`,
  `babycry/donateacry:<uuid>` (170 subjects), `babycry/recanvo:<YYMMDD_HHMM>` (14 sessions).
- **4 multi-class groups** — the W6 US8K fsIDs (180937, 77751, 106905, 176638) moved
  atomically as required.
- Outputs: `data/splits/clip_splits.csv`, `data/splits/group_splits.csv`.

### Per-class proportions vs 70/15/15 target (gate tolerance ±5 pp)

| Class | train | val | test | n |
|---|---|---|---|---|
| Aircraft | 0.692 | 0.151 | 0.157 | 338 |
| Alarm | 0.681 | 0.161 | 0.158 | 747 |
| Baby_Crying | 0.730 | 0.131 | 0.139 | 533 |
| Car_Engine | 0.674 | 0.146 | 0.180 | 1000 |
| Dog_Bark | 0.683 | 0.166 | 0.151 | 950 |
| Doorbell | 0.686 | 0.157 | 0.157 | 51 |
| Drilling | 0.678 | 0.169 | 0.153 | 993 |
| Footsteps | 0.687 | 0.155 | 0.158 | 562 |
| Glass_Breaking | 0.695 | 0.152 | 0.152 | 492 |
| Gunshot | 0.671 | 0.163 | 0.166 | 748 |
| Help_Shouting | 0.696 | 0.154 | 0.150 | 493 |
| Jackhammer | 0.673 | 0.148 | 0.179 | 999 |
| Knocking | 0.682 | 0.166 | 0.153 | 380 |
| Motorcycle | 0.685 | 0.147 | 0.168 | 238 |
| Siren | 0.684 | 0.157 | 0.159 | 904 |
| Train | 0.672 | 0.171 | 0.156 | 467 |
| Vehicle_Horn | 0.671 | 0.179 | 0.150 | 386 |
| **ALL** | **0.683** | **0.157** | **0.160** | **10,281** |

**Max absolute deviation: 3.00 pp** (Baby_Crying train 0.730, Car_Engine test
0.180). Well inside the ±5 pp tolerance; the loose end comes from large atomic
groups (e.g. one US8K fsID contributes 100 Siren clips).

## 3. Stage 3 — features (D2, D5, D6, D3/D7, OA1–OA3)

Pipeline per kept clip: soundfile read → mean-mono → soxr_hq 16 kHz → first 15.0 s →
windows of 3,200 samples every 1,600 → drop windows ≤ −60 dBFS → batched
`librosa.melspectrogram` (n_fft 1024, hop 512, 64 mels, center=False) → exactly 5
frames → `power_to_db(ref=1.0, amin=1e-10, top_db=None)` → scale to [0,1] with
**train-frozen dB bounds**.

| Split | Clips | Raw windows | Dropped (≤ −60 dBFS) | Kept windows |
|---|---|---|---|---|
| train | 7,021 | 348,412 | 23,799 | **324,613** |
| val | 1,618 | 80,554 | 5,724 | **74,830** |
| test | 1,642 | 81,370 | 5,882 | **75,488** |
| **total** | 10,281 | 510,336 | 35,405 (6.9%) | **474,931** |

- Frozen dB range: **[−100.00, +33.84]** (floor is the `amin=1e-10` limit of absolute
  `power_to_db`; `top_db=None` so no per-clip clipping), recorded with all parameters in
  `data/processed/feature_stats.json` (re-frozen from the D9 v2 train split).
- Artifacts (float32, shape (N, 64, 5, 1); labels int64):
  `features_train.npy` 415.5 MB, `features_val.npy` 95.8 MB, `features_test.npy`
  96.6 MB + `labels_*.npy` + `window_index_*.csv` (clip_id, class, start_sample for
  every window).
- Observed value ranges: train [0.0, 1.0], val [0.0, 1.0], test [0.0, 1.0].

## 4. Stage 4 — clip covariates (D9 v2 support)

`features.compute_clip_covariates()` derives one row per kept clip —
`processed_duration_s`, `rms_db`, `clip_feature_mean` (mel-dB mean over the clip,
affine-equivalent to the raw mel-dB clip mean under the frozen bounds) — into
`data/provenance/clip_covariates.csv` (10,281 rows). Feeds split phases 2–3 and the
Phase 2B C1/C2 gate cells; skipped when the CSV already exists (it never needs
refreshing when the split changes).

## 5. Stage 5 — D11 leakage gates

Details: `scripts/results/phase2a_gates.json`. **All passed:**

| Gate | Check | Result |
|---|---|---|
| G1 | clip_id in exactly one split; split covers all 10,281 kept clips | PASS |
| G2 | 5,201 groups each map to exactly one split; group_splits ≡ clip_splits | PASS |
| G3 | SHA-256 disjoint across splits | PASS — **0 shared hashes** |
| G4 | per-class + overall proportions within ±5 pp | PASS — max dev 3.00 pp |
| G5 | shapes (·,64,5,1), float32, finite ∈ [0,1], labels ∈ [0,17), window-index and stats counts consistent | PASS |

## 6. Notes, deviations, open items

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
- **Phase 2B executed (2026-10-08):** the §6 gate (`scripts/run_phase2b_split_gate.py`)
  initially FAILED the counts-only split (11 severe + 34 flag cells); D9 was amended to
  the three-phase covariate-balanced split described in §2 and the pipeline re-run.
  Verdict on the new split: **PASS WITH WARNINGS** (0 severe / 3 warnings, all other
  checks PASS) — see `scripts/results/phase2b_split_gate/phase2b_report.md` and the
  Change Log entries.

## 7. Reproduce

```powershell
& ".\.venv\Scripts\python.exe" scripts\run_phase2a_preprocessing.py                      # stages 1-3 + 5 (covariates only if missing)
& ".\.venv\Scripts\python.exe" scripts\run_phase2a_preprocessing.py --stage covariates   # cold start: build clip_covariates.csv, then re-run "all"
& ".\.venv\Scripts\python.exe" scripts\run_phase2a_preprocessing.py --stage gates
& ".\.venv\Scripts\python.exe" scripts\run_phase2b_split_gate.py                          # section 6 gate, exit 0 = PASS / PASS WITH WARNINGS
& ".\.venv\Scripts\python.exe" scripts\run_phase2a_preprocessing.py --stage features --limit 50   # smoke
```

Exit code is non-zero if any gate fails.
