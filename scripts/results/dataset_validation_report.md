# Phase 1.5 — Dataset Sanity & Validation Report

**Date:** 2026-10-07
**Manifest:** `data/manifests/master_audio_manifest.csv` (10,565 rows — NOT modified)
**Scope:** validation only. No downloads, no audio modification, no preprocessing, no split creation.
**Artifacts:** `scripts/results/dataset_validation/file_health.csv` (per-file metrics), `scripts/results/dataset_validation/validation_data.json`, `scripts/results/dataset_validation/diagnostics.csv`, `scripts/results/dataset_validation_summary.csv`, `scripts/results/dataset_validation/spectrograms/*.png` (17 figures, 4 clips per class).

**Thresholds used** (all reported below):
- extremely short: duration < 0.5 s; short: < 1.0 s; long: > 15 s; very long: > 30 s
- silent: RMS ≤ −60 dBFS; low-energy: −60 < RMS ≤ −45 dBFS
- clipped: ≥ 0.1 % of samples at |x| ≥ 0.999
- integrity: SHA-256 recomputed for every file and compared against manifest `v_sha256`

---

## 1. Dataset inventory

| Metric | Value |
|---|---:|
| Manifest rows | 10,565 |
| Unique paths | 10,565 |
| Files present | **10,565 / 10,565** |
| Missing files | 0 |
| Readable | 10,565 |
| Unreadable / corrupt | 0 |
| Zero-length files | 0 |
| SHA-256 mismatches vs manifest | 0 |
| Exact duplicate files | 0 |
| Extra audio on disk not in manifest | 0 (under `data/raw/`) |
| Total audio | 16.26 hours |

**Warning W1 — US8K file location:** all 5,732 UrbanSound8K `local_path` values are **absolute paths into the Kaggle cache** (`C:\Users\BITPATNA\.cache\kagglehub\datasets\chrisfilo\urbansound8k\versions\1\foldN\*.wav`), not project files. They exist and validate, but the dataset is not self-contained: if the Kaggle cache is cleaned, 54 % of the dataset disappears. FSD50K (4,027), ESC-50 (306) and babycry (500) live under `data/raw/`. Recommend copying US8K into `data/raw/` before preprocessing.

## 2. Class counts (17 of 18 planned classes)

| Project Class | Clips | Source(s) |
|---|---:|---|
| Dog_Bark | 1000 | UrbanSound8K |
| Car_Engine | 1000 | UrbanSound8K |
| Drilling | 1000 | UrbanSound8K |
| Jackhammer | 1000 | UrbanSound8K |
| Siren | 929 | UrbanSound8K |
| Gunshot | 793 | UrbanSound8K, FSD50K |
| Alarm | 755 | FSD50K, ESC-50 |
| Footsteps | 585 | FSD50K, ESC-50 |
| Baby_Crying | 540 | owlgebra_babycry, ESC-50 |
| Glass_Breaking | 515 | FSD50K, ESC-50 |
| Help_Shouting | 498 | FSD50K |
| Train | 468 | FSD50K, ESC-50 |
| Vehicle_Horn | 466 | UrbanSound8K, FSD50K, ESC-50 |
| Knocking | 387 | FSD50K, ESC-50 |
| Aircraft | 338 | FSD50K, ESC-50 |
| Motorcycle | 238 | FSD50K |
| Doorbell | 53 | FSD50K |
| **Smoke_Fire_Alarm** | **0** | **UNRESOLVED — no clips** |

Strongest: Dog_Bark/Car_Engine/Drilling/Jackhammer (1,000). Weakest: Doorbell (53), Motorcycle (238), Aircraft (338), Knocking (387). Imbalance ratio ≈ 19:1. Single-source classes: Doorbell, Motorcycle, Help_Shouting (FSD50K-only) and Siren, Dog_Bark, Car_Engine, Drilling, Jackhammer (US8K-only) — 8/17 classes depend on one dataset's recording characteristics.

## 3. Source counts

| Source dataset | Clips | Unique source IDs |
|---|---:|---:|
| UrbanSound8K | 5,732 | 910 fsIDs |
| FSD50K | 4,027 | 4,027 (fname-level) |
| owlgebra_babycry | 500 | 500 (338 subject UUIDs) |
| ESC-50 | 306 | 220 src_file IDs |

No `source_id` appears in more than one dataset (cross-dataset overlap check: none).

## 4. Duplicate analysis

- Exact duplicate SHA-256 groups: **0** (recomputed for all 10,565 files; matches manifest).
- No duplicates within or across source datasets. The dataset contains no redundant audio.

## 5. Source-level leakage analysis

- 5,657 source groups total; **707 groups contain >1 clip** (all UrbanSound8K fsID slices and ESC-50 src_file takes).
- Largest groups: US8K fsID 24347 → 100 Siren clips; fsID 180937 → 95 clips; fsID 72259 → 73 Vehicle_Horn; fsID 203929 → 72 Jackhammer.
- **4 US8K source groups span two project classes** (same underlying recording, differently labeled slices):
  - `fsID 180937` — 95 clips: Drilling + Jackhammer
  - `fsID 77751` — 30 clips: Drilling + Jackhammer
  - `fsID 106905` — 7 clips: Car_Engine + Siren
  - `fsID 176638` — 5 clips: Car_Engine + Vehicle_Horn
  These groups must stay intact in any train/val/test split (they would otherwise leak across classes).
- ESC-50: no source group spans more than one class.
- babycry: 32 of 338 subject UUIDs have >1 clip (max 8 clips/subject) — subject-aware grouping advisable for Baby_Crying.
- **Caveat:** FSD50K `source_id` equals the FSD50K file name (unique per clip), so deeper sharing (same Freesound upload / YouTube video) cannot be detected from the manifest; treat FSD50K rows as independent sources but expect some residual hidden correlation.

## 6. Duration statistics

Overall: min 0.05 s, median 4.00 s, mean 5.54 s, max 34.06 s, σ 5.65 s.

| Class | min | median | mean | max | hours |
|---|---:|---:|---:|---:|---:|
| Aircraft | 0.91 | 11.14 | 13.63 | 30.00 | 1.28 |
| Alarm | 0.30 | 6.62 | 9.51 | 30.02 | 1.99 |
| Baby_Crying | 0.26 | 5.00 | 4.42 | 7.39 | 0.66 |
| Car_Engine | 0.77 | 4.00 | 3.94 | 4.00 | 1.09 |
| Dog_Bark | 0.12 | 4.00 | 3.15 | 4.00 | 0.87 |
| Doorbell | 0.37 | 6.66 | 8.50 | 29.28 | 0.13 |
| Drilling | 0.42 | 4.00 | 3.55 | 4.01 | 0.99 |
| Footsteps | 0.30 | 6.70 | 8.51 | 30.00 | 1.38 |
| Glass_Breaking | 0.31 | 2.75 | 4.22 | 27.74 | 0.60 |
| Gunshot | 0.17 | 1.82 | 3.11 | 30.00 | 0.69 |
| Help_Shouting | 0.41 | 3.96 | 6.26 | 29.75 | 0.87 |
| Jackhammer | 0.39 | 4.00 | 3.61 | 4.00 | 1.00 |
| Knocking | 0.33 | 3.00 | 5.07 | 27.89 | 0.55 |
| Motorcycle | 1.32 | 13.01 | 14.36 | 30.00 | 0.95 |
| Siren | 0.26 | 4.00 | 3.91 | 4.00 | 1.01 |
| Train | 0.60 | 13.95 | 13.88 | 34.06 | 1.80 |
| Vehicle_Horn | 0.05 | 3.86 | 3.03 | 30.00 | 0.39 |

Unusual distributions: US8K classes are fixed ~4 s slices (but 135 Dog_Bark and 137 Vehicle_Horn clips are <1 s — US8K contains sub-second slices); FSD50K classes have a long 1–30 s tail (875 clips >15 s, 3 clips >30 s); Train/Motorcycle/Aircraft are long-form (medians 11–14 s). Preprocessing must handle both sub-second and 30 s clips (trimming/windowing).

## 7. Sample-rate / channel consistency

- **10 distinct sample rates:** 8,000 Hz ×263 (babycry), 11,025 ×31, 16,000 ×39, 22,050 ×37, 24,000 ×32, 32,000 ×4, 44,100 ×7,989, 48,000 ×1,601, 96,000 ×552, 192,000 ×17.
- Per dataset: FSD50K uniformly 44.1 kHz; ESC-50 44.1 kHz; babycry 8 kHz; **UrbanSound8K is mixed** (11 kHz–192 kHz, incl. 17 files at 192 kHz).
- Channels: 5,199 mono / 5,366 stereo (US8K mostly stereo; FSD50K/ESC-50/babycry mono).
- No conversion performed. Resampling + downmix is required preprocessing work (not a defect).

## 8. Silence / low-energy findings

- Silent (RMS ≤ −60 dBFS): **30 clips (0.28 %)** — Siren 21, Footsteps 5, Alarm 1, Doorbell 1, Train 1, Gunshot 1.
- Low-energy (−60 < RMS ≤ −45 dBFS): **308 clips (2.9 %)** — Siren 81, Footsteps 99, Train 7, Glass_Breaking 14, etc.
- Worst examples: Doorbell clip at −76.8 dB (2.5 s, 100 % silent frames), Footsteps clip at −71.6 dB (98.6 % silent frames), Train clip at −63.5 dB (15 s, 82.5 % silent), Alarm clip at −60.8 dB (19 s, 75 % silent).
- Many FSD50K excerpts contain long room-tone gaps (e.g. Knocking median clip 65 % silent frames, Doorbell 58 %, Glass_Breaking longest 63 %) — expected for un-trimmed Freesound excerpts; matters for windowing, not for integrity.

## 9. Suspicious / missing / corrupt files

- Missing/corrupt/zero-length/unreadable: **0**.
- Clips with ≥1 flag: **2,580 (24.4 %)**: short(<1 s) 862 (254 <0.5 s), long(>15 s) 875 + very long 3, clipped 685, silent 30, low-energy 308 (flags overlap).
- Clipping concentrated in **Gunshot: 326/793 (41 %)** — impulsive recordings frequently hit full scale (US8K gunshots are known to clip); also Car_Engine 65, Glass_Breaking 52, Dog_Bark 51.
- Notable extremes: Vehicle_Horn clip of **0.05 s** (50 ms), Dog_Bark 0.12 s, Baby_Crying 0.26 s, Siren 0.26 s, Alarm 0.30 s.
- DC offset: negligible overall (|DC| < 0.005) except some US8K Gunshot clips (up to 0.035).
- No rows were removed. These are candidates for preprocessing-time trimming/filtering decisions.

## 10. Label / mapping concerns

- **UrbanSound8K (clean 1:1):** only 7 of 10 US8K labels used — `car_horn→Vehicle_Horn`, `dog_bark→Dog_Bark`, `drilling→Drilling`, `engine_idling→Car_Engine`, `gun_shot→Gunshot`, `jackhammer→Jackhammer`, `siren→Siren`. `air_conditioner`, `children_playing`, `street_music` correctly **not** mapped — in particular there is **no `children_playing→Baby_Crying` mapping**. Note `engine_idling` is narrow (stationary engines only; no revving/acceleration).
- **ESC-50 (clean):** `airplane`+`helicopter→Aircraft` (rotorcraft-as-aircraft is a semantic choice, acceptable), `clock_alarm→Alarm`, `crying_baby→Baby_Crying`, `door_wood_knock→Knocking`, `footsteps`, `glass_breaking`, `train`, `car_horn→Vehicle_Horn`.
- **babycry (clean):** all 8 labels are cry-reason categories (hungry, belly_pain, discomfort, tired, frustrated, dysregulated, dysregulation-sick, dysregulation-bathroom) — all genuine crying.
- **FSD50K (multi-label ontology — review needed):**
  - `Alarm` (733 clips) is **semantically broad**: 241/733 co-labeled with bicycle bells, doorbells, clocks, engines, trains, fireworks, etc. (e.g. `Bicycle_bell,Alarm,Bicycle,Vehicle,Bell`). The Alarm class is heterogeneous.
  - `Help_Shouting`: 77/498 co-labeled with Applause/Crowd/Cheering/Clapping (party/crowd context, e.g. `Applause,Crowd,Cheering,Clapping,Shout,...`); one row also matched a music clip (`Acoustic_guitar,Yell,Strum,...,Music,Shout`).
  - Minor: Knocking 2/347 and Footsteps 2/545 co-labeled with Music/Percussion; Train 7/428 with Hiss/Alarm.
  - Normal co-occurrence (not defects): Gunshot↔Explosion (419/419), Doorbell↔Bell/Alarm (53/53), Vehicle_Horn↔Alarm (33/33), Motorcycle↔Engine (114/238).
  - No generic `noise`-type labels were mapped to any class.

## 11. Split feasibility (70/15/15, source-aware)

Rule: FEASIBLE if ≥20 source groups and largest group ≤20 % of class; MARGINAL if ≥7 groups; else INFEASIBLE.

**All 17 classes are FEASIBLE.** Weakest case: Jackhammer (45 groups, largest 79 clips = 7.9 %), Siren (74 groups, largest 100 = 10.8 %), Vehicle_Horn (162 groups, largest 73 = 15.7 %). Doorbell (53 groups of 1), Motorcycle (238 groups of 1), Help_Shouting (498 of 1) are ideal. Requirements for the split phase: (a) keep the 4 cross-class US8K fsID groups intact; (b) group babycry by subject UUID (32 multi-clip subjects); (c) treat FSD50K rows as independent (fname-granularity caveat, §5); (d) do not use US8K's built-in `fold` column as the val/test split — it is not source-aware for our class set.

## 12. Diagnostic observations

Diagnostic figures (waveform + spectrogram, 4 clips per class: shortest/median/longest/lowest-energy) are saved under `scripts/results/dataset_validation/spectrograms/`. They were generated for human review; **they could not be machine-inspected in this session (this model has no image input)**, so the visual check was performed numerically instead (`diagnostics.csv`):
- No broken/flat/garbage audio: every file decodes; energy, ZCR and spectral-centroid ranges are consistent with class content (e.g. Siren centroid ~1 kHz tonal; Doorbell 7–9 kHz; Drilling/Jackhammer 2.5–9 kHz with high ZCR 0.12–0.29; Train/Footsteps low ZCR ~0.02–0.03).
- Clipping visible as full-scale plateaus in impulsive classes (Gunshot 41 % of clips ≥0.1 % full-scale samples).
- Silent/low-energy outliers listed in §8 are the main "obvious" anomalies (e.g. the fully silent 2.5 s Doorbell clip).
- Recording characteristics differ strongly by source (US8K mixed-rate stereo slices vs FSD50K 44.1 kHz mono excerpts vs 8 kHz babycry) — expected; normalization is preprocessing's job.

## 13. Overall decision

### **PASS WITH WARNINGS**

The dataset is structurally healthy — 10,565/10,565 files present, readable, integrity-verified, zero duplicates, zero cross-dataset ID leakage, and every class supports a source-aware 70/15/15 split. It can proceed to preprocessing **provided the following are handled/documented there**:

- **W1:** US8K audio lives in the Kaggle cache, not the project tree (copy into `data/raw/`).
- **W2:** `Smoke_Fire_Alarm` has **0 clips** (17/18 classes) — unresolved acquisition gap.
- **W3:** 862 clips <1 s and 878 clips >15 s — trimming/windowing policy needed.
- **W4:** 685 clipped clips (Gunshot 41 %) — clipping tolerance/removal policy needed.
- **W5:** 30 silent + 308 low-energy clips — removal candidates.
- **W6:** 4 US8K source groups span two classes; babycry has 32 multi-clip subjects — group-aware splitting mandatory.
- **W7:** 10 sample rates (8 kHz–192 kHz) and mono/stereo mix — resample/downmix in preprocessing.
- **W8:** FSD50K `Alarm` class is semantically broad (bicycle bells, doorbells, clocks co-labeled); Help_Shouting has 15 % crowd-context clips — review mapping before training.
- **W9:** Class imbalance 19:1; Doorbell (53) and Motorcycle (238) are thin.
- **W10:** FSD50K source granularity is file-level; residual hidden source correlation possible.

Not a FAIL: nothing is missing, corrupt, duplicated, or unreadable. Not a clean PASS: the warnings above are material and must be addressed in preprocessing/splitting.
