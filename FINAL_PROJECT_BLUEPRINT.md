# Edge AI Earbud Supervisor — FINAL PROJECT BLUEPRINT

**Title:** Smart Context-Aware Acoustic Supervisor Framework — Edge AI Controller for Next-Generation Earphones

**Project type:** BTech AIML Final Year Project

**Status:** Living specification. This document is the single source of truth for the project. It consolidates and supersedes
`Edge_AI_Earbud_Project_Blueprint .pdf` and `Edge_AI_Earbud_Phase_Plan.pdf`, preserving their architecture, goals,
terminology, and constraints, and records any intentional changes made during the project.

> Rule: Every change to requirements, datasets, or approach MUST be recorded below under
> **[Change Log](#change-log)** with the reason. Never change requirements silently. This document must stay
> consistent with the actual implementation as the project evolves.
>
> **Auto-update standing rule:** Whenever any project-level change occurs or is discovered — covering source,
> dataset strategy, architecture, phases, preprocessing, model, evaluation, or requirements — this document MUST
> be updated automatically to reflect the new final decision. This is a standing instruction that applies to all
> future sessions and contexts; do not wait for the user (or any reminder) to request the update.

---

## 1. Executive Project Summary

This project builds a Smart, Context-Aware **Edge AI Controller** for next-generation earphones. Instead of using heavy,
latency-prone ML pipelines to directly filter or manipulate continuous raw audio, the system acts as an intelligent
**Acoustic Supervisor**.

Operating entirely locally on a host device (simulated via laptop deployment), the framework:

1. Samples ambient audio via a standard, non-ANC wired microphone.
2. Processes input into compact 2D structural arrays (mel-spectrograms).
3. Executes real-time classification inferences on ~200 ms macro-windows.
4. Based on the environmental acoustic class, updates an execution matrix and dynamically routes audio output to
   simulate three distinct operational states.

### The Three Operational States

| Detected Class | Triggered Mode | Intended Behavior |
|---|---|---|
| **Class A — Continuous Low-Freq Hum** | **Maximum ANC Mode** | Signal detected → trigger Max ANC state (e.g., phase-masking/noise profile) |
| **Class B — Transient Danger Profile** | **Safety Transparency Mode** | Signal detected → trigger immediate transparency / direct mic passthrough |
| **Class C — Low-Amplitude Quiet** | **Low-Power Standby Mode** | Signal detected → scale down DSP processing / reduce sampling to conserve power |

> **Important clarification (signal vs. analogy):** The masking noise played during the demo in Max ANC Mode is NOT real
> noise cancellation — it is an audible stand-in so a human watching the demo can perceive that a mode-switch decision
> fired. The actual system output is a **discrete decision signal** (e.g., "Class A detected → trigger Max ANC state"). On
> real ANC hardware, this decision would be sent as a command/API call to that hardware's own ANC system. The project's
> contribution is the **classification-and-decision layer**, not audio cancellation itself. Do not present masking noise as
> the project's noise-cancellation mechanism.

---

## 2. Technical & Machine Learning Architecture

The system treats 1D audio sequences as **2D spatial feature profiles**, building on foundations from Andrew Ng's ML
Specialization.

- **Input Spatialization:** Raw time-series signals are converted into Mel-Spectrograms (or MFCCs) using `librosa`.
  This maps acoustic frequency distributions over brief temporal steps (200 ms blocks).
- **Neural Network Design:** A lightweight CNN accepts the 2D feature grids. Feature-extraction layers extract acoustic
  invariants, fully-connected layers map the properties, and a final **Softmax** layer resolves classification into the
  three classes.
- **Evaluation Framework:** Optimization tracks **Precision, Recall, and F1-score** across edge conditions, not just
  accuracy. **Minimizing False Negatives** for critical target profiles (e.g., emergency/danger signals) is a core boundary
  constraint.

### Machine-Learning Gap Bridging Matrix (from Blueprint)

| Foundation Concept (from Specialization) | Project-Specific Implementation Extension |
|---|---|
| Supervised Learning & Image Inputs | 1D raw audio → `librosa.stft` / `librosa.feature.melspectrogram` → 2D visual matrices |
| Vectorization via NumPy Matrices | Multi-dimensional array ops process real-time buffer blocks; no heavy Python loops |
| Static Tensor Evaluation | Continuous, low-latency streaming pipeline via PyAudio / sounddevice |
| Multi-Class Cross-Entropy Optimization | Softmax classification across dynamic environmental categories |

### Why a Supervisor (latency rationale — viva defense)

The model does NOT perform real-time wave inversion/phase cancellation (which would violate physical buffer-latency
limits of ADC/DAC pipelines). Instead, ML performs classification inferences every **200–300 ms**, while the audio routing
path is handled instantaneously by low-level scripts. This hybrid design keeps latencies within real-world engineering
standards.

---

## 3. Hardware Simulation Strategy & Device Requirements

100% of this architecture runs locally as an **On-Device Offline Edge AI Node** using consumer hardware. No GPU or
external microcontroller required.

| Component | Requirement |
|---|---|
| Processing Engine | Consumer laptop CPU (Intel Core i5/i7, AMD Ryzen 5/7, or Apple Silicon M-series). No external GPU required. |
| Acoustic Sensor Node | Built-in laptop mic array or standard wired earphone/headphone inline microphone |
| Dynamic Evaluation Rig | A standalone smartphone emitting target audio distributions (siren arrays, industrial hums, car honks) at controlled distances from the mic during live demos |

### Software Simulation Execution Matrix (Blueprint Section 3)

| Target Operational State | AI Classification Output | Python Simulation Execution Path |
|---|---|---|
| Maximum ANC | Continuous Low-Frequency Hum | Trigger a local looping phase-masking audio profile (e.g., optimized brown or pink noise) |
| Safety Transparency | Critical Transient / Siren Profile | Open a direct, unbuffered microphone line loop piping live ambient feed to speakers |
| Low-Power Standby | Low-Amplitude Quiet Profile | Scale down processing loops and reduce microphone sampling frequency |

---

## 4. Acoustic Physics: Bypassing Built-in Noise Suppression (**Raw PCM Streams**)

**Problem:** Consumer devices ship with built-in Environmental Noise Cancellation (ENC) and automatic gain control. These
assume distant, low-amplitude, or non-vocal audio is "background garbage" and scrub it out. Distant shouts (50 m) or car
horns (15 m) drop in physical power per the **Inverse-Square Law**, and stock drivers often remove them — which would
blind the neural network.

**Solution:** Request raw PCM capture with a **clean input flag** (via PyAudio/sounddevice) to bypass OS-level filtering.
This converts the microphone into a true raw acoustic sensor. If a wave can physically vibrate the diaphragm, it is
captured — making distant horns/shouts visible as clear frequency components for AI classification.

---

## 5. Dataset Architecture

### 5.1 Class Definition

Three balance target classes:

| Target Class | Semantic Group | Source Labels (initial mapping) |
|---|---|---|
| **Class A — Continuous Low-Freq Hum** (→ Max ANC) | Stationary, continuous low-frequency interference | engine_idling, jackhammer, drilling, air_conditioner, street_music (low-freq subset) |
| **Class B — Transient Danger Profile** (→ Safety Transparency) | Sudden, non-stationary danger signals | siren, car_horn, gun_shot, scream/shout categories if available |
| **Class C — Low-Amplitude Quiet** (→ Low-Power Standby) | Quiet / near-silence environments | silence/near-silence clips, quiet-room equivalents, low-RMS background clips |

### 5.2 Dataset Requirements (AUTHORITATIVE — updated)

These requirements update the original Phase Plan targets and take priority.

1. **Quantity target:** Aim for **~1,000 good-quality clips per class**.
2. **Final balance rule:** Final A/B/C counts **must be equal**.
3. **Shortage handling order:**
   - If a class has fewer samples, FIRST search other suitable/allowed public sources for additional valid samples,
     BEFORE reducing the larger classes.
   - Do NOT sacrifice quality just to reach 1,000.
4. **Provenance:** Keep source dataset, original label, and sample information for every clip (see §5.4).
5. **Padding policy (updated):** **Avoid heavy zero-padding of short clips.** Previously, excessive zero-padding made
   clips mostly artificial silence and distorted class characteristics. Prefer naturally suitable clips/segments and
   **reject unsuitable samples**.

### 5.3 Source Datasets (initial candidates)

| Dataset | Relevance | License / Usage Notes (verify before download) |
|---|---|---|
| **UrbanSound8K** | car horns, sirens, engine idling, street/drilling noise — primary source | Academic. Fixed label taxonomy; inspect against class map. |
| **ESC-50** | broader ambient sounds: rain, wind, quiet-room, hums | CC-BY-NC (check for academic use). |
| **Google AudioSet (optional)** | YouTube-tagged "siren", "alarm", "traffic noise" | Large; optional for shortage fill. |
| *(Additional public sources as needed)* | Any suitable/allowed source used to fill a shortage | Must be logged under provenance + change log. |

*License verification is a prerequisite before downloading/using any source.*

### 5.4 Provenance & Metadata

For each accepted clip, record (persisted under `data/provenance/`):

- Source dataset & version
- Original label / class label in source
- Sample/filename identifier
- Duration, sample rate, derived acoustic features (RMS/energy, etc.)
- Accepted target class (A / B / C)

---

## 6. Train / Validation / Test Split Validation (MANDATORY GATE — updated)

After balancing, use a **70 / 15 / 15 stratified split**.

Before ANY model training, for each class separately, the following MUST be verified:

1. **Proportion check:** Mathematically verify train/val/test have approximately the expected 70/15/15 proportions.
2. **Distribution check:** Verify that the **actual audio characteristics/distributions** of train, validation, and test
   are reasonably similar for the same class (e.g., duration, RMS/energy, and relevant features) — not merely similar
   sample counts.
3. **Leakage check:** Check for duplicates or data leakage between splits.
4. **Visual check:** Generate spectrogram comparisons for the same class across Train / Validation / Test:
   - Class A → Train vs Val vs Test spectrograms
   - Class B → Train vs Val vs Test spectrograms
   - Class C → Train vs Val vs Test spectrograms

These checks target: excessive silence, bad padding, preprocessing differences, and distribution mismatch.

> **Gate rule:** Do NOT proceed to training if a significant problem is detected in any of the above checks.
> Resolve the issue and document it before attempting training.

---

## 7. Phase-Wise Execution Plan

### Phase 1 — Data Sourcing & Class Mapping

**Estimate:** 10–15 hrs

- Download and inspect UrbanSound8K + ESC-50; verify label taxonomy against the 3-class map (§5.1).
- Write a filtering script to bucket source labels into Class A / B / C; discard irrelevant classes.
- Enforce dataset requirements §5.2 (balance, provenance, padding policy, shortage handling).
- **Hold out ~15–20 real-mic recordings per class** (phone-emitted, laptop-mic captured) for later live demo validation
  (the entire pitch is "raw PCM captures what stock filtering misses"). These are held out and not used in train split.
- Stratified 70/15/15 split.
- Run the split-validation gate (§6) before proceeding.

### Phase 2 — Signal Processing Pipeline

**Estimate:** 10–12 hrs

Convert raw audio into the 2D spectrogram inputs the CNN consumes.

| Parameter | Value | Notes |
|---|---|---|
| Sample rate | 16,000 Hz | Standard; keeps compute light |
| Window size (analysis block) | 200 ms | Matches blueprint inference cadence |
| Frame overlap / hop | 50% overlap, hop = 512 samples | Smooths classification, avoids boundary artifacts |
| Feature type | Mel-spectrogram, 64 mel bands | Chosen over raw MFCC for CNNs |
| FFT window (n_fft) | 1024 | `librosa.stft`-adjacent |
| Normalization | Per-clip min-max or log-scale (dB) | `librosa.power_to_db`, then scale to [0,1] |

Tasks:
- Write `librosa`-based conversion: raw wav → mel-spectrogram → fixed-size 2D array (pad/truncate to constant shape,
  e.g., 64 × 63). **Note:** padding must respect §5.2 padding policy — prefer naturally suitable segments; avoid
  excessive zero-padding.
- Batch-convert the whole dataset; cache as `.npy` arrays to avoid recomputing each training run.
- Visualize a few spectrograms per class to sanity-check separability (siren should look distinct from hum).

### Phase 3 — Model Design, Training & Evaluation

**Estimate:** 20–25 hrs

Train a lightweight CNN and evaluate against the false-negative constraint.

**Architecture:**
- Input: 64 × 63 × 1 mel-spectrogram array
- Conv2D (16 filters, 3×3, ReLU) → MaxPool(2×2)
- Conv2D (32 filters, 3×3, ReLU) → MaxPool(2×2)
- Conv2D (64 filters, 3×3, ReLU) → GlobalAveragePooling2D
- Dense (32, ReLU) → Dropout(0.3) → Dense(3, Softmax)
- **Target: under 100K parameters total** (genuinely edge-deployable)

**Training config:**

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam, lr=1e-3 (ReduceLROnPlateau) |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Epochs | 30–50 (early stopping on val loss, patience=5) |
| Class weighting | Weighted loss favoring Class B (danger) to bias against false negatives |

**Evaluation plan:**
- Report accuracy (secondary); **headline metrics are per-class Precision, Recall, F1.**
- Confusion matrix — explicitly check Class B (danger) false-negative rate (core safety claim).
- **Target: Recall on Class B ≥ 90%** even at some precision cost (a false alarm is acceptable; missing a real siren is not).
- Latency benchmark: measure inference time per 200 ms window on target laptop CPU; **target < 50 ms** (inside the
  200–300 ms budget).

### Phase 4 — Live Capture & Raw PCM Pipeline

**Estimate:** 12–15 hrs

- Set up PyAudio/sounddevice stream with raw input flag, bypassing OS-level ENC/AGC where the driver allows.
- Implement a rolling 200 ms buffer with the SAME hop/overlap settings as training preprocessing (mismatch is the most
  common real-world bug).
- Run held-out real-mic recordings (Phase 1) through the live pipeline first — isolates model bugs from pipeline bugs.
- Validate the blueprint's core physics claim: confirm a distant (10–15 m) phone-emitted horn/siren is picked up and
  classified; document actual vs claimed detection range honestly.

### Phase 5 — Mode Routing Logic

**Estimate:** 8–10 hrs

- Argmax on softmax output selects the candidate mode per 200 ms window.
- Add debounce/smoothing: require same class for 2–3 consecutive windows before switching, to avoid mode flicker on
  noisy boundary frames.
- **Exception:** Safety Transparency (Class B) should NOT be debounced — trigger instantly on a single high-confidence
  danger detection (latency here is a safety issue).
- Implement the 3 output behaviors per §3 matrix.
- Output is a **decision signal** to be handed to existing ANC hardware (see §1 clarification).

### Phase 6 — Demo, Report & Deployment Polish

**Estimate:** 15–20 hrs

- Live demo script: phone plays test sounds at set distances; laptop mic + pipeline classifies and switches modes visibly.
- GitHub repo: clean README, architecture diagram, metrics table (precision/recall/F1 per class), sample spectrograms,
  demo GIF/video.
- Final report: merge blueprint conceptual sections with actual results (real numbers) and viva defense answers.
- **Do NOT attempt embedded/ARM Cortex-M deployment this semester** — mention only as future work.

---

## 8. Scope Guardrails (Fixed Constraints)

- **No physical hardware / PCB build** — laptop simulation only.
- **No embedded / microcontroller deployment** — future work only (mention in report/viva as portability path via
  TensorFlow Lite / NanoEdge AI Studio for ARM Cortex-M).
- **No custom audio dataset recording as primary data source** — public datasets first; real-mic recording limited to
  demo validation samples only.
- If timeline slips, cut Phase 6 depth BEFORE cutting Phase 3 evaluation rigor — a working, honestly-evaluated model
  matters more than a polished GitHub page.

---

## 9. Acoustic Physics & "Edge AI" Viva Defense Points

Reference answers preserved from Blueprint §5:

- **Q1 (Why isn't ML noise cancellation slow?)** — ML is a high-level supervisor on 200–300 ms macro-windows; audio
  routing is done instantly by low-level scripts. Not real-time inversion.
- **Q2 (How is laptop = Edge AI?)** — All compute runs locally on the node where data is collected; offline + private,
  no network dependence. Portable to embedded later.
- **Q3 (How does a cheap wired mic capture distant danger?)** — Software noise-gates/driver filters, not the capsule, are
  the limitation. Raw PCM bypasses them; any wave that vibrates the diaphragm is captured.
- **Q4 (Why isn't this a basic filter project?)** — Static linear filters (Wiener/spectral subtraction) fail on
  non-stationary/transient noises. This uses non-linear deep CNN acoustic signature processing + host behavior change.

### Production path (practicality answer)

The system is a **decision layer** over existing ANC hardware (AirPods/Sony/Bose etc.). On real devices it would call the
device's existing ANC/transparency control API instead of running masking-noise/mic-passthrough scripts. Pipeline:
**classify → decide → command existing hardware** — directly portable without building ANC hardware.

---

## 10. Git / GitHub Workflow — Standing Rule

GitHub serves as the project's **development history and backup**, not just a final-upload target.

1. **Checkpoint reminders.** Before moving between meaningful milestones, check whether current work should be
   committed/pushed. Remind at checkpoints such as: completing a phase; completing a major part of a phase; finishing a
   significant verified-and-working implementation; completing an important validation/checkpoint; before a major
   architectural or project-level change. Do NOT interrupt trivial edits. At a real checkpoint, say explicitly:
   *"Git checkpoint reached. The current work is stable enough to commit and push before we continue."* and suggest a
   concise commit message. Reminders are advisory only.
2. **No automatic commits/pushes.** Never run `git commit` / `git push` unless the user explicitly asks.
3. **Phase transitions.** Before starting a new major phase: (a) verify the current phase is complete per this document;
   (b) confirm implementation and documentation are consistent; (c) state whether this is a good Git checkpoint; if yes,
   recommend committing and pushing before continuing.
4. **Incomplete/incorrect earlier work.** Do not treat earlier committed work as permanently fixed. If work was
   implemented unnecessarily, a blueprint requirement was missed, an implementation differs from the spec, a better
   dataset/source/approach is justified, a validation step fails, or a decision needs to change — **stop and assess**,
   then correct the project properly (never preserve a mistake merely to keep Git history clean). Any project-level change
   is recorded in this document + Change Log. Git history reflects genuine evolution; do not rewrite history to hide
   mistakes unless explicitly requested.
5. **Checkpoint principle.** Prefer: *work → verify → document → Git checkpoint → continue* over one giant final commit.
6. **Repository cleanliness.** Before recommending a major push, verify: no generated datasets/audio committed; no
   secrets/API keys; temp files ignored; important source + docs included; `.gitignore` appropriate; repo is
   understandable to a GitHub viewer.
7. **Final project.** The repo should show the genuine development process and contain the final clean implementation,
   documentation, methodology, evaluation results, and relevant materials. Do not fabricate commits, results, metrics,
   experiments, or history.

---

## 11. Change Log

Every material change to the original Blueprint + Phase Plan is recorded here. New entries are appended at the top.

| Date | Change | Reason | Status |
|---|---|---|---|
| 2026-09-07 | **Git/GitHub standing workflow established (§10):** checkpoint-based commit reminders (advisory, never automatic); explicit user request required to commit/push; pre-phase completeness + consistency check; correct-mistakes-over-preserve-history policy; cleanliness checks before major pushes; genuine-history principle. | User governance instruction. Defines how the project's development history and backup are maintained on GitHub throughout the project. | Accepted |
| 2026-09-07 | **Standing auto-update rule established:** every project-level change (source, dataset strategy, architecture, phases, preprocessing, model, evaluation, or requirements) is recorded in this document automatically at the time the change is made or discovered — no reminder required. Applies to all future sessions. | User governance instruction. Makes the change-log discipline self-executing so the spec never drifts from the implementation. | Accepted |
| 2026-09-07 | **Final dataset targets set to ~1,000 clips/class (from 300–500), with equal final A/B/C counts; shortage-handling order defined; padding policy tightened (avoid heavy zero-padding); provenance requirement added.** | User specification update. Encoded as authoritative §5.2. | Accepted |
| 2026-09-07 | **Mandatory pre-training split-validation gate added (§6): proportion check, per-class audio-distribution similarity, leakage check, and Train/Val/Test spectrogram comparison, with a do-not-train rule on failure.** | User specification update to guarantee split quality and catch bad padding/silence/distribution mismatch before training. | Accepted |
| 2026-09-07 | **Initial scoped change — "modify the plan when necessary using additional appropriate public data/sources" is explicitly permitted, but every such use must be documented (provenance §5.4 + this change log), never silently.** | User specification. Sets the rule for legitimate future deviation from the original Phase Plan's dataset scope. | Accepted |

*Future entries: whenever we use additional public data sources, adjust class mapping, alter hyperparameters, or change
any requirement, record it here with the reason.*

---

*Companion sources (initial, now superseded by this living document): Edge_AI_Earbud_Project_Blueprint .pdf,
Edge_AI_Earbud_Phase_Plan.pdf.*