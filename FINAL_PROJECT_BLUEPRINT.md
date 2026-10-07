# Edge AI Earbud Supervisor — FINAL PROJECT BLUEPRINT

**Title:** Smart Context-Aware Acoustic Supervisor Framework — Edge AI Controller for Next-Generation Earphones

**Project type:** BTech AIML Final Year Project

**Status:** Living specification. This document is the single source of truth for the project. It consolidates and
supersedes `Edge_AI_Earbud_Project_Blueprint .pdf` and `Edge_AI_Earbud_Phase_Plan.pdf`, preserving their architecture,
goals, terminology, and constraints, and records any intentional changes made during the project.

**Revision:** Rev 3 — **Complete Context-Aware Acoustic Supervisor.** The system is specified as four conceptual
stages — **Perception → Context/Policy → Decision → Response** — with a dedicated **Context Response Engine**
coordinating audio-mode changes, notifications, priority warnings, user-confirmed quick actions, and state
restoration. (Rev 2 established the user-selectable acoustic trigger architecture; Rev 1's three broad
acoustic-class framing is retained below as historical record. All three revisions are recorded in
[§11 Change Log](#11-change-log).)

**Prior-art / design reference (not reproduced):** "Context-Aware Adjustment of Noise Cancellation for Wearable
Audio Devices" — Shailesh Maheshwari, Vedant Maheshwari, Rayansh Maheshwari, Technical Disclosure Commons,
27 October 2025. <https://www.tdcommons.org/cgi/viewcontent.cgi?article=10034&context=dpubs_series>

> **Scope of the reference:** this publication is used as a **design reference / prior-art reference** for the
> context-aware ANC-adjustment problem setting only. The project does **not** claim to reproduce, reimplement, or
> derive its dataset, trained model, or experimental results from it. Any overlap is conceptual, and any actual
> implementation is the project's own work.

> Rule: Every change to requirements, datasets, or approach MUST be recorded below under
> **[Change Log](#11-change-log)** with the reason. Never change requirements silently. This document must stay
> consistent with the actual implementation as the project evolves.
>
> **Auto-update standing rule:** Whenever any project-level change occurs or is discovered — covering source,
> dataset strategy, architecture, phases, preprocessing, model, evaluation, or requirements — this document MUST
> be updated automatically to reflect the new final decision. This is a standing instruction that applies to all
> future sessions and contexts; do not wait for the user (or any reminder) to request the update.

---

## 1. Executive Project Summary

This project builds a Smart, Context-Aware **Edge AI Controller** for next-generation earphones. Rather than
attempting to filter or manipulate continuous raw audio with a heavy ML pipeline, the system acts as an intelligent
**Acoustic Supervisor**.

The system is specified as **four deliberately separated conceptual stages** (the core architectural framing of
Rev 3):

1. **Perception (detection)** — *"What sound is currently present?"* A lightweight acoustic event classifier scores
   the supported environmental sound classes for each short window.
2. **Context / policy** — *"Does it matter to this user, right now?"* A user-configurable trigger policy decides
   which detected classes are enabled as triggers (per-class enable flags, confidence threshold, profile context).
3. **Decision (verification)** — *"Is the evidence strong and stable enough to act?"* Temporal verification requires
   sustained evidence across consecutive windows (per policy entry, so safety-relevant triggers may act on a single
   high-confidence window).
4. **Response** — *"What should the system do about it?"* A **Context Response Engine** converts a verified trigger
   into coordinated response actions: audio-mode change, notification, priority warning, optional user-confirmed
   quick action, and state management (release timer + restoration).

These four questions — detection / policy / decision / response — are distinct and must never be conflated in
documentation, demos, or code structure. The Context Response Engine is specified in
[§2A Context Response Engine](#2a-context-response-engine-supervisor-response-subsystem).

Operating entirely locally on a host device (simulated via laptop deployment), the framework:

1. Samples ambient audio via a standard, non-ANC wired microphone.
2. Processes input into compact 2D structural arrays (mel-spectrograms).
3. Executes real-time classification inferences on ~200 ms macro-windows (perception).
4. Consults the user's trigger profile (which classes are enabled as triggers) (context/policy).
5. Verifies the candidate trigger temporally (sustained evidence across consecutive windows) (decision).
6. Hands the verified trigger to the Context Response Engine, which routes the configured response actions —
   audio-mode change (Maximum ANC / Transparency / Safety), notification, warning, optional user-confirmed quick
   action — and manages state (trigger release timer → restore preferred mode; duplicate/cooldown suppression).

### 1.1 Conceptual flow (Rev 3 architecture)

Overall flow: **Ambient Audio → Acoustic Event Detection → User Trigger Policy → Confidence + Temporal
Verification → Context Response Engine → Response Actions → State Restoration.**

```
Ambient microphone
   ↓  [PERCEPTION]
Audio preprocessing (raw PCM → mel-spectrogram, ~200 ms windows)
   ↓
Lightweight acoustic event classifier
   ↓
Probability / confidence per supported sound class
   ↓  [CONTEXT / POLICY]
Check user-selected trigger profile (which classes are enabled)
   ↓  [DECISION]
Temporal / event verification (sustained evidence? confidence threshold met?)
   ↓
IF a selected trigger is verified:
   ↓  [RESPONSE]
Context Response Engine
   ├─→ Audio-mode response (Max ANC / Transparency / Safety / Standby)
   ├─→ Notification (e.g., "Siren detected — Transparency mode enabled")
   ├─→ Warning / priority alert (configured high-priority events)
   ├─→ User-confirmed quick action (configured critical events — requires EXPLICIT confirmation)
   └─→ State management: release timer, cooldown / duplicate suppression,
       restore preferred mode when the trigger is absent for a configured period
   ↓
IF a detected sound is NOT an enabled trigger:
   → no response; preferred audio mode continues unchanged
```

### 1.2 Supported trigger classes

The system detects **specific environmental sound events that the user selects as triggers**. Candidate classes
include, but are not limited to:

| Candidate trigger class | Typical relevance |
|---|---|
| Emergency siren | Road / emergency-vehicle awareness |
| Vehicle horn | Traffic awareness |
| Doorbell | Home / delivery awareness |
| Baby or child crying | Home / family awareness |
| Smoke alarm | Home safety awareness |

> **Scope statement (must be honoured in all documentation and demos):** the system does **not** claim to detect
> every possible environmental sound. The user **selects which supported acoustic events are enabled as triggers**
> from the classes the trained model supports. The **final supported class list is not fixed yet** — it is
> finalized in **Phase 1** from dataset investigation (see §5.1).

### 1.3 Prior three-class framing — SUPERSEDED (historical record, retained)

> **STATUS: SUPERSEDED by the user-selectable acoustic trigger architecture (§1.1–§1.2). Retained as the
> historical record of Rev 1.** It must not be reintroduced as a requirement.

| Detected Class | Triggered Mode | Intended Behavior |
|---|---|---|
| **Class A — Continuous Low-Freq Hum** | **Maximum ANC Mode** | Signal detected → trigger Max ANC state (e.g., phase-masking/noise profile) |
| **Class B — Transient Danger Profile** | **Safety Transparency Mode** | Signal detected → trigger immediate transparency / direct mic passthrough |
| **Class C — Low-Amplitude Quiet** | **Low-Power Standby Mode** | Signal detected → scale down DSP processing / reduce sampling to conserve power |

Why it was superseded is recorded in the [Change Log](#11-change-log) entry of 2026-10-05. In short: forcing
unrelated sounds into three overly broad buckets (e.g. lumping a doorbell and an emergency siren into one
"transient danger" class) reduced interpretability, made class balance and dataset sourcing harder, and weakened the
evaluation story. The modes themselves (Maximum ANC / Transparency / Safety / Low-Power Standby) remain valid —
what changed is **what triggers them**.

> **Important clarification (signal vs. analogy):** The masking noise played during a demo in Max ANC Mode is NOT real
> noise cancellation — it is an audible stand-in so a human watching the demo can perceive that a mode-switch decision
> fired. The actual system output is a **discrete decision signal** (e.g., "siren 0.94 and siren is an enabled
> trigger → enter Transparency mode"). On real ANC hardware, this decision would be sent as a command/API call to
> that hardware's own ANC system. The project's contribution is the **classification-and-decision layer**, not audio
> cancellation itself. Do not present masking noise as the project's noise-cancellation mechanism.

### 1.4 Example behaviour (ILLUSTRATIVE — not measured results)

The following examples show how the four stages interact. The probabilities are **illustrative examples only and
are not measured model outputs** — real numbers come only from Phase 3 evaluation.

**Example 1 — enabled, verified trigger (response fires):**

```
[Perception]    Model output:      {siren: 0.94, horn: 0.03, doorbell: 0.01, crying: 0.02}
[Context/Policy] Active profile:   {siren: enabled, horn: enabled, doorbell: disabled, crying: disabled}
[Decision]      Verification:      0.94 ≥ threshold, sustained across N consecutive windows → VERIFIED
[Response]      Response Engine:   audio mode → Transparency
                                   notification → "Siren detected — Transparency mode enabled"
                                   state → release timer armed; preferred mode restored after the
                                   siren is absent for the configured period; cooldown suppresses duplicates
```

**Example 2 — correctly detected but disabled (no response):**

```
[Perception]    Model output:      {doorbell: 0.91, ...}          ← detection is correct
[Context/Policy] Active profile:   {doorbell: disabled}           ← user did not enable it
[Decision]      Verification:      not attempted (policy gate failed first)
[Response]      Response Engine:   NO action; preferred audio mode unchanged
```

A correct detection with a disabled policy entry must produce **no response at all** — this is the practical
value of the detection/policy separation, and it is a required demo case.

---

## 2. Technical & Machine Learning Architecture

The system treats 1D audio sequences as **2D spatial feature profiles**, building on foundations from Andrew Ng's ML
Specialization.

- **Input Spatialization:** Raw time-series signals are converted into Mel-Spectrograms (or MFCCs) using `librosa`.
  This maps acoustic frequency distributions over brief temporal steps (200 ms blocks).
- **Neural Network Design:** A lightweight CNN accepts the 2D feature grids. Feature-extraction layers extract
  acoustic invariants, fully-connected layers map the properties, and a final **Softmax** layer resolves
  classification across the **N supported acoustic event classes** (`N` finalized in Phase 1 — see §5.1).
- **Evaluation Framework:** Optimization tracks **Precision, Recall, and F1-score** per class across edge
  conditions, not just accuracy. **Minimizing False Negatives** for safety-relevant selected triggers is a core
  boundary constraint.
- **No invented targets:** architecture depth/width, class count, and per-class performance targets are finalized
  only after Phase 1 establishes the class set and the dataset. Only the *edge constraints* (short windows, low
  latency, lightweight model, local processing) are fixed in advance.

### 2.1 Four-stage separation (core architectural principle)

The project distinguishes **four** stages. Each answers a different question and lives in a different layer.
Never collapse these into a single "classifier decides" story — in particular, **response is not detection, and
decision is not policy**.

| Stage | Question answered | Responsibility | Changes when the user edits settings? |
|---|---|---|---|
| **Perception — detection** | "What sound is currently present?" | Single lightweight CNN over supported acoustic event classes | No — model is trained once |
| **Context / policy** | "Does it matter to this user?" | Per-class enable flags, active profile (Commuting / Home-Family / Custom), confidence threshold | Yes — instantly, at runtime |
| **Decision — verification** | "Is the evidence strong and stable enough to act?" | Temporal verification across consecutive windows (per-entry rules; single-window allowed for safety-relevant triggers) | Partially — per-entry persistence rules are configuration; the algorithm itself is fixed |
| **Response — Context Response Engine** | "What should the system do?" | Audio-mode change, notification, warning, optional user-confirmed quick action, release timer + restoration, cooldown | Yes — response mapping is configuration |

Example (stages annotated):

```
[Perception]     Model output:      {siren: 0.94, horn: 0.03, doorbell: 0.01, crying: 0.02}
[Context/Policy] User profile:      {siren: enabled, horn: enabled, doorbell: disabled, crying: disabled}
[Decision]       Verification:      sustained evidence + confidence satisfied → VERIFIED
[Response]       System behaviour:  TRIGGER → Context Response Engine executes the configured response
```

If the model detects a **doorbell** under that same profile, the detection may be perfectly correct, but **no
response of any kind occurs** (no mode change, no notification) because the user did not enable the doorbell as a
trigger. This separation is what makes the system practical and customizable.

### 2.2 Do not retrain per profile

**A separate model must NOT be trained every time the user changes their trigger profile.** The project ships
**one model** trained on the supported acoustic event classes; a user-configurable **policy layer** decides which
detected classes cause an action. Changing a profile is a configuration change (no retraining, no re-export).

### 2.3 Trigger profiles (OPTIONAL higher-level demonstrations)

Profiles are **collections of user-selected trigger policies** built on top of the acoustic classifier. They are
**not** model classes and **not** separate models, and they are **optional demonstrations**, not mandatory
requirements.

| Example profile | Enabled triggers (illustrative) |
|---|---|
| **Commuting** | Siren, Vehicle horn |
| **Travel** | Siren, Vehicle horn, Station/announcement-related sounds — *only if a suitable dataset and class can be supported* |
| **Home / Family** | Doorbell, Baby/child crying, Smoke alarm |
| **Custom** | Any user-defined subset of supported classes (profiles are free-form configurations, not a fixed menu) |

> A profile is a **configuration policy only**: it sets which classes are triggers and how the Context Response
> Engine responds to them. Switching profiles is a runtime configuration change — **never** a model change, never
> a retraining event (§2.2).

### 2.4 Machine-Learning Gap Bridging Matrix (from Blueprint)

| Foundation Concept (from Specialization) | Project-Specific Implementation Extension |
|---|---|
| Supervised Learning & Image Inputs | 1D raw audio → `librosa.stft` / `librosa.feature.melspectrogram` → 2D visual matrices |
| Vectorization via NumPy Matrices | Multi-dimensional array ops process real-time buffer blocks; no heavy Python loops |
| Static Tensor Evaluation | Continuous, low-latency streaming pipeline via PyAudio / sounddevice |
| Multi-Class Cross-Entropy Optimization | Softmax classification across the supported environmental event classes |

### 2.5 Why a Supervisor (latency rationale — viva defense)

The model does NOT perform real-time wave inversion/phase cancellation (which would violate physical buffer-latency
limits of ADC/DAC pipelines). Instead, ML performs classification inferences every **200–300 ms**, while the audio
routing path is handled instantaneously by low-level scripts. This hybrid design keeps latencies within real-world
engineering standards.

### 2.6 ML architecture — LOCKED

The machine-learning architecture is **frozen and does not change** as documentation evolves:

| Parameter | Fixed value |
|---|---|
| Model | One lightweight **CNN** (Conv2D/Dense stack, Softmax over N classes) |
| Parameter budget | **< 100K parameters** target (edge-deployable) |
| Inference | CPU / local only, every ~200–300 ms cadence |
| Sample rate | 16 kHz |
| Window | 200 ms, 50% overlap (hop 512 samples) |
| Features | Mel-spectrogram, 64 mel bands, FFT 1024, hop 512 |
| Class count `N` | Finalized in **Phase 1** (§5.1) — not fixed here |

**Do NOT replace the CNN with large pretrained/audio-transformer architectures** (e.g., AudioSet-style AST models
at 86M+ parameters). Such models violate the edge/parameter budget. Hugging Face / pretrained audio models may be
used **only as optional external baselines** for comparison in the report — they are never the project's deployed
model. The four-stage supervisor architecture (§2.1, §2A) is independent of the classifier internals: whatever N
and whatever profile is active, the same one model feeds the same policy → decision → response chain.

---

## 2A. Context Response Engine (supervisor response subsystem)

> Numbered `2A` so that all existing section numbers and cross-references (§3, §5.1, §6, §7, §8, §10, §11) stay
> stable.

The **Context Response Engine** is the fourth stage of the architecture: it consumes a **verified trigger** (from
perception + policy + decision) and produces **coordinated response actions**, then manages the resulting state.
It is pure software logic — no ML inside it; the CNN never decides *what to do*, only *what is present*.

### 2A.1 Response types

| # | Response type | Description | Example |
|---|---|---|---|
| **A** | **Audio-mode response** | Switch to the mode mapped in the active policy entry (Max ANC / Transparency / Safety / Standby / none) | Siren verified → Transparency |
| **B** | **Notification** | Human-readable, non-urgent on-screen/log message accompanying a response | "Siren detected — Transparency mode enabled" |
| **C** | **Warning / priority alert** | Elevated-urgency alert for configured **high-priority** events (e.g., smoke alarm, siren), regardless of whether an audio mode changed | Persistent warning banner + log entry |
| **D** | **User-confirmed quick action** | For configured **critical** events only: surface an action shortcut that executes **only after explicit user confirmation** (e.g., a "call emergency services" shortcut the user must click/tap) | Siren verified → prompt shown → **user confirms** → action runs |
| **E** | **State management** | Release timer (restore preferred mode after trigger absent for a configured period), cooldown / duplicate-alert suppression, response log | Siren ends → configured period later → preferred mode restored, no duplicate alerts |

### 2A.2 Response E — state management details

- **Release timer:** armed when a response fires; when the trigger has been absent for the configured period, the
  preferred (user-default) audio mode is restored.
- **Cooldown / duplicate suppression:** a verified trigger that persists does not re-fire notifications/warnings
  every window; responses are edge-triggered with a configurable cooldown.
- **Restoration is part of the response, not an afterthought** — every mode change must have a defined path back
  to the preferred mode.

### 2A.3 Response D — user-confirmed quick action: HARD BOUNDARY

This is the most safety-sensitive feature; the following rules are **normative**:

1. The quick action **NEVER executes automatically.** It may only be *presented* by the system.
2. Execution requires **explicit, deliberate user confirmation** (a click/tap/keypress on the surfaced action).
   No countdown-to-auto-send, no implicit confirmation, no confirmation-on-behalf-of-user.
3. It applies **only to events the user has explicitly configured as critical** in the active profile. It is
   disabled by default.
4. Documentation and demos MUST state: this is a **UI shortcut prototype**, not a certified emergency system; the
   project makes **no claim of guaranteed emergency detection** (§8) and no claim of reliability sufficient for
   real emergency use.
5. "Automatic emergency calling" is **out of scope and forbidden** as a described capability.

### 2A.4 Notification / warning wording

- Warnings and notifications state what was detected, the active policy decision, and the response taken — they
  must never imply certainty beyond the model (e.g., prefer "Siren detected" as an event report, not "Ambulance
  approaching").
- All displayed probabilities/confidences in the UI are **live model outputs**, and any numbers shown in
  documentation are illustrative examples only (§1.4).

### 2A.5 Supervisor interface (demo-visible state)

The demo makes the supervisor's internal state **visible** in a software interface. The interface MUST show at
least:

- detected event (class) and confidence
- active profile
- trigger enabled / disabled for that event (policy gate result)
- verification state (pending / verified / released)
- current audio mode (+ preferred mode)
- notification / warning state
- available user-confirmed quick action (if configured for this event)
- state-management status (release timer running, cooldown active, restored)

**Demo pipeline:** Speaker/phone → laptop mic → raw PCM → preprocessing → CNN → event probability → user policy →
temporal verification → Context Response Engine → supervisor interface (fields above).

---

## 3. Hardware Simulation Strategy & Device Requirements

100% of this architecture runs locally as an **On-Device Offline Edge AI Node** using consumer hardware. No GPU or
external microcontroller required.

| Component | Requirement |
|---|---|
| Processing Engine | Consumer laptop CPU (Intel Core i5/i7, AMD Ryzen 5/7, or Apple Silicon M-series). No external GPU required. |
| Acoustic Sensor Node | Built-in laptop mic array or standard wired earphone/headphone inline microphone |
| Dynamic Evaluation Rig | A standalone smartphone emitting candidate trigger sounds (sirens, vehicle horns, doorbell, crying, smoke alarm) at controlled distances from the mic during live demos |

### Software Simulation Execution Matrix

The **mode is no longer bound to a class identity**. It is bound to the user's policy entry for the detected
class.

| Target Operational State | Trigger source (revised) | Python Simulation Execution Path |
|---|---|---|
| Maximum ANC | A selected trigger (or ambient-context rule) that the user mapped to Maximum ANC | Trigger a local looping phase-masking audio profile (e.g., optimized brown or pink noise) |
| Safety Transparency | A selected high-priority trigger (e.g., siren, smoke alarm) mapped to Safety Transparency | Open a direct, unbuffered microphone line loop piping live ambient feed to speakers |
| Low-Power Standby | A selected trigger mapped to standby (e.g., sustained quiet-ambient class) | Scale down processing loops and reduce microphone sampling frequency |
| **No action** | Detected class that is **not enabled** in the active profile | Preferred audio mode continues unchanged |

> The matrix above covers **response type A (audio-mode)** only. Notifications, warnings, user-confirmed quick
> actions, and state management (types B–E) are specified in [§2A](#2a-context-response-engine-supervisor-response-subsystem)
> and are also visible in the supervisor interface.

---

## 4. Acoustic Physics: Bypassing Built-in Noise Suppression (**Raw PCM Streams**)

**Problem:** Consumer devices ship with built-in Environmental Noise Cancellation (ENC) and automatic gain control. These
assume distant, low-amplitude, or non-vocal audio is "background garbage" and scrub it out. Distant shouts (50 m) or car
horns (15 m) drop in physical power per the **Inverse-Square Law**, and stock drivers often remove them — which would
blind the neural network.

**Solution:** Request raw PCM capture with a **clean input flag** (via PyAudio/sounddevice) to bypass OS-level filtering
*where the driver allows it*. This is intended to convert the microphone into a truer raw acoustic sensor. If a wave
can physically vibrate the diaphragm, it is captured — making distant horns/shouts more likely to appear as clear
frequency components for AI classification.

> **Honesty constraint:** raw-PCM "bypass" is driver- and platform-dependent and must be **verified and reported
> empirically** in Phase 4 (see §7 Phase 4). Never present it as guaranteed.

---

## 5. Dataset Architecture

### 5.1 Candidate trigger classes and how the final class list is decided

The model is trained over a set of **specific environmental sound events**. The final class list is **NOT fixed
yet**. It is determined during **Phase 1** against the following criteria:

1. Availability of suitable **public datasets**
2. **Audio quality** of the available material
3. **Sufficient sample counts** to train and evaluate reliably
4. **Class balance** across the retained classes
5. **Relevance** to earbud situational awareness
6. Ability to **evaluate the model reliably** (well-defined labels, realistic negatives, measurable metrics)

Priority is given to classes for which **established public datasets already exist**. UrbanSound8K / ESC-50 are
**not assumed to be automatically sufficient** — Phase 1 must investigate multiple public datasets and select the
combination giving the strongest coverage for the chosen trigger classes.

### 5.2 Dataset Requirements (AUTHORITATIVE — revised 2026-10-05)

> These requirements supersede the earlier "~1,000 clips per class with exactly equal final A/B/C counts" rule
> (see §1.3 and the [Change Log](#11-change-log)). The fixed numerical target no longer applies to the revised
> architecture.

1. **Sufficiency requirement:** Each selected trigger class should have **sufficient high-quality, diverse public
   data for reliable training and evaluation**. Classes should be **reasonably balanced where practical**.
2. **Shortage handling:** If one class has insufficient data, **investigate additional appropriate public sources**
   before reducing or compromising dataset quality.
3. **No target-driven dataset inflation:** A dataset is **never added merely to reach a numerical target**. Dataset
   quality, diversity, class balance, provenance, and suitability for the actual trigger task take priority over
   hitting a fixed count.
4. **Provenance:** Keep source dataset, original label, and sample information for every accepted clip (§5.4).
5. **Padding policy:** **Avoid heavy zero-padding of short clips.** Excessive zero-padding makes clips mostly
   artificial silence and distorts class characteristics. Prefer naturally suitable clips/segments and **reject
   unsuitable clips rather than manufacturing large amounts of artificial silence**. Any necessary padding must be
   **limited and documented**.
6. **Real-mic recording:** Not a primary data source; limited to demo / live validation samples (see §8).

### 5.3 Source Datasets (candidates — investigate, do not assume)

| Dataset | Potential relevance | Notes (verify license/access before download) |
|---|---|---|
| **UrbanSound8K** | vehicle horns, sirens, engine idling, machinery | Academic. Fixed label taxonomy; inspect against the candidate trigger list. |
| **ESC-50** | broader ambient sounds: rain, wind, water, hums, quiet ambience | CC-BY-NC (check academic-use terms). |
| **AudioSet-derived resources** | YouTube-tagged "siren", "alarm", "baby crying", "horn" | Large; access/licensing and label noise must be assessed. |
| **Other public environmental-sound datasets** | any suitable/allowed source that improves coverage of a chosen trigger class | Must pass license/access check and be logged under provenance + change log. |

*License/access verification is a prerequisite before downloading or using any source. No dataset is adopted merely
to increase the sample count.*

### 5.4 Provenance & Metadata

For each accepted clip, record (persisted under `data/provenance/`):

- Source dataset & version / release
- Original label / class label in the source
- Sample / filename identifier (and grouping id, to prevent near-duplicate leakage between splits)
- Duration, sample rate, derived acoustic features (RMS/energy, etc.)
- Accepted trigger class (the final class id/name)
- Any filtering, trimming, or limited padding applied (with the parameters used)

---

## 6. Train / Validation / Test Split Validation (MANDATORY GATE)

After dataset curation, use a **70 / 15 / 15 stratified split**, stratified per trigger class.

Before ANY model training, the following MUST be verified for each class:

1. **Proportion check:** Mathematically verify train/val/test have approximately the expected 70/15/15 proportions.
2. **Distribution check:** Verify that the **actual audio characteristics/distributions** of train, validation, and
   test are reasonably similar for the same class (e.g., duration, RMS/energy, and relevant features) — not merely
   similar sample counts.
3. **Leakage check:** Check for duplicates or data leakage between splits (including near-duplicates from the same
   source recording/session).
4. **Visual check:** Generate spectrogram comparisons for the same class across Train / Validation / Test (one class
   → Train vs Val vs Test panel per supported class).
5. **Artifact check:** Detect **excessive silence** or **preprocessing artifacts** introduced by trimming, padding, or
   conversion.

These checks target: excessive silence, bad padding, preprocessing differences, and distribution mismatch.

> **Gate rule:** Do NOT train if a significant split/distribution problem is found. Resolve the issue and document it
> before attempting training.

---

## 7. Phase-Wise Execution Plan

### Phase 1 — Dataset Investigation, Trigger-Class Definition & Class Mapping

**Estimate:** 10–15 hrs

- Investigate **multiple public datasets** (not only UrbanSound8K/ESC-50) and verify license/access conditions.
- Select the **final trigger class list** against the six criteria in §5.1, and record the decision + reasons here.
- Write the source-label → trigger-class mapping; discard irrelevant classes.
- Enforce dataset requirements §5.2 (sufficiency, balance where practical, provenance, limited/documented padding).
- **Hold out ~15–20 real-mic recordings per selected class** (phone-emitted, laptop-mic captured) for later live demo
  validation. These are held out and not used in any training split.
- Stratified 70/15/15 clip-level split (all windows derived from one source clip stay in the same split).
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
- Write `librosa`-based conversion: raw wav → mel-spectrogram → fixed-size 2D array (pad/truncate to constant shape).
  **Note:** padding must respect §5.2 padding policy — prefer naturally suitable segments; avoid excessive
  zero-padding; any necessary padding must be limited and documented.
- Batch-convert the whole dataset; cache as `.npy` arrays to avoid recomputing each training run.
- Visualize a few spectrograms per class to sanity-check separability (e.g., a siren should look distinct from a
  doorbell chime and from a hum).

### Phase 3 — Model Design, Training & Evaluation

**Estimate:** 20–25 hrs

Train a lightweight CNN for the supported acoustic event classes and evaluate it against the false-negative
constraint for safety-relevant triggers.

**Architecture:** final layer width (`N` supported classes), filter counts, and depth are **finalized after Phase 1**,
keeping the project's fixed edge constraints: short windows, low inference latency, lightweight model, local
processing. The original Conv2D/Dense stack from the source blueprint remains the starting candidate and must stay
genuinely edge-deployable (roughly a sub-100K-parameter target).

**Training config:** Adam optimizer with learning-rate scheduling, categorical cross-entropy, batch size ~32, early
stopping on validation loss. **Class weighting** is decided after the class set is known, biasing against false
negatives for safety-relevant selected triggers.

**Evaluation plan:**
- Report accuracy (secondary); **headline metrics are per-class Precision, Recall, F1.**
- Confusion matrix — explicitly inspect the false-negative rate for safety-relevant trigger classes.
- **Do not invent performance targets before the class set and dataset are established** (Phase 1). Any numeric
  accuracy/recall target must be derived from the real dataset and reported honestly.
- Latency benchmark: measure inference time per 200 ms window on the target laptop CPU against the inference budget
  (design budget: single-digit-to-low-tens of ms per window, well inside the 200–300 ms cadence).

### Phase 4 — Live Capture & Raw PCM Pipeline

**Estimate:** 12–15 hrs

- Set up PyAudio/sounddevice stream with a raw input flag, bypassing OS-level ENC/AGC **where the driver allows**.
- Implement a rolling 200 ms buffer with the SAME hop/overlap settings as training preprocessing (mismatch is the
  most common real-world bug).
- Run the held-out real-mic recordings (Phase 1) through the live pipeline first — isolates model bugs from pipeline
  bugs.
- Validate the physics claim empirically: confirm a phone-emitted siren/horn at distance is picked up and detected;
  **document actual vs claimed detection range honestly**.

### Phase 5 — Trigger Policy, Temporal Verification & Context Response Engine

**Estimate:** 8–10 hrs

- Build the **user trigger policy layer**: a per-class enable/disable configuration, per-class confidence threshold,
  and a per-class target mode mapping (Max ANC / Transparency / Safety / Standby / none). **No retraining** when the
  policy changes.
- Implement **temporal verification**: require the candidate class to persist across consecutive windows before
  switching, to avoid flicker on noisy boundary frames. The persistence requirement is **per policy entry**, so
  safety-relevant triggers can require only a single high-confidence window.
- Build the **Context Response Engine** (§2A): response types A–E —
  - audio-mode response per §3 matrix, including the explicit **no-action** case for detected-but-not-enabled
    classes;
  - **notifications** (type B) with the §2A.4 wording rules;
  - **warning / priority alerts** (type C) for configured high-priority events;
  - **user-confirmed quick action** (type D) strictly per the §2A.3 hard boundary — presentation only, explicit
    user confirmation required, disabled by default, no automatic execution;
  - **state management** (type E): release timer + preferred-mode restoration, cooldown / duplicate-alert
    suppression, response log.
- Implement the **release timer**: after a trigger is absent for a configured period, restore the preferred audio
  mode.
- Output is a **decision signal** to be handed to existing ANC hardware (see §1.3 clarification).

### Phase 6 — Demo, Report & Deployment Polish

**Estimate:** 15–20 hrs

- Live demo script: phone plays candidate trigger sounds at set distances; laptop mic + pipeline classifies, applies
  the active user policy, verifies temporally, and drives the Context Response Engine — modes switch visibly,
  notifications/warnings appear, the release timer restores the preferred mode, and the case where a correctly
  detected sound causes **no** change because it is not an enabled trigger is demonstrated.
- Demonstrate the **supervisor interface** (§2A.5) showing the full state: detected event + confidence, active
  profile, trigger enabled/disabled, verification state, current audio mode, notification/warning state,
  available user-confirmed quick action (if configured), and state restoration.
- Demonstrate **response diversity**: at least one audio-mode response (A), one notification (B), one warning (C),
  and the user-confirmed quick action flow (D) presented-but-confirmed-only — plus restoration/cooldown (E).
- Demonstrate **profile switching** (Commuting / Home-Family / Custom) as a **policy switch**, not a model switch.
- GitHub repo: clean README, architecture diagram, honest metrics table (precision/recall/F1 per class), sample
  spectrograms, demo GIF/video.
- Final report: merge blueprint conceptual sections with actual results (real numbers only) and viva defense answers.
- **Do NOT attempt embedded/ARM Cortex-M deployment this semester** — mention only as future work.

---

## 8. Scope Guardrails & Safety / Scope Claims (Fixed Constraints)

- **No physical hardware / PCB build** — laptop simulation only.
- **No embedded / microcontroller deployment** — future work only (mention in report/viva as a portability path via
  TensorFlow Lite / NanoEdge AI Studio for ARM Cortex-M).
- **No custom audio dataset recording as primary data source** — public datasets first; real-mic recording limited
  to demo validation samples only.
- **No claim of guaranteed emergency-sound detection.** This is a BTech prototype, not a certified safety-critical
  product.
- **No claim that ANC is actually being controlled on commercial earbuds** unless real hardware integration exists.
  Today the ANC/Transparency/Safety routing is simulated at software level.
- **No universal sound coverage claim** — only the supported classes can be detected, and only user-enabled
  classes act as triggers.
- **Profiles are policy configurations, not models.** No model is retrained when a profile changes.
- **No automatic emergency action of any kind.** The user-confirmed quick action (§2A.3) is presentation-only and
  requires explicit user confirmation; "automatic emergency calling" is forbidden as a described capability. The
  quick action is a UI-shortcut prototype, disabled by default, never presented as safety-certified or reliable
  enough for real emergency use.
- **Do not frame the project as "merely sound classification."** It is a four-stage supervisor
  (perception → context/policy → decision → response) with a Context Response Engine; the classifier is one
  component of it.
- If the timeline slips, cut Phase 6 depth BEFORE cutting Phase 3 evaluation rigor — a working, honestly-evaluated
  model matters more than a polished GitHub page.

---

## 9. Acoustic Physics & "Edge AI" Viva Defense Points

Reference answers preserved from Blueprint §5:

- **Q1 (Why isn't ML noise cancellation slow?)** — ML is a high-level supervisor on 200–300 ms macro-windows; audio
  routing is done instantly by low-level scripts. Not real-time inversion.
- **Q2 (How is laptop = Edge AI?)** — All compute runs locally on the node where data is collected; offline + private,
  no network dependence. Portable to embedded later.
- **Q3 (How does a cheap wired mic capture distant danger?)** — Software noise-gates/driver filters, not the capsule, are
  the limitation. Raw PCM capture attempts to bypass them; any wave that vibrates the diaphragm can be captured. (Verify
  driver behaviour empirically; do not overclaim.)
- **Q4 (Why isn't this a basic filter project?)** — Static linear filters (Wiener/spectral subtraction) fail on
  non-stationary/transient noises. This uses a non-linear CNN acoustic signature classifier + a host behaviour change.
- **Q5 (Why user-selectable triggers instead of fixed classes?)** — Real users care about different events in different
  situations. Separating *detection* from *trigger policy* means one trained model serves commuting, travel, and
  home/family needs, and the system only acts on events the user actually cares about. It also improves
  interpretability: each class is a specific, nameable acoustic event rather than a broad bucket.
- **Q6 (Why not one model per profile?)** — Retraining per profile is impractical on-device and unnecessary. A single
  multi-class model plus a configurable policy layer gives instant customization with one training run.
- **Q7 (Isn't this just sound classification?)** — The classifier only answers *perception* ("what sound is
  present?"). The project is a four-stage **supervisor**: policy decides *whether it matters to this user*,
  temporal verification decides *whether the evidence is strong enough to act*, and the Context Response Engine
  decides *what to do* — audio-mode change, notification, warning, user-confirmed quick action — and manages state
  (restoration, cooldown). None of those stages live inside the model.

### Production path (practicality answer)

The system is a **decision layer** over existing ANC hardware (AirPods/Sony/Bose etc.). On real devices it would call
the device's existing ANC/transparency control API instead of running masking-noise/mic-passthrough scripts. Pipeline:
**classify → check user policy → verify → Context Response Engine → command existing hardware (plus notification /
warning / user-confirmed action)** — directly portable without building ANC hardware.

---

## 10. Git / GitHub Workflow — Standing Rule

GitHub serves as the project's **development history and backup**, not just a final-upload target.

1. **Checkpoint reminders.** Before moving between meaningful milestones, check whether current work should be
   committed/pushed. Remind at checkpoints such as: completing a phase; completing a major part of a phase; finishing
   a significant verified-and-working implementation; completing an important validation/checkpoint; before a major
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
| 2026-10-07 | **Dataset acquisition phase executed — 17/18 project classes now have audio; `Smoke_Fire_Alarm` marked UNRESOLVED.** New sources: **FSD50K** (`Fhrozen/FSD50k`, 4,027 clips — 733 Alarm + 3,294 across Vehicle_Horn/Doorbell/Train/Aircraft/Motorcycle/Help_Shouting/Glass_Breaking/Gunshot/Knocking/Footsteps), **ESC-50** (306 clips of 9 categories, CC-BY-NC-3.0), **owlgebra-ai/babycry** (500 Baby_Crying clips from one parquet shard: DonateACry all labels except `burping` + ReCANVo `distress` rows of dysregulated/frustrated labels; ESC-50-source rows excluded as duplicates; per-row licenses ODbL/CC-BY-4.0 preserved), UrbanSound8K (5,732, pre-existing). Conservative ESC-50 mappings fixed: `crying_baby→Baby_Crying`, `clock_alarm→Alarm`, `door_wood_knock→Knocking` (an earlier `Alarm_Beep`/`Child_Crying`/`Knock` mapping had target classes outside the taxonomy). **Zenodo 3689288** (142 MB OSR/FSL domestic alarm corpus) downloaded but NOT used: its 24 alarm classes are anonymized (`pattern_01..24`) with no documented fire-alarm/doorbell mapping — zip retained unused. Leakage guards: FSD50K `fname` vs US8K `fsID`; ESC-50 `src_file` vs US8K `fsID` + FSD50K `source_id` (54 ESC-50 clips dropped); cross-dataset duplicate SHA-256 removed (4). Final master manifest `data/manifests/master_audio_manifest.csv`: **10,565 clips, all validation `ok`, 0 duplicate hashes, 16.3 audio hours** — Aircraft 338, Alarm 755, Baby_Crying 540, Car_Engine 1000, Dog_Bark 1000, Doorbell 53, Drilling 1000, Footsteps 585, Glass_Breaking 515, Gunshot 793, Help_Shouting 498, Jackhammer 1000, Knocking 387, Motorcycle 238, Siren 929, Train 468, Vehicle_Horn 466, **Smoke_Fire_Alarm 0 (UNRESOLVED — no conservative public source found; FSD50K `Fire` = crackling fire, not alarms)**. Doorbell (53) and Motorcycle (238) are below the ~500 sufficiency guide; additional sources may be investigated before Phase 1 training per the flexible sufficiency rule. Disk: ~3.7 GB. | Living spec standing rule 1: dataset strategy, mappings, and gaps recorded as decided. | Accepted |
| 2026-10-07 | **MAJOR — architecture expanded from a two-layer classifier+trigger scheme into a four-stage "Complete Context-Aware Acoustic Supervisor": Perception (detection) → Context/Policy (does it matter to this user?) → Decision (confidence + temporal verification) → Response.** Flow restated as: Ambient Audio → Acoustic Event Detection → User Trigger Policy → Confidence + Temporal Verification → Context Response Engine → Response Actions → State Restoration. New normative section **§2A Context Response Engine** defines response types **A** audio-mode, **B** notification ("Siren detected — Transparency mode enabled"), **C** warning/priority alert for configured high-priority events, **D** user-confirmed quick action for configured critical events (PRESENTATION ONLY — explicit user confirmation required, disabled by default, never automatic, no guaranteed emergency detection, "automatic emergency calling" forbidden), **E** state management (release timer, preferred-mode restoration, cooldown/duplicate suppression). §2.1 rewritten as the four-stage separation table (detection / policy / decision / response). Profiles documented as **configuration policies** (added **Custom** example) — never models; profile switch never requires retraining. §1.4 illustrative example behaviour added (probabilities labelled illustrative, not measured). §2.6 **ML architecture LOCKED**: CNN, <100K parameters, 16 kHz, 200 ms windows, 50% overlap, 64 mel, FFT 1024, hop 512, N finalized Phase 1 — explicitly NOT replaced by large audio transformers (e.g., AST 86M+); Hugging Face models only as optional external baselines. Supervisor interface + demo pipeline added (§2A.5); Phase 5 retitled and extended with response types A–E; Phase 6 demo updated (supervisor interface, response diversity, profile switching); §8 adds no-automatic-emergency-action and not-merely-classification guardrails; §9 adds Q7. **Documentation-only change** (README + blueprint): no source code, no datasets, no training, no implementation, no results, no commits. Section numbering of existing sections preserved (new section inserted as 2A) so all existing cross-references (§3, §5.1, §6, §7, §8, §10, §11) remain valid. | User-directed concept expansion: the deliverable is a complete context-aware acoustic supervisor with a response subsystem, not just an event classifier. Clarifies the decision/response boundary, makes the emergency-action safety boundary explicit, and freezes the ML architecture so documentation growth cannot drift the model design. | Accepted |
| 2026-10-05 | **MAJOR — three broad acoustic classes SUPERSEDED by a user-selectable acoustic trigger architecture.** The fixed 3-class scheme (A continuous hum / B transient danger / C quiet) is replaced by detection of *specific environmental sound events* (candidates: emergency siren, vehicle horn, doorbell, baby/child crying, smoke alarm) that the **user selects as triggers**. New two-layer separation is normative: **(1) model detection** = "what sound is present?" and **(2) user trigger policy** = "which detected sounds should cause an ANC/mode change?" Added: per-class confidence thresholds, temporal/event verification, per-trigger release timer, explicit no-action path for detected-but-not-enabled classes, and a hard rule that **changing a profile must not require retraining** (one model + configurable policy layer). Trigger profiles (Commuting / Travel / Home-Family) are documented as **optional** policy demonstrations, not model classes. Final class list is **not fixed** and is determined in Phase 1 against six criteria (§5.1). Dataset requirement replaced: the "exactly ~1,000 per class / equal A/B/C counts" rule is superseded by a **flexible sufficiency requirement** — each selected trigger class needs sufficient high-quality, diverse public data; classes reasonably balanced where practical; if a class is short, additional appropriate public sources are investigated before reducing or compromising quality; datasets are **never added merely to reach a numerical target**. §7 Phase 1 redefined as multi-dataset investigation + trigger-class selection; Phase 3/5 reframed around N classes and the policy layer; §6 gate generalized from A/B/C to per-class and extended with an explicit artifact/silence check; §8 adds safety/scope claims (no guaranteed emergency detection, no claim of controlling ANC on commercial earbuds, no universal sound coverage). Rev 1 three-class framing retained verbatim as superseded historical record in §1.3. Design/prior-art reference added: "Context-Aware Adjustment of Noise Cancellation for Wearable Audio Devices" (Maheshwari et al., Technical Disclosure Commons, 27 Oct 2025) — used as a **design reference only**, not reproduced; it provides no dataset, model, or results for this project. | The revised architecture (a) better matches the context-aware ANC use case, (b) allows use of established environmental sound datasets instead of forcing them into arbitrary buckets, (c) avoids forcing unrelated sounds into overly broad classes, (d) enables user-specific trigger policies, (e) should improve interpretability and potentially classification reliability, and (f) gives the project a clearer, more defensible evaluation strategy. | Accepted |
| 2026-09-07 | **Git/GitHub standing workflow established (§10):** checkpoint-based commit reminders (advisory, never automatic); explicit user request required to commit/push; pre-phase completeness + consistency check; correct-mistakes-over-preserve-history policy; cleanliness checks before major pushes; genuine-history principle. | User governance instruction. Defines how the project's development history and backup are maintained on GitHub throughout the project. | Accepted |
| 2026-09-07 | **Standing auto-update rule established:** every project-level change (source, dataset strategy, architecture, phases, preprocessing, model, evaluation, or requirements) is recorded in this document automatically at the time the change is made or discovered — no reminder required. Applies to all future sessions. | User governance instruction. Makes the change-log discipline self-executing so the spec never drifts from the implementation. | Accepted |
| 2026-09-07 | **Final dataset targets set to ~1,000 clips/class (from 300–500), with equal final A/B/C counts; shortage-handling order defined; padding policy tightened (avoid heavy zero-padding); provenance requirement added.** | User specification update. Encoded as authoritative §5.2. | **Superseded (2026-10-05)** — fixed 1,000/class and exact equality replaced by the flexible sufficiency requirement (§5.2); shortage-handling, padding and provenance rules retained in spirit and strengthened. |
| 2026-09-07 | **Mandatory pre-training split-validation gate added (§6): proportion check, per-class audio-distribution similarity, leakage check, and Train/Val/Test spectrogram comparison, with a do-not-train rule on failure.** | User specification update to guarantee split quality and catch bad padding/silence/distribution mismatch before training. | Accepted (generalized to per-class + artifact check, 2026-10-05) |
| 2026-09-07 | **Initial scoped change — "modify the plan when necessary using additional appropriate public data/sources" is explicitly permitted, but every such use must be documented (provenance §5.4 + this change log), never silently.** | User specification. Sets the rule for legitimate future deviation from the original Phase Plan's dataset scope. | Accepted |

*Future entries: whenever we use additional public data sources, finalize the trigger class list, adjust class mapping,
alter hyperparameters, or change any requirement, record it here with the reason.*

---

## 12. References

1. **Design / prior-art reference (not reproduced):** Shailesh Maheshwari, Vedant Maheshwari, Rayansh Maheshwari.
   *"Context-Aware Adjustment of Noise Cancellation for Wearable Audio Devices."* Technical Disclosure Commons,
   27 October 2025. <https://www.tdcommons.org/cgi/viewcontent.cgi?article=10034&context=dpubs_series>
   — Cited as inspiration for the context-aware ANC-adjustment problem setting. The project does **not** claim to
   reproduce its implementation, dataset, model, or experimental results.
2. Andrew Ng — Machine Learning Specialization (foundations for the supervised-learning / image-input approach).
3. Companion sources (initial, now superseded by this living document): `Edge_AI_Earbud_Project_Blueprint .pdf`,
   `Edge_AI_Earbud_Phase_Plan.pdf`.

---

*End of living specification. Rev 3.*