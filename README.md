# Edge AI Earbud Supervisor — Smart Context-Aware Acoustic Supervisor

A lightweight, **locally-running** Edge AI system that samples ambient audio from a standard microphone, converts
short windows into 2D mel-spectrogram grids, and runs a compact Convolutional Neural Network (CNN) to **detect specific
environmental sound events**. The classifier is only the first stage: the system is specified as four stages —
**Perception → Context/Policy → Decision → Response** — ending in a **Context Response Engine** that turns a verified
trigger into coordinated actions (audio-mode change, notification, warning, optional user-confirmed quick action, and
state restoration).

The AI acts as a **high-level supervisor / decision layer** that sits on top of existing ANC hardware. It classifies
macro-windows (~200 ms) to detect context changes, while the actual audio path is handled by low-level scripts. It does
**not** perform real-time wave inversion.

---

## 1. Problem

Active Noise Cancellation (ANC) improves listening comfort in steady background noise — but it can also **reduce
awareness of important environmental sounds**. A user on a busy street may need to hear an approaching emergency siren
or a vehicle horn; a parent may need to hear a doorbell or a child crying. Because ANC reduces everything uniformly,
the user has to remove the earbud or disable ANC manually — which is exactly the moment awareness matters most.

The core problem is therefore not "remove noise better". It is: **decide, locally and quickly, when the audio scene
contains a sound the user actually cares about right now.**

## 2. Proposed Solution

A lightweight local acoustic-event detection + response system that allows users to **select which supported
environmental sounds should cause a response — and what that response should be**.

Four stages are kept deliberately separate:

| Stage | Question it answers |
|---|---|
| **Perception — detection** | "What sound is currently present?" |
| **Context / policy** | "Does it matter to this user right now?" (which classes are enabled as triggers) |
| **Decision — verification** | "Is the evidence strong and stable enough to act?" (confidence + temporal rules) |
| **Response — Context Response Engine** | "What should the system do?" (audio mode, notification, warning, quick action, restoration) |

This separation is what makes the system practical: the model does the hard audio work once, the user decides what
matters, verification prevents flicker, and the response engine handles the actions — all instantly, without
retraining anything.

## 3. Core Architecture

Overall flow: **Ambient Audio → Detection → User Trigger Policy → Confidence + Temporal Verification → Context
Response Engine → Response Actions → State Restoration.**

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
Temporal / event verification (sustained evidence? threshold met?)
   ↓
IF verified:
   ↓  [RESPONSE]
Context Response Engine
   ├─→ Audio mode (Max ANC / Transparency / Safety / Standby)
   ├─→ Notification ("Siren detected — Transparency mode enabled")
   ├─→ Warning / priority alert (configured high-priority events)
   ├─→ User-confirmed quick action (configured critical events — user must confirm)
   └─→ State management: release timer, cooldown, restore preferred mode
   ↓
IF detected but NOT an enabled trigger → no response; preferred mode unchanged
```

The AI sits on top of existing ANC hardware and outputs a **decision signal**. On real ANC hardware this decision would
be sent as a command to that hardware's own ANC system; in this project the mode change is simulated at software level.

**ML architecture (fixed):** one lightweight CNN, **<100K parameters** target, 16 kHz input, 200 ms windows with 50%
overlap, 64 mel bands, FFT 1024, hop 512, CPU/local inference. Large pretrained audio transformers (e.g., AST at
86M+ params) are **not** the project model; Hugging Face models may appear only as optional external baselines.

## 4. Context Response Engine

The fourth stage: pure software logic that consumes a **verified trigger** and produces coordinated response
actions. The CNN never decides what to do — only what is present.

| # | Response type | Description |
|---|---|---|
| **A** | Audio-mode response | Switch to the policy-mapped mode (Max ANC / Transparency / Safety / Standby / none) |
| **B** | Notification | Human-readable message accompanying the response (e.g., "Siren detected — Transparency mode enabled") |
| **C** | Warning / priority alert | Elevated-urgency alert for configured high-priority events (e.g., smoke alarm) |
| **D** | User-confirmed quick action | For configured **critical** events: surfaces an action shortcut (e.g., "call emergency services") that runs **only after explicit user confirmation** |
| **E** | State management | Release timer → restore preferred mode after the trigger disappears; cooldown / duplicate-alert suppression; response log |

**Safety boundary (normative):** the quick action **never executes automatically** — the system may only *present*
it, and the user must explicitly confirm. It is disabled by default, applies only to user-configured critical
events, and is a UI-shortcut prototype: **no claim of guaranteed emergency detection**, and "automatic emergency
calling" is out of scope.

**Supervisor interface (demo):** the demo makes the pipeline visible — detected event + confidence, active profile,
trigger enabled/disabled, verification state, current audio mode, notification/warning state, available
user-confirmed quick action, and state restoration.

Demo pipeline: Speaker/phone → laptop mic → raw PCM → preprocessing → CNN → event probability → user policy →
temporal verification → Context Response Engine → supervisor interface.

## 5. Example Trigger Classes

Candidate classes include:

| Trigger class | Typical relevance |
|---|---|
| Emergency siren | Road / emergency-vehicle awareness |
| Vehicle horn | Traffic awareness |
| Doorbell | Home / delivery awareness |
| Baby or child crying | Home / family awareness |
| Smoke alarm | Home safety awareness |

**The final supported class list is not fixed yet.** It will be determined after dataset investigation in Phase 1,
based on availability of suitable public datasets, audio quality, sufficient sample counts, class balance, relevance to
earbud situational awareness, and the ability to evaluate the model reliably.

The system does **not** claim to detect every possible environmental sound. Users enable only the supported acoustic
events they want as triggers.

## 6. User Customization

Users can create or select **profiles** — simple policy configurations that enable the trigger classes relevant to
their current situation, and configure how the Context Response Engine responds to them:

| Example profile | Enabled triggers (illustrative) |
|---|---|
| **Commuting** | Siren, Vehicle horn |
| **Travel** | Siren, Vehicle horn, station/announcement-related sounds *(only if a suitable dataset and class can be supported)* |
| **Home / Family** | Doorbell, Baby/child crying, Smoke alarm |
| **Custom** | Any user-defined subset of the supported classes |

Example behaviour (probabilities are illustrative examples, **not measured results**):

```
[Perception]     Model output:   {siren: 0.94, horn: 0.03, doorbell: 0.01, crying: 0.02}
[Context/Policy] User profile:   {siren: enabled, horn: enabled, doorbell: disabled, crying: disabled}
[Decision]       Verification:   threshold met, sustained across windows → VERIFIED
[Response]       Result:         TRIGGER — audio mode switches, notification shown,
                                 release timer armed, preferred mode restored later
```

If the model correctly detects a **doorbell** under that same profile, **no response of any kind occurs** (no mode
change, no notification), because the user did not enable the doorbell as a trigger — a required demo case.

**Architectural principle: profiles are policy configurations, not separate ML models.** One model is trained on the
supported acoustic event classes; switching or editing a profile is an instant configuration change. **No model is
retrained when the user changes their trigger profile.**

## 7. Edge-AI Motivation

Local inference is intended to **reduce dependence on cloud processing** and to provide **low-latency responses**:

- All compute runs locally on the device where audio is captured — **offline and private**, no network round-trip.
- Response latency is bounded by the inference cadence (~200–300 ms), which matters when the user may be missing an
  approaching vehicle or an alarm.
- A **lightweight model** and **short audio windows** keep the system viable on consumer hardware (and potentially
  portable to embedded targets later).
- Audio never needs to leave the device.

## 8. Project Status

**Current state (2026-10-09):** dataset acquisition, validation, preprocessing, and the initial Phase 3A model
training + validation analysis are EXECUTED. The Phase 3A controlled experiments (preset A / preset D training runs),
the YAMNet embedding-baseline fit, and all test-split (Phase 3B) evaluation are prepared but NOT yet run - they are
gated on explicit user authorization.

In phase order, what is actually in place:

1. **Phase 1 - Dataset Investigation, Trigger-Class Definition & Class Mapping** (`data/`, `src/data/`): 10,565
   curated clips across 17 classes acquired (UrbanSound8K, FSD50K, ESC-50, owlgebra-ai/babycry), mappings fixed and
   provenance manifest built; full read-only validation verdict **PASS WITH WARNINGS**; `Smoke_Fire_Alarm`
   unresolved (0 conservative public clips found).
2. **Phase 2 - Signal Processing Pipeline** (`src/preprocessing/`): EXECUTED - 10,281 kept clips (284 excluded per
   audit flags), 15 s truncation, mono 16 kHz, 200 ms / 50%-overlap mel windows (64,5,1) scaled with train-frozen
   dB bounds, source-group-aware covariate-balanced 70/15/15 split (train 7,021 / val 1,618 / test 1,642), all D11
   leakage gates PASS; Phase 2B distribution gate **PASS WITH WARNINGS** (warnings documented, not blocking).
3. **Phase 3 - Model Design, Training & Evaluation** (`src/models/`, `src/training/`): locked 56,657-param CNN
   trained (2026-10-08, saved `data/models/lightweight_cnn.keras`) and analysed on validation: val accuracy 0.4243,
   macro-F1 0.4048 (74,830 windows); clip majority-vote mean-softmax accuracy 0.5414 / macro-F1 0.5113 /
   weighted-F1 0.5321 (window macro-F1 0.3861 / accuracy 0.3804). Analysis presets B/C/D approved; preset D
   train+val features built (train 127,398 / val 29,363 windows). YAMNet frozen-embedding baseline scaffolded and
   its extract executed (8,639 clips, 87,985 frames, 0 failures) - experimental baseline only, **NOT adopted as the
   perception layer**.

Splits must pass the mandatory validation gate (proportion + per-class distribution + leakage + spectrogram +
artifact checks) before any training - Phase 2A/2B gates currently PASS. Phases 4-6 not started.

Before any training, dataset splits must pass a mandatory validation gate: proportion check, per-class distribution
similarity, duplicate/leakage check, representative spectrogram comparison across splits, and detection of excessive
silence or preprocessing artifacts.

## 9. Research / Reference Basis

The concept is informed by the following publication, used as a **design reference / prior-art reference** for the
context-aware noise-cancellation adjustment problem setting:

> Shailesh Maheshwari, Vedant Maheshwari, Rayansh Maheshwari.
> *"Context-Aware Adjustment of Noise Cancellation for Wearable Audio Devices."*
> Technical Disclosure Commons, 27 October 2025.
> <https://www.tdcommons.org/cgi/viewcontent.cgi?article=10034&context=dpubs_series>

This project **does not reproduce or reimplement** that work, and does not use its dataset, trained model, or
experimental results. Any implementation here is the project's own work.

The project is also a BTech prototype, not a certified safety-critical product: it does not guarantee detection of
emergency sounds, and it does not claim to control ANC on commercial earbuds unless real hardware integration exists.

---

## Repository Layout

```
data/               Raw + processed datasets (gitignored)
  raw/              Downloaded dataset archives (validated copies)
  processed/        Cached .npy mel-spectrogram arrays (Phase 2, preset A)
  processed_preset_D/  Preset D train+val mel arrays (Phase 3A controlled experiment)
  embeddings/       YAMNet frozen-embedding cache (Phase 3A baseline)
  splits/           Stratified train/val/test split metadata (Phase 1)
  provenance/       Per-clip source/label/metadata records (Phase 1)
src/                All source code
  config.py         Single source of truth for paths, hyperparameters
  data/             Phase 1 - dataset investigation, filtering, class mapping, splitting
  preprocessing/    Phase 2 - audio -> mel-spectrogram conversion
  models/           Phase 3 - CNN architecture definition
  training/         Phase 3 - training loop, evaluation, metrics
  inference/        Phase 4 - live mic capture + real-time classification
  routing/          Phase 5 - trigger policy, temporal verification, Context Response Engine
notebooks/          Jupyter notebooks for exploration & visualization
scripts/            CLI entry points to run phases end-to-end
tests/              Unit tests
results/            Saved models, metrics, plots (gitignored)
docs/               Additional documentation (design notes, guides, etc.)
```

## Source Documents

- **`FINAL_PROJECT_BLUEPRINT.md`** — the project's **living specification** and single source of truth.
  Consolidates and supersedes the two PDFs; records every change with a reason.
- `Edge_AI_Earbud_Project_Blueprint .pdf` — original architecture & viva defense guide (initial source)
- `Edge_AI_Earbud_Phase_Plan.pdf` — original phase-wise implementation roadmap (initial source)

## Scope Guardrails

- Laptop-only simulation; no physical hardware / PCB build; no custom earbud hardware development.
- Software-level ANC/Transparency/Safety mode routing; no claim of controlling ANC on commercial earbuds.
- **No automatic emergency action**: the quick action is presentation-only and requires explicit user confirmation;
  no guaranteed emergency detection; not a certified safety-critical product.
- The project is a four-stage **supervisor**, not merely a sound classifier.
- No embedded / microcontroller deployment this semester (future work only).
- Training data from suitable public datasets (UrbanSound8K, ESC-50, AudioSet-derived, and other appropriate public
  sources as justified) — chosen for quality, diversity, balance, provenance and task suitability, not to hit a fixed
  count. Real-mic recording is limited to demo/live validation samples only.
- No cloud dependency as the primary path; local inference is the project goal.