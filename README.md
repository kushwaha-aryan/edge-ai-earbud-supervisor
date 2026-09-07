# Edge AI Earbud Supervisor

**Edge AI Earbud Supervisor** — Smart Context-Aware Acoustic Supervisor Framework.

A lightweight, locally-running Edge AI system that samples ambient audio from a
standard wired microphone, converts short windows into 2D mel-spectrogram grids,
and runs a compact Convolutional Neural Network (CNN) to classify three acoustic
contexts. Based on the acoustic class detected, the supervisor dynamically routes
audio output to simulate three operational states of next-generation earphones:

| Detected Class | Triggered Mode |
|---|---|
| Class A — Continuous Low-Frequency Hum | Maximum ANC Mode |
| Class B — Transient Danger Profile | Safety Transparency Mode |
| Class C — Low-Amplitude Quiet | Low-Power Standby Mode |

The AI acts as a **high-level supervisor / decision layer** that sits on top of
existing ANC hardware. It classifies macro-windows (~200 ms) to detect context
changes, while the actual audio path is handled by low-level scripts. It does not
perform real-time wave inversion.

## Project Status

**Work in progress — currently scaffolding only.** No phase has been implemented yet.
The project will be built strictly phase-by-phase:

1. **Phase 1 — Data Sourcing & Class Mapping** (`data/`, `src/data/`)
2. **Phase 2 — Signal Processing Pipeline** (`src/preprocessing/`)
3. **Phase 3 — Model Design, Training & Evaluation** (`src/models/`, `src/training/`)
4. **Phase 4 — Live Capture & Raw PCM Pipeline** (`src/inference/`)
5. **Phase 5 — Mode Routing Logic** (`src/routing/`)
6. **Phase 6 — Demo, Report & Deployment Polish**

## Repository Layout

```
data/               Raw + processed datasets (gitignored, not yet downloaded)
  raw/              Downloaded dataset archives
  processed/        Cached .npy spectrogram arrays (Phase 2)
  splits/           Stratified train/val/test split metadata (Phase 1)
  provenance/       Per-clip source/label/metadata records (Phase 1)
src/                All source code
  config.py         Single source of truth for paths, hyperparameters, classes
  data/             Phase 1 - dataset download, filtering, class mapping, splitting
  preprocessing/    Phase 2 - audio -> mel-spectrogram conversion
  models/           Phase 3 - CNN architecture definition
  training/         Phase 3 - training loop, evaluation, metrics
  inference/        Phase 4 - live mic capture + real-time classification
  routing/          Phase 5 - mode switching with debounce logic
notebooks/          Jupyter notebooks for exploration & visualization
scripts/            CLI entry points to run phases end-to-end
tests/              Unit tests
results/            Saved models, metrics, plots (gitignored)
docs/               Additional documentation (design notes, guilds, etc.)
```

## Source Documents

- **`FINAL_PROJECT_BLUEPRINT.md`** — the project's **living specification** and single source of truth.
  Consolidates and supersedes the two PDFs; records every change with a reason.
- `Edge_AI_Earbud_Project_Blueprint .pdf` — original architecture & viva defense guide (initial source)
- `Edge_AI_Earbud_Phase_Plan.pdf` — original phase-wise implementation roadmap (initial source)

## Scope Guardrails

- Laptop-only simulation; no physical hardware / PCB build.
- No embedded / microcontroller deployment this semester (future work only).
- Training data from public datasets (UrbanSound8K, ESC-50) first; real-mic
  recording limited to demo validation samples only.