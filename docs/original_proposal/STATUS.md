# Original Proposal Documents — ARCHIVED / SUPERSEDED

## Status

The two documents in this directory are the **original project proposal (Rev 1)**, written before dataset work
began. They are **archived as the historical record only** and are **superseded** by the project's living
specification:

> **`FINAL_PROJECT_BLUEPRINT.md` (repository root) is the single source of truth — Rev 3.**

They must not be cited at the viva or in the report as describing the current system. All current requirements,
architecture, phases, decisions, and measured results are in the blueprint and its Change Log (§11), which records
every deviation (with reasons and dates).

## Files

| File | Content |
|---|---|
| `Edge_AI_Earbud_Project_Blueprint .pdf` | Original architecture & viva defense guide (Rev 1): fixed **3-class** scheme (Class A hum → Max ANC / Class B transient danger → Safety Transparency / Class C quiet → Standby), raw-PCM capture intent, balanced-dataset rule (~1,000/class, equal A/B/C). |
| `Edge_AI_Earbud_Phase_Plan.pdf` | Original phase roadmap: 3-class CNN (`Dense(3, Softmax)`), epochs 30–50 / early-stop patience 5, **"Recall on Class B ≥ 90%"** target, ~50 ms per-window latency target, 70/15/15 split, ~90–100 h effort estimate. |

## What these documents do NOT contain

- No 17-class taxonomy, no preset experiment definitions, no measured results of any kind.
- Their architecture, class scheme, performance targets, and effort estimates are **superseded**, not measured.

## How each superseded claim maps to the current system (see blueprint)

| Original claim (PDF) | Current position (blueprint / code) |
|---|---|
| Fixed 3 classes A/B/C | Superseded 2026-10-05 → **17-class user-selectable trigger taxonomy**, ratified D8 (config.py `TRIGGER_CLASS_NAMES`); Rev-1 framing retained only as history in §1.3. |
| `Dense(3, Softmax)` head | Current CNN: Conv2D 16/32/64 → Dense 32 → Dropout 0.3 → **`Dense(17, Softmax)`**, 56,657 params; `--pool-freq-only` presets use a GlobalAveragePooling2D head (25,937 params). |
| Recall on Class B ≥ 90% | No per-class targets are invented (§7 Phase 3); metrics come from the actual locked split. |
| Latency "under 50 ms" | Design target only (`config.LATENCY_BUDGET_MS=50`); **not yet measured** — do not claim as measured. |
| Raw PCM / "clean input flag" bypass of OS filters | Proposed design intent; implementation (`scripts/demo_yamnet.py --mic`) uses `sounddevice` default input without such a bypass; **presence/value is unverified** — Phase 4 pending. |
| Epochs 30–50 / patience 5 | `config.MAX_EPOCHS=50`, `EARLY_STOPPING_PATIENCE=5`, driver default 4; **saved models used ≤ a handful of epochs** — do not claim "trained 30–50 epochs". |
| Microcontroller portability | Explicitly **future work only** (blueprint §8 / README guardrails). |

## Measured results (correct attribution)

Measured validation results live in `scripts/results/` and are summarized in the blueprint §11 Change Log. Note the
two distinct Phase 3A models — do **not** conflate them:

- `data/models/baseline_A.keras` — preset-A defaults: window acc **0.4243** / macro-F1 **0.4048** (74,830 windows); clip mean-softmax 0.4994 / 0.4689 / 0.5193.
- `data/models/lightweight_cnn.keras` — byte-identical to `archive/weighted_epoch1.keras` (epoch-1 weighted checkpoint): window acc **0.3804** / macro-F1 **0.3861**; clip mean-softmax 0.5414 / 0.5113 / 0.5321.

All figures above are **validation-set** results; the test split has never been opened (Phase 3B pending).

_This archive was created and the blueprint Change Log updated on 2026-10-10._