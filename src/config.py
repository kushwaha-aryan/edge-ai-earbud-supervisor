"""Global configuration and constants for the Edge AI Earbud Supervisor.

Single source of truth for project paths and fixed hyperparameters.
Referenced by all other modules. Keep this file consistent with the
FINAL_PROJECT_BLUEPRINT.md living specification (Rev 2).

Rev 2 note: the original fixed 3-class scheme (Class A continuous hum /
Class B transient danger / Class C quiet) and the "exactly ~1,000 clips per
class with equal final counts" dataset rule are SUPERSEDED by the
user-selectable acoustic trigger architecture. See blueprint §1.3
(historical record) and §5.2 (authoritative dataset requirements).
The trigger class list and all per-class constants are therefore finalized
in Phase 1 and are intentionally left unset here until then.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLIT_DATA_DIR = DATA_DIR / "splits"
PROVENANCE_DIR = DATA_DIR / "provenance"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

# ---------------------------------------------------------------------------
# Audio signal parameters (Phase 2 fixed parameters)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000                # Hz, standard for speech/environmental audio
WINDOW_SIZE_MS = 200                # analysis block length
HOP_LENGTH = 512                    # 50% overlap at 16kHz
N_FFT = 1024                        # FFT window size
N_MELS = 64                         # mel bands
SPEC_SHAPE = (N_MELS, 63, 1)        # fixed 2D feature grid, padded/truncated

# ---------------------------------------------------------------------------
# Trigger class mapping (Phase 1)
# ---------------------------------------------------------------------------
# The final supported trigger class list is NOT fixed yet. It is selected in
# Phase 1 from dataset investigation against the six criteria in blueprint
# §5.1. Populate these from src/data/class_map.py once Phase 1 decides the
# class set; do not hard-code a class list before then.
TRIGGER_CLASS_NAMES: tuple[str, ...] = ()   # finalized in Phase 1
TRIGGER_CLASS_TO_ID: dict[str, int] = {}    # finalized in Phase 1
NUM_TRIGGER_CLASSES: int | None = None      # finalized in Phase 1

# Dataset requirements (blueprint §5.2, revised 2026-10-05):
#   - each selected trigger class needs sufficient high-quality, diverse public
#     data for reliable training and evaluation
#   - classes reasonably balanced where practical
#   - if a class is short, additional appropriate public sources are
#     investigated before reducing or compromising quality
#   - a dataset is never added merely to reach a numerical target
# There is deliberately no fixed per-class sample-count constant here.
MAX_ZERO_PAD_FRACTION: float | None = None  # limited-padding threshold, decided in Phase 1

# Stratified split ratios (blueprint §6)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Model architecture (Phase 3) — final shape decided after Phase 1
# ---------------------------------------------------------------------------
# These are the original starting-candidate values from the source blueprint.
# The output layer width must equal NUM_TRIGGER_CLASSES, which is not fixed yet,
# and per-class weighting / recall targets are decided only after the class set
# and dataset are established (blueprint §7 Phase 3).
CONV1_FILTERS = 16
CONV2_FILTERS = 32
CONV3_FILTERS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.3

# Training configuration
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
CLASS_WEIGHTS: dict[int, float] | None = None   # decided in Phase 1/3

# Evaluation targets
LATENCY_BUDGET_MS = 50                            # inference budget per window (design target)
INFERENCE_CADENCE_MS = 200                        # match WINDOW_SIZE_MS
# No per-class accuracy/recall targets are defined here: they must be derived
# from the real dataset after Phase 1 (blueprint §7 Phase 3).

# ---------------------------------------------------------------------------
# Live capture (Phase 4) / trigger policy + routing (Phase 5)
# ---------------------------------------------------------------------------
MAX_DEVICE_CHANNELS = 1
ROLLING_BUFFER_WINDOWS = 1
# Temporal verification: a candidate trigger must persist for this many
# consecutive windows before a mode switch (blueprint §7 Phase 5). Safety-
# relevant triggers may instead require a single high-confidence window.
TRIGGER_CONFIRM_WINDOWS = 3
TRIGGER_RELEASE_SECONDS = 5                       # restore preferred mode after trigger absent
TRIGGER_MIN_CONFIDENCE = 0.80                     # per-policy default threshold

# Audio-mode outputs (simulated; see blueprint §3). Modes remain valid; what
# triggers them is the user's policy, not a fixed class identity.
MODE_MAX_ANC = "Maximum ANC Mode"          # brown/pink noise mask playback
MODE_SAFETY = "Safety Transparency Mode"   # direct unbuffered mic passthrough
MODE_STANDBY = "Low-Power Standby Mode"    # reduced sampling rate loop