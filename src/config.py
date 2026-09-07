"""Global configuration and constants for the Edge AI Earbud Supervisor.

Single source of truth for project paths, hyperparameters, and class labels.
Referenced by all other modules. Keep this file consistent with the
FINAL_PROJECT_BLUEPRINT.md living specification.
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
# Class mapping (Phase 1)
# ---------------------------------------------------------------------------
CLASS_LABELS = {
    0: "Class A - Continuous Low-Freq Hum",
    1: "Class B - Transient Danger Profile",
    2: "Class C - Low-Amplitude Quiet",
}
CLASS_INDICES = {v: k for k, v in CLASS_LABELS.items()}

# Authoritative dataset targets (FINAL_PROJECT_BLUEPRINT.md §5.2)
TARGET_SAMPLES_PER_CLASS = 1000      # aim ~1000 good-quality clips per class
FINAL_CLASS_COUNTS_EQUAL = True      # final A/B/C counts must be equal

# Stratified split ratios (blueprint: 70/15/15)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Padding policy (blueprint §5.2): avoid heavy zero-padding of short clips
MAX_ZERO_PAD_FRACTION = 0.20         # reject clips whose padding would exceed this

# ---------------------------------------------------------------------------
# Model architecture (Phase 3)
# ---------------------------------------------------------------------------
CONV1_FILTERS = 16
CONV2_FILTERS = 32
CONV3_FILTERS = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.3
NUM_CLASSES = 3

# Training configuration
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
CLASS_WEIGHTS = {0: 1.0, 1: 2.0, 2: 1.0}   # weight Class B higher to bias vs false negatives

# Evaluation targets
TARGET_CLASS_B_RECALL = 0.90
LATENCY_BUDGET_MS = 50
INFERENCE_CADENCE_MS = 200                 # match WINDOW_SIZE_MS

# ---------------------------------------------------------------------------
# Live capture (Phase 4) / routing (Phase 5)
# ---------------------------------------------------------------------------
MAX_DEVICE_CHANNELS = 1
ROLLING_BUFFER_WINDOWS = 1
DEBOUNCE_CONFIRM_WINDOWS = 3               # same class for N consecutive windows
NO_DEBOUNCE_DANGER_THRESHOLD = 0.80        # high-confidence trigger for Class B

# 3-mode output behaviors (Phase 5, per Blueprint Section 3)
MODE_MAX_ANC = "Maximum ANC Mode"          # brown/pink noise mask playback
MODE_SAFETY = "Safety Transparency Mode"   # direct unbuffered mic passthrough
MODE_STANDBY = "Low-Power Standby Mode"    # reduced sampling rate loop