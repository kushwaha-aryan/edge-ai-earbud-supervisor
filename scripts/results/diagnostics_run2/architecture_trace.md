# Architecture time-axis trace (build_cnn)

## Source references

| code reference | file:line |
| --- | --- |
| build_cnn definition | scripts/run_phase3a_training.py:150 |
| pool layers | scripts/run_phase3a_training.py:161, scripts/run_phase3a_training.py:165 |

## Layer trace

| layer | output shape | time frames | params |
| --- | --- | --- | --- |
| input mel_window | (64, 5, 1) | 5 | 0 |
| Conv2D 16 3x3 same relu | (64, 5, 16) | 5 | 160 |
| MaxPooling2D 2x2 | (32, 2, 16) | 2 | 0 |
| Conv2D 32 3x3 same relu | (32, 2, 32) | 2 | 4,640 |
| MaxPooling2D 2x2 | (16, 1, 32) | 1 | 0 |
| Conv2D 64 3x3 same relu | (16, 1, 64) | 1 | 18,496 |
| Flatten | (1024,) | 1 | 0 |
| Dense 32 relu | (32,) | 1 | 32,800 |
| Dropout 0.3 | (32,) | 1 | 0 |
| Dense 17 softmax | (17,) | 1 | 561 |

## Time-axis summary

- frames through the network: 5 -> 5 -> 2 -> 2 -> 1 -> 1 -> 1 -> 1 -> 1 -> 1
- the 5-frame time axis collapses to 1 frame at the second pool (64x5 -> 32x2 -> 16x1); flatten sees a single frame per window (1024 values), so temporal evolution inside the 200 ms window is discarded before the dense layers.
- total parameters: 56,657 (under the 100,000 parameter ceiling)
- input from config.SPEC_SHAPE = (64, 5, 1), filters 16/32/64, dense 32, classes 17
