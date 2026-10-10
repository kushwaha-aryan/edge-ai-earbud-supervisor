# Phase 3A Clip-Level Evaluation (analysis only)

Model: `data\models\presetD_freqpool.keras` (360,514 bytes, 25,937 parameters). Analysis preset `D`: 500 ms window, 50% overlap, FFT 1024, hop 512, 64 Mel -> (64, 14, 1) by the extraction formula. Frozen Phase 2B clip-level split, window and clip levels reported side by side. No retraining; model, preprocessing, configuration and dataset unchanged. The test split is never opened.

- validation windows: **28,470**
- validation clips: **1,337**
- clips with zero windows: **0**
- top-k size for clip_topk_mean: **5**
- evaluated on the validation clips that have windows in preset `E` (identical clip set across compared runs)

## Aggregation definitions

- window: one prediction per window (no aggregation).
- clip_mean_softmax: mean over the clip's softmax probability vectors, then argmax.
- clip_topk_mean: per class, the mean softmax probability across that class's --top-k highest-probability windows in the clip (k = min(--top-k, windows in the clip)), then argmax over classes.
- clip_majority_vote: majority vote of per-window argmax; ties are broken by the higher mean softmax over the clip.

## Overall metrics (window and clip levels)

| metric | window | clip mean-softmax | clip top-k mean | clip majority vote |
|---|---|---|---|---|
| samples | 28,470 | 28,470 | 1,337 | 1,337 | 1,337 |
| correct | 13,247 | 13,247 | 743 | 731 | 732 |
| accuracy | 0.4653 | 0.4653 | 0.5557 | 0.5467 | 0.5475 |
| macro precision | 0.4515 | 0.4515 | 0.5461 | 0.5327 | 0.5462 |
| macro recall | 0.4144 | 0.4144 | 0.4934 | 0.4765 | 0.4852 |
| macro F1 | 0.4203 | 0.4203 | 0.4927 | 0.4748 | 0.4876 |
| weighted precision | 0.4632 | 0.4632 | 0.6211 | 0.5928 | 0.6115 |
| weighted recall | 0.4653 | 0.4653 | 0.5557 | 0.5467 | 0.5475 |
| weighted F1 | 0.4551 | 0.4551 | 0.5608 | 0.5412 | 0.5524 |

## Per-class metrics - window

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 2029 | 0.4107 | 0.5190 | 0.4585 |
| 1 | Alarm | 3310 | 0.4327 | 0.4456 | 0.4391 |
| 2 | Baby_Crying | 1169 | 0.8619 | 0.7151 | 0.7817 |
| 3 | Car_Engine | 2122 | 0.4525 | 0.3073 | 0.3660 |
| 4 | Dog_Bark | 1725 | 0.4697 | 0.4354 | 0.4519 |
| 5 | Doorbell | 171 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 2162 | 0.5075 | 0.5809 | 0.5417 |
| 7 | Footsteps | 2353 | 0.4532 | 0.6791 | 0.5436 |
| 8 | Glass_Breaking | 768 | 0.5248 | 0.2760 | 0.3618 |
| 9 | Gunshot | 935 | 0.5292 | 0.3390 | 0.4133 |
| 10 | Help_Shouting | 1377 | 0.2741 | 0.2847 | 0.2793 |
| 11 | Jackhammer | 1909 | 0.5659 | 0.7831 | 0.6570 |
| 12 | Knocking | 743 | 0.6882 | 0.3297 | 0.4459 |
| 13 | Motorcycle | 1515 | 0.4430 | 0.3485 | 0.3901 |
| 14 | Siren | 2012 | 0.6167 | 0.6158 | 0.6163 |
| 15 | Train | 3406 | 0.3261 | 0.3417 | 0.3338 |
| 16 | Vehicle_Horn | 764 | 0.1193 | 0.0445 | 0.0648 |

## Per-class metrics - clip_mean_softmax

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 50 | 0.3415 | 0.5600 | 0.4242 |
| 1 | Alarm | 103 | 0.3526 | 0.5922 | 0.4420 |
| 2 | Baby_Crying | 55 | 0.8605 | 0.6727 | 0.7551 |
| 3 | Car_Engine | 143 | 0.7541 | 0.3217 | 0.4510 |
| 4 | Dog_Bark | 123 | 0.8026 | 0.4959 | 0.6131 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 150 | 0.6383 | 0.6000 | 0.6186 |
| 7 | Footsteps | 79 | 0.4656 | 0.7722 | 0.5810 |
| 8 | Glass_Breaking | 48 | 0.7600 | 0.3958 | 0.5205 |
| 9 | Gunshot | 56 | 0.8421 | 0.5714 | 0.6809 |
| 10 | Help_Shouting | 54 | 0.2727 | 0.4444 | 0.3380 |
| 11 | Jackhammer | 133 | 0.7832 | 0.8421 | 0.8116 |
| 12 | Knocking | 45 | 0.8148 | 0.4889 | 0.6111 |
| 13 | Motorcycle | 35 | 0.3636 | 0.4571 | 0.4051 |
| 14 | Siren | 137 | 0.8264 | 0.7299 | 0.7752 |
| 15 | Train | 79 | 0.2391 | 0.4177 | 0.3041 |
| 16 | Vehicle_Horn | 39 | 0.1667 | 0.0256 | 0.0444 |

## Per-class metrics - clip_topk_mean

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 50 | 0.3371 | 0.6000 | 0.4317 |
| 1 | Alarm | 103 | 0.3497 | 0.6214 | 0.4476 |
| 2 | Baby_Crying | 55 | 0.8222 | 0.6727 | 0.7400 |
| 3 | Car_Engine | 143 | 0.7015 | 0.3287 | 0.4476 |
| 4 | Dog_Bark | 123 | 0.8082 | 0.4797 | 0.6020 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 150 | 0.6414 | 0.6200 | 0.6305 |
| 7 | Footsteps | 79 | 0.4437 | 0.7975 | 0.5701 |
| 8 | Glass_Breaking | 48 | 0.8095 | 0.3542 | 0.4928 |
| 9 | Gunshot | 56 | 0.7250 | 0.5179 | 0.6042 |
| 10 | Help_Shouting | 54 | 0.3067 | 0.4259 | 0.3566 |
| 11 | Jackhammer | 133 | 0.6648 | 0.8797 | 0.7573 |
| 12 | Knocking | 45 | 0.8400 | 0.4667 | 0.6000 |
| 13 | Motorcycle | 35 | 0.3529 | 0.3429 | 0.3478 |
| 14 | Siren | 137 | 0.7464 | 0.7518 | 0.7491 |
| 15 | Train | 79 | 0.1733 | 0.1646 | 0.1688 |
| 16 | Vehicle_Horn | 39 | 0.3333 | 0.0769 | 0.1250 |

## Per-class metrics - clip_majority_vote

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 50 | 0.3537 | 0.5800 | 0.4394 |
| 1 | Alarm | 103 | 0.3220 | 0.5534 | 0.4071 |
| 2 | Baby_Crying | 55 | 0.8837 | 0.6909 | 0.7755 |
| 3 | Car_Engine | 143 | 0.7419 | 0.3217 | 0.4488 |
| 4 | Dog_Bark | 123 | 0.7262 | 0.4959 | 0.5894 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 150 | 0.6319 | 0.6067 | 0.6190 |
| 7 | Footsteps | 79 | 0.4485 | 0.7722 | 0.5674 |
| 8 | Glass_Breaking | 48 | 0.7500 | 0.3750 | 0.5000 |
| 9 | Gunshot | 56 | 0.8571 | 0.5357 | 0.6593 |
| 10 | Help_Shouting | 54 | 0.2532 | 0.3704 | 0.3008 |
| 11 | Jackhammer | 133 | 0.7740 | 0.8496 | 0.8100 |
| 12 | Knocking | 45 | 0.8148 | 0.4889 | 0.6111 |
| 13 | Motorcycle | 35 | 0.3902 | 0.4571 | 0.4211 |
| 14 | Siren | 137 | 0.8151 | 0.7080 | 0.7578 |
| 15 | Train | 79 | 0.2366 | 0.3924 | 0.2952 |
| 16 | Vehicle_Horn | 39 | 0.2857 | 0.0513 | 0.0870 |

## Confusion matrix - window (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **1053** | 24 | 0 | 120 | 81 | 0 | 40 | 130 | 0 | 19 | 3 | 142 | 6 | 103 | 27 | 280 | 1 |
| Alarm | 282 | **1475** | 1 | 61 | 147 | 0 | 167 | 253 | 17 | 8 | 102 | 77 | 13 | 47 | 279 | 311 | 70 |
| Baby_Crying | 0 | 185 | **836** | 0 | 47 | 0 | 13 | 13 | 7 | 0 | 59 | 0 | 0 | 0 | 1 | 2 | 6 |
| Car_Engine | 89 | 0 | 121 | **652** | 4 | 0 | 33 | 2 | 0 | 9 | 332 | 33 | 0 | 267 | 52 | 528 | 0 |
| Dog_Bark | 10 | 144 | 0 | 69 | **751** | 0 | 91 | 87 | 14 | 3 | 189 | 33 | 3 | 4 | 134 | 190 | 3 |
| Doorbell | 0 | 109 | 0 | 3 | 1 | **0** | 8 | 47 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| Drilling | 192 | 63 | 0 | 2 | 6 | 0 | **1256** | 36 | 1 | 7 | 128 | 246 | 0 | 13 | 26 | 131 | 55 |
| Footsteps | 51 | 32 | 1 | 20 | 134 | 0 | 19 | **1598** | 29 | 48 | 11 | 71 | 31 | 28 | 89 | 188 | 3 |
| Glass_Breaking | 9 | 44 | 0 | 5 | 11 | 0 | 82 | 213 | **212** | 39 | 3 | 57 | 3 | 6 | 2 | 82 | 0 |
| Gunshot | 277 | 17 | 0 | 0 | 12 | 0 | 19 | 181 | 41 | **317** | 3 | 6 | 36 | 1 | 4 | 21 | 0 |
| Help_Shouting | 5 | 383 | 2 | 38 | 77 | 0 | 131 | 65 | 28 | 9 | **392** | 7 | 8 | 18 | 60 | 127 | 27 |
| Jackhammer | 0 | 0 | 0 | 38 | 0 | 0 | 138 | 188 | 0 | 0 | 0 | **1495** | 0 | 20 | 0 | 30 | 0 |
| Knocking | 22 | 13 | 0 | 1 | 73 | 0 | 0 | 249 | 19 | 70 | 4 | 30 | **245** | 3 | 0 | 14 | 0 |
| Motorcycle | 82 | 62 | 4 | 202 | 68 | 0 | 49 | 127 | 1 | 9 | 2 | 110 | 5 | **528** | 9 | 257 | 0 |
| Siren | 37 | 420 | 0 | 0 | 90 | 0 | 4 | 0 | 0 | 4 | 67 | 0 | 4 | 2 | **1239** | 63 | 82 |
| Train | 447 | 73 | 5 | 202 | 86 | 0 | 387 | 303 | 34 | 57 | 134 | 321 | 2 | 137 | 50 | **1164** | 4 |
| Vehicle_Horn | 8 | 365 | 0 | 28 | 11 | 0 | 38 | 34 | 0 | 0 | 1 | 14 | 0 | 15 | 36 | 180 | **34** |

## Confusion matrix - clip_mean_softmax (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **28** | 0 | 0 | 2 | 2 | 0 | 1 | 3 | 0 | 0 | 0 | 3 | 0 | 2 | 0 | 9 | 0 |
| Alarm | 6 | **61** | 0 | 1 | 4 | 0 | 5 | 5 | 0 | 0 | 6 | 1 | 0 | 2 | 3 | 8 | 1 |
| Baby_Crying | 0 | 13 | **37** | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 7 | 0 | 6 | **46** | 0 | 0 | 2 | 0 | 0 | 0 | 25 | 0 | 0 | 18 | 2 | 37 | 0 |
| Dog_Bark | 0 | 12 | 0 | 5 | **61** | 0 | 8 | 3 | 0 | 0 | 11 | 1 | 0 | 0 | 8 | 14 | 0 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 14 | 5 | 0 | 0 | 0 | 0 | **90** | 0 | 0 | 0 | 11 | 13 | 0 | 0 | 1 | 13 | 3 |
| Footsteps | 1 | 0 | 0 | 0 | 2 | 0 | 0 | **61** | 2 | 2 | 0 | 2 | 3 | 1 | 3 | 2 | 0 |
| Glass_Breaking | 1 | 4 | 0 | 0 | 1 | 0 | 6 | 12 | **19** | 1 | 0 | 2 | 0 | 0 | 0 | 2 | 0 |
| Gunshot | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 2 | **32** | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| Help_Shouting | 0 | 19 | 0 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | **24** | 0 | 0 | 0 | 1 | 4 | 1 |
| Jackhammer | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 10 | 0 | 0 | 0 | **112** | 0 | 1 | 0 | 0 | 0 |
| Knocking | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 17 | 1 | 3 | 0 | 1 | **22** | 0 | 0 | 0 | 0 |
| Motorcycle | 2 | 0 | 0 | 4 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 0 | **16** | 0 | 7 | 0 |
| Siren | 4 | 28 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | **100** | 1 | 0 |
| Train | 9 | 2 | 0 | 2 | 1 | 0 | 11 | 6 | 1 | 0 | 5 | 5 | 0 | 3 | 1 | **33** | 0 |
| Vehicle_Horn | 0 | 24 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 2 | 8 | **1** |

## Confusion matrix - clip_topk_mean (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **30** | 0 | 0 | 3 | 2 | 0 | 1 | 3 | 0 | 1 | 0 | 7 | 0 | 1 | 0 | 2 | 0 |
| Alarm | 8 | **64** | 0 | 2 | 3 | 0 | 4 | 5 | 0 | 0 | 3 | 2 | 0 | 0 | 7 | 3 | 2 |
| Baby_Crying | 0 | 14 | **37** | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 4 | 0 | 8 | **47** | 0 | 0 | 2 | 0 | 0 | 0 | 23 | 3 | 0 | 18 | 5 | 33 | 0 |
| Dog_Bark | 1 | 12 | 0 | 4 | **59** | 0 | 6 | 5 | 0 | 0 | 11 | 4 | 0 | 0 | 11 | 10 | 0 |
| Doorbell | 0 | 6 | 0 | 0 | 0 | **0** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 13 | 5 | 0 | 0 | 0 | 0 | **93** | 0 | 0 | 0 | 10 | 16 | 0 | 0 | 3 | 7 | 3 |
| Footsteps | 2 | 0 | 0 | 0 | 2 | 0 | 0 | **63** | 1 | 2 | 0 | 2 | 3 | 0 | 3 | 1 | 0 |
| Glass_Breaking | 0 | 3 | 0 | 0 | 1 | 0 | 8 | 13 | **17** | 1 | 0 | 4 | 0 | 0 | 0 | 1 | 0 |
| Gunshot | 12 | 1 | 0 | 0 | 1 | 0 | 1 | 10 | 1 | **29** | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Help_Shouting | 0 | 20 | 0 | 1 | 0 | 0 | 4 | 1 | 0 | 0 | **23** | 1 | 0 | 0 | 3 | 0 | 1 |
| Jackhammer | 0 | 0 | 0 | 0 | 0 | 0 | 7 | 9 | 0 | 0 | 0 | **117** | 0 | 0 | 0 | 0 | 0 |
| Knocking | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18 | 1 | 4 | 0 | 1 | **21** | 0 | 0 | 0 | 0 |
| Motorcycle | 1 | 1 | 0 | 5 | 1 | 0 | 3 | 4 | 0 | 0 | 0 | 7 | 0 | **12** | 0 | 1 | 0 |
| Siren | 4 | 29 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **103** | 0 | 0 |
| Train | 14 | 4 | 0 | 4 | 2 | 0 | 12 | 8 | 1 | 3 | 4 | 11 | 0 | 2 | 1 | **13** | 0 |
| Vehicle_Horn | 0 | 24 | 0 | 1 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 2 | 4 | **3** |

## Confusion matrix - clip_majority_vote (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **29** | 0 | 0 | 2 | 3 | 0 | 2 | 2 | 0 | 0 | 0 | 2 | 0 | 2 | 0 | 8 | 0 |
| Alarm | 7 | **57** | 0 | 1 | 5 | 0 | 5 | 7 | 0 | 0 | 4 | 2 | 0 | 1 | 5 | 8 | 1 |
| Baby_Crying | 0 | 14 | **38** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 7 | 0 | 5 | **46** | 0 | 0 | 2 | 0 | 0 | 0 | 26 | 0 | 0 | 18 | 1 | 38 | 0 |
| Dog_Bark | 0 | 12 | 0 | 5 | **61** | 0 | 8 | 2 | 0 | 0 | 10 | 1 | 0 | 0 | 8 | 16 | 0 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 14 | 5 | 0 | 0 | 0 | 0 | **91** | 0 | 0 | 0 | 13 | 14 | 0 | 0 | 1 | 9 | 3 |
| Footsteps | 1 | 0 | 0 | 0 | 2 | 0 | 0 | **61** | 2 | 2 | 0 | 2 | 3 | 1 | 2 | 3 | 0 |
| Glass_Breaking | 0 | 5 | 0 | 0 | 1 | 0 | 6 | 12 | **18** | 1 | 0 | 3 | 0 | 0 | 0 | 2 | 0 |
| Gunshot | 10 | 0 | 0 | 0 | 1 | 0 | 0 | 11 | 2 | **30** | 0 | 0 | 2 | 0 | 0 | 0 | 0 |
| Help_Shouting | 0 | 20 | 0 | 0 | 1 | 0 | 4 | 4 | 0 | 0 | **20** | 0 | 0 | 0 | 1 | 3 | 1 |
| Jackhammer | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 10 | 0 | 0 | 0 | **113** | 0 | 0 | 0 | 0 | 0 |
| Knocking | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 17 | 1 | 2 | 0 | 1 | **22** | 0 | 0 | 0 | 0 |
| Motorcycle | 1 | 0 | 0 | 4 | 2 | 0 | 1 | 1 | 0 | 0 | 0 | 3 | 0 | **16** | 0 | 7 | 0 |
| Siren | 4 | 31 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **97** | 1 | 0 |
| Train | 9 | 3 | 0 | 3 | 1 | 0 | 13 | 6 | 1 | 0 | 4 | 5 | 0 | 2 | 1 | **31** | 0 |
| Vehicle_Horn | 0 | 25 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 3 | 5 | **2** |

## Top 15 clip-level confusions (mean-softmax)

largest off-diagonal cells of the clip_mean_softmax matrix.

| true | predicted | clips |
|---|---|---:|
| Car_Engine | Train | 37 |
| Siren | Alarm | 28 |
| Car_Engine | Help_Shouting | 25 |
| Vehicle_Horn | Alarm | 24 |
| Help_Shouting | Alarm | 19 |
| Car_Engine | Motorcycle | 18 |
| Knocking | Footsteps | 17 |
| Dog_Bark | Train | 14 |
| Drilling | Aircraft | 14 |
| Baby_Crying | Alarm | 13 |
| Drilling | Jackhammer | 13 |
| Drilling | Train | 13 |
| Dog_Bark | Alarm | 12 |
| Glass_Breaking | Footsteps | 12 |
| Dog_Bark | Help_Shouting | 11 |

## Baby_Crying clip recall by sample-rate group (mean-softmax)

ledger sample_rate is the original clip sample rate; clips with no prediction (zero-window clips) are excluded.

| sample-rate group | clips (total) | clips (assessed) | correct | recall |
|---:|---:|---:|---:|---:|
| below 16000 | 32 | 32 | 32 | 1.0000 |
| 16000 and above | 23 | 23 | 5 | 0.2174 |

## Misclassified clips - within vs cross dominant source_dataset (mean-softmax)

A class's dominant source_dataset is the dataset with the most clips of that class in the ledger. An error is within-dataset when the predicted class has the same dominant source_dataset as the true class, and cross-dataset otherwise.

- misclassified clips: **594**
- within-dataset errors: **242**
- cross-dataset errors: **352**
- errors with no dominant dataset for either class: **0**

| true class | predicted class | kind | clips |
|---|---|---:|---:|
| Car_Engine | Train | cross | 37 |
| Siren | Alarm | cross | 28 |
| Car_Engine | Help_Shouting | cross | 25 |
| Vehicle_Horn | Alarm | cross | 24 |
| Help_Shouting | Alarm | within | 19 |
| Car_Engine | Motorcycle | cross | 18 |
| Knocking | Footsteps | within | 17 |
| Dog_Bark | Train | cross | 14 |
| Drilling | Aircraft | cross | 14 |
| Baby_Crying | Alarm | cross | 13 |
| Drilling | Jackhammer | within | 13 |
| Drilling | Train | cross | 13 |
| Glass_Breaking | Footsteps | within | 12 |
| Dog_Bark | Alarm | cross | 12 |
| Train | Drilling | cross | 11 |
| Dog_Bark | Help_Shouting | cross | 11 |
| Drilling | Help_Shouting | cross | 11 |
| Gunshot | Footsteps | within | 10 |
| Gunshot | Aircraft | within | 10 |
| Jackhammer | Drilling | within | 10 |
| Jackhammer | Footsteps | cross | 10 |
| Train | Aircraft | within | 9 |
| Aircraft | Train | within | 9 |
| Vehicle_Horn | Train | cross | 8 |
| Alarm | Train | within | 8 |
| Dog_Bark | Siren | within | 8 |
| Dog_Bark | Drilling | within | 8 |
| Motorcycle | Train | within | 7 |
| Car_Engine | Aircraft | cross | 7 |
| Glass_Breaking | Drilling | cross | 6 |
| Alarm | Help_Shouting | within | 6 |
| Train | Footsteps | within | 6 |
| Alarm | Aircraft | within | 6 |
| Car_Engine | Baby_Crying | cross | 6 |
| Alarm | Drilling | cross | 5 |
| Train | Jackhammer | cross | 5 |
| Alarm | Footsteps | within | 5 |
| Train | Help_Shouting | within | 5 |
| Doorbell | Alarm | within | 5 |
| Dog_Bark | Car_Engine | within | 5 |
| Drilling | Alarm | cross | 5 |
| Help_Shouting | Train | within | 4 |
| Alarm | Dog_Bark | cross | 4 |
| Glass_Breaking | Alarm | within | 4 |
| Help_Shouting | Drilling | cross | 4 |
| Motorcycle | Car_Engine | cross | 4 |
| Siren | Aircraft | cross | 4 |
| Aircraft | Jackhammer | cross | 3 |
| Baby_Crying | Help_Shouting | cross | 3 |
| Footsteps | Knocking | within | 3 |
| Motorcycle | Jackhammer | cross | 3 |
| Alarm | Siren | cross | 3 |
| Knocking | Gunshot | within | 3 |
| Footsteps | Siren | cross | 3 |
| Aircraft | Footsteps | within | 3 |
| Train | Motorcycle | within | 3 |
| Dog_Bark | Footsteps | cross | 3 |
| Siren | Help_Shouting | cross | 3 |
| Drilling | Vehicle_Horn | within | 3 |
| Footsteps | Dog_Bark | cross | 2 |
| Aircraft | Dog_Bark | cross | 2 |
| Footsteps | Gunshot | within | 2 |
| Aircraft | Motorcycle | within | 2 |
| Motorcycle | Dog_Bark | cross | 2 |
| Train | Alarm | within | 2 |
| Gunshot | Glass_Breaking | within | 2 |
| Footsteps | Jackhammer | cross | 2 |
| Footsteps | Glass_Breaking | within | 2 |
| Gunshot | Knocking | within | 2 |
| Vehicle_Horn | Siren | within | 2 |
| Train | Car_Engine | cross | 2 |
| Alarm | Motorcycle | within | 2 |
| Aircraft | Car_Engine | cross | 2 |
| Vehicle_Horn | Drilling | within | 2 |
| Glass_Breaking | Train | within | 2 |
| Glass_Breaking | Jackhammer | cross | 2 |
| Doorbell | Footsteps | within | 2 |
| Footsteps | Train | within | 2 |
| Motorcycle | Aircraft | within | 2 |
| Car_Engine | Drilling | within | 2 |
| Car_Engine | Siren | within | 2 |
| Vehicle_Horn | Car_Engine | within | 1 |
| Baby_Crying | Dog_Bark | cross | 1 |
| Alarm | Jackhammer | cross | 1 |
| Help_Shouting | Siren | cross | 1 |
| Alarm | Vehicle_Horn | cross | 1 |
| Train | Dog_Bark | cross | 1 |
| Glass_Breaking | Dog_Bark | cross | 1 |
| Footsteps | Motorcycle | within | 1 |
| Aircraft | Drilling | cross | 1 |
| Motorcycle | Drilling | cross | 1 |
| Knocking | Glass_Breaking | within | 1 |
| Knocking | Dog_Bark | cross | 1 |
| Alarm | Car_Engine | cross | 1 |
| Footsteps | Aircraft | within | 1 |
| Glass_Breaking | Aircraft | within | 1 |
| Help_Shouting | Footsteps | within | 1 |
| Glass_Breaking | Gunshot | within | 1 |
| Vehicle_Horn | Motorcycle | cross | 1 |
| Knocking | Jackhammer | cross | 1 |
| Help_Shouting | Vehicle_Horn | cross | 1 |
| Train | Siren | cross | 1 |
| Doorbell | Drilling | cross | 1 |
| Train | Glass_Breaking | within | 1 |
| Siren | Dog_Bark | within | 1 |
| Siren | Train | cross | 1 |
| Dog_Bark | Jackhammer | within | 1 |
| Jackhammer | Motorcycle | cross | 1 |
| Drilling | Siren | within | 1 |
| Baby_Crying | Footsteps | cross | 1 |

## Per-class clip recall by source_dataset (mean-softmax)

classes with more than one source_dataset in the validation split.

| class | source_dataset | clips | correct | recall |
|---|---|---:|---:|---:|
| Aircraft | ESC-50 | 13 | 7 | 0.5385 |
| Aircraft | FSD50K | 37 | 21 | 0.5676 |
| Alarm | ESC-50 | 3 | 2 | 0.6667 |
| Alarm | FSD50K | 100 | 59 | 0.5900 |
| Baby_Crying | ESC-50 | 17 | 0 | 0.0000 |
| Baby_Crying | owlgebra_babycry | 38 | 37 | 0.9737 |
| Footsteps | ESC-50 | 6 | 5 | 0.8333 |
| Footsteps | FSD50K | 73 | 56 | 0.7671 |
| Glass_Breaking | ESC-50 | 4 | 2 | 0.5000 |
| Glass_Breaking | FSD50K | 44 | 17 | 0.3864 |
| Gunshot | FSD50K | 33 | 15 | 0.4545 |
| Gunshot | UrbanSound8K | 23 | 17 | 0.7391 |
| Knocking | ESC-50 | 4 | 1 | 0.2500 |
| Knocking | FSD50K | 41 | 21 | 0.5122 |
| Train | ESC-50 | 6 | 4 | 0.6667 |
| Train | FSD50K | 73 | 29 | 0.3973 |
| Vehicle_Horn | ESC-50 | 3 | 0 | 0.0000 |
| Vehicle_Horn | FSD50K | 11 | 1 | 0.0909 |
| Vehicle_Horn | UrbanSound8K | 25 | 0 | 0.0000 |

## Clips with zero windows

No validation clip has zero windows.

model.predict: 2.1 s