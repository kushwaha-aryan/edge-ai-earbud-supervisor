# Phase 3A Clip-Level Evaluation (analysis only)

Model: `data\models\presetE_freqpool.keras` (360,514 bytes, 25,937 parameters). Analysis preset `E`: 2000 ms window, 50% overlap, FFT 1024, hop 512, 64 Mel -> (64, 61, 1) by the extraction formula. Frozen Phase 2B clip-level split, window and clip levels reported side by side. No retraining; model, preprocessing, configuration and dataset unchanged. The test split is never opened.

- validation windows: **6,229**
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
| samples | 6,229 | 6,229 | 1,337 | 1,337 | 1,337 |
| correct | 2,850 | 2,850 | 648 | 648 | 645 |
| accuracy | 0.4575 | 0.4575 | 0.4847 | 0.4847 | 0.4824 |
| macro precision | 0.5094 | 0.5094 | 0.5612 | 0.5573 | 0.5550 |
| macro recall | 0.4078 | 0.4078 | 0.4338 | 0.4347 | 0.4303 |
| macro F1 | 0.4263 | 0.4263 | 0.4482 | 0.4473 | 0.4437 |
| weighted precision | 0.5004 | 0.5004 | 0.6111 | 0.6031 | 0.6085 |
| weighted recall | 0.4575 | 0.4575 | 0.4847 | 0.4847 | 0.4824 |
| weighted F1 | 0.4521 | 0.4521 | 0.5026 | 0.5003 | 0.5004 |

## Per-class metrics - window

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 471 | 0.3684 | 0.6093 | 0.4592 |
| 1 | Alarm | 764 | 0.4978 | 0.4529 | 0.4743 |
| 2 | Baby_Crying | 247 | 1.0000 | 0.7004 | 0.8238 |
| 3 | Car_Engine | 421 | 0.3266 | 0.3468 | 0.3364 |
| 4 | Dog_Bark | 355 | 0.6062 | 0.5465 | 0.5748 |
| 5 | Doorbell | 37 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 425 | 0.6311 | 0.3341 | 0.4369 |
| 7 | Footsteps | 525 | 0.5387 | 0.5962 | 0.5660 |
| 8 | Glass_Breaking | 185 | 0.6277 | 0.3189 | 0.4229 |
| 9 | Gunshot | 206 | 0.5287 | 0.2233 | 0.3140 |
| 10 | Help_Shouting | 314 | 0.3455 | 0.1815 | 0.2380 |
| 11 | Jackhammer | 374 | 0.4965 | 0.7540 | 0.5987 |
| 12 | Knocking | 204 | 0.6687 | 0.5245 | 0.5879 |
| 13 | Motorcycle | 347 | 0.5897 | 0.1988 | 0.2974 |
| 14 | Siren | 405 | 0.5983 | 0.5111 | 0.5513 |
| 15 | Train | 788 | 0.2799 | 0.5102 | 0.3615 |
| 16 | Vehicle_Horn | 161 | 0.5556 | 0.1242 | 0.2030 |

## Per-class metrics - clip_mean_softmax

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 50 | 0.2266 | 0.5800 | 0.3258 |
| 1 | Alarm | 103 | 0.3789 | 0.5922 | 0.4621 |
| 2 | Baby_Crying | 55 | 1.0000 | 0.5818 | 0.7356 |
| 3 | Car_Engine | 143 | 0.4950 | 0.3497 | 0.4098 |
| 4 | Dog_Bark | 123 | 0.8046 | 0.5691 | 0.6667 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 150 | 0.7797 | 0.3067 | 0.4402 |
| 7 | Footsteps | 79 | 0.5327 | 0.7215 | 0.6129 |
| 8 | Glass_Breaking | 48 | 0.8636 | 0.3958 | 0.5429 |
| 9 | Gunshot | 56 | 0.6087 | 0.2500 | 0.3544 |
| 10 | Help_Shouting | 54 | 0.2222 | 0.1852 | 0.2020 |
| 11 | Jackhammer | 133 | 0.6913 | 0.7744 | 0.7305 |
| 12 | Knocking | 45 | 0.7879 | 0.5778 | 0.6667 |
| 13 | Motorcycle | 35 | 0.4000 | 0.2286 | 0.2909 |
| 14 | Siren | 137 | 0.8090 | 0.5255 | 0.6372 |
| 15 | Train | 79 | 0.1618 | 0.5570 | 0.2507 |
| 16 | Vehicle_Horn | 39 | 0.7778 | 0.1795 | 0.2917 |

## Per-class metrics - clip_topk_mean

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 50 | 0.2443 | 0.6400 | 0.3536 |
| 1 | Alarm | 103 | 0.3750 | 0.5825 | 0.4563 |
| 2 | Baby_Crying | 55 | 1.0000 | 0.5818 | 0.7356 |
| 3 | Car_Engine | 143 | 0.5155 | 0.3497 | 0.4167 |
| 4 | Dog_Bark | 123 | 0.7955 | 0.5691 | 0.6635 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 150 | 0.7419 | 0.3067 | 0.4340 |
| 7 | Footsteps | 79 | 0.5327 | 0.7215 | 0.6129 |
| 8 | Glass_Breaking | 48 | 0.8636 | 0.3958 | 0.5429 |
| 9 | Gunshot | 56 | 0.6250 | 0.2679 | 0.3750 |
| 10 | Help_Shouting | 54 | 0.2222 | 0.1852 | 0.2020 |
| 11 | Jackhammer | 133 | 0.6603 | 0.7744 | 0.7128 |
| 12 | Knocking | 45 | 0.7879 | 0.5778 | 0.6667 |
| 13 | Motorcycle | 35 | 0.3889 | 0.2000 | 0.2642 |
| 14 | Siren | 137 | 0.7826 | 0.5255 | 0.6288 |
| 15 | Train | 79 | 0.1609 | 0.5316 | 0.2471 |
| 16 | Vehicle_Horn | 39 | 0.7778 | 0.1795 | 0.2917 |

## Per-class metrics - clip_majority_vote

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 50 | 0.2290 | 0.6000 | 0.3315 |
| 1 | Alarm | 103 | 0.3765 | 0.5922 | 0.4604 |
| 2 | Baby_Crying | 55 | 1.0000 | 0.5818 | 0.7356 |
| 3 | Car_Engine | 143 | 0.4950 | 0.3497 | 0.4098 |
| 4 | Dog_Bark | 123 | 0.7931 | 0.5610 | 0.6571 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 150 | 0.8033 | 0.3267 | 0.4645 |
| 7 | Footsteps | 79 | 0.5327 | 0.7215 | 0.6129 |
| 8 | Glass_Breaking | 48 | 0.8261 | 0.3958 | 0.5352 |
| 9 | Gunshot | 56 | 0.6087 | 0.2500 | 0.3544 |
| 10 | Help_Shouting | 54 | 0.2174 | 0.1852 | 0.2000 |
| 11 | Jackhammer | 133 | 0.6913 | 0.7744 | 0.7305 |
| 12 | Knocking | 45 | 0.7879 | 0.5778 | 0.6667 |
| 13 | Motorcycle | 35 | 0.3333 | 0.1714 | 0.2264 |
| 14 | Siren | 137 | 0.8023 | 0.5036 | 0.6188 |
| 15 | Train | 79 | 0.1599 | 0.5443 | 0.2471 |
| 16 | Vehicle_Horn | 39 | 0.7778 | 0.1795 | 0.2917 |

## Confusion matrix - window (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **287** | 1 | 0 | 29 | 7 | 0 | 2 | 19 | 0 | 2 | 0 | 41 | 0 | 2 | 1 | 80 | 0 |
| Alarm | 65 | **346** | 0 | 16 | 34 | 0 | 5 | 31 | 1 | 1 | 4 | 25 | 2 | 0 | 76 | 146 | 12 |
| Baby_Crying | 0 | 41 | **173** | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 4 | 1 | 1 |
| Car_Engine | 60 | 0 | 0 | **146** | 0 | 0 | 0 | 0 | 0 | 0 | 26 | 0 | 0 | 22 | 0 | 167 | 0 |
| Dog_Bark | 0 | 24 | 0 | 7 | **194** | 0 | 10 | 13 | 1 | 4 | 13 | 6 | 2 | 2 | 3 | 76 | 0 |
| Doorbell | 0 | 23 | 0 | 0 | 1 | **0** | 1 | 9 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| Drilling | 108 | 8 | 0 | 6 | 0 | 0 | **142** | 0 | 0 | 0 | 13 | 52 | 0 | 0 | 4 | 92 | 0 |
| Footsteps | 14 | 5 | 0 | 13 | 17 | 0 | 3 | **313** | 1 | 10 | 0 | 23 | 26 | 4 | 12 | 84 | 0 |
| Glass_Breaking | 2 | 12 | 0 | 6 | 4 | 0 | 14 | 38 | **59** | 13 | 2 | 6 | 1 | 0 | 0 | 28 | 0 |
| Gunshot | 67 | 0 | 0 | 4 | 5 | 0 | 1 | 37 | 14 | **46** | 1 | 0 | 14 | 0 | 0 | 17 | 0 |
| Help_Shouting | 3 | 100 | 0 | 7 | 24 | 0 | 23 | 7 | 4 | 0 | **57** | 1 | 7 | 0 | 14 | 66 | 1 |
| Jackhammer | 0 | 0 | 0 | 76 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | **282** | 0 | 0 | 0 | 12 | 0 |
| Knocking | 0 | 2 | 0 | 1 | 13 | 0 | 0 | 59 | 2 | 6 | 0 | 3 | **107** | 0 | 0 | 11 | 0 |
| Motorcycle | 38 | 0 | 0 | 64 | 2 | 0 | 2 | 13 | 0 | 3 | 0 | 57 | 1 | **69** | 6 | 90 | 2 |
| Siren | 7 | 73 | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 3 | 0 | 0 | 6 | **207** | 105 | 0 |
| Train | 128 | 8 | 0 | 68 | 10 | 0 | 18 | 28 | 11 | 2 | 26 | 71 | 0 | 5 | 11 | **402** | 0 |
| Vehicle_Horn | 0 | 52 | 0 | 4 | 0 | 0 | 4 | 8 | 0 | 0 | 0 | 1 | 0 | 7 | 8 | 57 | **20** |

## Confusion matrix - clip_mean_softmax (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **29** | 0 | 0 | 5 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 11 | 0 |
| Alarm | 4 | **61** | 0 | 3 | 5 | 0 | 1 | 2 | 0 | 0 | 1 | 2 | 0 | 0 | 6 | 17 | 1 |
| Baby_Crying | 0 | 13 | **32** | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 1 | 1 |
| Car_Engine | 19 | 0 | 0 | **50** | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 7 | 0 | 57 | 0 |
| Dog_Bark | 0 | 5 | 0 | 2 | **70** | 0 | 4 | 4 | 0 | 1 | 5 | 3 | 0 | 1 | 1 | 27 | 0 |
| Doorbell | 0 | 6 | 0 | 0 | 0 | **0** | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 36 | 3 | 0 | 2 | 0 | 0 | **46** | 0 | 0 | 0 | 8 | 20 | 0 | 0 | 0 | 35 | 0 |
| Footsteps | 2 | 0 | 0 | 1 | 0 | 0 | 1 | **57** | 1 | 2 | 0 | 2 | 5 | 1 | 1 | 6 | 0 |
| Glass_Breaking | 0 | 4 | 0 | 1 | 1 | 0 | 3 | 10 | **19** | 3 | 1 | 1 | 0 | 0 | 0 | 5 | 0 |
| Gunshot | 20 | 0 | 0 | 0 | 3 | 0 | 1 | 10 | 1 | **14** | 1 | 0 | 1 | 0 | 0 | 5 | 0 |
| Help_Shouting | 0 | 25 | 0 | 0 | 5 | 0 | 3 | 1 | 0 | 0 | **10** | 0 | 1 | 0 | 1 | 8 | 0 |
| Jackhammer | 0 | 0 | 0 | 27 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | **103** | 0 | 0 | 0 | 2 | 0 |
| Knocking | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 13 | 0 | 3 | 0 | 0 | **26** | 0 | 0 | 1 | 0 |
| Motorcycle | 2 | 0 | 0 | 5 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 6 | 0 | **8** | 1 | 12 | 0 |
| Siren | 3 | 27 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | **72** | 32 | 0 |
| Train | 13 | 1 | 0 | 4 | 0 | 0 | 0 | 3 | 1 | 0 | 3 | 8 | 0 | 0 | 2 | **44** | 0 |
| Vehicle_Horn | 0 | 15 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 9 | **7** |

## Confusion matrix - clip_topk_mean (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **32** | 0 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 8 | 0 |
| Alarm | 6 | **60** | 0 | 3 | 5 | 0 | 1 | 2 | 0 | 0 | 1 | 2 | 0 | 0 | 8 | 14 | 1 |
| Baby_Crying | 0 | 13 | **32** | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 1 | 1 |
| Car_Engine | 19 | 0 | 0 | **50** | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 7 | 0 | 57 | 0 |
| Dog_Bark | 0 | 5 | 0 | 2 | **70** | 0 | 4 | 4 | 0 | 1 | 5 | 3 | 0 | 1 | 1 | 27 | 0 |
| Doorbell | 0 | 6 | 0 | 0 | 0 | **0** | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 36 | 3 | 0 | 2 | 0 | 0 | **46** | 0 | 0 | 0 | 8 | 20 | 0 | 0 | 0 | 35 | 0 |
| Footsteps | 1 | 0 | 0 | 1 | 1 | 0 | 1 | **57** | 1 | 2 | 0 | 3 | 5 | 0 | 1 | 6 | 0 |
| Glass_Breaking | 0 | 4 | 0 | 0 | 1 | 0 | 3 | 10 | **19** | 3 | 1 | 2 | 0 | 0 | 0 | 5 | 0 |
| Gunshot | 19 | 0 | 0 | 0 | 3 | 0 | 1 | 10 | 1 | **15** | 1 | 0 | 1 | 0 | 0 | 5 | 0 |
| Help_Shouting | 0 | 25 | 0 | 0 | 5 | 0 | 3 | 1 | 0 | 0 | **10** | 0 | 1 | 0 | 1 | 8 | 0 |
| Jackhammer | 0 | 0 | 0 | 27 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | **103** | 0 | 0 | 0 | 2 | 0 |
| Knocking | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 13 | 0 | 3 | 0 | 0 | **26** | 0 | 0 | 1 | 0 |
| Motorcycle | 2 | 0 | 0 | 4 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 9 | 0 | **7** | 1 | 10 | 0 |
| Siren | 3 | 27 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 2 | **72** | 32 | 0 |
| Train | 13 | 1 | 0 | 3 | 0 | 0 | 2 | 3 | 1 | 0 | 3 | 9 | 0 | 0 | 2 | **42** | 0 |
| Vehicle_Horn | 0 | 15 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 6 | 8 | **7** |

## Confusion matrix - clip_majority_vote (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **30** | 0 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 11 | 0 |
| Alarm | 5 | **61** | 0 | 3 | 6 | 0 | 1 | 2 | 0 | 0 | 1 | 2 | 0 | 0 | 5 | 16 | 1 |
| Baby_Crying | 0 | 12 | **32** | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 1 | 1 | 1 |
| Car_Engine | 19 | 0 | 0 | **50** | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 0 | 0 | 8 | 0 | 55 | 0 |
| Dog_Bark | 0 | 9 | 0 | 2 | **69** | 0 | 3 | 4 | 0 | 1 | 4 | 3 | 0 | 1 | 0 | 27 | 0 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 36 | 3 | 0 | 2 | 0 | 0 | **49** | 0 | 0 | 0 | 8 | 18 | 0 | 0 | 1 | 33 | 0 |
| Footsteps | 1 | 0 | 0 | 1 | 0 | 0 | 0 | **57** | 1 | 2 | 0 | 3 | 5 | 0 | 1 | 8 | 0 |
| Glass_Breaking | 1 | 4 | 0 | 1 | 1 | 0 | 3 | 10 | **19** | 3 | 1 | 1 | 0 | 0 | 0 | 4 | 0 |
| Gunshot | 19 | 0 | 0 | 0 | 3 | 0 | 1 | 10 | 1 | **14** | 1 | 0 | 1 | 0 | 0 | 6 | 0 |
| Help_Shouting | 0 | 25 | 0 | 0 | 5 | 0 | 3 | 1 | 0 | 0 | **10** | 0 | 1 | 0 | 1 | 8 | 0 |
| Jackhammer | 0 | 0 | 0 | 27 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | **103** | 0 | 0 | 0 | 2 | 0 |
| Knocking | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 13 | 0 | 3 | 0 | 0 | **26** | 0 | 0 | 1 | 0 |
| Motorcycle | 3 | 0 | 0 | 6 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 8 | 0 | **6** | 1 | 10 | 0 |
| Siren | 3 | 26 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 2 | **69** | 35 | 0 |
| Train | 14 | 1 | 0 | 4 | 0 | 0 | 1 | 3 | 1 | 0 | 3 | 7 | 0 | 0 | 2 | **43** | 0 |
| Vehicle_Horn | 0 | 15 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 9 | **7** |

## Top 15 clip-level confusions (mean-softmax)

largest off-diagonal cells of the clip_mean_softmax matrix.

| true | predicted | clips |
|---|---|---:|
| Car_Engine | Train | 57 |
| Drilling | Aircraft | 36 |
| Drilling | Train | 35 |
| Siren | Train | 32 |
| Dog_Bark | Train | 27 |
| Jackhammer | Car_Engine | 27 |
| Siren | Alarm | 27 |
| Help_Shouting | Alarm | 25 |
| Drilling | Jackhammer | 20 |
| Gunshot | Aircraft | 20 |
| Car_Engine | Aircraft | 19 |
| Alarm | Train | 17 |
| Vehicle_Horn | Alarm | 15 |
| Baby_Crying | Alarm | 13 |
| Knocking | Footsteps | 13 |

## Baby_Crying clip recall by sample-rate group (mean-softmax)

ledger sample_rate is the original clip sample rate; clips with no prediction (zero-window clips) are excluded.

| sample-rate group | clips (total) | clips (assessed) | correct | recall |
|---:|---:|---:|---:|---:|
| below 16000 | 32 | 32 | 32 | 1.0000 |
| 16000 and above | 23 | 23 | 0 | 0.0000 |

## Misclassified clips - within vs cross dominant source_dataset (mean-softmax)

A class's dominant source_dataset is the dataset with the most clips of that class in the ledger. An error is within-dataset when the predicted class has the same dominant source_dataset as the true class, and cross-dataset otherwise.

- misclassified clips: **689**
- within-dataset errors: **276**
- cross-dataset errors: **413**
- errors with no dominant dataset for either class: **0**

| true class | predicted class | kind | clips |
|---|---|---:|---:|
| Car_Engine | Train | cross | 57 |
| Drilling | Aircraft | cross | 36 |
| Drilling | Train | cross | 35 |
| Siren | Train | cross | 32 |
| Jackhammer | Car_Engine | within | 27 |
| Dog_Bark | Train | cross | 27 |
| Siren | Alarm | cross | 27 |
| Help_Shouting | Alarm | within | 25 |
| Gunshot | Aircraft | within | 20 |
| Drilling | Jackhammer | within | 20 |
| Car_Engine | Aircraft | cross | 19 |
| Alarm | Train | within | 17 |
| Vehicle_Horn | Alarm | cross | 15 |
| Baby_Crying | Alarm | cross | 13 |
| Train | Aircraft | within | 13 |
| Knocking | Footsteps | within | 13 |
| Motorcycle | Train | within | 12 |
| Aircraft | Train | within | 11 |
| Gunshot | Footsteps | within | 10 |
| Glass_Breaking | Footsteps | within | 10 |
| Car_Engine | Help_Shouting | cross | 10 |
| Vehicle_Horn | Train | cross | 9 |
| Help_Shouting | Train | within | 8 |
| Train | Jackhammer | cross | 8 |
| Drilling | Help_Shouting | cross | 8 |
| Car_Engine | Motorcycle | cross | 7 |
| Baby_Crying | Help_Shouting | cross | 6 |
| Footsteps | Train | within | 6 |
| Motorcycle | Jackhammer | cross | 6 |
| Alarm | Siren | cross | 6 |
| Doorbell | Alarm | within | 6 |
| Aircraft | Car_Engine | cross | 5 |
| Footsteps | Knocking | within | 5 |
| Alarm | Dog_Bark | cross | 5 |
| Motorcycle | Car_Engine | cross | 5 |
| Vehicle_Horn | Siren | within | 5 |
| Glass_Breaking | Train | within | 5 |
| Help_Shouting | Dog_Bark | cross | 5 |
| Gunshot | Train | within | 5 |
| Dog_Bark | Alarm | cross | 5 |
| Dog_Bark | Help_Shouting | cross | 5 |
| Aircraft | Jackhammer | cross | 4 |
| Alarm | Aircraft | within | 4 |
| Glass_Breaking | Alarm | within | 4 |
| Train | Car_Engine | cross | 4 |
| Dog_Bark | Footsteps | cross | 4 |
| Dog_Bark | Drilling | within | 4 |
| Gunshot | Dog_Bark | cross | 3 |
| Train | Footsteps | within | 3 |
| Glass_Breaking | Gunshot | within | 3 |
| Knocking | Gunshot | within | 3 |
| Glass_Breaking | Drilling | cross | 3 |
| Help_Shouting | Drilling | cross | 3 |
| Alarm | Car_Engine | cross | 3 |
| Train | Help_Shouting | within | 3 |
| Dog_Bark | Jackhammer | within | 3 |
| Siren | Aircraft | cross | 3 |
| Drilling | Alarm | cross | 3 |
| Baby_Crying | Dog_Bark | cross | 2 |
| Footsteps | Gunshot | within | 2 |
| Alarm | Jackhammer | cross | 2 |
| Footsteps | Jackhammer | cross | 2 |
| Footsteps | Aircraft | within | 2 |
| Alarm | Footsteps | within | 2 |
| Doorbell | Footsteps | within | 2 |
| Motorcycle | Aircraft | within | 2 |
| Train | Siren | cross | 2 |
| Jackhammer | Train | cross | 2 |
| Dog_Bark | Car_Engine | within | 2 |
| Siren | Motorcycle | cross | 2 |
| Drilling | Car_Engine | within | 2 |
| Vehicle_Horn | Car_Engine | within | 1 |
| Alarm | Drilling | cross | 1 |
| Alarm | Vehicle_Horn | cross | 1 |
| Motorcycle | Siren | cross | 1 |
| Glass_Breaking | Dog_Bark | cross | 1 |
| Train | Alarm | within | 1 |
| Gunshot | Glass_Breaking | within | 1 |
| Footsteps | Glass_Breaking | within | 1 |
| Footsteps | Motorcycle | within | 1 |
| Glass_Breaking | Jackhammer | cross | 1 |
| Glass_Breaking | Help_Shouting | within | 1 |
| Vehicle_Horn | Footsteps | cross | 1 |
| Help_Shouting | Knocking | within | 1 |
| Knocking | Alarm | within | 1 |
| Knocking | Dog_Bark | cross | 1 |
| Gunshot | Knocking | within | 1 |
| Footsteps | Siren | cross | 1 |
| Gunshot | Drilling | cross | 1 |
| Motorcycle | Footsteps | within | 1 |
| Help_Shouting | Footsteps | within | 1 |
| Help_Shouting | Siren | cross | 1 |
| Vehicle_Horn | Motorcycle | cross | 1 |
| Knocking | Train | within | 1 |
| Footsteps | Drilling | cross | 1 |
| Gunshot | Help_Shouting | within | 1 |
| Glass_Breaking | Car_Engine | cross | 1 |
| Alarm | Help_Shouting | within | 1 |
| Footsteps | Car_Engine | cross | 1 |
| Train | Glass_Breaking | within | 1 |
| Aircraft | Footsteps | within | 1 |
| Siren | Footsteps | cross | 1 |
| Jackhammer | Footsteps | cross | 1 |
| Dog_Bark | Motorcycle | cross | 1 |
| Dog_Bark | Gunshot | cross | 1 |
| Dog_Bark | Siren | within | 1 |
| Baby_Crying | Train | cross | 1 |
| Baby_Crying | Vehicle_Horn | cross | 1 |

## Per-class clip recall by source_dataset (mean-softmax)

classes with more than one source_dataset in the validation split.

| class | source_dataset | clips | correct | recall |
|---|---|---:|---:|---:|
| Aircraft | ESC-50 | 13 | 7 | 0.5385 |
| Aircraft | FSD50K | 37 | 22 | 0.5946 |
| Alarm | ESC-50 | 3 | 3 | 1.0000 |
| Alarm | FSD50K | 100 | 58 | 0.5800 |
| Baby_Crying | ESC-50 | 17 | 0 | 0.0000 |
| Baby_Crying | owlgebra_babycry | 38 | 32 | 0.8421 |
| Footsteps | ESC-50 | 6 | 5 | 0.8333 |
| Footsteps | FSD50K | 73 | 52 | 0.7123 |
| Glass_Breaking | ESC-50 | 4 | 4 | 1.0000 |
| Glass_Breaking | FSD50K | 44 | 15 | 0.3409 |
| Gunshot | FSD50K | 33 | 9 | 0.2727 |
| Gunshot | UrbanSound8K | 23 | 5 | 0.2174 |
| Knocking | ESC-50 | 4 | 2 | 0.5000 |
| Knocking | FSD50K | 41 | 24 | 0.5854 |
| Train | ESC-50 | 6 | 3 | 0.5000 |
| Train | FSD50K | 73 | 41 | 0.5616 |
| Vehicle_Horn | ESC-50 | 3 | 0 | 0.0000 |
| Vehicle_Horn | FSD50K | 11 | 0 | 0.0000 |
| Vehicle_Horn | UrbanSound8K | 25 | 7 | 0.2800 |

## Clips with zero windows

No validation clip has zero windows.

model.predict: 2.0 s