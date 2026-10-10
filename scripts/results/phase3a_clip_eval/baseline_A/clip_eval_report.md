# Phase 3A Clip-Level Evaluation (analysis only)

Model: `data\models\baseline_A.keras` (729,178 bytes, 56,657 parameters). Analysis preset `A`: locked baseline: 200 ms, 50% overlap, FFT 1024, hop 512, 64 Mel -> (64, 5, 1). Frozen Phase 2B clip-level split, window and clip levels reported side by side. No retraining; model, preprocessing, configuration and dataset unchanged. The test split is never opened.

- validation windows: **74,830**
- validation clips: **1,618**
- clips with zero windows: **0**
- top-k size for clip_topk_mean: **5**

## Aggregation definitions

- window: one prediction per window (no aggregation).
- clip_mean_softmax: mean over the clip's softmax probability vectors, then argmax.
- clip_topk_mean: per class, the mean softmax probability across that class's --top-k highest-probability windows in the clip (k = min(--top-k, windows in the clip)), then argmax over classes.
- clip_majority_vote: majority vote of per-window argmax; ties are broken by the higher mean softmax over the clip.

## Overall metrics (window and clip levels)

| metric | window | clip mean-softmax | clip top-k mean | clip majority vote |
|---|---|---|---|---|
| samples | 74,830 | 74,830 | 1,618 | 1,618 | 1,618 |
| correct | 31,749 | 31,749 | 808 | 837 | 792 |
| accuracy | 0.4243 | 0.4243 | 0.4994 | 0.5173 | 0.4895 |
| macro precision | 0.5142 | 0.5142 | 0.6273 | 0.5825 | 0.6078 |
| macro recall | 0.3824 | 0.3824 | 0.4552 | 0.4663 | 0.4420 |
| macro F1 | 0.4048 | 0.4048 | 0.4689 | 0.4639 | 0.4557 |
| weighted precision | 0.5033 | 0.5033 | 0.6774 | 0.6276 | 0.6657 |
| weighted recall | 0.4243 | 0.4243 | 0.4994 | 0.5173 | 0.4895 |
| weighted F1 | 0.4278 | 0.4278 | 0.5193 | 0.5166 | 0.5112 |

## Per-class metrics - window

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 5146 | 0.4622 | 0.4341 | 0.4477 |
| 1 | Alarm | 8465 | 0.5427 | 0.4004 | 0.4608 |
| 2 | Baby_Crying | 3086 | 0.9795 | 0.7119 | 0.8245 |
| 3 | Car_Engine | 5560 | 0.5229 | 0.2932 | 0.3757 |
| 4 | Dog_Bark | 4656 | 0.2874 | 0.4227 | 0.3421 |
| 5 | Doorbell | 428 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 5820 | 0.5685 | 0.3708 | 0.4488 |
| 7 | Footsteps | 5912 | 0.2558 | 0.7348 | 0.3795 |
| 8 | Glass_Breaking | 2069 | 0.5912 | 0.0909 | 0.1575 |
| 9 | Gunshot | 3022 | 0.4805 | 0.2204 | 0.3022 |
| 10 | Help_Shouting | 3626 | 0.4361 | 0.4173 | 0.4265 |
| 11 | Jackhammer | 5181 | 0.6349 | 0.5343 | 0.5802 |
| 12 | Knocking | 1926 | 0.5930 | 0.2731 | 0.3740 |
| 13 | Motorcycle | 3826 | 0.5733 | 0.1289 | 0.2104 |
| 14 | Siren | 5277 | 0.7176 | 0.6790 | 0.6978 |
| 15 | Train | 8624 | 0.2356 | 0.3664 | 0.2868 |
| 16 | Vehicle_Horn | 2206 | 0.8598 | 0.4225 | 0.5666 |

## Per-class metrics - clip_mean_softmax

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 51 | 0.3594 | 0.4510 | 0.4000 |
| 1 | Alarm | 120 | 0.5036 | 0.5833 | 0.5405 |
| 2 | Baby_Crying | 70 | 1.0000 | 0.7143 | 0.8333 |
| 3 | Car_Engine | 146 | 0.7797 | 0.3151 | 0.4488 |
| 4 | Dog_Bark | 158 | 0.5523 | 0.6013 | 0.5758 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 168 | 0.7033 | 0.3810 | 0.4942 |
| 7 | Footsteps | 87 | 0.2149 | 0.8621 | 0.3440 |
| 8 | Glass_Breaking | 75 | 0.9286 | 0.1733 | 0.2921 |
| 9 | Gunshot | 122 | 0.8462 | 0.1803 | 0.2973 |
| 10 | Help_Shouting | 76 | 0.5357 | 0.5921 | 0.5625 |
| 11 | Jackhammer | 148 | 0.7850 | 0.5676 | 0.6588 |
| 12 | Knocking | 63 | 0.9130 | 0.3333 | 0.4884 |
| 13 | Motorcycle | 35 | 0.5455 | 0.1714 | 0.2609 |
| 14 | Siren | 142 | 0.8750 | 0.8873 | 0.8811 |
| 15 | Train | 80 | 0.1215 | 0.3750 | 0.1835 |
| 16 | Vehicle_Horn | 69 | 1.0000 | 0.5507 | 0.7103 |

## Per-class metrics - clip_topk_mean

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 51 | 0.3043 | 0.5490 | 0.3916 |
| 1 | Alarm | 120 | 0.5401 | 0.6167 | 0.5759 |
| 2 | Baby_Crying | 70 | 1.0000 | 0.7286 | 0.8430 |
| 3 | Car_Engine | 146 | 0.6364 | 0.3356 | 0.4395 |
| 4 | Dog_Bark | 158 | 0.5939 | 0.6203 | 0.6068 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 168 | 0.6598 | 0.3810 | 0.4830 |
| 7 | Footsteps | 87 | 0.2125 | 0.8621 | 0.3409 |
| 8 | Glass_Breaking | 75 | 0.9167 | 0.1467 | 0.2529 |
| 9 | Gunshot | 122 | 0.6316 | 0.0984 | 0.1702 |
| 10 | Help_Shouting | 76 | 0.5256 | 0.5395 | 0.5325 |
| 11 | Jackhammer | 148 | 0.7237 | 0.7432 | 0.7333 |
| 12 | Knocking | 63 | 0.8788 | 0.4603 | 0.6042 |
| 13 | Motorcycle | 35 | 0.4444 | 0.1143 | 0.1818 |
| 14 | Siren | 142 | 0.7849 | 0.9507 | 0.8599 |
| 15 | Train | 80 | 0.1181 | 0.1875 | 0.1449 |
| 16 | Vehicle_Horn | 69 | 0.9318 | 0.5942 | 0.7257 |

## Per-class metrics - clip_majority_vote

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 51 | 0.3387 | 0.4118 | 0.3717 |
| 1 | Alarm | 120 | 0.5038 | 0.5500 | 0.5259 |
| 2 | Baby_Crying | 70 | 1.0000 | 0.7143 | 0.8333 |
| 3 | Car_Engine | 146 | 0.7541 | 0.3151 | 0.4444 |
| 4 | Dog_Bark | 158 | 0.5385 | 0.5759 | 0.5566 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 168 | 0.6957 | 0.3810 | 0.4923 |
| 7 | Footsteps | 87 | 0.2071 | 0.8736 | 0.3348 |
| 8 | Glass_Breaking | 75 | 0.8750 | 0.1867 | 0.3077 |
| 9 | Gunshot | 122 | 0.8889 | 0.1967 | 0.3221 |
| 10 | Help_Shouting | 76 | 0.5000 | 0.5658 | 0.5309 |
| 11 | Jackhammer | 148 | 0.7699 | 0.5878 | 0.6667 |
| 12 | Knocking | 63 | 0.8750 | 0.3333 | 0.4828 |
| 13 | Motorcycle | 35 | 0.3750 | 0.0857 | 0.1395 |
| 14 | Siren | 142 | 0.8897 | 0.8521 | 0.8705 |
| 15 | Train | 80 | 0.1208 | 0.3625 | 0.1812 |
| 16 | Vehicle_Horn | 69 | 1.0000 | 0.5217 | 0.6857 |

## Confusion matrix - window (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **2234** | 35 | 0 | 205 | 137 | 0 | 63 | 844 | 1 | 74 | 9 | 98 | 10 | 17 | 11 | 1405 | 3 |
| Alarm | 456 | **3389** | 20 | 146 | 472 | 1 | 164 | 1474 | 30 | 11 | 528 | 174 | 37 | 15 | 463 | 976 | 109 |
| Baby_Crying | 3 | 315 | **2197** | 0 | 133 | 0 | 39 | 76 | 4 | 2 | 279 | 0 | 1 | 2 | 16 | 15 | 4 |
| Car_Engine | 114 | 2 | 6 | **1630** | 1153 | 0 | 88 | 466 | 0 | 51 | 0 | 11 | 0 | 95 | 25 | 1919 | 0 |
| Dog_Bark | 34 | 315 | 0 | 70 | **1968** | 0 | 159 | 1142 | 2 | 31 | 350 | 74 | 52 | 24 | 166 | 260 | 9 |
| Doorbell | 0 | 248 | 0 | 0 | 10 | **0** | 14 | 138 | 2 | 0 | 2 | 1 | 1 | 0 | 12 | 0 | 0 |
| Drilling | 171 | 26 | 0 | 54 | 613 | 0 | **2158** | 513 | 11 | 74 | 363 | 395 | 2 | 0 | 139 | 1293 | 8 |
| Footsteps | 100 | 84 | 8 | 10 | 310 | 0 | 39 | **4344** | 18 | 77 | 34 | 99 | 60 | 7 | 125 | 595 | 2 |
| Glass_Breaking | 1 | 91 | 0 | 4 | 51 | 0 | 170 | 1167 | **188** | 84 | 20 | 147 | 6 | 3 | 16 | 121 | 0 |
| Gunshot | 631 | 35 | 0 | 6 | 77 | 0 | 60 | 1004 | 42 | **666** | 8 | 12 | 122 | 4 | 0 | 355 | 0 |
| Help_Shouting | 0 | 510 | 9 | 32 | 358 | 0 | 63 | 592 | 16 | 65 | **1513** | 19 | 15 | 3 | 177 | 249 | 5 |
| Jackhammer | 0 | 0 | 0 | 301 | 0 | 0 | 272 | 680 | 1 | 0 | 0 | **2768** | 0 | 12 | 0 | 1147 | 0 |
| Knocking | 16 | 29 | 0 | 8 | 212 | 0 | 4 | 863 | 1 | 73 | 6 | 15 | **526** | 0 | 32 | 141 | 0 |
| Motorcycle | 60 | 76 | 2 | 477 | 112 | 0 | 70 | 893 | 0 | 12 | 32 | 185 | 23 | **493** | 93 | 1297 | 1 |
| Siren | 8 | 504 | 0 | 6 | 868 | 0 | 2 | 52 | 0 | 16 | 76 | 0 | 0 | 3 | **3583** | 153 | 6 |
| Train | 994 | 360 | 0 | 117 | 337 | 0 | 383 | 2235 | 2 | 150 | 244 | 335 | 31 | 151 | 120 | **3160** | 5 |
| Vehicle_Horn | 11 | 226 | 1 | 51 | 37 | 0 | 48 | 497 | 0 | 0 | 5 | 27 | 1 | 31 | 15 | 324 | **932** |

## Confusion matrix - clip_mean_softmax (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **23** | 0 | 0 | 1 | 1 | 0 | 1 | 9 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 14 | 0 |
| Alarm | 4 | **70** | 0 | 1 | 6 | 0 | 1 | 12 | 0 | 0 | 8 | 2 | 0 | 0 | 4 | 12 | 0 |
| Baby_Crying | 0 | 12 | **50** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 2 | 0 | 0 | **46** | 30 | 0 | 2 | 10 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 54 | 0 |
| Dog_Bark | 1 | 10 | 0 | 0 | **95** | 0 | 4 | 23 | 0 | 1 | 7 | 2 | 0 | 1 | 3 | 11 | 0 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 4 | 1 | 0 | 0 | 16 | 0 | **64** | 16 | 0 | 1 | 14 | 11 | 0 | 0 | 6 | 35 | 0 |
| Footsteps | 0 | 0 | 0 | 0 | 3 | 0 | 0 | **75** | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 6 | 0 |
| Glass_Breaking | 0 | 4 | 0 | 0 | 1 | 0 | 4 | 50 | **13** | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 |
| Gunshot | 22 | 0 | 0 | 0 | 3 | 0 | 0 | 64 | 1 | **22** | 0 | 0 | 1 | 0 | 0 | 9 | 0 |
| Help_Shouting | 0 | 15 | 0 | 0 | 2 | 0 | 1 | 7 | 0 | 0 | **45** | 0 | 0 | 0 | 3 | 3 | 0 |
| Jackhammer | 0 | 0 | 0 | 5 | 0 | 0 | 7 | 11 | 0 | 0 | 0 | **84** | 0 | 0 | 0 | 41 | 0 |
| Knocking | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 38 | 0 | 0 | 0 | 0 | **21** | 0 | 0 | 1 | 0 |
| Motorcycle | 1 | 1 | 0 | 5 | 1 | 0 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | **6** | 0 | 16 | 0 |
| Siren | 0 | 5 | 0 | 0 | 8 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | **126** | 1 | 0 |
| Train | 7 | 6 | 0 | 1 | 3 | 0 | 3 | 23 | 0 | 0 | 2 | 3 | 1 | 1 | 0 | **30** | 0 |
| Vehicle_Horn | 0 | 9 | 0 | 0 | 0 | 0 | 3 | 4 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 13 | **38** |

## Confusion matrix - clip_topk_mean (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **28** | 0 | 0 | 4 | 2 | 0 | 1 | 9 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 3 | 0 |
| Alarm | 5 | **74** | 0 | 2 | 4 | 0 | 2 | 9 | 0 | 0 | 7 | 2 | 0 | 0 | 10 | 4 | 1 |
| Baby_Crying | 0 | 7 | **51** | 0 | 3 | 0 | 2 | 2 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 1 |
| Car_Engine | 2 | 0 | 0 | **49** | 30 | 0 | 4 | 12 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 46 | 0 |
| Dog_Bark | 0 | 10 | 0 | 1 | **98** | 0 | 4 | 20 | 0 | 1 | 9 | 2 | 0 | 1 | 7 | 4 | 1 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| Drilling | 6 | 1 | 0 | 0 | 12 | 0 | **64** | 22 | 0 | 2 | 13 | 15 | 0 | 0 | 7 | 26 | 0 |
| Footsteps | 3 | 0 | 0 | 0 | 2 | 0 | 0 | **75** | 0 | 1 | 0 | 1 | 1 | 0 | 2 | 2 | 0 |
| Glass_Breaking | 0 | 5 | 0 | 0 | 2 | 0 | 1 | 49 | **11** | 1 | 0 | 5 | 0 | 0 | 1 | 0 | 0 |
| Gunshot | 34 | 0 | 0 | 0 | 2 | 0 | 2 | 64 | 1 | **12** | 0 | 0 | 2 | 0 | 0 | 5 | 0 |
| Help_Shouting | 0 | 16 | 0 | 1 | 4 | 0 | 3 | 6 | 0 | 1 | **41** | 0 | 0 | 0 | 4 | 0 | 0 |
| Jackhammer | 0 | 0 | 0 | 11 | 0 | 0 | 7 | 14 | 0 | 0 | 0 | **110** | 0 | 0 | 0 | 6 | 0 |
| Knocking | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 31 | 0 | 0 | 0 | 0 | **29** | 0 | 1 | 1 | 0 |
| Motorcycle | 1 | 0 | 0 | 8 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 5 | 0 | **4** | 2 | 6 | 0 |
| Siren | 0 | 3 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | **135** | 1 | 0 |
| Train | 13 | 5 | 0 | 1 | 5 | 0 | 4 | 23 | 0 | 0 | 3 | 7 | 1 | 2 | 1 | **15** | 0 |
| Vehicle_Horn | 0 | 10 | 0 | 0 | 0 | 0 | 2 | 6 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 8 | **41** |

## Confusion matrix - clip_majority_vote (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **21** | 0 | 0 | 2 | 1 | 0 | 1 | 8 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 17 | 0 |
| Alarm | 4 | **66** | 0 | 1 | 4 | 0 | 2 | 15 | 0 | 0 | 11 | 1 | 0 | 0 | 4 | 12 | 0 |
| Baby_Crying | 0 | 11 | **50** | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 6 | 0 | 0 | 1 | 0 | 0 | 0 |
| Car_Engine | 2 | 0 | 0 | **46** | 30 | 0 | 2 | 10 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 54 | 0 |
| Dog_Bark | 1 | 6 | 0 | 1 | **91** | 0 | 4 | 30 | 1 | 0 | 8 | 2 | 0 | 1 | 3 | 10 | 0 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 5 | 1 | 0 | 0 | 16 | 0 | **64** | 16 | 0 | 1 | 15 | 13 | 0 | 0 | 4 | 33 | 0 |
| Footsteps | 0 | 0 | 0 | 0 | 3 | 0 | 0 | **76** | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 5 | 0 |
| Glass_Breaking | 0 | 3 | 0 | 0 | 1 | 0 | 3 | 49 | **14** | 0 | 0 | 3 | 0 | 0 | 0 | 2 | 0 |
| Gunshot | 20 | 0 | 0 | 0 | 2 | 0 | 0 | 65 | 1 | **24** | 0 | 0 | 1 | 0 | 0 | 9 | 0 |
| Help_Shouting | 0 | 16 | 0 | 0 | 2 | 0 | 2 | 7 | 0 | 0 | **43** | 0 | 1 | 0 | 3 | 2 | 0 |
| Jackhammer | 0 | 0 | 0 | 5 | 0 | 0 | 7 | 15 | 0 | 0 | 0 | **87** | 0 | 0 | 0 | 34 | 0 |
| Knocking | 0 | 1 | 0 | 0 | 3 | 0 | 0 | 36 | 0 | 0 | 0 | 0 | **21** | 0 | 0 | 2 | 0 |
| Motorcycle | 1 | 1 | 0 | 4 | 2 | 0 | 0 | 6 | 0 | 0 | 0 | 2 | 0 | **3** | 0 | 16 | 0 |
| Siren | 0 | 8 | 0 | 0 | 10 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | **121** | 1 | 0 |
| Train | 8 | 4 | 0 | 1 | 3 | 0 | 3 | 25 | 0 | 0 | 2 | 3 | 1 | 1 | 0 | **29** | 0 |
| Vehicle_Horn | 0 | 9 | 0 | 1 | 0 | 0 | 3 | 5 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 14 | **36** |

## Top 15 clip-level confusions (mean-softmax)

largest off-diagonal cells of the clip_mean_softmax matrix.

| true | predicted | clips |
|---|---|---:|
| Gunshot | Footsteps | 64 |
| Car_Engine | Train | 54 |
| Glass_Breaking | Footsteps | 50 |
| Jackhammer | Train | 41 |
| Knocking | Footsteps | 38 |
| Drilling | Train | 35 |
| Car_Engine | Dog_Bark | 30 |
| Dog_Bark | Footsteps | 23 |
| Train | Footsteps | 23 |
| Gunshot | Aircraft | 22 |
| Drilling | Dog_Bark | 16 |
| Drilling | Footsteps | 16 |
| Motorcycle | Train | 16 |
| Help_Shouting | Alarm | 15 |
| Aircraft | Train | 14 |

## Baby_Crying clip recall by sample-rate group (mean-softmax)

ledger sample_rate is the original clip sample rate; clips with no prediction (zero-window clips) are excluded.

| sample-rate group | clips (total) | clips (assessed) | correct | recall |
|---:|---:|---:|---:|---:|
| below 16000 | 32 | 32 | 32 | 1.0000 |
| 16000 and above | 38 | 38 | 18 | 0.4737 |

## Misclassified clips - within vs cross dominant source_dataset (mean-softmax)

A class's dominant source_dataset is the dataset with the most clips of that class in the ledger. An error is within-dataset when the predicted class has the same dominant source_dataset as the true class, and cross-dataset otherwise.

- misclassified clips: **810**
- within-dataset errors: **451**
- cross-dataset errors: **359**
- errors with no dominant dataset for either class: **0**

| true class | predicted class | kind | clips |
|---|---|---:|---:|
| Gunshot | Footsteps | within | 64 |
| Car_Engine | Train | cross | 54 |
| Glass_Breaking | Footsteps | within | 50 |
| Jackhammer | Train | cross | 41 |
| Knocking | Footsteps | within | 38 |
| Drilling | Train | cross | 35 |
| Car_Engine | Dog_Bark | within | 30 |
| Train | Footsteps | within | 23 |
| Dog_Bark | Footsteps | cross | 23 |
| Gunshot | Aircraft | within | 22 |
| Motorcycle | Train | within | 16 |
| Drilling | Dog_Bark | within | 16 |
| Drilling | Footsteps | cross | 16 |
| Help_Shouting | Alarm | within | 15 |
| Aircraft | Train | within | 14 |
| Drilling | Help_Shouting | cross | 14 |
| Vehicle_Horn | Train | cross | 13 |
| Baby_Crying | Alarm | cross | 12 |
| Alarm | Train | within | 12 |
| Alarm | Footsteps | within | 12 |
| Jackhammer | Footsteps | cross | 11 |
| Dog_Bark | Train | cross | 11 |
| Drilling | Jackhammer | within | 11 |
| Car_Engine | Footsteps | cross | 10 |
| Dog_Bark | Alarm | cross | 10 |
| Aircraft | Footsteps | within | 9 |
| Gunshot | Train | within | 9 |
| Vehicle_Horn | Alarm | cross | 9 |
| Alarm | Help_Shouting | within | 8 |
| Siren | Dog_Bark | within | 8 |
| Baby_Crying | Help_Shouting | cross | 7 |
| Train | Aircraft | within | 7 |
| Help_Shouting | Footsteps | within | 7 |
| Dog_Bark | Help_Shouting | cross | 7 |
| Jackhammer | Drilling | within | 7 |
| Alarm | Dog_Bark | cross | 6 |
| Train | Alarm | within | 6 |
| Footsteps | Train | within | 6 |
| Drilling | Siren | within | 6 |
| Motorcycle | Car_Engine | cross | 5 |
| Doorbell | Alarm | within | 5 |
| Jackhammer | Car_Engine | within | 5 |
| Siren | Alarm | cross | 5 |
| Glass_Breaking | Alarm | within | 4 |
| Vehicle_Horn | Footsteps | cross | 4 |
| Alarm | Siren | cross | 4 |
| Motorcycle | Footsteps | within | 4 |
| Alarm | Aircraft | within | 4 |
| Glass_Breaking | Drilling | cross | 4 |
| Dog_Bark | Drilling | within | 4 |
| Drilling | Aircraft | cross | 4 |
| Help_Shouting | Siren | cross | 3 |
| Footsteps | Dog_Bark | cross | 3 |
| Gunshot | Dog_Bark | cross | 3 |
| Train | Dog_Bark | cross | 3 |
| Train | Drilling | cross | 3 |
| Help_Shouting | Train | within | 3 |
| Vehicle_Horn | Drilling | within | 3 |
| Train | Jackhammer | cross | 3 |
| Dog_Bark | Siren | within | 3 |
| Alarm | Jackhammer | cross | 2 |
| Help_Shouting | Dog_Bark | cross | 2 |
| Glass_Breaking | Jackhammer | cross | 2 |
| Knocking | Dog_Bark | cross | 2 |
| Train | Help_Shouting | within | 2 |
| Doorbell | Footsteps | within | 2 |
| Car_Engine | Drilling | within | 2 |
| Dog_Bark | Jackhammer | within | 2 |
| Car_Engine | Aircraft | cross | 2 |
| Baby_Crying | Dog_Bark | cross | 1 |
| Gunshot | Glass_Breaking | within | 1 |
| Alarm | Drilling | cross | 1 |
| Train | Knocking | within | 1 |
| Motorcycle | Jackhammer | cross | 1 |
| Motorcycle | Alarm | within | 1 |
| Glass_Breaking | Dog_Bark | cross | 1 |
| Footsteps | Jackhammer | cross | 1 |
| Aircraft | Motorcycle | within | 1 |
| Aircraft | Drilling | cross | 1 |
| Footsteps | Gunshot | within | 1 |
| Train | Motorcycle | within | 1 |
| Knocking | Alarm | within | 1 |
| Vehicle_Horn | Siren | within | 1 |
| Motorcycle | Dog_Bark | cross | 1 |
| Footsteps | Siren | cross | 1 |
| Aircraft | Dog_Bark | cross | 1 |
| Glass_Breaking | Train | within | 1 |
| Vehicle_Horn | Motorcycle | cross | 1 |
| Knocking | Train | within | 1 |
| Gunshot | Knocking | within | 1 |
| Alarm | Car_Engine | cross | 1 |
| Motorcycle | Aircraft | within | 1 |
| Doorbell | Drilling | cross | 1 |
| Aircraft | Jackhammer | cross | 1 |
| Aircraft | Car_Engine | cross | 1 |
| Train | Car_Engine | cross | 1 |
| Help_Shouting | Drilling | cross | 1 |
| Car_Engine | Motorcycle | cross | 1 |
| Siren | Footsteps | cross | 1 |
| Siren | Help_Shouting | cross | 1 |
| Car_Engine | Gunshot | cross | 1 |
| Dog_Bark | Aircraft | cross | 1 |
| Dog_Bark | Motorcycle | cross | 1 |
| Drilling | Alarm | cross | 1 |
| Drilling | Gunshot | cross | 1 |
| Dog_Bark | Gunshot | cross | 1 |
| Siren | Train | cross | 1 |

## Per-class clip recall by source_dataset (mean-softmax)

classes with more than one source_dataset in the validation split.

| class | source_dataset | clips | correct | recall |
|---|---|---:|---:|---:|
| Aircraft | ESC-50 | 13 | 5 | 0.3846 |
| Aircraft | FSD50K | 38 | 18 | 0.4737 |
| Alarm | ESC-50 | 3 | 3 | 1.0000 |
| Alarm | FSD50K | 117 | 67 | 0.5726 |
| Baby_Crying | ESC-50 | 17 | 0 | 0.0000 |
| Baby_Crying | owlgebra_babycry | 53 | 50 | 0.9434 |
| Footsteps | ESC-50 | 6 | 6 | 1.0000 |
| Footsteps | FSD50K | 81 | 69 | 0.8519 |
| Glass_Breaking | ESC-50 | 4 | 1 | 0.2500 |
| Glass_Breaking | FSD50K | 71 | 12 | 0.1690 |
| Gunshot | FSD50K | 61 | 11 | 0.1803 |
| Gunshot | UrbanSound8K | 61 | 11 | 0.1803 |
| Knocking | ESC-50 | 4 | 1 | 0.2500 |
| Knocking | FSD50K | 59 | 20 | 0.3390 |
| Train | ESC-50 | 6 | 5 | 0.8333 |
| Train | FSD50K | 74 | 25 | 0.3378 |
| Vehicle_Horn | ESC-50 | 3 | 0 | 0.0000 |
| Vehicle_Horn | FSD50K | 11 | 1 | 0.0909 |
| Vehicle_Horn | UrbanSound8K | 55 | 37 | 0.6727 |

## Clips with zero windows

No validation clip has zero windows.

model.predict: 1.2 s