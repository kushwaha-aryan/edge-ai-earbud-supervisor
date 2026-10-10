# Phase 3A Clip-Level Evaluation (analysis only)

Model: `data\models\baseline_A.keras` (729,178 bytes, 56,657 parameters). Analysis preset `A`: locked baseline: 200 ms, 50% overlap, FFT 1024, hop 512, 64 Mel -> (64, 5, 1). Frozen Phase 2B clip-level split, window and clip levels reported side by side. No retraining; model, preprocessing, configuration and dataset unchanged. The test split is never opened.

- validation windows: **72,101**
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
| samples | 72,101 | 72,101 | 1,337 | 1,337 | 1,337 |
| correct | 30,660 | 30,660 | 685 | 715 | 669 |
| accuracy | 0.4252 | 0.4252 | 0.5123 | 0.5348 | 0.5004 |
| macro precision | 0.5116 | 0.5116 | 0.6389 | 0.5841 | 0.6216 |
| macro recall | 0.3808 | 0.3808 | 0.4558 | 0.4668 | 0.4409 |
| macro F1 | 0.4031 | 0.4031 | 0.4771 | 0.4694 | 0.4613 |
| weighted precision | 0.5025 | 0.5025 | 0.6678 | 0.6156 | 0.6568 |
| weighted recall | 0.4252 | 0.4252 | 0.5123 | 0.5348 | 0.5004 |
| weighted F1 | 0.4284 | 0.4284 | 0.5302 | 0.5302 | 0.5196 |

## Per-class metrics - window

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 5138 | 0.4713 | 0.4348 | 0.4523 |
| 1 | Alarm | 8282 | 0.5518 | 0.3959 | 0.4611 |
| 2 | Baby_Crying | 2943 | 0.9803 | 0.7085 | 0.8225 |
| 3 | Car_Engine | 5525 | 0.5238 | 0.2923 | 0.3752 |
| 4 | Dog_Bark | 4336 | 0.2724 | 0.4156 | 0.3291 |
| 5 | Doorbell | 428 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 5637 | 0.5784 | 0.3756 | 0.4554 |
| 7 | Footsteps | 5837 | 0.2642 | 0.7358 | 0.3888 |
| 8 | Glass_Breaking | 1861 | 0.5866 | 0.0801 | 0.1409 |
| 9 | Gunshot | 2368 | 0.4491 | 0.2386 | 0.3116 |
| 10 | Help_Shouting | 3391 | 0.4258 | 0.4096 | 0.4176 |
| 11 | Jackhammer | 4987 | 0.6403 | 0.5286 | 0.5791 |
| 12 | Knocking | 1751 | 0.5713 | 0.2656 | 0.3626 |
| 13 | Motorcycle | 3826 | 0.5793 | 0.1289 | 0.2108 |
| 14 | Siren | 5215 | 0.7180 | 0.6826 | 0.6999 |
| 15 | Train | 8610 | 0.2394 | 0.3670 | 0.2898 |
| 16 | Vehicle_Horn | 1966 | 0.8456 | 0.4151 | 0.5568 |

## Per-class metrics - clip_mean_softmax

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 50 | 0.3966 | 0.4600 | 0.4259 |
| 1 | Alarm | 103 | 0.5413 | 0.5728 | 0.5566 |
| 2 | Baby_Crying | 55 | 1.0000 | 0.6727 | 0.8043 |
| 3 | Car_Engine | 143 | 0.7759 | 0.3147 | 0.4478 |
| 4 | Dog_Bark | 123 | 0.4832 | 0.5854 | 0.5294 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 150 | 0.7284 | 0.3933 | 0.5108 |
| 7 | Footsteps | 79 | 0.2698 | 0.8608 | 0.4109 |
| 8 | Glass_Breaking | 48 | 1.0000 | 0.1875 | 0.3158 |
| 9 | Gunshot | 56 | 0.8125 | 0.2321 | 0.3611 |
| 10 | Help_Shouting | 54 | 0.4921 | 0.5741 | 0.5299 |
| 11 | Jackhammer | 133 | 0.8202 | 0.5489 | 0.6577 |
| 12 | Knocking | 45 | 0.9375 | 0.3333 | 0.4918 |
| 13 | Motorcycle | 35 | 0.6000 | 0.1714 | 0.2667 |
| 14 | Siren | 137 | 0.8723 | 0.8978 | 0.8849 |
| 15 | Train | 79 | 0.1322 | 0.3797 | 0.1961 |
| 16 | Vehicle_Horn | 39 | 1.0000 | 0.5641 | 0.7213 |

## Per-class metrics - clip_topk_mean

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 50 | 0.3373 | 0.5600 | 0.4211 |
| 1 | Alarm | 103 | 0.5833 | 0.6117 | 0.5972 |
| 2 | Baby_Crying | 55 | 1.0000 | 0.6909 | 0.8172 |
| 3 | Car_Engine | 143 | 0.6400 | 0.3357 | 0.4404 |
| 4 | Dog_Bark | 123 | 0.5319 | 0.6098 | 0.5682 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 150 | 0.6941 | 0.3933 | 0.5021 |
| 7 | Footsteps | 79 | 0.2636 | 0.8608 | 0.4036 |
| 8 | Glass_Breaking | 48 | 1.0000 | 0.1458 | 0.2545 |
| 9 | Gunshot | 56 | 0.5000 | 0.1071 | 0.1765 |
| 10 | Help_Shouting | 54 | 0.4737 | 0.5000 | 0.4865 |
| 11 | Jackhammer | 133 | 0.7388 | 0.7444 | 0.7416 |
| 12 | Knocking | 45 | 0.8750 | 0.4667 | 0.6087 |
| 13 | Motorcycle | 35 | 0.4444 | 0.1143 | 0.1818 |
| 14 | Siren | 137 | 0.7857 | 0.9635 | 0.8656 |
| 15 | Train | 79 | 0.1351 | 0.1899 | 0.1579 |
| 16 | Vehicle_Horn | 39 | 0.9259 | 0.6410 | 0.7576 |

## Per-class metrics - clip_majority_vote

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 50 | 0.3889 | 0.4200 | 0.4038 |
| 1 | Alarm | 103 | 0.5500 | 0.5340 | 0.5419 |
| 2 | Baby_Crying | 55 | 1.0000 | 0.6727 | 0.8043 |
| 3 | Car_Engine | 143 | 0.7627 | 0.3147 | 0.4455 |
| 4 | Dog_Bark | 123 | 0.4726 | 0.5610 | 0.5130 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 150 | 0.7195 | 0.3933 | 0.5086 |
| 7 | Footsteps | 79 | 0.2509 | 0.8734 | 0.3898 |
| 8 | Glass_Breaking | 48 | 1.0000 | 0.1667 | 0.2857 |
| 9 | Gunshot | 56 | 0.8235 | 0.2500 | 0.3836 |
| 10 | Help_Shouting | 54 | 0.4545 | 0.5556 | 0.5000 |
| 11 | Jackhammer | 133 | 0.8085 | 0.5714 | 0.6696 |
| 12 | Knocking | 45 | 0.8889 | 0.3556 | 0.5079 |
| 13 | Motorcycle | 35 | 0.4286 | 0.0857 | 0.1429 |
| 14 | Siren | 137 | 0.8872 | 0.8613 | 0.8741 |
| 15 | Train | 79 | 0.1312 | 0.3671 | 0.1933 |
| 16 | Vehicle_Horn | 39 | 1.0000 | 0.5128 | 0.6780 |

## Confusion matrix - window (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **2234** | 35 | 0 | 202 | 137 | 0 | 63 | 842 | 1 | 74 | 9 | 98 | 10 | 15 | 11 | 1404 | 3 |
| Alarm | 448 | **3279** | 17 | 146 | 470 | 1 | 164 | 1466 | 27 | 11 | 489 | 174 | 36 | 15 | 462 | 968 | 109 |
| Baby_Crying | 2 | 308 | **2085** | 0 | 128 | 0 | 38 | 76 | 4 | 1 | 272 | 0 | 0 | 0 | 16 | 11 | 2 |
| Car_Engine | 114 | 2 | 6 | **1615** | 1151 | 0 | 88 | 454 | 0 | 51 | 0 | 11 | 0 | 95 | 24 | 1914 | 0 |
| Dog_Bark | 34 | 299 | 0 | 65 | **1802** | 0 | 159 | 1078 | 0 | 24 | 336 | 74 | 50 | 22 | 160 | 224 | 9 |
| Doorbell | 0 | 248 | 0 | 0 | 10 | **0** | 14 | 138 | 2 | 0 | 2 | 1 | 1 | 0 | 12 | 0 | 0 |
| Drilling | 171 | 15 | 0 | 54 | 613 | 0 | **2117** | 466 | 7 | 72 | 361 | 319 | 2 | 0 | 139 | 1293 | 8 |
| Footsteps | 100 | 81 | 8 | 10 | 308 | 0 | 38 | **4295** | 17 | 74 | 34 | 94 | 60 | 4 | 125 | 587 | 2 |
| Glass_Breaking | 1 | 84 | 0 | 4 | 51 | 0 | 147 | 1063 | **149** | 76 | 19 | 124 | 4 | 3 | 16 | 120 | 0 |
| Gunshot | 551 | 21 | 0 | 2 | 58 | 0 | 49 | 681 | 34 | **565** | 4 | 10 | 117 | 4 | 0 | 272 | 0 |
| Help_Shouting | 0 | 441 | 9 | 32 | 347 | 0 | 56 | 581 | 10 | 61 | **1389** | 19 | 14 | 3 | 176 | 249 | 4 |
| Jackhammer | 0 | 0 | 0 | 301 | 0 | 0 | 231 | 669 | 0 | 0 | 0 | **2636** | 0 | 12 | 0 | 1138 | 0 |
| Knocking | 12 | 28 | 0 | 8 | 203 | 0 | 4 | 770 | 1 | 71 | 6 | 15 | **465** | 0 | 29 | 139 | 0 |
| Motorcycle | 60 | 76 | 2 | 477 | 112 | 0 | 70 | 893 | 0 | 12 | 32 | 185 | 23 | **493** | 93 | 1297 | 1 |
| Siren | 8 | 500 | 0 | 6 | 858 | 0 | 2 | 52 | 0 | 16 | 63 | 0 | 0 | 3 | **3560** | 141 | 6 |
| Train | 994 | 352 | 0 | 117 | 332 | 0 | 383 | 2235 | 2 | 150 | 243 | 335 | 31 | 151 | 120 | **3160** | 5 |
| Vehicle_Horn | 11 | 173 | 0 | 44 | 36 | 0 | 37 | 495 | 0 | 0 | 3 | 22 | 1 | 31 | 15 | 282 | **816** |

## Confusion matrix - clip_mean_softmax (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **23** | 0 | 0 | 1 | 1 | 0 | 1 | 9 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 14 | 0 |
| Alarm | 3 | **59** | 0 | 1 | 6 | 0 | 1 | 12 | 0 | 0 | 4 | 2 | 0 | 0 | 4 | 11 | 0 |
| Baby_Crying | 0 | 11 | **37** | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 2 | 0 | 0 | **45** | 30 | 0 | 2 | 9 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 53 | 0 |
| Dog_Bark | 1 | 8 | 0 | 0 | **72** | 0 | 4 | 18 | 0 | 0 | 7 | 2 | 0 | 1 | 3 | 7 | 0 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 4 | 0 | 0 | 0 | 16 | 0 | **59** | 12 | 0 | 1 | 13 | 4 | 0 | 0 | 6 | 35 | 0 |
| Footsteps | 0 | 0 | 0 | 0 | 3 | 0 | 0 | **68** | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 5 | 0 |
| Glass_Breaking | 0 | 3 | 0 | 0 | 1 | 0 | 4 | 28 | **9** | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 0 |
| Gunshot | 17 | 0 | 0 | 0 | 3 | 0 | 0 | 20 | 0 | **13** | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| Help_Shouting | 0 | 8 | 0 | 0 | 2 | 0 | 1 | 6 | 0 | 0 | **31** | 0 | 0 | 0 | 3 | 3 | 0 |
| Jackhammer | 0 | 0 | 0 | 5 | 0 | 0 | 4 | 10 | 0 | 0 | 0 | **73** | 0 | 0 | 0 | 41 | 0 |
| Knocking | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 26 | 0 | 0 | 0 | 0 | **15** | 0 | 0 | 1 | 0 |
| Motorcycle | 1 | 1 | 0 | 5 | 1 | 0 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | **6** | 0 | 16 | 0 |
| Siren | 0 | 5 | 0 | 0 | 8 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | **123** | 0 | 0 |
| Train | 7 | 5 | 0 | 1 | 3 | 0 | 3 | 23 | 0 | 0 | 2 | 3 | 1 | 1 | 0 | **30** | 0 |
| Vehicle_Horn | 0 | 3 | 0 | 0 | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 7 | **22** |

## Confusion matrix - clip_topk_mean (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **28** | 0 | 0 | 3 | 2 | 0 | 1 | 9 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 3 | 0 |
| Alarm | 4 | **63** | 0 | 2 | 4 | 0 | 2 | 9 | 0 | 0 | 3 | 2 | 0 | 0 | 10 | 3 | 1 |
| Baby_Crying | 0 | 7 | **38** | 0 | 3 | 0 | 2 | 2 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 2 | 0 | 0 | **48** | 30 | 0 | 4 | 11 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 45 | 0 |
| Dog_Bark | 0 | 8 | 0 | 1 | **75** | 0 | 4 | 15 | 0 | 0 | 9 | 2 | 0 | 1 | 6 | 1 | 1 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| Drilling | 6 | 0 | 0 | 0 | 12 | 0 | **59** | 18 | 0 | 2 | 12 | 8 | 0 | 0 | 7 | 26 | 0 |
| Footsteps | 3 | 0 | 0 | 0 | 2 | 0 | 0 | **68** | 0 | 1 | 0 | 1 | 1 | 0 | 2 | 1 | 0 |
| Glass_Breaking | 0 | 4 | 0 | 0 | 2 | 0 | 1 | 27 | **7** | 1 | 0 | 5 | 0 | 0 | 1 | 0 | 0 |
| Gunshot | 26 | 0 | 0 | 0 | 2 | 0 | 1 | 19 | 0 | **6** | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| Help_Shouting | 0 | 10 | 0 | 1 | 3 | 0 | 2 | 6 | 0 | 1 | **27** | 0 | 0 | 0 | 4 | 0 | 0 |
| Jackhammer | 0 | 0 | 0 | 11 | 0 | 0 | 4 | 13 | 0 | 0 | 0 | **99** | 0 | 0 | 0 | 6 | 0 |
| Knocking | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 21 | 0 | 0 | 0 | 0 | **21** | 0 | 1 | 1 | 0 |
| Motorcycle | 1 | 0 | 0 | 8 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 5 | 0 | **4** | 2 | 6 | 0 |
| Siren | 0 | 3 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | **132** | 0 | 0 |
| Train | 13 | 4 | 0 | 1 | 5 | 0 | 4 | 23 | 0 | 0 | 3 | 7 | 1 | 2 | 1 | **15** | 0 |
| Vehicle_Horn | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 3 | **25** |

## Confusion matrix - clip_majority_vote (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **21** | 0 | 0 | 1 | 1 | 0 | 1 | 8 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 17 | 0 |
| Alarm | 3 | **55** | 0 | 1 | 4 | 0 | 2 | 15 | 0 | 0 | 7 | 1 | 0 | 0 | 4 | 11 | 0 |
| Baby_Crying | 0 | 10 | **37** | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 2 | 0 | 0 | **45** | 30 | 0 | 2 | 9 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 53 | 0 |
| Dog_Bark | 1 | 5 | 0 | 1 | **69** | 0 | 4 | 24 | 0 | 0 | 7 | 2 | 0 | 1 | 3 | 6 | 0 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 5 | 0 | 0 | 0 | 16 | 0 | **59** | 13 | 0 | 1 | 14 | 5 | 0 | 0 | 4 | 33 | 0 |
| Footsteps | 0 | 0 | 0 | 0 | 3 | 0 | 0 | **69** | 0 | 1 | 0 | 1 | 0 | 0 | 1 | 4 | 0 |
| Glass_Breaking | 0 | 2 | 0 | 0 | 1 | 0 | 3 | 29 | **8** | 0 | 0 | 3 | 0 | 0 | 0 | 2 | 0 |
| Gunshot | 13 | 0 | 0 | 0 | 2 | 0 | 0 | 23 | 0 | **14** | 0 | 0 | 0 | 0 | 0 | 4 | 0 |
| Help_Shouting | 0 | 7 | 0 | 0 | 2 | 0 | 2 | 7 | 0 | 0 | **30** | 0 | 1 | 0 | 3 | 2 | 0 |
| Jackhammer | 0 | 0 | 0 | 5 | 0 | 0 | 4 | 14 | 0 | 0 | 0 | **76** | 0 | 0 | 0 | 34 | 0 |
| Knocking | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | **16** | 0 | 0 | 2 | 0 |
| Motorcycle | 1 | 1 | 0 | 4 | 2 | 0 | 0 | 6 | 0 | 0 | 0 | 2 | 0 | **3** | 0 | 16 | 0 |
| Siren | 0 | 8 | 0 | 0 | 10 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | **118** | 0 | 0 |
| Train | 8 | 3 | 0 | 1 | 3 | 0 | 3 | 25 | 0 | 0 | 2 | 3 | 1 | 1 | 0 | **29** | 0 |
| Vehicle_Horn | 0 | 3 | 0 | 1 | 0 | 0 | 1 | 5 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 8 | **20** |

## Top 15 clip-level confusions (mean-softmax)

largest off-diagonal cells of the clip_mean_softmax matrix.

| true | predicted | clips |
|---|---|---:|
| Car_Engine | Train | 53 |
| Jackhammer | Train | 41 |
| Drilling | Train | 35 |
| Car_Engine | Dog_Bark | 30 |
| Glass_Breaking | Footsteps | 28 |
| Knocking | Footsteps | 26 |
| Train | Footsteps | 23 |
| Gunshot | Footsteps | 20 |
| Dog_Bark | Footsteps | 18 |
| Gunshot | Aircraft | 17 |
| Drilling | Dog_Bark | 16 |
| Motorcycle | Train | 16 |
| Aircraft | Train | 14 |
| Drilling | Help_Shouting | 13 |
| Alarm | Footsteps | 12 |

## Baby_Crying clip recall by sample-rate group (mean-softmax)

ledger sample_rate is the original clip sample rate; clips with no prediction (zero-window clips) are excluded.

| sample-rate group | clips (total) | clips (assessed) | correct | recall |
|---:|---:|---:|---:|---:|
| below 16000 | 32 | 32 | 32 | 1.0000 |
| 16000 and above | 23 | 23 | 5 | 0.2174 |

## Misclassified clips - within vs cross dominant source_dataset (mean-softmax)

A class's dominant source_dataset is the dataset with the most clips of that class in the ledger. An error is within-dataset when the predicted class has the same dominant source_dataset as the true class, and cross-dataset otherwise.

- misclassified clips: **652**
- within-dataset errors: **330**
- cross-dataset errors: **322**
- errors with no dominant dataset for either class: **0**

| true class | predicted class | kind | clips |
|---|---|---:|---:|
| Car_Engine | Train | cross | 53 |
| Jackhammer | Train | cross | 41 |
| Drilling | Train | cross | 35 |
| Car_Engine | Dog_Bark | within | 30 |
| Glass_Breaking | Footsteps | within | 28 |
| Knocking | Footsteps | within | 26 |
| Train | Footsteps | within | 23 |
| Gunshot | Footsteps | within | 20 |
| Dog_Bark | Footsteps | cross | 18 |
| Gunshot | Aircraft | within | 17 |
| Motorcycle | Train | within | 16 |
| Drilling | Dog_Bark | within | 16 |
| Aircraft | Train | within | 14 |
| Drilling | Help_Shouting | cross | 13 |
| Alarm | Footsteps | within | 12 |
| Drilling | Footsteps | cross | 12 |
| Baby_Crying | Alarm | cross | 11 |
| Alarm | Train | within | 11 |
| Jackhammer | Footsteps | cross | 10 |
| Aircraft | Footsteps | within | 9 |
| Car_Engine | Footsteps | cross | 9 |
| Help_Shouting | Alarm | within | 8 |
| Siren | Dog_Bark | within | 8 |
| Dog_Bark | Alarm | cross | 8 |
| Train | Aircraft | within | 7 |
| Vehicle_Horn | Train | cross | 7 |
| Dog_Bark | Help_Shouting | cross | 7 |
| Dog_Bark | Train | cross | 7 |
| Baby_Crying | Help_Shouting | cross | 6 |
| Alarm | Dog_Bark | cross | 6 |
| Help_Shouting | Footsteps | within | 6 |
| Drilling | Siren | within | 6 |
| Train | Alarm | within | 5 |
| Footsteps | Train | within | 5 |
| Motorcycle | Car_Engine | cross | 5 |
| Doorbell | Alarm | within | 5 |
| Jackhammer | Car_Engine | within | 5 |
| Siren | Alarm | cross | 5 |
| Vehicle_Horn | Footsteps | cross | 4 |
| Alarm | Siren | cross | 4 |
| Motorcycle | Footsteps | within | 4 |
| Glass_Breaking | Drilling | cross | 4 |
| Alarm | Help_Shouting | within | 4 |
| Dog_Bark | Drilling | within | 4 |
| Jackhammer | Drilling | within | 4 |
| Drilling | Jackhammer | within | 4 |
| Drilling | Aircraft | cross | 4 |
| Glass_Breaking | Alarm | within | 3 |
| Help_Shouting | Siren | cross | 3 |
| Footsteps | Dog_Bark | cross | 3 |
| Gunshot | Dog_Bark | cross | 3 |
| Train | Dog_Bark | cross | 3 |
| Train | Drilling | cross | 3 |
| Alarm | Aircraft | within | 3 |
| Help_Shouting | Train | within | 3 |
| Gunshot | Train | within | 3 |
| Vehicle_Horn | Alarm | cross | 3 |
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
| Alarm | Drilling | cross | 1 |
| Train | Knocking | within | 1 |
| Motorcycle | Jackhammer | cross | 1 |
| Motorcycle | Alarm | within | 1 |
| Glass_Breaking | Dog_Bark | cross | 1 |
| Footsteps | Jackhammer | cross | 1 |
| Aircraft | Drilling | cross | 1 |
| Footsteps | Gunshot | within | 1 |
| Train | Motorcycle | within | 1 |
| Knocking | Alarm | within | 1 |
| Vehicle_Horn | Siren | within | 1 |
| Motorcycle | Dog_Bark | cross | 1 |
| Vehicle_Horn | Drilling | within | 1 |
| Footsteps | Siren | cross | 1 |
| Aircraft | Dog_Bark | cross | 1 |
| Glass_Breaking | Train | within | 1 |
| Vehicle_Horn | Motorcycle | cross | 1 |
| Knocking | Train | within | 1 |
| Alarm | Car_Engine | cross | 1 |
| Motorcycle | Aircraft | within | 1 |
| Doorbell | Drilling | cross | 1 |
| Aircraft | Jackhammer | cross | 1 |
| Aircraft | Car_Engine | cross | 1 |
| Train | Car_Engine | cross | 1 |
| Help_Shouting | Drilling | cross | 1 |
| Car_Engine | Motorcycle | cross | 1 |
| Siren | Footsteps | cross | 1 |
| Car_Engine | Gunshot | cross | 1 |
| Dog_Bark | Aircraft | cross | 1 |
| Dog_Bark | Motorcycle | cross | 1 |
| Drilling | Gunshot | cross | 1 |

## Per-class clip recall by source_dataset (mean-softmax)

classes with more than one source_dataset in the validation split.

| class | source_dataset | clips | correct | recall |
|---|---|---:|---:|---:|
| Aircraft | ESC-50 | 13 | 5 | 0.3846 |
| Aircraft | FSD50K | 37 | 18 | 0.4865 |
| Alarm | ESC-50 | 3 | 3 | 1.0000 |
| Alarm | FSD50K | 100 | 56 | 0.5600 |
| Baby_Crying | ESC-50 | 17 | 0 | 0.0000 |
| Baby_Crying | owlgebra_babycry | 38 | 37 | 0.9737 |
| Footsteps | ESC-50 | 6 | 6 | 1.0000 |
| Footsteps | FSD50K | 73 | 62 | 0.8493 |
| Glass_Breaking | ESC-50 | 4 | 1 | 0.2500 |
| Glass_Breaking | FSD50K | 44 | 8 | 0.1818 |
| Gunshot | FSD50K | 33 | 9 | 0.2727 |
| Gunshot | UrbanSound8K | 23 | 4 | 0.1739 |
| Knocking | ESC-50 | 4 | 1 | 0.2500 |
| Knocking | FSD50K | 41 | 14 | 0.3415 |
| Train | ESC-50 | 6 | 5 | 0.8333 |
| Train | FSD50K | 73 | 25 | 0.3425 |
| Vehicle_Horn | ESC-50 | 3 | 0 | 0.0000 |
| Vehicle_Horn | FSD50K | 11 | 1 | 0.0909 |
| Vehicle_Horn | UrbanSound8K | 25 | 21 | 0.8400 |

## Clips with zero windows

No validation clip has zero windows.

model.predict: 1.1 s