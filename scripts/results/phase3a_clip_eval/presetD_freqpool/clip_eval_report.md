# Phase 3A Clip-Level Evaluation (analysis only)

Model: `data\models\presetD_freqpool.keras` (360,514 bytes, 25,937 parameters). Analysis preset `D`: 500 ms window, 50% overlap, FFT 1024, hop 512, 64 Mel -> (64, 14, 1) by the extraction formula. Frozen Phase 2B clip-level split, window and clip levels reported side by side. No retraining; model, preprocessing, configuration and dataset unchanged. The test split is never opened.

- validation windows: **29,363**
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
| samples | 29,363 | 29,363 | 1,618 | 1,618 | 1,618 |
| correct | 13,668 | 13,668 | 890 | 877 | 877 |
| accuracy | 0.4655 | 0.4655 | 0.5501 | 0.5420 | 0.5420 |
| macro precision | 0.4560 | 0.4560 | 0.5505 | 0.5380 | 0.5483 |
| macro recall | 0.4177 | 0.4177 | 0.5005 | 0.4845 | 0.4928 |
| macro F1 | 0.4237 | 0.4237 | 0.4940 | 0.4785 | 0.4885 |
| weighted precision | 0.4655 | 0.4655 | 0.6261 | 0.6048 | 0.6172 |
| weighted recall | 0.4655 | 0.4655 | 0.5501 | 0.5420 | 0.5420 |
| weighted F1 | 0.4556 | 0.4556 | 0.5557 | 0.5403 | 0.5476 |

## Per-class metrics - window

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 2031 | 0.4067 | 0.5185 | 0.4558 |
| 1 | Alarm | 3371 | 0.4289 | 0.4533 | 0.4407 |
| 2 | Baby_Crying | 1215 | 0.8677 | 0.7235 | 0.7890 |
| 3 | Car_Engine | 2134 | 0.4512 | 0.3074 | 0.3657 |
| 4 | Dog_Bark | 1827 | 0.4847 | 0.4412 | 0.4619 |
| 5 | Doorbell | 171 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 2218 | 0.5022 | 0.5739 | 0.5357 |
| 7 | Footsteps | 2383 | 0.4400 | 0.6744 | 0.5326 |
| 8 | Glass_Breaking | 842 | 0.5491 | 0.2922 | 0.3814 |
| 9 | Gunshot | 1151 | 0.5778 | 0.3579 | 0.4421 |
| 10 | Help_Shouting | 1451 | 0.2839 | 0.2901 | 0.2870 |
| 11 | Jackhammer | 1972 | 0.5618 | 0.7794 | 0.6529 |
| 12 | Knocking | 806 | 0.6875 | 0.3412 | 0.4561 |
| 13 | Motorcycle | 1515 | 0.4396 | 0.3485 | 0.3888 |
| 14 | Siren | 2032 | 0.6160 | 0.6127 | 0.6144 |
| 15 | Train | 3411 | 0.3232 | 0.3412 | 0.3320 |
| 16 | Vehicle_Horn | 833 | 0.1310 | 0.0456 | 0.0677 |

## Per-class metrics - clip_mean_softmax

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 51 | 0.3256 | 0.5490 | 0.4088 |
| 1 | Alarm | 120 | 0.3438 | 0.6417 | 0.4477 |
| 2 | Baby_Crying | 70 | 0.8929 | 0.7143 | 0.7937 |
| 3 | Car_Engine | 146 | 0.7460 | 0.3219 | 0.4498 |
| 4 | Dog_Bark | 158 | 0.8351 | 0.5127 | 0.6353 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 168 | 0.6000 | 0.5714 | 0.5854 |
| 7 | Footsteps | 87 | 0.3824 | 0.7471 | 0.5058 |
| 8 | Glass_Breaking | 75 | 0.8222 | 0.4933 | 0.6167 |
| 9 | Gunshot | 122 | 0.8831 | 0.5574 | 0.6834 |
| 10 | Help_Shouting | 76 | 0.3429 | 0.4737 | 0.3978 |
| 11 | Jackhammer | 148 | 0.7289 | 0.8176 | 0.7707 |
| 12 | Knocking | 63 | 0.7949 | 0.4921 | 0.6078 |
| 13 | Motorcycle | 35 | 0.3478 | 0.4571 | 0.3951 |
| 14 | Siren | 142 | 0.8095 | 0.7183 | 0.7612 |
| 15 | Train | 80 | 0.2185 | 0.4125 | 0.2857 |
| 16 | Vehicle_Horn | 69 | 0.2857 | 0.0290 | 0.0526 |

## Per-class metrics - clip_topk_mean

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 51 | 0.3226 | 0.5882 | 0.4167 |
| 1 | Alarm | 120 | 0.3404 | 0.6667 | 0.4507 |
| 2 | Baby_Crying | 70 | 0.8621 | 0.7143 | 0.7812 |
| 3 | Car_Engine | 146 | 0.6957 | 0.3288 | 0.4465 |
| 4 | Dog_Bark | 158 | 0.8404 | 0.5000 | 0.6270 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 168 | 0.6037 | 0.5893 | 0.5964 |
| 7 | Footsteps | 87 | 0.3702 | 0.7701 | 0.5000 |
| 8 | Glass_Breaking | 75 | 0.8500 | 0.4533 | 0.5913 |
| 9 | Gunshot | 122 | 0.8228 | 0.5328 | 0.6468 |
| 10 | Help_Shouting | 76 | 0.3736 | 0.4474 | 0.4072 |
| 11 | Jackhammer | 148 | 0.6332 | 0.8514 | 0.7262 |
| 12 | Knocking | 63 | 0.8158 | 0.4921 | 0.6139 |
| 13 | Motorcycle | 35 | 0.3333 | 0.3429 | 0.3380 |
| 14 | Siren | 142 | 0.7343 | 0.7394 | 0.7368 |
| 15 | Train | 80 | 0.1477 | 0.1625 | 0.1548 |
| 16 | Vehicle_Horn | 69 | 0.4000 | 0.0580 | 0.1013 |

## Per-class metrics - clip_majority_vote

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 51 | 0.3372 | 0.5686 | 0.4234 |
| 1 | Alarm | 120 | 0.3172 | 0.6000 | 0.4150 |
| 2 | Baby_Crying | 70 | 0.9107 | 0.7286 | 0.8095 |
| 3 | Car_Engine | 146 | 0.7344 | 0.3219 | 0.4476 |
| 4 | Dog_Bark | 158 | 0.7714 | 0.5127 | 0.6160 |
| 5 | Doorbell | 8 | 0.0000 | 0.0000 | 0.0000 |
| 6 | Drilling | 168 | 0.5988 | 0.5774 | 0.5879 |
| 7 | Footsteps | 87 | 0.3757 | 0.7471 | 0.5000 |
| 8 | Glass_Breaking | 75 | 0.8182 | 0.4800 | 0.6050 |
| 9 | Gunshot | 122 | 0.8684 | 0.5410 | 0.6667 |
| 10 | Help_Shouting | 76 | 0.3196 | 0.4079 | 0.3584 |
| 11 | Jackhammer | 148 | 0.7193 | 0.8311 | 0.7712 |
| 12 | Knocking | 63 | 0.7895 | 0.4762 | 0.5941 |
| 13 | Motorcycle | 35 | 0.3721 | 0.4571 | 0.4103 |
| 14 | Siren | 142 | 0.7984 | 0.6972 | 0.7444 |
| 15 | Train | 80 | 0.2153 | 0.3875 | 0.2768 |
| 16 | Vehicle_Horn | 69 | 0.3750 | 0.0435 | 0.0779 |

## Confusion matrix - window (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **1053** | 24 | 0 | 120 | 81 | 0 | 40 | 132 | 0 | 19 | 3 | 142 | 6 | 103 | 27 | 280 | 1 |
| Alarm | 282 | **1528** | 1 | 61 | 147 | 0 | 167 | 255 | 17 | 8 | 108 | 77 | 13 | 47 | 279 | 311 | 70 |
| Baby_Crying | 0 | 186 | **879** | 0 | 47 | 0 | 14 | 13 | 7 | 0 | 60 | 0 | 0 | 0 | 1 | 2 | 6 |
| Car_Engine | 89 | 0 | 121 | **656** | 4 | 0 | 33 | 7 | 0 | 10 | 332 | 33 | 0 | 267 | 52 | 530 | 0 |
| Dog_Bark | 13 | 152 | 0 | 75 | **806** | 0 | 92 | 91 | 14 | 3 | 197 | 37 | 3 | 4 | 135 | 202 | 3 |
| Doorbell | 0 | 109 | 0 | 3 | 1 | **0** | 8 | 47 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| Drilling | 192 | 65 | 0 | 2 | 6 | 0 | **1273** | 36 | 2 | 7 | 131 | 279 | 0 | 13 | 26 | 131 | 55 |
| Footsteps | 51 | 32 | 1 | 20 | 134 | 0 | 20 | **1607** | 31 | 52 | 11 | 71 | 39 | 33 | 89 | 189 | 3 |
| Glass_Breaking | 9 | 46 | 0 | 5 | 11 | 0 | 92 | 231 | **246** | 42 | 3 | 64 | 3 | 6 | 2 | 82 | 0 |
| Gunshot | 299 | 17 | 0 | 1 | 14 | 0 | 23 | 246 | 48 | **412** | 5 | 13 | 42 | 2 | 4 | 25 | 0 |
| Help_Shouting | 5 | 424 | 2 | 38 | 79 | 0 | 131 | 66 | 28 | 9 | **421** | 7 | 8 | 18 | 60 | 127 | 28 |
| Jackhammer | 0 | 0 | 0 | 38 | 0 | 0 | 159 | 188 | 0 | 0 | 0 | **1537** | 0 | 20 | 0 | 30 | 0 |
| Knocking | 22 | 15 | 0 | 1 | 73 | 0 | 0 | 269 | 19 | 81 | 4 | 30 | **275** | 3 | 0 | 14 | 0 |
| Motorcycle | 82 | 62 | 4 | 202 | 68 | 0 | 49 | 127 | 1 | 9 | 2 | 110 | 5 | **528** | 9 | 257 | 0 |
| Siren | 37 | 429 | 0 | 0 | 90 | 0 | 4 | 0 | 0 | 4 | 68 | 0 | 4 | 5 | **1245** | 64 | 82 |
| Train | 447 | 75 | 5 | 202 | 88 | 0 | 387 | 303 | 34 | 57 | 135 | 321 | 2 | 137 | 50 | **1164** | 4 |
| Vehicle_Horn | 8 | 399 | 0 | 30 | 14 | 0 | 43 | 34 | 0 | 0 | 3 | 15 | 0 | 15 | 41 | 193 | **38** |

## Confusion matrix - clip_mean_softmax (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **28** | 0 | 0 | 2 | 2 | 0 | 1 | 4 | 0 | 0 | 0 | 3 | 0 | 2 | 0 | 9 | 0 |
| Alarm | 6 | **77** | 0 | 1 | 4 | 0 | 5 | 5 | 0 | 0 | 7 | 1 | 0 | 2 | 3 | 8 | 1 |
| Baby_Crying | 0 | 14 | **50** | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 7 | 0 | 6 | **47** | 0 | 0 | 2 | 1 | 0 | 0 | 25 | 0 | 0 | 18 | 2 | 38 | 0 |
| Dog_Bark | 1 | 16 | 0 | 6 | **81** | 0 | 8 | 4 | 0 | 0 | 13 | 2 | 0 | 0 | 8 | 19 | 0 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 14 | 6 | 0 | 0 | 0 | 0 | **96** | 0 | 0 | 0 | 12 | 23 | 0 | 0 | 1 | 13 | 3 |
| Footsteps | 1 | 0 | 0 | 0 | 2 | 0 | 0 | **65** | 2 | 3 | 0 | 2 | 5 | 2 | 3 | 2 | 0 |
| Glass_Breaking | 1 | 5 | 0 | 0 | 1 | 0 | 8 | 17 | **37** | 1 | 0 | 3 | 0 | 0 | 0 | 2 | 0 |
| Gunshot | 13 | 0 | 0 | 0 | 0 | 0 | 2 | 30 | 4 | **68** | 0 | 2 | 3 | 0 | 0 | 0 | 0 |
| Help_Shouting | 0 | 29 | 0 | 0 | 0 | 0 | 4 | 1 | 0 | 0 | **36** | 0 | 0 | 0 | 1 | 4 | 1 |
| Jackhammer | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 10 | 0 | 0 | 0 | **121** | 0 | 1 | 0 | 0 | 0 |
| Knocking | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 24 | 1 | 5 | 0 | 1 | **31** | 0 | 0 | 0 | 0 |
| Motorcycle | 2 | 0 | 0 | 4 | 2 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 0 | **16** | 0 | 7 | 0 |
| Siren | 4 | 30 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 1 | **102** | 1 | 0 |
| Train | 9 | 3 | 0 | 2 | 1 | 0 | 11 | 6 | 1 | 0 | 5 | 5 | 0 | 3 | 1 | **33** | 0 |
| Vehicle_Horn | 0 | 39 | 0 | 1 | 1 | 0 | 4 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 5 | 15 | **2** |

## Confusion matrix - clip_topk_mean (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **30** | 0 | 0 | 3 | 2 | 0 | 1 | 4 | 0 | 1 | 0 | 7 | 0 | 1 | 0 | 2 | 0 |
| Alarm | 8 | **80** | 0 | 2 | 3 | 0 | 4 | 5 | 0 | 0 | 4 | 2 | 0 | 0 | 7 | 3 | 2 |
| Baby_Crying | 0 | 15 | **50** | 0 | 1 | 0 | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 4 | 0 | 8 | **48** | 0 | 0 | 2 | 1 | 0 | 0 | 23 | 3 | 0 | 18 | 5 | 34 | 0 |
| Dog_Bark | 2 | 16 | 0 | 5 | **79** | 0 | 6 | 6 | 0 | 0 | 13 | 5 | 0 | 0 | 11 | 15 | 0 |
| Doorbell | 0 | 6 | 0 | 0 | 0 | **0** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 13 | 6 | 0 | 0 | 0 | 0 | **99** | 0 | 0 | 0 | 11 | 26 | 0 | 0 | 3 | 7 | 3 |
| Footsteps | 2 | 0 | 0 | 0 | 2 | 0 | 0 | **67** | 1 | 3 | 0 | 2 | 5 | 1 | 3 | 1 | 0 |
| Glass_Breaking | 0 | 4 | 0 | 0 | 1 | 0 | 10 | 19 | **34** | 1 | 0 | 5 | 0 | 0 | 0 | 1 | 0 |
| Gunshot | 15 | 1 | 0 | 0 | 1 | 0 | 3 | 30 | 3 | **65** | 0 | 2 | 2 | 0 | 0 | 0 | 0 |
| Help_Shouting | 0 | 31 | 0 | 1 | 0 | 0 | 4 | 1 | 0 | 0 | **34** | 1 | 0 | 0 | 3 | 0 | 1 |
| Jackhammer | 0 | 0 | 0 | 0 | 0 | 0 | 13 | 9 | 0 | 0 | 0 | **126** | 0 | 0 | 0 | 0 | 0 |
| Knocking | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 | 1 | 6 | 0 | 1 | **31** | 0 | 0 | 0 | 0 |
| Motorcycle | 1 | 1 | 0 | 5 | 1 | 0 | 3 | 4 | 0 | 0 | 0 | 7 | 0 | **12** | 0 | 1 | 0 |
| Siren | 4 | 31 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | **105** | 0 | 0 |
| Train | 14 | 5 | 0 | 4 | 2 | 0 | 12 | 8 | 1 | 3 | 4 | 11 | 0 | 2 | 1 | **13** | 0 |
| Vehicle_Horn | 0 | 39 | 0 | 1 | 1 | 0 | 4 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 5 | 11 | **4** |

## Confusion matrix - clip_majority_vote (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **29** | 0 | 0 | 2 | 3 | 0 | 2 | 3 | 0 | 0 | 0 | 2 | 0 | 2 | 0 | 8 | 0 |
| Alarm | 7 | **72** | 0 | 1 | 5 | 0 | 5 | 7 | 0 | 0 | 6 | 2 | 0 | 1 | 5 | 8 | 1 |
| Baby_Crying | 0 | 15 | **51** | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 7 | 0 | 5 | **47** | 0 | 0 | 2 | 1 | 0 | 0 | 26 | 0 | 0 | 18 | 1 | 39 | 0 |
| Dog_Bark | 1 | 15 | 0 | 6 | **81** | 0 | 8 | 3 | 0 | 0 | 13 | 2 | 0 | 0 | 8 | 21 | 0 |
| Doorbell | 0 | 5 | 0 | 0 | 0 | **0** | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 14 | 6 | 0 | 0 | 0 | 0 | **97** | 0 | 0 | 0 | 14 | 24 | 0 | 0 | 1 | 9 | 3 |
| Footsteps | 1 | 0 | 0 | 0 | 2 | 0 | 0 | **65** | 2 | 3 | 0 | 2 | 5 | 2 | 2 | 3 | 0 |
| Glass_Breaking | 0 | 6 | 0 | 0 | 1 | 0 | 8 | 16 | **36** | 2 | 0 | 4 | 0 | 0 | 0 | 2 | 0 |
| Gunshot | 13 | 0 | 0 | 0 | 1 | 0 | 2 | 30 | 4 | **66** | 0 | 3 | 3 | 0 | 0 | 0 | 0 |
| Help_Shouting | 0 | 31 | 0 | 0 | 1 | 0 | 4 | 4 | 0 | 0 | **31** | 0 | 0 | 0 | 1 | 3 | 1 |
| Jackhammer | 0 | 0 | 0 | 0 | 0 | 0 | 15 | 10 | 0 | 0 | 0 | **123** | 0 | 0 | 0 | 0 | 0 |
| Knocking | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 24 | 1 | 5 | 0 | 1 | **30** | 0 | 0 | 0 | 0 |
| Motorcycle | 1 | 0 | 0 | 4 | 2 | 0 | 1 | 1 | 0 | 0 | 0 | 3 | 0 | **16** | 0 | 7 | 0 |
| Siren | 4 | 33 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | **99** | 1 | 0 |
| Train | 9 | 4 | 0 | 3 | 1 | 0 | 13 | 6 | 1 | 0 | 4 | 5 | 0 | 2 | 1 | **31** | 0 |
| Vehicle_Horn | 0 | 40 | 0 | 1 | 1 | 0 | 3 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 6 | 12 | **3** |

## Top 15 clip-level confusions (mean-softmax)

largest off-diagonal cells of the clip_mean_softmax matrix.

| true | predicted | clips |
|---|---|---:|
| Vehicle_Horn | Alarm | 39 |
| Car_Engine | Train | 38 |
| Gunshot | Footsteps | 30 |
| Siren | Alarm | 30 |
| Help_Shouting | Alarm | 29 |
| Car_Engine | Help_Shouting | 25 |
| Knocking | Footsteps | 24 |
| Drilling | Jackhammer | 23 |
| Dog_Bark | Train | 19 |
| Car_Engine | Motorcycle | 18 |
| Glass_Breaking | Footsteps | 17 |
| Dog_Bark | Alarm | 16 |
| Jackhammer | Drilling | 16 |
| Vehicle_Horn | Train | 15 |
| Baby_Crying | Alarm | 14 |

## Baby_Crying clip recall by sample-rate group (mean-softmax)

ledger sample_rate is the original clip sample rate; clips with no prediction (zero-window clips) are excluded.

| sample-rate group | clips (total) | clips (assessed) | correct | recall |
|---:|---:|---:|---:|---:|
| below 16000 | 32 | 32 | 32 | 1.0000 |
| 16000 and above | 38 | 38 | 18 | 0.4737 |

## Misclassified clips - within vs cross dominant source_dataset (mean-softmax)

A class's dominant source_dataset is the dataset with the most clips of that class in the ledger. An error is within-dataset when the predicted class has the same dominant source_dataset as the true class, and cross-dataset otherwise.

- misclassified clips: **728**
- within-dataset errors: **324**
- cross-dataset errors: **404**
- errors with no dominant dataset for either class: **0**

| true class | predicted class | kind | clips |
|---|---|---:|---:|
| Vehicle_Horn | Alarm | cross | 39 |
| Car_Engine | Train | cross | 38 |
| Gunshot | Footsteps | within | 30 |
| Siren | Alarm | cross | 30 |
| Help_Shouting | Alarm | within | 29 |
| Car_Engine | Help_Shouting | cross | 25 |
| Knocking | Footsteps | within | 24 |
| Drilling | Jackhammer | within | 23 |
| Dog_Bark | Train | cross | 19 |
| Car_Engine | Motorcycle | cross | 18 |
| Glass_Breaking | Footsteps | within | 17 |
| Dog_Bark | Alarm | cross | 16 |
| Jackhammer | Drilling | within | 16 |
| Vehicle_Horn | Train | cross | 15 |
| Baby_Crying | Alarm | cross | 14 |
| Drilling | Aircraft | cross | 14 |
| Gunshot | Aircraft | within | 13 |
| Dog_Bark | Help_Shouting | cross | 13 |
| Drilling | Train | cross | 13 |
| Drilling | Help_Shouting | cross | 12 |
| Train | Drilling | cross | 11 |
| Jackhammer | Footsteps | cross | 10 |
| Train | Aircraft | within | 9 |
| Aircraft | Train | within | 9 |
| Glass_Breaking | Drilling | cross | 8 |
| Alarm | Train | within | 8 |
| Dog_Bark | Siren | within | 8 |
| Dog_Bark | Drilling | within | 8 |
| Alarm | Help_Shouting | within | 7 |
| Motorcycle | Train | within | 7 |
| Car_Engine | Aircraft | cross | 7 |
| Train | Footsteps | within | 6 |
| Alarm | Aircraft | within | 6 |
| Dog_Bark | Car_Engine | within | 6 |
| Car_Engine | Baby_Crying | cross | 6 |
| Drilling | Alarm | cross | 6 |
| Alarm | Drilling | cross | 5 |
| Footsteps | Knocking | within | 5 |
| Knocking | Gunshot | within | 5 |
| Train | Jackhammer | cross | 5 |
| Alarm | Footsteps | within | 5 |
| Glass_Breaking | Alarm | within | 5 |
| Vehicle_Horn | Siren | within | 5 |
| Train | Help_Shouting | within | 5 |
| Doorbell | Alarm | within | 5 |
| Gunshot | Glass_Breaking | within | 4 |
| Help_Shouting | Train | within | 4 |
| Alarm | Dog_Bark | cross | 4 |
| Aircraft | Footsteps | within | 4 |
| Help_Shouting | Drilling | cross | 4 |
| Motorcycle | Car_Engine | cross | 4 |
| Vehicle_Horn | Drilling | within | 4 |
| Dog_Bark | Footsteps | cross | 4 |
| Siren | Aircraft | cross | 4 |
| Aircraft | Jackhammer | cross | 3 |
| Baby_Crying | Help_Shouting | cross | 3 |
| Motorcycle | Jackhammer | cross | 3 |
| Alarm | Siren | cross | 3 |
| Footsteps | Gunshot | within | 3 |
| Train | Alarm | within | 3 |
| Gunshot | Knocking | within | 3 |
| Glass_Breaking | Jackhammer | cross | 3 |
| Footsteps | Siren | cross | 3 |
| Train | Motorcycle | within | 3 |
| Siren | Help_Shouting | cross | 3 |
| Drilling | Vehicle_Horn | within | 3 |
| Footsteps | Dog_Bark | cross | 2 |
| Gunshot | Drilling | cross | 2 |
| Aircraft | Dog_Bark | cross | 2 |
| Aircraft | Motorcycle | within | 2 |
| Motorcycle | Dog_Bark | cross | 2 |
| Footsteps | Jackhammer | cross | 2 |
| Footsteps | Glass_Breaking | within | 2 |
| Footsteps | Motorcycle | within | 2 |
| Train | Car_Engine | cross | 2 |
| Alarm | Motorcycle | within | 2 |
| Aircraft | Car_Engine | cross | 2 |
| Glass_Breaking | Train | within | 2 |
| Doorbell | Footsteps | within | 2 |
| Footsteps | Train | within | 2 |
| Gunshot | Jackhammer | cross | 2 |
| Motorcycle | Aircraft | within | 2 |
| Car_Engine | Drilling | within | 2 |
| Dog_Bark | Jackhammer | within | 2 |
| Car_Engine | Siren | within | 2 |
| Vehicle_Horn | Car_Engine | within | 1 |
| Baby_Crying | Dog_Bark | cross | 1 |
| Alarm | Jackhammer | cross | 1 |
| Help_Shouting | Siren | cross | 1 |
| Alarm | Vehicle_Horn | cross | 1 |
| Train | Dog_Bark | cross | 1 |
| Glass_Breaking | Dog_Bark | cross | 1 |
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
| Car_Engine | Footsteps | cross | 1 |
| Siren | Dog_Bark | within | 1 |
| Siren | Train | cross | 1 |
| Jackhammer | Motorcycle | cross | 1 |
| Dog_Bark | Aircraft | cross | 1 |
| Vehicle_Horn | Dog_Bark | within | 1 |
| Drilling | Siren | within | 1 |
| Vehicle_Horn | Help_Shouting | cross | 1 |
| Siren | Motorcycle | cross | 1 |
| Baby_Crying | Footsteps | cross | 1 |
| Baby_Crying | Drilling | cross | 1 |

## Per-class clip recall by source_dataset (mean-softmax)

classes with more than one source_dataset in the validation split.

| class | source_dataset | clips | correct | recall |
|---|---|---:|---:|---:|
| Aircraft | ESC-50 | 13 | 7 | 0.5385 |
| Aircraft | FSD50K | 38 | 21 | 0.5526 |
| Alarm | ESC-50 | 3 | 2 | 0.6667 |
| Alarm | FSD50K | 117 | 75 | 0.6410 |
| Baby_Crying | ESC-50 | 17 | 0 | 0.0000 |
| Baby_Crying | owlgebra_babycry | 53 | 50 | 0.9434 |
| Footsteps | ESC-50 | 6 | 5 | 0.8333 |
| Footsteps | FSD50K | 81 | 60 | 0.7407 |
| Glass_Breaking | ESC-50 | 4 | 2 | 0.5000 |
| Glass_Breaking | FSD50K | 71 | 35 | 0.4930 |
| Gunshot | FSD50K | 61 | 28 | 0.4590 |
| Gunshot | UrbanSound8K | 61 | 40 | 0.6557 |
| Knocking | ESC-50 | 4 | 1 | 0.2500 |
| Knocking | FSD50K | 59 | 30 | 0.5085 |
| Train | ESC-50 | 6 | 4 | 0.6667 |
| Train | FSD50K | 74 | 29 | 0.3919 |
| Vehicle_Horn | ESC-50 | 3 | 0 | 0.0000 |
| Vehicle_Horn | FSD50K | 11 | 1 | 0.0909 |
| Vehicle_Horn | UrbanSound8K | 55 | 1 | 0.0182 |

## Clips with zero windows

No validation clip has zero windows.

model.predict: 3.0 s