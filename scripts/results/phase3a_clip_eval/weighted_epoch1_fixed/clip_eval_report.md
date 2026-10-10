# Phase 3A Clip-Level Evaluation (analysis only)

Model: `data\models\archive\weighted_epoch1.keras` (729,178 bytes, 56,657 parameters). Analysis preset `A`: locked baseline: 200 ms, 50% overlap, FFT 1024, hop 512, 64 Mel -> (64, 5, 1). Frozen Phase 2B clip-level split, window and clip levels reported side by side. No retraining; model, preprocessing, configuration and dataset unchanged. The test split is never opened.

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
| correct | 28,469 | 28,469 | 876 | 855 | 853 |
| accuracy | 0.3804 | 0.3804 | 0.5414 | 0.5284 | 0.5272 |
| macro precision | 0.4157 | 0.4157 | 0.5307 | 0.4908 | 0.5265 |
| macro recall | 0.4387 | 0.4387 | 0.5745 | 0.5624 | 0.5583 |
| macro F1 | 0.3861 | 0.3861 | 0.5113 | 0.4792 | 0.5003 |
| weighted precision | 0.4564 | 0.4564 | 0.5862 | 0.5510 | 0.5794 |
| weighted recall | 0.3804 | 0.3804 | 0.5414 | 0.5284 | 0.5272 |
| weighted F1 | 0.3829 | 0.3829 | 0.5321 | 0.5083 | 0.5207 |

## Per-class metrics - window

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 5146 | 0.4226 | 0.4668 | 0.4436 |
| 1 | Alarm | 8465 | 0.7566 | 0.2185 | 0.3391 |
| 2 | Baby_Crying | 3086 | 0.9706 | 0.7178 | 0.8253 |
| 3 | Car_Engine | 5560 | 0.2824 | 0.1725 | 0.2142 |
| 4 | Dog_Bark | 4656 | 0.3312 | 0.4824 | 0.3927 |
| 5 | Doorbell | 428 | 0.0821 | 0.6051 | 0.1445 |
| 6 | Drilling | 5820 | 0.4474 | 0.2251 | 0.2995 |
| 7 | Footsteps | 5912 | 0.2972 | 0.3259 | 0.3109 |
| 8 | Glass_Breaking | 2069 | 0.2481 | 0.4476 | 0.3192 |
| 9 | Gunshot | 3022 | 0.2367 | 0.4110 | 0.3004 |
| 10 | Help_Shouting | 3626 | 0.3800 | 0.4446 | 0.4098 |
| 11 | Jackhammer | 5181 | 0.6599 | 0.5202 | 0.5818 |
| 12 | Knocking | 1926 | 0.1505 | 0.6376 | 0.2435 |
| 13 | Motorcycle | 3826 | 0.3220 | 0.3680 | 0.3435 |
| 14 | Siren | 5277 | 0.6371 | 0.7015 | 0.6677 |
| 15 | Train | 8624 | 0.3588 | 0.1426 | 0.2041 |
| 16 | Vehicle_Horn | 2206 | 0.4842 | 0.5703 | 0.5237 |

## Per-class metrics - clip_mean_softmax

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 51 | 0.4058 | 0.5490 | 0.4667 |
| 1 | Alarm | 120 | 0.7778 | 0.3500 | 0.4828 |
| 2 | Baby_Crying | 70 | 1.0000 | 0.7429 | 0.8525 |
| 3 | Car_Engine | 146 | 0.4583 | 0.1507 | 0.2268 |
| 4 | Dog_Bark | 158 | 0.6607 | 0.7025 | 0.6810 |
| 5 | Doorbell | 8 | 0.0833 | 0.6250 | 0.1471 |
| 6 | Drilling | 168 | 0.5132 | 0.2321 | 0.3197 |
| 7 | Footsteps | 87 | 0.2885 | 0.3448 | 0.3141 |
| 8 | Glass_Breaking | 75 | 0.4812 | 0.8533 | 0.6154 |
| 9 | Gunshot | 122 | 0.5105 | 0.5984 | 0.5509 |
| 10 | Help_Shouting | 76 | 0.4696 | 0.7105 | 0.5654 |
| 11 | Jackhammer | 148 | 0.7757 | 0.5608 | 0.6510 |
| 12 | Knocking | 63 | 0.3741 | 0.8730 | 0.5238 |
| 13 | Motorcycle | 35 | 0.2754 | 0.5429 | 0.3654 |
| 14 | Siren | 142 | 0.8435 | 0.8732 | 0.8581 |
| 15 | Train | 80 | 0.2830 | 0.1875 | 0.2256 |
| 16 | Vehicle_Horn | 69 | 0.8219 | 0.8696 | 0.8451 |

## Per-class metrics - clip_topk_mean

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 51 | 0.3684 | 0.5490 | 0.4409 |
| 1 | Alarm | 120 | 0.7843 | 0.3333 | 0.4678 |
| 2 | Baby_Crying | 70 | 0.9815 | 0.7571 | 0.8548 |
| 3 | Car_Engine | 146 | 0.4098 | 0.1712 | 0.2415 |
| 4 | Dog_Bark | 158 | 0.7535 | 0.6772 | 0.7133 |
| 5 | Doorbell | 8 | 0.1000 | 0.8750 | 0.1795 |
| 6 | Drilling | 168 | 0.5132 | 0.2321 | 0.3197 |
| 7 | Footsteps | 87 | 0.3281 | 0.2414 | 0.2781 |
| 8 | Glass_Breaking | 75 | 0.4855 | 0.8933 | 0.6291 |
| 9 | Gunshot | 122 | 0.4667 | 0.4590 | 0.4628 |
| 10 | Help_Shouting | 76 | 0.4694 | 0.6053 | 0.5287 |
| 11 | Jackhammer | 148 | 0.7103 | 0.6959 | 0.7031 |
| 12 | Knocking | 63 | 0.3500 | 0.8889 | 0.5022 |
| 13 | Motorcycle | 35 | 0.2069 | 0.3429 | 0.2581 |
| 14 | Siren | 142 | 0.7293 | 0.9296 | 0.8173 |
| 15 | Train | 80 | 0.0952 | 0.0250 | 0.0396 |
| 16 | Vehicle_Horn | 69 | 0.5922 | 0.8841 | 0.7093 |

## Per-class metrics - clip_majority_vote

| id | class | support | precision | recall | F1 |
|---:|---|---:|---:|---:|---:|
| 0 | Aircraft | 51 | 0.3971 | 0.5294 | 0.4538 |
| 1 | Alarm | 120 | 0.8409 | 0.3083 | 0.4512 |
| 2 | Baby_Crying | 70 | 1.0000 | 0.7286 | 0.8430 |
| 3 | Car_Engine | 146 | 0.4107 | 0.1575 | 0.2277 |
| 4 | Dog_Bark | 158 | 0.6279 | 0.6835 | 0.6545 |
| 5 | Doorbell | 8 | 0.0862 | 0.6250 | 0.1515 |
| 6 | Drilling | 168 | 0.5067 | 0.2262 | 0.3128 |
| 7 | Footsteps | 87 | 0.2586 | 0.3448 | 0.2956 |
| 8 | Glass_Breaking | 75 | 0.4553 | 0.7467 | 0.5657 |
| 9 | Gunshot | 122 | 0.4929 | 0.5656 | 0.5267 |
| 10 | Help_Shouting | 76 | 0.5093 | 0.7237 | 0.5978 |
| 11 | Jackhammer | 148 | 0.7778 | 0.5676 | 0.6562 |
| 12 | Knocking | 63 | 0.3210 | 0.8254 | 0.4622 |
| 13 | Motorcycle | 35 | 0.2836 | 0.5429 | 0.3725 |
| 14 | Siren | 142 | 0.8289 | 0.8873 | 0.8571 |
| 15 | Train | 80 | 0.3000 | 0.1875 | 0.2308 |
| 16 | Vehicle_Horn | 69 | 0.8529 | 0.8406 | 0.8467 |

## Confusion matrix - window (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **2402** | 2 | 0 | 319 | 130 | 58 | 72 | 130 | 43 | 357 | 17 | 192 | 721 | 374 | 45 | 267 | 17 |
| Alarm | 623 | **1850** | 23 | 125 | 662 | 1216 | 219 | 427 | 289 | 121 | 605 | 141 | 747 | 147 | 573 | 132 | 565 |
| Baby_Crying | 1 | 64 | **2215** | 0 | 97 | 215 | 49 | 20 | 50 | 16 | 275 | 0 | 15 | 3 | 12 | 1 | 53 |
| Car_Engine | 251 | 9 | 2 | **959** | 499 | 131 | 121 | 430 | 6 | 1677 | 4 | 61 | 154 | 527 | 339 | 315 | 75 |
| Dog_Bark | 45 | 63 | 0 | 52 | **2246** | 224 | 134 | 259 | 171 | 86 | 339 | 8 | 569 | 131 | 231 | 68 | 30 |
| Doorbell | 0 | 7 | 0 | 0 | 6 | **259** | 1 | 49 | 32 | 6 | 6 | 1 | 56 | 1 | 4 | 0 | 0 |
| Drilling | 286 | 3 | 0 | 186 | 575 | 42 | **1310** | 316 | 807 | 105 | 493 | 316 | 36 | 482 | 144 | 542 | 177 |
| Footsteps | 150 | 24 | 11 | 19 | 397 | 61 | 23 | **1927** | 498 | 350 | 66 | 77 | 1764 | 117 | 206 | 188 | 34 |
| Glass_Breaking | 1 | 17 | 0 | 1 | 99 | 77 | 45 | 298 | **926** | 158 | 16 | 78 | 314 | 11 | 11 | 14 | 3 |
| Gunshot | 368 | 1 | 0 | 36 | 46 | 17 | 5 | 184 | 386 | **1242** | 5 | 23 | 666 | 16 | 2 | 25 | 0 |
| Help_Shouting | 8 | 177 | 16 | 93 | 423 | 155 | 77 | 124 | 137 | 113 | **1612** | 16 | 170 | 72 | 248 | 87 | 98 |
| Jackhammer | 0 | 0 | 0 | 723 | 2 | 0 | 464 | 734 | 34 | 21 | 0 | **2695** | 21 | 282 | 0 | 200 | 5 |
| Knocking | 36 | 8 | 0 | 22 | 98 | 55 | 0 | 173 | 44 | 165 | 3 | 7 | **1228** | 39 | 12 | 36 | 0 |
| Motorcycle | 108 | 13 | 13 | 503 | 193 | 89 | 97 | 328 | 32 | 104 | 55 | 121 | 418 | **1408** | 80 | 254 | 10 |
| Siren | 16 | 109 | 0 | 2 | 650 | 266 | 0 | 4 | 4 | 76 | 194 | 0 | 150 | 15 | **3702** | 8 | 81 |
| Train | 1339 | 72 | 1 | 312 | 522 | 188 | 292 | 926 | 269 | 644 | 530 | 337 | 939 | 649 | 182 | **1230** | 192 |
| Vehicle_Horn | 50 | 26 | 1 | 44 | 137 | 103 | 19 | 154 | 5 | 6 | 22 | 11 | 191 | 98 | 20 | 61 | **1258** |

## Confusion matrix - clip_mean_softmax (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **28** | 0 | 0 | 2 | 1 | 0 | 1 | 2 | 0 | 3 | 0 | 2 | 5 | 5 | 0 | 2 | 0 |
| Alarm | 7 | **42** | 0 | 0 | 4 | 28 | 1 | 2 | 3 | 0 | 13 | 1 | 7 | 1 | 3 | 2 | 6 |
| Baby_Crying | 0 | 2 | **52** | 0 | 1 | 5 | 1 | 0 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 6 | 0 | 0 | **22** | 20 | 5 | 5 | 9 | 0 | 47 | 0 | 0 | 5 | 12 | 6 | 9 | 0 |
| Dog_Bark | 1 | 1 | 0 | 1 | **111** | 3 | 4 | 6 | 5 | 1 | 12 | 0 | 5 | 3 | 3 | 2 | 0 |
| Doorbell | 0 | 0 | 0 | 0 | 0 | **5** | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Drilling | 8 | 0 | 0 | 3 | 14 | 0 | **39** | 8 | 24 | 2 | 18 | 15 | 0 | 16 | 5 | 11 | 5 |
| Footsteps | 2 | 0 | 0 | 0 | 2 | 0 | 0 | **30** | 8 | 4 | 0 | 1 | 35 | 2 | 2 | 1 | 0 |
| Glass_Breaking | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 2 | **64** | 1 | 0 | 1 | 5 | 0 | 0 | 0 | 0 |
| Gunshot | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 9 | 21 | **73** | 0 | 0 | 15 | 0 | 0 | 1 | 0 |
| Help_Shouting | 0 | 6 | 0 | 0 | 2 | 1 | 2 | 0 | 2 | 1 | **54** | 0 | 2 | 1 | 3 | 1 | 1 |
| Jackhammer | 0 | 0 | 0 | 16 | 0 | 0 | 20 | 22 | 0 | 0 | 0 | **83** | 0 | 3 | 0 | 4 | 0 |
| Knocking | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 1 | 2 | 0 | 0 | **55** | 0 | 0 | 1 | 0 |
| Motorcycle | 0 | 0 | 0 | 3 | 2 | 1 | 1 | 2 | 0 | 1 | 0 | 1 | 1 | **19** | 0 | 4 | 0 |
| Siren | 0 | 3 | 0 | 0 | 2 | 7 | 0 | 0 | 0 | 1 | 0 | 0 | 4 | 1 | **124** | 0 | 0 |
| Train | 15 | 0 | 0 | 0 | 4 | 3 | 2 | 7 | 4 | 7 | 8 | 3 | 5 | 5 | 1 | **15** | 1 |
| Vehicle_Horn | 0 | 0 | 0 | 1 | 1 | 2 | 0 | 1 | 0 | 0 | 1 | 0 | 2 | 1 | 0 | 0 | **60** |

## Confusion matrix - clip_topk_mean (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **28** | 0 | 0 | 5 | 1 | 2 | 1 | 1 | 0 | 1 | 0 | 5 | 6 | 0 | 0 | 0 | 1 |
| Alarm | 7 | **40** | 0 | 1 | 2 | 29 | 0 | 0 | 3 | 0 | 11 | 2 | 5 | 1 | 7 | 0 | 12 |
| Baby_Crying | 0 | 2 | **53** | 0 | 1 | 6 | 2 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| Car_Engine | 6 | 0 | 0 | **25** | 9 | 5 | 4 | 6 | 0 | 45 | 0 | 2 | 5 | 14 | 18 | 4 | 3 |
| Dog_Bark | 0 | 0 | 0 | 2 | **107** | 3 | 3 | 2 | 3 | 1 | 15 | 1 | 6 | 3 | 10 | 2 | 0 |
| Doorbell | 0 | 0 | 0 | 0 | 0 | **7** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Drilling | 5 | 0 | 0 | 5 | 12 | 0 | **39** | 8 | 22 | 2 | 17 | 18 | 0 | 15 | 6 | 10 | 9 |
| Footsteps | 2 | 1 | 0 | 0 | 2 | 1 | 0 | **21** | 13 | 2 | 0 | 1 | 39 | 1 | 2 | 0 | 2 |
| Glass_Breaking | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | **67** | 1 | 0 | 3 | 2 | 0 | 0 | 0 | 0 |
| Gunshot | 14 | 0 | 0 | 0 | 1 | 0 | 0 | 6 | 21 | **56** | 0 | 0 | 23 | 0 | 0 | 1 | 0 |
| Help_Shouting | 0 | 7 | 1 | 1 | 1 | 5 | 3 | 0 | 3 | 2 | **46** | 0 | 0 | 0 | 3 | 0 | 4 |
| Jackhammer | 0 | 0 | 0 | 14 | 0 | 0 | 18 | 11 | 0 | 0 | 0 | **103** | 0 | 0 | 0 | 2 | 0 |
| Knocking | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | 3 | 0 | 1 | **56** | 0 | 0 | 0 | 0 |
| Motorcycle | 1 | 0 | 0 | 7 | 0 | 1 | 3 | 1 | 0 | 1 | 0 | 5 | 3 | **12** | 1 | 0 | 0 |
| Siren | 0 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 1 | **132** | 0 | 0 |
| Train | 13 | 1 | 0 | 0 | 4 | 2 | 3 | 4 | 4 | 6 | 5 | 4 | 12 | 10 | 2 | **2** | 8 |
| Vehicle_Horn | 0 | 0 | 0 | 1 | 0 | 2 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | **61** |

## Confusion matrix - clip_majority_vote (rows = actual, columns = predicted)

| actual \ predicted | Aircraft | Alarm | Baby_Crying | Car_Engine | Dog_Bark | Doorbell | Drilling | Footsteps | Glass_Breaking | Gunshot | Help_Shouting | Jackhammer | Knocking | Motorcycle | Siren | Train | Vehicle_Horn |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Aircraft | **27** | 0 | 0 | 3 | 1 | 0 | 1 | 2 | 0 | 3 | 0 | 2 | 7 | 2 | 0 | 3 | 0 |
| Alarm | 6 | **37** | 0 | 2 | 5 | 29 | 2 | 3 | 3 | 0 | 12 | 1 | 8 | 2 | 4 | 1 | 5 |
| Baby_Crying | 0 | 1 | **51** | 0 | 1 | 6 | 1 | 0 | 0 | 1 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| Car_Engine | 7 | 0 | 0 | **23** | 22 | 5 | 4 | 8 | 0 | 47 | 0 | 0 | 4 | 12 | 7 | 7 | 0 |
| Dog_Bark | 1 | 2 | 0 | 1 | **108** | 3 | 4 | 6 | 6 | 1 | 8 | 0 | 8 | 4 | 4 | 2 | 0 |
| Doorbell | 0 | 0 | 0 | 0 | 0 | **5** | 0 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| Drilling | 10 | 0 | 0 | 4 | 15 | 0 | **38** | 9 | 24 | 2 | 17 | 15 | 0 | 15 | 5 | 11 | 3 |
| Footsteps | 2 | 0 | 0 | 0 | 2 | 0 | 0 | **30** | 7 | 5 | 0 | 1 | 35 | 2 | 2 | 1 | 0 |
| Glass_Breaking | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 8 | **56** | 1 | 0 | 1 | 7 | 0 | 0 | 0 | 0 |
| Gunshot | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 9 | 20 | **69** | 0 | 0 | 20 | 0 | 0 | 1 | 0 |
| Help_Shouting | 0 | 3 | 0 | 2 | 2 | 1 | 2 | 0 | 2 | 1 | **55** | 0 | 3 | 0 | 3 | 1 | 1 |
| Jackhammer | 0 | 0 | 0 | 15 | 0 | 0 | 20 | 24 | 0 | 0 | 0 | **84** | 0 | 1 | 0 | 4 | 0 |
| Knocking | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 4 | 1 | 2 | 0 | 0 | **52** | 0 | 0 | 1 | 0 |
| Motorcycle | 0 | 0 | 0 | 5 | 2 | 0 | 0 | 2 | 0 | 1 | 0 | 1 | 2 | **19** | 0 | 3 | 0 |
| Siren | 0 | 1 | 0 | 0 | 4 | 5 | 0 | 0 | 0 | 1 | 0 | 0 | 4 | 1 | **126** | 0 | 0 |
| Train | 13 | 0 | 0 | 0 | 4 | 2 | 3 | 8 | 3 | 6 | 6 | 3 | 8 | 7 | 1 | **15** | 1 |
| Vehicle_Horn | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 2 | 0 | 0 | 1 | 0 | 3 | 2 | 0 | 0 | **58** |

## Top 15 clip-level confusions (mean-softmax)

largest off-diagonal cells of the clip_mean_softmax matrix.

| true | predicted | clips |
|---|---|---:|
| Car_Engine | Gunshot | 47 |
| Footsteps | Knocking | 35 |
| Alarm | Doorbell | 28 |
| Drilling | Glass_Breaking | 24 |
| Jackhammer | Footsteps | 22 |
| Gunshot | Glass_Breaking | 21 |
| Car_Engine | Dog_Bark | 20 |
| Jackhammer | Drilling | 20 |
| Drilling | Help_Shouting | 18 |
| Drilling | Motorcycle | 16 |
| Jackhammer | Car_Engine | 16 |
| Drilling | Jackhammer | 15 |
| Gunshot | Knocking | 15 |
| Train | Aircraft | 15 |
| Drilling | Dog_Bark | 14 |

## Baby_Crying clip recall by sample-rate group (mean-softmax)

ledger sample_rate is the original clip sample rate; clips with no prediction (zero-window clips) are excluded.

| sample-rate group | clips (total) | clips (assessed) | correct | recall |
|---:|---:|---:|---:|---:|
| below 16000 | 32 | 32 | 32 | 1.0000 |
| 16000 and above | 38 | 38 | 20 | 0.5263 |

## Misclassified clips - within vs cross dominant source_dataset (mean-softmax)

A class's dominant source_dataset is the dataset with the most clips of that class in the ledger. An error is within-dataset when the predicted class has the same dominant source_dataset as the true class, and cross-dataset otherwise.

- misclassified clips: **742**
- within-dataset errors: **396**
- cross-dataset errors: **346**
- errors with no dominant dataset for either class: **0**

| true class | predicted class | kind | clips |
|---|---|---:|---:|
| Car_Engine | Gunshot | cross | 47 |
| Footsteps | Knocking | within | 35 |
| Alarm | Doorbell | within | 28 |
| Drilling | Glass_Breaking | cross | 24 |
| Jackhammer | Footsteps | cross | 22 |
| Gunshot | Glass_Breaking | within | 21 |
| Jackhammer | Drilling | within | 20 |
| Car_Engine | Dog_Bark | within | 20 |
| Drilling | Help_Shouting | cross | 18 |
| Jackhammer | Car_Engine | within | 16 |
| Drilling | Motorcycle | cross | 16 |
| Train | Aircraft | within | 15 |
| Gunshot | Knocking | within | 15 |
| Drilling | Jackhammer | within | 15 |
| Drilling | Dog_Bark | within | 14 |
| Alarm | Help_Shouting | within | 13 |
| Car_Engine | Motorcycle | cross | 12 |
| Dog_Bark | Help_Shouting | cross | 12 |
| Drilling | Train | cross | 11 |
| Baby_Crying | Help_Shouting | cross | 9 |
| Gunshot | Footsteps | within | 9 |
| Car_Engine | Footsteps | cross | 9 |
| Car_Engine | Train | cross | 9 |
| Footsteps | Glass_Breaking | within | 8 |
| Train | Help_Shouting | within | 8 |
| Drilling | Footsteps | cross | 8 |
| Drilling | Aircraft | cross | 8 |
| Train | Gunshot | within | 7 |
| Train | Footsteps | within | 7 |
| Alarm | Aircraft | within | 7 |
| Alarm | Knocking | within | 7 |
| Siren | Doorbell | cross | 7 |
| Alarm | Vehicle_Horn | cross | 6 |
| Help_Shouting | Alarm | within | 6 |
| Dog_Bark | Footsteps | cross | 6 |
| Car_Engine | Aircraft | cross | 6 |
| Car_Engine | Siren | within | 6 |
| Aircraft | Knocking | within | 5 |
| Baby_Crying | Doorbell | cross | 5 |
| Train | Motorcycle | within | 5 |
| Aircraft | Motorcycle | within | 5 |
| Glass_Breaking | Knocking | within | 5 |
| Train | Knocking | within | 5 |
| Dog_Bark | Knocking | cross | 5 |
| Car_Engine | Doorbell | cross | 5 |
| Dog_Bark | Glass_Breaking | cross | 5 |
| Car_Engine | Knocking | cross | 5 |
| Car_Engine | Drilling | within | 5 |
| Drilling | Vehicle_Horn | within | 5 |
| Drilling | Siren | within | 5 |
| Train | Glass_Breaking | within | 4 |
| Motorcycle | Train | within | 4 |
| Alarm | Dog_Bark | cross | 4 |
| Train | Dog_Bark | cross | 4 |
| Footsteps | Gunshot | within | 4 |
| Jackhammer | Train | cross | 4 |
| Dog_Bark | Drilling | within | 4 |
| Siren | Knocking | cross | 4 |
| Knocking | Footsteps | within | 3 |
| Aircraft | Gunshot | within | 3 |
| Alarm | Glass_Breaking | within | 3 |
| Alarm | Siren | cross | 3 |
| Help_Shouting | Siren | cross | 3 |
| Train | Doorbell | within | 3 |
| Motorcycle | Car_Engine | cross | 3 |
| Train | Jackhammer | cross | 3 |
| Jackhammer | Motorcycle | cross | 3 |
| Dog_Bark | Siren | within | 3 |
| Dog_Bark | Doorbell | cross | 3 |
| Dog_Bark | Motorcycle | cross | 3 |
| Drilling | Car_Engine | within | 3 |
| Siren | Alarm | cross | 3 |
| Aircraft | Footsteps | within | 2 |
| Vehicle_Horn | Knocking | cross | 2 |
| Baby_Crying | Alarm | cross | 2 |
| Aircraft | Train | within | 2 |
| Aircraft | Jackhammer | cross | 2 |
| Help_Shouting | Dog_Bark | cross | 2 |
| Footsteps | Dog_Bark | cross | 2 |
| Gunshot | Aircraft | within | 2 |
| Motorcycle | Dog_Bark | cross | 2 |
| Glass_Breaking | Dog_Bark | cross | 2 |
| Footsteps | Motorcycle | within | 2 |
| Aircraft | Car_Engine | cross | 2 |
| Vehicle_Horn | Doorbell | cross | 2 |
| Help_Shouting | Glass_Breaking | within | 2 |
| Train | Drilling | cross | 2 |
| Alarm | Footsteps | within | 2 |
| Help_Shouting | Drilling | cross | 2 |
| Footsteps | Siren | cross | 2 |
| Footsteps | Aircraft | within | 2 |
| Motorcycle | Footsteps | within | 2 |
| Help_Shouting | Knocking | within | 2 |
| Knocking | Gunshot | within | 2 |
| Glass_Breaking | Footsteps | within | 2 |
| Alarm | Train | within | 2 |
| Siren | Dog_Bark | within | 2 |
| Dog_Bark | Train | cross | 2 |
| Drilling | Gunshot | cross | 2 |
| Baby_Crying | Drilling | cross | 1 |
| Baby_Crying | Dog_Bark | cross | 1 |
| Gunshot | Dog_Bark | cross | 1 |
| Help_Shouting | Vehicle_Horn | cross | 1 |
| Motorcycle | Jackhammer | cross | 1 |
| Footsteps | Jackhammer | cross | 1 |
| Aircraft | Drilling | cross | 1 |
| Help_Shouting | Motorcycle | within | 1 |
| Motorcycle | Drilling | cross | 1 |
| Knocking | Glass_Breaking | within | 1 |
| Knocking | Dog_Bark | cross | 1 |
| Vehicle_Horn | Dog_Bark | within | 1 |
| Vehicle_Horn | Car_Engine | within | 1 |
| Motorcycle | Doorbell | within | 1 |
| Alarm | Motorcycle | within | 1 |
| Aircraft | Dog_Bark | cross | 1 |
| Doorbell | Knocking | within | 1 |
| Train | Siren | cross | 1 |
| Help_Shouting | Doorbell | within | 1 |
| Alarm | Jackhammer | cross | 1 |
| Help_Shouting | Gunshot | within | 1 |
| Motorcycle | Gunshot | within | 1 |
| Gunshot | Train | within | 1 |
| Glass_Breaking | Gunshot | within | 1 |
| Vehicle_Horn | Motorcycle | cross | 1 |
| Knocking | Train | within | 1 |
| Vehicle_Horn | Footsteps | cross | 1 |
| Doorbell | Footsteps | within | 1 |
| Glass_Breaking | Jackhammer | cross | 1 |
| Alarm | Drilling | cross | 1 |
| Motorcycle | Knocking | within | 1 |
| Doorbell | Glass_Breaking | within | 1 |
| Footsteps | Train | within | 1 |
| Train | Vehicle_Horn | cross | 1 |
| Help_Shouting | Train | within | 1 |
| Dog_Bark | Gunshot | cross | 1 |
| Siren | Gunshot | cross | 1 |
| Dog_Bark | Car_Engine | within | 1 |
| Dog_Bark | Aircraft | cross | 1 |
| Vehicle_Horn | Help_Shouting | cross | 1 |
| Dog_Bark | Alarm | cross | 1 |
| Siren | Motorcycle | cross | 1 |

## Per-class clip recall by source_dataset (mean-softmax)

classes with more than one source_dataset in the validation split.

| class | source_dataset | clips | correct | recall |
|---|---|---:|---:|---:|
| Aircraft | ESC-50 | 13 | 7 | 0.5385 |
| Aircraft | FSD50K | 38 | 21 | 0.5526 |
| Alarm | ESC-50 | 3 | 3 | 1.0000 |
| Alarm | FSD50K | 117 | 39 | 0.3333 |
| Baby_Crying | ESC-50 | 17 | 0 | 0.0000 |
| Baby_Crying | owlgebra_babycry | 53 | 52 | 0.9811 |
| Footsteps | ESC-50 | 6 | 4 | 0.6667 |
| Footsteps | FSD50K | 81 | 26 | 0.3210 |
| Glass_Breaking | ESC-50 | 4 | 4 | 1.0000 |
| Glass_Breaking | FSD50K | 71 | 60 | 0.8451 |
| Gunshot | FSD50K | 61 | 29 | 0.4754 |
| Gunshot | UrbanSound8K | 61 | 44 | 0.7213 |
| Knocking | ESC-50 | 4 | 3 | 0.7500 |
| Knocking | FSD50K | 59 | 52 | 0.8814 |
| Train | ESC-50 | 6 | 3 | 0.5000 |
| Train | FSD50K | 74 | 12 | 0.1622 |
| Vehicle_Horn | ESC-50 | 3 | 2 | 0.6667 |
| Vehicle_Horn | FSD50K | 11 | 4 | 0.3636 |
| Vehicle_Horn | UrbanSound8K | 55 | 54 | 0.9818 |

## Clips with zero windows

No validation clip has zero windows.

model.predict: 1.2 s