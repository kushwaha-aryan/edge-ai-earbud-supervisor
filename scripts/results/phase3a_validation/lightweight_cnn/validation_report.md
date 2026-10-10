# Phase 3A Validation Evaluation (analysis only)

Model: `data/models/lightweight_cnn.keras` (729,178 bytes, 56,657 parameters). Locked Phase 2B validation split: 74830 windows, 17 classes. No retraining; model, preprocessing, configuration and dataset unchanged.

## Overall metrics

- val loss: **2.1834**
- val accuracy: **0.3804** (28469/74830)
- macro precision / recall / F1: 0.4157 / 0.4387 / 0.3861
- weighted precision / recall / F1: 0.4564 / 0.3804 / 0.3829
- model.evaluate: 3.9 s, predict: 1.0 s

## Per-class metrics

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

## Confusion matrix (rows = actual, columns = predicted)

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

## Top confusions (off-diagonal, top 15)

| actual | predicted | windows | rate of actual class |
|---|---|---:|---:|
| Footsteps | Knocking | 1764 | 0.298 |
| Car_Engine | Gunshot | 1677 | 0.302 |
| Train | Aircraft | 1339 | 0.155 |
| Alarm | Doorbell | 1216 | 0.144 |
| Train | Knocking | 939 | 0.109 |
| Train | Footsteps | 926 | 0.107 |
| Drilling | Glass_Breaking | 807 | 0.139 |
| Alarm | Knocking | 747 | 0.088 |
| Jackhammer | Footsteps | 734 | 0.142 |
| Jackhammer | Car_Engine | 723 | 0.140 |
| Aircraft | Knocking | 721 | 0.140 |
| Gunshot | Knocking | 666 | 0.220 |
| Alarm | Dog_Bark | 662 | 0.078 |
| Siren | Dog_Bark | 650 | 0.123 |
| Train | Motorcycle | 649 | 0.075 |
