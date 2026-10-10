# YAMNet embedding baseline - fit report (yamnet_fit2_fixed)

- extract: C:\Users\BITPATNA\minorProject\data\embeddings\yamnet\yamnet_extract (train 7021 / val 1618 clips)
- CNN comparator: C:\Users\BITPATNA\minorProject\scripts\results\phase3a_clip_eval\baseline_A (clip_mean_softmax level)
- YAMNet: frozen, https://tfhub.dev/google/yamnet/1
- logistic head: StandardScaler fit on train, multinomial logistic regression on standardized per-clip mean embeddings, seed 42
- test split: never loaded

## Side-by-side (validation clip level)

| variant | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| logistic | 0.7769 | 0.7426 | 0.7791 |
| zero_shot | 0.6273 | 0.5955 | 0.6135 |
| clip_mean_softmax (CNN) | 0.4994 | 0.4689 | 0.5193 |

Logistic uses mean frame embeddings; zero-shot uses mean YAMNet class scores summed over a hand-mapped subset of the 521 classes.

## logistic

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Aircraft | 0.6481 | 0.6863 | 0.6667 | 51 |
| Alarm | 0.6389 | 0.575 | 0.6053 | 120 |
| Baby_Crying | 0.9559 | 0.9286 | 0.942 | 70 |
| Car_Engine | 0.6296 | 0.5822 | 0.605 | 146 |
| Dog_Bark | 0.9342 | 0.8987 | 0.9161 | 158 |
| Doorbell | 0.375 | 0.375 | 0.375 | 8 |
| Drilling | 0.9091 | 0.8333 | 0.8696 | 168 |
| Footsteps | 0.7158 | 0.7816 | 0.7473 | 87 |
| Glass_Breaking | 0.8571 | 0.8 | 0.8276 | 75 |
| Gunshot | 0.8943 | 0.9016 | 0.898 | 122 |
| Help_Shouting | 0.6979 | 0.8816 | 0.7791 | 76 |
| Jackhammer | 0.9167 | 0.7432 | 0.8209 | 148 |
| Knocking | 0.8667 | 0.8254 | 0.8455 | 63 |
| Motorcycle | 0.4531 | 0.8286 | 0.5859 | 35 |
| Siren | 0.7898 | 0.8732 | 0.8294 | 142 |
| Train | 0.5532 | 0.65 | 0.5977 | 80 |
| Vehicle_Horn | 0.7667 | 0.6667 | 0.7132 | 69 |

Top clip-level confusions:

| true class | predicted class | count |
|---|---|---|
| Jackhammer | Car_Engine | 34 |
| Car_Engine | Siren | 20 |
| Car_Engine | Motorcycle | 17 |
| Alarm | Train | 13 |
| Vehicle_Horn | Alarm | 11 |
| Siren | Help_Shouting | 9 |
| Alarm | Vehicle_Horn | 8 |
| Siren | Alarm | 8 |
| Alarm | Help_Shouting | 7 |
| Drilling | Train | 7 |
| Knocking | Footsteps | 7 |
| Train | Alarm | 7 |
| Train | Drilling | 7 |
| Drilling | Car_Engine | 6 |
| Glass_Breaking | Gunshot | 6 |

Within vs cross dominant source_dataset errors (over misclassified clips only): within 210, cross 151, excluded 0 of 361 misclassified clips (same definition as run_phase3a_clip_eval.py).

Baby_Crying clip recall by source sample-rate group:

| group | correct | total |
|---|---|---|
| below_16000 | 31 | 32 |
| at_least_16000 | 34 | 38 |

## zero_shot

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| Aircraft | 0.3277 | 0.7647 | 0.4588 | 51 |
| Alarm | 0.9091 | 0.0833 | 0.1527 | 120 |
| Baby_Crying | 0.7361 | 0.7571 | 0.7465 | 70 |
| Car_Engine | 0.3256 | 0.5753 | 0.4158 | 146 |
| Dog_Bark | 0.9079 | 0.8734 | 0.8903 | 158 |
| Doorbell | 0.2 | 0.5 | 0.2857 | 8 |
| Drilling | 0.918 | 0.3333 | 0.4891 | 168 |
| Footsteps | 0.8235 | 0.4828 | 0.6087 | 87 |
| Glass_Breaking | 0.6273 | 0.92 | 0.7459 | 75 |
| Gunshot | 0.887 | 0.8361 | 0.8608 | 122 |
| Help_Shouting | 0.8 | 0.6842 | 0.7376 | 76 |
| Jackhammer | 0.9804 | 0.3378 | 0.5025 | 148 |
| Knocking | 0.7324 | 0.8254 | 0.7761 | 63 |
| Motorcycle | 0.375 | 0.5143 | 0.4337 | 35 |
| Siren | 0.695 | 0.9789 | 0.8129 | 142 |
| Train | 0.4154 | 0.675 | 0.5143 | 80 |
| Vehicle_Horn | 0.631 | 0.7681 | 0.6928 | 69 |

Zero-shot mapping (project class -> resolved YAMNet classes):

| project class | mapped YAMNet classes |
|---|---|
| Aircraft | Aircraft (index 329), Aircraft engine (index 330), Jet engine (index 331), Helicopter (index 333) |
| Alarm | Fire alarm (index 394) |
| Baby_Crying | Baby cry, infant cry (index 20), Crying, sobbing (index 19) |
| Car_Engine | Engine (index 337) |
| Dog_Bark | Bark (index 70), Bow-wow (index 73), Yip (index 71), Growling (index 74), Howl (index 72) |
| Doorbell | Doorbell (index 349) |
| Drilling | Drill (index 419), Power tool (index 418) |
| Footsteps | Walk, footsteps (index 48) |
| Glass_Breaking | Glass (index 435), Shatter (index 437) |
| Gunshot | Gunshot, gunfire (index 421) |
| Help_Shouting | Shout (index 6), Screaming (index 11), Yell (index 9) |
| Jackhammer | Jackhammer (index 414) |
| Knocking | Knock (index 353) |
| Motorcycle | Motorcycle (index 320) |
| Siren | Siren (index 390), Civil defense siren (index 391) |
| Train | Train (index 323), Train wheels squealing (index 327) |
| Vehicle_Horn | Vehicle horn, car horn, honking (index 302), Air horn, truck horn (index 312) |

Unmapped classes (never predicted): 

Top clip-level confusions:

| true class | predicted class | count |
|---|---|---|
| Jackhammer | Car_Engine | 88 |
| Drilling | Car_Engine | 51 |
| Car_Engine | Train | 29 |
| Alarm | Siren | 25 |
| Car_Engine | Aircraft | 23 |
| Drilling | Aircraft | 21 |
| Alarm | Vehicle_Horn | 18 |
| Footsteps | Knocking | 16 |
| Alarm | Doorbell | 15 |
| Alarm | Aircraft | 14 |
| Alarm | Glass_Breaking | 14 |
| Drilling | Motorcycle | 13 |
| Drilling | Train | 13 |
| Help_Shouting | Baby_Crying | 13 |
| Motorcycle | Car_Engine | 13 |

Within vs cross dominant source_dataset errors (over misclassified clips only): within 338, cross 265, excluded 0 of 603 misclassified clips (same definition as run_phase3a_clip_eval.py).

Baby_Crying clip recall by source sample-rate group:

| group | correct | total |
|---|---|---|
| below_16000 | 29 | 32 |
| at_least_16000 | 24 | 38 |
