# Class x dataset confounding (train, val)

## Method

Per class, each source_dataset's share of that class's clips and windows over splits (train, val). A pair is flagged when either share exceeds 80%. The test split is excluded; test-split metadata is never read.

## Flagged pairs (share > 80%)

| class_id | class_name | source_dataset | n_clips | n_windows | clip_share | window_share |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Aircraft | FSD50K | 216 | 25695 | 0.7579 | 0.8855 |
| 1 | Alarm | FSD50K | 612 | 43823 | 0.9730 | 0.9837 |
| 2 | Baby_Crying | owlgebra_babycry | 428 | 16987 | 0.9325 | 0.9201 |
| 3 | Car_Engine | UrbanSound8K | 820 | 31432 | 1.0000 | 1.0000 |
| 4 | Dog_Bark | UrbanSound8K | 807 | 23837 | 1.0000 | 1.0000 |
| 5 | Doorbell | FSD50K | 43 | 2369 | 1.0000 | 1.0000 |
| 6 | Drilling | UrbanSound8K | 841 | 29085 | 1.0000 | 1.0000 |
| 7 | Footsteps | FSD50K | 442 | 29952 | 0.9345 | 0.9533 |
| 8 | Glass_Breaking | FSD50K | 383 | 10037 | 0.9185 | 0.9462 |
| 10 | Help_Shouting | FSD50K | 419 | 20233 | 1.0000 | 1.0000 |
| 11 | Jackhammer | UrbanSound8K | 820 | 29106 | 1.0000 | 1.0000 |
| 12 | Knocking | FSD50K | 290 | 9264 | 0.9006 | 0.9190 |
| 13 | Motorcycle | FSD50K | 198 | 21925 | 1.0000 | 1.0000 |
| 14 | Siren | UrbanSound8K | 760 | 28703 | 1.0000 | 1.0000 |
| 15 | Train | FSD50K | 358 | 40905 | 0.9086 | 0.9587 |
| 16 | Vehicle_Horn | UrbanSound8K | 300 | 8340 | 0.9146 | 0.7931 |

## All pairs

| class_id | class_name | source_dataset | n_clips | n_windows | clip_share | window_share |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Aircraft | FSD50K | 216 | 25695 | 0.7579 | 0.8855 |
| 0 | Aircraft | ESC-50 | 69 | 3324 | 0.2421 | 0.1145 |
| 1 | Alarm | FSD50K | 612 | 43823 | 0.9730 | 0.9837 |
| 1 | Alarm | ESC-50 | 17 | 728 | 0.0270 | 0.0163 |
| 2 | Baby_Crying | owlgebra_babycry | 428 | 16987 | 0.9325 | 0.9201 |
| 2 | Baby_Crying | ESC-50 | 31 | 1476 | 0.0675 | 0.0799 |
| 3 | Car_Engine | UrbanSound8K | 820 | 31432 | 1.0000 | 1.0000 |
| 4 | Dog_Bark | UrbanSound8K | 807 | 23837 | 1.0000 | 1.0000 |
| 5 | Doorbell | FSD50K | 43 | 2369 | 1.0000 | 1.0000 |
| 6 | Drilling | UrbanSound8K | 841 | 29085 | 1.0000 | 1.0000 |
| 7 | Footsteps | FSD50K | 442 | 29952 | 0.9345 | 0.9533 |
| 7 | Footsteps | ESC-50 | 31 | 1467 | 0.0655 | 0.0467 |
| 8 | Glass_Breaking | FSD50K | 383 | 10037 | 0.9185 | 0.9462 |
| 8 | Glass_Breaking | ESC-50 | 34 | 571 | 0.0815 | 0.0538 |
| 9 | Gunshot | FSD50K | 321 | 11059 | 0.5144 | 0.7169 |
| 9 | Gunshot | UrbanSound8K | 303 | 4368 | 0.4856 | 0.2831 |
| 10 | Help_Shouting | FSD50K | 419 | 20233 | 1.0000 | 1.0000 |
| 11 | Jackhammer | UrbanSound8K | 820 | 29106 | 1.0000 | 1.0000 |
| 12 | Knocking | FSD50K | 290 | 9264 | 0.9006 | 0.9190 |
| 12 | Knocking | ESC-50 | 32 | 817 | 0.0994 | 0.0810 |
| 13 | Motorcycle | FSD50K | 198 | 21925 | 1.0000 | 1.0000 |
| 14 | Siren | UrbanSound8K | 760 | 28703 | 1.0000 | 1.0000 |
| 15 | Train | FSD50K | 358 | 40905 | 0.9086 | 0.9587 |
| 15 | Train | ESC-50 | 36 | 1764 | 0.0914 | 0.0413 |
| 16 | Vehicle_Horn | UrbanSound8K | 300 | 8340 | 0.9146 | 0.7931 |
| 16 | Vehicle_Horn | FSD50K | 25 | 2029 | 0.0762 | 0.1929 |
| 16 | Vehicle_Horn | ESC-50 | 3 | 147 | 0.0091 | 0.0140 |

## Zero-window clips (kept clips with no extracted windows)

- total kept clips with zero windows: **0** (these clips contribute 0 labels but no windows; n_windows is filled with 0, not an error)

No kept clip in these splits has zero windows.

## Class x original sample rate

A class is flagged when one sample_rate contributes more than 80% of its kept clips (missing sample_rate rows are reported as 'missing').

| class_id | class_name | sample_rate | n_clips | clip_share | flag_single_rate |
| --- | --- | --- | --- | --- | --- |
| 0 | Aircraft | 44100 | 285 | 1.0000 | yes |
| 1 | Alarm | 44100 | 629 | 1.0000 | yes |
| 2 | Baby_Crying | 44100 | 235 | 0.5120 | no |
| 2 | Baby_Crying | 8000 | 224 | 0.4880 | no |
| 3 | Car_Engine | 44100 | 458 | 0.5585 | no |
| 3 | Car_Engine | 48000 | 309 | 0.3768 | no |
| 3 | Car_Engine | 96000 | 53 | 0.0646 | no |
| 4 | Dog_Bark | 44100 | 526 | 0.6518 | no |
| 4 | Dog_Bark | 48000 | 175 | 0.2169 | no |
| 4 | Dog_Bark | 96000 | 70 | 0.0867 | no |
| 4 | Dog_Bark | 11025 | 22 | 0.0273 | no |
| 4 | Dog_Bark | 22050 | 6 | 0.0074 | no |
| 4 | Dog_Bark | 24000 | 4 | 0.0050 | no |
| 4 | Dog_Bark | 32000 | 4 | 0.0050 | no |
| 5 | Doorbell | 44100 | 43 | 1.0000 | yes |
| 6 | Drilling | 44100 | 602 | 0.7158 | no |
| 6 | Drilling | 48000 | 96 | 0.1141 | no |
| 6 | Drilling | 96000 | 84 | 0.0999 | no |
| 6 | Drilling | 24000 | 28 | 0.0333 | no |
| 6 | Drilling | 192000 | 17 | 0.0202 | no |
| 6 | Drilling | 22050 | 8 | 0.0095 | no |
| 6 | Drilling | 16000 | 6 | 0.0071 | no |
| 7 | Footsteps | 44100 | 473 | 1.0000 | yes |
| 8 | Glass_Breaking | 44100 | 417 | 1.0000 | yes |
| 9 | Gunshot | 44100 | 492 | 0.7885 | no |
| 9 | Gunshot | 48000 | 84 | 0.1346 | no |
| 9 | Gunshot | 96000 | 47 | 0.0753 | no |
| 9 | Gunshot | 22050 | 1 | 0.0016 | no |
| 10 | Help_Shouting | 44100 | 419 | 1.0000 | yes |
| 11 | Jackhammer | 44100 | 487 | 0.5939 | no |
| 11 | Jackhammer | 96000 | 209 | 0.2549 | no |
| 11 | Jackhammer | 48000 | 124 | 0.1512 | no |
| 12 | Knocking | 44100 | 322 | 1.0000 | yes |
| 13 | Motorcycle | 44100 | 198 | 1.0000 | yes |
| 14 | Siren | 44100 | 396 | 0.5211 | no |
| 14 | Siren | 48000 | 343 | 0.4513 | no |
| 14 | Siren | 16000 | 12 | 0.0158 | no |
| 14 | Siren | 11025 | 9 | 0.0118 | no |
| 15 | Train | 44100 | 394 | 1.0000 | yes |
| 16 | Vehicle_Horn | 44100 | 235 | 0.7165 | no |
| 16 | Vehicle_Horn | 48000 | 66 | 0.2012 | no |
| 16 | Vehicle_Horn | 16000 | 19 | 0.0579 | no |
| 16 | Vehicle_Horn | 96000 | 8 | 0.0244 | no |

- classes with a single dominant sample rate: 9
- ledger rows with missing sample_rate: 0

## Share of clips below 16000 Hz per class

| class_name: clips below 16000 Hz (share) |
| --- |
| Aircraft: 0/285 (0.0000) |
| Alarm: 0/629 (0.0000) |
| Baby_Crying: 224/459 (0.4880) |
| Car_Engine: 0/820 (0.0000) |
| Dog_Bark: 22/807 (0.0273) |
| Doorbell: 0/43 (0.0000) |
| Drilling: 0/841 (0.0000) |
| Footsteps: 0/473 (0.0000) |
| Glass_Breaking: 0/417 (0.0000) |
| Gunshot: 0/624 (0.0000) |
| Help_Shouting: 0/419 (0.0000) |
| Jackhammer: 0/820 (0.0000) |
| Knocking: 0/322 (0.0000) |
| Motorcycle: 0/198 (0.0000) |
| Siren: 9/760 (0.0118) |
| Train: 0/394 (0.0000) |
| Vehicle_Horn: 0/328 (0.0000) |

## Notes

- classes with no clips in these splits: none
- flagged pairs: 16
- source: C:\Users\BITPATNA\minorProject\data\provenance\phase2a_clip_ledger.csv (status=kept) joined to clip_splits.csv and window_index_*.csv on clip_id
