# Normalization audit

## Verdict

- feature_stats.json contains one global db_min/db_max pair: **GLOBAL: one fixed (db_min, db_max) pair frozen from the train split**
- exact formula:

```
    db = librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=None)
    normalized = clip((db - (-100.0)) / ((33.83852005004883) - (-100.0)), 0.0, 1.0)
    [source: src/preprocessing/features.py lines 92, 93, 94, 10, 139, 162, 191]
```

- normalization is **global (fixed train-frozen dB bounds), NOT per-window**: a single (db_min, db_max) pair, collected once from the train split pass A and recorded in data/processed/feature_stats.json, is applied to every window of every split (train, val).
- floor: db_min=-100.0 maps to normalized 0.0; any value at 0.0 is dB <= -100.0 (the -100 dB floor).
- saturation check: fraction of values >= 0.95.

## Code references

| code reference | file:line |
| --- | --- |
| _scaled definition | src/preprocessing/features.py:92 |
| scaling formula | src/preprocessing/features.py:93 |
| clip to [0, 1] | src/preprocessing/features.py:94 |
| dB conversion (top_db=None) | src/preprocessing/features.py:10 |
| bounds frozen from train pass A | src/preprocessing/features.py:139 |
| train scaling applies frozen bounds | src/preprocessing/features.py:162 |
| val/test scaling applies frozen bounds | src/preprocessing/features.py:191 |

## Per-class value statistics

Test split not examined. Non-finite stats mean the class has no windows in that split.

Bin columns: mean_bins_0_47 is the mean normalized value over mel bins 0-47 (below about 4 kHz), mean_bins_48_63 over bins 48-63 (about 4 kHz and above), and frac_at_floor_bins_48_63 is the fraction of values exactly at 0.0 (-100 dB floor) within bins 48-63 only.

| split | class_id | class_name | n_windows | n_at_floor | frac_at_floor | mean_norm | std_norm | mean_bins_0_47 | mean_bins_48_63 | frac_at_floor_bins_48_63 | n_ge_095 | frac_ge_095 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | Aircraft | 23873 | 2658 | 0.000348 | 0.5713 | 0.1495 | 0.6185 | 0.4298 | 0.000361 | 662 | 0.000087 |
| train | 1 | Alarm | 36086 | 77748 | 0.006733 | 0.5024 | 0.1733 | 0.5264 | 0.4302 | 0.006207 | 1824 | 0.000158 |
| train | 2 | Baby_Crying | 15377 | 802280 | 0.163044 | 0.4717 | 0.2423 | 0.5779 | 0.1529 | 0.645801 | 261 | 0.000053 |
| train | 3 | Car_Engine | 25872 | 22 | 0.000003 | 0.5857 | 0.1169 | 0.6153 | 0.4968 | 0.000008 | 49 | 0.000006 |
| train | 4 | Dog_Bark | 19181 | 32517 | 0.005298 | 0.5034 | 0.1427 | 0.5413 | 0.3896 | 0.017936 | 506 | 0.000082 |
| train | 5 | Doorbell | 1941 | 3100 | 0.004991 | 0.4180 | 0.1551 | 0.4363 | 0.3632 | 0.005835 | 32 | 0.000052 |
| train | 6 | Drilling | 23265 | 0 | 0.000000 | 0.6315 | 0.0882 | 0.6462 | 0.5874 | 0.000000 | 67 | 0.000009 |
| train | 7 | Footsteps | 25507 | 23508 | 0.002880 | 0.4828 | 0.1279 | 0.5065 | 0.4118 | 0.002963 | 122 | 0.000015 |
| train | 8 | Glass_Breaking | 8539 | 6797 | 0.002487 | 0.5080 | 0.1570 | 0.5161 | 0.4836 | 0.002568 | 98 | 0.000036 |
| train | 9 | Gunshot | 12405 | 12350 | 0.003111 | 0.5522 | 0.1819 | 0.5854 | 0.4526 | 0.003244 | 793 | 0.000200 |
| train | 10 | Help_Shouting | 16607 | 5156 | 0.000970 | 0.5661 | 0.1468 | 0.5919 | 0.4884 | 0.001068 | 1313 | 0.000247 |
| train | 11 | Jackhammer | 23925 | 0 | 0.000000 | 0.6420 | 0.0753 | 0.6601 | 0.5875 | 0.000000 | 0 | 0.000000 |
| train | 12 | Knocking | 8155 | 4037 | 0.001547 | 0.4380 | 0.1651 | 0.4732 | 0.3324 | 0.001586 | 158 | 0.000061 |
| train | 13 | Motorcycle | 18099 | 1191 | 0.000206 | 0.5547 | 0.1309 | 0.5854 | 0.4625 | 0.000215 | 148 | 0.000026 |
| train | 14 | Siren | 23426 | 12449 | 0.001661 | 0.5368 | 0.1464 | 0.5771 | 0.4158 | 0.006581 | 103 | 0.000014 |
| train | 15 | Train | 34045 | 3320 | 0.000305 | 0.5759 | 0.1265 | 0.6094 | 0.4755 | 0.000315 | 271 | 0.000025 |
| train | 16 | Vehicle_Horn | 8310 | 66 | 0.000025 | 0.6068 | 0.1153 | 0.6420 | 0.5011 | 0.000024 | 61 | 0.000023 |
| val | 0 | Aircraft | 5146 | 128 | 0.000078 | 0.5679 | 0.1497 | 0.6138 | 0.4302 | 0.000078 | 25 | 0.000015 |
| val | 1 | Alarm | 8465 | 17780 | 0.006564 | 0.5118 | 0.1727 | 0.5379 | 0.4337 | 0.006113 | 1085 | 0.000401 |
| val | 2 | Baby_Crying | 3086 | 133277 | 0.134961 | 0.4905 | 0.2298 | 0.5801 | 0.2217 | 0.535446 | 235 | 0.000238 |
| val | 3 | Car_Engine | 5560 | 13 | 0.000007 | 0.5815 | 0.1319 | 0.6189 | 0.4690 | 0.000000 | 3 | 0.000002 |
| val | 4 | Dog_Bark | 4656 | 4793 | 0.003217 | 0.5071 | 0.1367 | 0.5404 | 0.4074 | 0.003479 | 105 | 0.000070 |
| val | 5 | Doorbell | 428 | 307 | 0.002242 | 0.4333 | 0.1641 | 0.4508 | 0.3810 | 0.002453 | 0 | 0.000000 |
| val | 6 | Drilling | 5820 | 0 | 0.000000 | 0.6337 | 0.0913 | 0.6524 | 0.5776 | 0.000000 | 0 | 0.000000 |
| val | 7 | Footsteps | 5912 | 2092 | 0.001106 | 0.4837 | 0.1297 | 0.5096 | 0.4059 | 0.001116 | 27 | 0.000014 |
| val | 8 | Glass_Breaking | 2069 | 725 | 0.001095 | 0.5166 | 0.1498 | 0.5287 | 0.4805 | 0.001100 | 5 | 0.000008 |
| val | 9 | Gunshot | 3022 | 966 | 0.000999 | 0.5434 | 0.1928 | 0.5776 | 0.4408 | 0.001059 | 29 | 0.000030 |
| val | 10 | Help_Shouting | 3626 | 489 | 0.000421 | 0.5634 | 0.1409 | 0.5872 | 0.4918 | 0.000493 | 255 | 0.000220 |
| val | 11 | Jackhammer | 5181 | 0 | 0.000000 | 0.6392 | 0.0668 | 0.6549 | 0.5921 | 0.000000 | 0 | 0.000000 |
| val | 12 | Knocking | 1926 | 1152 | 0.001869 | 0.4455 | 0.1692 | 0.4813 | 0.3384 | 0.001921 | 15 | 0.000024 |
| val | 13 | Motorcycle | 3826 | 413 | 0.000337 | 0.5469 | 0.1317 | 0.5776 | 0.4548 | 0.000343 | 0 | 0.000000 |
| val | 14 | Siren | 5277 | 0 | 0.000000 | 0.5420 | 0.1372 | 0.5861 | 0.4098 | 0.000000 | 5 | 0.000003 |
| val | 15 | Train | 8624 | 1765 | 0.000640 | 0.5772 | 0.1271 | 0.6108 | 0.4764 | 0.000651 | 31 | 0.000011 |
| val | 16 | Vehicle_Horn | 2206 | 128 | 0.000181 | 0.5736 | 0.1273 | 0.6074 | 0.4722 | 0.000181 | 0 | 0.000000 |

- highest floor fraction: train/Baby_Crying = 0.163044 (15377 windows)
- highest saturation fraction: val/Alarm = 0.000401 (8465 windows)
- classes absent from the reported splits: none

## Artifacts

- normalization_per_class.csv
- source of bounds: C:\Users\BITPATNA\minorProject\data\processed\feature_stats.json
