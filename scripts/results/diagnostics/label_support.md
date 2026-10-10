# Window label-support proxy (nearest-centroid margin)

## Definition

Centroids mu_c are mean train features per class (320 values = 64 x 5 x 1). For each window x with clip label c:

```
margin(x) = min_{j != c} ||x - mu_j||^2  -  ||x - mu_c||^2
centroid-confusable window  <=>  margin(x) <= 0
```

A positive margin means the window is closer to its own class centroid than to every other class centroid. The proxy is content based: all windows inherit the clip label, this measures whether the window's spectrogram supports that label.
This is a separability proxy, not a noise estimate.
Centroids always come from the train split, even when only val is reported.

## Per class and split

| split | class_id | class_name | n_windows | frac_centroid_confusable | mean_margin | median_margin | train_windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | Aircraft | 23873 | 0.7431 | -2.1191 | -0.9359 | 23873 |
| train | 1 | Alarm | 36086 | 0.9823 | -4.4176 | -3.2539 | 36086 |
| train | 2 | Baby_Crying | 15377 | 0.2508 | 3.3060 | 6.9535 | 15377 |
| train | 3 | Car_Engine | 25872 | 0.8430 | -1.6861 | -1.0433 | 25872 |
| train | 4 | Dog_Bark | 19181 | 0.8460 | -2.2800 | -1.5134 | 19181 |
| train | 5 | Doorbell | 1941 | 0.4472 | -1.4716 | 0.5622 | 1941 |
| train | 6 | Drilling | 23265 | 0.6398 | -0.7733 | -0.1911 | 23265 |
| train | 7 | Footsteps | 25507 | 0.8703 | -2.4277 | -1.5386 | 25507 |
| train | 8 | Glass_Breaking | 8539 | 0.8008 | -4.3447 | -3.2698 | 8539 |
| train | 9 | Gunshot | 12405 | 0.9733 | -5.7494 | -4.6486 | 12405 |
| train | 10 | Help_Shouting | 16607 | 0.8425 | -3.1460 | -1.8170 | 16607 |
| train | 11 | Jackhammer | 23925 | 0.4415 | -0.2166 | 0.1376 | 23925 |
| train | 12 | Knocking | 8155 | 0.7064 | -2.1240 | -0.8644 | 8155 |
| train | 13 | Motorcycle | 18099 | 0.8833 | -2.6637 | -1.4956 | 18099 |
| train | 14 | Siren | 23426 | 0.8811 | -2.7359 | -2.0138 | 23426 |
| train | 15 | Train | 34045 | 0.9600 | -2.2823 | -1.2512 | 34045 |
| train | 16 | Vehicle_Horn | 8310 | 0.6939 | -1.0400 | -0.5039 | 8310 |
| val | 0 | Aircraft | 5146 | 0.6862 | -2.2787 | -0.8677 | 23873 |
| val | 1 | Alarm | 8465 | 0.9797 | -4.3361 | -3.1392 | 36086 |
| val | 2 | Baby_Crying | 3086 | 0.3756 | 0.3331 | 5.8201 | 15377 |
| val | 3 | Car_Engine | 5560 | 0.8129 | -2.1714 | -2.0906 | 25872 |
| val | 4 | Dog_Bark | 4656 | 0.8402 | -2.2379 | -1.1316 | 19181 |
| val | 5 | Doorbell | 428 | 0.6332 | -1.8157 | -0.8411 | 1941 |
| val | 6 | Drilling | 5820 | 0.7323 | -0.7040 | -0.3734 | 23265 |
| val | 7 | Footsteps | 5912 | 0.8850 | -2.6119 | -1.5512 | 25507 |
| val | 8 | Glass_Breaking | 2069 | 0.8376 | -4.3202 | -3.0356 | 8539 |
| val | 9 | Gunshot | 3022 | 0.9854 | -6.4652 | -4.7109 | 12405 |
| val | 10 | Help_Shouting | 3626 | 0.8434 | -2.7582 | -1.4868 | 16607 |
| val | 11 | Jackhammer | 5181 | 0.2291 | 0.0903 | 0.1813 | 23925 |
| val | 12 | Knocking | 1926 | 0.7290 | -2.5846 | -0.8768 | 8155 |
| val | 13 | Motorcycle | 3826 | 0.8876 | -2.7981 | -1.4973 | 18099 |
| val | 14 | Siren | 5277 | 0.8672 | -1.8695 | -0.8498 | 23426 |
| val | 15 | Train | 8624 | 0.9766 | -2.3499 | -1.4205 | 34045 |
| val | 16 | Vehicle_Horn | 2206 | 0.6306 | -2.0677 | -0.6097 | 8310 |

## Highest centroid-confusable fractions

- val/Gunshot: 0.9854 centroid-confusable (3022 windows)
- train/Alarm: 0.9823 centroid-confusable (36086 windows)
- val/Alarm: 0.9797 centroid-confusable (8465 windows)
- val/Train: 0.9766 centroid-confusable (8624 windows)
- train/Gunshot: 0.9733 centroid-confusable (12405 windows)

## Artifacts

- label_support.csv
