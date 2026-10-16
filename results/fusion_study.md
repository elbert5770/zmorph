Combining heat2 and the SDF method (no post-filter). bd = mean interface distance in px; noisy = mean of seeds.

| method | arch Dice / bd | tilted Dice / bd | arch noisy Dice / bd | split3 gap Dice | real smoothed acc / bd / Dice | real raw acc / bd / Dice |
|---|---|---|---|---|---|---|
| nearest (repeat) | 0.962 / 0.40 | 0.880 / 0.81 | 0.936 / 0.74 | 0.590 | 0.8368 / 3.46 / 0.761 | 0.8372 / 3.39 / 0.762 |
| heat, published | 0.964 / 0.36 | 0.914 / 0.66 | 0.942 / 0.69 | 0.628 | 0.9166 / 1.75 / 0.881 | 0.9160 / 1.70 / 0.881 |
| heat2 | 0.980 / 0.18 | 0.961 / 0.26 | 0.964 / 0.33 | 0.665 | 0.9336 / 1.40 / 0.909 | 0.9355 / 1.34 / 0.911 |
| sdf (proposed) | 0.983 / 0.18 | 0.883 / 0.71 | 0.971 / 0.29 | 0.000 | 0.9396 / 1.30 / 0.914 | 0.9432 / 1.20 / 0.918 |
| sdf + translation | 0.983 / 0.18 | 0.938 / 0.39 | 0.971 / 0.29 | 0.121 | 0.9420 / 1.27 / 0.922 | 0.9456 / 1.18 / 0.925 |
| sdf + persistence guard | 0.983 / 0.18 | 0.883 / 0.71 | 0.971 / 0.29 | 0.580 | 0.9396 / 1.30 / 0.914 | 0.9432 / 1.20 / 0.918 |
| sdf + tube links + guard | 0.983 / 0.18 | 0.883 / 0.71 | 0.971 / 0.29 | 0.927 | 0.9396 / 1.30 / 0.914 | 0.9432 / 1.20 / 0.918 |
| FUSION sdf_rank (heat switching, SDF order) | 0.986 / 0.12 | 0.970 / 0.20 | 0.968 / 0.31 | 0.555 | 0.9368 / 1.39 / 0.909 | 0.9391 / 1.31 / 0.912 |
| FUSION rank_avg (mean of heat2 and sdf_rank times) | 0.986 / 0.13 | 0.969 / 0.21 | 0.967 / 0.31 | 0.621 | 0.9377 / 1.33 / 0.913 | 0.9398 / 1.25 / 0.915 |
| FUSION time_avg (heat2 and raw SDF crossing, w=0.5) | 0.984 / 0.15 | 0.942 / 0.37 | 0.970 / 0.30 | 0.537 | 0.9382 / 1.32 / 0.914 | 0.9407 / 1.25 / 0.917 |
| FUSION refine (heat2 frames, SDF smoothing) | 0.980 / 0.18 | 0.963 / 0.24 | 0.969 / 0.28 | 0.649 | 0.9341 / 1.40 / 0.910 | 0.9360 / 1.34 / 0.912 |
| FUSION guard (SDF, vanishing labels restored from heat2) | 0.983 / 0.18 | 0.883 / 0.71 | 0.971 / 0.29 | 0.649 | 0.9396 / 1.30 / 0.914 | 0.9432 / 1.20 / 0.918 |
| FUSION field (SDF blend fused with SDF of heat2 slice, w=0.5) | 0.982 / 0.17 | 0.935 / 0.41 | 0.972 / 0.26 | 0.166 | 0.9406 / 1.28 / 0.918 | 0.9427 / 1.21 / 0.920 |
