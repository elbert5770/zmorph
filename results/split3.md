Ground-truth components of the neurite in slices 16-19: [1, 1, 1, 3] (sections 15 / 20 / 25: [1, 3, 3]).


### clean sections

| method | neurite Dice in gap 16-19 | Dice all intermediate slices (all cells) | interface dist (px) | neurite components in slices 16-19 (clean only) |
|---|---|---|---|---|
| nearest (repeat) | 0.590 | 0.908 | 0.54 | [1, 1, 3, 3] |
| Raya-Udupa linear SDF | 0.085 | 0.816 | 1.69 | [6, 0, 0, 8] |
| cubic-spline SDF (binary fields) | 0.462 | 0.892 | 0.54 | [1, 7, 5, 7] |
| 4-point SDF (binary fields) | 0.361 | 0.897 | 0.49 | [1, 3, 3, 7] |
| harmonic | 0.417 | 0.872 | 1.66 | [1, 1, 0, 3] |
| heat-paths port (raw, no filter) | 0.628 | 0.892 | 0.68 | [1, 6, 9, 8] |
| proposed (linear) | 0.000 | 0.798 | 3.02 | [0, 0, 0, 0] |
| proposed, negative side truncated at 2 | 0.052 | 0.813 | 2.32 | [1, 0, 0, 1] |
| proposed + label translation | 0.128 | 0.834 | 1.64 | [1, 1, 1, 1] |
| proposed + component translation | 0.294 | 0.855 | 0.78 | [1, 3, 3, 7] |
| proposed + persistence guard | 0.580 | 0.911 | 0.50 | [1, 1, 3, 3] |
| component translation + persistence guard | 0.483 | 0.907 | 0.47 | [1, 1, 3, 7] |

### noisy sections

| method | neurite Dice in gap 16-19 | Dice all intermediate slices (all cells) | interface dist (px) | neurite components in slices 16-19 (clean only) |
|---|---|---|---|---|
| nearest (repeat) | 0.571 | 0.875 | 0.74 |  |
| Raya-Udupa linear SDF | 0.053 | 0.788 | 1.85 |  |
| cubic-spline SDF (binary fields) | 0.429 | 0.862 | 0.74 |  |
| 4-point SDF (binary fields) | 0.330 | 0.867 | 0.66 |  |
| harmonic | 0.381 | 0.854 | 1.78 |  |
| heat-paths port (raw, no filter) | 0.620 | 0.868 | 0.82 |  |
| proposed (linear) | 0.000 | 0.786 | 3.25 |  |
| proposed, negative side truncated at 2 | 0.024 | 0.796 | 2.88 |  |
| proposed + label translation | 0.000 | 0.813 | 2.71 |  |
| proposed + component translation | 0.173 | 0.830 | 1.29 |  |
| proposed + persistence guard | 0.568 | 0.897 | 0.60 |  |
| component translation + persistence guard | 0.559 | 0.898 | 0.57 |  |

### How each method carries the neurite through the gap (clean sections): neurite pixels / components per slice

| method | z=15 (section) | 16 | 17 | 18 | 19 | z=20 (section) |
|---|---|---|---|---|---|---|
| **ground truth** | 115 / 1 | 180 / 1 | 255 / 1 | 309 / 1 | 309 / 3 | 309 / 3 |
| nearest (repeat) | 115 / 1 | 115 / 1 | 115 / 1 | 309 / 3 | 309 / 3 | 309 / 3 |
| Raya-Udupa linear SDF | 115 / 1 | 21 / 6 | 0 / 0 | 0 / 0 | 26 / 8 | 309 / 3 |
| cubic-spline SDF (binary fields) | 115 / 1 | 98 / 1 | 38 / 7 | 28 / 5 | 163 / 7 | 309 / 3 |
| 4-point SDF (binary fields) | 115 / 1 | 74 / 1 | 11 / 3 | 11 / 3 | 138 / 7 | 309 / 3 |
| harmonic | 115 / 1 | 95 / 1 | 12 / 1 | 0 / 0 | 244 / 3 | 309 / 3 |
| heat-paths port (raw, no filter) | 115 / 1 | 101 / 1 | 97 / 6 | 183 / 9 | 283 / 8 | 309 / 3 |
| proposed (linear) | 111 / 1 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 295 / 3 |
| proposed, negative side truncated at 2 | 111 / 1 | 21 / 1 | 0 / 0 | 0 / 0 | 7 / 1 | 295 / 3 |
| proposed + label translation | 111 / 1 | 35 / 1 | 5 / 1 | 4 / 1 | 28 / 1 | 295 / 3 |
| proposed + component translation | 111 / 1 | 85 / 1 | 25 / 3 | 14 / 3 | 60 / 7 | 295 / 3 |
| proposed + persistence guard | 111 / 1 | 111 / 1 | 111 / 1 | 295 / 3 | 295 / 3 | 295 / 3 |
| component translation + persistence guard | 111 / 1 | 85 / 1 | 133 / 1 | 303 / 3 | 60 / 7 | 295 / 3 |
