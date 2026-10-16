Ground-truth components of the neurite in slices 16-19: [1, 1, 1, 3] (sections 15 / 20 / 25: [1, 3, 3]).


### clean sections

| method | neurite Dice in gap 16-19 | Dice all intermediate slices (all cells) | interface dist (px) | neurite components in slices 16-19 (clean only) |
|---|---|---|---|---|
| nearest (repeat) | 0.590 | 0.908 | 0.54 | [1, 1, 3, 3] |
| Raya-Udupa linear SDF | 0.085 | 0.816 | 1.69 | [6, 0, 0, 8] |
| cubic-spline SDF (binary fields) | 0.462 | 0.892 | 0.54 | [1, 7, 5, 7] |
| 4-point SDF (binary fields) | 0.361 | 0.897 | 0.49 | [1, 3, 3, 7] |
| harmonic | 0.417 | 0.872 | 1.66 | [1, 1, 0, 3] |
| heat-paths port + majority | 0.527 | 0.881 | 0.66 | [1, 1, 4, 4] |
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
| heat-paths port + majority | 0.507 | 0.872 | 0.71 |  |
| proposed (linear) | 0.000 | 0.786 | 3.25 |  |
| proposed, negative side truncated at 2 | 0.024 | 0.796 | 2.88 |  |
| proposed + label translation | 0.000 | 0.813 | 2.71 |  |
| proposed + component translation | 0.173 | 0.830 | 1.29 |  |
| proposed + persistence guard | 0.568 | 0.897 | 0.60 |  |
| component translation + persistence guard | 0.559 | 0.898 | 0.57 |  |
