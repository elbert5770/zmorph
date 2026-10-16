160 held-out sections, tiles: ['170398_60741', '170398_61241', '170898_60741', '170898_61241']

| method | acc | acc where inputs differ | boundary dist (px) | Dice | comp. error / section | wins vs sdf_align (acc) |
|---|---|---|---|---|---|---|
| nearest | 0.8506 | 0.4691 | 2.725 | 0.7863 | 44.3 | 0.00 |
| sdf | 0.9369 | 0.7841 | 1.217 | 0.9100 | 35.3 | 0.40 |
| sdf_align | 0.9372 | 0.7856 | 1.213 | 0.9110 | 34.7 | 0.00 |
| sdf_align_tube_persist | 0.9372 | 0.7854 | 1.214 | 0.9110 | 34.8 | 0.08 |
| heat_port | 0.9173 | 0.7135 | 1.446 | 0.8831 | 591.6 | 0.00 |
| heat2 | 0.9289 | 0.7549 | 1.319 | 0.9025 | 73.4 | 0.00 |
| rank_avg | 0.9338 | 0.7730 | 1.229 | 0.9086 | 69.4 | 0.11 |
| tube_guard | 0.9369 | 0.7842 | 1.216 | 0.9101 | 35.2 | 0.40 |

Per-tile accuracy (mean):

| method | 170398_60741 | 170398_61241 | 170898_60741 | 170898_61241 |
|---|---|---|---|---|
| nearest | 0.8436 | 0.8534 | 0.8606 | 0.8448 |
| sdf | 0.9334 | 0.9374 | 0.9414 | 0.9353 |
| sdf_align | 0.9336 | 0.9376 | 0.9419 | 0.9359 |
| sdf_align_tube_persist | 0.9336 | 0.9377 | 0.9416 | 0.9358 |
| heat_port | 0.9123 | 0.9186 | 0.9236 | 0.9146 |
| heat2 | 0.9255 | 0.9302 | 0.9333 | 0.9265 |
| rank_avg | 0.9307 | 0.9347 | 0.9382 | 0.9316 |
| tube_guard | 0.9333 | 0.9375 | 0.9415 | 0.9352 |

Scan, pair (i-1, i+1), 80 nm, totals over 160 sections: labels in both 17586, labels with no overlap 90 (0.51%), non-overlapping component pairs 17652, accepted tube links 133

Scan, pair (i-1, i), 40 nm, totals over 160 sections: labels in both 18218, labels with no overlap 15 (0.08%), non-overlapping component pairs 18337, accepted tube links 59

Run time per method (s, mean): nearest 0.3, sdf 6.2, sdf_align 11.3, sdf_align_tube_persist 12.5, heat_port 9.1, heat2 8.2, rank_avg 12.8, tube_guard 15.4

Paired differences in pixel accuracy (160 sections, mean ± s.e.m.):
- sdf_align − sdf: +3.4e-4 ± 0.8e-4 (better in 59%)
- sdf_align − heat2: +83e-4 ± 2.6e-4 (better in 100%)
- sdf_align − rank_avg: +34e-4 ± 2.4e-4 (better in 89%)
- heat2 − heat_port: +117e-4 ± 4.2e-4 (better in 100%)
- sdf_align_tube_persist − sdf_align: −0.7e-4 ± 0.6e-4 (no gain; −1.1e-4 in the 88 sections with an accepted link)
