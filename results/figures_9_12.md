# heat_port vs sdf_align: the Figure 9 and Figure 12 gaps

## Figure 9 (zframe 87070-87075, sections 17414->17415)

Whole frame (501x501), smoothed inputs unless stated. 'step share' = fraction of the changed pixels converted in each of the 5 steps (ideal 0.2 each).

| method | sources kept | changed px | step shares | step CV | flicker px (share) | frags <12 px | changed px done by frame 4 | purple cell comps (6 frames) | purple area |
|---|---|---|---|---|---|---|---|---|---|
| heat_port [smoothed in] | True | 40966 | [0.108, 0.175, 0.191, 0.186, 0.34] | 0.38 | 0 (0.0%) | 2067 | 0.66 | [2, 10, 30, 27, 38, 2] | [15053, 15119, 15090, 15041, 15062, 14968] |
| heat_port + majority [smoothed in] | False | 40966 | [0.095, 0.162, 0.189, 0.242, 0.213] | 0.278 | 78 (0.2%) | 18 | 0.712 | [2, 2, 2, 2, 2, 2] | [15077, 15096, 15085, 15061, 15024, 15017] |
| sdf_align [smoothed in] | False | 40966 | [0.198, 0.217, 0.223, 0.211, 0.191] | 0.057 | 1637 (4.0%) | 12 | 0.809 | [2, 2, 2, 2, 2, 2] | [15053, 15026, 15003, 15029, 15004, 14968] |
| heat_port + majority [raw in] | False | 40887 | [0.095, 0.163, 0.188, 0.24, 0.208] | 0.274 | 95 (0.2%) | 58 | 0.711 | [3, 3, 2, 2, 2, 2] | [15069, 15090, 15073, 15056, 15033, 15026] |
| sdf_align [raw in] | False | 40887 | [0.2, 0.217, 0.214, 0.208, 0.195] | 0.041 | 1480 (3.6%) | 12 | 0.803 | [2, 2, 2, 2, 2, 2] | [15052, 15021, 14986, 15014, 14997, 14966] |

## Figure 12 (zframe 87075-87080, sections 17415->17416)

Whole frame (501x501), smoothed inputs unless stated. 'step share' = fraction of the changed pixels converted in each of the 5 steps (ideal 0.2 each).

| method | sources kept | changed px | step shares | step CV | flicker px (share) | frags <12 px | changed px done by frame 4 | purple cell comps (6 frames) | purple area |
|---|---|---|---|---|---|---|---|---|---|
| heat_port [smoothed in] | True | 38912 | [0.096, 0.178, 0.2, 0.187, 0.339] | 0.393 | 0 (0.0%) | 2081 | 0.661 | [2, 6, 25, 44, 62, 1] | [14968, 15102, 15214, 15206, 15243, 15191] |
| heat_port + majority [smoothed in] | False | 38912 | [0.089, 0.164, 0.196, 0.247, 0.212] | 0.294 | 83 (0.2%) | 50 | 0.714 | [2, 2, 2, 2, 10, 1] | [14996, 15100, 15208, 15239, 15209, 15197] |
| sdf_align [smoothed in] | False | 38912 | [0.185, 0.21, 0.246, 0.213, 0.195] | 0.099 | 1873 (4.8%) | 12 | 0.805 | [2, 2, 2, 1, 1, 1] | [14968, 15033, 15117, 15221, 15252, 15191] |
| heat_port + majority [raw in] | False | 38618 | [0.089, 0.161, 0.192, 0.249, 0.212] | 0.3 | 101 (0.3%) | 70 | 0.71 | [2, 2, 2, 2, 6, 1] | [14992, 15096, 15231, 15275, 15239, 15177] |
| sdf_align [raw in] | False | 38618 | [0.192, 0.219, 0.234, 0.201, 0.189] | 0.082 | 1412 (3.7%) | 34 | 0.809 | [2, 2, 2, 1, 1, 1] | [14966, 15028, 15075, 15178, 15208, 15162] |

## Held-out check: predict section 17415 from 17414 and 17416 (80 nm gap), scored against the real 17415

| method | input | acc (frame) | acc where inputs differ | boundary dist | Dice | acc (crop) |
|---|---|---|---|---|---|---|
| heat_port | smoothed | 0.9166 | 0.7267 | 1.752 | 0.8813 | 0.8740 |
| heat_port + majority | smoothed | 0.9202 | 0.7389 | 1.670 | 0.8863 | 0.8786 |
| sdf_align | smoothed | 0.9420 | 0.8122 | 1.269 | 0.9218 | 0.9161 |
| heat_port | raw | 0.9159 | 0.7243 | 1.748 | 0.8808 | 0.8728 |
| heat_port + majority | raw | 0.9202 | 0.7388 | 1.626 | 0.8866 | 0.8777 |
| sdf_align | raw | 0.9453 | 0.8231 | 1.180 | 0.9250 | 0.9205 |
