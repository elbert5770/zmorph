## Synthetic phantom (mean of seeds for noisy sections)

### branch / clean sections

| method | acc_mid | dice_mid | bdist_mid | component_err | acc_sec | area_ratio |
|---|---|---|---|---|---|---|
| Raya-Udupa / Herman: linear binary SDF | 0.989 | 0.972 | 0.257 | 0.0 | 1.000 | 0.994 |
| cubic-spline SDF, binary fields (Grevera-Udupa) | 0.990 | 0.974 | 0.234 | 0.0 | 1.000 | 0.998 |
| 4-point SDF, binary fields (Kels-Dyn weights) | 0.990 | 0.973 | 0.234 | 0.0 | 1.000 | 1.000 |
| linear, absent depth 4 | 0.989 | 0.975 | 0.237 | 0.0 | 1.000 | 0.995 |
| linear, + half-pixel offset | 0.991 | 0.979 | 0.215 | 0.0 | 1.000 | 0.994 |
| linear, + smoothing = proposed (sdf) | 0.993 | 0.983 | 0.176 | 0.0 | 1.000 | 0.990 |
| cubic spline, proposed fields | 0.994 | 0.984 | 0.153 | 0.0 | 1.000 | 0.993 |
| 4-point, proposed fields | 0.994 | 0.985 | 0.145 | 0.0 | 1.000 | 0.993 |
| 4-point, proposed fields, sigma=2 | 0.992 | 0.979 | 0.198 | 0.0 | 0.995 | 0.988 |
| proposed + translation | 0.993 | 0.983 | 0.176 | 0.0 | 1.000 | 0.990 |

### branch / noisy sections

| method | acc_mid | dice_mid | bdist_mid | component_err | acc_sec | area_ratio |
|---|---|---|---|---|---|---|
| Raya-Udupa / Herman: linear binary SDF | 0.977 | 0.948 | 0.512 | 64.4 | 0.981 | 1.292 |
| cubic-spline SDF, binary fields (Grevera-Udupa) | 0.976 | 0.946 | 0.585 | 47.8 | 0.981 | 1.326 |
| 4-point SDF, binary fields (Kels-Dyn weights) | 0.976 | 0.946 | 0.585 | 52.4 | 0.981 | 1.322 |
| linear, absent depth 4 | 0.977 | 0.951 | 0.505 | 64.4 | 0.981 | 1.294 |
| linear, + half-pixel offset | 0.979 | 0.955 | 0.463 | 64.4 | 0.981 | 1.265 |
| linear, + smoothing = proposed (sdf) | 0.987 | 0.970 | 0.298 | 1.6 | 0.990 | 1.025 |
| cubic spline, proposed fields | 0.986 | 0.968 | 0.396 | 6.0 | 0.990 | 1.050 |
| 4-point, proposed fields | 0.986 | 0.968 | 0.386 | 6.0 | 0.990 | 1.051 |
| 4-point, proposed fields, sigma=2 | 0.987 | 0.969 | 0.310 | 1.0 | 0.989 | 1.006 |
| proposed + translation | 0.987 | 0.970 | 0.298 | 1.6 | 0.990 | 1.025 |

### tilted / clean sections

| method | acc_mid | dice_mid | bdist_mid | component_err | acc_sec | area_ratio |
|---|---|---|---|---|---|---|
| Raya-Udupa / Herman: linear binary SDF | 0.986 | 0.894 | 0.636 | 0.0 | 1.000 | 0.965 |
| cubic-spline SDF, binary fields (Grevera-Udupa) | 0.992 | 0.947 | 0.341 | 0.0 | 1.000 | 1.006 |
| 4-point SDF, binary fields (Kels-Dyn weights) | 0.992 | 0.946 | 0.350 | 0.0 | 1.000 | 1.005 |
| linear, absent depth 4 | 0.986 | 0.894 | 0.636 | 0.0 | 1.000 | 0.965 |
| linear, + half-pixel offset | 0.985 | 0.888 | 0.671 | 0.0 | 1.000 | 0.961 |
| linear, + smoothing = proposed (sdf) | 0.985 | 0.883 | 0.706 | 0.0 | 1.000 | 0.947 |
| cubic spline, proposed fields | 0.989 | 0.921 | 0.514 | 0.0 | 1.000 | 1.033 |
| 4-point, proposed fields | 0.989 | 0.921 | 0.516 | 0.0 | 1.000 | 1.052 |
| 4-point, proposed fields, sigma=2 | 0.986 | 0.902 | 0.647 | 0.0 | 0.996 | 1.015 |
| proposed + translation | 0.991 | 0.938 | 0.390 | 0.0 | 1.000 | 0.972 |

### tilted / noisy sections

| method | acc_mid | dice_mid | bdist_mid | component_err | acc_sec | area_ratio |
|---|---|---|---|---|---|---|
| Raya-Udupa / Herman: linear binary SDF | 0.982 | 0.866 | 0.775 | 26.4 | 0.991 | 1.185 |
| cubic-spline SDF, binary fields (Grevera-Udupa) | 0.987 | 0.912 | 0.542 | 15.2 | 0.991 | 1.241 |
| 4-point SDF, binary fields (Kels-Dyn weights) | 0.987 | 0.911 | 0.550 | 17.2 | 0.991 | 1.247 |
| linear, absent depth 4 | 0.982 | 0.866 | 0.775 | 26.4 | 0.991 | 1.185 |
| linear, + half-pixel offset | 0.982 | 0.863 | 0.793 | 26.4 | 0.991 | 1.166 |
| linear, + smoothing = proposed (sdf) | 0.983 | 0.870 | 0.794 | 0.4 | 0.995 | 0.972 |
| cubic spline, proposed fields | 0.985 | 0.902 | 0.654 | 0.2 | 0.995 | 1.081 |
| 4-point, proposed fields | 0.985 | 0.902 | 0.658 | 0.6 | 0.995 | 1.099 |
| 4-point, proposed fields, sigma=2 | 0.984 | 0.890 | 0.738 | 0.2 | 0.993 | 1.039 |
| proposed + translation | 0.989 | 0.922 | 0.494 | 0.4 | 0.995 | 0.995 |

## Real sections (MICrONS 17414-17416, 80 nm held-out)

Only two input sections exist for the held-out middle section, so the higher-order variants reduce to linear interpolation and are not shown; the ladder isolates the field construction.


### smoothed 17414/17416 -> smoothed 17415

| method | pixel acc. | acc. where inputs differ | boundary dist (px) | mean Dice |
|---|---|---|---|---|
| Raya-Udupa / Herman: linear binary SDF | 0.9403 | 0.8060 | 1.346 | 0.9139 |
| linear, absent depth 4 | 0.9384 | 0.7997 | 1.312 | 0.9121 |
| linear, + half-pixel offset | 0.9390 | 0.8017 | 1.309 | 0.9130 |
| linear, + smoothing = proposed (sdf) | 0.9396 | 0.8037 | 1.299 | 0.9141 |
| proposed + translation | 0.9420 | 0.8122 | 1.269 | 0.9218 |

### raw 17414/17416 -> raw 17415 (500x500)

| method | pixel acc. | acc. where inputs differ | boundary dist (px) | mean Dice |
|---|---|---|---|---|
| Raya-Udupa / Herman: linear binary SDF | 0.9440 | 0.8182 | 1.251 | 0.9170 |
| linear, absent depth 4 | 0.9418 | 0.8108 | 1.213 | 0.9157 |
| linear, + half-pixel offset | 0.9426 | 0.8136 | 1.205 | 0.9170 |
| linear, + smoothing = proposed (sdf) | 0.9432 | 0.8154 | 1.199 | 0.9180 |
| proposed + translation | 0.9456 | 0.8238 | 1.182 | 0.9251 |
