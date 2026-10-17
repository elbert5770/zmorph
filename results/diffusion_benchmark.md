# Diffusion in the gap between cells: Brownian dynamics vs continuum solvers

D = 1, lengths in nm, voxel size of the data 8 nm. Brownian dynamics (BD) runs on the implicit gap {margin < w} of the signed-distance fields (`zmorph/bd.py`); the continuum solvers are in `zmorph/diffusion_ref.py`. Errors are relative to the analytic or fine-grid answer; BD errors are +- one standard error over 10 particle groups.

## 1. Reflecting walls (planar gap, w = 20 nm)

| step sigma (nm) | density in the outer 20% of the gap / uniform | in-plane D / D_free | particles outside the gap |
|---|---|---|---|
| 0.5 | 0.989 | 1.000 +- 0.009 | 0 |
| 1.0 | 0.990 | 1.005 +- 0.007 | 0 |
| 2.0 | 0.991 | 1.004 +- 0.011 | 0 |
| 4.0 | 1.029 | 1.003 +- 0.012 | 0 |

A reflecting wall that biased the density would show a ratio away from 1.0. None does, up to steps of w/5.

## 2. Periodic lattice of cubic cells (cell side 200 nm, gap w on every face)

Tortuosity lambda^2 = D / D_ECS (thin-sheet limit 1.5). 'ECS fraction' is the share of the volume the solver treats as gap; the true value is 1 - ((a-w)/a)^3.

| w (nm) | true ECS fraction | BD, analytic geometry | BD, from labels | voxel FV 8 nm | voxel FV 4 nm | voxel FV 2 nm |
|---|---|---|---|---|---|---|
| 12 | 0.169 | 1.406 +- 0.028 | - | 1.426 (alpha 0.221) | 1.463 (alpha 0.115) | - |
| 20 | 0.271 | 1.386 +- 0.027 | 1.376 +- 0.021 | 1.426 (alpha 0.221) | 1.417 (alpha 0.221) | 1.392 (alpha 0.271) |
| 32 | 0.407 | 1.332 +- 0.022 | - | 1.341 (alpha 0.407) | 1.334 (alpha 0.407) | - |

Voxel solvers quantise the gap: at 8 nm the 12 nm and 20 nm gaps give the identical 16 nm mask (same answer in both rows). lambda^2 is forgiving because it is a ratio, but the ECS volume fraction, which any flow or osmotic calculation needs, is not.

## 3. Ball in socket: gap shell around a sphere of radius 200 nm (w = 20 nm)

Slowest decay rate of diffusion in the shell. Exact (spherical shell, reflecting walls): 4.9958e-05 per time unit (thin-film limit 2D/R^2 = 5.0000e-05).

| method | geometry source | resolution | rate error vs exact |
|---|---|---|---|
| Brownian dynamics | analytic | h = 8 nm | -0.8% +- 1.0% |
| Brownian dynamics | analytic | h = 2 nm | +1.5% +- 0.9% |
| Brownian dynamics | voxelised labels | h = 8 nm | -0.9% +- 0.6% |
| Brownian dynamics | labels, smoothed field | h = 8 nm | -0.1% +- 0.9% |
| voxel finite volumes | analytic mask | h = 8 nm | -18.7% |
| voxel finite volumes | analytic mask | h = 4 nm | -9.1% |
| voxel finite volumes | analytic mask | h = 3 nm | -6.8% |
| surface mesh, cotangent Laplacian | analytic | h = 8 nm | -2.7% |
| surface mesh, cotangent Laplacian | analytic | h = 4 nm | -0.6% |
| surface mesh, cotangent Laplacian | analytic | h = 2 nm | -0.1% |
| surface mesh, cotangent Laplacian | voxelised labels, raw | h = 8 nm | -7.3% |
| surface mesh, cotangent Laplacian | voxelised labels, raw | h = 4 nm | -7.7% |
| surface mesh, cotangent Laplacian | labels + smoothed field | h = 8 nm | +0.6% |
| surface mesh, cotangent Laplacian | labels + smoothed field | h = 4 nm | +0.1% |
| surface mesh, cotangent Laplacian | labels + Taubin | h = 8 nm | -0.4% |
| surface mesh, cotangent Laplacian | labels + Taubin | h = 4 nm | -0.6% |
