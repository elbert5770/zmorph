# Moving walls and membrane patches in Brownian dynamics

D = 1, kT = 1, lengths in nm; errors are single runs unless stated.

## Wall force from reflections (planar gap)

Measured total force on both walls / (2 N kT / w), 20000 particles.

| w (nm) | step 0.5 nm | step 1 nm | step 2 nm |
|---|---|---|---|
| 12 | 1.0002 | 0.9994 | 1.0005 |
| 20 | 1.0006 | 1.0001 | 0.9997 |
| 40 | 1.0016 | 1.0024 | 0.9954 |

## Piston: work against the ideal-gas isotherm

Work done on the gas / (N kT ln(w0/w1)); 1.0 = reversible. Pe = wall speed x gap / D.

| run | Pe | work / ideal | max density deviation from uniform | particles outside |
|---|---|---|---|---|
| compress 40->20 T=10000 | 0.06 | 1.006 | 0.044 | 0 |
| compress 40->20 T=1000 | 0.60 | 1.048 | 0.030 | 0 |
| compress 40->20 T=100 | 6.00 | 1.482 | 0.260 | 0 |
| expand 20->40 T=10000 | 0.06 | 0.995 | 0.050 | 0 |
| expand 20->40 T=1000 | 0.60 | 0.955 | 0.038 | 0 |
| expand 20->40 T=100 | 6.00 | 0.657 | 0.324 | 0 |

## Partially absorbing wall (Robin), decay of survivors

Absorption probability per hit p = kappa sqrt(pi dt / D). Exact rate from k tan(kL) = kappa/D, rate = D k^2.

| Bi = kappa w / D | step (nm) | p per hit | measured rate | exact rate | error |
|---|---|---|---|---|---|
| 0.5 | 1.0 | 0.031 | 1.078e-03 | 1.067e-03 | +1.0% |
| 0.5 | 0.5 | 0.016 | 1.081e-03 | 1.067e-03 | +1.4% |
| 0.5 | 0.25 | 0.008 | 1.072e-03 | 1.067e-03 | +0.5% |
| 2 | 1.0 | 0.125 | 3.033e-03 | 2.899e-03 | +4.6% |
| 2 | 0.5 | 0.063 | 2.940e-03 | 2.899e-03 | +1.4% |
| 2 | 0.25 | 0.031 | 2.945e-03 | 2.899e-03 | +1.6% |

## Checkerboard of absorbing patches (half of one wall)

Fine mosaic (patch tiles 8 nm vs gap 20 nm) should act like a uniform wall with kappa/2.

| Bi | measured rate | effective-Robin prediction | error | full coverage would be |
|---|---|---|---|---|
| 0.5 | 5.723e-04 | 5.762e-04 | -0.7% | +85.2% |
| 1 | 1.064e-03 | 1.067e-03 | -0.3% | +73.4% |

## Absorbing cap on the spherical shell

| cap half-angle | cap area fraction | absorbed fraction | largest polar angle of an absorption |
|---|---|---|---|
| 20 deg | 0.030 | 0.293 | 21.0 deg |
| 40 deg | 0.117 | 0.492 | 41.4 deg |
| 60 deg | 0.250 | 0.687 | 61.5 deg |
