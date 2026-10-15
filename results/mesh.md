| pipeline | triangles | min-angle p5 (deg) | dihedral mean / p95 (deg) | interface vertices shared (<1e-3 vox) | interface median / p95 vertex distance (vox) |
|---|---|---|---|---|---|
| repeat x5 + binary marching cubes (stair-step) | 212626 | 30.0 | 11.8 / 54.7 | 95.9% | 0.000 / 0.000 |
| proposed labels + smoothed-indicator meshes (sigma=1) | 177428 | 10.1 | 3.4 / 11.9 | 90.2% | 0.000 / 0.232 |
| proposed competition-field meshes | 179864 | 7.4 | 4.1 / 16.2 | 96.2% | 0.000 / 0.000 |
| proposed + joint Taubin (15 it.) | 179864 | 34.5 | 3.9 / 14.8 | 95.7% | 0.000 / 0.000 |