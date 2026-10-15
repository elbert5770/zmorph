# MICrONS sections 17414–17416 (tile 170398_61241, 4 µm)

Three consecutive 40 nm sections of the MICrONS cortical mm³ segmentation, exported as segment-ID images
(about 100 cells per section, space-filling, no unlabelled background). Source: MICrONS Consortium data,
publicly released through MICrONS Explorer (https://www.microns-explorer.org/cortical-mm3). Please check the
current terms of use and citation requirements there; the exports below were made with the pipeline of
https://github.com/elbert5770/neuron_transport_3D_v3_archive and are processed derivatives, not the original release.

Files are gzip-compressed CSV (`np.loadtxt` reads `.csv.gz` directly). First row is a header of column indices;
each following row is one image row (first array axis). IDs are about 1e18, so load as int64.

| file | content | size |
|---|---|---|
| `sframe_{17414,17415,17416}_170398_61241_4.csv.gz` | smoothed by the original pipeline's mode filter (`Image_mode_filter_7.1a.jl`) | 501×501 |
| `frame_{17414,17416}_170398_61241_4.csv.gz` | raw segmentation, same tile | 501×501 |
| `frame_17415_170398_61241.csv.gz` | raw 17415 from an **older export** with a 500×500 grid. It is the top-left 500×500 crop of the 501×501 grid (99.6% agreement with the smoothed 17415 cropped that way, versus ≤97.4% for any other offset or flip), so the raw test crops everything to 500×500. | 500×500 |

The 501st row and column are the one-pixel overlap with neighbouring tiles in the original pipeline.
