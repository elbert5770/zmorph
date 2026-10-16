"""Overlap-tile processing: run a z-interpolation method tile by tile with a halo of real neighbour data.

Every method here has a limited lateral reach (smoothing, search windows, the lateral decay of the heat field), but
none of them can see across the edge of the array it is given: smoothing replicates the border, the heat solve imposes
no-flux (mirror) conditions there, and cells cut by the edge look as if they appeared or ended. When the adjacent
frames are available, pad each tile with a halo taken from them, interpolate the padded tile, and keep only the centre.
"""
import numpy as np


def tile_slices(n, tile):
    """Split range(n) into consecutive chunks of about `tile` (last chunk absorbs the remainder)."""
    k = max(1, int(round(n / tile)))
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def reconstruct_tiled(sections, factor, method, tile=250, halo=32):
    """`method(sections, factor) -> volume`. `sections` are full-size 2-D label images (the neighbours of every
    tile are inside them). Returns the assembled volume of shape ((S-1)*factor+1, ny, nx)."""
    sections = [np.asarray(s) for s in sections]
    ny, nx = sections[0].shape
    nz = (len(sections) - 1) * factor + 1
    out = np.zeros((nz, ny, nx), sections[0].dtype)
    for (y0, y1) in tile_slices(ny, tile):
        for (x0, x1) in tile_slices(nx, tile):
            ya, yb = max(0, y0 - halo), min(ny, y1 + halo)
            xa, xb = max(0, x0 - halo), min(nx, x1 + halo)
            vol = method([s[ya:yb, xa:xb] for s in sections], factor)
            out[:, y0:y1, x0:x1] = vol[:, y0 - ya:y1 - ya, x0 - xa:x1 - xa]
    return out
