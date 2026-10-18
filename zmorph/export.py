"""Write fields and meshes in formats ParaView (and most 3-D viewers) open directly.

  write_vti(path, arrays, spacing, origin)   VTK XML ImageData: one or more 3-D arrays on a regular grid (signed distance,
                                             margin, labels, masks). Open in ParaView, then Slice, Contour (isosurface at 0, or at
                                             the gap level), Volume rendering or Clip.
  write_ply(path, meshes, spacing, origin)   binary little-endian PLY with one colour per cell (all cells in one file).

Arrays are indexed (z, y, x) as everywhere in zmorph; the file stores them with x fastest, so ParaView's X axis is the last
array axis. No dependencies beyond numpy.
"""
import base64
import colorsys
import struct
import zlib
import numpy as np

_VTK = {np.dtype(np.float32): "Float32", np.dtype(np.float64): "Float64", np.dtype(np.uint8): "UInt8", np.dtype(np.int32): "Int32",
        np.dtype(np.uint32): "UInt32", np.dtype(np.int64): "Int64", np.dtype(np.int16): "Int16", np.dtype(np.uint16): "UInt16"}


def _b64(a, compress=True, block=2 ** 20):
    """VTK inline binary: base64(header) + base64(data). Uncompressed the header is the byte count; with zlib it is
    [number of blocks, block size, size of the last block, compressed size of each block] and the data are the compressed blocks."""
    raw = np.ascontiguousarray(a).tobytes()
    if not compress:
        return base64.b64encode(struct.pack("<I", len(raw))).decode() + base64.b64encode(raw).decode()
    blocks = [zlib.compress(raw[i:i + block], 6) for i in range(0, len(raw), block)] or [zlib.compress(b"")]
    last = len(raw) % block or (block if raw else 0)
    head = struct.pack(f"<{3 + len(blocks)}I", len(blocks), block, last, *[len(b) for b in blocks])
    return base64.b64encode(head).decode() + base64.b64encode(b"".join(blocks)).decode()


def write_vti(path, arrays, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), compress=True):
    """arrays: {name: 3-D array (z, y, x)}; spacing and origin are given as (x, y, z). Float64 is stored as float32 and int64
    labels as uint32 or int32 (check `names` printed by ParaView if IDs are larger than 2^31)."""
    shapes = {np.shape(a) for a in arrays.values()}
    assert len(shapes) == 1, "all arrays must share one shape"
    nz, ny, nx = shapes.pop()
    items = []
    for name, a in arrays.items():
        a = np.asarray(a)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        elif a.dtype == np.int64:
            a = a.astype(np.uint32) if a.min() >= 0 and a.max() < 2 ** 32 else a.astype(np.int32)
        elif a.dtype == np.bool_:
            a = a.astype(np.uint8)
        items.append(f'        <DataArray type="{_VTK[a.dtype]}" Name="{name}" format="binary">{_b64(a, compress)}</DataArray>')
    comp = ' compressor="vtkZLibDataCompressor"' if compress else ""
    extent = f"0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"
    first = next(iter(arrays))
    xml = f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt32"{comp}>
  <ImageData WholeExtent="{extent}" Origin="{origin[0]} {origin[1]} {origin[2]}" Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">
    <Piece Extent="{extent}">
      <PointData Scalars="{first}">
{chr(10).join(items)}
      </PointData>
      <CellData/>
    </Piece>
  </ImageData>
</VTKFile>
"""
    with open(path, "w") as f:
        f.write(xml)


def write_ply(path, meshes, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), colors=None):
    """meshes: {label: (vertices (n, 3) in (z, y, x) voxel units, faces (m, 3))}. Vertices are written as (x, y, z) scaled by
    `spacing` (x, y, z). `colors` maps label -> (r, g, b) in 0-1; by default hues are spread over the full spectrum."""
    V, F, C = [], [], []
    off = 0
    labels = sorted(meshes)
    for i, k in enumerate(labels):
        v, f = meshes[k]
        v = np.asarray(v)[:, ::-1] * np.asarray(spacing) + np.asarray(origin)
        rgb = colors[k] if colors and k in colors else colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.85, 0.95)
        V.append(v.astype(np.float32)); F.append(np.asarray(f) + off); off += len(v)
        C.append(np.tile((np.array(rgb) * 255).astype(np.uint8), (len(v), 1)))
    V, F, C = np.concatenate(V), np.concatenate(F).astype(np.int32), np.concatenate(C)
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\nelement face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
    vert = np.zeros(len(V), dtype=[("p", "<f4", 3), ("c", "u1", 3)]); vert["p"], vert["c"] = V, C
    face = np.zeros(len(F), dtype=[("n", "u1"), ("i", "<i4", 3)]); face["n"], face["i"] = 3, F
    with open(path, "wb") as f:
        f.write(head.encode()); f.write(vert.tobytes()); f.write(face.tobytes())
