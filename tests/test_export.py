"""Export of fields (VTK ImageData) and meshes (PLY), and gapped conforming meshes."""
import base64
import struct
import xml.etree.ElementTree as ET
import numpy as np
import pytest

from zmorph.export import write_vti, write_ply
from zmorph.mesh import competition_meshes


def _decode(txt, dt, compressed):
    import zlib
    if not compressed:
        n = struct.unpack("<I", base64.b64decode(txt[:8]))[0]
        return np.frombuffer(base64.b64decode(txt[8:]), dtype=dt)[: n // np.dtype(dt).itemsize]
    nb = struct.unpack("<I", base64.b64decode(txt[:8])[:4])[0]
    hlen = 4 * (3 + nb); hchars = (hlen + 2) // 3 * 4                       # base64 length of the header
    head = struct.unpack(f"<{3 + nb}I", base64.b64decode(txt[:hchars])); data = base64.b64decode(txt[hchars:])
    out, pos = [], 0
    for size in head[3:]:
        out.append(zlib.decompress(data[pos:pos + size])); pos += size
    return np.frombuffer(b"".join(out), dtype=dt)


def _read_vti(path):
    root = ET.parse(path).getroot(); piece = root.find("ImageData/Piece"); compressed = root.get("compressor") is not None
    ext = [int(x) for x in piece.get("Extent").split()]; shape = (ext[5] + 1, ext[3] + 1, ext[1] + 1)
    out = {}
    for da in piece.iter("DataArray"):
        dt = {"Float32": "<f4", "UInt8": "u1", "UInt32": "<u4", "Int32": "<i4"}[da.get("type")]
        out[da.get("Name")] = _decode(da.text.strip(), dt, compressed).reshape(shape)
    return root, out


@pytest.mark.parametrize("compress", [False, True])
def test_vti_round_trip(tmp_path, compress):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(4, 5, 6)).astype(np.float32); lab = rng.integers(0, 9, (4, 5, 6)); mask = a > 0
    p = tmp_path / "f.vti"; write_vti(p, {"sdf": a, "label": lab, "mask": mask}, spacing=(8, 8, 8), origin=(1, 2, 3), compress=compress)
    root, out = _read_vti(p)
    assert np.array_equal(out["sdf"], a) and np.array_equal(out["label"], lab) and np.array_equal(out["mask"], mask)
    assert root.find("ImageData").get("Spacing") == "8 8 8"


def test_ply_header_and_size(tmp_path):
    v = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], float); f = np.array([[0, 1, 2], [0, 1, 3]])
    p = tmp_path / "m.ply"; write_ply(p, {5: (v, f), 9: (v + 3, f)}, spacing=(2, 2, 2))
    data = p.read_bytes(); head, body = data.split(b"end_header\n", 1)
    assert b"element vertex 8" in head and b"element face 4" in head
    assert len(body) == 8 * 15 + 4 * 13
    assert struct.unpack("<3f", body[:12]) == (0.0, 0.0, 0.0) and struct.unpack("<3f", body[15 * 1:15 * 1 + 12]) == (2.0, 0.0, 0.0)


def test_gapped_meshes_are_separated_by_the_gap():
    a = np.ones((40, 40), np.int64); a[:, 20:] = 2
    secs = [a] * 3
    for gap in (0.0, 4.0):
        lab, meshes = competition_meshes(secs, 5, gap=gap, sigma=0.0)
        x1 = meshes[1][0][:, 2].max(); x2 = meshes[2][0][:, 2].min()        # vertex x is the last axis
        assert x2 - x1 == pytest.approx(gap, abs=0.6)
