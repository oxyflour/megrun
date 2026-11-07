"""Read AEDT text .aedtplt VOLUME mesh plots, preserving TET4/TET10 corner nodes.

AEDT's 10-node tetrahedron convention uses corners at node positions 0, 2, 5, 9.
Only element type 4 (tetrahedron) is accepted. Coordinates are rescaled to SI.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


class MeshError(ValueError):
    pass


_FLOAT = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?")
_DRAW = re.compile(r"\$begin\s+Drawing[^\r\n]*([\s\S]*?)\$end\s+Drawing[^\r\n]*", re.I)

UNITS_M = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 1e-3,
    "millimeter": 1e-3,
    "cm": 1e-2,
    "um": 1e-6,
    "micrometer": 1e-6,
    "nm": 1e-9,
    "in": 0.0254,
    "inch": 0.0254,
    "mil": 2.54e-5,
    "ft": 0.3048,
}


def get_unit_scale(unit: str) -> float:
    try:
        return UNITS_M[str(unit).strip().lower()]
    except KeyError as e:
        raise MeshError(f"Unsupported model unit {unit!r}; add it to UNITS_M") from e


def _payload(txt: str, key: str) -> str:
    match = re.search(r"\b" + re.escape(key) + r"\s*\(([^()]*)\)", txt, re.I | re.S)
    if not match:
        raise MeshError(f"Missing {key}(...) block in .aedtplt drawing")
    return match.group(1)


def _numbers(s: str, as_int: bool = False):
    tokens = _FLOAT.findall(s)
    if not tokens:
        raise MeshError("Empty numeric block")
    if as_int:
        out = []
        for t in tokens:
            f = float(t.replace('D', 'E').replace('d', 'e'))
            if int(f) != f:
                raise MeshError(f"Expected integer, found {t}")
            out.append(int(f))
        return out
    return [float(x.replace('D', 'E').replace('d', 'e')) for x in tokens]


def tetra_volume(vertices_m: np.ndarray) -> np.ndarray:
    """Signed TET4 volume, vertices shape (N,4,3) in meters."""
    v = np.asarray(vertices_m, float)
    return np.einsum('ij,ij->i', np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]),
                     v[:, 3] - v[:, 0]) / 6.0


def read_aedtplt(path: str | Path, scale_to_m: float) -> tuple[np.ndarray, dict]:
    """Return (tetra[N,4,3] in meters, diagnostic dict).

    No silent surface->volume conversion. Curved second-order tet geometry is
    linearized by dropping mid-side nodes; diagnostics record this.
    """
    text = Path(path).read_text(encoding='utf-8-sig', errors='replace')
    drawings = _DRAW.findall(text)
    if not drawings:
        raise MeshError(f"No $begin Drawing ... $end Drawing sections found: {path}")
    result, counts = [], {'drawings': len(drawings), 'tet4': 0, 'tet10': 0,
                           'surface_elements_skipped': 0, 'curved': False}
    for drawing in drawings:
        ns = _numbers(_payload(drawing, 'Nodes'))
        es = _numbers(_payload(drawing, 'Elements'), as_int=True)
        if len(ns) % 3 != 0:
            raise MeshError('Node block must have 3 values per vertex')
        nodes = np.array(ns, dtype=np.float64).reshape(-1, 3) * scale_to_m
        if len(es) < 7:
            raise MeshError('Elements header too short')
        # AEDT: Elements(n_nodes, n_elements, element_type, unused x3,
        #                n_element_nodes, id_1, ..., id_n, ...)
        declared_node_count, n_elem = es[0:2]
        if declared_node_count != len(nodes):
            raise MeshError(f'Nodes count mismatch: {declared_node_count} vs {len(nodes)}')
        p = 2
        for _ in range(n_elem):
            if p + 5 > len(es):
                raise MeshError('Unexpected end of Elements array')
            element_type, _a, _b, _c, nn = es[p:p + 5]
            p += 5
            if nn < 1 or p + nn > len(es):
                raise MeshError(f'Invalid element connectivity length {nn}')
            conn = np.asarray(es[p:p + nn], dtype=np.int64) - 1  # 1-based AEDT IDs
            p += nn
            if element_type != 4:
                # Surface elements may share drawing; they are not 3D tiles.
                if element_type in (2, 3):
                    counts['surface_elements_skipped'] += 1
                    continue
                raise MeshError(f'Unsupported volume element type {element_type} (not tetra)')
            if nn == 10:
                ids = conn[[0, 2, 5, 9]]
                counts['tet10'] += 1
            elif nn == 4:
                ids = conn
                counts['tet4'] += 1
            else:
                raise MeshError(f'Tetra element has {nn} nodes, only TET4/TET10 supported')
            if np.min(ids) < 0 or np.max(ids) >= len(nodes):
                raise MeshError('Connectivity ID out of range; expected AEDT 1-based IDs')
            result.append(nodes[ids])
        if p != len(es):
            raise MeshError(f'Elements block has {len(es)-p} trailing integer fields')
        curve_match = re.search(r'\bHasCurvElem\s*\(\s*([^)]*)\)', drawing, re.I)
        if curve_match and curve_match.group(1).strip().lower() not in ('0', 'false', 'no'):
            counts['curved'] = True
    if not result:
        raise MeshError('No tetrahedral *volume* cells found. Export volume, not surfaces.')
    vertices = np.asarray(result, dtype=np.float64)
    vol = tetra_volume(vertices)
    scale = max(np.linalg.norm(vertices.max(axis=(0, 1)) - vertices.min(axis=(0, 1))), 1e-12)
    if np.any(np.abs(vol) < max((scale**3)*1e-18, 1e-32)):
        raise MeshError('Zero/near-zero tetrahedra found; cannot generate MagTense Tiles')
    # Ensure consistent orientation. For tetrahedra, only the 4 corner coordinates matter.
    neg = vol < 0
    if np.any(neg):
        tmp = vertices[neg, 1, :].copy()
        vertices[neg, 1, :] = vertices[neg, 2, :]
        vertices[neg, 2, :] = tmp
    counts['tetra_count'] = len(vertices)
    counts['orientation_flipped'] = int(np.sum(neg))
    counts['volume_m3'] = float(np.sum(np.abs(vol)))
    counts['bounds_min_m'] = vertices.min(axis=(0, 1)).tolist()
    counts['bounds_max_m'] = vertices.max(axis=(0, 1)).tolist()
    return vertices, counts
