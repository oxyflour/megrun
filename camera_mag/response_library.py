"""Single-camera field reuse: SI volume quadrature and exported-field checks."""
from pathlib import Path
import numpy as np


def coil_quadrature(spec, side_segments=32, radial_order=4, height_order=16):
    """Integrate a rectangular annulus using nested rectangular contours.

    Returns points [m], positive volume weights [m³], and ideal CCW J/NI
    [m⁻²]. Actual predictions use Maxwell's exported J, not this ideal J.
    Gauss points stay strictly away from conductor faces and corner seams.
    """
    inner, outer = np.asarray(spec["inner_half_m"]), np.asarray(spec["outer_half_m"])
    zlo, zhi = spec["z_m"]
    if side_segments < 1 or radial_order < 1 or height_order < 1:
        raise ValueError("Positive quadrature orders required")
    if np.any(outer <= inner) or zhi <= zlo:
        raise ValueError("Positive annular thickness required")
    g2, w2 = np.polynomial.legendre.leggauss(2)
    ts = ((np.arange(side_segments)[:, None] + (g2 + 1) / 2) / side_segments).ravel()
    tw = np.tile(w2 / (2 * side_segments), side_segments)
    gr, wr = np.polynomial.legendre.leggauss(radial_order)
    gz, wz = np.polynomial.legendre.leggauss(height_order)
    rs, rw = (gr + 1) / 2, wr / 2
    zs, zw = zlo + (gz + 1) * (zhi - zlo) / 2, wz * (zhi - zlo) / 2
    r, t, z = np.meshgrid(rs, ts, zs, indexing="ij")
    base = (rw[:, None, None] * tw[None, :, None] * zw[None, None, :]).ravel()
    hx = (inner[0] + r * (outer[0] - inner[0])).ravel()
    hy = (inner[1] + r * (outer[1] - inner[1])).ravel()
    t, z = t.ravel(), z.ravel()
    cx, cy = spec["center_m"]
    xyz, volume, ideal = [], [], []
    for side in range(4):
        if side in (0, 2):
            sign = 1 if side == 0 else -1
            pts = np.c_[cx + sign * (2*t-1)*hx, cy - sign*hy, z]
            jac = 2 * hx * (outer[1] - inner[1])
            j = np.tile([sign / ((outer[1] - inner[1]) * (zhi-zlo)), 0, 0], (len(t), 1))
        else:
            sign = 1 if side == 1 else -1
            pts = np.c_[cx + sign*hx, cy + sign*(2*t-1)*hy, z]
            jac = 2 * hy * (outer[0] - inner[0])
            j = np.tile([0, sign / ((outer[0] - inner[0]) * (zhi-zlo)), 0], (len(t), 1))
        xyz.append(pts)
        volume.append(base * jac)
        ideal.append(j)
    points, weights, ideal_j = np.concatenate(xyz), np.concatenate(volume), np.concatenate(ideal)
    exact = 4 * (np.prod(outer) - np.prod(inner)) * (zhi - zlo)
    if not np.isclose(weights.sum(), exact, rtol=1e-12):
        raise ValueError("Annular quadrature volume does not match geometry")
    return points, weights, ideal_j


def read_vector_field(path, expected_points=None):
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            row = [float(item) for item in line.split()]
        except ValueError:
            continue
        if len(row) == 6:
            rows.append(row)
    array = np.asarray(rows)
    if array.ndim != 2 or array.shape[1] != 6 or not np.isfinite(array).all():
        raise ValueError(f"Invalid vector field: {path}")
    if expected_points is not None:
        expected = np.asarray(expected_points)
        if array.shape[0] != len(expected) or not np.allclose(array[:, :3], expected, atol=1e-10, rtol=0):
            raise ValueError(f"Exported field positions differ from requested SI positions: {path}")
    return array[:, 3:]


def integrate_force(current_density, flux_density, weights):
    j, b, w = np.asarray(current_density), np.asarray(flux_density), np.asarray(weights)
    if j.shape != b.shape or j.shape != (len(w), 3) or np.any(w <= 0):
        raise ValueError("Invalid Lorentz quadrature arrays")
    if not (np.isfinite(j).all() and np.isfinite(b).all() and np.isfinite(w).all()):
        raise ValueError("Non-finite Lorentz quadrature")
    return np.sum(np.cross(j, b) * w[:, None], axis=0)


def force_response_operator(points, current_density, weights, centers, volumes, block=128):
    """Map incremental volume magnetization to coil Lorentz force.

    Only the induced *change* uses the Gaussian coarse volume approximation;
    the unperturbed field and direct incident field remain native Maxwell data.
    """
    from .interaction import volume_operator
    result = np.zeros((3, 3*len(centers)))
    for start in range(0, len(points), block):
        sl = slice(start, start+block)
        kernel = volume_operator(points[sl], centers, volumes, flux_density=True)
        kernel = kernel.reshape(-1, 3, 3*len(centers))
        j = current_density[sl]
        cross = np.zeros((len(j), 3, 3))
        cross[:, 0, 1], cross[:, 0, 2] = -j[:, 2], j[:, 1]
        cross[:, 1, 0], cross[:, 1, 2] = j[:, 2], -j[:, 0]
        cross[:, 2, 0], cross[:, 2, 1] = -j[:, 1], j[:, 0]
        result += np.einsum("p,pij,pjk->ik", weights[sl], cross, kernel, optimize=True)
    return result


def native_susceptibility(h0, material_ids, bh_table, recoil_mu_r=1.05):
    """Derivative of M(H) at the exported isolated operating point.

    Isotropic B(|H|) has different longitudinal and transverse derivatives;
    using only dB/dH as a scalar would lose the field-direction response.
    """
    from .interaction import MU0
    h0, ids, table = np.asarray(h0), np.asarray(material_ids), np.asarray(bh_table)
    tangent = np.zeros((len(h0), 3, 3))
    for index, h in enumerate(h0):
        if ids[index] == "magnet":
            tangent[index] = (recoil_mu_r-1)*np.eye(3)
        elif ids[index] == "yoke":
            hm = np.linalg.norm(h)
            k = np.clip(np.searchsorted(table[:, 0], hm)-1, 0, len(table)-2)
            slope = np.diff(table[k:k+2, 1])[0]/np.diff(table[k:k+2, 0])[0]
            bn = np.interp(hm, table[:, 0], table[:, 1])
            if hm>table[-1, 0]:
                bn += slope*(hm-table[-1, 0])
            longitudinal = slope/MU0-1
            transverse = bn/(MU0*hm)-1 if hm>1e-10 else longitudinal
            unit = h/hm if hm>1e-10 else np.zeros(3)
            tangent[index] = transverse*np.eye(3)+(longitudinal-transverse)*np.outer(unit, unit)
        else:
            raise ValueError(f"Unsupported magnetic material: {ids[index]}")
    if np.linalg.eigvalsh(tangent).min() < -1e-8:
        raise ValueError("Negative native susceptibility tangent")
    return tangent
