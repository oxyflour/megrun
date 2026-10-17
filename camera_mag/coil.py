"""Distributed rectangular winding quadrature and Lorentz force in SI units."""
from dataclasses import dataclass
import numpy as np
from .interaction import MU0, volume_flux_density


@dataclass
class Winding:
    points: np.ndarray
    dl: np.ndarray
    ampere_turns: np.ndarray
    core_radius: float

    def translated(self, offset):
        return Winding(self.points + offset, self.dl, self.ampere_turns, self.core_radius)


def make_winding(camera, current_a=.1, turns=100, segments_per_side=20, cross_samples=3):
    if turns <= 0 or segments_per_side < 4 or cross_samples < 1:
        raise ValueError("Positive turns and valid winding quadrature required")
    spec = camera.metadata["coil"]
    inner, outer = np.array(spec["inner_half_m"]), np.array(spec["outer_half_m"])
    zlo, zhi = spec["z_m"]
    points, dl = [], []
    for k in range(cross_samples):
        half = inner + (outer - inner) * (k + .5) / cross_samples
        cx, cy = spec["center_m"]
        corners = np.array([[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]]) + [cx, cy]
        for j in range(cross_samples):
            z = zlo + (zhi - zlo) * (j + .5) / cross_samples
            for a, b in zip(corners, np.roll(corners, -1, axis=0)):
                midpoint = a + ((np.arange(segments_per_side) + .5) / segments_per_side)[:, None] * (b - a)
                points.extend(np.c_[midpoint, np.full(segments_per_side, z)])
                dl.extend(np.tile(np.r_[((b - a) / segments_per_side), 0], (segments_per_side, 1)))
    count = len(points)
    radius = min(np.min(outer - inner), zhi - zlo) / (2 * cross_samples)
    return Winding(np.array(points), np.array(dl), np.full(count, current_a * turns / cross_samples**2), radius)


def winding_field(points, winding):
    """H from finite-core midpoint Biot-Savart; current sources include both coils."""
    output = np.empty_like(points, dtype=float)
    for start in range(0, len(points), 128):
        r = points[start:start + 128, None, :] - winding.points[None, :, :]
        r2 = np.einsum("...i,...i->...", r, r) + winding.core_radius**2
        output[start:start + len(r)] = np.sum(np.cross(winding.dl[None], r) *
              (winding.ampere_turns / (4 * np.pi) / r2**1.5)[..., None], axis=1)
    return output


def lorentz_force(winding, b):
    return np.sum(winding.ampere_turns[:, None] * np.cross(winding.dl, b), axis=0)


def coil_forces(solver, result, isolated):
    shift = result.offset_b
    ma, mb = result.state
    ia, ib = isolated
    ca, cb = solver.camera_a, solver.camera_b
    wa, wb = solver.winding_a, solver.winding_b.translated(shift)
    out = {}
    for label, winding, own, other, own_m, other_m, own_offset, other_offset, own_iso, other_winding in [
        ("A", wa, ca, cb, ma, mb, np.zeros(3), shift, ia, wb),
        ("B", wb, cb, ca, mb, ma, shift, np.zeros(3), ib, wa),
    ]:
        b_own = volume_flux_density(winding.points, own.centers + own_offset, own.volumes, own_m)
        b_other = volume_flux_density(winding.points, other.centers + other_offset, other.volumes, other_m)
        b_current = MU0 * winding_field(winding.points, other_winding)
        b_iso = volume_flux_density(winding.points, own.centers + own_offset, own.volumes, own_iso)
        total = lorentz_force(winding, b_own + b_other + b_current)
        baseline = lorentz_force(winding, b_iso)
        out[label] = {"total_N": total.tolist(), "isolated_N": baseline.tolist(),
                      "interaction_delta_N": (total - baseline).tolist(),
                      "direct_other_module_N": lorentz_force(winding, b_other + b_current).tolist()}
    return out
