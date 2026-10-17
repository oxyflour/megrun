"""Single-valued isotropic B-H response, with local demag preconditioning."""
from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import PchipInterpolator
from .interaction import MU0


@dataclass
class LinearMaterial:
    mu_r: float = 1.0

    def __post_init__(self):
        if self.mu_r <= 0:
            raise ValueError("mu_r must be positive")

    def response(self, h):
        return (self.mu_r - 1) * np.asarray(h)

    def local_response(self, external_h, remanence):
        chi = self.mu_r - 1
        return (remanence + chi * external_h) / (1 + chi / 3)


@dataclass
class SoftMagneticMaterial:
    h: np.ndarray
    b: np.ndarray
    saturation_t: float = 1.6
    interpolation: str = "pchip"
    _curve: object = field(init=False, repr=False)
    _inverse: object = field(init=False, repr=False)

    def __post_init__(self):
        self.h, self.b = np.asarray(self.h, float), np.asarray(self.b, float)
        if self.h.shape != self.b.shape or len(self.h) < 3 or self.h[0] != 0 or self.b[0] != 0:
            raise ValueError("B-H table must start at (0,0), with >=3 samples")
        if np.any(np.diff(self.h) <= 0) or np.any(np.diff(self.b) <= 0):
            raise ValueError("B-H table must be strictly increasing")
        if self.interpolation not in ("pchip", "linear"):
            raise ValueError("Unsupported B-H interpolation")
        self._curve = PchipInterpolator(self.h, self.b, extrapolate=False)
        # Dense monotonic inverse of H_ext = H + M(H)/3 removes stiff self feedback.
        grid = np.unique(np.r_[0, np.geomspace(max(self.h[1] * 1e-4, 1e-6), self.h[-1], 12000), self.h])
        ext = grid + (self.b_value(grid) / MU0 - grid) / 3
        if np.any(np.diff(ext) <= 0):
            raise ValueError("B-H curve gives non-monotone local demagnetization response")
        self._inverse = PchipInterpolator(ext, grid, extrapolate=False)
        self._ext_max = ext[-1]

    def b_value(self, hmag):
        hmag = np.asarray(hmag)
        if self.interpolation == "linear":
            return np.where(hmag <= self.h[-1], np.interp(hmag, self.h, self.b),
                            self.b[-1] + self.high_field_slope * (hmag - self.h[-1]))
        return np.where(hmag <= self.h[-1], self._curve(np.minimum(hmag, self.h[-1])),
                        self.b[-1] + MU0 * (hmag - self.h[-1]))

    @property
    def high_field_slope(self):
        return (self.b[-1] - self.b[-2]) / (self.h[-1] - self.h[-2]) if self.interpolation == "linear" else MU0

    def response(self, h):
        h = np.asarray(h)
        mag = np.linalg.norm(h, axis=-1)
        m = self.b_value(mag) / MU0 - mag
        return h * np.divide(m, mag, out=np.zeros_like(mag), where=mag > 0)[..., None]

    def local_response(self, external_h, remanence=None):
        mag = np.linalg.norm(external_h, axis=-1)
        hmag = self._inverse(np.minimum(mag, self._ext_max))
        beyond = self.h[-1] + (mag - self._ext_max) / (1 + (self.high_field_slope / MU0 - 1) / 3)
        hmag = np.where(mag > self._ext_max, beyond, hmag)
        mmag = self.b_value(hmag) / MU0 - hmag
        return external_h * np.divide(mmag, mag, out=np.zeros_like(mag), where=mag > 0)[..., None]

    def tangent_mu(self, hmag):
        if self.interpolation == "linear":
            indices = np.clip(np.searchsorted(self.h, hmag, side="right") - 1, 0, len(self.h) - 2)
            return (np.diff(self.b) / np.diff(self.h))[indices] / MU0
        return np.where(hmag <= self.h[-1], self._curve.derivative()(np.minimum(hmag, self.h[-1])) / MU0, 1.0)


def benchmark_materials(br_t=1.2, recoil_mu_r=1.05, bh_file=None):
    # Illustrative steel, not a measured material certificate.
    table = np.array([[0, 0], [50, .25], [100, .50], [200, .85], [400, 1.15],
                      [800, 1.35], [1600, 1.48], [3200, 1.56], [8000, 1.62], [20000, 1.68]])
    if bh_file:
        table = np.loadtxt(bh_file, delimiter=",", skiprows=1)
    return {
        "yoke": SoftMagneticMaterial(table[:, 0], table[:, 1]),
        "magnet": LinearMaterial(recoil_mu_r),
        "copper": LinearMaterial(1.0),
    }
