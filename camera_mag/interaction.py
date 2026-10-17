"""Point-volume dipoles with spherical (N=1/3) self-cell demagnetization.

This intentionally approximates finite voxel interactions. No singular point
kernel is evaluated on the diagonal. Arrays map magnetization [A/m] to H [A/m].
"""
import numpy as np
from scipy.special import erf

MU0 = 4e-7 * np.pi
KERNEL_VERSION = "gaussian-volume-average-spherical-self-v2"


def dipole_operator(targets, sources, volumes, *, self_interaction=False, dtype="float64"):
    targets, sources = np.asarray(targets), np.asarray(sources)
    n, m = len(targets), len(sources)
    result = np.empty((n, 3, m, 3), dtype=dtype)
    # Limit temporary memory; block loops are not per-cell Python loops.
    for start in range(0, n, 128):
        r = targets[start:start + 128, None, :] - sources[None, :, :]
        r2 = np.einsum("...i,...i->...", r, r)
        zero = r2 < 1e-30
        if zero.any() and not self_interaction:
            raise ValueError("Coincident source and observation cells")
        safe = np.where(zero, 1.0, r2)
        tensor = (3 * r[..., :, None] * r[..., None, :] / safe[..., None, None]
                  - np.eye(3)) * (np.asarray(volumes)[None, :] /
                  (4 * np.pi * safe ** 1.5))[..., None, None]
        tensor[zero] = 0
        result[start:start + len(r)] = tensor.transpose(0, 2, 1, 3)
    if self_interaction:
        if n != m or not np.array_equal(targets, sources):
            raise ValueError("Self operator requires identical source/target centers")
        result[np.arange(n), :, np.arange(n), :] = -np.eye(3) / 3
    return result.reshape(3 * n, 3 * m)


def dipole_field(points, centers, volumes, magnetization):
    return (dipole_operator(points, centers, volumes) @ np.asarray(magnetization).ravel()).reshape(-1, 3)


def volume_operator(targets, sources, volumes, target_volumes=None, *, dtype="float64", flux_density=False):
    """Gaussian volume-averaged dipoles, including a consistent N=1/3 self term.

    Each volume V has Gaussian width sigma = V**(1/3)/sqrt(4*pi).
    Pair averaging convolves source and target Gaussians. The Hessian of
    erf(r/(sqrt(2)*sigma))/(4*pi*r) tends to the point dipole tensor far away.
    Unlike bare point interactions, its volume-weighted magnetic energy is
    negative semidefinite, avoiding artificial high-permeability polarization.
    This is a coarse finite-volume approximation, not exact CAD demagnetization.
    flux_density uses point observations and includes the source M density.
    """
    targets, sources, volumes = np.asarray(targets), np.asarray(sources), np.asarray(volumes)
    if np.any(volumes <= 0):
        raise ValueError("Cell volumes must be positive")
    if flux_density and target_volumes is not None:
        raise ValueError("Flux-density evaluation expects point observations")
    source_sigma = np.cbrt(volumes) / np.sqrt(4 * np.pi)
    target_sigma = np.zeros(len(targets)) if target_volumes is None else np.cbrt(target_volumes) / np.sqrt(4 * np.pi)
    result = np.empty((len(targets), 3, len(sources), 3), dtype=dtype)
    for start in range(0, len(targets), 128):
        r = targets[start:start + 128, None, :] - sources[None, :, :]
        r2 = np.einsum("...i,...i->...", r, r)
        length = np.sqrt(r2)
        sigma = np.sqrt(target_sigma[start:start + len(r), None]**2 + source_sigma[None, :]**2)
        u = length / (np.sqrt(2) * sigma)
        exponential = np.exp(-u*u)
        f = erf(u) - 2 / np.sqrt(np.pi) * u * exponential
        # Series avoids cancellation in the isotropic r -> 0 limit.
        small = u < .05
        f[small] = 2 / np.sqrt(np.pi) * u[small]**3 * (2/3 - 2/5*u[small]**2 + u[small]**4/7 - u[small]**6/27)
        a = np.divide(f, length**3, out=np.zeros_like(f), where=length > 0)
        a[length == 0] = np.sqrt(2 / np.pi) / (3 * sigma[length == 0]**3)
        c = np.sqrt(2 / np.pi) * exponential / sigma**3
        unit = np.divide(r, length[..., None], out=np.zeros_like(r), where=length[..., None] > 0)
        tensor = ((3*a-c)[..., None, None] * unit[..., :, None] * unit[..., None, :]
                  - a[..., None, None] * np.eye(3)) * volumes[None, :, None, None] / (4*np.pi)
        if flux_density:
            density = volumes[None, :] * exponential / ((2*np.pi)**1.5 * sigma**3)
            tensor = MU0 * (tensor + density[..., None, None] * np.eye(3))
        result[start:start + len(r)] = tensor.transpose(0, 2, 1, 3)
    return result.reshape(3 * len(targets), 3 * len(sources))


def volume_flux_density(points, centers, volumes, magnetization):
    return (volume_operator(points, centers, volumes, flux_density=True) @ np.asarray(magnetization).ravel()).reshape(-1, 3)


def dipole_pair_force(r, moment_a, moment_b):
    """Force on dipole a; r points from b to a, moments in A m²."""
    r = np.asarray(r, dtype=float)
    length = np.linalg.norm(r)
    if length <= 0:
        raise ValueError("Dipoles must be separated")
    u = r / length
    a, b = np.asarray(moment_a), np.asarray(moment_b)
    au, bu = a @ u, b @ u
    return 3 * MU0 / (4 * np.pi * length**4) * (au * b + bu * a + (a @ b - 5 * au * bu) * u)
