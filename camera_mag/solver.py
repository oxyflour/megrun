"""Reusable camera self operators and distance-dependent nonlinear coupling."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from .cache import self_operator
from .coil import make_winding, winding_field
from .interaction import MU0, volume_operator


@dataclass
class SolveResult:
    distance_mm: float
    offset_b: np.ndarray
    state: tuple
    h: tuple
    b: tuple
    iterations: int
    residual_history: list
    converged: bool
    runtime: dict
    constitutive_residual: tuple


class CameraPairSolver:
    def __init__(self, camera_a, camera_b, materials, *, cache_dir=".cache", dtype="float64",
                 current_a=.1, turns=100, omega=.7, tol=1e-7, max_iterations=1500,
                 method="anderson", segments_per_side=20, cross_samples=3, blas_threads=1):
        if dtype not in ("float32", "float64") or method not in ("anderson", "fixed"):
            raise ValueError("Unsupported dtype or iteration method")
        if not 0 < omega <= 1 or tol <= 0 or max_iterations < 1:
            raise ValueError("Invalid nonlinear settings")
        self.camera_a, self.camera_b, self.materials = camera_a, camera_b, materials
        unknown = (set(camera_a.material_ids) | set(camera_b.material_ids)) - set(materials)
        if unknown:
            raise ValueError(f"Missing material models: {unknown}")
        self.cache_dir, self.dtype = cache_dir, dtype
        self.omega, self.tol, self.max_iterations, self.method = omega, tol, max_iterations, method
        self.blas_threads = blas_threads
        self.winding_a = make_winding(camera_a, current_a, turns, segments_per_side, cross_samples)
        self.winding_b = make_winding(camera_b, current_a, turns, segments_per_side, cross_samples)
        self.current_a, self.turns = current_a, turns
        self.self_a = self.self_b = None
        self.self_cache = {}
        self.self_build_calls = 0
        self._isolated = None
        self.isolated_runtime = []

    def precompute_self_operators(self):
        if self.self_a is not None:
            return self.self_cache
        self.self_a, self.self_cache["A"] = self_operator(self.camera_a, self.cache_dir, self.dtype)
        if np.array_equal(self.camera_a.centers, self.camera_b.centers) and np.array_equal(self.camera_a.volumes, self.camera_b.volumes):
            self.self_b = self.self_a
            self.self_cache["B"] = {**self.self_cache["A"], "shared_with_A": True, "seconds": 0.0}
        else:
            self.self_b, self.self_cache["B"] = self_operator(self.camera_b, self.cache_dir, self.dtype)
        self.self_build_calls += 1
        self.src_a = winding_field(self.camera_a.centers, self.winding_a)
        self.src_b = winding_field(self.camera_b.centers, self.winding_b)
        return self.self_cache

    def _response(self, external_h, camera):
        response = np.empty_like(external_h)
        for name, material in self.materials.items():
            mask = camera.material_ids == name
            response[mask] = material.local_response(external_h[mask], camera.remanence[mask])
        return response

    def material_residual(self, m, h, camera):
        residual = np.empty_like(m)
        for name, material in self.materials.items():
            mask = camera.material_ids == name
            residual[mask] = m[mask] - camera.remanence[mask] - material.response(h[mask])
        return residual

    def _iterate(self, mapping, initial):
        start = perf_counter()
        x = np.asarray(initial, dtype=self.dtype).ravel().copy()
        history, xs, rs = [], [], []
        converged = False
        with threadpool_limits(limits=self.blas_threads, user_api="blas"):
            for iteration in range(self.max_iterations + 1):
                target = mapping(x).ravel()
                r = target - x
                relative = float(np.linalg.norm(r) / max(np.linalg.norm(target), 1.0))
                history.append(relative)
                if not np.isfinite(relative):
                    raise RuntimeError("Non-finite nonlinear iteration")
                if relative < self.tol:
                    converged = True
                    break
                if iteration == self.max_iterations:
                    break
                step = self.omega * r
                if self.method == "anderson" and xs:
                    dx = np.stack([b - a for a, b in zip(xs, xs[1:] + [x])], axis=1)
                    dr = np.stack([b - a for a, b in zip(rs, rs[1:] + [r])], axis=1)
                    # Small regularized history solve, no dense Jacobian/Newton solve.
                    gram = dr.T @ dr
                    regularization = max(float(np.trace(gram)) * 1e-10, 1e-15)
                    gamma = np.linalg.solve(gram + regularization * np.eye(len(gram)), dr.T @ r)
                    candidate = step - (dx + self.omega * dr) @ gamma
                    if np.linalg.norm(candidate) <= 5 * max(np.linalg.norm(step), 1.0):
                        step = candidate
                xs.append(x.copy())
                rs.append(r.copy())
                if len(xs) > 8:
                    xs.pop(0)
                    rs.pop(0)
                x = np.asarray(x + step, dtype=self.dtype)
        return x, iteration, history, converged, perf_counter() - start

    def isolated_states(self):
        self.precompute_self_operators()
        if self._isolated is not None:
            return self._isolated
        states = []
        for camera, operator, src in [(self.camera_a, self.self_a, self.src_a),
                                      (self.camera_b, self.self_b, self.src_b)]:
            if (states and self.self_b is self.self_a and
                    np.array_equal(self.camera_a.remanence, self.camera_b.remanence) and
                    np.array_equal(self.camera_a.material_ids, self.camera_b.material_ids) and
                    np.array_equal(self.src_a, self.src_b)):
                states.append(states[0].copy())
                self.isolated_runtime.append({**self.isolated_runtime[0], "shared_with_A": True})
                continue
            def mapping(flat):
                m = flat.reshape(-1, 3)
                external = (operator @ flat).reshape(-1, 3) + m / 3 + src
                return self._response(external, camera)
            x, count, history, ok, seconds = self._iterate(mapping, camera.remanence)
            if not ok:
                raise RuntimeError(f"Isolated module failed to converge: residual={history[-1]:.3g}")
            states.append(x.reshape(-1, 3))
            self.isolated_runtime.append({"iterations": count, "solve_seconds": seconds, "residual": history[-1]})
        self._isolated = tuple(states)
        return self._isolated

    def solve(self, distance, initial_state=None):
        """distance is the nonnegative outer-envelope gap along +X, in mm."""
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Outer-envelope gap must be finite and nonnegative")
        self.precompute_self_operators()
        total_start = perf_counter()
        ca, cb = self.camera_a, self.camera_b
        shift = np.array([ca.metadata["bounds_m"][1][0] - cb.metadata["bounds_m"][0][0] + distance * 1e-3, 0, 0])
        start = perf_counter()
        cross_ab = volume_operator(ca.centers, cb.centers + shift, cb.volumes,
                                   target_volumes=ca.volumes, dtype=self.dtype)
        # Dipole kernel reciprocity, accounting for unequal cell volumes.
        blocks = cross_ab.reshape(len(ca.centers), 3, len(cb.centers), 3).transpose(2, 3, 0, 1)
        cross_ba = np.ascontiguousarray(blocks * (ca.volumes[None, None, :, None] /
                                                   cb.volumes[:, None, None, None]), dtype=self.dtype).reshape(3 * len(cb.centers), -1)
        source_a = self.src_a + winding_field(ca.centers, self.winding_b.translated(shift))
        source_b = self.src_b + winding_field(cb.centers + shift, self.winding_a)
        cross_seconds = perf_counter() - start
        split = ca.centers.size

        def fields(flat):
            ma, mb = flat[:split], flat[split:]
            ha = (self.self_a @ ma + cross_ab @ mb).reshape(-1, 3) + source_a
            hb = (self.self_b @ mb + cross_ba @ ma).reshape(-1, 3) + source_b
            return ha, hb

        def mapping(flat):
            ha, hb = fields(flat)
            ma, mb = flat[:split].reshape(-1, 3), flat[split:].reshape(-1, 3)
            return np.r_[self._response(ha + ma / 3, ca).ravel(), self._response(hb + mb / 3, cb).ravel()]

        initial = (ca.remanence, cb.remanence) if initial_state is None else initial_state
        if len(initial) != 2 or initial[0].shape != ca.remanence.shape or initial[1].shape != cb.remanence.shape:
            raise ValueError("Initial state does not match module cells")
        flat, count, history, ok, seconds = self._iterate(mapping, np.r_[initial[0].ravel(), initial[1].ravel()])
        with threadpool_limits(limits=self.blas_threads, user_api="blas"):
            h = fields(flat)
        state = flat[:split].reshape(-1, 3), flat[split:].reshape(-1, 3)
        b = tuple(MU0 * (hi + mi) for hi, mi in zip(h, state))
        residual = tuple(self.material_residual(mi, hi, camera) for mi, hi, camera in zip(state, h, (ca, cb)))
        return SolveResult(float(distance), shift, state, h, b, count, history, ok,
                           {"cross_build_seconds": cross_seconds, "solve_seconds": seconds,
                            "total_seconds": perf_counter() - total_start, "warm_start": initial_state is not None}, residual)
