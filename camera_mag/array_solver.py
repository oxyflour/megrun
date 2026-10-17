"""Fully coupled N-camera coarse model; source attribution includes cancellation."""
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .array_config import monitor_values, validate_layout
from .cache import self_operator
from .coil import lorentz_force, make_winding, winding_field
from .interaction import MU0, volume_flux_density, volume_operator
from .solver import CameraPairSolver


class CameraArraySolver(CameraPairSolver):
    """Reuse the existing material law/Anderson iteration and single-camera cache.

    Every camera feels every other camera in the same nonlinear solve. Field
    attribution splits the final coupled state by source body; it is not a
    counterfactual solve with a magnet physically removed.
    """

    def __init__(self, camera, materials, *, cache_dir=".cache", tol=1e-7,
                 max_iterations=1500, max_operator_mb=512):
        super().__init__(camera, camera, materials, cache_dir=cache_dir, tol=tol,
                         max_iterations=max_iterations, blas_threads=1)
        self.camera = camera
        self.max_operator_mb = max_operator_mb
        self.operator, self.cache_info = None, None
        self.deadline = None

    def check_budget(self):
        if self.deadline is not None and perf_counter() >= self.deadline:
            raise TimeoutError("Low-fidelity campaign budget exhausted; no automatic extension")

    def solve_layout(self, config, initial_state=None):
        start = perf_counter()
        self.check_budget()
        validate_layout(config, self.camera)
        camera, count = self.camera, config["N"]
        size = camera.centers.size
        estimated_mb = (1 + count * (count - 1)) * size**2 * 8 / 2**20
        if estimated_mb > self.max_operator_mb:
            raise MemoryError(f"Coarse operators need {estimated_mb:.1f} MiB; limit {self.max_operator_mb:g}")
        if self.operator is None:
            self.operator, self.cache_info = self_operator(camera, self.cache_dir)
        positions = np.array([c["position_mm"] for c in config["cameras"]]) * 1e-3
        centers = [camera.centers + p for p in positions]
        windings = [make_winding(camera, c["current_A"], c["turns"]).translated(p)
                    for c, p in zip(config["cameras"], positions)]
        cross = {}
        for i in range(count):
            for j in range(i):
                self.check_budget()
                op = volume_operator(centers[i], centers[j], camera.volumes, target_volumes=camera.volumes)
                cross[i, j] = op
                blocks = op.reshape(-1, 3, len(camera.centers), 3).transpose(2, 3, 0, 1)
                cross[j, i] = np.ascontiguousarray(blocks * (camera.volumes[None, None, :, None] /
                                                         camera.volumes[:, None, None, None])).reshape(size, size)
        sources = [sum((winding_field(p, w) for w in windings), np.zeros_like(p)) for p in centers]
        build_seconds = perf_counter() - start

        def fields(flat):
            state = flat.reshape(count, size)
            return [(self.operator @ state[i] + sum((cross[i, j] @ state[j] for j in range(count) if j != i),
                                                   np.zeros(size))).reshape(-1, 3) + sources[i] for i in range(count)]

        def mapping(flat):
            self.check_budget()
            state = flat.reshape(count, -1, 3)
            return np.array([self._response(h + m / 3, camera) for h, m in zip(fields(flat), state)])

        initial = np.tile(camera.remanence, (count, 1, 1)) if initial_state is None else np.asarray(initial_state)
        if initial.shape != (count, len(camera.centers), 3):
            raise ValueError("Initial state does not match camera array")
        flat, iterations, history, converged, seconds = self._iterate(mapping, initial)
        with threadpool_limits(limits=1, user_api="blas"):
            h = np.asarray(fields(flat))
        state = flat.reshape(count, -1, 3)
        b = MU0 * (h + state)
        residual = np.array([self.material_residual(m, hi, camera) for m, hi in zip(state, h)])
        relative = float(np.linalg.norm(residual) / max(np.linalg.norm(state), 1.))
        return {"state": state, "h": h, "b": b, "centers": centers, "windings": windings,
                "converged": converged, "iterations": iterations, "residual_history": history,
                "constitutive_relative_residual": relative,
                "runtime": {"operator_build_seconds": build_seconds, "solve_seconds": seconds,
                            "total_seconds": perf_counter() - start}, "operator_estimated_mb": estimated_mb}

    def analyze(self, config, solution):
        start = perf_counter()
        camera = self.camera
        ids = [c["id"] for c in config["cameras"]]
        sources = []
        # Individual magnet fields, induced yoke fields, and currents all sum
        # to the monitor value. A current never exerts force on its own winding.
        for i, cid in enumerate(ids):
            for oid, obj in enumerate(camera.metadata["objects"]):
                material = self.materials[obj.get("material_id", obj["material"])]
                if obj["material"] != "copper" or getattr(material, "mu_r", 1.) != 1.:
                    sources.append((i, cid, obj["name"], obj["material"], camera.object_ids == oid))
            sources.append((i, cid, "WINDING", "current", None))
        values = np.zeros((len(sources), len(config["monitors"]), 3))
        for k, (i, cid, name, kind, mask) in enumerate(sources):
            self.check_budget()
            for j, monitor in enumerate(config["monitors"]):
                if monitor["quantity"] == "B":
                    points = np.array([monitor["position_mm"]]) * 1e-3
                    receiver = None
                else:
                    receiver = ids.index(monitor["camera_id"])
                    points = solution["windings"][receiver].points
                if kind == "current":
                    field = (np.zeros_like(points) if receiver == i else
                             MU0 * winding_field(points, solution["windings"][i]))
                else:
                    field = volume_flux_density(points, solution["centers"][i][mask], camera.volumes[mask],
                                                solution["state"][i][mask])
                values[k, j] = field[0] if receiver is None else lorentz_force(solution["windings"][receiver], field)
        total = values.sum(axis=0)
        monitors, score = monitor_values(config, total)
        for row, monitor in zip(monitors, config["monitors"]):
            if monitor["quantity"] == "B":
                row["position_mm"] = monitor["position_mm"]
            else:
                receiver = config["cameras"][ids.index(monitor["camera_id"])]
                coil = camera.metadata["coil"]
                row.update(camera_id=monitor["camera_id"], position_mm=(np.asarray(receiver["position_mm"]) +
                           np.r_[coil["center_m"], np.mean(coil["z_m"])] * 1e3).tolist(),
                           position_note="Coil center for display; force is integrated over the entire winding")

        def record(cid, name, kind, value):
            _, without = monitor_values(config, total - value)
            strengths = [m["weight"] * float(np.linalg.norm(v)) / m["scale"] for m, v in zip(config["monitors"], value)]
            return {"camera_id": cid, "object": name, "kind": kind, "vectors": value.tolist(),
                    "normalized_strength": max(strengths), "frozen_removal_score_change": without - score}

        ranked = [record(cid, name, kind, v) for (_, cid, name, kind, _), v in zip(sources, values)]
        ranked.sort(key=lambda row: (-row["normalized_strength"], row["camera_id"], row["object"]))
        cameras = [record(cid, "ALL", "camera", values[[s[1] == cid for s in sources]].sum(axis=0)) for cid in ids]
        cameras.sort(key=lambda row: (-row["normalized_strength"], row["camera_id"]))
        return {"fidelity": "low-fidelity", "native_verified": False,
                "physics_source": camera.metadata.get("aedt_source", {"kind": "legacy STEP with benchmark material assumptions"}),
                "objective": "max(weight * abs(value-target) / scale)",
                "score": score, "monitors": monitors, "sources": ranked, "cameras": cameras,
                "magnets": [r for r in ranked if r["kind"] == "magnet"],
                "attribution": "Source decomposition at the fully coupled state; frozen removal is not a new physical solve",
                "converged": solution["converged"], "iterations": solution["iterations"],
                "residual_history": solution["residual_history"],
                "constitutive_relative_residual": solution["constitutive_relative_residual"],
                "runtime": solution["runtime"] | {"monitor_seconds": perf_counter() - start},
                "self_cache": self.cache_info, "operator_estimated_mb": solution["operator_estimated_mb"]}

    def evaluate(self, config, initial_state=None):
        result = self.solve_layout(config, initial_state)
        report = self.analyze(config, result)
        return report, result
