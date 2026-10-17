import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .interaction import MU0


def feature_table(solver, result):
    cameras = (solver.camera_a, solver.camera_b)
    centers = np.r_[cameras[0].centers, cameras[1].centers + result.offset_b]
    h, b, m = np.concatenate(result.h), np.concatenate(result.b), np.concatenate(result.state)
    volumes = np.r_[cameras[0].volumes, cameras[1].volumes]
    mats = np.r_[cameras[0].material_ids, cameras[1].material_ids]
    names = np.concatenate([np.array([c.metadata["objects"][i]["name"] for i in c.object_ids]) for c in cameras])
    module = np.r_[np.full(len(cameras[0].centers), "A"), np.full(len(cameras[1].centers), "B")]
    distances, neighbors = cKDTree(centers).query(centers, k=min(9, len(centers)))
    neighbors, distances = neighbors[:, 1:], distances[:, 1:]
    grad_b = np.sqrt(np.mean((np.linalg.norm(b[neighbors] - b[:, None], axis=-1) / np.maximum(distances, 1e-12))**2, axis=1))
    grad_h = np.sqrt(np.mean((np.linalg.norm(h[neighbors] - h[:, None], axis=-1) / np.maximum(distances, 1e-12))**2, axis=1))
    variation = np.sqrt(np.mean(np.linalg.norm(b[neighbors] - b[:, None], axis=-1)**2, axis=1))
    hmag, bmag, mmag = [np.linalg.norm(v, axis=1) for v in (h, b, m)]
    mu_tangent = np.ones(len(centers))
    saturation = np.zeros(len(centers))
    for name, material in solver.materials.items():
        mask = mats == name
        if hasattr(material, "tangent_mu"):
            mu_tangent[mask] = material.tangent_mu(hmag[mask])
            saturation[mask] = bmag[mask] / material.saturation_t
        else:
            mu_tangent[mask] = material.mu_r
    other_distance = np.empty(len(centers))
    for label in ("A", "B"):
        mask = module == label
        nearest, indices = cKDTree(centers[~mask]).query(centers[mask])
        other_distance[mask] = np.maximum(0, nearest - .5 * np.cbrt(volumes[~mask][indices]))
    magnet_distance, mag_indices = cKDTree(centers[mats == "magnet"]).query(centers)
    magnet_distance = np.maximum(0, magnet_distance - .5 * np.cbrt(volumes[mats == "magnet"][mag_indices]))
    interface = np.r_[cameras[0].interface_distance, cameras[1].interface_distance]
    score = np.maximum.reduce([bmag / 1.6, grad_b * np.cbrt(volumes) / .5, saturation])
    level = np.digitize(score, [.25, .6, .9]).astype(int)
    data = {"x": centers[:, 0], "y": centers[:, 1], "z": centers[:, 2], "volume": volumes,
            "module": module, "object": names, "material": mats,
            "Bmag": bmag, "Hmag": hmag, "Mmag": mmag, "gradB": grad_b, "gradH": grad_h,
            "local_field_variation": variation, "mu_secant": bmag / (MU0 * np.maximum(hmag, 1e-9)),
            "mu_tangent": mu_tangent, "saturation": saturation, "interface_distance": interface,
            "other_camera_distance": other_distance, "nearest_magnet_distance": magnet_distance,
            "residual": np.linalg.norm(np.concatenate(result.constitutive_residual), axis=1),
            "refinement_level": level}
    for prefix, vectors in [("B", b), ("H", h), ("M", m)]:
        for i, axis in enumerate("xyz"):
            data[prefix + axis] = vectors[:, i]
    return pd.DataFrame(data)
