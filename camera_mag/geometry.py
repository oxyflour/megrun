"""Named STEP solids -> boundary-sampled coarse volume cells using OCCT."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .cache import fingerprint
from .interaction import MU0

DIRECTIONS = {"MAGNET_X_POS": [-1, 0, 0], "MAGNET_X_NEG": [1, 0, 0],
              "MAGNET_Y_POS": [0, -1, 0], "MAGNET_Y_NEG": [0, 1, 0]}
EXPECTED = {"YOKE_SOFT_MAGNETIC", "AF_COIL_COPPER", *DIRECTIONS}


@dataclass
class Camera:
    centers: np.ndarray
    volumes: np.ndarray
    material_ids: np.ndarray
    object_ids: np.ndarray
    remanence: np.ndarray
    interface_distance: np.ndarray
    metadata: dict

    @property
    def extent(self):
        return np.diff(np.asarray(self.metadata["bounds_m"]), axis=0)[0]


def load_camera(step, cell_mm=.7, cache_dir=".cache", br_t=1.2, subsamples=3, body_materials=None):
    import cadquery as cq
    step = Path(step)
    if cell_mm <= 0 or subsamples < 1 or br_t < 0:
        raise ValueError("Invalid cell size, subsampling, or remanence")
    if body_materials is not None and set(body_materials) != EXPECTED:
        raise ValueError("Native material assignments must cover every camera body")
    key = fingerprint(step.read_bytes(), {"version": 3, "cq": cq.__version__, "cell_mm": cell_mm,
                                         "br_t": br_t if body_materials is None else None,
                                         "body_materials": body_materials, "subsamples": subsamples})
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"geometry_{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            meta["geometry_cache_hit"] = True
            return Camera(**{k: data[k] for k in ["centers", "volumes", "material_ids", "object_ids",
                          "remanence", "interface_distance"]}, metadata=meta)
    # CadQuery 2.7 imports STEP into its default millimetre coordinates.
    assembly = cq.Assembly.importStep(str(step))
    solids = {}

    def visit(node, parent_location):
        location = parent_location * node.loc
        if node.obj is not None:
            shape = node.obj.val() if isinstance(node.obj, cq.Workplane) else node.obj
            name = node.name
            if name in EXPECTED:
                if name in solids or len(shape.Solids()) != 1:
                    raise ValueError(f"Expected one independent solid named {name}")
                solids[name] = shape.located(location * shape.location())
        for child in node.children:
            visit(child, location)

    visit(assembly, cq.Location())
    if set(solids) != EXPECTED:
        raise ValueError(f"Required named solids missing: {EXPECTED - set(solids)}; imported {list(solids)}")
    centers, volumes, materials, objects, remanence, interface = [], [], [], [], [], []
    object_meta = []
    offsets = np.stack(np.meshgrid(*([((np.arange(subsamples) + .5) / subsamples)] * 3), indexing="ij"), -1).reshape(-1, 3)
    for oid, (name, shape) in enumerate(solids.items()):
        box = shape.BoundingBox()
        lower = np.array([box.xmin, box.ymin, box.zmin])
        upper = np.array([box.xmax, box.ymax, box.zmax])
        counts = np.maximum(1, np.ceil((upper - lower) / cell_mm).astype(int))
        widths = (upper - lower) / counts
        starts = np.stack(np.meshgrid(*(np.arange(n) for n in counts), indexing="ij"), -1).reshape(-1, 3)
        local_centers, local_volumes, local_interface = [], [], []
        boundary = cq.Compound.makeCompound(shape.Faces())
        for index in starts:
            probes = lower + (index + offsets) * widths
            inside = np.array([shape.isInside(tuple(p), tolerance=1e-7) for p in probes])
            if not inside.any():
                continue
            center = probes[inside].mean(axis=0)
            local_centers.append(center * 1e-3)
            local_volumes.append(np.prod(widths) * inside.mean() * 1e-9)
            local_interface.append(cq.Vertex.makeVertex(*center).distance(boundary) * 1e-3)
        if not local_centers:
            raise ValueError(f"No cells in {name}; decrease --cell-mm")
        local_volumes = np.asarray(local_volumes)
        sampled_volume = local_volumes.sum()
        exact_volume = shape.Volume() * 1e-9
        # Preserve the true dipole moment even for partially occupied boundary voxels.
        local_volumes *= exact_volume / sampled_volume
        material = "magnet" if name in DIRECTIONS else "yoke" if name.startswith("YOKE") else "copper"
        material_id = material
        mr = np.array(DIRECTIONS.get(name, [0, 0, 0])) * br_t / MU0
        if body_materials is not None:
            assignment = body_materials[name]
            material, material_id = assignment["kind"], assignment["material_id"]
            mr = np.asarray(assignment["remanence_A_per_m"], float)
            if mr.shape != (3,) or not np.isfinite(mr).all():
                raise ValueError("Invalid native remanence vector")
        centers.extend(local_centers)
        volumes.extend(local_volumes)
        materials.extend([material_id] * len(local_centers))
        objects.extend([oid] * len(local_centers))
        remanence.extend([mr] * len(local_centers))
        interface.extend(local_interface)
        object_meta.append({"name": name, "material": material, "material_id": material_id, "cells": len(local_centers),
                            "cad_volume_mm3": exact_volume * 1e9, "sampled_volume_mm3": sampled_volume * 1e9,
                            "volume_scale": exact_volume / sampled_volume,
                            "bounds_mm": [lower.tolist(), upper.tolist()]})
    bounds = np.asarray([o["bounds_mm"] for o in object_meta])
    coil = solids["AF_COIL_COPPER"]
    coil_box = coil.BoundingBox()
    cx, cy = (coil_box.xmin + coil_box.xmax) / 2, (coil_box.ymin + coil_box.ymax) / 2
    coords = np.array([v.Center().toTuple() for v in coil.Vertices()])
    x = np.unique(np.round(np.abs(coords[:, 0] - cx), 7))
    y = np.unique(np.round(np.abs(coords[:, 1] - cy), 7))
    x, y = x[x > 1e-7], y[y > 1e-7]
    if len(x) != 2 or len(y) != 2:
        raise ValueError("MVP winding requires an axis-aligned rectangular annular copper solid")
    predicted_volume = 4 * (x[-1] * y[-1] - x[0] * y[0]) * coil_box.zlen
    if not np.isclose(predicted_volume, coil.Volume(), rtol=.005):
        raise ValueError("Copper solid is not a rectangular annulus; explicit winding path required")
    metadata = {"step": str(step.resolve()), "sha256_key": key, "geometry_cache_hit": False,
                "cell_mm": cell_mm, "subsamples": subsamples, "br_t": br_t if body_materials is None else None, "objects": object_meta,
                "bounds_m": [bounds[:, 0, :].min(axis=0).tolist(), bounds[:, 1, :].max(axis=0).tolist()],
                "coil": {"center_m": [cx * 1e-3, cy * 1e-3], "inner_half_m": [x[0] * 1e-3, y[0] * 1e-3],
                         "outer_half_m": [x[-1] * 1e-3, y[-1] * 1e-3],
                         "z_m": [coil_box.zmin * 1e-3, coil_box.zmax * 1e-3]}}
    metadata["bounds_m"] = (np.asarray(metadata["bounds_m"]) * 1e-3).tolist()
    camera = Camera(np.array(centers), np.array(volumes), np.array(materials), np.array(objects),
                    np.array(remanence), np.array(interface), metadata)
    np.savez_compressed(path, **{k: getattr(camera, k) for k in camera.__dataclass_fields__ if k != "metadata"},
                        metadata=json.dumps(metadata))
    return camera
