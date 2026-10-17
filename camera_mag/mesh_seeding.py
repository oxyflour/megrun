"""Budgeted, untrained face seeds from coarse fields; no Maxwell result inputs."""
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .materials import benchmark_materials
from .solver import CameraPairSolver


def _assign(app, assignment, length, name, *, faces=False, budget=None):
    if not assignment:
        return None
    operation = app.mesh.assign_length_mesh(assignment, inside_selection=not faces,
        maximum_length=f"{length:g}mm", maximum_elements=budget, name=name)
    if not operation:
        raise RuntimeError(f"Failed mesh operation {name}")
    return {"name": operation.name, "assignment": assignment, "on_faces": faces,
            "maximum_length_mm": length, "maximum_additional_elements": budget}


def _near_air(app, camera, offset_b):
    """Partition vacuum only; preserve the physical domain and STEP bodies."""
    original_volumes = {n: float(app.modeler[n].volume) for n in app.modeler.solid_names
                        if n.startswith(("A_", "B_"))}
    domain_volume = float(app.modeler["Region"].volume)
    # Use an explicit vacuum box for robust partition Booleans, preserving
    # exactly the automatic Region's outer coordinates and default boundary.
    outer = np.asarray(app.modeler["Region"].bounding_box)
    if not app.modeler.delete("Region"):
        raise RuntimeError("Could not replace automatic Region")
    if not app.modeler.create_box(outer[:3].tolist(), (outer[3:] - outer[:3]).tolist(),
                                 name="Region", material="vacuum"):
        raise RuntimeError("Could not restore outer vacuum box")
    bounds = np.asarray(camera.metadata["bounds_m"]) * 1e3
    shift = np.asarray(offset_b) * 1e3
    split_x = (bounds[1, 0] + bounds[0, 0] + shift[0]) / 2
    air = []
    for label, delta in zip("AB", (np.zeros(3), shift)):
        lo, hi = bounds[0] + delta - .6, bounds[1] + delta + .6
        if label == "A":
            hi[0] = min(hi[0], split_x)
        else:
            lo[0] = max(lo[0], split_x)
        name = "LF_NearAir_" + label
        box = app.modeler.create_box(lo.tolist(), (hi - lo).tolist(), name=name, material="vacuum")
        if not box:
            raise RuntimeError(f"Could not create near-field vacuum {name}")
        air.append({"object": name, "bounds_mm": [lo.tolist(), hi.tolist()]})
    if not app.modeler.subtract("Region", [a["object"] for a in air],
                                keep_originals=True):
        raise RuntimeError("Could not partition outer vacuum")
    for label, region in zip("AB", air):
        tools = [n for n in original_volumes if n.startswith(label + "_")]
        if not app.modeler.subtract(region["object"], tools, keep_originals=True):
            raise RuntimeError(f"Could not cut solids from {region['object']}")
    for name, volume in original_volumes.items():
        if not np.isclose(app.modeler[name].volume, volume, rtol=1e-9, atol=1e-8):
            raise RuntimeError(f"Air partition changed STEP body {name}")
    new_volume = sum(float(app.modeler[n].volume)
                     for n in ["Region"] + [a["object"] for a in air] + list(original_volumes))
    if not np.isclose(new_volume, domain_volume, rtol=1e-8, atol=1e-6):
        raise RuntimeError("Air partition changed domain volume")
    return {"regions": air, "STEP_body_volumes_preserved": True,
            "outer_domain_volume_mm3": domain_volume, "partition_volume_mm3": new_volume}


def _local_regions(app, camera, result):
    """Non-model primitive boxes constrain mesh without partitioning physics."""
    model_objects = set(app.modeler.model_objects)
    volumes = {n: float(app.modeler[n].volume) for n in app.modeler.solid_names if n in model_objects}
    regions, corners = [], []
    coil = camera.metadata["coil"]
    inner, outer = np.asarray(coil["inner_half_m"]) * 1e3, np.asarray(coil["outer_half_m"]) * 1e3
    zlo, zhi = np.asarray(coil["z_m"]) * 1e3

    def box(name, lo, hi, kind, score=None):
        obj = app.modeler.create_box(lo.tolist(), (hi - lo).tolist(), name=name, material="vacuum")
        if not obj:
            raise RuntimeError(f"Could not create mesh-only box {name}")
        obj.model = False
        if name not in app.modeler.non_model_objects:
            raise RuntimeError(f"Mesh-only box {name} became a physical body")
        regions.append({"object": name, "kind": kind, "bounds_mm": [lo.tolist(), hi.tolist()],
                        "model": False, "score": score})

    for label, shift, b in zip("AB", (np.zeros(3), result.offset_b * 1e3), result.b):
        # Thin bars surrounding the winding include the narrow magnet/coil air
        # gaps. A sub-mm scalar length there avoids refining all remote air.
        for axis in (0, 1):
            other = 1 - axis
            for sign in (-1, 1):
                lo = np.array([-outer[0] - .15, -outer[1] - .15, zlo - .1])
                hi = np.array([outer[0] + .15, outer[1] + .15, zhi + .1])
                endpoints = sorted((sign * (inner[axis] - .1), sign * (outer[axis] + .2)))
                lo[axis], hi[axis] = endpoints
                side = "P" if sign > 0 else "N"
                box(f"LF_LocalCoil_{label}_{axis}_{side}", lo + shift, hi + shift, "coil_and_gap")

        oid = next(i for i, obj in enumerate(camera.metadata["objects"]) if obj["material"] == "yoke")
        mask = camera.object_ids == oid
        points, field = camera.centers[mask] * 1e3 + shift, b[mask]
        tree = cKDTree(points)
        _, neighbors = tree.query(points, k=min(7, len(points)))
        variation = np.sqrt(np.mean(np.linalg.norm(field[neighbors[:, 1:]] - field[:, None], axis=2)**2, axis=1))
        magnitude = np.linalg.norm(field, axis=1)
        variation /= np.maximum(magnitude, .08)
        ranked = []
        positions = sorted({tuple(np.round(v.position, 7)) for v in app.modeler[label + "_YOKE_SOFT_MAGNETIC"].vertices})
        for position in positions:
            point = np.asarray(position)
            _, indices = tree.query(point, k=min(4, len(points)))
            local = point - shift
            # Distance to the actual annular winding, rather than its filled bbox.
            bar_distances = []
            for axis in (0, 1):
                other = 1 - axis
                for sign in (-1, 1):
                    blo, bhi = np.array([-outer[0], -outer[1], zlo]), np.array([outer[0], outer[1], zhi])
                    blo[axis], bhi[axis] = sorted((sign * inner[axis], sign * outer[axis]))
                    bar_distances.append(np.linalg.norm(np.maximum(np.maximum(blo - local, local - bhi), 0)))
            proximity = float(np.exp(-min(bar_distances)))
            grad = float(np.quantile(variation[indices], .8))
            flux = float(np.quantile(magnitude[indices], .8))
            score = (.65 * min(grad / .25, 2.) + .35 * min(flux / .8, 2.)) * (.3 + .7 * proximity)
            ranked.append({"module": label, "center_mm": point.tolist(), "relative_B_variation": grad,
                           "B_T": flux, "coil_proximity": proximity, "score": score})
        ranked.sort(key=lambda v: (-v["score"], tuple(v["center_mm"])))
        for i, corner in enumerate(ranked):
            corner["selected"] = i < 4
            corners.append(corner)
            if corner["selected"]:
                center = np.asarray(corner["center_mm"])
                box(f"LF_LocalCorner_{label}_{i:02d}", center - .12, center + .12, "yoke_corner", corner["score"])
    if set(app.modeler.model_objects) != model_objects:
        raise RuntimeError("Local seeds changed the physical model object set")
    for name, volume in volumes.items():
        if not np.isclose(app.modeler[name].volume, volume, rtol=1e-10, atol=1e-8):
            raise RuntimeError(f"Local seeds changed volume of {name}")
    return {"regions": regions, "corner_scores": corners, "physical_objects_unchanged": True,
            "physical_volumes_unchanged": True, "primitive_non_model_boxes": True}


def apply_mesh_policy(app, camera, policy, gap_mm, current_a, turns):
    start = perf_counter()
    objects = [name for name in app.modeler.solid_names if name.startswith(("A_", "B_"))]
    if policy == "adaptive":
        return {"policy": policy, "operations": [], "preprocessing_seconds": perf_counter() - start}
    if policy == "geometry":
        op = _assign(app, objects, .5, "Geometry_05mm")
        return {"policy": policy, "description": "Manual uniform 0.5 mm inside all camera solids",
                "operations": [op], "preprocessing_seconds": perf_counter() - start}
    if policy not in ("low-fidelity", "low-fidelity-aggressive", "low-fidelity-local"):
        raise ValueError(policy)
    aggressive = policy == "low-fidelity-aggressive"
    localized_policy = policy == "low-fidelity-local"

    solver = CameraPairSolver(camera, camera, benchmark_materials(), current_a=current_a,
                              turns=turns, blas_threads=1)
    solver.precompute_self_operators()
    result = solver.solve(gap_mm)
    if not result.converged:
        raise RuntimeError("Coarse fields did not converge")
    air_partition = _near_air(app, camera, result.offset_b) if aggressive else None
    local_regions = _local_regions(app, camera, result) if localized_policy else None
    # The old global heuristic mixes jumps across different materials. Rank
    # variation within each object instead; retain geometry and normal-flux cues.
    faces = []
    coil = camera.metadata["coil"]
    coil_xy = np.asarray(coil["outer_half_m"])
    coil_z = np.asarray(coil["z_m"])
    for label, b in zip("AB", result.b):
        shift = result.offset_b if label == "B" else np.zeros(3)
        for oid, obj_meta in enumerate(camera.metadata["objects"]):
            name = label + "_" + obj_meta["name"]
            if "COIL" in name:
                continue  # Coil interior has a shared geometric resolution floor.
            mask = camera.object_ids == oid
            points, fields = camera.centers[mask] + shift, b[mask]
            tree = cKDTree(points)
            k = min(7, len(points))
            distances, neighbors = tree.query(points, k=k)
            variation = np.sqrt(np.mean(np.linalg.norm(fields[neighbors[:, 1:]] - fields[:, None], axis=2)**2, axis=1))
            bmag = np.linalg.norm(fields, axis=1)
            relative_variation = variation / np.maximum(bmag, .08)
            for face in app.modeler[name].faces:
                center = np.asarray(face.center) * 1e-3
                vertices = [np.asarray(vertex.position) * 1e-3 for vertex in face.vertices]
                probes = np.r_[center[None], np.asarray(vertices)]
                _, indices = tree.query(probes)
                sampled_b = fields[indices]
                normal = np.asarray(face.normal)
                flux = float(np.quantile(np.abs(sampled_b @ normal), .8))
                grad = float(np.quantile(relative_variation[indices], .8))
                saturation = float(np.quantile(bmag[indices], .8) / 1.6) if obj_meta["material"] == "yoke" else 0.
                local = probes - shift
                outside_xy = np.maximum(np.abs(local[:, :2]) - coil_xy, 0)
                outside_z = np.maximum(np.maximum(coil_z[0] - local[:, 2], local[:, 2] - coil_z[1]), 0)
                proximity = float(np.exp(-np.min(np.sqrt(np.sum(outside_xy**2, axis=1) + outside_z**2)) / .001))
                score = (.55 * min(grad / .25, 2.) + .30 * min(flux / .6, 2.)
                         + .15 * max(saturation - .6, 0) / .4) * (.4 + .6 * proximity)
                faces.append({"object": name, "face_id": int(face.id), "center_mm": (center * 1e3).tolist(),
                    "area_mm2": float(face.area), "relative_B_variation": grad, "normal_B_T": flux,
                    "yoke_saturation_ratio": saturation, "coil_proximity": proximity, "score": score})
    # Select by physical surface area, not face count: STEP topology can split a
    # surface without changing its physical importance. Ties use geometry.
    faces.sort(key=lambda f: (-f["score"], f["object"], tuple(f["center_mm"])))
    total_area = sum(f["area_mm2"] for f in faces)
    fine_fraction, medium_fraction = (.30, .60) if aggressive else (.22, .45)
    accumulated = 0.
    groups = {"fine": [], "medium": []}
    for face in faces:
        fraction = accumulated / total_area
        level = "fine" if fraction < fine_fraction else "medium" if fraction < medium_fraction else "coarse"
        face["level"] = level
        if level in groups:
            groups[level].append(face["face_id"])
        accumulated += face["area_mm2"]
    if localized_policy:
        operations = [_assign(app, objects, .4, "LF3_Base_04mm"),
                      _assign(app, [n for n in objects if "COIL" in n], .28, "LF3_Coil_028mm"),
                      _assign(app, groups["fine"], .2, "LF3_FluxFaces", faces=True, budget=90000),
                      _assign(app, groups["medium"], .3, "LF3_MediumFaces", faces=True, budget=40000),
                      _assign(app, [r["object"] for r in local_regions["regions"] if r["kind"] == "coil_and_gap"],
                              .34, "LF3_CoilGapRegions", budget=60000),
                      _assign(app, [r["object"] for r in local_regions["regions"] if r["kind"] == "yoke_corner"],
                              .08, "LF3_CornerRegions", budget=30000)]
    elif aggressive:
        operations = [_assign(app, objects, .4, "LF2_Base_04mm"),
                      _assign(app, [n for n in objects if "COIL" in n], .28, "LF2_Coil_028mm"),
                      _assign(app, [a["object"] for a in air_partition["regions"]], .7, "LF2_NearAir_07mm")]
    else:
        operations = [_assign(app, objects, .7, "LF_Base_07mm"),
                      _assign(app, [n for n in objects if "COIL" in n], .5, "LF_Coil_05mm")]
    # Caps bound initial seed growth. Maxwell's adaptive stage remains free to
    # refine any region it finds necessary, including all unselected faces.
    if not localized_policy:
        operations.append(_assign(app, groups["medium"], .28 if aggressive else .4,
                                  "LF_MediumFaces", faces=True, budget=65000 if aggressive else 20000))
        operations.append(_assign(app, groups["fine"], .2 if aggressive else .3,
                                  "LF_FineFaces", faces=True, budget=130000 if aggressive else 35000))
    return {"policy": policy, "version": 4 if localized_policy else 2 if aggressive else 1,
            "source": "Untrained coarse-field ranking, no Maxwell solution or imported mesh inputs",
            "operations": [op for op in operations if op], "face_scores": faces,
            "air_partition": air_partition,
            "local_regions": local_regions,
            "fine_area_target_fraction": fine_fraction, "fine_and_medium_area_target_fraction": medium_fraction,
            "fine_area_actual_fraction": sum(f["area_mm2"] for f in faces if f["level"] == "fine") / total_area,
            "fine_and_medium_area_actual_fraction": sum(f["area_mm2"] for f in faces if f["level"] != "coarse") / total_area,
            "coarse_cell_mm": camera.metadata["cell_mm"], "coarse_iterations": result.iterations,
            "coarse_residual": result.residual_history[-1], "self_cache": solver.self_cache,
            "preprocessing_seconds": perf_counter() - start}
