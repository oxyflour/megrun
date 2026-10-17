"""Read a native AEDT camera template into a portable, hashed physics source.

The native project is opened only as a disposable copy. No analysis is submitted.
Unsupported material laws, coordinate systems and current paths fail explicitly.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter

import numpy as np

from .array_config import ROOT, write_json
from .geometry import EXPECTED, load_camera
from .interaction import MU0
from .materials import LinearMaterial, SoftMagneticMaterial


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scalar(value, unit=""):
    """Literal SI values only: expressions must never become guessed constants."""
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*([A-Za-z_]*)\s*", str(value))
    if not match or match[2] not in ("", unit):
        raise ValueError(f"Unsupported AEDT expression/units: {value!r}; require literal {unit}")
    result = float(match[1])
    if not np.isfinite(result):
        raise ValueError("Non-finite AEDT property")
    return result


def material_spec(name, props):
    """Preserve signed Hc from raw properties (old PyAEDT value loses its sign)."""
    if props.get("CoordinateSystemType", "Cartesian") != "Cartesian" or props.get("ModifierData"):
        raise ValueError(f"Unsupported coordinate system or material modifiers: {name}")
    permeability = props.get("permeability")
    if permeability is None:
        raise ValueError(f"Missing native permeability: {name}")
    coercivity = props.get("magnetic_coercivity")
    hc, direction = 0., np.zeros(3)
    if coercivity:
        if not isinstance(coercivity, dict) or coercivity.get("property_type") != "VectorProperty":
            raise ValueError(f"Unsupported coercivity: {name}")
        hc = scalar(coercivity["Magnitude"], "A_per_meter")
        direction = np.array([scalar(coercivity[f"DirComp{i}"]) for i in (1, 2, 3)])
        if hc and not np.isclose(np.linalg.norm(direction), 1.):
            raise ValueError(f"Native coercivity direction is not a unit vector: {name}")
    result = {"name": name, "raw_properties": deepcopy(props), "Hc_A_per_m": hc,
              "Hc_direction_global": direction.tolist()}
    if isinstance(permeability, dict):
        if (permeability.get("property_type") != "nonlinear" or
                permeability.get("BTypeForSingleCurve") != "normal" or
                permeability.get("IsTemperatureDependent") or hc):
            raise ValueError(f"Only isotropic, single normal B-H curves are supported: {name}")
        if permeability.get("HUnit") != "A_per_meter" or permeability.get("BUnit") != "tesla":
            raise ValueError(f"Unsupported native B-H units: {name}")
        coords = permeability["BHCoordinates"]
        points = coords.get("Point")
        if points is None:
            points = np.asarray(coords["Points"], float).reshape(-1, 2).tolist()
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError(f"Invalid native B-H coordinates: {name}")
        # Installed AEDT 2022 R2 serializes BHCoordinates as [H, B].
        SoftMagneticMaterial(points[:, 0], points[:, 1], interpolation="linear")
        result.update(law="nonlinear_BH", BH_H_A_per_m_B_T=points.tolist(),
                      interpolation="linear", remanence_A_per_m=[0., 0., 0.],
                      high_field_extension="last B-H segment; coarse-model approximation")
    else:
        mu = scalar(permeability)
        LinearMaterial(mu)
        result.update(law="linear", mu_r=mu, remanence_A_per_m=(-mu * hc * direction).tolist())
    return result


def read_source(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("kind") != "aedt_camera_template":
        raise ValueError("Unsupported AEDT source bundle")
    for name, spec in data["materials"].items():
        derived = material_spec(name, spec["raw_properties"])
        if derived != spec:
            raise ValueError(f"Normalized material differs from native properties: {name}")
    for name, expected in data["files_sha256"].items():
        item = (path.parent / name).resolve()
        if not item.is_relative_to(path.parent) or sha256(item) != expected:
            raise ValueError(f"AEDT source bundle checksum mismatch: {name}")
    return data


def load_source_camera(path, cell_mm=.7, cache_dir=".cache"):
    source = read_source(path)
    specs = source["materials"]
    materials = {}
    for name, spec in specs.items():
        if spec["law"] == "linear":
            materials[name] = LinearMaterial(spec["mu_r"])
        else:
            table = np.asarray(spec["BH_H_A_per_m_B_T"])
            materials[name] = SoftMagneticMaterial(table[:, 0], table[:, 1], interpolation="linear")
    assignments = {obj["name"]: {"material_id": obj["material_name"], "kind": obj["kind"],
                    "remanence_A_per_m": specs[obj["material_name"]]["remanence_A_per_m"]}
                   for obj in source["objects"]}
    camera = load_camera(Path(path).parent / "template.step", cell_mm, cache_dir,
                         body_materials=assignments)
    camera.metadata["aedt_source"] = {"bundle": str(Path(path).resolve()), "bundle_sha256": sha256(path),
                                      **source["provenance"]}
    return camera, materials


def audit_saved_materials(project, source):
    """Read the saved AEDT bytes, independently of PyAEDT's material cache."""
    from pyaedt.generic.LoadAEDTFile import load_keyword_in_aedt_file
    props = load_keyword_in_aedt_file(str(project), "Materials")["Materials"]
    folded = {n.lower(): p for n, p in props.items()}
    checked = []
    for name, expected in source["materials"].items():
        actual = material_spec(name, folded[name.lower()])
        for key in ("law", "mu_r", "BH_H_A_per_m_B_T", "Hc_A_per_m", "Hc_direction_global", "remanence_A_per_m"):
            if actual.get(key) != expected.get(key):
                raise ValueError(f"Saved native material differs from AEDT source: {name}/{key}")
        checked.append(name)
    return {"saved_native_materials_verified": True, "materials": checked,
            "project_sha256": sha256(project), "method": "Parse saved AEDT Materials, independent of in-memory cache"}


def extract_worker(descriptor, output):
    from .array_maxwell import AEDT, launch_owned_desktop, require
    import cadquery as cq
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings

    output = Path(output).resolve()
    original = Path(descriptor["path"]).resolve()
    before = sha256(original)
    native_copy = output / "source_copy" / "Source.aedt"
    native_copy.parent.mkdir()
    shutil.copy2(original, native_copy)
    if sha256(native_copy) != before:
        raise IOError("Native project copy checksum mismatch")
    os.environ["ANSYSEM_ROOT222"] = str(AEDT.parent)
    settings.use_grpc_api, settings.enable_error_handler = True, False
    desktop, port = launch_owned_desktop(output)
    app = None
    try:
        app = Maxwell3d(projectname=str(native_copy), designname=descriptor["design"],
                        specified_version="2022.2", non_graphical=True, new_desktop_session=False,
                        machine="localhost", port=port, aedt_process_id=desktop.pid, close_on_exit=False)
        if app.design_name != descriptor["design"] or app.solution_type != "Magnetostatic":
            raise ValueError("Require an existing Magnetostatic design")
        if app.modeler.model_units != "mm":
            raise ValueError("This extractor currently requires native model units mm")
        setups = {s.name: dict(s.props) for s in app.setups}
        if any(p.get("SmoothBHCurve", False) for p in setups.values()):
            raise ValueError("Smoothed native B-H curves need a separate material adapter")
        mapping = descriptor.get("objects") or {n: descriptor.get("template_prefix", "") + n for n in EXPECTED}
        if set(mapping) != EXPECTED or len(set(mapping.values())) != len(EXPECTED):
            raise ValueError("Explicit template mapping must map the six camera body roles one-to-one")
        inventory = [{"name": n, "material_name": app.modeler[n].material_name,
                      "bounds_mm": np.asarray(app.modeler[n].bounding_box).reshape(2, 3).tolist(),
                      "volume_mm3": float(app.modeler[n].volume)} for n in app.modeler.solid_names]
        raw = output / "native_geometry"
        raw.mkdir()
        materials, objects, hashes = {}, [], {}
        origin = np.asarray(descriptor.get("origin_mm", [0., 0., 0.]), float)
        assembly = cq.Assembly(name="AEDTCameraTemplate")
        for logical, native in sorted(mapping.items()):
            if native not in app.modeler.solid_names:
                raise ValueError(f"Missing native solid: {native}")
            obj = app.modeler[native]
            if obj.part_coordinate_system != "Global" or not obj.model or not obj.solve_inside:
                raise ValueError(f"Unsupported local coordinates/non-model/non-volume body: {native}")
            mat_name = obj.material_name
            if mat_name not in materials:
                materials[mat_name] = material_spec(mat_name, dict(app.materials[mat_name]._props))
            spec = materials[mat_name]
            kind = "magnet" if np.linalg.norm(spec["remanence_A_per_m"]) else "yoke" if spec["law"] == "nonlinear_BH" else "linear"
            if logical == "AF_COIL_COPPER":
                if kind != "linear":
                    raise ValueError("Winding must have a linear non-permanent material")
                kind = "copper"
            require(app.export_3d_model(logical, str(raw), ".step", object_list=[native]), "native solid export")
            file = raw / (logical + ".step")
            shape = cq.importers.importStep(str(file)).val()
            if len(shape.Solids()) != 1:
                raise ValueError(f"Native CAD export is not one solid: {native}")
            box = shape.BoundingBox()
            bounds = np.array([[box.xmin, box.ymin, box.zmin], [box.xmax, box.ymax, box.zmax]])
            actual = np.asarray(obj.bounding_box).reshape(2, 3)
            if not np.allclose(bounds, actual, atol=2e-5, rtol=1e-7) or not np.isclose(shape.Volume(), float(obj.volume), rtol=1e-6):
                raise ValueError(f"Exported geometry differs from AEDT: {native}")
            assembly.add(shape.translate(tuple(-origin)), name=logical)
            hashes[str(file.relative_to(output))] = sha256(file)
            objects.append({"name": logical, "native_name": native, "material_name": mat_name, "kind": kind,
                            "part_coordinate_system": "Global", "solve_inside": bool(obj.solve_inside),
                            "bounds_mm": (actual - origin).tolist(), "volume_mm3": float(obj.volume)})
        assembly.save(str(output / "template.step"))
        hashes["template.step"] = sha256(output / "template.step")
        boundaries = [{"name": b.name, "type": b.type, "properties": dict(b.props)} for b in app.boundaries]
        write_json(output / "native_boundaries.json", boundaries)
        hashes["native_boundaries.json"] = sha256(output / "native_boundaries.json")
        # Save enough geometry to audit the current terminal and its normal.
        terminals = []
        for b in boundaries:
            if b["type"] == "Current":
                for n in b["properties"].get("Objects", []):
                    obj = app.modeler[n]
                    terminals.append({"boundary": b["name"], "object": n,
                                      "bounds_mm": np.asarray(obj.bounding_box).reshape(2, 3).tolist(),
                                      "normals": [list(f.normal) for f in obj.faces]})
        source = {"schema_version": 1, "kind": "aedt_camera_template", "units": "mm",
                  "provenance": {"project": str(original), "project_sha256": before,
                                 "design": descriptor["design"], "native_version": "2022.2",
                                 "template_origin_mm": origin.tolist(), "native_solve_submitted": False},
                  "scope": "Selected native camera template, replicated by translation; other assembly bodies are inventoried, not included",
                  "objects": objects, "materials": materials, "native_solid_inventory": inventory,
                  "native_setups": setups, "boundaries": boundaries, "current_terminals": terminals,
                  "files_sha256": hashes}
        # Freeze the exact native file used for the extraction as well.
        if sha256(native_copy) != before:
            raise IOError("Native source copy changed during geometry/material extraction")
        hashes[str(native_copy.relative_to(output))] = before
        # Validate the actual downstream voxelizer too, before publishing a source.
        pending = output / "source.pending.json"
        write_json(pending, source)
        camera, _ = load_source_camera(pending, cache_dir=output / "cache")
        for body in camera.metadata["objects"]:
            expected = next(o for o in objects if o["name"] == body["name"])
            if not np.allclose(body["bounds_mm"], expected["bounds_mm"], atol=2e-5) or not np.isclose(body["cad_volume_mm3"], expected["volume_mm3"], rtol=1e-6):
                raise ValueError("Downstream voxel geometry differs from native template")
        pending.replace(output / "source.json")
        write_json(output / "extraction_audit.json", {"geometry_and_materials_from_aedt": True,
                   "native_body_count": len(inventory), "template_body_count": len(objects),
                   "material_count": len(materials), "cells": len(camera.centers),
                   "volume_and_bounds_verified": True, "original_sha256": before,
                   "source_sha256": sha256(output / "source.json"), "native_solve_submitted": False})
    finally:
        try:
            if app is not None:
                app.release_desktop(close_projects=True, close_desktop=True)
        finally:
            if desktop.poll() is None:
                desktop.terminate()
                desktop.wait(timeout=10)
            after = sha256(original)
            write_json(output / "original_integrity.json", {"before": before, "after": after, "unchanged": before == after})
            if after != before:
                raise IOError("Original AEDT file changed during extraction")


def extract_source(descriptor, output, deadline):
    from .array_maxwell import run_stage
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "request.json", descriptor)
    begin = perf_counter()
    record = {"native_solve_submitted": False}
    try:
        record["stage"] = run_stage([sys.executable, str(ROOT / "scripts/extract_camera_aedt.py"),
            "--worker", "--request", str(output / "request.json"), "--output", str(output)],
            output / "extract.log", deadline)
        read_source(output / "source.json")
        record["status"] = "extracted_verified"
        return output / "source.json"
    except BaseException as exc:
        record.update(status="stopped", failure=repr(exc))
        raise
    finally:
        record["seconds"] = perf_counter() - begin
        write_json(output / "extraction_record.json", record)
