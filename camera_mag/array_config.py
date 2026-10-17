"""Versioned, translation-only camera assemblies and monitor contracts."""
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def known_fields(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unexpected = set(value) - set(allowed.split())
    if unexpected:
        raise ValueError(f"Unsupported {name} fields: {sorted(unexpected)}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return array


def positive(value, name, *, zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"Invalid {name}")
    return value


def integer(value, name, minimum=1):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"Invalid {name}")
    return int(value)


def normalize_config(raw, base_dir=ROOT):
    known_fields(raw, "schema_version units N step aedt aedt_source aedt_source_sha256 cameras monitors search low_fidelity maxwell", "config")
    config = deepcopy(raw)
    if config.get("schema_version", 1) != 1 or config.get("units", "mm") != "mm":
        raise ValueError("Require schema_version=1 and units=mm")
    config.update(schema_version=1, units="mm")
    config["N"] = integer(config.get("N", 3), "N")
    cameras = config.get("cameras", [])
    if len(cameras) != config["N"]:
        raise ValueError("N (default 3) must equal the number of camera coordinates")
    source_keys = set(config) & {"aedt", "aedt_source", "step"}
    if len(source_keys) != 1:
        raise ValueError("Select exactly one physics source: aedt, aedt_source, or explicit legacy step")
    if "aedt" in config:
        aedt = config["aedt"]
        known_fields(aedt, "path design template_prefix origin_mm objects", "aedt")
        aedt["path"] = str((Path(base_dir) / aedt["path"]).resolve())
        if not Path(aedt["path"]).is_file():
            raise FileNotFoundError(aedt["path"])
        if not isinstance(aedt.get("design"), str) or not aedt["design"]:
            raise ValueError("AEDT source requires an explicit existing design name")
        aedt["origin_mm"] = vector(aedt.get("origin_mm", [0., 0., 0.]), "template origin_mm").tolist()
    elif "aedt_source" in config:
        from .aedt_source import read_source, sha256
        path = (Path(base_dir) / config["aedt_source"]).resolve()
        source = read_source(path)
        checksum = sha256(path)
        if config.get("aedt_source_sha256", checksum) != checksum:
            raise ValueError("Frozen AEDT source changed; extract a new bundle")
        config.update(aedt_source=str(path), aedt_source_sha256=checksum)
    else:
        config["step"] = str((Path(base_dir) / config["step"]).resolve())
        if not Path(config["step"]).is_file():
            raise FileNotFoundError(config["step"])
    ids = []
    for i, item in enumerate(cameras):
        if not isinstance(item, dict):
            cameras[i] = item = {"position_mm": item}
        known_fields(item, "id position_mm current_A turns movable move_axes bounds_mm", "camera")
        item.setdefault("id", f"C{i + 1}")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", item["id"]):
            raise ValueError("Camera IDs must be short ASCII identifiers")
        ids.append(item["id"])
        pos = vector(item["position_mm"], "camera position_mm")
        item["position_mm"] = pos.tolist()
        # Native-source excitation is explicit until a verified terminal adapter
        # is available; never invent the turn count from a Current boundary.
        if "aedt" in config or "aedt_source" in config:
            if "current_A" not in item or "turns" not in item:
                raise ValueError("AEDT-source cameras require explicit current_A and turns (native N*I alone cannot identify each)")
        item["current_A"] = float(item.get("current_A", .1))
        if not np.isfinite(item["current_A"]):
            raise ValueError("Non-finite camera current")
        item["turns"] = integer(item.get("turns", 100), "turns")
        item.setdefault("movable", True)
        if not isinstance(item["movable"], bool):
            raise ValueError("movable must be a boolean")
        axes = item.setdefault("move_axes", ["x", "y"])
        if not isinstance(axes, list) or len(set(axes)) != len(axes) or not set(axes) <= {"x", "y", "z"}:
            raise ValueError("move_axes must be a subset of x/y/z")
        # Absolute bounds are frozen into every next.json; they never recenter.
        bounds = np.asarray(item.setdefault("bounds_mm", [(pos - 2).tolist(), (pos + 2).tolist()]), float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
            raise ValueError("bounds_mm must be finite [lower_xyz, upper_xyz]")
        if np.any(pos < bounds[0] - 1e-9) or np.any(pos > bounds[1] + 1e-9):
            raise ValueError("Camera lies outside its absolute movement bounds")
        item["bounds_mm"] = bounds.tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate camera IDs")
    monitors = config.get("monitors", [])
    if not monitors:
        raise ValueError("At least one B or F monitor is required")
    monitor_ids = []
    for i, item in enumerate(monitors):
        if not isinstance(item, dict):
            monitors[i] = item = {"position_mm": item}
        known_fields(item, "id quantity position_mm camera_id component target weight scale", "monitor")
        item.setdefault("id", f"M{i + 1}")
        if not isinstance(item["id"], str) or not item["id"]:
            raise ValueError("Monitor ID must be a nonempty string")
        monitor_ids.append(item["id"])
        quantity = item.setdefault("quantity", "B")
        if quantity not in ("B", "F"):
            raise ValueError("Monitor quantity must be B or F (coil Lorentz force)")
        item.setdefault("component", "magnitude" if quantity == "B" else "z")
        if item["component"] not in ("magnitude", "x", "y", "z"):
            raise ValueError("Monitor component must be magnitude/x/y/z")
        if quantity == "B":
            item["position_mm"] = vector(item["position_mm"], "B monitor position_mm").tolist()
        elif item.get("camera_id") not in ids:
            raise ValueError("F monitor requires an existing camera_id; force acts on that entire coil")
        elif "position_mm" in item:
            raise ValueError("F monitors use camera_id; their displayed position follows the coil, not an independent point")
        item["target"] = float(item.get("target", 0.))
        if not np.isfinite(item["target"]):
            raise ValueError("Monitor target must be finite (T for B, N for F)")
        if item["component"] == "magnitude" and item["target"] < 0:
            raise ValueError("Magnitude target must be nonnegative")
        item["weight"] = positive(item.get("weight", 1.), "monitor weight")
        item["scale"] = positive(item.get("scale", 1.), "monitor scale")
    if len(set(monitor_ids)) != len(monitors):
        raise ValueError("Duplicate monitor IDs")
    if len({m["quantity"] for m in monitors}) > 1 and any(not isinstance(m, dict) or "scale" not in m for m in raw["monitors"]):
        raise ValueError("Mixed B/F objectives require explicit per-monitor scale (T or N)")
    lf = config.setdefault("low_fidelity", {})
    known_fields(lf, "cell_mm tol max_iterations max_operator_mb", "low_fidelity")
    lf["cell_mm"] = positive(lf.get("cell_mm", .7), "cell_mm")
    lf["tol"] = positive(lf.get("tol", 1e-7), "tol")
    lf["max_iterations"] = integer(lf.get("max_iterations", 1500), "LF max_iterations")
    lf["max_operator_mb"] = positive(lf.get("max_operator_mb", 512), "max_operator_mb")
    search = config.setdefault("search", {})
    known_fields(search, "step_mm top_cameras max_candidates min_relative_improvement stop_score clearance_mm", "search")
    search["step_mm"] = positive(search.get("step_mm", .5), "step_mm")
    search["top_cameras"] = integer(search.get("top_cameras", 2), "top_cameras")
    search["max_candidates"] = integer(search.get("max_candidates", 12), "max_candidates")
    search["min_relative_improvement"] = positive(search.get("min_relative_improvement", .001), "min_relative_improvement", zero=True)
    if search["min_relative_improvement"] >= 1:
        raise ValueError("min_relative_improvement must be below 1")
    search["stop_score"] = positive(search.get("stop_score", 0.), "stop_score", zero=True)
    search["clearance_mm"] = positive(search.get("clearance_mm", .2), "clearance_mm", zero=True)
    native = config.setdefault("maxwell", {})
    known_fields(native, "mesh_policy air_mm percent_error max_passes cores force_delta_N", "maxwell")
    native.setdefault("mesh_policy", "adaptive")
    if native["mesh_policy"] not in ("adaptive", "low-fidelity"):
        raise ValueError("Array mesh_policy must be adaptive or low-fidelity")
    native["air_mm"] = positive(native.get("air_mm", 20), "air_mm")
    native["percent_error"] = positive(native.get("percent_error", .5), "percent_error")
    native["max_passes"] = integer(native.get("max_passes", 12), "max_passes", 2)
    native["cores"] = integer(native.get("cores", 4), "cores")
    native["force_delta_N"] = positive(native.get("force_delta_N", 2e-5), "force_delta_N")
    return config


def load_config(path):
    path = Path(path).resolve()
    return normalize_config(json.loads(path.read_text(encoding="utf-8-sig")), path.parent)


def validate_layout(config, camera):
    bounds = np.asarray(camera.metadata["bounds_m"]) * 1e3
    boxes = [bounds + c["position_mm"] for c in config["cameras"]]
    gap = config["search"]["clearance_mm"]
    for i, box in enumerate(boxes):
        for other in boxes[:i]:
            separation = np.maximum(box[0] - other[1], other[0] - box[1])
            if np.all(separation < gap + 1e-10):
                raise ValueError("Camera envelope overlap or insufficient clearance")
    for monitor in config["monitors"]:
        if monitor["quantity"] == "B":
            pos = np.asarray(monitor["position_mm"])
            if any(np.all(pos >= b[0] - 1e-8) and np.all(pos <= b[1] + 1e-8) for b in boxes):
                raise ValueError("B monitors must lie outside the camera EM envelopes")
    return boxes


def moved_config(config, positions):
    result = deepcopy(config)
    for camera in result["cameras"]:
        if camera["id"] not in positions:
            continue
        old, new = np.asarray(camera["position_mm"]), vector(positions[camera["id"]], "proposed position")
        diff = new - old
        if not camera["movable"] and np.linalg.norm(diff) > 1e-9:
            raise ValueError("Cannot move a fixed camera")
        for axis, name in enumerate("xyz"):
            if name not in camera["move_axes"] and abs(diff[axis]) > 1e-9:
                raise ValueError("Proposed motion uses a locked axis")
        if np.linalg.norm(diff) > config["search"]["step_mm"] + 1e-9:
            raise ValueError("Proposed motion exceeds per-iteration step_mm")
        camera["position_mm"] = new.tolist()
    if set(positions) - {c["id"] for c in result["cameras"]}:
        raise ValueError("Unknown proposed camera ID")
    return normalize_config(result)


def monitor_values(config, vectors):
    rows = []
    for monitor, value in zip(config["monitors"], vectors, strict=True):
        value = vector(value, "monitor field/force")
        scalar = float(np.linalg.norm(value)) if monitor["component"] == "magnitude" else float(value["xyz".index(monitor["component"])])
        score = monitor["weight"] * abs(scalar - monitor["target"]) / monitor["scale"]
        rows.append({"id": monitor["id"], "quantity": monitor["quantity"], "unit": "T" if monitor["quantity"] == "B" else "N",
                     "vector": value.tolist(), "component": monitor["component"], "value": scalar, "score": score})
    return rows, max(row["score"] for row in rows)
