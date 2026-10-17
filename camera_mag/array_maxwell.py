"""Native assembly/extraction worker and budgeted batch driver for new runs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np
from scipy.spatial import cKDTree

from .array_config import ROOT, load_config, monitor_values, validate_layout, write_json
from .geometry import DIRECTIONS, EXPECTED, load_camera
from .interaction import MU0
from .materials import benchmark_materials

AEDT = Path(r"C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe")
DESIGN, SETUP = "CameraArray", "Reference"


def require(value, action):
    if value is False or value is None:
        raise RuntimeError(f"AEDT failed: {action}")
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def launch_owned_desktop(output, timeout_seconds=60):
    """One explicit startup attempt, with exit evidence and no library retries."""
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    command = [str(AEDT), "-grpcsrv", str(port), "-ng"]
    record = {"command": command, "port": port, "timeout_seconds": timeout_seconds}
    start = perf_counter()
    with (output / "native_startup.log").open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=AEDT.parent, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        record["pid"] = process.pid
        write_json(output / "startup_record.json", record)
        try:
            while perf_counter() - start < timeout_seconds:
                if process.poll() is not None:
                    raise RuntimeError(f"AEDT startup exited with code {process.returncode}; inspect native_startup.log")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        record["status"] = "grpc_listening"
                        return process, port
                except OSError:
                    sleep(.25)
            raise TimeoutError("AEDT gRPC startup exceeded 60 seconds; no automatic retry")
        except BaseException as exc:
            record["failure"] = repr(exc)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            raise
        finally:
            record.update(seconds=perf_counter() - start, returncode=process.poll())
            write_json(output / "startup_record.json", record)


def apply_array_seeds(app, camera, config, fields):
    """Rank translated faces by within-body B variation, flux and saturation."""
    faces, operations = [], []
    for item, b in zip(config["cameras"], fields, strict=True):
        offset = np.asarray(item["position_mm"]) * 1e-3
        for oid, obj in enumerate(camera.metadata["objects"]):
            if obj["material"] == "copper":
                continue
            mask = camera.object_ids == oid
            points, values = camera.centers[mask] + offset, b[mask]
            tree = cKDTree(points)
            k = min(7, len(points))
            if k > 1:
                neighbors = tree.query(points, k=k)[1]
                variation = np.sqrt(np.mean(np.linalg.norm(values[neighbors[:, 1:]] - values[:, None], axis=2)**2, axis=1))
            else:
                variation = np.zeros(len(points))
            norms = np.linalg.norm(values, axis=1)
            variation /= np.maximum(norms, .08)
            name = item["id"] + "_" + obj["name"]
            for face in app.modeler[name].faces:
                probes = np.array([face.center] + [v.position for v in face.vertices]) * 1e-3
                indices = tree.query(probes)[1]
                flux = float(np.quantile(np.abs(values[indices] @ np.asarray(face.normal)), .8))
                gradient = float(np.quantile(variation[indices], .8))
                saturation = float(np.quantile(norms[indices], .8) / 1.6) if obj["material"] == "yoke" else 0.
                score = .55 * min(gradient / .25, 2) + .30 * min(flux / .6, 2) + .15 * max(saturation - .6, 0) / .4
                faces.append({"object": name, "face_id": int(face.id), "area_mm2": float(face.area),
                              "center_mm": list(face.center), "score": score,
                              "normal_B_T": flux, "relative_B_variation": gradient, "saturation_ratio": saturation})
    faces.sort(key=lambda f: (-f["score"], f["object"], tuple(f["center_mm"])))
    area, cumulative, chosen = sum(f["area_mm2"] for f in faces), 0., []
    for face in faces:
        face["selected"] = cumulative < .3 * area
        if face["selected"]:
            chosen.append(face["face_id"])
        cumulative += face["area_mm2"]
    for name, assignment, length, on_faces, budget in [
        ("ArrayLF_Base", [c["id"] + "_" + s for c in config["cameras"] for s in sorted(EXPECTED)], .7, False, None),
        ("ArrayLF_Coil", [c["id"] + "_AF_COIL_COPPER" for c in config["cameras"]], .5, False, None),
        ("ArrayLF_Faces", chosen, .3, True, 35000 * config["N"])]:
        op = require(app.mesh.assign_length_mesh(assignment, inside_selection=not on_faces,
                     maximum_length=f"{length}mm", maximum_elements=budget, name=name), "array LF mesh")
        operations.append({"name": op.name, "assignment": assignment, "length_mm": length,
                           "on_faces": on_faces, "maximum_additional_elements": budget})
    return {"policy": "low-fidelity", "source": "New N-camera fully coupled coarse fields; untrained rule",
            "face_scores": faces, "operations": operations, "fine_area_fraction": .3,
            "accuracy_or_speedup_proven": False}


def audit_array_build(app, config, project, output):
    """Check saved assembly and repair the legacy fuzzy-matched setup flag."""
    source = None
    if "aedt_source" in config:
        from .aedt_source import load_source_camera, material_spec, read_source
        camera, _ = load_source_camera(config["aedt_source"], config["low_fidelity"]["cell_mm"])
        source = read_source(config["aedt_source"])
    else:
        camera = load_camera(config["step"], config["low_fidelity"]["cell_mm"])
    setup = app.get_setup(SETUP)
    previous = setup.props.get("SolveFieldOnly")
    setup.props["SolveFieldOnly"] = False
    require(setup.update(), "explicit SolveFieldOnly=False")
    checked = []
    for item in config["cameras"]:
        for body in camera.metadata["objects"]:
            name = item["id"] + "_" + body["name"]
            obj = app.modeler[name]
            expected = np.asarray(body["bounds_mm"]) + np.asarray(item["position_mm"])
            actual = np.asarray(obj.bounding_box).reshape(2, 3)
            if not np.allclose(actual, expected, atol=1e-5, rtol=1e-8):
                raise ValueError(f"Native camera coordinates differ from JSON: {name}")
            if not np.isclose(float(obj.volume), body["cad_volume_mm3"], rtol=1e-6):
                raise ValueError(f"Native STEP volume changed: {name}")
            if source:
                expected_material = body["material_id"]
                if obj.material_name.lower() != expected_material.lower():
                    raise ValueError(f"Native material assignment changed: {name}")
                actual_spec = material_spec(obj.material_name, dict(app.materials[obj.material_name]._props))
                expected_spec = source["materials"][expected_material]
                for key in ("law", "mu_r", "BH_H_A_per_m_B_T", "Hc_A_per_m", "Hc_direction_global", "remanence_A_per_m"):
                    if actual_spec.get(key) != expected_spec.get(key):
                        raise ValueError(f"Native material property differs from AEDT source: {name}/{key}")
            checked.append({"object": name, "bounds_mm": actual.tolist(), "volume_mm3": float(obj.volume)})
    require(app.save_project(), "audited project save")
    if app.validate_simple(str(output / "validation.log")) != 1:
        raise ValueError("Native audited project failed validation")
    native_text = project.read_text(encoding="utf-8", errors="replace")
    if source:
        from .aedt_source import audit_saved_materials
        write_json(output / "material_persistence_audit.json", audit_saved_materials(project, source))
    if "SolveFieldOnly=false" not in native_text or "SolveFieldOnly=true" in native_text:
        raise ValueError("Saved native setup does not preserve SolveFieldOnly=False")
    for item in config["cameras"]:
        cid = item["id"]
        block = native_text.split("$begin '" + cid + "_AmpereTurns'", 1)[1].split("$end", 1)[0]
        expected_current = item["current_A"] * item["turns"]
        actual_current = float(block.split("Current='", 1)[1].split("A'", 1)[0])
        if not np.isclose(actual_current, expected_current) or "IsSolid=false" not in block or "'Point out of terminal'=false" not in block:
            raise ValueError("Native current excitation differs from requested N*I/polarity")
        force = native_text.split("$begin 'Force_" + cid + "'", 1)[1].split("$end", 1)[0]
        if "'Is Virtual'=false" not in force:
            raise ValueError("Native force parameter is not Lorentz")
    write_json(output / "build_audit.json", {"project": str(project), "project_sha256": digest(project),
        "native_build_verified": True, "native_solved": False, "camera_count": config["N"],
        "checked_bodies": checked, "coordinates_and_volumes_checked": True,
        "ampere_turns_and_polarity_checked": True, "lorentz_force_parameters_checked": True,
        "aedt_source_materials_checked": source is not None,
        "previous_SolveFieldOnly": previous, "SolveFieldOnly": False, "setup": dict(setup.props)})


def native_worker(config_path, project, phase, output):
    config = load_config(config_path)
    project, output = Path(project).resolve(), Path(output).resolve()
    if phase == "build" and project.exists():
        raise FileExistsError(project)
    if phase in ("extract", "audit") and not project.exists():
        raise FileNotFoundError(project)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["ANSYSEM_ROOT222"] = str(AEDT.parent)
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    desktop, port = launch_owned_desktop(output)
    try:
        app = Maxwell3d(projectname=str(project), designname=DESIGN, solution_type="Magnetostatic",
                        specified_version="2022.2", non_graphical=True, new_desktop_session=False,
                        machine="localhost", port=port, aedt_process_id=desktop.pid, close_on_exit=False)
    except BaseException:
        if desktop.poll() is None:
            desktop.terminate()
            desktop.wait(timeout=10)
        raise
    write_json(output / "session.json", {"pid": app.desktop_class.aedt_process_id, "project": str(project)})
    try:
        if phase == "audit":
            audit_array_build(app, config, project, output)
            return
        if phase == "build":
            source = None
            if "aedt_source" in config:
                from copy import deepcopy
                from .aedt_source import load_source_camera, read_source
                source = read_source(config["aedt_source"])
                camera, _ = load_source_camera(config["aedt_source"], config["low_fidelity"]["cell_mm"])
                step_path = str(Path(config["aedt_source"]).parent / "template.step")
            elif "aedt" in config:
                raise ValueError("Freeze the AEDT source before native build")
            else:
                camera = load_camera(config["step"], config["low_fidelity"]["cell_mm"])
                step_path = config["step"]
            boxes = validate_layout(config, camera)
            if app.modeler.solid_names:
                raise ValueError("Native build requires an empty design")
            app.modeler.model_units = "mm"
            if source:
                assignments = {o["name"]: o["material_name"] for o in source["objects"]}
                for name, spec in source["materials"].items():
                    mat = require(app.materials.add_material(name, properties=deepcopy(spec["raw_properties"])), "native source material")
                    # Copy raw props after construction: old PyAEDT vector parsing
                    # strips the sign of Hc when it creates cached MatProperty values.
                    mat._props = deepcopy(spec["raw_properties"])
                    require(mat.update(), "preserve exact native source properties")
            else:
                materials = benchmark_materials()
                steel = require(app.materials.add_material("ArraySteel"), "steel material")
                curve = materials["yoke"]
                hs = np.unique(np.r_[np.linspace(0, 8000, 401), curve.h, [100000, 200000]])
                bs = curve.b_value(hs)
                steel.permeability.value = np.c_[hs, bs].tolist()
                steel.conductivity = 0
                require(steel.update(), "steel update")
                np.savetxt(output / "maxwell_bh.csv", np.c_[hs, bs], delimiter=",", header="H_A_per_m,B_T", comments="")
                assignments = {"YOKE_SOFT_MAGNETIC": steel.name, "AF_COIL_COPPER": "copper"}
                for name, direction in DIRECTIONS.items():
                    mat = require(app.materials.add_material("Array_" + name), "magnet material")
                    mat.permeability, mat.conductivity = 1.05, 0
                    require(mat.set_magnetic_coercivity(-1.2 / (MU0 * 1.05), *direction), "magnet remanence")
                    require(mat.update(), "magnet update")
                    assignments[name] = mat.name
            objects = {}
            for c in config["cameras"]:
                cid, pos = c["id"], np.asarray(c["position_mm"])
                require(app.modeler.import_3d_cad(step_path, create_group=False), "native-source geometry import")
                if not EXPECTED <= set(app.modeler.solid_names):
                    raise ValueError("Native STEP body names not preserved")
                objects[cid] = []
                for name in sorted(EXPECTED):
                    obj = app.modeler[name]
                    obj.name = cid + "_" + name
                    objects[cid].append(obj.name)
                require(app.modeler.move(objects[cid], pos.tolist()), "camera translation")
                for name, material in assignments.items():
                    require(app.assign_material(cid + "_" + name, material), "magnet assignment")
                coil_name = cid + "_AF_COIL_COPPER"
                app.modeler[coil_name].solve_inside = True
                spec = camera.metadata["coil"]
                cx, cy = np.asarray(spec["center_m"]) * 1e3
                iy, oy = spec["inner_half_m"][1] * 1e3, spec["outer_half_m"][1] * 1e3
                zlo, zhi = np.asarray(spec["z_m"]) * 1e3
                terminal = require(app.modeler.create_rectangle("YZ", (pos + [cx, cy + iy, zlo]).tolist(),
                                   [oy - iy, zhi - zlo], name=cid + "_Terminal"), "terminal sheet")
                require(app.assign_current(terminal.name, amplitude=c["current_A"] * c["turns"], solid=False,
                                          swap_direction=False, name=cid + "_AmpereTurns"), "stranded current")
                require(app.assign_force(coil_name, is_virtual=False, force_name="Force_" + cid), "Lorentz force")
            # Include distant B monitors in the physical air domain, without
            # creating physical monitor bodies or modifying imported solids.
            points = np.array([m["position_mm"] for m in config["monitors"] if m["quantity"] == "B"])
            lo, hi = np.min(np.array(boxes)[:, 0], axis=0), np.max(np.array(boxes)[:, 1], axis=0)
            if len(points):
                lo, hi = np.minimum(lo, points.min(axis=0)), np.maximum(hi, points.max(axis=0))
            air = config["maxwell"]["air_mm"]
            require(app.modeler.create_box((lo - air).tolist(), (hi - lo + 2 * air).tolist(),
                                          name="Region", material="vacuum"), "air domain")
            plan = {"policy": "adaptive", "operations": []}
            if config["maxwell"]["mesh_policy"] == "low-fidelity":
                with np.load(output / "low_fidelity_state.npz", allow_pickle=False) as state:
                    fields = state["b"]
                plan = apply_array_seeds(app, camera, config, fields)
            write_json(output / "mesh_plan.json", plan)
            setup = require(app.create_setup(SETUP, MaximumPasses=config["maxwell"]["max_passes"], MinimumPasses=2,
                MinimumConvergedPasses=2, PercentError=config["maxwell"]["percent_error"], PercentRefinement=30,
                NonLinearResidual=1e-4, SmoothBHCurve=False, SolveFieldOnly=False), "setup")
            expressions = sorted({"Force_" + m["camera_id"] + ".Force_" + axis for m in config["monitors"]
                                  if m["quantity"] == "F" for axis in ("xyz" if m["component"] == "magnitude" else m["component"])})
            if expressions:
                require(setup.enable_expression_cache(expressions, report_type="Magnetostatic", isconvergence=True,
                    isrelativeconvergence=False, conv_criteria=config["maxwell"]["force_delta_N"], use_cache_for_freq=False), "force convergence")
            require(app.save_project(str(project)), "pre-solve project save")
            write_json(output / "native_manifest.json", {"project": str(project), "design": DESIGN, "setup": dict(setup.props),
                "camera_objects": objects, "config_sha256": digest(config_path), "geometry_sha256": digest(step_path),
                "physics_source": camera.metadata.get("aedt_source", {"kind": "legacy STEP / benchmark materials", "br_T": 1.2, "recoil_mu_r": 1.05}), "mesh_plan": plan,
                "current_convention": "CCW +Z; stranded N*I; PointOut=False", "BH_coordinate_order": "H,B"})
            audit_array_build(app, config, project, output)
            return
        require(app.export_profile(SETUP, file_path=str(output / "profile.txt")), "profile export")
        require(app.export_convergence(SETUP, file_path=str(output / "convergence.txt")), "convergence export")
        require(app.export_mesh_stats(SETUP, mesh_path=str(output / "mesh_stats.txt")), "mesh stats export")
        profile = (output / "profile.txt").read_text(errors="replace")
        convergence = (output / "convergence.txt").read_text(errors="replace")
        status = "Normal Completion" if "Status: Normal Completion" in profile else "Aborted" if "Status: Aborted" in profile else "Other"
        result = {"fidelity": "Maxwell", "native_status": status,
                  "native_adaptive_converged": status == "Normal Completion" and "Adaptive Passes converged" in profile,
                  "monitor_precision_proven": False, "project": str(project), "config_sha256": digest(config_path)}
        write_json(output / "native_result.json", result)
        if status != "Normal Completion":
            raise RuntimeError("Native solve failed or was aborted; no optimization proposal accepted")
        b_monitors = [m for m in config["monitors"] if m["quantity"] == "B"]
        b_values = iter([])
        if b_monitors:
            points = np.array([m["position_mm"] for m in b_monitors])
            pts = output / "monitor_points_mm.pts"
            with pts.open("w", encoding="utf-8") as stream:
                stream.write("Unit=mm\n")
                np.savetxt(stream, points, fmt="%.12g")
            field = output / "monitor_B.fld"
            require(app.post.export_field_file("B", solution=SETUP + " : LastAdaptive", output_dir=str(field),
                        sample_points_file=str(pts), export_in_si_system=True), "monitor B field")
            data = np.loadtxt(field, skiprows=1, ndmin=2)
            if data.shape != (len(points), 6) or not np.isfinite(data).all() or not np.allclose(data[:, :3], points * 1e-3, atol=1e-10):
                raise ValueError("Native monitor field has invalid values or SI coordinates")
            b_values = iter(data[:, 3:])
        force_ids = sorted({m["camera_id"] for m in config["monitors"] if m["quantity"] == "F"})
        forces = {}
        if force_ids:
            quantities = ["Force_" + cid + ".Force_" + axis for cid in force_ids for axis in "xyz"]
            data = require(app.post.get_solution_data(expressions=quantities, setup_sweep_name=SETUP + " : LastAdaptive",
                                                     report_category="Magnetostatic"), "coil force reports")
            for cid in force_ids:
                value = []
                for axis in "xyz":
                    expr = "Force_" + cid + ".Force_" + axis
                    if data.units_data.get(expr) != "N":
                        raise ValueError("Native force report is not in N")
                    values = data.data_real(expr)
                    if len(values) != 1:
                        raise ValueError("Expected one nominal force value")
                    value.append(float(values[0]))
                forces[cid] = value
            write_json(output / "forces_N.json", forces)
        vectors = [next(b_values) if m["quantity"] == "B" else forces[m["camera_id"]] for m in config["monitors"]]
        monitors, score = monitor_values(config, vectors)
        result.update(monitors=monitors, score=score, native_verified=True,
                      accuracy_note="Native adaptive completion does not establish a strict monitor or force-difference error bound")
        write_json(output / "native_result.json", result)
    finally:
        try:
            require(app.save_project(), "final project save")
        finally:
            app.release_desktop(close_projects=True, close_desktop=True)


def snapshot_native(project, destination):
    """Freeze all native results before reopening the project for extraction."""
    project, destination = Path(project), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    results = project.with_suffix(".aedtresults")
    if not results.is_dir() or not any(results.rglob("*.asol")):
        raise FileNotFoundError("No native .asol/result tree to preserve")
    sources = [project, results]
    batch = Path(str(project) + ".batchinfo")
    if batch.exists():
        sources.append(batch)
    needed = sum(p.stat().st_size for root in sources for p in ([root] if root.is_file() else root.rglob("*")) if p.is_file())
    if shutil.disk_usage(project.parent).free < needed * 1.1:
        raise OSError("Insufficient disk for a complete pre-extraction native snapshot")
    destination.mkdir()
    hashes = {}
    for source in sources:
        target = destination / source.name
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        for p in ([source] if source.is_file() else source.rglob("*")):
            if p.is_file():
                relative = p.relative_to(project.parent)
                sha = digest(p)
                if digest(destination / relative) != sha:
                    raise IOError(f"Native snapshot hash mismatch: {p}")
                hashes[str(relative)] = sha
    write_json(destination / "sha256.json", hashes)
    return {"directory": str(destination), "files": len(hashes), "bytes": needed, "hashes_verified": True}


def simulation_gate(*, estimated_minutes, estimate_basis, budget_minutes, confirmed_long=False):
    # This is the whole serial campaign, not a per-job exception.
    if estimated_minutes is None or not np.isfinite(estimated_minutes) or estimated_minutes <= 0 or not estimate_basis:
        raise ValueError("Native solve requires a reliable whole-campaign estimate and its basis; uncertain >30 min needs user confirmation")
    if not np.isfinite(budget_minutes) or budget_minutes <= 0 or budget_minutes < estimated_minutes:
        raise ValueError("Simulation budget must cover the whole estimated campaign")
    if (estimated_minutes > 30 or budget_minutes > 30) and not confirmed_long:
        raise PermissionError("AGENTS.md requires explicit confirmation for this >30 min simulation campaign")
    return {"estimated_total_minutes": estimated_minutes, "estimate_basis": estimate_basis,
            "budget_minutes": budget_minutes, "explicit_long_confirmation": confirmed_long}


def run_stage(command, log, deadline, *, project=None, solve=False):
    """Reserve time for native abort; never relabel an aborted solve as success."""
    begin = perf_counter()
    if deadline - begin < 100:
        raise TimeoutError("Insufficient remaining campaign budget before native stage")
    item = {"command": command, "started_UTC": datetime.now(timezone.utc).isoformat(), "log": str(log)}
    with Path(log).open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        item["pid"] = process.pid
        write_json(Path(str(log) + ".json"), item)
        try:
            while process.poll() is None:
                if perf_counter() >= deadline - 90:
                    raise TimeoutError("Whole-campaign budget reached; no additional computation")
                sleep(.5)
        except BaseException as exc:
            item["stop_reason"] = repr(exc)
            if solve:
                abort = [str(AEDT.parent / "desktopproxy.exe"), "-abort", str(project)]
                try:
                    item["native_abort_returncode"] = subprocess.run(abort, stdout=stream, stderr=stream,
                        timeout=30, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
                    process.wait(timeout=45)
                except (OSError, subprocess.TimeoutExpired) as abort_error:
                    # Leave native solver files/state intact for manual recovery.
                    item["native_abort_failure"] = repr(abort_error)
                    item["process_may_still_be_running"] = process.poll() is None
            else:
                # The worker may be blocked in a native RPC. Its separately
                # recorded desktop belongs only to this stage/campaign.
                startup = Path(log).parent / "startup_record.json"
                if startup.exists():
                    import psutil
                    owned = json.loads(startup.read_text(encoding="utf-8"))
                    try:
                        desktop = psutil.Process(owned["pid"])
                        expected_port = str(owned["port"])
                        command_line = desktop.cmdline()
                        if expected_port in command_line and "-grpcsrv" in command_line:
                            desktop.terminate()
                            item["owned_desktop_terminated"] = owned["pid"]
                    except psutil.NoSuchProcess:
                        pass
                process.terminate()
                process.wait(timeout=10)
            raise
        finally:
            item.update(returncode=process.poll(), seconds=perf_counter() - begin)
            write_json(Path(str(log) + ".json"), item)
    if item["returncode"]:
        raise RuntimeError(f"Native stage failed; inspect {log}")
    return item


def native_iteration(config_path, output, deadline, *, build_only=False):
    output = Path(output).resolve()
    project = output / "CameraArray.aedt"
    worker = [sys.executable, str(ROOT / "scripts/run_camera_array_native.py"), "--config", str(config_path),
              "--project", str(project), "--output", str(output)]
    record = {"stages": {}, "project": str(project)}
    start = perf_counter()
    try:
        record["stages"]["build"] = run_stage(worker + ["--phase", "build"], output / "native_build.log", deadline)
        if build_only:
            record["status"] = "built_unsolved"
            return record
        config = load_config(config_path)
        command = [str(AEDT), "-ng", "-BatchSolve", DESIGN + ":Nominal:" + SETUP,
                   "-machinelist", f"list=localhost:1:{config['maxwell']['cores']}:60%:0", "-Monitor", str(project)]
        record["stages"]["solve"] = run_stage(command, output / "native_solve.log", deadline, project=project, solve=True)
        record["snapshot"] = snapshot_native(project, output / "pre_extract_snapshot")
        write_json(output / "native_record.json", record)
        record["stages"]["extract"] = run_stage(worker + ["--phase", "extract"], output / "native_extract.log", deadline)
        result = json.loads((output / "native_result.json").read_text(encoding="utf-8"))
        record["status"] = "completed" if result["native_adaptive_converged"] else "unconverged"
        return result
    except BaseException as exc:
        record.update(status="stopped", failure=repr(exc))
        raise
    finally:
        record["seconds"] = perf_counter() - start
        write_json(output / "native_record.json", record)
