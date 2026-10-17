"""Bounded native Maxwell loading/response pilot; not a two-camera accuracy claim.

The low-order external mode is a truncation of cached native neighbour fields.
It is deliberately used only to test loading and half/full responses: its known
representation error forbids treating it as a replacement for the neighbour.
Original projects and the frozen production predictor remain unchanged.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.refine_pair_difference_reference import digest, read, save, native_convergence_rows
from scripts.prepare_native_external_field_modes import monomial_field, expression
from camera_mag.geometry import load_camera
from camera_mag.response_library import read_vector_field

AEDT = Path(r"C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe")
SOURCE = ROOT / "outputs/maxwell_verified/Isolated_CCW"
MODE = ROOT / "outputs/delta_precision_20261003/external_modes/gap1_A_degree2.npz"
CASES = [("VacuumCheck", 1.), ("NaturalControl", 0.),
         ("PilotZero", 0.), ("PilotHalf", .5), ("PilotFull", 1.)]


def utc():
    return datetime.now(timezone.utc).isoformat()


def mode_data():
    with np.load(MODE) as data:
        return {key: data[key] for key in data.files}


def incident(points):
    mode = mode_data()
    q = (np.asarray(points)-mode["center_m"])/mode["scale_m"]
    return monomial_field(q, mode["terms"]) @ mode["coefficients"]


def field_expressions():
    mode = mode_data()
    return [expression(mode["terms"], mode["coefficients"], axis,
                       mode["center_m"], float(mode["scale_m"])) for axis in range(3)]


def require(value, action):
    if value is False or value is None:
        raise RuntimeError(action)
    return value


def assign_mode(app, amplitude, reverse=False):
    """Face components are in local UV, not global XYZ."""
    expressions = field_expressions()
    record = []
    for face in app.modeler["Region"].faces:
        normal = np.asarray(face.normal, dtype=float)
        axis = int(np.argmax(np.abs(normal)))
        if not np.allclose(np.abs(normal), np.eye(3)[axis], atol=1e-8):
            raise ValueError("Pilot requires an axis-aligned outer region")
        u = np.eye(3)[(axis+1)%3]
        v = np.cross(normal, u)
        v_axis = int(np.argmax(np.abs(v)))
        origin = np.asarray(face.center)
        component_u = f"({amplitude:.12g})*({expressions[(axis+1)%3]})"
        component_v = f"({amplitude*v[v_axis]:.12g})*({expressions[v_axis]})"
        boundary = require(app.assign_tangential_h_field([face.id],
            x_component_real=component_u, y_component_real=component_v,
            origin=[f"{x:.12g}mm" for x in origin],
            u_pos=[f"{x:.12g}mm" for x in origin+u],
            reverse=reverse, bound_name=f"IncidentFace_{axis}_{'pos' if normal[axis]>0 else 'neg'}"),
            "Tangential H boundary creation failed")
        record.append(dict(face_id=face.id, normal=normal.tolist(), u=u.tolist(), v=v.tolist(),
                           boundary=boundary.name, properties=dict(boundary.props)))
    if len(record)!=6:
        raise ValueError("Expected all six outer region faces")
    return record


def build(output):
    source_provenance = read(SOURCE/"provenance.json")
    source_project = Path(source_provenance["project"])
    project = output/"NativeResponsePilot.aedt"
    if project.exists():
        raise FileExistsError(project)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_project, project)
    os.environ["ANSYSEM_ROOT222"] = str(AEDT.parent)
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(project), designname=source_provenance["design"],
                    specified_version="2022.2", non_graphical=True,
                    new_desktop_session=True, close_on_exit=False)
    try:
        solids = [name for name in app.modeler.solid_names if name!="Region"]
        volumes = {name:app.modeler[name].volume for name in solids}
        # Object3d.bounding_box returned a stale [0,0,0,1,1,1] placeholder in
        # this PyAEDT/AEDT version. Planar native face centres give the bounds.
        face_centres = np.asarray([face.center for face in app.modeler["Region"].faces])
        region_bounds = np.r_[face_centres.min(axis=0), face_centres.max(axis=0)]
        if np.min(region_bounds[3:]-region_bounds[:3])<30:
            raise ValueError("Unexpected source-region dimensions")
        records = {}
        for design, amplitude in CASES[1:]:
            app.set_active_design(source_provenance["design"])
            require(app.duplicate_design(design), "Native design duplication failed")
            if app.design_name != design:
                raise RuntimeError("Unexpected duplicated design")
            setup = app.get_setup("Reference")
            setup.auto_update = False
            setup.props.pop("ExpressionCache", None)
            setup.props.update(MaximumPasses=1, MinimumPasses=1, MinimumConvergedPasses=1,
                               NonLinearResidual=1e-5, PercentError=.5)
            require(setup.add_mesh_link(source_provenance["design"],
                solution="Reference : LastAdaptive", project=str(source_project),
                force_source_to_solve=False, preserve_partner_solution=True,
                apply_mesh_operations=False), "Source mesh import failed")
            require(setup.update(), "Pilot setup update failed")
            boundaries = [] if design=="NaturalControl" else assign_mode(app, amplitude)
            after = {name:app.modeler[name].volume for name in solids}
            if volumes != after:
                raise ValueError("Physical geometry changed")
            folder = output/design
            folder.mkdir(exist_ok=True)
            require(app.validate_simple(str(folder/"validation.log"))==1,
                    "Native pilot validation failed")
            records[design] = dict(amplitude=amplitude, setup=dict(setup.props),
                boundaries=boundaries, physical_volumes_mm3=after,
                mode_is_exact_neighbour_field=False, mesh_source=str(source_project),
                uses_same_geometry_mesh_link=True)
        app.insert_design("VacuumCheck", solution_type="Magnetostatic")
        low, high = region_bounds[:3], region_bounds[3:]
        require(app.modeler.create_box(low.tolist(), (high-low).tolist(), name="Region",
                material="vacuum"), "Vacuum test region failed")
        boundaries = assign_mode(app, 1.)
        require(app.mesh.assign_length_mesh(["Region"], maximum_length="5mm",
                    maximum_elements=None, name="Vacuum_Loading_Check"), "Vacuum mesh failed")
        setup = require(app.create_setup("Reference"), "Vacuum setup failed")
        setup.auto_update = False
        setup.props.update(MaximumPasses=1, MinimumPasses=1, MinimumConvergedPasses=1)
        require(setup.update(), "Vacuum setup update failed")
        folder = output/"VacuumCheck"
        folder.mkdir(exist_ok=True)
        require(app.validate_simple(str(folder/"validation.log"))==1, "Vacuum validation failed")
        records["VacuumCheck"] = dict(boundaries=boundaries, setup=dict(setup.props), amplitude=1.)
        require(app.save_project(), "Pilot project save failed")
        save(output/"build.json", dict(project=str(project), designs=records,
            source_project=str(source_project), source_design=source_provenance["design"],
            region_bounds_mm=region_bounds.tolist(), mode_sha256=digest(MODE),
            mode_note="Degree-2 harmonic truncation only: 38.14% held-out field RMS error; not accepted for pair prediction",
            no_paired_force_fit=True))
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


def extract(output, design):
    os.environ["ANSYSEM_ROOT222"] = str(AEDT.parent)
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    project = output/"NativeResponsePilot.aedt"
    app = Maxwell3d(projectname=str(project), designname=design, specified_version="2022.2",
                   non_graphical=True, new_desktop_session=True, close_on_exit=False)
    try:
        folder = output/design
        require(app.export_profile("Reference", file_path=str(folder/"profile.txt")), "Profile export failed")
        require(app.export_convergence("Reference", file_path=str(folder/"convergence.txt")), "Convergence export failed")
        app.export_mesh_stats("Reference", mesh_path=str(folder/"mesh_stats.txt"))
        profile = (folder/"profile.txt").read_text()
        if "Status: Normal Completion" not in profile:
            raise RuntimeError("Pilot solve lacks normal completion")
        native = (project.with_suffix(".aedtresults")/(design+".asol")).read_text()
        rows = native_convergence_rows(native)
        if max(rows)!=1:
            raise ValueError("Pilot must retain exactly one completed adaptive pass")
        metadata = dict(native_status="Normal Completion", completed_passes=1,
                        native_adaptive_converged=False, fixed_pass_pilot=True, native=rows[1])
        metadata["native_reported_adaptive_convergence"] = "Adaptive Passes converged" in profile
        metadata["force_stability_checked"] = False
        camera = load_camera(ROOT/"models/B0500_scale_AF_VCM_EM_only.step", cell_mm=.7)
        with np.load(ROOT/"outputs/single_camera_reuse/isolated_library/quadrature.npz") as q:
            coil = q["coarse_points"]
            weights = q["coarse_weights"]
        shift = camera.extent[0]+.001
        air = np.concatenate([camera.centers+[shift,0,0], camera.centers-[shift,0,0]])
        points = np.concatenate([camera.centers, coil, air])
        pts = folder/"samples_mm.pts"
        with pts.open("w") as stream:
            stream.write("Unit=mm\n")
            np.savetxt(stream, points*1e3, fmt="%.12g")
        for quantity in ("B", "H"):
            path = folder/(quantity+".fld")
            require(app.post.export_field_file(quantity, solution="Reference : LastAdaptive",
                    output_dir=str(path), sample_points_file=str(pts), export_in_si_system=True),
                    "Pilot field export failed")
            read_vector_field(path, points)
        np.savez_compressed(folder/"sampling.npz", points_m=points, body_count=len(camera.centers),
            coil_count=len(coil), coil_weights_m3=weights, body_centers_m=camera.centers,
            body_volumes_m3=camera.volumes, material_ids=camera.material_ids)
        if design!="VacuumCheck":
            data = require(app.post.get_solution_data(expressions=["Force_A.Force_z"],
                setup_sweep_name="Reference : LastAdaptive", report_category="Magnetostatic"),
                "Native Lorentz force export failed")
            if data.units_data["Force_A.Force_z"] not in ("N", "newton"):
                raise ValueError("Unexpected force units")
            metadata["Fz_N"] = float(data.data_real("Force_A.Force_z")[0])
            coil_pts = folder/"coil_mm.pts"
            with coil_pts.open("w") as stream:
                stream.write("Unit=mm\n")
                np.savetxt(stream, coil*1e3, fmt="%.12g")
            require(app.post.export_field_file("J", solution="Reference : LastAdaptive",
                output_dir=str(folder/"J.fld"), sample_points_file=str(coil_pts),
                export_in_si_system=True), "Pilot J export failed")
            read_vector_field(folder/"J.fld", coil)
        else:
            h = read_vector_field(folder/"H.fld", points)
            expected = incident(points)
            rms = float(np.linalg.norm(h-expected)/np.linalg.norm(expected))
            metadata.update(vacuum_analytic_field_relative_RMS=rms,
                            loading_gate_met=rms<=.02)
        save(folder/"result.json", metadata)
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


def repair_vacuum(output):
    """Archive the invalid geometry and solve, then correct only the vacuum control."""
    archive = output/"failed_vacuum_geometry"
    if archive.exists():
        raise FileExistsError(archive)
    archive.mkdir()
    project = output/"NativeResponsePilot.aedt"
    shutil.copy2(project, archive/project.name)
    shutil.copytree(project.with_suffix(".aedtresults"), archive/project.with_suffix(".aedtresults").name)
    shutil.copytree(output/"VacuumCheck", archive/"VacuumCheck")
    shutil.copy2(output/"run_record.json", archive/"run_record.json")
    os.environ["ANSYSEM_ROOT222"] = str(AEDT.parent)
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(project), designname="PilotZero", specified_version="2022.2",
        non_graphical=True, new_desktop_session=True, close_on_exit=False)
    try:
        centres = np.asarray([face.center for face in app.modeler["Region"].faces])
        low, high = centres.min(axis=0), centres.max(axis=0)
        if np.min(high-low)<30:
            raise ValueError("Invalid corrected vacuum extent")
        app.set_active_design("VacuumCheck")
        for boundary in list(app.boundaries):
            require(boundary.delete(), "Vacuum boundary removal failed")
        require(app.modeler.delete("Region"), "Vacuum placeholder removal failed")
        require(app.modeler.create_box(low.tolist(), (high-low).tolist(), name="Region", material="vacuum"),
                "Corrected vacuum region failed")
        boundaries = assign_mode(app, 1.)
        require(app.validate_simple(str(output/"VacuumCheck/validation_repaired.log"))==1,
                "Corrected vacuum validation failed")
        require(app.save_project(), "Corrected vacuum save failed")
        save(output/"vacuum_repair.json", dict(UTC=utc(), preserved_failed_project=str(archive/project.name),
            reason="Stale Object3d bounding_box created a 1 mm cube; requested sample points were outside",
            corrected_region_bounds_mm=np.r_[low,high].tolist(), boundaries=boundaries))
        record = read(output/"run_record.json")
        record["stages"]["VacuumCheck_solve"]["requires_rerun"] = True
        record["stages"]["VacuumCheck_solve"]["repair"] = str(output/"vacuum_repair.json")
        save(output/"run_record.json", record)
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


def repair_vacuum_mesh(output):
    os.environ["ANSYSEM_ROOT222"] = str(AEDT.parent)
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(output/"NativeResponsePilot.aedt"), designname="VacuumCheck",
        specified_version="2022.2", non_graphical=True, new_desktop_session=True, close_on_exit=False)
    try:
        require(app.mesh.assign_length_mesh(["Region"], inside_selection=True,
            maximum_length="5mm", maximum_elements=None, name="Vacuum_Volume_Check"), "Vacuum volume mesh failed")
        require(app.save_project(), "Vacuum mesh save failed")
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


def repair_loading_orientation(output):
    archive = output/"failed_loading_orientation"
    attempt=1
    while archive.exists():
        attempt+=1
        archive=output/f"failed_loading_orientation_{attempt}"
    archive.mkdir()
    project = output/"NativeResponsePilot.aedt"
    shutil.copy2(project,archive/project.name)
    shutil.copytree(project.with_suffix(".aedtresults"),archive/project.with_suffix(".aedtresults").name)
    shutil.copytree(output/"VacuumCheck",archive/"VacuumCheck")
    shutil.copy2(output/"run_record.json",archive/"run_record.json")
    os.environ["ANSYSEM_ROOT222"] = str(AEDT.parent)
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api=True
    settings.enable_error_handler=False
    app=Maxwell3d(projectname=str(project),designname="VacuumCheck",specified_version="2022.2",
        non_graphical=True,new_desktop_session=True,close_on_exit=False)
    try:
        for name in ("VacuumCheck","PilotZero","PilotHalf","PilotFull"):
            app.set_active_design(name)
            for boundary in list(app.boundaries):
                if boundary.name.startswith("IncidentFace_"):
                    require(boundary.delete(),"Boundary direction removal failed")
            amplitude=dict(CASES)[name]
            assign_mode(app,amplitude,reverse=True)
        require(app.save_project(),"Boundary direction repair save failed")
        save(output/"loading_orientation_repair.json",dict(UTC=utc(),
            preserved_failed_project=str(archive/project.name), ReverseV=True,
            note="Test native face-V orientation using vacuum analytic-field gate before camera solves"))
        record=read(output/"run_record.json")
        for key in ("VacuumCheck_solve","VacuumCheck_extract"):
            record["stages"][key]["requires_rerun"]=True
            record["stages"][key]["repair"]=str(output/"loading_orientation_repair.json")
        save(output/"run_record.json",record)
    finally:
        app.release_desktop(close_projects=True,close_desktop=True)


def run(output, budget_minutes, resume=False):
    output.mkdir(parents=True, exist_ok=True)
    provenance = read(SOURCE/"provenance.json")
    source_project = Path(provenance["project"])
    files = [source_project, source_project.with_suffix(".aedtresults")/(provenance["design"]+".asol"),
             SOURCE/"forces.json", SOURCE/"provenance.json", MODE]
    before = {str(path):digest(path) for path in files}
    # 165,544-cell source final pass was about two minutes on two cores.
    # One pass, four cores, five designs including a small vacuum control.
    record = dict(started_UTC=utc(), expected_minutes=[15,25], budget_minutes=budget_minutes,
        budget_basis="Existing native 165,544-tetrahedron final pass profile, fixed one-pass pilot",
        scope="Native loading, boundary control, half/full response and outward incremental field; not pair accuracy acceptance",
        original_hashes=before, stages={}, no_paired_force_fit=True,
        production_predictor_modified=False, max_passes=1, cores=4)
    record_path = output/"run_record.json"
    elapsed_before_resume = 0.
    if record_path.exists():
        if not resume:
            raise FileExistsError(record_path)
        record = read(record_path)
        if before!=record["original_hashes"]:
            raise ValueError("Original source hashes changed")
        if any(item["returncode"] and not key.endswith("_extract")
               or "stop_request" in item for key,item in record["stages"].items()):
            raise RuntimeError("Resume never retries failed native solves, builds or budget stops")
        elapsed_before_resume = (datetime.now(timezone.utc)-datetime.fromisoformat(record["started_UTC"])).total_seconds()
        record.setdefault("prior_stops", []).append(dict(failure=record.pop("failure", None),
            ended_UTC=record.get("ended_UTC"), elapsed_seconds=record.get("elapsed_seconds"),
            resume_UTC=utc(), reason="Correct extraction call arguments; retain native solve and original failure"))
    start = perf_counter()
    deadline = start+budget_minutes*60-elapsed_before_resume
    save(record_path, record)
    project = output/"NativeResponsePilot.aedt"
    def stage(key, command, native=False):
        previous = record["stages"].get(key)
        if previous and previous["returncode"]==0 and "stop_request" not in previous and not previous.get("requires_rerun"):
            return
        if perf_counter()>deadline-90:
            raise TimeoutError("Whole-pilot budget exhausted before next stage")
        prior_attempts = previous.get("prior_attempts", [])+[previous] if previous else []
        filename = key+(f"_attempt{len(prior_attempts)+1}" if previous else "")+".log"
        item = dict(command=command, started_UTC=utc(), log=str(output/filename))
        if previous:
            item["prior_attempts"] = prior_attempts
        record["active"] = key
        record["active_stage"] = item
        stage_start = perf_counter()
        with Path(item["log"]).open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            item["pid"] = process.pid
            save(record_path, record)
            while process.poll() is None:
                if perf_counter()>deadline-90 and "stop_request" not in item:
                    item["stop_request"] = dict(UTC=utc(), reason="Whole-pilot time budget")
                    if native:
                        abort = [str(AEDT.parent/"desktopproxy.exe"), "-abort", str(project)]
                        item["stop_request"]["command"] = abort
                        with (output/(key+"_abort.log")).open("w") as log:
                            item["stop_request"]["returncode"] = subprocess.run(abort,
                                stdout=log, stderr=subprocess.STDOUT, timeout=60,
                                creationflags=subprocess.CREATE_NO_WINDOW).returncode
                    else:
                        process.terminate()
                    save(record_path, record)
                if perf_counter()>deadline+60:
                    raise TimeoutError("Native cleanup exceeded pilot time budget; no further stages launched")
                sleep(3)
            item.update(returncode=process.returncode, seconds=perf_counter()-stage_start, ended_UTC=utc())
        record["stages"][key] = item
        record.pop("active_stage", None)
        save(record_path, record)
        if item["returncode"] or "stop_request" in item:
            raise RuntimeError("Pilot stage failed/stopped; native status retained: "+key)
    try:
        stage("build", [sys.executable, str(Path(__file__).resolve()), "--phase", "build", "--output", str(output)])
        if (output/"vacuum_repair.json").exists():
            stage("vacuum_mesh_repair", [sys.executable, str(Path(__file__).resolve()),
                "--phase", "repair-vacuum-mesh", "--output", str(output)])
        for design, amplitude in CASES:
            stage(design+"_solve", [str(AEDT), "-ng", "-BatchSolve", design+":Nominal:Reference",
                "-machinelist", "list=localhost:1:4:60%:0", "-Monitor", str(project)], native=True)
            stage(design+"_extract", [sys.executable, str(Path(__file__).resolve()), "--phase", "extract",
                                    "--design", design, "--output", str(output)])
            if design=="VacuumCheck" and not read(output/design/"result.json")["loading_gate_met"]:
                raise RuntimeError("Vacuum loading gate failed; no camera perturbation solves submitted")
            if design=="VacuumCheck" and deadline-perf_counter()<12*60:
                record.update(active="Loading verified; camera response deferred at budget checkpoint",
                    camera_response_not_submitted=True,
                    budget_checkpoint_reason="Less than the estimated complete camera-response batch remains")
                break
        else:
            record["active"] = "Pilot complete"
    except Exception as exc:
        record.update(active="Stopped", failure=repr(exc))
        raise
    finally:
        record.update(ended_UTC=utc(), elapsed_seconds=perf_counter()-start+elapsed_before_resume,
                      original_sources_unchanged={path:digest(Path(path))==sha for path,sha in before.items()})
        save(record_path, record)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="outputs/native_response_pilot_20261003")
    parser.add_argument("--phase", choices=["run", "build", "extract", "repair-vacuum", "repair-vacuum-mesh", "repair-orientation"], default="run")
    parser.add_argument("--design")
    parser.add_argument("--budget-minutes", type=float, default=25)
    parser.add_argument("--resume", action="store_true", help="Repair extraction only; do not repeat native solves")
    args = parser.parse_args()
    if args.budget_minutes>25 or args.budget_minutes<=0:
        raise ValueError("This pilot is capped at 25 minutes; longer jobs require user confirmation")
    output = Path(args.output).resolve()
    if args.phase=="build":
        build(output)
    elif args.phase=="extract":
        extract(output,args.design)
    elif args.phase=="repair-vacuum":
        repair_vacuum(output)
    elif args.phase=="repair-vacuum-mesh":
        repair_vacuum_mesh(output)
    elif args.phase=="repair-orientation":
        repair_loading_orientation(output)
    else:
        run(output,args.budget_minutes,args.resume)


if __name__=="__main__":
    main()
