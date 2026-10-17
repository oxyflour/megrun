"""Refine separate same-geometry Maxwell references and audit distance differences.

Original references remain immutable mesh-link sources. Pass changes and changes
between refinement levels are stability diagnostics, never certified error bounds.
"""
import argparse
from datetime import datetime, timezone
import hashlib
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
AEDT = Path(r"C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe")
SOURCES = {
    1: ROOT / "outputs/mesh_seeding_onepass_holdouts/gap_1mm/TightReference",
    2: ROOT / "outputs/mesh_seeding_2mm/TightReference",
}
LEVELS = [(0.12, 2e-6, 20), (0.08, 1e-6, 15)]


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot(source):
    manifest = read(source / "manifest.json")
    project = Path(manifest["project"])
    files = [project, project.with_suffix(".aedtresults") / (manifest["design"] + ".asol")]
    files += [source / name for name in ("manifest.json", "forces.json", "force_adaptive_history.json",
              "profile.txt", "convergence.txt", "mesh_stats.txt", "maxwell_bh.csv", "provenance.json")]
    return {str(path): digest(path) for path in files}


def native_convergence_rows(text):
    """AEDT changes argument order in c(...) records between saved designs."""
    result = {}
    for body in re.findall(r"^\s*c\((.*)\)\s*$", text, flags=re.MULTILINE):
        row = {}
        for key in ("tet", "te", "ee", "p", "de"):
            match = re.search(r"(?:^|,\s*)" + key + r"=([^,\s()]+)", body)
            if match:
                row[key] = float(match[1])
        if all(key in row for key in ("tet", "te", "ee", "p")):
            result[int(row["p"])] = row
    return result


def wait_for_queue_checkpoint(target):
    """Hold a future build before opening AEDT; never interrupt a native solve.

    Build subprocesses read this file afresh, so the checkpoint also applies to
    an orchestration process that was started before queue control was added.
    """
    level_match = re.fullmatch(r"level_(\d+)", target.parent.parent.name)
    if not level_match:
        return
    level = int(level_match[1])
    output = target.parent.parent.parent
    control_path = output / "queue_control.json"
    checkpoint_path = output / "queue_checkpoint.json"
    held = False
    while control_path.exists():
        control = read(control_path)
        if not control.get("defer_enabled", False) or level < int(control["defer_from_level"]):
            break
        if not held:
            save(checkpoint_path, dict(state="Deferred before native build", UTC=utc(),
                level=level, target=str(target), reason=control["reason"],
                native_project_created=False, native_solve_started=False))
            print(f"Deferring level {level} before project creation: {control['reason']}", flush=True)
            held = True
        sleep(20)
    if held:
        save(checkpoint_path, dict(state="Released", UTC=utc(), level=level, target=str(target)))


def build(source, target, energy, force, refinement, max_passes):
    wait_for_queue_checkpoint(target)
    manifest = read(source / "manifest.json")
    if manifest.get("isolated") or int(manifest["distance_mm"]) not in SOURCES:
        raise ValueError("Requires an existing paired reference at its recorded geometry")
    if "Status: Normal Completion" not in (source / "profile.txt").read_text():
        raise ValueError("Mesh source must have a native normal completion")
    if target.exists():
        raise FileExistsError(target)
    source_project = Path(manifest["project"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_project, target)
    os.environ["ANSYSEM_ROOT222"] = str(AEDT.parent)
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(target), designname=manifest["design"], specified_version="2022.2",
                    non_graphical=True, new_desktop_session=True, close_on_exit=False)
    try:
        physical = list(app.modeler.model_objects)
        before = {name: app.modeler[name].volume for name in app.modeler.solid_names}
        if not app.duplicate_design("PrecisionReference") or app.design_name != "PrecisionReference":
            raise RuntimeError("Duplicate design failed")
        setup = app.get_setup("Reference")
        setup.props.pop("ExpressionCache", None)
        setup.props.update(MaximumPasses=max_passes, MinimumPasses=2, MinimumConvergedPasses=2,
                           PercentError=energy, PercentRefinement=refinement, NonLinearResidual=1e-5)
        if not setup.add_mesh_link(manifest["design"], solution="Reference : LastAdaptive",
                project=str(source_project), force_source_to_solve=False,
                preserve_partner_solution=True, apply_mesh_operations=False):
            raise RuntimeError("Same-geometry mesh link failed")
        if not setup.update() or not setup.enable_expression_cache(["Force_A.Force_z", "Force_B.Force_z"],
                report_type="Magnetostatic", isconvergence=True, isrelativeconvergence=False,
                conv_criteria=force, use_cache_for_freq=False):
            raise RuntimeError("Precision setup failed")
        after = {name: app.modeler[name].volume for name in app.modeler.solid_names}
        if before != after or physical != list(app.modeler.model_objects):
            raise ValueError("Physical geometry changed")
        folder = target.parent / "PrecisionReference"
        folder.mkdir(exist_ok=True)
        manifest.update(project=str(target), design="PrecisionReference", percent_error=energy,
                        force_delta_n=force, max_passes=max_passes, maximum_passes=max_passes,
                        minimum_passes=2, minimum_converged_passes=2,
                        percent_refinement=refinement, nonlinear_residual=1e-5,
                        mesh_source_project=str(source_project), mesh_source_design=read(source / "manifest.json")["design"],
                        same_geometry_mesh_reuse=True, geometry_volume_check={"before": before, "after": after},
                        reference_purpose="Distance-difference stability refinement; no force fitting")
        save(folder / "manifest.json", manifest)
        shutil.copy2(source / "maxwell_bh.csv", folder / "maxwell_bh.csv")
        if (source / "mesh_plan.json").exists():
            shutil.copy2(source / "mesh_plan.json", folder / "mesh_plan.json")
        if app.validate_simple(str(folder / "validation.log")) != 1 or not app.save_project():
            raise RuntimeError("Native validation/save failed")
        save(folder / "build_setup.json", dict(setup.props))
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


def audit(folder):
    manifest = read(folder / "manifest.json")
    history = read(folder / "force_adaptive_history.json")
    forces = read(folder / "forces.json")
    text = (folder / "convergence.txt").read_text()
    profile = (folder / "profile.txt").read_text()
    mesh = (folder / "mesh_stats.txt").read_text()
    native = (Path(manifest["project"]).with_suffix(".aedtresults") / (manifest["design"] + ".asol")).read_text()
    completed = int(re.search(r"Completed\s*:\s*(\d+)", text)[1])
    last_pass = int(re.search(r"LP\(VID=\d+, Ver=\d+, P=(\d+)\)", native)[1])
    tetrahedra = int(re.search(r"Total number of mesh elements:\s*(\d+)", mesh)[1])
    native_rows = native_convergence_rows(native)
    native_mesh = native_rows.get(last_pass)
    if (completed != last_pass or list(map(int, history["passes"])) != list(range(1, completed+1))
            or not native_mesh or int(native_mesh["tet"]) != tetrahedra):
        raise ValueError("Native pass, force history or mesh mismatch")
    result = dict(completed_passes=completed, tetrahedra=tetrahedra,
                  energy_error_percent=native_mesh["ee"], delta_energy_percent=native_mesh.get("de"),
                  native_status="Normal Completion" if "Status: Normal Completion" in profile else "Aborted" if "Status: Aborted" in profile else "Other",
                  adaptive_passes_converged="Adaptive Passes converged" in profile,
                  stability_diagnostics_are_error_bounds=False, Fz_N={}, last_two_Fz_changes_uN={})
    for label in "AB":
        expression = f"Force_{label}.Force_z"
        values = np.asarray(history["forces"][expression], dtype=float)
        force = forces[expression]
        if force["unit"] not in ("N", "newton") or not np.isclose(force["values"][0], values[-1], rtol=0, atol=1e-13):
            raise ValueError("LastAdaptive force/history mismatch")
        result["Fz_N"][label] = float(values[-1])
        result["last_two_Fz_changes_uN"][label] = (np.abs(np.diff(values))[-2:] * 1e6).tolist()
    force_stable = all(len(values) == 2 and max(values) <= manifest["force_delta_n"] * 1e6
                       for values in result["last_two_Fz_changes_uN"].values())
    energy_last_two = [native_rows.get(p) for p in (last_pass-1, last_pass)]
    energy_met = all(row is not None and "de" in row and max(row["ee"], row["de"]) <= manifest["percent_error"]
                     for row in energy_last_two)
    result["last_two_energy_conditions_met"] = bool(energy_met)
    result["configured_stopping_conditions_met"] = bool(force_stable and energy_met)
    result["native_adaptive_converged"] = bool(result["native_status"] == "Normal Completion"
                                              and result["adaptive_passes_converged"] and force_stable and energy_met)
    save(folder / "precision_audit.json", result)
    return result


def stage(record, key, command, folder, record_path, monitor_project=None, max_tetrahedra=2050000):
    if key in record["stages"]:
        if record["stages"][key]["returncode"] != 0 or "abort_request" in record["stages"][key]:
            raise RuntimeError("Prior failed stage retained; use a fresh output: " + key)
        return
    record.update(active=key, active_started_UTC=utc())
    save(record_path, record)
    start = perf_counter()
    item = dict(command=command, started_UTC=utc(), log=str(folder / (key + ".log")))
    print("Starting " + key, flush=True)
    with Path(item["log"]).open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        item["pid"] = process.pid
        record["active_process"] = item
        save(record_path, record)
        low_memory_samples = 0
        while process.poll() is None:
            if monitor_project:
                import psutil
                free_gib = psutil.virtual_memory().available / 2**30
                low_memory_samples = low_memory_samples + 1 if free_gib < 1.5 else 0
                metadata = monitor_project.with_suffix(".aedtresults") / "PrecisionReference.asol"
                count = 0
                if metadata.exists():
                    counts = re.findall(r"\btet=(\d+)", metadata.read_text(errors="replace"))
                    count = max(map(int, counts), default=0)
                for batch_log in monitor_project.with_suffix(".aedt.batchinfo").glob("*.log"):
                    counts = re.findall(r"(?:# Tetrahedra:\s*|Tetrahedra:\s*)(\d+)", batch_log.read_text(errors="replace"))
                    count = max(count, max(map(int, counts), default=0))
                record["live_monitor"] = dict(UTC=utc(), available_memory_GiB=free_gib,
                                              latest_completed_max_tetrahedra=count)
                if (count > max_tetrahedra or low_memory_samples >= 3) and "abort_request" not in item:
                    command_abort = [str(AEDT.parent / "desktopproxy.exe"), "-abort", str(monitor_project)]
                    item["abort_request"] = dict(command=command_abort, UTC=utc(),
                        reason="Resource guard", available_memory_GiB=free_gib, tetrahedra=count)
                    with (folder / (key + "_abort.log")).open("w") as abort_log:
                        abort = subprocess.run(command_abort, stdout=abort_log, stderr=subprocess.STDOUT,
                                               creationflags=subprocess.CREATE_NO_WINDOW)
                    item["abort_request"]["returncode"] = abort.returncode
                save(record_path, record)
            sleep(20)
        item.update(returncode=process.returncode, seconds=perf_counter() - start, ended_UTC=utc())
    record["stages"][key] = item
    record.pop("active_process", None)
    save(record_path, record)
    if item["returncode"] != 0 or "abort_request" in item:
        raise RuntimeError("Solve failed/stopped; original status retained: " + key)
    print(f"Finished {key}: {item['seconds']:.1f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="outputs/delta_precision_20261003")
    parser.add_argument("--levels", type=int, choices=[1, 2], default=2)
    parser.add_argument("--max-passes", type=int, default=3)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--target")
    parser.add_argument("--energy", type=float)
    parser.add_argument("--force", type=float)
    parser.add_argument("--refinement", type=int)
    args = parser.parse_args()
    if args.build:
        build(Path(args.source).resolve(), Path(args.target).resolve(), args.energy, args.force,
              args.refinement, args.max_passes)
        return
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "run_record.json"
    config = dict(levels=args.levels, max_passes=args.max_passes, cores=args.cores,
                  targets=LEVELS[:args.levels], max_tetrahedra=2050000)
    record = read(record_path) if record_path.exists() else dict(started_UTC=utc(), config=config,
        metric="Fz(1 mm) - Fz(2 mm), separately for A/B", original_sources={str(g):snapshot(s) for g,s in SOURCES.items()},
        stages={}, levels={}, force_fit_applied=False, mesh_reused_between_different_gaps=False)
    if json.dumps(record["config"], sort_keys=True) != json.dumps(config, sort_keys=True):
        raise ValueError("Existing configuration differs")
    save(record_path, record)
    source_folders = dict(SOURCES)
    for index, (energy, force, refinement) in enumerate(LEVELS[:args.levels], 1):
        level = record["levels"].setdefault(str(index), {"gaps": {}})
        for gap in (1, 2):
            folder = output / f"level_{index}" / f"gap_{gap}mm"
            folder.mkdir(parents=True, exist_ok=True)
            project = folder / "PrecisionReference.aedt"
            prefix = f"level{index}_gap{gap}"
            stage(record, prefix + "_build", [sys.executable, str(Path(__file__).resolve()), "--build",
                "--source", str(source_folders[gap]), "--target", str(project), "--energy", str(energy),
                "--force", str(force), "--refinement", str(refinement), "--max-passes", str(args.max_passes)], folder, record_path)
            stage(record, prefix + "_solve", [str(AEDT), "-ng", "-BatchSolve", "PrecisionReference:Nominal:Reference",
                "-machinelist", f"list=localhost:1:{args.cores}:60%:0", "-Monitor", str(project)], folder, record_path,
                monitor_project=project)
            stage(record, prefix + "_extract", [sys.executable, str(ROOT / "scripts/run_maxwell_case.py"),
                "--phase", "extract", "--project", str(project), "--design", "PrecisionReference", "--distance-mm", str(gap)], folder, record_path)
            level["gaps"][str(gap)] = audit(folder / "PrecisionReference")
            save(record_path, record)
        level["delta_Fz_uN"] = {label: (level["gaps"]["1"]["Fz_N"][label] - level["gaps"]["2"]["Fz_N"][label]) * 1e6 for label in "AB"}
        if index == 1:
            original = {g: read(s / "forces.json") for g,s in SOURCES.items()}
            previous = {label: (original[1][f"Force_{label}.Force_z"]["values"][0] - original[2][f"Force_{label}.Force_z"]["values"][0])*1e6 for label in "AB"}
        else:
            previous = record["levels"][str(index-1)]["delta_Fz_uN"]
        level["previous_delta_Fz_uN"] = previous
        level["delta_change_from_previous_level_uN"] = {label: level["delta_Fz_uN"][label]-previous[label] for label in "AB"}
        level["max_absolute_delta_change_uN"] = max(abs(v) for v in level["delta_change_from_previous_level_uN"].values())
        level["further_refinement_stability_2uN_met"] = index > 1 and level["max_absolute_delta_change_uN"] <= 2
        level["stability_is_strict_error_bound"] = False
        print(f"Level {index} delta: {level['delta_Fz_uN']}, change: {level['delta_change_from_previous_level_uN']}", flush=True)
        source_folders = {g: output / f"level_{index}" / f"gap_{g}mm" / "PrecisionReference" for g in (1,2)}
        for gap, source in SOURCES.items():
            if snapshot(source) != record["original_sources"][str(gap)]:
                raise ValueError("Original source metadata changed during mesh reuse")
        level["original_sources_unchanged"] = True
        save(record_path, record)
    record.update(active="Complete", completed_UTC=utc())
    save(record_path, record)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
