"""Export a completed isolated Maxwell field; performs no field solve or force fit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_mag.geometry import load_camera
from camera_mag.response_library import coil_quadrature, read_vector_field


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", default="outputs/maxwell_verified/Isolated_CCW")
    p.add_argument("--output", default="outputs/single_camera_reuse/isolated_library")
    p.add_argument("--gaps", type=float, nargs="+", default=[1, 2, 3])
    p.add_argument("--cell-mm", type=float, default=.7)
    p.add_argument("--frozen-pass", type=int, help="Reuse only a specifically audited retained experimental pass")
    args = p.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((source / "provenance.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    frozen = None
    if args.frozen_pass is not None:
        frozen = json.loads((source / "frozen_source_audit.json").read_text())
        if frozen["selected_completed_pass"] != args.frozen_pass or frozen["source"] != str(source):
            raise ValueError("Frozen source audit does not match requested pass")
        for path, expected in frozen["hashes"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen native source changed: "+path)
    if frozen is None and "Status: Normal Completion" not in (source / "profile.txt").read_text():
        raise ValueError("Source Maxwell solve has no normal completion")
    if not manifest.get("isolated"):
        raise ValueError("Source is not an isolated camera")
    convergence = re.search(r"Current\s*:\s*\(([^,]+),\s*([^)]+)\)", (source / "convergence.txt").read_text())
    if not convergence or max(map(float, convergence.groups())) > provenance["setup"]["PercentError"]:
        raise ValueError("Isolated Maxwell source has not met its energy convergence target")
    source_history = json.loads((source / "force_adaptive_history.json").read_text())["forces"]["Force_A.Force_z"]
    force_delta = manifest.get("force_delta_n")
    if frozen is None and force_delta is not None and (len(source_history)<3 or max(abs(np.diff(source_history[-3:])))>force_delta*1.001):
        raise ValueError("Isolated source has not met its last-two-step force stability target")
    for key in ("SI_positions_checked", "current_polarity_and_ampere_turns_checked", "BH_field_median_error_below_5_percent"):
        if provenance.get(key) is not True:
            raise ValueError(f"Unverified isolated source: {key}")
    step = ROOT / "models/B0500_scale_AF_VCM_EM_only.step"
    step_hash = hashlib.sha256(step.read_bytes()).hexdigest()
    if step_hash != provenance["step_sha256"]:
        raise ValueError("STEP differs from isolated source")
    camera = load_camera(step, cell_mm=args.cell_mm)
    levels = [["coarse", 32, 4, 16], ["fine", 64, 6, 24]]
    blocks = []
    quadratures = {}
    for label, segments, radial, height in levels:
        xyz, weights, ideal_j = coil_quadrature(camera.metadata["coil"], segments, radial, height)
        quadratures[label] = dict(points=xyz, weights=weights, ideal_j=ideal_j)
        blocks.append(xyz)
    own_points = np.concatenate(blocks)
    np.savez_compressed(output / "quadrature.npz", **{
        label + "_" + name: value for label, quad in quadratures.items() for name, value in quad.items()})
    specification = dict(source=str(source), project=provenance["project"], design=provenance["design"],
                         step_sha256=step_hash, gaps_mm=args.gaps, levels=levels,
                         current_a=manifest["current_a"], turns=manifest["turns"],
                         cell_mm=args.cell_mm,
                         geometry_extent_m=camera.extent.tolist(),
                         source_provenance_sha256=hashlib.sha256((source / "provenance.json").read_bytes()).hexdigest(),
                         source_BH_sha256=hashlib.sha256((source / "maxwell_bh.csv").read_bytes()).hexdigest(),
                         prediction_method="Isolated Maxwell J times translated isolated Maxwell B; no paired force calibration")
    if frozen is not None:
        specification["frozen_source_audit_sha256"] = hashlib.sha256((source / "frozen_source_audit.json").read_bytes()).hexdigest()
    record_path = output / "export_record.json"
    previous = json.loads(record_path.read_text()) if record_path.exists() else None
    if previous:
        previous["specification"].setdefault("cell_mm", .7)
    if previous and previous["specification"] != specification:
        raise ValueError("Existing export uses a different specification")
    record = previous or dict(specification=specification, exports={})
    os.environ["ANSYSEM_ROOT222"] = r"C:\Program Files\AnsysEM\v222\Win64"
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    start = perf_counter()
    app = Maxwell3d(projectname=provenance["project"], designname=provenance["design"],
                    specified_version="2022.2", non_graphical=True, new_desktop_session=True,
                    close_on_exit=False)
    try:
        model_solids = set(app.modeler.solid_names) & set(app.modeler.model_objects)
        if model_solids != set(manifest["module_objects"]["A"]) | {"Region"}:
            raise ValueError("Unexpected solids in single-camera source")
        # Query the native model box; old PyAEDT SAT bounding-box parsing can
        # return inconsistent units for a solved Region in AEDT 2022 R2.
        bounds_mm = np.asarray(app.modeler.get_model_bounding_box(), dtype=float).reshape(2, 3)
        print(f"MODEL_UNITS: {app.modeler.model_units}; AIR_BOUNDS_MM: {bounds_mm.tolist()}", flush=True)
        if app.modeler.model_units != "mm":
            raise ValueError("Expected a millimetre source model")

        def export(name, quantity, points):
            path = output / (name + ".fld")
            if name in record["exports"]:
                expected_hash = record["exports"][name]["sha256"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                    raise ValueError(f"Cached field hash changed: {name}")
                read_vector_field(path, points)
                return
            if np.any(points * 1e3 <= bounds_mm[0]) or np.any(points * 1e3 >= bounds_mm[1]):
                raise ValueError("Field query outside solved air region")
            pts = output / (name + "_mm.pts")
            with pts.open("w") as stream:
                stream.write("Unit=mm\n")
                np.savetxt(stream, points * 1e3, fmt="%.12g")
            begin = perf_counter()
            if not app.post.export_field_file(quantity, solution="Reference : LastAdaptive", output_dir=str(path),
                        sample_points_file=str(pts), export_in_si_system=True):
                raise RuntimeError(f"Maxwell export failed: {name}")
            read_vector_field(path, points)
            record["exports"][name] = dict(quantity=quantity, points=len(points), seconds=perf_counter()-begin,
                SI_positions_checked=True, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"EXPORTED {name}: {len(points)} points", flush=True)

        export("own_J", "J", own_points)
        export("own_B", "B", own_points)
        # Own H/B at physical coarse cells support a later incremental-response experiment.
        export("own_cell_H", "H", camera.centers)
        export("own_cell_B", "B", camera.centers)
        for gap in args.gaps:
            shift = np.array([camera.extent[0] + gap * 1e-3, 0, 0])
            for label, displacement in [("A", -shift), ("B", shift)]:
                export(f"gap_{gap:g}_{label}_incident_B", "B", own_points + displacement)
                export(f"gap_{gap:g}_{label}_cell_incident_H", "H", camera.centers + displacement)
        # Regular exterior-air field library permits new gaps without opening
        # Maxwell again. Both domains are entirely outside the physical camera.
        grid_axes = dict(xneg=np.linspace(-.0166, -.005, 59), xpos=np.linspace(.005, .0166, 59),
                         y=np.linspace(-.0046, .0046, 47), z=np.linspace(.0012, .005, 20))
        grids = [np.stack(np.meshgrid(grid_axes[key], grid_axes["y"], grid_axes["z"], indexing="ij"), -1).reshape(-1, 3)
                 for key in ("xneg", "xpos")]
        grid_points = np.concatenate(grids)
        export("exterior_grid_B", "B", grid_points)
        grid_b = read_vector_field(output / "exterior_grid_B.fld", grid_points)
        grid_shape = (59, 47, 20, 3)
        half = len(grids[0])
        np.savez_compressed(output / "exterior_grid.npz", **grid_axes,
                            Bneg=grid_b[:half].reshape(grid_shape), Bpos=grid_b[half:].reshape(grid_shape))
        record["exterior_grid_npz_sha256"] = hashlib.sha256((output / "exterior_grid.npz").read_bytes()).hexdigest()
        record["source_force_file_sha256"] = hashlib.sha256((source / "forces.json").read_bytes()).hexdigest()
        record.update(completed_UTC=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=perf_counter()-start, native_field_solve_performed=False,
                      source_energy_error_percent=float(convergence[1]),
                      source_delta_energy_percent=float(convergence[2]),
                      source_passes=len(source_history),
                      source_last_pass_Fz_change_uN=abs(np.diff(source_history)[-1])*1e6)
        if frozen is not None:
            record["native_adaptive_converged"] = False
            record["experimental_source_status"] = frozen["source_status"]
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    finally:
        # No save: the completed source physics and solution remain unchanged.
        app.release_desktop(close_projects=True, close_desktop=True)
    print(json.dumps({"completed": True, "elapsed_seconds": record["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
