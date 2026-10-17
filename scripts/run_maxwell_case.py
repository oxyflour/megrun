"""Build and solve an independent AEDT 2022 R2 coil-force reference."""
import argparse
import json
import os
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_mag.geometry import load_camera, EXPECTED, DIRECTIONS
from camera_mag.materials import benchmark_materials
from camera_mag.interaction import MU0


def require(value, action):
    if value is False or value is None:
        raise RuntimeError(f"AEDT failed: {action}")
    return value


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--step", default="models/B0500_scale_AF_VCM_EM_only.step")
    p.add_argument("--distance-mm", type=float, default=2)
    p.add_argument("--isolated", action="store_true")
    p.add_argument("--design", default="Gap_2mm")
    p.add_argument("--project", default="outputs/maxwell_verified/camera_reference.aedt")
    p.add_argument("--phase", choices=["build", "solve", "extract", "all"], default="all")
    p.add_argument("--mesh-mm", type=float, default=.3)
    p.add_argument("--mesh-policy", choices=["uniform", "adaptive", "geometry", "low-fidelity", "low-fidelity-aggressive", "low-fidelity-local"], default="uniform")
    p.add_argument("--generate-initial-mesh", action="store_true")
    p.add_argument("--initial-mesh-limit", type=int, default=600000)
    p.add_argument("--minimum-passes", type=int, default=3)
    p.add_argument("--minimum-converged-passes", type=int, default=2)
    p.add_argument("--force-delta-n", type=float, help="Additional absolute per-pass Fz convergence, for each coil")
    p.add_argument("--air-mm", type=float, default=20)
    p.add_argument("--percent-error", type=float, default=1)
    p.add_argument("--max-passes", type=int, default=12)
    p.add_argument("--current-a", type=float, default=.1)
    p.add_argument("--turns", type=int, default=100)
    p.add_argument("--pid", type=int)
    args = p.parse_args()
    if args.isolated and args.mesh_policy.startswith("low-fidelity"):
        p.error("low-fidelity mesh policy currently requires the two-camera benchmark")
    os.environ["ANSYSEM_ROOT222"] = r"C:\Program Files\AnsysEM\v222\Win64"
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    project = Path(args.project).resolve()
    folder = project.parent / args.design
    folder.mkdir(parents=True, exist_ok=True)
    pid = args.pid
    if pid is not None:
        import psutil
        if not psutil.pid_exists(pid):
            pid = None
    app = Maxwell3d(projectname=str(project), designname=args.design, solution_type="Magnetostatic",
                   specified_version="2022.2", non_graphical=True, new_desktop_session=pid is None,
                   aedt_process_id=pid, close_on_exit=False)
    (project.parent / "session.json").write_text(json.dumps({"pid": app.desktop_class.aedt_process_id,
                       "project": str(project), "version": "2022.2"}, indent=2))
    try:
        if args.phase in ("build", "all"):
            if app.modeler.solid_names:
                raise RuntimeError("Design is not empty; choose another --design or use --phase solve")
            app.modeler.model_units = "mm"
            camera = load_camera(args.step)
            shift = camera.extent[0] * 1e3 + args.distance_mm
            module_objects = {}
            for label in ("A",) if args.isolated else ("A", "B"):
                require(app.modeler.import_3d_cad(str(Path(args.step).resolve()), create_group=False), "STEP import")
                available = set(app.modeler.solid_names)
                print("Imported solids:", sorted(available), flush=True)
                if not EXPECTED <= available:
                    raise RuntimeError(f"STEP names not preserved: missing {EXPECTED - available}")
                module_objects[label] = []
                for name in sorted(EXPECTED):
                    obj = app.modeler[name]
                    obj.name = label + "_" + name
                    module_objects[label].append(obj.name)
                if label == "B":
                    require(app.modeler.move(module_objects[label], [shift, 0, 0]), "module B translation")
            materials = benchmark_materials()
            steel = app.materials.add_material("Magrun_Steel_BH")
            curve = materials["yoke"]
            # Dense samples reproduce the PCHIP law in the operating range.
            # Beyond 8 kA/m retain the original table chord, ending with μ0 slope.
            hs = np.unique(np.r_[np.linspace(0, 8000, 401), curve.h, [100000, 200000]])
            bs = curve.b_value(hs)
            tail = (hs > 8000) & (hs < 20000)
            bs[tail] = np.interp(hs[tail], curve.h, curve.b)
            # AEDT 2022 R2 consumes coordinate pairs as [H, B]. The older
            # PyAEDT 0.8.11 docstring shows [B,H]; field checks exposed that
            # this inverts the actual law in this installed version.
            steel.permeability.value = np.c_[hs, bs].tolist()
            steel.conductivity = 0
            require(steel.update(), "B-H material")
            np.savetxt(folder / "maxwell_bh.csv", np.c_[hs, bs], delimiter=",", header="H_A_per_m,B_T", comments="")
            magnet_materials = {}
            for name, direction in DIRECTIONS.items():
                mat = app.materials.add_material("Magrun_" + name)
                mat.permeability = 1.05
                mat.conductivity = 0
                # B = μ0 μr (H - Hc), matching remanence Br=1.2 T.
                require(mat.set_magnetic_coercivity(-1.2 / (MU0 * 1.05), *direction), "magnet coercivity")
                require(mat.update(), "magnet material")
                magnet_materials[name] = mat.name
            for label, objects in module_objects.items():
                require(app.assign_material(label + "_YOKE_SOFT_MAGNETIC", steel.name), "steel assignment")
                for name, material in magnet_materials.items():
                    require(app.assign_material(label + "_" + name, material), "magnet assignment")
                coil_name = label + "_AF_COIL_COPPER"
                require(app.assign_material(coil_name, "copper"), "copper assignment")
                app.modeler[coil_name].solve_inside = True
                x = 0 if label == "A" else shift
                spec = camera.metadata["coil"]
                iy, oy = spec["inner_half_m"][1] * 1e3, spec["outer_half_m"][1] * 1e3
                zlo, zhi = np.array(spec["z_m"]) * 1e3
                terminal = require(app.modeler.create_rectangle("YZ", [x, iy, zlo], [oy - iy, zhi - zlo],
                                   name=label + "_Terminal"), "terminal sheet")
                print("Terminal normal:", terminal.faces[0].normal, flush=True)
                # 2022 R2 magnetostatic uses a stranded Current sheet. Its bulk
                # current is N*I, equivalent to the homogenized 100-turn winding.
                # A field export verified PointOut=False gives -X at the +Y bar.
                require(app.assign_current(terminal.name, amplitude=args.current_a * args.turns,
                        solid=False, swap_direction=False, name=label + "_AmpereTurns"), "stranded current")
                require(app.assign_force(coil_name, is_virtual=False, force_name="Force_" + label), "Lorentz force parameter")
            region = require(app.modeler.create_region(args.air_mm, pad_type="Absolute Offset", name="Region"), "air region")
            require(app.assign_material(region.name, "vacuum"), "air material")
            all_solids = [name for objects in module_objects.values() for name in objects]
            if args.mesh_policy == "uniform":
                require(app.mesh.assign_length_mesh(all_solids, maximum_length=f"{args.mesh_mm}mm",
                                                   maximum_elements=None, name="Reference_Length"), "reference mesh")
                mesh_plan = {"policy": "uniform", "maximum_length_mm": args.mesh_mm}
            else:
                from camera_mag.mesh_seeding import apply_mesh_policy
                mesh_plan = apply_mesh_policy(app, camera, args.mesh_policy, args.distance_mm,
                                              args.current_a, args.turns)
            (folder / "mesh_plan.json").write_text(json.dumps(mesh_plan, indent=2), encoding="utf-8")
            setup = require(app.create_setup("Reference", MaximumPasses=args.max_passes, MinimumPasses=args.minimum_passes,
                        MinimumConvergedPasses=args.minimum_converged_passes, PercentError=args.percent_error, PercentRefinement=30,
                        NonLinearResidual=1e-4, SmoothBHCurve=False), "magnetostatic setup")
            if args.force_delta_n is not None:
                if args.force_delta_n <= 0:
                    raise ValueError("Force delta must be positive")
                require(setup.enable_expression_cache(["Force_" + label + ".Force_z" for label in module_objects],
                        report_type="Magnetostatic", isconvergence=True, isrelativeconvergence=False,
                        conv_criteria=args.force_delta_n, use_cache_for_freq=False), "force convergence cache")
            require(app.save_project(str(project)), "project save")
            manifest = vars(args) | {"project": str(project), "aedt_version": "2022.2", "center_offset_mm": shift,
                       "module_objects": module_objects, "force_type": "Lorentz", "br_T": 1.2,
                       "mu_r_recoil": 1.05, "winding": "stranded bulk current N*I=10 A; PointOut=False gives -X at +Y (field-verified)",
                       "bh_alignment": "PCHIP sampled below 8000 A/m; original high-field chord and μ0 extrapolation"}
            manifest["bh_coordinate_order"] = "H_A_per_m, B_T (verified against exported constitutive fields)"
            manifest["mesh_plan"] = mesh_plan
            (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
            valid = app.validate_simple(str(folder / "validation.log"))
            print("VALIDATION:", valid, flush=True)
            if valid != 1:
                raise RuntimeError(f"Maxwell validation failed ({valid}); inspect validation.log")
            if args.generate_initial_mesh:
                require(app.mesh.generate_mesh("Reference"), "initial mesh generation")
                app.export_mesh_stats("Reference", mesh_path=str(folder / "initial_mesh_stats.txt"))
                import re
                mesh_text = (folder / "initial_mesh_stats.txt").read_text()
                count = int(re.search(r"Total number of mesh elements:\s*(\d+)", mesh_text)[1])
                require(app.save_project(), "initial mesh save")
                print(f"INITIAL_TETRAHEDRA: {count}", flush=True)
                if count > args.initial_mesh_limit:
                    raise RuntimeError(f"Initial mesh {count} exceeds preflight limit {args.initial_mesh_limit}")
        if args.phase in ("solve", "all"):
            start = perf_counter()
            require(app.analyze_setup("Reference", cores=2), "reference solve")
            seconds = perf_counter() - start
            app.save_project()
            (folder / "solve_runtime.json").write_text(json.dumps({"seconds": seconds}, indent=2))
        if args.phase in ("solve", "extract", "all"):
            app.export_convergence("Reference", file_path=str(folder / "convergence.txt"))
            app.export_mesh_stats("Reference", mesh_path=str(folder / "mesh_stats.txt"))
            app.export_profile("Reference", file_path=str(folder / "profile.txt"))
            print("REPORT TYPES:", app.post.available_report_types, flush=True)
            quantities = app.post.available_report_quantities(report_category="Magnetostatic", solution="Reference : LastAdaptive")
            print("REPORT QUANTITIES:", quantities, flush=True)
            (folder / "quantities.json").write_text(json.dumps(quantities, indent=2))
            force_expressions = [q for q in quantities if "Force_" in q]
            if not force_expressions:
                raise RuntimeError("Maxwell has no solved force report quantities")
            data = require(app.post.get_solution_data(expressions=force_expressions,
                         setup_sweep_name="Reference : LastAdaptive", report_category="Magnetostatic"), "force data")
            force_values = {q: {"values": data.data_real(q), "unit": data.units_data.get(q)} for q in force_expressions}
            (folder / "forces.json").write_text(json.dumps(force_values, indent=2))
            data.export_data_to_csv(str(folder / "forces.csv"))
            history = app.post.get_solution_data(expressions=force_expressions,
                      setup_sweep_name="Reference : AdaptivePass", report_category="Magnetostatic",
                      primary_sweep_variable="Pass", variations={"Pass": ["All"]})
            if history:
                history.export_data_to_csv(str(folder / "force_adaptive_history.csv"))
                (folder / "force_adaptive_history.json").write_text(json.dumps({"passes": history.primary_sweep_values,
                    "forces": {q: history.data_real(q) for q in force_expressions}}, indent=2))
            print("FORCES:", json.dumps(force_values), flush=True)
            points = [[0, 2.975, 2.85], [2.975, 0, 2.85]]
            require(app.post.export_field_file("J", solution="Reference : LastAdaptive", output_dir=str(folder / "current_samples.fld"),
                        sample_points=points, export_in_si_system=True), "current direction check")
            currents = np.loadtxt(folder / "current_samples.fld", skiprows=1)
            if not np.allclose(currents[:, :3], np.asarray(points) * 1e-3, atol=1e-10):
                raise RuntimeError("Current sampling coordinates have wrong units")
            if currents[0, 3] >= 0 or currents[1, 4] <= 0:
                raise RuntimeError("Current direction differs from counterclockwise coarse winding")
            require(app.post.export_field_file("B", solution="Reference : LastAdaptive", output_dir=str(folder / "coil_B_samples.fld"),
                        sample_points=points, export_in_si_system=True), "coil field samples")
            camera = load_camera(args.step)
            coil = camera.metadata["coil"]
            area = (coil["outer_half_m"][1] - coil["inner_half_m"][1]) * np.diff(coil["z_m"])[0]
            ampere_turns = np.array([-currents[0, 3], currents[1, 4]]) * area
            if not np.allclose(ampere_turns, args.current_a * args.turns, rtol=.01):
                raise RuntimeError("Exported current density does not match requested ampere-turns")
            (folder / "current_check.json").write_text(json.dumps({"direction": "CCW from +Z",
                   "sample_positions_m": currents[:, :3].tolist(),
                   "bar_current_density_A_per_m2": currents[:, 3:].tolist(),
                   "cross_section_m2": area, "ampere_turns_from_J": ampere_turns.tolist()}, indent=2))
            shift = camera.extent[0] + args.distance_mm * 1e-3
            centers = camera.centers if args.isolated else np.r_[camera.centers, camera.centers + [shift, 0, 0]]
            # AEDT 2022 R2 reads point-file numbers in model units even with a
            # Unit=m header. Use mm explicitly, and verify exported SI positions.
            pts = folder / "coarse_centers_mm.pts"
            with pts.open("w") as stream:
                stream.write("Unit=mm\n")
                np.savetxt(stream, centers * 1e3, fmt="%.12g")
            for quantity in ("B", "H"):
                field_path = folder / (quantity + "_coarse_centers.fld")
                require(app.post.export_field_file(quantity, solution="Reference : LastAdaptive",
                            output_dir=str(field_path),
                            sample_points_file=str(pts), export_in_si_system=True), "coarse center field samples")
                exported = np.loadtxt(field_path, skiprows=1)
                if (exported.shape != (len(centers), 6) or not np.isfinite(exported).all()
                        or not np.allclose(exported[:, :3], centers, atol=1e-10)):
                    raise RuntimeError("Maxwell field sampling coordinates do not match SI camera centers")
            bfield = np.loadtxt(folder / "B_coarse_centers.fld", skiprows=1)[:, 3:]
            hfield = np.loadtxt(folder / "H_coarse_centers.fld", skiprows=1)[:, 3:]
            ids = camera.material_ids if args.isolated else np.tile(camera.material_ids, 2)
            steel_mask = ids == "yoke"
            hnorm = np.linalg.norm(hfield[steel_mask], axis=1)
            bnorm = np.linalg.norm(bfield[steel_mask], axis=1)
            expected_b = benchmark_materials()["yoke"].b_value(hnorm)
            bh_relative_error = np.abs(bnorm - expected_b) / np.maximum(expected_b, 1e-9)
            diagnostics = {"H_A_per_m_quantiles": np.quantile(hnorm, [0, .5, .95, 1]).tolist(),
                           "B_T_quantiles": np.quantile(bnorm, [0, .5, .95, 1]).tolist(),
                           "relative_BH_error_quantiles": np.quantile(bh_relative_error, [0, .5, .95, 1]).tolist(),
                           "sample_coordinates_checked_in_SI": True}
            (folder / "constitutive_check.json").write_text(json.dumps(diagnostics, indent=2))
            if np.median(bh_relative_error) > .05:
                raise RuntimeError("Exported Maxwell B-H fields do not follow specified steel curve")
            # AEDT non-graphical previews can be empty; do not publish one.
            preview = folder / "geometry.jpg"
            if preview.exists() and preview.stat().st_size == 0:
                preview.unlink()
            (folder / "provenance.json").write_text(json.dumps({"project": str(project),
                    "design": app.design_name, "version": app.odesktop.GetVersion(),
                    "extracted_at_UTC": datetime.now(timezone.utc).isoformat(),
                    "step_sha256": hashlib.sha256(Path(args.step).read_bytes()).hexdigest(),
                    "SI_positions_checked": True, "current_polarity_and_ampere_turns_checked": True,
                    "BH_field_median_error_below_5_percent": True,
                    "physical_model_objects": app.modeler.model_objects,
                    "non_model_objects": app.modeler.non_model_objects,
                    "object_ids_by_name": {name: app.modeler[name].id for name in app.modeler.object_names},
                    "object_materials": {name: app.modeler[name].material_name for name in app.modeler.solid_names},
                    "mesh_operations": {op.name: dict(op.props) for op in app.mesh.meshoperations},
                    "setup": dict(app.get_setup("Reference").props)}, indent=2, default=str))
        return 0
    finally:
        app.save_project()
        app.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    raise SystemExit(main())
