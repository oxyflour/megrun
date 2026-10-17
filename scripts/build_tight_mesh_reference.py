"""Continue refinement from a completed same-geometry Maxwell mesh."""
import argparse
import json
import os
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("source")
    parser.add_argument("--design", required=True)
    parser.add_argument("--destination", default="outputs/mesh_seeding_2mm/TightReference.aedt")
    parser.add_argument("--percent-error", type=float, default=.25)
    args = parser.parse_args()
    source, target = Path(args.source).resolve(), Path(args.destination).resolve()
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(source, target)
    os.environ["ANSYSEM_ROOT222"] = r"C:\Program Files\AnsysEM\v222\Win64"
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(target), designname=args.design, specified_version="2022.2",
                    non_graphical=True, new_desktop_session=True, close_on_exit=False)
    try:
        if not app.duplicate_design("TightReference"):
            raise RuntimeError("Failed to duplicate design")
        setup = app.get_setup("Reference")
        # Remove parsed cache items before any setup update. AEDT reads their
        # report type back as a numeric ID; rebuild the cache through the API.
        setup.props.pop("ExpressionCache", None)
        setup.props["MaximumPasses"] = 15
        setup.props["MinimumPasses"] = 2
        setup.props["MinimumConvergedPasses"] = 2
        setup.props["PercentError"] = args.percent_error
        if not setup.add_mesh_link(args.design, solution="Reference : LastAdaptive", project=str(source),
                force_source_to_solve=False, preserve_partner_solution=True, apply_mesh_operations=False):
            raise RuntimeError("Failed to import completed mesh")
        if not setup.update():
            raise RuntimeError("Failed to update reference setup")
        if not setup.enable_expression_cache([f"Force_{label}.Force_z" for label in "AB"],
                report_type="Magnetostatic", isconvergence=True, isrelativeconvergence=False,
                conv_criteria=1e-5, use_cache_for_freq=False):
            raise RuntimeError("Failed reference force convergence")
        folder = target.parent / "TightReference"
        folder.mkdir(parents=True, exist_ok=True)
        source_folder = source.parent / args.design
        manifest = json.loads((source_folder / "manifest.json").read_text())
        manifest.update(project=str(target), design="TightReference", percent_error=args.percent_error,
                        force_delta_n=1e-5, max_passes=15, minimum_passes=2,
                        minimum_converged_passes=2,
                        mesh_source_project=str(source), mesh_source_design=args.design,
                        reference_purpose="Common tighter force reference, not included in strategy timing")
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        mesh_plan = source_folder / "mesh_plan.json"
        if mesh_plan.exists():
            shutil.copy2(mesh_plan, folder / "mesh_plan.json")
        else:
            (folder / "mesh_plan.json").write_text(json.dumps({
                "policy": "legacy-source-mesh", "source_project": str(source),
                "source_design": args.design, "source_manifest_mesh_mm": manifest.get("mesh_mm"),
                "note": "Original mesh plan unavailable; reference starts from the recorded native source mesh, not a rebuilt mesh"
            }, indent=2), encoding="utf-8")
        shutil.copy2(source_folder / "maxwell_bh.csv", folder / "maxwell_bh.csv")
        if app.validate_simple(str(folder / "validation.log")) != 1 or not app.save_project():
            raise RuntimeError("Reference validation/save failed")
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    main()
