"""Single active camera on the identical mesh used by the two-camera case."""
import argparse
import json
import os
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--design", default="Gap_2mm_CCW")
    p.add_argument("--baseline", default="Gap_2mm_OnlyA")
    args = p.parse_args()
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
        if not app.duplicate_design(args.baseline):
            raise RuntimeError("Design duplication failed")
        for name in app.modeler.solid_names:
            if name.startswith("B_") and "COIL" not in name:
                if not app.assign_material(name, "vacuum"):
                    raise RuntimeError("Failed to deactivate other magnetic body")
                app.modeler[name].solve_inside = True
        for boundary in app.boundaries:
            if boundary.name == "B_AmpereTurns":
                boundary.props["Current"] = "0A"
                if not boundary.update():
                    raise RuntimeError("Failed to deactivate other coil")
        setup = app.get_setup("Reference")
        setup.props["MaximumPasses"] = 1
        setup.props["MinimumPasses"] = 1
        setup.props["MinimumConvergedPasses"] = 1
        if not setup.add_mesh_link(args.design, solution="Reference : LastAdaptive", project=str(source),
                        force_source_to_solve=False, preserve_partner_solution=True, apply_mesh_operations=False):
            raise RuntimeError("Failed to link the identical source mesh")
        if not setup.update() or not app.save_project():
            raise RuntimeError("Failed to save matched baseline")
        folder = target.parent / args.baseline
        folder.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((source.parent / args.design / "manifest.json").read_text())
        manifest.update(project=str(target), design=args.baseline, active_camera="A",
                        inactive_camera="B: magnets/yoke vacuum, coil 0 A",
                        mesh_source_project=str(source), mesh_source_design=args.design,
                        maximum_passes=1, mesh_frozen=True)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
        if app.validate_simple(str(folder / "validation.log")) != 1:
            raise RuntimeError("Matched baseline validation failed")
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    main()
