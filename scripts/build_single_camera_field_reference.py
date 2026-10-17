"""Build an isolated Maxwell reference with refined air at all neighbour query positions."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", default="outputs/single_camera_reuse/refined_single")
    args = p.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    project, design = output / "SingleFieldReference.aedt", "SingleFieldReference"
    if project.exists():
        raise FileExistsError(project)
    command = [sys.executable, str(ROOT / "scripts/run_maxwell_case.py"), "--phase", "build",
        "--project", str(project), "--design", design, "--isolated", "--mesh-mm", ".3",
        "--minimum-passes", "2", "--minimum-converged-passes", "2", "--percent-error", ".25",
        "--force-delta-n", ".000005", "--max-passes", "12"]
    start = perf_counter()
    with (output / "build_base.log").open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    os.environ["ANSYSEM_ROOT222"] = r"C:\Program Files\AnsysEM\v222\Win64"
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(project), designname=design, specified_version="2022.2",
                    non_graphical=True, new_desktop_session=True, close_on_exit=False)
    try:
        physical = [name for name in app.modeler.solid_names if name!="Region"]
        volume_before = {name:app.modeler[name].volume for name in physical}
        query = app.modeler.create_box([-16.6, -4.6, 1.2], [33.2, 9.2, 3.8],
                                      name="FieldQueryMeshBox", matname="vacuum")
        if not query:
            raise RuntimeError("Failed to create air-query refinement primitive")
        query.model = False
        if query.name not in app.modeler.non_model_objects:
            raise ValueError("Query primitive was not marked non-model")
        if not app.mesh.assign_length_mesh([query.name], isinside=True, maximum_length="0.6mm",
                    maximum_elements=180000, name="Neighbour_Field_Coverage"):
            raise RuntimeError("Failed to refine neighbour-query air")
        if not app.mesh.assign_length_mesh(["A_AF_COIL_COPPER"], isinside=True, maximum_length="0.22mm",
                    maximum_elements=None, name="Coil_Field_Accuracy"):
            raise RuntimeError("Failed to refine own coil")
        volume_after = {name:app.modeler[name].volume for name in physical}
        if volume_before != volume_after:
            raise ValueError("Physical source solids changed during refinement")
        folder = output / design
        manifest = json.loads((folder / "manifest.json").read_text())
        manifest.update(field_library_purpose="Isolated operating point and off-body incident field for 1–3 mm pair predictions",
                        extra_mesh_only_primitives=[query.name], physical_volume_check=dict(before=volume_before, after=volume_after),
                        coil_mesh_mm=.22, neighbour_query_box_mesh_mm=.6,
                        neighbour_query_box_mm=[[-16.6,-4.6,1.2],[16.6,4.6,5]],
                        paired_force_fit_applied=False)
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if app.validate_simple(str(folder / "validation.log")) != 1:
            raise RuntimeError("Refined isolated source failed Maxwell validation")
        if not app.mesh.generate_mesh("Reference"):
            raise RuntimeError("Initial field-library mesh generation failed")
        app.export_mesh_stats("Reference", mesh_path=str(folder / "initial_mesh_stats.txt"))
        import re
        count = int(re.search(r"Total number of mesh elements:\s*(\d+)", (folder / "initial_mesh_stats.txt").read_text())[1])
        if not app.save_project():
            raise RuntimeError("Refined isolated source save failed")
        (output / "build_record.json").write_text(json.dumps(dict(command=command, seconds=perf_counter()-start,
            initial_tetrahedra=count, physics_modified=False, extra_mesh_only_primitives=[query.name]), indent=2))
        print(f"INITIAL_TETRAHEDRA: {count}", flush=True)
        if count>850000:
            raise RuntimeError("Single-camera library initial mesh exceeds 850k guard")
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    main()
