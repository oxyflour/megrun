"""Continue from an existing isolated adaptive mesh; refine exterior query air only."""
import argparse
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--source", default="outputs/maxwell_verified/Isolated_CCW")
    p.add_argument("--output", default="outputs/single_camera_reuse/reused_single")
    args = p.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    provenance = json.loads((source / "provenance.json").read_text())
    source_project, project = Path(provenance["project"]), output / "SingleFieldReference.aedt"
    if project.exists():
        raise FileExistsError(project)
    start = perf_counter()
    shutil.copy2(source_project,project)
    os.environ["ANSYSEM_ROOT222"] = r"C:\Program Files\AnsysEM\v222\Win64"
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api=True
    settings.enable_error_handler=False
    app = Maxwell3d(projectname=str(project),designname=provenance["design"],specified_version="2022.2",
                    non_graphical=True,new_desktop_session=True,close_on_exit=False)
    try:
        if not app.duplicate_design("SingleFieldReference"):
            raise RuntimeError("Failed to duplicate isolated source design")
        physical = [name for name in app.modeler.solid_names if name!="Region"]
        volumes = {name:app.modeler[name].volume for name in physical}
        query_boxes=[]
        for label,x in [("NEG",-16.6),("POS",5.)]:
            obj=app.modeler.create_box([x,-4.6,1.2],[11.6,9.2,3.8],name="FieldQuery_"+label,material="vacuum")
            if not obj:
                raise RuntimeError("Query air refinement primitive failed")
            obj.model=False
            if obj.name not in app.modeler.non_model_objects:
                raise ValueError("Query primitive changed physical model")
            query_boxes.append(obj.name)
            if not app.mesh.assign_length_mesh([obj.name],inside_selection=True,maximum_length="0.6mm",
                        maximum_elements=80000,name="Neighbour_Query_"+label):
                raise RuntimeError("Query air mesh operation failed")
        if not app.mesh.assign_length_mesh(["A_AF_COIL_COPPER"],inside_selection=True,maximum_length="0.22mm",
                    maximum_elements=None,name="Coil_Field_Accuracy"):
            raise RuntimeError("Coil mesh operation failed")
        setup=app.get_setup("Reference")
        setup.props.pop("ExpressionCache",None)
        setup.props["MaximumPasses"],setup.props["MinimumPasses"],setup.props["MinimumConvergedPasses"]=8,2,2
        setup.props["PercentError"]=.25
        if not setup.add_mesh_link(provenance["design"],solution="Reference : LastAdaptive",project=str(source_project),
                    force_source_to_solve=False,preserve_partner_solution=True,apply_mesh_operations=True):
            raise RuntimeError("Isolated adaptive mesh link failed")
        if not setup.update() or not setup.enable_expression_cache(["Force_A.Force_z"],report_type="Magnetostatic",
                isconvergence=True,isrelativeconvergence=False,conv_criteria=5e-6,use_cache_for_freq=False):
            raise RuntimeError("Isolated force convergence setup failed")
        after={name:app.modeler[name].volume for name in physical}
        if volumes!=after:
            raise ValueError("Physical camera volumes changed")
        folder=output / "SingleFieldReference"
        folder.mkdir(parents=True,exist_ok=True)
        manifest=json.loads((source / "manifest.json").read_text())
        manifest.update(project=str(project),design="SingleFieldReference",minimum_passes=2,minimum_converged_passes=2,
            max_passes=8,percent_error=.25,force_delta_n=5e-6,
            mesh_source_project=str(source_project),mesh_source_design=provenance["design"],
            extra_mesh_only_primitives=query_boxes,physical_volume_check=dict(before=volumes,after=after),
            coil_mesh_mm=.22,neighbour_query_box_mesh_mm=.6,paired_force_fit_applied=False,
            field_library_purpose="Isolated native working point plus exterior air field; reuse prior adaptive mesh")
        (folder / "manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        shutil.copy2(source / "maxwell_bh.csv",folder / "maxwell_bh.csv")
        if app.validate_simple(str(folder / "validation.log"))!=1:
            raise RuntimeError("Reused source failed native Maxwell validation")
        if not app.save_project() or not app.mesh.generate_mesh("Reference"):
            raise RuntimeError("Reused source initial mesh generation failed")
        app.export_mesh_stats("Reference",mesh_path=str(folder / "initial_mesh_stats.txt"))
        import re
        count=int(re.search(r"Total number of mesh elements:\s*(\d+)",(folder / "initial_mesh_stats.txt").read_text())[1])
        if not app.save_project():
            raise RuntimeError("Reused source save failed")
        (output / "build_record.json").write_text(json.dumps(dict(seconds=perf_counter()-start,initial_tetrahedra=count,
            source_project=str(source_project),source_design=provenance["design"],
            mesh_imported=True,physics_modified=False,extra_mesh_only_primitives=query_boxes),indent=2))
        print(f"INITIAL_TETRAHEDRA: {count}",flush=True)
        if count>650000:
            raise RuntimeError("Reused initial mesh exceeds 650k guard")
    finally:
        app.release_desktop(close_projects=True,close_desktop=True)


if __name__=="__main__":
    main()
