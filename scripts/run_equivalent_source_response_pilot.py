"""Native finite-source realization and camera response, capped at 25 minutes.

Virtual prescribed-magnetization sources (mu_r=1) realize a field-only fitted
loading basis. They are not the physical second camera. Native source-only
fields are validated first and used for all incremental-field subtractions.
The same augmented geometry and imported mesh are used at zero/half/full load.
"""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_native_response_pilot as framework
from scripts.refine_pair_difference_reference import read,save,native_convergence_rows,digest
from camera_mag.geometry import load_camera
from camera_mag.interaction import MU0
from camera_mag.response_library import read_vector_field,integrate_force

SOURCE=framework.SOURCE
BANK=ROOT/"outputs/equivalent_source_loading_20261003/sources_96_rcond_0.001.npz"
LIBRARY=ROOT/"outputs/single_camera_reuse/refined_library_0.5"
CASES=[("VacuumCheck",1.),("PilotZero",0.),("PilotHalf",.5),("PilotFull",1.)]


def session(project,design):
    os.environ["ANSYSEM_ROOT222"]=str(framework.AEDT.parent)
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api=True;settings.enable_error_handler=False
    return Maxwell3d(projectname=str(project),designname=design,specified_version="2022.2",
        non_graphical=True,new_desktop_session=True,close_on_exit=False)


def set_current(app,amplitude):
    for boundary in list(app.boundaries):
        if boundary.name=="A_AmpereTurns":
            framework.require(boundary.delete(),"Pilot current boundary removal failed")
    framework.require(app.assign_current("A_Terminal",amplitude=amplitude,
        solid=False,swap_direction=False,name="A_AmpereTurns"),"Pilot current boundary failed")


def set_fixed_pass(app,source_design=None,project=None):
    setup=app.get_setup("Reference")
    setup.auto_update=False
    setup.props.pop("ExpressionCache",None)
    setup.props.update(MaximumPasses=1,MinimumPasses=1,MinimumConvergedPasses=1,
        NonLinearResidual=1e-5,PercentError=.5)
    setup.props["MeshLink"]={"ImportMesh":False}
    if source_design:
        # An internal design link must use the native current-project token.
        # An absolute self-reference is an external-file dependency and may
        # invalidate results when the project is reopened. This correction is
        # for future builds; the failed experimental project is preserved.
        mesh_project="This Project*" if project and Path(project).resolve()==Path(app.project_file).resolve() else str(project)
        framework.require(setup.add_mesh_link(source_design,solution="Reference : LastAdaptive",
            project=mesh_project,force_source_to_solve=False,preserve_partner_solution=True,
            apply_mesh_operations=False),"Augmented-geometry mesh link failed")
    framework.require(setup.update(),"Fixed-pass setup failed")
    return dict(setup.props)


def build(output):
    source=read(SOURCE/"provenance.json")
    project=output/"NativeResponsePilot.aedt"
    if project.exists():raise FileExistsError(project)
    shutil.copy2(Path(source["project"]),project)
    app=session(project,source["design"])
    try:
        receiver=read(SOURCE/"manifest.json")["module_objects"]["A"]
        materials={name:app.modeler[name].material_name for name in receiver}
        volumes={name:app.modeler[name].volume for name in receiver}
        framework.require(app.duplicate_design("VacuumCheck"),"Source-only duplicate failed")
        for name in receiver:
            if name!="A_AF_COIL_COPPER":
                framework.require(app.assign_material(name,"vacuum"),"Source-only receiver removal failed")
        set_current(app,0)
        with np.load(BANK) as data:
            centres=data["centers_m"]
            moments=data["moments_A_m2"]
        separation=np.max(np.abs(centres[:,None,:]-centres[None,:,:]),axis=-1)
        np.fill_diagonal(separation,np.inf)
        edge_mm=min(.25,float(np.min(separation))*1e3*.4)
        cube_volume=(edge_mm*1e-3)**3
        shift=np.array([9.8,0,0])
        full_names=[];half_names=[];source_names=[]
        for index,(centre,moment) in enumerate(zip(centres,moments)):
            magnitude=float(np.linalg.norm(moment))
            direction=moment/magnitude
            for amplitude,names,label in ((1.,full_names,"Full"),(.5,half_names,"Half")):
                name=f"Loading_{label}_{index:03d}"
                material=framework.require(app.materials.add_material(name),"Source material creation failed")
                material.permeability=1
                material.conductivity=0
                framework.require(material.set_magnetic_coercivity(-amplitude*magnitude/cube_volume,
                                  *direction),"Fixed source magnetization failed")
                framework.require(material.update(),"Source material save failed")
                names.append(material.name)
            name=f"LoadingSource_{index:03d}"
            obj=framework.require(app.modeler.create_box((centre*1e3+shift-edge_mm/2).tolist(),
                [edge_mm]*3,name=name,material=full_names[-1]),"Finite loading source failed")
            source_names.append(obj.name)
        # A far boundary avoids imposing a second copy of the field library's
        # finite-boundary correction, which is already represented in the fit.
        framework.require(app.modeler.delete("Region"),"Original pilot air removal failed")
        framework.require(app.modeler.create_region(80,pad_type="Absolute Offset",name="Region"),
                          "Far-boundary pilot region failed")
        target=framework.require(app.modeler.create_box([-4.6,-4.6,1.2],[9.2,9.2,3.8],
            name="LoadingTargetQuery",material="vacuum"),"Loading query primitive failed")
        target.model=False
        framework.require(app.mesh.assign_length_mesh([target.name],inside_selection=True,
            maximum_length="0.4mm",maximum_elements=None,name="Loading_Target_Query"),"Loading query mesh failed")
        first_setup=set_fixed_pass(app)
        output.joinpath("VacuumCheck").mkdir(exist_ok=True)
        framework.require(app.validate_simple(str(output/"VacuumCheck/validation.log"))==1,
                          "Source-only native validation failed")
        framework.require(app.save_project(),"Source-only save failed")
        framework.require(app.mesh.generate_mesh("Reference"),"Source-only initial mesh failed")
        app.export_mesh_stats("Reference",mesh_path=str(output/"VacuumCheck/initial_mesh_stats.txt"))
        count=int(re.search(r"Total number of mesh elements:\s*(\d+)",
                  (output/"VacuumCheck/initial_mesh_stats.txt").read_text())[1])
        framework.require(app.save_project(),"Source-only mesh save failed")
        records={"VacuumCheck":dict(amplitude=1.,receiver_materials_removed=True,setup=first_setup)}
        if count>350000:
            save(output/"build_resource_stop.json",dict(initial_tetrahedra=count,
                 reason="Augmented mesh exceeds short-trial estimate; no field solve submitted"))
            raise RuntimeError("Initial mesh exceeds 350k short-trial guard")
        geometry=[name for name in app.modeler.solid_names if name!="Region"]
        geometry_volumes={name:app.modeler[name].volume for name in geometry}
        for design,amplitude in CASES[1:]:
            app.set_active_design("VacuumCheck")
            framework.require(app.duplicate_design(design),"Response design duplicate failed")
            for name,material in materials.items():
                framework.require(app.assign_material(name,material),"Receiver material restore failed")
            set_current(app,10)
            for index,name in enumerate(source_names):
                material="vacuum" if amplitude==0 else half_names[index] if amplitude==.5 else full_names[index]
                framework.require(app.assign_material(name,material),"Response source assignment failed")
            setup=set_fixed_pass(app,"VacuumCheck",project)
            if geometry_volumes!={name:app.modeler[name].volume for name in geometry}:
                raise ValueError("Augmented geometry differs between amplitudes")
            if volumes!={name:app.modeler[name].volume for name in receiver}:
                raise ValueError("Original receiving-camera volumes changed")
            folder=output/design;folder.mkdir(exist_ok=True)
            framework.require(app.validate_simple(str(folder/"validation.log"))==1,"Response native validation failed")
            records[design]=dict(amplitude=amplitude,setup=setup,receiver_volumes_mm3=volumes)
        framework.require(app.save_project(),"Response project save failed")
        save(output/"build.json",dict(project=str(project),initial_tetrahedra=count,designs=records,
            source_bank_sha256=digest(BANK),virtual_source_count=len(centres),virtual_source_edge_mm=edge_mm,
            virtual_sources_mu_r=1,virtual_sources_are_not_physical_camera_materials=True,
            same_augmented_geometry_between_amplitudes=True,original_camera_volumes_unchanged=True,
            max_proxy_remanence_T=float(MU0*np.max(np.linalg.norm(moments,axis=1))/cube_volume),
            source_only_loading_gate=dict(relative_field_RMS=.02,direct_force_difference_uN=1),
            no_force_fit=True,air_padding_mm=80))
    finally:
        app.release_desktop(close_projects=True,close_desktop=True)


def extract(output,design):
    project=output/"NativeResponsePilot.aedt"
    app=session(project,design)
    try:
        folder=output/design
        framework.require(app.export_profile("Reference",file_path=str(folder/"profile.txt")),"Profile export failed")
        framework.require(app.export_convergence("Reference",file_path=str(folder/"convergence.txt")),"Convergence export failed")
        app.export_mesh_stats("Reference",mesh_path=str(folder/"mesh_stats.txt"))
        profile=(folder/"profile.txt").read_text()
        if "Status: Normal Completion" not in profile:raise ValueError("Native solve lacks normal completion")
        rows=native_convergence_rows((project.with_suffix(".aedtresults")/(design+".asol")).read_text())
        if max(rows)!=1:raise ValueError("Expected exactly one completed pass")
        metadata=dict(native_status="Normal Completion",completed_passes=1,native=rows[1],
            fixed_pass_pilot=True,native_adaptive_converged=False,force_stability_checked=False,
            native_reported_adaptive_convergence="Adaptive Passes converged" in profile)
        camera=load_camera(ROOT/"models/B0500_scale_AF_VCM_EM_only.step",cell_mm=.5)
        with np.load(LIBRARY/"quadrature.npz") as q:
            coil,weights=q["coarse_points"],q["coarse_weights"]
        shift=np.array([9.8e-3,0,0])
        air=np.r_[camera.centers+shift,camera.centers-shift]
        points=np.r_[camera.centers,coil,air]
        path=folder/"samples_mm.pts"
        with path.open("w") as stream:
            stream.write("Unit=mm\n");np.savetxt(stream,points*1e3,fmt="%.12g")
        for quantity in ("B","H"):
            field_path=folder/(quantity+".fld")
            framework.require(app.post.export_field_file(quantity,solution="Reference : LastAdaptive",
                output_dir=str(field_path),sample_points_file=str(path),export_in_si_system=True),"Field export failed")
            read_vector_field(field_path,points)
        np.savez_compressed(folder/"sampling.npz",points_m=points,body_count=len(camera.centers),coil_count=len(coil),
            coil_weights_m3=weights,body_centers_m=camera.centers,body_volumes_m3=camera.volumes,
            material_ids=camera.material_ids.astype("U"))
        if design=="VacuumCheck":
            h=read_vector_field(folder/"H.fld",points)
            with np.load(LIBRARY/"quadrature.npz") as q:
                own=np.r_[q["coarse_points"],q["fine_points"]]
            reference_body=read_vector_field(LIBRARY/"gap_1_A_cell_incident_H.fld",camera.centers-shift)
            reference_coil_b=read_vector_field(LIBRARY/"gap_1_A_incident_B.fld",own-shift)[:len(coil)]
            reference_h=np.r_[reference_body,reference_coil_b/MU0]
            rms=float(np.linalg.norm(h[:len(reference_h)]-reference_h)/np.linalg.norm(reference_h))
            current=read_vector_field(LIBRARY/"own_J.fld",own)[:len(coil)]
            b=read_vector_field(folder/"B.fld",points)
            difference=float((integrate_force(current,b[len(camera.centers):len(camera.centers)+len(coil)],weights)[2]
                        -integrate_force(current,reference_coil_b,weights)[2])*1e6)
            metadata.update(native_loading_field_relative_RMS=rms,
                native_loading_direct_Fz_difference_uN=difference,loading_gate_met=rms<=.02 and abs(difference)<=1.)
        else:
            data=framework.require(app.post.get_solution_data(expressions=["Force_A.Force_z"],
                setup_sweep_name="Reference : LastAdaptive",report_category="Magnetostatic"),"Native force export failed")
            if data.units_data["Force_A.Force_z"] not in ("N","newton"):raise ValueError("Unexpected force units")
            metadata["Fz_N"]=float(data.data_real("Force_A.Force_z")[0])
            point_file=folder/"coil_mm.pts"
            with point_file.open("w") as stream:
                stream.write("Unit=mm\n");np.savetxt(stream,coil*1e3,fmt="%.12g")
            framework.require(app.post.export_field_file("J",solution="Reference : LastAdaptive",
                output_dir=str(folder/"J.fld"),sample_points_file=str(point_file),export_in_si_system=True),"J export failed")
            read_vector_field(folder/"J.fld",coil)
        save(folder/"result.json",metadata)
    finally:
        app.release_desktop(close_projects=True,close_desktop=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",default="outputs/native_equivalent_response_pilot_20261003")
    parser.add_argument("--phase",choices=["run","build","extract"],default="run")
    parser.add_argument("--design")
    parser.add_argument("--budget-minutes",type=float,default=25)
    args=parser.parse_args()
    if not 0<args.budget_minutes<=25:raise ValueError("Pilot budget must be at most 25 minutes")
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    if args.phase=="build":build(output)
    elif args.phase=="extract":extract(output,args.design)
    else:
        framework.MODE=BANK
        framework.CASES=CASES
        framework.__file__=__file__
        framework.run(output,args.budget_minutes)


if __name__=="__main__":main()
