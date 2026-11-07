"""Prepare a fixed-mesh no-shield control; no solve is submitted here."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
source=ROOT/'failure_snapshot_refinement24000/gap3mm_shield_sizes_v3.aedt'
project=ROOT/'matched_mesh_control_v6.aedt'
if project.exists():raise FileExistsError(project)
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
shutil.copy2(source,project)
os.environ['ANSYSEM_ROOT222']=r'C:\Program Files\AnsysEM\v222\Win64'
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings
settings.use_grpc_api=True;settings.enable_error_handler=False
app=Maxwell3d(projectname=str(project),designname='Shield_T010',specified_version='2022.2',
              non_graphical=True,new_desktop_session=True,close_on_exit=False)
try:
    assert app.duplicate_design('Shield_T010_NoShield')
    for name in list(app.design_list):
        # PyAEDT 0.8.11 checks a local source design even for an external link.
        if name not in ('Shield_T010_NoShield','Shield_T010'):app.delete_design(name)
    # Deleting any design can reset PyAEDT's active-design/modeler caches.
    app.set_active_design('Shield_T010_NoShield')
    assert app.design_name=='Shield_T010_NoShield'
    before={name:{'bbox':list(app.modeler[name].bounding_box),'volume':float(app.modeler[name].volume),
                  'material':app.modeler[name].material_name} for name in app.modeler.solid_names}
    assert app.assign_material('MID_SHIELD','vacuum')
    setup=app.get_setup('Reference')
    setup.auto_update=False
    setup.props.update(MaximumPasses=1,MinimumPasses=1,MinimumConvergedPasses=1,
                        PercentError=.5,NonLinearResidual=1e-5,SmoothBHCurve=False)
    setup.props['MeshLink']={'ImportMesh':True,'Product':'Maxwell','Design':'Shield_T010',
          'Project':str(source),'PathRelativeTo':'SourceProduct','Soln':'Reference : LastAdaptive',
          'Params':{},'ForceSourceToSolve':False,'PreservePartnerSoln':True,'ApplyMeshOp':False,'AdaptPort':True}
    assert setup.update()
    setup.auto_update=True
    # set_active_design reparses the saved .aedt, so persist before readback.
    assert app.save_project()
    app.set_active_design('Shield_T010_NoShield')
    assert app.design_name=='Shield_T010_NoShield'
    setup=app.get_setup('Reference')
    assert int(setup.props['MaximumPasses'])==1
    assert setup.props['MeshLink']['ImportMesh']
    assert setup.props['MeshLink']['Project']==str(source)
    after={name:{'bbox':list(app.modeler[name].bounding_box),'volume':float(app.modeler[name].volume),
                 'material':app.modeler[name].material_name} for name in app.modeler.solid_names}
    for name,obj in before.items():
        np.testing.assert_allclose(after[name]['bbox'],obj['bbox'],rtol=1e-10,atol=1e-9,err_msg=name)
        np.testing.assert_allclose(after[name]['volume'],obj['volume'],rtol=1e-10,atol=1e-9,err_msg=name)
        if name!='MID_SHIELD':assert after[name]['material']==obj['material']
    assert after['MID_SHIELD']['material']=='vacuum'
    assert app.validate_simple(str(ROOT/'matched_control_validation.log'))==1
    assert app.save_project()
    (ROOT/'matched_control_plan.json').write_text(json.dumps({'project':str(project),
       'design':'Shield_T010_NoShield','source_project':str(source),'source_design':'Shield_T010',
       'purpose':'Separate different-FEM-mesh subtraction effects from predicted shield-induced field changes',
       'geometry_unchanged':True,'changed_material_only':'MID_SHIELD to vacuum','setup':dict(setup.props),
       'fixed_mesh_control_not_independent_adaptive_convergence':True,
       'required_mesh_check':'Compare all exported TET4 corner sets and tetrahedron counts after import',
       'maximum_additional_seconds':480,'same_original_group_deadline':True,'solve_submitted':False},indent=2),encoding='utf-8')
finally:
    app.release_desktop(close_projects=True,close_desktop=True)
# Source may be saved by its active native worker; this script never writes it.
print('PREPARED FIXED-MESH CONTROL; NO SOLVE',flush=True)
