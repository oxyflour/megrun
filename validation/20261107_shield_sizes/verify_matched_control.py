"""Verify the saved fixed-mesh control after normalizing native path separators."""
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
project=ROOT/'matched_mesh_control_v6.aedt'
source=ROOT/'failure_snapshot_refinement24000/gap3mm_shield_sizes_v3.aedt'
os.environ['ANSYSEM_ROOT222']=r'C:\Program Files\AnsysEM\v222\Win64'
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings
settings.use_grpc_api=True;settings.enable_error_handler=False
app=Maxwell3d(projectname=str(project),designname='Shield_T010_NoShield',specified_version='2022.2',
             non_graphical=True,new_desktop_session=True,close_on_exit=False)
try:
    setup=app.get_setup('Reference');props=dict(setup.props)
    assert int(props['MaximumPasses'])==1 and int(props['MinimumPasses'])==1
    assert props['MeshLink']['ImportMesh']
    assert Path(props['MeshLink']['Project']).resolve()==source.resolve()
    assert props['MeshLink']['Design']=='Shield_T010'
    assert not props['MeshLink']['ForceSourceToSolve'] and not props['MeshLink']['ApplyMeshOp']
    assert app.modeler['MID_SHIELD'].material_name=='vacuum'
    np.testing.assert_allclose(app.modeler['MID_SHIELD'].bounding_box,[5.85,-4.4,1.4,5.95,4.4,4.8],atol=1e-9,rtol=0)
    np.testing.assert_allclose(app.modeler['Region'].bounding_box,[-24.4,-24.4,-18.6,36.2,24.4,24.8],atol=1e-9,rtol=0)
    for b in app.boundaries:
        if b.type=='Current':assert float(str(b.props['Current']).replace('A',''))==0
    assert app.validate_simple(str(ROOT/'matched_control_validation.log'))==1
    (ROOT/'matched_control_plan.json').write_text(json.dumps({'project':str(project),'design':'Shield_T010_NoShield',
      'source_project':str(source),'source_design':'Shield_T010','source_is_frozen_completed_snapshot':True,
      'purpose':'Check FEM mesh-subtraction contamination of shield Delta B','setup':props,
      'geometry_unchanged':True,'MID_SHIELD_material':'vacuum','maximum_additional_seconds':480,
      'same_original_group_deadline':True,'cores':2,'solve_submitted':False,
      'note':'Only a one-pass fixed-mesh diagnostic; actual full-corner mesh identity must be checked after solve'},indent=2),encoding='utf-8')
finally:
    app.release_desktop(close_projects=True,close_desktop=True)
print('FIXED-MESH CONTROL VERIFIED',flush=True)
