"""Prepare independent Maxwell controls, sharing the already solved coarse mesh."""
from pathlib import Path
import json
import os
import shutil

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '20261107_gap3mm_vacoflux50'
SOURCE = BASE / 'gap3mm_vacoflux50.aedt'
PROJECT = ROOT / 'staged_controls_v2.aedt'
if PROJECT.exists():
    raise FileExistsError(PROJECT)
shutil.copy2(SOURCE, PROJECT)
os.environ['ANSYSEM_ROOT222'] = r'C:\Program Files\AnsysEM\v222\Win64'
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings
settings.use_grpc_api = True
settings.enable_error_handler = False
app = Maxwell3d(projectname=str(PROJECT), designname='MT_Coarse', specified_version='2022.2',
                non_graphical=True, new_desktop_session=True, close_on_exit=False)
try:
    material = app.materials.add_material('Diag_Linear1000')
    material.permeability = 1000
    material.conductivity = 0
    assert material.update()
    studies = {}
    for name, case in [('Stage_PM_A', 'single_pm'), ('Stage_Linear1000_A', 'single_linear_1000')]:
        app.set_active_design('MT_Coarse')
        assert app.duplicate_design(name)
        for solid in app.modeler.solid_names:
            if (solid.startswith('B_') and ('MAGNET_' in solid or 'YOKE' in solid)) or (solid == 'A_YOKE_SOFT_MAGNETIC' and case == 'single_pm'):
                assert app.assign_material(solid, 'vacuum')
            elif solid == 'A_YOKE_SOFT_MAGNETIC':
                assert app.assign_material(solid, material.name)
        setup = app.get_setup('Reference')
        setup.props.update(MaximumPasses=6, MinimumPasses=2, MinimumConvergedPasses=2,
                           PercentError=.5, NonLinearResidual=1e-6)
        assert setup.add_mesh_link('MT_Coarse', project=str(SOURCE), solution='Reference : LastAdaptive',
             force_source_to_solve=False, preserve_partner_solution=True, apply_mesh_operations=False)
        assert setup.update()
        assert app.validate_simple(str(ROOT / f'{name}_validation_v2.log')) == 1
        studies[name] = {'case': case, 'setup': dict(setup.props), 'mesh_source': str(SOURCE),
                          'current_A_turn': 0, 'project': str(PROJECT)}
    assert app.save_project()
    (ROOT / 'native_studies.json').write_text(json.dumps(studies, indent=2, default=str))
    print('PREPARED NATIVE CONTROLS', flush=True)
finally:
    app.release_desktop(close_projects=True, close_desktop=True)
