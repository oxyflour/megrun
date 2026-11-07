"""Continue the two successful controls from their existing meshes within the same budget."""
from pathlib import Path
import json
import os
import shutil

ROOT = Path(__file__).resolve().parent
source = ROOT / 'staged_controls_v2.aedt'
project = ROOT / 'staged_controls_v3.aedt'
if project.exists():
    raise FileExistsError(project)
shutil.copy2(source, project)
os.environ['ANSYSEM_ROOT222'] = r'C:\Program Files\AnsysEM\v222\Win64'
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings
settings.use_grpc_api = True
settings.enable_error_handler = False
app = Maxwell3d(projectname=str(project), designname='Stage_PM_A', specified_version='2022.2',
                non_graphical=True, new_desktop_session=True, close_on_exit=False)
try:
    studies = json.loads((ROOT / 'native_studies.json').read_text())
    for name, configuration in studies.items():
        app.set_active_design(name)
        setup = app.get_setup('Reference')
        assert setup.add_mesh_link(name, project=str(source), solution='Reference : LastAdaptive',
            force_source_to_solve=False, preserve_partner_solution=True, apply_mesh_operations=False)
        assert setup.update()
        configuration.update(project=str(project), mesh_source=str(source),
                             continuation_of_6_pass_control=True)
        configuration['case'] = configuration['case'] + '_reference'
        (ROOT / configuration['case']).mkdir(exist_ok=True)
        shutil.copy2(ROOT / configuration['case'].removesuffix('_reference') / 'B.csv',
                     ROOT / configuration['case'] / 'B.csv')
    assert app.save_project()
    (ROOT / 'native_studies_refined.json').write_text(json.dumps(studies, indent=2))
finally:
    app.release_desktop(close_projects=True, close_desktop=True)
