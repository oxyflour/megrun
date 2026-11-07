"""Prepare an isolated Maxwell material/mesh comparison; never submit a solve."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = Path(r'C:\Projects\magrun\outputs\maxwell_verified\gap_3mm_reference.aedt')
PROJECT = ROOT / 'gap3mm_vacoflux50.aedt'
MU0 = 4e-7 * np.pi
URL = 'https://vacuumschmelze.com/03_Documents/Brochures/Product%20Information%20Cobalt-Iron%20Alloys.pdf'


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')


def require(value, message):
    if value is False or value is None:
        raise RuntimeError(message)
    return value


def main():
    started = time.perf_counter()
    if PROJECT.exists():
        raise FileExistsError(PROJECT)
    h = np.array([0, 300, 800, 1600, 4000, 8000, 16000, 200000.])
    b = np.array([0, 1.90, 2.10, 2.20, 2.255, 2.28, 2.30, 2.30 + MU0 * (200000 - 16000)])
    assert np.all(np.diff(b / MU0 - h) >= -1e-5)
    np.savetxt(ROOT / 'vacoflux50_bh.csv', np.c_[h, b], delimiter=',',
               header='H_A_per_m,B_T', comments='')
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    save('material_provenance.json', {
        'source_project': str(SOURCE), 'source_sha256': source_hash,
        'material': 'VACOFLUX 50', 'condition': 'Final magnetic annealing; strip thickness 0.35 mm; typical DC values',
        'url': URL, 'pdf_page': 8,
        'manufacturer_H_Apm': h[1:-1].tolist(), 'manufacturer_B_T': b[1:-1].tolist(),
        'model_assumptions': ['Origin (0,0) added for a single-valued non-hysteretic law',
            'Piecewise-linear interpolation; not a dense measured dataset',
            'Above 16000 A/m use constant M, dB/dH=mu0; final endpoint is a model assumption',
            'Isotropic DC approximation; typical values do not identify the actual camera alloy',
            'Original permanent-magnet Br=1.2 T, recoil mu_r=1.05 and directions retained'],
        'current_scope': 'Awaiting user choice; no solve submitted'})
    shutil.copy2(SOURCE, PROJECT)
    os.environ['ANSYSEM_ROOT222'] = r'C:\Program Files\AnsysEM\v222\Win64'
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(PROJECT), designname='Gap_3mm_CCW',
                    specified_version='2022.2', non_graphical=True,
                    new_desktop_session=True, close_on_exit=False)
    try:
        save('desktop_session.json', {'pid': app.desktop_class.aedt_process_id,
             'project': str(PROJECT), 'version': app.aedt_version_id})
        for design in list(app.design_list):
            if design != 'Gap_3mm_CCW':
                app.delete_design(design)
        assert app.design_name == 'Gap_3mm_CCW'
        geometry = {name: {'volume_mm3': app.modeler[name].volume,
                          'bbox_mm': list(app.modeler[name].bounding_box),
                          'material': app.modeler[name].material_name}
                    for name in app.modeler.solid_names}
        steel = app.materials['Magrun_Steel_BH']
        steel.permeability.value = np.c_[h, b].tolist()
        require(steel.update(), 'Material update failed')
        save('maxwell_material_readback.json', steel._props)
        setup = app.get_setup('Reference')
        setup.props.update(MaximumPasses=8, MinimumPasses=3, MinimumConvergedPasses=2,
                           PercentError=0.5, NonLinearResidual=1e-5, SmoothBHCurve=False)
        require(setup.update(), 'Reference setup update failed')
        save('fine_setup.json', dict(setup.props))
        require(app.duplicate_design('MT_Coarse'), 'Duplicate coarse design failed')
        assert app.design_name == 'MT_Coarse'
        setup = app.get_setup('Reference')
        setup.props.update(MaximumPasses=1, MinimumPasses=1, MinimumConvergedPasses=1,
                           NonLinearResidual=1e-5, SmoothBHCurve=False)
        require(setup.update(), 'Coarse setup update failed')
        meshes = []
        for operation in app.mesh.meshoperations:
            if operation.props.get('Type') == 'LengthBased':
                operation.props['MaxLength'] = '1.5mm'
                require(operation.update(), 'Coarse mesh operation update failed')
            meshes.append({'name': operation.name, 'properties': dict(operation.props)})
        save('coarse_setup.json', {'setup': dict(setup.props), 'mesh_operations': meshes})
        after = {name: {'volume_mm3': app.modeler[name].volume,
                       'bbox_mm': list(app.modeler[name].bounding_box),
                       'material': app.modeler[name].material_name}
                 for name in app.modeler.solid_names}
        assert geometry == after, 'Duplicating the design changed geometry/material assignment'
        save('model_inventory.json', {'source_design': 'Gap_3mm_CCW',
             'designs': list(app.design_list), 'units': app.modeler.model_units,
             'solids': geometry, 'boundaries': [{'name': x.name, 'type': x.type,
                'properties': dict(x.props)} for x in app.boundaries],
             'geometry_copy_verified': True, 'current_scope': 'Pending user selection',
             'solve_submitted': False})
        require(app.save_project(), 'Project save failed')
        assert source_hash == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        save('prepare_status.json', {'state': 'Prepared, no solve submitted',
             'seconds': time.perf_counter() - started, 'source_unchanged': True,
             'project': str(PROJECT)})
        print('PREPARED:', PROJECT, flush=True)
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


if __name__ == '__main__':
    main()
