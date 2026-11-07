"""Configure the user-approved zero-current comparison; no solve is launched."""
from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'gap3mm_vacoflux50.aedt'


def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def main():
    started = time.perf_counter()
    os.environ['ANSYSEM_ROOT222'] = r'C:\Program Files\AnsysEM\v222\Win64'
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(PROJECT), designname='Gap_3mm_CCW',
                    specified_version='2022.2', non_graphical=True,
                    new_desktop_session=True, close_on_exit=False)
    try:
        copper = app.materials['copper']
        old_mu = copper.permeability.value
        copper.permeability = 1.0
        assert copper.update()
        studies = {}
        for design in ['Gap_3mm_CCW', 'MT_Coarse']:
            app.set_active_design(design)
            assert app.design_name == design
            boundaries = []
            for boundary in app.boundaries:
                if boundary.type == 'Current':
                    boundary.props['Current'] = '0A'
                    assert boundary.update()
                boundaries.append({'name': boundary.name, 'type': boundary.type,
                                   'properties': dict(boundary.props)})
            if design == 'MT_Coarse':
                assignment = [name for name in app.modeler.solid_names if name != 'Region']
                app.odesign.GetModule('MeshSetup').EditLengthOp('Reference_Length', [
                    'NAME:Reference_Length', 'RefineInside:=', True, 'Enabled:=', True,
                    'Objects:=', assignment, 'RestrictElem:=', False, 'NumMaxElem:=', '1000',
                    'RestrictLength:=', True, 'MaxLength:=', '1.5mm'])
            validation = app.validate_simple(str(ROOT / f'{design}_validation.log'))
            assert validation == 1, f'{design}: native validation returned {validation}'
            studies[design] = {'boundaries': boundaries, 'setup': dict(app.get_setup('Reference').props),
                               'native_validation': validation}
        assert app.save_project()
        text = PROJECT.read_text(errors='replace')
        assert "MaxLength='1.5mm'" in text
        assert "Current='10A'" not in text
        assert text.count("Current='0A'") == 4
        inventory = json.loads((ROOT / 'model_inventory.json').read_text())
        magnetic = {name: data for name, data in inventory['solids'].items()
                    if 'YOKE' in name or 'MAGNET_' in name}
        bounds = np.array([v['bbox_mm'] for v in magnetic.values()], dtype=float)
        lo, hi = bounds[:, :3].min(axis=0), bounds[:, 3:].max(axis=0)
        yoke_a = np.array(magnetic['A_YOKE_SOFT_MAGNETIC']['bbox_mm'], dtype=float)
        yoke_b = np.array(magnetic['B_YOKE_SOFT_MAGNETIC']['bbox_mm'], dtype=float)
        gap_x = (yoke_a[3] + yoke_b[0]) / 2
        xs = np.linspace(lo[0] - 1, hi[0] + 1, 101)
        points, labels = [], []
        for y in [-3., 0., 3.]:
            points.extend(np.c_[xs, np.full_like(xs, y), np.full_like(xs, hi[2] + .5)])
            labels.extend(['near_top_0.5mm'] * len(xs))
        points.extend(np.c_[xs, np.zeros_like(xs), np.full_like(xs, hi[2] + 3)])
        labels.extend(['far_top_3mm'] * len(xs))
        zs = np.linspace(lo[2] - 1, hi[2] + 1, 81)
        points.extend(np.c_[np.full_like(zs, gap_x), np.zeros_like(zs), zs])
        labels.extend(['gap_vertical'] * len(zs))
        np.savetxt(ROOT / 'points.csv', np.array(points) * 1e-3, delimiter=',',
                   header='x_m,y_m,z_m', comments='')
        save('point_groups.json', {'labels': labels, 'count': len(labels),
                                   'coordinates_unit': 'm', 'source_geometry_bounds_mm': [lo.tolist(), hi.tolist()]})
        source = Path(json.loads((ROOT / 'material_provenance.json').read_text())['source_project'])
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        assert source_hash == json.loads((ROOT / 'material_provenance.json').read_text())['source_sha256']
        save('study_configuration.json', {'state': 'Ready; no solve submitted',
             'project': str(PROJECT), 'project_sha256': hashlib.sha256(PROJECT.read_bytes()).hexdigest(),
             'source_unchanged': True, 'current_A_turn': 0,
             'copper_mu_r_original': old_mu, 'copper_mu_r_for_both_solvers': 1.0,
             'copper_note': 'Weak diamagnetism omitted in both models; zero-current copper is nonmagnetic',
             'studies': studies, 'point_count': len(labels), 'seconds': time.perf_counter() - started,
             'fine_reference_is_not_the_old_solved_material': True})
        print('READY, NO SOLVE:', PROJECT, flush=True)
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


if __name__ == '__main__':
    main()
