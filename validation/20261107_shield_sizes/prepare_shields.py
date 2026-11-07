"""Prepare isolated shield-size references. Does not submit a solve."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / '20261107_repair_staged/full_pair_reference.aedt'
ORIGINAL = Path(r'C:\Projects\magrun\outputs\maxwell_verified\gap_3mm_reference.aedt')
PROJECT = ROOT / 'gap3mm_shield_sizes_v3.aedt'
CASES = [
    {'name': 'Shield_T010', 'thickness_mm': .10, 'width_mm': 8.8, 'height_mm': 3.4},
    {'name': 'Shield_T020', 'thickness_mm': .20, 'width_mm': 8.8, 'height_mm': 3.4},
    {'name': 'Shield_Large_T020', 'thickness_mm': .20, 'width_mm': 10., 'height_mm': 5.},
]

def save(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inside(points, lo, hi, tol=1e-8):
    return np.all((points >= np.asarray(lo)-tol) & (points <= np.asarray(hi)+tol), axis=1)

def make_points():
    base = ROOT.parent / '20261107_gap3mm_vacoflux50'
    previous = np.loadtxt(base / 'points.csv', delimiter=',', skiprows=1) * 1000
    labels = np.asarray(json.loads((base / 'point_groups.json').read_text())['labels'])
    # Keep all four original top lines except the three points in the largest shield.
    top = previous[labels != 'gap_vertical']
    top_labels = labels[labels != 'gap_vertical']
    conflict = inside(top, [5.8, -5, .6], [6, 5, 5.6])
    # Move the conflicting top-line x coordinates into verified gap air on the A side.
    top[conflict, 0] = 5.2
    z = np.linspace(.4, 5.8, 81)
    left = np.c_[np.full(len(z), 5.2), np.zeros(len(z)), z]
    right = np.c_[np.full(len(z), 6.6), np.zeros(len(z)), z]
    points = np.vstack([top, left, right])
    group = np.r_[top_labels, np.full(81, 'gap_A_side'), np.full(81, 'gap_B_side')]
    assert not np.any(inside(points, [5.8, -5, .6], [6, 5, 5.6]))
    assert len(points) == 566
    np.savetxt(ROOT / 'points.csv', points / 1000, delimiter=',', header='x_m,y_m,z_m', comments='')
    save('point_groups.json', {'count': len(points), 'labels': group.tolist(), 'units': 'm',
         'changed_top_point_count': int(conflict.sum()),
         'nominal_gap_midplane_mm': 5.9,
         'gap_measurement_x_mm': [5.2, 6.6],
         'reason': 'All cases, including no-shield baseline, use identical air coordinates; original central gap line intersects a shield'})
    return points

def inventory(app):
    return {name: {'volume_mm3': float(app.modeler[name].volume),
                   'bbox_mm': list(app.modeler[name].bounding_box),
                   'material': app.modeler[name].material_name}
            for name in app.modeler.solid_names}

def verify_inventory(actual, expected):
    assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
    for name, obj in expected.items():
        assert actual[name]['material'] == obj['material'], name
        np.testing.assert_allclose(actual[name]['volume_mm3'], obj['volume_mm3'], rtol=1e-10, atol=1e-10, err_msg=name)
        np.testing.assert_allclose(actual[name]['bbox_mm'], obj['bbox_mm'], rtol=1e-10, atol=1e-9, err_msg=name)

def main():
    started = time.perf_counter()
    if PROJECT.exists():
        raise FileExistsError(PROJECT)
    curve = np.loadtxt(ROOT / 'shield_bh.csv', delimiter=',', skiprows=1)
    assert curve.shape[1] == 2 and len(curve) >= 10
    assert np.all(np.diff(curve[:, 0]) > 0) and np.all(np.diff(curve[:, 1] / (4e-7*np.pi)-curve[:, 0]) >= -1e-5)
    make_points()
    original_hash, source_hash = sha(ORIGINAL), sha(SOURCE)
    shutil.copy2(SOURCE, PROJECT)
    os.environ['ANSYSEM_ROOT222'] = r'C:\Program Files\AnsysEM\v222\Win64'
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(PROJECT), designname='Gap_3mm_CCW', specified_version='2022.2',
                    non_graphical=True, new_desktop_session=True, close_on_exit=False)
    try:
        save('prepare_session.json', {'aedt_pid': app.desktop_class.aedt_process_id,
                                      'started_UTC': datetime.now(timezone.utc).isoformat()})
        original_inventory = inventory(app)
        assert all(str(b.props.get('Current', '0A')).replace(' ', '') in ('0A', '0', '0.0A')
                   for b in app.boundaries if b.type == 'Current')
        mat = app.materials.add_material('Shield_MUMETALL_VAC_curve')
        mat.permeability.value = curve.tolist()
        mat.conductivity = 1 / .6e-6
        mat.mass_density = 8700
        assert mat.update()
        save('shield_material_readback.json', mat._props)
        records = []
        for case in CASES:
            app.set_active_design('Gap_3mm_CCW')
            assert app.duplicate_design(case['name'])
            t, w, h = (case[k] for k in ('thickness_mm', 'width_mm', 'height_mm'))
            origin = [5.9-t/2, -w/2, 3.1-h/2]
            plate = app.modeler.create_box(origin, [t, w, h], name='MID_SHIELD', material=mat.name)
            assert plate
            assert abs(float(plate.volume)-t*w*h) < 1e-8
            # The original Region uses offsets from all solids. A larger shield
            # would silently enlarge it; freeze the original absolute boundary.
            bounds = original_inventory['Region']['bbox_mm']
            positions = [bounds[3], bounds[0], bounds[4], bounds[1], bounds[5], bounds[2]]
            assert app.modeler.change_region_padding([f'{v:.12g}mm' for v in positions],
                   ['Absolute Position']*6, ['+X','-X','+Y','-Y','+Z','-Z'])
            after = inventory(app)
            save(f"{case['name']}_inventory.json", {'before': original_inventory, 'after': after})
            verify_inventory({k: v for k, v in after.items() if k != 'MID_SHIELD'}, original_inventory)
            for name, obj in original_inventory.items():
                if obj['material'].lower() in ('air', 'vacuum'):
                    continue
                bounds = np.asarray(obj['bbox_mm'])
                overlap = np.minimum(bounds[3:], np.asarray(origin)+[t,w,h]) - np.maximum(bounds[:3], origin)
                assert np.any(overlap < -1e-7), f'Shield intersects {name}'
            setup = app.get_setup('Reference')
            setup.props.update(MaximumPasses=8, MinimumPasses=3, MinimumConvergedPasses=2,
                               PercentError=.5, NonLinearResidual=1e-5, SmoothBHCurve=False)
            setup.props['MeshLink'] = {'ImportMesh': False}
            assert setup.update()
            operation = app.mesh.assign_length_mesh(['MID_SHIELD'], inside_selection=True,
                         maximum_length='0.35mm', maximum_elements=None, name='Shield_Length')
            assert operation
            # Separate broad-face constraint controls resolution along thin sheets.
            report = ROOT / f"{case['name']}_native_validation.log"
            assert app.validate_simple(str(report)) == 1
            records.append({**case, 'project': str(PROJECT), 'design': case['name'],
                            'setup': 'Reference', 'origin_mm': origin, 'center_mm': [5.9, 0, 3.1],
                            'volume_mm3': float(plate.volume), 'current_A_turn': 0,
                            'material': mat.name, 'fine_setup': dict(setup.props),
                            'shield_mesh_operation': dict(operation.props),
                            'native_model_validation': 'Passed', 'solve_submitted': False})
        app.set_active_design('Gap_3mm_CCW')
        verify_inventory(inventory(app), original_inventory)
        assert app.save_project()
        assert sha(SOURCE) == source_hash and sha(ORIGINAL) == original_hash
        save('prepared_cases.json', {'source_project': str(SOURCE), 'source_sha256': source_hash,
             'original_project': str(ORIGINAL), 'original_sha256': original_hash,
             'original_geometry_unchanged': True, 'source_files_unchanged': True,
             'baseline_design': 'Gap_3mm_CCW', 'baseline_reference_results': str(SOURCE.with_suffix('.aedtresults')),
             'common_air_points': 566, 'cases': records,
             'seconds': time.perf_counter()-started, 'solve_submitted': False})
        print('Prepared and validated 3 shield sizes. No solve submitted.', flush=True)
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)

if __name__ == '__main__':
    main()
