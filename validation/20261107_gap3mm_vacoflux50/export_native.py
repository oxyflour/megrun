"""AEDT 2022 R2 compatibility wrapper around the skill's material/mesh readers."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
TOOL = ROOT.parents[1] / 'skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
sys.path.insert(0, str(TOOL))
from aedt_export import classify_material, _validate_mesh_units_against_cad
from aedtplt import read_aedtplt
from bridge import read_maxwell_fld, check_points_in_air


def save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', required=True, choices=['MT_Coarse', 'Gap_3mm_CCW'])
    parser.add_argument('--project', type=Path, default=ROOT/'gap3mm_vacoflux50.aedt')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--audit-bh', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.out or ROOT / ('coarse_export' if args.design == 'MT_Coarse' else 'fine_export')
    output.mkdir(exist_ok=True)
    points = np.loadtxt(ROOT / 'points.csv', delimiter=',', skiprows=1)
    np.savetxt(output / 'points.csv', points, delimiter=',', header='x_m,y_m,z_m', comments='')
    # AEDT 2022 R2 consumes point file numbers in model units; verify returned SI coordinates.
    with (output / 'points_mm.pts').open('w') as stream:
        stream.write('Unit=mm\n')
        np.savetxt(stream, points * 1000, fmt='%.15g')
    os.environ['ANSYSEM_ROOT222'] = r'C:\Program Files\AnsysEM\v222\Win64'
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(args.project), designname=args.design,
                    specified_version='2022.2', non_graphical=True,
                    new_desktop_session=True, close_on_exit=False)
    try:
        for bound in app.boundaries:
            if bound.type == 'Current' and float(str(bound.props['Current']).removesuffix('A')) != 0:
                raise RuntimeError('Nonzero current source: unsupported by this skill')
        for method, name in [('export_convergence', 'convergence.txt'),
                             ('export_profile', 'profile.txt'), ('export_mesh_stats', 'mesh_stats.txt')]:
            kwargs = {'mesh_path' if method == 'export_mesh_stats' else 'file_path': str(output / name)}
            getattr(app, method)('Reference', **kwargs)
        profile = (output / 'profile.txt').read_text(errors='replace')
        if 'Normal Completion' not in profile:
            raise RuntimeError('Native profile does not show Normal Completion')
        assert app.post.export_field_file('B', solution='Reference : LastAdaptive',
            output_dir=str(output / 'maxwell_B.fld'), sample_points_file=str(output / 'points_mm.pts'),
            export_with_sample_points=True, export_in_si_system=True,
            reference_coordinate_system='Global', export_field_in_reference=True)
        reference, coordinate_scale = read_maxwell_fld(output / 'maxwell_B.fld', points)
        if coordinate_scale != 1.0:
            raise RuntimeError(f'Exported SI coordinates require an unexpected scale: {coordinate_scale}')
        np.savetxt(output / 'maxwell_B.csv', np.c_[points, reference], delimiter=',',
                   header='x_m,y_m,z_m,Bx_T,By_T,Bz_T', comments='')
        if args.audit_bh:
            expected = np.loadtxt(ROOT/'vacoflux50_bh.csv',delimiter=',',skiprows=1)
            actual = classify_material(app.materials['Magrun_Steel_BH'])
            np.testing.assert_allclose(actual['bh_H_Apm_B_T'],expected,rtol=1e-12,atol=1e-12)
            coarse = json.loads((ROOT/'coarse_export/model.json').read_text())
            centers = np.concatenate([np.load(ROOT/'coarse_export'/obj['mesh_npz'])['tetra_m'].mean(1)
                                      for obj in coarse['objects'] if obj['properties']['kind']=='nonlinear'])
            centers = centers[np.linspace(0,len(centers)-1,200,dtype=int)]
            with (output/'soft_centers_mm.pts').open('w') as stream:
                stream.write('Unit=mm\n'); np.savetxt(stream,centers*1000,fmt='%.15g')
            fields = {}
            for quantity in ['B','H']:
                path = output/f'soft_{quantity}.fld'
                assert app.post.export_field_file(quantity,solution='Reference : LastAdaptive',
                    output_dir=str(path),sample_points_file=str(output/'soft_centers_mm.pts'),
                    export_in_si_system=True,export_with_sample_points=True)
                fields[quantity], scale = read_maxwell_fld(path,centers)
                assert scale == 1.
            h = np.linalg.norm(fields['H'],axis=1)
            b = np.linalg.norm(fields['B'],axis=1)
            predicted_b = np.interp(h,expected[:,0],expected[:,1])
            above = h > expected[-1,0]
            predicted_b[above] = expected[-1,1]+4e-7*np.pi*(h[above]-expected[-1,0])
            relative_b = abs(b-predicted_b)/np.maximum(predicted_b,1e-12)
            save(output/'constitutive_audit.json',{'material_readback_matches_csv':True,
                 'sample_count':len(h),'H_min_Apm':float(h.min()),'H_max_Apm':float(h.max()),
                 'B_H_curve_median_relative_difference':float(np.median(relative_b)),
                 'B_H_curve_p95_relative_difference':float(np.percentile(relative_b,95)),
                 'B_H_curve_max_relative_difference':float(relative_b.max()),
                 'material_properties':actual})
        if args.design == 'MT_Coarse':
            (output / 'mesh').mkdir(exist_ok=True)
            model = {'schema_version': 1, 'source': {'project': str(ROOT / 'gap3mm_vacoflux50.aedt'),
                     'design': args.design, 'solution': 'Reference : LastAdaptive', 'model_unit': 'mm',
                     'current_A_turn': 0, 'native_status': 'Normal Completion'},
                     'objects': [], 'skipped': [], 'points': 'points.csv',
                     'maxwell_field': 'maxwell_B.fld', 'mesh_unit': 'm'}
            total = 0
            for index, name in enumerate(app.modeler.solid_names):
                obj = app.modeler[name]
                if 'YOKE' not in name and 'MAGNET_' not in name:
                    model['skipped'].append({'name': name, 'reason': 'air or zero-current nonmagnetic copper'})
                    continue
                properties = classify_material(app.materials[obj.material_name])
                plot = app.post.create_fieldplot_volume([name], 'Mesh', setup='Reference : LastAdaptive',
                                                       plot_name=f'MT_EXPORT_{index}')
                if not plot:
                    raise RuntimeError('Cannot create volume mesh plot for ' + name)
                try:
                    raw = app.post.export_field_plot(plot.name, str(output / 'mesh'),
                                                     file_name=name, file_format='aedtplt')
                finally:
                    plot.delete()
                vertices, stats = read_aedtplt(raw, 1e-3)
                vertices, stats = _validate_mesh_units_against_cad(obj, vertices, stats, 1e-3)
                expected_volume = float(obj.volume) * 1e-9
                stats['CAD_volume_m3'] = expected_volume
                stats['volume_relative_difference'] = abs(stats['volume_m3'] - expected_volume) / expected_volume
                if stats['volume_relative_difference'] > .02:
                    raise RuntimeError(f'{name}: exported mesh volume differs from CAD by >2%')
                total += len(vertices)
                print(name, len(vertices), stats, flush=True)
                filename = f'mesh/{name}.npz'
                np.savez_compressed(output / filename, tetra_m=vertices)
                model['objects'].append({'name': name, 'material': obj.material_name,
                    'properties': properties, 'properties_source': 'AEDT project material',
                    'mesh_npz': filename, 'mesh_raw': f'mesh/{name}.aedtplt', 'mesh_stats': stats})
            model['total_magnetic_tetra'] = total
            save(output / 'model.json', model)
            check_points_in_air(output, model, points)
            save(output / 'mesh_budget.json', {'tiles': total, 'FP64_K_bytes': 72 * total ** 2,
                 'FP32_K_bytes': 36 * total ** 2, 'within_8000_tile_budget': total <= 8000})
            if total > 8000:
                raise RuntimeError(f'{total} magnetic tetrahedra exceed the approved 8000-tile stop limit')
        save(output / 'export_status.json', {'state': 'Exported and SI coordinates verified',
             'design': args.design, 'seconds': time.perf_counter() - started, 'point_count': len(points)})
        app.save_project()
        print('EXPORTED:', output, flush=True)
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


if __name__ == '__main__':
    main()
