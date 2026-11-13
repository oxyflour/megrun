"""Generate an isolated dense AEDT mesh for the current FMM scale study.

Input: ../20261107_repair_staged/full_pair_reference.aedt, Gap_3mm_CCW,
the existing zero-current 3 mm two-camera VACOFLUX 50 model. No source code is
copied. Geometry, materials and excitations must remain identical to this input.
Command (licensed AEDT 2022 R2, PyAEDT 0.8.11):
  C:/Projects/magrun/.venv/Scripts/python.exe prepare_mesh.py
Outputs: gap3mm_fmm_3m.aedt + its native aedtresults, mesh_stats.txt and
mesh_generation.json. Only GenerateMesh is submitted, not a field solution.
Pass: native validation succeeds, source hash/inventories are unchanged,
magnetic tetrahedra reach 90%-120% of 3 million and total tetrahedra reach
80%-135% of 9 million. Counts come from AEDT, not prescribed target values.
The pilot and at most four refinements require several GB of available RAM;
this is a mesh-capacity input, not a Maxwell accuracy or FMM performance result.
--prepare-only saves verified mesh settings without claiming a generated mesh.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'20261107_repair_staged/full_pair_reference.aedt'


def inventory(app):
    return {name:{'volume_mm3':float(app.modeler[name].volume),
                  'bounds_mm':list(app.modeler[name].bounding_box),
                  'material':app.modeler[name].material_name}
            for name in app.modeler.solid_names}


def read_counts(path, magnetic):
    text = path.read_text(errors='replace')
    found = re.search(r'Total number of mesh elements:\s*(\d+)',text)
    if not found:
        raise RuntimeError('AEDT did not export total tetrahedron count')
    per_object = {}
    for line in text.splitlines():
        match = re.match(r'\s*([^|]+?)\s*\|\s*(\d+)\s*\|',line)
        if match:
            per_object[match[1].strip()] = int(match[2])
    if not set(magnetic).issubset(per_object):
        raise RuntimeError('AEDT statistics omit magnetic objects')
    total = int(found.group(1))
    if sum(per_object.values()) != total:
        raise RuntimeError('Per-object tetrahedron counts do not sum to total')
    return {'total_tetrahedra':total,
            'magnetic_tetrahedra':sum(per_object[name] for name in magnetic),
            'per_object':per_object}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--design',default='Gap_3mm_CCW')
    parser.add_argument('--project',type=Path,default=ROOT/'gap3mm_fmm_3m.aedt')
    parser.add_argument('--target-magnetic',type=int,default=3000000)
    parser.add_argument('--target-total',type=int,default=9000000)
    parser.add_argument('--passes',type=int,default=5,help='Pilot plus bounded refinements')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--verify-saved',action='store_true',help='Reopen saved AEDT and verify persisted mesh counts')
    parser.add_argument('--resume',action='store_true',help='Continue this derived project only')
    args = parser.parse_args(argv)
    source,project = args.source.resolve(),args.project.resolve()
    if source == project or args.target_magnetic < 1 or args.target_total <= args.target_magnetic or args.passes < 1:
        parser.error('Require separate source/output and positive magnetic/total targets')
    project.parent.mkdir(parents=True,exist_ok=True)
    record_path = project.parent/'mesh_generation.json'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if project.exists() and not (args.resume or args.verify_saved):
        raise FileExistsError(f'{project}; use --resume only for this generated case')
    if not project.exists():
        shutil.copy2(source,project)
    os.environ['ANSYSEM_ROOT222'] = r'C:\Program Files\AnsysEM\v222\Win64'
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    settings.enable_desktop_logs = False
    record = {'source':str(source),'source_sha256':source_hash,'project':str(project),
              'design':args.design,'target_magnetic':args.target_magnetic,'target_total':args.target_total,
              'field_solve_submitted':False,'mesh_target_met':False,'state':'Opening derived project',
              'attempts':[]}
    previous = None
    if (args.resume or args.verify_saved) and record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous['source_sha256'] != source_hash or Path(previous['project']).resolve() != project:
            raise RuntimeError('Resume source/project do not match the existing mesh record')
        record['attempts'] = previous['attempts']
    if args.verify_saved:
        if previous is None or not previous['mesh_target_met']:
            raise RuntimeError('No completed mesh record is available for saved-project verification')
        record = previous
    def save():
        record_path.write_text(json.dumps(record,indent=2,default=str),encoding='utf-8')
    save()
    app = Maxwell3d(projectname=str(project),designname=args.design,specified_version='2022.2',
        non_graphical=True,new_desktop_session=True,close_on_exit=False)
    try:
        if args.verify_saved:
            assert inventory(app) == record['geometry']
            temp = ROOT.parents[1]/'.pytest_cache/saved_fmm_mesh.stats'
            temp.parent.mkdir(exist_ok=True)
            temp.unlink(missing_ok=True)
            app.export_mesh_stats('Reference',mesh_path=str(temp))
            actual = read_counts(temp,record['magnetic_objects'])
            expected = record['attempts'][-1]
            for key in ('total_tetrahedra','magnetic_tetrahedra','per_object'):
                assert actual[key] == expected[key], f'Saved mesh readback mismatch: {key}'
            record['saved_mesh_readback_verified'] = True
            record['project_sha256'] = hashlib.sha256(project.read_bytes()).hexdigest()
            temp.unlink()
            print('SAVED_MESH_VERIFIED '+json.dumps(actual),flush=True)
            return
        record['aedt_pid'] = app.desktop_class.aedt_process_id
        record['geometry'] = before = inventory(app)
        materials = {item['material'] for item in before.values()}
        properties = deepcopy({name:app.materials[name]._props for name in materials})
        boundaries = deepcopy([(b.name,b.type,dict(b.props)) for b in app.boundaries])
        for _,kind,props in boundaries:
            if kind == 'Current' and str(props.get('Current','0A')).replace(' ','') not in ('0','0A','0.0A'):
                raise RuntimeError('This FMM benchmark requires zero-current excitations')
        # Retain only the selected design in the derived file, with no mesh link
        # that could silently substitute the original coarse mesh.
        for name in list(app.design_list):
            if name != args.design:
                assert app.delete_design(name)
        app.set_active_design(args.design)
        magnetic = [name for name in before if 'YOKE' in name or 'MAGNET_' in name or name == 'MID_SHIELD']
        if len(magnetic) < 2:
            raise RuntimeError('No supported magnetic two-camera geometry identified')
        record['magnetic_objects'] = magnetic
        air = [name for name in before if name not in magnetic]
        setup = app.get_setup('Reference')
        setup.props['MeshLink'] = {'ImportMesh':False}
        assert setup.update()
        for operation in list(app.mesh.meshoperations):
            assert operation.delete()
        length_mag,length_air = .35,2.
        if record['attempts']:
            # Maxwell does not reapply an edited operation to an already
            # generated mesh. Unchanged counts cannot calibrate a new length.
            changed = [record['attempts'][0]]
            ignored = []
            for i,item in enumerate(record['attempts'][1:],start=2):
                if (item['magnetic_tetrahedra'],item['total_tetrahedra']) == (
                    changed[-1]['magnetic_tetrahedra'],changed[-1]['total_tetrahedra']):
                    ignored.append(i)
                else:
                    changed.append(item)
            record['ignored_unchanged_mesh_attempts'] = ignored
            last = changed[-1]
            length_mag,length_air = last['magnetic_length_mm'],last['air_length_mm']
            if last.get('magnetic_tetrahedra'):
                length_mag *= (last['magnetic_tetrahedra']/args.target_magnetic)**(1/3)
                nonmag = last['total_tetrahedra']-last['magnetic_tetrahedra']
                length_air *= (nonmag/(args.target_total-args.target_magnetic))**(1/3)
        mag_op = app.mesh.assign_length_mesh(magnetic,inside_selection=True,
            maximum_length=f'{length_mag:.9g}mm',maximum_elements=None,name='FMM_Magnetic_Length')
        air_op = app.mesh.assign_length_mesh(air,inside_selection=True,
            maximum_length=f'{length_air:.9g}mm',maximum_elements=None,name='FMM_Air_Length')
        assert mag_op and air_op
        assert app.validate_simple(str(project.parent/'model_validation.log')) == 1
        assert inventory(app) == before
        assert app.save_project()
        record['state'] = 'Configured; no mesh generated'
        record['magnetic_length_mm'],record['air_length_mm'] = length_mag,length_air
        save()
        if args.prepare_only:
            return
        for attempt in range(args.passes):
            import psutil
            available = psutil.virtual_memory().available
            if available < 2*1024**3:
                raise MemoryError('Less than 2 GiB RAM available; derived AEDT settings were saved')
            # Only this derived project's own mesh/results are removed. This
            # forces native remeshing; updated length properties alone are not
            # reapplied by Maxwell (Generating the Mesh Without Solving).
            app.odesign.DeleteFullVariation(['All'],False)
            assert mag_op.delete() and air_op.delete()
            suffix = len(record['attempts'])+1
            mag_op = app.mesh.assign_length_mesh(magnetic,inside_selection=True,
                maximum_length=f'{length_mag:.9g}mm',maximum_elements=None,
                name=f'FMM_Magnetic_Length_{suffix}')
            air_op = app.mesh.assign_length_mesh(air,inside_selection=True,
                maximum_length=f'{length_air:.9g}mm',maximum_elements=None,
                name=f'FMM_Air_Length_{suffix}')
            assert mag_op and air_op
            assert app.save_project()
            record['state'] = 'Generating mesh'
            record['magnetic_length_mm'],record['air_length_mm'] = length_mag,length_air
            save()
            print(f'MESH_START {attempt+1}: magnetic={length_mag:.6g}mm air={length_air:.6g}mm',flush=True)
            started = time.perf_counter()
            assert app.mesh.generate_mesh('Reference'), 'Native mesh generation failed'
            stats_path = project.parent/'mesh_stats.txt'
            stats_path.unlink(missing_ok=True)
            app.export_mesh_stats('Reference',mesh_path=str(stats_path))
            counts = read_counts(stats_path,magnetic)
            counts.update(magnetic_length_mm=length_mag,air_length_mm=length_air,
                          mesh_seconds=time.perf_counter()-started)
            if record['attempts'] and counts['per_object'] == record['attempts'][-1]['per_object']:
                raise RuntimeError('Native mesh counts did not change; do not reduce lengths against a cached mesh')
            record['attempts'].append(counts)
            record['state'] = 'Mesh generated; checking targets'
            print('MESH_RESULT '+json.dumps(counts),flush=True)
            assert app.save_project()
            save()
            mag_ok = .9*args.target_magnetic <= counts['magnetic_tetrahedra'] <= 1.2*args.target_magnetic
            total_ok = .8*args.target_total <= counts['total_tetrahedra'] <= 1.35*args.target_total
            if mag_ok and total_ok:
                record.update(state='Mesh target met',mesh_target_met=True)
                break
            # Calibrate against measured counts, generating a fresh mesh next.
            if counts['magnetic_tetrahedra'] > 1.2*args.target_magnetic or counts['total_tetrahedra'] > 1.35*args.target_total:
                raise RuntimeError('Mesh overshot target range; preserve evidence and revise on a fresh derived mesh')
            length_mag *= min(.98,(counts['magnetic_tetrahedra']/args.target_magnetic)**(1/3))
            nonmag = counts['total_tetrahedra']-counts['magnetic_tetrahedra']
            length_air *= min(.98,(nonmag/(args.target_total-args.target_magnetic))**(1/3))
        assert inventory(app) == before
        assert properties == {name:app.materials[name]._props for name in materials}
        assert boundaries == [(b.name,b.type,dict(b.props)) for b in app.boundaries]
        assert record['mesh_target_met'], 'Refinement pass limit reached before target'
        assert app.save_project()
    except BaseException as error:
        record.update(state='Failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['source_unchanged'] = hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
        app.release_desktop(close_projects=True,close_desktop=True)
        if args.verify_saved and record.get('saved_mesh_readback_verified'):
            record['project_unchanged_during_readback'] = (
                hashlib.sha256(project.read_bytes()).hexdigest() == record['project_sha256'])
            assert record['project_unchanged_during_readback']
        save()
    assert record['source_unchanged']


if __name__ == '__main__':
    main()
