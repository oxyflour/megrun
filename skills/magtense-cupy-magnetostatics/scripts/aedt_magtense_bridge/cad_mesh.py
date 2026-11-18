"""Independent per-part STEP tetrahedralization for bridge.py mesh-cad.

Reads the hashed CAD-only aedt-native-disk-1 bundle and reuses the existing
material/current/unit checks. Gmsh/OpenCASCADE imports STEP in millimetres, fragments
touching solids to share interfaces without merging material labels, and
writes the existing model.json + per-object TET4 NPZ input. No air volume,
Maxwell mesh, Ansys module or native solver is used. Positive-volume overlaps,
lost material labels, nonconforming interfaces and volume errors fail closed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from aedt_export import (ConversionError, _read_native_bundle, _select_native_solids,
                         load_points)
from aedtplt import get_unit_scale, tetra_volume


def _keys(rows):
    rows=np.ascontiguousarray(np.sort(rows,axis=1),dtype=np.int64)
    return rows.view(np.dtype((np.void,24))).ravel()


def mesh_cad_bundle(bundle,out,*,size_mm,part_sizes=None,points_csv=None,
                    line_mm=None,overrides_path=None,max_tiles=1000000,
                    volume_rtol=.005,threads=1):
    """Mesh magnetic STEP solids, maintaining their original assembly coordinates.

    size_mm is the maximum target edge size; optional part_sizes is a JSON
    mapping object names to target sizes in mm. CAD and mesh volumes must agree
    within volume_rtol (default 0.5%); STEP/native analytic volumes within 1e-6.
    This is a geometry gate, not an estimate of field discretization accuracy.
    """
    if (not np.isfinite(size_mm) or size_mm<=0 or not 0<volume_rtol<1
            or max_tiles<1 or threads<1):
        raise ValueError('Positive finite CAD mesh size, budgets and volume tolerance required')
    root,files,native,member=_read_native_bundle(bundle)
    if native.get('mesh_source')!='cad':
        raise ConversionError('mesh-cad requires a mesh_source=cad STEP bundle')
    overrides=json.loads(Path(overrides_path).read_text(encoding='utf-8')) if overrides_path else {}
    selection,skipped=_select_native_solids(native,overrides,'cad_step')
    if len({o['name'] for o,_,_ in selection})!=len(selection):
        raise ConversionError('CAD object names must be unique')
    sizes=json.loads(Path(part_sizes).read_text(encoding='utf-8')) if part_sizes else {}
    if (not isinstance(sizes,dict) or set(sizes)-{o['name'] for o,_,_ in selection}
            or any(not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0 for v in sizes.values())):
        raise ConversionError('Part sizes must map selected CAD object names to positive mm sizes')
    if points_csv or line_mm:
        points=load_points(points_csv,line_mm)
    elif native.get('points'):
        points=load_points(str(member(native['points'])),None)
    else:
        raise ConversionError('Provide air points using --points-csv or --line-mm')
    for obj,_,_ in selection:
        member(obj['cad_step'])
        if not np.isfinite(obj.get('cad_volume_model_units3',np.nan)) or obj['cad_volume_model_units3']<=0:
            raise ConversionError(obj['name']+': native CAD volume is required to verify STEP units')
    target=Path(out).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('CAD mesh output must be new; existing evidence is retained')
    import gmsh
    if gmsh.isInitialized():
        raise RuntimeError('CAD meshing requires its own Gmsh session')
    target.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    report=dict(state='running',passed=False,gmsh_version=gmsh.__version__,
        source_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
            for n in ('cad_mesh.py','aedt_export.py','fem_bem.py')},
        settings=dict(size_mm=size_mm,part_sizes_mm=sizes,volume_rtol=volume_rtol,
            threads=threads,algorithm_3d=1,element_order=1,STEP_target_unit='MM',
            output_unit='m',boolean_tolerance_m=1e-11),
        scope='CAD geometry and conforming TET4 checks; field/Maxwell accuracy requires a separate solve')
    def checkpoint():
        report['elapsed_seconds']=time.perf_counter()-started
        (target/'meshing_info.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    initialized=False
    try:
        checkpoint()
        gmsh.initialize([],readConfigFiles=False);initialized=True
        gmsh.option.setNumber('General.Terminal',0)
        gmsh.option.setNumber('General.NumThreads',threads)
        # OCC's absolute precision works better at mm-scale CAD dimensions.
        # Converting geometry to metres before booleans can collapse tiny gaps.
        gmsh.option.setString('Geometry.OCCTargetUnit','MM')
        gmsh.option.setNumber('Geometry.Tolerance',1e-8)
        gmsh.option.setNumber('Geometry.ToleranceBoolean',1e-8)
        gmsh.model.add('magnetic_parts')
        occ=gmsh.model.occ
        original=[];owners=[];cad_volumes=[]
        scale=get_unit_scale(native['model_unit'])
        for k,(obj,_,_) in enumerate(selection):
            tags=occ.importShapes(str(member(obj['cad_step'])),highestDimOnly=True)
            if not tags or any(d!=3 for d,t in tags):
                raise ConversionError(obj['name']+': STEP must contain closed volume solids')
            expected=float(obj['cad_volume_model_units3'])*scale**3
            actual=sum(occ.getMass(d,t) for d,t in tags)*1e-9
            if abs(actual-expected)>1e-6*expected:
                raise ConversionError(f'{obj["name"]}: STEP/native CAD volume or unit mismatch: {actual} vs {expected}')
            original.extend(tags);owners.extend([k]*len(tags));cad_volumes.append(actual)
        if len(original)>1:
            fragments,mapping=occ.fragment(original[:1],original[1:])
        else:
            fragments,mapping=original,[original]
        volume_owner={}
        for k,tags in zip(owners,mapping):
            if not tags:raise ConversionError('CAD part disappeared during interface fragmentation')
            for d,t in tags:
                if d!=3:continue
                if t in volume_owner and volume_owner[t]!=k:
                    raise ConversionError('Positive-volume overlap between distinct CAD parts')
                volume_owner[t]=k
        if set(volume_owner)!={t for d,t in fragments if d==3}:
            raise ConversionError('Fragmentation lost CAD material ownership')
        occ.synchronize()
        volumes_by_part=[[t for t,owner in volume_owner.items() if owner==k] for k in range(len(selection))]
        for k,tags in enumerate(volumes_by_part):
            volume=sum(occ.getMass(3,t) for t in tags)*1e-9
            if not tags or abs(volume-cad_volumes[k])>1e-6*cad_volumes[k]:
                raise ConversionError('Fragmentation changed a CAD part volume')
            group=gmsh.model.addPhysicalGroup(3,tags)
            gmsh.model.setPhysicalName(3,group,selection[k][0]['name'])
        # Shared points use the finest incident part's request. Interior fields
        # retain per-part sizing while a shared CAD surface is meshed only once.
        point_sizes={};fields=[]
        for k,tags in enumerate(volumes_by_part):
            h=float(sizes.get(selection[k][0]['name'],size_mm))
            for tag in tags:
                for _,point in gmsh.model.getBoundary([(3,tag)],recursive=True,oriented=False):
                    point_sizes[point]=min(point_sizes.get(point,h),h)
            constant=gmsh.model.mesh.field.add('MathEval')
            gmsh.model.mesh.field.setString(constant,'F',repr(h))
            restrict=gmsh.model.mesh.field.add('Restrict')
            gmsh.model.mesh.field.setNumber(restrict,'InField',constant)
            gmsh.model.mesh.field.setNumbers(restrict,'VolumesList',tags)
            fields.append(restrict)
        for tag,h in point_sizes.items():gmsh.model.mesh.setSize([(0,tag)],h)
        minimum=gmsh.model.mesh.field.add('Min')
        gmsh.model.mesh.field.setNumbers(minimum,'FieldsList',fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)
        for name,value in {'Mesh.Algorithm3D':1,'Mesh.ElementOrder':1,
                'Mesh.MeshSizeMax':max([size_mm,*sizes.values()]),
                'Mesh.MeshSizeFromCurvature':16,'Mesh.Optimize':1,
                'Mesh.MaxNumThreads3D':threads,'Mesh.RandomSeed':1}.items():
            gmsh.option.setNumber(name,value)
        gmsh.model.mesh.generate(3)
        node_tags,coords,_=gmsh.model.mesh.getNodes()
        order=np.argsort(node_tags);node_tags=np.asarray(node_tags)[order]
        coords=np.asarray(coords).reshape(-1,3)[order]*1e-3
        def connectivity(dim,tag,expected_type,width):
            kinds,_,nodes=gmsh.model.mesh.getElements(dim,tag)
            if any(int(kind)!=expected_type for kind in kinds):
                raise ConversionError('Only linear triangles and TET4 are supported')
            raw=np.concatenate(nodes).astype(np.int64).reshape(-1,width) if nodes else np.empty((0,width),dtype=np.int64)
            indices=np.searchsorted(node_tags,raw)
            if np.any(indices>=len(node_tags)) or not np.array_equal(node_tags[indices],raw):
                raise ConversionError('Gmsh element references an unknown node')
            return indices
        parts=[];connections=[];total=0
        for tags in volumes_by_part:
            conn=np.concatenate([connectivity(3,t,4,4) for t in tags])
            total+=len(conn)
            if total>max_tiles:raise MemoryError(f'CAD mesh exceeds --max-tiles {max_tiles}: {total}')
            if not len(conn):raise ConversionError('CAD part generated no tetrahedra')
            connections.append(conn);parts.append(coords[conn])
        all_conn=np.concatenate(connections)
        faces=all_conn[:,[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1,3)
        faces,counts=np.unique(np.sort(faces,axis=1),axis=0,return_counts=True)
        if np.any(counts>2):raise ConversionError('Nonmanifold tetrahedral face incidence')
        surface_owners={}
        for volume,k in volume_owner.items():
            for _,surface in gmsh.model.getBoundary([(3,volume)],oriented=False):
                surface_owners.setdefault(surface,[]).append(k)
        exterior=[];interface=[]
        for surface,adjacent in surface_owners.items():
            triangles=connectivity(2,surface,2,3)
            if len(adjacent)==1:exterior.append(triangles)
            elif len(adjacent)==2:
                if adjacent[0]!=adjacent[1]:interface.append(triangles)
            else:raise ConversionError('Nonmanifold CAD surface incidence')
        expected=np.concatenate(exterior) if exterior else np.empty((0,3),dtype=np.int64)
        if not np.array_equal(np.sort(_keys(expected)),np.sort(_keys(faces[counts==1]))):
            raise ConversionError('Tetrahedral boundary does not match CAD exterior triangles')
        shared=np.concatenate(interface) if interface else np.empty((0,3),dtype=np.int64)
        if not np.all(np.isin(_keys(shared),_keys(faces[counts==2]))):
            raise ConversionError('CAD contact surface is not shared by both volume meshes')
        from fem_bem import FEMBEM,material_laws
        ids=np.concatenate([np.full(len(t),k,np.int32) for k,t in enumerate(parts)])
        geometry=FEMBEM(np.concatenate(parts),material_laws([p for _,p,_ in selection],ids),boundary_backend='geometry')
        report.update(geometry=geometry.info,shared_interface_triangles=len(shared),
            shared_CAD_surfaces=sum(len(set(a))==2 for a in surface_owners.values()),
            exterior_triangles=len(expected),objects=[])
        model=dict(schema_version=1,mesh_unit='m',points='points.csv',skipped=skipped,objects=[],
            source=dict(native['source'],mesh_source='Gmsh per-part STEP / conforming contact interfaces',
                native_manifest_sha256=report['source_manifest_sha256'],source_check=native['excitations'],
                native_boundaries=native['boundaries'],native_variables=native['variables'],
                offline_exterior='Infinite air; no air volume mesh'),total_magnetic_tetra=total)
        (target/'mesh').mkdir()
        for k,((obj,properties,origin),tetra) in enumerate(zip(selection,parts)):
            volumes=tetra_volume(tetra);volume=float(volumes.sum())
            mismatch=abs(volume-cad_volumes[k])/cad_volumes[k]
            if not np.isfinite(volumes).all() or np.any(volumes<=0) or mismatch>volume_rtol:
                raise ConversionError(f'{obj["name"]}: tetra/CAD volume error {mismatch} exceeds {volume_rtol}')
            relative=f'mesh/{k:05d}.npz';np.savez_compressed(target/relative,tetra_m=tetra)
            stats=dict(tetra_count=len(tetra),volume_m3=volume,cad_volume_m3=cad_volumes[k],
                cad_volume_relative_mismatch=mismatch,curved=False,element_order=1,
                target_size_mm=float(sizes.get(obj['name'],size_mm)))
            model['objects'].append(dict(name=obj['name'],material=obj['material'],properties=properties,
                properties_source='native_'+origin,mesh_npz=relative,mesh_stats=stats,
                mesh_npz_sha256=hashlib.sha256((target/relative).read_bytes()).hexdigest(),
                source_CAD_sha256=files[obj['cad_step']]['sha256']))
            report['objects'].append(dict(name=obj['name'],gmsh_volumes=volumes_by_part[k],**stats))
        np.savetxt(target/'points.csv',points,delimiter=',',header='x_m,y_m,z_m',comments='')
        model['points_sha256']=hashlib.sha256((target/'points.csv').read_bytes()).hexdigest()
        if native.get('reference_fields',{}).get('B') and native.get('points') and np.array_equal(points,load_points(str(member(native['points'])),None)):
            from bridge import read_maxwell_fld
            reference=member(native['reference_fields']['B'])
            _,coordinate_scale=read_maxwell_fld(reference,points)
            if coordinate_scale!=1.:raise ConversionError('Reference coordinates must use SI metres')
            (target/'maxwell_B.fld').write_bytes(reference.read_bytes());model['maxwell_field']='maxwell_B.fld'
        report.update(state='complete',passed=True);checkpoint()
        model['meshing']=report
        (target/'model.json').write_text(json.dumps(model,indent=2,allow_nan=False),encoding='utf-8')
        return model
    except Exception as error:
        report.update(state='failed',passed=False,error=f'{type(error).__name__}: {error}');checkpoint()
        raise
    finally:
        if initialized:gmsh.finalize()
