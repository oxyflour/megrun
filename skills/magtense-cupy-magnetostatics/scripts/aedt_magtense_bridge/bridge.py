#!/usr/bin/env python3
"""AEDT / Maxwell -> MagTense tetrahedral solver / accuracy comparison.

Examples:
  python bridge.py demo --out demo
  python bridge.py export --project C:\\x\\camera.aedt --design Maxwell3DDesign1 \
      --out out --line-mm '0,0,2:10,0,2:101'
  python bridge.py inspect --out out
  python bridge.py solve --out out --max-tiles 10000
  python bridge.py compare --out out
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import permutations
from pathlib import Path
import re
import sys
import hashlib
import time

import numpy as np

from aedtplt import tetra_volume
from magtense_adapter import MU0, assemble_tiles, solve as mt_solve


def read_model(root: Path):
    p = root/'model.json'
    if not p.exists():
        raise FileNotFoundError(f'{p} not found. Run export or demo first.')
    model = json.loads(p.read_text(encoding='utf-8'))
    if model['schema_version'] != 1:
        raise RuntimeError(f'Unknown model schema_version {model["schema_version"]}')
    for relative,digest in [(o['mesh_npz'],o.get('mesh_npz_sha256')) for o in model['objects']]+[
            (model['points'],model.get('points_sha256'))]:
        if digest and hashlib.sha256((root/relative).read_bytes()).hexdigest()!=digest:
            raise ValueError('Imported input SHA-256 mismatch: '+relative)
    return model


def load_points(root: Path, model):
    a = np.loadtxt(root/model['points'], delimiter=',', skiprows=1,ndmin=2)
    if a.shape[1] != 3:
        raise ValueError('points.csv must contain x_m,y_m,z_m')
    return a


def check_points_in_air(root: Path, model: dict, pts: np.ndarray):
    """Fail if a sample point is in/on any magnetic TET4.

    Bounding box first, then exact tetra barycentric test. O(points * tiles)
    broad phase, but only narrow-phase solve near nearby tetra.
    """
    tolerance = 1e-10  # barycentric dimensionless tolerance
    for obj in model['objects']:
        verts = np.load(root/obj['mesh_npz'])['tetra_m']
        lo, hi = verts.min(axis=1), verts.max(axis=1)
        for k, p in enumerate(pts):
            cand = np.flatnonzero(np.all((p >= lo) & (p <= hi),axis=1))
            for idx in cand:
                v = verts[idx]
                mat = np.stack((v[1]-v[0],v[2]-v[0],v[3]-v[0]),axis=1)
                try:
                    c = np.linalg.solve(mat, p-v[0])
                except np.linalg.LinAlgError:
                    raise ValueError(f'Degenerate tetrahedron in {obj["name"]} #{idx}')
                if np.all(c >= -tolerance) and np.sum(c) <= 1+tolerance:
                    raise ValueError(f'Sample point #{k} {p} is on/in magnetic object '
                                     f'{obj["name"]} tetra #{idx}; B != mu0*H inside magnets.')


def summary(root: Path):
    model = read_model(root)
    n = model['total_magnetic_tetra']
    print(f'Source: {model["source"]}')
    print(f'Magnetic tetrahedra: {n:,} (air and mu_r=1 solids skipped)')
    print(f'Full GPU K: {36*n*n/1e9:.3f} GB FP32 / {72*n*n/1e9:.3f} GB FP64, plus workspace')
    for ob in model['objects']:
        s = ob['mesh_stats']
        print(f'  {ob["name"]}: {s["tetra_count"]:,} tiles, {ob["properties"]["kind"]}, '
              f'{ob["material"]}, curved={s["curved"]}')
    print(f'Skipped: {len(model["skipped"])} solids')
    for x in model['skipped'][:12]:
        print('  -',x['name'], ':',x['reason'])


def command_fem_bem(args):
    """Solve a portable SI model using current FEM-BEM; no AEDT or MagTense DLL."""
    from aedt_export import _native_physics
    from fem_bem import FEMBEM,material_laws
    root=Path(args.out).resolve();model=read_model(root);started=time.perf_counter()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.result_prefix):
        raise ValueError('--result-prefix must contain only letters, digits, underscores or hyphens')
    outputs=[root/(args.result_prefix+suffix) for suffix in ('_solution.npz','_B.csv','_solve_info.json')]
    if any(p.exists() for p in outputs):
        raise FileExistsError('Existing FEM-BEM evidence retained; use a separate imported job folder')
    source=model.get('source',{})
    _native_physics(dict(excitations=source.get('source_check',[]),boundaries=source.get('native_boundaries',[]),
        variables=source.get('native_variables',{})))
    info=dict(state='running',backend=args.boundary,source_model_sha256=hashlib.sha256((root/'model.json').read_bytes()).hexdigest(),
        scope='Offline P1/P0 magnetostatic field, zero free currents, infinite exterior air; native high-order equivalence not certified')
    def write():outputs[2].write_text(json.dumps(info,indent=2,default=str),encoding='utf-8')
    last_progress=[0.]
    def progress(phase,**details):
        info.update(phase=phase,elapsed_seconds=time.perf_counter()-started)
        now=time.perf_counter()
        if phase=='boundary_build' and now-last_progress[0]<10:return
        last_progress[0]=now;info['progress']=details;write()
        print('FEM_BEM',phase,json.dumps(details,default=str),flush=True)
    try:
        parts=[np.load(root/o['mesh_npz'])['tetra_m'] for o in model['objects']]
        ids=np.concatenate([np.full(len(t),i,np.int32) for i,t in enumerate(parts)])
        tetra=np.concatenate(parts);del parts
        options=dict(leaf_size=args.h_leaf,eta=args.h_eta,rtol=args.h_rtol,max_rank=args.h_rank,
            budget_gib=args.h_gib,storage_dtype=args.h_storage,far_quadrature_order=args.h_far_order,
            cache_path=Path(args.boundary_cache).resolve() if args.boundary_cache else root/'boundary_cache',progress=progress)
        mesh=FEMBEM(tetra,material_laws([o['properties'] for o in model['objects']],ids),quadrature_order=8,
            boundary_backend=args.boundary,boundary_options=options if args.boundary=='cuda_hmatrix' else None)
        del tetra,ids
        if args.boundary=='cuda_hmatrix':
            mesh.boundary.save_cache()
            info['boundary_refinement']=mesh.refine_boundary(rtol=args.h_rtol,progress=progress)
            rng=np.random.default_rng(827319)
            rows=np.unique(np.r_[rng.choice(len(mesh.faces),min(8,len(mesh.faces)),replace=False),
                np.argmin(mesh.areas),np.argmax(mesh.areas)])
            info['boundary_checks']=[]
            for label,q,u in [('random',rng.normal(size=len(mesh.faces)),rng.normal(size=len(mesh.boundary_nodes))),
                    ('flux_only',rng.normal(size=len(mesh.faces)),np.zeros(len(mesh.boundary_nodes))),
                    ('smooth',mesh.normals[:,2],mesh.vertices[mesh.boundary_nodes,2])]:
                approximate=mesh.boundary.apply(q,u)[rows];original=mesh.boundary.direct_rows(rows,q,u)
                error=float(np.linalg.norm(approximate-original)/max(np.linalg.norm(original),1e-300))
                info['boundary_checks'].append(dict(input=label,relative_L2=error,tolerance=20*args.h_rtol))
                if error>20*args.h_rtol:raise ArithmeticError(f'Original boundary probe failed: {label} {error}')
        u,q,solver=mesh.solve(tol=args.tol,maxiter=args.maxiter,linear_rtol=args.linear_rtol,
            linear_maxiter=args.linear_cycles,fem_preconditioner=args.fem_preconditioner,progress=progress)
        info['solver']=solver
        np.savez_compressed(outputs[0],potential_scaled_Apm=u,boundary_q_Apm=q)
        points=load_points(root,model)
        B,gradient=mesh.air_field(u,q,points,validate_air=mesh.validate_air)
        np.savetxt(outputs[1],np.column_stack((points,B)),delimiter=',',
            header='x_m,y_m,z_m,Bx_T,By_T,Bz_T',comments='')
        info.update(air_gradient=gradient,operator=mesh.info,
            solution_sha256=hashlib.sha256(outputs[0].read_bytes()).hexdigest())
        if model.get('maxwell_field'):
            reference,scale=read_maxwell_fld(root/model['maxwell_field'],points)
            if scale!=1.:raise ValueError('Native reference points must be in SI metres')
            error=float(np.linalg.norm(B-reference)/max(np.linalg.norm(reference),1e-300))
            info['native_air_B_sample_relative_L2']=error
            if args.max_reference_error is not None and error>args.max_reference_error:
                raise ArithmeticError(f'Native air-field sample error {error} exceeds {args.max_reference_error}')
        elif args.max_reference_error is not None:
            raise ValueError('--max-reference-error requires a native field reference')
        info.update(state='complete',passed=True,elapsed_seconds=time.perf_counter()-started);write()
        print('Saved offline FEM-BEM solution:',outputs[0])
    except Exception as error:
        info.update(state='failed',passed=False,error=f'{type(error).__name__}: {error}',elapsed_seconds=time.perf_counter()-started)
        write();raise


def command_solve(args):
    root = Path(args.out).resolve()
    model = read_model(root)
    pts = load_points(root,model)
    print(f'Loading {model["total_magnetic_tetra"]:,} magnetic tetra and {len(pts)} air evaluation points')
    check_points_in_air(root, model, pts)
    if args.dry_run:
        summary(root)
        return
    tiles, bh, stats = assemble_tiles(model['objects'],root,max_tiles=args.max_tiles)
    print('Assembled:',stats)
    if bh is not None:
        print(f'Custom B-H: {len(bh)} distinct curves, original point counts '
              f'{[len(c) for c in bh]} -> independent Fortran StateFunctions')
        for a in stats['nonlinear_material_assignments']:
            print(f'  {a["object"]} ({a["material"]}): state_function_index={a["state_function_index"]}')
    predicted, diagnostics = mt_solve(tiles,pts,bh,max_error=args.max_error,
                                     max_it=args.max_it,temperature_K=args.temperature_K)
    out = np.column_stack([pts,predicted])
    np.savetxt(root/'magtense_B.csv',out,delimiter=',',
               header='x_m,y_m,z_m,Bx_T,By_T,Bz_T',comments='')
    np.savez_compressed(root/'magnetization.npz',M_Apm=tiles.M)
    diagnostics['tile_stats'] = stats
    (root/'solve_info.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    print('Wrote',root/'magtense_B.csv')
    print('Final M_rel:',diagnostics['max_final_M_rel'])
    if diagnostics.get('warning'):
        print('WARNING:',diagnostics['warning'])


def _as_float_list(line: str):
    if not line.strip() or line.lstrip().startswith(('#','%','//')):
        return None
    # Separators can be whitespace, comma, or semicolon. Reject text headers.
    toks = re.split(r'[\s,;]+',line.strip())
    try:
        return [float(x.replace('D','E').replace('d','e')) for x in toks if x]
    except ValueError:
        return None


def read_maxwell_fld(path: Path, points_m: np.ndarray):
    """Convert AEDT .fld text (3 or 6 numerical columns) to Tesla vectors.

    For six columns require coordinate matches (with scale check). For unsupported
    formats refuse and ask the user to export 6-column reference CSV explicitly.
    """
    lines = path.read_text(encoding='utf-8-sig', errors='replace').splitlines()
    rows = []
    for line in lines:
        nums = _as_float_list(line)
        if nums is None:
            continue
        if len(nums) in (3, 6):
            rows.append(nums)
    if len(rows) != len(points_m):
        # Some AEDT files may have a leading single numerical line, which is
        # ignored above; other formats are refused rather than mis-compared.
        raise ValueError(f'{path}: parsed {len(rows)} numeric [3 or 6] rows, but '
                         f'expected {len(points_m)}. Inspect .fld format and export a '
                         f'6-column CSV [x_m,y_m,z_m,Bx_T,By_T,Bz_T] if needed.')
    if len({len(x) for x in rows}) > 1:
        raise ValueError('Mixed 3- and 6-column Maxwell reference rows unsupported')
    data = np.asarray(rows)
    if data.shape[1] == 6:
        coords = data[:,:3]
        for scale in (1.0,1e-3,1e3,1e-6):
            if np.allclose(coords*scale,points_m,rtol=1e-7,atol=1e-10):
                return data[:,3:], scale
        raise ValueError('Maxwell .fld XYZ do not align with points.csv; wrong units/solution/order. '
                         f'First Maxwell XYZ={coords[0]}, first point_m={points_m[0]}')
    return data, None


def command_compare(args):
    root=Path(args.out).resolve()
    model=read_model(root)
    pts=load_points(root,model)
    pred_path = Path(args.prediction)
    if not pred_path.is_absolute():
        pred_path = root / pred_path
    pred=np.loadtxt(pred_path,delimiter=',',skiprows=1,ndmin=2)
    if pred.shape!=(len(pts),6) or not np.allclose(pred[:,:3],pts,rtol=0,atol=1e-13):
        raise ValueError('Prediction coordinates do not match points.csv')
    if args.maxwell_csv:
        arr=np.loadtxt(args.maxwell_csv,delimiter=',',skiprows=1,ndmin=2)
        if arr.shape == (len(pts),6):
            if not np.allclose(arr[:,:3],pts,rtol=1e-7,atol=1e-10):
                raise ValueError('Maxwell CSV coordinates do not match points.csv')
            ref=arr[:,3:]
        elif arr.shape == (len(pts),3):
            ref=arr
        else:
            raise ValueError('Maxwell CSV must contain Bx,By,Bz or xyz,Bx,By,Bz')
        scale='CSV supplied'
    else:
        field = root/model['maxwell_field']
        if not field.exists():
            raise FileNotFoundError(f'No Maxwell reference: {field}. '
                 'Use --maxwell-csv after exporting aligned B vectors from Maxwell.')
        ref, scale=read_maxwell_fld(field,pts)
    if not np.all(np.isfinite(ref)):
        raise ValueError('Reference contains non-finite values')
    B=pred[:,3:]
    err=B-ref
    err_norm=np.linalg.norm(err,axis=1)
    base=np.linalg.norm(ref,axis=1)
    err_rms=float(np.sqrt(np.mean(np.sum(err**2,axis=1))))
    base_rms=float(np.sqrt(np.mean(np.sum(ref**2,axis=1))))
    percent=100*err_rms/base_rms if base_rms>1e-20 else None
    metrics={
      'points':len(pts), 'maxwell_coordinate_scale_to_m':scale,
      'rms_vector_abs_error_mT':float(err_rms*1000),
      'rms_vector_reference_mT':float(base_rms*1000),
      'rms_vector_relative_error_percent':percent,
      'mean_abs_vector_error_mT':float(np.mean(err_norm)*1000),
      'max_abs_vector_error_mT':float(np.max(err_norm)*1000),
      'rms_component_error_mT':(np.sqrt(np.mean(err**2,axis=0))*1000).tolist(),
      'maxwell_max_B_mT':float(np.max(base)*1000),
    }
    cols=np.column_stack((pts,ref,B,err,err_norm[:,None],base[:,None]))
    predicted_name = 'MagTense' if pred_path.name == 'magtense_B.csv' else 'Predicted'
    prefix = args.prefix
    np.savetxt(root/f'{prefix}comparison.csv',cols,delimiter=',',comments='',
        header='x_m,y_m,z_m,Bx_Maxwell_T,By_Maxwell_T,Bz_Maxwell_T,'
               f'Bx_{predicted_name}_T,By_{predicted_name}_T,Bz_{predicted_name}_T,'
               'dBx_T,dBy_T,dBz_T,dBnorm_T,BMaxwell_norm_T')
    (root/f'{prefix}metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))
    print('Output:',root/f'{prefix}comparison.csv')


def command_demo(args):
    """Offline TET4 cube test fixture only, NOT a real Maxwell accuracy comparison."""
    root = Path(args.out).resolve()
    (root/'mesh').mkdir(parents=True,exist_ok=True)
    o=np.zeros(3); end=np.ones(3)*1e-3
    tetra=[]
    for perm in permutations(range(3)):
        a=np.eye(3)[perm[0]]*1e-3
        b=(np.eye(3)[perm[0]]+np.eye(3)[perm[1]])*1e-3
        tetra.append([o,a,b,end])
    vertices=np.array(tetra)
    vol=tetra_volume(vertices)
    neg=vol<0
    if np.any(neg):
        tmp=vertices[neg,1,:].copy()
        vertices[neg,1,:]=vertices[neg,2,:]
        vertices[neg,2,:]=tmp
    vol=tetra_volume(vertices)
    assert np.all(vol>0)
    f=root/'mesh/demo_magnet.npz'
    np.savez_compressed(f,tetra_m=vertices)
    points=np.linspace([2e-3,0.5e-3,0.5e-3],[5e-3,0.5e-3,0.5e-3],50)
    np.savetxt(root/'points.csv',points,delimiter=',',header='x_m,y_m,z_m',comments='')
    model={'schema_version':1,'source':{'project':'OFFLINE DEMO - NOT MAXWELL',
         'design':'cube','solution':'none','model_unit':'m'},
         'mesh_unit':'m','points':'points.csv','maxwell_field':'maxwell_B.fld',
         'skipped':[{'name':'Air','reason':'air/vacuum'}],
         'total_magnetic_tetra':len(vertices),
         'objects':[{'name':'cube_magnet','material':'NdFeB_demo',
                     'properties':{'kind':'permanent','mu_r':1.05,'br_T':1.2,
                                   'direction':[1,0,0]},
                     'mesh_npz':'mesh/demo_magnet.npz',
                     'mesh_stats':{'tetra_count':len(vertices),
                         'volume_m3':float(vol.sum()),'tet10':0,'tet4':len(vertices),
                         'curved':False,'orientation_flipped':0}}]}
    (root/'model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    print('Offline demo created:',root)
    print('This dataset has NO Maxwell reference, so it cannot produce a numerical accuracy claim.')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=ap.add_subparsers(dest='cmd',required=True)
    exp=sub.add_parser('export',help='Run on Windows with AEDT; export magnetic TET meshes and B reference')
    exp.add_argument('--project',required=True,help='Existing .aedt project')
    exp.add_argument('--design',required=True)
    exp.add_argument('--out',required=True)
    exp.add_argument('--setup',help='e.g. Setup1 : LastAdaptive')
    exp.add_argument('--version',help='e.g. 2026.1')
    exp.add_argument('--objects',nargs='+',help='Specific magnetic object names; default auto-detect')
    samp=exp.add_mutually_exclusive_group(required=True)
    samp.add_argument('--line-mm',help='x0,y0,z0:x1,y1,z1:count, units mm')
    samp.add_argument('--points-csv',help='CSV of x_m,y_m,z_m (meters), optional header')
    exp.add_argument('--overrides',help='JSON object/material overrides')
    exp.add_argument('--max-tiles',type=int,default=10000)
    exp.add_argument('--no-reference',action='store_true')
    exp.add_argument('--ignore-current-check',action='store_true',
                     help='Only if verified all currents are zero; current sources are not converted!')
    exp.add_argument('--new-desktop',action='store_true',help='Create a separate AEDT session')

    insp=sub.add_parser('inspect',help='Read export and estimate tensor memory')
    insp.add_argument('--out',required=True)

    native=sub.add_parser('import-native',help='Convert native AEDT 18.x disk bundle without Maxwell/PyAEDT')
    native.add_argument('--bundle',required=True)
    native.add_argument('--out',required=True)
    native_points=native.add_mutually_exclusive_group()
    native_points.add_argument('--points-csv');native_points.add_argument('--line-mm')
    native.add_argument('--overrides');native.add_argument('--max-tiles',type=int,default=10000)

    fem=sub.add_parser('solve-fem-bem',help='Offline P1/P0 scalar-potential solve; no Maxwell/PyAEDT required')
    fem.add_argument('--out',required=True)
    fem.add_argument('--result-prefix',default='fem_bem',help='New prefix for retry evidence, reusing the same imported mesh/cache')
    fem.add_argument('--boundary-cache',help='Optional shared numeric H cache; source/geometry/settings signatures are verified')
    fem.add_argument('--boundary',choices=['dense','cuda_hmatrix'],default='dense')
    fem.add_argument('--tol',type=float,default=1e-8);fem.add_argument('--maxiter',type=int,default=20)
    fem.add_argument('--linear-rtol',type=float,default=1e-6);fem.add_argument('--linear-cycles',type=int,default=10)
    fem.add_argument('--fem-preconditioner',choices=['lu','amg'],default='lu')
    fem.add_argument('--h-leaf',type=int,default=32);fem.add_argument('--h-eta',type=float,default=3.)
    fem.add_argument('--h-rtol',type=float,default=1e-6);fem.add_argument('--h-rank',type=int,default=64)
    fem.add_argument('--h-gib',type=float,default=6.);fem.add_argument('--h-storage',choices=['float64','float32'],default='float32')
    fem.add_argument('--h-far-order',type=int,default=4)
    fem.add_argument('--max-reference-error',type=float,help='Maximum relative L2 at native air samples; fail if exceeded')

    sol=sub.add_parser('solve',help='Run MagTense on exported magnetic tetrahedral mesh')
    sol.add_argument('--out',required=True)
    sol.add_argument('--max-tiles',type=int,default=10000)
    sol.add_argument('--max-error',type=float,default=1e-5)
    sol.add_argument('--max-it',type=int,default=500)
    sol.add_argument('--temperature-K',type=float,default=300)
    sol.add_argument('--dry-run',action='store_true')

    cmp=sub.add_parser('compare',help='Compare Maxwell and MagTense air B vectors in Tesla')
    cmp.add_argument('--out',required=True)
    cmp.add_argument('--prediction',default='magtense_B.csv',help='Predicted xyz,Bx,By,Bz CSV, default MagTense CPU')
    cmp.add_argument('--prefix',default='',help='Prefix for comparison.csv and metrics.json, e.g. gpu_')
    cmp.add_argument('--maxwell-csv',help='Optional CSV x,y,z,Bx,By,Bz in SI or Bx,By,Bz')
    dem=sub.add_parser('demo',help='Create minimal synthetic tetra cube; no Maxwell')
    dem.add_argument('--out',required=True)
    args=ap.parse_args(argv)
    try:
        if args.cmd=='export':
            from aedt_export import export_project
            export_project(args.project,args.design,args.out,points_csv=args.points_csv,
                line_mm=args.line_mm,setup=args.setup,version=args.version,objects=args.objects,
                overrides_path=args.overrides,max_tiles=args.max_tiles,
                no_reference=args.no_reference,ignore_current_check=args.ignore_current_check,
                new_desktop=args.new_desktop)
        elif args.cmd=='inspect':
            summary(Path(args.out).resolve())
        elif args.cmd=='import-native':
            from aedt_export import import_native_bundle
            model=import_native_bundle(args.bundle,args.out,points_csv=args.points_csv,line_mm=args.line_mm,
                overrides_path=args.overrides,max_tiles=args.max_tiles)
            print('Imported native disk bundle:',len(model['objects']),'magnetic objects,',model['total_magnetic_tetra'],'cells')
        elif args.cmd=='solve-fem-bem':
            command_fem_bem(args)
        elif args.cmd=='solve':
            command_solve(args)
        elif args.cmd=='compare':
            command_compare(args)
        elif args.cmd=='demo':
            command_demo(args)
    except Exception as e:
        print(f'ERROR [{type(e).__name__}]: {e}',file=sys.stderr)
        return 2
    return 0


if __name__=='__main__':
    sys.exit(main())
