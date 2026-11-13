"""Measure one real FMM attempt on the saved 2,718,864-cell magnetic mesh.

Source entrypoints: current aedt_magtense_bridge/aedt_export.py and gpu_solver.py.
Input: gap3mm_fmm_3m.aedt + native results, mesh_generation.json, and the original
485 air points. No synthetic mesh or copied solver implementation is used.
Run: C:/Projects/magrun/.venv/Scripts/python.exe -u run_fmm.py
Resume using existing export and append an attempt: add --solve-only
--fmm-neighbours cuda --fmm-farfield cuda. Prior mixed CPU/GPU evidence is retained.
Small real-mesh point-FMM validation: --validate-cuda-small; appends its all-point
comparisons at tolerance 1e-6 to fmm_verification.json (no Maxwell certification).
Fixed near-coefficient compression experiment: run with the SOLVER interpreter
and --compress-near. Uses this large export, complete global source neighbours
for sampled receiver patches in every magnetic object, and current tetra_fmm
AdaptiveNearCompression (CPU LAPACK or GPU SVD, packed CUDA matvec). Records storage including indices,
coefficient/field errors, construction and warm matvec wall times in the SAME
fmm_verification.json. Assertions: reconstructed Frobenius error <= requested
rtol + roundoff, and both held-out fields <= 10*rtol. No full-model cache, full
nonlinear solve, or Maxwell accuracy/speedup is certified by this experiment.
FEM-BEM feasibility topology census: --assess-fem-bem uses the SAME exported
magnetic TET4 corner geometry and NumPy/SciPy, without installing a new solver.
It counts exact-coordinate welded vertices, tetrahedral edges and exterior
faces, checks nondegeneracy/face incidence, and estimates scalar-P1 FEM CSR and
dense surface-BEM storage. It writes fem_bem_feasibility in fmm_verification.json.
This is a FULL magnetic-mesh census, NOT a field solve or GPU-memory benchmark;
quadratic Maxwell elements, added local-air mesh and AMG memory are excluded.
Small scalar FEM-BEM solve: --prototype-fem-bem (SOLVER Python, NumPy/SciPy).
Tests PM/nonlinear spheres at 32/128/512 boundary faces against the analytical
demagnetizing factor 1/3, weak and pointwise interface diagnostics, quadrature
refinement and preconditioning; then the retained 3,629-cell/485-point Maxwell
case at target quadrature orders 8 and 12. --fem-bem-synthetic-only omits the
real case and explicitly cannot certify Maxwell accuracy. Assertions and
declared thresholds are stored in append-only fem_bem_prototype_runs; failures
retain evidence and exit nonzero. CPU/dense boundary reference only, not H/H2,
boundary FMM, AMG or a large-model GPU performance measurement.
Live AEDT case audit: --validate-fem-bem-aedt opens the solved, unchanged
full_pair_reference.aedt via the legacy PyAEDT environment, reads its actual
Reference : LastAdaptive solution and verifies currents, material assignments,
CAD bounds, B-H knots, convergence and normal completion. It compares FEM-BEM
with freshly exported B/H at all 485 air points, all 3,629 material-cell centers
and both sides of all 2,382 boundary faces. No synthetic case or new Maxwell
solve is used. Results, arrays and hashes append to fem_bem_aedt_runs in the
same verification JSON. Initial acceptance: air B <=5%, material B <=10%,
boundary-flux B.n <=10%, coupled residual <=1e-8; failure exits nonzero.
Requires licensed AEDT 2022 R2/PyAEDT for export, the existing MagTense/CuPy/
FMM3D environment for solve, and substantial RAM/pagefile and GPU memory.
Outputs: large_export/ (solver input/results), large_fmm_run.json (status,
stage times, exit codes, sampled memory peaks), large_fmm_resources.csv, logs.
Success: exactly 2,718,864 exported magnetic cells, current solver convergence
at 1e-8, and its mandatory final native field checks. Failed stages exit nonzero
and remain failures in the report. No Maxwell accuracy or speedup is certified.
GPU board usage includes other applications; CuPy pool bytes are separately
reported by the worker. RSS is sampled externally, including owned descendants.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TOOL = REPO/'skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
EXPORT = ROOT/'large_export'
STATUS = ROOT/'large_fmm_worker.json'
REPORT = ROOT/'large_fmm_run.json'
LEGACY = Path('C:/Projects/magrun/.venv/Scripts/python.exe')
SOLVER = REPO/'skill-test/.venv/Scripts/python.exe'


def save(path, data):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')
    os.replace(temporary,path)


def validate_cuda_small():
    """Real 3,629-cell geometry; compare ALL point fields to FMM3D at eps=1e-12.

    This checks the GPU far-field implementation, not Maxwell accuracy or
    nonlinear convergence. Both physical saved M and seeded random M must
    achieve relative vector L2 error <=1e-6; any failure exits nonzero.
    """
    import cupy as cp
    import numpy as np
    from fmm3dpy import lfmm3d
    from tetra_fmm import _CUDAPointFMM
    from tetra_volume import quadrature
    source = REPO/'validation/20261107_repair_staged/full_pair_repaired_runtime'
    model = json.loads((source/'model.json').read_text(encoding='utf-8'))
    vertices = np.concatenate([np.load(source/o['mesh_npz'])['tetra_m'] for o in model['objects']])
    assert len(vertices)==3629
    q = quadrature(vertices).reshape(-1,3)
    scale = np.ptp(vertices.reshape(-1,3),axis=0).max()
    points = np.ascontiguousarray((q-vertices.min(axis=(0,1)))/scale)
    weights = abs(np.linalg.det(vertices[:,1:]-vertices[:,:1]))/24/scale**3
    magnetization = np.load(ROOT/'fmm_small_model.npz')['M_Apm']
    fmm = _CUDAPointFMM(points,eps=1e-6,order=10)
    checks = []
    for name,m in [('saved_physical_M',magnetization),
                   ('seeded_random_M',np.random.default_rng(7143).normal(size=magnetization.shape)*20000)]:
        moments = np.ascontiguousarray(np.repeat(m*weights[:,None],4,axis=0))
        reference = lfmm3d(eps=1e-12,sources=np.asfortranarray(points.T),
                          dipvec=np.asfortranarray(moments.T),pg=2)
        assert reference.ier==0 and np.isfinite(reference.grad).all()
        started = time.perf_counter()
        field = cp.asnumpy(fmm(moments))
        elapsed = time.perf_counter()-started
        expected = -reference.grad.T
        error = float(np.linalg.norm(field-expected)/max(np.linalg.norm(expected),1.))
        assert np.isfinite(error) and error<=1e-6,(name,error)
        checks.append({'magnetization':name,'all_point_relative_L2_error':error,
                       'tolerance':1e-6,'passed':True,'GPU_evaluation_seconds':elapsed})
    record = {'input_magnetic_cells':len(vertices),'all_point_count':len(points),
              'reference':'FMM3D point-dipole field, eps=1e-12; same source quadrature and moments',
              'checks':checks,'GPU_plan':fmm.info,'cupy_version':cp.__version__,
              'source_sha256':{name:hashlib.sha256((TOOL/name).read_bytes()).hexdigest()
                               for name in ('tetra_fmm.py','tetra_cuda.cu')},
              'reproduce':'C:/Projects/magrun/.venv/Scripts/python.exe run_fmm.py --validate-cuda-small',
              'scope':'Point far-field validation only; does not certify Maxwell error or nonlinear convergence'}
    path = ROOT/'fmm_verification.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['cuda_point_validation'] = record
    save(path,report)
    print(json.dumps(record,indent=2),flush=True)


def compress_near(args):
    """Bounded, reproducible H-matrix experiment on the actual large-mesh rows."""
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ.setdefault('CUPY_CACHE_DIR', str(ROOT/'.cupy_cache'))
    sys.path.insert(0, str(TOOL))
    dll = os.add_dll_directory('C:/Program Files/AnsysEM/v222/Win64')
    import cupy as cp
    import numpy as np
    import psutil
    from scipy.spatial import cKDTree
    from magtense.magstatics import Tiles
    from tetra_fmm import VolumeFMM, AdaptiveNearCompression, _CUDANeighbours, _neighbours

    started = time.perf_counter()
    process = psutil.Process()
    path = ROOT/'fmm_verification.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    record = {'state':'running', 'started_utc':datetime.now(timezone.utc).isoformat(),
              'pid':os.getpid(),'command':[sys.executable,*sys.argv],
              'hardware':{'GPU':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                          'RAM_bytes':psutil.virtual_memory().total,
                          'CPU':os.environ.get('PROCESSOR_IDENTIFIER')},
              'scope':'Sampled receiver patches; each includes ALL near source neighbours from the 2,718,864-cell mesh, across object boundaries. Not full-model compression.',
              'settings':vars(args), 'patches':[], 'cupy_version':cp.__version__,
              'source_sha256':{name:hashlib.sha256((TOOL/name).read_bytes()).hexdigest()
                               for name in ('tetra_fmm.py','tetra_cuda.cu')},
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'reference':'Uncompressed FP64 CUDA analytical near-correction coefficients, same source4/receiving256 quadrature and correction_factor=3 as the actual large solve',
              'error_scope':'Coefficient Frobenius and two held-out near-only fields; not full field, nonlinear residual or Maxwell error',
              'timing_scope':'GPU synchronized wall seconds, including Python gather/scatter/launch overhead; warm matvec median of three batches of five calls',
              'tolerance_sweep_policy':'When a looser tolerance accepts NO low-rank block anywhere, stricter tolerances reuse this exact no-compression decision (ranks cannot decrease). Shared search cost is retained as build_seconds; additional_build_seconds is zero. Each reconstruction, error and matvec timing is measured again.',
              'input_npz_sha256':{}}
    report.setdefault('adaptive_near_compression_runs', []).append(record)
    def checkpoint(**details):
        record.update(details, elapsed_seconds=time.perf_counter()-started)
        memory = process.memory_info()
        record['process_peak_RSS_bytes'] = getattr(memory,'peak_wset',memory.rss)
        record['process_peak_commit_bytes'] = getattr(memory,'peak_pagefile',memory.vms)
        record['max_CuPy_reserved_bytes'] = max(record.get('max_CuPy_reserved_bytes',0),cp.get_default_memory_pool().total_bytes())
        save(path,report)
        print('COMPRESSION '+json.dumps({**details,'elapsed_seconds':record['elapsed_seconds'],
                                         'completed_patches':len(record['patches'])}),flush=True)

    def measure(function, vector):
        for _ in range(2):
            function(vector)
        cp.cuda.Stream.null.synchronize()
        samples = []
        for _ in range(3):
            before = time.perf_counter()
            for _ in range(5):
                function(vector)
            cp.cuda.Stream.null.synchronize()
            samples.append((time.perf_counter()-before)/5)
        return {'median_seconds':float(np.median(samples)), 'batch_seconds_per_call':samples}

    try:
        model = json.loads((EXPORT/'model.json').read_text(encoding='utf-8'))
        vertices, slices = [], []
        count = 0
        for obj in model['objects']:
            mesh_path = EXPORT/obj['mesh_npz']
            record['input_npz_sha256'][obj['mesh_npz']] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
            part = np.load(mesh_path)['tetra_m']
            vertices.append(part)
            slices.append((obj['name'],count,count+len(part)))
            count += len(part)
        vertices = np.concatenate(vertices)
        assert count==2718864
        centres = vertices.mean(axis=1)
        radii = np.linalg.norm(vertices-centres[:,None],axis=2).max(axis=1)
        search = _CUDANeighbours(centres,radii)
        record['magnetic_cells'] = count
        checkpoint(phase='global_geometry_ready')
        rng = np.random.default_rng(981)
        visited = []
        for object_index,(name,lo,hi) in enumerate(slices):
            if object_index>=args.compress_object_count:
                break
            local_radii = radii[lo:hi]
            anchors = [int(rng.integers(lo,hi)),int(rng.integers(lo,hi)),
                       lo+int(local_radii.argmin()),lo+int(local_radii.argmax())]
            tree = cKDTree(centres[lo:hi])
            for anchor in anchors[:args.compress_patches_per_object]:
                receiver_count = min(args.compress_patch_cells,hi-lo)
                _,nearest = tree.query(centres[anchor],k=receiver_count)
                target_ids = lo+np.atleast_1d(nearest).astype(np.int64)
                while True:
                    global_rows = [row for _,rows in _neighbours(centres,radii,3.,32,groups=search,
                        targets=centres[target_ids],target_radii=radii[target_ids]) for row in rows]
                    union = np.unique(np.concatenate(global_rows))
                    pairs = sum(map(len,global_rows))
                    if 9*8*len(target_ids)*len(union)<=args.compress_patch_mib*1024**2 and pairs<=2000000:
                        break
                    if len(target_ids)<=1:
                        raise MemoryError('One complete near row exceeds the experiment patch budget')
                    target_ids = target_ids[:max(1,len(target_ids)//2)]
                source_ids = np.r_[target_ids,np.setdiff1d(union,target_ids)].astype(np.int64)
                order = np.argsort(source_ids)
                rows = [np.sort(order[np.searchsorted(source_ids[order],row)]).astype(np.int32) for row in global_rows]
                indices = np.concatenate(rows)
                pointer = np.r_[0,np.cumsum([len(row) for row in rows])].astype(np.int64)
                nr, ns = len(target_ids),len(source_ids)
                patch = {'object':name,'anchor_global_id':anchor,'receiver_count':nr,'source_count':ns,
                         'near_pairs':pairs,'receiver_global_ids':target_ids.tolist(),'tolerances':[]}
                checkpoint(phase='patch_coefficients',current_object=name,anchor_global_id=anchor,near_pairs=pairs)
                preparation = time.perf_counter()
                pv = vertices[source_ids]
                tiles = Tiles(n=ns,vertices=np.asfortranarray(pv.transpose(0,2,1)),tile_type=5,
                              M_rem=0,easy_axis=[1,0,0],mu_r_ea=1,mu_r_oa=1)
                operator = VolumeFMM(tiles,cp,cache_bytes=0,index_bytes=8*(ns+1),
                                     neighbours='cuda',farfield='cpu',batch=32)
                cp.cuda.Stream.null.synchronize()
                patch['local_geometry_seconds'] = time.perf_counter()-preparation
                before = time.perf_counter()
                blocks = cp.concatenate([operator._blocks(a,min(nr,a+32),rows=rows[a:a+32])
                                         for a in range(0,nr,32)])
                cp.cuda.Stream.null.synchronize()
                patch['coefficient_generation_seconds'] = time.perf_counter()-before
                di,dp = cp.asarray(indices),cp.asarray(pointer)
                reference_apply = lambda x: operator._apply_blocks(blocks,di,dp,x)
                dense = cp.zeros((3*nr,3*ns),dtype=cp.float64)
                rr = cp.asarray(np.repeat(np.arange(nr),np.diff(pointer)))
                dense.reshape(nr,3,ns,3)[rr,:,di,:] = blocks
                matrix_norm = float(cp.linalg.norm(dense))
                vectors = {'random':cp.asarray(np.random.default_rng(981+anchor).normal(size=(ns,3))*20000).ravel(),
                           'uniform':cp.tile(cp.asarray([20000.,-13000.,17000.]),ns)}
                references = {key:reference_apply(x) for key,x in vectors.items()}
                # Independent dense multiplication checks BSR assembly/application.
                for key,x in vectors.items():
                    cp.testing.assert_allclose(references[key],dense@x,rtol=1e-11,atol=1e-7)
                patch['uncompressed_BSR_apply'] = measure(reference_apply,vectors['random'])
                patch['uncompressed_BSR_bytes'] = blocks.nbytes+indices.nbytes+pointer.nbytes
                patch['six_FP32_BSR_bytes'] = pairs*(6*4+4)+pointer.nbytes
                # Also compare to the CURRENT streamed-vector implementation, not
                # just a cheap cached baseline. No full nonlinear solve is run.
                charges = cp.empty((ns,4),dtype=cp.float64)
                operator.module.get_function('fmm_face_charges')(((ns+127)//128,),(128,),
                    (operator.faces,vectors['random'],np.int32(ns),charges))
                cp.cuda.Stream.null.synchronize()
                before = time.perf_counter()
                streamed = operator._correct_vectors(0,nr,vectors['random'],charges,rows=rows)
                cp.cuda.Stream.null.synchronize()
                patch['streamed_vector_apply_seconds'] = time.perf_counter()-before
                cp.testing.assert_allclose(streamed.ravel(),references['random'],rtol=1e-10,atol=1e-6)
                compressed, built_tolerance = None, None
                for tolerance in args.compress_tolerances:
                    reuse = compressed is not None and not compressed.info['low_rank_blocks'] and tolerance<=built_tolerance
                    if not reuse:
                        compressed = AdaptiveNearCompression(blocks,indices,pointer,pv[:nr],pv,cp,
                            rtol=tolerance,leaf_size=args.compress_leaf_cells,eta=args.compress_eta,
                            admissibility=args.compress_admissibility,error_budget=args.compress_error_budget,
                            decomposition=args.compress_decomposition)
                        built_tolerance = tolerance
                    reconstructed = compressed.toarray()
                    coefficient_error = float(cp.linalg.norm(reconstructed-dense))/matrix_norm
                    assert coefficient_error<=tolerance*(1+1e-8)+1e-12,coefficient_error
                    errors = {}
                    for key,x in vectors.items():
                        actual = compressed@x
                        cp.testing.assert_allclose(actual,reconstructed@x,rtol=1e-10,atol=1e-6)
                        error = float(cp.linalg.norm(actual-references[key])/cp.linalg.norm(references[key]))
                        errors[key] = error
                        assert np.isfinite(error) and error<=10*tolerance,(key,error,tolerance)
                    timing = measure(lambda x: compressed@x,vectors['random'])
                    summary = dict(compressed.info)
                    summary.update(rtol=tolerance,decision_reused_from_rtol=built_tolerance if reuse else None,
                                   additional_build_seconds=0. if reuse else summary['build_seconds'])
                    ranks = summary.pop('ranks')
                    summary['rank_histogram'] = {str(k):ranks.count(k) for k in sorted(set(ranks))}
                    summary.update(coefficient_relative_Frobenius_error=coefficient_error,
                                   near_field_relative_L2_errors=errors,
                                   compressed_apply=timing,
                                   apply_speedup_vs_cached_BSR=patch['uncompressed_BSR_apply']['median_seconds']/timing['median_seconds'],
                                   apply_speedup_vs_streamed_vectors=patch['streamed_vector_apply_seconds']/timing['median_seconds'],
                                   ratio_vs_six_FP32_BSR=patch['six_FP32_BSR_bytes']/compressed.nbytes)
                    patch['tolerances'].append(summary)
                    print('PATCH '+json.dumps({'object':name,'anchor':anchor,'rtol':tolerance,
                        'ratio':summary['compression_ratio'],'error':coefficient_error,
                        'build_seconds':summary['build_seconds'],'apply_seconds':timing['median_seconds'],
                        'speedup_vs_cached':summary['apply_speedup_vs_cached_BSR']}),flush=True)
                    del reconstructed
                record['patches'].append(patch)
                visited.extend(target_ids.tolist())
                checkpoint(phase='patch_complete',current_object=name)
                del operator,tiles,blocks,dense,di,dp,rr,vectors,references,charges,streamed,compressed
                cp.get_default_memory_pool().free_all_blocks()
        record['unique_receivers'] = len(set(visited))
        record['receiver_fraction'] = len(set(visited))/count
        record['sampled_near_pairs_with_patch_repetition'] = sum(p['near_pairs'] for p in record['patches'])
        record['aggregate'] = []
        for tolerance in args.compress_tolerances:
            items = [(p,next(t for t in p['tolerances'] if t['rtol']==tolerance)) for p in record['patches']]
            original = sum(p['uncompressed_BSR_bytes'] for p,t in items)
            compressed = sum(t['compressed_bytes'] for p,t in items)
            raw_time = sum(p['uncompressed_BSR_apply']['median_seconds'] for p,t in items)
            compressed_time = sum(t['compressed_apply']['median_seconds'] for p,t in items)
            record['aggregate'].append({'rtol':tolerance,'original_BSR_bytes':original,
                'compressed_bytes':compressed,'compression_ratio':original/compressed,
                'ratio_vs_six_FP32_BSR':sum(p['six_FP32_BSR_bytes'] for p,t in items)/compressed,
                'max_coefficient_relative_Frobenius_error':max(t['coefficient_relative_Frobenius_error'] for p,t in items),
                'max_held_out_near_field_relative_L2_error':max(max(t['near_field_relative_L2_errors'].values()) for p,t in items),
                'sum_compression_build_seconds':sum(t['build_seconds'] for p,t in items),
                'sum_additional_build_seconds_in_sweep':sum(t['additional_build_seconds'] for p,t in items),
                'sum_coefficient_generation_seconds':sum(p['coefficient_generation_seconds'] for p,t in items),
                'sum_cached_BSR_apply_seconds':raw_time,'sum_compressed_apply_seconds':compressed_time,
                'cached_BSR_apply_speedup':raw_time/compressed_time,
                'sum_streamed_vector_apply_seconds':sum(p['streamed_vector_apply_seconds'] for p,t in items),
                'streamed_vector_apply_speedup':sum(p['streamed_vector_apply_seconds'] for p,t in items)/compressed_time,
                'low_rank_blocks':sum(t['low_rank_blocks'] for p,t in items),
                'SVD_attempts':sum(t['svd_attempts'] for p,t in items),
                'retained_raw_pairs':sum(t['retained_raw_pairs'] for p,t in items)})
        checkpoint(state='completed',phase='complete')
        print(json.dumps(record['aggregate'],indent=2),flush=True)
    except BaseException as error:
        checkpoint(state='failed',phase='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise


def assess_fem_bem():
    """Measure full exported topology before choosing a new discretization."""
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    import numpy as np
    import psutil
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    started = time.perf_counter()
    process = psutil.Process()
    path = ROOT/'fmm_verification.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    record = {'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Full magnetic TET4 corner geometry only; scalar-P1 structural counts, not a FEM/BEM field solve',
        'command':[sys.executable,*sys.argv],
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'input_npz_sha256':{},'stages':{},
        'coordinate_welding':'Exact FP64 coordinate equality; no tolerance snapping or source mesh modification',
        'excluded_from_FEM_storage':['AMG hierarchy','vectors and Krylov basis','element gradients and nonlinear tangents','BEM operators','local-air remeshing','quadratic Maxwell field space']}
    report['fem_bem_feasibility'] = record
    def progress(phase,**details):
        record.update(phase=phase,**details,elapsed_seconds=time.perf_counter()-started)
        memory = process.memory_info()
        record['process_peak_RSS_bytes'] = getattr(memory,'peak_wset',memory.rss)
        record['process_peak_commit_bytes'] = getattr(memory,'peak_pagefile',memory.vms)
        save(path,report)
        print('FEM_BEM '+json.dumps({'phase':phase,**details,'elapsed_seconds':record['elapsed_seconds']}),flush=True)
    try:
        model = json.loads((EXPORT/'model.json').read_text(encoding='utf-8'))
        parts=[]
        for obj in model['objects']:
            meshpath=EXPORT/obj['mesh_npz']
            record['input_npz_sha256'][obj['mesh_npz']] = hashlib.sha256(meshpath.read_bytes()).hexdigest()
            parts.append(np.load(meshpath)['tetra_m'])
        tetra = np.concatenate(parts)
        del parts
        nt = len(tetra)
        assert nt==model['total_magnetic_tetra']==2718864
        determinants = np.linalg.det(tetra[:,1:]-tetra[:,:1])
        assert np.all(np.isfinite(determinants)) and np.all(determinants!=0)
        record['magnetic_volume_m3'] = float(np.abs(determinants).sum()/6)
        del determinants
        progress('vertex_welding',magnetic_tetrahedra=nt)
        before=time.perf_counter()
        nodes,inverse=np.unique(tetra.reshape(-1,3),axis=0,return_inverse=True)
        conn=inverse.reshape(-1,4).astype(np.int32)
        del tetra,inverse
        nv=len(nodes)
        record['stages']['vertex_welding_seconds']=time.perf_counter()-before
        progress('face_census',scalar_P1_nodes=nv)
        before=time.perf_counter()
        faces=conn[:,np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])].reshape(-1,3)
        faces.sort(axis=1)
        unique_faces,incidence=np.unique(faces,axis=0,return_counts=True)
        assert incidence.sum()==4*nt and incidence.max()<=2
        boundary=unique_faces[incidence==1]
        internal_faces=int(np.count_nonzero(incidence==2))
        assert len(boundary)+2*internal_faces==4*nt
        boundary_nodes=np.unique(boundary)
        del faces,unique_faces,incidence
        record['stages']['face_census_seconds']=time.perf_counter()-before
        progress('edge_census',boundary_triangles=len(boundary),boundary_vertices=len(boundary_nodes))
        before=time.perf_counter()
        edges=conn[:,np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])].reshape(-1,2)
        edges.sort(axis=1)
        edges=np.unique(edges,axis=0)
        ne=len(edges)
        nnz=nv+2*ne
        graph=csr_matrix((np.ones(ne,dtype=np.int8),(edges[:,0],edges[:,1])),shape=(nv,nv))
        components,labels=connected_components(graph,directed=False)
        record['volume_connected_component_sizes']=np.bincount(labels).tolist()
        del graph,labels,edges,conn
        surface_edges=boundary[:,np.array([[0,1],[0,2],[1,2]])].reshape(-1,2)
        surface_edges.sort(axis=1)
        surface_edges,surface_incidence=np.unique(surface_edges,axis=0,return_counts=True)
        open_edges=int(np.count_nonzero(surface_incidence==1))
        junction_edges=int(np.count_nonzero(surface_incidence>2))
        record['stages']['edge_census_seconds']=time.perf_counter()-before
        record.update(unique_volume_edges=ne,volume_connected_components=components,
            interior_faces=internal_faces,boundary_edges=len(surface_edges),
            boundary_open_edges=open_edges,boundary_junction_edges=junction_edges,
            closed_manifold_boundary=(open_edges==junction_edges==0),
            scalar_P1_FEM_structural_nnz=nnz,
            scalar_P1_FEM_CSR_FP64_int32_bytes=nnz*(8+4)+(nv+1)*4,
            scalar_P1_FEM_single_vector_FP64_bytes=nv*8,
            scalar_P2_geometric_DOFs=nv+ne,
            direct_material_surface_P0_flux_DOFs=len(boundary),
            direct_material_surface_P1_trace_DOFs=len(boundary_nodes),
            P1_FEM_plus_P0_flux_DOFs=nv+len(boundary),
            one_dense_P0_BEM_FP64_matrix_bytes=len(boundary)**2*8,
            one_dense_P1_BEM_FP64_matrix_bytes=len(boundary_nodes)**2*8,
            assertions={'nondegenerate_finite_tetrahedra':True,
                        'each_volume_face_has_one_or_two_incident_tetrahedra':True,
                        'four_faces_per_tetrahedron_accounted_for':True},
            interpretation='CSR is the structural graph of a scalar-P1 volume stiffness matrix, not assembled matrix values or total solve memory. Boundary faces are the exact-coordinate welded union of exported magnetic regions; an artificial outer air coupling surface must be remeshed and counted separately.')
        progress('complete',state='completed')
    except BaseException as error:
        progress('failed',state='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise


def prototype_fem_bem(synthetic_only=False):
    """Small-model acceptance of the current-source FEM--BEM reference backend."""
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    import numpy as np
    import psutil
    from scipy.optimize import brentq
    from scipy.spatial import ConvexHull
    from fem_bem import FEMBEM, material_laws
    from gpu_solver import MU0
    from bridge import load_points, check_points_in_air, read_maxwell_fld
    started = time.perf_counter()
    path = ROOT/'fmm_verification.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    limits = dict(fine_sphere_volume_H_relative_L2=.015,
        sphere_refinement_error_ratio=.5, fine_interface_potential_relative_L2=.05,
        fine_interface_normal_flux_relative_L2=.15,
        quadrature_field_relative_L2=.005, Maxwell_air_B_relative_L2=.10,
        weak_relative_residual=1e-8)
    record = dict(state='running', started_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], limits=limits, synthetic_only=synthetic_only,
        scope='CPU FP64 sparse P1 / dense P0 Galerkin reference; no large solve or GPU speedup certification',
        source_sha256={name:hashlib.sha256((TOOL/name).read_bytes()).hexdigest()
                       for name in ('fem_bem.py','gpu_solver.py')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analytical_reference='Uniform sphere H_internal=H0-M(H_internal)/3; '
            'PM recoil H_internal=-Br/(mu0*(mu_r+2)); exterior dipole potential',
        input_sha256={}, spheres=[], real_mesh=[])
    report.setdefault('fem_bem_prototype_runs', []).append(record)
    def checkpoint(phase, **extra):
        record.update(phase=phase, elapsed_seconds=time.perf_counter()-started, **extra)
        memory = psutil.Process().memory_info()
        record['process_peak_RSS_bytes'] = getattr(memory, 'peak_wset', memory.rss)
        save(path, report)
        print('FEM_BEM_PROTOTYPE '+json.dumps(dict(phase=phase, elapsed_seconds=record['elapsed_seconds'], **extra)), flush=True)
    def relative(error, reference, weights=None):
        if weights is None:
            return float(np.linalg.norm(error)/np.linalg.norm(reference))
        return float(np.sqrt(np.sum(weights[:, None]*error**2)/np.sum(weights[:, None]*reference**2)))
    def diagnostics(s, u, q, H0):
        _, B, *_ = s.fields(u, H0)
        inside = np.einsum('fi,fi->f', B[s.owners]/MU0, s.normals)
        outside = s.normals @ H0-q
        centers = (s.tri.mean(axis=1)+2e-6*s.normals)*s.length+s.origin
        trace = u[s.faces].mean(axis=1)*s.length
        exterior = s.potential(u, q, centers)
        return dict(interface_potential_relative_L2=relative((exterior-trace)[:, None], trace[:, None], s.areas),
            interface_normal_flux_relative_L2=relative((outside-inside)[:, None], inside[:, None], s.areas),
            interface_scope='Potential at outward offset 2e-6*L versus P1 centroid trace; '
                'raw owner-element B.n versus mu0*(H0.n-q), not a pointwise-enforced flux',
            component_relative_net_flux=[float(abs(s.areas[ix] @ q[ix]) /
                max(np.sum(s.areas[ix]*abs(q[ix])), 1e-30))
                for k in range(s.info['components'])
                for ix in [s.component_ids[s.faces[:, 0]] == k]])
    try:
        checkpoint('start')
        # Synthetic sphere geometry only: no Maxwell or GPU claim is attached.
        vertices = np.r_[np.eye(3), -np.eye(3)]
        vertices /= np.linalg.norm(vertices, axis=1)[:, None]
        steel = dict(kind='nonlinear', bh_H_Apm_B_T=[[0,0],[300,1.9],[800,2.1],
            [1600,2.2],[4000,2.255],[8000,2.28],[16000,2.3],[200000,2.53122121930421]])
        pm = dict(kind='permanent', br_T=1.2, mu_r=1.05, direction=[0,0,1])
        h_knots, b_knots = np.asarray(steel['bh_H_Apm_B_T']).T
        m_knots = b_knots/MU0-h_knots
        applied = 560000.
        h_steel = brentq(lambda h: h+np.interp(h, h_knots, m_knots)/3-applied, 0, applied)
        previous = {}
        for level in range(4):
            faces = ConvexHull(vertices).simplices
            if level:
                tetra = np.concatenate((np.zeros((len(faces),1,3)),vertices[faces]),axis=1)*1e-3
                for name, properties, H0, exact_h in [
                    ('permanent_sphere', pm, np.zeros(3), -pm['br_T']/(MU0*(pm['mu_r']+2))),
                    ('nonlinear_steel_sphere', steel, np.array([0.,0,applied]), h_steel)]:
                    s = FEMBEM(tetra, material_laws([properties], np.zeros(len(tetra),dtype=int)), quadrature_order=12)
                    u, q, info = s.solve(H_external=H0, tol=1e-9)
                    H = s.fields(u, H0)[0]
                    expected = np.tile([0.,0,exact_h], (len(tetra),1))
                    # For high-mu steel, H is tiny near a B-H knee. The sphere
                    # mesh must converge in H; B alone can conceal its error.
                    error = relative(H-expected, expected, s.volumes)
                    entry = dict(case=name, level=level, internal_H_reference_Apm=exact_h,
                        volume_H_relative_L2=error, solver=info, **diagnostics(s,u,q,H0))
                    record['spheres'].append(entry)
                    checkpoint(f'{name}_{len(faces)}', volume_H_relative_L2=error)
                    assert info['residual'] < limits['weak_relative_residual']
                    assert info['boundary_weak_relative_residual'] < limits['weak_relative_residual']
                    assert max(entry['component_relative_net_flux']) < 1e-8
                    if name in previous:
                        assert error < limits['sphere_refinement_error_ratio']*previous[name], (name, error, previous[name])
                    previous[name] = error
                    if level == 3:
                        assert error < limits['fine_sphere_volume_H_relative_L2'], (name, error)
                        assert entry['interface_potential_relative_L2'] < limits['fine_interface_potential_relative_L2']
                        assert entry['interface_normal_flux_relative_L2'] < limits['fine_interface_normal_flux_relative_L2']
                    if level == 2 and name == 'permanent_sphere':
                        un, qn, plain = s.solve(preconditioner='none')
                        assert relative(un-u, u) < 1e-7
                        assert plain['total_gmres_iterations'] > info['total_gmres_iterations']
                        entry['unpreconditioned_gmres_iterations'] = plain['total_gmres_iterations']
                        coarse = FEMBEM(tetra, s.laws, quadrature_order=8)
                        uc, qc, _ = coarse.solve()
                        change = relative(coarse.fields(uc)[0]-H, H, s.volumes)
                        entry['quadrature_8_to_12_H_relative_L2'] = change
                        assert change < limits['quadrature_field_relative_L2']
            mids = (vertices[faces]+np.roll(vertices[faces], -1, axis=1)).reshape(-1,3)/2
            vertices = np.unique(np.r_[vertices, mids], axis=0)
            vertices /= np.linalg.norm(vertices, axis=1)[:,None]
        if not synthetic_only:
            folder = REPO/'validation/20261107_repair_staged/full_pair_repaired_runtime'
            model = json.loads((folder/'model.json').read_text(encoding='utf-8'))
            assert model['total_magnetic_tetra'] == 3629
            assert model['reference_source']['current_A_turn'] == 0
            parts, material_ids = [], []
            for k, obj in enumerate(model['objects']):
                meshpath = folder/obj['mesh_npz']
                record['input_sha256'][str(meshpath.relative_to(REPO))] = hashlib.sha256(meshpath.read_bytes()).hexdigest()
                parts.append(np.load(meshpath)['tetra_m'])
                material_ids.extend([k]*len(parts[-1]))
            for filename in ('model.json',model['points'],model['maxwell_field']):
                file = folder/filename
                record['input_sha256'][str(file.relative_to(REPO))] = hashlib.sha256(file.read_bytes()).hexdigest()
            tetra = np.concatenate(parts)
            laws = material_laws([o['properties'] for o in model['objects']], material_ids)
            points = load_points(folder, model)
            reference, coordinate_scale = read_maxwell_fld(folder/model['maxwell_field'], points)
            assert len(points) == 485 and np.isfinite(reference).all()
            first_B = None
            for order in (8, 12):
                checkpoint(f'real_mesh_assembly_q{order}')
                s = FEMBEM(tetra, laws, quadrature_order=order)
                checkpoint(f'real_mesh_solve_q{order}', setup_seconds=s.info['setup_seconds'])
                u, q, info = s.solve(tol=1e-9, maxiter=60)
                B, field_info = s.air_field(u, q, points,
                    validate_air=lambda p: check_points_in_air(folder, model, p))
                error = relative(B-reference, reference)
                entry = dict(quadrature_order=order, solver=info, air_gradient=field_info,
                    points=len(points), Maxwell_coordinate_scale_to_m=coordinate_scale,
                    Maxwell_air_B_relative_L2=error,
                    rms_absolute_B_error_T=float(np.sqrt(np.mean(np.sum((B-reference)**2,axis=1)))),
                    max_absolute_B_error_T=float(np.linalg.norm(B-reference,axis=1).max()),
                    **diagnostics(s,u,q,np.zeros(3)))
                record['real_mesh'].append(entry)
                checkpoint(f'real_mesh_complete_q{order}', Maxwell_air_B_relative_L2=error)
                assert info['residual'] < limits['weak_relative_residual']
                assert info['boundary_weak_relative_residual'] < limits['weak_relative_residual']
                assert max(entry['component_relative_net_flux']) < 1e-8
                assert error < limits['Maxwell_air_B_relative_L2'], error
                if first_B is not None:
                    entry['quadrature_8_to_12_air_B_relative_L2'] = relative(B-first_B, B)
                    assert entry['quadrature_8_to_12_air_B_relative_L2'] < limits['quadrature_field_relative_L2']
                first_B = B
        checkpoint('complete', state='completed', passed=True)
    except BaseException as error:
        checkpoint('failed', state='failed', passed=False,
                   error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise


def fem_bem_aedt_export(request_path):
    """Native-environment worker: inspect solved AEDT and export exact queries."""
    import tempfile
    import numpy as np
    from aedt_export import classify_material
    from aedtplt import get_unit_scale
    from bridge import read_maxwell_fld
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    project = Path(request['project'])
    before = hashlib.sha256(project.read_bytes()).hexdigest()
    result = dict(state='running', project=str(project), project_sha256_before=before,
                  design=request['design'], solution=request['setup']+' : LastAdaptive')
    app = None
    try:
        os.environ['ANSYSEM_ROOT222'] = r'C:\Program Files\AnsysEM\v222\Win64'
        from pyaedt import Maxwell3d
        from pyaedt.generic.settings import settings
        settings.use_grpc_api = True
        settings.enable_error_handler = False
        app = Maxwell3d(projectname=str(project), designname=request['design'],
            specified_version='2022.2', non_graphical=True, new_desktop_session=True, close_on_exit=False)
        model = json.loads(Path(request['model']).read_text(encoding='utf-8'))
        result['AEDT_version'] = app.aedt_version_id
        result['currents'] = {}
        for boundary in app.boundaries:
            kind = boundary.type.lower()
            if 'current' in kind:
                value = str(boundary.props['Current']).strip()
                assert float(value.replace('A', '').strip()) == 0, (boundary.name, value)
                result['currents'][boundary.name] = value
            elif any(t in kind for t in ('winding', 'coil', 'circuit')):
                raise ValueError(f'Unsupported excitation during source audit: {boundary.name} ({kind})')
        assert set(result['currents']) == {'A_AmpereTurns', 'B_AmpereTurns'}
        unit = get_unit_scale(app.modeler.model_units)
        result['materials_and_geometry'] = []
        for obj in model['objects']:
            actual_obj = app.modeler[obj['name']]
            material = actual_obj.material_name
            assert material.lower() == obj['material'].lower(), (obj['name'], material)
            actual = classify_material(app.materials[material])
            expected = obj['properties']
            assert actual['kind'] == expected['kind']
            for key in expected.keys()-{'kind'}:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12)
            bounds = np.asarray(actual_obj.bounding_box)*unit
            stats = obj['mesh_stats']
            np.testing.assert_allclose(bounds, stats['bounds_min_m']+stats['bounds_max_m'], rtol=1e-10, atol=1e-12)
            result['materials_and_geometry'].append(dict(object=obj['name'], material=material,
                properties=actual, bounding_box_m=bounds.tolist(), matches_export=True))
        setup = app.get_setup(request['setup'])
        result['setup'] = dict(setup.props)
        assert not setup.props['SmoothBHCurve'], 'The FEM material law does not smooth B-H knots'
        with tempfile.TemporaryDirectory(prefix='magtense_aedt_fields_') as scratch:
            scratch = Path(scratch)
            for method, name in [('export_convergence','convergence.txt'), ('export_profile','profile.txt'),
                                 ('export_mesh_stats','mesh_stats.txt')]:
                keyword = 'mesh_path' if method=='export_mesh_stats' else 'file_path'
                getattr(app, method)(request['setup'], **{keyword:str(scratch/name)})
                result[name] = (scratch/name).read_text(errors='replace')
            assert 'Normal Completion' in result['profile.txt']
            rows = []
            for line in result['convergence.txt'].splitlines():
                if re.match(r'^\s*\d+\|', line):
                    values = [v.strip() for v in line.split('|')]
                    rows.append(dict(pass_number=int(values[0]), tetrahedra=int(values[1]),
                        energy_error_percent=float(values[3]),
                        delta_energy_percent=None if values[4]=='N/A' else float(values[4])))
            assert len(rows)>=3 and all(r['energy_error_percent']<=.5 and
                r['delta_energy_percent'] is not None and r['delta_energy_percent']<=.5 for r in rows[-2:])
            result['adaptive_passes'] = rows
            result['fields'] = {}
            for name, query in request['queries'].items():
                points = np.asarray(query['points_m'])
                point_file = scratch/(name+'.pts')
                with point_file.open('w') as stream:
                    stream.write('Unit=mm\n')
                    np.savetxt(stream, points*1000, fmt='%.15g')
                output = scratch/(name+'.fld')
                assert app.post.export_field_file(query['quantity'], solution=result['solution'],
                    output_dir=str(output), sample_points_file=str(point_file),
                    export_with_sample_points=True, export_in_si_system=True,
                    reference_coordinate_system='Global', export_field_in_reference=True)
                values, scale = read_maxwell_fld(output, points)
                assert scale == 1 and np.isfinite(values).all()
                result['fields'][name] = dict(quantity=query['quantity'], point_count=len(points),
                    coordinate_scale_to_m=scale, values=values.tolist(),
                    fld_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        result.update(state='completed', solved_again=False)
    except BaseException as error:
        result.update(state='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        raise
    finally:
        try:
            if app is not None:
                app.release_desktop(close_projects=True, close_desktop=True)
        finally:
            result['project_sha256_after'] = hashlib.sha256(project.read_bytes()).hexdigest()
            result['project_unchanged'] = result['project_sha256_after'] == before
            save(request_path.with_suffix('.result.json'), result)
        assert result['project_unchanged'], 'Reading the reference modified the AEDT project'


def validate_fem_bem_aedt():
    """Direct AEDT verification, including material and interface field evidence."""
    import tempfile
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    import numpy as np
    import psutil
    from fem_bem import FEMBEM, material_laws
    from gpu_solver import MU0
    from bridge import load_points, check_points_in_air, read_maxwell_fld
    started = time.perf_counter()
    path = ROOT/'fmm_verification.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    folder = REPO/'validation/20261107_repair_staged/full_pair_repaired_runtime'
    model = json.loads((folder/'model.json').read_text(encoding='utf-8'))
    source = model['reference_source']
    limits = dict(air_B_relative_L2=.05, material_B_relative_L2=.10,
        boundary_normal_B_relative_L2=.10, coupled_relative_residual=1e-8,
        fresh_vs_saved_Maxwell_air_relative_L2=1e-9)
    record = dict(state='running', started_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv], source_sha256={name:hashlib.sha256((TOOL/name).read_bytes()).hexdigest()
            for name in ('fem_bem.py','gpu_solver.py','aedt_export.py')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limits=limits,
        reference='Live AEDT solved full_pair_reference.aedt, Gap_3mm_CCW, Reference : LastAdaptive',
        scope='Real AEDT case only; CPU FEM-BEM; no Maxwell re-solve or large-model speed claim',
        input_sha256={}, native_audit={}, metrics={})
    report.setdefault('fem_bem_aedt_runs', []).append(record)
    def checkpoint(phase, **extra):
        record.update(phase=phase, elapsed_seconds=time.perf_counter()-started, **extra)
        memory = psutil.Process().memory_info()
        record['solver_process_peak_RSS_bytes'] = getattr(memory,'peak_wset',memory.rss)
        save(path, report)
        print('FEM_BEM_AEDT '+json.dumps(dict(phase=phase, **extra)), flush=True)
    def metrics(predicted, expected, weights=None):
        error = np.asarray(predicted)-expected
        e = np.linalg.norm(error,axis=1) if error.ndim==2 else abs(error)
        r = np.linalg.norm(expected,axis=1) if expected.ndim==2 else abs(expected)
        w = np.ones(len(e)) if weights is None else np.asarray(weights)
        return dict(relative_L2=float(np.sqrt(np.sum(w*e**2)/max(np.sum(w*r**2),1e-300))),
            rms_absolute=float(np.sqrt(np.sum(w*e**2)/np.sum(w))), max_absolute=float(e.max()))
    try:
        parts, ids = [], []
        for k, obj in enumerate(model['objects']):
            mesh = folder/obj['mesh_npz']
            record['input_sha256'][str(mesh.relative_to(REPO))] = hashlib.sha256(mesh.read_bytes()).hexdigest()
            parts.append(np.load(mesh)['tetra_m'])
            ids.extend([k]*len(parts[-1]))
        for file in (folder/'model.json', folder/model['points'], Path(source['project'])):
            record['input_sha256'][str(file.relative_to(REPO))] = hashlib.sha256(file.read_bytes()).hexdigest()
        tetra, ids = np.concatenate(parts), np.asarray(ids)
        points = load_points(folder, model)
        assert len(tetra)==3629 and len(points)==485
        checkpoint('assemble_reference_boundary')
        s = FEMBEM(tetra, material_laws([o['properties'] for o in model['objects']], ids), quadrature_order=8)
        checkpoint('solve_FEM_BEM', boundary_setup_seconds=s.info['setup_seconds'])
        u, q, info = s.solve(tol=1e-9, maxiter=60)
        record['solver'] = info
        centers = s.tri.mean(axis=1)*s.length+s.origin
        # An owner-based altitude cap keeps each inward probe inside its tetra.
        altitude = 3*s.volumes[s.owners]/s.areas*s.length
        offsets = np.minimum(2e-6*s.length, .02*altitude)
        inside, outside = centers-offsets[:,None]*s.normals, centers+offsets[:,None]*s.normals
        raw_owner = tetra[s.owners]
        local = np.linalg.solve((raw_owner[:,1:]-raw_owner[:,:1]).transpose(0,2,1),
                               (inside-raw_owner[:,0])[...,None])[...,0]
        assert np.all(local>0) and np.all(local.sum(axis=1)<1)
        check_points_in_air(folder, model, outside)
        Bpoints = np.r_[points,tetra.mean(axis=1),inside,outside]
        queries = dict(B=dict(quantity='B',points_m=Bpoints.tolist()),
                       H=dict(quantity='H',points_m=tetra.mean(axis=1).tolist()))
        with tempfile.TemporaryDirectory(prefix='magtense_fem_bem_aedt_') as scratch:
            request_path = Path(scratch)/'queries.json'
            save(request_path, dict(project=source['project'],design=source['design'],setup='Reference',
                model=str(folder/'model.json'),queries=queries))
            checkpoint('read_live_AEDT_fields', air_points=len(points), material_points=len(tetra),
                       interface_faces=len(s.faces))
            process = subprocess.run([str(LEGACY),'-u',str(Path(__file__).resolve()),
                '--fem-bem-aedt-worker',str(request_path)], cwd=REPO,
                capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
            response = request_path.with_suffix('.result.json')
            if response.exists():
                record['native_audit'] = json.loads(response.read_text(encoding='utf-8'))
            record['native_worker_exit_code'] = process.returncode
            if process.returncode:
                record['native_worker_output_tail'] = (process.stdout+process.stderr)[-10000:]
                raise RuntimeError('Live AEDT audit/export failed; see native audit and worker output')
        native = record['native_audit']
        assert native['state']=='completed' and native['project_unchanged']
        Bnative = np.asarray(native['fields']['B']['values'])
        Hair = np.asarray(native['fields']['H']['values'])
        count, cells, faces = len(points), len(tetra), len(s.faces)
        Bair, Bmaterial, Bin, Bout = np.split(Bnative, [count,count+cells,count+cells+faces])
        old_reference, _ = read_maxwell_fld(folder/model['maxwell_field'],points)
        predicted_air, gradient_info = s.air_field(u,q,points,
            validate_air=lambda p:check_points_in_air(folder,model,p))
        Hmaterial, Bfem, *_ = s.fields(u)
        normal_inside = np.einsum('fi,fi->f', Bin,s.normals)
        normal_outside = np.einsum('fi,fi->f', Bout,s.normals)
        normal_cell = np.einsum('fi,fi->f', Bfem[s.owners],s.normals)
        result = record['metrics']
        result.update(air_B_T=metrics(predicted_air,Bair), material_B_T=metrics(Bfem,Bmaterial,s.volumes),
            material_H_Apm=metrics(Hmaterial,Hair,s.volumes),
            fresh_vs_saved_Maxwell_air=metrics(Bair,old_reference),
            Maxwell_interface_normal_B_jump_T=metrics(normal_outside,normal_inside,s.areas),
            FEM_boundary_normal_B_vs_Maxwell_T=metrics(-MU0*q,normal_outside,s.areas),
            FEM_cell_normal_B_vs_Maxwell_T=metrics(normal_cell,normal_inside,s.areas))
        result['per_object'] = []
        for k,obj in enumerate(model['objects']):
            ix, fx = ids==k, ids[s.owners]==k
            result['per_object'].append(dict(object=obj['name'],kind=obj['properties']['kind'],
                material_B_T=metrics(Bfem[ix],Bmaterial[ix],s.volumes[ix]),
                material_H_Apm=metrics(Hmaterial[ix],Hair[ix],s.volumes[ix]),
                boundary_normal_B_T=metrics((-MU0*q)[fx],normal_outside[fx],s.areas[fx])))
        record['air_gradient'] = gradient_info
        record['query_layout'] = dict(B_order=['485 air points','3629 cell centers','2382 inside face probes','2382 outside face probes'],
            interface_offset_m_min=float(offsets.min()),interface_offset_m_max=float(offsets.max()),
            probe_rule='area centroid +/- outward unit normal * min(2e-6*L,0.02*owner altitude)')
        record['predictions'] = dict(air_B_T=predicted_air.tolist(),material_B_T=Bfem.tolist(),
            material_H_Apm=Hmaterial.tolist(),boundary_normal_B_T=(-MU0*q).tolist(),
            normal_vectors=s.normals.tolist(),physical_face_areas_m2=(s.areas*s.length**2).tolist())
        checks = dict(air_B=result['air_B_T']['relative_L2']<=limits['air_B_relative_L2'],
            material_B=result['material_B_T']['relative_L2']<=limits['material_B_relative_L2'],
            interface_flux=result['FEM_boundary_normal_B_vs_Maxwell_T']['relative_L2']<=limits['boundary_normal_B_relative_L2'],
            weak_residual=info['residual']<=limits['coupled_relative_residual'],
            saved_reference=result['fresh_vs_saved_Maxwell_air']['relative_L2']<=limits['fresh_vs_saved_Maxwell_air_relative_L2'])
        checkpoint('complete',state='completed',passed=all(checks.values()),acceptance_checks=checks)
        assert all(checks.values()), f'AEDT case acceptance failed: {checks}'
    except BaseException as error:
        checkpoint('failed',state='failed',passed=False,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise


def diagnose_fem_bem_aedt():
    """Read-only follow-up of a retained AEDT run; no repeated BEM assembly.

    Independent reference checks: local B(H) consistency and interface-probe
    offsets 10/100 times the original offset, bounded to remain in the owner.
    These diagnostics do not change or override the baseline acceptance.
    """
    import tempfile
    import numpy as np
    from fem_bem import material_laws
    from gpu_solver import MU0
    from bridge import check_points_in_air
    path = ROOT/'fmm_verification.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    index = len(report['fem_bem_aedt_runs'])-1
    base = report['fem_bem_aedt_runs'][index]
    assert base['native_audit']['state']=='completed' and base['native_audit']['project_unchanged']
    for name,sha in base['source_sha256'].items():
        assert hashlib.sha256((TOOL/name).read_bytes()).hexdigest()==sha
    record = dict(state='running', diagnostic_only=True, baseline_run_index=index,
        started_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable,*sys.argv],
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=base['source_sha256'], interface_offsets=[], material_reference_checks=[])
    report.setdefault('fem_bem_aedt_diagnostics',[]).append(record)
    try:
        folder = REPO/'validation/20261107_repair_staged/full_pair_repaired_runtime'
        model = json.loads((folder/'model.json').read_text(encoding='utf-8'))
        parts = [np.load(folder/o['mesh_npz'])['tetra_m'] for o in model['objects']]
        tetra = np.concatenate(parts)
        ids = np.concatenate([np.full(len(t),k,int) for k,t in enumerate(parts)])
        volumes = abs(np.linalg.det(tetra[:,1:]-tetra[:,:1]))/6
        reference_B = np.asarray(base['native_audit']['fields']['B']['values'])[485:485+len(tetra)]
        reference_H = np.asarray(base['native_audit']['fields']['H']['values'])
        laws = material_laws([o['properties'] for o in model['objects']],ids)
        law_B = MU0*(reference_H+laws.eval(reference_H)[0])
        def error(a,b,w):
            return float(np.sqrt(np.sum(w*(a-b)**2)/max(np.sum(w*b**2),1e-300)))
        for k,obj in enumerate(model['objects']):
            ix=ids==k
            h=np.linalg.norm(reference_H[ix],axis=1)
            item=dict(object=obj['name'],kind=obj['properties']['kind'],
                native_B_vs_law_of_native_H_relative_L2=error(law_B[ix],reference_B[ix],volumes[ix,None]),
                native_H_min_Apm=float(h.min()),native_H_max_Apm=float(h.max()))
            if item['kind']=='nonlinear':
                first_knee=float(obj['properties']['bh_H_Apm_B_T'][1][0])
                item.update(first_positive_BH_knot_H_Apm=first_knee,
                            sampled_centers_beyond_first_knot=int(np.sum(h>first_knee)))
            record['material_reference_checks'].append(item)
        # Query geometry only; FEM fields are the retained current-source solve.
        # Canonical boundary keys preserve the baseline ordering independent of
        # face winding. Physical normals and areas come from that actual solve.
        nodes, inverse=np.unique(tetra.reshape(-1,3),axis=0,return_inverse=True)
        cells=inverse.reshape(-1,4)
        keys=np.sort(cells[:,[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1,3),axis=1)
        keys,first,counts=np.unique(keys,axis=0,return_index=True,return_counts=True)
        faces=keys[counts==1];owners=first[counts==1]//4
        tri=nodes[faces];centers=tri.mean(axis=1)
        normals=np.asarray(base['predictions']['normal_vectors'])
        areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        np.testing.assert_allclose(areas,base['predictions']['physical_face_areas_m2'],rtol=1e-12)
        altitude=3*volumes[owners]/areas
        base_step=base['query_layout']['interface_offset_m_max']
        owner=tetra[owners]
        matrix=(owner[:,1:]-owner[:,:1]).transpose(0,2,1)
        queries=[];steps=[]
        for factor in (10,100):
            step=np.minimum(base_step*factor,.02*altitude)
            for _ in range(20):
                inside=centers-step[:,None]*normals
                bary=np.linalg.solve(matrix,(inside-owner[:,0])[...,None])[...,0]
                bad=(bary<=0).any(axis=1)|(bary.sum(axis=1)>=1)
                if not bad.any():
                    break
                step[bad]/=2
            else:
                raise AssertionError('Unable to certify interior diagnostic probes')
            outside=centers+step[:,None]*normals
            check_points_in_air(folder,model,outside)
            queries.extend([inside,outside]);steps.append(step)
        with tempfile.TemporaryDirectory(prefix='magtense_aedt_probe_diagnostic_') as scratch:
            request_path=Path(scratch)/'queries.json'
            source=model['reference_source']
            save(request_path,dict(project=source['project'],design=source['design'],setup='Reference',
                model=str(folder/'model.json'),queries=dict(B=dict(quantity='B',points_m=np.concatenate(queries).tolist()))))
            record['query_sha256']=hashlib.sha256(request_path.read_bytes()).hexdigest()
            save(path,report)
            print('FEM_BEM_AEDT_DIAGNOSTIC reading live offset probes',flush=True)
            process=subprocess.run([str(LEGACY),'-u',str(Path(__file__).resolve()),
                '--fem-bem-aedt-worker',str(request_path)],cwd=REPO,capture_output=True,text=True,
                creationflags=subprocess.CREATE_NO_WINDOW)
            response=request_path.with_suffix('.result.json')
            if response.exists():
                record['native_audit']=json.loads(response.read_text(encoding='utf-8'))
            if process.returncode:
                record['worker_output_tail']=(process.stdout+process.stderr)[-10000:]
                raise RuntimeError('AEDT offset diagnostic failed')
        native=record['native_audit']
        assert native['project_unchanged']
        assert native['project_sha256_before']==base['native_audit']['project_sha256_before']
        fields=np.asarray(native['fields']['B']['values']).reshape(2,2,len(faces),3)
        qB=np.asarray(base['predictions']['boundary_normal_B_T'])
        for factor,step,field in zip((10,100),steps,fields):
            inside=np.einsum('fi,fi->f',field[0],normals)
            outside=np.einsum('fi,fi->f',field[1],normals)
            record['interface_offsets'].append(dict(factor=factor,offset_m_min=float(step.min()),
                offset_m_max=float(step.max()),Maxwell_normal_B_jump_relative_L2=error(outside,inside,areas),
                FEM_boundary_normal_B_vs_Maxwell_relative_L2=error(qB,outside,areas)))
        record.update(state='completed',baseline_acceptance_unchanged=True)
        save(path,report)
        print(json.dumps({k:v for k,v in record.items() if k!='native_audit'},indent=2),flush=True)
    except BaseException as failure:
        record.update(state='failed',error=f'{type(failure).__name__}: {failure}',traceback=traceback.format_exc())
        save(path,report)
        raise


def diagnose_fem_bem_identity():
    """--diagnose-fem-bem-identity: exact closed-surface constant identity.

    Current FEMBEM geometry / cuda_potential, full large_export magnetic mesh.
    Reference: D(1)=-1/2 on closed flat surface interiors. Check 100 seeded
    rows, 32 smallest-area and 32 highest-aspect-ratio rows, plus recorded H
    worst rows. Uses no compressed matrix. Requires CUDA and several GiB RAM.
    Append errors/hashes to fem_bem_identity_diagnostics; max error <=5e-5,
    otherwise preserve evidence and exit nonzero. Does not certify a field.
    """
    import numpy as np
    from fem_bem import FEMBEM,material_laws,triangle_rule
    from boundary_hmatrix import cuda_potential
    model=json.loads((EXPORT/'model.json').read_text(encoding='utf-8'))
    parts=[np.load(EXPORT/o['mesh_npz'])['tetra_m'] for o in model['objects']]
    ids=np.concatenate([np.full(len(t),k,int) for k,t in enumerate(parts)])
    mesh=FEMBEM(np.concatenate(parts),material_laws([o['properties'] for o in model['objects']],ids),
                boundary_backend='geometry')
    del parts,ids
    path=ROOT/'fmm_verification.json';report=json.loads(path.read_text(encoding='utf-8'))
    previous=report.get('fem_bem_large_runs',[{}])[-1]
    rng=np.random.default_rng(827319)
    shape=mesh.edge_lengths.max(axis=1)**2/mesh.areas
    rows=np.unique(np.r_[rng.choice(len(mesh.faces),100,replace=False),np.argsort(mesh.areas)[:32],
        np.argsort(shape)[-32:],previous.get('constant_identity_worst_rows',[])]).astype(int)
    bary,weights=triangle_rule(8)
    points=np.einsum('pv,fvi->fpi',bary,mesh.tri[rows]).reshape(-1,3)
    value=cuda_potential(mesh,np.ones(len(mesh.boundary_nodes)),np.zeros(len(mesh.faces)),points).reshape(len(rows),-1)
    error=value@weights+.5
    record=dict(rows=rows.tolist(),exact_constant_errors=error.tolist(),maximum_exact_error=float(abs(error).max()),
        area_ratio=float(mesh.areas.max()/mesh.areas.min()),threshold=5e-5,passed=bool(abs(error).max()<=5e-5),
        scope='Original exact-panel constant sums, sampled seeded and geometric extremes; no compressed operator',
        source_sha256={n:hashlib.sha256((TOOL/n).read_bytes()).hexdigest()
            for n in ('fem_bem.py','boundary_hmatrix.py','tetra_cuda.cu')})
    report.setdefault('fem_bem_identity_diagnostics',[]).append(record);save(path,report)
    print(json.dumps(record),flush=True)
    assert record['passed'],record['maximum_exact_error']


def validate_native_transfer():
    """--validate-native-transfer: native IronPython -> disk -> offline FEM-BEM.

    Uses canonical aedt_export_native/aedt_export/bridge source and the retained
    full_pair_reference.aedt (333613 total cells). Requires licensed AEDT 2022
    R2 with its bundled IronPython, solver CuPy/PyAMG and the earlier retained
    native_transfer/export + offline H cache. Tests current exporter in a
    temporary directory, all-domain mesh counts, normalized materials/geometry,
    unchanged original project, and a separate process whose import hook blocks
    all Ansys/PyAEDT/COM/MagTense native modules. Require weak residual <=1e-8
    and native air-B sample error <=10%; this is transfer/solver integration,
    not high-order equivalence or actual AEDT18/Linux-host certification.
    Append source hashes and assertions to native_transfer_runs. Failure exits
    nonzero; missing AEDT is failure, not a passing skip. Temporary duplicate
    data are removed; existing native export and solutions remain untouched.
    """
    import tempfile
    import shutil
    import numpy as np
    from aedt_export import import_native_bundle
    from aedtplt import read_aedtplt,get_unit_scale
    from bridge import read_maxwell_fld,read_model
    path=ROOT/'fmm_verification.json';report=json.loads(path.read_text(encoding='utf-8'))
    started=time.perf_counter();source=REPO/'validation/20261107_repair_staged/full_pair_reference.aedt'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before=digest(source);bin_path=Path(os.environ['ANSYSEM_ROOT222'])
    record=dict(state='running',started_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv],source_project_sha256=before,
        source_sha256={n:digest(TOOL/n) for n in ('aedt_export_native.py','aedt_export.py','bridge.py','fem_bem.py','boundary_hmatrix.py','tetra_cuda.cu')},
        scope='Actual AEDT 2022 R2 IronPython API + blocked-dependency offline process; AEDT18 and separate server OS not yet exercised')
    report.setdefault('native_transfer_runs',[]).append(record);save(path,report)
    try:
        with tempfile.TemporaryDirectory(prefix='magtense-native-final-') as temporary:
            work=Path(temporary).resolve()
            assert work.parent==Path(tempfile.gettempdir()).resolve() and work.name.startswith('magtense-native-final-')
            env=os.environ.copy();env['PATH']=str(bin_path)+os.pathsep+env['PATH']
            env['MAGTENSE_NATIVE_BIN']=str(bin_path);env['MAGTENSE_CANONICAL_TOOL']=str(TOOL)
            config=dict(project=str(source),design='Gap_3mm_CCW',solution='Reference : LastAdaptive',
                out=str(work/'bundle'),reference_quantities=['B','H'],
                points_csv=str(REPO/'validation/20261107_repair_staged/full_pair_repaired_runtime/points.csv'))
            env['MAGTENSE_NATIVE_CONFIG']=json.dumps(config)
            script="""import sys,os,json
sys.path.extend([os.environ['MAGTENSE_NATIVE_BIN'],os.path.join(os.environ['MAGTENSE_NATIVE_BIN'],'PythonFiles','DesktopPlugin'),os.environ['MAGTENSE_CANONICAL_TOOL']])
import ScriptEnv
ScriptEnv.InitializeNew(NonGraphical=True)
try:
 import aedt_export_native
 aedt_export_native.export_native(oDesktop,json.loads(os.environ['MAGTENSE_NATIVE_CONFIG']))
finally:
 ScriptEnv.Shutdown()
"""
            print('NATIVE_TRANSFER native_export',flush=True)
            result=subprocess.run([str(bin_path/'common/IronPython/ipy64.exe'),'-c',script],env=env,
                capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
            record['native_exit_code']=result.returncode;record['native_stdout']=result.stdout;record['native_stderr']=result.stderr
            assert result.returncode==0,result.stderr
            info=json.loads((work/'bundle/native_export.json').read_text(encoding='utf-8'))
            assert info['source']['exporter_sha256']==record['source_sha256']['aedt_export_native.py']
            record['native_version']=info['source']['aedt_version'];record['native_export_seconds']=info['elapsed_seconds']
            model=import_native_bundle(work/'bundle',work/'job',max_tiles=120000)
            prior=read_model(ROOT/'native_transfer/offline')
            by_name={o['name']:o for o in prior['objects']}
            for obj in model['objects']:
                old=by_name[obj['name']];assert obj['properties']==old['properties']
                np.testing.assert_array_equal(np.load(work/'job'/obj['mesh_npz'])['tetra_m'],
                    np.load(ROOT/'native_transfer/offline'/old['mesh_npz'])['tetra_m'])
            cells=0;solids=0
            for obj in info['objects']:
                if obj.get('mesh_raw'):
                    vertices,stats=read_aedtplt(work/'bundle'/obj['mesh_raw'],get_unit_scale(info['model_unit']))
                    cells+=stats['tetra_count'];solids+=1;del vertices
            assert cells==333613 and solids==13 and model['total_magnetic_tetra']==99093
            record.update(total_domain_cells=cells,exported_volume_solids=solids,magnetic_cells=99093,
                normalized_inputs_equal_retained_export=True)
            for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):env[name]='4'
            env['MAGTENSE_OFFLINE_ARGS']=json.dumps(['solve-fem-bem','--out',str(work/'job'),
                '--boundary','cuda_hmatrix','--fem-preconditioner','amg','--max-reference-error','0.10',
                '--boundary-cache',str(ROOT/'native_transfer/offline/boundary_cache')])
            script="""import sys,os,json,importlib.abc
class NoAnsys(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('ansys','pyaedt','ScriptEnv','clr','win32com','pythoncom','magtense'):
   raise RuntimeError('Offline dependency violation: '+fullname)
sys.meta_path.insert(0,NoAnsys());sys.path.insert(0,os.environ['MAGTENSE_CANONICAL_TOOL'])
from bridge import main
sys.exit(main(json.loads(os.environ['MAGTENSE_OFFLINE_ARGS'])))
"""
            print('NATIVE_TRANSFER offline_solve_no_ansys',flush=True)
            offline_env=env.copy()
            offline_env['PATH']=os.pathsep.join(p for p in os.environ['PATH'].split(os.pathsep) if 'ansys' not in p.lower())
            for name in list(offline_env):
                if name.startswith('ANSYSEM_ROOT') or name in ('ANSYSLMD_LICENSE_FILE','ANSYSLI_SERVERS','ANSYSLIC_DIR'):
                    del offline_env[name]
            result=subprocess.run([sys.executable,'-c',script],env=offline_env,capture_output=True,text=True,
                encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
            record['offline_exit_code']=result.returncode;record['offline_stdout']=result.stdout;record['offline_stderr']=result.stderr
            assert result.returncode==0,result.stderr
            solver=json.loads((work/'job/fem_bem_solve_info.json').read_text(encoding='utf-8'))
            assert solver['passed'] and solver['solver']['residual']<=1e-8 and solver['native_air_B_sample_relative_L2']<=.10
            record['offline_solve']=solver
            record['offline_environment']='Ansys module imports blocked; Ansys PATH/root/license variables removed'
            record['native_manifest']=json.loads((work/'bundle/manifest.json').read_text(encoding='utf-8'))
            # A fresh project copy has no result database. GenerateMesh is
            # submitted only to that disposable copy; never submit Analyze.
            mesh_project=work/'mesh_only.aedt';shutil.copyfile(source,mesh_project)
            config.update(project=str(mesh_project),out=str(work/'mesh_only_bundle'),reference_quantities=[])
            env['MAGTENSE_NATIVE_CONFIG']=json.dumps(config)
            script="""import sys,os,json
sys.path.extend([os.environ['MAGTENSE_NATIVE_BIN'],os.path.join(os.environ['MAGTENSE_NATIVE_BIN'],'PythonFiles','DesktopPlugin'),os.environ['MAGTENSE_CANONICAL_TOOL']])
import ScriptEnv
ScriptEnv.InitializeNew(NonGraphical=True)
try:
 import aedt_export_native
 config=json.loads(os.environ['MAGTENSE_NATIVE_CONFIG'])
 p=oDesktop.OpenProject(config['project']);d=p.SetActiveDesign(config['design'])
 d.GenerateMesh('Reference')
 aedt_export_native.export_native(oDesktop,config)
 oDesktop.CloseProject(p.GetName())
finally:
 ScriptEnv.Shutdown()
"""
            print('NATIVE_TRANSFER mesh_only_export_no_field_solve',flush=True)
            result=subprocess.run([str(bin_path/'common/IronPython/ipy64.exe'),'-c',script],env=env,
                capture_output=True,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
            record['mesh_only_stdout']=result.stdout;record['mesh_only_stderr']=result.stderr
            assert result.returncode==0,result.stderr
            inputs=json.loads((work/'mesh_only_bundle/native_export.json').read_text(encoding='utf-8'))
            assert inputs['mode']=='inputs_only' and not inputs.get('reference_fields')
            job=import_native_bundle(work/'mesh_only_bundle',work/'mesh_only_job',max_tiles=120000)
            assert job['total_magnetic_tetra']>0 and not job.get('maxwell_field')
            from fem_bem import FEMBEM,material_laws
            parts=[np.load(work/'mesh_only_job'/o['mesh_npz'])['tetra_m'] for o in job['objects']]
            ids=np.concatenate([np.full(len(t),k,int) for k,t in enumerate(parts)])
            geometry=FEMBEM(np.concatenate(parts),material_laws([o['properties'] for o in job['objects']],ids),boundary_backend='geometry')
            assert geometry.info['components']==10
            record['mesh_only']=dict(field_solve_submitted=False,disposable_project_only=True,
                source_mode=inputs['mode'],imported_without_reference=True,geometry=geometry.info,
                native_evidence=inputs.get('native_evidence',{}))
            assert digest(source)==before
            record.update(source_project_unchanged=True,state='complete',passed=True,elapsed_seconds=time.perf_counter()-started)
            save(path,report)
    except BaseException as error:
        record.update(state='failed',passed=False,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-started)
        save(path,report);raise


def benchmark_fem_bem_batches(boundary,report,record,path,batch_sizes):
    """ACA batch-size probe: actual retained full-AEDT terminal blocks.

    Invoked by --benchmark-fem-bem-assembly --fem-bem-assembly-batches.
    Compare the current _aca_batch with 64/256/1024-block dispatch, including
    row-plan preparation. Independent expectations are the retained full H
    cache's ranks and FP32 factor bytes; failed blocks must still return rank
    zero and require exact fallback. Three rotated warm timings per family;
    append evidence only, without changing the production batch cap or cache.
    Numerical mismatch raises. This does not measure a complete new H build.
    """
    import numpy as np
    import cupy as cp
    b=boundary
    origin=next(r for r in reversed(report['fem_bem_large_runs'])
        if r.get('passed') and r.get('operator',{}).get('hmatrix_cache_prefix'))
    assert origin['input_sha256']==record['input_sha256']
    prefix=Path(origin['operator']['hmatrix_cache_prefix'])
    header=json.loads(prefix.with_suffix('.json').read_text(encoding='utf-8'))
    manifest=header.pop('manifest_sha256')
    assert hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest()==manifest
    for name,key in [('boundary_hmatrix.py','builder_sha256'),('tetra_cuda.cu','cuda_sha256')]:
        assert header['signature'][key]==record['source_sha256'][name]
    dtype=np.dtype(header['payload_dtype']);assert dtype==np.dtype('float32')
    offsets=np.r_[0,np.cumsum(header['payload_sizes'],dtype=np.int64)]*dtype.itemsize
    position=int(offsets[-1])
    for name in header['fixed_order']:
        spec=header['fixed'][name]
        if name=='metadata_logical':break
        position+=spec['count']*np.dtype(spec['dtype']).itemsize
    with prefix.with_suffix('.bin').open('rb') as stream:
        stream.seek(position)
        metadata=np.fromfile(stream,dtype=spec['dtype'],count=spec['count']).reshape(spec['shape'])
    near=np.asarray([[b.tree.nodes[a]['start'],b.tree.nodes[a]['end']-b.tree.nodes[a]['start'],
        b.tree.column_nodes[c]['start'],b.tree.column_nodes[c]['end']-b.tree.column_nodes[c]['start']]
        for a,c,far in b.tree.blocks if not far],dtype=np.int64)
    np.testing.assert_array_equal(metadata[:len(near),:4],near)
    terminal=metadata[len(near):];samples=[];rng=np.random.default_rng(981735)
    for label,selection in [('fallback',terminal[:,4]<0),('retained',terminal[:,4]>0)]:
        group=terminal[selection]
        shapes,counts=np.unique(group[:,[1,3]],axis=0,return_counts=True)
        for index in np.argsort(counts)[-2:][::-1]:
            m,n=map(int,shapes[index]);eligible=group[(group[:,1]==m)&(group[:,3]==n)]
            count=min(1024,len(eligible));chosen=rng.choice(len(eligible),count,replace=False)
            samples.append((f'{label}_{m}x{n}',eligible[chosen],int(counts[index])))
    del near,metadata,terminal
    record.update(batch_sizes=batch_sizes,source_cache_prefix=str(prefix),
        source_binary_sha256=header['binary_sha256'],
        scope='Bounded full-AEDT ACA dispatch batches; exact retained ranks/factor bytes and failed-block decisions, not complete build or Maxwell speedup')
    for label,rows,population in samples:
        mismatch_before=record.get('equivalence_mismatch_count',0)
        desc=rows[:,:4].copy();m,n=map(int,desc[0,[1,3]])
        expected=np.maximum(rows[:,4],0).astype(np.int32);factor_hashes=[]
        with prefix.with_suffix('.bin').open('rb') as stream:
            for row,rank in zip(rows,expected):
                digest=hashlib.sha256()
                for pointer,offset,size in [(row[5],row[6],m*rank),(row[7],row[8],n*rank)] if rank else []:
                    stream.seek(int(offsets[pointer]+offset));data=stream.read(int(size)*dtype.itemsize)
                    assert len(data)==int(size)*dtype.itemsize;digest.update(data)
                factor_hashes.append(digest.hexdigest())
        print('ACA_BATCH_PROBE '+json.dumps(dict(case=label,blocks=len(desc),population=population)),flush=True)
        def evaluate(cap):
            elapsed=0.
            for first in range(0,len(desc),cap):
                chunk=desc[first:first+cap]
                cp.cuda.Stream.null.synchronize();tick=time.perf_counter()
                U,W,ranks,errors=b._aca_batch(chunk,m,n)
                cp.cuda.Stream.null.synchronize();elapsed+=time.perf_counter()-tick
                host_u=cp.asnumpy(U.astype(cp.float32));host_w=cp.asnumpy(W.astype(cp.float32))
                for local,rank in enumerate(ranks):
                    digest=hashlib.sha256()
                    if rank:
                        digest.update(np.ascontiguousarray(host_u[local,:,:rank]).tobytes())
                        digest.update(np.ascontiguousarray(host_w[local,:rank,:]).tobytes())
                    changed=(int(rank)!=int(expected[first+local]) or digest.hexdigest()!=factor_hashes[first+local])
                    if changed:
                        record['equivalence_mismatch_count']=record.get('equivalence_mismatch_count',0)+1
                        diagnostics=record.setdefault('equivalence_mismatches',[])
                        key=(label,cap,first+local)
                        if len(diagnostics)<16 and not any((a['case'],a['batch_size'],a['sample'])==key for a in diagnostics):
                            mismatch=dict(case=label,batch_size=cap,sample=first+local,
                                descriptor=chunk[local].tolist(),retained_rank=int(expected[first+local]),probe_rank=int(rank))
                            if rank:
                                original=cp.asnumpy(b._batch(chunk[local:local+1],0,prefix=np.array([0,m*n]))).reshape(m,n)
                                candidate=host_u[local,:,:rank].astype(float)@host_w[local,:rank,:].astype(float)
                                columns=b.tree.column_permutation[int(chunk[local,2]):int(chunk[local,2]+n)]
                                flux=columns<len(b.mesh.faces);difference=candidate-original
                                for channel,mask in [('P0',flux),('P1',~flux)]:
                                    mismatch[channel+'_full_block_relative_L2']=float(np.linalg.norm(difference[:,mask])/
                                        max(np.linalg.norm(original[:,mask]),1e-300))
                                mismatch['K_max_absolute_row_L1_error']=float(abs(difference[:,~flux]).sum(axis=1).max())
                                mismatch['full_block_threshold_passed']=bool(max(mismatch[c+'_full_block_relative_L2']
                                    for c in ('P0','P1'))<=b.rtol and mismatch['K_max_absolute_row_L1_error']<=b.rtol)
                            diagnostics.append(mismatch)
                del U,W,host_u,host_w
            return elapsed
        for cap in batch_sizes:evaluate(cap)
        timings={str(cap):[] for cap in batch_sizes}
        for repeat in range(3):
            for cap in batch_sizes[repeat:]+batch_sizes[:repeat]:timings[str(cap)].append(evaluate(cap))
        medians={cap:float(np.median(values)) for cap,values in timings.items()}
        entry=dict(case=label,blocks=len(desc),full_cache_family_blocks=population,descriptors_sha256=hashlib.sha256(desc.tobytes()).hexdigest(),
            ranks_and_FP32_factor_bytes_equal_full_cache=record.get('equivalence_mismatch_count',0)==mismatch_before,
            mismatch_observations_including_warmup=record.get('equivalence_mismatch_count',0)-mismatch_before,
            seconds=timings,median_seconds=medians,
            speedup_vs_64={cap:medians['64']/value for cap,value in medians.items()})
        record['cases'].append(entry);save(path,report)
        print('ACA_BATCH_RESULT '+json.dumps(entry),flush=True)
        cp.get_default_memory_pool().free_all_blocks()
    passed=not record.get('equivalence_mismatch_count',0)
    record.update(state='completed' if passed else 'failed',passed=passed,row_reuse_stats=b.row_reuse_stats,
        mismatch_count_scope='Observations include warmup and repeated timings; not a count of unique blocks',
        timing_scope='Sum of synchronized ACA calls including CPU row-plan cost; full warmup, three rotated repetitions; factor-byte validation outside timed interval')
    save(path,report)
    assert passed,'Batch-size probe changed retained rank/factor bytes; see original-integral diagnostics'


def benchmark_fem_bem_assembly(batch_sizes=None):
    """--benchmark-fem-bem-assembly: bounded large-AEDT ACA comparison.

    Uses current FEMBEM/BoundaryHMatrix/tetra_cuda.cu and the full large_export
    geometry, with no assembled H matrix or native project modification. Actual
    spatial blocks cover five size bands and a retained coplanar-cross outlier.
    Compare direct, inactive-masked and shared-panel ACA, including plan cost.
    Require identical pivot-row FP64 coefficients/ranks and action differences
    <=1e-10. Append raw timings, counts and hashes to fem_bem_assembly_probes;
    numerical failure exits nonzero. Requires CUDA and several GiB host RAM.
    This sample does not certify full-build speed or new Maxwell accuracy.
    """
    os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
    import numpy as np
    import cupy as cp
    from fem_bem import FEMBEM,material_laws
    from boundary_hmatrix import BoundaryHMatrix
    path=ROOT/'fmm_verification.json';report=json.loads(path.read_text(encoding='utf-8'))
    record=dict(state='running',started_utc=datetime.now(timezone.utc).isoformat(),cases=[],
        source_sha256={n:hashlib.sha256((TOOL/n).read_bytes()).hexdigest()
            for n in ('fem_bem.py','boundary_hmatrix.py','tetra_cuda.cu')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Bounded ACA blocks from full 2718864-cell AEDT geometry; not full H build or field speedup')
    key='fem_bem_batch_probes' if batch_sizes is not None else 'fem_bem_assembly_probes'
    report.setdefault(key,[]).append(record);save(path,report)
    started=time.perf_counter()
    try:
        model=json.loads((EXPORT/'model.json').read_text(encoding='utf-8'))
        record['input_sha256']={n:hashlib.sha256((EXPORT/n).read_bytes()).hexdigest()
            for n in ['model.json',*[o['mesh_npz'] for o in model['objects']]]}
        parts=[np.load(EXPORT/o['mesh_npz'])['tetra_m'] for o in model['objects']]
        ids=np.concatenate([np.full(len(t),k,int) for k,t in enumerate(parts)])
        mesh=FEMBEM(np.concatenate(parts),material_laws([o['properties'] for o in model['objects']],ids),
            boundary_backend='geometry');del parts,ids
        b=BoundaryHMatrix(mesh,quadrature_order=8,leaf_size=32,eta=3.,rtol=1e-6,max_rank=64,
            storage_dtype='float32',far_quadrature_order=4,_assemble=False)
        record['geometry']=mesh.info;record['partition']=b.info
        device=cp.cuda.runtime.getDeviceProperties(0)['name']
        record['GPU']=device.decode() if isinstance(device,bytes) else device
        if batch_sizes is not None:
            benchmark_fem_bem_batches(b,report,record,path,batch_sizes)
            record['elapsed_seconds']=time.perf_counter()-started;save(path,report)
            return
        rng=np.random.default_rng(87132);groups={}
        for a,c,far in b.tree.blocks:
            if not far: continue
            x,y=b.tree.nodes[a],b.tree.column_nodes[c]
            nr,nc=x['end']-x['start'],y['end']-y['start']
            if nc<16 or nr+nc>131072: continue
            groups.setdefault((nr,nc),[]).append([x['start'],nr,y['start'],nc])
        samples=[];used=set()
        for target in (128,512,2048,8192,32768):
            shape=min(groups,key=lambda s:abs(np.log2((s[0]+s[1])/target)))
            if shape in used: continue
            used.add(shape);nr,nc=shape
            # Use the production ACA workspace/batch cap rather than tiny
            # batches, including variation in convergence among 64 blocks.
            count=min(64,len(groups[shape]),max(1,(128*2**20)//(8*64*(nr+nc))))
            chosen=rng.choice(len(groups[shape]),count,replace=False)
            samples.append((f'size_{target}',np.asarray([groups[shape][i] for i in chosen],dtype=np.int64)))
        outlier=report['fem_bem_compression_diagnostics'][-1]['worst_row_blocks'][0]
        samples.append(('retained_K_outlier',np.array([[outlier[n] for n in ('rs','nr','cs','nc')]],dtype=np.int64)))
        variants=[('direct',False),('direct',True),('shared',True)]
        for label,desc in samples:
            count=len(desc);m,n=map(int,desc[0,[1,3]])
            print('ACA_PROBE '+json.dumps(dict(case=label,blocks=count,rows=m,columns=n)),flush=True)
            plan=b._row_plan(desc,n,force=True)
            pivots=cp.asarray([rng.integers(m) for _ in range(count)],dtype=cp.int32)
            direct=b._batch(desc,1,nr=m,nc=n,pivots=pivots,far=True)
            shared=b._pivot_rows(plan,pivots)
            np.testing.assert_array_equal(cp.asnumpy(direct),cp.asnumpy(shared))
            del direct,shared
            entry=dict(case=label,descriptors=desc.tolist(),expanded_panel_slots=plan['expanded_panels'],
                shared_panel_slots=plan['shared_panels'],pivot_rows_bitwise_equal=True,variants={})
            del plan
            vector=cp.asarray(rng.normal(size=(count,n,4)));reference=None;reference_ranks=None
            times={str(v):[] for v in variants}
            # Warm the complete ACA path (cuBLAS, allocation and shape-specific
            # elementwise kernels), then rotate order across three repetitions.
            # Timings include plan preparation and final synchronization.
            for backend,skip in variants:
                b.pivot_row_backend=backend;b.skip_inactive=skip
                U,W,_,_=b._aca_batch(desc,m,n)
                cp.cuda.Stream.null.synchronize();del U,W
            for repeat in range(3):
                order=variants[repeat:]+variants[:repeat]
                for backend,skip in order:
                    name=f'{backend}_skip{int(skip)}';b.pivot_row_backend=backend;b.skip_inactive=skip
                    cp.cuda.Stream.null.synchronize();tick=time.perf_counter()
                    U,W,ranks,errors=b._aca_batch(desc,m,n)
                    cp.cuda.Stream.null.synchronize();elapsed=time.perf_counter()-tick
                    times[str((backend,skip))].append(elapsed)
                    mask=cp.arange(U.shape[2])[None,:]<cp.asarray(ranks)[:,None]
                    action=cp.asnumpy((U*mask[:,None,:])@(W@vector))
                    if reference is None:
                        reference=action;reference_ranks=ranks.copy()
                    np.testing.assert_array_equal(ranks,reference_ranks)
                    relative=float(np.linalg.norm(action-reference)/max(np.linalg.norm(reference),1e-300))
                    assert relative<=1e-10,(label,name,relative)
                    entry['variants'].setdefault(name,dict(ranks=ranks.tolist(),action_relative_L2_max=0.))
                    entry['variants'][name]['action_relative_L2_max']=max(entry['variants'][name]['action_relative_L2_max'],relative)
                    del U,W,action
            for backend,skip in variants:
                name=f'{backend}_skip{int(skip)}';timings=times[str((backend,skip))]
                entry['variants'][name].update(seconds=timings,median_seconds=float(np.median(timings)))
            base=entry['variants']['direct_skip0']['median_seconds']
            entry['inactive_speedup']=base/entry['variants']['direct_skip1']['median_seconds']
            entry['combined_speedup']=base/entry['variants']['shared_skip1']['median_seconds']
            record['cases'].append(entry);save(path,report)
            print('ACA_RESULT '+json.dumps(entry),flush=True)
            del vector,reference;cp.get_default_memory_pool().free_all_blocks()
        totals={name:sum(c['variants'][name]['median_seconds'] for c in record['cases'])
            for name in record['cases'][0]['variants']}
        record.update(state='completed',passed=True,elapsed_seconds=time.perf_counter()-started,
            sum_sample_median_seconds=totals,combined_sample_speedup=totals['direct_skip0']/totals['shared_skip1'],
            row_reuse_stats=b.row_reuse_stats,
            compiled_CUDA_cache='full ACA warmup per variant/shape, then three rotated timings; no H cache loaded')
        save(path,report)
    except BaseException as error:
        record.update(state='failed',passed=False,elapsed_seconds=time.perf_counter()-started,
            error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        save(path,report);raise


def benchmark_fem_bem_large(args):
    """Full 235,652-face H-matrix setup/matvec, with optional coupled solve.

    Current-source entries: FEMBEM + BoundaryHMatrix + tetra_cuda.cu. Actual
    large_export geometry only; no volume-dipole FMM or copied implementation.
    Direct original-operator sums on seeded global rows must agree within
    20*H tolerance for random and smooth inputs. Constant double-layer row
    identity must agree within 50*H tolerance. Resource/timing records append
    to fem_bem_large_runs, including failures. A boundary benchmark alone is
    explicitly NOT a converged large field or Maxwell accuracy certification.
    Requires CuPy/CUDA, ~several GiB free GPU/host RAM; coupled solve uses PyAMG.
    """
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['MKL_NUM_THREADS']='4'
    os.environ['OPENBLAS_NUM_THREADS']='4'
    import numpy as np
    import cupy as cp
    import psutil
    from fem_bem import FEMBEM,material_laws
    started=time.perf_counter();path=ROOT/'fmm_verification.json'
    report=json.loads(path.read_text(encoding='utf-8'))
    record=dict(state='running',started_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv],requested_coupled_solve=args.solve_fem_bem_large,
        field_solve_completed=False,source_sha256={name:hashlib.sha256((TOOL/name).read_bytes()).hexdigest()
            for name in ('fem_bem.py','boundary_hmatrix.py','tetra_cuda.cu','gpu_solver.py')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={},
        scope='Full magnetic mesh CPU sparse FEM / CUDA H-matrix boundary; no full native Maxwell timing comparison',
        settings=dict(leaf_size=args.fem_bem_h_leaf,eta=args.fem_bem_h_eta,rtol=args.fem_bem_h_rtol,
            max_rank=args.fem_bem_h_rank,budget_gib=args.fem_bem_h_gib,quadrature_order=8,
            storage_dtype=args.fem_bem_h_storage,compute_dtype='float64',
            far_pivot_quadrature_order=args.fem_bem_h_far_order,
            pivot_row_backend=args.fem_bem_h_row_backend,
            skip_inactive=args.fem_bem_h_skip_inactive,
            adaptive_exact_K_blocks=args.fem_bem_refine_boundary),checks=[])
    report.setdefault('fem_bem_large_runs',[]).append(record)
    try:
        gpu=GPUUsage()
    except (OSError,AssertionError):
        gpu=None
    device=cp.cuda.runtime.getDeviceProperties(0)['name']
    record['hardware']=dict(GPU=device.decode() if isinstance(device,bytes) else device,
        system_RAM_bytes=psutil.virtual_memory().total,Python=sys.version,CuPy=cp.__version__,NumPy=np.__version__)
    record['compiled_CUDA_cache']='warm after current-source regression; H cache status recorded separately'
    record['resource_sampling_scope']='At progress writes (10 s throttle during build); whole-board use includes other apps'
    last_write=0.
    def progress(phase,**extra):
        nonlocal last_write
        record.update(extra)
        record.update(phase=phase,elapsed_seconds=time.perf_counter()-started)
        now=time.perf_counter()
        if phase=='boundary_build' and now-last_write<10:
            return
        last_write=now
        mem=psutil.Process().memory_info()
        record['process_peak_RSS_bytes']=getattr(mem,'peak_wset',mem.rss)
        record['cupy_pool_peak_reserved_bytes']=max(record.get('cupy_pool_peak_reserved_bytes',0),cp.get_default_memory_pool().total_bytes())
        record['cupy_pool_used_bytes']=cp.get_default_memory_pool().used_bytes()
        record['system_available_RAM_bytes']=psutil.virtual_memory().available
        if gpu is not None:
            used=gpu.used()
            record['sampled_GPU_board_peak_bytes']=max(record.get('sampled_GPU_board_peak_bytes',0),used)
            record.setdefault('GPU_board_samples',[]).append(dict(elapsed_seconds=record['elapsed_seconds'],used_bytes=used,phase=phase))
        save(path,report)
        print('FEM_BEM_LARGE '+json.dumps({**extra,'phase':phase,'elapsed_seconds':record['elapsed_seconds']}),flush=True)
    try:
        model=json.loads((EXPORT/'model.json').read_text(encoding='utf-8'))
        record['input_sha256']['model.json']=hashlib.sha256((EXPORT/'model.json').read_bytes()).hexdigest()
        parts=[];ids=[]
        for k,obj in enumerate(model['objects']):
            file=EXPORT/obj['mesh_npz']
            record['input_sha256'][obj['mesh_npz']]=hashlib.sha256(file.read_bytes()).hexdigest()
            part=np.load(file)['tetra_m'];parts.append(part);ids.append(np.full(len(part),k,dtype=np.int32))
        tetra=np.concatenate(parts);cell_ids=np.concatenate(ids);del parts,ids
        assert len(tetra)==2718864
        progress('geometry_and_boundary_setup')
        mesh=FEMBEM(tetra,material_laws([o['properties'] for o in model['objects']],cell_ids),
            boundary_backend='cuda_hmatrix',quadrature_order=8,
            boundary_options=dict(leaf_size=args.fem_bem_h_leaf,eta=args.fem_bem_h_eta,
                rtol=args.fem_bem_h_rtol,max_rank=args.fem_bem_h_rank,budget_gib=args.fem_bem_h_gib,
                progress=progress,cache_path=EXPORT/'fem_bem_boundary_cache',storage_dtype=args.fem_bem_h_storage,
                far_quadrature_order=args.fem_bem_h_far_order,pivot_row_backend=args.fem_bem_h_row_backend,
                skip_inactive=args.fem_bem_h_skip_inactive))
        assert len(mesh.vertices)==522922 and len(mesh.faces)==235652
        record['operator']=mesh.info
        # Persist numeric data before numerical gates. Every load repeats all
        # gates; a rejected cache is diagnostic data, never a certified field.
        if not mesh.boundary.info.get('hmatrix_cache_hit'):
            progress('cache_numeric_boundary_pending_validation')
            cache_started=time.perf_counter();mesh.boundary.save_cache()
            record['boundary_cache_write_seconds']=time.perf_counter()-cache_started
            mesh.info.update(mesh.boundary.info)
        record['boundary_cache_accuracy_checks_passed']=False
        if args.fem_bem_refine_boundary:
            progress('adaptive_exact_boundary_blocks')
            record['boundary_refinement']=mesh.refine_boundary(rtol=args.fem_bem_h_rtol,progress=progress)
        progress('direct_global_operator_probes')
        rng=np.random.default_rng(827319)
        rows=np.unique(np.r_[rng.choice(len(mesh.faces),16,replace=False),
            np.argmin(mesh.areas),np.argmax(mesh.areas)])
        record['direct_probe_rows']=rows.tolist()
        for name,q,u in [('random',rng.normal(size=len(mesh.faces)),rng.normal(size=len(mesh.boundary_nodes))),
                         ('flux_only',rng.normal(size=len(mesh.faces)),np.zeros(len(mesh.boundary_nodes))),
                         ('smooth',mesh.normals[:,2],mesh.vertices[mesh.boundary_nodes,2])]:
            approximate=mesh.boundary.apply(q,u)[rows]
            direct=mesh.boundary.direct_rows(rows,q,u)
            error=float(np.linalg.norm(approximate-direct)/max(np.linalg.norm(direct),1e-300))
            record['checks'].append(dict(input=name,direct_rows_relative_L2=error,
                                          tolerance=20*args.fem_bem_h_rtol,passed=error<=20*args.fem_bem_h_rtol))
            progress('direct_probe_complete',probe=name,relative_L2=error)
            assert error<=20*args.fem_bem_h_rtol,(name,error)
        value=mesh.boundary.apply(np.zeros(len(mesh.faces)),np.ones(len(mesh.boundary_nodes)))
        constant_errors=value/mesh.areas-.5
        worst=np.argsort(abs(constant_errors))[-32:]
        constant_error=float(np.max(abs(constant_errors)))
        record['constant_double_layer_identity_max_error']=constant_error
        record['constant_identity_worst_rows']=worst.tolist()
        record['constant_identity_worst_errors']=constant_errors[worst].tolist()
        if constant_error>50*args.fem_bem_h_rtol:
            direct=mesh.boundary.direct_rows(worst,np.zeros(len(mesh.faces)),np.ones(len(mesh.boundary_nodes)))
            record['constant_identity_exact_worst_row_errors']=(direct/mesh.areas[worst]-.5).tolist()
            progress('constant_identity_rejected',constant_double_layer_identity_max_error=constant_error)
        assert constant_error<=50*args.fem_bem_h_rtol,constant_error
        record['boundary_cache_accuracy_checks_passed']=True
        q=rng.normal(size=len(mesh.faces));u=rng.normal(size=len(mesh.boundary_nodes))
        timings=[]
        for repetition in range(5):
            tick=time.perf_counter();mesh.boundary.apply(q,u)
            elapsed=time.perf_counter()-tick
            if repetition>=2:
                timings.append(elapsed)
        record['warm_boundary_matvec_seconds']=timings
        record['warm_boundary_matvec_median_seconds']=float(np.median(timings))
        progress('boundary_benchmark_complete',operator=mesh.info,
            warm_boundary_matvec_median_seconds=record['warm_boundary_matvec_median_seconds'])
        if args.solve_fem_bem_large or args.fem_bem_check_solution:
            if args.solve_fem_bem_large:
                if psutil.virtual_memory().available<2*2**30:
                    raise MemoryError('Insufficient host RAM headroom before sparse FEM/AMG setup')
                cp.get_default_memory_pool().free_all_blocks()
                u,q,info=mesh.solve(tol=1e-8,maxiter=20,linear_rtol=1e-6,
                    linear_maxiter=10,fem_preconditioner='amg',progress=progress)
                solution_file=ROOT/f'fem_bem_large_solution_{len(report["fem_bem_large_runs"])}.npz'
                if solution_file.exists():
                    raise FileExistsError('Existing full solution retained: '+str(solution_file))
                np.savez_compressed(solution_file,potential_scaled_Apm=u,boundary_q_Apm=q)
                record.update(field_solve_completed=True,solver=info,solution_npz=solution_file.name,
                    solution_sha256=hashlib.sha256(solution_file.read_bytes()).hexdigest())
                progress('coupled_solve_complete',coupled_relative_residual=info['residual'])
                assert info['converged'] and info['residual']<=1e-8
            else:
                progress('check_retained_solution')
                solution_file=args.fem_bem_check_solution.resolve()
                digest=hashlib.sha256(solution_file.read_bytes()).hexdigest()
                origin=next(r for r in report['fem_bem_large_runs'][:-1]
                    if r.get('solution_sha256')==digest and r.get('passed') and r.get('field_solve_completed'))
                assert origin['input_sha256']==record['input_sha256'],'Retained solution input mismatch'
                with np.load(solution_file) as arrays:
                    u=arrays['potential_scaled_Apm'];q=arrays['boundary_q_Apm']
                assert u.shape==(len(mesh.vertices),) and q.shape==(len(mesh.faces),)
                assert np.isfinite(u).all() and np.isfinite(q).all()
                r0,A0=mesh.residual(np.zeros_like(u),np.zeros_like(q),np.zeros(3));del A0
                residual,A=mesh.residual(u,q,np.zeros(3));del A
                relative_residual=float(np.linalg.norm(residual)/max(np.linalg.norm(r0),1e-300))
                record['retained_solution_check']=dict(solution_npz=solution_file.name,sha256=digest,
                    coupled_relative_residual=relative_residual,tolerance=1e-8,passed=relative_residual<=1e-8,
                    scope='Existing converged solution evaluated with newly built boundary; no new nonlinear solve')
                assert relative_residual<=1e-8,relative_residual
                del r0,residual
            exact_boundary=mesh.boundary.direct_rows(rows,q,u[mesh.boundary_nodes])
            mass_trace=.5*(mesh.mass.T@u)[rows]
            record['original_boundary_probe_relative_residual']=float(np.linalg.norm(exact_boundary+mass_trace)/
                max(np.linalg.norm(exact_boundary)+np.linalg.norm(mass_trace),1e-300))
            assert record['original_boundary_probe_relative_residual']<=20*args.fem_bem_h_rtol
            # Same physical project, independently solved adaptive Maxwell
            # reference; this is not a same-8.68M-mesh native timing comparison.
            reference_run=report['fem_bem_aedt_runs'][-1]
            native=reference_run['native_audit']
            assert native['state']=='completed' and native['project_unchanged']
            generation=json.loads((ROOT/'mesh_generation.json').read_text(encoding='utf-8'))
            assert native['project_sha256_before']==generation['source_sha256']
            reference_folder=REPO/'validation/20261107_repair_staged/full_pair_repaired_runtime'
            reference_model=json.loads((reference_folder/'model.json').read_text(encoding='utf-8'))
            by_name={obj['name']:obj for obj in reference_model['objects']}
            for obj in model['objects']:
                actual,expected=obj['properties'],by_name[obj['name']]['properties']
                assert actual['kind']==expected['kind']
                for key in expected.keys()-{'kind'}:
                    np.testing.assert_allclose(actual[key],expected[key],rtol=1e-12,atol=1e-12)
            points=np.loadtxt(EXPORT/model['points'],delimiter=',',skiprows=1)
            old_points=np.loadtxt(reference_folder/reference_model['points'],delimiter=',',skiprows=1)
            np.testing.assert_allclose(points,old_points,rtol=0,atol=1e-14)
            progress('validate_large_air_field')
            predicted,gradient_info=mesh.air_field(u,q,points,validate_air=mesh.validate_air)
            Bnative=np.asarray(native['fields']['B']['values'])
            air_reference=Bnative[:485]
            def relative(a,b,weights=None):
                w=np.ones(len(a)) if weights is None else weights
                return float(np.sqrt(np.sum(w[:,None]*(a-b)**2)/max(np.sum(w[:,None]*b**2),1e-300)))
            accuracy=dict(air_B_relative_L2=relative(predicted,air_reference),
                air_RMS_vector_error_T=float(np.sqrt(np.mean(np.sum((predicted-air_reference)**2,axis=1)))),
                air_max_vector_error_T=float(np.linalg.norm(predicted-air_reference,axis=1).max()),
                reference_scope='Native converged 333613-total-cell Maxwell solution, same CAD/material/current inputs; '
                    'not a field solve on the 8679245-total-cell mesh')
            progress('validate_large_material_and_interface_samples',air_B_relative_L2=accuracy['air_B_relative_L2'])
            coarse_parts=[np.load(reference_folder/o['mesh_npz'])['tetra_m'] for o in reference_model['objects']]
            coarse_tetra=np.concatenate(coarse_parts)
            coarse_ids=np.concatenate([np.full(len(t),k,int) for k,t in enumerate(coarse_parts)])
            positions=coarse_tetra.mean(axis=1)
            located=mesh.locate_points(positions)
            assert np.all(located>=0),'A retained material sample left the large FEM mesh'
            Hfull,Bfull,*_=mesh.fields(u)
            Hsample,Bsample=Hfull[located],Bfull[located]
            weights=abs(np.linalg.det(coarse_tetra[:,1:]-coarse_tetra[:,:1]))/6
            B_reference=Bnative[485:485+len(positions)]
            H_reference=np.asarray(native['fields']['H']['values'])
            accuracy.update(material_B_sample_relative_L2=relative(Bsample,B_reference,weights),
                            material_H_sample_relative_L2=relative(Hsample,H_reference,weights))
            coarse=FEMBEM(coarse_tetra,material_laws([o['properties'] for o in reference_model['objects']],coarse_ids),
                          boundary_backend='geometry')
            face_positions=coarse.tri.mean(axis=1)*coarse.length+coarse.origin
            face_index=mesh.locate_boundary(face_positions,coarse.normals)
            assert np.all(face_index>=0),'A retained boundary sample left the large CAD surface'
            reference_normal=np.einsum('fi,fi->f',Bnative[485+len(positions)+len(coarse.faces):],coarse.normals)
            predicted_normal=-4e-7*np.pi*q[face_index]
            area=coarse.areas*coarse.length**2
            accuracy['boundary_normal_B_sample_relative_L2']=float(np.sqrt(np.sum(area*(predicted_normal-reference_normal)**2)/
                max(np.sum(area*reference_normal**2),1e-300)))
            accuracy['material_samples']=len(positions);accuracy['boundary_samples']=len(face_index)
            record.update(accuracy=accuracy,air_gradient=gradient_info,
                air_prediction_B_T=predicted.tolist(),material_prediction_B_T=Bsample.tolist(),
                material_prediction_H_Apm=Hsample.tolist(),boundary_prediction_normal_B_T=predicted_normal.tolist())
            checks=dict(air=accuracy['air_B_relative_L2']<=.05,
                material=accuracy['material_B_sample_relative_L2']<=.10,
                boundary=accuracy['boundary_normal_B_sample_relative_L2']<=.10)
            record['physical_acceptance_checks']=checks
            progress('physical_comparison_complete',accuracy=accuracy,physical_acceptance_checks=checks)
            assert all(checks.values()),checks
        progress('complete',state='completed',passed=True)
    except BaseException as error:
        progress('failed',state='failed',passed=False,error=f'{type(error).__name__}: {error}',
                 traceback=traceback.format_exc())
        raise


def worker(stage,neighbours,farfield):
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    os.environ['CUPY_CACHE_DIR'] = str(ROOT/'.cupy_cache')
    sys.path.insert(0,str(TOOL))
    state = {'stage':stage,'pid':os.getpid(),'phase':'starting','started_utc':datetime.now(timezone.utc).isoformat()}
    def progress(phase,**details):
        state.update(phase=phase,updated_utc=datetime.now(timezone.utc).isoformat(),**details)
        save(STATUS,state)
        print('PROGRESS '+json.dumps({'phase':phase,**details}),flush=True)
    try:
        if stage=='validate_cuda':
            validate_cuda_small()
            return
        if stage == 'export':
            from aedt_export import export_project
            mesh = json.loads((ROOT/'mesh_generation.json').read_text())
            progress('native_mesh_export')
            model = export_project(str(ROOT/'gap3mm_fmm_3m.aedt'),'Gap_3mm_CCW',str(EXPORT),
                points_csv=str(REPO/'validation/20261107_repair_staged/full_pair_repaired_runtime/points.csv'),
                setup='Reference : LastAdaptive',version='2022.2',no_reference=True,
                ignore_current_check=True,new_desktop=True,max_tiles=2718864)
            actual = {o['name']:o['mesh_stats']['tetra_count'] for o in model['objects']}
            expected = mesh['attempts'][-1]['per_object']
            assert model['total_magnetic_tetra'] == 2718864
            assert all(actual[name] == expected[name] for name in mesh['magnetic_objects'])
            progress('export_complete',magnetic_cells=model['total_magnetic_tetra'])
            return
        # Preserve native DLL directory handles throughout the worker lifetime.
        dll = os.add_dll_directory('C:/Program Files/AnsysEM/v222/Win64')
        import cupy as cp
        import gpu_solver as solver
        import tetra_fmm
        from magtense_adapter import assemble_tiles
        import magtense_adapter
        def gpu_state():
            return {'cupy_used_bytes':cp.get_default_memory_pool().used_bytes(),
                    'cupy_reserved_bytes':cp.get_default_memory_pool().total_bytes(),
                    'gpu_free_bytes':cp.cuda.Device().mem_info[0]}
        original_tiles = assemble_tiles
        def measured_tiles(*args,**kwargs):
            progress('tile_assembly')
            result = original_tiles(*args,**kwargs)
            progress('tiles_ready',magnetic_cells=result[0].n)
            return result
        magtense_adapter.assemble_tiles = measured_tiles
        original_neighbours = tetra_fmm._neighbours
        last_update = [0.]
        def measured_neighbours(*args,**kwargs):
            for lo,rows in original_neighbours(*args,**kwargs):
                if time.monotonic()-last_update[0] > 30:
                    progress(state['phase'],receivers_processed=lo+len(rows),**gpu_state())
                    last_update[0] = time.monotonic()
                yield lo,rows
        tetra_fmm._neighbours = measured_neighbours
        original_counts = tetra_fmm._CUDANeighbours.counts
        def measured_counts(search,factor):
            import numpy as np
            progress('GPU_neighbour_counting',receivers=len(search.centres),**gpu_state())
            started = time.perf_counter()
            result = original_counts(search,factor)
            count_seconds = time.perf_counter()-started
            # Independent SciPy radius queries on actual exported geometry,
            # including the smallest/largest cells; exact ID/count equality.
            centres,radii = cp.asnumpy(search.centres),cp.asnumpy(search.radii)
            selected = np.unique(np.r_[np.random.default_rng(981).choice(len(radii),128,replace=False),
                                      np.argsort(radii)[:16],np.argsort(radii)[-16:]])
            validation_started = time.perf_counter()
            cpu_groups = tetra_fmm._source_trees(centres,radii)
            query = dict(targets=centres[selected],target_radii=radii[selected])
            expected = [row for _,rows in original_neighbours(centres,radii,factor,32,groups=cpu_groups,**query)
                        for row in rows]
            actual = [row for _,rows in original_neighbours(centres,radii,factor,32,groups=search,**query)
                      for row in rows]
            assert len(actual)==len(expected)==len(selected)
            for receiver,a,b in zip(selected,actual,expected):
                np.testing.assert_array_equal(a,b)
                assert result[receiver]==len(b)
            progress('GPU_neighbour_counts_ready',total_neighbour_pairs=int(result.sum()),
                     neighbour_count_seconds=count_seconds,
                     gpu_neighbour_validation={'actual_mesh_cells':len(radii),'sample_receivers':len(selected),
                         'exact_ids_and_counts_match':True,'reference':'SciPy cKDTree with FP64 radius filter',
                         'seconds':time.perf_counter()-validation_started},**gpu_state())
            return result
        tetra_fmm._CUDANeighbours.counts = measured_counts
        original_gpu_setup = tetra_fmm._CUDAPointFMM.__init__
        def measured_gpu_setup(plan,*args,**kwargs):
            progress('GPU_FMM_tree_setup',**gpu_state())
            original_gpu_setup(plan,*args,**kwargs)
            progress('GPU_FMM_tree_ready',GPU_FMM_plan=plan.info,**gpu_state())
        tetra_fmm._CUDAPointFMM.__init__ = measured_gpu_setup
        original_gpu_field = tetra_fmm._CUDAPointFMM.__call__
        def measured_gpu_field(plan,moments):
            progress('GPU_farfield',gpu_farfield_state='running',receivers_processed=0,
                     GPU_target_points=len(plan.targets),**gpu_state())
            started = time.perf_counter()
            result = original_gpu_field(plan,moments)
            progress('GPU_near_correction',gpu_farfield_state='complete',
                     gpu_farfield_seconds=time.perf_counter()-started,GPU_FMM_plan=plan.info,**gpu_state())
            return result
        tetra_fmm._CUDAPointFMM.__call__ = measured_gpu_field
        original_assemble = tetra_fmm.assemble
        def measured_assemble(*args,**kwargs):
            progress('FMM_operator_setup',**gpu_state())
            result = original_assemble(*args,**kwargs)
            progress('FMM_operator_ready',operator_info=result[2],**gpu_state())
            return result
        tetra_fmm.assemble = measured_assemble
        original_matvec = tetra_fmm.VolumeFMM.__matmul__
        calls = [0]
        def measured_matvec(operator,vector):
            calls[0] += 1
            started = time.perf_counter()
            progress('FMM_matvec',matvec_call=calls[0],matvec_state='running',receivers_processed=0,**gpu_state())
            result = original_matvec(operator,vector)
            progress('FMM_matvec',matvec_call=calls[0],matvec_state='complete',
                     last_matvec_seconds=time.perf_counter()-started,**gpu_state())
            return result
        tetra_fmm.VolumeFMM.__matmul__ = measured_matvec
        original_solve = solver.solve_nonlinear
        def measured_solve(*args,**kwargs):
            progress('nonlinear_solve',**gpu_state())
            result = original_solve(*args,**kwargs)
            progress('nonlinear_converged',solve_info=result[1],**gpu_state())
            return result
        solver.solve_nonlinear = measured_solve
        progress('solver_start',**gpu_state())
        solver.main(['--out',str(EXPORT),'--operator','volume','--k-builder','fmm',
            '--backend','cupy','--dtype','float64','--method','proximal','--tol','1e-8',
            '--fmm-eps','1e-6','--anderson-history','6','--fmm-index-gib','1',
            '--fmm-near-cache-mib','512','--max-tiles','2718864','--maxiter','6000',
            '--fmm-neighbours',neighbours,'--fmm-farfield',farfield])
        progress('complete',**gpu_state())
    except BaseException as error:
        progress('failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
        raise


class GPUUsage:
    class Memory(ctypes.Structure):
        _fields_ = [('total',ctypes.c_ulonglong),('free',ctypes.c_ulonglong),('used',ctypes.c_ulonglong)]
    def __init__(self):
        self.library = ctypes.WinDLL('nvml.dll')
        self.handle = ctypes.c_void_p()
        assert self.library.nvmlInit_v2() == 0
        assert self.library.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.handle)) == 0
    def used(self):
        memory = self.Memory()
        if self.library.nvmlDeviceGetMemoryInfo(self.handle,ctypes.byref(memory)) != 0:
            return None
        return memory.used


def monitor(stage, writer, stream, report, gpu,neighbours,farfield):
    import psutil
    interpreter = LEGACY if stage == 'export' else SOLVER
    command = [str(interpreter),'-u',str(Path(__file__).resolve()),'--worker',stage,
               '--fmm-neighbours',neighbours,'--fmm-farfield',farfield]
    started = time.monotonic()
    attempt = sum(item['stage']==stage for item in report['stages'])+1
    label = f'{stage}_{attempt}_{neighbours}_{farfield}'
    log_path = ROOT/f'large_fmm_{label}.log'
    with log_path.open('w',encoding='utf-8') as log:
        process = subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        root = psutil.Process(process.pid)
        record = {'stage':stage,'command':command,'pid':process.pid,'log':log_path.name,
                  'started_utc':datetime.now(timezone.utc).isoformat(),
                  'attempt_label':label,'neighbour_backend':neighbours,'farfield_backend':farfield,
                  'source_sha256':dict(report['current_source_sha256']),
                  'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'gpu_board_used_baseline_bytes':gpu.used() if gpu else None,
                  'peak_rss_bytes':0,'peak_private_bytes':0,'peak_gpu_board_used_bytes':0}
        report['stages'].append(record)
        next_save = 0.
        while True:
            try:
                processes = [root]+root.children(recursive=True)
            except psutil.NoSuchProcess:
                processes = []
            rss = private = alive = 0
            for item in processes:
                try:
                    memory = item.memory_info()
                    rss += memory.rss
                    private += getattr(memory,'private',memory.vms)
                    alive += 1
                except (psutil.NoSuchProcess,psutil.AccessDenied):
                    pass
            board = gpu.used() if gpu is not None else None
            now = time.monotonic()
            row = [datetime.now(timezone.utc).isoformat(),label,now-started,alive,rss,private,
                   psutil.virtual_memory().available,board]
            writer.writerow(row);stream.flush()
            record.update(elapsed_seconds=now-started,
                          peak_rss_bytes=max(record['peak_rss_bytes'],rss),
                          peak_private_bytes=max(record['peak_private_bytes'],private),
                          peak_gpu_board_used_bytes=max(record['peak_gpu_board_used_bytes'],board or 0))
            if STATUS.exists():
                try:
                    record['worker'] = json.loads(STATUS.read_text())
                except (OSError,ValueError):
                    pass
            # Preserve accepted proximal checkpoints independently of CLI success.
            text = log_path.read_text(encoding='utf-8',errors='replace')
            record['residual_history'] = [{'iteration':int(i),'residual':float(r)} for i,r in
                re.findall(r'prox (\d+): (?:FMM|true) constitutive residual=([\deE+.-]+)',text)]
            if now >= next_save or process.poll() is not None:
                save(REPORT,report);next_save = now+30
            if process.poll() is not None:
                break
            time.sleep(5)
        record['exit_code'] = process.returncode
        record['elapsed_seconds'] = time.monotonic()-started
        record['state'] = 'completed' if process.returncode == 0 else 'failed'
        save(REPORT,report)
    if process.returncode:
        raise RuntimeError(f'{stage} failed with exit code {process.returncode}; see {log_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['export','solve','validate_cuda'])
    parser.add_argument('--validate-cuda-small',action='store_true')
    parser.add_argument('--compress-near',action='store_true')
    parser.add_argument('--assess-fem-bem',action='store_true')
    parser.add_argument('--prototype-fem-bem',action='store_true')
    parser.add_argument('--fem-bem-synthetic-only',action='store_true')
    parser.add_argument('--validate-fem-bem-aedt',action='store_true')
    parser.add_argument('--diagnose-fem-bem-aedt',action='store_true')
    parser.add_argument('--diagnose-fem-bem-identity',action='store_true')
    parser.add_argument('--benchmark-fem-bem-large',action='store_true')
    parser.add_argument('--benchmark-fem-bem-assembly',action='store_true')
    parser.add_argument('--validate-native-transfer',action='store_true')
    parser.add_argument('--fem-bem-assembly-batches',type=int,nargs='+',choices=[64,256,1024],
        help='With assembly probe, compare ACA batch sizes against retained full-cache ranks and factor bytes')
    parser.add_argument('--solve-fem-bem-large',action='store_true')
    parser.add_argument('--fem-bem-h-leaf',type=int,default=64)
    parser.add_argument('--fem-bem-h-eta',type=float,default=2.)
    parser.add_argument('--fem-bem-h-rtol',type=float,default=1e-6)
    parser.add_argument('--fem-bem-h-rank',type=int,default=48)
    parser.add_argument('--fem-bem-h-gib',type=float,default=6.)
    parser.add_argument('--fem-bem-h-storage',choices=['float64','float32'],default='float64')
    parser.add_argument('--fem-bem-h-far-order',type=int,default=8)
    parser.add_argument('--fem-bem-h-row-backend',choices=['auto','direct','shared'],default='auto')
    parser.add_argument('--fem-bem-h-skip-inactive',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--fem-bem-refine-boundary',action='store_true')
    parser.add_argument('--fem-bem-check-solution',type=Path,
        help='With boundary benchmark, check an unchanged retained converged solution and native field samples')
    parser.add_argument('--fem-bem-aedt-worker',type=Path)
    parser.add_argument('--compress-object-count',type=int,default=10,choices=range(1,11))
    parser.add_argument('--compress-patches-per-object',type=int,default=4,choices=range(1,5))
    parser.add_argument('--compress-patch-cells',type=int,default=128)
    parser.add_argument('--compress-patch-mib',type=int,default=256)
    parser.add_argument('--compress-leaf-cells',type=int,default=8)
    parser.add_argument('--compress-eta',type=float,default=2.)
    parser.add_argument('--compress-admissibility',choices=['strong','weak'],default='weak')
    parser.add_argument('--compress-error-budget',choices=['relative','global'],default='global')
    parser.add_argument('--compress-decomposition',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--compress-tolerances',type=float,nargs='+',default=[1e-3,1e-4,1e-6])
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--solve-only',action='store_true')
    parser.add_argument('--fmm-neighbours',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--fmm-farfield',choices=['cpu','cuda'],default='cpu')
    args = parser.parse_args()
    if args.validate_native_transfer:
        sys.path.insert(0,str(TOOL));validate_native_transfer();return
    if args.fem_bem_check_solution and (not args.benchmark_fem_bem_large or args.solve_fem_bem_large):
        parser.error('--fem-bem-check-solution requires --benchmark-fem-bem-large without --solve-fem-bem-large')
    if args.fem_bem_assembly_batches and (not args.benchmark_fem_bem_assembly or
            sorted(args.fem_bem_assembly_batches)!=[64,256,1024]):
        parser.error('--fem-bem-assembly-batches requires --benchmark-fem-bem-assembly and 64 256 1024')
    if args.benchmark_fem_bem_assembly:
        sys.path.insert(0,str(TOOL))
        benchmark_fem_bem_assembly(args.fem_bem_assembly_batches)
        return
    if args.diagnose_fem_bem_identity:
        sys.path.insert(0,str(TOOL))
        diagnose_fem_bem_identity()
        return
    if args.benchmark_fem_bem_large or args.solve_fem_bem_large:
        sys.path.insert(0,str(TOOL))
        benchmark_fem_bem_large(args)
        return
    if args.fem_bem_aedt_worker:
        sys.path.insert(0,str(TOOL))
        fem_bem_aedt_export(args.fem_bem_aedt_worker)
        return
    if args.validate_fem_bem_aedt:
        sys.path.insert(0,str(TOOL))
        validate_fem_bem_aedt()
        return
    if args.diagnose_fem_bem_aedt:
        sys.path.insert(0,str(TOOL))
        diagnose_fem_bem_aedt()
        return
    if args.fem_bem_synthetic_only and not args.prototype_fem_bem:
        parser.error('--fem-bem-synthetic-only requires --prototype-fem-bem')
    if args.prototype_fem_bem:
        sys.path.insert(0,str(TOOL))
        prototype_fem_bem(args.fem_bem_synthetic_only)
        return
    if args.assess_fem_bem:
        assess_fem_bem()
        return
    if args.compress_near:
        if args.compress_patch_cells<1 or args.compress_patch_mib<1:
            parser.error('Compression patch size and memory budget must be positive')
        compress_near(args)
        return
    if args.worker:
        worker(args.worker,args.fmm_neighbours,args.fmm_farfield)
        return
    if args.validate_cuda_small:
        subprocess.run([str(SOLVER),'-u',str(Path(__file__).resolve()),'--worker','validate_cuda'],
                       cwd=REPO,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        return
    import psutil
    try:
        gpu = GPUUsage()
    except (OSError,AssertionError,AttributeError):
        gpu = None
    report = {'state':'running','started_utc':datetime.now(timezone.utc).isoformat(),
              'expected_magnetic_cells':2718864,'expected_total_mesh_cells':8679245,
              'memory_sample_seconds':5,'gpu_usage_scope':'whole board, includes other applications',
              'hardware':{'RAM_bytes':psutil.virtual_memory().total,'cpu':os.environ.get('PROCESSOR_IDENTIFIER')},
              'gpu_board_used_baseline_bytes':gpu.used() if gpu is not None else None,'stages':[]}
    if REPORT.exists():
        if not args.solve_only:
            raise FileExistsError('Existing run report retained; use --solve-only to append a new solve attempt')
        previous = json.loads(REPORT.read_text(encoding='utf-8'))
        if previous['state']=='running' and psutil.pid_exists(previous['stages'][-1]['pid']):
            raise RuntimeError('An owned run is still active')
        for item in previous['stages']:
            item.setdefault('source_sha256',dict(previous['current_source_sha256']))
        report = previous
        report['state'] = 'running'
        report.pop('error',None)
    for path in ['tetra_fmm.py','gpu_solver.py','constitutive_prox.py','tetra_cuda.cu','aedtplt.py','aedt_export.py']:
        report.setdefault('current_source_sha256',{})[path] = hashlib.sha256((TOOL/path).read_bytes()).hexdigest()
    if args.solve_only:
        assert json.loads((EXPORT/'model.json').read_text(encoding='utf-8'))['total_magnetic_tetra'] == 2718864
    resource_path = ROOT/'large_fmm_resources.csv'
    append = args.solve_only and resource_path.exists()
    with resource_path.open('a' if append else 'w',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        if not append:
            writer.writerow(['UTC','stage','elapsed_seconds','processes','RSS_bytes','private_commit_bytes',
                             'system_available_RAM_bytes','GPU_board_used_bytes'])
        try:
            if not args.solve_only:
                monitor('export',writer,stream,report,gpu,args.fmm_neighbours,args.fmm_farfield)
            if not args.export_only:
                monitor('solve',writer,stream,report,gpu,args.fmm_neighbours,args.fmm_farfield)
                info = json.loads((EXPORT/'gpu_solve_info.json').read_text())
                assert info['converged'] and info['residual'] <= 1e-8
                report['solve_info'] = info
            report['state'] = 'export_complete' if args.export_only else 'completed'
        except BaseException as error:
            report.update(state='failed',error=f'{type(error).__name__}: {error}')
            save(REPORT,report)
            raise
    save(REPORT,report)


if __name__ == '__main__':
    main()
