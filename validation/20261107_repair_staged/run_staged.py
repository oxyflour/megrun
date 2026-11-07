"""Bounded staged diagnosis using the unchanged native tetrahedron operator."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '20261107_gap3mm_vacoflux50'
TOOL = Path(r'C:\Projects\magtense-gpu\skill-test\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path[:0] = [str(BASE), str(TOOL)]
from gpu_solver import MaterialLaws, verify_tensor_against_magtense, hash_geometry
from magtense_adapter import assemble_tiles
from magtense.magstatics import get_H_field
import gpu_solver_dense_newton as dense

ROOT.mkdir(exist_ok=True)
started = time.perf_counter()
record = {'started_UTC': datetime.now(timezone.utc).isoformat(), 'budget_seconds': 1200,
          'scope': 'Staged diagnosis and repair; cached geometry; original source project untouched', 'cases': []}
(ROOT / 'session.json').write_text(json.dumps(record, indent=2))
model = json.loads((BASE / 'coarse_export/model.json').read_text())
points = np.loadtxt(BASE / 'points.csv', delimiter=',', skiprows=1)
full_K = cp.asarray(np.load(BASE / 'coarse_export/K_float64.npy', mmap_mode='r'))
offset = 0
for obj in model['objects']:
    obj['_start'] = offset
    offset += obj['mesh_stats']['tetra_count']

cases = [('single_pm', 'A', None), ('pair_pm', None, None),
         ('single_linear_10', 'A', 10), ('single_linear_100', 'A', 100),
         ('single_linear_1000', 'A', 1000), ('single_real_bh', 'A', 'BH')]
for name, camera, mu in cases:
    if time.perf_counter() - started > 600:
        raise RuntimeError('Staged pilot exceeded its 10-minute allocation')
    objects, selection = [], []
    for original in model['objects']:
        if camera and not original['name'].startswith(camera + '_'):
            continue
        is_yoke = 'YOKE' in original['name']
        if is_yoke and mu is None:
            continue
        obj = copy.deepcopy(original)
        count = obj['mesh_stats']['tetra_count']
        selection.extend(range(obj.pop('_start'), original['_start'] + count))
        obj['mesh_npz'] = str(BASE / 'coarse_export' / obj['mesh_npz'])
        if is_yoke and mu != 'BH':
            obj['properties'] = {'kind': 'linear', 'mu_r': mu}
        objects.append(obj)
    folder = ROOT / name
    folder.mkdir(exist_ok=True)
    case_model = {**model, 'objects': objects, 'total_magnetic_tetra': len(selection)}
    (folder / 'model.json').write_text(json.dumps(case_model, indent=2))
    tiles, curves, stats = assemble_tiles(objects, folder)
    indices = cp.asarray((3 * np.asarray(selection)[:, None] + np.arange(3)).ravel())
    K = full_K[indices[:, None], indices[None, :]]
    diag = cp.stack([K[3*i:3*i+3, 3*i:3*i+3] for i in range(tiles.n)])
    check = verify_tensor_against_magtense(tiles, K, cp, get_H_field)
    laws = MaterialLaws(tiles, curves, cp, 'float64')
    item = {'name': name, 'tiles': tiles.n, 'K_native_check_relative': check,
            'material': mu, 'geometry_sha256': hash_geometry(tiles)}
    t0 = time.perf_counter()
    try:
        if mu != 'BH':
            row_chi = cp.repeat(laws.chi, 3)
            J = -row_chi[:, None] * K
            index = cp.arange(3*tiles.n)
            J[index, index] += 1
            M = cp.linalg.solve(J, laws.rem.ravel()).reshape(tiles.n, 3)
            del J
            item['solver'] = 'Exact dense solution of the linear constitutive system'
        else:
            M, nonlinear_info = dense.solve_nonlinear(K, diag, laws, tol=1e-8,
                                                       maxiter=60, verbose=True)
            item['nonlinear_info'] = nonlinear_info
        H = (K @ M.ravel()).reshape(tiles.n, 3)
        R = M - laws.eval(H)[0]
        residual = float(cp.linalg.norm(R).item()) / max(float(cp.linalg.norm(M).item()), 1.)
        item.update(converged=residual < 1e-8, residual=residual,
                    max_M_Apm=float(cp.max(cp.linalg.norm(M, axis=1)).item()))
        if not item['converged']:
            raise RuntimeError('Independent residual check failed')
        tiles.M = cp.asnumpy(M)
        B = 4e-7 * np.pi * get_H_field(tiles, points)
        np.savetxt(folder / 'B.csv', np.c_[points, B], delimiter=',',
                   header='x_m,y_m,z_m,Bx_T,By_T,Bz_T', comments='')
        np.savez_compressed(folder / 'solution.npz', M_Apm=tiles.M)
    except Exception as error:
        item.update(converged=False, failure=repr(error))
    item['seconds'] = time.perf_counter() - t0
    record['cases'].append(item)
    record['elapsed_seconds'] = time.perf_counter() - started
    (ROOT / 'session.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(item), flush=True)
    del K, diag, laws
print('STAGED PILOT FINISHED', flush=True)
