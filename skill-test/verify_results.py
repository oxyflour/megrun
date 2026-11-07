"""Tighter CPU/GPU checks and residual recomputation for synthetic fixtures."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
TOOL = ROOT / 'magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
RESULTS = ROOT / 'results'
sys.path.insert(0, str(TOOL))
from bridge import read_model
from gpu_solver import MaterialLaws
from magtense_adapter import assemble_tiles

records = []


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def run(name, *args):
    process = subprocess.run([sys.executable, *map(str, args)], cwd=TOOL,
                             capture_output=True, encoding='utf-8', errors='replace',
                             timeout=120, env={**os.environ, 'OMP_NUM_THREADS': '2',
                                              'MKL_NUM_THREADS': '2', 'PYTHONIOENCODING': 'utf-8'})
    (RESULTS / f'{name}.log').write_text(process.stdout + process.stderr, encoding='utf-8')
    if process.returncode:
        raise RuntimeError(f'{name} failed: {process.stdout}\n{process.stderr}')
    print(f'{name}: PASS', flush=True)


for model_name, field in [('demo', [0, 0, 0]), ('two_bh', [65000, 0, 0])]:
    folder = RESULTS / model_name
    model = read_model(folder)
    tiles, curves, _ = assemble_tiles(model['objects'], folder)
    solutions = {}
    for backend, dtype in [('numpy', 'float64'), ('cupy', 'float64'), ('cupy', 'float32')]:
        name = f'{model_name}_{backend}_{dtype}_verified'
        tolerance = '1e-10' if dtype == 'float64' else '1e-5'
        run(name, 'gpu_solver.py', '--out', folder, '--backend', backend, '--dtype', dtype,
            '--cache-k', '--tol', tolerance, '--gmres-rtol', '1e-6' if dtype == 'float64' else '5e-4',
            '--uniform-H', *field)
        info = json.loads((folder / 'gpu_solve_info.json').read_text())
        m = np.load(folder / 'gpu_magnetization.npz')['M_Apm'].astype(float)
        b = np.loadtxt(folder / 'gpu_B.csv', delimiter=',', skiprows=1)[:, 3:]
        k = np.load(folder / f'K_{dtype}.npy').astype(float)
        law = MaterialLaws(tiles, curves or [], np, 'float64')
        h = (k @ m.ravel()).reshape(tiles.n, 3) + field
        residual = m - law.eval(h)[0]
        scale = max(float(np.linalg.norm(m)), float(np.linalg.norm(law.rem)), np.sqrt(tiles.n))
        independent_residual = float(np.linalg.norm(residual) / scale)
        assert independent_residual < (1e-9 if dtype == 'float64' else 1e-5)
        key = f'{backend}_{dtype}'
        solutions[key] = (m, b)
        record = {'fixture': model_name, 'backend': backend, 'dtype': dtype,
                  'tiles': tiles.n, 'reported_residual': info['residual'],
                  'recomputed_fp64_residual': independent_residual,
                  'iterations': info['iterations'], 'cache_hit': info['loaded_from_cache'],
                  'native_K_H_relative_error': info['K_magtense_field_check_relative_error'],
                  'M_norm_Apm_by_tile': np.linalg.norm(m, axis=1).tolist(),
                  'H_norm_Apm_by_tile': np.linalg.norm(h, axis=1).tolist(),
                  'K_build_or_load_seconds': info['K_build_or_load_seconds'],
                  'solve_seconds': info['solve_seconds']}
        reference_m, reference_b = solutions['numpy_float64']
        record['M_relative_L2_vs_numpy_fp64'] = relative(m, reference_m)
        record['B_relative_L2_vs_numpy_fp64'] = relative(b, reference_b)
        for filename in ['gpu_B.csv', 'gpu_magnetization.npz', 'gpu_solve_info.json']:
            (RESULTS / f'{name}_{filename}').write_bytes((folder / filename).read_bytes())
        records.append(record)
        print(json.dumps(record), flush=True)
    assert relative(solutions['cupy_float64'][0], solutions['numpy_float64'][0]) < 1e-9
    assert relative(solutions['cupy_float32'][0], solutions['numpy_float64'][0]) < 1e-5

run('demo_native_tight', 'bridge.py', 'solve', '--out', RESULTS / 'demo', '--max-error', '1e-10')
native = np.loadtxt(RESULTS / 'demo/magtense_B.csv', delimiter=',', skiprows=1)[:, 3:]
gpu = np.loadtxt(RESULTS / 'demo_cupy_float64_verified_gpu_B.csv', delimiter=',', skiprows=1)[:, 3:]
native_error = relative(gpu, native)
assert native_error < 1e-8
summary = {'source': 'Synthetic fixtures only; no Maxwell reference', 'checks': records,
           'demo_B_relative_L2_gpu_vs_native_CPU': native_error}
(RESULTS / 'verification.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps({'demo_B_relative_L2_gpu_vs_native_CPU': native_error}), flush=True)
