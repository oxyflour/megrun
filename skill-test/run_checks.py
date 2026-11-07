"""Reproducible checks against the unmodified skill extracted from the ZIP."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
TOOL = ROOT / 'magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
records = []


def run(name, *args, timeout=180):
    started = time.perf_counter()
    command = [sys.executable, *map(str, args)]
    try:
        process = subprocess.run(command, cwd=TOOL, capture_output=True, text=True,
                                 encoding='utf-8', errors='replace', timeout=timeout,
                                 env={**os.environ, 'PYTHONIOENCODING': 'utf-8',
                                      'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
        code, output = process.returncode, process.stdout + process.stderr
    except subprocess.TimeoutExpired as error:
        code, output = -1, f'Timed out after {timeout}s: {error}'
    (RESULTS / f'{name}.log').write_text(output, encoding='utf-8')
    record = {'name': name, 'command': command, 'exit_code': code,
              'seconds': time.perf_counter() - started}
    records.append(record)
    if code == 0 and args[0] == 'gpu_solver.py' and '--out' in args:
        model_out = Path(args[args.index('--out') + 1])
        for filename in ['gpu_B.csv', 'gpu_magnetization.npz', 'gpu_solve_info.json']:
            (RESULTS / f'{name}_{filename}').write_bytes((model_out / filename).read_bytes())
    (RESULTS / 'checks.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'{name}: exit={code}, {record["seconds"]:.2f}s', flush=True)
    print(output[-2500:], flush=True)
    return code


def make_nonlinear_fixture(out):
    import numpy as np
    (out / 'mesh').mkdir(parents=True, exist_ok=True)
    mu0 = 4e-7 * np.pi
    base = np.array([[0, 0, 0], [.001, 0, 0], [0, .001, 0], [0, 0, .001]])
    objects = []
    curves = [([0, 5000, 20000, 100000], [0, 60000, 80000, 100000]),
              ([0, 6000, 15000, 120000], [0, 30000, 45000, 60000])]
    for i, (h, m) in enumerate(curves):
        mesh = f'mesh/soft_{i}.npz'
        np.savez_compressed(out / mesh, tetra_m=(base + [(-.02 if i == 0 else .02), 0, 0])[None])
        bh = np.column_stack((h, mu0 * (np.array(h) + m))).tolist()
        objects.append({'name': f'soft_{i}', 'material': f'SYNTHETIC_BH_{i}',
                        'properties': {'kind': 'nonlinear', 'bh_H_Apm_B_T': bh},
                        'mesh_npz': mesh, 'mesh_stats': {'tetra_count': 1,
                        'volume_m3': .001 ** 3 / 6, 'tet4': 1, 'tet10': 0,
                        'curved': False, 'orientation_flipped': 0}})
    np.savetxt(out / 'points.csv', [[0, .01, .01], [.001, .01, .02]], delimiter=',',
               header='x_m,y_m,z_m', comments='')
    model = {'schema_version': 1, 'source': {'project': 'SYNTHETIC TEST - NOT MAXWELL',
             'design': 'two_distinct_bh', 'solution': 'none', 'model_unit': 'm'},
             'mesh_unit': 'm', 'points': 'points.csv', 'maxwell_field': 'maxwell_B.fld',
             'skipped': [], 'total_magnetic_tetra': 2, 'objects': objects}
    (out / 'model.json').write_text(json.dumps(model, indent=2), encoding='utf-8')


versions = {'python': sys.version, 'executable': sys.executable}
for package in ['numpy', 'scipy', 'pytest', 'magtense', 'pyaedt', 'cupy-cuda13x']:
    versions[package] = importlib.metadata.version(package)
(RESULTS / 'versions.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')

run('numpy_self_test', 'gpu_solver.py', '--self-test', '--backend', 'numpy')
run('pytest', '-m', 'pytest', '-q', '-ra', 'tests', f'--junitxml={RESULTS / "pytest.xml"}')
run('cuda_environment', '-c', 'import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); '
    'print(cp.cuda.runtime.getDeviceProperties(0)); cp.show_config(); '
    'print("CUDA reduction:", float(cp.sum(cp.arange(1000, dtype=cp.float64)).get()))')
demo = RESULTS / 'demo'
run('demo_create', 'bridge.py', 'demo', '--out', demo)
run('demo_inspect', 'bridge.py', 'inspect', '--out', demo)
run('demo_dry_run', 'bridge.py', 'solve', '--out', demo, '--dry-run')
run('demo_native_cpu', 'bridge.py', 'solve', '--out', demo)
run('demo_numpy_fp64', 'gpu_solver.py', '--out', demo, '--backend', 'numpy',
    '--dtype', 'float64', '--cache-k')
if (demo / 'gpu_B.csv').exists():
    (RESULTS / 'demo_numpy_B.csv').write_bytes((demo / 'gpu_B.csv').read_bytes())
gpu_status = run('demo_cupy_fp64', 'gpu_solver.py', '--out', demo, '--backend', 'cupy',
                 '--dtype', 'float64', '--cache-k')
if gpu_status == 0:
    (RESULTS / 'demo_cupy_fp64_B.csv').write_bytes((demo / 'gpu_B.csv').read_bytes())
    run('demo_cupy_fp32', 'gpu_solver.py', '--out', demo, '--backend', 'cupy',
        '--dtype', 'float32', '--cache-k')

nonlinear = RESULTS / 'two_bh'
make_nonlinear_fixture(nonlinear)
run('two_bh_numpy_fp64', 'gpu_solver.py', '--out', nonlinear, '--backend', 'numpy',
    '--dtype', 'float64', '--uniform-H', '65000', '0', '0', '--cache-k')
if (nonlinear / 'gpu_B.csv').exists():
    (RESULTS / 'two_bh_numpy_B.csv').write_bytes((nonlinear / 'gpu_B.csv').read_bytes())
run('two_bh_cupy_fp64', 'gpu_solver.py', '--out', nonlinear, '--backend', 'cupy',
    '--dtype', 'float64', '--uniform-H', '65000', '0', '0', '--cache-k')
print('Finished. Evidence:', RESULTS, flush=True)
