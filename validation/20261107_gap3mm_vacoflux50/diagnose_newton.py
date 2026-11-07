"""Audit the failed two Newton steps; does not launch a new simulation."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import cupy as cp

ROOT = Path(__file__).resolve().parent
TOOL = ROOT.parents[1] / 'skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
sys.path.insert(0, str(TOOL))
# Replay the failed v2 steps with the case's existing frozen original solver.
from gpu_solver_v2_original import MaterialLaws, make_preconditioner, gmres_solve, apply_tangent
from magtense_adapter import assemble_tiles


def scalar(x):
    return float(x.item())


started = time.perf_counter()
folder = ROOT / 'coarse_export'
model = json.loads((folder / 'model.json').read_text())
tiles, curves, _ = assemble_tiles(model['objects'], folder)
K = cp.asarray(np.load(folder / 'K_float64.npy', mmap_mode='r'))
diag = cp.stack([K[3*i:3*i+3, 3*i:3*i+3] for i in range(tiles.n)])
law = MaterialLaws(tiles, curves, cp, 'float64')
M = law.rem.copy()
rem_norm = scalar(cp.linalg.norm(law.rem))


def state(m):
    h = (K @ m.ravel()).reshape(tiles.n, 3)
    f, a, b, u = law.eval(h)
    r = m - f
    norm = scalar(cp.linalg.norm(r))
    scale = max(scalar(cp.linalg.norm(m)), rem_norm, np.sqrt(tiles.n))
    return r, (h, a, b, u), norm, norm / scale


records = []
for iteration in range(2):
    R, (H, a, b, u), absolute, relative = state(M)
    inverse = make_preconditioner(cp, diag, a, b, u)

    def pinv(z):
        return (inverse @ z.reshape(tiles.n, 3, 1)).ravel()

    def J(z):
        dm = pinv(z)
        dh = (K @ dm).reshape(tiles.n, 3)
        return (dm.reshape(tiles.n, 3) - apply_tangent(cp, a, b, u, dh)).ravel()

    z, info = gmres_solve(cp, J, -R, rtol=1e-5, restart=30, maxiter=100)
    delta = pinv(z).reshape(tiles.n, 3)
    linear_error = scalar(cp.linalg.norm(J(z) + R.ravel())) / absolute
    trials = []
    selected = None
    for j in range(13):
        step = 2. ** -j
        _, _, new_absolute, new_relative = state(M + step * delta)
        trials.append({'step': step, 'absolute_residual': new_absolute,
                       'relative_residual': new_relative,
                       'absolute_ratio': new_absolute / absolute,
                       'relative_ratio': new_relative / relative})
        if selected is None and new_relative < relative:
            selected = step
    record = {'iteration': iteration, 'relative_residual': relative,
              'absolute_residual': absolute, 'gmres_info': int(info),
              'actual_linear_relative_residual': linear_error,
              'M_norm': scalar(cp.linalg.norm(M)), 'delta_norm': scalar(cp.linalg.norm(delta)),
              'selected_original_step': selected, 'trials': trials}
    records.append(record)
    print(json.dumps(record), flush=True)
    if selected is None:
        break
    M += selected * delta
output = {'purpose': 'Read-only replay/diagnosis of the failed first two Newton steps',
          'seconds': time.perf_counter() - started, 'records': records}
(ROOT / 'nonconvergence_diagnostic.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
