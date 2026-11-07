"""Diagnostic excitation continuation, ending at the exact original Br and law."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
TOOL = Path(r'C:\Projects\magtense-gpu\skill-test\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0, str(TOOL))
import gpu_solver_dense_newton as baseline

base_solve = baseline.solve_nonlinear


def continuation(K, diagK, laws, **kwargs):
    xp, n = laws.xp, laws.n
    if kwargs.get('H_external') is not None and np.any(np.asarray(kwargs['H_external']) != 0):
        raise ValueError('This diagnostic continuation is scoped to zero applied field')
    original_rem = laws.rem.copy()
    start = time.perf_counter()
    # Exact small-excitation linearization gives an initial state without Maxwell fields.
    _, alpha, beta, u = laws.eval(xp.zeros((n, 3), dtype=K.dtype))
    tangent = alpha[:, None, None] * xp.eye(3, dtype=K.dtype)[None]
    jacobian = -xp.matmul(tangent, K.reshape(n, 3, 3*n)).reshape(3*n, 3*n)
    index = xp.arange(3*n)
    jacobian[index, index] += 1
    m = xp.linalg.solve(jacobian, original_rem.reshape(-1) * .01).reshape(n, 3)
    del jacobian, tangent
    previous = .01
    stages = []
    try:
        for factor in np.r_[.01, np.arange(.05, 1.001, .05)]:
            record = json.loads((ROOT / 'run_record.json').read_text())
            deadline = datetime.fromisoformat(record['study_started_UTC']).timestamp() + 3000
            if time.time() > deadline:
                raise RuntimeError('Approved study time exhausted during diagnostic continuation')
            laws.rem = original_rem * float(factor)
            initial = m * (float(factor) / previous)
            options = {**kwargs, 'M_initial': initial}
            print(f'Excitation continuation Br scale={factor:.3f}', flush=True)
            try:
                m, info = base_solve(K, diagK, laws, **options)
            except Exception as error:
                failure = {'state': 'Unconverged diagnostic initial guess; NOT a solution',
                           'Br_scale': float(factor), 'error': str(error), 'completed_stages': stages}
                (ROOT / 'continuation_failure.json').write_text(json.dumps(failure, indent=2))
                raise
            stages.append({'Br_scale': float(factor), **info})
            previous = float(factor)
    finally:
        laws.rem = original_rem
    info['continuation_stages'] = stages
    info['continuation_uses_Maxwell_initial_guess'] = False
    info['iterations'] = sum(s['iterations'] for s in stages)
    info['solve_seconds'] = time.perf_counter() - start
    info['linear_solver'] = 'dense_lu_with_excitation_continuation_diagnostic'
    (ROOT / 'continuation_stages.json').write_text(json.dumps(stages, indent=2), encoding='utf-8')
    return m, info


baseline.solve_nonlinear = continuation

if __name__ == '__main__':
    baseline.main()
