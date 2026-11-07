"""Analytic hard/soft two-tile case exposing the changing-denominator bug."""
from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
TOOL = ROOT.parents[1] / 'skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
sys.path.insert(0, str(TOOL))
# The failure oracle is the case's existing, byte-identical v2 snapshot.
original_spec = importlib.util.spec_from_file_location('original_unmodified_solver', ROOT / 'gpu_solver_v2_original.py')
original = importlib.util.module_from_spec(original_spec)
original_spec.loader.exec_module(original)
spec = importlib.util.spec_from_file_location('corrected', ROOT / 'gpu_solver_corrected.py')
corrected = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corrected)


def test_absolute_merit_converges_to_analytic_solution():
    t = SimpleNamespace(n=2, magnet_type=np.array([1, 2]), stfcn_index=np.array([1, 1]),
        mu_r_ea=np.ones(2), mu_r_oa=np.ones(2), M_rem=np.array([1.2 / original.MU0, 0.]),
        u_ea=np.tile([1., 0., 0.], (2, 1)))
    curve = np.loadtxt(ROOT / 'vacoflux50_bh.csv', delimiter=',', skiprows=1)
    K = np.kron(np.array([[-1/3, .1], [.1, -1/3]]), np.eye(3))
    diag = np.stack([-np.eye(3) / 3] * 2)
    old_law = original.MaterialLaws(t, [curve], np, 'float64')
    with pytest.raises(original.NonConvergence, match='line search failed'):
        original.solve_nonlinear(K, diag, old_law, tol=1e-10, gmres_rtol=1e-8, verbose=False)
    law = corrected.MaterialLaws(t, [curve], np, 'float64')
    m, info = corrected.solve_nonlinear(K, diag, law, tol=1e-10, gmres_rtol=1e-8, verbose=False)
    chi = curve[1, 1] / (original.MU0 * curve[1, 0]) - 1
    expected = .1 * t.M_rem[0] / (1/3 + 1/chi)
    np.testing.assert_allclose(m, [[t.M_rem[0], 0, 0], [expected, 0, 0]], rtol=1e-9, atol=1e-8)
    assert info['residual'] < 1e-10
    h = (K @ m.ravel()).reshape(2, 3)
    assert 0 < h[1, 0] < 300  # Verify the exact first-segment solution applies.
