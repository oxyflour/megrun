from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

TOOL = Path(r'C:\Projects\magtense-gpu\skill-test\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0, str(TOOL))
from gpu_solver import MaterialLaws, MU0
from proximal_solver import solve


def test_recoil_magnet_sphere_analytic():
    tiles = SimpleNamespace(magnet_type=np.array([1]), stfcn_index=np.array([1]),
        mu_r_ea=np.array([1.1]), mu_r_oa=np.array([1.1]),
        M_rem=np.array([1.2/MU0]), u_ea=np.array([[1., 0, 0]]))
    laws = MaterialLaws(tiles, [], np, 'float64')
    M, info = solve(-np.eye(3)/3, laws, tolerance=1e-11)
    np.testing.assert_allclose(M[0], [1.2/MU0/(1+.1/3), 0, 0], rtol=1e-10, atol=1e-7)
    assert info['residual'] < 1e-11


@pytest.mark.parametrize('remanence', [1.2/MU0, 1e7])
def test_soft_magnet_unsaturated_and_saturated_analytic(remanence):
    tiles = SimpleNamespace(magnet_type=np.array([1, 2]), stfcn_index=np.array([1, 1]),
        mu_r_ea=np.ones(2), mu_r_oa=np.ones(2), M_rem=np.array([remanence, 0]),
        u_ea=np.array([[1., 0, 0], [1., 0, 0]]))
    h = np.array([0., 300., 16000., 200000.])
    magnetization = np.array([0., 1.5e6, 1.8e6, 1.8e6])
    curve = np.c_[h, MU0*(h+magnetization)]
    laws = MaterialLaws(tiles, [curve], np, 'float64')
    K = np.kron(np.array([[-1/3, .1], [.1, -1/3]]), np.eye(3))
    M, info = solve(K, laws, tolerance=1e-10)
    if remanence < 1e6:
        expected = .1*remanence / (1/3 + 1/5000)
        assert .1*remanence-expected/3 < 300
    else:
        expected = magnetization[-1]
        assert .1*remanence-expected/3 > h[-1]
    np.testing.assert_allclose(M[1], [expected, 0, 0], rtol=1e-8, atol=1e-5)
    assert info['residual'] < 1e-10
