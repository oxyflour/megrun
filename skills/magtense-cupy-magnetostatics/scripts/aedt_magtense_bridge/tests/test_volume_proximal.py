from itertools import permutations
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_solver import MU0, MaterialLaws, NonConvergence, solve_nonlinear
from tetra_volume import assemble, quadrature, verify


def material(kinds, rem, mu=None, ids=None):
    n = len(kinds)
    return SimpleNamespace(magnet_type=np.array(kinds), stfcn_index=np.array(ids or [1]*n),
        mu_r_ea=np.array(mu or [1.]*n), mu_r_oa=np.array(mu or [1.]*n),
        M_rem=np.array(rem), u_ea=np.tile([1.,0.,0.],(n,1)))


def test_positive_quadrature_exact_tetrahedron_moments():
    v = np.array([[[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]]])
    for level in range(4):
        q = quadrature(v, level)[0]
        assert len(q) == 4*8**level
        assert np.all(q > 0) and np.all(q.sum(1) < 1)
        np.testing.assert_allclose(q.mean(0), .25, atol=1e-14)
        np.testing.assert_allclose((q*q).mean(0), .1, atol=1e-14)
        np.testing.assert_allclose(np.mean(q[:,0]*q[:,1]), .05, atol=1e-14)


def test_newton_globalization_has_analytic_hard_soft_solution():
    t = material([1,2],[1.2/MU0,0])
    h = np.array([0.,300,16000,200000])
    curve = np.c_[h,MU0*(h+[0,1.5e6,1.8e6,1.8e6])]
    law = MaterialLaws(t,[curve],np,'float64')
    K = np.kron([[-1/3,.1],[.1,-1/3]],np.eye(3))
    m, info = solve_nonlinear(K,np.stack([-np.eye(3)/3]*2),law,tol=1e-10,verbose=False)
    expected = .1*t.M_rem[0]/(1/3+1/5000)
    np.testing.assert_allclose(m,[[t.M_rem[0],0,0],[expected,0,0]],rtol=1e-9,atol=1e-6)
    assert info['converged']


def test_proximal_external_field_exact_linear_sphere():
    law = MaterialLaws(material([3],[0],mu=[10]),[],np,'float64')
    K = -np.eye(3)/3
    m, info = solve_nonlinear(K,K[None],law,method='proximal',H_external=[100,0,0],
                              tol=1e-11,verbose=False)
    np.testing.assert_allclose(m,[[225,0,0]],rtol=1e-10,atol=1e-8)
    assert info['residual'] < 1e-11


def test_proximal_distinct_curves_and_saturation():
    law_tiles = material([1,2,2],[1e7,0,0],ids=[1,1,2])
    h = np.array([0.,300.,16000.,200000.])
    curves = [np.c_[h,MU0*(h+m)] for m in [np.array([0,1.5e6,1.8e6,1.8e6]),
                                         np.array([0,.5e6,.9e6,.9e6])]]
    law = MaterialLaws(law_tiles,curves,np,'float64')
    coupling = np.array([[-1/3,.1,.1],[.1,-1/3,0],[.1,0,-1/3]])
    K = np.kron(coupling,np.eye(3))
    m, info = solve_nonlinear(K,np.stack([-np.eye(3)/3]*3),law,
                              method='proximal',tol=1e-10,verbose=False)
    np.testing.assert_allclose(m[:,0],[1e7,1.8e6,.9e6],rtol=1e-9)
    np.testing.assert_allclose(m[:,1:],0,atol=1e-9)
    assert info['residual'] < 1e-10


def test_proximal_nonconvergence_is_not_success():
    law = MaterialLaws(material([3],[0],mu=[10]),[],np,'float64')
    K = -np.eye(3)/3
    with pytest.raises(NonConvergence):
        solve_nonlinear(K,K[None],law,method='proximal',H_external=[100,0,0],maxiter=0,verbose=False)


def test_native_volume_cache_and_direct_field_consistency(tmp_path):
    ms = pytest.importorskip('magtense.magstatics')
    v = np.array([[[0,0,0],[.001,0,0],[0,.001,0],[0,0,.001]]],float)
    tiles = ms.Tiles(n=1,vertices=np.asfortranarray(v.transpose(0,2,1)),tile_type=5,
                     M_rem=1e5,easy_axis=[1,0,0],mu_r_ea=1,mu_r_oa=1)
    cache = tmp_path/'K.npy'
    K,diag,info = assemble(tiles,np,cache=cache,near_levels=0)
    assert not info['loaded_from_cache']
    assert verify(tiles,K,np,near_levels=0) < 1e-12
    before = tiles.M.copy()
    np.testing.assert_allclose(np.trace(diag[0]),-1,atol=1e-12)
    K2,_,info2 = assemble(tiles,np,cache=cache,near_levels=0)
    assert info2['loaded_from_cache']
    np.testing.assert_array_equal(K,K2)
    # A point-operator cache must never be mistaken for the volume operator.
    metadata_path = Path(str(cache)+'.json')
    meta = json.loads(metadata_path.read_text()); meta['schema']='magtense-denseK-1'
    metadata_path.write_text(json.dumps(meta))
    _,_,info3 = assemble(tiles,np,cache=cache,near_levels=0)
    assert not info3['loaded_from_cache']
    np.testing.assert_array_equal(tiles.M,before)


def test_volume_integration_against_independent_analytic_prism():
    ms = pytest.importorskip('magtense.magstatics')
    side = .001
    v = np.array([[[0,0,0],np.eye(3)[p[0]]*side,
                    (np.eye(3)[p[0]]+np.eye(3)[p[1]])*side,[side]*3]
                  for p in permutations(range(3))])
    tiles = ms.Tiles(n=6,vertices=np.asfortranarray(v.transpose(0,2,1)),tile_type=5,
                     M_rem=0,easy_axis=[1,0,0],mu_r_ea=1,mu_r_oa=1)
    prism = ms.Tiles(n=1,tile_type=8,size=[side]*3,offset=[side/2]*3,
                     M_rem=0,easy_axis=[1,0,0],mu_r_ea=1,mu_r_oa=1)
    for shift in ([0,0,0],[side,0,0]):
        target = np.array([[side/2]*3])+shift
        reference = ms.get_demag_tensor(prism,target,obs_size=np.array([[side]*3]))[0,0]
        q = quadrature(v+shift,3)
        averaged = ms.get_demag_tensor(tiles,q.reshape(-1,3)).sum(0).mean(0)
        assert np.linalg.norm(averaged-reference)/np.linalg.norm(reference) < 5e-4
