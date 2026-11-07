"""CPU offline tests for MagTense/CuPy solver; no AEDT, CUDA, or MagTense required."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_solver import (MU0, MaterialLaws, apply_tangent, build_tet_centers,
                        hash_geometry, make_preconditioner, pack_tensor_block,
                        solve_nonlinear, tensor_assembly, NonConvergence)


def tiles(kinds, ids=None, mu=None, br=None, directions=None):
    kinds = np.asarray(kinds, dtype=int)
    n = len(kinds)
    return SimpleNamespace(n=n,magnet_type=kinds,
                           stfcn_index=np.ones(n,dtype=int) if ids is None else np.asarray(ids),
                           mu_r_ea=np.ones(n) if mu is None else np.asarray(mu),
                           mu_r_oa=np.ones(n) if mu is None else np.asarray(mu),
                           M_rem=np.zeros(n) if br is None else np.asarray(br)/MU0,
                           u_ea=np.tile([1.,0,0],(n,1)) if directions is None else np.asarray(directions))


def bh_from_m(h, m):
    h = np.asarray(h, dtype=float)
    m = np.asarray(m, dtype=float)
    return np.column_stack((h, MU0*(h+m)))


def test_linear_hard_pm_recoil_exact():
    t = tiles([1],mu=[1.1], br=[1.2])
    laws=MaterialLaws(t,[],np,'float64')
    K=-np.eye(3)/3
    M,info=solve_nonlinear(K, np.array([K]),laws,method='newton',tol=1e-12,verbose=False)
    np.testing.assert_allclose(M.ravel(), [1.2/MU0/(1+.1/3),0,0],rtol=1e-11)
    assert info['converged']


def test_soft_linear_exact():
    t=tiles([3],mu=[10])
    laws=MaterialLaws(t,[],np,'float64')
    K=-np.eye(3)/3
    M,info=solve_nonlinear(K,np.array([K]),laws, H_external=[100.,0,0],
                           method='newton',tol=1e-12,verbose=False)
    np.testing.assert_allclose(M[0],[225,0,0],rtol=1e-10)


def test_two_distinct_bh_curves_coupled_nonlin():
    t=tiles([2,2],ids=[1,2])
    curves=[bh_from_m([0,5000,20000,100000],[0,60000,80000,100000]),
            bh_from_m([0,6000,15000,120000],[0,30000,45000,60000])]
    laws=MaterialLaws(t,curves,np,'float64')
    # Synthetic symmetric coupling with negative self-demag, other coupling small.
    K=np.zeros((6,6),dtype=float)
    for i in range(2):
        K[3*i:3*i+3,3*i:3*i+3]=-np.eye(3)/3
    K[:3,3:]=-np.eye(3)*0.04
    K[3:,:3]=-np.eye(3)*0.04
    M,info=solve_nonlinear(K, np.stack([-np.eye(3)/3]*2), laws,
                           H_external=[65000,0,0], method='newton',
                           tol=1e-9,maxiter=30,verbose=False)
    assert info['residual'] < 1e-9
    assert M[0,0] > M[1,0]*1.3
    H= (K@M.ravel()).reshape(2,3)+np.array([65000,0,0])
    R=M-laws.eval(H)[0]
    assert np.max(np.abs(R)) < 1e-3


def test_piecewise_tangent_isotropic_matches_finite_diff():
    t=tiles([2],ids=[1])
    curves=[bh_from_m([0,2000,12000,150000],[0,20000,60000,90000])]
    laws=MaterialLaws(t,curves,np,'float64')
    H=np.array([[7000,3000,-2000.]])
    dv=np.array([[1.,-.8,.5]])
    F,a,b,u=laws.eval(H)
    approx=apply_tangent(np,a,b,u,dv)
    eps=1e-3
    finite=(laws.eval(H+eps*dv)[0]-laws.eval(H-eps*dv)[0])/(2*eps)
    np.testing.assert_allclose(approx,finite,rtol=1e-6,atol=1e-6)


def test_constant_m_extrapolation_after_last_h():
    t=tiles([2],ids=[1])
    law=MaterialLaws(t,[bh_from_m([0,10000,25000],[0,20000,30000])],np,'float64')
    F,a,b,u=law.eval(np.array([[1e6,0.,0.]]))
    np.testing.assert_allclose(F,[[30000,0,0]])
    assert b[0]==0
    assert a[0]>0


def test_low_h_slope_and_zero():
    t=tiles([2],ids=[1])
    law=MaterialLaws(t,[bh_from_m([0,10000,20000],[0,20000,30000])],np,'float64')
    F,a,b,u=law.eval(np.array([[0.,0.,0.]]))
    np.testing.assert_allclose(F,0)
    np.testing.assert_allclose(a,[2.])
    np.testing.assert_allclose(b,[2.])


def test_invalid_bh_rejected():
    t=tiles([2],ids=[1])
    with pytest.raises(ValueError):
        MaterialLaws(t,[np.array([[0,0],[10000,0.1],[20000,0.02]])],np)
    with pytest.raises(ValueError):
        MaterialLaws(t,[np.array([[0,0],[20000,0.3],[10000,0.4]])],np)


def test_pack_tensor_order_9_components_not_symmetric():
    n=3; nr=2
    t=np.arange(n*nr*9,dtype=float).reshape(n,nr,3,3)
    K=pack_tensor_block(t)
    assert K.shape==(3*nr,3*n)
    m=np.arange(n*3,dtype=float).reshape(n,3)+1
    y=(K@m.reshape(-1)).reshape(nr,3)
    expected=np.einsum('stac,sc->ta',t,m)
    np.testing.assert_array_equal(y,expected)
    assert t[0,0,0,1] != t[0,0,1,0]


def mk_mesh():
    t=tiles([1,1],mu=[1,1],br=[1,1])
    # Two shifted tetrahedra to identify target centroid in the fake tensor.
    base=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    verts=np.stack([base,base+np.array([3,0,0])],axis=0)
    t.vertices=verts.transpose(0,2,1).copy()
    t.offset=np.zeros((2,3))
    t.rot=np.zeros((2,3))
    t.tile_type=np.full(2,5)
    return t


def test_build_tensor_cache_roundtrip_and_shape(tmp_path):
    t=mk_mesh()
    tensor=np.zeros((2,2,3,3))
    for s in range(2):
        for i in range(2):
            tensor[s,i]=np.eye(3)*(-0.2 if s==i else 0.04*(s+1))
    centers=build_tet_centers(t)
    assert np.allclose(centers[:,0],[.25,3.25])
    calls=[]
    def getter(t,pts):
        calls.append(len(pts))
        idx=[int(np.argmin(abs(centers[:,0]-x))) for x in pts[:,0]]
        return tensor[:,idx]
    cache=tmp_path/'K.npy'
    K,d,info=tensor_assembly(t,np,dtype='float64',batch=1,cache=cache,tensor_fn=getter)
    assert len(calls)==2
    assert not info['loaded_from_cache']
    assert np.allclose(K,pack_tensor_block(tensor))
    assert np.allclose(d[0],tensor[0,0])
    assert np.allclose(d[1],tensor[1,1])
    calls.clear()
    K2,d2,info=tensor_assembly(t,np,dtype='float64',batch=1,cache=cache,
                                tensor_fn=lambda *a: (_ for _ in ()).throw(AssertionError()))
    assert info['loaded_from_cache']
    np.testing.assert_array_equal(K,K2)
    np.testing.assert_array_equal(d,d2)
    t.offset[0,0]=1
    assert hash_geometry(t)!=json_geometry_sha(cache)


def json_geometry_sha(cache):
    import json
    return json.loads((Path(str(cache)+'.json')).read_text())['geometry_sha256']


def test_linear_preconditioner_matches_local_jacobian():
    t=tiles([3,3],mu=[5,10]);law=MaterialLaws(t,[],np,'float64')
    H=np.array([[0.,4.,1.],[3.,0.,1.]])
    F,a,b,u=law.eval(H)
    diag=np.stack([-np.eye(3)/3, -np.eye(3)*.25])
    inv=make_preconditioner(np,diag,a,b,u)
    D=np.eye(3)[None,:,:]-law.chi[:,None,None]*diag
    np.testing.assert_allclose(inv@D, np.broadcast_to(np.eye(3),(2,3,3)),atol=1e-12)


def test_picard_converges_for_small_chi():
    t=tiles([3],mu=[2])
    laws=MaterialLaws(t,[],np,'float64')
    K=-np.eye(3)/3
    M,info=solve_nonlinear(K,np.array([K]),laws,H_external=[100,0,0],
                           method='picard',tol=1e-7,maxiter=150,damping=.5,verbose=False)
    np.testing.assert_allclose(M[0],[75,0,0],atol=1e-4)
    assert info['converged']


def test_nonconvergence_fails_loudly():
    t=tiles([3],mu=[100]); laws=MaterialLaws(t,[],np,'float64')
    K=-np.eye(3)/3
    with pytest.raises(NonConvergence):
        solve_nonlinear(K,np.array([K]),laws,H_external=[1e5,0,0],
                        method='picard',tol=1e-7,maxiter=2,damping=1,verbose=False)


def test_native_tensor_crosscheck_fake_provider():
    from gpu_solver import verify_tensor_against_magtense
    t=mk_mesh()
    g=np.random.default_rng(42)
    # Full 9-component tensor, no symmetry assumptions.
    T=g.normal(scale=.03,size=(2,2,3,3))
    K=pack_tensor_block(T)
    saved=np.asarray(t.M_rem)[:,None]*t.u_ea
    t.M=saved.copy()
    c=build_tet_centers(t)

    def fake_H(t, pts):
        targets=[int(np.argmin(np.linalg.norm(c-p,axis=1))) for p in pts]
        return np.einsum('stac,sc->ta',T[:,targets,:,:],t.M)

    err=verify_tensor_against_magtense(t,K,np,fake_H)
    assert err < 1e-12
    np.testing.assert_array_equal(t.M,saved)
    wrong_K=np.transpose(K)
    with pytest.raises(RuntimeError,match='check failed'):
        verify_tensor_against_magtense(t,wrong_K,np,fake_H)
    np.testing.assert_array_equal(t.M,saved)


def test_bridge_compare_prediction_argument(tmp_path):
    from bridge import main as bridge_main
    bridge_main(['demo','--out',str(tmp_path)])
    pts=np.loadtxt(tmp_path/'points.csv',skiprows=1,delimiter=',')
    B=np.column_stack((np.zeros(len(pts)),np.zeros(len(pts)),
                       np.linspace(.001,.002,len(pts))))
    header='x_m,y_m,z_m,Bx_T,By_T,Bz_T'
    np.savetxt(tmp_path/'gpu_B.csv',np.column_stack((pts,B)),header=header,
               comments='',delimiter=',')
    np.savetxt(tmp_path/'maxwell_ref.csv',np.column_stack((pts,B+np.array([1e-5,0,0]))),
               header=header,comments='',delimiter=',')
    bridge_main(['compare','--out',str(tmp_path),
                 '--prediction','gpu_B.csv','--prefix','gpu_',
                 '--maxwell-csv',str(tmp_path/'maxwell_ref.csv')])
    import json
    m=json.loads((tmp_path/'gpu_metrics.json').read_text())
    assert m['rms_vector_abs_error_mT'] == pytest.approx(.01,abs=1e-10)
    assert (tmp_path/'gpu_comparison.csv').exists()
