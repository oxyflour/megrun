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


def fmm_tiles(vertices):
    pytest.importorskip('fmm3dpy')
    ms = pytest.importorskip('magtense.magstatics')
    return ms.Tiles(n=len(vertices),vertices=np.asfortranarray(vertices.transpose(0,2,1)),
                    tile_type=5,M_rem=0,easy_axis=[1,0,0],mu_r_ea=1,mu_r_oa=1)


@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_adaptive_near_compression_known_rank_and_masked_self_blocks(backend):
    """Independent rank-two oracle, plus a cutoff/self matrix that must stay exact."""
    from tetra_fmm import AdaptiveNearCompression
    xp = np if backend=='numpy' else pytest.importorskip('cupy')
    rng = np.random.default_rng(419)
    nr, ns = 16, 24
    left, right = rng.normal(size=(3*nr, 2)), rng.normal(size=(2, 3*ns))
    expected = left @ right
    v = np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]], float)
    receivers = v[None]+np.arange(nr)[:,None,None]*[0,.02,0]
    sources = v[None]+[10,0,0]+np.arange(ns)[:,None,None]*[0,.02,0]
    indices = np.tile(np.arange(ns, dtype=np.int32), nr)
    pointer = np.arange(nr+1, dtype=np.int64)*ns
    blocks = expected.reshape(nr,3,ns,3).transpose(0,2,1,3).reshape(-1,3,3)
    op = AdaptiveNearCompression(xp.asarray(blocks), indices, pointer, receivers, sources, xp,
                                  rtol=1e-10, leaf_size=4)
    tohost = np.asarray if xp is np else xp.asnumpy
    assert op.info['ranks']==[2]
    assert op.info['compression_ratio']>4
    np.testing.assert_allclose(tohost(op.toarray()),expected,rtol=1e-11,atol=1e-11)
    vector = rng.normal(size=3*ns)
    np.testing.assert_allclose(tohost(op @ xp.asarray(vector)),expected@vector,rtol=1e-11,atol=1e-11)
    # Force several source clusters to contribute to the same receiver rows;
    # this checks the packed CUDA right/left reductions and atomic accumulation.
    op = AdaptiveNearCompression(xp.asarray(blocks),indices,pointer,receivers,sources,xp,
                                  rtol=1e-10,leaf_size=4,max_svd_entries=400,decomposition='cpu')
    assert len(op.low_rank)>2
    np.testing.assert_allclose(tohost(op @ xp.asarray(vector)),expected@vector,rtol=1e-11,atol=1e-11)
    # A full-rank separated block must split, exercise batched small SVDs,
    # then fall back to the original coefficients without storage inflation.
    expected = rng.normal(size=(3*nr,3*ns))
    blocks = expected.reshape(nr,3,ns,3).transpose(0,2,1,3).reshape(-1,3,3)
    op = AdaptiveNearCompression(xp.asarray(blocks),indices,pointer,receivers,sources,xp,
                                  rtol=1e-10,leaf_size=4)
    assert op.info['svd_attempts']>4
    assert op.info['retained_raw_pairs']==nr*ns
    assert op.info['compression_ratio']==1.
    np.testing.assert_allclose(tohost(op.toarray()),expected,rtol=0,atol=0)
    # Sparse near mask: only self and one neighbour, including a sharp cutoff.
    nr = ns = 16
    rows = [np.unique([i, min(i+1,ns-1)]).astype(np.int32) for i in range(nr)]
    indices = np.concatenate(rows)
    pointer = np.r_[0,np.cumsum([len(row) for row in rows])]
    blocks = rng.normal(size=(len(indices),3,3))
    expected = np.zeros((3*nr,3*ns))
    for row in range(nr):
        for p in range(pointer[row],pointer[row+1]):
            expected[3*row:3*row+3,3*indices[p]:3*indices[p]+3] = blocks[p]
    op = AdaptiveNearCompression(xp.asarray(blocks),indices,pointer,receivers,receivers,xp,
                                  rtol=1e-10,leaf_size=4)
    assert op.info['retained_raw_pairs']==len(indices)
    np.testing.assert_allclose(tohost(op.toarray()),expected,rtol=0,atol=0)
    vector = rng.normal(size=3*ns)
    np.testing.assert_allclose(tohost(op @ xp.asarray(vector)),expected@vector,rtol=1e-13,atol=1e-13)
    # An absolute budget allocated from total patch energy can discard tiny
    # off-diagonal coefficients while preserving every self block exactly.
    nr = ns = 8
    expected = np.eye(3*nr)
    expected[:12,12:] = rng.normal(size=(12,12))*1e-9
    rows = [np.unique([i]+(list(range(4,8)) if i<4 else [])).astype(np.int32) for i in range(nr)]
    indices = np.concatenate(rows)
    pointer = np.r_[0,np.cumsum([len(row) for row in rows])]
    blocks = np.stack([expected[3*i:3*i+3,3*j:3*j+3] for i,row in enumerate(rows) for j in row])
    op = AdaptiveNearCompression(xp.asarray(blocks),indices,pointer,receivers[:nr],receivers[:nr],xp,
                                  rtol=1e-5,leaf_size=2,error_budget='global',decomposition='cpu')
    actual = tohost(op.toarray())
    assert op.info['dropped_pairs']>0
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<1e-5
    np.testing.assert_allclose(np.diag(actual),np.ones(3*nr),rtol=0,atol=0)


def fmm_mesh():
    unit = np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])*.001
    return np.array([unit,unit+[.003,.001,.001],unit+[.012,.003,-.002]])


@pytest.mark.parametrize('cache_bytes',[0,1024**2])
@pytest.mark.parametrize('source_levels',[0,1])
def test_fmm_analytical_correction_matches_original_volume_operator(cache_bytes,source_levels):
    """All corrected pairs must reproduce analytical K, including finite self H."""
    from tetra_fmm import assemble as fmm_assemble
    t = fmm_tiles(fmm_mesh())
    reference,diagonal,_ = assemble(t,np,near_levels=1)
    before = t.M.copy()
    op,diag,info = fmm_assemble(t,np,near_levels=1,source_levels=source_levels,
                               correction_factor=100.,cache_bytes=cache_bytes,batch=2)
    rng = np.random.default_rng(241)
    for _ in range(2):
        m = rng.normal(size=3*t.n)*20000
        np.testing.assert_allclose(op@m,reference@m,rtol=2e-9,atol=1e-6)
    np.testing.assert_allclose(diag,diagonal,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(t.M,before)
    assert info['dense_K_allocated'] is False and info['K_GB'] == 0
    assert info['fmm_near_storage'] == ('cached' if cache_bytes else 'streamed')
    assert info['fmm_matvec_calls'] == 2


def test_fmm_far_source_quadrature_error_reduces_with_refinement():
    """Native analytical source is independent of FMM point-dipole quadrature."""
    from tetra_fmm import assemble as fmm_assemble
    t = fmm_tiles(fmm_mesh()[:2])
    reference,_,_ = assemble(t,np,near_levels=1)
    m = np.array([1.,.3,-.2,0,0,0])
    target = (reference@m)[3:]
    errors = []
    for level in (0,1,2):
        op,_,info = fmm_assemble(t,np,near_levels=1,correction_factor=1.5,source_levels=level)
        assert info['fmm_near_pairs'] == 2  # Cross pair really exercises the far field.
        errors.append(np.linalg.norm((op@m)[3:]-target)/np.linalg.norm(target))
    assert errors[0] < 1e-3 and errors[1] < errors[0]/4 and errors[2] < errors[1]/4
    op.verify_rtol = 1e-10
    with pytest.raises(RuntimeError,match='source quadrature error'):
        op.verify(np.array([[1.,.3,-.2],[0,0,0]]))


def test_fmm_streams_neighbour_indices_when_they_exceed_budget():
    from tetra_fmm import assemble as fmm_assemble
    t = fmm_tiles(fmm_mesh())
    reference,_,_ = assemble(t,np,near_levels=1)
    op,_,info = fmm_assemble(t,np,near_levels=1,correction_factor=100.,
                            index_bytes=40,batch=1)
    assert op.indices is None and op.blocks is None
    assert info['fmm_index_storage'] == info['fmm_near_storage'] == 'streamed'
    m = np.random.default_rng(79).normal(size=9)
    np.testing.assert_allclose(op@m,reference@m,rtol=1e-9,atol=1e-10)
    assert op.verify() < 1e-8


def test_fmm_rotation_translation_scale_and_batched_high_quadrature(monkeypatch):
    from tetra_fmm import assemble as fmm_assemble
    import tetra_fmm
    t = fmm_tiles(fmm_mesh())
    original = tetra_fmm.quadrature
    high_sizes = []
    def bounded(vertices,levels=0):
        if levels == 2:
            high_sizes.append(len(vertices))
        return original(vertices,levels)
    monkeypatch.setattr(tetra_fmm,'quadrature',bounded)
    op,_,_ = fmm_assemble(t,np,batch=1,cache_bytes=0)
    m = np.random.default_rng(9).normal(size=(3,3))
    result = (op@m.ravel()).reshape(3,3)
    q,_ = np.linalg.qr(np.random.default_rng(73).normal(size=(3,3)))
    transformed = fmm_tiles(fmm_mesh()@q.T*7+[.012,-.007,.003])
    other,_,_ = fmm_assemble(transformed,np,batch=1,cache_bytes=0)
    np.testing.assert_allclose((other@(m@q.T).ravel()).reshape(3,3),result@q.T,
                               rtol=1e-8,atol=1e-10)
    assert high_sizes and max(high_sizes) == 1


def test_fmm_multimaterial_nonlinear_solution_matches_dense():
    from tetra_fmm import assemble as fmm_assemble
    t = fmm_tiles(fmm_mesh())
    t.magnet_type = np.array([1,2,2])
    t.M_rem = np.array([1e6,0,0])
    t.stfcn_index = np.array([1,1,2])
    h = np.array([0.,100.,10000.,200000.])
    curves = [np.c_[h,MU0*(h+np.array(m))] for m in
              ([0,1000,60000,80000],[0,3000,100000,150000])]
    laws = MaterialLaws(t,curves,np,'float64')
    dense,diagonal,_ = assemble(t,np,near_levels=1)
    op,diag,_ = fmm_assemble(t,np,near_levels=1,correction_factor=100.)
    for method in ('newton','proximal'):
        expected,_ = solve_nonlinear(dense,diagonal,laws,method=method,H_external=[500,200,0],
                                     tol=1e-10,verbose=False)
        actual,info = solve_nonlinear(op,diag,laws,method=method,H_external=[500,200,0],
                                     tol=1e-10,anderson_history=4,verbose=False)
        np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-5)
        assert info['converged']
        if method == 'proximal':
            assert 'approximate volume FMM' in info['equations']


def test_fmm_rejects_bad_geometry_memory_budget_and_failed_library(monkeypatch):
    from tetra_fmm import assemble as fmm_assemble
    t = fmm_tiles(fmm_mesh())
    for arguments in ({'eps':0},{'correction_factor':1.},{'index_bytes':1},{'batch':0}):
        with pytest.raises((ValueError,MemoryError)):
            fmm_assemble(t,np,**arguments)
    degenerate = fmm_mesh();degenerate[0,3] = degenerate[0,0]
    with pytest.raises(ValueError,match='nondegenerate'):
        fmm_assemble(fmm_tiles(degenerate),np)
    no_memory = SimpleNamespace(cuda=SimpleNamespace(Device=lambda:SimpleNamespace(mem_info=(1,1))))
    with pytest.raises(MemoryError,match='FMM CUDA geometry'):
        fmm_assemble(t,no_memory,source_levels=2)
    op,_,_ = fmm_assemble(t,np,near_levels=0)
    monkeypatch.setattr(op,'_lfmm',lambda **kwargs:SimpleNamespace(ier=4,grad=None))
    with pytest.raises(RuntimeError,match='ier=4'):
        op@np.ones(9)
    monkeypatch.setattr(op,'_lfmm',lambda **kwargs:SimpleNamespace(ier=0,grad=np.full((3,12),np.nan)))
    with pytest.raises(FloatingPointError):
        op@np.ones(9)


def test_fmm_thousands_of_sources_match_sampled_native_field():
    """Exercise the compiled FMM on 4096 sources; check original volume field."""
    from tetra_fmm import assemble as fmm_assemble
    unit = fmm_mesh()[0]
    offsets = np.indices((16,8,8)).reshape(3,-1).T*.003
    t = fmm_tiles(unit[None]+offsets[:,None,:])
    op,_,info = fmm_assemble(t,np,near_levels=0,correction_factor=1.5,batch=16,
                            verify_rtol=1e-4)
    assert info['fmm_source_points'] == 4096
    assert op.verify() < 1e-4


def test_fmm_air_probes_match_independent_native_field():
    from tetra_fmm import assemble as fmm_assemble
    from magtense.magstatics import get_H_field
    t = fmm_tiles(fmm_mesh())
    m = np.random.default_rng(357).normal(size=(t.n,3))*20000
    t.M = m.copy()
    points = np.array([[.0005,.0015,.0005],[.006,-.002,.003],[.1,.02,.03]])
    expected = get_H_field(t,points)
    op,_,info = fmm_assemble(t,np,correction_factor=100.)
    actual = op.field_at(m,points)
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
    assert op.verify_air(m,points,actual) < 1e-6
    np.testing.assert_array_equal(t.M,m)
    assert info['fmm_air_field_calls'] == 1


def test_fmm_coplanar_air_point_requires_limit_and_matches_analytic_prism():
    """A finite near-field pole must not hide behind the finite FMM far field."""
    from tetra_fmm import assemble as fmm_assemble
    from air_field import evaluate
    ms = pytest.importorskip('magtense.magstatics')
    side = .001
    vertices = np.array([[[0,0,0],np.eye(3)[p[0]]*side,
        (np.eye(3)[p[0]]+np.eye(3)[p[1]])*side,[side]*3] for p in permutations(range(3))])
    t = fmm_tiles(vertices)
    m = np.tile([100000.,0,0],(6,1))
    point = np.array([[.002,0,.0015]])
    op,_,_ = fmm_assemble(t,np,near_levels=0)
    assert not np.isfinite(op.field_at(m,point)).all()
    def validate(points):
        assert np.all(points[:,0] > side)
    actual,info = evaluate(t,point,validate_air=validate,get_field=lambda source,p:op.field_at(m,p))
    prism = ms.Tiles(n=1,tile_type=2,size=[side]*3,offset=[side/2]*3,
                     M_rem=100000.,easy_axis=[1,0,0],mu_r_ea=1,mu_r_oa=1)
    expected = ms.get_H_field(prism,point)
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)
    assert info['regularized_indices'] == [0]
    assert op.verify_air(m,point,actual,validate_air=validate,force_limit_points=point) < 1e-6
