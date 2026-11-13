"""CUDA/native agreement, independent invariants, and cache isolation."""
from pathlib import Path
from itertools import permutations
import json
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tetra_volume import assemble as native_assemble, verify
from tetra_cuda import assemble


@pytest.fixture(scope='module')
def cp():
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CUDA device unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    return cp


def tiles(vertices):
    ms = pytest.importorskip('magtense.magstatics')
    return ms.Tiles(n=len(vertices),vertices=np.asfortranarray(vertices.transpose(0,2,1)),
                    tile_type=5,M_rem=0,easy_axis=[1,0,0],mu_r_ea=1,mu_r_oa=1)


def mesh():
    unit = np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
    return np.array([unit*.001, unit*.001+[.0008,.0007,-.0001],
                     unit*[.0001,.003,.001]+[.003,0,0]])


@pytest.mark.parametrize('levels',[0,1,2])
@pytest.mark.parametrize('dtype',['float64','float32'])
def test_native_tensor_agreement_and_trace(cp,levels,dtype):
    t = tiles(mesh())
    reference,_,_ = native_assemble(t,np,dtype=dtype,near_levels=levels)
    K,diag,info = assemble(t,cp,dtype=dtype,near_levels=levels,receiver_batch=1,near_batch=2)
    np.testing.assert_allclose(cp.asnumpy(K),reference,rtol=2e-6 if dtype=='float32' else 1e-9,
                               atol=2e-8 if dtype=='float32' else 1e-12)
    np.testing.assert_allclose(np.trace(cp.asnumpy(diag),axis1=1,axis2=2),-1,
                               atol=2e-7 if dtype=='float32' else 1e-12)
    assert verify(t,K,cp,near_levels=levels) < (1e-5 if dtype=='float32' else 1e-10)
    assert info['K_near_seconds']>0 and info['K_far_seconds']>0


def test_rotation_translation_scale_and_vertex_order(cp):
    v = mesh()
    q,_ = np.linalg.qr(np.random.default_rng(97).normal(size=(3,3)))
    K,_,_ = assemble(tiles(v),cp)
    moved = (v@q.T)*7.+[.031,-.012,.003]
    rotated,_,_ = assemble(tiles(moved),cp)
    transform = np.kron(np.eye(len(v)),q)
    expected = transform@cp.asnumpy(K)@transform.T
    np.testing.assert_allclose(cp.asnumpy(rotated),expected,rtol=1e-9,atol=2e-12)
    # Red8 chooses an interior diagonal from the vertex order. A permutation
    # therefore changes the discrete quadrature, but CUDA must still match CPU.
    permuted = tiles(moved[:,[2,0,3,1]])
    reference,_,_ = native_assemble(permuted,np)
    result,_,_ = assemble(permuted,cp)
    np.testing.assert_allclose(cp.asnumpy(result),reference,rtol=1e-9,atol=2e-12)


def test_cache_cannot_relabel_native_or_old_kernel(cp,tmp_path):
    t = tiles(mesh()[:1]);cache=tmp_path/'operator.npy'
    K,_,info = assemble(t,cp,cache=cache)
    assert not info['loaded_from_cache']
    K2,_,info = assemble(t,cp,cache=cache)
    assert info['loaded_from_cache']
    np.testing.assert_array_equal(cp.asnumpy(K),cp.asnumpy(K2))
    meta_path=Path(str(cache)+'.json')
    for key,value in [('schema','magtense-tetra-volume-1'),('kernel_sha256','outdated')]:
        meta=json.loads(meta_path.read_text());meta[key]=value
        meta_path.write_text(json.dumps(meta))
        _,_,info=assemble(t,cp,cache=cache)
        assert not info['loaded_from_cache']


def test_reject_numpy_or_expired_deadline(cp):
    t = tiles(mesh()[:1])
    with pytest.raises(ValueError,match='backend cupy'):
        assemble(t,np)
    with pytest.raises(TimeoutError):
        assemble(t,cp,deadline=1.)


def test_independent_cube_average_demagnetization(cp):
    side=.001
    v=np.array([[[0,0,0],np.eye(3)[p[0]]*side,
                 (np.eye(3)[p[0]]+np.eye(3)[p[1]])*side,[side]*3]
                for p in permutations(range(3))])
    K,_,_=assemble(tiles(v),cp,near_levels=3)
    averaged=cp.asnumpy(K).reshape(6,3,6,3).sum(axis=2).mean(axis=0)
    np.testing.assert_allclose(averaged,-np.eye(3)/3,rtol=0,atol=2e-4)


def test_cli_cuda_build_and_cache_keep_native_verification(cp,tmp_path):
    from bridge import command_demo
    from gpu_solver import main
    command_demo(SimpleNamespace(out=tmp_path))
    args=['--out',str(tmp_path),'--backend','cupy','--dtype','float64',
          '--operator','volume','--k-builder','cuda','--method','proximal',
          '--tol','1e-8','--cache-k']
    for expected_hit in (False,True):
        main(args)
        info=json.loads((tmp_path/'gpu_solve_info.json').read_text())
        assert info['K_builder']=='cuda' and info['loaded_from_cache']==expected_hit
        assert info['converged'] and info['K_magtense_field_check_relative_error']<1e-9
        assert info['air_field_seconds']>0 and info['pipeline_seconds_before_info_write']>0


@pytest.mark.parametrize('source_levels',[0,1])
@pytest.mark.parametrize('cache_bytes',[0,1024**2])
@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('farfield',['cpu','cuda'])
def test_fmm_cuda_near_cache_and_stream_match_native(cp,source_levels,cache_bytes,dtype,farfield):
    pytest.importorskip('fmm3dpy')
    from tetra_fmm import assemble as fmm_assemble
    v = mesh()
    # Include a genuinely far pair, not only analytical near corrections.
    v[2] += [.03,.01,.02]
    t = tiles(v)
    cpu,diag,_ = fmm_assemble(t,np,dtype=dtype,source_levels=source_levels,
                             near_levels=1,cache_bytes=cache_bytes,batch=2)
    gpu,gdiag,info = fmm_assemble(t,cp,dtype=dtype,source_levels=source_levels,
                                 near_levels=1,cache_bytes=cache_bytes,batch=2,farfield=farfield,gpu_order=4)
    m = np.random.default_rng(12).normal(size=9).astype(dtype)*20000
    tolerance = 3e-6 if dtype == 'float32' else 1e-9
    np.testing.assert_allclose(cp.asnumpy(gpu@cp.asarray(m)),cpu@m,rtol=tolerance,
                               atol=2e-3 if dtype == 'float32' else 1e-7)
    np.testing.assert_allclose(cp.asnumpy(gdiag),diag,rtol=tolerance,atol=1e-12)
    assert info['fmm_near_storage'] == ('cached' if cache_bytes else 'streamed')
    assert info['fmm_tree_reused'] == (farfield=='cuda')


@pytest.mark.parametrize('farfield',['cpu','cuda'])
def test_cli_fmm_keeps_native_checks_and_writes_approximation_scope(cp,tmp_path,monkeypatch,farfield):
    if farfield=='cpu':
        pytest.importorskip('fmm3dpy')
    else:
        import sys
        monkeypatch.setitem(sys.modules,'fmm3dpy',None)
    from bridge import command_demo
    from gpu_solver import main
    command_demo(SimpleNamespace(out=tmp_path))
    main(['--out',str(tmp_path),'--backend','cupy','--dtype','float64',
          '--operator','volume','--k-builder','fmm','--method','proximal',
          '--tol','1e-8','--anderson-history','4','--fmm-farfield',farfield,'--fmm-gpu-order','4'])
    info = json.loads((tmp_path/'gpu_solve_info.json').read_text())
    assert info['converged'] and info['K_builder'] == 'fmm'
    assert not info['dense_K_allocated'] and not info['full_native_constitutive_residual_verified']
    assert info['K_magtense_field_check_relative_error'] < 1e-4
    assert info['fmm_final_native_sample_relative_error'] < 1e-4
    assert info['fmm_air_native_sample_relative_error'] < 1e-4
    assert info['fmm_air_field_calls'] >= 1
    assert info['fmm_matvec_calls'] > info['iterations']
    assert len(info['iteration_history'])==info['iterations']+1
    if farfield=='cuda':
        assert info['fmm_backend']=='CUDA interpolation FMM' and info['fmm_tree_reused']
        assert info['fmm_air_point_backend']=='cuda_direct_batched'
        assert info['fmm_air_correction_backend']=='cuda_analytic'
    assert not list(tmp_path.glob('K*.npy'))


def test_fmm_cuda_streams_neighbour_indices_with_tiny_budget(cp):
    pytest.importorskip('fmm3dpy')
    from tetra_fmm import assemble as fmm_assemble
    t = tiles(mesh())
    reference,_,_ = native_assemble(t,np,near_levels=1)
    op,_,info = fmm_assemble(t,cp,near_levels=1,correction_factor=100.,index_bytes=40,batch=1)
    m = np.random.default_rng(65).normal(size=9)
    np.testing.assert_allclose(cp.asnumpy(op@cp.asarray(m)),reference@m,rtol=1e-8,atol=1e-10)
    assert op.indices is None and info['fmm_index_storage'] == 'streamed'


def test_gpu_radius_queries_match_cpu_for_graded_and_duplicate_centres(cp):
    from tetra_fmm import _CUDANeighbours,_neighbours,_source_trees
    rng = np.random.default_rng(981)
    centres = rng.normal(size=(257,3))
    centres[1] = centres[0]
    radii = np.exp(rng.uniform(-5,0,len(centres)))
    # Tiles geometry commonly exposes Fortran-strided arrays; kernels require
    # contiguous FP64 coordinates rather than interpreting those strides away.
    gpu = _CUDANeighbours(np.asfortranarray(centres),radii,query_batch=41,max_pairs=2000)
    cpu = _source_trees(centres,radii)
    expected = [row for _,rows in _neighbours(centres,radii,1.5,17,groups=cpu) for row in rows]
    actual = [row for _,rows in _neighbours(centres,radii,1.5,17,groups=gpu) for row in rows]
    np.testing.assert_array_equal(gpu.counts(1.5),[len(row) for row in expected])
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a,b)
    targets = rng.normal(size=(23,3))
    for target_radii in (np.zeros(23),np.exp(rng.uniform(-5,0,23))):
        expected = list(_neighbours(centres,radii,3.,7,groups=cpu,targets=targets,target_radii=target_radii))
        actual = list(_neighbours(centres,radii,3.,7,groups=gpu,targets=targets,target_radii=target_radii))
        for (_,aa),(_,bb) in zip(actual,expected):
            for a,b in zip(aa,bb):
                np.testing.assert_array_equal(a,b)


def test_gpu_radius_boundary_and_pair_budget_never_drop_neighbours(cp):
    from tetra_fmm import _CUDANeighbours
    centres = np.zeros((5,3))
    centres[:,0] = [0,3,np.nextafter(3.,0),np.nextafter(3.,np.inf),-3]
    search = _CUDANeighbours(centres,np.ones(5),max_pairs=5)
    result = list(search.neighbours(1.5,1,targets=centres[:1],target_radii=np.ones(1)))[0][1][0]
    np.testing.assert_array_equal(result,[0,1,2,4])
    search.max_pairs = 2
    with pytest.raises(MemoryError,match='pair budget'):
        list(search.neighbours(1.5,1,targets=centres[:1],target_radii=np.ones(1)))


def test_cuda_point_fmm_multilevel_matches_independent_dipole_sum_and_reuses_tree(cp):
    from tetra_fmm import _CUDAPointFMM
    rng = np.random.default_rng(73)
    sources = rng.uniform(.02,.98,(577,3))
    targets = rng.uniform(.01,.99,(37,3))
    fmm = _CUDAPointFMM(sources,targets,eps=2e-5,order=8,leaf_size=4)
    assert fmm.depth>=3 and fmm.interactions
    original_tree = fmm.source_tree
    for moments in (rng.normal(size=sources.shape),rng.normal(size=sources.shape)*3):
        delta = targets[:,None,:]-sources[None,:,:]
        radius = np.linalg.norm(delta,axis=2)
        expected = np.sum((3*delta*np.sum(delta*moments[None,:,:],axis=2)[:,:,None]/radius[:,:,None]**5
                           -moments[None,:,:]/radius[:,:,None]**3)/(4*np.pi),axis=1)
        actual = cp.asnumpy(fmm(moments))
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-5
        subset = fmm.subset(targets[[0,17,36]])
        assert subset.source_tree is fmm.source_tree
        np.testing.assert_allclose(cp.asnumpy(subset(moments)),actual[[0,17,36]],rtol=1e-12,atol=1e-9)
    assert fmm.source_tree is original_tree and fmm.info['fmm_gpu_evaluations']==2
    fmm.eps = 1e-30
    with pytest.raises(RuntimeError,match='sampled point-field error'):
        fmm(moments)


def test_cuda_point_fmm_coincident_sources_and_budget_are_explicit(cp):
    from tetra_fmm import _CUDAPointFMM
    sources = np.array([[.2,.3,.4],[.2,.3,.4],[.8,.7,.6]])
    moments = np.array([[1.,2.,3.],[-2.,1.,4.],[3.,2.,-1.]])
    fmm = _CUDAPointFMM(sources,eps=1e-12,order=3)
    actual = fmm(moments)
    np.testing.assert_allclose(cp.asnumpy(actual),cp.asnumpy(fmm.direct(cp.asarray(moments),cp.asarray(sources))),
                               rtol=1e-12,atol=1e-12)
    with pytest.raises(MemoryError,match='budget'):
        _CUDAPointFMM(sources,budget_bytes=1)
    with pytest.raises(ValueError,match='normalized'):
        _CUDAPointFMM(sources+2)
