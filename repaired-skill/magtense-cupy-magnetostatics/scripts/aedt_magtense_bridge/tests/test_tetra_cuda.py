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
