from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from air_field import evaluate


def tile():
    return SimpleNamespace(vertices=np.array([[[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]]))


def test_finite_air_field_is_unchanged():
    p = np.array([[2.,0,0],[3.,0,0]])
    expected = np.array([[1.,2,3],[4.,5,6]])
    h,info = evaluate(tile(),p,validate_air=lambda p:None,get_field=lambda t,p:expected.copy())
    np.testing.assert_array_equal(h,expected)
    assert info['regularized_indices'] == []


def test_removable_polynomial_limit_preserves_nominal_coordinates():
    points = np.array([[0.,0,0],[2.,0,0]])
    original = points.copy()
    checks = []
    def field(t,p):
        h = np.c_[1+p[:,0]**2+p[:,1]**2,2+p[:,2]**2,3+p[:,0]*p[:,1]]
        h[np.all(p==0,axis=1)] = np.nan
        return h
    h,info = evaluate(tile(),points,validate_air=lambda p:checks.append(p.copy()),get_field=field)
    np.testing.assert_allclose(h[0],[1,2,3],atol=1e-12)
    np.testing.assert_array_equal(points,original)
    assert info['regularized_indices'] == [0]
    assert len(checks) >= 2


def test_nonconvergent_limit_is_rejected():
    def field(t,p):
        with np.errstate(divide='ignore'):
            value = 1/np.linalg.norm(p,axis=1)
        return np.repeat(value[:,None],3,axis=1)
    with pytest.raises(FloatingPointError,match='did not converge'):
        evaluate(tile(),np.zeros((1,3)),validate_air=lambda p:None,get_field=field)


def test_batched_limits_preserve_each_point_and_exclude_invalid_air_offsets():
    points = np.c_[np.arange(2.,11.,2.),np.zeros(5),np.zeros(5)]
    calls = []
    def field(t,p):
        calls.append(len(p))
        result = np.c_[1+p[:,0]**2,2+p[:,1]**2,3+p[:,2]**2]
        result[p[:,2] == 0] = np.nan
        return result
    def validate(p):
        if np.any((p[:,0] > 9) & (abs(p[:,1]) > 2e-4)):
            raise ValueError('Offset left air domain')
    result,info = evaluate(tile(),points,validate_air=validate,get_field=field,limit_batch=5)
    np.testing.assert_allclose(result,np.c_[1+points[:,0]**2,np.full(5,2.),np.full(5,3.)],atol=1e-10)
    assert info['regularized_indices'] == list(range(5))
    assert max(calls) == 48  # Four valid offset sets share one field call.
    assert info['limits'][-1]['step_m'] < info['limits'][0]['step_m']
