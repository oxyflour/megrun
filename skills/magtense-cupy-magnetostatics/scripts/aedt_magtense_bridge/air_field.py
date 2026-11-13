"""Resolve removable analytic-kernel singularities at verified air observations."""
from __future__ import annotations
import numpy as np


def evaluate(tiles, points, *, validate_air, get_field=None, rtol=1e-7, atol_H=1e-4, limit_batch=1):
    if limit_batch < 1:
        raise ValueError('Air limit batch must be positive')
    if get_field is None:
        from magtense.magstatics import get_H_field
        get_field = get_H_field
    points = np.asarray(points,dtype=float)
    validate_air(points)
    H = np.asarray(get_field(tiles,points),dtype=float)
    bad = np.flatnonzero(~np.isfinite(H).all(axis=1))
    info = {'regularized_indices':bad.tolist(), 'coordinates_unchanged':True,
            'method':'two-direction symmetric Richardson limit in verified air',
            'limits':[]}
    if not len(bad):
        return H, info
    vertices = np.asarray(tiles.vertices).transpose(0,2,1)
    scale = max(float(np.linalg.norm(np.ptp(vertices.reshape(-1,3),axis=0))),1e-6)
    directions = np.array([[1.,np.sqrt(2),np.sqrt(3)],[np.sqrt(5),-1.,np.sqrt(7)]])
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    for first in range(0,len(bad),limit_batch):
        remaining = bad[first:first+limit_batch]
        for level in range(13):
            # Over-small offsets amplify cancellation in removable kernel singularities.
            # Start farther away, then require both step and direction convergence.
            step = scale*1e-3 / 2**level
            offsets = np.array([sign*step/2**j*d for d in directions for j in range(3) for sign in [-1,1]])
            observations = points[remaining,None,:]+offsets[None,:,:]
            try:
                validate_air(observations.reshape(-1,3))
                eligible = np.arange(len(remaining))
            except ValueError:
                eligible = []
                for j,candidate in enumerate(observations):
                    try:
                        validate_air(candidate)
                        eligible.append(j)
                    except ValueError:
                        pass
                eligible = np.asarray(eligible,dtype=int)
            if not len(eligible):
                continue
            fields = np.asarray(get_field(tiles,observations[eligible].reshape(-1,3)),dtype=float).reshape(-1,12,3)
            finite = np.isfinite(fields).all(axis=(1,2))
            eligible,fields = eligible[finite],fields[finite]
            if not len(eligible):
                continue
            averages = fields.reshape(-1,2,3,2,3).mean(axis=3)
            limits1 = (4*averages[:,:,1]-averages[:,:,0])/3
            limits2 = (4*averages[:,:,2]-averages[:,:,1])/3
            result = limits2.mean(axis=1)
            difference = np.maximum(np.linalg.norm(limits2-limits1,axis=2).max(axis=1),
                                    np.linalg.norm(limits2[:,0]-limits2[:,1],axis=1))
            accepted = difference <= atol_H+rtol*np.linalg.norm(result,axis=1)
            resolved = remaining[eligible[accepted]]
            H[resolved] = result[accepted]
            for index,error in zip(resolved,difference[accepted]):
                info['limits'].append({'index':int(index),'step_m':step,'limit_difference_Apm':float(error)})
            remaining = remaining[~np.isin(remaining,resolved)]
            if not len(remaining):
                break
        if len(remaining):
            raise FloatingPointError(f'Native field limit did not converge at air point {remaining[0]}')
    info['limits'].sort(key=lambda item:item['index'])
    if not np.isfinite(H).all():
        raise FloatingPointError('Nonfinite field remains after air limit evaluation')
    return H, info
