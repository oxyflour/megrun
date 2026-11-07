"""Resolve removable analytic-kernel singularities at verified air observations."""
from __future__ import annotations
import numpy as np


def evaluate(tiles, points, *, validate_air, get_field=None, rtol=1e-7, atol_H=1e-4):
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
    for index in bad:
        accepted = False
        for level in range(13):
            # Over-small offsets amplify cancellation in removable kernel singularities.
            # Start farther away, then require both step and direction convergence.
            step = scale*1e-3 / 2**level
            offsets = np.array([sign*step/2**j*d for d in directions for j in range(3) for sign in [-1,1]])
            observations = points[index]+offsets
            try:
                validate_air(observations)
            except ValueError:
                continue
            fields = np.asarray(get_field(tiles,observations),dtype=float)
            if not np.isfinite(fields).all():
                continue
            averages = fields.reshape(2,3,2,3).mean(axis=2)
            limits1 = (4*averages[:,1]-averages[:,0])/3
            limits2 = (4*averages[:,2]-averages[:,1])/3
            result = limits2.mean(axis=0)
            difference = max(float(np.linalg.norm(limits2-limits1,axis=1).max()),
                             float(np.linalg.norm(limits2[0]-limits2[1])))
            if difference <= atol_H+rtol*float(np.linalg.norm(result)):
                H[index] = result
                info['limits'].append({'index':int(index),'step_m':step,
                                       'limit_difference_Apm':difference})
                accepted = True
                break
        if not accepted:
            raise FloatingPointError(f'Native field limit did not converge at air point {index}')
    if not np.isfinite(H).all():
        raise FloatingPointError('Nonfinite field remains after air limit evaluation')
    return H, info
