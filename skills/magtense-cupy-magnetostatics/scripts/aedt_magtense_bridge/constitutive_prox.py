"""Constitutive proximal iteration: same M=F(KM), bounded local nonlinear steps."""
import time
import numpy as np


def solve(K, law, *, tolerance=1e-8, maxiter=6000, tau=.8, history=20, deadline=None,
          H_external=None, M_initial=None, verbose=True):
    xp, n = law.xp, law.n
    if (K.shape != (3*n, 3*n) or not 0 < tolerance < 1 or not 0 < tau <= 1 or
        maxiter < 0 or history < 0):
        raise ValueError('Invalid proximal operator shape, tolerance, step or iteration limit')
    applied = xp.zeros((n,3), dtype=K.dtype) if H_external is None else xp.asarray(H_external,dtype=K.dtype)
    if applied.shape == (3,):
        applied = xp.broadcast_to(applied,(n,3))
    if applied.shape != (n,3) or not bool(xp.all(xp.isfinite(applied))):
        raise ValueError('External H must be finite with shape (3,) or (N,3)')
    soft = law.kind == 2
    M = xp.asarray(law.rem if M_initial is None else M_initial,dtype=K.dtype).reshape(n,3).copy()
    if not bool(xp.all(xp.isfinite(M))):
        raise ValueError('Initial magnetization must be finite')
    initial_norm = max(float(xp.linalg.norm(law.rem).item()), np.sqrt(n))
    checkpoints = []
    old_x, old_f = None, None
    dxs, dfs = [], []
    t0 = time.perf_counter()
    checkpoint_interval = 1 if getattr(K,'matrix_free',False) else 25

    def mapping(m):
        H = (K @ m.ravel()).reshape(n, 3) + applied
        z = m + tau * H
        prox = (law.chi[:, None]*z + tau*law.rem) / (law.chi[:, None] + tau)
        radius = xp.linalg.norm(z, axis=1)
        for k, (h, magnetization) in enumerate(law.curves):
            knots = magnetization + tau*h
            ix = xp.clip(xp.searchsorted(knots, radius, side='right')-1, 0, len(h)-2)
            m_radius = magnetization[ix] + (radius-knots[ix]) * (
                magnetization[ix+1]-magnetization[ix]) / (knots[ix+1]-knots[ix])
            m_radius = xp.where(radius >= knots[-1], magnetization[-1], m_radius)
            direction = z / xp.maximum(radius[:, None], 1e-30)
            mask = soft & (law.curve_ids == k)
            prox = xp.where(mask[:, None], m_radius[:, None]*direction, prox)
        return prox, H

    def project(m):
        radius = xp.linalg.norm(m, axis=1)
        for k, (_, magnetization) in enumerate(law.curves):
            mask = soft & (law.curve_ids == k)
            factor = xp.minimum(1., magnetization[-1] / xp.maximum(radius, 1e-30))
            m = xp.where(mask[:, None], m*factor[:, None], m)
        return m

    for iteration in range(maxiter+1):
        if deadline and time.time() > deadline:
            raise RuntimeError('Study deadline reached in proximal solver')
        G, H = mapping(M)
        f = G-M
        f_norm = float(xp.linalg.norm(f).item())
        if not np.isfinite(f_norm):
            raise RuntimeError('Nonfinite proximal fixed-point residual')
        if iteration % checkpoint_interval == 0 or f_norm < 1e-10 * initial_norm or iteration == maxiter:
            R = M - law.eval(H)[0]
            residual = float(xp.linalg.norm(R).item()) / max(float(xp.linalg.norm(M).item()), initial_norm)
            if iteration % checkpoint_interval == 0 or residual <= tolerance:
                checkpoints.append({'iteration': iteration, 'residual': residual})
                if verbose:
                    label = 'FMM constitutive residual' if getattr(K,'matrix_free',False) else 'true constitutive residual'
                    print(f'prox {iteration}: {label}={residual:.6e}', flush=True)
            if residual <= tolerance:
                return M, {'converged': True, 'iterations': iteration, 'residual': residual,
                           'solve_seconds': time.perf_counter()-t0, 'method': 'proximal',
                           'residual_history': [p['residual'] for p in checkpoints],
                           'iteration_history': checkpoints,
                           'equations': ('M=F(H_external + approximate volume FMM(M)); unchanged local material law'
                               if getattr(K,'matrix_free',False) else
                               'Original M=F(KM), no change in physical material or K')}
        if iteration == maxiter:
            break
        if old_x is not None and history:
            dxs.append((M-old_x).ravel()); dfs.append((f-old_f).ravel())
            dxs, dfs = dxs[-history:], dfs[-history:]
        candidate = G
        if dfs:
            DF = xp.stack(dfs, axis=1)
            normal = DF.T @ DF
            ridge = max(1e-12,10*np.finfo(np.dtype(K.dtype)).eps)
            regularization = max(float(xp.trace(normal).item())/len(dfs)*ridge, 1e-16)
            normal += xp.eye(len(dfs), dtype=K.dtype)*regularization
            gamma = xp.linalg.solve(normal, DF.T @ f.ravel())
            if getattr(K, 'matrix_free', False):
                # Avoid two extra (3*N, history) temporaries at million-cell scale.
                correction = DF @ gamma
                for i, dx in enumerate(dxs):
                    correction += gamma[i]*dx
                candidate = project(G-correction.reshape(n, 3))
            else:
                DX = xp.stack(dxs, axis=1)
                candidate = project(G - ((DX+DF) @ gamma).reshape(n, 3))
            trial_G, _ = mapping(candidate)
            if float(xp.linalg.norm(trial_G-candidate).item()) > 1.5*f_norm:
                candidate = G
                dxs, dfs = [], []
        old_x, old_f = M, f
        M = candidate
    label = 'FMM' if getattr(K,'matrix_free',False) else 'true'
    raise RuntimeError(f'Proximal iteration did not converge; last {label} residual={residual}')
