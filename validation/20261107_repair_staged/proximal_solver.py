"""Constitutive proximal iteration: same M=F(KM), bounded local nonlinear steps."""
import time
import numpy as np


def solve(K, law, *, tolerance=1e-8, maxiter=6000, tau=.8, history=20, deadline=None):
    xp, n = law.xp, law.n
    soft = law.kind == 2
    M = law.rem.copy()
    initial_norm = max(float(xp.linalg.norm(M).item()), np.sqrt(n))
    checkpoints = []
    old_x, old_f = None, None
    dxs, dfs = [], []
    t0 = time.perf_counter()

    def mapping(m):
        H = (K @ m.ravel()).reshape(n, 3)
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
        if iteration % 25 == 0 or f_norm < 1e-10 * initial_norm:
            R = M - law.eval(H)[0]
            residual = float(xp.linalg.norm(R).item()) / max(float(xp.linalg.norm(M).item()), initial_norm)
            if iteration % 25 == 0 or residual <= tolerance:
                checkpoints.append({'iteration': iteration, 'residual': residual})
                print(f'prox {iteration}: true constitutive residual={residual:.6e}', flush=True)
            if residual <= tolerance:
                return M, {'converged': True, 'iterations': iteration, 'residual': residual,
                           'seconds': time.perf_counter()-t0, 'history': checkpoints,
                           'equations': 'Original M=F(KM), no change in physical material or K'}
        if old_x is not None:
            dxs.append((M-old_x).ravel()); dfs.append((f-old_f).ravel())
            dxs, dfs = dxs[-history:], dfs[-history:]
        candidate = G
        if dfs:
            DF = xp.stack(dfs, axis=1)
            DX = xp.stack(dxs, axis=1)
            normal = DF.T @ DF
            regularization = max(float(xp.trace(normal).item())/len(dfs)*1e-12, 1e-16)
            normal += xp.eye(len(dfs), dtype=K.dtype)*regularization
            gamma = xp.linalg.solve(normal, DF.T @ f.ravel())
            candidate = project(G - ((DX+DF) @ gamma).reshape(n, 3))
            trial_G, _ = mapping(candidate)
            if float(xp.linalg.norm(trial_G-candidate).item()) > 1.5*f_norm:
                candidate = G
                dxs, dfs = [], []
        old_x, old_f = M, f
        M = candidate
    raise RuntimeError(f'Proximal iteration did not converge; last true residual={residual}')
