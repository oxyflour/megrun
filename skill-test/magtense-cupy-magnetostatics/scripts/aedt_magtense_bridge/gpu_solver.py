#!/usr/bin/env python3
"""MagTense analytical TET tensor + CuPy/NumPy nonlinear GPU magnetostatics.

Reuses the export format of bridge.py (model.json, mesh/*.npz, points.csv).
Only isotropic, single-valued, non-hysteretic B(H); no current-driven coils.
Units: length m, H/M A/m, B T, relative permeability dimensionless.

Examples:
    python gpu_solver.py --out out --backend cupy --method newton --cache-k
    python bridge.py compare --out out --prediction gpu_B.csv --prefix gpu_
    python gpu_solver.py --self-test --backend numpy
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

MU0 = 4e-7 * np.pi


class NonConvergence(RuntimeError):
    pass


def get_backend(name: str):
    if name == 'numpy':
        return np
    if name == 'cupy':
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() <= 0:
                raise RuntimeError('No CUDA device detected')
            return cp
        except (ImportError, RuntimeError) as e:
            raise RuntimeError('CuPy backend unavailable; install cupy-cuda13x or '
                               'cupy-cuda12x matching your CUDA installation') from e
    raise ValueError(f'Unknown backend: {name}')


def as_float(x) -> float:
    """Only scalar diagnostics cross the CPU/GPU boundary."""
    return float(x.item())


def build_tet_centers(tiles) -> np.ndarray:
    if not np.all(np.asarray(tiles.tile_type) == 5):
        raise ValueError('This first GPU implementation requires all tiles be TET4')
    vertices = np.asarray(tiles.vertices, dtype=np.float64)
    if vertices.shape != (tiles.n, 3, 4):
        raise ValueError(f'Expected MagTense vertices (N,3,4), got {vertices.shape}')
    return vertices.mean(axis=2)


def hash_geometry(tiles) -> str:
    """Include geometry, rotations, offsets and tile types, exclude magnetic laws."""
    h = hashlib.sha256()
    for v in (tiles.vertices, tiles.offset, tiles.rot, tiles.tile_type):
        a = np.ascontiguousarray(v)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()



def hash_tile_order(tiles) -> str:
    """Stable material/index signature for explicit warm-start after a rigid move."""
    h = hashlib.sha256()
    for v in (tiles.tile_type, tiles.magnet_type, tiles.stfcn_index):
        a = np.ascontiguousarray(v)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def verify_tensor_against_magtense(tiles, K, xp, get_h_field, probes=5):
    """Check a nonzero random M field at tetra centers against native MagTense.

    Verifies (source,target,component) indexing and H vs B conventions.
    Restores tile.M on success/failure. This is a verification, not a benchmark.
    """
    centers = build_tet_centers(tiles)
    rng = np.random.default_rng(314159)
    M0 = np.asarray(tiles.M, dtype=float).copy()
    mag = rng.standard_normal((tiles.n,3)) * 2e4
    try:
        tiles.M = mag
        selected = np.linspace(0, tiles.n-1, min(probes,tiles.n),dtype=int)
        direct = np.asarray(get_h_field(tiles, centers[selected]), dtype=float)
        from_op = (K @ xp.asarray(mag, dtype=K.dtype).reshape(-1)).reshape(tiles.n,3)
        from_op = xp.asnumpy(from_op[selected]) if xp is not np else from_op[selected]
        scale = max(float(np.linalg.norm(direct)), 1.0)
        rel = float(np.linalg.norm(direct - from_op)/scale)
        # FP32 matrix multiplication can differ by >1e-6 for cancellation.
        threshold = 5e-4 if K.dtype == np.float32 else 1e-9
        if not np.isfinite(rel) or rel > threshold:
            raise RuntimeError(f'K tensor layout/native H check failed: relative error={rel:.3e} '
                               f'(allowed {threshold:g}). Do not trust this solve.')
        return rel
    finally:
        tiles.M = M0

def pack_tensor_block(source_target: np.ndarray) -> np.ndarray:
    """(source, target, out_component, in_component) -> dense row-major K.

    K[3*i+a, 3*j+b] = tensor[source=j,target=i,a,b].
    No assumption about 3x3 tensor symmetry or target/source reciprocity.
    """
    a = np.asarray(source_target)
    if a.ndim != 4 or a.shape[2:] != (3, 3):
        raise ValueError('Expected (nSource,nTarget,3,3) demag tensor')
    return np.ascontiguousarray(a.transpose(1, 2, 0, 3).reshape(3*a.shape[1], 3*a.shape[0]))


def tensor_assembly(tiles, xp, *, dtype: str = 'float32', batch: int = 32,
                    cache: Path | None = None,
                    tensor_fn: Callable | None = None):
    """Chunk MagTense's exact CPU source->target tensor into a dense GPU K.

    Memory: 9*N*N*itemsize on target device, plus O(N*batch) CPU work.
    A .npy disk cache is optional and validated using a geometry SHA-256.
    """
    n = int(tiles.n)
    if n < 1 or batch < 1:
        raise ValueError('tiles and batch must be positive')
    dt = np.dtype(dtype)
    if dt not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Only float32/float64 tensors are supported')
    if tensor_fn is None:
        from magtense.magstatics import get_demag_tensor
        tensor_fn = get_demag_tensor
    bytes_required = 9 * n * n * dt.itemsize
    if xp is not np:
        free, total = xp.cuda.Device().mem_info
        if bytes_required > free * 0.82:
            raise MemoryError(f'Dense K alone needs {bytes_required/1e9:.2f} GB; '
                              f'only {free/1e9:.2f} GB CUDA memory free. '
                              'Use fewer magnetic tiles or a future matrix-free/FMM kernel.')
    geometry_sha = hash_geometry(tiles)
    t0 = time.perf_counter()
    K = xp.empty((3*n, 3*n), dtype=dt)
    diag_cpu = np.empty((n, 3, 3), dtype=dt)
    metadata = dict(geometry_sha256=geometry_sha, dtype=dt.name, n=n,
                    schema='magtense-denseK-1')
    loaded = False
    if cache is not None:
        cache = Path(cache)
        meta_path = Path(str(cache) + '.json')
        if cache.exists() and meta_path.exists():
            old = json.loads(meta_path.read_text())
            if old == metadata:
                mmap = np.load(cache, mmap_mode='r', allow_pickle=False)
                if mmap.shape != (3*n, 3*n) or mmap.dtype != dt:
                    raise ValueError('Dense K cache dimensions/dtype do not match')
                for lo in range(0, 3*n, 3*batch):
                    hi = min(3*n, lo + 3*batch)
                    K[lo:hi] = xp.asarray(np.asarray(mmap[lo:hi]))
                    for row in range(lo//3, hi//3):
                        diag_cpu[row] = mmap[3*row:3*row+3, 3*row:3*row+3]
                loaded = True
                del mmap
            else:
                print(f'Ignoring old K cache: metadata mismatched ({cache})')
    if not loaded:
        centers = build_tet_centers(tiles)
        mmap = None
        if cache is not None:
            from numpy.lib.format import open_memmap
            cache.parent.mkdir(parents=True, exist_ok=True)
            # Metadata is deleted first to invalidate interrupted writes.
            if meta_path.exists():
                meta_path.unlink()
            mmap = open_memmap(cache, mode='w+', dtype=dt, shape=(3*n, 3*n))
        for lo in range(0, n, batch):
            hi = min(n, lo + batch)
            tensor = np.asarray(tensor_fn(tiles, centers[lo:hi]), dtype=np.float64)
            if tensor.shape != (n, hi-lo, 3, 3):
                raise ValueError(f'MagTense tensor returned {tensor.shape}; '
                                 f'expected {(n,hi-lo,3,3)}')
            if not np.all(np.isfinite(tensor)):
                raise FloatingPointError('Nonfinite demagnetization tensor; check tetrahedra')
            for local, target in enumerate(range(lo, hi)):
                diag_cpu[target] = tensor[target, local]
            rows = pack_tensor_block(tensor).astype(dt, copy=False)
            K[3*lo:3*hi, :] = xp.asarray(rows)
            if mmap is not None:
                mmap[3*lo:3*hi, :] = rows
            print(f'  K rows: {hi:,}/{n:,} tiles', flush=True)
        if mmap is not None:
            mmap.flush()
            del mmap
            meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf8')
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    return K, xp.asarray(diag_cpu), {'loaded_from_cache': loaded,
                                    'K_build_or_load_seconds': time.perf_counter()-t0,
                                    'K_GB': bytes_required/1e9,
                                    'dtype': dt.name, 'tiles': n}


class MaterialLaws:
    """Single-valued isotropic B(H), linear soft, and recoil-PM constitutive laws.

    Magnet_type: 1 hard PM; 2 nonlinear soft; 3 constant-mu soft.
    Curves: lists of [H(A/m),B(T)] with H strictly increasing, first H=B=0.
    Nonlinear extrapolation beyond last H: constant M (B = mu0*(H+Ms)).
    """
    def __init__(self, tiles, curves, xp, dtype='float32'):
        self.xp = xp
        self.dtype = np.dtype(dtype)
        self.kind = xp.asarray(np.asarray(tiles.magnet_type, dtype=np.int32))
        self.curve_ids = xp.asarray(np.asarray(tiles.stfcn_index, dtype=np.int32) - 1)
        self.kind_cpu = np.asarray(tiles.magnet_type, dtype=np.int32)
        self.ids_cpu = np.asarray(tiles.stfcn_index, dtype=np.int32) - 1
        self.n = len(self.kind_cpu)
        if not set(self.kind_cpu.tolist()).issubset({1,2,3}):
            raise ValueError('Supported magnet_type: 1 permanent, 2 nonlinear, 3 linear')
        mu = np.asarray(tiles.mu_r_ea, dtype=float)
        mu_other = np.asarray(tiles.mu_r_oa, dtype=float)
        if (np.any(mu < 1) or not np.allclose(mu, mu_other) or
            not np.all(np.isfinite(mu))):
            raise ValueError('Only isotropic relative permeability >=1 supported')
        rem = np.asarray(tiles.M_rem, dtype=float)[:, None] * np.asarray(tiles.u_ea, dtype=float)
        rem[self.kind_cpu != 1] = 0
        self.rem = xp.asarray(rem, dtype=self.dtype)
        self.chi = xp.asarray(mu - 1, dtype=self.dtype)
        self.curves = []
        for k, raw in enumerate(curves or []):
            a = np.asarray(raw, dtype=float)
            if a.ndim != 2 or a.shape[1] != 2 or len(a) < 3:
                raise ValueError(f'B-H curve {k} needs >=3 [H,B] pairs')
            H, B = a[:, 0], a[:, 1]
            M = B/MU0 - H
            if (not np.all(np.isfinite(a)) or np.any(H < 0) or
                np.any(np.diff(H) <= 0) or abs(H[0]) > 1e-10 or abs(B[0]) > 1e-10 or
                np.any(M < -1e-5) or np.any(np.diff(M) < -1e-5)):
                raise ValueError(f'B-H curve {k} invalid/nonmonotonic M(H), or has hysteresis')
            self.curves.append((xp.asarray(H, dtype=self.dtype),
                                xp.asarray(np.maximum(0, M), dtype=self.dtype)))
        active = self.ids_cpu[self.kind_cpu == 2]
        if np.any(active < 0) or np.any(active >= len(self.curves)):
            raise ValueError(f'B-H indices {active.tolist()} are out of range for '
                             f'{len(self.curves)} curves')
        self.mask = [self.curve_ids == k for k in range(len(self.curves))]

    def eval(self, H):
        """Return F(H), dF/dH transverse, longitudinal, and H direction.

        dF_i = alpha_i * dH_i + (beta_i-alpha_i)*(u_i·dH_i)*u_i.
        """
        xp = self.xp
        H = xp.asarray(H, dtype=self.dtype).reshape(self.n, 3)
        r = xp.linalg.norm(H, axis=1)
        u = H / xp.maximum(r[:, None], 1e-30)
        F = self.rem + self.chi[:, None] * H
        alpha = self.chi.copy()
        beta = self.chi.copy()
        for k, (h_k, m_k) in enumerate(self.curves):
            ix = xp.clip(xp.searchsorted(h_k, r, side='right')-1, 0, len(h_k)-2)
            slope = (m_k[ix+1] - m_k[ix])/(h_k[ix+1] - h_k[ix])
            mi = m_k[ix] + (r - h_k[ix])*slope
            above = r >= h_k[-1]
            mi = xp.where(above, m_k[-1], mi)
            slope = xp.where(above, 0.0, slope)
            trans = xp.where(r > 1e-20, mi/xp.maximum(r,1e-20), slope)
            mask = self.mask[k] & (self.kind == 2)
            F = xp.where(mask[:, None], trans[:, None]*H, F)
            alpha = xp.where(mask, trans, alpha)
            beta = xp.where(mask, slope, beta)
        return F, alpha, beta, u


def apply_tangent(xp, alpha, beta, u, dH):
    proj = xp.sum(u*dH, axis=1)
    return alpha[:, None]*dH + ((beta-alpha)*proj)[:,None]*u


def make_preconditioner(xp, diagK, alpha, beta, u):
    """Exact 3x3 block-Jacobi of I - F'(H) K, used as right preconditioner."""
    n = len(alpha)
    eye = xp.eye(3, dtype=diagK.dtype)[None, :, :]
    jac = alpha[:,None,None]*eye + (beta-alpha)[:,None,None]*u[:,:,None]*u[:,None,:]
    block = eye - jac @ diagK
    try:
        inv = xp.linalg.inv(block)
    except Exception as e:
        raise NonConvergence('Singular local block-Jacobi preconditioner') from e
    if not bool(xp.all(xp.isfinite(inv))):
        raise NonConvergence('Non-finite local block-Jacobi preconditioner')
    return inv


def gmres_solve(xp, matvec, rhs, *, rtol=5e-4, restart=30, maxiter=200):
    if xp is np:
        from scipy.sparse.linalg import LinearOperator, gmres
    else:
        from cupyx.scipy.sparse.linalg import LinearOperator, gmres
    rhs = rhs.reshape(-1)
    op = LinearOperator((rhs.size,rhs.size), matvec=matvec, dtype=rhs.dtype)
    sol, info = gmres(op, rhs, rtol=rtol, atol=0.0,
                      restart=restart, maxiter=maxiter)
    return sol, info


def solve_nonlinear(K, diagK, laws: MaterialLaws, *, H_external=None, M_initial=None,
                    method='newton', tol=1e-5, maxiter=80, damping=0.15,
                    gmres_rtol=5e-4, gmres_maxiter=100, verbose=True):
    """Solve R(M)=M-F(H_ext+K M)=0; all matrices/vectors stay on selected device."""
    xp = laws.xp
    n = laws.n
    if K.shape != (3*n,3*n) or diagK.shape != (n,3,3):
        raise ValueError(f'K shape {K.shape} or diag {diagK.shape} incorrect')
    if method not in ('newton','picard'):
        raise ValueError('method must be newton/picard')
    if not 0 < damping <= 1 or not 0 < tol < 1:
        raise ValueError('Invalid damping/tolerance')
    if H_external is None:
        H_external = xp.zeros((n,3), dtype=K.dtype)
    else:
        H_external = xp.asarray(H_external, dtype=K.dtype)
        if H_external.shape == (3,):
            H_external = xp.broadcast_to(H_external, (n,3)).copy()
        if H_external.shape != (n,3):
            raise ValueError('H_external must be (3,) or (N,3)')
    M = xp.asarray(laws.rem if M_initial is None else M_initial, dtype=K.dtype).reshape(n,3).copy()
    M0_norm = as_float(xp.linalg.norm(laws.rem))
    history = []
    t0 = time.perf_counter()

    def state(m):
        H = (K @ m.reshape(-1)).reshape(n,3) + H_external
        F, a, b, u = laws.eval(H)
        R = m-F
        rms = as_float(xp.linalg.norm(R))
        scale = max(as_float(xp.linalg.norm(m)), M0_norm, np.sqrt(n)*1.0)
        return R, (H,a,b,u), rms/scale

    for it in range(maxiter+1):
        R, (H, alpha, beta, u), rel = state(M)
        history.append(rel)
        if verbose:
            print(f'  {it:3d}: relative nonlinear residual = {rel:.4e}', flush=True)
        if not np.isfinite(rel):
            raise NonConvergence('NaN/Inf residual; check B-H and mesh')
        if rel <= tol:
            if xp is not np:
                xp.cuda.Stream.null.synchronize()
            return M, {'converged':True,'iterations':it, 'residual':rel,
                       'residual_history':history,
                       'solve_seconds':time.perf_counter()-t0,'method':method}
        if it >= maxiter:
            break
        if method == 'picard':
            M = M - damping*R
            continue
        pinv = make_preconditioner(xp, diagK, alpha, beta, u)

        def pinv_apply(z):
            return (pinv @ z.reshape(n,3,1)).reshape(-1)

        def J_times_z(z):
            dm = pinv_apply(z)
            dh = (K @ dm).reshape(n,3)
            return (dm.reshape(n,3) - apply_tangent(xp, alpha,beta,u,dh)).reshape(-1)

        z, info = gmres_solve(xp, J_times_z, -R,
                              rtol=gmres_rtol, restart=30, maxiter=gmres_maxiter)
        if info != 0:
            raise NonConvergence(f'Newton GMRES failed: info={info} at iteration {it}; '
                                 'try float64, more iterations, or Picard damping')
        delta = pinv_apply(z).reshape(n,3)
        step = 1.0
        success = False
        for _ in range(13):
            trial = M + step*delta
            _, _, new_rel = state(trial)
            if np.isfinite(new_rel) and new_rel < rel:
                M = trial
                success = True
                break
            step *= 0.5
        if not success:
            raise NonConvergence(f'Newton line search failed at iteration {it}. '
                                 'Try float64 or damped Picard.')
    raise NonConvergence(f'Nonlinear solver did not converge after {maxiter} steps; '
                         f'final relative residual={history[-1]:.4g}')


def self_test() -> dict:
    """No MagTense and no GPU needed: analytic single-tile and two-curve tests."""
    from types import SimpleNamespace
    # Test: permanent magnet with mu_r=1.1 and synthetic spherical demag K=-I/3.
    tile = SimpleNamespace(n=1, magnet_type=np.array([1]), stfcn_index=np.array([1]),
                           mu_r_ea=np.array([1.1]),mu_r_oa=np.array([1.1]),
                           M_rem=np.array([1.2/MU0]), u_ea=np.array([[0,0,1.]]))
    law = MaterialLaws(tile, [], np, 'float64')
    K = -np.eye(3,dtype='float64')/3
    m, info = solve_nonlinear(K, np.array([-np.eye(3)/3]), law,
                              tol=1e-11, method='newton', verbose=False)
    exact = (1.2/MU0)/(1+0.1/3)
    np.testing.assert_allclose(m[0,2], exact, rtol=1e-10)

    # Different nonlinear B-H curves, same demag coefficient, same external field.
    tile = SimpleNamespace(n=2, magnet_type=np.array([2,2]),stfcn_index=np.array([1,2]),
                           mu_r_ea=np.ones(2),mu_r_oa=np.ones(2),
                           M_rem=np.zeros(2),u_ea=np.tile([1.,0.,0.],(2,1)))
    curves = [np.array([[0,0],[2e4,MU0*2.2e5],[8e4,MU0*3.0e5]]),
              np.array([[0,0],[1e4,MU0*1.1e5],[1.2e5,MU0*3.0e5]])]
    law = MaterialLaws(tile, curves, np,'float64')
    K = np.zeros((6,6))
    K[:3,:3] = -np.eye(3)/3
    K[3:,3:] = -np.eye(3)/3
    m,info2 = solve_nonlinear(K, np.stack([-np.eye(3)/3]*2), law,
                             H_external=np.array([2e5,0,0]),tol=1e-10,
                             maxiter=40,verbose=False)
    assert abs(m[0,0]-m[1,0])>1e3, 'Distinct B-H curves yielded identical M'
    H_final = (K @ m.reshape(-1)).reshape(2,3) + np.array([2e5,0,0])
    R = m - law.eval(H_final)[0]
    assert np.max(np.abs(R)) < 1e-4, R
    return {'permanent_mag_M_Apm':float(exact), 'nonlinear_M_Apm':m[:,0].tolist(),
            'permanent_newton_steps':info['iterations'],
            'nonlinear_newton_steps':info2['iterations']}


def run_from_export(args):
    from bridge import read_model, load_points, check_points_in_air
    from magtense_adapter import assemble_tiles
    from magtense.magstatics import get_H_field

    root = Path(args.out).resolve()
    model = read_model(root)
    pts = load_points(root, model)
    check_points_in_air(root, model, pts)
    tiles, curves, stats = assemble_tiles(model['objects'],root,max_tiles=args.max_tiles)
    xp = get_backend(args.backend)
    n = tiles.n
    print(f'Backend={args.backend}, nonlinear B-H curves={len(curves or [])}, tiles={n:,}')
    K,diag, kinfo = tensor_assembly(tiles,xp,dtype=args.dtype,batch=args.batch,
                                  cache=(root/f'K_{args.dtype}.npy') if args.cache_k else None)
    material = MaterialLaws(tiles,curves or [],xp,args.dtype)
    tensor_check = None
    if not args.skip_k_verify:
        tensor_check = verify_tensor_against_magtense(tiles,K,xp,get_H_field)
        print(f'K vs native MagTense H verified: relative={tensor_check:.3e}')
    m0 = None
    if args.warm_start:
        with np.load(args.warm_start) as warm:
            m0 = warm['M_Apm']
            same_geometry = ('geometry_sha256' in warm.files and
                             str(warm['geometry_sha256'].item()) == hash_geometry(tiles))
            if not same_geometry:
                same_order = ('tile_order_sha256' in warm.files and
                              str(warm['tile_order_sha256'].item()) == hash_tile_order(tiles))
                if not args.allow_warm_geometry_change or not same_order:
                    raise ValueError('Warm-start geometry changed. To reuse M across rigid '
                                     'moves with identical tile ordering, specify '
                                     '--allow-warm-geometry-change; otherwise use same geometry.')
        if m0.shape != (n,3) or not np.all(np.isfinite(m0)):
            raise ValueError('Warm-start magnetization shape/values invalid')
    H_ext = np.asarray(args.uniform_H, dtype=float)
    M, sinfo = solve_nonlinear(K,diag,material,H_external=H_ext,
                              M_initial=m0, method=args.method,tol=args.tol,
                              maxiter=args.maxiter,damping=args.damping,
                              gmres_rtol=args.gmres_rtol,gmres_maxiter=args.gmres_maxiter,
                              verbose=True)
    final_M = xp.asnumpy(M) if xp is not np else M
    tiles.M = np.asarray(final_M, dtype=np.float64)
    # Fortran evaluation is done only at user air probes, not on every iteration.
    H_air = get_H_field(tiles, pts) + H_ext[None,:]
    B_air = MU0 * H_air
    if not np.all(np.isfinite(B_air)):
        raise FloatingPointError('Non-finite field at air probes')
    np.savetxt(root/'gpu_B.csv',np.column_stack([pts,B_air]),delimiter=',',comments='',
               header='x_m,y_m,z_m,Bx_T,By_T,Bz_T')
    np.savez_compressed(root/'gpu_magnetization.npz',M_Apm=final_M,
                        geometry_sha256=hash_geometry(tiles),
                        tile_order_sha256=hash_tile_order(tiles))
    sinfo.update(kinfo)
    sinfo.update({'backend':args.backend,'max_tiles':args.max_tiles,
                  'H_uniform_Apm':list(args.uniform_H),
                  'nonlinear_curves':len(curves or []), 'source':model.get('source'),
                  'tile_stats':stats,'output':'gpu_B.csv',
                  'K_magtense_field_check_relative_error':tensor_check})
    (root/'gpu_solve_info.json').write_text(json.dumps(sinfo,indent=2),encoding='utf-8')
    print(f'Written: {root/"gpu_B.csv"}, residual={sinfo["residual"]:.4e}')
    print(f'Times: K={kinfo["K_build_or_load_seconds"]:.2f}s, '
          f'nonlinear={sinfo["solve_seconds"]:.2f}s')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out',help='Directory previously exported by bridge.py')
    parser.add_argument('--self-test',action='store_true',help='Analytic CPU tests (no MagTense needed)')
    parser.add_argument('--backend',choices=['cupy','numpy'],default='cupy')
    parser.add_argument('--dtype',choices=['float32','float64'],default='float32')
    parser.add_argument('--method',choices=['newton','picard'],default='newton')
    parser.add_argument('--tol',type=float,default=1e-5)
    parser.add_argument('--maxiter',type=int,default=80)
    parser.add_argument('--damping',type=float,default=0.15)
    parser.add_argument('--gmres-rtol',type=float,default=5e-4)
    parser.add_argument('--gmres-maxiter',type=int,default=100)
    parser.add_argument('--batch',type=int,default=32,help='CPU MagTense tensor target batch')
    parser.add_argument('--max-tiles',type=int,default=10000)
    parser.add_argument('--cache-k',action='store_true',help='Cache full dense K to disk (large!)')
    parser.add_argument('--warm-start',help='Path to gpu_magnetization.npz')
    parser.add_argument('--allow-warm-geometry-change',action='store_true',
                        help='Permit warm-start across moved geometry with identical tile ordering')
    parser.add_argument('--skip-k-verify',action='store_true',
                        help='Skip native MagTense K vs H field verification (not recommended)')
    parser.add_argument('--uniform-H',type=float,nargs=3,default=[0.,0.,0.],metavar=('Hx','Hy','Hz'),
                        help='Uniform applied H in A/m, outside MagTense magnet sources')
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(),indent=2))
        return
    if not args.out:
        parser.error('Supply --out or --self-test')
    run_from_export(args)


if __name__ == '__main__':
    main()
