"""True receiving-tetrahedron quadrature; no eigenvalue clipping or material change."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import time

import cupy as cp
import numpy as np
from cupyx.scipy.sparse.linalg import LinearOperator, eigsh

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '20261107_gap3mm_vacoflux50'
TOOL = ROOT.parents[1] / 'skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
sys.path[:0] = [str(BASE), str(TOOL)]
from magtense.magstatics import Tiles, get_demag_tensor, get_H_field
from magtense_adapter import assemble_tiles
from gpu_solver import MaterialLaws, pack_tensor_block, hash_geometry
from aedtplt import tetra_volume
import gpu_solver_dense_newton as solver


def quadrature(vertices, levels):
    cells = vertices[:, None]
    for _ in range(levels):
        center = cells.mean(axis=2)
        children = []
        for excluded in range(4):
            face = [i for i in range(4) if i != excluded]
            children.append(np.concatenate([center[:, :, None], cells[:, :, face]], axis=2))
        cells = np.concatenate(children, axis=1)
    a = (5 + 3*np.sqrt(5)) / 20
    b = (5 - np.sqrt(5)) / 20
    bary = np.full((4, 4), b)
    np.fill_diagonal(bary, a)
    points = np.einsum('qv,ncvd->ncqd', bary, cells).reshape(len(vertices), -1, 3)
    return points


def red_quadrature(vertices, levels):
    """Eight equal-volume children; every boundary edge is bisected each level."""
    cells = vertices[:, None]
    for _ in range(levels):
        v0, v1, v2, v3 = [cells[:, :, i] for i in range(4)]
        m01, m02, m03 = (v0+v1)/2, (v0+v2)/2, (v0+v3)/2
        m12, m13, m23 = (v1+v2)/2, (v1+v3)/2, (v2+v3)/2
        definitions = [(v0,m01,m02,m03), (m01,v1,m12,m13),
                       (m02,m12,v2,m23), (m03,m13,m23,v3),
                       (m01,m23,m02,m03), (m01,m23,m03,m13),
                       (m01,m23,m13,m12), (m01,m23,m12,m02)]
        cells = np.concatenate([np.stack(d, axis=2) for d in definitions], axis=1)
    a, b = (5+3*np.sqrt(5))/20, (5-np.sqrt(5))/20
    bary = np.full((4,4), b); np.fill_diagonal(bary, a)
    return np.einsum('qv,ncvd->ncqd', bary, cells).reshape(len(vertices), -1, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', type=int, default=0)
    parser.add_argument('--pair', action='store_true')
    parser.add_argument('--no-solve', action='store_true')
    parser.add_argument('--near-levels', type=int)
    parser.add_argument('--near-rule', choices=['fan','red'], default='fan')
    parser.add_argument('--budget-record', type=Path, default=ROOT/'session.json')
    args = parser.parse_args()
    begin = time.perf_counter()
    config = json.loads(args.budget_record.read_text())
    deadline = datetime.fromisoformat(config['started_UTC']).timestamp() + config['budget_seconds']
    model = json.loads((BASE / 'coarse_export/model.json').read_text())
    objects = []
    for original in model['objects']:
        if not args.pair and not original['name'].startswith('A_'):
            continue
        obj = dict(original)
        obj['mesh_npz'] = str(BASE / 'coarse_export' / original['mesh_npz'])
        objects.append(obj)
    tiles, curves, _ = assemble_tiles(objects, ROOT)
    vertices = np.asarray(tiles.vertices).transpose(0, 2, 1)
    q = quadrature(vertices, args.levels)
    qn = q.shape[1]
    base_output = ROOT / f'{"pair" if args.pair else "single"}_volume_q{qn}'
    near_function = red_quadrature if args.near_rule == 'red' else quadrature
    near_n = 4 * (8 if args.near_rule == 'red' else 4) ** (args.near_levels or 0)
    suffix = f'near_red_q{near_n}' if args.near_rule == 'red' else f'near_q{near_n}'
    output = base_output if args.near_levels is None else ROOT / f'{base_output.name}_{suffix}'
    output.mkdir(exist_ok=True)
    dt = np.float64
    cache = output / 'K.npy'
    t0 = time.perf_counter()
    if cache.exists() and not (output / 'operator_info.json').exists():
        raise RuntimeError('Existing operator has no completion metadata; retaining it for audit')
    if cache.exists():
        K = cp.asarray(np.load(cache, mmap_mode='r'))
        built = False
    elif args.near_levels is not None:
        base_cache = base_output / 'K.npy'
        if not (base_output / 'operator_info.json').exists():
            raise RuntimeError('A completed base quadrature operator is required')
        from scipy.spatial import cKDTree
        centers = vertices.mean(axis=1)
        radii = np.linalg.norm(vertices-centers[:, None], axis=2).max(axis=1)
        tree = cKDTree(centers)
        near_lists = []
        for i in range(tiles.n):
            candidates = np.asarray(tree.query_ball_point(centers[i], 1.5*(radii[i]+radii.max())), dtype=int)
            distances = np.linalg.norm(centers[candidates]-centers[i], axis=1)
            near_lists.append(candidates[distances <= 1.5*(radii[i]+radii[candidates])])
        high_q = near_function(vertices, args.near_levels)
        print(f'Near quadrature {high_q.shape[1]}: mean sources {np.mean([len(x) for x in near_lists]):.1f}', flush=True)
        matrix = np.lib.format.open_memmap(cache, mode='w+', dtype=dt, shape=(3*tiles.n, 3*tiles.n))
        matrix[:] = np.load(base_cache, mmap_mode='r')
        K = cp.asarray(matrix)
        for i, near in enumerate(near_lists):
            if time.time() > deadline:
                raise RuntimeError('Study budget exhausted during near-pair quadrature')
            sub = Tiles(n=len(near), vertices=np.asfortranarray(np.asarray(tiles.vertices)[near]),
                        tile_type=5, M_rem=0., easy_axis=[1., 0., 0.], mu_r_ea=1., mu_r_oa=1.)
            block = pack_tensor_block(np.asarray(get_demag_tensor(sub, high_q[i])).mean(axis=1)[:, None])
            columns = (3*near[:, None]+np.arange(3)).ravel()
            matrix[3*i:3*i+3, columns] = block
            K[3*i:3*i+3, cp.asarray(columns)] = cp.asarray(block)
            if i % 128 == 0:
                print(f'Near q={high_q.shape[1]}: {i+1}/{tiles.n} receiver cells', flush=True)
        matrix.flush()
        del matrix
        built = True
    else:
        matrix = np.lib.format.open_memmap(cache, mode='w+', dtype=dt, shape=(3*tiles.n, 3*tiles.n))
        K = cp.empty((3*tiles.n, 3*tiles.n), dtype=dt)
        for lo in range(0, tiles.n, 16):
            if time.time() > deadline:
                raise RuntimeError('Study budget exhausted during volume quadrature')
            hi = min(lo+16, tiles.n)
            T = get_demag_tensor(tiles, q[lo:hi].reshape(-1, 3))
            T = np.asarray(T).reshape(tiles.n, hi-lo, qn, 3, 3).mean(axis=2)
            block = pack_tensor_block(T)
            if not np.isfinite(block).all():
                raise FloatingPointError('Non-finite native target quadrature tensor')
            matrix[3*lo:3*hi] = block
            K[3*lo:3*hi] = cp.asarray(block)
            if lo % 256 == 0:
                print(f'Volume q={qn}: {hi}/{tiles.n} receiver cells', flush=True)
        matrix.flush()
        del matrix
        built = True
    cp.cuda.Stream.null.synchronize()
    build_seconds = time.perf_counter() - t0
    # A changed observation functional needs its own native direct-field check.
    rng = np.random.default_rng(7143)
    random_m = rng.normal(size=(tiles.n, 3)) * 20000
    saved_m = tiles.M.copy()
    tiles.M = random_m
    selected = np.linspace(0, tiles.n-1, 5, dtype=int)
    reference = get_H_field(tiles, q[selected].reshape(-1, 3)).reshape(5, qn, 3).mean(axis=1)
    if args.near_levels is not None:
        if 'high_q' not in locals():
            from scipy.spatial import cKDTree
            centers = vertices.mean(axis=1)
            radii = np.linalg.norm(vertices-centers[:, None], axis=2).max(axis=1)
            near_lists = [np.flatnonzero(np.linalg.norm(centers-c, axis=1) <= 1.5*(r+radii)) for c, r in zip(centers, radii)]
            high_q = near_function(vertices, args.near_levels)
        for j, i in enumerate(selected):
            near = near_lists[i]
            sub = Tiles(n=len(near), vertices=np.asfortranarray(np.asarray(tiles.vertices)[near]),
                        tile_type=5, M_rem=0., easy_axis=[1., 0., 0.], mu_r_ea=1., mu_r_oa=1.)
            sub.M = random_m[near]
            reference[j] += get_H_field(sub, high_q[i]).mean(axis=0) - get_H_field(sub, q[i]).mean(axis=0)
    predicted = cp.asnumpy((K @ cp.asarray(random_m).ravel()).reshape(tiles.n, 3)[selected])
    check = np.linalg.norm(reference-predicted) / max(np.linalg.norm(reference), 1.)
    assert check < 1e-9, check
    tiles.M = saved_m
    volumes = np.abs(tetra_volume(vertices))
    w = cp.repeat(cp.sqrt(cp.asarray(volumes)), 3)

    def symmetric(v):
        return .5 * (w * (K @ (v/w)) + (K.T @ (w*v)) / w)

    op = LinearOperator(K.shape, matvec=symmetric, dtype=K.dtype)
    values, vectors = eigsh(op, k=1, which='LA', ncv=80, tol=1e-7, maxiter=2000)
    eigenvalue = float(values[0].item())
    eigen_residual = float(cp.linalg.norm(symmetric(vectors[:, 0])-values[0]*vectors[:, 0]).item())
    result = {'receiver_quadrature_points': qn, 'geometry_sha256': hash_geometry(tiles),
              'tiles': tiles.n, 'K_build_seconds': build_seconds, 'new_K_built': built,
              'native_averaged_H_check': float(check), 'symmetric_K_largest_eigenvalue': eigenvalue,
              'eigen_residual': eigen_residual, 'uses_eigenvalue_clipping': False,
              'eigen_estimate_converged': eigen_residual < 1e-7,
              'same_source_tensor': 'MagTense native exact tetrahedron kernel'}
    if args.near_levels is not None:
        result['near_quadrature_points'] = high_q.shape[1]
        result['near_source_count_mean'] = float(np.mean([len(x) for x in near_lists]))
        result['near_quadrature_rule'] = args.near_rule
    (output / 'operator_info.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if args.no_solve:
        return
    laws = MaterialLaws(tiles, curves, cp, 'float64')
    diag = cp.stack([K[3*i:3*i+3, 3*i:3*i+3] for i in range(tiles.n)])
    for label, mu in [('linear1000', 1000), ('real_bh', None)]:
        if time.time() > deadline:
            raise RuntimeError('Study budget exhausted before solve')
        try:
            if mu:
                old_kind = tiles.magnet_type.copy()
                old_mu_ea, old_mu_oa = tiles.mu_r_ea.copy(), tiles.mu_r_oa.copy()
                mask = old_kind == 2
                kinds = old_kind.copy(); kinds[mask] = 3
                mur = old_mu_ea.copy(); mur[mask] = mu
                tiles.magnet_type = kinds; tiles.mu_r_ea = mur; tiles.mu_r_oa = mur
                law = MaterialLaws(tiles, [], cp, 'float64')
                chi = cp.repeat(law.chi, 3)
                J = -chi[:, None] * K
                idx = cp.arange(3*tiles.n); J[idx, idx] += 1
                M = cp.linalg.solve(J, law.rem.ravel()).reshape(tiles.n, 3)
                del J
                tiles.magnet_type = old_kind; tiles.mu_r_ea = old_mu_ea; tiles.mu_r_oa = old_mu_oa
                info = {'solver': 'dense exact linear solve'}
            else:
                law = laws
                M, info = solver.solve_nonlinear(K, diag, law, tol=1e-8, maxiter=80, verbose=True)
            H = (K @ M.ravel()).reshape(tiles.n, 3)
            R = M - law.eval(H)[0]
            residual = float(cp.linalg.norm(R).item()) / max(float(cp.linalg.norm(M).item()), 1.)
            assert residual < 1e-8
            tiles.M = cp.asnumpy(M)
            points = np.loadtxt(BASE / 'points.csv', delimiter=',', skiprows=1)
            B = 4e-7*np.pi*get_H_field(tiles, points)
            np.savetxt(output / f'{label}_B.csv', np.c_[points, B], delimiter=',',
                       header='x_m,y_m,z_m,Bx_T,By_T,Bz_T', comments='')
            np.savez_compressed(output / f'{label}_solution.npz', M_Apm=tiles.M)
            result[label] = {'converged': True, 'residual': residual,
                            'max_M_Apm': float(np.max(np.linalg.norm(tiles.M, axis=1))), **info}
        except Exception as error:
            result[label] = {'converged': False, 'error': repr(error)}
        (output / 'operator_info.json').write_text(json.dumps(result, indent=2))
    result['elapsed_seconds'] = time.perf_counter()-begin
    (output / 'operator_info.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
