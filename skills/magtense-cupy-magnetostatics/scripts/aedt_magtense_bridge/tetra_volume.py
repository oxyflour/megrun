"""Receiving-volume quadrature of MagTense's native tetrahedral field kernel.

Far pairs use degree-two four-point tetrahedron quadrature. Near pairs use
uniform eight-child refinement of the receiving tetrahedron. No spectrum
filtering, material modification, or GPU tensor generation is performed.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree


def quadrature(vertices, levels=0):
    if levels not in range(4):
        raise ValueError('Quadrature refinement must be 0, 1, 2 or 3')
    cells = np.asarray(vertices, dtype=float)[:, None]
    for _ in range(levels):
        v0, v1, v2, v3 = [cells[:, :, i] for i in range(4)]
        m01, m02, m03 = (v0+v1)/2, (v0+v2)/2, (v0+v3)/2
        m12, m13, m23 = (v1+v2)/2, (v1+v3)/2, (v2+v3)/2
        children = [(v0,m01,m02,m03), (m01,v1,m12,m13), (m02,m12,v2,m23),
                    (m03,m13,m23,v3), (m01,m23,m02,m03), (m01,m23,m03,m13),
                    (m01,m23,m13,m12), (m01,m23,m12,m02)]
        cells = np.concatenate([np.stack(c, axis=2) for c in children], axis=1)
    a, b = (5+3*np.sqrt(5))/20, (5-np.sqrt(5))/20
    bary = np.full((4,4), b)
    np.fill_diagonal(bary, a)
    return np.einsum('qv,ncvd->ncqd', bary, cells).reshape(len(vertices), -1, 3)


def geometry(tiles, near_factor):
    if near_factor <= 0 or not np.isfinite(near_factor):
        raise ValueError('near_factor must be finite and positive')
    if not np.all(np.asarray(tiles.tile_type) == 5):
        raise ValueError('Volume quadrature requires TET4 tiles')
    if np.any(np.asarray(tiles.offset) != 0) or np.any(np.asarray(tiles.rot) != 0):
        raise ValueError('Export tetrahedron vertices in global coordinates; zero offsets/rotations required')
    vertices = np.asarray(tiles.vertices).transpose(0, 2, 1)
    centers = vertices.mean(axis=1)
    radii = np.linalg.norm(vertices-centers[:, None], axis=2).max(axis=1)
    tree = cKDTree(centers)
    near = []
    for c, r in zip(centers, radii):
        candidates = np.asarray(tree.query_ball_point(c, near_factor*(r+radii.max())), dtype=int)
        distances = np.linalg.norm(centers[candidates]-c, axis=1)
        near.append(candidates[distances <= near_factor*(r+radii[candidates])])
    return vertices, near


def subset(tiles, indices):
    from magtense.magstatics import Tiles
    return Tiles(n=len(indices), vertices=np.asfortranarray(np.asarray(tiles.vertices)[indices]),
                 tile_type=5, M_rem=0., easy_axis=[1.,0.,0.], mu_r_ea=1., mu_r_oa=1.)


def diagonal_blocks(K, xp, n):
    indices = xp.arange(n)*3
    return xp.stack([K[indices+a, indices+b] for a in range(3) for b in range(3)], axis=1).reshape(n,3,3)


def assemble(tiles, xp, *, dtype='float64', batch=16, cache=None,
             near_levels=2, near_factor=1.5, deadline=None):
    total_start = time.perf_counter()
    from magtense.magstatics import get_demag_tensor
    from gpu_solver import hash_geometry, pack_tensor_block
    n = int(tiles.n)
    dt = np.dtype(dtype)
    if n < 1 or batch < 1 or dt not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('Positive mesh/batch and float32/float64 are required')
    vertices, near = geometry(tiles, near_factor)
    q4, high = quadrature(vertices), quadrature(vertices, near_levels)
    size = 9*n*n*dt.itemsize
    if xp is not np and size > .82*xp.cuda.Device().mem_info[0]:
        raise MemoryError(f'Volume K requires {size/1e9:.3f} GB; insufficient free GPU memory')
    meta = {'schema': 'magtense-tetra-volume-1', 'geometry_sha256': hash_geometry(tiles),
            'n': n, 'dtype': dt.name, 'receiver_rule': 'gauss4', 'near_rule': 'red8',
            'near_levels': near_levels, 'near_factor': near_factor,
            'magtense_version': importlib.metadata.version('magtense')}
    cache = Path(cache) if cache else None
    meta_path = Path(str(cache)+'.json') if cache else None
    start = time.perf_counter()
    prepare_seconds = start-total_start
    far_seconds = near_seconds = cache_finalize_seconds = 0.
    loaded = False
    if cache and cache.exists() and meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text())
        except (ValueError, OSError):
            old_meta = None
        if old_meta == meta:
            array = np.load(cache, mmap_mode='r', allow_pickle=False)
            if array.shape != (3*n,3*n) or array.dtype != dt:
                raise ValueError('Volume K cache shape/dtype mismatch')
            K = xp.asarray(array).copy()
            del array
            loaded = True
    if not loaded:
        K = xp.empty((3*n,3*n), dtype=dt)
        disk = None
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            if meta_path.exists():
                meta_path.unlink()
            disk = np.lib.format.open_memmap(cache, mode='w+', dtype=dt, shape=(3*n,3*n))
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        phase_start = time.perf_counter()
        for lo in range(0, n, batch):
            if deadline and time.time() > deadline:
                raise TimeoutError('Volume K assembly exceeded its deadline')
            hi = min(n, lo+batch)
            tensor = np.asarray(get_demag_tensor(tiles, q4[lo:hi].reshape(-1,3)))
            tensor = tensor.reshape(n,hi-lo,4,3,3).mean(axis=2)
            block = pack_tensor_block(tensor).astype(dt, copy=False)
            if not np.isfinite(block).all():
                raise FloatingPointError('Nonfinite native volume tensor')
            K[3*lo:3*hi] = xp.asarray(block)
            if disk is not None:
                disk[3*lo:3*hi] = block
            print(f'  Volume K: {hi:,}/{n:,} receivers', flush=True)
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        far_seconds = time.perf_counter()-phase_start
        phase_start = time.perf_counter()
        for i, sources in enumerate(near):
            if deadline and time.time() > deadline:
                raise TimeoutError('Near-pair K assembly exceeded its deadline')
            block = np.asarray(get_demag_tensor(subset(tiles,sources), high[i])).mean(axis=1)
            block = pack_tensor_block(block[:,None]).astype(dt, copy=False)
            if not np.isfinite(block).all():
                raise FloatingPointError('Nonfinite near-pair volume tensor')
            columns = (3*sources[:,None]+np.arange(3)).ravel()
            K[3*i:3*i+3, xp.asarray(columns)] = xp.asarray(block)
            if disk is not None:
                disk[3*i:3*i+3, columns] = block
            if i % 128 == 0:
                print(f'  Near K q={high.shape[1]}: {i+1:,}/{n:,}', flush=True)
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        near_seconds = time.perf_counter()-phase_start
        phase_start = time.perf_counter()
        if disk is not None:
            disk.flush()
            del disk
            meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        cache_finalize_seconds = time.perf_counter()-phase_start
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    info = {'operator': 'tetra_volume', 'loaded_from_cache': loaded, 'tiles': n,
            'K_build_or_load_seconds': time.perf_counter()-start, 'K_GB': size/1e9,
            'dtype': dt.name, 'quadrature_points_far': 4, 'quadrature_points_near': high.shape[1],
            'near_factor': near_factor, 'source_kernel': 'CPU native MagTense tetrahedron',
            'spectrum_clipping': False, 'K_builder': 'native',
            'K_prepare_seconds': prepare_seconds, 'K_far_seconds': far_seconds,
            'K_near_seconds': near_seconds, 'K_cache_finalize_seconds': cache_finalize_seconds,
            'K_total_seconds': time.perf_counter()-total_start}
    return K, diagonal_blocks(K,xp,n), info


def verify(tiles, K, xp, *, near_levels=2, near_factor=1.5, probes=5):
    from magtense.magstatics import get_H_field
    vertices, near = geometry(tiles, near_factor)
    q4, high = quadrature(vertices), quadrature(vertices, near_levels)
    selected = np.linspace(0, tiles.n-1, min(probes,tiles.n), dtype=int)
    old = tiles.M.copy()
    random_m = np.random.default_rng(7143).normal(size=(tiles.n,3))*20000
    try:
        tiles.M = random_m
        reference = get_H_field(tiles,q4[selected].reshape(-1,3)).reshape(len(selected),4,3).mean(axis=1)
        for j, i in enumerate(selected):
            sub = subset(tiles,near[i])
            sub.M = random_m[near[i]]
            reference[j] += get_H_field(sub,high[i]).mean(axis=0)-get_H_field(sub,q4[i]).mean(axis=0)
        predicted = (K @ xp.asarray(random_m,dtype=K.dtype).ravel()).reshape(tiles.n,3)[selected]
        if xp is not np:
            predicted = xp.asnumpy(predicted)
        relative = float(np.linalg.norm(reference-predicted)/max(np.linalg.norm(reference),1.))
        threshold = 5e-4 if K.dtype == np.float32 else 1e-9
        if not np.isfinite(relative) or relative > threshold:
            raise RuntimeError(f'Volume K/native averaged H check failed: {relative:.3e}')
        return relative
    finally:
        tiles.M = old
