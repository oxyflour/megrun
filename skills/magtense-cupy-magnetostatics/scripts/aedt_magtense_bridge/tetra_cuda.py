"""CUDA analytical TET4 volume operator; same 4/256 receiving quadrature as CPU.

Only geometry preparation and near-pair enumeration remain on the host.
All source triangle integrals and receiving-point reductions execute on CUDA.
The native MagTense verification remains independent and mandatory by default.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tetra_volume import geometry, quadrature, diagonal_blocks

SOURCE = Path(__file__).with_suffix('.cu')


@lru_cache(maxsize=8)
def _module(device, source_hash):
    import cupy as cp
    module = cp.RawModule(code=SOURCE.read_text(encoding='utf-8'),
                          options=('--std=c++17', '--fmad=false'))
    module.compile()
    return module


def assemble(tiles, xp, *, dtype='float64', cache=None, near_levels=2,
             near_factor=1.5, deadline=None, receiver_batch=32, near_batch=4096):
    """Build a full dense GPU K, never loading a native CPU cache as CUDA work.

    Bounded launches and synchronization also enforce deadlines between batches
    and avoid a single long-running Windows display-GPU kernel. FP32 changes
    matrix storage only; geometry/integrals/reductions always use FP64.
    """
    total_start = time.perf_counter()
    import cupy as cp
    from gpu_solver import hash_geometry
    if xp is not cp:
        raise ValueError('CUDA K construction requires --backend cupy')
    dt = np.dtype(dtype)
    if dt not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('CUDA operator dtype must be float32 or float64')
    if receiver_batch < 1 or near_batch < 1:
        raise ValueError('CUDA batch sizes must be positive')
    n = int(tiles.n)
    if n < 1:
        raise ValueError('At least one magnetic tetrahedron is required')
    vertices, near = geometry(tiles, near_factor)
    edge = vertices[:,1:]-vertices[:,:1]
    determinants = np.linalg.det(edge)
    if not np.all(np.isfinite(vertices)) or np.any(determinants == 0):
        raise ValueError('CUDA operator requires finite, nondegenerate TET4 geometry')
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    meta = {'schema':'cuda-tetra-volume-1', 'geometry_sha256':hash_geometry(tiles),
            'n':n, 'dtype':dt.name, 'integration_dtype':'float64',
            'receiver_rule':'gauss4', 'near_rule':'red8', 'near_levels':near_levels,
            'near_factor':near_factor, 'kernel_sha256':source_hash}
    # Validate/refine the same rule even when a cache is present.
    q4, high = quadrature(vertices), quadrature(vertices, near_levels)
    cache = Path(cache) if cache else None
    meta_path = Path(str(cache)+'.json') if cache else None
    size = 9*n*n*dt.itemsize
    scratch = q4.nbytes+high.nbytes+len(vertices)*4*24*8
    if size+scratch > .82*cp.cuda.Device().mem_info[0]:
        raise MemoryError(f'CUDA volume K and quadrature need at least {(size+scratch)/1e9:.3f} GB')
    info = {'operator':'tetra_volume', 'K_builder':'cuda', 'tiles':n, 'K_GB':size/1e9,
            'dtype':dt.name, 'quadrature_points_far':4, 'quadrature_points_near':high.shape[1],
            'near_factor':near_factor, 'near_pairs':sum(map(len,near)),
            'source_kernel':'CUDA analytical tetrahedron surface-charge integral',
            'integration_dtype':'float64', 'kernel_sha256':source_hash,
            'spectrum_clipping':False, 'loaded_from_cache':False,
            'K_prepare_seconds':time.perf_counter()-total_start,
            'K_compile_seconds':0., 'K_geometry_upload_seconds':0.,
            'K_far_seconds':0., 'K_near_seconds':0., 'K_cache_write_seconds':0.,
            'K_cache_load_seconds':0.}

    def sync():
        cp.cuda.Stream.null.synchronize()

    def check_deadline():
        if deadline and time.time() > deadline:
            raise TimeoutError('CUDA K assembly exceeded its deadline')

    check_deadline()
    build_start = time.perf_counter()
    old = None
    if cache and cache.exists() and meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            pass
    if old == meta:
        phase = time.perf_counter()
        array = np.load(cache,mmap_mode='r',allow_pickle=False)
        if array.shape != (3*n,3*n) or array.dtype != dt:
            raise ValueError('CUDA cache shape/dtype mismatch')
        K = cp.asarray(array)
        del array
        sync()
        info.update(loaded_from_cache=True,K_cache_load_seconds=time.perf_counter()-phase)
    else:
        phase = time.perf_counter()
        module = _module(int(cp.cuda.Device().id),source_hash)
        suffix = 'f64' if dt.itemsize==8 else 'f32'
        prepare = module.get_function('prepare_faces')
        far = module.get_function('far_'+suffix)
        close = module.get_function('near_'+suffix)
        info['K_compile_seconds'] = time.perf_counter()-phase
        phase = time.perf_counter()
        gpu_vertices = cp.asarray(np.ascontiguousarray(vertices))
        gpu_q4 = cp.asarray(np.ascontiguousarray(q4))
        gpu_high = cp.asarray(np.ascontiguousarray(high))
        counts = np.array([len(x) for x in near],dtype=np.int32)
        receivers = cp.asarray(np.repeat(np.arange(n,dtype=np.int32),counts))
        sources = cp.asarray(np.concatenate(near).astype(np.int32))
        faces = cp.empty((n,4,24),dtype=cp.float64)
        K = cp.empty((3*n,3*n),dtype=dt)
        prepare(((n+127)//128,),(128,),(gpu_vertices,faces,np.int32(n)))
        sync()
        info['K_geometry_upload_seconds'] = time.perf_counter()-phase
        phase = time.perf_counter()
        for lo in range(0,n,receiver_batch):
            check_deadline()
            count = min(receiver_batch,n-lo)
            far(((count*n+127)//128,),(128,),
                (faces,gpu_q4,K,np.int32(n),np.int32(lo),np.int32(count)))
            sync()
        info['K_far_seconds'] = time.perf_counter()-phase
        phase = time.perf_counter()
        for lo in range(0,len(sources),near_batch):
            check_deadline()
            count = min(near_batch,len(sources)-lo)
            close(((count*32+127)//128,),(128,),
                  (faces,gpu_high,receivers,sources,K,np.int32(n),np.int32(high.shape[1]),
                   np.int32(lo),np.int32(count)))
            sync()
        info['K_near_seconds'] = time.perf_counter()-phase
        if not bool(cp.all(cp.isfinite(K))):
            raise FloatingPointError('CUDA tetrahedral integral produced a nonfinite K')
        if cache:
            phase = time.perf_counter()
            cache.parent.mkdir(parents=True,exist_ok=True)
            partial = cache.with_name(cache.name+'.partial.npy')
            disk = np.lib.format.open_memmap(partial,mode='w+',dtype=dt,shape=K.shape)
            K.get(out=disk)
            disk.flush()
            del disk
            # An interrupted replacement must never leave an old valid sidecar
            # pointing to a newly written matrix for different geometry.
            meta_path.unlink(missing_ok=True)
            partial.replace(cache)
            meta_path.write_text(json.dumps(meta,indent=2),encoding='utf-8')
            info['K_cache_write_seconds'] = time.perf_counter()-phase
    if info['loaded_from_cache'] and not bool(cp.all(cp.isfinite(K))):
        raise FloatingPointError('CUDA operator cache contains nonfinite values')
    sync()
    info['K_build_or_load_seconds'] = time.perf_counter()-build_start
    diag = diagonal_blocks(K,cp,n)
    sync()
    info['K_total_seconds'] = time.perf_counter()-total_start
    return K,diag,info
