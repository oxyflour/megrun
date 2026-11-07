"""Build only operator blocks involving an added shield; keep camera K exact.

Validation-local helper, not a change to the packaged skill's cache semantics.
Output metadata uses the normal volume-cache schema only after completion.
"""
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

TOOL = Path(r'C:\Projects\magtense-gpu\repaired-skill\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0, str(TOOL))
from gpu_solver import hash_geometry, pack_tensor_block
from tetra_volume import geometry, quadrature, subset


def assemble_added(tiles, old_cache, output, *, batch=16, deadline=None):
    from magtense.magstatics import get_demag_tensor
    old_cache, output = Path(old_cache), Path(output)
    if output.exists() or Path(str(output)+'.json').exists():
        raise FileExistsError(output)
    old_meta = json.loads(Path(str(old_cache)+'.json').read_text())
    nold = int(old_meta['n'])
    expected = {'schema':'magtense-tetra-volume-1', 'dtype':'float64', 'receiver_rule':'gauss4',
                'near_rule':'red8', 'near_levels':2, 'near_factor':1.5,
                'magtense_version':importlib.metadata.version('magtense')}
    for key,value in expected.items():
        if old_meta[key] != value:
            raise ValueError(f'Camera cache setting mismatch: {key}')
    assert 0 < nold < tiles.n
    prefix = subset(tiles, np.arange(nold))
    assert hash_geometry(prefix) == old_meta['geometry_sha256'], 'Old magnetic geometry/order changed'
    old = np.load(old_cache, mmap_mode='r', allow_pickle=False)
    assert old.shape == (3*nold,3*nold) and old.dtype == np.float64
    vertices, near = geometry(tiles,1.5)
    q4, high = quadrature(vertices,0), quadrature(vertices,2)
    n = tiles.n
    new_sources = np.arange(nold,n)
    new_tiles = subset(tiles,new_sources)
    K = np.lib.format.open_memmap(output, mode='w+', dtype='float64', shape=(3*n,3*n))
    start = time.perf_counter()

    def check_time():
        if deadline and time.time() > deadline:
            raise TimeoutError('Added-shield operator exceeded authorized group deadline')

    for lo in range(0,3*nold,3*batch):
        check_time()
        hi=min(3*nold,lo+3*batch)
        K[lo:hi,:3*nold] = old[lo:hi]
    for lo in range(0,nold,batch):
        check_time()
        hi=min(nold,lo+batch)
        tensor=np.asarray(get_demag_tensor(new_tiles,q4[lo:hi].reshape(-1,3)))
        block=pack_tensor_block(tensor.reshape(n-nold,hi-lo,4,3,3).mean(2))
        if not np.isfinite(block).all():
            raise FloatingPointError('Nonfinite camera-receiver/shield-source tensor')
        K[3*lo:3*hi,3*nold:] = block
    for lo in range(nold,n,batch):
        check_time()
        hi=min(n,lo+batch)
        tensor=np.asarray(get_demag_tensor(tiles,q4[lo:hi].reshape(-1,3)))
        block=pack_tensor_block(tensor.reshape(n,hi-lo,4,3,3).mean(2))
        if not np.isfinite(block).all():
            raise FloatingPointError('Nonfinite shield-receiver tensor')
        K[3*lo:3*hi,:] = block
    new_near_pairs = 0
    for i,sources in enumerate(near):
        check_time()
        if i < nold:
            sources=sources[sources >= nold]
        if not len(sources):
            continue
        block=np.asarray(get_demag_tensor(subset(tiles,sources),high[i])).mean(1)
        block=pack_tensor_block(block[:,None])
        if not np.isfinite(block).all():
            raise FloatingPointError('Nonfinite added near-pair tensor')
        columns=(3*sources[:,None]+np.arange(3)).ravel()
        K[3*i:3*i+3,columns]=block
        new_near_pairs += len(sources)
        if i % 256 == 0:
            print(f'Added-shield near pairs: receiver {i+1}/{n}',flush=True)
    K.flush()
    # This invariant catches accidental writes into the reused block.
    assert np.array_equal(K[:3*nold,:3*nold],old)
    meta={**expected, 'n':n, 'geometry_sha256':hash_geometry(tiles)}
    Path(str(output)+'.json').write_text(json.dumps(meta,indent=2))
    info={'old_camera_tiles':nold,'new_shield_tiles':n-nold,'total_tiles':n,
          'new_near_pairs':int(new_near_pairs), 'new_block_seconds':time.perf_counter()-start,
          'old_block_copied_bitwise':True, 'old_geometry_and_settings_verified':True,
          'old_cache':str(old_cache),'cache':str(output),'full_K_GB':K.nbytes/1e9,
          'full_native_average_field_check_required_after_load':True}
    Path(str(output)+'.reuse.json').write_text(json.dumps(info,indent=2))
    return K,info


def box_tetra(origin, size, max_in_plane=.8):
    """Conforming Freudenthal TET4 mesh, one cell through plate thickness."""
    from itertools import permutations
    origin,size=np.asarray(origin,dtype=float),np.asarray(size,dtype=float)
    counts=[1,int(np.ceil(size[1]/max_in_plane)),int(np.ceil(size[2]/max_in_plane))]
    cells=[]
    for ix in range(counts[0]):
        for iy in range(counts[1]):
            for iz in range(counts[2]):
                base=origin+size*np.array([ix,iy,iz])/counts
                for order in permutations(range(3)):
                    path=[np.zeros(3)]
                    for axis in order:
                        point=path[-1].copy();point[axis]=1;path.append(point)
                    tet=base+np.asarray(path)*size/counts
                    if np.linalg.det(tet[1:]-tet[0]) < 0:
                        tet[[1,2]]=tet[[2,1]]
                    cells.append(tet)
    vertices=np.asarray(cells)*1e-3
    volumes=np.linalg.det(vertices[:,1:]-vertices[:,:1])/6
    assert np.all(volumes > 0)
    np.testing.assert_allclose(volumes.sum(),np.prod(size)*1e-9,rtol=1e-12)
    return vertices,{'cells_through_thickness':1,'cell_counts':counts,
                     'maximum_in_plane_target_mm':max_in_plane,'tetrahedra':len(vertices),
                     'volume_m3':float(volumes.sum())}


if __name__ == '__main__':
    # Independent small native-kernel test against the ordinary full assembler.
    import tempfile
    from magtense.magstatics import Tiles
    from tetra_volume import assemble,verify
    first,_=box_tetra([0,0,0],[.2,.3,.4],1.)
    added,_=box_tetra([.3,.05,.05],[.1,.2,.2],1.)
    def tiles_of(v):
        return Tiles(n=len(v),vertices=np.asfortranarray(v.transpose(0,2,1)),tile_type=5,
                     M_rem=0.,easy_axis=[1.,0.,0.],mu_r_ea=1.,mu_r_oa=1.)
    out=Path(__file__).resolve().parent/'operator_reuse_test'
    out.mkdir(exist_ok=True)
    camera=tiles_of(first);both=tiles_of(np.concatenate([first,added]))
    camera_K,_,_=assemble(camera,np,cache=out/'camera.npy')
    full_K,_,_=assemble(both,np,cache=out/'full.npy')
    new_path=out/'reused.npy'
    reused,info=assemble_added(both,out/'camera.npy',new_path)
    np.testing.assert_allclose(reused,full_K,rtol=2e-12,atol=2e-13)
    error=float(np.linalg.norm(reused-full_K)/np.linalg.norm(full_K))
    check=verify(both,reused,np,probes=12)
    result={'full_vs_reused_relative_error':error,'native_K_H_relative_error':check,**info}
    (out/'test_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
