"""Matrix-free TET4 volume field: FMM3D dipoles plus analytical near correction.

Used by gpu_solver --operator volume --k-builder fmm. Far fields use CPU
FMM3D or a reusable CUDA interpolation FMM; CuPy also accelerates analytical
correction, neighbour queries and the material iteration. Sources
are positive tetrahedral quadrature dipoles, NOT one centre dipole per tile.
eps controls only the point FMM; source quadrature and correction radius have
independent errors. No dense fallback or dense K cache is used here.
"""
from __future__ import annotations

import hashlib
import itertools
import time

import numpy as np
from scipy.sparse import bsr_matrix
from scipy.spatial import cKDTree

from tetra_volume import quadrature, subset


def _source_trees(centres, radii):
    bins = np.floor(np.log2(radii / radii.min())).astype(np.int32)
    groups = []
    for key in np.unique(bins):
        ids = np.flatnonzero(bins == key)
        groups.append((ids, cKDTree(centres[ids]), radii[ids].max()))
    return groups


def _neighbours(centres, radii, factor, batch, *, groups=None, targets=None, target_radii=None):
    """Bound query work by receiver batches and bins of comparable source size."""
    groups = _source_trees(centres, radii) if groups is None else groups
    if isinstance(groups,_CUDANeighbours):
        yield from groups.neighbours(factor,batch,targets,target_radii)
        return
    targets = centres if targets is None else targets
    target_radii = radii if target_radii is None else target_radii
    for lo in range(0, len(targets), batch):
        hi = min(lo + batch, len(targets))
        rows = [[] for _ in range(hi-lo)]
        for ids, tree, maximum in groups:
            candidates = tree.query_ball_point(targets[lo:hi],
                factor*(target_radii[lo:hi]+maximum), return_sorted=True)
            for j, candidate in enumerate(candidates):
                selected = ids[np.asarray(candidate, dtype=np.intp)]
                distance = np.linalg.norm(centres[selected]-targets[lo+j], axis=1)
                rows[j].append(selected[distance <= factor*(target_radii[lo+j]+radii[selected])])
        yield lo, [np.sort(np.concatenate(row)).astype(np.int32) for row in rows]


class _CUDANeighbours:
    """GPU KD trees and bounded count/fill queries with the original radius rule."""
    def __init__(self,centres,radii,*,query_batch=2048,max_pairs=4000000):
        import cupy as cp
        from cupyx.scipy.spatial import KDTree
        from tetra_cuda import SOURCE,_module
        self.cp,self.query_batch,self.max_pairs = cp,query_batch,max_pairs
        self.centres = cp.ascontiguousarray(cp.asarray(centres,dtype=cp.float64))
        self.radii = cp.ascontiguousarray(cp.asarray(radii,dtype=cp.float64))
        self.kernel = _module(int(cp.cuda.Device().id),hashlib.sha256(SOURCE.read_bytes()).hexdigest()).get_function('fmm_neighbours')
        bins = np.floor(np.log2(radii/radii.min())).astype(np.int32)
        self.groups = []
        for key in np.unique(bins):
            ids = np.flatnonzero(bins == key).astype(np.int32)
            device_ids = cp.asarray(ids)
            tree = KDTree(cp.ascontiguousarray(self.centres[device_ids]))
            if not hasattr(tree,'tree') or not hasattr(tree,'index'):
                raise RuntimeError('GPU neighbour search requires CuPy KDTree tree/index arrays')
            self.groups.append((device_ids,cp.ascontiguousarray(tree.tree),
                                cp.ascontiguousarray(tree.index),float(radii[ids].max())))
        self.dummy = cp.empty(1,dtype=cp.uint64)

    def _counts(self,points,radii,factor):
        cp = self.cp
        counts = cp.zeros(len(points),dtype=cp.int64)
        for ids,tree,order,maximum in self.groups:
            self.kernel(((len(points)+127)//128,),(128,),
                (tree,order,ids,self.radii,points,radii,np.int32(len(tree)),np.int32(len(points)),
                 np.float64(maximum),np.float64(factor),np.int32(0),counts,self.dummy,self.dummy))
        return counts

    def counts(self,factor):
        """Count all rows without allocating neighbour IDs or Python lists."""
        cp = self.cp
        result = np.empty(len(self.centres),dtype=np.int64)
        for lo in range(0,len(result),self.query_batch):
            hi = min(len(result),lo+self.query_batch)
            result[lo:hi] = cp.asnumpy(self._counts(self.centres[lo:hi],self.radii[lo:hi],factor))
        return result

    def neighbours(self,factor,batch,targets=None,target_radii=None):
        cp = self.cp
        points = self.centres if targets is None else cp.ascontiguousarray(cp.asarray(targets,dtype=cp.float64))
        radii = self.radii if target_radii is None else cp.asarray(target_radii,dtype=cp.float64)
        lo = 0
        while lo<len(points):
            hi = min(len(points),lo+self.query_batch)
            while True:
                counts = self._counts(points[lo:hi],radii[lo:hi],factor)
                pointer = cp.empty(len(counts)+1,dtype=cp.int64)
                pointer[0] = 0
                cp.cumsum(counts,out=pointer[1:])
                pairs = int(pointer[-1].item())
                if pairs<=self.max_pairs:
                    break
                if hi-lo==1:
                    raise MemoryError('One GPU neighbour row exceeds the compact query pair budget')
                hi = lo+max(1,(hi-lo)//2)
            keys = cp.empty(pairs,dtype=cp.uint64)
            cursor = pointer[:-1].copy()
            for ids,tree,order,maximum in self.groups:
                self.kernel(((hi-lo+127)//128,),(128,),
                    (tree,order,ids,self.radii,points[lo:hi],radii[lo:hi],np.int32(len(tree)),
                     np.int32(hi-lo),np.float64(maximum),np.float64(factor),np.int32(1),
                     counts,cursor,keys))
            if not bool(cp.all(cursor == pointer[1:])):
                raise RuntimeError('GPU neighbour count/fill mismatch')
            keys.sort()
            ids = cp.asnumpy((keys & cp.uint64(0xffffffff)).astype(cp.int32))
            offsets = cp.asnumpy(pointer)
            for first in range(0,hi-lo,batch):
                stop = min(hi-lo,first+batch)
                yield lo+first,[ids[offsets[j]:offsets[j+1]] for j in range(first,stop)]
            lo = hi


class _CUDAPointFMM:
    """GPU interpolation FMM for the 3-D dipole-field tensor, on a fixed tree.

    Tensor Chebyshev interpolation supplies P2M/M2M/M2L/L2L/L2P. Adjacent
    leaf boxes use exact point interactions. No centre-dipole treecode is used.
    eps is checked against direct GPU point sums at deterministic probes;
    this sampled check is not a uniform mathematical error bound.
    """
    def __init__(self,sources,targets=None,*,eps=1e-6,order=None,
                 budget_bytes=12*1024**3,leaf_size=512):
        import cupy as cp
        from tetra_cuda import SOURCE,_module
        started = time.perf_counter()
        self.cp,self.eps,self.budget = cp,eps,budget_bytes
        self.p = int(order if order is not None else max(4,np.ceil(-np.log10(eps))+4))
        if not (0<eps<1 and 3<=self.p<=16 and budget_bytes>0 and leaf_size>0):
            raise ValueError('Invalid CUDA FMM accuracy, interpolation order or memory budget')
        self.k = self.p**3
        self.module = _module(int(cp.cuda.Device().id),hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        self.kernels = {key:self.module.get_function('fmm_'+key) for key in
                        ('morton','box_lookup','p2m','l2p','p2p','point_reference')}
        nt = len(sources) if targets is None else len(targets)
        if (len(sources)<1 or nt<1 or max(len(sources),nt)>np.iinfo(np.int32).max):
            raise ValueError('CUDA FMM requires nonempty point sets within int32 indexing')
        self._check_budget((len(sources)+nt)*128,'point geometry and sorting')
        self.sources = cp.ascontiguousarray(cp.asarray(sources,dtype=cp.float64))
        self.targets = self.sources if targets is None else cp.ascontiguousarray(cp.asarray(targets,dtype=cp.float64))
        for points in (self.sources,self.targets):
            if points.ndim!=2 or points.shape[1]!=3 or not bool(cp.all(cp.isfinite(points)&(points>=0)&(points<=1))):
                raise ValueError('CUDA point FMM coordinates must be finite and normalized into [0,1]^3')
        source_keys,source_order = self._sort(self.sources)
        for depth in range(21):
            leaf_keys,pointer = self._leaves(source_keys,depth)
            if (len(sources)/len(leaf_keys)<=leaf_size and int(cp.diff(pointer).max())<=4096) or depth==20:
                break
        if int(cp.diff(pointer).max())>65536:
            raise MemoryError('CUDA FMM has an excessively occupied leaf, including coincident points')
        self.depth = depth
        self.source_tree = self._hierarchy(leaf_keys)
        self.source_order,self.source_pointer = source_order,pointer
        if targets is None:
            self.target_tree,self.target_order,self.target_pointer = self.source_tree,source_order,pointer
        else:
            target_keys,self.target_order = self._sort(self.targets)
            leaf_keys,self.target_pointer = self._leaves(target_keys,depth)
            self.target_tree = self._hierarchy(leaf_keys)
        self.target_boxes = cp.repeat(cp.arange(len(self.target_tree[-1]['keys']),dtype=cp.int32),
                                      cp.diff(self.target_pointer))
        expansion_bytes = sum(len(t['keys']) for tree in (self.source_tree,self.target_tree) for t in tree)*self.k*3*8
        geometry_bytes = (len(sources)+nt)*80 + 8*self.k*self.k*8
        self.estimated_bytes = expansion_bytes+geometry_bytes+256*1024**2
        self._check_budget(self.estimated_bytes,'moments and local expansions')
        nodes = np.cos((2*np.arange(self.p)+1)*np.pi/(2*self.p))
        weights = (-1.)**np.arange(self.p)*np.sin((2*np.arange(self.p)+1)*np.pi/(2*self.p))
        self.nodes,self.weights = cp.asarray(nodes),cp.asarray(weights)
        self.node_points = cp.asarray(np.stack(np.meshgrid(nodes,nodes,nodes,indexing='ij'),axis=-1).reshape(-1,3)/2)
        transfers = []
        for child in range(8):
            basis = []
            for axis in range(3):
                x = nodes/2 + (.5 if child&(1<<axis) else -.5)
                delta = x[:,None]-nodes[None,:]
                with np.errstate(divide='ignore',invalid='ignore'):
                    values = weights[None,:]/delta
                    values /= values.sum(axis=1,keepdims=True)
                for row in np.flatnonzero(np.any(delta==0,axis=1)):
                    values[row] = delta[row]==0
                basis.append(values.T)
            transfers.append(np.kron(basis[0],np.kron(basis[1],basis[2])))
        self.transfers = cp.asarray(np.stack(transfers))
        interaction_bytes = self._connect_targets()
        cp.cuda.Stream.null.synchronize()
        self.info = {'fmm_gpu_depth':depth,'fmm_gpu_order':self.p,
                     'fmm_gpu_target_points':len(self.targets),
                     'fmm_gpu_leaf_boxes':len(self.source_tree[-1]['keys']),
                     'fmm_gpu_interaction_index_bytes':interaction_bytes,
                     'fmm_gpu_estimated_peak_bytes':self.estimated_bytes,
                     'fmm_gpu_setup_seconds':time.perf_counter()-started,
                     'fmm_gpu_point_probe_error':None,'fmm_gpu_evaluations':0,
                     'fmm_gpu_accuracy_scope':'Direct point-field samples, not a uniform error bound'}

    def _connect_targets(self):
        cp,depth = self.cp,self.depth
        self.near = cp.empty((len(self.target_tree[-1]['keys']),27),dtype=cp.int32)
        for column,offset in enumerate(itertools.product(range(-1,2),repeat=3)):
            self.near[:,column] = self._lookup(depth,offset,False)
        self.interactions = {}
        interaction_bytes = 0
        for offset in itertools.product(range(-3,4),repeat=3):
            if max(map(abs,offset))<=1:
                continue
            records = []
            for level in range(2,depth+1):
                lookup = self._lookup(level,offset,True)
                ids = cp.flatnonzero(lookup>=0).astype(cp.int32)
                if len(ids):
                    sources_at = lookup[ids]
                    interaction_bytes += ids.nbytes+sources_at.nbytes
                    self._check_budget(self.estimated_bytes+interaction_bytes,'interaction lists')
                    records.append((level,ids,sources_at))
            if records:
                self.interactions[offset] = records
        self.estimated_bytes += interaction_bytes
        return interaction_bytes

    def subset(self,targets):
        """Evaluate a target subset with identical source tree, order and box depth."""
        cp = self.cp
        plan = object.__new__(type(self))
        for name in ('cp','eps','budget','p','k','module','kernels','sources','depth','source_tree',
                     'source_order','source_pointer','nodes','weights','node_points','transfers','estimated_bytes'):
            setattr(plan,name,getattr(self,name))
        plan.targets = cp.ascontiguousarray(cp.asarray(targets,dtype=cp.float64))
        if (plan.targets.ndim!=2 or plan.targets.shape[1]!=3 or not len(plan.targets) or
            not bool(cp.all(cp.isfinite(plan.targets)&(plan.targets>=0)&(plan.targets<=1)))):
            raise ValueError('CUDA FMM subset targets must be normalized into [0,1]^3')
        keys,plan.target_order = plan._sort(plan.targets)
        leaf_keys,plan.target_pointer = plan._leaves(keys,plan.depth)
        plan.target_tree = plan._hierarchy(leaf_keys)
        plan.target_boxes = cp.repeat(cp.arange(len(leaf_keys),dtype=cp.int32),cp.diff(plan.target_pointer))
        index_bytes = plan._connect_targets()
        plan.info = dict(self.info,fmm_gpu_target_points=len(plan.targets),fmm_gpu_evaluations=0,
                         fmm_gpu_interaction_index_bytes=index_bytes,fmm_gpu_shared_source_tree=True)
        return plan

    def _check_budget(self,required,stage):
        cp = self.cp
        free = cp.cuda.Device().mem_info[0]+cp.get_default_memory_pool().free_bytes()
        if required>min(self.budget,int(free*.85)):
            raise MemoryError(f'CUDA FMM {stage} needs approximately {required/1024**3:.2f} GiB; '
                              'exceeds the far-field budget or available GPU memory')

    def _sort(self,points):
        cp = self.cp
        keys = cp.empty(len(points),dtype=cp.uint64)
        self.kernels['morton'](((len(points)+127)//128,),(128,),
                              (points,np.int32(len(points)),np.int32(20),keys))
        order = cp.argsort(keys).astype(cp.int64)
        return keys[order],order

    def _leaves(self,sorted_keys,depth):
        cp = self.cp
        keys = sorted_keys>>cp.uint64(3*(20-depth))
        starts = cp.flatnonzero(cp.r_[cp.asarray([True]),keys[1:]!=keys[:-1]]).astype(cp.int64)
        return keys[starts],cp.r_[starts,cp.asarray([len(keys)],dtype=cp.int64)]

    def _hierarchy(self,leaf_keys):
        cp = self.cp
        levels = [None]*(self.depth+1)
        levels[-1] = {'keys':leaf_keys}
        for level in range(self.depth,0,-1):
            parent,indices = cp.unique(levels[level]['keys']>>cp.uint64(3),return_inverse=True)
            levels[level]['parents'] = indices.astype(cp.int32)
            levels[level-1] = {'keys':parent}
        return levels

    def _lookup(self,level,offset,interaction):
        cp = self.cp
        target,source = self.target_tree[level]['keys'],self.source_tree[level]['keys']
        result = cp.empty(len(target),dtype=cp.int32)
        self.kernels['box_lookup'](((len(target)+127)//128,),(128,),
            (target,source,np.int32(len(target)),np.int32(len(source)),np.int32(level),
             *map(np.int32,offset),np.int32(interaction),result))
        return result

    def _translate(self,values,tree,level,up):
        cp = self.cp
        children = tree[level]
        rows = len(tree[level-1]['keys']) if up else len(children['keys'])
        result = cp.zeros((rows,self.k,3),dtype=cp.float64)
        chunk = max(1,(64*1024**2)//(self.k*3*8))
        for parity in range(8):
            ids = cp.flatnonzero((children['keys']&cp.uint64(7))==parity).astype(cp.int32)
            for lo in range(0,len(ids),chunk):
                selected = ids[lo:lo+chunk]
                parents = children['parents'][selected]
                source = values[selected if up else parents]
                matrix = source.transpose(1,0,2).reshape(self.k,-1)
                transfer = self.transfers[parity] if up else self.transfers[parity].T
                transformed = (transfer@matrix).reshape(self.k,len(selected),3).transpose(1,0,2)
                result[parents if up else selected] += transformed
        return result

    def _m2l_matrix(self,offset):
        cp = self.cp
        delta = self.node_points[:,None,:]-self.node_points[None,:,:]+cp.asarray(offset)
        radius2 = cp.sum(delta*delta,axis=2)
        inverse = 1/cp.sqrt(radius2)
        result = cp.empty((self.k,3,self.k,3),dtype=cp.float64)
        for a in range(3):
            for b in range(3):
                result[:,a,:,b] = (3*delta[:,:,a]*delta[:,:,b]*inverse**5
                                  - (inverse**3 if a==b else 0))/(4*np.pi)
        return result.reshape(3*self.k,3*self.k)

    def direct(self,moments,targets):
        cp = self.cp
        targets = cp.ascontiguousarray(cp.asarray(targets,dtype=cp.float64))
        result = cp.zeros((len(targets),3),dtype=cp.float64)
        for first in range(0,len(targets),128):
            stop = min(len(targets),first+128)
            for lo in range(0,len(self.sources),32768):
                hi = min(len(self.sources),lo+32768)
                self.kernels['point_reference'](((stop-first+3)//4,),(128,),
                    (self.sources,targets[first:stop],moments,np.int32(lo),np.int32(hi),
                     np.int32(stop-first),result[first:stop]))
        return result

    def __call__(self,moments):
        cp = self.cp
        self._check_budget(self.estimated_bytes,'evaluation workspace')
        moments = cp.ascontiguousarray(cp.asarray(moments,dtype=cp.float64))
        if moments.shape!=self.sources.shape or not bool(cp.all(cp.isfinite(moments))):
            raise ValueError('Invalid CUDA FMM point moments')
        multipoles = [None]*(self.depth+1)
        multipoles[-1] = cp.empty((len(self.source_tree[-1]['keys']),self.k,3),dtype=cp.float64)
        for first in range(0,len(multipoles[-1]),128):
            count = min(128,len(multipoles[-1])-first)
            self.kernels['p2m']((count,),(128,),
                (self.sources,moments,self.source_order,self.source_pointer,self.source_tree[-1]['keys'],
                 np.int32(first),np.int32(len(multipoles[-1])),np.int32(self.depth),np.int32(self.p),
                 self.nodes,self.weights,multipoles[-1]))
        for level in range(self.depth,0,-1):
            multipoles[level-1] = self._translate(multipoles[level],self.source_tree,level,True)
        local = [cp.zeros((len(t['keys']),self.k,3),dtype=cp.float64) for t in self.target_tree]
        chunk = max(1,(64*1024**2)//(self.k*3*8))
        for offset,records in self.interactions.items():
            matrix = self._m2l_matrix(offset)
            for level,targets,sources in records:
                for first in range(0,len(targets),chunk):
                    stop = min(len(targets),first+chunk)
                    field = multipoles[level][sources[first:stop]].reshape(stop-first,3*self.k)@matrix.T
                    local[level][targets[first:stop]] += field.reshape(-1,self.k,3)*8.**level
        del multipoles
        for level in range(1,self.depth+1):
            local[level] += self._translate(local[level-1],self.target_tree,level,False)
        field = cp.empty((len(self.targets),3),dtype=cp.float64)
        for first in range(0,len(self.targets),4096):
            stop = min(len(self.targets),first+4096)
            self.kernels['l2p'](((stop-first+3)//4,),(128,),
                (self.targets,self.target_order,self.target_boxes,self.target_tree[-1]['keys'],local[-1],
                 np.int32(first),np.int32(stop),np.int32(self.depth),np.int32(self.p),self.nodes,self.weights,field))
            self.kernels['p2p'](((stop-first+3)//4,),(128,),
                (self.sources,self.targets,moments,self.source_order,self.target_order,self.source_pointer,
                 self.target_boxes,self.near,np.int32(first),np.int32(stop),field))
        selected = cp.asarray(np.linspace(0,len(field)-1,min(17,len(field)),dtype=np.int32))
        reference = self.direct(moments,self.targets[selected])
        error = float(cp.linalg.norm(field[selected]-reference)/cp.maximum(cp.linalg.norm(reference),1.))
        if not np.isfinite(error) or error>self.eps:
            raise RuntimeError(f'CUDA FMM sampled point-field error {error:.3e} exceeds {self.eps:g}; '
                               'increase --fmm-gpu-order, with additional GPU memory cost')
        if not bool(cp.all(cp.isfinite(field))):
            raise FloatingPointError('Nonfinite CUDA FMM field')
        self.info.update(fmm_gpu_point_probe_error=error,fmm_gpu_evaluations=self.info['fmm_gpu_evaluations']+1)
        return field


def _dipole_blocks(targets, sources, weights):
    """Independent direct point-dipole tensor; coincident points are omitted."""
    delta = targets[None, :, None, :] - sources[:, None, :, :]
    radius2 = np.sum(delta*delta, axis=-1)
    inverse = np.zeros_like(radius2)
    np.divide(1., np.sqrt(radius2), out=inverse, where=radius2 > 0)
    tensor = (3*delta[..., :, None]*delta[..., None, :]*inverse[..., None, None]**5
              - np.eye(3)*inverse[..., None, None]**3)
    return tensor.sum(axis=(1, 2)) * (weights/(4*np.pi*len(targets)))[:, None, None]


def _apply_near_blocks(blocks, indices, pointer, vector, source_count, xp):
    """Apply rectangular near-correction BSR data without expanding scalar CSR."""
    if xp is np:
        return bsr_matrix((blocks, indices, pointer),
                          shape=(3*(len(pointer)-1), 3*source_count)) @ vector
    from tetra_cuda import SOURCE, _module
    module = _module(int(xp.cuda.Device().id), hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    result = xp.empty((len(pointer)-1, 3), dtype=blocks.dtype)
    suffix = 'f64' if blocks.dtype.itemsize == 8 else 'f32'
    module.get_function('fmm_apply_'+suffix)(((len(pointer)-1+3)//4,), (128,),
        (blocks, xp.asarray(indices), xp.asarray(pointer), vector,
         np.int32(len(pointer)-1), result))
    return result.ravel()


class AdaptiveNearCompression:
    """Experimental fixed-coefficient H-matrix for a bounded receiver patch.

    Input is the ORIGINAL masked near-correction BSR, including self blocks.
    Geometry uses full tetrahedron boxes. Strong admissibility tests separation;
    weak admissibility also tests touching blocks without shared tetrahedra.
    Error budgets are either relative per block or allocated from the whole
    patch Frobenius budget in proportion to original nonzero pair counts.
    Unprofitable blocks split again, and touching leaves retain exact BSR data.
    This is not wired into the production nonlinear solver. The real-mesh
    experiment in validation/20261109_fmm_refined/run_fmm.py uses this class.
    """
    def __init__(self, blocks, indices, pointer, receiver_vertices, source_vertices,
                 xp, *, rtol=1e-4, leaf_size=8, eta=2., max_svd_entries=1000000,
                 admissibility='weak', error_budget='global', decomposition='cuda'):
        if (not 0 < rtol < 1 or leaf_size < 1 or eta <= 0 or max_svd_entries < 9 or
                admissibility not in ('strong','weak') or error_budget not in ('global','relative') or
                decomposition not in ('cpu','cuda')):
            raise ValueError('Invalid near-compression tolerance or partition settings')
        started = time.perf_counter()
        self.xp = xp
        nr, ns = len(receiver_vertices), len(source_vertices)
        self.shape = (3*nr, 3*ns)
        indices, pointer = np.asarray(indices, dtype=np.int32), np.asarray(pointer, dtype=np.int64)
        if (blocks.shape != (len(indices), 3, 3) or pointer.shape != (nr+1,) or
                pointer[0] != 0 or pointer[-1] != len(indices) or np.any(np.diff(pointer)<0) or
                np.any(indices<0) or np.any(indices>=ns) or blocks.dtype != np.float64):
            raise ValueError('Expected rectangular FP64 near-correction BSR')
        rowids = np.repeat(np.arange(nr, dtype=np.int32), np.diff(pointer))
        pairmap = np.full((nr, ns), -1, dtype=np.int64)
        pairmap[rowids, indices] = np.arange(len(indices))
        if np.count_nonzero(pairmap>=0) != len(indices):
            raise ValueError('Duplicate near-correction pair')
        dense = xp.zeros(self.shape, dtype=xp.float64)
        dense.reshape(nr, 3, ns, 3)[xp.asarray(rowids), :, xp.asarray(indices), :] = blocks
        total_energy = float(xp.sum(blocks*blocks))
        self.low_rank = []
        retained = []
        source_lookup = {v.tobytes():i for i,v in enumerate(source_vertices)}
        self_columns = np.array([source_lookup.get(v.tobytes(),-1) for v in receiver_vertices])
        stats = {'svd_attempts':0, 'low_rank_blocks':0, 'raw_leaf_blocks':0,
                 'zero_blocks':0, 'svd_seconds':0., 'ranks':[],
                 'rtol':rtol, 'leaf_size':leaf_size, 'admissibility_eta':eta,
                 'max_svd_entries':max_svd_entries, 'admissibility':admissibility,
                 'error_budget':error_budget, 'decomposition_backend':decomposition if xp is not np else 'cpu',
                 'zero_rank_blocks':0, 'dropped_pairs':0}

        def cluster(vertices, ids):
            selected = vertices[ids]
            return (ids, selected.min(axis=(0, 1)), selected.max(axis=(0, 1)))

        def split(vertices, node):
            ids, lower, upper = node
            axis = int(np.argmax(upper-lower))
            order = np.argsort(vertices[ids].mean(axis=1)[:, axis], kind='stable')
            middle = len(ids)//2
            return [cluster(vertices, ids[order[:middle]]), cluster(vertices, ids[order[middle:]])]

        def descend(rnode, snode, active, pending):
            ri, rlo, rhi = rnode
            si, slo, shi = snode
            if len(ri)<=leaf_size and len(si)<=leaf_size:
                retained.append(active)
                stats['raw_leaf_blocks'] += 1
            elif len(ri)>leaf_size and (len(si)<=leaf_size or np.linalg.norm(rhi-rlo)>=np.linalg.norm(shi-slo)):
                pending.extend((child,snode) for child in split(receiver_vertices,rnode))
            else:
                pending.extend((rnode,child) for child in split(source_vertices,snode))

        def classify(rnode, snode, pending, candidates):
            ri, rlo, rhi = rnode
            si, slo, shi = snode
            pairs = pairmap[np.ix_(ri, si)]
            active = pairs[pairs>=0]
            if not len(active):
                stats['zero_blocks'] += 1
                return
            distance = float(np.linalg.norm(np.maximum(0., np.maximum(rlo-shi, slo-rhi))))
            separated = max(np.linalg.norm(rhi-rlo), np.linalg.norm(shi-slo)) <= eta*distance
            if admissibility=='weak':
                separated = separated or not np.intersect1d(self_columns[ri],si).size
            rdofs = (3*ri[:, None]+np.arange(3)).ravel().astype(np.int32)
            sdofs = (3*si[:, None]+np.arange(3)).ravel().astype(np.int32)
            if separated and len(rdofs)*len(sdofs) <= max_svd_entries:
                candidates.setdefault((len(rdofs),len(sdofs)),[]).append(
                    (rnode,snode,active,rdofs,sdofs))
            else:
                descend(rnode,snode,active,pending)

        pending = [(cluster(receiver_vertices,np.arange(nr,dtype=np.int32)),
                    cluster(source_vertices,np.arange(ns,dtype=np.int32)))]
        while pending:
            current, pending, candidates = pending, [], {}
            for rnode,snode in current:
                classify(rnode,snode,pending,candidates)
            for (nrows,ncols),group in candidates.items():
                # Small blocks use batched Jacobi SVD; CuPy loops larger gesvd
                # calls without per-matrix host synchronization. Bound batches.
                batch_size = max(1,min(256,16*1024**2//(8*nrows*ncols)))
                for offset in range(0,len(group),batch_size):
                    batch = group[offset:offset+batch_size]
                    dr = xp.asarray(np.stack([item[3] for item in batch]))
                    ds = xp.asarray(np.stack([item[4] for item in batch]))
                    matrices = dense[dr[:,:,None],ds[:,None,:]]
                    transposed = nrows<ncols
                    if xp is not np:
                        xp.cuda.Stream.null.synchronize()
                    svd_start = time.perf_counter()
                    matrix_input = matrices.swapaxes(-1,-2) if transposed else matrices
                    if decomposition=='cpu' and xp is not np:
                        matrix_input = xp.asnumpy(matrix_input)
                    engine = np if decomposition=='cpu' else xp
                    u,s,vh = engine.linalg.svd(matrix_input,full_matrices=False)
                    singular = np.asarray(s) if engine is np else xp.asnumpy(s)
                    stats['svd_seconds'] += time.perf_counter()-svd_start
                    stats['svd_attempts'] += len(batch)
                    for j,(rnode,snode,active,rdofs,sdofs) in enumerate(batch):
                        tail = np.r_[np.cumsum((singular[j]**2)[::-1])[::-1],0.]
                        budget = tail[0] if error_budget=='relative' else total_energy*len(active)/len(indices)
                        rank = int(np.flatnonzero(tail<=rtol**2*budget)[0])
                        factor_bytes = (nrows+ncols)*(8*rank+4)+64
                        if xp is not np:
                            factor_bytes += 4*nrows+12*rank
                        if rank==0:
                            stats['low_rank_blocks'] += 1
                            stats['zero_rank_blocks'] += 1
                            stats['dropped_pairs'] += len(active)
                            stats['ranks'].append(0)
                            continue
                        # Compare actual nonzero FP64 BSR, not dense padding.
                        if factor_bytes<.9*len(active)*(9*8+4):
                            if transposed:
                                left,right = vh[j,:rank].T*s[j,:rank],u[j,:,:rank].T
                            else:
                                left,right = u[j,:,:rank]*s[j,:rank],vh[j,:rank]
                            self.low_rank.append((dr[j].copy(),ds[j].copy(),xp.asarray(left).copy(),xp.asarray(right).copy()))
                            stats['low_rank_blocks'] += 1
                            stats['ranks'].append(rank)
                        else:
                            descend(rnode,snode,active,pending)
        kept = np.sort(np.concatenate(retained)) if retained else np.empty(0, dtype=np.int64)
        self.raw_blocks = blocks[xp.asarray(kept)].copy()
        self.raw_indices = xp.asarray(indices[kept])
        self.raw_pointer = xp.asarray(np.r_[0, np.cumsum(np.bincount(rowids[kept], minlength=nr))].astype(np.int64))
        self.nbytes = self.raw_blocks.nbytes+self.raw_indices.nbytes+self.raw_pointer.nbytes
        self.nbytes += sum(sum(a.nbytes for a in part) for part in self.low_rank)
        # Serialized block directory: offsets/counts for the four numeric arrays.
        self.directory_bytes = 8*8*len(self.low_rank)
        self.nbytes += self.directory_bytes
        if xp is not np and self.low_rank:
            self._pack_gpu()
        original_bytes = blocks.nbytes+indices.nbytes+pointer.nbytes
        if xp is not np:
            xp.cuda.Stream.null.synchronize()
        self.info = {**stats, 'original_pairs':len(indices), 'retained_raw_pairs':len(kept),
                     'original_BSR_bytes':original_bytes, 'compressed_bytes':self.nbytes,
                     'compression_ratio':original_bytes/self.nbytes,
                     'build_seconds':time.perf_counter()-started,
                     'storage_scope':'numeric data, device indices/pointers, 64-byte directory per nonzero low-rank block, packed-kernel maps and intermediate workspace; Python objects and allocator reserve excluded'}

    def _pack_gpu(self):
        """Store factors contiguously and use two kernels instead of many GEMVs."""
        cp = self.xp
        metadata, rank_blocks, row_blocks = [], [], []
        row0=col0=z0=left0=right0=0
        for block,(rows,columns,left,right) in enumerate(self.low_rank):
            nr,nc,rank = len(rows),len(columns),left.shape[1]
            metadata.append([row0,col0,z0,left0,right0,nr,nc,rank])
            rank_blocks.extend([block]*rank)
            row_blocks.extend([block]*nr)
            row0+=nr;col0+=nc;z0+=rank;left0+=nr*rank;right0+=rank*nc
        self.packed_rows = cp.concatenate([part[0] for part in self.low_rank])
        self.packed_columns = cp.concatenate([part[1] for part in self.low_rank])
        self.packed_left = cp.concatenate([part[2].ravel() for part in self.low_rank])
        self.packed_right = cp.concatenate([part[3].ravel() for part in self.low_rank])
        self.packed_metadata = cp.asarray(metadata,dtype=cp.int64)
        self.rank_blocks = cp.asarray(rank_blocks,dtype=cp.int32)
        self.row_blocks = cp.asarray(row_blocks,dtype=cp.int32)
        self.intermediate = cp.empty(z0,dtype=cp.float64)
        self.nbytes += self.rank_blocks.nbytes+self.row_blocks.nbytes+self.intermediate.nbytes
        # Keep lightweight views for audit reconstruction, without duplicate data.
        self.low_rank = [(self.packed_rows[r:r+nr],self.packed_columns[c:c+nc],
                          self.packed_left[l:l+nr*k].reshape(nr,k),
                          self.packed_right[v:v+k*nc].reshape(k,nc))
                         for r,c,z,l,v,nr,nc,k in metadata]
        from tetra_cuda import SOURCE,_module
        self.compressed_module = _module(int(cp.cuda.Device().id),hashlib.sha256(SOURCE.read_bytes()).hexdigest())

    def __matmul__(self, vector):
        xp = self.xp
        vector = xp.asarray(vector, dtype=xp.float64).ravel()
        if len(vector) != self.shape[1]:
            raise ValueError('Near-compression vector shape mismatch')
        result = _apply_near_blocks(self.raw_blocks, self.raw_indices, self.raw_pointer,
                                    vector, self.shape[1]//3, xp)
        if xp is not np and self.low_rank:
            count = len(self.intermediate)
            self.compressed_module.get_function('fmm_compressed_right')(((count+3)//4,),(128,),
                (self.packed_right,self.packed_columns,self.packed_metadata,self.rank_blocks,
                 vector,np.int32(count),self.intermediate))
            count = len(self.packed_rows)
            self.compressed_module.get_function('fmm_compressed_left')(((count+3)//4,),(128,),
                (self.packed_left,self.packed_rows,self.packed_metadata,self.row_blocks,
                 self.intermediate,np.int32(count),result))
            return result
        for rows, columns, left, right in self.low_rank:
            result[rows] += left @ (right @ vector[columns])
        return result

    def toarray(self):
        """Explicit bounded-patch reconstruction for independent error auditing."""
        xp = self.xp
        result = xp.zeros(self.shape, dtype=xp.float64)
        pointer = self.raw_pointer if xp is np else xp.asnumpy(self.raw_pointer)
        rows = np.repeat(np.arange(self.shape[0]//3), np.diff(pointer))
        result.reshape(self.shape[0]//3, 3, self.shape[1]//3, 3)[
            xp.asarray(rows), :, self.raw_indices, :] = self.raw_blocks
        for rows, columns, left, right in self.low_rank:
            result[rows[:, None], columns] += left @ right
        return result


class VolumeFMM:
    """Array-like matvec interface for the existing Newton/proximal solvers.

    Persistent correction storage is O(P), P = number of corrected pairs, not
    automatically O(N) for overlapping, very graded or degenerate geometries.
    Above cache_bytes, near blocks are evaluated in bounded batches per matvec.
    """
    matrix_free = True

    def __init__(self, tiles, xp, *, dtype='float64', eps=1e-10,
                 near_levels=2, near_factor=1.5, correction_factor=3.,
                 source_levels=0, batch=32, cache_bytes=512*1024**2,
                 index_bytes=4*1024**3, verify_rtol=1e-4,neighbours='auto',
                 farfield='cpu',gpu_order=None,gpu_bytes=12*1024**3,gpu_leaf_size=512):
        start = time.perf_counter()
        # No fallback: a missing/broken dependency must fail before geometry work.
        if farfield not in ('cpu','cuda') or (farfield=='cuda' and xp is np):
            raise ValueError('FMM farfield must be cpu/cuda; cuda requires --backend cupy')
        self.farfield = farfield
        self._lfmm = None
        if farfield=='cpu':
            try:
                from fmm3dpy import lfmm3d
            except ImportError as error:
                raise RuntimeError('CPU FMM requires fmm3dpy>=2.1; install it in the active Python environment') from error
            self._lfmm = lfmm3d
        self.xp, self.tiles = xp, tiles
        self.dtype = np.dtype(dtype)
        self.n = n = int(tiles.n)
        self.shape = (3*n, 3*n)
        if (n < 1 or n > np.iinfo(np.int32).max or
            self.dtype not in (np.dtype('float32'), np.dtype('float64')) or
            not np.isfinite(eps) or not 0 < eps < 1 or
            not np.isfinite(near_factor) or near_factor <= 0 or
            not np.isfinite(correction_factor) or correction_factor < near_factor or
            near_levels not in range(4) or source_levels not in range(3) or
            batch < 1 or cache_bytes < 0 or index_bytes < 1 or
            not np.isfinite(verify_rtol) or not 0 < verify_rtol < 1):
            raise ValueError('Invalid FMM geometry, dtype, accuracy, radius, batch or memory settings')
        if neighbours not in ('auto','cpu','cuda') or (neighbours=='cuda' and xp is np):
            raise ValueError('FMM neighbours must be auto/cpu/cuda; cuda requires --backend cupy')
        if (not np.all(np.asarray(tiles.tile_type) == 5) or
            np.any(tiles.offset) or np.any(tiles.rot)):
            raise ValueError('FMM requires global-coordinate, unrotated TET4 tiles')
        self.vertices = v = np.asarray(tiles.vertices, dtype=float).transpose(0, 2, 1)
        if v.shape != (n, 4, 3) or not np.isfinite(v).all():
            raise ValueError('FMM requires finite TET4 vertices')
        volumes = abs(np.linalg.det(v[:, 1:]-v[:, :1]))/6
        if not np.isfinite(volumes).all() or np.any(volumes <= 0):
            raise ValueError('FMM requires nondegenerate tetrahedra')
        self.centres = v.mean(axis=1)
        self.radii = np.linalg.norm(v-self.centres[:, None], axis=2).max(axis=1)
        self.near_levels, self.near_factor = near_levels, near_factor
        self.correction_factor = correction_factor
        # Air targets have no receiving-volume radius. Use a wider direct
        # neighbourhood to resolve strong fields close to material interfaces.
        self.air_correction_factor = 4*correction_factor
        self.batch, self.eps, self.verify_rtol = batch, eps, verify_rtol
        self.source_levels = source_levels
        if xp is not np:
            # Reject a source rule that cannot fit before allocating/refining
            # its potentially multi-GB host quadrature arrays.
            free,_ = xp.cuda.Device().mem_info
            source_bytes = n*(4*8**source_levels)*3*8
            required = n*(4*24*8+12*8+9*self.dtype.itemsize+8)+source_bytes
            if source_levels:
                required += n*4*3*8
            if required > free*.8:
                raise MemoryError(f'FMM CUDA geometry needs at least {required/1e9:.2f} GB; '
                                  'reduce --fmm-source-levels or use --backend numpy')
        self.source_q = quadrature(v, source_levels)
        self.target_q = self.source_q if source_levels == 0 else quadrature(v)
        self.nq = self.source_q.shape[1]
        self.weights = volumes/self.nq
        self.origin = v.min(axis=(0, 1))
        self.scale = float(np.ptp(v.reshape(-1, 3), axis=0).max())
        self.sources = np.asfortranarray(((self.source_q-self.origin)/self.scale).reshape(-1, 3).T)
        self.targets = None if source_levels == 0 else np.asfortranarray(
            ((self.target_q-self.origin)/self.scale).reshape(-1, 3).T)
        # Only compact CSR indices are kept on the host. Never materialize a
        # global N x 256 x 3 array of receiving quadrature coordinates.
        pointer = np.zeros(n+1, dtype=np.int64)
        if pointer.nbytes > index_bytes:
            raise MemoryError('FMM row pointers exceed --fmm-index-gib')
        chunks = []
        self.stream_indices = False
        use_cuda = xp is not np and neighbours!='cpu'
        if use_cuda:
            try:
                from cupyx.scipy.spatial import KDTree
            except ImportError:
                if neighbours=='cuda':
                    raise RuntimeError('GPU neighbour enumeration requires a CuPy version with KDTree support')
                use_cuda = False
        self.groups = (_CUDANeighbours(self.centres,self.radii) if use_cuda
                       else _source_trees(self.centres,self.radii))
        if use_cuda:
            counts = self.groups.counts(correction_factor)
            np.cumsum(counts,out=pointer[1:])
            self.stream_indices = pointer[-1]*4+pointer.nbytes>index_bytes
            if not self.stream_indices:
                chunks = [np.concatenate(rows) for _,rows in _neighbours(
                    self.centres,self.radii,correction_factor,batch,groups=self.groups)]
        else:
            for lo, rows in _neighbours(self.centres, self.radii, correction_factor, batch, groups=self.groups):
                counts = np.array([len(row) for row in rows], dtype=np.int64)
                pointer[lo+1:lo+1+len(rows)] = pointer[lo] + np.cumsum(counts)
                if pointer[lo+len(rows)]*4 + pointer.nbytes > index_bytes:
                    self.stream_indices = True
                    chunks.clear()
                if not self.stream_indices:
                    chunks.append(np.concatenate(rows))
        self.indptr = pointer
        self.indices = None if self.stream_indices else np.concatenate(chunks)
        del chunks
        pairs = int(pointer[-1])
        self.diag = xp.empty((n, 3, 3), dtype=self.dtype)
        self.blocks = None
        self.cuda = xp is not np
        if self.cuda:
            from tetra_cuda import SOURCE, _module
            free, _ = xp.cuda.Device().mem_info
            required = n*4*24*8 + self.source_q.nbytes + self.diag.nbytes
            if source_levels:
                required += self.target_q.nbytes
            if required > free*.8:
                raise MemoryError(f'FMM CUDA geometry needs at least {required/1e9:.2f} GB; '
                                  'reduce --fmm-source-levels or use --backend numpy')
            self.module = _module(int(xp.cuda.Device().id), hashlib.sha256(SOURCE.read_bytes()).hexdigest())
            self.gpu_v = xp.asarray(np.ascontiguousarray(v))
            self.faces = xp.empty((n, 4, 24), dtype=xp.float64)
            self.module.get_function('prepare_faces')(((n+127)//128,), (128,),
                (self.gpu_v, self.faces, np.int32(n)))
            self.gpu_source_q = xp.asarray(np.ascontiguousarray(self.source_q))
            self.gpu_target_q = self.gpu_source_q if source_levels == 0 else xp.asarray(self.target_q)
            self.gpu_weights = xp.asarray(self.weights)
            self.gpu_centres = (self.groups.centres if isinstance(self.groups,_CUDANeighbours)
                                else xp.ascontiguousarray(xp.asarray(self.centres)))
            self.gpu_radii = (self.groups.radii if isinstance(self.groups,_CUDANeighbours)
                              else xp.asarray(self.radii))
        block_bytes = pairs*9*self.dtype.itemsize
        if self.cuda and not self.stream_indices:
            free, _ = xp.cuda.Device().mem_info
            cache_bytes = min(cache_bytes, max(0, int(free*.65)-self.indices.nbytes-pointer.nbytes))
        cache = not self.stream_indices and block_bytes <= cache_bytes
        if cache:
            self.blocks = xp.empty((pairs, 3, 3), dtype=self.dtype)
            for lo in range(0, n, batch):
                hi = min(n, lo+batch)
                self.blocks[pointer[lo]:pointer[hi]] = self._blocks(lo, hi)
            if self.cuda:
                self.device_indices = xp.asarray(self.indices)
                self.device_indptr = xp.asarray(pointer)
            else:
                self.sparse = bsr_matrix((self.blocks, self.indices, pointer), shape=self.shape)
        else:
            # Exact self blocks are required by the existing block preconditioner.
            for lo in range(0, n, batch):
                self._blocks(lo, min(n, lo+batch), self_only=True)
        if self.cuda:
            xp.cuda.Stream.null.synchronize()
        if not bool(xp.all(xp.isfinite(self.diag))):
            raise FloatingPointError('Nonfinite FMM self tensor')
        if self.blocks is not None and not bool(xp.all(xp.isfinite(self.blocks))):
            raise FloatingPointError('Nonfinite FMM near correction')
        self.gpu_fmm = None
        self._verification_gpu_plan = None
        if farfield=='cuda':
            self.gpu_fmm = _CUDAPointFMM(self.sources.T,None if self.targets is None else self.targets.T,
                eps=eps,order=gpu_order,budget_bytes=gpu_bytes,leaf_size=gpu_leaf_size)
        self.info = {'operator':'tetra_volume_fmm', 'K_builder':'fmm', 'tiles':n,
            'dtype':self.dtype.name, 'integration_dtype':'float64', 'loaded_from_cache':False,
            'K_GB':0., 'dense_K_allocated':False,
            'fmm_backend':'CUDA interpolation FMM' if self.gpu_fmm is not None else 'fmm3dpy CPU',
            'fmm_eps':eps, 'fmm_source_levels':source_levels,
            'fmm_source_points':n*self.nq, 'fmm_target_points':4*n,
            'fmm_correction_factor':correction_factor, 'near_factor':near_factor,
            'fmm_air_correction_factor':self.air_correction_factor,
            'quadrature_points_far':4, 'quadrature_points_near':4*8**near_levels,
            'fmm_near_pairs':pairs, 'fmm_near_storage':'cached' if cache else 'streamed',
            'fmm_near_apply':'cuda_fused_vectors' if self.cuda and not cache else 'tensor_blocks',
            'fmm_near_block_bytes':block_bytes if cache else 0,
            'fmm_index_bytes':(0 if self.stream_indices else self.indices.nbytes)+pointer.nbytes,
            'fmm_index_storage':'streamed' if self.stream_indices else 'cached',
            'fmm_neighbour_backend':'cuda_kdtree_csr' if use_cuda else 'scipy_cpu',
            'fmm_verify_rtol':verify_rtol, 'fmm_matvec_calls':0, 'fmm_matvec_seconds':0.,
            'fmm_tree_reused':self.gpu_fmm is not None,
            'fmm_air_field_calls':0,
            'approximation':'Volume quadrature dipoles; analytical correction; eps excludes quadrature error',
            'K_build_or_load_seconds':time.perf_counter()-start}
        if self.gpu_fmm is not None:
            self.info.update(self.gpu_fmm.info)

    def _blocks(self, lo, hi, self_only=False, rows=None):
        xp = self.xp
        if self_only:
            receivers = np.arange(lo, hi, dtype=np.int32)
            sources = receivers
        else:
            counts = np.diff(self.indptr[lo:hi+1]) if rows is None else [len(row) for row in rows]
            receivers = np.repeat(np.arange(lo, hi, dtype=np.int32), counts)
            sources = (self.indices[self.indptr[lo]:self.indptr[hi]] if rows is None
                       else np.concatenate(rows))
        # A receiver batch bounds all high-order quadrature temporaries.
        high = quadrature(self.vertices[lo:hi], self.near_levels)
        if self.cuda:
            result = xp.empty((len(sources), 3, 3), dtype=self.dtype)
            high_gpu = xp.asarray(high)
            suffix = 'f64' if self.dtype.itemsize == 8 else 'f32'
            kernel = self.module.get_function('fmm_blocks_'+suffix)
            # Limit pair temporaries and Windows display-GPU kernel duration.
            for offset in range(0, len(sources), 4096):
                stop = min(len(sources), offset+4096)
                rr, ss = receivers[offset:stop], sources[offset:stop]
                use_high = (np.linalg.norm(self.centres[rr]-self.centres[ss], axis=1)
                            <= self.near_factor*(self.radii[rr]+self.radii[ss])).astype(np.int32)
                kernel(((stop-offset+3)//4,), (128,),
                    (self.faces, self.gpu_source_q, self.gpu_target_q, self.gpu_weights,
                     high_gpu, xp.asarray(rr), xp.asarray(ss), xp.asarray(use_high),
                     np.int32(lo), np.int32(self.nq), np.int32(high.shape[1]),
                     np.int32(stop-offset), result[offset:], self.diag))
            return result
        from magtense.magstatics import get_demag_tensor
        result = np.empty((len(sources), 3, 3), dtype=self.dtype)
        for receiver in range(lo, hi):
            positions = np.flatnonzero(receivers == receiver)
            for offset in range(0, len(positions), 128):
                select = positions[offset:offset+128]
                ids = sources[select]
                is_near = (np.linalg.norm(self.centres[ids]-self.centres[receiver], axis=1)
                           <= self.near_factor*(self.radii[ids]+self.radii[receiver]))
                exact = np.empty((len(ids), 3, 3))
                for mask, q in ((is_near, high[receiver-lo]), (~is_near, self.target_q[receiver])):
                    if mask.any():
                        exact[mask] = get_demag_tensor(subset(self.tiles, ids[mask]), q).mean(axis=1)
                own = ids == receiver
                if own.any():
                    self.diag[receiver] = exact[own][0]
                result[select] = exact - _dipole_blocks(self.target_q[receiver],
                                                       self.source_q[ids], self.weights[ids])
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite FMM near correction')
        return result

    def _apply_blocks(self, blocks, indices, pointer, vector):
        return _apply_near_blocks(blocks, indices, pointer, vector, self.n, self.xp)

    def _correct_vectors(self,lo,hi,magnetization,charges,rows=None,selected=None):
        """Stream H directly from face charges; avoid nine tensor components per pair."""
        cp = self.xp
        if rows is None:
            pointer = self.indptr[lo:hi+1]-self.indptr[lo]
            indices = self.indices[self.indptr[lo]:self.indptr[hi]]
            counts = np.diff(pointer)
        else:
            counts = np.array([len(row) for row in rows],dtype=np.int64)
            pointer = np.r_[np.int64(0),np.cumsum(counts)]
            indices = np.concatenate(rows)
        receiving = np.arange(lo,hi,dtype=np.int32) if selected is None else np.asarray(selected,dtype=np.int32)
        receivers = np.repeat(receiving,counts)
        high_rows = cp.asarray(np.repeat(np.arange(hi-lo,dtype=np.int32),counts))
        high = cp.asarray(quadrature(self.vertices[receiving],self.near_levels))
        values = cp.empty((len(indices),3),dtype=cp.float64)
        device_indices,device_receivers = cp.asarray(indices),cp.asarray(receivers)
        kernel = self.module.get_function('fmm_correct_vectors')
        for first in range(0,len(indices),4096):
            stop = min(len(indices),first+4096)
            kernel(((stop-first+3)//4,),(128,),
                (self.faces,charges,magnetization,self.gpu_source_q,self.gpu_target_q,high,
                 self.gpu_centres,self.gpu_radii,self.gpu_weights,device_receivers[first:],
                 device_indices[first:],high_rows[first:],np.int32(self.nq),np.int32(high.shape[1]),
                 np.int32(stop-first),np.float64(self.near_factor),values[first:]))
        result = cp.empty((hi-lo,3),dtype=cp.float64)
        self.module.get_function('fmm_sum_vectors')(((hi-lo+3)//4,),(128,),
            (values,cp.asarray(pointer),np.int32(hi-lo),result))
        return result.ravel()

    def __matmul__(self, vector):
        start = time.perf_counter()
        xp = self.xp
        vector = xp.asarray(vector,dtype=self.dtype)
        if vector.shape != (3*self.n,):
            raise ValueError('FMM matvec expects a flat 3*N vector')
        if not bool(xp.all(xp.isfinite(vector))):
            raise FloatingPointError('Nonfinite FMM magnetization')
        if self.gpu_fmm is not None:
            moments = vector.reshape(self.n,3).astype(xp.float64)*(self.gpu_weights/self.scale**3)[:,None]
            field = self.gpu_fmm(xp.repeat(moments,self.nq,axis=0)).reshape(self.n,4,3).mean(axis=1)
            field = field.astype(self.dtype).ravel()
            self.info.update(self.gpu_fmm.info)
        else:
            host = np.asarray(vector if xp is np else xp.asnumpy(vector), dtype=np.float64).reshape(self.n,3)
            moments = host * (self.weights/self.scale**3)[:, None]
            dipvec = np.asfortranarray(np.repeat(moments,self.nq,axis=0).T)
            arguments = dict(eps=self.eps,sources=self.sources,dipvec=dipvec)
            if self.targets is None:
                output = self._lfmm(**arguments,pg=2)
                gradient = output.grad
            else:
                output = self._lfmm(**arguments,targets=self.targets,pgt=2)
                gradient = output.gradtarg
            if output.ier!=0:
                raise RuntimeError(f'FMM3D failed with ier={output.ier}')
            if gradient is None or not np.isfinite(gradient).all():
                raise FloatingPointError('FMM3D returned a nonfinite or missing gradient')
            field = xp.asarray(-gradient.T.reshape(self.n,4,3).mean(axis=1),dtype=self.dtype).ravel()
        if self.blocks is not None:
            correction = (self.sparse @ vector if not self.cuda else self._apply_blocks(
                self.blocks, self.device_indices, self.device_indptr, vector))
            field += correction
        elif self.cuda:
            magnetization = xp.ascontiguousarray(vector.reshape(self.n,3),dtype=xp.float64)
            charges = xp.empty((self.n,4),dtype=xp.float64)
            self.module.get_function('fmm_face_charges')(((self.n+127)//128,),(128,),
                (self.faces,magnetization,np.int32(self.n),charges))
            batches = (_neighbours(self.centres,self.radii,self.correction_factor,self.batch,groups=self.groups)
                       if self.stream_indices else ((lo,None) for lo in range(0,self.n,self.batch)))
            for lo,rows in batches:
                hi = lo+len(rows) if rows is not None else min(self.n,lo+self.batch)
                field[3*lo:3*hi] += self._correct_vectors(lo,hi,magnetization,charges,rows).astype(self.dtype)
        elif self.stream_indices:
            for lo,rows in _neighbours(self.centres,self.radii,self.correction_factor,self.batch,groups=self.groups):
                pointer = np.r_[np.int64(0),np.cumsum([len(row) for row in rows],dtype=np.int64)]
                correction = self._apply_blocks(self._blocks(lo,lo+len(rows),rows=rows),
                    np.concatenate(rows),pointer,vector)
                field[3*lo:3*(lo+len(rows))] += correction
        else:
            for lo in range(0, self.n, self.batch):
                hi = min(self.n, lo+self.batch)
                first, last = self.indptr[lo], self.indptr[hi]
                correction = self._apply_blocks(self._blocks(lo, hi), self.indices[first:last],
                    self.indptr[lo:hi+1]-first, vector)
                field[3*lo:3*hi] += correction
        if not bool(xp.all(xp.isfinite(field))):
            raise FloatingPointError('Nonfinite corrected FMM field')
        self.info['fmm_matvec_calls'] += 1
        self.info['fmm_matvec_seconds'] += time.perf_counter()-start
        return field

    def direct_samples(self, magnetization, selected):
        """Bounded-memory native reference, O(N * probes), no dense tensor."""
        from magtense.magstatics import get_H_field
        selected = np.asarray(selected, dtype=int)
        low = self.target_q[selected].reshape(-1, 3)
        reference = np.zeros((len(selected), 3))
        for lo in range(0, self.n, 256):
            ids = np.arange(lo, min(self.n, lo+256))
            part = subset(self.tiles, ids)
            part.M = magnetization[ids]
            reference += get_H_field(part, low).reshape(len(selected), 4, 3).mean(axis=1)
        rows = next(_neighbours(self.centres,self.radii,self.near_factor,len(selected),
            groups=self.groups,targets=self.centres[selected],target_radii=self.radii[selected]))[1]
        for j, receiver in enumerate(selected):
            ids = rows[j]
            high = quadrature(self.vertices[receiver:receiver+1], self.near_levels)[0]
            for lo in range(0, len(ids), 128):
                chunk = ids[lo:lo+128]
                part = subset(self.tiles, chunk)
                part.M = magnetization[chunk]
                reference[j] += (get_H_field(part, high).mean(axis=0)
                                 - get_H_field(part, self.target_q[receiver]).mean(axis=0))
        if not np.isfinite(reference).all():
            raise FloatingPointError('Nonfinite native FMM sample reference')
        return reference

    def verify(self, magnetization=None, probes=5, *, return_reference=False):
        """Sample original volume operator; eps alone never certifies its error."""
        xp = self.xp
        if magnetization is None:
            magnetization = np.random.default_rng(7143).normal(size=(self.n, 3))*20000
        selected = np.linspace(0, self.n-1, min(probes, self.n), dtype=int)
        reference = self.direct_samples(magnetization, selected)
        if self.gpu_fmm is None:
            predicted = (self @ xp.asarray(magnetization, dtype=self.dtype).ravel()).reshape(self.n,3)[selected]
        else:
            key = tuple(selected)
            if self._verification_gpu_plan is None or self._verification_gpu_plan[0]!=key:
                points = (self.target_q[selected].reshape(-1,3)-self.origin)/self.scale
                self._verification_gpu_plan = (key,self.gpu_fmm.subset(points))
            plan = self._verification_gpu_plan[1]
            input_m = xp.asarray(magnetization,dtype=self.dtype).ravel()
            m = xp.ascontiguousarray(input_m.reshape(self.n,3),dtype=xp.float64)
            moments = xp.repeat(m*(self.gpu_weights/self.scale**3)[:,None],self.nq,axis=0)
            predicted = plan(moments).reshape(len(selected),4,3).mean(axis=1).astype(self.dtype)
            if self.blocks is not None:
                starts,ends = self.indptr[selected],self.indptr[selected+1]
                pointer = np.r_[np.int64(0),np.cumsum(ends-starts)]
                indices = np.concatenate([self.indices[a:b] for a,b in zip(starts,ends)])
                blocks = xp.concatenate([self.blocks[a:b] for a,b in zip(starts,ends)])
                predicted += self._apply_blocks(blocks,indices,pointer,input_m).reshape(-1,3)
            else:
                rows = [row for _,batch in _neighbours(self.centres,self.radii,self.correction_factor,self.batch,
                        groups=self.groups,targets=self.centres[selected],target_radii=self.radii[selected]) for row in batch]
                charges = xp.empty((self.n,4),dtype=xp.float64)
                self.module.get_function('fmm_face_charges')(((self.n+127)//128,),(128,),
                    (self.faces,m,np.int32(self.n),charges))
                predicted += self._correct_vectors(0,len(selected),m,charges,rows,selected).reshape(-1,3).astype(self.dtype)
            self.info.update(fmm_verification_selected_receivers=len(selected),
                             fmm_verification_target_points=4*len(selected),
                             fmm_verification_uses_full_matvec=False,
                             fmm_verification_point_probe_error=plan.info['fmm_gpu_point_probe_error'])
        if self.cuda:
            predicted = xp.asnumpy(predicted)
        relative = float(np.linalg.norm(reference-predicted)/max(np.linalg.norm(reference), 1.))
        if not np.isfinite(relative) or relative > self.verify_rtol:
            raise RuntimeError(f'FMM/native sampled volume field error {relative:.3e} exceeds '
                f'{self.verify_rtol:g}; increase --fmm-source-levels or --fmm-correction-factor. '
                'Reducing --fmm-eps alone does not fix source quadrature error.')
        if return_reference:
            return relative, selected, reference
        return relative

    def field_at(self, magnetization, points):
        """FMM air probes with exact point-field correction, bounded native work.

        The caller must validate that points are in air (air_field.evaluate does
        this, including every offset used for removable-singularity limits).
        """
        from magtense.magstatics import get_H_field
        points = np.asarray(points, dtype=float)
        m = np.asarray(magnetization, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or m.shape != (self.n,3) or not np.isfinite(m).all()):
            raise ValueError('Invalid FMM air points or magnetization')
        if not len(points):
            return np.empty((0,3))
        if self.gpu_fmm is not None:
            return self._field_at_cuda(m,points)
        moments = m*(self.weights/self.scale**3)[:,None]
        output = self._lfmm(eps=self.eps,sources=self.sources,
            dipvec=np.asfortranarray(np.repeat(moments,self.nq,axis=0).T),
            targets=np.asfortranarray(((points-self.origin)/self.scale).T),pgt=2)
        if output.ier != 0:
            raise RuntimeError(f'FMM3D air field failed with ier={output.ier}')
        if output.gradtarg is None or not np.isfinite(output.gradtarg).all():
            raise FloatingPointError('Nonfinite FMM air gradient')
        result = -output.gradtarg.T.copy()
        for lo, rows in _neighbours(self.centres,self.radii,self.air_correction_factor,self.batch,
                                    groups=self.groups,targets=points,target_radii=np.zeros(len(points))):
            for j, ids in enumerate(rows):
                point = points[lo+j:lo+j+1]
                for first in range(0,len(ids),128):
                    selected = ids[first:first+128]
                    # Native triangle formulas can produce a finite, incorrect
                    # value on a face-plane extension as well as NaN. Force the
                    # existing two-direction limit check for these coordinates;
                    # a finite FMM far field must not mask a near-field pole.
                    vertices = self.vertices[selected]
                    coplanar = False
                    for face in ((1,2,3),(0,2,3),(0,1,3),(0,1,2)):
                        a,b,c = (vertices[:,index] for index in face)
                        normal = np.cross(b-a,c-a)
                        distance = abs(np.einsum('ij,ij->i',normal,point[0]-a))
                        tolerance = 64*np.finfo(float).eps*self.scale*np.linalg.norm(normal,axis=1)
                        if np.any(distance <= tolerance):
                            coplanar = True
                            break
                    if coplanar:
                        result[lo+j] = np.nan
                        break
                    part = subset(self.tiles,selected)
                    part.M = m[selected]
                    approximate = np.einsum('sab,sb->a',
                        _dipole_blocks(point,self.source_q[selected],self.weights[selected]),m[selected])
                    result[lo+j] += get_H_field(part,point)[0]-approximate
        self.info['fmm_air_field_calls'] += 1
        # Nonfinite analytical values are deliberately returned to the existing
        # air evaluator, which verifies symmetric limits before accepting them.
        return result

    def _field_at_cuda(self,magnetization,points):
        """Small air-probe sets: batched GPU dipoles plus analytical correction."""
        cp = self.xp
        m = cp.ascontiguousarray(cp.asarray(magnetization,dtype=cp.float64))
        device_points = cp.ascontiguousarray(cp.asarray(points,dtype=cp.float64))
        moments = m*(self.gpu_weights/self.scale**3)[:,None]
        result = self.gpu_fmm.direct(cp.repeat(moments,self.nq,axis=0),
                                     cp.ascontiguousarray((device_points-cp.asarray(self.origin))/self.scale))
        kernel = self.module.get_function('fmm_air_correction')
        for lo,rows in _neighbours(self.centres,self.radii,self.air_correction_factor,self.batch,
                                  groups=self.groups,targets=points,target_radii=np.zeros(len(points))):
            counts = np.array([len(row) for row in rows],dtype=np.int64)
            pointer = cp.asarray(np.r_[np.int64(0),np.cumsum(counts)])
            indices = cp.asarray(np.concatenate(rows))
            for offset in range(0,int(counts.max()),8192):
                kernel(((len(rows)+3)//4,),(128,),
                    (self.faces,device_points[lo:lo+len(rows)],m,self.gpu_source_q,self.gpu_weights,
                     indices,pointer,np.int32(len(rows)),np.int32(self.nq),np.int32(offset),
                     np.float64(64*np.finfo(float).eps*self.scale),result[lo:lo+len(rows)]))
        self.info.update(fmm_air_field_calls=self.info['fmm_air_field_calls']+1,
                         fmm_air_point_backend='cuda_direct_batched',fmm_air_correction_backend='cuda_analytic')
        return cp.asnumpy(result)

    def verify_air(self, magnetization, points, field, probes=5, *, validate_air=None,
                   force_limit_points=None):
        """Independent native air-field samples without N*len(points)*9 storage."""
        from magtense.magstatics import get_H_field
        # Check all requested points on affordable small problems; large jobs
        # keep bounded O(N*probes) verification instead of an O(N*Npoints) pass.
        count = len(points) if self.n*len(points) <= 2000000 else min(probes,len(points))
        selected = np.linspace(0,len(points)-1,count,dtype=int)
        self.info['fmm_air_native_reference_points'] = count
        force_limit_points = np.asarray([] if force_limit_points is None else force_limit_points).reshape(-1,3)
        def native(source,observations):
            result = np.zeros((len(observations),3))
            for lo in range(0,self.n,256):
                ids = np.arange(lo,min(self.n,lo+256))
                part = subset(self.tiles,ids)
                part.M = magnetization[ids]
                result += get_H_field(part,observations)
            if len(force_limit_points):
                forced = np.any(np.all(observations[:,None,:] == force_limit_points[None,:,:],axis=2),axis=1)
                result[forced] = np.nan
            return result
        if validate_air is None:
            reference = native(self.tiles,points[selected])
        else:
            from air_field import evaluate
            reference,_ = evaluate(self.tiles,points[selected],validate_air=validate_air,
                                  get_field=native,limit_batch=self.batch)
        if not np.isfinite(reference).all():
            raise FloatingPointError('Nonfinite native air reference; supply an air validator for limit evaluation')
        relative = float(np.linalg.norm(reference-field[selected])/max(np.linalg.norm(reference),1.))
        if not np.isfinite(relative) or relative > self.verify_rtol:
            raise RuntimeError(f'FMM/native sampled air field error {relative:.3e} exceeds {self.verify_rtol:g}')
        return relative


def assemble(tiles, xp, **kwargs):
    operator = VolumeFMM(tiles, xp, **kwargs)
    return operator, operator.diag, operator.info
