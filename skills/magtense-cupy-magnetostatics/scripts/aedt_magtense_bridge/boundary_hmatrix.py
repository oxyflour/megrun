"""Bounded CUDA H-matrix for the current P1/P0 FEM--BEM boundary equation.

Called by fem_bem.FEMBEM(boundary_backend='cuda_hmatrix'). This stores exact
near blocks and ACA far factors of [V,-K] on global P0 faces and P1 nodes,
never a full boundary matrix. Row averaging and block column equilibration
balance the two boundary kernels.
Geometry, source integration, quadrature and traces belong to that same FEMBEM
instance. ACA stopping is heuristic and independently sampled; direct global
operator probes must be checked before using the compressed coupled residual.
The block-Jacobi V inverse is a preconditioner, not an exact boundary inverse.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def panel_geometry(mesh):
    return np.column_stack((mesh.tri.reshape(-1,9),mesh.normals,
        mesh.trace_gradients.reshape(-1,9),mesh.edge_lengths,mesh.conormals.reshape(-1,9)))


def cuda_potential(mesh,trace,q,points,*,geometry=None):
    """Exact source-panel potential, bounded target batches; no H truncation."""
    import cupy as cp
    from tetra_cuda import SOURCE,_module
    module=_module(int(cp.cuda.Device().id),hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    geometry=cp.asarray(panel_geometry(mesh)) if geometry is None else geometry
    vector=cp.column_stack((cp.asarray(q),cp.asarray(trace)[cp.asarray(mesh.trace_faces)]))
    result=[]
    for first in range(0,len(points),32):
        target=cp.asarray(points[first:first+32]);out=cp.empty(len(target),dtype=cp.float64)
        module.get_function('bem_potential')((len(target),),(256,),
            (geometry,vector,np.int32(len(mesh.faces)),target,out))
        result.append(cp.asnumpy(out))
    return np.concatenate(result) if result else np.empty(0)


class _Clusters:
    def __init__(self,triangles,leaf_size):
        tri=np.asarray(triangles,dtype=float)
        self.permutation=np.arange(len(tri),dtype=np.int32)
        self.nodes=[];self.leaves=[]
        centers=tri.mean(axis=1)
        def cluster(start,end):
            ids=self.permutation[start:end]
            lo=tri[ids].min(axis=(0,1));hi=tri[ids].max(axis=(0,1))
            node=dict(start=start,end=end,lo=lo,hi=hi,diam=float(np.linalg.norm(hi-lo)),children=[])
            self.nodes.append(node);index=len(self.nodes)-1
            if end-start>leaf_size:
                axis=int(np.argmax(np.ptp(centers[ids],axis=0)))
                middle=(end-start)//2
                self.permutation[start:end]=ids[np.argpartition(centers[ids,axis],middle)]
                node['children']=[cluster(start,start+middle),cluster(start+middle,end)]
            else:
                self.leaves.append(index)
            return index
        self.root=cluster(0,len(tri))


class BoundaryTree:
    """Whole-support bounds for rectangular P0 tests / global P0+P1 trials."""
    def __init__(self,triangles,*,leaf_size=64,eta=2.,trial_triangles=None):
        if leaf_size<4 or not np.isfinite(eta) or eta<=0:
            raise ValueError('Positive H-matrix separation and leaf size >=4 required')
        row=_Clusters(triangles,leaf_size)
        column=row if trial_triangles is None else _Clusters(trial_triangles,leaf_size)
        self.nodes=row.nodes;self.column_nodes=column.nodes
        self.permutation=row.permutation;self.column_permutation=column.permutation
        self.leaves=row.leaves;self.root=row.root;self.eta=eta
        self.channels=4 if trial_triangles is None else 1
        self.blocks=[]
        def partition(a,b):
            x,y=self.nodes[a],self.column_nodes[b]
            gap=np.maximum(0,np.maximum(x['lo']-y['hi'],y['lo']-x['hi']))
            admissible=max(x['diam'],y['diam'])<=eta*np.linalg.norm(gap)
            if admissible or (not x['children'] and not y['children']):
                self.blocks.append((a,b,bool(admissible)))
            else:
                for aa,bb in self.split(a,b):
                    partition(aa,bb)
        partition(row.root,column.root)

    def split(self,a,b):
        x,y=self.nodes[a],self.column_nodes[b]
        if x['children'] and (not y['children'] or x['diam']>=y['diam']):
            return [(c,b) for c in x['children']]
        if y['children']:
            return [(a,c) for c in y['children']]
        return []

    def summary(self):
        dense=sum((self.nodes[a]['end']-self.nodes[a]['start'])*
                  (self.column_nodes[b]['end']-self.column_nodes[b]['start']) for a,b,far in self.blocks if not far)
        return dict(boundary_faces=len(self.permutation),clusters=len(self.nodes),
            trial_dofs=len(self.column_permutation),trial_clusters=len(self.column_nodes),
            leaf_clusters=len(self.leaves),terminal_blocks=len(self.blocks),
            admissible_blocks=sum(far for a,b,far in self.blocks),
            exact_near_entries=dense,exact_near_storage_bytes=8*self.channels*dense)


class BoundaryHMatrix:
    def __init__(self, mesh, *, quadrature_order=8, leaf_size=64, eta=2.,
                 rtol=1e-6, max_rank=48, budget_gib=6., progress=None,cache_path=None,
                 storage_dtype='float64',far_quadrature_order=None,
                 pivot_row_backend='auto',skip_inactive=True,_assemble=True):
        import cupy as cp
        from fem_bem import triangle_rule
        from tetra_cuda import SOURCE,_module
        if not 0<rtol<.1 or not 4<=max_rank<=128 or not np.isfinite(budget_gib) or budget_gib<=0:
            raise ValueError('Invalid H-matrix tolerance/rank/budget')
        self.storage_dtype=np.dtype(storage_dtype)
        if self.storage_dtype not in (np.dtype('float32'),np.dtype('float64')):
            raise ValueError('H coefficient storage must be float32 or float64; computations remain float64')
        far_quadrature_order=quadrature_order if far_quadrature_order is None else far_quadrature_order
        if not isinstance(far_quadrature_order,int) or not 2<=far_quadrature_order<=quadrature_order:
            raise ValueError('Far pivot quadrature must be between 2 and full target order')
        if pivot_row_backend not in ('direct','shared','auto') or (not _assemble and cache_path is not None):
            raise ValueError('Invalid pivot-row backend or cache on coefficient-only probe')
        started=time.perf_counter()
        self.cp=cp;self.mesh=mesh;self.rtol=rtol;self.max_rank=max_rank
        self.budget=int(budget_gib*2**30);self.progress=progress
        self.pivot_row_backend=pivot_row_backend
        self.skip_inactive=bool(skip_inactive)
        self.row_reuse_stats=dict(plans=0,plan_seconds=0.,pivot_calls=0,expanded_panel_slots=0,shared_panel_slots=0)
        self.module=_module(int(cp.cuda.Device().id),hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        self._cache_prefix=None
        if cache_path is not None:
            bary,weights=triangle_rule(quadrature_order)
            digest=hashlib.sha256(panel_geometry(mesh).tobytes())
            digest.update(mesh.trace_faces.tobytes());digest.update(mesh.areas.tobytes())
            digest.update(bary.tobytes());digest.update(weights.tobytes())
            self._cache_signature=dict(schema='global-P0-P1-H-3',geometry_sha256=digest.hexdigest(),
                cuda_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                leaf_size=leaf_size,eta=eta,rtol=rtol,max_rank=max_rank,quadrature_order=quadrature_order,
                storage_dtype=self.storage_dtype.name,far_quadrature_order=far_quadrature_order,
                pivot_row_backend=pivot_row_backend,skip_inactive=self.skip_inactive)
            key=hashlib.sha256(json.dumps(self._cache_signature,sort_keys=True).encode()).hexdigest()
            self._cache_prefix=Path(cache_path)/key
            if self._cache_prefix.with_suffix('.json').exists():
                self._load_cache()
                self.info.update(hmatrix_cache_hit=True,hmatrix_cache_load_seconds=time.perf_counter()-started)
                return
            if any(self._cache_prefix.with_suffix(s).exists() for s in ('.bin','.bin.tmp','.json.tmp')):
                raise FileExistsError('Incomplete boundary cache retained for inspection: '+str(self._cache_prefix))
        m=len(mesh.faces);nb=len(mesh.boundary_nodes)
        lo=np.full((nb,3),np.inf);hi=np.full((nb,3),-np.inf)
        for j in range(3):
            np.minimum.at(lo,mesh.trace_faces[:,j],mesh.tri.min(axis=1))
            np.maximum.at(hi,mesh.trace_faces[:,j],mesh.tri.max(axis=1))
        # Bounding proxies only: each nodal trial support contains its incident
        # triangles. Integrals always use the actual, unchanged source panels.
        trial=np.concatenate((mesh.tri,np.stack((lo,hi,(lo+hi)/2),axis=1)))
        self.tree=BoundaryTree(mesh.tri,leaf_size=leaf_size,eta=eta,trial_triangles=trial)
        self.info=dict(self.tree.summary(),boundary_backend='cuda_hmatrix',hmatrix_rtol=rtol,
            hmatrix_leaf_size=leaf_size,hmatrix_eta=eta,hmatrix_max_rank=max_rank,
            hmatrix_coefficient_storage_dtype=self.storage_dtype.name,hmatrix_compute_dtype='float64',
            hmatrix_far_pivot_quadrature_order=far_quadrature_order,
            hmatrix_reference_quadrature_order=quadrature_order,
            hmatrix_pivot_row_backend=pivot_row_backend,
            hmatrix_skip_inactive=self.skip_inactive,
            hmatrix_error_scope='ACA sampled block residual; independent global direct probes required')
        self.info['exact_near_storage_bytes']=self.info['exact_near_entries']*self.storage_dtype.itemsize
        if progress:
            progress('boundary_partition',**self.info)
        if self.info['exact_near_storage_bytes']>self.budget:
            raise MemoryError(f'Exact boundary near blocks require {self.info["exact_near_storage_bytes"]/2**30:.3f} GiB, '
                              f'exceeding {budget_gib} GiB before allocation')
        self.permutation=cp.asarray(self.tree.permutation)
        self.columns=cp.asarray(self.tree.column_permutation)
        self.reverse=cp.empty(m,dtype=cp.int32);self.reverse[self.permutation]=cp.arange(m,dtype=cp.int32)
        self.scale_cpu=np.r_[1/np.sqrt(mesh.areas),np.ones(nb)]
        self.scales=cp.asarray(self.scale_cpu)
        order=np.argsort(mesh.trace_faces.ravel(),kind='stable')
        self.node_pointer=cp.asarray(np.r_[0,np.cumsum(np.bincount(mesh.trace_faces.ravel(),minlength=nb))],dtype=cp.int32)
        self.node_triangles=cp.asarray(order//3,dtype=cp.int32)
        self.node_basis=cp.asarray(order%3,dtype=cp.int32)
        self.geometry=cp.asarray(panel_geometry(mesh));self.areas=cp.asarray(mesh.areas)
        bary,weights=triangle_rule(quadrature_order)
        self.bary=cp.asarray(bary);self.weights=cp.asarray(weights);self.nquad=len(weights)
        far_bary,far_weights=triangle_rule(far_quadrature_order)
        self.far_bary=cp.asarray(far_bary);self.far_weights=cp.asarray(far_weights);self.far_nquad=len(far_weights)
        self.descriptors=[];self.payload=[];inverse=[];inverse_meta=[]
        inv_offset=0;self.diagonal=np.empty(m)
        self.stored_bytes=0;self.ranks=[];self.sampled_errors=[];self.coefficient_pairs=0
        self._empty_int=cp.empty(0,dtype=cp.int32);self._empty_long=cp.empty(0,dtype=cp.int64)
        self.metadata=None
        if not _assemble:
            self.info['scope']='Coefficient/ACA probes only; no assembled boundary operator'
            return
        cp.cuda.Stream.null.synchronize()
        phases=dict(partition_and_geometry=time.perf_counter()-started)
        pending_count=len(self.tree.blocks)
        def audit():
            if self.stored_bytes>self.budget:
                raise MemoryError(f'Boundary H-matrix storage exceeded {budget_gib} GiB; no pairs discarded')
            if progress:
                progress('boundary_build',completed_blocks=len(self.descriptors),pending_blocks=pending_count,
                    stored_bytes=self.stored_bytes,coefficient_pairs=self.coefficient_pairs,
                    boundary_elapsed_seconds=time.perf_counter()-started)
        def descriptor(a,b):
            x,y=self.tree.nodes[a],self.tree.column_nodes[b]
            return [x['start'],x['end']-x['start'],y['start'],y['end']-y['start']]
        def retain_raw(tasks):
            nonlocal pending_count
            first=0
            while first<len(tasks):
                end=first;pair_count=0
                while end<len(tasks):
                    rs,nr,cs,nc=descriptor(*tasks[end]);size=nr*nc
                    if end>first and pair_count+size>131072:
                        break
                    pair_count+=size;end+=1
                chosen=tasks[first:end]
                descriptions=np.asarray([descriptor(*task) for task in chosen],dtype=np.int64)
                prefix=np.r_[0,np.cumsum(descriptions[:,1]*descriptions[:,3])]
                buffer=self._batch(descriptions,0,prefix=prefix).astype(self.storage_dtype,order='C',copy=False)
                self.payload.append(buffer);self.stored_bytes+=buffer.nbytes
                for k,(a,b) in enumerate(chosen):
                    rs,nr,cs,nc=map(int,descriptions[k]);position=int(prefix[k])
                    self.descriptors.append([rs,nr,cs,nc,-1,int(buffer.data.ptr)+self.storage_dtype.itemsize*position,0])
                pending_count-=len(chosen);first=end;audit()
        # The rectangular trial tree differs from the test tree. Build exact
        # physical V blocks on test leaves independently for block Jacobi.
        phase_started=time.perf_counter()
        diagonal_tasks=[]
        for a in self.tree.leaves:
            node=self.tree.nodes[a];start=node['start'];size=node['end']-start
            diagonal_tasks.append([start,size,start,size])
        for first in range(0,len(diagonal_tasks),64):
            descriptions=np.asarray(diagonal_tasks[first:first+64],dtype=np.int64)
            prefix=np.r_[0,np.cumsum(descriptions[:,1]**2)]
            buffer=self._batch(descriptions,4,prefix=prefix)
            for k,(rs,nr,cs,nc) in enumerate(descriptions):
                rows=self.tree.permutation[rs:rs+nr]
                local=cp.asnumpy(buffer[prefix[k]:prefix[k+1]].reshape(nr,nr))
                local*=mesh.areas[rows][:,None]/self.scale_cpu[rows][None,:]
                local=(local+local.T)/2
                self.diagonal[rows]=local.diagonal()
                inv=cho_solve(cho_factor(local,lower=True),np.eye(nr))
                inverse.append(cp.asarray(inv.ravel()));inverse_meta.append([rs,nr,inv_offset])
                inv_offset+=inv.size;self.stored_bytes+=inv.nbytes
        cp.cuda.Stream.null.synchronize()
        phases['leaf_preconditioner']=time.perf_counter()-phase_started
        phase_started=time.perf_counter()
        groups={};near=[]
        for a,b,far in self.tree.blocks:
            if far:
                rs,nr,cs,nc=descriptor(a,b)
                groups.setdefault((nr,nc),[]).append((a,b))
            else:
                near.append((a,b))
        retain_raw(near);near.clear()
        cp.cuda.Stream.null.synchronize()
        phases['exact_near_blocks']=time.perf_counter()-phase_started
        phase_started=time.perf_counter()
        while groups:
            shape=max(groups,key=lambda v:v[0]+v[1]);tasks=groups.pop(shape);nr,nc=shape
            rank_limit=min(max_rank,nr,nc)
            batch_size=max(1,min(64,(128*2**20)//max(8*rank_limit*(nr+nc),1)))
            for first in range(0,len(tasks),batch_size):
                chosen=tasks[first:first+batch_size]
                descriptions=np.asarray([descriptor(a,b) for a,b in chosen],dtype=np.int64)
                U,W,ranks,errors=self._aca_batch(descriptions,nr,nc)
                fallback=[]
                for k,((a,b),rank,error) in enumerate(zip(chosen,ranks,errors)):
                    rs,nr,cs,nc=map(int,descriptions[k]);r=int(rank)
                    if r:
                        # Copy rather than retain a view into padded batch workspace.
                        uu=U[k,:,:r].astype(self.storage_dtype,order='C',copy=True)
                        ww=W[k,:r,:].astype(self.storage_dtype,order='C',copy=True)
                        self.payload.extend((uu,ww))
                        self.descriptors.append([rs,nr,cs,nc,r,int(uu.data.ptr),int(ww.data.ptr)])
                        self.stored_bytes+=uu.nbytes+ww.nbytes
                        self.ranks.append(r);self.sampled_errors.append(float(error));pending_count-=1
                    else:
                        children=self.tree.split(a,b)
                        if children:
                            pending_count+=len(children)-1
                            for aa,bb in children:
                                rr,nn,cc,mm=descriptor(aa,bb)
                                groups.setdefault((nn,mm),[]).append((aa,bb))
                        else:
                            fallback.append((a,b))
                del U,W
                if fallback:
                    retain_raw(fallback)
                audit()
        cp.cuda.Stream.null.synchronize()
        phases['far_ACA_and_failed_block_fallback']=time.perf_counter()-phase_started
        phase_started=time.perf_counter()
        def pack(items):
            return cp.concatenate(items) if items else cp.empty(0,dtype=cp.float64)
        self.inverse=pack(inverse);inverse.clear()
        self.metadata=cp.asarray(self.descriptors,dtype=cp.int64)
        self.inverse_metadata=cp.asarray(inverse_meta,dtype=cp.int64)
        self.descriptors=None
        assert len(inverse_meta)==len(self.tree.leaves), 'Missing exact diagonal leaf inverse'
        assert np.all(self.diagonal>0) and np.isfinite(self.diagonal).all()
        cp.cuda.Stream.null.synchronize()
        phases['finalize']=time.perf_counter()-phase_started
        self.info.update(hmatrix_storage_bytes=sum(a.nbytes for a in self.payload)+sum(a.nbytes for a in (
            self.inverse,self.metadata,self.inverse_metadata,self.geometry,self.areas,self.permutation,self.columns,
            self.reverse,self.scales,self.node_pointer,self.node_triangles,self.node_basis)),
            hmatrix_trial_layout='global P0 flux faces + global P1 trace nodes; area-averaged tests and equilibrated kernel columns',
            hmatrix_payload_layout='persistent individual buffers; no concatenation copy of the H-matrix payload',
            hmatrix_low_rank_blocks=len(self.ranks),hmatrix_retained_dense_blocks=len(self.metadata)-len(self.ranks),
            hmatrix_max_retained_rank=max(self.ranks,default=0),
            hmatrix_max_heldout_relative_error=max(self.sampled_errors,default=0.),
            hmatrix_coefficient_entry_queries=self.coefficient_pairs,
            hmatrix_setup_seconds=time.perf_counter()-started,
            hmatrix_setup_phase_seconds=phases,
            hmatrix_cache_hit=False,
            hmatrix_row_reuse=dict(self.row_reuse_stats),
            boundary_inverse='exact diagonal-leaf block Jacobi; no global dense factorization')

    def save_cache(self):
        """Stream numeric operator data to ignored case output, without a host copy.

        This cache is not an accuracy certificate. The case entrypoint must run
        independent probes and the constant identity before any coupled solve.
        Retaining a rejected operator permits diagnosis without another cold
        build. Logical references replace process-local GPU addresses.
        """
        if self._cache_prefix is None or self.info.get('hmatrix_cache_hit'):
            return
        cp=self.cp;prefix=self._cache_prefix;prefix.parent.mkdir(parents=True,exist_ok=True)
        metadata_path=prefix.with_suffix('.json');binary_path=prefix.with_suffix('.bin')
        if metadata_path.exists() or binary_path.exists():
            raise FileExistsError('Existing boundary cache retained: '+str(prefix))
        original=cp.asnumpy(self.metadata)
        starts=np.asarray([a.data.ptr for a in self.payload],dtype=np.int64)
        sizes=np.asarray([a.nbytes for a in self.payload],dtype=np.int64)
        order=np.argsort(starts);sorted_starts=starts[order]
        logical=np.zeros((len(original),9),dtype=np.int64);logical[:,:5]=original[:,:5]
        for source_col,dest_col,mask in [(5,5,np.ones(len(original),dtype=bool)),(6,7,original[:,4]>0)]:
            pointer=original[mask,source_col]
            index=order[np.searchsorted(sorted_starts,pointer,side='right')-1]
            delta=pointer-starts[index]
            assert np.all(delta>=0) and np.all(delta<sizes[index])
            logical[mask,dest_col]=index;logical[mask,dest_col+1]=delta
        fixed={name:getattr(self,name) for name in ('geometry','areas','permutation','columns','reverse',
            'scales','node_pointer','node_triangles','node_basis','bary','weights','far_bary','far_weights','inverse','inverse_metadata')}
        fixed.update(diagonal=self.diagonal,metadata_logical=logical)
        header=dict(signature=self._cache_signature,info=self.info,stored_bytes=self.stored_bytes,
                    payload_sizes=[a.size for a in self.payload],payload_dtype=self.storage_dtype.str,
                    fixed={},fixed_order=list(fixed))
        temporary=prefix.with_suffix('.bin.tmp');hash_state=hashlib.sha256()
        def write_array(stream,array):
            host=cp.asnumpy(array) if isinstance(array,cp.ndarray) else np.ascontiguousarray(array)
            data=memoryview(host).cast('B');stream.write(data);hash_state.update(data)
        try:
            with temporary.open('xb') as stream:
                for array in self.payload:
                    write_array(stream,array.ravel())
                for name,array in fixed.items():
                    header['fixed'][name]=dict(dtype=array.dtype.str,shape=list(array.shape),
                        count=array.size,device=isinstance(array,cp.ndarray))
                    write_array(stream,array)
            header['binary_sha256']=hash_state.hexdigest()
            header['manifest_sha256']=hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest()
            os.replace(temporary,binary_path)
            meta_tmp=prefix.with_suffix('.json.tmp')
            meta_tmp.write_text(json.dumps(header,indent=2),encoding='utf-8');os.replace(meta_tmp,metadata_path)
            self.info['hmatrix_cache_prefix']=str(prefix)
        except BaseException:
            if temporary.exists():
                temporary.unlink()
            raise

    def _load_cache(self):
        cp=self.cp;prefix=self._cache_prefix
        header=json.loads(prefix.with_suffix('.json').read_text(encoding='utf-8'))
        manifest=header.pop('manifest_sha256')
        if hashlib.sha256(json.dumps(header,sort_keys=True).encode()).hexdigest()!=manifest:
            raise ValueError('Boundary-cache manifest hash mismatch')
        if header['signature']!=self._cache_signature:
            raise ValueError('Boundary cache source/geometry/settings signature mismatch')
        if header['stored_bytes']>self.budget:
            raise MemoryError('Cached boundary data exceeds requested storage budget')
        free=cp.cuda.Device().mem_info[0]+cp.get_default_memory_pool().free_bytes()
        if header['info']['hmatrix_storage_bytes']+256*2**20>free:
            raise MemoryError('Insufficient device memory for cached boundary operator')
        self.payload=[];hash_state=hashlib.sha256();logical=None
        def read_array(stream,dtype,count,shape):
            array=np.fromfile(stream,dtype=np.dtype(dtype),count=count)
            if array.size!=count:
                raise ValueError('Truncated boundary cache')
            hash_state.update(memoryview(array).cast('B'));return array.reshape(shape)
        with prefix.with_suffix('.bin').open('rb') as stream:
            for size in header['payload_sizes']:
                host=read_array(stream,header['payload_dtype'],size,(size,))
                self.payload.append(cp.asarray(host,blocking=True))
            for name in header['fixed_order']:
                spec=header['fixed'][name]
                host=read_array(stream,spec['dtype'],spec['count'],spec['shape'])
                if name=='metadata_logical':
                    logical=host
                else:
                    setattr(self,name,cp.asarray(host,blocking=True) if spec['device'] else host)
            if stream.read(1):
                raise ValueError('Unexpected trailing boundary-cache data')
        if hash_state.hexdigest()!=header['binary_sha256']:
            raise ValueError('Boundary-cache content hash mismatch')
        pointers=np.asarray([a.data.ptr for a in self.payload],dtype=np.int64)
        metadata=np.zeros((len(logical),7),dtype=np.int64);metadata[:,:5]=logical[:,:5]
        metadata[:,5]=pointers[logical[:,5]]+logical[:,6]
        far=metadata[:,4]>0
        metadata[far,6]=pointers[logical[far,7]]+logical[far,8]
        self.metadata=cp.asarray(metadata);self.diagonal=np.asarray(self.diagonal)
        self.nquad=len(self.weights);self.far_nquad=len(self.far_weights)
        self.stored_bytes=header['stored_bytes'];self.coefficient_pairs=0
        self.info=header['info'];self.info['hmatrix_cache_prefix']=str(prefix)

    def coefficients(self,rows,columns):
        cp=self.cp
        rows=cp.asarray(rows,dtype=cp.int32);columns=cp.asarray(columns,dtype=cp.int32)
        out=cp.empty((len(rows),len(columns)),dtype=cp.float64)
        count=len(rows)*len(columns);self.coefficient_pairs+=count
        self.module.get_function('bem_coefficients')(((count+127)//128,),(128,),
            (self.geometry,self.areas,self.node_pointer,self.node_triangles,self.node_basis,self.scales,
             np.int32(len(self.mesh.faces)),rows,columns,np.int32(len(rows)),np.int32(len(columns)),
             self.bary,self.weights,np.int32(self.nquad),out))
        return out*self.areas[rows][:,None]

    def _batch(self,descriptions,mode,*,nr=0,nc=0,pivots=None,checks=None,prefix=None,far=False,active=None):
        cp=self.cp;count=len(descriptions)
        descriptions=cp.asarray(descriptions,dtype=cp.int64)
        pr=pc=0
        if mode in (0,4):
            total=int(prefix[-1]);per=0;out=cp.empty(total,dtype=cp.float64)
        elif mode==1:
            per=nc;total=count*per;out=cp.empty((count,nc),dtype=cp.float64)
        elif mode==2:
            per=nr;total=count*per;out=cp.empty((count,nr),dtype=cp.float64)
        else:
            pr,pc=checks[0].shape[1],checks[1].shape[1]
            per=pr*pc;total=count*per;out=cp.empty((count,pr,pc),dtype=cp.float64)
        r,c=checks if checks is not None else (self._empty_int,self._empty_int)
        self.coefficient_pairs+=total
        bary,weights,nquad=(self.far_bary,self.far_weights,self.far_nquad) if far else (self.bary,self.weights,self.nquad)
        self.module.get_function('bem_batch_coefficients')(((total+127)//128,),(128,),
            (self.geometry,self.areas,self.node_pointer,self.node_triangles,self.node_basis,self.scales,
             np.int32(len(self.mesh.faces)),self.permutation,self.columns,descriptions,np.int32(count),np.int32(mode),
             self._empty_int if pivots is None else pivots,r,c,np.int32(pr),np.int32(pc),
             self._empty_long if prefix is None else cp.asarray(prefix,dtype=cp.int64),
             np.int64(per),np.int64(total),bary,weights,np.int32(nquad),
             self._empty_int if active is None else active,np.int32(active is not None),out))
        return out

    def _row_plan(self,descriptions,n,*,force=False):
        """Shared source-panel/incidence plan, reused for every ACA pivot row.

        This changes evaluation reuse only. Panel integrals and each nodal
        incidence sum retain the direct FP64 formula and accumulation order.
        """
        started=time.perf_counter();cp=self.cp;descriptions=np.asarray(descriptions,dtype=np.int64)
        count=len(descriptions)*n
        if not force and (n<16 or count<256):
            return None
        columns=np.concatenate([self.tree.column_permutation[int(a[2]):int(a[2]+a[3])] for a in descriptions])
        faces=len(self.mesh.faces);is_node=columns>=faces
        if not force and not is_node.any():
            return None
        if not hasattr(self,'_node_cpu'):
            self._node_cpu=tuple(cp.asnumpy(a) for a in (self.node_pointer,self.node_triangles,self.node_basis))
        node_pointer,node_triangles,node_basis=self._node_cpu
        degrees=np.ones(count,dtype=np.int64)
        nodes=columns[is_node]-faces
        degrees[is_node]=node_pointer[nodes+1]-node_pointer[nodes]
        pointer=np.r_[0,np.cumsum(degrees,dtype=np.int64)]
        query=np.repeat(np.arange(count,dtype=np.int64),degrees)
        local=np.arange(int(pointer[-1]),dtype=np.int64)-pointer[query]
        expanded_node=is_node[query]
        source=np.empty(len(query),dtype=np.int32);component=np.zeros(len(query),dtype=np.int32)
        source[~expanded_node]=columns[query[~expanded_node]]
        position=node_pointer[columns[query[expanded_node]]-faces]+local[expanded_node]
        source[expanded_node]=node_triangles[position]
        component[expanded_node]=node_basis[position]+1
        keys=(query//n)*faces+source
        unique,inverse=np.unique(keys,return_inverse=True)
        if not force and len(keys)<1.5*len(unique):
            return None
        single=np.zeros(len(unique),dtype=np.int32);single[inverse[component==0]]=1
        plan=dict(descriptions=cp.asarray(descriptions),columns=cp.asarray(columns,dtype=cp.int32),
            pointer=cp.asarray(pointer),panel_index=cp.asarray(inverse,dtype=cp.int32),
            component=cp.asarray(component),keys=cp.asarray(unique),single=cp.asarray(single),
            values=cp.empty((len(unique),4),dtype=cp.float64),
            coplanar=cp.empty(len(unique),dtype=cp.uint8),count=count,n=n,
            expanded_panels=len(keys),shared_panels=len(unique))
        self.row_reuse_stats['plans']+=1
        self.row_reuse_stats['plan_seconds']+=time.perf_counter()-started
        return plan

    def _pivot_rows(self,plan,pivots,active=None):
        cp=self.cp;size=plan['shared_panels'];count=plan['count']
        self.coefficient_pairs+=count
        self.row_reuse_stats['pivot_calls']+=1
        self.row_reuse_stats['expanded_panel_slots']+=plan['expanded_panels']
        self.row_reuse_stats['shared_panel_slots']+=size
        self.module.get_function('bem_pivot_panels')(((size+127)//128,),(128,),
            (self.geometry,self.areas,self.permutation,plan['descriptions'],pivots,
             plan['keys'],plan['single'],np.int32(len(self.mesh.faces)),np.int64(size),
             self.far_bary,self.far_weights,np.int32(self.far_nquad),
             self._empty_int if active is None else active,np.int32(active is not None),plan['values'],plan['coplanar']))
        out=cp.empty((count//plan['n'],plan['n']),dtype=cp.float64)
        self.module.get_function('bem_pivot_gather')(((count+127)//128,),(128,),
            (self.areas,self.scales,self.permutation,plan['descriptions'],pivots,plan['columns'],
             plan['pointer'],plan['panel_index'],plan['component'],plan['values'],plan['coplanar'],
             np.int32(plan['n']),np.int64(count),out))
        return out

    def _aca_batch(self,descriptions,m,n):
        cp=self.cp;count=len(descriptions);limit=min(self.max_rank,m,n)
        plan=(None if self.pivot_row_backend=='direct' else
              self._row_plan(descriptions,n,force=self.pivot_row_backend=='shared'))
        rr=[];cc=[];pr=min(m,12);pc=min(n,12)
        for rs,nr,cs,nc in descriptions:
            rng=np.random.default_rng(319+int(self.tree.permutation[rs])+17*int(self.tree.column_permutation[cs]))
            rr.append(rng.choice(m,pr,replace=False))
            # Both kernels must participate even if one is a minority channel.
            columns=self.tree.column_permutation[int(cs):int(cs+nc)]
            qcols=np.flatnonzero(columns<len(self.mesh.faces));ucols=np.flatnonzero(columns>=len(self.mesh.faces))
            nq=min(len(qcols),pc//2);nu=min(len(ucols),pc-nq)
            nq=min(len(qcols),pc-nu)
            cc.append(np.r_[rng.choice(qcols,nq,replace=False),rng.choice(ucols,nu,replace=False)])
        check_rows=cp.asarray(rr,dtype=cp.int32);check_faces=cp.asarray(cc,dtype=cp.int32)
        check_cols=check_faces
        desc=cp.asarray(descriptions,dtype=cp.int64)
        reference=self._batch(desc,3,checks=(check_rows,check_faces))
        full_columns=self.columns[desc[:,2,None]+cp.arange(n)[None,:]]
        flux=full_columns<len(self.mesh.faces)
        sampled_flux=cp.take_along_axis(flux,check_faces,axis=1)
        norm_q=cp.sqrt(cp.sum(reference**2*sampled_flux[:,None,:],axis=(1,2)))
        norm_u=cp.sqrt(cp.sum(reference**2*(~sampled_flux[:,None,:]),axis=(1,2)))
        balance=cp.where((norm_q>0)&(norm_u>0),norm_q/cp.maximum(norm_u,1e-300),1.)
        column_scale=cp.where(flux,1.,balance[:,None])
        sample_scale=cp.take_along_axis(column_scale,check_faces,axis=1)
        reference*=sample_scale[:,None,:]
        denominator=cp.sqrt(cp.sum(reference**2,axis=(1,2)))
        U=cp.zeros((count,m,limit));W=cp.zeros((count,limit,n))
        used=cp.zeros((count,m),dtype=bool);active=denominator>0
        ranks=cp.zeros(count,dtype=cp.int32);errors=cp.full(count,cp.inf)
        batch=cp.arange(count);pivot=cp.zeros(count,dtype=cp.int32)
        for k in range(limit):
            activity=active if self.skip_inactive else None
            row=(self._batch(desc,1,nr=m,nc=n,pivots=pivot,far=True,active=activity)
                 if plan is None else self._pivot_rows(plan,pivot,activity))
            row*=column_scale
            if k:
                row-=cp.einsum('bk,bkj->bj',U[batch,pivot,:k],W[:,:k,:])
            j=cp.argmax(cp.abs(row),axis=1).astype(cp.int32)
            p=row[batch,j];active&=cp.abs(p)>1e-30
            col=self._batch(desc,2,nr=m,nc=n,pivots=j,far=True,active=active if self.skip_inactive else None)
            col*=column_scale[batch,j,None]
            if k:
                wcol=W[batch[:,None],cp.arange(k)[None,:],j[:,None]]
                col-=cp.einsum('bmk,bk->bm',U[:,:,:k],wcol)
            U[:,:,k]=cp.where(active[:,None],col,0.)
            W[:,k,:]=cp.where(active[:,None],row/cp.where(active,p,1.)[:,None],0.)
            used[batch,pivot]=True
            if (k+1)%4==0 or k+1==limit:
                a=U[batch[:,None],check_rows,:k+1]
                b=W[batch[:,None,None],cp.arange(k+1)[None,:,None],check_cols[:,None,:]]
                residual=reference-a@b
                eq=cp.sqrt(cp.sum(residual**2*sampled_flux[:,None,:],axis=(1,2)))/cp.maximum(norm_q,1e-300)
                eu=cp.sqrt(cp.sum(residual**2*(~sampled_flux[:,None,:]),axis=(1,2)))/cp.maximum(norm_u*balance,1e-300)
                error=cp.maximum(eq,eu)
                done=active&(error<self.rtol*.2)
                economical=(k+1)*(m+n)<m*n
                ranks=cp.where(done&economical,k+1,ranks)
                errors=cp.where(done,error,errors);active&=~done
                if not bool(cp.any(active).get()):
                    break
            candidate=cp.where(used,0.,cp.abs(col))
            pivot=cp.argmax(candidate,axis=1).astype(cp.int32)
        W/=column_scale[:,None,:]
        if self.storage_dtype.itemsize==4:
            # Certify the retained factors AFTER coefficient rounding, while
            # source integrals, ACA and this residual evaluation stay FP64.
            rounded_u=U.astype(cp.float32).astype(cp.float64)
            rounded_w=W.astype(cp.float32).astype(cp.float64)
            approximation=(rounded_u[batch[:,None],check_rows,:]
                @rounded_w[batch[:,None,None],cp.arange(limit)[None,:,None],check_cols[:,None,:]])
            residual=reference-approximation*sample_scale[:,None,:]
            eq=cp.sqrt(cp.sum(residual**2*sampled_flux[:,None,:],axis=(1,2)))/cp.maximum(norm_q,1e-300)
            eu=cp.sqrt(cp.sum(residual**2*(~sampled_flux[:,None,:]),axis=(1,2)))/cp.maximum(norm_u*balance,1e-300)
            quantized_error=cp.maximum(eq,eu)
            ranks=cp.where(quantized_error<=self.rtol,ranks,0)
            errors=cp.maximum(errors,quantized_error)
        return U,W,cp.asnumpy(ranks),cp.asnumpy(errors)

    def apply(self,q,trace):
        if self.metadata is None:
            raise ValueError('Coefficient-only probe has no assembled boundary operator')
        cp=self.cp;m=len(self.permutation)
        q=cp.asarray(q,dtype=cp.float64)
        vector=cp.concatenate((q/(2*self.scales[:m]),cp.asarray(trace,dtype=cp.float64)))[self.columns]
        result=cp.zeros(m,dtype=cp.float64)
        count=len(self.metadata)
        self.module.get_function('bem_h_forward')((count,),(128,),
            (self.metadata,np.int32(count),np.int32(self.storage_dtype.itemsize),self.areas[self.permutation],vector,result))
        self.module.get_function('bem_h_transpose_v')((count,),(128,),
            (self.metadata,np.int32(count),np.int32(self.storage_dtype.itemsize),self.columns,self.reverse,self.scales,np.int32(m),
             q[self.permutation]*self.areas[self.permutation],result))
        original=cp.empty_like(result);original[self.permutation]=result
        return cp.asnumpy(original)

    def single(self,q):
        return self.apply(q,np.zeros(len(self.mesh.boundary_nodes)))

    def precondition(self,vector):
        cp=self.cp
        x=cp.asarray(vector)[self.permutation];y=cp.empty_like(x)
        self.module.get_function('bem_h_inverse')((len(self.inverse_metadata),),(128,),
            (self.inverse_metadata,self.inverse,x,y))
        original=cp.empty_like(y);original[self.permutation]=y
        return cp.asnumpy(original)

    def direct_rows(self,rows,q,trace):
        """Independent original-operator row sums, including V symmetrization."""
        cp=self.cp;rows=np.asarray(rows,dtype=np.int32)
        m=len(self.mesh.faces);q=cp.asarray(q)
        vector=cp.concatenate((q/(2*self.scales[:m]),cp.asarray(trace)))
        result=cp.zeros(len(rows),dtype=cp.float64)
        for first in range(0,len(vector),2048):
            ids=np.arange(first,min(len(vector),first+2048),dtype=np.int32)
            block=self.coefficients(rows,ids)
            result+=block@vector[ids]
        for first in range(0,m,2048):
            ids=np.arange(first,min(m,first+2048),dtype=np.int32)
            transpose=self.coefficients(ids,rows)/self.scales[cp.asarray(rows)][None,:]
            result+=transpose.T@(q[ids]/2)
        return cp.asnumpy(result)
