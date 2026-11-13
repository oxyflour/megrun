"""Source-free scalar-potential FEM--BEM with dense or CUDA H boundary operators.

P1 tetrahedra / P0 exterior normal derivatives, Johnson--Nedelec coupling:
  -int (B(H)/mu0-H0).grad(v) - <q,v> = 0,
  <psi,(1/2-K)u+Vq> = 0,  H=H0-grad(u),  u_ext=D u-S q.
Normals point OUT of the FEM union; q is d_n u_ext, NOT B.n or charge density.
Only the scattered potential decays at infinity. No component is pinned.

Exact flat-triangle source integrals (including single-layer self integrals),
tensor Gauss/Duffy target integration. Dense CPU boundary storage is bounded;
the optional CUDA H backend keeps sparse FEM and AMG on CPU. Full-model
accuracy and performance require the independent case verification gates.
The current MaterialLaws supplies the unchanged M(H) and its consistent tangent.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, gmres, splu

from gpu_solver import MU0, MaterialLaws, NonConvergence


def material_laws(properties, cell_material):
    """Adapt exported properties without requiring a MagTense native installation."""
    ids = np.asarray(cell_material, dtype=int)
    if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= len(properties)):
        raise ValueError('Invalid cell material indices')
    kinds, mus, rems, dirs, curve_ids, curves = [], [], [], [], [], []
    for p in properties:
        kind = p['kind']
        if kind not in ('permanent', 'linear', 'nonlinear'):
            raise ValueError(f'Unsupported FEM material {kind!r}')
        direction = np.asarray(p.get('direction', [1., 0, 0]), dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
            raise ValueError('Invalid remanence direction')
        br = float(p.get('br_T', 0.))
        if not np.isfinite(br) or br < 0:
            raise ValueError('Invalid remanence')
        kinds.append({'permanent': 1, 'nonlinear': 2, 'linear': 3}[kind])
        mus.append(float(p.get('mu_r', 1.)))
        rems.append(br / MU0)
        dirs.append(direction / np.linalg.norm(direction))
        if kind == 'nonlinear':
            curves.append(p['bh_H_Apm_B_T'])
        curve_ids.append(max(1, len(curves)))
    tiles = SimpleNamespace(magnet_type=np.asarray(kinds)[ids],
        mu_r_ea=np.asarray(mus)[ids], mu_r_oa=np.asarray(mus)[ids],
        M_rem=np.asarray(rems)[ids], u_ea=np.asarray(dirs)[ids],
        stfcn_index=np.asarray(curve_ids)[ids])
    return MaterialLaws(tiles, curves, np, 'float64')


def triangle_rule(order):
    """Interior tensor Gauss points in barycentric coordinates; weights sum to 1."""
    if not isinstance(order, int) or not 2 <= order <= 64:
        raise ValueError('Target quadrature order must be an integer in [2,64]')
    g, w = np.polynomial.legendre.leggauss(order)
    a, b = np.meshgrid((g + 1) / 2, (g + 1) / 2, indexing='ij')
    weights = (np.outer(w, w) * a / 2).ravel()
    return np.column_stack((1 - a.ravel(), (a * (1 - b)).ravel(), (a * b).ravel())), weights


class FEMBEM:
    """Fixed topology and exterior operators, reused across material Newton steps.

    Coordinates are normalized by a length L; the stored unknown is phi/L
    (A/m), so gradients and q retain SI H units. Returned potential is in A.
    Input tetrahedra must form a conforming, nonoverlapping volume mesh.
    Exact-coordinate welding never moves the exported corners.
    """
    def __init__(self, tetra_m, laws, *, quadrature_order=10,
                 max_boundary_faces=4096, max_dense_mib=256,
                 boundary_backend='dense', boundary_options=None):
        started = time.perf_counter()
        raw = np.asarray(tetra_m, dtype=float)
        if (raw.ndim != 3 or raw.shape[1:] != (4, 3) or not len(raw)
                or not np.isfinite(raw).all() or len(raw) != laws.n):
            raise ValueError('Expected finite (material cells,4,3) tetrahedra')
        if max_boundary_faces < 1 or not np.isfinite(max_dense_mib) or max_dense_mib <= 0:
            raise ValueError('Boundary budgets must be positive and finite')
        if boundary_backend not in ('dense', 'geometry', 'cuda_hmatrix'):
            raise ValueError('Unknown FEM-BEM boundary backend')
        self.boundary_backend = boundary_backend
        self.laws = laws
        self.origin = raw.min(axis=(0, 1))
        self.length = float(np.ptp(raw.reshape(-1, 3), axis=0).max())
        if self.length <= 0:
            raise ValueError('Zero geometry scale')
        vertices, inverse = np.unique(raw.reshape(-1, 3), axis=0, return_inverse=True)
        self.vertices = (vertices - self.origin) / self.length
        self.tets = inverse.reshape(-1, 4)
        t = self.vertices[self.tets]
        edges = t[:, 1:] - t[:, :1]
        determinant = np.linalg.det(edges)
        # Scale-relative element test: permits small elements in a graded mesh.
        if np.any(abs(determinant) <= 1e-13 * np.prod(np.linalg.norm(edges, axis=2), axis=1)):
            raise ValueError('Degenerate tetrahedron')
        self.volumes = abs(determinant) / 6
        self.gradients = np.empty_like(t)
        self.gradients[:, 1:] = np.linalg.inv(edges).transpose(0, 2, 1)
        self.gradients[:, 0] = -self.gradients[:, 1:].sum(axis=1)
        faces = self.tets[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
        _, first, counts = np.unique(np.sort(faces, axis=1), axis=0,
                                     return_index=True, return_counts=True)
        if np.any(counts > 2):
            raise ValueError('Nonmanifold volume face')
        self.faces = faces[first[counts == 1]].copy()
        self.owners = first[counts == 1] // 4
        m = len(self.faces)
        self.boundary_nodes = np.unique(self.faces)
        nb = len(self.boundary_nodes)
        # V and its factor, C, plus assembly/factorization copies. This excludes
        # sparse LU fill and bounded quadrature workspaces; never a peak estimate.
        dense_bytes = 8 * (3 * m * m + 2 * m * nb)
        if m == 0 or (boundary_backend == 'dense' and
                      (m > max_boundary_faces or dense_bytes > max_dense_mib * 2**20)):
            raise MemoryError(f'Dense reference boundary rejected: {m} faces, '
                              f'{dense_bytes / 2**20:.1f} MiB budgeted arrays; '
                              'use an accelerated boundary backend for large meshes')
        boundary_edges = np.sort(self.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
        _, ec = np.unique(boundary_edges, axis=0, return_counts=True)
        if np.any(ec != 2):
            raise ValueError('Boundary must be a closed two-manifold')
        tri = self.vertices[self.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        inward = np.einsum('ij,ij->i', normals, tri.mean(axis=1) - t[self.owners].mean(axis=1)) < 0
        self.faces[inward] = self.faces[inward, ::-1]
        self.tri = self.vertices[self.faces]
        a, b = self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0]
        cross = np.cross(a, b)
        self.areas = np.linalg.norm(cross, axis=1) / 2
        self.normals = cross / (2 * self.areas[:, None])
        aa, ab, bb = np.einsum('ij,ij->i', a, a), np.einsum('ij,ij->i', a, b), np.einsum('ij,ij->i', b, b)
        gramdet = (2 * self.areas)**2
        g1 = (bb[:, None] * a - ab[:, None] * b) / gramdet[:, None]
        g2 = (aa[:, None] * b - ab[:, None] * a) / gramdet[:, None]
        self.trace_gradients = np.stack((-g1-g2, g1, g2), axis=1)
        self.edge_vectors = np.roll(self.tri, -1, axis=1) - self.tri
        self.edge_lengths = np.linalg.norm(self.edge_vectors, axis=2)
        self.conormals = np.cross(self.edge_vectors / self.edge_lengths[:, :, None], self.normals[:, None])
        self.trace_faces = np.searchsorted(self.boundary_nodes, self.faces)
        n = len(vertices)
        self.mass = coo_matrix((np.repeat(self.areas / 3, 3),
            (self.faces.ravel(), np.repeat(np.arange(m), 3))), shape=(n, m)).tocsr()
        self.rows = np.broadcast_to(self.tets[:, :, None], (len(t), 4, 4)).ravel()
        self.cols = np.broadcast_to(self.tets[:, None, :], (len(t), 4, 4)).ravel()
        graph = coo_matrix((np.ones(len(self.rows)), (self.rows, self.cols)), shape=(n, n)).tocsr()
        components, self.component_ids = connected_components(graph, directed=False)
        del graph
        self.info = dict(tetrahedra=len(t), P1_nodes=n, P0_boundary_faces=m,
            components=components, quadrature_order=quadrature_order,
            boundary_backend=boundary_backend,
            geometry_seconds=time.perf_counter()-started,
            gauge='scattered potential zero at infinity; no pinned nodes or per-component zero means')
        self.boundary = None
        if boundary_backend == 'geometry':
            self.V = self.C = None
            return
        if boundary_backend == 'cuda_hmatrix':
            from boundary_hmatrix import BoundaryHMatrix
            self.boundary = BoundaryHMatrix(self, quadrature_order=quadrature_order,
                                            **(boundary_options or {}))
            self.V = LinearOperator((m,m), matvec=self.boundary.single, dtype=float)
            self.C = LinearOperator((m,nb), matvec=lambda v:
                .5*self.mass[self.boundary_nodes].T @ v + self.boundary.apply(np.zeros(m),v), dtype=float)
            self.v_diagonal = self.boundary.diagonal
            self.info.update(self.boundary.info, setup_seconds=time.perf_counter()-started)
            return
        bary, weights = triangle_rule(quadrature_order)
        V = np.empty((m, m))
        K = np.zeros((m, nb))
        for i in range(m):
            single, double = self.layers(bary @ self.tri[i])
            V[i] = self.areas[i] * (weights @ single)
            local = self.areas[i] * np.einsum('p,psj->sj', weights, double)
            np.add.at(K[i], self.trace_faces.ravel(), local.ravel())
        symmetry_error = float(np.linalg.norm(V - V.T) / np.linalg.norm(V))
        self.V = (V + V.T) / 2
        self.C = .5 * self.mass[self.boundary_nodes].T.toarray() - K
        self.V_factor = cho_factor(self.V, lower=True, check_finite=True)
        self.v_diagonal = self.V.diagonal()
        constant_error = np.max(abs(K.sum(axis=1) / self.areas + .5))
        if constant_error > 1e-10:
            raise ValueError(f'Closed-surface double-layer identity failed: {constant_error}')
        self.info.update(
            dense_budgeted_array_bytes=dense_bytes,
            single_layer_unsymmetrized_relative_error=symmetry_error,
            double_layer_constant_identity_max_error=float(constant_error),
            setup_seconds=time.perf_counter()-started,
            backend='CPU FP64 sparse FEM / bounded dense Galerkin BEM reference',
            )

    def layers(self, points):
        """Exact source-panel S(P0), D(P1) at arbitrary non-edge observations.

        Divergence theorem reduces S to edge logarithms and a solid angle;
        tangential integration by parts gives D's linear-basis moments.
        Coplanar D is its principal value, zero on each flat source panel.
        """
        x = np.asarray(points, dtype=float)
        r = self.tri[None] - x[:, None, None, :]
        distances = np.linalg.norm(r, axis=3)
        d = -np.einsum('psi,si->ps', r[:, :, 0], self.normals)
        # Coordinates are globally normalized. Rounding of a target's barycentric
        # sum scales with its position, not only with a tiny source panel's edge.
        coordinate_scale=np.maximum(1.,np.linalg.norm(x,axis=1))[:,None]
        coplanar = abs(d) < 32*np.finfo(float).eps*coordinate_scale
        determinant = np.einsum('psi,psi->ps', r[:, :, 0], np.cross(r[:, :, 1], r[:, :, 2]))
        denominator = (np.prod(distances, axis=2)
            + np.einsum('psi,psi->ps', r[:, :, 0], r[:, :, 1]) * distances[:, :, 2]
            + np.einsum('psi,psi->ps', r[:, :, 1], r[:, :, 2]) * distances[:, :, 0]
            + np.einsum('psi,psi->ps', r[:, :, 2], r[:, :, 0]) * distances[:, :, 1])
        omega = -2 * np.arctan2(determinant, denominator)
        omega[coplanar] = 0
        d = np.where(coplanar, 0., d)
        sums = distances + np.roll(distances, -1, axis=2)
        with np.errstate(divide='raise', invalid='raise'):
            logarithms = np.log1p(2 * self.edge_lengths / (sums - self.edge_lengths))
        height = np.einsum('psei,sei->pse', r, self.conormals)
        single = (np.sum(height * logarithms, axis=2) - d * omega) / (4 * np.pi)
        edge_sum = np.einsum('pse,sei->psi', logarithms, self.conormals)
        # lambda_j at the orthogonal projection of x (grad lambda is tangential).
        bary = np.einsum('psi,sji->psj', -r[:, :, 0], self.trace_gradients)
        bary[:, :, 0] += 1
        double = (bary * omega[:, :, None]
            - d[:, :, None] * np.einsum('psi,sji->psj', edge_sum, self.trace_gradients)) / (4 * np.pi)
        if not np.isfinite(single).all() or not np.isfinite(double).all():
            raise FloatingPointError('Nonfinite boundary source integral')
        return single, double

    def fields(self, u, H_external=(0., 0, 0)):
        H = np.asarray(H_external) - np.einsum('tj,tji->ti', u[self.tets], self.gradients)
        M, alpha, beta, direction = self.laws.eval(H)
        return H, MU0 * (H + M), alpha, beta, direction

    def volume_residual(self, u, H_external):
        gradient = np.einsum('tj,tji->ti', u[self.tets], self.gradients)
        M, alpha, beta, direction = self.laws.eval(np.asarray(H_external)-gradient)
        # Evaluate the scattered induction without B/mu0-H0 cancellation.
        local = self.volumes[:, None] * np.einsum('ti,tji->tj', gradient-M, self.gradients)
        residual = np.bincount(self.tets.ravel(), weights=local.ravel(), minlength=len(u))
        tangent = ((1+alpha)[:, None, None] * np.eye(3)
            + (beta-alpha)[:, None, None] * direction[:, :, None] * direction[:, None, :])
        values = self.volumes[:, None, None] * np.einsum('tia,tab,tjb->tij', self.gradients, tangent, self.gradients)
        A = coo_matrix((values.ravel(), (self.rows, self.cols)), shape=(len(u), len(u))).tocsr()
        return residual, A

    def residual(self, u, q, H_external):
        r, A = self.volume_residual(u, H_external)
        return np.r_[r-self.mass @ q, self.boundary_residual(u,q)], A

    def boundary_residual(self, u, q):
        if self.boundary is None:
            return self.C @ u[self.boundary_nodes]+self.V @ q
        return .5*(self.mass.T @ u)+self.boundary.apply(q,u[self.boundary_nodes])

    def boundary_inverse(self, vector):
        if self.boundary is None:
            return cho_solve(self.V_factor, vector, check_finite=False)
        return self.boundary.precondition(vector)

    def refine_boundary(self, *, rtol=1e-6, max_probe_rows=512,
                        max_extra_mib=512, max_passes=6, progress=None):
        """Replace inaccurate H-matrix K blocks by full-order exact coefficients.

        A sampled ACA cross can miss nonzero K entries when most selected
        rows/columns are coplanar. The closed-surface identity locates suspect
        rows; ALL P1 coefficients at those rows are then independently summed.
        Offending original blocks receive an additive exact-minus-approximate
        K block, for every row/basis in that block. V is untouched. This does
        not project or clamp the constant vector. The unmodified numeric cache
        remains reusable; refinements repeat deterministically after loading.
        """
        if self.boundary is None or not 0<rtol<.1 or min(max_probe_rows,max_passes)<1 or max_extra_mib<=0:
            raise ValueError('Expected a CUDA H boundary and positive refinement controls')
        b=self.boundary;cp=b.cp;m=len(self.faces);nb=len(self.boundary_nodes)
        metadata=cp.asnumpy(b.metadata);columns=cp.asnumpy(b.columns)
        permutation=cp.asnumpy(b.permutation);reverse=cp.asnumpy(b.reverse)
        starts=np.asarray([a.data.ptr for a in b.payload],dtype=np.int64)
        order=np.argsort(starts);sorted_starts=starts[order]
        def view(pointer,count):
            index=order[np.searchsorted(sorted_starts,pointer,side='right')-1]
            delta=pointer-starts[index]
            if delta<0 or delta%b.storage_dtype.itemsize or delta+count*b.storage_dtype.itemsize>b.payload[index].nbytes:
                raise ValueError('Invalid H payload descriptor during exact refinement')
            first=delta//b.storage_dtype.itemsize
            return b.payload[index].ravel()[first:first+count]
        def block_rows(a,local):
            rs,nr,cs,nc,rank,left,right=map(int,a)
            if rank<0:
                return view(left,nr*nc).reshape(nr,nc)[local].astype(cp.float64)
            return (view(left,nr*rank).reshape(nr,rank)[local].astype(cp.float64)
                @view(right,rank*nc).reshape(rank,nc).astype(cp.float64))
        started=time.perf_counter();initial_queries=b.coefficient_pairs
        info=dict(kind='full-order exact K blocks selected by independent coefficient errors',
            target=25*rtol,block_absolute_row_L1_threshold=rtol,passes=[],blocks=[],added_bytes=0)
        refined=set();base_storage=b.info['hmatrix_storage_bytes']
        trace_components=self.component_ids[self.boundary_nodes]
        face_components=self.component_ids[self.faces[:,0]]
        for iteration in range(max_passes+1):
            error=b.apply(np.zeros(m),np.ones(nb))/self.areas-.5
            score=abs(error);component_errors=[]
            for component in range(self.info['components']):
                value=b.apply(np.zeros(m),(trace_components==component).astype(float))/self.areas
                component_error=abs(value-.5*(face_components==component))
                component_errors.append(float(component_error.max()))
                score=np.maximum(score,component_error)
            bad=np.flatnonzero(score>25*rtol)
            info['passes'].append(dict(maximum_constant_error=float(abs(error).max()),
                component_constant_max_errors=component_errors,bad_rows=len(bad)))
            if progress:
                progress('boundary_refinement_identity',refinement_pass=iteration,
                         maximum_constant_error=float(abs(error).max()),bad_rows=len(bad))
            if not len(bad):
                info.update(seconds=time.perf_counter()-started,exact_entry_queries=b.coefficient_pairs-initial_queries,
                            added_metadata_bytes=len(info['blocks'])*7*8)
                self.info.update(boundary_refinement=info,hmatrix_cached_base_storage_bytes=base_storage,
                                 hmatrix_storage_bytes=base_storage+info['added_bytes']+info['added_metadata_bytes'])
                return info
            if iteration==max_passes:
                break
            bad=bad[np.argsort(score[bad])[-max_probe_rows:]]
            if progress:
                progress('boundary_refinement_exact_probes',refinement_pass=iteration,rows=len(bad))
            exact=b.coefficients(bad,np.arange(m,m+nb,dtype=np.int32))/cp.asarray(self.areas[bad])[:,None]
            pos=reverse[bad];sort=np.argsort(pos);sorted_pos=pos[sort]
            first=np.searchsorted(sorted_pos,metadata[:,0]);last=np.searchsorted(sorted_pos,metadata[:,0]+metadata[:,1])
            selected=[]
            for count,k in enumerate(np.flatnonzero(first<last)):
                if int(k) in refined:
                    continue
                rs,nr,cs,nc,rank,left,right=map(int,metadata[k])
                nodes=columns[cs:cs+nc];keep=nodes>=m
                if not keep.any():
                    continue
                row_index=sort[first[k]:last[k]];local=pos[row_index]-rs
                actual=block_rows(metadata[k],local)[:,keep]
                reference=exact[row_index[:,None],(nodes[keep]-m)[None,:]]
                score=float(cp.max(cp.sum(abs(actual-reference),axis=1)).get())
                if score>rtol:
                    selected.append((int(k),score))
                if progress and count and count%2000==0:
                    progress('boundary_refinement_scan',blocks_scanned=count,selected_blocks=len(selected))
            del exact
            if not selected:
                raise NonConvergence('Constant identity failed without a resolvable inaccurate K block')
            new_metadata=[]
            for k,score in selected:
                rs,nr,cs,nc,rank,left,right=map(int,metadata[k]);size=nr*nc*b.storage_dtype.itemsize
                if (info['added_bytes']+size>max_extra_mib*2**20 or b.stored_bytes+size>b.budget):
                    raise MemoryError('Exact K block refinement exceeds declared additional/operator budget')
                nodes=columns[cs:cs+nc];keep=nodes>=m;rows=permutation[rs:rs+nr]
                expected=b.coefficients(rows,nodes[keep])/cp.asarray(self.areas[rows])[:,None]
                original=block_rows(metadata[k],np.arange(nr))[:,keep]
                correction=cp.zeros((nr,nc),dtype=b.storage_dtype)
                correction[:,keep]=(expected-original).astype(b.storage_dtype)
                b.payload.append(correction)
                new_metadata.append([rs,nr,cs,nc,-1,int(correction.data.ptr),0])
                refined.add(k);b.stored_bytes+=size;info['added_bytes']+=size
                info['blocks'].append(dict(block=k,rows=nr,columns=nc,P1_columns=int(keep.sum()),
                                           sampled_absolute_row_L1_error=score,bytes=size))
            b.metadata=cp.concatenate((b.metadata,cp.asarray(new_metadata,dtype=cp.int64)))
            if progress:
                progress('boundary_refinement_blocks_replaced',exact_blocks=len(refined),added_bytes=info['added_bytes'])
        raise NonConvergence(f'Exact K refinement did not pass the constant identity: {info["passes"][-1]}')

    def solve(self, *, H_external=(0., 0, 0), tol=1e-9, maxiter=40,
              preconditioner='block', linear_rtol=1e-10, linear_maxiter=30,
              fem_preconditioner='lu', progress=None):
        """Damped Newton and GMRES; failures raise, never return an uncertified field.

        The dense reference uses exact V Cholesky and sparse FEM LU. The H
        backend uses exact diagonal-leaf V inverses and optional fixed CPU
        AMG V-cycles for A + E diag(V)^-1 E.T. Robin terms lift each FEM
        constant mode while retaining relative component potentials.
        """
        if self.V is None:
            raise ValueError('Geometry-only preparation has no boundary operator')
        if (fem_preconditioner not in ('lu','amg') or preconditioner not in ('block', 'none') or not 0 < tol < 1
                or not 0 < linear_rtol < 1 or maxiter < 1 or linear_maxiter < 1):
            raise ValueError('Invalid FEM-BEM solver options')
        H0 = np.asarray(H_external, dtype=float)
        if H0.shape != (3,) or not np.isfinite(H0).all():
            raise ValueError('Expected a finite uniform applied field')
        started = time.perf_counter()
        n, m = self.mass.shape
        u, q = np.zeros(n), np.zeros(m)
        r, A = self.residual(u, q, H0)
        scale = max(float(np.linalg.norm(r)), np.finfo(float).tiny)
        history = []
        for iteration in range(maxiter+1):
            relative = float(np.linalg.norm(r)/scale)
            if progress:
                progress('newton',iteration=iteration,residual=relative)
            if relative <= tol:
                info = dict(self.info, converged=True, residual=relative,
                    nonlinear_updates=iteration, history=history,
                    preconditioner=preconditioner, solve_seconds=time.perf_counter()-started,
                    total_gmres_iterations=sum(h['gmres_iterations'] for h in history),
                    volume_weak_relative_residual=float(np.linalg.norm(r[:n])/scale),
                    boundary_weak_relative_residual=float(np.linalg.norm(r[n:]) /
                        max(np.linalg.norm(self.C @ u[self.boundary_nodes])+np.linalg.norm(self.V @ q), scale*1e-12)))
                return u, q, info
            if iteration == maxiter:
                break
            def multiply(z):
                return np.r_[A @ z[:n]-self.mass @ z[n:],
                    self.boundary_residual(z[:n],z[n:])]
            operator = LinearOperator((n+m, n+m), matvec=multiply, dtype=float)
            pc_started = time.perf_counter()
            pc = None
            if preconditioner == 'block':
                robin = self.mass @ diags(1 / self.v_diagonal) @ self.mass.T
                if fem_preconditioner == 'lu':
                    factor = splu((A+robin).tocsc())
                    fem_apply = factor.solve
                else:
                    import pyamg
                    hierarchy = pyamg.smoothed_aggregation_solver((A+robin).tocsr(),
                        symmetry='symmetric', max_coarse=200)
                    fem_apply = hierarchy.aspreconditioner(cycle='V').matvec
                    self.info['AMG_levels'] = len(hierarchy.levels)
                    self.info['AMG_operator_complexity'] = hierarchy.operator_complexity()
                def apply(z):
                    vq = self.boundary_inverse(z[n:])
                    du = fem_apply(z[:n]+self.mass @ vq)
                    dq = self.boundary_inverse(z[n:]-self.C @ du[self.boundary_nodes])
                    return np.r_[du, dq]
                pc = LinearOperator(operator.shape, matvec=apply, dtype=float)
            pc_seconds = time.perf_counter()-pc_started
            krylov = []
            def linear_progress(value):
                krylov.append(value)
                if progress and len(krylov)%25==0:
                    progress('gmres',newton_iteration=iteration,linear_iterations=len(krylov),
                             preconditioned_relative_residual=float(value))
            step, status = gmres(operator, -r, M=pc, rtol=linear_rtol, atol=0.,
                restart=min(100, n+m), maxiter=linear_maxiter,
                callback=linear_progress, callback_type='pr_norm')
            if status or not np.isfinite(step).all():
                raise NonConvergence(f'FEM-BEM GMRES failed at Newton {iteration}: status={status}, '
                                     f'iterations={len(krylov)}')
            damping = 1.
            for _ in range(24):
                un, qn = u+damping*step[:n], q+damping*step[n:]
                rn, An = self.residual(un, qn, H0)
                if np.linalg.norm(rn) <= (1-1e-4*damping)*np.linalg.norm(r):
                    break
                damping /= 2
            else:
                raise NonConvergence(f'FEM-BEM Newton line search failed: residual={relative}')
            history.append(dict(residual_before=relative, damping=damping,
                                gmres_iterations=len(krylov), preconditioner_seconds=pc_seconds))
            u, q, r, A = un, qn, rn, An
        raise NonConvergence(f'FEM-BEM Newton did not converge after {maxiter} updates: {relative}')

    def potential(self, u, q, points_m):
        """Decaying scattered exterior potential, in A; caller must supply air points."""
        points = (np.asarray(points_m, dtype=float)-self.origin) / self.length
        if self.boundary is not None:
            from boundary_hmatrix import cuda_potential
            return self.length*cuda_potential(self,u[self.boundary_nodes],q,points,geometry=self.boundary.geometry)
        out = []
        for first in range(0, len(points), 32):
            S, D = self.layers(points[first:first+32])
            out.extend(np.einsum('psj,sj->p', D, u[self.faces])-S @ q)
        return np.asarray(out) * self.length

    def locate_points(self,points_m):
        """Find material tetrahedra using size-binned trees and exact barycentrics.

        Returns -1 in air. Points on magnetic faces count as in/on material.
        The trees and barycentric gradients use this same unmodified FEM mesh.
        """
        if not hasattr(self,'_locator_groups'):
            from tetra_fmm import _source_trees
            cells=self.vertices[self.tets]
            centers=cells.mean(axis=1)
            radii=np.linalg.norm(cells-centers[:,None],axis=2).max(axis=1)
            self._locator_groups=_source_trees(centers,radii)
        points=(np.asarray(points_m,dtype=float)-self.origin)/self.length
        result=np.full(len(points),-1,dtype=np.int64)
        for ids,tree,radius in self._locator_groups:
            for k,candidates in enumerate(tree.query_ball_point(points,radius*(1+1e-9),return_sorted=True)):
                if result[k]>=0 or not candidates:
                    continue
                candidates=ids[np.asarray(candidates,dtype=int)]
                delta=points[k]-self.vertices[self.tets[candidates,0]]
                bary=np.einsum('ni,nji->nj',delta,self.gradients[candidates,1:])
                contained=np.all(bary>=-1e-10,axis=1)&(bary.sum(axis=1)<=1+1e-10)
                if contained.any():
                    result[k]=candidates[np.flatnonzero(contained)[0]]
        return result

    def validate_air(self,points_m):
        contained=self.locate_points(points_m)
        if np.any(contained>=0):
            raise ValueError(f'Observation in/on magnetic tetrahedron: {contained[contained>=0][0]}')

    def locate_boundary(self,points_m,normals):
        """Locate P0 boundary traces at physical points on the same CAD surface."""
        if not hasattr(self,'_boundary_locator_groups'):
            from tetra_fmm import _source_trees
            centers=self.tri.mean(axis=1)
            radii=np.linalg.norm(self.tri-centers[:,None],axis=2).max(axis=1)
            self._boundary_locator_groups=_source_trees(centers,radii)
        points=(np.asarray(points_m)-self.origin)/self.length
        result=np.full(len(points),-1,dtype=np.int64)
        for ids,tree,radius in self._boundary_locator_groups:
            for k,candidates in enumerate(tree.query_ball_point(points,radius*(1+1e-9),return_sorted=True)):
                if result[k]>=0 or not candidates:
                    continue
                candidates=ids[np.asarray(candidates,dtype=int)]
                delta=points[k]-self.tri[candidates,0]
                bary=np.einsum('ni,nji->nj',delta,self.trace_gradients[candidates]);bary[:,0]+=1
                distance=abs(np.einsum('ni,ni->n',delta,self.normals[candidates]))
                direction=self.normals[candidates]@np.asarray(normals[k])
                contained=(bary>=-1e-9).all(axis=1)&(distance<1e-10)&(direction>1-1e-7)
                if contained.any():
                    result[k]=candidates[np.flatnonzero(contained)[0]]
        return result

    def air_field(self, u, q, points_m, *, validate_air, H_external=(0., 0, 0), rtol=1e-5):
        """CPU reference gradient: central differences with independent step refinement.

        ALL stencil points must be verified in air by the supplied validator.
        Near interfaces use a smaller step until both geometry and field agree;
        step disagreement raises. No points or components are discarded.
        """
        points = np.asarray(points_m, dtype=float)
        validate_air(points)
        axis = np.eye(3)
        for level in range(10):
            step = self.length * 1e-4 / 2**level
            stencils = np.array([points + s*step/f*d for f in (1, 2, 4)
                                 for d in axis for s in (-1, 1)])
            try:
                validate_air(stencils.reshape(-1, 3))
            except ValueError:
                continue
            values = self.potential(u, q, stencils.reshape(-1, 3)).reshape(3, 3, 2, len(points))
            gradients = (values[:, :, 1]-values[:, :, 0]) / (2*step/np.array([1, 2, 4]))[:, None, None]
            g1, g2 = (4*gradients[1]-gradients[0])/3, (4*gradients[2]-gradients[1])/3
            H = np.asarray(H_external)-g2.T
            error = np.linalg.norm(g2-g1, axis=0)
            if np.all(error <= 1e-4+rtol*np.linalg.norm(H, axis=1)):
                return MU0*H, dict(step_m=step, max_step_difference_Apm=float(error.max()),
                                  method='central difference / Richardson with verified air stencils')
        raise FloatingPointError('Exterior BEM potential gradient failed step/air certification')
