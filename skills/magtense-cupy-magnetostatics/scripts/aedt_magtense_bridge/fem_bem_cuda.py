"""Device-resident context for FEMBEM.solve(compute_backend='cupy').

Reuses the production FEM/material/Newton/acceptance code in fem_bem.py.
Geometry is uploaded once; FEM CSR assembly, boundary actions, AMG cycles,
Arnoldi vectors and the small triangular solve stay on CUDA in FP64. Only
scalar convergence/telemetry and the final returned u/q cross to the host.
AmgX is loaded through its stable C API; AMGX_LIBRARY may name its DLL/SO.
No CPU AMG, sparse factorization or least-squares fallback is used here.
"""
from __future__ import annotations

import copy
import ctypes as ct
import atexit
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

import numpy as np

from fem_bem import FEMBEM
from gpu_solver import NonConvergence


@lru_cache(maxsize=8)
def _rotations(device):
    import cupy as cp
    return cp.RawKernel(r'''
extern "C" __global__ void gmres_rotations(double* H, double* c, double* s,
                                          double* g, int ld, int j) {
    if (threadIdx.x || blockIdx.x) return;
    double* h=H+ld*j;
    for (int i=0;i<j;++i) {
        double a=h[i],b=h[i+1];
        h[i]=c[i]*a+s[i]*b; h[i+1]=-s[i]*a+c[i]*b;
    }
    double d=hypot(h[j],h[j+1]);
    c[j]=d ? h[j]/d : 1.; s[j]=d ? h[j+1]/d : 0.;
    h[j]=d; h[j+1]=0.;
    double a=g[j]; g[j]=c[j]*a; g[j+1]=-s[j]*a;
}
''', 'gmres_rotations', options=('--std=c++17', '--fmad=false'))


def gpu_gmres(A, b, *, M=None, rtol=1e-10, atol=0., restart=100,
              maxiter=30, callback=None, callback_type='pr_norm'):
    """Restarted left-preconditioned GMRES; maxiter counts restart cycles.

    Two GPU Gram--Schmidt passes, GPU Givens rotations and cuBLAS triangular
    solve. The full unpreconditioned residual is checked before success.
    Unlike cupyx GMRES, this never downloads the Hessenberg matrix to NumPy.
    """
    import cupy as cp
    from cupyx.scipy.linalg import solve_triangular
    if callback_type != 'pr_norm' or restart < 1 or maxiter < 1:
        raise ValueError('Invalid resident GMRES options')
    n = len(b)
    restart = min(restart, n)
    x = cp.zeros_like(b)
    norm_b = float(cp.linalg.norm(b))
    target = max(atol, rtol*norm_b)
    if norm_b == 0:
        return x, 0
    pre = (lambda v: v) if M is None else M.matvec
    norm_pb = float(cp.linalg.norm(pre(b)))
    if not np.isfinite(norm_pb) or norm_pb == 0:
        raise NonConvergence('Resident preconditioner annihilated a nonzero RHS')
    ptol = rtol*norm_pb
    V = cp.empty((n, restart+1), dtype=cp.float64, order='F')
    rotate = _rotations(int(cp.cuda.Device().id))
    residual = b.copy()
    for cycle in range(maxiter):
        if float(cp.linalg.norm(residual)) <= target:
            return x, 0
        z = pre(residual)
        beta = cp.linalg.norm(z)
        if not np.isfinite(float(beta)) or float(beta) == 0:
            return x, cycle+1
        V[:, 0] = z/beta
        H = cp.zeros((restart+1, restart), dtype=cp.float64, order='F')
        cs, sn, g = cp.zeros(restart), cp.zeros(restart), cp.zeros(restart+1)
        g[0] = beta
        for j in range(restart):
            w = pre(A @ V[:, j])
            basis = V[:, :j+1]
            h = basis.T @ w
            w -= basis @ h
            extra = basis.T @ w
            w -= basis @ extra
            H[:j+1, j] = h+extra
            length = cp.linalg.norm(w)
            H[j+1, j] = length
            V[:, j+1] = w/cp.maximum(length, np.finfo(float).tiny)
            rotate((1,), (1,), (H, cs, sn, g, np.int32(restart+1), np.int32(j)))
            estimate = float(cp.abs(g[j+1]))
            if callback:
                callback(estimate/norm_b)
            if estimate <= ptol or j == restart-1:
                k = j+1
                break
        y = solve_triangular(H[:k, :k], g[:k], lower=False, check_finite=False)
        x += V[:, :k] @ y
        if not bool(cp.isfinite(x).all()):
            return x, cycle+1
        residual = b-A @ x
        if float(cp.linalg.norm(residual)) <= target:
            return x, 0
        ptol *= .1
    return x, maxiter


@lru_cache(maxsize=4)
def _amgx_library(path):
    """Keep DLL search-directory handles alive with the loaded library."""
    locations = [Path(path).parent]
    if os.environ.get('CUDA_PATH'):
        cuda_bin = Path(os.environ['CUDA_PATH'])/'bin'
        locations.extend((cuda_bin, cuda_bin/'x64'))
    nvcc = shutil.which('nvcc')
    if nvcc:
        locations.extend((Path(nvcc).parent, Path(nvcc).parent/'x64'))
    directories = []
    if hasattr(os, 'add_dll_directory'):
        directories = [os.add_dll_directory(str(p)) for p in locations if p.is_dir()]
    lib = ct.CDLL(path)
    lib._sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    lib._directories = directories
    P, I = ct.c_void_p, ct.c_int
    signatures = dict(
        AMGX_initialize=[], AMGX_finalize=[],
        AMGX_config_create=[ct.POINTER(P), ct.c_char_p], AMGX_config_destroy=[P],
        AMGX_resources_create=[ct.POINTER(P), P, P, I, ct.POINTER(I)], AMGX_resources_destroy=[P],
        AMGX_matrix_create=[ct.POINTER(P), P, I], AMGX_matrix_destroy=[P],
        AMGX_matrix_upload_all=[P, I, I, I, I, P, P, P, P],
        AMGX_vector_create=[ct.POINTER(P), P, I], AMGX_vector_destroy=[P],
        AMGX_vector_bind=[P, P], AMGX_vector_upload=[P, I, I, P], AMGX_vector_download=[P, P],
        AMGX_vector_set_zero=[P, I, I],
        AMGX_solver_create=[ct.POINTER(P), P, I, P], AMGX_solver_destroy=[P],
        AMGX_solver_setup=[P, P], AMGX_solver_solve_with_0_initial_guess=[P, P, P],
        AMGX_solver_get_status=[P, ct.POINTER(I)],
        AMGX_get_error_string=[I, P, I])
    for name, args in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = args, I
    lib._messages = []
    callback_type = ct.CFUNCTYPE(None, ct.c_char_p, ct.c_int)
    lib._print_callback = callback_type(lambda message, length:
        lib._messages.append(ct.string_at(message, length).decode(errors='replace')))
    lib.AMGX_register_print_callback.argtypes = [callback_type]
    lib.AMGX_register_print_callback.restype = ct.c_int
    if lib.AMGX_register_print_callback(lib._print_callback):
        raise RuntimeError('Cannot install AmgX telemetry callback')
    if lib.AMGX_initialize():
        raise RuntimeError('AMGX_initialize failed')
    atexit.register(lib.AMGX_finalize)
    return lib


class GPUAMG:
    """One fixed FP64 AmgX V-cycle, including GPU hierarchy construction.

    Upload/download API names also accept CUDA pointers. All supplied matrix
    and vector pointers here belong to CuPy device allocations (D2D copies).
    """
    def __init__(self):
        import cupy as cp
        default = Path.home()/'.cache/magtense-gpu/amgx/build'/('amgxsh.dll' if os.name == 'nt' else 'libamgxsh.so')
        path = Path(os.environ.get('AMGX_LIBRARY', str(default))).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError('GPU AMG requires AmgX; set AMGX_LIBRARY to its DLL/SO')
        self.lib = _amgx_library(str(path))
        self.info = dict(AMG_backend='AmgX FP64 device V-cycle',
            AMGX_library=str(path), AMGX_library_sha256=self.lib._sha256,
            AMG_setup_location='GPU', AMG_apply_location='GPU')
        self.cfg, self.resources, self.matrix, self.rhs, self.solution, self.solver = (ct.c_void_p() for _ in range(6))
        configuration = dict(config_version=2, determinism_flag=1,
            solver=dict(solver='AMG', scope='resident', algorithm='AGGREGATION', cycle='V',
                # P1 cell Grams have at most four local DOFs (Robin faces
                # three), so rho(D^-1 P) <= 4. The default AmgX weight .9
                # can amplify obtuse-tet modes. Explicitly scope a stable
                # .5 weight; an unscoped smoother reads AmgX defaults.
                smoother=dict(solver='BLOCK_JACOBI',scope='resident_jacobi',
                    relaxation_factor=.5,max_iters=1,monitor_residual=0,print_solve_stats=0),
                selector='SIZE_2', presweeps=2, postsweeps=2,
                min_coarse_rows=200, max_levels=30, coarse_solver='DENSE_LU_SOLVER',
                amg_host_levels_rows=-1, verbosity_level=3,
                max_iters=1, monitor_residual=0, print_grid_stats=1, print_solve_stats=0,
                obtain_timings=0, tolerance=0.))
        self.info['AMGX_config'] = configuration
        try:
            self.call('AMGX_config_create', ct.byref(self.cfg), json.dumps(configuration).encode())
            device = ct.c_int(int(cp.cuda.Device().id))
            self.call('AMGX_resources_create', ct.byref(self.resources), self.cfg, None, 1, ct.byref(device))
            # AMGX_mode_dDDI from the public amgx_config.h enum.
            for name, handle in [('matrix', self.matrix), ('vector', self.rhs), ('vector', self.solution)]:
                self.call(f'AMGX_{name}_create', ct.byref(handle), self.resources, 8193)
            self.call('AMGX_solver_create', ct.byref(self.solver), self.resources, 8193, self.cfg)
        except Exception:
            self.close()
            raise

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            text = ct.create_string_buffer(4096)
            self.lib.AMGX_get_error_string(code, text, len(text))
            raise NonConvergence(f'{name}: {text.value.decode(errors="replace")}')

    def setup(self, matrix):
        import cupy as cp
        matrix = matrix.astype(cp.float64,copy=False).tocsr()
        if matrix.shape[0] < 1 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('GPU AMG requires a nonempty square matrix')
        if not bool(cp.isfinite(matrix.data).all()):
            raise NonConvergence('Nonfinite GPU AMG operator')
        matrix.sort_indices()
        if matrix.shape[0] >= 2**31 or matrix.nnz >= 2**31:
            raise MemoryError('AmgX dDDI requires 32-bit CSR indices')
        self.n = matrix.shape[0]
        ptr = matrix.indptr.astype(cp.int32, copy=False)
        col = matrix.indices.astype(cp.int32, copy=False)
        self.call('AMGX_matrix_upload_all', self.matrix, self.n, matrix.nnz, 1, 1,
                  ptr.data.ptr, col.data.ptr, matrix.data.data.ptr, None)
        self.call('AMGX_vector_bind', self.rhs, self.matrix)
        self.call('AMGX_vector_bind', self.solution, self.matrix)
        # Binding supplies distribution metadata, not vector storage. Even the
        # zero-initial-guess solve requires a sized device solution allocation.
        self.call('AMGX_vector_set_zero', self.solution, self.n, 1)
        self.lib._messages.clear()
        self.call('AMGX_solver_setup', self.solver, self.matrix)
        telemetry = ''.join(self.lib._messages)
        levels = re.search(r'Number of Levels:\s+(\d+)', telemetry)
        complexity = re.search(r'Operator Complexity:\s+([\d.eE+-]+)', telemetry)
        if levels:
            self.info['AMG_levels'] = int(levels.group(1))
        if complexity:
            self.info['AMG_operator_complexity'] = float(complexity.group(1))

    def apply(self, vector):
        import cupy as cp
        vector = cp.ascontiguousarray(vector, dtype=cp.float64)
        if vector.shape != (self.n,):
            raise ValueError('GPU AMG RHS shape does not match its operator')
        result = cp.empty_like(vector)
        self.call('AMGX_vector_upload', self.rhs, self.n, 1, vector.data.ptr)
        self.call('AMGX_solver_solve_with_0_initial_guess', self.solver, self.rhs, self.solution)
        status = ct.c_int()
        self.call('AMGX_solver_get_status', self.solver, ct.byref(status))
        if status.value not in (0, 3):
            raise NonConvergence(f'AmgX V-cycle failed: status={status.value}')
        self.call('AMGX_vector_download', self.solution, result.data.ptr)
        return result

    def close(self):
        for kind, handle in [('solver', self.solver), ('vector', self.solution), ('vector', self.rhs),
                             ('matrix', self.matrix), ('resources', self.resources), ('config', self.cfg)]:
            if handle.value:
                getattr(self.lib, f'AMGX_{kind}_destroy')(handle)
                handle.value = None


class CUDAFEM(FEMBEM):
    """GPU context consumed by the shared FEMBEM Newton driver."""
    def __init__(self, source):
        import cupy as cp
        from cupyx.scipy import sparse
        from cupyx.scipy.sparse.linalg import LinearOperator
        started = time.perf_counter()
        if source.V is None:
            raise ValueError('Geometry-only preparation has no boundary operator')
        if cp.cuda.get_current_stream().ptr != 0:
            raise ValueError('AmgX interoperability currently requires the default CUDA stream')
        self._xp, self._coo_matrix, self._diags = cp, sparse.coo_matrix, sparse.diags
        self._LinearOperator, self._gmres = LinearOperator, gpu_gmres
        self.to_host = cp.asnumpy
        self._amgx = None
        if len(source.vertices) >= 2**31 or source.mass.nnz >= 2**31:
            raise MemoryError('Resident AmgX path requires 32-bit sparse indices')
        indices = {'tets', 'faces', 'component_ids', 'boundary_nodes', 'rows', 'cols'}
        for name in ('vertices', 'tets', 'faces', 'gradients', 'volumes', 'component_ids',
                     'boundary_nodes', 'areas', 'rows', 'cols', 'v_diagonal'):
            setattr(self, name, cp.asarray(getattr(source, name),
                dtype=cp.int32 if name in indices else cp.float64))
        self.mass = sparse.csr_matrix(source.mass)
        self.laws = copy.copy(source.laws)
        self.laws.xp = cp
        self.laws.dtype = np.dtype('float64')
        for name in ('kind', 'curve_ids', 'rem', 'chi', 'cell_material'):
            if hasattr(self.laws, name):
                setattr(self.laws, name, cp.asarray(getattr(self.laws, name),
                    dtype=cp.float64 if name in ('rem','chi') else cp.int32))
        if not hasattr(self.laws, 'cell_material'):
            signature = np.column_stack((source.laws.kind_cpu,source.laws.ids_cpu,
                                         source.laws.chi,source.laws.rem))
            _, groups = np.unique(signature,axis=0,return_inverse=True)
            self.laws.cell_material = cp.asarray(groups,dtype=cp.int32)
        self.laws.curves = [(cp.asarray(h,dtype=cp.float64), cp.asarray(m,dtype=cp.float64))
                            for h, m in source.laws.curves]
        self.laws.mask = [self.laws.curve_ids == k for k in range(len(self.laws.curves))]
        self.boundary = source.boundary
        self.info = dict(source.info, compute_backend='cupy',
            backend='CUDA FP64 FEM/material/GMRES + device AmgX + CUDA boundary',
            iteration_vector_host_transfers=0, result_vector_host_transfers=2,
            scalar_convergence_telemetry_on_host=True)
        if self.boundary is None:
            self.V, self.C = cp.asarray(source.V), cp.asarray(source.C)
            self._V_cholesky = cp.linalg.cholesky(self.V)
        else:
            self._zero_q = cp.zeros(len(self.faces))
            self._zero_trace = cp.zeros(len(self.boundary_nodes))
            self._trace_mass = self.mass[self.boundary_nodes].T.tocsr()
            self.V = LinearOperator((len(self.faces), len(self.faces)), matvec=lambda q:
                self.boundary.apply(q, self._zero_trace, return_device=True), dtype=float)
            self.C = LinearOperator((len(self.faces), len(self.boundary_nodes)), matvec=lambda u:
                .5*(self._trace_mass @ u)+self.boundary.apply(self._zero_q, u, return_device=True), dtype=float)
        cp.cuda.get_current_stream().synchronize()
        self.info['device_upload_seconds'] = time.perf_counter()-started

    def boundary_residual(self, u, q):
        if self.boundary is None:
            return self.C @ u[self.boundary_nodes]+self.V @ q
        return .5*(self.mass.T @ u)+self.boundary.apply(q, u[self.boundary_nodes], return_device=True)

    def boundary_inverse(self, vector):
        if self.boundary is not None:
            return self.boundary.precondition(vector, return_device=True)
        from cupyx.scipy.linalg import solve_triangular
        intermediate = solve_triangular(self._V_cholesky, vector, lower=True, check_finite=False)
        return solve_triangular(self._V_cholesky.T, intermediate, lower=False, check_finite=False)

    def _fem_preconditioner(self, matrix, kind):
        if kind != 'amg':
            raise ValueError('Resident GPU block preconditioning requires fem_preconditioner="amg"')
        if self._amgx is None:
            self._amgx = GPUAMG()
        self._amgx.setup(matrix)
        self.info.update(self._amgx.info)
        return self._amgx.apply

    def close(self):
        if self._amgx is not None:
            self._amgx.close()
            self._amgx = None
        # LinearOperator callbacks capture this context. Break those cycles
        # so a completed solve releases its uploaded FEM and Krylov storage.
        self.C = self.V = None
