"""Inspect the volume-weighted symmetric part of the centroid collocation K."""
import json
from pathlib import Path
import time

import numpy as np
import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator, eigsh

ROOT = Path(__file__).resolve().parent
folder = ROOT / 'coarse_export'
model = json.loads((folder / 'model.json').read_text())
volumes = []
for obj in model['objects']:
    v = np.load(folder / obj['mesh_npz'])['tetra_m']
    volumes.extend(np.abs(np.einsum('ij,ij->i', np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0]),
                                        v[:, 3]-v[:, 0])) / 6)
K = cp.asarray(np.load(folder / 'K_float64.npy', mmap_mode='r'))
w = cp.repeat(cp.sqrt(cp.asarray(volumes)), 3)


def symmetric(v):
    return .5 * (w * (K @ (v / w)) + (K.T @ (w * v)) / w)


start = time.perf_counter()
op = LinearOperator(K.shape, matvec=symmetric, dtype=K.dtype)
values, vectors = eigsh(op, k=4, which='LA', tol=1e-6, maxiter=400)
residuals = [float(cp.linalg.norm(symmetric(vectors[:, i]) - values[i] * vectors[:, i]).get())
             for i in range(4)]
result = {'operator': 'Symmetric part of sqrt(V) K inv(sqrt(V)); tetra-centroid collocation',
          'largest_eigenvalues': cp.asnumpy(values).tolist(),
          'eigen_residuals': residuals, 'seconds': time.perf_counter() - start,
          'interpretation': 'Diagnostic of collocation operator; not a Maxwell accuracy measurement'}
(ROOT / 'operator_spectrum.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
