# CUDA volume K

Select `--operator volume --k-builder cuda --backend cupy`. The native CPU
builder remains the compatible default and the independent check.

For constant magnetization in a tetrahedron, each outward-oriented triangular
face has charge density `sigma = M dot n`. Its vector integral is

```
I(r) = integral_face (r-r') / |r-r'|^3 dS
     = sum_edges nu_e log((R_a + R_b + length_e) / (R_a + R_b - length_e)) - n Omega
K_ab = sum_faces I_a n_b / (4 pi)
```

`nu_e` points outward within the face. `Omega` is its oriented solid angle
computed with `2 atan2(det, denominator)` using vertex vectors from the probe.
The determinant uses two face edges to reduce far-observation cancellation.
The edge integral uses `log1p`. This implementation was developed independently
of the MagTense Fortran source and is checked against its installed native TET4
kernel. It evaluates the analytical field, rather than replacing far sources
by point dipoles.

The original four-point far quadrature, red8 subdivision, source/receiver order,
and radius-based near-pair criterion are unchanged. A CUDA thread handles one
far pair; a warp handles one near pair and reduces all receiving samples in
double precision. The changed summation order permits small floating-point
differences. FP32 only changes K storage; all integral calculations use FP64.
No fast-math or spectral modification is used.

CPU work remains for geometry/quadrature preparation and near-pair enumeration.
CUDA source-face preparation, integrals and reductions execute on GPU. Bounded
launches permit deadline checks and avoid one long Windows display-GPU kernel.
Independent native field checks and the final air field remain on CPU.

CUDA K caches are distinct from native K caches and from CuPy's compiled-kernel
cache. They include geometry, integral settings, storage precision and the CUDA
source SHA256. An interrupted cache replacement invalidates its sidecar.

Read `validation-gpu-k-20261107.json` for the measured scope and limitations.
