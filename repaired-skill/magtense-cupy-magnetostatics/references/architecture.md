# Implementation notes

## Mathematical model

Uniform per-magnetic-tetrahedron magnetization, with analytical MagTense demagnetizing field tensor `K`:

```
H = H_external + K @ M
R(M) = M - F(H; material_id)
J(v) = v - dF/dH @ (K @ v)
```

The bundled CuPy prototype implements multi-material magnetization laws, damped Picard, Newton–GMRES with local 3x3 preconditioning, and implicit local-material proximal updates with safeguarded Anderson mixing. CPU MagTense builds the tetrahedral K; CuPy stores the dense full `3N x 3N` K on GPU for nonlinear iterations. The full tensor avoids assuming the source/target pair block is symmetric for arbitrary tetrahedral geometry.

Centroid mode observes H at each tetrahedron center. Volume mode averages the same native source kernel over receiving tetrahedra: four far-field points, with 256 near/self points by default. Native K/H verification uses the identical observation rule. These are different discretizations and need separate accuracy checks. Proximal mode retains the supplied material law and K; convergence is judged by the original constitutive residual, not the update size. No spectral clipping is used.

The 2026-11-07 real-Maxwell benchmark and its limits are recorded in [validation-20261107.json](validation-20261107.json).

## Scaling

- Dense FP32 K: `36*N^2` bytes (9 components per ordered tile pair).
- Dense FP64 K: `72*N^2` bytes.
- Example 10,000 tiles: 3.6 GB / 7.2 GB for K alone; allow more memory for workspaces and caches.
- The current code does not compute K on GPU or use FMM; K construction stays O(N^2) CPU time.
- K may be reused for identical magnetic geometry; moving any component changes its cross-interaction tensors and triggers rebuild in this prototype.

## Supported constitutive behavior

- Fixed-geometry permanent magnets with remanent magnetization and isotropic recoil approximation.
- Linear soft magnets.
- Multiple distinct isotropic, single-valued, non-hysteretic B–H curves. `B(H)` converted to `M(H)=B(H)/mu0-H`; the implementation uses piecewise-linear interpolation and constant-M extrapolation beyond the last curve point.

Unsupported: hysteresis and remanent state variables for soft steel, anisotropic constitutive tensors, nonlinear permanent-magnet demagnetization, current-driven winding geometry export, Maxwell stress force integration, transient magnetics.

## Three separate verification levels

1. **Offline math tests**: can run without AEDT, GPU or MagTense native binaries.
2. **Native kernel correctness**: needs a working MagTense native extension and real tetrahedral meshes. The GPU script checks K against MagTense's native field calculation by default.
3. **Physical fidelity / performance**: needs solved Maxwell AEDT reference, actual NVIDIA device, and carefully controlled mesh convergence. Neither of the first two proves the third.
