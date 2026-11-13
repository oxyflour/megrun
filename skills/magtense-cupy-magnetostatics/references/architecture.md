# Implementation notes

## Mathematical model

Uniform per-magnetic-tetrahedron magnetization, with analytical MagTense demagnetizing field tensor `K`:

```
H = H_external + K @ M
R(M) = M - F(H; material_id)
J(v) = v - dF/dH @ (K @ v)
```

The bundled CuPy prototype implements multi-material magnetization laws, damped Picard, Newton–GMRES with local 3x3 preconditioning, and implicit local-material proximal updates with safeguarded Anderson mixing. The default native builder uses CPU MagTense. The explicit `--operator volume --k-builder cuda` builder evaluates analytical triangle surface-charge integrals on CUDA with FP64 geometry, transcendental functions and receiving-quadrature reductions. These dense paths store the full `3N x 3N` K on the selected device. The full tensor avoids assuming the source/target pair block is symmetric for arbitrary tetrahedral geometry.

Centroid mode observes H at each tetrahedron center. Volume mode averages the analytical tetrahedral source field over receiving tetrahedra: four far-field points, with 256 near/self points by default. Native K/H verification uses the identical observation rule and remains independent of the CUDA formula. These are different discretizations and need separate accuracy checks. Proximal mode retains the supplied material law and K; convergence is judged by the original constitutive residual, not the update size. No spectral clipping is used.

The 2026-11-07 real-Maxwell benchmark and its limits are recorded in [validation-20261107.json](validation-20261107.json).

## Scaling

- Dense FP32 K: `36*N^2` bytes (9 components per ordered tile pair).
- Dense FP64 K: `72*N^2` bytes.
- Example 10,000 tiles: 3.6 GB / 7.2 GB for K alone; allow more memory for workspaces and caches.
- The CUDA volume builder computes all integrals on GPU, with one warp per near pair and bounded launches; geometry, quadrature coordinates and near-pair enumeration remain on CPU. The native/cuda builders remain O(N^2). Their final air field and independent native checks remain on CPU.
- The optional `--operator volume --k-builder fmm` backend in `tetra_fmm.py` instead applies a matrix-free volume-dipole FMM with analytical near correction. The compatible default FMM3D backend runs on CPU and rebuilds the tree each matvec. Explicit --fmm-farfield cuda uses native-Windows/CuPy Chebyshev interpolation FMM (P2M/M2M/M2L/L2L/L2P, exact adjacent-leaf point sums), with persistent Morton trees and interaction lists; GPU point accuracy is checked against direct point-sum probes, rather than claimed as a uniform bound. M2L matrices are applied one offset at a time under a GPU memory budget. CuPy also accelerates neighbour enumeration, correction and material steps. Streamed correction applies face charges directly to H, avoiding nine tensor components per pair. Small CUDA air probes use batched direct GPU point sums plus GPU analytical correction. GPU neighbours use CuPy KDTree construction with bounded custom count/fill traversal, preserving the CPU FP64 radius criterion without a dense query array; `--fmm-neighbours cpu/cuda` selects explicitly. Near blocks are cached within a budget or streamed; high receiving quadrature is batched. Near CSR storage is O(P), and P can grow excessively for poorly graded geometry. Source quadrature error is separate from FMM expansion error. The reported iterative residual belongs to this approximate operator; sampled native constitutive residual and field checks are separate, with no full native residual certification. See [the FMM CLI and limits](../scripts/aedt_magtense_bridge/README_GPU.md#无矩阵-fmm-模式).
- K may be reused for identical magnetic geometry; moving any component changes its cross-interaction tensors and triggers rebuild in this prototype.

## Supported constitutive behavior

- Fixed-geometry permanent magnets with remanent magnetization and isotropic recoil approximation.
- Linear soft magnets.
- Multiple distinct isotropic, single-valued, non-hysteretic B–H curves. `B(H)` converted to `M(H)=B(H)/mu0-H`; the implementation uses piecewise-linear interpolation and constant-M extrapolation beyond the last curve point.

Unsupported: hysteresis and remanent state variables for soft steel, anisotropic constitutive tensors, nonlinear permanent-magnet demagnetization, current-driven winding geometry export, Maxwell stress force integration, transient magnetics.

## FEM–BEM replacement: topology and small CPU prototype

This section records the replacement decision for maintainers after the large
volume-moment run failed practical runtime acceptance. The reproducible full
magnetic-mesh census is `validation/20261109_fmm_refined/run_fmm.py
--assess-fem-bem`, using the solver Python and the existing `large_export`.
Counts, assertions, input hashes and timing are in `fmm_verification.json` under
`fem_bem_feasibility`. It welds exactly equal corner coordinates without moving
vertices. This is a topology census, not a coupled field or GPU-memory test.

| Quantity on the actual 2,718,864 magnetic tetrahedra | Count / storage |
| --- | --- |
| Scalar P1 FEM nodal potentials | 522,922 |
| Structural FEM matrix nonzeros | 7,242,130 |
| FP64 values + int32 CSR indices/pointers | 88,997,252 bytes (84.87 MiB) |
| Material/air boundary triangles / vertices | 235,652 / 117,842 |
| P1 volume potential + P0 boundary flux unknowns | 758,574 |
| One dense FP64 P0 boundary matrix | 413.74 GiB |
| Geometric P2 nodal degrees of freedom | 3,882,526 |

All volume faces have one or two incident tetrahedra; every boundary edge has
two incident triangles. There are ten disconnected magnetic volume components.
The census took 46.26 seconds. CSR storage excludes element geometry, nonlinear
tangents, AMG hierarchy, Krylov vectors and every BEM operator. P1 counts do not
establish accuracy equivalent to the original Maxwell TET10 discretization.

The current AEDT project has both `A_AmpereTurns` and `B_AmpereTurns` equal to
`0A`. A scalar magnetic potential is therefore an appropriate first formulation:
H = -grad(phi), div(B) = 0; magnets use B = mu0*mu_r*H + Br and steel uses its
nonlinear B(H). Nonlinear material tangents belong to sparse local FEM terms.
The fixed exterior Laplace BEM and its compressed geometry can be reused across
Newton steps. Direct nonlinear 3D magnetostatic FEM–BEM coupling with H-matrix
compression has a published precedent: [Bruckner et al.](https://www.sfb-vicom.at/fileadmin/user_upload/proj_comp_mat_lab/0_vicom_publications_pdfs/Bruckner_et_al._-_2012_-_3D_FEM%E2%80%93BEM-coupling_method_to_solve_magnetostatic_.pdf).
Nonzero winding currents would require a suitable reduced potential/source-field
or vector-potential treatment; the present source-free assumption is scoped to
this saved project.

Two layouts merit comparison. Coupling directly on the material surfaces can
reuse the measured mesh and removes all exterior-air volume elements, but needs
accelerated BEM on 235,652 triangles, including accurate close-surface quadrature.
Putting magnets, steel and critical local air/gaps inside the FEM domain allows
a smoother artificial outer BEM interface with a separately coarsened surface.
That layout trades some sparse air elements for a smaller boundary problem; its
actual unknowns require a new local-air mesh and have not been measured here.
The latter is the preferred production candidate if material-surface BEM setup
or close-surface interactions dominate. A plain sparse-FEM open-boundary baseline
should also be timed so BEM's cost and benefit can be assessed independently.

The coupled operator needs block preconditioning, not only local 3x3 material
blocks. Use coefficient-aware AMG for FEM, an appropriate boundary operator
preconditioner, and a Krylov method matching the coupling: a symmetric Costabel
system is generally indefinite (MINRES with an SPD preconditioner, or GMRES),
while a nonsymmetric coupling requires GMRES/FGMRES. Preserve relative component
potentials, handle the ten FEM-block constant modes consistently, and set the
global reference at infinity. Newton damping/continuation and monotone B(H)
tangents still require validation. A symmetric indefinite example is documented
in [NGSolve's FEM–BEM tutorial](https://docu.ngsolve.org/latest/i-tutorials/wta/fembem.html).

### Implemented reference formulation

[`fem_bem.py`](../scripts/aedt_magtense_bridge/fem_bem.py) now implements a
**small CPU FP64 reference**, with sparse P1 tetrahedral FEM, dense P0-tested
Laplace BEM, consistent material Newton updates and GMRES. It uses the
nonsymmetric [Johnson–Nédélec coupling](https://arxiv.org/abs/1211.4225), not the
prospective symmetric layout above. With outward material-union normal n,
scattered potential u and q = d_n u_ext, the equations are:

```
H = H0 - grad(u),   b(H) = B(H)/mu0
- integral_volume (b(H)-H0).grad(v) - <q,v>_boundary = 0
<psi, (1/2-K) trace(u) + V q>_boundary = 0
u_ext = D trace(u) - S q,     u_ext -> 0 at infinity
```

H0 is an optional prescribed uniform applied field, not a converted winding.
Thus exterior B.n/mu0 = H0.n-q; q is not magnetic charge or B.n. Potentials
share the FEM boundary trace. The exterior decay condition fixes every relative
component potential without pinning a node or imposing separate zero means.
Coordinates use one global length L; the internal unknown is u/L (A/m), its
physical potential is in A, and q and H are A/m. `MaterialLaws` from the current
volume solver supplies M(H), transverse/longitudinal tangents and the same
constant-M extrapolation. The FEM Jacobian is the local integral of
grad(v).(I+dM/dH).grad(delta_u).

Source-panel integrals use exact edge-logarithm/solid-angle formulas for S(P0)
and D(P1), including planar self terms. Targets use tensor Gauss/Duffy rules;
V is symmetrized after recording the raw symmetry defect. The closed-union
identity K*1=-1/2 and Cholesky of V are checked. This does **not** certify
close-gap target integration for arbitrary meshes: increase target order and
compare the resulting fields. Exterior fields differentiate the exact-source
potential using step refinement and Richardson agreement; all nominal and
stencil coordinates must pass the existing air validator.

The reference block preconditioner factors V once and uses sparse LU of
A + E diag(V)^-1 E.T, where E is the P1/P0 boundary mass pairing. These Robin
terms lift all disconnected FEM constant modes. This is a checkable reference
for a future coefficient-aware AMG / accelerated-boundary implementation;
sparse LU and dense Cholesky are not the production preconditioner. The
constructor rejects more than 4,096 boundary faces or over 256 MiB of budgeted
dense arrays before assembling BEM. The budget excludes sparse fill and
quadrature workspaces and must not be reported as peak memory.

### Reproduce small-model acceptance

From the repository root, using the retained solver environment:

```powershell
& .\skill-test\.venv\Scripts\python.exe -m pytest -q skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge/tests/test_fem_bem.py
& .\skill-test\.venv\Scripts\python.exe -u validation/20261109_fmm_refined/run_fmm.py --prototype-fem-bem
```

The existing runner appends `fem_bem_prototype_runs` to `fmm_verification.json`;
it preserves the topology census and previous successes/failures. Source/input
hashes, limits, Newton history, weak residuals and interface diagnostics are
retained. It tests 32/128/512-face PM and nonlinear steel spheres against the
analytic demagnetizing factor 1/3, target-order refinement and GMRES counts,
then the existing 3,629-cell / 485-air-point solved Maxwell case at orders 8/12.
The unit suite additionally checks exact/independent panel integrals, assembled
material tangents, two-component relative potentials, per-component net flux,
length/translation invariance, failure propagation and the allocation guard.
An independent exterior dipole test checks absolute potential and B-field
convergence from 128 to 512 faces, with a 3% fine-mesh tolerance for each.
`--fem-bem-synthetic-only` explicitly omits Maxwell verification. Weak residuals
and raw element/interface flux differences are distinct metrics: P1/P0 does
not enforce pointwise flux continuity. Acceptance limits in the runner are
prototype limits, not a high-order Maxwell equivalence claim.

The executed sphere acceptance gives the following volume-weighted relative
L2 errors in H (the nonlinear sphere uses H0=560,000 A/m):

| Boundary faces | PM sphere | Nonlinear steel sphere |
| --- | --- | --- |
| 32 | 7.210% | 13.666% |
| 128 | 1.976% | 3.431% |
| 512 | 0.586% | 0.895% |

On the 128-face PM case, block preconditioning reduces GMRES from 87 to 13
iterations with agreement of the converged solutions. On the 512-face cases,
the offset interface potential differences are below 0.4%, and the raw normal
flux differences are below 5%. These are synthetic geometry convergence tests;
the separate small Maxwell run uses its own retained geometry and materials.

The completed real case has 3,629 tetrahedra, 1,340 P1 nodes, 2,382 P0 faces
and ten connected components. All 485 original air points and all gradient
stencils passed the air checks. Both target orders converged in three Newton
updates; the coupled residual was below 1.4e-13 and the weak boundary relative
residual below 8e-16.

| Target order | Maxwell air B relative L2 | RMS / max vector error | BEM setup | Nonlinear solve |
| --- | --- | --- | --- | --- |
| 8 | 3.92967% | 1.59280 / 7.22707 mT | 269.41 s | 18.43 s |
| 12 | 3.92958% | 1.59276 / 7.22685 mT | 636.36 s | 15.23 s |

The order-8/12 air-field relative difference is 9.14e-6 (0.000914%). The complete
sphere-plus-real validation took 1,040.43 s; measured process peak RSS was
388.80 MiB. These are CPU reference measurements, not GPU memory or a matched
Maxwell timing comparison. Source and input hashes are in the appended record.

**Interface limitation:** on this coarse real mesh the offset potential-trace
difference is 4.31%, while the raw owner-element versus boundary normal-flux
relative L2 difference is 78.97%. The ten component net fluxes balance to below
4e-17 relative, but that does not certify local interface flux accuracy. The
prototype's declared real-case acceptance checks weak residuals, air-field
error and quadrature refinement; its pointwise interface diagnostics are not
passed production thresholds. Real-interface mesh convergence and suitable
flux recovery or a higher-order space remain necessary. Neither the 3.93%
air error nor the much smaller quadrature difference establishes Maxwell
TET10 accuracy.

### Direct AEDT case verification

For validation against the actual solved project, use the existing runner
without synthetic cases:

```powershell
& .\skill-test\.venv\Scripts\python.exe -u validation/20261109_fmm_refined/run_fmm.py --validate-fem-bem-aedt
```

This opens `validation/20261107_repair_staged/full_pair_reference.aedt` in
licensed AEDT 2022 R2 through the existing legacy PyAEDT environment. It reads
`Gap_3mm_CCW / Reference : LastAdaptive`; it does not submit another solve or
save the project. Before comparing fields, it verifies both zero-current
sources, all ten material assignments and CAD bounds, the actual B-H knots,
disabled B-H smoothing, convergence and normal completion. Project hashes
before and after access must match.

The FEM mesh remains the retained 3,629-cell magnetic mesh. The reference is
freshly read from the actual adaptive Maxwell solution, rather than treating
that coarse P1 discretization as equivalent to Maxwell's field space. Queries
cover all 485 air points, B/H at all 3,629 material-cell centers, and B just
inside/outside all 2,382 boundary faces. The inward samples must lie within
their owner tetrahedra and the outward samples must pass the air validator.
Per-object center errors use volume weights; they are sampled errors, not
continuous-volume error integrals. Interface errors use face-area weights.

Results append to `fmm_verification.json / fem_bem_aedt_runs`, including fresh
Maxwell and predicted vectors, material/convergence audits and source/input
hashes. Temporary sampling files are removed after the vectors are retained.
The initial case limits are 5% relative air B error, 10% material-center B
error, 10% boundary-normal B error and 1e-8 coupled residual; the fresh 485-point
Maxwell field must also match the retained export within 1e-9 relative. A failed
limit exits nonzero and retains the failed record. Interior H and Maxwell's
own inside/outside normal-flux jump are independent diagnostics, enabling the
coarse FEM/interface error to be distinguished from reference sampling issues.

The executed direct AEDT run **failed the material and interface limits**;
its nonzero exit is an accuracy failure, not a native export or Newton failure.
The native project hash stayed unchanged, both currents read back as 0 A,
all material/CAD checks passed, and the fresh air vectors were identical to
the retained export. Maxwell reported normal completion and five converged
adaptive passes, ending with 333,613 total tetrahedra, 0.35169% energy error
and 0.010764% delta energy. FEM-BEM still used 3,629 magnetic tetrahedra.

| Sampled quantity | Relative L2 difference | Initial limit / interpretation |
| --- | --- | --- |
| All 485 air B vectors | 3.92967% | 5%, passed |
| Volume-weighted material-center B | 38.581% | 10%, failed |
| Volume-weighted material-center H | 10.818% | Diagnostic; dominated by PM H scale |
| BEM boundary normal B versus Maxwell outside probe | 38.404% | 10%, failed |
| Owner-element normal B versus Maxwell inside probe | 76.851% | Diagnostic |
| Maxwell's own inside/outside normal B difference | 11.914% | Reference trace diagnostic |

Material-center B errors are 43.15%/44.65% in the two steel yokes and
16.70%–18.75% in the eight permanent magnets. The case took 340.32 s including
fresh AEDT access; solver-process peak RSS was 242.63 MiB, excluding the AEDT
process. This is not a new native solve timing or a large-model benchmark.

For a retained run, the independent reference checks can be reproduced without
assembling the BEM matrix again:

```powershell
& .\skill-test\.venv\Scripts\python.exe -u validation/20261109_fmm_refined/run_fmm.py --diagnose-fem-bem-aedt
```

This appends `fem_bem_aedt_diagnostics`; it preserves the failed acceptance.
Increasing the 0.0412 um interface offset to 0.412 um and then 1.75–4.12 um
(owner-altitude capped) leaves the BEM/Maxwell normal-B error at
38.426% and 38.304%. Maxwell's own normal-trace difference is 11.921% and
11.439%. Thus these point traces also need reference mesh/trace-convergence
checks; the failed 38% material error cannot be certified by the air result.
Maxwell B compared with the exported law evaluated at Maxwell H differs by
only 0.680%/0.464% in the yokes and about 3.25e-12 relative in the PMs.
All sampled steel-center H values remain below the first positive B-H knot
(300 A/m; measured maxima 244.92/254.98 A/m), so these center samples do not
validate the higher-field nonlinear branches. The origin of the FEM errors
still needs mesh/space and flux-discretization comparisons before production
scaling. The [case comparison figure](../../../validation/20261109_fmm_refined/fem_bem_aedt_comparison.png)
uses the retained native vectors and per-object metrics; all arrays and hashes
are in the existing verification JSON.

NGSolve/ngsolve.bem is a candidate reference implementation, not an installed
nonlinear magnetostatic solver in this repository. Its documented coupling demo
is linear electrostatics, and Windows pip availability does not establish CUDA
FEM/BEM coverage: current [wheel configuration](https://github.com/NGSolve/ngsolve/blob/master/pyproject.toml)
explicitly enables CUDA for Linux. For native Windows GPU sparse solves,
[NVIDIA AmgX](https://github.com/NVIDIA/AMGX) supports FP64/mixed precision, AMG
and Windows; a compatible FEM assembly, binding and accelerated boundary backend
must still be integrated and tested. Bempp OpenCL dense assembly similarly does
not prove GPU FMM availability: its [operator guide](https://bempp.com/handbook/core/assembling_operators.html)
uses ExaFMM for FMM assembly.

Before committing to the full solver, validate an analytical permanent-magnet
case, nonlinear steel cases against the retained Maxwell air-field reference,
mesh convergence, interface potential/normal-flux continuity, and independently
checked boundary approximation errors. Then measure full-model peak memory,
linear/nonlinear iterations and end-to-end wall time at matched field accuracy.
### CUDA hierarchical boundary and sparse AMG

[`boundary_hmatrix.py`](../scripts/aedt_magtense_bridge/boundary_hmatrix.py)
adds `FEMBEM(..., boundary_backend='cuda_hmatrix')`. Its test space is P0 on
235,652 faces and its trial space is the global P0 flux faces plus 117,842 P1
trace nodes (353,494 columns). Each nodal source sums its incident panel-basis
integrals. Whole triangle/patch support boxes drive rectangular clustering;
the bounding proxies do not change source geometry. Repeating one P1 value
per triangle corner would inflate near blocks and low-rank factors.

Exact near blocks and ACA far factors are built in CUDA batches. Far blocks
that fail the sampled truncation check are subdivided. Compression uses
area-averaged test rows and inverse square-root area scaling on P0 columns,
then balances P0/P1 columns within each block. Stratified held-out columns
include both kernels, with a separate relative error for each channel.
Applications restore physical Galerkin rows and original q. The single layer uses a
separate transpose application to match the symmetric dense reference.
Persistent per-block buffers avoid concatenating a second full operator copy.
Storage limits exclude allocator reserves and workspaces. Optional FP32
coefficient storage retains FP64 source integrals, ACA, material/FEM terms,
preconditioning and accumulation. Rounded factors are checked against the
full-order held-out coefficients before acceptance. Far pivots may use a
lower target quadrature order, while validation, exact blocks and direct-row
probes retain the full order; failing candidates subdivide/fall back.
ACA tolerance is
not a uniform error guarantee: independent original-operator row sums and the
closed-surface constant identity are required before a coupled solve.

ACA pivot rows now reuse a source-panel integral across its P0 column and
incident P1 basis columns. CUDA computes the same four FP64 panel moments once;
the gather retains the original incidence accumulation order and coplanar
zeros. CPU plans map columns to panels once per ACA batch. Default
`pivot_row_backend='auto'` enables sharing for at least 256 row queries,
at least 16 columns and an estimated panel reuse of 1.5 or greater; other
batches use direct evaluation. GPU activity masks skip panel integrals for
blocks that have already completed ACA. Neither optimization changes target
quadrature, held-out checks, rank limits or coefficient precision.

`run_fmm.py --benchmark-fem-bem-assembly` compares direct, activity-masked and
shared-panel ACA on spatial blocks from the actual full AEDT mesh. The warmed
102-block comparison (five size bands plus a retained K outlier, three rotated
timings per variant) required bitwise-equal pivot rows, identical retained
ranks and relative action differences <=1e-10; observed differences were zero.
Ordinary sampled families took 34.2%--65.4% of baseline time, while the single
small outlier had no stable gain and is below the automatic sharing threshold.
These bounded timings are in `fem_bem_assembly_probes`; their sum is not a
weighted estimate of a complete build. For a baseline use
`--fem-bem-h-row-backend direct --no-fem-bem-h-skip-inactive`; defaults select
automatic sharing and skip inactive blocks. Source fingerprints require a
fresh numeric H cache after these kernel changes; earlier evidence is retained.

FEM assembly, sparse AMG, boundary trees, batch scheduling and leaf Cholesky
currently run on CPU; panel integrals, ACA arithmetic, compressed multiplication
and exact-panel air potentials run on CUDA. ACA convergence control periodically
reads GPU flags back to the CPU. The
fixed block preconditioner uses a PyAMG V-cycle for the coefficient-dependent
FEM plus Robin stabilization, and exact diagonal-leaf V block inverses. It
does not factor a global dense V. This is a hybrid implementation. A separate
regression checks two disconnected bodies at permeability contrast 5,000
using a genuine multilevel AMG hierarchy.

From the repository root, the current experimental large-export entrypoint is:

```powershell
& .\skill-test\.venv\Scripts\python.exe -u validation/20261109_fmm_refined/run_fmm.py --solve-fem-bem-large --fem-bem-h-leaf 32 --fem-bem-h-eta 3 --fem-bem-h-rank 64 --fem-bem-h-gib 18 --fem-bem-h-storage float32 --fem-bem-h-far-order 4 --fem-bem-refine-boundary
```

Use `--benchmark-fem-bem-large` instead for boundary setup, direct-row accuracy
and warm matvec checks only. Both append `fem_bem_large_runs`, retaining rejected
budgets, failed runs and stop reasons. A boundary benchmark is not a converged
large field. Add `--fem-bem-check-solution validation/20261109_fmm_refined/fem_bem_large_solution_8.npz`
to the boundary benchmark to verify an unchanged retained solution against the
current operator and repeat native field sampling. This checks its input/content
hashes and a <=1e-8 coupled residual, without claiming a new nonlinear solve.
Fresh builds record separate partition/geometry, leaf preconditioner, exact
near, far ACA/fallback and finalization wall times.
The earlier leaf-64 / eta-2 build exceeded its declared 18 GiB
operator budget after 2,677.49 s, before field iteration. Reducing the leaves
to 32 and using eta 3 reduces planned exact near-block storage from 5.92 to
2.13 GiB in FP64. The subsequent leaf-32 / eta-3 FP64 build completed in
5,952.33 s with 18.48 GiB operator data. Three original-operator row probes
gave relative L2 differences 6.56e-7 / 6.86e-8 / 3.32e-7. However, the maximum
area-normalized constant identity error was 2.539e-3, exceeding 5e-5, so no
coupled solve was attempted. Whole-board sampled peak was 22.42 GiB; process
shared GPU memory stayed at 150 MiB. This passed the storage test and failed
the operator accuracy gate.

Original source-panel sums at 164 seeded/small-area/high-aspect-ratio faces
gave maximum constant error 1.83e-14. Current row/kernel equilibration
addresses this discrepancy; a separate three-body regression with face-area
ratio >1e6 reproduces the former failure and checks the fix. This is not a
completed full-mesh accuracy claim. Reproduce original-integral diagnostics
with `run_fmm.py --diagnose-fem-bem-identity`; records append separately.

The revised mixed-storage build took 2,508.63 s and 11.61 GiB operator data;
whole-board peak was 15.37 GiB. It still failed the constant gate on 158 of
235,652 faces, despite seeded probe differences of 2.55e-8--6.54e-8. Median
constant error was 4.86e-9. At the worst face, four compressed blocks omitted
nonzero K coefficients completely; original sums at all 32 worst faces were
within 3.67e-15. These raw failures remain failures in `fem_bem_large_runs`.

`FEMBEM.refine_boundary()` uses the global and each connected component's
closed-surface constant identity to locate suspect rows. It computes every
original P1 coefficient at those rows, identifies inaccurate blocks, and adds
full-order exact-minus-compressed K blocks for all their test rows and P1
bases. V is unchanged. It does not clamp constants or set component means.
The additional arrays have a separate 512 MiB default budget. Independent
random/smooth actions, the full-surface identity and all component constant
modes must pass again before field iteration. Refinement repeats on loading
the retained raw numeric cache; a cache is not a certified solve. A regression
injects a local coefficient error and checks recovery for arbitrary traces
against the independent dense operator, in both storage precisions.

After construction, the runner streams numeric data under the ignored
`large_export/fem_bem_boundary_cache/`, before numerical acceptance gates.
Geometry, source hashes, precision, integration and compression settings key the cache;
manifest and payload hashes are checked on load. Process-local GPU addresses
are reconstructed from logical buffer references. This allows preconditioner
retries and rejection diagnosis without repeating a cold build. Every load
repeats independent operator and constant-identity checks before a coupled
solve; the cache itself is not an accuracy certificate. An incomplete cache raises
and retains its evidence rather than silently reusing partial data.

A converged coupled solve retains phi/L and q in a uniquely named
NPZ before testing physical accuracy. Its reference is the retained native
333,613-total-cell solved Maxwell project, with identical CAD, materials and
zero currents checked. This is not a native field solution on the 8,679,245-cell
mesh or a same-mesh Maxwell timing comparison.

Run 8 completed the full 2,718,864-magnetic-cell problem on the RTX 3090 Ti
(24 GiB) with CPU FEM/PyAMG and CUDA boundary operations. Source/input/solution
hashes and the unchanged native project were checked after completion. All
four Newton steps used unit damping; residuals were
`1 -> 1.22576e-2 -> 7.18025e-4 -> 9.28746e-6 -> 5.84127e-10`.

| Completed prototype measurement | Result |
| --- | --- |
| Coupled P1/P0 unknowns | 758,574 |
| GMRES iterations by Newton update | 500 / 451 / 200 / 338 (1,489 total) |
| FEM preconditioner | Four-level CPU AMG; operator complexity 1.1681 |
| Exact K block replacements | 119; 755,832 payload + 6,664 metadata bytes |
| Global / worst component constant-identity error | 1.59775e-5 / 1.59762e-5 |
| Final original-operator boundary probe residual | 6.94093e-8 (18 rows) |
| H operator arrays / warm boundary matvec median | 11.61 GiB / 0.10565 s |
| Cache load / exact-block refinement | 141.04 s / 37.15 s |
| Nonlinear solve / complete cached workflow | 810.68 s / 1,042.40 s |
| Sampled whole-board GPU peak / process peak RSS | 14.71 GiB / 4.68 GiB |
| 485 air points: vector B relative L2 | 3.24265%; RMS 1.3143 mT, max 7.0461 mT |
| 3,629 material-center samples, volume weighted | B 7.36961%, H 2.55532% |
| 2,382 surface samples: area-weighted normal B | 9.33422% |

The air/material-B/boundary-B sampling gates were 5%/10%/10% and passed.
Those are sampled comparisons to the retained native solution, not a global
field error bound. The raw-cache cold boundary construction (2,508.63 s) and
the successful cached workflow above were separate runs; do not add them and
label the sum as an observed single cold end-to-end solve. The raw-cache build
used a warm compiled-CUDA cache after regression tests. Resource CSVs, package
versions, solver history and all compared vectors are in `fmm_verification.json`
under `fem_bem_large_runs[7]`; the retained result is
`fem_bem_large_solution_8.npz` (phi/L and q, both FP64).

Run 9 measured the optimized full boundary build on the same exported geometry
and RTX 3090 Ti, with the same leaf size, separation, quadrature, tolerance,
rank cap and coefficient precision as run 7. Both used a warm compiled-CUDA
cache and built a fresh numeric H matrix. H setup fell from **2,508.63 s to
1,639.62 s** (41 min 49 s to 27 min 20 s): **34.64% less time**, saving
869.01 s. The entire raw numeric cache binary is bitwise identical to the old
cache: both full file SHA-256 hashes and manifests were rechecked. The 5,525,340,941
logical entry queries, 207,079 low-rank blocks, 809,676 dense blocks and 11.61 GiB
of raw operator arrays are unchanged. Sharing reduces repeated panel evaluation;
logical entry counts do not measure actual nonzero panel integrals.

| Optimized H setup phase | Wall seconds |
| --- | ---: |
| Boundary partition and GPU geometry preparation | 14.70 |
| Leaf V preconditioner | 3.31 |
| Initial exact near blocks | 169.13 |
| Far ACA, subdivision and failed-block exact fallback | 1,451.80 |
| Finalization | 0.61 |

These are mixed CPU/GPU phase wall times, not a CUDA kernel profile. CPU reuse
plans took 20.64 s within the far stage. The far stage remains the main cost.
Run 9 also wrote the new cache in 73.85 s, repeated the 119 exact K block
replacements in 37.52 s, and passed the original-operator probes, global and all
component constants. It re-evaluated the unchanged run-8 solution: coupled
residual and the same native air/material/boundary sampling gates passed,
with unchanged field errors. The complete boundary benchmark plus retained-field
checks took 1,819.84 s; **no new nonlinear solve was run**, so this is not a
new cold end-to-end solve time. Source/input/solution/native-project hashes
and the full binary comparison are retained in `fem_bem_large_runs[8]`.
The regression suite passed 111 tests; the existing gfortran-dependent test
was skipped because gfortran is unavailable, and is not counted as passed.

The retained native Maxwell batch run completed in **615.24 s** including
process startup (profile solution elapsed 596 s), after five adaptive passes
and 333,613 total tetrahedra, with final energy error 0.35169%. Evidence is
[`full_native_record.json`](../../../validation/20261107_repair_staged/full_native_record.json)
and its referenced profile/convergence files. This is the same CAD/material
problem, with a different mesh and discretization. Our completed large cached
workflow took 1,042.40 s; the 1,639.62 s fresh H build is an additional setup
measurement from another run. There is no observed complete optimized cold
solve time. The 8,679,245-total-cell Maxwell project was generated with
`field_solve_submitted=false`, so no same-large-mesh native field timing exists.
These results do **not** establish a Maxwell speed advantage. Energy error is
not a sampled-field error bound, and matching tetrahedron counts would not
establish matching field accuracy.

Further dispatch headroom can be reproduced with
`run_fmm.py --benchmark-fem-bem-assembly --fem-bem-assembly-batches 64 256 1024`.
This reads actual terminal descriptors and FP32 factor bytes from the retained
full cache without loading its entire payload. Four families of 1,024 blocks
were tested with warmup and three rotated repetitions. Increasing the batch
from 64 to 1,024 gave local ACA speedups of 2.94--3.54, including row-plan cost
but excluding factor-byte validation. Two families retained identical ranks
and factor bytes; the other two changed some ranks or factor bytes, so the
strict equivalence probe **failed and exited nonzero**. Sixteen changed-block
diagnostics against full original panel-integral matrices passed the declared
1e-6 per-channel norm / absolute K row-L1 checks; that limited sample is not
full-operator certification. Failed evidence is retained in
`fem_bem_batch_probes`. The production cap stays at 64 and the validated H
cache/source remain unchanged. A full larger-batch build and all independent
global/component/field gates are required before claiming a new total time.
The 1,489 GMRES iterations / 810.68 s solve also remain a separate bottleneck.

Post-solve diagnostics in `fem_bem_large_solution_diagnostics` found maximum
component relative net flux 7.14e-12 and an area-weighted potential-trace
difference of 0.1142% at 514 face centroids displaced outward by 41.2 nm, with
all observations checked in air. However, the raw owner-element normal B
differs from BEM's exterior normal B by **27.61%** over all boundary faces
(area-weighted L2, normalized by the owner-element values). This P1/P0 method
enforces normal continuity weakly; the raw pointwise flux trace remains
insufficient for a high-order interface-accuracy claim. Passing the declared
sampling gates does not remove that limitation. Higher-order/flux-space
accuracy and a stronger boundary preconditioner remain production work;
1,489 GMRES iterations and cold boundary setup still carry significant cost.
No Maxwell speedup or high-order accuracy equivalence has been certified.

The source-integral regression also fixes a fine-panel defect: a coplanarity
tolerance based only on source edge length gave spurious planar double-layer
values up to about 0.49 on tiny translated triangles. CPU and CUDA now include
the normalized coordinate scale in the planar principal-value check. The
regression requires the exact zero self term; historical reports keep their
original source hashes.

Total large-model memory, convergence, physical error and any Maxwell speed
advantage must be read from completed run records. None follows from the
topology census, a partial build or a small-model regression.

## Three separate verification levels

1. **Offline math tests**: can run without AEDT, GPU or MagTense native binaries.
2. **Native kernel correctness**: needs a working MagTense native extension and real tetrahedral meshes. The GPU script checks K against MagTense's native field calculation by default.
3. **Physical fidelity / performance**: needs solved Maxwell AEDT reference, actual NVIDIA device, and carefully controlled mesh convergence. Neither of the first two proves the third.
