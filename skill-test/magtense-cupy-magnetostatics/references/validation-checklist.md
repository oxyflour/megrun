# Validation checklist

Use these before calling a model accurate or performant.

1. **Inputs**: report AEDT design/setup/version, actual solved result status, magnetic tetrahedra per object, material mapping, sample points and their units.
2. **Geometry**: confirm all magnetic TET4/TET10 corner-node conversions have valid nonzero volume and matching global coordinates; document TET10 linearization and material/object exclusions.
3. **Materials**: confirm every material ID has its own B–H curve, H in A/m and B in tesla; record curve extrapolation and whether H operated beyond sampled range.
4. **Operator**: keep native MagTense K-vs-H randomized-vector verification enabled. If it fails, stop rather than switch it off.
5. **Solver**: report method (Newton/Picard), dtype, convergence reason, residual, iterations and GPU device; independently recompute final nonlinear residual and use tighter tolerance as a diagnostic.
6. **Accuracy**: compare same air points with solved Maxwell reference; report B vector L2, RMS, max absolute errors, and near-zero-field absolute error. Check Maxwell boundary conditions and mesh convergence.
7. **Speed**: report K construction separately from nonlinear iteration; do not equate CuPy iteration speed to total solve speed. Report tensor cache hit/miss and device memory.
8. **Scope**: never claim arbitrary current coils, hysteresis, anisotropic B–H, forces, arbitrary large meshes or per-block distance-update acceleration.
9. **Evidence**: synthetic NumPy tests are not Maxwell or GPU benchmarks; keep these validations separate in reports.
