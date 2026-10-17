# Maxwell mesh seeding experiment

The first experiment fixes the supplied STEP geometry, two identical cameras
along +X with a 2 mm EM-envelope gap, 0.1 A × 100 turns, counterclockwise coil
current, the previously verified illustrative steel BH curve, and 20 mm air
padding. It tests mesh selection, not the low-fidelity force prediction.

Three independently built designs are solved sequentially using Maxwell 2022 R2
and four CPU cores. They use identical energy error targets (0.5%), nonlinear
residuals (1e-4), 30% adaptive refinement, minimum two passes, minimum two
converged passes, and absolute per-pass Fz convergence of 20 µN for each coil.
Maximum 30 passes is a safety limit, not an accuracy criterion. A result stopped
at the limit without convergence is rejected.

1. Adaptive: no added length mesh operations.
2. Geometry: manual 0.5 mm interior mesh on all camera solids.
3. LowFidelity: 0.7 mm general interior and 0.5 mm coil interior, with selected
   magnetic faces seeded at 0.4 and 0.3 mm. Face ranking uses within-object
   coarse B variation, normal flux, yoke saturation, and proximity to the coil.
   Added surface elements are capped at 20,000 and 35,000 respectively. These
   are seed caps; Maxwell adaptivity can refine any region afterwards.

The ranking has no Maxwell solution or force labels. It is an untrained
heuristic. Geometry cues complement coarse fields because coarse voxels cannot
resolve all edges and material interfaces. The coarse solver's self operator is
cached, and the measured preprocessing time and cache status are recorded.

Timing includes project construction, solver launch and solution, and output
extraction; each stage is also reported separately. All projects start without
imported meshes or existing solutions. A separately tightened Maxwell solution
is used as a common force reference; the initial acceptance target is 0.2%
for each coil's Fz. Reference computation is excluded from all three strategy
timings. Per-pass convergence alone is not an absolute error bound.

The final reference uses a 0.25% energy target and 10 µN force convergence,
starting from the Adaptive mesh and allowing further refinement. The original
draft proposed 0.2% energy; this was revised before the reference solve to limit
validation cost. Candidate settings, candidate timings and the per-coil 0.2%
force acceptance target were unchanged. The unrun reference copied from
LowFidelity is marked UNSOLVED and is not a reference result.

Primary success: at least 1.5× end-to-end speedup with reference force accuracy
maintained. Compare against both Adaptive and Geometry. Beating Adaptive alone
does not establish that coarse fields are more useful than manual seeding.

Run:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_mesh_seeding.py
.\.venv\Scripts\python.exe scripts/compare_mesh_seeding.py --reference outputs/mesh_seeding_2mm/TightReference
```

Results are under `outputs/mesh_seeding_2mm`. One geometry and one gap constitute
a pilot, not proof of speedup across camera types, orientations, or gaps. Do not
count future tuning cases as independent validation.

## Aggressive two-pass development experiment

The new `low-fidelity-aggressive` policy retains the untrained coarse-field
face ranking. It expands the fine surface-area target to 30%, and fine plus
medium to 60%, with 0.2 / 0.28 mm face lengths and 130,000 / 65,000 added-element
caps. Solid interiors use 0.4 mm, coils 0.28 mm. Two vacuum subdomains around
the modules receive a 0.7 mm length floor; the surrounding air is unrestricted.
Vacuum partitioning checks unchanged STEP body volumes and unchanged total
outer-domain volume. No solved Maxwell mesh is imported.

This experiment has a hard maximum of two passes, and changes
MinimumConvergedPasses from two to one. This allows the first available
cross-pass force delta to satisfy the stopping rule. Energy error and delta
remain 0.5%, absolute per-coil Fz delta remains 20 µN, and independent accuracy
against the existing tighter Maxwell reference remains 0.2% per coil.
Hitting the maximum pass count is not acceptance: the native profile must
report adaptive convergence, and all independent checks must pass.

Preflight generates and counts the initial mesh before solving, with a
600,000-element upper limit. A 0.45 mm near-air length generated 749,722
elements and was rejected before any field solve. The final 0.7 mm choice
is a budget adjustment on this known case, not held-out validation.
Successful end-to-end timing includes preflight mesh generation, so moving
meshing earlier does not hide its cost. Earlier failed development attempts
are retained separately and are not production per-case timings.

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_mesh_seeding.py --output outputs/mesh_seeding_aggressive_repeat --cases Aggressive --max-passes 2 --minimum-converged-passes 1 --generate-initial-mesh
.\.venv\Scripts\python.exe scripts/audit_aggressive_seeding.py --output outputs/mesh_seeding_aggressive_repeat
```

The comparison with previous timings changes both mesh policy and the number
of consecutive convergence checks. It measures the complete new workflow;
it does not isolate the causal speedup of the coarse-field ranking.
One-pass results can be compared with the tighter reference on this case,
but cannot provide a cross-pass force or energy-change check. A one-pass
production workflow would require separate validation over its operating range.

Measured result: `outputs/mesh_seeding_aggressive_2mm_v3` starts with 546,205
tetrahedra and reaches 710,069 after two passes. The initial force differs
from the tighter reference by at most 0.13050%, and the second-pass force by
0.04705%. Energy error falls from 0.61659% to 0.33327%. However, the force
change is 28.143 µN, above the unchanged 20 µN criterion; Maxwell reports
normal completion but **adaptive passes did not converge**. The two-pass
attempt fails the protocol and costs 26.88 minutes end to end, versus 25.86
minutes for the original Adaptive and 21.83 for the original LowFidelity.
This is not a speedup.

The first pass nevertheless meets the external 0.2% total-Fz accuracy target
on this known case. Its native logged stage lasts 509 seconds, with another
153.84 seconds for build and initial mesh generation. A separately capped
one-pass end-to-end run has not been measured, so these partial timings are
not a measured one-pass speedup. The first pass also misses the 0.5% energy
target and has no cross-pass force check. A one-pass force-focused workflow
therefore remains a candidate for independent validation, not a native
converged solution or validated distance-interaction force predictor.

## Local seeds and one-pass force validation

The `low-fidelity-local` policy uses primitive non-model boxes. The boxes
constrain length refinement inside the existing solids and air without
becoming material objects or splitting the physical domain. The saved plan
checks unchanged physical objects and volumes; extraction records the native
non-model flags, object IDs and mesh operations. Thin boxes cover the winding
bars and adjacent gaps. Yoke CAD vertices are ranked using the coarse field
variation, flux and distance to the annular winding; only the highest four
vertices per module receive small 0.24 mm boxes with 0.08 mm length seeds.

One pass is evaluated as a **fixed-mesh force calculation**. It must finish
normally, preserve the geometry and physics checks, import no Maxwell mesh,
and differ from an independently refined same-gap Maxwell reference by at
most 0.2% in each coil's total Fz. It cannot provide a cross-pass energy or
force convergence check. Neither meeting a force accuracy target nor
reaching MaximumPasses=1 is native adaptive convergence.

The light trial `outputs/mesh_seeding_local_2mm_v4` used 0.5 mm solids,
0.45 mm coils, 0.35 / 0.5 mm selected faces and 0.34 mm winding/gap boxes.
It generated 302,230 tetrahedra and completed in 7.07 minutes including
initial meshing and extraction, but its worst per-coil Fz error was 0.57898%.
It **failed** the accuracy target; its short runtime is not an accepted
speedup result. Further changes on this 2 mm case remain development.

The denser local trial `outputs/mesh_seeding_local_2mm_v5` used 0.4 mm solids,
0.28 mm coils and 0.2 / 0.3 mm selected faces, with the same small corner and
winding/gap boxes. It generated 479,895 tetrahedra and completed in 11.35
minutes, but still failed: worst per-coil Fz error 0.52777%. Its A / B yokes
had 66,238 / 60,450 tetrahedra and coil forces differed by 47.873 µN, whereas
the tighter reference coil forces differ by only 0.161 µN. Increased total
element count alone did not resolve the observed force error. The next
experiment separately measures MaximumPasses=1 with the earlier aggressive
face and near-air policy whose first pass had already met the force target.

After an accepted development result, `validate_fixed_mesh_holdouts.py`
checks the geometry/material/LF implementation hashes against its source
snapshot, runs cold one-pass candidates at 1 and 3 mm sequentially on four
cores, and builds independent tighter references from previously solved
same-gap native meshes. Reference construction and solves are recorded
separately and excluded from candidate runtime, equally to the original
2 mm benchmark. Candidate meshes never import those native solutions.
The references use 0.25% energy error and two consecutive per-coil Fz changes
of at most 10 µN. Held-out forces must not be used to retune the policy while
continuing to call those cases held out.

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_mesh_seeding.py --output outputs/local_repeat --cases Local --max-passes 1 --minimum-passes 1 --minimum-converged-passes 1 --generate-initial-mesh --initial-mesh-limit 650000
.\.venv\Scripts\python.exe scripts/audit_fixed_mesh.py --output outputs/local_repeat
```

The full workflow timing comparison also changes the stopping rule. Without
a matched geometry-only one-pass control, it does not isolate the causal
contribution of low-fidelity ranking. Total Fz accuracy also does not establish
relative accuracy of the much smaller difference between camera distances.

## Separately measured aggressive one-pass workflow

`outputs/mesh_seeding_onepass_2mm` regenerated exactly 546,205 tetrahedra
with the unchanged aggressive face / near-air policy. MaximumPasses,
MinimumPasses and MinimumConvergedPasses were all one. Build and initial
mesh generation took 150.185 s, the native four-core batch solve 533.412 s,
and extraction 43.172 s: **726.768 s / 12.113 min** end to end. Each coil's
Fz matches the earlier first-pass result; worst error versus the tighter
reference is 0.13050%, passing the fixed-mesh force protocol.

The complete workflow is 2.135 times faster than the prior Adaptive and
1.803 times faster than the prior six-pass LowFidelity. Native energy error
is 0.61659%, and native adaptive convergence is false. These runtime ratios
compare different stopping rules; they are not proof that LF ranking alone
caused the improvement. This remains development on the known 2 mm case.

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_mesh_seeding.py --output outputs/onepass_repeat --cases Aggressive --max-passes 1 --minimum-passes 1 --minimum-converged-passes 1 --generate-initial-mesh --initial-mesh-limit 650000
.\.venv\Scripts\python.exe scripts/audit_fixed_mesh.py --output outputs/onepass_repeat --case Aggressive
.\.venv\Scripts\python.exe scripts/validate_fixed_mesh_holdouts.py --output outputs/holdout_repeat --development outputs/mesh_seeding_onepass_2mm
```

The hold-out command freezes candidate construction and LF/physics source
hashes as well as mesh rules. The earlier 1 mm source used two solver cores,
and the earlier 3 mm source had different stopping settings, so neither is
used as a matched end-to-end timing baseline for the new four-core one-pass
candidates. Their completed same-gap meshes are used only to start the
independent tighter accuracy references.

## Frozen-rule distance results

`outputs/mesh_seeding_onepass_holdouts` completed both independent reference
solves and audits. Candidate and reference physics guards pass, source hashes
were rechecked at summary time, and there are no native field-solver errors.
The only candidate warnings are the expected absence of adaptive convergence
with a one-pass cap. The native sessions have closed.

| Gap mm | Passes | Candidate tetrahedra | Maximum per-coil total Fz error | End-to-end min |
|---:|---:|---:|---:|---:|
| 1 | 1 | 542,301 | 0.12312% | 15.13 |
| 2, development | 1 | 546,205 | 0.13050% | 12.11 |
| 3 | 1 | 542,071 | 0.10719% | 11.91 |

The 1 / 3 mm references each converge in five passes, with 907,564 / 952,853
tetrahedra and final energy errors of 0.16363% / 0.16224%. Each reference
also has two consecutive per-coil Fz changes within 10 µN. Additional
hold-out reference build, solve and extraction cost is 93.66 minutes;
candidate runtimes exclude that separately recorded validation cost, just
as the original 2 mm strategy timings exclude their common reference cost.

The final force-accuracy tests pass at all three discrete operating points.
This is not continuous-gap, different-camera, material, current or orientation
validation. It does not establish native one-pass adaptive convergence or
isolate the contribution of LF ranking from geometry and changed stopping.

Force differences relative to 2 mm have maximum per-coil absolute deviations
of 4.999 µN at 1 mm and 10.088 µN at 3 mm versus the current finer references.
The largest relative difference deviation is about 29%, so the 0.2% total-Fz
criterion cannot be reused as a relative interaction-force guarantee.
Reference pass deltas within 10 µN are not rigorous absolute error bounds.
JSON, CSV and Chinese report include both total forces and these diagnostics.

```powershell
.\.venv\Scripts\python.exe scripts/summarize_fixed_mesh_validation.py
```
