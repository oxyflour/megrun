---
name: magtense-cupy-magnetostatics
description: Convert solved Ansys Maxwell AEDT 3D magnetostatic tetrahedral meshes to MagTense, solve isotropic nonlinear magnetostatic fields with multiple B-H curves on an NVIDIA GPU using CuPy, and compare magnetic B-field accuracy and runtime with Maxwell. Use for camera magnet interference, soft-magnetic shields, tetrahedral-mesh accuracy checks, and distance/geometry parameter-sweep prototypes. Not for transient EM, magnetic hysteresis, or unsupported coil-current sources.
metadata:
  compatibility: Python 3.12+, MagTense native Python extension; optional Windows with Ansys AEDT/PyAEDT for export and NVIDIA CUDA with CuPy for GPU solve.
---

# MagTense + CuPy 非线性静磁场 Skill

This skill ships an actual Python solver and Maxwell import/export/validation tools, not only prompting instructions. Resolve the directory containing this `SKILL.md` as `SKILL_ROOT`; the executable project is at `SKILL_ROOT/scripts/aedt_magtense_bridge/` (`TOOL_DIR`). Execute all example commands **from `TOOL_DIR`**. Never require a nonexistent system-installed `bridge.py`.

## Use this skill when

- A user wants to import **solved** Ansys Maxwell 3D Magnetostatic (`.aedt` plus results) tetrahedral magnetic meshes, automatically omit air/vacuum, and compare fields with a MagTense/CuPy solver.
- A user wants **multiple distinct isotropic, single-valued, non-hysteretic B–H curves**, linear soft magnets, and permanent magnets solved in a shared magnetostatic model.
- A user wants to benchmark CPU MagTense analytical demagnetizing tensor generation versus GPU-resident CuPy nonlinear iteration, or investigate reuse of a dense K matrix across parameter sweeps.

Do **not** claim the package supports non-isotropic or hysteretic materials, coil-current geometry conversion, transient solutions, GPU tensor generation, FMM, or magnetic force calculation. These are not implemented. A CUDA backend being selected does not imply that constructing the tetrahedral K matrix ran on GPU.

## Preflight / environment

1. Read the local project docs when necessary: `scripts/aedt_magtense_bridge/README_GPU.md` (GPU solver), `README.md` (AEDT bridge), and `README_MULTI_BH.md` (optional *native Fortran* multi-BH patch). See `references/validation-checklist.md` for delivery requirements.
2. Verify a Python 3.12+ interpreter. For the real mesh path, install local requirements from the project directory: `python -m pip install -r requirements.txt`. For GPU choose exactly one CuPy wheel compatible with the machine's CUDA toolkit/driver, e.g. CUDA 13.x `python -m pip install "cupy-cuda13x[ctk]"`. Never install both cu12x and cu13x.
3. On AEDT 2022 R2, an existing PyAEDT 0.8.11 export environment can be reused separately from the solver environment. The exporter detects the legacy API, writes point-file numbers in model units, and checks the returned SI coordinates. On machines without AEDT, use an already exported directory (`model.json`, `mesh/*.npz`, `points.csv`) or `bridge.py demo`; do not pretend that AEDT export succeeded. On machines without NVIDIA/CuPy use `--backend numpy` to validate math, but do not claim GPU performance.
4. Run the offline checks before trusting a new setup:

```bash
python gpu_solver.py --self-test --backend numpy
python -m pytest -q tests
```

5. If running the native MagTense reference `bridge.py solve` **with multiple B–H curves**, the upstream Fortran extension must be explicitly patched/rebuilt. **The CuPy path does not require that patch** because it uses MagTense only for tensors and computes material laws in Python. Never silently alter or uninstall the user's MagTense installation; request authorization before running the optional `scripts/install_multibh.ps1` or `.sh`.

## Workflow 1: Import solved Maxwell 3D model

Require a **real project path**, design name, existing analysis sweep, and observation points inside air. Unless the user provided them, discover locally with AEDT when possible; never manufacture names/paths. Work in a new output directory to avoid modifying the `.aedt` project.

Run from `TOOL_DIR`:

```powershell
python bridge.py export --project "D:\sim\camera.aedt" --design "Maxwell3DDesign1" --setup "Setup1 : LastAdaptive" --out out --line-mm "-4,0,3:14,0,3:101" --max-tiles 10000
python bridge.py inspect --out out
```

The project/design/coordinates in this example are placeholders. `--points-csv` is another option, with point coordinates in **meters**. Air/vacuum mesh elements are omitted while physical distances remain unchanged. TET10 is linearized to four corner nodes: disclose the geometric approximation. Never force `--ignore-current-check` unless zero current sources are independently verified. For material expressions or rotated remanence that cannot be interpreted, use `overrides.example.json` and ask for physical clarification rather than guessing.

Selecting `--objects` only limits exported sources; it does not deactivate omitted magnets in Maxwell. A comparison requires matching active sources in both solvers.

## Workflow 2: GPU nonlinear solve

Inspect mesh count before allocating a dense operator. The implementation stores a **full 3N × 3N K matrix**, not a six-component compressed tensor. K itself consumes about `36*N*N` bytes in FP32 or `72*N*N` bytes in FP64, excluding solver/GPU and temporary memory. At N=10,000, this means **3.6 GB FP32 or 7.2 GB FP64**, plus overhead. Recommend beginning with 1k–5k magnetic tiles for validation.

```powershell
python gpu_solver.py --out out --backend cupy --dtype float64 --operator volume --method proximal --tol 1e-8 --cache-k
```

Prefer FP64 for an initial accuracy validation; after a converged reference result, test FP32 performance. For high-permeability soft materials, `--operator volume --method proximal` uses receiving-volume quadrature and implicit material updates; `centroid` with Newton remains available as a baseline. Volume mode uses four far-field receiving points and, by default, 256 near/self receiving points (`--near-levels 2`), all evaluated with the native CPU source kernel. It changes the collocation discretization and needs its own quadrature/mesh accuracy check. K construction can take minutes even when GPU iteration takes seconds.

Keep the default **native K vs field consistency check** enabled; do not add `--skip-k-verify` to hide failure. The check matches the chosen observation: centroid H for centroid K, the identical volume quadrature of native H for volume K. Neither establishes Maxwell accuracy. Both methods raise on non-convergence. Use `--backend numpy` to debug if GPU is unavailable; the real mesh still requires MagTense's native library for K generation. See `README_GPU.md` for mode selection and cache semantics.

`--cache-k` keys K on geometry and precision; volume caches additionally key the native version, quadrature rule and near-pair settings. Centroid and volume caches are distinct. Geometry changes require recomputation; the package does **not** selectively update inter-component blocks. Warm start across changed geometry is permitted only if element ordering/material correspondence is independently verified and the user agrees.

## Workflow 3: Compare against Maxwell

```powershell
python bridge.py compare --out out --prediction gpu_B.csv --prefix gpu_
```

Report at least: magnetic tile count, mesh linearization warnings, B-field relative L2 error and RMS/max absolute error, nonlinear residual and convergence, K-generation time, nonlinear iteration time, GPU model/device/precision if available, and whether the Maxwell reference is from an actual solved `.aedt`. If no reference exists, only report solver self-consistency; do not imply Maxwell accuracy.

The air-field evaluator can resolve removable native-kernel singularities using two-direction symmetric Richardson limits. It verifies every offset remains in air, keeps the nominal coordinates, checks step/direction agreement, and records affected indices in `gpu_solve_info.json`. An uncertified limit raises instead of dropping the point. Report any such evaluations.

The output files include `gpu_B.csv`, `gpu_magnetization.npz`, `gpu_solve_info.json`, `gpu_comparison.csv`, and `gpu_metrics.json`. Always label units: coordinates m, magnetization and H A/m, B tesla. For severe non-convergence or disagreement, evaluate demagnetizing tensor orientation, material curves, sampling in air, boundary conditions and Maxwell mesh convergence before tuning parameters.

## Workflow 4: Offline example (no AEDT, no CUDA)

```bash
python bridge.py demo --out demo_out
python bridge.py inspect --out demo_out
python gpu_solver.py --self-test --backend numpy
```

`demo_out` is synthetic and has **no Maxwell reference**. `gpu_solver.py --self-test` runs an analytic mathematics self-test and does not exercise the native MagTense tetrahedral tensor implementation. For real solves, `magtense` native library is required even with `--backend numpy`.

## Guardrails for automation and code modifications

- Preserve user data and do not modify original `.aedt` files. Avoid running large solves without checking mesh size and available memory.
- Separate **tested** functions from assumptions. Never claim physical accuracy from synthetic tests alone, nor a GPU benchmark from NumPy CPU runs.
- Do not silently approximate unsupported excitations, use one B–H curve for all objects, or truncate geometry to bounding boxes. When the exporter refuses, explain the exact unsupported capability.
- Before editing native MagTense sources, read `README_MULTI_BH.md`; its optional patch updates the Fortran ABI and requires rebuilding. MagTense upstream is GPL-3.0-or-later; assess compliance when distributing modified upstream code.
- When implementing new CUDA features, first preserve the MagTense CPU tensor agreement tests and add a real Maxwell B-field comparison. The existing script's GPU implementation currently accelerates *iteration*, not K construction.

## Further references

- `scripts/aedt_magtense_bridge/README_GPU.md`: full solver CLI, outputs, limitations and debugging.
- `scripts/aedt_magtense_bridge/README.md`: mesh conversion, material overrides and Maxwell export details.
- `scripts/aedt_magtense_bridge/README_MULTI_BH.md`: optional native Fortran multi-material patch for comparison path.
- `references/validation-checklist.md`: required checks before reporting a result.
- `references/architecture.md`: mathematical method, scaling, and known limitations.
