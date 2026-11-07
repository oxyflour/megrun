# Maxwell AEDT → MagTense: **multiple different nonlinear B–H materials**

This update changes the Maxwell converter **and the MagTense Fortran native extension**. You may use many nonlinear soft-magnetic materials in one solve. Every tetrahedron stores its 1-based `stateFunctionIndex`; two solids with the same curve share an index. Curves with different numbers of B–H samples are passed without interpolation: each table slice contains a `[count, temperature_K]` header and the original `[H, M]` knots. Inside the patched Fortran `loadStateFunction`, the slice `data_stateFcn(:,:,i)` is loaded into `stateFcn(i)` independently.

> Requires a **newly compiled MagTense native library**. Simply installing the stock `magtense` wheel or editing `n_statefcn` does **not** enable this capability. The bridge rejects multiple curves if it detects the unpatched Python package.

## One-command installation after toolchain setup

Windows PowerShell: `powershell -ExecutionPolicy Bypass -File .\scripts\install_multibh.ps1`

Linux / WSL2: `bash scripts/install_multibh.sh`

Both scripts clone MagTense, apply the source patch, rebuild the native Fortran module, install the patched package and run a two-material native smoke test. Toolchain prerequisites still apply (below).

## Installation — Windows (native AEDT + MagTense)

Prepare a Python 3.12+ environment, [Intel oneAPI Fortran `ifx`](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html), Intel MKL, and `make` (the MagTense Windows Makefile expects a compatible Conda/MSYS environment). You can also run the solve stage in WSL2/Linux after exporting from Maxwell on Windows.

```powershell
# 1. Clone the public MagTense sources
 git clone https://github.com/cmt-dtu-energy/MagTense.git

# 2. Apply native/Python multi-material patch and compile it.
 python patch_magtense.py --repo .\MagTense --build --fc ifx

# 3. Install the patched local checkout *instead of* the PyPI wheel
 python -m pip uninstall -y magtense
 python -m pip install -e .\MagTense\python

# 4. Check which MagTense you imported
 python -c "import magtense.magstatics as mt; print(mt.__file__, mt.MAGTENSE_MULTIBH_PATCH_V1)"

# 5. Verify two different B-H materials alter the native solution independently
 python native_smoke.py
```

If `make` is unavailable, install the native build toolchain first. The `--build` option calls the upstream `make python` target; it does *not* download/build an Intel compiler. A previous MagTense build might require `make clean` before recompiling (prefer a fresh clone). See [upstream build instructions](https://github.com/cmt-dtu-energy/MagTense#building-from-source).

### Alternative — Linux / WSL2

After installing `gfortran`, `make`, `numpy`, and the upstream Python/F2PY build dependencies (e.g. `meson-python`, `ninja`):

```bash
 git clone https://github.com/cmt-dtu-energy/MagTense.git
 python patch_magtense.py --repo ./MagTense --build --fc gfortran
 python -m pip uninstall -y magtense
 python -m pip install -e ./MagTense/python
 python native_smoke.py
```

`make python` is an upstream target with external compiler/library requirements; these commands were **not** exercised against an actual MagTense checkout in the packaging environment. An existing precompiled MagTense wheel is not sufficient for the modified F2PY ABI.

## Maxwell model: export, solve, compare

```powershell
 python bridge.py export --project "D:\sim\camera.aedt" --design "Maxwell3DDesign1" --setup "Setup1 : LastAdaptive" --out out --line-mm "-4,0,3:14,0,3:101"
 python bridge.py inspect --out out
 python bridge.py solve --out out --max-tiles 10000
 python bridge.py compare --out out
```

`solve` now prints every nonlinear object and its `state_function_index` and writes `nonlinear_material_assignments` into `solve_info.json`, so it is easy to check that different shields did **not** accidentally use one B–H table. A distinct curve may have any number of knots >=3. Curve values are converted from Maxwell `[H A/m, B T]` to MagTense `[H, M=B/μ0−H]`.

### Tested and untested

- **Tested here:** pure-Python full pipeline/mocks, B–H knot preservation (including different lengths), material index reuse, rank-3 F2PY call arguments, old-binary detection, patch idempotence, and a compiled `gfortran` standalone smoke test of the exact new Fortran state-function loader loop.
- **Not tested here:** compiling the **full upstream MagTense library**, running its real native solver, or comparing against your AEDT project. `native_smoke.py` does a physical integration check after you build the patched extension; Maxwell accuracy requires your `.aedt` and solver results.
- **Physics scope:** static, isotropic, single-valued soft-magnetic B–H curves; no hysteresis-loop history, anisotropic nonlinear curves, multi-temperature tables, eddy currents, or coil excitations.

### What the patch changes

1. `python/FortranToPythonIO.f90` — `data_stateFcn(nH,nT,n_stateFcn)` in both `IterateTiles` and `runSimulation`.
2. `source/DemagField/IterateMagnetSolution.f90` — `loadStateFunction` creates one curve per third-axis slice and reads each curve's own knot count from header row; legacy built-in Fe CSV header `0` remains supported.
3. `python/src/magtense/magstatics.py` — packaged Fe curve is wrapped as `[:, :, None]` so stock high-level calls remain compatible after the ABI change; a marker flags patched Python.
4. `magtense_adapter.py` and `bridge.py` — de-duplicate identical B–H curves, assign 1-based material indices to all tetrahedral tiles, transmit independent curves, validate a rebuilt native library, report assignments.

Source patch is **idempotent** and refuses to edit any unexpected upstream structure rather than silently patching the wrong version. To verify an existing patch:

```powershell
python patch_magtense.py --repo .\MagTense --check
```
