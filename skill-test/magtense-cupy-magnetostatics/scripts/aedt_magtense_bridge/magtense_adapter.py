"""MagTense 3.x tetrahedral magnetostatics adapter (SI units).

The upstream MagTense Fortran loader reads a single 2D state-function table.
Apply the accompanying patch_magtense.py and rebuild the Fortran module to accept
multiple *independent*, variable-length isotropic M(H) curves in one solve.
Old native binaries still work for 0 or 1 nonlinear curve only.
"""
from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi


class UnsupportedPhysics(RuntimeError):
    pass


def validate_bh(bh: list[list[float]]) -> np.ndarray:
    """Input [H(A/m), B(T)] monotonically increasing, single-valued, isotropic BH."""
    a = np.asarray(bh, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 3:
        raise UnsupportedPhysics('B-H must contain at least 3 [H(A/m), B(T)] pairs')
    if not np.all(np.isfinite(a)) or np.any(a[:, 0] < 0) or np.any(np.diff(a[:, 0]) <= 0):
        raise UnsupportedPhysics('B-H must have finite values and strictly increasing nonnegative H')
    if np.max(abs(a[0])) > 1e-9:
        raise UnsupportedPhysics('B-H needs the origin [H=0, B=0] (no remanence/hysteresis)')
    if np.any(np.diff(a[:, 1]) < 0):
        raise UnsupportedPhysics('B(H) must be nondecreasing for this isotropic model')
    return a


def state_function_from_bh(bh: list[list[float]] | np.ndarray,
                           temperature_K: float = 300.0) -> np.ndarray:
    """Convert B(H) into a lossless MagTense [header; H, M] array.

    Header [count, T] is accepted by our patched loader, and also by the
    original single-curve loader (which ignores header column 1).
    """
    a = validate_bh(bh)
    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise UnsupportedPhysics('Temperature must be positive and finite (K)')
    mh = np.column_stack((a[:, 0], a[:, 1] / MU0 - a[:, 0]))
    if np.any(mh[:, 1] < -1e-3):
        raise UnsupportedPhysics('B-H contains B < mu0 H, giving negative M')
    # The first value is a curve-specific count, NOT an actual H coordinate.
    return np.asfortranarray(np.vstack(([float(len(a)), float(temperature_K)], mh)))


def state_functions_from_bh(curves: list[np.ndarray], temperature_K=300.0) -> np.ndarray:
    """Ragged curves -> padded Fortran (nHmax + 1, 2, nCurves) table.

    The original (H,M) knot values are NEVER resampled or blended. The patched
    Fortran loader reads the per-curve row count from the header and ignores
    padding. A 2D table remains accepted by unpatched MagTense for one curve.
    """
    if not curves:
        raise ValueError('Expected at least one B-H curve')
    tables = [state_function_from_bh(c, temperature_K) for c in curves]
    nrows = max(len(t) for t in tables)
    data = np.zeros((nrows, 2, len(tables)), dtype=np.float64, order='F')
    for i, table in enumerate(tables):
        data[:len(table), :, i] = table
    return data


def assemble_tiles(objects: list[dict], folder, max_tiles: int = 10000):
    """Return Tiles instance, nonlinear B-H (or None), metadata about tile counts."""
    from pathlib import Path
    try:
        from magtense.magstatics import Tiles
    except ImportError as ex:
        raise RuntimeError('MagTense not installed: pip install magtense (Python >=3.12)') from ex

    arrays = []
    kinds, mu, br, dirs, sf_ids = [], [], [], [], []
    curve_index_by_bytes: dict[tuple[tuple[int, ...], bytes], int] = {}
    bh_curves: list[np.ndarray] = []
    material_assignments: list[dict] = []
    per_object = {}
    for o in objects:
        f = Path(folder) / o['mesh_npz']
        v = np.load(f)['tetra_m']
        if v.ndim != 3 or v.shape[1:] != (4, 3):
            raise ValueError(f'Bad mesh shape {v.shape} in {f}')
        if not len(v):
            continue
        mat = o['properties']
        kind = mat['kind']
        if kind not in ('permanent', 'linear', 'nonlinear'):
            raise UnsupportedPhysics(f"Unexpected material kind {kind!r} in {o['name']}")
        curve_id = 1  # Fortran 1-based index; ignored by permanent/linear tiles.
        if kind == 'nonlinear':
            c = validate_bh(mat['bh_H_Apm_B_T'])
            key = (c.shape, c.tobytes())
            if key not in curve_index_by_bytes:
                curve_index_by_bytes[key] = len(bh_curves) + 1
                bh_curves.append(c)
            curve_id = curve_index_by_bytes[key]
            material_assignments.append({
                'object': o['name'], 'material': o.get('material', ''),
                'state_function_index': curve_id, 'original_bh_knots': len(c),
            })
        per_object[o['name']] = len(v)
        arrays.append(v)
        n = len(v)
        kinds.extend([kind] * n)
        mu.extend([float(mat.get('mu_r', 1.0))] * n)
        br.extend([float(mat.get('br_T', 0.0))] * n)
        dirs.extend([mat.get('direction', [1, 0, 0])] * n)
        sf_ids.extend([curve_id] * n)

    n_total = len(kinds)
    if n_total == 0:
        raise UnsupportedPhysics('No magnetic tetrahedra. Check material conversion / exports.')
    if n_total > max_tiles:
        raise UnsupportedPhysics(f'{n_total:,} magnetic tetra > --max-tiles {max_tiles:,}. '
                                 f'Naive dense O(N²) interaction can exhaust memory.')

    vv = np.concatenate(arrays, axis=0)
    # Each tetrahedron is 4x3; Tiles vertices property requires 3x4.
    vt = np.asfortranarray(np.transpose(vv, (0, 2, 1)))
    direction = np.asarray(dirs, dtype=float)
    if np.any(np.linalg.norm(direction, axis=1) < 1e-9):
        raise UnsupportedPhysics('Permanent magnet direction cannot be zero')
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    mu = np.asarray(mu, dtype=np.float64)
    br = np.asarray(br, dtype=np.float64)
    if np.any(mu < 1):
        raise UnsupportedPhysics('Relative permeability <1 (diamagnet) not supported by this adapter')
    # Explicit easy axis avoids the random default direction from Tiles.set_easy_axis().
    tiles = Tiles(n=n_total, vertices=vt, tile_type=5,
                  M_rem=br / MU0, easy_axis=direction,
                  mu_r_ea=mu, mu_r_oa=mu)
    tiles.magnet_type = np.asarray([1 if k == 'permanent' else 2 if k == 'nonlinear' else 3
                                    for k in kinds], dtype=np.int32)
    # MagTense hard magnets should iterate as well to include mu_r>1 recoil response.
    tiles.incl_it = np.ones(n_total, dtype=np.int32)
    # Nonlinear and linear soft-magnet M starts at zero by construction (M_rem=0).
    tiles.stfcn_index = np.asarray(sf_ids, dtype=np.int32)
    return tiles, bh_curves or None, {
        'tiles': n_total, 'per_object': per_object,
        'nonlinear_state_functions': len(bh_curves),
        'nonlinear_material_assignments': material_assignments,
        'dense_tensor_GB_reference': round(24 * n_total * n_total / 1e9, 3),
    }


def solve(tiles, pts_m: np.ndarray, bh: list[np.ndarray] | np.ndarray | None = None,
          *, max_error=1e-5, max_it=500, temperature_K=300.0):
    """Return B(N,3) in Tesla at points in AIR only, plus convergence diagnostics."""
    try:
        from magtense.lib import magtensesource
        import magtense.magstatics as magstatics
    except ImportError as ex:
        raise RuntimeError('MagTense 3.x is required; pip install magtense') from ex
    pts = np.asarray(pts_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('Expected pts shape (N,3)')
    if bh is None or (isinstance(bh, (list, tuple)) and not bh):
        curves = []
        tiles, H = magstatics.run_simulation(tiles, pts, max_error=max_error,
                                            max_it=max_it, T=temperature_K)
    else:
        # Preserve backward compatibility with solve(tiles, points, one_curve).
        curves = [np.asarray(bh)] if isinstance(bh, np.ndarray) and bh.ndim == 2 else list(bh)
        count = len(curves)
        patched = bool(getattr(magstatics, 'MAGTENSE_MULTIBH_PATCH_V1', False))
        if count > 1 and not patched:
            raise UnsupportedPhysics('MagTense native multi-BH patch is not installed. '
                                     'Run patch_magtense.py on a MagTense clone, rebuild its '
                                     'Fortran extension, and reinstall that package.')
        ids = np.asarray(tiles.stfcn_index, dtype=int)[np.asarray(tiles.magnet_type) == 2]
        if np.any((ids < 1) | (ids > count)):
            raise UnsupportedPhysics(f'Invalid nonlinear stateFunctionIndex: {ids.tolist()} '
                                     f'for {count} available B-H curves')
        sf3 = state_functions_from_bh(curves, temperature_K)
        # Original MagTense binary expects a 2D table. Patched binary expects
        # a 3D table even if only one material is present.
        sf = sf3 if patched else np.asfortranarray(sf3[:, :, 0])
        api_doc = (getattr(magtensesource.fortrantopythonio.runsimulation, '__doc__', '') or '').lower()
        if patched and 'data_statefcn : input rank-2' in api_doc:
            raise UnsupportedPhysics('Patched Python MagTense was imported but its Fortran '
                                     'extension is still the old rank-2 binary: rebuild native module.')
        try:
            H, M, Mrel = magtensesource.fortrantopythonio.runsimulation(
            centerpos=tiles.center_pos,
            dev_center=tiles.dev_center,
            tile_size=tiles.size,
            vertices=tiles.vertices,
            mag=tiles.M,
            u_ea=tiles.u_ea,
            u_oa1=tiles.u_oa1,
            u_oa2=tiles.u_oa2,
            mu_r_ea=tiles.mu_r_ea,
            mu_r_oa=tiles.mu_r_oa,
            mrem=tiles.M_rem,
            tiletype=tiles.tile_type,
            offset=tiles.offset,
            rotangles=tiles.rot,
            color=tiles.color,
            magnettype=tiles.magnet_type,
            statefunctionindex=tiles.stfcn_index,
            includeiniteration=tiles.incl_it,
            exploitsymmetry=tiles.use_sym,
            symmetryops=tiles.sym_op,
            mrel=tiles.M_rel,
            pts=pts,
            data_statefcn=sf,
            n_statefcn=count,
            t=float(temperature_K),
            maxerr=float(max_error),
            nitemax=int(max_it),
            iteratesolution=True,
            returnsolution=True,
            obs_size=None,
            )
        except (TypeError, ValueError) as ex:
            if patched:
                raise RuntimeError('MagTense Fortran extension rejected multi-BH table. '
                                   'The installed native .pyd/.so may be outdated; rebuild it '
                                   'from your patched source tree.') from ex
            raise
        tiles.M = M
        tiles.M_rel = Mrel
    H = np.asarray(H, dtype=float)
    if H.shape != pts.shape:
        raise RuntimeError(f'MagTense returned H shape {H.shape}, expected {pts.shape}')
    if not np.all(np.isfinite(H)):
        raise RuntimeError('MagTense returned nonfinite field values')
    rel = np.asarray(tiles.M_rel, dtype=float)
    diagnostics = {'max_final_M_rel': float(np.max(np.abs(rel))),
                   'solver_max_error': max_error, 'solver_max_iterations': max_it,
                   'n_nonlinear_bh_curves': len(curves),
                   'original_bh_point_counts': [int(len(c)) for c in curves]}
    if np.max(np.abs(rel)) > 20 * max_error:
        diagnostics['warning'] = 'Magnetization may not have converged; verify iterations.'
    # In air/vacuum: B = mu0 H. Inside magnetic material: WRONG; never use here.
    return MU0 * H, diagnostics
