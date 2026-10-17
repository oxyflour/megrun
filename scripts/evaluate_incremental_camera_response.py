"""Hybrid incremental response around an isolated native Maxwell operating point.

No pair force fitting. The coarse Green operator is an approximation, not a
reduced native Maxwell stiffness matrix or a native perturbation-response solve.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_mag.geometry import load_camera
from camera_mag.interaction import MU0, volume_operator
from camera_mag.materials import benchmark_materials
from camera_mag.solver import CameraPairSolver
from camera_mag.response_library import read_vector_field, force_response_operator, native_susceptibility


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--library", default="outputs/single_camera_reuse/isolated_library")
    p.add_argument("--output", default="outputs/single_camera_reuse/incremental")
    p.add_argument("--direct-summary", default="outputs/single_camera_reuse/direct_prediction_summary.json")
    args = p.parse_args()
    library, output = Path(args.library).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = json.loads((library / "export_record.json").read_text())
    direct = json.loads(Path(args.direct_summary).read_text())
    if not record.get("completed_UTC"):
        raise ValueError("Library export is incomplete")
    for name, export in record["exports"].items():
        if hashlib.sha256((library / (name+".fld")).read_bytes()).hexdigest() != export["sha256"]:
            raise ValueError("Library data hash changed")
    camera = load_camera(ROOT / "models/B0500_scale_AF_VCM_EM_only.step", cell_mm=record["specification"].get("cell_mm", .7))
    materials = benchmark_materials()
    active = camera.material_ids != "copper"
    centers, volumes = camera.centers[active], camera.volumes[active]
    ids = camera.material_ids[active]
    h0 = read_vector_field(library / "own_cell_H.fld", camera.centers)[active]
    n, size = len(centers), centers.size
    # Match the actual dense native piecewise-linear B-H table for tangents.
    source = Path(record["specification"]["source"])
    table = np.loadtxt(source / "maxwell_bh.csv", delimiter=",", skiprows=1)
    hnorm = np.linalg.norm(h0, axis=1)
    tangent = native_susceptibility(h0, ids, table)
    # Local spherical self preconditioning removes the high-chi stiffness.
    local = np.linalg.solve(np.eye(3)+tangent/3, tangent)
    begin = perf_counter()
    with threadpool_limits(limits=4, user_api="blas"):
        self_k = volume_operator(centers, centers, volumes, target_volumes=volumes)
        off_self = self_k + np.eye(size)/3
        self_matrix = np.eye(size) - np.einsum("nij,njmk->nimk", local, off_self.reshape(n, 3, n, 3)).reshape(size, size)
        factor = lu_factor(self_matrix)
    offline = perf_counter()-begin
    arrays = np.load(library / "quadrature.npz")
    coarse_n = len(arrays["coarse_points"])
    all_points = np.r_[arrays["coarse_points"], arrays["fine_points"]]
    all_j = read_vector_field(library / "own_J.fld", all_points)
    # Fixed coarse quadrature is tested separately against fine quadrature below.
    coil_points, coil_weights, coil_j = arrays["coarse_points"], arrays["coarse_weights"], all_j[:coarse_n]
    own_force = force_response_operator(coil_points, coil_j, coil_weights, centers, volumes)
    offline += perf_counter()-begin-offline

    def mat_response(h):
        result = np.empty_like(h)
        for material in ("yoke", "magnet"):
            mask = ids == material
            result[mask] = materials[material].response(h[mask])
        return result

    base_m = mat_response(h0)
    base_ext = h0+base_m/3

    def local_response(h):
        result = np.empty_like(h)
        for material in ("yoke", "magnet"):
            mask = ids == material
            result[mask] = materials[material].local_response(h[mask], np.zeros((mask.sum(), 3)))
        return result

    local_base = local_response(base_ext)
    rows, states = [], {}
    spec = record["specification"]
    for gap in spec["gaps_mm"]:
        start = perf_counter()
        shift = np.array([camera.extent[0]+gap*1e-3, 0, 0])
        hab = read_vector_field(library / f"gap_{gap:g}_A_cell_incident_H.fld", camera.centers-shift)[active]
        hba = read_vector_field(library / f"gap_{gap:g}_B_cell_incident_H.fld", camera.centers+shift)[active]
        kab = volume_operator(centers, centers+shift, volumes, target_volumes=volumes)
        blocks = kab.reshape(n, 3, n, 3).transpose(2, 3, 0, 1)
        kba = (blocks*(volumes[None,None,:,None]/volumes[:,None,None,None])).reshape(size, size)

        def response(h):
            rhs = np.einsum("nij,nj->ni", local, h).ravel()
            return lu_solve(factor, rhs)

        ma, mb = np.zeros(size), np.zeros(size)
        linear_history = []
        with threadpool_limits(limits=4, user_api="blas"):
            for iteration in range(60):
                a = response(hab+(kab@mb).reshape(n, 3))
                b = response(hba+(kba@ma).reshape(n, 3))
                err = max(np.linalg.norm(a-ma), np.linalg.norm(b-mb))/max(np.linalg.norm(a), np.linalg.norm(b), 1)
                linear_history.append(float(err))
                ma, mb = a, b
                if err < 1e-8:
                    break
            if linear_history[-1] >= 1e-8:
                raise RuntimeError("Mutual incremental linear response did not converge")
        linear_seconds = perf_counter()-start
        # One-way response diagnostic: isolated response to the native incident
        # field, with no mutual induced feedback. This is not force fitting.
        with threadpool_limits(limits=4, user_api="blas"):
            one_way_a, one_way_b = response(hab), response(hba)
        states[f"gap_{gap:g}_linear_A"], states[f"gap_{gap:g}_linear_B"] = ma.reshape(n, 3), mb.reshape(n, 3)
        # Anchored nonlinear increment leaves the exported native operating point
        # exactly unchanged when the incident field is zero.
        def mapping(flat):
            a, b = flat[:size], flat[size:]
            ha = base_ext + hab + (off_self@a+kab@b).reshape(n, 3)
            hb = base_ext + hba + (off_self@b+kba@a).reshape(n, 3)
            return np.r_[(local_response(ha)-local_base).ravel(), (local_response(hb)-local_base).ravel()]
        iterator = object.__new__(CameraPairSolver)
        iterator.dtype, iterator.omega, iterator.tol = "float64", .7, 1e-7
        iterator.max_iterations, iterator.method, iterator.blas_threads = 1500, "anderson", 4
        flat, count, hist, ok, nonlinear_seconds = iterator._iterate(mapping, np.r_[ma, mb])
        if not ok:
            raise RuntimeError(f"Anchored nonlinear increment did not converge: {hist[-1]}")
        nonlinear_a, nonlinear_b = flat[:size], flat[size:]
        states[f"gap_{gap:g}_nonlinear_A"], states[f"gap_{gap:g}_nonlinear_B"] = nonlinear_a.reshape(n, 3), nonlinear_b.reshape(n, 3)
        force_ab = force_response_operator(coil_points, coil_j, coil_weights, centers+shift, volumes)
        force_ba = force_response_operator(coil_points, coil_j, coil_weights, centers-shift, volumes)
        for mode, a, b in [("linear", ma, mb), ("anchored-nonlinear", nonlinear_a, nonlinear_b)]:
            for label, own_m, other_m, other_force, incident, cross_k in [
                ("A", a, b, force_ab, hab, kab), ("B", b, a, force_ba, hba, kba)]:
                original = next(row for row in direct["rows"] if row["gap_mm"]==gap and row["camera"]==label)
                own_correction, neighbour_correction = own_force@own_m, other_force@other_m
                correction = own_correction+neighbour_correction
                pred = original["predicted_Fz_N"]+correction[2]
                dh = incident+(self_k@own_m+cross_k@other_m).reshape(n, 3)
                ratio = np.linalg.norm(dh, axis=1)/np.maximum(hnorm, 1e-6)
                rows.append(dict(mode=mode, gap_mm=gap, camera=label,
                    direct_Fz_uN=original["direct_Fz_uN"], induced_Fz_uN=correction[2]*1e6,
                    own_induced_Fz_uN=own_correction[2]*1e6,
                    neighbour_induced_Fz_uN=neighbour_correction[2]*1e6,
                    one_way_own_response_Fz_uN=float(own_force[2]@(one_way_a if label=="A" else one_way_b))*1e6,
                    interaction_Fz_uN=original["direct_Fz_uN"]+correction[2]*1e6,
                    predicted_Fz_mN=pred*1e3, reference_Fz_mN=original["reference_Fz_mN"],
                    total_error_percent=abs(pred/original["reference_Fz_N"]-1)*100,
                    linear_response_iterations=len(linear_history), nonlinear_iterations=count,
                    response_H_perturbation_relative_quantiles=np.quantile(ratio,[.5,.95,1]).tolist(),
                    response_delta_M_norm_A_per_m=float(np.linalg.norm(own_m)),
                    linear_seconds=linear_seconds, nonlinear_seconds=nonlinear_seconds,
                    complete_response_seconds=perf_counter()-start))
        print(f"gap={gap:g}: linear mutual iterations={len(linear_history)} nonlinear iterations={count}", flush=True)
    for row in rows:
        anchor = next(item for item in rows if item["mode"]==row["mode"] and item["gap_mm"]==2 and item["camera"]==row["camera"])
        row["predicted_change_from_2mm_uN"] = row["interaction_Fz_uN"]-anchor["interaction_Fz_uN"]
        row["reference_change_from_2mm_uN"] = (row["reference_Fz_mN"]-anchor["reference_Fz_mN"])*1e3
        row["absolute_distance_change_error_uN"] = abs(row["predicted_change_from_2mm_uN"]-row["reference_change_from_2mm_uN"])
        one_way = row["direct_Fz_uN"]+row["one_way_own_response_Fz_uN"]
        anchor_one_way = anchor["direct_Fz_uN"]+anchor["one_way_own_response_Fz_uN"]
        row["one_way_change_from_2mm_uN"] = one_way-anchor_one_way
        row["one_way_vs_full_distance_change_difference_uN"] = abs(row["one_way_change_from_2mm_uN"]-row["predicted_change_from_2mm_uN"])
    np.savez_compressed(output / "incremental_states.npz", centers=centers, volumes=volumes, **states)
    # Resolve sampling independently of the native solution or volume-cell size.
    check_start = perf_counter()
    fine_own_force = force_response_operator(arrays["fine_points"], all_j[coarse_n:], arrays["fine_weights"], centers, volumes)
    for row in rows:
        mode = "nonlinear" if row["mode"]=="anchored-nonlinear" else "linear"
        own_m = states[f"gap_{row['gap_mm']:g}_{mode}_{row['camera']}"].ravel()
        row["own_response_quadrature_change_uN"] = abs((fine_own_force-own_force)[2]@own_m)*1e6
    pd.DataFrame(rows).to_csv(output / "prediction.csv", index=False)
    summary = dict(method="Maxwell isolated baseline plus coarse incremental polarization response",
        native_Maxwell_perturbation_solves_performed=False, paired_force_fit_applied=False,
        gaussian_volume_increment_approximation=True, cells=n,
        offline_response_precompute_seconds=offline, rows=rows,
        own_response_quadrature_check_seconds=perf_counter()-check_start,
        max_own_response_quadrature_change_uN=max(row["own_response_quadrature_change_uN"] for row in rows),
        max_one_way_vs_full_distance_change_difference_uN=max(row["one_way_vs_full_distance_change_difference_uN"] for row in rows if row["mode"]=="linear"),
        max_distance_change_error_uN_by_mode={mode:max(row["absolute_distance_change_error_uN"] for row in rows if row["mode"]==mode)
                                            for mode in ("linear", "anchored-nonlinear")},
        completed_UTC=datetime.now(timezone.utc).isoformat())
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["max_distance_change_error_uN_by_mode"], indent=2))


if __name__ == "__main__":
    main()
