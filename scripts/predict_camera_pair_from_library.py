"""Predict new 1–3 mm gaps from a saved single-camera field, without running Maxwell."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import lu_factor, lu_solve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_mag.geometry import load_camera
from camera_mag.interaction import MU0, KERNEL_VERSION, volume_operator
from camera_mag.response_library import read_vector_field, integrate_force, force_response_operator, native_susceptibility


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--library", default="outputs/single_camera_reuse/refined_library_0.5")
    p.add_argument("--gaps", type=float, nargs="+", default=[1, 1.5, 2, 2.5, 3])
    p.add_argument("--output", default="outputs/single_camera_reuse/new_gap_predictions_0.5.json")
    args = p.parse_args()
    if any(not np.isfinite(gap) or not 1<=gap<=3 for gap in args.gaps):
        raise ValueError("This prototype is restricted to same-camera +X edge gaps 1–3 mm")
    library = Path(args.library).resolve()
    record = json.loads((library / "export_record.json").read_text())
    if not record.get("completed_UTC"):
        raise ValueError("Single-camera field library is incomplete")
    spec = record["specification"]
    if hashlib.sha256((ROOT / "models/B0500_scale_AF_VCM_EM_only.step").read_bytes()).hexdigest()!=spec["step_sha256"]:
        raise ValueError("STEP geometry differs from library")
    camera = load_camera(ROOT / "models/B0500_scale_AF_VCM_EM_only.step", cell_mm=spec.get("cell_mm", .7))
    source = Path(spec["source"])
    for name in ("own_J", "own_cell_H"):
        if hashlib.sha256((library / (name+".fld")).read_bytes()).hexdigest()!=record["exports"][name]["sha256"]:
            raise ValueError("Native library field changed: "+name)
    if hashlib.sha256((source / "maxwell_bh.csv").read_bytes()).hexdigest()!=spec["source_BH_sha256"]:
        raise ValueError("Native B-H source changed")
    if hashlib.sha256((library / "exterior_grid.npz").read_bytes()).hexdigest()!=record["exterior_grid_npz_sha256"]:
        raise ValueError("Exterior field-grid data changed")
    if hashlib.sha256((source / "forces.json").read_bytes()).hexdigest()!=record["source_force_file_sha256"]:
        raise ValueError("Source native force changed")
    active = camera.material_ids != "copper"
    centers, volumes, ids = camera.centers[active], camera.volumes[active], camera.material_ids[active]
    n, size = len(centers), centers.size
    arrays = np.load(library / "quadrature.npz")
    all_points = np.r_[arrays["coarse_points"], arrays["fine_points"]]
    current = read_vector_field(library / "own_J.fld", all_points)
    coarse_count = len(arrays["coarse_points"])
    points, weights, current_coarse = arrays["coarse_points"], arrays["coarse_weights"], current[:coarse_count]
    fine_points, fine_weights, current_fine = arrays["fine_points"], arrays["fine_weights"], current[coarse_count:]
    h0 = read_vector_field(library / "own_cell_H.fld", camera.centers)[active]
    grid = np.load(library / "exterior_grid.npz")
    neg = RegularGridInterpolator((grid["xneg"], grid["y"], grid["z"]), grid["Bneg"], bounds_error=True)
    pos = RegularGridInterpolator((grid["xpos"], grid["y"], grid["z"]), grid["Bpos"], bounds_error=True)
    forces = json.loads((source / "forces.json").read_text())
    isolated = np.array([forces[f"Force_A.Force_{axis}"]["values"][0] for axis in "xyz"])
    cache_inputs = [library / "own_J.fld", library / "own_cell_H.fld", library / "quadrature.npz", source / "maxwell_bh.csv"]
    cache_hash = hashlib.sha256((KERNEL_VERSION+str(spec.get("cell_mm", .7))+"single-response-v1").encode())
    for path in cache_inputs:
        cache_hash.update(hashlib.sha256(path.read_bytes()).digest())
    cache_hash.update(spec["step_sha256"].encode())
    key = cache_hash.hexdigest()
    cache = library / ("incremental_operator_"+key+".npz")
    start = perf_counter()
    with threadpool_limits(limits=4, user_api="blas"):
        if cache.exists():
            with np.load(cache) as data:
                local, factor, pivots, own_force = [data[name] for name in ("local", "factor", "pivots", "own_force")]
            cache_hit = True
        else:
            table = np.loadtxt(source / "maxwell_bh.csv", delimiter=",", skiprows=1)
            tangent = native_susceptibility(h0, ids, table)
            local = np.linalg.solve(np.eye(3)+tangent/3, tangent)
            self_k = volume_operator(centers, centers, volumes, target_volumes=volumes)+np.eye(size)/3
            matrix = np.eye(size)-np.einsum("nij,njmk->nimk", local, self_k.reshape(n, 3, n, 3)).reshape(size,size)
            factor, pivots = lu_factor(matrix)
            own_force = force_response_operator(points, current_coarse, weights, centers, volumes)
            np.savez_compressed(cache, local=local, factor=factor, pivots=pivots, own_force=own_force)
            cache_hit = False
        offline_seconds = perf_counter()-start
        rows = []
        for gap in args.gaps:
            begin = perf_counter()
            shift = np.array([camera.extent[0]+gap*1e-3, 0, 0])
            incident_a, incident_b = neg(centers-shift)/MU0, pos(centers+shift)/MU0
            direct_a = integrate_force(current_fine, neg(fine_points-shift), fine_weights)
            direct_b = integrate_force(current_fine, pos(fine_points+shift), fine_weights)
            kab = volume_operator(centers, centers+shift, volumes, target_volumes=volumes)
            kba = (kab.reshape(n,3,n,3).transpose(2,3,0,1)*(volumes[None,None,:,None]/volumes[:,None,None,None])).reshape(size,size)
            ma, mb = np.zeros(size), np.zeros(size)
            for iteration in range(60):
                a = lu_solve((factor,pivots), np.einsum("nij,nj->ni", local, incident_a+(kab@mb).reshape(n,3)).ravel())
                b = lu_solve((factor,pivots), np.einsum("nij,nj->ni", local, incident_b+(kba@ma).reshape(n,3)).ravel())
                residual = max(np.linalg.norm(a-ma),np.linalg.norm(b-mb))/max(np.linalg.norm(a),np.linalg.norm(b),1)
                ma, mb = a,b
                if residual < 1e-8:
                    break
            if residual >= 1e-8:
                raise RuntimeError("Incremental mutual response did not converge")
            cross_key = hashlib.sha256((key+float(gap).hex()).encode()).hexdigest()
            cross_cache = library / ("cross_force_"+cross_key+".npz")
            cross_cache_hit = cross_cache.exists()
            if cross_cache_hit:
                with np.load(cross_cache) as data:
                    other_a, other_b = data["A"], data["B"]
            else:
                other_a = force_response_operator(points,current_coarse,weights,centers+shift,volumes)
                other_b = force_response_operator(points,current_coarse,weights,centers-shift,volumes)
                np.savez_compressed(cross_cache,A=other_a,B=other_b)
            correction_a, correction_b = own_force@ma+other_a@mb, own_force@mb+other_b@ma
            rows.append(dict(gap_mm=gap, A_force_N=(isolated+direct_a+correction_a).tolist(),
                B_force_N=(isolated+direct_b+correction_b).tolist(),
                A_direct_force_N=direct_a.tolist(), B_direct_force_N=direct_b.tolist(),
                A_response_force_N=correction_a.tolist(), B_response_force_N=correction_b.tolist(),
                response_iterations=iteration+1, response_residual=float(residual),
                cross_force_cache_hit=cross_cache_hit,
                new_gap_query_seconds=perf_counter()-begin))
            print(f"gap={gap:g} mm: A_Fz={rows[-1]['A_force_N'][2]*1e3:.6f} mN, query={rows[-1]['new_gap_query_seconds']:.2f}s", flush=True)
    result = dict(library=str(library), method="Single-camera native field interpolation plus incremental volume response",
        paired_force_data_read=False, Maxwell_started=False, current_A=spec["current_a"], turns=spec["turns"],
        same_camera_model_only=True, cache_hit=cache_hit, operator_load_or_build_seconds=offline_seconds,
        field_grid_spacing_mm=.2, rows=rows)
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
