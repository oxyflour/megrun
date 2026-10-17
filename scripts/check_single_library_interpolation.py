"""Check saved air-grid interpolation against native exports and full prototype queries."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from camera_mag.geometry import load_camera
from camera_mag.interaction import MU0
from camera_mag.response_library import read_vector_field, integrate_force


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--library", default="outputs/single_camera_reuse/refined_library")
    parser.add_argument("--queries", default="outputs/single_camera_reuse/new_gap_predictions.json")
    parser.add_argument("--increment", default="outputs/single_camera_reuse/refined_prediction/incremental/summary.json")
    parser.add_argument("--output", default="outputs/single_camera_reuse/interpolation_check.json")
    args = parser.parse_args()
    out = ROOT / "outputs/single_camera_reuse"
    library = Path(args.library).resolve()
    record = json.loads((library / "export_record.json").read_text())
    spec = record["specification"]
    camera = load_camera(ROOT / "models/B0500_scale_AF_VCM_EM_only.step", cell_mm=spec.get("cell_mm",.7))
    q = np.load(library / "quadrature.npz")
    all_points = np.r_[q["coarse_points"], q["fine_points"]]
    current = read_vector_field(library / "own_J.fld",all_points)
    offset = len(q["coarse_points"])
    grid = np.load(library / "exterior_grid.npz")
    interpolators = {"A":RegularGridInterpolator((grid["xneg"],grid["y"],grid["z"]),grid["Bneg"]),
                     "B":RegularGridInterpolator((grid["xpos"],grid["y"],grid["z"]),grid["Bpos"])}
    direct = json.loads((out / "refined_prediction/direct_prediction_summary.json").read_text())
    native_increment = json.loads(Path(args.increment).read_text())
    queries = json.loads(Path(args.queries).read_text())
    rows=[]
    for gap in spec["gaps_mm"]:
        shift = np.array([camera.extent[0]+gap*1e-3,0,0])
        query = next(row for row in queries["rows"] if row["gap_mm"]==gap)
        for label, displacement in [("A",-shift),("B",shift)]:
            interpolator = interpolators[label]
            b = interpolator(q["fine_points"]+displacement)
            pred = integrate_force(current[offset:],b,q["fine_weights"])
            original = next(row for row in direct["rows"] if row["camera"]==label and row["gap_mm"]==gap)
            inc = next(row for row in native_increment["rows"] if row["camera"]==label and row["gap_mm"]==gap and row["mode"]=="linear")
            original_h = read_vector_field(library / f"gap_{gap:g}_{label}_cell_incident_H.fld",camera.centers+displacement)
            h_interp = interpolator(camera.centers+displacement)/MU0
            rows.append(dict(gap_mm=gap,camera=label,
                direct_force_interpolation_difference_uN=abs(pred[2]-original["direct_Fz_N"])*1e6,
                incident_H_rms_relative_error_percent=np.linalg.norm(h_interp-original_h)/np.linalg.norm(original_h)*100,
                full_prototype_vs_native_queries_Fz_difference_uN=abs(query[f"{label}_force_N"][2]*1e3-inc["predicted_Fz_mN"])*1e3))
    summary=dict(library=str(library), query_file=str(Path(args.queries).resolve()), rows=rows,
        max_direct_force_interpolation_difference_uN=max(row["direct_force_interpolation_difference_uN"] for row in rows),
        max_full_prototype_vs_native_queries_difference_uN=max(row["full_prototype_vs_native_queries_Fz_difference_uN"] for row in rows))
    summary["interpolation_comparison_within_2uN"] = summary["max_full_prototype_vs_native_queries_difference_uN"]<=2
    Path(args.output).write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
