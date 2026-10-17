"""Compare intended, analytic-source, and native-source loads at equal samples.

This is an excitation-fidelity check, not a camera response or distance-force
accuracy validation. No force is fitted, and no Maxwell solve is submitted.
"""
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from camera_mag.interaction import MU0,dipole_operator
from camera_mag.response_library import read_vector_field,integrate_force
from scripts.refine_pair_difference_reference import save,digest,utc


def main():
    started=perf_counter()
    output=ROOT/"outputs/native_equivalent_response_pilot_20261003"
    library=ROOT/"outputs/single_camera_reuse/refined_library_0.5"
    bank=ROOT/"outputs/equivalent_source_loading_20261003/sources_96_rcond_0.001.npz"
    snapshot=digest(bank)
    with np.load(output/"VacuumCheck/sampling.npz") as data:
        nb,nc=int(data["body_count"]),int(data["coil_count"])
        native_points=data["points_m"]
        points=native_points[:nb+nc]
        weights=data["coil_weights_m3"]
    with np.load(library/"quadrature.npz") as data:
        own=np.r_[data["coarse_points"],data["fine_points"]]
    with np.load(bank) as data:
        sources=data["centers_m"]+np.array([9.8e-3,0,0])
        moments=data["moments_A_m2"]
    native=read_vector_field(output/"VacuumCheck/H.fld",native_points)[:nb+nc]
    shift=np.array([9.8e-3,0,0])
    body=read_vector_field(library/"gap_1_A_cell_incident_H.fld",points[:nb]-shift)
    coil_b=read_vector_field(library/"gap_1_A_incident_B.fld",own-shift)[:nc]
    target=np.r_[body,coil_b/MU0]
    current=read_vector_field(library/"own_J.fld",own)[:nc]
    analytic=np.empty_like(target)
    with threadpool_limits(limits=4,user_api="blas"):
        for start in range(0,len(points),512):
            sl=slice(start,start+512)
            analytic[sl]=(dipole_operator(points[sl],sources,np.ones(len(sources)))@moments.ravel()).reshape(-1,3)
    rows={}
    for label,field in (("desired_native_neighbour",target),("analytic_source_basis",analytic),("native_finite_sources",native)):
        rows[label]=dict(relative_field_RMS_to_desired=float(np.linalg.norm(field-target)/np.linalg.norm(target)),
            direct_Fz_uN=float(integrate_force(current,MU0*field[nb:],weights)[2]*1e6))
        rows[label]["direct_Fz_difference_to_desired_uN"]=rows[label]["direct_Fz_uN"]-rows["desired_native_neighbour"]["direct_Fz_uN"]
    assert digest(bank)==snapshot
    save(output/"equal_sample_loading_comparison.json",dict(completed_UTC=utc(),seconds=perf_counter()-started,
        gap_mm=1,receiver="A",body_samples=nb,coil_samples=nc,rows=rows,
        native_minus_analytic_field_relative_RMS=float(np.linalg.norm(native-analytic)/np.linalg.norm(analytic)),
        native_minus_analytic_direct_Fz_uN=rows["native_finite_sources"]["direct_Fz_uN"]-rows["analytic_source_basis"]["direct_Fz_uN"],
        source_bank_sha256=snapshot,source_bank_unchanged=True,no_force_fit=True,
        same_coordinates_same_quadrature_same_frozen_native_current=True,
        limitation="Loading fidelity only; source-only one-pass native control is not adaptively converged and does not validate distance Delta F"))


if __name__=="__main__":main()
