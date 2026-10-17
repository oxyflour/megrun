"""Audit disjoint field holdouts without modifying the selected source bank."""
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from camera_mag.geometry import load_camera
from camera_mag.interaction import MU0,dipole_operator
from camera_mag.response_library import read_vector_field
from scripts.refine_pair_difference_reference import read,save,digest,utc


def main():
    output=ROOT/"outputs/equivalent_source_loading_20261003"
    bank=output/"sources_96_rcond_0.001.npz"
    bank_hash=digest(bank)
    library=ROOT/"outputs/single_camera_reuse/refined_library_0.5"
    camera=load_camera(ROOT/"models/B0500_scale_AF_VCM_EM_only.step",cell_mm=.5)
    with np.load(library/"quadrature.npz") as q:
        coarse,fine=q["coarse_points"],q["fine_points"]
    with np.load(bank) as data:
        sources,moments=data["centers_m"],data["moments_A_m2"]
    coil_train=np.arange(7,len(fine),151)
    initial_validate=np.arange(13,len(fine),157)
    coil_validate=np.setdiff1d(initial_validate,coil_train)
    body_validate=np.arange(len(camera.centers))%4==0
    assert not len(np.intersect1d(coil_train,coil_validate))
    rows=[]
    with threadpool_limits(limits=4,user_api="blas"):
        for gap in (1,2):
            for receiver in "AB":
                shift=np.array([camera.extent[0]+gap*1e-3,0,0])*(-1 if receiver=="A" else 1)
                points=np.r_[camera.centers[body_validate]+shift,fine[coil_validate]+shift]
                body=read_vector_field(library/f"gap_{gap}_{receiver}_cell_incident_H.fld",camera.centers+shift)
                coil_b=read_vector_field(library/f"gap_{gap}_{receiver}_incident_B.fld",np.r_[coarse,fine]+shift)[len(coarse):]
                reference=np.r_[body[body_validate],coil_b[coil_validate]/MU0]
                prediction=(dipole_operator(points,sources,np.ones(len(sources)))@moments.ravel()).reshape(-1,3)
                rows.append(dict(gap_mm=gap,receiver=receiver,
                    disjoint_heldout_field_relative_RMS=float(np.linalg.norm(prediction-reference)/np.linalg.norm(reference))))
    assert digest(bank)==bank_hash
    save(output/"heldout_audit.json",dict(completed_UTC=utc(),source_bank_sha256=bank_hash,
        source_bank_unchanged=True,removed_overlapping_coil_samples=int(len(initial_validate)-len(coil_validate)),
        body_holdout_count=int(body_validate.sum()),coil_holdout_count=len(coil_validate),
        all_field_holdouts_disjoint_from_training=True,
        max_disjoint_heldout_field_relative_RMS=max(r["disjoint_heldout_field_relative_RMS"] for r in rows),
        metrics=rows,coefficients_refit=False,force_objective_used=False,
        limitation="Holdouts sample the same frozen native field solution, not independent FEM accuracy"))


if __name__=="__main__":main()
