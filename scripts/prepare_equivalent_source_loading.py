"""Fit an external loading basis to native fields, never to paired forces.

Point sources are confined to the neighbour envelope. This is a candidate
excitation representation, not a self-response model or a Maxwell-equivalent
solver. A native source-only loading test is still required before force trials.
"""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.linalg import lstsq
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from camera_mag.geometry import load_camera
from camera_mag.interaction import MU0,dipole_operator
from camera_mag.response_library import read_vector_field,integrate_force
from scripts.refine_pair_difference_reference import save,digest,utc


def source_order(points,count):
    selected=[int(np.argmax(np.linalg.norm(points-points.mean(0),axis=1)))]
    distance=np.full(len(points),np.inf)
    for _ in range(count-1):
        distance=np.minimum(distance,np.linalg.norm(points-points[selected[-1]],axis=1))
        selected.append(int(np.argmax(distance)))
    return np.asarray(selected)


def direct_force_operator(points,current,weights,sources,volumes):
    result=np.zeros((3,3*len(sources)))
    for start in range(0,len(points),512):
        sl=slice(start,start+512)
        kernel=MU0*dipole_operator(points[sl],sources,volumes).reshape(-1,3,3*len(sources))
        j=current[sl]
        cross=np.zeros((len(j),3,3))
        cross[:,0,1],cross[:,0,2]=-j[:,2],j[:,1]
        cross[:,1,0],cross[:,1,2]=j[:,2],-j[:,0]
        cross[:,2,0],cross[:,2,1]=-j[:,1],j[:,0]
        result+=np.einsum("p,pij,pjk->ik",weights[sl],cross,kernel,optimize=True)
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",default="outputs/equivalent_source_loading_20261003")
    args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    library=ROOT/"outputs/single_camera_reuse/refined_library_0.5"
    camera=load_camera(ROOT/"models/B0500_scale_AF_VCM_EM_only.step",cell_mm=.5)
    with np.load(library/"quadrature.npz") as q:
        coarse,fine,weights=q["coarse_points"],q["fine_points"],q["fine_weights"]
    own=np.r_[coarse,fine]
    current=read_vector_field(library/"own_J.fld",own)[len(coarse):]
    active=camera.material_ids!="copper"
    candidates=camera.centers[active]
    order=source_order(candidates,96)
    sources=candidates[order]
    volumes=np.full(len(sources),1e-9)  # coefficient normalization; moments are M*V
    body_train=np.arange(len(camera.centers))%4!=0
    coil_train=np.arange(7,len(fine),151)
    coil_validate=np.setdiff1d(np.arange(13,len(fine),157),coil_train)
    blocks=[]; hashes={};start=perf_counter()
    for gap in (1,2):
        for receiver in "AB":
            shift=np.array([camera.extent[0]+gap*1e-3,0,0])*(-1 if receiver=="A" else 1)
            cell_file=library/f"gap_{gap}_{receiver}_cell_incident_H.fld"
            coil_file=library/f"gap_{gap}_{receiver}_incident_B.fld"
            cell=read_vector_field(cell_file,camera.centers+shift)
            coil_b=read_vector_field(coil_file,own+shift)[len(coarse):]
            train_points=np.r_[camera.centers[body_train]+shift,fine[coil_train]+shift]
            validate_points=np.r_[camera.centers[~body_train]+shift,fine[coil_validate]+shift]
            train_h=np.r_[cell[body_train],coil_b[coil_train]/MU0]
            validate_h=np.r_[cell[~body_train],coil_b[coil_validate]/MU0]
            w=np.r_[np.full(body_train.sum(),.5/body_train.sum()),np.full(len(coil_train),.5/len(coil_train))]
            blocks.append(dict(gap=gap,receiver=receiver,train_points=train_points,validate_points=validate_points,
                train_h=train_h,validate_h=validate_h,weight=np.repeat(np.sqrt(w),3),
                true_force=integrate_force(current,coil_b,weights),fine_points=fine+shift))
            hashes[str(cell_file)]=digest(cell_file);hashes[str(coil_file)]=digest(coil_file)
    with threadpool_limits(limits=4,user_api="blas"):
        for block in blocks:
            block["train_op"]=dipole_operator(block["train_points"],sources,volumes)
            block["validate_op"]=dipole_operator(block["validate_points"],sources,volumes)
            block["force_op"]=direct_force_operator(block["fine_points"],current,weights,sources,volumes)
        matrix=np.concatenate([b["train_op"]*b["weight"][:,None] for b in blocks])
        target=np.concatenate([b["train_h"].ravel()*b["weight"] for b in blocks])
        rows=[];selected=None
        for count in (12,24,48,96):
            for tolerance in (1e-3,1e-5,1e-7):
                coef,_,rank,singular=lstsq(matrix[:,:3*count],target,cond=tolerance,lapack_driver="gelsd")
                metrics=[]
                for b in blocks:
                    prediction=(b["validate_op"][:,:3*count]@coef).reshape(-1,3)
                    fitted_force=b["force_op"][:,:3*count]@coef
                    metrics.append(dict(gap_mm=b["gap"],receiver=b["receiver"],
                        heldout_field_relative_RMS=float(np.linalg.norm(prediction-b["validate_h"])/np.linalg.norm(b["validate_h"])),
                        direct_Fz_representation_difference_uN=float((fitted_force[2]-b["true_force"][2])*1e6)))
                maximum_field=max(m["heldout_field_relative_RMS"] for m in metrics)
                maximum_force=max(abs(m["direct_Fz_representation_difference_uN"]) for m in metrics)
                coeff_m=coef.reshape(-1,3)
                row=dict(source_count=count,relative_svd_cutoff=tolerance,rank=int(rank),
                    max_heldout_field_relative_RMS=maximum_field,max_direct_Fz_difference_uN=maximum_force,
                    max_source_moment_A_m2=float(np.linalg.norm(coeff_m*volumes[:count,None],axis=1).max()),
                    field_representation_gate_met=maximum_field<=.02 and maximum_force<=.5,metrics=metrics)
                rows.append(row)
                np.savez_compressed(output/f"sources_{count}_rcond_{tolerance:g}.npz",
                    centers_m=sources[:count],moments_A_m2=coeff_m*volumes[:count,None])
                print(f"sources={count}, cutoff={tolerance:g}, field={maximum_field:.4%}, direct_Fz={maximum_force:.4f} uN",flush=True)
                if row["field_representation_gate_met"] and selected is None:
                    selected=dict(row,file=str(output/f"sources_{count}_rcond_{tolerance:g}.npz"))
    result=dict(completed_UTC=utc(),seconds=perf_counter()-start,source_hashes=hashes,
        source_hashes_unchanged=all(digest(Path(p))==h for p,h in hashes.items()),
        candidates=rows,selected=selected,paired_force_fit=False,
        coefficients_fit_to="Native neighbour H/B samples at 1/2 mm, both sides; no force objective",
        same_cached_samples_only=True,native_loading_validated=False,native_response_library_built=False,
        final_distance_accuracy_accepted=False,
        limitations=["Point-source loading basis only; physical native source realization still requires validation",
            "Field holdouts use the same frozen native field solution; not independent FEM accuracy checks",
            "No new camera perturbation solves, response coupling or pair-force accuracy acceptance"])
    save(output/"comparison.json",result)


if __name__=="__main__":
    main()
