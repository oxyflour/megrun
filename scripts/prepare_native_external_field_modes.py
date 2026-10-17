"""Test source-free polynomial representations of cached native neighbour H.

Fits field samples only, never paired forces. This is an excitation feasibility
check, not a magnetic solver or proof of a Maxwell response-library accuracy.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import null_space

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_mag.geometry import load_camera
from camera_mag.interaction import MU0
from camera_mag.response_library import read_vector_field, integrate_force


def exponents(degree):
    return [(x, y, degree-x-y) for x in range(degree+1) for y in range(degree-x+1)]


def harmonic_basis(max_degree):
    all_terms, blocks = [], []
    for degree in range(1, max_degree+1):
        terms = exponents(degree)
        if degree < 2:
            basis = np.eye(len(terms))
        else:
            lower = exponents(degree-2)
            laplace = np.zeros((len(lower), len(terms)))
            for col, term in enumerate(terms):
                for axis in range(3):
                    if term[axis] >= 2:
                        reduced = list(term)
                        reduced[axis] -= 2
                        laplace[lower.index(tuple(reduced)), col] += term[axis]*(term[axis]-1)
            basis = null_space(laplace)
            if np.max(np.abs(laplace @ basis), initial=0) > 1e-12:
                raise ValueError("Nonharmonic basis")
        all_terms.extend(terms)
        blocks.append(basis)
    result = np.zeros((len(all_terms), sum(b.shape[1] for b in blocks)))
    row = col = 0
    for block in blocks:
        result[row:row+block.shape[0], col:col+block.shape[1]] = block
        row += block.shape[0]
        col += block.shape[1]
    return np.asarray(all_terms, dtype=int), result


def monomial_field(q, terms):
    field = np.zeros((len(q), 3, len(terms)))
    for col, term in enumerate(terms):
        for axis in range(3):
            if term[axis]:
                reduced = term.copy()
                reduced[axis] -= 1
                field[:, axis, col] = -term[axis] * np.prod(q ** reduced, axis=1)
    return field


def expression(terms, coefficients, axis, center, scale):
    symbols = [f"(({name}-({center[i]*1e3:.12g}mm))/({scale*1e3:.12g}mm))" for i,name in enumerate("XYZ")]
    parts = []
    for term, coefficient in zip(terms, coefficients):
        if not term[axis] or abs(coefficient) < 1e-12:
            continue
        power = term.copy()
        power[axis] -= 1
        factors = [symbols[i] if p==1 else f"({symbols[i]})^{p}" for i,p in enumerate(power) if p]
        value = -coefficient * term[axis]
        parts.append(f"({value:.16g}A_per_meter)" + ("*" + "*".join(factors) if factors else ""))
    return "+".join(parts) or "0A_per_meter"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--library", default="outputs/single_camera_reuse/refined_library_0.5")
    parser.add_argument("--output", default="outputs/delta_precision_20261003/external_modes")
    parser.add_argument("--max-degree", type=int, default=6)
    args = parser.parse_args()
    library, output = ROOT / args.library, ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    camera = load_camera(ROOT / "models/B0500_scale_AF_VCM_EM_only.step", cell_mm=.5)
    with np.load(library / "quadrature.npz") as quadrature:
        own_points = np.concatenate([quadrature[label + "_points"] for label in ("coarse", "fine")])
        fine_points, fine_weights = quadrature["fine_points"], quadrature["fine_weights"]
    current = read_vector_field(library / "own_J.fld", own_points)[-len(fine_points):]
    center = np.mean(np.asarray(camera.metadata["bounds_m"]), axis=0)
    scale = .0044
    # Deterministic withheld native samples test representation, not new FEM discretization.
    training_cells = np.arange(len(camera.centers)) % 4 != 0
    coil_training = np.arange(7, len(fine_points), 29)
    coil_validation = np.arange(13, len(fine_points), 31)
    q_train = (np.concatenate([camera.centers[training_cells], fine_points[coil_training]]) - center)/scale
    q_validate = (np.concatenate([camera.centers[~training_cells], fine_points[coil_validation]]) - center)/scale
    q_full_coil = (fine_points-center)/scale
    corners = np.asarray(np.meshgrid([-24.4,24.4],[-24.4,24.4],[-18.6,24.8], indexing="ij")).reshape(3,-1).T * 1e-3
    q_corners = (corners-center)/scale
    # Balance body and conductor field data; field least squares contains no force objective.
    weights = np.r_[np.full(training_cells.sum(), .5/training_cells.sum()), np.full(len(coil_training), .5/len(coil_training))]
    root_weights = np.repeat(np.sqrt(weights), 3)
    record = dict(started_UTC=datetime.now(timezone.utc).isoformat(), field_source=str(library),
                  paired_force_data_read=False, force_fit_applied=False,
                  source_free_by_construction=True, same_cached_field_samples_only=True,
                  purpose="External excitation feasibility; not a native response solve", modes=[])
    for gap in (1,2):
        for label in "AB":
            cell_file = library / f"gap_{gap}_{label}_cell_incident_H.fld"
            coil_file = library / f"gap_{gap}_{label}_incident_B.fld"
            shift = np.array([camera.extent[0]+gap*1e-3,0,0]) * (-1 if label=="A" else 1)
            body_h = read_vector_field(cell_file, camera.centers+shift)
            coil_h = read_vector_field(coil_file, own_points+shift)[-len(fine_points):]/MU0
            h_train = np.concatenate([body_h[training_cells], coil_h[coil_training]])
            h_validate = np.concatenate([body_h[~training_cells], coil_h[coil_validation]])
            true_force = integrate_force(current, coil_h*MU0, fine_weights)
            row = dict(gap_mm=gap, receiver=label, source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (cell_file,coil_file)}, candidates=[])
            for degree in range(1,args.max_degree+1):
                terms, basis = harmonic_basis(degree)
                train_matrix = (monomial_field(q_train,terms) @ basis).reshape(-1,basis.shape[1])
                fitted, _, rank, singular = np.linalg.lstsq(train_matrix*root_weights[:,None], h_train.ravel()*root_weights, rcond=1e-12)
                coefficients = basis @ fitted
                prediction = (monomial_field(q_validate,terms) @ coefficients)
                relative_rms = float(np.linalg.norm(prediction-h_validate)/np.linalg.norm(h_validate))
                # Full quadrature is a separate force diagnostic, not a fit target.
                full_force = integrate_force(current, (monomial_field(q_full_coil,terms) @ coefficients)*MU0, fine_weights)
                boundary_max_b = float(np.linalg.norm((monomial_field(q_corners,terms) @ coefficients)*MU0,axis=1).max())
                candidate = dict(potential_degree=degree, harmonic_modes=basis.shape[1], rank=int(rank),
                    condition_number=float(singular[0]/singular[-1]), heldout_field_relative_RMS=relative_rms,
                    direct_Fz_representation_difference_uN=float((full_force[2]-true_force[2])*1e6),
                    corner_max_incident_B_T=boundary_max_b,
                    passes_provisional_excitation_gate=relative_rms <= .02 and abs(full_force[2]-true_force[2])*1e6 <= .5 and boundary_max_b <= .2)
                row["candidates"].append(candidate)
                np.savez_compressed(output / f"gap{gap}_{label}_degree{degree}.npz", terms=terms, coefficients=coefficients,
                                    center_m=center, scale_m=scale)
                if candidate["passes_provisional_excitation_gate"] and "selected_degree" not in row:
                    row["selected_degree"] = degree
                    row["H_expressions"] = [expression(terms,coefficients,axis,center,scale) for axis in range(3)]
            row["excitation_representation_accepted"] = "selected_degree" in row
            record["modes"].append(row)
            print(f"gap {gap}, {label}: " + json.dumps(row["candidates"][-1]), flush=True)
    record.update(completed_UTC=datetime.now(timezone.utc).isoformat(), native_Maxwell_solve_performed=False)
    (output / "mode_feasibility.json").write_text(json.dumps(record,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
