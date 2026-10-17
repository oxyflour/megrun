"""Collect measured slide evidence and tessellate the supplied CAD, without solving."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_mag.geometry import load_camera, EXPECTED


def main():
    import cadquery as cq
    output = ROOT / "outputs/two_ideas_slides"
    output.mkdir(parents=True, exist_ok=True)
    source_names = {
        "matched": "outputs/mesh_seeding_2mm/comparison_summary.json",
        "onepass": "outputs/mesh_seeding_onepass_holdouts/summary.json",
        "mesh_plan": "outputs/mesh_seeding_onepass_2mm/Aggressive/mesh_plan.json",
        "reuse": "outputs/single_camera_reuse/summary.json",
        "direct": "outputs/single_camera_reuse/refined_prediction/direct_prediction_summary.json",
        "increment": "outputs/single_camera_reuse/refined_prediction/incremental_0.5/summary.json",
        "coarse_increment": "outputs/single_camera_reuse/refined_prediction/incremental/summary.json",
        "interpolation": "outputs/single_camera_reuse/interpolation_check.json",
        "queries": "outputs/single_camera_reuse/new_gap_predictions_0.5.json",
        "cached": "outputs/single_camera_reuse/new_gap_predictions_0.5_cached.json",
        "cli": "outputs/single_camera_reuse/cached_cli_runtime_0.5.json",
        "frozen": "outputs/single_camera_reuse/reused_single/SingleFieldReference/frozen_source_audit.json",
        "build": "outputs/single_camera_reuse/reused_single/build_record.json",
        "native_record": "outputs/single_camera_reuse/reused_validation_record.json",
    }
    data = {key: json.loads((ROOT / path).read_text(encoding="utf-8")) for key, path in source_names.items()}
    # The full summary includes large duplicate fields; keep the measured rows
    # and the provenance needed by the slides, never invent new simulation data.
    data["reuse"] = {key: data["reuse"][key] for key in (
        "experiments", "max_response_cell_refinement_distance_change_uN",
        "stopped_from_scratch_trial_seconds", "refined_isolated_stages_seconds")}
    data["frozen"] = {key: value for key, value in data["frozen"].items() if key != "hashes"}
    data["mesh_plan"] = {key: data["mesh_plan"][key] for key in (
        "face_scores", "fine_area_actual_fraction", "fine_and_medium_area_actual_fraction",
        "preprocessing_seconds", "coarse_cell_mm", "source")}
    step = ROOT / "models/B0500_scale_AF_VCM_EM_only.step"
    camera = load_camera(step)
    assembly = cq.Assembly.importStep(str(step))
    cad_faces = []

    def visit(node, parent_location):
        location = parent_location * node.loc
        if node.obj is not None and node.name in EXPECTED:
            raw = node.obj.val() if isinstance(node.obj, cq.Workplane) else node.obj
            solid = raw.located(location * raw.location())
            material = "copper" if "COIL" in node.name else "yoke" if "YOKE" in node.name else "magnet"
            for face in solid.Faces():
                vertices, indices = face.tessellate(.02, .15)
                cad_faces.append(dict(object=node.name, material=material,
                    center_mm=list(face.Center().toTuple()), area_mm2=float(face.Area()),
                    vertices_mm=[list(point.toTuple()) for point in vertices], triangles=indices))
        for child in node.children:
            visit(child, location)

    visit(assembly, cq.Location())
    if {face["object"] for face in cad_faces} != EXPECTED:
        raise ValueError("CAD slide evidence lost one of the named physical bodies")
    # Map the actual native seed records onto matching CAD faces, by object and
    # measured face centre. A mismatch is visible as an unassigned face.
    records = data["mesh_plan"]["face_scores"]
    match_count = 0
    for face in cad_faces:
        if face["material"] == "copper":
            face["seed_level"] = "coil"
            continue
        group = [item for item in records if item["object"] == "A_"+face["object"]]
        if not group:
            raise ValueError("Native mesh record missing a CAD body")
        nearest = min(group, key=lambda item: np.linalg.norm(np.array(item["center_mm"])-face["center_mm"]))
        error = float(np.linalg.norm(np.array(nearest["center_mm"])-face["center_mm"]))
        face["seed_match_distance_mm"] = error
        face["seed_level"] = nearest["level"] if error < 1e-5 else "unassigned"
        face["seed_score"] = nearest["score"]
        match_count += error < 1e-5
    data["cad"] = dict(faces=cad_faces, bounds_mm=(np.asarray(camera.metadata["bounds_m"])*1000).tolist(),
        matched_seed_faces=match_count, seed_faces_total=sum(face["material"]!="copper" for face in cad_faces),
        vertices_from_STEP=True, tetrahedral_mesh=False)
    if match_count != data["cad"]["seed_faces_total"]:
        raise ValueError("Mesh colour overlay does not exactly match native face-centre records")
    state_path = ROOT / "outputs/single/state.npz"
    with np.load(state_path) as state:
        # One closest-to-midplane sample per XY position keeps this a faithful
        # discrete field plot rather than a fabricated smooth field image.
        points = []
        for label in "AB":
            xyz = state["centers_"+label]*1000
            bnorm = np.linalg.norm(state["B_"+label], axis=1)
            candidates = {}
            for index, point in enumerate(xyz):
                key = tuple(np.round(point[:2], 7))
                previous = candidates.get(key)
                if previous is None or abs(point[2]-3) < abs(xyz[previous,2]-3):
                    candidates[key] = index
            points += [[*xyz[index].tolist(), float(bnorm[index])] for index in candidates.values()]
        data["lf_field"] = dict(points=points, gap_mm=float(state["distance_mm"]), unit="T", method="low fidelity")
    grid_path = ROOT / "outputs/single_camera_reuse/refined_library_0.5/exterior_grid.npz"
    with np.load(grid_path) as grid:
        zi = int(np.argmin(abs(grid["z"]-.003)))
        data["native_air"] = dict(x_mm=(grid["xpos"]*1000).tolist(), y_mm=(grid["y"]*1000).tolist(),
            z_mm=float(grid["z"][zi]*1000), B_mT=(np.linalg.norm(grid["Bpos"][:,:,zi],axis=-1)*1000).tolist(),
            field="isolated native Maxwell B in air")
    source_names.update(step=str(step.relative_to(ROOT)), lf_field=str(state_path.relative_to(ROOT)),
                        native_air=str(grid_path.relative_to(ROOT)))
    data["sources"] = [{"key":key,"path":path,"sha256":hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}
                       for key,path in source_names.items()]
    data["meta"] = dict(title="摄像头磁仿真的两个加速思路", date="2026-10-03", current_A=.1, turns=100,
        same_model_same_orientation=True, figure_scope="CAD surfaces, discrete field samples and measured curves",
        native_Maxwell_solve_performed=False)
    target = output / "slide_data.json"
    target.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(dict(data_file=str(target), bytes=target.stat().st_size,
        CAD_faces=len(cad_faces), matched_seed_faces=match_count,
        native_solve_performed=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
