from pathlib import Path
import json
import numpy as np
import pandas as pd
from camera_mag.geometry import load_camera
from camera_mag.materials import benchmark_materials
from camera_mag.solver import CameraPairSolver
from camera_mag.output import save_result


MODEL = Path(__file__).resolve().parents[1] / "models" / "B0500_scale_AF_VCM_EM_only.step"


def test_actual_step_pair_and_exports(tmp_path):
    camera = load_camera(MODEL, cache_dir=tmp_path)
    assert 500 <= len(camera.centers) <= 3000
    np.testing.assert_allclose(camera.extent * 1e3, [8.8, 8.8, 3.4])
    for oid, obj in enumerate(camera.metadata["objects"]):
        np.testing.assert_allclose(camera.volumes[camera.object_ids == oid].sum() * 1e9, obj["cad_volume_mm3"])
    solver = CameraPairSolver(camera, camera, benchmark_materials(), cache_dir=tmp_path)
    cold = solver.solve(2)
    assert cold.converged
    initial = solver.solve(1.5)
    warm = solver.solve(2, initial.state)
    assert warm.converged and warm.iterations < cold.iterations
    relative = np.linalg.norm(np.concatenate(warm.state) - np.concatenate(cold.state)) / np.linalg.norm(np.concatenate(cold.state))
    assert relative < 1e-5
    summary = save_result(solver, warm, tmp_path / "result")
    for name in ["state.npz", "features.parquet", "field.vtk", "field.png", "summary.json"]:
        assert (tmp_path / "result" / name).stat().st_size > 100
    features = pd.read_parquet(tmp_path / "result" / "features.parquet")
    assert len(features) == 2 * len(camera.centers)
    assert np.isfinite(features.select_dtypes("number")).all().all()
    assert summary["constitutive_relative_residual"] < .001
    assert summary["self_precompute_calls"] == 1
    for label in "AB":
        assert np.isfinite(summary["coil_force"][label]["total_N"]).all()
