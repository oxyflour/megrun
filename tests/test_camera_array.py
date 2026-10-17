"""Physics regression and orchestration guards for the N-camera workflow."""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

from camera_mag.array_config import load_config, monitor_values, moved_config, normalize_config, validate_layout, write_json
from camera_mag.array_maxwell import simulation_gate, snapshot_native
from camera_mag.array_search import accept_selection, candidate_search, improving_candidates
from camera_mag.array_solver import CameraArraySolver
from camera_mag.array_workflow import run_workflow
from camera_mag.coil import coil_forces
from camera_mag.geometry import load_camera
from camera_mag.materials import benchmark_materials
from camera_mag.solver import CameraPairSolver

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def camera():
    return load_camera(ROOT / "models/B0500_scale_AF_VCM_EM_only.step", cache_dir=ROOT / ".cache")


def config():
    return load_config(ROOT / "examples/camera_array_B.json")


def test_two_camera_force_and_state_match_existing_pair(camera, tmp_path):
    cfg = config()
    cfg["N"], cfg["cameras"] = 2, cfg["cameras"][:2]
    cfg["monitors"] = [{"id": "F", "quantity": "F", "camera_id": "C1", "component": "z"}]
    cfg = normalize_config(cfg)
    array = CameraArraySolver(camera, benchmark_materials(), cache_dir=tmp_path)
    report, result = array.evaluate(cfg)
    pair = CameraPairSolver(camera, camera, benchmark_materials(), cache_dir=tmp_path)
    old = pair.solve(2.)
    assert old.converged and result["converged"]
    np.testing.assert_allclose(result["state"], np.asarray(old.state), rtol=2e-8, atol=1e-7)
    force = coil_forces(pair, old, pair.isolated_states())["A"]["total_N"]
    np.testing.assert_allclose(report["monitors"][0]["vector"], force, rtol=2e-8, atol=1e-12)


def test_three_camera_covariance_and_source_sum(camera, tmp_path):
    cfg = config()
    solver = CameraArraySolver(camera, benchmark_materials(), cache_dir=tmp_path)
    report, result = solver.evaluate(cfg)
    assert result["converged"] and report["constitutive_relative_residual"] < 1e-3
    summed = np.array([r["vectors"] for r in report["sources"]]).sum(axis=0)
    np.testing.assert_allclose(summed, [r["vector"] for r in report["monitors"]], atol=1e-17)
    shifted = deepcopy(cfg)
    shift = np.array([3., -2., 1.])
    for c in shifted["cameras"]:
        c["position_mm"] = (c["position_mm"] + shift).tolist()
        c["bounds_mm"] = (c["bounds_mm"] + shift).tolist()
    for m in shifted["monitors"]:
        m["position_mm"] = (m["position_mm"] + shift).tolist()
    moved, new = solver.evaluate(shifted)
    np.testing.assert_allclose(new["state"], result["state"], rtol=3e-8, atol=1e-6)
    np.testing.assert_allclose(moved["score"], report["score"], rtol=2e-9)
    # A removed magnet may increase the monitor magnitude: ranking must
    # retain signed cancellation evidence rather than claiming positive harm.
    assert any(r["frozen_removal_score_change"] > 0 for r in report["magnets"])


def test_single_camera_matches_isolated_state(camera, tmp_path):
    cfg = config()
    cfg["N"], cfg["cameras"] = 1, cfg["cameras"][:1]
    cfg = normalize_config(cfg)
    array = CameraArraySolver(camera, benchmark_materials(), cache_dir=tmp_path)
    _, result = array.evaluate(cfg)
    pair = CameraPairSolver(camera, camera, benchmark_materials(), cache_dir=tmp_path)
    np.testing.assert_allclose(result["state"][0], pair.isolated_states()[0], rtol=2e-8, atol=1e-7)


def test_default_count_bounds_and_locked_motion(camera):
    cfg = config()
    del cfg["N"]
    assert normalize_config(cfg)["N"] == 3
    cfg["cameras"].pop()
    with pytest.raises(ValueError, match="N"):
        normalize_config(cfg)
    cfg = config()
    with pytest.raises(ValueError, match="fixed"):
        moved_config(cfg, {"C1": [.1, 0, 0]})
    with pytest.raises(ValueError, match="locked"):
        moved_config(cfg, {"C2": [10.8, 0, .1]})
    with pytest.raises(ValueError, match="step_mm"):
        moved_config(cfg, {"C2": [11.4, 0, 0]})
    moved = moved_config(cfg, {"C2": [11.3, 0, 0]})
    assert moved["cameras"][1]["bounds_mm"] == cfg["cameras"][1]["bounds_mm"]
    for _ in range(3):
        moved = moved_config(moved, {"C2": [moved["cameras"][1]["position_mm"][0] + .5, 0, 0]})
    with pytest.raises(ValueError, match="outside"):
        moved_config(moved, {"C2": [13.3, 0, 0]})
    overlap = config()
    overlap["cameras"][1]["position_mm"] = [1, 1, 0]
    with pytest.raises(ValueError, match="overlap"):
        validate_layout(overlap, camera)


def test_monitor_units_targets_and_no_point_force():
    cfg = config()
    cfg["monitors"] = [{"id": "B", "quantity": "B", "position_mm": [0, 0, 10], "scale": .001},
                       {"id": "F", "quantity": "F", "camera_id": "C1", "component": "z", "scale": 1e-5, "target": .03}]
    cfg = normalize_config(cfg)
    rows, score = monitor_values(cfg, [[0, .002, 0], [0, 0, .03003]])
    assert rows[0]["unit"] == "T" and rows[1]["unit"] == "N"
    assert score == pytest.approx(3.)
    invalid = deepcopy(cfg)
    del invalid["monitors"][1]["scale"]
    with pytest.raises(ValueError, match="explicit"):
        normalize_config(invalid)
    cfg["monitors"][1]["position_mm"] = [1, 2, 3]
    with pytest.raises(ValueError, match="camera_id"):
        normalize_config(cfg)


def test_search_and_agent_gate(camera, tmp_path):
    cfg = config()
    cfg["search"]["max_candidates"] = 4
    solver = CameraArraySolver(camera, benchmark_materials(), cache_dir=tmp_path)
    baseline, state = solver.evaluate(cfg)
    search = candidate_search(solver, cfg, baseline, state, deadline=perf_counter() + 45)
    choices = improving_candidates(cfg, search)
    assert choices
    next_cfg, audit = accept_selection(cfg, search, {"candidate_id": choices[0]["id"], "reason": "test"})
    verified, _ = solver.evaluate(next_cfg)
    assert verified["score"] == pytest.approx(choices[0]["score"], rel=1e-6)
    assert audit["native_verified"] is False
    assert next_cfg["monitors"] == cfg["monitors"]
    with pytest.raises(ValueError, match="non-improving"):
        accept_selection(cfg, search, {"candidate_id": "candidate_001", "reason": "bad"})
    solver.deadline = perf_counter() - 1
    with pytest.raises(TimeoutError):
        solver.evaluate(cfg)
    too_big = CameraArraySolver(camera, benchmark_materials(), cache_dir=tmp_path, max_operator_mb=.1)
    with pytest.raises(MemoryError):
        too_big.evaluate(cfg)


def test_whole_campaign_budget_and_no_submission(tmp_path):
    with pytest.raises(ValueError, match="reliable"):
        simulation_gate(estimated_minutes=None, estimate_basis=None, budget_minutes=25)
    with pytest.raises(PermissionError):
        simulation_gate(estimated_minutes=31, estimate_basis="measured", budget_minutes=40)
    with pytest.raises(PermissionError):
        simulation_gate(estimated_minutes=29, estimate_basis="measured", budget_minutes=40)
    with pytest.raises(ValueError, match="cover"):
        simulation_gate(estimated_minutes=25, estimate_basis="measured", budget_minutes=20)
    assert simulation_gate(estimated_minutes=31, estimate_basis="measured", budget_minutes=40, confirmed_long=True)["explicit_long_confirmation"]
    output = tmp_path / "blocked"
    with pytest.raises(ValueError):
        run_workflow(ROOT / "examples/camera_array_B.json", output, mode="run")
    assert not output.exists()
    with pytest.raises(ValueError, match="surrogate_only"):
        run_workflow(ROOT / "examples/camera_array_B.json", output, mode="build", surrogate_only=True)
    assert not output.exists()


def test_native_snapshot_survives_extraction_mutation(tmp_path):
    project = tmp_path / "test.aedt"
    project.write_bytes(b"native project")
    results = tmp_path / "test.aedtresults"
    results.mkdir()
    (results / "CameraArray.asol").write_bytes(b"native convergence")
    (results / "field.bin").write_bytes(b"original field evidence")
    copy = tmp_path / "pre_extract_snapshot"
    record = snapshot_native(project, copy)
    (results / "field.bin").write_bytes(b"mutated on reopening")
    assert record["hashes_verified"]
    assert (copy / "test.aedtresults/field.bin").read_bytes() == b"original field evidence"
    assert len(json.loads((copy / "sha256.json").read_text())) == 3


def test_native_snapshot_precedes_extraction_and_stops_on_failure(monkeypatch, tmp_path):
    import camera_mag.array_maxwell as native
    cfg = tmp_path / "config.json"
    write_json(cfg, config())
    calls = []

    def stage(command, log, deadline, **kwargs):
        phase = command[-1] if "--phase" in command else "solve"
        calls.append(phase)
        if phase == "extract":
            raise RuntimeError("controlled extraction failure")
        return {"returncode": 0}

    def snapshot(*args):
        calls.append("snapshot")
        return {"hashes_verified": True}

    monkeypatch.setattr(native, "run_stage", stage)
    monkeypatch.setattr(native, "snapshot_native", snapshot)
    with pytest.raises(RuntimeError, match="controlled"):
        native.native_iteration(cfg, tmp_path, perf_counter() + 300)
    assert calls == ["build", "solve", "snapshot", "extract"]
    record = json.loads((tmp_path / "native_record.json").read_text())
    assert record["status"] == "stopped" and record["snapshot"]["hashes_verified"]


def test_owned_desktop_startup_exits_once_with_evidence(monkeypatch, tmp_path):
    import camera_mag.array_maxwell as native
    calls = []

    class Exited:
        pid, returncode = 12345, 7

        def poll(self):
            return self.returncode

    def popen(command, **kwargs):
        calls.append(command)
        return Exited()

    monkeypatch.setattr(native.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="code 7"):
        native.launch_owned_desktop(tmp_path)
    assert len(calls) == 1
    record = json.loads((tmp_path / "startup_record.json").read_text())
    assert record["returncode"] == 7 and "failure" in record


@pytest.mark.parametrize("native_scores,converged,expected", [([10., 11.], True, "no_verified_improvement_stop"),
                                                           ([10.], False, "native_unconverged_stop")])
def test_native_feedback_stops_loop_and_keeps_best(monkeypatch, tmp_path, native_scores, converged, expected):
    import camera_mag.array_workflow as flow
    calls = []
    monkeypatch.setattr(flow, "make_solver", lambda *a: object())
    monkeypatch.setattr(flow, "analyze_layout", lambda *a: ({"score": 5.}, {}))
    monkeypatch.setattr(flow, "prepare_candidates", lambda *a: {"candidates": [1]})
    monkeypatch.setattr(flow, "select_next", lambda cfg, *a, **k: moved_config(cfg, {"C2": [11.3, 0, 0]}))

    def native(*args, **kwargs):
        calls.append(args)
        return {"score": native_scores[len(calls) - 1], "native_adaptive_converged": converged}

    monkeypatch.setattr(flow, "native_iteration", native)
    output = tmp_path / "campaign"
    record = flow.run_workflow(ROOT / "examples/camera_array_B.json", output, mode="run", max_iterations=5,
                               estimated_minutes=10, estimate_basis="mock native regression", budget_minutes=15)
    assert record["status"] == expected and len(calls) == len(native_scores)
    if converged:
        best = load_config(output / "best.json")
        assert best["cameras"][1]["position_mm"] == [10.8, 0, 0]
    else:
        assert not (output / "best.json").exists()


@pytest.mark.parametrize("bad", [None, "coordinates", "polarity"])
def test_saved_native_assembly_audit(monkeypatch, tmp_path, camera, bad):
    from types import SimpleNamespace
    import camera_mag.array_maxwell as native
    cfg = config()
    monkeypatch.setattr(native, "load_camera", lambda *args: camera)
    bodies = {}
    for item in cfg["cameras"]:
        for body in camera.metadata["objects"]:
            bounds = np.asarray(body["bounds_mm"]) + item["position_mm"]
            if bad == "coordinates":
                bounds = bounds + .01
            bodies[item["id"] + "_" + body["name"]] = SimpleNamespace(
                bounding_box=bounds.ravel(), volume=body["cad_volume_mm3"])
    setup = SimpleNamespace(props={"SolveFieldOnly": True}, update=lambda: True)
    project = tmp_path / "CameraArray.aedt"

    def save():
        text = "SolveFieldOnly=" + str(setup.props["SolveFieldOnly"]).lower() + "\n"
        for item in cfg["cameras"]:
            cid = item["id"]
            point_out = "true" if bad == "polarity" else "false"
            text += f"$begin '{cid}_AmpereTurns'\nCurrent='10A'\nIsSolid=false\n'Point out of terminal'={point_out}\n$end\n"
            text += f"$begin 'Force_{cid}'\n'Is Virtual'=false\n$end\n"
        project.write_text(text, encoding="utf-8")
        return True

    app = SimpleNamespace(get_setup=lambda name: setup, modeler=bodies, save_project=save,
                          validate_simple=lambda path: 1)
    if bad:
        with pytest.raises(ValueError):
            native.audit_array_build(app, cfg, project, tmp_path)
    else:
        native.audit_array_build(app, cfg, project, tmp_path)
        audit = json.loads((tmp_path / "build_audit.json").read_text())
        assert audit["native_build_verified"] and audit["SolveFieldOnly"] is False
        assert audit["previous_SolveFieldOnly"] is True
        assert len(audit["checked_bodies"]) == 18
