"""Prevent max-pass/aborted references from being reported as convergence."""
import json
from pathlib import Path

import pytest
from scripts.refine_pair_difference_reference import audit, native_convergence_rows


def reference(tmp_path, profile="Adaptive Passes converged\nStatus: Normal Completion",
              values=(.033, .033001, .0330015), passes=(1,2,3), mesh=1000):
    folder = tmp_path / "PrecisionReference"
    folder.mkdir()
    project = tmp_path / "PrecisionReference.aedt"
    results = project.with_suffix(".aedtresults")
    results.mkdir()
    manifest = dict(project=str(project), design="PrecisionReference", force_delta_n=2e-6, percent_error=.12)
    history = dict(passes=list(passes), forces={f"Force_{label}.Force_z": list(values) for label in "AB"})
    forces = {f"Force_{label}.Force_z": dict(values=[values[-1]], unit="newton") for label in "AB"}
    for name, data in (("manifest.json", manifest), ("force_adaptive_history.json",history), ("forces.json",forces)):
        (folder/name).write_text(json.dumps(data))
    (folder/"convergence.txt").write_text("Completed : 3")
    (folder/"mesh_stats.txt").write_text(f"Total number of mesh elements: {mesh}")
    (folder/"profile.txt").write_text(profile)
    (results/"PrecisionReference.asol").write_text("LP(VID=22, Ver=7, P=3)\n" + "\n".join([
        "c(de_ee='(0.1, N/A)', ee=0.1, p=1, te=0.01, tet=1000)",
        "c(de_ee='(0.1, 0.01)', tet=1000, p=2, de=0.01, te=0.01, ee=0.1)",
        "c(de_ee='(0.1, 0.01)', ee=0.1, de=0.01, p=3, te=0.01, tet=1000)"]))
    return folder


def test_normal_completion_without_native_convergence(tmp_path):
    result = audit(reference(tmp_path, profile="Maximum number of passes reached\nStatus: Normal Completion"))
    assert result["native_status"] == "Normal Completion"
    assert not result["native_adaptive_converged"]


def test_aborted_retained_pass_never_certifies_convergence(tmp_path):
    result = audit(reference(tmp_path, profile="Adaptive Passes converged\nStatus: Aborted"))
    assert result["native_status"] == "Aborted"
    assert not result["native_adaptive_converged"]


def test_native_message_alone_does_not_certify_force_stability(tmp_path):
    result = audit(reference(tmp_path, values=(.033, .033010, .033020)))
    assert not result["configured_stopping_conditions_met"]
    assert not result["native_adaptive_converged"]


def test_consistent_native_convergence_is_a_diagnostic(tmp_path):
    result = audit(reference(tmp_path))
    assert result["native_adaptive_converged"]
    assert not result["stability_diagnostics_are_error_bounds"]


def test_incomplete_force_history_rejected(tmp_path):
    with pytest.raises(ValueError, match="mismatch"):
        audit(reference(tmp_path, passes=(1,3), values=(.033,.033001)))


def test_wrong_mesh_rejected(tmp_path):
    with pytest.raises(ValueError, match="mismatch"):
        audit(reference(tmp_path, mesh=999))


@pytest.mark.parametrize("relative", [
    "outputs/mesh_seeding_onepass_holdouts/gap_1mm/TightReference.aedtresults/TightReference.asol",
    "outputs/mesh_seeding_2mm/TightReference_Default.aedtresults/TightReference.asol",
])
def test_existing_native_metadata_order(relative):
    path = Path(__file__).resolve().parents[1]/relative
    if not path.exists():
        pytest.skip("Native experiment unavailable")
    rows = native_convergence_rows(path.read_text())
    expected = 907564 if "gap_1mm" in relative else 818966
    assert rows[max(rows)]["tet"] == expected
    assert "de" in rows[max(rows)]
