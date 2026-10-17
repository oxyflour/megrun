"""Native-source material/geometry provenance and constitutive regressions."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from camera_mag.aedt_source import load_source_camera, material_spec, read_source, sha256
from camera_mag.array_config import load_config, normalize_config, write_json
from camera_mag.array_workflow import make_solver
from camera_mag.geometry import EXPECTED
from camera_mag.interaction import MU0
from camera_mag.materials import SoftMagneticMaterial

ROOT = Path(__file__).resolve().parents[1]


def test_native_signed_coercivity_and_actual_permeability():
    spec = material_spec("custom_magnet", {"permeability": "1.17", "magnetic_coercivity": {
        "property_type": "VectorProperty", "Magnitude": "-720000A_per_meter",
        "DirComp1": "0", "DirComp2": "0.6", "DirComp3": "-0.8"}})
    np.testing.assert_allclose(spec["remanence_A_per_m"], [0, 505440, -673920])
    assert spec["Hc_A_per_m"] == -720000
    assert material_spec("native_copper", {"permeability": "0.999991"})["mu_r"] == .999991
    for value in ("1+offset", "pwl($BH,Temp)", "1tesla"):
        with pytest.raises(ValueError, match="Unsupported"):
            material_spec("expression", {"permeability": value})
    with pytest.raises(ValueError, match="Only isotropic"):
        material_spec("anisotropic", {"permeability": {"property_type": "AnisoProperty"}})


def test_native_piecewise_curve_and_local_response():
    h, b = [0, 100, 200, 1000], [0, .3, .8, 1.2]
    mat = SoftMagneticMaterial(h, b, interpolation="linear")
    np.testing.assert_allclose(mat.b_value([50, 150, 600]), [.15, .55, 1.])
    # Constitutive consistency at the actual native nodes and interpolated H.
    fields = np.array([[50., 0, 0], [0, 150., 0], [0, 0, 1500.]])
    m = mat.response(fields)
    np.testing.assert_allclose(mat.local_response(fields + m / 3), m, rtol=2e-6, atol=.02)


@pytest.fixture
def source_bundle(tmp_path):
    # Portable fixture uses a real CAD shape; material values deliberately
    # differ from benchmark assumptions to detect accidental fallback.
    folder = tmp_path / "source"
    folder.mkdir()
    shutil.copy2(ROOT / "models/B0500_scale_AF_VCM_EM_only.step", folder / "template.step")
    mat = material_spec("native_test", {"permeability": "1.17", "magnetic_coercivity": {
        "property_type": "VectorProperty", "Magnitude": "-720000A_per_meter",
        "DirComp1": "0", "DirComp2": "0", "DirComp3": "1"}})
    copper = material_spec("native_copper", {"permeability": "0.999991"})
    data = {"schema_version": 1, "kind": "aedt_camera_template", "provenance": {"project_sha256": "fixture"},
            "materials": {"native_test": mat, "native_copper": copper},
            "objects": [{"name": n, "material_name": "native_copper" if n == "AF_COIL_COPPER" else "native_test",
                         "kind": "copper" if n == "AF_COIL_COPPER" else "magnet"} for n in sorted(EXPECTED)],
            "files_sha256": {"template.step": sha256(folder / "template.step")}}
    path = folder / "source.json"
    write_json(path, data)
    return path


def test_real_geometry_uses_native_assignments_without_benchmark(monkeypatch, source_bundle, tmp_path):
    import camera_mag.array_workflow as workflow
    cfg = load_config(ROOT / "examples/camera_array_B.json")
    cfg.pop("step")
    cfg["aedt_source"] = str(source_bundle)
    cfg = normalize_config(cfg)
    monkeypatch.setattr(workflow, "benchmark_materials", lambda: pytest.fail("Native source must not call benchmark materials"))
    solver = make_solver(cfg, tmp_path / "cache")
    camera = solver.camera
    assert set(camera.material_ids) == {"native_test", "native_copper"}
    magnetic = camera.material_ids == "native_test"
    np.testing.assert_allclose(camera.remanence[magnetic], np.tile([0, 0, 842400], (sum(magnetic), 1)))
    assert solver.materials["native_test"].mu_r == 1.17
    assert camera.metadata["br_t"] is None
    report, result = solver.evaluate(cfg)
    assert result["converged"] and report["physics_source"]["bundle_sha256"] == sha256(source_bundle)
    # Actual copper permeability contributes its (small) induced field.
    assert any(r["object"] == "AF_COIL_COPPER" for r in report["sources"])
    previous_key = camera.metadata["sha256_key"]
    source = read_source(source_bundle)
    changed = deepcopy(source["materials"]["native_test"]["raw_properties"])
    changed["magnetic_coercivity"]["Magnitude"] = "-360000A_per_meter"
    source["materials"]["native_test"] = material_spec("native_test", changed)
    write_json(source_bundle, source)
    with pytest.raises(ValueError, match="Frozen AEDT source changed"):
        normalize_config(cfg)
    newer, _ = load_source_camera(source_bundle, cache_dir=tmp_path / "cache")
    assert newer.metadata["sha256_key"] != previous_key
    np.testing.assert_allclose(newer.remanence[magnetic], camera.remanence[magnetic] / 2)


def test_source_has_no_implicit_step_and_rejects_geometry_tampering(source_bundle):
    cfg = load_config(ROOT / "examples/camera_array_B.json")
    del cfg["step"]
    with pytest.raises(ValueError, match="exactly one"):
        normalize_config(cfg)
    cfg["aedt_source"] = str(source_bundle)
    frozen = normalize_config(cfg)
    assert "step" not in frozen
    cfg["step"] = str(ROOT / "models/B0500_scale_AF_VCM_EM_only.step")
    with pytest.raises(ValueError, match="exactly one"):
        normalize_config(cfg)
    with (source_bundle.parent / "template.step").open("ab") as stream:
        stream.write(b"\nmodified\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        read_source(source_bundle)


def test_normalized_material_cannot_disagree_with_native_properties(source_bundle):
    data = read_source(source_bundle)
    data["materials"]["native_test"]["mu_r"] = 1.05
    write_json(source_bundle, data)
    with pytest.raises(ValueError, match="differs from native properties"):
        read_source(source_bundle)
