"""Validate a single Maxwell field solve against a tighter independent solution."""
import argparse
import json
from pathlib import Path
import re

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def force_vector(folder):
    data = read_json(folder / "forces.json")
    values = []
    for label in "AB":
        item = data[f"Force_{label}.Force_z"]
        if item["unit"] not in ("N", "newton") or len(item["values"]) != 1:
            raise ValueError("Invalid force units or values")
        values.append(item["values"][0])
    return np.asarray(values)


def energy(folder):
    text = (folder / "convergence.txt").read_text()
    match = re.search(r"Current\s*:\s*\(([^,]+),\s*([^)]+)\)", text)
    return [float(v) if v.strip() != "N/A" else None for v in match.groups()]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", default="Local")
    parser.add_argument("--reference", default="outputs/mesh_seeding_2mm/TightReference")
    parser.add_argument("--baseline", default="outputs/mesh_seeding_2mm")
    args = parser.parse_args()
    root, reference = Path(args.output).resolve(), Path(args.reference).resolve()
    folder = root / args.case
    record = read_json(root / "run_record.json")
    timing = record["cases"][args.case]
    if timing.get("extract", {}).get("returncode", 1):
        raise ValueError("Candidate extraction incomplete")
    provenance, rp = [read_json(f / "provenance.json") for f in (folder, reference)]
    manifest, rm = [read_json(f / "manifest.json") for f in (folder, reference)]
    for key in ("distance_mm", "current_a", "turns", "air_mm", "br_T", "mu_r_recoil", "force_type"):
        if manifest[key] != rm[key]:
            raise ValueError(f"Reference differs in {key}")
    if provenance["step_sha256"] != rp["step_sha256"]:
        raise ValueError("Different STEP geometry")
    if (folder / "maxwell_bh.csv").read_bytes() != (reference / "maxwell_bh.csv").read_bytes():
        raise ValueError("Different BH curves")
    for source in (provenance, rp):
        for key in ("SI_positions_checked", "current_polarity_and_ampere_turns_checked", "BH_field_median_error_below_5_percent"):
            if not source.get(key):
                raise ValueError(f"Failed physics guard {key}")
    profile, ref_profile = [(f / "profile.txt").read_text() for f in (folder, reference)]
    if "Status: Normal Completion" not in profile:
        raise ValueError("Single-pass field solve did not complete normally")
    if "Status: Normal Completion" not in ref_profile or "Adaptive Passes converged" not in ref_profile:
        raise ValueError("Reference did not converge")
    if rp["setup"]["PercentError"] > .25 or max(energy(reference)) > rp["setup"]["PercentError"]:
        raise ValueError("Reference energy criterion is insufficient")
    rh = read_json(reference / "force_adaptive_history.json")
    rf = np.array([rh["forces"][f"Force_{label}.Force_z"] for label in "AB"])
    if rf.shape[1] < 3 or np.max(np.abs(np.diff(rf, axis=1)[:, -2:])) > 1.000001e-5:
        raise ValueError("Reference force deltas exceed 10 uN")
    setup = provenance["setup"]
    for key, target in {"MaximumPasses": 1, "MinimumPasses": 1, "MinimumConvergedPasses": 1,
                        "NonLinearResidual": 1e-4, "PercentError": .5}.items():
        if setup[key] != target:
            raise ValueError(f"Unexpected candidate {key}")
    if setup["MeshLink"]["ImportMesh"]:
        raise ValueError("Candidate imported a solved Maxwell mesh")
    history = read_json(folder / "force_adaptive_history.json")
    if len(history["passes"]) != 1:
        raise ValueError("Not a one-pass field solve")
    plan = read_json(folder / "mesh_plan.json")
    local = plan.get("local_regions")
    if local:
        if not (local["physical_objects_unchanged"] and local["physical_volumes_unchanged"]):
            raise ValueError("Physical geometry changed")
        box_names = {region["object"] for region in local["regions"]}
        if not box_names <= set(provenance["non_model_objects"]) or box_names & set(provenance["physical_model_objects"]):
            raise ValueError("Refinement boxes became physical material")
        required_operations = ("LF3_CoilGapRegions", "LF3_CornerRegions")
        selection_names = box_names
    else:
        air = plan["air_partition"]
        if not air["STEP_body_volumes_preserved"] or not np.isclose(
                air["outer_domain_volume_mm3"], air["partition_volume_mm3"], rtol=1e-8, atol=1e-6):
            raise ValueError("Vacuum partition changed physical geometry")
        selection_names = {region["object"] for region in air["regions"]}
        if not selection_names <= set(provenance["physical_model_objects"]):
            raise ValueError("Near vacuum regions missing")
        if any(provenance["object_materials"][name] != "vacuum" for name in selection_names):
            raise ValueError("Near regions are not vacuum")
        box_names = set()
        required_operations = ("LF2_NearAir_07mm",)
    object_ids = provenance.get("object_ids_by_name")
    if object_ids is None:
        # AEDT 2022 reads assignments back as IDs, and older extraction records
        # did not include the name mapping. Read it from the saved native CAD.
        native = Path(provenance["project"]).read_text(encoding="utf-8", errors="replace")
        object_ids = {}
        for part in re.findall(r"\$begin 'GeometryPart'(.*?)\$end 'GeometryPart'", native, re.S):
            name = re.search(r"Name='([^']+)'", part)
            body = re.search(r"ParentPartID=(\d+)", part)
            if name and body:
                object_ids[name[1]] = int(body[1])
    selected_ids = {object_ids[name] for name in selection_names}
    for name in required_operations:
        operation = provenance["mesh_operations"][name]
        inside = operation.get("RefineInside", operation.get("Region") == "Inside Selection")
        assigned = set(operation["Objects"])
        if not operation["Enabled"] or not inside or not assigned or not (assigned <= selection_names or assigned <= selected_ids):
            raise ValueError("Local mesh operation absent or invalid")
    actual, target = force_vector(folder), force_vector(reference)
    if not np.allclose(rf[:, -1], target, atol=1e-12, rtol=1e-9):
        raise ValueError("Reference history and final forces disagree")
    errors = np.abs(actual / target - 1) * 100
    meshes = [(folder / name).read_text() for name in ("initial_mesh_stats.txt", "mesh_stats.txt")]
    counts = [int(re.search(r"Total number of mesh elements:\s*(\d+)", text)[1]) for text in meshes]
    if counts[0] != counts[1]:
        raise ValueError("Single-pass solve changed the initial mesh")
    if any(name in meshes[1] for name in box_names):
        raise ValueError("Non-model box appears as physical meshed region")
    accepted = bool(errors.max() <= .2)
    result = {"case": args.case, "distance_mm": manifest["distance_mm"], "passes": 1,
              "tetrahedra": counts[0], "max_per_coil_Fz_error_percent": float(errors.max()),
              "A_Fz_mN": float(actual[0] * 1e3), "B_Fz_mN": float(actual[1] * 1e3),
              "per_coil_errors_percent": errors.tolist(), "energy_error_percent": energy(folder)[0],
              "force_accuracy_accepted": accepted, "native_adaptive_converged": "Adaptive Passes converged" in profile,
              "end_to_end_seconds": timing["end_to_end_seconds"],
              "preprocessing_seconds": plan["preprocessing_seconds"],
              "reference_directory": str(reference), "reference_tetrahedra": int(re.search(r"Total number of mesh elements:\s*(\d+)", (reference / "mesh_stats.txt").read_text())[1]),
              "acceptance": "Max per-coil total Fz error <= 0.2% vs tighter Maxwell, normal field completion and physics guards; not native adaptive convergence",
              "known_development_gap_mm": 2, "held_out_distance": manifest["distance_mm"] != 2,
              "interaction_force_accuracy_validated": False}
    baseline = Path(args.baseline)
    if (baseline / "run_record.json").exists():
        base = read_json(baseline / "run_record.json")
        if base["distance_mm"] == manifest["distance_mm"] and base["cores"] == record["cores"]:
            result["speedup_vs_previous_cases"] = {name: item["end_to_end_seconds"] / timing["end_to_end_seconds"] for name, item in base["cases"].items()}
    (root / "fixed_mesh_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = ["# 低保真加密，一轮 Maxwell 力精度试验", "",
              f"间隙 {manifest['distance_mm']:g} mm；一轮力精度验收：{'通过' if accepted else '未通过'}。", "",
              f"四面体 {counts[0]:,}；最大单线圈总 Fz 参考误差 {errors.max():.5f}%；端到端耗时 {timing['end_to_end_seconds']/60:.2f} min。", "",
              f"线圈 A / B 的 Fz：{actual[0]*1e3:.6f} / {actual[1]*1e3:.6f} mN。参考网格 {result['reference_tetrahedra']:,} 单元，能量误差目标 ≤0.25%，连续两次力变化 ≤10 µN。", "",
              "实际核查了采样单位、电流方向与安匝数、B-H 场和原生网格操作。实体与空气外域体积保持不变；局部盒保持非模型标志，或新增分区确认为真空。未导入已求解 Maxwell 网格。时间包括初始网格生成。", "",
              f"原生能量误差 {energy(folder)[0]:.5g}%；此流程仅作一次固定网格场解，未获得跨轮收敛证明。其验收依据是相对更细 Maxwell 参考的总 Fz 误差 ≤0.2%。不能据此声称原生自适应收敛，或距离变化产生的小干扰力也有同样相对精度。", ""]
    if "speedup_vs_previous_cases" in result:
        report += ["与原相同间隙、四核全流程的时间比：" + "，".join(f"{name} {value:.3f}×" for name, value in result["speedup_vs_previous_cases"].items()) + "。这是完整流程之比，未隔离低保真排序本身的贡献。", ""]
    if result["held_out_distance"]:
        report += ["本距离使用 2 mm 开发后固定的规则，没有根据此距离的 Maxwell 力输出调参。", ""]
    (root / "report_zh.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
