"""Report a two-pass attempt, including failure; never equate a pass cap with convergence."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from compare_mesh_seeding import read_case


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pass_start_seconds(log, number):
    match = re.search(rf"\[PROFILE\] Pass {number} \((\d+):(\d+):(\d+) (上午|下午|AM|PM)", log)
    if not match:
        return None
    hour, minute, second = map(int, match.groups()[:3])
    hour %= 12
    if match[4] in ("下午", "PM"):
        hour += 12
    return hour * 3600 + minute * 60 + second


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline", default="outputs/mesh_seeding_2mm")
    args = parser.parse_args()
    root, baseline = Path(args.output).resolve(), Path(args.baseline).resolve()
    folder, reference = root / "Aggressive", baseline / "TightReference"
    record = read_json(root / "run_record.json")
    timing = record["cases"]["Aggressive"]
    if timing.get("extract", {}).get("returncode", 1):
        raise ValueError("No successfully extracted candidate")
    manifest = read_json(folder / "manifest.json")
    ref_manifest = read_json(reference / "manifest.json")
    provenance = read_json(folder / "provenance.json")
    ref_provenance = read_json(reference / "provenance.json")
    for key in ("distance_mm", "current_a", "turns", "air_mm", "br_T", "mu_r_recoil", "force_type"):
        if manifest[key] != ref_manifest[key]:
            raise ValueError(f"Reference differs in {key}")
    if provenance["step_sha256"] != ref_provenance["step_sha256"]:
        raise ValueError("Reference STEP differs")
    for source in (provenance, ref_provenance):
        for key in ("SI_positions_checked", "current_polarity_and_ampere_turns_checked",
                    "BH_field_median_error_below_5_percent"):
            if not source.get(key):
                raise ValueError(f"Failed physics guard {key}")
    if (folder / "maxwell_bh.csv").read_bytes() != (reference / "maxwell_bh.csv").read_bytes():
        raise ValueError("Reference BH curve differs")
    if provenance["setup"]["MeshLink"]["ImportMesh"]:
        raise ValueError("This experiment must not import a solved Maxwell mesh")
    setup = provenance["setup"]
    for key, value in {"MaximumPasses": 2, "MinimumPasses": 2, "MinimumConvergedPasses": 1,
                       "PercentError": .5, "PercentRefinement": 30, "NonLinearResidual": 1e-4}.items():
        if setup[key] != value:
            raise ValueError(f"Unexpected native setting {key}")
    cache = setup["ExpressionCache"]["CacheItem"]
    if isinstance(cache, dict):
        cache = [cache]
    for label in "AB":
        items = [item for item in cache if item["Expression"] == f"Force_{label}.Force_z"]
        if (len(items) != 1 or not items[0]["IsConvergence"] or items[0]["UseRelativeConvergence"] != 0
                or not np.isclose(items[0]["MaxConvergenceDelta"], 2e-5, rtol=1e-9, atol=0)):
            raise ValueError("Native force convergence criterion differs")
    plan = read_json(folder / "mesh_plan.json")
    if not plan["air_partition"]["STEP_body_volumes_preserved"]:
        raise ValueError("STEP body geometry changed")
    ref_profile = (reference / "profile.txt").read_text()
    ref_history = read_json(reference / "force_adaptive_history.json")
    if "Adaptive Passes converged" not in ref_profile or "Status: Normal Completion" not in ref_profile:
        raise ValueError("Reference did not converge")
    ref_values = np.array([ref_history["forces"][f"Force_{label}.Force_z"] for label in "AB"])
    if ref_values.shape[1] < 3 or np.max(np.abs(np.diff(ref_values, axis=1)[:, -2:])) > 1.000001e-5:
        raise ValueError("Reference last two force deltas exceed 10 uN")
    history = read_json(folder / "force_adaptive_history.json")
    forces = np.array([history["forces"][f"Force_{label}.Force_z"] for label in "AB"])
    final_data = read_json(folder / "forces.json")
    for i, label in enumerate("AB"):
        quantity = final_data[f"Force_{label}.Force_z"]
        if quantity["unit"] not in ("N", "newton") or not np.isclose(quantity["values"][0], forces[i, -1], atol=1e-12, rtol=1e-9):
            raise ValueError("Force history and final force disagree")
    deviations = np.max(np.abs(forces / ref_values[:, -1, None] - 1), axis=0) * 100
    delta = float(np.max(np.abs(np.diff(forces, axis=1))) * 1e6)
    convergence = (folder / "convergence.txt").read_text()
    energy = re.search(r"Current\s*:\s*\(([^,]+),\s*([^)]+)\)", convergence)
    initial = int(re.search(r"Total number of mesh elements:\s*(\d+)",
                            (folder / "initial_mesh_stats.txt").read_text())[1])
    final = int(re.search(r"Total number of mesh elements:\s*(\d+)",
                          (folder / "mesh_stats.txt").read_text())[1])
    try:
        row, _, _, _ = read_case(root, "Aggressive", timing)
        native_accepted, native_failure = True, None
    except ValueError as error:
        row = None
        native_accepted, native_failure = False, str(error)
    force_accepted = bool(deviations[-1] <= .2)
    accepted = native_accepted and force_accepted and len(history["passes"]) <= 2
    baseline_record = read_json(baseline / "run_record.json")
    speeds = {name: value["end_to_end_seconds"] / timing["end_to_end_seconds"]
              for name, value in baseline_record["cases"].items()}
    log = next((root / "Aggressive.aedt.batchinfo").glob("*.log")).read_text(encoding="utf-8")
    pass_starts = [pass_start_seconds(log, i) for i in (1, 2)]
    first_seconds = (pass_starts[1] - pass_starts[0]) % 86400 if None not in pass_starts else None
    pass_energy = []
    for line in convergence.splitlines():
        parts = [part.strip() for part in line.strip().rstrip("|").split("|")]
        if len(parts) == 5 and parts[0].isdigit():
            pass_energy.append(float(parts[3]))
    if len(pass_energy) != len(history["passes"]):
        raise ValueError("Energy and force histories have different pass counts")
    force_delta_accepted = delta <= 20 * (1 + 1e-6)
    result = {"native_accepted": native_accepted, "native_failure": native_failure,
              "within_02_percent_reference": force_accepted, "two_pass_target_met": bool(accepted),
              "passes": len(history["passes"]), "initial_tetrahedra": initial, "final_tetrahedra": final,
              "energy_error_percent": float(energy[1]), "energy_delta_percent": float(energy[2]),
              "max_pass_Fz_change_uN": delta, "per_pass_max_Fz_error_percent": deviations.tolist(),
              "force_delta_criterion_met": bool(force_delta_accepted),
              "per_pass_energy_error_percent": pass_energy,
              "first_pass_stage_seconds_from_native_log": first_seconds,
              "first_pass_meets_02_percent_Fz_reference": bool(deviations[0] <= .2),
              "one_pass_pipeline_separately_measured": False,
              "A_Fz_mN": float(forces[0, -1] * 1e3), "B_Fz_mN": float(forces[1, -1] * 1e3),
              "end_to_end_seconds": timing["end_to_end_seconds"], "speedup_vs_previous_cases": speeds,
              "speedup_15x_target_met": bool(accepted and speeds['Adaptive'] >= 1.5 and speeds['Geometry'] >= 1.5),
              "policy_preprocessing_seconds": plan["preprocessing_seconds"],
              "imported_Maxwell_mesh": False, "native_audit": row,
              "stopping_rule_change": "MinimumConvergedPasses 2 -> 1; same energy/delta and external Fz thresholds",
              "scope": "Development on known 2 mm gap. No held-out generalization or lateral/interaction-force accuracy claim."}
    (root / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    report = ["# 激进初始网格：两轮试验", "",
              f"结论：{'达到两轮验收目标' if accepted else '未达到两轮验收目标'}。", "",
              "相同 STEP、2 mm 间隙、0.1 A、100 匝、材料和空气外边界。四核串行求解。未导入 Maxwell 已求解网格。", "",
              "| 指标 | 实测 |", "|---|---:|",
              f"| 自适应轮数 | {len(history['passes'])} |",
              f"| 初始 / 最终四面体 | {initial:,} / {final:,} |",
              f"| 能量误差 / 能量变化 | {float(energy[1]):.5g}% / {float(energy[2]):.5g}% |",
              f"| 两轮最大单线圈 Fz 变化 | {delta:.3f} µN |",
              f"| 首轮最大单线圈 Fz 参考误差 | {deviations[0]:.5f}% |",
              f"| 最终最大单线圈 Fz 参考误差 | {deviations[-1]:.5f}% |",
              f"| 建模、初始网格生成、求解、提取总时间 | {timing['end_to_end_seconds']/60:.2f} min |",
              f"| 相对旧 Adaptive / LowFidelity 时间比 | {speeds['Adaptive']:.3f}× / {speeds['LowFidelity']:.3f}× |", "",
              "验收要求：Maxwell 正常且自适应收敛；能量误差及变化均 ≤0.5%；单线圈 Fz 变化 ≤20 µN；两线圈各自相对原 81.9 万单元参考误差 ≤0.2%。", "",
              "MinimumConvergedPasses 从 2 改为 1，所以只有一次跨轮力差检查。旧基线要求两次连续力差检查。时间比是整个新流程与旧流程之比，不能全部归因为加密策略。", "",
              "规则：实体内部 0.4 mm、线圈 0.28 mm；低保真排序前 30% 表面积约 0.2 mm，累计前 60% 表面积约 0.28 mm；邻近空气 0.7 mm，远处空气继续粗网格。局部面操作有增长预算上限。", "",
              "这次属于在已知案例上的开发试验。通过后仍需用未调参的距离/姿态验收。首轮力误差达标仅证明该案例固定网格的力精度，不能替代跨轮收敛检查，也不证明微小横向力或模组间干扰力的相对精度。", ""]
    if first_seconds is not None:
        report += [f"首轮阶段按原生日志中 Pass 1 到 Pass 2 的时间戳计为 {first_seconds/60:.2f} min；建模及初始网格生成另计 {timing['build']['seconds']/60:.2f} min。没有单独跑 MaximumPasses=1 的端到端流程，因此不把这些阶段相加当作已实测的一轮流程加速比。", ""]
    if native_failure:
        report += [f"未通过的原生检查：{native_failure}。", ""]
    if not force_delta_accepted:
        report += [f"明确失败项：跨轮 Fz 变化 {delta:.3f} µN 超过 20 µN，不能把两轮达到上限称为已收敛。", ""]
    (root / "report_zh.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
