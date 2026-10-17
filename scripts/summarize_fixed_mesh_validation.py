"""Summarize measured one-pass force validation and distance-change diagnostics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--development", default="outputs/mesh_seeding_onepass_2mm")
    parser.add_argument("--holdouts", default="outputs/mesh_seeding_onepass_holdouts")
    args = parser.parse_args()
    development, holdouts = [(ROOT / p).resolve() for p in (args.development, args.holdouts)]
    record = read(holdouts / "holdout_record.json")
    if "completed_UTC" not in record:
        raise ValueError("Held-out reference validation is incomplete")
    for name, digest in record["frozen_physics_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen implementation changed: {name}")
    rows = []
    for folder in [development] + [holdouts / f"gap_{g}mm" for g in record["gaps_mm"]]:
        audit = read(folder / "fixed_mesh_audit.json")
        ref = read(Path(audit["reference_directory"]) / "forces.json")
        row = {key: audit[key] for key in ("distance_mm", "passes", "tetrahedra", "A_Fz_mN", "B_Fz_mN",
               "max_per_coil_Fz_error_percent", "end_to_end_seconds", "energy_error_percent",
               "force_accuracy_accepted", "native_adaptive_converged", "reference_tetrahedra")}
        row.update({f"reference_{m}_Fz_mN": ref[f"Force_{m}.Force_z"]["values"][0] * 1e3 for m in "AB"})
        rows.append(row)
    rows.sort(key=lambda row: row["distance_mm"])
    anchor = next(row for row in rows if row["distance_mm"] == 2)
    diagnostics = []
    for row in rows:
        if row["distance_mm"] == 2:
            continue
        delta = {"distance_mm": row["distance_mm"], "anchor_distance_mm": 2}
        for m in "AB":
            predicted = (row[f"{m}_Fz_mN"] - anchor[f"{m}_Fz_mN"]) * 1e3
            reference = (row[f"reference_{m}_Fz_mN"] - anchor[f"reference_{m}_Fz_mN"]) * 1e3
            delta.update({f"{m}_distance_change_uN": predicted, f"{m}_reference_change_uN": reference,
                          f"{m}_absolute_change_error_uN": abs(predicted - reference)})
        diagnostics.append(delta)
    reference_cost = sum(item["seconds"] for key, item in record.items()
                         if key.startswith(("reference_build_", "reference_solve_", "reference_extract_")))
    result = {"results": rows, "all_total_Fz_accuracy_accepted": all(row["force_accuracy_accepted"] for row in rows),
              "frozen_sources_rechecked": True, "distance_change_diagnostics": diagnostics,
              "held_out_reference_cost_seconds": reference_cost,
              "interaction_force_accuracy_validated": False}
    (holdouts / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (holdouts / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ["# 低保真网格播种：一轮 Maxwell 实测", "",
        "同一 STEP、两个相同相机沿 +X 放置，间隙为 EM 包围盒间隙；0.1 A、100 匝。最终力均由 Maxwell 2022 R2 的 Lorentz 力参数计算。四核串行执行，计时包含建模、低保真预处理、初始网格生成、求解和提取。", "",
        "## 总 Fz 验收", "", "| 间隙 mm | 轮数 | 四面体 | A / B Fz mN | 最大参考误差 | 完整时间 min | 通过 ≤0.2% |",
        "|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in rows:
        report.append(f"| {row['distance_mm']:g} | {row['passes']} | {row['tetrahedra']:,} | {row['A_Fz_mN']:.6f} / {row['B_Fz_mN']:.6f} | {row['max_per_coil_Fz_error_percent']:.5f}% | {row['end_to_end_seconds']/60:.2f} | {'是' if row['force_accuracy_accepted'] else '否'} |")
    report += ["", "2 mm 是本轮参数开发点。1 mm 和 3 mm 使用相同的冻结规则，没有用这次结果回调参数。源码哈希在运行前和汇总时均核查。候选网格不导入任何 Maxwell 已求解网格。参考解从各自同距离的旧网格独立继续细化，要求原生收敛、能量误差 ≤0.25%、连续两次单线圈 Fz 变化 ≤10 µN，并通过材料、电流、采样单位检查。", "",
        "2 mm 的原默认流程为 25.86 min / 22 轮，原低保真流程为 21.83 min / 6 轮；本次一轮流程为 12.11 min，完整时间比为 2.135× / 1.803×。1 / 3 mm 没有采用相同停止设置和完整计时的匹配基线，因此只报告新流程实测时间。", "",
        "## 收敛和适用范围", "",
        "一轮计算通过的是相对更细 Maxwell 参考的总 Fz 精度。所有候选都没有跨轮能量或力变化检查，也未获得原生自适应收敛。新旧流程停止规则不同；上述时间比尚未隔离低保真排序自身的贡献。这三个操作点不证明其他相机、材料、电流、姿态或所有中间间隙的精度。", "",
        f"1 / 3 mm 更细参考解的额外建模、求解及提取合计 {reference_cost/60:.2f} min，单独记为本轮验证成本。候选完整耗时未包含参考成本；原基线同样未包含参考成本。", "",
        "## 距离变化的力：诊断", "",
        "总 Fz 的 0.2% 验收不等于距离差分力的相同相对精度。下表单独列出与 2 mm 相比的变化，单位 µN；仅为相对当前参考网格的诊断，参考解逐轮变化 ≤10 µN 也不是严格误差上界。", "",
        "| 间隙 mm | A 一轮 / 参考 ΔFz | B 一轮 / 参考 ΔFz | 最大差分误差 µN |",
        "|---:|---:|---:|---:|"]
    for d in diagnostics:
        report.append(f"| {d['distance_mm']:g} | {d['A_distance_change_uN']:.3f} / {d['A_reference_change_uN']:.3f} | {d['B_distance_change_uN']:.3f} / {d['B_reference_change_uN']:.3f} | {max(d['A_absolute_change_error_uN'],d['B_absolute_change_error_uN']):.3f} |")
    report += ["", "## 采用的规则和失败试验", "",
        "采用激进面和近场空气规则：实体 0.4 mm，线圈 0.28 mm，低保真评分前 30% 表面积 0.2 mm，累计前 60% 为 0.28 mm；模组邻近真空 0.7 mm，远处空气保持粗网格。单模组自作用矩阵复用，低保真预处理约 2–3 秒。", "",
        "两次局部尖角/气隙盒试验均失败：30.2 万单元、7.07 min、误差 0.579%；48.0 万单元、11.35 min、误差 0.528%。局部方案的面积预算截断使两侧对应面分到不同密度级别，记录见开发目录；这不能单独解释全部力偏差。全部失败结果和源码快照保留。", "",
        "原始工程、原生日志、力历史、参考网格、各阶段命令和时间均保留在每个案例目录。数据汇总另见 summary.json / summary.csv。", ""]
    (holdouts / "report_zh.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
