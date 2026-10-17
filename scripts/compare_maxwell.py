"""Compare independently solved Maxwell forces with the uncalibrated MVP."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "outputs/maxwell_verified"
CASES = [(1., "Gap_1mm_CCW"), (1.5, "Gap_15mm_CCW"),
         (2., "Gap_2mm_CCW"), (2.5, "Gap_25mm_CCW"), (3., "Gap_3mm_CCW")]


def force(folder, label):
    provenance = json.loads((folder / "provenance.json").read_text())
    if "Status: Normal Completion" not in (folder / "profile.txt").read_text():
        raise ValueError(f"Maxwell did not finish normally: {folder}")
    setup = provenance["setup"]
    if setup["MaximumPasses"] > 1:
        match = re.search(r"Current\s*:\s*\(([^,]+),\s*([^)]+)\)",
                          (folder / "convergence.txt").read_text())
        if not match or max(map(float, match.groups())) > setup["PercentError"]:
            raise ValueError(f"Unconverged Maxwell reference: {folder}")
    elif not setup["MeshLink"].get("ImportMesh"):
        raise ValueError(f"Single-pass reference has no source mesh: {folder}")
    for key in ["SI_positions_checked", "current_polarity_and_ampere_turns_checked",
                "BH_field_median_error_below_5_percent"]:
        if provenance.get(key) is not True:
            raise ValueError(f"Unverified Maxwell reference: {folder}")
    values = json.loads((folder / "forces.json").read_text())
    vector = []
    for axis in "xyz":
        found = [v for k, v in values.items()
                 if k.startswith(f"Force_{label}.") and k.lower().endswith("_" + axis)]
        if len(found) != 1 or found[0]["unit"] not in ("N", "newton") or len(found[0]["values"]) != 1:
            raise ValueError(f"Unexpected {label}/{axis} report or force unit in {folder}: {values}")
        vector.append(found[0]["values"][0])
    result = np.array(vector, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite Maxwell force in {folder}")
    return result


def field(path):
    # Maxwell FLD exports begin with two header lines, followed by SI positions/vectors.
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = [float(v) for v in line.split()]
        except ValueError:
            continue
        if len(row) == 6:
            rows.append(row)
    result = np.asarray(rows)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"Invalid Maxwell field export: {path}")
    return result


def history_change(folder):
    hist = json.loads((folder / "force_adaptive_history.json").read_text())
    axial = np.array([v for k, v in hist["forces"].items() if k.lower().endswith("_z")])
    if axial.shape[1] < 2:
        raise ValueError(f"No adaptive force history in {folder}")
    return float(np.max(np.abs(axial[:, -1] - axial[:, -2])) * 1e3)


def main():
    low = pd.read_csv(ROOT / "outputs/sweep/coil_force_sweep.csv").set_index("distance_mm")
    isolated = force(REF / "Isolated_CCW", "A")
    rows = []
    for distance, design in CASES:
        folder = REF / design
        a, b = force(folder, "A"), force(folder, "B")
        mean = (a[2] + b[2]) / 2
        coarse = low.loc[distance, "coil_A_total_z_N"]
        delta = mean - isolated[2]
        coarse_delta = low.loc[distance, "coil_A_interaction_delta_z_N"]
        rows.append({"gap_mm": distance, "center_distance_mm": distance + 8.8,
                     "maxwell_A_z_mN": a[2] * 1e3, "maxwell_B_z_mN": b[2] * 1e3,
                     "maxwell_mean_z_mN": mean * 1e3, "coarse_z_mN": coarse * 1e3,
                     "signed_total_error_percent": (coarse / mean - 1) * 100,
                     "maxwell_delta_z_mN": delta * 1e3, "coarse_delta_z_mN": coarse_delta * 1e3,
                     "absolute_delta_error_mN": abs(coarse_delta - delta) * 1e3,
                     "relative_delta_error_percent": abs((coarse_delta - delta) / delta) * 100,
                     "maxwell_AB_z_spread_mN": abs(a[2] - b[2]) * 1e3,
                     "maxwell_last_pass_z_change_mN": history_change(folder),
                     **{f"maxwell_{label}_{axis}_mN": vec[i] * 1e3
                        for label, vec in [("A", a), ("B", b)] for i, axis in enumerate("xy")}})
    table = pd.DataFrame(rows)
    table.to_csv(REF / "comparison.csv", index=False)
    fine = json.loads((ROOT / "outputs/validation_low_035/summary.json").read_text())
    fine_z = fine["coil_force"]["A"]["total_N"][2]
    main_z = float(table.loc[table.gap_mm == 2, "maxwell_mean_z_mN"].iloc[0]) / 1e3
    air40_z = np.mean([force(REF / "Gap_2mm_Air40", label)[2] for label in "AB"])
    current = field(REF / "Gap_2mm_CCW/current_samples.fld")
    expected_positions = np.array([[0, .002975, .00285], [.002975, 0, .00285]])
    if not np.allclose(current[:, :3], expected_positions, atol=1e-10):
        raise ValueError("Current sampling points are not in expected SI coordinates")
    if current[0, 3] >= 0 or current[1, 4] <= 0:
        raise ValueError("Maxwell current is not counterclockwise from +Z")
    paired_baseline = force(REF / "Gap_2mm_OnlyA", "A")[2]
    paired_delta = force(REF / "Gap_2mm_CCW", "A")[2] - paired_baseline
    coarse_delta = low.loc[2., "coil_A_interaction_delta_z_N"]
    paired_current = field(REF / "Gap_2mm_OnlyA/current_samples.fld")
    current_difference = np.linalg.norm(current[:, 3:] - paired_current[:, 3:]) / np.linalg.norm(current[:, 3:])
    def tet_count(folder):
        match = re.search(r"Total number of mesh elements:\s*(\d+)", (folder / "mesh_stats.txt").read_text())
        if not match:
            raise ValueError(f"No tetrahedron count in {folder}")
        return int(match.group(1))
    source_tets = tet_count(REF / "Gap_2mm_CCW")
    baseline_tets = tet_count(REF / "Gap_2mm_OnlyA")
    if source_tets != baseline_tets:
        raise ValueError("Matched baseline did not preserve the reference mesh element count")
    diagnostics = {"maxwell_version": "2022 R2", "current_A": .1, "turns": 100,
                   "maxwell_isolated_z_mN": isolated[2] * 1e3,
                   "coarse_isolated_z_mN": low.iloc[0].coil_A_isolated_z_N * 1e3,
                   "max_total_error_percent": float(table.signed_total_error_percent.abs().max()),
                   "max_delta_error_mN": float(table.absolute_delta_error_mN.max()),
                   "max_delta_relative_error_percent": float(table.relative_delta_error_percent.max()),
                   "air40_mean_z_mN": air40_z * 1e3,
                   "air20_to_air40_z_change_mN": (air40_z - main_z) * 1e3,
                   "air20_to_air40_z_change_percent": (air40_z / main_z - 1) * 100,
                   "fine_035mm_z_mN": fine_z * 1e3,
                   "fine_035mm_error_percent": (fine_z / main_z - 1) * 100,
                   "matched_baseline_A_z_mN": paired_baseline * 1e3,
                   "matched_delta_A_z_mN": paired_delta * 1e3,
                   "matched_delta_absolute_error_mN": abs(coarse_delta - paired_delta) * 1e3,
                   "matched_delta_relative_error_percent": abs((coarse_delta - paired_delta) / paired_delta) * 100,
                   "matched_source_and_baseline_tetrahedra": source_tets,
                   "matched_current_sample_relative_difference": float(current_difference),
                   "fine_035mm_delta_relative_error_percent": abs((fine["coil_force"]["A"]["interaction_delta_N"][2]
                                                               - paired_delta) / paired_delta) * 100,
                   "current_samples_SI": current.tolist(),
                   "force_fit_or_scale_applied": False}
    (REF / "comparison_summary.json").write_text(json.dumps(diagnostics, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for ax, max_name, coarse_name, ylabel in [
            (axes[0], "maxwell_mean_z_mN", "coarse_z_mN", "Axial coil force Fz (mN)"),
            (axes[1], "maxwell_delta_z_mN", "coarse_delta_z_mN", "Fz - isolated Fz (mN)")]:
        ax.plot(table.gap_mm, table[max_name], "o-", label="Maxwell 2022 R2")
        ax.plot(table.gap_mm, table[coarse_name], "s--", label="Uncalibrated coarse model")
        ax.set_xlabel("EM envelope gap along X (mm)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(REF / "comparison.png", dpi=180)
    plt.close(fig)
    lines = ["# Maxwell 实算对比：当前 MVP 的距离增量尚不一致", "",
             "已实际使用 Ansys Maxwell 3D 2022 R2 对相同 STEP 做磁静态有限元求解。"
             "五组双摄、独立单摄和 40 mm 空气域参考正常完成并满足能量收敛条件。"
             "所有接受数据通过 SI 采样坐标、电流方向/安匝和材料场检查；固定网格对照另行记载。"
             "原粗算法结果保留，没有用 Maxwell 力做拟合、缩放或替换。", "",
             f"五个间隙的轴向总力相对误差最大为 {diagnostics['max_total_error_percent']:.2f}%。"
             f"2 mm 同网格对照的轴向力增量误差为 {diagnostics['matched_delta_relative_error_percent']:.2f}%，"
             "所以当前结果不能证明距离变化时的受力与 Maxwell 一致。", "",
             "## 五点扫距", "",
             "两个相同模组同向沿 +X 放置。电流 0.1 A、100 匝，从 +Z 看逆时针；"
             "间隙按 EM 包围盒定义，中心距为间隙 + 8.8 mm。Fz 为 A/B 平均。"
             "本表 ΔFz 与独立单摄基线做差；最后一列为双摄最后两轮的最大轴向力变化，属于网格敏感性记录。", "",
             "|间隙 mm|粗算 Fz mN|Maxwell Fz mN|总力误差 %|粗算 ΔFz mN|Maxwell ΔFz mN|末轮力变化 mN|",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['gap_mm']:g}|{row['coarse_z_mN']:.3f}|{row['maxwell_mean_z_mN']:.3f}|"
                     f"{row['signed_total_error_percent']:.2f}|{row['coarse_delta_z_mN']:.4f}|"
                     f"{row['maxwell_delta_z_mN']:.4f}|{row['maxwell_last_pass_z_change_mN']:.4f}|")
    lines += ["", f"独立单摄：粗算法 {diagnostics['coarse_isolated_z_mN']:.5f} mN，"
              f"Maxwell {diagnostics['maxwell_isolated_z_mN']:.5f} mN。", "",
              "![力与增量对比](" + (REF / "comparison.png").as_posix() + ")", "",
              "## 2 mm 同网格对照", "",
              f"双摄和只激活 A 摄像头的对照使用同一 {source_tets:,} 个四面体的网格。"
              "B 模组的磁铁、轭铁设为 vacuum，B 线圈电流为零。"
              "对照在导入的固定网格上求解一次，不作再次细化；它是同网格差分检查。", "",
              f"A 线圈 Maxwell 总力 {force(REF / 'Gap_2mm_CCW', 'A')[2] * 1e3:.5f} mN，"
              f"对照 {paired_baseline * 1e3:.5f} mN，增量 {paired_delta * 1e3:.5f} mN。"
              f"粗算法增量 {coarse_delta * 1e3:.5f} mN，绝对差 {diagnostics['matched_delta_absolute_error_mN']:.5f} mN，"
              f"相对差 {diagnostics['matched_delta_relative_error_percent']:.2f}%。"
              f"两个解的 A 线圈 J 采样相对差为 {current_difference:.3g}。", "",
              "## 误差检查", "",
              f"空气留白由 20 mm 扩大到 40 mm 后，2 mm 平均 Fz 为 {air40_z * 1e3:.5f} mN，"
              f"变化 {diagnostics['air20_to_air40_z_change_mN']:.5f} mN（{diagnostics['air20_to_air40_z_change_percent']:.4f}%）。"
              "这次变化同时包含两套自适应网格的差异。", "",
              f"粗算法细化到每模组 3264 单元（0.35 mm），绕组截面 5×5、每边 40 段后，"
              f"Fz 为 {fine_z * 1e3:.5f} mN，总力误差 {diagnostics['fine_035mm_error_percent']:.2f}%；"
              f"增量误差仍为 {diagnostics['fine_035mm_delta_relative_error_percent']:.2f}%。"
              "细化未消除距离增量的偏差。需要进一步改进并验证粗磁场近似。", "",
              "小的横向分量在自适应历史中有明显波动，完整 XYZ 数值保存在 CSV/JSON，"
              "本次没有据此认定横向力准确。能量门限为 0.5%，不等于力误差的界限。", "",
              "材料为相同的示例参数：Br=1.2 T、回线 μr=1.05 和示例钢 B-H 曲线。"
              "这是两个求解方法在同一假设下的比较，没有真实器件材料测量或实验校准。", "",
              "## 原始依据与项目", "",
              "[完整数值 CSV](" + (REF / "comparison.csv").as_posix() + ") · "
              "[汇总 JSON](" + (REF / "comparison_summary.json").as_posix() + ") · "
              "[方法与复现命令](" + (ROOT / "docs/maxwell_validation.md").as_posix() + ")", ""]
    for _, design in CASES + [(None, "Isolated_CCW"), (None, "Gap_2mm_OnlyA"), (None, "Gap_2mm_Air40")]:
        folder = REF / design
        provenance = json.loads((folder / "provenance.json").read_text())
        project_path = Path(provenance["project"]).as_posix()
        lines.append(f"- {design}：[Maxwell 项目]({project_path}) · "
                     f"[力]({(folder / 'forces.json').as_posix()}) · "
                     f"[收敛]({(folder / 'convergence.txt').as_posix()}) · "
                     f"[运行记录]({(folder / 'profile.txt').as_posix()}) · "
                     f"[材料场检查]({(folder / 'constitutive_check.json').as_posix()})")
    lines += ["", "旧 outputs/maxwell 开发目录的材料坐标试算已标记无效。"
              "扫距批项目中曾异常退出的 2.5 mm 记录不参与比较；正式该点来自独立重建后的正常收敛项目。"
              "已完成的参考设计按各自 profile 核验，批项目中的剩余重复算例已停止。", ""]
    (REF / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
