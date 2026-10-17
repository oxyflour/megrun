"""Write the single-camera reuse experiment report from saved measured artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/single_camera_reuse"


def main():
    original = json.loads((OUT / "direct_prediction_summary.json").read_text())
    early_increment = json.loads((OUT / "incremental/summary.json").read_text())
    refined = json.loads((OUT / "refined_prediction/direct_prediction_summary.json").read_text())
    increment = json.loads((OUT / "refined_prediction/incremental/summary.json").read_text())
    record_file = OUT / "reused_validation_record.json"
    if not record_file.exists():
        record_file = OUT / "refined_validation_record.json"
    record = json.loads(record_file.read_text())
    if not record.get("completed_UTC"):
        raise ValueError("Native refined-single pipeline is not complete")
    experiments = [dict(name="旧单摄场直接复用", summary=original, mode=None),
                   dict(name="旧单摄场＋增量响应", summary=early_increment, mode="linear"),
                   dict(name="加密单摄场直接复用", summary=refined, mode=None),
                   dict(name="加密单摄场＋增量响应", summary=increment, mode="linear")]
    # Resolution checks are reported separately; they are not pair-force fitting.
    for cell in (.5, .35):
        path = OUT / f"refined_prediction/incremental_{cell:g}/summary.json"
        if path.exists():
            experiments.append(dict(name=f"加密单摄场＋{cell:g} mm 增量单元", summary=json.loads(path.read_text()), mode="linear"))
    rows = []
    for experiment in experiments:
        summary, mode = experiment["summary"], experiment["mode"]
        included = [row for row in summary["rows"] if mode is None or row.get("mode")==mode]
        rows.append(dict(experiment=experiment["name"],
            max_distance_change_error_uN=max(row["absolute_distance_change_error_uN"] for row in included),
            max_total_error_percent=max(row["total_error_percent"] for row in included),
            reference_comparison_within_10uN=all(row["absolute_distance_change_error_uN"]<=10 for row in included),
            cells=summary.get("cells"), data=included,
            own_response_quadrature_change_uN=summary.get("max_own_response_quadrature_change_uN")))
    final = rows[-1]
    resolution_change = None
    if len(rows) > 4:
        coarse_data = rows[3]["data"]
        resolution_change = max(abs(row["predicted_change_from_2mm_uN"] - next(
            item["predicted_change_from_2mm_uN"] for item in coarse_data
            if item["camera"] == row["camera"] and item["gap_mm"] == row["gap_mm"])) for row in final["data"])
    build_path = Path(record["build_record"])
    build = json.loads(build_path.read_text())
    stages = {item["name"]:item["seconds"] for item in record["stages"]}
    interpolation_path = OUT / "interpolation_check.json"
    interpolation = json.loads(interpolation_path.read_text()) if interpolation_path.exists() else None
    query_path = Path(interpolation["query_file"]) if interpolation else OUT / "new_gap_predictions.json"
    query = json.loads(query_path.read_text()) if query_path.exists() else None
    cached_path = query_path.with_stem(query_path.stem+"_cached")
    cached = json.loads(cached_path.read_text()) if cached_path.exists() else None
    cli_runtime_path = OUT / "cached_cli_runtime_0.5.json"
    cli_runtime = json.loads(cli_runtime_path.read_text()) if cli_runtime_path.exists() else None
    source = build_path.parent / "SingleFieldReference"
    frozen_path = source / "frozen_source_audit.json"
    frozen = json.loads(frozen_path.read_text()) if frozen_path.exists() else None
    result = dict(experiments=rows, final_experiment=final["experiment"],
        final_reference_distance_comparison_passed=final["reference_comparison_within_10uN"],
        accuracy_of_absolute_isolated_to_pair_increment_certified=False,
        exact_or_hardware_accuracy_certified=False, independent_other_camera_models_validated=False,
        native_perturbation_response_library_created=False,
        paired_force_fit_or_scale_applied=False,
        refined_isolated_build_seconds=build["seconds"], refined_isolated_stages_seconds=stages,
        native_source_record=str(record_file), source_project=str(build_path.parent / "SingleFieldReference.aedt"),
        frozen_source_audit=frozen, interpolation_check=interpolation,
        max_response_cell_refinement_distance_change_uN=resolution_change,
        new_gap_predictions=query, cached_gap_predictions=cached,
        cached_cli_runtime=cli_runtime,
        stopped_from_scratch_trial_seconds=1235.72)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    for name, data, style in [("Direct single-field reuse", refined["rows"], "s--"),
                              ("Incremental response", final["data"], "o--")]:
        for label in "AB":
            part = sorted([row for row in data if row["camera"]==label], key=lambda row:row["gap_mm"])
            axes[0].plot([row["gap_mm"] for row in part], [row["predicted_change_from_2mm_uN"] for row in part], style, label=f"{name} {label}")
            axes[1].plot([row["gap_mm"] for row in part], [row["absolute_distance_change_error_uN"] for row in part], style, label=f"{name} {label}")
    for label in "AB":
        part = sorted([row for row in refined["rows"] if row["camera"]==label], key=lambda row:row["gap_mm"])
        axes[0].plot([row["gap_mm"] for row in part], [row["reference_change_from_2mm_uN"] for row in part], "^-", label=f"Pair Maxwell {label}")
    axes[1].axhline(10, color="black", linestyle=":", label="10 microN comparison target")
    axes[0].set(ylabel="Fz(gap) - Fz(2 mm) (microN)")
    axes[1].set(ylabel="Absolute reference difference (microN)")
    for ax in axes:
        ax.set(xlabel="EM envelope gap (mm)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(OUT / "single_camera_reuse.png", dpi=180)
    plt.close(fig)
    lines = ["# 单摄 Maxwell 结果预测双摄力：实算验证", "",
        "两个相同 B0500 模组同向沿 +X 放置，0.1 A、100 匝。"
        "间隙按 8.8 mm 电磁包围盒定义。本次只补算单摄 Maxwell，双摄使用之前保存的独立细参考。", "",
        "本次补算使用正常中止后保留并审计的第四轮解；原生状态为 Aborted，"
        "未达到原设定的完整自适应停止条件。固定基线相消不能消除局部场网格偏差。", "",
        f"最终方法：**{final['experiment']}**。三个间隙相对 2 mm 的 Fz 变化，"
        f"最大参考差 **{final['max_distance_change_error_uN']:.3f} μN**；"
        f"≤10 μN 数值比较通过：**{final['reference_comparison_within_10uN']}**。", "",
        "## 方法与独立性", "",
        "直接复用从单摄 Maxwell 导出线圈 J 和邻居位置的空气域 B，积分 ∫J×B dV，"
        "加上单摄原生 Lorentz 力。它固定单摄磁化状态，不能包含两模组互相改变轭铁磁化的效应。", "",
        "增量修正在单摄 Maxwell 的 H 工作点计算 B-H 切线响应，使用近似体积格林算子"
        "求磁化增量和相互反馈，只把增量场加到原生单摄场上。"
        "未使用双摄力拟合、乘比例或调材料。本次尚未建立由 Maxwell 外场扰动求解得到的响应库；"
        "增量空间算子仍是高斯体积近似，不能称为原生 Maxwell 有限元矩阵降阶。", "",
        "另算锚定的非线性增量作为线性化检查。该检查只在同一近似空间算子内部进行，"
        "不能独立证明对 Maxwell 的精度。", "",
        "## 各阶段结果", "",
        "|方法|磁性单元数|最大总 Fz 误差 %|最大距离变化参考差 μN|≤10 μN 比较|",
        "|:---|---:|---:|---:|:---:|"]
    for row in rows:
        lines.append(f"|{row['experiment']}|{row['cells'] or '—'}|{row['max_total_error_percent']:.4f}|"
            f"{row['max_distance_change_error_uN']:.3f}|{row['reference_comparison_within_10uN']}|")
    if resolution_change is not None:
        lines += ["", f"响应单元由 0.7 mm 细化至最后报告的分辨率，距离差分最多变化 {resolution_change:.3f} μN。"
            "这只是离散敏感性检查，不能证明空间近似算子的正确性。"]
    lines += ["", "总力约 33 mN，距离变化只有几十 μN。总力接近不能代替距离效应的验收。", "",
        "## 最终距离结果", "",
        "|间隙 mm|模组|预测 Fz mN|双摄参考 Fz mN|预测相对 2 mm 的变化 μN|参考变化 μN|绝对差 μN|",
        "|---:|:---:|---:|---:|---:|---:|---:|"]
    for row in final["data"]:
        lines.append(f"|{row['gap_mm']:g}|{row['camera']}|{row['predicted_Fz_mN']:.6f}|{row['reference_Fz_mN']:.6f}|"
            f"{row['predicted_change_from_2mm_uN']:.3f}|{row['reference_change_from_2mm_uN']:.3f}|"
            f"{row['absolute_distance_change_error_uN']:.3f}|")
    lines += ["", "![距离变化及误差](" + (OUT / "single_camera_reuse.png").as_posix() + ")", "",
        "## 场与积分精度", "",
        "旧单摄解最后一轮 Fz 还变化了 18.89 μN，且空气域没有专门覆盖邻居查询位置。"
        "最终单摄参考复用旧单摄已完成的自适应网格，线圈目标长度 0.22 mm，并在 5–16.6 mm 两侧查询范围内以 0.6 mm 加密空气，"
        "使用非模型 primitive box 实施网格约束，物理实体体积检查前后一致。"
        "非线性残差保持 1e-4，原定能量目标 0.25%、逐轮 Fz 变化目标 5 μN。", "",
        "最初从头生成的 440,935 单元版本，在第二轮仍需较长非线性求解，"
        "运行 1235.72 秒后由代理停止，未作为接受数据。该工程和失败记录保留在 refined_single 与 refined_validation_record.json。"
        "最终版本通过导入旧单摄网格保留已有局部加密；初始单元数见 build_record。", "",
        f"新场库直接漏磁积分粗/细采样差最大为 {refined['max_direct_quadrature_change_uN']:.3f} μN。"
        "自响应增量粗/细积分检查见各版本 summary。这些差异只检查积分采样，不是有限元或空间近似误差界限。", "",
        "## 成本与复用", "",
        f"单摄场库新建与初始网格：{build['seconds']:.2f} 秒；"
        f"原生 Maxwell 求解：{stages['native_solve']:.2f} 秒；提取：{stages['native_extract']:.2f} 秒；"
        f"场库导出：{stages['library_export']:.2f} 秒。"
        "这些属于一次离线成本。增量算子和力响应系数可以复用；当前原型仍为每个距离构建跨模组矩阵和积分系数。"
        "不能把积分时间直接当作完整新距离查询成本，也未据此宣称普遍加速倍数。", "",
        "## 结论边界", "",
        "相对 2 mm 的做差消除了固定单摄力基线。它不验证每个间隙相对无限远单摄的绝对干扰增量。"
        "双摄参考最后两步 Fz 变化≤10 μN，但这不是严格误差上界；接近这一尺度的结果需要更细或同网格差分验证。", "",
        "1/2/3 mm 双摄结果已经在之前的工作中见过，本次用它们作数值比较。"
        "公式不使用双摄力参数，但这也不是完全未知数据上的盲测。"
        "仅验证当前模组、方向、电流和材料；不同 camera 型号需要各自单摄工作点和响应描述。"
        "横向小力、转角、位移方向改变、强饱和变化尚未验证。", "",
        "当前原型适合快速探索间隙趋势，但尚未达到 ≤10 μN 的差分目标。"
        "继续工作的重点应是用单摄 Maxwell 外场扰动求解建立响应描述，"
        "再用未参与构建的双摄距离检验。该路线的精度与成本本轮尚未验证。", "",
        "## 文件与复现", "",
        f"- [汇总数据]({(OUT / 'summary.json').as_posix()})",
        f"- [原生求解与导出记录]({record_file.as_posix()})",
        f"- [加密单摄 Maxwell 项目]({(build_path.parent / 'SingleFieldReference.aedt').as_posix()})",
        f"- [直接复用对照]({(OUT / 'refined_prediction/report_zh.md').as_posix()})", "",
        "先运行 scripts/build_reused_single_camera_field_reference.py，再运行 scripts/validate_single_camera_field_library.py "
        "--source-directory reused_single --record-name reused_validation_record.json。"
        "两者对既有文件采用显式恢复检查。新实验应选择新目录。"
        "增量方法可用 scripts/evaluate_incremental_camera_response.py 单独复算，场库导出支持 --cell-mm 指定增量离散。", ""]
    if frozen is not None:
        lines += ["保留第四轮场的本次恢复命令：", "",
            "```powershell", ".venv\\Scripts\\python.exe scripts\\validate_single_camera_field_library.py "
            "--source-directory reused_single --record-name reused_validation_record.json --frozen-pass 4", "```", ""]
    if interpolation is not None:
        lines += ["## 无需启动 Maxwell 的新距离查询", "",
            "scripts/predict_camera_pair_from_library.py 读取单摄空气域规则网格，"
            "不读取双摄力文件，也不启动 Maxwell。",
            f"相同 1/2/3 mm 下，完整插值查询与原生逐点采样方法的 Fz 最大差 "
            f"{interpolation['max_full_prototype_vs_native_queries_difference_uN']:.3f} μN；"
            f"≤2 μN 插值目标通过：{interpolation['interpolation_comparison_within_2uN']}。",
            f"[新距离查询结果]({Path(interpolation['query_file']).as_posix()}) · "
            f"[插值检查]({interpolation_path.as_posix()})", ""]
    if query is not None:
        lines += ["首次新间隙查询耗时（含跨模组矩阵与力系数生成）："
            +", ".join(f"{row['gap_mm']:g} mm / {row['new_gap_query_seconds']:.3f} s" for row in query["rows"])+"。"
            f"单摄响应算子加载或构建另计 {query['operator_load_or_build_seconds']:.3f} s。", ""]
    if cached is not None:
        lines += ["已有间隙系数缓存后的重复查询："
            +", ".join(f"{row['gap_mm']:g} mm / {row['new_gap_query_seconds']:.3f} s" for row in cached["rows"])+"。"
            "以上是进程内查询计时，不含 Python 启动、STEP 加载与场文件读取，不能解释为整个 CLI 的延迟。", ""]
    if cli_runtime is not None:
        lines += [f"完整启动 CLI 并重复计算五个缓存距离的实测时间为 {cli_runtime['cli_wall_seconds']:.3f} s，"
            "包含解释器启动、STEP 与场文件读取。", ""]
    if frozen is not None:
        lines += ["## 保留场解状态", "",
            f"本次使用第 {frozen['selected_completed_pass']} 轮已完成解，"
            f"{frozen['retained_tetrahedra']:,} 个四面体；能量误差 {frozen['source_energy_error_percent']:.5f}%，"
            f"最后一步 Fz 变化 {frozen['last_Fz_change_uN']:.3f} μN。"
            "后续求解通过 Ansys 自带 desktopproxy -abort 正常中止。"
            "原生状态为 Aborted，未通过完整自适应停止条件，不能称为原生收敛。", "",
            "核对已保留的 LastAdaptive 力与第 4 轮力缓存、convergence 与原生 asol 轮号、"
            "导出网格与原生记录的单元数、场目录、SI 坐标、安匝与本构关系后，"
            "才接受它用于本轮差分探索。原生场文件与提取记录均保存 SHA-256。"
            "固定基线相消并不能消除局部场的网格误差；新旧单摄场结果差异单独列在上表。", "",
            f"[保留场解审计]({frozen_path.as_posix()})", ""]
    (OUT / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["final_experiment", "final_reference_distance_comparison_passed"]}))


if __name__ == "__main__":
    main()
