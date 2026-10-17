"""Publish the achieved comparisons and failed response attempt faithfully."""
from pathlib import Path
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.refine_pair_difference_reference import read,save,digest,utc


def main():
    out=ROOT/"outputs/native_equivalent_response_pilot_20261003"
    equal=read(out/"equal_sample_loading_comparison.json")
    loading=read(out/"run_record.json")
    diag=read(out/"diagnostic_record.json")
    tangential=read(ROOT/"outputs/native_response_pilot_20261003/run_record.json")
    precision=read(ROOT/"outputs/delta_precision_20261003/precision_summary.json")
    holdout=read(ROOT/"outputs/equivalent_source_loading_20261003/heldout_audit.json")
    raw_log=out/"NativeResponsePilot.aedt.batchinfo/NativeResponsePilot-RSM_16953.log"
    raw=raw_log.read_text(encoding="utf-8",errors="replace")
    assert "Design:PilotZero" in raw and "Status: Normal Completion" in raw
    assert "0.66347" in raw and "264695" in raw
    assert "PilotHalf_solve" not in diag["stages"] and "PilotFull_solve" not in diag["stages"]
    failure=dict(audited_UTC=utc(),batch_native_status="Normal Completion",
        adaptive_converged=False,completed_passes=1,tetrahedra=264695,energy_error_percent=0.66347,
        native_log=str(raw_log),native_log_sha256=digest(raw_log),
        postprocess_failure=diag["failure"],profile_missing_after_reopen=not (out/"PilotZero/profile.txt").exists(),
        native_asol_after_failed_reopen_sha256=digest(out/"NativeResponsePilot.aedtresults/PilotZero.asol"),
        paired_response_available=False,half_or_full_solve_submitted=False,
        root_cause_confirmed=False,
        hypothesis="Absolute self-project mesh dependency may invalidate solved data on project reopen",
        future_build_change="Use This Project* for internal mesh links; not validated by a new solve",
        automatically_retried=False)
    save(out/"response_extraction_failure_audit.json",failure)
    stage_min=sum(r["elapsed_seconds"]/60 for r in (tangential,loading,diag))
    calendar=(datetime.fromisoformat(diag["ended_UTC"])-datetime.fromisoformat(tangential["started_UTC"])).total_seconds()/60
    report_utc=utc()
    report_calendar=(datetime.fromisoformat(report_utc)-datetime.fromisoformat(tangential["started_UTC"])).total_seconds()/60
    ana=equal["rows"]["analytic_source_basis"]
    native=equal["rows"]["native_finite_sources"]
    ref=precision["levels"]["1"]
    lines=["# 摄像头磁仿真加速：本轮可比较结果", "",
        "本轮取得了外场加载的同条件比较，尚未取得 Maxwell 原生半幅/全幅力响应库，也没有证明双摄距离差分提精度。", "",
        "## 直接可比的加载结果", "",
        "g=1 mm、接收模组 A。两种实现使用相同的 1176 个体积采样点、16384 个线圈积分点、积分权重和冻结的 Maxwell 单摄电流密度。目标为冻结单摄场库中的实际邻居场。", "",
        "| 外场实现 | 场 RMS 参考差 | 直接漏磁 Fz 表达差 / μN |",
        "|---|---:|---:|",
        f"| 离线 96 源表示 | {ana['relative_field_RMS_to_desired']*100:.4f}% | {ana['direct_Fz_difference_to_desired_uN']:+.6f} |",
        f"| Maxwell 原生有限源加载 | {native['relative_field_RMS_to_desired']*100:.4f}% | {native['direct_Fz_difference_to_desired_uN']:+.6f} |", "",
        "原生加载门槛为场 RMS 差 ≤2%、直接力表达差 ≤1 μN，两项均失败。输入只有原生场样本，未拟合双摄力。这个直接力项尚未包括接收模组磁化响应，不能当成距离 ΔF。", "",
        f"离线源表示在 1/2 mm、A/B 四种位置的独立采样点最大场 RMS 差为 {holdout['max_disjoint_heldout_field_relative_RMS']*100:.4f}%。审计排除了 3 个与训练重合的线圈点，未重新拟合或修改源系数。它只验证相同冻结场的表达，不是独立 FEM 精度验证。", "",
        "原生仅外部源对照为固定一轮、264695 四面体；状态 Normal Completion，自适应未收敛，能量误差指标 359.866%。虚拟源 μr=1，最大等效剩磁约69.8 T，是场表达代理，不能解释为真实材料参数。网格解析、有限源表达和边界共同引入的差异尚未分离。", "",
        "## 力响应试验为何没有产出", "",
        "加载失败后，单独尝试比较同一个实际原生源场下的自响应，未放宽邻居加载验收门槛。零幅基准原生批处理正常结束，固定一轮，能量误差0.66347%，自适应未收敛；重新打开工程提取时，profile 不存在，asol 变为无已完成解的状态，未取得可用 Fz/B/H。半幅、全幅均未提交。", "",
        "仅保存了批处理原生日志与后处理失败证据，不能声称保留了可复用的零幅完整场解。原始单摄、双摄工程及冻结预测均未修改。当前怀疑绝对路径指向自身工程的网格依赖参与了失效，原因尚未验证；只修正了未来构建脚本的内部工程引用，没有重跑。", "",
        "另一路六面切向 H 加载试验也未通过空气域核验，默认/翻转 V 后场 RMS 差分别为100.81%/185.66%。无效几何、原生记录和各次失败日志保留，没有把该路线记为加载成功。", "",
        "## 距离差分目标的当前证据", "",
        "ΔF(g)=Fz(g)−Fz(2 mm)，与上面的外场直接漏磁力、加载响应 dF 分开。", "",
        f"第一档补算的1 mm距离差分：A={ref['delta_Fz_uN']['A']:.6f} μN，B={ref['delta_Fz_uN']['B']:.6f} μN。相对旧参考移动最多{ref['max_absolute_delta_change_uN']:.6f} μN。",
        f"原冻结近似预测相对该档1 mm参考的最大差为{max(abs(x) for x in ref['existing_approximate_prediction_difference_uN'].values()):.6f} μN，仍超过10 μN通过线。这是更新参考后的比较，不是加速模型改进。1–2 μN参考稳定性尚未证明。", "",
        "思路①的相同停止条件加速仍只有1.18倍；2.135倍涉及同时改变停止规则。本轮没有新增思路①验证结果，也没有进行1.5/2.5 mm盲验。", "",
        "## 完整计时与停止状态", "",
        f"- 前面的1/2 mm参考补算：{precision['completed_stage_seconds']/60:.2f} min。",
        f"- 切向 H 加载尝试：{tangential['elapsed_seconds']/60:.2f} min，包含失败、修复和等待。",
        f"- 有限源构建及加载核验：{loading['elapsed_seconds']/60:.2f} min。",
        f"- 实测源场响应诊断至后处理失败：{diag['elapsed_seconds']/60:.2f} min。",
        f"- 上述加载/响应尝试累计记录时间：{stage_min:.2f} min。从首次加载构建到最后诊断停止的日历时间：{calendar:.2f} min，含开发排错与离线分析间隔。",
        f"- 从首次加载构建到本比较报告保存：{report_calendar:.2f} min，另包含失败审计和报告整理。报告保存时间（UTC）：{report_utc}。",
        "- 原生任务已停止；第二档双摄参考仍在创建原生工程前搁置。未自动重试零幅，未生成正式响应库。", "",
        "## 下一步的投入顺序", "",
        "先修复解保存/重开提取，并用已知均匀场和一个非均匀场核验加载。再解决真实邻居场的原生加载差，通过上述2%/1 μN门槛后，才值得投入半幅/全幅力与向外增量场的成对响应。响应及双向耦合构建完成后，以未用于构建的1.5/2.5 mm双摄结果验收ΔF。≤10 μN为通过线，≤5 μN仍是研发目标。", "",
        "预计完整仿真任务超过30分钟或耗时不确定且可能超过30分钟时，提交前先获得用户确认；按整组等待计时，不拆分规避。", "",
        "证据：equal_sample_loading_comparison.json、response_extraction_failure_audit.json、run_record.json、diagnostic_record.json，以及各阶段原生日志。"]
    (out/"comparison_report_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(out/"handoff_status.json",dict(updated_UTC=utc(),external_loading_comparison_available=True,
        native_response_library_built=False,final_distance_accuracy_accepted=False,
        simulation_queue_state="Stopped; reference level 2 held before native build",
        prototype_recorded_minutes=stage_min,prototype_calendar_minutes=calendar,
        calendar_minutes_through_report=report_calendar,
        original_sources_unchanged=all(diag["original_sources_unchanged"].values()),
        report=str(out/"comparison_report_zh.md")))


if __name__=="__main__":main()
