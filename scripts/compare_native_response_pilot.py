"""Compare native perturbations and the existing Gaussian response on one mode.

External-load response dF is distinct from the camera distance difference Delta F.
No native or paired force is used to fit an operator or alter a prediction.
"""
import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_native_response_pilot import incident, SOURCE, utc
from scripts.refine_pair_difference_reference import read, save, digest
from camera_mag.interaction import MU0, volume_operator
from camera_mag.response_library import read_vector_field, integrate_force, native_susceptibility, force_response_operator


def compare(output):
    loading_record = read(output/"run_record.json")
    diagnostic_path = output/"diagnostic_record.json"
    diagnostic = diagnostic_path.exists()
    record = read(diagnostic_path) if diagnostic else loading_record
    build = read(output/"build.json") if (output/"build.json").exists() else {}
    finite_source = "source_bank_sha256" in build
    summary = dict(updated_UTC=utc(), run_active=record["active"],
        total_wall_minutes=record["elapsed_seconds"]/60,
        original_sources_unchanged=all(record["original_sources_unchanged"].values()),
        production_prediction_changed=False, paired_force_fit=False,
        metric="External-mode dF(load)=Fz(load)-Fz(0); distinct from distance Delta F(g)",
        final_distance_accuracy_accepted=False,
        neighbour_representation_accepted=False, production_library_accepted=False,
        diagnostic_on_measured_native_load=diagnostic,
        limitations=["165,544-cell one-pass pilot; no force stability acceptance",
            "Degree-2 truncation has 38.14% field representation error; it is not the full neighbour field",
            "Finite tangential-H boundary differs from the original natural boundary",
            "No blind 1.5/2.5 mm pair validation performed"])
    if finite_source:
        summary["limitations"] = ["One-pass augmented-geometry pilot; no adaptive force stability acceptance",
            "Virtual mu_r=1 sources realize a field-only fitted loading mode; they are not the physical second camera",
            "The far air boundary and additional source geometry differ from the frozen single-camera baseline",
            "Only one external mode is tested; no full two-way response library or blind pair validation"]
        summary["loading_method"] = "Native prescribed-magnetization finite sources; actual source-only field supplied to both response models"
    if diagnostic:
        summary["loading_attempt_wall_minutes"] = loading_record["elapsed_seconds"]/60
        summary["simulation_stages_wall_minutes"] = summary["loading_attempt_wall_minutes"]+summary["total_wall_minutes"]
        from datetime import datetime
        summary["calendar_minutes_from_loading_start"] = (datetime.fromisoformat(record["ended_UTC"])-datetime.fromisoformat(loading_record["started_UTC"])).total_seconds()/60
        summary["diagnostic_failure"] = record.get("failure")
        summary["limitations"].insert(0,"Desired neighbour loading gate failed; this compares responses to the actual measured native source field only")
    cases = {}
    for name in ("VacuumCheck", "NaturalControl", "PilotZero", "PilotHalf", "PilotFull"):
        path = output/name/"result.json"
        if path.exists():
            cases[name] = read(path)
    summary["native_cases"] = cases
    if "VacuumCheck" in cases:
        summary["vacuum_loading_gate_met"] = cases["VacuumCheck"]["loading_gate_met"]
        if finite_source:
            summary["native_loading_field_relative_RMS"] = cases["VacuumCheck"]["native_loading_field_relative_RMS"]
            summary["native_loading_direct_Fz_difference_uN"] = cases["VacuumCheck"]["native_loading_direct_Fz_difference_uN"]
    old_fz = read(SOURCE/"forces.json")["Force_A.Force_z"]["values"][0]
    if "NaturalControl" in cases:
        summary["natural_control_minus_old_source_uN"] = (cases["NaturalControl"]["Fz_N"]-old_fz)*1e6
    if "PilotZero" in cases and "NaturalControl" in cases:
        summary["zero_tangential_minus_natural_control_uN"] = (cases["PilotZero"]["Fz_N"]-cases["NaturalControl"]["Fz_N"])*1e6
    if "PilotZero" in cases:
        summary["zero_load_baseline_minus_frozen_source_uN"] = (cases["PilotZero"]["Fz_N"]-old_fz)*1e6
    loads = [(name,amplitude) for name,amplitude in (("PilotHalf",.5),("PilotFull",1.)) if name in cases]
    if "PilotZero" in cases and loads:
        counts = [int(cases[name]["native"]["tet"]) for name in ["PilotZero"]+[name for name,_ in loads]]
        expected_count = int(build.get("initial_tetrahedra",165544))
        summary["same_imported_mesh_counts"] = len(set(counts))==1 and counts[0]==expected_count
        summary["camera_case_tetrahedra"] = counts
        zero_force = cases["PilotZero"]["Fz_N"]
        zero = output/"PilotZero"
        with np.load(zero/"sampling.npz", allow_pickle=True) as data:
            points = data["points_m"]
            nb, nc = int(data["body_count"]), int(data["coil_count"])
            weights = data["coil_weights_m3"]
            centers = data["body_centers_m"]
            volumes = data["body_volumes_m3"]
            ids = data["material_ids"]
        body, coil, air = slice(0,nb), slice(nb,nb+nc), slice(nb+nc,None)
        b0 = read_vector_field(zero/"B.fld", points)
        h0 = read_vector_field(zero/"H.fld", points)
        current = read_vector_field(zero/"J.fld", points[coil])
        if finite_source:
            incident_b = read_vector_field(output/"VacuumCheck/B.fld",points)
            incident_h = read_vector_field(output/"VacuumCheck/H.fld",points)
        else:
            incident_h = incident(points)
            incident_b = MU0*incident_h
        response = dict(air_points_m=points[air],body_points_m=centers,
            incident_H_A_per_m=incident_h[air])
        increments = {}
        summary["current_max_relative_change"] = 0.
        for name,amplitude in loads:
            label = "half" if amplitude==.5 else "full"
            loaded_b = read_vector_field(output/name/"B.fld",points)
            loaded_h = read_vector_field(output/name/"H.fld",points)
            loaded_j = read_vector_field(output/name/"J.fld",points[coil])
            df = cases[name]["Fz_N"]-zero_force
            induced = loaded_b[air]-b0[air]-amplitude*incident_b[air]
            increments[label] = (df,induced)
            response["outward_"+label+"_delta_B_T"] = induced
            response[label+"_delta_M_A_per_m"] = (loaded_b[body]-b0[body])/MU0-(loaded_h[body]-h0[body])
            response[label+"_dF_N"] = df
            summary["native_"+label+"_load_dF_uN"] = df*1e6
            summary["outward_incremental_field_"+label+"_RMS_uT"] = float(np.sqrt(np.mean(np.sum(induced**2,axis=1)))*1e6)
            summary["native_force_quadrature_"+label+"_difference_uN"] = float((integrate_force(current,loaded_b[coil]-b0[coil],weights)[2]-df)*1e6)
            summary["current_max_relative_change"] = max(summary["current_max_relative_change"],float(np.linalg.norm(loaded_j-current)/np.linalg.norm(current)))
        if len(increments)==2:
            summary["native_force_linearity_defect_uN"] = (increments["full"][0]-2*increments["half"][0])*1e6
            summary["outward_incremental_field_linearity_relative_RMS"] = float(np.linalg.norm(increments["full"][1]-2*increments["half"][1])/max(np.linalg.norm(increments["full"][1]),1e-30))
        np.savez_compressed(output/"native_incremental_response.npz",**response)
        summary["response_file"] = str(output/"native_incremental_response.npz")
        # Use the existing model equations and native zero-load H/J; no force fit.
        begin = perf_counter()
        active = ids!="copper"
        c, v, material = centers[active], volumes[active], ids[active]
        n, size = len(c), c.size
        table = np.loadtxt(SOURCE/"maxwell_bh.csv",delimiter=",",skiprows=1)
        tangent = native_susceptibility(h0[body][active],material,table)
        local = np.linalg.solve(np.eye(3)+tangent/3,tangent)
        with threadpool_limits(limits=4,user_api="blas"):
            self_k = volume_operator(c,c,v,target_volumes=v)
            off_self = self_k+np.eye(size)/3
            matrix = np.eye(size)-np.einsum("nij,njmk->nimk",local,off_self.reshape(n,3,n,3)).reshape(size,size)
            factor = lu_factor(matrix)
            dm = lu_solve(factor,np.einsum("nij,nj->ni",local,incident_h[body][active]).ravel())
            force_op = force_response_operator(points[coil],current,weights,c,v)
            direct = integrate_force(current,incident_b[coil],weights)
            approximate_force = direct+force_op@dm
            outward_k = volume_operator(points[air],c,v,flux_density=True)
            approximate_outward = (outward_k@dm).reshape(-1,3)
        summary["approximate_model_evaluation_seconds"] = perf_counter()-begin
        summary["gaussian_full_load_dF_uN"] = float(approximate_force[2]*1e6)
        summary["direct_incident_full_dF_uN"] = float(direct[2]*1e6)
        summary["gaussian_induced_full_dF_uN"] = float((force_op@dm)[2]*1e6)
        for name,amplitude in loads:
            label = "half" if amplitude==.5 else "full"
            df,induced = increments[label]
            summary["gaussian_minus_native_"+label+"_dF_uN"] = float((amplitude*approximate_force[2]-df)*1e6)
            summary["gaussian_outward_"+label+"_field_difference_relative_RMS"] = float(np.linalg.norm(amplitude*approximate_outward-induced)/max(np.linalg.norm(induced),1e-30))
            summary["gaussian_outward_"+label+"_field_difference_RMS_uT"] = float(np.sqrt(np.mean(np.sum((amplitude*approximate_outward-induced)**2,axis=1)))*1e6)
        summary["native_incremental_response_sha256"] = digest(output/"native_incremental_response.npz")
    save(output/"comparison.json",summary)
    lines = ["# Maxwell 单摄非均匀外场响应诊断", "", f"本次响应诊断完整时间：{summary['total_wall_minutes']:.2f} min。",
        "原始工程和冻结预测未修改；失败记录和原生求解状态已保留。", "",
        "本试验比较外场加载响应 dF(load)=Fz(load)-Fz(0)，不是距离差分 Delta F(g)。",
        "固定一轮，未验收自适应收敛或力的加密稳定性。", ""]
    if diagnostic:
        lines += [f"此前外场加载构建与核验：{summary['loading_attempt_wall_minutes']:.2f} min；两组执行合计：{summary['simulation_stages_wall_minutes']:.2f} min。",
            f"从加载构建开始至诊断完成的日历时间：{summary['calendar_minutes_from_loading_start']:.2f} min，包含诊断设计和离线分析等待。", "",
            ("**目标邻居场的原生加载核验失败，未放宽门槛。响应诊断使用实际导出的原生源场，未取得可用的成对响应结果。**" if not loads else
             "**目标邻居场的原生加载核验失败，未放宽门槛。以下只比较实际导出的同一个原生源场激励下的响应，不是完整邻居场响应库。**"), "",
            f"原生加载场 RMS 差：{summary['native_loading_field_relative_RMS']*100:.4f}%；直接漏磁 Fz 表达差：{summary['native_loading_direct_Fz_difference_uN']:.6f} uN。", ""]
    if "gaussian_full_load_dF_uN" in summary:
        lines += ["| 实际外场幅度 | 原生 Maxwell dF / uN | 现有高斯近似 dF / uN | 近似减原生 / uN |","|---|---:|---:|---:|"]
        for label,amplitude in (("half",.5),("full",1.)):
            if "native_"+label+"_load_dF_uN" in summary:
                lines.append(f"| {'半幅' if label=='half' else '全幅'} | {summary['native_'+label+'_load_dF_uN']:.6f} | {amplitude*summary['gaussian_full_load_dF_uN']:.6f} | {summary['gaussian_minus_native_'+label+'_dF_uN']:.6f} |")
        lines += ["",f"导入网格单元数：{summary['camera_case_tetrahedra']}；与仅外部源网格数一致：{summary['same_imported_mesh_counts']}。",
            f"近似响应求值：{summary['approximate_model_evaluation_seconds']:.2f} s。",
            f"零加载基准相对旧粗单摄工程 Fz 差：{summary['zero_load_baseline_minus_frozen_source_uN']:.6f} uN（同时改变网格、气域与求解残差，不能归因于一项）。", ""]
    if "native_force_linearity_defect_uN" in summary:
        lines += [f"全幅原生力与半幅外推的差：{summary['native_force_linearity_defect_uN']:.6f} uN。",
            f"外部增量场半幅/全幅线性偏差 RMS：{summary['outward_incremental_field_linearity_relative_RMS']*100:.4f}%。", ""]
    lines += ["## 限制", ""]+["- "+item for item in summary["limitations"]]
    lines += ["", ("加载采用场数据拟合的虚拟有限源，并先核验原生源场；虚拟源磁化固定，不代表邻居的磁化反馈。" if finite_source else
              "加载使用从邻居原生场提取的二阶谐和截断模式；完整邻居场的表达精度检查未通过。"),
              ("力响应和外部增量场已分开保存。" if "response_file" in summary else "未取得可用的成对响应，未生成增量响应库。")+"没有拟合双摄力，没有重新运行双摄参考。",
              "超过 30 min 的后续仿真须在提交前得到用户确认。"]
    (output/"report_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",default="outputs/native_response_pilot_20261003")
    args=parser.parse_args()
    result=compare(Path(args.output).resolve())
    print({key:value for key,value in result.items() if key!="native_cases"})
