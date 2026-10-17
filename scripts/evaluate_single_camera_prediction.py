"""Audit untranslated/translated single-camera fields against independent pair solves."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_mag.response_library import read_vector_field, integrate_force

REFERENCES = {
    1.: ROOT / "outputs/mesh_seeding_onepass_holdouts/gap_1mm/TightReference",
    2.: ROOT / "outputs/mesh_seeding_2mm/TightReference",
    3.: ROOT / "outputs/mesh_seeding_onepass_holdouts/gap_3mm/TightReference",
}


def native_force(folder, label):
    forces = json.loads((folder / "forces.json").read_text())
    entries = [forces[f"Force_{label}.Force_{axis}"] for axis in "xyz"]
    if any(item["unit"] not in ("N", "newton") or len(item["values"])!=1 for item in entries):
        raise ValueError("Native force report is not a single SI-newton vector")
    vector = np.array([item["values"][0] for item in entries])
    if not np.isfinite(vector).all():
        raise ValueError("Native force contains non-finite values")
    return vector


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--library", default="outputs/single_camera_reuse/isolated_library")
    p.add_argument("--output", default="outputs/single_camera_reuse")
    args = p.parse_args()
    library, output = Path(args.library).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = json.loads((library / "export_record.json").read_text())
    if not record.get("completed_UTC"):
        raise ValueError("Single-camera field exports not complete")
    spec = record["specification"]
    for name, export in record["exports"].items():
        if hashlib.sha256((library / (name + ".fld")).read_bytes()).hexdigest() != export["sha256"]:
            raise ValueError(f"Field export hash changed: {name}")
    source = Path(spec["source"])
    if hashlib.sha256((source / "provenance.json").read_bytes()).hexdigest() != spec["source_provenance_sha256"]:
        raise ValueError("Isolated source provenance changed")
    iso = native_force(source, "A")
    with np.load(library / "quadrature.npz") as arrays:
        quadratures = {level[0]: {key: arrays[level[0]+"_"+key] for key in ("points", "weights", "ideal_j")}
                       for level in spec["levels"]}
    own_points = np.concatenate([q["points"] for q in quadratures.values()])
    j = read_vector_field(library / "own_J.fld", own_points)
    b = read_vector_field(library / "own_B.fld", own_points)
    start = perf_counter()
    own_integration = {}
    cursor = 0
    for level, q in quadratures.items():
        count = len(q["points"])
        own_integration[level] = integrate_force(j[cursor:cursor+count], b[cursor:cursor+count], q["weights"]).tolist()
        cursor += count
    rows = []
    reference_records = {}
    for gap in spec["gaps_mm"]:
        reference = REFERENCES[gap]
        provenance = json.loads((reference / "provenance.json").read_text())
        manifest = json.loads((reference / "manifest.json").read_text())
        if provenance["step_sha256"] != spec["step_sha256"]:
            raise ValueError("Pair reference STEP differs from library")
        if hashlib.sha256((reference / "maxwell_bh.csv").read_bytes()).hexdigest() != spec["source_BH_sha256"]:
            raise ValueError("Pair reference B-H table differs from library")
        if (manifest["current_a"], manifest["turns"], manifest["distance_mm"]) != (spec["current_a"], spec["turns"], gap):
            raise ValueError("Reference current/turns/distance differ")
        if "Status: Normal Completion" not in (reference / "profile.txt").read_text():
            raise ValueError("Pair reference field solve failed")
        history = json.loads((reference / "force_adaptive_history.json").read_text())
        axial = np.array([history["forces"][f"Force_{label}.Force_z"] for label in "AB"])
        last_changes = np.abs(np.diff(axial[:, -3:], axis=1))*1e6
        if axial.shape[1] < 3 or last_changes.max() > 10.001:
            raise ValueError("Pair reference force history is not stable within 10 μN")
        reference_records[str(gap)] = dict(folder=str(reference), last_two_step_max_Fz_change_uN=float(last_changes.max()),
            project=provenance["project"], design=provenance["design"],
            force_file_sha256=hashlib.sha256((reference / "forces.json").read_bytes()).hexdigest())
        shift = np.array([spec["geometry_extent_m"][0] + gap*1e-3, 0, 0])
        for label, displacement in [("A", -shift), ("B", shift)]:
            field = read_vector_field(library / f"gap_{gap:g}_{label}_incident_B.fld", own_points+displacement)
            cursor = 0
            estimates = {}
            for level, q in quadratures.items():
                count = len(q["points"])
                estimates[level] = integrate_force(j[cursor:cursor+count], field[cursor:cursor+count], q["weights"])
                cursor += count
            direct = estimates["fine"]
            predicted = iso + direct
            ref = native_force(reference, label)
            row = dict(gap_mm=gap, camera=label, isolated_Fz_mN=iso[2]*1e3,
                direct_Fz_uN=direct[2]*1e6, predicted_Fz_mN=predicted[2]*1e3,
                reference_Fz_mN=ref[2]*1e3, signed_total_error_uN=(predicted[2]-ref[2])*1e6,
                total_error_percent=abs(predicted[2]/ref[2]-1)*100,
                direct_quadrature_change_uN=abs(estimates["fine"][2]-estimates["coarse"][2])*1e6)
            for axis, d, pred, actual in zip("xyz", direct, predicted, ref):
                row[f"direct_F{axis}_N"] = d
                row[f"predicted_F{axis}_N"] = pred
                row[f"reference_F{axis}_N"] = actual
            rows.append(row)
    for row in rows:
        anchor = next(item for item in rows if item["gap_mm"] == 2 and item["camera"] == row["camera"])
        row["predicted_change_from_2mm_uN"] = row["direct_Fz_uN"]-anchor["direct_Fz_uN"]
        row["reference_change_from_2mm_uN"] = (row["reference_Fz_mN"]-anchor["reference_Fz_mN"])*1e3
        row["absolute_distance_change_error_uN"] = abs(row["predicted_change_from_2mm_uN"]-row["reference_change_from_2mm_uN"])
    table = pd.DataFrame(rows)
    table.to_csv(output / "direct_prediction.csv", index=False)
    max_quad = float(table.direct_quadrature_change_uN.max())
    max_error = float(table.absolute_distance_change_error_uN.max())
    summary = dict(method="Frozen isolated Maxwell field, translated onto the other camera coil",
                   paired_force_fit_applied=False, additional_Maxwell_field_solves=0,
                   current_A=spec["current_a"], turns=spec["turns"],
                   max_distance_change_error_uN=max_error, max_direct_quadrature_change_uN=max_quad,
                   target_distance_change_error_uN=10,
                   numerical_distance_change_comparison_passed=bool(max_error<=10 and max_quad<=2),
                   interaction_response_model_included=False,
                   exact_or_hardware_accuracy_certified=False,
                   native_isolated_force_N=iso.tolist(), own_volume_integration_N=own_integration,
                   source_last_pass_Fz_change_uN=record["source_last_pass_Fz_change_uN"],
                   isolated_library_export_seconds=record["elapsed_seconds"],
                   integration_seconds=perf_counter()-start,
                   reference_records=reference_records, rows=rows)
    (output / "direct_prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for label, marker in [("A", "o"), ("B", "s")]:
        part = table[table.camera==label]
        axes[0].plot(part.gap_mm, part.predicted_Fz_mN, marker+"--", label=f"Single-camera reuse {label}")
        axes[0].plot(part.gap_mm, part.reference_Fz_mN, marker+"-", label=f"Pair Maxwell {label}")
        axes[1].plot(part.gap_mm, part.predicted_change_from_2mm_uN, marker+"--", label=f"Single-camera reuse {label}")
        axes[1].plot(part.gap_mm, part.reference_change_from_2mm_uN, marker+"-", label=f"Pair Maxwell {label}")
    for ax, ylabel in zip(axes, ["Total coil Fz (mN)", "Fz(gap) - Fz(2 mm) (microN)"]):
        ax.set(xlabel="EM envelope gap (mm)", ylabel=ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(output / "direct_prediction.png", dpi=180)
    plt.close(fig)
    lines = ["# 单摄 Maxwell 场复用：直接漏磁作用验证", "",
        "使用现有已完成的单摄 Maxwell 2022 R2 解导出场，并在两个模组的线圈位置平移采样。"
        "本次不求解新的双摄案例，不使用双摄力拟合或缩放预测。双摄参考只用于验收。", "",
        "两个相同模组同向沿 +X，0.1 A、100 匝；距离指 8.8 mm 电磁包围盒的边缘间隙。", "",
        "预测为单摄原生 Lorentz 力加上 ∫J_single×B_other_single dV。"
        "J 使用 Maxwell 导出的实际 stranded 电流密度，B 使用邻居单摄解的空气域场。"
        "这种预测固定了两个摄像头的磁化状态，尚未计入轭铁相互磁化及电流分布变化。", "",
        "## 结果", "",
        "|间隙 mm|模组|直接漏磁 ΔFz μN|预测总 Fz mN|双摄参考 Fz mN|总力误差 %|相对 2 mm 的变化误差 μN|",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{row['gap_mm']:g}|{row['camera']}|{row['direct_Fz_uN']:.3f}|{row['predicted_Fz_mN']:.6f}|"
            f"{row['reference_Fz_mN']:.6f}|{row['total_error_percent']:.4f}|{row['absolute_distance_change_error_uN']:.3f}|")
    lines += ["", f"相对 2 mm 的距离变化，最大参考差为 **{max_error:.3f} μN**；"
        f"粗/细两组体积积分的最大直接漏磁力差为 **{max_quad:.3f} μN**。",
        f"≤10 μN 距离变化目标的数值比较通过：**{summary['numerical_distance_change_comparison_passed']}**。", "",
        f"单摄旧解最后一轮 Fz 变化为 {record['source_last_pass_Fz_change_uN']:.3f} μN，"
        "因此总力预测存在单摄基线的网格误差；相对 2 mm 做差可以抵消固定的基线偏差。"
        "积分粗细比较只检查采样误差，不是 Maxwell 场解误差的界限。", "",
        "双摄参考使用此前独立加密且原生收敛的 1/2/3 mm 解，最后两步 Fz 变化均≤10 μN。"
        "这些变化量不是严格误差界限；靠近 5–10 μN 的预测差仍需要更精细或同网格差分验证。", "",
        f"现有单摄库导出用时 {record['elapsed_seconds']:.2f} 秒，"
        f"读取场并积分、比较用时 {summary['integration_seconds']:.2f} 秒。"
        "原有单摄 Maxwell 求解成本未计入，也没有据此宣称端到端加速倍数。", "",
        "![总力与距离变化](" + (output / "direct_prediction.png").as_posix() + ")", "",
        "## 原始依据", "",
        f"- [场导出记录]({(library / 'export_record.json').as_posix()})",
        f"- [完整数值]({(output / 'direct_prediction.csv').as_posix()})",
        f"- [汇总]({(output / 'direct_prediction_summary.json').as_posix()})",
        f"- [单摄原生力]({(source / 'forces.json').as_posix()})"]
    for gap, reference in REFERENCES.items():
        lines.append(f"- [{gap:g} mm 双摄参考]({(reference / 'forces.json').as_posix()})")
    (output / "report_zh.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(table[["gap_mm", "camera", "direct_Fz_uN", "total_error_percent", "absolute_distance_change_error_uN"]].to_string(index=False))
    print(json.dumps({key:summary[key] for key in ("max_distance_change_error_uN", "max_direct_quadrature_change_uN", "numerical_distance_change_comparison_passed")}))


if __name__ == "__main__":
    main()
