"""Audit completed mesh benchmarks and write measured comparisons, without fitting."""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def seconds(value):
    h, m, s = map(int, value.split(":"))
    return 3600 * h + 60 * m + s


def read_case(root, name, timing):
    folder = root / name
    profile = (folder / "profile.txt").read_text()
    if "Status: Normal Completion" not in profile:
        raise ValueError(f"{name}: Maxwell did not complete normally")
    if "Adaptive Passes converged" not in profile:
        raise ValueError(f"{name}: Maxwell adaptive passes did not converge")
    provenance = json.loads((folder / "provenance.json").read_text())
    for key in ("SI_positions_checked", "current_polarity_and_ampere_turns_checked", "BH_field_median_error_below_5_percent"):
        if not provenance.get(key):
            raise ValueError(f"{name}: failed {key}")
    setup = provenance["setup"]
    convergence = (folder / "convergence.txt").read_text()
    match = re.search(r"Current\s*:\s*\(([^,]+),\s*([^)]+)\)", convergence)
    if not match or max(map(float, match.groups())) > setup["PercentError"]:
        raise ValueError(f"{name}: energy error target not achieved")
    force_data = json.loads((folder / "forces.json").read_text())
    vectors = {}
    for label in "AB":
        vectors[label] = []
        for axis in "xyz":
            matches = [v for k, v in force_data.items() if k == f"Force_{label}.Force_{axis}"]
            if len(matches) != 1 or matches[0]["unit"] not in ("N", "newton") or len(matches[0]["values"]) != 1:
                raise ValueError(f"{name}: invalid force units or data")
            vectors[label].append(matches[0]["values"][0])
    history = json.loads((folder / "force_adaptive_history.json").read_text())
    forces = np.array([history["forces"][f"Force_{label}.Force_z"] for label in "AB"])
    delta = np.max(np.abs(np.diff(forces, axis=1)), axis=0)
    manifest = json.loads((folder / "manifest.json").read_text())
    force_target = manifest["force_delta_n"]
    cache = setup.get("ExpressionCache", {}).get("CacheItem", [])
    if isinstance(cache, dict):
        cache = [cache]
    for label in "AB":
        items = [c for c in cache if c.get("Expression") == f"Force_{label}.Force_z"]
        if (len(items) != 1 or not items[0].get("IsConvergence") or items[0].get("UseRelativeConvergence") != 0
                or not np.isclose(items[0].get("MaxConvergenceDelta", 0), force_target, rtol=1e-9, atol=0)):
            raise ValueError(f"{name}: native force criterion differs from declared target")
    required_deltas = int(setup["MinimumConvergedPasses"])
    if required_deltas < 1 or len(delta) < required_deltas or np.max(delta[-required_deltas:]) > force_target * (1 + 1e-6):
        raise ValueError(f"{name}: last {required_deltas} force deltas exceed {force_target} N")
    # Confirm the solver had this criterion enabled, rather than just checking
    # forces afterwards. The saved native file is the actual setup input.
    project_text = Path(timing["project"]).read_text(errors="replace")
    if project_text.count("IsConvergence=true") < 2 or "UseRelativeConvergence=0" not in project_text:
        raise ValueError(f"{name}: force expression convergence missing from native setup")
    plan = json.loads((folder / "mesh_plan.json").read_text())
    meshes = [int(n) for n in re.findall(r"Tetrahedra:\s*(\d+)", profile)]
    mesh_stats = (folder / "mesh_stats.txt").read_text()
    final_mesh = int(re.search(r"Total number of mesh elements:\s*(\d+)", mesh_stats)[1])
    object_tets = {n.strip(): int(k) for n, k in re.findall(r"^\s*([^|\n]+)\|\s*(\d+)\s*\|", mesh_stats, re.M)}
    volume = re.search(r"Mesh \(volume, seed\).*Tetrahedra:\s*(\d+)", profile)
    initial_mesh = int(volume[1]) if volume else meshes[0]
    first_pass = re.search(r"Pass 1\s.*?\badapt\s+[^\n]*Tetrahedra:\s*(\d+)", profile, flags=re.S)
    if first_pass:
        initial_mesh = int(first_pass[1])
    row = {"case": name, "policy": plan["policy"], "gap_mm": manifest["distance_mm"],
           "passes": len(history["passes"]), "initial_tetrahedra": initial_mesh,
           "final_tetrahedra": final_mesh, "energy_error_percent": float(match[1]),
           "air_tetrahedra": object_tets.get("Region", 0),
           "coil_tetrahedra": sum(v for k, v in object_tets.items() if "_AF_COIL_" in k),
           "yoke_tetrahedra": sum(v for k, v in object_tets.items() if "_YOKE_" in k),
           "magnet_tetrahedra": sum(v for k, v in object_tets.items() if "_MAGNET_" in k),
           "energy_delta_percent": float(match[2]), "last_two_max_Fz_delta_uN": float(np.max(delta[-2:]) * 1e6),
           "required_consecutive_force_deltas": required_deltas,
           "last_required_max_Fz_delta_uN": float(np.max(delta[-required_deltas:]) * 1e6),
           "A_Fz_mN": vectors["A"][2] * 1e3, "B_Fz_mN": vectors["B"][2] * 1e3,
           "mean_Fz_mN": (vectors["A"][2] + vectors["B"][2]) * 500,
           "AB_Fz_difference_uN": abs(vectors["A"][2] - vectors["B"][2]) * 1e6,
           "build_seconds": timing["build"]["seconds"], "solve_seconds": timing["solve"]["seconds"],
           "extract_seconds": timing["extract"]["seconds"], "end_to_end_seconds": timing["end_to_end_seconds"],
           "policy_preprocessing_seconds": plan["preprocessing_seconds"]}
    return row, history, vectors, provenance


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", default="outputs/mesh_seeding_2mm")
    p.add_argument("--reference", help="Directory of a tighter solved common reference")
    args = p.parse_args()
    root = Path(args.output).resolve()
    record = json.loads((root / "run_record.json").read_text())
    rows, histories, provenance_list = [], {}, []
    for name, timing in record["cases"].items():
        if "extract" not in timing or timing["extract"]["returncode"]:
            raise ValueError(f"Incomplete {name}")
        row, history, vectors, provenance = read_case(root, name, timing)
        rows.append(row)
        histories[name] = history
        provenance_list.append(provenance)
    if len({p["step_sha256"] for p in provenance_list}) != 1:
        raise ValueError("Different STEP geometry")
    frame = pd.DataFrame(rows).set_index("case", drop=False)
    if "Adaptive" in frame.index:
        frame["speedup_vs_adaptive"] = frame.loc["Adaptive", "end_to_end_seconds"] / frame.end_to_end_seconds
    if "Geometry" in frame.index:
        frame["speedup_vs_geometry"] = frame.loc["Geometry", "end_to_end_seconds"] / frame.end_to_end_seconds
    summary = {"measured": True, "cores": record["cores"], "energy_target_percent": record["energy_error_percent"],
               "Fz_delta_target_uN": record["force_delta_N"] * 1e6,
               "force_spread_percent": 100 * (frame.mean_Fz_mN.max() - frame.mean_Fz_mN.min()) / frame.mean_Fz_mN.mean(),
               "reference_validated": False, "rows": frame.to_dict("records")}
    if args.reference:
        folder = Path(args.reference)
        profile = (folder / "profile.txt").read_text()
        if "Status: Normal Completion" not in profile or "Adaptive Passes converged" not in profile:
            raise ValueError("Reference did not converge")
        ref_provenance = json.loads((folder / "provenance.json").read_text())
        if ref_provenance["step_sha256"] != provenance_list[0]["step_sha256"]:
            raise ValueError("Reference STEP differs")
        convergence = (folder / "convergence.txt").read_text()
        current = re.search(r"Current\s*:\s*\(([^,]+),\s*([^)]+)\)", convergence)
        ref_energy_target = ref_provenance["setup"]["PercentError"]
        if not current or ref_energy_target > .25 or max(map(float, current.groups())) > ref_energy_target:
            raise ValueError("Reference energy error exceeds tighter criterion")
        ref_history = json.loads((folder / "force_adaptive_history.json").read_text())
        ref_pass_forces = np.array([ref_history["forces"][f"Force_{label}.Force_z"] for label in "AB"])
        if ref_pass_forces.shape[1] < 3 or np.max(np.abs(np.diff(ref_pass_forces, axis=1)[:, -2:])) > 1.000001e-5:
            raise ValueError("Reference force is not converged to 10 µN for two consecutive deltas")
        force = json.loads((folder / "forces.json").read_text())
        if any(force[f"Force_{label}.Force_z"]["unit"] not in ("N", "newton") for label in "AB"):
            raise ValueError("Reference force units differ")
        ref_vector = np.array([force[f"Force_{label}.Force_z"]["values"][0] for label in "AB"]) * 1e3
        ref_mean = float(ref_vector.mean())
        frame["mean_Fz_error_percent"] = (frame.mean_Fz_mN / ref_mean - 1) * 100
        frame["max_coil_Fz_error_percent"] = np.max(np.abs(frame[["A_Fz_mN", "B_Fz_mN"]].to_numpy() / ref_vector - 1), axis=1) * 100
        frame["within_02_percent_reference"] = frame.max_coil_Fz_error_percent <= .2
        summary.update(reference_validated=True, reference_mean_Fz_mN=ref_mean,
                       reference_directory=str(folder.resolve()), reference_project=ref_provenance["project"],
                       reference_energy_target_percent=ref_energy_target, rows=frame.to_dict("records"))
    frame.to_csv(root / "comparison.csv", index=False)
    (root / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    colors = ["#7b8794", "#3176a8", "#209c72", "#a06fba"][:len(frame)]
    x = np.arange(len(frame))
    axes[0].bar(x, frame.end_to_end_seconds / 60, color=colors)
    axes[0].set(xticks=x, xticklabels=frame.case, ylabel="End-to-end time [min]", title="Measured, sequential, 4 cores")
    for i, row in enumerate(frame.to_dict("records")):
        axes[0].text(i, row["end_to_end_seconds"] / 60, f"{row['passes']} passes", ha="center", va="bottom")
    axes[1].bar(x - .18, frame.initial_tetrahedra / 1e3, width=.36, label="Initial")
    axes[1].bar(x + .18, frame.final_tetrahedra / 1e3, width=.36, label="Final")
    axes[1].set(xticks=x, xticklabels=frame.case, ylabel="Tetrahedra [thousands]", title="Mesh cost")
    axes[1].legend()
    for (name, history), color in zip(histories.items(), colors):
        vectors = np.array([history["forces"][f"Force_{label}.Force_z"] for label in "AB"])
        delta = np.max(np.abs(np.diff(vectors, axis=1)), axis=0) * 1e6
        axes[2].semilogy(np.asarray(history["passes"], float)[1:], np.maximum(delta, .001), ".-", color=color, label=name)
    axes[2].axhline(record["force_delta_N"] * 1e6, color="#bf4b45", linestyle="--", label="20 uN target")
    axes[2].set(xlabel="Adaptive pass", ylabel="Max per-coil Fz pass change [uN]", title="Force stopping criterion")
    axes[2].legend()
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    report = ["# Maxwell mesh seeding: measured pilot", "",
        "Fixed STEP geometry, two identical cameras, +X gap %.3g mm; 0.1 A, 100 turns. " % record["distance_mm"]
        + "Maxwell 2022 R2, four CPU cores, sequential independent solves. "
        + "Materials, current direction and sampling units were checked against the solved fields.", "",
        "All strategies use 0.5% energy error and 20 µN per-pass Fz delta for each coil. "
        + f"The last {record['minimum_converged_passes']} force deltas are checked independently after solving. "
        + "The comparison measures the coil's main-axis force; it does not establish relative accuracy of tiny lateral forces.", "",
        "| Strategy | Total min | Solve min | Passes | Initial tets | Final tets | Mean Fz mN | Speedup vs adaptive | Speedup vs geometry |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in frame.to_dict("records"):
        report.append(f"| {row['case']} | {row['end_to_end_seconds']/60:.2f} | {row['solve_seconds']/60:.2f} "
            f"| {row['passes']} | {row['initial_tetrahedra']:,} | {row['final_tetrahedra']:,} "
            f"| {row['mean_Fz_mN']:.6f} | {row.get('speedup_vs_adaptive', float('nan')):.3f} "
            f"| {row.get('speedup_vs_geometry', float('nan')):.3f} |")
    report += ["", "End-to-end time includes project construction, solver launch/solution, and output extraction. "
        + "Reference computation is excluded equally. Detailed raw timings and commands are in `run_record.json`.", ""]
    if summary["reference_validated"]:
        report += [f"Common tighter reference: mean Fz = {summary['reference_mean_Fz_mN']:.6f} mN, "
            + f"energy target {ref_energy_target:g}%, absolute Fz pass delta 10 µN, with two successive deltas checked.", "",
            "| Strategy | Maximum per-coil Fz deviation from tighter reference | Within 0.2% target |",
            "|---|---:|---|"]
        for row in frame.to_dict("records"):
            report.append(f"| {row['case']} | {row['max_coil_Fz_error_percent']:.4f}% "
                          f"| {row['within_02_percent_reference']} |")
    else:
        report += ["Tighter reference has not been supplied. Force convergence has been checked, "
                   + "but absolute force accuracy has not yet been validated."]
    lf = frame.loc["LowFidelity"] if "LowFidelity" in frame.index else None
    if lf is not None:
        report += ["", f"Low-fidelity computation and face-operation mapping cost {lf.policy_preprocessing_seconds:.3f} s. "
            + "The source self operator was cached; this is a reuse measurement. The rules use no Maxwell solution labels."]
        report += ["", f"Air contains {lf.air_tetrahedra:,} of the {lf.final_tetrahedra:,} final tetrahedra "
            f"({100*lf.air_tetrahedra/lf.final_tetrahedra:.1f}%) in LowFidelity. These are not automatically redundant: "
            + "near-field air supports fringe fields and interface continuity. The current policy influences air indirectly through face seeds."]
        if summary["reference_validated"]:
            accepted = bool(lf.within_02_percent_reference)
            adaptive_success = accepted and lf.speedup_vs_adaptive >= 1.5
            geometry_success = accepted and lf.speedup_vs_geometry >= 1.5
            report += ["", f"Pilot 1.5x target vs Adaptive: {'met' if adaptive_success else 'not met'}. "
                f"Pilot 1.5x target vs Geometry: {'met' if geometry_success else 'not met'}."]
    report += ["", "One geometry, one distance and one run per strategy are a pilot. "
        + "They do not establish generalization across camera types or orientations, or a timing confidence interval. "
        + "Any later policy tuning is development, and must be evaluated on a separate case.", "",
        "![Measured comparison](comparison.png)", "",
        "Native projects: `Adaptive.aedt`, `Geometry.aedt`, `LowFidelity.aedt`, "
        + (f"`{Path(summary['reference_project']).name}`. " if summary["reference_validated"] else "reference pending. ")
        + "Per-case directories retain setup provenance, mesh operations, force histories, profiles, mesh statistics and field checks."]
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
