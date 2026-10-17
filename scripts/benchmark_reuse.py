from pathlib import Path
import json
import numpy as np
import pandas as pd
from _common import parser, make_solver
from camera_mag.output import save_result, plt


def main():
    p = parser("Sweep camera gap: cold vs warm, reusable self operators, coil force")
    p.add_argument("--distances-mm", type=float, nargs="+", default=[1, 1.5, 2, 2.5, 3])
    p.add_argument("--output", default="outputs/sweep")
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    solver = make_solver(args)
    isolated = solver.isolated_states()
    rows, previous = [], None
    all_ok = True
    for index, distance in enumerate(args.distances_mm):
        cold = solver.solve(distance)
        warm = solver.solve(distance, previous)
        comparison = np.linalg.norm(np.concatenate(cold.state) - np.concatenate(warm.state)) / max(np.linalg.norm(np.concatenate(cold.state)), 1)
        summary = save_result(solver, warm, out / f"case_{index:02d}_gap_{distance:g}mm", isolated)
        ok = bool(cold.converged and warm.converged and comparison < max(50 * args.tol, 1e-5))
        row = {"distance_mm": distance, "center_distance_mm": warm.offset_b[0] * 1e3,
               "cold_iterations": cold.iterations, "warm_iterations": warm.iterations,
               "cold_solve_seconds": cold.runtime["solve_seconds"], "warm_solve_seconds": warm.runtime["solve_seconds"],
               "cold_cross_seconds": cold.runtime["cross_build_seconds"], "warm_cross_seconds": warm.runtime["cross_build_seconds"],
               "cold_warm_relative_difference": float(comparison), "cold_converged": cold.converged,
               "warm_converged": warm.converged, "verified": ok,
               "relative_M_A_change": summary["relative_M_change"]["A"], "relative_M_B_change": summary["relative_M_change"]["B"]}
        for module, forces in summary["coil_force"].items():
            for quantity in ["total_N", "interaction_delta_N", "isolated_N", "direct_other_module_N"]:
                for axis, value in zip("xyz", forces[quantity]):
                    row[f"coil_{module}_{quantity[:-2]}_{axis}_N"] = value
        rows.append(row)
        previous = warm.state if warm.converged else None
        all_ok = all_ok and ok
        print(f"gap={distance:g} mm: cold={cold.iterations} ({row['cold_solve_seconds']:.3f}s), "
              f"warm={warm.iterations} ({row['warm_solve_seconds']:.3f}s), state difference={comparison:.2g}, verified={ok}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "reuse_benchmark.csv", index=False)
    force_cols = [c for c in table if c.startswith("coil_")]
    table[["distance_mm", "center_distance_mm", *force_cols]].to_csv(out / "coil_force_sweep.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, quantity, title in zip(axes, ["total", "interaction_delta"], ["Total coil force", "Change from isolated module"]):
        for module, style in [("A", "-"), ("B", "--")]:
            for axis, color in zip("xyz", ["tab:red", "tab:green", "tab:blue"]):
                ax.plot(table.distance_mm, table[f"coil_{module}_{quantity}_{axis}_N"] * 1e3,
                        style, color=color, marker="o", label=f"{module}: F{axis}")
        ax.set(title=title, xlabel="Outer-envelope gap [mm]", ylabel="Force [mN]")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"LOW FIDELITY: {args.current_a:g} A, {args.turns} turns, {args.cell_mm:g} mm cells")
    fig.savefig(out / "coil_force_sweep.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode in ["cold", "warm"]:
        axes[0].plot(table.distance_mm, table[f"{mode}_iterations"], "-o", label=mode)
        axes[1].plot(table.distance_mm, table[f"{mode}_solve_seconds"], "-o", label=mode)
    axes[0].set(xlabel="Gap [mm]", ylabel="Iterations")
    axes[1].set(xlabel="Gap [mm]", ylabel="Solve time [s]")
    for ax in axes:
        ax.grid(alpha=.3)
        ax.legend()
    fig.savefig(out / "reuse_benchmark.png", dpi=160)
    plt.close(fig)
    (out / "benchmark_summary.json").write_text(json.dumps({"all_verified": bool(all_ok),
              "self_precompute_calls": solver.self_build_calls, "self_cache": solver.self_cache,
              "isolated": solver.isolated_runtime, "rows": rows}, indent=2), encoding="utf-8")
    print(table[["distance_mm", "cold_iterations", "warm_iterations", "relative_M_A_change", "relative_M_B_change"]].to_string(index=False))
    print(f"Saved {out}; self operators were precomputed {solver.self_build_calls} time(s).")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
