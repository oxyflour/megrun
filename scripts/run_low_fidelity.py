from _common import parser, make_solver
from camera_mag.output import save_result


def main():
    p = parser("Coarse camera-pair fields and coil Lorentz-force diagnostics")
    p.add_argument("--distance-mm", type=float, required=True, help="Outer-envelope gap along +X")
    p.add_argument("--output", default="outputs/single")
    args = p.parse_args()
    solver = make_solver(args)
    result = solver.solve(args.distance_mm)
    summary = save_result(solver, result, args.output)
    print(f"converged={result.converged}, iterations={result.iterations}, residual={result.residual_history[-1]:.3g}")
    for module, force in summary["coil_force"].items():
        print(f"Coil {module}: total {force['total_N']} N; interaction delta {force['interaction_delta_N']} N")
    print(f"Saved to {args.output}; low-fidelity diagnostic, not Maxwell ground truth.")
    return 0 if result.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
