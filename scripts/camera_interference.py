"""Multi-camera JSON workflow: prepare / propose / build / run."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_mag.array_workflow import propose_prepared, run_workflow


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["prepare", "propose", "build", "run"])
    parser.add_argument("--config", help="Input JSON; required except for propose")
    parser.add_argument("--output", required=True, help="Fresh output directory; propose uses a prepared directory")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--max-iterations", type=int, default=2, help="Maximum moves; native solves = moves+1")
    parser.add_argument("--surrogate-only", action="store_true", help="Preview loop with no native build or solve")
    parser.add_argument("--budget-minutes", type=float, default=5, help="Whole campaign wall-clock limit, including cleanup")
    parser.add_argument("--estimated-total-minutes", type=float, help="Required for native run, all serial work included")
    parser.add_argument("--estimate-basis", help="Measured basis for complete native campaign estimate")
    parser.add_argument("--confirm-long-simulation", action="store_true", help="Use only after explicit user confirmation for this campaign")
    parser.add_argument("--proposal", help="JSON candidate_id/reason supplied by gpt-6-luna; otherwise call Codex CLI")
    args = parser.parse_args()
    if args.command != "propose" and not args.config:
        parser.error("--config is required")
    if args.command == "propose":
        result = propose_prepared(args.output, proposal=args.proposal, budget_seconds=args.budget_minutes * 60)
    else:
        result = run_workflow(args.config, args.output, mode=args.command, surrogate_only=args.surrogate_only,
            max_iterations=args.max_iterations, budget_minutes=args.budget_minutes,
            estimated_minutes=args.estimated_total_minutes, estimate_basis=args.estimate_basis,
            confirmed_long=args.confirm_long_simulation, cache_dir=args.cache_dir, proposal=args.proposal)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["status"] in ("stopped", "native_unconverged_stop") else 0


if __name__ == "__main__":
    raise SystemExit(main())
