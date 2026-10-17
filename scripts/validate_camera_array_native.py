"""Compare two frozen camera layouts under one confirmed native campaign budget."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_mag.array_config import ROOT, load_config, write_json
from camera_mag.array_maxwell import digest, native_iteration, simulation_gate
from camera_mag.array_workflow import analyze_layout, make_solver


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budget-minutes", type=float, required=True)
    parser.add_argument("--estimated-total-minutes", type=float, required=True)
    parser.add_argument("--estimate-basis", required=True)
    parser.add_argument("--confirm-long-simulation", action="store_true")
    args = parser.parse_args()
    gate = simulation_gate(estimated_minutes=args.estimated_total_minutes, estimate_basis=args.estimate_basis,
                           budget_minutes=args.budget_minutes, confirmed_long=args.confirm_long_simulation)
    configs = {"baseline": load_config(args.baseline_config), "candidate": load_config(args.candidate_config)}
    baseline, candidate = configs.values()
    if baseline["monitors"] != candidate["monitors"] or baseline["maxwell"] != candidate["maxwell"]:
        raise ValueError("Comparison requires identical monitors and Maxwell settings")
    if "aedt" in baseline or "aedt" in candidate:
        raise ValueError("Extract one frozen aedt_source bundle before comparing layouts")
    source_key = "aedt_source" if "aedt_source" in baseline else "step"
    if (baseline["N"] != candidate["N"] or source_key not in candidate or
            digest(baseline[source_key]) != digest(candidate[source_key])):
        raise ValueError("Comparison requires the same camera count and physics source")
    for left, right in zip(baseline["cameras"], candidate["cameras"], strict=True):
        for key in ("id", "current_A", "turns"):
            if left[key] != right[key]:
                raise ValueError("Comparison requires identical camera IDs and excitations")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    start = perf_counter()
    deadline = start + args.budget_minutes * 60
    record = {"started_UTC": datetime.now(timezone.utc).isoformat(), "budget": gate, "cases": {},
              "scope": "Exactly two frozen three-camera B layouts; no position search or extra solve",
              "native_solve_limit": 2, "status": "preparing"}
    path = output / "comparison_record.json"
    try:
        sources = list((ROOT / "camera_mag").glob("*.py")) + [ROOT / "scripts/run_camera_array_native.py", Path(__file__).resolve()]
        hashes = {}
        for source in sources:
            target = output / "source_snapshot" / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[str(source.relative_to(ROOT))] = digest(target)
        write_json(output / "source_snapshot/sha256.json", hashes)
        solver = make_solver(baseline, ".cache")
        for name, config in configs.items():
            folder = output / name
            record.update(active_case=name, status="analyzing_" + name)
            write_json(path, record)
            lf, _ = analyze_layout(solver, config, folder, deadline)
            record["cases"][name] = {"directory": str(folder), "config_sha256": digest(folder / "config.json"),
                                     "low_fidelity_score": lf["score"]}
            record["status"] = "native_" + name
            write_json(path, record)
            native = native_iteration(folder / "config.json", folder, deadline)
            record["cases"][name]["native_result"] = native
            write_json(path, record)
            if not native["native_adaptive_converged"]:
                record["status"] = "native_unconverged_stop"
                return 2
        first = record["cases"]["baseline"]["native_result"]
        last = record["cases"]["candidate"]["native_result"]
        reduction = 1 - last["score"] / first["score"] if first["score"] else None
        report = {"both_native_adaptive_converged": True, "baseline_score": first["score"], "candidate_score": last["score"],
                  "native_relative_score_reduction": reduction, "native_improvement": last["score"] < first["score"],
                  "low_fidelity_relative_score_reduction": 1 - record["cases"]["candidate"]["low_fidelity_score"] / record["cases"]["baseline"]["low_fidelity_score"],
                  "monitor_comparison": [{"id": a["id"], "unit": a["unit"], "baseline": a["value"],
                                          "candidate": b["value"], "change": b["value"] - a["value"]}
                                         for a, b in zip(first["monitors"], last["monitors"], strict=True)],
                  "precision_note": "Native adaptive convergence is not a strict monitor error bound; no F-difference precision claim"}
        write_json(output / "comparison_summary.json", report)
        record["status"] = "compared"
        return 0
    except BaseException as exc:
        record.update(status="stopped", failure=repr(exc))
        raise
    finally:
        record.update(ended_UTC=datetime.now(timezone.utc).isoformat(), elapsed_seconds=perf_counter() - start,
                      native_jobs_submitted=len(list(output.rglob("native_solve.log.json"))))
        write_json(path, record)


if __name__ == "__main__":
    raise SystemExit(main())
