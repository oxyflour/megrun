"""Run independent cold Maxwell mesh strategies sequentially on fixed hardware."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
AEDT = Path(r"C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe")
CASES = [("Adaptive", "adaptive"), ("Geometry", "geometry"), ("LowFidelity", "low-fidelity"),
         ("Aggressive", "low-fidelity-aggressive"), ("Local", "low-fidelity-local")]


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run(command, log):
    start = perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    return {"command": command, "returncode": result.returncode, "seconds": perf_counter() - start}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="outputs/mesh_seeding_2mm")
    parser.add_argument("--distance-mm", type=float, default=2)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--cases", nargs="+", choices=[n for n, _ in CASES], default=[n for n, _ in CASES[:3]])
    parser.add_argument("--max-passes", type=int, default=30)
    parser.add_argument("--minimum-converged-passes", type=int, default=2)
    parser.add_argument("--minimum-passes", type=int, default=2)
    parser.add_argument("--initial-mesh-limit", type=int, default=600000)
    parser.add_argument("--generate-initial-mesh", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.minimum_passes <= args.max_passes or args.minimum_converged_passes < 1:
        parser.error("Require 1 <= minimum-passes <= max-passes and minimum-converged-passes >= 1")
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "run_record.json"
    record = json.loads(record_path.read_text()) if record_path.exists() else {
        "started_UTC": datetime.now(timezone.utc).isoformat(), "distance_mm": args.distance_mm,
        "cores": args.cores, "maxwell_version": "2022.2", "force_delta_N": 2e-5,
        "energy_error_percent": .5, "minimum_passes": args.minimum_passes,
        "initial_mesh_limit": args.initial_mesh_limit,
        "minimum_converged_passes": args.minimum_converged_passes, "maximum_passes": args.max_passes,
        "generate_initial_mesh": args.generate_initial_mesh,
        "cases": {}}
    if record["cores"] != args.cores or record["distance_mm"] != args.distance_mm:
        raise ValueError("Existing experiment uses different hardware or gap")
    for key, value, old_default in [("minimum_converged_passes", args.minimum_converged_passes, 2),
                                    ("minimum_passes", args.minimum_passes, 2),
                                    ("initial_mesh_limit", args.initial_mesh_limit, 600000),
                                    ("maximum_passes", args.max_passes, 30),
                                    ("generate_initial_mesh", args.generate_initial_mesh, False)]:
        if record.get(key, old_default) != value:
            raise ValueError(f"Existing experiment uses different {key}")
    # Build everything before solving, so no second desktop competes with solver.
    for name, policy in CASES:
        if name not in args.cases:
            continue
        project = output / (name + ".aedt")
        item = record["cases"].setdefault(name, {"policy": policy, "project": str(project)})
        if "build" not in item:
            if project.exists():
                raise FileExistsError(f"Unrecorded project: {project}")
            record["active"] = f"Building {name}"
            write_json(record_path, record)
            command = [sys.executable, str(ROOT / "scripts/run_maxwell_case.py"), "--phase", "build",
                "--project", str(project), "--design", name, "--distance-mm", str(args.distance_mm),
                "--mesh-policy", policy, "--percent-error", ".5", "--max-passes", str(args.max_passes),
                "--minimum-passes", str(args.minimum_passes), "--minimum-converged-passes", str(args.minimum_converged_passes),
                "--force-delta-n", "2e-5"]
            if args.generate_initial_mesh:
                command += ["--generate-initial-mesh", "--initial-mesh-limit", str(args.initial_mesh_limit)]
            item["build"] = run(command, output / (name + "_build.log"))
            write_json(record_path, record)
            if item["build"]["returncode"]:
                raise RuntimeError(f"Build failed: {name}")
            print(f"Built {name} in {item['build']['seconds']:.1f}s", flush=True)
        if item["build"]["returncode"]:
            raise RuntimeError(f"Recorded build failed: {name}; inspect the log and use a fresh output")
    for name, policy in CASES:
        if name not in args.cases:
            continue
        item = record["cases"][name]
        if "extract" in item and item["extract"]["returncode"] == 0:
            print(f"Already completed {name}", flush=True)
            continue
        if "solve" not in item:
            record["active"] = f"Solving {name}"
            record["active_started_UTC"] = datetime.now(timezone.utc).isoformat()
            write_json(record_path, record)
            command = [str(AEDT), "-ng", "-BatchSolve", f"{name}:Nominal:Reference", "-machinelist",
                       f"list=localhost:1:{args.cores}:60%:0", "-Monitor", item["project"]]
            item["solve"] = run(command, output / (name + "_solve.log"))
            write_json(record_path, record)
            if item["solve"]["returncode"]:
                raise RuntimeError(f"Solve failed: {name}")
            print(f"Solved {name} in {item['solve']['seconds']:.1f}s", flush=True)
        if item["solve"]["returncode"]:
            raise RuntimeError(f"Recorded solve failed: {name}; inspect the log")
        record["active"] = f"Extracting {name}"
        write_json(record_path, record)
        command = [sys.executable, str(ROOT / "scripts/run_maxwell_case.py"), "--phase", "extract",
                   "--project", item["project"], "--design", name, "--distance-mm", str(args.distance_mm)]
        item["extract"] = run(command, output / (name + "_extract.log"))
        item["end_to_end_seconds"] = sum(item[k]["seconds"] for k in ("build", "solve", "extract"))
        write_json(record_path, record)
        if item["extract"]["returncode"]:
            raise RuntimeError(f"Extract failed: {name}")
        print(f"Extracted {name}, end to end {item['end_to_end_seconds']:.1f}s", flush=True)
    record["active"] = "Complete"
    record["completed_UTC"] = datetime.now(timezone.utc).isoformat()
    write_json(record_path, record)


if __name__ == "__main__":
    main()
