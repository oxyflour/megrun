"""Validate frozen one-pass seeds at 1 / 3 mm against tighter native Maxwell."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
AEDT = Path(r"C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe")
SOURCES = {
    1: (ROOT / "outputs/maxwell_verified/gap_1mm_reference.aedt", "Gap_1mm_CCW"),
    3: (ROOT / "outputs/maxwell_verified/sweep_reference.aedt", "Gap_3mm_CCW"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, record):
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def run_stage(record, key, command, log, record_path):
    if key in record:
        if record[key]["returncode"]:
            raise RuntimeError(f"Previously failed {key}; inspect {log}")
        return
    record["active"] = key
    save(record_path, record)
    start = perf_counter()
    print(f"Starting {key}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record[key] = {"command": command, "seconds": perf_counter() - start,
                   "returncode": result.returncode}
    save(record_path, record)
    if result.returncode:
        raise RuntimeError(f"Failed {key}; inspect {log}")
    print(f"Finished {key} in {record[key]['seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--development", required=True)
    parser.add_argument("--gaps-mm", nargs="+", type=int, choices=sorted(SOURCES), default=[1, 3])
    args = parser.parse_args()
    output, development = (ROOT / args.output).resolve(), (ROOT / args.development).resolve()
    audit = read(development / "fixed_mesh_audit.json")
    if not audit["force_accuracy_accepted"] or audit["distance_mm"] != 2:
        raise ValueError("Requires accepted 2 mm development result before hold-outs")
    case = audit["case"]
    if case not in ("Local", "Aggressive"):
        raise ValueError("Unsupported development mesh policy")
    hashes = read(development / "source_snapshot/sha256.json")
    # Analysis tooling can evolve, but all geometry, material and LF policy
    # files must be exactly the implementation used for development.
    frozen = {name: digest for name, digest in hashes.items()
              if name.startswith("camera_mag") or Path(name).name in ("run_maxwell_case.py", "benchmark_mesh_seeding.py")}
    for name, digest in frozen.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen implementation changed: {name}")
    output.mkdir(parents=True, exist_ok=True)
    record_path = output / "holdout_record.json"
    record = read(record_path) if record_path.exists() else {
        "started_UTC": datetime.now(timezone.utc).isoformat(), "development": str(development),
        "frozen_physics_hashes": frozen, "gaps_mm": args.gaps_mm, "cores": 4, "case": case,
        "reference_cost_excluded_from_candidate_timing": True,
        "note": "No tuning using held-out Maxwell force outputs; native reference uses its own same-gap prior mesh",
    }
    if record["development"] != str(development) or record["frozen_physics_hashes"] != frozen or record["gaps_mm"] != args.gaps_mm or record["case"] != case:
        raise ValueError("Existing hold-out configuration differs")
    save(record_path, record)
    for gap in args.gaps_mm:
        folder = output / f"gap_{gap}mm"
        folder.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts/benchmark_mesh_seeding.py"),
            "--output", str(folder), "--distance-mm", str(gap), "--cases", case,
            "--max-passes", "1", "--minimum-passes", "1", "--minimum-converged-passes", "1",
            "--generate-initial-mesh", "--initial-mesh-limit", "650000"]
        run_stage(record, f"candidate_{gap}mm", command, folder / "candidate_pipeline.log", record_path)
    for gap in args.gaps_mm:
        folder = output / f"gap_{gap}mm"
        source, design = SOURCES[gap]
        target = folder / "TightReference.aedt"
        run_stage(record, f"reference_build_{gap}mm", [sys.executable,
            str(ROOT / "scripts/build_tight_mesh_reference.py"), str(source), "--design", design,
            "--destination", str(target)], folder / "reference_build.log", record_path)
        run_stage(record, f"reference_solve_{gap}mm", [str(AEDT), "-ng", "-BatchSolve",
            "TightReference:Nominal:Reference", "-machinelist", "list=localhost:1:4:60%:0",
            "-Monitor", str(target)], folder / "reference_solve.log", record_path)
        run_stage(record, f"reference_extract_{gap}mm", [sys.executable,
            str(ROOT / "scripts/run_maxwell_case.py"), "--phase", "extract", "--project", str(target),
            "--design", "TightReference", "--distance-mm", str(gap)], folder / "reference_extract.log", record_path)
        run_stage(record, f"audit_{gap}mm", [sys.executable, str(ROOT / "scripts/audit_fixed_mesh.py"),
            "--output", str(folder), "--case", case, "--reference", str(folder / "TightReference")],
            folder / "audit.log", record_path)
    record.pop("active", None)
    record["completed_UTC"] = datetime.now(timezone.utc).isoformat()
    record["all_force_accuracy_accepted"] = all(read(output / f"gap_{g}mm/fixed_mesh_audit.json")["force_accuracy_accepted"] for g in args.gaps_mm)
    save(record_path, record)
    print(f"Held-out validation complete; accepted = {record['all_force_accuracy_accepted']}", flush=True)


if __name__ == "__main__":
    main()
