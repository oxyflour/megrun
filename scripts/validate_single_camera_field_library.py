"""Sequential native isolated solve, field export, and independent pair-force audits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
AEDT = r"C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe"


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", default="outputs/single_camera_reuse")
    p.add_argument("--source-directory", default="refined_single")
    p.add_argument("--record-name", default="refined_validation_record.json")
    p.add_argument("--frozen-pass", type=int, help="Postprocess an aborted source after auditing this retained completed pass")
    args = p.parse_args()
    output = Path(args.output).resolve()
    project = output / args.source_directory / "SingleFieldReference.aedt"
    source = project.parent / "SingleFieldReference"
    library = output / "refined_library"
    result = output / "refined_prediction"
    record_path = output / args.record_name
    if not (project.parent / "build_record.json").exists():
        raise ValueError("Isolated field source build must be complete")
    record = json.loads(record_path.read_text()) if record_path.exists() else dict(stages=[],
        purpose="Single-camera-only offline data, paired force references used only for evaluation",
        build_record=str(project.parent / "build_record.json"))

    def stage(name, command):
        previous = next((item for item in record["stages"] if item["name"]==name), None)
        if previous:
            if previous["command"] != command:
                raise ValueError(f"Resume command changed: {name}")
            if previous.get("returncode")==0:
                print(f"COMPLETED ALREADY: {name}", flush=True)
                return
            record["stages"].remove(previous)
        item = dict(name=name, command=command, started_UTC=datetime.now(timezone.utc).isoformat())
        record["stages"].append(item)
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"START {name}", flush=True)
        start = perf_counter()
        log_prefix = "reused" if args.source_directory=="reused_single" else "refined"
        with (output / (log_prefix+"_"+name+".log")).open("w") as stream:
            proc = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        item.update(returncode=proc.returncode, seconds=perf_counter()-start,
                    completed_UTC=datetime.now(timezone.utc).isoformat())
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"FINISH {name}: return={proc.returncode}, seconds={item['seconds']:.2f}", flush=True)
        if proc.returncode:
            raise RuntimeError(f"Stage {name} failed; see its log")

    if args.frozen_pass is None:
        stage("native_solve", [AEDT, "-ng", "-BatchSolve", "SingleFieldReference:Nominal:Reference",
            "-machinelist", "list=localhost:1:4:60%:0", "-Monitor", str(project)])
    else:
        record["source_acceptance"] = dict(selected_completed_pass=args.frozen_pass,
            native_adaptive_converged=False, use="Experimental distance differences; fixed baseline cancels")
    stage("native_extract", [sys.executable, str(ROOT / "scripts/run_maxwell_case.py"), "--phase", "extract",
        "--project", str(project), "--design", "SingleFieldReference", "--isolated"])
    export_command = [sys.executable, str(ROOT / "scripts/export_single_camera_library.py"),
        "--source", str(source), "--output", str(library)]
    if args.frozen_pass is not None:
        stage("frozen_source_audit", [sys.executable, str(ROOT / "scripts/audit_frozen_single_camera.py"),
            "--source", str(source), "--pass", str(args.frozen_pass)])
        export_command += ["--frozen-pass", str(args.frozen_pass)]
    stage("library_export", export_command)
    stage("direct_evaluate", [sys.executable, str(ROOT / "scripts/evaluate_single_camera_prediction.py"),
        "--library", str(library), "--output", str(result)])
    stage("incremental_evaluate", [sys.executable, str(ROOT / "scripts/evaluate_incremental_camera_response.py"),
        "--library", str(library), "--output", str(result / "incremental"),
        "--direct-summary", str(result / "direct_prediction_summary.json")])
    stage("new_gap_predict", [sys.executable, str(ROOT / "scripts/predict_camera_pair_from_library.py"),
        "--library", str(library), "--output", str(output / "new_gap_predictions.json")])
    record["completed_UTC"] = datetime.now(timezone.utc).isoformat()
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
