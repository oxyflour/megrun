"""Retain outcomes and resume completed stages after a metadata-parser correction.

This supervisor never retries a failed/aborted Maxwell solve. It only resumes
the active orchestration once when recorded native stages all succeeded.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from time import sleep

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.refine_pair_difference_reference import audit, digest, read, save, snapshot, SOURCES


def summarize(output):
    record = read(output / "run_record.json")
    result = dict(updated_UTC=datetime.now(timezone.utc).isoformat(), active=record["active"],
                  native_response_library_built=False, original_sources_unchanged={}, levels={},
                  stability_diagnostics_are_strict_error_bounds=False,
                  aborted_or_failed_stages=[key for key,item in record["stages"].items()
                                            if item["returncode"] or "abort_request" in item],
                  completed_stage_seconds=sum(item["seconds"] for item in record["stages"].values()))
    for gap, source in SOURCES.items():
        result["original_sources_unchanged"][str(gap)] = snapshot(source)==record["original_sources"][str(gap)]
    prediction = read(ROOT / "outputs/single_camera_reuse/refined_prediction/incremental_0.5/summary.json")
    prediction_by_label = {row["camera"]:row["predicted_change_from_2mm_uN"] for row in prediction["rows"]
                           if row["mode"]=="linear" and row["gap_mm"]==1}
    previous = {label:(read(SOURCES[1]/"forces.json")[f"Force_{label}.Force_z"]["values"][0]
                      -read(SOURCES[2]/"forces.json")[f"Force_{label}.Force_z"]["values"][0])*1e6 for label in "AB"}
    for level in (1,2):
        level_result = dict(gaps={})
        for gap in (1,2):
            folder = output / f"level_{level}" / f"gap_{gap}mm" / "PrecisionReference"
            if (folder/"forces.json").exists() and (folder/"provenance.json").exists():
                try:
                    level_result["gaps"][str(gap)] = audit(folder)
                except Exception as exc:
                    level_result["gaps"][str(gap)] = dict(audit_error=repr(exc))
        if all(str(gap) in level_result["gaps"] and "Fz_N" in level_result["gaps"][str(gap)]
               and level_result["gaps"][str(gap)]["native_status"]=="Normal Completion" for gap in (1,2)):
            delta = {label:(level_result["gaps"]["1"]["Fz_N"][label]-level_result["gaps"]["2"]["Fz_N"][label])*1e6 for label in "AB"}
            level_result.update(delta_Fz_uN=delta,
                change_from_previous_level_uN={label:delta[label]-previous[label] for label in "AB"},
                existing_approximate_prediction_difference_uN={label:prediction_by_label[label]-delta[label] for label in "AB"})
            level_result["max_absolute_delta_change_uN"] = max(abs(v) for v in level_result["change_from_previous_level_uN"].values())
            level_result["further_refinement_2uN_stability_met"] = level==2 and level_result["max_absolute_delta_change_uN"]<=2
            previous = delta
        result["levels"][str(level)] = level_result
    modes = output/"external_modes/mode_feasibility.json"
    if modes.exists():
        feasibility = read(modes)
        result["external_mode_representation_accepted"] = all(m["excitation_representation_accepted"] for m in feasibility["modes"])
        result["external_mode_note"] = "Polynomial boundary excitation feasibility only; no native perturbation solve"
    save(output/"precision_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", default="outputs/delta_precision_20261003")
    parser.add_argument("--pid", type=int, required=True)
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    supervision = dict(started_UTC=datetime.now(timezone.utc).isoformat(), watched_pid=args.pid,
                       restart_only_successful_native_stages=True,
                       parser_correction="Accept native convergence-record arguments in any order",
                       code_sha256={str(p):digest(p) for p in (Path(__file__).resolve(),ROOT/"scripts/refine_pair_difference_reference.py")})
    save(output/"supervision.json", supervision)
    try:
        process = psutil.Process(args.pid)
        created = process.create_time()
    except psutil.NoSuchProcess:
        process = None
    while process is not None and process.is_running() and process.create_time()==created:
        record = read(output/"run_record.json")
        if record["active"]=="Complete":
            break
        sleep(20)
    record = read(output/"run_record.json")
    expected_extract = record["stages"].get("level1_gap1_extract", {})
    can_resume = (record["active"]!="Complete" and expected_extract.get("returncode")==0
                  and all(item["returncode"]==0 and "abort_request" not in item for item in record["stages"].values()))
    if can_resume:
        command = [sys.executable,str(ROOT/"scripts/refine_pair_difference_reference.py"),"--output",str(output),
                   "--levels","2","--max-passes","3","--cores","4"]
        supervision["resume"] = dict(command=command, started_UTC=datetime.now(timezone.utc).isoformat(),
            reason="Correct native metadata parsing; retain completed build/solve/extraction and their timings")
        save(output/"supervision.json", supervision)
        with (output/"resumed_pipeline.log").open("w",encoding="utf-8") as stream:
            completed = subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
        supervision["resume"].update(returncode=completed.returncode, ended_UTC=datetime.now(timezone.utc).isoformat())
    summary = summarize(output)
    supervision.update(completed_UTC=datetime.now(timezone.utc).isoformat(), final_active=summary["active"],
                       summary=str(output/"precision_summary.json"))
    save(output/"supervision.json", supervision)


if __name__=="__main__":
    main()
