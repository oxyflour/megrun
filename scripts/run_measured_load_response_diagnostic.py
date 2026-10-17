"""Compare self responses on the measured native loading field, not its target.

The neighbour representation gate remains failed. This diagnostic does not
relax that gate: both response models receive the actual exported source-only
field, so their self-response difference can be isolated. No pair force fit.
"""
from pathlib import Path
import subprocess
import sys
from time import perf_counter,sleep

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.refine_pair_difference_reference import read,save,digest,utc
from scripts.run_native_response_pilot import AEDT,SOURCE


def main():
    output=ROOT/"outputs/native_equivalent_response_pilot_20261003"
    project=output/"NativeResponsePilot.aedt"
    path=output/"diagnostic_record.json"
    if path.exists():raise FileExistsError(path)
    first=read(output/"run_record.json")
    source=read(output/"VacuumCheck/result.json")
    if source["native_status"]!="Normal Completion" or source["loading_gate_met"]:
        raise ValueError("Diagnostic specifically retains the failed neighbour-loading gate")
    record=dict(started_UTC=utc(),expected_minutes=[8,12],budget_minutes=15,stages={},
        previous_loading_run_minutes=first["elapsed_seconds"]/60,
        neighbour_representation_accepted=False,production_library_accepted=False,
        scope="Self-response diagnostic on the actual measured native source field; not the intended full neighbour field",
        actual_source_field_sha256={str(output/"VacuumCheck"/(q+".fld")):digest(output/"VacuumCheck"/(q+".fld")) for q in ("B","H")},
        no_paired_force_fit=True,original_hashes=first["original_hashes"])
    start=perf_counter();deadline=start+15*60
    try:
        for design in ("PilotZero","PilotHalf","PilotFull"):
            commands=[("solve",[str(AEDT),"-ng","-BatchSolve",design+":Nominal:Reference",
                       "-machinelist","list=localhost:1:4:60%:0","-Monitor",str(project)]),
                      ("extract",[sys.executable,str(ROOT/"scripts/run_equivalent_source_response_pilot.py"),
                        "--phase","extract","--design",design,"--output",str(output)])]
            for phase,command in commands:
                if perf_counter()>deadline-90:raise TimeoutError("Diagnostic budget exhausted before next stage")
                key=design+"_"+phase
                item=dict(command=command,started_UTC=utc(),log=str(output/("diagnostic_"+key+".log")))
                record.update(active=key,active_stage=item)
                begin=perf_counter()
                with Path(item["log"]).open("w",encoding="utf-8") as stream:
                    process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    item["pid"]=process.pid;save(path,record)
                    while process.poll() is None:
                        if perf_counter()>deadline-90 and "stop_request" not in item:
                            item["stop_request"]=dict(UTC=utc(),reason="15-minute diagnostic budget")
                            if phase=="solve":
                                abort=[str(AEDT.parent/"desktopproxy.exe"),"-abort",str(project)]
                                with (output/("diagnostic_"+key+"_abort.log")).open("w") as log:
                                    item["stop_request"]["returncode"]=subprocess.run(abort,stdout=log,
                                        stderr=subprocess.STDOUT,timeout=60,creationflags=subprocess.CREATE_NO_WINDOW).returncode
                            else:process.terminate()
                            save(path,record)
                        if perf_counter()>deadline+60:raise TimeoutError("Cleanup budget exceeded; no further job submitted")
                        sleep(3)
                    item.update(returncode=process.returncode,seconds=perf_counter()-begin,ended_UTC=utc())
                record["stages"][key]=item;record.pop("active_stage",None);save(path,record)
                if item["returncode"] or "stop_request" in item:raise RuntimeError("Diagnostic failed/stopped: "+key)
        record["active"]="Diagnostic complete"
    except Exception as exc:
        record.update(active="Stopped",failure=repr(exc));raise
    finally:
        record.update(ended_UTC=utc(),elapsed_seconds=perf_counter()-start,
            original_sources_unchanged={p:digest(Path(p))==h for p,h in record["original_hashes"].items()})
        save(path,record)


if __name__=="__main__":main()
