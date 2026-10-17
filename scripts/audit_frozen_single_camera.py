"""Audit a completed Maxwell pass retained by a documented desktopproxy abort.

This records experimental field reuse, never adaptive convergence certification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--pass", dest="pass_number", required=True, type=int)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    provenance = json.loads((source / "provenance.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    project = Path(provenance["project"])
    design = provenance["design"]
    results = project.with_suffix(".aedtresults")
    metadata = results / (design + ".asol")
    text = metadata.read_text()
    profile = (source / "profile.txt").read_text()
    if not manifest.get("isolated") or "Status: Aborted" not in profile:
        raise ValueError("Expected an isolated, cleanly aborted source")
    p = args.pass_number
    last_pass = re.search(r"LP\(VID=\d+, Ver=\d+, P=(\d+)\)", text)
    completed = re.search(r"Completed\s*:\s*(\d+)", (source / "convergence.txt").read_text())
    history = json.loads((source / "force_adaptive_history.json").read_text())
    if not last_pass or not completed or int(last_pass[1]) != p or int(completed[1]) != p:
        raise ValueError("Retained native metadata and convergence pass differ")
    if list(map(int, history["passes"])) != list(range(1, p+1)):
        raise ValueError("Force history does not match retained passes")
    force = json.loads((source / "forces.json").read_text())["Force_A.Force_z"]
    values = history["forces"]["Force_A.Force_z"]
    if force["unit"] not in ("newton", "N") or len(force["values"]) != 1 or not np.isclose(force["values"][0], values[-1], atol=1e-13, rtol=0):
        raise ValueError("Native LastAdaptive force differs from last completed pass")
    convergence = re.search(r"c\(tet=(\d+), te=([^,]+), ee=([^,]+), p="+str(p)+r", de=([^,]+),", text)
    mesh_count = re.search(r"Total number of mesh elements:\s*(\d+)", (source / "mesh_stats.txt").read_text())
    if not convergence or not mesh_count or int(convergence[1]) != int(mesh_count[1]):
        raise ValueError("Completed native pass and exported mesh do not match")
    if max(float(convergence[3]), float(convergence[4])) > provenance["setup"]["PercentError"]:
        raise ValueError("Frozen pass has not met the energy target")
    last_adaptive = text.split("SolutionName='LastAdaptive'", 1)[1]
    field_dir = re.search(r"Field\(DblID=\d+, Dir='([^']+)'\)", last_adaptive)
    if not field_dir:
        raise ValueError("Retained field directory missing")
    fields = results / (design + ".results") / field_dir[1]
    if not fields.is_dir() or not any(fields.rglob("*")):
        raise ValueError("Native retained field data missing")
    for key in ("SI_positions_checked", "current_polarity_and_ampere_turns_checked", "BH_field_median_error_below_5_percent"):
        if provenance.get(key) is not True:
            raise ValueError("Native extraction guard failed: "+key)
    files = [project, metadata]+[source / name for name in (
        "provenance.json", "manifest.json", "forces.json", "force_adaptive_history.json",
        "convergence.txt", "profile.txt", "mesh_stats.txt", "maxwell_bh.csv",
        "current_check.json", "constitutive_check.json")]
    files += sorted(path for path in fields.rglob("*") if path.is_file())
    result = dict(completed_UTC=datetime.now(timezone.utc).isoformat(), source=str(source),
        selected_completed_pass=p, retained_tetrahedra=int(mesh_count[1]),
        source_energy_error_percent=float(convergence[3]), source_delta_energy_percent=float(convergence[4]),
        last_Fz_N=float(values[-1]), last_Fz_change_uN=abs(values[-1]-values[-2])*1e6,
        native_adaptive_converged=False, original_force_stability_target_met=False,
        source_status="Aborted after completed pass; retained field verified",
        accepted_use="Exploratory distance differences against a fixed 2 mm baseline",
        baseline_cancellation_does_not_certify_field_accuracy=True,
        stop_command=[r"C:\Program Files\AnsysEM\v222\Win64\desktopproxy.exe", "-abort", str(project)],
        stop_documentation="https://ansyshelp.ansys.com/public/Views/Secured/Electronics/v242/en/Subsystems/Mechanical/Content/HPC/desktopproxy%20Utility%20to%20Abort%20or%20Cleanstop%20a%20BatchSolve.htm",
        hashes={str(path):digest(path) for path in files})
    (source / "frozen_source_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({key:value for key,value in result.items() if key != "hashes"},indent=2))


if __name__ == "__main__":
    main()
