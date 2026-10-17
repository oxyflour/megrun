"""Build independent Maxwell cases for all coarse sweep gaps and isolated baseline."""
import subprocess
import sys
import argparse
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(__doc__)
p.add_argument("--project", default="outputs/maxwell_verified/sweep_reference.aedt")
args = p.parse_args()
project = (root / args.project).resolve()
cases = [("Isolated_CCW", None), ("Gap_1mm_CCW", 1), ("Gap_15mm_CCW", 1.5),
         ("Gap_25mm_CCW", 2.5), ("Gap_3mm_CCW", 3)]
for name, gap in cases:
    command = [sys.executable, str(root / "scripts/run_maxwell_case.py"), "--phase", "build",
               "--project", str(project), "--design", name, "--mesh-mm", ".5", "--percent-error", ".5"]
    command += ["--isolated"] if gap is None else ["--distance-mm", str(gap)]
    subprocess.run(command, cwd=root, check=True)
