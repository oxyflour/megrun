"""Internal build/audit/extract worker; solves use the budgeted driver."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_mag.array_maxwell import native_worker


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["build", "extract", "audit"], required=True)
    args = parser.parse_args()
    native_worker(args.config, args.project, args.phase, args.output)


if __name__ == "__main__":
    main()
