"""Bounded AEDT geometry/material extraction; no solver invocation."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_mag.aedt_source import extract_source, extract_worker


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--project")
    p.add_argument("--design", default="CameraArray")
    p.add_argument("--template-prefix", default="C1_")
    p.add_argument("--origin-mm", type=float, nargs=3, default=[0., 0., 0.])
    p.add_argument("--output", required=True)
    p.add_argument("--budget-minutes", type=float, default=5.)
    p.add_argument("--request", help="Explicit source descriptor, including optional body-role mapping")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.request:
        descriptor = json.loads(Path(args.request).read_text(encoding="utf-8"))
    else:
        if not args.project:
            p.error("--project or --request is required")
        descriptor = {"path": str(Path(args.project).resolve()), "design": args.design,
                      "template_prefix": args.template_prefix, "origin_mm": args.origin_mm}
    if args.worker:
        extract_worker(descriptor, args.output)
    else:
        if not 0 < args.budget_minutes <= 30:
            p.error("Extraction budget must be positive and <=30 minutes")
        result = extract_source(descriptor, args.output, perf_counter() + args.budget_minutes * 60)
        print(str(result))


if __name__ == "__main__":
    main()
