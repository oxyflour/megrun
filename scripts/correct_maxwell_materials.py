"""Reuse CAD and setups in a fresh project, without copying old solution data."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_mag.materials import benchmark_materials


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("source")
    p.add_argument("destination")
    p.add_argument("--design", default="Gap_2mm_CCW")
    args = p.parse_args()
    source, destination = Path(args.source).resolve(), Path(args.destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    os.environ["ANSYSEM_ROOT222"] = r"C:\Program Files\AnsysEM\v222\Win64"
    from pyaedt import Maxwell3d
    from pyaedt.generic.settings import settings
    settings.use_grpc_api = True
    settings.enable_error_handler = False
    app = Maxwell3d(projectname=str(destination), designname=args.design,
                    specified_version="2022.2", non_graphical=True,
                    new_desktop_session=True, close_on_exit=False)
    try:
        curve = benchmark_materials()["yoke"]
        hs = np.unique(np.r_[np.linspace(0, 8000, 401), curve.h, [100000, 200000]])
        bs = curve.b_value(hs)
        steel = app.materials["Magrun_Steel_BH"]
        steel.permeability.value = np.c_[hs, bs].tolist()
        if not steel.update():
            raise RuntimeError("Material update failed")
        if not app.save_project():
            raise RuntimeError("Save failed")
        for design in app.design_list:
            old = source.parent / design
            if not (old / "manifest.json").exists():
                continue
            folder = destination.parent / design
            folder.mkdir(parents=True, exist_ok=True)
            manifest = json.loads((old / "manifest.json").read_text())
            manifest.update(project=str(destination), source_geometry_project=str(source),
                            bh_coordinate_order="H_A_per_m, B_T",
                            bh_alignment="Dense PCHIP samples plus original knots and μ0 extrapolation")
            (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
            np.savetxt(folder / "maxwell_bh.csv", np.c_[hs, bs], delimiter=",",
                       header="H_A_per_m,B_T", comments="")
    finally:
        app.release_desktop(close_projects=True, close_desktop=True)


if __name__ == "__main__":
    main()
