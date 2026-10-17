import os
import json
from pathlib import Path
os.environ["ANSYSEM_ROOT222"] = r"C:\Program Files\AnsysEM\v222\Win64"
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings
settings.use_grpc_api = True
session = json.load(open("outputs/maxwell/session.json"))
app = Maxwell3d(projectname=session["project"], designname="Gap_2mm_Ref", specified_version="2022.2",
                aedt_process_id=session["pid"], non_graphical=True, close_on_exit=False)
folder = Path("outputs/maxwell/Gap_2mm_Ref").resolve()
# PyAEDT 0.8 writes Unit=mm in sample files; arguments must be model units.
points = [[0, 2.975, 2.85], [2.975, 0, 2.85], [3.5, 0, 3]]
for quantity in ["J", "B"]:
    path = folder / (quantity + "_direction.fld")
    print(app.post.export_field_file(quantity, solution="Reference : LastAdaptive", output_dir=str(path),
                    sample_points=points, export_in_si_system=True), flush=True)
    if path.exists():
        print(path.read_text(), flush=True)
app.release_desktop(close_projects=False, close_desktop=False)
