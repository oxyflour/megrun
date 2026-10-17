import os
import json
from pathlib import Path
os.environ["ANSYSEM_ROOT222"] = r"C:\Program Files\AnsysEM\v222\Win64"
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings

settings.use_grpc_api = True
settings.enable_screen_logs = True
app = Maxwell3d(projectname="Magrun_Reference_Probe", designname="Probe", solution_type="Magnetostatic",
                specified_version="2022.2", non_graphical=True, new_desktop_session=True, close_on_exit=False)
folder = Path("outputs/maxwell")
folder.mkdir(parents=True, exist_ok=True)
app.save_project(str((folder / "probe.aedt").resolve()))
info = {"version": app.aedt_version_id, "pid": app.desktop_class.aedt_process_id,
        "project": app.project_name, "design": app.design_name, "solution_type": app.solution_type,
        "project_path": str((folder / "probe.aedt").resolve())}
print(json.dumps(info), flush=True)
(folder / "session.json").write_text(json.dumps(info, indent=2))
app.release_desktop(close_projects=False, close_desktop=False)
