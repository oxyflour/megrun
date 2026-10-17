import os
import json
os.environ["ANSYSEM_ROOT222"] = r"C:\Program Files\AnsysEM\v222\Win64"
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings
settings.use_grpc_api = True
session = json.load(open("outputs/maxwell/session.json"))
app = Maxwell3d(projectname=session["project"], designname="Gap_2mm", specified_version="2022.2",
                aedt_process_id=session["pid"], non_graphical=True, close_on_exit=False)
print("MESSAGES", app.odesktop.GetMessages(app.project_name, app.design_name, 0), flush=True)
print("TERMINAL", app.modeler["A_Terminal"].bounding_box, app.modeler["A_Terminal"].faces[0].normal, flush=True)
print("COIL", app.modeler["A_AF_COIL_COPPER"].bounding_box, flush=True)
print("MATERIAL", app.materials["Magrun_MAGNET_X_POS"]._props, flush=True)
app.release_desktop(close_projects=False, close_desktop=False)
