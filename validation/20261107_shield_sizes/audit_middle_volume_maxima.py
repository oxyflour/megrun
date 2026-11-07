"""Read-only full-volume peak audit for the completed middle-size reference."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
project=ROOT/'gap3mm_shield_sizes_v3.aedt'
before=hashlib.sha256(project.read_bytes()).hexdigest()
os.environ['ANSYSEM_ROOT222']=r'C:\Program Files\AnsysEM\v222\Win64'
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings
settings.use_grpc_api=True;settings.enable_error_handler=False
app=Maxwell3d(projectname=str(project),designname='Shield_T020',specified_version='2022.2',
              non_graphical=True,new_desktop_session=True,close_on_exit=False)
try:
    values={q:app.post.get_scalar_field_value(q,scalar_function='Maximum',
           solution='Reference : LastAdaptive',object_name='MID_SHIELD',object_type='volume')
           for q in ['Mag_H','Mag_B']}
    assert all(isinstance(v,(int,float)) and not isinstance(v,bool) and v>0 for v in values.values())
    out=ROOT/'Shield_T020_reference'
    (out/'volume_field_maxima.json').write_text(json.dumps(values,indent=2),encoding='utf-8')
    path=out/'constitutive_audit.json';audit=json.loads(path.read_text())
    audit['MID_SHIELD'].update({'volume_max_'+q:v for q,v in values.items()})
    path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
finally:
    app.release_desktop(close_projects=True,close_desktop=True)
assert hashlib.sha256(project.read_bytes()).hexdigest()==before
print(json.dumps(values),flush=True)
