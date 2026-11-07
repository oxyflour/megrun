"""Check worst finite field errors against stable symmetric air limits."""
import json
from pathlib import Path
import sys

import numpy as np
from magtense.magstatics import get_H_field

ROOT=Path(__file__).resolve().parent
TOOL=Path(r'C:\Projects\magtense-gpu\repaired-skill\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0,str(TOOL))
from magtense_adapter import assemble_tiles
from bridge import check_points_in_air
from air_field import evaluate

results={}
for name in ['Shield_T010','Shield_T020','Shield_Large_T020','Shield_T020_refined']:
    out=ROOT/name
    saved=out/'finite_worst_point_audit.json'
    if saved.exists():
        results[name]=json.loads(saved.read_text());continue
    refname='Shield_T020' if name.endswith('_refined') else name
    reference=ROOT/f'{refname}_reference/maxwell_B.csv'
    if not reference.exists() or not (out/'gpu_B.csv').exists():
        continue
    gpu=np.loadtxt(out/'gpu_B.csv',delimiter=',',skiprows=1)
    ref=np.loadtxt(reference,delimiter=',',skiprows=1)
    error=np.linalg.norm(gpu[:,3:]-ref[:,3:],axis=1)
    indices=np.argsort(error)[-10:]
    model=json.loads((out/'model.json').read_text())
    tiles,_,_=assemble_tiles(model['objects'],out)
    tiles.M=np.load(out/'gpu_magnetization.npz')['M_Apm']
    calls=0
    def force_limit(t,p):
        global calls
        calls+=1
        return np.full((len(p),3),np.nan) if calls==1 else get_H_field(t,p)
    H,limit=evaluate(tiles,gpu[indices,:3],validate_air=lambda p:check_points_in_air(out,model,p),get_field=force_limit)
    difference=np.linalg.norm(4e-7*np.pi*H-gpu[indices,3:],axis=1)*1000
    result={'point_indices':indices.tolist(),'nominal_coordinates_unchanged':True,
            'native_vs_certified_limit_difference_mT':difference.tolist(),
            'maximum_difference_mT':float(difference.max()),
            'maximum_Maxwell_error_mT':float(error.max()*1000),
            'limit_checks':limit}
    saved.write_text(json.dumps(result,indent=2),encoding='utf-8');results[name]=result
(ROOT/'finite_field_checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps({n:{'native_limit_max_difference_mT':v['maximum_difference_mT'],
                   'Maxwell_max_difference_mT':v['maximum_Maxwell_error_mT']} for n,v in results.items()},indent=2))
