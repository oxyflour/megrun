"""Bounded diagnostic of the default iteration cap; no Maxwell submission."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
TOOL=Path(r'C:\Projects\magtense-gpu\repaired-skill\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0,str(TOOL))
from magtense_adapter import assemble_tiles
from gpu_solver import MaterialLaws

source=ROOT/'Shield_T010'
out=ROOT/'Shield_T010_iteration_check'
out.mkdir(exist_ok=False)
model=json.loads((source/'model.json').read_text())
for obj in model['objects']:
    obj['mesh_npz']=str(source/obj['mesh_npz'])
(out/'model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
shutil.copy2(ROOT/'points.csv',out/'points.csv')
cache='K_volume_r2_f1.5_float64.npy'
os.link(source/cache,out/cache)
shutil.copy2(source/(cache+'.json'),out/(cache+'.json'))
command=[sys.executable,'-X','utf8',str(TOOL/'gpu_solver.py'),'--out',str(out),
         '--backend','cupy','--dtype','float64','--operator','volume','--method','proximal',
         '--tol','1e-8','--maxiter','12000','--cache-k']
session=json.loads((ROOT/'group_session.json').read_text())
remaining=datetime.fromisoformat(session['started_UTC']).timestamp()+session['budget_seconds']-time.time()
start=time.perf_counter()
with (out/'diagnostic.log').open('w',encoding='utf-8') as log:
    result=subprocess.run(command,cwd=TOOL,stdout=log,stderr=subprocess.STDOUT,
           creationflags=subprocess.CREATE_NO_WINDOW,timeout=min(180,remaining))
record={'purpose':'Determine whether default 6000-step cap caused the failure; no Maxwell retry',
        'maxiter':12000,'tolerance':1e-8,'original_K_and_materials_reused':True,
        'seconds':time.perf_counter()-start,'returncode':result.returncode}
if result.returncode==0:
    info=json.loads((out/'gpu_solve_info.json').read_text())
    tiles,curves,_=assemble_tiles(model['objects'],out)
    M=np.load(out/'gpu_magnetization.npz')['M_Apm']
    K=np.load(out/cache,mmap_mode='r')
    H=(K@M.ravel()).reshape(-1,3)
    laws=MaterialLaws(tiles,curves,np,'float64')
    residual=float(np.linalg.norm(M-laws.eval(H)[0])/max(np.linalg.norm(M),1))
    assert residual<=1e-8
    record.update(converged=info['converged'],iterations=info['iterations'],
                  residual=info['residual'],independent_CPU_material_residual=residual,
                  native_K_H_relative_error=info['K_magtense_field_check_relative_error'])
(out/'diagnostic_result.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2),flush=True)
if result.returncode:
    raise RuntimeError('Bounded iteration-cap diagnostic did not converge; no further retries')
