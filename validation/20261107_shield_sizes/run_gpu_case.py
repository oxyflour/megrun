"""One shield discretization: attest reused block, build new blocks, solve, audit."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Six physical CPU cores are shared with Maxwell's four-core job. The final
# refinement's CPU kernel uses the other two; CuPy iteration remains on GPU.
if sys.argv[1].endswith('_refined'):
    os.environ['OMP_NUM_THREADS']='2'
    os.environ['MKL_NUM_THREADS']='2'
import numpy as np

ROOT=Path(__file__).resolve().parent
TOOL=Path(r'C:\Projects\magtense-gpu\repaired-skill\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0,str(TOOL))
from magtense_adapter import assemble_tiles
from gpu_solver import MaterialLaws
from reuse_camera_operator import assemble_added

name=sys.argv[1]
out=ROOT/name
model=json.loads((out/'model.json').read_text())
session=json.loads((ROOT/'group_session.json').read_text())
deadline=datetime.fromisoformat(session['started_UTC']).timestamp()+session['budget_seconds']
assert time.time()<deadline and not (ROOT/'STOP_GROUP.json').exists()
tiles,curves,stats=assemble_tiles(model['objects'],out)
old=ROOT.parent/'20261107_repair_staged/full_pair_repaired_runtime/K_volume_r2_f1.5_float64.npy'
cache=out/'K_volume_r2_f1.5_float64.npy'
if cache.exists() and Path(str(cache)+'.json').exists():
    # A fully written cache is checked again by the packaged solver below.
    reuse=json.loads(Path(str(cache)+'.reuse.json').read_text())
else:
    K,reuse=assemble_added(tiles,old,cache,deadline=deadline)
    del K
if not (out/'gpu_solve_info.json').exists():
    command=[sys.executable,'-X','utf8',str(TOOL/'gpu_solver.py'),'--out',str(out),
             '--backend','cupy','--dtype','float64','--operator','volume','--method','proximal',
             '--tol','1e-8','--maxiter','24000','--cache-k']
    with (out/'gpu_cli.log').open('w',encoding='utf-8') as log:
        p=subprocess.run(command,cwd=TOOL,stdout=log,stderr=subprocess.STDOUT,
           creationflags=subprocess.CREATE_NO_WINDOW,timeout=max(1,min(420,deadline-time.time())))
    if p.returncode:
        raise RuntimeError(f'CuPy CLI failed for {name}; inspect gpu_cli.log')
info=json.loads((out/'gpu_solve_info.json').read_text())
assert info['converged'] and info['residual']<=1e-8
assert info['K_magtense_field_check_relative_error']<=1e-9
M=np.load(out/'gpu_magnetization.npz')['M_Apm']
K=np.load(cache,mmap_mode='r')
H=(K@M.ravel()).reshape(-1,3)
laws=MaterialLaws(tiles,curves,np,'float64')
residual=float(np.linalg.norm(M-laws.eval(H)[0])/max(np.linalg.norm(M),1))
assert residual<=1e-8
field=np.loadtxt(out/'gpu_B.csv',delimiter=',',skiprows=1)
assert field.shape==(566,6) and np.isfinite(field).all()
per_object={};start=0
for obj in model['objects']:
    n=stats['per_object'][obj['name']]
    h=np.linalg.norm(H[start:start+n],axis=1)
    b=np.linalg.norm(4e-7*np.pi*(H[start:start+n]+M[start:start+n]),axis=1)
    values={'tiles':n,'H_min_Apm':float(h.min()),'H_max_Apm':float(h.max()),
            'B_min_T':float(b.min()),'B_max_T':float(b.max()),'B_mean_T':float(b.mean())}
    if obj['properties']['kind']=='nonlinear':
        material_curve=np.asarray(obj['properties']['bh_H_Apm_B_T'])
        measured_min,measured_max=material_curve[1,0],material_curve[-2,0]
        values.update(chart_H_min_Apm=float(measured_min),chart_H_max_Apm=float(measured_max),
                      below_first_measured_knot=int(np.sum(h<measured_min)),
                      above_last_measured_knot=int(np.sum(h>measured_max)),
                      above_model_endpoint=int(np.sum(h>material_curve[-1,0])))
    per_object[obj['name']]=values;start+=n
audit={'case':name,'independent_CPU_material_residual':residual,
       'native_K_H_relative_error':info['K_magtense_field_check_relative_error'],
       'new_operator_blocks':reuse,'gpu_solve_seconds':info['solve_seconds'],
       'iterations':info['iterations'],'per_object':per_object,
       'CPU_thread_environment':{key:os.environ.get(key) for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS']},
       'air_field_regularization':info['air_field_regularization']}
(out/'independent_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps({'case':name,'tiles':tiles.n,'residual':residual,'iterations':info['iterations'],
      'new_K_seconds':reuse['new_block_seconds'],'gpu_seconds':info['solve_seconds']},indent=2),flush=True)
