"""Exercise the packaged CUDA CLI on the real middle shield, with a fresh K."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TOOL=ROOT/'skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
source=ROOT/'validation/20261107_shield_sizes/Shield_T020'
out=HERE/'cli_Shield_T020'
out.mkdir(exist_ok=False)
model=json.loads((source/'model.json').read_text())
for obj in model['objects']:
    obj['mesh_npz']=str((source/obj['mesh_npz']).resolve())
(out/'model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
shutil.copy2(source/'points.csv',out/'points.csv')
command=[sys.executable,'-X','utf8',str(TOOL/'gpu_solver.py'),'--out',str(out),
         '--backend','cupy','--dtype','float64','--operator','volume','--k-builder','cuda',
         '--method','proximal','--tol','1e-8','--maxiter','24000','--cache-k']
start=time.perf_counter()
env={**os.environ,'CUPY_CACHE_DIR':str(HERE/'.cupy_cache')}
with (out/'cli.log').open('w',encoding='utf-8') as log:
    p=subprocess.run(command,cwd=TOOL,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=420)
record={'command':command,'returncode':p.returncode,'process_wall_seconds':time.perf_counter()-start}
if p.returncode:
    (out/'cli_record.json').write_text(json.dumps(record,indent=2))
    raise RuntimeError('CUDA CLI failed; see cli.log')
info=json.loads((out/'gpu_solve_info.json').read_text())
assert info['K_builder']=='cuda' and not info['loaded_from_cache'] and info['converged']
assert info['K_magtense_field_check_relative_error']<1e-9
before=np.loadtxt(source/'gpu_B.csv',delimiter=',',skiprows=1)
after=np.loadtxt(out/'gpu_B.csv',delimiter=',',skiprows=1)
np.testing.assert_allclose(before[:,:3],after[:,:3],rtol=0,atol=1e-14)
record['B_relative_difference_from_previous_CPU_K']=float(np.linalg.norm(before[:,3:]-after[:,3:])/np.linalg.norm(before[:,3:]))
record['B_max_absolute_difference_mT']=float(np.linalg.norm(before[:,3:]-after[:,3:],axis=1).max()*1000)
record['timings']={k:v for k,v in info.items() if k.endswith('seconds') or k in ['K_builder','loaded_from_cache','iterations','residual']}
(out/'cli_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2),flush=True)
