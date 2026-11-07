"""Sequential CuPy cases, concurrently monitored against the same group cap."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
TOOL=Path(r'C:\Projects\magtense-gpu\skills\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0,str(TOOL))
from magtense_adapter import assemble_tiles
from bridge import check_points_in_air
from air_field import evaluate

session=json.loads((ROOT/'group_session.json').read_text())
deadline=datetime.fromisoformat(session['started_UTC']).timestamp()+session['budget_seconds']
def save(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

try:
    baseline=ROOT.parent/'20261107_repair_staged/full_pair_repaired_runtime'
    model=json.loads((baseline/'model.json').read_text())
    tiles,_,_=assemble_tiles(model['objects'],baseline)
    tiles.M=np.load(baseline/'gpu_magnetization.npz')['M_Apm']
    points=np.loadtxt(ROOT/'points.csv',delimiter=',',skiprows=1)
    H,limit=evaluate(tiles,points,validate_air=lambda p:check_points_in_air(baseline,model,p))
    np.savetxt(ROOT/'baseline_gpu_B.csv',np.c_[points,4e-7*np.pi*H],delimiter=',',
               header='x_m,y_m,z_m,Bx_T,By_T,Bz_T',comments='')
    save('baseline_gpu_evaluation.json',{'source':str(baseline),'reused_previous_M':True,
                                        'air_field_regularization':limit,'points':566})
    for name in ['Shield_T010','Shield_T020','Shield_Large_T020','Shield_T020_refined']:
        if time.time()>deadline or (ROOT/'STOP_GROUP.json').exists():
            raise RuntimeError('Stopped before next GPU case')
        record={'state':'Running','case':name,'started_UTC':datetime.now(timezone.utc).isoformat()}
        start=time.perf_counter()
        with (ROOT/f'{name}_gpu_worker.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-X','utf8',str(ROOT/'run_gpu_case.py'),name],
               cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            record['pid']=process.pid
            while process.poll() is None:
                if time.time()>deadline or (ROOT/'STOP_GROUP.json').exists():
                    process.terminate();process.wait(timeout=30)
                    record['terminated_reason']='Deadline or another worker failure'
                record['seconds']=time.perf_counter()-start
                save(f'{name}_gpu_record.json',record)
                time.sleep(3)
        record.update(state='Finished',returncode=process.returncode,seconds=time.perf_counter()-start)
        save(f'{name}_gpu_record.json',record)
        if process.returncode or 'terminated_reason' in record:
            raise RuntimeError(f'GPU case failed/stopped: {name}')
        print(f'COMPLETE {name}: {record["seconds"]:.1f}s',flush=True)
    save('gpu_group_status.json',{'state':'Completed','ended_UTC':datetime.now(timezone.utc).isoformat()})
except Exception as error:
    save('STOP_GROUP.json',{'worker':'gpu','reason':str(error),'UTC':datetime.now(timezone.utc).isoformat()})
    save('gpu_group_status.json',{'state':'Stopped','reason':str(error)})
    raise
