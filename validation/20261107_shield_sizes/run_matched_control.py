"""One bounded fixed-mesh Delta-B diagnostic after the three main references."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parent
LEGACY=Path(r'C:\Projects\magrun\.venv\Scripts\python.exe')
AEDT=Path(r'C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe')
group=json.loads((ROOT/'group_session.json').read_text())
group_end=datetime.fromisoformat(group['started_UTC']).timestamp()+group['budget_seconds']

def save(value):
    (ROOT/'matched_control_status.json').write_text(json.dumps(value,indent=2),encoding='utf-8')

def field_metrics(g,r):
    e=g-r;v=np.linalg.norm(e,axis=1)
    return {'points':len(g),'relative_L2_percent':float(np.linalg.norm(e)/np.linalg.norm(r)*100),
        'RMS_vector_error_mT':float(np.sqrt(np.mean(v*v))*1000),'max_vector_error_mT':float(v.max()*1000),
        'reference_change_RMS_mT':float(np.linalg.norm(r)/np.sqrt(len(r))*1000),
        'predicted_change_RMS_mT':float(np.linalg.norm(g)/np.sqrt(len(g))*1000)}

try:
    save({'state':'Waiting for frozen-source control plan','same_original_group_deadline':True})
    while True:
        if (ROOT/'STOP_GROUP.json').exists():
            raise RuntimeError('Main group stop marker exists; control not submitted')
        if time.time()>group_end-600:
            save({'state':'Not submitted','reason':'Less than ten minutes remain in the original budget'});sys.exit(0)
        if (ROOT/'matched_control_plan.json').exists() and (ROOT/'Shield_T010_reference/export_status.json').exists():
            break
        time.sleep(3)
    plan=json.loads((ROOT/'matched_control_plan.json').read_text())
    start=time.perf_counter();end=min(group_end,time.time()+480)
    project=Path(plan['project'])
    cmd=[str(AEDT),'-ng','-BatchSolve','Shield_T010_NoShield:Nominal:Reference',
         '-machinelist','list=localhost:1:2:60%:0','-Monitor',str(project)]
    record={'state':'Native fixed-mesh control running','command':cmd,
            'started_UTC':datetime.now(timezone.utc).isoformat(),'independent_adaptive_reference':False,
            'source_is_frozen_completed_snapshot':True,'cores':2}
    with (ROOT/'matched_control_native.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        record['pid']=p.pid
        while p.poll() is None:
            if time.time()>end and 'abort' not in record:
                with (ROOT/'matched_control_abort.log').open('w') as stream:
                    result=subprocess.run([str(AEDT.parent/'desktopproxy.exe'),'-abort',str(project)],
                       stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,timeout=60)
                record['abort']={'reason':'Eight-minute control/original group deadline','request_returncode':result.returncode}
            record['seconds']=time.perf_counter()-start;save(record);time.sleep(3)
    record.update(returncode=p.returncode,native_seconds=time.perf_counter()-start)
    if p.returncode or 'abort' in record:
        raise RuntimeError('Fixed-mesh control failed or was aborted')
    outputs=[]
    for input_project,design,name in [(Path(plan['source_project']),'Shield_T010','matched_source_grid'),
                                      (project,'Shield_T010_NoShield','matched_no_shield_reference')]:
        output=ROOT/name
        if time.time()>end:raise TimeoutError('Control export deadline')
        with (ROOT/f'{name}_export.log').open('w') as stream:
            exported=subprocess.run([str(LEGACY),'-X','utf8',str(ROOT/'export_control_mesh.py'),
               '--project',str(input_project),'--design',design,'--out',str(output)],
               cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW,
               timeout=max(1,min(180,end-time.time())))
        if exported.returncode:raise RuntimeError(f'Control mesh/field export failed: {design}')
        outputs.append(output)
    a,b=[json.loads((p/'mesh_fingerprint.json').read_text()) for p in outputs]
    same=(a['tetrahedra']==b['tetrahedra'] and a['corner_coordinate_set_sha256']==b['corner_coordinate_set_sha256'])
    if not same:
        save({**record,'state':'Mesh check failed','source_mesh':a,'control_mesh':b,'used_for_accuracy':False});sys.exit(0)
    source=np.loadtxt(outputs[0]/'maxwell_B.csv',delimiter=',',skiprows=1)
    frozen_original=np.loadtxt(ROOT/'Shield_T010_reference/maxwell_B.csv',delimiter=',',skiprows=1)
    np.testing.assert_allclose(source,frozen_original,rtol=0,atol=1e-11)
    control=np.loadtxt(outputs[1]/'maxwell_B.csv',delimiter=',',skiprows=1)
    gpu=np.loadtxt(ROOT/'Shield_T010/gpu_B.csv',delimiter=',',skiprows=1)
    gpu0=np.loadtxt(ROOT/'baseline_gpu_B.csv',delimiter=',',skiprows=1)
    for field in [source,control,gpu,gpu0]:
        assert field.shape==(566,6) and np.isfinite(field).all()
        np.testing.assert_allclose(field[:,:3],source[:,:3],rtol=0,atol=1e-13)
    dg,dr=gpu[:,3:]-gpu0[:,3:],source[:,3:]-control[:,3:]
    groups=np.asarray(json.loads((ROOT/'point_groups.json').read_text())['labels'])
    result={'case':'Shield_T010','same_corner_tetrahedron_set_verified':True,'source_mesh':a,'control_mesh':b,
      'all':field_metrics(dg,dr),
      'groups':{g:field_metrics(dg[groups==g],dr[groups==g]) for g in sorted(set(groups))},
      'note':'Same exported TET corner set and counts; quadratic midside nodes were not compared. No-shield control is one fixed-mesh pass, not independent adaptive convergence.'}
    (ROOT/'matched_mesh_delta_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save({**record,'state':'Completed and mesh-audited','total_seconds':time.perf_counter()-start,
          'same_corner_tetrahedron_set_verified':True,'metrics':result['all']})
except Exception as error:
    save({'state':'Control unavailable','reason':str(error),'used_for_accuracy':False})
    raise
