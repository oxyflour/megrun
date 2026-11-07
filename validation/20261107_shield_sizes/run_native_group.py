"""Sequential real Maxwell references, under the user's 75-minute group cap."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
AEDT=Path(r'C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe')
LEGACY=Path(r'C:\Projects\magrun\.venv\Scripts\python.exe')
session=json.loads((ROOT/'group_session.json').read_text())
deadline=datetime.fromisoformat(session['started_UTC']).timestamp()+session['budget_seconds']
cases=json.loads((ROOT/'prepared_cases.json').read_text())['cases']

def save(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

def export(project,design,out):
    if time.time()>deadline or (ROOT/'STOP_GROUP.json').exists():
        raise RuntimeError('Stopped before native field export')
    with (ROOT/f'{design}_export.log').open('w',encoding='utf-8') as log:
        p=subprocess.run([str(LEGACY),'-X','utf8',str(ROOT/'export_reference.py'),
              '--project',str(project),'--design',design,'--out',str(out)],
              cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
              creationflags=subprocess.CREATE_NO_WINDOW,timeout=min(180,deadline-time.time()))
    if p.returncode:
        raise RuntimeError(f'Native field export failed: {design}')

try:
    # Read existing fine baseline. No new baseline solve is submitted.
    previous=ROOT.parent/'20261107_repair_staged/full_pair_reference.aedt'
    baseline_status=ROOT/'baseline_reference/export_status.json'
    if not baseline_status.exists():
        export(previous,'Gap_3mm_CCW',ROOT/'baseline_reference')
    for case in cases:
        completed=ROOT/f"{case['name']}_reference/export_status.json"
        if completed.exists() and json.loads(completed.read_text()).get('adaptive_converged'):
            print(f"REUSE completed reference {case['name']}",flush=True)
            continue
        if time.time()>deadline or (ROOT/'STOP_GROUP.json').exists():
            raise RuntimeError('Stopped before starting next Maxwell design')
        name,project=case['name'],Path(case['project'])
        command=[str(AEDT),'-ng','-BatchSolve',f'{name}:Nominal:Reference',
                 '-machinelist','list=localhost:1:4:60%:0','-Monitor',str(project)]
        record={'command':command,'started_UTC':datetime.now(timezone.utc).isoformat(),
                'state':'Native solve running','case':name}
        start=time.perf_counter()
        with (ROOT/f'{name}_native.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            record['pid']=process.pid
            while process.poll() is None:
                reason='75-minute group deadline' if time.time()>deadline else (
                       'Another validation worker failed' if (ROOT/'STOP_GROUP.json').exists() else None)
                if reason and 'abort' not in record:
                    with (ROOT/f'{name}_abort.log').open('w') as abort_log:
                        aborted=subprocess.run([str(AEDT.parent/'desktopproxy.exe'),'-abort',str(project)],
                          cwd=ROOT,stdout=abort_log,stderr=subprocess.STDOUT,
                          creationflags=subprocess.CREATE_NO_WINDOW,timeout=60)
                    record['abort']={'reason':reason,'request_returncode':aborted.returncode}
                record['seconds']=time.perf_counter()-start
                save(f'{name}_native_record.json',record)
                time.sleep(3)
        record.update(state='Native process finished; audit pending',returncode=process.returncode,
                      seconds=time.perf_counter()-start,ended_UTC=datetime.now(timezone.utc).isoformat())
        save(f'{name}_native_record.json',record)
        if process.returncode or 'abort' in record:
            raise RuntimeError(f'Native solve failed/aborted: {name}')
        out=ROOT/f'{name}_reference'
        export(project,name,out)
        audit=json.loads((out/'export_status.json').read_text())
        if not audit['adaptive_converged']:
            raise RuntimeError(f'Maxwell did not meet adaptive convergence: {name}')
        record.update(state='Completed and audited',adaptive_converged=True,
                      adaptive_passes=audit['passes'],final_energy_error_percent=audit['energy_error_percent'])
        save(f'{name}_native_record.json',record)
        print(f'COMPLETE {name}: {record["seconds"]:.1f}s, E={audit["energy_error_percent"]}%',flush=True)
    save('native_group_status.json',{'state':'Completed','ended_UTC':datetime.now(timezone.utc).isoformat()})
except Exception as error:
    save('STOP_GROUP.json',{'worker':'native','reason':str(error),'UTC':datetime.now(timezone.utc).isoformat()})
    save('native_group_status.json',{'state':'Stopped','reason':str(error)})
    raise
