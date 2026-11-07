"""Budgeted native solve. User approved one 50-minute study on 2026-11-07."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'gap3mm_vacoflux50.aedt'
AEDT = Path(r'C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe')
RECORD = ROOT / 'run_record.json'


def save(data):
    RECORD.write_text(json.dumps(data, indent=2), encoding='utf-8')


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', required=True, choices=['MT_Coarse', 'Gap_3mm_CCW'])
    args = parser.parse_args()
    if RECORD.exists():
        record = json.loads(RECORD.read_text())
    else:
        record = {'study_started_UTC': utc(), 'budget_seconds': 3000,
                  'authorization': 'User explicitly approved 50-minute group budget on 2026-11-07',
                  'stages': {}, 'state': 'Authorized'}
    if args.design in record['stages']:
        raise RuntimeError('This native stage already has a record; no automatic repeat allowed')
    deadline = datetime.fromisoformat(record['study_started_UTC']).timestamp() + record['budget_seconds']
    if time.time() >= deadline:
        raise RuntimeError('Whole-study budget exhausted before submission')
    command = [str(AEDT), '-ng', '-BatchSolve', args.design + ':Nominal:Reference',
               '-machinelist', 'list=localhost:1:4:60%:0', '-Monitor', str(PROJECT)]
    item = {'command': command, 'started_UTC': utc(), 'log': str(ROOT / (args.design + '_native.log'))}
    record['stages'][args.design] = item
    record['state'] = 'Solving ' + args.design
    save(record)
    start = time.perf_counter()
    with Path(item['log']).open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        item['pid'] = process.pid
        save(record)
        while process.poll() is None:
            if time.time() >= deadline and 'abort' not in item:
                abort_command = [str(AEDT.parent / 'desktopproxy.exe'), '-abort', str(PROJECT)]
                with (ROOT / (args.design + '_abort.log')).open('w') as abort_log:
                    abort = subprocess.run(abort_command, stdout=abort_log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
                item['abort'] = {'UTC': utc(), 'reason': 'User-approved total budget exhausted',
                                 'command': abort_command, 'returncode': abort.returncode}
                record['state'] = 'Native abort requested; preserving actual outcome'
                save(record)
            record['elapsed_study_seconds'] = time.time() - (deadline - 3000)
            save(record)
            time.sleep(3)
    item.update(returncode=process.returncode, seconds=time.perf_counter() - start, ended_UTC=utc())
    record['state'] = 'Native stage finished; outcome audit pending'
    save(record)
    print(json.dumps(item, indent=2), flush=True)
    if process.returncode or 'abort' in item:
        raise RuntimeError('Native stage failed/aborted; no automatic retry')


if __name__ == '__main__':
    main()
