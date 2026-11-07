"""Final full-material Maxwell reference, under the explicitly extended budget."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
project = ROOT / 'full_pair_reference.aedt'
budget = json.loads((ROOT / 'full_session.json').read_text())
deadline = datetime.fromisoformat(budget['started_UTC']).timestamp() + budget['budget_seconds']
AEDT = Path(r'C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe')
command = [str(AEDT), '-ng', '-BatchSolve', 'Gap_3mm_CCW:Nominal:Reference',
           '-machinelist', 'list=localhost:1:4:60%:0', '-Monitor', str(project)]
record = {'command': command, 'started_UTC': datetime.now(timezone.utc).isoformat(),
          'state': 'Native reference running'}
t0 = time.perf_counter()
with (ROOT / 'full_pair_native.log').open('w') as stream:
    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    record['pid'] = process.pid
    while process.poll() is None:
        if time.time() > deadline and 'abort' not in record:
            with (ROOT / 'full_pair_abort.log').open('w') as log:
                result = subprocess.run([str(AEDT.parent/'desktopproxy.exe'), '-abort', str(project)],
                    stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
            record['abort'] = {'reason': 'User-approved 45-minute group budget', 'returncode': result.returncode}
        record['seconds'] = time.perf_counter()-t0
        (ROOT / 'full_native_record.json').write_text(json.dumps(record, indent=2))
        time.sleep(3)
record.update(state='Native process finished; audit pending', returncode=process.returncode,
              seconds=time.perf_counter()-t0, ended_UTC=datetime.now(timezone.utc).isoformat())
(ROOT / 'full_native_record.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
if process.returncode or 'abort' in record:
    raise RuntimeError('Full native reference failed or aborted; no automatic repeat')
