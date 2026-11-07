"""Native controls with a shared whole-study 20-minute deadline."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / '20261107_gap3mm_vacoflux50'
TOOL = Path(r'C:\Projects\magtense-gpu\skill-test\magtense-cupy-magnetostatics\scripts\aedt_magtense_bridge')
sys.path.insert(0, str(TOOL))
from bridge import read_maxwell_fld

AEDT = Path(r'C:\Program Files\AnsysEM\v222\Win64\ansysedt.exe')
parser = argparse.ArgumentParser()
parser.add_argument('--refined', action='store_true')
args = parser.parse_args()
studies = json.loads((ROOT / ('native_studies_refined.json' if args.refined else 'native_studies.json')).read_text())
session = json.loads((ROOT / 'session.json').read_text())
deadline = datetime.fromisoformat(session['started_UTC']).timestamp() + session['budget_seconds']
record = {'deadline_UTC': datetime.fromtimestamp(deadline, timezone.utc).isoformat(), 'stages': []}
for design, configuration in studies.items():
    if time.time() > deadline:
        raise RuntimeError('Whole study budget exhausted before native submission')
    folder = ROOT / configuration['case']
    project = Path(configuration['project'])
    command = [str(AEDT), '-ng', '-BatchSolve', design + ':Nominal:Reference',
               '-machinelist', 'list=localhost:1:4:60%:0', '-Monitor', str(project)]
    item = {'design': design, 'command': command, 'started_UTC': datetime.now(timezone.utc).isoformat()}
    t0 = time.perf_counter()
    with (folder / 'native.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        item['pid'] = process.pid
        while process.poll() is None:
            if time.time() > deadline and 'abort' not in item:
                abort = subprocess.run([str(AEDT.parent / 'desktopproxy.exe'), '-abort', str(project)],
                    capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW, timeout=60)
                item['abort'] = {'reason': 'Whole-study budget', 'returncode': abort.returncode}
            time.sleep(2)
    item.update(returncode=process.returncode, seconds=time.perf_counter()-t0)
    record['stages'].append(item)
    (ROOT / ('native_run_refined.json' if args.refined else 'native_run.json')).write_text(json.dumps(record, indent=2))
    for logfile in project.with_suffix('.aedt.batchinfo').glob('*'):
        if logfile.is_file():
            shutil.copy2(logfile, folder / logfile.name)
    print(json.dumps(item), flush=True)
    if process.returncode or 'abort' in item:
        raise RuntimeError('Native control failed/aborted; retaining records')

os.environ['ANSYSEM_ROOT222'] = str(AEDT.parent)
from pyaedt import Maxwell3d
from pyaedt.generic.settings import settings
settings.use_grpc_api = True
settings.enable_error_handler = False
app = Maxwell3d(projectname=str(project), designname=next(iter(studies)), specified_version='2022.2',
                non_graphical=True, new_desktop_session=True, close_on_exit=False)
try:
    points = np.loadtxt(BASE / 'points.csv', delimiter=',', skiprows=1)
    for design, configuration in studies.items():
        if time.time() > deadline:
            raise RuntimeError('Whole-study budget exhausted before field extraction')
        app.set_active_design(design)
        folder = ROOT / configuration['case']
        for method, filename in [('export_convergence', 'convergence.txt'),
                                 ('export_profile', 'profile.txt'), ('export_mesh_stats', 'mesh_stats.txt')]:
            getattr(app, method)('Reference', **{'mesh_path' if method == 'export_mesh_stats' else 'file_path': str(folder / filename)})
        with (folder / 'points_mm.pts').open('w') as stream:
            stream.write('Unit=mm\n')
            np.savetxt(stream, points*1000, fmt='%.15g')
        assert app.post.export_field_file('B', solution='Reference : LastAdaptive',
            output_dir=str(folder / 'maxwell_B.fld'), sample_points_file=str(folder / 'points_mm.pts'),
            export_in_si_system=True, export_with_sample_points=True)
        B, scale = read_maxwell_fld(folder / 'maxwell_B.fld', points)
        assert scale == 1
        np.savetxt(folder / 'maxwell_B.csv', np.c_[points, B], delimiter=',',
                   header='x_m,y_m,z_m,Bx_T,By_T,Bz_T', comments='')
        predicted = np.loadtxt(folder / 'B.csv', delimiter=',', skiprows=1)[:, 3:]
        error = np.linalg.norm(predicted-B, axis=1)
        metrics = {'reference': 'Maxwell independently solved staged material, same air points',
                   'B_relative_L2': float(np.linalg.norm(predicted-B)/np.linalg.norm(B)),
                   'B_RMS_vector_error_T': float(np.sqrt(np.mean(error**2))),
                   'B_max_vector_error_T': float(error.max()),
                   'native_normal_completion': 'Normal Completion' in (folder / 'profile.txt').read_text(),
                   'native_convergence_text': (folder / 'convergence.txt').read_text()}
        (folder / 'comparison.json').write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
finally:
    app.release_desktop(close_projects=True, close_desktop=True)
