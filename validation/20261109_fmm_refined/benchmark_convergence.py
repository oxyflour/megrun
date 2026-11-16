"""Small FEM--BEM convergence benchmarks, without AEDT or MagTense binaries.

Target: current aedt_magtense_bridge/fem_bem.py Newton/GMRES, material laws,
and optional CUDA H boundary. Inputs are deterministic tetrahedral spheres,
boxes and synthetic B-H tables below; no solver source is copied here.
Reference: SciPy HYBR / trust-region least squares (finite differences), after dense Cholesky
elimination of q, on the SAME mesh/operator. A second reference doubles target
quadrature order to separate iteration error from integration sensitivity.
Sphere H also has an independently evaluated demagnetizing-factor 1/3 oracle;
its error is diagnostic, since these small polyhedra are not exact spheres.

Run from repository root (Python with NumPy/SciPy):
  python validation/20261109_fmm_refined/benchmark_convergence.py --list
  python validation/20261109_fmm_refined/benchmark_convergence.py --case knee
  python validation/20261109_fmm_refined/benchmark_convergence.py --output NEW.json
Optional: --backend cuda_hmatrix --fem-preconditioner amg needs CuPy/CUDA/PyAMG.
Add --compute-backend cupy for resident FEM/Krylov/device AmgX; AMGX_LIBRARY
may name its DLL/SO. Independent reference solves still use CPU NumPy/SciPy.
--compare-backends runs isolated CPU/LU, hybrid CUDA/PyAMG, and resident
CUDA/AmgX workers with identical inputs/gates and retains one combined JSON.
--preconditioner none measures Krylov sensitivity under the same acceptance.
--level 3 raises sphere boundary faces from 128 to 512; dense reference storage
is quadratic and bounded by FEMBEM's 256 MiB/4096-face guards. No large-model
GPU throughput, Maxwell accuracy, mesh convergence or hysteresis is certified.

Outputs: JSON to stdout, or an exclusively created --output file, including
inputs/hashes, dependency versions, phase timings, histories and gate failures.
Pass: freshly recomputed full dense weak residual <=1e-8; same-operator H/B/q
agreement <=1e-5 relative with absolute floors; per-component net flux <=1e-8;
quadrature H/B change <=0.5% with absolute floors. Gates are separately named.
Every selected case runs even after another fails; any failed gate/reference,
exception or missing dependency gives nonzero exit. No skips count as passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / 'skills/magtense-cupy-magnetostatics/scripts/aedt_magtense_bridge'
LIMITS = dict(weak_residual=1e-8, reference_residual=1e-10,
              solution_rtol=1e-5, H_atol_Apm=1e-6, B_atol_T=1e-10,
              q_atol_Apm=1e-6, quadrature_rtol=.005, component_net_flux=1e-8)


def cases():
    """Freeze difficulty ladders before execution; B-H curves are test models."""
    mu0 = 4e-7 * 3.141592653589793
    steel = dict(kind='nonlinear', bh_H_Apm_B_T=[
        [0., 0.], [300., 1.9], [800., 2.1], [4000., 2.3],
        [200000., 2.3+mu0*(200000.-4000.)]])
    specs = [dict(name='vacuum', family='control', shape='sphere',
                  properties=[dict(kind='linear', mu_r=1.)], H0=[200., -30., 45.])]
    for mu in (100., 1e4, 1e6):
        specs.append(dict(name=f'high_mu_{mu:g}', family='high_mu', shape='sphere',
                          properties=[dict(kind='linear', mu_r=mu)], H0=[0., 0., 1000.]))
    # Continuous piecewise linear M(H), exact knee at Hk=Ms/chi.
    # Applied field putting an exact sphere at that knee is Hk+Ms/3.
    chi, ms = 4999., 1.2e6
    hk = ms / chi
    curve = [[h, mu0*(h+min(chi*h, ms))] for h in (0., hk, 10*hk, 100*hk)]
    for factor in (.98, 1., 1.02):
        specs.append(dict(name=f'knee_{factor:g}', family='knee', shape='sphere',
                          properties=[dict(kind='nonlinear', bh_H_Apm_B_T=curve)],
                          H0=[0., 0., factor*(hk+ms/3)], chi=chi, Ms_Apm=ms))
    for gap in (.1, .01, .001):
        specs.append(dict(name=f'gap_{gap:g}', family='gap', shape='pair', gap_ratio=gap,
                          properties=[dict(kind='linear', mu_r=1e4)]*2,
                          H0=[1000., 170., -80.]))
    for thickness in (.1, .01, .001):
        specs.append(dict(name=f'thin_{thickness:g}', family='thin', shape='sheet',
                          thickness_ratio=thickness,
                          properties=[dict(kind='linear', mu_r=1e5)], H0=[1000., 0., 100.]))
    pm = dict(kind='permanent', br_T=1.2, mu_r=1.05, direction=[1., 0., 0.])
    for br in (.01, 1.2):
        specs.append(dict(name=f'mixed_{br:g}', family='mixed', shape='triplet',
                          gap_ratio=.02, properties=[dict(pm, br_T=br), steel,
                              dict(kind='linear', mu_r=1e5)],
                          H0=[0., 0., 0.] if br < 1 else [6e5, 4e4, -2e4]))
    return specs


def geometry(spec, level):
    import numpy as np
    from scipy.spatial import ConvexHull
    if spec['shape'] == 'sphere':
        vertices = np.r_[np.eye(3), -np.eye(3)]
        for _ in range(level):
            faces = ConvexHull(vertices).simplices
            mids = (vertices[faces]+np.roll(vertices[faces], -1, axis=1)).reshape(-1, 3)/2
            vertices = np.unique(np.r_[vertices, mids], axis=0)
            vertices /= np.linalg.norm(vertices, axis=1)[:, None]
        faces = ConvexHull(vertices).simplices
        tetra = np.concatenate((np.zeros((len(faces), 1, 3)), vertices[faces]), axis=1)*1e-3
        return tetra, np.zeros(len(tetra), dtype=int)
    unit = np.array([[np.zeros(3), np.eye(3)[p[0]],
                      np.eye(3)[p[0]]+np.eye(3)[p[1]], np.ones(3)]
                     for p in permutations(range(3))])
    if spec['shape'] == 'sheet':
        return unit*[1e-3, 1e-3, spec['thickness_ratio']*1e-3], np.zeros(6, dtype=int)
    shifts = [0., 1.+spec['gap_ratio']]
    if spec['shape'] == 'triplet':
        shifts.append(-1.-spec['gap_ratio'])
    return (np.concatenate([unit+[x, 0., 0.] for x in shifts])*1e-3,
            np.repeat(np.arange(len(shifts)), 6))


def sphere_internal_H(spec):
    applied = spec['H0'][2]
    if spec['family'] == 'high_mu':
        return 3*applied/(spec['properties'][0]['mu_r']+2)
    return (applied/(1+spec['chi']/3) if applied <= spec['Ms_Apm']/spec['chi']+spec['Ms_Apm']/3
            else applied-spec['Ms_Apm']/3)


def direct_reference(mesh, H0, spec):
    """Independent nonlinear driver: eliminate BEM, HYBR finite differences.

    This shares discretization/material evaluation, but never calls solve(),
    GMRES, its block preconditioner, line search or material tangent.
    """
    import numpy as np
    from scipy.linalg import cho_solve
    from scipy.optimize import least_squares, root
    n = len(mesh.vertices)
    C = np.zeros((len(mesh.faces), n))
    C[:, mesh.boundary_nodes] = mesh.C
    trace_to_q = -cho_solve(mesh.V_factor, C)
    coupling = mesh.mass @ trace_to_q
    zero = np.zeros(n)
    initial = mesh.residual(zero, np.zeros(len(mesh.faces)), H0)[0]
    scale = max(float(np.linalg.norm(initial)), np.finfo(float).tiny)
    if not np.any(initial):
        return zero, np.zeros(len(mesh.faces)), dict(residual=0., attempts=[])
    amplitude = max(float(np.linalg.norm(H0)), float(np.max(abs(mesh.laws.rem))), 1.)
    def fun(x):
        u = x*amplitude
        return (mesh.volume_residual(u, H0)[0]-coupling @ u)/scale
    seed = zero
    if spec['shape'] == 'sphere':
        # Independent continuum guess; never initialize with the tested solve.
        seed = (mesh.vertices-.5) @ (H0-[0., 0., sphere_internal_H(spec)])/amplitude
    result = root(fun, seed, method='hybr', options=dict(xtol=1e-10, maxfev=5000))
    attempts = [dict(method='hybr', scipy_nfev=int(result.nfev),
                     scipy_success=bool(result.success), message=str(result.message))]
    if np.linalg.norm(fun(result.x)) > LIMITS['reference_residual']:
        # A genuine kink can stall HYBR. A different SciPy globalization is
        # independent of the tested Armijo/GMRES path; acceptance is unchanged.
        result = least_squares(fun, result.x, jac='3-point', x_scale='jac',
                               ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=500)
        attempts.append(dict(method='trf_3point', scipy_nfev=int(result.nfev),
                             scipy_success=bool(result.success), message=str(result.message)))
    u = result.x*amplitude
    q = trace_to_q @ u
    residual = float(np.linalg.norm(mesh.residual(u, q, H0)[0])/scale)
    if not np.isfinite(residual) or residual > LIMITS['reference_residual']:
        raise RuntimeError(f'Unresolved independent reference: residual={residual:g}; {attempts}')
    return u, q, dict(residual=residual, attempts=attempts)


def rms(value, weights):
    import numpy as np
    return float(np.sqrt(np.sum(weights*np.sum(np.asarray(value)**2, axis=-1))/np.sum(weights)))


def compare(actual, reference, weights, atol, rtol):
    error, amplitude = rms(actual-reference, weights), rms(reference, weights)
    return dict(error_RMS=error, reference_RMS=amplitude, atol=atol, rtol=rtol,
                relative_L2=error/amplitude if amplitude else None,
                passed=error <= atol+rtol*amplitude)


def run_case(spec, args, record, checkpoint):
    import numpy as np
    from fem_bem import FEMBEM, material_laws
    tetra, ids = geometry(spec, args.level)
    record.update(input=spec, tetrahedra=len(tetra),
                  geometry_sha256=hashlib.sha256(tetra.tobytes()+ids.tobytes()).hexdigest(),
                  newton_history=[], linear_progress=[], field_correction_history=[])
    def timed(name, function):
        if args.backend == 'cuda_hmatrix' or args.compute_backend == 'cupy':
            import cupy as cp
            cp.cuda.get_current_stream().synchronize()
        t = time.perf_counter()
        try:
            return function()
        finally:
            if args.backend == 'cuda_hmatrix' or args.compute_backend == 'cupy':
                cp.cuda.get_current_stream().synchronize()
            record.setdefault('timings_seconds', {})[name] = time.perf_counter()-t
    def build(order, backend='dense'):
        options = dict(leaf_size=8, eta=2., rtol=1e-8, max_rank=64, budget_gib=.25,
                       storage_dtype='float64', far_quadrature_order=order)
        return FEMBEM(tetra, material_laws(spec['properties'], ids), quadrature_order=order,
                      boundary_backend=backend, boundary_options=options if backend != 'dense' else None)
    record['phase'] = 'dense_reference_setup'
    dense = timed('dense_reference_setup', lambda: build(args.order))
    record['P1_nodes'], record['boundary_faces'] = len(dense.vertices), len(dense.faces)
    H0 = np.asarray(spec['H0'])
    mesh = dense
    if args.backend != 'dense':
        record['phase'] = 'tested_operator_setup'
        mesh = timed('tested_operator_setup', lambda: build(args.order, args.backend))
    else:
        record.setdefault('timings_seconds', {})['tested_operator_setup'] = 0.
        record['tested_setup_reuses_dense_reference'] = True
    def progress(event, **values):
        target = ('newton_history' if event == 'newton' else
                  'field_correction_history' if event == 'field_correction' else 'linear_progress')
        record[target].append(values)
        print(json.dumps(dict(case=spec['name'], event=event, **values)), file=sys.stderr, flush=True)
        checkpoint()
    record['phase'] = 'solve'
    u, q, info = timed('solve', lambda: mesh.solve(
        H_external=H0, tol=1e-9, maxiter=args.maxiter, linear_rtol=1e-10,
        linear_maxiter=args.linear_maxiter, preconditioner=args.preconditioner,
        fem_preconditioner=args.fem_preconditioner, progress=progress, compute_backend=args.compute_backend))
    record['solver'] = info
    record['phase'] = 'certify'
    initial = dense.residual(np.zeros_like(u), np.zeros_like(q), H0)[0]
    scale = max(float(np.linalg.norm(initial)), np.finfo(float).tiny)
    residual = float(np.linalg.norm(dense.residual(u, q, H0)[0])/scale)
    H, B, *_ = dense.fields(u, H0)
    record['dense_weak_residual'] = residual
    record['phase'] = 'same_operator_reference'
    ur, qr, ri = timed('same_operator_reference', lambda: direct_reference(dense, H0, spec))
    record['reference'] = ri
    Hr, Br, *_ = dense.fields(ur, H0)
    gates = dict(dense_weak_residual=dict(value=residual, limit=LIMITS['weak_residual'],
                                         passed=np.isfinite(residual) and residual <= LIMITS['weak_residual']))
    for name, value, reference, weights, atol in (
            ('same_operator_H', H, Hr, dense.volumes, LIMITS['H_atol_Apm']),
            ('same_operator_B', B, Br, dense.volumes, LIMITS['B_atol_T']),
            ('same_operator_q', q[:, None], qr[:, None], dense.areas, LIMITS['q_atol_Apm'])):
        gates[name] = compare(value, reference, weights, atol, LIMITS['solution_rtol'])
    for k in range(len(spec['properties'])):
        keep = ids == k
        for name, value, reference, atol in (
                ('H', H, Hr, LIMITS['H_atol_Apm']), ('B', B, Br, LIMITS['B_atol_T'])):
            gates[f'material_{k}_{name}'] = compare(value[keep], reference[keep],
                dense.volumes[keep], atol, LIMITS['solution_rtol'])
    flux = []
    for k in range(dense.info['components']):
        keep = dense.component_ids[dense.faces[:, 0]] == k
        denominator = max(float(np.sum(dense.areas[keep]*abs(q[keep]))), 1e-30)
        flux.append(float(abs(dense.areas[keep] @ q[keep])/denominator))
    gates['component_net_flux'] = dict(values=flux, limit=LIMITS['component_net_flux'],
                                       passed=max(flux) <= LIMITS['component_net_flux'])
    record['gates'] = gates
    record['iteration_passed'] = all(g['passed'] for g in gates.values())
    if spec['shape'] == 'sphere' and spec['family'] != 'control':
        exact = sphere_internal_H(spec)
        expected = np.broadcast_to([0., 0., exact], H.shape)
        record['sphere_analytic_H'] = dict(internal_H_Apm=exact,
            relative_L2=rms(H-expected, dense.volumes)/abs(exact),
            scope='Diagnostic polyhedron-versus-sphere error, not an iteration gate')
    magnitudes = np.linalg.norm(H, axis=1)
    record['internal_H_range_Apm'] = [float(magnitudes.min()), float(magnitudes.max())]
    record['material_workpoints'] = []
    for k, prop in enumerate(spec['properties']):
        h = magnitudes[ids == k]
        entry = dict(material=k, kind=prop['kind'], H_range_Apm=[float(h.min()), float(h.max())])
        if prop['kind'] == 'nonlinear':
            knots = np.array(prop['bh_H_Apm_B_T'])[:, 0]
            branch = np.searchsorted(knots, h, side='right')-1
            entry['BH_interval_cell_counts'] = np.bincount(branch, minlength=len(knots)).tolist()
        record['material_workpoints'].append(entry)
    if 'chi' in spec:
        record['cells_at_or_above_knee'] = int(np.sum(magnitudes >= spec['Ms_Apm']/spec['chi']))
    record['phase'] = 'quadrature_reference'
    fine = timed('quadrature_reference_setup', lambda: build(args.reference_order))
    uf, qf, refinfo = timed('quadrature_reference', lambda: direct_reference(fine, H0, spec))
    Hf, Bf, *_ = fine.fields(uf, H0)
    record['quadrature_reference'] = refinfo
    record['quadrature_gates'] = dict(
        H=compare(Hr, Hf, dense.volumes, LIMITS['H_atol_Apm'], LIMITS['quadrature_rtol']),
        B=compare(Br, Bf, dense.volumes, LIMITS['B_atol_T'], LIMITS['quadrature_rtol']))
    for k in range(len(spec['properties'])):
        keep = ids == k
        for name, value, reference, atol in (
                ('H', Hr, Hf, LIMITS['H_atol_Apm']), ('B', Br, Bf, LIMITS['B_atol_T'])):
            record['quadrature_gates'][f'material_{k}_{name}'] = compare(value[keep], reference[keep],
                dense.volumes[keep], atol, LIMITS['quadrature_rtol'])
    record['quadrature_passed'] = all(g['passed'] for g in record['quadrature_gates'].values())
    record['state'] = 'passed' if record['iteration_passed'] and record['quadrature_passed'] else 'failed'
    record['failed_gates'] = [f'{group}.{key}' for group in ('gates', 'quadrature_gates')
                             for key, gate in record[group].items() if not gate['passed']]
    record['phase'] = 'completed'


def compare_backends(args, selected):
    """Paired current-source controls, not copies of historical result files."""
    report = dict(schema='fem-bem-resident-comparison-v1',
        started_utc=datetime.now(timezone.utc).isoformat(), command=[sys.executable,*sys.argv],
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Same-input/gate current-source comparison; one isolated process per backend; phase timings include telemetry',
        passed=False, runs=[])
    output = args.output.open('x',encoding='utf-8') if args.output else None
    def checkpoint():
        if output:
            output.seek(0); json.dump(report,output,indent=2,allow_nan=False)
            output.truncate(); output.flush()
    try:
        with tempfile.TemporaryDirectory(prefix='magtense-resident-comparison-') as temporary:
            shared = ['--level',str(args.level),'--order',str(args.order),
                '--reference-order',str(args.reference_order),'--maxiter',str(args.maxiter),
                '--linear-maxiter',str(args.linear_maxiter),'--preconditioner',args.preconditioner]
            for case in args.case or []:
                shared.extend(['--case',case])
            for name, boundary, fem, compute in [('cpu','dense','lu','numpy'),
                    ('hybrid','cuda_hmatrix','amg','numpy'),('resident','cuda_hmatrix','amg','cupy')]:
                report['phase'] = name
                checkpoint()
                path = Path(temporary)/(name+'.json')
                command = [sys.executable,str(Path(__file__).resolve()),*shared,
                    '--backend',boundary,'--fem-preconditioner',fem,'--compute-backend',compute,'--output',str(path)]
                result = subprocess.run(command,check=False)
                if not path.is_file():
                    raise RuntimeError(f'{name} worker produced no evidence: exit {result.returncode}')
                record = json.loads(path.read_text(encoding='utf-8'))
                report['runs'].append(dict(name=name,exit_code=result.returncode,report=record))
                checkpoint()
        baseline = report['runs'][0]['report']
        for run in report['runs']:
            data = run['report']
            if (data['limits'] != baseline['limits'] or data['source_sha256'] != baseline['source_sha256']
                    or data['runner_sha256'] != report['runner_sha256']
                    or len(data['cases']) != len(selected)):
                raise AssertionError('Backend comparison lost a case, input gate, or source identity')
            for before, after in zip(baseline['cases'],data['cases']):
                if (before['name'] != after['name'] or before.get('input') != after.get('input')
                        or before.get('geometry_sha256') != after.get('geometry_sha256')):
                    raise AssertionError('Backend comparison changed case geometry or materials')
        report['identical_inputs_and_limits'] = True
        report['passed'] = all(r['exit_code']==0 and r['report'].get('passed') for r in report['runs'])
        report['phase'] = 'completed'
    except Exception:
        report['failure'] = traceback.format_exc()
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        checkpoint()
        if output:
            output.close()
        else:
            print(json.dumps(report,indent=2,allow_nan=False))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--case', action='append', help='Exact name or family; repeat to combine (default: all)')
    parser.add_argument('--backend', choices=['dense', 'cuda_hmatrix'], default='dense')
    parser.add_argument('--preconditioner', choices=['block', 'none'], default='block')
    parser.add_argument('--fem-preconditioner', choices=['lu', 'amg'], default='lu')
    parser.add_argument('--compute-backend', choices=['numpy', 'cupy'], default='numpy')
    parser.add_argument('--compare-backends',action='store_true',help='Paired CPU, hybrid and resident workers; one combined report')
    parser.add_argument('--level', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--order', type=int, default=12)
    parser.add_argument('--reference-order', type=int, default=24)
    parser.add_argument('--maxiter', type=int, default=40)
    parser.add_argument('--linear-maxiter', type=int, default=30)
    parser.add_argument('--output', type=Path, help='New JSON path; existing files are never overwritten')
    args = parser.parse_args()
    if args.list:
        for spec in cases():
            print(f"{spec['name']:18s} {spec['family']}")
        return 0
    if not 2 <= args.order < args.reference_order <= 64 or min(args.maxiter, args.linear_maxiter) < 1:
        parser.error('Require 2 <= order < reference-order <= 64 and positive iteration limits')
    available = cases()
    if args.case:
        unknown = set(args.case)-{s['name'] for s in available}-{s['family'] for s in available}
        if unknown:
            parser.error(f'Unknown case/family: {sorted(unknown)}')
    selected = [s for s in available if not args.case or s['name'] in args.case or s['family'] in args.case]
    if args.compare_backends:
        return compare_backends(args, selected)
    report = dict(schema='fem-bem-convergence-v1', started_utc=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv], options={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  limits=LIMITS, scope='Synthetic small-model solver and quadrature benchmark; no Maxwell or large-GPU certification',
                  source_sha256={name: hashlib.sha256((TOOL/name).read_bytes()).hexdigest()
                                 for name in ('fem_bem.py', 'fem_bem_cuda.py', 'gpu_solver.py', 'boundary_hmatrix.py', 'tetra_cuda.py', 'tetra_cuda.cu')},
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=[])
    output = args.output.open('x', encoding='utf-8') if args.output else None
    def checkpoint():
        if output:
            output.seek(0)
            json.dump(report, output, indent=2, allow_nan=False, default=lambda v: v.item())
            output.truncate()
            output.flush()
    try:
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ.setdefault(name, '4')
        sys.path.insert(0, str(TOOL))
        import numpy as np
        import scipy
        report['environment'] = dict(python=sys.version, platform=platform.platform(),
                                     numpy=np.__version__, scipy=scipy.__version__,
                                     thread_environment={k: os.environ[k] for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
        if args.fem_preconditioner == 'amg' and args.compute_backend == 'numpy':
            import pyamg
            report['environment']['pyamg'] = pyamg.__version__
        if args.backend == 'cuda_hmatrix' or args.compute_backend == 'cupy':
            import cupy as cp
            device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            report['environment'].update(cupy=cp.__version__, GPU=device['name'].decode(),
                                         free_total_GPU_bytes=list(cp.cuda.Device().mem_info))
        checkpoint()
        for spec in selected:
            record = dict(name=spec['name'], state='running')
            report['cases'].append(record)
            checkpoint()
            started = time.perf_counter()
            try:
                run_case(spec, args, record, checkpoint)
            except Exception:
                phase = record.get('phase', 'initialization')
                kind = ('reference_unresolved' if phase in ('same_operator_reference', 'quadrature_reference')
                        else 'solver_failure' if phase == 'solve' else 'setup_or_verification_failure')
                record.update(state='failed', failure_kind=kind, failure=traceback.format_exc())
            record['elapsed_seconds'] = time.perf_counter()-started
            print(f"{record['name']}: {record['state']} ({record.get('phase', 'initialization')})", file=sys.stderr, flush=True)
            checkpoint()
        report['passed'] = len(report['cases']) == len(selected) and all(r['state'] == 'passed' for r in report['cases'])
    except Exception:
        report.update(passed=False, failure=traceback.format_exc())
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        checkpoint()
        if output:
            output.close()
        else:
            print(json.dumps(report, indent=2, allow_nan=False, default=lambda v: v.item()))
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
