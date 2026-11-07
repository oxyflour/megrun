"""Multi-nonlinear-material offline contract tests (no AEDT or MagTense native required)."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from magtense_adapter import (
    MU0, UnsupportedPhysics, assemble_tiles, solve, state_functions_from_bh,
)
from patch_magtense import (
    FORTRAN_ARRAY_NEW, FORTRAN_LOADER_NEW, MARKER, apply_patch,
)

CURVE_A = np.array([[0, 0], [75, .12], [180, .37]], dtype=float)
CURVE_B = np.array([[0, 0], [50, .035], [110, .09], [290, .15], [1000, .5]], dtype=float)


def create_obj(folder: Path, name: str, curve=None, kind='nonlinear') -> dict:
    mesh = folder / (name + '.npz')
    verts = np.array([[[0., 0., 0.], [0.001, 0., 0.],
                       [0., 0.001, 0.], [0., 0., 0.001]]])
    np.savez_compressed(mesh, tetra_m=verts)
    if kind == 'nonlinear':
        props = {'kind': 'nonlinear', 'bh_H_Apm_B_T': curve.tolist()}
    elif kind == 'permanent':
        props = {'kind': 'permanent', 'mu_r': 1.05,
                 'br_T': 1.1, 'direction': [0, 0, 1]}
    else:
        props = {'kind': 'linear', 'mu_r': 75}
    return {'name': name, 'mesh_npz': mesh.name,
            'material': name, 'properties': props}


class FakeTiles:
    def __init__(self, **kw):
        self.n = kw['n']
        self.vertices = kw['vertices']
        self.M_rem = np.asarray(kw['M_rem'])
        self.u_ea = np.asarray(kw['easy_axis'])
        self.mu_r_ea = np.asarray(kw['mu_r_ea'])
        self.mu_r_oa = np.asarray(kw['mu_r_oa'])
        self.tile_type = np.full(self.n, 5)
        self.M = self.M_rem[:, None] * self.u_ea
        self.center_pos = np.zeros((self.n, 3))
        self.dev_center = np.zeros((self.n, 3))
        self.size = np.zeros((self.n, 3))
        self.u_oa1 = np.zeros((self.n, 3))
        self.u_oa2 = np.zeros((self.n, 3))
        self.offset = np.zeros((self.n, 3))
        self.rot = np.zeros((self.n, 3))
        self.color = np.zeros((self.n, 3))
        self.M_rel = np.zeros(self.n)
        self.use_sym = np.zeros(self.n, dtype=np.int32)
        self.sym_op = np.ones((self.n, 3))


def fake_magtense(monkeypatch, patched=True, native_doc=''):
    calls = []
    def native_run(**kw):
        calls.append(kw)
        n = len(kw['mrem'])
        return (np.tile([100., 0., 0.], (len(kw['pts']), 1)),
                np.ones((n, 3)), np.full(n, 1e-7))
    native_run.__doc__ = native_doc
    native = types.SimpleNamespace(runsimulation=native_run)
    mag = types.ModuleType('magtense.magstatics')
    mag.Tiles = FakeTiles
    mag.run_simulation = lambda *_a, **_kw: (_ for _ in ()).throw(
        AssertionError('custom B-H should use native solver directly'))
    if patched:
        mag.MAGTENSE_MULTIBH_PATCH_V1 = True
    lib = types.ModuleType('magtense.lib')
    lib.magtensesource = types.SimpleNamespace(fortrantopythonio=native)
    pkg = types.ModuleType('magtense')
    pkg.__path__ = []
    pkg.magstatics, pkg.lib = mag, lib
    for name, mod in [('magtense', pkg), ('magtense.magstatics', mag), ('magtense.lib', lib)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_ragged_bh_tables_preserve_every_original_knots():
    x = state_functions_from_bh([CURVE_A, CURVE_B], temperature_K=298.15)
    assert x.shape == (len(CURVE_B) + 1, 2, 2)
    assert x.flags.f_contiguous
    assert x[0, 0, :].tolist() == [len(CURVE_A), len(CURVE_B)]
    assert x[0, 1, :].tolist() == [298.15, 298.15]
    np.testing.assert_allclose(x[1:4, 0, 0], CURVE_A[:, 0])
    np.testing.assert_allclose(x[1:4, 1, 0], CURVE_A[:, 1] / MU0 - CURVE_A[:, 0])
    np.testing.assert_allclose(x[1:6, 0, 1], CURVE_B[:, 0])
    np.testing.assert_allclose(x[1:6, 1, 1], CURVE_B[:, 1] / MU0 - CURVE_B[:, 0])
    assert np.all(x[4:, :, 0] == 0.0)  # padding ignored by Fortran loader


def test_multimaterial_deduplicate_indices_and_native_arguments(tmp_path, monkeypatch):
    calls = fake_magtense(monkeypatch, patched=True)
    objs = [create_obj(tmp_path, 'ShieldA', CURVE_A),
            create_obj(tmp_path, 'ShieldB', CURVE_B),
            create_obj(tmp_path, 'ShieldC', CURVE_A),
            create_obj(tmp_path, 'NdFeB', kind='permanent')]
    tiles, curves, meta = assemble_tiles(objs, tmp_path, max_tiles=100)
    assert meta['tiles'] == 4
    assert meta['nonlinear_state_functions'] == 2
    np.testing.assert_array_equal(tiles.stfcn_index, [1, 2, 1, 1])
    np.testing.assert_array_equal(tiles.magnet_type, [2, 2, 2, 1])
    assert len(curves) == 2
    pts = np.array([[.008, 0, 0]])
    B, diagnostics = solve(tiles, pts, curves, temperature_K=298.15)
    assert len(calls) == 1
    kw = calls[0]
    assert kw['n_statefcn'] == 2
    np.testing.assert_array_equal(kw['statefunctionindex'], [1, 2, 1, 1])
    assert kw['data_statefcn'].ndim == 3
    assert kw['data_statefcn'].shape == (6, 2, 2)
    assert kw['data_statefcn'].flags.f_contiguous
    np.testing.assert_allclose(B, [[100 * MU0, 0, 0]])
    assert diagnostics['n_nonlinear_bh_curves'] == 2
    assert diagnostics['original_bh_point_counts'] == [3, 5]


def test_unpatched_magtense_fails_loudly_for_two_curves(tmp_path, monkeypatch):
    calls = fake_magtense(monkeypatch, patched=False)
    objs = [create_obj(tmp_path, 'Soft1', CURVE_A),
            create_obj(tmp_path, 'Soft2', CURVE_B)]
    tiles, curves, _ = assemble_tiles(objs, tmp_path, max_tiles=10)
    with pytest.raises(UnsupportedPhysics, match='patch is not installed'):
        solve(tiles, np.array([[.005, 0, 0]]), curves)
    assert calls == []


def test_detect_old_native_rank_2_with_new_python(monkeypatch):
    calls = fake_magtense(monkeypatch, patched=True,
                          native_doc='data_statefcn : input rank-2 array')
    tiles = FakeTiles(n=2, vertices=np.zeros((2, 3, 4)), M_rem=[0., 0.],
                      easy_axis=[[1.,0.,0.], [1.,0.,0.]],
                      mu_r_ea=[1.,1.], mu_r_oa=[1.,1.])
    tiles.magnet_type = np.array([2, 2])
    tiles.stfcn_index = np.array([1, 2])
    tiles.incl_it = np.ones(2, int)
    with pytest.raises(UnsupportedPhysics, match='old rank-2 binary'):
        solve(tiles, np.zeros((1, 3)), [CURVE_A, CURVE_B])
    assert calls == []


def test_legacy_single_curve_still_uses_rank_2(monkeypatch):
    calls = fake_magtense(monkeypatch, patched=False)
    tiles = FakeTiles(n=1, vertices=np.zeros((1,3,4)), M_rem=[0],
                      easy_axis=[[1.,0.,0.]],mu_r_ea=[1.],mu_r_oa=[1.])
    tiles.magnet_type = np.array([2])
    tiles.stfcn_index = np.array([1])
    tiles.incl_it = np.ones(1, int)
    solve(tiles, np.zeros((1,3)), [CURVE_A])
    assert calls[0]['data_statefcn'].shape == (4, 2)
    assert calls[0]['n_statefcn'] == 1


def synthetic_source_checkout(tmp_path):
    """Small but structurally faithful snippets for upstream patch-script regression."""
    repo = tmp_path / 'MagTense'
    (repo / 'source' / 'DemagField').mkdir(parents=True)
    (repo / 'python' / 'src' / 'magtense').mkdir(parents=True)
    (repo / 'python' / 'FortranToPythonIO.f90').write_text(
        ('subroutine runSimulation\n    real(8),dimension(nH,nT),intent(in) :: data_stateFcn\nend\n'
         'subroutine IterateTiles\n    real(8),dimension(nH,nT),intent(in) :: data_stateFcn\nend\n'))
    (repo / 'source' / 'DemagField' / 'IterateMagnetSolution.f90').write_text('''
    subroutine loadStateFunction( nT, nH, stateFcn, data_stateFcn, n_stateFcn )
        real(8),dimension(nH,nT),intent(in) :: data_stateFcn
        do i=1,n_stateFcn
            stateFcn(i)%nT = nT-1
            stateFcn(i)%nH = nH-1
            allocate( stateFcn(i)%M(stateFcn(i)%nT,stateFcn(i)%nH) )
            allocate( stateFcn(i)%T(stateFcn(i)%nT) )
            allocate( stateFcn(i)%H(stateFcn(i)%nH) )
            allocate( stateFcn(i)%y2a(stateFcn(i)%nH) )
            stateFcn(i)%T(1) = data_stateFcn(1,2)
            do j=2,nH
                k = j - 1
                stateFcn(i)%H(k) = data_stateFcn(j,1)
                stateFcn(i)%M(1,k) = data_stateFcn(j,2)
            enddo
        enddo
    end subroutine loadStateFunction
''')
    (repo / 'python' / 'src' / 'magtense' / 'magstatics.py').write_text('''
import numpy as np
from magtense.lib import magtensesource

def run_simulation():
    x = f(data_statefcn=data_stateFcn, n_statefcn=1)

def iterate_magnetization():
    x = f(data_statefcn=data_stateFcn, n_statefcn=1)
''')
    return repo


def test_patcher_changes_all_three_files_and_is_idempotent(tmp_path):
    repo = synthetic_source_checkout(tmp_path)
    assert apply_patch(repo) is True
    assert apply_patch(repo, check_only=True) is False
    assert apply_patch(repo) is False
    f = (repo / 'source' / 'DemagField' / 'IterateMagnetSolution.f90').read_text()
    w = (repo / 'python' / 'FortranToPythonIO.f90').read_text()
    p = (repo / 'python' / 'src' / 'magtense' / 'magstatics.py').read_text()
    assert FORTRAN_ARRAY_NEW in f and w.count(FORTRAN_ARRAY_NEW) == 2
    assert 'data_stateFcn(j,1,i)' in f and 'data_stateFcn(j,2,i)' in f
    assert MARKER in p


def test_patcher_refuses_unexpected_upstream_without_writes(tmp_path):
    repo = synthetic_source_checkout(tmp_path)
    path = repo / 'python' / 'FortranToPythonIO.f90'
    original = path.read_text()
    path.write_text(original.replace('dimension(nH,nT)', 'dimension(nH,nT+1)', 1))
    f = repo / 'source' / 'DemagField' / 'IterateMagnetSolution.f90'
    before = f.read_text()
    with pytest.raises(RuntimeError, match='upstream sources changed'):
        apply_patch(repo)
    assert f.read_text() == before


@pytest.mark.skipif(shutil.which('gfortran') is None, reason='gfortran not installed')
def test_fortran_loader_compiles_and_reads_two_distinct_ragged_curves(tmp_path):
    """Compile and execute the exact patched Fortran loader logic in a minimal module."""
    src = tmp_path / 'loader_smoke.f90'
    program = '''
module loader_test
  implicit none
  type MagStateFunction
     integer :: nT, nH
     real(8), allocatable :: M(:,:), T(:), H(:), y2a(:)
  end type
contains
  subroutine loadStateFunction(nT,nH,stateFcn,data_stateFcn,n_stateFcn)
    integer,intent(in) :: nT,nH,n_stateFcn
    type(MagStateFunction),dimension(n_stateFcn),intent(inout) :: stateFcn
    real(8),dimension(nH,nT,n_stateFcn),intent(in) :: data_stateFcn
    integer :: i,j,k
''' + FORTRAN_LOADER_NEW + '''
  end subroutine
end module
program check
 use loader_test
 implicit none
 real(8) :: a(6,2,2)
 type(MagStateFunction) :: m(2)
 a=0.0d0
 a(1,1,1)=3
 a(1,2,1)=300
 a(2,1,1)=0
 a(3,1,1)=75
 a(4,1,1)=180
 a(2,2,1)=0
 a(3,2,1)=1000
 a(4,2,1)=2000
 a(1,1,2)=5
 a(1,2,2)=300
 a(2:6,1,2)=[0.d0,50.d0,110.d0,290.d0,1000.d0]
 a(2:6,2,2)=[0.d0,100.d0,200.d0,300.d0,400.d0]
 call loadStateFunction(2,6,m,a,2)
 if (m(1)%nH/=3 .or. m(2)%nH/=5) stop 1
 if (m(1)%H(3)/=180.d0 .or. m(2)%H(5)/=1000.d0) stop 2
 if (m(1)%M(1,2)/=1000.d0 .or. m(2)%M(1,2)/=100.d0) stop 3
 ! Legacy packaged Fe files set the first header field to zero, no count.
 a(1,1,1)=0.d0
 a(5:6,1,1)=[300.d0,400.d0]
 a(5:6,2,1)=[3000.d0,4000.d0]
 deallocate(m(1)%M, m(1)%T, m(1)%H, m(1)%y2a)
 call loadStateFunction(2,6,m(1:1),a(:,:,1:1),1)
 if (m(1)%nH/=5 .or. m(1)%H(5)/=400.d0) stop 4
end program
'''
    src.write_text(program)
    binary = tmp_path / 'loader_smoke'
    subprocess.run(['gfortran', '-ffree-line-length-none', str(src), '-o', str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_full_cli_solve_writes_multibh_material_mapping(tmp_path, monkeypatch):
    from bridge import main
    calls = fake_magtense(monkeypatch, patched=True)
    objs = [create_obj(tmp_path, 'Shield_FeA', CURVE_A),
            create_obj(tmp_path, 'Shield_FeB', CURVE_B)]
    points = np.array([[0.004, 0.001, 0.], [0.005, 0.002, 0.]])
    np.savetxt(tmp_path / 'points.csv', points, delimiter=',',
               header='x_m,y_m,z_m', comments='')
    model = {
        'schema_version': 1, 'source': {'project': 'synthetic'},
        'points': 'points.csv', 'total_magnetic_tetra': 2,
        'objects': objs, 'skipped': [], 'maxwell_field': 'unused.fld',
    }
    import json
    (tmp_path / 'model.json').write_text(json.dumps(model))
    assert main(['solve', '--out', str(tmp_path), '--max-tiles', '10']) == 0
    assert len(calls) == 1
    info = json.loads((tmp_path / 'solve_info.json').read_text())
    assert info['n_nonlinear_bh_curves'] == 2
    mapping = info['tile_stats']['nonlinear_material_assignments']
    assert [(x['material'], x['state_function_index']) for x in mapping] == [
        ('Shield_FeA', 1), ('Shield_FeB', 2)]
    field = np.loadtxt(tmp_path / 'magtense_B.csv', delimiter=',', skiprows=1)
    assert field.shape == (2, 6)
    np.testing.assert_allclose(field[:, 3], 100 * MU0)
