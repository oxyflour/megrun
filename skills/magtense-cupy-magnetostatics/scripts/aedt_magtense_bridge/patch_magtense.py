#!/usr/bin/env python3
"""Patch an upstream MagTense checkout to accept distinct, ragged B-H tables.

Example:
    git clone https://github.com/cmt-dtu-energy/MagTense.git
    python patch_magtense.py --repo MagTense --build --fc ifx

The patch modifies the Fortran F2PY ABI. You MUST rebuild its native module.
No installation is attempted by default; see README_MULTI_BH.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


MARKER = 'MAGTENSE_MULTIBH_PATCH_V1 = True'
FORTRAN_ARRAY_OLD = 'real(8),dimension(nH,nT),intent(in) :: data_stateFcn'
FORTRAN_ARRAY_NEW = 'real(8),dimension(nH,nT,n_stateFcn),intent(in) :: data_stateFcn'

FORTRAN_LOADER_NEW = '''        do i=1,n_stateFcn
            stateFcn(i)%nT = nT-1
            ! Header column 1 is the number of H,M data rows. The legacy
            ! single-curve files use 0 in that header, meaning nH-1.
            stateFcn(i)%nH = nH-1
            if (data_stateFcn(1,1,i) > 0.0d0) then
                stateFcn(i)%nH = int(data_stateFcn(1,1,i))
            endif
            allocate( stateFcn(i)%M(stateFcn(i)%nT,stateFcn(i)%nH) )
            allocate( stateFcn(i)%T(stateFcn(i)%nT) )
            allocate( stateFcn(i)%H(stateFcn(i)%nH) )
            allocate( stateFcn(i)%y2a(stateFcn(i)%nH) )

            stateFcn(i)%T(1) = data_stateFcn(1,2,i)

            do j=2,stateFcn(i)%nH+1
                k = j - 1
                stateFcn(i)%H(k) = data_stateFcn(j,1,i)
                stateFcn(i)%M(1,k) = data_stateFcn(j,2,i)
            enddo
        enddo'''


def _exact_replace(data: str, old: str, new: str, count: int, file: Path) -> str:
    n = data.count(old)
    if n != count:
        raise RuntimeError(f'{file}: expected {count} instances of {old!r}, found {n}; '
                           'upstream sources changed. No files written.')
    return data.replace(old, new)


def patch_sources(repo: Path) -> dict[Path, str]:
    """Prepare modified file content; raise before writing if source is unexpected."""
    w = repo / 'python' / 'FortranToPythonIO.f90'
    f = repo / 'source' / 'DemagField' / 'IterateMagnetSolution.f90'
    p = repo / 'python' / 'src' / 'magtense' / 'magstatics.py'
    for x in (w, f, p):
        if not x.is_file():
            raise FileNotFoundError(f'MagTense source file missing: {x}')

    data = {x: x.read_text(encoding='utf-8') for x in (w, f, p)}
    if MARKER in data[p]:
        verify_sources(data)
        return {}

    data[w] = _exact_replace(data[w], FORTRAN_ARRAY_OLD, FORTRAN_ARRAY_NEW, 2, w)
    data[f] = _exact_replace(data[f], FORTRAN_ARRAY_OLD, FORTRAN_ARRAY_NEW, 1, f)
    # Change the loader only; leave physics equations and material lookup alone.
    start = data[f].find('    subroutine loadStateFunction(')
    if start < 0:
        raise RuntimeError('loadStateFunction not found in Fortran module')
    pos_a = data[f].find('        do i=1,n_stateFcn\n', start)
    pos_b = data[f].find('\n        enddo', pos_a)
    if pos_a < 0 or pos_b < 0:
        raise RuntimeError('Could not identify Fortran loader loop')
    pos_b += len('\n        enddo')
    old_block = data[f][pos_a:pos_b]
    if ('stateFcn(i)%H(k) = data_stateFcn(j,1)' not in old_block or
            'stateFcn(i)%M(1,k) = data_stateFcn(j,2)' not in old_block or
            'stateFcn(i)%nH = nH-1' not in old_block):
        raise RuntimeError('Unexpected loader structure; refusing to overwrite')
    data[f] = data[f][:pos_a] + FORTRAN_LOADER_NEW + data[f][pos_b:]

    py = data[p]
    py = _exact_replace(py, 'from magtense.lib import magtensesource',
                        'from magtense.lib import magtensesource\n\n' + MARKER, 1, p)
    # The stock high-level single-material API must still use the new ABI.
    py = _exact_replace(py, 'data_statefcn=data_stateFcn,',
                        'data_statefcn=np.asfortranarray(data_stateFcn[:, :, None]),', 2, p)
    data[p] = py
    verify_sources(data)
    return data


def verify_sources(data: dict[Path, str]) -> None:
    fs = {x.name: s for x, s in data.items()}
    io = fs['FortranToPythonIO.f90']
    core = fs['IterateMagnetSolution.f90']
    py = fs['magstatics.py']
    checks = {
        'Fortran wrapper 3D table': io.count(FORTRAN_ARRAY_NEW) == 2,
        'Fortran loader 3D table': core.count(FORTRAN_ARRAY_NEW) == 1,
        'Fortran per-material lookup': 'data_stateFcn(j,1,i)' in core and 'data_stateFcn(j,2,i)' in core,
        'Fortran ragged table count': 'int(data_stateFcn(1,1,i))' in core,
        'Python marker': MARKER in py,
        'Python packaged curve backward compatibility':
            py.count('data_statefcn=np.asfortranarray(data_stateFcn[:, :, None]),') == 2,
    }
    fail = [n for n, ok in checks.items() if not ok]
    if fail:
        raise RuntimeError('Invalid or partially patched MagTense: ' + ', '.join(fail))


def apply_patch(repo: Path, check_only: bool = False):
    changes = patch_sources(repo)
    if check_only:
        if changes:
            raise RuntimeError('Patch has not been applied yet')
        print('PASS: MagTense multi-BH source patch verified')
        return False
    if not changes:
        print('Already patched; source structure verified')
        return False
    # All modified texts validated before writing a single source file.
    for path, text in changes.items():
        path.write_text(text, encoding='utf-8')
        print('Patched', path)
    print('IMPORTANT: Recompile the MagTense Fortran extension (ABI changed).')
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=True, type=Path, help='Root of the MagTense Git clone')
    ap.add_argument('--check', action='store_true', help='Verify that patch is already applied')
    ap.add_argument('--build', action='store_true', help='Run upstream make python after patch')
    ap.add_argument('--fc', choices=['ifx','gfortran'],
                    default='ifx' if sys.platform == 'win32' else 'gfortran')
    ap.add_argument('--make', default='make', help='Make executable')
    args = ap.parse_args(argv)
    repo = args.repo.expanduser().resolve()
    apply_patch(repo, args.check)
    if args.build:
        cmd = [args.make, 'python', f'FC={args.fc}', 'USE_CUDA=0', 'USE_CVODE=0',
               'USE_MICROMAG=0', 'USE_MATLAB=0', 'USE_FMM3D=0',
               f'USE_MKL={1 if args.fc == "ifx" else 0}']
        print('Building patched native extension:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=repo, check=True)
        print('Built native module. Install from cloned ./python directory and run native smoke test.')


if __name__ == '__main__':
    main()
