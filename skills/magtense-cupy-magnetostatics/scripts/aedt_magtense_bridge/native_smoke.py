#!/usr/bin/env python3
"""Native integration check for the *rebuilt* MagTense multi-BH Fortran extension.

No Maxwell installation needed. The two soft tetrahedra have different B-H
curves, driven by a uniform H field. Swapping stateFunctionIndex must swap
which physical tetrahedron responds more strongly.
"""
from __future__ import annotations

import numpy as np

from magtense_adapter import MU0, solve


def verify_native_multibh():
    import magtense.magstatics as ms
    if not getattr(ms, 'MAGTENSE_MULTIBH_PATCH_V1', False):
        raise RuntimeError('Patched MagTense not imported. See README_MULTI_BH.md.')

    # Synthetic curves, deliberately very different; not real shield data.
    def bh_with_mu_r(mu_r):
        h = np.array([0., 100., 500., 1500., 5000., 10000.])
        m = (mu_r - 1.) * np.minimum(h, 3000.)
        return np.column_stack((h, MU0 * (h + m)))

    curves = [bh_with_mu_r(2.), bh_with_mu_r(150.)]
    base = np.array([[0., 0., 0.], [0.001, 0., 0.],
                     [0., 0.001, 0.], [0., 0., 0.001]])
    # Separation >> tile size: small mutual coupling relative to own demag field.
    vertices = np.stack((base + [-.02, 0., 0.], base + [.02, 0., 0.]))
    vertices = np.transpose(vertices, (0, 2, 1))

    results = []
    for ids in ([1, 2], [2, 1]):
        tiles = ms.Tiles(n=2, vertices=np.asfortranarray(vertices), tile_type=5,
                         M_rem=0., easy_axis=[1., 0., 0.],
                         mu_r_ea=1., mu_r_oa=1., magnet_type=2)
        tiles.stfcn_index = np.asarray(ids, dtype=np.int32)
        # Ensure this is the patched MagTense >=3.0.1, supporting uniform field.
        tiles.add_uniform_field([500., 0., 0.])
        _, diag = solve(tiles, np.array([[0., .01, .01]]), curves,
                        max_error=1e-6, max_it=500)
        m = np.linalg.norm(tiles.M[:2], axis=1)
        results.append(m)
        print(f'indices={ids}, |M|=[{m[0]:.3f}, {m[1]:.3f}] A/m, '
              f'convergence={diag["max_final_M_rel"]:.3g}')
        if not np.isfinite(m).all():
            raise RuntimeError('Native solver produced non-finite magnetization')

    # Material 2 (mu_r ~150 near origin) must magnetize more strongly than material 1.
    first, swapped = results
    if not (first[1] > 1.25 * first[0] and swapped[0] > 1.25 * swapped[1]):
        raise RuntimeError('Native multi-material behavior failed: B-H response does '
                           'not follow stateFunctionIndex. Check native .pyd/.so rebuild.')
    print('PASS: Two distinct B-H curves changed the native solution independently.')


if __name__ == '__main__':
    verify_native_multibh()
