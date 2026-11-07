"""Independent cube-average benchmark and exact tetrahedron moments."""
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from magtense.magstatics import Tiles, get_demag_tensor
from average_pilot import red_quadrature as quadrature

ROOT = Path(__file__).resolve().parent


def test_tetrahedron_moments():
    tetra = np.array([[[0., 0, 0], [1., 0, 0], [0, 1., 0], [0, 0, 1.]]])
    for level in range(4):
        q = quadrature(tetra, level)[0]
        np.testing.assert_allclose(q.mean(0), [.25]*3, atol=1e-14)
        np.testing.assert_allclose((q*q).mean(0), [.1]*3, atol=1e-14)
        np.testing.assert_allclose((q[:, 0]*q[:, 1]).mean(), .05, atol=1e-14)
        assert np.all(q > 0) and np.all(q.sum(1) < 1)


def test_against_analytic_averaged_prism():
    side = .001
    vertices = []
    for permutation in permutations(range(3)):
        a = np.eye(3)[permutation[0]] * side
        b = (np.eye(3)[permutation[0]] + np.eye(3)[permutation[1]]) * side
        vertices.append([[0, 0, 0], a, b, [side]*3])
    vertices = np.asarray(vertices)
    sources = Tiles(n=6, vertices=np.asfortranarray(vertices.transpose(0, 2, 1)),
                    tile_type=5, M_rem=0, easy_axis=[1, 0, 0], mu_r_ea=1, mu_r_oa=1)
    prism = Tiles(n=1, tile_type=8, size=[side]*3, offset=[side/2]*3,
                  M_rem=0, easy_axis=[1, 0, 0], mu_r_ea=1, mu_r_oa=1)
    results = {}
    for label, shift in [('self', [0, 0, 0]), ('touching', [side, 0, 0]), ('separated', [2*side, 0, 0])]:
        reference = get_demag_tensor(prism, np.array([[side/2]*3])+shift, obs_size=np.array([[side]*3]))[0, 0]
        errors = []
        for level in range(4):
            q = quadrature(vertices+shift, level)
            matrix = get_demag_tensor(sources, q.reshape(-1, 3)).sum(axis=0).mean(axis=0)
            errors.append(float(np.linalg.norm(matrix-reference)/np.linalg.norm(reference)))
        results[label] = {'relative_errors_q4_q16_q64_q256': errors, 'reference': reference.tolist()}
    (ROOT / 'quadrature_red_cube_validation.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results))
    for label, result in results.items():
        assert result['relative_errors_q4_q16_q64_q256'][-1] < 5e-4, label
