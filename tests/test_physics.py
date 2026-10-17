import numpy as np
import pytest
from camera_mag.interaction import dipole_operator, dipole_field, dipole_pair_force, volume_operator, MU0
from camera_mag.materials import LinearMaterial, SoftMagneticMaterial
from camera_mag.coil import Winding, lorentz_force, make_winding, winding_field
from camera_mag.geometry import Camera
from camera_mag.solver import CameraPairSolver


def tiny_camera(mu_r=1.1):
    return Camera(np.array([[0., 0., 0.]]), np.array([1e-9]), np.array(["magnet"]), np.array([0]),
                  np.array([[0., 0., 1e5]]), np.array([.0005]),
                  {"bounds_m": [[-.0005]*3, [.0005]*3], "coil": {"center_m": [0, 0],
                   "inner_half_m": [.002, .002], "outer_half_m": [.0022, .0022], "z_m": [-.0001, .0001]}})


def test_single_dipole_analytical():
    # Axial and equatorial fields of a z dipole, from independent closed forms.
    moment, distance, volume = 2e-5, .01, 1e-9
    h = dipole_field(np.array([[0, 0, distance], [distance, 0, 0]]), np.zeros((1, 3)),
                     np.array([volume]), np.array([[0, 0, moment / volume]]))
    expected = np.array([[0, 0, moment / (2 * np.pi * distance**3)],
                         [0, 0, -moment / (4 * np.pi * distance**3)]])
    np.testing.assert_allclose(h, expected, atol=1e-14)


def test_dipole_force_direction_symmetry_and_inverse_fourth():
    moment = np.array([0., 0., 1.])
    r = np.array([0., 0., 2.])
    force = dipole_pair_force(r, moment, moment)
    assert force[2] < 0  # aligned axial moments attract
    np.testing.assert_allclose(dipole_pair_force(-r, moment, moment), -force)
    np.testing.assert_allclose(dipole_pair_force(2 * r, moment, moment), force / 16)
    assert dipole_pair_force(np.array([2., 0, 0]), moment, moment)[0] > 0


def test_spherical_self_term_and_volume_reciprocity():
    centers = np.array([[0, 0, 0], [.001, .002, .003]])
    volumes = np.array([1e-10, 2e-10])
    g = dipole_operator(centers, centers, volumes, self_interaction=True).reshape(2, 3, 2, 3)
    np.testing.assert_allclose(g[0, :, 0, :], -np.eye(3) / 3)
    np.testing.assert_allclose(g[0, :, 1, :] * volumes[0], g[1, :, 0, :].T * volumes[1])


def test_volume_regularization_energy_self_and_far_field():
    # Close high-permeability cells must not develop an unphysical positive-energy mode.
    centers = np.array([[0, 0, 0], [.00001, 0, 0], [.0002, .0002, 0]])
    volumes = np.array([1e-10, 2e-10, 1.5e-10])
    g = volume_operator(centers, centers, volumes, target_volumes=volumes)
    weights = np.repeat(np.sqrt(volumes), 3)
    symmetric = g * weights[:, None] / weights[None, :]
    np.testing.assert_allclose(symmetric, symmetric.T, atol=1e-15)
    assert np.linalg.eigvalsh(symmetric).max() <= 1e-12
    np.testing.assert_allclose(g[:3, :3], -np.eye(3)/3, atol=1e-15)
    far = np.array([[.01, .02, .03]])
    np.testing.assert_allclose(volume_operator(far, centers, volumes), dipole_operator(far, centers, volumes), rtol=1e-12)


@pytest.mark.parametrize("mu_r", [.95, 1., 1.2])
def test_linear_material_response_and_demag_solution(mu_r):
    mat = LinearMaterial(mu_r)
    external = np.array([[30., -20, 50]])
    remanence = np.array([[0., 0, 1e5]])
    m = mat.local_response(external, remanence)
    total_h = external - m / 3
    np.testing.assert_allclose(m, remanence + mat.response(total_h), rtol=1e-12, atol=1e-10)


def test_nonlinear_bh_curve_and_local_inverse():
    h = np.array([0, 100, 500, 2000, 20000.])
    b = np.array([0, .5, 1.2, 1.5, 1.68])
    mat = SoftMagneticMaterial(h, b)
    vector_h = np.c_[h, np.zeros((len(h), 2))]
    np.testing.assert_allclose(MU0 * (vector_h + mat.response(vector_h))[:, 0], b)
    external = vector_h + mat.response(vector_h) / 3
    np.testing.assert_allclose(mat.local_response(external), mat.response(vector_h), rtol=1e-8, atol=1e-7)
    high_h = np.array([[1e7, 0, 0]])
    assert mat.b_value(np.array([1e7]))[0] == pytest.approx(b[-1] + MU0 * (1e7 - h[-1]))
    assert np.isfinite(mat.local_response(high_h)).all()


def test_linear_fixed_point_converges_to_spherical_analytical(tmp_path):
    camera = tiny_camera()
    solver = CameraPairSolver(camera, camera, {"magnet": LinearMaterial(1.1)}, current_a=0,
                               cache_dir=tmp_path, method="fixed", omega=.5, tol=1e-11)
    state = solver.isolated_states()[0]
    np.testing.assert_allclose(state, camera.remanence / (1 + .1 / 3), rtol=1e-9, atol=1e-6)


def test_cache_unchanged_distance_and_warm_state(tmp_path):
    camera = tiny_camera()
    materials = {"magnet": LinearMaterial(1.1)}
    solver = CameraPairSolver(camera, camera, materials, current_a=0, cache_dir=tmp_path, tol=1e-10)
    solver.precompute_self_operators()
    before = solver.self_a.copy()
    first = solver.solve(1)
    cold = solver.solve(2)
    warm = solver.solve(2, first.state)
    assert cold.converged and warm.converged
    assert solver.self_build_calls == 1
    assert solver.self_b is solver.self_a
    np.testing.assert_array_equal(solver.self_a, before)
    np.testing.assert_allclose(warm.state, cold.state, rtol=1e-8, atol=1e-5)
    again = CameraPairSolver(camera, camera, materials, current_a=0, cache_dir=tmp_path)
    assert again.precompute_self_operators()["A"]["cache_hit"]


def test_closed_loop_uniform_field_has_zero_net_force():
    winding = make_winding(tiny_camera())
    np.testing.assert_allclose(winding.dl.sum(axis=0), 0, atol=1e-15)
    force = lorentz_force(winding, np.tile([.2, -.3, .7], (len(winding.points), 1)))
    np.testing.assert_allclose(force, 0, atol=1e-14)


def test_square_loop_center_field_and_current_polarity():
    # Independent analytical field at center of a square loop.
    a, current = .002, .1
    points, dl = [], []
    corners = np.array([[-a,-a,0],[a,-a,0],[a,a,0],[-a,a,0]])
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        points.extend(start + ((np.arange(200)+.5)/200)[:,None] * (end-start))
        dl.extend(np.tile((end-start)/200, (200,1)))
    winding = Winding(np.array(points), np.array(dl), np.full(len(points), current), 0)
    field = winding_field(np.zeros((1,3)), winding)[0]
    np.testing.assert_allclose(field, [0,0,np.sqrt(2)*current/(np.pi*a)], rtol=1e-5)
    negative = Winding(winding.points, winding.dl, -winding.ampere_turns, 0)
    np.testing.assert_allclose(winding_field(np.zeros((1,3)), negative)[0], -field)


def test_lorentz_force_in_radial_field():
    # A counterclockwise square loop in inward in-plane B has +Z axial force.
    winding = make_winding(tiny_camera(), current_a=.1, turns=100, cross_samples=1)
    b = -winding.points.copy()
    b[:, 2] = 0
    force = lorentz_force(winding, b)
    assert force[2] > 0
    np.testing.assert_allclose(force[:2], 0, atol=1e-15)
