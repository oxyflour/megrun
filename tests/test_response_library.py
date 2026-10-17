import numpy as np
import pytest
from camera_mag.response_library import coil_quadrature, integrate_force, read_vector_field, force_response_operator, native_susceptibility


SPEC = dict(inner_half_m=[.002825, .002825], outer_half_m=[.003125, .003125],
            center_m=[0, 0], z_m=[.0017, .004])


def test_closed_winding_has_zero_force_in_uniform_field_and_correct_gradient_force():
    points, weights, ideal = coil_quadrature(SPEC, 12, 3, 4)
    ni = 10
    uniform = np.tile([.2, -.1, .3], (len(points), 1))
    assert np.linalg.norm(integrate_force(ni*ideal, uniform, weights)) < 1e-15
    # Divergence-free B=(g*x,0,-g*z) yields Fz=-NI*g*mean enclosed area.
    g = 2.0
    field = np.c_[g*points[:, 0], np.zeros(len(points)), -g*points[:, 2]]
    a, b = SPEC["inner_half_m"][0], SPEC["outer_half_m"][0]
    expected = -ni*g*4*(a*a+a*b+b*b)/3
    force = integrate_force(ni*ideal, field, weights)
    assert force[2] == pytest.approx(expected, rel=1e-12)
    assert np.linalg.norm(force[:2]) < 1e-15


def test_native_field_requires_matching_si_positions(tmp_path):
    path = tmp_path / "field.fld"
    path.write_text("header\n0.001 0 0 1 2 3\n")
    assert np.array_equal(read_vector_field(path, [[.001, 0, 0]]), [[1, 2, 3]])
    with pytest.raises(ValueError, match="SI positions"):
        read_vector_field(path, [[1, 0, 0]])


def test_incremental_force_operator_matches_far_axial_dipole_attraction():
    # Independent small-loop/far-dipole formula, including the winding height.
    from camera_mag.interaction import MU0
    points, weights, ideal = coil_quadrature(SPEC, 6, 3, 4)
    ni, moment, zsource, volume = 10, 1e-3, -.5, 1e-9
    operator = force_response_operator(points, ni*ideal, weights,
                                      np.array([[0, 0, zsource]]), np.array([volume]))
    actual = operator @ np.array([0, 0, moment/volume])
    a, b = SPEC["inner_half_m"][0], SPEC["outer_half_m"][0]
    area = 4*(a*a+a*b+b*b)/3
    lower, upper = np.array(SPEC["z_m"])-zsource
    height_average = (lower**-3-upper**-3)/(3*(upper-lower))
    expected_z = -3*MU0*moment*ni*area*height_average/(2*np.pi)
    assert actual[2] == pytest.approx(expected_z, rel=1e-3)
    assert np.linalg.norm(actual[:2]) < abs(actual[2])*1e-10


def test_native_tangent_matches_vector_finite_difference():
    from camera_mag.interaction import MU0
    table = np.array([[0,0],[100,.3],[1000,1.]])
    h = np.array([300.,400,0])
    def response(vector):
        magnitude = np.linalg.norm(vector)
        return vector*(np.interp(magnitude,table[:,0],table[:,1])/(MU0*magnitude)-1)
    numerical = np.column_stack([(response(h+np.eye(3)[k]*1e-3)-response(h-np.eye(3)[k]*1e-3))/(2e-3) for k in range(3)])
    tangent = native_susceptibility(h[None], ["yoke"], table)[0]
    np.testing.assert_allclose(tangent,numerical,rtol=1e-8,atol=1e-5)
