"""CPU FEM--BEM regression: source integrals, tangent, gauge and block solver.

Run with python -m pytest -q tests/test_fem_bem.py. Analytical panel integrals,
independent source quadrature and finite differences provide numerical oracles.
Sphere/Maxwell field acceptance lives in the existing run_fmm.py validation
entrypoint, not in a second implementation of this solver.
"""
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fem_bem import FEMBEM, material_laws, triangle_rule
from gpu_solver import NonConvergence


def cube(shift=(0., 0, 0)):
    return np.array([[np.zeros(3), np.eye(3)[p[0]],
                      np.eye(3)[p[0]]+np.eye(3)[p[1]], np.ones(3)]
                     for p in permutations(range(3))]) + shift


def test_source_integrals_against_independent_quadrature_and_exact_self():
    tet = np.array([[[0., 0, 0], [1, 0, 0], [.5, np.sqrt(3)/2, 0], [.5, .3, 1]]])
    s = FEMBEM(tet, material_laws([dict(kind='linear', mu_r=1)], [0]))
    face = np.flatnonzero(np.all(s.tri[:, :, 2] == 0, axis=1))[0]
    S, D = s.layers(s.tri[face].mean(axis=0)[None])
    # Equilateral unit triangle, observation at its centroid: the radial
    # integral reduces to 6*inradius*asinh(sqrt(3)). D has zero planar PV.
    assert S[0, face] == pytest.approx(np.sqrt(3)*np.arcsinh(np.sqrt(3))/(4*np.pi), rel=2e-14)
    np.testing.assert_array_equal(D[0, face], 0)
    points = np.array([[.2, -.8, .4], [1.7, 1.5, -.8]])
    S, D = s.layers(points)
    bary, w = triangle_rule(36)
    source = np.einsum('qj,sji->sqi', bary, s.tri)
    delta = points[:, None, None, :] - source[None]
    r = np.linalg.norm(delta, axis=3)
    expected_S = np.einsum('q,psq,s->ps', w, 1/r, s.areas)/(4*np.pi)
    kernel_D = np.einsum('psqi,si->psq', delta, s.normals)/r**3
    expected_D = np.einsum('q,psq,qj,s->psj', w, kernel_D, bary, s.areas)/(4*np.pi)
    np.testing.assert_allclose(S, expected_S, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(D, expected_D, rtol=3e-11, atol=2e-14)


def test_material_tangent_and_fem_nullspace():
    properties = [dict(kind='permanent', mu_r=1.05, br_T=1.2),
                  dict(kind='linear', mu_r=12),
                  dict(kind='nonlinear', bh_H_Apm_B_T=[[0, 0], [300, 1.9], [800, 2.1], [4000, 2.3]])]
    s = FEMBEM(cube(), material_laws(properties, [0, 1, 2, 0, 1, 2]))
    rng = np.random.default_rng(617)
    u, v = rng.normal(size=(2, len(s.vertices))) * 13
    H0 = np.array([421., -87, 63])
    _, A = s.volume_residual(u, H0)
    step = 1e-4
    finite = (s.volume_residual(u+step*v, H0)[0]-s.volume_residual(u-step*v, H0)[0])/(2*step)
    np.testing.assert_allclose(A @ v, finite, rtol=3e-7, atol=3e-6)
    np.testing.assert_allclose(A.toarray(), A.T.toarray(), atol=1e-12)
    np.testing.assert_allclose(A @ np.ones(len(u)), 0, atol=1e-12)
    assert np.linalg.eigvalsh(A.toarray())[1] > 0


def test_disconnected_components_keep_relative_potential_and_conserve_flux():
    geometry = np.r_[cube(), cube([3.3, .2, .7])]*1e-3
    law = material_laws([dict(kind='permanent', br_T=1.2, mu_r=1.05),
                         dict(kind='linear', mu_r=100)], [0]*6+[1]*6)
    s = FEMBEM(geometry, law, quadrature_order=12)
    u, q, block = s.solve(tol=1e-10, linear_rtol=1e-13)
    un, qn, plain = s.solve(tol=1e-10, preconditioner='none', linear_rtol=1e-13)
    np.testing.assert_allclose(u, un, rtol=2e-8, atol=1e-4)
    np.testing.assert_allclose(q, qn, rtol=2e-8, atol=1e-4)
    assert block['total_gmres_iterations'] < plain['total_gmres_iterations']
    assert block['boundary_weak_relative_residual'] < 1e-9
    assert s.info['components'] == 2
    shifted = u.copy()
    for k in range(2):
        nodes = s.component_ids == k
        shifted[nodes] -= u[nodes].mean()
        faces = s.component_ids[s.faces[:, 0]] == k
        assert abs(s.areas[faces] @ q[faces]) < 1e-8*np.linalg.norm(s.areas[faces]*q[faces])
    # Independently pinning each component changes the coupled boundary equation.
    assert np.linalg.norm(s.C @ shifted[s.boundary_nodes]+s.V @ q) > 1e-3*np.linalg.norm(s.C @ u[s.boundary_nodes])
    # Rigid translation and physical length scaling preserve H and q; phi scales.
    moved = FEMBEM(geometry*7+[.07, -.02, .03], law, quadrature_order=12)
    um, qm, _ = moved.solve(tol=1e-10)
    np.testing.assert_allclose(s.fields(u)[0], moved.fields(um)[0], rtol=2e-9, atol=1e-4)
    np.testing.assert_allclose(q, qm, rtol=2e-9, atol=1e-4)


def test_zero_contrast_has_zero_scattering_and_failure_is_not_a_result():
    s = FEMBEM(cube(), material_laws([dict(kind='linear', mu_r=1)], [0]*6))
    u, q, info = s.solve(H_external=[200, -30, 45])
    np.testing.assert_array_equal(u, 0)
    np.testing.assert_array_equal(q, 0)
    assert info['nonlinear_updates'] == 0
    nonlinear = FEMBEM(cube(), material_laws([dict(kind='nonlinear',
        bh_H_Apm_B_T=[[0, 0], [300, 1.9], [800, 2.1], [4000, 2.3]])], [0]*6))
    with pytest.raises(NonConvergence):
        nonlinear.solve(H_external=[3e5, 1e4, 0], maxiter=1)


def test_dense_reference_budget_is_checked_before_boundary_assembly(monkeypatch):
    def forbidden(*args):
        raise AssertionError('Boundary allocation reached despite the budget')
    monkeypatch.setattr(FEMBEM, 'layers', forbidden)
    with pytest.raises(MemoryError, match='accelerated boundary backend'):
        FEMBEM(cube(), material_laws([dict(kind='linear', mu_r=1)], [0]*6), max_boundary_faces=4)


def test_air_gradient_requires_valid_stencils_and_matches_different_steps():
    s = FEMBEM(cube()*1e-3, material_laws([dict(kind='permanent', br_T=1.2)], [0]*6))
    u, q, _ = s.solve()
    checks = []
    def validate(points):
        checks.append(len(points))
        if np.any(points[:, 0] <= 1e-3):
            raise ValueError('Not verified air')
    points = np.array([[2e-3, .5e-3, .5e-3], [4e-3, .7e-3, .2e-3]])
    B, info = s.air_field(u, q, points, validate_air=validate)
    assert checks == [2, 36]
    assert np.isfinite(B).all() and B[0, 0] > 0
    assert info['max_step_difference_Apm'] < 1e-3
    with pytest.raises(FloatingPointError, match='certification'):
        s.air_field(u, q, np.array([[1e-3+1e-12, .5e-3, .5e-3]]), validate_air=validate)


def test_exterior_sphere_potential_and_field_converge_to_analytic_dipole():
    from scipy.spatial import ConvexHull
    from gpu_solver import MU0
    radius = 1e-3
    points = np.array([[2.4, 0, 0], [0, 0, 2.4], [.9, 1.5, 1.6]])*radius
    r = np.linalg.norm(points, axis=1)
    coefficient = 1.2/(MU0*(1.05+2))
    exact_phi = coefficient*radius**3*points[:, 2]/r**3
    exact_H = coefficient*radius**3*(3*points[:, 2, None]*points/r[:, None]**5
                                    - np.array([0, 0, 1])/r[:, None]**3)
    def validate(points):
        # The entire polyhedral FEM mesh lies inside this sphere.
        if np.any(np.linalg.norm(points, axis=1) <= radius):
            raise ValueError('Stencil is not certified exterior to the sphere')
    vertices = np.r_[np.eye(3), -np.eye(3)]
    errors = []
    for level in range(4):
        faces = ConvexHull(vertices).simplices
        if level >= 2:
            tetra = np.concatenate((np.zeros((len(faces), 1, 3)), vertices[faces]), axis=1)*radius
            s = FEMBEM(tetra, material_laws([dict(kind='permanent', br_T=1.2,
                mu_r=1.05, direction=[0, 0, 1])], [0]*len(tetra)), quadrature_order=8)
            u, q, _ = s.solve()
            B, _ = s.air_field(u, q, points, validate_air=validate)
            errors.append(np.array([np.linalg.norm(s.potential(u, q, points)-exact_phi)/np.linalg.norm(exact_phi),
                                    np.linalg.norm(B/MU0-exact_H)/np.linalg.norm(exact_H)]))
        mids = (vertices[faces]+np.roll(vertices[faces], -1, axis=1)).reshape(-1, 3)/2
        vertices = np.unique(np.r_[vertices, mids], axis=0)
        vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    assert np.all(errors[1] < .03), errors
    assert np.all(errors[1] < .4*errors[0]), errors


def test_geometry_only_and_boundary_tree_partition_without_dense_allocation():
    from boundary_hmatrix import BoundaryTree
    mesh=FEMBEM(np.r_[cube(),cube([5.,0,0])],material_laws([dict(kind='linear',mu_r=1)],[0]*12),
                boundary_backend='geometry')
    with pytest.raises(ValueError,match='Geometry-only'):
        mesh.solve()
    tree=BoundaryTree(mesh.tri,leaf_size=4,eta=2.)
    covered=np.zeros((len(mesh.faces),len(mesh.faces)),dtype=int)
    for a,b,far in tree.blocks:
        x,y=tree.nodes[a],tree.nodes[b]
        covered[x['start']:x['end'],y['start']:y['end']]+=1
        if far:
            assert np.any((x['hi']<y['lo'])|(y['hi']<x['lo']))
    np.testing.assert_array_equal(covered,1)
    assert tree.summary()['admissible_blocks']>0


@pytest.mark.parametrize('storage_dtype',['float64','float32'])
def test_cuda_boundary_integrals_compression_and_amg_against_dense_reference(tmp_path,storage_dtype):
    pytest.importorskip('cupy')
    pytest.importorskip('pyamg')
    from scipy.spatial import ConvexHull
    vertices=np.r_[np.eye(3),-np.eye(3)]
    # Whole nodal patches need enough faces for economical far-field factors.
    for _ in range(2):
        faces=ConvexHull(vertices).simplices
        mids=(vertices[faces]+np.roll(vertices[faces],-1,axis=1)).reshape(-1,3)/2
        vertices=np.unique(np.r_[vertices,mids],axis=0)
        vertices/=np.linalg.norm(vertices,axis=1)[:,None]
    faces=ConvexHull(vertices).simplices
    one=np.concatenate((np.zeros((len(faces),1,3)),vertices[faces]),axis=1)
    tetra=np.r_[one,one+[8.,.2,.4]]*1e-3
    laws=material_laws([dict(kind='permanent',mu_r=1.05,br_T=1.2),dict(kind='linear',mu_r=100)],
                       [0]*len(one)+[1]*len(one))
    dense=FEMBEM(tetra,laws,quadrature_order=8)
    compressed=FEMBEM(tetra,laws,quadrature_order=8,boundary_backend='cuda_hmatrix',
        boundary_options=dict(leaf_size=8,eta=2.,rtol=1e-7,max_rank=48,budget_gib=.1,cache_path=tmp_path,
                              storage_dtype=storage_dtype,far_quadrature_order=4,pivot_row_backend='shared',skip_inactive=True))
    rng=np.random.default_rng(443)
    u=rng.normal(size=len(dense.vertices));q=rng.normal(size=len(dense.faces))
    expected=dense.boundary_residual(u,q)
    actual=compressed.boundary_residual(u,q)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<2e-6
    ids=np.array([0,3,11,len(q)-1])
    direct=compressed.boundary.direct_rows(ids,q,u[dense.boundary_nodes])
    expected_rows=expected[ids]-.5*(dense.mass.T@u)[ids]
    np.testing.assert_allclose(direct,expected_rows,rtol=1e-10,atol=1e-13)
    assert compressed.info['hmatrix_low_rank_blocks']>0
    assert q@compressed.boundary.precondition(q)>0
    ud,qd,_=dense.solve(tol=1e-9)
    uh,qh,info=compressed.solve(tol=1e-9,fem_preconditioner='amg',linear_rtol=1e-10)
    assert np.linalg.norm(uh-ud)/np.linalg.norm(ud)<2e-5
    assert np.linalg.norm(qh-qd)/np.linalg.norm(qd)<2e-5
    assert info['residual']<1e-9 and info['AMG_levels']>=1
    points=np.array([[0,0,3e-3],[4e-3,2e-3,-1e-3]])
    np.testing.assert_allclose(compressed.potential(ud,qd,points),dense.potential(ud,qd,points),rtol=1e-9,atol=1e-9)
    compressed.boundary.save_cache()
    loaded=FEMBEM(tetra,laws,quadrature_order=8,boundary_backend='cuda_hmatrix',
        boundary_options=dict(leaf_size=8,eta=2.,rtol=1e-7,max_rank=48,budget_gib=.1,cache_path=tmp_path,
                              storage_dtype=storage_dtype,far_quadrature_order=4,pivot_row_backend='shared',skip_inactive=True))
    assert loaded.info['hmatrix_cache_hit']
    np.testing.assert_allclose(loaded.boundary_residual(u,q),actual,rtol=1e-10,atol=1e-13)
    cached=next(tmp_path.glob('*.bin'))
    with cached.open('r+b') as stream:
        byte=stream.read(1);stream.seek(0);stream.write(bytes([byte[0]^1]))
    with pytest.raises(ValueError,match='content hash'):
        FEMBEM(tetra,laws,quadrature_order=8,boundary_backend='cuda_hmatrix',
            boundary_options=dict(leaf_size=8,eta=2.,rtol=1e-7,max_rank=48,budget_gib=.1,cache_path=tmp_path,
                                  storage_dtype=storage_dtype,far_quadrature_order=4,pivot_row_backend='shared',skip_inactive=True))


def test_shared_pivot_panels_match_direct_integrals_for_all_incident_bases():
    """Panel reuse must preserve P0/P1 values, incidence order and planar zeros."""
    cp=pytest.importorskip('cupy')
    from boundary_hmatrix import BoundaryHMatrix
    tetra=np.r_[cube(),cube([3.,.1,.2])]
    mesh=FEMBEM(tetra,material_laws([dict(kind='linear',mu_r=1)],[0]*len(tetra)),boundary_backend='geometry')
    probe=BoundaryHMatrix(mesh,leaf_size=4,quadrature_order=8,far_quadrature_order=4,_assemble=False)
    m=len(mesh.faces);n=len(probe.columns)
    desc=np.array([[0,m//2,0,n],[m//2,m-m//2,0,n]],dtype=np.int64)
    plan=probe._row_plan(desc,n,force=True)
    assert plan['expanded_panels']>2*plan['shared_panels']
    for selected in ([0,0],[3,4],[m//2-1,m//2-1]):
        pivots=cp.asarray(selected,dtype=cp.int32)
        expected=probe._batch(desc,1,nr=m//2,nc=n,pivots=pivots,far=True)
        actual=probe._pivot_rows(plan,pivots)
        np.testing.assert_array_equal(cp.asnumpy(actual),cp.asnumpy(expected))
        mask=cp.asarray([True,False])
        skipped=probe._pivot_rows(plan,pivots,mask)
        expected[1]=0
        np.testing.assert_array_equal(cp.asnumpy(skipped),cp.asnumpy(expected))
        skipped_direct=probe._batch(desc,1,nr=m//2,nc=n,pivots=pivots,far=True,active=mask)
        np.testing.assert_array_equal(cp.asnumpy(skipped_direct),cp.asnumpy(expected))
    with pytest.raises(ValueError,match='Coefficient-only'):
        probe.apply(np.zeros(m),np.ones(len(mesh.boundary_nodes)))


def test_tiny_translated_panels_have_zero_planar_double_layer_and_mesh_locator():
    base=np.array([[[0.,0,0],[1,0,0],[0,1,0],[0,0,1]]])*1e-5
    tetra=np.r_[base,base+[1.,.3,.7]]
    mesh=FEMBEM(tetra,material_laws([dict(kind='linear',mu_r=1)],[0,0]),boundary_backend='geometry')
    bary,_=triangle_rule(8)
    for i in range(len(mesh.faces)):
        _,double=mesh.layers(bary@mesh.tri[i])
        np.testing.assert_array_equal(double[:,i],0)
    centers=tetra.mean(axis=1)
    np.testing.assert_array_equal(mesh.locate_points(centers),[0,1])
    np.testing.assert_array_equal(mesh.locate_points([[.4,.2,.3],[-1,0,0]]),[-1,-1])
    centers=mesh.tri.mean(axis=1)*mesh.length+mesh.origin
    np.testing.assert_array_equal(mesh.locate_boundary(centers,mesh.normals),np.arange(len(mesh.faces)))
    with pytest.raises(ValueError,match='Observation'):
        mesh.validate_air(centers)


@pytest.mark.parametrize('storage_dtype',['float64','float32'])
def test_graded_boundary_preserves_pointwise_constant_identity_after_compression(storage_dtype):
    """Small rows and minority K columns must retain the closed-surface identity.

    Three closed polyhedral spheres have an area ratio >1e6. The analytical
    constant double layer is -1/2 on every face. The former mixed-channel ACA
    gives 7.22e-5 here, exceeding the independent 50*rtol acceptance limit.
    """
    pytest.importorskip('cupy')
    from scipy.spatial import ConvexHull
    vertices=np.r_[np.eye(3),-np.eye(3)]
    for _ in range(2):
        faces=ConvexHull(vertices).simplices
        mids=(vertices[faces]+np.roll(vertices[faces],-1,axis=1)).reshape(-1,3)/2
        vertices=np.unique(np.r_[vertices,mids],axis=0)
        vertices/=np.linalg.norm(vertices,axis=1)[:,None]
    faces=ConvexHull(vertices).simplices
    one=np.concatenate((np.zeros((len(faces),1,3)),vertices[faces]),axis=1)
    tetra=np.r_[one,one*1e-3+[1.01,.013,.007],one+[10.,.2,.1]]*1e-3
    mesh=FEMBEM(tetra,material_laws([dict(kind='linear',mu_r=1)],[0]*len(tetra)),
        quadrature_order=8,boundary_backend='cuda_hmatrix',
        boundary_options=dict(leaf_size=8,eta=3.,rtol=1e-6,max_rank=64,budget_gib=.25,
                              storage_dtype=storage_dtype,far_quadrature_order=4))
    assert mesh.areas.max()/mesh.areas.min()>1e6
    assert mesh.info['hmatrix_low_rank_blocks']>0
    value=mesh.boundary.apply(np.zeros(len(mesh.faces)),np.ones(len(mesh.boundary_nodes)))
    assert np.max(abs(value/mesh.areas-.5))<5e-5


@pytest.mark.parametrize('storage_dtype',['float64','float32'])
def test_exact_K_refinement_restores_full_operator_after_local_coefficient_error(storage_dtype):
    """A faulty K coefficient must be repaired for arbitrary traces, not clamped.

    Independent reference is the dense original-panel operator on two closed
    cubes. Inject one known row-averaged K error, then require restored random
    q/P1 actions and the constant identity at the same numerical tolerances.
    """
    cp=pytest.importorskip('cupy')
    geometry=np.r_[cube(),cube([3.,.1,.2])]
    laws=material_laws([dict(kind='linear',mu_r=1)],[0]*len(geometry))
    dense=FEMBEM(geometry,laws,quadrature_order=8)
    mesh=FEMBEM(geometry,laws,quadrature_order=8,boundary_backend='cuda_hmatrix',
        boundary_options=dict(leaf_size=4,eta=2.,rtol=1e-6,max_rank=32,budget_gib=.1,storage_dtype=storage_dtype))
    b=mesh.boundary;metadata=cp.asnumpy(b.metadata);columns=cp.asnumpy(b.columns);m=len(mesh.faces)
    for rs,nr,cs,nc,rank,left,right in metadata:
        keep=np.flatnonzero(columns[cs:cs+nc]>=m)
        if rank<0 and len(keep):
            for buffer in b.payload:
                delta=int(left)-int(buffer.data.ptr)
                if 0<=delta<buffer.nbytes:
                    first=delta//buffer.dtype.itemsize
                    buffer.ravel()[first+int(keep[0])]+=.01
                    break
            break
    rng=np.random.default_rng(291)
    u=rng.normal(size=len(mesh.vertices));q=rng.normal(size=m)
    expected=dense.boundary_residual(u,q)
    assert np.linalg.norm(mesh.boundary_residual(u,q)-expected)>1e-6*np.linalg.norm(expected)
    info=mesh.refine_boundary(rtol=1e-6,max_extra_mib=1)
    assert len(info['blocks'])>=1 and info['added_bytes']>0
    actual=mesh.boundary_residual(u,q)
    np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=1e-9)
    value=b.apply(np.zeros(m),np.ones(len(mesh.boundary_nodes)))/mesh.areas
    assert np.max(abs(value-.5))<5e-5


def test_multilevel_amg_preserves_two_constant_modes_with_large_material_contrast():
    pyamg=pytest.importorskip('pyamg')
    from itertools import product
    from scipy.sparse import diags
    from scipy.sparse.linalg import cg
    n=6
    one=np.concatenate([cube(shift)/n for shift in product(range(n),repeat=3)])
    geometry=np.r_[one,one+[3.,.1,.2]]
    laws=material_laws([dict(kind='linear',mu_r=1),dict(kind='linear',mu_r=5000)],
                       [0]*len(one)+[1]*len(one))
    mesh=FEMBEM(geometry,laws,boundary_backend='geometry')
    _,A=mesh.volume_residual(np.zeros(len(mesh.vertices)),np.zeros(3))
    P=A+mesh.mass@diags(1/mesh.areas)@mesh.mass.T
    exact=np.sin(3*mesh.vertices[:,0])+np.where(mesh.component_ids==0,2.,-3.)
    rhs=P@exact
    hierarchy=pyamg.smoothed_aggregation_solver(P,symmetry='symmetric',max_coarse=200)
    answer,status=cg(P,rhs,M=hierarchy.aspreconditioner(),rtol=1e-12,maxiter=120)
    assert len(hierarchy.levels)>1
    assert status==0 and np.linalg.norm(P@answer-rhs)/np.linalg.norm(rhs)<1e-10
    assert np.linalg.norm(answer-exact)/np.linalg.norm(exact)<1e-6
