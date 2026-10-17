# Coarse magnetic model

All internal lengths are metres, H and M are A/m, B is tesla, volumes are m³ and forces are newtons.

## Why the near field is regularized

The requested bare point-volume dipole operator is provided as `dipole_operator` and tested against analytical dipoles. With spherical self terms and nearby partially occupied thin-wall cells, its volume-weighted self matrix can have positive eigenvalues. High-permeability material then develops spurious polarization modes. A small nonlinear residual alone cannot distinguish those modes from meaningful solutions. In the supplied STEP, this caused strong baseline dependence on floating-point precision and a misleading large change from isolated-camera magnetization.

The camera solver instead uses normalized Gaussian volume distributions. For a cell of volume V, its width is σ = V^(1/3) / √(4π). The interaction between target i and source j is the Hessian of

    erf(r / (√2 s)) / (4π r),  with s² = σ_i² + σ_j²,

multiplied by the source volume. This is the source-field and target-volume convolution. Its volume-weighted matrix is symmetric and negative semidefinite: in Fourier space the interaction is the negative longitudinal projector, multiplied by Gaussian factors. This removes the artificial positive-energy polarization mode without clipping eigenvalues or silently changing the material curve.

The diagonal limit is exactly -I/3 with this width choice. At large distances the operator becomes the conventional dipole tensor. A small-r series avoids cancellation. For air point observations, only the source Gaussian is convolved. Flux density includes both H and the Gaussian source magnetization density. This is still a smoothed volume approximation; it does not reproduce an exact cuboid demagnetizing field or sharply resolve CAD interfaces.

## Constitutive iteration

For each cell split H into H_ext - M/3. Solve the local isotropic relation against H_ext first:

    H_ext = H + M(H)/3.

Linear and recoil materials have a closed form. Soft magnetic materials use a monotonic interpolated inverse of the scalar relation. Above the last supplied B-H sample, B is extrapolated with slope μ0 so magnetization stays finite. The inverse is sampled densely and retains every original knot. The solver accelerates the remaining off-diagonal fixed point with a regularized eight-step Anderson history. The residual API also evaluates M - M_rem - χ(H), independently of the preconditioned stopping test.

## Force and winding assumptions

The copper CAD is a bulk envelope, not a trace-resolved winding. Uniformly distributed rectangular loops approximate its cross-section. Positive current is counterclockwise viewed from +Z; changing current reverses the loop source. Both coil source fields are included when solving induced magnetization. Biot–Savart midpoint integration uses a finite core radius based on cross-section quadrature spacing. The source field is sampled at coarse cell centers rather than integrated over each target Gaussian.

Coil force is the distributed line integral of I dℓ × B from the magnetic cell distributions and the other coil current. The same coil's own current field is omitted. There is no force calculation on the mechanical camera assembly, Maxwell stress integration, FEM, eddy current, hysteresis or mechanical motion solve. Force on the two coils alone need not satisfy Newton's third law; magnet/yoke reactions are not included in those reported vectors.

## Geometry and features

OCCT point-in-solid classification supplies sampled occupied fractions and their centroid per coarse grid cell. Per-object total volume is normalized to the CAD volume; metadata preserves the raw sampled volume and correction. Gaussian cell shapes do not reproduce partial-voxel shape exactly.

Interface distances use actual CAD faces. Field gradients use RMS neighbor differences divided by sample separation; these are approximate local variation features, including discontinuities across material interfaces. Other-camera and nearest-magnet distances use cell-center distances corrected by an equivalent half cell width. Secant permeability is |B|/(μ0|H|), including remanence for magnet cells, so it is a feature ratio rather than their recoil permeability. Saturation is |B|/1.6 T for yoke cells only. Refinement levels and object lengths are deterministic heuristics, not trained predictions.

## Libraries

Geometry uses [CadQuery/OCCT](https://cadquery.readthedocs.io/en/stable/apireference.html), and monotonic B-H interpolation uses [SciPy PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html). Neither constitutes a Maxwell replacement or a reference-quality force solver.
