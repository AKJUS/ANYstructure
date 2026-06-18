# FE Solver Theory Notes

This note records the theoretical basis and validity limits for the production
`fe_solver` package.  The implementation target is an ANYstructure-oriented
beam-shell solver, not a general-purpose nonlinear FE code.

## Units and DOFs

- SI units are used internally: m, N, Pa, kg.
- Every node has six DOFs ordered as `ux, uy, uz, rx, ry, rz`.
- Shells, beams, boundary conditions and MPCs share the same global DOF space.
- Fixed DOFs and beam-shell MPC slave DOFs are eliminated by the transformation
  `u = T q + u0` before static, buckling and transient solves.

## Shell Element

`ShellElement` is a 4/8-node Mindlin-Reissner quadrilateral shell.  At each
integration point the element builds a local orthonormal frame from the surface
tangents:

```text
local x = projected xi tangent
local z = shell normal
local y = local z cross local x
```

Global nodal translations and rotations are transformed to that local frame
before evaluating membrane strain, bending curvature and transverse shear.
The elastic constitutive blocks are:

```text
D_membrane = E h / (1 - nu^2) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu)/2]]
D_bending  = E h^3 / (12 (1 - nu^2)) times the same plane-stress block
D_shear    = kappa G h I, kappa = 5/6
```

The 4-node shell uses an MITC4-style assumed natural shear interpolation.  The
covariant shear strains are sampled at the four edge midpoints and interpolated
to the 2x2 integration points.  This avoids the excessive thin-plate shear
stiffness of a fully integrated displacement Q4 without introducing the
one-point reduced-shear hourglass mode.

The 8-node shell uses full 3x3 membrane/bending integration and reduced 2x2
transverse shear integration.  When `ShellElement(..., reduced_integration=True)`
is used for S8R/Q8R, membrane/bending, mass, geometric stiffness, and stress
recovery are evaluated at the reduced 2x2 rule.  The S8R/Q8R path adds a small
nullspace-projection hourglass stiffness: the reduced-integration free-element
nullspace is computed, the six rigid-body modes are projected out, and the
remaining modes receive a small positive stiffness scaled from the element
stiffness.  The intent is to remove spurious zero-energy modes while keeping
rigid motion and reduced-point patch behavior unchanged.  Broad production use
still requires external benchmark coverage for representative distorted shell
panels.  Both shell topologies include a small drilling stabilization strain:

```text
theta_z - 0.5 * (dv/dx - du/dy)
```

This gives the otherwise free drilling rotation a finite stiffness while
leaving rigid rotation about the shell normal strain-free.

## Mass and Pressure Loading

Shell mass is integrated consistently with the shell shape functions.  The
translational mass scales with `rho h`; rotary inertia scales with
`rho h^3 / 12`.  Beam mass uses translational lumping plus section rotary
inertia for torsion and bending rotations.

Shell pressure is assembled as a consistent nodal load:

```text
f_i = integral_A N_i p n dA
```

where `n` is the element normal implied by the element node order.  Follower
load moments are not included in the linear load vector.

## Linear Transient Dynamics and Slamming V1

The transient solver advances the constrained/reduced linear system:

```text
M qdd + C qd + K q = F(t)
C = alpha M + beta_R K
```

The default method is Newmark average acceleration:

```text
beta_N = 1/4
gamma_N = 1/2
```

For constant `dt`, `K`, `M` and Rayleigh damping, the solver reuses the sparse
factorization of the effective stiffness:

```text
K_eff = K + a0 M + a1 C
```

The slamming v1 load model is prescribed pressure over selected shell elements.
`PressurePatch` supports explicit element IDs, a centroid-selected rectangular
or circular area, or a custom selector callback.  Pressure magnitude is a
constant, a time table or a callable.  This is a structural response model to a
given pressure history; it is not fluid-structure interaction and does not add
hydrodynamic added mass.

Validity limits for v1 slamming:

- linear transient response about the undeformed structure,
- prescribed shell-normal pressure only,
- centroid inclusion for patch area selection,
- fixed time-step integration,
- no contact, cavitation, water-entry kinematics or pressure feedback.

## Buckling and Nonlinear Statics

Linear buckling solves:

```text
K phi = lambda KG phi
```

with positive `KG` representing destabilizing compression in the supplied
reference stress/resultant state.

The incremental nonlinear static path uses von Karman shell kinematics,
beam-column geometric coupling and optional layered J2 plane-stress plasticity
with DNV-RP-C208 style material curves.  It is suitable for restrained
plate/stiffened-panel response and pre/post-buckling capacity checks in the
implemented range.  It is not a corotational large-rigid-rotation formulation.

### DNV-RP-C208 Capacity Workflow Anchors

The nonlinear capacity workflow is aligned with the DNV-RP-C208 guidance
reviewed from the supplied PDF:

- key parameters such as element type, mesh, material curve, imperfections and
  residual-stress representation should be selected conservatively or
  calibrated against a comparable standard/test case,
- true stress / true plastic strain material curves should be used consistently
  with the element formulation,
- permanent loads should be applied before environmental/proportional loads in
  nonlinear analyses,
- buckling capacity checks should include equivalent imperfections and may use
  scaled eigenmodes or standard imperfection patterns,
- displacement control is useful when the force-controlled run must be driven
  beyond the load limit to identify the peak response.

Implemented interfaces:

- `dnv_c208_steel_curve(grade, thickness, fractile="low")` returns the built-in
  low-fractile S235/S275/S355/S420/S460 section 4.6.6 curves.  Mean curves are
  deliberately not guessed; supply explicit data through `curve_from_properties`
  when a mean-capacity study is required.
- `ImperfectionField`, `EigenmodeImperfection`, `StandardImperfection` and
  `CompositeImperfection` describe stress-free nodal reference-geometry offsets.
  `apply_imperfection()` modifies coordinates before the nonlinear solve, so
  zero displacement in the imperfect model has zero internal force.
- Standard deterministic imperfections include member bow (default `L/300`),
  plate sinusoidal half-wave (default `s/200`) and flange/outstand twist
  (default `0.02 rad`).  The defaults correspond to the reviewed DNV table, but
  users should still calibrate or override amplitudes when the failure mode or
  tolerance class requires it.
- `NonlinearLoadProgram` applies ordered stages.  The common DNV sequence is a
  permanent stage first and an environmental/pressure/compression stage second.
- `DisplacementControl` augments the Newton system with a scalar displacement
  constraint and a load proportionality factor unknown, allowing monotonic
  capacity tracing past a simple force-control limit.

Material modelling:

- Shells use layered plane-stress J2 plasticity through Gauss-Lobatto thickness
  layers.  Result diagnostics include equivalent plastic strain, compressed-side
  plastic strain and layer strain extrema when plastic state is available.
- Beam/stiffener plasticity is opt-in through `FiberSectionPlasticityConfig`.
  The v1 beam fiber model integrates uniaxial axial/bending stress over a
  section grid scaled to the supplied `A`, `Iy` and `Iz`; shear and torsion stay
  elastic.

Residual stresses:

- v1 treats calibrated equivalent geometric imperfections as the practical
  residual-stress proxy for buckling capacity workflows.
- `initial_element_states` is reserved for future residual stress/prestrain
  fields.  A full residual-stress implementation must contribute to both
  internal-force equilibrium and tangent stiffness, and must report diagnostics
  separately from geometric imperfections.

## Verification Anchors

The solver test suite protects:

- DOF ordering and separated K/M/F assembly,
- shell rigid-body modes and patch behavior,
- MITC4 zero-mode and thin-strip shear-locking behavior,
- pressure resultants and transient pressure-patch impulse,
- Newmark SDOF response and undamped energy conservation,
- CalculiX shell-reference discovery and internal convergence-table comparison.

External-reference workflow:

- CalculiX is the default practical FE reference for shell convergence and
  prescribed dynamic pressure workflows.
- The official CalculiX manual documents shell elements, dynamic procedures,
  amplitudes and `*DLOAD`: https://www.dhondt.de/ccx_2.22.pdf
- Mixed-shear-projected Reissner-Mindlin/MITC-family behavior is used as the
  theoretical anchor for the S4 shear treatment, for example:
  https://arxiv.org/abs/1410.3683
