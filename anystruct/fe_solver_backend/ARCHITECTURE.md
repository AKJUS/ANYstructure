# FE solver architecture

This directory is the production path for the ANYstructure-oriented finite-element solver.
The top-level `fea_solver.py` prototype is legacy/reference material only and should not be extended.

## Scope

The solver is intentionally not a general-purpose pre/post processor.  It targets generated
ANYstructure-style geometry first:

- flat stiffened panels,
- cylindrical shell panels,
- beam stiffeners coupled to shell plating,
- linear static, eigenvalue buckling, controlled nonlinear capacity checks with
  DNV-style imperfections/material curves, and linear transient pressure-patch
  response.

General CAD topology, arbitrary contact, cyclic plasticity, fracture modelling
and full arc-length post-buckling path following are out of scope for the first
production version.

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `fe_core.py` | FE model, mesh, nodes, materials and DOF numbering. |
| `elements.py` | Element formulations and element-level stiffness, mass and stress recovery. |
| `boundary.py` | Boundary conditions, load cases, nodal loads, pressure loads and load combinations. |
| `matrix_assembly.py` | Explicit assembly of K, M and F without mixing stiffness, mass and external loads. |
| `buckling.py` | Linear eigenvalue buckling solve using separately assembled elastic and geometric stiffness matrices. |
| `nonlinear.py` | Controlled proportional load stepping with tangent-stiffness stability monitoring and limit-point stop. |
| `dynamics.py` | Linear Newmark transient response and prescribed shell pressure-patch loading for slamming v1. |
| `material_curves.py` | DNV-RP-C208 section 4.6.6 flow curves (stepwise linear + yield plateau + power law), vectorized. |
| `imperfections.py` | Stress-free geometric imperfection fields, eigenmode scaling, standard member/plate patterns and calibration helper. |
| `plasticity.py` | Vectorized J2 plane-stress return mapping with isotropic hardening and Gauss-Lobatto thickness layers. |
| `nonlinear_static.py` | Incremental Newton-Raphson solver with geometric (von Karman) and material (layered J2) nonlinearity. |
| `anystructure_fem_mode.py` | Full generated-geometry ANYstructure FEM workflow: static solve, prestress recovery and buckling. |
| `assembly.py` | Constraint transformation, nullspace solves, legacy assembly wrapper and solver routines. |
| `mesh_gen.py` | Limited mesh generation for ANYstructure-style panels and stiffeners. |
| `results.py` | Result containers and post-processing helpers. |
| `validation.py` | Verification utilities and benchmark helpers used by pytest and future benchmark scripts. |
| `reference_cases.py` | Local CalculiX/PrePoMax reference discovery, upstream CalculiX shell manifest and shell convergence table parsing. |
| `shell_benchmarks.py` | Internal shell benchmark sweeps, internal convergence table writer and loose external/internal comparison helpers. |
| `cylinder_benchmarks.py` | Internal cylindrical shell pressure benchmark with nominal hoop/axial stresses and FE von Mises percentiles. |
| `linalg.py` | Shared sparse linear solver backend, matrix-class declarations and reusable factorization handles. |
| `cases.py` | Minimal analysis/load/result/prestress case provenance objects. |
| `baselines.py` | Deterministic local FE baseline generation and comparison helpers. |

## Required invariants

These invariants should be protected by tests before adding new solver features.

1. Nodes have six DOFs in this order: `ux, uy, uz, rx, ry, rz`.
2. Internal solver units are SI: metres, Newtons and Pascals.
3. Shell and beam element matrices are assembled into the same global DOF space.
4. Beam-shell eccentricity is handled by explicit MPC constraints and the global transformation, not by large penalty springs.
5. Fixed supports and MPC slave relations are eliminated by the same transformation path:

   ```text
   u = T q + u0
   K_red = T.T K T
   F_red = T.T (F - K u0)
   ```

6. Free-free static models are handled with rigid-body nullspace augmentation, not artificial support stiffness.
7. Stiffness, mass and load assembly must remain separable:

   ```text
   K = assemble stiffness only
   M = assemble mass only
   F = assemble load only
   ```

8. Solver diagnostics must report constraint method, free DOF count, MPC slave count and nullspace/load-imbalance information when relevant.
9. Linear buckling uses the convention `K phi = lambda KG phi`, where positive `KG` represents the destabilizing part of the supplied reference compression state.
10. Linear transient dynamics must use the same fixed-DOF/MPC transformation as static analysis and must not mix mass into stiffness assembly.
11. The legacy `nonlinear.py` stability monitor stops when the reduced tangent stiffness loses, or nearly loses, positive definiteness. `nonlinear_static.py` may trace capacity with force control or single-DOF displacement control, but must report the control mode and last converged point explicitly.
12. The ANYstructure FEM mode consumes geometry generated by ANYstructure. It must not depend on raw IFC parsing in the solver package.
13. `matrix_assembly.py` is the canonical K, M, KG, damping and load-vector/load-matrix assembly path.  `assembly.py` may wrap it for compatibility, but must not duplicate global assembly implementations.
14. Ordinary analyses should use `linalg.factorize()` and `FactorizationHandle.solve()`/`solve_many()` instead of direct `scipy.sparse.linalg.spsolve` calls.
15. Mesh/model revision counters must distinguish topology, geometry, material, load, boundary/MPC and result-state changes.  Geometry changes invalidate geometry-dependent element caches while preserving sparsity cache signatures when topology and MPC topology are unchanged.
16. Free-free static loads with non-zero rigid-body generalized force are rejected by default.  The old gauged/balanced behavior is available only when explicitly requested.

## Local verification infrastructure

The first SESTRA-inspired infrastructure batch is local-script driven and does
not add GitHub Actions.  Use these commands from the repository root:

```text
python scripts/run_fe_verification.py
python scripts/generate_fe_baselines.py
python scripts/compare_fe_baselines.py
```

`run_fe_verification.py` writes machine-readable and Markdown reports under
`reports/verification/` by default.  The report captures date, commit SHA when
available, platform, Python/dependency versions, command status, timings,
warnings and known limitations.  Frozen smoke baselines live under
`tests/fixtures/fe_baselines/`; reference values and tolerances are separate so
intentional solver improvements can update values without burying tolerance
policy.

Batch 02 adds independent verification families and infrastructure benchmarks:

```text
python scripts/run_fe_verification.py --static
python scripts/run_fe_verification.py --buckling
python scripts/run_fe_verification.py --nonlinear
python scripts/run_fe_verification.py --transient
python scripts/run_fe_verification.py --performance
python scripts/run_fe_benchmarks.py
```

Static, multiple-RHS static, buckling, nonlinear-static and transient results
now carry `ResultCase` provenance through their existing result/diagnostic
dictionaries.  Provenance includes the analysis type, load references, matrix
and load signatures, solver backend, recovery settings and matrix revision
metadata.  Benchmark reports intentionally measure local infrastructure timing
and Python allocation peaks only; they are trend evidence, not validation
references.

Batch 03 adds S4 validity hardening around the current 4-node shell theory:

```text
python scripts/run_fe_verification.py --s4
python scripts/run_s4_validity.py
```

The S4 validity report covers free-element rigid-body modes, exact affine
membrane/bending/shear patch metrics, skew and mild-warp diagnostic metrics,
thin-plate shear-locking strip sweeps and S4/S8 displacement/stress comparison
sweeps.  The skew membrane metric is diagnostic because it reports local shell
stress components; exact affine patch assertions are enforced for square and
parallelogram quads where the component basis is unambiguous.

Batch 04 adds the nonlinear capacity workflow wrapper:

```text
python scripts/run_fe_verification.py --capacity
python scripts/run_capacity_workflow.py
```

The workflow composes existing solver features in the intended DNV-style order:
linear static solve, prestress recovery, eigenvalue buckling, stress-free
imperfection application, then nonlinear static capacity solve.  It reports the
critical buckling factor, selected imperfection amplitude, nonlinear peak/last
converged load factors, prestress summary, and a conservative mesh/mode
adequacy diagnostic based on active elements per estimated half-wave.  It does
not alter nonlinear physics; it packages the existing solver path into a
repeatable evidence workflow.

Batch 05 adds an opt-in 2-node corotational beam geometry path for member and
stiffener nonlinear checks:

```text
python scripts/run_fe_verification.py --beam
python scripts/run_beam_validity.py
```

Enable it per element with
`cross_section["geometric_nonlinearity"] = "corotational"`.  The v1
implementation keeps the element frame attached to the current chord and
subtracts finite rigid chord rotation from nodal rotation vectors before
forming local elastic beam forces.  It is aimed at large rigid-rotation
validity and member geometric response; fiber-section plasticity keeps the
existing beam-column path in this batch.

Batch 06 adds the mass and modal foundation:

```text
python scripts/run_fe_verification.py --modal
python scripts/run_mass_modal_validity.py
```

`mass_properties.py` integrates beam and shell mass points to report total
mass, centre of mass, first moments and inertia tensors, then forms a 6x6
rigid-body mass matrix from the assembled global mass matrix.  This makes
element rotary inertia and constraint-compatible rigid fields visible in one
diagnostic.  `modal.py` solves the constrained generalized eigenproblem
`K phi = omega^2 M phi` using the same fixed/MPC transformation as static,
buckling and transient analysis.  Modes are mass-normalized, sign-stabilized,
and reported with eigenpair residuals, mass-orthogonality error, rigid-body
correlation and `ResultCase` provenance.  Full repeated-mode basis
stabilization and external modal benchmarks remain later Phase 7/14 work.

Batch 07 continues the priority-5 eigenvalue work by hardening sparse buckling:

```text
python scripts/run_fe_verification.py --buckling
python scripts/run_buckling_validity.py
```

`solve_eigenvalue_buckling()` remains source-compatible, but returned modes now
carry eigenpair residuals, validity status and repeated-mode group IDs.  The
solver reports sparse/dense route selection, rejected roots, maximum residual,
load-factor range filtering and repeated-mode groups in `BucklingResult`
diagnostics.  Optional `shift_load_factor`, `load_factor_range`,
`search_factor`, `repeated_tolerance` and `allow_dense_fallback` controls make
the sparse path more explicit.  Large reduced systems no longer fall back to a
dense solve after sparse failure unless that is explicitly allowed.  External
plate/shell/cylinder buckling validation remains later Phase 14 work.

Batch 08 starts the priority-6 qualification expansion for Q8, beams and mass:

```text
python scripts/run_fe_verification.py --element
python scripts/run_element_qualification.py
```

`element_qualification.py` gives Q8 a first-class local qualification report:
free-element rigid modes, exact affine membrane/bending/shear patch metrics for
the square reference element, skew and distorted-midside finite/symmetry
diagnostics, Q8 mass checks, and a Q4/Q8 displacement/stress/cost sweep.
The same report adds beam strong-axis orientation checks for 2-node and
3-node beams, torsion response, and assembled/integrated beam mass diagnostics.
This is still local evidence; external Q8 shell, beam dynamic, and nonlinear
tangent qualification remain later priority-6/14 work.

Batch 09 expands priority-6 material and nonlinear tangent qualification:

```text
python scripts/run_fe_verification.py --plasticity
python scripts/run_plasticity_qualification.py
```

`plasticity_qualification.py` checks plane-stress return mapping over elastic,
uniaxial, biaxial/shear, pure-shear and unloading paths, reports scaled yield
residuals and material-point tangent finite-difference errors, and records DNV
curve parameters used by the current factory.  Element-level finite-difference
tangent checks cover elastic beams, fiber-plastic beams, elastic shells and
layered-plastic shells.  Layered shell plasticity now uses a consistent
numerical algorithmic tangent of the discrete return map, so the local tangent
check is tight.  This is correctness-first and intentionally more expensive
than a closed-form analytical algorithmic tangent; replacing it analytically is
a later speed/formulation task guarded by these checks.

## Roadmap gate order

The work sequence is:

1. Freeze architecture and package API.
2. Add verification tests and validation helpers.
3. Split and harden K/M/F assembly APIs.
4. Verify shell element behaviour against patch tests and reference models.
5. Turn the cylinder example into a benchmark with nominal/percentile stress reporting.
6. Add geometric stiffness and eigenvalue buckling.
7. Add nonlinear limit-point detection, stopping before full post-buckling path following.
8. Add linear transient slamming as prescribed shell-normal pressure over a
   selected area, then validate it against CalculiX dynamic reference cases.

Do not start step 6 or 7 before the verification tests in steps 2-4 are in place and passing.

## Reference cases

`reference_cases.py` discovers local reference inputs from repository-relative
roots such as `tests/reference_cases/`, `reference_cases/` and
`examples/reference_cases/`.  A local reference case is an `*.inp` file with an
optional matching `*.frd` result file and optional JSON metadata sidecar.

The module also registers the upstream CalculiX shell convergence example from
`calculix/CalculiX-Examples/Elements/Shell`.  This upstream case is intentionally
kept as a manifest/reference candidate, not a vendored dependency.  The parser
for generated upstream convergence files reads text tables such as `S4.txt` and
`S8.txt` with rows shaped as:

```text
# size NoN smax umax
100 12 0.616489 0.013121
```

## Shell Benchmarks

`shell_benchmarks.py` runs a small internal simply supported shell benchmark and
returns `ShellBenchmarkResult` rows with maximum von Mises stress, maximum
out-of-plane displacement and normalized values.  Internal sweeps can be written
to CalculiX-like tables with two extra normalized columns:

```text
# size NoN smax umax s_norm u_norm
```

Use the script below to write `S4_internal.txt` and `S8_internal.txt`:

```powershell
python scripts/run_internal_shell_benchmark.py --output-dir reports/shell_benchmarks
```

`compare_shell_benchmark_to_reference()` can compare a parsed upstream
`ShellConvergenceTable` with internal benchmark results.  The comparison is
informational only for now because the internal model does not yet exactly
reproduce the upstream CalculiX geometry, loading or supports.

## Optional CalculiX Workflow

The upstream shell reference files are optional.  If needed, fetch them with:

```powershell
python scripts/fetch_calculix_shell_reference.py
```

Run the upstream CalculiX example workflow outside the core test suite to
generate concrete `S4.txt`, `S8.txt`, etc.  Once generated, parse them with
`discover_calculix_shell_convergence_tables()` and compare them to internal
tables as an informational trend check.

Batch 14 adds generated external-reference decks:

```powershell
python scripts/run_fe_verification.py --reference
python scripts/run_external_references.py --output reports/external_references/external_reference_report.json --deck-dir reports/external_references/decks --markdown reports/external_references/external_reference_report.md
```

`external_references.py` writes deterministic CalculiX/Abaqus-style `.inp`
decks plus JSON sidecars for a pressure S4 plate, a beam-column buckling case
and a pressure S4 cylinder.  The generated decks are parsed back through the
same discovery path used for local `.inp/.frd` references, so malformed node or
element sections fail in the local gate.  This is a handoff artifact for
external solver comparison; it intentionally does not execute CalculiX or claim
numerical agreement until `.frd` parsing/comparison is added.

## Beam-Shell Verification Manifest

`beam_shell_verification.py` implements the supplied beam/shell verification
manifest as a stable case-ID ledger.  It runs deterministic checks for the
currently supported algebra, beam, shell patch, coupling relation, nullspace,
mass/eigen, buckling, nonlinear tangent and plasticity cases, and records
unsupported or reference-dataset-dependent items as explicit `XFAIL` entries
with reasons.  A passing report means there are no unexpected required
failures; it does not turn XFAIL literature/cross-solver items into validation
claims.

```powershell
python scripts/run_fe_verification.py --beam-shell
python scripts/run_beam_shell_verification.py --output reports/beam_shell_verification/beam_shell_verification_report.json --markdown reports/beam_shell_verification/beam_shell_verification_report.md
```

## Cylinder Benchmark

`cylinder_benchmarks.py` turns the previous exploratory cylinder scripts into a
package-level benchmark.  It builds a closed cylindrical shell from shell
elements, applies self-equilibrated external pressure and optional closed-end
axial cap loads, and reports:

- nominal thin-cylinder hoop stress `-pR/t`,
- nominal closed-end axial stress `-pR/(2t)`,
- nominal von Mises stress,
- FE von Mises maximum, p95 and mid-height p95,
- displacement and solver diagnostics.

Run the script below to write a JSON report:

```powershell
python scripts/run_cylinder_benchmark.py --output reports/cylinder_benchmark.json
```

The FE values are benchmark diagnostics rather than acceptance criteria at this
stage.  They establish stable percentile reporting before geometric stiffness
and eigenvalue buckling are added in step 6.

## Linear Buckling

`matrix_assembly.py` exposes `assemble_geometric_stiffness_matrix()` as a
separate API alongside K, M and F assembly.  Element state input is supplied as
a mapping or callback keyed by element id.  Implemented element theories are
beam-column geometric stiffness for axial reference compression (consistent
cubic matrix for 2-node beams, lateral-gradient matrix for 3-node beams) and
shell membrane-stress geometric stiffness from membrane resultants:

```python
element_states = {
    1: {"axial_compression": 1000.0},      # positive compression
    2: {"axial_force": -1000.0},           # axial_force remains positive in tension
    3: {"membrane_compression_x": 500.0},  # shell Nx, compression positive [N/m]
    4: {"membrane_forces": (-500.0, 0.0, 0.0)},  # tension-positive alternative
}
```

`buckling.py` solves the constrained generalized eigenproblem:

```text
K phi = lambda KG phi
```

The same transformation used by static analysis eliminates fixed supports and
MPC slave DOFs before the eigen solve, and full-system mode vectors are
reconstructed afterward.  Positive eigenvalues are load multipliers relative to
the supplied reference compression state.  For example, if every beam element
is given `axial_compression = 1 N`, the first load factor is the critical
compression in Newtons for that reference shape.

Shell membrane-stress geometric stiffness integrates the transverse-deflection
gradient against the local membrane resultant tensor `[Nx Nxy; Nxy Ny]` at the
full bending integration points.  It is verified against the simply supported
plate uniaxial buckling coefficient `k = 4`.

Beam elements take their bending axes from the section convention: `Iy` is the
second moment about local y (governs deflection in local z), and the optional
`cross_section["orientation"]` vector pins local z to the section web
direction.  Asymmetric stiffener sections should always supply an orientation;
without it a global-Y/Z heuristic chooses the frame and the strong/weak axes of
the section are not guaranteed to land where intended.

## Nonlinear Limit-Point Checks

`nonlinear.py` adds a controlled load-step stability path for roadmap step 7.
It uses the same assembled elastic stiffness `K`, geometric stiffness `KG`,
load vector `F` and constraint transformation as the static and buckling
solvers.  At proportional load factor `lambda`, the monitored tangent is:

```text
KT(lambda) = K - lambda KG
```

Each step solves the tangent system `KT(lambda) q = lambda F`, so reported
displacements follow the linearized pre-buckling amplification, and evaluates
the smallest eigenvalue of the reduced tangent stiffness.  The run stops when
that eigenvalue becomes non-positive or when its normalized stability index
falls below the requested tolerance:

```python
result = solve_nonlinear_load_stepping(
    model,
    load_case,
    element_states={1: {"axial_compression": 1.0}},
    max_load_factor=1.2 * expected_critical_factor,
    num_steps=40,
    stability_tolerance=0.03,
)
```

This is intentionally a pre-limit stability detector, not an arc-length or
post-buckling continuation solver.  Until displacement-dependent element
tangents are implemented, the tangent update is the linearized geometric
stiffness path from step 6.

## Linear Transient Slamming V1

`dynamics.py` adds the first transient pressure-loading path.  It solves:

```text
M qdd + C qd + K q = F(t)
C = alpha M + beta_R K
```

after applying the same transformation used by static analysis:

```text
u = T q + u0
K_red = T.T K T
M_red = T.T M T
F_red(t) = T.T (F(t) - K u0)
```

The default time integrator is Newmark average acceleration
(`beta = 1/4`, `gamma = 1/2`).  For constant time step and constant matrices,
the sparse factorization of the effective stiffness is reused:

```text
K_eff = K_red + a0 M_red + a1 C_red
```

`PressurePatch` supplies prescribed shell pressure over a selected area:

```python
patch = PressurePatch.rectangular_pulse(
    name="slam",
    pressure=250_000.0,
    start_time=0.002,
    end_time=0.012,
    center=(1.0, 0.5, 0.0),
    box_size=(0.5, 0.25),
)
config = TransientConfig(dt=0.0005, t_end=0.05, rayleigh_alpha=0.0, rayleigh_beta=0.0)
result = solve_transient_newmark(model, config, pressure_patches=[patch])
```

Patch selection is centroid-based in v1.  Explicit element ids are accepted
when exact patch membership is required.  This is a prescribed pressure
response model, not a fluid-structure interaction model: there is no added
mass, pressure feedback, cavitation or water-entry kinematics.

The detailed theory and validity notes live in `THEORY.md`.

## Incremental Geometric/Material Nonlinear Statics

`nonlinear_static.py` adds a true incremental nonlinear solver on top of the
same model, load and constraint machinery:

```python
from fe_solver import (
    DisplacementControl,
    NonlinearLoadProgram,
    NonlinearLoadStage,
    dnv_c208_steel_curve,
    solve_static_nonlinear,
    standard_member_bow,
)

model.add_material("S355", 210e9, 0.3,
                   hardening_curve=dnv_c208_steel_curve("S355", 0.012))

program = NonlinearLoadProgram([
    NonlinearLoadStage("permanent", dead_load),
    NonlinearLoadStage("environmental", live_load),
])

result = solve_static_nonlinear(
    model,
    load_program=program,
    imperfection=standard_member_bow(model, member_nodes),
    control="displacement",
    displacement_control=DisplacementControl(node_id=top_node, dof="uz", target_displacement=-0.08),
    num_steps=20,
    num_layers=5,
)
```

Theory and scope:

- **Geometric nonlinearity**: total-Lagrangian von Karman kinematics in the
  shell elements (membrane strain includes transverse-deflection gradient
  terms; the tangent contains the consistent initial-stress stiffness from
  the current membrane resultants) and a consistent von Karman beam-column
  axial coupling in the 2-node beam.  This is the established formulation for
  restrained plate/stiffened-panel response including post-buckling; it is
  not a corotational large-rotation formulation, so free-edge members
  undergoing large rigid rotations are out of scope.
- **Material nonlinearity**: layered J2 plane-stress plasticity in the
  shells.  The flow curve is the DNV-RP-C208 section 4.6.6 model (stepwise
  linear with yield plateau plus power law, true stress / true plastic
  strain) attached per material via ``Material.hardening_curve``.  The return
  mapping is the plane-stress projected algorithm, vectorized over all
  integration points and Gauss-Lobatto thickness layers of an element at
  once.  Materials without a curve stay elastic.  Beam/stiffener plasticity is
  opt-in through ``cross_section["fiber_plasticity"]``; axial and bending
  stresses are integrated over a uniaxial fiber grid using the same material
  curve, while shear and torsion remain elastic in v1.
- **Imperfections**: `imperfections.py` applies stress-free nodal reference
  coordinate offsets.  Supported sources are explicit fields, scaled buckling
  modes, standard member bows, standard plate half-waves, flange twists and
  composites.  Applying an imperfection changes the reference geometry; zero
  displacement in the imperfect model produces zero internal force.
- **Solution control**: full Newton-Raphson per force increment with a
  backtracking-line-search rescue retry, adaptive load-step cutback/growth,
  ``F = F_constant + lambda * F_proportional`` load split, ordered
  ``NonlinearLoadProgram`` stages, and plastic state committed only on
  increment convergence.  Displacement control solves an augmented Newton
  system with the load factor as an additional unknown.  Results report
  ``last_converged_load_factor``, ``peak_load_factor``, force-displacement
  history, active stage and failure reason separately.
- **Speed**: cached reference-geometry per element, residual-only assemblies
  in the line search, COO-triplet assembly, one sparse factorization per
  iteration.

Validation anchors in ``tests/test_fe_solver_nonlinear_static.py``: RP-C208
curve knots/continuity, uniaxial tension landing exactly on the curve, plate
strip bending capacity at the quadrature plastic moment (shape factor ~1.5),
beam-column amplification 2x at half the Euler load, and von Karman membrane
stiffening of an immovable strip.

Additional DNV workflow anchors in ``tests/test_fe_solver_nonlinear_dnv.py``:
RP-C208 low-fractile factory table values, exact eigenmode imperfection
scaling, stress-free imperfect reference geometry, standard member/plate
imperfection amplitudes, beam fiber axial yield, displacement-control load
tracing and staged permanent/environmental load sequencing.

## ANYstructure Generated-Geometry FEM Mode

`anystructure_fem_mode.py` is the production-facing full-geometry workflow.  It
does not import FEM files and does not parse IFC.  ANYstructure generates the
complete cylinder or panel geometry first, and the solver consumes a normalized
duck-typed geometry object containing:

- nodes with stable ids and coordinates,
- shell faces with node ids, thickness and material,
- beam members for stiffeners, girders, frames, longitudinals and transverses,
- optional beam-shell coupling metadata,
- optional support metadata.

IfcOpenShell-derived geometry may represent stiffeners and girders as plates
upstream, but that representation is not the solver idealization.  The default
ANYstructure FEM mode is a beam-shell model: plating remains shells, while
stiffeners and girders are supplied as line beam members with section
properties and eccentric/coupling metadata.  Plates tagged as stiffener/girder
parts are excluded from the shell mesh by default and require matching beam
members, so the solver fails closed instead of silently double-counting member
stiffness.

`idealize_generated_geometry_members()` performs this handoff normalization.
Explicit line members are preserved.  Tagged stiffener/girder plate surfaces can
also be collapsed automatically into equivalent beam members by inferring a
centerline from the generated node coordinates and a plate-strip cross-section
from surface area and thickness.  This is intended for ANYstructure-generated
member geometry; arbitrary untagged IFC plate facets are not guessed into beams.

The workflow is:

```text
generated ANYstructure geometry
  -> FEModel
  -> symmetric loads from existing ANYstructure input fields
  -> static solve
  -> shell/beam prestress recovery from the static result
  -> K phi = lambda KG(result) phi
```

The v1 output is buckling plus stress: critical load factors, mode data,
stress percentiles, displacement maxima, load resultants and solver
diagnostics.  Plastic collapse and post-buckling tracing remain out of scope.

## Selective Recovery And Resource Policy

`recovery.py` defines the Batch 10 foundation for bounded result recovery and
resource provenance:

- `RecoveryConfig` selects node ids, element ids and stress components, and
  records whether displacement, stress and reaction recovery are requested.
- `ResourceConfig` records requested solver/assembly/recovery thread counts,
  worker counts, deterministic mode and optional memory limits.  Recovery
  threading is opt-in and measured; solver and assembly thread controls remain
  deferred.
- `MemoryEstimate` gives deterministic estimates for sparse CSR storage,
  right-hand sides, transient histories, eigenvectors and nonlinear state
  blocks.
- `RecoveryExecutionReport` records the recovery backend, requested/used
  workers, item count, deterministic flag, timing and serial/threaded reason.

Legacy static result creation still recovers all nodes/elements by default.
Callers can pass a `RecoveryConfig` to `create_fe_result()` for selected
results.  If a `ResourceConfig(recovery_threads=N)` with `N > 1` is supplied,
element stress recovery uses a deterministic thread pool and records an
execution report.  Transient analysis accepts the same policy object and
includes the serialized recovery/resource/memory metadata in diagnostics and
result-case provenance while preserving existing full-history arrays for API
compatibility.

Local evidence:

```text
python scripts/run_recovery_policy.py --output reports/recovery_policy/recovery_policy_report.json --markdown reports/recovery_policy/recovery_policy_report.md
python scripts/run_fe_verification.py --recovery
```

Batch 12 completes the first resource-control pass:

- `ResourcePolicyError` and `enforce_memory_limit()` fail early when a supplied
  `ResourceConfig.memory_limit_bytes` is exceeded by deterministic storage
  estimates.
- `TransientConfig.recovery.history_mode` controls saved history storage:
  `full` preserves legacy full `u/v/a` arrays, `selected` stores only selected
  node DOFs in the result arrays, and `envelope` stores peak absolute
  displacement/velocity/acceleration envelopes plus requested node histories.
- `TransientResult` reports `history_storage_mode`, optional
  `history_dof_indices`, and optional envelope arrays so reduced storage is
  explicit rather than implicit.
- The recovery policy report records full, selected and envelope transient
  history byte estimates alongside measured serial/threaded recovery.

## Factorization Reuse

Batch 13 adds an explicit local sparse factorization cache:

- `FactorizationCache` owns cached `FactorizationHandle` objects and reports
  hits, misses, entries and failures.  It is local/opt-in rather than global,
  so analyses control cache lifetime.
- Cache keys use sparse matrix content signatures by default, not sparsity
  alone, because unchanged topology can still have changed stiffness values.
- `factorize_cached()` and `cached_inverse_operator()` route repeated direct
  solves and shift-invert operators through the common `linalg.py` backend.
- `solve_linear_many()` now reports cache diagnostics for its one-factorization
  multi-RHS path.
- Shifted sparse modal and shifted sparse buckling analyses build explicit
  cache-backed inverse operators for `eigsh`; unshifted `eigsh` routes still
  use SciPy's internal operator policy.

Local evidence:

```text
python -m pytest tests/test_fe_solver_infrastructure.py tests/test_fe_solver_mass_modal.py tests/test_fe_solver_buckling.py -q -p no:cacheprovider
python scripts/run_fe_benchmarks.py --output reports/benchmarks/fe_infrastructure_benchmarks.json --markdown reports/benchmarks/fe_infrastructure_benchmarks.md
```
