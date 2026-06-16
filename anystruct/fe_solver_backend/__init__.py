"""
Modular FE solver package for ANYstructure-style stiffened panels.

Production solver path
----------------------
Use this package, not the legacy top-level ``fea_solver.py`` prototype.  The
package is intentionally split by responsibility:

- ``fe_core``: model, mesh, node, material and DOF bookkeeping
- ``elements``: shell, beam and MPC/coupling element formulations
- ``boundary``: supports, nodal loads, pressure loads and load cases
- ``matrix_assembly``: explicit K/M/F assembly APIs
- ``buckling``: geometric stiffness and linear eigenvalue buckling helpers
- ``nonlinear``: load stepping with tangent-stability limit-point detection
- ``anystructure_fem_mode``: full generated-geometry FEM mode workflow
- ``assembly``: constraint transformation, nullspace solve and solver routines
- ``mesh_gen``: limited ANYstructure-oriented mesh generation
- ``results``: result objects, stress extraction and post-processing helpers
- ``validation``: verification utilities and benchmark helpers
- ``reference_cases``: local and upstream CalculiX reference case discovery
- ``shell_benchmarks``: internal shell benchmark runners

Architecture invariants
-----------------------
- Six DOFs per node: ux, uy, uz, rx, ry, rz.
- SI units internally: metres, Newtons, Pascals.
- Beam-shell eccentricity is handled with MPC/constraint transformation, not
  shared-node eccentricity hacks and not penalty springs.
- Boundary conditions and MPCs are eliminated through the same transformation
  machinery before solving.
- Free-free static problems use explicit rigid-body/nullspace handling instead
  of artificial support stiffness.
- Stiffness, mass and load assembly must remain separable for modal and
  buckling work.
- Linear buckling solves use ``K phi = lambda KG phi`` with positive ``KG`` for
  destabilizing reference compression.
- Nonlinear stability checks stop near the first tangent-stiffness limit point;
  full post-buckling continuation is out of scope.
"""

from .fe_core import DOFManager, FEMesh, FEModel, Material, Node
from .elements import BeamElement, CoupledBeamShellElement, QuadraticBeamElement, ShellElement, create_element
from .boundary import (
    BoundaryCondition,
    FixedSupport,
    InPlaneLoad,
    LoadCase,
    LoadCombination,
    PinnedSupport,
    RollerSupport,
    SymmetryBC,
)
from .cylinder_benchmarks import (
    CylinderBenchmarkConfig,
    CylinderBenchmarkResult,
    CylinderNominalStress,
    CylinderStressStatistics,
    build_cylindrical_shell_benchmark_model,
    nominal_cylinder_membrane_stress,
    run_cylindrical_shell_benchmark,
)
from .buckling import BucklingMode, BucklingResult, solve_eigenvalue_buckling
from .nonlinear import NonlinearLimitPointResult, NonlinearLoadStep, solve_nonlinear_load_stepping
from .material_curves import DNVC208MaterialCurve, curve_from_properties
from .nonlinear_static import (
    NonlinearStaticResult,
    NonlinearStaticStep,
    solve_static_nonlinear,
)
from .anystructure_fem_mode import (
    AnyStructureFEMConfig,
    AnyStructureFEMResult,
    build_fe_model_from_generated_geometry,
    build_symmetric_load_case,
    idealize_generated_geometry_members,
    recover_prestress_from_static_result,
    run_anystructure_fem_mode,
)
from .matrix_assembly import (
    AssemblyError,
    assemble_geometric_stiffness_matrix,
    assemble_load_vector,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    assemble_system,
)
from .assembly import (
    build_constraint_transformation,
    reconstruct_full_solution,
    solve_linear,
    solve_nonlinear,
)
from .mesh_gen import (
    MeshConfig,
    PanelGeometry,
    StiffenerCrossSection,
    generate_beam_mesh,
    generate_simple_panel_mesh,
    generate_stiffened_panel_mesh,
    verify_mesh_quality,
)
from .results import (
    DisplacementResult,
    FEResult,
    StressResult,
    compare_with_analytical,
    create_fe_result,
    post_process_results,
)
from .validation import (
    LoadResultant,
    ShellPatchSummary,
    dof_order_signature,
    load_case_resultant,
    load_vector_resultant,
    max_abs,
    mpc_constraint_residuals,
    nullspace_diagnostics,
    shell_element_patch_summary,
)
from .reference_cases import (
    CalculixReferenceCase,
    ShellConvergencePoint,
    ShellConvergenceTable,
    classify_reference_case_from_nodes,
    discover_calculix_reference_cases,
    discover_calculix_shell_convergence_tables,
    parse_calculix_shell_convergence_file,
    summarize_inp_geometry,
    upstream_calculix_reference_manifest,
    upstream_calculix_shell_reference_values,
)
from .shell_benchmarks import (
    ShellBenchmarkComparison,
    ShellBenchmarkComparisonPoint,
    ShellBenchmarkResult,
    compare_shell_benchmark_to_reference,
    run_simple_supported_shell_benchmark,
    run_simple_supported_shell_convergence,
    shell_benchmark_results_to_convergence_table,
    write_internal_shell_convergence_table,
)

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "DOFManager",
    "FEMesh",
    "FEModel",
    "Material",
    "Node",
    # Elements
    "BeamElement",
    "CoupledBeamShellElement",
    "QuadraticBeamElement",
    "ShellElement",
    "create_element",
    # Boundary and loads
    "BoundaryCondition",
    "FixedSupport",
    "InPlaneLoad",
    "LoadCase",
    "LoadCombination",
    "PinnedSupport",
    "RollerSupport",
    "SymmetryBC",
    # Cylinder benchmarks
    "CylinderBenchmarkConfig",
    "CylinderBenchmarkResult",
    "CylinderNominalStress",
    "CylinderStressStatistics",
    "build_cylindrical_shell_benchmark_model",
    "nominal_cylinder_membrane_stress",
    "run_cylindrical_shell_benchmark",
    # Buckling
    "BucklingMode",
    "BucklingResult",
    "solve_eigenvalue_buckling",
    # Nonlinear stability
    "NonlinearLimitPointResult",
    "NonlinearLoadStep",
    "solve_nonlinear_load_stepping",
    # Incremental geometric/material nonlinear statics
    "DNVC208MaterialCurve",
    "curve_from_properties",
    "NonlinearStaticResult",
    "NonlinearStaticStep",
    "solve_static_nonlinear",
    # ANYstructure generated-geometry FEM mode
    "AnyStructureFEMConfig",
    "AnyStructureFEMResult",
    "build_fe_model_from_generated_geometry",
    "build_symmetric_load_case",
    "idealize_generated_geometry_members",
    "recover_prestress_from_static_result",
    "run_anystructure_fem_mode",
    # Assembly and solving
    "AssemblyError",
    "assemble_geometric_stiffness_matrix",
    "assemble_load_vector",
    "assemble_mass_matrix",
    "assemble_stiffness_matrix",
    "assemble_system",
    "build_constraint_transformation",
    "reconstruct_full_solution",
    "solve_linear",
    "solve_nonlinear",
    # Mesh generation
    "MeshConfig",
    "PanelGeometry",
    "StiffenerCrossSection",
    "generate_beam_mesh",
    "generate_simple_panel_mesh",
    "generate_stiffened_panel_mesh",
    "verify_mesh_quality",
    # Results
    "DisplacementResult",
    "FEResult",
    "StressResult",
    "compare_with_analytical",
    "create_fe_result",
    "post_process_results",
    # Validation helpers
    "LoadResultant",
    "ShellPatchSummary",
    "dof_order_signature",
    "load_case_resultant",
    "load_vector_resultant",
    "max_abs",
    "mpc_constraint_residuals",
    "nullspace_diagnostics",
    "shell_element_patch_summary",
    # Reference cases
    "CalculixReferenceCase",
    "ShellConvergencePoint",
    "ShellConvergenceTable",
    "classify_reference_case_from_nodes",
    "discover_calculix_reference_cases",
    "discover_calculix_shell_convergence_tables",
    "parse_calculix_shell_convergence_file",
    "summarize_inp_geometry",
    "upstream_calculix_reference_manifest",
    "upstream_calculix_shell_reference_values",
    # Shell benchmarks
    "ShellBenchmarkComparison",
    "ShellBenchmarkComparisonPoint",
    "ShellBenchmarkResult",
    "compare_shell_benchmark_to_reference",
    "run_simple_supported_shell_benchmark",
    "run_simple_supported_shell_convergence",
    "shell_benchmark_results_to_convergence_table",
    "write_internal_shell_convergence_table",
]
