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
- ``dynamics``: linear Newmark transient response with pressure patches
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
- Linear transient dynamics uses separable K/M/F assembly and the same
  constraint transformation as static analysis.
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
from .buckling import BucklingMode, BucklingResult, solve_eigenvalue_buckling
from .nonlinear import NonlinearLimitPointResult, NonlinearLoadStep, solve_nonlinear_load_stepping
from .capacity_workflow import (
    DEFAULT_CAPACITY_WORKFLOW_PATH,
    CapacityWorkflowConfig,
    CapacityWorkflowResult,
    MeshModeAdequacy,
    default_eigenmode_imperfection,
    evaluate_mode_mesh_adequacy,
    run_capacity_workflow_from_builder,
    run_nonlinear_capacity_workflow,
    write_capacity_workflow_report,
)
from .cases import (
    AnalysisCase,
    LoadCaseRef,
    PrestressCase,
    ResultCase,
    load_case_ref,
    load_signature_from_info,
    make_result_case,
    matrix_signature_from_info,
    solver_backend_from_info,
)
from .dynamics import (
    PressurePatch,
    TransientConfig,
    TransientResult,
    assemble_pressure_patch_load_vector,
    solve_transient_newmark,
)
from .recovery import (
    MemoryEstimate,
    RecoveryExecutionReport,
    RecoveryConfig,
    ResourcePolicyError,
    ResourceConfig,
    default_recovery_config,
    enforce_memory_limit,
    estimate_model_memory,
    filter_reactions,
    recover_element_stresses,
    recover_element_stresses_with_report,
    recovery_metadata,
    select_node_displacements,
)
from .recovery_policy import (
    DEFAULT_RECOVERY_POLICY_PATH,
    generate_recovery_policy_report,
    write_recovery_policy_report,
)
from .material_curves import (
    DNVC208MaterialCurve,
    FiberSectionPlasticityConfig,
    curve_from_properties,
    dnv_c208_steel_curve,
)
from .imperfections import (
    CompositeImperfection,
    EigenmodeImperfection,
    ImperfectionCalibrationResult,
    ImperfectionField,
    StandardImperfection,
    apply_imperfection,
    calibrate_imperfection_amplitude,
    imperfection_from_buckling_mode,
    standard_flange_twist,
    standard_member_bow,
    standard_plate_mode,
)
from .nonlinear_static import (
    DisplacementControl,
    NonlinearLoadProgram,
    NonlinearLoadStage,
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
    assemble_damping_matrix,
    assemble_geometric_stiffness_matrix,
    assemble_load_matrix,
    assemble_load_vector,
    assemble_mass_matrix,
    assemble_stiffness_matrix,
    assemble_system,
)
from .assembly import (
    build_constraint_transformation,
    compute_constraint_force_diagnostics,
    reconstruct_full_solution,
    solve_linear,
    solve_linear_many,
    solve_nonlinear,
)
from .linalg import (
    FactorizationCache,
    FactorizationHandle,
    MatrixClass,
    SparseSolverBackend,
    cached_inverse_operator,
    factorize,
    factorize_cached,
    solve_many,
    sparse_matrix_signature,
)
from .mass_properties import MassProperties, calculate_mass_properties, element_mass_points
from .modal import ModalMode, ModalResult, solve_free_vibration
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
    "DEFAULT_BUCKLING_VALIDITY_PATH",
    "generate_buckling_validity_report",
    "write_buckling_validity_report",
    # Nonlinear stability
    "NonlinearLimitPointResult",
    "NonlinearLoadStep",
    "solve_nonlinear_load_stepping",
    # Case/provenance model
    "AnalysisCase",
    "LoadCaseRef",
    "PrestressCase",
    "ResultCase",
    "load_case_ref",
    "load_signature_from_info",
    "make_result_case",
    "matrix_signature_from_info",
    "solver_backend_from_info",
    "DEFAULT_BENCHMARK_PATH",
    "run_infrastructure_benchmarks",
    "write_benchmark_report",
    "DEFAULT_BEAM_SHELL_VERIFICATION_PATH",
    "VerificationCase",
    "VerificationCaseResult",
    "run_beam_shell_verification",
    "verification_manifest_cases",
    "write_beam_shell_verification_report",
    "DEFAULT_BEAM_VALIDITY_PATH",
    "corotational_axial_extension_metric",
    "corotational_rigid_rotation_metric",
    "generate_beam_validity_report",
    "write_beam_validity_report",
    "DEFAULT_CAPACITY_WORKFLOW_PATH",
    "CapacityWorkflowConfig",
    "CapacityWorkflowResult",
    "MeshModeAdequacy",
    "default_eigenmode_imperfection",
    "evaluate_mode_mesh_adequacy",
    "run_capacity_workflow_from_builder",
    "run_nonlinear_capacity_workflow",
    "write_capacity_workflow_report",
    # Linear transient dynamics / slamming v1
    "PressurePatch",
    "TransientConfig",
    "TransientResult",
    "assemble_pressure_patch_load_vector",
    "solve_transient_newmark",
    "DEFAULT_ELEMENT_QUALIFICATION_PATH",
    "beam_qualification_metrics",
    "generate_element_qualification_report",
    "q4_q8_convergence_cost_sweep",
    "q8_free_mode_metric",
    "q8_mass_metric",
    "q8_patch_metric",
    "reference_q8_geometries",
    "write_element_qualification_report",
    "DEFAULT_EXTERNAL_REFERENCE_PATH",
    "ExternalReferenceCase",
    "generate_external_reference_cases",
    "generate_external_reference_report",
    "write_calculix_input_deck",
    "write_external_reference_report",
    "DEFAULT_PLASTICITY_QUALIFICATION_PATH",
    "dnv_curve_metric",
    "element_tangent_metrics",
    "generate_plasticity_qualification_report",
    "material_point_path_metrics",
    "reference_plastic_curve",
    "write_plasticity_qualification_report",
    "yield_function_residual",
    "MemoryEstimate",
    "RecoveryExecutionReport",
    "RecoveryConfig",
    "ResourcePolicyError",
    "ResourceConfig",
    "default_recovery_config",
    "enforce_memory_limit",
    "estimate_model_memory",
    "filter_reactions",
    "recover_element_stresses",
    "recover_element_stresses_with_report",
    "recovery_metadata",
    "select_node_displacements",
    "DEFAULT_RECOVERY_POLICY_PATH",
    "generate_recovery_policy_report",
    "write_recovery_policy_report",
    # Incremental geometric/material nonlinear statics
    "DNVC208MaterialCurve",
    "FiberSectionPlasticityConfig",
    "curve_from_properties",
    "dnv_c208_steel_curve",
    "CompositeImperfection",
    "EigenmodeImperfection",
    "ImperfectionCalibrationResult",
    "ImperfectionField",
    "StandardImperfection",
    "apply_imperfection",
    "calibrate_imperfection_amplitude",
    "imperfection_from_buckling_mode",
    "standard_flange_twist",
    "standard_member_bow",
    "standard_plate_mode",
    "DisplacementControl",
    "NonlinearLoadProgram",
    "NonlinearLoadStage",
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
    "assemble_damping_matrix",
    "assemble_geometric_stiffness_matrix",
    "assemble_load_matrix",
    "assemble_load_vector",
    "assemble_mass_matrix",
    "assemble_stiffness_matrix",
    "assemble_system",
    "build_constraint_transformation",
    "compute_constraint_force_diagnostics",
    "reconstruct_full_solution",
    "solve_linear",
    "solve_linear_many",
    "solve_nonlinear",
    "MatrixClass",
    "SparseSolverBackend",
    "FactorizationCache",
    "FactorizationHandle",
    "cached_inverse_operator",
    "factorize",
    "factorize_cached",
    "solve_many",
    "sparse_matrix_signature",
    "MassProperties",
    "calculate_mass_properties",
    "element_mass_points",
    "ModalMode",
    "ModalResult",
    "solve_free_vibration",
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
    # S4 validity
    "DEFAULT_S4_VALIDITY_PATH",
    "bending_patch_metric",
    "free_element_mode_metric",
    "generate_s4_validity_report",
    "membrane_patch_metric",
    "reference_s4_geometries",
    "s4_s8_comparison",
    "shear_patch_metric",
    "thin_plate_locking_sweep",
    "write_s4_validity_report",
]
