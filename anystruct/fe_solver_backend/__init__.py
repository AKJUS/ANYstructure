"""Runtime-only FE solver backend embedded in ANYstructure.

This package intentionally exposes only solver runtime APIs required by
``anystruct.fe_solver``. Non-runtime helpers belong in ANYintelligent and are not shipped in this embedded backend.

Imports are lazy so opening ANYstructure does not import every solver module.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Tuple

__version__ = "0.1.0-anystructure-runtime"

_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Core model and elements
    "DOFManager": ("fe_core", "DOFManager"),
    "FEMesh": ("fe_core", "FEMesh"),
    "FEModel": ("fe_core", "FEModel"),
    "Material": ("fe_core", "Material"),
    "Node": ("fe_core", "Node"),
    "BeamElement": ("elements", "BeamElement"),
    "CoupledBeamShellElement": ("elements", "CoupledBeamShellElement"),
    "QuadraticBeamElement": ("elements", "QuadraticBeamElement"),
    "ShellElement": ("elements", "ShellElement"),
    "create_element": ("elements", "create_element"),
    # Boundary and loads
    "BoundaryCondition": ("boundary", "BoundaryCondition"),
    "FixedSupport": ("boundary", "FixedSupport"),
    "InPlaneLoad": ("boundary", "InPlaneLoad"),
    "LoadCase": ("boundary", "LoadCase"),
    "LoadCombination": ("boundary", "LoadCombination"),
    "PinnedSupport": ("boundary", "PinnedSupport"),
    "RollerSupport": ("boundary", "RollerSupport"),
    "SymmetryBC": ("boundary", "SymmetryBC"),
    # Assembly and solvers
    "build_constraint_transformation": ("assembly", "build_constraint_transformation"),
    "compute_constraint_force_diagnostics": ("assembly", "compute_constraint_force_diagnostics"),
    "compute_stresses": ("assembly", "compute_stresses"),
    "reconstruct_full_solution": ("assembly", "reconstruct_full_solution"),
    "solve_linear": ("assembly", "solve_linear"),
    "solve_linear_many": ("assembly", "solve_linear_many"),
    "solve_nonlinear": ("assembly", "solve_nonlinear"),
    "AssemblyError": ("matrix_assembly", "AssemblyError"),
    "assemble_damping_matrix": ("matrix_assembly", "assemble_damping_matrix"),
    "assemble_geometric_stiffness_matrix": ("matrix_assembly", "assemble_geometric_stiffness_matrix"),
    "assemble_load_matrix": ("matrix_assembly", "assemble_load_matrix"),
    "assemble_load_vector": ("matrix_assembly", "assemble_load_vector"),
    "assemble_mass_matrix": ("matrix_assembly", "assemble_mass_matrix"),
    "assemble_stiffness_matrix": ("matrix_assembly", "assemble_stiffness_matrix"),
    "assemble_system": ("matrix_assembly", "assemble_system"),
    "MatrixClass": ("linalg", "MatrixClass"),
    "factorize": ("linalg", "factorize"),
    "solve_many": ("linalg", "solve_many"),
    # Analysis APIs
    "BucklingMode": ("buckling", "BucklingMode"),
    "BucklingResult": ("buckling", "BucklingResult"),
    "solve_eigenvalue_buckling": ("buckling", "solve_eigenvalue_buckling"),
    "ModalMode": ("modal", "ModalMode"),
    "ModalResult": ("modal", "ModalResult"),
    "solve_free_vibration": ("modal", "solve_free_vibration"),
    "NonlinearLimitPointResult": ("nonlinear", "NonlinearLimitPointResult"),
    "NonlinearLoadStep": ("nonlinear", "NonlinearLoadStep"),
    "solve_nonlinear_load_stepping": ("nonlinear", "solve_nonlinear_load_stepping"),
    "DisplacementControl": ("nonlinear_static", "DisplacementControl"),
    "NonlinearConvergenceSettings": ("nonlinear_static", "NonlinearConvergenceSettings"),
    "NonlinearLoadProgram": ("nonlinear_static", "NonlinearLoadProgram"),
    "NonlinearLoadStage": ("nonlinear_static", "NonlinearLoadStage"),
    "NonlinearStaticResult": ("nonlinear_static", "NonlinearStaticResult"),
    "NonlinearStaticStep": ("nonlinear_static", "NonlinearStaticStep"),
    "solve_static_nonlinear": ("nonlinear_static", "solve_static_nonlinear"),
    "ArcLengthControl": ("arc_length", "ArcLengthControl"),
    "ArcLengthResult": ("arc_length", "ArcLengthResult"),
    "ArcLengthStep": ("arc_length", "ArcLengthStep"),
    "solve_static_arc_length": ("arc_length", "solve_static_arc_length"),
    "PressurePatch": ("dynamics", "PressurePatch"),
    "TransientConfig": ("dynamics", "TransientConfig"),
    "TransientResult": ("dynamics", "TransientResult"),
    "assemble_pressure_patch_load_vector": ("dynamics", "assemble_pressure_patch_load_vector"),
    "solve_transient_newmark": ("dynamics", "solve_transient_newmark"),
    # Runtime model building and recovery
    "MeshConfig": ("mesh_gen", "MeshConfig"),
    "PanelGeometry": ("mesh_gen", "PanelGeometry"),
    "StiffenerCrossSection": ("mesh_gen", "StiffenerCrossSection"),
    "generate_beam_mesh": ("mesh_gen", "generate_beam_mesh"),
    "generate_simple_panel_mesh": ("mesh_gen", "generate_simple_panel_mesh"),
    "generate_stiffened_panel_mesh": ("mesh_gen", "generate_stiffened_panel_mesh"),
    "verify_mesh_quality": ("mesh_gen", "verify_mesh_quality"),
    "MassProperties": ("mass_properties", "MassProperties"),
    "calculate_mass_properties": ("mass_properties", "calculate_mass_properties"),
    "element_mass_points": ("mass_properties", "element_mass_points"),
    "DisplacementResult": ("results", "DisplacementResult"),
    "FEResult": ("results", "FEResult"),
    "StressResult": ("results", "StressResult"),
    "compare_with_analytical": ("results", "compare_with_analytical"),
    "create_fe_result": ("results", "create_fe_result"),
    "RecoveryConfig": ("recovery", "RecoveryConfig"),
    "ResourceConfig": ("recovery", "ResourceConfig"),
    "recover_element_stresses": ("recovery", "recover_element_stresses"),
    "recover_element_stresses_with_report": ("recovery", "recover_element_stresses_with_report"),
    # Materials, imperfections, capacity workflow
    "DNVC208MaterialCurve": ("material_curves", "DNVC208MaterialCurve"),
    "FiberSectionPlasticityConfig": ("material_curves", "FiberSectionPlasticityConfig"),
    "curve_from_properties": ("material_curves", "curve_from_properties"),
    "dnv_c208_steel_curve": ("material_curves", "dnv_c208_steel_curve"),
    "CompositeImperfection": ("imperfections", "CompositeImperfection"),
    "EigenmodeImperfection": ("imperfections", "EigenmodeImperfection"),
    "ImperfectionCalibrationResult": ("imperfections", "ImperfectionCalibrationResult"),
    "ImperfectionField": ("imperfections", "ImperfectionField"),
    "StandardImperfection": ("imperfections", "StandardImperfection"),
    "apply_imperfection": ("imperfections", "apply_imperfection"),
    "calibrate_imperfection_amplitude": ("imperfections", "calibrate_imperfection_amplitude"),
    "imperfection_from_buckling_mode": ("imperfections", "imperfection_from_buckling_mode"),
    "standard_flange_twist": ("imperfections", "standard_flange_twist"),
    "standard_member_bow": ("imperfections", "standard_member_bow"),
    "standard_plate_mode": ("imperfections", "standard_plate_mode"),
    "CapacityWorkflowConfig": ("capacity_workflow", "CapacityWorkflowConfig"),
    "CapacityWorkflowResult": ("capacity_workflow", "CapacityWorkflowResult"),
    "MeshModeAdequacy": ("capacity_workflow", "MeshModeAdequacy"),
    "default_eigenmode_imperfection": ("capacity_workflow", "default_eigenmode_imperfection"),
    "evaluate_mode_mesh_adequacy": ("capacity_workflow", "evaluate_mode_mesh_adequacy"),
    "run_capacity_workflow_from_builder": ("capacity_workflow", "run_capacity_workflow_from_builder"),
    "run_nonlinear_capacity_workflow": ("capacity_workflow", "run_nonlinear_capacity_workflow"),
    # ANYstructure generated-geometry workflow
    "AnyStructureFEMConfig": ("anystructure_fem_mode", "AnyStructureFEMConfig"),
    "AnyStructureFEMResult": ("anystructure_fem_mode", "AnyStructureFEMResult"),
    "build_fe_model_from_generated_geometry": ("anystructure_fem_mode", "build_fe_model_from_generated_geometry"),
    "build_symmetric_load_case": ("anystructure_fem_mode", "build_symmetric_load_case"),
    "idealize_generated_geometry_members": ("anystructure_fem_mode", "idealize_generated_geometry_members"),
    "recover_prestress_from_static_result": ("anystructure_fem_mode", "recover_prestress_from_static_result"),
    "run_anystructure_fem_mode": ("anystructure_fem_mode", "run_anystructure_fem_mode"),
    # Runtime load resultants
    "LoadResultant": ("validation", "LoadResultant"),
    "load_case_resultant": ("validation", "load_case_resultant"),
    "load_vector_resultant": ("validation", "load_vector_resultant"),
    # SESAM formatted FEM document import/export
    "FemDiagnostic": ("sesam_fem", "FemDiagnostic"),
    "FemRawRecord": ("sesam_fem", "FemRawRecord"),
    "SesamFemDocument": ("sesam_fem", "SesamFemDocument"),
    "SesamFemError": ("sesam_fem", "SesamFemError"),
    "SesamFemExportReport": ("sesam_fem", "SesamFemExportReport"),
    "SesamFemImportResult": ("sesam_fem", "SesamFemImportResult"),
    "export_sesam_fem": ("sesam_fem", "export_sesam_fem"),
    "import_sesam_fem": ("sesam_fem", "import_sesam_fem"),
    "read_raw_records": ("sesam_fem", "read_raw_records"),
    "read_sesam_fem_document": ("sesam_fem", "read_sesam_fem_document"),
    "validate_sesam_fem_document": ("sesam_fem", "validate_sesam_fem_document"),
    "write_sesam_fem_document": ("sesam_fem", "write_sesam_fem_document"),
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))




