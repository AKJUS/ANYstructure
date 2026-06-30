"""Runtime-only FE backend exports for ANYstructure.

This package mirrors the solver runtime from ANYintelligent, but deliberately
excludes verification, QC, benchmark and development modules.
"""

from __future__ import annotations

from .anystructure_fem_mode import (
    AnyStructureFEMConfig,
    build_fe_model_from_generated_geometry,
    build_symmetric_load_case,
    recover_prestress_from_static_result,
)
from .arc_length import ArcLengthControl, ArcLengthResult, solve_static_arc_length
from .boundary import BoundaryCondition, LoadCase
from .buckling import BucklingMode, BucklingResult, solve_eigenvalue_buckling
from .capacity_workflow import (
    CapacityWorkflowConfig,
    CapacityWorkflowResult,
    MeshModeAdequacy,
    default_eigenmode_imperfection,
    evaluate_mode_mesh_adequacy,
    run_capacity_workflow_from_builder,
    run_nonlinear_capacity_workflow,
)
from .dynamics import PressurePatch, TransientConfig, TransientResult, solve_transient_newmark
from .elements import BeamElement, CoupledBeamShellElement, QuadraticBeamElement, ShellElement, create_element
from .fe_core import DOFManager, FEModel, FEMesh, Material, Node
from .imperfections import (
    CompositeImperfection,
    EigenmodeImperfection,
    ImperfectionField,
    StandardImperfection,
    apply_imperfection,
)
from .material_curves import DNVC208MaterialCurve, curve_from_properties, dnv_c208_steel_curve
from .nonlinear import NonlinearLimitPointResult, NonlinearLoadStep, solve_nonlinear_load_stepping
from .nonlinear_static import (
    DisplacementControl,
    NonlinearConvergenceSettings,
    NonlinearLoadProgram,
    NonlinearLoadStage,
    NonlinearStaticResult,
    NonlinearStaticStep,
    solve_static_nonlinear,
)
from .recovery import RecoveryConfig, ResourceConfig
from .validation import load_case_resultant, validate_production_model

__version__ = "0.1.0-anystructure-runtime"

__all__ = [
    "ArcLengthControl",
    "ArcLengthResult",
    "AnyStructureFEMConfig",
    "BeamElement",
    "BoundaryCondition",
    "BucklingMode",
    "BucklingResult",
    "CapacityWorkflowConfig",
    "CapacityWorkflowResult",
    "CompositeImperfection",
    "CoupledBeamShellElement",
    "DNVC208MaterialCurve",
    "DOFManager",
    "DisplacementControl",
    "EigenmodeImperfection",
    "FEModel",
    "FEMesh",
    "ImperfectionField",
    "LoadCase",
    "Material",
    "MeshModeAdequacy",
    "Node",
    "NonlinearConvergenceSettings",
    "NonlinearLimitPointResult",
    "NonlinearLoadProgram",
    "NonlinearLoadStage",
    "NonlinearLoadStep",
    "NonlinearStaticResult",
    "NonlinearStaticStep",
    "PressurePatch",
    "QuadraticBeamElement",
    "RecoveryConfig",
    "ResourceConfig",
    "ShellElement",
    "StandardImperfection",
    "TransientConfig",
    "TransientResult",
    "apply_imperfection",
    "build_fe_model_from_generated_geometry",
    "build_symmetric_load_case",
    "create_element",
    "curve_from_properties",
    "default_eigenmode_imperfection",
    "dnv_c208_steel_curve",
    "evaluate_mode_mesh_adequacy",
    "load_case_resultant",
    "recover_prestress_from_static_result",
    "run_capacity_workflow_from_builder",
    "run_nonlinear_capacity_workflow",
    "solve_eigenvalue_buckling",
    "solve_nonlinear_load_stepping",
    "solve_static_arc_length",
    "solve_static_nonlinear",
    "solve_transient_newmark",
    "validate_production_model",
]
