"""Runtime FE solver backend package used by ANYstructure.

This runtime package exposes analysis/model APIs only.  Development-only
verification, benchmark, and report generators are intentionally not imported
here; they remain in ANYintelligent.
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
from .assembly import compute_stresses, solve_linear
from .buckling import BucklingMode, BucklingResult, solve_eigenvalue_buckling
from .nonlinear import NonlinearLimitPointResult, NonlinearLoadStep, solve_nonlinear_load_stepping
from .nonlinear_static import (
    DisplacementControl,
    NonlinearLoadProgram,
    NonlinearLoadStage,
    solve_static_nonlinear,
)
from .arc_length import ArcLengthControl, ArcLengthResult, solve_static_arc_length
from .anystructure_fem_mode import (
    AnyStructureFEMConfig,
    build_fe_model_from_generated_geometry,
    build_symmetric_load_case,
    recover_prestress_from_static_result,
)
from .capacity_workflow import (
    CapacityWorkflowConfig,
    CapacityWorkflowResult,
    MeshModeAdequacy,
    default_eigenmode_imperfection,
    evaluate_mode_mesh_adequacy,
    run_capacity_workflow_from_builder,
    run_nonlinear_capacity_workflow,
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
from .contact import (
    NonlinearTransientConfig,
    RigidSphereImpact,
    SphereContactConfig,
    SphereContactRecord,
    SphereImpactResult,
    assemble_sphere_contact_load_vector,
    recommend_sphere_contact_penalty,
    solve_transient_sphere_impact,
    validate_contact_configuration,
)
from .fracture import (
    DeletedElementRecord,
    ElementDeletionConfig,
    FractureConfig,
    ImpactDamageConfig,
    ImpactFractureConfig,
    PlasticImpactDamageConfig,
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
from .kernel_warmup import warm_fe_solver_kernels
from .linalg import FactorizationHandle, MatrixClass, SparseSolverBackend, factorize, solve_many
from .material_curves import DNVC208MaterialCurve, FiberSectionPlasticityConfig, curve_from_properties, dnv_c208_steel_curve
from .matrix_assembly import assemble_damping_matrix, assemble_geometric_stiffness_matrix, assemble_load_matrix, assemble_load_vector, assemble_mass_matrix, assemble_stiffness_matrix
from .modal import ModalMode, ModalResult, solve_free_vibration
from .recovery import (
    MemoryEstimate,
    RecoveryConfig,
    RecoveryExecutionReport,
    ResourceConfig,
    ResourcePolicyError,
    default_recovery_config,
    enforce_memory_limit,
    estimate_model_memory,
    filter_reactions,
    recover_element_stresses,
    recover_element_stresses_with_report,
    recovery_metadata,
    select_node_displacements,
)
from .results import FEResult, StressResult, DisplacementResult, create_fe_result, post_process_results
from .validation import ProductionValidationIssue, ProductionValidationReport, load_case_resultant, load_vector_resultant

__all__ = [name for name in globals() if not name.startswith('_')]