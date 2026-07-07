"""Traceable nonlinear capacity workflow.

The workflow composes existing solver pieces without changing nonlinear
physics:

1. linear static solve,
2. prestress recovery,
3. eigenvalue buckling,
4. stress-free imperfection application,
5. nonlinear static capacity solve.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .assembly import solve_linear
from .buckling import BucklingResult, solve_eigenvalue_buckling
from .imperfections import EigenmodeImperfection, ImperfectionField, apply_imperfection, to_imperfection_field
from .nonlinear_static import (
    NonlinearConvergenceSettings,
    NonlinearLoadProgram,
    NonlinearStaticResult,
    solve_static_nonlinear,
)
from .recovery import ResourceConfig

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


DEFAULT_CAPACITY_WORKFLOW_PATH = Path("reports/capacity_workflow/capacity_workflow_report.json")


@dataclass(frozen=True)
class CapacityWorkflowConfig:
    """Settings for the nonlinear capacity workflow."""

    num_buckling_modes: int = 3
    buckling_mode_number: int = 1
    eigenmode_imperfection_amplitude: float = 0.0
    imperfection_dof_filter: str = "translations"
    nonlinear_num_steps: int = 10
    nonlinear_max_load_factor: float = 1.0
    nonlinear_max_iterations: int = 25
    nonlinear_tolerance: float = 1.0e-6
    nonlinear_num_layers: int = 5
    nonlinear_convergence_settings: Optional[NonlinearConvergenceSettings | str | Mapping[str, Any]] = None
    nonlinear_resource_config: Optional[ResourceConfig] = None
    mesh_min_elements_per_half_wave: int = 4
    copy_model: bool = True


@dataclass(frozen=True)
class MeshModeAdequacy:
    """Coarse diagnostic for whether a mode is represented by enough nodes."""

    status: str
    active_node_count: int
    active_element_count: int
    estimated_half_waves: int
    elements_per_half_wave: float
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "active_node_count": self.active_node_count,
            "active_element_count": self.active_element_count,
            "estimated_half_waves": self.estimated_half_waves,
            "elements_per_half_wave": self.elements_per_half_wave,
            "warnings": list(self.warnings),
        }


@dataclass
class CapacityWorkflowResult:
    """Complete result from the capacity workflow."""

    status: str
    static_displacements: np.ndarray
    static_solver_info: Dict[str, Any]
    prestress_states: Dict[int, Dict[str, float]]
    prestress_summary: Dict[str, Any]
    buckling_result: BucklingResult
    imperfection: ImperfectionField
    imperfect_model: "FEModel"
    nonlinear_result: NonlinearStaticResult
    mesh_adequacy: MeshModeAdequacy
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_load_factor(self) -> Optional[float]:
        return self.buckling_result.critical_load_factor

    @property
    def capacity_factor(self) -> float:
        return float(self.nonlinear_result.peak_load_factor)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "static_solver_status": str((self.static_solver_info.get("convergence_info") or {}).get("status", "unknown")),
            "critical_load_factor": self.critical_load_factor,
            "buckling_solver_status": self.buckling_result.solver_status,
            "capacity_factor": self.capacity_factor,
            "nonlinear_status": self.nonlinear_result.status,
            "last_converged_load_factor": self.nonlinear_result.last_converged_load_factor,
            "failure_reason": self.nonlinear_result.failure_reason,
            "imperfection": {
                "name": self.imperfection.name,
                "max_offset": self.imperfection.max_offset,
                "metadata": dict(self.imperfection.metadata),
            },
            "mesh_adequacy": self.mesh_adequacy.to_dict(),
            "prestress_summary": self.prestress_summary,
            "diagnostics": self.diagnostics,
            "buckling_result": self.buckling_result.to_dict(),
            "nonlinear_result": self.nonlinear_result.to_dict(),
        }


def _git_sha() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _recover_prestress(model: "FEModel", displacements: np.ndarray) -> Tuple[Dict[int, Dict[str, float]], Dict[str, Any]]:
    from .anystructure_fem_mode import recover_prestress_from_static_result

    return recover_prestress_from_static_result(model, displacements)


def _mode_shape_for(buckling_result: BucklingResult, mode_number: int) -> Optional[np.ndarray]:
    mode = next((item for item in buckling_result.modes if int(item.mode_number) == int(mode_number)), None)
    if mode is None:
        return None
    return np.asarray(mode.mode_shape, dtype=float).reshape(-1)


def evaluate_mode_mesh_adequacy(
    model: "FEModel",
    buckling_result: BucklingResult,
    mode_number: int = 1,
    min_elements_per_half_wave: int = 4,
    activity_tolerance: float = 0.05,
) -> MeshModeAdequacy:
    """Estimate whether the selected mode is adequately represented by the mesh.

    The diagnostic is deliberately conservative and topology-agnostic.  It
    counts nodes with translational mode amplitude above a fraction of the peak
    and estimates half-waves from sign changes along the longest coordinate
    axis among active nodes.
    """
    shape = _mode_shape_for(buckling_result, mode_number)
    if shape is None or shape.size == 0:
        return MeshModeAdequacy("no_mode", 0, 0, 0, 0.0, ("Selected buckling mode is unavailable.",))

    node_values = []
    for node_id, node in model.mesh.nodes.items():
        translation = shape[node.dofs[:3]]
        node_values.append((int(node_id), node.coords(), float(np.linalg.norm(translation)), float(translation[np.argmax(np.abs(translation))]) if translation.size else 0.0))
    peak = max((item[2] for item in node_values), default=0.0)
    if peak <= 0.0:
        return MeshModeAdequacy("zero_mode_amplitude", 0, 0, 0, 0.0, ("Selected buckling mode has no translational amplitude.",))

    active = [item for item in node_values if item[2] >= float(activity_tolerance) * peak]
    active_ids = {item[0] for item in active}
    active_element_count = sum(
        1
        for element in model.mesh.elements.values()
        if any(int(node_id) in active_ids for node_id in getattr(element, "node_ids", ()))
    )
    if not active:
        return MeshModeAdequacy("no_active_nodes", 0, active_element_count, 0, 0.0, ("No active nodes exceeded mode activity threshold.",))

    coords = np.asarray([item[1] for item in active], dtype=float)
    spans = np.ptp(coords, axis=0)
    axis = int(np.argmax(spans))
    active_sorted = sorted(active, key=lambda item: (item[1][axis], item[0]))
    signs = []
    for _node_id, _coord, _norm, signed_value in active_sorted:
        if abs(signed_value) < 1.0e-12 * peak:
            continue
        signs.append(1 if signed_value > 0.0 else -1)
    sign_changes = sum(1 for previous, current in zip(signs, signs[1:]) if previous != current)
    estimated_half_waves = max(sign_changes + 1, 1 if signs else 0)
    elements_per_half_wave = float(active_element_count / max(estimated_half_waves, 1))
    warnings = []
    if elements_per_half_wave < float(min_elements_per_half_wave):
        warnings.append(
            f"Mode mesh representation is coarse: {elements_per_half_wave:.2f} active elements per estimated half-wave "
            f"(target {min_elements_per_half_wave})."
        )
    status = "ok" if not warnings else "warning"
    return MeshModeAdequacy(
        status=status,
        active_node_count=len(active),
        active_element_count=int(active_element_count),
        estimated_half_waves=int(estimated_half_waves),
        elements_per_half_wave=elements_per_half_wave,
        warnings=tuple(warnings),
    )


def default_eigenmode_imperfection(
    buckling_result: BucklingResult,
    config: CapacityWorkflowConfig,
) -> EigenmodeImperfection:
    """Create the configured eigenmode imperfection."""
    return EigenmodeImperfection(
        buckling_result=buckling_result,
        mode_number=config.buckling_mode_number,
        amplitude=config.eigenmode_imperfection_amplitude,
        dof_filter=config.imperfection_dof_filter,
    )


def run_nonlinear_capacity_workflow(
    model: "FEModel",
    reference_load_case: "LoadCase",
    *,
    nonlinear_load_case: Optional["LoadCase"] = None,
    nonlinear_load_program: Optional[NonlinearLoadProgram] = None,
    imperfection: Optional[Any] = None,
    config: Optional[CapacityWorkflowConfig] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> CapacityWorkflowResult:
    """Run linear static -> buckling -> imperfection -> nonlinear capacity."""
    config = config or CapacityWorkflowConfig()
    start = time.perf_counter()
    static_displacements, static_info = solve_linear(model, reference_load_case)
    static_status = str((static_info.get("convergence_info") or {}).get("status", "unknown"))
    if static_status != "converged":
        raise RuntimeError(f"Static prestress solve did not converge: {static_status}")

    prestress_states, prestress_summary = _recover_prestress(model, static_displacements)
    buckling = solve_eigenvalue_buckling(model, prestress_states, num_modes=config.num_buckling_modes)
    if not buckling.modes:
        raise RuntimeError(f"Buckling solve returned no usable modes: {buckling.solver_status}")

    mesh_adequacy = evaluate_mode_mesh_adequacy(
        model,
        buckling,
        mode_number=config.buckling_mode_number,
        min_elements_per_half_wave=config.mesh_min_elements_per_half_wave,
    )
    selected_imperfection = imperfection
    if selected_imperfection is None:
        selected_imperfection = default_eigenmode_imperfection(buckling, config)
    imperfection_field = to_imperfection_field(model, selected_imperfection)
    imperfect_model = apply_imperfection(model, imperfection_field, copy_model=config.copy_model)

    if nonlinear_load_program is not None:
        nonlinear_result = solve_static_nonlinear(
            imperfect_model,
            load_program=nonlinear_load_program,
            max_load_factor=config.nonlinear_max_load_factor,
            num_steps=config.nonlinear_num_steps,
            max_iterations=config.nonlinear_max_iterations,
            tolerance=config.nonlinear_tolerance,
            num_layers=config.nonlinear_num_layers,
            convergence_settings=config.nonlinear_convergence_settings,
            resource_config=config.nonlinear_resource_config,
            status_callback=status_callback,
        )
    else:
        nonlinear_result = solve_static_nonlinear(
            imperfect_model,
            load_case=nonlinear_load_case or reference_load_case,
            max_load_factor=config.nonlinear_max_load_factor,
            num_steps=config.nonlinear_num_steps,
            max_iterations=config.nonlinear_max_iterations,
            tolerance=config.nonlinear_tolerance,
            num_layers=config.nonlinear_num_layers,
            convergence_settings=config.nonlinear_convergence_settings,
            resource_config=config.nonlinear_resource_config,
            status_callback=status_callback,
        )

    status = "completed" if nonlinear_result.converged else "nonlinear_not_converged"
    diagnostics = {
        "workflow_seconds": float(time.perf_counter() - start),
        "model_name": model.name,
        "node_count": int(model.mesh.num_nodes),
        "element_count": int(model.mesh.num_elements),
        "reference_load_case": getattr(reference_load_case, "name", None),
        "nonlinear_load_case": getattr(nonlinear_load_case or reference_load_case, "name", None),
        "config": config.__dict__,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "commit": _git_sha(),
        },
    }
    if mesh_adequacy.warnings:
        diagnostics["warnings"] = list(mesh_adequacy.warnings)

    return CapacityWorkflowResult(
        status=status,
        static_displacements=static_displacements,
        static_solver_info=static_info,
        prestress_states=prestress_states,
        prestress_summary=prestress_summary,
        buckling_result=buckling,
        imperfection=imperfection_field,
        imperfect_model=imperfect_model,
        nonlinear_result=nonlinear_result,
        mesh_adequacy=mesh_adequacy,
        diagnostics=diagnostics,
    )


def write_capacity_workflow_report(
    result: CapacityWorkflowResult,
    path: Path | str = DEFAULT_CAPACITY_WORKFLOW_PATH,
) -> Path:
    """Write a capacity workflow result report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return output


def run_capacity_workflow_from_builder(
    model_builder: Callable[[], "FEModel"],
    load_case_builder: Callable[["FEModel"], "LoadCase"],
    *,
    config: Optional[CapacityWorkflowConfig] = None,
    report_path: Path | str = DEFAULT_CAPACITY_WORKFLOW_PATH,
) -> CapacityWorkflowResult:
    """Convenience helper for scripted examples/tests."""
    model = model_builder()
    load_case = load_case_builder(model)
    result = run_nonlinear_capacity_workflow(model, load_case, config=config)
    write_capacity_workflow_report(result, report_path)
    return result
