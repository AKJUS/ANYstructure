"""Incremental Newton-Raphson solver with geometric and material nonlinearity.

Geometric nonlinearity: total-Lagrangian von Karman kinematics in the shell
elements (membrane-bending coupling from transverse-deflection gradients,
initial-stress stiffness from the current membrane resultants) and a
consistent beam-column axial coupling in the 2-node beam.

Material nonlinearity: layered J2 plane-stress plasticity in the shells with
the isotropic hardening curve attached to the material
(``Material.hardening_curve``, e.g. a DNV-RP-C208 curve from
:mod:`fe_solver.material_curves`).  Materials without a curve stay elastic.

Solution strategy (chosen for speed):

* full Newton-Raphson per load increment (quadratic-ish convergence, one
  sparse factorization per iteration),
* vectorized element kernels with cached reference geometry,
* COO-triplet assembly of tangent and internal force in a single element loop,
* adaptive load stepping: the increment halves on non-convergence and grows
  again after fast steps, so the run survives limit points gracefully and
  reports the last converged load factor as the capacity estimate.

The external load is ``F = F_constant + lambda * F_proportional`` so dead
loads or imperfection loads can be held while the proportional part ramps.
"""

from __future__ import annotations

import time
import copy
from dataclasses import dataclass, field, replace as dataclass_replace
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse

from .assembly import build_constraint_transformation
from .cases import make_result_case
from .fracture import (
    DeletedElementRecord,
    FractureConfig,
    detect_new_deletions,
    deleted_pressure_load_resultant,
    element_fracture_category,
    filtered_load_case_for_deleted_elements,
    fracture_summary,
    mpc_warning_for_deleted_shells,
)
from .linalg import MatrixClass, factorize
from .jit_compiler import numba_thread_scope
from .matrix_assembly import (
    _scatter_element_matrix,
    _triplets_to_csr,
    assemble_load_vector,
    assemble_stiffness_matrix,
)
from .recovery import ResourceConfig

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


_DOF_INDEX = {"ux": 0, "uy": 1, "uz": 2, "rx": 3, "ry": 4, "rz": 5}


@dataclass(frozen=True)
class NonlinearLoadStage:
    """One ordered load stage in a nonlinear load program."""

    name: str
    load_case: "LoadCase"
    target_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.target_factor <= 0.0:
            raise ValueError("target_factor must be positive")


@dataclass(frozen=True)
class NonlinearLoadProgram:
    """Ordered nonlinear load path, e.g. permanent then environmental load."""

    stages: Sequence[NonlinearLoadStage]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("NonlinearLoadProgram requires at least one stage")

    @property
    def total_factor(self) -> float:
        return float(sum(stage.target_factor for stage in self.stages))

    def stage_factors(self, path_factor: float) -> Dict[str, float]:
        remaining = max(float(path_factor), 0.0)
        factors: Dict[str, float] = {}
        for stage in self.stages:
            value = min(remaining, stage.target_factor)
            factors[stage.name] = max(value, 0.0)
            remaining -= value
        return factors

    def active_stage(self, path_factor: float) -> str:
        remaining = max(float(path_factor), 0.0)
        for stage in self.stages:
            if remaining <= stage.target_factor + 1.0e-12:
                return stage.name
            remaining -= stage.target_factor
        return self.stages[-1].name


@dataclass(frozen=True)
class DisplacementControl:
    """Scalar displacement constraint used with load-factor continuation."""

    node_id: Optional[int] = None
    dof: Optional[Union[str, int]] = None
    target_displacement: float = 0.0
    weighted_dofs: Optional[Mapping[Any, float]] = None

    def full_row(self, model: "FEModel") -> np.ndarray:
        row = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
        if self.weighted_dofs:
            for key, weight in self.weighted_dofs.items():
                if isinstance(key, tuple):
                    node_id, dof = key
                    dof_index = _local_dof_index(dof)
                    node = model.mesh.get_node(int(node_id))
                    if node is None:
                        raise ValueError(f"Displacement control references missing node {node_id}")
                    row[node.dofs[dof_index]] += float(weight)
                else:
                    row[int(key)] += float(weight)
        else:
            if self.node_id is None or self.dof is None:
                raise ValueError("DisplacementControl requires node_id and dof, or weighted_dofs")
            node = model.mesh.get_node(int(self.node_id))
            if node is None:
                raise ValueError(f"Displacement control references missing node {self.node_id}")
            row[node.dofs[_local_dof_index(self.dof)]] = 1.0
        if float(np.linalg.norm(row)) <= 0.0:
            raise ValueError("Displacement control row is empty")
        return row


@dataclass(frozen=True)
class NonlinearConvergenceSettings:
    """Automatic convergence controls for force-control nonlinear static solves.

    The settings do not change element theory.  They tune globalization and load
    increment adaptation: line-search usage, step growth after fast convergence,
    and cutback after difficult increments.
    """

    profile: str = "auto"
    line_search: str = "auto"
    fast_iterations: int = 4
    slow_iterations: int = 9
    growth_factor: float = 1.5
    cutback_factor: float = 0.5
    max_step_factor: float = 2.0
    min_step_fraction: Optional[float] = None
    max_line_search_cuts: int = 16
    line_search_reduction: float = 0.5

    def __post_init__(self) -> None:
        profile = str(self.profile).lower()
        line_search = str(self.line_search).lower()
        if profile not in {"legacy", "auto", "balanced", "fast", "robust"}:
            raise ValueError("profile must be one of 'legacy', 'auto', 'balanced', 'fast', or 'robust'")
        if line_search not in {"never", "rescue", "auto", "always"}:
            raise ValueError("line_search must be one of 'never', 'rescue', 'auto', or 'always'")
        if self.fast_iterations <= 0 or self.slow_iterations <= 0:
            raise ValueError("iteration thresholds must be positive")
        if self.growth_factor < 1.0:
            raise ValueError("growth_factor must be at least 1.0")
        if not (0.0 < self.cutback_factor < 1.0):
            raise ValueError("cutback_factor must be between 0 and 1")
        if self.max_step_factor <= 0.0:
            raise ValueError("max_step_factor must be positive")
        if self.min_step_fraction is not None and self.min_step_fraction <= 0.0:
            raise ValueError("min_step_fraction must be positive when supplied")
        if self.max_line_search_cuts <= 0:
            raise ValueError("max_line_search_cuts must be positive")
        if not (0.0 < self.line_search_reduction < 1.0):
            raise ValueError("line_search_reduction must be between 0 and 1")

    @staticmethod
    def for_profile(profile: str) -> "NonlinearConvergenceSettings":
        name = str(profile).lower()
        if name in {"auto", "balanced"}:
            return NonlinearConvergenceSettings(profile=name)
        if name == "fast":
            return NonlinearConvergenceSettings(
                profile="fast",
                line_search="auto",
                fast_iterations=3,
                slow_iterations=8,
                growth_factor=2.0,
                cutback_factor=0.5,
                max_step_factor=4.0,
                max_line_search_cuts=10,
            )
        if name == "robust":
            return NonlinearConvergenceSettings(
                profile="robust",
                line_search="always",
                fast_iterations=5,
                slow_iterations=7,
                growth_factor=1.25,
                cutback_factor=0.5,
                max_step_factor=1.0,
                max_line_search_cuts=20,
            )
        if name == "legacy":
            return NonlinearConvergenceSettings(
                profile="legacy",
                line_search="rescue",
                fast_iterations=5,
                slow_iterations=25,
                growth_factor=2.0,
                cutback_factor=0.5,
                max_step_factor=1.0,
                max_line_search_cuts=16,
            )
        raise ValueError("Unknown nonlinear convergence profile")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "line_search": self.line_search,
            "fast_iterations": int(self.fast_iterations),
            "slow_iterations": int(self.slow_iterations),
            "growth_factor": float(self.growth_factor),
            "cutback_factor": float(self.cutback_factor),
            "max_step_factor": float(self.max_step_factor),
            "min_step_fraction": self.min_step_fraction,
            "max_line_search_cuts": int(self.max_line_search_cuts),
            "line_search_reduction": float(self.line_search_reduction),
        }


def _coerce_convergence_settings(value: Optional[Union[str, Mapping[str, Any], NonlinearConvergenceSettings]]) -> NonlinearConvergenceSettings:
    if value is None:
        return NonlinearConvergenceSettings.for_profile("auto")
    if isinstance(value, NonlinearConvergenceSettings):
        return value
    if isinstance(value, str):
        return NonlinearConvergenceSettings.for_profile(value)
    if isinstance(value, Mapping):
        data = dict(value)
        profile = str(data.pop("profile", "auto")).lower()
        base = NonlinearConvergenceSettings.for_profile(profile).to_dict()
        base.update(data)
        return NonlinearConvergenceSettings(**base)
    raise TypeError("convergence_settings must be None, a profile string, a mapping, or NonlinearConvergenceSettings")


@dataclass
class NonlinearStaticStep:
    """One converged load increment."""

    step_index: int
    load_factor: float
    iterations: int
    residual_norm: float
    displacement_norm: float
    max_equivalent_plastic_strain: float
    control_value: Optional[float] = None
    active_stage: Optional[str] = None
    deleted_element_count: int = 0
    max_fracture_utilization: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "load_factor": self.load_factor,
            "iterations": self.iterations,
            "residual_norm": self.residual_norm,
            "displacement_norm": self.displacement_norm,
            "max_equivalent_plastic_strain": self.max_equivalent_plastic_strain,
            "control_value": self.control_value,
            "active_stage": self.active_stage,
            "deleted_element_count": int(self.deleted_element_count),
            "max_fracture_utilization": float(self.max_fracture_utilization),
        }


@dataclass
class NonlinearStaticResult:
    """Result of the incremental geometric/material nonlinear solve."""

    steps: List[NonlinearStaticStep]
    status: str
    displacements: np.ndarray
    load_factor: float
    element_states: Dict[int, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.status in {"completed", "stopped_at_limit"}

    @property
    def capacity_estimate(self) -> float:
        """Last converged proportional load factor."""
        return self.load_factor

    @property
    def peak_load_factor(self) -> float:
        return float(self.info.get("peak_load_factor", max((step.load_factor for step in self.steps), default=self.load_factor)))

    @property
    def last_converged_load_factor(self) -> float:
        return float(self.info.get("last_converged_load_factor", self.load_factor))

    @property
    def failure_reason(self) -> Optional[str]:
        return self.info.get("failure_reason")

    @property
    def status_category(self) -> str:
        return str(self.info.get("status_category", self.status))

    @property
    def stop_reason(self) -> Optional[str]:
        return self.info.get("stop_reason", self.failure_reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "status_category": self.status_category,
            "converged": self.converged,
            "load_factor": self.load_factor,
            "peak_load_factor": self.peak_load_factor,
            "last_converged_load_factor": self.last_converged_load_factor,
            "failure_reason": self.failure_reason,
            "stop_reason": self.stop_reason,
            "info": self.info,
            "steps": [step.to_dict() for step in self.steps],
        }


def _local_dof_index(dof: Union[str, int]) -> int:
    if isinstance(dof, str):
        key = dof.lower()
        if key not in _DOF_INDEX:
            raise ValueError(f"Unknown DOF {dof!r}; use one of {sorted(_DOF_INDEX)}")
        return _DOF_INDEX[key]
    index = int(dof)
    if not (0 <= index < 6):
        raise ValueError("DOF index must be in [0, 5]")
    return index


def _assemble_nonlinear_system(
    model: "FEModel",
    displacements: np.ndarray,
    committed_states: Dict[int, Any],
    num_layers: int,
    tangent: bool = True,
    deleted_element_ids: Optional[Sequence[int]] = None,
    residual_stiffness_fraction: float = 1.0,
    element_stiffness_scales: Optional[Mapping[int, float]] = None,
    kinematics: str = "von_karman",
) -> Tuple[np.ndarray, Any, Dict[int, Any]]:
    """Assemble F_int (and the tangent K_T when requested) at a state.

    ``kinematics`` selects between the default von Karman element response and
    the element-independent corotational formulation for large rigid rotations
    (linear-elastic local response; see :mod:`fe_solver.corotational`).
    """
    mesh = model.mesh
    total_dofs = mesh.dof_manager.total_dofs
    F_int = np.zeros(total_dofs, dtype=float)
    data: list = []
    trial_states: Dict[int, Any] = {}
    deleted_set = {int(element_id) for element_id in (deleted_element_ids or ())}
    residual_fraction = float(residual_stiffness_fraction)
    element_scales = {
        int(element_id): min(max(float(scale), 0.0), 1.0)
        for element_id, scale in (element_stiffness_scales or {}).items()
    }

    if tangent:
        from .matrix_assembly import _get_cached_sparsity_pattern
        rows_concat, cols_concat = _get_cached_sparsity_pattern(mesh, "tangent_stiffness")

    from .elements import ShellElement
    from .vectorized_nonlinear import batch_shell_nonlinear_response

    groups = {}
    for elem_id, element in mesh.elements.items():
        if kinematics == "corotational":
            break
        if isinstance(element, ShellElement) and getattr(element, "_is_quadrilateral", False):
            key = (
                element.num_nodes,
                element.thickness,
                element.drilling_stabilization,
                element.reduced_integration,
                element.hourglass_stabilization,
                element.material_name,
            )
            if key not in groups:
                groups[key] = []
            groups[key].append((elem_id, element))

    precomputed_F = {}
    precomputed_K = {}

    for key, elem_list in groups.items():
        num_nodes, thickness, drilling_stabilization, _reduced_integration, _hourglass_stabilization, material_name = key
        material = model.get_material(material_name)
        E = float(material.elastic_modulus)
        nu = float(material.poisson_ratio)
        G_mod = float(material.shear_modulus)
        curve = getattr(material, "hardening_curve", None)

        n_elem = len(elem_list)
        first_element = elem_list[0][1]
        n_dof = first_element.total_dofs
        
        # We need to extract the caches
        cache_first = first_element._nonlinear_geometry(mesh)
        n_gp = cache_first["detw_all"].shape[0]
        n_shear = cache_first["detw_shear_all"].shape[0]

        u_elem_batch = np.zeros((n_elem, n_dof))
        T0_batch = np.zeros((n_elem, n_dof, n_dof))
        B_m_all_batch = np.zeros((n_elem, n_gp, 3, n_dof))
        B_b_all_batch = np.zeros((n_elem, n_gp, 3, n_dof))
        B_d_all_batch = np.zeros((n_elem, n_gp, 1, n_dof))
        Gw_all_batch = np.zeros((n_elem, n_gp, 2, n_dof))
        detw_all_batch = np.zeros((n_elem, n_gp))
        B_s_all_batch = np.zeros((n_elem, n_shear, 2, n_dof))
        detw_shear_all_batch = np.zeros((n_elem, n_shear))

        plastic_strain_batch = np.zeros((n_elem, n_gp * num_layers, 3))
        alpha_batch = np.zeros((n_elem, n_gp * num_layers))

        dof_mappings = []
        for idx, (elem_id, element) in enumerate(elem_list):
            dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
            dof_mappings.append(dof_mapping)
            if dof_mapping.size > 0:
                u_elem_batch[idx] = displacements[dof_mapping]

            cache = element._nonlinear_geometry(mesh)
            T0_batch[idx] = cache["T0"]
            B_m_all_batch[idx] = cache["B_m_all"]
            B_b_all_batch[idx] = cache["B_b_all"]
            B_d_all_batch[idx] = cache["B_d_all"]
            Gw_all_batch[idx] = cache["Gw_all"]
            detw_all_batch[idx] = cache["detw_all"]
            B_s_all_batch[idx] = cache["B_s_all"]
            detw_shear_all_batch[idx] = cache["detw_shear_all"]

            state = committed_states.get(elem_id)
            if state is None:
                state = element.init_nonlinear_state(num_layers)
            plastic_strain_batch[idx] = state["plastic_strain"]
            alpha_batch[idx] = state["alpha"]

        F_int_batch, K_T_batch, ep_new, alpha_new, layer_strain_batch = batch_shell_nonlinear_response(
            u_elem_batch,
            T0_batch,
            B_m_all_batch,
            B_b_all_batch,
            B_d_all_batch,
            Gw_all_batch,
            detw_all_batch,
            B_s_all_batch,
            detw_shear_all_batch,
            E,
            nu,
            G_mod,
            thickness,
            drilling_stabilization,
            tangent,
            curve,
            plastic_strain_batch,
            alpha_batch,
            num_layers,
        )

        for idx, (elem_id, element) in enumerate(elem_list):
            precomputed_F[elem_id] = F_int_batch[idx]
            if tangent:
                precomputed_K[elem_id] = K_T_batch[idx]
            
            # Reconstruct trial state to be compatible with single-element API
            trial_state = {
                "plastic_strain": ep_new[idx],
                "alpha": alpha_new[idx],
                "layer_strain": layer_strain_batch[idx * n_gp * num_layers : (idx + 1) * n_gp * num_layers].copy(),
            }
            if curve is not None:
                # We need layer_stress for outputs if people use it, let's keep it simple: we can omit it if not strictly required, 
                # but let's see if elements.py returned it. We didn't return it from batch_shell_nonlinear_response.
                pass
            if elem_id in deleted_set:
                trial_state = committed_states.get(elem_id, trial_state)
            trial_states[elem_id] = trial_state

    for elem_id, element in mesh.elements.items():
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            continue
        
        if elem_id in precomputed_F:
            f_elem = precomputed_F[elem_id]
            k_elem = precomputed_K.get(elem_id) if tangent else None
            # trial_state already in trial_states dict
        else:
            f_elem = None
            k_elem = None
            if kinematics == "corotational":
                from .corotational import corotational_element_response

                f_elem, k_elem, cr_trial_state = corotational_element_response(
                    model,
                    int(elem_id),
                    element,
                    displacements[dof_mapping],
                    tangent,
                    committed_state=committed_states.get(elem_id),
                    num_layers=num_layers,
                )
                if f_elem is not None and cr_trial_state is not None:
                    trial_states[elem_id] = cr_trial_state
            if f_elem is None:
                material = model.get_material(element.material_name)
                u_elem = displacements[dof_mapping]
                f_elem, k_elem, trial_state = element.compute_nonlinear_response(
                    mesh, material, u_elem, committed_states.get(elem_id), num_layers, tangent
                )
                if trial_state is not None:
                    trial_states[elem_id] = trial_state

        if elem_id in deleted_set:
            f_elem = residual_fraction * np.asarray(f_elem, dtype=float)
            if tangent and k_elem is not None:
                k_elem = residual_fraction * np.asarray(k_elem, dtype=float)
            if elem_id in committed_states:
                trial_states[elem_id] = committed_states[elem_id]
        elif elem_id in element_scales:
            scale = float(element_scales[elem_id])
            f_elem = scale * np.asarray(f_elem, dtype=float)
            if tangent and k_elem is not None:
                k_elem = scale * np.asarray(k_elem, dtype=float)

        np.add.at(F_int, dof_mapping, np.asarray(f_elem, dtype=float))
        if tangent and k_elem is not None:
            data.append(np.asarray(k_elem, dtype=float).ravel())

    if tangent:
        if not data:
            K_T = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
        else:
            K_T = sparse.coo_matrix(
                (np.concatenate(data), (rows_concat, cols_concat)),
                shape=(total_dofs, total_dofs),
                dtype=float,
            ).tocsr()
    else:
        K_T = None
        
    return F_int, K_T, trial_states


def _max_plastic_strain(states: Dict[int, Any]) -> float:
    return float(_nonlinear_state_summary(states)["max_equivalent_plastic_strain"])


def _nonlinear_state_summary(states: Dict[int, Any]) -> Dict[str, Any]:
    """Summarize plastic strain and layer/fiber strain data from element states."""
    max_alpha = 0.0
    max_plastic_component = 0.0
    max_compressed_alpha = 0.0
    layer_min = float("inf")
    layer_max = float("-inf")
    fiber_min = float("inf")
    fiber_max = float("-inf")
    yielded = 0

    for state in states.values():
        if not isinstance(state, dict):
            continue
        alpha = np.asarray(state.get("alpha", []), dtype=float).reshape(-1)
        if alpha.size:
            local_max = float(np.max(alpha))
            max_alpha = max(max_alpha, local_max)
            if local_max > 0.0:
                yielded += 1
        plastic = np.asarray(state.get("plastic_strain", []), dtype=float)
        if plastic.size:
            max_plastic_component = max(max_plastic_component, float(np.max(np.abs(plastic))))
        layer = np.asarray(state.get("layer_strain", []), dtype=float)
        if layer.size:
            layer_2d = layer.reshape((-1, layer.shape[-1] if layer.ndim > 1 else 1))
            layer_min = min(layer_min, float(np.min(layer_2d)))
            layer_max = max(layer_max, float(np.max(layer_2d)))
            if alpha.size == layer_2d.shape[0]:
                compression = np.min(layer_2d[:, : min(2, layer_2d.shape[1])], axis=1) < 0.0
                if np.any(compression):
                    max_compressed_alpha = max(max_compressed_alpha, float(np.max(alpha[compression])))
        fiber = np.asarray(state.get("fiber_strain", []), dtype=float)
        if fiber.size:
            fiber_min = min(fiber_min, float(np.min(fiber)))
            fiber_max = max(fiber_max, float(np.max(fiber)))
            if alpha.size == fiber.size:
                compression = fiber < 0.0
                if np.any(compression):
                    max_compressed_alpha = max(max_compressed_alpha, float(np.max(alpha[compression])))

    return {
        "max_equivalent_plastic_strain": max_alpha,
        "max_plastic_strain_component": max_plastic_component,
        "max_compressed_side_plastic_strain": max_compressed_alpha,
        "layer_strain_min": None if layer_min == float("inf") else layer_min,
        "layer_strain_max": None if layer_max == float("-inf") else layer_max,
        "fiber_strain_min": None if fiber_min == float("inf") else fiber_min,
        "fiber_strain_max": None if fiber_max == float("-inf") else fiber_max,
        "yielded_element_count": yielded,
    }


def _copy_initial_states(initial_element_states: Optional[Mapping[int, Any]]) -> Dict[int, Any]:
    return copy.deepcopy(dict(initial_element_states or {}))


def _nonlinear_status_category(status: str, failure_reason: Optional[str]) -> str:
    if status == "completed":
        return "converged"
    if status == "empty_reduced_system":
        return "invalid_model"
    reason = str(failure_reason or "")
    if reason.startswith("fracture_") or "deleted_fraction" in reason:
        return "fracture_limit"
    if "singular" in reason or "factorization" in reason:
        return "singular_tangent"
    if "nonfinite" in reason:
        return "numerical_instability"
    if "maximum_iterations" in reason:
        return "iteration_failure"
    if "minimum_load_increment" in reason:
        return "limit_point_or_nonconvergence"
    if status == "stopped_at_limit":
        return "limit_point_or_nonconvergence"
    return "failed"


def _solve_static_displacement_control(
    *,
    model: "FEModel",
    T: sparse.csr_matrix,
    u0: np.ndarray,
    F_const: np.ndarray,
    F_prop: np.ndarray,
    stage_vectors: Sequence[np.ndarray],
    load_program: Optional[NonlinearLoadProgram],
    displacement_control: DisplacementControl,
    committed_states: Dict[int, Any],
    num_layers: int,
    num_steps: int,
    max_iterations: int,
    tolerance: float,
    info: Dict[str, Any],
    start_time: float,
    resource_config: Optional[ResourceConfig] = None,
    kinematics: str = "von_karman",
) -> NonlinearStaticResult:
    """Displacement-control Newton solve with load factor as an unknown."""
    if load_program is not None:
        if len(load_program.stages) == 1:
            F_const_dc = F_const
            F_prop_dc = load_program.stages[0].target_factor * stage_vectors[0]
            active_stage = load_program.stages[0].name
        else:
            F_const_dc = F_const.copy()
            for stage, vector in zip(load_program.stages[:-1], stage_vectors[:-1]):
                F_const_dc += stage.target_factor * vector
            F_prop_dc = load_program.stages[-1].target_factor * stage_vectors[-1]
            active_stage = load_program.stages[-1].name
        info["displacement_control_load_split"] = {
            "constant_stages": [stage.name for stage in load_program.stages[:-1]],
            "proportional_stage": active_stage,
        }
    else:
        F_const_dc = F_const
        F_prop_dc = F_prop
        active_stage = "displacement_control"

    n_red = int(T.shape[1])
    q = np.zeros(n_red, dtype=float)
    lam = 0.0
    steps: List[NonlinearStaticStep] = []
    history: List[Dict[str, Any]] = []
    status = "completed"
    failure_reason: Optional[str] = None
    total_iterations = 0

    row_full = displacement_control.full_row(model)
    row_red = np.asarray(row_full @ T, dtype=float).reshape(-1)
    row_u0 = float(row_full @ u0)
    if float(np.linalg.norm(row_red)) <= 0.0:
        raise ValueError("Displacement control target is fixed or dependent and cannot be used as an unknown")

    F_prop_red = np.asarray(T.T @ F_prop_dc, dtype=float).reshape(-1)
    if float(np.linalg.norm(F_prop_red)) <= 0.0:
        raise ValueError("Displacement control requires a non-zero proportional load vector")

    target_total = float(displacement_control.target_displacement)
    target_scale = max(abs(target_total), 1.0e-9)

    assembly_threads = None if resource_config is None else resource_config.assembly_threads
    with numba_thread_scope(assembly_threads):
        for step_index in range(1, num_steps + 1):
            target = target_total * step_index / num_steps
            residual_norm = float("inf")
            constraint_error = float("inf")
            states_new = committed_states

            for iteration in range(1, max_iterations + 1):
                total_iterations += 1
                u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
                F_int, K_T, trial_states = _assemble_nonlinear_system(
                    model, u, committed_states, num_layers, kinematics=kinematics
                )
                residual = np.asarray(T.T @ (F_const_dc + lam * F_prop_dc - F_int), dtype=float).reshape(-1)
                residual_norm = float(np.linalg.norm(residual))
                current = float(row_red @ q + row_u0)
                constraint = target - current
                constraint_error = abs(constraint)
                reference = max(float(np.linalg.norm(np.asarray(T.T @ (F_const_dc + max(abs(lam), 1.0) * F_prop_dc), dtype=float))), 1.0)

                if residual_norm <= tolerance * reference and constraint_error <= tolerance * target_scale:
                    states_new = trial_states
                    break

                K_red = (T.T @ K_T @ T).tocsr()
                aug = sparse.bmat(
                    [
                        [K_red, sparse.csr_matrix((-F_prop_red).reshape(-1, 1))],
                        [sparse.csr_matrix(row_red.reshape(1, -1)), sparse.csr_matrix((1, 1))],
                    ],
                    format="csr",
                )
                rhs = np.concatenate([residual, np.array([constraint], dtype=float)])
                try:
                    with np.errstate(all="ignore"):
                        handle = factorize(
                            aug,
                            MatrixClass.SYMMETRIC_INDEFINITE,
                            signature=f"nonlinear.displacement_control:{step_index}:{iteration}",
                        )
                        delta = np.asarray(handle.solve(rhs), dtype=float).reshape(-1)
                except Exception:
                    failure_reason = "singular_augmented_tangent"
                    break
                if np.any(~np.isfinite(delta)):
                    failure_reason = "nonfinite_augmented_solution"
                    break
                q += delta[:-1]
                lam += float(delta[-1])
            else:
                failure_reason = "maximum_iterations_reached"

            if failure_reason is not None:
                status = "stopped_at_limit" if steps else "diverged"
                break

            committed_states = states_new
            u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
            current = float(row_red @ q + row_u0)
            steps.append(
                NonlinearStaticStep(
                    step_index=step_index,
                    load_factor=float(lam),
                    iterations=iteration,
                    residual_norm=residual_norm,
                    displacement_norm=float(np.linalg.norm(u)),
                    max_equivalent_plastic_strain=_max_plastic_strain(committed_states),
                    control_value=current,
                    active_stage=active_stage,
                )
            )
            history.append(
                {
                    "step_index": step_index,
                    "load_factor": float(lam),
                    "control_value": current,
                    "target_displacement": target,
                    "residual_norm": residual_norm,
                    "constraint_error": constraint_error,
                    "iterations": iteration,
                    "active_stage": active_stage,
                }
            )

    u_final = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    info["failure_reason"] = failure_reason
    info["stop_reason"] = "target_displacement_reached" if failure_reason is None else failure_reason
    info["status_category"] = _nonlinear_status_category(status, failure_reason)
    info["last_converged_load_factor"] = float(lam)
    info["peak_load_factor"] = max((step.load_factor for step in steps), default=float(lam))
    info["force_displacement_history"] = history
    info["strain_summary"] = _nonlinear_state_summary(committed_states)
    info["total_newton_iterations"] = total_iterations
    info["solve_time"] = time.time() - start_time
    info["result_case"] = make_result_case(
        name="nonlinear_static_displacement_control",
        analysis_type="nonlinear_static",
        load_cases=tuple(stage.load_case for stage in load_program.stages) if load_program is not None else (),
        assembly_info={"load": {"vector_type": "load_program" if load_program is not None else "load"}, **info},
        solver_info={"convergence_info": {"status": status}},
        recovery={"displacements": True, "element_states": True, "force_displacement_history": True},
        settings={"control": "displacement", "num_steps": num_steps, "num_layers": num_layers, "kinematics": kinematics},
    ).to_dict()
    return NonlinearStaticResult(steps, status, u_final, float(lam), committed_states, info)


def solve_static_nonlinear(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    constant_load_case: Optional["LoadCase"] = None,
    max_load_factor: float = 1.0,
    num_steps: int = 10,
    max_iterations: int = 25,
    tolerance: float = 1.0e-6,
    num_layers: int = 5,
    min_step_fraction: float = 1.0 / 1024.0,
    imperfection: Optional[Any] = None,
    load_program: Optional[NonlinearLoadProgram] = None,
    control: str = "force",
    displacement_control: Optional[DisplacementControl] = None,
    initial_element_states: Optional[Mapping[int, Any]] = None,
    convergence_settings: Optional[Union[str, Mapping[str, Any], NonlinearConvergenceSettings]] = None,
    resource_config: Optional[ResourceConfig] = None,
    fracture_config: Optional[FractureConfig] = None,
    kinematics: str = "von_karman",
    status_callback: Optional[Callable[[str], None]] = None,
) -> NonlinearStaticResult:
    """Incremental nonlinear static solve with adaptive load stepping.

    The proportional load case is ramped from 0 to ``max_load_factor`` while
    ``constant_load_case`` (if given) is applied in full from the first
    increment.  Plastic state is committed per element only on increment
    convergence, so every Newton iteration return-maps from the last
    converged state (standard backward-Euler incremental plasticity).
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    kinematics = str(kinematics).lower()
    if kinematics not in {"von_karman", "corotational"}:
        raise ValueError("kinematics must be 'von_karman' or 'corotational'")
    if kinematics == "corotational":
        from .corotational import validate_corotational_scope

        validate_corotational_scope(model)
        if fracture_config is not None:
            raise ValueError("Corotational kinematics v1 does not support fracture/erosion")
    if max_load_factor <= 0.0:
        raise ValueError("max_load_factor must be positive")
    if fracture_config is not None and not isinstance(fracture_config, FractureConfig):
        raise TypeError("fracture_config must be a FractureConfig or None")
    settings = _coerce_convergence_settings(convergence_settings)
    if kinematics == "corotational" and settings.line_search in {"auto", "rescue"}:
        # Corotational Newton necessarily passes through a large intermediate
        # residual while the element frames rotate toward the new state;
        # residual-norm backtracking rejects that excursion and grinds the
        # increment adaptation.  Plain Newton converges in a few iterations.
        settings = dataclass_replace(settings, line_search="never")
    effective_min_step_fraction = settings.min_step_fraction if settings.min_step_fraction is not None else min_step_fraction

    start_time = time.time()
    if imperfection is not None:
        from .imperfections import apply_imperfection

        model = apply_imperfection(model, imperfection, copy_model=True)
    model.apply_boundary_conditions()

    # The constraint transformation only depends on supports/MPCs; the
    # elastic stiffness is assembled once to build it (and warms the element
    # caches used by the nonlinear kernels).
    K0, stiffness_info = assemble_stiffness_matrix(model)
    stage_vectors: List[np.ndarray] = []
    stage_infos: List[Dict[str, Any]] = []
    if load_program is not None:
        for stage in load_program.stages:
            vector, stage_info = assemble_load_vector(model, stage.load_case)
            stage_vectors.append(vector)
            stage_infos.append({"name": stage.name, "target_factor": stage.target_factor, **stage_info})
        F_prop = np.sum(np.vstack(stage_vectors), axis=0) if stage_vectors else np.zeros(K0.shape[0], dtype=float)
        load_info = {"vector_type": "load_program", "stages": stage_infos}
    else:
        F_prop, load_info = assemble_load_vector(model, load_case)

    if constant_load_case is not None:
        F_const, constant_load_info = assemble_load_vector(model, constant_load_case)
    else:
        F_const = np.zeros_like(F_prop)
        constant_load_info = None
    _, _, T, u0, _, constraint_info = build_constraint_transformation(K0, F_prop, model)

    info: Dict[str, Any] = {
        "stiffness": stiffness_info,
        "load": load_info,
        "constant_load": constant_load_info,
        "constraint_info": constraint_info,
        "num_layers": int(num_layers),
        "total_dofs": model.mesh.dof_manager.total_dofs,
        "reduced_dofs": int(T.shape[1]),
        "control": str(control),
        "kinematics": kinematics,
        "convergence_settings": settings.to_dict(),
        "resource_config": None if resource_config is None else resource_config.to_dict(),
    }
    if imperfection is not None:
        info["imperfection"] = getattr(model, "imperfection_metadata", [])

    control_name = str(control).lower()
    if control_name not in {"force", "displacement"}:
        raise ValueError("control must be 'force' or 'displacement'")
    if fracture_config is not None and control_name != "force":
        raise ValueError("fracture_config is currently supported only with force control")

    n_red = int(T.shape[1])
    if n_red == 0:
        info["failure_reason"] = "empty_reduced_system"
        info["stop_reason"] = "empty_reduced_system"
        info["status_category"] = _nonlinear_status_category("empty_reduced_system", "empty_reduced_system")
        info["result_case"] = make_result_case(
            name="nonlinear_static",
            analysis_type="nonlinear_static",
            load_cases=tuple(stage.load_case for stage in load_program.stages) if load_program is not None else (() if load_case is None else (load_case,)),
            assembly_info={"stiffness": stiffness_info, "load": load_info},
            solver_info={"convergence_info": {"status": "empty_reduced_system"}},
            recovery={"displacements": True, "element_states": True},
            settings={"control": control_name, "num_steps": num_steps, "num_layers": num_layers, "kinematics": kinematics},
        ).to_dict()
        return NonlinearStaticResult([], "empty_reduced_system", u0.copy(), 0.0, {}, info)

    q = np.zeros(n_red, dtype=float)
    committed_states: Dict[int, Any] = _copy_initial_states(initial_element_states)
    steps: List[NonlinearStaticStep] = []
    status = "completed"
    deleted_element_ids: set[int] = set()
    deletion_records: List[DeletedElementRecord] = []
    fracture_warnings: List[str] = []
    max_fracture_utilization = 0.0

    if load_program is not None and max_load_factor == 1.0:
        target_load_factor = load_program.total_factor
    else:
        target_load_factor = float(max_load_factor)

    def _load_vector_with_deleted(load: Optional["LoadCase"]) -> np.ndarray:
        filtered = filtered_load_case_for_deleted_elements(load, deleted_element_ids)
        vector, _info = assemble_load_vector(model, filtered)
        return vector

    def external_load_at(path_factor: float) -> Tuple[np.ndarray, Dict[str, float], Optional[str]]:
        if not deleted_element_ids:
            if load_program is None:
                return F_const + float(path_factor) * F_prop, {"proportional": float(path_factor)}, None
            factors = load_program.stage_factors(path_factor)
            F_ext = F_const.copy()
            for stage, vector in zip(load_program.stages, stage_vectors):
                F_ext += factors[stage.name] * vector
            return F_ext, factors, load_program.active_stage(path_factor)

        F_const_current = _load_vector_with_deleted(constant_load_case) if constant_load_case is not None else np.zeros_like(F_prop)
        if load_program is None:
            return F_const_current + float(path_factor) * _load_vector_with_deleted(load_case), {"proportional": float(path_factor)}, None
        factors = load_program.stage_factors(path_factor)
        F_ext = F_const_current.copy()
        for stage in load_program.stages:
            F_ext += factors[stage.name] * _load_vector_with_deleted(stage.load_case)
        return F_ext, factors, load_program.active_stage(path_factor)

    if control_name == "displacement":
        if displacement_control is None:
            raise ValueError("displacement_control is required when control='displacement'")
        return _solve_static_displacement_control(
            model=model,
            T=T,
            u0=u0,
            F_const=F_const,
            F_prop=F_prop,
            stage_vectors=stage_vectors,
            load_program=load_program,
            displacement_control=displacement_control,
            committed_states=committed_states,
            num_layers=num_layers,
            num_steps=num_steps,
            max_iterations=max_iterations,
            tolerance=tolerance,
            kinematics=kinematics,
            info=info,
            start_time=start_time,
            resource_config=resource_config,
        )

    base_step = target_load_factor / num_steps
    min_step = max(float(effective_min_step_fraction) * base_step, 1.0e-12)
    max_step = max(base_step * settings.max_step_factor, min_step)
    step_size = base_step
    lam = 0.0
    step_index = 0
    total_iterations = 0

    def newton_increment(q_start, F_ext_red, reference, line_search):
        """One load increment.  Plain full Newton when ``line_search`` is
        False (the fast path); backtracking-line-search Newton otherwise.
        Returns (converged, q, states, residual_norm, iterations_used, failure_reason).
        """
        nonlocal total_iterations
        q_trial = q_start.copy()
        u = np.asarray(T @ q_trial + u0, dtype=float).reshape(-1)
        F_int, K_T, trial_states = _assemble_nonlinear_system(
            model,
            u,
            committed_states,
            num_layers,
            kinematics=kinematics,
            deleted_element_ids=tuple(deleted_element_ids),
            residual_stiffness_fraction=(
                fracture_config.residual_stiffness_fraction if fracture_config is not None else 1.0
            ),
        )
        residual = F_ext_red - np.asarray(T.T @ F_int, dtype=float).reshape(-1)
        residual_norm = float(np.linalg.norm(residual))

        for iteration in range(1, max_iterations + 1):
            if status_callback:
                status_callback(f"\r  Step {step_index}/{num_steps}, Iteration {iteration}: Res {residual_norm:.2e}")
            total_iterations += 1
            if residual_norm <= tolerance * reference:
                return True, q_trial, trial_states, residual_norm, iteration, None

            K_red = (T.T @ K_T @ T).tocsr()
            try:
                with np.errstate(all="ignore"):
                    handle = factorize(
                        K_red,
                        MatrixClass.SYMMETRIC_INDEFINITE,
                        signature=f"nonlinear.static_newton:{lam:.16g}:{iteration}",
                    )
                    dq = np.asarray(handle.solve(residual), dtype=float).reshape(-1)
            except Exception:
                return False, q_start, committed_states, residual_norm, iteration, "singular_tangent_factorization"
            if np.any(~np.isfinite(dq)):
                return False, q_start, committed_states, residual_norm, iteration, "nonfinite_newton_increment"

            if not line_search:
                q_trial = q_trial + dq
                u = np.asarray(T @ q_trial + u0, dtype=float).reshape(-1)
                F_int, K_T, trial_states = _assemble_nonlinear_system(
                    model,
                    u,
                    committed_states,
                    num_layers,
                    kinematics=kinematics,
            deleted_element_ids=tuple(deleted_element_ids),
                    residual_stiffness_fraction=(
                        fracture_config.residual_stiffness_fraction if fracture_config is not None else 1.0
                    ),
                )
                residual = F_ext_red - np.asarray(T.T @ F_int, dtype=float).reshape(-1)
                residual_norm = float(np.linalg.norm(residual))
                if not np.isfinite(residual_norm):
                    return False, q_start, committed_states, residual_norm, iteration, "nonfinite_residual"
                continue

            # Backtracking line search on the residual norm.  Von Karman
            # membrane terms can make full Newton steps overshoot violently
            # when an iterate moves many plate thicknesses at once; halving
            # until the residual decreases restores global convergence.
            # Rejected trials skip the tangent assembly (residual only).
            accepted = False
            scale = 1.0
            for trial in range(settings.max_line_search_cuts):
                q_candidate = q_trial + scale * dq
                u = np.asarray(T @ q_candidate + u0, dtype=float).reshape(-1)
                with_tangent = trial == 0
                F_c, K_c, states_c = _assemble_nonlinear_system(
                    model,
                    u,
                    committed_states,
                    num_layers,
                    tangent=with_tangent,
                    kinematics=kinematics,
            deleted_element_ids=tuple(deleted_element_ids),
                    residual_stiffness_fraction=(
                        fracture_config.residual_stiffness_fraction if fracture_config is not None else 1.0
                    ),
                )
                r_c = F_ext_red - np.asarray(T.T @ F_c, dtype=float).reshape(-1)
                rn_c = float(np.linalg.norm(r_c))
                if np.isfinite(rn_c) and rn_c < residual_norm:
                    if not with_tangent:
                        F_c, K_c, states_c = _assemble_nonlinear_system(
                            model,
                            u,
                            committed_states,
                            num_layers,
                            tangent=True,
                            kinematics=kinematics,
            deleted_element_ids=tuple(deleted_element_ids),
                            residual_stiffness_fraction=(
                                fracture_config.residual_stiffness_fraction if fracture_config is not None else 1.0
                            ),
                        )
                        r_c = F_ext_red - np.asarray(T.T @ F_c, dtype=float).reshape(-1)
                        rn_c = float(np.linalg.norm(r_c))
                    q_trial = q_candidate
                    F_int, K_T, trial_states = F_c, K_c, states_c
                    residual, residual_norm = r_c, rn_c
                    accepted = True
                    break
                scale *= settings.line_search_reduction
            if not accepted:
                return False, q_start, committed_states, residual_norm, iteration, "line_search_failed"

        return False, q_start, committed_states, residual_norm, max_iterations, "maximum_iterations_reached"

    force_displacement_history: List[Dict[str, Any]] = []
    convergence_adaptation: List[Dict[str, Any]] = []
    force_line_search_next = False

    assembly_threads = None if resource_config is None else resource_config.assembly_threads
    with numba_thread_scope(assembly_threads):
        while lam < target_load_factor - 1.0e-12:
            step_size = min(step_size, max(target_load_factor - lam, min_step))
            lam_trial = min(lam + step_size, target_load_factor)
            attempted_step_size = lam_trial - lam
            F_ext, stage_factors, active_stage = external_load_at(lam_trial)
            F_ext_red = np.asarray(T.T @ F_ext, dtype=float).reshape(-1)
            reference = max(float(np.linalg.norm(F_ext_red)), 1.0)

            policy = settings.line_search
            line_search_first = policy == "always" or (
                policy == "auto" and (force_line_search_next or attempted_step_size > base_step * 1.000001)
            )
            converged, q_new, states_new, residual_norm, iterations_used, failure_reason = newton_increment(
                q, F_ext_red, reference, line_search=line_search_first
            )
            line_search_used = bool(line_search_first)
            if not converged and not line_search_first and policy in {"rescue", "auto", "always"}:
                # Rescue retry with globalized (line-search) Newton before
                # cutting the load increment.
                converged, q_new, states_new, residual_norm, extra, failure_reason = newton_increment(
                    q, F_ext_red, reference, line_search=True
                )
                iterations_used += extra
                line_search_used = True

            if converged:
                q = q_new
                lam = lam_trial
                committed_states = states_new
                force_line_search_next = False
                step_index += 1
                u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
                control_value = float(np.linalg.norm(u))
                new_records: Tuple[DeletedElementRecord, ...] = ()
                if fracture_config is not None:
                    new_records, step_fracture_utilization = detect_new_deletions(
                        model,
                        committed_states,
                        fracture_config,
                        deleted_element_ids,
                        step_index=step_index,
                        load_factor=float(lam),
                    )
                    max_fracture_utilization = max(max_fracture_utilization, step_fracture_utilization)
                    if new_records:
                        deletion_records.extend(new_records)
                        deleted_element_ids.update(record.element_id for record in new_records)
                        warning = mpc_warning_for_deleted_shells(model, (record.element_id for record in new_records))
                        if warning is not None and warning not in fracture_warnings:
                            fracture_warnings.append(warning)
                steps.append(
                    NonlinearStaticStep(
                        step_index=step_index,
                        load_factor=float(lam),
                        iterations=iterations_used,
                        residual_norm=residual_norm,
                        displacement_norm=float(np.linalg.norm(u)),
                        max_equivalent_plastic_strain=_max_plastic_strain(committed_states),
                        control_value=control_value,
                        active_stage=active_stage,
                        deleted_element_count=len(deleted_element_ids),
                        max_fracture_utilization=max_fracture_utilization,
                    )
                )
                removed_load = np.zeros(3, dtype=float)
                if fracture_config is not None and deleted_element_ids:
                    if load_program is None:
                        removed_load += float(lam) * deleted_pressure_load_resultant(model, load_case, deleted_element_ids)
                    else:
                        for stage in load_program.stages:
                            removed_load += stage_factors[stage.name] * deleted_pressure_load_resultant(
                                model, stage.load_case, deleted_element_ids
                            )
                    if constant_load_case is not None:
                        removed_load += deleted_pressure_load_resultant(model, constant_load_case, deleted_element_ids)
                force_displacement_history.append(
                    {
                        "step_index": step_index,
                        "load_factor": float(lam),
                        "control_value": control_value,
                        "displacement_norm": float(np.linalg.norm(u)),
                        "residual_norm": residual_norm,
                        "iterations": iterations_used,
                        "step_size": float(attempted_step_size),
                        "line_search_used": line_search_used,
                        "stage_factors": stage_factors,
                        "active_stage": active_stage,
                        "deleted_element_count": len(deleted_element_ids),
                        "newly_deleted_element_ids": [record.element_id for record in new_records],
                        "max_fracture_utilization": max_fracture_utilization,
                        "deleted_pressure_force_resultant": removed_load.tolist(),
                    }
                )
                if fracture_config is not None and deleted_element_ids:
                    scoped_total = sum(
                        1
                        for element in model.mesh.elements.values()
                        if element_fracture_category(element) in fracture_config.element_scope
                    )
                    deleted_fraction = len(deleted_element_ids) / max(scoped_total, 1)
                    if deleted_fraction > fracture_config.max_deleted_fraction + 1.0e-12:
                        status = "stopped_at_limit"
                        info["failure_reason"] = "max_deleted_fraction_reached"
                        info["deleted_fraction"] = float(deleted_fraction)
                        break
                next_step = step_size
                action = "keep"
                if iterations_used <= settings.fast_iterations and step_size < max_step:
                    next_step = min(step_size * settings.growth_factor, max_step)
                    action = "grow"
                elif iterations_used >= settings.slow_iterations:
                    next_step = max(step_size * settings.cutback_factor, min_step)
                    action = "shrink_after_slow_convergence"
                    if policy == "auto":
                        force_line_search_next = True
                convergence_adaptation.append(
                    {
                        "step_index": step_index,
                        "load_factor": float(lam),
                        "iterations": int(iterations_used),
                        "line_search_used": line_search_used,
                        "attempted_step_size": float(attempted_step_size),
                        "next_step_size": float(next_step),
                        "action": action,
                    }
                )
                step_size = next_step
            else:
                if fracture_config is not None and deleted_element_ids and failure_reason in {
                    "singular_tangent_factorization",
                    "maximum_iterations_reached",
                    "line_search_failed",
                }:
                    failure_reason = "fracture_instability"
                    status = "stopped_at_limit" if steps else "diverged"
                    info["failure_reason"] = failure_reason
                    info["first_failed_load_factor"] = float(lam_trial)
                    info["first_failed_step_size"] = float(attempted_step_size)
                    break
                previous_step_size = step_size
                step_size *= 0.5
                force_line_search_next = True
                convergence_adaptation.append(
                    {
                        "step_index": step_index + 1,
                        "load_factor": float(lam_trial),
                        "iterations": int(iterations_used),
                        "line_search_used": line_search_used,
                        "attempted_step_size": float(attempted_step_size),
                        "next_step_size": float(step_size),
                        "action": "cutback_after_nonconvergence",
                        "previous_step_size": float(previous_step_size),
                        "residual_norm": float(residual_norm),
                    }
                )
                if step_size < min_step:
                    status = "stopped_at_limit" if steps else "diverged"
                    info["failure_reason"] = "minimum_load_increment_reached"
                    info["first_failed_load_factor"] = float(lam_trial)
                    info["first_failed_step_size"] = float(attempted_step_size)
                    info["first_failed_iteration_reason"] = failure_reason
                    break

    u_final = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    if "failure_reason" not in info and status == "completed":
        info["failure_reason"] = None
    failure_reason = info.get("failure_reason")
    info["stop_reason"] = "target_load_factor_reached" if failure_reason is None else failure_reason
    info["status_category"] = _nonlinear_status_category(status, failure_reason)
    info["last_converged_load_factor"] = float(lam)
    info["peak_load_factor"] = max((step.load_factor for step in steps), default=float(lam))
    info["force_displacement_history"] = force_displacement_history
    info["convergence_adaptation"] = convergence_adaptation
    info["strain_summary"] = _nonlinear_state_summary(committed_states)
    if fracture_config is not None:
        info["fracture_summary"] = fracture_summary(
            model,
            fracture_config,
            deletion_records,
            deleted_element_ids,
            max_utilization=max_fracture_utilization,
            warnings=fracture_warnings,
        )
    if load_program is not None:
        info["load_program_stage_factors"] = load_program.stage_factors(lam)
    info["total_newton_iterations"] = total_iterations
    info["solve_time"] = time.time() - start_time
    info["result_case"] = make_result_case(
        name="nonlinear_static",
        analysis_type="nonlinear_static",
        load_cases=tuple(stage.load_case for stage in load_program.stages) if load_program is not None else (() if load_case is None else (load_case,)),
        assembly_info={"stiffness": stiffness_info, "load": load_info},
        solver_info={"convergence_info": {"status": status}},
        recovery={"displacements": True, "element_states": True, "force_displacement_history": True},
        settings={
            "control": control_name,
            "max_load_factor": max_load_factor,
            "num_steps": num_steps,
            "num_layers": num_layers,
            "convergence_settings": settings.to_dict(),
            "fracture": None if fracture_config is None else fracture_config.to_dict(),
        },
    ).to_dict()
    return NonlinearStaticResult(steps, status, u_final, float(lam), committed_states, info)
