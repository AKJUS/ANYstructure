"""Bounded Crisfield-style arc-length continuation for nonlinear static analysis.

This module follows one proportional reference load through the first limit
point.  It reuses the production nonlinear element response, constraint
transformation and committed material-state machinery from
:mod:`fe_solver.nonlinear_static`.

Scope is deliberately limited to the ANYstructure capacity workflow:

* one proportional load pattern plus an optional constant preload,
* constrained models (no nonlinear free-free nullspace solve),
* geometric and material nonlinearity already supported by the elements,
* continuation only far enough beyond the peak to confirm the descending
  branch.

The equilibrium equations are

    R(q, lambda) = F_constant + lambda F_reference - F_internal(q) = 0

with the spherical constraint

    dq.T W dq + alpha**2 dlambda**2 = ds**2.

Newton corrections use block elimination.  The tangent is factorized once per
iteration and solved for two right-hand sides instead of assembling a generally
nonsymmetric bordered matrix.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .assembly import build_constraint_transformation
from .cases import make_result_case
from .linalg import MatrixClass, factorize
from .matrix_assembly import assemble_load_vector, assemble_stiffness_matrix
from .nonlinear_static import (
    _assemble_nonlinear_system,
    _copy_initial_states,
    _max_plastic_strain,
    _nonlinear_state_summary,
    solve_static_nonlinear,
)

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


_SMALL = 1.0e-14


@dataclass(frozen=True)
class ArcLengthControl:
    """Controls for bounded spherical arc-length continuation.

    ``initial_load_increment`` is used only to construct the first arc radius
    from the initial tangent direction.  Thereafter the radius is adapted in
    path space, not forced to produce a particular load increment.
    """

    initial_load_increment: float = 0.05
    minimum_load_increment: float = 5.0e-4
    maximum_load_increment: float = 0.20
    load_scaling: Optional[float] = None
    rotation_length_scale: Optional[float] = None
    target_iterations: int = 5
    growth_factor: float = 1.25
    cutback_factor: float = 0.5
    max_steps: int = 100
    max_retries_per_step: int = 8
    stop_after_peak_steps: int = 4
    peak_drop_tolerance: float = 1.0e-3
    maximum_absolute_load_factor: Optional[float] = None
    preload_steps: int = 10
    # Post-buckling continuation controls.  When ``post_peak_load_fraction``
    # is set the trace continues past the limit point and stops automatically
    # once the load factor has fallen to that fraction of the recorded peak
    # (set ``stop_after_peak_steps`` high to allow the descending branch).
    # ``max_translation`` is an absolute displacement guard in metres on the
    # largest nodal translation, protecting against runaway post-peak paths.
    post_peak_load_fraction: Optional[float] = None
    max_translation: Optional[float] = None

    def __post_init__(self) -> None:
        if self.initial_load_increment <= 0.0:
            raise ValueError("initial_load_increment must be positive")
        if self.minimum_load_increment <= 0.0:
            raise ValueError("minimum_load_increment must be positive")
        if self.maximum_load_increment < self.initial_load_increment:
            raise ValueError("maximum_load_increment must be at least initial_load_increment")
        if self.minimum_load_increment > self.initial_load_increment:
            raise ValueError("minimum_load_increment must not exceed initial_load_increment")
        if self.load_scaling is not None and self.load_scaling <= 0.0:
            raise ValueError("load_scaling must be positive when supplied")
        if self.rotation_length_scale is not None and self.rotation_length_scale <= 0.0:
            raise ValueError("rotation_length_scale must be positive when supplied")
        if self.target_iterations <= 0:
            raise ValueError("target_iterations must be positive")
        if self.growth_factor < 1.0:
            raise ValueError("growth_factor must be at least 1.0")
        if not (0.0 < self.cutback_factor < 1.0):
            raise ValueError("cutback_factor must be between 0 and 1")
        if self.max_steps <= 0 or self.max_retries_per_step <= 0:
            raise ValueError("max_steps and max_retries_per_step must be positive")
        if self.stop_after_peak_steps <= 0:
            raise ValueError("stop_after_peak_steps must be positive")
        if self.peak_drop_tolerance < 0.0:
            raise ValueError("peak_drop_tolerance must be non-negative")
        if self.maximum_absolute_load_factor is not None and self.maximum_absolute_load_factor <= 0.0:
            raise ValueError("maximum_absolute_load_factor must be positive when supplied")
        if self.preload_steps <= 0:
            raise ValueError("preload_steps must be positive")
        if self.post_peak_load_fraction is not None and not (0.0 < self.post_peak_load_fraction < 1.0):
            raise ValueError("post_peak_load_fraction must be in (0, 1) when supplied")
        if self.max_translation is not None and self.max_translation <= 0.0:
            raise ValueError("max_translation must be positive when supplied")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_load_increment": float(self.initial_load_increment),
            "minimum_load_increment": float(self.minimum_load_increment),
            "maximum_load_increment": float(self.maximum_load_increment),
            "load_scaling": self.load_scaling,
            "rotation_length_scale": self.rotation_length_scale,
            "target_iterations": int(self.target_iterations),
            "growth_factor": float(self.growth_factor),
            "cutback_factor": float(self.cutback_factor),
            "max_steps": int(self.max_steps),
            "max_retries_per_step": int(self.max_retries_per_step),
            "stop_after_peak_steps": int(self.stop_after_peak_steps),
            "peak_drop_tolerance": float(self.peak_drop_tolerance),
            "maximum_absolute_load_factor": self.maximum_absolute_load_factor,
            "preload_steps": int(self.preload_steps),
            "post_peak_load_fraction": self.post_peak_load_fraction,
            "max_translation": self.max_translation,
        }


@dataclass
class ArcLengthStep:
    """One converged point on the equilibrium path."""

    step_index: int
    load_factor: float
    iterations: int
    retries: int
    arc_radius: float
    residual_norm: float
    arc_residual: float
    displacement_norm: float
    load_increment: float
    path_increment_norm: float
    max_equivalent_plastic_strain: float
    is_peak: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "load_factor": float(self.load_factor),
            "iterations": int(self.iterations),
            "retries": int(self.retries),
            "arc_radius": float(self.arc_radius),
            "residual_norm": float(self.residual_norm),
            "arc_residual": float(self.arc_residual),
            "displacement_norm": float(self.displacement_norm),
            "load_increment": float(self.load_increment),
            "path_increment_norm": float(self.path_increment_norm),
            "max_equivalent_plastic_strain": float(self.max_equivalent_plastic_strain),
            "is_peak": bool(self.is_peak),
        }


@dataclass
class ArcLengthResult:
    """Result from bounded arc-length continuation."""

    steps: List[ArcLengthStep]
    status: str
    displacements: np.ndarray
    load_factor: float
    peak_load_factor: float
    peak_step_index: Optional[int]
    element_states: Dict[int, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def converged(self) -> bool:
        return self.status in {
            "peak_confirmed",
            "maximum_steps_reached",
            "load_factor_limit_reached",
            "post_buckling_traced",
            "displacement_limit_reached",
        }

    @property
    def capacity_estimate(self) -> float:
        return float(self.peak_load_factor)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "converged": self.converged,
            "load_factor": float(self.load_factor),
            "peak_load_factor": float(self.peak_load_factor),
            "peak_step_index": self.peak_step_index,
            "capacity_estimate": self.capacity_estimate,
            "info": self.info,
            "steps": [step.to_dict() for step in self.steps],
        }


def _characteristic_length(model: "FEModel") -> float:
    coords = np.asarray(model.mesh.get_node_coordinates(), dtype=float)
    if coords.size == 0:
        return 1.0
    spans = np.ptp(coords, axis=0)
    value = float(np.max(spans))
    return value if value > _SMALL else 1.0


def _reduced_metric(model: "FEModel", T: sparse.spmatrix, rotation_length_scale: float) -> sparse.csr_matrix:
    """Project a translation-equivalent full-DOF metric to reduced coordinates."""
    total_dofs = model.mesh.dof_manager.total_dofs
    weights = np.ones(total_dofs, dtype=float)
    rotation_weight = float(rotation_length_scale) ** 2
    for dof in range(total_dofs):
        _node_id, local_index, _name = model.mesh.dof_manager.get_dof_info(dof)
        if local_index >= 3:
            weights[dof] = rotation_weight
    W_full = sparse.diags(weights, format="csr")
    return (T.T @ W_full @ T).tocsr()


def _metric_dot(W: sparse.spmatrix, left: np.ndarray, right: np.ndarray) -> float:
    return float(np.asarray(left, dtype=float) @ np.asarray(W @ right, dtype=float))


def _metric_norm(W: sparse.spmatrix, vector: np.ndarray) -> float:
    return float(np.sqrt(max(_metric_dot(W, vector, vector), 0.0)))


def _factorized_solve(matrix: sparse.spmatrix, rhs: np.ndarray, signature: str) -> np.ndarray:
    handle = factorize(matrix, MatrixClass.SYMMETRIC_INDEFINITE, signature=signature)
    solution = np.asarray(handle.solve(np.asarray(rhs, dtype=float)), dtype=float).reshape(-1)
    if np.any(~np.isfinite(solution)):
        raise np.linalg.LinAlgError("non-finite tangent solution")
    return solution


def _recover_reduced_coordinates(T: sparse.spmatrix, u0: np.ndarray, displacements: np.ndarray) -> np.ndarray:
    rhs = np.asarray(displacements, dtype=float).reshape(-1) - np.asarray(u0, dtype=float).reshape(-1)
    result = sparse_linalg.lsqr(T, rhs, atol=1.0e-12, btol=1.0e-12)
    q = np.asarray(result[0], dtype=float).reshape(-1)
    mismatch = np.asarray(T @ q + u0 - displacements, dtype=float).reshape(-1)
    scale = max(float(np.linalg.norm(displacements)), 1.0)
    if float(np.linalg.norm(mismatch)) > 1.0e-8 * scale:
        raise RuntimeError("could not recover reduced coordinates from preloaded displacement state")
    return q


def _copy_model_with_imperfection(model: "FEModel", imperfection: Optional[Any]) -> "FEModel":
    if imperfection is None:
        return model
    from .imperfections import apply_imperfection

    return apply_imperfection(model, imperfection, copy_model=True)


def _max_nodal_translation(model: "FEModel", displacements: np.ndarray) -> float:
    """Largest nodal translation magnitude in the displacement vector."""
    peak = 0.0
    size = int(displacements.size)
    for node in model.mesh.nodes.values():
        dofs = np.asarray(node.dofs[:3], dtype=np.intp)
        if dofs.size == 0 or int(dofs.max()) >= size:
            continue
        value = float(np.linalg.norm(displacements[dofs]))
        if value > peak:
            peak = value
    return peak


def solve_static_arc_length(
    model: "FEModel",
    load_case: "LoadCase",
    *,
    constant_load_case: Optional["LoadCase"] = None,
    control: Optional[ArcLengthControl] = None,
    max_iterations: int = 25,
    tolerance: float = 1.0e-6,
    arc_tolerance: float = 1.0e-6,
    num_layers: int = 5,
    imperfection: Optional[Any] = None,
    initial_element_states: Optional[Mapping[int, Any]] = None,
    progress_callback: Optional[Any] = None,
) -> ArcLengthResult:
    """Trace the first nonlinear limit point with spherical arc-length control.

    The optional ``constant_load_case`` is first brought to equilibrium using
    the existing adaptive force-control solver.  Arc-length continuation then
    scales only ``load_case``.  Material states are committed only after a full
    equilibrium-plus-constraint convergence.
    """
    if load_case is None:
        raise ValueError("load_case is required for arc-length continuation")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0.0 or arc_tolerance <= 0.0:
        raise ValueError("tolerances must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")

    settings = control or ArcLengthControl()
    start_time = time.time()
    working_model = _copy_model_with_imperfection(model, imperfection)
    working_model.apply_boundary_conditions()

    K0, stiffness_info = assemble_stiffness_matrix(working_model)
    F_prop, load_info = assemble_load_vector(working_model, load_case)
    if constant_load_case is None:
        F_const = np.zeros_like(F_prop)
        constant_load_info = None
    else:
        F_const, constant_load_info = assemble_load_vector(working_model, constant_load_case)

    _, _, T, u0, _, constraint_info = build_constraint_transformation(K0, F_prop, working_model)
    n_red = int(T.shape[1])
    assembly_info = {
        "stiffness": stiffness_info,
        "load": load_info,
        "constant_load": constant_load_info,
        "constraint_info": constraint_info,
        "total_dofs": int(working_model.mesh.dof_manager.total_dofs),
        "reduced_dofs": n_red,
    }

    info: Dict[str, Any] = {
        **assembly_info,
        "control": settings.to_dict(),
        "num_layers": int(num_layers),
        "formulation": "crisfield_spherical_block_elimination",
    }
    if imperfection is not None:
        info["imperfection"] = getattr(working_model, "imperfection_metadata", [])

    if n_red == 0:
        info["failure_reason"] = "empty_reduced_system"
        return ArcLengthResult([], "empty_reduced_system", u0.copy(), 0.0, 0.0, None, {}, info)

    F_prop_red = np.asarray(T.T @ F_prop, dtype=float).reshape(-1)
    F_const_red = np.asarray(T.T @ F_const, dtype=float).reshape(-1)
    if float(np.linalg.norm(F_prop_red)) <= _SMALL:
        info["failure_reason"] = "zero_reduced_reference_load"
        return ArcLengthResult([], "zero_reference_load", u0.copy(), 0.0, 0.0, None, {}, info)

    committed_states: Dict[int, Any] = _copy_initial_states(initial_element_states)
    q = np.zeros(n_red, dtype=float)
    lam = 0.0
    preload_info = None

    if constant_load_case is not None and float(np.linalg.norm(F_const_red)) > _SMALL:
        preload = solve_static_nonlinear(
            working_model,
            load_case=constant_load_case,
            max_load_factor=1.0,
            num_steps=settings.preload_steps,
            max_iterations=max_iterations,
            tolerance=tolerance,
            num_layers=num_layers,
            initial_element_states=committed_states,
        )
        preload_info = preload.to_dict()
        if preload.status != "completed":
            info["preload"] = preload_info
            info["failure_reason"] = "constant_preload_not_converged"
            return ArcLengthResult(
                [],
                "preload_failed",
                preload.displacements,
                0.0,
                0.0,
                None,
                preload.element_states,
                info,
            )
        q = _recover_reduced_coordinates(T, u0, preload.displacements)
        committed_states = copy.deepcopy(preload.element_states)

    rotation_scale = settings.rotation_length_scale or _characteristic_length(working_model)
    W = _reduced_metric(working_model, T, rotation_scale)

    # Establish the first tangent direction and derive fixed path-space radius
    # limits from the user-facing load-increment settings.
    u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    F_int, K_T, _trial_states = _assemble_nonlinear_system(
        working_model, u, committed_states, num_layers, tangent=True
    )
    residual0 = F_const_red + lam * F_prop_red - np.asarray(T.T @ F_int, dtype=float).reshape(-1)
    reference0 = max(float(np.linalg.norm(F_const_red + F_prop_red)), 1.0)
    if float(np.linalg.norm(residual0)) > 10.0 * tolerance * reference0:
        info["failure_reason"] = "initial_state_not_in_equilibrium"
        info["initial_residual_norm"] = float(np.linalg.norm(residual0))
        return ArcLengthResult([], "initial_equilibrium_failed", u, lam, lam, None, committed_states, info)

    K_red = (T.T @ K_T @ T).tocsr()
    try:
        tangent_direction = _factorized_solve(K_red, F_prop_red, "arc_length.initial_tangent")
    except Exception as exc:
        info["failure_reason"] = "initial_tangent_factorization_failed"
        info["factorization_error"] = str(exc)
        return ArcLengthResult([], "initial_tangent_failed", u, lam, lam, None, committed_states, info)

    tangent_norm = _metric_norm(W, tangent_direction)
    load_scaling = float(settings.load_scaling) if settings.load_scaling is not None else max(tangent_norm, 1.0e-12)
    predictor_norm = float(np.sqrt(tangent_norm * tangent_norm + load_scaling * load_scaling))
    radius = settings.initial_load_increment * predictor_norm
    min_radius = radius * settings.minimum_load_increment / settings.initial_load_increment
    max_radius = radius * settings.maximum_load_increment / settings.initial_load_increment

    steps: List[ArcLengthStep] = []
    previous_dq: Optional[np.ndarray] = None
    previous_dlambda: Optional[float] = None
    peak_load_factor = float(lam)
    peak_step_index: Optional[int] = None
    max_translation = 0.0
    descending_steps = 0
    status = "maximum_steps_reached"
    failure_reason: Optional[str] = None
    total_iterations = 0
    total_retries = 0
    adaptation_history: List[Dict[str, Any]] = []

    for step_index in range(1, settings.max_steps + 1):
        q_base = q.copy()
        lambda_base = float(lam)
        states_base = copy.deepcopy(committed_states)
        accepted = False
        step_failure = "unknown"

        for retry in range(settings.max_retries_per_step + 1):
            total_retries += int(retry > 0)
            u_base = np.asarray(T @ q_base + u0, dtype=float).reshape(-1)
            F_base, K_base, _ = _assemble_nonlinear_system(
                working_model, u_base, states_base, num_layers, tangent=True
            )
            K_base_red = (T.T @ K_base @ T).tocsr()
            try:
                load_direction = _factorized_solve(
                    K_base_red,
                    F_prop_red,
                    f"arc_length.predictor:{step_index}:{retry}",
                )
            except Exception:
                step_failure = "singular_predictor_tangent"
                radius *= settings.cutback_factor
                if radius < min_radius:
                    break
                continue

            sign = 1.0
            if previous_dq is not None and previous_dlambda is not None:
                orientation = _metric_dot(W, previous_dq, load_direction) + (
                    load_scaling * load_scaling * previous_dlambda
                )
                sign = 1.0 if orientation >= 0.0 else -1.0

            direction_norm = float(
                np.sqrt(
                    max(_metric_dot(W, load_direction, load_direction), 0.0)
                    + load_scaling * load_scaling
                )
            )
            if direction_norm <= _SMALL:
                step_failure = "zero_predictor_direction"
                break

            dlambda_total = sign * radius / direction_norm
            dq_total = dlambda_total * load_direction
            q_trial = q_base + dq_total
            lambda_trial = lambda_base + dlambda_total
            residual_norm = float("inf")
            arc_residual = float("inf")
            trial_states = states_base

            for iteration in range(1, max_iterations + 1):
                total_iterations += 1
                u_trial = np.asarray(T @ q_trial + u0, dtype=float).reshape(-1)
                F_internal, K_trial, states_candidate = _assemble_nonlinear_system(
                    working_model, u_trial, states_base, num_layers, tangent=True
                )
                residual = (
                    F_const_red
                    + lambda_trial * F_prop_red
                    - np.asarray(T.T @ F_internal, dtype=float).reshape(-1)
                )
                residual_norm = float(np.linalg.norm(residual))
                arc_residual = float(
                    _metric_dot(W, dq_total, dq_total)
                    + (load_scaling * dlambda_total) ** 2
                    - radius * radius
                )
                force_reference = max(
                    float(np.linalg.norm(F_const_red + lambda_trial * F_prop_red)),
                    float(np.linalg.norm(F_prop_red)),
                    1.0,
                )
                arc_reference = max(radius * radius, 1.0e-24)

                if (
                    residual_norm <= tolerance * force_reference
                    and abs(arc_residual) <= arc_tolerance * arc_reference
                ):
                    trial_states = states_candidate
                    accepted = True
                    break

                K_trial_red = (T.T @ K_trial @ T).tocsr()
                try:
                    handle = factorize(
                        K_trial_red,
                        MatrixClass.SYMMETRIC_INDEFINITE,
                        signature=f"arc_length.corrector:{step_index}:{retry}:{iteration}",
                    )
                    correction_at_fixed_load = np.asarray(handle.solve(residual), dtype=float).reshape(-1)
                    correction_per_load = np.asarray(handle.solve(F_prop_red), dtype=float).reshape(-1)
                except Exception:
                    step_failure = "singular_corrector_tangent"
                    break
                if (
                    np.any(~np.isfinite(correction_at_fixed_load))
                    or np.any(~np.isfinite(correction_per_load))
                ):
                    step_failure = "nonfinite_corrector_solution"
                    break

                denominator = 2.0 * (
                    _metric_dot(W, dq_total, correction_per_load)
                    + load_scaling * load_scaling * dlambda_total
                )
                denominator_scale = max(
                    2.0 * radius * max(_metric_norm(W, correction_per_load), load_scaling),
                    1.0,
                )
                if abs(denominator) <= 1.0e-14 * denominator_scale:
                    step_failure = "singular_arc_constraint_linearization"
                    break

                numerator = -arc_residual - 2.0 * _metric_dot(
                    W, dq_total, correction_at_fixed_load
                )
                dlambda_correction = numerator / denominator
                dq_correction = correction_at_fixed_load + correction_per_load * dlambda_correction
                if (
                    np.any(~np.isfinite(dq_correction))
                    or not np.isfinite(dlambda_correction)
                ):
                    step_failure = "nonfinite_arc_correction"
                    break

                q_trial += dq_correction
                lambda_trial += float(dlambda_correction)
                dq_total = q_trial - q_base
                dlambda_total = lambda_trial - lambda_base
            else:
                step_failure = "maximum_iterations_reached"

            if accepted:
                q = q_trial
                lam = float(lambda_trial)
                committed_states = trial_states
                u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
                path_increment_norm = float(
                    np.sqrt(
                        max(_metric_dot(W, dq_total, dq_total), 0.0)
                        + (load_scaling * dlambda_total) ** 2
                    )
                )
                is_new_peak = lam > peak_load_factor
                if is_new_peak:
                    peak_load_factor = float(lam)
                    peak_step_index = step_index
                    descending_steps = 0
                    for old_step in steps:
                        old_step.is_peak = False
                else:
                    required_drop = settings.peak_drop_tolerance * max(abs(peak_load_factor), 1.0)
                    if peak_step_index is not None and lam < peak_load_factor - required_drop:
                        descending_steps += 1
                    else:
                        descending_steps = 0

                step = ArcLengthStep(
                    step_index=step_index,
                    load_factor=float(lam),
                    iterations=iteration,
                    retries=retry,
                    arc_radius=float(radius),
                    residual_norm=float(residual_norm),
                    arc_residual=float(arc_residual),
                    displacement_norm=float(np.linalg.norm(u)),
                    load_increment=float(dlambda_total),
                    path_increment_norm=path_increment_norm,
                    max_equivalent_plastic_strain=_max_plastic_strain(committed_states),
                    is_peak=is_new_peak,
                )
                steps.append(step)
                previous_dq = dq_total.copy()
                previous_dlambda = float(dlambda_total)
                max_translation = _max_nodal_translation(working_model, u)
                if progress_callback is not None:
                    try:
                        progress_callback(
                            {
                                "type": "nonlinear_static_step",
                                "control": "arc length",
                                "step_index": int(step_index),
                                "load_factor": float(lam),
                                "peak_load_factor": float(peak_load_factor),
                                "displacement_norm": float(np.linalg.norm(u)),
                                "max_translation": float(max_translation),
                                "iterations": int(iteration),
                                "max_equivalent_plastic_strain": float(step.max_equivalent_plastic_strain),
                            }
                        )
                    except Exception:
                        pass

                old_radius = radius
                if iteration <= max(settings.target_iterations // 2, 1):
                    radius = min(radius * settings.growth_factor, max_radius)
                    action = "grow"
                elif iteration > settings.target_iterations:
                    radius = max(
                        radius * max(settings.cutback_factor, np.sqrt(settings.target_iterations / iteration)),
                        min_radius,
                    )
                    action = "shrink_after_slow_convergence"
                else:
                    action = "keep"
                adaptation_history.append(
                    {
                        "step_index": step_index,
                        "iterations": int(iteration),
                        "retries": int(retry),
                        "accepted_radius": float(old_radius),
                        "next_radius": float(radius),
                        "action": action,
                    }
                )
                break

            radius *= settings.cutback_factor
            adaptation_history.append(
                {
                    "step_index": step_index,
                    "retry": int(retry),
                    "accepted": False,
                    "next_radius": float(radius),
                    "action": "cutback_after_nonconvergence",
                    "failure_reason": step_failure,
                }
            )
            if radius < min_radius:
                break

        if not accepted:
            q = q_base
            lam = lambda_base
            committed_states = states_base
            status = "stopped_at_limit" if steps else "diverged"
            failure_reason = step_failure if radius >= min_radius else "minimum_arc_radius_reached"
            break

        if (
            settings.post_peak_load_fraction is not None
            and peak_step_index is not None
            and step_index > peak_step_index
            and lam <= settings.post_peak_load_fraction * peak_load_factor
        ):
            # Automatic post-buckling stop: the descending branch has shed
            # the requested fraction of the peak load, so the post-buckling
            # response is traced and further continuation adds no insight.
            status = "post_buckling_traced"
            break
        if settings.max_translation is not None and max_translation > settings.max_translation:
            status = "displacement_limit_reached"
            break
        if descending_steps >= settings.stop_after_peak_steps:
            status = "peak_confirmed"
            break
        if (
            settings.maximum_absolute_load_factor is not None
            and abs(lam) >= settings.maximum_absolute_load_factor
        ):
            status = "load_factor_limit_reached"
            break
    else:
        status = "maximum_steps_reached"

    u_final = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    info["failure_reason"] = failure_reason
    info["last_converged_load_factor"] = float(lam)
    info["peak_load_factor"] = float(peak_load_factor)
    info["peak_step_index"] = peak_step_index
    info["descending_steps_after_peak"] = int(descending_steps)
    info["final_max_translation"] = float(max_translation)
    info["load_scaling"] = float(load_scaling)
    info["rotation_length_scale"] = float(rotation_scale)
    info["initial_arc_radius"] = float(settings.initial_load_increment * predictor_norm)
    info["minimum_arc_radius"] = float(min_radius)
    info["maximum_arc_radius"] = float(max_radius)
    info["adaptation_history"] = adaptation_history
    info["strain_summary"] = _nonlinear_state_summary(committed_states)
    info["preload"] = preload_info
    info["total_newton_iterations"] = int(total_iterations)
    info["total_retries"] = int(total_retries)
    info["solve_time"] = float(time.time() - start_time)
    info["result_case"] = make_result_case(
        name="nonlinear_static_arc_length",
        analysis_type="nonlinear_static",
        load_cases=(load_case,) if constant_load_case is None else (constant_load_case, load_case),
        assembly_info=assembly_info,
        solver_info={"convergence_info": {"status": status, "failure_reason": failure_reason}},
        recovery={
            "displacements": True,
            "element_states": True,
            "force_displacement_history": True,
            "arc_length_history": True,
        },
        settings={
            "control": "arc_length",
            "arc_length": settings.to_dict(),
            "max_iterations": int(max_iterations),
            "tolerance": float(tolerance),
            "arc_tolerance": float(arc_tolerance),
            "num_layers": int(num_layers),
        },
    ).to_dict()

    return ArcLengthResult(
        steps=steps,
        status=status,
        displacements=u_final,
        load_factor=float(lam),
        peak_load_factor=float(peak_load_factor),
        peak_step_index=peak_step_index,
        element_states=committed_states,
        info=info,
    )
