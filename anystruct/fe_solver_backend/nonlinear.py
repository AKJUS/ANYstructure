"""Controlled nonlinear stability checks.

This module deliberately stops at the first tangent-stiffness limit point.  It
does not attempt arc-length continuation or post-buckling path following.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigsh

from .assembly import build_constraint_transformation
from .cases import make_result_case
from .linalg import MatrixClass, factorize
from .matrix_assembly import (
    assemble_geometric_stiffness_matrix,
    assemble_load_vector,
    assemble_stiffness_matrix,
)

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


@dataclass
class NonlinearLoadStep:
    """One proportional load step with tangent-stability diagnostics."""

    step_index: int
    load_factor: float
    converged: bool
    iterations: int
    displacement_norm: float
    residual_norm: float
    tangent_min_eigenvalue: float
    tangent_stability_index: float
    tangent_status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "load_factor": self.load_factor,
            "converged": self.converged,
            "iterations": self.iterations,
            "displacement_norm": self.displacement_norm,
            "residual_norm": self.residual_norm,
            "tangent_min_eigenvalue": self.tangent_min_eigenvalue,
            "tangent_stability_index": self.tangent_stability_index,
            "tangent_status": self.tangent_status,
        }


@dataclass
class NonlinearLimitPointResult:
    """Result from load stepping stopped near the first stiffness limit."""

    steps: List[NonlinearLoadStep]
    status: str
    final_displacements: np.ndarray
    critical_load_factor_estimate: Optional[float]
    assembly_info: Dict[str, Any]
    constraint_info: Dict[str, Any]
    result_case: Optional[Dict[str, Any]] = None

    @property
    def last_load_factor(self) -> float:
        if not self.steps:
            return 0.0
        return self.steps[-1].load_factor

    @property
    def converged(self) -> bool:
        return self.status in {"completed", "near_limit_point", "limit_point_detected"}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "converged": self.converged,
            "last_load_factor": self.last_load_factor,
            "critical_load_factor_estimate": self.critical_load_factor_estimate,
            "assembly_info": self.assembly_info,
            "constraint_info": self.constraint_info,
            "result_case": self.result_case,
            "steps": [step.to_dict() for step in self.steps],
        }


_DENSE_EIGEN_LIMIT = 600


def _minimum_tangent_eigenvalue(K_red: Any, KG_red: Any, load_factor: float) -> float:
    """Smallest eigenvalue of the symmetrized reduced tangent stiffness.

    Small systems use dense ``eigvalsh``.  Larger systems use shift-invert
    Lanczos around zero, which targets exactly the eigenvalue that drives the
    limit-point check, with a dense fallback if the factorization fails
    (e.g. an exactly singular tangent).
    """
    if load_factor != 0.0 and KG_red.nnz > 0:
        tangent = (K_red - float(load_factor) * KG_red).tocsr()
    else:
        tangent = K_red.tocsr()
    tangent = 0.5 * (tangent + tangent.T)
    n = int(tangent.shape[0])
    if n == 0:
        return 0.0
    if n > _DENSE_EIGEN_LIMIT:
        try:
            k = min(2, n - 1)
            values = eigsh(tangent.tocsc(), k=k, sigma=0.0, which="LM", return_eigenvectors=False)
            return float(np.min(values))
        except Exception:
            pass
    eigenvalues = linalg.eigvalsh(tangent.toarray())
    if eigenvalues.size == 0:
        return 0.0
    return float(eigenvalues[0])


def _estimate_limit_load_factor(
    previous_step: Optional[NonlinearLoadStep],
    current_step: NonlinearLoadStep,
) -> Optional[float]:
    if previous_step is None:
        return current_step.load_factor
    previous_value = previous_step.tangent_min_eigenvalue
    current_value = current_step.tangent_min_eigenvalue
    if previous_value > 0.0 and current_value <= 0.0 and previous_value != current_value:
        span = current_step.load_factor - previous_step.load_factor
        return previous_step.load_factor + span * previous_value / (previous_value - current_value)
    return current_step.load_factor


def solve_nonlinear_load_stepping(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    element_states: Optional[Any] = None,
    max_load_factor: float = 1.0,
    num_steps: int = 20,
    stability_tolerance: float = 1.0e-3,
    stop_at_limit: bool = True,
) -> NonlinearLimitPointResult:
    """Run proportional load stepping and stop near the first limit point.

    The tangent matrix is linearized as ``KT(lambda) = K - lambda KG``.  Each
    step solves the tangent system ``KT(lambda) q = lambda F``, so the reported
    displacements follow the classical linearized pre-buckling amplification
    and grow super-linearly as the limit point is approached.  This is a
    controlled stability check for the current linear elastic/geometric
    stiffness theory, not a post-buckling continuation method; past the limit
    point the tangent is indefinite and displacements are not meaningful.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if max_load_factor < 0.0:
        raise ValueError("max_load_factor must be non-negative")
    if stability_tolerance < 0.0:
        raise ValueError("stability_tolerance must be non-negative")

    model.apply_boundary_conditions()
    K, stiffness_info = assemble_stiffness_matrix(model)
    KG, geometric_info = assemble_geometric_stiffness_matrix(model, element_states)
    F_ref, load_info = assemble_load_vector(model, load_case)

    K_red, F_red, T, u0, _, constraint_info = build_constraint_transformation(K, F_ref, model)
    KG_red = (T.T @ KG @ T).tocsr()

    assembly_info = {
        "stiffness": stiffness_info,
        "geometric_stiffness": geometric_info,
        "load": load_info,
        "total_dofs": model.mesh.dof_manager.total_dofs,
        "reduced_dofs": int(K_red.shape[0]),
    }

    if K_red.shape[0] == 0:
        result_case = make_result_case(
            name="nonlinear_limit_point",
            analysis_type="nonlinear_limit_point",
            load_cases=() if load_case is None else (load_case,),
            assembly_info=assembly_info,
            solver_info={"convergence_info": {"status": "empty_reduced_system"}},
            recovery={"steps": 0, "limit_point": True},
            settings={"max_load_factor": max_load_factor, "num_steps": num_steps},
        ).to_dict()
        return NonlinearLimitPointResult([], "empty_reduced_system", u0.copy(), None, assembly_info, constraint_info, result_case)

    initial_min = _minimum_tangent_eigenvalue(K_red, KG_red, 0.0)
    if initial_min <= 0.0:
        result_case = make_result_case(
            name="nonlinear_limit_point",
            analysis_type="nonlinear_limit_point",
            load_cases=() if load_case is None else (load_case,),
            assembly_info=assembly_info,
            solver_info={"convergence_info": {"status": "initial_tangent_not_positive"}},
            recovery={"steps": 0, "limit_point": True},
            settings={"max_load_factor": max_load_factor, "num_steps": num_steps},
        ).to_dict()
        return NonlinearLimitPointResult(
            [],
            "initial_tangent_not_positive",
            u0.copy(),
            0.0,
            assembly_info,
            constraint_info,
            result_case,
        )

    steps: List[NonlinearLoadStep] = []
    q = np.zeros(K_red.shape[0], dtype=float)
    u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    previous_step: Optional[NonlinearLoadStep] = None

    initial_step = NonlinearLoadStep(
        step_index=0,
        load_factor=0.0,
        converged=True,
        iterations=0,
        displacement_norm=float(np.linalg.norm(u)),
        residual_norm=0.0,
        tangent_min_eigenvalue=initial_min,
        tangent_stability_index=1.0,
        tangent_status="stable",
    )
    steps.append(initial_step)
    previous_step = initial_step

    status = "completed"
    critical_estimate: Optional[float] = None
    load_factors = np.linspace(max_load_factor / num_steps, max_load_factor, num_steps)

    for step_index, load_factor in enumerate(load_factors, start=1):
        rhs = float(load_factor) * F_red
        KT_red = (K_red - float(load_factor) * KG_red).tocsr()
        try:
            with np.errstate(all="ignore"):
                handle = factorize(
                    KT_red,
                    MatrixClass.SYMMETRIC_INDEFINITE,
                    signature=f"nonlinear.limit_step:{step_index}:{float(load_factor):.16g}",
                )
                q = np.asarray(handle.solve(rhs), dtype=float).reshape(-1)
        except Exception:
            status = "solver_failed"
            break
        if np.any(~np.isfinite(q)):
            status = "solver_failed"
            break

        u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
        residual = np.asarray(KT_red @ q - rhs, dtype=float).reshape(-1)
        tangent_min = _minimum_tangent_eigenvalue(K_red, KG_red, float(load_factor))
        stability_index = tangent_min / initial_min

        if tangent_min <= 0.0:
            tangent_status = "unstable"
        elif stability_index <= stability_tolerance:
            tangent_status = "near_limit"
        else:
            tangent_status = "stable"

        step = NonlinearLoadStep(
            step_index=step_index,
            load_factor=float(load_factor),
            converged=True,
            iterations=1,
            displacement_norm=float(np.linalg.norm(u)),
            residual_norm=float(np.linalg.norm(residual)),
            tangent_min_eigenvalue=float(tangent_min),
            tangent_stability_index=float(stability_index),
            tangent_status=tangent_status,
        )
        steps.append(step)

        if stop_at_limit and tangent_status in {"near_limit", "unstable"}:
            status = "limit_point_detected" if tangent_status == "unstable" else "near_limit_point"
            critical_estimate = _estimate_limit_load_factor(previous_step, step)
            break
        previous_step = step

    result_case = make_result_case(
        name="nonlinear_limit_point",
        analysis_type="nonlinear_limit_point",
        load_cases=() if load_case is None else (load_case,),
        assembly_info=assembly_info,
        solver_info={"convergence_info": {"status": status}},
        recovery={"steps": len(steps), "limit_point": True},
        settings={
            "max_load_factor": max_load_factor,
            "num_steps": num_steps,
            "stability_tolerance": stability_tolerance,
            "stop_at_limit": stop_at_limit,
        },
    ).to_dict()
    return NonlinearLimitPointResult(steps, status, u, critical_estimate, assembly_info, constraint_info, result_case)
