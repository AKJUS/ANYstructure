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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import spsolve

from .assembly import build_constraint_transformation
from .matrix_assembly import (
    _scatter_element_matrix,
    _triplets_to_csr,
    assemble_load_vector,
    assemble_stiffness_matrix,
)

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


@dataclass
class NonlinearStaticStep:
    """One converged load increment."""

    step_index: int
    load_factor: float
    iterations: int
    residual_norm: float
    displacement_norm: float
    max_equivalent_plastic_strain: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "load_factor": self.load_factor,
            "iterations": self.iterations,
            "residual_norm": self.residual_norm,
            "displacement_norm": self.displacement_norm,
            "max_equivalent_plastic_strain": self.max_equivalent_plastic_strain,
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "converged": self.converged,
            "load_factor": self.load_factor,
            "info": self.info,
            "steps": [step.to_dict() for step in self.steps],
        }


def _assemble_nonlinear_system(
    model: "FEModel",
    displacements: np.ndarray,
    committed_states: Dict[int, Any],
    num_layers: int,
    tangent: bool = True,
) -> Tuple[np.ndarray, Any, Dict[int, Any]]:
    """Assemble F_int (and the tangent K_T when requested) at a state."""
    mesh = model.mesh
    total_dofs = mesh.dof_manager.total_dofs
    F_int = np.zeros(total_dofs, dtype=float)
    rows: list = []
    cols: list = []
    data: list = []
    trial_states: Dict[int, Any] = {}

    for elem_id, element in mesh.elements.items():
        material = model.get_material(element.material_name)
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            continue
        u_elem = displacements[dof_mapping]
        f_elem, k_elem, trial_state = element.compute_nonlinear_response(
            mesh, material, u_elem, committed_states.get(elem_id), num_layers, tangent
        )
        np.add.at(F_int, dof_mapping, np.asarray(f_elem, dtype=float))
        if tangent and k_elem is not None:
            _scatter_element_matrix(np.asarray(k_elem, dtype=float), dof_mapping, rows, cols, data)
        if trial_state is not None:
            trial_states[elem_id] = trial_state

    K_T = _triplets_to_csr(rows, cols, data, total_dofs) if tangent else None
    return F_int, K_T, trial_states


def _max_plastic_strain(states: Dict[int, Any]) -> float:
    value = 0.0
    for state in states.values():
        if isinstance(state, dict) and "alpha" in state:
            alpha = np.asarray(state["alpha"], dtype=float)
            if alpha.size:
                value = max(value, float(np.max(alpha)))
    return value


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
    if max_load_factor <= 0.0:
        raise ValueError("max_load_factor must be positive")

    start_time = time.time()
    model.apply_boundary_conditions()

    # The constraint transformation only depends on supports/MPCs; the
    # elastic stiffness is assembled once to build it (and warms the element
    # caches used by the nonlinear kernels).
    K0, stiffness_info = assemble_stiffness_matrix(model)
    F_prop, load_info = assemble_load_vector(model, load_case)
    if constant_load_case is not None:
        F_const, _ = assemble_load_vector(model, constant_load_case)
    else:
        F_const = np.zeros_like(F_prop)
    _, _, T, u0, _, constraint_info = build_constraint_transformation(K0, F_prop, model)

    info: Dict[str, Any] = {
        "stiffness": stiffness_info,
        "load": load_info,
        "constraint_info": constraint_info,
        "num_layers": int(num_layers),
        "total_dofs": model.mesh.dof_manager.total_dofs,
        "reduced_dofs": int(T.shape[1]),
    }

    n_red = int(T.shape[1])
    if n_red == 0:
        return NonlinearStaticResult([], "empty_reduced_system", u0.copy(), 0.0, {}, info)

    q = np.zeros(n_red, dtype=float)
    committed_states: Dict[int, Any] = {}
    steps: List[NonlinearStaticStep] = []
    status = "completed"

    base_step = max_load_factor / num_steps
    min_step = max(min_step_fraction * base_step, 1.0e-12)
    step_size = base_step
    lam = 0.0
    step_index = 0
    total_iterations = 0

    def newton_increment(q_start, F_ext_red, reference, line_search):
        """One load increment.  Plain full Newton when ``line_search`` is
        False (the fast path); backtracking-line-search Newton otherwise.
        Returns (converged, q, states, residual_norm, iterations_used).
        """
        nonlocal total_iterations
        q_trial = q_start.copy()
        u = np.asarray(T @ q_trial + u0, dtype=float).reshape(-1)
        F_int, K_T, trial_states = _assemble_nonlinear_system(
            model, u, committed_states, num_layers
        )
        residual = F_ext_red - np.asarray(T.T @ F_int, dtype=float).reshape(-1)
        residual_norm = float(np.linalg.norm(residual))

        for iteration in range(1, max_iterations + 1):
            total_iterations += 1
            if residual_norm <= tolerance * reference:
                return True, q_trial, trial_states, residual_norm, iteration

            K_red = (T.T @ K_T @ T).tocsr()
            try:
                with np.errstate(all="ignore"):
                    dq = np.asarray(spsolve(K_red, residual), dtype=float).reshape(-1)
            except Exception:
                return False, q_start, committed_states, residual_norm, iteration
            if np.any(~np.isfinite(dq)):
                return False, q_start, committed_states, residual_norm, iteration

            if not line_search:
                q_trial = q_trial + dq
                u = np.asarray(T @ q_trial + u0, dtype=float).reshape(-1)
                F_int, K_T, trial_states = _assemble_nonlinear_system(
                    model, u, committed_states, num_layers
                )
                residual = F_ext_red - np.asarray(T.T @ F_int, dtype=float).reshape(-1)
                residual_norm = float(np.linalg.norm(residual))
                if not np.isfinite(residual_norm):
                    return False, q_start, committed_states, residual_norm, iteration
                continue

            # Backtracking line search on the residual norm.  Von Karman
            # membrane terms can make full Newton steps overshoot violently
            # when an iterate moves many plate thicknesses at once; halving
            # until the residual decreases restores global convergence.
            # Rejected trials skip the tangent assembly (residual only).
            accepted = False
            scale = 1.0
            for trial in range(16):
                q_candidate = q_trial + scale * dq
                u = np.asarray(T @ q_candidate + u0, dtype=float).reshape(-1)
                with_tangent = trial == 0
                F_c, K_c, states_c = _assemble_nonlinear_system(
                    model, u, committed_states, num_layers, tangent=with_tangent
                )
                r_c = F_ext_red - np.asarray(T.T @ F_c, dtype=float).reshape(-1)
                rn_c = float(np.linalg.norm(r_c))
                if np.isfinite(rn_c) and rn_c < residual_norm:
                    if not with_tangent:
                        F_c, K_c, states_c = _assemble_nonlinear_system(
                            model, u, committed_states, num_layers, tangent=True
                        )
                        r_c = F_ext_red - np.asarray(T.T @ F_c, dtype=float).reshape(-1)
                        rn_c = float(np.linalg.norm(r_c))
                    q_trial = q_candidate
                    F_int, K_T, trial_states = F_c, K_c, states_c
                    residual, residual_norm = r_c, rn_c
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                return False, q_start, committed_states, residual_norm, iteration

        return False, q_start, committed_states, residual_norm, max_iterations

    while lam < max_load_factor - 1.0e-12:
        lam_trial = min(lam + step_size, max_load_factor)
        F_ext = F_const + lam_trial * F_prop
        F_ext_red = np.asarray(T.T @ F_ext, dtype=float).reshape(-1)
        reference = max(float(np.linalg.norm(F_ext_red)), 1.0)

        converged, q_new, states_new, residual_norm, iterations_used = newton_increment(
            q, F_ext_red, reference, line_search=False
        )
        if not converged:
            # Rescue retry with globalized (line-search) Newton before
            # cutting the load increment.
            converged, q_new, states_new, residual_norm, extra = newton_increment(
                q, F_ext_red, reference, line_search=True
            )
            iterations_used += extra

        if converged:
            q = q_new
            lam = lam_trial
            committed_states = states_new
            step_index += 1
            u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
            steps.append(
                NonlinearStaticStep(
                    step_index=step_index,
                    load_factor=float(lam),
                    iterations=iterations_used,
                    residual_norm=residual_norm,
                    displacement_norm=float(np.linalg.norm(u)),
                    max_equivalent_plastic_strain=_max_plastic_strain(committed_states),
                )
            )
            # Grow the step again after a fast increment.
            if iterations_used <= 5 and step_size < base_step:
                step_size = min(step_size * 2.0, base_step)
        else:
            step_size *= 0.5
            if step_size < min_step:
                status = "stopped_at_limit" if steps else "diverged"
                break

    u_final = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    info["total_newton_iterations"] = total_iterations
    info["solve_time"] = time.time() - start_time
    return NonlinearStaticResult(steps, status, u_final, float(lam), committed_states, info)
