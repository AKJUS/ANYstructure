"""Validation helpers for FE solver verification tests and benchmarks.

The functions in this module are intentionally lightweight.  They do not define
new solver behaviour; they inspect assembled models, load vectors, solver
metadata and reconstructed displacement fields so tests can lock the solver
architecture before eigenvalue buckling and nonlinear work are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


@dataclass(frozen=True)
class LoadResultant:
    """Global resultant force and moment of a load vector."""

    force: np.ndarray
    moment: np.ndarray

    @property
    def force_norm(self) -> float:
        return float(np.linalg.norm(self.force))

    @property
    def moment_norm(self) -> float:
        return float(np.linalg.norm(self.moment))


@dataclass(frozen=True)
class ShellPatchSummary:
    """Compact shell verification summary."""

    element_id: int
    strain_energy: float
    stiffness_symmetry_error: float
    max_membrane_spread: float
    max_bending_spread: float


def dof_order_signature(model: "FEModel") -> Dict[int, List[Tuple[int, str]]]:
    """Return node DOF indices with local DOF names.

    This is used by tests to protect the required ordering:
    ux, uy, uz, rx, ry, rz.
    """
    signature: Dict[int, List[Tuple[int, str]]] = {}
    dof_manager = model.mesh.dof_manager
    for node_id, node in model.mesh.nodes.items():
        entries: List[Tuple[int, str]] = []
        for dof in node.dofs:
            _node_id, _local_index, name = dof_manager.get_dof_info(dof)
            entries.append((int(dof), name))
        signature[int(node_id)] = entries
    return signature


def load_vector_resultant(model: "FEModel", load_vector: np.ndarray) -> LoadResultant:
    """Compute global force and moment resultants from a full load vector.

    The moment is taken about the model coordinate origin.
    """
    load_vector = np.asarray(load_vector, dtype=float).reshape(-1)
    force = np.zeros(3, dtype=float)
    moment = np.zeros(3, dtype=float)
    for node in model.mesh.nodes.values():
        node_force = load_vector[node.dofs[:3]]
        node_moment = load_vector[node.dofs[3:6]]
        r = node.coords()
        force += node_force
        moment += np.cross(r, node_force) + node_moment
    return LoadResultant(force=force, moment=moment)


def load_case_resultant(model: "FEModel", load_case: "LoadCase") -> LoadResultant:
    """Assemble a load case and return global force/moment resultants."""
    load_vector = load_case.get_load_vector(model.mesh, model.mesh.dof_manager, model.get_material)
    return load_vector_resultant(model, load_vector)


def mpc_constraint_residuals(model: "FEModel", displacements: np.ndarray) -> Dict[str, float]:
    """Return residuals for all element-provided linear MPC constraints.

    A constraint is interpreted as:

        u_slave = sum(coeff_i * u_master_i) + value

    The residual returned is lhs - rhs.  A correct reconstructed displacement
    field should give residuals close to zero for every MPC.
    """
    u = np.asarray(displacements, dtype=float).reshape(-1)
    residuals: Dict[str, float] = {}
    counter = 0
    for element in model.mesh.elements.values():
        getter = getattr(element, "get_mpc_constraints", None)
        if getter is None:
            continue
        for constraint in getter(model.mesh) or []:
            slave = int(constraint["slave"])
            masters = constraint.get("masters", {})
            value = float(constraint.get("value", 0.0))
            rhs = value
            for master, coefficient in masters.items():
                rhs += float(coefficient) * float(u[int(master)])
            label = str(constraint.get("label", f"mpc_{counter}"))
            residuals[label] = float(u[slave] - rhs)
            counter += 1
    return residuals


def shell_element_patch_summary(model: "FEModel", element_id: int, element_displacements: np.ndarray) -> ShellPatchSummary:
    """Return compact shell element diagnostics for a supplied element displacement field."""
    element = model.mesh.get_element(element_id)
    if element is None:
        raise ValueError(f"Element {element_id} not found")
    material = model.get_material(element.material_name)
    u = np.asarray(element_displacements, dtype=float).reshape(-1)
    if u.shape != (element.total_dofs,):
        raise ValueError(f"Element displacement shape {u.shape} does not match {(element.total_dofs,)}")

    K = element.compute_stiffness_matrix(model.mesh, material)
    stresses = element.compute_stresses(model.mesh, u, material)
    membrane_arrays = [
        np.asarray(stresses[key], dtype=float)
        for key in ("membrane_xx", "membrane_yy", "membrane_xy")
        if key in stresses
    ]
    bending_arrays = [
        np.asarray(stresses[key], dtype=float)
        for key in ("bending_xx", "bending_yy", "bending_xy")
        if key in stresses
    ]

    max_membrane_spread = max((float(np.max(values) - np.min(values)) for values in membrane_arrays), default=0.0)
    max_bending_spread = max((float(np.max(values) - np.min(values)) for values in bending_arrays), default=0.0)
    return ShellPatchSummary(
        element_id=int(element_id),
        strain_energy=float(u @ K @ u),
        stiffness_symmetry_error=float(np.linalg.norm(K - K.T)),
        max_membrane_spread=max_membrane_spread,
        max_bending_spread=max_bending_spread,
    )


def max_abs(values: Iterable[float]) -> float:
    """Return max absolute value, or 0.0 for an empty iterable."""
    data = [abs(float(value)) for value in values]
    return max(data) if data else 0.0


def nullspace_diagnostics(solver_info: Dict[str, Any]) -> Dict[str, Any]:
    """Extract nullspace-related diagnostics in a stable shape for tests."""
    convergence = solver_info.get("convergence_info", {}) or {}
    nullspace = solver_info.get("nullspace_info", {}) or {}
    return {
        "constraint_method": solver_info.get("constraint_method", ""),
        "rank": int(nullspace.get("rank", convergence.get("nullspace_rank", 0)) or 0),
        "relative_rigid_body_load_imbalance": float(convergence.get("relative_rigid_body_load_imbalance", 0.0) or 0.0),
        "augmented_residual_norm": float(convergence.get("augmented_residual_norm", 0.0) or 0.0),
        "gauge_residual_norm": float(convergence.get("gauge_residual_norm", 0.0) or 0.0),
        "status": convergence.get("status", "unknown"),
        "warnings": list(convergence.get("warnings", []) or []),
    }
