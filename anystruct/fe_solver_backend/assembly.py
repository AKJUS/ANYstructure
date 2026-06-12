"""
Assembly and Solver Module

This module provides functions for:
- assembling global stiffness and load vectors,
- applying constraints,
- solving linear systems,
- post-processing reactions, stresses and displacements.

Constraint handling
-------------------
The linear solver uses an explicit transformation/reduction of the global
system. Fixed boundary conditions and beam-shell MPC/eccentricity constraints
are eliminated before solving:

    u = T q + u0
    T.T K T q = T.T (F - K u0)

For unsupported free-free models, the solver can additionally suppress the six
rigid body modes by solving an augmented nullspace system:

    [K_red  Q] [q]      [F_red]
    [Q.T    0] [lambda] [0    ]

where Q contains orthonormal reduced rigid-body modes. This gives a unique gauge
for the displacement field without introducing artificial support stiffness. If
the load vector has a rigid-body component, the solve still returns the gauged
solution and reports the imbalance through solver diagnostics.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab, gmres, minres, spsolve

from .matrix_assembly import _scatter_element_matrix, _triplets_to_csr

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel, FEMesh


def assemble_system(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    include_mass: bool = False,
) -> Tuple[sparse.csr_matrix, np.ndarray, Dict[str, Any]]:
    """
    Assemble the global stiffness matrix and load vector.

    If include_mass is true, the mass matrix is assembled separately and stored
    in assembly_info["mass_matrix"]. It is not added to stiffness.
    """
    mesh = model.mesh
    dof_manager = mesh.dof_manager
    total_dofs = dof_manager.total_dofs

    F_global = np.zeros(total_dofs, dtype=float)
    k_rows: List[np.ndarray] = []
    k_cols: List[np.ndarray] = []
    k_data: List[np.ndarray] = []
    m_rows: List[np.ndarray] = []
    m_cols: List[np.ndarray] = []
    m_data: List[np.ndarray] = []

    assembly_info: Dict[str, Any] = {
        "num_elements": 0,
        "num_nodes": mesh.num_nodes,
        "total_dofs": total_dofs,
        "includes_mass_matrix": bool(include_mass),
        "assembly_time": 0.0,
        "element_times": {},
    }
    start_time = time.time()

    for elem_id, element in mesh.elements.items():
        elem_start = time.time()
        material = model.get_material(element.material_name)
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            continue

        K_elem = np.asarray(element.compute_stiffness_matrix(mesh, material), dtype=float)
        _scatter_element_matrix(K_elem, dof_mapping, k_rows, k_cols, k_data)

        if include_mass:
            M_elem = np.asarray(element.compute_mass_matrix(mesh, material), dtype=float)
            _scatter_element_matrix(M_elem, dof_mapping, m_rows, m_cols, m_data)

        assembly_info["element_times"][elem_id] = time.time() - elem_start
        assembly_info["num_elements"] += 1

    if load_case is not None:
        F_global = load_case.get_load_vector(mesh, dof_manager, model.get_material)

    assembly_info["assembly_time"] = time.time() - start_time
    if include_mass:
        assembly_info["mass_matrix"] = _triplets_to_csr(m_rows, m_cols, m_data, total_dofs)
    return _triplets_to_csr(k_rows, k_cols, k_data, total_dofs), F_global, assembly_info


def assemble_mass_matrix(model: "FEModel") -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble the global mass matrix without mixing it into stiffness."""
    mesh = model.mesh
    total_dofs = mesh.dof_manager.total_dofs
    m_rows: List[np.ndarray] = []
    m_cols: List[np.ndarray] = []
    m_data: List[np.ndarray] = []
    info: Dict[str, Any] = {
        "num_elements": 0,
        "num_nodes": mesh.num_nodes,
        "total_dofs": total_dofs,
        "assembly_time": 0.0,
        "element_times": {},
    }
    start_time = time.time()

    for elem_id, element in mesh.elements.items():
        elem_start = time.time()
        material = model.get_material(element.material_name)
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            continue

        M_elem = np.asarray(element.compute_mass_matrix(mesh, material), dtype=float)
        _scatter_element_matrix(M_elem, dof_mapping, m_rows, m_cols, m_data)

        info["element_times"][elem_id] = time.time() - elem_start
        info["num_elements"] += 1

    info["assembly_time"] = time.time() - start_time
    return _triplets_to_csr(m_rows, m_cols, m_data, total_dofs), info


def _constraint_value_map(model: "FEModel") -> Dict[int, float]:
    """Return fixed prescribed displacement values keyed by global DOF."""
    dof_manager = model.mesh.dof_manager
    values: Dict[int, float] = {}
    for bc in model.boundary_conditions:
        for dof, value in bc.get_constrained_dofs(dof_manager):
            if dof in values and not np.isclose(values[dof], value):
                raise ValueError(f"Conflicting prescribed displacement for DOF {dof}: {values[dof]} vs {value}")
            values[dof] = float(value)
    return values


def _collect_mpc_constraints(model: "FEModel") -> List[Dict[str, Any]]:
    """Collect linear slave/master constraints from elements."""
    constraints: List[Dict[str, Any]] = []
    for element in model.mesh.elements.values():
        getter = getattr(element, "get_mpc_constraints", None)
        if getter is None:
            continue
        element_constraints = getter(model.mesh)
        if element_constraints:
            constraints.extend(element_constraints)
    return constraints


def build_constraint_transformation(
    K: sparse.csr_matrix,
    F: np.ndarray,
    model: "FEModel",
) -> Tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build the reduced system using fixed DOFs and linear MPC slave relations.

    The returned system is K_red q = F_red and the full displacement vector is
    recovered from u = T q + u0.
    """
    total_dofs = int(K.shape[0])
    fixed_values = _constraint_value_map(model)
    mpc_constraints = _collect_mpc_constraints(model)

    slave_constraints: Dict[int, Dict[str, Any]] = {}
    for constraint in mpc_constraints:
        slave = int(constraint["slave"])
        if slave in fixed_values:
            raise ValueError(f"DOF {slave} is both fixed and used as an MPC slave")
        if slave in slave_constraints:
            raise ValueError(f"DOF {slave} has multiple MPC slave definitions")
        masters = {int(k): float(v) for k, v in constraint.get("masters", {}).items() if abs(float(v)) > 0.0}
        if slave in masters:
            raise ValueError(f"MPC constraint for DOF {slave} references itself as a master")
        slave_constraints[slave] = {
            "masters": masters,
            "value": float(constraint.get("value", 0.0)),
            "label": constraint.get("label", "mpc"),
        }

    fixed_dofs = set(int(dof) for dof in fixed_values)
    slave_dofs = set(slave_constraints)
    dependent_dofs = fixed_dofs | slave_dofs
    independent_dofs = np.array([dof for dof in range(total_dofs) if dof not in dependent_dofs], dtype=int)
    independent_index = {int(dof): i for i, dof in enumerate(independent_dofs)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    u0 = np.zeros(total_dofs, dtype=float)

    for dof, value in fixed_values.items():
        u0[int(dof)] = float(value)

    for col, dof in enumerate(independent_dofs):
        rows.append(int(dof))
        cols.append(col)
        data.append(1.0)

    for slave, constraint in slave_constraints.items():
        value = constraint["value"]
        for master, coefficient in constraint["masters"].items():
            if master in fixed_values:
                value += coefficient * fixed_values[master]
            elif master in slave_dofs:
                raise ValueError(
                    f"Nested MPC dependency detected: slave DOF {slave} depends on slave master DOF {master}. "
                    "Only one-level beam-shell constraints are currently supported."
                )
            else:
                try:
                    col = independent_index[master]
                except KeyError as exc:
                    raise ValueError(f"MPC master DOF {master} is outside the active system") from exc
                rows.append(slave)
                cols.append(col)
                data.append(coefficient)
        u0[slave] = value

    T = sparse.csr_matrix((data, (rows, cols)), shape=(total_dofs, len(independent_dofs)))
    residual_offset = F - K @ u0
    K_red = (T.T @ K @ T).tocsr()
    F_red = np.asarray(T.T @ residual_offset, dtype=float).reshape(-1)

    info = {
        "method": "transformation",
        "num_total_dofs": total_dofs,
        "num_independent_dofs": int(len(independent_dofs)),
        "num_fixed_dofs": int(len(fixed_dofs)),
        "num_mpc_slave_dofs": int(len(slave_dofs)),
        "num_mpc_constraints": int(len(mpc_constraints)),
        "fixed_dofs": sorted(fixed_dofs),
        "slave_dofs": sorted(slave_dofs),
    }
    return K_red, F_red, T, u0, independent_dofs, info


def reconstruct_full_solution(T: sparse.csr_matrix, q: np.ndarray, u0: np.ndarray) -> np.ndarray:
    """Reconstruct full displacement vector from reduced unknowns."""
    return np.asarray(T @ q + u0, dtype=float).reshape(-1)


def _rigid_body_modes_full(model: "FEModel", total_dofs: int) -> np.ndarray:
    """Build six full-system rigid body modes for 6-DOF nodes."""
    modes = np.zeros((total_dofs, 6), dtype=float)
    if not model.mesh.nodes:
        return modes

    coords = np.asarray([node.coords() for node in model.mesh.nodes.values()], dtype=float)
    origin = np.mean(coords, axis=0)

    for node in model.mesh.nodes.values():
        x, y, z = node.coords() - origin
        ux, uy, uz, rx, ry, rz = node.dofs[:6]

        # Translations.
        modes[ux, 0] = 1.0
        modes[uy, 1] = 1.0
        modes[uz, 2] = 1.0

        # Rigid rotations: u = omega x r, theta = omega.
        modes[uy, 3] = -z
        modes[uz, 3] = y
        modes[rx, 3] = 1.0

        modes[ux, 4] = z
        modes[uz, 4] = -x
        modes[ry, 4] = 1.0

        modes[ux, 5] = -y
        modes[uy, 5] = x
        modes[rz, 5] = 1.0

    return modes


def _orthonormalize_columns(matrix: np.ndarray, tolerance: float = 1.0e-10) -> Tuple[np.ndarray, np.ndarray]:
    """Return independent orthonormal columns and the kept column indices."""
    if matrix.size == 0 or matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float), np.zeros(0, dtype=int)

    kept: List[int] = []
    basis: List[np.ndarray] = []
    scale = max(float(np.linalg.norm(matrix)), 1.0)
    for col in range(matrix.shape[1]):
        vector = np.asarray(matrix[:, col], dtype=float).copy()
        for q in basis:
            vector -= q * float(q @ vector)
        for q in basis:
            vector -= q * float(q @ vector)
        norm = float(np.linalg.norm(vector))
        if norm > tolerance * scale:
            basis.append(vector / norm)
            kept.append(col)

    if not basis:
        return np.zeros((matrix.shape[0], 0), dtype=float), np.zeros(0, dtype=int)
    return np.column_stack(basis), np.asarray(kept, dtype=int)


def build_reduced_rigid_body_modes(
    model: "FEModel",
    independent_dofs: np.ndarray,
    total_dofs: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build orthonormal rigid-body modes in reduced independent DOF space."""
    full_modes = _rigid_body_modes_full(model, total_dofs)
    if len(independent_dofs) == 0:
        return np.zeros((0, 0), dtype=float), {"rank": 0, "kept_mode_indices": []}
    reduced_modes = full_modes[np.asarray(independent_dofs, dtype=int), :]
    q_modes, kept = _orthonormalize_columns(reduced_modes)
    return q_modes, {
        "rank": int(q_modes.shape[1]),
        "kept_mode_indices": [int(i) for i in kept],
        "description": "Translations Tx, Ty, Tz followed by rotations Rx, Ry, Rz about the mesh centroid.",
    }


def apply_boundary_conditions(K: sparse.csr_matrix, F: np.ndarray, dof_manager: "DOFManager") -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Legacy penalty boundary-condition application."""
    penalty = 1.0e15
    constrained_dofs = getattr(dof_manager, "_constrained_dofs", set())
    if not constrained_dofs:
        return K, F
    K_modified = K.tolil()
    F_modified = F.copy()
    for dof in constrained_dofs:
        K_modified[dof, dof] = penalty
        F_modified[dof] = 0.0
    return K_modified.tocsr(), F_modified


def _solve_reduced_system(K_red: sparse.csr_matrix, F_red: np.ndarray, solver_type: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    if K_red.shape[0] == 0:
        return np.zeros(0), {"status": "converged", "note": "no independent DOFs"}

    if solver_type == "direct":
        try:
            with np.errstate(all="ignore"):
                q = spsolve(K_red, F_red)
            q = np.asarray(q, dtype=float)
            if np.any(np.isnan(q)) or np.any(np.isinf(q)):
                return np.zeros(K_red.shape[0]), {"status": "singular", "error": "NaN/Inf solution"}
            return q, {"status": "converged"}
        except Exception as exc:
            return np.zeros(K_red.shape[0]), {"status": "failed", "error": str(exc)}

    if solver_type == "gmres":
        q, info = gmres(K_red, F_red, rtol=1.0e-8, atol=1.0e-12, maxiter=1000)
        return np.asarray(q, dtype=float), {"status": "converged" if info == 0 else "not_converged", "iterations": info}

    if solver_type == "minres":
        q, info = minres(K_red, F_red, rtol=1.0e-8, maxiter=1000)
        return np.asarray(q, dtype=float), {"status": "converged" if info == 0 else "not_converged", "iterations": info}

    if solver_type == "bicgstab":
        q, info = bicgstab(K_red, F_red, rtol=1.0e-8, atol=1.0e-12, maxiter=1000)
        return np.asarray(q, dtype=float), {"status": "converged" if info == 0 else "not_converged", "iterations": info}

    raise ValueError(f"Unknown solver type: {solver_type}")


def _solve_nullspace_augmented_system(
    K_red: sparse.csr_matrix,
    F_red: np.ndarray,
    Q: np.ndarray,
    load_imbalance_tolerance: float = 1.0e-7,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve free-free reduced system with rigid body modes constrained by Q.T q = 0."""
    n = int(K_red.shape[0])
    r = int(Q.shape[1])
    if n == 0:
        return np.zeros(0), {"status": "converged", "note": "no independent DOFs", "nullspace_rank": r}
    if r == 0:
        q, info = _solve_reduced_system(K_red, F_red, "direct")
        info["nullspace_rank"] = 0
        return q, info

    Q_sparse = sparse.csr_matrix(Q)
    zero = sparse.csr_matrix((r, r), dtype=float)
    augmented = sparse.bmat([[K_red, Q_sparse], [Q_sparse.T, zero]], format="csr")
    rhs = np.concatenate([F_red, np.zeros(r, dtype=float)])

    load_components = np.asarray(Q.T @ F_red, dtype=float).reshape(-1)
    load_imbalance_norm = float(np.linalg.norm(load_components))
    load_norm = float(np.linalg.norm(F_red))
    relative_imbalance = load_imbalance_norm / max(load_norm, 1.0)

    try:
        with np.errstate(all="ignore"):
            solution = spsolve(augmented, rhs)
        solution = np.asarray(solution, dtype=float).reshape(-1)
        if np.any(np.isnan(solution)) or np.any(np.isinf(solution)):
            return np.zeros(n), {
                "status": "singular",
                "error": "NaN/Inf solution in nullspace augmented solve",
                "nullspace_rank": r,
                "rigid_body_load_components": load_components.tolist(),
                "relative_rigid_body_load_imbalance": relative_imbalance,
            }
    except Exception as exc:
        return np.zeros(n), {
            "status": "failed",
            "error": str(exc),
            "nullspace_rank": r,
            "rigid_body_load_components": load_components.tolist(),
            "relative_rigid_body_load_imbalance": relative_imbalance,
        }

    q = solution[:n]
    multipliers = solution[n:]
    residual = np.asarray(K_red @ q + Q @ multipliers - F_red, dtype=float).reshape(-1)
    gauge = np.asarray(Q.T @ q, dtype=float).reshape(-1)

    warnings: List[str] = []
    if relative_imbalance > load_imbalance_tolerance:
        warnings.append(
            "The external load vector has a non-zero rigid-body component. "
            "The nullspace solve returned a gauged displacement field and balancing generalized reactions. "
            "For a physical free-free static solution, use self-equilibrated loads."
        )

    return q, {
        "status": "converged",
        "method": "rigid_body_nullspace_augmented",
        "nullspace_rank": r,
        "rigid_body_load_components": load_components.tolist(),
        "rigid_body_lagrange_multipliers": multipliers.tolist(),
        "rigid_body_load_imbalance_norm": load_imbalance_norm,
        "relative_rigid_body_load_imbalance": relative_imbalance,
        "augmented_residual_norm": float(np.linalg.norm(residual)),
        "gauge_residual_norm": float(np.linalg.norm(gauge)),
        "warnings": warnings,
    }


def solve_linear(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    solver_type: str = "direct",
    precond: bool = True,
    constraint_mode: str = "auto",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve a linear FE problem using fixed-DOF/MPC transformation.

    constraint_mode:
        - "auto": use the ordinary reduced solve when fixed DOFs exist, otherwise
          use rigid-body nullspace augmentation.
        - "transformation": always use the ordinary reduced solve.
        - "nullspace": always use nullspace augmentation after MPC/fixed reduction.
    """
    mesh = model.mesh
    dof_manager = mesh.dof_manager

    model.apply_boundary_conditions()
    K, F, assembly_info = assemble_system(model, load_case)
    K_red, F_red, T, u0, independent_dofs, constraint_info = build_constraint_transformation(K, F, model)

    total_dofs = int(K.shape[0])
    Q, nullspace_info = build_reduced_rigid_body_modes(model, independent_dofs, total_dofs)
    mode = (constraint_mode or "auto").strip().lower()
    if mode not in {"auto", "transformation", "nullspace"}:
        raise ValueError(f"Unknown constraint_mode '{constraint_mode}'. Use auto, transformation or nullspace.")
    use_nullspace = mode == "nullspace" or (mode == "auto" and int(constraint_info["num_fixed_dofs"]) == 0 and Q.shape[1] > 0)

    solver_info: Dict[str, Any] = {
        "assembly": assembly_info,
        "solver_type": solver_type,
        "constraint_method": "transformation_fixed_plus_mpc_nullspace" if use_nullspace else "transformation_fixed_plus_mpc",
        "constraint_mode": mode,
        "num_free_dofs": int(len(independent_dofs)),
        "num_constrained_dofs": int(constraint_info["num_fixed_dofs"]),
        "num_mpc_slave_dofs": int(constraint_info["num_mpc_slave_dofs"]),
        "solve_time": 0.0,
        "constraint_info": constraint_info,
        "nullspace_info": nullspace_info,
        "convergence_info": {},
    }

    start_time = time.time()
    if use_nullspace:
        q, convergence_info = _solve_nullspace_augmented_system(K_red, F_red, Q)
    else:
        q, convergence_info = _solve_reduced_system(K_red, F_red, solver_type)
    solver_info["solve_time"] = time.time() - start_time
    solver_info["convergence_info"] = convergence_info

    if convergence_info.get("status") != "converged":
        return u0.copy(), solver_info
    return reconstruct_full_solution(T, q, u0), solver_info


def solve_nonlinear(
    model: "FEModel",
    load_case: "LoadCase",
    max_iterations: int = 20,
    tolerance: float = 1.0e-6,
    method: str = "newton_raphson",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Placeholder nonlinear solve using the same transformation constraints."""
    mesh = model.mesh
    dof_manager = mesh.dof_manager
    model.apply_boundary_conditions()

    K, F_ext, assembly_info = assemble_system(model, load_case)
    K_red, F_red, T, u0, independent_dofs, constraint_info = build_constraint_transformation(K, F_ext, model)
    q = np.zeros(len(independent_dofs), dtype=float)
    u = reconstruct_full_solution(T, q, u0)

    solver_info: Dict[str, Any] = {
        "assembly": assembly_info,
        "method": method,
        "constraint_method": "transformation_fixed_plus_mpc",
        "constraint_info": constraint_info,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "iterations": 0,
        "converged": False,
        "residual_history": [],
        "displacement_norm_history": [],
        "iteration_times": [],
    }

    for iteration in range(max_iterations):
        iter_start = time.time()
        F_int = compute_internal_forces(model, u)
        residual_full = F_ext - F_int
        residual_red = np.asarray(T.T @ residual_full, dtype=float).reshape(-1)
        residual_norm = float(np.linalg.norm(residual_red))
        solver_info["residual_history"].append(residual_norm)

        if residual_norm < tolerance:
            solver_info["converged"] = True
            solver_info["iterations"] = iteration + 1
            break

        if method == "newton_raphson":
            K, _, _ = assemble_system(model, load_case)
            K_red, _, T, u0, independent_dofs, constraint_info = build_constraint_transformation(K, np.zeros_like(F_ext), model)

        dq, convergence = _solve_reduced_system(K_red, residual_red, "direct")
        if convergence.get("status") != "converged":
            solver_info["convergence_info"] = convergence
            break
        q += dq
        u = reconstruct_full_solution(T, q, u0)
        solver_info["displacement_norm_history"].append(float(np.linalg.norm(dq)))
        solver_info["iteration_times"].append(time.time() - iter_start)

    return u, solver_info


def compute_internal_forces(model: "FEModel", displacements: np.ndarray) -> np.ndarray:
    """Compute internal forces for all elements."""
    mesh = model.mesh
    total_dofs = mesh.dof_manager.total_dofs
    F_int = np.zeros(total_dofs, dtype=float)

    for elem_id, element in mesh.elements.items():
        material = model.get_material(element.material_name)
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            continue
        u_elem = displacements[dof_mapping]
        F_elem = np.asarray(element.compute_internal_forces(mesh, u_elem, material), dtype=float)
        np.add.at(F_int, dof_mapping, F_elem)
    return F_int


def compute_reactions(model: "FEModel", displacements: np.ndarray, load_case: "LoadCase") -> Dict[int, np.ndarray]:
    """Compute reactions at fixed and MPC slave DOFs from the unreduced residual."""
    mesh = model.mesh
    dof_manager = mesh.dof_manager
    model.apply_boundary_conditions()

    K, _, _ = assemble_system(model)
    F_ext = load_case.get_load_vector(mesh, dof_manager, model.get_material)
    residual = K @ displacements - F_ext
    fixed_dofs = set(getattr(dof_manager, "_constrained_dofs", set()))
    mpc_slave_dofs = {int(c["slave"]) for c in _collect_mpc_constraints(model)}
    reported_dofs = fixed_dofs | mpc_slave_dofs

    reactions: Dict[int, np.ndarray] = {}
    for node_id, node in mesh.nodes.items():
        node_reactions = np.zeros(6, dtype=float)
        for local_dof in range(6):
            global_dof = node.dofs[local_dof]
            if global_dof in reported_dofs:
                node_reactions[local_dof] = residual[global_dof]
        if np.any(np.abs(node_reactions) > 0.0):
            reactions[node_id] = node_reactions
    return reactions


def compute_stresses(model: "FEModel", displacements: np.ndarray) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute stresses for all elements."""
    mesh = model.mesh
    stresses: Dict[int, Dict[str, np.ndarray]] = {}
    displacements = np.asarray(displacements, dtype=float)
    for elem_id, element in mesh.elements.items():
        material = model.get_material(element.material_name)
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0 or int(dof_mapping.max()) >= displacements.size:
            continue
        try:
            stresses[elem_id] = element.compute_stresses(mesh, displacements[dof_mapping], material)
        except (IndexError, ValueError):
            continue
    return stresses


def extract_node_displacements(displacements: np.ndarray, mesh: "FEMesh") -> Dict[int, np.ndarray]:
    """Extract displacements for each node from the global vector."""
    displacements = np.asarray(displacements, dtype=float)
    return {
        node_id: displacements[np.asarray(node.dofs, dtype=np.intp)]
        for node_id, node in mesh.nodes.items()
    }


def extract_element_displacements(displacements: np.ndarray, mesh: "FEMesh") -> Dict[int, np.ndarray]:
    """Extract element displacement vectors."""
    displacements = np.asarray(displacements, dtype=float)
    elem_displacements: Dict[int, np.ndarray] = {}
    for elem_id, element in mesh.elements.items():
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size:
            elem_displacements[elem_id] = displacements[dof_mapping]
    return elem_displacements
