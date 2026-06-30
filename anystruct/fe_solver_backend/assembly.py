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

import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab, gmres, minres

from .cases import make_result_case
from .linalg import FactorizationCache, MatrixClass, factorize, factorize_cached
from .matrix_assembly import (
    assemble_load_matrix,
    assemble_mass_matrix as _canonical_assemble_mass_matrix,
    assemble_system as _canonical_assemble_system,
)

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel, FEMesh


def assemble_system(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    include_mass: bool = False,
) -> Tuple[sparse.csr_matrix, np.ndarray, Dict[str, Any]]:
    """Compatibility wrapper around :mod:`fe_solver.matrix_assembly`."""
    return _canonical_assemble_system(model, load_case, include_mass)


def assemble_mass_matrix(model: "FEModel") -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Compatibility wrapper around :func:`matrix_assembly.assemble_mass_matrix`."""
    return _canonical_assemble_mass_matrix(model)


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

    # Topological sort of slave_constraints to resolve cascading dependencies
    visited = {}  # 0 = unvisited, 1 = visiting, 2 = visited
    topo_order = []

    def visit(node: int):
        state = visited.get(node, 0)
        if state == 1:
            raise ValueError(f"Circular MPC dependency detected containing DOF {node}")
        if state == 2:
            return
        
        visited[node] = 1
        if node in slave_constraints:
            for master in slave_constraints[node]["masters"]:
                if master in slave_constraints:
                    visit(master)
        visited[node] = 2
        topo_order.append(node)

    for slave in slave_constraints:
        if visited.get(slave, 0) == 0:
            visit(slave)

    # Process and resolve master-slave dependencies recursively in topological order
    for slave in topo_order:
        constraint = slave_constraints[slave]
        resolved_masters: Dict[int, float] = {}
        resolved_value = constraint["value"]

        for master, coefficient in constraint["masters"].items():
            if master in slave_constraints:
                sub_masters = slave_constraints[master]["resolved_masters"]
                sub_value = slave_constraints[master]["resolved_value"]
                resolved_value += coefficient * sub_value
                for sub_m, sub_coeff in sub_masters.items():
                    resolved_masters[sub_m] = resolved_masters.get(sub_m, 0.0) + coefficient * sub_coeff
            else:
                resolved_masters[master] = resolved_masters.get(master, 0.0) + coefficient

        constraint["resolved_masters"] = resolved_masters
        constraint["resolved_value"] = resolved_value

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
        value = constraint["resolved_value"]
        for master, coefficient in constraint["resolved_masters"].items():
            if master in fixed_values:
                value += coefficient * fixed_values[master]
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


def _node_components(model: "FEModel") -> List[List[int]]:
    """Connected components from elements and MPC node relationships."""
    mesh = model.mesh
    adjacency: Dict[int, set] = {int(node_id): set() for node_id in mesh.nodes}

    def connect(node_ids: List[int]) -> None:
        ids = [int(node_id) for node_id in node_ids if int(node_id) in adjacency]
        for node_id in ids:
            adjacency[node_id].update(other for other in ids if other != node_id)

    for element in mesh.elements.values():
        connect(list(getattr(element, "node_ids", [])))
    for constraint in _collect_mpc_constraints(model):
        slave_node, _, _ = mesh.dof_manager.get_dof_info(int(constraint["slave"]))
        nodes = [slave_node]
        for master in constraint.get("masters", {}):
            master_node, _, _ = mesh.dof_manager.get_dof_info(int(master))
            nodes.append(master_node)
        connect([node_id for node_id in nodes if node_id >= 0])

    components: List[List[int]] = []
    visited = set()
    for node_id in sorted(adjacency):
        if node_id in visited:
            continue
        stack = [node_id]
        component = []
        visited.add(node_id)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current]):
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    return components


def _rigid_body_modes_full(model: "FEModel", total_dofs: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Build six full-system rigid body modes per connected component."""
    components = _node_components(model)
    modes = np.zeros((total_dofs, 6 * len(components)), dtype=float)
    component_info: List[Dict[str, Any]] = []
    constrained_dofs = set(getattr(model.mesh.dof_manager, "_constrained_dofs", set()))
    for component_index, node_ids in enumerate(components):
        if not node_ids:
            continue
        component_dofs: List[int] = []
        for node_id in node_ids:
            node = model.mesh.get_node(node_id)
            if node is not None:
                component_dofs.extend(int(dof) for dof in node.dofs)
        component_supported = any(dof in constrained_dofs for dof in component_dofs)
        coords = np.asarray([model.mesh.get_node(node_id).coords() for node_id in node_ids], dtype=float)
        origin = np.mean(coords, axis=0)
        base = 6 * component_index
        if component_supported:
            component_info.append(
                {
                    "component_index": component_index,
                    "node_ids": [int(node_id) for node_id in node_ids],
                    "centroid": origin.tolist(),
                    "candidate_modes": 0,
                    "supported": True,
                }
            )
            continue
        for node_id in node_ids:
            node = model.mesh.get_node(node_id)
            x, y, z = node.coords() - origin
            ux, uy, uz, rx, ry, rz = node.dofs[:6]

            modes[ux, base + 0] = 1.0
            modes[uy, base + 1] = 1.0
            modes[uz, base + 2] = 1.0

            modes[uy, base + 3] = -z
            modes[uz, base + 3] = y
            modes[rx, base + 3] = 1.0

            modes[ux, base + 4] = z
            modes[uz, base + 4] = -x
            modes[ry, base + 4] = 1.0

            modes[ux, base + 5] = -y
            modes[uy, base + 5] = x
            modes[rz, base + 5] = 1.0
        component_info.append(
            {
                "component_index": component_index,
                "node_ids": [int(node_id) for node_id in node_ids],
                "centroid": origin.tolist(),
                "candidate_modes": 6,
                "supported": False,
            }
        )
    return modes, component_info


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
    full_modes, component_info = _rigid_body_modes_full(model, total_dofs)
    if len(independent_dofs) == 0:
        return np.zeros((0, 0), dtype=float), {"rank": 0, "kept_mode_indices": [], "components": component_info}
    reduced_modes = full_modes[np.asarray(independent_dofs, dtype=int), :]
    q_modes, kept = _orthonormalize_columns(reduced_modes)
    return q_modes, {
        "rank": int(q_modes.shape[1]),
        "kept_mode_indices": [int(i) for i in kept],
        "component_count": len(component_info),
        "components": component_info,
        "rank_method": "modified_gram_schmidt_with_rank_tolerance",
        "description": "Six rigid-body candidates per connected component: Tx, Ty, Tz, Rx, Ry, Rz.",
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
            handle = factorize(K_red, MatrixClass.GENERAL)
            if handle.status != "ok":
                return np.zeros(K_red.shape[0]), {
                    "status": "failed",
                    "error": handle.failure_reason,
                    "backend": handle.diagnostics(),
                }
            q = handle.solve(F_red)
            return q, {"status": "converged", "backend": handle.diagnostics()}
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
    allow_unbalanced_loads: bool = False,
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

    warnings: List[str] = []
    if relative_imbalance > load_imbalance_tolerance:
        message = (
            "The external load vector has a non-zero rigid-body component. "
            "For a physical free-free static solution, use self-equilibrated loads."
        )
        if not allow_unbalanced_loads:
            return np.zeros(n), {
                "status": "incompatible_free_free_load",
                "error": message,
                "nullspace_rank": r,
                "rigid_body_load_components": load_components.tolist(),
                "rigid_body_load_imbalance_norm": load_imbalance_norm,
                "relative_rigid_body_load_imbalance": relative_imbalance,
            }
        warnings.append(
            message
            + " The nullspace solve returned a gauged displacement field and balancing generalized reactions."
        )

    try:
        handle = factorize(augmented, MatrixClass.SYMMETRIC_INDEFINITE)
        if handle.status != "ok":
            return np.zeros(n), {
                "status": "failed",
                "error": handle.failure_reason,
                "nullspace_rank": r,
                "rigid_body_load_components": load_components.tolist(),
                "relative_rigid_body_load_imbalance": relative_imbalance,
                "backend": handle.diagnostics(),
            }
        solution = handle.solve(rhs)
        solution = np.asarray(solution, dtype=float).reshape(-1)
        if np.any(np.isnan(solution)) or np.any(np.isinf(solution)):
            return np.zeros(n), {
                "status": "singular",
                "error": "NaN/Inf solution in nullspace augmented solve",
                "nullspace_rank": r,
                "rigid_body_load_components": load_components.tolist(),
                "relative_rigid_body_load_imbalance": relative_imbalance,
                "backend": handle.diagnostics(),
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
        "backend": handle.diagnostics(),
    }


def solve_linear(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    solver_type: str = "direct",
    precond: bool = True,
    constraint_mode: str = "auto",
    allow_unbalanced_free_free: bool = False,
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
        q, convergence_info = _solve_nullspace_augmented_system(
            K_red,
            F_red,
            Q,
            allow_unbalanced_loads=allow_unbalanced_free_free,
        )
    else:
        q, convergence_info = _solve_reduced_system(K_red, F_red, solver_type)
    solver_info["solve_time"] = time.time() - start_time
    solver_info["convergence_info"] = convergence_info
    result_case = make_result_case(
        name=f"linear_static:{getattr(load_case, 'name', 'none')}",
        analysis_type="linear_static",
        load_cases=() if load_case is None else (load_case,),
        assembly_info=assembly_info,
        solver_info=solver_info,
        recovery={"displacements": True, "stresses": "on_demand", "reactions": "on_demand"},
        settings={"constraint_mode": mode, "solver_type": solver_type},
        warnings=convergence_info.get("warnings", ()),
    )
    solver_info["result_case"] = result_case.to_dict()

    if convergence_info.get("status") != "converged":
        return u0.copy(), solver_info
    return reconstruct_full_solution(T, q, u0), solver_info


def solve_linear_many(
    model: "FEModel",
    load_cases: List[Optional["LoadCase"]],
    constraint_mode: str = "auto",
    factorization_cache: Optional[FactorizationCache] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve several unchanged-stiffness static load cases with one factorization.

    The returned displacement matrix has one column per load case.
    """
    model.apply_boundary_conditions()
    K, _, assembly_info = assemble_system(model, None)
    F_matrix, load_matrix_info = assemble_load_matrix(model, load_cases)
    zero_load = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    K_red, _, T, u0, independent_dofs, constraint_info = build_constraint_transformation(K, zero_load, model)
    F_red = np.asarray(T.T @ (F_matrix - (K @ u0)[:, None]), dtype=float)

    total_dofs = int(K.shape[0])
    Q, nullspace_info = build_reduced_rigid_body_modes(model, independent_dofs, total_dofs)
    mode = (constraint_mode or "auto").strip().lower()
    if mode not in {"auto", "transformation"}:
        raise ValueError("solve_linear_many supports constraint_mode 'auto' or 'transformation'")
    if mode == "auto" and int(constraint_info["num_fixed_dofs"]) == 0 and Q.shape[1] > 0:
        raise ValueError("solve_linear_many requires supports; use individual solve_linear for free-free nullspace solves")

    start = time.time()
    local_cache = factorization_cache or FactorizationCache(name="linear_static_many", max_entries=1)
    revisions = getattr(model, "revision_signature", lambda: getattr(model.mesh, "revision_signature", lambda: {})())()
    stiffness_signature = json.dumps(
        {
            "analysis": "linear_static_many",
            "matrix": "K_reduced",
            "shape": tuple(int(v) for v in K_red.shape),
            "nnz": int(K_red.nnz),
            "model_revision": revisions,
            "constraint_method": constraint_info.get("method"),
            "num_independent_dofs": int(len(independent_dofs)),
        },
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    handle = factorize_cached(
        K_red,
        MatrixClass.GENERAL,
        cache=local_cache,
        signature=stiffness_signature,
    )
    if handle.status != "ok":
        info = {
            "status": "failed",
            "error": handle.failure_reason,
            "assembly": assembly_info,
            "load_matrix": load_matrix_info,
            "constraint_info": constraint_info,
            "nullspace_info": nullspace_info,
            "backend": handle.diagnostics(),
            "factorization_cache": local_cache.diagnostics(),
            "solve_time": time.time() - start,
        }
        info["result_case"] = make_result_case(
            name="linear_static_many",
            analysis_type="linear_static_many",
            load_cases=tuple(load_case for load_case in load_cases if load_case is not None),
            assembly_info={**assembly_info, "load_matrix": load_matrix_info},
            solver_info=info,
            recovery={"displacements": True, "stresses": "on_demand", "reactions": "on_demand"},
            settings={"constraint_mode": mode, "num_load_cases": len(load_cases)},
        ).to_dict()
        return np.tile(u0.reshape(-1, 1), (1, len(load_cases))), info
    q_matrix = handle.solve_many(F_red)
    full = np.asarray(T @ q_matrix + u0[:, None], dtype=float)
    info = {
        "status": "converged",
        "assembly": assembly_info,
        "load_matrix": load_matrix_info,
        "constraint_info": constraint_info,
        "nullspace_info": nullspace_info,
        "backend": handle.diagnostics(),
        "factorization_cache": local_cache.diagnostics(),
        "solve_time": time.time() - start,
        "num_result_cases": len(load_cases),
    }
    info["result_case"] = make_result_case(
        name="linear_static_many",
        analysis_type="linear_static_many",
        load_cases=tuple(load_case for load_case in load_cases if load_case is not None),
        assembly_info={**assembly_info, "load_matrix": load_matrix_info},
        solver_info=info,
        recovery={"displacements": True, "stresses": "on_demand", "reactions": "on_demand"},
        settings={"constraint_mode": mode, "num_load_cases": len(load_cases)},
    ).to_dict()
    return full, info


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


def _add_dof_force(force_map: Dict[int, np.ndarray], model: "FEModel", dof: int, value: float) -> None:
    """Accumulate one global-DOF force into a node-indexed six-component map."""
    node_id, local_index, _name = model.mesh.dof_manager.get_dof_info(int(dof))
    if node_id < 0 or local_index < 0:
        return
    force_map.setdefault(int(node_id), np.zeros(6, dtype=float))[int(local_index)] += float(value)


def _compact_force_map(force_map: Dict[int, np.ndarray], tolerance: float = 0.0) -> Dict[int, np.ndarray]:
    """Drop near-zero entries from a node force map."""
    compact: Dict[int, np.ndarray] = {}
    for node_id, values in force_map.items():
        vector = np.asarray(values, dtype=float).reshape(6)
        if np.any(np.abs(vector) > float(tolerance)):
            compact[int(node_id)] = vector
    return compact


def compute_constraint_force_diagnostics(
    model: "FEModel",
    displacements: np.ndarray,
    load_case: Optional["LoadCase"] = None,
    *,
    force_tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Return separated support, MPC and nullspace force diagnostics.

    The raw residual convention is ``K u - F`` on the unreduced global system.
    ``support_reactions`` contains only fixed-DOF residuals. ``mpc_slave_forces``
    contains residuals on slave DOFs, while ``mpc_master_equivalent_forces``
    pushes those slave residuals through the linear MPC coefficients to show
    the equivalent force/moment components seen by master DOFs.
    """
    mesh = model.mesh
    dof_manager = mesh.dof_manager
    model.apply_boundary_conditions()

    K, _, _ = assemble_system(model)
    if load_case is None:
        F_ext = np.zeros(dof_manager.total_dofs, dtype=float)
    else:
        F_ext = load_case.get_load_vector(mesh, dof_manager, model.get_material)

    u = np.asarray(displacements, dtype=float).reshape(-1)
    if u.shape[0] != int(K.shape[0]):
        raise ValueError(f"Displacement vector length {u.shape[0]} does not match system size {K.shape[0]}")

    residual = np.asarray(K @ u - F_ext, dtype=float).reshape(-1)
    fixed_dofs = sorted(int(dof) for dof in getattr(dof_manager, "_constrained_dofs", set()))
    mpc_constraints = _collect_mpc_constraints(model)

    support_reactions: Dict[int, np.ndarray] = {}
    mpc_slave_forces: Dict[int, np.ndarray] = {}
    mpc_master_equivalent_forces: Dict[int, np.ndarray] = {}
    mpc_constraint_forces: List[Dict[str, Any]] = []

    for dof in fixed_dofs:
        _add_dof_force(support_reactions, model, dof, residual[dof])

    for index, constraint in enumerate(mpc_constraints):
        slave = int(constraint["slave"])
        slave_force = float(residual[slave])
        slave_node, slave_local, slave_component = dof_manager.get_dof_info(slave)
        _add_dof_force(mpc_slave_forces, model, slave, slave_force)

        master_entries: List[Dict[str, Any]] = []
        for master, coefficient in constraint.get("masters", {}).items():
            master_dof = int(master)
            value = float(coefficient) * slave_force
            _add_dof_force(mpc_master_equivalent_forces, model, master_dof, value)
            master_node, master_local, master_component = dof_manager.get_dof_info(master_dof)
            master_entries.append(
                {
                    "dof": master_dof,
                    "node_id": int(master_node),
                    "local_index": int(master_local),
                    "component": str(master_component),
                    "coefficient": float(coefficient),
                    "equivalent_force": value,
                }
            )

        mpc_constraint_forces.append(
            {
                "index": int(index),
                "label": str(constraint.get("label", f"mpc_{index}")),
                "slave_dof": slave,
                "slave_node_id": int(slave_node),
                "slave_local_index": int(slave_local),
                "slave_component": str(slave_component),
                "slave_force": slave_force,
                "master_equivalent_forces": master_entries,
                "master_equivalent_norm": float(np.linalg.norm([entry["equivalent_force"] for entry in master_entries])),
            }
        )

    K_red, _, T, _, independent_dofs, constraint_info = build_constraint_transformation(K, F_ext, model)
    reduced_residual = np.asarray(T.T @ residual, dtype=float).reshape(-1)
    Q, nullspace_info = build_reduced_rigid_body_modes(model, independent_dofs, int(K.shape[0]))
    if Q.shape[1] > 0:
        nullspace_generalized_forces = np.asarray(Q.T @ reduced_residual, dtype=float).reshape(-1)
    else:
        nullspace_generalized_forces = np.zeros(0, dtype=float)

    return {
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "reduced_residual_norm": float(np.linalg.norm(reduced_residual)),
        "fixed_dofs": fixed_dofs,
        "mpc_slave_dofs": sorted(int(constraint["slave"]) for constraint in mpc_constraints),
        "support_reactions": _compact_force_map(support_reactions, force_tolerance),
        "mpc_slave_forces": _compact_force_map(mpc_slave_forces, force_tolerance),
        "mpc_master_equivalent_forces": _compact_force_map(mpc_master_equivalent_forces, force_tolerance),
        "mpc_constraint_forces": mpc_constraint_forces,
        "support_reaction_norm": float(np.linalg.norm(np.concatenate(list(support_reactions.values()))) if support_reactions else 0.0),
        "mpc_slave_force_norm": float(np.linalg.norm(np.concatenate(list(mpc_slave_forces.values()))) if mpc_slave_forces else 0.0),
        "mpc_master_equivalent_force_norm": float(
            np.linalg.norm(np.concatenate(list(mpc_master_equivalent_forces.values()))) if mpc_master_equivalent_forces else 0.0
        ),
        "nullspace_generalized_forces": nullspace_generalized_forces,
        "nullspace_generalized_force_norm": float(np.linalg.norm(nullspace_generalized_forces)),
        "constraint_info": constraint_info,
        "nullspace_info": nullspace_info,
    }


def compute_reactions(model: "FEModel", displacements: np.ndarray, load_case: "LoadCase") -> Dict[int, np.ndarray]:
    """Compute legacy combined reactions at fixed and MPC slave DOFs."""
    diagnostics = compute_constraint_force_diagnostics(model, displacements, load_case)
    reactions: Dict[int, np.ndarray] = {}
    for bucket in ("support_reactions", "mpc_slave_forces"):
        for node_id, values in diagnostics[bucket].items():
            reactions.setdefault(int(node_id), np.zeros(6, dtype=float))
            reactions[int(node_id)] += np.asarray(values, dtype=float).reshape(6)
    return _compact_force_map(reactions)


def compute_stresses(
    model: "FEModel",
    displacements: np.ndarray,
    return_global: bool = False,
    element_ids: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute stresses for all or selected elements."""
    mesh = model.mesh
    stresses: Dict[int, Dict[str, np.ndarray]] = {}
    displacements = np.asarray(displacements, dtype=float)
    selected = None if element_ids is None else {int(element_id) for element_id in element_ids}
    for elem_id, element in mesh.elements.items():
        if selected is not None and int(elem_id) not in selected:
            continue
        material = model.get_material(element.material_name)
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0 or int(dof_mapping.max()) >= displacements.size:
            continue
        try:
            stresses[elem_id] = element.compute_stresses(
                mesh, displacements[dof_mapping], material, return_global=return_global
            )
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
