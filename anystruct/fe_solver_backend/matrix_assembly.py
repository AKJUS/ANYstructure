"""Explicit stiffness, mass and load assembly APIs.

This module is the step-3 public assembly interface.  It keeps K, M and F
assembly separate so modal, buckling and nonlinear solvers can choose exactly
which matrices they need without side effects.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from .boundary import LoadCase
    from .fe_core import FEModel


class AssemblyError(ValueError):
    """Raised when an element returns an invalid matrix or load contribution."""


def _base_info(model: "FEModel", matrix_type: str) -> Dict[str, Any]:
    mesh = model.mesh
    return {
        "matrix_type": matrix_type,
        "num_elements": 0,
        "num_nodes": mesh.num_nodes,
        "total_dofs": mesh.dof_manager.total_dofs,
        "assembly_time": 0.0,
        "element_times": {},
        "skipped_elements": [],
        "diagnostics": {},
        "revision_signature": getattr(mesh, "revision_signature", lambda: {})(),
    }


def _check_element_matrix_shape(element_id: int, matrix_name: str, matrix: np.ndarray, expected_size: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    expected_shape = (expected_size, expected_size)
    if matrix.shape != expected_shape:
        raise AssemblyError(
            f"Element {element_id} returned {matrix_name} with shape {matrix.shape}; "
            f"expected {expected_shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise AssemblyError(f"Element {element_id} returned non-finite values in {matrix_name}.")
    return matrix


def _relative_symmetry_error(matrix: sparse.spmatrix | np.ndarray) -> float:
    if sparse.issparse(matrix):
        diff = matrix - matrix.T
        numerator = float(sparse.linalg.norm(diff))
        denominator = max(float(sparse.linalg.norm(matrix)), 1.0)
        return numerator / denominator
    dense = np.asarray(matrix, dtype=float)
    return float(np.linalg.norm(dense - dense.T) / max(np.linalg.norm(dense), 1.0))


def _topology_signature(mesh: Any, matrix_type: str) -> str:
    revisions = getattr(mesh, "revision_signature", lambda: {})()
    payload = {
        "matrix_type": matrix_type,
        "topology_revision": revisions.get("topology", 0),
        "mpc_revision": revisions.get("mpc", 0),
        "elements": [
            {
                "id": int(elem_id),
                "class": element.__class__.__name__,
                "node_ids": [int(node_id) for node_id in getattr(element, "node_ids", [])],
                "dofs": [int(dof) for dof in element.get_dof_mapping(mesh)],
            }
            for elem_id, element in mesh.elements.items()
        ],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scatter_element_matrix(
    element_matrix: np.ndarray,
    dof_mapping: np.ndarray,
    rows: list,
    cols: list,
    data: list,
) -> None:
    """Append element matrix entries to COO triplet buffers (vectorized)."""
    n_local = dof_mapping.size
    values = element_matrix.ravel()
    mask = values != 0.0
    if not np.any(mask):
        return
    rows.append(np.repeat(dof_mapping, n_local)[mask])
    cols.append(np.tile(dof_mapping, n_local)[mask])
    data.append(values[mask])


def _triplets_to_csr(rows: list, cols: list, data: list, total_dofs: int) -> sparse.csr_matrix:
    """Build a CSR matrix from COO triplet buffers; duplicates are summed."""
    if not data:
        return sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
    coo = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(total_dofs, total_dofs),
        dtype=float,
    )
    return coo.tocsr()


def _get_cached_sparsity_pattern(mesh: "FEMesh", matrix_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve or build the cached row and column indices for global matrix COO assembly."""
    if not hasattr(mesh, "_sparsity_cache"):
        mesh._sparsity_cache = {}

    signature = _topology_signature(mesh, matrix_type)

    if matrix_type in mesh._sparsity_cache:
        cached = mesh._sparsity_cache[matrix_type]
        if cached.get("signature") == signature:
            return cached["rows"], cached["cols"]

    rows_list = []
    cols_list = []
    for _, element in mesh.elements.items():
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            continue
        n_local = dof_mapping.size
        rows_list.append(np.repeat(dof_mapping, n_local))
        cols_list.append(np.tile(dof_mapping, n_local))

    rows_concat = np.concatenate(rows_list) if rows_list else np.empty(0, dtype=np.intp)
    cols_concat = np.concatenate(cols_list) if cols_list else np.empty(0, dtype=np.intp)

    mesh._sparsity_cache[matrix_type] = {
        "rows": rows_concat,
        "cols": cols_concat,
        "signature": signature,
    }
    return rows_concat, cols_concat


def _assemble_element_matrix(
    model: "FEModel",
    matrix_type: str,
    element_matrix_getter: Callable[[Any, Any, Any], np.ndarray],
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    mesh = model.mesh
    total_dofs = mesh.dof_manager.total_dofs
    info = _base_info(model, matrix_type)
    start_time = time.time()

    # Precompute shell stiffnesses in a JIT-compiled batch if doing stiffness matrix assembly
    precomputed = {}
    vectorized_shell_groups = []
    if matrix_type == "stiffness":
        from .elements import ShellElement
        from .jit_compiler import JIT_ENABLED, JIT_DISABLED_REASON, jit_diagnostics
        from .vectorized_stiffness import compute_shell_stiffness_matrices_jit

        groups = {}
        for elem_id, element in mesh.elements.items():
            if (
                isinstance(element, ShellElement)
                and getattr(element, "_is_quadrilateral", False)
                and not (getattr(element, "_is_8node", False) and bool(getattr(element, "reduced_integration", False)))
            ):
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

        for key, elem_list in groups.items():
            num_nodes, thickness, drilling_stabilization, _reduced_integration, _hourglass_stabilization, material_name = key
            material = model.get_material(material_name)
            E = float(material.elastic_modulus)
            nu = float(material.poisson_ratio)
            G = float(material.shear_modulus)

            n_elem = len(elem_list)
            coords_all = np.zeros((n_elem, num_nodes, 3))
            for idx, (elem_id, element) in enumerate(elem_list):
                coords_all[idx] = element.get_node_coordinates(mesh)

            first_element = elem_list[0][1]
            is_4node = first_element._is_4node
            gauss_points = first_element.gauss_points
            gauss_weights = first_element.gauss_weights
            if is_4node:
                shear_points = np.empty((0, 2))
                shear_weights = np.empty(0)
            else:
                shear_points = first_element.shear_gauss_points
                shear_weights = first_element.shear_gauss_weights

            stiffnesses = compute_shell_stiffness_matrices_jit(
                coords_all,
                is_4node,
                thickness,
                drilling_stabilization,
                E,
                nu,
                G,
                gauss_points,
                gauss_weights,
                shear_points,
                shear_weights,
            )

            for idx, (elem_id, element) in enumerate(elem_list):
                precomputed[elem_id] = stiffnesses[idx]
            jit_info = jit_diagnostics()
            vectorized_shell_groups.append(
                {
                    "shell_order": "S4" if is_4node else "Q8",
                    "num_elements": int(n_elem),
                    "num_nodes": int(num_nodes),
                    "material": str(material_name),
                    "thickness": float(thickness),
                    "jit_enabled": bool(JIT_ENABLED),
                    "jit_disabled_reason": JIT_DISABLED_REASON,
                    "kernel": "compute_shell_stiffness_matrices_jit",
                    "parallel_kernel": True,
                    "parallel_threads": jit_info.get("num_threads"),
                    "backend": jit_info.get("backend"),
                }
            )

    # Retrieve or build cached sparsity pattern
    rows_concat, cols_concat = _get_cached_sparsity_pattern(mesh, matrix_type)

    data_list = []
    for elem_id, element in mesh.elements.items():
        elem_start = time.time()
        material = model.get_material(element.material_name)
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            info["skipped_elements"].append(int(elem_id))
            continue

        if elem_id in precomputed:
            element_matrix = precomputed[elem_id]
        else:
            element_matrix = element_matrix_getter(element, mesh, material)

        element_matrix = _check_element_matrix_shape(
            int(elem_id),
            matrix_type,
            element_matrix,
            int(dof_mapping.size),
        )
        if matrix_type in {"stiffness", "mass", "geometric_stiffness"}:
            local_symmetry = _relative_symmetry_error(element_matrix)
            if local_symmetry > 1.0e-8:
                raise AssemblyError(
                    f"Element {elem_id} returned nonsymmetric {matrix_type}; "
                    f"relative symmetry error {local_symmetry:.3e}."
                )
        data_list.append(np.asarray(element_matrix, dtype=float).ravel())

        info["element_times"][int(elem_id)] = time.time() - elem_start
        info["num_elements"] += 1

    info["assembly_time"] = time.time() - start_time
    
    if not data_list:
        matrix = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
        info["diagnostics"]["assembled_symmetry_error"] = 0.0
        return matrix, info
        
    data_concat = np.concatenate(data_list)
    coo = sparse.coo_matrix(
        (data_concat, (rows_concat, cols_concat)),
        shape=(total_dofs, total_dofs),
        dtype=float,
    )
    matrix = coo.tocsr()
    info["diagnostics"]["assembled_symmetry_error"] = _relative_symmetry_error(matrix)
    if matrix_type == "stiffness":
        info["diagnostics"]["vectorized_shell_groups"] = vectorized_shell_groups
        info["diagnostics"]["vectorized_shell_element_count"] = int(len(precomputed))
        info["diagnostics"]["scalar_shell_element_count"] = int(info["num_elements"] - len(precomputed))
    info["sparsity_signature"] = _topology_signature(mesh, matrix_type)
    return matrix, info


def assemble_stiffness_matrix(model: "FEModel") -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble the global stiffness matrix K only."""
    return _assemble_element_matrix(
        model,
        "stiffness",
        lambda element, mesh, material: element.compute_stiffness_matrix(mesh, material),
    )


def assemble_mass_matrix(model: "FEModel") -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble the global mass matrix M only."""
    return _assemble_element_matrix(
        model,
        "mass",
        lambda element, mesh, material: element.compute_mass_matrix(mesh, material),
    )


def _get_element_state(element_states: Optional[Any], element_id: int, element: Any) -> Any:
    if element_states is None:
        return None
    if callable(element_states):
        try:
            return element_states(element_id, element)
        except TypeError:
            return element_states(element_id)
    if isinstance(element_states, Mapping):
        if element_id in element_states:
            return element_states[element_id]
        element_id_text = str(element_id)
        if element_id_text in element_states:
            return element_states[element_id_text]
    return None


def assemble_geometric_stiffness_matrix(
    model: "FEModel",
    element_states: Optional[Any] = None,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble the global geometric stiffness matrix KG only.

    ``element_states`` supplies the reference stress/resultant state for each
    element.  The current beam-column implementation accepts a numeric value or
    a mapping with ``axial_compression`` positive in compression.
    """
    mesh = model.mesh
    total_dofs = mesh.dof_manager.total_dofs
    info = _base_info(model, "geometric_stiffness")
    start_time = time.time()

    # Retrieve or build cached sparsity pattern
    rows_concat, cols_concat = _get_cached_sparsity_pattern(mesh, "geometric_stiffness")

    data_list = []
    for elem_id, element in mesh.elements.items():
        elem_start = time.time()
        material = model.get_material(element.material_name)
        dof_mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp)
        if dof_mapping.size == 0:
            info["skipped_elements"].append(int(elem_id))
            continue

        state = _get_element_state(element_states, int(elem_id), element)
        getter = getattr(element, "compute_geometric_stiffness_matrix", None)
        if getter is None:
            element_matrix = np.zeros((dof_mapping.size, dof_mapping.size), dtype=float)
        else:
            element_matrix = getter(mesh, material, state)
        element_matrix = _check_element_matrix_shape(
            int(elem_id),
            "geometric_stiffness",
            element_matrix,
            int(dof_mapping.size),
        )
        data_list.append(np.asarray(element_matrix, dtype=float).ravel())

        info["element_times"][int(elem_id)] = time.time() - elem_start
        info["num_elements"] += 1

    info["assembly_time"] = time.time() - start_time
    info["state_source"] = "none" if element_states is None else type(element_states).__name__
    
    if not data_list:
        matrix = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
        info["diagnostics"]["assembled_symmetry_error"] = 0.0
        return matrix, info
        
    data_concat = np.concatenate(data_list)
    coo = sparse.coo_matrix(
        (data_concat, (rows_concat, cols_concat)),
        shape=(total_dofs, total_dofs),
        dtype=float,
    )
    matrix = coo.tocsr()
    info["diagnostics"]["assembled_symmetry_error"] = _relative_symmetry_error(matrix)
    info["sparsity_signature"] = _topology_signature(mesh, "geometric_stiffness")
    return matrix, info


def assemble_load_vector(model: "FEModel", load_case: Optional["LoadCase"] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Assemble the global external load vector F only."""
    total_dofs = model.mesh.dof_manager.total_dofs
    start_time = time.time()
    if load_case is None:
        load_vector = np.zeros(total_dofs, dtype=float)
        load_name = None
    else:
        load_vector = load_case.get_load_vector(model.mesh, model.mesh.dof_manager, model.get_material)
        load_vector = np.asarray(load_vector, dtype=float).reshape(-1)
        load_name = load_case.name

    if load_vector.shape != (total_dofs,):
        raise AssemblyError(f"Load vector shape {load_vector.shape} does not match total DOFs {(total_dofs,)}.")
    if not np.all(np.isfinite(load_vector)):
        raise AssemblyError(f"Load case {load_name!r} produced non-finite load vector values.")

    return load_vector, {
        "vector_type": "load",
        "load_case": load_name,
        "num_nodes": model.mesh.num_nodes,
        "total_dofs": total_dofs,
        "assembly_time": time.time() - start_time,
        "load_norm": float(np.linalg.norm(load_vector)),
    }


def assemble_load_matrix(
    model: "FEModel",
    load_cases: Sequence[Optional["LoadCase"]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Assemble a dense load matrix with one column per load case."""
    start = time.time()
    vectors = []
    infos = []
    names = []
    for load_case in load_cases:
        vector, info = assemble_load_vector(model, load_case)
        vectors.append(vector)
        infos.append(info)
        names.append(None if load_case is None else load_case.name)
    total_dofs = model.mesh.dof_manager.total_dofs
    matrix = np.column_stack(vectors) if vectors else np.zeros((total_dofs, 0), dtype=float)
    return matrix, {
        "vector_type": "load_matrix",
        "load_cases": names,
        "num_load_cases": len(names),
        "total_dofs": total_dofs,
        "assembly_time": time.time() - start,
        "columns": infos,
        "load_norms": [float(np.linalg.norm(matrix[:, idx])) for idx in range(matrix.shape[1])],
        "revision_signature": getattr(model.mesh, "revision_signature", lambda: {})(),
    }


def assemble_damping_matrix(
    model: "FEModel",
    rayleigh_alpha: float = 0.0,
    rayleigh_beta: float = 0.0,
) -> Tuple[sparse.csr_matrix, Dict[str, Any]]:
    """Assemble Rayleigh damping C = alpha M + beta K."""
    start = time.time()
    M, mass_info = assemble_mass_matrix(model)
    K, stiffness_info = assemble_stiffness_matrix(model)
    C = (float(rayleigh_alpha) * M + float(rayleigh_beta) * K).tocsr()
    return C, {
        "matrix_type": "damping",
        "rayleigh_alpha": float(rayleigh_alpha),
        "rayleigh_beta": float(rayleigh_beta),
        "mass": mass_info,
        "stiffness": stiffness_info,
        "assembly_time": time.time() - start,
        "diagnostics": {"assembled_symmetry_error": _relative_symmetry_error(C)},
        "revision_signature": getattr(model.mesh, "revision_signature", lambda: {})(),
    }


def assemble_system(
    model: "FEModel",
    load_case: Optional["LoadCase"] = None,
    include_mass: bool = False,
) -> Tuple[sparse.csr_matrix, np.ndarray, Dict[str, Any]]:
    """Compatibility wrapper returning K, F and assembly metadata.

    The mass matrix is assembled separately and returned in info["mass_matrix"]
    only when include_mass is true.  It is never added to stiffness.
    """
    start_time = time.time()
    K, stiffness_info = assemble_stiffness_matrix(model)
    F, load_info = assemble_load_vector(model, load_case)

    info: Dict[str, Any] = {
        "num_elements": stiffness_info["num_elements"],
        "num_nodes": model.mesh.num_nodes,
        "total_dofs": model.mesh.dof_manager.total_dofs,
        "includes_mass_matrix": bool(include_mass),
        "assembly_time": 0.0,
        "stiffness": stiffness_info,
        "load": load_info,
        # Backwards-compatible keys used by older diagnostics/tests.
        "element_times": stiffness_info.get("element_times", {}),
    }

    if include_mass:
        M, mass_info = assemble_mass_matrix(model)
        info["mass_matrix"] = M
        info["mass"] = mass_info

    info["assembly_time"] = time.time() - start_time
    return K, F, info
