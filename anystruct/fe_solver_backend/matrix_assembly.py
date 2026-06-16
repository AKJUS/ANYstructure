"""Explicit stiffness, mass and load assembly APIs.

This module is the step-3 public assembly interface.  It keeps K, M and F
assembly separate so modal, buckling and nonlinear solvers can choose exactly
which matrices they need without side effects.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Optional, Tuple

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
    }


def _check_element_matrix_shape(element_id: int, matrix_name: str, matrix: np.ndarray, expected_size: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    expected_shape = (expected_size, expected_size)
    if matrix.shape != expected_shape:
        raise AssemblyError(
            f"Element {element_id} returned {matrix_name} with shape {matrix.shape}; "
            f"expected {expected_shape}."
        )
    return matrix


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

    current_elem_ids = list(mesh.elements.keys())

    if matrix_type in mesh._sparsity_cache:
        cached = mesh._sparsity_cache[matrix_type]
        if cached["elem_ids"] == current_elem_ids:
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
        "elem_ids": current_elem_ids,
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

        element_matrix = element_matrix_getter(element, mesh, material)
        element_matrix = _check_element_matrix_shape(
            int(elem_id),
            matrix_type,
            element_matrix,
            int(dof_mapping.size),
        )
        data_list.append(np.asarray(element_matrix, dtype=float).ravel())

        info["element_times"][int(elem_id)] = time.time() - elem_start
        info["num_elements"] += 1

    info["assembly_time"] = time.time() - start_time
    
    if not data_list:
        return sparse.csr_matrix((total_dofs, total_dofs), dtype=float), info
        
    data_concat = np.concatenate(data_list)
    coo = sparse.coo_matrix(
        (data_concat, (rows_concat, cols_concat)),
        shape=(total_dofs, total_dofs),
        dtype=float,
    )
    return coo.tocsr(), info


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
        return sparse.csr_matrix((total_dofs, total_dofs), dtype=float), info
        
    data_concat = np.concatenate(data_list)
    coo = sparse.coo_matrix(
        (data_concat, (rows_concat, cols_concat)),
        shape=(total_dofs, total_dofs),
        dtype=float,
    )
    return coo.tocsr(), info


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

    return load_vector, {
        "vector_type": "load",
        "load_case": load_name,
        "num_nodes": model.mesh.num_nodes,
        "total_dofs": total_dofs,
        "assembly_time": time.time() - start_time,
        "load_norm": float(np.linalg.norm(load_vector)),
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
