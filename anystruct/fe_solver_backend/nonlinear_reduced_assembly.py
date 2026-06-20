"""Constraint-aware direct reduced-coordinate nonlinear assembly core."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse

from .jit_compiler import njit


@njit(cache=True)
def _scatter_reduced_values(
    local_values: np.ndarray,
    unit_positions: np.ndarray,
    weighted_sources: np.ndarray,
    weighted_positions: np.ndarray,
    weighted_coefficients: np.ndarray,
    output: np.ndarray,
) -> None:
    """Scatter local values into a retained reduced-coordinate output buffer."""

    output.fill(0.0)
    for source in range(unit_positions.size):
        position = int(unit_positions[source])
        if position >= 0:
            output[position] += local_values[source]
    for index in range(weighted_sources.size):
        output[int(weighted_positions[index])] += (
            local_values[int(weighted_sources[index])] * weighted_coefficients[index]
        )


@dataclass
class ReducedAssemblyTimings:
    builds: int = 0
    assemblies: int = 0
    residual_only_assemblies: int = 0
    local_response_seconds: float = 0.0
    reduced_force_scatter_seconds: float = 0.0
    reduced_tangent_scatter_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "builds": int(self.builds),
            "assemblies": int(self.assemblies),
            "residual_only_assemblies": int(self.residual_only_assemblies),
            "local_response_seconds": float(self.local_response_seconds),
            "reduced_force_scatter_seconds": float(self.reduced_force_scatter_seconds),
            "reduced_tangent_scatter_seconds": float(self.reduced_tangent_scatter_seconds),
            "total_seconds": float(self.total_seconds),
        }


@dataclass
class ReducedAssemblyPlan:
    """Precomputed direct scatter from element-local values to reduced CSR data."""

    transformation: sparse.csr_matrix
    total_dofs: int
    reduced_dofs: int
    force_unit_positions: np.ndarray
    force_weighted_sources: np.ndarray
    force_weighted_positions: np.ndarray
    force_weighted_coefficients: np.ndarray
    tangent_unit_positions: np.ndarray
    tangent_weighted_sources: np.ndarray
    tangent_weighted_positions: np.ndarray
    tangent_weighted_coefficients: np.ndarray
    csr_indptr: np.ndarray
    csr_indices: np.ndarray
    force_buffer: np.ndarray
    tangent_buffer: np.ndarray
    tangent_matrix: sparse.csr_matrix
    setup_seconds: float
    force_contributions: int
    tangent_contributions: int
    tangent_expansion_ratio: float
    estimated_map_bytes: int
    mapping_kind: str
    source_plan: Any = field(repr=False)
    timings: ReducedAssemblyTimings = field(default_factory=ReducedAssemblyTimings)

    @property
    def nnz(self) -> int:
        return int(self.csr_indices.size)

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "total_dofs": int(self.total_dofs),
            "reduced_dofs": int(self.reduced_dofs),
            "reduction_ratio": float(self.reduced_dofs / max(self.total_dofs, 1)),
            "reduced_tangent_nnz": self.nnz,
            "force_contributions": int(self.force_contributions),
            "tangent_contributions": int(self.tangent_contributions),
            "tangent_expansion_ratio": float(self.tangent_expansion_ratio),
            "estimated_map_bytes": int(self.estimated_map_bytes),
            "mapping_kind": str(self.mapping_kind),
            "setup_seconds": float(self.setup_seconds),
            "timings": self.timings.to_dict(),
        }


class ReducedAssemblyPlanLimit(RuntimeError):
    """Raised when a direct reduced scatter map exceeds the configured memory cap."""


def _maximum_map_bytes() -> int:
    raw = os.environ.get("FE_SOLVER_BATCH_C_MAX_MAP_MB", "512").strip()
    try:
        megabytes = max(float(raw), 1.0)
    except ValueError:
        megabytes = 512.0
    return int(megabytes * 1024.0 * 1024.0)


def _identity_transformation(transformation: sparse.spmatrix) -> bool:
    T = sparse.csr_matrix(transformation, dtype=float)
    if T.shape[0] != T.shape[1] or T.nnz != T.shape[0]:
        return False
    T.sum_duplicates()
    T.sort_indices()
    expected = np.arange(T.shape[0], dtype=T.indices.dtype)
    return bool(
        np.array_equal(T.indptr, np.arange(T.shape[0] + 1))
        and np.array_equal(T.indices, expected)
        and np.all(T.data == 1.0)
    )


def _preflight_reduced_map_bytes(
    nonlinear_plan: Any,
    transformation: sparse.csr_matrix,
) -> Tuple[int, int]:
    """Estimate retained map memory before allocating expanded MPC arrays."""

    force_dofs = np.asarray(nonlinear_plan.force_dofs_flat, dtype=np.intp)
    row_nnz = np.diff(transformation.indptr)
    force_counts = row_nnz[force_dofs]
    force_unit = np.zeros(force_counts.size, dtype=bool)
    force_simple = np.flatnonzero(force_counts == 1)
    if force_simple.size:
        pointers = transformation.indptr[force_dofs[force_simple]]
        force_unit[force_simple] = transformation.data[pointers] == 1.0
    force_contributions = int(np.sum(force_counts, dtype=np.int64))
    force_weighted = force_contributions - int(np.count_nonzero(force_unit))

    source_positions = np.asarray(nonlinear_plan.tangent_scatter, dtype=np.intp)
    source_rows = np.searchsorted(
        np.asarray(nonlinear_plan.csr_indptr, dtype=np.intp),
        source_positions,
        side="right",
    ) - 1
    source_columns = nonlinear_plan.csr_indices[source_positions]
    tangent_counts = (
        row_nnz[source_rows].astype(np.int64)
        * row_nnz[source_columns].astype(np.int64)
    )
    tangent_contributions = int(np.sum(tangent_counts, dtype=np.int64))
    tangent_unit = np.zeros(tangent_counts.size, dtype=bool)
    simple_sources = np.flatnonzero(tangent_counts == 1)
    if simple_sources.size:
        row_ptr = transformation.indptr[source_rows[simple_sources]]
        column_ptr = transformation.indptr[source_columns[simple_sources]]
        tangent_unit[simple_sources] = (
            transformation.data[row_ptr] * transformation.data[column_ptr]
        ) == 1.0
    tangent_weighted = tangent_contributions - int(np.count_nonzero(tangent_unit))

    max_row_nnz = int(np.max(row_nnz)) if row_nnz.size else 0
    estimated_reduced_nnz = min(
        tangent_contributions,
        int(nonlinear_plan.nnz) * max(max_row_nnz * max_row_nnz, 1),
    )
    estimated = int(
        4 * int(nonlinear_plan.force_values.size)
        + 16 * force_weighted
        + 4 * int(nonlinear_plan.tangent_values.size)
        + 16 * tangent_weighted
        + 16 * estimated_reduced_nnz
        + 8 * int(transformation.shape[1])
        + 8 * (int(transformation.shape[1]) + 1)
    )
    return estimated, tangent_contributions


def _append_force_mapping(
    transformation: sparse.csr_matrix,
    full_dofs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row_counts = np.diff(transformation.indptr)[full_dofs]
    unit_sources: List[np.ndarray] = []
    unit_keys: List[np.ndarray] = []
    weighted_sources: List[np.ndarray] = []
    weighted_keys: List[np.ndarray] = []
    weighted_coefficients: List[np.ndarray] = []

    simple_sources = np.flatnonzero(row_counts == 1)
    if simple_sources.size:
        pointers = transformation.indptr[full_dofs[simple_sources]]
        columns = transformation.indices[pointers]
        coefficients = transformation.data[pointers]
        unit_mask = coefficients == 1.0
        if np.any(unit_mask):
            unit_sources.append(simple_sources[unit_mask].astype(np.int32, copy=False))
            unit_keys.append(columns[unit_mask].astype(np.int64, copy=False))
        if np.any(~unit_mask):
            weighted_sources.append(
                simple_sources[~unit_mask].astype(np.int32, copy=False)
            )
            weighted_keys.append(columns[~unit_mask].astype(np.int64, copy=False))
            weighted_coefficients.append(
                coefficients[~unit_mask].astype(float, copy=False)
            )

    complex_sources = np.flatnonzero(row_counts > 1)
    for source in complex_sources:
        full_dof = int(full_dofs[source])
        start = int(transformation.indptr[full_dof])
        stop = int(transformation.indptr[full_dof + 1])
        columns = transformation.indices[start:stop]
        coefficients = transformation.data[start:stop]
        weighted_sources.append(np.full(columns.size, int(source), dtype=np.int32))
        weighted_keys.append(columns.astype(np.int64, copy=False))
        weighted_coefficients.append(coefficients.astype(float, copy=False))

    return (
        np.concatenate(unit_sources) if unit_sources else np.empty(0, dtype=np.int32),
        np.concatenate(unit_keys) if unit_keys else np.empty(0, dtype=np.int64),
        np.concatenate(weighted_sources)
        if weighted_sources
        else np.empty(0, dtype=np.int32),
        np.concatenate(weighted_keys)
        if weighted_keys
        else np.empty(0, dtype=np.int64),
        np.concatenate(weighted_coefficients)
        if weighted_coefficients
        else np.empty(0, dtype=float),
    )


def _append_tangent_mapping(
    nonlinear_plan: Any,
    transformation: sparse.csr_matrix,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    source_positions = np.asarray(nonlinear_plan.tangent_scatter, dtype=np.intp)
    source_rows = np.searchsorted(
        np.asarray(nonlinear_plan.csr_indptr, dtype=np.intp),
        source_positions,
        side="right",
    ) - 1
    source_columns = nonlinear_plan.csr_indices[source_positions]
    row_counts = np.diff(transformation.indptr)[source_rows]
    column_counts = np.diff(transformation.indptr)[source_columns]
    expansion_counts = row_counts.astype(np.int64) * column_counts.astype(np.int64)
    total_contributions = int(np.sum(expansion_counts, dtype=np.int64))

    unit_sources: List[np.ndarray] = []
    unit_keys: List[np.ndarray] = []
    weighted_sources: List[np.ndarray] = []
    weighted_keys: List[np.ndarray] = []
    weighted_coefficients: List[np.ndarray] = []
    n_reduced = int(transformation.shape[1])

    simple = (row_counts == 1) & (column_counts == 1)
    if np.any(simple):
        sources = np.flatnonzero(simple)
        row_ptr = transformation.indptr[source_rows[sources]]
        column_ptr = transformation.indptr[source_columns[sources]]
        reduced_rows = transformation.indices[row_ptr]
        reduced_columns = transformation.indices[column_ptr]
        coefficients = transformation.data[row_ptr] * transformation.data[column_ptr]
        keys = (
            reduced_rows.astype(np.int64) * np.int64(n_reduced)
            + reduced_columns.astype(np.int64)
        )
        unit_mask = coefficients == 1.0
        if np.any(unit_mask):
            unit_sources.append(sources[unit_mask].astype(np.int32, copy=False))
            unit_keys.append(keys[unit_mask])
        if np.any(~unit_mask):
            weighted_sources.append(sources[~unit_mask].astype(np.int32, copy=False))
            weighted_keys.append(keys[~unit_mask])
            weighted_coefficients.append(
                coefficients[~unit_mask].astype(float, copy=False)
            )

    complex_sources = np.flatnonzero((expansion_counts > 0) & ~simple)
    for source in complex_sources:
        row = int(source_rows[source])
        column = int(source_columns[source])
        row_start = int(transformation.indptr[row])
        row_stop = int(transformation.indptr[row + 1])
        column_start = int(transformation.indptr[column])
        column_stop = int(transformation.indptr[column + 1])
        reduced_rows = transformation.indices[row_start:row_stop]
        reduced_columns = transformation.indices[column_start:column_stop]
        row_coefficients = transformation.data[row_start:row_stop]
        column_coefficients = transformation.data[column_start:column_stop]
        count = int(reduced_rows.size * reduced_columns.size)
        keys = (
            np.repeat(reduced_rows, reduced_columns.size).astype(np.int64)
            * np.int64(n_reduced)
            + np.tile(reduced_columns, reduced_rows.size).astype(np.int64)
        )
        coefficients = np.repeat(row_coefficients, reduced_columns.size) * np.tile(
            column_coefficients, reduced_rows.size
        )
        weighted_sources.append(np.full(count, int(source), dtype=np.int32))
        weighted_keys.append(keys)
        weighted_coefficients.append(coefficients.astype(float, copy=False))

    return (
        np.concatenate(unit_sources) if unit_sources else np.empty(0, dtype=np.int32),
        np.concatenate(unit_keys) if unit_keys else np.empty(0, dtype=np.int64),
        np.concatenate(weighted_sources)
        if weighted_sources
        else np.empty(0, dtype=np.int32),
        np.concatenate(weighted_keys)
        if weighted_keys
        else np.empty(0, dtype=np.int64),
        np.concatenate(weighted_coefficients)
        if weighted_coefficients
        else np.empty(0, dtype=float),
        total_contributions,
    )


def _csr_pattern_from_keys(
    unique_keys: np.ndarray,
    reduced_dofs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if unique_keys.size == 0:
        return (
            np.zeros(reduced_dofs + 1, dtype=np.intp),
            np.empty(0, dtype=np.intp),
        )
    reduced_rows = unique_keys // np.int64(reduced_dofs)
    reduced_columns = unique_keys % np.int64(reduced_dofs)
    row_counts = np.bincount(reduced_rows.astype(np.intp), minlength=reduced_dofs)
    csr_indptr = np.empty(reduced_dofs + 1, dtype=np.intp)
    csr_indptr[0] = 0
    np.cumsum(row_counts, out=csr_indptr[1:])
    return csr_indptr, reduced_columns.astype(np.intp, copy=False)


def _finalize_reduced_plan(
    *,
    nonlinear_plan: Any,
    transformation: sparse.csr_matrix,
    force_unit_positions: np.ndarray,
    force_weighted_sources: np.ndarray,
    force_weighted_positions: np.ndarray,
    force_weighted_coefficients: np.ndarray,
    tangent_unit_positions: np.ndarray,
    tangent_weighted_sources: np.ndarray,
    tangent_weighted_positions: np.ndarray,
    tangent_weighted_coefficients: np.ndarray,
    csr_indptr: np.ndarray,
    csr_indices: np.ndarray,
    force_contributions: int,
    tangent_contributions: int,
    mapping_kind: str,
    setup_start: float,
) -> ReducedAssemblyPlan:
    force_buffer = np.zeros(transformation.shape[1], dtype=float)
    tangent_buffer = np.zeros(csr_indices.size, dtype=float)
    tangent_matrix = sparse.csr_matrix(
        (tangent_buffer, csr_indices, csr_indptr),
        shape=(transformation.shape[1], transformation.shape[1]),
        copy=False,
    )
    tangent_buffer = tangent_matrix.data
    estimated_map_bytes = int(
        force_unit_positions.nbytes
        + force_weighted_sources.nbytes
        + force_weighted_positions.nbytes
        + force_weighted_coefficients.nbytes
        + tangent_unit_positions.nbytes
        + tangent_weighted_sources.nbytes
        + tangent_weighted_positions.nbytes
        + tangent_weighted_coefficients.nbytes
        + csr_indptr.nbytes
        + csr_indices.nbytes
        + force_buffer.nbytes
        + tangent_buffer.nbytes
    )
    if estimated_map_bytes > _maximum_map_bytes():
        raise ReducedAssemblyPlanLimit(
            "direct reduced scatter map would require "
            f"{estimated_map_bytes / (1024.0 ** 2):.1f} MiB; "
            "increase FE_SOLVER_BATCH_C_MAX_MAP_MB or retain full-coordinate reduction"
        )

    plan = ReducedAssemblyPlan(
        transformation=transformation,
        total_dofs=int(transformation.shape[0]),
        reduced_dofs=int(transformation.shape[1]),
        force_unit_positions=force_unit_positions.astype(np.int32, copy=False),
        force_weighted_sources=force_weighted_sources.astype(np.int32, copy=False),
        force_weighted_positions=force_weighted_positions.astype(np.int32, copy=False),
        force_weighted_coefficients=force_weighted_coefficients.astype(float, copy=False),
        tangent_unit_positions=tangent_unit_positions.astype(np.int32, copy=False),
        tangent_weighted_sources=tangent_weighted_sources.astype(np.int32, copy=False),
        tangent_weighted_positions=tangent_weighted_positions.astype(np.int32, copy=False),
        tangent_weighted_coefficients=tangent_weighted_coefficients.astype(float, copy=False),
        csr_indptr=csr_indptr,
        csr_indices=csr_indices,
        force_buffer=force_buffer,
        tangent_buffer=tangent_buffer,
        tangent_matrix=tangent_matrix,
        setup_seconds=float(time.perf_counter() - setup_start),
        force_contributions=int(force_contributions),
        tangent_contributions=int(tangent_contributions),
        tangent_expansion_ratio=float(
            tangent_contributions / max(int(nonlinear_plan.tangent_values.size), 1)
        ),
        estimated_map_bytes=estimated_map_bytes,
        mapping_kind=mapping_kind,
        source_plan=nonlinear_plan,
    )
    plan.timings.builds = 1
    return plan


def _build_selector_reduced_plan(
    nonlinear_plan: Any,
    transformation: sparse.csr_matrix,
    setup_start: float,
) -> ReducedAssemblyPlan:
    """Fast setup for fixed-support/selector transformations."""

    reduced_dofs = int(transformation.shape[1])
    row_counts = np.diff(transformation.indptr)
    full_to_reduced = np.full(transformation.shape[0], -1, dtype=np.int32)
    active_rows = np.flatnonzero(row_counts == 1)
    if active_rows.size:
        pointers = transformation.indptr[active_rows]
        full_to_reduced[active_rows] = transformation.indices[pointers].astype(
            np.int32, copy=False
        )

    force_unit_positions = full_to_reduced[
        np.asarray(nonlinear_plan.force_dofs_flat, dtype=np.intp)
    ].copy()

    source_positions = np.asarray(nonlinear_plan.tangent_scatter, dtype=np.intp)
    source_rows = np.searchsorted(
        np.asarray(nonlinear_plan.csr_indptr, dtype=np.intp),
        source_positions,
        side="right",
    ) - 1
    source_columns = np.asarray(nonlinear_plan.csr_indices, dtype=np.intp)[
        source_positions
    ]
    reduced_rows = full_to_reduced[source_rows]
    reduced_columns = full_to_reduced[source_columns]
    valid = (reduced_rows >= 0) & (reduced_columns >= 0)

    tangent_unit_positions = np.full(
        nonlinear_plan.tangent_values.size, -1, dtype=np.int32
    )
    if np.any(valid):
        keys = (
            reduced_rows[valid].astype(np.int64) * np.int64(reduced_dofs)
            + reduced_columns[valid].astype(np.int64)
        )
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        if unique_keys.size > np.iinfo(np.int32).max:
            raise ReducedAssemblyPlanLimit(
                "reduced tangent pattern exceeds the supported 32-bit scatter index range"
            )
        tangent_unit_positions[valid] = inverse.astype(np.int32, copy=False)
        csr_indptr, csr_indices = _csr_pattern_from_keys(unique_keys, reduced_dofs)
    else:
        csr_indptr, csr_indices = _csr_pattern_from_keys(
            np.empty(0, dtype=np.int64), reduced_dofs
        )

    empty_i32 = np.empty(0, dtype=np.int32)
    empty_float = np.empty(0, dtype=float)
    return _finalize_reduced_plan(
        nonlinear_plan=nonlinear_plan,
        transformation=transformation,
        force_unit_positions=force_unit_positions,
        force_weighted_sources=empty_i32,
        force_weighted_positions=empty_i32,
        force_weighted_coefficients=empty_float,
        tangent_unit_positions=tangent_unit_positions,
        tangent_weighted_sources=empty_i32,
        tangent_weighted_positions=empty_i32,
        tangent_weighted_coefficients=empty_float,
        csr_indptr=csr_indptr,
        csr_indices=csr_indices,
        force_contributions=int(np.count_nonzero(force_unit_positions >= 0)),
        tangent_contributions=int(np.count_nonzero(valid)),
        mapping_kind="selector",
        setup_start=setup_start,
    )


def _build_weighted_reduced_plan(
    nonlinear_plan: Any,
    transformation: sparse.csr_matrix,
    setup_start: float,
    preflight_tangent_contributions: int,
) -> ReducedAssemblyPlan:
    (
        force_unit_sources,
        force_unit_keys,
        force_weighted_sources,
        force_weighted_keys,
        force_weighted_coefficients,
    ) = _append_force_mapping(
        transformation,
        np.asarray(nonlinear_plan.force_dofs_flat, dtype=np.intp),
    )
    (
        tangent_unit_sources,
        tangent_unit_keys,
        tangent_weighted_sources,
        tangent_weighted_keys,
        tangent_weighted_coefficients,
        tangent_contributions,
    ) = _append_tangent_mapping(nonlinear_plan, transformation)
    if tangent_contributions != preflight_tangent_contributions:
        raise RuntimeError(
            "reduced tangent contribution count changed during plan construction"
        )

    all_tangent_keys = np.concatenate((tangent_unit_keys, tangent_weighted_keys))
    if all_tangent_keys.size:
        unique_keys, inverse = np.unique(all_tangent_keys, return_inverse=True)
        if unique_keys.size > np.iinfo(np.int32).max:
            raise ReducedAssemblyPlanLimit(
                "reduced tangent pattern exceeds the supported 32-bit scatter index range"
            )
        split = tangent_unit_keys.size
        tangent_unit_reduced_positions = inverse[:split]
        tangent_weighted_positions = inverse[split:]
        csr_indptr, csr_indices = _csr_pattern_from_keys(
            unique_keys, int(transformation.shape[1])
        )
    else:
        tangent_unit_reduced_positions = np.empty(0, dtype=np.intp)
        tangent_weighted_positions = np.empty(0, dtype=np.intp)
        csr_indptr, csr_indices = _csr_pattern_from_keys(
            np.empty(0, dtype=np.int64), int(transformation.shape[1])
        )

    tangent_unit_positions = np.full(
        nonlinear_plan.tangent_values.size, -1, dtype=np.int32
    )
    if tangent_unit_sources.size:
        tangent_unit_positions[tangent_unit_sources] = (
            tangent_unit_reduced_positions.astype(np.int32, copy=False)
        )

    force_unit_positions = np.full(
        nonlinear_plan.force_values.size, -1, dtype=np.int32
    )
    if force_unit_sources.size:
        force_unit_positions[force_unit_sources] = force_unit_keys.astype(
            np.int32, copy=False
        )

    return _finalize_reduced_plan(
        nonlinear_plan=nonlinear_plan,
        transformation=transformation,
        force_unit_positions=force_unit_positions,
        force_weighted_sources=force_weighted_sources,
        force_weighted_positions=force_weighted_keys,
        force_weighted_coefficients=force_weighted_coefficients,
        tangent_unit_positions=tangent_unit_positions,
        tangent_weighted_sources=tangent_weighted_sources,
        tangent_weighted_positions=tangent_weighted_positions,
        tangent_weighted_coefficients=tangent_weighted_coefficients,
        csr_indptr=csr_indptr,
        csr_indices=csr_indices,
        force_contributions=int(force_unit_sources.size + force_weighted_sources.size),
        tangent_contributions=int(tangent_contributions),
        mapping_kind="weighted_mpc",
        setup_start=setup_start,
    )


def build_reduced_assembly_plan(
    nonlinear_plan: Any,
    transformation: sparse.spmatrix,
) -> ReducedAssemblyPlan:
    """Build a retained direct scatter map for one constraint transformation."""

    start = time.perf_counter()
    T = sparse.csr_matrix(transformation, dtype=float, copy=True)
    T.sum_duplicates()
    T.eliminate_zeros()
    T.sort_indices()
    if T.shape[0] != nonlinear_plan.total_dofs:
        raise ValueError(
            f"Constraint transformation has {T.shape[0]} rows; "
            f"expected {nonlinear_plan.total_dofs}"
        )
    if T.shape[1] > np.iinfo(np.int32).max:
        raise ReducedAssemblyPlanLimit(
            "reduced system exceeds the supported 32-bit scatter index range"
        )

    estimated_preflight_bytes, preflight_tangent_contributions = (
        _preflight_reduced_map_bytes(nonlinear_plan, T)
    )
    if estimated_preflight_bytes > _maximum_map_bytes():
        raise ReducedAssemblyPlanLimit(
            "direct reduced scatter map is estimated to require "
            f"{estimated_preflight_bytes / (1024.0 ** 2):.1f} MiB; "
            "increase FE_SOLVER_BATCH_C_MAX_MAP_MB or retain full-coordinate reduction"
        )

    row_counts = np.diff(T.indptr)
    selector = bool(
        (row_counts.size == 0 or int(np.max(row_counts)) <= 1)
        and np.all(T.data == 1.0)
    )
    if selector:
        return _build_selector_reduced_plan(nonlinear_plan, T, start)
    return _build_weighted_reduced_plan(
        nonlinear_plan,
        T,
        start,
        preflight_tangent_contributions,
    )


def _evaluate_local_responses(
    nonlinear_plan: Any,
    displacements: np.ndarray,
    committed_states: Mapping[int, Any],
    tangent: bool,
) -> Tuple[Dict[int, Any], float]:
    """Fill the persistent element-local buffers without global sparse assembly."""

    start = time.perf_counter()
    nonlinear_plan.force_values.fill(0.0)
    if tangent:
        nonlinear_plan.tangent_values.fill(0.0)
    trial_states: Dict[int, Any] = {}

    for batch in nonlinear_plan.shell_batches:
        force_batch, tangent_batch, batch_states, kernel_seconds = batch.evaluate(
            displacements,
            committed_states,
            tangent,
        )
        nonlinear_plan.timings.shell_kernel_seconds += kernel_seconds
        nonlinear_plan.force_values[batch.force_positions.reshape(-1)] = np.asarray(
            force_batch, dtype=float
        ).reshape(-1)
        if tangent and tangent_batch is not None:
            nonlinear_plan.tangent_values[
                batch.tangent_positions.reshape(-1)
            ] = np.asarray(tangent_batch, dtype=float).reshape(-1)
        trial_states.update(batch_states)

    non_shell_start = time.perf_counter()
    model = nonlinear_plan.model
    mesh = model.mesh
    for record in nonlinear_plan.non_shell_elements:
        material = model.get_material(record.element.material_name)
        element_displacement = np.asarray(displacements, dtype=float)[
            record.dof_mapping
        ]
        force_element, tangent_element, trial_state = (
            record.element.compute_nonlinear_response(
                mesh,
                material,
                element_displacement,
                committed_states.get(record.element_id),
                nonlinear_plan.num_layers,
                tangent,
            )
        )
        nonlinear_plan.force_values[record.force_positions] = np.asarray(
            force_element, dtype=float
        ).reshape(-1)
        if tangent and tangent_element is not None:
            nonlinear_plan.tangent_values[record.tangent_positions] = np.asarray(
                tangent_element, dtype=float
            ).reshape(-1)
        if trial_state is not None:
            trial_states[record.element_id] = trial_state
    nonlinear_plan.timings.non_shell_seconds += time.perf_counter() - non_shell_start
    return trial_states, time.perf_counter() - start


def assemble_reduced_system(
    nonlinear_plan: Any,
    reduced_plan: ReducedAssemblyPlan,
    displacements: np.ndarray,
    committed_states: Mapping[int, Any],
    tangent: bool = True,
) -> Tuple[np.ndarray, Optional[sparse.csr_matrix], Dict[int, Any]]:
    """Assemble reduced internal force and tangent directly from local buffers."""

    with nonlinear_plan._lock:
        start_total = time.perf_counter()
        nonlinear_plan.timings.calls += 1
        reduced_plan.timings.assemblies += 1
        if tangent:
            nonlinear_plan.timings.tangent_calls += 1
        else:
            nonlinear_plan.timings.residual_only_calls += 1
            reduced_plan.timings.residual_only_assemblies += 1

        trial_states, local_seconds = _evaluate_local_responses(
            nonlinear_plan,
            displacements,
            committed_states,
            tangent,
        )
        reduced_plan.timings.local_response_seconds += local_seconds

        scatter_start = time.perf_counter()
        _scatter_reduced_values(
            nonlinear_plan.force_values,
            reduced_plan.force_unit_positions,
            reduced_plan.force_weighted_sources,
            reduced_plan.force_weighted_positions,
            reduced_plan.force_weighted_coefficients,
            reduced_plan.force_buffer,
        )
        force_seconds = time.perf_counter() - scatter_start
        reduced_plan.timings.reduced_force_scatter_seconds += force_seconds
        nonlinear_plan.timings.force_scatter_seconds += force_seconds

        tangent_matrix: Optional[sparse.csr_matrix]
        if tangent:
            scatter_start = time.perf_counter()
            _scatter_reduced_values(
                nonlinear_plan.tangent_values,
                reduced_plan.tangent_unit_positions,
                reduced_plan.tangent_weighted_sources,
                reduced_plan.tangent_weighted_positions,
                reduced_plan.tangent_weighted_coefficients,
                reduced_plan.tangent_buffer,
            )
            tangent_seconds = time.perf_counter() - scatter_start
            reduced_plan.timings.reduced_tangent_scatter_seconds += tangent_seconds
            nonlinear_plan.timings.tangent_scatter_seconds += tangent_seconds
            tangent_matrix = reduced_plan.tangent_matrix
        else:
            tangent_matrix = None

        elapsed = time.perf_counter() - start_total
        reduced_plan.timings.total_seconds += elapsed
        nonlinear_plan.timings.total_seconds += elapsed
        return reduced_plan.force_buffer, tangent_matrix, trial_states
