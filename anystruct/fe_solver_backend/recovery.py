"""Selective result recovery and resource-policy helpers."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .fe_core import FEModel


_DOF_COMPONENTS = ("ux", "uy", "uz", "rx", "ry", "rz")
_HISTORY_MODES = {"full", "selected", "envelope"}


def _optional_int_tuple(values: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    if values is None:
        return None
    return tuple(int(value) for value in values)


def _optional_str_tuple(values: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if values is None:
        return None
    return tuple(str(value) for value in values)


@dataclass(frozen=True)
class RecoveryConfig:
    """Requested result-recovery scope.

    ``None`` for node or element ids means recover all available items, matching
    the legacy result behavior.  Components filter stress/result dictionaries by
    key; displacement arrays remain six-DOF node vectors.
    """

    node_ids: Optional[Sequence[int]] = None
    element_ids: Optional[Sequence[int]] = None
    components: Optional[Sequence[str]] = None
    include_displacements: bool = True
    include_stresses: bool = True
    include_reactions: bool = True
    history_mode: str = "full"
    store_full_histories: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.history_mode not in _HISTORY_MODES:
            raise ValueError(f"history_mode must be one of {sorted(_HISTORY_MODES)}")

    def selected_node_ids(self, model: "FEModel") -> Tuple[int, ...]:
        if self.node_ids is None:
            return tuple(int(node_id) for node_id in model.mesh.nodes)
        missing = [int(node_id) for node_id in self.node_ids if int(node_id) not in model.mesh.nodes]
        if missing:
            raise ValueError(f"Requested recovery node ids not found: {missing}")
        return _optional_int_tuple(self.node_ids) or ()

    def selected_element_ids(self, model: "FEModel") -> Tuple[int, ...]:
        if self.element_ids is None:
            return tuple(int(element_id) for element_id in model.mesh.elements)
        missing = [int(element_id) for element_id in self.element_ids if int(element_id) not in model.mesh.elements]
        if missing:
            raise ValueError(f"Requested recovery element ids not found: {missing}")
        return _optional_int_tuple(self.element_ids) or ()

    def selected_components(self) -> Optional[Tuple[str, ...]]:
        return _optional_str_tuple(self.components)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_ids": None if self.node_ids is None else list(_optional_int_tuple(self.node_ids) or ()),
            "element_ids": None if self.element_ids is None else list(_optional_int_tuple(self.element_ids) or ()),
            "components": None if self.components is None else list(_optional_str_tuple(self.components) or ()),
            "include_displacements": bool(self.include_displacements),
            "include_stresses": bool(self.include_stresses),
            "include_reactions": bool(self.include_reactions),
            "history_mode": self.history_mode,
            "store_full_histories": bool(self.store_full_histories),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResourceConfig:
    """Bounded resource policy for solver phases.

    This batch records requested limits and deterministic behavior.  It does not
    force parallel execution; later measured-parallelism work can consume the
    same contract.
    """

    solver_threads: Optional[int] = None
    assembly_threads: Optional[int] = None
    recovery_threads: Optional[int] = None
    process_workers: Optional[int] = None
    deterministic: bool = True
    memory_limit_bytes: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("solver_threads", "assembly_threads", "recovery_threads", "process_workers"):
            value = getattr(self, name)
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if self.memory_limit_bytes is not None and int(self.memory_limit_bytes) <= 0:
            raise ValueError("memory_limit_bytes must be positive when provided")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_threads": None if self.solver_threads is None else int(self.solver_threads),
            "assembly_threads": None if self.assembly_threads is None else int(self.assembly_threads),
            "recovery_threads": None if self.recovery_threads is None else int(self.recovery_threads),
            "process_workers": None if self.process_workers is None else int(self.process_workers),
            "deterministic": bool(self.deterministic),
            "memory_limit_bytes": None if self.memory_limit_bytes is None else int(self.memory_limit_bytes),
            "metadata": dict(self.metadata),
        }


class ResourcePolicyError(ValueError):
    """Raised when a requested resource policy cannot be satisfied."""

    def __init__(
        self,
        message: str,
        *,
        context: str,
        memory_estimate: Optional["MemoryEstimate"] = None,
        resource_config: Optional[ResourceConfig] = None,
    ) -> None:
        super().__init__(message)
        self.context = context
        self.memory_estimate = memory_estimate
        self.resource_config = resource_config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context": self.context,
            "message": str(self),
            "memory_estimate": None if self.memory_estimate is None else self.memory_estimate.to_dict(),
            "resource_config": None if self.resource_config is None else self.resource_config.to_dict(),
        }


@dataclass(frozen=True)
class MemoryEstimate:
    """Conservative byte estimates for common FE storage blocks."""

    total_dofs: int
    num_nodes: int
    num_elements: int
    matrix_nnz_estimate: int
    csr_bytes_estimate: int
    rhs_bytes_estimate: int
    history_bytes_estimate: int
    eigenvector_bytes_estimate: int
    nonlinear_state_bytes_estimate: int
    total_bytes_estimate: int
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_dofs": int(self.total_dofs),
            "num_nodes": int(self.num_nodes),
            "num_elements": int(self.num_elements),
            "matrix_nnz_estimate": int(self.matrix_nnz_estimate),
            "csr_bytes_estimate": int(self.csr_bytes_estimate),
            "rhs_bytes_estimate": int(self.rhs_bytes_estimate),
            "history_bytes_estimate": int(self.history_bytes_estimate),
            "eigenvector_bytes_estimate": int(self.eigenvector_bytes_estimate),
            "nonlinear_state_bytes_estimate": int(self.nonlinear_state_bytes_estimate),
            "total_bytes_estimate": int(self.total_bytes_estimate),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RecoveryExecutionReport:
    """Execution diagnostics for a recovery phase."""

    phase: str
    item_count: int
    requested_workers: int
    used_workers: int
    backend: str
    deterministic: bool
    elapsed_seconds: float
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "item_count": int(self.item_count),
            "requested_workers": int(self.requested_workers),
            "used_workers": int(self.used_workers),
            "backend": self.backend,
            "deterministic": bool(self.deterministic),
            "elapsed_seconds": float(self.elapsed_seconds),
            "reason": self.reason,
        }


def default_recovery_config(config: Optional[RecoveryConfig] = None) -> RecoveryConfig:
    """Return a full-recovery config when none is supplied."""

    return config if config is not None else RecoveryConfig()


def estimate_model_memory(
    model: "FEModel",
    *,
    num_rhs: int = 1,
    num_modes: int = 0,
    transient_saved_steps: int = 0,
    store_full_history: bool = True,
    recovery_config: Optional[RecoveryConfig] = None,
    nonlinear_state: bool = False,
) -> MemoryEstimate:
    """Estimate matrix/result storage for a model and recovery request."""

    total_dofs = int(model.mesh.dof_manager.total_dofs)
    num_nodes = int(len(model.mesh.nodes))
    num_elements = int(len(model.mesh.elements))
    nnz_pairs = set()
    notes = []
    for element in model.mesh.elements.values():
        try:
            mapping = tuple(int(dof) for dof in element.get_dof_mapping(model.mesh))
        except Exception:
            continue
        for row in mapping:
            for col in mapping:
                nnz_pairs.add((row, col))
    matrix_nnz = len(nnz_pairs)
    if matrix_nnz == 0 and total_dofs:
        matrix_nnz = total_dofs
        notes.append("matrix sparsity estimated as diagonal because no element mapping was available")

    csr_bytes = int(matrix_nnz * (8 + 4) + (total_dofs + 1) * 4)
    rhs_bytes = int(max(int(num_rhs), 0) * total_dofs * 8)

    recovery = default_recovery_config(recovery_config)
    if transient_saved_steps > 0:
        if recovery.history_mode == "envelope":
            selected_history_dofs = 6 * len(recovery.selected_node_ids(model)) if recovery.node_ids is not None and recovery.include_displacements else 0
            history_bytes = int(total_dofs * 8 * 3 + max(int(transient_saved_steps), 0) * selected_history_dofs * 8)
        elif store_full_history and recovery.store_full_histories:
            history_dofs = total_dofs
            history_bytes = int(max(int(transient_saved_steps), 0) * max(history_dofs, 0) * 8 * 3)
        else:
            history_dofs = 6 * len(recovery.selected_node_ids(model)) if recovery.include_displacements else 0
            history_bytes = int(max(int(transient_saved_steps), 0) * max(history_dofs, 0) * 8 * 3)
    else:
        history_bytes = 0

    eigenvector_bytes = int(max(int(num_modes), 0) * total_dofs * 8)
    if nonlinear_state:
        nonlinear_bytes = int(max(num_elements, 0) * 8 * 64)
    else:
        nonlinear_bytes = 0
    total = int(csr_bytes + rhs_bytes + history_bytes + eigenvector_bytes + nonlinear_bytes)
    return MemoryEstimate(
        total_dofs=total_dofs,
        num_nodes=num_nodes,
        num_elements=num_elements,
        matrix_nnz_estimate=matrix_nnz,
        csr_bytes_estimate=csr_bytes,
        rhs_bytes_estimate=rhs_bytes,
        history_bytes_estimate=history_bytes,
        eigenvector_bytes_estimate=eigenvector_bytes,
        nonlinear_state_bytes_estimate=nonlinear_bytes,
        total_bytes_estimate=total,
        notes=tuple(notes),
    )


def enforce_memory_limit(
    memory_estimate: MemoryEstimate,
    resource_config: Optional[ResourceConfig],
    *,
    context: str,
) -> None:
    """Raise when an estimate exceeds ``ResourceConfig.memory_limit_bytes``."""

    if resource_config is None or resource_config.memory_limit_bytes is None:
        return
    limit = int(resource_config.memory_limit_bytes)
    estimated = int(memory_estimate.total_bytes_estimate)
    if estimated > limit:
        raise ResourcePolicyError(
            f"{context} estimated memory {estimated} bytes exceeds configured limit {limit} bytes",
            context=context,
            memory_estimate=memory_estimate,
            resource_config=resource_config,
        )


def select_node_displacements(
    model: "FEModel",
    displacements: np.ndarray,
    recovery_config: Optional[RecoveryConfig] = None,
) -> Dict[int, np.ndarray]:
    """Extract selected nodal displacement vectors."""

    recovery = default_recovery_config(recovery_config)
    if not recovery.include_displacements:
        return {}
    vector = np.asarray(displacements, dtype=float)
    selected: Dict[int, np.ndarray] = {}
    for node_id in recovery.selected_node_ids(model):
        node = model.mesh.nodes[int(node_id)]
        selected[int(node_id)] = vector[np.asarray(node.dofs, dtype=np.intp)]
    return selected


def _filter_components(values: Mapping[str, Any], components: Optional[Tuple[str, ...]]) -> Dict[str, Any]:
    if components is None:
        return dict(values)
    wanted = set(components)
    return {key: value for key, value in values.items() if str(key) in wanted}


def _recovery_worker_count(resource_config: Optional[ResourceConfig], item_count: int) -> Tuple[int, int, str]:
    requested = 1 if resource_config is None or resource_config.recovery_threads is None else int(resource_config.recovery_threads)
    if item_count <= 1:
        return requested, 1, "serial: item count <= 1"
    if requested <= 1:
        return requested, 1, "serial: recovery_threads not requested"
    return requested, min(requested, int(item_count)), "thread_pool"


def _ordered_element_ids(model: "FEModel", selected: Sequence[int]) -> Tuple[int, ...]:
    wanted = {int(element_id) for element_id in selected}
    return tuple(int(element_id) for element_id in model.mesh.elements if int(element_id) in wanted)


def _compute_one_element_stress(
    model: "FEModel",
    displacements: np.ndarray,
    element_id: int,
    *,
    return_global: bool,
) -> Optional[Tuple[int, Dict[str, np.ndarray]]]:
    element = model.mesh.elements[int(element_id)]
    material = model.get_material(element.material_name)
    dof_mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp)
    if dof_mapping.size == 0 or int(dof_mapping.max()) >= displacements.size:
        return None
    try:
        return int(element_id), element.compute_stresses(
            model.mesh,
            displacements[dof_mapping],
            material,
            return_global=return_global,
        )
    except (IndexError, ValueError):
        return None


def recover_element_stresses_with_report(
    model: "FEModel",
    displacements: np.ndarray,
    recovery_config: Optional[RecoveryConfig] = None,
    *,
    return_global: bool = False,
    resource_config: Optional[ResourceConfig] = None,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], RecoveryExecutionReport]:
    """Recover element stresses and return bounded execution diagnostics."""

    recovery = default_recovery_config(recovery_config)
    if not recovery.include_stresses:
        report = RecoveryExecutionReport(
            phase="element_stress_recovery",
            item_count=0,
            requested_workers=1 if resource_config is None or resource_config.recovery_threads is None else int(resource_config.recovery_threads),
            used_workers=0,
            backend="disabled",
            deterministic=True if resource_config is None else bool(resource_config.deterministic),
            elapsed_seconds=0.0,
            reason="stress recovery disabled",
        )
        return {}, report

    selected_ids = _ordered_element_ids(model, recovery.selected_element_ids(model))
    displacements = np.asarray(displacements, dtype=float)
    requested, used_workers, reason = _recovery_worker_count(resource_config, len(selected_ids))
    deterministic = True if resource_config is None else bool(resource_config.deterministic)
    backend = "serial" if used_workers <= 1 else "thread_pool"
    start = time.perf_counter()
    stresses: Dict[int, Dict[str, np.ndarray]] = {}
    if used_workers <= 1:
        for element_id in selected_ids:
            item = _compute_one_element_stress(model, displacements, element_id, return_global=return_global)
            if item is not None:
                stresses[item[0]] = item[1]
    else:
        with ThreadPoolExecutor(max_workers=used_workers) as executor:
            futures = [
                executor.submit(_compute_one_element_stress, model, displacements, element_id, return_global=return_global)
                for element_id in selected_ids
            ]
            results = [future.result() for future in futures]
        for item in results:
            if item is not None:
                stresses[item[0]] = item[1]

    components = recovery.selected_components()
    if components is not None:
        stresses = {int(element_id): _filter_components(values, components) for element_id, values in stresses.items()}
    elapsed = time.perf_counter() - start
    report = RecoveryExecutionReport(
        phase="element_stress_recovery",
        item_count=len(selected_ids),
        requested_workers=requested,
        used_workers=used_workers,
        backend=backend,
        deterministic=deterministic,
        elapsed_seconds=float(elapsed),
        reason=reason,
    )
    return stresses, report


def recover_element_stresses(
    model: "FEModel",
    displacements: np.ndarray,
    recovery_config: Optional[RecoveryConfig] = None,
    *,
    return_global: bool = False,
    resource_config: Optional[ResourceConfig] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Recover selected element stresses with optional component filtering."""

    stresses, _report = recover_element_stresses_with_report(
        model,
        displacements,
        recovery_config,
        return_global=return_global,
        resource_config=resource_config,
    )
    return stresses


def filter_reactions(
    reactions: Mapping[int, np.ndarray],
    recovery_config: Optional[RecoveryConfig] = None,
    model: Optional["FEModel"] = None,
) -> Dict[int, np.ndarray]:
    """Filter reaction dictionary by requested node ids."""

    recovery = default_recovery_config(recovery_config)
    if not recovery.include_reactions:
        return {}
    if recovery.node_ids is None:
        return {int(node_id): np.asarray(values, dtype=float) for node_id, values in reactions.items()}
    if model is not None:
        selected = set(recovery.selected_node_ids(model))
    else:
        selected = set(_optional_int_tuple(recovery.node_ids) or ())
    return {int(node_id): np.asarray(values, dtype=float) for node_id, values in reactions.items() if int(node_id) in selected}


def recovery_metadata(
    recovery_config: Optional[RecoveryConfig] = None,
    resource_config: Optional[ResourceConfig] = None,
    memory_estimate: Optional[MemoryEstimate] = None,
) -> Dict[str, Any]:
    """Serialize recovery/resource policy metadata for provenance records."""

    payload: Dict[str, Any] = {"recovery": default_recovery_config(recovery_config).to_dict()}
    if resource_config is not None:
        payload["resources"] = resource_config.to_dict()
    if memory_estimate is not None:
        payload["memory_estimate"] = memory_estimate.to_dict()
    return payload
