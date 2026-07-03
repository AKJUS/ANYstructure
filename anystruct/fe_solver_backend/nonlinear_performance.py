"""Performance layer for nonlinear shell/beam analyses.

The production nonlinear formulations remain in :mod:`fe_solver.nonlinear_static`
and the element modules.  This module removes repeated Python and sparse-matrix
bookkeeping from every Newton iteration by building a persistent assembly plan
for each model/layer-count pair.

The plan caches:

* shell groups and reference-geometry arrays,
* element DOF mappings,
* reusable shell input work arrays,
* global force scatter positions,
* the unique global CSR tangent pattern, and
* local-entry-to-CSR-data scatter positions.

It also installs two low-risk solver improvements:

* revision-counter sparsity caching instead of rebuilding a JSON/SHA topology
  signature on every cache lookup;
* displacement-control block elimination using two right-hand sides on the
  ordinary structural tangent instead of an augmented sparse matrix.

The installation is idempotent.  The FE package activates it during normal
package import on the performance branch.  Set ``FE_SOLVER_DISABLE_FAST_NL=1``
before importing :mod:`fe_solver` to retain the legacy assembly path for
comparison and debugging.
"""

from __future__ import annotations

import copy
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .jit_compiler import njit

if TYPE_CHECKING:
    from .fe_core import FEModel, FEMesh


@njit(cache=True)
def _scatter_sum(values: np.ndarray, positions: np.ndarray, output_size: int) -> np.ndarray:
    """Sum flat local values into a precomputed unique global index space."""
    result = np.zeros(output_size, dtype=np.float64)
    for index in range(values.size):
        result[positions[index]] += values[index]
    return result


def _revision_tuple(mesh: "FEMesh") -> Tuple[int, ...]:
    revisions = getattr(mesh, "revision_signature", lambda: {})()
    return tuple(
        int(revisions.get(name, 0))
        for name in ("topology", "geometry", "material", "mpc")
    )


def _sparsity_revision_tuple(mesh: "FEMesh") -> Tuple[int, int]:
    revisions = getattr(mesh, "revision_signature", lambda: {})()
    return int(revisions.get("topology", 0)), int(revisions.get("mpc", 0))


@dataclass
class NonlinearAssemblyTimings:
    calls: int = 0
    tangent_calls: int = 0
    residual_only_calls: int = 0
    shell_kernel_seconds: float = 0.0
    non_shell_seconds: float = 0.0
    state_pack_seconds: float = 0.0
    force_scatter_seconds: float = 0.0
    tangent_scatter_seconds: float = 0.0
    total_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": int(self.calls),
            "tangent_calls": int(self.tangent_calls),
            "residual_only_calls": int(self.residual_only_calls),
            "shell_kernel_seconds": float(self.shell_kernel_seconds),
            "non_shell_seconds": float(self.non_shell_seconds),
            "state_pack_seconds": float(self.state_pack_seconds),
            "force_scatter_seconds": float(self.force_scatter_seconds),
            "tangent_scatter_seconds": float(self.tangent_scatter_seconds),
            "total_seconds": float(self.total_seconds),
        }


@dataclass
class _ElementScatter:
    element_id: int
    element: Any
    dof_mapping: np.ndarray
    force_positions: np.ndarray
    tangent_positions: np.ndarray


@dataclass
class _ShellBatchPlan:
    key: Tuple[Any, ...]
    element_ids: np.ndarray
    elements: Tuple[Any, ...]
    dof_mappings: np.ndarray
    force_positions: np.ndarray
    tangent_positions: np.ndarray
    T0: np.ndarray
    B_m: np.ndarray
    B_b: np.ndarray
    B_d: np.ndarray
    Gw: np.ndarray
    detw: np.ndarray
    B_s: np.ndarray
    detw_shear: np.ndarray
    material: Any
    thickness: float
    drilling_stabilization: float
    n_gp: int
    n_dof: int
    num_layers: int
    u_work: np.ndarray
    plastic_work: np.ndarray
    alpha_work: np.ndarray
    elastic_states: Tuple[Dict[str, np.ndarray], ...]

    @property
    def has_plasticity(self) -> bool:
        return getattr(self.material, "hardening_curve", None) is not None

    @classmethod
    def build(
        cls,
        model: "FEModel",
        key: Tuple[Any, ...],
        items: Sequence[Tuple[int, Any, _ElementScatter]],
        num_layers: int,
    ) -> "_ShellBatchPlan":
        first = items[0][1]
        first_cache = first._nonlinear_geometry(model.mesh)
        n_elem = len(items)
        n_dof = int(first.total_dofs)
        n_gp = int(first_cache["detw_all"].shape[0])
        n_shear = int(first_cache["detw_shear_all"].shape[0])

        dof_mappings = np.empty((n_elem, n_dof), dtype=np.intp)
        force_positions = np.empty((n_elem, n_dof), dtype=np.intp)
        tangent_positions = np.empty((n_elem, n_dof * n_dof), dtype=np.intp)
        T0 = np.empty((n_elem, n_dof, n_dof), dtype=float)
        B_m = np.empty((n_elem, n_gp, 3, n_dof), dtype=float)
        B_b = np.empty((n_elem, n_gp, 3, n_dof), dtype=float)
        B_d = np.empty((n_elem, n_gp, 1, n_dof), dtype=float)
        Gw = np.empty((n_elem, n_gp, 2, n_dof), dtype=float)
        detw = np.empty((n_elem, n_gp), dtype=float)
        B_s = np.empty((n_elem, n_shear, 2, n_dof), dtype=float)
        detw_shear = np.empty((n_elem, n_shear), dtype=float)
        element_ids = np.empty(n_elem, dtype=np.int64)
        elements: List[Any] = []
        elastic_states: List[Dict[str, np.ndarray]] = []

        for batch_index, (element_id, element, scatter) in enumerate(items):
            cache = element._nonlinear_geometry(model.mesh)
            element_ids[batch_index] = int(element_id)
            elements.append(element)
            dof_mappings[batch_index] = scatter.dof_mapping
            force_positions[batch_index] = scatter.force_positions
            tangent_positions[batch_index] = scatter.tangent_positions
            T0[batch_index] = cache["T0"]
            B_m[batch_index] = cache["B_m_all"]
            B_b[batch_index] = cache["B_b_all"]
            B_d[batch_index] = cache["B_d_all"]
            Gw[batch_index] = cache["Gw_all"]
            detw[batch_index] = cache["detw_all"]
            B_s[batch_index] = cache["B_s_all"]
            detw_shear[batch_index] = cache["detw_shear_all"]
            # Elastic shell states never evolve.  Stable per-element arrays avoid
            # retaining views into reusable batch work buffers.
            elastic_states.append(
                {
                    "plastic_strain": np.zeros((n_gp * num_layers, 3), dtype=float),
                    "alpha": np.zeros(n_gp * num_layers, dtype=float),
                    "layer_strain": np.zeros((n_gp * num_layers, 3), dtype=float),
                }
            )

        material = model.get_material(first.material_name)
        return cls(
            key=key,
            element_ids=element_ids,
            elements=tuple(elements),
            dof_mappings=dof_mappings,
            force_positions=force_positions,
            tangent_positions=tangent_positions,
            T0=T0,
            B_m=B_m,
            B_b=B_b,
            B_d=B_d,
            Gw=Gw,
            detw=detw,
            B_s=B_s,
            detw_shear=detw_shear,
            material=material,
            thickness=float(first.thickness),
            drilling_stabilization=float(first.drilling_stabilization),
            n_gp=n_gp,
            n_dof=n_dof,
            num_layers=int(num_layers),
            u_work=np.zeros((n_elem, n_dof), dtype=float),
            plastic_work=np.zeros((n_elem, n_gp * num_layers, 3), dtype=float),
            alpha_work=np.zeros((n_elem, n_gp * num_layers), dtype=float),
            elastic_states=tuple(elastic_states),
        )

    def evaluate(
        self,
        displacements: np.ndarray,
        committed_states: Mapping[int, Any],
        tangent: bool,
        deleted_element_ids: Sequence[int] = (),
        residual_stiffness_fraction: float = 1.0,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[int, Any], float]:
        from .vectorized_nonlinear import batch_shell_nonlinear_response

        start = time.perf_counter()
        deleted = {int(element_id) for element_id in deleted_element_ids}
        residual_fraction = float(residual_stiffness_fraction)
        self.u_work[:] = np.asarray(displacements, dtype=float)[self.dof_mappings]

        if self.has_plasticity:
            self.plastic_work.fill(0.0)
            self.alpha_work.fill(0.0)
            for index, (element_id, element) in enumerate(zip(self.element_ids, self.elements)):
                state = committed_states.get(int(element_id))
                if state is None:
                    state = element.init_nonlinear_state(self.num_layers)
                self.plastic_work[index] = np.asarray(state["plastic_strain"], dtype=float)
                self.alpha_work[index] = np.asarray(state["alpha"], dtype=float)

        F_batch, K_batch, ep_new, alpha_new, layer_strain = batch_shell_nonlinear_response(
            self.u_work,
            self.T0,
            self.B_m,
            self.B_b,
            self.B_d,
            self.Gw,
            self.detw,
            self.B_s,
            self.detw_shear,
            float(self.material.elastic_modulus),
            float(self.material.poisson_ratio),
            float(self.material.shear_modulus),
            self.thickness,
            self.drilling_stabilization,
            bool(tangent),
            getattr(self.material, "hardening_curve", None),
            self.plastic_work,
            self.alpha_work,
            self.num_layers,
        )

        states: Dict[int, Any] = {}
        if self.has_plasticity:
            points_per_element = self.n_gp * self.num_layers
            for index, element_id in enumerate(self.element_ids):
                start_layer = index * points_per_element
                stop_layer = start_layer + points_per_element
                if int(element_id) in deleted:
                    existing = committed_states.get(int(element_id))
                    if isinstance(existing, dict):
                        states[int(element_id)] = existing
                    else:
                        states[int(element_id)] = {
                            "plastic_strain": ep_new[index],
                            "alpha": alpha_new[index],
                            "layer_strain": layer_strain[start_layer:stop_layer].copy(),
                        }
                    continue
                states[int(element_id)] = {
                    "plastic_strain": ep_new[index],
                    "alpha": alpha_new[index],
                    "layer_strain": layer_strain[start_layer:stop_layer].copy(),
                }
        else:
            for index, element_id in enumerate(self.element_ids):
                existing = committed_states.get(int(element_id))
                states[int(element_id)] = existing if isinstance(existing, dict) else self.elastic_states[index]

        if deleted:
            for index, element_id in enumerate(self.element_ids):
                if int(element_id) not in deleted:
                    continue
                F_batch[index] *= residual_fraction
                if tangent and K_batch is not None:
                    K_batch[index] *= residual_fraction

        return F_batch, K_batch if tangent else None, states, time.perf_counter() - start


@dataclass
class NonlinearAssemblyPlan:
    """Persistent nonlinear global assembly data for one FE model."""

    model_ref: "weakref.ReferenceType[FEModel]"
    num_layers: int
    revision: Tuple[int, ...]
    shell_batches: Tuple[_ShellBatchPlan, ...]
    non_shell_elements: Tuple[_ElementScatter, ...]
    force_dofs_flat: np.ndarray
    tangent_scatter: np.ndarray
    csr_indptr: np.ndarray
    csr_indices: np.ndarray
    force_values: np.ndarray
    tangent_values: np.ndarray
    setup_seconds: float
    timings: NonlinearAssemblyTimings = field(default_factory=NonlinearAssemblyTimings)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def model(self) -> "FEModel":
        model = self.model_ref()
        if model is None:
            raise RuntimeError("The FE model associated with this assembly plan no longer exists")
        return model

    @property
    def total_dofs(self) -> int:
        return int(self.csr_indptr.size - 1)

    @property
    def nnz(self) -> int:
        return int(self.csr_indices.size)

    @classmethod
    def build(cls, model: "FEModel", num_layers: int) -> "NonlinearAssemblyPlan":
        from .elements import ShellElement

        start = time.perf_counter()
        mesh = model.mesh
        total_dofs = int(mesh.dof_manager.total_dofs)
        element_records: List[Tuple[int, Any, np.ndarray, int, int, int, int]] = []
        all_rows: List[np.ndarray] = []
        all_cols: List[np.ndarray] = []
        force_offset = 0
        tangent_offset = 0

        for element_id, element in mesh.elements.items():
            mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp).reshape(-1)
            if mapping.size == 0:
                continue
            n_local = int(mapping.size)
            rows = np.repeat(mapping, n_local)
            cols = np.tile(mapping, n_local)
            all_rows.append(rows)
            all_cols.append(cols)
            element_records.append(
                (
                    int(element_id),
                    element,
                    mapping,
                    force_offset,
                    force_offset + n_local,
                    tangent_offset,
                    tangent_offset + n_local * n_local,
                )
            )
            force_offset += n_local
            tangent_offset += n_local * n_local

        rows_concat = np.concatenate(all_rows) if all_rows else np.empty(0, dtype=np.intp)
        cols_concat = np.concatenate(all_cols) if all_cols else np.empty(0, dtype=np.intp)
        if rows_concat.size:
            pair_keys = rows_concat.astype(np.int64) * np.int64(total_dofs) + cols_concat.astype(np.int64)
            unique_keys, tangent_scatter = np.unique(pair_keys, return_inverse=True)
            unique_rows = (unique_keys // np.int64(total_dofs)).astype(np.intp)
            unique_cols = (unique_keys % np.int64(total_dofs)).astype(np.intp)
            row_counts = np.bincount(unique_rows, minlength=total_dofs)
            csr_indptr = np.empty(total_dofs + 1, dtype=np.intp)
            csr_indptr[0] = 0
            np.cumsum(row_counts, out=csr_indptr[1:])
            csr_indices = unique_cols
        else:
            tangent_scatter = np.empty(0, dtype=np.intp)
            csr_indptr = np.zeros(total_dofs + 1, dtype=np.intp)
            csr_indices = np.empty(0, dtype=np.intp)

        scatter_records: Dict[int, _ElementScatter] = {}
        force_dofs: List[np.ndarray] = []
        for element_id, element, mapping, f0, f1, k0, k1 in element_records:
            force_positions = np.arange(f0, f1, dtype=np.intp)
            tangent_positions = np.arange(k0, k1, dtype=np.intp)
            scatter = _ElementScatter(
                element_id=element_id,
                element=element,
                dof_mapping=mapping,
                force_positions=force_positions,
                tangent_positions=tangent_positions,
            )
            scatter_records[element_id] = scatter
            force_dofs.append(mapping)

        shell_groups: Dict[Tuple[Any, ...], List[Tuple[int, Any, _ElementScatter]]] = {}
        non_shell: List[_ElementScatter] = []
        for element_id, element, *_rest in element_records:
            scatter = scatter_records[element_id]
            if isinstance(element, ShellElement):
                key = (
                    int(element.num_nodes),
                    float(element.thickness),
                    float(element.drilling_stabilization),
                    str(element.material_name),
                )
                shell_groups.setdefault(key, []).append((element_id, element, scatter))
            else:
                non_shell.append(scatter)

        shell_batches = tuple(
            _ShellBatchPlan.build(model, key, items, int(num_layers))
            for key, items in shell_groups.items()
        )

        return cls(
            model_ref=weakref.ref(model),
            num_layers=int(num_layers),
            revision=_revision_tuple(mesh),
            shell_batches=shell_batches,
            non_shell_elements=tuple(non_shell),
            force_dofs_flat=np.concatenate(force_dofs) if force_dofs else np.empty(0, dtype=np.intp),
            tangent_scatter=np.asarray(tangent_scatter, dtype=np.intp),
            csr_indptr=csr_indptr,
            csr_indices=csr_indices,
            force_values=np.zeros(force_offset, dtype=float),
            tangent_values=np.zeros(tangent_offset, dtype=float),
            setup_seconds=float(time.perf_counter() - start),
        )

    def is_valid(self, model: "FEModel", num_layers: int) -> bool:
        return (
            self.model_ref() is model
            and int(num_layers) == self.num_layers
            and _revision_tuple(model.mesh) == self.revision
        )

    def assemble(
        self,
        displacements: np.ndarray,
        committed_states: Mapping[int, Any],
        tangent: bool = True,
        deleted_element_ids: Sequence[int] = (),
        residual_stiffness_fraction: float = 1.0,
    ) -> Tuple[np.ndarray, Optional[sparse.csr_matrix], Dict[int, Any]]:
        """Assemble internal force and tangent using persistent batch/scatter data."""
        with self._lock:
            start_total = time.perf_counter()
            self.timings.calls += 1
            if tangent:
                self.timings.tangent_calls += 1
            else:
                self.timings.residual_only_calls += 1

            self.force_values.fill(0.0)
            if tangent:
                self.tangent_values.fill(0.0)
            trial_states: Dict[int, Any] = {}
            deleted = {int(element_id) for element_id in deleted_element_ids}
            residual_fraction = float(residual_stiffness_fraction)

            state_start = time.perf_counter()
            for batch in self.shell_batches:
                F_batch, K_batch, batch_states, kernel_seconds = batch.evaluate(
                    displacements,
                    committed_states,
                    tangent,
                    deleted_element_ids=tuple(deleted),
                    residual_stiffness_fraction=residual_fraction,
                )
                self.timings.shell_kernel_seconds += kernel_seconds
                self.force_values[batch.force_positions.reshape(-1)] = np.asarray(F_batch, dtype=float).reshape(-1)
                if tangent and K_batch is not None:
                    self.tangent_values[batch.tangent_positions.reshape(-1)] = np.asarray(K_batch, dtype=float).reshape(-1)
                trial_states.update(batch_states)
            self.timings.state_pack_seconds += time.perf_counter() - state_start

            non_shell_start = time.perf_counter()
            model = self.model
            mesh = model.mesh
            for record in self.non_shell_elements:
                material = model.get_material(record.element.material_name)
                u_element = np.asarray(displacements, dtype=float)[record.dof_mapping]
                f_element, k_element, trial_state = record.element.compute_nonlinear_response(
                    mesh,
                    material,
                    u_element,
                    committed_states.get(record.element_id),
                    self.num_layers,
                    tangent,
                )
                if record.element_id in deleted:
                    f_element = residual_fraction * np.asarray(f_element, dtype=float)
                    if tangent and k_element is not None:
                        k_element = residual_fraction * np.asarray(k_element, dtype=float)
                    if record.element_id in committed_states:
                        trial_state = committed_states[record.element_id]
                self.force_values[record.force_positions] = np.asarray(f_element, dtype=float).reshape(-1)
                if tangent and k_element is not None:
                    self.tangent_values[record.tangent_positions] = np.asarray(k_element, dtype=float).reshape(-1)
                if trial_state is not None:
                    trial_states[record.element_id] = trial_state
            self.timings.non_shell_seconds += time.perf_counter() - non_shell_start

            scatter_start = time.perf_counter()
            force = _scatter_sum(self.force_values, self.force_dofs_flat, self.total_dofs)
            self.timings.force_scatter_seconds += time.perf_counter() - scatter_start

            tangent_matrix: Optional[sparse.csr_matrix]
            if tangent:
                scatter_start = time.perf_counter()
                csr_data = _scatter_sum(self.tangent_values, self.tangent_scatter, self.nnz)
                tangent_matrix = sparse.csr_matrix(
                    (csr_data, self.csr_indices, self.csr_indptr),
                    shape=(self.total_dofs, self.total_dofs),
                    copy=False,
                )
                self.timings.tangent_scatter_seconds += time.perf_counter() - scatter_start
            else:
                tangent_matrix = None

            self.timings.total_seconds += time.perf_counter() - start_total
            return force, tangent_matrix, trial_states

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "num_layers": int(self.num_layers),
            "revision": list(self.revision),
            "shell_batch_count": len(self.shell_batches),
            "shell_element_count": int(sum(batch.element_ids.size for batch in self.shell_batches)),
            "non_shell_element_count": len(self.non_shell_elements),
            "total_dofs": self.total_dofs,
            "tangent_nnz": self.nnz,
            "local_force_entries": int(self.force_values.size),
            "local_tangent_entries": int(self.tangent_values.size),
            "setup_seconds": float(self.setup_seconds),
            "timings": self.timings.to_dict(),
        }


_PLAN_CACHE: "weakref.WeakKeyDictionary[FEModel, Dict[int, NonlinearAssemblyPlan]]" = weakref.WeakKeyDictionary()
_CACHE_LOCK = threading.RLock()
_ORIGINAL_ASSEMBLER = None
_ORIGINAL_SPARSITY_GETTER = None
_ORIGINAL_DISPLACEMENT_SOLVER = None
_INSTALLED = False


def get_nonlinear_assembly_plan(model: "FEModel", num_layers: int) -> NonlinearAssemblyPlan:
    """Return a valid cached plan or build a new one."""
    with _CACHE_LOCK:
        by_layers = _PLAN_CACHE.setdefault(model, {})
        plan = by_layers.get(int(num_layers))
        if plan is None or not plan.is_valid(model, int(num_layers)):
            plan = NonlinearAssemblyPlan.build(model, int(num_layers))
            by_layers[int(num_layers)] = plan
        return plan


def clear_nonlinear_assembly_cache(model: Optional["FEModel"] = None) -> None:
    with _CACHE_LOCK:
        if model is None:
            _PLAN_CACHE.clear()
        else:
            _PLAN_CACHE.pop(model, None)


def nonlinear_assembly_diagnostics(model: Optional["FEModel"] = None) -> Dict[str, Any]:
    with _CACHE_LOCK:
        if model is not None:
            plans = _PLAN_CACHE.get(model, {})
            return {str(layers): plan.diagnostics() for layers, plan in plans.items()}
        result: Dict[str, Any] = {}
        for cached_model, plans in list(_PLAN_CACHE.items()):
            result[str(id(cached_model))] = {
                "model_name": getattr(cached_model, "name", None),
                "plans": {str(layers): plan.diagnostics() for layers, plan in plans.items()},
            }
        return result


def _optimized_assemble_nonlinear_system(
    model: "FEModel",
    displacements: np.ndarray,
    committed_states: Dict[int, Any],
    num_layers: int,
    tangent: bool = True,
    deleted_element_ids: Optional[Sequence[int]] = None,
    residual_stiffness_fraction: float = 1.0,
    **extra,
):
    kinematics = str(extra.pop("kinematics", "von_karman"))
    if extra or kinematics != "von_karman":
        # The assembly plan encodes the von Karman element response; other
        # kinematics or per-element scale options use the original assembler.
        assembler = _ORIGINAL_ASSEMBLER
        if assembler is None:
            from . import nonlinear_static as _nonlinear_static

            assembler = _nonlinear_static._assemble_nonlinear_system
        return assembler(
            model,
            displacements,
            committed_states,
            num_layers,
            tangent=tangent,
            deleted_element_ids=tuple(deleted_element_ids or ()),
            residual_stiffness_fraction=float(residual_stiffness_fraction),
            kinematics=kinematics,
            **extra,
        )
    plan = get_nonlinear_assembly_plan(model, int(num_layers))
    return plan.assemble(
        displacements,
        committed_states,
        tangent=tangent,
        deleted_element_ids=tuple(deleted_element_ids or ()),
        residual_stiffness_fraction=float(residual_stiffness_fraction),
    )


def _revision_cached_sparsity_pattern(mesh: "FEMesh", matrix_type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Cached COO row/column pattern using mesh revisions rather than JSON/SHA."""
    if not hasattr(mesh, "_sparsity_cache"):
        mesh._sparsity_cache = {}
    signature = _sparsity_revision_tuple(mesh)
    cached = mesh._sparsity_cache.get(matrix_type)
    if cached is not None and cached.get("signature") == signature:
        return cached["rows"], cached["cols"]

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for element in mesh.elements.values():
        mapping = np.asarray(element.get_dof_mapping(mesh), dtype=np.intp).reshape(-1)
        if mapping.size == 0:
            continue
        rows.append(np.repeat(mapping, mapping.size))
        cols.append(np.tile(mapping, mapping.size))
    rows_concat = np.concatenate(rows) if rows else np.empty(0, dtype=np.intp)
    cols_concat = np.concatenate(cols) if cols else np.empty(0, dtype=np.intp)
    mesh._sparsity_cache[matrix_type] = {
        "rows": rows_concat,
        "cols": cols_concat,
        "signature": signature,
    }
    return rows_concat, cols_concat


def _solve_static_displacement_control_block(
    *,
    model,
    T,
    u0,
    F_const,
    F_prop,
    stage_vectors,
    load_program,
    displacement_control,
    committed_states,
    num_layers,
    num_steps,
    max_iterations,
    tolerance,
    info,
    start_time,
    resource_config=None,
    kinematics="von_karman",
):
    """Displacement control using block elimination on the structural tangent."""
    from . import nonlinear_static as ns
    from .cases import make_result_case
    from .jit_compiler import numba_thread_scope
    from .linalg import MatrixClass, factorize

    if str(kinematics) != "von_karman":
        # The block-elimination fast path encodes the von Karman assembly plan;
        # corotational displacement control uses the original solver.
        solver = _ORIGINAL_DISPLACEMENT_SOLVER
        if solver is None:
            solver = ns._solve_static_displacement_control
        return solver(
            model=model,
            T=T,
            u0=u0,
            F_const=F_const,
            F_prop=F_prop,
            stage_vectors=stage_vectors,
            load_program=load_program,
            displacement_control=displacement_control,
            committed_states=committed_states,
            num_layers=num_layers,
            num_steps=num_steps,
            max_iterations=max_iterations,
            tolerance=tolerance,
            info=info,
            start_time=start_time,
            resource_config=resource_config,
            kinematics=kinematics,
        )

    if load_program is not None:
        if len(load_program.stages) == 1:
            F_const_dc = F_const
            F_prop_dc = load_program.stages[0].target_factor * stage_vectors[0]
            active_stage = load_program.stages[0].name
        else:
            F_const_dc = F_const.copy()
            for stage, vector in zip(load_program.stages[:-1], stage_vectors[:-1]):
                F_const_dc += stage.target_factor * vector
            F_prop_dc = load_program.stages[-1].target_factor * stage_vectors[-1]
            active_stage = load_program.stages[-1].name
        info["displacement_control_load_split"] = {
            "constant_stages": [stage.name for stage in load_program.stages[:-1]],
            "proportional_stage": active_stage,
        }
    else:
        F_const_dc = F_const
        F_prop_dc = F_prop
        active_stage = "displacement_control"

    n_red = int(T.shape[1])
    q = np.zeros(n_red, dtype=float)
    lam = 0.0
    steps = []
    history: List[Dict[str, Any]] = []
    status = "completed"
    failure_reason: Optional[str] = None
    total_iterations = 0

    row_full = displacement_control.full_row(model)
    row_red = np.asarray(row_full @ T, dtype=float).reshape(-1)
    row_u0 = float(row_full @ u0)
    if float(np.linalg.norm(row_red)) <= 0.0:
        raise ValueError("Displacement control target is fixed or dependent and cannot be used as an unknown")

    F_const_red = np.asarray(T.T @ F_const_dc, dtype=float).reshape(-1)
    F_prop_red = np.asarray(T.T @ F_prop_dc, dtype=float).reshape(-1)
    if float(np.linalg.norm(F_prop_red)) <= 0.0:
        raise ValueError("Displacement control requires a non-zero proportional load vector")

    target_total = float(displacement_control.target_displacement)
    target_scale = max(abs(target_total), 1.0e-9)

    assembly_threads = None if resource_config is None else resource_config.assembly_threads
    with numba_thread_scope(assembly_threads):
        for step_index in range(1, num_steps + 1):
            target = target_total * step_index / num_steps
            residual_norm = float("inf")
            constraint_error = float("inf")
            states_new = committed_states

            for iteration in range(1, max_iterations + 1):
                total_iterations += 1
                u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
                F_int, K_T, trial_states = ns._assemble_nonlinear_system(
                    model, u, committed_states, num_layers
                )
                residual = F_const_red + lam * F_prop_red - np.asarray(T.T @ F_int, dtype=float).reshape(-1)
                residual_norm = float(np.linalg.norm(residual))
                current = float(row_red @ q + row_u0)
                constraint = target - current
                constraint_error = abs(constraint)
                reference = max(
                    float(np.linalg.norm(F_const_red + max(abs(lam), 1.0) * F_prop_red)),
                    1.0,
                )

                if residual_norm <= tolerance * reference and constraint_error <= tolerance * target_scale:
                    states_new = trial_states
                    break

                K_red = (T.T @ K_T @ T).tocsr()
                try:
                    handle = factorize(
                        K_red,
                        MatrixClass.SYMMETRIC_INDEFINITE,
                        signature=f"nonlinear.displacement_control.block:{step_index}:{iteration}",
                    )
                    rhs = np.column_stack((residual, F_prop_red))
                    solutions = np.asarray(handle.solve_many(rhs), dtype=float)
                    fixed_load_correction = solutions[:, 0]
                    load_direction = solutions[:, 1]
                except Exception:
                    failure_reason = "singular_structural_tangent"
                    break

                denominator = float(row_red @ load_direction)
                denominator_scale = max(float(np.linalg.norm(row_red)) * float(np.linalg.norm(load_direction)), 1.0)
                if abs(denominator) <= 1.0e-14 * denominator_scale:
                    failure_reason = "singular_displacement_constraint"
                    break
                d_lambda = (constraint - float(row_red @ fixed_load_correction)) / denominator
                d_q = fixed_load_correction + load_direction * d_lambda
                if np.any(~np.isfinite(d_q)) or not np.isfinite(d_lambda):
                    failure_reason = "nonfinite_block_solution"
                    break
                q += d_q
                lam += float(d_lambda)
            else:
                failure_reason = "maximum_iterations_reached"

            if failure_reason is not None:
                status = "stopped_at_limit" if steps else "diverged"
                break

            committed_states = states_new
            u = np.asarray(T @ q + u0, dtype=float).reshape(-1)
            current = float(row_red @ q + row_u0)
            steps.append(
                ns.NonlinearStaticStep(
                    step_index=step_index,
                    load_factor=float(lam),
                    iterations=iteration,
                    residual_norm=residual_norm,
                    displacement_norm=float(np.linalg.norm(u)),
                    max_equivalent_plastic_strain=ns._max_plastic_strain(committed_states),
                    control_value=current,
                    active_stage=active_stage,
                )
            )
            history.append(
                {
                    "step_index": step_index,
                    "load_factor": float(lam),
                    "control_value": current,
                    "target_displacement": target,
                    "residual_norm": residual_norm,
                    "constraint_error": constraint_error,
                    "iterations": iteration,
                    "active_stage": active_stage,
                    "linearization": "block_elimination",
                }
            )

    u_final = np.asarray(T @ q + u0, dtype=float).reshape(-1)
    info["failure_reason"] = failure_reason
    info["last_converged_load_factor"] = float(lam)
    info["peak_load_factor"] = max((step.load_factor for step in steps), default=float(lam))
    info["force_displacement_history"] = history
    info["strain_summary"] = ns._nonlinear_state_summary(committed_states)
    info["total_newton_iterations"] = total_iterations
    info["displacement_control_linearization"] = "block_elimination"
    info["solve_time"] = time.time() - start_time
    info["result_case"] = make_result_case(
        name="nonlinear_static_displacement_control",
        analysis_type="nonlinear_static",
        load_cases=tuple(stage.load_case for stage in load_program.stages) if load_program is not None else (),
        assembly_info={"load": {"vector_type": "load_program" if load_program is not None else "load"}, **info},
        solver_info={"convergence_info": {"status": status}},
        recovery={"displacements": True, "element_states": True, "force_displacement_history": True},
        settings={
            "control": "displacement",
            "num_steps": num_steps,
            "num_layers": num_layers,
            "linearization": "block_elimination",
        },
    ).to_dict()
    return ns.NonlinearStaticResult(steps, status, u_final, float(lam), committed_states, info)


def install_nonlinear_performance_optimizations() -> bool:
    """Install the optimized assembly and continuation helpers.

    Returns ``True`` when the performance layer is active and ``False`` when it
    is disabled through ``FE_SOLVER_DISABLE_FAST_NL``.
    """
    global _INSTALLED, _ORIGINAL_ASSEMBLER, _ORIGINAL_SPARSITY_GETTER, _ORIGINAL_DISPLACEMENT_SOLVER
    if os.environ.get("FE_SOLVER_DISABLE_FAST_NL", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if _INSTALLED:
        return True

    from . import matrix_assembly
    from . import nonlinear_static

    _ORIGINAL_ASSEMBLER = nonlinear_static._assemble_nonlinear_system
    _ORIGINAL_SPARSITY_GETTER = matrix_assembly._get_cached_sparsity_pattern
    _ORIGINAL_DISPLACEMENT_SOLVER = nonlinear_static._solve_static_displacement_control
    nonlinear_static._assemble_nonlinear_system = _optimized_assemble_nonlinear_system
    nonlinear_static._solve_static_displacement_control = _solve_static_displacement_control_block
    matrix_assembly._get_cached_sparsity_pattern = _revision_cached_sparsity_pattern
    _INSTALLED = True
    return True


def uninstall_nonlinear_performance_optimizations() -> None:
    """Restore legacy functions, primarily for A/B benchmark tests."""
    global _INSTALLED
    if not _INSTALLED:
        return
    from . import matrix_assembly
    from . import nonlinear_static

    if _ORIGINAL_ASSEMBLER is not None:
        nonlinear_static._assemble_nonlinear_system = _ORIGINAL_ASSEMBLER
    if _ORIGINAL_DISPLACEMENT_SOLVER is not None:
        nonlinear_static._solve_static_displacement_control = _ORIGINAL_DISPLACEMENT_SOLVER
    if _ORIGINAL_SPARSITY_GETTER is not None:
        matrix_assembly._get_cached_sparsity_pattern = _ORIGINAL_SPARSITY_GETTER
    _INSTALLED = False
    clear_nonlinear_assembly_cache()


def nonlinear_performance_status() -> Dict[str, Any]:
    return {
        "installed": bool(_INSTALLED),
        "disabled_by_environment": os.environ.get("FE_SOLVER_DISABLE_FAST_NL", "").strip().lower()
        in {"1", "true", "yes", "on"},
        "cached_models": len(_PLAN_CACHE),
        "diagnostics": nonlinear_assembly_diagnostics(),
    }
