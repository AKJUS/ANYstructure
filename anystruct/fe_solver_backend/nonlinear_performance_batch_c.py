"""Batch C runtime integration for direct reduced-coordinate assembly."""

from __future__ import annotations

import functools
import sys
import threading
import time
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy import sparse

from . import nonlinear_performance_batch_b as _batch_b
from . import nonlinear_reduced_assembly as _reduced
from .jit_compiler import JIT_ENABLED
from .nonlinear_reduced_assembly import (
    ReducedAssemblyPlan,
    ReducedAssemblyPlanLimit,
    _identity_transformation,
    _maximum_map_bytes,
    assemble_reduced_system,
    build_reduced_assembly_plan,
)


@dataclass(frozen=True)
class _ReducedVectorPayload:
    values: np.ndarray
    token: object


@dataclass(frozen=True)
class _ReducedMatrixPayload:
    matrix: sparse.csr_matrix
    token: object


@dataclass(frozen=True)
class _ReducedLeftProduct:
    matrix: sparse.csr_matrix
    token: object

    def __matmul__(self, other: Any):
        if (
            isinstance(other, _DirectReductionTransform)
            and other._batch_c_token is self.token
        ):
            return self.matrix
        return NotImplemented


class _DirectReductionTranspose(sparse.csc_matrix):
    """Sparse transpose that recognizes already-reduced assembly payloads."""

    def __init__(
        self,
        arg1: Any,
        *,
        token: Optional[object] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(arg1, **kwargs)
        self._batch_c_token = token

    def __matmul__(self, other: Any):
        if (
            isinstance(other, _ReducedVectorPayload)
            and other.token is self._batch_c_token
        ):
            return other.values
        if (
            isinstance(other, _ReducedMatrixPayload)
            and other.token is self._batch_c_token
        ):
            return _ReducedLeftProduct(other.matrix, self._batch_c_token)
        return sparse.csc_matrix(self).__matmul__(other)


class _DirectReductionTransform(sparse.csr_matrix):
    """CSR transformation retaining a token for direct-reduction interception."""

    def __init__(
        self,
        arg1: Any,
        *,
        token: Optional[object] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(arg1, **kwargs)
        self._batch_c_token = token

    def transpose(self, axes=None, copy: bool = False):
        base = sparse.csr_matrix(self).transpose(axes=axes, copy=copy)
        return _DirectReductionTranspose(base, token=self._batch_c_token)


@dataclass
class _SolveContext:
    requested_model: Any
    token: object = field(default_factory=object)
    bound_model: Any = None
    transformation: Optional[_DirectReductionTransform] = None
    reduced_plan: Optional[ReducedAssemblyPlan] = None
    fallback_reason: Optional[str] = None


_CONTEXT = threading.local()
_STATUS_LOCK = threading.RLock()
_STATUS: Dict[str, Any] = {
    "installed": False,
    "contexts_entered": 0,
    "reduced_plan_builds": 0,
    "reduced_assemblies": 0,
    "full_coordinate_fallbacks": 0,
    "last_fallback_reason": None,
    "last_plan": None,
}
_PATCHES: List[Tuple[ModuleType, str, Any, Any]] = []
_INSTALLED = False
_BASE_ASSEMBLER = None
_ORIGINAL_STATIC_SOLVER = None
_ORIGINAL_ARC_SOLVER = None
_ORIGINAL_CONSTRAINT_BUILDER = None
_ORIGINAL_LOCAL_EVALUATOR = _reduced._evaluate_local_responses


def _context_stack() -> List[_SolveContext]:
    stack = getattr(_CONTEXT, "stack", None)
    if stack is None:
        stack = []
        _CONTEXT.stack = stack
    return stack


def _active_context(model: Any) -> Optional[_SolveContext]:
    for context in reversed(_context_stack()):
        if context.bound_model is model:
            return context
    return None


def _run_with_context(original, *args: Any, **kwargs: Any):
    model = args[0] if args else kwargs.get("model")
    context = _SolveContext(requested_model=model)
    stack = _context_stack()
    stack.append(context)
    with _STATUS_LOCK:
        _STATUS["contexts_entered"] += 1
    try:
        return original(*args, **kwargs)
    finally:
        if stack and stack[-1] is context:
            stack.pop()
        else:
            try:
                stack.remove(context)
            except ValueError:
                pass


def _make_solver_wrapper(original):
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any):
        return _run_with_context(original, *args, **kwargs)

    wrapper._batch_c_original = original
    return wrapper


def _constraint_builder_wrapper(K, F, model):
    result = _ORIGINAL_CONSTRAINT_BUILDER(K, F, model)
    K_red, F_red, transformation, u0, independent_dofs, info = result
    stack = _context_stack()
    if not stack:
        return result
    context = stack[-1]
    if context.bound_model is None:
        context.bound_model = model
    if context.bound_model is not model or _identity_transformation(transformation):
        return result
    wrapped = _DirectReductionTransform(transformation, token=context.token)
    context.transformation = wrapped
    return K_red, F_red, wrapped, u0, independent_dofs, info


def _batch_c_evaluate_local_responses(
    nonlinear_plan: Any,
    displacements: np.ndarray,
    committed_states: Mapping[int, Any],
    tangent: bool,
):
    """Fill local buffers while retaining Batch B's elastic shell fast path."""

    start = time.perf_counter()
    nonlinear_plan.force_values.fill(0.0)
    if tangent:
        nonlinear_plan.tangent_values.fill(0.0)
    trial_states: Dict[int, Any] = {}
    displacement_array = np.asarray(displacements, dtype=float)

    for batch in nonlinear_plan.shell_batches:
        if getattr(batch, "_batch_b_elastic", False):
            kernel_start = time.perf_counter()
            _batch_b._elastic_shell_batch_into_buffers(
                displacement_array,
                batch.dof_mappings,
                batch.T0,
                batch.B_m,
                batch.B_b,
                batch.B_d,
                batch.Gw,
                batch.detw,
                batch.B_s,
                batch.detw_shear,
                batch._batch_b_membrane_matrix,
                batch._batch_b_bending_matrix,
                batch._batch_b_shear_matrix,
                float(batch._batch_b_drilling_stiffness),
                batch.force_positions,
                batch.tangent_positions,
                nonlinear_plan.force_values,
                nonlinear_plan.tangent_values,
                batch.u_work,
                bool(tangent),
            )
            nonlinear_plan.timings.shell_kernel_seconds += (
                time.perf_counter() - kernel_start
            )
            elastic_states = batch._batch_b_elastic_state_mapping
            use_cached_mapping = True
            for element_id in batch.element_ids:
                existing = committed_states.get(int(element_id))
                if (
                    isinstance(existing, dict)
                    and existing is not elastic_states[int(element_id)]
                ):
                    use_cached_mapping = False
                    break
            if use_cached_mapping:
                trial_states.update(elastic_states)
            else:
                for element_id in batch.element_ids:
                    element_key = int(element_id)
                    existing = committed_states.get(element_key)
                    trial_states[element_key] = (
                        existing
                        if isinstance(existing, dict)
                        else elastic_states[element_key]
                    )
            continue

        force_batch, tangent_batch, batch_states, kernel_seconds = batch.evaluate(
            displacement_array,
            committed_states,
            tangent,
        )
        nonlinear_plan.timings.shell_kernel_seconds += kernel_seconds
        nonlinear_plan.force_values[batch.force_positions.reshape(-1)] = np.asarray(
            force_batch,
            dtype=float,
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
        element_displacement = displacement_array[record.dof_mapping]
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
            force_element,
            dtype=float,
        ).reshape(-1)
        if tangent and tangent_element is not None:
            nonlinear_plan.tangent_values[record.tangent_positions] = np.asarray(
                tangent_element,
                dtype=float,
            ).reshape(-1)
        if trial_state is not None:
            trial_states[record.element_id] = trial_state
    nonlinear_plan.timings.non_shell_seconds += (
        time.perf_counter() - non_shell_start
    )
    return trial_states, time.perf_counter() - start


def _batch_c_assemble_nonlinear_system(
    model,
    displacements: np.ndarray,
    committed_states: Dict[int, Any],
    num_layers: int,
    tangent: bool = True,
):
    context = _active_context(model)
    if (
        context is None
        or context.transformation is None
        or context.fallback_reason is not None
    ):
        return _BASE_ASSEMBLER(
            model,
            displacements,
            committed_states,
            num_layers,
            tangent=tangent,
        )

    from .nonlinear_performance_bootstrap import get_nonlinear_assembly_plan

    nonlinear_plan = get_nonlinear_assembly_plan(model, int(num_layers))
    if (
        context.reduced_plan is None
        or context.reduced_plan.source_plan is not nonlinear_plan
    ):
        try:
            context.reduced_plan = build_reduced_assembly_plan(
                nonlinear_plan,
                context.transformation,
            )
            with _STATUS_LOCK:
                _STATUS["reduced_plan_builds"] += 1
                _STATUS["last_plan"] = context.reduced_plan.diagnostics()
        except ReducedAssemblyPlanLimit as exc:
            context.fallback_reason = str(exc)
            with _STATUS_LOCK:
                _STATUS["full_coordinate_fallbacks"] += 1
                _STATUS["last_fallback_reason"] = context.fallback_reason
            return _BASE_ASSEMBLER(
                model,
                displacements,
                committed_states,
                num_layers,
                tangent=tangent,
            )

    force_reduced, tangent_reduced, trial_states = assemble_reduced_system(
        nonlinear_plan,
        context.reduced_plan,
        displacements,
        committed_states,
        tangent=tangent,
    )
    with _STATUS_LOCK:
        _STATUS["reduced_assemblies"] += 1
        _STATUS["last_plan"] = context.reduced_plan.diagnostics()
    force_payload = _ReducedVectorPayload(force_reduced, context.token)
    tangent_payload = (
        _ReducedMatrixPayload(tangent_reduced, context.token)
        if tangent_reduced is not None
        else None
    )
    return force_payload, tangent_payload, trial_states


def _record_patch(module: ModuleType, name: str, replacement: Any) -> None:
    current = getattr(module, name, None)
    if current is replacement:
        return
    _PATCHES.append((module, name, current, replacement))
    setattr(module, name, replacement)


def _replace_function_references(original: Any, replacement: Any) -> None:
    for module in list(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("fe_solver") or module_name == __name__:
            continue
        for name, value in list(vars(module).items()):
            if value is original:
                _record_patch(module, name, replacement)


def install_batch_c_optimizations() -> bool:
    """Install direct reduced-coordinate assembly for Numba-enabled runs."""

    global _INSTALLED, _BASE_ASSEMBLER, _ORIGINAL_STATIC_SOLVER
    global _ORIGINAL_ARC_SOLVER, _ORIGINAL_CONSTRAINT_BUILDER
    if _INSTALLED:
        return True
    if not JIT_ENABLED:
        return False

    from . import arc_length
    from . import assembly
    from . import nonlinear_static

    _BASE_ASSEMBLER = nonlinear_static._assemble_nonlinear_system
    _ORIGINAL_STATIC_SOLVER = nonlinear_static.solve_static_nonlinear
    _ORIGINAL_ARC_SOLVER = arc_length.solve_static_arc_length
    _ORIGINAL_CONSTRAINT_BUILDER = assembly.build_constraint_transformation

    static_wrapper = _make_solver_wrapper(_ORIGINAL_STATIC_SOLVER)
    arc_wrapper = _make_solver_wrapper(_ORIGINAL_ARC_SOLVER)
    _replace_function_references(_ORIGINAL_STATIC_SOLVER, static_wrapper)
    _replace_function_references(_ORIGINAL_ARC_SOLVER, arc_wrapper)
    _replace_function_references(
        _ORIGINAL_CONSTRAINT_BUILDER,
        _constraint_builder_wrapper,
    )

    for module in list(sys.modules.values()):
        if not isinstance(module, ModuleType):
            continue
        module_name = getattr(module, "__name__", "")
        if not module_name.startswith("fe_solver") or module_name == __name__:
            continue
        if hasattr(module, "_assemble_nonlinear_system"):
            _record_patch(
                module,
                "_assemble_nonlinear_system",
                _batch_c_assemble_nonlinear_system,
            )

    _reduced._evaluate_local_responses = _batch_c_evaluate_local_responses
    _INSTALLED = True
    with _STATUS_LOCK:
        _STATUS["installed"] = True
    return True


def uninstall_batch_c_optimizations() -> None:
    """Restore references changed by :func:`install_batch_c_optimizations`."""

    global _INSTALLED
    if not _INSTALLED:
        return
    _reduced._evaluate_local_responses = _ORIGINAL_LOCAL_EVALUATOR
    for module, name, original, replacement in reversed(_PATCHES):
        if getattr(module, name, None) is replacement:
            setattr(module, name, original)
    _PATCHES.clear()
    _context_stack().clear()
    _INSTALLED = False
    with _STATUS_LOCK:
        _STATUS["installed"] = False


def reset_batch_c_counters() -> None:
    with _STATUS_LOCK:
        installed = bool(_STATUS["installed"])
        _STATUS.update(
            {
                "installed": installed,
                "contexts_entered": 0,
                "reduced_plan_builds": 0,
                "reduced_assemblies": 0,
                "full_coordinate_fallbacks": 0,
                "last_fallback_reason": None,
                "last_plan": None,
            }
        )


def batch_c_status() -> Dict[str, Any]:
    with _STATUS_LOCK:
        result = dict(_STATUS)
    result["eligible"] = bool(JIT_ENABLED)
    result["map_limit_mb"] = float(_maximum_map_bytes() / (1024.0**2))
    result["active_context_depth"] = len(_context_stack())
    result["batch_b_local_kernel_retained"] = bool(
        _reduced._evaluate_local_responses is _batch_c_evaluate_local_responses
    )
    return result
