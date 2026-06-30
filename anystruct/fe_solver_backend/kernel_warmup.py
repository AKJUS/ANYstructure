"""Optional runtime warmup for FE solver compiled kernels."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Tuple

from scipy import sparse

from .jit_compiler import JIT_DISABLED_REASON, JIT_ENABLED, jit_diagnostics
from .matrix_assembly import assemble_stiffness_matrix
from .mesh_gen import generate_simple_panel_mesh


def _normalize_shell_order(shell_order: str) -> str:
    order = str(shell_order).strip().upper()
    aliases = {"S8": "Q8", "S8R": "Q8R"}
    return aliases.get(order, order)


def _warmup_model(shell_order: str):
    order = _normalize_shell_order(shell_order)
    if order == "S4":
        model = generate_simple_panel_mesh(1.0, 0.75, 0.01, num_divisions_x=1, num_divisions_y=1)
    elif order in {"Q8", "Q8R"}:
        model = generate_simple_panel_mesh(
            1.0,
            0.75,
            0.01,
            num_divisions_x=1,
            num_divisions_y=1,
            use_8node_elements=True,
        )
        if order == "Q8R":
            for element in model.mesh.elements.values():
                if getattr(element, "_is_8node", False):
                    element.reduced_integration = True
    else:
        raise ValueError(f"Unsupported shell order for warmup: {shell_order!r}")
    return model


def _matrix_difference_norm(left: sparse.spmatrix, right: sparse.spmatrix) -> float:
    denominator = max(float(sparse.linalg.norm(left)), 1.0)
    return float(sparse.linalg.norm(left - right) / denominator)


def warm_fe_solver_kernels(shell_orders: Iterable[str] = ("S4", "Q8", "Q8R")) -> Dict[str, Any]:
    """Warm representative FE kernels and return timing/correctness diagnostics.

    The helper is intentionally optional and side-effect free apart from Numba's
    normal in-process/disk cache behavior.  It is suitable for runtime
    applications that want to absorb first-call compilation before an analysis.
    """

    results: Dict[str, Any] = {}
    total_start = time.perf_counter()
    jit = jit_diagnostics()
    for requested_order in shell_orders:
        order = _normalize_shell_order(requested_order)
        model = _warmup_model(order)
        start = time.perf_counter()
        K_first, first_info = assemble_stiffness_matrix(model)
        first_seconds = time.perf_counter() - start
        start = time.perf_counter()
        K_second, second_info = assemble_stiffness_matrix(model)
        second_seconds = time.perf_counter() - start
        results[order] = {
            "status": "completed",
            "shell_order": order,
            "element_count": int(model.mesh.num_elements),
            "jit_enabled": bool(JIT_ENABLED),
            "jit_disabled_reason": JIT_DISABLED_REASON,
            "jit_backend": jit.get("backend"),
            "parallel_threads": jit.get("num_threads"),
            "cold_assembly_seconds": float(first_seconds),
            "warm_assembly_seconds": float(second_seconds),
            "warm_speedup": float(first_seconds / second_seconds) if second_seconds > 0.0 else 0.0,
            "matrix_difference_norm": _matrix_difference_norm(K_first, K_second),
            "first_assembly": first_info.get("diagnostics", {}),
            "second_assembly": second_info.get("diagnostics", {}),
        }
    return {
        "status": "completed",
        "jit": jit,
        "total_seconds": float(time.perf_counter() - total_start),
        "shell_orders": results,
    }


__all__: Tuple[str, ...] = ("warm_fe_solver_kernels",)
