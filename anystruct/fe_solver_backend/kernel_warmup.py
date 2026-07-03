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


def _warm_nonlinear_impact_kernel() -> Dict[str, Any]:
    """Touch the material-nonlinear impact path without running a real case."""

    import numpy as np

    from .boundary import BoundaryCondition
    from .contact import (
        NonlinearTransientConfig,
        RigidSphereImpact,
        SphereContactConfig,
        solve_transient_sphere_impact,
    )
    from .dynamics import TransientConfig
    from .elements import ShellElement
    from .fe_core import FEModel
    from .fracture import PlasticImpactDamageConfig
    from .material_curves import DNVC208MaterialCurve

    model = FEModel("kernel_warmup_nonlinear_impact")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    model.materials["soft"].hardening_curve = DNVC208MaterialCurve(
        sigma_prop=800.0,
        sigma_yield=1000.0,
        sigma_yield_2=1200.0,
        eps_p_y1=1.0e-5,
        eps_p_y2=1.0e-3,
        K=2000.0,
        n=0.1,
    )
    for node_id, xyz in {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (1.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
    }.items():
        model.add_node(node_id, *xyz)
    model.add_element(1, ShellElement(1, [1, 2, 3, 4], "soft", thickness=0.05))
    model.add_boundary_condition(
        BoundaryCondition(
            "warmup_restrain",
            [1, 2, 3, 4],
            {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )

    start = time.perf_counter()
    result = solve_transient_sphere_impact(
        model,
        TransientConfig(dt=0.01, t_end=0.01),
        RigidSphereImpact(
            "warmup",
            radius=0.2,
            mass=5.0,
            start_point=(0.5, 0.5, 0.25),
            travel_direction=(0.0, 0.0, -1.0),
            speed=1.0,
        ),
        SphereContactConfig(penalty_stiffness=500.0, max_contact_iterations=8),
        nonlinear_config=NonlinearTransientConfig(enabled=True, max_iterations=6, max_cutbacks=1),
        plastic_damage_config=PlasticImpactDamageConfig(threshold=0.01, max_deleted_fraction=1.0),
    )
    elapsed = time.perf_counter() - start
    return {
        "status": str(result.status),
        "seconds": float(elapsed),
        "method": str(result.diagnostics.get("method", "")),
        "stiffness_assembly_skipped": bool((result.diagnostics.get("stiffness") or {}).get("assembly_skipped", False)),
        "strain_summary_available": "strain_summary" in result.diagnostics,
        "num_saved_steps": int(len(result.times)),
    }


def warm_fe_solver_kernels(
    shell_orders: Iterable[str] = ("S4", "Q8", "Q8R"),
    *,
    include_nonlinear_impact: bool = False,
) -> Dict[str, Any]:
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
    nonlinear_impact = None
    if include_nonlinear_impact:
        nonlinear_impact = _warm_nonlinear_impact_kernel()
    return {
        "status": "completed",
        "jit": jit,
        "total_seconds": float(time.perf_counter() - total_start),
        "shell_orders": results,
        "nonlinear_impact": nonlinear_impact,
    }


__all__: Tuple[str, ...] = ("warm_fe_solver_kernels",)
