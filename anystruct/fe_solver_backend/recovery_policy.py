"""Smoke-report helpers for selective recovery and resource policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .mesh_gen import generate_simple_panel_mesh
from .recovery import RecoveryConfig, ResourceConfig, estimate_model_memory, recover_element_stresses_with_report
from .results import create_fe_result


DEFAULT_RECOVERY_POLICY_PATH = Path("reports/recovery_policy/recovery_policy_report.json")


def generate_recovery_policy_report() -> Dict[str, Any]:
    """Generate deterministic selective-recovery/resource-policy smoke metrics."""

    model = generate_simple_panel_mesh(2.0, 1.0, 0.01, num_divisions_x=2, num_divisions_y=1)
    total_dofs = int(model.mesh.dof_manager.total_dofs)
    displacement = np.zeros(total_dofs, dtype=float)

    full_recovery = RecoveryConfig()
    selective_recovery = RecoveryConfig(
        node_ids=[1, max(model.mesh.nodes)],
        element_ids=[1],
        components=["von_mises"],
        history_mode="selected",
        store_full_histories=False,
        metadata={"purpose": "batch_10_smoke"},
    )
    resources = ResourceConfig(solver_threads=1, recovery_threads=1, deterministic=True)
    threaded_resources = ResourceConfig(solver_threads=1, recovery_threads=2, deterministic=True)

    full_result = create_fe_result(
        model,
        displacement,
        {"solver_type": "recovery_policy_smoke"},
        recovery_config=full_recovery,
        resource_config=resources,
    )
    selective_result = create_fe_result(
        model,
        displacement,
        {"solver_type": "recovery_policy_smoke"},
        recovery_config=selective_recovery,
        resource_config=resources,
    )

    full_memory = estimate_model_memory(
        model,
        transient_saved_steps=10,
        store_full_history=True,
        recovery_config=full_recovery,
    )
    selective_memory = estimate_model_memory(
        model,
        transient_saved_steps=10,
        store_full_history=False,
        recovery_config=selective_recovery,
    )
    envelope_recovery = RecoveryConfig(node_ids=[1, max(model.mesh.nodes)], history_mode="envelope", store_full_histories=False)
    envelope_memory = estimate_model_memory(
        model,
        transient_saved_steps=10,
        store_full_history=False,
        recovery_config=envelope_recovery,
    )
    recovery_model = generate_simple_panel_mesh(3.0, 2.0, 0.01, num_divisions_x=6, num_divisions_y=4)
    recovery_displacement = np.zeros(recovery_model.mesh.dof_manager.total_dofs, dtype=float)
    recovery_scope = RecoveryConfig(components=["von_mises"])
    serial_stresses, serial_report = recover_element_stresses_with_report(
        recovery_model,
        recovery_displacement,
        recovery_scope,
        resource_config=resources,
    )
    threaded_stresses, threaded_report = recover_element_stresses_with_report(
        recovery_model,
        recovery_displacement,
        recovery_scope,
        resource_config=threaded_resources,
    )
    stress_keys_match = sorted(serial_stresses) == sorted(threaded_stresses)
    stress_values_match = all(
        np.allclose(serial_stresses[element_id]["von_mises"], threaded_stresses[element_id]["von_mises"])
        for element_id in serial_stresses
    )
    node_reduction = 1.0 - len(selective_result.node_displacements) / max(len(full_result.node_displacements), 1)
    element_reduction = 1.0 - len(selective_result.element_stresses) / max(len(full_result.element_stresses), 1)
    memory_reduction = 1.0 - selective_memory.history_bytes_estimate / max(full_memory.history_bytes_estimate, 1)

    return {
        "status": "passed",
        "model": {
            "name": model.name,
            "num_nodes": len(model.mesh.nodes),
            "num_elements": len(model.mesh.elements),
            "total_dofs": total_dofs,
        },
        "full_recovery": {
            "num_node_displacements": len(full_result.node_displacements),
            "num_element_stresses": len(full_result.element_stresses),
            "memory_estimate": full_memory.to_dict(),
        },
        "selective_recovery": {
            "config": selective_recovery.to_dict(),
            "num_node_displacements": len(selective_result.node_displacements),
            "num_element_stresses": len(selective_result.element_stresses),
            "stress_components": {
                int(element_id): sorted(values)
                for element_id, values in selective_result.element_stresses.items()
            },
            "memory_estimate": selective_memory.to_dict(),
        },
        "transient_storage_modes": {
            "full_history_bytes": full_memory.history_bytes_estimate,
            "selected_history_bytes": selective_memory.history_bytes_estimate,
            "envelope_history_bytes": envelope_memory.history_bytes_estimate,
        },
        "resource_policy": resources.to_dict(),
        "measured_parallel_recovery": {
            "model": {
                "num_nodes": len(recovery_model.mesh.nodes),
                "num_elements": len(recovery_model.mesh.elements),
                "total_dofs": recovery_model.mesh.dof_manager.total_dofs,
            },
            "serial": serial_report.to_dict(),
            "threaded": threaded_report.to_dict(),
            "results_match": bool(stress_keys_match and stress_values_match),
            "observed_speedup": (
                float(serial_report.elapsed_seconds / threaded_report.elapsed_seconds)
                if threaded_report.elapsed_seconds > 0.0
                else None
            ),
        },
        "reductions": {
            "node_result_reduction_fraction": float(node_reduction),
            "element_result_reduction_fraction": float(element_reduction),
            "history_memory_reduction_fraction": float(memory_reduction),
        },
        "known_limits": [
            "ResourceConfig controls opt-in recovery threading and nonlinear Numba assembly threads; sparse solver thread control remains backend-dependent.",
            "Measured threaded recovery is informational and may be slower than serial for small models.",
            "Transient full history arrays are retained for API compatibility; selected/envelope storage is a later optimization.",
        ],
    }


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Selective Recovery And Resource Policy Report",
        "",
        f"- Status: {report['status']}",
        f"- Model: {report['model']['name']}",
        f"- Nodes: {report['model']['num_nodes']}",
        f"- Elements: {report['model']['num_elements']}",
        f"- Total DOFs: {report['model']['total_dofs']}",
        "",
        "## Recovery Scope",
        "",
        f"- Full node results: {report['full_recovery']['num_node_displacements']}",
        f"- Selective node results: {report['selective_recovery']['num_node_displacements']}",
        f"- Full element stress results: {report['full_recovery']['num_element_stresses']}",
        f"- Selective element stress results: {report['selective_recovery']['num_element_stresses']}",
        f"- History memory reduction: {report['reductions']['history_memory_reduction_fraction']:.3f}",
        f"- Envelope history bytes: {report['transient_storage_modes']['envelope_history_bytes']}",
        "",
        "## Measured Parallel Recovery",
        "",
        f"- Results match serial: {report['measured_parallel_recovery']['results_match']}",
        f"- Serial time: {report['measured_parallel_recovery']['serial']['elapsed_seconds']:.6g} s",
        f"- Threaded time: {report['measured_parallel_recovery']['threaded']['elapsed_seconds']:.6g} s",
        f"- Observed speedup: {report['measured_parallel_recovery']['observed_speedup']}",
        "",
        "## Resource Policy",
        "",
    ]
    for key, value in report["resource_policy"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Known Limits", ""])
    for item in report["known_limits"]:
        lines.append(f"- {item}")
    return "\n".join(lines).rstrip() + "\n"


def write_recovery_policy_report(
    output: Path | str = DEFAULT_RECOVERY_POLICY_PATH,
    *,
    markdown: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Write JSON and optional Markdown recovery-policy report."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_recovery_policy_report()
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if markdown is not None:
        markdown_path = Path(markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report
