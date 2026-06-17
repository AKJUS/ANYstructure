"""Local FE solver infrastructure benchmark helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy import sparse

from .assembly import solve_linear, solve_linear_many
from .boundary import BoundaryCondition, FixedSupport, LoadCase
from .buckling import solve_eigenvalue_buckling
from .dynamics import TransientConfig, solve_transient_newmark
from .elements import BeamElement
from .fe_core import FEModel
from .matrix_assembly import assemble_load_vector, assemble_mass_matrix, assemble_stiffness_matrix
from .mesh_gen import generate_beam_mesh, generate_simple_panel_mesh
from .recovery import RecoveryConfig, ResourceConfig, recover_element_stresses_with_report
from .linalg import FactorizationCache, MatrixClass, factorize_cached


DEFAULT_BENCHMARK_PATH = Path("reports/benchmarks/fe_infrastructure_benchmarks.json")


def _git_sha() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _measure(func: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    tracemalloc.start()
    start = time.perf_counter()
    try:
        payload = func()
    finally:
        elapsed = time.perf_counter() - start
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    payload.setdefault("timing", {})
    payload["timing"]["wall_seconds"] = float(elapsed)
    payload["memory"] = {"current_bytes": int(current), "peak_bytes": int(peak)}
    return payload


def _topology(model: FEModel) -> Dict[str, int]:
    return {
        "nodes": int(model.mesh.num_nodes),
        "elements": int(model.mesh.num_elements),
        "dofs": int(model.mesh.dof_manager.total_dofs),
    }


def _static_beam_case() -> Dict[str, Any]:
    model = generate_beam_mesh(2.0, num_divisions=8, cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6})
    load_case = LoadCase("tip_load")
    load_case.add_nodal_load(9, [0.0, 0.0, -1000.0, 0.0, 0.0, 0.0])
    K, stiffness_info = assemble_stiffness_matrix(model)
    M, mass_info = assemble_mass_matrix(model)
    F, load_info = assemble_load_vector(model, load_case)
    u, solver_info = solve_linear(model, load_case)
    backend = (solver_info.get("convergence_info") or {}).get("backend", {})
    return {
        "topology": _topology(model),
        "matrix_nnz": {"K": int(K.nnz), "M": int(M.nnz)},
        "load_norm": float(np.linalg.norm(F)),
        "timing": {
            "stiffness_assembly_seconds": float(stiffness_info.get("assembly_time", 0.0)),
            "mass_assembly_seconds": float(mass_info.get("assembly_time", 0.0)),
            "load_assembly_seconds": float(load_info.get("assembly_time", 0.0)),
            "solve_seconds": float(solver_info.get("solve_time", 0.0)),
            "factorization_seconds": float(backend.get("factorization_time", 0.0)) if isinstance(backend, dict) else 0.0,
            "backend_solve_seconds": float(backend.get("solve_time", 0.0)) if isinstance(backend, dict) else 0.0,
        },
        "results": {"max_abs_displacement": float(np.max(np.abs(u)))},
        "status": str((solver_info.get("convergence_info") or {}).get("status", "unknown")),
    }


def _multi_rhs_case() -> Dict[str, Any]:
    model = generate_beam_mesh(2.0, num_divisions=8, cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6})
    tip = 9
    load_x = LoadCase("tip_x")
    load_x.add_nodal_load(tip, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_z = LoadCase("tip_z")
    load_z.add_nodal_load(tip, [0.0, 0.0, -100.0, 0.0, 0.0, 0.0])
    load_m = LoadCase("tip_m")
    load_m.add_nodal_load(tip, [0.0, 0.0, 0.0, 0.0, 50.0, 0.0])
    U, info = solve_linear_many(model, [load_x, load_z, load_m])
    backend = info.get("backend", {})
    return {
        "topology": _topology(model),
        "num_rhs": 3,
        "timing": {
            "solve_many_seconds": float(info.get("solve_time", 0.0)),
            "factorization_seconds": float(backend.get("factorization_time", 0.0)) if isinstance(backend, dict) else 0.0,
            "backend_solve_seconds": float(backend.get("solve_time", 0.0)) if isinstance(backend, dict) else 0.0,
        },
        "results": {"solution_matrix_norm": float(np.linalg.norm(U))},
        "backend": backend,
        "status": str(info.get("status", "unknown")),
    }


def _shell_assembly_case() -> Dict[str, Any]:
    model = generate_simple_panel_mesh(2.0, 1.0, 0.01, num_divisions_x=8, num_divisions_y=4)
    load_case = LoadCase("pressure")
    for element_id in model.mesh.elements:
        load_case.add_pressure_load(element_id, 1000.0)
    K, stiffness_info = assemble_stiffness_matrix(model)
    M, mass_info = assemble_mass_matrix(model)
    _F, load_info = assemble_load_vector(model, load_case)
    return {
        "topology": _topology(model),
        "matrix_nnz": {"K": int(K.nnz), "M": int(M.nnz)},
        "timing": {
            "stiffness_assembly_seconds": float(stiffness_info.get("assembly_time", 0.0)),
            "mass_assembly_seconds": float(mass_info.get("assembly_time", 0.0)),
            "load_assembly_seconds": float(load_info.get("assembly_time", 0.0)),
        },
        "diagnostics": {"stiffness_symmetry": stiffness_info.get("diagnostics", {}).get("assembled_symmetry_error")},
        "status": "completed",
    }


def _buckling_case() -> Dict[str, Any]:
    model = FEModel("benchmark_column")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for i in range(9):
        model.add_node(i + 1, 4.0 * i / 8, 0.0, 0.0)
    for i in range(8):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    all_nodes = list(range(1, 10))
    model.add_boundary_condition(BoundaryCondition("suppress", all_nodes, {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pins", [1, 9], {"uy": 0.0}))
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    result = solve_eigenvalue_buckling(model, states, num_modes=2)
    return {
        "topology": _topology(model),
        "timing": {
            "stiffness_assembly_seconds": float(result.assembly_info.get("stiffness", {}).get("assembly_time", 0.0)),
            "geometric_assembly_seconds": float(result.assembly_info.get("geometric_stiffness", {}).get("assembly_time", 0.0)),
        },
        "results": {"critical_load_factor": float(result.critical_load_factor or 0.0), "num_modes": int(result.num_modes_returned)},
        "status": result.solver_status,
    }


def _transient_case() -> Dict[str, Any]:
    model = FEModel("benchmark_sdof")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_element(1, BeamElement(1, [1, 2], "steel", {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    load_case = LoadCase("step")
    load_case.add_nodal_load(2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = solve_transient_newmark(model, TransientConfig(dt=0.001, t_end=0.05), base_load_case=load_case)
    return {
        "topology": _topology(model),
        "timing": {
            "factorization_count": int(result.diagnostics.get("factorization_count", 0)),
            "solve_count": int(result.diagnostics.get("solve_count", 0)),
        },
        "results": {"peak_displacement": float(result.peak_displacement), "energy_drift": float(result.diagnostics.get("max_relative_energy_drift", 0.0))},
        "status": result.status,
    }


def _selective_recovery_case() -> Dict[str, Any]:
    model = generate_simple_panel_mesh(3.0, 2.0, 0.01, num_divisions_x=6, num_divisions_y=4)
    displacement = np.zeros(model.mesh.dof_manager.total_dofs, dtype=float)
    recovery = RecoveryConfig(components=["von_mises"])
    serial, serial_report = recover_element_stresses_with_report(
        model,
        displacement,
        recovery,
        resource_config=ResourceConfig(recovery_threads=1),
    )
    threaded, threaded_report = recover_element_stresses_with_report(
        model,
        displacement,
        recovery,
        resource_config=ResourceConfig(recovery_threads=2),
    )
    results_match = sorted(serial) == sorted(threaded) and all(
        np.allclose(serial[element_id]["von_mises"], threaded[element_id]["von_mises"])
        for element_id in serial
    )
    return {
        "topology": _topology(model),
        "timing": {
            "serial_recovery_seconds": float(serial_report.elapsed_seconds),
            "threaded_recovery_seconds": float(threaded_report.elapsed_seconds),
            "observed_speedup": (
                float(serial_report.elapsed_seconds / threaded_report.elapsed_seconds)
                if threaded_report.elapsed_seconds > 0.0
                else 0.0
            ),
        },
        "resources": {
            "serial": serial_report.to_dict(),
            "threaded": threaded_report.to_dict(),
        },
        "results": {"num_stress_results": len(serial), "results_match": bool(results_match)},
        "status": "completed" if results_match else "failed",
    }


def _factorization_reuse_case() -> Dict[str, Any]:
    matrix = sparse.diags([1.0, 4.0, 1.0], offsets=[-1, 0, 1], shape=(200, 200), format="csr")
    rhs = np.ones(200, dtype=float)
    cache = FactorizationCache(name="benchmark_factorization_reuse", max_entries=2)
    first = factorize_cached(matrix, MatrixClass.SPD, cache=cache)
    first.solve(rhs)
    second = factorize_cached(matrix.copy(), MatrixClass.SPD, cache=cache)
    second.solve(rhs)
    changed = factorize_cached(matrix + sparse.eye(200, format="csr") * 0.01, MatrixClass.SPD, cache=cache)
    changed.solve(rhs)
    return {
        "topology": {"dofs": 200, "nnz": int(matrix.nnz)},
        "timing": {
            "first_factorization_seconds": float(first.factorization_time),
            "reused_solve_seconds_total": float(second.solve_time),
            "changed_factorization_seconds": float(changed.factorization_time),
        },
        "cache": cache.diagnostics(),
        "results": {"same_handle_reused": bool(first is second), "changed_matrix_new_handle": bool(changed is not first)},
        "status": "completed" if first is second and changed is not first else "failed",
    }


BENCHMARK_CASES: Dict[str, Callable[[], Dict[str, Any]]] = {
    "static_beam": _static_beam_case,
    "multi_rhs_static": _multi_rhs_case,
    "shell_assembly": _shell_assembly_case,
    "beam_column_buckling": _buckling_case,
    "transient_newmark": _transient_case,
    "selective_recovery": _selective_recovery_case,
    "factorization_reuse": _factorization_reuse_case,
}


def run_infrastructure_benchmarks() -> Dict[str, Any]:
    """Run local benchmark smoke cases and return a serializable report."""
    cases = {name: _measure(builder) for name, builder in BENCHMARK_CASES.items()}
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_sha(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "cases": cases,
        "known_limitations": [
            "Benchmarks are local smoke measurements; compare trends on the same machine and Python environment.",
            "tracemalloc reports Python allocation peaks, not full process resident memory.",
            "Threaded recovery speedups are informational; deterministic correctness is the gate.",
        ],
    }


def write_benchmark_report(path: Path | str = DEFAULT_BENCHMARK_PATH) -> Dict[str, Any]:
    report = run_infrastructure_benchmarks()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
