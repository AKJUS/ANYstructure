"""Deterministic local FE baseline cases and comparison helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from .assembly import solve_linear
from .boundary import BoundaryCondition, FixedSupport, LoadCase
from .buckling import solve_eigenvalue_buckling
from .cylinder_benchmarks import CylinderBenchmarkConfig, run_cylindrical_shell_benchmark
from .dynamics import TransientConfig, solve_transient_newmark
from .elements import BeamElement
from .fe_core import FEModel
from .mesh_gen import generate_beam_mesh, generate_simple_panel_mesh
from .nonlinear_static import solve_static_nonlinear


DEFAULT_BASELINE_PATH = Path("tests/fixtures/fe_baselines/baseline.json")


def _git_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _topology(model: FEModel) -> Dict[str, int]:
    return {
        "nodes": int(model.mesh.num_nodes),
        "elements": int(model.mesh.num_elements),
        "dofs": int(model.mesh.dof_manager.total_dofs),
        "boundary_conditions": int(len(model.boundary_conditions)),
    }


def _float(value: Any) -> float:
    return float(np.asarray(value, dtype=float))


def _max_translation(model: FEModel, displacements: np.ndarray) -> float:
    peak = 0.0
    for node in model.mesh.nodes.values():
        peak = max(peak, float(np.linalg.norm(displacements[node.dofs[:3]])))
    return peak


def _beam_static_case() -> Dict[str, Any]:
    model = generate_beam_mesh(1.0, num_divisions=2, cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6})
    load_case = LoadCase("beam_tip")
    load_case.add_nodal_load(3, [0.0, 0.0, -1000.0, 0.0, 0.0, 0.0])
    u, info = solve_linear(model, load_case)
    tip = model.mesh.get_node(3)
    return {
        "topology": _topology(model),
        "results": {
            "tip_uz": _float(u[tip.dofs[2]]),
            "max_translation": _max_translation(model, u),
            "solution_norm": _float(np.linalg.norm(u)),
        },
        "residuals": {
            "status": str((info.get("convergence_info") or {}).get("status", "")),
            "relative_residual": _float((info.get("convergence_info") or {}).get("relative_residual", 0.0)),
        },
    }


def _plate_shell_case() -> Dict[str, Any]:
    model = generate_simple_panel_mesh(1.0, 0.6, 0.01, num_divisions_x=2, num_divisions_y=1)
    load_case = LoadCase("plate_pressure")
    for element_id in model.mesh.elements:
        load_case.add_pressure_load(element_id, 1000.0)
    u, info = solve_linear(model, load_case)
    return {
        "topology": _topology(model),
        "results": {
            "max_translation": _max_translation(model, u),
            "solution_norm": _float(np.linalg.norm(u)),
        },
        "residuals": {
            "status": str((info.get("convergence_info") or {}).get("status", "")),
            "relative_residual": _float((info.get("convergence_info") or {}).get("relative_residual", 0.0)),
        },
    }


def _cylinder_case() -> Dict[str, Any]:
    config = CylinderBenchmarkConfig(radius=1.0, height=1.0, thickness=0.02, pressure=1000.0, num_circumferential=8, num_height=2)
    result = run_cylindrical_shell_benchmark(config)
    return {
        "topology": {
            "nodes": int(result.node_count),
            "elements": int(result.element_count),
            "shell_elements": int(result.shell_element_count),
        },
        "results": {
            "max_displacement_norm": float(result.max_displacement_norm),
            "max_radial_displacement": float(result.max_radial_displacement),
            "von_mises_mean": float(result.all_von_mises.mean),
            "nominal_von_mises": float(result.nominal.von_mises_stress),
        },
        "residuals": {
            "status": str(result.solver_status),
            "relative_rigid_body_load_imbalance": float(result.relative_rigid_body_load_imbalance),
        },
    }


def _beam_column_model(num_elements: int = 4) -> FEModel:
    length = 4.0
    model = FEModel("baseline_beam_column")
    model.add_material("steel", 210.0e9, 0.3, density=7850.0)
    model.current_material = "steel"
    for i in range(num_elements + 1):
        model.add_node(i + 1, length * i / num_elements, 0.0, 0.0)
    section = {"area": 0.02, "Iy": 3.0e-6, "Iz": 5.0e-6, "J": 2.0e-6}
    for i in range(num_elements):
        model.add_element(i + 1, BeamElement(i + 1, [i + 1, i + 2], "steel", section))
    all_nodes = list(range(1, num_elements + 2))
    end_nodes = [1, num_elements + 1]
    model.add_boundary_condition(BoundaryCondition("suppress_unrelated_dofs", all_nodes, {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0}))
    model.add_boundary_condition(BoundaryCondition("pinned_lateral_ends", end_nodes, {"uy": 0.0}))
    return model


def _buckling_case() -> Dict[str, Any]:
    model = _beam_column_model(num_elements=4)
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    result = solve_eigenvalue_buckling(model, states, num_modes=2)
    return {
        "topology": _topology(model),
        "results": {
            "critical_load_factor": float(result.critical_load_factor or 0.0),
            "num_modes_returned": int(result.num_modes_returned),
        },
        "residuals": {"status": str(result.solver_status)},
    }


def _nonlinear_case() -> Dict[str, Any]:
    model = generate_beam_mesh(1.0, num_divisions=1, cross_section={"area": 0.01, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6})
    load_case = LoadCase("small_axial")
    load_case.add_nodal_load(2, [100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = solve_static_nonlinear(model, load_case, num_steps=2, max_iterations=8, num_layers=3)
    return {
        "topology": _topology(model),
        "results": {
            "load_factor": float(result.load_factor),
            "peak_load_factor": float(result.peak_load_factor),
            "max_translation": _max_translation(model, result.displacements),
            "steps": int(len(result.steps)),
        },
        "residuals": {"status": str(result.status), "failure_reason": str(result.failure_reason)},
    }


def _transient_case() -> Dict[str, Any]:
    model = FEModel("baseline_axial_sdof")
    model.add_material("steel", elastic_modulus=100.0, poisson_ratio=0.3, density=2.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    section = {"area": 1.0, "Iy": 1.0e-6, "Iz": 1.0e-6, "J": 1.0e-6}
    model.add_element(1, BeamElement(1, [1, 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    model.add_boundary_condition(BoundaryCondition("slider", [2], {"uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}))
    load_case = LoadCase("step")
    load_case.add_nodal_load(2, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = solve_transient_newmark(model, TransientConfig(dt=0.001, t_end=0.01), base_load_case=load_case)
    ux = model.mesh.get_node(2).dofs[0]
    return {
        "topology": _topology(model),
        "results": {
            "final_ux": _float(result.displacements[-1, ux]),
            "peak_displacement": float(result.peak_displacement),
            "energy_drift": float(result.diagnostics.get("max_relative_energy_drift", 0.0)),
        },
        "residuals": {"status": str(result.status)},
    }


CASE_BUILDERS: Mapping[str, Callable[[], Dict[str, Any]]] = {
    "beam_static": _beam_static_case,
    "plate_shell_static": _plate_shell_case,
    "cylinder_static": _cylinder_case,
    "beam_column_buckling": _buckling_case,
    "nonlinear_static": _nonlinear_case,
    "transient_newmark": _transient_case,
}


DEFAULT_TOLERANCES: Dict[str, Dict[str, Dict[str, float]]] = {
    name: {
        "results.*": {"rtol": 1.0e-8, "atol": 1.0e-10},
        "residuals.relative_residual": {"rtol": 0.0, "atol": 1.0e-8},
        "residuals.relative_rigid_body_load_imbalance": {"rtol": 0.0, "atol": 1.0e-8},
    }
    for name in CASE_BUILDERS
}


def generate_baseline_document(*, include_timing: bool = True, selected_cases: Optional[Mapping[str, Callable[[], Dict[str, Any]]]] = None) -> Dict[str, Any]:
    """Run deterministic baseline cases and return a serializable document."""
    cases: Dict[str, Any] = {}
    for name, builder in (selected_cases or CASE_BUILDERS).items():
        start = time.perf_counter()
        case = builder()
        elapsed = time.perf_counter() - start
        case.setdefault("timing", {})["solve_seconds"] = float(elapsed) if include_timing else None
        cases[name] = case
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
        "tolerances": DEFAULT_TOLERANCES,
        "known_limitations": [
            "Baselines are local deterministic smoke anchors, not external validation references.",
            "Timing fields are informational and excluded from numeric baseline comparison.",
        ],
    }


def write_baseline(path: Path | str = DEFAULT_BASELINE_PATH, *, include_timing: bool = True) -> Dict[str, Any]:
    """Generate and write a baseline JSON document."""
    document = generate_baseline_document(include_timing=include_timing)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def load_baseline(path: Path | str = DEFAULT_BASELINE_PATH) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tolerance_for(tolerances: Mapping[str, Mapping[str, Mapping[str, float]]], case_name: str, section: str, metric: str) -> Dict[str, float]:
    case_tolerances = tolerances.get(case_name, {})
    return dict(case_tolerances.get(f"{section}.{metric}", case_tolerances.get(f"{section}.*", {"rtol": 1.0e-8, "atol": 1.0e-10})))


def compare_baseline_documents(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare baseline values while excluding metadata and timings."""
    failures = []
    warnings = []
    ref_cases = reference.get("cases", {})
    cand_cases = candidate.get("cases", {})
    tolerances = reference.get("tolerances", DEFAULT_TOLERANCES)

    for case_name, ref_case in ref_cases.items():
        cand_case = cand_cases.get(case_name)
        if cand_case is None:
            failures.append({"case": case_name, "reason": "missing_candidate_case"})
            continue
        if ref_case.get("topology") != cand_case.get("topology"):
            failures.append({"case": case_name, "reason": "topology_mismatch", "reference": ref_case.get("topology"), "candidate": cand_case.get("topology")})
        for section in ("results", "residuals"):
            ref_section = ref_case.get(section, {})
            cand_section = cand_case.get(section, {})
            for metric, ref_value in ref_section.items():
                cand_value = cand_section.get(metric)
                if isinstance(ref_value, (int, float)) and isinstance(cand_value, (int, float)):
                    tol = _tolerance_for(tolerances, case_name, section, metric)
                    rtol = float(tol.get("rtol", 0.0))
                    atol = float(tol.get("atol", 0.0))
                    if not np.isclose(float(cand_value), float(ref_value), rtol=rtol, atol=atol):
                        failures.append(
                            {
                                "case": case_name,
                                "metric": f"{section}.{metric}",
                                "reference": float(ref_value),
                                "candidate": float(cand_value),
                                "rtol": rtol,
                                "atol": atol,
                            }
                        )
                elif cand_value != ref_value:
                    failures.append({"case": case_name, "metric": f"{section}.{metric}", "reference": ref_value, "candidate": cand_value})

    for case_name in sorted(set(cand_cases) - set(ref_cases)):
        warnings.append({"case": case_name, "reason": "extra_candidate_case"})

    return {"status": "passed" if not failures else "failed", "failures": failures, "warnings": warnings}
