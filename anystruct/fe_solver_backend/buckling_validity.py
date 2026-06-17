"""Sparse buckling validity smoke metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .boundary import BoundaryCondition
from .buckling import solve_eigenvalue_buckling
from .elements import BeamElement
from .fe_core import FEModel

DEFAULT_BUCKLING_VALIDITY_PATH = Path("reports/buckling_validity/buckling_validity_report.json")


def _column_model(num_elements: int = 12, symmetric: bool = False) -> Tuple[FEModel, float, Dict[str, float]]:
    length = 4.0
    model = FEModel("buckling_validity_column")
    model.add_material("steel", 210.0e9, 0.3)
    for i in range(num_elements + 1):
        model.add_node(i + 1, length * i / num_elements, 0.0, 0.0)
    inertia_y = 4.0e-6 if symmetric else 3.0e-6
    inertia_z = 4.0e-6 if symmetric else 5.0e-6
    section = {"area": 0.02, "Iy": inertia_y, "Iz": inertia_z, "J": 2.0e-6}
    for element_id in range(1, num_elements + 1):
        model.add_element(element_id, BeamElement(element_id, [element_id, element_id + 1], "steel", section))
    all_nodes = list(range(1, num_elements + 2))
    end_nodes = [1, num_elements + 1]
    if symmetric:
        model.add_boundary_condition(BoundaryCondition("suppress_axial_torsion", all_nodes, {"ux": 0.0, "rx": 0.0}))
        model.add_boundary_condition(BoundaryCondition("pinned_lateral_ends", end_nodes, {"uy": 0.0, "uz": 0.0}))
    else:
        model.add_boundary_condition(
            BoundaryCondition("suppress_unrelated_dofs", all_nodes, {"ux": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0})
        )
        model.add_boundary_condition(BoundaryCondition("pinned_lateral_ends", end_nodes, {"uy": 0.0}))
    return model, length, section


def generate_buckling_validity_report() -> Dict[str, Any]:
    model, length, section = _column_model()
    states = {element_id: {"axial_compression": 1.0} for element_id in model.mesh.elements}
    result = solve_eigenvalue_buckling(model, states, num_modes=3, dense_size_limit=2)
    expected = np.pi**2 * model.get_material("steel").elastic_modulus * section["Iz"] / length**2

    higher = solve_eigenvalue_buckling(
        model,
        states,
        num_modes=1,
        load_factor_range=(3.0 * float(result.critical_load_factor), 5.0 * float(result.critical_load_factor)),
        dense_size_limit=2,
    )

    symmetric, _, _ = _column_model(symmetric=True)
    symmetric_states = {element_id: {"axial_compression": 1.0} for element_id in symmetric.mesh.elements}
    repeated = solve_eigenvalue_buckling(symmetric, symmetric_states, num_modes=4, dense_size_limit=2, repeated_tolerance=1.0e-2)

    return {
        "status": "passed" if result.solver_status == "ok" and repeated.diagnostics["num_repeated_mode_groups"] >= 1 else "failed",
        "euler_column": {
            "critical_load_factor": result.critical_load_factor,
            "expected_euler_load": float(expected),
            "relative_error": abs(float(result.critical_load_factor) - float(expected)) / float(expected),
            "diagnostics": result.diagnostics,
        },
        "higher_mode_range": {
            "critical_load_factor": higher.critical_load_factor,
            "ratio_to_first": float(higher.critical_load_factor) / float(result.critical_load_factor),
            "diagnostics": higher.diagnostics,
        },
        "repeated_modes": {
            "load_factors": [mode.load_factor for mode in repeated.modes],
            "groups": repeated.diagnostics["repeated_mode_groups"],
            "diagnostics": repeated.diagnostics,
        },
        "known_limitations": [
            "Shifted sparse buckling now exposes an explicit cached shift-invert factorization; unshifted eigsh still uses SciPy's internal operator policy.",
            "Repeated modes are grouped by load-factor proximity; deterministic basis projection is later modal/buckling work.",
            "External plate, shell and cylinder buckling references remain Phase 14 validation work.",
        ],
    }


def write_buckling_validity_report(path: Path = DEFAULT_BUCKLING_VALIDITY_PATH) -> Dict[str, Any]:
    report = generate_buckling_validity_report()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
