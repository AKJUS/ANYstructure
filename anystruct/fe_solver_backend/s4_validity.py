"""S4 shell validity metrics and local report generation."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .assembly import solve_linear
from .boundary import FixedSupport, LoadCase
from .elements import ShellElement
from .fe_core import FEModel
from .mesh_gen import generate_simple_panel_mesh
from .shell_benchmarks import run_simple_supported_shell_benchmark


DEFAULT_S4_VALIDITY_PATH = Path("reports/s4_validity/s4_validity_report.json")

E_STEEL = 210.0e9
NU_STEEL = 0.3


@dataclass(frozen=True)
class S4Metric:
    """One scalar S4 validity metric."""

    name: str
    value: float
    unit: str = ""
    metadata: Mapping[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "unit": self.unit,
            "metadata": dict(self.metadata or {}),
        }


def _git_sha() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _single_s4_model(coords: Sequence[Sequence[float]], thickness: float = 0.01) -> FEModel:
    model = FEModel("single_s4")
    model.add_material("steel", E_STEEL, NU_STEEL)
    for node_id, coord in enumerate(coords, start=1):
        model.add_node(node_id, float(coord[0]), float(coord[1]), float(coord[2]))
    model.add_element(1, ShellElement(1, [1, 2, 3, 4], "steel", thickness))
    return model


def reference_s4_geometries() -> Dict[str, Tuple[Tuple[float, float, float], ...]]:
    """Representative S4 geometries used by validity metrics."""
    return {
        "square": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
        "parallelogram": ((0.0, 0.0, 0.0), (1.2, 0.0, 0.0), (1.55, 0.9, 0.0), (0.35, 0.9, 0.0)),
        "skew": ((0.0, 0.0, 0.0), (1.35, 0.08, 0.0), (1.08, 0.95, 0.0), (-0.18, 1.12, 0.0)),
        "mild_warp": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.015), (1.0, 1.0, -0.012), (0.0, 1.0, 0.008)),
    }


def free_element_mode_metric(coords: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Return free S4 eigenvalue/null-mode diagnostics."""
    model = _single_s4_model(coords)
    element = model.mesh.get_element(1)
    K = element.compute_stiffness_matrix(model.mesh, model.get_material("steel"))
    eig = np.linalg.eigvalsh(0.5 * (K + K.T))
    max_eig = max(float(np.max(np.abs(eig))), 1.0)
    threshold = 1.0e-9 * max_eig
    return {
        "zero_mode_count": int(np.sum(np.abs(eig) < threshold)),
        "threshold": float(threshold),
        "min_eigenvalue": float(np.min(eig)),
        "max_eigenvalue": float(np.max(eig)),
        "relative_negative_eigenvalue": float(min(np.min(eig), 0.0) / max_eig),
    }


def _membrane_displacement(model: FEModel, eps_x: float, eps_y: float, gamma_xy: float) -> np.ndarray:
    element = model.mesh.get_element(1)
    coords = element.get_node_coordinates(model.mesh)
    u = np.zeros(element.total_dofs, dtype=float)
    for local_node_index, coord in enumerate(coords):
        x, y, _z = coord
        base = local_node_index * 6
        u[base + 0] = eps_x * x
        u[base + 1] = eps_y * y + gamma_xy * x
    return u


def membrane_patch_metric(coords: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Constant membrane strain patch error on one S4 element."""
    model = _single_s4_model(coords)
    element = model.mesh.get_element(1)
    material = model.get_material("steel")
    eps_x, eps_y, gamma_xy = 1.2e-4, -0.4e-4, 0.7e-4
    u = _membrane_displacement(model, eps_x, eps_y, gamma_xy)
    stresses = element.compute_stresses(model.mesh, u, material)
    E = material.elastic_modulus
    nu = material.poisson_ratio
    expected = {
        "membrane_xx": E / (1.0 - nu**2) * (eps_x + nu * eps_y),
        "membrane_yy": E / (1.0 - nu**2) * (eps_y + nu * eps_x),
        "membrane_xy": E / (2.0 * (1.0 + nu)) * gamma_xy,
    }
    errors = {}
    spreads = {}
    for key, target in expected.items():
        values = np.asarray(stresses[key], dtype=float)
        errors[key] = float(np.max(np.abs(values - target)) / max(abs(float(target)), 1.0))
        spreads[key] = float((np.max(values) - np.min(values)) / max(abs(float(target)), 1.0))
    return {"relative_errors": errors, "relative_spreads": spreads}


def bending_patch_metric(coords: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Constant curvature patch error on one S4 element."""
    model = _single_s4_model(coords)
    element = model.mesh.get_element(1)
    material = model.get_material("steel")
    kappa_x = 1.1e-3
    coords_array = element.get_node_coordinates(model.mesh)
    u = np.zeros(element.total_dofs, dtype=float)
    for local_node_index, coord in enumerate(coords_array):
        x = coord[0]
        u[local_node_index * 6 + 4] = kappa_x * x
    stresses = element.compute_stresses(model.mesh, u, material)
    expected = material.elastic_modulus * element.thickness / (2.0 * (1.0 - material.poisson_ratio**2)) * kappa_x
    values = np.asarray(stresses["bending_xx"], dtype=float)
    return {
        "relative_error": float(np.max(np.abs(values - expected)) / max(abs(expected), 1.0)),
        "relative_spread": float((np.max(values) - np.min(values)) / max(abs(expected), 1.0)),
    }


def shear_patch_metric(coords: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Constant transverse shear patch on one S4 element."""
    model = _single_s4_model(coords)
    element = model.mesh.get_element(1)
    material = model.get_material("steel")
    gamma_xz = 2.0e-4
    coords_array = element.get_node_coordinates(model.mesh)
    u = np.zeros(element.total_dofs, dtype=float)
    for local_node_index, coord in enumerate(coords_array):
        x = coord[0]
        u[local_node_index * 6 + 2] = gamma_xz * x
    stresses = element.compute_stresses(model.mesh, u, material)
    expected = material.shear_modulus * (5.0 / 6.0) * gamma_xz
    values = np.asarray(stresses["shear_xz"], dtype=float)
    return {
        "relative_error": float(np.max(np.abs(values - expected)) / max(abs(expected), 1.0)),
        "relative_spread": float((np.max(values) - np.min(values)) / max(abs(expected), 1.0)),
        "target_shear_pa": float(expected),
    }


def thin_plate_locking_sweep(
    thicknesses: Sequence[float] = (0.01, 0.003, 0.001),
    length: float = 1.0,
    width: float = 0.1,
    num_divisions: int = 10,
) -> Tuple[Dict[str, Any], ...]:
    """Cantilever strip sweep against beam bending reference."""
    rows: List[Dict[str, Any]] = []
    for thickness in thicknesses:
        model = FEModel(name="s4_locking_strip")
        model.add_material("steel", E_STEEL, NU_STEEL)
        nid = {}
        node_id = 1
        for i in range(num_divisions + 1):
            for j in range(2):
                model.add_node(node_id, length * i / num_divisions, width * j, 0.0)
                nid[(i, j)] = node_id
                node_id += 1
        for i in range(num_divisions):
            model.add_element(
                i + 1,
                ShellElement(i + 1, [nid[(i, 0)], nid[(i + 1, 0)], nid[(i + 1, 1)], nid[(i, 1)]], "steel", thickness),
            )
        model.add_boundary_condition(FixedSupport("fixed", [nid[(0, 0)], nid[(0, 1)]]))
        load_case = LoadCase("tip")
        load_case.add_nodal_load(nid[(num_divisions, 0)], [0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        load_case.add_nodal_load(nid[(num_divisions, 1)], [0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
        displacements, solver_info = solve_linear(model, load_case)
        w_tip = 0.5 * (
            displacements[model.mesh.get_node(nid[(num_divisions, 0)]).dofs[2]]
            + displacements[model.mesh.get_node(nid[(num_divisions, 1)]).dofs[2]]
        )
        reference = length**3 / (3.0 * E_STEEL * width * thickness**3 / 12.0)
        rows.append(
            {
                "thickness": float(thickness),
                "span_to_thickness": float(length / thickness),
                "tip_displacement": float(w_tip),
                "beam_reference_displacement": float(reference),
                "ratio_to_reference": float(w_tip / reference),
                "relative_error": float(abs(w_tip - reference) / max(abs(reference), 1.0e-30)),
                "solver_status": str((solver_info.get("convergence_info") or {}).get("status", "unknown")),
            }
        )
    return tuple(rows)


def s4_s8_comparison(divisions: Sequence[int] = (2, 4), thickness: float = 0.01) -> Tuple[Dict[str, Any], ...]:
    """Compare S4 and S8 responses for matching simple supported panel sweeps."""
    rows: List[Dict[str, Any]] = []
    for division in divisions:
        s4 = run_simple_supported_shell_benchmark(divisions_x=int(division), divisions_y=int(division), thickness=thickness, use_8node_elements=False)
        s8 = run_simple_supported_shell_benchmark(divisions_x=int(division), divisions_y=int(division), thickness=thickness, use_8node_elements=True)
        rows.append(
            {
                "division": int(division),
                "s4_nodes": int(s4.node_count),
                "s8_nodes": int(s8.node_count),
                "s4_displacement": float(s4.max_out_of_plane_displacement),
                "s8_displacement": float(s8.max_out_of_plane_displacement),
                "displacement_ratio_s4_to_s8": float(s4.max_out_of_plane_displacement / max(s8.max_out_of_plane_displacement, 1.0e-30)),
                "s4_von_mises": float(s4.max_von_mises_stress),
                "s8_von_mises": float(s8.max_von_mises_stress),
                "stress_ratio_s4_to_s8": float(s4.max_von_mises_stress / max(s8.max_von_mises_stress, 1.0e-30)),
                "s4_status": s4.solver_status,
                "s8_status": s8.solver_status,
            }
        )
    return tuple(rows)


def generate_s4_validity_report() -> Dict[str, Any]:
    """Generate a local S4 validity report."""
    geometries = reference_s4_geometries()
    geometry_metrics: Dict[str, Any] = {}
    for name, coords in geometries.items():
        metrics = {"free_modes": free_element_mode_metric(coords)}
        if name != "mild_warp":
            metrics["membrane_patch"] = membrane_patch_metric(coords)
            metrics["bending_patch"] = bending_patch_metric(coords)
            metrics["shear_patch"] = shear_patch_metric(coords)
        else:
            model = _single_s4_model(coords)
            element = model.mesh.get_element(1)
            K = element.compute_stiffness_matrix(model.mesh, model.get_material("steel"))
            metrics["warped_quad"] = {
                "stiffness_finite": bool(np.all(np.isfinite(K))),
                "relative_symmetry_error": float(np.linalg.norm(K - K.T) / max(np.linalg.norm(K), 1.0)),
            }
        geometry_metrics[name] = metrics

    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_sha(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "element": "S4",
        "theory_notes": [
            "S4 uses a Mindlin-Reissner shell with MITC-style assumed transverse shear for 4-node quadrilaterals.",
            "Metrics are local regression evidence; external CalculiX/SESTRA comparisons remain a separate validation layer.",
        ],
        "geometry_metrics": geometry_metrics,
        "thin_plate_locking_sweep": list(thin_plate_locking_sweep()),
        "s4_s8_comparison": list(s4_s8_comparison()),
    }


def write_s4_validity_report(path: Path | str = DEFAULT_S4_VALIDITY_PATH) -> Dict[str, Any]:
    """Write the S4 validity report as JSON."""
    report = generate_s4_validity_report()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
