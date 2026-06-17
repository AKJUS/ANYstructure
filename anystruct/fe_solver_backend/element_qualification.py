"""Q8 shell, beam, and mass qualification metrics."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Type

import numpy as np

from .assembly import solve_linear
from .boundary import BoundaryCondition, FixedSupport, LoadCase
from .elements import BeamElement, QuadraticBeamElement, ShellElement
from .fe_core import FEModel
from .mass_properties import calculate_mass_properties
from .shell_benchmarks import run_simple_supported_shell_benchmark

DEFAULT_ELEMENT_QUALIFICATION_PATH = Path("reports/element_qualification/element_qualification_report.json")

E_STEEL = 210.0e9
NU_STEEL = 0.3


def reference_q8_geometries() -> Dict[str, Tuple[Tuple[float, float, float], ...]]:
    """Representative Q8 geometries with corner and midside node ordering."""
    square = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (0.5, 1.0, 0.0),
        (0.0, 0.5, 0.0),
    )
    skew = (
        (0.0, 0.0, 0.0),
        (1.35, 0.08, 0.0),
        (1.08, 0.95, 0.0),
        (-0.18, 1.12, 0.0),
        (0.675, 0.04, 0.0),
        (1.215, 0.515, 0.0),
        (0.45, 1.035, 0.0),
        (-0.09, 0.56, 0.0),
    )
    distorted_midside = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.52, -0.025, 0.0),
        (1.035, 0.47, 0.0),
        (0.46, 1.02, 0.0),
        (-0.025, 0.55, 0.0),
    )
    return {"square": square, "skew": skew, "distorted_midside": distorted_midside}


def _single_q8_model(coords: Sequence[Sequence[float]], thickness: float = 0.01) -> FEModel:
    model = FEModel("single_q8")
    model.add_material("steel", E_STEEL, NU_STEEL, density=7850.0)
    for node_id, coord in enumerate(coords, start=1):
        model.add_node(node_id, float(coord[0]), float(coord[1]), float(coord[2]))
    model.add_element(1, ShellElement(1, list(range(1, 9)), "steel", thickness))
    return model


def q8_free_mode_metric(coords: Sequence[Sequence[float]]) -> Dict[str, Any]:
    model = _single_q8_model(coords)
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
        "relative_symmetry_error": float(np.linalg.norm(K - K.T) / max(np.linalg.norm(K), 1.0)),
    }


def q8_patch_metric(coords: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """Affine membrane, bending, and shear patch metrics for one Q8."""
    model = _single_q8_model(coords)
    element = model.mesh.get_element(1)
    material = model.get_material("steel")
    xyz = element.get_node_coordinates(model.mesh)

    eps_x, eps_y, gamma_xy = 1.2e-4, -0.4e-4, 0.7e-4
    u = np.zeros(element.total_dofs, dtype=float)
    for local, coord in enumerate(xyz):
        x, y, _ = coord
        base = local * 6
        u[base + 0] = eps_x * x
        u[base + 1] = eps_y * y + gamma_xy * x
    stresses = element.compute_stresses(model.mesh, u, material)
    E = material.elastic_modulus
    nu = material.poisson_ratio
    expected = {
        "membrane_xx": E / (1.0 - nu**2) * (eps_x + nu * eps_y),
        "membrane_yy": E / (1.0 - nu**2) * (eps_y + nu * eps_x),
        "membrane_xy": E / (2.0 * (1.0 + nu)) * gamma_xy,
    }
    membrane_errors = {
        key: float(np.max(np.abs(np.asarray(stresses[key], dtype=float) - target)) / max(abs(target), 1.0))
        for key, target in expected.items()
    }

    kappa_x = 1.1e-3
    u = np.zeros(element.total_dofs, dtype=float)
    for local, coord in enumerate(xyz):
        u[local * 6 + 4] = kappa_x * coord[0]
    stresses = element.compute_stresses(model.mesh, u, material)
    bending_expected = E * element.thickness / (2.0 * (1.0 - nu**2)) * kappa_x
    bending_values = np.asarray(stresses["bending_xx"], dtype=float)

    gamma_xz = 2.0e-4
    u = np.zeros(element.total_dofs, dtype=float)
    for local, coord in enumerate(xyz):
        u[local * 6 + 2] = gamma_xz * coord[0]
    stresses = element.compute_stresses(model.mesh, u, material)
    shear_expected = material.shear_modulus * (5.0 / 6.0) * gamma_xz
    shear_values = np.asarray(stresses["shear_xz"], dtype=float)

    return {
        "membrane_relative_errors": membrane_errors,
        "membrane_max_relative_error": max(membrane_errors.values()),
        "bending_relative_error": float(np.max(np.abs(bending_values - bending_expected)) / max(abs(bending_expected), 1.0)),
        "bending_relative_spread": float((np.max(bending_values) - np.min(bending_values)) / max(abs(bending_expected), 1.0)),
        "shear_relative_error": float(np.max(np.abs(shear_values - shear_expected)) / max(abs(shear_expected), 1.0)),
        "shear_relative_spread": float((np.max(shear_values) - np.min(shear_values)) / max(abs(shear_expected), 1.0)),
    }


def q8_mass_metric(coords: Sequence[Sequence[float]], thickness: float = 0.01) -> Dict[str, Any]:
    model = _single_q8_model(coords, thickness=thickness)
    props = calculate_mass_properties(model)
    expected = 7850.0 * thickness
    if len(coords) == 8 and np.allclose(np.asarray(coords, dtype=float)[:, 2], 0.0):
        corners = np.asarray(coords[:4], dtype=float)
        area = 0.5 * np.linalg.norm(np.cross(corners[1] - corners[0], corners[2] - corners[0]))
        area += 0.5 * np.linalg.norm(np.cross(corners[2] - corners[0], corners[3] - corners[0]))
        expected *= area
    return {
        "total_mass": float(props.total_mass),
        "expected_mass_from_corner_area": float(expected),
        "relative_mass_error": float(abs(props.total_mass - expected) / max(abs(expected), 1.0)),
        "assembled_translation_masses": props.assembled_translation_masses,
        "center_of_mass": props.center_of_mass.tolist(),
    }


def q4_q8_convergence_cost_sweep(divisions: Sequence[int] = (2, 4)) -> Tuple[Dict[str, Any], ...]:
    rows: List[Dict[str, Any]] = []
    for div in divisions:
        q4 = run_simple_supported_shell_benchmark(divisions_x=int(div), divisions_y=int(div), use_8node_elements=False)
        q8 = run_simple_supported_shell_benchmark(divisions_x=int(div), divisions_y=int(div), use_8node_elements=True)
        rows.append(
            {
                "division": int(div),
                "q4_nodes": int(q4.node_count),
                "q8_nodes": int(q8.node_count),
                "q4_elements": int(q4.element_count),
                "q8_elements": int(q8.element_count),
                "q4_dofs": int(q4.node_count * 6),
                "q8_dofs": int(q8.node_count * 6),
                "q4_displacement": float(q4.max_out_of_plane_displacement),
                "q8_displacement": float(q8.max_out_of_plane_displacement),
                "displacement_ratio_q4_to_q8": float(q4.max_out_of_plane_displacement / max(q8.max_out_of_plane_displacement, 1.0e-30)),
                "q4_von_mises": float(q4.max_von_mises_stress),
                "q8_von_mises": float(q8.max_von_mises_stress),
                "stress_ratio_q4_to_q8": float(q4.max_von_mises_stress / max(q8.max_von_mises_stress, 1.0e-30)),
                "q4_status": q4.solver_status,
                "q8_status": q8.solver_status,
            }
        )
    return tuple(rows)


def _cantilever_model(element_cls: Type[BeamElement], axis: str, n_elem: int = 12) -> Tuple[FEModel, int, Dict[str, float]]:
    length = 2.0
    section = {
        "area": 0.01,
        "Iy": 8.0e-6,
        "Iz": 1.0e-6,
        "J": 1.0e-6,
        "shear_factor_y": 5.0 / 6.0,
        "shear_factor_z": 5.0 / 6.0,
        "orientation": (0.0, 0.0, 1.0),
    }
    model = FEModel(f"{element_cls.__name__}_{axis}_cantilever")
    model.add_material("steel", E_STEEL, NU_STEEL, density=7850.0)
    num_nodes = n_elem + 1 if element_cls is BeamElement else 2 * n_elem + 1
    for i in range(num_nodes):
        s = length * i / (num_nodes - 1)
        model.add_node(i + 1, s if axis == "X" else 0.0, s if axis == "Y" else 0.0, 0.0)
    if element_cls is BeamElement:
        for e in range(n_elem):
            model.add_element(e + 1, BeamElement(e + 1, [e + 1, e + 2], "steel", section))
    else:
        for e in range(n_elem):
            base = 2 * e + 1
            model.add_element(e + 1, QuadraticBeamElement(e + 1, [base, base + 1, base + 2], "steel", section))
    model.add_boundary_condition(FixedSupport("fixed", [1]))
    return model, num_nodes, section


def beam_qualification_metrics() -> Dict[str, Any]:
    length = 2.0
    force = 1.0e4
    shear = 5.0 / 6.0
    G = E_STEEL / (2.0 * (1.0 + NU_STEEL))
    rows = []
    for element_cls in (BeamElement, QuadraticBeamElement):
        for axis in ("X", "Y"):
            model, tip, section = _cantilever_model(element_cls, axis)
            load = LoadCase("strong_axis_tip")
            load.add_nodal_load(tip, [0.0, 0.0, force, 0.0, 0.0, 0.0])
            u, info = solve_linear(model, load)
            observed = float(u[model.mesh.get_node(tip).dofs[2]])
            expected = force * length**3 / (3.0 * E_STEEL * section["Iy"]) + force * length / (shear * G * section["area"])
            rows.append(
                {
                    "element": element_cls.__name__,
                    "axis": axis,
                    "case": "strong_axis_tip",
                    "observed": observed,
                    "expected": expected,
                    "relative_error": abs(observed - expected) / expected,
                    "solver_status": str((info.get("convergence_info") or {}).get("status", "unknown")),
                }
            )
    model, tip, section = _cantilever_model(BeamElement, "X")
    torque = 100.0
    load = LoadCase("torsion")
    load.add_nodal_load(tip, [0.0, 0.0, 0.0, torque, 0.0, 0.0])
    u, info = solve_linear(model, load)
    theta = float(u[model.mesh.get_node(tip).dofs[3]])
    expected_theta = torque * length / (G * section["J"])
    rows.append(
        {
            "element": "BeamElement",
            "axis": "X",
            "case": "torsion",
            "observed": theta,
            "expected": expected_theta,
            "relative_error": abs(theta - expected_theta) / expected_theta,
            "solver_status": str((info.get("convergence_info") or {}).get("status", "unknown")),
        }
    )
    mass_model, _, _ = _cantilever_model(BeamElement, "X", n_elem=1)
    mass_props = calculate_mass_properties(mass_model)
    return {
        "response": rows,
        "max_relative_error": max(row["relative_error"] for row in rows),
        "beam_mass": {
            "total_mass": mass_props.total_mass,
            "assembled_translation_masses": mass_props.assembled_translation_masses,
            "center_of_mass": mass_props.center_of_mass.tolist(),
        },
    }


def generate_element_qualification_report() -> Dict[str, Any]:
    geometries = reference_q8_geometries()
    q8_metrics: Dict[str, Any] = {}
    for name, coords in geometries.items():
        entry = {
            "free_modes": q8_free_mode_metric(coords),
            "mass": q8_mass_metric(coords),
        }
        if name == "square":
            entry["patch"] = q8_patch_metric(coords)
        else:
            model = _single_q8_model(coords)
            element = model.mesh.get_element(1)
            K = element.compute_stiffness_matrix(model.mesh, model.get_material("steel"))
            entry["distortion"] = {
                "stiffness_finite": bool(np.all(np.isfinite(K))),
                "relative_symmetry_error": float(np.linalg.norm(K - K.T) / max(np.linalg.norm(K), 1.0)),
            }
        q8_metrics[name] = entry
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "q8": {
            "geometry_metrics": q8_metrics,
            "q4_q8_convergence_cost_sweep": list(q4_q8_convergence_cost_sweep()),
        },
        "beam": beam_qualification_metrics(),
        "known_limitations": [
            "Q8 metrics are local algebraic and internal benchmark evidence, not external CalculiX validation.",
            "2-node beam mass remains lumped; 3-node beam mass is consistently integrated.",
            "Real-section beam fibers and nonlinear tangent finite-difference checks remain later priority-6 work.",
        ],
    }


def write_element_qualification_report(path: Path | str = DEFAULT_ELEMENT_QUALIFICATION_PATH) -> Dict[str, Any]:
    report = generate_element_qualification_report()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
