"""Plasticity and nonlinear tangent qualification metrics."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .elements import BeamElement, ShellElement
from .fe_core import FEModel
from .material_curves import DNVC208MaterialCurve, FiberSectionPlasticityConfig, dnv_c208_steel_curve
from .plasticity import plane_stress_elastic_matrix, plane_stress_return_map

DEFAULT_PLASTICITY_QUALIFICATION_PATH = Path("reports/plasticity_qualification/plasticity_qualification_report.json")

E_STEEL = 210.0e9
NU_STEEL = 0.3

_P_MATRIX = np.array(
    [[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 6.0]],
    dtype=float,
) / 3.0


def reference_plastic_curve() -> DNVC208MaterialCurve:
    """Nearly elastic-perfectly plastic reference curve used by local checks."""
    return DNVC208MaterialCurve(
        sigma_prop=354.0e6,
        sigma_yield=355.0e6,
        sigma_yield_2=355.5e6,
        eps_p_y1=0.004,
        eps_p_y2=0.1,
        K=400.0e6,
        n=0.2,
    )


def yield_function_residual(stress: np.ndarray, alpha: float, curve: DNVC208MaterialCurve) -> float:
    """Scaled plane-stress J2 yield residual."""
    sigma = np.asarray(stress, dtype=float).reshape(3)
    sy = float(curve.flow_stress(np.array([float(alpha)], dtype=float))[0])
    residual = 0.5 * float(sigma @ _P_MATRIX @ sigma) - sy**2 / 3.0
    return float(residual / max(sy**2, 1.0))


def _material_tangent_fd_error(
    strain: np.ndarray,
    curve: DNVC208MaterialCurve,
    plastic: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    step: float = 1.0e-7,
) -> Dict[str, Any]:
    strain = np.asarray(strain, dtype=float).reshape(1, 3)
    plastic = np.zeros_like(strain) if plastic is None else np.asarray(plastic, dtype=float).reshape(1, 3)
    alpha = np.zeros(1, dtype=float) if alpha is None else np.asarray(alpha, dtype=float).reshape(1)
    stress, tangent, plastic_new, alpha_new = plane_stress_return_map(
        strain, plastic, alpha, E_STEEL, NU_STEEL, curve
    )
    fd = np.zeros((3, 3), dtype=float)
    for col in range(3):
        perturb = np.zeros_like(strain)
        perturb[0, col] = step
        sp = plane_stress_return_map(strain + perturb, plastic, alpha, E_STEEL, NU_STEEL, curve)[0][0]
        sm = plane_stress_return_map(strain - perturb, plastic, alpha, E_STEEL, NU_STEEL, curve)[0][0]
        fd[:, col] = (sp - sm) / (2.0 * step)
    error = float(np.linalg.norm(tangent[0] - fd) / max(np.linalg.norm(fd), 1.0))
    return {
        "stress": stress[0].tolist(),
        "alpha": float(alpha_new[0]),
        "max_plastic_strain_component": float(np.max(np.abs(plastic_new))),
        "yield_residual": yield_function_residual(stress[0], float(alpha_new[0]), curve),
        "tangent_fd_relative_error": error,
        "tangent_status": "tight" if error < 1.0e-3 else "diagnostic_current_continuum_tangent",
    }


def material_point_path_metrics() -> Dict[str, Any]:
    """Qualify return-map yield consistency over representative strain paths."""
    curve = reference_plastic_curve()
    elastic_curve = None
    elastic_strain = np.array([[1.0e-4, -0.5e-4, 0.25e-4]], dtype=float)
    elastic_stress, elastic_tangent, _, _ = plane_stress_return_map(
        elastic_strain, np.zeros_like(elastic_strain), np.zeros(1), E_STEEL, NU_STEEL, elastic_curve
    )
    elastic_expected = elastic_strain @ plane_stress_elastic_matrix(E_STEEL, NU_STEEL).T

    paths = {
        "uniaxial": _material_tangent_fd_error(np.array([0.002, 0.0, 0.0]), curve),
        "biaxial_shear": _material_tangent_fd_error(np.array([0.003, 0.001, 0.0005]), curve),
        "pure_shear": _material_tangent_fd_error(np.array([0.0, 0.0, 0.004]), curve),
    }

    first = _material_tangent_fd_error(np.array([0.003, 0.0, 0.0]), curve)
    plastic_old = np.asarray(first["stress"], dtype=float).reshape(1, 3) * 0.0
    stress, _tangent, plastic_new, alpha_new = plane_stress_return_map(
        np.array([[0.003, 0.0, 0.0]], dtype=float),
        np.zeros((1, 3), dtype=float),
        np.zeros(1, dtype=float),
        E_STEEL,
        NU_STEEL,
        curve,
    )
    unload_stress, _, _, unload_alpha = plane_stress_return_map(
        np.array([[0.001, 0.0, 0.0]], dtype=float),
        plastic_new,
        alpha_new,
        E_STEEL,
        NU_STEEL,
        curve,
    )
    paths["unload_from_plastic"] = {
        "stress": unload_stress[0].tolist(),
        "alpha": float(unload_alpha[0]),
        "alpha_change": float(unload_alpha[0] - alpha_new[0]),
        "yield_residual": yield_function_residual(stress[0], float(alpha_new[0]), curve),
    }

    return {
        "elastic": {
            "stress_relative_error": float(np.linalg.norm(elastic_stress - elastic_expected) / max(np.linalg.norm(elastic_expected), 1.0)),
            "tangent_relative_error": float(
                np.linalg.norm(elastic_tangent[0] - plane_stress_elastic_matrix(E_STEEL, NU_STEEL))
                / max(np.linalg.norm(plane_stress_elastic_matrix(E_STEEL, NU_STEEL)), 1.0)
            ),
        },
        "plastic_paths": paths,
        "max_abs_yield_residual": max(abs(path.get("yield_residual", 0.0)) for path in paths.values()),
        "max_material_tangent_fd_error": max(
            path.get("tangent_fd_relative_error", 0.0) for path in paths.values()
        ),
    }


def _finite_difference_element_tangent(
    element: Any,
    model: FEModel,
    u_elem: np.ndarray,
    state: Any = None,
    num_layers: int = 5,
    step: float = 1.0e-7,
) -> Dict[str, Any]:
    material = model.get_material(element.material_name)
    f, K, trial_state = element.compute_nonlinear_response(
        model.mesh, material, u_elem, state, num_layers=num_layers, tangent=True
    )
    fd = np.zeros_like(K)
    for col in range(K.shape[1]):
        perturb = np.zeros_like(u_elem)
        perturb[col] = step
        fp = element.compute_nonlinear_response(
            model.mesh, material, u_elem + perturb, state, num_layers=num_layers, tangent=False
        )[0]
        fm = element.compute_nonlinear_response(
            model.mesh, material, u_elem - perturb, state, num_layers=num_layers, tangent=False
        )[0]
        fd[:, col] = (fp - fm) / (2.0 * step)
    error = float(np.linalg.norm(K - fd) / max(np.linalg.norm(fd), 1.0))
    return {
        "tangent_fd_relative_error": error,
        "force_norm": float(np.linalg.norm(f)),
        "tangent_norm": float(np.linalg.norm(K)),
        "fd_tangent_norm": float(np.linalg.norm(fd)),
        "state_summary": _state_summary(trial_state),
    }


def _state_summary(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    summary: Dict[str, Any] = {}
    for key in ("alpha", "plastic_strain", "layer_strain", "fiber_strain"):
        value = np.asarray(state.get(key, []), dtype=float)
        if value.size:
            summary[f"{key}_max"] = float(np.max(value))
            summary[f"{key}_min"] = float(np.min(value))
            summary[f"{key}_max_abs"] = float(np.max(np.abs(value)))
    if "axial_force" in state:
        summary["axial_force"] = float(state["axial_force"])
    return summary


def _beam_model(curve: DNVC208MaterialCurve | None = None, fiber: bool = False) -> Tuple[FEModel, BeamElement]:
    model = FEModel("beam_tangent_metric")
    model.add_material("steel", E_STEEL, NU_STEEL, hardening_curve=curve)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    section = {"area": 0.01, "Iy": 1.0e-5, "Iz": 1.0e-5, "J": 1.0e-5}
    if fiber:
        section["fiber_plasticity"] = FiberSectionPlasticityConfig(5, 5)
    element = BeamElement(1, [1, 2], "steel", section)
    model.add_element(1, element)
    return model, element


def _shell_model(curve: DNVC208MaterialCurve | None = None) -> Tuple[FEModel, ShellElement]:
    model = FEModel("shell_tangent_metric")
    model.add_material("steel", E_STEEL, NU_STEEL, hardening_curve=curve)
    for node_id, coord in enumerate(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)), start=1):
        model.add_node(node_id, *coord)
    element = ShellElement(1, [1, 2, 3, 4], "steel", 0.01)
    model.add_element(1, element)
    return model, element


def element_tangent_metrics() -> Dict[str, Any]:
    curve = reference_plastic_curve()

    beam_elastic_model, beam_elastic = _beam_model()
    u_beam_elastic = np.zeros(12, dtype=float)
    u_beam_elastic[6] = 0.001
    u_beam_elastic[7] = 0.0003
    beam_elastic_metric = _finite_difference_element_tangent(beam_elastic, beam_elastic_model, u_beam_elastic)

    beam_plastic_model, beam_plastic = _beam_model(curve, fiber=True)
    u_beam_plastic = np.zeros(12, dtype=float)
    u_beam_plastic[6] = 0.006
    u_beam_plastic[7] = 0.0003
    beam_plastic_metric = _finite_difference_element_tangent(beam_plastic, beam_plastic_model, u_beam_plastic)

    shell_elastic_model, shell_elastic = _shell_model()
    u_shell_elastic = np.zeros(24, dtype=float)
    u_shell_elastic[2::6] = [0.0, 0.001, 0.001, 0.0]
    u_shell_elastic[4::6] = [0.001, 0.001, 0.001, 0.001]
    shell_elastic_metric = _finite_difference_element_tangent(
        shell_elastic, shell_elastic_model, u_shell_elastic, step=1.0e-8
    )

    shell_plastic_model, shell_plastic = _shell_model(curve)
    u_shell_plastic = np.zeros(24, dtype=float)
    coords = shell_plastic.get_node_coordinates(shell_plastic_model.mesh)
    for local, coord in enumerate(coords):
        x, y, _ = coord
        base = local * 6
        u_shell_plastic[base + 0] = 0.003 * x
        u_shell_plastic[base + 1] = -0.0008 * y
        u_shell_plastic[base + 4] = 0.002 * x
    shell_plastic_metric = _finite_difference_element_tangent(
        shell_plastic, shell_plastic_model, u_shell_plastic, num_layers=5
    )
    shell_plastic_metric["tangent_status"] = (
        "tight" if shell_plastic_metric["tangent_fd_relative_error"] < 1.0e-4 else "diagnostic_high_tangent_error"
    )

    max_algorithmic_error = max(
        beam_elastic_metric["tangent_fd_relative_error"],
        beam_plastic_metric["tangent_fd_relative_error"],
        shell_elastic_metric["tangent_fd_relative_error"],
        shell_plastic_metric["tangent_fd_relative_error"],
    )
    return {
        "beam_elastic": beam_elastic_metric,
        "beam_fiber_plastic": beam_plastic_metric,
        "shell_elastic": shell_elastic_metric,
        "shell_layered_plastic": shell_plastic_metric,
        "max_tight_tangent_error": max(
            beam_elastic_metric["tangent_fd_relative_error"],
            beam_plastic_metric["tangent_fd_relative_error"],
            shell_elastic_metric["tangent_fd_relative_error"],
        ),
        "max_algorithmic_tangent_error": max_algorithmic_error,
    }


def dnv_curve_metric() -> Dict[str, Any]:
    curves = {}
    for grade, thickness in (("S355", 0.010), ("S420", 0.020), ("S460", 0.050)):
        curve = dnv_c208_steel_curve(grade, thickness)
        curves[f"{grade}_{thickness:g}"] = {
            "sigma_prop": curve.sigma_prop,
            "sigma_yield": curve.sigma_yield,
            "sigma_yield_2": curve.sigma_yield_2,
            "eps_p_y1": curve.eps_p_y1,
            "eps_p_y2": curve.eps_p_y2,
            "K": curve.K,
            "n": curve.n,
        }
    return curves


def generate_plasticity_qualification_report() -> Dict[str, Any]:
    material = material_point_path_metrics()
    element = element_tangent_metrics()
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "dnv_curves": dnv_curve_metric(),
        "material_point": material,
        "element_tangents": element,
        "status": "passed"
        if material["max_abs_yield_residual"] < 1.0e-8 and element["max_algorithmic_tangent_error"] < 1.0e-4
        else "diagnostic",
        "known_limitations": [
            "Plastic material tangents are now consistent numerical algorithmic tangents of the discrete return map.",
            "The numerical tangent is intentionally correctness-first and more expensive than a closed-form analytical algorithmic tangent.",
            "A future speed batch should replace the numerical tangent with an analytical derivative after preserving these finite-difference checks.",
        ],
    }


def write_plasticity_qualification_report(path: Path | str = DEFAULT_PLASTICITY_QUALIFICATION_PATH) -> Dict[str, Any]:
    report = generate_plasticity_qualification_report()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
