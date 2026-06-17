"""Lightweight experimental FEM solver owned by ANYstructure.

This module is intentionally small and dependency-light.  It provides the
runtime API used by the experimental FEM popup while the production solver is
developed in ANYintelligent.  Future solver updates can replace this module
without changing the GUI handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import collections
import math

import numpy as np

try:
    from anystruct import fe_solver_backend as _full_backend
    from anystruct.fe_solver_backend.assembly import compute_stresses as _backend_compute_stresses
    from anystruct.fe_solver_backend.assembly import solve_linear as _backend_solve_linear
    from anystruct.fe_solver_backend.buckling import solve_eigenvalue_buckling as _backend_solve_buckling
    from anystruct.fe_solver_backend.material_curves import curve_from_properties as _backend_curve_from_properties
    from anystruct.fe_solver_backend.nonlinear import solve_nonlinear_load_stepping as _backend_solve_nonlinear_limit
    from anystruct.fe_solver_backend.nonlinear_static import solve_static_nonlinear as _backend_solve_static_nonlinear
    from anystruct.fe_solver_backend.dynamics import solve_transient_newmark as _backend_solve_transient_newmark
    from anystruct.fe_solver_backend.validation import load_case_resultant as _backend_load_case_resultant
except ModuleNotFoundError:
    try:
        from ANYstructure.anystruct import fe_solver_backend as _full_backend
        from ANYstructure.anystruct.fe_solver_backend.assembly import compute_stresses as _backend_compute_stresses
        from ANYstructure.anystruct.fe_solver_backend.assembly import solve_linear as _backend_solve_linear
        from ANYstructure.anystruct.fe_solver_backend.buckling import solve_eigenvalue_buckling as _backend_solve_buckling
        from ANYstructure.anystruct.fe_solver_backend.material_curves import curve_from_properties as _backend_curve_from_properties
        from ANYstructure.anystruct.fe_solver_backend.nonlinear import solve_nonlinear_load_stepping as _backend_solve_nonlinear_limit
        from ANYstructure.anystruct.fe_solver_backend.nonlinear_static import solve_static_nonlinear as _backend_solve_static_nonlinear
        from ANYstructure.anystruct.fe_solver_backend.dynamics import solve_transient_newmark as _backend_solve_transient_newmark
        from ANYstructure.anystruct.fe_solver_backend.validation import load_case_resultant as _backend_load_case_resultant
    except ModuleNotFoundError:
        _full_backend = None
        _backend_compute_stresses = None
        _backend_solve_linear = None
        _backend_solve_buckling = None
        _backend_curve_from_properties = None
        _backend_solve_nonlinear_limit = None
        _backend_solve_static_nonlinear = None
        _backend_solve_transient_newmark = None
        _backend_load_case_resultant = None


@dataclass(frozen=True)
class LightweightFEMConfig:
    """Runtime options for the local lightweight solver."""

    mesh_fidelity: str = "coarse"
    pressure_pa: float = 0.0
    load_scale: float = 1.0
    include_stiffeners: bool = True
    include_girders: bool = True
    include_end_lids: bool = False
    num_buckling_modes: int = 5
    mesh_size_m: float = 0.0
    top_bottom_moment_nm: float = 0.0
    boundary_condition: str = "auto"
    symmetry_mode: str = "none"
    shell_element_order: str = "S4"
    analysis_type: str = "linear eigenvalue"
    buckling_analysis_type: str = "linear eigenvalue"
    pressure_direction: str = "external"
    axial_force_n: float = 0.0
    enforced_displacement_m: float = 0.0
    stiffener_eccentricity_m: float = 0.0
    girder_eccentricity_m: float = 0.0
    member_orientation: str = "auto"
    solver_type: str = "direct"
    stress_percentile: float = 95.0
    elastic_modulus_pa: float = 210.0e9
    poisson_ratio: float = 0.3
    yield_stress_pa: float = 355.0e6
    material_model: str = "linear elastic"
    steel_grade: str = "S355"
    steel_thickness_class: str = "auto"
    nonlinear_max_load_factor: float = 3.0
    nonlinear_steps: int = 12
    nonlinear_max_iterations: int = 25
    nonlinear_tolerance: float = 1.0e-6
    nonlinear_layers: int = 5
    custom_load_bc_enabled: bool = False
    custom_loads_add_to_imported: bool = False
    custom_use_nullspace_projection: bool = False
    custom_pressure_pa: float = 0.0
    plate_edge_x0_support: str = "free"
    plate_edge_x1_support: str = "free"
    plate_edge_y0_support: str = "free"
    plate_edge_y1_support: str = "free"
    cylinder_lower_support: str = "free"
    cylinder_upper_support: str = "free"
    plate_edge_x0_load_n_per_m: float = 0.0
    plate_edge_x1_load_n_per_m: float = 0.0
    plate_edge_y0_load_n_per_m: float = 0.0
    plate_edge_y1_load_n_per_m: float = 0.0
    cylinder_lower_edge_load_n_per_m: float = 0.0
    cylinder_upper_edge_load_n_per_m: float = 0.0
    slamming_enabled: bool = False
    slamming_pressure_pa: float = 0.0
    slamming_duration_s: float = 0.01
    slamming_total_time_s: float = 0.05
    slamming_dt_s: float = 0.0005
    slamming_patch_center_a_m: float = 0.0
    slamming_patch_center_b_m: float = 0.0
    slamming_patch_size_a_m: float = 0.0
    slamming_patch_size_b_m: float = 0.0
    slamming_include_static_load: bool = False
    imperfection_enabled: bool = False
    imperfection_shape: str = "standard plate/cylinder"
    imperfection_amplitude_m: float = 0.0
    imperfection_wave_a: int = 1
    imperfection_wave_b: int = 1
    runtime_solver: str = "stepwise"
    allow_unbalanced_free_free: bool = False
    buckling_shift_load_factor: float = 0.0
    buckling_min_load_factor: float = 0.0
    buckling_max_load_factor: float = 0.0
    buckling_repeated_tolerance: float = 1.0e-3
    buckling_allow_dense_fallback: bool = False
    recovery_history_mode: str = "full"
    recovery_threads: int = 0
    memory_limit_mb: float = 0.0
    capacity_buckling_mode_number: int = 1
    capacity_mesh_min_elements_per_half_wave: int = 4


@dataclass(frozen=True)
class LightweightFEMResult:
    """Result contract returned to the ANYstructure runtime popup."""

    status: str
    stress_max_pa: float
    stress_p95_pa: float
    displacement_max_m: float
    buckling_factors: tuple[float, ...] = field(default_factory=tuple)
    diagnostics: tuple[str, ...] = field(default_factory=tuple)
    mesh_info: dict[str, int] = field(default_factory=dict)
    prestress_summary: dict[str, float] = field(default_factory=dict)
    load_resultant: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    visualization: dict[str, object] = field(default_factory=dict)
    solver_name: str = "ANYstructure lightweight"


def _positive(value: float, fallback: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback
    return value if value > 0.0 else fallback


_DNV_C208_STEEL_TABLES_MPA: dict[str, tuple[dict[str, float | str], ...]] = {
    "S235": (
        {"label": "t <= 16", "max_t_mm": 16.0, "E": 210000.0, "sigma_prop": 211.7, "sigma_yield": 236.2, "sigma_yield_2": 243.4, "eps_p_y1": 0.004, "eps_p_y2": 0.020, "K": 520.0, "n": 0.166},
        {"label": "16 < t <= 40", "max_t_mm": 40.0, "E": 210000.0, "sigma_prop": 202.7, "sigma_yield": 226.1, "sigma_yield_2": 233.2, "eps_p_y1": 0.004, "eps_p_y2": 0.020, "K": 520.0, "n": 0.166},
        {"label": "40 < t <= 63", "max_t_mm": 63.0, "E": 210000.0, "sigma_prop": 193.7, "sigma_yield": 216.1, "sigma_yield_2": 223.0, "eps_p_y1": 0.004, "eps_p_y2": 0.020, "K": 520.0, "n": 0.166},
        {"label": "63 < t <= 100", "max_t_mm": 100.0, "E": 210000.0, "sigma_prop": 193.7, "sigma_yield": 216.1, "sigma_yield_2": 223.0, "eps_p_y1": 0.004, "eps_p_y2": 0.020, "K": 520.0, "n": 0.166},
    ),
    "S275": (
        {"label": "t <= 16", "max_t_mm": 16.0, "E": 210000.0, "sigma_prop": 247.8, "sigma_yield": 276.5, "sigma_yield_2": 282.8, "eps_p_y1": 0.004, "eps_p_y2": 0.017, "K": 620.0, "n": 0.166},
        {"label": "16 < t <= 40", "max_t_mm": 40.0, "E": 210000.0, "sigma_prop": 238.8, "sigma_yield": 266.4, "sigma_yield_2": 272.6, "eps_p_y1": 0.004, "eps_p_y2": 0.017, "K": 620.0, "n": 0.166},
        {"label": "40 < t <= 63", "max_t_mm": 63.0, "E": 210000.0, "sigma_prop": 229.8, "sigma_yield": 256.3, "sigma_yield_2": 262.4, "eps_p_y1": 0.004, "eps_p_y2": 0.017, "K": 620.0, "n": 0.166},
    ),
    "S355": (
        {"label": "t <= 16", "max_t_mm": 16.0, "E": 210000.0, "sigma_prop": 320.0, "sigma_yield": 357.0, "sigma_yield_2": 363.3, "eps_p_y1": 0.004, "eps_p_y2": 0.015, "K": 740.0, "n": 0.166},
        {"label": "16 < t <= 40", "max_t_mm": 40.0, "E": 210000.0, "sigma_prop": 311.0, "sigma_yield": 346.9, "sigma_yield_2": 353.1, "eps_p_y1": 0.004, "eps_p_y2": 0.015, "K": 740.0, "n": 0.166},
        {"label": "40 < t <= 63", "max_t_mm": 63.0, "E": 210000.0, "sigma_prop": 301.9, "sigma_yield": 336.9, "sigma_yield_2": 342.9, "eps_p_y1": 0.004, "eps_p_y2": 0.015, "K": 725.0, "n": 0.166},
        {"label": "63 < t <= 100", "max_t_mm": 100.0, "E": 210000.0, "sigma_prop": 283.9, "sigma_yield": 316.7, "sigma_yield_2": 322.5, "eps_p_y1": 0.004, "eps_p_y2": 0.015, "K": 725.0, "n": 0.166},
    ),
    "S420": (
        {"label": "t <= 16", "max_t_mm": 16.0, "E": 210000.0, "sigma_prop": 378.7, "sigma_yield": 422.5, "sigma_yield_2": 427.6, "eps_p_y1": 0.004, "eps_p_y2": 0.012, "K": 738.0, "n": 0.140},
        {"label": "16 < t <= 40", "max_t_mm": 40.0, "E": 210000.0, "sigma_prop": 360.6, "sigma_yield": 402.4, "sigma_yield_2": 407.3, "eps_p_y1": 0.004, "eps_p_y2": 0.012, "K": 703.0, "n": 0.140},
        {"label": "40 < t <= 63", "max_t_mm": 63.0, "E": 210000.0, "sigma_prop": 351.6, "sigma_yield": 392.3, "sigma_yield_2": 397.1, "eps_p_y1": 0.004, "eps_p_y2": 0.012, "K": 686.0, "n": 0.140},
    ),
    "S460": (
        {"label": "t <= 16", "max_t_mm": 16.0, "E": 210000.0, "sigma_prop": 414.8, "sigma_yield": 462.8, "sigma_yield_2": 466.9, "eps_p_y1": 0.004, "eps_p_y2": 0.010, "K": 772.0, "n": 0.120},
        {"label": "16 < t <= 40", "max_t_mm": 40.0, "E": 210000.0, "sigma_prop": 396.7, "sigma_yield": 442.7, "sigma_yield_2": 446.6, "eps_p_y1": 0.004, "eps_p_y2": 0.010, "K": 745.0, "n": 0.120},
        {"label": "40 < t <= 63", "max_t_mm": 63.0, "E": 210000.0, "sigma_prop": 374.2, "sigma_yield": 417.5, "sigma_yield_2": 421.2, "eps_p_y1": 0.004, "eps_p_y2": 0.010, "K": 703.0, "n": 0.120},
    ),
}


def dnv_c208_steel_properties(
    grade: str = "S355",
    thickness_m: float = 0.0,
    thickness_class: str = "auto",
) -> dict[str, float | str]:
    """Return DNV-RP-C208 low-fractile steel true-stress curve properties.

    Stress values and K are returned in Pa, E in Pa and strain values as true
    plastic strain.  ``thickness_class`` may be one of the table labels or
    ``auto`` to select by plate thickness.
    """

    grade_key = str(grade or "S355").strip().upper()
    rows = _DNV_C208_STEEL_TABLES_MPA.get(grade_key, _DNV_C208_STEEL_TABLES_MPA["S355"])
    class_choice = _normalized_choice(thickness_class, "auto")
    try:
        thickness_mm = float(thickness_m) * 1000.0
    except (TypeError, ValueError):
        thickness_mm = 0.0
    if thickness_mm <= 0.0:
        thickness_mm = 16.0

    selected = rows[-1]
    if class_choice not in {"auto", "automatic", "by thickness", "auto by plate thickness"}:
        simplified = class_choice.replace(" ", "")
        for row in rows:
            if simplified == str(row["label"]).lower().replace(" ", ""):
                selected = row
                break
    else:
        for row in rows:
            if thickness_mm <= float(row["max_t_mm"]) + 1.0e-9:
                selected = row
                break

    result: dict[str, float | str] = {
        "grade": grade_key if grade_key in _DNV_C208_STEEL_TABLES_MPA else "S355",
        "thickness_class": str(selected["label"]),
        "thickness_mm": float(thickness_mm),
        "source": "DNV-RP-C208 Table 4-2 to 4-6 low-fractile true stress-strain values",
    }
    for key, value in selected.items():
        if key in {"label", "max_t_mm"}:
            continue
        if key in {"E", "sigma_prop", "sigma_yield", "sigma_yield_2", "K"}:
            result[key if key != "E" else "E_pa"] = float(value) * 1.0e6
        else:
            result[key] = float(value)
    return result


def _mesh_divisions(mesh_fidelity: str) -> int:
    return {"coarse": 8, "medium": 16, "fine": 32, "very fine": 48, "very_fine": 48}.get(str(mesh_fidelity).lower(), 8)


def _production_divisions(mesh_fidelity: str) -> int:
    return {"coarse": 4, "medium": 8, "fine": 12, "very fine": 20, "very_fine": 20}.get(str(mesh_fidelity).lower(), 4)


def _fidelity_refinement(mesh_fidelity: str) -> int:
    return {"coarse": 1, "medium": 2, "fine": 3, "very fine": 4, "very_fine": 4}.get(str(mesh_fidelity).lower(), 1)


def _requested_mesh_size(config: LightweightFEMConfig) -> float:
    try:
        size = float(config.mesh_size_m)
    except (TypeError, ValueError):
        return 0.0
    return size if size > 0.0 else 0.0


def _line_divisions(
    length: float,
    config: LightweightFEMConfig,
    fallback: int,
    max_element_size: float = 0.0,
) -> int:
    mesh_size = _requested_mesh_size(config)
    max_element_size = _positive(max_element_size, 0.0)
    if mesh_size > 0.0:
        if max_element_size > 0.0 and mesh_size > max_element_size:
            mesh_size = max_element_size
        return max(int(math.ceil(max(length, 1.0e-9) / mesh_size)), 1)
    divisions = max(int(fallback), 1)
    if max_element_size > 0.0:
        target_size = max_element_size / max(_fidelity_refinement(config.mesh_fidelity), 1)
        divisions = max(divisions, int(math.ceil(max(length, 1.0e-9) / target_size)))
    return divisions


def _axis_breaks(
    length: float,
    divisions: int,
    mandatory: tuple[float, ...] = (),
    max_element_size: float = 0.0,
) -> list[float]:
    length = max(float(length), 1.0e-9)
    divisions = max(int(divisions), 1)
    max_element_size = _positive(max_element_size, 0.0)
    if max_element_size > 0.0:
        divisions = max(divisions, int(math.ceil(length / max_element_size)))
    values = [length * idx / divisions for idx in range(divisions + 1)]
    tol = max(length * 1.0e-9, 1.0e-9)
    for value in mandatory:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if tol < value < length - tol:
            values.append(value)
    clean = []
    for value in sorted(values):
        value = min(max(float(value), 0.0), length)
        if not clean or abs(value - clean[-1]) > tol:
            clean.append(value)
    clean[0] = 0.0
    clean[-1] = length
    return clean


def _positive_spacing(value: object) -> float:
    try:
        spacing = float(value)
    except (TypeError, ValueError):
        return 0.0
    return spacing if spacing > 1.0e-9 else 0.0


def _member_positions(total_length: float, spacing: float, fallback_midpoint: bool = True) -> tuple[float, ...]:
    total_length = max(float(total_length), 1.0e-9)
    spacing = _positive_spacing(spacing)
    tol = max(total_length * 1.0e-9, 1.0e-9)
    positions: list[float] = []
    if spacing > 0.0:
        value = spacing
        while value < total_length - tol and len(positions) < 1000:
            positions.append(value)
            value += spacing
    if not positions and fallback_midpoint:
        positions = [0.5 * total_length]
    return tuple(positions)


def _centered_member_positions(total_length: float, spacing: float, fallback_midpoint: bool = True) -> tuple[float, ...]:
    """Return member stations with any cut length shared symmetrically."""

    total_length = max(float(total_length), 1.0e-9)
    spacing = _positive_spacing(spacing)
    tol = max(total_length * 1.0e-9, 1.0e-9)
    if spacing <= 0.0:
        return (0.5 * total_length,) if fallback_midpoint else ()
    full_count = int(math.floor(total_length / spacing))
    if full_count <= 0:
        return (0.5 * total_length,) if fallback_midpoint else ()
    offset = 0.5 * (total_length - full_count * spacing)
    if offset <= tol:
        positions = [spacing * idx for idx in range(1, full_count)]
    else:
        positions = [offset + spacing * idx for idx in range(full_count + 1)]
    return tuple(position for position in positions if tol < position < total_length - tol)


def _index_of_break(breaks: list[float], value: float) -> int:
    return min(range(len(breaks)), key=lambda index: abs(float(breaks[index]) - float(value)))


def _member_count_from_spacing(total_length: float, spacing: float) -> int:
    try:
        total_length = float(total_length)
        spacing = float(spacing)
    except (TypeError, ValueError):
        return 0
    if total_length <= 0.0 or spacing <= 1.0e-9:
        return 0
    return max(int(round(total_length / spacing)), 1)


def _multiple_at_least(value: int, factor: int) -> int:
    value = max(int(value), 1)
    factor = max(int(factor), 1)
    return max(factor, int(math.ceil(value / factor)) * factor)


def _sorted_positive_factors(base_factor: float, count: int) -> tuple[float, ...]:
    count = max(int(count), 1)
    base = max(float(base_factor), 1.0e-6)
    return tuple(base * (1.0 + 0.35 * mode) for mode in range(count))


def _plate_critical_stress(E: float, nu: float, thickness: float, width: float, k: float = 4.0) -> float:
    slenderness = thickness / max(width, 1.0e-9)
    return k * math.pi**2 * E * slenderness**2 / (12.0 * (1.0 - nu**2))


def _cylinder_critical_pressure(E: float, nu: float, thickness: float, radius: float) -> float:
    radius = max(radius, 1.0e-9)
    thickness = max(thickness, 1.0e-9)
    return 0.605 * E / ((1.0 - nu**2) ** 0.75) * (thickness / radius) ** 2.5


def _grid(rows: int, cols: int, value_at) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value_at(row, col)) for col in range(cols)) for row in range(rows))


def _flat_visualization(
    length: float,
    width: float,
    displacement: float,
    stress: float,
    div: int,
) -> dict[str, object]:
    rows = div + 1
    cols = div + 1

    def shape(row: int, col: int) -> float:
        x_norm = row / max(rows - 1, 1)
        y_norm = col / max(cols - 1, 1)
        return math.sin(math.pi * x_norm) * math.sin(math.pi * y_norm)

    return {
        "type": "flat",
        "x_m": _grid(rows, cols, lambda row, _col: length * row / max(rows - 1, 1)),
        "y_m": _grid(rows, cols, lambda _row, col: width * col / max(cols - 1, 1)),
        "w_m": _grid(rows, cols, lambda row, col: displacement * shape(row, col)),
        "stress_pa": _grid(rows, cols, lambda row, col: stress * (0.55 + 0.45 * shape(row, col))),
    }


def _cylinder_visualization(
    radius: float,
    length: float,
    displacement: float,
    stress: float,
    circumferential_div: int,
    axial_div: int,
) -> dict[str, object]:
    rows = axial_div + 1
    cols = circumferential_div + 1

    def axial_shape(row: int) -> float:
        x_norm = row / max(rows - 1, 1)
        return math.sin(math.pi * x_norm) ** 2

    def radial_pattern(row: int, col: int) -> float:
        theta = 2.0 * math.pi * col / max(cols - 1, 1)
        return displacement * (0.45 + 0.55 * axial_shape(row)) * (1.0 + 0.08 * math.cos(3.0 * theta))

    return {
        "type": "cylinder",
        "radius_m": radius,
        "axial_m": _grid(rows, cols, lambda row, _col: length * row / max(rows - 1, 1)),
        "theta_rad": _grid(rows, cols, lambda _row, col: 2.0 * math.pi * col / max(cols - 1, 1)),
        "radial_displacement_m": _grid(rows, cols, radial_pattern),
        "stress_pa": _grid(rows, cols, lambda row, col: stress * (0.80 + 0.20 * axial_shape(row)) * (1.0 + 0.03 * math.cos(2.0 * math.pi * col / max(cols - 1, 1)))),
    }


def _beam_section(thickness: float, reference: float, depth_factor: float) -> dict[str, float]:
    depth = max(depth_factor * reference, 6.0 * thickness, 0.05)
    width = max(2.5 * thickness, 0.03)
    area = width * depth
    iy = width * depth**3 / 12.0
    iz = depth * width**3 / 12.0
    return {
        "area": area,
        "Iy": max(iy, 1.0e-10),
        "Iz": max(iz, 1.0e-10),
        "J": max(iy + iz, 1.0e-10),
        "shear_factor_y": 5.0 / 6.0,
        "shear_factor_z": 5.0 / 6.0,
        "web_height": depth,
        "web_thickness": thickness,
        "flange_width": 0.0,
        "flange_thickness": 0.0,
    }


def _section_or_default(section: object, thickness: float, reference: float, depth_factor: float) -> dict[str, float]:
    if isinstance(section, dict):
        try:
            area = float(section.get("area", section.get("A", 0.0)))
            iy = float(section.get("Iy", section.get("iy", 0.0)))
            iz = float(section.get("Iz", section.get("iz", 0.0)))
            j = float(section.get("J", section.get("torsion_constant", iy + iz)))
        except (TypeError, ValueError):
            area = 0.0
            iy = 0.0
            iz = 0.0
            j = 0.0
        if area > 0.0 and iy > 0.0 and iz > 0.0:
            result = {
                "area": area,
                "Iy": max(iy, 1.0e-12),
                "Iz": max(iz, 1.0e-12),
                "J": max(j, 1.0e-12),
                "shear_factor_y": float(section.get("shear_factor_y", 5.0 / 6.0)),
                "shear_factor_z": float(section.get("shear_factor_z", 5.0 / 6.0)),
                "web_height": float(section.get("web_height") or section.get("web_h") or 0.1),
                "web_thickness": float(section.get("web_thickness") or section.get("web_thk") or 0.01),
                "flange_width": float(section.get("flange_width") or section.get("flange_w") or 0.0),
                "flange_thickness": float(section.get("flange_thickness") or section.get("flange_thk") or 0.0),
            }
            if section.get("label"):
                result["label"] = str(section.get("label"))
            return result
    return _beam_section(thickness, reference, depth_factor)


def _normalized_choice(value: object, default: str = "auto") -> str:
    text = str(value or default).strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split()) or default


def _wants_s8(config: LightweightFEMConfig) -> bool:
    return _normalized_choice(config.shell_element_order, "s4") in {"s8", "8 node", "8 node shell", "quadratic"}


def _shell_order_from_geometry(generated_geometry: dict) -> str:
    for shell in generated_geometry.get("shells", []) or []:
        return "S8" if len(shell.get("node_ids", [])) == 8 else "S4"
    return "S4"


def _node_lookup(nodes: list[dict[str, object]]) -> dict[int, np.ndarray]:
    return {int(node["id"]): np.asarray(node["coords"], dtype=float) for node in nodes}


def _project_cylinder_midpoint(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    midpoint = 0.5 * (a + b)
    if radius <= 0.0:
        return midpoint
    radial = midpoint[:2]
    norm = float(np.linalg.norm(radial))
    if norm > 1.0e-12:
        midpoint[:2] = radial / norm * radius
    return midpoint


def _upgrade_shells_to_s8(nodes: list[dict[str, object]], shells: list[dict[str, object]], radius: float = 0.0) -> None:
    """Convert generated 4-node quads to 8-node serendipity quads in place."""
    node_coords = _node_lookup(nodes)
    next_node_id = max(node_coords, default=0) + 1
    midside_nodes: dict[tuple[int, int], int] = {}

    def midside_id(n1: int, n2: int) -> int:
        nonlocal next_node_id
        key = tuple(sorted((int(n1), int(n2))))
        if key in midside_nodes:
            return midside_nodes[key]
        a = node_coords[int(n1)]
        b = node_coords[int(n2)]
        coords = _project_cylinder_midpoint(a, b, radius) if radius > 0.0 else 0.5 * (a + b)
        node_id = next_node_id
        next_node_id += 1
        midside_nodes[key] = node_id
        node_coords[node_id] = coords
        nodes.append({"id": node_id, "coords": coords.tolist()})
        return node_id

    for shell in shells:
        node_ids = [int(node_id) for node_id in shell.get("node_ids", [])]
        if len(node_ids) != 4:
            continue
        n1, n2, n3, n4 = node_ids
        shell["node_ids"] = [
            n1,
            n2,
            n3,
            n4,
            midside_id(n1, n2),
            midside_id(n2, n3),
            midside_id(n3, n4),
            midside_id(n4, n1),
        ]


def _axis_symmetry_constraints(axis: str) -> dict[str, float]:
    if axis == "x":
        return {"ux": 0.0, "ry": 0.0, "rz": 0.0}
    if axis == "y":
        return {"uy": 0.0, "rx": 0.0, "rz": 0.0}
    if axis == "z":
        return {"uz": 0.0, "rx": 0.0, "ry": 0.0}
    return {}


def _symmetry_supports(nodes: list[dict[str, object]], config: LightweightFEMConfig) -> list[dict[str, object]]:
    mode = _normalized_choice(config.symmetry_mode, "none")
    if mode in {"none", "off", "cyclic"}:
        return []
    axis_index = {"x": 0, "y": 1, "z": 2}.get(mode)
    constraints = _axis_symmetry_constraints(mode)
    if axis_index is None or not constraints:
        return []
    coords = _node_lookup(nodes)
    values = np.asarray([coord[axis_index] for coord in coords.values()], dtype=float)
    if values.size == 0:
        return []
    span = float(np.max(values) - np.min(values))
    tol = max(span * 1.0e-8, 1.0e-8)
    zero_nodes = [node_id for node_id, coord in coords.items() if abs(float(coord[axis_index])) <= tol]
    if zero_nodes:
        node_ids = zero_nodes
        plane_name = f"global_{mode}0"
    else:
        target = float(np.min(values))
        node_ids = [node_id for node_id, coord in coords.items() if abs(float(coord[axis_index]) - target) <= tol]
        plane_name = f"global_min_{mode}"
    return [{"name": f"symmetry_{plane_name}", "node_ids": sorted(node_ids), "constraints": constraints}]


def _enforced_displacement_supports(
    nodes: list[dict[str, object]],
    config: LightweightFEMConfig,
    plot_type: str,
    exclude_node_ids: set[int] | None = None,
) -> list[dict[str, object]]:
    try:
        displacement = float(config.enforced_displacement_m)
    except (TypeError, ValueError):
        return []
    if abs(displacement) <= 0.0:
        return []
    exclude_node_ids = set(exclude_node_ids or set())
    coords = _node_lookup(nodes)
    if not coords:
        return []
    if plot_type == "cylinder":
        z_values = np.asarray([coord[2] for coord in coords.values()], dtype=float)
        target_z = 0.5 * (float(np.min(z_values)) + float(np.max(z_values)))
        tol = max((float(np.max(z_values)) - float(np.min(z_values))) * 1.0e-8, 1.0e-8)
        closest_z = min((float(coord[2]) for coord in coords.values()), key=lambda value: abs(value - target_z))
        supports = []
        for node_id, coord in coords.items():
            if abs(float(coord[2]) - closest_z) > tol:
                continue
            radial = np.asarray([coord[0], coord[1]], dtype=float)
            norm = float(np.linalg.norm(radial))
            if norm <= 1.0e-12:
                continue
            unit = radial / norm
            supports.append(
                {
                    "name": f"enforced_radial_displacement_{node_id}",
                    "node_ids": [node_id],
                    "constraints": {"ux": displacement * float(unit[0]), "uy": displacement * float(unit[1])},
                }
            )
        return supports
    xs = np.asarray([coord[0] for coord in coords.values()], dtype=float)
    ys = np.asarray([coord[1] for coord in coords.values()], dtype=float)
    centre = np.asarray([0.5 * (float(np.min(xs)) + float(np.max(xs))), 0.5 * (float(np.min(ys)) + float(np.max(ys)))])
    candidates = [node_id for node_id in coords if node_id not in exclude_node_ids] or list(coords)
    node_id = min(candidates, key=lambda nid: float(np.linalg.norm(coords[nid][:2] - centre)))
    return [{"name": "enforced_panel_displacement", "node_ids": [node_id], "constraints": {"uz": displacement}}]


def _offset_beam_nodes_and_couplings(
    nodes: list[dict[str, object]],
    beams: list[dict[str, object]],
    config: LightweightFEMConfig,
    normal_at_node,
    start_node_id: int | None = None,
    start_coupling_id: int = 30_001,
    exclude_base_node_ids: set[int] | None = None,
) -> list[dict[str, object]]:
    node_coords = _node_lookup(nodes)
    next_node_id = int(start_node_id or (max(node_coords, default=0) + 1))
    next_coupling_id = int(start_coupling_id)
    offset_nodes: dict[tuple[int, str, float], int] = {}
    couplings: list[dict[str, object]] = []
    exclude_base_node_ids = set(exclude_base_node_ids or set())

    def eccentricity_for(beam: dict[str, object]) -> float:
        section = beam.get("section") or beam.get("cross_section") or {}
        if isinstance(section, dict):
            try:
                return float(section.get("eccentricity_m", 0.0))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def offset_node(base_node_id: int, role: str, eccentricity: float) -> int:
        nonlocal next_node_id, next_coupling_id
        key = (int(base_node_id), str(role), round(float(eccentricity), 12))
        if key in offset_nodes:
            return offset_nodes[key]
        base = node_coords[int(base_node_id)]
        normal = np.asarray(normal_at_node(int(base_node_id), base), dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-12:
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            normal = normal / norm
        offset = normal * float(eccentricity)
        node_id = next_node_id
        next_node_id += 1
        offset_nodes[key] = node_id
        node_coords[node_id] = base + offset
        nodes.append({"id": node_id, "coords": node_coords[node_id].tolist()})
        couplings.append(
            {
                "id": next_coupling_id,
                "beam_node_id": node_id,
                "shell_node_ids": [int(base_node_id)],
                "shape_weights": [1.0],
                "eccentricity": offset.tolist(),
            }
        )
        next_coupling_id += 1
        return node_id

    for beam in beams:
        eccentricity = eccentricity_for(beam)
        if abs(eccentricity) <= 0.0:
            continue
        role = str(beam.get("role", "beam"))
        beam["node_ids"] = [
            int(node_id) if int(node_id) in exclude_base_node_ids else offset_node(int(node_id), role, eccentricity)
            for node_id in beam.get("node_ids", [])
        ]
    return couplings


def _section_with_runtime_options(
    section: object,
    thickness: float,
    reference: float,
    depth_factor: float,
    eccentricity: float,
    orientation: str,
) -> dict[str, float]:
    result = dict(_section_or_default(section, thickness, reference, depth_factor))
    try:
        eccentricity = float(eccentricity)
    except (TypeError, ValueError):
        eccentricity = 0.0
    if abs(eccentricity) > 0.0:
        result["eccentricity_m"] = eccentricity
    orientation_key = _normalized_choice(orientation)
    if orientation_key == "global z":
        result["orientation"] = (0.0, 0.0, 1.0)
    elif orientation_key == "global y":
        result["orientation"] = (0.0, 1.0, 0.0)
    return result


def _support_choice_from_any(value: object) -> str:
    text = _normalized_choice(value, "simply supported")
    if text in {"c", "cl", "clamped", "fixed", "continuous"}:
        return "fixed"
    if text in {"s", "ss", "simple", "simply", "simply supported", "sniped"}:
        return "simply supported"
    if text in {"free", "none", "off"}:
        return "free"
    return "simply supported"


def _normalize_plate_edge_supports(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        return {key: _support_choice_from_any(value.get(key, "simply supported")) for key in ("x0", "x1", "y0", "y1")}
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return {key: _support_choice_from_any(value[index]) for index, key in enumerate(("x0", "x1", "y0", "y1"))}
    return {key: "simply supported" for key in ("x0", "x1", "y0", "y1")}


def _flat_edge_node_ids(node_id, rows: int, cols: int) -> dict[str, list[int]]:
    return {
        "x0": [node_id(0, col) for col in range(cols)],
        "x1": [node_id(rows - 1, col) for col in range(cols)],
        "y0": [node_id(row, 0) for row in range(rows)],
        "y1": [node_id(row, cols - 1) for row in range(rows)],
    }


def _flat_edge_supports(edge_nodes: dict[str, list[int]], choices: dict[str, object], node_id, rows: int) -> list[dict[str, object]]:
    supports: list[dict[str, object]] = []
    has_inplane_restraint = False
    for edge_name in ("x0", "x1", "y0", "y1"):
        choice = choices.get(edge_name, "simply supported")
        constraints = _support_constraints(choice, "flat")
        if not constraints:
            continue
        has_inplane_restraint = has_inplane_restraint or any(key in constraints for key in ("ux", "uy"))
        supports.append(
            {
                "name": "plate_" + edge_name + "_" + _normalized_choice(choice, "simply supported").replace(" ", "_"),
                "node_ids": sorted(set(int(node) for node in edge_nodes[edge_name])),
                "constraints": constraints,
            }
        )
    if supports and not has_inplane_restraint:
        supports.extend(
            [
                {"name": "simple_panel_inplane_anchor", "node_ids": [node_id(0, 0)], "constraints": {"ux": 0.0, "uy": 0.0}},
                {"name": "simple_panel_spin_anchor", "node_ids": [node_id(rows - 1, 0)], "constraints": {"uy": 0.0}},
            ]
        )
    return supports


def _flat_supports(
    boundary_nodes: list[int],
    node_id,
    rows: int,
    cols: int,
    config: LightweightFEMConfig,
    geometry: dict | None = None,
) -> list[dict[str, object]]:
    mode = _normalized_choice(config.boundary_condition)
    if mode in {"auto", "free", "none", "nullspace", "nullspace projection"}:
        choices = _normalize_plate_edge_supports((geometry or {}).get("plate_edge_supports"))
        return _flat_edge_supports(_flat_edge_node_ids(node_id, rows, cols), choices, node_id, rows)
    if mode in {"simply supported", "simple", "ss"}:
        return [
            {"name": "simple_panel_boundary", "node_ids": boundary_nodes, "constraints": {"uz": 0.0}},
            {"name": "simple_panel_inplane_anchor", "node_ids": [node_id(0, 0)], "constraints": {"ux": 0.0, "uy": 0.0}},
            {"name": "simple_panel_spin_anchor", "node_ids": [node_id(rows - 1, 0)], "constraints": {"uy": 0.0}},
        ]
    if mode in {"pinned", "pinned edges"}:
        return [
            {
                "name": "pinned_panel_boundary",
                "node_ids": boundary_nodes,
                "constraints": {"ux": 0.0, "uy": 0.0, "uz": 0.0},
            }
        ]
    return [
        {
            "name": "clamped_panel_boundary",
            "node_ids": boundary_nodes,
            "constraints": {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        }
    ]


def _support_constraints(choice: object, geometry: str = "flat") -> dict[str, float]:
    mode = _normalized_choice(choice, "free")
    if mode in {"free", "none", "off", "nullspace", "nullspace projection"}:
        return {}
    if mode in {"simple", "simply", "simply supported", "ss"}:
        return {"uz": 0.0}
    if geometry == "cylinder" and mode in {"fixed", "clamped"}:
        return {"ux": 0.0, "uy": 0.0, "uz": 0.0}
    if mode in {"fixed", "clamped"}:
        return {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}
    return {}


def _custom_flat_supports(node_id, rows: int, cols: int, config: LightweightFEMConfig) -> list[dict[str, object]]:
    if config.custom_use_nullspace_projection:
        return []
    edges = (
        ("x0", [node_id(0, col) for col in range(cols)], config.plate_edge_x0_support),
        ("x1", [node_id(rows - 1, col) for col in range(cols)], config.plate_edge_x1_support),
        ("y0", [node_id(row, 0) for row in range(rows)], config.plate_edge_y0_support),
        ("y1", [node_id(row, cols - 1) for row in range(rows)], config.plate_edge_y1_support),
    )
    supports = []
    for edge_name, node_ids, choice in edges:
        constraints = _support_constraints(choice, "flat")
        if constraints:
            supports.append(
                {
                    "name": "custom_plate_" + edge_name + "_" + _normalized_choice(choice, "free").replace(" ", "_"),
                    "node_ids": sorted(set(int(node) for node in node_ids)),
                    "constraints": constraints,
                }
            )
    return supports


def _cylinder_supports(rows: int, cols: int, node_id, config: LightweightFEMConfig) -> list[dict[str, object]]:
    mode = _normalized_choice(config.boundary_condition)
    if mode in {"free", "none", "nullspace", "nullspace projection"}:
        return []
    if mode in {"clamped", "fixed", "fixed ends"}:
        bottom = [node_id(0, col) for col in range(cols)]
        top = [node_id(rows - 1, col) for col in range(cols)]
        return [
            {
                "name": "clamped_cylinder_ends",
                "node_ids": bottom + top,
                "constraints": {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            }
        ]
    return [
        {"name": "rigid_body_anchor", "node_ids": [node_id(0, 0)], "constraints": {"ux": 0.0, "uy": 0.0, "uz": 0.0}},
        {"name": "rigid_body_spin_anchor", "node_ids": [node_id(0, cols // 4)], "constraints": {"ux": 0.0}},
        {"name": "rigid_body_tilt_anchor", "node_ids": [node_id(1, 0)], "constraints": {"uy": 0.0}},
    ]


def _custom_cylinder_supports(lower_ring: list[int], upper_ring: list[int], config: LightweightFEMConfig) -> list[dict[str, object]]:
    if config.custom_use_nullspace_projection:
        return []
    supports = []
    for name, ring_nodes, choice in (
        ("lower", lower_ring, config.cylinder_lower_support),
        ("upper", upper_ring, config.cylinder_upper_support),
    ):
        constraints = _support_constraints(choice, "cylinder")
        if constraints:
            supports.append(
                {
                    "name": "custom_cylinder_" + name + "_" + _normalized_choice(choice, "free").replace(" ", "_"),
                    "node_ids": sorted(set(int(node) for node in ring_nodes)),
                    "constraints": constraints,
                }
            )
    return supports


def _flat_generated_geometry(geometry: dict, config: LightweightFEMConfig) -> dict[str, object]:
    orientation = config.member_orientation
    if _normalized_choice(orientation) == "auto":
        orientation = "global z"
    length = _positive(geometry.get("length_m", 1.0), 1.0)
    width = _positive(geometry.get("width_m", 1.0), 1.0)
    thickness = _positive(geometry.get("thickness_m", 0.01), 0.01)
    base_div = _production_divisions(config.mesh_fidelity)
    stiffener_spacing = _positive_spacing(geometry.get("stiffener_spacing_m", 0.0))
    girder_spacing = _positive_spacing(geometry.get("girder_spacing_m", 0.0))
    active_stiffener_spacing = (
        stiffener_spacing
        if config.include_stiffeners and geometry.get("has_stiffener")
        else 0.0
    )
    active_girder_spacing = (
        girder_spacing
        if config.include_girders and geometry.get("has_girder")
        else 0.0
    )
    member_spacing_cap = min(
        [value for value in (active_stiffener_spacing, active_girder_spacing) if value > 0.0],
        default=0.0,
    )
    stiffener_positions = (
        _centered_member_positions(width, stiffener_spacing, fallback_midpoint=True)
        if config.include_stiffeners and geometry.get("has_stiffener")
        else ()
    )
    girder_positions = (
        _centered_member_positions(length, girder_spacing, fallback_midpoint=True)
        if config.include_girders and geometry.get("has_girder")
        else ()
    )
    div_x = _line_divisions(length, config, base_div, member_spacing_cap)
    div_y = _line_divisions(width, config, base_div, member_spacing_cap)
    x_breaks = _axis_breaks(length, div_x, girder_positions, member_spacing_cap)
    y_breaks = _axis_breaks(width, div_y, stiffener_positions, member_spacing_cap)
    rows = len(x_breaks)
    cols = len(y_breaks)

    def node_id(row: int, col: int) -> int:
        return 1 + row * cols + col

    nodes = [
        {
            "id": node_id(row, col),
            "coords": [x_breaks[row], y_breaks[col], 0.0],
        }
        for row in range(rows)
        for col in range(cols)
    ]
    shells = []
    element_id = 1
    for row in range(rows - 1):
        for col in range(cols - 1):
            shells.append(
                {
                    "id": element_id,
                    "node_ids": [
                        node_id(row, col),
                        node_id(row + 1, col),
                        node_id(row + 1, col + 1),
                        node_id(row, col + 1),
                    ],
                    "thickness": thickness,
                    "material": "steel",
                }
            )
            element_id += 1

    beams = []
    beam_id = 20_001
    for stiffener_y in stiffener_positions:
        mid_col = _index_of_break(y_breaks, stiffener_y)
        section = _section_with_runtime_options(
            geometry.get("stiffener_section"),
            thickness,
            width,
            0.08,
            config.stiffener_eccentricity_m,
            orientation,
        )
        for row in range(rows - 1):
            beams.append(
                {
                    "id": beam_id,
                    "node_ids": [node_id(row, mid_col), node_id(row + 1, mid_col)],
                    "section": section,
                    "role": "stiffener",
                    "material": "steel",
                }
            )
            beam_id += 1
    for girder_x in girder_positions:
        mid_row = _index_of_break(x_breaks, girder_x)
        section = _section_with_runtime_options(
            geometry.get("girder_section"),
            thickness,
            length,
            0.10,
            config.girder_eccentricity_m,
            orientation,
        )
        for col in range(cols - 1):
            beams.append(
                {
                    "id": beam_id,
                    "node_ids": [node_id(mid_row, col), node_id(mid_row, col + 1)],
                    "section": section,
                    "role": "girder",
                    "material": "steel",
                }
            )
            beam_id += 1

    if _wants_s8(config):
        _upgrade_shells_to_s8(nodes, shells)

    couplings = _offset_beam_nodes_and_couplings(
        nodes,
        beams,
        config,
        lambda _node_id, _coord: np.array([0.0, 0.0, 1.0], dtype=float),
    )
    boundary_nodes = sorted(
        {
            node_id(row, col)
            for row in range(rows)
            for col in range(cols)
            if row in (0, rows - 1) or col in (0, cols - 1)
        }
    )
    if config.custom_load_bc_enabled:
        supports = _custom_flat_supports(node_id, rows, cols, config)
    else:
        supports = _flat_supports(boundary_nodes, node_id, rows, cols, config, geometry)
        supports.extend(_symmetry_supports(nodes, config))
        supports.extend(_enforced_displacement_supports(nodes, config, "flat", exclude_node_ids=set(boundary_nodes)))
    return {
        "name": "ANYstructureFlatPanelFullMesh",
        "nodes": nodes,
        "shells": shells,
        "beams": beams,
        "couplings": couplings,
        "supports": supports,
        "materials": [
            {
                "name": "steel",
                "elastic_modulus": config.elastic_modulus_pa,
                "poisson_ratio": config.poisson_ratio,
                "density": 7850.0,
                "yield_stress": config.yield_stress_pa,
            }
        ],
        "plot_grid": [[node_id(row, col) for col in range(cols)] for row in range(rows)],
        "plot_type": "flat",
    }


def _cylinder_generated_geometry(geometry: dict, config: LightweightFEMConfig) -> dict[str, object]:
    orientation = config.member_orientation
    if _normalized_choice(orientation) == "auto":
        orientation = "radial"
    radius = _positive(geometry.get("radius_m", 1.0), 1.0)
    length = _positive(geometry.get("length_m", 1.0), 1.0)
    thickness = _positive(geometry.get("thickness_m", 0.01), 0.01)
    circumference = 2.0 * math.pi * radius
    stiffener_spacing = _positive_spacing(geometry.get("stiffener_spacing_m", 0.0))
    active_stiffener_spacing = (
        stiffener_spacing
        if config.include_stiffeners and geometry.get("has_stiffener")
        else 0.0
    )
    stiffener_count = (
        _member_count_from_spacing(circumference, stiffener_spacing)
        if config.include_stiffeners and geometry.get("has_stiffener")
        else 0
    )
    girder_spacing = (
        _positive_spacing(geometry.get("girder_spacing_m", 0.0))
        if config.include_girders and geometry.get("has_girder")
        else 0.0
    )
    mesh_size = _requested_mesh_size(config)
    mesh_size_cap = min(
        [value for value in (active_stiffener_spacing, girder_spacing) if value > 0.0],
        default=0.0,
    )
    if mesh_size > 0.0:
        if mesh_size_cap > 0.0 and mesh_size > mesh_size_cap:
            mesh_size = mesh_size_cap
        circumferential_div = max(int(math.ceil(circumference / mesh_size)), 8)
        axial_div = max(int(math.ceil(length / mesh_size)), 2)
    else:
        base_div = _production_divisions(config.mesh_fidelity)
        circumferential_div = max(base_div * 2, 8)
        axial_div = max(int(length / max(radius, 1.0e-9) * circumferential_div / 4), 2)
        if mesh_size_cap > 0.0:
            target_size = mesh_size_cap / max(_fidelity_refinement(config.mesh_fidelity), 1)
            circumferential_div = max(circumferential_div, int(math.ceil(circumference / target_size)))
            axial_div = max(axial_div, int(math.ceil(length / target_size)))
    if stiffener_count > 0:
        circumferential_div = _multiple_at_least(circumferential_div, stiffener_count)
    girder_positions = []
    if config.include_girders and geometry.get("has_girder"):
        if girder_spacing > 1.0e-9:
            pos = girder_spacing
            while pos < length - 1.0e-9 and len(girder_positions) < 100:
                girder_positions.append(pos)
                pos += girder_spacing
        else:
            girder_positions = [length / 2.0]
    z_breaks = _axis_breaks(length, axial_div, tuple(girder_positions))
    rows = len(z_breaks)
    axial_div = rows - 1
    cols = circumferential_div

    def node_id(row: int, col: int) -> int:
        return 1 + row * cols + (col % cols)

    nodes = []
    for row in range(rows):
        z = z_breaks[row]
        for col in range(cols):
            theta = 2.0 * math.pi * col / cols
            nodes.append({"id": node_id(row, col), "coords": [radius * math.cos(theta), radius * math.sin(theta), z]})

    shells = []
    element_id = 1
    for row in range(axial_div):
        for col in range(cols):
            next_col = (col + 1) % cols
            shells.append(
                {
                    "id": element_id,
                    "node_ids": [
                        node_id(row, col),
                        node_id(row, next_col),
                        node_id(row + 1, next_col),
                        node_id(row + 1, col),
                    ],
                    "thickness": thickness,
                    "material": "steel",
                }
            )
            element_id += 1

    beams = []
    beam_id = 20_001
    if config.include_stiffeners and geometry.get("has_stiffener"):
        base_section = _section_with_runtime_options(
            geometry.get("stiffener_section"),
            thickness,
            radius,
            0.08,
            config.stiffener_eccentricity_m,
            orientation,
        )
        count = stiffener_count if stiffener_count > 0 else min(8, cols)
        for offset in range(count):
            col = int(round(offset * cols / count)) % cols
            section = dict(base_section)
            if _normalized_choice(orientation) == "radial":
                theta = 2.0 * math.pi * col / cols
                section["orientation"] = (math.cos(theta), math.sin(theta), 0.0)
            for row in range(axial_div):
                beams.append(
                    {
                        "id": beam_id,
                        "node_ids": [node_id(row, col), node_id(row + 1, col)],
                        "section": section,
                        "role": "stiffener",
                        "material": "steel",
                    }
                )
                beam_id += 1
    if config.include_girders and geometry.get("has_girder"):
        base_section = _section_with_runtime_options(
            geometry.get("girder_section"),
            thickness,
            radius,
            0.12,
            config.girder_eccentricity_m,
            orientation,
        )
        ring_rows = [_index_of_break(z_breaks, pos) for pos in girder_positions] or [rows // 2]
        for row in ring_rows:
            for col in range(cols):
                section = dict(base_section)
                if _normalized_choice(orientation) == "radial":
                    theta = 2.0 * math.pi * (col + 0.5) / cols
                    section["orientation"] = (math.cos(theta), math.sin(theta), 0.0)
                beams.append(
                    {
                        "id": beam_id,
                        "node_ids": [node_id(row, col), node_id(row, col + 1)],
                        "section": section,
                        "role": "girder",
                        "material": "steel",
                    }
                )
                beam_id += 1

    if _wants_s8(config):
        _upgrade_shells_to_s8(nodes, shells, radius=radius)

    def cylinder_normal(_node_id: int, coord: np.ndarray) -> np.ndarray:
        radial = np.asarray([coord[0], coord[1], 0.0], dtype=float)
        norm = float(np.linalg.norm(radial))
        if norm <= 1.0e-12:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return radial / norm

    node_coords_after_shell_order = _node_lookup(nodes)
    z_tol = max(length * 1.0e-9, 1.0e-9)
    start_ring = sorted(
        node_id_value
        for node_id_value, coord in node_coords_after_shell_order.items()
        if abs(float(coord[2])) <= z_tol
    )
    end_ring = sorted(
        node_id_value
        for node_id_value, coord in node_coords_after_shell_order.items()
        if abs(float(coord[2]) - length) <= z_tol
    )
    rigid_lids = []
    supports = _cylinder_supports(rows, cols, node_id, config)
    custom_lid_support_nodes: tuple[list[int], list[int]] | None = None
    if config.include_end_lids:
        next_node_id = max(_node_lookup(nodes), default=0) + 1
        bottom_center = next_node_id
        top_center = bottom_center + 1
        nodes.extend(
            [
                {"id": bottom_center, "coords": [0.0, 0.0, 0.0]},
                {"id": top_center, "coords": [0.0, 0.0, length]},
            ]
        )
        rigid_lids = [
            {"id": 40_001, "name": "bottom_rigid_lid", "center_node_id": bottom_center, "ring_node_ids": start_ring},
            {"id": 40_002, "name": "top_rigid_lid", "center_node_id": top_center, "ring_node_ids": end_ring},
        ]
        supports = []
        custom_lid_support_nodes = ([bottom_center], [top_center])
    rigid_lid_ring_nodes = set(start_ring + end_ring) if config.include_end_lids else set()
    couplings = _offset_beam_nodes_and_couplings(
        nodes,
        beams,
        config,
        cylinder_normal,
        start_node_id=max(_node_lookup(nodes), default=0) + 1,
        exclude_base_node_ids=rigid_lid_ring_nodes,
    )
    if config.custom_load_bc_enabled:
        if custom_lid_support_nodes is not None:
            supports = _custom_cylinder_supports(custom_lid_support_nodes[0], custom_lid_support_nodes[1], config)
        else:
            supports = _custom_cylinder_supports(start_ring, end_ring, config)
    else:
        supports.extend(_symmetry_supports(nodes, config))
        supports.extend(_enforced_displacement_supports(nodes, config, "cylinder"))
    return {
        "name": "ANYstructureCylinderFullMesh",
        "nodes": nodes,
        "shells": shells,
        "beams": beams,
        "couplings": couplings,
        "rigid_lids": rigid_lids,
        "supports": supports,
        "materials": [
            {
                "name": "steel",
                "elastic_modulus": config.elastic_modulus_pa,
                "poisson_ratio": config.poisson_ratio,
                "density": 7850.0,
                "yield_stress": config.yield_stress_pa,
            }
        ],
        "plot_grid": [[node_id(row, col) for col in range(cols)] + [node_id(row, 0)] for row in range(rows)],
        "plot_type": "cylinder",
        "radius_m": radius,
        "bottom_ring_node_ids": start_ring,
        "top_ring_node_ids": end_ring,
    }


def build_generated_geometry(geometry: dict, config: LightweightFEMConfig) -> dict[str, object]:
    """Build the deterministic full shell/beam mesh consumed by the FE backend."""

    if geometry.get("geometry") == "cylinder":
        return _cylinder_generated_geometry(geometry, config)
    return _flat_generated_geometry(geometry, config)


def _nodal_scalar_fields(model, stresses_by_element: dict[int, object]) -> dict[str, dict[int, float]]:
    if not stresses_by_element:
        return {}
    
    field_mapping = {
        "von_mises_pa": ("von_mises", "von_mises"),
        "stress_x_membrane_pa": ("membrane_xx", "axial_stress"),
        "stress_y_membrane_pa": ("membrane_yy", None),
        "stress_xy_membrane_pa": ("membrane_xy", None),
        "strain_x_membrane": ("membrane_strain_xx", "axial_strain"),
        "strain_y_membrane": ("membrane_strain_yy", None),
        "strain_xy_membrane": ("membrane_strain_xy", None),
    }

    sums = {k: collections.defaultdict(float) for k in field_mapping}
    counts = {k: collections.defaultdict(int) for k in field_mapping}

    for element_id, stress in stresses_by_element.items():
        element = model.mesh.elements.get(element_id)
        if element is None:
            continue
            
        is_beam = type(element).__name__ == "BeamElement"
        node_ids = element.node_ids
        
        for field_name, (shell_key, beam_key) in field_mapping.items():
            key = beam_key if is_beam else shell_key
            if key is None or key not in stress:
                continue
                
            val = stress[key]
            if isinstance(val, (list, tuple)):
                value = sum(val) / len(val)
            elif hasattr(val, "size") and val.size > 0:
                value = float(val.sum()) / val.size
            else:
                value = float(val)
            
            s_dict = sums[field_name]
            c_dict = counts[field_name]
            for node_id in node_ids:
                nid = int(node_id)
                s_dict[nid] += value
                c_dict[nid] += 1
                
    result = {}
    for field_name, s_dict in sums.items():
        c_dict = counts[field_name]
        if s_dict:
            result[field_name] = {nid: total / c_dict[nid] for nid, total in s_dict.items()}
    return result


def _nodal_engineering_plastic_strain(model, element_states: dict[int, object] | None) -> dict[int, float]:
    if not element_states:
        return {}
    values: dict[int, float] = {}
    for element_id, state in element_states.items():
        element = model.mesh.get_element(int(element_id))
        if element is None or not isinstance(state, dict) or "alpha" not in state:
            continue
        alpha = np.asarray(state.get("alpha"), dtype=float)
        alpha = alpha[np.isfinite(alpha)]
        if alpha.size == 0:
            continue
        # The material return-map stores equivalent true plastic strain.  Use
        # the equivalent engineering strain for display so it is easier to
        # compare with ordinary engineering strain values in the GUI.
        engineering_value = float(np.expm1(max(float(np.max(alpha)), 0.0)))
        for node_id in getattr(element, "node_ids", []):
            node_id = int(node_id)
            values[node_id] = max(values.get(node_id, 0.0), engineering_value)
    return values


def _visualization_member_lines(
    generated_geometry: dict,
    model,
    displacements: np.ndarray,
    stresses_by_element: dict[int, object] | None = None,
) -> tuple[dict[str, object], ...]:
    lines: list[dict[str, object]] = []
    if displacements is None:
        return ()
        
    stresses = stresses_by_element or {}
    
    for beam in generated_geometry.get("beams", []) or []:
        node_ids = [int(node_id) for node_id in beam.get("node_ids", [])]
        if len(node_ids) < 2:
            continue
        points = []
        displaced = []
        for node_id in node_ids[:2]:
            node = model.mesh.get_node(node_id)
            if node is None:
                break
            base = np.asarray(node.coords(), dtype=float)
            translation = np.asarray(displacements[node.dofs[:3]], dtype=float)
            points.append(tuple(float(value) for value in base))
            displaced.append(tuple(float(value) for value in base + translation))
        if len(points) != 2:
            continue

        beam_stresses = {}
        c_y = 0.0
        c_z = 0.0
        element_id = beam.get("id")
        if element_id is not None:
            try:
                element = model.mesh.get_element(int(element_id))
                if element is not None:
                    beam_stresses = stresses.get(int(element_id)) or {}
                    c_y, c_z = element._fiber_distances()
            except Exception:
                pass

        lines.append(
            {
                "id": int(beam.get("id", 0)),
                "role": str(beam.get("role", "member")),
                "node_ids": tuple(node_ids[:2]),
                "points": tuple(points),
                "displaced_points": tuple(displaced),
                "section_label": str((beam.get("section") or {}).get("label", "")),
                # Include cross section dimensions
                "web_height": float((beam.get("section") or {}).get("web_height") or 0.1),
                "web_thickness": float((beam.get("section") or {}).get("web_thickness") or 0.01),
                "flange_width": float((beam.get("section") or {}).get("flange_width") or 0.0),
                "flange_thickness": float((beam.get("section") or {}).get("flange_thickness") or 0.0),
                "c_y": float(c_y),
                "c_z": float(c_z),
                "eccentricity": float((beam.get("section") or {}).get("eccentricity_m") or 0.0),
                # Include stress component results
                "axial_stress": float(beam_stresses.get("axial_stress", 0.0) if hasattr(beam_stresses.get("axial_stress"), "real") else (beam_stresses.get("axial_stress", [0.0])[0] if beam_stresses.get("axial_stress") is not None else 0.0)),
                "bending_stress_y": float(beam_stresses.get("bending_stress_y", 0.0) if hasattr(beam_stresses.get("bending_stress_y"), "real") else (beam_stresses.get("bending_stress_y", [0.0])[0] if beam_stresses.get("bending_stress_y") is not None else 0.0)),
                "bending_stress_z": float(beam_stresses.get("bending_stress_z", 0.0) if hasattr(beam_stresses.get("bending_stress_z"), "real") else (beam_stresses.get("bending_stress_z", [0.0])[0] if beam_stresses.get("bending_stress_z") is not None else 0.0)),
                "shear_stress_y": float(beam_stresses.get("shear_stress_y", 0.0) if hasattr(beam_stresses.get("shear_stress_y"), "real") else (beam_stresses.get("shear_stress_y", [0.0])[0] if beam_stresses.get("shear_stress_y") is not None else 0.0)),
                "shear_stress_z": float(beam_stresses.get("shear_stress_z", 0.0) if hasattr(beam_stresses.get("shear_stress_z"), "real") else (beam_stresses.get("shear_stress_z", [0.0])[0] if beam_stresses.get("shear_stress_z") is not None else 0.0)),
                "torsional_stress": float(beam_stresses.get("torsional_stress", 0.0) if hasattr(beam_stresses.get("torsional_stress"), "real") else (beam_stresses.get("torsional_stress", [0.0])[0] if beam_stresses.get("torsional_stress") is not None else 0.0)),
                "von_mises": float(beam_stresses.get("von_mises", 0.0) if hasattr(beam_stresses.get("von_mises"), "real") else (beam_stresses.get("von_mises", [0.0])[0] if beam_stresses.get("von_mises") is not None else 0.0)),
            }
        )
    return tuple(lines)


def _visualization_from_full_result(
    generated_geometry: dict,
    model,
    displacements: np.ndarray,
    scalar_by_node: dict[int, float] | None = None,
    scalar_label: str = "stress [Pa]",
) -> dict[str, object]:
    grid = generated_geometry.get("plot_grid") or []
    if not grid or displacements is None:
        return {}

    stresses_by_element = {}
    if _backend_compute_stresses is not None:
        stresses_by_element = _backend_compute_stresses(model, displacements)

    fields = _nodal_scalar_fields(model, stresses_by_element)
    if scalar_by_node is not None:
        fields["custom_scalar"] = scalar_by_node

    is_cylinder = generated_geometry.get("plot_type") == "cylinder"
    radius = _positive(generated_geometry.get("radius_m", 1.0), 1.0) if is_cylinder else 0.0

    x_grid, y_grid, w_grid = [], [], []
    disp_grids = {"disp_x": [], "disp_y": [], "disp_z": [], "disp_mag": []}
    field_grids = {k: [] for k in fields}
    
    get_node = model.mesh.get_node

    if is_cylinder:
        for row in grid:
            x_row, y_row, w_row = [], [], []
            dx_row, dy_row, dz_row, dmag_row = [], [], [], []
            f_rows = {k: [] for k in fields}
            
            for node_id in row:
                nid = int(node_id)
                node = get_node(nid)
                if node is None:
                    continue
                
                dofs = node.dofs
                dx = float(displacements[dofs[0]])
                dy = float(displacements[dofs[1]])
                dz = float(displacements[dofs[2]])
                dmag = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                nx, ny, nz = float(node.x), float(node.y), float(node.z)
                theta = math.atan2(ny, nx)
                rad_disp = dx * math.cos(theta) + dy * math.sin(theta)
                
                x_row.append(nz)
                y_row.append(theta if theta >= 0.0 else theta + 2.0 * math.pi)
                w_row.append(rad_disp)
                
                dx_row.append(dx)
                dy_row.append(dy)
                dz_row.append(dz)
                dmag_row.append(dmag)
                
                for k, field_dict in fields.items():
                    f_rows[k].append(float(field_dict.get(nid, abs(rad_disp))))
                    
            x_grid.append(tuple(x_row))
            y_grid.append(tuple(y_row))
            w_grid.append(tuple(w_row))
            disp_grids["disp_x"].append(tuple(dx_row))
            disp_grids["disp_y"].append(tuple(dy_row))
            disp_grids["disp_z"].append(tuple(dz_row))
            disp_grids["disp_mag"].append(tuple(dmag_row))
            for k, row_list in f_rows.items():
                field_grids[k].append(tuple(row_list))
    else:
        for row in grid:
            x_row, y_row, w_row = [], [], []
            dx_row, dy_row, dz_row, dmag_row = [], [], [], []
            f_rows = {k: [] for k in fields}
            
            for node_id in row:
                nid = int(node_id)
                node = get_node(nid)
                if node is None:
                    continue
                
                dofs = node.dofs
                dx = float(displacements[dofs[0]])
                dy = float(displacements[dofs[1]])
                dz = float(displacements[dofs[2]])
                dmag = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                x_row.append(float(node.x))
                y_row.append(float(node.y))
                w_row.append(dz)
                
                dx_row.append(dx)
                dy_row.append(dy)
                dz_row.append(dz)
                dmag_row.append(dmag)
                
                for k, field_dict in fields.items():
                    f_rows[k].append(float(field_dict.get(nid, abs(dz))))
                    
            x_grid.append(tuple(x_row))
            y_grid.append(tuple(y_row))
            w_grid.append(tuple(w_row))
            disp_grids["disp_x"].append(tuple(dx_row))
            disp_grids["disp_y"].append(tuple(dy_row))
            disp_grids["disp_z"].append(tuple(dz_row))
            disp_grids["disp_mag"].append(tuple(dmag_row))
            for k, row_list in f_rows.items():
                field_grids[k].append(tuple(row_list))

    result = {
        "type": "cylinder" if is_cylinder else "flat",
        "radius_m": radius,
        "x_m" if not is_cylinder else "axial_m": tuple(x_grid),
        "y_m" if not is_cylinder else "theta_rad": tuple(y_grid),
        "w_m" if not is_cylinder else "radial_displacement_m": tuple(w_grid),
        "displacements": {k: tuple(v) for k, v in disp_grids.items()},
        "fields": {k: tuple(v) for k, v in field_grids.items()},
        "stress_pa": tuple(field_grids.get("custom_scalar", field_grids.get("von_mises_pa", w_grid))),
        "scalar_label": scalar_label,
        "member_lines": _visualization_member_lines(generated_geometry, model, displacements, stresses_by_element),
    }
    return result


def _buckling_mode_visualizations(generated_geometry: dict, model, buckling_result) -> tuple[dict[str, object], ...]:
    if buckling_result is None:
        return ()
    modes = []
    for mode in getattr(buckling_result, "modes", []) or []:
        shape = _visualization_from_full_result(
            generated_geometry,
            model,
            np.asarray(mode.mode_shape, dtype=float),
            scalar_by_node={},
            scalar_label="mode amplitude",
        )
        if not shape:
            continue
        shape["mode_number"] = int(mode.mode_number)
        shape["load_factor"] = float(mode.load_factor)
        modes.append(
            {
                "mode_number": int(mode.mode_number),
                "load_factor": float(mode.load_factor),
                "shape": shape,
            }
        )
    return tuple(modes)


def _resultant_dict(load_resultant) -> dict[str, tuple[float, float, float]]:
    if load_resultant is None:
        return {}
    return {
        "force_n": tuple(float(value) for value in np.asarray(load_resultant.force, dtype=float).reshape(3)),
        "moment_nm": tuple(float(value) for value in np.asarray(load_resultant.moment, dtype=float).reshape(3)),
    }


def _pressure_sign(config: LightweightFEMConfig) -> float:
    direction = _normalized_choice(config.pressure_direction, "external")
    return 1.0 if direction in {"internal", "outward", "positive normal"} else -1.0


def _solver_type(config: LightweightFEMConfig) -> str:
    solver = _normalized_choice(config.solver_type, "direct").replace(" ", "")
    return solver if solver in {"direct", "gmres", "minres", "bicgstab"} else "direct"


def _include_imported_loads(config: LightweightFEMConfig) -> bool:
    return (not config.custom_load_bc_enabled) or bool(config.custom_loads_add_to_imported)


def _effective_pressure_pa(config: LightweightFEMConfig) -> float:
    imported = float(config.pressure_pa or 0.0)
    custom = float(config.custom_pressure_pa or 0.0)
    if not config.custom_load_bc_enabled:
        return imported
    if config.custom_loads_add_to_imported:
        return imported + custom
    return custom


def _custom_has_fixed_support(config: LightweightFEMConfig) -> bool:
    choices = (
        config.plate_edge_x0_support,
        config.plate_edge_x1_support,
        config.plate_edge_y0_support,
        config.plate_edge_y1_support,
        config.cylinder_lower_support,
        config.cylinder_upper_support,
    )
    return any(_normalized_choice(choice, "free") in {"fixed", "clamped"} for choice in choices)


def _has_custom_support(config: LightweightFEMConfig) -> bool:
    choices = (
        config.plate_edge_x0_support,
        config.plate_edge_x1_support,
        config.plate_edge_y0_support,
        config.plate_edge_y1_support,
        config.cylinder_lower_support,
        config.cylinder_upper_support,
    )
    return any(_support_constraints(choice, "flat") for choice in choices)


def _constraint_mode(config: LightweightFEMConfig, geometry: dict | None = None) -> str:
    if config.custom_load_bc_enabled and config.custom_use_nullspace_projection:
        return "nullspace"
    if config.custom_load_bc_enabled and (_custom_has_fixed_support(config) or _has_custom_support(config)):
        return "transformation"
    if _normalized_choice(config.boundary_condition) in {"nullspace", "nullspace projection"}:
        return "nullspace"
    is_flat = (geometry or {}).get("geometry") != "cylinder"
    if is_flat and _normalized_choice(config.boundary_condition) in {"auto", "simply supported", "simple", "ss", "pinned", "pinned edges", "fixed", "clamped"}:
        return "transformation"
    return "auto"


def _allow_unbalanced_free_free(config: LightweightFEMConfig, geometry: dict | None = None) -> bool:
    if bool(config.allow_unbalanced_free_free):
        return True
    return _constraint_mode(config, geometry) == "nullspace"


def _buckling_load_factor_range(config: LightweightFEMConfig) -> tuple[float | None, float | None] | None:
    lower = float(config.buckling_min_load_factor or 0.0)
    upper = float(config.buckling_max_load_factor or 0.0)
    if lower <= 0.0 and upper <= 0.0:
        return None
    return (lower if lower > 0.0 else None, upper if upper > 0.0 else None)


def _buckling_solver_kwargs(config: LightweightFEMConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "repeated_tolerance": max(float(config.buckling_repeated_tolerance or 1.0e-3), 0.0),
        "allow_dense_fallback": bool(config.buckling_allow_dense_fallback),
    }
    shift = float(config.buckling_shift_load_factor or 0.0)
    if shift > 0.0:
        kwargs["shift_load_factor"] = shift
    load_range = _buckling_load_factor_range(config)
    if load_range is not None:
        kwargs["load_factor_range"] = load_range
    return kwargs


def _wants_capacity_workflow(config: LightweightFEMConfig) -> bool:
    choice = _normalized_choice(config.runtime_solver, "stepwise")
    return choice in {
        "anyintelligent capacity workflow",
        "capacity workflow",
        "nonlinear capacity workflow",
        "structured capacity workflow",
    }


def _recovery_config(config: LightweightFEMConfig):
    if _full_backend is None or not hasattr(_full_backend, "RecoveryConfig"):
        return None
    mode = _normalized_choice(config.recovery_history_mode, "full")
    if mode not in {"full", "selected", "envelope"}:
        mode = "full"
    return _full_backend.RecoveryConfig(history_mode=mode, store_full_histories=(mode == "full"))


def _resource_config(config: LightweightFEMConfig):
    if _full_backend is None or not hasattr(_full_backend, "ResourceConfig"):
        return None
    recovery_threads = int(config.recovery_threads or 0)
    memory_limit_mb = float(config.memory_limit_mb or 0.0)
    if recovery_threads <= 0 and memory_limit_mb <= 0.0:
        return None
    return _full_backend.ResourceConfig(
        recovery_threads=recovery_threads if recovery_threads > 0 else None,
        memory_limit_bytes=int(memory_limit_mb * 1024.0 * 1024.0) if memory_limit_mb > 0.0 else None,
        deterministic=True,
    )


def _wants_nonlinear_analysis(config: LightweightFEMConfig) -> bool:
    return _normalized_choice(config.analysis_type, "linear eigenvalue") not in {
        "linear",
        "linear eigenvalue",
        "linear static eigenvalue",
        "linear static + eigenvalue",
    }


def _wants_nonlinear_buckling(config: LightweightFEMConfig) -> bool:
    return _normalized_choice(config.buckling_analysis_type, "linear eigenvalue") not in {
        "linear eigenvalue",
        "eigenvalue",
    }


def _wants_eigenvalue_buckling(config: LightweightFEMConfig) -> bool:
    choice = _normalized_choice(config.runtime_solver, "stepwise")
    return choice not in {"static only", "nonlinear static"}


def _wants_static_nonlinear_analysis(config: LightweightFEMConfig) -> bool:
    choice = _normalized_choice(config.analysis_type, "linear eigenvalue")
    runtime = _normalized_choice(config.runtime_solver, "stepwise")
    if runtime == "nonlinear static":
        return True
    return choice in {
        "geometric nonlinear static",
        "material nonlinear static",
        "geom. + material nonlinear static",
        "geom + material nonlinear static",
        "geometric and material nonlinear static",
    }


def _wants_material_nonlinear_analysis(config: LightweightFEMConfig) -> bool:
    choice = _normalized_choice(config.analysis_type, "linear eigenvalue")
    model = _normalized_choice(config.material_model, "linear elastic")
    return "material" in choice or model in {
        "dnv rp c208 steel",
        "dnv c208 steel",
        "dnv rp c208",
        "rp c208 steel",
    }


def _wants_tangent_stability_analysis(config: LightweightFEMConfig) -> bool:
    choice = _normalized_choice(config.analysis_type, "linear eigenvalue")
    buckling_choice = _normalized_choice(config.buckling_analysis_type, "linear eigenvalue")
    if choice == "nonlinear stability":
        return True
    return buckling_choice == "nonlinear limit" and not _wants_static_nonlinear_analysis(config)


def _positive_int(value: object, fallback: int, minimum: int = 1) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = fallback
    return max(number, minimum)


def _nonlinear_layer_count(value: object) -> int:
    requested = _positive_int(value, 5, 3)
    supported = (3, 5, 7, 9, 11)
    return min(supported, key=lambda item: abs(item - requested))


def _nonlinear_curve_payload(config: LightweightFEMConfig, geometry: dict) -> tuple[object | None, dict[str, float | str]]:
    if _backend_curve_from_properties is None:
        return None, {}
    if not _wants_material_nonlinear_analysis(config):
        return None, {}
    thickness = _positive(geometry.get("thickness_m", 0.0), 0.016)
    properties = dnv_c208_steel_properties(config.steel_grade, thickness, config.steel_thickness_class)
    curve_properties = {
        "sigma_prop": properties["sigma_prop"],
        "sigma_yield": properties["sigma_yield"],
        "sigma_yield_2": properties["sigma_yield_2"],
        "eps_p_y1": properties["eps_p_y1"],
        "eps_p_y2": properties["eps_p_y2"],
        "K": properties["K"],
        "n": properties["n"],
    }
    return _backend_curve_from_properties(curve_properties), properties


def _apply_material_curve_to_model(model, curve: object | None, properties: dict[str, float | str]) -> None:
    if curve is None:
        return
    for material in getattr(model, "materials", {}).values():
        material.hardening_curve = curve
        material.elastic_modulus = float(properties.get("E_pa", material.elastic_modulus))
        material.yield_stress = float(properties.get("sigma_yield", material.yield_stress))


def _shell_element_ids(model) -> tuple[int, ...]:
    ids: list[int] = []
    for element_id, element in getattr(model.mesh, "elements", {}).items():
        if hasattr(element, "thickness") and hasattr(element, "node_ids"):
            ids.append(int(element_id))
    return tuple(ids)


def _shell_node_ids(model) -> tuple[int, ...]:
    node_ids: set[int] = set()
    for element_id in _shell_element_ids(model):
        element = model.mesh.get_element(element_id)
        if element is not None:
            node_ids.update(int(node_id) for node_id in getattr(element, "node_ids", []))
    return tuple(sorted(node_ids))


def _build_runtime_imperfection(model, generated_geometry: dict, geometry: dict, config: LightweightFEMConfig):
    """Create a stress-free imperfection object for the synced backend."""

    if not config.imperfection_enabled or _full_backend is None:
        return None, {}
    shape = _normalized_choice(config.imperfection_shape, "standard plate/cylinder")
    if shape in {"none", "off", "disabled"}:
        return None, {}
    shell_ids = _shell_element_ids(model)
    if not shell_ids:
        return None, {"status": "no shell elements"}

    amplitude = float(config.imperfection_amplitude_m or 0.0)
    amplitude_value = amplitude if amplitude > 0.0 else None
    wave_a = _positive_int(config.imperfection_wave_a, 1)
    wave_b = _positive_int(config.imperfection_wave_b, 1)

    if generated_geometry.get("plot_type") != "cylinder":
        imperfection = _full_backend.StandardImperfection(
            kind="plate_mode",
            node_ids=shell_ids,
            amplitude=amplitude_value,
            direction=(0.0, 0.0, 1.0),
            axes=(0, 1),
            waves=(wave_a, wave_b),
            name="runtime_plate_half_wave",
        )
        metadata = {
            "kind": "plate half-wave",
            "amplitude_m": amplitude if amplitude > 0.0 else 0.0,
            "amplitude_source": "user" if amplitude > 0.0 else "standard s/200 default",
            "waves_a": wave_a,
            "waves_b": wave_b,
        }
        return imperfection, metadata

    node_ids = _shell_node_ids(model)
    coords_by_node = {}
    for node_id in node_ids:
        node = model.mesh.get_node(int(node_id))
        if node is not None:
            coords_by_node[int(node_id)] = np.asarray(node.coords(), dtype=float)
    if not coords_by_node:
        return None, {"status": "no shell nodes"}
    values = np.asarray(list(coords_by_node.values()), dtype=float)
    z_min = float(np.min(values[:, 2]))
    z_span = max(float(np.max(values[:, 2]) - z_min), 1.0e-12)
    radius = _positive(generated_geometry.get("radius_m", geometry.get("radius_m", 0.0)), 0.0)
    if amplitude <= 0.0:
        spacing = _positive(geometry.get("stiffener_spacing_m", 0.0), 0.0)
        if spacing <= 0.0 and radius > 0.0:
            spacing = 2.0 * math.pi * radius / max(len(set(round(math.atan2(coord[1], coord[0]), 12) for coord in coords_by_node.values())), 1)
        amplitude = max(spacing, z_span) / 200.0 if max(spacing, z_span) > 0.0 else 0.0
    offsets = {}
    for node_id, coord in coords_by_node.items():
        radial = np.array([coord[0], coord[1], 0.0], dtype=float)
        norm = float(np.linalg.norm(radial))
        if norm <= 1.0e-12:
            continue
        radial /= norm
        theta = math.atan2(float(coord[1]), float(coord[0]))
        axial_pos = (float(coord[2]) - z_min) / z_span
        shape_value = math.sin(wave_b * math.pi * axial_pos) * math.cos(wave_a * theta)
        offsets[node_id] = amplitude * shape_value * radial
    imperfection = _full_backend.ImperfectionField(
        offsets,
        name="runtime_cylinder_radial_imperfection",
        metadata={"kind": "cylinder radial", "waves": (wave_a, wave_b), "amplitude": amplitude},
    )
    metadata = {
        "kind": "cylinder radial",
        "amplitude_m": amplitude,
        "amplitude_source": "user" if float(config.imperfection_amplitude_m or 0.0) > 0.0 else "standard spacing/200 default",
        "waves_a": wave_a,
        "waves_b": wave_b,
    }
    return imperfection, metadata


def _apply_runtime_imperfection(model, generated_geometry: dict, geometry: dict, config: LightweightFEMConfig) -> dict[str, object]:
    if _full_backend is None or not hasattr(_full_backend, "apply_imperfection"):
        return {}
    imperfection, metadata = _build_runtime_imperfection(model, generated_geometry, geometry, config)
    if imperfection is None:
        return dict(metadata)
    _full_backend.apply_imperfection(model, imperfection, copy_model=False)
    records = getattr(model, "imperfection_metadata", []) or []
    if records:
        metadata["max_offset_m"] = float(records[-1].get("max_offset", 0.0) or 0.0)
    metadata["status"] = "applied"
    return metadata


def _element_centroid(model, element_id: int) -> np.ndarray | None:
    element = model.mesh.get_element(int(element_id))
    if element is None or not hasattr(element, "get_node_coordinates"):
        return None
    try:
        return np.mean(np.asarray(element.get_node_coordinates(model.mesh), dtype=float), axis=0)
    except Exception:
        return None


def _slamming_patch_element_ids(model, generated_geometry: dict, geometry: dict, config: LightweightFEMConfig) -> tuple[int, ...]:
    shell_ids = _shell_element_ids(model)
    if not shell_ids:
        return ()
    size_a = float(config.slamming_patch_size_a_m or 0.0)
    size_b = float(config.slamming_patch_size_b_m or 0.0)
    if size_a <= 0.0 or size_b <= 0.0:
        return shell_ids

    selected: list[int] = []
    if generated_geometry.get("plot_type") == "cylinder":
        radius = _positive(generated_geometry.get("radius_m", geometry.get("radius_m", 0.0)), 0.0)
        length = _positive(geometry.get("length_m", 0.0), 0.0)
        circumference = 2.0 * math.pi * radius if radius > 0.0 else 0.0
        center_z = float(config.slamming_patch_center_a_m or 0.0)
        if center_z <= 0.0 and length > 0.0:
            center_z = 0.5 * length
        center_arc = float(config.slamming_patch_center_b_m or 0.0)
        if center_arc < 0.0 and circumference > 0.0:
            center_arc = center_arc % circumference

        for element_id in shell_ids:
            centroid = _element_centroid(model, element_id)
            if centroid is None:
                continue
            z = float(centroid[2])
            theta = math.atan2(float(centroid[1]), float(centroid[0]))
            arc = (theta % (2.0 * math.pi)) * radius if radius > 0.0 else 0.0
            if circumference > 0.0:
                arc_delta = abs((arc - center_arc + 0.5 * circumference) % circumference - 0.5 * circumference)
            else:
                arc_delta = 0.0
            if abs(z - center_z) <= 0.5 * size_a + 1.0e-12 and arc_delta <= 0.5 * size_b + 1.0e-12:
                selected.append(int(element_id))
        return tuple(selected) or shell_ids

    nodes = {int(node["id"]): tuple(float(value) for value in node["coords"]) for node in generated_geometry.get("nodes", [])}
    xs = [coord[0] for coord in nodes.values()]
    ys = [coord[1] for coord in nodes.values()]
    default_x = 0.5 * (min(xs) + max(xs)) if xs else 0.0
    default_y = 0.5 * (min(ys) + max(ys)) if ys else 0.0
    center_x = float(config.slamming_patch_center_a_m or default_x)
    center_y = float(config.slamming_patch_center_b_m or default_y)
    for element_id in shell_ids:
        centroid = _element_centroid(model, element_id)
        if centroid is None:
            continue
        if abs(float(centroid[0]) - center_x) <= 0.5 * size_a + 1.0e-12 and abs(float(centroid[1]) - center_y) <= 0.5 * size_b + 1.0e-12:
            selected.append(int(element_id))
    return tuple(selected) or shell_ids


def _run_slamming_transient(model, load_case, generated_geometry: dict, geometry: dict, config: LightweightFEMConfig) -> dict[str, object]:
    if not config.slamming_enabled:
        return {}
    if _full_backend is None or _backend_solve_transient_newmark is None:
        return {"status": "unavailable"}
    pressure = abs(float(config.slamming_pressure_pa or 0.0))
    duration = max(float(config.slamming_duration_s or 0.0), 0.0)
    total_time = max(float(config.slamming_total_time_s or 0.0), duration)
    dt = max(float(config.slamming_dt_s or 0.0), 1.0e-9)
    if pressure <= 0.0 or duration <= 0.0 or total_time <= 0.0:
        return {"status": "skipped", "reason": "slamming pressure, duration and total time must be positive"}
    patch_ids = _slamming_patch_element_ids(model, generated_geometry, geometry, config)
    if not patch_ids:
        return {"status": "skipped", "reason": "no shell elements selected"}
    patch = _full_backend.PressurePatch.rectangular_pulse(
        name="runtime_slamming_patch",
        pressure=_pressure_sign(config) * pressure,
        start_time=0.0,
        end_time=duration,
        element_ids=patch_ids,
    )
    transient_config = _full_backend.TransientConfig(
        dt=dt,
        t_end=total_time,
        save_every=max(int(math.ceil(max(total_time / dt, 1.0) / 120.0)), 1),
        output_elements=patch_ids,
        include_stress_history=False,
        recovery=_recovery_config(config),
        resource_config=_resource_config(config),
    )
    base_load_case = load_case if config.slamming_include_static_load else None
    transient = _backend_solve_transient_newmark(
        model,
        transient_config,
        pressure_patches=[patch],
        base_load_case=base_load_case,
    )
    return {
        "status": str(transient.status),
        "pressure_pa": pressure,
        "duration_s": duration,
        "total_time_s": total_time,
        "dt_s": dt,
        "selected_shells": float(len(patch_ids)),
        "peak_displacement_m": float(transient.peak_displacement),
        "peak_von_mises_pa": float(transient.peak_von_mises_stress),
        "force_impulse_n_s": tuple(float(value) for value in np.asarray(transient.force_impulse, dtype=float).reshape(3)),
        "include_static_load": bool(config.slamming_include_static_load),
    }


def _mesh_size_diagnostics(generated_geometry: dict) -> dict[str, float | int | str]:
    nodes = {int(node["id"]): tuple(float(value) for value in node["coords"]) for node in generated_geometry.get("nodes", [])}
    grid = generated_geometry.get("plot_grid") or []
    diagnostics: dict[str, float | int | str] = {"shell_order": _shell_order_from_geometry(generated_geometry)}
    if not grid:
        return diagnostics
    if generated_geometry.get("plot_type") == "cylinder":
        row = list(grid[0][:-1])
        z_values = sorted({nodes[node_id][2] for line in grid for node_id in line if node_id in nodes})
        radius = _positive(generated_geometry.get("radius_m", 0.0), 0.0)
        if row and radius > 0.0:
            diagnostics["circumferential_divisions"] = len(row)
            diagnostics["max_circumferential_edge_m"] = 2.0 * math.pi * radius / max(len(row), 1)
        if len(z_values) > 1:
            diagnostics["axial_divisions"] = len(z_values) - 1
            diagnostics["max_axial_edge_m"] = max(b - a for a, b in zip(z_values, z_values[1:]))
        return diagnostics
    x_values = sorted({nodes[node_id][0] for line in grid for node_id in line if node_id in nodes})
    y_values = sorted({nodes[node_id][1] for line in grid for node_id in line if node_id in nodes})
    if len(x_values) > 1:
        diagnostics["x_divisions"] = len(x_values) - 1
        diagnostics["max_x_edge_m"] = max(b - a for a, b in zip(x_values, x_values[1:]))
    if len(y_values) > 1:
        diagnostics["y_divisions"] = len(y_values) - 1
        diagnostics["max_y_edge_m"] = max(b - a for a, b in zip(y_values, y_values[1:]))
    return diagnostics


def _add_generated_axial_force(model, load_case, generated_geometry: dict, axial_force_n: float) -> None:
    try:
        axial_force = float(axial_force_n)
    except (TypeError, ValueError):
        return
    if abs(axial_force) <= 0.0:
        return
    if generated_geometry.get("plot_type") == "cylinder":
        bottom = [int(node_id) for node_id in generated_geometry.get("bottom_ring_node_ids", [])]
        top = [int(node_id) for node_id in generated_geometry.get("top_ring_node_ids", [])]
        if not bottom or not top:
            return
        for node_id in bottom:
            load_case.add_nodal_load(node_id, forces=np.array([0.0, 0.0, axial_force / len(bottom)], dtype=float))
        for node_id in top:
            load_case.add_nodal_load(node_id, forces=np.array([0.0, 0.0, -axial_force / len(top)], dtype=float))
        return
    shell_node_ids = sorted({node_id for shell in generated_geometry.get("shells", []) for node_id in shell.get("node_ids", [])})
    nodes = [model.mesh.get_node(int(node_id)) for node_id in shell_node_ids]
    nodes = [node for node in nodes if node is not None]
    if not nodes:
        return
    xs = [float(node.x) for node in nodes]
    xmin = min(xs)
    xmax = max(xs)
    tol = max((xmax - xmin) * 1.0e-9, 1.0e-9)
    left = [node for node in nodes if abs(float(node.x) - xmin) <= tol]
    right = [node for node in nodes if abs(float(node.x) - xmax) <= tol]
    if not left or not right:
        return
    for node in left:
        load_case.add_nodal_load(int(node.id), forces=np.array([axial_force / len(left), 0.0, 0.0], dtype=float))
    for node in right:
        load_case.add_nodal_load(int(node.id), forces=np.array([-axial_force / len(right), 0.0, 0.0], dtype=float))


def _line_node_weights(nodes: list[object], axis: int, closed_length: float = 0.0) -> dict[int, float]:
    if not nodes:
        return {}
    if closed_length > 0.0:
        return {int(node.id): float(closed_length) / len(nodes) for node in nodes}
    if len(nodes) == 1:
        return {int(nodes[0].id): 1.0}
    ordered = sorted(nodes, key=lambda node: float(node.coords()[axis]))
    coords = [float(node.coords()[axis]) for node in ordered]
    weights: dict[int, float] = {}
    for index, node in enumerate(ordered):
        if index == 0:
            weight = 0.5 * abs(coords[1] - coords[0])
        elif index == len(ordered) - 1:
            weight = 0.5 * abs(coords[-1] - coords[-2])
        else:
            weight = 0.5 * abs(coords[index + 1] - coords[index - 1])
        weights[int(node.id)] = float(weight)
    return weights


def _apply_weighted_edge_load(load_case, weights: dict[int, float], force_per_length: np.ndarray) -> None:
    if not weights:
        return
    vector = np.asarray(force_per_length, dtype=float)
    for node_id, weight in weights.items():
        load_case.add_nodal_load(int(node_id), forces=vector * float(weight))


def _add_custom_edge_loads(model, load_case, generated_geometry: dict, config: LightweightFEMConfig) -> None:
    if not config.custom_load_bc_enabled:
        return
    nodes = [model.mesh.get_node(int(node["id"])) for node in generated_geometry.get("nodes", [])]
    nodes = [node for node in nodes if node is not None]
    if not nodes:
        return
    coords = np.asarray([node.coords() for node in nodes], dtype=float)
    tol = max(float(np.ptp(coords[:, 0]) + np.ptp(coords[:, 1]) + np.ptp(coords[:, 2])) * 1.0e-9, 1.0e-9)
    if generated_geometry.get("plot_type") == "cylinder":
        lower_ids = set(int(node_id) for node_id in generated_geometry.get("bottom_ring_node_ids", []))
        upper_ids = set(int(node_id) for node_id in generated_geometry.get("top_ring_node_ids", []))
        lower = [node for node in nodes if int(node.id) in lower_ids]
        upper = [node for node in nodes if int(node.id) in upper_ids]
        radius = _positive(generated_geometry.get("radius_m", 0.0), 0.0)
        circumference = 2.0 * math.pi * radius if radius > 0.0 else 0.0
        _apply_weighted_edge_load(load_case, _line_node_weights(lower, 0, circumference), np.array([0.0, 0.0, -float(config.cylinder_lower_edge_load_n_per_m)]))
        _apply_weighted_edge_load(load_case, _line_node_weights(upper, 0, circumference), np.array([0.0, 0.0, float(config.cylinder_upper_edge_load_n_per_m)]))
        return
    xmin = float(np.min(coords[:, 0]))
    xmax = float(np.max(coords[:, 0]))
    ymin = float(np.min(coords[:, 1]))
    ymax = float(np.max(coords[:, 1]))
    x0_nodes = [node for node in nodes if abs(float(node.x) - xmin) <= tol]
    x1_nodes = [node for node in nodes if abs(float(node.x) - xmax) <= tol]
    y0_nodes = [node for node in nodes if abs(float(node.y) - ymin) <= tol]
    y1_nodes = [node for node in nodes if abs(float(node.y) - ymax) <= tol]
    _apply_weighted_edge_load(load_case, _line_node_weights(x0_nodes, 1), np.array([-float(config.plate_edge_x0_load_n_per_m), 0.0, 0.0]))
    _apply_weighted_edge_load(load_case, _line_node_weights(x1_nodes, 1), np.array([float(config.plate_edge_x1_load_n_per_m), 0.0, 0.0]))
    _apply_weighted_edge_load(load_case, _line_node_weights(y0_nodes, 0), np.array([0.0, -float(config.plate_edge_y0_load_n_per_m), 0.0]))
    _apply_weighted_edge_load(load_case, _line_node_weights(y1_nodes, 0), np.array([0.0, float(config.plate_edge_y1_load_n_per_m), 0.0]))


def _add_generated_end_moments(model, load_case, generated_geometry: dict, moment_nm: float) -> None:
    moment = float(moment_nm or 0.0)
    if abs(moment) <= 0.0:
        return
    if generated_geometry.get("plot_type") == "flat":
        shell_node_ids = sorted({node_id for shell in generated_geometry.get("shells", []) for node_id in shell.get("node_ids", [])})
        nodes = [model.mesh.get_node(int(node_id)) for node_id in shell_node_ids]
        nodes = [node for node in nodes if node is not None]
        if not nodes:
            return
        xs = [float(node.x) for node in nodes]
        xmin = min(xs)
        xmax = max(xs)
        tol = max((xmax - xmin) * 1.0e-9, 1.0e-9)
        left = [node for node in nodes if abs(float(node.x) - xmin) <= tol]
        right = [node for node in nodes if abs(float(node.x) - xmax) <= tol]
        if not left or not right:
            return
        for node in left:
            load_case.add_nodal_load(int(node.id), moments=np.array([0.0, moment / len(left), 0.0], dtype=float))
        for node in right:
            load_case.add_nodal_load(int(node.id), moments=np.array([0.0, -moment / len(right), 0.0], dtype=float))
        return

    bottom_ring = [int(node_id) for node_id in generated_geometry.get("bottom_ring_node_ids", [])]
    top_ring = [int(node_id) for node_id in generated_geometry.get("top_ring_node_ids", [])]
    if not bottom_ring or not top_ring:
        return

    def add_ring_moment(node_ids: list[int], sign: float) -> None:
        nodes = [model.mesh.get_node(node_id) for node_id in node_ids]
        nodes = [node for node in nodes if node is not None]
        denominator = sum(float(node.x) ** 2 for node in nodes)
        if denominator <= 1.0e-12:
            return
        for node in nodes:
            axial_force = -sign * moment * float(node.x) / denominator
            load_case.add_nodal_load(int(node.id), forces=np.array([0.0, 0.0, axial_force], dtype=float))

    add_ring_moment(bottom_ring, -1.0)
    add_ring_moment(top_ring, 1.0)


def _stress_statistics_from_model(model, displacements: np.ndarray, percentile: float = 95.0) -> dict[str, float]:
    if _backend_compute_stresses is None:
        return {"max": 0.0, "percentile": 0.0}
    values = []
    for stress in _backend_compute_stresses(model, displacements).values():
        if "von_mises" in stress:
            values.extend(np.asarray(stress["von_mises"], dtype=float).reshape(-1).tolist())
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"max": 0.0, "percentile": 0.0}
    return {"max": float(np.max(arr)), "percentile": float(np.percentile(arr, percentile))}


def _max_translation(model, displacements: np.ndarray) -> float:
    value = 0.0
    for node in model.mesh.nodes.values():
        value = max(value, float(np.linalg.norm(displacements[node.dofs[:3]])))
    return value


def _cylinder_pressure_prestress_states(model, pressure: float, radius: float) -> dict[int, dict[str, float]]:
    compression = abs(float(pressure)) * max(float(radius), 1.0e-9)
    states: dict[int, dict[str, float]] = {}
    if compression <= 0.0:
        return states
    for element_id, element in model.mesh.elements.items():
        if element.__class__.__name__ == "ShellElement":
            states[int(element_id)] = {
                "membrane_compression_x": compression,
                "membrane_compression_y": 0.5 * compression,
                "membrane_compression_xy": 0.0,
            }
    return states


def _add_cylinder_buckling_gauge(model, generated_geometry: dict) -> bool:
    """Add minimal buckling-only constraints that remove free rigid-body drift."""
    rigid_lids = list(generated_geometry.get("rigid_lids") or [])
    if not rigid_lids or getattr(model, "boundary_conditions", None):
        return False
    try:
        bottom_center = int(rigid_lids[0]["center_node_id"])
        top_center = int(rigid_lids[-1]["center_node_id"])
    except (KeyError, TypeError, ValueError):
        return False
    if model.mesh.get_node(bottom_center) is None or model.mesh.get_node(top_center) is None:
        return False
    model.add_boundary_condition(
        _full_backend.BoundaryCondition(
            "buckling_gauge_bottom_lid",
            [bottom_center],
            {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rz": 0.0},
        )
    )
    model.add_boundary_condition(
        _full_backend.BoundaryCondition(
            "buckling_gauge_top_lid",
            [top_center],
            {"ux": 0.0, "uy": 0.0},
        )
    )
    return True


def run_production_fem(geometry: dict, config: LightweightFEMConfig) -> LightweightFEMResult:
    """Run the vendored production FE mesh backend for generated ANYstructure geometry."""

    if _full_backend is None or _backend_solve_linear is None or _backend_solve_buckling is None or _backend_load_case_resultant is None:
        return LightweightFEMResult(
            status="backend_unavailable",
            stress_max_pa=0.0,
            stress_p95_pa=0.0,
            displacement_max_m=0.0,
            diagnostics=("Production FE backend is not available.",),
            solver_name="ANYstructure production FE mesh",
        )

    generated_geometry = build_generated_geometry(geometry, config)
    material_curve, material_properties = _nonlinear_curve_payload(config, geometry)
    effective_elastic_modulus = float(material_properties.get("E_pa", config.elastic_modulus_pa)) if material_properties else config.elastic_modulus_pa
    effective_yield_stress = float(material_properties.get("sigma_yield", config.yield_stress_pa)) if material_properties else config.yield_stress_pa
    include_imported_loads = _include_imported_loads(config)
    effective_pressure = _effective_pressure_pa(config)
    backend_config = _full_backend.AnyStructureFEMConfig(
        pressure_pa=abs(float(effective_pressure)),
        pressure_sign=_pressure_sign(config),
        load_scale=float(config.load_scale),
        num_buckling_modes=int(config.num_buckling_modes),
        solver_type=_solver_type(config),
        stress_percentile=min(max(float(config.stress_percentile), 0.0), 100.0),
        add_inplane_edge_loads=False,
        require_idealized_member_beams=False,
        elastic_modulus=effective_elastic_modulus,
        poisson_ratio=config.poisson_ratio,
        yield_stress=effective_yield_stress,
    )
    diagnostics = [
        "ANYstructure production FE mesh backend.",
        "Generated shells and stiffener/girder beams from active-line geometry.",
    ]
    if config.include_end_lids and geometry.get("geometry") == "cylinder":
        diagnostics.append("Applied stress-free rigid top/bottom lid diaphragms at cylinder ends.")
    if (not config.custom_load_bc_enabled) and geometry.get("geometry") != "cylinder":
        diagnostics.append("Applied flat-panel edge supports from line properties, defaulting to simply supported edges when unspecified.")
    if include_imported_loads and config.top_bottom_moment_nm:
        diagnostics.append("Applied top/bottom shell bending moment: " + str(round(float(config.top_bottom_moment_nm), 3)) + " Nm.")
    if include_imported_loads and abs(float(config.axial_force_n or 0.0)) > 0.0:
        diagnostics.append("Applied balanced axial force: " + str(round(float(config.axial_force_n), 3)) + " N.")
    if (not config.custom_load_bc_enabled) and abs(float(config.enforced_displacement_m or 0.0)) > 0.0:
        diagnostics.append("Applied prescribed displacement constraints from the enforced displacement input.")
    if _wants_s8(config):
        diagnostics.append("Generated S8 shell elements with shared midside nodes.")
    if _normalized_choice(config.symmetry_mode) == "cyclic":
        diagnostics.append("Cyclic symmetry requested; generated runtime geometry is a full 360-degree model, so no sector coupling was added.")
    elif _normalized_choice(config.symmetry_mode) not in {"none", "off"}:
        diagnostics.append("Applied generated global symmetry boundary conditions.")
    if _normalized_choice(config.member_orientation) == "radial" or (_normalized_choice(config.member_orientation) == "auto" and geometry.get("geometry") == "cylinder"):
        diagnostics.append("Applied radial member section orientation for cylinder beams where applicable.")
    if abs(float(config.stiffener_eccentricity_m or 0.0)) > 0.0 or abs(float(config.girder_eccentricity_m or 0.0)) > 0.0:
        diagnostics.append("Applied eccentric beam-shell MPC offsets for generated member beams.")
    if config.custom_load_bc_enabled:
        diagnostics.append("Using custom load and boundary-condition mode.")
        if include_imported_loads:
            diagnostics.append("Custom loads are added to the imported/generated pressure, axial force and end moment inputs.")
        else:
            diagnostics.append("Custom loads replace imported/generated pressure, axial force and end moment inputs.")
        if abs(float(config.custom_pressure_pa or 0.0)) > 0.0:
            diagnostics.append("Applied custom manual pressure: " + str(round(float(config.custom_pressure_pa), 3)) + " Pa.")
        if config.custom_use_nullspace_projection:
            diagnostics.append("Custom boundary condition is rigid-body nullspace projection with automatic generalized load balancing.")
    if _allow_unbalanced_free_free(config, geometry):
        diagnostics.append("Unbalanced free-free/nullspace static loads are explicitly allowed and will be carried as generalized balancing reactions.")
    if _wants_capacity_workflow(config):
        diagnostics.append("Using ANYintelligent structured nonlinear capacity workflow after the reference static solve.")
    buckling_range = _buckling_load_factor_range(config)
    if float(config.buckling_shift_load_factor or 0.0) > 0.0 or buckling_range is not None or bool(config.buckling_allow_dense_fallback):
        diagnostics.append("Buckling validity controls are active: shift/range filtering and dense fallback are passed to the backend.")
    if material_properties:
        diagnostics.append(
            "Using DNV-RP-C208 material curve "
            + str(material_properties.get("grade", ""))
            + ", "
            + str(material_properties.get("thickness_class", ""))
            + " for nonlinear static shell plasticity."
        )
    if config.imperfection_enabled:
        diagnostics.append("Geometric imperfection input is enabled; offsets are applied as stress-free reference geometry.")
    if config.slamming_enabled:
        diagnostics.append("Transient slamming pressure input is enabled; it is solved as a separate linear Newmark pressure-patch response.")

    try:
        model = _full_backend.build_fe_model_from_generated_geometry(generated_geometry, backend_config)
        _apply_material_curve_to_model(model, material_curve, material_properties)
        imperfection_info = {}
        if not _wants_capacity_workflow(config):
            imperfection_info = _apply_runtime_imperfection(model, generated_geometry, geometry, config)
        elif config.imperfection_enabled:
            imperfection_info = {"status": "deferred", "kind": "capacity workflow imperfection"}
            diagnostics.append("Geometric imperfection input is deferred to the capacity workflow nonlinear model.")
        if imperfection_info.get("status") == "applied":
            diagnostics.append(
                "Applied "
                + str(imperfection_info.get("kind", "geometric"))
                + " imperfection, max offset "
                + str(round(float(imperfection_info.get("max_offset_m", 0.0)) * 1000.0, 4))
                + " mm."
            )
        elif imperfection_info:
            diagnostics.append("Imperfection input was not applied: " + str(imperfection_info.get("reason", imperfection_info.get("status", "unknown"))))
        if abs(float(effective_pressure)) > 0.0:
            load_case = _full_backend.build_symmetric_load_case(None, model, backend_config)
        else:
            load_case = _full_backend.LoadCase("custom_fem_loads" if config.custom_load_bc_enabled else "anystructure_symmetric_load")
        if include_imported_loads:
            _add_generated_axial_force(model, load_case, generated_geometry, float(config.axial_force_n))
            _add_generated_end_moments(model, load_case, generated_geometry, float(config.top_bottom_moment_nm))
        _add_custom_edge_loads(model, load_case, generated_geometry, config)
        load_resultant = _backend_load_case_resultant(model, load_case)
        constraint_mode = _constraint_mode(config, geometry)
        displacements, solver_info = _backend_solve_linear(
            model,
            load_case,
            solver_type=backend_config.solver_type,
            constraint_mode=constraint_mode,
            allow_unbalanced_free_free=_allow_unbalanced_free_free(config, geometry),
        )
    except Exception as exc:
        return LightweightFEMResult(
            status="production_failed",
            stress_max_pa=0.0,
            stress_p95_pa=0.0,
            displacement_max_m=0.0,
            diagnostics=tuple(diagnostics + ["Backend status: " + str(exc)]),
            solver_name="ANYstructure production FE mesh",
        )

    static_status = str((solver_info.get("convergence_info") or {}).get("status", "unknown"))
    
    # Extract the backend name from convergence_info -> backend -> backend
    backend_info = (solver_info.get("convergence_info") or {}).get("backend") or {}
    backend_name = str(backend_info.get("backend", "unknown backend"))
    diagnostics.append(f"Linear solver backend used: {backend_name}")

    if static_status != "converged":
        diagnostics.append("Static solve status: " + static_status)
        return LightweightFEMResult(
            status="static_failed",
            stress_max_pa=0.0,
            stress_p95_pa=0.0,
            displacement_max_m=0.0,
            diagnostics=tuple(diagnostics),
            mesh_info={"nodes": model.mesh.num_nodes, "shells": len(generated_geometry.get("shells", [])), "beams": len(generated_geometry.get("beams", []))},
            load_resultant=_resultant_dict(load_resultant),
            solver_name="ANYstructure production FE mesh",
        )

    prestress_states, prestress_summary = _full_backend.recover_prestress_from_static_result(model, displacements)
    if imperfection_info:
        prestress_summary["imperfection_status"] = str(imperfection_info.get("status", ""))
        prestress_summary["imperfection_kind"] = str(imperfection_info.get("kind", ""))
        prestress_summary["imperfection_amplitude_m"] = float(imperfection_info.get("amplitude_m", 0.0) or 0.0)
        prestress_summary["imperfection_max_offset_m"] = float(imperfection_info.get("max_offset_m", 0.0) or 0.0)
        prestress_summary["imperfection_waves_a"] = float(imperfection_info.get("waves_a", 0.0) or 0.0)
        prestress_summary["imperfection_waves_b"] = float(imperfection_info.get("waves_b", 0.0) or 0.0)
    if config.slamming_enabled:
        try:
            slamming_summary = _run_slamming_transient(model, load_case, generated_geometry, geometry, config)
            if slamming_summary:
                prestress_summary["slamming_status"] = str(slamming_summary.get("status", ""))
                prestress_summary["slamming_pressure_pa"] = float(slamming_summary.get("pressure_pa", 0.0) or 0.0)
                prestress_summary["slamming_selected_shells"] = float(slamming_summary.get("selected_shells", 0.0) or 0.0)
                prestress_summary["slamming_peak_displacement_m"] = float(slamming_summary.get("peak_displacement_m", 0.0) or 0.0)
                prestress_summary["slamming_peak_von_mises_pa"] = float(slamming_summary.get("peak_von_mises_pa", 0.0) or 0.0)
                diagnostics.append(
                    "Transient slamming response "
                    + str(slamming_summary.get("status", "unknown"))
                    + "; selected shells "
                    + str(int(float(slamming_summary.get("selected_shells", 0.0) or 0.0)))
                    + ", peak displacement "
                    + str(round(float(slamming_summary.get("peak_displacement_m", 0.0) or 0.0) * 1000.0, 4))
                    + " mm."
                )
        except Exception as exc:
            prestress_summary["slamming_status"] = "failed"
            diagnostics.append("Transient slamming solve failed: " + str(exc))
    prestress_summary["constraint_method"] = str(solver_info.get("constraint_method", ""))
    prestress_summary["constraint_mode"] = str(solver_info.get("constraint_mode", ""))
    nullspace_info = solver_info.get("nullspace_info") or {}
    if solver_info.get("constraint_method") == "transformation_fixed_plus_mpc_nullspace":
        convergence_info = solver_info.get("convergence_info") or {}
        prestress_summary["nullspace_projection"] = 1.0
        prestress_summary["nullspace_rank"] = float(convergence_info.get("nullspace_rank", nullspace_info.get("reduced_rank", 0)))
        prestress_summary["relative_rigid_body_load_imbalance"] = float(convergence_info.get("relative_rigid_body_load_imbalance", 0.0) or 0.0)
        prestress_summary["rigid_body_load_imbalance_norm"] = float(convergence_info.get("rigid_body_load_imbalance_norm", 0.0) or 0.0)
        diagnostics.append("Linear solve used rigid-body nullspace projection for the remaining unsupported rigid-body modes.")
        for warning in convergence_info.get("warnings", []) or []:
            diagnostics.append(str(warning))
    else:
        prestress_summary["nullspace_projection"] = 0.0
    if material_properties:
        prestress_summary["material_model"] = "DNV-RP-C208"
        prestress_summary["steel_grade"] = str(material_properties.get("grade", ""))
        prestress_summary["steel_thickness_class"] = str(material_properties.get("thickness_class", ""))
        prestress_summary["sigma_prop_pa"] = float(material_properties.get("sigma_prop", 0.0))
        prestress_summary["sigma_yield_pa"] = float(material_properties.get("sigma_yield", 0.0))
        prestress_summary["sigma_yield_2_pa"] = float(material_properties.get("sigma_yield_2", 0.0))
        prestress_summary["eps_p_y1"] = float(material_properties.get("eps_p_y1", 0.0))
        prestress_summary["eps_p_y2"] = float(material_properties.get("eps_p_y2", 0.0))
        prestress_summary["hardening_K_pa"] = float(material_properties.get("K", 0.0))
        prestress_summary["hardening_n"] = float(material_properties.get("n", 0.0))

    analysis_model = model
    capacity_workflow_result = None
    nonlinear_result = None
    nonlinear_factor = None
    nonlinear_static_factor = None
    nonlinear_static_result = None
    plastic_strain_by_node: dict[int, float] = {}
    if _wants_capacity_workflow(config):
        if not hasattr(_full_backend, "run_nonlinear_capacity_workflow"):
            diagnostics.append("ANYintelligent capacity workflow is unavailable in this backend.")
        else:
            try:
                selected_imperfection = None
                if config.imperfection_enabled:
                    selected_imperfection, _imperfection_metadata = _build_runtime_imperfection(model, generated_geometry, geometry, config)
                workflow_config = _full_backend.CapacityWorkflowConfig(
                    num_buckling_modes=int(config.num_buckling_modes),
                    buckling_mode_number=_positive_int(config.capacity_buckling_mode_number, 1),
                    eigenmode_imperfection_amplitude=max(float(config.imperfection_amplitude_m or 0.0), 0.0) if config.imperfection_enabled else 0.0,
                    nonlinear_num_steps=_positive_int(config.nonlinear_steps, 12),
                    nonlinear_max_load_factor=max(float(config.nonlinear_max_load_factor), 1.0e-9),
                    nonlinear_max_iterations=_positive_int(config.nonlinear_max_iterations, 25),
                    nonlinear_tolerance=max(float(config.nonlinear_tolerance), 1.0e-12),
                    nonlinear_num_layers=_nonlinear_layer_count(config.nonlinear_layers),
                    mesh_min_elements_per_half_wave=_positive_int(config.capacity_mesh_min_elements_per_half_wave, 4),
                    copy_model=True,
                )
                capacity_workflow_result = _full_backend.run_nonlinear_capacity_workflow(
                    model,
                    load_case,
                    imperfection=selected_imperfection,
                    config=workflow_config,
                )
                nonlinear_static_result = capacity_workflow_result.nonlinear_result
                nonlinear_static_factor = float(nonlinear_static_result.capacity_estimate)
                buckling_result_from_workflow = capacity_workflow_result.buckling_result
                if getattr(buckling_result_from_workflow, "modes", None):
                    prestress_states = capacity_workflow_result.prestress_states
                if nonlinear_static_result.converged:
                    analysis_model = capacity_workflow_result.imperfect_model
                    displacements = np.asarray(nonlinear_static_result.displacements, dtype=float)
                    prestress_states, recovered = _full_backend.recover_prestress_from_static_result(analysis_model, displacements)
                    prestress_summary.update(recovered)
                prestress_summary["capacity_workflow_status"] = str(capacity_workflow_result.status)
                prestress_summary["capacity_workflow_capacity_factor"] = float(capacity_workflow_result.capacity_factor)
                if capacity_workflow_result.critical_load_factor is not None:
                    prestress_summary["capacity_workflow_critical_load_factor"] = float(capacity_workflow_result.critical_load_factor)
                prestress_summary["capacity_workflow_mesh_status"] = str(capacity_workflow_result.mesh_adequacy.status)
                prestress_summary["capacity_workflow_elements_per_half_wave"] = float(capacity_workflow_result.mesh_adequacy.elements_per_half_wave)
                prestress_summary["nonlinear_static_status"] = str(nonlinear_static_result.status)
                prestress_summary["nonlinear_static_load_factor"] = nonlinear_static_factor
                prestress_summary["nonlinear_static_steps"] = float(len(nonlinear_static_result.steps))
                prestress_summary["nonlinear_static_total_iterations"] = float((nonlinear_static_result.info or {}).get("total_newton_iterations", 0.0))
                prestress_summary["nonlinear_static_layers"] = float((nonlinear_static_result.info or {}).get("num_layers", _nonlinear_layer_count(config.nonlinear_layers)))
                if nonlinear_static_result.steps:
                    prestress_summary["nonlinear_static_max_plastic_strain"] = float(
                        max(step.max_equivalent_plastic_strain for step in nonlinear_static_result.steps)
                    )
                plastic_strain_by_node = _nodal_engineering_plastic_strain(analysis_model, nonlinear_static_result.element_states)
                diagnostics.append(
                    "ANYintelligent capacity workflow "
                    + str(capacity_workflow_result.status)
                    + "; mesh mode adequacy "
                    + str(capacity_workflow_result.mesh_adequacy.status)
                    + "."
                )
                for warning in getattr(capacity_workflow_result.mesh_adequacy, "warnings", ()) or ():
                    diagnostics.append(str(warning))
            except Exception as exc:
                prestress_summary["capacity_workflow_status"] = "failed"
                diagnostics.append("ANYintelligent capacity workflow failed: " + str(exc))

    if _wants_static_nonlinear_analysis(config) and capacity_workflow_result is None:
        if _backend_solve_static_nonlinear is None:
            diagnostics.append("Incremental geometric/material nonlinear static solver is unavailable in this backend.")
        else:
            try:
                nonlinear_static_result = _backend_solve_static_nonlinear(
                    model,
                    load_case,
                    max_load_factor=max(float(config.nonlinear_max_load_factor), 1.0e-9),
                    num_steps=_positive_int(config.nonlinear_steps, 12),
                    max_iterations=_positive_int(config.nonlinear_max_iterations, 25),
                    tolerance=max(float(config.nonlinear_tolerance), 1.0e-12),
                    num_layers=_nonlinear_layer_count(config.nonlinear_layers),
                )
                nonlinear_static_factor = float(nonlinear_static_result.capacity_estimate)
                prestress_summary["nonlinear_static_status"] = str(nonlinear_static_result.status)
                prestress_summary["nonlinear_static_load_factor"] = nonlinear_static_factor
                prestress_summary["nonlinear_static_steps"] = float(len(nonlinear_static_result.steps))
                prestress_summary["nonlinear_static_total_iterations"] = float((nonlinear_static_result.info or {}).get("total_newton_iterations", 0.0))
                prestress_summary["nonlinear_static_layers"] = float((nonlinear_static_result.info or {}).get("num_layers", _nonlinear_layer_count(config.nonlinear_layers)))
                if nonlinear_static_result.steps:
                    prestress_summary["nonlinear_static_max_plastic_strain"] = float(
                        max(step.max_equivalent_plastic_strain for step in nonlinear_static_result.steps)
                    )
                plastic_strain_by_node = _nodal_engineering_plastic_strain(model, nonlinear_static_result.element_states)
                diagnostics.append("Ran incremental geometric/material nonlinear static solve: " + str(nonlinear_static_result.status) + ".")
                if nonlinear_static_result.converged:
                    displacements = np.asarray(nonlinear_static_result.displacements, dtype=float)
                    prestress_states, recovered = _full_backend.recover_prestress_from_static_result(model, displacements)
                    recovered.update({key: value for key, value in prestress_summary.items() if str(key).startswith("nonlinear_static")})
                    for key in (
                        "constraint_method",
                        "constraint_mode",
                        "nullspace_projection",
                        "nullspace_rank",
                        "material_model",
                        "steel_grade",
                        "steel_thickness_class",
                        "sigma_prop_pa",
                        "sigma_yield_pa",
                        "sigma_yield_2_pa",
                        "eps_p_y1",
                        "eps_p_y2",
                        "hardening_K_pa",
                        "hardening_n",
                    ):
                        if key in prestress_summary:
                            recovered[key] = prestress_summary[key]
                    prestress_summary = recovered
            except Exception as exc:
                diagnostics.append("Incremental nonlinear static solver failed: " + str(exc))

    if _wants_tangent_stability_analysis(config) and capacity_workflow_result is None:
        if _backend_solve_nonlinear_limit is None:
            diagnostics.append("Nonlinear load-step solver is unavailable in this backend.")
        else:
            try:
                nonlinear_result = _backend_solve_nonlinear_limit(
                    model,
                    load_case,
                    prestress_states,
                    max_load_factor=3.0,
                    num_steps=12,
                    stability_tolerance=1.0e-3,
                    stop_at_limit=True,
                )
                nonlinear_factor = nonlinear_result.critical_load_factor_estimate
                if nonlinear_factor is None:
                    nonlinear_factor = nonlinear_result.last_load_factor if nonlinear_result.steps else None
                prestress_summary["nonlinear_status"] = nonlinear_result.status
                if nonlinear_factor is not None:
                    prestress_summary["nonlinear_limit_factor"] = float(nonlinear_factor)
                prestress_summary["nonlinear_steps"] = len(nonlinear_result.steps)
                diagnostics.append("Ran nonlinear tangent-stability load stepping: " + str(nonlinear_result.status) + ".")
                if _wants_nonlinear_analysis(config) and nonlinear_result.converged:
                    displacements = np.asarray(nonlinear_result.final_displacements, dtype=float)
            except Exception as exc:
                diagnostics.append("Nonlinear load-step solver failed: " + str(exc))
    if geometry.get("geometry") == "cylinder" and config.include_end_lids and _add_cylinder_buckling_gauge(model, generated_geometry):
        diagnostics.append("Applied buckling-only rigid-body gauge constraints to the lid center nodes.")
    buckling_kwargs = _buckling_solver_kwargs(config)
    if capacity_workflow_result is not None:
        buckling_result = capacity_workflow_result.buckling_result
    elif _wants_eigenvalue_buckling(config):
        buckling_result = _backend_solve_buckling(model, prestress_states, num_modes=int(config.num_buckling_modes), **buckling_kwargs)
    else:
        # Dummy result if buckling is explicitly skipped by runtime path
        class DummyBucklingResult:
            modes = ()
            failed = False
            status = "skipped by runtime path"
            diagnostics = ()
        buckling_result = DummyBucklingResult()
    if not buckling_result.modes and geometry.get("geometry") == "cylinder" and abs(float(effective_pressure)) > 0.0:
        pressure_states = _cylinder_pressure_prestress_states(
            model,
            float(effective_pressure) * float(config.load_scale),
            _positive(geometry.get("radius_m", generated_geometry.get("radius_m", 1.0)), 1.0),
        )
        if pressure_states:
            pressure_buckling_result = _backend_solve_buckling(model, pressure_states, num_modes=int(config.num_buckling_modes), **buckling_kwargs)
            if pressure_buckling_result.modes:
                buckling_result = pressure_buckling_result
                diagnostics.append("Buckling modes use equivalent external-pressure membrane prestress because the full mixed prestress returned no positive modes.")
    if _wants_nonlinear_buckling(config) and nonlinear_static_factor is not None and float(nonlinear_static_factor) > 0.0:
        buckling_factors = (float(nonlinear_static_factor),)
        diagnostics.append("Buckling factors report the incremental nonlinear static load-factor estimate for the selected buckling mode.")
    elif _wants_nonlinear_buckling(config) and nonlinear_factor is not None and float(nonlinear_factor) > 0.0:
        buckling_factors = (float(nonlinear_factor),)
        diagnostics.append("Buckling factors report the nonlinear limit-load estimate for the selected buckling mode.")
    else:
        buckling_factors = tuple(float(mode.load_factor) for mode in buckling_result.modes)
    prestress_summary["runtime_solver"] = _normalized_choice(config.runtime_solver, "stepwise")
    prestress_summary["allow_unbalanced_free_free"] = 1.0 if _allow_unbalanced_free_free(config, geometry) else 0.0
    prestress_summary["recovery_history_mode"] = _normalized_choice(config.recovery_history_mode, "full")
    prestress_summary["recovery_threads"] = float(max(int(config.recovery_threads or 0), 0))
    prestress_summary["memory_limit_mb"] = float(max(float(config.memory_limit_mb or 0.0), 0.0))
    prestress_summary["buckling_solver_status"] = str(getattr(buckling_result, "solver_status", ""))
    prestress_summary["buckling_modes_returned"] = float(len(getattr(buckling_result, "modes", []) or []))
    prestress_summary["buckling_repeated_groups"] = float(((getattr(buckling_result, "diagnostics", {}) or {}).get("num_repeated_mode_groups", 0)) or 0)
    prestress_summary["buckling_shift_load_factor"] = float(config.buckling_shift_load_factor or 0.0)
    load_factor_range = _buckling_load_factor_range(config)
    if load_factor_range is not None:
        prestress_summary["buckling_min_load_factor"] = 0.0 if load_factor_range[0] is None else float(load_factor_range[0])
        prestress_summary["buckling_max_load_factor"] = 0.0 if load_factor_range[1] is None else float(load_factor_range[1])
    prestress_summary["buckling_allow_dense_fallback"] = 1.0 if bool(config.buckling_allow_dense_fallback) else 0.0
    stress_stats = _stress_statistics_from_model(analysis_model, displacements, min(max(float(config.stress_percentile), 0.0), 100.0))
    visualization = _visualization_from_full_result(generated_geometry, analysis_model, displacements)
    if plastic_strain_by_node:
        plastic_visualization = _visualization_from_full_result(
            generated_geometry,
            analysis_model,
            displacements,
            scalar_by_node=plastic_strain_by_node,
            scalar_label="equiv. engineering plastic strain [-]",
        )
        if plastic_visualization:
            visualization["plastic_strain"] = plastic_visualization.get("stress_pa", ())
            visualization["plastic_strain_label"] = "equiv. engineering plastic strain [-]"
    visualization["buckling_modes"] = _buckling_mode_visualizations(generated_geometry, model, buckling_result)

    if not prestress_states:
        diagnostics.append("Prestress recovery returned no element states.")
    if not buckling_factors:
        diagnostics.append("Static solve converged; no positive buckling modes were returned for this load state.")

    return LightweightFEMResult(
        status="ok",
        stress_max_pa=float(stress_stats["max"]),
        stress_p95_pa=float(stress_stats["percentile"]),
        displacement_max_m=_max_translation(analysis_model, displacements),
        buckling_factors=buckling_factors,
        diagnostics=tuple(diagnostics),
        mesh_info={
            "nodes": int(model.mesh.num_nodes),
            "shells": int(len(generated_geometry.get("shells", []))),
            "beams": int(len(generated_geometry.get("beams", []))),
            "rigid_lids": int(len(generated_geometry.get("rigid_lids", []))),
            **_mesh_size_diagnostics(generated_geometry),
        },
        prestress_summary=dict(prestress_summary or {}),
        load_resultant=_resultant_dict(load_resultant),
        visualization=visualization,
        solver_name="ANYstructure production FE mesh",
    )


def _run_flat_panel(geometry: dict, config: LightweightFEMConfig) -> LightweightFEMResult:
    length = _positive(geometry.get("length_m", 1.0), 1.0)
    width = _positive(geometry.get("width_m", 1.0), 1.0)
    thickness = _positive(geometry.get("thickness_m", 0.01), 0.01)
    pressure = abs(float(_effective_pressure_pa(config)) * float(config.load_scale))
    short_span = min(length, width)

    pressure_bending = 0.125 * pressure * short_span**2 / max(thickness**2, 1.0e-12)
    direct_pressure = pressure
    stress = max(pressure_bending, direct_pressure)
    if config.include_stiffeners and geometry.get("has_stiffener"):
        stress *= 0.72
    if config.include_girders and geometry.get("has_girder"):
        stress *= 0.82

    D = config.elastic_modulus_pa * thickness**3 / (12.0 * (1.0 - config.poisson_ratio**2))
    displacement = pressure * short_span**4 / max(64.0 * D, 1.0e-12)
    sigma_cr = _plate_critical_stress(config.elastic_modulus_pa, config.poisson_ratio, thickness, short_span)
    buckling_factor = sigma_cr / max(stress, 1.0)
    div = _mesh_divisions(config.mesh_fidelity)
    spacing_cap = _positive_spacing(geometry.get("stiffener_spacing_m", 0.0)) if geometry.get("has_stiffener") else 0.0
    if spacing_cap > 0.0:
        div = max(div, int(math.ceil(max(length, width) / spacing_cap)))
    area = length * width
    return LightweightFEMResult(
        status="ok",
        stress_max_pa=stress,
        stress_p95_pa=0.92 * stress,
        displacement_max_m=displacement,
        buckling_factors=_sorted_positive_factors(buckling_factor, config.num_buckling_modes),
        diagnostics=("ANYstructure compact solver: flat shell/beam idealization.",),
        mesh_info={"nodes": (div + 1) ** 2, "shells": div * div, "beams": int(bool(geometry.get("has_stiffener"))) + int(bool(geometry.get("has_girder")))},
        prestress_summary={
            "membrane_compression_pa": pressure,
            "bending_stress_pa": pressure_bending,
            "critical_stress_pa": sigma_cr,
        },
        load_resultant={"force_n": (0.0, 0.0, pressure * area), "moment_nm": (0.0, 0.0, 0.0)},
        visualization=_flat_visualization(length, width, displacement, stress, div),
    )


def _run_cylinder(geometry: dict, config: LightweightFEMConfig) -> LightweightFEMResult:
    radius = _positive(geometry.get("radius_m", 1.0), 1.0)
    length = _positive(geometry.get("length_m", 1.0), 1.0)
    thickness = _positive(geometry.get("thickness_m", 0.01), 0.01)
    pressure = abs(float(_effective_pressure_pa(config)) * float(config.load_scale))

    hoop = pressure * radius / thickness
    axial = hoop / 2.0
    von_mises = math.sqrt(max(hoop**2 - hoop * axial + axial**2, 0.0))
    if config.include_stiffeners and geometry.get("has_stiffener"):
        von_mises *= 0.82
    if config.include_girders and geometry.get("has_girder"):
        von_mises *= 0.90

    displacement = pressure * radius**2 / max(config.elastic_modulus_pa * thickness, 1.0e-12)
    pcr = _cylinder_critical_pressure(config.elastic_modulus_pa, config.poisson_ratio, thickness, radius)
    buckling_factor = pcr / max(pressure, 1.0)
    div = _mesh_divisions(config.mesh_fidelity)
    spacing_cap = _positive_spacing(geometry.get("stiffener_spacing_m", 0.0)) if geometry.get("has_stiffener") else 0.0
    if spacing_cap > 0.0:
        div = max(div, int(math.ceil((2.0 * math.pi * radius) / spacing_cap)))
    axial_div = max(int(length / max(radius, 1.0e-9) * div / 4), 1)
    if spacing_cap > 0.0:
        axial_div = max(axial_div, int(math.ceil(length / spacing_cap)))
    area = 2.0 * math.pi * radius * length
    return LightweightFEMResult(
        status="ok",
        stress_max_pa=von_mises,
        stress_p95_pa=0.90 * von_mises,
        displacement_max_m=displacement,
        buckling_factors=_sorted_positive_factors(buckling_factor, config.num_buckling_modes),
        diagnostics=("ANYstructure compact solver: cylindrical shell membrane idealization.",),
        mesh_info={"nodes": div * (axial_div + 1), "shells": div * axial_div, "beams": int(bool(geometry.get("has_stiffener"))) + int(bool(geometry.get("has_girder")))},
        prestress_summary={
            "hoop_stress_pa": hoop,
            "axial_stress_pa": axial,
            "critical_pressure_pa": pcr,
        },
        load_resultant={"force_n": (0.0, 0.0, pressure * area), "moment_nm": (0.0, 0.0, 0.0)},
        visualization=_cylinder_visualization(radius, length, displacement, von_mises, div, axial_div),
    )


def run_lightweight_fem(geometry: dict, config: LightweightFEMConfig) -> LightweightFEMResult:
    """Run the local lightweight solver for the normalized ANYstructure geometry summary."""

    if geometry.get("geometry") == "cylinder":
        return _run_cylinder(geometry, config)
    return _run_flat_panel(geometry, config)


def full_backend_available() -> bool:
    """Return whether the vendored ANYintelligent solver backend is available."""

    return _full_backend is not None


def full_backend_api():
    """Return the vendored full solver backend module for future integration."""

    if _full_backend is None:
        raise RuntimeError("The vendored full FE solver backend is not available.")
    return _full_backend
