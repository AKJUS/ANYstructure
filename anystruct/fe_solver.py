"""Lightweight experimental FEM solver owned by ANYstructure.

This module is intentionally small and dependency-light.  It provides the
runtime API used by the experimental FEM popup while the production solver is
developed in ANYintelligent.  Future solver updates can replace this module
without changing the GUI handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
        from ANYstructure.anystruct.fe_solver_backend.validation import load_case_resultant as _backend_load_case_resultant
    except ModuleNotFoundError:
        _full_backend = None
        _backend_compute_stresses = None
        _backend_solve_linear = None
        _backend_solve_buckling = None
        _backend_curve_from_properties = None
        _backend_solve_nonlinear_limit = None
        _backend_solve_static_nonlinear = None
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
            return {
                "area": area,
                "Iy": max(iy, 1.0e-12),
                "Iz": max(iz, 1.0e-12),
                "J": max(j, 1.0e-12),
                "shear_factor_y": float(section.get("shear_factor_y", 5.0 / 6.0)),
                "shear_factor_z": float(section.get("shear_factor_z", 5.0 / 6.0)),
            }
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


def _flat_supports(boundary_nodes: list[int], node_id, rows: int, cols: int, config: LightweightFEMConfig) -> list[dict[str, object]]:
    mode = _normalized_choice(config.boundary_condition)
    if mode in {"free", "none"}:
        return []
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
    if mode in {"free", "none", "off"}:
        return {}
    if mode in {"simple", "simply", "simply supported", "ss"}:
        return {"uz": 0.0}
    if geometry == "cylinder" and mode in {"fixed", "clamped"}:
        return {"ux": 0.0, "uy": 0.0, "uz": 0.0}
    if mode in {"fixed", "clamped"}:
        return {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}
    return {}


def _custom_flat_supports(node_id, rows: int, cols: int, config: LightweightFEMConfig) -> list[dict[str, object]]:
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
    if mode in {"free", "none"}:
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
        _member_positions(width, stiffener_spacing, fallback_midpoint=True)
        if config.include_stiffeners and geometry.get("has_stiffener")
        else ()
    )
    girder_positions = (
        _member_positions(length, girder_spacing, fallback_midpoint=True)
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
            config.member_orientation,
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
            config.member_orientation,
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
    supports = _custom_flat_supports(node_id, rows, cols, config) if config.custom_load_bc_enabled else _flat_supports(boundary_nodes, node_id, rows, cols, config)
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
            config.member_orientation,
        )
        count = stiffener_count if stiffener_count > 0 else min(8, cols)
        for offset in range(count):
            col = int(round(offset * cols / count)) % cols
            section = dict(base_section)
            if _normalized_choice(config.member_orientation) == "radial":
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
            config.member_orientation,
        )
        ring_rows = [_index_of_break(z_breaks, pos) for pos in girder_positions] or [rows // 2]
        for row in ring_rows:
            for col in range(cols):
                section = dict(base_section)
                if _normalized_choice(config.member_orientation) == "radial":
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


def _nodal_von_mises(model, displacements: np.ndarray) -> dict[int, float]:
    if _backend_compute_stresses is None or displacements is None:
        return {}
    stresses_by_element = _backend_compute_stresses(model, displacements)
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for element_id, stress in stresses_by_element.items():
        element = model.mesh.get_element(element_id)
        if element is None or "von_mises" not in stress:
            continue
        value = float(np.mean(np.asarray(stress["von_mises"], dtype=float)))
        for node_id in getattr(element, "node_ids", []):
            node_id = int(node_id)
            sums[node_id] = sums.get(node_id, 0.0) + value
            counts[node_id] = counts.get(node_id, 0) + 1
    return {node_id: sums[node_id] / max(counts[node_id], 1) for node_id in sums}


def _visualization_from_full_result(
    generated_geometry: dict,
    model,
    displacements: np.ndarray,
    scalar_by_node: dict[int, float] | None = None,
    scalar_label: str = "stress [Pa]",
) -> dict[str, object]:
    grid = generated_geometry.get("plot_grid") or []
    scalar_by_node = _nodal_von_mises(model, displacements) if scalar_by_node is None else scalar_by_node
    if not grid or displacements is None:
        return {}

    if generated_geometry.get("plot_type") == "cylinder":
        radius = _positive(generated_geometry.get("radius_m", 1.0), 1.0)
        axial_grid = []
        theta_grid = []
        radial_grid = []
        stress_grid = []
        for row in grid:
            axial_row = []
            theta_row = []
            radial_row = []
            stress_row = []
            for node_id in row:
                node = model.mesh.get_node(int(node_id))
                if node is None:
                    continue
                x = float(node.x)
                y = float(node.y)
                theta = math.atan2(y, x)
                radial = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
                translation = np.asarray(displacements[node.dofs[:3]], dtype=float)
                radial_displacement = float(translation @ radial)
                axial_row.append(float(node.z))
                theta_row.append(theta if theta >= 0.0 else theta + 2.0 * math.pi)
                radial_row.append(radial_displacement)
                stress_row.append(float(scalar_by_node.get(int(node_id), abs(radial_displacement))))
            axial_grid.append(tuple(axial_row))
            theta_grid.append(tuple(theta_row))
            radial_grid.append(tuple(radial_row))
            stress_grid.append(tuple(stress_row))
        return {
            "type": "cylinder",
            "radius_m": radius,
            "axial_m": tuple(axial_grid),
            "theta_rad": tuple(theta_grid),
            "radial_displacement_m": tuple(radial_grid),
            "stress_pa": tuple(stress_grid),
            "scalar_label": scalar_label,
        }

    x_grid = []
    y_grid = []
    w_grid = []
    stress_grid = []
    for row in grid:
        x_row = []
        y_row = []
        w_row = []
        stress_row = []
        for node_id in row:
            node = model.mesh.get_node(int(node_id))
            if node is None:
                continue
            x_row.append(float(node.x))
            y_row.append(float(node.y))
            w = float(displacements[node.dofs[2]])
            w_row.append(w)
            stress_row.append(float(scalar_by_node.get(int(node_id), abs(w))))
        x_grid.append(tuple(x_row))
        y_grid.append(tuple(y_row))
        w_grid.append(tuple(w_row))
        stress_grid.append(tuple(stress_row))
    return {
        "type": "flat",
        "x_m": tuple(x_grid),
        "y_m": tuple(y_grid),
        "w_m": tuple(w_grid),
        "stress_pa": tuple(stress_grid),
        "scalar_label": scalar_label,
    }


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


def _wants_static_nonlinear_analysis(config: LightweightFEMConfig) -> bool:
    choice = _normalized_choice(config.analysis_type, "linear eigenvalue")
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


def _add_cylinder_end_moments(model, load_case, generated_geometry: dict, moment_nm: float) -> None:
    moment = float(moment_nm or 0.0)
    if abs(moment) <= 0.0:
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
    backend_config = _full_backend.AnyStructureFEMConfig(
        pressure_pa=abs(float(config.pressure_pa)),
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
    if config.top_bottom_moment_nm:
        diagnostics.append("Applied top/bottom shell bending moment: " + str(round(float(config.top_bottom_moment_nm), 3)) + " Nm.")
    if abs(float(config.axial_force_n or 0.0)) > 0.0:
        diagnostics.append("Applied balanced axial force: " + str(round(float(config.axial_force_n), 3)) + " N.")
    if abs(float(config.enforced_displacement_m or 0.0)) > 0.0:
        diagnostics.append("Applied prescribed displacement constraints from the enforced displacement input.")
    if _wants_s8(config):
        diagnostics.append("Generated S8 shell elements with shared midside nodes.")
    if _normalized_choice(config.symmetry_mode) == "cyclic":
        diagnostics.append("Cyclic symmetry requested; generated runtime geometry is a full 360-degree model, so no sector coupling was added.")
    elif _normalized_choice(config.symmetry_mode) not in {"none", "off"}:
        diagnostics.append("Applied generated global symmetry boundary conditions.")
    if _normalized_choice(config.member_orientation) == "radial":
        diagnostics.append("Applied radial member section orientation for cylinder beams where applicable.")
    if abs(float(config.stiffener_eccentricity_m or 0.0)) > 0.0 or abs(float(config.girder_eccentricity_m or 0.0)) > 0.0:
        diagnostics.append("Applied eccentric beam-shell MPC offsets for generated member beams.")
    if config.custom_load_bc_enabled:
        diagnostics.append("Using custom load and boundary-condition mode.")
    if material_properties:
        diagnostics.append(
            "Using DNV-RP-C208 material curve "
            + str(material_properties.get("grade", ""))
            + ", "
            + str(material_properties.get("thickness_class", ""))
            + " for nonlinear static shell plasticity."
        )

    try:
        model = _full_backend.build_fe_model_from_generated_geometry(generated_geometry, backend_config)
        _apply_material_curve_to_model(model, material_curve, material_properties)
        load_case = _full_backend.build_symmetric_load_case(None, model, backend_config)
        _add_generated_axial_force(model, load_case, generated_geometry, float(config.axial_force_n))
        _add_custom_edge_loads(model, load_case, generated_geometry, config)
        _add_cylinder_end_moments(model, load_case, generated_geometry, float(config.top_bottom_moment_nm))
        load_resultant = _backend_load_case_resultant(model, load_case)
        displacements, solver_info = _backend_solve_linear(model, load_case, solver_type=backend_config.solver_type, constraint_mode="auto")
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
    prestress_summary["constraint_method"] = str(solver_info.get("constraint_method", ""))
    prestress_summary["constraint_mode"] = str(solver_info.get("constraint_mode", ""))
    nullspace_info = solver_info.get("nullspace_info") or {}
    if solver_info.get("constraint_method") == "transformation_fixed_plus_mpc_nullspace":
        prestress_summary["nullspace_projection"] = 1.0
        prestress_summary["nullspace_rank"] = float((solver_info.get("convergence_info") or {}).get("nullspace_rank", nullspace_info.get("reduced_rank", 0)))
        diagnostics.append("Linear solve used rigid-body nullspace projection because no fixed DOFs remained after MPC/fixed-DOF reduction.")
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

    nonlinear_result = None
    nonlinear_factor = None
    nonlinear_static_factor = None
    nonlinear_static_result = None
    if _wants_static_nonlinear_analysis(config):
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

    if _wants_tangent_stability_analysis(config):
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
    buckling_result = _backend_solve_buckling(model, prestress_states, num_modes=int(config.num_buckling_modes))
    if not buckling_result.modes and geometry.get("geometry") == "cylinder" and abs(float(config.pressure_pa)) > 0.0:
        pressure_states = _cylinder_pressure_prestress_states(
            model,
            float(config.pressure_pa) * float(config.load_scale),
            _positive(geometry.get("radius_m", generated_geometry.get("radius_m", 1.0)), 1.0),
        )
        if pressure_states:
            pressure_buckling_result = _backend_solve_buckling(model, pressure_states, num_modes=int(config.num_buckling_modes))
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
    stress_stats = _stress_statistics_from_model(model, displacements, min(max(float(config.stress_percentile), 0.0), 100.0))
    visualization = _visualization_from_full_result(generated_geometry, model, displacements)
    visualization["buckling_modes"] = _buckling_mode_visualizations(generated_geometry, model, buckling_result)

    if not prestress_states:
        diagnostics.append("Prestress recovery returned no element states.")
    if not buckling_factors:
        diagnostics.append("Static solve converged; no positive buckling modes were returned for this load state.")

    return LightweightFEMResult(
        status="ok",
        stress_max_pa=float(stress_stats["max"]),
        stress_p95_pa=float(stress_stats["percentile"]),
        displacement_max_m=_max_translation(model, displacements),
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
    pressure = abs(float(config.pressure_pa) * float(config.load_scale))
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
    pressure = abs(float(config.pressure_pa) * float(config.load_scale))

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
