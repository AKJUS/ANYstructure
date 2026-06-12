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
    from anystruct.fe_solver_backend.validation import load_case_resultant as _backend_load_case_resultant
except ModuleNotFoundError:
    try:
        from ANYstructure.anystruct import fe_solver_backend as _full_backend
        from ANYstructure.anystruct.fe_solver_backend.assembly import compute_stresses as _backend_compute_stresses
        from ANYstructure.anystruct.fe_solver_backend.assembly import solve_linear as _backend_solve_linear
        from ANYstructure.anystruct.fe_solver_backend.buckling import solve_eigenvalue_buckling as _backend_solve_buckling
        from ANYstructure.anystruct.fe_solver_backend.validation import load_case_resultant as _backend_load_case_resultant
    except ModuleNotFoundError:
        _full_backend = None
        _backend_compute_stresses = None
        _backend_solve_linear = None
        _backend_solve_buckling = None
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
    elastic_modulus_pa: float = 210.0e9
    poisson_ratio: float = 0.3
    yield_stress_pa: float = 355.0e6


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


def _mesh_divisions(mesh_fidelity: str) -> int:
    return {"coarse": 8, "medium": 16, "fine": 32, "very fine": 48, "very_fine": 48}.get(str(mesh_fidelity).lower(), 8)


def _production_divisions(mesh_fidelity: str) -> int:
    return {"coarse": 4, "medium": 8, "fine": 12, "very fine": 20, "very_fine": 20}.get(str(mesh_fidelity).lower(), 4)


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
    if max_element_size > 0.0 and (mesh_size <= 0.0 or mesh_size > max_element_size):
        mesh_size = max_element_size
    if mesh_size > 0.0:
        return max(int(math.ceil(max(length, 1.0e-9) / mesh_size)), 1)
    return max(int(fallback), 1)


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
        section = _section_or_default(geometry.get("stiffener_section"), thickness, width, 0.08)
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
        section = _section_or_default(geometry.get("girder_section"), thickness, length, 0.10)
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

    boundary_nodes = sorted(
        {
            node_id(row, col)
            for row in range(rows)
            for col in range(cols)
            if row in (0, rows - 1) or col in (0, cols - 1)
        }
    )
    return {
        "name": "ANYstructureFlatPanelFullMesh",
        "nodes": nodes,
        "shells": shells,
        "beams": beams,
        "supports": [
            {
                "name": "clamped_panel_boundary",
                "node_ids": boundary_nodes,
                "constraints": {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
            }
        ],
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
    if mesh_size_cap > 0.0 and (mesh_size <= 0.0 or mesh_size > mesh_size_cap):
        mesh_size = mesh_size_cap
    if mesh_size > 0.0:
        circumferential_div = max(int(math.ceil(circumference / mesh_size)), 8)
        axial_div = max(int(math.ceil(length / mesh_size)), 2)
    else:
        base_div = _production_divisions(config.mesh_fidelity)
        circumferential_div = max(base_div * 2, 8)
        axial_div = max(int(length / max(radius, 1.0e-9) * circumferential_div / 4), 2)
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
        section = _section_or_default(geometry.get("stiffener_section"), thickness, radius, 0.08)
        count = stiffener_count if stiffener_count > 0 else min(8, cols)
        for offset in range(count):
            col = int(round(offset * cols / count)) % cols
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
        section = _section_or_default(geometry.get("girder_section"), thickness, radius, 0.12)
        ring_rows = [_index_of_break(z_breaks, pos) for pos in girder_positions] or [rows // 2]
        for row in ring_rows:
            for col in range(cols):
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

    start_ring = [node_id(0, col) for col in range(cols)]
    end_ring = [node_id(rows - 1, col) for col in range(cols)]
    rigid_lids = []
    supports = [
        {"name": "rigid_body_anchor", "node_ids": [node_id(0, 0)], "constraints": {"ux": 0.0, "uy": 0.0, "uz": 0.0}},
        {"name": "rigid_body_spin_anchor", "node_ids": [node_id(0, cols // 4)], "constraints": {"ux": 0.0}},
        {"name": "rigid_body_tilt_anchor", "node_ids": [node_id(1, 0)], "constraints": {"uy": 0.0}},
    ]
    if config.include_end_lids:
        bottom_center = rows * cols + 1
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
    return {
        "name": "ANYstructureCylinderFullMesh",
        "nodes": nodes,
        "shells": shells,
        "beams": beams,
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
    backend_config = _full_backend.AnyStructureFEMConfig(
        pressure_pa=abs(float(config.pressure_pa)),
        pressure_sign=-1.0,
        load_scale=float(config.load_scale),
        num_buckling_modes=int(config.num_buckling_modes),
        solver_type="direct",
        stress_percentile=95.0,
        add_inplane_edge_loads=False,
        require_idealized_member_beams=False,
        elastic_modulus=config.elastic_modulus_pa,
        poisson_ratio=config.poisson_ratio,
        yield_stress=config.yield_stress_pa,
    )
    diagnostics = [
        "ANYstructure production FE mesh backend.",
        "Generated shells and stiffener/girder beams from active-line geometry.",
    ]
    if config.include_end_lids and geometry.get("geometry") == "cylinder":
        diagnostics.append("Applied stress-free rigid top/bottom lid diaphragms at cylinder ends.")
    if config.top_bottom_moment_nm:
        diagnostics.append("Applied top/bottom shell bending moment: " + str(round(float(config.top_bottom_moment_nm), 3)) + " Nm.")

    try:
        model = _full_backend.build_fe_model_from_generated_geometry(generated_geometry, backend_config)
        load_case = _full_backend.build_symmetric_load_case(None, model, backend_config)
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
    buckling_factors = tuple(float(mode.load_factor) for mode in buckling_result.modes)
    stress_stats = _stress_statistics_from_model(model, displacements, 95.0)
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
