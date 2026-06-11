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
except ModuleNotFoundError:
    try:
        from ANYstructure.anystruct import fe_solver_backend as _full_backend
        from ANYstructure.anystruct.fe_solver_backend.assembly import compute_stresses as _backend_compute_stresses
    except ModuleNotFoundError:
        _full_backend = None
        _backend_compute_stresses = None


@dataclass(frozen=True)
class LightweightFEMConfig:
    """Runtime options for the local lightweight solver."""

    mesh_fidelity: str = "coarse"
    pressure_pa: float = 0.0
    load_scale: float = 1.0
    include_stiffeners: bool = True
    include_girders: bool = True
    num_buckling_modes: int = 5
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
    return {"coarse": 8, "medium": 16, "fine": 32}.get(str(mesh_fidelity).lower(), 8)


def _production_divisions(mesh_fidelity: str) -> int:
    return {"coarse": 4, "medium": 8, "fine": 12}.get(str(mesh_fidelity).lower(), 4)


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


def _flat_generated_geometry(geometry: dict, config: LightweightFEMConfig) -> dict[str, object]:
    length = _positive(geometry.get("length_m", 1.0), 1.0)
    width = _positive(geometry.get("width_m", 1.0), 1.0)
    thickness = _positive(geometry.get("thickness_m", 0.01), 0.01)
    div = _production_divisions(config.mesh_fidelity)
    rows = div + 1
    cols = div + 1

    def node_id(row: int, col: int) -> int:
        return 1 + row * cols + col

    nodes = [
        {
            "id": node_id(row, col),
            "coords": [length * row / div, width * col / div, 0.0],
        }
        for row in range(rows)
        for col in range(cols)
    ]
    shells = []
    element_id = 1
    for row in range(div):
        for col in range(div):
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
    if config.include_stiffeners and geometry.get("has_stiffener"):
        mid_col = cols // 2
        section = _beam_section(thickness, width, 0.08)
        for row in range(div):
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
    if config.include_girders and geometry.get("has_girder"):
        mid_row = rows // 2
        section = _beam_section(thickness, length, 0.10)
        for col in range(div):
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
    circumferential_div = max(_production_divisions(config.mesh_fidelity) * 2, 8)
    axial_div = max(int(length / max(radius, 1.0e-9) * circumferential_div / 4), 2)
    rows = axial_div + 1
    cols = circumferential_div

    def node_id(row: int, col: int) -> int:
        return 1 + row * cols + (col % cols)

    nodes = []
    for row in range(rows):
        x = length * row / axial_div
        for col in range(cols):
            theta = 2.0 * math.pi * col / cols
            nodes.append({"id": node_id(row, col), "coords": [x, radius * math.cos(theta), radius * math.sin(theta)]})

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
        section = _beam_section(thickness, radius, 0.08)
        count = min(8, cols)
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
        section = _beam_section(thickness, radius, 0.12)
        ring_rows = sorted({0, rows // 2, rows - 1})
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
    return {
        "name": "ANYstructureCylinderFullMesh",
        "nodes": nodes,
        "shells": shells,
        "beams": beams,
        "supports": [
            {"name": "axial_diaphragm_start", "node_ids": start_ring, "constraints": {"ux": 0.0}},
            {"name": "axial_diaphragm_end", "node_ids": end_ring, "constraints": {"ux": 0.0}},
            {"name": "rigid_body_anchor", "node_ids": [node_id(0, 0)], "constraints": {"uy": 0.0, "uz": 0.0}},
            {"name": "rigid_body_spin_anchor", "node_ids": [node_id(0, cols // 4)], "constraints": {"uz": 0.0}},
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
        "plot_grid": [[node_id(row, col) for col in range(cols)] + [node_id(row, 0)] for row in range(rows)],
        "plot_type": "cylinder",
        "radius_m": radius,
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


def _visualization_from_full_result(generated_geometry: dict, model, displacements: np.ndarray) -> dict[str, object]:
    grid = generated_geometry.get("plot_grid") or []
    stress_by_node = _nodal_von_mises(model, displacements)
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
                y = float(node.y)
                z = float(node.z)
                theta = math.atan2(z, y)
                radial = np.array([0.0, math.cos(theta), math.sin(theta)], dtype=float)
                translation = np.asarray(displacements[node.dofs[:3]], dtype=float)
                axial_row.append(float(node.x))
                theta_row.append(theta if theta >= 0.0 else theta + 2.0 * math.pi)
                radial_row.append(float(translation @ radial))
                stress_row.append(float(stress_by_node.get(int(node_id), 0.0)))
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
            w_row.append(float(displacements[node.dofs[2]]))
            stress_row.append(float(stress_by_node.get(int(node_id), 0.0)))
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
    }


def _resultant_dict(load_resultant) -> dict[str, tuple[float, float, float]]:
    if load_resultant is None:
        return {}
    return {
        "force_n": tuple(float(value) for value in np.asarray(load_resultant.force, dtype=float).reshape(3)),
        "moment_nm": tuple(float(value) for value in np.asarray(load_resultant.moment, dtype=float).reshape(3)),
    }


def run_production_fem(geometry: dict, config: LightweightFEMConfig) -> LightweightFEMResult:
    """Run the vendored production FE mesh backend for generated ANYstructure geometry."""

    if _full_backend is None:
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
    backend_result = _full_backend.run_anystructure_fem_mode(None, generated_geometry, backend_config)
    diagnostics = [
        "ANYstructure production FE mesh backend.",
        "Generated shells and stiffener/girder beams from active-line geometry.",
    ]
    if backend_result.invalid_reason:
        diagnostics.append("Backend status: " + str(backend_result.invalid_reason))

    visualization = {}
    if backend_result.displacements is not None:
        try:
            model = _full_backend.build_fe_model_from_generated_geometry(generated_geometry, backend_config)
            visualization = _visualization_from_full_result(generated_geometry, model, backend_result.displacements)
        except Exception as exc:
            diagnostics.append("Visualization recovery failed: " + str(exc))

    static_ok = backend_result.static_solver_status == "converged"
    buckling_factors = tuple(float(value) for value in backend_result.buckling_load_factors)
    status = "ok" if static_ok else backend_result.status
    if static_ok and not buckling_factors:
        diagnostics.append("Static solve converged; no positive buckling modes were returned for this load state.")

    return LightweightFEMResult(
        status=status,
        stress_max_pa=float(backend_result.stress_max),
        stress_p95_pa=float(backend_result.stress_percentile),
        displacement_max_m=float(backend_result.max_translation or backend_result.max_displacement),
        buckling_factors=buckling_factors,
        diagnostics=tuple(diagnostics),
        mesh_info={
            "nodes": int(backend_result.node_count),
            "shells": int(backend_result.shell_element_count),
            "beams": int(backend_result.beam_element_count),
        },
        prestress_summary=dict(backend_result.prestress_summary or {}),
        load_resultant=_resultant_dict(backend_result.load_resultant),
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
    axial_div = max(int(length / max(radius, 1.0e-9) * div / 4), 1)
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
