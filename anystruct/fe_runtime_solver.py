"""Experimental runtime FEM solver window for active ANYstructure lines.

The module owns the active-line handoff, user options and result visualization
for the experimental full-geometry FEM mode.  It calls the ANYstructure-local
``anystruct.fe_solver`` module; solver development happens in ANYintelligent
and can later be copied into that local module without changing this GUI layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import queue
import math
import os
import sys
import threading
import types

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from matplotlib import cm, colormaps, colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from anystruct import fe_solver
except ModuleNotFoundError:
    from ANYstructure.anystruct import fe_solver


@dataclass(frozen=True)
class RuntimeFEMLineSnapshot:
    """Minimal active-line payload passed from ANYstructure to the runtime FEM UI."""

    line_name: str
    line_points: Any
    structure_bundle: Any
    pressure_pa: float = 0.0
    domain: str = ""
    is_cylinder: bool = False
    diagnostics: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RuntimeFEMOptions:
    """User-selected runtime FEM options from the popup."""

    mesh_fidelity: str = "coarse"
    pressure_pa: float = 0.0
    load_scale: float = 1.0
    include_stiffeners: bool = True
    include_girders: bool = True
    include_end_lids: bool = True
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
    deformation_scale: float = 0.0
    custom_load_bc_enabled: bool = False
    custom_loads_add_to_imported: bool = False
    custom_use_nullspace_projection: bool = False
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
class RuntimeFEMRunResult:
    """Structured runtime FEM result used by text and Matplotlib visualization."""

    status: str
    summary: dict[str, Any]
    diagnostics: tuple[str, ...] = field(default_factory=tuple)
    buckling_factors: tuple[float, ...] = field(default_factory=tuple)
    stress_percentiles: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    displacement_scale: float = 0.0
    visualization: dict[str, Any] = field(default_factory=dict)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _nearest_nonlinear_layer_count(value: Any) -> int:
    requested = max(_safe_int(value, 5), 3)
    supported = (3, 5, 7, 9, 11)
    return min(supported, key=lambda item: abs(item - requested))


def _read_attr_or_call(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is None:
            break
        value = getattr(obj, name, None)
        if value is None:
            continue
        try:
            return value() if callable(value) else value
        except Exception:
            continue
    return default


def _mm_or_m_to_m(value: Any, default: float = 0.0) -> float:
    number = _safe_float(value, default)
    if number <= 0.0:
        return default
    return number / 1000.0 if number > 1.0 else number


def _member_section(member: Any) -> dict[str, float] | None:
    if member is None:
        return None
    section = _read_attr_or_call(member, "cross_section", "section", default=None)
    if isinstance(section, dict):
        return dict(section)

    stiffener_type = str(_read_attr_or_call(member, "stiffener_type", "stf_type", default="")).upper()
    height = _mm_or_m_to_m(_read_attr_or_call(member, "hw", "height", "web_height", default=None), 0.0)
    thickness = _mm_or_m_to_m(_read_attr_or_call(member, "tw", "thickness", "web_thickness", default=None), 0.0)
    flange_width = _mm_or_m_to_m(_read_attr_or_call(member, "b", "bf", "flange_width", default=None), 0.0)
    flange_thickness = _mm_or_m_to_m(_read_attr_or_call(member, "tf", "flange_thickness", default=None), 0.0)
    if stiffener_type in {"T", "TEE", "T-BAR"} and height > 0.0 and thickness > 0.0 and flange_width > 0.0 and flange_thickness > 0.0:
        web_area = height * thickness
        flange_area = flange_width * flange_thickness
        area = web_area + flange_area
        web_y = 0.5 * height
        flange_y = height + 0.5 * flange_thickness
        centroid = (web_area * web_y + flange_area * flange_y) / area
        iy = thickness * height**3 / 12.0 + web_area * (web_y - centroid) ** 2
        iy += flange_width * flange_thickness**3 / 12.0 + flange_area * (flange_y - centroid) ** 2
        iz = height * thickness**3 / 12.0 + flange_thickness * flange_width**3 / 12.0
        return {
            "area": area,
            "Iy": max(iy, 1.0e-12),
            "Iz": max(iz, 1.0e-12),
            "J": max(iy + iz, 1.0e-12),
            "shear_factor_y": 5.0 / 6.0,
            "shear_factor_z": 5.0 / 6.0,
            "label": "T" + str(round(height * 1000.0)) + "x" + str(round(thickness * 1000.0))
            + "+" + str(round(flange_width * 1000.0)) + "x" + str(round(flange_thickness * 1000.0)),
        }

    if stiffener_type not in {"FB", "FLAT", "FLATBAR", "FLAT BAR"} and not (height > 0.0 and thickness > 0.0):
        return None
    if height <= 0.0 or thickness <= 0.0:
        return None

    area = height * thickness
    iy = thickness * height**3 / 12.0
    iz = height * thickness**3 / 12.0
    return {
        "area": area,
        "Iy": max(iy, 1.0e-12),
        "Iz": max(iz, 1.0e-12),
        "J": max(iy + iz, 1.0e-12),
        "shear_factor_y": 5.0 / 6.0,
        "shear_factor_z": 5.0 / 6.0,
        "label": "FB" + str(round(height * 1000.0)) + "x" + str(round(thickness * 1000.0)),
    }


def active_line_snapshot(app: Any) -> RuntimeFEMLineSnapshot:
    """Collect current active-line data for the experimental runtime FEM mode."""

    active_line = getattr(app, "_active_line", "")
    if not active_line:
        raise ValueError("No active line selected.")
    line_dict = getattr(app, "_line_dict", {})
    line_to_struc = getattr(app, "_line_to_struc", {})
    if active_line not in line_dict:
        raise ValueError("The active line is not available in the geometry model.")
    if active_line not in line_to_struc:
        raise ValueError("The active line has no assigned structure properties.")

    diagnostics = []
    pressure = 0.0
    try:
        pressure_data = app.get_highest_pressure(active_line)
        pressure = _safe_float(pressure_data.get("normal", 0.0))
    except Exception as error:
        diagnostics.append("Pressure unavailable: " + str(error))

    bundle = line_to_struc[active_line]
    cylinder_obj = None
    try:
        cylinder_obj = bundle[5]
    except Exception:
        cylinder_obj = None

    domain = ""
    try:
        if cylinder_obj is not None:
            domain = "cylinder"
        elif bundle[0] is not None:
            domain = str(bundle[0].Plate.get_structure_type())
    except Exception:
        domain = ""

    return RuntimeFEMLineSnapshot(
        line_name=active_line,
        line_points=line_dict[active_line],
        structure_bundle=bundle,
        pressure_pa=pressure,
        domain=domain,
        is_cylinder=cylinder_obj is not None,
        diagnostics=tuple(diagnostics),
    )


def _flat_geometry_summary(snapshot: RuntimeFEMLineSnapshot) -> dict[str, Any]:
    bundle = snapshot.structure_bundle
    all_obj = bundle[0] if bundle and len(bundle) > 0 else None
    plate = getattr(all_obj, "Plate", None)
    stiffener = getattr(all_obj, "Stiffener", None)
    girder = getattr(all_obj, "Girder", None)
    return {
        "geometry": "flat panel",
        "length_m": _safe_float(_read_attr_or_call(plate, "get_span", "span"), 1.0),
        "width_m": _safe_float(_read_attr_or_call(plate, "get_s", default=None), 1.0),
        "thickness_m": _safe_float(_read_attr_or_call(plate, "get_pl_thk", default=None), 0.0),
        "has_stiffener": stiffener is not None,
        "has_girder": girder is not None,
        "stiffener_spacing_m": _safe_float(_read_attr_or_call(plate, "get_s", default=None), 0.0),
        "girder_spacing_m": _safe_float(_read_attr_or_call(girder, "get_s", "spacing", "s", default=None), 0.0),
    }


def _cylinder_geometry_summary(snapshot: RuntimeFEMLineSnapshot) -> dict[str, Any]:
    bundle = snapshot.structure_bundle
    cyl_obj = bundle[5] if bundle and len(bundle) > 5 else None
    shell = getattr(cyl_obj, "ShellObj", None)
    stiffener = getattr(cyl_obj, "LongStfObj", None)
    girder = getattr(cyl_obj, "RingFrameObj", None)
    return {
        "geometry": "cylinder",
        "radius_m": _safe_float(_read_attr_or_call(shell, "radius", default=None), 1.0),
        "length_m": _safe_float(_read_attr_or_call(shell, "length_of_shell", "tot_cyl_length", default=None), 1.0),
        "thickness_m": _safe_float(_read_attr_or_call(shell, "thk", default=None), 0.0),
        "has_stiffener": stiffener is not None,
        "has_girder": girder is not None,
        "stiffener_spacing_m": _safe_float(_read_attr_or_call(stiffener, "get_s", "spacing", "s", default=None), 0.0),
        "ring_spacing_m": _safe_float(_read_attr_or_call(shell, "_dist_between_rings", "dist_between_rings", default=None), 0.0),
        "girder_spacing_m": _safe_float(_read_attr_or_call(cyl_obj, "length_between_girders", default=None), 0.0),
        "stiffener_section": _member_section(stiffener),
        "girder_section": _member_section(girder),
    }


def runtime_geometry_summary(snapshot: RuntimeFEMLineSnapshot) -> dict[str, Any]:
    """Return compact geometry metadata for plotting and future solver handoff."""

    if snapshot.is_cylinder:
        return _cylinder_geometry_summary(snapshot)
    return _flat_geometry_summary(snapshot)


def run_runtime_fem(snapshot: RuntimeFEMLineSnapshot, options: RuntimeFEMOptions) -> RuntimeFEMRunResult:
    """Run the ANYstructure-owned lightweight FEM solver."""

    geometry = runtime_geometry_summary(snapshot)
    diagnostics = list(snapshot.diagnostics)
    solver_config = fe_solver.LightweightFEMConfig(
        mesh_fidelity=options.mesh_fidelity,
        pressure_pa=options.pressure_pa,
        load_scale=options.load_scale,
        include_stiffeners=options.include_stiffeners,
        include_girders=options.include_girders,
        include_end_lids=options.include_end_lids,
        num_buckling_modes=options.num_buckling_modes,
        mesh_size_m=options.mesh_size_m,
        top_bottom_moment_nm=options.top_bottom_moment_nm,
        boundary_condition=options.boundary_condition,
        symmetry_mode=options.symmetry_mode,
        shell_element_order=options.shell_element_order,
        analysis_type=options.analysis_type,
        buckling_analysis_type=options.buckling_analysis_type,
        pressure_direction=options.pressure_direction,
        axial_force_n=options.axial_force_n,
        enforced_displacement_m=options.enforced_displacement_m,
        stiffener_eccentricity_m=options.stiffener_eccentricity_m,
        girder_eccentricity_m=options.girder_eccentricity_m,
        member_orientation=options.member_orientation,
        solver_type=options.solver_type,
        stress_percentile=options.stress_percentile,
        elastic_modulus_pa=options.elastic_modulus_pa,
        poisson_ratio=options.poisson_ratio,
        yield_stress_pa=options.yield_stress_pa,
        material_model=options.material_model,
        steel_grade=options.steel_grade,
        steel_thickness_class=options.steel_thickness_class,
        nonlinear_max_load_factor=options.nonlinear_max_load_factor,
        nonlinear_steps=options.nonlinear_steps,
        nonlinear_max_iterations=options.nonlinear_max_iterations,
        nonlinear_tolerance=options.nonlinear_tolerance,
        nonlinear_layers=options.nonlinear_layers,
        custom_loads_add_to_imported=options.custom_loads_add_to_imported,
        custom_use_nullspace_projection=options.custom_use_nullspace_projection,
        custom_load_bc_enabled=options.custom_load_bc_enabled,
        plate_edge_x0_support=options.plate_edge_x0_support,
        plate_edge_x1_support=options.plate_edge_x1_support,
        plate_edge_y0_support=options.plate_edge_y0_support,
        plate_edge_y1_support=options.plate_edge_y1_support,
        cylinder_lower_support=options.cylinder_lower_support,
        cylinder_upper_support=options.cylinder_upper_support,
        plate_edge_x0_load_n_per_m=options.plate_edge_x0_load_n_per_m,
        plate_edge_x1_load_n_per_m=options.plate_edge_x1_load_n_per_m,
        plate_edge_y0_load_n_per_m=options.plate_edge_y0_load_n_per_m,
        plate_edge_y1_load_n_per_m=options.plate_edge_y1_load_n_per_m,
        cylinder_lower_edge_load_n_per_m=options.cylinder_lower_edge_load_n_per_m,
        cylinder_upper_edge_load_n_per_m=options.cylinder_upper_edge_load_n_per_m,
    )
    if fe_solver.full_backend_available():
        solver_result = fe_solver.run_production_fem(geometry, solver_config)
        if solver_result.status in {"backend_unavailable", "invalid", "static_failed", "production_failed"}:
            fallback = fe_solver.run_lightweight_fem(geometry, solver_config)
            diagnostics.extend(solver_result.diagnostics)
            diagnostics.append("Production FE mesh failed; using compact fallback result.")
            solver_result = fallback
    else:
        solver_result = fe_solver.run_lightweight_fem(geometry, solver_config)
    diagnostics.extend(solver_result.diagnostics)

    summary = {
        **geometry,
        "line": snapshot.line_name,
        "mesh_fidelity": options.mesh_fidelity,
        "pressure_pa": float(options.pressure_pa) * float(options.load_scale),
        "include_stiffeners": bool(options.include_stiffeners),
        "include_girders": bool(options.include_girders),
        "include_end_lids": bool(options.include_end_lids),
        "num_buckling_modes": int(options.num_buckling_modes),
        "mesh_size_m": float(options.mesh_size_m),
        "top_bottom_moment_nm": float(options.top_bottom_moment_nm),
        "boundary_condition": str(options.boundary_condition),
        "symmetry_mode": str(options.symmetry_mode),
        "shell_element_order": str(options.shell_element_order),
        "analysis_type": str(options.analysis_type),
        "buckling_analysis_type": str(options.buckling_analysis_type),
        "pressure_direction": str(options.pressure_direction),
        "axial_force_n": float(options.axial_force_n),
        "enforced_displacement_m": float(options.enforced_displacement_m),
        "stiffener_eccentricity_m": float(options.stiffener_eccentricity_m),
        "girder_eccentricity_m": float(options.girder_eccentricity_m),
        "member_orientation": str(options.member_orientation),
        "solver_type": str(options.solver_type),
        "stress_percentile": float(options.stress_percentile),
        "elastic_modulus_pa": float(options.elastic_modulus_pa),
        "poisson_ratio": float(options.poisson_ratio),
        "yield_stress_pa": float(options.yield_stress_pa),
        "material_model": str(options.material_model),
        "steel_grade": str(options.steel_grade),
        "steel_thickness_class": str(options.steel_thickness_class),
        "nonlinear_max_load_factor": float(options.nonlinear_max_load_factor),
        "nonlinear_steps": int(options.nonlinear_steps),
        "nonlinear_max_iterations": int(options.nonlinear_max_iterations),
        "nonlinear_tolerance": float(options.nonlinear_tolerance),
        "nonlinear_layers": int(options.nonlinear_layers),
        "deformation_scale": float(options.deformation_scale),
        "custom_load_bc_enabled": bool(options.custom_load_bc_enabled),
        "custom_loads_add_to_imported": bool(options.custom_loads_add_to_imported),
        "custom_use_nullspace_projection": bool(options.custom_use_nullspace_projection),
        "plate_edge_x0_support": str(options.plate_edge_x0_support),
        "plate_edge_x1_support": str(options.plate_edge_x1_support),
        "plate_edge_y0_support": str(options.plate_edge_y0_support),
        "plate_edge_y1_support": str(options.plate_edge_y1_support),
        "cylinder_lower_support": str(options.cylinder_lower_support),
        "cylinder_upper_support": str(options.cylinder_upper_support),
        "plate_edge_x0_load_n_per_m": float(options.plate_edge_x0_load_n_per_m),
        "plate_edge_x1_load_n_per_m": float(options.plate_edge_x1_load_n_per_m),
        "plate_edge_y0_load_n_per_m": float(options.plate_edge_y0_load_n_per_m),
        "plate_edge_y1_load_n_per_m": float(options.plate_edge_y1_load_n_per_m),
        "cylinder_lower_edge_load_n_per_m": float(options.cylinder_lower_edge_load_n_per_m),
        "cylinder_upper_edge_load_n_per_m": float(options.cylinder_upper_edge_load_n_per_m),
        "solver": solver_result.solver_name,
        "mesh_info": dict(solver_result.mesh_info),
        "max_displacement_m": solver_result.displacement_max_m,
        "prestress_summary": dict(solver_result.prestress_summary),
        "load_resultant": dict(solver_result.load_resultant),
    }
    return RuntimeFEMRunResult(
        status=solver_result.status,
        summary=summary,
        diagnostics=tuple(diagnostics),
        buckling_factors=solver_result.buckling_factors,
        stress_percentiles=(
            ("p95", solver_result.stress_p95_pa),
            ("max", solver_result.stress_max_pa),
        ),
        displacement_scale=solver_result.displacement_max_m,
        visualization=dict(solver_result.visualization),
    )


def _plot_grid_values(grid: Any) -> list[list[float]]:
    try:
        return [[_safe_float(value) for value in row] for row in grid]
    except TypeError:
        return []


def _all_grid_values(grid: list[list[float]]) -> list[float]:
    return [value for row in grid for value in row]


def _surface_facecolors(values_grid: list[list[float]]):
    values = _all_grid_values(values_grid) or [0.0]
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values) if max(values) > min(values) else min(values) + 1.0)
    cmap = colormaps["jet"]
    return [[cmap(norm(value)) for value in row] for row in values_grid], norm, cmap


def _visualization_displacement_extent(visualization: dict[str, Any]) -> float:
    if visualization.get("type") == "cylinder":
        values = _all_grid_values(_plot_grid_values(visualization.get("radial_displacement_m")))
    else:
        values = _all_grid_values(_plot_grid_values(visualization.get("w_m")))
    return max((abs(value) for value in values), default=0.0)


def _displacement_plot_scale(
    geometry: dict[str, Any],
    result: RuntimeFEMRunResult | None,
    visualization: dict[str, Any] | None = None,
    override_scale: float | None = None,
) -> float:
    if override_scale is not None and override_scale > 0.0:
        return float(override_scale)
    summary_scale = _safe_float((result.summary if result is not None else {}).get("deformation_scale"), 0.0)
    if summary_scale > 0.0:
        return summary_scale
    display_displacement = _visualization_displacement_extent(visualization or {})
    result_displacement = 0.0 if result is None else result.displacement_scale
    displacement = max(display_displacement, result_displacement)
    if displacement <= 0.0:
        return 1.0
    length = _safe_float(geometry.get("length_m"), 1.0)
    width = _safe_float(geometry.get("width_m"), _safe_float(geometry.get("radius_m"), 1.0))
    reference = max(length, width, _safe_float(geometry.get("radius_m"), 0.0), 1.0e-9)
    return min(max(0.08 * reference / max(displacement, 1.0e-12), 1.0), 1.0e5)


def _set_3d_axes_limits(axis: Any, x: list[list[float]], y: list[list[float]], z: list[list[float]]) -> None:
    xs = _all_grid_values(x)
    ys = _all_grid_values(y)
    zs = _all_grid_values(z)
    if not xs or not ys or not zs:
        return
    axis.set_xlim(min(xs), max(xs))
    axis.set_ylim(min(ys), max(ys))
    axis.set_zlim(min(zs), max(zs) if max(zs) > min(zs) else min(zs) + 1.0e-9)
    try:
        axis.set_box_aspect((max(xs) - min(xs) or 1.0, max(ys) - min(ys) or 1.0, max(zs) - min(zs) or 1.0))
    except Exception:
        pass


def _buckling_mode_shapes(result: RuntimeFEMRunResult | None) -> list[dict[str, Any]]:
    if result is None:
        return []
    return list((result.visualization or {}).get("buckling_modes") or [])


def _selected_visualization(result: RuntimeFEMRunResult, display_mode: str) -> tuple[dict[str, Any], str, bool]:
    if display_mode == "plastic":
        visualization = dict(result.visualization or {})
        if visualization.get("plastic_strain"):
            visualization["stress_pa"] = visualization.get("plastic_strain")
            visualization["scalar_label"] = visualization.get("plastic_strain_label") or "equiv. engineering plastic strain [-]"
            visualization["scalar_kind"] = "raw"
            return visualization, "Engineering plastic strain", False
    if display_mode.startswith("mode:"):
        try:
            mode_number = int(display_mode.split(":", 1)[1])
        except (IndexError, ValueError):
            mode_number = -1
        for mode in _buckling_mode_shapes(result):
            if int(mode.get("mode_number", -1)) == mode_number:
                factor = _safe_float(mode.get("load_factor"))
                title = "Buckling mode " + str(mode_number) + "  LF=" + str(round(factor, 4))
                return dict(mode.get("shape") or {}), title, True
    return dict(result.visualization or {}), "Static stress/displacement", False


def _plot_visualization_surface(
    figure: Figure,
    axis: Any,
    geometry: dict[str, Any],
    result: RuntimeFEMRunResult,
    display_mode: str = "static",
    deformation_scale: float | None = None,
) -> None:
    visualization, title, is_mode = _selected_visualization(result, display_mode)
    scalar_values = _plot_grid_values(visualization.get("stress_pa"))
    if is_mode:
        color_grid = scalar_values
        colorbar_label = str(visualization.get("scalar_label") or "mode amplitude")
    elif visualization.get("scalar_kind") == "raw":
        color_grid = scalar_values
        colorbar_label = str(visualization.get("scalar_label") or "value")
    else:
        color_grid = [[value / 1.0e6 for value in row] for row in scalar_values]
        colorbar_label = "stress [MPa]"
    facecolors, norm, cmap = _surface_facecolors(color_grid)
    scale = _displacement_plot_scale(geometry, result, visualization, deformation_scale)

    if visualization.get("type") == "cylinder":
        axial = _plot_grid_values(visualization.get("axial_m"))
        theta = _plot_grid_values(visualization.get("theta_rad"))
        radial_displacement = _plot_grid_values(visualization.get("radial_displacement_m"))
        radius = max(_safe_float(visualization.get("radius_m"), _safe_float(geometry.get("radius_m"), 1.0)), 1.0e-9)
        x = [
            [(radius + radial_displacement[row_index][col_index] * scale) * math.cos(theta[row_index][col_index])
             for col_index in range(len(theta[row_index]))]
            for row_index in range(len(theta))
        ]
        y = [
            [(radius + radial_displacement[row_index][col_index] * scale) * math.sin(theta[row_index][col_index])
             for col_index in range(len(theta[row_index]))]
            for row_index in range(len(theta))
        ]
        axis.plot_surface(
            np.asarray(x),
            np.asarray(y),
            np.asarray(axial),
            facecolors=np.asarray(facecolors),
            rstride=1,
            cstride=1,
            linewidth=0.15,
            antialiased=True,
            shade=False,
        )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("height [m]")
        _set_3d_axes_limits(axis, x, y, axial)
        try:
            axis.view_init(elev=18.0, azim=-45.0)
        except Exception:
            pass
    else:
        x = _plot_grid_values(visualization.get("x_m"))
        y = _plot_grid_values(visualization.get("y_m"))
        w = _plot_grid_values(visualization.get("w_m"))
        z = [[value * scale for value in row] for row in w]
        axis.plot_surface(
            np.asarray(x),
            np.asarray(y),
            np.asarray(z),
            facecolors=np.asarray(facecolors),
            rstride=1,
            cstride=1,
            linewidth=0.15,
            antialiased=True,
            shade=False,
        )
        axis.set_xlabel("length [m]")
        axis.set_ylabel("width [m]")
        axis.set_zlabel("w x" + str(round(scale, 1)))
        _set_3d_axes_limits(axis, x, y, z)

    axis.set_title(title)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(_all_grid_values(color_grid))
    figure.colorbar(mappable, ax=axis, shrink=0.68, pad=0.1, label=colorbar_label)


def create_runtime_fem_result_figure(
    snapshot: RuntimeFEMLineSnapshot,
    result: RuntimeFEMRunResult | None = None,
    display_mode: str = "static",
    deformation_scale: float | None = None,
) -> Figure:
    """Create the Matplotlib result visualization used in the runtime popup."""

    figure = Figure(figsize=(8.0, 4.1), dpi=100)
    geometry_ax = figure.add_subplot(121, projection="3d")
    result_ax = figure.add_subplot(122)
    geometry = runtime_geometry_summary(snapshot) if result is None else result.summary

    if result is None or not result.visualization:
        geometry_ax.set_title("Static stress/displacement")
        geometry_ax.text2D(0.08, 0.56, "Run FEM to plot stresses and displacements.", transform=geometry_ax.transAxes)
        geometry_ax.set_xlabel("length [m]")
        geometry_ax.set_ylabel("width/radius [m]")
        geometry_ax.set_zlabel("displacement")
        result_ax.text(0.5, 0.55, "No FEM run yet", ha="center", va="center", fontsize=12)
        result_ax.text(0.5, 0.42, "Results will appear here after Run FEM.", ha="center", va="center", fontsize=9)
        result_ax.set_axis_off()
    else:
        _plot_visualization_surface(figure, geometry_ax, geometry, result, display_mode, deformation_scale)
        result_ax.set_title("Buckling modes")
        result_ax.set_axis_off()
        summary_lines = [
            "status: " + result.status.replace("_", " "),
            "solver: " + str(geometry.get("solver", "")),
            "max disp [mm]: " + str(round(1000.0 * _safe_float(geometry.get("max_displacement_m")), 4)),
        ]
        for label, value in result.stress_percentiles:
            summary_lines.append(label + " stress [MPa]: " + str(round(value / 1.0e6, 3)))
        result_ax.text(0.02, 0.98, "\n".join(summary_lines), transform=result_ax.transAxes,
                       ha="left", va="top", fontsize=9)
        if result.buckling_factors:
            rows = [
                [str(index), str(round(factor, 4))]
                for index, factor in enumerate(result.buckling_factors, start=1)
            ]
        else:
            rows = [["-", "No positive modes"]]
        table = result_ax.table(
            cellText=rows,
            colLabels=["Mode", "Load factor"],
            cellLoc="center",
            colLoc="center",
            bbox=[0.02, 0.08, 0.76, 0.58],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for index in range(1, len(rows) + 1):
            if display_mode == "mode:" + rows[index - 1][0]:
                for col in range(2):
                    table[(index, col)].set_facecolor("#dbeafe")
    figure.tight_layout()
    return figure


def create_runtime_fem_geometry_preview_figure(snapshot: RuntimeFEMLineSnapshot, app: Any | None = None) -> Figure:
    """Create the 3D geometry preview shown in the runtime FEM popup."""

    if app is not None and hasattr(app, "create_prop_3d_figure_for_line"):
        try:
            preview = app.create_prop_3d_figure_for_line(snapshot.line_name)
            if isinstance(preview, tuple) and preview:
                figure = preview[0]
                if isinstance(figure, Figure):
                    return figure
        except Exception:
            pass

    geometry = runtime_geometry_summary(snapshot)
    figure = Figure(figsize=(3.0, 2.05), dpi=100)
    axis = figure.add_subplot(111, projection="3d")

    if snapshot.is_cylinder:
        radius = max(_safe_float(geometry.get("radius_m"), 1.0), 1.0e-6)
        length = max(_safe_float(geometry.get("length_m"), 1.0), 1.0e-6)
        theta = np.linspace(0.0, 2.0 * math.pi, 38)
        z = np.linspace(0.0, length, 8)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)
        axis.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            color="#c7d2fe",
            edgecolor="#64748b",
            linewidth=0.12,
            alpha=0.78,
            shade=False,
        )
        axis.set_xlabel("x", fontsize=6)
        axis.set_ylabel("y", fontsize=6)
        axis.set_zlabel("L", fontsize=6)
        try:
            axis.set_box_aspect((1.0, 1.0, max(length / max(2.0 * radius, 1.0e-6), 0.35)))
        except Exception:
            pass
    else:
        length = max(_safe_float(geometry.get("length_m"), 1.0), 1.0e-6)
        width = max(_safe_float(geometry.get("width_m"), 1.0), 1.0e-6)
        thickness = max(_safe_float(geometry.get("thickness_m"), 0.0), 0.0)
        x_grid, y_grid = np.meshgrid([0.0, length], [0.0, width])
        z_grid = np.zeros_like(x_grid)
        axis.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            color="#d1d5db",
            edgecolor="#64748b",
            linewidth=0.25,
            alpha=0.92,
            shade=False,
        )
        web_height = max(width * 0.18, thickness * 10.0, 0.08)
        if geometry.get("has_stiffener"):
            y_mid = 0.5 * width
            axis.plot([0.0, length], [y_mid, y_mid], [web_height, web_height], color="#334155", linewidth=2.0)
            axis.plot([0.0, length], [y_mid, y_mid], [0.0, web_height], color="#475569", linewidth=1.2)
        if geometry.get("has_girder"):
            x_mid = 0.5 * length
            axis.plot([x_mid, x_mid], [0.0, width], [web_height * 1.35, web_height * 1.35], color="#7f1d1d", linewidth=2.0)
            axis.plot([x_mid, x_mid], [0.0, width], [0.0, web_height * 1.35], color="#991b1b", linewidth=1.2)
        axis.set_xlabel("L", fontsize=6)
        axis.set_ylabel("s", fontsize=6)
        axis.set_zlabel("h", fontsize=6)
        try:
            axis.set_box_aspect((length, width, max(web_height * 1.6, width * 0.18)))
        except Exception:
            pass

    axis.set_title("3D section view", fontsize=8)
    axis.tick_params(labelsize=5, pad=0)
    try:
        axis.view_init(elev=22, azim=-55)
    except Exception:
        pass
    figure.tight_layout(pad=0.3)
    return figure


def format_runtime_fem_result(result: RuntimeFEMRunResult) -> str:
    """Format runtime FEM result text for the popup."""

    summary = result.summary
    lines = [
        "Runtime FEM status: " + result.status.replace("_", " "),
        "",
        "Line: " + str(summary.get("line", "")),
        "Geometry: " + str(summary.get("geometry", "")),
        "Mesh fidelity: " + str(summary.get("mesh_fidelity", "")),
        "Shell element: " + str(summary.get("shell_element_order", "")),
        "Boundary condition: " + str(summary.get("boundary_condition", "")),
        "Symmetry: " + str(summary.get("symmetry_mode", "")),
        "Analysis type: " + str(summary.get("analysis_type", "")),
        "Buckling type: " + str(summary.get("buckling_analysis_type", "")),
        "Linear solver: " + str(summary.get("solver_type", "")),
        "Pressure [Pa]: " + str(round(_safe_float(summary.get("pressure_pa")), 3)),
        "Pressure direction: " + str(summary.get("pressure_direction", "")),
        "Axial force [N]: " + str(round(_safe_float(summary.get("axial_force_n")), 3)),
        "Enforced displacement [m]: " + str(round(_safe_float(summary.get("enforced_displacement_m")), 6)),
        "Mesh size override [m]: " + str(round(_safe_float(summary.get("mesh_size_m")), 4)),
        "Top/bottom moment [Nm]: " + str(round(_safe_float(summary.get("top_bottom_moment_nm")), 3)),
        "Include stiffener beams: " + str(bool(summary.get("include_stiffeners"))),
        "Include girder/frame beams: " + str(bool(summary.get("include_girders"))),
        "Include top/bottom lid: " + str(bool(summary.get("include_end_lids"))),
        "Member orientation: " + str(summary.get("member_orientation", "")),
        "Stiffener eccentricity [m]: " + str(round(_safe_float(summary.get("stiffener_eccentricity_m")), 6)),
        "Girder eccentricity [m]: " + str(round(_safe_float(summary.get("girder_eccentricity_m")), 6)),
        "Material model: " + str(summary.get("material_model", "")),
        "Steel grade/class: " + str(summary.get("steel_grade", "")) + " / " + str(summary.get("steel_thickness_class", "")),
        "Material E [GPa]: " + str(round(_safe_float(summary.get("elastic_modulus_pa")) / 1.0e9, 3)),
        "Poisson ratio: " + str(round(_safe_float(summary.get("poisson_ratio")), 4)),
        "Yield stress [MPa]: " + str(round(_safe_float(summary.get("yield_stress_pa")) / 1.0e6, 3)),
        "Stress percentile: " + str(round(_safe_float(summary.get("stress_percentile")), 2)),
        "Nonlinear max LF / steps: "
        + str(round(_safe_float(summary.get("nonlinear_max_load_factor")), 3))
        + " / "
        + str(_safe_int(summary.get("nonlinear_steps"), 0)),
        "Nonlinear layers / max iterations: "
        + str(_safe_int(summary.get("nonlinear_layers"), 0))
        + " / "
        + str(_safe_int(summary.get("nonlinear_max_iterations"), 0)),
        "Deformation plot scale: " + ("auto" if _safe_float(summary.get("deformation_scale"), 0.0) <= 0.0 else str(round(_safe_float(summary.get("deformation_scale")), 3))),
        "Custom load/BC mode: " + str(bool(summary.get("custom_load_bc_enabled"))),
        "Buckling modes: " + str(summary.get("num_buckling_modes", "")),
        "Max displacement [mm]: " + str(round(1000.0 * _safe_float(summary.get("max_displacement_m")), 4)),
    ]
    if summary.get("custom_load_bc_enabled"):
        lines.append("Custom loads add to imported/generated loads: " + str(bool(summary.get("custom_loads_add_to_imported"))))
        lines.append("Custom nullspace boundary: " + str(bool(summary.get("custom_use_nullspace_projection"))))
        if str(summary.get("geometry", "")).lower().startswith("cylinder"):
            lines.extend(
                [
                    "Cylinder lower/upper support: "
                    + str(summary.get("cylinder_lower_support", ""))
                    + " / "
                    + str(summary.get("cylinder_upper_support", "")),
                    "Cylinder lower/upper edge load [N/m]: "
                    + str(round(_safe_float(summary.get("cylinder_lower_edge_load_n_per_m")), 3))
                    + " / "
                    + str(round(_safe_float(summary.get("cylinder_upper_edge_load_n_per_m")), 3)),
                ]
            )
        else:
            lines.extend(
                [
                    "Plate edge supports x0/x1/y0/y1: "
                    + ", ".join(
                        str(summary.get(key, ""))
                        for key in ("plate_edge_x0_support", "plate_edge_x1_support", "plate_edge_y0_support", "plate_edge_y1_support")
                    ),
                    "Plate edge loads x0/x1/y0/y1 [N/m]: "
                    + ", ".join(
                        str(round(_safe_float(summary.get(key)), 3))
                        for key in (
                            "plate_edge_x0_load_n_per_m",
                            "plate_edge_x1_load_n_per_m",
                            "plate_edge_y0_load_n_per_m",
                            "plate_edge_y1_load_n_per_m",
                        )
                    ),
                ]
            )
    if result.buckling_factors:
        lines.append("Critical load factor: " + str(round(result.buckling_factors[0], 4)))
    mesh_info = summary.get("mesh_info") or {}
    if mesh_info:
        lines.extend([
            "",
            "FE mesh:",
            " - nodes: " + str(mesh_info.get("nodes", 0)),
            " - shells: " + str(mesh_info.get("shells", 0)),
            " - beams: " + str(mesh_info.get("beams", 0)),
            " - rigid lids: " + str(mesh_info.get("rigid_lids", 0)),
            " - shell order: " + str(mesh_info.get("shell_order", "")),
        ])
        for key in ("max_x_edge_m", "max_y_edge_m", "max_circumferential_edge_m", "max_axial_edge_m"):
            if key in mesh_info:
                lines.append(" - " + key + ": " + str(round(_safe_float(mesh_info.get(key)), 4)))
    prestress = summary.get("prestress_summary") or {}
    if prestress:
        constraint_method = str(prestress.get("constraint_method", "") or "")
        if constraint_method:
            lines.extend(["", "Linear constraint handling:"])
            lines.append(" - method: " + constraint_method)
            if _safe_float(prestress.get("nullspace_projection"), 0.0) > 0.0:
                lines.append(" - nullspace projection: used")
                lines.append(" - remaining rigid-body modes: " + str(_safe_int(prestress.get("nullspace_rank"), 0)))
                lines.append(" - relative load imbalance: " + str(round(_safe_float(prestress.get("relative_rigid_body_load_imbalance")), 6)))
                lines.append(" - meaning: remaining rigid-body modes were projected out and any rigid-body load imbalance was carried as generalized balancing reactions.")
            else:
                lines.append(" - nullspace projection: not used")
        lines.extend(["", "Recovered prestress / reference state:"])
        material_keys = {
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
        }
        nonlinear_static_keys = {
            "nonlinear_static_status",
            "nonlinear_static_load_factor",
            "nonlinear_static_steps",
            "nonlinear_static_total_iterations",
            "nonlinear_static_layers",
            "nonlinear_static_max_plastic_strain",
        }
        special_keys = {
            "nonlinear_status",
            "nonlinear_limit_factor",
            "nonlinear_steps",
            "constraint_method",
            "constraint_mode",
            "nullspace_projection",
            "nullspace_rank",
            "relative_rigid_body_load_imbalance",
            "rigid_body_load_imbalance_norm",
            *material_keys,
            *nonlinear_static_keys,
        }
        if prestress.get("material_model"):
            lines.extend(["", "DNV-RP-C208 material curve:"])
            lines.append(" - grade: " + str(prestress.get("steel_grade", "")))
            lines.append(" - thickness class: " + str(prestress.get("steel_thickness_class", "")))
            lines.append(" - sigma_prop/yield/yield2 [MPa]: " + " / ".join(
                str(round(_safe_float(prestress.get(key)) / 1.0e6, 3))
                for key in ("sigma_prop_pa", "sigma_yield_pa", "sigma_yield_2_pa")
            ))
            lines.append(" - eps_p_y1/eps_p_y2: " + str(prestress.get("eps_p_y1", "")) + " / " + str(prestress.get("eps_p_y2", "")))
            lines.append(" - K [MPa] / n: " + str(round(_safe_float(prestress.get("hardening_K_pa")) / 1.0e6, 3)) + " / " + str(prestress.get("hardening_n", "")))
        nonlinear_static_status = str(prestress.get("nonlinear_static_status", "") or "")
        if nonlinear_static_status:
            lines.extend(["", "Incremental nonlinear static solve:"])
            lines.append(" - status: " + nonlinear_static_status.replace("_", " "))
            lines.append(" - last converged load factor: " + str(round(_safe_float(prestress.get("nonlinear_static_load_factor")), 4)))
            lines.append(" - completed steps: " + str(_safe_int(prestress.get("nonlinear_static_steps"), 0)))
            lines.append(" - Newton iterations: " + str(_safe_int(prestress.get("nonlinear_static_total_iterations"), 0)))
            lines.append(" - through-thickness layers: " + str(_safe_int(prestress.get("nonlinear_static_layers"), 0)))
            lines.append(" - max equivalent plastic strain: " + str(round(_safe_float(prestress.get("nonlinear_static_max_plastic_strain")), 6)))
            if nonlinear_static_status == "completed":
                lines.append(" - interpretation: all requested proportional load was reached; this is not necessarily a collapse load.")
            elif nonlinear_static_status == "stopped_at_limit":
                lines.append(" - interpretation: the adaptive Newton solve stopped at the last stable converged load increment.")
        for key, value in prestress.items():
            if key in special_keys:
                continue
            lines.append(" - " + key + ": " + str(round(_safe_float(value), 3)))
        nonlinear_status = str(prestress.get("nonlinear_status", "") or "")
        if nonlinear_status:
            lines.extend(["", "Nonlinear tangent-stability check:"])
            lines.append(" - status: " + nonlinear_status.replace("_", " "))
            steps = _safe_float(prestress.get("nonlinear_steps"), 0.0)
            lines.append(" - completed load steps: " + str(int(steps)))
            limit_factor = _safe_float(prestress.get("nonlinear_limit_factor"), 0.0)
            if nonlinear_status in {"limit_point_detected", "near_limit_point", "completed"} and limit_factor > 0.0:
                lines.append(" - estimated nonlinear load factor: " + str(round(limit_factor, 4)))
            else:
                lines.append(" - estimated nonlinear load factor: not available")
                if nonlinear_status == "initial_tangent_not_positive":
                    lines.append(" - explanation: the initial tangent stiffness was not positive for the selected prestress state.")
                elif steps == 0:
                    lines.append(" - explanation: the nonlinear check stopped before the first load increment.")
                else:
                    lines.append(" - explanation: no usable limit point was found in the configured load-step range.")
    load_resultant = summary.get("load_resultant") or {}
    if load_resultant:
        force = load_resultant.get("force_n", (0.0, 0.0, 0.0))
        lines.extend(["", "Load resultant force [N]: " + ", ".join(str(round(_safe_float(component), 3)) for component in force)])
    if result.diagnostics:
        lines.extend(["", "Diagnostics:"])
        lines.extend(" - " + item for item in result.diagnostics)
    return "\n".join(lines)


FEM_OPTION_INFO: dict[str, dict[str, str]] = {
    "mesh_fidelity": {
        "title": "Mesh Fidelity",
        "purpose": "Controls the default shell mesh density when no explicit mesh size is given.",
        "use": "The runtime generator refines the plating mesh and always inserts mesh lines at stiffeners and girders. Coarser meshes run faster; finer meshes give better stress and buckling-mode resolution.",
        "output": "Changes node, shell and beam counts, stress recovery, displacement shape and eigenvalue buckling factors.",
        "caution": "Very fine meshes can become expensive for cylinders with many stiffeners. Use explicit mesh size when you need a repeatable target element length.",
    },
    "mesh_size_m": {
        "title": "Mesh Size",
        "purpose": "Optional target element edge length in metres.",
        "use": "When greater than zero, generated mesh divisions are based on this size. The generator still limits the size so stiffener and girder lines remain represented by mesh edges.",
        "output": "Reported in FE mesh diagnostics as maximum axial, circumferential, x or y edge size.",
        "caution": "A value of zero lets the selected mesh fidelity decide. A very small value can make the solve slow.",
    },
    "pressure_pa": {
        "title": "Pressure",
        "purpose": "Uniform pressure magnitude used for the static prestress calculation.",
        "use": "The pressure is applied to shell elements using the selected pressure direction and load scale. The recovered membrane stresses are then used for buckling.",
        "output": "Affects displacements, stresses, recovered prestress and buckling load factors.",
        "caution": "When analysing imported FEA result stresses, avoid double counting pressure unless pressure is intentionally part of the buckling load case.",
    },
    "load_scale": {
        "title": "Load Scale",
        "purpose": "Multiplier on pressure and generated design loads.",
        "use": "The solver multiplies the input pressure by this factor before the static solve.",
        "output": "Changes load resultant, static stresses and buckling prestress.",
        "caution": "Keep at 1.0 unless running a sensitivity or intentionally scaling the load case.",
    },
    "top_bottom_moment_nm": {
        "title": "Top/Bottom Moment",
        "purpose": "Applies an opposite shell bending-style moment couple at cylinder ends.",
        "use": "The generated load case distributes nodal moments on the end rings. It is mainly useful for cylinder examples and controlled studies.",
        "output": "Shown in diagnostics and contributes to static displacement/stress recovery.",
        "caution": "This is a simplified end moment input, not a full end-cap or external frame model.",
    },
    "include_stiffeners": {
        "title": "Include Stiffener Beams",
        "purpose": "Controls whether generated stiffener beam members are included in the FE model.",
        "use": "When enabled, stiffeners are represented as beam elements tied to shell plating. When disabled, only shell plating and other enabled members are solved.",
        "output": "Changes beam count, local stiffness, stress distribution and buckling modes.",
        "caution": "Disabling stiffeners is useful for comparison only; it usually does not represent the real structure.",
    },
    "include_girders": {
        "title": "Include Girder/Frame Beams",
        "purpose": "Controls whether generated girders, frames or ring members are included.",
        "use": "When enabled, girders are represented as beam elements. Cylinder ring frames are tied into the shell model.",
        "output": "Changes beam count, global stiffness, local buckling boundary behaviour and recovered stresses.",
        "caution": "For stiffened cylinders, girder/frame modelling has a strong effect on global shell modes.",
    },
    "include_end_lids": {
        "title": "Top/Bottom Lid",
        "purpose": "Adds stress-free rigid diaphragm constraints at cylinder ends.",
        "use": "The end ring nodes are tied to free reference nodes. The lid adds local diaphragm behaviour without shell elements, pressure loads or lid stress recovery.",
        "output": "Shown as rigid lids in mesh diagnostics and affects cylinder end deformation.",
        "caution": "At least one global motion remains free before buckling gauge constraints are added, avoiding artificial axial membrane locking.",
    },
    "num_buckling_modes": {
        "title": "Buckling Modes",
        "purpose": "Number of positive buckling factors/mode shapes requested.",
        "use": "The eigensolver returns the lowest positive modes available from the recovered prestress state.",
        "output": "Controls the number of mode entries in the display selector and load-factor table.",
        "caution": "More modes require more solver work and may include local modes that are not design governing.",
    },
    "boundary_condition": {
        "title": "Boundary Condition",
        "purpose": "Defines the generated global support assumptions.",
        "use": "Flat panels can be free, simply supported, pinned or clamped. Cylinders use rigid-body anchors unless clamped/free is selected or rigid lids are active.",
        "output": "Affects stiffness, displacement, stress recovery and buckling factors.",
        "caution": "Boundary conditions are often the largest modelling assumption. Check whether the generated support matches the intended physical restraint.",
    },
    "symmetry_mode": {
        "title": "Symmetry",
        "purpose": "Applies global symmetry constraints for x, y or z symmetry planes.",
        "use": "The generator constrains normal displacement and compatible rotations on the detected symmetry plane. Cyclic is recorded for full 360 degree cylinder models.",
        "output": "Shown in diagnostics and changes the constrained DOF set.",
        "caution": "Only use symmetry if the geometry, load and expected buckling mode are symmetric.",
    },
    "shell_element_order": {
        "title": "Shell Element",
        "purpose": "Selects 4-node or 8-node quadrilateral shell elements.",
        "use": "S4 is faster. S8 adds shared midside nodes and uses higher-order shell interpolation in the core solver.",
        "output": "Mesh diagnostics report the shell order. S8 usually increases node count and runtime.",
        "caution": "S8 can improve curvature and bending representation, but it should be verified with mesh convergence.",
    },
    "analysis_type": {
        "title": "Analysis Type",
        "purpose": "Controls how the reference stress/prestress state is established before buckling capacity is interpreted.",
        "use": "Linear eigenvalue runs one linear static solve, recovers membrane prestress, and sends that state to the eigenvalue buckling solver. Nonlinear stability is the older tangent-stability check: it scales the linear prestress and monitors K - lambda KG. Geometric nonlinear static uses incremental Newton-Raphson with von Karman shell kinematics and beam-column axial coupling. Geometric + material nonlinear static also attaches the DNV-RP-C208 layered J2 plasticity curve to shell elements.",
        "output": "Linear analysis produces static displacement/stress and eigenvalue buckling factors. Tangent stability prints a tangent limit estimate when one is found. Incremental nonlinear static prints status, completed load factor, Newton iterations, through-thickness layers and maximum equivalent plastic strain.",
        "caution": "Incremental nonlinear static is more expensive than linear eigenvalue analysis. A completed nonlinear static run at the requested max load factor means the target load was reached; it is not automatically a collapse load unless the run stops at a limit.",
    },
    "buckling_analysis_type": {
        "title": "Buckling Type",
        "purpose": "Controls how the instability result is reported after the reference stress state has been recovered.",
        "use": "Linear eigenvalue solves K phi = lambda KG phi and reports positive eigenvalues as load factors with corresponding mode shapes. Nonlinear limit uses the tangent-stability load-step estimate when available; it is a capacity estimate from stiffness loss rather than an eigenmode table.",
        "output": "Linear eigenvalue output gives several mode numbers and load factors. Nonlinear limit output gives one estimated limit factor when the load-step procedure finds a limit point; otherwise the output explains why the estimate is unavailable.",
        "caution": "Eigenvalue factors are elastic bifurcation factors around the current prestress state. Nonlinear limit estimates include tangent-stiffness degradation in the current simplified solver, but they are not a full post-buckling collapse trace.",
    },
    "pressure_direction": {
        "title": "Pressure Direction",
        "purpose": "Selects whether pressure acts with or against the shell normal.",
        "use": "External pressure is destabilizing for typical cylinders; internal pressure reverses the sign.",
        "output": "Changes load resultant, stress signs and buckling prestress.",
        "caution": "Shell normal direction follows generated element ordering. Verify sign using displacement direction and diagnostics.",
    },
    "axial_force_n": {
        "title": "Axial Force",
        "purpose": "Adds a balanced axial force to the generated model.",
        "use": "For flat panels the force is applied on opposite x edges. For cylinders it is applied to the end rings.",
        "output": "Shown in diagnostics and included in load resultant/prestress recovery.",
        "caution": "Positive sign follows the current runtime convention; verify whether it produces tension or compression for the case.",
    },
    "enforced_displacement_m": {
        "title": "Enforced Displacement",
        "purpose": "Adds a prescribed displacement constraint to study displacement-controlled response.",
        "use": "Flat panels prescribe out-of-plane displacement near the panel centre. Cylinders prescribe radial displacement on a representative ring.",
        "output": "Appears as prescribed displacement constraints and affects static stress recovery.",
        "caution": "This is a modelling study input. Avoid combining with incompatible supports that over-constrain the same DOF.",
    },
    "stiffener_eccentricity_m": {
        "title": "Stiffener Eccentricity",
        "purpose": "Offsets generated stiffener beam nodes from the shell midsurface.",
        "use": "The offset is represented with exact beam-shell MPC constraints, so beam nodes are separate from shell nodes.",
        "output": "Changes beam-shell coupling stiffness and member stress recovery.",
        "caution": "Positive eccentricity follows the generated shell normal/radial direction.",
    },
    "girder_eccentricity_m": {
        "title": "Girder Eccentricity",
        "purpose": "Offsets generated girder/frame beam nodes from the shell midsurface.",
        "use": "The runtime creates separate girder nodes and beam-shell MPC coupling where applicable.",
        "output": "Changes frame/girder stiffness contribution and stress recovery.",
        "caution": "For cylinders with rigid lids, lid-ring nodes are kept compatible with one-level MPC constraints.",
    },
    "member_orientation": {
        "title": "Member Orientation",
        "purpose": "Controls beam local section orientation for asymmetric members.",
        "use": "Auto uses the backend default. Global Y/Z prescribe section local direction. Radial aligns cylinder members with the local radial direction.",
        "output": "Affects bending stiffness axes and member stress recovery.",
        "caution": "Wrong orientation can swap strong/weak axes. Use the 3D preview and diagnostics to verify.",
    },
    "solver_type": {
        "title": "Linear Solver",
        "purpose": "Selects the linear equation solver used by the static solve.",
        "use": "Direct is robust for normal model sizes. Iterative solvers are available for experiments on larger sparse systems.",
        "output": "Solver status and convergence information are printed in run status.",
        "caution": "Use direct unless memory or runtime requires experimenting with iterative solvers.",
    },
    "stress_percentile": {
        "title": "Stress Percentile",
        "purpose": "Controls the reported percentile stress used for summary output.",
        "use": "The solver samples recovered von Mises stresses and reports the requested percentile.",
        "output": "Affects p95/pXX stress summaries and plot annotations.",
        "caution": "Percentile stress is for summary robustness; design checks may require location-specific stresses.",
    },
    "elastic_modulus_gpa": {
        "title": "Elastic Modulus",
        "purpose": "Young's modulus used for shell and beam material stiffness.",
        "use": "Converted from GPa to Pa before the solver is called.",
        "output": "Affects stiffness, displacement, stress recovery and buckling factors.",
        "caution": "Typical steel value is about 210 GPa.",
    },
    "poisson_ratio": {
        "title": "Poisson Ratio",
        "purpose": "Material Poisson ratio used in shell constitutive stiffness.",
        "use": "The GUI clamps it below 0.5 for numerical stability.",
        "output": "Affects shell bending/membrane stiffness and buckling estimates.",
        "caution": "Typical steel value is about 0.3.",
    },
    "yield_stress_mpa": {
        "title": "Yield Stress",
        "purpose": "Material yield stress stored in the FE material model.",
        "use": "Converted from MPa to Pa and passed to the backend. For linear elastic material this is metadata. For DNV-RP-C208 material, the selected table row supplies sigma_yield and overrides this value in the nonlinear shell material.",
        "output": "Included in run summary and used by downstream utilization checks. In material nonlinear static the DNV curve section reports the active low-fractile yield values.",
        "caution": "Linear eigenvalue buckling is still elastic. Select geometric + material nonlinear static and a DNV-RP-C208 material model to include yielding in the incremental static solve.",
    },
    "material_model": {
        "title": "Material Model",
        "purpose": "Selects whether the FE material remains linear elastic or uses the DNV-RP-C208 low-fractile steel curve in nonlinear static analysis.",
        "use": "Linear elastic keeps shell and beam stiffness elastic. DNV-RP-C208 steel attaches a true-stress versus true-plastic-strain curve to shell elements. The curve combines a proportional/yield transition, a yield plateau and a power-law hardening branch.",
        "output": "Affects incremental nonlinear static response, plastic strain output and the nonlinear load-factor estimate. Linear static and eigenvalue buckling stay elastic.",
        "caution": "The current material nonlinearity is for shell plane-stress layers. Beam plasticity is not included in this local runtime formulation.",
    },
    "steel_grade": {
        "title": "Steel Grade",
        "purpose": "DNV-RP-C208 steel grade used for nonlinear shell plasticity.",
        "use": "Available presets are S235, S275, S355, S420 and S460. The values are the low-fractile true stress-strain properties from DNV-RP-C208 Table 4-2 to Table 4-6.",
        "output": "Sets E, sigma_prop, sigma_yield, sigma_yield_2, eps_p_y1, eps_p_y2, K and n for the nonlinear shell material curve.",
        "caution": "Use the grade matching the plate material. If an imported model has mixed steels, this single setting is still a simplification.",
    },
    "steel_thickness_class": {
        "title": "Steel Thickness Class",
        "purpose": "Selects the DNV-RP-C208 table column for the chosen steel grade.",
        "use": "Auto uses the generated plate thickness. Manual choices force a table class such as t <= 16 mm or 16 < t <= 40 mm.",
        "output": "Changes the low-fractile yield and hardening values used by the material nonlinear solver.",
        "caution": "Some grades in the RP tables are only provided up to 63 mm in the attached values; auto uses the largest available class when the thickness exceeds the listed range.",
    },
    "nonlinear_max_load_factor": {
        "title": "Nonlinear Max Load Factor",
        "purpose": "Maximum proportional load multiplier attempted by the incremental nonlinear static solver.",
        "use": "The load case is ramped from zero to this factor using adaptive increments. If convergence fails repeatedly, the solver reports the last stable factor.",
        "output": "Controls whether the result is a reached target load or a stopped-at-limit estimate.",
        "caution": "A very high value can increase runtime. A value of 1.0 checks the design load only; it may not find collapse.",
    },
    "nonlinear_steps": {
        "title": "Nonlinear Steps",
        "purpose": "Initial number of proportional load increments for nonlinear static analysis.",
        "use": "The solver starts with max load factor divided by this count, then halves or grows the increment adaptively depending on Newton convergence.",
        "output": "Affects runtime, convergence robustness and the load factor resolution near a limit.",
        "caution": "Too few steps can make a nonlinear solve fail early. Too many steps can be slow.",
    },
    "nonlinear_max_iterations": {
        "title": "Nonlinear Max Iterations",
        "purpose": "Maximum Newton iterations allowed inside one nonlinear load increment.",
        "use": "Each increment solves internal force equilibrium. The solver retries difficult increments with line search before cutting the step size.",
        "output": "Printed as total Newton iterations in the nonlinear static result.",
        "caution": "Increasing this may help difficult cases but can also spend time on a step that should be cut smaller.",
    },
    "nonlinear_tolerance": {
        "title": "Nonlinear Tolerance",
        "purpose": "Relative residual tolerance for Newton convergence in nonlinear static analysis.",
        "use": "The increment converges when the reduced residual norm is below this tolerance times the reference load norm.",
        "output": "Controls numerical convergence strictness of the nonlinear static status.",
        "caution": "Very tight tolerances can make plastic or near-limit steps expensive. Very loose tolerances can hide residual imbalance.",
    },
    "nonlinear_layers": {
        "title": "Nonlinear Layers",
        "purpose": "Number of through-thickness Gauss-Lobatto layers used for shell plasticity.",
        "use": "Layered integration captures bending plastification through the plate thickness. Supported values are odd Lobatto rules such as 3, 5, 7, 9 or 11.",
        "output": "Affects plastic strain, bending capacity and nonlinear static runtime.",
        "caution": "Five layers is a practical default. More layers cost more but can improve plastic bending resolution.",
    },
    "display_choice": {
        "title": "Display",
        "purpose": "Chooses which result visualization is shown after a run.",
        "use": "Static view shows displacement/stress. Engineering plastic strain is available after a material nonlinear run. Buckling mode views show mode shape and load factor.",
        "output": "Only affects plotting; it does not rerun the solver.",
        "caution": "Mode amplitudes are normalized for visualization, not physical displacement magnitudes.",
    },
    "deformation_scale": {
        "title": "Deformation Scale",
        "purpose": "Controls the visual magnification of displacement in the 3D result plot.",
        "use": "Set 0 for automatic scaling. Set a positive value to multiply physical displacements by that value in the plot.",
        "output": "Only affects the visualization. The solver, stresses and load factors are not changed.",
        "caution": "A very large scale can make the shape easier to see but can also make the geometry look physically misleading.",
    },
    "nullspace_projection": {
        "title": "Nullspace Projection",
        "purpose": "Explains how the linear solver handles a free-free or under-constrained model.",
        "use": "After fixed supports and MPCs are applied, the solver checks for rigid-body modes. If no fixed DOFs remain and rigid-body modes are present, the static solve uses an augmented nullspace system. This projects out pure rigid-body motion and returns a gauged deformation field.",
        "output": "The run print says whether nullspace projection was used. If it is used with non-self-equilibrated loads, the core solver may also report balancing generalized reactions.",
        "caution": "Nullspace projection is not a physical support. It prevents numerical rigid-body drift so stress recovery can proceed. For a physical static solution, use self-equilibrated loads or define real supports.",
    },
    "custom_load_bc_enabled": {
        "title": "Custom Load/BC Mode",
        "purpose": "Switches from automatic generated boundary/load assumptions to user-defined supports and edge loads.",
        "use": "When enabled, plate side supports or cylinder end supports are taken only from the custom fields below. By default the pressure, axial force and end moment inputs are not used; tick the additive-load option if the custom edge loads should be added to those imported/generated loads.",
        "output": "The run summary prints the custom support choices and edge loads. Diagnostics show that custom mode is active.",
        "caution": "This mode can easily create free-free or over-constrained models. Check the nullspace projection status and load resultant after each run.",
    },
    "custom_loads_add_to_imported": {
        "title": "Add To Imported Loads",
        "purpose": "Controls whether custom edge loads replace or supplement the pressure and other generated load inputs.",
        "use": "Unchecked means custom loads are the complete load case. Checked means pressure, axial force and top/bottom moment are applied first, then the custom edge loads are added.",
        "output": "Changes load resultant, stress recovery and buckling prestress.",
        "caution": "Use this deliberately. Leaving it unchecked is safest when studying a clean custom load path.",
    },
    "custom_use_nullspace_projection": {
        "title": "Nullspace Boundary",
        "purpose": "Uses rigid-body nullspace projection as the boundary condition for custom load/BC mode.",
        "use": "No explicit support edges are applied. The solver projects out rigid-body motion and reports any rigid-body load imbalance as balancing generalized reactions.",
        "output": "The result print reports nullspace rank and relative rigid-body load imbalance.",
        "caution": "This is a mathematical free-body gauge, not a physical support. It is useful for understanding self-equilibrated loads and free-body behaviour.",
    },
    "plate_edge_supports": {
        "title": "Plate Edge Supports",
        "purpose": "Defines supports on the four generated plate sides x0, x1, y0 and y1.",
        "use": "Free applies no restraint. Simply supported restrains out-of-plane displacement. Fixed restrains translations and rotations on the selected edge.",
        "output": "Changes support constraints, displacement, stress recovery and buckling factors.",
        "caution": "The side names follow generated coordinates: x0/x1 are the low/high x edges, y0/y1 are the low/high y edges.",
    },
    "cylinder_end_supports": {
        "title": "Cylinder End Supports",
        "purpose": "Defines support assumptions at the lower and upper cylinder ends.",
        "use": "Free applies no restraint. Simply supported restrains axial displacement at the end. Fixed restrains translations at the end. If rigid lids are active, support is applied to the lid reference node.",
        "output": "Changes cylinder global stiffness, membrane stress recovery and buckling factors.",
        "caution": "Fixed cylinder ends can significantly increase membrane stiffness. Use free or simple ends when global motion should be allowed.",
    },
    "plate_edge_loads": {
        "title": "Plate Edge Loads",
        "purpose": "Adds in-plane normal line loads to plate edges in N/m.",
        "use": "Positive x0/x1 loads act outward from the low/high x sides. Positive y0/y1 loads act outward from the low/high y sides.",
        "output": "Loads are distributed to edge nodes and included in the load resultant, recovered stresses and buckling prestress.",
        "caution": "Use sign carefully. Compressive edge load usually means load directed into the panel, which may require a negative value on an outward-positive edge.",
    },
    "cylinder_edge_loads": {
        "title": "Cylinder End Edge Loads",
        "purpose": "Adds axial line loads to lower and upper cylinder end rings in N/m.",
        "use": "Positive lower load acts in negative global z, and positive upper load acts in positive global z, giving an outward end-pull convention.",
        "output": "Loads are distributed around the end ring and included in load resultant, stresses and buckling prestress.",
        "caution": "For axial compression, choose signs so the loads act toward the cylinder mid-height.",
    },
}


class RuntimeFEMWindow:
    """Popup window for the experimental full-geometry FEM runtime solver."""

    def __init__(self, parent: Any, app: Any, use_parent_as_window: bool = False):
        self.app = app
        self.snapshot = active_line_snapshot(app)
        if use_parent_as_window:
            self.window = parent
            self.window.configure(background=getattr(app, "_general_color", "#f0f0f0"))
        else:
            self.window = tk.Toplevel(parent, background=getattr(app, "_general_color", "#f0f0f0"))
        self.window.title("Experimental FEM solver")
        self.window.geometry("1100x760")
        self.window.minsize(980, 640)
        self.window.resizable(True, True)
        if not use_parent_as_window:
            self.window.transient(parent)
        try:
            self.window.attributes("-toolwindow", False)
        except Exception:
            pass

        self.mesh_fidelity = tk.StringVar(value="coarse")
        self.mesh_size_m = tk.DoubleVar(value=0.0)
        self.load_scale = tk.DoubleVar(value=1.0)
        self.pressure_pa = tk.DoubleVar(value=self.snapshot.pressure_pa)
        self.top_bottom_moment_nm = tk.DoubleVar(value=_safe_float(getattr(app, "_fem_default_top_bottom_moment_nm", 0.0)))
        self.include_stiffeners = tk.BooleanVar(value=True)
        self.include_girders = tk.BooleanVar(value=True)
        self.include_end_lids = tk.BooleanVar(value=bool(self.snapshot.is_cylinder))
        self.num_buckling_modes = tk.IntVar(value=5)
        self.boundary_condition = tk.StringVar(value="auto")
        self.symmetry_mode = tk.StringVar(value="none")
        self.shell_element_order = tk.StringVar(value="S4")
        self.analysis_type = tk.StringVar(value="linear eigenvalue")
        self.buckling_analysis_type = tk.StringVar(value="linear eigenvalue")
        self.pressure_direction = tk.StringVar(value="external")
        self.axial_force_n = tk.DoubleVar(value=0.0)
        self.enforced_displacement_m = tk.DoubleVar(value=0.0)
        self.stiffener_eccentricity_m = tk.DoubleVar(value=0.0)
        self.girder_eccentricity_m = tk.DoubleVar(value=0.0)
        self.member_orientation = tk.StringVar(value="auto")
        self.solver_type = tk.StringVar(value="direct")
        self.stress_percentile = tk.DoubleVar(value=95.0)
        self.elastic_modulus_gpa = tk.DoubleVar(value=210.0)
        self.poisson_ratio = tk.DoubleVar(value=0.3)
        self.yield_stress_mpa = tk.DoubleVar(value=355.0)
        self.material_model = tk.StringVar(value="linear elastic")
        self.steel_grade = tk.StringVar(value="S355")
        self.steel_thickness_class = tk.StringVar(value="auto")
        self.nonlinear_max_load_factor = tk.DoubleVar(value=3.0)
        self.nonlinear_steps = tk.IntVar(value=12)
        self.nonlinear_max_iterations = tk.IntVar(value=25)
        self.nonlinear_tolerance = tk.DoubleVar(value=1.0e-6)
        self.nonlinear_layers = tk.IntVar(value=5)
        self.deformation_scale = tk.DoubleVar(value=0.0)
        self.custom_load_bc_enabled = tk.BooleanVar(value=False)
        self.custom_loads_add_to_imported = tk.BooleanVar(value=False)
        self.custom_use_nullspace_projection = tk.BooleanVar(value=False)
        self.plate_edge_x0_support = tk.StringVar(value="free")
        self.plate_edge_x1_support = tk.StringVar(value="free")
        self.plate_edge_y0_support = tk.StringVar(value="free")
        self.plate_edge_y1_support = tk.StringVar(value="free")
        self.cylinder_lower_support = tk.StringVar(value="free")
        self.cylinder_upper_support = tk.StringVar(value="free")
        self.plate_edge_x0_load_n_per_m = tk.DoubleVar(value=0.0)
        self.plate_edge_x1_load_n_per_m = tk.DoubleVar(value=0.0)
        self.plate_edge_y0_load_n_per_m = tk.DoubleVar(value=0.0)
        self.plate_edge_y1_load_n_per_m = tk.DoubleVar(value=0.0)
        self.cylinder_lower_edge_load_n_per_m = tk.DoubleVar(value=0.0)
        self.cylinder_upper_edge_load_n_per_m = tk.DoubleVar(value=0.0)
        self.display_choice = tk.StringVar(value="Static displacement/stress")
        self.display_mode_labels: dict[str, str] = {"Static displacement/stress": "static"}
        self.current_result: RuntimeFEMRunResult | None = None
        self.result_text = None
        self.figure_canvas = None
        self.figure_toolbar = None
        self.figure_toolbar_frame = None
        self.preview_canvas = None
        self.figure_parent = None
        self.display_selector = None
        self.run_button = None
        self.progress_bar = None
        self.solver_thread = None
        self.solver_queue = queue.Queue()
        try:
            self.deformation_scale.trace_add("write", lambda *_args: self._refresh_figure())
        except Exception:
            pass

        self._build()

    def _info_button(self, parent: Any, key: str) -> ttk.Button:
        return ttk.Button(parent, text="i", width=2, command=lambda info_key=key: self._show_solver_info(info_key))

    def _add_control_row(
        self,
        parent: Any,
        row: int,
        key: str,
        label: str,
        control: Any,
        sticky: str = tk.EW,
    ) -> Any:
        self._info_button(parent, key).grid(row=row, column=0, sticky=tk.W, padx=(8, 4), pady=4)
        ttk.Label(parent, text=label).grid(row=row, column=1, sticky=tk.W, padx=(0, 8), pady=4)
        control.grid(row=row, column=2, sticky=sticky, padx=(0, 8), pady=4)
        return control

    def _add_option_row(
        self,
        parent: Any,
        row: int,
        key: str,
        label: str,
        variable: tk.Variable,
        values: tuple[str, ...],
        width: int | None = None,
    ) -> ttk.OptionMenu:
        control = ttk.OptionMenu(parent, variable, variable.get(), *values)
        if width is not None:
            try:
                control.configure(width=width)
            except Exception:
                pass
        return self._add_control_row(parent, row, key, label, control)

    def _add_entry_row(
        self,
        parent: Any,
        row: int,
        key: str,
        label: str,
        variable: tk.Variable,
        width: int = 12,
    ) -> ttk.Entry:
        control = ttk.Entry(parent, textvariable=variable, width=width)
        return self._add_control_row(parent, row, key, label, control)

    def _add_check_row(self, parent: Any, row: int, key: str, text: str, variable: tk.BooleanVar) -> ttk.Checkbutton:
        self._info_button(parent, key).grid(row=row, column=0, sticky=tk.W, padx=(8, 4), pady=4)
        control = ttk.Checkbutton(parent, text=text, variable=variable)
        control.grid(row=row, column=1, columnspan=2, sticky=tk.W, padx=(0, 8), pady=4)
        return control

    @staticmethod
    def _configure_option_grid(parent: Any) -> None:
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=0)
        parent.columnconfigure(2, weight=1)

    def _show_solver_info(self, key: str) -> None:
        info = FEM_OPTION_INFO.get(key)
        if not info:
            messagebox.showinfo("FEM option", "No detailed information is available for this option.")
            return
        dialog = tk.Toplevel(self.window)
        dialog.title(str(info.get("title", "FEM option")))
        dialog.geometry("560x520")
        dialog.minsize(480, 420)
        dialog.transient(self.window)
        dialog.configure(background="#f5f7fb")

        header = tk.Frame(dialog, background="#172033")
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=str(info.get("title", "FEM option")),
            background="#172033",
            foreground="white",
            font=("Segoe UI", 14, "bold"),
            padx=16,
            pady=12,
        ).pack(anchor=tk.W)

        body = ttk.Frame(dialog, padding=14)
        body.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(
            body,
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            padx=12,
            pady=10,
            background="white",
            foreground="#111827",
            font=("Segoe UI", 10),
        )
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(body, orient=tk.VERTICAL, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.configure(yscrollcommand=scrollbar.set)
        text.tag_configure("section", font=("Segoe UI", 10, "bold"), foreground="#172033", spacing1=8, spacing3=2)
        text.tag_configure("body", lmargin1=10, lmargin2=10, spacing3=6)
        for title, field in (
            ("Purpose", "purpose"),
            ("How It Is Used", "use"),
            ("Output Affected", "output"),
            ("Cautions", "caution"),
        ):
            text.insert(tk.END, title + "\n", "section")
            text.insert(tk.END, str(info.get(field, "")) + "\n", "body")
        text.configure(state=tk.DISABLED)

        footer = ttk.Frame(dialog, padding=(14, 0, 14, 14))
        footer.pack(fill=tk.X)
        ttk.Button(footer, text="Close", command=dialog.destroy).pack(side=tk.RIGHT)
        try:
            dialog.grab_set()
            dialog.focus_set()
        except Exception:
            pass

    def _build(self) -> None:
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        header = tk.Frame(outer, background="#172033", bd=0, highlightthickness=0)
        header.pack(fill=tk.X, pady=(0, 12))
        header_inner = tk.Frame(header, background="#172033")
        header_inner.pack(fill=tk.X, padx=16, pady=12)
        tk.Label(
            header_inner,
            text="Experimental full-geometry FEM",
            background="#172033",
            foreground="white",
            font=("Segoe UI", 15, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            header_inner,
            text="ANYstructure local solver",
            background="#2563eb",
            foreground="white",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=3,
        ).pack(side=tk.RIGHT)
        tk.Label(
            header,
            text="Active-line analysis with shell plating, stiffener/girder beams, symmetric pressure and eigenvalue-style buckling factors.",
            background="#172033",
            foreground="#d7deeb",
            font=("Segoe UI", 9),
        ).pack(anchor=tk.W, padx=16, pady=(0, 12))

        body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(body)
        mid_panel = ttk.Frame(body)
        right_panel = ttk.Frame(body)
        body.add(left_panel, weight=2)
        body.add(mid_panel, weight=2)
        body.add(right_panel, weight=3)

        summary = ttk.LabelFrame(left_panel, text="Active line")
        summary.pack(fill=tk.X, pady=(0, 10))
        summary_text = (
            "Line: " + self.snapshot.line_name
            + "\nDomain: " + (self.snapshot.domain or "unknown")
            + "\nGeometry: " + ("cylinder/panel" if self.snapshot.is_cylinder else "flat panel")
            + "\nPressure [Pa]: " + str(round(self.snapshot.pressure_pa, 3))
        )
        ttk.Label(summary, text=summary_text, justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=8)

        options = ttk.LabelFrame(left_panel, text="Run setup")
        options.pack(fill=tk.X, pady=(0, 10))

        mesh_loads = ttk.LabelFrame(options, text="Mesh and loads")
        mesh_loads.pack(fill=tk.X, padx=8, pady=(8, 6))
        self._configure_option_grid(mesh_loads)
        self._add_option_row(mesh_loads, 0, "mesh_fidelity", "Mesh fidelity", self.mesh_fidelity, ("coarse", "medium", "fine", "very fine"))
        self._add_entry_row(mesh_loads, 1, "mesh_size_m", "Mesh size [m]", self.mesh_size_m)
        self._add_entry_row(mesh_loads, 2, "pressure_pa", "Pressure [Pa]", self.pressure_pa)
        self._add_entry_row(mesh_loads, 3, "load_scale", "Load scale", self.load_scale)
        self._add_entry_row(mesh_loads, 4, "top_bottom_moment_nm", "Top/bottom moment [Nm]", self.top_bottom_moment_nm)
        self._add_entry_row(mesh_loads, 5, "num_buckling_modes", "Buckling modes", self.num_buckling_modes, width=8)
        self._add_entry_row(mesh_loads, 6, "deformation_scale", "Def. scale", self.deformation_scale, width=8)

        contents = ttk.LabelFrame(options, text="Model contents")
        contents.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._configure_option_grid(contents)
        self._add_check_row(contents, 0, "include_stiffeners", "Include stiffener beams", self.include_stiffeners)
        self._add_check_row(contents, 1, "include_girders", "Include girder/frame beams", self.include_girders)
        self._add_check_row(contents, 2, "include_end_lids", "Top/bottom lid", self.include_end_lids)

        buttons = ttk.Frame(left_panel)
        buttons.pack(fill=tk.X, pady=(0, 10))
        self.run_button = ttk.Button(buttons, text="Run FEM", command=self.run)
        self.run_button.pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(buttons, mode="indeterminate", length=140)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        ttk.Button(buttons, text="Close", command=self.window.destroy).pack(side=tk.RIGHT)

        status_frame = ttk.LabelFrame(left_panel, text="Run status")
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(status_frame, height=12, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        future_inputs = ttk.LabelFrame(mid_panel, text="Analysis options")
        future_inputs.pack(fill=tk.BOTH, expand=True)

        constraints = ttk.LabelFrame(future_inputs, text="Supports and load path")
        constraints.pack(fill=tk.X, padx=8, pady=(8, 6))
        self._configure_option_grid(constraints)
        self._add_option_row(constraints, 0, "boundary_condition", "Boundary", self.boundary_condition, ("auto", "free", "simply supported", "pinned", "clamped"))
        self._add_option_row(constraints, 1, "symmetry_mode", "Symmetry", self.symmetry_mode, ("none", "x", "y", "z", "cyclic"))
        self._add_option_row(constraints, 2, "pressure_direction", "Pressure dir.", self.pressure_direction, ("external", "internal"))
        self._add_entry_row(constraints, 3, "axial_force_n", "Axial force [N]", self.axial_force_n)
        self._add_entry_row(constraints, 4, "enforced_displacement_m", "Enforced disp. [m]", self.enforced_displacement_m)

        solver_options = ttk.LabelFrame(future_inputs, text="Solver")
        solver_options.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._configure_option_grid(solver_options)
        self._add_option_row(solver_options, 0, "shell_element_order", "Shell element", self.shell_element_order, ("S4", "S8"))
        self._add_option_row(
            solver_options,
            1,
            "analysis_type",
            "Analysis",
            self.analysis_type,
            ("linear eigenvalue", "nonlinear stability", "geometric nonlinear static", "geom. + material nonlinear static"),
        )
        self._add_option_row(solver_options, 2, "buckling_analysis_type", "Buckling", self.buckling_analysis_type, ("linear eigenvalue", "nonlinear limit"))
        self._add_option_row(solver_options, 3, "solver_type", "Linear solver", self.solver_type, ("direct", "gmres", "minres", "bicgstab"))
        self._add_entry_row(solver_options, 4, "nonlinear_max_load_factor", "NL max LF", self.nonlinear_max_load_factor)
        self._add_entry_row(solver_options, 5, "nonlinear_steps", "NL steps", self.nonlinear_steps, width=8)
        self._add_entry_row(solver_options, 6, "nonlinear_max_iterations", "NL iterations", self.nonlinear_max_iterations, width=8)
        self._add_entry_row(solver_options, 7, "nonlinear_tolerance", "NL tolerance", self.nonlinear_tolerance)

        members = ttk.LabelFrame(future_inputs, text="Member modelling")
        members.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._configure_option_grid(members)
        self._add_entry_row(members, 0, "stiffener_eccentricity_m", "Stf. ecc. [m]", self.stiffener_eccentricity_m)
        self._add_entry_row(members, 1, "girder_eccentricity_m", "Girder ecc. [m]", self.girder_eccentricity_m)
        self._add_option_row(members, 2, "member_orientation", "Member orient.", self.member_orientation, ("auto", "global Y", "global Z", "radial"))

        material = ttk.LabelFrame(future_inputs, text="Material and recovery")
        material.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._configure_option_grid(material)
        self._add_entry_row(material, 0, "stress_percentile", "Stress pct.", self.stress_percentile)
        self._add_option_row(material, 1, "material_model", "Material", self.material_model, ("linear elastic", "DNV-RP-C208 steel"))
        self._add_option_row(material, 2, "steel_grade", "Steel grade", self.steel_grade, ("S235", "S275", "S355", "S420", "S460"))
        self._add_option_row(
            material,
            3,
            "steel_thickness_class",
            "Steel class",
            self.steel_thickness_class,
            ("auto", "t <= 16", "16 < t <= 40", "40 < t <= 63", "63 < t <= 100"),
        )
        self._add_entry_row(material, 4, "elastic_modulus_gpa", "E [GPa]", self.elastic_modulus_gpa)
        self._add_entry_row(material, 5, "poisson_ratio", "Poisson", self.poisson_ratio)
        self._add_entry_row(material, 6, "yield_stress_mpa", "Yield [MPa]", self.yield_stress_mpa)
        self._add_entry_row(material, 7, "nonlinear_layers", "NL layers", self.nonlinear_layers, width=8)

        custom = ttk.LabelFrame(future_inputs, text="Custom loads and boundary conditions")
        custom.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._configure_option_grid(custom)
        self._add_check_row(custom, 0, "custom_load_bc_enabled", "Use custom load/BC mode", self.custom_load_bc_enabled)
        self._add_check_row(custom, 1, "custom_loads_add_to_imported", "Add custom loads to imported/generated loads", self.custom_loads_add_to_imported)
        self._add_check_row(custom, 2, "custom_use_nullspace_projection", "Use nullspace projection as boundary", self.custom_use_nullspace_projection)
        self._add_option_row(custom, 3, "plate_edge_supports", "Plate x0 / x1", self.plate_edge_x0_support, ("free", "simply supported", "fixed"))
        self._add_option_row(custom, 4, "plate_edge_supports", "Plate y0 / y1", self.plate_edge_y0_support, ("free", "simply supported", "fixed"))
        ttk.OptionMenu(custom, self.plate_edge_x1_support, self.plate_edge_x1_support.get(), "free", "simply supported", "fixed").grid(row=3, column=3, sticky=tk.EW, padx=(0, 8), pady=4)
        ttk.OptionMenu(custom, self.plate_edge_y1_support, self.plate_edge_y1_support.get(), "free", "simply supported", "fixed").grid(row=4, column=3, sticky=tk.EW, padx=(0, 8), pady=4)
        self._add_option_row(custom, 5, "cylinder_end_supports", "Cyl. lower / upper", self.cylinder_lower_support, ("free", "simply supported", "fixed"))
        ttk.OptionMenu(custom, self.cylinder_upper_support, self.cylinder_upper_support.get(), "free", "simply supported", "fixed").grid(row=5, column=3, sticky=tk.EW, padx=(0, 8), pady=4)
        self._add_entry_row(custom, 6, "plate_edge_loads", "Plate x0 / x1 [N/m]", self.plate_edge_x0_load_n_per_m)
        ttk.Entry(custom, textvariable=self.plate_edge_x1_load_n_per_m, width=12).grid(row=6, column=3, sticky=tk.EW, padx=(0, 8), pady=4)
        self._add_entry_row(custom, 7, "plate_edge_loads", "Plate y0 / y1 [N/m]", self.plate_edge_y0_load_n_per_m)
        ttk.Entry(custom, textvariable=self.plate_edge_y1_load_n_per_m, width=12).grid(row=7, column=3, sticky=tk.EW, padx=(0, 8), pady=4)
        self._add_entry_row(custom, 8, "cylinder_edge_loads", "Cyl. lower / upper [N/m]", self.cylinder_lower_edge_load_n_per_m)
        ttk.Entry(custom, textvariable=self.cylinder_upper_edge_load_n_per_m, width=12).grid(row=8, column=3, sticky=tk.EW, padx=(0, 8), pady=4)
        custom.columnconfigure(3, weight=1)

        preview = ttk.LabelFrame(right_panel, text="3D section view")
        preview.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self._show_preview_figure(create_runtime_fem_geometry_preview_figure(self.snapshot, self.app), preview)

        result_frame = ttk.LabelFrame(right_panel, text="Run visualization")
        result_frame.pack(fill=tk.BOTH, expand=True)

        plot_holder = ttk.Frame(result_frame)
        plot_holder.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.figure_parent = plot_holder
        selector_bar = ttk.Frame(plot_holder)
        selector_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        self._info_button(selector_bar, "display_choice").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(selector_bar, text="Display").pack(side=tk.LEFT, padx=(0, 6))
        self.display_selector = ttk.Combobox(
            selector_bar,
            textvariable=self.display_choice,
            state="readonly",
            values=tuple(self.display_mode_labels),
            width=34,
        )
        self.display_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.display_selector.bind("<<ComboboxSelected>>", lambda _event: self._refresh_figure())
        self._show_figure(create_runtime_fem_result_figure(self.snapshot), plot_holder)
        self._write_status("Ready. ANYstructure production FE mesh solver is available.")

    def _show_figure(self, figure: Figure, parent: Any | None = None) -> None:
        if parent is None and self.figure_canvas is not None:
            parent = self.figure_canvas.get_tk_widget().master
        if parent is None:
            return
        if self.figure_canvas is not None:
            self.figure_canvas.get_tk_widget().destroy()
            self.figure_canvas = None
        if self.figure_toolbar is not None:
            self.figure_toolbar.destroy()
            self.figure_toolbar = None
        if self.figure_toolbar_frame is not None:
            self.figure_toolbar_frame.destroy()
            self.figure_toolbar_frame = None

        toolbar_frame = ttk.Frame(parent)
        self.figure_toolbar_frame = toolbar_frame
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.figure_canvas = FigureCanvasTkAgg(figure, master=parent)
        self.figure_toolbar = NavigationToolbar2Tk(self.figure_canvas, toolbar_frame, pack_toolbar=False)
        self.figure_toolbar.update()
        self.figure_toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    @staticmethod
    def _fit_preview_figure_to_canvas(figure: Figure, width: int, height: int) -> None:
        if width < 80 or height < 80:
            return
        figure.set_size_inches(width / figure.dpi, height / figure.dpi, forward=False)
        try:
            figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.96)
        except Exception:
            pass
        zoom = 2.25
        if min(width, height) > 340:
            zoom = 2.75
        if min(width, height) > 520:
            zoom = 3.25
        for axis in figure.axes:
            if not hasattr(axis, "get_zlim"):
                continue
            axis.set_position([-0.08, -0.12, 1.16, 1.16])
            axis.margins(0.0)
            try:
                axis.set_anchor("C")
            except Exception:
                pass
            try:
                axis.set_proj_type("ortho")
            except Exception:
                pass
            try:
                extents = RuntimeFEMWindow._preview_axis_data_extents(axis)
                if extents is not None:
                    x_limits, y_limits, z_limits = extents
                    x_span_raw = max(abs(x_limits[1] - x_limits[0]), 1.0e-6)
                    y_span_raw = max(abs(y_limits[1] - y_limits[0]), 1.0e-6)
                    z_span_raw = max(abs(z_limits[1] - z_limits[0]), 1.0e-6)
                    pad = 0.04 * max(x_span_raw, y_span_raw, z_span_raw)
                    x_mid = 0.5 * (x_limits[0] + x_limits[1])
                    y_mid = 0.5 * (y_limits[0] + y_limits[1])
                    z_mid = 0.5 * (z_limits[0] + z_limits[1])
                    axis.set_xlim3d(x_mid - 0.5 * x_span_raw - pad, x_mid + 0.5 * x_span_raw + pad)
                    axis.set_ylim3d(y_mid - 0.5 * y_span_raw - pad, y_mid + 0.5 * y_span_raw + pad)
                    axis.set_zlim3d(z_mid - 0.5 * z_span_raw - pad, z_mid + 0.5 * z_span_raw + pad)
                x_limits = axis.get_xlim3d()
                y_limits = axis.get_ylim3d()
                z_limits = axis.get_zlim3d()
                x_span = max(abs(x_limits[1] - x_limits[0]), 1.0e-6)
                y_span = max(abs(y_limits[1] - y_limits[0]), 1.0e-6)
                z_span = max(abs(z_limits[1] - z_limits[0]), 1.0e-6)
                axis.set_box_aspect((x_span, y_span, z_span), zoom=zoom)
            except TypeError:
                try:
                    axis.set_box_aspect((x_span, y_span, z_span))
                except Exception:
                    pass
            except Exception:
                pass

    @staticmethod
    def _preview_axis_data_extents(axis: Any) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []

        def add(values: Any, target: list[float]) -> None:
            try:
                arr = np.asarray(values, dtype=float).reshape(-1)
            except Exception:
                return
            arr = arr[np.isfinite(arr)]
            target.extend(float(value) for value in arr)

        for line in getattr(axis, "lines", []):
            try:
                x_data, y_data, z_data = line.get_data_3d()
            except Exception:
                continue
            add(x_data, xs)
            add(y_data, ys)
            add(z_data, zs)
        for collection in getattr(axis, "collections", []):
            segments = getattr(collection, "_segments3d", None)
            if segments is not None:
                for segment in segments:
                    arr = np.asarray(segment, dtype=float)
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                        add(arr[:, 0], xs)
                        add(arr[:, 1], ys)
                        add(arr[:, 2], zs)
                continue
            vec = getattr(collection, "_vec", None)
            if vec is not None:
                arr = np.asarray(vec, dtype=float)
                if arr.ndim == 2 and arr.shape[0] >= 3:
                    add(arr[0], xs)
                    add(arr[1], ys)
                    add(arr[2], zs)
        if not xs or not ys or not zs:
            return None
        return ((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)))

    def _show_preview_figure(self, figure: Figure, parent: Any) -> None:
        if self.preview_canvas is not None:
            self.preview_canvas.get_tk_widget().destroy()
            self.preview_canvas = None

        self.preview_canvas = FigureCanvasTkAgg(figure, master=parent)
        self.preview_canvas.draw()
        widget = self.preview_canvas.get_tk_widget()
        redraw_after_id = {"value": None}

        def resize_preview(event: Any) -> None:
            if event.width < 80 or event.height < 80:
                return
            if redraw_after_id["value"] is not None:
                try:
                    widget.after_cancel(redraw_after_id["value"])
                except Exception:
                    pass

            def redraw() -> None:
                redraw_after_id["value"] = None
                if self.preview_canvas is None:
                    return
                canvas_widget = self.preview_canvas.get_tk_widget()
                if canvas_widget.winfo_width() < 80 or canvas_widget.winfo_height() < 80:
                    return
                try:
                    self._fit_preview_figure_to_canvas(
                        figure,
                        canvas_widget.winfo_width(),
                        canvas_widget.winfo_height(),
                    )
                    self.preview_canvas.draw_idle()
                except Exception:
                    pass

            try:
                redraw_after_id["value"] = widget.after(80, redraw)
            except Exception:
                pass

        widget.bind("<Configure>", resize_preview)
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)
        try:
            widget.after(120, lambda: (
                self.preview_canvas is not None
                and self._fit_preview_figure_to_canvas(figure, widget.winfo_width(), widget.winfo_height()) is None
                and self.preview_canvas.draw_idle()
            ))
        except Exception:
            pass

    def _selected_display_mode(self) -> str:
        return self.display_mode_labels.get(str(self.display_choice.get()), "static")

    def _set_display_modes(self, result: RuntimeFEMRunResult) -> None:
        labels = {"Static displacement/stress": "static"}
        if (result.visualization or {}).get("plastic_strain"):
            labels["Engineering plastic strain"] = "plastic"
        for mode in _buckling_mode_shapes(result):
            mode_number = int(mode.get("mode_number", 0))
            load_factor = _safe_float(mode.get("load_factor"))
            label = "Mode " + str(mode_number) + "  LF " + str(round(load_factor, 4))
            labels[label] = "mode:" + str(mode_number)
        self.display_mode_labels = labels
        self.display_choice.set("Static displacement/stress")
        if self.display_selector is not None:
            self.display_selector.configure(values=tuple(labels))

    def _refresh_figure(self) -> None:
        if self.figure_parent is None:
            return
        self._show_figure(
            create_runtime_fem_result_figure(
                self.snapshot,
                self.current_result,
                self._selected_display_mode(),
                max(_safe_float(self.deformation_scale.get(), 0.0), 0.0),
            ),
            self.figure_parent,
        )

    def _write_status(self, text: str) -> None:
        if self.result_text is None:
            return
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)

    def _set_solver_running(self, is_running: bool) -> None:
        if self.run_button is not None:
            self.run_button.configure(state=tk.DISABLED if is_running else tk.NORMAL)
        if self.progress_bar is not None:
            if is_running:
                self.progress_bar.start(12)
            else:
                self.progress_bar.stop()

    def _options(self) -> RuntimeFEMOptions:
        return RuntimeFEMOptions(
            mesh_fidelity=str(self.mesh_fidelity.get()),
            pressure_pa=_safe_float(self.pressure_pa.get()),
            load_scale=_safe_float(self.load_scale.get(), 1.0),
            include_stiffeners=bool(self.include_stiffeners.get()),
            include_girders=bool(self.include_girders.get()),
            include_end_lids=bool(self.include_end_lids.get()),
            num_buckling_modes=max(_safe_int(self.num_buckling_modes.get(), 5), 1),
            mesh_size_m=max(_safe_float(self.mesh_size_m.get(), 0.0), 0.0),
            top_bottom_moment_nm=_safe_float(self.top_bottom_moment_nm.get(), 0.0),
            boundary_condition=str(self.boundary_condition.get()),
            symmetry_mode=str(self.symmetry_mode.get()),
            shell_element_order=str(self.shell_element_order.get()),
            analysis_type=str(self.analysis_type.get()),
            buckling_analysis_type=str(self.buckling_analysis_type.get()),
            pressure_direction=str(self.pressure_direction.get()),
            axial_force_n=_safe_float(self.axial_force_n.get(), 0.0),
            enforced_displacement_m=_safe_float(self.enforced_displacement_m.get(), 0.0),
            stiffener_eccentricity_m=_safe_float(self.stiffener_eccentricity_m.get(), 0.0),
            girder_eccentricity_m=_safe_float(self.girder_eccentricity_m.get(), 0.0),
            member_orientation=str(self.member_orientation.get()),
            solver_type=str(self.solver_type.get()),
            stress_percentile=min(max(_safe_float(self.stress_percentile.get(), 95.0), 0.0), 100.0),
            elastic_modulus_pa=max(_safe_float(self.elastic_modulus_gpa.get(), 210.0), 1.0e-9) * 1.0e9,
            poisson_ratio=min(max(_safe_float(self.poisson_ratio.get(), 0.3), 0.0), 0.49),
            yield_stress_pa=max(_safe_float(self.yield_stress_mpa.get(), 355.0), 0.0) * 1.0e6,
            material_model=str(self.material_model.get()),
            steel_grade=str(self.steel_grade.get()),
            steel_thickness_class=str(self.steel_thickness_class.get()),
            nonlinear_max_load_factor=max(_safe_float(self.nonlinear_max_load_factor.get(), 3.0), 1.0e-9),
            nonlinear_steps=max(_safe_int(self.nonlinear_steps.get(), 12), 1),
            nonlinear_max_iterations=max(_safe_int(self.nonlinear_max_iterations.get(), 25), 1),
            nonlinear_tolerance=max(_safe_float(self.nonlinear_tolerance.get(), 1.0e-6), 1.0e-12),
            nonlinear_layers=_nearest_nonlinear_layer_count(self.nonlinear_layers.get()),
            deformation_scale=max(_safe_float(self.deformation_scale.get(), 0.0), 0.0),
            custom_load_bc_enabled=bool(self.custom_load_bc_enabled.get()),
            custom_loads_add_to_imported=bool(self.custom_loads_add_to_imported.get()),
            custom_use_nullspace_projection=bool(self.custom_use_nullspace_projection.get()),
            plate_edge_x0_support=str(self.plate_edge_x0_support.get()),
            plate_edge_x1_support=str(self.plate_edge_x1_support.get()),
            plate_edge_y0_support=str(self.plate_edge_y0_support.get()),
            plate_edge_y1_support=str(self.plate_edge_y1_support.get()),
            cylinder_lower_support=str(self.cylinder_lower_support.get()),
            cylinder_upper_support=str(self.cylinder_upper_support.get()),
            plate_edge_x0_load_n_per_m=_safe_float(self.plate_edge_x0_load_n_per_m.get(), 0.0),
            plate_edge_x1_load_n_per_m=_safe_float(self.plate_edge_x1_load_n_per_m.get(), 0.0),
            plate_edge_y0_load_n_per_m=_safe_float(self.plate_edge_y0_load_n_per_m.get(), 0.0),
            plate_edge_y1_load_n_per_m=_safe_float(self.plate_edge_y1_load_n_per_m.get(), 0.0),
            cylinder_lower_edge_load_n_per_m=_safe_float(self.cylinder_lower_edge_load_n_per_m.get(), 0.0),
            cylinder_upper_edge_load_n_per_m=_safe_float(self.cylinder_upper_edge_load_n_per_m.get(), 0.0),
        )

    def run(self) -> None:
        """Prepare/run the runtime FEM request and render Matplotlib results."""

        if self.solver_thread is not None and self.solver_thread.is_alive():
            return
        if not self.include_stiffeners.get() and not self.include_girders.get():
            messagebox.showwarning("FEM solver", "At least one member beam family should normally be included.")

        options = self._options()
        self._set_solver_running(True)
        self._write_status("Running FEM solver...\n\n" + "The result plot will update when the solver finishes.")

        def worker() -> None:
            try:
                self.solver_queue.put((run_runtime_fem(self.snapshot, options), None))
            except Exception as error:
                self.solver_queue.put((None, error))

        self.solver_thread = threading.Thread(target=worker, daemon=True)
        self.solver_thread.start()
        self.window.after(100, self._poll_solver_result)

    def _poll_solver_result(self) -> None:
        try:
            result, error = self.solver_queue.get_nowait()
        except queue.Empty:
            if self.solver_thread is not None and self.solver_thread.is_alive():
                self.window.after(100, self._poll_solver_result)
                return
            self._set_solver_running(False)
            return

        self._set_solver_running(False)
        if error is not None:
            self._write_status("Runtime FEM failed:\n" + str(error))
            messagebox.showerror("FEM solver", str(error))
            return

        self.current_result = result
        self._set_display_modes(result)
        self._write_status(format_runtime_fem_result(result))
        self._refresh_figure()


def open_runtime_fem_window(parent: Any, app: Any) -> RuntimeFEMWindow | None:
    """Open the experimental runtime FEM popup for the app active line."""

    try:
        return RuntimeFEMWindow(parent, app)
    except Exception as error:
        messagebox.showinfo("Experimental FEM solver", str(error))
        return None


class _ExamplePlate:
    def get_structure_type(self):
        return "Flat plate, stiffened with girder"

    def get_span(self):
        return 2.8

    def get_s(self):
        return 0.72

    def get_pl_thk(self):
        return 0.012


class _ExampleAllStructure:
    Plate = _ExamplePlate()
    Stiffener = object()
    Girder = object()


class _ExampleShell:
    radius = 2.0
    length_of_shell = 8.0
    tot_cyl_length = 8.0
    thk = 0.012
    cone_r1 = None
    cone_r2 = None
    cone_length = None
    _dist_between_rings = 2.0


class _ExampleLongitudinalStiffener:
    stiffener_type = "FB"
    spacing = 0.5
    hw = 0.150
    tw = 0.010
    b = 0.0
    tf = 0.0


class _ExampleRingGirder:
    stiffener_type = "T"
    hw = 0.400
    tw = 0.010
    b = 0.150
    tf = 0.020


class _ExampleCylinder:
    geometry = 7
    ShellObj = _ExampleShell()
    LongStfObj = _ExampleLongitudinalStiffener()
    RingStfObj = None
    RingFrameObj = _ExampleRingGirder()
    length_between_girders = 4.0


class _ExampleTkVariable:
    def __init__(self, value: Any):
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


class _ExampleRuntimeApp:
    _general_color = "#f0f0f0"
    _fem_default_top_bottom_moment_nm = 30_000_000.0
    _active_line = "line_example"
    _line_is_active = True
    _line_dict = {"line_example": [1, 2]}
    _line_to_struc = {"line_example": [_ExampleAllStructure(), None, None, object(), None, _ExampleCylinder()]}
    _simplified_calculation_mode = True

    def __init__(self):
        self._new_prop_3d_opposite_side = _ExampleTkVariable(False)
        self._new_shell_ring_frame_length_between_girders = _ExampleTkVariable(4.0)
        self._new_shell_dist_rings = _ExampleTkVariable(2.0)
        self._new_panel_length_Lp = _ExampleTkVariable(0.0)
        self._new_girder_length_LG = _ExampleTkVariable(8.0)

    def __getattr__(self, name: str) -> Any:
        try:
            from anystruct.main_application import Application
        except ModuleNotFoundError:
            from ANYstructure.anystruct.main_application import Application
        descriptor = Application.__dict__.get(name)
        if isinstance(descriptor, staticmethod):
            return descriptor.__func__
        if isinstance(descriptor, classmethod):
            return types.MethodType(descriptor.__func__, Application)
        attr = getattr(Application, name)
        if callable(attr):
            return types.MethodType(attr, self)
        return attr

    def get_highest_pressure(self, line):
        return {"normal": 100_000.0 if line == self._active_line else 0.0}


def example_runtime_app() -> _ExampleRuntimeApp:
    """Return a tiny active-line app fixture for running this module directly."""

    return _ExampleRuntimeApp()


if __name__ == "__main__":
    root = tk.Tk()
    my_app = RuntimeFEMWindow(root, example_runtime_app(), use_parent_as_window=True)
    my_app.window.protocol("WM_DELETE_WINDOW", root.destroy)
    my_app.window.focus_force()
    root.mainloop()
