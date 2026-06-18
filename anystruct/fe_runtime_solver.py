"""Experimental runtime FEM solver window for active ANYstructure lines.

The module owns the active-line handoff, user options and result visualization
for the experimental full-geometry FEM mode.  It calls the ANYstructure-local
``anystruct.fe_solver`` module; solver development happens in ANYintelligent
and can later be copied into that local module without changing this GUI layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import argparse
import json
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

try:
    from anystruct import tkinter_3d_canvas_thickness_v6 as _tk3d_canvas_module
except ModuleNotFoundError:
    from ANYstructure.anystruct import tkinter_3d_canvas_thickness_v6 as _tk3d_canvas_module

Tkinter3DCanvas = _tk3d_canvas_module.Tkinter3DCanvas
Point3D = _tk3d_canvas_module.Point3D
_interpolate_thickness_color = _tk3d_canvas_module._interpolate_thickness_color


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
    beam_element_order: str = "B2"
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
    custom_loads_json: str = ""
    custom_pressure_patches_json: str = ""
    custom_edge_segments_json: str = ""
    custom_selected_edge_load_n_per_m: float = 0.0
    custom_time_domain_enabled: bool = False
    custom_time_domain_duration_s: float = 0.01
    custom_time_domain_total_time_s: float = 0.05
    custom_time_domain_dt_s: float = 0.0005
    custom_time_domain_include_static_load: bool = False
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


def _length_input_to_m(value: Any, default: float = 0.0) -> float:
    """Return a structural length in metres from saved-object or GUI-style values."""

    number = _safe_float(value, default)
    if number <= 0.0:
        return default
    return number / 1000.0 if number > 100.0 else number


def _flat_lg_from_objects(plate: Any, stiffener: Any, girder: Any, spacing: float) -> float:
    for obj in (girder, stiffener, plate):
        if obj is None:
            continue
        value = _read_attr_or_call(obj, "get_girder_lg", "get_lg", "get_LG", "girder_lg", "lg", "LG", default=None)
        lg = _length_input_to_m(value, 0.0)
        if lg > 1.0e-9:
            return lg
    return max(4.0 * spacing, 0.8)


def _flat_lp_from_structure(all_obj: Any, span: float, spacing: float, has_girder: bool) -> float:
    for obj in (all_obj,):
        if obj is None:
            continue
        value = _read_attr_or_call(obj, "panel_length_Lp", "_panel_length_Lp", "panel_length", "lp", "Lp", default=None)
        lp = _length_input_to_m(value, 0.0)
        if lp > 1.0e-9:
            return lp
    if has_girder:
        return max(2.0 * span, 2.0 * spacing, 0.8)
    return max(span, spacing, 0.8)


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
    if stiffener_type in {"T", "TEE",
                          "T-BAR"} and height > 0.0 and thickness > 0.0 and flange_width > 0.0 and flange_thickness > 0.0:
        web_area = height * thickness
        flange_area = flange_width * flange_thickness
        area = web_area + flange_area
        web_y = 0.5 * height
        flange_y = height + 0.5 * flange_thickness
        centroid = (web_area * web_y + flange_area * flange_y) / area
        iy = thickness * height ** 3 / 12.0 + web_area * (web_y - centroid) ** 2
        iy += flange_width * flange_thickness ** 3 / 12.0 + flange_area * (flange_y - centroid) ** 2
        iz = height * thickness ** 3 / 12.0 + flange_thickness * flange_width ** 3 / 12.0
        return {
            "area": area,
            "Iy": max(iy, 1.0e-12),
            "Iz": max(iz, 1.0e-12),
            "J": max(iy + iz, 1.0e-12),
            "shear_factor_y": 5.0 / 6.0,
            "shear_factor_z": 5.0 / 6.0,
            "web_height": height,
            "web_thickness": thickness,
            "flange_width": flange_width,
            "flange_thickness": flange_thickness,
            "label": "T" + str(round(height * 1000.0)) + "x" + str(round(thickness * 1000.0))
                     + "+" + str(round(flange_width * 1000.0)) + "x" + str(round(flange_thickness * 1000.0)),
        }

    if stiffener_type not in {"FB", "FLAT", "FLATBAR", "FLAT BAR"} and not (height > 0.0 and thickness > 0.0):
        return None
    if height <= 0.0 or thickness <= 0.0:
        return None

    area = height * thickness
    iy = thickness * height ** 3 / 12.0
    iz = height * thickness ** 3 / 12.0
    return {
        "area": area,
        "Iy": max(iy, 1.0e-12),
        "Iz": max(iz, 1.0e-12),
        "J": max(iy + iz, 1.0e-12),
        "shear_factor_y": 5.0 / 6.0,
        "shear_factor_z": 5.0 / 6.0,
        "web_height": height,
        "web_thickness": thickness,
        "flange_width": 0.0,
        "flange_thickness": 0.0,
        "label": "FB" + str(round(height * 1000.0)) + "x" + str(round(thickness * 1000.0)),
    }


def _plate_edge_supports_from_line_properties(plate: Any) -> tuple[str, str, str, str]:
    raw = _read_attr_or_call(plate, "get_puls_up_boundary", "_puls_up_boundary", "puls_up_boundary", default=None)
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    text = str(raw or "").strip().upper().replace(" ", "")
    if len(text) < 4 or any(letter not in {"S", "C"} for letter in text[:4]):
        return ("simply supported", "simply supported", "simply supported", "simply supported")
    return tuple("fixed" if letter == "C" else "simply supported" for letter in text[:4])  # type: ignore[return-value]


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
    span = _safe_float(_read_attr_or_call(plate, "get_span", "span"), 1.0)
    spacing = _safe_float(_read_attr_or_call(plate, "get_s", default=None), 1.0)
    has_girder = girder is not None
    length = _flat_lp_from_structure(all_obj, span, spacing, has_girder)
    width = _flat_lg_from_objects(plate, stiffener, girder, spacing) if has_girder else spacing
    girder_spacing = span if has_girder else _safe_float(
        _read_attr_or_call(girder, "get_s", "spacing", "s", default=None), 0.0)
    return {
        "geometry": "flat panel",
        "length_m": length,
        "width_m": width,
        "thickness_m": _safe_float(_read_attr_or_call(plate, "get_pl_thk", default=None), 0.0),
        "has_stiffener": stiffener is not None,
        "has_girder": has_girder,
        "stiffener_spacing_m": spacing,
        "girder_spacing_m": girder_spacing,
        "stiffener_span_m": span,
        "girder_length_m": width if has_girder else 0.0,
        "panel_length_m": length,
        "stiffener_section": _member_section(stiffener),
        "girder_section": _member_section(girder),
        "plate_edge_supports": _plate_edge_supports_from_line_properties(plate),
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
        "ring_spacing_m": _safe_float(
            _read_attr_or_call(shell, "_dist_between_rings", "dist_between_rings", default=None), 0.0),
        "girder_spacing_m": _safe_float(_read_attr_or_call(cyl_obj, "length_between_girders", default=None), 0.0),
        "stiffener_section": _member_section(stiffener),
        "girder_section": _member_section(girder),
    }


def runtime_geometry_summary(snapshot: RuntimeFEMLineSnapshot) -> dict[str, Any]:
    """Return compact geometry metadata for plotting and future solver handoff."""

    if snapshot.is_cylinder:
        return _cylinder_geometry_summary(snapshot)
    return _flat_geometry_summary(snapshot)


def _runtime_custom_load_entries(raw_json: str) -> list[dict[str, Any]]:
    try:
        raw_entries = json.loads(raw_json or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(raw_entries, list):
        return []
    return [entry for entry in raw_entries if isinstance(entry, dict)]


def _runtime_custom_pressure_patches(options: RuntimeFEMOptions) -> list[dict[str, float]]:
    entries = _runtime_custom_load_entries(options.custom_loads_json)
    raw_patches: list[Any] = []
    if entries:
        for entry in entries:
            if str(entry.get("type", "")).lower() not in {"pressure", "panel_pressure"}:
                continue
            patches = entry.get("patches", [])
            if isinstance(patches, list):
                raw_patches.extend(patches)
    else:
        try:
            raw_loaded = json.loads(options.custom_pressure_patches_json or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_loaded = []
        if isinstance(raw_loaded, list):
            raw_patches = raw_loaded

    patches: list[dict[str, float]] = []
    for raw_patch in raw_patches:
        if not isinstance(raw_patch, dict):
            continue
        patch = {
            "min_a": _safe_float(raw_patch.get("min_a")),
            "max_a": _safe_float(raw_patch.get("max_a")),
            "min_b": _safe_float(raw_patch.get("min_b")),
            "max_b": _safe_float(raw_patch.get("max_b")),
        }
        if patch["max_a"] > patch["min_a"] and patch["max_b"] > patch["min_b"]:
            patches.append(patch)
    return patches


def _runtime_custom_edges(options: RuntimeFEMOptions) -> list[dict[str, float | str]]:
    entries = _runtime_custom_load_entries(options.custom_loads_json)
    raw_edges: list[Any] = []
    if entries:
        for entry in entries:
            if str(entry.get("type", "")).lower() not in {"edge", "edge_load"}:
                continue
            edges = entry.get("edges", [])
            if isinstance(edges, list):
                raw_edges.extend(edges)
    else:
        try:
            raw_loaded = json.loads(options.custom_edge_segments_json or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_loaded = []
        if isinstance(raw_loaded, list):
            raw_edges = raw_loaded

    custom_edges: list[dict[str, float | str]] = []
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict):
            continue
        start = _safe_float(raw_edge.get("start_coordinate"))
        end = _safe_float(raw_edge.get("end_coordinate"))
        if abs(end - start) <= 1.0e-12:
            continue
        custom_edges.append({
            "varying_axis": str(raw_edge.get("varying_axis", "a")),
            "fixed_coordinate": _safe_float(raw_edge.get("fixed_coordinate")),
            "start_coordinate": min(start, end),
            "end_coordinate": max(start, end),
        })
    return custom_edges


def run_runtime_fem(snapshot: RuntimeFEMLineSnapshot, options: RuntimeFEMOptions,
                    status_callback=None) -> RuntimeFEMRunResult:
    """Run the ANYstructure-owned lightweight FEM solver."""

    geometry = runtime_geometry_summary(snapshot)
    diagnostics = list(snapshot.diagnostics)
    custom_load_entries = _runtime_custom_load_entries(options.custom_loads_json)
    listed_pressure = sum(
        abs(_safe_float(entry.get("pressure_pa"), 0.0))
        for entry in custom_load_entries
        if str(entry.get("type", "")).lower() in {"pressure", "panel_pressure"}
    )
    effective_pressure = float(
        options.pressure_pa if (not options.custom_load_bc_enabled or options.custom_loads_add_to_imported) else 0.0)
    if options.custom_load_bc_enabled:
        effective_pressure += float(listed_pressure if custom_load_entries else options.custom_pressure_pa)
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
        beam_element_order=options.beam_element_order,
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
        custom_pressure_pa=options.custom_pressure_pa,
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
        custom_loads_json=options.custom_loads_json,
        custom_pressure_patches_json=options.custom_pressure_patches_json,
        custom_edge_segments_json=options.custom_edge_segments_json,
        custom_selected_edge_load_n_per_m=options.custom_selected_edge_load_n_per_m,
        custom_time_domain_enabled=options.custom_time_domain_enabled,
        custom_time_domain_duration_s=options.custom_time_domain_duration_s,
        custom_time_domain_total_time_s=options.custom_time_domain_total_time_s,
        custom_time_domain_dt_s=options.custom_time_domain_dt_s,
        custom_time_domain_include_static_load=options.custom_time_domain_include_static_load,
        imperfection_enabled=options.imperfection_enabled,
        imperfection_shape=options.imperfection_shape,
        imperfection_amplitude_m=options.imperfection_amplitude_m,
        imperfection_wave_a=options.imperfection_wave_a,
        imperfection_wave_b=options.imperfection_wave_b,
        runtime_solver=options.runtime_solver,
        allow_unbalanced_free_free=options.allow_unbalanced_free_free,
        buckling_shift_load_factor=options.buckling_shift_load_factor,
        buckling_min_load_factor=options.buckling_min_load_factor,
        buckling_max_load_factor=options.buckling_max_load_factor,
        buckling_repeated_tolerance=options.buckling_repeated_tolerance,
        buckling_allow_dense_fallback=options.buckling_allow_dense_fallback,
        recovery_history_mode=options.recovery_history_mode,
        recovery_threads=options.recovery_threads,
        memory_limit_mb=options.memory_limit_mb,
        capacity_buckling_mode_number=options.capacity_buckling_mode_number,
        capacity_mesh_min_elements_per_half_wave=options.capacity_mesh_min_elements_per_half_wave,
    )
    if fe_solver.full_backend_available():
        solver_result = fe_solver.run_production_fem(geometry, solver_config, status_callback=status_callback)
        if solver_result.status in {"backend_unavailable", "invalid", "static_failed", "production_failed"}:
            fallback = fe_solver.run_lightweight_fem(geometry, solver_config, status_callback=status_callback)
            diagnostics.extend(solver_result.diagnostics)
            diagnostics.append("Production FE mesh failed; using compact fallback result.")
            solver_result = fallback
    else:
        solver_result = fe_solver.run_lightweight_fem(geometry, solver_config, status_callback=status_callback)
    diagnostics.extend(solver_result.diagnostics)

    custom_patches = _runtime_custom_pressure_patches(options)
    custom_edges = _runtime_custom_edges(options)

    summary = {
        **geometry,
        "line": snapshot.line_name,
        "mesh_fidelity": options.mesh_fidelity,
        "pressure_pa": effective_pressure * float(options.load_scale),
        "imported_pressure_pa": float(options.pressure_pa),
        "include_stiffeners": bool(options.include_stiffeners),
        "include_girders": bool(options.include_girders),
        "include_end_lids": bool(options.include_end_lids),
        "num_buckling_modes": int(options.num_buckling_modes),
        "mesh_size_m": float(options.mesh_size_m),
        "top_bottom_moment_nm": float(options.top_bottom_moment_nm),
        "boundary_condition": str(options.boundary_condition),
        "symmetry_mode": str(options.symmetry_mode),
        "shell_element_order": str(options.shell_element_order),
        "beam_element_order": str(options.beam_element_order),
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
        "custom_pressure_pa": float(options.custom_pressure_pa),
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
        "custom_loads_json": str(options.custom_loads_json),
        "custom_load_count": len(custom_load_entries),
        "custom_pressure_patches_json": str(options.custom_pressure_patches_json),
        "custom_edge_segments_json": str(options.custom_edge_segments_json),
        "custom_selected_edge_load_n_per_m": float(options.custom_selected_edge_load_n_per_m),
        "custom_pressure_patch_count": len(custom_patches),
        "custom_pressure_patch_area_m2": sum(
            max(0.0, patch["max_a"] - patch["min_a"])
            * max(0.0, patch["max_b"] - patch["min_b"])
            for patch in custom_patches
        ),
        "custom_edge_segment_count": len(custom_edges),
        "custom_time_domain_enabled": bool(options.custom_time_domain_enabled),
        "custom_time_domain_duration_s": float(options.custom_time_domain_duration_s),
        "custom_time_domain_total_time_s": float(options.custom_time_domain_total_time_s),
        "custom_time_domain_dt_s": float(options.custom_time_domain_dt_s),
        "custom_time_domain_include_static_load": bool(options.custom_time_domain_include_static_load),
        "imperfection_enabled": bool(options.imperfection_enabled),
        "imperfection_shape": str(options.imperfection_shape),
        "imperfection_amplitude_m": float(options.imperfection_amplitude_m),
        "imperfection_wave_a": int(options.imperfection_wave_a),
        "imperfection_wave_b": int(options.imperfection_wave_b),
        "runtime_solver": str(options.runtime_solver),
        "allow_unbalanced_free_free": bool(options.allow_unbalanced_free_free),
        "buckling_shift_load_factor": float(options.buckling_shift_load_factor),
        "buckling_min_load_factor": float(options.buckling_min_load_factor),
        "buckling_max_load_factor": float(options.buckling_max_load_factor),
        "buckling_repeated_tolerance": float(options.buckling_repeated_tolerance),
        "buckling_allow_dense_fallback": bool(options.buckling_allow_dense_fallback),
        "recovery_history_mode": str(options.recovery_history_mode),
        "recovery_threads": int(options.recovery_threads),
        "memory_limit_mb": float(options.memory_limit_mb),
        "capacity_buckling_mode_number": int(options.capacity_buckling_mode_number),
        "capacity_mesh_min_elements_per_half_wave": int(options.capacity_mesh_min_elements_per_half_wave),
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


def _clamped_alpha(value: Any, default: float = 1.0) -> float:
    return min(max(_safe_float(value, default), 0.0), 1.0)


def _alpha_to_stipple(alpha: float) -> str:
    """Approximate alpha in Tk Canvas, which has no polygon alpha channel."""

    alpha = _clamped_alpha(alpha, 1.0)
    if alpha >= 0.94:
        return ""
    if alpha >= 0.68:
        return "gray75"
    if alpha >= 0.43:
        return "gray50"
    if alpha >= 0.18:
        return "gray25"
    return "gray12"


def _blend_hex_color(color: str, alpha: float, background: str = "#ffffff") -> str:
    """Blend a solid Tk colour towards the background to emulate opacity."""

    alpha = _clamped_alpha(alpha, 1.0)
    try:
        foreground_rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
        background_rgb = np.asarray(mcolors.to_rgb(background), dtype=float)
        return mcolors.to_hex(alpha * foreground_rgb + (1.0 - alpha) * background_rgb)
    except Exception:
        return color


def _configure_tk_canvas_colormap(colormap: str) -> None:
    """Keep the interactive canvas surface colours and legend in sync."""

    name = str(colormap or "jet")
    cmap = colormaps.get(name, colormaps["jet"])
    existing = getattr(_tk3d_canvas_module, "_THICKNESS_COLOR_STOPS", ())
    positions = tuple(float(item[0]) for item in existing if isinstance(item, (tuple, list)) and len(item) >= 2)
    if len(positions) < 2:
        positions = (0.0, 0.18, 0.36, 0.52, 0.66, 0.78, 0.88, 0.95, 1.0)
    stops = tuple((position, mcolors.to_hex(cmap(position), keep_alpha=False)) for position in positions)
    try:
        setattr(_tk3d_canvas_module, "_THICKNESS_COLOR_STOPS", stops)
    except Exception:
        pass


def _surface_facecolors(values_grid: list[list[float]], colormap: str = "jet"):
    values = _all_grid_values(values_grid) or [0.0]
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values) if max(values) > min(values) else min(values) + 1.0)
    cmap = colormaps.get(colormap, colormaps["jet"])
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


def _plot_member_lines(
        axis: Any,
        visualization: dict[str, Any],
        scale: float,
        show_stiffeners: bool = True,
        show_girders: bool = True,
        member_alpha: float = 0.95,
) -> None:
    role_style = {
        "stiffener": ("#1f2937", 1.7),
        "girder": ("#7f1d1d", 2.2),
    }
    for line in visualization.get("member_lines") or ():
        role = str(line.get("role", "member")).lower()
        if role == "stiffener" and not show_stiffeners:
            continue
        if role == "girder" and not show_girders:
            continue

        points = list(line.get("points") or ())
        displaced = list(line.get("displaced_points") or ())
        if len(points) < 2:
            continue
        plot_points = []
        for index, point in enumerate(points):
            try:
                base = np.asarray(point, dtype=float)
                moved = np.asarray(displaced[index], dtype=float) if index < len(displaced) else base
            except Exception:
                continue
            plot_points.append(base + (moved - base) * float(scale))
        if len(plot_points) < 2:
            continue
        color, width = role_style.get(role, ("#475569", 1.4))
        for start, end in zip(plot_points, plot_points[1:]):
            axis.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                [start[2], end[2]],
                color=color,
                linewidth=width,
                alpha=member_alpha,
                solid_capstyle="round",
            )


def _buckling_mode_shapes(result: RuntimeFEMRunResult | None) -> list[dict[str, Any]]:
    if result is None:
        return []
    return list((result.visualization or {}).get("buckling_modes") or [])


def _selected_visualization(result: RuntimeFEMRunResult, display_mode: str, component: str = "von_mises_pa") -> tuple[
    dict[str, Any], str, bool]:
    if display_mode == "plastic":
        visualization = dict(result.visualization or {})
        if visualization.get("plastic_strain"):
            visualization["stress_pa"] = visualization.get("plastic_strain")
            visualization["scalar_label"] = visualization.get(
                "plastic_strain_label") or "equiv. engineering plastic strain [-]"
            visualization["scalar_kind"] = "raw"
            return visualization, "Engineering plastic strain", False

    def apply_component(vis: dict[str, Any], title: str) -> tuple[dict[str, Any], str]:
        fields = vis.get("fields", {})
        disps = vis.get("displacements", {})
        if component == "plastic_strain" and vis.get("plastic_strain"):
            vis["stress_pa"] = vis["plastic_strain"]
            vis["scalar_label"] = vis.get("plastic_strain_label", "equiv. engineering plastic strain [-]")
            vis["scalar_kind"] = "raw"
        elif component in fields:
            vis["stress_pa"] = fields[component]
            vis["scalar_label"] = component.replace("_", " ")
        elif component in disps:
            vis["stress_pa"] = disps[component]
            vis["scalar_label"] = component.replace("_", " ") + " [m]"
        return vis, title

    if display_mode.startswith("mode:"):
        try:
            mode_number = int(display_mode.split(":", 1)[1])
        except (IndexError, ValueError):
            mode_number = -1
        for mode in _buckling_mode_shapes(result):
            if int(mode.get("mode_number", -1)) == mode_number:
                factor = _safe_float(mode.get("load_factor"))
                title = "Buckling mode " + str(mode_number) + "  LF=" + str(round(factor, 4))
                vis, _ = apply_component(dict(mode.get("shape") or {}), title)
                return vis, title, True

    vis, title = apply_component(dict(result.visualization or {}), "Static stress/displacement")
    return vis, title, False


def _plot_visualization_surface(
        figure: Figure,
        axis: Any,
        geometry: dict[str, Any],
        result: RuntimeFEMRunResult,
        display_mode: str = "static",
        deformation_scale: float | None = None,
        show_plate: bool = True,
        show_stiffeners: bool = True,
        show_girders: bool = True,
        plate_alpha: float = 1.0,
        member_alpha: float = 0.95,
        colormap: str = "jet",
        component: str = "von_mises_pa",
) -> None:
    plate_alpha = _clamped_alpha(plate_alpha, 1.0)
    member_alpha = _clamped_alpha(member_alpha, 0.95)
    visualization, title, is_mode = _selected_visualization(result, display_mode, component)
    scalar_values = _plot_grid_values(visualization.get("stress_pa"))
    if is_mode:
        color_grid = scalar_values
        colorbar_label = str(visualization.get("scalar_label") or "mode amplitude")
    elif visualization.get("scalar_kind") == "raw":
        color_grid = scalar_values
        colorbar_label = str(visualization.get("scalar_label") or "value")
    else:
        if component.endswith("_pa"):
            color_grid = [[value / 1.0e6 for value in row] for row in scalar_values]
            colorbar_label = str(visualization.get("scalar_label", "stress")).replace("_pa", "") + " [MPa]"
        elif "disp" in component:
            color_grid = [[value * 1000.0 for value in row] for row in scalar_values]
            colorbar_label = str(visualization.get("scalar_label", "displacement")).replace(" [m]", " [mm]")
        else:
            color_grid = scalar_values
            colorbar_label = str(visualization.get("scalar_label", component))
    facecolors, norm, cmap = _surface_facecolors(color_grid, colormap)
    scale = _displacement_plot_scale(geometry, result, visualization, deformation_scale)

    if visualization.get("type") == "cylinder":
        axial = _plot_grid_values(visualization.get("axial_m"))
        theta = _plot_grid_values(visualization.get("theta_rad"))
        radius = max(_safe_float(visualization.get("radius_m"), _safe_float(geometry.get("radius_m"), 1.0)), 1.0e-9)

        disps = visualization.get("displacements", {})
        dx_grid = _plot_grid_values(disps.get("disp_x", []))
        dy_grid = _plot_grid_values(disps.get("disp_y", []))
        dz_grid = _plot_grid_values(disps.get("disp_z", []))

        if not dx_grid or not dy_grid or not dz_grid:
            radial_displacement = _plot_grid_values(visualization.get("radial_displacement_m"))
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
            z = axial
        else:
            x = [
                [(radius * math.cos(theta[row_index][col_index]) + dx_grid[row_index][col_index] * scale)
                 for col_index in range(len(theta[row_index]))]
                for row_index in range(len(theta))
            ]
            y = [
                [(radius * math.sin(theta[row_index][col_index]) + dy_grid[row_index][col_index] * scale)
                 for col_index in range(len(theta[row_index]))]
                for row_index in range(len(theta))
            ]
            z = [
                [(axial[row_index][col_index] + dz_grid[row_index][col_index] * scale)
                 for col_index in range(len(axial[row_index]))]
                for row_index in range(len(axial))
            ]

        if show_plate:
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
                alpha=plate_alpha,
            )
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("height [m]")
        _plot_member_lines(axis, visualization, scale, show_stiffeners, show_girders, member_alpha)
        _set_3d_axes_limits(axis, x, y, z)
        try:
            axis.view_init(elev=18.0, azim=-45.0)
        except Exception:
            pass
    else:
        x = _plot_grid_values(visualization.get("x_m"))
        y = _plot_grid_values(visualization.get("y_m"))
        w = _plot_grid_values(visualization.get("w_m"))
        z = [[value * scale for value in row] for row in w]
        if show_plate:
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
                alpha=_clamped_alpha(plate_alpha, 1.0),
            )
        axis.set_xlabel("length [m]")
        axis.set_ylabel("width [m]")
        axis.set_zlabel("w x" + str(round(scale, 1)))
        _plot_member_lines(axis, visualization, scale, show_stiffeners, show_girders, member_alpha)
        _set_3d_axes_limits(axis, x, y, z)

    axis.set_title(title)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(_all_grid_values(color_grid))
    figure.colorbar(mappable, ax=axis, shrink=0.68, pad=0.1, label=colorbar_label)


def _plot_base_geometry_surface(
        axis: Any,
        geometry: dict[str, Any],
        is_cylinder: bool,
        show_plate: bool = True,
        show_stiffeners: bool = True,
        show_girders: bool = True,
        plate_alpha: float = 1.0,
        member_alpha: float = 0.95,
) -> None:
    """Draw the unsolved model with the same visibility controls as results."""

    plate_alpha = _clamped_alpha(plate_alpha, 1.0)
    member_alpha = _clamped_alpha(member_alpha, 0.95)
    drew_anything = False

    if is_cylinder:
        radius = max(_safe_float(geometry.get("radius_m"), 1.0), 1.0e-6)
        length = max(_safe_float(geometry.get("length_m"), 1.0), 1.0e-6)
        theta = np.linspace(0.0, 2.0 * math.pi, 49)
        axial = np.linspace(-0.5 * length, 0.5 * length, 13)
        theta_grid, axial_grid = np.meshgrid(theta, axial)
        x_grid = radius * np.cos(theta_grid)
        y_grid = radius * np.sin(theta_grid)

        if show_plate and plate_alpha > 0.0:
            axis.plot_surface(
                x_grid,
                y_grid,
                axial_grid,
                color="#d8e2ea",
                edgecolor="#708090",
                linewidth=0.18,
                alpha=plate_alpha,
                antialiased=True,
                shade=False,
            )
            drew_anything = True

        if show_stiffeners and member_alpha > 0.0 and geometry.get("has_stiffener"):
            spacing = _safe_float(geometry.get("stiffener_spacing_m"), 0.0)
            if spacing > 0.0:
                count = max(1, int(round(2.0 * math.pi * radius / spacing)))
                section = geometry.get("stiffener_section") or {}
                web_height = max(_safe_float(section.get("web_height"), 0.1), 0.0)
                member_radius = max(radius - 0.5 * web_height, 1.0e-6)
                for index in range(count):
                    angle = 2.0 * math.pi * index / count
                    axis.plot(
                        [member_radius * math.cos(angle)] * 2,
                        [member_radius * math.sin(angle)] * 2,
                        [-0.5 * length, 0.5 * length],
                        color="#334155",
                        linewidth=1.3,
                        alpha=member_alpha,
                    )
                drew_anything = True

        if show_girders and member_alpha > 0.0 and geometry.get("has_girder"):
            spacing = _safe_float(geometry.get("girder_spacing_m"), 0.0)
            if spacing > 0.0:
                section = geometry.get("girder_section") or {}
                web_height = max(_safe_float(section.get("web_height"), 0.12), 0.0)
                member_radius = max(radius - 0.5 * web_height, 1.0e-6)
                ring_theta = np.linspace(0.0, 2.0 * math.pi, 97)
                count = max(1, int(math.floor(length / spacing)) + 1)
                for index in range(count):
                    z_position = -0.5 * length + index * spacing
                    if z_position > 0.5 * length + 1.0e-9:
                        break
                    axis.plot(
                        member_radius * np.cos(ring_theta),
                        member_radius * np.sin(ring_theta),
                        np.full_like(ring_theta, z_position),
                        color="#7f1d1d",
                        linewidth=1.8,
                        alpha=member_alpha,
                    )
                drew_anything = True

        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("height [m]")
        try:
            axis.set_box_aspect((2.0 * radius, 2.0 * radius, length))
            axis.view_init(elev=18.0, azim=-45.0)
        except Exception:
            pass
    else:
        length = max(_safe_float(geometry.get("length_m"), 1.0), 1.0e-6)
        width = max(_safe_float(geometry.get("width_m"), 1.0), 1.0e-6)
        x_grid, y_grid = np.meshgrid([0.0, length], [0.0, width])
        z_grid = np.zeros_like(x_grid)

        if show_plate and plate_alpha > 0.0:
            axis.plot_surface(
                x_grid,
                y_grid,
                z_grid,
                color="#d1d5db",
                edgecolor="#64748b",
                linewidth=0.3,
                alpha=plate_alpha,
                shade=False,
            )
            drew_anything = True

        if show_stiffeners and member_alpha > 0.0 and geometry.get("has_stiffener"):
            spacing = _safe_float(geometry.get("stiffener_spacing_m"), 0.0)
            section = geometry.get("stiffener_section") or {}
            web_height = max(_safe_float(section.get("web_height"), 0.1), 0.0)
            if spacing > 0.0:
                count = max(1, int(math.floor(width / spacing)) + 1)
                for index in range(count):
                    y_position = index * spacing
                    if y_position > width + 1.0e-9:
                        break
                    axis.plot(
                        [0.0, length],
                        [y_position, y_position],
                        [web_height, web_height],
                        color="#334155",
                        linewidth=1.5,
                        alpha=member_alpha,
                    )
                drew_anything = True

        if show_girders and member_alpha > 0.0 and geometry.get("has_girder"):
            section = geometry.get("girder_section") or {}
            web_height = max(_safe_float(section.get("web_height"), 0.15), 0.0)
            x_position = 0.5 * length
            axis.plot(
                [x_position, x_position],
                [0.0, width],
                [web_height, web_height],
                color="#7f1d1d",
                linewidth=2.0,
                alpha=member_alpha,
            )
            drew_anything = True

        axis.set_xlabel("length [m]")
        axis.set_ylabel("width [m]")
        axis.set_zlabel("height [m]")
        try:
            axis.set_box_aspect((length, width, max(0.18 * min(length, width), 0.1)))
            axis.view_init(elev=22.0, azim=-55.0)
        except Exception:
            pass

    axis.set_title("Base model geometry")
    if not drew_anything:
        axis.text2D(0.08, 0.56, "All model display items are hidden.", transform=axis.transAxes)


def create_runtime_fem_result_figure(
        snapshot: RuntimeFEMLineSnapshot,
        result: RuntimeFEMRunResult | None = None,
        display_mode: str = "static",
        deformation_scale: float | None = None,
        show_plate: bool = True,
        show_stiffeners: bool = True,
        show_girders: bool = True,
        plate_alpha: float = 1.0,
        member_alpha: float = 0.95,
        colormap: str = "jet",
        component: str = "von_mises_pa",
) -> Figure:
    """Create the Matplotlib result visualization used in the runtime popup."""

    figure = Figure(figsize=(8.0, 4.1), dpi=100)
    geometry_ax = figure.add_subplot(111, projection="3d")
    geometry = runtime_geometry_summary(snapshot) if result is None else result.summary

    if result is None or not result.visualization:
        _plot_base_geometry_surface(
            geometry_ax,
            geometry,
            snapshot.is_cylinder,
            show_plate=show_plate,
            show_stiffeners=show_stiffeners,
            show_girders=show_girders,
            plate_alpha=plate_alpha,
            member_alpha=member_alpha,
        )
    else:
        _plot_visualization_surface(
            figure, geometry_ax, geometry, result, display_mode, deformation_scale,
            show_plate=show_plate, show_stiffeners=show_stiffeners, show_girders=show_girders,
            plate_alpha=plate_alpha, member_alpha=member_alpha, colormap=colormap,
            component=component
        )

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
            axis.plot([x_mid, x_mid], [0.0, width], [web_height * 1.35, web_height * 1.35], color="#7f1d1d",
                      linewidth=2.0)
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
        "Beam element: " + str(summary.get("beam_element_order", "")),
        "Boundary condition: " + str(summary.get("boundary_condition", "")),
        "Symmetry: " + str(summary.get("symmetry_mode", "")),
        "Analysis type: " + str(summary.get("analysis_type", "")),
        "Buckling type: " + str(summary.get("buckling_analysis_type", "")),
        "Runtime solver: " + str(summary.get("runtime_solver", "stepwise")),
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
        "Steel grade/class: " + str(summary.get("steel_grade", "")) + " / " + str(
            summary.get("steel_thickness_class", "")),
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
        "Deformation plot scale: " + ("auto" if _safe_float(summary.get("deformation_scale"), 0.0) <= 0.0 else str(
            round(_safe_float(summary.get("deformation_scale")), 3))),
        "Recovery history: " + str(summary.get("recovery_history_mode", "full")),
        "Memory limit [MB]: " + ("none" if _safe_float(summary.get("memory_limit_mb"), 0.0) <= 0.0 else str(
            round(_safe_float(summary.get("memory_limit_mb")), 1))),
        "Custom load/BC mode: " + str(bool(summary.get("custom_load_bc_enabled"))),
        "Buckling modes: " + str(summary.get("num_buckling_modes", "")),
        "Max displacement [mm]: " + str(round(1000.0 * _safe_float(summary.get("max_displacement_m")), 4)),
    ]
    if summary.get("custom_load_bc_enabled"):
        lines.append(
            "Custom loads add to imported/generated loads: " + str(bool(summary.get("custom_loads_add_to_imported"))))
        lines.append("Custom nullspace boundary: " + str(bool(summary.get("custom_use_nullspace_projection"))))
        lines.append("Allow unbalanced free-free loads: " + str(bool(summary.get("allow_unbalanced_free_free"))))
        lines.append("Custom pressure [Pa]: " + str(round(_safe_float(summary.get("custom_pressure_pa")), 3)))
        lines.append("Selected pressure panels: " + str(_safe_int(summary.get("custom_pressure_patch_count"), 0)))
        lines.append("Selected pressure panel area [m2]: " + str(round(_safe_float(summary.get("custom_pressure_patch_area_m2")), 4)))
        lines.append("Selected edge segments: " + str(_safe_int(summary.get("custom_edge_segment_count"), 0)))
        lines.append("Selected edge load [N/m]: " + str(round(_safe_float(summary.get("custom_selected_edge_load_n_per_m")), 3)))
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
                        for key in ("plate_edge_x0_support", "plate_edge_x1_support", "plate_edge_y0_support",
                                    "plate_edge_y1_support")
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
    if summary.get("imperfection_enabled"):
        lines.extend(
            [
                "",
                "Geometric imperfection input:",
                " - shape: " + str(summary.get("imperfection_shape", "")),
                " - amplitude [mm]: "
                + ("standard default" if _safe_float(summary.get("imperfection_amplitude_m"), 0.0) <= 0.0 else str(
                    round(1000.0 * _safe_float(summary.get("imperfection_amplitude_m")), 4))),
                " - waves A/B: " + str(_safe_int(summary.get("imperfection_wave_a"), 1)) + " / " + str(
                    _safe_int(summary.get("imperfection_wave_b"), 1)),
            ]
        )
    if summary.get("custom_time_domain_enabled"):
        lines.extend(
            [
                "",
                "Custom time-domain input:",
                " - pressure [Pa]: " + str(round(_safe_float(summary.get("custom_pressure_pa")), 3)),
                " - duration / total time [s]: "
                + str(round(_safe_float(summary.get("custom_time_domain_duration_s")), 6))
                + " / "
                + str(round(_safe_float(summary.get("custom_time_domain_total_time_s")), 6)),
                " - dt [s]: " + str(round(_safe_float(summary.get("custom_time_domain_dt_s")), 8)),
                " - selected patches: " + str(_safe_int(summary.get("custom_pressure_patch_count"), 0)),
                " - selected patch area [m2]: " + str(round(_safe_float(summary.get("custom_pressure_patch_area_m2")), 4)),
                " - include static load in time domain: " + str(bool(summary.get("custom_time_domain_include_static_load"))),
            ]
        )
    if (
            str(summary.get("runtime_solver", "stepwise")).lower() != "stepwise"
            or _safe_float(summary.get("buckling_shift_load_factor"), 0.0) > 0.0
            or _safe_float(summary.get("buckling_min_load_factor"), 0.0) > 0.0
            or _safe_float(summary.get("buckling_max_load_factor"), 0.0) > 0.0
            or bool(summary.get("buckling_allow_dense_fallback"))
    ):
        lines.extend(
            [
                "",
                "Solver validity controls:",
                " - runtime path: " + str(summary.get("runtime_solver", "stepwise")),
                " - buckling shift/range: "
                + str(round(_safe_float(summary.get("buckling_shift_load_factor")), 4))
                + " / "
                + str(round(_safe_float(summary.get("buckling_min_load_factor")), 4))
                + "-"
                + str(round(_safe_float(summary.get("buckling_max_load_factor")), 4)),
                " - repeated-mode tolerance: " + str(
                    round(_safe_float(summary.get("buckling_repeated_tolerance"), 1.0e-3), 6)),
                " - dense fallback: " + str(bool(summary.get("buckling_allow_dense_fallback"))),
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
            " - beam order: " + str(mesh_info.get("beam_order", "")),
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
                lines.append(" - relative load imbalance: " + str(
                    round(_safe_float(prestress.get("relative_rigid_body_load_imbalance")), 6)))
                lines.append(
                    " - meaning: remaining rigid-body modes were projected out and any rigid-body load imbalance was carried as generalized balancing reactions.")
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
        imperfection_keys = {
            "imperfection_status",
            "imperfection_kind",
            "imperfection_amplitude_m",
            "imperfection_max_offset_m",
            "imperfection_waves_a",
            "imperfection_waves_b",
        }
        custom_time_domain_keys = {
            "custom_time_domain_status",
            "custom_time_domain_pressure_pa",
            "custom_time_domain_selected_shells",
            "custom_time_domain_peak_displacement_m",
            "custom_time_domain_peak_von_mises_pa",
        }
        runtime_keys = {
            "runtime_solver",
            "allow_unbalanced_free_free",
            "recovery_history_mode",
            "recovery_threads",
            "memory_limit_mb",
            "buckling_solver_status",
            "buckling_modes_returned",
            "buckling_repeated_groups",
            "buckling_shift_load_factor",
            "buckling_min_load_factor",
            "buckling_max_load_factor",
            "buckling_allow_dense_fallback",
            "capacity_workflow_status",
            "capacity_workflow_capacity_factor",
            "capacity_workflow_critical_load_factor",
            "capacity_workflow_mesh_status",
            "capacity_workflow_elements_per_half_wave",
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
            *imperfection_keys,
            *custom_time_domain_keys,
            *runtime_keys,
        }
        if prestress.get("material_model"):
            lines.extend(["", "DNV-RP-C208 material curve:"])
            lines.append(" - grade: " + str(prestress.get("steel_grade", "")))
            lines.append(" - thickness class: " + str(prestress.get("steel_thickness_class", "")))
            lines.append(" - sigma_prop/yield/yield2 [MPa]: " + " / ".join(
                str(round(_safe_float(prestress.get(key)) / 1.0e6, 3))
                for key in ("sigma_prop_pa", "sigma_yield_pa", "sigma_yield_2_pa")
            ))
            lines.append(" - eps_p_y1/eps_p_y2: " + str(prestress.get("eps_p_y1", "")) + " / " + str(
                prestress.get("eps_p_y2", "")))
            lines.append(
                " - K [MPa] / n: " + str(round(_safe_float(prestress.get("hardening_K_pa")) / 1.0e6, 3)) + " / " + str(
                    prestress.get("hardening_n", "")))
        runtime_solver = str(prestress.get("runtime_solver", summary.get("runtime_solver", "")) or "")
        if runtime_solver:
            lines.extend(["", "Runtime solver provenance:"])
            lines.append(" - runtime path: " + runtime_solver)
            lines.append(" - buckling status: " + str(prestress.get("buckling_solver_status", "")))
            lines.append(" - modes returned: " + str(_safe_int(prestress.get("buckling_modes_returned"), 0)))
            lines.append(" - repeated mode groups: " + str(_safe_int(prestress.get("buckling_repeated_groups"), 0)))
            if _safe_float(prestress.get("allow_unbalanced_free_free"), 0.0) > 0.0:
                lines.append(" - free-free load handling: explicit generalized balancing reactions allowed")
            lines.append(" - recovery history: " + str(
                prestress.get("recovery_history_mode", summary.get("recovery_history_mode", "full"))))
            if _safe_float(prestress.get("memory_limit_mb"), 0.0) > 0.0:
                lines.append(" - memory limit [MB]: " + str(round(_safe_float(prestress.get("memory_limit_mb")), 1)))
        capacity_status = str(prestress.get("capacity_workflow_status", "") or "")
        if capacity_status:
            lines.extend(["", "ANYintelligent capacity workflow:"])
            lines.append(" - status: " + capacity_status.replace("_", " "))
            lines.append(
                " - capacity factor: " + str(round(_safe_float(prestress.get("capacity_workflow_capacity_factor")), 4)))
            critical = _safe_float(prestress.get("capacity_workflow_critical_load_factor"), 0.0)
            lines.append(
                " - eigenvalue critical factor: " + ("not available" if critical <= 0.0 else str(round(critical, 4))))
            lines.append(" - mode mesh adequacy: " + str(prestress.get("capacity_workflow_mesh_status", "")))
            lines.append(" - active elements per estimated half-wave: " + str(
                round(_safe_float(prestress.get("capacity_workflow_elements_per_half_wave")), 3)))
            lines.append(
                " - meaning: linear static prestress, eigenmode buckling, stress-free imperfection and nonlinear static capacity were run as one traceable workflow.")
        nonlinear_static_status = str(prestress.get("nonlinear_static_status", "") or "")
        if nonlinear_static_status:
            lines.extend(["", "Incremental nonlinear static solve:"])
            lines.append(" - status: " + nonlinear_static_status.replace("_", " "))
            lines.append(" - last converged load factor: " + str(
                round(_safe_float(prestress.get("nonlinear_static_load_factor")), 4)))
            lines.append(" - completed steps: " + str(_safe_int(prestress.get("nonlinear_static_steps"), 0)))
            lines.append(
                " - Newton iterations: " + str(_safe_int(prestress.get("nonlinear_static_total_iterations"), 0)))
            lines.append(" - through-thickness layers: " + str(_safe_int(prestress.get("nonlinear_static_layers"), 0)))
            lines.append(" - max equivalent plastic strain: " + str(
                round(_safe_float(prestress.get("nonlinear_static_max_plastic_strain")), 6)))
            if nonlinear_static_status == "completed":
                lines.append(
                    " - interpretation: all requested proportional load was reached; this is not necessarily a collapse load.")
            elif nonlinear_static_status == "stopped_at_limit":
                lines.append(
                    " - interpretation: the adaptive Newton solve stopped at the last stable converged load increment.")
        imperfection_status = str(prestress.get("imperfection_status", "") or "")
        if imperfection_status:
            lines.extend(["", "Applied geometric imperfection:"])
            lines.append(" - status: " + imperfection_status)
            lines.append(" - kind: " + str(prestress.get("imperfection_kind", "")))
            lines.append(" - input amplitude [mm]: " + str(
                round(1000.0 * _safe_float(prestress.get("imperfection_amplitude_m")), 4)))
            lines.append(" - max offset [mm]: " + str(
                round(1000.0 * _safe_float(prestress.get("imperfection_max_offset_m")), 4)))
            lines.append(" - waves A/B: " + str(_safe_int(prestress.get("imperfection_waves_a"), 0)) + " / " + str(
                _safe_int(prestress.get("imperfection_waves_b"), 0)))
            lines.append(
                " - meaning: the coordinates were offset before solving, so zero displacement in the imperfect model is stress free.")
        custom_time_domain_status = str(prestress.get("custom_time_domain_status", "") or "")
        if custom_time_domain_status:
            lines.extend(["", "Custom time-domain response:"])
            lines.append(" - status: " + custom_time_domain_status)
            lines.append(" - selected shell elements: " + str(_safe_int(prestress.get("custom_time_domain_selected_shells"), 0)))
            lines.append(" - peak displacement [mm]: " + str(
                round(1000.0 * _safe_float(prestress.get("custom_time_domain_peak_displacement_m")), 4)))
            lines.append(" - peak von Mises [MPa]: " + str(
                round(_safe_float(prestress.get("custom_time_domain_peak_von_mises_pa")) / 1.0e6, 3)))
            lines.append(
                " - meaning: this is a linear Newmark response to the prescribed pressure pulse; it is reported separately from the static buckling prestress.")
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
                    lines.append(
                        " - explanation: the initial tangent stiffness was not positive for the selected prestress state.")
                elif steps == 0:
                    lines.append(" - explanation: the nonlinear check stopped before the first load increment.")
                else:
                    lines.append(" - explanation: no usable limit point was found in the configured load-step range.")
    load_resultant = summary.get("load_resultant") or {}
    if load_resultant:
        force = load_resultant.get("force_n", (0.0, 0.0, 0.0))
        lines.extend(["", "Load resultant force [N]: " + ", ".join(
            str(round(_safe_float(component), 3)) for component in force)])
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
    "custom_time_domain_enabled": {
        "title": "Custom Time-Domain Load",
        "purpose": "Runs the synced ANYintelligent linear time-domain pressure-patch solver for a custom load pulse.",
        "use": "When enabled, a prescribed shell-normal pressure pulse is applied to the selected shell patch and advanced with Newmark average acceleration. This is a separate transient response calculation after the normal static solve.",
        "output": "Adds custom load status, selected shell count, peak transient displacement and peak transient von Mises stress to the result print.",
        "caution": "This is prescribed structural response only: no fluid-structure interaction, added mass, water entry, cavitation or pressure feedback is included.",
    },
    "custom_pressure_pa": {
        "title": "Custom Load Pressure",
        "purpose": "Peak pressure magnitude for the transient custom load pulse.",
        "use": "The pressure sign follows the selected pressure direction. The pulse starts at t = 0 and remains constant until the custom load duration.",
        "output": "Controls the transient impulse, peak displacement and transient stress response.",
        "caution": "Do not also include the same pressure as a static pressure unless that is the intended load history.",
    },
    "custom_time_domain_duration_s": {
        "title": "Custom Load Duration",
        "purpose": "Length of the constant pressure pulse in seconds.",
        "use": "The transient load is active from t = 0 to this time, then returns to zero.",
        "output": "Controls impulse and dynamic amplification.",
        "caution": "Use a time step small enough to resolve the pulse duration.",
    },
    "custom_time_domain_total_time_s": {
        "title": "Custom Load Total Time",
        "purpose": "Total transient analysis time.",
        "use": "The solver saves the response until this time, including the free vibration after the pulse has ended.",
        "output": "Controls how much of the transient response is searched for peak displacement and stress.",
        "caution": "Long runs with small dt increase runtime.",
    },
    "custom_time_domain_dt_s": {
        "title": "Custom Load Time Step",
        "purpose": "Fixed Newmark time step.",
        "use": "The transient solver reuses the effective stiffness factorization when possible, but the number of steps still scales with total_time / dt.",
        "output": "Affects transient accuracy and runtime.",
        "caution": "The Newmark method is stable for the default parameters, but a coarse time step can miss the pressure pulse and peak response.",
    },
    "custom_pressure_patch_center": {
        "title": "Custom Pressure Patch Centre",
        "purpose": "Patch centre coordinates used to select shell elements by centroid.",
        "use": "For flat panels, A is x and B is y. For cylinders, A is axial z and B is circumferential arc length measured from the positive X direction. Zero uses the model centre for A and the positive X seam for B.",
        "output": "Changes which shell elements receive the transient custom load pulse.",
        "caution": "Selection is centroid based. If exact patch membership is required, verify selected shell count and plot resolution.",
    },
    "custom_pressure_patch_size": {
        "title": "Custom Pressure Patch Size",
        "purpose": "Patch dimensions used for centroid-based shell selection.",
        "use": "For flat panels, A/B are x/y dimensions. For cylinders, A is axial length and B is circumferential arc length. If either value is zero, all shell elements are loaded.",
        "output": "Changes the loaded area, impulse and transient response.",
        "caution": "Very small patches may select no element on a coarse mesh; the wrapper then falls back to loading all shells to avoid a silent zero-load run.",
    },
    "custom_time_domain_include_static_load": {
        "title": "Custom Load Base Load",
        "purpose": "Adds the current static load vector as a constant base load in the transient custom load run.",
        "use": "Leave off to study the custom load pulse alone. Enable when the transient pressure is intentionally superposed on the static load case.",
        "output": "Changes transient displacement, stress and impulse resultants.",
        "caution": "This can double count pressure if the static pressure already represents the same custom load event.",
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
    "imperfection_enabled": {
        "title": "Geometric Imperfection",
        "purpose": "Applies a stress-free initial geometry offset before the static/nonlinear solve.",
        "use": "Flat panels use a sinusoidal plate half-wave over the shell region. Cylinders use a radial out-of-roundness field. An amplitude of zero uses the standard default scale.",
        "output": "Changes static stress recovery, nonlinear response and buckling factors because the reference geometry is imperfect.",
        "caution": "This is a geometric imperfection, not a residual-stress field. Verify the selected shape and amplitude before using nonlinear capacity results.",
    },
    "imperfection_shape": {
        "title": "Imperfection Shape",
        "purpose": "Chooses the standard imperfection family applied to the generated shell model.",
        "use": "The current runtime option maps flat panels to plate half-wave imperfections and cylinders to radial out-of-roundness. Future solver work can add eigenmode-scaled imperfections.",
        "output": "Affects the offset field applied to nodes before the solve.",
        "caution": "Member-only bows are not yet separately exposed in this runtime GUI; the shape is intended as a panel/cylinder equivalent imperfection.",
    },
    "imperfection_amplitude_m": {
        "title": "Imperfection Amplitude",
        "purpose": "Maximum geometric imperfection amplitude in metres.",
        "use": "Enter a positive value to force the amplitude. Enter zero to use the standard default in the wrapper: s/200 for flat plate mode or spacing/200 for cylinder radial mode.",
        "output": "Reported in the result print together with the actual maximum applied nodal offset.",
        "caution": "Capacity can be sensitive to this value. Use code/rule tolerance values or calibrated amplitudes for final assessments.",
    },
    "imperfection_waves": {
        "title": "Imperfection Waves",
        "purpose": "Number of half waves in the generated imperfection field.",
        "use": "For flat panels, A/B are x/y half-wave counts. For cylinders, A is circumferential waves and B is axial half waves.",
        "output": "Changes the shape of the imperfect reference geometry.",
        "caution": "Use wave counts consistent with the expected buckling mode or rule requirement.",
    },
    "num_buckling_modes": {
        "title": "Buckling Modes",
        "purpose": "Number of positive buckling factors/mode shapes requested.",
        "use": "The eigensolver returns the lowest positive modes available from the recovered prestress state.",
        "output": "Controls the number of mode entries in the display selector and load-factor table.",
        "caution": "More modes require more solver work and may include local modes that are not design governing.",
    },
    "runtime_solver": {
        "title": "Runtime Solver",
        "purpose": "Selects the high-level runtime path used after the FE model and loads are generated.",
        "use": "Stepwise keeps the familiar ANYstructure sequence: linear static, prestress recovery, optional nonlinear solve and buckling. ANYintelligent capacity workflow runs the new traceable solver-wide sequence: linear static, eigenvalue buckling, stress-free imperfection and nonlinear static capacity in one workflow.",
        "output": "The result print records the selected path, workflow status, capacity factor and mesh-mode adequacy when the capacity workflow is selected.",
        "caution": "The capacity workflow is intentionally opt-in because it can be slower and applies a different capacity interpretation than the default static/eigenvalue result.",
    },
    "buckling_shift_load_factor": {
        "title": "Buckling Shift",
        "purpose": "Optional shift target for the sparse buckling eigensolver.",
        "use": "A positive value asks the backend to search near that load factor. Zero uses the default lowest-positive-factor search.",
        "output": "Affects which buckling modes are returned and is recorded in the solver provenance.",
        "caution": "Use only when you understand the expected load-factor range; an unsuitable shift can hide lower modes.",
    },
    "buckling_load_factor_range": {
        "title": "Buckling Range",
        "purpose": "Optional accepted load-factor interval for buckling modes.",
        "use": "Positive lower and/or upper values filter returned eigenmodes. Zero means open-ended.",
        "output": "Controls which modes appear in the display selector and load-factor table.",
        "caution": "Filtering is useful for numerical searches but can remove physically relevant low modes.",
    },
    "buckling_repeated_tolerance": {
        "title": "Repeated Mode Tolerance",
        "purpose": "Tolerance used to group nearly repeated buckling factors.",
        "use": "The backend marks close eigenvalues as repeated-mode groups for validity diagnostics.",
        "output": "The result print reports the number of repeated mode groups found.",
        "caution": "Repeated modes are common in symmetric cylinders and panels; they are not automatically an error.",
    },
    "buckling_allow_dense_fallback": {
        "title": "Dense Buckling Fallback",
        "purpose": "Allows a dense eigenvalue fallback when the sparse buckling solve cannot return useful modes.",
        "use": "Leave off for normal runs. Enable for small models when diagnosing sparse solver behaviour.",
        "output": "May return modes for small reduced systems that sparse search did not resolve.",
        "caution": "Dense fallback is not suitable for large models and may consume significant memory.",
    },
    "recovery_history_mode": {
        "title": "Recovery History",
        "purpose": "Controls how much transient/nonlinear history data the backend is allowed to retain.",
        "use": "Full keeps all selected histories, selected limits recovery scope, and envelope stores peak/envelope style histories where supported.",
        "output": "Recorded in result provenance and passed to transient custom load recovery policy.",
        "caution": "This is mostly a memory-management control; static stress plots still recover the full current field needed for the display.",
    },
    "recovery_threads": {
        "title": "Recovery Threads",
        "purpose": "Optional thread count for backend result-recovery phases that support measured parallel recovery.",
        "use": "Zero lets the backend choose the default serial path. Positive values request that many recovery workers.",
        "output": "Can change recovery timing; result values should remain deterministic.",
        "caution": "Small models may run slower with threads due to overhead.",
    },
    "memory_limit_mb": {
        "title": "Memory Limit",
        "purpose": "Optional soft memory limit for recovery/resource-policy checks.",
        "use": "Zero disables the limit. Positive values are converted to bytes and passed to backend resource policy objects.",
        "output": "If supported recovery would exceed the limit, the backend can reject the run before allocating large histories.",
        "caution": "This is a guardrail, not a full process memory sandbox.",
    },
    "capacity_buckling_mode_number": {
        "title": "Capacity Mode Number",
        "purpose": "Selects which eigenmode seeds the capacity workflow imperfection.",
        "use": "Mode 1 normally means the lowest positive buckling factor. Higher values can be used to study a known local/global mode.",
        "output": "Affects the stress-free imperfection used by the ANYintelligent capacity workflow.",
        "caution": "Choosing a non-governing mode can produce a non-conservative or misleading capacity estimate.",
    },
    "capacity_mesh_min_elements_per_half_wave": {
        "title": "Mode Mesh Adequacy",
        "purpose": "Minimum target number of active elements per estimated half-wave in the selected buckling mode.",
        "use": "The capacity workflow estimates active half-waves and reports whether the mesh is coarse for the selected mode.",
        "output": "Printed as mode mesh adequacy and active elements per estimated half-wave.",
        "caution": "This is a coarse screening metric, not a replacement for mesh convergence.",
    },
    "boundary_condition": {
        "title": "Boundary Condition",
        "purpose": "Defines the generated global support assumptions.",
        "use": "Flat automatic mode always gives pressure-loaded plating physical edge supports: line-property edge supports are used when available, otherwise all four edges default to simply supported. Cylinders keep the existing rigid-body anchor/lid behaviour unless custom mode is used.",
        "output": "Affects stiffness, displacement, stress recovery and buckling factors.",
        "caution": "Nullspace projection is only a numerical gauge, not the plate bending support. For manual flat-plate support studies, use custom load/BC mode.",
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
        "use": "S4 is faster. S8 adds shared midside nodes and uses higher-order shell interpolation in the core solver. S8R is the reduced integration version of S8.",
        "output": "Mesh diagnostics report the shell order. S8 usually increases node count and runtime. S8R significantly speeds up S8 by reducing integration points.",
        "caution": "S8 can improve curvature and bending representation, but it should be verified with mesh convergence. S8R exhibits hourglass modes, so use with caution for unconstrained models.",
    },
    "beam_element_order": {
        "title": "Beam Element",
        "purpose": "Selects 2-node or 3-node beam elements for generated stiffeners, girders and frames.",
        "use": "B2 keeps the current two-node Timoshenko beam mesh. B3 inserts beam-direction mid nodes and emits quadratic three-node beam elements.",
        "output": "Mesh diagnostics report the beam order. B3 increases beam interpolation order and adds nodes along member lines.",
        "caution": "B3 is useful for member curvature studies but adds degrees of freedom. Use B2 for faster routine runs.",
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
        "use": "When enabled, plate side supports or cylinder end supports are taken only from the custom fields below. By default the imported pressure, axial force and end moment inputs are not used; enter manual pressure and edge loads here instead. Tick the additive-load option if the custom loads should be added to those imported/generated loads.",
        "output": "The run summary prints the custom support choices and edge loads. Diagnostics show that custom mode is active.",
        "caution": "This mode can easily create free-free or over-constrained models. Check the nullspace projection status and load resultant after each run.",
    },
    "custom_loads_add_to_imported": {
        "title": "Add To Imported Loads",
        "purpose": "Controls whether custom edge loads replace or supplement the pressure and other generated load inputs.",
        "use": "Unchecked means manual pressure and custom edge loads are the complete load case. Checked means imported/generated pressure, axial force and top/bottom moment are applied first, then manual pressure and custom edge loads are added.",
        "output": "Changes load resultant, stress recovery and buckling prestress.",
        "caution": "Use this deliberately. Leaving it unchecked is safest when studying a clean custom load path.",
    },
    "custom_pressure_pa": {
        "title": "Manual Pressure",
        "purpose": "Pressure value used inside custom load/BC mode.",
        "use": "When custom mode is active and additive loads are unchecked, this pressure replaces the imported pressure from the active line. When additive loads are checked, it is added to the imported/generated pressure.",
        "output": "Affects shell pressure loads, displacement, stress recovery and buckling prestress.",
        "caution": "Manual mode assumes the user has evaluated whether pressure, edge loads and supports form the intended load case.",
    },
    "custom_use_nullspace_projection": {
        "title": "Nullspace Boundary",
        "purpose": "Uses rigid-body nullspace projection as the boundary condition for custom load/BC mode.",
        "use": "No explicit support edges are applied. The solver projects out rigid-body motion and reports any rigid-body load imbalance as balancing generalized reactions.",
        "output": "The result print reports nullspace rank and relative rigid-body load imbalance.",
        "caution": "This is a mathematical free-body gauge, not a physical support. It is useful for understanding self-equilibrated loads and free-body behaviour.",
    },
    "allow_unbalanced_free_free": {
        "title": "Allow Free-Free Imbalance",
        "purpose": "Explicitly allows the backend to solve a free-free or nullspace-gauged static problem even when loads are not self-equilibrated.",
        "use": "When enabled, rigid-body generalized load components are not rejected; they are reported as balancing generalized reactions. Selecting the nullspace boundary also enables this behaviour because the user has explicitly chosen a mathematical free-body gauge.",
        "output": "The result print reports relative rigid-body load imbalance and notes that generalized balancing reactions were allowed.",
        "caution": "This does not create a physical support. For pressure-loaded flat plates, define real edge supports unless the goal is a deliberate free-body load-path study.",
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
            try:
                self.window.group(parent)
            except Exception:
                pass
        try:
            self.window.attributes("-toolwindow", False)
        except Exception:
            pass

        self.mesh_fidelity = tk.StringVar(value="coarse")
        self.mesh_size_m = tk.DoubleVar(value=0.0)
        self.load_scale = tk.DoubleVar(value=1.0)
        self.pressure_pa = tk.DoubleVar(value=self.snapshot.pressure_pa)
        self.top_bottom_moment_nm = tk.DoubleVar(
            value=_safe_float(getattr(app, "_fem_default_top_bottom_moment_nm", 0.0)))
        self.include_stiffeners = tk.BooleanVar(value=True)
        self.include_girders = tk.BooleanVar(value=True)
        self.include_end_lids = tk.BooleanVar(value=bool(self.snapshot.is_cylinder))
        self.num_buckling_modes = tk.IntVar(value=5)
        self.boundary_condition = tk.StringVar(value="auto")
        self.symmetry_mode = tk.StringVar(value="none")
        self.shell_element_order = tk.StringVar(value="S4")
        self.beam_element_order = tk.StringVar(value="B2")
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
        self.deformation_scale = tk.StringVar(value="0.0")
        self.custom_load_bc_enabled = tk.BooleanVar(value=False)
        self.custom_loads_add_to_imported = tk.BooleanVar(value=False)
        self.custom_use_nullspace_projection = tk.BooleanVar(value=False)
        self.custom_pressure_pa = tk.DoubleVar(value=0.0)
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
        self.custom_time_domain_enabled = tk.BooleanVar(value=False)
        self.custom_time_domain_duration_s = tk.DoubleVar(value=0.01)
        self.custom_time_domain_total_time_s = tk.DoubleVar(value=0.05)
        self.custom_time_domain_dt_s = tk.DoubleVar(value=0.0005)
        self.custom_loads_json = tk.StringVar(value="[]")
        self.custom_pressure_patches_json = tk.StringVar(value="[]")
        self.custom_edge_segments_json = tk.StringVar(value="[]")
        self.custom_selected_edge_load_n_per_m = tk.DoubleVar(value=0.0)
        self._custom_load_entries: list[dict[str, Any]] = []
        self._custom_selected_edge_keys: set[tuple[str, float, float, float]] = set()
        self._custom_load_tree: ttk.Treeview | None = None
        self._custom_load_tree_scrollbar: ttk.Scrollbar | None = None
        self._custom_load_trace_ids: list[tuple[tk.Variable, str]] = []
        self._custom_load_patches: list[dict] = []
        self._custom_load_manual_cuts: list[dict[str, Any]] = []
        self._custom_load_selected_index: int = -1
        self._custom_load_selection_active = False
        self._custom_load_click_origin: tuple[int, int] | None = None
        self._custom_load_edge_click_origin: tuple[int, int] | None = None
        self._custom_load_selection_button = None
        self._generate_default_custom_load_patches()
        self.custom_time_domain_include_static_load = tk.BooleanVar(value=False)
        self.imperfection_enabled = tk.BooleanVar(value=False)
        self.imperfection_shape = tk.StringVar(value="standard plate/cylinder")
        self.imperfection_amplitude_m = tk.DoubleVar(value=0.0)
        self.imperfection_wave_a = tk.IntVar(value=1)
        self.imperfection_wave_b = tk.IntVar(value=1)
        self.runtime_solver = tk.StringVar(value="stepwise")
        self.allow_unbalanced_free_free = tk.BooleanVar(value=False)
        self.buckling_shift_load_factor = tk.DoubleVar(value=0.0)
        self.buckling_min_load_factor = tk.DoubleVar(value=0.0)
        self.buckling_max_load_factor = tk.DoubleVar(value=0.0)
        self.buckling_repeated_tolerance = tk.DoubleVar(value=1.0e-3)
        self.buckling_allow_dense_fallback = tk.BooleanVar(value=False)
        self.recovery_history_mode = tk.StringVar(value="full")
        self.recovery_threads = tk.IntVar(value=0)
        self.memory_limit_mb = tk.DoubleVar(value=0.0)
        self.capacity_buckling_mode_number = tk.IntVar(value=1)
        self.capacity_mesh_min_elements_per_half_wave = tk.IntVar(value=4)
        self.result_case_choice = tk.StringVar(value="Static displacement/stress")
        self.result_case_labels: dict[str, str] = {"Static displacement/stress": "static"}
        self.component_choice = tk.StringVar(value="von_mises_pa")
        self.component_labels: dict[str, str] = {
            "Stress von Mises": "von_mises_pa",
            "Displacement Magnitude": "disp_mag",
            "Displacement X": "disp_x",
            "Displacement Y": "disp_y",
            "Displacement Z": "disp_z",
            "Stress X (membrane)": "stress_x_membrane_pa",
            "Stress Y (membrane)": "stress_y_membrane_pa",
            "Stress XY (membrane)": "stress_xy_membrane_pa",
            "Strain X (membrane)": "strain_x_membrane",
            "Strain Y (membrane)": "strain_y_membrane",
            "Strain XY (membrane)": "strain_xy_membrane",
            "Equivalent Plastic Strain": "plastic_strain",
        }
        self.current_result: RuntimeFEMRunResult | None = None
        self.result_text = None
        self.figure_canvas = None
        self.figure_toolbar = None
        self.figure_toolbar_frame = None
        self.preview_canvas = None
        self.figure_parent = None
        self.result_case_selector = None
        self.component_selector = None
        self.run_button = None
        self.cancel_button = None
        self._cancel_requested = False
        self.progress_bar = None
        self.result_canvas = None
        self.use_interactive_3d = tk.BooleanVar(value=True)
        self.show_plate_vis = tk.BooleanVar(value=True)
        self.show_stiffener_vis = tk.BooleanVar(value=True)
        self.show_girder_vis = tk.BooleanVar(value=True)
        self.plate_alpha_vis = tk.StringVar(value="1.0")
        self.member_alpha_vis = tk.StringVar(value="0.95")
        self.colormap_vis = tk.StringVar(value="jet")
        self.upper_result_frame = None
        self.upper_result_text = None
        self.solver_thread = None
        self.solver_queue = queue.Queue()
        self._plot_refresh_after_id: str | None = None
        self._plot_trace_ids: list[tuple[tk.Variable, str]] = []
        self._force_fit_next_refresh = True
        self._display_base_geometry = True

        self._build()
        self._bind_plot_configuration_traces()
        self._show_as_normal_maximizable_window()

    def _bind_plot_configuration_traces(self) -> None:
        """Redraw both the base model and solved result when plot options change."""

        variables = (
            self.deformation_scale,
            self.show_plate_vis,
            self.show_stiffener_vis,
            self.show_girder_vis,
            self.plate_alpha_vis,
            self.member_alpha_vis,
            self.colormap_vis,
        )
        for variable in variables:
            try:
                trace_id = variable.trace_add("write", self._schedule_plot_refresh)
                self._plot_trace_ids.append((variable, trace_id))
            except Exception:
                pass

    def _schedule_plot_refresh(self, *_args: Any) -> None:
        """Debounce entry edits so typing does not rebuild the 3D scene per key."""

        if self.figure_parent is None:
            return
        if self._plot_refresh_after_id is not None:
            try:
                self.window.after_cancel(self._plot_refresh_after_id)
            except Exception:
                pass
            self._plot_refresh_after_id = None
        try:
            self._plot_refresh_after_id = self.window.after(90, self._run_scheduled_plot_refresh)
        except Exception:
            self._plot_refresh_after_id = None

    def _run_scheduled_plot_refresh(self) -> None:
        self._plot_refresh_after_id = None
        try:
            self._refresh_figure(preserve_view=True)
        except tk.TclError:
            pass

    def _show_as_normal_maximizable_window(self) -> None:
        """Keep the solver as a normal window with maximize controls available."""

        try:
            self.window.update_idletasks()
        except Exception:
            pass
        try:
            self.window.state("zoomed")
            return
        except Exception:
            pass
        try:
            self.window.attributes("-zoomed", True)
            return
        except Exception:
            pass
        try:
            screen_width = max(int(self.window.winfo_screenwidth()), 1100)
            screen_height = max(int(self.window.winfo_screenheight()), 760)
            self.window.geometry(str(screen_width - 80) + "x" + str(screen_height - 100) + "+20+20")
        except Exception:
            pass

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
        self._add_option_row(mesh_loads, 0, "mesh_fidelity", "Mesh fidelity", self.mesh_fidelity,
                             ("coarse", "medium", "fine", "very fine"))
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
        self.cancel_button = ttk.Button(buttons, text="Stop", command=self.cancel_run, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=(4, 0))
        self.progress_bar = ttk.Progressbar(buttons, mode="indeterminate", length=140)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        ttk.Button(buttons, text="Close", command=self.window.destroy).pack(side=tk.RIGHT)

        status_frame = ttk.LabelFrame(left_panel, text="Run status")
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(status_frame, height=12, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.options_notebook = ttk.Notebook(mid_panel)
        self.options_notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        tab_general = ttk.Frame(self.options_notebook)
        self.options_notebook.add(tab_general, text="General")

        tab_properties = ttk.Frame(self.options_notebook)
        self.options_notebook.add(tab_properties, text="Properties")

        tab_visualization = ttk.Frame(self.options_notebook)
        self.options_notebook.add(tab_visualization, text="Visualization")

        tab_advanced = ttk.Frame(self.options_notebook)
        self.options_notebook.add(tab_advanced, text="Advanced")

        constraints = ttk.LabelFrame(tab_general, text="Supports and load path")
        constraints.pack(fill=tk.X, padx=8, pady=(8, 6))
        self._configure_option_grid(constraints)
        self._add_option_row(constraints, 0, "boundary_condition", "Boundary", self.boundary_condition,
                             ("auto", "free", "simply supported", "pinned", "clamped"))
        self._add_option_row(constraints, 1, "symmetry_mode", "Symmetry", self.symmetry_mode,
                             ("none", "x", "y", "z", "cyclic"))
        self._add_option_row(constraints, 2, "pressure_direction", "Pressure dir.", self.pressure_direction,
                             ("external", "internal"))
        self._add_entry_row(constraints, 3, "axial_force_n", "Axial force [N]", self.axial_force_n)
        self._add_entry_row(constraints, 4, "enforced_displacement_m", "Enforced disp. [m]",
                            self.enforced_displacement_m)

        solver_options = ttk.LabelFrame(tab_general, text="Solver")
        solver_options.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._configure_option_grid(solver_options)
        self._add_option_row(
            solver_options,
            0,
            "runtime_solver",
            "Runtime path",
            self.runtime_solver,
            ("stepwise", "static only", "nonlinear static", "ANYintelligent capacity workflow"),
        )
        self._add_option_row(solver_options, 1, "shell_element_order", "Shell element", self.shell_element_order,
                             ("S4", "S8", "S8R"))
        self._add_option_row(solver_options, 2, "beam_element_order", "Beam element", self.beam_element_order,
                             ("B2", "B3"))
        self._add_option_row(
            solver_options,
            3,
            "analysis_type",
            "Analysis",
            self.analysis_type,
            ("linear eigenvalue", "nonlinear stability", "geometric nonlinear static",
             "geom. + material nonlinear static"),
        )
        self._add_option_row(solver_options, 4, "buckling_analysis_type", "Buckling", self.buckling_analysis_type,
                             ("linear eigenvalue", "nonlinear limit"))
        self._add_option_row(solver_options, 5, "solver_type", "Linear solver", self.solver_type,
                             ("direct", "gmres", "minres", "bicgstab"))
        self._add_entry_row(solver_options, 6, "nonlinear_max_load_factor", "NL max LF", self.nonlinear_max_load_factor)
        self._add_entry_row(solver_options, 7, "nonlinear_steps", "NL steps", self.nonlinear_steps, width=8)
        self._add_entry_row(solver_options, 8, "nonlinear_max_iterations", "NL iterations",
                            self.nonlinear_max_iterations, width=8)
        self._add_entry_row(solver_options, 9, "nonlinear_tolerance", "NL tolerance", self.nonlinear_tolerance)

        buckling_validity = ttk.LabelFrame(tab_general, text="Buckling validity and resources")
        buckling_validity.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._configure_option_grid(buckling_validity)
        self._add_entry_row(buckling_validity, 0, "buckling_shift_load_factor", "Buckling shift LF",
                            self.buckling_shift_load_factor, width=8)
        self._add_entry_row(buckling_validity, 1, "buckling_load_factor_range", "Buckling LF min/max",
                            self.buckling_min_load_factor, width=8)
        ttk.Entry(buckling_validity, textvariable=self.buckling_max_load_factor, width=8).grid(row=1, column=3,
                                                                                               sticky=tk.EW,
                                                                                               padx=(0, 8), pady=4)
        self._add_entry_row(buckling_validity, 2, "buckling_repeated_tolerance", "Repeated tol.",
                            self.buckling_repeated_tolerance, width=8)
        self._add_check_row(buckling_validity, 3, "buckling_allow_dense_fallback", "Allow dense buckling fallback",
                            self.buckling_allow_dense_fallback)
        self._add_option_row(buckling_validity, 4, "recovery_history_mode", "Recovery history",
                             self.recovery_history_mode, ("full", "selected", "envelope"))
        self._add_entry_row(buckling_validity, 5, "recovery_threads", "Recovery threads", self.recovery_threads,
                            width=8)
        self._add_entry_row(buckling_validity, 6, "memory_limit_mb", "Memory limit [MB]", self.memory_limit_mb, width=8)
        self._add_entry_row(buckling_validity, 7, "capacity_buckling_mode_number", "Capacity mode no.",
                            self.capacity_buckling_mode_number, width=8)
        self._add_entry_row(buckling_validity, 8, "capacity_mesh_min_elements_per_half_wave", "Mode elems/half-wave",
                            self.capacity_mesh_min_elements_per_half_wave, width=8)
        buckling_validity.columnconfigure(3, weight=1)

        members = ttk.LabelFrame(tab_properties, text="Member modelling")
        members.pack(fill=tk.X, padx=8, pady=(0, 6))
        self._configure_option_grid(members)
        self._add_entry_row(members, 0, "stiffener_eccentricity_m", "Stf. ecc. [m]", self.stiffener_eccentricity_m)
        self._add_entry_row(members, 1, "girder_eccentricity_m", "Girder ecc. [m]", self.girder_eccentricity_m)
        self._add_option_row(members, 2, "member_orientation", "Member orient.", self.member_orientation,
                             ("auto", "global Y", "global Z", "radial"))

        material = ttk.LabelFrame(tab_properties, text="Material and recovery")
        material.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._configure_option_grid(material)
        self._add_entry_row(material, 0, "stress_percentile", "Stress pct.", self.stress_percentile)
        self._add_option_row(material, 1, "material_model", "Material", self.material_model,
                             ("linear elastic", "DNV-RP-C208 steel"))
        self._add_option_row(material, 2, "steel_grade", "Steel grade", self.steel_grade,
                             ("S235", "S275", "S355", "S420", "S460"))
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

        imperfections = ttk.LabelFrame(tab_properties, text="Imperfections")
        imperfections.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._configure_option_grid(imperfections)
        self._add_check_row(imperfections, 0, "imperfection_enabled", "Use geometric imperfection",
                            self.imperfection_enabled)
        self._add_option_row(
            imperfections,
            1,
            "imperfection_shape",
            "Shape",
            self.imperfection_shape,
            ("standard plate/cylinder", "none"),
        )
        self._add_entry_row(imperfections, 2, "imperfection_amplitude_m", "Amplitude [m]",
                            self.imperfection_amplitude_m)
        self._add_entry_row(imperfections, 3, "imperfection_waves", "Waves A / B", self.imperfection_wave_a, width=8)
        ttk.Entry(imperfections, textvariable=self.imperfection_wave_b, width=8).grid(row=3, column=3, sticky=tk.EW,
                                                                                      padx=(0, 8), pady=4)
        imperfections.columnconfigure(3, weight=1)

        time_domain = ttk.LabelFrame(tab_advanced, text="Custom time-domain load")
        time_domain.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._configure_option_grid(time_domain)
        self._add_check_row(time_domain, 0, "custom_time_domain_enabled", "Run custom load in time domain", self.custom_time_domain_enabled)
        self._add_check_row(time_domain, 1, "custom_time_domain_include_static_load", "Include static load in time domain",
                            self.custom_time_domain_include_static_load)
        self._add_entry_row(time_domain, 2, "custom_pressure_pa", "Pressure [Pa]", self.custom_pressure_pa)
        self._add_entry_row(time_domain, 3, "custom_time_domain_duration_s", "Duration [s]", self.custom_time_domain_duration_s)
        self._add_entry_row(time_domain, 4, "custom_time_domain_total_time_s", "Total time [s]", self.custom_time_domain_total_time_s)
        self._add_entry_row(time_domain, 5, "custom_time_domain_dt_s", "dt [s]", self.custom_time_domain_dt_s)

        self.custom_load_area_var = tk.StringVar(value="Patch Area: 0.00 mÂ²")
        ttk.Label(time_domain, textvariable=self.custom_load_area_var).grid(row=6, column=0, sticky=tk.W, padx=8, pady=4)

        self._custom_load_selection_button = ttk.Button(
            time_domain,
            text="Start selection",
            command=self._toggle_custom_load_selection,
        )
        self._custom_load_selection_button.grid(row=6, column=1, sticky=tk.EW, padx=(0, 4), pady=4)
        ttk.Button(time_domain, text="Select All", command=self._custom_load_select_all).grid(
            row=6, column=2, sticky=tk.EW, padx=(0, 4), pady=4
        )
        ttk.Button(time_domain, text="Clear", command=self._custom_load_clear_all).grid(
            row=6, column=3, sticky=tk.EW, padx=(0, 8), pady=4
        )

        btn_frame = ttk.Frame(time_domain)
        btn_frame.grid(
            row=7,
            column=0,
            columnspan=4,
            sticky=tk.EW,
            padx=8,
            pady=4,
        )
        ttk.Button(
            btn_frame,
            text="Split A (Z/X)",
            command=lambda: self._custom_load_split_field("a"),
        ).pack(
            side=tk.LEFT,
            expand=True,
            fill=tk.X,
            padx=(0, 2),
        )
        ttk.Button(
            btn_frame,
            text="Split B (Arc/Y)",
            command=lambda: self._custom_load_split_field("b"),
        ).pack(
            side=tk.LEFT,
            expand=True,
            fill=tk.X,
            padx=(2, 0),
        )

        time_domain.columnconfigure(3, weight=1)

        custom = ttk.LabelFrame(tab_advanced, text="Custom loads and boundary conditions")
        custom.pack(fill=tk.X, padx=8, pady=(0, 8))
        self._configure_option_grid(custom)
        self._add_check_row(custom, 0, "custom_load_bc_enabled", "Use custom load/BC mode", self.custom_load_bc_enabled)
        self._add_check_row(custom, 1, "custom_loads_add_to_imported", "Add custom loads to imported/generated loads",
                            self.custom_loads_add_to_imported)
        self._add_check_row(custom, 2, "custom_use_nullspace_projection", "Use nullspace projection as boundary",
                            self.custom_use_nullspace_projection)
        self._add_check_row(custom, 3, "allow_unbalanced_free_free", "Allow unbalanced free-free loads",
                            self.allow_unbalanced_free_free)
        self._add_entry_row(custom, 4, "custom_pressure_pa", "Manual pressure [Pa]", self.custom_pressure_pa)
        self._add_option_row(custom, 5, "plate_edge_supports", "Plate x0 / x1", self.plate_edge_x0_support,
                             ("free", "simply supported", "fixed"))
        self._add_option_row(custom, 6, "plate_edge_supports", "Plate y0 / y1", self.plate_edge_y0_support,
                             ("free", "simply supported", "fixed"))
        ttk.OptionMenu(custom, self.plate_edge_x1_support, self.plate_edge_x1_support.get(), "free", "simply supported",
                       "fixed").grid(row=5, column=3, sticky=tk.EW, padx=(0, 8), pady=4)
        ttk.OptionMenu(custom, self.plate_edge_y1_support, self.plate_edge_y1_support.get(), "free", "simply supported",
                       "fixed").grid(row=6, column=3, sticky=tk.EW, padx=(0, 8), pady=4)
        self._add_option_row(custom, 7, "cylinder_end_supports", "Cyl. lower / upper", self.cylinder_lower_support,
                             ("free", "simply supported", "fixed"))
        ttk.OptionMenu(custom, self.cylinder_upper_support, self.cylinder_upper_support.get(), "free",
                       "simply supported", "fixed").grid(row=7, column=3, sticky=tk.EW, padx=(0, 8), pady=4)
        self._add_entry_row(custom, 8, "plate_edge_loads", "Plate x0 / x1 [N/m]", self.plate_edge_x0_load_n_per_m)
        ttk.Entry(custom, textvariable=self.plate_edge_x1_load_n_per_m, width=12).grid(row=8, column=3, sticky=tk.EW,
                                                                                       padx=(0, 8), pady=4)
        self._add_entry_row(custom, 9, "plate_edge_loads", "Plate y0 / y1 [N/m]", self.plate_edge_y0_load_n_per_m)
        ttk.Entry(custom, textvariable=self.plate_edge_y1_load_n_per_m, width=12).grid(row=9, column=3, sticky=tk.EW,
                                                                                       padx=(0, 8), pady=4)
        self._add_entry_row(custom, 10, "cylinder_edge_loads", "Cyl. lower / upper [N/m]",
                            self.cylinder_lower_edge_load_n_per_m)
        ttk.Entry(custom, textvariable=self.cylinder_upper_edge_load_n_per_m, width=12).grid(row=10, column=3,
                                                                                             sticky=tk.EW, padx=(0, 8),
                                                                                             pady=4)
        self._add_entry_row(custom, 11, "custom_selected_edge_load", "Selected edges [N/m]",
                            self.custom_selected_edge_load_n_per_m)
        custom.columnconfigure(3, weight=1)

        load_list = ttk.LabelFrame(tab_advanced, text="Loads to run")
        load_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        actions = ttk.Frame(load_list)
        actions.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(actions, text="Add load", command=self._add_custom_load_from_selection).pack(
            side=tk.LEFT,
            padx=(0, 4),
        )
        ttk.Button(actions, text="Delete load", command=self._delete_selected_custom_load).pack(side=tk.LEFT)
        columns = ("type", "value", "selection", "notes")
        self._custom_load_tree = ttk.Treeview(
            load_list,
            columns=columns,
            show="headings",
            height=9,
            selectmode="browse",
        )
        self._custom_load_tree.heading("type", text="Type")
        self._custom_load_tree.heading("value", text="Value")
        self._custom_load_tree.heading("selection", text="Selection")
        self._custom_load_tree.heading("notes", text="Boundary / notes")
        self._custom_load_tree.column("type", width=88, stretch=False, anchor=tk.W)
        self._custom_load_tree.column("value", width=110, stretch=False, anchor=tk.W)
        self._custom_load_tree.column("selection", width=180, stretch=True, anchor=tk.W)
        self._custom_load_tree.column("notes", width=240, stretch=True, anchor=tk.W)
        self._custom_load_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))
        self._custom_load_tree_scrollbar = ttk.Scrollbar(
            load_list,
            orient=tk.VERTICAL,
            command=self._custom_load_tree.yview,
        )
        self._custom_load_tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=(0, 8))
        self._custom_load_tree.configure(yscrollcommand=self._custom_load_tree_scrollbar.set)
        self._bind_custom_load_list_traces()
        self._refresh_custom_load_list()

        vis_group = ttk.LabelFrame(tab_visualization, text="Plot configuration")
        vis_group.pack(fill=tk.X, padx=8, pady=(8, 8))
        self._configure_option_grid(vis_group)
        self._add_check_row(vis_group, 0, "show_plate", "Show plate surface", self.show_plate_vis)
        self._add_check_row(vis_group, 1, "show_stiffeners", "Show stiffeners", self.show_stiffener_vis)
        self._add_check_row(vis_group, 2, "show_girders", "Show girders/frames", self.show_girder_vis)
        self._add_entry_row(vis_group, 3, "plate_alpha", "Plate alpha [0-1]", self.plate_alpha_vis, width=8)
        self._add_entry_row(vis_group, 4, "member_alpha", "Member alpha [0-1]", self.member_alpha_vis, width=8)
        self._add_option_row(vis_group, 5, "colormap", "Colormap", self.colormap_vis,
                             ("jet", "viridis", "plasma", "inferno", "coolwarm"))
        vis_actions = ttk.Frame(vis_group)
        vis_actions.grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=8, pady=4)
        ttk.Button(vis_actions, text="Redraw base 3D", command=self._redraw_base_3d).pack(side=tk.LEFT)
        ttk.Button(vis_actions, text="Show results", command=self._show_results).pack(side=tk.LEFT, padx=(6, 0))

        self.upper_result_frame = ttk.LabelFrame(right_panel, text="Result text")
        self.upper_result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.upper_result_text = tk.Text(self.upper_result_frame, wrap=tk.WORD, height=10)
        self.upper_result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        scrollbar = ttk.Scrollbar(self.upper_result_frame, orient=tk.VERTICAL, command=self.upper_result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        self.upper_result_text.configure(yscrollcommand=scrollbar.set)

        result_frame = ttk.LabelFrame(right_panel, text="Run visualization")
        result_frame.pack(fill=tk.BOTH, expand=True)

        plot_holder = ttk.Frame(result_frame)
        plot_holder.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.figure_parent = plot_holder
        selector_bar = ttk.Frame(plot_holder)
        selector_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        self._info_button(selector_bar, "display_choice").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(selector_bar, text="Result Case:").pack(side=tk.LEFT, padx=(0, 6))
        self.result_case_selector = ttk.Combobox(
            selector_bar,
            textvariable=self.result_case_choice,
            state="readonly",
            values=tuple(self.result_case_labels),
            width=20,
        )
        self.result_case_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.result_case_selector.bind("<<ComboboxSelected>>", lambda _event: self._refresh_figure(preserve_view=True))

        ttk.Label(selector_bar, text=" Component:").pack(side=tk.LEFT, padx=(6, 6))
        self.component_selector = ttk.Combobox(
            selector_bar,
            textvariable=self.component_choice,
            state="readonly",
            values=tuple(self.component_labels.keys()),
            width=26,
        )
        self.component_selector.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.component_selector.bind("<<ComboboxSelected>>", lambda _event: self._refresh_figure(preserve_view=True))
        self.interactive_3d_checkbox = ttk.Checkbutton(
            selector_bar,
            text="Interactive 3D",
            variable=self.use_interactive_3d,
            command=self._refresh_figure,
        )
        self.interactive_3d_checkbox.pack(side=tk.RIGHT, padx=6)
        self._refresh_figure()
        self._write_status("Ready. ANYstructure production FE mesh solver is available.")

    def cancel_run(self) -> None:
        self._cancel_requested = True
        if self.cancel_button is not None:
            self.cancel_button.configure(state=tk.DISABLED)

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
    def _preview_axis_data_extents(axis: Any) -> tuple[tuple[float, float], tuple[float, float], tuple[
        float, float]] | None:
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
        return self.result_case_labels.get(str(self.result_case_choice.get()), "static")

    def _selected_component(self) -> str:
        return self.component_labels.get(str(self.component_choice.get()), "von_mises_pa")

    def _set_display_modes(self, result: RuntimeFEMRunResult) -> None:
        labels = {"Static displacement/stress": "static"}
        if (result.visualization or {}).get("plastic_strain"):
            labels["Engineering plastic strain"] = "plastic"
        for mode in _buckling_mode_shapes(result):
            mode_number = int(mode.get("mode_number", 0))
            load_factor = _safe_float(mode.get("load_factor"))
            label = "Mode " + str(mode_number) + "  LF " + str(round(load_factor, 4))
            labels[label] = "mode:" + str(mode_number)
        self.result_case_labels = labels
        self.result_case_choice.set("Static displacement/stress")
        if self.result_case_selector is not None:
            self.result_case_selector.configure(values=tuple(labels))

    def _get_shell_normal(self, p: np.ndarray, is_cylinder: bool) -> np.ndarray:
        if is_cylinder:
            r = np.array([p[0], p[1], 0.0], dtype=float)
            norm_r = np.linalg.norm(r)
            if norm_r > 1.0e-9:
                return r / norm_r
            return np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            return np.array([0.0, 0.0, 1.0], dtype=float)

    def _populate_canvas_with_geometry(self, canvas: Tkinter3DCanvas, fit_view: bool = True) -> None:
        geometry = runtime_geometry_summary(self.snapshot)
        show_plate_var = getattr(self, "show_plate_vis", None)
        show_plate = show_plate_var.get() if show_plate_var is not None else True
        show_stiffeners_var = getattr(self, "show_stiffener_vis", None)
        show_stiffeners = show_stiffeners_var.get() if show_stiffeners_var is not None else True
        show_girders_var = getattr(self, "show_girder_vis", None)
        show_girders = show_girders_var.get() if show_girders_var is not None else True
        plate_alpha_var = getattr(self, "plate_alpha_vis", None)
        member_alpha_var = getattr(self, "member_alpha_vis", None)
        colormap_var = getattr(self, "colormap_vis", None)
        plate_alpha = _clamped_alpha(plate_alpha_var.get() if plate_alpha_var is not None else 1.0, 1.0)
        member_alpha = _clamped_alpha(member_alpha_var.get() if member_alpha_var is not None else 0.95, 0.95)
        plate_stipple = _alpha_to_stipple(plate_alpha)
        member_stipple = _alpha_to_stipple(member_alpha)
        _configure_tk_canvas_colormap(str(colormap_var.get() if colormap_var is not None else "jet"))

        if self.snapshot.is_cylinder:
            radius = max(_safe_float(geometry.get("radius_m"), 1.0), 1.0e-6)
            length = max(_safe_float(geometry.get("length_m"), 1.0), 1.0e-6)
            if show_plate and plate_alpha > 0.0:
                canvas.add_cylinder(
                    radius=radius,
                    height=length,
                    center=Point3D(0.0, 0.0, 0.0),
                    color="#d8e2ea",
                    outline="",
                    segments=32,
                    height_segments=12,
                    capped=False,
                    opacity=plate_alpha,
                    show_backfaces=plate_alpha < 0.94,
                    show_thickness_legend=False
                )
            if show_stiffeners and member_alpha > 0.0 and geometry.get("has_stiffener"):
                stf_spacing = _safe_float(geometry.get("stiffener_spacing_m"))
                if stf_spacing > 0.0:
                    num_longs = int(2.0 * math.pi * radius / stf_spacing)
                    stf_sec = geometry.get("stiffener_section") or {}
                    hw = _safe_float(stf_sec.get("web_height") or stf_sec.get("web_h") or 0.1)
                    tw = _safe_float(stf_sec.get("web_thickness") or stf_sec.get("web_t") or 0.01)
                    b = _safe_float(stf_sec.get("flange_width") or stf_sec.get("flange_w") or 0.0)
                    tf = _safe_float(stf_sec.get("flange_thickness") or stf_sec.get("flange_t") or 0.0)
                    for i in range(num_longs):
                        angle = 2.0 * math.pi * i / num_longs
                        canvas.add_longitudinal_stiffener(
                            radius=radius,
                            height=length,
                            angle=angle,
                            web_height=hw,
                            web_thickness=tw,
                            flange_width=b,
                            flange_thickness=tf,
                            color=_blend_hex_color("#a0a0ff", member_alpha),
                            outline=_blend_hex_color("#404080", member_alpha),
                            segments=4,
                            height_segments=8,
                            inside=True,
                            z_offset=0.0,
                        )
            if show_girders and member_alpha > 0.0 and geometry.get("has_girder"):
                gir_spacing = _safe_float(geometry.get("girder_spacing_m"))
                gir_sec = geometry.get("girder_section") or {}
                ghw = _safe_float(gir_sec.get("web_height") or gir_sec.get("web_h") or 0.12)
                gtw = _safe_float(gir_sec.get("web_thickness") or gir_sec.get("web_t") or 0.02)
                gb = _safe_float(gir_sec.get("flange_width") or gir_sec.get("flange_w") or 0.08)
                gtf = _safe_float(gir_sec.get("flange_thickness") or gir_sec.get("flange_t") or 0.015)
                if gir_spacing > 0.0:
                    num_girders = int(length / gir_spacing) + 1
                    for i in range(num_girders):
                        z_pos = -0.5 * length + i * gir_spacing
                        if abs(z_pos) <= 0.5 * length + 1.0e-3:
                            canvas.add_ring_stiffener(
                                radius=radius,
                                z_position=z_pos,
                                web_height=ghw,
                                web_thickness=gtw,
                                flange_width=gb,
                                flange_thickness=gtf,
                                color=_blend_hex_color("#ffa0a0", member_alpha),
                                outline=_blend_hex_color("#804040", member_alpha),
                                segments=32,
                                inside=True,
                            )
        else:
            length = max(_safe_float(geometry.get("length_m"), 1.0), 1.0e-6)
            width = max(_safe_float(geometry.get("width_m"), 1.0), 1.0e-6)
            if show_plate and plate_alpha > 0.0:
                canvas.add_polygon(
                    [
                        Point3D(0.0, 0.0, 0.0),
                        Point3D(length, 0.0, 0.0),
                        Point3D(length, width, 0.0),
                        Point3D(0.0, width, 0.0)
                    ],
                    color="#d1d5db",
                    outline="#64748b",
                    layer=5,
                    stipple=plate_stipple
                )
            if show_stiffeners and member_alpha > 0.0 and geometry.get("has_stiffener"):
                spacing = _safe_float(geometry.get("stiffener_spacing_m"))
                stf_sec = geometry.get("stiffener_section") or {}
                hw = _safe_float(stf_sec.get("web_height") or stf_sec.get("web_h") or 0.1)
                b = _safe_float(stf_sec.get("flange_width") or stf_sec.get("flange_w") or 0.0)
                if spacing > 0.0:
                    num_stiffeners = int(width / spacing) + 1
                    valid_indices = []
                    for i in range(num_stiffeners):
                        y_pos = i * spacing
                        if y_pos <= width + 1.0e-3:
                            valid_indices.append((i, y_pos))
                    for idx, (i, y_pos) in enumerate(valid_indices):
                        spans = [(0.0, 0.5 * length), (0.5 * length, length)]
                        for stf_start, stf_end in spans:
                            canvas.add_polygon(
                                [
                                    Point3D(stf_start, y_pos, 0.0),
                                    Point3D(stf_end, y_pos, 0.0),
                                    Point3D(stf_end, y_pos, hw),
                                    Point3D(stf_start, y_pos, hw)
                                ],
                                color="#94a3b8",
                                outline="#1f2937",
                                width=2,
                                layer=12,
                                stipple=member_stipple
                            )
                            if b > 0.0:
                                canvas.add_polygon(
                                    [
                                        Point3D(stf_start, y_pos - 0.5 * b, hw),
                                        Point3D(stf_end, y_pos - 0.5 * b, hw),
                                        Point3D(stf_end, y_pos + 0.5 * b, hw),
                                        Point3D(stf_start, y_pos + 0.5 * b, hw)
                                    ],
                                    color="#334155",
                                    outline="#111827",
                                    width=2,
                                    layer=13,
                                    stipple=member_stipple
                                )
            if show_girders and member_alpha > 0.0 and geometry.get("has_girder"):
                gir_sec = geometry.get("girder_section") or {}
                ghw = _safe_float(gir_sec.get("web_height") or gir_sec.get("web_h") or 0.15)
                gb = _safe_float(gir_sec.get("flange_width") or gir_sec.get("flange_w") or 0.08)
                x_mid = 0.5 * length
                canvas.add_polygon(
                    [
                        Point3D(x_mid, 0.0, 0.0),
                        Point3D(x_mid, width, 0.0),
                        Point3D(x_mid, width, ghw),
                        Point3D(x_mid, 0.0, ghw)
                    ],
                    color="#fca5a5",
                    outline="#991b1b",
                    width=2,
                    layer=12,
                    stipple=member_stipple
                )
                if gb > 0.0:
                    canvas.add_polygon(
                        [
                            Point3D(x_mid - 0.5 * gb, 0.0, ghw),
                            Point3D(x_mid - 0.5 * gb, width, ghw),
                            Point3D(x_mid + 0.5 * gb, width, ghw),
                            Point3D(x_mid + 0.5 * gb, 0.0, ghw)
                        ],
                        color="#b91c1c",
                        outline="#7f1d1d",
                        width=2,
                        layer=13,
                        stipple=member_stipple
                    )
        if hasattr(self, "_custom_load_patches"):
            self._draw_custom_load_patch_outlines(canvas)
        if fit_view:
            canvas.after_idle(canvas.fit_to_scene)

    def _populate_canvas_with_results(self, canvas: Tkinter3DCanvas, fit_view: bool = True) -> None:
        result = self.current_result
        geometry = result.summary
        display_mode = self._selected_display_mode()
        deformation_scale = max(_safe_float(self.deformation_scale.get(), 0.0), 0.0)
        component = self._selected_component()
        plate_alpha = _clamped_alpha(self.plate_alpha_vis.get(), 1.0)
        member_alpha = _clamped_alpha(self.member_alpha_vis.get(), 0.95)
        plate_stipple = _alpha_to_stipple(plate_alpha)
        member_stipple = _alpha_to_stipple(member_alpha)
        _configure_tk_canvas_colormap(str(self.colormap_vis.get()))
        if self.snapshot.is_cylinder and plate_alpha >= 0.94:
            set_occluder = getattr(canvas, "set_opaque_cylinder_occluder", None)
            if callable(set_occluder):
                set_occluder(
                    radius=max(
                        _safe_float(geometry.get("radius_m"), 1.0),
                        1.0e-9,
                    ),
                    height=max(_safe_float(geometry.get("length_m"), 1.0), 1.0e-9),
                    center=Point3D(0.0, 0.0, 0.0),
                )


        visualization, title, is_mode = _selected_visualization(result, display_mode, component)

        scalar_values = _plot_grid_values(visualization.get("stress_pa"))
        if is_mode:
            color_grid = scalar_values
            colorbar_label = str(visualization.get("scalar_label") or "mode amplitude")
        elif visualization.get("scalar_kind") == "raw":
            color_grid = scalar_values
            colorbar_label = str(visualization.get("scalar_label") or "value")
        else:
            if component.endswith("_pa"):
                color_grid = [[value / 1.0e6 for value in row] for row in scalar_values]
                colorbar_label = str(visualization.get("scalar_label", "stress")).replace("_pa", "") + " [MPa]"
            elif "disp" in component:
                color_grid = [[value * 1000.0 for value in row] for row in scalar_values]
                colorbar_label = str(visualization.get("scalar_label", "displacement")).replace(" [m]", " [mm]")
            else:
                color_grid = scalar_values
                colorbar_label = str(visualization.get("scalar_label", component))

        all_vals = _all_grid_values(color_grid)
        if all_vals:
            vmin = min(all_vals)
            vmax = max(all_vals)
        else:
            vmin, vmax = 0.0, 1.0

        if vmax <= vmin:
            vmax = vmin + 1.0

        scale = _displacement_plot_scale(geometry, result, visualization, deformation_scale)

        if visualization.get("type") == "cylinder":
            axial = _plot_grid_values(visualization.get("axial_m"))
            theta = _plot_grid_values(visualization.get("theta_rad"))
            radius = max(_safe_float(visualization.get("radius_m"), _safe_float(geometry.get("radius_m"), 1.0)), 1.0e-9)

            disps = visualization.get("displacements", {})
            dx_grid = _plot_grid_values(disps.get("disp_x", []))
            dy_grid = _plot_grid_values(disps.get("disp_y", []))
            dz_grid = _plot_grid_values(disps.get("disp_z", []))

            if not dx_grid or not dy_grid or not dz_grid:
                radial_displacement = _plot_grid_values(visualization.get("radial_displacement_m"))
                x = [
                    [(radius + radial_displacement[row_index][col_index] * scale) * math.cos(
                        theta[row_index][col_index])
                     for col_index in range(len(theta[row_index]))]
                    for row_index in range(len(theta))
                ]
                y = [
                    [(radius + radial_displacement[row_index][col_index] * scale) * math.sin(
                        theta[row_index][col_index])
                     for col_index in range(len(theta[row_index]))]
                    for row_index in range(len(theta))
                ]
                z = axial
            else:
                x = [
                    [(radius * math.cos(theta[row_index][col_index]) + dx_grid[row_index][col_index] * scale)
                     for col_index in range(len(theta[row_index]))]
                    for row_index in range(len(theta))
                ]
                y = [
                    [(radius * math.sin(theta[row_index][col_index]) + dy_grid[row_index][col_index] * scale)
                     for col_index in range(len(theta[row_index]))]
                    for row_index in range(len(theta))
                ]
                z = [
                    [(axial[row_index][col_index] + dz_grid[row_index][col_index] * scale)
                     for col_index in range(len(axial[row_index]))]
                    for row_index in range(len(axial))
                ]
        else:
            x = _plot_grid_values(visualization.get("x_m"))
            y = _plot_grid_values(visualization.get("y_m"))
            w = _plot_grid_values(visualization.get("w_m"))
            z = [[value * scale for value in row] for row in w]

        show_plate_var = getattr(self, "show_plate_vis", None)
        show_plate = show_plate_var.get() if show_plate_var is not None else True
        if show_plate and plate_alpha > 0.0:
            R = len(x)
            C = len(x[0]) if R > 0 else 0
            for i in range(R - 1):
                for j in range(C - 1):
                    p1 = Point3D(x[i][j], y[i][j], z[i][j])
                    p2 = Point3D(x[i + 1][j], y[i + 1][j], z[i + 1][j])
                    p3 = Point3D(x[i + 1][j + 1], y[i + 1][j + 1], z[i + 1][j + 1])
                    p4 = Point3D(x[i][j + 1], y[i][j + 1], z[i][j + 1])

                    avg_val = 0.25 * (
                                color_grid[i][j] + color_grid[i + 1][j] + color_grid[i + 1][j + 1] + color_grid[i][
                            j + 1])
                    color = _interpolate_thickness_color(avg_val, vmin, vmax)
                    canvas.add_polygon(
                        [p1, p2, p3, p4],
                        color=color,
                        outline="#64748b",
                        layer=5,
                        stipple=plate_stipple,
                    )

        show_stiffeners_var = getattr(self, "show_stiffener_vis", None)
        show_stiffeners = show_stiffeners_var.get() if show_stiffeners_var is not None else True
        show_girders_var = getattr(self, "show_girder_vis", None)
        show_girders = show_girders_var.get() if show_girders_var is not None else True

        for line in visualization.get("member_lines") or ():
            role = str(line.get("role", "member")).lower()
            if role == "stiffener" and (not show_stiffeners or member_alpha <= 0.0):
                continue
            if role == "girder" and (not show_girders or member_alpha <= 0.0):
                continue

            points = list(line.get("points") or ())
            displaced = list(line.get("displaced_points") or ())
            if len(points) < 2 or len(displaced) < 2:
                continue

            pts = []
            for idx in range(2):
                base = np.asarray(points[idx], dtype=float)
                moved = np.asarray(displaced[idx], dtype=float)
                pts.append(base + (moved - base) * float(scale))

            pA = pts[0]
            pB = pts[1]
            vec_AB = pB - pA
            len_AB = np.linalg.norm(vec_AB)
            if len_AB < 1.0e-9:
                continue
            u = vec_AB / len_AB

            nA = self._get_shell_normal(pA, self.snapshot.is_cylinder)
            nB = self._get_shell_normal(pB, self.snapshot.is_cylinder)

            e = line.get("eccentricity", 0.0)
            if abs(e) > 1.0e-9:
                web_sign = np.sign(e)
            else:
                web_sign = -1.0 if self.snapshot.is_cylinder else 1.0

            wA = web_sign * nA
            wB = web_sign * nB

            hw = line.get("web_height", 0.1)
            b_f = line.get("flange_width", 0.0)

            s_stops = [0.0, 1.0]
            centroids = [0.5]

            for k in range(1):
                z_start = s_stops[k] * hw
                z_end = s_stops[k + 1] * hw

                pAk = pA + z_start * wA
                pAk1 = pA + z_end * wA
                pBk = pB + z_start * wB
                pBk1 = pB + z_end * wB

                q1 = Point3D(pAk[0], pAk[1], pAk[2])
                q2 = Point3D(pBk[0], pBk[1], pBk[2])
                q3 = Point3D(pBk1[0], pBk1[1], pBk1[2])
                q4 = Point3D(pAk1[0], pAk1[1], pAk1[2])

                if is_mode:
                    baseA = np.asarray(points[0])
                    dispA = np.asarray(displaced[0]) - baseA
                    baseB = np.asarray(points[1])
                    dispB = np.asarray(displaced[1]) - baseB
                    val = 0.5 * (np.linalg.norm(dispA) + np.linalg.norm(dispB))
                else:
                    s_val = centroids[k]
                    sig_axial = line.get("axial_stress", 0.0)
                    sig_bend_y = line.get("bending_stress_y", 0.0)
                    sig_x = sig_axial + (2.0 * s_val - 1.0) * sig_bend_y

                    tau_y = line.get("shear_stress_y", 0.0)
                    tau_z = line.get("shear_stress_z", 0.0)
                    tau_t = line.get("torsional_stress", 0.0)

                    val = np.sqrt(sig_x ** 2 + 3.0 * (tau_y ** 2 + tau_z ** 2 + tau_t ** 2)) / 1.0e6

                color = _interpolate_thickness_color(val, vmin, vmax)
                canvas.add_polygon(
                    [q1, q2, q3, q4],
                    color=color,
                    outline="#000000",
                    width=2,
                    layer=12,
                    stipple=member_stipple,
                )

            if b_f > 0.0:
                pA_top = pA + hw * wA
                pB_top = pB + hw * wB

                v_flange_A = np.cross(u, wA)
                v_flange_B = np.cross(u, wB)

                fA1 = pA_top - 0.5 * b_f * v_flange_A
                fA2 = pA_top + 0.5 * b_f * v_flange_A
                fB1 = pB_top - 0.5 * b_f * v_flange_B
                fB2 = pB_top + 0.5 * b_f * v_flange_B

                qf1 = Point3D(fA1[0], fA1[1], fA1[2])
                qf2 = Point3D(fB1[0], fB1[1], fB1[2])
                qf3 = Point3D(fB2[0], fB2[1], fB2[2])
                qf4 = Point3D(fA2[0], fA2[1], fA2[2])

                if is_mode:
                    baseA = np.asarray(points[0])
                    dispA = np.asarray(displaced[0]) - baseA
                    baseB = np.asarray(points[1])
                    dispB = np.asarray(displaced[1]) - baseB
                    val = 0.5 * (np.linalg.norm(dispA) + np.linalg.norm(dispB))
                else:
                    sig_axial = line.get("axial_stress", 0.0)
                    sig_bend_y = line.get("bending_stress_y", 0.0)
                    sig_x = sig_axial + sig_bend_y

                    tau_y = line.get("shear_stress_y", 0.0)
                    tau_z = line.get("shear_stress_z", 0.0)
                    tau_t = line.get("torsional_stress", 0.0)

                    val = np.sqrt(sig_x ** 2 + 3.0 * (tau_y ** 2 + tau_z ** 2 + tau_t ** 2)) / 1.0e6

                color = _interpolate_thickness_color(val, vmin, vmax)
                canvas.add_polygon(
                    [qf1, qf2, qf3, qf4],
                    color=color,
                    outline="#000000",
                    width=2,
                    layer=13,
                    stipple=member_stipple,
                )

        self._draw_custom_load_patch_outlines(canvas)
        canvas.set_thickness_legend(
            values=all_vals,
            unit=colorbar_label,
            title=title
        )
        if fit_view:
            canvas.after_idle(canvas.fit_to_scene)

    def _update_result_text(self) -> None:
        if self.upper_result_text is None:
            return
        self.upper_result_text.config(state=tk.NORMAL)
        self.upper_result_text.delete("1.0", tk.END)

        if self.current_result is None:
            self.upper_result_text.insert(tk.END, "No solver results available yet. Run FEM to calculate results.")
            self.upper_result_text.config(state=tk.DISABLED)
            return

        result = self.current_result
        geometry = result.summary
        display_mode = self._selected_display_mode()

        lines = []
        if str(display_mode).startswith("mode:"):
            lines.extend([
                "--- BUCKLING MODES ANALYSIS ---",
                "Status: " + result.status.replace("_", " "),
                "Solver: " + str(geometry.get("solver", "")),
                "Max displacement [mm]: " + str(round(1000.0 * _safe_float(geometry.get("max_displacement_m")), 4)),
            ])
            for label, value in result.stress_percentiles:
                lines.append(label.upper() + " stress [MPa]: " + str(round(value / 1.0e6, 3)))
            lines.append("")
            lines.append("Mode\tLoad factor")
            lines.append("----\t-----------")

            active_mode_str = display_mode.split(":", 1)[1]
            active_idx = -1
            try:
                active_idx = int(active_mode_str)
            except ValueError:
                pass

            if result.buckling_factors:
                for idx, factor in enumerate(result.buckling_factors, start=1):
                    prefix = "--> " if idx == active_idx else "    "
                    lines.append(f"{prefix}{idx}\t{round(factor, 4)}")
            else:
                lines.append("No positive buckling modes found.")
        else:
            force = tuple((geometry.get("load_resultant") or {}).get("force_n") or ())
            applied_pressure = _safe_float(geometry.get("pressure_pa"))
            lines.extend([
                "--- STATIC RESPONSE ANALYSIS ---",
                "Status: " + result.status.replace("_", " "),
                "Solver: " + str(geometry.get("solver", "")),
                "Pressure [Pa]: " + str(round(applied_pressure, 3)),
                "Load scale: " + str(round(_safe_float(geometry.get("load_scale"), 1.0), 4)),
                "Max displacement [mm]: " + str(round(1000.0 * _safe_float(geometry.get("max_displacement_m")), 4)),
            ])
            for label, value in result.stress_percentiles:
                lines.append(label.upper() + " stress [MPa]: " + str(round(value / 1.0e6, 3)))
            if force:
                lines.append("Resultant force [N]: " + ", ".join(str(round(_safe_float(v), 2)) for v in force))

            lines.extend([
                "",
                "Buckling mode shapes are available from the Display selector.",
            ])

        self.upper_result_text.insert(tk.END, "\n".join(lines))
        self.upper_result_text.config(state=tk.DISABLED)

    def _refresh_figure(self, preserve_view: bool = False) -> None:
        self._update_result_text()
        if self.figure_parent is None:
            return

        show_base_geometry = self.current_result is None or self._display_base_geometry

        if hasattr(self, "preview_canvas") and self.preview_canvas is not None:
            try:
                self.preview_canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self.preview_canvas = None

        if self.use_interactive_3d.get():
            if self.figure_canvas is not None:
                try:
                    self.figure_canvas.get_tk_widget().destroy()
                except Exception:
                    pass
                self.figure_canvas = None
            if self.figure_toolbar is not None:
                try:
                    self.figure_toolbar.destroy()
                except Exception:
                    pass
                self.figure_toolbar = None
            if self.figure_toolbar_frame is not None:
                try:
                    self.figure_toolbar_frame.destroy()
                except Exception:
                    pass
                self.figure_toolbar_frame = None

            canvas_created = self.result_canvas is None
            if canvas_created:
                self.result_canvas = Tkinter3DCanvas(self.figure_parent, bg="white")
                self.result_canvas.pack(fill=tk.BOTH, expand=True)
                self._bind_custom_load_canvas_selection(self.result_canvas)

            try:
                self.result_canvas.canvas.configure(
                    cursor="crosshair" if self._custom_load_selection_active else ""
                )
            except Exception:
                pass

            fit_view = bool(canvas_created or self._force_fit_next_refresh or not preserve_view)
            self._force_fit_next_refresh = False
            self.result_canvas.clear()
            self.result_canvas.clear_thickness_legend()

            if show_base_geometry:
                self._populate_canvas_with_geometry(self.result_canvas, fit_view=fit_view)
            else:
                self._populate_canvas_with_results(self.result_canvas, fit_view=fit_view)
        else:
            if self.result_canvas is not None:
                try:
                    self.result_canvas.destroy()
                except Exception:
                    pass
                self.result_canvas = None

            self._force_fit_next_refresh = False
            self._show_figure(
                create_runtime_fem_result_figure(
                    self.snapshot,
                    None if show_base_geometry else self.current_result,
                    self._selected_display_mode(),
                    max(_safe_float(self.deformation_scale.get(), 0.0), 0.0),
                    show_plate=self.show_plate_vis.get(),
                    show_stiffeners=self.show_stiffener_vis.get(),
                    show_girders=self.show_girder_vis.get(),
                    plate_alpha=_clamped_alpha(self.plate_alpha_vis.get(), 1.0),
                    member_alpha=_clamped_alpha(self.member_alpha_vis.get(), 0.95),
                    colormap=str(self.colormap_vis.get()),
                    component=self._selected_component(),
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
        if self.cancel_button is not None:
            self.cancel_button.configure(state=tk.NORMAL if is_running else tk.DISABLED)
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
            beam_element_order=str(self.beam_element_order.get()),
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
            custom_pressure_pa=_safe_float(self.custom_pressure_pa.get(), 0.0),
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
            custom_loads_json=str(self.custom_loads_json.get()),
            custom_pressure_patches_json=str(self.custom_pressure_patches_json.get()),
            custom_edge_segments_json=str(self.custom_edge_segments_json.get()),
            custom_selected_edge_load_n_per_m=_safe_float(self.custom_selected_edge_load_n_per_m.get(), 0.0),
            custom_time_domain_enabled=bool(self.custom_time_domain_enabled.get()),
            custom_time_domain_duration_s=max(_safe_float(self.custom_time_domain_duration_s.get(), 0.01), 0.0),
            custom_time_domain_total_time_s=max(_safe_float(self.custom_time_domain_total_time_s.get(), 0.05), 0.0),
            custom_time_domain_dt_s=max(_safe_float(self.custom_time_domain_dt_s.get(), 0.0005), 1.0e-9),
            custom_time_domain_include_static_load=bool(self.custom_time_domain_include_static_load.get()),
            imperfection_enabled=bool(self.imperfection_enabled.get()),
            imperfection_shape=str(self.imperfection_shape.get()),
            imperfection_amplitude_m=max(_safe_float(self.imperfection_amplitude_m.get(), 0.0), 0.0),
            imperfection_wave_a=max(_safe_int(self.imperfection_wave_a.get(), 1), 1),
            imperfection_wave_b=max(_safe_int(self.imperfection_wave_b.get(), 1), 1),
            runtime_solver=str(self.runtime_solver.get()),
            allow_unbalanced_free_free=bool(self.allow_unbalanced_free_free.get()),
            buckling_shift_load_factor=max(_safe_float(self.buckling_shift_load_factor.get(), 0.0), 0.0),
            buckling_min_load_factor=max(_safe_float(self.buckling_min_load_factor.get(), 0.0), 0.0),
            buckling_max_load_factor=max(_safe_float(self.buckling_max_load_factor.get(), 0.0), 0.0),
            buckling_repeated_tolerance=max(_safe_float(self.buckling_repeated_tolerance.get(), 1.0e-3), 0.0),
            buckling_allow_dense_fallback=bool(self.buckling_allow_dense_fallback.get()),
            recovery_history_mode=str(self.recovery_history_mode.get()),
            recovery_threads=max(_safe_int(self.recovery_threads.get(), 0), 0),
            memory_limit_mb=max(_safe_float(self.memory_limit_mb.get(), 0.0), 0.0),
            capacity_buckling_mode_number=max(_safe_int(self.capacity_buckling_mode_number.get(), 1), 1),
            capacity_mesh_min_elements_per_half_wave=max(
                _safe_int(self.capacity_mesh_min_elements_per_half_wave.get(), 4), 1),
        )

    def _bind_custom_load_list_traces(self) -> None:
        if self._custom_load_trace_ids:
            return
        variables: tuple[tk.Variable, ...] = (
            self.custom_load_bc_enabled,
            self.custom_loads_add_to_imported,
            self.custom_use_nullspace_projection,
            self.allow_unbalanced_free_free,
            self.plate_edge_x0_support,
            self.plate_edge_x1_support,
            self.plate_edge_y0_support,
            self.plate_edge_y1_support,
            self.cylinder_lower_support,
            self.cylinder_upper_support,
            self.plate_edge_x0_load_n_per_m,
            self.plate_edge_x1_load_n_per_m,
            self.plate_edge_y0_load_n_per_m,
            self.plate_edge_y1_load_n_per_m,
            self.cylinder_lower_edge_load_n_per_m,
            self.cylinder_upper_edge_load_n_per_m,
            self.custom_time_domain_enabled,
            self.custom_time_domain_duration_s,
            self.custom_time_domain_total_time_s,
            self.custom_time_domain_dt_s,
            self.custom_time_domain_include_static_load,
        )
        for variable in variables:
            trace_id = variable.trace_add("write", lambda *_args: self._refresh_custom_load_list())
            self._custom_load_trace_ids.append((variable, trace_id))

    def _custom_boundary_summary(self) -> tuple[str, str]:
        flags = [
            "custom on" if bool(self.custom_load_bc_enabled.get()) else "custom off",
            "add imported" if bool(self.custom_loads_add_to_imported.get()) else "replace imported",
        ]
        if bool(self.custom_use_nullspace_projection.get()):
            flags.append("nullspace")
        if bool(self.allow_unbalanced_free_free.get()):
            flags.append("allow free-free")
        if self.snapshot.is_cylinder:
            supports = (
                "lower=" + str(self.cylinder_lower_support.get())
                + ", upper=" + str(self.cylinder_upper_support.get())
            )
            edge_loads = (
                "lower "
                + f"{_safe_float(self.cylinder_lower_edge_load_n_per_m.get(), 0.0):g}"
                + ", upper "
                + f"{_safe_float(self.cylinder_upper_edge_load_n_per_m.get(), 0.0):g}"
                + " N/m"
            )
        else:
            supports = (
                "x0=" + str(self.plate_edge_x0_support.get())
                + ", x1=" + str(self.plate_edge_x1_support.get())
                + ", y0=" + str(self.plate_edge_y0_support.get())
                + ", y1=" + str(self.plate_edge_y1_support.get())
            )
            edge_loads = (
                "x0 "
                + f"{_safe_float(self.plate_edge_x0_load_n_per_m.get(), 0.0):g}"
                + ", x1 "
                + f"{_safe_float(self.plate_edge_x1_load_n_per_m.get(), 0.0):g}"
                + ", y0 "
                + f"{_safe_float(self.plate_edge_y0_load_n_per_m.get(), 0.0):g}"
                + ", y1 "
                + f"{_safe_float(self.plate_edge_y1_load_n_per_m.get(), 0.0):g}"
                + " N/m"
            )
        time_domain = "time off"
        if bool(self.custom_time_domain_enabled.get()):
            time_domain = (
                "time dt="
                + f"{_safe_float(self.custom_time_domain_dt_s.get(), 0.0):g}"
                + " s, duration="
                + f"{_safe_float(self.custom_time_domain_duration_s.get(), 0.0):g}"
                + " s, total="
                + f"{_safe_float(self.custom_time_domain_total_time_s.get(), 0.0):g}"
                + " s"
            )
            if bool(self.custom_time_domain_include_static_load.get()):
                time_domain += ", static included"
        return supports, "; ".join(flags + [edge_loads, time_domain])

    @staticmethod
    def _custom_load_entry_selection_text(entry: dict[str, Any]) -> str:
        if str(entry.get("type", "")).lower() in {"pressure", "panel_pressure"}:
            patches = entry.get("patches", [])
            count = len(patches) if isinstance(patches, list) else 0
            area = 0.0
            if isinstance(patches, list):
                for patch in patches:
                    if not isinstance(patch, dict):
                        continue
                    area += max(0.0, _safe_float(patch.get("max_a")) - _safe_float(patch.get("min_a"))) * max(
                        0.0,
                        _safe_float(patch.get("max_b")) - _safe_float(patch.get("min_b")),
                    )
            return f"{count} panel(s), {area:.3f} m2"
        edges = entry.get("edges", [])
        count = len(edges) if isinstance(edges, list) else 0
        return f"{count} edge segment(s)"

    @staticmethod
    def _custom_load_entry_value_text(entry: dict[str, Any]) -> str:
        if str(entry.get("type", "")).lower() in {"pressure", "panel_pressure"}:
            return f"{_safe_float(entry.get('pressure_pa'), 0.0):g} Pa"
        return f"{_safe_float(entry.get('line_load_n_per_m'), 0.0):g} N/m"

    def _refresh_custom_load_list(self) -> None:
        tree = getattr(self, "_custom_load_tree", None)
        if tree is None:
            return
        selected = tree.selection()
        selected_iid = selected[0] if selected else ""
        for item in tree.get_children():
            tree.delete(item)
        boundary_selection, boundary_notes = self._custom_boundary_summary()
        tree.insert("", tk.END, iid="boundary", values=("Boundary", "", boundary_selection, boundary_notes))
        for index, entry in enumerate(self._custom_load_entries):
            entry_type = str(entry.get("type", "")).lower()
            type_text = "Pressure" if entry_type in {"pressure", "panel_pressure"} else "Edge load"
            notes = "time-domain capable" if type_text == "Pressure" and bool(self.custom_time_domain_enabled.get()) else ""
            tree.insert(
                "",
                tk.END,
                iid=f"load:{index}",
                values=(
                    type_text,
                    self._custom_load_entry_value_text(entry),
                    self._custom_load_entry_selection_text(entry),
                    notes,
                ),
            )
        if selected_iid and tree.exists(selected_iid):
            tree.selection_set(selected_iid)

    def _sync_custom_load_payloads(self) -> None:
        entries = [dict(entry) for entry in self._custom_load_entries]
        self.custom_loads_json.set(json.dumps(entries))
        pressure_patches: list[dict[str, Any]] = []
        edge_segments: list[dict[str, Any]] = []
        for entry in entries:
            if str(entry.get("type", "")).lower() in {"pressure", "panel_pressure"}:
                patches = entry.get("patches", [])
                if isinstance(patches, list):
                    pressure_patches.extend([dict(patch) for patch in patches if isinstance(patch, dict)])
            elif str(entry.get("type", "")).lower() in {"edge", "edge_load"}:
                edges = entry.get("edges", [])
                if isinstance(edges, list):
                    edge_segments.extend([dict(edge) for edge in edges if isinstance(edge, dict)])
        self.custom_pressure_patches_json.set(json.dumps(pressure_patches))
        self.custom_edge_segments_json.set(json.dumps(edge_segments))
        self._refresh_custom_load_list()

    def _add_custom_load_from_selection(self) -> None:
        selected_patches = [
            dict(patch)
            for patch in getattr(self, "_custom_load_patches", ())
            if bool(patch.get("selected", False))
        ]
        selected_edges = self._selected_custom_load_edges()
        pressure = _safe_float(self.custom_pressure_pa.get(), 0.0)
        line_load = _safe_float(self.custom_selected_edge_load_n_per_m.get(), 0.0)
        added = 0
        if selected_patches and abs(pressure) > 0.0:
            self._custom_load_entries.append({
                "type": "pressure",
                "pressure_pa": float(pressure),
                "patches": selected_patches,
            })
            added += 1
        if selected_edges and abs(line_load) > 0.0:
            self._custom_load_entries.append({
                "type": "edge",
                "line_load_n_per_m": float(line_load),
                "edges": selected_edges,
            })
            added += 1
        if added == 0:
            self._write_status(
                "Select at least one panel with non-zero pressure or right-click at least one edge with non-zero edge load before adding."
            )
            return
        self.custom_load_bc_enabled.set(True)
        self._sync_custom_load_payloads()
        self._write_status(f"Added {added} custom load item(s) to the run list.")

    def _delete_selected_custom_load(self) -> None:
        tree = getattr(self, "_custom_load_tree", None)
        if tree is None:
            return
        selected = tree.selection()
        if not selected:
            self._write_status("Select a pressure or edge load in the list before deleting.")
            return
        iid = selected[0]
        if not iid.startswith("load:"):
            self._write_status("Boundary-condition information cannot be deleted from the load list.")
            return
        try:
            index = int(iid.split(":", 1)[1])
        except (IndexError, ValueError):
            return
        if 0 <= index < len(self._custom_load_entries):
            del self._custom_load_entries[index]
            self._sync_custom_load_payloads()
            self._write_status("Deleted selected custom load from the run list.")

    def run(self) -> None:
        """Prepare/run the runtime FEM request and render Matplotlib results."""

        if self.solver_thread is not None and self.solver_thread.is_alive():
            return
        if not self.include_stiffeners.get() and not self.include_girders.get():
            messagebox.showwarning("FEM solver", "At least one member beam family should normally be included.")

        options = self._options()
        self._set_solver_running(True)
        self._cancel_requested = False
        self._run_status_history = ["The result plot will update when the solver finishes.", ""]
        self._write_status("Running FEM solver...\n\n" + "\n".join(self._run_status_history))

        def worker() -> None:
            def status_cb(msg: str):
                if getattr(self, "_cancel_requested", False):
                    raise RuntimeError("Run stopped by user.")
                self.solver_queue.put(msg)

            try:
                self.solver_queue.put((run_runtime_fem(self.snapshot, options, status_callback=status_cb), None))
            except Exception as error:
                self.solver_queue.put((None, error))

        self.solver_thread = threading.Thread(target=worker, daemon=True)
        self.solver_thread.start()
        self.window.after(100, self._poll_solver_result)

    def _poll_solver_result(self) -> None:
        try:
            msg = self.solver_queue.get_nowait()
            if isinstance(msg, str):
                if not hasattr(self, "_run_status_history"):
                    self._run_status_history = []
                self._run_status_history.append(msg)
                self._write_status("Running FEM solver...\n\n" + "\n".join(self._run_status_history))
                self.window.after(100, self._poll_solver_result)
                return
            result, error = msg
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
        self._display_base_geometry = False
        self._set_custom_load_selection_active(False, refresh=False)
        self._force_fit_next_refresh = True
        self._set_display_modes(result)
        self._write_status(format_runtime_fem_result(result))
        self._refresh_figure()

    def _set_custom_load_selection_active(self, active: bool, refresh: bool = True) -> None:
        self._custom_load_selection_active = bool(active)
        self._custom_load_click_origin = None
        self._custom_load_edge_click_origin = None
        if self._custom_load_selection_button is not None:
            self._custom_load_selection_button.configure(
                text="Finish selection" if self._custom_load_selection_active else "Start selection"
            )
        if self.result_canvas is not None:
            try:
                self.result_canvas.canvas.configure(
                    cursor="crosshair" if self._custom_load_selection_active else ""
                )
            except Exception:
                pass
        if refresh:
            self._refresh_figure(preserve_view=True)

    def _toggle_custom_load_selection(self) -> None:
        start_selection = not self._custom_load_selection_active
        if start_selection:
            self.use_interactive_3d.set(True)
            self._display_base_geometry = True
            self._force_fit_next_refresh = self.result_canvas is None
            self._set_custom_load_selection_active(True, refresh=False)
            self._write_status(
                "Custom load selection is active. Left-click panels for pressure; right-click edges for line loads; drag to rotate the view."
            )
        else:
            self._set_custom_load_selection_active(False, refresh=False)
            self._write_status("Custom load selection finished.")
        self._refresh_figure(preserve_view=True)

    def _bind_custom_load_canvas_selection(self, canvas: Tkinter3DCanvas) -> None:
        canvas.canvas.bind("<ButtonPress-1>", self._on_custom_load_canvas_press, add="+")
        canvas.canvas.bind("<ButtonRelease-1>", self._on_custom_load_canvas_release, add="+")
        canvas.canvas.bind("<ButtonPress-3>", self._on_custom_load_edge_canvas_press, add="+")
        canvas.canvas.bind("<ButtonRelease-3>", self._on_custom_load_edge_canvas_release, add="+")

    def _on_custom_load_canvas_press(self, event: Any) -> None:
        if self._custom_load_selection_active:
            self._custom_load_click_origin = (int(event.x), int(event.y))

    def _on_custom_load_canvas_release(self, event: Any) -> None:
        origin = self._custom_load_click_origin
        self._custom_load_click_origin = None
        if not self._custom_load_selection_active or origin is None or self.result_canvas is None:
            return
        if math.hypot(float(event.x) - origin[0], float(event.y) - origin[1]) > 5.0:
            return

        patch_index = self._pick_custom_load_patch(self.result_canvas, float(event.x), float(event.y))
        if patch_index is None:
            self._write_status("Custom load selection: no panel was found at the clicked position.")
            return

        self._custom_load_selected_index = patch_index
        patch = self._custom_load_patches[patch_index]
        patch["selected"] = not bool(patch.get("selected", False))
        self._update_custom_load_summary()
        state = "selected" if patch["selected"] else "cleared"
        self._write_status(f"Custom load panel {patch_index + 1} {state}.")
        self._refresh_figure(preserve_view=True)

    def _on_custom_load_edge_canvas_press(self, event: Any) -> None:
        if self._custom_load_selection_active:
            self._custom_load_edge_click_origin = (int(event.x), int(event.y))

    def _on_custom_load_edge_canvas_release(self, event: Any) -> None:
        origin = self._custom_load_edge_click_origin
        self._custom_load_edge_click_origin = None
        if not self._custom_load_selection_active or origin is None or self.result_canvas is None:
            return
        if math.hypot(float(event.x) - origin[0], float(event.y) - origin[1]) > 5.0:
            return

        edge = self._pick_custom_load_edge(self.result_canvas, float(event.x), float(event.y))
        if edge is None:
            self._write_status("Custom load selection: no edge was found at the clicked position.")
            return

        key = self._custom_load_edge_key(*edge)
        if key in self._custom_selected_edge_keys:
            self._custom_selected_edge_keys.remove(key)
            state = "cleared"
        else:
            self._custom_selected_edge_keys.add(key)
            state = "selected"
        self._update_custom_load_summary()
        self._write_status(f"Custom load edge {state}.")
        self._refresh_figure(preserve_view=True)

    @staticmethod
    def _point_segment_distance_2d(
            px: float,
            py: float,
            ax: float,
            ay: float,
            bx: float,
            by: float,
    ) -> float:
        dx = bx - ax
        dy = by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1.0e-12:
            return math.hypot(px - ax, py - ay)
        fraction = ((px - ax) * dx + (py - ay) * dy) / length_sq
        fraction = min(max(fraction, 0.0), 1.0)
        return math.hypot(px - (ax + fraction * dx), py - (ay + fraction * dy))

    @classmethod
    def _point_in_polygon_2d(cls, x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
        if len(polygon) < 3:
            return False
        inside = False
        previous = polygon[-1]
        for current in polygon:
            if cls._point_segment_distance_2d(x, y, previous[0], previous[1], current[0], current[1]) <= 4.0:
                return True
            x1, y1 = previous
            x2, y2 = current
            if (y1 > y) != (y2 > y):
                crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < crossing_x:
                    inside = not inside
            previous = current
        return inside

    def _custom_load_patch_boundary_points(
            self,
            patch: dict[str, Any],
            surface_offset: float = 0.0,
    ) -> list[Point3D]:
        geometry = runtime_geometry_summary(self.snapshot)
        min_a = _safe_float(patch.get("min_a"))
        max_a = _safe_float(patch.get("max_a"))
        min_b = _safe_float(patch.get("min_b"))
        max_b = _safe_float(patch.get("max_b"))

        if self.snapshot.is_cylinder:
            radius = max(_safe_float(geometry.get("radius_m"), 1.0), 1.0e-9)
            draw_radius = max(radius + surface_offset, 1.0e-9)
            theta_0 = min_b / radius
            theta_1 = max_b / radius
            arc_steps = max(2, int(math.ceil(abs(theta_1 - theta_0) / (math.pi / 24.0))))
            angles = [theta_0 + (theta_1 - theta_0) * index / arc_steps for index in range(arc_steps + 1)]
            lower = [Point3D(draw_radius * math.cos(theta), draw_radius * math.sin(theta), min_a)
                     for theta in angles]
            upper = [Point3D(draw_radius * math.cos(theta), draw_radius * math.sin(theta), max_a)
                     for theta in reversed(angles)]
            return lower + upper

        return [
            Point3D(min_a, min_b, surface_offset),
            Point3D(max_a, min_b, surface_offset),
            Point3D(max_a, max_b, surface_offset),
            Point3D(min_a, max_b, surface_offset),
        ]

    def _pick_custom_load_patch(
            self,
            canvas: Tkinter3DCanvas,
            screen_x: float,
            screen_y: float,
    ) -> int | None:
        if not self._custom_load_patches:
            return None

        try:
            plot_width = max(1, int(canvas._plot_width()))
        except Exception:
            plot_width = max(1, int(canvas.width))
        height = max(1, int(canvas.height))
        right, camera_up, forward = canvas.camera.basis()
        position = canvas.camera.position
        scale = 1.0 / math.tan(canvas.camera.fov / 2.0)
        aspect = plot_width / height
        x_scale = scale / aspect
        half_width = 0.5 * plot_width
        half_height = 0.5 * height

        best_index: int | None = None
        best_depth = float("inf")
        for index, patch in enumerate(self._custom_load_patches):
            projected: list[tuple[float, float]] = []
            depths: list[float] = []
            for point in self._custom_load_patch_boundary_points(patch):
                relative = point - position
                depth = relative.dot(forward)
                if depth <= canvas.camera.near or depth >= canvas.camera.far:
                    projected = []
                    break
                camera_x = relative.dot(right)
                camera_y = relative.dot(camera_up)
                projected.append((
                    (camera_x * x_scale / depth + 1.0) * half_width,
                    (1.0 - camera_y * scale / depth) * half_height,
                ))
                depths.append(depth)
            if projected and self._point_in_polygon_2d(screen_x, screen_y, projected):
                mean_depth = sum(depths) / len(depths)
                if mean_depth < best_depth:
                    best_depth = mean_depth
                    best_index = index
        return best_index

    def _project_custom_load_points(
            self,
            canvas: Tkinter3DCanvas,
            points: list[Point3D],
    ) -> list[tuple[float, float, float]]:
        try:
            plot_width = max(1, int(canvas._plot_width()))
        except Exception:
            plot_width = max(1, int(canvas.width))
        height = max(1, int(canvas.height))
        right, camera_up, forward = canvas.camera.basis()
        position = canvas.camera.position
        scale = 1.0 / math.tan(canvas.camera.fov / 2.0)
        aspect = plot_width / height
        x_scale = scale / aspect
        half_width = 0.5 * plot_width
        half_height = 0.5 * height
        projected: list[tuple[float, float, float]] = []
        for point in points:
            relative = point - position
            depth = relative.dot(forward)
            if depth <= canvas.camera.near or depth >= canvas.camera.far:
                return []
            camera_x = relative.dot(right)
            camera_y = relative.dot(camera_up)
            projected.append((
                (camera_x * x_scale / depth + 1.0) * half_width,
                (1.0 - camera_y * scale / depth) * half_height,
                float(depth),
            ))
        return projected

    def _custom_load_outline_edge_points(
            self,
            varying_axis: str,
            fixed_coordinate: float,
            start_coordinate: float,
            end_coordinate: float,
            surface_offset: float,
    ) -> list[Point3D]:
        # Convert one boundary segment in patch coordinates to 3D points.
        geometry = runtime_geometry_summary(self.snapshot)
        start_coordinate = float(start_coordinate)
        end_coordinate = float(end_coordinate)
        fixed_coordinate = float(fixed_coordinate)

        if self.snapshot.is_cylinder:
            radius = max(_safe_float(geometry.get("radius_m"), 1.0), 1.0e-9)
            draw_radius = max(radius + surface_offset, 1.0e-9)
            if varying_axis == "a":
                theta = fixed_coordinate / radius
                return [
                    Point3D(
                        draw_radius * math.cos(theta),
                        draw_radius * math.sin(theta),
                        start_coordinate,
                    ),
                    Point3D(
                        draw_radius * math.cos(theta),
                        draw_radius * math.sin(theta),
                        end_coordinate,
                    ),
                ]

            theta_0 = start_coordinate / radius
            theta_1 = end_coordinate / radius
            arc_steps = max(
                2,
                int(math.ceil(abs(theta_1 - theta_0) / (math.pi / 36.0))),
            )
            return [
                Point3D(
                    draw_radius * math.cos(
                        theta_0 + (theta_1 - theta_0) * index / arc_steps
                    ),
                    draw_radius * math.sin(
                        theta_0 + (theta_1 - theta_0) * index / arc_steps
                    ),
                    fixed_coordinate,
                )
                for index in range(arc_steps + 1)
            ]

        if varying_axis == "a":
            return [
                Point3D(start_coordinate, fixed_coordinate, surface_offset),
                Point3D(end_coordinate, fixed_coordinate, surface_offset),
            ]
        return [
            Point3D(fixed_coordinate, start_coordinate, surface_offset),
            Point3D(fixed_coordinate, end_coordinate, surface_offset),
        ]

    @staticmethod
    def _custom_load_edge_key(
            varying_axis: str,
            fixed_coordinate: float,
            start_coordinate: float,
            end_coordinate: float,
    ) -> tuple[str, float, float, float]:
        start = float(start_coordinate)
        end = float(end_coordinate)
        if end < start:
            start, end = end, start
        return (
            str(varying_axis).lower(),
            round(float(fixed_coordinate), 10),
            round(start, 10),
            round(end, 10),
        )

    @staticmethod
    def _custom_load_edge_payload_from_key(
            key: tuple[str, float, float, float],
    ) -> dict[str, float | str]:
        return {
            "varying_axis": key[0],
            "fixed_coordinate": float(key[1]),
            "start_coordinate": float(key[2]),
            "end_coordinate": float(key[3]),
        }

    def _all_custom_load_edges(self) -> list[tuple[str, float, float, float]]:
        edges: dict[tuple[str, float, float, float], tuple[str, float, float, float]] = {}
        for patch in getattr(self, "_custom_load_patches", ()):
            min_a = _safe_float(patch.get("min_a"))
            max_a = _safe_float(patch.get("max_a"))
            min_b = _safe_float(patch.get("min_b"))
            max_b = _safe_float(patch.get("max_b"))
            candidates = (
                ("a", min_b, min_a, max_a),
                ("a", max_b, min_a, max_a),
                ("b", min_a, min_b, max_b),
                ("b", max_a, min_b, max_b),
            )
            for candidate in candidates:
                key = self._custom_load_edge_key(*candidate)
                edges[key] = candidate
        return [edges[key] for key in sorted(edges)]

    def _selected_custom_load_edges(self) -> list[dict[str, float | str]]:
        all_keys = {self._custom_load_edge_key(*edge) for edge in self._all_custom_load_edges()}
        self._custom_selected_edge_keys.intersection_update(all_keys)
        return [
            self._custom_load_edge_payload_from_key(key)
            for key in sorted(self._custom_selected_edge_keys)
        ]

    def _pick_custom_load_edge(
            self,
            canvas: Tkinter3DCanvas,
            screen_x: float,
            screen_y: float,
    ) -> tuple[str, float, float, float] | None:
        best_edge: tuple[str, float, float, float] | None = None
        best_distance = 9.0
        best_depth = float("inf")
        geometry = runtime_geometry_summary(self.snapshot)
        surface_offset = (
            max(_safe_float(geometry.get("radius_m"), 1.0) * 1.0e-3, 1.0e-5)
            if self.snapshot.is_cylinder
            else max(_safe_float(geometry.get("length_m"), 1.0), _safe_float(geometry.get("width_m"), 1.0)) * 1.0e-5
        )
        for edge in self._all_custom_load_edges():
            points = self._custom_load_outline_edge_points(*edge, surface_offset=surface_offset)
            projected = self._project_custom_load_points(canvas, points)
            if len(projected) < 2:
                continue
            min_distance = min(
                self._point_segment_distance_2d(
                    screen_x,
                    screen_y,
                    projected[index][0],
                    projected[index][1],
                    projected[index + 1][0],
                    projected[index + 1][1],
                )
                for index in range(len(projected) - 1)
            )
            mean_depth = sum(item[2] for item in projected) / len(projected)
            if min_distance <= best_distance and mean_depth < best_depth:
                best_distance = min_distance
                best_depth = mean_depth
                best_edge = edge
        return best_edge

    def _selected_custom_load_boundary_edges(
            self,
    ) -> list[tuple[str, float, float, float]]:
        # Return only the external boundary of the combined selected patch area.
        selected_patches = [
            patch
            for patch in self._custom_load_patches
            if bool(patch.get("selected", False))
        ]
        if not selected_patches:
            return []

        tolerance_digits = 10

        def clean(value: Any) -> float:
            return round(_safe_float(value), tolerance_digits)

        a_values = sorted(
            {
                clean(patch.get(key))
                for patch in self._custom_load_patches
                for key in ("min_a", "max_a")
            }
        )
        b_values = sorted(
            {
                clean(patch.get(key))
                for patch in self._custom_load_patches
                for key in ("min_b", "max_b")
            }
        )
        if len(a_values) < 2 or len(b_values) < 2:
            return []

        selected_cells: set[tuple[int, int]] = set()
        for index_a in range(len(a_values) - 1):
            mid_a = 0.5 * (a_values[index_a] + a_values[index_a + 1])
            for index_b in range(len(b_values) - 1):
                mid_b = 0.5 * (b_values[index_b] + b_values[index_b + 1])
                for patch in selected_patches:
                    if (
                        _safe_float(patch.get("min_a")) - 1.0e-9
                        <= mid_a
                        <= _safe_float(patch.get("max_a")) + 1.0e-9
                        and _safe_float(patch.get("min_b")) - 1.0e-9
                        <= mid_b
                        <= _safe_float(patch.get("max_b")) + 1.0e-9
                    ):
                        selected_cells.add((index_a, index_b))
                        break

        is_cylinder = bool(self.snapshot.is_cylinder)
        number_b_cells = len(b_values) - 1
        edges: list[tuple[str, float, float, float]] = []

        for index_a, index_b in sorted(selected_cells):
            a_0 = a_values[index_a]
            a_1 = a_values[index_a + 1]
            b_0 = b_values[index_b]
            b_1 = b_values[index_b + 1]

            below = (index_a - 1, index_b)
            above = (index_a + 1, index_b)
            left_b = (
                index_a,
                (index_b - 1) % number_b_cells
                if is_cylinder
                else index_b - 1,
            )
            right_b = (
                index_a,
                (index_b + 1) % number_b_cells
                if is_cylinder
                else index_b + 1,
            )

            if below not in selected_cells:
                edges.append(("b", a_0, b_0, b_1))
            if above not in selected_cells:
                edges.append(("b", a_1, b_0, b_1))
            if left_b not in selected_cells:
                edges.append(("a", b_0, a_0, a_1))
            if right_b not in selected_cells:
                edges.append(("a", b_1, a_0, a_1))

        return edges

    def _draw_custom_load_patch_outlines(self, canvas: Tkinter3DCanvas) -> None:
        # Draw selectable panels, selected perimeter and user-created cuts.
        if not self._custom_load_patches:
            return

        geometry = runtime_geometry_summary(self.snapshot)
        if self.snapshot.is_cylinder:
            surface_offset = max(
                _safe_float(geometry.get("radius_m"), 1.0) * 1.0e-3,
                1.0e-5,
            )
        else:
            surface_offset = (
                max(
                    _safe_float(geometry.get("length_m"), 1.0),
                    _safe_float(geometry.get("width_m"), 1.0),
                )
                * 1.0e-5
            )

        if self._custom_load_selection_active:
            for patch in self._custom_load_patches:
                points = self._custom_load_patch_boundary_points(
                    patch,
                    surface_offset=surface_offset,
                )
                if len(points) < 2:
                    continue
                for start, end in zip(points, points[1:] + points[:1]):
                    canvas.add_line(
                        start,
                        end,
                        color="#94a3b8",
                        width=1,
                    )

        for varying_axis, fixed_coordinate, start_coordinate, end_coordinate in (
                self._selected_custom_load_boundary_edges()
        ):
            points = self._custom_load_outline_edge_points(
                varying_axis,
                fixed_coordinate,
                start_coordinate,
                end_coordinate,
                surface_offset,
            )
            for start, end in zip(points, points[1:]):
                canvas.add_line(
                    start,
                    end,
                    color="#dc2626",
                    width=4,
                )

        for edge in self._selected_custom_load_edges():
            points = self._custom_load_outline_edge_points(
                str(edge.get("varying_axis", "a")),
                _safe_float(edge.get("fixed_coordinate")),
                _safe_float(edge.get("start_coordinate")),
                _safe_float(edge.get("end_coordinate")),
                surface_offset,
            )
            for start, end in zip(points, points[1:]):
                canvas.add_line(
                    start,
                    end,
                    color="#16a34a",
                    width=4,
                )

        for cut in getattr(self, "_custom_load_manual_cuts", ()):
            points = self._custom_load_outline_edge_points(
                str(cut.get("varying_axis", "a")),
                _safe_float(cut.get("fixed_coordinate")),
                _safe_float(cut.get("start_coordinate")),
                _safe_float(cut.get("end_coordinate")),
                surface_offset,
            )
            for start, end in zip(points, points[1:]):
                canvas.add_line(
                    start,
                    end,
                    color="#f59e0b",
                    width=3,
                )

    def _custom_load_select_all(self) -> None:
        if not hasattr(self, "_custom_load_patches"): return
        for f in self._custom_load_patches: f["selected"] = True
        self._update_custom_load_summary()
        self._refresh_figure()

    def _custom_load_clear_all(self) -> None:
        # Remove the selection and restore the unsplit default panel layout.
        self._custom_load_manual_cuts.clear()
        self._custom_selected_edge_keys.clear()
        self._generate_default_custom_load_patches()
        self._custom_load_selected_index = -1
        self._display_base_geometry = True
        self._force_fit_next_refresh = False
        self._update_custom_load_summary()
        self._write_status(
            "Custom load selection and all manually created panel cuts were cleared."
        )
        self._refresh_figure(preserve_view=True)

    def _custom_load_split_field(self, axis: str) -> None:
        # Split the active selected panel exactly at its local midpoint.
        if not self._custom_load_patches:
            return
        if (
                self._custom_load_selected_index < 0
                or self._custom_load_selected_index >= len(self._custom_load_patches)
        ):
            self._write_status(
                "Select a custom load panel before using Split A or Split B."
            )
            return

        field = self._custom_load_patches[self._custom_load_selected_index]
        if not bool(field.get("selected", False)):
            self._write_status(
                "The active panel is not selected. Select it before cutting."
            )
            return

        axis = str(axis).lower()
        min_a = _safe_float(field.get("min_a"))
        max_a = _safe_float(field.get("max_a"))
        min_b = _safe_float(field.get("min_b"))
        max_b = _safe_float(field.get("max_b"))

        if axis == "a":
            split_coordinate = 0.5 * (min_a + max_a)
            first = dict(field)
            first["max_a"] = split_coordinate
            second = dict(field)
            second["min_a"] = split_coordinate
            cut = {
                "varying_axis": "b",
                "fixed_coordinate": split_coordinate,
                "start_coordinate": min_b,
                "end_coordinate": max_b,
            }
            direction_label = "A"
        elif axis == "b":
            split_coordinate = 0.5 * (min_b + max_b)
            first = dict(field)
            first["max_b"] = split_coordinate
            second = dict(field)
            second["min_b"] = split_coordinate
            cut = {
                "varying_axis": "a",
                "fixed_coordinate": split_coordinate,
                "start_coordinate": min_a,
                "end_coordinate": max_a,
            }
            direction_label = "B"
        else:
            raise ValueError(f"Unknown custom load split axis: {axis!r}")

        cut_key = (
            str(cut["varying_axis"]),
            round(float(cut["fixed_coordinate"]), 10),
            round(float(cut["start_coordinate"]), 10),
            round(float(cut["end_coordinate"]), 10),
        )
        existing_keys = {
            (
                str(item.get("varying_axis", "")),
                round(_safe_float(item.get("fixed_coordinate")), 10),
                round(_safe_float(item.get("start_coordinate")), 10),
                round(_safe_float(item.get("end_coordinate")), 10),
            )
            for item in getattr(self, "_custom_load_manual_cuts", ())
        }
        if cut_key not in existing_keys:
            self._custom_load_manual_cuts.append(cut)

        index = self._custom_load_selected_index
        self._custom_load_patches[index:index + 1] = [first, second]
        self._custom_load_selected_index = index
        self._display_base_geometry = True
        self._force_fit_next_refresh = False
        self._update_custom_load_summary()
        self._write_status(
            f"Selected custom load panel was split in local {direction_label} "
            "at its midpoint. Press Clear to remove all cuts."
        )
        self._refresh_figure(preserve_view=True)

    def _update_custom_load_summary(self) -> None:
        if not hasattr(self, "_custom_load_patches"): return
        total = 0.0
        for f in self._custom_load_patches:
            if f.get("selected", False):
                total += (f["max_a"] - f["min_a"]) * (f["max_b"] - f["min_b"])
        if hasattr(self, "custom_load_area_var") and self.custom_load_area_var is not None:
            self.custom_load_area_var.set(f"Selected Area: {total:.3f} m2")
        self._sync_custom_load_payloads()

    def _generate_default_custom_load_patches(self) -> None:
        geometry = runtime_geometry_summary(self.snapshot)
        is_cylinder = str(geometry.get("geometry", "")).lower() == "cylinder"
        if is_cylinder:
            length = max(float(geometry.get("length_m") or 1.0), 1.0e-6)
            min_a = -0.5 * length
            max_a = 0.5 * length
            radius = max(float(geometry.get("radius_m") or 1.0), 1.0e-6)
            circumference = 2.0 * math.pi * radius
            min_b = 0.0
            max_b = circumference
        else:
            min_a = 0.0
            max_a = max(float(geometry.get("length_m") or 1.0), 1.0e-6)
            min_b = 0.0
            max_b = max(float(geometry.get("width_m") or 1.0), 1.0e-6)

        a_breaks = [min_a, max_a]
        b_breaks = [min_b, max_b]

        if is_cylinder:
            if geometry.get("has_girder"):
                girder_spacing = float(geometry.get("girder_spacing_m") or 0.0)
                if girder_spacing > 0.0:
                    axial_bays = max(1, int(round((max_a - min_a) / girder_spacing)))
                    a_breaks = [
                        min_a + (max_a - min_a) * index / axial_bays
                        for index in range(axial_bays + 1)
                    ]

            stiffener_spacing = float(geometry.get("stiffener_spacing_m") or 0.0)
            if geometry.get("has_stiffener") and stiffener_spacing > 0.0:
                circumferential_bays = max(1, int((max_b - min_b) / stiffener_spacing))
            else:
                circumferential_bays = 16
            b_breaks = [
                min_b + (max_b - min_b) * index / circumferential_bays
                for index in range(circumferential_bays + 1)
            ]
        elif geometry.get("has_stiffener"):
            spacing = float(geometry.get("stiffener_spacing_m") or 0.0)
            if spacing > 0.0:
                b_breaks = [min_b]
                count = int((max_b - min_b) / spacing) + 1
                for index in range(count):
                    coordinate = min_b + index * spacing
                    if min_b < coordinate < max_b:
                        b_breaks.append(coordinate)
                b_breaks.append(max_b)

        a_breaks = sorted(set(a_breaks))
        b_breaks = sorted(set(b_breaks))

        self._custom_load_patches = []
        self._custom_load_selected_index = -1
        for index_a in range(len(a_breaks) - 1):
            for index_b in range(len(b_breaks) - 1):
                self._custom_load_patches.append({
                    "min_a": a_breaks[index_a],
                    "max_a": a_breaks[index_a + 1],
                    "min_b": b_breaks[index_b],
                    "max_b": b_breaks[index_b + 1],
                    "selected": False,
                })
        self._update_custom_load_summary()

    def _redraw_base_3d(self) -> None:
        self._display_base_geometry = True
        self._force_fit_next_refresh = True
        self._refresh_figure()
        self._write_status("Displaying base 3D model geometry.")

    def _show_results(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("FEM results", "No solver results are available yet. Run FEM first.")
            return
        self._display_base_geometry = False
        self._set_custom_load_selection_active(False, refresh=False)
        self._force_fit_next_refresh = True
        self._refresh_figure()
        self._write_status("Displaying the latest FEM results.")


def open_runtime_fem_window(parent: Any, app: Any) -> RuntimeFEMWindow | None:
    """Open the experimental runtime FEM popup for the app active line."""

    try:
        return RuntimeFEMWindow(parent, app)
    except Exception as error:
        messagebox.showinfo("Experimental FEM solver", str(error))
        return None


class _ExamplePlate:
    girder_lg = 10.0

    def get_structure_type(self):
        return "Flat plate, stiffened with girder"

    def get_span(self):
        return 3.5

    def get_s(self):
        return 0.75

    def get_pl_thk(self):
        return 0.018

    def get_puls_up_boundary(self):
        return "SSSS"


class _ExampleTMember:
    stiffener_type = "T"

    def __init__(self, spacing: float, web_h: float, web_t: float, flange_w: float, flange_t: float):
        self.spacing = spacing
        self.hw = web_h
        self.tw = web_t
        self.b = flange_w
        self.tf = flange_t
        self.girder_lg = 10.0

    def get_s(self):
        return self.spacing

    def get_web_h(self):
        return self.hw

    def get_web_thk(self):
        return self.tw

    def get_fl_w(self):
        return self.b

    def get_fl_thk(self):
        return self.tf

    def get_stiffener_type(self):
        return self.stiffener_type


class _ExampleAllStructure:
    Plate = _ExamplePlate()
    Stiffener = _ExampleTMember(spacing=0.75, web_h=0.400, web_t=0.012, flange_w=0.250, flange_t=0.012)
    Girder = _ExampleTMember(spacing=3.5, web_h=0.800, web_t=0.020, flange_w=0.200, flange_t=0.030)
    _panel_length_Lp = 10.0


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
    _active_line = "line_example"
    _line_is_active = True
    _line_dict = {"line_example": [1, 2]}
    _simplified_calculation_mode = True

    def __init__(self, example: str = "girder_panel"):
        self.example = "cylinder" if str(example).lower() == "cylinder" else "girder_panel"
        cylinder = _ExampleCylinder() if self.example == "cylinder" else None
        self._fem_default_top_bottom_moment_nm = 30_000_000.0 if self.example == "cylinder" else 0.0
        self._line_to_struc = {"line_example": [_ExampleAllStructure(), None, None, object(), None, cylinder]}
        self._new_prop_3d_opposite_side = _ExampleTkVariable(False)
        self._new_shell_ring_frame_length_between_girders = _ExampleTkVariable(4.0)
        self._new_shell_dist_rings = _ExampleTkVariable(2.0)
        self._new_panel_length_Lp = _ExampleTkVariable(10_000.0)
        self._new_girder_length_LG = _ExampleTkVariable(10_000.0)

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
        if line != self._active_line:
            return {"normal": 0.0}
        return {"normal": 100_000.0 if self.example == "cylinder" else 459_639.0}


def example_runtime_app(example: str = "girder_panel") -> _ExampleRuntimeApp:
    """Return a tiny active-line app fixture for running this module directly."""

    return _ExampleRuntimeApp(example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the standalone experimental ANYstructure FEM window.")
    parser.add_argument(
        "--example",
        choices=("girder_panel", "cylinder"),
        default="cylinder",
        help="Standalone example to open. Default is the flat stiffened girder panel.",
    )
    args = parser.parse_args()
    root = tk.Tk()
    my_app = RuntimeFEMWindow(root, example_runtime_app(args.example), use_parent_as_window=True)
    my_app.window.protocol("WM_DELETE_WINDOW", root.destroy)
    my_app.window.focus_force()
    root.mainloop()
