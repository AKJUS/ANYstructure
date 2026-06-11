"""Experimental runtime FEM solver window for active ANYstructure lines.

The module owns the active-line handoff, user options and result visualization
for the experimental full-geometry FEM mode.  It calls the ANYstructure-local
``anystruct.fe_solver`` module; solver development happens in ANYintelligent
and can later be copied into that local module without changing this GUI layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import os
import sys

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
    num_buckling_modes: int = 5


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
    }


def _cylinder_geometry_summary(snapshot: RuntimeFEMLineSnapshot) -> dict[str, Any]:
    bundle = snapshot.structure_bundle
    cyl_obj = bundle[5] if bundle and len(bundle) > 5 else None
    shell = getattr(cyl_obj, "ShellObj", None)
    return {
        "geometry": "cylinder",
        "radius_m": _safe_float(_read_attr_or_call(shell, "radius", default=None), 1.0),
        "length_m": _safe_float(_read_attr_or_call(shell, "length_of_shell", "tot_cyl_length", default=None), 1.0),
        "thickness_m": _safe_float(_read_attr_or_call(shell, "thk", default=None), 0.0),
        "has_stiffener": getattr(cyl_obj, "LongStfObj", None) is not None,
        "has_girder": getattr(cyl_obj, "RingFrameObj", None) is not None,
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
        num_buckling_modes=options.num_buckling_modes,
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
        "num_buckling_modes": int(options.num_buckling_modes),
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


def _stress_facecolors(stress_mpa: list[list[float]]):
    values = _all_grid_values(stress_mpa) or [0.0]
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values) if max(values) > min(values) else min(values) + 1.0)
    cmap = colormaps["viridis"]
    return [[cmap(norm(value)) for value in row] for row in stress_mpa], norm, cmap


def _displacement_plot_scale(geometry: dict[str, Any], result: RuntimeFEMRunResult | None) -> float:
    if result is None or result.displacement_scale <= 0.0:
        return 1.0
    length = _safe_float(geometry.get("length_m"), 1.0)
    width = _safe_float(geometry.get("width_m"), _safe_float(geometry.get("radius_m"), 1.0))
    reference = max(length, width, _safe_float(geometry.get("radius_m"), 0.0), 1.0e-9)
    return min(max(0.08 * reference / max(result.displacement_scale, 1.0e-12), 1.0), 1.0e5)


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


def _plot_visualization_surface(figure: Figure, axis: Any, geometry: dict[str, Any], result: RuntimeFEMRunResult) -> None:
    visualization = result.visualization or {}
    stress_pa = _plot_grid_values(visualization.get("stress_pa"))
    stress_mpa = [[value / 1.0e6 for value in row] for row in stress_pa]
    facecolors, norm, cmap = _stress_facecolors(stress_mpa)
    scale = _displacement_plot_scale(geometry, result)

    if visualization.get("type") == "cylinder":
        axial = _plot_grid_values(visualization.get("axial_m"))
        theta = _plot_grid_values(visualization.get("theta_rad"))
        radial_displacement = _plot_grid_values(visualization.get("radial_displacement_m"))
        radius = max(_safe_float(visualization.get("radius_m"), _safe_float(geometry.get("radius_m"), 1.0)), 1.0e-9)
        y = [
            [(radius + radial_displacement[row_index][col_index] * scale) * math.cos(theta[row_index][col_index])
             for col_index in range(len(theta[row_index]))]
            for row_index in range(len(theta))
        ]
        z = [
            [(radius + radial_displacement[row_index][col_index] * scale) * math.sin(theta[row_index][col_index])
             for col_index in range(len(theta[row_index]))]
            for row_index in range(len(theta))
        ]
        axis.plot_surface(
            np.asarray(axial),
            np.asarray(y),
            np.asarray(z),
            facecolors=np.asarray(facecolors),
            linewidth=0.15,
            antialiased=True,
            shade=False,
        )
        axis.set_xlabel("length [m]")
        axis.set_ylabel("y [m]")
        axis.set_zlabel("z [m]")
        _set_3d_axes_limits(axis, axial, y, z)
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
            linewidth=0.15,
            antialiased=True,
            shade=False,
        )
        axis.set_xlabel("length [m]")
        axis.set_ylabel("width [m]")
        axis.set_zlabel("w x" + str(round(scale, 1)))
        _set_3d_axes_limits(axis, x, y, z)

    axis.set_title("3D stress/displacement")
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(_all_grid_values(stress_mpa))
    figure.colorbar(mappable, ax=axis, shrink=0.68, pad=0.1, label="stress [MPa]")


def create_runtime_fem_result_figure(snapshot: RuntimeFEMLineSnapshot, result: RuntimeFEMRunResult | None = None) -> Figure:
    """Create the Matplotlib result visualization used in the runtime popup."""

    figure = Figure(figsize=(8.0, 4.1), dpi=100)
    geometry_ax = figure.add_subplot(121, projection="3d")
    result_ax = figure.add_subplot(122)
    geometry = runtime_geometry_summary(snapshot) if result is None else result.summary

    if result is None or not result.visualization:
        geometry_ax.set_title("3D stress/displacement")
        geometry_ax.text2D(0.08, 0.56, "Run FEM to plot stresses and displacements.", transform=geometry_ax.transAxes)
        geometry_ax.set_xlabel("length [m]")
        geometry_ax.set_ylabel("width/radius [m]")
        geometry_ax.set_zlabel("displacement")
        result_ax.text(0.5, 0.55, "No FEM run yet", ha="center", va="center", fontsize=12)
        result_ax.text(0.5, 0.42, "Results will appear here after Run FEM.", ha="center", va="center", fontsize=9)
        result_ax.set_axis_off()
    else:
        _plot_visualization_surface(figure, geometry_ax, geometry, result)
        labels = [label for label, _value in result.stress_percentiles]
        values = [value / 1.0e6 for _label, value in result.stress_percentiles]
        colors = ["#2563eb", "#0f766e"][: len(values)]
        result_ax.bar(labels, values, color=colors)
        result_ax.set_title("Result summary")
        result_ax.set_ylabel("MPa")
        result_ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
        result_ax.text(0.02, 0.96, result.status.replace("_", " "), transform=result_ax.transAxes,
                       ha="left", va="top", fontsize=9)
        if result.buckling_factors:
            result_ax.text(
                0.02,
                0.86,
                "critical factor: " + str(round(result.buckling_factors[0], 3)),
                transform=result_ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
            )
    figure.tight_layout()
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
        "Pressure [Pa]: " + str(round(_safe_float(summary.get("pressure_pa")), 3)),
        "Include stiffener beams: " + str(bool(summary.get("include_stiffeners"))),
        "Include girder/frame beams: " + str(bool(summary.get("include_girders"))),
        "Buckling modes: " + str(summary.get("num_buckling_modes", "")),
        "Max displacement [mm]: " + str(round(1000.0 * _safe_float(summary.get("max_displacement_m")), 4)),
    ]
    if result.buckling_factors:
        lines.append("Critical load factor: " + str(round(result.buckling_factors[0], 4)))
    mesh_info = summary.get("mesh_info") or {}
    if mesh_info:
        lines.extend([
            "",
            "Lightweight mesh estimate:",
            " - nodes: " + str(mesh_info.get("nodes", 0)),
            " - shells: " + str(mesh_info.get("shells", 0)),
            " - beams: " + str(mesh_info.get("beams", 0)),
        ])
    prestress = summary.get("prestress_summary") or {}
    if prestress:
        lines.extend(["", "Recovered prestress / reference state:"])
        for key, value in prestress.items():
            lines.append(" - " + key + ": " + str(round(_safe_float(value), 3)))
    load_resultant = summary.get("load_resultant") or {}
    if load_resultant:
        force = load_resultant.get("force_n", (0.0, 0.0, 0.0))
        lines.extend(["", "Load resultant force [N]: " + ", ".join(str(round(_safe_float(component), 3)) for component in force)])
    if result.diagnostics:
        lines.extend(["", "Diagnostics:"])
        lines.extend(" - " + item for item in result.diagnostics)
    return "\n".join(lines)


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
        self.load_scale = tk.DoubleVar(value=1.0)
        self.pressure_pa = tk.DoubleVar(value=self.snapshot.pressure_pa)
        self.include_stiffeners = tk.BooleanVar(value=True)
        self.include_girders = tk.BooleanVar(value=True)
        self.num_buckling_modes = tk.IntVar(value=5)
        self.result_text = None
        self.figure_canvas = None
        self.figure_toolbar = None

        self._build()

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

        summary = ttk.LabelFrame(outer, text="Active line")
        summary.pack(fill=tk.X, pady=(0, 10))
        summary_text = (
            "Line: " + self.snapshot.line_name
            + "\nDomain: " + (self.snapshot.domain or "unknown")
            + "\nGeometry: " + ("cylinder/panel" if self.snapshot.is_cylinder else "flat panel")
            + "\nPressure [Pa]: " + str(round(self.snapshot.pressure_pa, 3))
        )
        ttk.Label(summary, text=summary_text, justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=8)

        options = ttk.LabelFrame(outer, text="Run options")
        options.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(options, text="Mesh fidelity").grid(row=0, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.OptionMenu(options, self.mesh_fidelity, self.mesh_fidelity.get(), "coarse", "medium", "fine").grid(
            row=0, column=1, sticky=tk.W, padx=8, pady=6
        )
        ttk.Label(options, text="Pressure [Pa]").grid(row=1, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(options, textvariable=self.pressure_pa, width=14).grid(row=1, column=1, sticky=tk.W, padx=8, pady=6)
        ttk.Label(options, text="Load scale").grid(row=1, column=2, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(options, textvariable=self.load_scale, width=10).grid(row=1, column=3, sticky=tk.W, padx=8, pady=6)
        ttk.Checkbutton(options, text="Include stiffener beams", variable=self.include_stiffeners).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=8, pady=6
        )
        ttk.Checkbutton(options, text="Include girder/frame beams", variable=self.include_girders).grid(
            row=2, column=2, columnspan=2, sticky=tk.W, padx=8, pady=6
        )
        ttk.Label(options, text="Buckling modes").grid(row=3, column=0, sticky=tk.W, padx=8, pady=6)
        ttk.Entry(options, textvariable=self.num_buckling_modes, width=8).grid(row=3, column=1, sticky=tk.W, padx=8, pady=6)

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(buttons, text="Run FEM", command=self.run).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Close", command=self.window.destroy).pack(side=tk.RIGHT)

        result_frame = ttk.LabelFrame(outer, text="Run status and visualization")
        result_frame.pack(fill=tk.BOTH, expand=True)
        result_frame.columnconfigure(0, weight=1)
        result_frame.columnconfigure(1, weight=2)
        result_frame.rowconfigure(0, weight=1)

        self.result_text = tk.Text(result_frame, height=12, wrap=tk.WORD)
        self.result_text.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        plot_holder = ttk.Frame(result_frame)
        plot_holder.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self._show_figure(create_runtime_fem_result_figure(self.snapshot), plot_holder)
        self._write_status("Ready. ANYstructure lightweight runtime solver is available.")

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

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        self.figure_canvas = FigureCanvasTkAgg(figure, master=parent)
        self.figure_toolbar = NavigationToolbar2Tk(self.figure_canvas, toolbar_frame, pack_toolbar=False)
        self.figure_toolbar.update()
        self.figure_toolbar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _write_status(self, text: str) -> None:
        if self.result_text is None:
            return
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)

    def _options(self) -> RuntimeFEMOptions:
        return RuntimeFEMOptions(
            mesh_fidelity=str(self.mesh_fidelity.get()),
            pressure_pa=_safe_float(self.pressure_pa.get()),
            load_scale=_safe_float(self.load_scale.get(), 1.0),
            include_stiffeners=bool(self.include_stiffeners.get()),
            include_girders=bool(self.include_girders.get()),
            num_buckling_modes=max(_safe_int(self.num_buckling_modes.get(), 5), 1),
        )

    def run(self) -> None:
        """Prepare/run the runtime FEM request and render Matplotlib results."""

        if not self.include_stiffeners.get() and not self.include_girders.get():
            messagebox.showwarning("FEM solver", "At least one member beam family should normally be included.")

        result = run_runtime_fem(self.snapshot, self._options())
        self._write_status(format_runtime_fem_result(result))
        self._show_figure(create_runtime_fem_result_figure(self.snapshot, result))


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


class _ExampleRuntimeApp:
    _general_color = "#f0f0f0"
    _active_line = "line_example"
    _line_is_active = True
    _line_dict = {"line_example": [1, 2]}
    _line_to_struc = {"line_example": [_ExampleAllStructure(), None, None, object(), None, None]}

    def get_highest_pressure(self, line):
        return {"normal": 521_418.0 if line == self._active_line else 0.0}


def example_runtime_app() -> _ExampleRuntimeApp:
    """Return a tiny active-line app fixture for running this module directly."""

    return _ExampleRuntimeApp()


if __name__ == "__main__":
    root = tk.Tk()
    my_app = RuntimeFEMWindow(root, example_runtime_app(), use_parent_as_window=True)
    my_app.window.protocol("WM_DELETE_WINDOW", root.destroy)
    my_app.window.focus_force()
    root.mainloop()
