from matplotlib.figure import Figure
from pathlib import Path
import math

import pytest

from anystruct import fe_runtime_solver, fe_solver


class _Plate:
    def get_structure_type(self):
        return "Flat plate, stiffened"

    def get_span(self):
        return 2.5

    def get_s(self):
        return 0.75

    def get_pl_thk(self):
        return 0.012


class _AllStructure:
    Plate = _Plate()
    Stiffener = object()
    Girder = object()


class _FakeApp:
    _active_line = "line1"
    _line_dict = {"line1": [1, 2]}
    _line_to_struc = {"line1": [_AllStructure(), None, None, object(), None, None]}

    def get_highest_pressure(self, line):
        assert line == "line1"
        return {"normal": 12345.0}


def test_active_line_snapshot_uses_current_anystructure_line():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeApp())

    assert snapshot.line_name == "line1"
    assert snapshot.line_points == [1, 2]
    assert snapshot.pressure_pa == 12345.0
    assert snapshot.domain == "Flat plate, stiffened"
    assert snapshot.is_cylinder is False


def test_runtime_geometry_summary_reads_flat_panel_dimensions_and_members():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeApp())

    summary = fe_runtime_solver.runtime_geometry_summary(snapshot)

    assert summary["geometry"] == "flat panel"
    assert summary["length_m"] == 2.5
    assert summary["width_m"] == 0.75
    assert summary["thickness_m"] == 0.012
    assert summary["has_stiffener"] is True
    assert summary["has_girder"] is True
    assert summary["stiffener_spacing_m"] == 0.75


def test_run_runtime_fem_returns_backend_status_and_visualization_payload():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeApp())
    options = fe_runtime_solver.RuntimeFEMOptions(
        mesh_fidelity="medium",
        pressure_pa=100_000.0,
        load_scale=1.2,
        include_stiffeners=True,
        include_girders=True,
        num_buckling_modes=4,
    )

    result = fe_runtime_solver.run_runtime_fem(snapshot, options)

    assert result.status == "ok"
    assert result.summary["pressure_pa"] == 120_000.0
    assert result.summary["mesh_fidelity"] == "medium"
    assert result.summary["solver"] == "ANYstructure production FE mesh"
    assert result.summary["mesh_info"]["shells"] > 0
    assert result.summary["prestress_summary"]
    assert result.summary["load_resultant"]
    assert result.visualization["type"] == "flat"
    assert result.visualization["stress_pa"]
    assert result.stress_percentiles[0][0] == "p95"
    assert result.stress_percentiles[0][1] > 0.0
    assert result.buckling_factors == tuple(sorted(result.buckling_factors))


def test_runtime_fem_matplotlib_figure_contains_geometry_and_result_axes():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeApp())
    result = fe_runtime_solver.run_runtime_fem(
        snapshot,
        fe_runtime_solver.RuntimeFEMOptions(
            mesh_fidelity="coarse",
            pressure_pa=100_000.0,
            load_scale=1.0,
            include_stiffeners=True,
            include_girders=True,
            num_buckling_modes=3,
        ),
    )

    figure = fe_runtime_solver.create_runtime_fem_result_figure(snapshot, result)

    assert isinstance(figure, Figure)
    assert len(figure.axes) >= 2
    assert figure.axes[0].get_title() == "Static stress/displacement"
    assert figure.axes[1].get_title() == "Buckling modes"
    assert not figure.axes[1].patches
    assert figure.axes[1].tables


def test_runtime_fem_result_print_explains_unavailable_nonlinear_factor():
    result = fe_runtime_solver.RuntimeFEMRunResult(
        status="ok",
        summary={
            "line": "line1",
            "geometry": "cylinder",
            "mesh_fidelity": "coarse",
            "shell_element_order": "S8",
            "boundary_condition": "auto",
            "symmetry_mode": "none",
            "analysis_type": "nonlinear stability",
            "buckling_analysis_type": "nonlinear limit",
            "solver_type": "direct",
            "pressure_pa": 1000.0,
            "pressure_direction": "external",
            "axial_force_n": 0.0,
            "enforced_displacement_m": 0.0,
            "mesh_size_m": 0.0,
            "top_bottom_moment_nm": 0.0,
            "include_stiffeners": True,
            "include_girders": True,
            "include_end_lids": True,
            "member_orientation": "auto",
            "stiffener_eccentricity_m": 0.0,
            "girder_eccentricity_m": 0.0,
            "elastic_modulus_pa": 210.0e9,
            "poisson_ratio": 0.3,
            "yield_stress_pa": 355.0e6,
            "stress_percentile": 95.0,
            "num_buckling_modes": 5,
            "max_displacement_m": 0.0,
            "prestress_summary": {
                "shell_elements": 800,
                "nonlinear_status": "initial_tangent_not_positive",
                "nonlinear_limit_factor": 0.0,
                "nonlinear_steps": 0,
            },
        },
        stress_percentiles=(),
        buckling_factors=(),
        diagnostics=(),
        visualization={},
    )

    text = fe_runtime_solver.format_runtime_fem_result(result)

    assert "Nonlinear tangent-stability check:" in text
    assert "estimated nonlinear load factor: not available" in text
    assert "initial tangent stiffness was not positive" in text
    assert " - nonlinear_limit_factor: 0.0" not in text


def test_runtime_fem_result_print_explains_nullspace_projection():
    result = fe_runtime_solver.RuntimeFEMRunResult(
        status="ok",
        summary={
            "line": "line1",
            "geometry": "cylinder",
            "mesh_fidelity": "coarse",
            "shell_element_order": "S4",
            "boundary_condition": "auto",
            "symmetry_mode": "none",
            "analysis_type": "linear eigenvalue",
            "buckling_analysis_type": "linear eigenvalue",
            "solver_type": "direct",
            "pressure_pa": 1000.0,
            "pressure_direction": "external",
            "axial_force_n": 0.0,
            "enforced_displacement_m": 0.0,
            "mesh_size_m": 0.0,
            "top_bottom_moment_nm": 0.0,
            "include_stiffeners": True,
            "include_girders": True,
            "include_end_lids": True,
            "member_orientation": "auto",
            "stiffener_eccentricity_m": 0.0,
            "girder_eccentricity_m": 0.0,
            "elastic_modulus_pa": 210.0e9,
            "poisson_ratio": 0.3,
            "yield_stress_pa": 355.0e6,
            "stress_percentile": 95.0,
            "custom_load_bc_enabled": True,
            "cylinder_lower_support": "free",
            "cylinder_upper_support": "free",
            "cylinder_lower_edge_load_n_per_m": 0.0,
            "cylinder_upper_edge_load_n_per_m": 0.0,
            "num_buckling_modes": 5,
            "max_displacement_m": 0.0,
            "prestress_summary": {
                "shell_elements": 800,
                "constraint_method": "transformation_fixed_plus_mpc_nullspace",
                "nullspace_projection": 1.0,
            },
        },
        stress_percentiles=(),
        buckling_factors=(),
        diagnostics=(),
        visualization={},
    )

    text = fe_runtime_solver.format_runtime_fem_result(result)

    assert "Custom load/BC mode: True" in text
    assert "Linear constraint handling:" in text
    assert "nullspace projection: used" in text
    assert "rigid-body modes were projected out" in text


def test_dnv_c208_steel_properties_use_grade_and_thickness_class():
    props = fe_solver.dnv_c208_steel_properties("S355", thickness_m=0.018, thickness_class="auto")

    assert props["grade"] == "S355"
    assert props["thickness_class"] == "16 < t <= 40"
    assert props["sigma_prop"] == pytest.approx(311.0e6)
    assert props["sigma_yield"] == pytest.approx(346.9e6)
    assert props["sigma_yield_2"] == pytest.approx(353.1e6)
    assert props["eps_p_y1"] == pytest.approx(0.004)
    assert props["eps_p_y2"] == pytest.approx(0.015)
    assert props["K"] == pytest.approx(740.0e6)
    assert props["n"] == pytest.approx(0.166)


def test_production_solver_runs_incremental_material_nonlinear_static_path():
    result = fe_solver.run_production_fem(
        {
            "geometry": "flat panel",
            "length_m": 0.6,
            "width_m": 0.3,
            "thickness_m": 0.01,
            "has_stiffener": False,
            "has_girder": False,
        },
        fe_solver.LightweightFEMConfig(
            pressure_pa=1000.0,
            mesh_fidelity="coarse",
            num_buckling_modes=1,
            analysis_type="geom. + material nonlinear static",
            material_model="DNV-RP-C208 steel",
            steel_grade="S355",
            nonlinear_max_load_factor=1.0,
            nonlinear_steps=2,
            nonlinear_layers=4,
        ),
    )

    prestress = result.prestress_summary

    assert result.status == "ok"
    assert prestress["material_model"] == "DNV-RP-C208"
    assert prestress["steel_grade"] == "S355"
    assert prestress["nonlinear_static_status"] == "completed"
    assert prestress["nonlinear_static_load_factor"] == pytest.approx(1.0)
    assert prestress["nonlinear_static_layers"] in {3.0, 5.0}
    assert result.visualization["plastic_strain"]
    assert result.visualization["plastic_strain_label"] == "equiv. engineering plastic strain [-]"
    assert "Ran incremental geometric/material nonlinear static solve: completed." in result.diagnostics


def test_runtime_fem_plots_engineering_plastic_strain_and_uses_deformation_scale():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeApp())
    result = fe_runtime_solver.run_runtime_fem(
        snapshot,
        fe_runtime_solver.RuntimeFEMOptions(
            pressure_pa=1000.0,
            mesh_fidelity="coarse",
            num_buckling_modes=1,
            analysis_type="geom. + material nonlinear static",
            material_model="DNV-RP-C208 steel",
            nonlinear_max_load_factor=1.0,
            nonlinear_steps=2,
            nonlinear_layers=3,
            deformation_scale=12.0,
        ),
    )

    figure = fe_runtime_solver.create_runtime_fem_result_figure(snapshot, result, "plastic", deformation_scale=12.0)

    assert result.visualization["plastic_strain"]
    assert figure.axes[0].get_title() == "Engineering plastic strain"
    assert any(getattr(axis, "get_ylabel", lambda: "")() == "equiv. engineering plastic strain [-]" for axis in figure.axes)


def test_runtime_result_print_includes_dnv_curve_and_nonlinear_static_summary():
    result = fe_runtime_solver.RuntimeFEMRunResult(
        status="ok",
        summary={
            "line": "line1",
            "geometry": "flat panel",
            "mesh_fidelity": "coarse",
            "shell_element_order": "S4",
            "boundary_condition": "auto",
            "symmetry_mode": "none",
            "analysis_type": "geom. + material nonlinear static",
            "buckling_analysis_type": "linear eigenvalue",
            "solver_type": "direct",
            "pressure_pa": 1000.0,
            "pressure_direction": "external",
            "axial_force_n": 0.0,
            "enforced_displacement_m": 0.0,
            "mesh_size_m": 0.0,
            "top_bottom_moment_nm": 0.0,
            "include_stiffeners": False,
            "include_girders": False,
            "include_end_lids": False,
            "member_orientation": "auto",
            "stiffener_eccentricity_m": 0.0,
            "girder_eccentricity_m": 0.0,
            "material_model": "DNV-RP-C208 steel",
            "steel_grade": "S355",
            "steel_thickness_class": "auto",
            "elastic_modulus_pa": 210.0e9,
            "poisson_ratio": 0.3,
            "yield_stress_pa": 355.0e6,
            "stress_percentile": 95.0,
            "nonlinear_max_load_factor": 1.0,
            "nonlinear_steps": 2,
            "nonlinear_max_iterations": 25,
            "nonlinear_layers": 5,
            "custom_load_bc_enabled": False,
            "num_buckling_modes": 1,
            "max_displacement_m": 0.0,
            "prestress_summary": {
                "material_model": "DNV-RP-C208",
                "steel_grade": "S355",
                "steel_thickness_class": "t <= 16",
                "sigma_prop_pa": 320.0e6,
                "sigma_yield_pa": 357.0e6,
                "sigma_yield_2_pa": 363.3e6,
                "eps_p_y1": 0.004,
                "eps_p_y2": 0.015,
                "hardening_K_pa": 740.0e6,
                "hardening_n": 0.166,
                "nonlinear_static_status": "completed",
                "nonlinear_static_load_factor": 1.0,
                "nonlinear_static_steps": 2,
                "nonlinear_static_total_iterations": 6,
                "nonlinear_static_layers": 5,
                "nonlinear_static_max_plastic_strain": 0.0,
            },
        },
        stress_percentiles=(),
        buckling_factors=(),
        diagnostics=(),
        visualization={},
    )

    text = fe_runtime_solver.format_runtime_fem_result(result)

    assert "Material model: DNV-RP-C208 steel" in text
    assert "DNV-RP-C208 material curve:" in text
    assert "sigma_prop/yield/yield2 [MPa]: 320.0 / 357.0 / 363.3" in text
    assert "Incremental nonlinear static solve:" in text
    assert "last converged load factor: 1.0" in text


def test_runtime_fem_popup_has_compact_3d_section_preview():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeApp())

    figure = fe_runtime_solver.create_runtime_fem_geometry_preview_figure(snapshot)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert figure.axes[0].get_title() == "3D section view"
    assert hasattr(figure.axes[0], "get_zlim")


def test_runtime_fem_popup_wires_preview_canvas_in_upper_right():
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "fe_runtime_solver.py").read_text(encoding="utf-8")

    assert "import queue" in source
    assert "import threading" in source
    assert "body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)" in source
    assert "body.add(left_panel, weight=2)" in source
    assert "body.add(mid_panel, weight=2)" in source
    assert "body.add(right_panel, weight=3)" in source
    assert "FEM_OPTION_INFO: dict[str, dict[str, str]]" in source
    assert "def _info_button(self, parent: Any, key: str) -> ttk.Button:" in source
    assert "def _show_solver_info(self, key: str) -> None:" in source
    assert "ttk.Button(parent, text=\"i\", width=2" in source
    assert "future_inputs = ttk.LabelFrame(mid_panel, text=\"Analysis options\")" in source
    assert "constraints = ttk.LabelFrame(future_inputs, text=\"Supports and load path\")" in source
    assert "solver_options = ttk.LabelFrame(future_inputs, text=\"Solver\")" in source
    assert "members = ttk.LabelFrame(future_inputs, text=\"Member modelling\")" in source
    assert "material = ttk.LabelFrame(future_inputs, text=\"Material and recovery\")" in source
    assert "preview = ttk.LabelFrame(right_panel, text=\"3D section view\")" in source
    assert "preview.pack(fill=tk.BOTH, expand=True, pady=(0, 10))" in source
    assert "self._show_preview_figure(create_runtime_fem_geometry_preview_figure(self.snapshot, self.app), preview)" in source
    assert "self.preview_canvas = FigureCanvasTkAgg(figure, master=parent)" in source
    assert "self.figure_toolbar_frame = toolbar_frame" in source
    assert "self.figure_toolbar_frame.destroy()" in source
    assert "redraw_after_id" in source
    assert "def _fit_preview_figure_to_canvas" in source
    assert "figure.set_size_inches(width / figure.dpi, height / figure.dpi, forward=False)" in source
    assert "figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.96)" in source
    assert "axis.set_position([-0.08, -0.12, 1.16, 1.16])" in source
    assert "def _preview_axis_data_extents" in source
    assert "axis.set_xlim3d" in source
    assert "zoom = 2.25" in source
    assert "axis.set_anchor(\"C\")" in source
    assert "axis.set_box_aspect((x_span, y_span, z_span), zoom=zoom)" in source
    assert "self.run_button = ttk.Button(buttons, text=\"Run FEM\", command=self.run)" in source
    assert "self.progress_bar = ttk.Progressbar(buttons, mode=\"indeterminate\", length=140)" in source
    assert "self.include_end_lids = tk.BooleanVar(value=bool(self.snapshot.is_cylinder))" in source
    assert "self._add_check_row(contents, 2, \"include_end_lids\", \"Top/bottom lid\", self.include_end_lids)" in source
    assert "include_end_lids=bool(self.include_end_lids.get())" in source
    assert "self.boundary_condition = tk.StringVar(value=\"auto\")" in source
    assert "self.shell_element_order = tk.StringVar(value=\"S4\")" in source
    assert "self.analysis_type = tk.StringVar(value=\"linear eigenvalue\")" in source
    assert "self.pressure_direction = tk.StringVar(value=\"external\")" in source
    assert "self.axial_force_n = tk.DoubleVar(value=0.0)" in source
    assert "self.elastic_modulus_gpa = tk.DoubleVar(value=210.0)" in source
    assert "boundary_condition=str(self.boundary_condition.get())" in source
    assert "elastic_modulus_pa=max(_safe_float(self.elastic_modulus_gpa.get(), 210.0), 1.0e-9) * 1.0e9" in source
    assert "self.progress_bar.start(12)" in source
    assert "threading.Thread(target=worker, daemon=True)" in source
    assert "self.window.after(100, self._poll_solver_result)" in source
    assert "def _poll_solver_result(self) -> None:" in source
    assert "except queue.Empty:" in source
    assert "self._info_button(selector_bar, \"display_choice\").pack" in source
    assert "\"mesh_fidelity\"" in source
    assert "\"pressure_pa\"" in source
    assert "\"yield_stress_mpa\"" in source
    assert "self.custom_load_bc_enabled = tk.BooleanVar(value=False)" in source
    assert "self.custom_loads_add_to_imported = tk.BooleanVar(value=False)" in source
    assert "self.custom_use_nullspace_projection = tk.BooleanVar(value=False)" in source
    assert "self.deformation_scale = tk.DoubleVar(value=0.0)" in source
    assert "custom = ttk.LabelFrame(future_inputs, text=\"Custom loads and boundary conditions\")" in source
    assert "custom_loads_add_to_imported=bool(self.custom_loads_add_to_imported.get())" in source
    assert "custom_use_nullspace_projection=bool(self.custom_use_nullspace_projection.get())" in source
    assert "plate_edge_x0_support=str(self.plate_edge_x0_support.get())" in source
    assert "cylinder_upper_edge_load_n_per_m=_safe_float(self.cylinder_upper_edge_load_n_per_m.get(), 0.0)" in source
    assert "\"nullspace_projection\"" in source
    assert "horizontal_span" not in source


def test_runtime_fem_standalone_example_uses_main_application_section_preview():
    app = fe_runtime_solver.example_runtime_app()
    snapshot = fe_runtime_solver.active_line_snapshot(app)

    figure = fe_runtime_solver.create_runtime_fem_geometry_preview_figure(snapshot, app)

    assert isinstance(figure, Figure)
    assert figure.axes[0].get_title() == "3D cylinder / curved plate preview"
    assert len(figure.axes[0].collections) > 3


def test_lightweight_solver_returns_positive_fast_panel_results():
    result = fe_solver.run_lightweight_fem(
        {
            "geometry": "flat panel",
            "length_m": 2.5,
            "width_m": 0.75,
            "thickness_m": 0.012,
            "has_stiffener": True,
            "has_girder": True,
        },
        fe_solver.LightweightFEMConfig(pressure_pa=120_000.0, mesh_fidelity="coarse", num_buckling_modes=3),
    )

    assert result.status == "ok"
    assert result.stress_max_pa > 0.0
    assert result.displacement_max_m > 0.0
    assert len(result.buckling_factors) == 3
    assert result.mesh_info["beams"] == 2
    assert result.prestress_summary["critical_stress_pa"] > 0.0
    assert result.load_resultant["force_n"][2] > 0.0
    assert result.visualization["type"] == "flat"
    assert len(result.visualization["x_m"]) == 9
    assert len(result.visualization["stress_pa"]) == 9


def test_production_solver_runs_full_panel_mesh_backend():
    result = fe_solver.run_production_fem(
        {
            "geometry": "flat panel",
            "length_m": 1.2,
            "width_m": 0.6,
            "thickness_m": 0.01,
            "has_stiffener": True,
            "has_girder": True,
        },
        fe_solver.LightweightFEMConfig(pressure_pa=25_000.0, mesh_fidelity="coarse", num_buckling_modes=2),
    )

    assert result.status == "ok"
    assert result.solver_name == "ANYstructure production FE mesh"
    assert result.mesh_info["nodes"] > 0
    assert result.mesh_info["shells"] > 0
    assert result.mesh_info["beams"] > 0
    assert result.stress_max_pa >= 0.0
    assert result.displacement_max_m >= 0.0
    assert result.visualization["type"] == "flat"
    assert result.visualization["stress_pa"]


def test_production_solver_runs_full_cylinder_mesh_with_beams_and_buckling():
    result = fe_solver.run_production_fem(
        {
            "geometry": "cylinder",
            "radius_m": 1.0,
            "length_m": 1.5,
            "thickness_m": 0.02,
            "has_stiffener": True,
            "has_girder": True,
        },
        fe_solver.LightweightFEMConfig(pressure_pa=10_000.0, mesh_fidelity="coarse", num_buckling_modes=2),
    )

    assert result.status == "ok"
    assert result.mesh_info["shells"] > 0
    assert result.mesh_info["beams"] > 0
    assert result.stress_max_pa > 0.0
    assert result.displacement_max_m > 0.0
    assert result.buckling_factors == tuple(sorted(result.buckling_factors))
    assert result.buckling_factors[0] > 0.0
    assert result.visualization["type"] == "cylinder"
    assert result.visualization["stress_pa"]
    assert len(result.visualization["buckling_modes"]) >= 1
    assert result.visualization["buckling_modes"][0]["shape"]["type"] == "cylinder"


def test_generated_cylinder_mesh_uses_fb100x10_stiffener_section():
    geometry = {
        "geometry": "cylinder",
        "radius_m": 1.0,
        "length_m": 5.0,
        "thickness_m": 0.02,
        "has_stiffener": True,
        "has_girder": False,
        "stiffener_section": {
            "area": 0.1 * 0.01,
            "Iy": 0.01 * 0.1**3 / 12.0,
            "Iz": 0.1 * 0.01**3 / 12.0,
            "J": 0.01 * 0.1**3 / 12.0 + 0.1 * 0.01**3 / 12.0,
        },
    }

    generated = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="coarse"),
    )
    first_stiffener = next(beam for beam in generated["beams"] if beam["role"] == "stiffener")

    assert first_stiffener["section"]["area"] == 0.001
    assert first_stiffener["section"]["Iy"] == 0.01 * 0.1**3 / 12.0
    assert first_stiffener["section"]["Iz"] == 0.1 * 0.01**3 / 12.0


def test_flat_generated_mesh_forces_edges_at_member_lines_when_mesh_is_coarse():
    generated = fe_solver.build_generated_geometry(
        {
            "geometry": "flat panel",
            "length_m": 4.0,
            "width_m": 0.7,
            "thickness_m": 0.012,
            "has_stiffener": True,
            "has_girder": True,
        },
        fe_solver.LightweightFEMConfig(mesh_size_m=5.0),
    )

    coords = {node["id"]: tuple(node["coords"]) for node in generated["nodes"]}
    y_values = {round(coords[node_id][1], 6) for node_id in coords}
    x_values = {round(coords[node_id][0], 6) for node_id in coords}
    stiffener_beams = [beam for beam in generated["beams"] if beam["role"] == "stiffener"]
    girder_beams = [beam for beam in generated["beams"] if beam["role"] == "girder"]

    assert 0.35 in y_values
    assert 2.0 in x_values
    assert stiffener_beams
    assert girder_beams
    assert all(round(coords[node_id][1], 6) == 0.35 for beam in stiffener_beams for node_id in beam["node_ids"])
    assert all(round(coords[node_id][0], 6) == 2.0 for beam in girder_beams for node_id in beam["node_ids"])


def test_flat_generated_mesh_caps_element_size_to_stiffener_spacing():
    spacing = 0.7
    generated = fe_solver.build_generated_geometry(
        {
            "geometry": "flat panel",
            "length_m": 4.0,
            "width_m": spacing,
            "thickness_m": 0.012,
            "has_stiffener": True,
            "has_girder": False,
            "stiffener_spacing_m": spacing,
        },
        fe_solver.LightweightFEMConfig(mesh_size_m=5.0),
    )

    coords = {node["id"]: tuple(node["coords"]) for node in generated["nodes"]}
    x_values = sorted({coords[node_id][0] for node_id in coords})
    y_values = sorted({coords[node_id][1] for node_id in coords})

    assert max(b - a for a, b in zip(x_values, x_values[1:])) <= spacing + 1.0e-9
    assert max(b - a for a, b in zip(y_values, y_values[1:])) <= spacing + 1.0e-9


def test_flat_mesh_size_is_not_capped_by_disabled_members():
    requested = 3.0
    generated = fe_solver.build_generated_geometry(
        {
            "geometry": "flat panel",
            "length_m": 10.0,
            "width_m": 2.0,
            "thickness_m": 0.012,
            "has_stiffener": True,
            "has_girder": True,
            "stiffener_spacing_m": 0.5,
            "girder_spacing_m": 0.5,
        },
        fe_solver.LightweightFEMConfig(
            mesh_size_m=requested,
            include_stiffeners=False,
            include_girders=False,
        ),
    )

    coords = {node["id"]: tuple(node["coords"]) for node in generated["nodes"]}
    x_values = sorted({coords[node_id][0] for node_id in coords})
    y_values = sorted({coords[node_id][1] for node_id in coords})

    assert not generated["beams"]
    assert max(b - a for a, b in zip(x_values, x_values[1:])) <= requested + 1.0e-9
    assert max(b - a for a, b in zip(y_values, y_values[1:])) <= requested + 1.0e-9
    assert max(b - a for a, b in zip(x_values, x_values[1:])) > 0.5


def test_runtime_generated_mesh_uses_boundary_and_member_orientation_options():
    generated = fe_solver.build_generated_geometry(
        {
            "geometry": "flat panel",
            "length_m": 4.0,
            "width_m": 1.0,
            "thickness_m": 0.012,
            "has_stiffener": True,
            "has_girder": False,
            "stiffener_spacing_m": 0.5,
        },
        fe_solver.LightweightFEMConfig(
            boundary_condition="simply supported",
            member_orientation="global Z",
            stiffener_eccentricity_m=0.08,
        ),
    )

    assert generated["supports"][0]["name"] == "simple_panel_boundary"
    assert generated["supports"][0]["constraints"] == {"uz": 0.0}
    first_stiffener = next(beam for beam in generated["beams"] if beam["role"] == "stiffener")
    assert first_stiffener["section"]["orientation"] == (0.0, 0.0, 1.0)
    assert first_stiffener["section"]["eccentricity_m"] == 0.08
    assert generated["couplings"]
    assert first_stiffener["node_ids"][0] != generated["couplings"][0]["shell_node_ids"][0]
    assert generated["couplings"][0]["eccentricity"] == [0.0, 0.0, 0.08]


def test_runtime_generated_mesh_supports_s8_shells_and_enforced_displacement():
    generated = fe_solver.build_generated_geometry(
        {
            "geometry": "flat panel",
            "length_m": 2.0,
            "width_m": 1.0,
            "thickness_m": 0.012,
            "has_stiffener": False,
            "has_girder": False,
        },
        fe_solver.LightweightFEMConfig(
            mesh_fidelity="coarse",
            shell_element_order="S8",
            boundary_condition="simply supported",
            symmetry_mode="x",
            enforced_displacement_m=0.003,
        ),
    )

    assert all(len(shell["node_ids"]) == 8 for shell in generated["shells"])
    assert len(generated["nodes"]) > len(generated["plot_grid"]) * len(generated["plot_grid"][0])
    assert any(support["name"].startswith("symmetry_") for support in generated["supports"])
    assert any(support["name"] == "enforced_panel_displacement" for support in generated["supports"])


def test_custom_plate_supports_and_edge_loads_are_applied():
    geometry = {
        "geometry": "flat panel",
        "length_m": 2.0,
        "width_m": 1.0,
        "thickness_m": 0.012,
        "has_stiffener": False,
        "has_girder": False,
    }
    config = fe_solver.LightweightFEMConfig(
        pressure_pa=100.0,
        custom_load_bc_enabled=True,
        plate_edge_x0_support="fixed",
        plate_edge_x1_support="simply supported",
        plate_edge_x1_load_n_per_m=-1000.0,
    )

    generated = fe_solver.build_generated_geometry(geometry, config)
    result = fe_solver.run_production_fem(geometry, config)

    assert generated["supports"][0]["name"] == "custom_plate_x0_fixed"
    assert generated["supports"][1]["name"] == "custom_plate_x1_simply_supported"
    assert result.status == "ok"
    assert result.load_resultant["force_n"][0] == pytest.approx(-1000.0)
    assert result.load_resultant["force_n"][2] == pytest.approx(0.0)
    assert result.prestress_summary["constraint_method"] == "transformation_fixed_plus_mpc"
    assert result.prestress_summary["nullspace_projection"] == 0.0
    assert any("custom load and boundary-condition mode" in item.lower() for item in result.diagnostics)
    assert any("replace imported/generated" in item.lower() for item in result.diagnostics)


def test_custom_plate_loads_can_be_added_to_imported_pressure():
    geometry = {
        "geometry": "flat panel",
        "length_m": 2.0,
        "width_m": 1.0,
        "thickness_m": 0.012,
        "has_stiffener": False,
        "has_girder": False,
    }
    config = fe_solver.LightweightFEMConfig(
        pressure_pa=100.0,
        custom_load_bc_enabled=True,
        custom_loads_add_to_imported=True,
        plate_edge_x0_support="fixed",
        plate_edge_x1_load_n_per_m=-1000.0,
    )

    result = fe_solver.run_production_fem(geometry, config)

    assert result.status == "ok"
    assert result.load_resultant["force_n"][0] == pytest.approx(-1000.0)
    assert result.load_resultant["force_n"][2] == pytest.approx(-200.0)
    assert any("added to the imported/generated" in item.lower() for item in result.diagnostics)


def test_custom_nullspace_boundary_balances_free_body_loads():
    geometry = {
        "geometry": "flat panel",
        "length_m": 2.0,
        "width_m": 1.0,
        "thickness_m": 0.012,
        "has_stiffener": False,
        "has_girder": False,
    }
    config = fe_solver.LightweightFEMConfig(
        custom_load_bc_enabled=True,
        custom_use_nullspace_projection=True,
        plate_edge_x1_load_n_per_m=-1000.0,
    )

    generated = fe_solver.build_generated_geometry(geometry, config)
    result = fe_solver.run_production_fem(geometry, config)

    assert generated["supports"] == []
    assert result.status == "ok"
    assert result.prestress_summary["constraint_method"] == "transformation_fixed_plus_mpc_nullspace"
    assert result.prestress_summary["constraint_mode"] == "nullspace"
    assert result.prestress_summary["relative_rigid_body_load_imbalance"] > 0.0
    assert any("automatic generalized load balancing" in item.lower() for item in result.diagnostics)


def test_custom_cylinder_lid_support_and_edge_loads_are_applied_to_reference_node():
    geometry = {
        "geometry": "cylinder",
        "radius_m": 1.0,
        "length_m": 2.0,
        "thickness_m": 0.012,
        "has_stiffener": False,
        "has_girder": False,
    }
    config = fe_solver.LightweightFEMConfig(
        pressure_pa=100.0,
        include_end_lids=True,
        custom_load_bc_enabled=True,
        cylinder_lower_support="free",
        cylinder_upper_support="simply supported",
        cylinder_upper_edge_load_n_per_m=-500.0,
    )

    generated = fe_solver.build_generated_geometry(geometry, config)
    result = fe_solver.run_production_fem(geometry, config)

    assert len(generated["supports"]) == 1
    assert generated["supports"][0]["name"] == "custom_cylinder_upper_simply_supported"
    assert generated["supports"][0]["node_ids"] == [generated["rigid_lids"][1]["center_node_id"]]
    assert result.status == "ok"
    assert result.load_resultant["force_n"][2] == pytest.approx(-2.0 * math.pi * 1.0 * 500.0)


def test_cylinder_generated_mesh_forces_edges_at_stiffener_spacing_when_mesh_is_coarse():
    generated = fe_solver.build_generated_geometry(
        {
            "geometry": "cylinder",
            "radius_m": 1.0,
            "length_m": 2.0,
            "thickness_m": 0.012,
            "has_stiffener": True,
            "has_girder": False,
            "stiffener_spacing_m": 0.5,
        },
        fe_solver.LightweightFEMConfig(mesh_size_m=5.0),
    )

    row_node_ids = generated["plot_grid"][0][:-1]
    stiffener_beams = [beam for beam in generated["beams"] if beam["role"] == "stiffener"]
    stiffener_columns = {beam["node_ids"][0] for beam in stiffener_beams if beam["node_ids"][0] in row_node_ids}

    assert len(row_node_ids) == round(2.0 * math.pi / 0.5)
    assert len(stiffener_columns) == len(row_node_ids)
    assert set(row_node_ids) == stiffener_columns


def test_cylinder_generated_mesh_caps_axial_and_circumferential_size_to_stiffener_spacing():
    spacing = 0.5
    radius = 1.0
    generated = fe_solver.build_generated_geometry(
        {
            "geometry": "cylinder",
            "radius_m": radius,
            "length_m": 2.0,
            "thickness_m": 0.012,
            "has_stiffener": True,
            "has_girder": False,
            "stiffener_spacing_m": spacing,
        },
        fe_solver.LightweightFEMConfig(mesh_size_m=5.0),
    )

    row_node_ids = generated["plot_grid"][0][:-1]
    axial_values = sorted({node["coords"][2] for node in generated["nodes"]})
    circumferential_segment = 2.0 * math.pi * radius / len(row_node_ids)

    assert circumferential_segment <= spacing + 1.0e-9
    assert max(b - a for a, b in zip(axial_values, axial_values[1:])) <= spacing + 1.0e-9


def test_cylinder_mesh_fidelity_refines_real_mesh_below_member_spacing_cap():
    geometry = {
        "geometry": "cylinder",
        "radius_m": 2.0,
        "length_m": 8.0,
        "thickness_m": 0.012,
        "has_stiffener": True,
        "has_girder": True,
        "stiffener_spacing_m": 0.5,
        "girder_spacing_m": 4.0,
    }

    coarse = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="coarse", include_end_lids=True),
    )
    medium = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="medium", include_end_lids=True),
    )
    fine = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="fine", include_end_lids=True),
    )

    assert len(coarse["shells"]) < len(medium["shells"]) < len(fine["shells"])
    assert len(coarse["plot_grid"]) < len(medium["plot_grid"]) < len(fine["plot_grid"])
    assert len(coarse["plot_grid"][0]) < len(medium["plot_grid"][0]) < len(fine["plot_grid"][0])


def test_cylinder_mesh_size_is_not_capped_by_disabled_members():
    requested = 3.0
    radius = 2.0
    generated = fe_solver.build_generated_geometry(
        {
            "geometry": "cylinder",
            "radius_m": radius,
            "length_m": 8.0,
            "thickness_m": 0.012,
            "has_stiffener": True,
            "has_girder": True,
            "stiffener_spacing_m": 0.5,
            "girder_spacing_m": 0.5,
        },
        fe_solver.LightweightFEMConfig(
            mesh_size_m=requested,
            include_stiffeners=False,
            include_girders=False,
        ),
    )

    row_node_ids = generated["plot_grid"][0][:-1]
    axial_values = sorted({node["coords"][2] for node in generated["nodes"]})
    circumferential_segment = 2.0 * math.pi * radius / len(row_node_ids)

    assert not generated["beams"]
    assert circumferential_segment <= requested + 1.0e-9
    assert max(b - a for a, b in zip(axial_values, axial_values[1:])) <= requested + 1.0e-9
    assert circumferential_segment > 0.5
    assert max(b - a for a, b in zip(axial_values, axial_values[1:])) > 0.5


def test_cylinder_end_lids_are_stress_free_rigid_diaphragms():
    geometry = {
        "geometry": "cylinder",
        "radius_m": 1.0,
        "length_m": 2.0,
        "thickness_m": 0.012,
        "has_stiffener": False,
        "has_girder": False,
    }
    open_generated = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="coarse", include_end_lids=False),
    )
    lidded_generated = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="coarse", include_end_lids=True),
    )

    assert len(lidded_generated["shells"]) == len(open_generated["shells"])
    assert len(lidded_generated["nodes"]) == len(open_generated["nodes"]) + 2
    assert len(lidded_generated["rigid_lids"]) == 2
    bottom_lid, top_lid = lidded_generated["rigid_lids"]
    assert lidded_generated["supports"] == []

    backend = fe_solver.full_backend_api()
    backend_config = backend.AnyStructureFEMConfig(pressure_pa=1000.0, require_idealized_member_beams=False)
    model = backend.build_fe_model_from_generated_geometry(lidded_generated, backend_config)
    load_case = backend.build_symmetric_load_case(None, model, backend_config)
    displacements, solver_info = backend.solve_linear(model, load_case, solver_type="direct", constraint_mode="auto")
    lid_elements = [
        element
        for element in model.mesh.elements.values()
        if element.__class__.__name__ == "RigidLidMPCElement"
    ]

    assert len(lid_elements) == 2
    assert len(load_case.pressure_loads) == len(lidded_generated["shells"])
    assert not ({element.element_id for element in lid_elements} & set(load_case.pressure_loads))
    assert sum(len(element.get_mpc_constraints(model.mesh)) for element in lid_elements) > 0
    assert solver_info["convergence_info"]["status"] == "converged"
    assert solver_info["constraint_method"] == "transformation_fixed_plus_mpc_nullspace"
    assert solver_info["constraint_info"]["num_fixed_dofs"] == 0
    assert displacements[model.mesh.get_node(top_lid["center_node_id"]).dofs[2]] != 0.0
    assert displacements[model.mesh.get_node(bottom_lid["center_node_id"]).dofs[2]] != 0.0


def test_runtime_solver_records_new_analysis_material_and_load_options():
    result = fe_solver.run_production_fem(
        {
            "geometry": "cylinder",
            "radius_m": 1.0,
            "length_m": 1.5,
            "thickness_m": 0.02,
            "has_stiffener": False,
            "has_girder": False,
        },
        fe_solver.LightweightFEMConfig(
            pressure_pa=10_000.0,
            pressure_direction="internal",
            axial_force_n=25_000.0,
            shell_element_order="S8",
            analysis_type="nonlinear stability",
            buckling_analysis_type="nonlinear limit",
            symmetry_mode="cyclic",
            solver_type="direct",
            stress_percentile=90.0,
            elastic_modulus_pa=200.0e9,
            poisson_ratio=0.29,
            yield_stress_pa=300.0e6,
        ),
    )

    assert result.status == "ok"
    assert result.mesh_info["shell_order"] == "S8"
    assert "max_axial_edge_m" in result.mesh_info
    assert any("Applied balanced axial force" in item for item in result.diagnostics)
    assert any("Generated S8 shell elements" in item for item in result.diagnostics)
    assert any("Ran nonlinear tangent-stability load stepping" in item for item in result.diagnostics)
    assert any("Cyclic symmetry requested" in item for item in result.diagnostics)
    assert result.prestress_summary["nonlinear_steps"] >= 1


def test_cylinder_s8_lids_and_eccentric_members_solve_without_mpc_id_collision():
    result = fe_solver.run_production_fem(
        {
            "geometry": "cylinder",
            "radius_m": 1.0,
            "length_m": 1.5,
            "thickness_m": 0.02,
            "has_stiffener": True,
            "has_girder": True,
            "stiffener_spacing_m": 0.8,
            "girder_spacing_m": 0.75,
        },
        fe_solver.LightweightFEMConfig(
            pressure_pa=1000.0,
            include_end_lids=True,
            shell_element_order="S8",
            member_orientation="radial",
            stiffener_eccentricity_m=0.02,
            girder_eccentricity_m=0.03,
        ),
    )

    assert result.status == "ok"
    assert result.mesh_info["shell_order"] == "S8"
    assert result.mesh_info["rigid_lids"] == 2
    assert any("Applied eccentric beam-shell MPC offsets" in item for item in result.diagnostics)


def test_generated_cylinder_mesh_honors_mesh_size_and_middle_t_ring_girder():
    generated = fe_solver.build_generated_geometry(
        {
            "geometry": "cylinder",
            "radius_m": 2.0,
            "length_m": 8.0,
            "thickness_m": 0.012,
            "has_stiffener": True,
            "has_girder": True,
            "stiffener_section": {
                "area": 0.150 * 0.010,
                "Iy": 0.010 * 0.150**3 / 12.0,
                "Iz": 0.150 * 0.010**3 / 12.0,
                "J": 0.010 * 0.150**3 / 12.0 + 0.150 * 0.010**3 / 12.0,
            },
            "girder_section": {
                "area": 0.400 * 0.010 + 0.150 * 0.020,
                "Iy": 1.0e-4,
                "Iz": 1.0e-5,
                "J": 1.1e-4,
            },
        },
        fe_solver.LightweightFEMConfig(mesh_size_m=1.0),
    )

    assert len(generated["nodes"]) == 9 * 13
    assert len(generated["shells"]) == 8 * 13
    girder_rows = {
        tuple(generated["nodes"][node_id - 1]["coords"] for node_id in beam["node_ids"])[0][2]
        for beam in generated["beams"]
        if beam["role"] == "girder"
    }
    assert girder_rows == {4.0}
    assert all(beam["section"]["area"] == 0.007 for beam in generated["beams"] if beam["role"] == "girder")


def test_runtime_fem_figure_can_display_cylinder_buckling_modes():
    result = fe_solver.run_production_fem(
        {
            "geometry": "cylinder",
            "radius_m": 1.0,
            "length_m": 1.5,
            "thickness_m": 0.02,
            "has_stiffener": True,
            "has_girder": True,
        },
        fe_solver.LightweightFEMConfig(pressure_pa=10_000.0, mesh_fidelity="coarse", num_buckling_modes=2),
    )
    app = fe_runtime_solver.example_runtime_app()
    snapshot = fe_runtime_solver.active_line_snapshot(app)
    runtime_result = fe_runtime_solver.RuntimeFEMRunResult(
        status=result.status,
        summary={
            **fe_runtime_solver.runtime_geometry_summary(snapshot),
            "solver": result.solver_name,
            "max_displacement_m": result.displacement_max_m,
        },
        buckling_factors=result.buckling_factors,
        stress_percentiles=(("p95", result.stress_p95_pa), ("max", result.stress_max_pa)),
        displacement_scale=result.displacement_max_m,
        visualization=dict(result.visualization),
    )

    figure = fe_runtime_solver.create_runtime_fem_result_figure(snapshot, runtime_result, display_mode="mode:1")

    assert figure.axes[0].get_title().startswith("Buckling mode 1")
    assert figure.axes[1].get_title() == "Buckling modes"


def test_anystructure_contains_vendored_full_fe_solver_backend():
    assert fe_solver.full_backend_available() is True

    backend = fe_solver.full_backend_api()

    assert backend.AnyStructureFEMConfig.__name__ == "AnyStructureFEMConfig"
    assert callable(backend.run_anystructure_fem_mode)


def test_runtime_fem_module_has_ready_to_run_main_example():
    app = fe_runtime_solver.example_runtime_app()
    snapshot = fe_runtime_solver.active_line_snapshot(app)

    assert snapshot.line_name == "line_example"
    assert snapshot.pressure_pa == 100_000.0
    assert snapshot.domain == "cylinder"
    assert snapshot.is_cylinder is True
    summary = fe_runtime_solver.runtime_geometry_summary(snapshot)
    assert summary["radius_m"] == 2.0
    assert summary["length_m"] == 8.0
    assert summary["thickness_m"] == 0.012
    assert summary["stiffener_section"]["label"] == "FB150x10"
    assert summary["stiffener_section"]["area"] == 0.0015
    assert summary["girder_section"]["label"] == "T400x10+150x20"
    assert app._fem_default_top_bottom_moment_nm == 30_000_000.0


def test_startup_cylinder_example_runs_near_200_mpa_with_buckling_modes():
    app = fe_runtime_solver.example_runtime_app()
    snapshot = fe_runtime_solver.active_line_snapshot(app)

    result = fe_runtime_solver.run_runtime_fem(
        snapshot,
        fe_runtime_solver.RuntimeFEMOptions(
            mesh_fidelity="coarse",
            pressure_pa=snapshot.pressure_pa,
            load_scale=1.0,
            include_stiffeners=True,
            include_girders=True,
            num_buckling_modes=5,
            mesh_size_m=0.0,
            top_bottom_moment_nm=app._fem_default_top_bottom_moment_nm,
        ),
    )

    assert result.status == "ok"
    assert 150.0 <= result.stress_percentiles[0][1] / 1.0e6 <= 250.0
    assert len(result.buckling_factors) == 5
    assert result.buckling_factors[0] > 0.1
    assert len(result.visualization["buckling_modes"]) == 5


def test_runtime_fem_file_can_be_run_directly_from_pycharm():
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "fe_runtime_solver.py").read_text(encoding="utf-8")

    assert 'if __package__ in (None, ""):' in source
    assert "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))" in source
    assert 'if __name__ == "__main__":' in source
    assert "root.withdraw()" not in source
    assert "RuntimeFEMWindow(root, example_runtime_app(), use_parent_as_window=True)" in source


def test_active_line_snapshot_rejects_missing_structure():
    app = _FakeApp()
    app._active_line = "line2"

    try:
        fe_runtime_solver.active_line_snapshot(app)
    except ValueError as error:
        assert "active line is not available" in str(error)
    else:
        raise AssertionError("missing active line should fail")
