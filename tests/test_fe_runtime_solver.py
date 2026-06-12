from matplotlib.figure import Figure
from pathlib import Path
import math

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


def test_runtime_fem_popup_has_compact_3d_section_preview():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeApp())

    figure = fe_runtime_solver.create_runtime_fem_geometry_preview_figure(snapshot)

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 1
    assert figure.axes[0].get_title() == "3D section view"
    assert hasattr(figure.axes[0], "get_zlim")


def test_runtime_fem_popup_wires_preview_canvas_in_upper_right():
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "fe_runtime_solver.py").read_text(encoding="utf-8")

    assert "body = ttk.Panedwindow(outer, orient=tk.HORIZONTAL)" in source
    assert "body.add(left_panel, weight=2)" in source
    assert "body.add(mid_panel, weight=2)" in source
    assert "body.add(right_panel, weight=3)" in source
    assert "future_inputs = ttk.LabelFrame(mid_panel, text=\"Additional inputs\")" in source
    assert "preview = ttk.LabelFrame(right_panel, text=\"3D section view\")" in source
    assert "preview.pack(fill=tk.BOTH, expand=True, pady=(0, 10))" in source
    assert "self._show_preview_figure(create_runtime_fem_geometry_preview_figure(self.snapshot, self.app), preview)" in source
    assert "self.preview_canvas = FigureCanvasTkAgg(figure, master=parent)" in source
    assert "redraw_after_id" in source
    assert "def _fit_preview_figure_to_canvas" in source
    assert "figure.set_size_inches(width / figure.dpi, height / figure.dpi, forward=False)" in source
    assert "figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)" in source
    assert "axis.set_position([0.01, 0.03, 0.98, 0.93])" in source
    assert "axis.set_box_aspect((x_span, y_span, z_span), zoom=zoom)" in source
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
