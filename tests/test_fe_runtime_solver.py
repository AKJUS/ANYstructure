from matplotlib.figure import Figure
from pathlib import Path
import json
import math

import pytest

from anystruct import api, fe_plate_fields, fe_runtime_solver, fe_solver


class _Plate:
    girder_lg = 3.5

    def get_structure_type(self):
        return "Flat plate, stiffened"

    def get_span(self):
        return 2.5

    def get_s(self):
        return 0.75

    def get_pl_thk(self):
        return 0.012

    def get_puls_up_boundary(self):
        return "SSSS"


class _PlateWithMixedSupports(_Plate):
    def get_puls_up_boundary(self):
        return "CSSC"


class _TMember:
    stiffener_type = "T"

    def __init__(self, spacing=0.75, web_h=0.4, web_t=0.012, flange_w=0.15, flange_t=0.02):
        self._spacing = spacing
        self._web_h = web_h
        self._web_t = web_t
        self._flange_w = flange_w
        self._flange_t = flange_t

    @property
    def hw(self):
        return self._web_h * 1000.0

    @property
    def tw(self):
        return self._web_t * 1000.0

    @property
    def b(self):
        return self._flange_w * 1000.0

    @property
    def tf(self):
        return self._flange_t * 1000.0

    def get_s(self):
        return self._spacing


class _AllStructure:
    Plate = _Plate()
    Stiffener = object()
    Girder = object()
    _panel_length_Lp = None


class _FakeApp:
    _active_line = "line1"
    _line_dict = {"line1": [1, 2]}
    _line_to_struc = {"line1": [_AllStructure(), None, None, object(), None, None]}

    def get_highest_pressure(self, line):
        assert line == "line1"
        return {"normal": 12345.0}


class _AllStructureMixedSupports(_AllStructure):
    Plate = _PlateWithMixedSupports()


class _FakeAppMixedSupports(_FakeApp):
    _line_to_struc = {"line1": [_AllStructureMixedSupports(), None, None, object(), None, None]}


class _AllStructureWithFlatMembers:
    Plate = _Plate()
    Stiffener = _TMember(spacing=0.75, web_h=0.4, web_t=0.012, flange_w=0.15, flange_t=0.02)
    Girder = _TMember(spacing=1.25, web_h=0.8, web_t=0.02, flange_w=0.20, flange_t=0.03)
    _panel_length_Lp = 7.5


class _FakeAppFlatMembers(_FakeApp):
    _line_to_struc = {"line1": [_AllStructureWithFlatMembers(), None, None, object(), None, None]}


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
    assert summary["length_m"] == 5.0
    assert summary["width_m"] == 3.5
    assert summary["thickness_m"] == 0.012
    assert summary["has_stiffener"] is True
    assert summary["has_girder"] is True
    assert summary["stiffener_spacing_m"] == 0.75
    assert summary["girder_spacing_m"] == 2.5
    assert summary["girder_length_m"] == 3.5
    assert summary["plate_edge_supports"] == ("simply supported", "simply supported", "simply supported", "simply supported")


def test_runtime_geometry_summary_reads_flat_plate_edge_supports_from_line_properties():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeAppMixedSupports())

    summary = fe_runtime_solver.runtime_geometry_summary(snapshot)

    assert summary["plate_edge_supports"] == ("fixed", "simply supported", "simply supported", "fixed")


def test_runtime_geometry_summary_imports_flat_stiffener_and_girder_sections():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeAppFlatMembers())

    summary = fe_runtime_solver.runtime_geometry_summary(snapshot)

    assert summary["stiffener_section"]["label"] == "T400x12+150x20"
    assert summary["stiffener_section"]["area"] == pytest.approx(0.0048 + 0.003)
    assert summary["length_m"] == pytest.approx(7.5)
    assert summary["width_m"] == pytest.approx(3.5)
    assert summary["girder_spacing_m"] == pytest.approx(2.5)
    assert summary["girder_section"]["label"] == "T800x20+200x30"
    assert summary["girder_section"]["area"] == pytest.approx(0.016 + 0.006)


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
    import_payload = result.visualization["fea_result_import"]
    assert import_payload["format"] == "anystructure-runtime-fe-results-v1"
    assert len(import_payload["shells"]) == result.summary["mesh_info"]["shells"]
    assert import_payload["stress_components"] == ("SXX", "SYY", "SZZ", "SXY", "SYZ", "SZX")
    assert import_payload["nodal_stress_pa"]
    session = fe_plate_fields.create_runtime_fea_buckling_session(result, run_buckling=False, geometry_type="flat")
    api_summary = api.analyze_runtime_fea_result_buckling(result, run_buckling=False, geometry_type="flat")
    assert session.panel_count > 0
    assert session.panels[0].stress is not None
    assert session.panels[0].stress.sample_count > 0
    assert api_summary["field_count"] == session.field_count
    assert result.stress_percentiles[0][0] == "p95"
    assert result.stress_percentiles[0][1] > 0.0
    assert result.buckling_factors == tuple(sorted(result.buckling_factors))


def test_run_runtime_fem_flat_member_geometry_matches_generated_fe_model():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeAppFlatMembers())
    summary = fe_runtime_solver.runtime_geometry_summary(snapshot)

    generated = fe_solver.build_generated_geometry(
        summary,
        fe_solver.LightweightFEMConfig(mesh_fidelity="coarse", include_stiffeners=True, include_girders=True),
    )
    result = fe_runtime_solver.run_runtime_fem(
        snapshot,
        fe_runtime_solver.RuntimeFEMOptions(
            mesh_fidelity="coarse",
            pressure_pa=10_000.0,
            include_stiffeners=True,
            include_girders=True,
            num_buckling_modes=1,
        ),
    )

    stiffener = next(beam for beam in generated["beams"] if beam["role"] == "stiffener")
    girder = next(beam for beam in generated["beams"] if beam["role"] == "girder")
    roles = {line["role"] for line in result.visualization["member_lines"]}

    assert stiffener["section"]["label"] == "T400x12+150x20"
    assert stiffener["section"]["area"] == pytest.approx(0.0078)
    assert girder["section"]["label"] == "T800x20+200x30"
    assert girder["section"]["area"] == pytest.approx(0.022)
    assert result.summary["mesh_info"]["beams"] == len(generated["beams"])
    assert {"stiffener", "girder"} <= roles


def test_runtime_flat_girder_panel_uses_lg_width_and_span_girder_stations():
    snapshot = fe_runtime_solver.active_line_snapshot(_FakeAppFlatMembers())
    summary = fe_runtime_solver.runtime_geometry_summary(snapshot)

    generated = fe_solver.build_generated_geometry(
        summary,
        fe_solver.LightweightFEMConfig(mesh_size_m=5.0, include_stiffeners=True, include_girders=True),
    )
    coords = {node["id"]: tuple(node["coords"]) for node in generated["nodes"]}
    x_values = {round(coord[0], 6) for coord in coords.values()}
    y_values = {round(coord[1], 6) for coord in coords.values()}
    girder_x_values = {
        round(coords[node_id][0], 6)
        for beam in generated["beams"]
        if beam["role"] == "girder"
        for node_id in beam["node_ids"]
    }
    stiffener_y_values = {
        round(coords[node_id][1], 6)
        for beam in generated["beams"]
        if beam["role"] == "stiffener"
        for node_id in beam["node_ids"]
    }

    assert max(x_values) == pytest.approx(7.5)
    assert max(y_values) == pytest.approx(3.5)
    assert girder_x_values == {2.5, 5.0}
    assert stiffener_y_values == {0.25, 1.0, 1.75, 2.5, 3.25}


def test_standalone_girder_panel_centers_cut_bays_for_symmetric_stress_model():
    snapshot = fe_runtime_solver.active_line_snapshot(fe_runtime_solver.example_runtime_app())
    summary = fe_runtime_solver.runtime_geometry_summary(snapshot)

    generated = fe_solver.build_generated_geometry(
        summary,
        fe_solver.LightweightFEMConfig(mesh_size_m=5.0, include_stiffeners=True, include_girders=True),
    )
    coords = {node["id"]: tuple(node["coords"]) for node in generated["nodes"]}
    girder_x_values = sorted(
        {
            round(coords[node_id][0], 6)
            for beam in generated["beams"]
            if beam["role"] == "girder"
            for node_id in beam["node_ids"]
        }
    )
    stiffener_y_values = sorted(
        {
            round(coords[node_id][1], 6)
            for beam in generated["beams"]
            if beam["role"] == "stiffener"
            for node_id in beam["node_ids"]
        }
    )

    assert summary["length_m"] == pytest.approx(10.0)
    assert summary["width_m"] == pytest.approx(10.0)
    assert girder_x_values == [1.5, 5.0, 8.5]
    assert stiffener_y_values[0] == pytest.approx(0.125)
    assert stiffener_y_values[-1] == pytest.approx(9.875)
    assert all(
        (left + right) == pytest.approx(10.0)
        for left, right in zip(stiffener_y_values, reversed(stiffener_y_values))
    )


def test_standalone_girder_pressure_static_deflection_is_downward_not_nullspace_balanced():
    snapshot = fe_runtime_solver.active_line_snapshot(fe_runtime_solver.example_runtime_app())
    result = fe_runtime_solver.run_runtime_fem(
        snapshot,
        fe_runtime_solver.RuntimeFEMOptions(
            mesh_fidelity="coarse",
            shell_element_order="S4",
            pressure_pa=snapshot.pressure_pa,
            include_stiffeners=True,
            include_girders=True,
            num_buckling_modes=1,
        ),
    )
    w_values = [value for row in result.visualization.get("w_m", ()) for value in row]
    prestress = result.summary.get("prestress_summary", {})

    assert result.status == "ok"
    assert prestress["constraint_method"] == "transformation_fixed_plus_mpc"
    assert prestress["constraint_mode"] == "transformation"
    assert prestress["nullspace_projection"] == pytest.approx(0.0)
    assert min(w_values) < 0.0
    assert max(w_values) <= 1.0e-4


def test_runtime_fem_matplotlib_figure_contains_geometry_axis():
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
    assert len(figure.axes) >= 1
    assert figure.axes[0].get_title() == "Static stress/displacement"
    assert figure.axes[0].lines


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


def test_flat_automatic_nullspace_keeps_physical_edge_pressure_supports():
    geometry = {
        "geometry": "flat panel",
        "length_m": 1.0,
        "width_m": 1.0,
        "thickness_m": 0.01,
        "has_stiffener": False,
        "has_girder": False,
    }
    config = fe_solver.LightweightFEMConfig(boundary_condition="nullspace", pressure_pa=1000.0, mesh_fidelity="coarse")

    generated = fe_solver.build_generated_geometry(geometry, config)
    result = fe_solver.run_production_fem(geometry, config)

    assert {support["name"] for support in generated["supports"]} >= {
        "plate_x0_simply_supported",
        "plate_x1_simply_supported",
        "plate_y0_simply_supported",
        "plate_y1_simply_supported",
    }
    assert all({"uz": 0.0}.items() <= support["constraints"].items() for support in generated["supports"] if support["name"].startswith("plate_"))
    assert result.status == "ok"
    assert result.prestress_summary["constraint_mode"] == "nullspace"
    assert result.prestress_summary["nullspace_projection"] == pytest.approx(1.0)
    assert "Applied flat-panel edge supports from line properties, defaulting to simply supported edges when unspecified." in result.diagnostics


def test_flat_automatic_supports_use_imported_line_property_pattern():
    generated = fe_solver.build_generated_geometry(
        {
            "geometry": "flat panel",
            "length_m": 2.0,
            "width_m": 1.0,
            "thickness_m": 0.01,
            "has_stiffener": False,
            "has_girder": False,
            "plate_edge_supports": ("fixed", "simply supported", "simply supported", "fixed"),
        },
        fe_solver.LightweightFEMConfig(boundary_condition="auto", mesh_fidelity="coarse"),
    )

    supports = {support["name"]: support["constraints"] for support in generated["supports"]}

    assert supports["plate_x0_fixed"] == {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}
    assert supports["plate_x1_simply_supported"] == {"uz": 0.0}
    assert supports["plate_y0_simply_supported"] == {"uz": 0.0}
    assert supports["plate_y1_fixed"] == {"ux": 0.0, "uy": 0.0, "uz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0}


def test_custom_manual_pressure_replaces_imported_flat_pressure():
    result = fe_solver.run_production_fem(
        {
            "geometry": "flat panel",
            "length_m": 2.0,
            "width_m": 1.0,
            "thickness_m": 0.01,
            "has_stiffener": False,
            "has_girder": False,
        },
        fe_solver.LightweightFEMConfig(
            pressure_pa=1000.0,
            custom_load_bc_enabled=True,
            custom_pressure_pa=500.0,
            plate_edge_x0_support="simply supported",
            plate_edge_x1_support="simply supported",
            plate_edge_y0_support="simply supported",
            plate_edge_y1_support="simply supported",
        ),
    )

    assert result.status == "ok"
    assert result.load_resultant["force_n"][2] == pytest.approx(-1000.0)
    assert "Custom loads replace imported/generated pressure, axial force and end moment inputs." in result.diagnostics
    assert "Applied custom manual pressure: 500.0 Pa." in result.diagnostics


def test_imported_flat_force_and_moment_are_balanced_on_opposite_edges():
    result = fe_solver.run_production_fem(
        {
            "geometry": "flat panel",
            "length_m": 2.0,
            "width_m": 1.0,
            "thickness_m": 0.01,
            "has_stiffener": False,
            "has_girder": False,
        },
        fe_solver.LightweightFEMConfig(pressure_pa=0.0, axial_force_n=1000.0, top_bottom_moment_nm=200.0, mesh_fidelity="coarse"),
    )

    assert result.status == "ok"
    assert result.load_resultant["force_n"] == pytest.approx((0.0, 0.0, 0.0))
    assert result.load_resultant["moment_nm"] == pytest.approx((0.0, 0.0, 0.0))


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
    assert "self.options_notebook = ttk.Notebook(mid_panel)" in source
    assert "constraints = ttk.LabelFrame(tab_general, text=\"Supports and load path\")" in source
    assert "solver_options = ttk.LabelFrame(tab_general, text=\"Solver\")" in source
    assert "members = ttk.LabelFrame(tab_properties, text=\"Member modelling\")" in source
    assert "material = ttk.LabelFrame(tab_properties, text=\"Material and recovery\")" in source
    assert "self.upper_result_frame = ttk.LabelFrame(right_panel, text=\"Result text\")" in source
    assert "self.upper_result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))" in source
    assert "self.upper_result_text = tk.Text(" in source
    assert "self.result_canvas = Tkinter3DCanvas(" in source
    assert "self.interactive_3d_checkbox = ttk.Checkbutton(" in source
    assert "def _populate_canvas_with_geometry(" in source
    assert "def _populate_canvas_with_results(" in source
    assert "self.run_button = ttk.Button(buttons, text=\"Run FEM\", command=self.run)" in source
    assert "text=\"Use for buckling\"" in source
    assert "command=self._send_results_to_fea_buckling" in source
    assert "def _send_results_to_fea_buckling(self) -> None:" in source
    assert "import_runtime_fem_buckling_result" in source
    assert "self.progress_bar = ttk.Progressbar(buttons, mode=\"indeterminate\", length=140)" in source
    assert "self.include_end_lids = tk.BooleanVar(value=bool(self.snapshot.is_cylinder))" in source
    assert "self._add_check_row(contents, 2, \"include_end_lids\", \"Top/bottom lid\", self.include_end_lids)" in source
    assert "include_end_lids=bool(self.include_end_lids.get())" in source
    assert "self.boundary_condition = tk.StringVar(value=\"auto\")" in source
    assert "self.shell_element_order = tk.StringVar(value=\"S4\")" in source
    assert "self.beam_element_order = tk.StringVar(value=\"B2\")" in source
    assert "self.member_model = tk.StringVar(value=\"plates as shell, girders as beams\")" in source
    assert "self.analysis_type = tk.StringVar(value=\"linear eigenvalue\")" in source
    assert "self.pressure_direction = tk.StringVar(value=\"external\")" in source
    assert "self.axial_force_n = tk.DoubleVar(value=0.0)" in source
    assert "self.elastic_modulus_gpa = tk.DoubleVar(value=210.0)" in source
    assert "self.plate_alpha_vis = tk.StringVar(value=\"0.99\")" in source
    assert "boundary_condition=str(self.boundary_condition.get())" in source
    assert "beam_element_order=str(self.beam_element_order.get())" in source
    assert "member_model=str(self.member_model.get())" in source
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
    assert "self.custom_pressure_pa = tk.DoubleVar(value=0.0)" in source
    assert "self.custom_loads_json = tk.StringVar(value=\"[]\")" in source
    assert "self._custom_load_entries: list[dict[str, Any]] = []" in source
    assert "self._custom_selected_edge_keys: set[tuple[str, float, float, float]] = set()" in source
    assert "self.deformation_scale = tk.StringVar(value=\"0.0\")" in source
    assert "custom = ttk.LabelFrame(tab_advanced, text=\"Custom loads and boundary conditions\")" in source
    assert "selection = ttk.LabelFrame(custom, text=\"Panel and edge selection\")" in source
    assert "load_list = ttk.LabelFrame(tab_advanced, text=\"Loads to run\")" in source
    assert "ttk.Button(actions, text=\"Add load\", command=self._add_custom_load_from_selection)" in source
    assert "ttk.Button(actions, text=\"Delete load\", command=self._delete_selected_custom_load)" in source
    assert "self._custom_load_tree = ttk.Treeview(" in source
    assert "canvas.canvas.bind(\"<ButtonRelease-3>\", self._on_custom_load_edge_canvas_release, add=\"+\")" in source
    assert "def _custom_load_selection_visual_offset(self) -> float:" in source
    assert "draw_overlay=True" in source
    assert "custom_loads_add_to_imported=bool(self.custom_loads_add_to_imported.get())" in source
    assert "custom_use_nullspace_projection=bool(self.custom_use_nullspace_projection.get())" in source
    assert "custom_pressure_pa=_safe_float(self.custom_pressure_pa.get(), 0.0)" in source
    assert "custom_loads_json=str(self.custom_loads_json.get())" in source
    assert "self._add_entry_row(time_domain, 2, \"custom_pressure_pa\"" not in source
    assert "ttk.Button(view_actions, text=\"ISO\", command=lambda: self._set_runtime_3d_view(\"iso\"))" in source
    assert "ttk.Button(view_actions, text=\"Front\", command=lambda: self._set_runtime_3d_view(\"front\"))" in source
    assert "ttk.Button(view_actions, text=\"Top\", command=lambda: self._set_runtime_3d_view(\"top\"))" in source
    assert "def _set_runtime_3d_view(self, view_name: str) -> None:" in source
    assert "self._add_option_row(" in source and "\"member_model\"" in source
    assert "\"webs as shells, flanges as beams\"" in source
    assert "\"all shell\"" in source
    assert "plate_edge_x0_support=str(self.plate_edge_x0_support.get())" in source
    assert "cylinder_upper_edge_load_n_per_m=_safe_float(self.cylinder_upper_edge_load_n_per_m.get(), 0.0)" in source
    assert "\"nullspace_projection\"" in source
    assert "horizontal_span" not in source


def test_runtime_fem_standalone_example_uses_main_application_section_preview():
    app = fe_runtime_solver.example_runtime_app()
    snapshot = fe_runtime_solver.active_line_snapshot(app)

    figure = fe_runtime_solver.create_runtime_fem_geometry_preview_figure(snapshot, app)

    assert isinstance(figure, Figure)
    assert figure.axes[0].get_title() == "3D stiffened panel with girder"
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


def test_runtime_generated_mesh_supports_b2_and_b3_beam_elements():
    geometry = {
        "geometry": "flat panel",
        "length_m": 2.0,
        "width_m": 1.0,
        "thickness_m": 0.012,
        "has_stiffener": True,
        "has_girder": True,
        "stiffener_spacing_m": 0.5,
        "girder_spacing_m": 1.0,
    }

    b2 = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="coarse", beam_element_order="B2"),
    )
    b3_config = fe_solver.LightweightFEMConfig(mesh_fidelity="coarse", beam_element_order="B3")
    b3 = fe_solver.build_generated_geometry(geometry, b3_config)

    assert b2["beams"]
    assert b3["beams"]
    assert {len(beam["node_ids"]) for beam in b2["beams"]} == {2}
    assert {len(beam["node_ids"]) for beam in b3["beams"]} == {3}
    assert fe_solver._mesh_size_diagnostics(b2)["beam_order"] == "B2"
    assert fe_solver._mesh_size_diagnostics(b3)["beam_order"] == "B3"


def test_runtime_generated_mesh_supports_member_shell_modelling_modes():
    geometry = {
        "geometry": "flat panel",
        "length_m": 2.0,
        "width_m": 1.0,
        "thickness_m": 0.012,
        "has_stiffener": True,
        "has_girder": True,
        "stiffener_spacing_m": 0.5,
        "girder_spacing_m": 1.0,
        "stiffener_section": {
            "web_height": 0.2,
            "web_thickness": 0.01,
            "flange_width": 0.08,
            "flange_thickness": 0.012,
        },
        "girder_section": {
            "web_height": 0.3,
            "web_thickness": 0.012,
            "flange_width": 0.12,
            "flange_thickness": 0.014,
        },
    }

    current = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="coarse", member_model="plates as shell, girders as beams"),
    )
    mixed = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="coarse", member_model="webs as shells, flanges as beams"),
    )
    all_shell = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="coarse", member_model="all shell"),
    )

    assert len(mixed["shells"]) > len(current["shells"])
    assert any(str(shell.get("role", "")).endswith("_web") for shell in mixed["shells"])
    assert any(str(beam.get("role", "")).endswith("_flange") for beam in mixed["beams"])
    assert len(all_shell["shells"]) > len(mixed["shells"])
    assert not all_shell["beams"]
    assert any(str(shell.get("role", "")).endswith("_flange") for shell in all_shell["shells"])
    node_by_id = {int(node["id"]): node for node in mixed["nodes"]}
    web_roles_by_node: dict[int, set[str]] = {}
    for shell in mixed["shells"]:
        role = str(shell.get("role", ""))
        if role in {"stiffener_web", "girder_web"}:
            for node_id in shell["node_ids"]:
                web_roles_by_node.setdefault(int(node_id), set()).add(role)
    shared_above_plate = [
        node_id
        for node_id, roles in web_roles_by_node.items()
        if {"stiffener_web", "girder_web"} <= roles
        and abs(float(node_by_id[node_id]["coords"][2])) > 1.0e-9
    ]
    assert shared_above_plate

    if fe_solver._full_backend is not None:
        for member_model in ("webs as shells, flanges as beams", "all shell"):
            result = fe_solver.run_production_fem(
                geometry,
                fe_solver.LightweightFEMConfig(
                    mesh_fidelity="coarse",
                    pressure_pa=1000.0,
                    member_model=member_model,
                    runtime_solver="static only",
                    boundary_condition="fixed",
                ),
            )
            assert result.status == "ok"
            assert any("Member modelling:" in item for item in result.diagnostics)


def test_cylinder_member_shell_web_crossings_share_connection_nodes():
    geometry = {
        "geometry": "cylinder",
        "radius_m": 1.0,
        "length_m": 2.0,
        "thickness_m": 0.012,
        "has_stiffener": True,
        "has_girder": True,
        "stiffener_spacing_m": math.pi / 2.0,
        "girder_spacing_m": 1.0,
        "stiffener_section": {
            "web_height": 0.2,
            "web_thickness": 0.01,
            "flange_width": 0.08,
            "flange_thickness": 0.012,
        },
        "girder_section": {
            "web_height": 0.35,
            "web_thickness": 0.012,
            "flange_width": 0.12,
            "flange_thickness": 0.014,
        },
    }

    generated = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(mesh_fidelity="coarse", member_model="webs as shells, flanges as beams"),
    )

    node_by_id = {int(node["id"]): node for node in generated["nodes"]}
    web_roles_by_node: dict[int, set[str]] = {}
    for shell in generated["shells"]:
        role = str(shell.get("role", ""))
        if role in {"stiffener_web", "girder_web"}:
            for node_id in shell["node_ids"]:
                web_roles_by_node.setdefault(int(node_id), set()).add(role)
    shared_inside_shell = []
    for node_id, roles in web_roles_by_node.items():
        if not {"stiffener_web", "girder_web"} <= roles:
            continue
        x, y, _z = (float(value) for value in node_by_id[node_id]["coords"])
        if math.hypot(x, y) < 1.0 - 1.0e-9:
            shared_inside_shell.append(node_id)

    assert shared_inside_shell


def test_flat_member_shell_boundary_conditions_include_generated_edge_shell_nodes():
    geometry = {
        "geometry": "flat panel",
        "length_m": 2.0,
        "width_m": 1.0,
        "thickness_m": 0.012,
        "has_stiffener": True,
        "has_girder": True,
        "stiffener_spacing_m": 0.5,
        "girder_spacing_m": 1.0,
        "stiffener_section": {
            "web_height": 0.2,
            "web_thickness": 0.01,
            "flange_width": 0.08,
            "flange_thickness": 0.012,
        },
        "girder_section": {
            "web_height": 0.3,
            "web_thickness": 0.012,
            "flange_width": 0.12,
            "flange_thickness": 0.014,
        },
    }
    configs = (
        fe_solver.LightweightFEMConfig(
            mesh_fidelity="coarse",
            member_model="all shell",
            boundary_condition="fixed",
        ),
        fe_solver.LightweightFEMConfig(
            mesh_fidelity="coarse",
            member_model="all shell",
            custom_load_bc_enabled=True,
            plate_edge_x0_support="fixed",
            plate_edge_x1_support="fixed",
            plate_edge_y0_support="fixed",
            plate_edge_y1_support="fixed",
        ),
    )

    for config in configs:
        generated = fe_solver.build_generated_geometry(geometry, config)
        support_nodes = {int(node_id) for support in generated["supports"] for node_id in support["node_ids"]}
        shell_nodes = {int(node_id) for shell in generated["shells"] for node_id in shell["node_ids"]}
        node_by_id = {int(node["id"]): node for node in generated["nodes"]}
        edge_shell_nodes = {
            node_id
            for node_id in shell_nodes
            if abs(float(node_by_id[node_id]["coords"][0])) <= 1.0e-8
            or abs(float(node_by_id[node_id]["coords"][0]) - geometry["length_m"]) <= 1.0e-8
            or abs(float(node_by_id[node_id]["coords"][1])) <= 1.0e-8
            or abs(float(node_by_id[node_id]["coords"][1]) - geometry["width_m"]) <= 1.0e-8
        }

        assert edge_shell_nodes
        assert edge_shell_nodes <= support_nodes


def test_cylinder_member_shell_boundary_conditions_cover_generated_end_shell_nodes():
    geometry = {
        "geometry": "cylinder",
        "radius_m": 1.0,
        "length_m": 2.0,
        "thickness_m": 0.012,
        "has_stiffener": True,
        "has_girder": True,
        "stiffener_spacing_m": math.pi / 2.0,
        "girder_spacing_m": 1.0,
        "stiffener_section": {
            "web_height": 0.2,
            "web_thickness": 0.01,
            "flange_width": 0.08,
            "flange_thickness": 0.012,
        },
        "girder_section": {
            "web_height": 0.35,
            "web_thickness": 0.012,
            "flange_width": 0.12,
            "flange_thickness": 0.014,
        },
    }

    no_lid = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(
            mesh_fidelity="coarse",
            member_model="all shell",
            boundary_condition="fixed",
            include_end_lids=False,
        ),
    )
    support_nodes = {int(node_id) for support in no_lid["supports"] for node_id in support["node_ids"]}
    node_by_id = {int(node["id"]): node for node in no_lid["nodes"]}
    end_shell_nodes = {
        int(node_id)
        for shell in no_lid["shells"]
        for node_id in shell["node_ids"]
        if abs(float(node_by_id[int(node_id)]["coords"][2])) <= 1.0e-8
        or abs(float(node_by_id[int(node_id)]["coords"][2]) - geometry["length_m"]) <= 1.0e-8
    }

    assert end_shell_nodes
    assert end_shell_nodes <= support_nodes

    with_lid = fe_solver.build_generated_geometry(
        geometry,
        fe_solver.LightweightFEMConfig(
            mesh_fidelity="coarse",
            member_model="all shell",
            boundary_condition="fixed",
            include_end_lids=True,
        ),
    )
    node_by_id = {int(node["id"]): node for node in with_lid["nodes"]}
    end_shell_nodes = {
        int(node_id)
        for shell in with_lid["shells"]
        for node_id in shell["node_ids"]
        if abs(float(node_by_id[int(node_id)]["coords"][2])) <= 1.0e-8
        or abs(float(node_by_id[int(node_id)]["coords"][2]) - geometry["length_m"]) <= 1.0e-8
    }
    lid_ring_nodes = {int(node_id) for lid in with_lid["rigid_lids"] for node_id in lid["ring_node_ids"]}
    support_nodes = {int(node_id) for support in with_lid["supports"] for node_id in support["node_ids"]}
    lid_reference_nodes = {int(lid["center_node_id"]) for lid in with_lid["rigid_lids"]}

    assert end_shell_nodes <= lid_ring_nodes
    assert support_nodes == lid_reference_nodes


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


def test_custom_selected_internal_edge_load_adds_mesh_breaks_and_resultant():
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
        plate_edge_x0_support="fixed",
        custom_selected_edge_load_n_per_m=250.0,
        custom_edge_segments_json=(
            '[{"varying_axis":"a","fixed_coordinate":0.5,'
            '"start_coordinate":0.5,"end_coordinate":1.5}]'
        ),
    )

    generated = fe_solver.build_generated_geometry(geometry, config)
    coords = {node["id"]: tuple(node["coords"]) for node in generated["nodes"]}
    result = fe_solver.run_production_fem(geometry, config)

    assert 0.5 in {round(coord[0], 6) for coord in coords.values()}
    assert 1.5 in {round(coord[0], 6) for coord in coords.values()}
    assert 0.5 in {round(coord[1], 6) for coord in coords.values()}
    assert result.status == "ok"
    assert result.load_resultant["force_n"][1] == pytest.approx(250.0)
    assert any("selected edge segments" in item.lower() for item in result.diagnostics)


def test_saved_custom_load_entries_add_panel_and_edge_breaks_to_mesh():
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
        custom_loads_json=json.dumps([
            {
                "type": "pressure",
                "pressure_pa": 500.0,
                "patches": [{"min_a": 0.25, "max_a": 0.75, "min_b": 0.2, "max_b": 0.6}],
            },
            {
                "type": "edge",
                "line_load_n_per_m": 125.0,
                "edges": [{"varying_axis": "a", "fixed_coordinate": 0.5, "start_coordinate": 0.4, "end_coordinate": 0.8}],
            },
        ]),
    )

    generated = fe_solver.build_generated_geometry(geometry, config)
    coords = {node["id"]: tuple(node["coords"]) for node in generated["nodes"]}

    assert {0.25, 0.75, 0.4, 0.8}.issubset({round(coord[0], 6) for coord in coords.values()})
    assert {0.2, 0.5, 0.6}.issubset({round(coord[1], 6) for coord in coords.values()})
    assert fe_solver._custom_pressure_patch_count(config) == 1
    assert len(fe_solver._custom_edge_segments(config)) == 1


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


def test_custom_cylinder_lid_support_and_edge_loads_constrain_reference_node_kinematics():
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
    assert generated["supports"][0]["constraints"] == {"uz": 0.0, "rx": 0.0, "ry": 0.0}
    assert result.status == "ok"
    assert result.load_resultant["force_n"][2] == pytest.approx(-2.0 * math.pi * 1.0 * 500.0)


def test_custom_cylinder_single_fixed_lid_constrains_reference_rotations():
    geometry = {
        "geometry": "cylinder",
        "radius_m": 1.0,
        "length_m": 2.0,
        "thickness_m": 0.012,
        "has_stiffener": False,
        "has_girder": False,
    }
    config = fe_solver.LightweightFEMConfig(
        include_end_lids=True,
        custom_load_bc_enabled=True,
        cylinder_lower_support="fixed",
        cylinder_upper_support="free",
    )

    generated = fe_solver.build_generated_geometry(geometry, config)

    assert len(generated["supports"]) == 1
    assert generated["supports"][0]["name"] == "custom_cylinder_lower_fixed"
    assert generated["supports"][0]["node_ids"] == [generated["rigid_lids"][0]["center_node_id"]]
    assert generated["supports"][0]["constraints"] == {
        "ux": 0.0,
        "uy": 0.0,
        "uz": 0.0,
        "rx": 0.0,
        "ry": 0.0,
        "rz": 0.0,
    }


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
    assert "nonlinear_steps" in result.prestress_summary
    assert result.prestress_summary["nonlinear_status"] in {
        "completed",
        "limit_point_detected",
        "near_limit_point",
        "initial_tangent_not_positive",
    }


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
        fe_solver.LightweightFEMConfig(pressure_pa=10_000.0, mesh_fidelity="coarse", num_buckling_modes=2, include_end_lids=True),
    )
    app = fe_runtime_solver.example_runtime_app("cylinder")
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


def test_anystructure_contains_vendored_full_fe_solver_backend():
    assert fe_solver.full_backend_available() is True

    backend = fe_solver.full_backend_api()

    assert backend.AnyStructureFEMConfig.__name__ == "AnyStructureFEMConfig"
    assert callable(backend.run_anystructure_fem_mode)
    assert callable(backend.solve_transient_newmark)
    assert backend.PressurePatch.__name__ == "PressurePatch"
    assert callable(backend.apply_imperfection)
    assert backend.StandardImperfection.__name__ == "StandardImperfection"
    assert backend.CapacityWorkflowConfig.__name__ == "CapacityWorkflowConfig"
    assert callable(backend.run_nonlinear_capacity_workflow)
    assert backend.RecoveryConfig.__name__ == "RecoveryConfig"
    assert backend.ResourceConfig.__name__ == "ResourceConfig"
    assert backend.FactorizationCache.__name__ == "FactorizationCache"
    assert callable(backend.solve_free_vibration)


def test_production_solver_can_use_anyintelligent_capacity_workflow_path():
    result = fe_solver.run_production_fem(
        {
            "geometry": "cylinder",
            "radius_m": 1.0,
            "length_m": 1.0,
            "thickness_m": 0.02,
            "has_stiffener": False,
            "has_girder": False,
        },
        fe_solver.LightweightFEMConfig(
            pressure_pa=10_000.0,
            mesh_fidelity="coarse",
            num_buckling_modes=1,
            include_end_lids=True,
            runtime_solver="ANYintelligent capacity workflow",
            imperfection_enabled=True,
            imperfection_amplitude_m=0.0001,
            nonlinear_max_load_factor=0.5,
            nonlinear_steps=1,
            capacity_mesh_min_elements_per_half_wave=1,
        ),
    )

    prestress = result.prestress_summary

    assert result.status == "ok"
    assert prestress["runtime_solver"] == "anyintelligent capacity workflow"
    assert prestress["capacity_workflow_status"] == "completed"
    assert prestress["capacity_workflow_capacity_factor"] == pytest.approx(0.5)
    assert prestress["capacity_workflow_mesh_status"] == "ok"
    assert prestress["buckling_solver_status"] == "ok"
    assert result.buckling_factors and result.buckling_factors[0] > 0.0
    assert any("capacity workflow completed" in item.lower() for item in result.diagnostics)


def test_production_solver_runs_custom_time_domain_response_and_stress_free_imperfection():
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
            pressure_pa=100.0,
            mesh_fidelity="coarse",
            num_buckling_modes=1,
            imperfection_enabled=True,
            imperfection_amplitude_m=0.001,
            imperfection_wave_a=1,
            imperfection_wave_b=1,
            custom_load_bc_enabled=True,
            custom_pressure_pa=1000.0,
            custom_time_domain_enabled=True,
            custom_time_domain_duration_s=0.001,
            custom_time_domain_total_time_s=0.002,
            custom_time_domain_dt_s=0.001,
        ),
    )

    prestress = result.prestress_summary

    assert result.status == "ok"
    assert prestress["imperfection_status"] == "applied"
    assert prestress["imperfection_kind"] == "plate half-wave"
    assert prestress["imperfection_max_offset_m"] == pytest.approx(0.001)
    assert prestress["custom_time_domain_status"] == "completed"
    assert prestress["custom_time_domain_selected_shells"] == pytest.approx(result.mesh_info["shells"])
    assert prestress["custom_time_domain_peak_displacement_m"] > 0.0
    assert any("Custom time-domain response completed" in item for item in result.diagnostics)


def test_runtime_result_print_and_gui_source_include_custom_time_domain_and_imperfection_inputs():
    result = fe_runtime_solver.RuntimeFEMRunResult(
        status="ok",
        summary={
            "geometry": "flat panel",
            "line": "line1",
            "mesh_fidelity": "coarse",
            "shell_element_order": "S4",
            "boundary_condition": "auto",
            "symmetry_mode": "none",
            "analysis_type": "linear eigenvalue",
            "buckling_analysis_type": "linear eigenvalue",
            "solver_type": "direct",
            "pressure_pa": 0.0,
            "pressure_direction": "external",
            "axial_force_n": 0.0,
            "enforced_displacement_m": 0.0,
            "mesh_size_m": 0.0,
            "top_bottom_moment_nm": 0.0,
            "include_stiffeners": True,
            "include_girders": True,
            "include_end_lids": False,
            "member_orientation": "auto",
            "stiffener_eccentricity_m": 0.0,
            "girder_eccentricity_m": 0.0,
            "material_model": "linear elastic",
            "steel_grade": "S355",
            "steel_thickness_class": "auto",
            "elastic_modulus_pa": 210.0e9,
            "poisson_ratio": 0.3,
            "yield_stress_pa": 355.0e6,
            "stress_percentile": 95.0,
            "nonlinear_max_load_factor": 3.0,
            "nonlinear_steps": 12,
            "nonlinear_layers": 5,
            "nonlinear_max_iterations": 25,
            "deformation_scale": 0.0,
            "custom_load_bc_enabled": False,
            "num_buckling_modes": 1,
            "max_displacement_m": 0.0,
            "imperfection_enabled": True,
            "imperfection_shape": "standard plate/cylinder",
            "imperfection_amplitude_m": 0.001,
            "imperfection_wave_a": 1,
            "imperfection_wave_b": 2,
            "custom_time_domain_enabled": True,
            "custom_pressure_pa": 1000.0,
            "custom_time_domain_duration_s": 0.001,
            "custom_time_domain_total_time_s": 0.002,
            "custom_time_domain_dt_s": 0.001,
            "custom_pressure_patch_count": 1,
            "custom_pressure_patch_area_m2": 0.18,
            "custom_edge_segment_count": 2,
            "custom_selected_edge_load_n_per_m": 300.0,
            "custom_time_domain_include_static_load": False,
            "prestress_summary": {
                "imperfection_status": "applied",
                "imperfection_kind": "plate half-wave",
                "imperfection_amplitude_m": 0.001,
                "imperfection_max_offset_m": 0.001,
                "imperfection_waves_a": 1,
                "imperfection_waves_b": 2,
                "custom_time_domain_status": "completed",
                "custom_time_domain_selected_shells": 16,
                "custom_time_domain_peak_displacement_m": 0.0002,
                "custom_time_domain_peak_von_mises_pa": 2.5e6,
            },
        },
    )

    text = fe_runtime_solver.format_runtime_fem_result(result)
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "fe_runtime_solver.py").read_text(encoding="utf-8")

    assert "Geometric imperfection input:" in text
    assert "Custom time-domain input:" in text
    assert "Applied geometric imperfection:" in text
    assert "Custom time-domain response:" in text
    assert "self.custom_time_domain_enabled = tk.BooleanVar(value=False)" in source
    assert "self.imperfection_enabled = tk.BooleanVar(value=False)" in source
    assert "Custom time-domain load" in source
    assert "Imperfections" in source


def test_runtime_fem_module_has_ready_to_run_main_example():
    app = fe_runtime_solver.example_runtime_app()
    snapshot = fe_runtime_solver.active_line_snapshot(app)

    assert snapshot.line_name == "line_example"
    assert snapshot.pressure_pa == 459_639.0
    assert snapshot.domain == "Flat plate, stiffened with girder"
    assert snapshot.is_cylinder is False
    summary = fe_runtime_solver.runtime_geometry_summary(snapshot)
    assert summary["length_m"] == pytest.approx(10.0)
    assert summary["width_m"] == pytest.approx(10.0)
    assert summary["thickness_m"] == pytest.approx(0.018)
    assert summary["stiffener_spacing_m"] == pytest.approx(0.75)
    assert summary["girder_spacing_m"] == pytest.approx(3.5)
    assert summary["stiffener_section"]["label"] == "T400x12+250x12"
    assert summary["girder_section"]["label"] == "T800x20+200x30"
    assert app._fem_default_top_bottom_moment_nm == 0.0


def test_runtime_fem_module_keeps_cylinder_standalone_example_option():
    app = fe_runtime_solver.example_runtime_app("cylinder")
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


def test_startup_cylinder_example_runs_near_200_mpa_with_buckling_modes():
    app = fe_runtime_solver.example_runtime_app("cylinder")
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
    assert "self.window.transient(parent)" not in source
    assert "state(\"zoomed\")" in source
    assert 'if __name__ == "__main__":' in source
    assert "root.withdraw()" not in source
    assert "argparse.ArgumentParser" in source
    assert "choices=(\"girder_panel\", \"cylinder\")" in source
    assert "RuntimeFEMWindow(root, example_runtime_app(args.example), use_parent_as_window=True)" in source


def test_active_line_snapshot_rejects_missing_structure():
    app = _FakeApp()
    app._active_line = "line2"

    try:
        fe_runtime_solver.active_line_snapshot(app)
    except ValueError as error:
        assert "active line is not available" in str(error)
    else:
        raise AssertionError("missing active line should fail")


def test_populate_canvas_with_geometry_outer_vs_intermediate_stiffeners():
    class DummyCanvas:
        def __init__(self):
            self.polygons = []

        def add_polygon(self, points, color=None, outline=None, **kwargs):
            self.polygons.append((points, color, outline))

        def add_cylinder(self, *args, **kwargs):
            pass

        def add_ring_stiffener(self, *args, **kwargs):
            pass

        def add_longitudinal_stiffener(self, *args, **kwargs):
            pass

        def after_idle(self, func):
            pass

        def fit_to_scene(self):
            pass

    class DummyWindow:
        def __init__(self):
            self.snapshot = fe_runtime_solver.active_line_snapshot(fe_runtime_solver.example_runtime_app())

        def _populate_canvas_with_geometry(self, canvas):
            fe_runtime_solver.RuntimeFEMWindow._populate_canvas_with_geometry(self, canvas)

    window = DummyWindow()
    canvas = DummyCanvas()
    window._populate_canvas_with_geometry(canvas)

    stiffeners = [
        poly for poly in canvas.polygons
        if poly[1] == "#94a3b8"  # Web color
    ]
    
    assert len(stiffeners) > 2
    from collections import defaultdict
    stf_by_y = defaultdict(list)
    for stf in stiffeners:
        points = stf[0]
        y_val = round(points[0].y, 4)
        stf_by_y[y_val].append(points)

    assert len(stf_by_y) > 2
    for y_val, segments in stf_by_y.items():
        assert len(segments) == 2
        spans = sorted([(round(min(p.x for p in seg), 2), round(max(p.x for p in seg), 2)) for seg in segments])
        assert spans == [(0.0, 5.0), (5.0, 10.0)]
