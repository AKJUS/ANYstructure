from pathlib import Path

import pytest

from anystruct import fe_plate_fields


SAMPLE_INP = Path(r"C:\Users\AudunArnesenNyhus\OneDrive - Cefront\Desktop\Analysis-1.inp")
SAMPLE_FRD = Path(r"C:\Users\AudunArnesenNyhus\OneDrive - Cefront\Desktop\Analysis-1.frd")
ADVANCED_SAMPLE_INP = Path(r"C:\Users\AudunArnesenNyhus\OneDrive - Cefront\Desktop\Analysis-2.inp")
ADVANCED_SAMPLE_FRD = Path(r"C:\Users\AudunArnesenNyhus\OneDrive - Cefront\Desktop\Analysis-2.frd")


def _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange="T"):
    nodes = {}
    elements = {}
    next_node = 1
    next_element = 1

    def add_node(point):
        nonlocal next_node
        nodes[next_node] = point
        next_node += 1
        return next_node - 1

    def add_quad(points):
        nonlocal next_element
        node_ids = [add_node(point) for point in points]
        elements[next_element] = node_ids
        next_element += 1

    # Base plate: one element per bay, keeping the intended bay split visible.
    x0, x1 = 0.0, 4.0
    for y0, y1 in zip(web_stations[:-1], web_stations[1:]):
        add_quad([(x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)])

    # Webs: vertical shell plates on each station.
    for y in web_stations:
        add_quad([(x0, y, 0.0), (x1, y, 0.0), (x1, y, 0.4), (x0, y, 0.4)])

    # Optional flanges: horizontal shell plates at the web tips.
    if flange:
        for y in web_stations:
            if flange == "T":
                y0, y1 = y - 0.075, y + 0.075
            elif flange == "L":
                y0, y1 = y, y + 0.15
            else:
                raise AssertionError("Unexpected flange type")
            add_quad([(x0, y0, 0.4), (x1, y0, 0.4), (x1, y1, 0.4), (x0, y1, 0.4)])

    lines = [
        "*Heading",
        "*Node",
    ]
    for node_id, (x, y, z) in nodes.items():
        lines.append(f"{node_id}, {x:.9g}, {y:.9g}, {z:.9g}")
    lines.append("*Element, Type=S4, Elset=Compound-1")
    for element_id, node_ids in elements.items():
        lines.append(f"{element_id}, " + ", ".join(str(node_id) for node_id in node_ids))
    lines.append("*Elset, Elset=ShellSection")
    lines.append(", ".join(str(element_id) for element_id in elements))
    lines.append("*Shell section, Elset=ShellSection, Material=S355, Offset=0")
    lines.append("0.015")
    filename = tmp_path / "panel.inp"
    filename.write_text("\n".join(lines), encoding="utf-8")
    return filename


def _unique_members(fields):
    members = {}
    for field in fields:
        for member in field.members:
            members[member.member_id] = member
    return [members[key] for key in sorted(members)]


def _write_synthetic_frd(path, model, stress_by_node=None):
    stress_by_node = stress_by_node or {}
    lines = [
        f"    2C{len(model.nodes):>30}         1",
    ]
    for node_id, point in sorted(model.nodes.items()):
        lines.append(f" -1{node_id:10d}{point[0]:12.5E}{point[1]:12.5E}{point[2]:12.5E}")
    lines.append(" -3")
    lines.append(f"    3C{len(model.shell_elements):>30}         1")
    for element_id, element in sorted(model.shell_elements.items()):
        lines.append(f" -1{element_id:10d}    4    0    1")
        lines.append(" -2" + "".join(f"{node_id:10d}" for node_id in element.corner_node_ids))
    lines.append(" -3")
    lines.extend(
        [
            " -4  STRESS      6    1",
            " -5  SXX         1    4    1    1",
            " -5  SYY         1    4    2    2",
            " -5  SZZ         1    4    3    3",
            " -5  SXY         1    4    1    2",
            " -5  SYZ         1    4    2    3",
            " -5  SZX         1    4    3    1",
        ]
    )
    for node_id in sorted(model.nodes):
        values = stress_by_node.get(node_id, (-100.0e6, -50.0e6, 0.0, 5.0e6, 0.0, 0.0))
        lines.append(f" -1{node_id:10d}" + "".join(f"{value:12.5E}" for value in values))
    lines.append(" -3")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_read_calculix_inp_keeps_shell_section_metadata(tmp_path):
    inp = _write_panel_inp(tmp_path)

    model = fe_plate_fields.read_calculix_inp(inp)

    assert len(model.nodes) == 32
    assert len(model.shell_elements) == 8
    assert model.shell_sections[0].thickness_m == 0.015
    assert model.shell_sections[0].material == "S355"


def test_infer_t_stiffened_panel_from_synthetic_mesh(tmp_path):
    inp = _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange="T")
    model = fe_plate_fields.read_calculix_inp(inp)

    fields = fe_plate_fields.infer_plate_fields(model)
    members = _unique_members(fields)

    assert len(fields) == 2
    assert len(members) == 3
    assert {member.section_type for member in members} == {"T"}
    assert fields[0].span_m == pytest.approx(4.0)
    assert fields[0].spacing_m == pytest.approx(0.7)
    assert members[0].web_height_m == pytest.approx(0.4)
    assert members[0].flange_width_m == pytest.approx(0.15)
    assert members[0].web_thickness_m == pytest.approx(0.015)
    assert members[0].flange_thickness_m == pytest.approx(0.015)
    assert members[0].thickness_source == "shell section metadata"
    assert fields[0].shell_section_thickness_m == 0.015


def test_infer_flatbar_stiffeners_without_flanges(tmp_path):
    inp = _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange=None)
    model = fe_plate_fields.read_calculix_inp(inp)

    fields = fe_plate_fields.infer_plate_fields(model)
    members = _unique_members(fields)

    assert len(fields) == 2
    assert len(members) == 3
    assert {member.section_type for member in members} == {"FB"}
    assert all(member.flange_width_m is None for member in members)


def test_infer_one_sided_l_stiffeners(tmp_path):
    inp = _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange="L")
    model = fe_plate_fields.read_calculix_inp(inp)

    fields = fe_plate_fields.infer_plate_fields(model)
    members = _unique_members(fields)

    assert len(fields) == 2
    assert len(members) == 3
    assert {member.section_type for member in members} == {"L"}
    assert members[0].flange_width_m == pytest.approx(0.15)


def test_infer_unstiffened_base_plate(tmp_path):
    inp = _write_panel_inp(tmp_path, web_stations=(0.0,), flange=None)
    model = fe_plate_fields.read_calculix_inp(inp)

    fields = fe_plate_fields.infer_plate_fields(model)

    assert len(fields) == 1
    assert fields[0].attached_member_ids == ()
    assert fields[0].span_m == pytest.approx(4.0)


def test_plot_plate_fields_saves_png(tmp_path):
    inp = _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange="T")
    model = fe_plate_fields.read_calculix_inp(inp)
    fields = fe_plate_fields.infer_plate_fields(model)
    plot_path = tmp_path / "panels.png"

    fig = fe_plate_fields.plot_plate_fields(model, fields, output_path=plot_path)

    assert fig is not None
    assert plot_path.exists()
    assert plot_path.read_bytes().startswith(b"\x89PNG")


def test_plot_plate_fields_can_color_by_uf(tmp_path):
    inp = _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange="T")
    model = fe_plate_fields.read_calculix_inp(inp)
    fields = fe_plate_fields.infer_plate_fields(model)
    plot_path = tmp_path / "panels_uf.png"
    field_values = {field.field_id: index + 0.5 for index, field in enumerate(fields)}

    fig = fe_plate_fields.plot_plate_fields(
        model,
        fields,
        output_path=plot_path,
        field_values=field_values,
        value_label="UF",
    )

    assert fig is not None
    assert len(fig.axes) == 2
    assert plot_path.exists()
    assert plot_path.read_bytes().startswith(b"\x89PNG")


def test_panel_3d_records_preserve_elevation_and_orientation(tmp_path):
    model = fe_plate_fields.FeShellModel(
        nodes={
            1: (0.0, 0.0, 0.0),
            2: (1.0, 0.0, 0.0),
            3: (1.0, 1.0, 0.0),
            4: (0.0, 1.0, 0.0),
            5: (2.0, 0.0, 0.5),
            6: (2.0, 1.0, 0.5),
            7: (2.0, 1.0, 1.5),
            8: (2.0, 0.0, 1.5),
        },
        shell_elements={
            1: fe_plate_fields.ShellElement(1, (1, 2, 3, 4), element_type="S4"),
            2: fe_plate_fields.ShellElement(2, (5, 6, 7, 8), element_type="S4"),
        },
    )
    fields = [
        fe_plate_fields.PlateField(
            field_id="field_low",
            base_patch_id="patch_low",
            element_ids=(1,),
            bbox=((0.0, 1.0), (0.0, 1.0), (0.0, 0.0)),
            span_m=1.0,
            spacing_m=1.0,
            transverse_bounds=(0.0, 1.0),
            attached_member_ids=(),
        ),
        fe_plate_fields.PlateField(
            field_id="field_vertical",
            base_patch_id="patch_vertical",
            element_ids=(2,),
            bbox=((2.0, 2.0), (0.0, 1.0), (0.5, 1.5)),
            span_m=1.0,
            spacing_m=1.0,
            transverse_bounds=(0.0, 1.0),
            attached_member_ids=(),
        ),
    ]

    records = fe_plate_fields.panel_3d_records(model, fields)

    assert records[0]["bbox"][2] == (0.0, 0.0)
    assert records[1]["bbox"][0] == (2.0, 2.0)
    assert records[1]["bbox"][2] == (0.5, 1.5)
    assert records[0]["normal"] == pytest.approx((0.0, 0.0, 1.0))
    assert abs(records[1]["normal"][0]) == pytest.approx(1.0)
    assert len(records[0]["polygons"]) == 1
    assert len(records[1]["polygons"]) == 1
    assert len(records[0]["polygons"][0]) == 4


def test_buckling_usage_factor_is_read_from_nested_anystructure_results():
    item = {
        "result": {
            "Buckling strength": {
                "Actual usage Factor": [None, "0.87"],
            }
        }
    }

    assert fe_plate_fields._selected_uf_from_buckling_result(item) == pytest.approx(0.87)


def test_plot_plate_fields_3d_saves_png(tmp_path):
    inp = _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange="T")
    model = fe_plate_fields.read_calculix_inp(inp)
    fields = fe_plate_fields.infer_plate_fields(model)
    plot_path = tmp_path / "panels_3d.png"

    fig = fe_plate_fields.plot_plate_fields_3d(model, fields, output_path=plot_path)

    assert fig is not None
    assert plot_path.exists()
    assert plot_path.read_bytes().startswith(b"\x89PNG")


def test_read_calculix_frd_summary_discovers_result_blocks(tmp_path):
    frd = tmp_path / "sample.frd"
    frd.write_text(
        "\n".join(
            [
                "    1UMAT    1S355",
                "    2C                         12                                     1",
                "    1PSTEP                         2           1           1",
                "  100CL  101 1.000000000          12                     0    1           1",
                " -4  STRESS      6    1",
                " -5  SXX         1    4    1    1",
                " -5  SYY         1    4    2    2",
            ]
        ),
        encoding="utf-8",
    )

    summary = fe_plate_fields.read_calculix_frd_summary(frd)

    assert summary["node_count"] == 12
    assert summary["materials"] == ["1S355"]
    assert summary["result_blocks"][0]["name"] == "STRESS"
    assert summary["result_blocks"][0]["components"] == ["SXX", "SYY"]


def test_reduce_field_stresses_projects_frd_to_compression_positive_mpa(tmp_path):
    inp = _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange="T")
    model = fe_plate_fields.read_calculix_inp(inp)
    fields = fe_plate_fields.infer_plate_fields(model)
    frd = _write_synthetic_frd(tmp_path / "panel.frd", model)

    frd_stress = fe_plate_fields.read_calculix_frd_stress(frd)
    panel_stresses = fe_plate_fields.reduce_field_stresses(model, fields, frd_stress)

    assert len(panel_stresses) == 2
    assert panel_stresses[0].sigma_x1_mpa == pytest.approx(100.0)
    assert panel_stresses[0].sigma_x2_mpa == pytest.approx(100.0)
    assert panel_stresses[0].sigma_y1_mpa == pytest.approx(50.0)
    assert panel_stresses[0].sigma_y2_mpa == pytest.approx(50.0)
    assert panel_stresses[0].tau_xy_mpa == pytest.approx(5.0)
    assert panel_stresses[0].sample_count > 0


def test_calculate_field_buckling_returns_per_field_results(tmp_path):
    inp = _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange="T")
    model = fe_plate_fields.read_calculix_inp(inp)
    fields = fe_plate_fields.infer_plate_fields(model)
    frd = _write_synthetic_frd(tmp_path / "panel.frd", model)
    frd_stress = fe_plate_fields.read_calculix_frd_stress(frd)
    panel_stresses = fe_plate_fields.reduce_field_stresses(model, fields, frd_stress)

    buckling_results = fe_plate_fields.calculate_field_buckling(
        fields,
        panel_stresses,
        calculation_method="SemiAnalytical S3/U3",
    )

    assert len(buckling_results) == len(fields)
    assert {result["field_id"] for result in buckling_results} == {field.field_id for field in fields}
    assert all("available" in result for result in buckling_results)


def test_create_fea_buckling_session_prepares_selectable_anystructure_panels(tmp_path):
    inp = _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange="T")
    model = fe_plate_fields.read_calculix_inp(inp)
    frd = _write_synthetic_frd(tmp_path / "panel.frd", model)

    session = fe_plate_fields.create_fea_buckling_session(inp, frd, run_buckling=False)

    assert session.panel_count == 2
    assert session.field_count == 2
    first = session.panels[0]
    assert first.field_id == "field_001"
    assert first.stress.sigma_x1_mpa == pytest.approx(100.0)
    assert first.anystructure_input["calculation_domain"] == "Flat plate, stiffened"
    assert first.anystructure_input["geometry"]["span_mm"] == pytest.approx(4000.0)
    assert first.anystructure_input["geometry"]["spacing_mm"] == pytest.approx(700.0)
    assert first.anystructure_input["geometry"]["plate_thickness_mm"] == pytest.approx(15.0)
    assert first.anystructure_input["section"]["web_height_mm"] == pytest.approx(400.0)
    assert first.anystructure_input["section"]["flange_width_mm"] == pytest.approx(150.0)
    assert first.anystructure_input["stresses"]["sigma_y1_mpa"] == pytest.approx(50.0)
    assert len(first.plot_bounds) == 4
    assert session.summary()["panels"][0]["field_id"] == "field_001"
    assert session.summary()["panels"][0]["surface_3d"]["polygons"]


def test_public_api_exposes_fea_result_buckling_session(tmp_path):
    from anystruct import api

    inp = _write_panel_inp(tmp_path, web_stations=(0.0, 0.7, 1.4), flange="T")
    model = fe_plate_fields.read_calculix_inp(inp)
    frd = _write_synthetic_frd(tmp_path / "panel.frd", model)

    session = api.create_fea_result_buckling_session(inp, frd, run_buckling=False)
    summary = api.analyze_fea_result_buckling(inp, frd, run_buckling=False)

    assert session.panel_count == 2
    assert summary["field_count"] == 2
    assert summary["panels"][0]["anystructure_input"]["stresses"]["tau_xy_mpa"] == pytest.approx(5.0)


@pytest.mark.skipif(not SAMPLE_INP.exists(), reason="Provided PrePoMax sample is not available")
def test_provided_prepomax_sample_geometry_counts():
    model = fe_plate_fields.read_calculix_inp(SAMPLE_INP)
    fields = fe_plate_fields.infer_plate_fields(model)
    members = _unique_members(fields)
    frd_summary = fe_plate_fields.read_calculix_frd_summary(SAMPLE_FRD) if SAMPLE_FRD.exists() else None

    assert len(model.nodes) == 24725
    assert len(model.shell_elements) == 8080
    assert len(members) == 15
    assert sum(1 for member in members if member.flange_patch_id is not None) == 15
    assert len(fields) == 14
    assert fields[0].span_m == pytest.approx(4.0)
    assert fields[0].spacing_m == pytest.approx(0.714286, abs=1.0e-6)
    assert members[0].web_height_m == pytest.approx(0.4)
    assert members[0].flange_width_m == pytest.approx(0.15)
    assert {member.section_type for member in members} == {"T"}
    if frd_summary is not None:
        assert {"DISP", "STRESS", "TOSTRAIN", "FORC"}.issubset(
            {block["name"] for block in frd_summary["result_blocks"]}
        )


@pytest.mark.skipif(not ADVANCED_SAMPLE_INP.exists(), reason="Advanced PrePoMax sample is not available")
def test_advanced_prepomax_sample_detects_stiffeners_girder_and_thicknesses():
    model = fe_plate_fields.read_calculix_inp(ADVANCED_SAMPLE_INP)
    fields = fe_plate_fields.infer_plate_fields(model)
    members = _unique_members(fields)
    stiffeners = [member for member in members if member.role == "stiffener"]
    girders = [member for member in members if member.role == "girder"]
    frd_summary = (
        fe_plate_fields.read_calculix_frd_summary(ADVANCED_SAMPLE_FRD)
        if ADVANCED_SAMPLE_FRD.exists()
        else None
    )

    assert len(model.nodes) == 3976
    assert len(model.shell_elements) == 1323
    assert len(fields) == 28
    assert len(stiffeners) == 15
    assert len(girders) == 1
    assert sum(1 for member in members if member.flange_patch_id is not None) == 16
    assert fields[0].shell_section_thickness_m == pytest.approx(0.01)
    assert fields[0].span_m == pytest.approx(4.0)
    assert fields[0].spacing_m == pytest.approx(0.714286, abs=1.0e-6)
    assert {member.section_type for member in members} == {"T"}
    assert stiffeners[0].web_height_m == pytest.approx(0.4)
    assert stiffeners[0].flange_width_m == pytest.approx(0.15)
    assert stiffeners[0].web_thickness_m == pytest.approx(0.015)
    assert stiffeners[0].flange_thickness_m == pytest.approx(0.02)
    assert girders[0].web_height_m == pytest.approx(0.8)
    assert girders[0].flange_width_m == pytest.approx(0.2)
    assert girders[0].web_thickness_m == pytest.approx(0.015)
    assert girders[0].flange_thickness_m == pytest.approx(0.02)
    if frd_summary is not None:
        assert {"DISP", "STRESS", "TOSTRAIN", "FORC"}.issubset(
            {block["name"] for block in frd_summary["result_blocks"]}
        )
