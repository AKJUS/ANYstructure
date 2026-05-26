import copy
from pathlib import Path

from anystruct import (
    calc_loads,
    calc_structure,
    example_data,
    line_structure,
    project_application,
    project_services,
)
from anystruct.project_state import ProjectState


class DummyStructure:
    def __init__(self, calculation_domain="Flat plate, stiffened"):
        self.need_recalc = False
        self.calculation_domain = calculation_domain
        self.properties = None

    def set_main_properties(self, properties):
        self.properties = properties

    def get_main_properties(self):
        return {"domain": self.calculation_domain}


class DummyFatigue:
    def __init__(self):
        self.main_properties = None

    def set_main_properties(self, properties):
        self.main_properties = properties

    def get_fatigue_properties(self):
        return {"fatigue": "properties"}


class DummyLoad:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def get_name(self):
        return self._name

    def __str__(self):
        return self._value

    def get_load_parmeters(self):
        return [self._name, self._value]


class DummyCylinder:
    def get_all_properties(self):
        return {"shell": "properties"}


class DummyTank:
    def get_parameters(self):
        return {"tank": "properties"}


class DummySesamExport:
    def __init__(self, points, lines, sections, line_to_struc):
        self.points = points
        self.lines = lines
        self.sections = sections
        self.line_bundles = line_to_struc
        self.output_lines = []

    def write_points(self):
        self.output_lines.append("points\n")

    def write_lines(self):
        self.output_lines.append("lines\n")

    def write_sections(self):
        self.output_lines.append("sections\n")

    def write_beams(self):
        self.output_lines.append("beams\n")


class DummyReport:
    def __init__(self, filename, title, seconds, source_data):
        self.arguments = (filename, title, seconds, source_data)
        self.document_created = False
        self.pdf_saved = False

    def createDocument(self):
        self.document_created = True

    def savePDF(self):
        self.pdf_saved = True

    def createTable(self):
        return ["table"]


class DummyReportDocument:
    def __init__(self, filename):
        self.filename = filename
        self.elements = None

    def build(self, elements):
        self.elements = elements


class DummyExcelWorkbook:
    def __init__(self, path, visible, read_only):
        self.path = path
        self.visible = visible
        self.read_only = read_only
        self.closed = False

    def get_sheet_data(self, sheet):
        return {
            "flat_plate": [["flat header"], [f"flat {idx}" for idx in range(43)]],
            "cylinder": [["cylinder header"], [f"cylinder {idx}" for idx in range(43)]],
        }[sheet]

    def close_book(self):
        self.closed = True


def test_project_edit_service_uses_typed_records_for_legacy_geometry_mutations():
    points = {"point1": [0.0, 0.0], "point2": [1.0, 0.0]}
    lines = {}
    service = project_services.ProjectEditService(points, lines)

    point = service.add_point(service.next_point_name(), (1.0, 1.0))
    line = service.add_line("point1", "point3")

    assert point == project_services.PointRecord(name="point3", x=1.0, y=1.0)
    assert points["point3"] == [1.0, 1.0]
    assert line == project_services.LineRecord(name="line1", first_point="point1", second_point="point3")
    assert line.endpoint_keys == ("p1p3", "p3p1")
    assert lines["line1"] == [1, 3]


def test_project_edit_service_rejects_duplicate_points_and_reversed_lines():
    points = {"point1": [0.0, 0.0], "point2": [1.0, 0.0]}
    lines = {"line1": [1, 2]}
    service = project_services.ProjectEditService(points, lines)

    assert service.add_point(service.next_point_name(), (1.0, 0.0)) is None
    assert service.add_line("point2", "point1") is None
    assert lines == {"line1": [1, 2]}


def test_project_edit_service_removes_connected_lines_and_points():
    points = {"point1": [0.0, 0.0], "point2": [1.0, 0.0], "point3": [1.0, 1.0]}
    lines = {"line1": [1, 2], "line2": [2, 3]}
    service = project_services.ProjectEditService(points, lines)

    assert service.connected_line_names("point2") == ["line1", "line2"]
    assert service.remove_line("line1").endpoint_keys == ("p1p2", "p2p1")
    assert service.remove_point("point1") == project_services.PointRecord("point1", 0.0, 0.0)
    assert "line1" not in lines
    assert "point1" not in points


def test_recalculation_service_uses_named_bundle_access_and_skips_empty_slots():
    assigned = DummyStructure()
    line_bundles = {
        "line1": [assigned, None, None, [], {}, None],
        "line2": [None, None, None, [], {}, None],
    }

    invalidated = project_services.mark_lines_for_recalculation(line_bundles)

    assert invalidated == ["line1"]
    assert assigned.need_recalc


def test_line_structure_service_assigns_and_updates_legacy_bundle_slots():
    structure = DummyStructure()
    line_bundles = {}
    service = project_services.LineStructureService(line_bundles)

    service.assign_structure("line1", structure, cylinder="shell")
    service.update_structure_properties("line1", {"Plate": {"span": 1}})
    service.set_cylinder("line1", "updated shell")

    assert line_bundles["line1"] == [structure, None, None, [None], {}, "updated shell"]
    assert structure.properties == {"Plate": {"span": 1}}
    assert structure.need_recalc


def test_line_structure_service_syncs_or_removes_fatigue_for_updated_domain():
    fatigue = DummyFatigue()
    line_bundles = {"line1": [DummyStructure(), None, fatigue, [], {}, None]}
    service = project_services.LineStructureService(line_bundles)

    service.sync_fatigue_after_structure_update("line1", {"Stiffener": {"spacing": 0.7}})
    service.replace_structure("line1", DummyStructure("Flat plate, unstiffened"))
    service.sync_fatigue_after_structure_update("line1", {"Stiffener": {"spacing": 1.0}})

    assert fatigue.main_properties == {"spacing": 0.7}
    assert line_bundles["line1"][2] is None


def _flat_structure_property_request(calculation_domain="Flat plate, stiffened with girder"):
    return project_services.FlatStructurePropertyRequest(
        calculation_domain=calculation_domain,
        base_values={
            "material": 355,
            "material_factor": 1.15,
            "span": 4000,
            "spacing": 700,
            "plate_thk": 20,
            "stf_web_h": 400,
            "stf_web_t": 12,
            "stf_fl_w": 150,
            "stf_fl_t": 20,
            "structure_type": "GENERAL_INTERNAL_WT",
            "stf_type": "T",
            "sigma_y1": 90,
            "sigma_y2": 90,
            "sigma_x1": 40,
            "sigma_x2": 40,
            "tau_xy": 5,
            "plate_kpp": 1,
            "stf_kps": 1,
            "stf_km1": 12,
            "stf_km2": 24,
            "stf_km3": 12,
            "pressure_side": "both sides",
            "zstar_optimization": True,
            "puls_method": "ultimate",
            "puls_boundary": "Int",
            "puls_stiffener_end": "Continuous",
            "puls_sp_or_up": "SP",
            "puls_up_boundary": "SSSS",
            "panel_or_shell": "panel",
            "girder_lg": 10000,
        },
        girder_values={
            "web_h": 700,
            "web_t": 15,
            "fl_w": 250,
            "fl_t": 25,
            "type": "L",
        },
        buckling_values={
            "min_pressure_adjacent_spans": 0,
            "load_factor_stresses": 1,
            "stiffener_end_support": "Continuous",
            "girder_end_support": "Sniped",
            "tension_field": "not allowed",
            "plate_effective_against_sigy": True,
            "buckling_length_factor_stf": 0,
            "buckling_length_factor_girder": 0,
            "km3": 12,
            "km2": 24,
            "girder_dist_lateral_support": 0,
            "stiffener_dist_lateral_support": 0,
            "panel_length": 0,
            "fabrication_method_stiffener": "welded",
            "fabrication_method_girder": "welded",
        },
        structure_types={"vertical": ["GENERAL_INTERNAL_WT"]},
    )


def test_flat_structure_property_service_builds_legacy_property_bundle_from_plain_values():
    properties, stiffener_properties = project_services.FlatStructurePropertyService.build(
        _flat_structure_property_request()
    )

    assert properties["main dict"]["calculation domain"] == ["Flat plate, stiffened with girder", ""]
    assert properties["main dict"]["material yield"] == [355e6, "Pa"]
    assert properties["Plate"]["span"] == [4.0, "m"]
    assert properties["Plate"]["plate_thk"] == [0.02, "m"]
    assert properties["Stiffener"]["spacing"] == [0.7, "m"]
    assert properties["Girder"]["stf_web_height"] == [0.7, "m"]
    assert properties["Girder"]["stf_type"] == ["L", ""]
    assert stiffener_properties is properties["Stiffener"]


def test_flat_structure_property_service_omits_stiffener_and_girder_for_unstiffened_domain():
    properties, stiffener_properties = project_services.FlatStructurePropertyService.build(
        _flat_structure_property_request("Flat plate, unstiffened")
    )

    assert properties["Stiffener"] is None
    assert properties["Girder"] is None
    assert stiffener_properties["stf_web_height"] == [0.4, "m"]


def _cylinder_structure_property_request(load_mode=0):
    return project_services.CylinderStructurePropertyRequest(
        calculation_domain="Longitudinal Stiffened shell (Force input)",
        dummy_values={
            "span": 4000,
            "plate_thk": 20,
            "structure_type": "GENERAL_INTERNAL_WT",
            "sigma_y1": 90,
            "sigma_y2": 90,
            "sigma_x1": 40,
            "sigma_x2": 40,
            "tau_xy": 5,
            "plate_kpp": 1,
            "stf_kps": 1,
            "stf_km1": 12,
            "stf_km2": 24,
            "stf_km3": 12,
            "pressure_side": "both sides",
            "zstar_optimization": True,
            "puls_method": "ultimate",
            "puls_boundary": "Int",
            "puls_stiffener_end": "Continuous",
            "puls_sp_or_up": "SP",
            "puls_up_boundary": "SSSS",
            "panel_or_shell": "shell",
            "material_factor": 1.15,
            "spacing": 700,
        },
        shell_values={
            "thickness": 20,
            "radius": 5000,
            "distance_between_rings": 5000,
            "length": 5000,
            "total_length": 5000,
            "k_factor": 1.0,
        },
        longitudinal_values={
            "spacing": 700,
            "web_h": 450,
            "web_t": 12,
            "fl_w": 150,
            "fl_t": 20,
            "type": "T",
        },
        ring_stiffener_values={
            "web_h": 0,
            "web_t": 0,
            "fl_w": 0,
            "fl_t": 0,
            "type": "T",
        },
        ring_frame_values={
            "web_h": 0,
            "web_t": 0,
            "fl_w": 0,
            "fl_t": 0,
            "type": "T",
        },
        load_input={
            "mode": load_mode,
            "Nsd": 1000,
            "Msd": 2000,
            "Tsd": 3000,
            "Qsd": 4000,
            "sasd": 40,
            "smsd": 195,
            "tTsd": -12.7,
            "tQsd": 4.8,
            "psd": 0.2,
            "shsd": 0,
        },
        main_values={
            "material_factor": 1.15,
            "fab_method_ring_stiffener": "welded",
            "fab_method_ring_frame": "welded",
            "e_module": 210000000000,
            "poisson": 0.3,
            "yield": 355,
            "length_between_girders": 5000,
            "panel_spacing": 700,
            "ring_stiffener_excluded": True,
            "ring_frame_excluded": True,
            "uls_or_als": "ULS",
            "end_cap_pressure": False,
        },
        structure_types={"vertical": ["GENERAL_INTERNAL_WT"]},
    )


def test_cylinder_structure_property_service_builds_legacy_property_bundle_from_stress_values():
    result = project_services.CylinderStructurePropertyService.build(
        _cylinder_structure_property_request(load_mode=0)
    )

    assert result.geometry == 3
    assert result.main_dict["geometry"] == [3, ""]
    assert result.main_dict["sasd"] == [40e6, "Pa"]
    assert result.main_dict["smsd"] == [195e6, "Pa"]
    assert result.main_dict["tTsd"] == [12.7e6, "Pa"]
    assert result.main_dict["psd"] == [0.2e6, "Pa"]
    assert result.shell_dict["plate_thk"] == [0.02, "m"]
    assert result.shell_dict["radius"] == [5.0, "m"]
    assert result.longitudinal_dict["spacing"] == [0.7, "m"]
    assert result.longitudinal_dict["stf_web_height"] == [0.45, "m"]
    assert result.ring_frame_dict["span"] == [4.0, "m"]
    assert result.derived_stresses == (40, 195, 12.7, 4.8, 0)
    assert len(result.derived_forces) == 4


def test_cylinder_structure_property_service_derives_stresses_from_force_values():
    result = project_services.CylinderStructurePropertyService.build(
        _cylinder_structure_property_request(load_mode=1)
    )

    assert len(result.derived_stresses) == 5
    assert result.derived_forces == (1000, 2000, 3000, 4000)
    assert result.main_dict["sasd"][0] == result.derived_stresses[0] * 1e6
    assert result.main_dict["shsd"][0] == result.derived_stresses[4] * 1e6


def _cylinder_excel_import_defaults(end_cap_pressure="not included in axial force"):
    return project_services.CylinderExcelImportDefaults(
        plate_thk=20,
        structure_type="GENERAL_INTERNAL_WT",
        sigma_y1=90,
        sigma_y2=90,
        sigma_x1=40,
        sigma_x2=40,
        tau_xy=5,
        plate_kpp=1,
        stf_kps=1,
        stf_km1=12,
        stf_km2=24,
        stf_km3=12,
        pressure_side="both sides",
        zstar_optimization=True,
        puls_method="ultimate",
        puls_boundary="Int",
        puls_stiffener_end="Continuous",
        puls_sp_or_up="SP",
        puls_up_boundary="SSSS",
        panel_or_shell="shell",
        material_factor=1.15,
        design_pressure=-0.2,
        shear_stress=0,
        e_module=210000000000,
        poisson=0.3,
        length_between_girders=2500,
        fab_method_ring_stiffener=1,
        fab_method_ring_frame=2,
        end_cap_pressure=end_cap_pressure,
        structure_types={"vertical": ["GENERAL_INTERNAL_WT"]},
    )


def _cylinder_excel_import_record(
    *,
    stress_values=(40, 195, -12.7, 4.8, 0.2, 0),
    force_values=(None, None, None, None),
    ring_stiffener_values=(300, 12, 120, 20, "T"),
    ring_frame_values=(400, 14, 160, 22, 5000, "L"),
):
    return project_services.CylinderExcelImportRecord(
        calculation_domain="Longitudinal Stiffened shell (Force input)",
        first_point=(0, 0),
        second_point=(0, 5000),
        shell_values=(20, 5000, 5000, 5000, 5000, 1.0, 1.25),
        longitudinal_values=(450, 12, 150, 20, 700, "T"),
        ring_stiffener_values=ring_stiffener_values,
        ring_frame_values=ring_frame_values,
        stress_values=stress_values,
        force_values=force_values,
        end_values=("ULS", 460, "Fabricated", "Cold formed"),
    )


def test_cylinder_excel_import_property_service_maps_stress_record_to_cylinder_request():
    request = project_services.CylinderExcelImportPropertyService.build_request(
        _cylinder_excel_import_record(),
        _cylinder_excel_import_defaults(),
    )
    result = project_services.CylinderStructurePropertyService.build(request)

    assert request.load_input["mode"] == 2
    assert request.dummy_values["span"] == 5000
    assert request.shell_values["thickness"] == 20
    assert request.longitudinal_values["spacing"] == 700
    assert request.main_values["material_factor"] == 1.25
    assert request.main_values["yield"] == 460
    assert request.main_values["ring_stiffener_excluded"] is False
    assert request.main_values["ring_frame_excluded"] is False
    assert result.main_dict["sasd"] == [40e6, "Pa"]
    assert result.main_dict["mat_yield"] == [460e6, "Pa"]
    assert result.shell_dict["radius"] == [5.0, "m"]


def test_cylinder_excel_import_property_service_prefers_force_values_when_present():
    request = project_services.CylinderExcelImportPropertyService.build_request(
        _cylinder_excel_import_record(force_values=(1000, 2000, 3000, 4000)),
        _cylinder_excel_import_defaults(),
    )
    result = project_services.CylinderStructurePropertyService.build(request)

    assert request.load_input["mode"] == 1
    assert result.derived_forces == (1000, 2000, 3000, 4000)
    assert result.main_dict["psd"] == [0.2e6, "Pa"]


def test_cylinder_excel_import_property_service_excludes_missing_ring_components():
    request = project_services.CylinderExcelImportPropertyService.build_request(
        _cylinder_excel_import_record(
            ring_stiffener_values=(None, None, None, None, None),
            ring_frame_values=(None, None, None, None, None, None),
        ),
        _cylinder_excel_import_defaults(),
    )
    result = project_services.CylinderStructurePropertyService.build(request)

    assert request.main_values["ring_stiffener_excluded"] is True
    assert request.main_values["ring_frame_excluded"] is True
    assert request.ring_stiffener_values["web_h"] == 0
    assert request.ring_frame_values["web_h"] == 0
    assert result.ring_stiffener_dict["stf_web_height"] == [0.0, "m"]
    assert result.ring_frame_dict["stf_web_height"] == [0.0, "m"]


def test_cylinder_excel_import_property_service_uses_default_end_cap_pressure_not_shell_yield():
    request = project_services.CylinderExcelImportPropertyService.build_request(
        _cylinder_excel_import_record(),
        _cylinder_excel_import_defaults(end_cap_pressure="included in axial force"),
    )
    result = project_services.CylinderStructurePropertyService.build(request)

    assert request.main_values["yield"] == 460
    assert request.main_values["end_cap_pressure"] == "included in axial force"
    assert result.main_dict["end cap pressure"] == ["included in axial force", ""]
    assert result.main_dict["end cap pressure"][0] != request.main_values["yield"]


def test_line_load_service_rebuilds_line_loads_and_reports_changed_lines():
    first_structure = DummyStructure()
    second_structure = DummyStructure()
    line_bundles = {
        "line1": [first_structure, None, None, ["old"], {}, None],
        "line2": [second_structure, None, None, ["old"], {}, None],
    }
    changed_load = DummyLoad("changed", "new pressure")
    new_load = DummyLoad("new", "fresh pressure")
    previous_loads = {"changed": [DummyLoad("changed", "old pressure"), ["line1"]]}
    current_loads = {
        "changed": [changed_load, ["line1"]],
        "new": [new_load, ["line2"]],
    }

    result = project_services.LineLoadService(line_bundles).rebuild_line_loads(
        ["line1", "line2", "line3"], current_loads, previous_loads)

    assert result.invalidated_lines == ("line1", "line2")
    assert result.changed_lines == ("line1", "line2")
    assert line_bundles["line1"][3] == [changed_load]
    assert line_bundles["line2"][3] == [new_load]
    assert first_structure.need_recalc
    assert second_structure.need_recalc


def test_line_pressure_service_calculates_combinations_from_plain_load_factors():
    base_bundle = example_data.get_line_to_struc()["line1"]
    load = calc_loads.Loads(example_data.loa_uls)
    line_input = project_services.LinePressureInput(
        line_name="line1",
        line_bundle=[base_bundle[0], None, None, [load], {}, None],
        coordinate=(4.0, 0.0),
        defined_tanks=(),
        accelerations={"static": 9.81, "dyn_loaded": 3.0, "dyn_ballast": 3.0},
        load_factors=project_services.load_factor_records(
            {
                ("dnva", "line1", load.get_name()): (1.2, 0.7, 1),
                ("dnvb", "line1", load.get_name()): (1.0, 1.3, 1),
                ("tanktest", "line1", load.get_name()): (1.0, 0.0, 1),
                ("manual", "line1", "manual"): (11.0, 2.0, 1),
            }
        ),
    )

    results = project_services.LinePressureService.calculate_combinations(line_input)
    highest = project_services.LinePressureService.highest_pressure(line_input)

    assert results["dnva"][0] > 0
    assert results["dnvb"][0] > 0
    assert results["manual"] == [22.0]
    assert highest["normal"] == max(max(values) for key, values in results.items() if key != "slamming")
    assert highest["slamming"] is None


def test_sesam_export_service_builds_supported_lines_from_plain_request():
    captured = {}

    def export_factory(points, lines, sections, line_to_struc):
        captured["request"] = (points, lines, sections, line_to_struc)
        return DummySesamExport(points, lines, sections, line_to_struc)

    request = project_services.SesamExportRequest(
        points={"point1": [0, 0]},
        lines={"line1": [1, 2]},
        sections=("T",),
        line_bundles={"line1": ["bundle"]},
    )

    lines = project_services.SesamExportService.build_js_lines(request, export_factory)

    assert captured["request"] == (request.points, request.lines, request.sections, request.line_bundles)
    assert lines == ("points\n", "lines\n", "sections\n", "beams\n")


def test_sesam_export_service_writes_supported_lines_to_path(tmp_path):
    request = project_services.SesamExportRequest(
        points={"point1": [0, 0]},
        lines={"line1": [1, 2]},
        sections=("T",),
        line_bundles={"line1": ["bundle"]},
    )

    export_path = project_services.SesamExportService.write_js_path(
        request,
        tmp_path / "geometry.js",
        DummySesamExport,
    )

    assert export_path == tmp_path / "geometry.js"
    assert export_path.read_text(encoding="utf-8") == "points\nlines\nsections\nbeams\n"


def test_excel_project_import_service_reads_supported_sheets_and_closes_workbook():
    workbooks = []

    def workbook_factory(path, visible, read_only):
        workbook = DummyExcelWorkbook(path, visible, read_only)
        workbooks.append(workbook)
        return workbook

    import_data = project_services.ExcelProjectImportService.read_path(
        Path("import.xlsx"),
        workbook_factory=workbook_factory,
    )

    assert import_data.flat_plate_rows[0] == ["flat header"]
    assert import_data.cylinder_rows[0] == ["cylinder header"]
    assert import_data.flat_plate_records[0].first_point == ("flat 1", "flat 2")
    assert import_data.flat_plate_records[0].plate_values == tuple(f"flat {idx}" for idx in range(5, 11))
    assert import_data.flat_plate_records[0].manual_pressure == "flat 16"
    assert import_data.cylinder_records[0].shell_values == tuple(
        f"cylinder {idx}" for idx in range(5, 12)
    )
    assert import_data.cylinder_records[0].end_values == tuple(
        f"cylinder {idx}" for idx in range(39, 43)
    )
    assert workbooks[0].path == "import.xlsx"
    assert not workbooks[0].visible
    assert workbooks[0].read_only
    assert workbooks[0].closed


def test_excel_project_import_service_opens_example_workbook_visibly():
    workbook = project_services.ExcelProjectImportService.open_example_path(
        "example.xlsx",
        workbook_factory=DummyExcelWorkbook,
    )

    assert workbook.path == "example.xlsx"
    assert workbook.visible
    assert workbook.read_only
    assert not workbook.closed


def test_excel_project_geometry_import_service_adds_typed_record_geometry_in_meters():
    first = project_services.FlatPlateExcelImportRecord(
        calculation_domain="Flat plate, stiffened",
        first_point=(0, 0),
        second_point=(3000, 0),
        plate_values=(),
        stress_values=(),
        manual_pressure=0,
        girder_values=(),
        buckling_values=(),
    )
    second = project_services.CylinderExcelImportRecord(
        calculation_domain="Longitudinal Stiffened shell  (Stress input)",
        first_point=(3000, 0),
        second_point=(3000, 5000),
        shell_values=(),
        longitudinal_values=(),
        ring_stiffener_values=(),
        ring_frame_values=(),
        stress_values=(),
        force_values=(),
        end_values=(),
    )
    points, lines = {}, {}

    result = project_services.ExcelProjectGeometryImportService.add_records(
        points,
        lines,
        (first, second),
    )

    assert points == {
        "point1": [0.0, 0.0],
        "point2": [3.0, 0.0],
        "point3": [3.0, 5.0],
    }
    assert lines == {"line1": [1, 2], "line2": [2, 3]}
    assert tuple(point.name for point in result.created_points) == ("point1", "point2", "point3")
    assert tuple(imported.line.name for imported in result.imported_lines) == ("line1", "line2")
    assert result.imported_lines[0].record is first
    assert result.imported_lines[1].record is second


def test_report_request_service_runs_pdf_and_table_adapters():
    request = project_services.ReportRequest("results.pdf", "Section results", 10, {"plain": "data"})

    pdf_report = project_services.ReportRequestService.create_pdf(request, DummyReport)
    table_document = project_services.ReportRequestService.create_table(
        request,
        DummyReport,
        DummyReportDocument,
    )

    assert pdf_report.arguments == ("results.pdf", "Section results", 10, {"plain": "data"})
    assert pdf_report.document_created
    assert pdf_report.pdf_saved
    assert table_document.filename == "results.pdf"
    assert table_document.elements == ["table"]


def test_report_data_snapshot_keeps_plain_report_input_data():
    snapshot = project_services.ReportDataSnapshot(
        project_information="Project text",
        buckling_method="ML-CL (PULS based)",
        points={"point1": [0, 0]},
        lines={"line1": [1, 2]},
        line_bundles={"line1": ["bundle"]},
        tanks={"comp1": "tank"},
        loads={"load1": "load"},
        result_state={"colors": {"line1": "green"}},
        highest_pressures={"line1": {"normal": 3}},
        ml_classes=("ok", "not ok"),
    )

    assert snapshot.project_information == "Project text"
    assert snapshot.buckling_method == "ML-CL (PULS based)"
    assert snapshot.points == {"point1": [0, 0]}
    assert snapshot.lines == {"line1": [1, 2]}
    assert snapshot.line_bundles == {"line1": ["bundle"]}
    assert snapshot.tanks == {"comp1": "tank"}
    assert snapshot.loads == {"load1": "load"}
    assert snapshot.result_state == {"colors": {"line1": "green"}}
    assert snapshot.highest_pressures == {"line1": {"normal": 3}}
    assert snapshot.ml_classes == ("ok", "not ok")


def test_project_services_keeps_project_application_compatibility_exports():
    assert project_services.ProjectFileCodec is project_application.ProjectFileCodec
    assert project_services.ProjectPersistenceService is project_application.ProjectPersistenceService
    assert project_services.ProjectFileDialogService is project_application.ProjectFileDialogService
    assert project_services.ProjectOpenService is project_application.ProjectOpenService
    assert project_services.ProjectSaveService is project_application.ProjectSaveService
    assert project_services.ProjectSnapshotService is project_application.ProjectSnapshotService
    assert project_services.ProjectHydrationService is project_application.ProjectHydrationService
    assert project_services.ProjectOpenTransferService is project_application.ProjectOpenTransferService


def test_project_snapshot_service_serializes_domain_objects_without_tkinter():
    line_bundles = {
        "line1": [DummyStructure(), None, DummyFatigue(), [], {}, DummyCylinder()],
        "line2": [DummyStructure("Flat plate, unstiffened"), None, None, [], {}, None],
    }
    load_assignments = {"load1": [DummyLoad("loaded", "pressure"), ["line1"]]}
    load_combinations = [
        project_application.LoadCombinationRecord(("loaded", "ULS"), 1.2, 0.7, 1),
        project_application.LoadCombinationRecord(("manual", "FLS"), 0.9, 0.5, 0),
    ]

    state = project_application.ProjectSnapshotService.create_state(
        project_information="Snapshot service",
        theme="dark",
        points={"point1": [0.0, 0.0]},
        lines={"line1": [1, 2]},
        line_bundles=line_bundles,
        load_assignments=load_assignments,
        accelerations={"static": 9.81},
        load_combinations=load_combinations,
        tanks={"comp2": DummyTank()},
        tank_grid=[[0, 1]],
        tank_search_data={"2": {"min_el": 0}},
        buckling_method="ML-CL (PULS based)",
        shifting={"shifted checked": True, "shift hor": 1, "shift ver": 2},
        weight_and_cog={"new structure": {"weight": 3}},
    )

    assert state.structures == {
        "line1": {"domain": "Flat plate, stiffened"},
        "line2": {"domain": "Flat plate, unstiffened"},
    }
    assert state.shell_structures == {"line1": {"shell": "properties"}, "line2": None}
    assert state.fatigue == {"line1": {"fatigue": "properties"}, "line2": None}
    assert state.loads == {"load1": [["loaded", "pressure"], ["line1"]]}
    assert state.tanks == {
        "grid": [[0, 1]],
        "search_data": {"2": {"min_el": 0}},
        "comp2": {"tank": "properties"},
    }
    assert state.load_combinations == {
        0: [("loaded", "ULS"), 1.2, 0.7, 1],
        1: [("manual", "FLS"), 0.9, 0.5, 0],
    }


def test_project_save_service_creates_state_from_plain_save_input():
    save_input = project_application.ProjectSaveInput(
        project_information="Save service",
        theme="dark",
        points={"point1": [0.0, 0.0]},
        lines={"line1": [1, 2]},
        line_bundles={"line1": [DummyStructure(), None, None, [], {}, None]},
        load_assignments={"load1": [DummyLoad("loaded", "pressure"), ["line1"]]},
        accelerations={"static": 9.81},
        load_combinations=(
            project_application.LoadCombinationRecord(("loaded", "ULS"), 1.2, 0.7, 1),
        ),
        tanks={"comp2": DummyTank()},
        tank_grid=[[0, 1]],
        tank_search_data={"2": {"min_el": 0}},
        buckling_method="DNV-RP-C201 - prescriptive",
        shifting={"shifted checked": True},
        weight_and_cog={"new structure": {"weight": [1]}},
    )

    state = project_application.ProjectSaveService.create_state(save_input)

    assert state.project_information == "Save service"
    assert state.structures == {"line1": {"domain": "Flat plate, stiffened"}}
    assert state.loads == {"load1": [["loaded", "pressure"], ["line1"]]}
    assert state.load_combinations == {0: [("loaded", "ULS"), 1.2, 0.7, 1]}
    assert state.tanks["comp2"] == {"tank": "properties"}


def test_project_hydration_service_rebuilds_saved_structures_loads_and_cylinders():
    base_bundle = example_data.get_line_to_struc()["line1"]
    load = calc_loads.Loads(example_data.loa_uls)
    cylinder = calc_structure.CylinderAndCurvedPlate(
        main_dict=example_data.shell_main_dict,
        shell=calc_structure.Shell(example_data.shell_dict),
        long_stf=calc_structure.Structure(example_data.obj_dict_cyl_long2),
        ring_stf=None,
        ring_frame=None,
    )
    state = project_application.ProjectSnapshotService.create_state(
        project_information="hydrate",
        theme="default",
        points={"point1": [0.0, 0.0], "point2": [1.0, 0.0]},
        lines={"line1": [1, 2]},
        line_bundles={
            "line1": [
                base_bundle[0],
                None,
                example_data.get_fatigue_object(),
                [],
                {},
                cylinder,
            ]
        },
        load_assignments={"load1": [load, ["line1"]]},
        accelerations={"static": 9.81},
        load_combinations=[],
        tanks={},
        tank_grid=[],
        tank_search_data=None,
        buckling_method="DNV-RP-C201 - prescriptive",
        shifting={},
        weight_and_cog={},
    )

    hydrated = project_application.ProjectHydrationService.hydrate_objects(
        state,
        project_application.ProjectHydrationDefaults(structure_types=example_data.structure_types),
    )
    line_bundle = hydrated.line_bundles["line1"]

    assert isinstance(line_structure.structure(line_bundle), calc_structure.AllStructure)
    assert isinstance(line_structure.fatigue(line_bundle), calc_structure.CalcFatigue)
    assert isinstance(line_structure.cylinder(line_bundle), calc_structure.CylinderAndCurvedPlate)
    assert hydrated.load_assignments["load1"][0].get_name() == load.get_name()
    assert line_structure.loads(line_bundle)[0].get_name() == load.get_name()
    assert len(hydrated.section_properties) == 1


def test_project_open_service_assembles_transfer_and_hydrated_objects():
    base_bundle = example_data.get_line_to_struc()["line1"]
    state = project_application.ProjectSnapshotService.create_state(
        project_information="opened",
        theme="dark",
        points={"point1": [0.0, 0.0], "point2": [1.0, 0.0]},
        lines={"line1": [1, 2]},
        line_bundles={"line1": [base_bundle[0], None, None, [], {}, None]},
        load_assignments={},
        accelerations={"static": 10.0},
        load_combinations=[],
        tanks={},
        tank_grid=[],
        tank_search_data=None,
        buckling_method="DNV-RP-C201 - prescriptive",
        shifting={},
        weight_and_cog={},
    )

    opened = project_application.ProjectOpenService.assemble(
        state,
        project_application.ProjectHydrationDefaults(structure_types=example_data.structure_types),
    )

    assert opened.state is state
    assert opened.transfer.project_information == "opened"
    assert opened.transfer.accelerations["static"] == 10.0
    assert isinstance(
        line_structure.structure(opened.hydration.line_bundles["line1"]),
        calc_structure.AllStructure,
    )


def test_project_hydration_service_migrates_old_flat_structure_defaults():
    legacy_structure = copy.deepcopy(example_data.obj_dict)
    legacy_structure.pop("structure_types")
    legacy_structure["sigma_x"] = [80, "MPa"]
    legacy_structure.pop("sigma_x1")
    legacy_structure.pop("sigma_x2")
    state = ProjectState(
        points={"point1": [0.0, 0.0], "point2": [1.0, 0.0]},
        lines={"line1": [1, 2]},
        structures={"line1": legacy_structure},
        fatigue={"line1": None},
    )

    hydrated = project_application.ProjectHydrationService.hydrate_objects(
        state,
        project_application.ProjectHydrationDefaults(structure_types=example_data.structure_types),
    )
    line_structure_object = line_structure.structure(hydrated.line_bundles["line1"])

    assert isinstance(line_structure_object, calc_structure.AllStructure)
    assert line_structure_object.Stiffener is not None
    assert line_structure_object.Plate.sigma_x1 == 80
    assert hydrated.section_properties[0]["structure_types"][0] == example_data.structure_types


def test_project_open_transfer_service_normalizes_saved_view_payloads():
    state = ProjectState(
        project_information="Open transfer",
        theme="dark",
        points={"point1": [0.0, 0.0]},
        lines={"line1": [1, 2]},
        accelerations={"static": 10.0},
        load_combinations={
            0: [["dnva", "line1", "load1"], 1.2, 0.7, 1],
            1: [["legacy", "line1", "load2"], 0.8, 0],
        },
        tanks={
            "grid": [[0, 1]],
            "search_data": {"2": {"min_el": 0}},
            "comp2": {"tank": "properties"},
        },
        buckling_method="ML-CL (PULS based)",
        weight_and_cog={"new structure": {"weight": [1]}},
    )

    transfer = project_application.ProjectOpenTransferService.create_transfer(state)

    assert transfer.accelerations == {"static": 10.0, "dyn_loaded": 0, "dyn_ballast": 0}
    assert transfer.load_combinations == (
        project_application.OpenLoadCombinationRecord(("dnva", "line1", "load1"), 1.2, 0.7, 1),
        project_application.OpenLoadCombinationRecord(("legacy", "line1", "load2"), 0.8, 0),
    )
    assert transfer.load_combinations[0].has_include
    assert not transfer.load_combinations[1].has_include
    assert transfer.tank_grid == [[0, 1]]
    assert transfer.tank_search_data == {2: {"min_el": 0}}
    assert transfer.tank_properties == {"comp2": {"tank": "properties"}}
    assert transfer.buckling_method == "ML-CL (PULS based)"
    assert transfer.weight_and_cog == {"new structure": {"weight": [1]}}


def test_project_open_transfer_service_ignores_invalid_tank_search_payload():
    transfer = project_application.ProjectOpenTransferService.create_transfer(
        ProjectState(tanks={"search_data": ["not", "a mapping"]})
    )

    assert transfer.tank_search_data is None
