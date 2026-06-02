from pathlib import Path
import re
from types import SimpleNamespace


def test_main_application_uses_shared_geometry_menu_helpers():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")

    assert "api_helpers.CYLINDER_STRUCTURE_DOMAINS_WITH_INPUT" in source
    assert "api_helpers.FLAT_GEOMETRY_IDS" in source
    assert "api_helpers.CYLINDER_GEOMETRY_IDS" in source
    assert "CylinderAndCurvedPlate.geomeries.values()" not in source
    assert "CylinderAndCurvedPlate.geomeries_map" not in source
    assert "Longitudinal Stiffened shell  (Force input)" not in source


def test_functional_modes_keep_3d_section_checkbox_visible():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")

    assert "def _place_3d_section_view_checkbox(self):" in source
    assert "self._chk_show_prop_3d.place(relx=0.637, rely=0.705)" in source
    assert "self._chk_show_prop_3d.lift()" in source
    assert source.count("self._place_3d_section_view_checkbox()") >= 3


def test_help_tab_includes_cylinder_panel_buckling_image():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    shell_image_index = source.index("Buckling_Strength_of_Shells.png")
    panel_heading_index = source.index("Buckling cylinder panels")
    panel_image_index = source.index("buckling_cylinder_panel.png")

    assert shell_image_index < panel_heading_index < panel_image_index


def test_main_application_uses_tcl9_compatible_variable_traces():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")

    assert ".trace('w', self.trace_acceptance_change)" not in source
    assert ".trace_add('write', self.trace_acceptance_change)" in source


def test_resize_state_is_initialized_before_configure_binding():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    init_source = source[
        source.index("def __init__(self, parent):"):
        source.index("self._root_dir =")
    ]

    assert init_source.index("self._last_resize_size = (0, 0)") < init_source.index('parent.bind("<Configure>"')
    assert init_source.index("self._resize_after_id = None") < init_source.index('parent.bind("<Configure>"')


def test_main_gui_prompts_for_simplified_single_line_mode_with_standard_default():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")

    assert "self._simplified_calculation_mode = False" in source
    assert "self._single_line_name = 'line1'" in source
    assert "def _prompt_startup_calculation_mode(self):" in source
    assert "from importlib import metadata as importlib_metadata" in source
    assert "def _get_application_version_from_metadata():" in source
    assert "importlib_metadata.version(package_name)" in source
    assert "__version__" not in source[source.index("def _show_startup_calculation_mode_dialog"):source.index("def switch_to_single_calculation_mode")]
    assert "def _show_startup_calculation_mode_dialog(self):" in source
    assert "tk.Toplevel(self._parent, background='#f5f7fb')" in source
    assert "ANYstructure_logo.jpg" in source
    assert "from PIL import Image, ImageTk" in source
    assert "ImageTk.PhotoImage(logo_image)" in source
    assert "logo_image.thumbnail((132, 76), Image.LANCZOS)" in source
    assert "Version ' + app_version" in source
    assert "Choose calculation workflow" in source
    assert "Multiple panels" in source
    assert "subtitle='Default'" in source
    assert "Recommended default" not in source
    assert "Single panel/cylinder" in source
    assert "dialog.bind('<Return>', lambda _event: choose(False))" in source
    assert "dialog.bind('<Escape>', lambda _event: choose(False))" in source
    assert "self._parent.wait_window(dialog)" in source
    assert "Mode - Single panel/cylinder" in source
    assert "Mode - Multiple panels" in source
    assert "def switch_to_single_calculation_mode(self):" in source
    assert "def switch_to_multiple_calculation_mode(self):" in source
    assert "self._single_line_name = selected_line" in source
    assert "self._activate_simplified_calculation_pipeline()" in source
    assert "def _ensure_single_dummy_line(self):" in source
    assert "self._line_dict[self._single_line_name] = [1, 2]" in source
    assert "def _ensure_manual_pressure_combination(self, line, default_enabled=False):" in source
    assert "def _gui_single_line_manual_pressure(self):" in source
    assert "Manual pressure [Pa]" in source
    assert "if self._line_to_struc[self._active_line][5] is not None:" in source
    assert "self._result_label_manual, self._lab_pressure" in source
    assert "def _sync_simplified_domain_selection(self):" in source
    assert "self._sync_simplified_domain_selection()" in source
    assert "if not getattr(self, '_simplified_calculation_mode', False):\n            self.set_selected_variables(self._active_line)" in source
    assert "self._tabControl.hide(self._tab_geo)" in source
    assert "self._tabControl.hide(self._tab_comp)" in source
    assert "def _show_standard_calculation_layout(self):" in source
    assert "self._tabControl.add(self._tab_geo, text='Geometry')" in source
    assert "self._tabControl.add(self._tab_comp, text='Compartments and loads')" in source
    assert "self.gui_load_combinations(self._combination_slider.get())" in source


def test_single_line_optimizer_return_refreshes_hidden_line():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")

    assert "def _prepare_simplified_optimizer_replacement(self):" in source
    assert "def _refresh_simplified_optimizer_replacement(self):" in source
    assert "def _replace_active_line_with_optimized_structure(self, optimized_structure):" in source
    assert "self._ensure_manual_pressure_combination(self._active_line, default_enabled=True)" in source
    assert "self.set_selected_variables(self._active_line)" in source

    flat_close = source[
        source.index("def on_close_opt_window"):
        source.index("def on_close_opt_cyl_window")
    ]
    cylinder_close = source[
        source.index("def on_close_opt_cyl_window"):
        source.index("def on_close_opt_multiple_window")
    ]

    assert "self._prepare_simplified_optimizer_replacement()" in flat_close
    assert "self._replace_active_line_with_optimized_structure(returned_object[0])" in flat_close
    assert "else:\n            self.new_structure(multi_return=returned_object[0:2])" in flat_close
    assert "if not self._refresh_simplified_optimizer_replacement():" in flat_close
    assert "self._prepare_simplified_optimizer_replacement()" in cylinder_close
    assert "self.new_structure(cylinder_return=returned_object[0])" in cylinder_close
    assert "if not self._refresh_simplified_optimizer_replacement():" in cylinder_close


def test_single_line_mode_keeps_active_line_on_selected_or_dummy_line():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")

    assert "def _single_mode_active_line_candidate(self):" in source
    selector_block = source[
        source.index("def _single_mode_active_line_candidate"):
        source.index("def _ensure_single_dummy_line")
    ]
    select_block = source[
        source.index("def _select_single_calculation_line"):
        source.index("def _ensure_manual_pressure_combination")
    ]

    assert "if self._active_line in self._line_dict:" in selector_block
    assert "return self._active_line" in selector_block
    assert "if self._single_line_name in self._line_dict:" in selector_block
    assert "return self._single_line_name" in selector_block
    assert "return sorted(self._line_dict.keys(), key=get_num)[0]" in selector_block
    assert "self._single_line_name = self._single_mode_active_line_candidate()" in select_block
    assert "self._active_line = self._single_line_name" in select_block


def test_cylinder_optimizer_return_bypasses_missing_flat_input_guard():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    new_structure = source[
        source.index("def new_structure"):
        source.index("def option_meny_structure_type_trace")
    ]

    assert "cylinder_return == None" in new_structure
    assert "self._show_missing_structure_input_warning()" in new_structure


def test_simplified_3d_preview_uses_main_canvas_place():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    placement_block = source[
        source.index("def _get_prop_3d_bottom_place"):
        source.index("def _resize_prop_3d_figure")
    ]

    assert "getattr(self, '_simplified_calculation_mode', False)" in placement_block
    assert "self._place_info_float(self._main_canvas, 'relx', 0.26)" in placement_block
    assert "self._place_info_float(self._main_canvas, 'relheight', 0.73)" in placement_block


def test_cylinder_panel_domains_render_as_angular_sector_preview():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")

    assert "def _is_cylinder_panel_preview(cyl_obj):" in source
    assert "api_helpers.domain_for_geometry_id(cyl_obj.geometry)" in source
    assert "return 'panel' in domain.lower() and 'shell' not in domain.lower()" in source
    assert "def _cylinder_preview_theta_range(self, cyl_obj):" in source
    assert "math.radians(60.0)" in source
    assert "theta_range=theta_range" in source
    assert "3D cylinder panel preview (60 deg)" in source
    assert "arc_length = abs(theta_end - theta_start) * radius" in source


def test_main_application_uses_geometry_helpers_for_active_lookups():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")

    assert "api_helpers.geometry_id_for_domain(self._new_calculation_domain.get())" in source
    assert "api_helpers.domain_for_geometry_id(main_dict_cyl['geometry'][0])" in source
    assert "api_helpers.domain_for_geometry_id(self._line_to_struc[self._active_line][5].geometry)" in source
    assert "self._shell_geometries_map[self._new_calculation_domain.get()]" not in source
    assert "CylinderAndCurvedPlate.geomeries[main_dict_cyl['geometry'][0]]" not in source
    assert "CylinderAndCurvedPlate\n                                                 .geomeries[" not in source


def test_main_application_uses_helpers_for_structure_property_unit_conversions():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    services_source = (Path(__file__).resolve().parents[1] / "anystruct" / "project_services.py").read_text(
        encoding="utf-8"
    )
    source = main_source.read_text(encoding="utf-8")
    flat_builder = source[
        source.index("def _build_flat_structure_properties"):
        source.index("def _build_cylinder_structure_property_request")
    ]
    cylinder_builder = source[
        source.index("def _build_cylinder_structure_properties"):
        source.index("def new_structure")
    ]
    property_block = flat_builder + cylinder_builder

    assert "FlatStructurePropertyService.build(" in flat_builder
    assert "CylinderStructurePropertyService.build(" in cylinder_builder
    assert "api_helpers.mpa_to_pa" in services_source
    assert "api_helpers.mm_to_m" in services_source
    assert "helper_cylinder_stress_to_force_to_stress(" in services_source
    assert "api_helpers.mpa_to_pa" not in property_block
    assert "api_helpers.mm_to_m" not in property_block
    assert "helper_cylinder_stress_to_force_to_stress(" not in property_block
    assert not re.search(r"[\w.)\]]\s*\*\s*1e6", property_block)
    assert not re.search(r"[\w.)\]]\s*/\s*1000", property_block)


def test_main_application_uses_shared_ml_model_loader():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    ml_loader_block = source[
        source.index("self._ML_buckling ="):
        source.index("# Used to select parameter")
    ]

    assert "ml_models.load_buckling_models((self._root_dir,))" in ml_loader_block
    assert "pickle.load(" not in ml_loader_block


def test_new_structure_delegates_property_building():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    new_structure = source[
        source.index("def new_structure"):
        source.index("def option_meny_structure_type_trace")
    ]
    resolver = source[
        source.index("def _resolve_new_structure_properties"):
        source.index("def _add_structure_to_active_line")
    ]
    flat_builder = source[
        source.index("def _build_flat_structure_property_request"):
        source.index("def _build_cylinder_structure_property_request")
    ]
    cylinder_builder = source[
        source.index("def _build_cylinder_structure_property_request"):
        source.index("def _structure_input_is_missing")
    ]
    add_structure = source[
        source.index("def _add_structure_to_active_line"):
        source.index("def _scale_existing_flat_structure_if_needed")
    ]
    update_structure = source[
        source.index("def _update_existing_active_line_structure"):
        source.index("def _calculate_load_combinations_after_structure_update")
    ]

    assert "self._build_flat_structure_properties()" in resolver
    assert "elif isinstance(toggle_multi, tuple):" in resolver
    assert "prop_dict, obj_dict_stf = toggle_multi" in resolver
    assert "if cylinder_return is not None:" in resolver
    assert "CylinderObj = cylinder_return" in resolver
    assert "FlatStructurePropertyRequest(" in flat_builder
    assert "FlatStructurePropertyService.build(" in flat_builder
    assert "api_helpers.mpa_to_pa" not in flat_builder
    assert "api_helpers.mm_to_m" not in flat_builder
    assert "CylinderStructurePropertyRequest(" in cylinder_builder
    assert "CylinderStructurePropertyService.build(" in cylinder_builder
    assert "helper_cylinder_stress_to_force_to_stress(" not in cylinder_builder
    assert "api_helpers.mpa_to_pa" not in cylinder_builder
    assert "api_helpers.mm_to_m" not in cylinder_builder
    assert "self._build_cylinder_structure_properties()" in resolver
    assert "self._structure_input_is_missing()" in new_structure
    assert "self._create_all_structure_from_properties(prop_dict)" in add_structure
    assert "self._create_cylinder_structure_from_properties(" in add_structure
    assert "self._clear_tanks_and_grid()" in add_structure
    assert "self._clear_tanks_and_grid()" in update_structure
    assert "self._refresh_after_structure_change(suspend_recalc)" in new_structure
    assert "self._resolve_new_structure_properties(" in new_structure
    assert "self._add_structure_to_active_line(" in new_structure
    assert "self._update_existing_active_line_structure(" in new_structure
    assert "self._calculate_load_combinations_after_structure_update()" in new_structure
    assert "obj_dict = {" not in new_structure
    assert "shell_dict = {" not in new_structure
    assert "AllStructure(" not in new_structure
    assert "CylinderAndCurvedPlate(" not in new_structure
    assert "self._tank_dict = {}" not in new_structure
    assert "self.update_frame()" not in new_structure
    assert "set_main_properties(prop_dict)" not in new_structure
    assert "calculate_all_load_combinations_for_line_all_lines()" not in new_structure


def test_savefile_delegates_save_command_assembly_and_persistence():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    save_no_dialogue = source[
        source.index("def save_no_dialogue"):
        source.index("def savefile")
    ]
    savefile = source[
        source.index("def savefile"):
        source.index("def _build_project_save_input")
    ]
    save_input_builder = source[
        source.index("def _build_project_save_input"):
        source.index("def openfile")
    ]

    assert "ProjectFileDialogService.backup_save_target(" in save_no_dialogue
    assert "ProjectFileDialogService.remembered_save_target(" in save_no_dialogue
    assert "ProjectSaveService.save_path(" in savefile
    assert "ProjectFileDialogService.selected_save_target(" in savefile
    assert "self._build_project_save_input()" in savefile
    assert "ProjectSnapshotService.create_state(" not in savefile
    assert "save_state_to_path(" not in savefile
    assert "ProjectSaveInput(" in save_input_builder


def test_openfile_delegates_project_open_application_steps():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    openfile = source[
        source.index("def openfile"):
        source.index("def restore_previous")
    ]

    assert "ProjectOpenService.open_path(" in openfile
    assert "ProjectFileDialogService.selected_open_target(" in openfile
    assert "self._build_project_hydration_defaults()" in openfile
    assert "self._apply_open_project_text_and_theme(open_transfer)" in openfile
    assert "self._apply_open_project_geometry_and_objects(open_transfer, hydration)" in openfile
    assert "self._apply_open_project_accelerations(open_transfer)" in openfile
    assert "self._apply_open_project_load_combinations(open_transfer)" in openfile
    assert "self._apply_open_project_tanks(open_transfer)" in openfile
    assert "self._apply_open_project_canvas_scale()" in openfile
    assert "self._finalize_open_project(open_transfer, target.path)" in openfile
    assert "ProjectHydrationDefaults(" not in openfile
    assert "load_state_from_path(" not in source
    assert "json.load(" not in source


def test_restore_and_example_open_delegate_file_target_resolution():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    restore_block = source[
        source.index("def restore_previous"):
        source.index("def open_example")
    ]
    example_block = source[
        source.index("def open_example"):
        source.index("def open_example_excel_file")
    ]

    assert "ProjectFileDialogService.restore_target(" in restore_block
    assert "ProjectPersistenceService.backup_exists(" not in restore_block
    assert "ProjectPersistenceService.backup_path(" not in restore_block
    assert "ProjectFileDialogService.example_open_target(" in example_block
    assert "os.path.isfile(file_name)" not in example_block
    assert "self._root_dir + '/' + file_name" not in example_block


def test_example_excel_open_delegates_file_target_resolution():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    example_excel_block = source[
        source.index("def open_example_excel_file"):
        source.index("def _sync_excel_import_geometry")
    ]

    assert "ProjectFileDialogService.example_open_target(" in example_excel_block
    assert "ExcelProjectImportService.open_example_path(target.path)" in example_excel_block
    assert "os.path.isfile(file_name)" not in example_excel_block
    assert "self._root_dir + '/' + file_name" not in example_excel_block


def test_line_pressure_calculation_delegates_to_project_service():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    calculation_block = source[
        source.index("def calculate_all_load_combinations_for_line"):
        source.index("def run_optimizer_for_line")
    ]
    pressure_block = source[
        source.index("def get_highest_pressure"):
        source.index("def get_fatigue_pressures")
    ]

    assert "LinePressureService.calculate_combinations(" in calculation_block
    assert "LinePressureService.calculate_one(" in calculation_block
    assert "LinePressureInput(" in calculation_block
    assert not re.search(r"\bone_load_combination\(", calculation_block)
    assert "LinePressureService.highest_pressure(" in pressure_block


def test_report_and_sesam_callbacks_delegate_request_orchestration():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    report_block = source[
        source.index("def _build_report_data_snapshot"):
        source.index("def create_accelerations")
    ]
    export_block = source[
        source.index("def export_to_js"):
        source.index("if __name__ == '__main__':")
    ]

    assert "ReportRequestService.create_pdf(" in report_block
    assert "ReportRequestService.create_table(" in report_block
    assert "def _build_report_data_snapshot" in report_block
    assert "def _get_ml_classes" in source
    assert "ReportDataSnapshot(" in report_block
    assert "self._build_report_data_snapshot()" in report_block
    assert "ml_classes=self._get_ml_classes()" in report_block
    assert "ml_classes=self._ML_classes" not in report_block
    assert 'ReportRequest(filename, "Section results", 10, self)' not in report_block
    assert "LetterMaker" not in source
    assert "SimpleDocTemplate" not in source
    assert "reportlab" not in source
    assert "ProjectFileDialogService.selected_output_target(" in report_block
    assert "filedialog.asksaveasfilename(defaultextension=\".pdf\")" in report_block
    assert not re.search(r"filedialog\.asksaveasfile\(", report_block)
    assert "SesamExportService.write_js_path(" in export_block
    assert "sesam.JSfile(" not in export_block
    assert "save_file.writelines(" not in export_block
    assert not re.search(r"filedialog\.asksaveasfile\(", export_block)
    assert "ProjectFileDialogService.selected_output_target(" in export_block


def test_excel_callbacks_delegate_workbook_adapter_access():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")
    excel_block = source[
        source.index("def open_example_excel_file"):
        source.index("def on_open_structure_window")
    ]

    assert "ExcelProjectImportService.open_example_path(" in excel_block
    assert "ExcelProjectImportService.read_path(" in excel_block
    assert "ProjectFileDialogService.selected_open_target(" in excel_block
    assert "ExcelProjectGeometryImportService.add_records(" in excel_block
    assert "def _sync_excel_import_geometry" in excel_block
    assert "def _build_flat_structure_property_request_from_excel_record" in source
    assert "def _build_flat_structure_property_request_from_cylinder_excel_record" in source
    assert "def _build_cylinder_excel_import_defaults" in source
    assert "FlatStructurePropertyService.build(flat_request)" in excel_block
    assert "CylinderExcelImportPropertyService.build_request(" in excel_block
    assert "CylinderStructurePropertyService.build(cylinder_request)" in excel_block
    assert "cylinder_return=cylinder_obj" in excel_block
    assert "flat_plate_records" in excel_block
    assert "cylinder_records" in excel_block
    assert "row_data[" not in excel_block
    assert "self.new_point()" not in excel_block
    assert "this_line = self.new_line()" not in excel_block
    assert "ExcelInterface(" not in source
    assert "excel_inteface" not in source

    flat_import_block = excel_block[
        excel_block.index("# Flat"):
        excel_block.index("# Cylinders")
    ]
    assert "_new_plate_thk.set(" not in flat_import_block
    assert "_new_sigma_x1.set(" not in flat_import_block
    assert "_new_girder_web_h.set(" not in flat_import_block

    cylinder_import_block = excel_block[
        excel_block.index("# Cylinders"):
        excel_block.index("def button_load_info_click")
    ]
    assert "_new_shell_thk.set(" not in cylinder_import_block
    assert "_new_shell_radius.set(" not in cylinder_import_block
    assert "_new_shell_Nsd.set(" not in cylinder_import_block
    assert "_new_shell_end_cap_pressure_included.set(shell_yield)" not in cylinder_import_block


def test_csr_requirement_is_shared_by_numeric_and_semianalytical_methods():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")

    assert "use_semi_analytical_equation=False" in source
    assert "selected_buckling_method in ['ML-Numeric (PULS based)', 'SemiAnalytical S3/U3']" in source
    assert "use_semi_analytical_equation=selected_buckling_method == 'SemiAnalytical S3/U3'" in source
    assert "return_dict['ML buckling class'][current_line]['CSR'] = csr_values" in source
    assert "return_dict['ML buckling colors'][current_line]['CSR requirement'] = csr_color" in source


def test_semianalytical_csr_helper_uses_equation_predictor(monkeypatch):
    from anystruct import main_application
    from anystruct.main_application import Application

    calls = {}

    def fake_predict(calc_object, design_pressure):
        calls["plate"] = calc_object.Plate
        calls["stiffener"] = calc_object.Stiffener
        calls["design_pressure"] = design_pressure
        return [1, 0, 1, 1], "red", {"source": "equation"}

    monkeypatch.setattr(
        main_application.op.semi_analytical,
        "predict_anystructure_csr_requirement",
        fake_predict,
    )

    app = object.__new__(Application)
    plate = SimpleNamespace(mat_factor=1.15)
    stiffener = SimpleNamespace()

    csr, color = app._predict_csr_requirement(
        plate,
        stiffener,
        design_pressure=123.0,
        material_factor=1.15,
        use_semi_analytical_equation=True,
    )

    assert csr == [1, 0, 1, 1]
    assert color == "red"
    assert calls == {
        "plate": plate,
        "stiffener": stiffener,
        "design_pressure": 123.0,
    }
