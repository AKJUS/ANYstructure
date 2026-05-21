from pathlib import Path
import re


def test_main_application_uses_shared_geometry_menu_helpers():
    main_source = Path(__file__).resolve().parents[1] / "anystruct" / "main_application.py"
    source = main_source.read_text(encoding="utf-8")

    assert "api_helpers.CYLINDER_STRUCTURE_DOMAINS_WITH_INPUT" in source
    assert "api_helpers.FLAT_GEOMETRY_IDS" in source
    assert "api_helpers.CYLINDER_GEOMETRY_IDS" in source
    assert "CylinderAndCurvedPlate.geomeries.values()" not in source
    assert "CylinderAndCurvedPlate.geomeries_map" not in source
    assert "Longitudinal Stiffened shell  (Force input)" not in source


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
    source = main_source.read_text(encoding="utf-8")
    flat_builder = source[
        source.index("def _build_flat_structure_properties"):
        source.index("def _build_cylinder_structure_properties")
    ]
    cylinder_builder = source[
        source.index("def _build_cylinder_structure_properties"):
        source.index("def new_structure")
    ]
    property_block = flat_builder + cylinder_builder

    assert "api_helpers.mpa_to_pa" in property_block
    assert "api_helpers.mm_to_m" in property_block
    assert not re.search(r"[\w.)\]]\s*\*\s*1e6", property_block)
    assert not re.search(r"[\w.)\]]\s*/\s*1000", property_block)


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
    add_structure = source[
        source.index("def _add_structure_to_active_line"):
        source.index("def _scale_existing_flat_structure_if_needed")
    ]
    update_structure = source[
        source.index("def _update_existing_active_line_structure"):
        source.index("def _calculate_load_combinations_after_structure_update")
    ]

    assert "self._build_flat_structure_properties()" in resolver
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
    savefile = source[
        source.index("def savefile"):
        source.index("def _build_project_save_input")
    ]
    save_input_builder = source[
        source.index("def _build_project_save_input"):
        source.index("def openfile")
    ]

    assert "ProjectSaveService.save_path(" in savefile
    assert "self._build_project_save_input()" in savefile
    assert "ProjectSnapshotService.create_state(" not in savefile
    assert "save_state_to_path(" not in savefile
    assert "ProjectSaveInput(" in save_input_builder


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
