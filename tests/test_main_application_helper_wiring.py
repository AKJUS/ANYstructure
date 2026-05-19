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
    assert "self._shell_geometries_map[self._new_calculation_domain.get()]" not in source
    assert "CylinderAndCurvedPlate.geomeries[main_dict_cyl['geometry'][0]]" not in source


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

    assert "self._build_flat_structure_properties()" in new_structure
    assert "self._build_cylinder_structure_properties()" in new_structure
    assert "self._structure_input_is_missing()" in new_structure
    assert "self._create_all_structure_from_properties(prop_dict)" in new_structure
    assert "self._create_cylinder_structure_from_properties(" in new_structure
    assert "self._clear_tanks_and_grid()" in new_structure
    assert "self._refresh_after_structure_change(suspend_recalc)" in new_structure
    assert "obj_dict = {" not in new_structure
    assert "shell_dict = {" not in new_structure
    assert "AllStructure(" not in new_structure
    assert "CylinderAndCurvedPlate(" not in new_structure
    assert "self._tank_dict = {}" not in new_structure
    assert "self.update_frame()" not in new_structure
