from pathlib import Path


def test_span_optimizer_reads_pressure_side_from_line_structure_bundle():
    optimize_geometry_source = Path(__file__).resolve().parents[1] / "anystruct" / "optimize_geometry.py"
    source = optimize_geometry_source.read_text(encoding="utf-8")

    assert "def _line_overpressure_side(self, line):" in source
    assert "return self._line_structure(line).overpressure_side" in source
    assert "pressure_side = self._line_overpressure_side(closet_line)" in source
    assert "self._line_to_struc[closet_line][0].overpressure_side" not in source
    assert "self._line_to_struc[closet_line].overpressure_side" not in source


def test_span_optimizer_reads_predefined_sections_from_stiffener_component():
    optimize_source = Path(__file__).resolve().parents[1] / "anystruct" / "optimize.py"
    source = optimize_source.read_text(encoding="utf-8")

    assert "hlp.helper_read_section_file(predefiened_stiffener_iter, struc_obj.Stiffener)" in source
    assert "hlp.helper_read_section_file(predefiened_stiffener_iter, struc_obj)" not in source


def test_span_optimizer_uses_helpers_for_line_structure_bundle_access():
    optimize_geometry_source = Path(__file__).resolve().parents[1] / "anystruct" / "optimize_geometry.py"
    source = optimize_geometry_source.read_text(encoding="utf-8")

    assert "def _line_structure_bundle(self, line):" in source
    assert "def _line_structure(self, line):" in source
    assert "def _copy_line_structure_bundle(self, line):" in source
    assert "_line_to_struc[key][0]" not in source
    assert "_line_to_struc[line][0]" not in source
    assert "_line_to_struc[closet_line][0]" not in source


def test_span_optimizer_uses_imported_allstructure_type_in_result_drawing():
    optimize_geometry_source = Path(__file__).resolve().parents[1] / "anystruct" / "optimize_geometry.py"
    source = optimize_geometry_source.read_text(encoding="utf-8")

    assert "isinstance(stuc_info, AllStructure)" in source
    assert "calc_structure.AllStructure" not in source
