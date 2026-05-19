from pathlib import Path


def test_span_optimizer_reads_pressure_side_from_line_structure_bundle():
    optimize_geometry_source = Path(__file__).resolve().parents[1] / "anystruct" / "optimize_geometry.py"
    source = optimize_geometry_source.read_text(encoding="utf-8")

    assert "self._line_to_struc[closet_line][0].overpressure_side" in source
    assert "self._line_to_struc[closet_line].overpressure_side" not in source
