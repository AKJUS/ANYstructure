from pathlib import Path
from tempfile import TemporaryDirectory

from anystruct import example_data, report_generator, sesam_interface


class _LegacyReportStructure:
    def __init__(self, structure):
        self.Plate = structure.Plate
        self.Stiffener = structure.Stiffener

    def get_s(self):
        return self.Plate.get_s()

    def get_report_stresses(self):
        return self.Plate.get_report_stresses()

    def get_results_for_report(self):
        return "Representative section results"


def test_sesam_export_writes_geometry_sections_and_beams():
    export = sesam_interface.JSfile(
        example_data.get_point_dict(),
        example_data.get_line_dict(),
        example_data.get_section_list(),
        line_to_struc=example_data.get_line_to_struc(),
    )

    export.write_points()
    export.write_lines()
    export.write_sections()
    export.write_beams()

    output = "".join(export.output_lines)
    assert "point1 = Point(" in output
    assert "line1 = CreateLineTwoPoints(" in output
    assert "ANYbm" in output
    assert ".section =" in output


def test_summary_report_generation_contract(monkeypatch):
    line_bundle = example_data.get_line_to_struc()["line1"]
    report_structure = _LegacyReportStructure(line_bundle[0])

    with TemporaryDirectory(dir=Path.cwd()) as report_dir:
        report_path = Path(report_dir)
        with monkeypatch.context() as report_context:
            report_context.chdir(report_path)

            report_generator.create_report(
                {
                    "lines": {"line1": example_data.get_line_dict()["line1"]},
                    "calc_structure": {"line1": report_structure},
                    "calc_fatigue": {"line1": line_bundle[2]},
                    "pressures": {"line1": 0},
                }
            )

            report = report_path / "Report_current_results.pdf"
            assert report.is_file()
            assert report.stat().st_size > 0


def test_letter_report_renderer_reads_snapshot_shape_directly():
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "report_generator.py").read_text(
        encoding="utf-8"
    )

    assert "self.data._" not in source
    assert "self.data.get_color_and_calc_state(" not in source
    assert "self.data.get_highest_pressure(" not in source


def test_ifc_export_has_conical_shell_model_branch():
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "ifc_model_export.py").read_text(
        encoding="utf-8"
    )

    assert "def _conical_faces(" in source
    assert "def _conical_wall_solid_faces(" in source
    assert "def _add_conical_shell_surface(" in source
    assert "def _add_conical_wall_solid(" in source
    assert 'getattr(cyl_obj, "geometry", None) == 9' in source
    assert "Unstiffened conical shell" in source


def test_report_generator_accounts_for_conical_shell_results():
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "report_generator.py").read_text(
        encoding="utf-8"
    )

    assert "Unstiffened conical shell r1/r2" in source
    assert "Unstiffened conical shell detailed" in source
    assert "results['Unstiffened conical shell'] if cyl_obj.geometry == 9" in source
