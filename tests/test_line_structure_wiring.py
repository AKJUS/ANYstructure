from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_source(path):
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_pr27_target_files_import_line_structure_helpers():
    for path in [
        "anystruct/optimize_geometry.py",
        "anystruct/optimize_window.py",
        "anystruct/optimize_cylinder.py",
        "anystruct/fatigue_window.py",
        "anystruct/report_generator.py",
    ]:
        assert "line_structure" in read_source(path)


def test_pr27_constructor_paths_use_line_structure_helpers():
    optimize_window = read_source("anystruct/optimize_window.py")
    optimize_window_app_branch = optimize_window[
        optimize_window.index("        else:\n            self.app = app"):
        optimize_window.index("            try:\n                self._fatigue_pressure")
    ]
    assert "line_structure.structure(active_bundle)" in optimize_window_app_branch
    assert "line_structure.fatigue(active_bundle)" in optimize_window_app_branch
    assert not re.search(r"_line_to_struc\[[^\n]+]\[[025]]", optimize_window_app_branch)

    optimize_cylinder = read_source("anystruct/optimize_cylinder.py")
    optimize_cylinder_app_branch = optimize_cylinder[
        optimize_cylinder.index("        else:\n            self.app = app"):
        optimize_cylinder.index("            try:\n                self._fatigue_pressure")
    ]
    assert "line_structure.structure(active_bundle)" in optimize_cylinder_app_branch
    assert "line_structure.cylinder(active_bundle)" in optimize_cylinder_app_branch
    assert "line_structure.fatigue(active_bundle)" in optimize_cylinder_app_branch
    assert not re.search(r"_line_to_struc\[[^\n]+]\[[0125]]", optimize_cylinder_app_branch)


def test_pr27_report_generator_uses_line_structure_for_read_only_access():
    report_source = read_source("anystruct/report_generator.py")

    assert "line_structure.has_cylinder(line_bundle)" in report_source
    assert "line_structure.structure(line_bundle)" in report_source
    assert "line_structure.cylinder(line_bundle)" in report_source
    assert not re.search(r"_line_to_struc\[[^\n]+]\[[025]]", report_source)
