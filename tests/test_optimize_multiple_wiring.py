from pathlib import Path
import re

import anystruct.optimize_multiple_window as optimize_multiple_window


def test_multiple_optimizer_uses_line_structure_helpers_for_bundle_reads():
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "optimize_multiple_window.py").read_text(
        encoding="utf-8")

    assert "import anystruct.line_structure as line_structure" in source
    assert "def _line_structure(self, line):" in source
    assert "def _line_stiffener(self, line):" in source
    assert "def _update_harmonized_fatigue_result(self, line, x):" in source
    assert "_line_to_struc[line][1]" not in source
    assert "_line_to_struc[line][0]" not in source
    assert "_line_to_struc[key][0]" not in source
    assert not re.search(r"_line_to_struc\[[^\n]+]\[[012345]]", source)


def test_multiple_optimizer_updates_harmonized_fatigue_from_stiffener(monkeypatch):
    class FakeStructure:
        Stiffener = object()

    class FakeFatigue:
        def get_fatigue_properties(self):
            return {"fatigue": "properties"}

    captured = {}

    def fake_create_new_calc_obj(init_obj, x, fat_dict):
        captured["init_obj"] = init_obj
        captured["x"] = x
        captured["fat_dict"] = fat_dict
        return None, "updated fatigue"

    monkeypatch.setattr(optimize_multiple_window.opt, "create_new_calc_obj", fake_create_new_calc_obj)
    window = optimize_multiple_window.CreateOptimizeMultipleWindow.__new__(
        optimize_multiple_window.CreateOptimizeMultipleWindow)
    legacy_slot = object()
    window._line_to_struc = {"line1": [FakeStructure(), legacy_slot, FakeFatigue(), [], {}, None]}
    window._opt_results = {"line1": [None, None, None]}

    window._update_harmonized_fatigue_result("line1", [1, 2, 3])

    assert captured["init_obj"] is FakeStructure.Stiffener
    assert captured["init_obj"] is not legacy_slot
    assert captured["x"] == [1, 2, 3]
    assert captured["fat_dict"] == {"fatigue": "properties"}
    assert window._opt_results["line1"][2] == "updated fatigue"
