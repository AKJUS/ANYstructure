from copy import deepcopy
from pathlib import Path

import numpy as np

from anystruct import calc_structure as calc
from anystruct import example_data as ex
from anystruct import helper as hlp
from anystruct import optimize as opt


CSV_PATH = Path(__file__).resolve().parents[1] / "anystruct" / "bulb_anglebar_tbar_flatbar.csv"


def _flat_benchmark_object():
    obj_dict = deepcopy(ex.obj_dict)
    obj_dict.setdefault("panel or shell", ["panel", ""])
    scantlings = calc.CalcScantlings(obj_dict)
    return calc.AllStructure(
        Plate=scantlings,
        Stiffener=scantlings,
        Girder=None,
        main_dict=ex.prescriptive_main_dict,
    )


def test_anysmart_predefined_stiffeners_work_with_weld_metrics():
    obj = _flat_benchmark_object()
    predefined = hlp.helper_read_section_file(str(CSV_PATH), obj.Stiffener)

    min_var = np.array([0.6, 0.015, 0.240817178, 0.010, 0.058, 0.029182822, 3.5, 10.0])
    max_var = np.array([0.6, 0.015, 0.240817178, 0.010, 0.058, 0.029182822, 3.5, 10.0])
    deltas = np.array([0.1, 0.005, 0.1, 0.005, 0.05, 0.005, 1.0, 1.0])

    for metric in ("weld_consumables", "weld_length"):
        result = opt.run_optmizataion(
            obj,
            min_var,
            max_var,
            271.124,
            deltas,
            algorithm="anysmart",
            const_chk=(False, False, False, False, False, False, False, False, False, False),
            predefined_stiffener_iter=predefined,
            processes=1,
            use_weight_filter=True,
            weld_bias=1.0,
            weld_metric=metric,
        )

        assert result[0] is not None
        assert result[2] is True
        assert result[0].Stiffener.get_stiffener_type() == "L-bulb"
        assert opt.calc_weld_objective(
            result[0].Stiffener.get_tuple(),
            weld_metric=metric,
        ) < float("inf")


def test_predefined_candidate_type_controls_weld_line_count():
    t_candidate = (0.6, 0.015, 0.240817178, 0.010, 0.058, 0.029182822, 3.5, 10.0, "T")
    bulb_candidate = (0.6, 0.015, 0.240817178, 0.010, 0.058, 0.029182822, 3.5, 10.0, "L-bulb")

    assert opt.calc_weld_length(bulb_candidate) == opt.calc_weld_length(t_candidate) / 2.0
    assert opt.calc_weld_consumable(bulb_candidate) == opt.calc_weld_consumable(t_candidate) / 2.0


def test_multiple_optimizer_reads_predefined_sections_from_stiffener_component():
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "optimize_multiple_window.py").read_text(
        encoding="utf-8"
    )

    assert "hlp.helper_read_section_file(files=found_files, obj=obj.Stiffener)" in source
    assert "hlp.helper_read_section_file(files=found_files, obj=obj.Plate)" not in source


def test_span_optimizer_forwards_predefined_file_with_weld_metric():
    source = (Path(__file__).resolve().parents[1] / "anystruct" / "optimize_geometry.py").read_text(
        encoding="utf-8"
    )

    assert "predefined_stiffener_iter=self._filez" in source
    assert "weld_metric=self._get_weld_metric_for_optimization()" in source
