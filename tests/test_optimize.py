from copy import deepcopy

from anystruct import optimize as opt, example_data as ex, calc_structure as calc
import numpy as np
import pytest

# Testing the Structure class

def _with_panel_default(obj_dict):
    obj_dict = deepcopy(obj_dict)
    obj_dict.setdefault('panel or shell', ['panel', ''])
    return obj_dict

@pytest.fixture
def opt_input():
    obj_dict = _with_panel_default(ex.obj_dict)
    fat_obj = ex.get_fatigue_object()
    fp = ex.get_fatigue_pressures()
    fat_press = ((fp['p_ext']['loaded'],fp['p_ext']['ballast'],fp['p_ext']['part']),
                 (fp['p_int']['loaded'],fp['p_int']['ballast'],fp['p_int']['part']))
    x0 = [obj_dict['spacing'][0], obj_dict['plate_thk'][0], obj_dict['stf_web_height'][0], obj_dict['stf_web_thk'][0],
          obj_dict['stf_flange_width'][0], obj_dict['stf_flange_thk'][0], obj_dict['span'][0], 10]
    scantlings = calc.CalcScantlings(obj_dict)
    obj = calc.AllStructure(Plate=scantlings, Stiffener=scantlings, Girder=None, main_dict=ex.prescriptive_main_dict)
    lat_press = 271.124
    upper_bounds = np.array([0.6, 0.01, 0.3, 0.01, 0.1, 0.01, 3.5, 10])
    lower_bounds = np.array([0.8, 0.02, 0.5, 0.02, 0.22, 0.03, 3.5, 10])
    deltas = np.array([0.05, 0.005, 0.05, 0.005, 0.05, 0.005])
    return obj, upper_bounds, lower_bounds, lat_press, deltas, fat_obj, fat_press, x0

def test_optimization(opt_input):
    obj, upper_bounds, lower_bounds, lat_press, deltas, fat_obj, fat_press, x0 = opt_input
    results = opt.run_optmizataion(obj, upper_bounds, lower_bounds, lat_press, deltas, algorithm='anysmart',
                                   fatigue_obj=fat_obj, fat_press_ext_int=fat_press)

    assert len(results) == 5
    assert results[3] in (True, False)

def test_weight_calc(opt_input):
    assert opt.calc_weight(opt_input[-1]) == pytest.approx(8125.711486965601)


def test_external_excel_puls_sheet_argument_is_removed(opt_input):
    obj, upper_bounds, lower_bounds, lat_press, deltas, fat_obj, fat_press, _ = opt_input

    with pytest.raises(NotImplementedError, match="External Excel-sheet PULS optimization was removed"):
        opt.run_optmizataion(
            obj,
            upper_bounds,
            lower_bounds,
            lat_press,
            deltas,
            puls_sheet="removed-puls.xlsm",
            fatigue_obj=fat_obj,
            fat_press_ext_int=fat_press,
        )


def test_puls_s3_optimizer_replacement_is_available(opt_input):
    obj, _, _, lat_press, _, _, _, x0 = opt_input

    calc_object_stf = opt.create_new_calc_obj(obj.Stiffener, x0, None)
    calc_object_pl = opt.create_new_calc_obj(obj.Plate, x0, None)
    calc_object = [
        calc.AllStructure(
            Plate=calc_object_pl[0],
            Stiffener=calc_object_stf[0],
            Girder=None,
            main_dict=obj.get_main_properties()['main dict'],
        ),
        calc_object_pl[1],
    ]

    puls_result = opt._predict_puls_s3_uf(calc_object, lat_press)

    assert puls_result[2] == 1
    assert np.isfinite(puls_result[0])
    assert np.isfinite(puls_result[1])

    check_ok, check_not_ok = opt.get_filtered_results(
        [x0],
        obj,
        lat_press,
        float('inf'),
        chk=(False, False, False, False, False, False, False, True, False, False),
        processes=1,
    )

    assert len(check_ok) + len(check_not_ok) == 1
    assert (check_ok or check_not_ok)[0][1] in ('Check OK', 'PULS-S3')
