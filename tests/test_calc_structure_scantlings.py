from copy import deepcopy

import pytest
from anystruct import example_data as ex, calc_structure as calc


# Testing the Structure class

def _with_panel_default(obj_dict):
    obj_dict = deepcopy(obj_dict)
    obj_dict.setdefault('panel or shell', ['panel', ''])
    return obj_dict

@pytest.fixture
def scantling_cls():
    return (calc.CalcScantlings(_with_panel_default(ex.obj_dict)),
            calc.CalcScantlings(_with_panel_default(ex.obj_dict2)),
            calc.CalcScantlings(_with_panel_default(ex.obj_dict_L)))

def test_eff_moment_of_intertia(scantling_cls):
    pressure = 200

    results = [item.get_moment_of_intertia(efficent_se=item.get_plate_efficent_b(design_lat_press=pressure))
               for item in scantling_cls]

    assert all(result > 0 for result in results)
    assert results[0] == pytest.approx(0.00015746749125364508)

def test_buckling_methods_live_on_all_structure():
    assert not hasattr(calc.CalcScantlings(_with_panel_default(ex.obj_dict)), 'calculate_buckling_all')
    assert hasattr(calc.AllStructure(), 'plate_buckling')

def test_minimum_plate_thickenss(scantling_cls):
    pressure = 200
    results = [item.get_dnv_min_thickness(pressure) for item in scantling_cls]

    assert all(result > 0 for result in results)
    assert results[0] == pytest.approx(8.964617963748333)

def test_minimum_section_module(scantling_cls):
    pressure = 200
    results = [item.get_dnv_min_section_modulus(pressure) for item in scantling_cls]

    assert all(result > 0 for result in results)
    assert results[0] == pytest.approx(0.0005585093810653399)
