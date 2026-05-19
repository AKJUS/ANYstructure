from copy import deepcopy

import pytest
from anystruct import example_data as ex, calc_structure as calc


# Testing the Structure class

def _with_panel_default(obj_dict):
    obj_dict = deepcopy(obj_dict)
    obj_dict.setdefault('panel or shell', ['panel', ''])
    return obj_dict

@pytest.fixture
def structure_cls():
    return (calc.Structure(_with_panel_default(ex.obj_dict)),
            calc.Structure(_with_panel_default(ex.obj_dict2)),
            calc.Structure(_with_panel_default(ex.obj_dict_L)))

def test_section_modulus(structure_cls):
    results = [item.get_section_modulus() for item in structure_cls]

    assert all(min(result) > 0 for result in results)
    assert results[0] == pytest.approx((0.0006303271987905085, 0.003096298566158356))

def test_shear_center(structure_cls):
    results = [item.get_shear_center() for item in structure_cls]

    assert all(result >= 0 for result in results)
    assert results[0] == pytest.approx(0.03894073197367731)

def test_shear_area(structure_cls):
    results = [item.get_shear_area() for item in structure_cls]

    assert all(result > 0 for result in results)
    assert results[0] == pytest.approx(0.0036600000000000005)

def test_plastic_sec_mod(structure_cls):
    results = [item.get_plasic_section_modulus() for item in structure_cls]

    assert all(result > 0 for result in results)
    assert results[0] == pytest.approx(0.01781929526454241)

def test_moment_of_intertia(structure_cls):
    results = [item.get_moment_of_intertia() for item in structure_cls]

    assert all(result > 0 for result in results)
    assert results[0] == pytest.approx(0.0001597323702732991)

def test_weight(structure_cls):
    results = [item.get_weight() for item in structure_cls]

    assert all(result > 0 for result in results)
    assert results[0] == pytest.approx(558.2036776404001)

def test_cross_section_area(structure_cls):
    results = [item.get_cross_section_area() for item in structure_cls]

    assert all(result > 0 for result in results)
    assert results[0] == pytest.approx(0.021548105680000002)

def test_input_properties(structure_cls):
    props = structure_cls[0].get_structure_prop()

    assert props['mat_yield'] == ex.obj_dict['mat_yield']
    assert props['span'] == ex.obj_dict['span']
    assert props['spacing'] == ex.obj_dict['spacing']
    assert props['plate_thk'] == ex.obj_dict['plate_thk']
    assert props['panel or shell'] == ['panel', '']
