from copy import deepcopy

from anystruct import calc_loads as calcl, example_data as ex
import pytest


# Testing the Structure class

@pytest.fixture
def load_cls():
    slamming_load = deepcopy(ex.load_slamming)
    slamming_load.setdefault('slamming mult pl', 1)
    slamming_load.setdefault('slamming mult stf', 1)
    return calcl.Loads(ex.load_bottom), calcl.Loads(ex.load_side), \
           calcl.Loads(ex.load_static), calcl.Loads(slamming_load)

def test_get_bottom_pressure(load_cls):
    for load, type, result in zip(load_cls, ['BOTTOM', 'SIDE_SHELL', '', ''], [48070.0, 145800.0, 15375.0, 1000000.0]):
        assert load.get_calculated_pressure((10,10), 3, type) == result
