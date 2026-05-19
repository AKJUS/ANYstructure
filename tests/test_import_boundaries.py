import importlib
import sys

import pytest


CORE_MODULES = [
    "anystruct.calc_loads",
    "anystruct.calc_structure",
    "anystruct.helper",
    "anystruct.make_grid_numpy",
    "anystruct.optimize",
]

PUBLIC_MODULES = [
    "anystruct.api",
    "anystruct.report_generator",
]


@pytest.mark.parametrize("module_name", CORE_MODULES + PUBLIC_MODULES)
def test_module_imports(module_name):
    module = importlib.import_module(module_name)

    assert module is not None


@pytest.mark.parametrize("module_name", CORE_MODULES)
def test_core_module_imports_without_tkinter_side_effect(module_name):
    sys.modules.pop("tkinter", None)
    sys.modules.pop("_tkinter", None)

    importlib.import_module(module_name)

    assert "tkinter" not in sys.modules
    assert "_tkinter" not in sys.modules
