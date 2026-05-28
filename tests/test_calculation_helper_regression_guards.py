from copy import deepcopy

import pytest

from anystruct import calculate_semianalytical as semi
from anystruct import calc_structure as calc
from anystruct import example_data as ex
from anystruct import helper as hlp


def _panel_dict(source):
    data = deepcopy(source)
    data.setdefault("panel or shell", ["panel", ""])
    return data


def test_cylinder_force_converter_documents_meter_and_millimetre_calling_conventions():
    forces = (1000, 0, 0, 0)

    api_units = hlp.helper_cylinder_stress_to_force_to_stress(
        forces=forces,
        geometry=1,
        shell_t=0.02,
        shell_radius=6.5,
        shell_spacing=3.0,
        CylinderAndCurvedPlate=calc.CylinderAndCurvedPlate,
    )
    project_service_units = hlp.helper_cylinder_stress_to_force_to_stress(
        forces=forces,
        geometry=1,
        shell_t=20,
        shell_radius=6500,
        shell_spacing=3000,
        CylinderAndCurvedPlate=calc.CylinderAndCurvedPlate,
    )

    assert api_units[0] == pytest.approx(project_service_units[0] * 1e6)
    assert api_units[1:] == pytest.approx(project_service_units[1:])


def test_semi_analytical_helper_maps_stiffened_anystructure_input_units():
    plate = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    stiffener = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    structure = calc.AllStructure(
        Plate=plate,
        Stiffener=stiffener,
        Girder=None,
        main_dict=deepcopy(ex.prescriptive_main_dict),
    )
    structure.E = 210e9
    structure.v = 0.3

    panel = semi.anystructure_panel_input(structure, lat_press=200)

    assert isinstance(panel, semi.S3PanelInput)
    assert panel.length == pytest.approx(stiffener.span * 1000)
    assert panel.stiffener_spacing == pytest.approx(stiffener.spacing)
    assert panel.plate_thickness == pytest.approx(stiffener.t)
    assert panel.yield_stress_plate == pytest.approx(stiffener.mat_yield / 1e6)
    assert panel.pressure == pytest.approx(0.2)
    assert panel.elastic_modulus == pytest.approx(210000)


def test_semi_analytical_helper_maps_unstiffened_anystructure_input_units():
    plate = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    plate._puls_sp_or_up = "UP"
    structure = calc.AllStructure(
        Plate=plate,
        Stiffener=None,
        Girder=None,
        main_dict=deepcopy(ex.prescriptive_main_dict),
    )
    structure.E = 210e9
    structure.v = 0.3

    panel = semi.anystructure_panel_input(structure, lat_press=200)

    assert isinstance(panel, semi.U3PanelInput)
    assert panel.length == pytest.approx(plate.span * 1000)
    assert panel.width == pytest.approx(plate.spacing)
    assert panel.plate_thickness == pytest.approx(plate.t)
    assert panel.yield_stress_plate == pytest.approx(plate.mat_yield / 1e6)
    assert panel.pressure == pytest.approx(0.2)
    assert panel.elastic_modulus == pytest.approx(210000)
