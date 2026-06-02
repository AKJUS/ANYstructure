from copy import deepcopy
import math
from types import SimpleNamespace

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


def _csr_s3_panel(**overrides):
    values = {
        "length": 3000.0,
        "stiffener_spacing": 700.0,
        "plate_thickness": 14.0,
        "stiffener_type": "T-bar",
        "stiffener_boundary": "Cont",
        "stiffener_height": 250.0,
        "web_thickness": 8.0,
        "flange_width": 120.0,
        "flange_thickness": 10.0,
        "yield_stress_plate": 355.0,
        "yield_stress_stiffener": 355.0,
        "axial_stress": 0.0,
        "transverse_stress_1": 0.0,
        "transverse_stress_2": 0.0,
        "shear_stress": 0.0,
        "pressure": 0.0,
        "in_plane_support": "Integrated",
    }
    values.update(overrides)
    return semi.S3PanelInput(**values)


def test_semianalytical_csr_equations_return_stiffened_component_flags():
    ok_result = semi.calculate_csr_requirement(_csr_s3_panel())

    assert ok_result["panel_family"] == "S3"
    assert ok_result["csr_vector"] == [1, 1, 1, 1]
    assert ok_result["within_csr_proportions"] is True
    assert ok_result["failed"] == []

    narrow_flange = semi.calculate_csr_requirement(_csr_s3_panel(flange_width=40.0))

    assert narrow_flange["csr_vector"] == [1, 1, 0, 1]
    assert narrow_flange["within_csr_proportions"] is False
    assert narrow_flange["failed"] == ["web_flange_ratio"]


def test_semianalytical_csr_equations_use_net_thickness_after_corrosion_addition():
    gross_only = semi.calculate_csr_requirement(_csr_s3_panel(plate_thickness=10.0))
    corroded = semi.calculate_csr_requirement(
        _csr_s3_panel(plate_thickness=10.0, plate_corrosion_addition=2.0)
    )

    assert gross_only["checks"]["plate_slenderness"] is True
    assert gross_only["csr_vector"][0] == 1
    assert corroded["checks"]["plate_slenderness"] is False
    assert corroded["csr_vector"][0] == 0
    assert corroded["values"]["plate_net_thickness"] == pytest.approx(8.0)


def test_semianalytical_csr_equations_return_plate_only_flags_for_u3():
    result = semi.calculate_csr_requirement(
        semi.U3PanelInput(
            length=3000.0,
            width=800.0,
            plate_thickness=10.0,
            yield_stress_plate=235.0,
            axial_stress_1=0.0,
            axial_stress_2=0.0,
            transverse_stress_1=0.0,
            transverse_stress_2=0.0,
            shear_stress=0.0,
            pressure=0.0,
            in_plane_support="Integrated",
        )
    )

    assert result["panel_family"] == "U3"
    assert result["csr_vector"][0] == 1
    assert all(math.isinf(value) for value in result["csr_vector"][1:])
    assert result["within_csr_proportions"] is True


def test_anystructure_csr_equation_adapter_returns_gui_vector_and_color():
    plate = SimpleNamespace(get_puls_sp_or_up=lambda: "SP")
    stiffener = SimpleNamespace(
        span=3.0,
        spacing=700.0,
        t=14.0,
        hw=250.0,
        tw=8.0,
        b=120.0,
        tf=10.0,
        mat_yield=355.0e6,
        sigma_x1=0.0,
        sigma_x2=0.0,
        sigma_y1=0.0,
        sigma_y2=0.0,
        tau_xy=0.0,
        get_puls_boundary=lambda: "Int",
        get_stiffener_type=lambda: "T",
        get_puls_stf_end=lambda: "Continuous",
    )
    structure = SimpleNamespace(Plate=plate, Stiffener=stiffener, E=210.0e9, v=0.3)

    csr, color, diagnostics = semi.predict_anystructure_csr_requirement(structure, lat_press=200.0)

    assert csr == [1, 1, 1, 1]
    assert color == "green"
    assert diagnostics["within_csr_proportions"] is True
    assert diagnostics["values"]["plate_corrosion_addition"] == pytest.approx(
        semi.DEFAULT_ANYSTRUCTURE_CSR_CORROSION_ADDITION_MM
    )
