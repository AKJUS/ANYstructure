import pytest

from anystruct.api import (
    CylStru,
    FlatStru,
    ProjectFileCodec,
    ProjectHydrationDefaults,
    ProjectState,
    load_project_state,
    open_project,
    save_project_state,
)
from anystruct.project_state import PROJECT_FORMAT_VERSION


def test_flat_structure_api_returns_special_provisions():
    flat = FlatStru("Flat plate, stiffened")
    flat.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    flat.set_plate_geometry(spacing=700, thickness=18, span=4000)
    flat.set_stresses(pressure=0.2, sigma_x1=50, sigma_x2=50, sigma_y1=100, sigma_y2=100, tau_xy=5)
    flat.set_stiffener(hw=360, tw=12, bf=150, tf=20, stf_type="T", spacing=700)
    flat.set_fixation_parameters(kpp=1, kps=1, km1=12, km2=24, km3=12)
    flat.set_buckling_parameters(
        calculation_method="DNV-RP-C201 - prescriptive",
        buckling_acceptance="ultimate",
        stiffened_plate_effective_aginst_sigy=True,
    )

    results = flat.get_special_provisions_results()

    assert set(results) == {"Plate thickness", "Stiffener section modulus", "Stiffener shear area"}
    assert results == {
        "Plate thickness": {"minimum": pytest.approx(9.214115076089064), "actual": 18.0},
        "Stiffener section modulus": {
            "minimum": pytest.approx(842126.1216230252),
            "actual": pytest.approx(1514995.9581737241),
        },
        "Stiffener shear area": {"minimum": pytest.approx(1593.0323568831573), "actual": 4776.0},
    }

    buckling = flat.get_buckling_results()

    assert buckling["Plate"]["Plate buckling"] == pytest.approx(0.2819426684980083)
    assert buckling["Stiffener"]["Overpressure plate side"] == pytest.approx(0.8627465560212727)


def test_flat_structure_api_rejects_unknown_domain():
    with pytest.raises(AssertionError):
        FlatStru("not a domain")


def test_cylinder_api_sets_basic_unstiffened_shell_properties():
    cyl = CylStru(calculation_domain="Unstiffened shell")
    cyl.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    cyl.set_imperfection(delta_0=0.005)
    cyl.set_fabrication_method(stiffener="Fabricated", girder="Fabricated")
    cyl.set_end_cap_pressure_included_in_stress(is_included=True)
    cyl.set_uls_or_als(kind="ULS")
    cyl.set_shell_geometry(radius=6500, thickness=20, distance_between_rings=3000, tot_length_of_shell=12000)
    cyl.set_shell_buckling_parmeters(eff_buckling_length_factor=1.0)
    cyl.set_stresses(sasd=-50, smsd=0, tTsd=0, tQsd=10, psd=0.1, shsd=0)

    props = cyl._CylinderMain.get_main_properties()

    assert props["geometry"] == [1, ""]
    assert props["mat_yield"] == [355e6, "Pa"]
    assert props["ULS or ALS"] == ["ULS", ""]
    assert props["psd"] == [100000.0, "Pa"]


def test_cylinder_api_returns_unstiffened_shell_buckling_golden_result():
    cyl = CylStru(calculation_domain="Unstiffened shell")
    cyl.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    cyl.set_imperfection(delta_0=0.005)
    cyl.set_fabrication_method(stiffener="Fabricated", girder="Fabricated")
    cyl.set_end_cap_pressure_included_in_stress(is_included=True)
    cyl.set_uls_or_als(kind="ULS")
    cyl.set_shell_geometry(radius=6500, thickness=20, distance_between_rings=3000, tot_length_of_shell=12000)
    cyl.set_panel_spacing(700)
    cyl.set_shell_buckling_parmeters(eff_buckling_length_factor=1.0)
    cyl.set_stresses(sasd=-50, smsd=0, tTsd=0, tQsd=10, psd=0.1, shsd=0)

    results = cyl.get_buckling_results()

    assert results["Unstiffened shell"] == pytest.approx(0.8455659884505807)
    assert results["Longitudinal stiffened shell"] is None
    assert results["Ring stiffened shell"] is None
    assert results["Column stability UF"] is None


def test_cylinder_api_rejects_unknown_domain():
    with pytest.raises(AssertionError):
        CylStru(calculation_domain="not a domain")


def test_public_api_project_file_facade_round_trips_state(tmp_path):
    project_path = tmp_path / "api_project.txt"
    state = ProjectState(
        project_information="public api",
        theme="dark",
        points={"point1": [0, 0]},
        extras={"PULS results": {"obsolete": True}},
    )

    saved_path = save_project_state(state, project_path)
    loaded = load_project_state(project_path)
    opened = open_project(project_path, ProjectHydrationDefaults(structure_types={}))
    encoded = ProjectFileCodec.encode_mapping(loaded)

    assert saved_path == project_path
    assert loaded.project_information == "public api"
    assert opened.transfer.theme == "dark"
    assert encoded["format version"] == PROJECT_FORMAT_VERSION
    assert "PULS results" not in encoded
