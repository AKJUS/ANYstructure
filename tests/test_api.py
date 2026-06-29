import math

import pytest
import numpy as np

from anystruct.api import (
    CylStru,
    FlatStru,
    ProjectFileCodec,
    ProjectHydrationDefaults,
    ProjectState,
    available_fea_stress_reduction_methods,
    load_project_state,
    open_project,
    save_project_state,
)
from anystruct.project_state import PROJECT_FORMAT_VERSION


class _IdentityScaler:
    def transform(self, rows):
        return np.asarray(rows, dtype=float)

    def inverse_transform(self, rows):
        return np.asarray(rows, dtype=float)


class _ValidPredictionModel:
    def predict(self, rows):
        return np.ones(len(rows), dtype=int)


class _NumericUfModel:
    def predict(self, rows):
        return np.asarray([[0.4, 0.5] for _ in rows], dtype=float)


def _numeric_ml_bundle(prefix):
    return {
        f"{prefix} validity predictor": _ValidPredictionModel(),
        f"{prefix} validity xscaler": _IdentityScaler(),
        f"{prefix} UF reg predictor": _NumericUfModel(),
        f"{prefix} UF reg xscaler": _IdentityScaler(),
        f"{prefix} UF reg yscaler": _IdentityScaler(),
    }


def _configured_stiffened_flat():
    flat = FlatStru("Flat plate, stiffened")
    flat.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    flat.set_plate_geometry(spacing=700, thickness=18, span=4000)
    flat.set_stresses(pressure=0.2, sigma_x1=50, sigma_x2=50, sigma_y1=100, sigma_y2=100, tau_xy=5)
    flat.set_stiffener(hw=360, tw=12, bf=150, tf=20, stf_type="T", spacing=700)
    flat.set_fixation_parameters(kpp=1, kps=1, km1=12, km2=24, km3=12)
    flat.set_puls_parameters(sp_or_up="SP", puls_boundary="Int", stiffener_end="Continuous", up_boundary="SSSS")
    return flat


def _configured_flat_domain(domain):
    flat = FlatStru(domain)
    flat.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    flat.set_plate_geometry(spacing=700, thickness=18, span=4000)
    flat.set_stresses(pressure=0.2, sigma_x1=50, sigma_x2=50, sigma_y1=80, sigma_y2=80, tau_xy=5)
    if domain != "Flat plate, unstiffened":
        flat.set_stiffener(hw=360, tw=12, bf=150, tf=20, stf_type="T", spacing=700)
    if domain == "Flat plate, stiffened with girder":
        flat.set_girder(hw=600, tw=15, bf=220, tf=25, stf_type="T", spacing=2800)
    flat.set_puls_parameters(sp_or_up=None, puls_boundary="Int", stiffener_end="Continuous", up_boundary="SSSS")
    return flat


def test_available_fea_stress_reduction_methods_are_exposed():
    assert available_fea_stress_reduction_methods() == (
        "CSR area weighted mean",
        "Whole panel nodal mean",
        "Centre strip mean",
    )


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


def test_flat_structure_api_selects_all_buckling_methods():
    flat = _configured_stiffened_flat()

    assert set(flat.get_available_buckling_methods()) == {
        "DNV-RP-C201 - prescriptive",
        "SemiAnalytical S3/U3",
        "ML-Numeric (PULS based)",
    }

    flat.set_buckling_parameters()
    dnv = flat.get_buckling_results()
    assert "Plate" in dnv
    assert "Stiffener" in dnv

    flat.set_buckling_parameters(calculation_method="SemiAnalytical S3/U3", buckling_acceptance="buckling")
    semi = flat.get_buckling_results()
    assert semi["method"] == "SemiAnalytical S3/U3"
    assert semi["panel family"] == "S3"
    assert semi["available"] is True
    assert semi["valid prediction"] == 1
    assert semi["buckling UF"] == pytest.approx(0.7962763046223863)
    assert semi["CSR"] == [1, 1, 1, 1]
    assert semi["CSR color"] == "green"
    assert semi["CSR requirement"]["within_csr_proportions"] is True
    assert "controlling limit" in semi
    assert "critical mode" in semi
    assert "critical failure family" in semi

    flat.set_buckling_parameters(calculation_method="ML-Numeric (PULS based)", buckling_acceptance="ultimate")
    ml_missing_model = flat.get_buckling_results()
    assert ml_missing_model["method"] == "ML-Numeric (PULS based)"
    assert ml_missing_model["available"] is False
    assert ml_missing_model["valid prediction"] is None
    assert ml_missing_model["valid label"] == "numeric ML unavailable"


def test_flat_structure_api_runs_numeric_ml_buckling_bundle():
    flat = _configured_stiffened_flat()
    flat.set_ml_buckling_model(_numeric_ml_bundle("num SP int"))
    flat.set_buckling_parameters(calculation_method="ML-Numeric (PULS based)", buckling_acceptance="ultimate")

    results = flat.get_buckling_results()

    assert results["available"] is True
    assert results["valid prediction"] == 1
    assert results["pipeline prefix"] == "num SP int"
    assert results["buckling UF raw"] == pytest.approx(0.4)
    assert results["ultimate UF raw"] == pytest.approx(0.5)
    assert results["buckling UF"] == pytest.approx(0.46)
    assert results["ultimate UF"] == pytest.approx(0.575)
    assert results["selected UF"] == pytest.approx(0.575)


@pytest.mark.parametrize(
    ("domain", "sp_or_up", "puls_boundary", "expected_prefix"),
    [
        ("Flat plate, stiffened", "SP", "Int", "num SP int"),
        ("Flat plate, stiffened", "SP", "GL", "num SP GLGT"),
        ("Flat plate, unstiffened", "UP", "Int", "num UP int"),
        ("Flat plate, unstiffened", "UP", "GT", "num UP GLGT"),
    ],
)
def test_flat_structure_api_runs_each_numeric_ml_pipeline_prefix(
    domain,
    sp_or_up,
    puls_boundary,
    expected_prefix,
):
    flat = _configured_flat_domain(domain)
    flat.set_puls_parameters(sp_or_up=sp_or_up, puls_boundary=puls_boundary)
    flat.set_buckling_parameters(
        calculation_method="ML-Numeric (PULS based)",
        buckling_acceptance="buckling",
        ml_algo=_numeric_ml_bundle(expected_prefix),
    )

    results = flat.get_buckling_results()

    assert results["available"] is True
    assert results["valid prediction"] == 1
    assert results["pipeline prefix"] == expected_prefix
    assert results["selected UF"] == pytest.approx(0.46)


@pytest.mark.parametrize("domain", [
    "Flat plate, unstiffened",
    "Flat plate, stiffened",
    "Flat plate, stiffened with girder",
])
@pytest.mark.parametrize("method", [
    "DNV-RP-C201 - prescriptive",
    "SemiAnalytical S3/U3",
    "ML-Numeric (PULS based)",
])
def test_flat_structure_api_applies_every_buckling_method_to_every_flat_domain(domain, method):
    flat = _configured_flat_domain(domain)

    flat.set_buckling_parameters(calculation_method=method, buckling_acceptance="ultimate")
    results = flat.get_buckling_results()

    assert isinstance(results, dict)
    if method == "DNV-RP-C201 - prescriptive":
        assert "Plate" in results
    else:
        assert results["method"] == method
        assert "buckling UF" in results
        assert "ultimate UF" in results


def test_flat_structure_api_accepts_unstiffened_up_semianalytical_input():
    flat = FlatStru("Flat plate, unstiffened")
    flat.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    flat.set_plate_geometry(spacing=700, thickness=18, span=4000)
    flat.set_stresses(pressure=0.2, sigma_x1=50, sigma_x2=50, sigma_y1=100, sigma_y2=100, tau_xy=5)
    flat.set_puls_parameters(sp_or_up="UP", puls_boundary="GL", up_boundary="SCSC")
    flat.set_buckling_parameters(calculation_method="SemiAnalytical S3/U3", buckling_acceptance="buckling")

    results = flat.get_buckling_results()

    assert flat.Plate.get_puls_sp_or_up() == "UP"
    assert flat.Plate.get_puls_boundary() == "GL"
    assert flat.Plate.get_puls_up_boundary() == "SCSC"
    assert results["panel family"] == "U3"
    assert results["available"] is True
    assert results["valid prediction"] == 1


def test_flat_structure_api_sets_all_flat_buckling_input_knobs():
    flat = FlatStru("Flat plate, stiffened with girder")
    flat.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    flat.set_plate_geometry(spacing=700, thickness=18, span=4000)
    flat.set_stresses(pressure=0.2, sigma_x1=50, sigma_x2=60, sigma_y1=100, sigma_y2=110, tau_xy=5)
    flat.set_stiffener(hw=360, tw=12, bf=150, tf=20, stf_type="T", spacing=700)
    flat.set_girder(hw=600, tw=15, bf=220, tf=25, stf_type="bulb", spacing=2800)
    flat.set_stresses(pressure=0.25, sigma_x1=70, sigma_x2=80, sigma_y1=120, sigma_y2=130, tau_xy=6)
    flat.set_buckling_parameters(
        calculation_method="DNV-RP-C201 - prescriptive",
        buckling_acceptance="ultimate",
        pressure_side="plate side",
        load_factor_stresses=1.2,
        load_factor_pressure=1.4,
        fabrication_method_stiffener="Fabricated",
        fabrication_method_girder="Cold formed",
        stiffener_support="Sniped",
        girder_support="Sniped",
    )

    assert flat._FlatStructure._overpressure_side == "plate side"
    assert flat._FlatStructure._stress_load_factor == pytest.approx(1.2)
    assert flat._FlatStructure._lat_load_factor == pytest.approx(1.4)
    assert flat._FlatStructure._fab_method_stiffener == "welded"
    assert flat._FlatStructure._fab_method_girder == "cold formed"
    assert flat._FlatStructure._stf_end_support == "Sniped"
    assert flat._FlatStructure._girder_end_support == "Sniped"
    assert flat._FlatStructure.Girder.get_stiffener_type() == "L-bulb"
    assert flat._FlatStructure.Girder.spacing == 2800
    assert flat._FlatStructure.Girder.span == flat._FlatStructure.Plate.span
    assert flat._FlatStructure.Girder.E == flat._FlatStructure.Plate.E
    assert flat._FlatStructure.Girder.sigma_x1 == flat._FlatStructure.Plate.sigma_x1
    assert flat._FlatStructure.Stiffener.sigma_x1 == flat._FlatStructure.Plate.sigma_x1


def test_flat_structure_api_rejects_unknown_domain():
    with pytest.raises(AssertionError):
        FlatStru("not a domain")


def test_flat_unstiffened_api_accepts_plate_only_stresses():
    flat = FlatStru("Flat plate, unstiffened")
    flat.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    flat.set_plate_geometry(spacing=700, thickness=18, span=4000)

    flat.set_stresses(pressure=0.2, sigma_x1=50, sigma_x2=50, sigma_y1=100, sigma_y2=100, tau_xy=5)

    assert flat._FlatStructure.Plate.tau_xy == 5
    assert flat._FlatStructure.Stiffener is None
    assert flat._FlatStructure.lat_press == 0.2


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


def test_cylinder_api_returns_unstiffened_conical_shell_buckling_result():
    cyl = CylStru(calculation_domain="Unstiffened conical shell")
    cyl.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    cyl.set_imperfection(delta_0=0.005)
    cyl.set_fabrication_method(stiffener="Fabricated", girder="Fabricated")
    cyl.set_end_cap_pressure_included_in_stress(is_included=True)
    cyl.set_uls_or_als(kind="ULS")
    cyl.set_conical_shell_geometry(r1=4000, r2=6500, length=5000, thickness=20)
    cyl.set_shell_buckling_parmeters(eff_buckling_length_factor=1.0)
    cyl.set_conical_forces(Nsd=-1000, M1sd=2000, M2sd=1000, Tsd=500, Q1sd=200, Q2sd=100, psd=-0.1)

    results = cyl.get_buckling_results()
    details = results["Unstiffened conical shell detailed"]

    assert results["Unstiffened conical shell"] == pytest.approx(1.5186769907266833)
    assert results["Unstiffened shell"] is None
    assert details["equivalent radius"] == pytest.approx(5.869678440936948)
    assert details["equivalent length"] == pytest.approx(5.5901699437494745)
    assert details["governing radius"] == pytest.approx(4.0)


def test_cylinder_api_accepts_labelled_conical_domain():
    cyl = CylStru(calculation_domain="Unstiffened conical shell (Force input)")

    assert cyl._calculation_domain == "Unstiffened conical shell (Force input)"
    assert cyl._load_type == "Force"
    assert cyl._CylinderMain.geometry == 9
    assert cyl._CylinderMain.LongStfObj is None
    assert cyl._CylinderMain.RingStfObj is None
    assert cyl._CylinderMain.RingFrameObj is None


def test_cylinder_api_generic_force_input_routes_to_conical_adapter():
    cyl = CylStru(calculation_domain="Unstiffened conical shell")
    cyl.set_conical_shell_geometry(r1=4000, r2=6500, length=5000, thickness=20)

    cyl.set_forces(Nsd=-1000, Msd=2000, Tsd=500, Qsd=200, psd=-0.1)

    assert cyl._CylinderMain._cone_Nsd == -1000
    assert cyl._CylinderMain._cone_M1sd == 2000
    assert cyl._CylinderMain._cone_M2sd == 0
    assert cyl._CylinderMain._cone_Tsd == 500
    assert cyl._CylinderMain._cone_Q1sd == 200
    assert cyl._CylinderMain._cone_Q2sd == 0
    assert cyl._CylinderMain.psd == pytest.approx(-0.1e6)


def test_cylinder_api_accepts_conical_stress_input():
    cyl = CylStru(calculation_domain="Unstiffened conical shell")
    cyl.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    cyl.set_imperfection(delta_0=0.005)
    cyl.set_fabrication_method(stiffener="Fabricated", girder="Fabricated")
    cyl.set_end_cap_pressure_included_in_stress(is_included=True)
    cyl.set_uls_or_als(kind="ULS")
    cyl.set_conical_shell_geometry(r1=4000, r2=6500, length=5000, thickness=20)
    cyl.set_shell_buckling_parmeters(eff_buckling_length_factor=1.0)

    cyl.set_stresses(sasd=-60, smsd=20, tTsd=3, tQsd=2, psd=0, shsd=0)
    results = cyl.get_buckling_results()
    details = results["Unstiffened conical shell detailed"]

    assert math.isfinite(results["Unstiffened conical shell"])
    assert cyl._CylinderMain._cone_M2sd == 0
    assert cyl._CylinderMain._cone_Q2sd == 0
    assert details["governing radius"] == pytest.approx(4.0)
    assert details["sasd"] == pytest.approx(-60e6)
    assert details["smsd"] == pytest.approx(20e6)
    assert details["tTsd"] == pytest.approx(3e6)
    assert details["tQsd"] == pytest.approx(2e6)


def test_cylinder_api_force_input_converts_pressure_to_pa():
    cyl = CylStru(calculation_domain="Unstiffened shell")
    cyl.set_shell_geometry(radius=6500, thickness=20, distance_between_rings=3000, tot_length_of_shell=12000)

    cyl.set_forces(Nsd=0, Msd=0, Tsd=0, Qsd=0, psd=0.1)

    assert cyl._CylinderMain.psd == pytest.approx(0.1e6)


def test_cylinder_api_force_input_uses_longitudinal_stiffener_meter_units():
    cyl = CylStru(calculation_domain="Longitudinal Stiffened shell")
    cyl.set_shell_geometry(radius=6500, thickness=20, distance_between_rings=3000, tot_length_of_shell=12000)
    cyl.set_longitudinal_stiffener(hw=260, tw=12, bf=80, tf=20, spacing=700)

    cyl.set_forces(Nsd=1000, Msd=0, Tsd=0, Qsd=0, psd=0.1)

    assert cyl._CylinderMain.sasd == pytest.approx(915585.6358015017)
    assert cyl._CylinderMain.psd == pytest.approx(0.1e6)


def test_cylinder_api_force_input_supports_ring_stiffened_without_longitudinal_stiffener():
    cyl = CylStru(calculation_domain="Ring Stiffened shell")
    cyl.set_shell_geometry(radius=6500, thickness=20, distance_between_rings=3000, tot_length_of_shell=12000)
    cyl.set_ring_stiffener(hw=300, tw=12, bf=120, tf=20, stf_type="T", spacing=3000)

    cyl.set_forces(Nsd=1000, Msd=0, Tsd=0, Qsd=0, psd=0.1)

    assert cyl._CylinderMain.sasd == pytest.approx(1224268.7930145795)
    assert cyl._CylinderMain.psd == pytest.approx(0.1e6)


def test_cylinder_api_rejects_unknown_domain():
    with pytest.raises(KeyError):
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
