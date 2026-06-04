from copy import deepcopy
import math

import pytest

from anystruct import calc_structure as calc, example_data as ex
from anystruct.api import CylStru


def _panel_dict(source):
    data = deepcopy(source)
    data.setdefault("panel or shell", ["panel", ""])
    return data


def _numeric_values(value):
    if isinstance(value, dict):
        for sub_value in value.values():
            yield from _numeric_values(sub_value)
    elif isinstance(value, (list, tuple)):
        for sub_value in value:
            yield from _numeric_values(sub_value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def test_local_buckling_uses_mpa_yield_with_mm_dimensions():
    scantling = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    structure = calc.AllStructure(
        Plate=scantling,
        Stiffener=scantling,
        main_dict=deepcopy(ex.prescriptive_main_dict),
    )

    local = structure.local_buckling()
    acceptable, utilization = scantling.buckling_local_stiffener()

    assert acceptable
    assert utilization == pytest.approx(0.610387982352254)
    assert local["Stiffener"][0] == pytest.approx(410.0627227872807)
    assert local["Stiffener"][1] == pytest.approx(338.33184034710676)


def test_hp_inertia_path_returns_value_without_demo_output(capsys):
    structure = calc.Structure(_panel_dict(ex.obj_dict))

    inertia = structure.get_moment_of_intertia_hp()

    assert inertia == pytest.approx(structure.get_moment_of_intertia())
    assert inertia > 0
    assert capsys.readouterr().out == ""


def test_stiffener_centroid_uses_flange_area_not_web_thickness():
    structure = calc.Structure(_panel_dict(ex.obj_dict))

    assert structure.get_stf_cog_eccentricity() == pytest.approx(0.17269270039573859)


def test_overstressed_special_provision_requirements_do_not_collapse_to_minimum():
    scantling = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    scantling.set_stresses(sigy1=700, sigy2=700, sigx1=700, sigx2=700, tauxy=0)

    assert math.isinf(scantling.get_dnv_min_section_modulus(200))
    assert math.isinf(scantling.get_dnv_min_thickness(200))


def test_overstressed_shear_area_requirement_returns_infinity():
    scantling = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    scantling.set_stresses(sigy1=0, sigy2=0, sigx1=700, sigx2=700, tauxy=0)

    assert math.isinf(scantling.get_minimum_shear_area(200))


def test_effective_plate_width_clamps_out_of_range_reduction_terms():
    scantling = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    scantling._spacing = 0.2
    scantling._plate_th = 0.03

    stocky = scantling.get_plate_efficent_b(
        design_lat_press=200,
        axial_stress=50,
        trans_stress_small=100,
        trans_stress_large=100,
    )
    overloaded = scantling.get_plate_efficent_b(
        design_lat_press=200,
        axial_stress=50,
        trans_stress_small=700,
        trans_stress_large=700,
    )

    assert stocky >= 0
    assert overloaded == 0


def test_slamming_failure_ratios_are_dimensionless_not_scaled_by_millions():
    scantling = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    stf_result = scantling.calculate_slamming_stiffener(1_000_000)

    ok, ratio = scantling.check_all_slamming(1_000_000)

    assert not ok
    assert ratio == pytest.approx(stf_result["tw_req"] / scantling.tw)


def test_report_summary_uses_all_structure_buckling_path():
    scantling = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    report = scantling.get_results_for_report(lat_press=200)

    assert "Minimum section modulus:" in report
    assert "Buckling results:" in report


def test_sniped_girder_results_are_read_from_girder_branch():
    plate = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    stiffener = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    girder = calc.CalcScantlings(_panel_dict(ex.obj_dict_heavy))
    main = deepcopy(ex.prescriptive_main_dict)
    main["calculation domain"] = ["Flat plate, stiffened with girder", ""]
    main["girder end support"] = ["Sniped", ""]
    structure = calc.AllStructure(Plate=plate, Stiffener=stiffener, Girder=girder, main_dict=main)
    structure.lat_press = 0.2

    results = structure.plate_buckling()

    assert results["Girder"]["Overpressure plate side"] > 0
    assert results["Girder"]["Overpressure girder side"] > 0


def test_unstiffened_plate_handles_mixed_longitudinal_stress_gradient():
    plate = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    plate.set_stresses(sigy1=0, sigy2=0, sigx1=100, sigx2=-50, tauxy=0)
    structure = calc.AllStructure(
        Plate=plate,
        Stiffener=None,
        Girder=None,
        main_dict=deepcopy(ex.prescriptive_main_dict),
    )

    results = structure.unstiffened_plate_buckling()

    assert results["UF Longitudinal stress"] >= 0


def test_transverse_stress_uses_required_seventy_five_percent_floor():
    plate = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    plate.set_stresses(sigy1=100, sigy2=-300, sigx1=0, sigx2=0, tauxy=0)
    structure = calc.AllStructure(
        Plate=plate,
        Stiffener=None,
        Girder=None,
        main_dict=deepcopy(ex.prescriptive_main_dict),
    )

    results = structure.unstiffened_plate_buckling()

    assert results["sysd"] == pytest.approx(75)


def test_stocky_plate_longitudinal_reduction_factor_is_capped_at_yield():
    plate = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    plate._spacing = 0.2
    plate._plate_th = 0.03
    plate.set_stresses(sigy1=0, sigy2=0, sigx1=10, sigx2=10, tauxy=0)
    structure = calc.AllStructure(
        Plate=plate,
        Stiffener=None,
        Girder=None,
        main_dict=deepcopy(ex.prescriptive_main_dict),
    )

    results = structure.unstiffened_plate_buckling()

    assert results["UF Longitudinal stress"] == pytest.approx(
        abs(results["sxsd"]) / (plate.mat_yield / 1e6 / plate.mat_factor)
    )


def test_flat_plate_buckling_is_symmetric_for_shear_sign():
    plate_positive = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    plate_negative = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    plate_positive.set_stresses(sigy1=20, sigy2=10, sigx1=50, sigx2=30, tauxy=25)
    plate_negative.set_stresses(sigy1=20, sigy2=10, sigx1=50, sigx2=30, tauxy=-25)

    structure_positive = calc.AllStructure(
        Plate=plate_positive,
        Stiffener=plate_positive,
        main_dict=deepcopy(ex.prescriptive_main_dict),
    )
    structure_negative = calc.AllStructure(
        Plate=plate_negative,
        Stiffener=plate_negative,
        main_dict=deepcopy(ex.prescriptive_main_dict),
    )
    structure_positive.lat_press = 0.1
    structure_negative.lat_press = 0.1

    positive = structure_positive.plate_buckling()
    negative = structure_negative.plate_buckling()

    assert negative["Plate"]["Plate buckling"] == pytest.approx(positive["Plate"]["Plate buckling"])
    assert negative["Stiffener"]["Resistance between stiffeners"] == pytest.approx(
        positive["Stiffener"]["Resistance between stiffeners"]
    )
    assert negative["Stiffener"]["Shear capacity"] == pytest.approx(positive["Stiffener"]["Shear capacity"])


def test_unstiffened_plate_shear_uses_stress_load_factor():
    plate = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    plate.set_stresses(sigy1=0, sigy2=0, sigx1=0, sigx2=0, tauxy=20)
    base_main = deepcopy(ex.prescriptive_main_dict)
    factored_main = deepcopy(ex.prescriptive_main_dict)
    base_main["load factor on stresses"] = [1, ""]
    factored_main["load factor on stresses"] = [2, ""]

    base = calc.AllStructure(
        Plate=plate,
        Stiffener=None,
        Girder=None,
        main_dict=base_main,
    ).unstiffened_plate_buckling()
    factored = calc.AllStructure(
        Plate=plate,
        Stiffener=None,
        Girder=None,
        main_dict=factored_main,
    ).unstiffened_plate_buckling()

    assert factored["UF Shear stresses"] == pytest.approx(2 * base["UF Shear stresses"])


def test_stiffened_panel_zero_load_returns_zero_utilization():
    plate = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    plate.set_stresses(sigy1=0, sigy2=0, sigx1=0, sigx2=0, tauxy=0)
    structure = calc.AllStructure(
        Plate=plate,
        Stiffener=plate,
        main_dict=deepcopy(ex.prescriptive_main_dict),
    )
    structure.lat_press = 0

    results = structure.plate_buckling()

    assert results["Plate"]["Plate buckling"] == pytest.approx(0)
    assert results["Stiffener"]["Overpressure plate side"] == pytest.approx(0)
    assert results["Stiffener"]["Overpressure stiffener side"] == pytest.approx(0)
    assert results["Stiffener"]["Resistance between stiffeners"] == pytest.approx(0)
    assert results["Stiffener"]["Shear capacity"] == pytest.approx(0)


def test_girder_branch_handles_mixed_transverse_stress_gradient():
    plate = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    stiffener = calc.CalcScantlings(_panel_dict(ex.obj_dict))
    girder = calc.CalcScantlings(_panel_dict(ex.obj_dict_heavy))
    plate.span = 2.0
    stiffener.span = 2.0
    girder.span = 2.0
    plate.spacing = 0.95
    stiffener.spacing = 0.95
    girder.spacing = 0.95
    plate.set_stresses(sigy1=15, sigy2=-50, sigx1=50, sigx2=50, tauxy=5)
    main = deepcopy(ex.prescriptive_main_dict)
    main["calculation domain"] = ["Flat plate, stiffened with girder", ""]
    structure = calc.AllStructure(Plate=plate, Stiffener=stiffener, Girder=girder, main_dict=main)
    structure.lat_press = 0.1

    results = structure.plate_buckling()

    assert results["Girder"]["Overpressure plate side"] >= 0


def test_column_buckling_can_build_missing_shell_data_internally():
    cylinder = calc.CylinderAndCurvedPlate(
        main_dict=deepcopy(ex.shell_main_dict2),
        shell=calc.Shell(deepcopy(ex.shell_dict)),
        long_stf=None,
        ring_stf=calc.Structure(deepcopy(ex.obj_dict_cyl_ring2)),
        ring_frame=None,
    )

    results = cylinder.column_buckling()

    assert "Column stability check" in results
    assert "Column stability UF" in results


def test_column_buckling_reduction_formula_stays_positive_for_moderate_slenderness():
    fak = 100.0
    lambda_value = 0.1

    corrected = calc.CylinderAndCurvedPlate._column_reduced_buckling_strength(fak, lambda_value)

    assert corrected == pytest.approx(99.72)
    assert corrected > 0


@pytest.mark.parametrize(
    "domain",
    [
        "Unstiffened shell",
        "Unstiffened panel",
        "Longitudinal Stiffened shell",
        "Longitudinal Stiffened panel",
        "Ring Stiffened shell",
        "Ring Stiffened panel",
        "Orthogonally Stiffened shell",
        "Orthogonally Stiffened panel",
        "Unstiffened conical shell",
    ],
)
def test_cylinder_api_domains_return_finite_results(domain):
    cylinder = CylStru(calculation_domain=domain)
    cylinder.set_material(mat_yield=355, emodule=210000, material_factor=1.15, poisson=0.3)
    cylinder.set_imperfection(delta_0=0.005)
    cylinder.set_fabrication_method(stiffener="Fabricated", girder="Fabricated")
    cylinder.set_end_cap_pressure_included_in_stress(is_included=True)
    cylinder.set_uls_or_als(kind="ULS")
    if domain == "Unstiffened conical shell":
        cylinder.set_conical_shell_geometry(r1=4000, r2=6500, length=5000, thickness=20)
    else:
        cylinder.set_shell_geometry(radius=6500, thickness=20, distance_between_rings=3300, tot_length_of_shell=20000)
    cylinder.set_panel_spacing(700)
    cylinder.set_shell_buckling_parmeters(eff_buckling_length_factor=1.0)
    if domain == "Unstiffened conical shell":
        cylinder.set_conical_forces(Nsd=-1000, M1sd=2000, M2sd=1000, Tsd=500, Q1sd=200, Q2sd=100, psd=-0.1)
    else:
        cylinder.set_stresses(sasd=-80, smsd=20, tTsd=10, tQsd=5, psd=-0.1, shsd=0)
    if "Longitudinal" in domain or "Orthogonally" in domain:
        cylinder.set_longitudinal_stiffener(hw=260, tw=12, bf=80, tf=20, spacing=700)
    if "Ring Stiffened" in domain:
        cylinder.set_ring_stiffener(hw=300, tw=12, bf=120, tf=20, stf_type="T", spacing=3300)
    if "Orthogonally" in domain:
        cylinder.set_ring_girder(hw=500, tw=15, bf=200, tf=25, stf_type="T", spacing=3300)

    results = cylinder.get_buckling_results()
    numeric_values = _numeric_values(results)

    assert numeric_values
    assert all(math.isfinite(value) for value in numeric_values)


def test_fatigue_slope2_applies_same_thickness_reference_as_slope1():
    fatigue = calc.CalcFatigue(_panel_dict(ex.obj_dict), deepcopy(ex.fat_obj_dict))
    baseline = fatigue.get_damage_slope2(0, "Ec", int_press=0, ext_press=50_000)

    fatigue._plate_th = 0.05
    doubled_thickness = fatigue.get_damage_slope2(0, "Ec", int_press=0, ext_press=50_000)

    assert doubled_thickness < baseline
    assert doubled_thickness == pytest.approx(4.6011397118183774e-06)
