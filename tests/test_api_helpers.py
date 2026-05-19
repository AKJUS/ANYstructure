import pytest

from anystruct import api_helpers


def test_assert_choice_accepts_known_value():
    api_helpers.assert_choice("ULS", api_helpers.LIMIT_STATE_TYPES, "limit state")


def test_assert_choice_rejects_unknown_value():
    with pytest.raises(AssertionError, match="limit state must be one of"):
        api_helpers.assert_choice("SLS", api_helpers.LIMIT_STATE_TYPES, "limit state")


def test_geometry_id_for_flat_domain():
    assert api_helpers.geometry_id_for_domain("Flat plate, unstiffened") == 10
    assert api_helpers.geometry_id_for_domain("Flat plate, stiffened") == 11
    assert api_helpers.geometry_id_for_domain("Flat plate, stiffened with girder") == 12


def test_geometry_id_for_cylinder_domain_with_input_mode():
    assert api_helpers.geometry_id_for_domain("Unstiffened shell (Force input)") == 1
    assert api_helpers.geometry_id_for_domain("Unstiffened panel (Stress input)") == 2
    assert api_helpers.geometry_id_for_domain("Orthogonally Stiffened shell (Force input)") == 7
    assert api_helpers.geometry_id_for_domain("Orthogonally Stiffened panel (Stress input)") == 8


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("Unstiffened shell", "Force"),
        ("Longitudinal Stiffened shell", "Force"),
        ("Unstiffened panel", "Stress"),
        ("Orthogonally Stiffened panel", "Stress"),
    ],
)
def test_cylinder_input_mode(domain, expected):
    assert api_helpers.cylinder_input_mode(domain) == expected


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        ("Unstiffened shell", "Unstiffened shell (Force input)"),
        ("Unstiffened panel", "Unstiffened panel (Stress input)"),
    ],
)
def test_cylinder_domain_with_input_mode(domain, expected):
    assert api_helpers.cylinder_domain_with_input_mode(domain) == expected


def test_geometry_id_rejects_unknown_domain():
    with pytest.raises(KeyError):
        api_helpers.geometry_id_for_domain("Unknown geometry")


def test_cylinder_input_mode_rejects_unknown_domain():
    with pytest.raises(AssertionError, match="calculation_domain must be one of"):
        api_helpers.cylinder_input_mode("Unknown geometry")


def test_unit_conversions():
    assert api_helpers.mpa_to_pa(355) == 355e6
    assert api_helpers.mm_to_m(6500) == 6.5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hp", "L-bulb"),
        ("HP", "L-bulb"),
        ("HP-bulb", "L-bulb"),
        ("bulb", "L-bulb"),
        ("T", "T"),
    ],
)
def test_normalize_bulb_stiffener_type(raw, expected):
    assert api_helpers.normalize_bulb_stiffener_type(raw) == expected


def test_domain_constants_include_public_api_values():
    assert "Flat plate, stiffened" in api_helpers.FLAT_STRUCTURE_DOMAINS
    assert "Orthogonally Stiffened shell" in api_helpers.CYLINDER_STRUCTURE_DOMAINS
