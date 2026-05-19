import pytest

from anystruct import api_helpers


def test_assert_choice_accepts_known_value():
    api_helpers.assert_choice("ULS", api_helpers.LIMIT_STATE_TYPES, "limit state")


def test_assert_choice_rejects_unknown_value():
    with pytest.raises(AssertionError, match="limit state must be one of"):
        api_helpers.assert_choice("SLS", api_helpers.LIMIT_STATE_TYPES, "limit state")


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
