import pytest

from anystruct import api_helpers


def test_assert_choice_accepts_known_value():
    api_helpers.assert_choice("ULS", api_helpers.LIMIT_STATE_TYPES, "limit state")


def test_assert_choice_rejects_unknown_value():
    with pytest.raises(AssertionError, match="limit state must be one of"):
        api_helpers.assert_choice("SLS", api_helpers.LIMIT_STATE_TYPES, "limit state")


@pytest.mark.parametrize(("domain", "expected"), api_helpers.FLAT_GEOMETRY_IDS.items())
def test_geometry_id_for_flat_domain(domain, expected):
    assert api_helpers.geometry_id_for_domain(domain) == expected


@pytest.mark.parametrize(("domain", "expected"), api_helpers.CYLINDER_GEOMETRY_IDS.items())
def test_geometry_id_for_cylinder_domain_with_input_mode(domain, expected):
    assert api_helpers.geometry_id_for_domain(domain) == expected


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        (domain, "Stress" if "panel" in domain else "Force")
        for domain in api_helpers.CYLINDER_STRUCTURE_DOMAINS
    ],
)
def test_cylinder_input_mode(domain, expected):
    assert api_helpers.cylinder_input_mode(domain) == expected


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        (domain, expected)
        for domain, expected in zip(
            api_helpers.CYLINDER_STRUCTURE_DOMAINS,
            api_helpers.CYLINDER_STRUCTURE_DOMAINS_WITH_INPUT,
        )
    ],
)
def test_cylinder_domain_with_input_mode(domain, expected):
    assert api_helpers.cylinder_domain_with_input_mode(domain) == expected


def test_cylinder_domains_with_input_are_canonical_labels():
    assert api_helpers.CYLINDER_STRUCTURE_DOMAINS_WITH_INPUT == tuple(api_helpers.CYLINDER_GEOMETRY_IDS)
    assert all("  " not in domain for domain in api_helpers.CYLINDER_STRUCTURE_DOMAINS_WITH_INPUT)


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
