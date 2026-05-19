from pathlib import Path


def test_api_uses_shared_helper_constants():
    api_source = Path(__file__).resolve().parents[1] / "anystruct" / "api.py"
    source = api_source.read_text(encoding="utf-8")

    assert "api_helpers.FLAT_STRUCTURE_DOMAINS" in source
    assert "api_helpers.CYLINDER_STRUCTURE_DOMAINS" in source
    assert "api_helpers.BUCKLING_CALCULATION_METHODS" in source
    assert "api_helpers.FABRICATION_METHODS" in source
    assert "api_helpers.LIMIT_STATE_TYPES" in source


def test_api_uses_shared_stiffener_normalizer():
    api_source = Path(__file__).resolve().parents[1] / "anystruct" / "api.py"
    source = api_source.read_text(encoding="utf-8")

    assert source.count("api_helpers.normalize_bulb_stiffener_type") == 4
