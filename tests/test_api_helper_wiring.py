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

    assert source.count("api_helpers.normalize_bulb_stiffener_type") == 5


def test_api_uses_shared_cylinder_domain_helpers():
    api_source = Path(__file__).resolve().parents[1] / "anystruct" / "api.py"
    source = api_source.read_text(encoding="utf-8")

    assert "api_helpers.cylinder_input_mode(calculation_domain)" in source
    assert "api_helpers.cylinder_domain_with_input_mode(calculation_domain)" in source
    assert "api_helpers.geometry_id_for_domain(self._calculation_domain)" in source
    assert "geomeries_map_no_input_spec" not in source
    assert "geomeries = {" not in source


def test_api_uses_shared_unit_conversions():
    api_source = Path(__file__).resolve().parents[1] / "anystruct" / "api.py"
    source = api_source.read_text(encoding="utf-8")

    assert source.count("api_helpers.mpa_to_pa") == 13
    assert source.count("api_helpers.mm_to_m") == 8


def test_api_exposes_project_file_facade_through_application_services():
    api_source = Path(__file__).resolve().parents[1] / "anystruct" / "api.py"
    source = api_source.read_text(encoding="utf-8")

    assert "ProjectFileCodec" in source
    assert "ProjectHydrationDefaults" in source
    assert "ProjectPersistenceService.load_state_from_path(path)" in source
    assert "ProjectPersistenceService.save_state_to_path(project_state, path)" in source
    assert "ProjectOpenService.open_path(path, hydration_defaults)" in source
    assert "ProjectSaveService.save_path(path, save_input)" in source
