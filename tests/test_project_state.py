import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from anystruct import project_application, project_io
from anystruct.project_state import PROJECT_FORMAT_VERSION, ProjectState


def test_project_state_round_trips_compatible_json_shape_without_external_puls_results():
    legacy = {
        "project information": "demo",
        "theme": "dark",
        "point_dict": {"point1": [0, 0]},
        "line_dict": {"line1": [1, 2]},
        "structure_properties": {"line1": {"Plate": {}}},
        "shell structure properties": {},
        "load_properties": {},
        "accelerations_dict": {"static": 9.81},
        "load_combinations": {},
        "tank_properties": {},
        "fatigue_properties": {},
        "buckling method": "DNV-RP-C201 - prescriptive",
        "shifting": {"shifted checked": False},
        "Weight and COG": {"new structure": {}},
        "PULS results": {"line1": {"obsolete": True}, "sheet location": "dead.xlsm"},
    }

    state = ProjectState.from_legacy_mapping(legacy)
    written = io.StringIO()
    project_io.dump_project_state(state, written)
    payload = json.loads(written.getvalue())

    assert payload["format version"] == PROJECT_FORMAT_VERSION
    assert payload["point_dict"] == legacy["point_dict"]
    assert "PULS results" not in payload


def test_project_io_loads_old_puls_selected_projects_with_supported_method():
    old_project = io.StringIO(
        json.dumps(
            {
                "point_dict": {},
                "line_dict": {},
                "structure_properties": {},
                "PULS results": {"sheet location": "obsolete.xlsm"},
                "buckling method": "DNV PULS",
            }
        )
    )

    state = project_io.load_project_state(old_project)

    assert state.buckling_method == "DNV-RP-C201 - prescriptive"
    assert "PULS results" not in state.to_legacy_mapping()


def test_project_io_codec_migrates_legacy_payloads_at_the_file_boundary():
    legacy = {
        "format version": 0,
        "point_dict": {},
        "line_dict": {},
        "structure_properties": {},
        "PULS results": {"sheet location": "obsolete.xlsm"},
        "buckling method": "DNV PULS",
        "custom adapter state": {"kept": True},
    }

    state = project_io.decode_project_mapping(legacy)
    payload = project_io.encode_project_mapping(state)

    assert legacy["format version"] == 0
    assert "PULS results" in legacy
    assert state.format_version == PROJECT_FORMAT_VERSION
    assert state.buckling_method == "DNV-RP-C201 - prescriptive"
    assert payload["format version"] == PROJECT_FORMAT_VERSION
    assert payload["custom adapter state"] == {"kept": True}
    assert "PULS results" not in payload


def test_project_io_migrates_deactivated_ml_cl_to_ml_numeric():
    legacy = {
        "format version": 0,
        "point_dict": {},
        "line_dict": {},
        "structure_properties": {},
        "buckling method": "ML-CL (SemiAnalytical based)",
    }

    state = project_io.decode_project_mapping(legacy)

    assert state.buckling_method == "ML-Numeric (PULS based)"


def test_project_file_codec_is_the_versioned_migration_entry_point():
    legacy = {
        "format version": 0,
        "buckling method": "DNV PULS",
        "PULS results": {"sheet location": "obsolete.xlsm"},
        "custom adapter state": {"kept": True},
    }

    migrated = project_io.ProjectFileCodec.migrate_mapping(legacy)
    state = project_io.ProjectFileCodec.decode_mapping(legacy)
    payload = project_io.ProjectFileCodec.encode_mapping(state)

    assert legacy["format version"] == 0
    assert legacy["buckling method"] == "DNV PULS"
    assert "PULS results" in legacy
    assert migrated["format version"] == PROJECT_FORMAT_VERSION
    assert migrated["buckling method"] == "DNV-RP-C201 - prescriptive"
    assert "PULS results" not in migrated
    assert state.buckling_method == "DNV-RP-C201 - prescriptive"
    assert payload["custom adapter state"] == {"kept": True}


def test_project_file_codec_rejects_future_project_formats():
    future_project = {
        "format version": PROJECT_FORMAT_VERSION + 1,
        "point_dict": {},
    }

    with pytest.raises(ValueError, match="newer than supported"):
        project_io.ProjectFileCodec.decode_mapping(future_project)


def test_project_state_keeps_plain_legacy_mapping_while_codec_migrates_file_fields():
    state = ProjectState.from_legacy_mapping(
        {
            "buckling method": "DNV PULS",
            "PULS results": {"line1": {"obsolete": True}},
        }
    )

    raw_payload = state.to_legacy_mapping()
    encoded_payload = project_io.ProjectFileCodec.encode_mapping(state)

    assert raw_payload["buckling method"] == "DNV PULS"
    assert raw_payload["PULS results"] == {"line1": {"obsolete": True}}
    assert encoded_payload["buckling method"] == "DNV-RP-C201 - prescriptive"
    assert "PULS results" not in encoded_payload


def test_real_legacy_ship_fixture_round_trips_without_external_puls_results():
    fixture = Path(__file__).resolve().parents[1] / "anystruct" / "ship_section_example.txt"
    with fixture.open(encoding="utf-8") as project_file:
        state = project_io.load_project_state(project_file)

    written = io.StringIO()
    project_io.dump_project_state(state, written)
    payload = json.loads(written.getvalue())

    assert len(state.points) > 50
    assert len(state.lines) > 70
    assert "line14" in state.structures
    assert state.loads
    assert state.tanks
    assert "PULS results" not in payload
    assert "sheet location" not in written.getvalue()


def test_project_persistence_service_saves_and_loads_paths_and_locates_backup():
    with TemporaryDirectory(dir=Path.cwd()) as project_dir:
        project_path = Path(project_dir) / "project.txt"
        root_dir = Path(project_dir) / "anystruct"
        root_dir.mkdir()
        state = ProjectState(project_information="phase 2", points={"point1": [0, 0]})

        saved_path = project_application.ProjectPersistenceService.save_state_to_path(state, project_path)
        loaded = project_application.ProjectPersistenceService.load_state_from_path(project_path)

        assert saved_path == project_path
        assert loaded.project_information == "phase 2"
        assert loaded.points == {"point1": [0, 0]}
        assert project_application.ProjectPersistenceService.backup_path(root_dir) == (
            Path(project_dir) / "backup.txt"
        )
        assert not project_application.ProjectPersistenceService.backup_exists(root_dir)

        project_application.ProjectPersistenceService.save_state_to_path(
            state,
            project_application.ProjectPersistenceService.backup_path(root_dir),
        )

        assert project_application.ProjectPersistenceService.backup_exists(root_dir)


def test_project_persistence_service_uses_configured_file_codec(monkeypatch):
    class FakeCodec:
        saved_state = None
        loaded_text = None

        @classmethod
        def dump(cls, state, project_file):
            cls.saved_state = state
            project_file.write('{"encoded": true}')

        @classmethod
        def load(cls, project_file):
            cls.loaded_text = project_file.read()
            return ProjectState(project_information="loaded through fake codec")

    with TemporaryDirectory(dir=Path.cwd()) as project_dir:
        project_path = Path(project_dir) / "codec_project.txt"
        monkeypatch.setattr(project_application.ProjectPersistenceService, "codec", FakeCodec)

        state = ProjectState(project_information="codec save")
        saved_path = project_application.ProjectPersistenceService.save_state_to_path(state, project_path)
        loaded = project_application.ProjectPersistenceService.load_state_from_path(project_path)

        assert saved_path == project_path
        assert FakeCodec.saved_state is state
        assert project_path.read_text(encoding="utf-8") == '{"encoded": true}'
        assert FakeCodec.loaded_text == '{"encoded": true}'
        assert loaded.project_information == "loaded through fake codec"


def test_project_file_dialog_service_resolves_save_restore_and_example_targets():
    with TemporaryDirectory(dir=Path.cwd()) as project_dir:
        project_dir = Path(project_dir)
        root_dir = project_dir / "anystruct"
        root_dir.mkdir()
        local_example = project_dir / "local_example.txt"
        local_example.write_text("{}", encoding="utf-8")
        bundled_example = root_dir / "bundled_example.txt"
        bundled_example.write_text("{}", encoding="utf-8")
        bundled_workbook = root_dir / "excel_input_example.xlsx"
        bundled_workbook.write_text("", encoding="utf-8")

        selected = project_application.ProjectFileDialogService.selected_save_target(project_dir / "save.txt")
        selected_output = project_application.ProjectFileDialogService.selected_output_target(project_dir / "report.pdf")
        cancelled_output = project_application.ProjectFileDialogService.selected_output_target("")
        selected_open = project_application.ProjectFileDialogService.selected_open_target(project_dir / "open.txt")
        cancelled_open = project_application.ProjectFileDialogService.selected_open_target(None)
        backup = project_application.ProjectFileDialogService.backup_save_target(root_dir)
        remembered = project_application.ProjectFileDialogService.remembered_save_target(project_dir / "save.txt")
        missing_remembered = project_application.ProjectFileDialogService.remembered_save_target(None)
        missing_restore = project_application.ProjectFileDialogService.restore_target(root_dir)
        project_application.ProjectPersistenceService.save_state_to_path(ProjectState(), backup.path)
        restore = project_application.ProjectFileDialogService.restore_target(root_dir)
        direct_example = project_application.ProjectFileDialogService.example_open_target(local_example, root_dir)
        root_example = project_application.ProjectFileDialogService.example_open_target("bundled_example.txt", root_dir)
        workbook_example = project_application.ProjectFileDialogService.example_open_target(
            "excel_input_example.xlsx",
            root_dir,
        )

        assert selected.path == project_dir / "save.txt"
        assert selected.remember_as_last_save
        assert selected_output.path == project_dir / "report.pdf"
        assert not selected_output.remember_as_last_save
        assert cancelled_output is None
        assert selected_open.path == project_dir / "open.txt"
        assert not selected_open.remember_as_last_save
        assert cancelled_open is None
        assert backup.path == project_dir / "backup.txt"
        assert not backup.remember_as_last_save
        assert remembered.path == project_dir / "save.txt"
        assert missing_remembered is None
        assert missing_restore is None
        assert restore.path == project_dir / "backup.txt"
        assert direct_example.path == local_example
        assert root_example.path == bundled_example
        assert workbook_example.path == bundled_workbook


def test_project_persistence_service_wraps_invalid_project_json():
    with TemporaryDirectory(dir=Path.cwd()) as project_dir:
        project_path = Path(project_dir) / "broken.txt"
        project_path.write_text("{", encoding="utf-8")

        with pytest.raises(project_application.ProjectPersistenceError, match="Could not load project file"):
            project_application.ProjectPersistenceService.load_state_from_path(project_path)


def test_project_save_service_returns_written_state_and_path():
    with TemporaryDirectory(dir=Path.cwd()) as project_dir:
        project_path = Path(project_dir) / "save.txt"
        save_result = project_application.ProjectSaveService.save_path(
            project_path,
            project_application.ProjectSaveInput(
                project_information="save path",
                theme="dark",
                points={},
                lines={},
                line_bundles={},
                load_assignments={},
                accelerations={},
                load_combinations=(),
                tanks={},
                tank_grid=[],
                tank_search_data=None,
                buckling_method="DNV-RP-C201 - prescriptive",
                shifting={},
                weight_and_cog={},
            ),
        )

        loaded = project_application.ProjectPersistenceService.load_state_from_path(project_path)

        assert save_result.path == project_path
        assert save_result.state.project_information == "save path"
        assert loaded.theme == "dark"


def test_project_open_service_loads_path_before_creating_transfer_and_hydration():
    with TemporaryDirectory(dir=Path.cwd()) as project_dir:
        project_path = Path(project_dir) / "open.txt"
        project_application.ProjectPersistenceService.save_state_to_path(
            ProjectState(project_information="open path", theme="dark"),
            project_path,
        )

        opened = project_application.ProjectOpenService.open_path(
            project_path,
            project_application.ProjectHydrationDefaults(structure_types={}),
        )

        assert opened.state.project_information == "open path"
        assert opened.transfer.theme == "dark"
        assert opened.hydration.line_bundles == {}
