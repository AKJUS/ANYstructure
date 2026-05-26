"""Versioned JSON project persistence helpers."""

import json
from typing import Any, IO

from anystruct.project_state import PROJECT_FORMAT_VERSION, ProjectState


PROJECT_FORMAT_KEY = "format version"
REMOVED_EXTERNAL_PULS_METHOD = "DNV PULS"
DEFAULT_BUCKLING_METHOD = "DNV-RP-C201 - prescriptive"


class ProjectFileCodec:
    """Versioned project-file codec and migration entry point."""

    current_version = PROJECT_FORMAT_VERSION
    format_key = PROJECT_FORMAT_KEY
    removed_external_puls_method = REMOVED_EXTERNAL_PULS_METHOD
    default_buckling_method = DEFAULT_BUCKLING_METHOD
    obsolete_external_puls_keys = ("PULS results",)

    @classmethod
    def migrate_mapping(cls, project_data: dict[str, Any]) -> dict[str, Any]:
        """Normalize a loaded project mapping into the current supported shape."""
        source_version = cls._source_version(project_data)
        if source_version > cls.current_version:
            raise ValueError(
                f"Project format version {source_version} is newer than supported version {cls.current_version}"
            )
        migrated = dict(project_data)
        for obsolete_key in cls.obsolete_external_puls_keys:
            migrated.pop(obsolete_key, None)
        if migrated.get("buckling method") == cls.removed_external_puls_method:
            migrated["buckling method"] = cls.default_buckling_method
        migrated[cls.format_key] = cls.current_version
        return migrated

    @classmethod
    def _source_version(cls, project_data: dict[str, Any]) -> int:
        if cls.format_key not in project_data:
            return 0
        return int(project_data[cls.format_key])

    @classmethod
    def decode_mapping(cls, project_data: dict[str, Any]) -> ProjectState:
        """Decode current or legacy project payloads into canonical project state."""
        return ProjectState.from_legacy_mapping(cls.migrate_mapping(project_data))

    @classmethod
    def encode_mapping(cls, project_state: ProjectState) -> dict[str, Any]:
        """Encode canonical project state using the compatible JSON project shape."""
        payload = cls.migrate_mapping(project_state.to_legacy_mapping())
        payload[cls.format_key] = cls.current_version
        return payload

    @classmethod
    def load(cls, project_file: IO[str]) -> ProjectState:
        """Load current or legacy project JSON and drop obsolete external PULS state."""
        return cls.decode_mapping(json.load(project_file))

    @classmethod
    def dump(cls, project_state: ProjectState, project_file: IO[str]) -> None:
        """Write the canonical state using the compatible JSON project shape."""
        json.dump(cls.encode_mapping(project_state), project_file)


def migrate_project_mapping(project_data: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the versioned project-file codec."""
    return ProjectFileCodec.migrate_mapping(project_data)


def decode_project_mapping(project_data: dict[str, Any]) -> ProjectState:
    """Compatibility wrapper for the versioned project-file codec."""
    return ProjectFileCodec.decode_mapping(project_data)


def encode_project_mapping(project_state: ProjectState) -> dict[str, Any]:
    """Compatibility wrapper for the versioned project-file codec."""
    return ProjectFileCodec.encode_mapping(project_state)


def load_project_state(project_file: IO[str]) -> ProjectState:
    """Compatibility wrapper for the versioned project-file codec."""
    return ProjectFileCodec.load(project_file)


def dump_project_state(project_state: ProjectState, project_file: IO[str]) -> None:
    """Compatibility wrapper for the versioned project-file codec."""
    ProjectFileCodec.dump(project_state, project_file)
