"""Versioned JSON project persistence helpers."""

import json
from typing import Any, IO

from anystruct.project_state import PROJECT_FORMAT_VERSION, ProjectState


PROJECT_FORMAT_KEY = "format version"
REMOVED_EXTERNAL_PULS_METHOD = "DNV PULS"
DEFAULT_BUCKLING_METHOD = "DNV-RP-C201 - prescriptive"


def migrate_project_mapping(project_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize a loaded project mapping into the current supported shape."""
    migrated = dict(project_data)
    migrated.pop("PULS results", None)
    if migrated.get("buckling method") == REMOVED_EXTERNAL_PULS_METHOD:
        migrated["buckling method"] = DEFAULT_BUCKLING_METHOD
    migrated[PROJECT_FORMAT_KEY] = PROJECT_FORMAT_VERSION
    return migrated


def decode_project_mapping(project_data: dict[str, Any]) -> ProjectState:
    """Decode current or legacy project payloads into canonical project state."""
    return ProjectState.from_legacy_mapping(migrate_project_mapping(project_data))


def encode_project_mapping(project_state: ProjectState) -> dict[str, Any]:
    """Encode canonical project state using the compatible JSON project shape."""
    payload = project_state.to_legacy_mapping()
    payload[PROJECT_FORMAT_KEY] = PROJECT_FORMAT_VERSION
    payload.pop("PULS results", None)
    return payload


def load_project_state(project_file: IO[str]) -> ProjectState:
    """Load current or legacy project JSON and drop obsolete external PULS state."""
    return decode_project_mapping(json.load(project_file))


def dump_project_state(project_state: ProjectState, project_file: IO[str]) -> None:
    """Write the canonical state using the compatible JSON project shape."""
    json.dump(encode_project_mapping(project_state), project_file)
