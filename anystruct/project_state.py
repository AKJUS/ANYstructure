"""Canonical project data used at persistence and application boundaries."""

from dataclasses import dataclass, field
from typing import Any


PROJECT_FORMAT_VERSION = 1


@dataclass
class ProjectState:
    """JSON-safe project snapshot independent of Tkinter widget state."""

    project_information: str = ""
    theme: str = "default"
    points: dict[str, Any] = field(default_factory=dict)
    lines: dict[str, Any] = field(default_factory=dict)
    structures: dict[str, Any] = field(default_factory=dict)
    shell_structures: dict[str, Any] = field(default_factory=dict)
    loads: dict[str, Any] = field(default_factory=dict)
    accelerations: dict[str, Any] = field(default_factory=dict)
    load_combinations: dict[str, Any] = field(default_factory=dict)
    tanks: dict[str, Any] = field(default_factory=dict)
    fatigue: dict[str, Any] = field(default_factory=dict)
    buckling_method: str = "DNV-RP-C201 - prescriptive"
    shifting: dict[str, Any] = field(default_factory=dict)
    weight_and_cog: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    format_version: int = PROJECT_FORMAT_VERSION

    @classmethod
    def from_legacy_mapping(cls, data: dict[str, Any]) -> "ProjectState":
        known_keys = {
            "format version",
            "project information",
            "theme",
            "point_dict",
            "line_dict",
            "structure_properties",
            "shell structure properties",
            "load_properties",
            "accelerations_dict",
            "load_combinations",
            "tank_properties",
            "fatigue_properties",
            "buckling method",
            "shifting",
            "Weight and COG",
            "PULS results",
        }
        buckling_method = data.get("buckling method", "DNV-RP-C201 - prescriptive")
        if buckling_method == "DNV PULS":
            buckling_method = "DNV-RP-C201 - prescriptive"

        return cls(
            project_information=data.get("project information", ""),
            theme=data.get("theme", "default"),
            points=data.get("point_dict", {}),
            lines=data.get("line_dict", {}),
            structures=data.get("structure_properties", {}),
            shell_structures=data.get("shell structure properties", {}),
            loads=data.get("load_properties", {}),
            accelerations=data.get("accelerations_dict", {}),
            load_combinations=data.get("load_combinations", {}),
            tanks=data.get("tank_properties", {}),
            fatigue=data.get("fatigue_properties", {}),
            buckling_method=buckling_method,
            shifting=data.get("shifting", {}),
            weight_and_cog=data.get("Weight and COG", {}),
            extras={key: value for key, value in data.items() if key not in known_keys},
            format_version=data.get("format version", PROJECT_FORMAT_VERSION),
        )

    def to_legacy_mapping(self) -> dict[str, Any]:
        data = dict(self.extras)
        data.update(
            {
                "format version": self.format_version,
                "project information": self.project_information,
                "theme": self.theme,
                "point_dict": self.points,
                "line_dict": self.lines,
                "structure_properties": self.structures,
                "shell structure properties": self.shell_structures,
                "load_properties": self.loads,
                "accelerations_dict": self.accelerations,
                "load_combinations": self.load_combinations,
                "tank_properties": self.tanks,
                "fatigue_properties": self.fatigue,
                "buckling method": self.buckling_method,
                "shifting": self.shifting,
                "Weight and COG": self.weight_and_cog,
            }
        )
        return data
