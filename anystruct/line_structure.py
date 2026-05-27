"""Named accessors for the existing line-to-structure bundle format."""

import copy
from dataclasses import dataclass, field
from typing import Any


STRUCTURE = 0
LEGACY_CALC_OBJECT = 1
FATIGUE = 2
LOADS = 3
LOAD_COMBINATIONS = 4
CYLINDER = 5


@dataclass
class LineStructureBundle:
    """Typed adapter for the legacy line-to-structure list representation."""

    line_structure: Any = None
    legacy_calc_object: Any = None
    fatigue: Any = None
    loads: list[Any] = field(default_factory=list)
    load_combinations: dict[str, Any] = field(default_factory=dict)
    cylinder: Any = None
    legacy_length: int = CYLINDER + 1

    @classmethod
    def from_legacy_bundle(cls, bundle):
        return cls(
            line_structure=None if len(bundle) <= STRUCTURE else bundle[STRUCTURE],
            legacy_calc_object=None if len(bundle) <= LEGACY_CALC_OBJECT else bundle[LEGACY_CALC_OBJECT],
            fatigue=None if len(bundle) <= FATIGUE else bundle[FATIGUE],
            loads=[] if len(bundle) <= LOADS else bundle[LOADS],
            load_combinations={} if len(bundle) <= LOAD_COMBINATIONS else bundle[LOAD_COMBINATIONS],
            cylinder=None if len(bundle) <= CYLINDER else bundle[CYLINDER],
            legacy_length=len(bundle),
        )

    def to_legacy_bundle(self):
        bundle = [
            self.line_structure,
            self.legacy_calc_object,
            self.fatigue,
            self.loads,
            self.load_combinations,
            self.cylinder,
        ]
        return bundle[:self.legacy_length]


def structure(bundle):
    return LineStructureBundle.from_legacy_bundle(bundle).line_structure


def plate(bundle):
    line_structure = structure(bundle)
    return None if line_structure is None else line_structure.Plate


def stiffener(bundle):
    line_structure = structure(bundle)
    return None if line_structure is None else line_structure.Stiffener


def girder(bundle):
    line_structure = structure(bundle)
    return None if line_structure is None else line_structure.Girder


def fatigue(bundle):
    return LineStructureBundle.from_legacy_bundle(bundle).fatigue


def loads(bundle):
    return LineStructureBundle.from_legacy_bundle(bundle).loads


def load_combinations(bundle):
    return LineStructureBundle.from_legacy_bundle(bundle).load_combinations


def cylinder(bundle):
    return LineStructureBundle.from_legacy_bundle(bundle).cylinder


def has_cylinder(bundle):
    return cylinder(bundle) is not None


def has_stiffener(bundle):
    return stiffener(bundle) is not None


def copy_bundle(bundle):
    return [copy.deepcopy(item) if item is not None else None for item in bundle]
