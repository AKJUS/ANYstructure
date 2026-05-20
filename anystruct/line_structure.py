"""Named accessors for the existing line-to-structure bundle format."""

import copy


STRUCTURE = 0
LEGACY_CALC_OBJECT = 1
FATIGUE = 2
LOADS = 3
LOAD_COMBINATIONS = 4
CYLINDER = 5


def structure(bundle):
    return bundle[STRUCTURE]


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
    return bundle[FATIGUE]


def loads(bundle):
    return bundle[LOADS]


def load_combinations(bundle):
    return None if len(bundle) <= LOAD_COMBINATIONS else bundle[LOAD_COMBINATIONS]


def cylinder(bundle):
    return None if len(bundle) <= CYLINDER else bundle[CYLINDER]


def has_cylinder(bundle):
    return cylinder(bundle) is not None


def has_stiffener(bundle):
    return stiffener(bundle) is not None


def copy_bundle(bundle):
    return [copy.deepcopy(item) if item is not None else None for item in bundle]
