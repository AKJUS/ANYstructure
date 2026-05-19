FLAT_STRUCTURE_DOMAINS = (
    "Flat plate, unstiffened",
    "Flat plate, stiffened",
    "Flat plate, stiffened with girder",
)

CYLINDER_STRUCTURE_DOMAINS = (
    "Unstiffened shell",
    "Unstiffened panel",
    "Longitudinal Stiffened shell",
    "Longitudinal Stiffened panel",
    "Ring Stiffened shell",
    "Ring Stiffened panel",
    "Orthogonally Stiffened shell",
    "Orthogonally Stiffened panel",
)

BUCKLING_CALCULATION_METHODS = (
    "DNV-RP-C201 - prescriptive",
    "ML-CL (PULS based)",
)

BUCKLING_ACCEPTANCE_TYPES = (
    "buckling",
    "ultimate",
)

SUPPORT_TYPES = (
    "Continuous",
    "Sniped",
)

FABRICATION_METHODS = (
    "Fabricated",
    "Cold formed",
)

LIMIT_STATE_TYPES = (
    "ULS",
    "ALS",
)


def assert_choice(value, choices, label):
    assert value in choices, f"{label} must be one of: {list(choices)}"


def mpa_to_pa(value):
    return value * 1e6


def mm_to_m(value):
    return value / 1000


def normalize_bulb_stiffener_type(stiffener_type):
    return "L-bulb" if stiffener_type in ["hp", "HP", "HP-bulb", "bulb"] else stiffener_type
