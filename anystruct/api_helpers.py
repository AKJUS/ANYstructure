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
    "Unstiffened conical shell",
)

BUCKLING_CALCULATION_METHODS = (
    "DNV-RP-C201 - prescriptive",
    "SemiAnalytical S3/U3",
    "ML-Numeric (PULS based)",
)

BUCKLING_ACCEPTANCE_TYPES = (
    "buckling",
    "ultimate",
)

FLAT_PRESSURE_SIDES = (
    "stiffener side",
    "plate side",
    "both sides",
)

FLAT_FABRICATION_METHODS = (
    "welded",
    "cold formed",
    "Fabricated",
    "Cold formed",
)

PULS_PANEL_TYPES = (
    "SP",
    "UP",
)

PULS_BOUNDARY_TYPES = (
    "Int",
    "GL",
    "GT",
    "Integrated",
    "Girder - long",
    "Girder - trans",
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

FLAT_GEOMETRY_IDS = {
    "Flat plate, unstiffened": 10,
    "Flat plate, stiffened": 11,
    "Flat plate, stiffened with girder": 12,
}

CYLINDER_GEOMETRY_IDS = {
    "Unstiffened shell (Force input)": 1,
    "Unstiffened panel (Stress input)": 2,
    "Longitudinal Stiffened shell (Force input)": 3,
    "Longitudinal Stiffened panel (Stress input)": 4,
    "Ring Stiffened shell (Force input)": 5,
    "Ring Stiffened panel (Stress input)": 6,
    "Orthogonally Stiffened shell (Force input)": 7,
    "Orthogonally Stiffened panel (Stress input)": 8,
    "Unstiffened conical shell (Force input)": 9,
}

CYLINDER_GEOMETRY_ALIASES = {
    "Unstiffened conical shell (Stress input)": 9,
}

CYLINDER_STRUCTURE_DOMAINS_WITH_INPUT = tuple(CYLINDER_GEOMETRY_IDS.keys())

GEOMETRY_IDS = {
    **FLAT_GEOMETRY_IDS,
    **CYLINDER_GEOMETRY_IDS,
}

GEOMETRY_DOMAINS = {value: key for key, value in GEOMETRY_IDS.items()}


def normalize_domain_string(calculation_domain):
    return " ".join(str(calculation_domain).split())


def assert_choice(value, choices, label):
    assert value in choices, f"{label} must be one of: {list(choices)}"


def cylinder_input_mode(calculation_domain):
    calculation_domain = normalize_domain_string(calculation_domain)
    if calculation_domain in CYLINDER_GEOMETRY_IDS or calculation_domain in CYLINDER_GEOMETRY_ALIASES:
        return "Stress" if "Stress input" in calculation_domain else "Force"
    assert_choice(calculation_domain, CYLINDER_STRUCTURE_DOMAINS, "calculation_domain")
    return "Stress" if "panel" in calculation_domain.lower() else "Force"


def cylinder_domain_with_input_mode(calculation_domain):
    calculation_domain = normalize_domain_string(calculation_domain)
    input_mode = cylinder_input_mode(calculation_domain)
    return f"{calculation_domain} ({input_mode} input)"


def geometry_id_for_domain(calculation_domain):
    calculation_domain = normalize_domain_string(calculation_domain)

    if calculation_domain in GEOMETRY_IDS:
        return GEOMETRY_IDS[calculation_domain]

    if calculation_domain in CYLINDER_GEOMETRY_ALIASES:
        return CYLINDER_GEOMETRY_ALIASES[calculation_domain]

    if calculation_domain in CYLINDER_STRUCTURE_DOMAINS:
        calculation_domain = cylinder_domain_with_input_mode(calculation_domain)
        return GEOMETRY_IDS[calculation_domain]

    raise KeyError(
        f"Unknown calculation domain: {calculation_domain!r}. "
        f"Expected one of: {list(GEOMETRY_IDS)} or {list(CYLINDER_STRUCTURE_DOMAINS)}"
    )


def domain_for_geometry_id(geometry_id):
    return GEOMETRY_DOMAINS[geometry_id]


def mpa_to_pa(value):
    return value * 1e6


def mm_to_m(value):
    return value / 1000


def normalize_bulb_stiffener_type(stiffener_type):
    return "L-bulb" if stiffener_type in ["hp", "HP", "HP-bulb", "bulb"] else stiffener_type


def normalize_flat_fabrication_method(method):
    if method == "Fabricated":
        return "welded"
    if method == "Cold formed":
        return "cold formed"
    return method


def normalize_puls_boundary(boundary):
    return {
        "Integrated": "Int",
        "Girder - long": "GL",
        "Girder - trans": "GT",
    }.get(boundary, boundary)


def normalize_puls_stiffener_end(support):
    return {
        "C": "Continuous",
        "Cont": "Continuous",
        "Continuous": "Continuous",
        "S": "Sniped",
        "Sniped": "Sniped",
    }.get(support, support)
