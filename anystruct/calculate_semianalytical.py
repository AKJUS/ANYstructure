"""Runtime S3/U3 semi-analytical PULS solver for ANYstructure.

This file intentionally contains only the solver, validity diagnostics, and
ANYstructure adapter functions.  Benchmarks, verification reports, CLI commands,
and regression fixtures live in C:/Github/ANYintelligent/calculate_puls.py.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np


EPS = 1.0e-12
SUPPORTED_IN_PLANE_SUPPORTS = {
    "Integrated": "method-a",
    "Girder - long": "method-b-long",
    "Girder - trans": "method-b-trans",
}
SUPPORTED_STIFFENER_TYPES = {"T-bar", "L-bulb", "Angle", "Flatbar"}
SUPPORTED_STIFFENER_BOUNDARIES = {"Cont", "Sniped"}
SUPPORTED_ROTATIONAL_SUPPORTS = {"SS", "CL", "FS", ""}


@dataclass(frozen=True)
class S3PanelInput:
    """Input surface for the regular stiffened S3 panel milestone.

    Geometric dimensions are millimetres and stresses/pressure are MPa.  CSV
    PULS exports use positive in-plane normal stress for compression; the
    helper functions in this module preserve that sign convention.
    """

    length: float
    stiffener_spacing: float
    plate_thickness: float
    stiffener_type: str
    stiffener_boundary: str
    stiffener_height: float
    web_thickness: float
    flange_width: float
    flange_thickness: float
    yield_stress_plate: float
    yield_stress_stiffener: float
    axial_stress: float
    transverse_stress_1: float
    transverse_stress_2: float
    shear_stress: float
    pressure: float
    in_plane_support: str
    elastic_modulus: float = 210000.0
    poisson_ratio: float = 0.3

    @property
    def width(self) -> float:
        return self.stiffener_spacing

    @property
    def mean_transverse_stress(self) -> float:
        return 0.5 * (self.transverse_stress_1 + self.transverse_stress_2)


@dataclass(frozen=True)
class U3PanelInput:
    """Input surface for the regular unstiffened U3 plate milestone.

    Geometric dimensions are millimetres and stresses/pressure are MPa.  The
    ANYstructure/PULS exports use two end values for longitudinal and
    transverse stress; the current U3 Ritz path uses their linear plate-field
    interpolation in yield checks and their mean values in elastic buckling.
    """

    length: float
    width: float
    plate_thickness: float
    yield_stress_plate: float
    axial_stress_1: float
    axial_stress_2: float
    transverse_stress_1: float
    transverse_stress_2: float
    shear_stress: float
    pressure: float
    in_plane_support: str
    rotational_support_1: str = "SS"
    rotational_support_2: str = "SS"
    elastic_modulus: float = 210000.0
    poisson_ratio: float = 0.3

    @property
    def axial_stress(self) -> float:
        return 0.5 * (self.axial_stress_1 + self.axial_stress_2)

    @property
    def mean_transverse_stress(self) -> float:
        return 0.5 * (self.transverse_stress_1 + self.transverse_stress_2)


@dataclass(frozen=True)
class S3SolverConfig:
    """Numerical and covered-domain controls for the reduced S3 solver."""

    longitudinal_modes: tuple[int, ...] = (1, 2, 3)
    transverse_modes: tuple[int, ...] = (1, 2)
    web_longitudinal_modes: tuple[int, ...] = (1, 2, 3)
    web_depth_modes: tuple[int, ...] = (1, 2)
    initial_imperfection_ratio: float = 0.001
    initial_imperfection_floor_mm: float = 0.02
    nonlinear_membrane_factor: float = 0.75
    local_global_coupling_floor: float = 0.25
    local_global_coupling_gain: float = 1.0
    web_shear_interaction_exponent: float = 1.0
    local_plate_web_interaction_exponent: float = 2.0
    flanged_local_plate_restraint_factor: float = 1.10
    use_effective_stiffener_width: bool = False
    sniped_eccentricity_factor: float = 1.2
    torsional_imperfection_scale: float = 1.0
    pressure_local_share: float = 0.0
    pressure_global_share: float = 1.0
    include_pressure_dominated_yield_in_buckling_strength: bool = False
    include_lateral_deformation_in_ultimate_yield: bool = False
    include_global_curvature_in_plate_yield: bool = False
    pressure_dominated_yield_preload_ratio: float = 0.05
    yield_utilization_limit: float = 1.0
    pressure_yield_limit: float = 1.0
    max_load_factor: float = 100.0
    initial_load_step: float = 0.05
    load_step_growth: float = 1.08
    max_load_step: float = 1.5
    min_load_step: float = 1.0e-4
    load_step_cutback: float = 0.5
    max_load_step_cutbacks: int = 12
    newton_max_iterations: int = 40
    newton_tolerance: float = 1.0e-7
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 12.0
    max_plate_slenderness: float = 250.0
    max_web_slenderness: float = 180.0
    max_flange_slenderness: float = 45.0
    max_web_to_flange_ratio: float = 5.0
    hot_spot_grid: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 0.875)
    check_mode_convergence: bool = False
    medium_longitudinal_modes: tuple[int, ...] = (1, 2, 3, 4)
    medium_transverse_modes: tuple[int, ...] = (1, 2, 3)
    high_longitudinal_modes: tuple[int, ...] = (1, 2, 3, 4, 5)
    high_transverse_modes: tuple[int, ...] = (1, 2, 3, 4)
    high_confidence_drift_limit: float = 0.05
    medium_confidence_drift_limit: float = 0.12


@dataclass(frozen=True)
class S3SectionProperties:
    area: float
    centroid_from_plate_midplane: float
    inertia_x: float
    top_distance: float
    bottom_distance: float
    plate_area: float
    stiffener_area: float

    @property
    def section_modulus(self) -> float:
        return self.inertia_x / max(self.top_distance, self.bottom_distance, EPS)

    @property
    def top_section_modulus(self) -> float:
        return self.inertia_x / max(self.top_distance, EPS)

    @property
    def attached_plate_section_modulus(self) -> float:
        return self.inertia_x / max(self.bottom_distance, EPS)


@dataclass(frozen=True)
class OrthotropicStiffness:
    d11: float
    d12: float
    d22: float
    d66: float
    membrane_thickness: float


@dataclass(frozen=True)
class RitzMode:
    family: str
    m: int
    n: int
    kx: float
    ky: float
    linear_stiffness: float
    geometric_stiffness: float
    pressure_force: float
    imperfection: float
    nonlinear_stiffness: float

    @property
    def label(self) -> str:
        return f"{self.family}:{self.m},{self.n}"


@dataclass
class S3Result:
    buckling_usage_factor: float | None
    ultimate_usage_factor: float | None
    valid: bool
    elastic_buckling_usage_factor: float | None = None
    invalid_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    covered_domain_notes: list[str] = field(default_factory=list)
    confidence: str = "low"
    confidence_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "buckling_usage_factor": self.buckling_usage_factor,
            "ultimate_usage_factor": self.ultimate_usage_factor,
            "elastic_buckling_usage_factor": self.elastic_buckling_usage_factor,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "diagnostics": self.diagnostics,
            "covered_domain_notes": list(self.covered_domain_notes),
            "confidence": self.confidence,
            "confidence_reasons": list(self.confidence_reasons),
        }


@dataclass(frozen=True)
class _Rectangle:
    area: float
    centroid: float
    height: float

    @property
    def local_inertia_x(self) -> float:
        return self.area * self.height * self.height / 12.0


def _float_from_row(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value is None:
        raise ValueError(f"Missing numeric value for {key}")
    text = str(value).strip()
    if text == "":
        raise ValueError(f"Missing numeric value for {key}")
    return float(text)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def row_to_s3_input(row: Mapping[str, Any]) -> S3PanelInput:
    """Map a PULS stiffened-panel CSV row into the solver input type."""

    elastic_modulus = _optional_float(row.get("Modulus of elasticity"))
    poisson_ratio = _optional_float(row.get("Poisson's ratio"))
    if poisson_ratio is None:
        poisson_ratio = _optional_float(row.get("Poisson ratio"))
    return S3PanelInput(
        length=_float_from_row(row, "Length of panel"),
        stiffener_spacing=_float_from_row(row, "Stiffener spacing"),
        plate_thickness=_float_from_row(row, "Plate thick."),
        stiffener_type=str(row.get("Stiffener type", "")).strip(),
        stiffener_boundary=str(row.get("Stiffener boundary", "")).strip(),
        stiffener_height=_float_from_row(row, "Stiff. Height"),
        web_thickness=_float_from_row(row, "Web thick."),
        flange_width=_float_from_row(row, "Flange width"),
        flange_thickness=_float_from_row(row, "Flange thick."),
        yield_stress_plate=_float_from_row(row, "Yield stress plate"),
        yield_stress_stiffener=_float_from_row(row, "Yield stress stiffener"),
        axial_stress=_float_from_row(row, "Axial stress"),
        transverse_stress_1=_float_from_row(row, "Trans. stress 1"),
        transverse_stress_2=_float_from_row(row, "Trans. stress 2"),
        shear_stress=_float_from_row(row, "Shear stress"),
        pressure=_float_from_row(row, "Pressure (fixed)"),
        in_plane_support=str(row.get("In-plane support", "")).strip(),
        elastic_modulus=(
            elastic_modulus
            if elastic_modulus is not None
            else S3PanelInput.__dataclass_fields__["elastic_modulus"].default
        ),
        poisson_ratio=(
            poisson_ratio
            if poisson_ratio is not None
            else S3PanelInput.__dataclass_fields__["poisson_ratio"].default
        ),
    )


def row_to_u3_input(row: Mapping[str, Any]) -> U3PanelInput:
    """Map a PULS unstiffened-panel row/export into the solver input type."""

    elastic_modulus = _optional_float(row.get("Modulus of elasticity"))
    poisson_ratio = _optional_float(row.get("Poisson's ratio"))
    if poisson_ratio is None:
        poisson_ratio = _optional_float(row.get("Poisson ratio"))
    axial_1 = _float_from_row(row, "Axial stress")
    axial_2 = _optional_float(row.get("Axial stress 2"))
    transverse_1 = _optional_float(row.get("Trans. stress 1"))
    if transverse_1 is None:
        transverse_1 = _optional_float(row.get("Trans. Stress"))
    if transverse_1 is None:
        transverse_1 = _optional_float(row.get("Trans. stress"))
    if transverse_1 is None:
        raise ValueError("Missing numeric value for Trans. stress 1")
    transverse_2 = _optional_float(row.get("Trans. stress 2"))
    if transverse_2 is None:
        transverse_2 = _optional_float(row.get("Trans. Stress 2"))
    if transverse_2 is None:
        transverse_2 = transverse_1
    return U3PanelInput(
        length=_float_from_row(row, "Plate length"),
        width=_float_from_row(row, "Plate width"),
        plate_thickness=_float_from_row(row, "Plate thick."),
        yield_stress_plate=_float_from_row(row, "Yield stress plate"),
        axial_stress_1=axial_1,
        axial_stress_2=axial_1 if axial_2 is None else axial_2,
        transverse_stress_1=transverse_1,
        transverse_stress_2=transverse_2,
        shear_stress=_float_from_row(row, "Shear stress"),
        pressure=_float_from_row(row, "Pressure (fixed)"),
        in_plane_support=str(row.get("In-plane support", "")).strip(),
        rotational_support_1=str(row.get("Rotational support", "SS") or "SS").strip(),
        rotational_support_2=str(row.get("Rotational support 2", "SS") or "SS").strip(),
        elastic_modulus=(
            elastic_modulus
            if elastic_modulus is not None
            else U3PanelInput.__dataclass_fields__["elastic_modulus"].default
        ),
        poisson_ratio=(
            poisson_ratio
            if poisson_ratio is not None
            else U3PanelInput.__dataclass_fields__["poisson_ratio"].default
        ),
    )


_MISSING = object()


def _ship_section_value(section: Mapping[str, Any], key: str, default: Any = _MISSING) -> Any:
    if key in section:
        value = section[key]
    elif default is not _MISSING:
        value = default
    else:
        value = section[key]
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def ship_section_record_to_csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one ANYstructure ship-section PULS result record to CSV-like S3 fields."""

    plate = record["Plate geometry"]
    stiffener = record["Primary stiffeners"]
    material = record["Material"]
    loads = record["Applied loads"]
    support = record.get("Bound cond.", {})
    buckling = record.get("Buckling strength", {})
    ultimate = record.get("Ultimate capacity", {})
    transverse_stress = _ship_section_value(loads, "Trans. stress")

    return {
        "_ship_line": record.get("Identification", ""),
        "Length of panel": _ship_section_value(plate, "Length of panel"),
        "Stiffener spacing": _ship_section_value(plate, "Stiffener spacing"),
        "Plate thick.": _ship_section_value(plate, "Plate thick."),
        "Stiffener type": _ship_section_value(stiffener, "Stiffener type"),
        "Stiffener boundary": _ship_section_value(stiffener, "Stiffener boundary"),
        "Stiff. Height": _ship_section_value(stiffener, "Stiff. Height"),
        "Web thick.": _ship_section_value(stiffener, "Web thick."),
        "Flange width": _ship_section_value(stiffener, "Flange width"),
        "Flange thick.": _ship_section_value(stiffener, "Flange thick."),
        "Yield stress plate": _ship_section_value(material, "Yield stress plate"),
        "Yield stress stiffener": _ship_section_value(material, "Yield stress stiffener"),
        "Modulus of elasticity": _ship_section_value(material, "Modulus of elasticity"),
        "Poisson's ratio": _ship_section_value(material, "Poisson's ratio"),
        "Axial stress": _ship_section_value(loads, "Axial stress"),
        "Trans. stress 1": transverse_stress,
        "Trans. stress 2": _ship_section_value(loads, "Trans. stress 2", transverse_stress),
        "Shear stress": _ship_section_value(loads, "Shear stress"),
        "Pressure (fixed)": _ship_section_value(loads, "Pressure (fixed)"),
        "In-plane support": _ship_section_value(support, "In-plane support") if support else "",
        "Buckling Actual usage Factor inc NaN": (
            _ship_section_value(buckling, "Actual usage Factor") if buckling else ""
        ),
        "Ultimate Actual usage Factor inc NaN": (
            _ship_section_value(ultimate, "Actual usage Factor") if ultimate else ""
        ),
        "output cl str buc": _ship_section_value(buckling, "Status") if buckling else "",
    }


def ship_section_record_to_u3_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one ANYstructure ship-section PULS result record to CSV-like U3 fields."""

    plate = record["Geometry"]
    material = record["Material"]
    loads = record["Applied loads"]
    support = record.get("Boundary conditions", record.get("Bound cond.", {}))
    buckling = record.get("Buckling strength", {})
    ultimate = record.get("Ultimate capacity", {})
    transverse_stress = _ship_section_value(loads, "Trans. Stress", _ship_section_value(loads, "Trans. stress", 0.0))

    return {
        "_ship_line": record.get("Identification", ""),
        "Panel family": "U3",
        "Plate length": _ship_section_value(plate, "Plate length"),
        "Plate width": _ship_section_value(plate, "Plate width"),
        "Plate thick.": _ship_section_value(plate, "Plate thick."),
        "Yield stress plate": _ship_section_value(material, "Yield st. plate", _ship_section_value(material, "Yield stress plate", "")),
        "Modulus of elasticity": _ship_section_value(material, "Modulus of elasticity"),
        "Poisson's ratio": _ship_section_value(material, "Poisson's ratio"),
        "Axial stress": _ship_section_value(loads, "Axial stress"),
        "Axial stress 2": _ship_section_value(loads, "Axial stress 2", _ship_section_value(loads, "Axial stress")),
        "Trans. stress 1": transverse_stress,
        "Trans. stress 2": _ship_section_value(loads, "Trans. Stress 2", transverse_stress),
        "Shear stress": _ship_section_value(loads, "Shear stress"),
        "Pressure (fixed)": _ship_section_value(loads, "Pressure (fixed)"),
        "In-plane support": _ship_section_value(support, "In-plane support") if support else "",
        "Rotational support": _ship_section_value(support, "Rotational support", "SS") if support else "SS",
        "Rotational support 2": _ship_section_value(support, "Rotational support 2", "SS") if support else "SS",
        "Buckling Actual usage Factor inc NaN": (
            _ship_section_value(buckling, "Actual usage Factor") if buckling else ""
        ),
        "Ultimate Actual usage Factor inc NaN": (
            _ship_section_value(ultimate, "Actual usage Factor") if ultimate else ""
        ),
        "output cl str buc": _ship_section_value(buckling, "Status") if buckling else "",
    }


def _anystructure_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _anystructure_stiffener_type(value: Any) -> str:
    return {
        "T": "T-bar",
        "T-bar": "T-bar",
        "L": "Angle",
        "Angle": "Angle",
        "L-bulb": "L-bulb",
        "FB": "Flatbar",
        "F": "Flatbar",
        "Flatbar": "Flatbar",
    }.get(str(value), str(value))


def _anystructure_stiffener_boundary(value: Any) -> str:
    return {
        "C": "Cont",
        "Cont": "Cont",
        "Continuous": "Cont",
        "S": "Sniped",
        "Sniped": "Sniped",
    }.get(str(value), str(value))


def _anystructure_in_plane_support(value: Any) -> str:
    return {
        "Int": "Integrated",
        "Integrated": "Integrated",
        "GL": "Girder - long",
        "Girder - long": "Girder - long",
        "GT": "Girder - trans",
        "Girder - trans": "Girder - trans",
    }.get(str(value), str(value))


def _anystructure_rotational_supports(value: Any) -> tuple[str, str]:
    text = str(value or "SSSS").strip().upper().replace("-", "")
    if text in {"CCCC", "CLCL", "CC", "CL"}:
        return "CL", "CL"
    if text in {"FSFS", "FFFF", "FS"}:
        return "FS", "FS"
    return "SS", "SS"


def _anystructure_selected_method(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "buckling"}:
        return "buckling"
    if text in {"2", "ultimate"}:
        return "ultimate"
    return text


def _anystructure_material_factor(all_structure: Any) -> float | None:
    for candidate in (
        getattr(all_structure, "Plate", None),
        getattr(all_structure, "Stiffener", None),
        all_structure,
    ):
        if candidate is None:
            continue
        for name in ("mat_factor", "_mat_factor"):
            try:
                value = getattr(candidate, name)
            except Exception:
                continue
            parsed = _optional_float(value)
            if parsed is not None and parsed > EPS:
                return parsed
    return None


def _anystructure_axial_design_stress(sigma_x1: float, sigma_x2: float) -> float:
    if sigma_x1 * sigma_x2 >= 0:
        return sigma_x1 if abs(sigma_x1) > abs(sigma_x2) else sigma_x2
    return max(sigma_x1, sigma_x2)


def anystructure_panel_input(calc_object: Any, lat_press: float = 0.0) -> S3PanelInput | U3PanelInput | None:
    """Build an S3/U3 input from an ANYstructure AllStructure-like object.

    This is intentionally duck-typed so ANYstructure can call it without making
    this repository import ANYstructure.  The `lat_press` convention follows
    ANYstructure optimization: kPa in, MPa on the solver input surface.
    """

    all_structure = calc_object[0] if isinstance(calc_object, (list, tuple)) else calc_object
    plate = getattr(all_structure, "Plate", None)
    stiffener = getattr(all_structure, "Stiffener", None)
    if plate is None:
        return None

    sp_or_up = str(plate.get_puls_sp_or_up()).strip().upper()
    pressure_mpa = _anystructure_float(lat_press) / 1000.0
    elastic_modulus = _anystructure_float(getattr(all_structure, "E", None), 210000.0e6) / 1.0e6
    poisson_ratio = _anystructure_float(getattr(all_structure, "v", None), 0.3)

    if sp_or_up == "SP" and stiffener is not None:
        puls_boundary = stiffener.get_puls_boundary()
        sigxd = _anystructure_axial_design_stress(stiffener.sigma_x1, stiffener.sigma_x2)
        return S3PanelInput(
            length=stiffener.span * 1000.0,
            stiffener_spacing=stiffener.spacing,
            plate_thickness=stiffener.t,
            stiffener_type=_anystructure_stiffener_type(stiffener.get_stiffener_type()),
            stiffener_boundary=_anystructure_stiffener_boundary(stiffener.get_puls_stf_end()),
            stiffener_height=stiffener.hw,
            web_thickness=stiffener.tw,
            flange_width=stiffener.b,
            flange_thickness=stiffener.tf,
            yield_stress_plate=stiffener.mat_yield / 1.0e6,
            yield_stress_stiffener=stiffener.mat_yield / 1.0e6,
            axial_stress=0.0 if puls_boundary == "GT" else sigxd,
            transverse_stress_1=0.0 if puls_boundary == "GL" else stiffener.sigma_y1,
            transverse_stress_2=0.0 if puls_boundary == "GL" else stiffener.sigma_y2,
            shear_stress=stiffener.tau_xy,
            pressure=pressure_mpa,
            in_plane_support=_anystructure_in_plane_support(puls_boundary),
            elastic_modulus=elastic_modulus,
            poisson_ratio=poisson_ratio,
        )

    up_boundary = plate.get_puls_up_boundary() if hasattr(plate, "get_puls_up_boundary") else "SSSS"
    rotational_1, rotational_2 = _anystructure_rotational_supports(up_boundary)
    puls_boundary = plate.get_puls_boundary() if hasattr(plate, "get_puls_boundary") else "GL"
    return U3PanelInput(
        length=plate.span * 1000.0,
        width=plate.spacing,
        plate_thickness=plate.t,
        yield_stress_plate=plate.mat_yield / 1.0e6,
        axial_stress_1=plate.sigma_x1,
        axial_stress_2=plate.sigma_x2,
        transverse_stress_1=plate.sigma_y1,
        transverse_stress_2=plate.sigma_y2,
        shear_stress=plate.tau_xy,
        pressure=pressure_mpa,
        in_plane_support=_anystructure_in_plane_support(puls_boundary),
        rotational_support_1=rotational_1,
        rotational_support_2=rotational_2,
        elastic_modulus=elastic_modulus,
        poisson_ratio=poisson_ratio,
    )


def solve_anystructure_panel(
    calc_object: Any,
    lat_press: float = 0.0,
    config: S3SolverConfig | None = None,
) -> dict[str, Any]:
    """Return an ANYstructure-friendly SemiAnalytical result dictionary."""

    config = config or S3SolverConfig()
    all_structure = calc_object[0] if isinstance(calc_object, (list, tuple)) else calc_object
    material_factor = _anystructure_material_factor(all_structure)
    acceptance_limit = None if material_factor is None else 1.0 / material_factor
    panel = anystructure_panel_input(calc_object, lat_press)
    if panel is None:
        return {
            "panel_family": None,
            "buckling_usage_factor": None,
            "ultimate_usage_factor": None,
            "material_factor": material_factor,
            "acceptance_limit": acceptance_limit,
            "valid": False,
            "available": False,
            "valid_prediction": 0,
            "valid_label": "SemiAnalytical S3/U3 unsupported or invalid",
            "invalid_reason": "unsupported-anystructure-input",
            "confidence": "low",
            "confidence_reasons": ["unsupported-anystructure-input"],
            "diagnostics": {},
        }

    panel_family = "S3" if isinstance(panel, S3PanelInput) else "U3"
    solved = solve_s3_panel(panel, config) if panel_family == "S3" else solve_u3_panel(panel, config)
    valid_prediction = (
        solved.valid
        and solved.buckling_usage_factor is not None
        and solved.ultimate_usage_factor is not None
    )
    valid_label = (
        f"valid SemiAnalytical {panel_family} UF predicted ({solved.confidence} confidence)"
        if valid_prediction
        else f"SemiAnalytical {panel_family} unsupported or invalid"
    )
    return {
        "panel_family": panel_family,
        "buckling_usage_factor": solved.buckling_usage_factor,
        "ultimate_usage_factor": solved.ultimate_usage_factor,
        "elastic_buckling_usage_factor": solved.elastic_buckling_usage_factor,
        "material_factor": material_factor,
        "acceptance_limit": acceptance_limit,
        "valid": solved.valid,
        "available": bool(valid_prediction),
        "valid_prediction": 1 if valid_prediction else 0,
        "valid_label": valid_label,
        "invalid_reason": solved.invalid_reason,
        "confidence": solved.confidence,
        "confidence_reasons": list(solved.confidence_reasons),
        "diagnostics": solved.diagnostics,
        "result": solved.to_dict(),
    }


def predict_anystructure_uf(
    calc_object: Any,
    lat_press: float = 0.0,
    config: S3SolverConfig | None = None,
) -> np.ndarray:
    """Return legacy ANYstructure vector [buckling UF, ultimate UF, valid].

    The factors are intentionally returned un-factored.  ANYstructure compares
    them against a separate PULS acceptance limit, typically 1 / material factor.
    """

    result = solve_anystructure_panel(calc_object, lat_press, config)
    if result["available"]:
        return np.array(
            [
                float(result["buckling_usage_factor"]),
                float(result["ultimate_usage_factor"]),
                1.0,
            ],
            dtype=float,
        )
    return np.array([float("inf"), float("inf"), 0.0], dtype=float)


def predict_anystructure_uf_with_acceptance(
    calc_object: Any,
    lat_press: float = 0.0,
    config: S3SolverConfig | None = None,
    default_acceptance: float = 0.87,
) -> np.ndarray:
    """Return [buckling UF, ultimate UF, valid, acceptance limit] for ANYstructure."""

    result = solve_anystructure_panel(calc_object, lat_press, config)
    acceptance = _optional_float(result.get("acceptance_limit"))
    if acceptance is None:
        acceptance = float(default_acceptance)
    vector = np.array([float("inf"), float("inf"), 0.0, acceptance], dtype=float)
    if result["available"]:
        vector[0] = float(result["buckling_usage_factor"])
        vector[1] = float(result["ultimate_usage_factor"])
        vector[2] = 1.0
    return vector


def normalized_load_components(panel: S3PanelInput) -> dict[str, float]:
    """Return stress resultants and destabilizing load components.

    Positive values in the `compression_*` fields are compression resultants
    per unit length.  Negative normal stresses still enter the yield checks
    with their signed values, but are not treated as elastic buckling drivers.
    """

    plate_t = panel.plate_thickness
    return {
        "signed_axial_stress": panel.axial_stress,
        "signed_transverse_stress": panel.mean_transverse_stress,
        "signed_shear_stress": panel.shear_stress,
        "compression_nx": max(panel.axial_stress, 0.0) * plate_t,
        "compression_ny": max(panel.mean_transverse_stress, 0.0) * plate_t,
        "shear_nxy": abs(panel.shear_stress) * plate_t,
        "pressure": max(panel.pressure, 0.0),
    }


def build_section_properties(
    panel: S3PanelInput,
    attached_plate_width: float | None = None,
    stiffener_web_thickness: float | None = None,
) -> S3SectionProperties:
    """Return strip section properties about the plate/stiffener axis."""

    plate_t = panel.plate_thickness
    web_h = panel.stiffener_height
    web_thickness = panel.web_thickness if stiffener_web_thickness is None else stiffener_web_thickness
    plate_width = panel.width if attached_plate_width is None else min(attached_plate_width, panel.width)
    rectangles = [
        _Rectangle(
            area=max(plate_width, EPS) * plate_t,
            centroid=0.0,
            height=plate_t,
        ),
        _Rectangle(
            area=web_h * max(web_thickness, EPS),
            centroid=0.5 * plate_t + 0.5 * web_h,
            height=web_h,
        ),
    ]

    if panel.stiffener_type != "Flatbar" and panel.flange_width > 0.0 and panel.flange_thickness > 0.0:
        rectangles.append(
            _Rectangle(
                area=panel.flange_width * panel.flange_thickness,
                centroid=0.5 * plate_t + web_h + 0.5 * panel.flange_thickness,
                height=panel.flange_thickness,
            )
        )

    total_area = sum(rectangle.area for rectangle in rectangles)
    centroid = sum(rectangle.area * rectangle.centroid for rectangle in rectangles) / max(total_area, EPS)
    inertia = sum(
        rectangle.local_inertia_x + rectangle.area * (rectangle.centroid - centroid) ** 2
        for rectangle in rectangles
    )

    top_coordinate = max(
        0.5 * plate_t,
        0.5 * plate_t + web_h,
        0.5 * plate_t + web_h + max(panel.flange_thickness, 0.0),
    )
    bottom_coordinate = -0.5 * plate_t
    plate_area = rectangles[0].area
    return S3SectionProperties(
        area=total_area,
        centroid_from_plate_midplane=centroid,
        inertia_x=inertia,
        top_distance=max(top_coordinate - centroid, EPS),
        bottom_distance=max(centroid - bottom_coordinate, EPS),
        plate_area=plate_area,
        stiffener_area=max(total_area - plate_area, 0.0),
    )


def effective_stiffener_plate_width(panel: S3PanelInput) -> dict[str, float]:
    """Return the shear-lag effective width used by stiffener checks.

    IACS S35 and the matching DNV stiffener text reduce attached plating by an
    effective-width coefficient based on stiffener effective length.  The S3 CSV
    surface has a regular strip width but not separate plate reduction factors
    for the two sides of a stiffener, so this helper applies the length-based
    coefficient only and records that width in diagnostics.
    """

    effective_length = (
        panel.length / math.sqrt(3.0)
        if panel.stiffener_boundary == "Cont"
        else panel.length
    )
    length_ratio = effective_length / max(panel.width, EPS)
    if length_ratio >= 1.0:
        coefficient = 1.12 / (1.0 + 1.75 * length_ratio**1.6)
        coefficient = min(coefficient, 1.0)
    else:
        coefficient = 0.407 * length_ratio
    coefficient = min(1.0, max(coefficient, EPS))
    return {
        "width": coefficient * panel.width,
        "coefficient": coefficient,
        "effective_length": effective_length,
        "length_ratio": length_ratio,
    }


def effective_flatbar_web_thickness(
    panel: S3PanelInput,
    attached_plate_width: float,
) -> dict[str, float]:
    """Return the effective flat-bar web thickness for stiffener checks."""

    if panel.stiffener_type != "Flatbar":
        return {
            "gross_thickness": panel.web_thickness,
            "effective_thickness": panel.web_thickness,
            "reduction_factor": 1.0,
            "attached_plate_width": attached_plate_width,
        }

    width_ratio = min(max(attached_plate_width / max(panel.width, EPS), 0.0), 1.0)
    reduction_factor = (
        1.0
        - 2.0
        * math.pi**2
        / 3.0
        * (panel.stiffener_height / max(panel.width, EPS)) ** 2
        * (1.0 - width_ratio)
    )
    reduction_factor = min(1.0, max(reduction_factor, EPS))
    return {
        "gross_thickness": panel.web_thickness,
        "effective_thickness": reduction_factor * panel.web_thickness,
        "reduction_factor": reduction_factor,
        "attached_plate_width": attached_plate_width,
    }


def build_effective_stiffener_section(
    panel: S3PanelInput,
    config: S3SolverConfig,
) -> tuple[S3SectionProperties, dict[str, Any]]:
    effective_width = effective_stiffener_plate_width(panel)
    attached_width = effective_width["width"] if config.use_effective_stiffener_width else panel.width
    flatbar_web = effective_flatbar_web_thickness(panel, attached_width)
    if not config.use_effective_stiffener_width:
        effective_width = {**effective_width, "applied_width": panel.width}
    else:
        effective_width = {**effective_width, "applied_width": attached_width}
    effective_width = {**effective_width, "flatbar_web_thickness": flatbar_web}
    return (
        build_section_properties(
            panel,
            attached_plate_width=attached_width,
            stiffener_web_thickness=flatbar_web["effective_thickness"],
        ),
        effective_width,
    )


def _plate_bending_rigidity(panel: S3PanelInput) -> float:
    return panel.elastic_modulus * panel.plate_thickness**3 / (
        12.0 * (1.0 - panel.poisson_ratio**2)
    )


def build_orthotropic_stiffness(
    panel: S3PanelInput,
    section: S3SectionProperties,
    family: str,
    config: S3SolverConfig | None = None,
) -> OrthotropicStiffness:
    """Build local-plate or distributed stiffened-strip stiffness terms."""

    config = config or S3SolverConfig()
    plate_d = _plate_bending_rigidity(panel)
    if family == "local":
        restraint_factor = (
            max(config.flanged_local_plate_restraint_factor, EPS)
            if panel.stiffener_type != "Flatbar"
            else 1.0
        )
        d11 = plate_d * restraint_factor
        d22 = plate_d * restraint_factor
        membrane_thickness = panel.plate_thickness
    elif family == "global":
        d11 = max(panel.elastic_modulus * section.inertia_x / panel.width, plate_d)
        d22 = plate_d
        membrane_thickness = max(section.area / panel.width, panel.plate_thickness)
    else:
        raise ValueError(f"Unknown stiffness family: {family}")

    d12 = panel.poisson_ratio * math.sqrt(d11 * d22)
    d66 = 0.5 * (1.0 - panel.poisson_ratio) * math.sqrt(d11 * d22)
    return OrthotropicStiffness(
        d11=d11,
        d12=d12,
        d22=d22,
        d66=d66,
        membrane_thickness=membrane_thickness,
    )


def _with_longitudinal_stiffness_scale(
    stiffness: OrthotropicStiffness,
    scale: float,
) -> OrthotropicStiffness:
    scale = max(scale, EPS)
    d11 = stiffness.d11 * scale
    d22 = stiffness.d22
    return OrthotropicStiffness(
        d11=d11,
        d12=math.sqrt(scale) * stiffness.d12,
        d22=d22,
        d66=math.sqrt(scale) * stiffness.d66,
        membrane_thickness=stiffness.membrane_thickness,
    )


def _support_membrane_factor(panel: S3PanelInput) -> float:
    if panel.in_plane_support == "Integrated":
        return 1.0
    if panel.in_plane_support == "Girder - long":
        return 0.90
    if panel.in_plane_support == "Girder - trans":
        return 0.90
    return 1.0


def _load_family(panel: S3PanelInput | U3PanelInput) -> str:
    components = []
    if panel.axial_stress > EPS:
        components.append("axial-compression")
    elif panel.axial_stress < -EPS:
        components.append("axial-tension")
    if panel.mean_transverse_stress > EPS:
        components.append("transverse-compression")
    elif panel.mean_transverse_stress < -EPS:
        components.append("transverse-tension")
    if abs(panel.shear_stress) > EPS:
        components.append("shear")
    if panel.pressure > EPS:
        components.append("pressure")
    return "+".join(components) if components else "zero-variable-load"


def _validation_domain(
    panel: S3PanelInput | U3PanelInput,
    config: S3SolverConfig,
    panel_family: str,
    reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    width = panel.width
    plate_slenderness = width / panel.plate_thickness if panel.plate_thickness > EPS else None
    aspect_ratio = panel.length / width if width > EPS else None
    domain: dict[str, Any] = {
        "panel_family": panel_family,
        "aspect_ratio": aspect_ratio,
        "aspect_ratio_limits": [config.min_aspect_ratio, config.max_aspect_ratio],
        "plate_slenderness": plate_slenderness,
        "max_plate_slenderness": config.max_plate_slenderness,
        "in_plane_support": panel.in_plane_support,
        "support_model": SUPPORTED_IN_PLANE_SUPPORTS.get(panel.in_plane_support),
        "pressure": panel.pressure,
        "pressure_category": "nonzero" if panel.pressure > EPS else "zero",
        "load_family": _load_family(panel),
        "reasons": list(reasons or ()),
    }
    if isinstance(panel, S3PanelInput):
        domain.update(
            {
                "stiffener_type": panel.stiffener_type,
                "stiffener_boundary": panel.stiffener_boundary,
                "web_slenderness": (
                    panel.stiffener_height / panel.web_thickness
                    if panel.web_thickness > EPS
                    else None
                ),
                "max_web_slenderness": config.max_web_slenderness,
                "flange_slenderness": (
                    panel.flange_width / panel.flange_thickness
                    if panel.flange_thickness > EPS
                    else None
                ),
                "max_flange_slenderness": config.max_flange_slenderness,
                "web_to_flange_ratio": (
                    panel.stiffener_height / max(panel.flange_width, panel.web_thickness)
                    if max(panel.flange_width, panel.web_thickness) > EPS
                    else None
                ),
                "max_web_to_flange_ratio": config.max_web_to_flange_ratio,
            }
        )
    else:
        domain.update(
            {
                "rotational_support": {
                    "x_edges": panel.rotational_support_1,
                    "y_edges": panel.rotational_support_2,
                },
            }
        )
    return domain


def collect_s3_validation_reasons(panel: S3PanelInput, config: S3SolverConfig) -> list[str]:
    """Return all explicit S3 domain/validity reasons in stable first-reason order."""

    numeric_fields = {
        "length": panel.length,
        "stiffener_spacing": panel.stiffener_spacing,
        "plate_thickness": panel.plate_thickness,
        "stiffener_height": panel.stiffener_height,
        "web_thickness": panel.web_thickness,
        "flange_width": panel.flange_width,
        "flange_thickness": panel.flange_thickness,
        "yield_stress_plate": panel.yield_stress_plate,
        "yield_stress_stiffener": panel.yield_stress_stiffener,
        "elastic_modulus": panel.elastic_modulus,
        "poisson_ratio": panel.poisson_ratio,
        "axial_stress": panel.axial_stress,
        "transverse_stress_1": panel.transverse_stress_1,
        "transverse_stress_2": panel.transverse_stress_2,
        "shear_stress": panel.shear_stress,
        "pressure": panel.pressure,
    }
    reasons: list[str] = []
    if any(not math.isfinite(value) for value in numeric_fields.values()):
        return ["non-finite-input"]
    if panel.poisson_ratio < 0.0 or panel.poisson_ratio >= 0.5:
        reasons.append("unsupported-material")
    positive_fields = {
        "length": panel.length,
        "stiffener_spacing": panel.stiffener_spacing,
        "plate_thickness": panel.plate_thickness,
        "stiffener_height": panel.stiffener_height,
        "web_thickness": panel.web_thickness,
        "yield_stress_plate": panel.yield_stress_plate,
        "yield_stress_stiffener": panel.yield_stress_stiffener,
        "elastic_modulus": panel.elastic_modulus,
    }
    if any(value <= 0.0 for value in positive_fields.values()):
        reasons.append("non-positive-geometry-or-material")
    if panel.pressure < 0.0:
        reasons.append("negative-pressure-unsupported")
    if panel.stiffener_type not in SUPPORTED_STIFFENER_TYPES:
        reasons.append("unsupported-stiffener-type")
    if panel.stiffener_boundary not in SUPPORTED_STIFFENER_BOUNDARIES:
        reasons.append("unsupported-stiffener-boundary")
    if panel.in_plane_support not in SUPPORTED_IN_PLANE_SUPPORTS:
        reasons.append("unsupported-in-plane-support")
    if panel.width > EPS:
        aspect_ratio = panel.length / panel.width
        if aspect_ratio < config.min_aspect_ratio or aspect_ratio > config.max_aspect_ratio:
            reasons.append("aspect ratio")
    if panel.plate_thickness > EPS and panel.width / panel.plate_thickness > config.max_plate_slenderness:
        reasons.append("slenderness")
    if panel.web_thickness > EPS and panel.stiffener_height / panel.web_thickness > config.max_web_slenderness:
        reasons.append("slenderness")
    if panel.stiffener_type != "Flatbar":
        if panel.flange_width <= 0.0 or panel.flange_thickness <= 0.0:
            reasons.append("web-flange-ratio")
        elif panel.flange_width / panel.flange_thickness > config.max_flange_slenderness:
            reasons.append("web-flange-ratio")
        if (
            panel.stiffener_type in {"Angle", "T-bar"}
            and panel.stiffener_height / max(panel.flange_width, panel.web_thickness, EPS)
            > config.max_web_to_flange_ratio
        ):
            reasons.append("web-flange-ratio")
    return list(dict.fromkeys(reasons))


def collect_u3_validation_reasons(panel: U3PanelInput, config: S3SolverConfig) -> list[str]:
    """Return all explicit U3 domain/validity reasons in stable first-reason order."""

    numeric_fields = {
        "length": panel.length,
        "width": panel.width,
        "plate_thickness": panel.plate_thickness,
        "yield_stress_plate": panel.yield_stress_plate,
        "elastic_modulus": panel.elastic_modulus,
        "poisson_ratio": panel.poisson_ratio,
        "axial_stress_1": panel.axial_stress_1,
        "axial_stress_2": panel.axial_stress_2,
        "transverse_stress_1": panel.transverse_stress_1,
        "transverse_stress_2": panel.transverse_stress_2,
        "shear_stress": panel.shear_stress,
        "pressure": panel.pressure,
    }
    reasons: list[str] = []
    if any(not math.isfinite(value) for value in numeric_fields.values()):
        return ["non-finite-input"]
    if panel.poisson_ratio < 0.0 or panel.poisson_ratio >= 0.5:
        reasons.append("unsupported-material")
    positive_fields = {
        "length": panel.length,
        "width": panel.width,
        "plate_thickness": panel.plate_thickness,
        "yield_stress_plate": panel.yield_stress_plate,
        "elastic_modulus": panel.elastic_modulus,
    }
    if any(value <= 0.0 for value in positive_fields.values()):
        reasons.append("non-positive-geometry-or-material")
    if panel.pressure < 0.0:
        reasons.append("negative-pressure-unsupported")
    if panel.in_plane_support not in SUPPORTED_IN_PLANE_SUPPORTS:
        reasons.append("unsupported-in-plane-support")
    if panel.rotational_support_1 not in SUPPORTED_ROTATIONAL_SUPPORTS:
        reasons.append("unsupported-rotational-support")
    if panel.rotational_support_2 not in SUPPORTED_ROTATIONAL_SUPPORTS:
        reasons.append("unsupported-rotational-support")
    if panel.width > EPS:
        aspect_ratio = panel.length / panel.width
        if aspect_ratio < config.min_aspect_ratio or aspect_ratio > config.max_aspect_ratio:
            reasons.append("aspect ratio")
    if panel.plate_thickness > EPS and panel.width / panel.plate_thickness > config.max_plate_slenderness:
        reasons.append("slenderness")
    return list(dict.fromkeys(reasons))


def validate_s3_input(panel: S3PanelInput, config: S3SolverConfig) -> str | None:
    """Validate the explicit covered domain before solving."""

    reasons = collect_s3_validation_reasons(panel, config)
    return reasons[0] if reasons else None


def validate_u3_input(panel: U3PanelInput, config: S3SolverConfig) -> str | None:
    """Validate the explicit U3 covered domain before solving."""

    reasons = collect_u3_validation_reasons(panel, config)
    return reasons[0] if reasons else None


def _pressure_generalized_force(panel: S3PanelInput, m: int, n: int, share: float) -> float:
    if panel.pressure <= 0.0 or m % 2 == 0 or n % 2 == 0:
        return 0.0
    return share * panel.pressure * 4.0 * panel.length * panel.width / (m * n * math.pi**2)


def _mode_linear_terms(
    panel: S3PanelInput,
    stiffness: OrthotropicStiffness,
    config: S3SolverConfig,
    m: int,
    n: int,
) -> tuple[float, float, float, float]:
    kx = m * math.pi / panel.length
    ky = n * math.pi / panel.width
    area_factor = panel.length * panel.width / 4.0
    bending = (
        stiffness.d11 * kx**4
        + 2.0 * (stiffness.d12 + 2.0 * stiffness.d66) * kx * kx * ky * ky
        + stiffness.d22 * ky**4
    )

    loads = normalized_load_components(panel)
    geometric = (
        loads["compression_nx"] * kx * kx
        + loads["compression_ny"] * ky * ky
    )
    linear_stiffness = area_factor * bending
    geometric_stiffness = area_factor * geometric
    wave_norm = kx * kx + ky * ky
    nonlinear_stiffness = (
        config.nonlinear_membrane_factor
        * _support_membrane_factor(panel)
        * panel.elastic_modulus
        * stiffness.membrane_thickness
        * panel.length
        * panel.width
        * wave_norm
        * wave_norm
        / 16.0
    )
    return kx, ky, linear_stiffness, max(geometric_stiffness, 0.0), nonlinear_stiffness


def _sin_cos_integral(sine_mode: int, cosine_mode: int, length: float) -> float:
    """Return the Fourier integral int_0^L sin(p*pi*x/L) cos(q*pi*x/L) dx."""

    def harmonic_integral(harmonic: int) -> float:
        if harmonic == 0 or harmonic % 2 == 0:
            return 0.0
        return 2.0 / harmonic

    return (
        0.5
        * length
        / math.pi
        * (
            harmonic_integral(sine_mode + cosine_mode)
            + harmonic_integral(sine_mode - cosine_mode)
        )
    )


def _rectangular_mode_shear_geometric_integral(
    length: float,
    width: float,
    first: RitzMode,
    second: RitzMode,
) -> float:
    """Return symmetric unit-Nxy geometric coupling between Ritz modes."""

    first_x_second_y = (
        first.kx
        * second.ky
        * _sin_cos_integral(second.m, first.m, length)
        * _sin_cos_integral(first.n, second.n, width)
    )
    second_x_first_y = (
        second.kx
        * first.ky
        * _sin_cos_integral(first.m, second.m, length)
        * _sin_cos_integral(second.n, first.n, width)
    )
    return first_x_second_y + second_x_first_y


def _mode_shear_geometric_integral(
    panel: S3PanelInput,
    first: RitzMode,
    second: RitzMode,
) -> float:
    return _rectangular_mode_shear_geometric_integral(
        panel.length,
        panel.width,
        first,
        second,
    )


def build_ritz_modes(
    panel: S3PanelInput,
    section: S3SectionProperties,
    config: S3SolverConfig,
    global_stiffness_scale: float = 1.0,
) -> list[RitzMode]:
    modes: list[RitzMode] = []
    imperfection = max(
        config.initial_imperfection_floor_mm,
        config.initial_imperfection_ratio * min(panel.length, panel.width),
    )
    for family, pressure_share in (
        ("local", config.pressure_local_share),
        ("global", config.pressure_global_share),
    ):
        stiffness = build_orthotropic_stiffness(panel, section, family, config)
        if family == "global":
            stiffness = _with_longitudinal_stiffness_scale(stiffness, global_stiffness_scale)
        for m in config.longitudinal_modes:
            for n in config.transverse_modes:
                kx, ky, linear, geometric, nonlinear = _mode_linear_terms(
                    panel,
                    stiffness,
                    config,
                    m,
                    n,
                )
                modes.append(
                    RitzMode(
                        family=family,
                        m=m,
                        n=n,
                        kx=kx,
                        ky=ky,
                        linear_stiffness=linear,
                        geometric_stiffness=geometric,
                        pressure_force=_pressure_generalized_force(panel, m, n, pressure_share),
                        imperfection=imperfection,
                        nonlinear_stiffness=max(nonlinear, EPS),
                    )
                )
    return modes


def _isotropic_plate_stiffness(panel: U3PanelInput | S3PanelInput) -> OrthotropicStiffness:
    d = _plate_bending_rigidity(panel)
    return OrthotropicStiffness(
        d11=d,
        d12=panel.poisson_ratio * d,
        d22=d,
        d66=0.5 * (1.0 - panel.poisson_ratio) * d,
        membrane_thickness=panel.plate_thickness,
    )


def build_u3_ritz_modes(panel: U3PanelInput, config: S3SolverConfig) -> list[RitzMode]:
    """Return the plate-only Rayleigh-Ritz modes used by the U3 solver."""

    modes: list[RitzMode] = []
    imperfection = max(
        config.initial_imperfection_floor_mm,
        config.initial_imperfection_ratio * min(panel.length, panel.width),
    )
    stiffness = _isotropic_plate_stiffness(panel)
    for m in config.longitudinal_modes:
        for n in config.transverse_modes:
            kx, ky, linear, geometric, nonlinear = _mode_linear_terms(
                panel,
                stiffness,
                config,
                m,
                n,
            )
            modes.append(
                RitzMode(
                    family="plate",
                    m=m,
                    n=n,
                    kx=kx,
                    ky=ky,
                    linear_stiffness=linear,
                    geometric_stiffness=geometric,
                    pressure_force=_pressure_generalized_force(panel, m, n, 1.0),
                    imperfection=imperfection,
                    nonlinear_stiffness=max(nonlinear, EPS),
                )
            )
    return modes


def ritz_combined_buckling_factor(
    panel: S3PanelInput,
    modes: Sequence[RitzMode],
    family: str,
) -> dict[str, Any] | None:
    """Return a coupled linear Ritz factor for compression plus panel shear.

    Normal compression is diagonal in the sine basis.  Panel shear couples
    opposite-parity Fourier terms and is the mechanism that lets a double
    Fourier expansion describe inclined elastic shear shapes.  This truncated
    candidate makes that interaction explicit without replacing the classical
    pure-shear plate diagnostic retained for a narrow reduced basis.
    """

    family_modes = [mode for mode in modes if mode.family == family]
    if len(family_modes) < 2:
        return None

    linear = np.diag([mode.linear_stiffness for mode in family_modes])
    geometric, coupling = _ritz_geometric_matrix(panel, family_modes)
    if coupling["coupled_terms"] == 0:
        return None

    linear_diagonal = np.diag(linear)
    inverse_sqrt_linear = np.diag(1.0 / np.sqrt(np.maximum(linear_diagonal, EPS)))
    transformed = inverse_sqrt_linear @ geometric @ inverse_sqrt_linear
    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    positive_indices = [
        index
        for index, eigenvalue in enumerate(eigenvalues)
        if eigenvalue > EPS and math.isfinite(float(eigenvalue))
    ]
    if not positive_indices:
        return None

    critical_index = max(positive_indices, key=lambda index: float(eigenvalues[index]))
    eigenvalue = float(eigenvalues[critical_index])
    transformed_vector = eigenvectors[:, critical_index]
    physical_vector = inverse_sqrt_linear @ transformed_vector
    modal_weights = np.abs(physical_vector)
    weight_sum = float(np.sum(modal_weights))
    composition = []
    if weight_sum > EPS:
        composition = [
            {
                "mode": mode.label,
                "weight": float(weight / weight_sum),
            }
            for mode, weight in sorted(
                zip(family_modes, modal_weights),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            if weight > EPS
        ]
    return {
        "factor": 1.0 / eigenvalue,
        "eigenvalue": eigenvalue,
        "family": family,
        "signed_shear_resultant": coupling["signed_shear_resultant"],
        "coupled_terms": coupling["coupled_terms"],
        "mode_composition": composition[:4],
    }


def _ritz_geometric_matrix(
    panel: S3PanelInput,
    modes: Sequence[RitzMode],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return normal-plus-shear geometric stiffness for Ritz modes.

    The local and distributed-global families share the trigonometric panel
    coordinates but represent different reduced S3 displacement fields.  Shear
    coupling is therefore kept inside each family block.
    """

    geometric = np.diag([mode.geometric_stiffness for mode in modes])
    signed_shear_resultant = panel.shear_stress * panel.plate_thickness
    family_terms: dict[str, int] = {}
    if abs(signed_shear_resultant) <= EPS or len(modes) < 2:
        return geometric, {
            "signed_shear_resultant": signed_shear_resultant,
            "coupled_terms": 0,
            "family_terms": family_terms,
        }

    coupled_terms = 0
    for row, first in enumerate(modes):
        for column in range(row + 1, len(modes)):
            second = modes[column]
            if first.family != second.family:
                continue
            coupling = (
                signed_shear_resultant
                * _mode_shear_geometric_integral(panel, first, second)
            )
            if abs(coupling) <= EPS:
                continue
            geometric[row, column] += coupling
            geometric[column, row] += coupling
            coupled_terms += 1
            family_terms[first.family] = family_terms.get(first.family, 0) + 1
    return geometric, {
        "signed_shear_resultant": signed_shear_resultant,
        "coupled_terms": coupled_terms,
        "family_terms": family_terms,
    }


def _local_amplitude_ratio(panel: S3PanelInput, modes: Sequence[RitzMode], amplitudes: Sequence[float]) -> float:
    local_amplitude = sum(abs(amplitude) for mode, amplitude in zip(modes, amplitudes) if mode.family == "local")
    local_span = max(min(panel.length, panel.width), EPS)
    return local_amplitude / local_span


def local_global_stiffness_scale(
    panel: S3PanelInput,
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
    load_factor: float,
    config: S3SolverConfig,
) -> dict[str, float]:
    """Return reduced global longitudinal stiffness from local response.

    The Byklum model family uses local response to supply anisotropic stiffness
    to a global orthotropic plate model.  This reduced pass applies a scalar
    longitudinal degradation driven by local-family elastic utilization so the
    interaction is visible and configurable.  Local amplitude is still
    reported, but not used as a degradation driver because pressure deflection
    in the reduced basis makes that scalar too noisy.
    """

    local_mode_factors = [
        mode.linear_stiffness / mode.geometric_stiffness
        for mode in modes
        if mode.family == "local" and mode.geometric_stiffness > EPS
    ]
    local_elastic_utilization = (
        load_factor / max(min(local_mode_factors), EPS)
        if local_mode_factors
        else 0.0
    )
    amplitude_ratio = _local_amplitude_ratio(panel, modes, amplitudes)
    amplitude_utilization = amplitude_ratio / max(config.initial_imperfection_ratio, EPS)
    interaction_driver = local_elastic_utilization
    scale = 1.0 / (1.0 + config.local_global_coupling_gain * max(interaction_driver - 1.0, 0.0))
    scale = min(1.0, max(config.local_global_coupling_floor, scale))
    return {
        "scale": scale,
        "local_elastic_utilization": local_elastic_utilization,
        "local_amplitude_ratio": amplitude_ratio,
        "amplitude_utilization": amplitude_utilization,
        "interaction_driver": interaction_driver,
    }


def _stiffener_column_factor(panel: S3PanelInput, section: S3SectionProperties) -> float | None:
    compression = max(panel.axial_stress, 0.0)
    if compression <= EPS:
        return None
    effective_length = panel.length * (0.70 if panel.stiffener_boundary == "Cont" else 1.0)
    euler_force = math.pi**2 * panel.elastic_modulus * section.inertia_x / max(effective_length**2, EPS)
    reference_force = compression * section.area
    if reference_force <= EPS:
        return None
    return euler_force / reference_force


def _plate_strip_shear_buckling(
    panel: S3PanelInput,
    length: float,
    width: float,
    thickness: float,
    shear_stress: float,
) -> dict[str, float] | None:
    """Return the classical elastic shear buckling factor for a plate strip.

    Notes
    -----
    This uses the classical simply-supported plate shear buckling coefficient

        k_tau = 5.34 + 4 / alpha^2

    where alpha = long_side / short_side >= 1.

    The elastic shear buckling stress is calculated directly as

        tau_cr = k_tau * pi^2 * E / (12 * (1 - nu^2)) * (t / short_side)^2

    This is the classical elastic plate coefficient, not the DNV-CG-0128
    Sec.3 CFM Table 3 Case 15 K_tau expression with the sqrt(3) factor.
    """

    shear = abs(shear_stress)
    if shear <= EPS:
        return None

    short_side = min(length, width)
    long_side = max(length, width)
    alpha = long_side / max(short_side, EPS)

    shear_coefficient = 5.34 + 4.0 / max(alpha * alpha, EPS)

    elastic_reference = (
        math.pi**2
        * panel.elastic_modulus
        / (12.0 * (1.0 - panel.poisson_ratio**2))
        * (thickness / max(short_side, EPS)) ** 2
    )

    critical_stress = shear_coefficient * elastic_reference

    return {
        "factor": critical_stress / shear,
        "critical_stress": critical_stress,
        "coefficient": shear_coefficient,
        "aspect_ratio": alpha,
        "short_side": short_side,
        "long_side": long_side,
    }


def _local_plate_shear_buckling(panel: S3PanelInput) -> dict[str, float] | None:
    """Return the classical local plate shear factor for the unit bay.

    Same-mode diagonal sine terms are not a sound representation of plate shear
    buckling.  The reduced solver therefore reports a separate simply
    supported plate shear candidate while the nonlinear path keeps shear in
    the yield stress state.
    """

    return _plate_strip_shear_buckling(
        panel,
        panel.length,
        panel.width,
        panel.plate_thickness,
        panel.shear_stress,
    )


def _web_ritz_modes(
    panel: S3PanelInput,
    config: S3SolverConfig,
    compression_stress: float,
) -> list[RitzMode]:
    """Return a bounded isotropic Ritz surface for open stiffener webs."""

    bending_rigidity = panel.elastic_modulus * panel.web_thickness**3 / (
        12.0 * (1.0 - panel.poisson_ratio**2)
    )
    compression_resultant = max(compression_stress, 0.0) * panel.web_thickness
    area_factor = panel.length * panel.stiffener_height / 4.0
    modes: list[RitzMode] = []
    for m in config.web_longitudinal_modes:
        for n in config.web_depth_modes:
            kx = m * math.pi / panel.length
            ky = n * math.pi / panel.stiffener_height
            wave_norm = kx * kx + ky * ky
            modes.append(
                RitzMode(
                    family="web",
                    m=m,
                    n=n,
                    kx=kx,
                    ky=ky,
                    linear_stiffness=area_factor * bending_rigidity * wave_norm * wave_norm,
                    geometric_stiffness=area_factor * compression_resultant * kx * kx,
                    pressure_force=0.0,
                    imperfection=0.0,
                    nonlinear_stiffness=EPS,
                )
            )
    return modes


def _ritz_factor_summary(
    linear: np.ndarray,
    geometric: np.ndarray,
    modes: Sequence[RitzMode],
    metadata: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the positive critical factor for a Ritz geometric matrix."""

    linear_diagonal = np.diag(linear)
    inverse_sqrt_linear = np.diag(1.0 / np.sqrt(np.maximum(linear_diagonal, EPS)))
    transformed = inverse_sqrt_linear @ geometric @ inverse_sqrt_linear
    eigenvalues, eigenvectors = np.linalg.eigh(transformed)
    positive_indices = [
        index
        for index, eigenvalue in enumerate(eigenvalues)
        if eigenvalue > EPS and math.isfinite(float(eigenvalue))
    ]
    if not positive_indices:
        return None

    critical_index = max(positive_indices, key=lambda index: float(eigenvalues[index]))
    eigenvalue = float(eigenvalues[critical_index])
    physical_vector = inverse_sqrt_linear @ eigenvectors[:, critical_index]
    modal_weights = np.abs(physical_vector)
    weight_sum = float(np.sum(modal_weights))
    composition = []
    if weight_sum > EPS:
        composition = [
            {
                "mode": mode.label,
                "weight": float(weight / weight_sum),
            }
            for mode, weight in sorted(
                zip(modes, modal_weights),
                key=lambda item: float(item[1]),
                reverse=True,
            )
            if weight > EPS
        ]
    return {
        "factor": 1.0 / eigenvalue,
        "eigenvalue": eigenvalue,
        "mode_composition": composition[:4],
        **metadata,
    }


def _web_ritz_buckling(
    panel: S3PanelInput,
    config: S3SolverConfig,
    compression_demand: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a web-surface Ritz factor for axial web compression and shear."""

    modes = _web_ritz_modes(panel, config, float(compression_demand["stress"]))
    if not modes:
        return None

    linear = np.diag([mode.linear_stiffness for mode in modes])
    geometric = np.diag([mode.geometric_stiffness for mode in modes])
    signed_shear_resultant = panel.shear_stress * panel.web_thickness
    coupled_terms = 0
    if abs(signed_shear_resultant) > EPS:
        for row, first in enumerate(modes):
            for column in range(row + 1, len(modes)):
                second = modes[column]
                coupling = (
                    signed_shear_resultant
                    * _rectangular_mode_shear_geometric_integral(
                        panel.length,
                        panel.stiffener_height,
                        first,
                        second,
                    )
                )
                if abs(coupling) <= EPS:
                    continue
                geometric[row, column] += coupling
                geometric[column, row] += coupling
                coupled_terms += 1

    return _ritz_factor_summary(
        linear,
        geometric,
        modes,
        {
            "family": "web",
            "signed_shear_resultant": signed_shear_resultant,
            "coupled_terms": coupled_terms,
            "mode_count": len(modes),
            "longitudinal_modes": list(config.web_longitudinal_modes),
            "depth_modes": list(config.web_depth_modes),
            "surface": {
                "length": panel.length,
                "depth": panel.stiffener_height,
                "thickness": panel.web_thickness,
            },
            "compression_demand": compression_demand,
        },
    )


def _stiffener_web_reference_compression(
    panel: S3PanelInput,
    gross_section: S3SectionProperties,
    stiffener_section: S3SectionProperties,
    config: S3SolverConfig,
) -> dict[str, Any]:
    """Return reference web-edge compression for the local web check.

    The panel axial stress is the base proportional compression driver.  A
    sniped stiffener also sees the public axial-eccentricity moment already
    used by the SI/PI yield branches, so the local web strip uses the
    compressive edge envelope of that section stress instead of treating the
    web as an isolated plate under nominal panel stress only.
    """

    effective_axial = _stiffener_effective_axial_stress(
        panel,
        gross_section,
        stiffener_section,
        1.0,
    )
    axial_compression = max(float(effective_axial["stress"]), 0.0)
    sniped = _sniped_stiffener_eccentricity_moments(
        panel,
        gross_section,
        stiffener_section,
        1.0,
        config,
    )
    web_coordinates = {
        "root": 0.5 * panel.plate_thickness,
        "tip": 0.5 * panel.plate_thickness + panel.stiffener_height,
    }
    absolute_moment = abs(float(sniped["absolute"]))
    edge_bending = {
        edge: absolute_moment
        * abs(coordinate - stiffener_section.centroid_from_plate_midplane)
        / max(stiffener_section.inertia_x, EPS)
        for edge, coordinate in web_coordinates.items()
    }
    edge_compression = {
        edge: axial_compression + bending
        for edge, bending in edge_bending.items()
    }
    controlling_edge = max(edge_compression, key=edge_compression.get)
    return {
        "stress": edge_compression[controlling_edge],
        "source": "stiffener-section-axial-plus-sniped-web-edge-envelope",
        "controlling_edge": controlling_edge,
        "edge_compression": edge_compression,
        "edge_bending_stress": edge_bending,
        "effective_axial_stress": effective_axial,
        "sniped_eccentricity_moment": sniped,
    }


def _stiffener_web_local_buckling(
    panel: S3PanelInput,
    gross_section: S3SectionProperties,
    stiffener_section: S3SectionProperties,
    config: S3SolverConfig,
) -> dict[str, Any] | None:
    """Return an open-profile web compression-shear candidate.

    The web is treated as a long simply supported plate strip under a reference
    web-edge compression envelope and panel shear.  The candidate keeps the
    full PULS web/stiffener stress redistribution out of scope, but the explicit
    interaction avoids treating tall loaded webs as compression-only strips.
    """

    compression_demand = _stiffener_web_reference_compression(
        panel,
        gross_section,
        stiffener_section,
        config,
    )
    compression = float(compression_demand["stress"])
    compression_factor = None
    compression_critical_stress = None
    compression_coefficient = 4.0
    if compression > EPS:
        elastic_reference = (
            math.pi**2
            * panel.elastic_modulus
            / (12.0 * (1.0 - panel.poisson_ratio**2))
            * (panel.web_thickness / panel.stiffener_height) ** 2
        )
        compression_critical_stress = compression_coefficient * elastic_reference
        compression_factor = compression_critical_stress / compression

    shear = _plate_strip_shear_buckling(
        panel,
        panel.length,
        panel.stiffener_height,
        panel.web_thickness,
        panel.shear_stress,
    )
    web_ritz = _web_ritz_buckling(panel, config, compression_demand)
    factor_rows = [
        factor
        for factor in (
            compression_factor,
            None if shear is None else shear["factor"],
        )
        if factor is not None and factor > EPS and math.isfinite(factor)
    ]
    if not factor_rows:
        return None

    exponent = max(config.web_shear_interaction_exponent, EPS)
    interaction_usage = sum((1.0 / factor) ** exponent for factor in factor_rows) ** (1.0 / exponent)
    approximation_notes = [
        "web modeled as an isolated simply supported compression-shear strip",
        "S3 web buckling load shedding with plate, flange, and torsional displacement fields is not reproduced",
    ]
    return {
        "factor": 1.0 / max(interaction_usage, EPS),
        "critical_stress": compression_critical_stress,
        "coefficient": compression_coefficient,
        "compression_demand": compression_demand,
        "compression_factor": compression_factor,
        "shear_factor": None if shear is None else shear["factor"],
        "shear_critical_stress": None if shear is None else shear["critical_stress"],
        "shear_coefficient": None if shear is None else shear["coefficient"],
        "interaction_exponent": exponent,
        "factor_source": "strip-compression-shear-interaction",
        "web_ritz": web_ritz,
        "coverage": "reduced-strip-approximation",
        "approximation_notes": approximation_notes,
    }


def _local_plate_web_interaction(
    plate_shear: Mapping[str, Any] | None,
    web_local: Mapping[str, Any] | None,
    config: S3SolverConfig,
) -> dict[str, Any] | None:
    """Return a reduced mixed local interaction for plate and web response.

    The public S3 description treats panel failure as a mixed local stiffened
    response with load shedding between plating and primary stiffeners.  The
    first reduced solver still has separate plate-shear and web-local
    candidates, so this explicit interaction candidate keeps those concurrent
    local shear usages visible while the full coupled displacement family is
    not yet implemented.
    """

    if plate_shear is None or web_local is None:
        return None
    plate_factor = _optional_float(plate_shear.get("factor"))
    web_factor = _optional_float(web_local.get("factor"))
    if (
        plate_factor is None
        or web_factor is None
        or plate_factor <= EPS
        or web_factor <= EPS
    ):
        return None

    exponent = max(config.local_plate_web_interaction_exponent, EPS)
    plate_usage = 1.0 / plate_factor
    web_usage = 1.0 / web_factor
    interaction_usage = (
        plate_usage**exponent + web_usage**exponent
    ) ** (1.0 / exponent)
    return {
        "factor": 1.0 / max(interaction_usage, EPS),
        "interaction_usage": interaction_usage,
        "interaction_exponent": exponent,
        "plate_shear_factor": plate_factor,
        "plate_shear_usage": plate_usage,
        "web_local_factor": web_factor,
        "web_local_usage": web_usage,
        "factor_source": "plate-shear-web-local-usage-interaction",
        "coverage": "reduced-local-interaction",
        "approximation_notes": [
            "mixed local interaction keeps separate plate-shear and web-local reduced candidates",
            "full S3 plate, web, flange, and torsional local displacement coupling is not reproduced",
        ],
    }


def _stiffener_torsional_buckling(panel: S3PanelInput) -> dict[str, float] | None:
    """Return a reduced open-profile tripping/torsional stress candidate.

    DNV-CG-0128 uses a torsional reference stress based on St. Venant torsion,
    polar inertia, sectorial inertia, the stiffener span, and attachment
    restraint.  The reduced S3 solver uses the same ingredients in a
    conservative gross-section estimate about the web root.  It is exposed as
    an approximate candidate in diagnostics.
    """

    compression = max(panel.axial_stress, 0.0)
    if compression <= EPS:
        return None

    web_height = panel.stiffener_height
    web_thickness = panel.web_thickness
    flange_width = 0.0 if panel.stiffener_type == "Flatbar" else max(panel.flange_width, 0.0)
    flange_thickness = 0.0 if panel.stiffener_type == "Flatbar" else max(panel.flange_thickness, 0.0)
    web_area = web_height * web_thickness
    flange_area = flange_width * flange_thickness
    shear_modulus = panel.elastic_modulus / (2.0 * (1.0 + panel.poisson_ratio))

    torsion_constant = web_height * web_thickness**3 / 3.0
    if flange_area > 0.0:
        torsion_constant += flange_width * flange_thickness**3 / 3.0

    polar_inertia = web_area * web_height**2 / 3.0
    flange_offset = web_height + 0.5 * flange_thickness
    if flange_area > 0.0:
        polar_inertia += flange_area * (flange_offset**2 + flange_width**2 / 12.0)
    polar_inertia = max(polar_inertia, EPS)

    sectorial_inertia = 0.0
    if flange_area > 0.0:
        sectorial_inertia = flange_area * flange_width**2 * flange_offset**2 / 12.0

    effective_length = panel.length * (0.70 if panel.stiffener_boundary == "Cont" else 1.0)
    wave_number = math.pi / max(effective_length, EPS)
    critical_stress = (
        shear_modulus * torsion_constant
        + panel.elastic_modulus * sectorial_inertia * wave_number**2
    ) / polar_inertia
    return {
        "factor": critical_stress / compression,
        "critical_stress": critical_stress,
        "torsion_constant": torsion_constant,
        "polar_inertia": polar_inertia,
        "sectorial_inertia": sectorial_inertia,
        "effective_length": effective_length,
        "half_waves": 1.0,
    }


def elastic_buckling_factors(
    panel: S3PanelInput,
    section: S3SectionProperties,
    modes: Sequence[RitzMode],
    config: S3SolverConfig | None = None,
    stiffener_section: S3SectionProperties | None = None,
) -> dict[str, Any]:
    config = config or S3SolverConfig()
    stiffener_section = stiffener_section or section
    factor_rows = [
        {
            "factor": mode.linear_stiffness / mode.geometric_stiffness,
            "label": mode.label,
            "family": mode.family,
            "failure_family": "plate" if mode.family == "local" else "global-stiffened-strip",
        }
        for mode in modes
        if mode.geometric_stiffness > EPS
    ]
    column_factor = _stiffener_column_factor(panel, stiffener_section)
    if column_factor is not None:
        factor_rows.append(
            {
                "factor": column_factor,
                "label": "stiffener-column",
                "family": "stiffener-column",
                "failure_family": "global-stiffener-cutoff",
            }
        )
    shear_factor = _local_plate_shear_buckling(panel)
    if shear_factor is not None:
        factor_rows.append(
            {
                "factor": shear_factor["factor"],
                "label": "local-plate-shear",
                "family": "local-shear",
                "failure_family": "plate-shear",
            }
        )
    coupled_shear_factors = {
        family: ritz_combined_buckling_factor(panel, modes, family)
        for family in ("local", "global")
    }
    for family, factor in coupled_shear_factors.items():
        if factor is None:
            continue
        factor_rows.append(
            {
                "factor": factor["factor"],
                "label": f"{family}-ritz-combined-shear",
                "family": family,
                "failure_family": "plate-shear" if family == "local" else "global-stiffened-strip",
            }
        )
    web_factor = _stiffener_web_local_buckling(panel, section, stiffener_section, config)
    if web_factor is not None:
        factor_rows.append(
            {
                "factor": web_factor["factor"],
                "label": "stiffener-web-local",
                "family": "stiffener-web",
                "failure_family": "web-local",
            }
        )
    local_interaction = _local_plate_web_interaction(shear_factor, web_factor, config)
    if local_interaction is not None:
        factor_rows.append(
            {
                "factor": local_interaction["factor"],
                "label": "local-plate-web-interaction",
                "family": "local-interaction",
                "failure_family": "plate-web-local-interaction",
            }
        )
    torsional_factor = _stiffener_torsional_buckling(panel)
    if torsional_factor is not None:
        factor_rows.append(
            {
                "factor": torsional_factor["factor"],
                "label": "stiffener-torsional",
                "family": "stiffener-torsional",
                "failure_family": "torsional-stiffener",
            }
        )
    factor_rows = [
        row
        for row in factor_rows
        if row["factor"] > EPS and math.isfinite(row["factor"])
    ]
    local_plate_rows = [
        row
        for row in factor_rows
        if row["failure_family"] in {"plate", "plate-shear"}
    ]
    local_reference_factor = (
        min(float(row["factor"]) for row in local_plate_rows)
        if local_plate_rows
        else None
    )
    elastic_global_coupling_rows: list[dict[str, float | str]] = []
    if local_reference_factor is not None:
        for row in factor_rows:
            if row["failure_family"] != "global-stiffened-strip":
                continue
            uncoupled_factor = float(row["factor"])
            interaction_driver = uncoupled_factor / max(local_reference_factor, EPS)
            scale = 1.0 / (
                1.0
                + config.local_global_coupling_gain * max(interaction_driver - 1.0, 0.0)
            )
            scale = min(1.0, max(config.local_global_coupling_floor, scale))
            row["uncoupled_factor"] = uncoupled_factor
            row["elastic_coupling_scale"] = scale
            row["factor"] = uncoupled_factor * scale
            elastic_global_coupling_rows.append(
                {
                    "mode": str(row["label"]),
                    "uncoupled_factor": uncoupled_factor,
                    "coupled_factor": float(row["factor"]),
                    "scale": scale,
                    "interaction_driver": interaction_driver,
                }
            )
    if not factor_rows:
        return {
            "critical_factor": None,
            "critical_mode": "no-compressive-or-shear-buckling-driver",
            "critical_failure_family": "none",
            "stiffener_column_factor": column_factor,
            "local_plate_shear": shear_factor,
            "ritz_combined_shear": coupled_shear_factors,
            "stiffener_web_local": web_factor,
            "local_plate_web_interaction": local_interaction,
            "stiffener_torsional": torsional_factor,
            "elastic_global_coupling": {
                "local_reference_factor": local_reference_factor,
                "global_modes": elastic_global_coupling_rows,
            },
            "modeled_failure_families": {},
            "approximate_failure_families": [
                "torsional-stiffener",
                "web-local",
                "plate-web-local-interaction",
            ],
            "unmodeled_failure_families": ["stiffener-local-global-coupling"],
        }

    family_minima: dict[str, dict[str, Any]] = {}
    for row in factor_rows:
        failure_family = str(row["failure_family"])
        current = family_minima.get(failure_family)
        if current is None or row["factor"] < current["factor"]:
            family_minima[failure_family] = dict(row)

    family_usages = {
        name: 1.0 / max(float(row["factor"]), EPS)
        for name, row in family_minima.items()
    }
    usage_total = sum(family_usages.values())
    modeled_failure_families = {}
    for name, row in sorted(family_minima.items()):
        family_summary = {
            "critical_factor": row["factor"],
            "critical_mode": row["label"],
            "usage_share_percent": 100.0 * family_usages[name] / max(usage_total, EPS),
        }
        if "uncoupled_factor" in row:
            family_summary["uncoupled_factor"] = row["uncoupled_factor"]
            family_summary["elastic_coupling_scale"] = row["elastic_coupling_scale"]
        modeled_failure_families[name] = family_summary
    critical = min(factor_rows, key=lambda item: item["factor"])
    return {
        "critical_factor": critical["factor"],
        "critical_mode": critical["label"],
        "critical_failure_family": critical["failure_family"],
        "stiffener_column_factor": column_factor,
        "local_plate_shear": shear_factor,
        "ritz_combined_shear": coupled_shear_factors,
        "stiffener_web_local": web_factor,
        "local_plate_web_interaction": local_interaction,
        "stiffener_torsional": torsional_factor,
        "elastic_global_coupling": {
            "local_reference_factor": local_reference_factor,
            "global_modes": elastic_global_coupling_rows,
        },
        "modeled_failure_families": modeled_failure_families,
        "approximate_failure_families": [
            "torsional-stiffener",
            "web-local",
            "plate-web-local-interaction",
        ],
        "unmodeled_failure_families": ["stiffener-local-global-coupling"],
    }


def elastic_u3_buckling_factors(
    panel: U3PanelInput,
    modes: Sequence[RitzMode],
    config: S3SolverConfig | None = None,
) -> dict[str, Any]:
    """Return elastic buckling candidates for the U3 unstiffened plate."""

    config = config or S3SolverConfig()
    factor_rows = [
        {
            "factor": mode.linear_stiffness / mode.geometric_stiffness,
            "label": mode.label,
            "family": mode.family,
            "failure_family": "plate",
        }
        for mode in modes
        if mode.geometric_stiffness > EPS
    ]
    shear_factor = _plate_strip_shear_buckling(
        panel,
        panel.length,
        panel.width,
        panel.plate_thickness,
        panel.shear_stress,
    )
    if shear_factor is not None:
        factor_rows.append(
            {
                "factor": shear_factor["factor"],
                "label": "plate-shear",
                "family": "plate-shear",
                "failure_family": "plate-shear",
            }
        )
    coupled_shear = {"plate": ritz_combined_buckling_factor(panel, modes, "plate")}
    if coupled_shear["plate"] is not None:
        factor_rows.append(
            {
                "factor": coupled_shear["plate"]["factor"],
                "label": "plate-ritz-combined-shear",
                "family": "plate",
                "failure_family": "plate-shear",
            }
        )
    factor_rows = [
        row
        for row in factor_rows
        if row["factor"] > EPS and math.isfinite(row["factor"])
    ]
    if not factor_rows:
        return {
            "critical_factor": None,
            "critical_mode": "no-compressive-or-shear-buckling-driver",
            "critical_failure_family": "none",
            "local_plate_shear": shear_factor,
            "ritz_combined_shear": coupled_shear,
            "modeled_failure_families": {},
            "approximate_failure_families": [],
            "unmodeled_failure_families": ["production-U3-elasto-plastic-postbuckling-calibration"],
        }

    family_minima: dict[str, dict[str, Any]] = {}
    for row in factor_rows:
        failure_family = str(row["failure_family"])
        current = family_minima.get(failure_family)
        if current is None or row["factor"] < current["factor"]:
            family_minima[failure_family] = dict(row)

    family_usages = {
        name: 1.0 / max(float(row["factor"]), EPS)
        for name, row in family_minima.items()
    }
    usage_total = sum(family_usages.values())
    modeled_failure_families = {
        name: {
            "critical_factor": row["factor"],
            "critical_mode": row["label"],
            "usage_share_percent": 100.0 * family_usages[name] / max(usage_total, EPS),
        }
        for name, row in sorted(family_minima.items())
    }
    critical = min(factor_rows, key=lambda item: item["factor"])
    return {
        "critical_factor": critical["factor"],
        "critical_mode": critical["label"],
        "critical_failure_family": critical["failure_family"],
        "local_plate_shear": shear_factor,
        "ritz_combined_shear": coupled_shear,
        "modeled_failure_families": modeled_failure_families,
        "approximate_failure_families": [],
        "unmodeled_failure_families": ["production-U3-elasto-plastic-postbuckling-calibration"],
    }


def _limit_newton_delta(
    modes: Sequence[RitzMode],
    amplitudes: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    limited = delta.copy()
    for index, mode in enumerate(modes):
        max_delta = max(
            1.0,
            0.5 * abs(float(amplitudes[index])),
            0.10 / max(mode.kx, mode.ky, EPS),
        )
        if abs(float(limited[index])) > max_delta:
            limited[index] = math.copysign(max_delta, float(limited[index]))
    return limited


def solve_equilibrium_amplitudes(
    panel: S3PanelInput,
    modes: Sequence[RitzMode],
    load_factor: float,
    previous_amplitudes: Sequence[float],
    config: S3SolverConfig,
) -> tuple[list[float], bool, int]:
    """Solve the coupled reduced Ritz continuation equilibrium.

    Normal resultants keep diagonal geometric terms in this basis.  Panel shear
    contributes off-diagonal geometric coupling between opposite-parity modes,
    so the load-path residual and Newton tangent are assembled as vectors and
    matrices instead of solving each amplitude independently.
    """

    if not modes:
        return [], True, 0

    q = np.asarray(previous_amplitudes, dtype=float)
    if q.shape != (len(modes),):
        q = np.zeros(len(modes), dtype=float)

    linear = np.diag([mode.linear_stiffness for mode in modes])
    geometric, _ = _ritz_geometric_matrix(panel, modes)
    nonlinear = np.asarray([mode.nonlinear_stiffness for mode in modes], dtype=float)
    pressure = np.asarray([mode.pressure_force for mode in modes], dtype=float)
    imperfection = np.asarray([mode.imperfection for mode in modes], dtype=float)
    force = pressure + load_factor * (geometric @ imperfection)
    tangent_linear = linear - load_factor * geometric
    if np.max(np.abs(force)) <= EPS and np.max(np.abs(q)) <= EPS:
        return [0.0 for _ in modes], True, 0

    for iteration in range(1, config.newton_max_iterations + 1):
        nonlinear_response = nonlinear * q**3
        residual = tangent_linear @ q + nonlinear_response - force
        scale = max(
            float(np.max(np.abs(force))),
            float(np.max(np.abs(tangent_linear @ q))),
            float(np.max(np.abs(nonlinear_response))),
            1.0,
        )
        if float(np.max(np.abs(residual))) <= config.newton_tolerance * scale:
            return q.tolist(), True, iteration

        tangent = tangent_linear + np.diag(3.0 * nonlinear * q * q)
        try:
            delta = np.linalg.solve(tangent, -residual)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(tangent, -residual, rcond=None)[0]
        delta = _limit_newton_delta(modes, q, delta)
        q += delta
        if not np.all(np.isfinite(q)):
            return list(previous_amplitudes), False, iteration

    return q.tolist(), False, config.newton_max_iterations


def _mode_amplitude_summary(
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
) -> dict[str, Any]:
    family_maxima: dict[str, dict[str, float | str]] = {}
    max_amplitude = 0.0
    for mode, amplitude in zip(modes, amplitudes):
        magnitude = abs(amplitude)
        max_amplitude = max(max_amplitude, magnitude)
        current = family_maxima.get(mode.family)
        if current is None or magnitude > float(current["amplitude"]):
            family_maxima[mode.family] = {
                "mode": mode.label,
                "amplitude": magnitude,
                "signed_amplitude": amplitude,
            }
    return {
        "max_amplitude": max_amplitude,
        "families": family_maxima,
    }


def _stress_von_mises(sigma_x: float, sigma_y: float, tau_xy: float) -> float:
    return math.sqrt(max(sigma_x**2 - sigma_x * sigma_y + sigma_y**2 + 3.0 * tau_xy**2, 0.0))


def _mode_curvatures(
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
    x: float,
    y: float,
    family: str | None = None,
) -> tuple[float, float, float]:
    d2x = 0.0
    d2y = 0.0
    dxy = 0.0
    for mode, amplitude in zip(modes, amplitudes):
        if family is not None and mode.family != family:
            continue
        sin_x = math.sin(mode.kx * x)
        sin_y = math.sin(mode.ky * y)
        cos_x = math.cos(mode.kx * x)
        cos_y = math.cos(mode.ky * y)
        d2x -= amplitude * mode.kx * mode.kx * sin_x * sin_y
        d2y -= amplitude * mode.ky * mode.ky * sin_x * sin_y
        dxy += amplitude * mode.kx * mode.ky * cos_x * cos_y
    return d2x, d2y, dxy


def _pressure_stiffener_bending_moment(panel: S3PanelInput) -> float:
    if panel.pressure <= 0.0:
        return 0.0
    span_factor = 12.0 if panel.stiffener_boundary == "Cont" else 8.0
    return panel.pressure * panel.width * panel.length**2 / span_factor


def _sniped_stiffener_eccentricity_moments(
    panel: S3PanelInput,
    gross_section: S3SectionProperties,
    stiffener_section: S3SectionProperties,
    load_factor: float,
    config: S3SolverConfig,
) -> dict[str, float]:
    """Return branch-signed axial eccentricity moments for sniped stiffeners."""

    compression = max(load_factor * panel.axial_stress, 0.0)
    if panel.stiffener_boundary != "Sniped" or compression <= EPS:
        return {
            "absolute": 0.0,
            "stiffener_induced": 0.0,
            "plate_induced": 0.0,
        }
    moment = (
        config.sniped_eccentricity_factor
        * abs(stiffener_section.centroid_from_plate_midplane)
        * compression
        * gross_section.area
    )
    return {
        "absolute": moment,
        "stiffener_induced": -moment,
        "plate_induced": moment,
    }


def _stiffener_torsional_edge_distance(
    panel: S3PanelInput,
    stiffener_section: S3SectionProperties,
) -> float:
    """Return a reduced free-edge distance for SI torsional deformation."""

    if panel.stiffener_type == "Flatbar":
        return 0.5 * panel.web_thickness
    if panel.stiffener_type == "T-bar":
        return 0.5 * panel.flange_width

    stiffener_area = max(stiffener_section.stiffener_area, EPS)
    edge_distance = panel.flange_width - (
        panel.stiffener_height * panel.web_thickness**2
        + panel.flange_thickness * panel.flange_width**2
    ) / (
        2.0 * stiffener_area
    )
    return max(edge_distance, EPS)


def _stiffener_torsional_deformation_stress(
    panel: S3PanelInput,
    stiffener_section: S3SectionProperties,
    effective_axial_stress: float,
    config: S3SolverConfig,
) -> dict[str, float]:
    """Return the SI-only torsional deformation stress used in stiffener yield.

    Public stiffener interaction rules add a stress amplification term for
    stiffener-induced failure when the axial stress approaches the torsional
    reference stress.  This reduced pass uses the existing gross-section
    torsional reference stress and one half-wave along the supported span.
    """

    axial_compression = max(effective_axial_stress, 0.0)
    torsional = _stiffener_torsional_buckling(panel)
    if axial_compression <= EPS or torsional is None:
        return {
            "stress": 0.0,
            "edge_distance": 0.0,
            "imperfection_rotation": 0.0,
            "reference_stress": 0.0,
            "stress_ratio": 0.0,
        }

    reference_stress = torsional["critical_stress"]
    stress_ratio = min(axial_compression / max(reference_stress, EPS), 1.0 - 1.0e-6)
    edge_distance = _stiffener_torsional_edge_distance(panel, stiffener_section)
    half_waves = torsional["half_waves"]
    effective_length = torsional["effective_length"]
    imperfection_rotation = (
        config.torsional_imperfection_scale
        * effective_length
        / max(half_waves * panel.stiffener_height, EPS)
        * 1.0e-4
    )
    wave_number = half_waves * math.pi / max(effective_length, EPS)
    stress = (
        panel.elastic_modulus
        * edge_distance
        * imperfection_rotation
        * wave_number**2
        * (1.0 / max(1.0 - stress_ratio, EPS) - 1.0)
    )
    return {
        "stress": stress,
        "edge_distance": edge_distance,
        "imperfection_rotation": imperfection_rotation,
        "reference_stress": reference_stress,
        "stress_ratio": stress_ratio,
    }


def _plate_yield_ratio(
    panel: S3PanelInput,
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
    load_factor: float,
    config: S3SolverConfig,
) -> float:
    modulus = panel.elastic_modulus / (1.0 - panel.poisson_ratio**2)
    shear_modulus = panel.elastic_modulus / (2.0 * (1.0 + panel.poisson_ratio))
    z_values = (-0.5 * panel.plate_thickness, 0.5 * panel.plate_thickness)
    max_ratio = 0.0
    curvature_family = None if config.include_global_curvature_in_plate_yield else "local"

    for x_fraction in config.hot_spot_grid:
        x = panel.length * x_fraction
        for y_fraction in config.hot_spot_grid:
            y = panel.width * y_fraction
            d2x, d2y, dxy = _mode_curvatures(
                modes,
                amplitudes,
                x,
                y,
                family=curvature_family,
            )
            transverse_stress = (
                panel.transverse_stress_1
                + (panel.transverse_stress_2 - panel.transverse_stress_1) * y_fraction
            )
            for z in z_values:
                bending_x = -modulus * z * (d2x + panel.poisson_ratio * d2y)
                bending_y = -modulus * z * (d2y + panel.poisson_ratio * d2x)
                bending_tau = -2.0 * shear_modulus * z * dxy
                vm = _stress_von_mises(
                    load_factor * panel.axial_stress + bending_x,
                    load_factor * transverse_stress + bending_y,
                    load_factor * panel.shear_stress + bending_tau,
                )
                max_ratio = max(max_ratio, vm / panel.yield_stress_plate)
    return max_ratio


def _u3_plate_yield_ratio(
    panel: U3PanelInput,
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
    load_factor: float,
    config: S3SolverConfig,
) -> float:
    modulus = panel.elastic_modulus / (1.0 - panel.poisson_ratio**2)
    shear_modulus = panel.elastic_modulus / (2.0 * (1.0 + panel.poisson_ratio))
    z_values = (-0.5 * panel.plate_thickness, 0.5 * panel.plate_thickness)
    max_ratio = 0.0

    for x_fraction in config.hot_spot_grid:
        x = panel.length * x_fraction
        axial_stress = (
            panel.axial_stress_1
            + (panel.axial_stress_2 - panel.axial_stress_1) * x_fraction
        )
        for y_fraction in config.hot_spot_grid:
            y = panel.width * y_fraction
            d2x, d2y, dxy = _mode_curvatures(modes, amplitudes, x, y)
            transverse_stress = (
                panel.transverse_stress_1
                + (panel.transverse_stress_2 - panel.transverse_stress_1) * y_fraction
            )
            for z in z_values:
                bending_x = -modulus * z * (d2x + panel.poisson_ratio * d2y)
                bending_y = -modulus * z * (d2y + panel.poisson_ratio * d2x)
                bending_tau = -2.0 * shear_modulus * z * dxy
                vm = _stress_von_mises(
                    load_factor * axial_stress + bending_x,
                    load_factor * transverse_stress + bending_y,
                    load_factor * panel.shear_stress + bending_tau,
                )
                max_ratio = max(max_ratio, vm / panel.yield_stress_plate)
    return max_ratio


def u3_yield_utilization(
    panel: U3PanelInput,
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
    load_factor: float,
    config: S3SolverConfig,
) -> dict[str, Any]:
    plate_ratio = _u3_plate_yield_ratio(panel, modes, amplitudes, load_factor, config)
    return {
        "max": plate_ratio,
        "plate": plate_ratio,
        "stiffener": None,
        "stiffener_induced": None,
        "plate_induced": None,
    }


def _stiffener_branch_stress_ratio(
    axial_stress: float,
    signed_bending_stress: float,
    yield_stress: float,
) -> dict[str, float]:
    stress = axial_stress + signed_bending_stress
    return {
        "stress": stress,
        "signed_bending_stress": signed_bending_stress,
        "ratio": abs(stress) / max(yield_stress, EPS),
    }


def _stiffener_effective_axial_stress(
    panel: S3PanelInput,
    gross_section: S3SectionProperties,
    stiffener_section: S3SectionProperties,
    load_factor: float,
) -> dict[str, float]:
    """Return attached-plating effective axial stress for stiffener checks."""

    nominal_stress = load_factor * panel.axial_stress
    area_factor = gross_section.area / max(stiffener_section.area, EPS)
    return {
        "stress": nominal_stress * area_factor,
        "nominal_stress": nominal_stress,
        "area_factor": area_factor,
    }


def _global_slenderness_reduction(
    panel: S3PanelInput,
    global_elastic_factor: float | None,
) -> dict[str, float | None]:
    """Return the public stiffener lateral-deformation reduction terms."""

    if global_elastic_factor is None or global_elastic_factor <= EPS:
        return {
            "gamma_reh": None,
            "global_elastic_factor": global_elastic_factor,
            "slenderness": None,
            "reduction_factor": 0.0,
        }

    reference_stress = _stress_von_mises(
        panel.axial_stress,
        panel.mean_transverse_stress,
        panel.shear_stress,
    )
    if reference_stress <= EPS:
        return {
            "gamma_reh": None,
            "global_elastic_factor": global_elastic_factor,
            "slenderness": None,
            "reduction_factor": 0.0,
        }

    gamma_reh = min(panel.yield_stress_plate, panel.yield_stress_stiffener) / reference_stress
    slenderness = math.sqrt(max(gamma_reh / global_elastic_factor, 0.0))
    if slenderness <= 1.56:
        reduction = 1.0 - slenderness**4 / 12.0
    else:
        reduction = 3.0 / max(4.0 * slenderness, EPS)
    return {
        "gamma_reh": gamma_reh,
        "global_elastic_factor": global_elastic_factor,
        "slenderness": slenderness,
        "reduction_factor": min(1.0, max(reduction, 0.0)),
    }


def _stiffener_lateral_deformation_moment(
    panel: S3PanelInput,
    stiffener_section: S3SectionProperties,
    load_factor: float,
    global_elastic_factor: float | None,
) -> dict[str, float | None]:
    """Return guide-style stiffener moment from lateral deformation."""

    slenderness = _global_slenderness_reduction(panel, global_elastic_factor)
    if (
        load_factor <= EPS
        or global_elastic_factor is None
        or global_elastic_factor <= load_factor + EPS
        or slenderness["reduction_factor"] <= EPS
    ):
        return {
            "moment": 0.0,
            "ideal_elastic_buckling_force": 0.0,
            "assumed_imperfection": panel.length / 1000.0,
            "amplification": 0.0,
            **slenderness,
        }

    effective_length = panel.length
    ideal_elastic_force = (
        math.pi**2
        * panel.elastic_modulus
        * stiffener_section.inertia_x
        / max(effective_length**2, EPS)
    )
    amplification = load_factor / max(global_elastic_factor - load_factor, EPS)
    assumed_imperfection = effective_length / 1000.0
    moment = (
        ideal_elastic_force
        * float(slenderness["reduction_factor"])
        * amplification
        * assumed_imperfection
    )
    return {
        "moment": moment,
        "ideal_elastic_buckling_force": ideal_elastic_force,
        "assumed_imperfection": assumed_imperfection,
        "amplification": amplification,
        **slenderness,
    }


def _stiffener_yield_ratios(
    panel: S3PanelInput,
    gross_section: S3SectionProperties,
    stiffener_section: S3SectionProperties,
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
    load_factor: float,
    config: S3SolverConfig,
    global_elastic_factor: float | None = None,
) -> dict[str, Any]:
    pressure_moment = _pressure_stiffener_bending_moment(panel)
    sniped_moments = _sniped_stiffener_eccentricity_moments(
        panel,
        gross_section,
        stiffener_section,
        load_factor,
        config,
    )
    effective_axial = _stiffener_effective_axial_stress(
        panel,
        gross_section,
        stiffener_section,
        load_factor,
    )
    torsional_deformation = _stiffener_torsional_deformation_stress(
        panel,
        stiffener_section,
        effective_axial["stress"],
        config,
    )
    lateral_deformation = _stiffener_lateral_deformation_moment(
        panel,
        stiffener_section,
        load_factor,
        global_elastic_factor,
    )
    max_curvature = 0.0
    for x_fraction in config.hot_spot_grid:
        d2x, _, _ = _mode_curvatures(
            modes,
            amplitudes,
            panel.length * x_fraction,
            0.5 * panel.width,
            family="global",
        )
        max_curvature = max(max_curvature, abs(d2x))
    si_deflection_stress = panel.elastic_modulus * stiffener_section.top_distance * max_curvature
    pi_deflection_stress = panel.elastic_modulus * stiffener_section.bottom_distance * max_curvature
    axial_stress = effective_axial["stress"]
    common_moment = pressure_moment + float(lateral_deformation["moment"])
    si_moment_stress = (
        common_moment + sniped_moments["stiffener_induced"]
    ) / max(stiffener_section.top_section_modulus, EPS)
    pi_moment_stress = (
        -common_moment + sniped_moments["plate_induced"]
    ) / max(stiffener_section.attached_plate_section_modulus, EPS)
    si_bending_stress = si_moment_stress + torsional_deformation["stress"]
    pi_bending_stress = pi_moment_stress
    si_stress = _stiffener_branch_stress_ratio(
        axial_stress,
        si_bending_stress,
        panel.yield_stress_stiffener,
    )
    pi_stress = _stiffener_branch_stress_ratio(
        axial_stress,
        pi_bending_stress,
        panel.yield_stress_plate,
    )
    si_ratio = si_stress["ratio"]
    pi_ratio = pi_stress["ratio"]
    return {
        "max": max(si_ratio, pi_ratio),
        "stiffener_induced": si_ratio,
        "plate_induced": pi_ratio,
        "effective_axial_stress": effective_axial,
        "stiffener_induced_stress": si_stress,
        "plate_induced_stress": pi_stress,
        "signed_bending_stress": {
            "stiffener_moment": si_moment_stress,
            "plate_moment": pi_moment_stress,
            "ritz_stiffener_deflection": si_deflection_stress,
            "ritz_plate_deflection": -pi_deflection_stress,
        },
        "pressure_moment": pressure_moment,
        "sniped_eccentricity_moment": sniped_moments,
        "lateral_deformation_moment": lateral_deformation,
        "torsional_deformation_stress": torsional_deformation["stress"],
        "torsional_deformation": torsional_deformation,
    }


def yield_utilization(
    panel: S3PanelInput,
    section: S3SectionProperties,
    stiffener_section: S3SectionProperties,
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
    load_factor: float,
    config: S3SolverConfig,
    global_elastic_factor: float | None = None,
) -> dict[str, Any]:
    plate_ratio = _plate_yield_ratio(panel, modes, amplitudes, load_factor, config)
    stiffener_ratios = _stiffener_yield_ratios(
        panel,
        section,
        stiffener_section,
        modes,
        amplitudes,
        load_factor,
        config,
        global_elastic_factor,
    )
    return {
        "max": max(plate_ratio, stiffener_ratios["max"]),
        "plate": plate_ratio,
        "stiffener": stiffener_ratios["max"],
        "stiffener_induced": stiffener_ratios["stiffener_induced"],
        "plate_induced": stiffener_ratios["plate_induced"],
        "effective_axial_stress": stiffener_ratios["effective_axial_stress"],
        "stiffener_induced_stress": stiffener_ratios["stiffener_induced_stress"],
        "plate_induced_stress": stiffener_ratios["plate_induced_stress"],
        "signed_bending_stress": stiffener_ratios["signed_bending_stress"],
        "pressure_moment": stiffener_ratios["pressure_moment"],
        "sniped_eccentricity_moment": stiffener_ratios["sniped_eccentricity_moment"],
        "lateral_deformation_moment": stiffener_ratios["lateral_deformation_moment"],
        "torsional_deformation_stress": stiffener_ratios["torsional_deformation_stress"],
        "torsional_deformation": stiffener_ratios["torsional_deformation"],
    }


def _continuation_summary(
    accepted_steps: int,
    rejected_steps: int,
    cutbacks: int,
    last_accepted_load_factor: float,
    min_accepted_step: float | None,
    max_accepted_step: float | None,
    current_step: float,
    config: S3SolverConfig,
) -> dict[str, Any]:
    return {
        "accepted_steps": accepted_steps,
        "rejected_steps": rejected_steps,
        "cutbacks": cutbacks,
        "last_accepted_load_factor": last_accepted_load_factor,
        "min_accepted_step": min_accepted_step,
        "max_accepted_step": max_accepted_step,
        "current_step": current_step,
        "configured_initial_step": config.initial_load_step,
        "configured_max_step": config.max_load_step,
        "configured_min_step": config.min_load_step,
        "configured_cutback": config.load_step_cutback,
        "configured_max_cutbacks": config.max_load_step_cutbacks,
    }


def _interpolate_capacity(
    previous_load: float,
    previous_ratio: float,
    load_factor: float,
    ratio: float,
    limit: float,
) -> float:
    if ratio <= previous_ratio + EPS:
        return load_factor
    fraction = (limit - previous_ratio) / (ratio - previous_ratio)
    fraction = min(max(fraction, 0.0), 1.0)
    return previous_load + fraction * (load_factor - previous_load)


def _notes() -> list[str]:
    return [
        "regular S3 unit strip only; T1, K3, corrugation and FRP are outside this milestone",
        "positive PULS CSV normal stress is compression; signed stresses scale while lateral pressure remains fixed",
        "Rayleigh-Ritz sine modes use a reduced local/global strip basis, not the full production PULS basis",
        "buckling usage is the reduced buckling-strength envelope over ultimate capacity and elastic local/global buckling limits; fixed-pressure first-yield with material preload contribution is reported in ultimate diagnostics but excluded from buckling-strength control by default",
        "shear-normal Ritz coupling is truncated in elastic and continuation checks; classical local plate shear remains a fallback candidate",
        "web-local compression-shear uses a reduced stiffener-section web-edge compression envelope and a reduced local plate-web interaction; torsional stiffener remains a reduced gross-section estimate",
        "stiffener yield exposes SI/PI section branches, lateral-deformation and sniped bending, SI-only torsional stress, and effective attached plate width",
        "global longitudinal strip stiffness degrades from local elastic utilization on the nonlinear load path",
        "validity limits are explicit covered-domain checks because the PULS user manual is not assumed",
    ]


def _u3_notes() -> list[str]:
    return [
        "regular U3 unstiffened rectangular panels only; T1, K3, corrugation and FRP are outside this milestone",
        "positive PULS export normal stress is compression; signed stresses scale while lateral pressure remains fixed",
        "Rayleigh-Ritz sine modes use an isotropic plate basis with simply-supported trigonometric shapes",
        "buckling usage is the reduced buckling-strength envelope over first-yield capacity and elastic plate/shear buckling limits",
        "U3 end stresses are interpolated in yield checks and averaged in elastic buckling checks",
        "validity limits are explicit covered-domain checks because the PULS user manual is not assumed",
    ]


def _invalid_result(reason: str, diagnostics: dict[str, Any] | None = None) -> S3Result:
    return S3Result(
        buckling_usage_factor=None,
        ultimate_usage_factor=None,
        valid=False,
        elastic_buckling_usage_factor=None,
        invalid_reason=reason,
        diagnostics=diagnostics or {},
        covered_domain_notes=_notes(),
    )


def _invalid_u3_result(reason: str, diagnostics: dict[str, Any] | None = None) -> S3Result:
    return S3Result(
        buckling_usage_factor=None,
        ultimate_usage_factor=None,
        valid=False,
        elastic_buckling_usage_factor=None,
        invalid_reason=reason,
        diagnostics=diagnostics or {},
        covered_domain_notes=_u3_notes(),
    )


def _pressure_dominated_yield_limit(
    panel: S3PanelInput,
    pressure_yield: Mapping[str, Any],
    final_yield: Mapping[str, Any],
    config: S3SolverConfig,
) -> bool:
    if panel.pressure <= EPS:
        return False
    preload_max = float(pressure_yield.get("max") or 0.0)
    if preload_max <= EPS:
        return False
    preload_share = preload_max / max(config.yield_utilization_limit, EPS)
    return preload_share >= config.pressure_dominated_yield_preload_ratio


def _relative_drift(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None:
        return None
    return abs(candidate - reference) / max(abs(reference), EPS)


def _mode_convergence_config(
    config: S3SolverConfig,
    longitudinal_modes: tuple[int, ...],
    transverse_modes: tuple[int, ...],
) -> S3SolverConfig:
    return replace(
        config,
        longitudinal_modes=longitudinal_modes,
        transverse_modes=transverse_modes,
        check_mode_convergence=False,
    )


def _summarize_mode_convergence(
    base_result: S3Result,
    medium_result: S3Result,
    high_result: S3Result,
    config: S3SolverConfig,
) -> dict[str, Any]:
    medium_buckling = _relative_drift(
        base_result.buckling_usage_factor,
        medium_result.buckling_usage_factor,
    )
    medium_ultimate = _relative_drift(
        base_result.ultimate_usage_factor,
        medium_result.ultimate_usage_factor,
    )
    high_buckling = _relative_drift(
        base_result.buckling_usage_factor,
        high_result.buckling_usage_factor,
    )
    high_ultimate = _relative_drift(
        base_result.ultimate_usage_factor,
        high_result.ultimate_usage_factor,
    )
    finite_drifts = [
        value
        for value in (
            medium_buckling,
            medium_ultimate,
            high_buckling,
            high_ultimate,
        )
        if value is not None and math.isfinite(value)
    ]
    return {
        "enabled": True,
        "medium_basis": {
            "longitudinal_modes": list(config.medium_longitudinal_modes),
            "transverse_modes": list(config.medium_transverse_modes),
            "valid": medium_result.valid,
            "buckling_usage_factor": medium_result.buckling_usage_factor,
            "ultimate_usage_factor": medium_result.ultimate_usage_factor,
            "buckling_relative_drift": medium_buckling,
            "ultimate_relative_drift": medium_ultimate,
        },
        "high_basis": {
            "longitudinal_modes": list(config.high_longitudinal_modes),
            "transverse_modes": list(config.high_transverse_modes),
            "valid": high_result.valid,
            "buckling_usage_factor": high_result.buckling_usage_factor,
            "ultimate_usage_factor": high_result.ultimate_usage_factor,
            "buckling_relative_drift": high_buckling,
            "ultimate_relative_drift": high_ultimate,
        },
        "max_relative_drift": max(finite_drifts) if finite_drifts else None,
        "high_confidence_drift_limit": config.high_confidence_drift_limit,
        "medium_confidence_drift_limit": config.medium_confidence_drift_limit,
    }


def _classify_confidence(
    result: S3Result,
    validation_domain: Mapping[str, Any],
    mode_convergence: Mapping[str, Any] | None,
    config: S3SolverConfig,
) -> tuple[str, list[str]]:
    if not result.valid:
        return "low", [f"invalid:{result.invalid_reason or 'unknown'}"]

    reasons: list[str] = []
    domain_reasons = list(validation_domain.get("reasons") or [])
    if domain_reasons:
        reasons.extend(f"domain:{reason}" for reason in domain_reasons)

    if result.ultimate_usage_factor is not None and result.buckling_usage_factor is not None:
        if result.ultimate_usage_factor > result.buckling_usage_factor + EPS:
            reasons.append("usage-order:ultimate-exceeds-buckling")

    if mode_convergence is None or not mode_convergence.get("enabled"):
        reasons.append("mode-convergence:not-run")
        return ("low" if any(reason.startswith("usage-order:") for reason in reasons) else "medium", reasons)

    max_drift = _optional_float(mode_convergence.get("max_relative_drift"))
    medium = mode_convergence.get("medium_basis", {})
    high = mode_convergence.get("high_basis", {})
    if not medium.get("valid") or not high.get("valid"):
        reasons.append("mode-convergence:non-valid-refined-basis")
        return "low", reasons
    if max_drift is None:
        reasons.append("mode-convergence:unavailable")
        return "low", reasons
    reasons.append(f"mode-convergence:max-drift={max_drift:.6g}")
    if max_drift <= config.high_confidence_drift_limit:
        return "high", reasons
    if max_drift <= config.medium_confidence_drift_limit:
        return "medium", reasons
    return "low", reasons


def _attach_reliability(
    result: S3Result,
    panel: S3PanelInput | U3PanelInput,
    config: S3SolverConfig,
    panel_family: str,
    solver: Any,
    validation_reasons: Sequence[str] | None = None,
) -> S3Result:
    validation_domain = _validation_domain(panel, config, panel_family, validation_reasons)
    diagnostics = dict(result.diagnostics)
    diagnostics["validation_domain"] = validation_domain
    mode_convergence = diagnostics.get("mode_convergence")
    if result.valid and config.check_mode_convergence:
        medium_result = solver(
            panel,
            _mode_convergence_config(
                config,
                config.medium_longitudinal_modes,
                config.medium_transverse_modes,
            ),
        )
        high_result = solver(
            panel,
            _mode_convergence_config(
                config,
                config.high_longitudinal_modes,
                config.high_transverse_modes,
            ),
        )
        mode_convergence = _summarize_mode_convergence(
            result,
            medium_result,
            high_result,
            config,
        )
    elif mode_convergence is None:
        mode_convergence = {"enabled": False}
    diagnostics["mode_convergence"] = mode_convergence
    confidence, confidence_reasons = _classify_confidence(
        result,
        validation_domain,
        mode_convergence,
        config,
    )
    diagnostics["confidence"] = confidence
    diagnostics["confidence_reasons"] = confidence_reasons
    return S3Result(
        buckling_usage_factor=result.buckling_usage_factor,
        ultimate_usage_factor=result.ultimate_usage_factor,
        elastic_buckling_usage_factor=result.elastic_buckling_usage_factor,
        valid=result.valid,
        invalid_reason=result.invalid_reason,
        diagnostics=diagnostics,
        covered_domain_notes=result.covered_domain_notes,
        confidence=confidence,
        confidence_reasons=confidence_reasons,
    )


def solve_s3_panel(panel: S3PanelInput, config: S3SolverConfig | None = None) -> S3Result:
    """Solve the reduced S3 load path and return usage-factor diagnostics."""

    config = config or S3SolverConfig()
    validation_reasons = collect_s3_validation_reasons(panel, config)
    validation_error = validation_reasons[0] if validation_reasons else None
    if validation_error is not None:
        return _attach_reliability(
            _invalid_result(validation_error),
            panel,
            config,
            "S3",
            solve_s3_panel,
            validation_reasons,
        )

    if all(
        abs(value) <= EPS
        for value in (
            panel.axial_stress,
            panel.mean_transverse_stress,
            panel.shear_stress,
        )
    ):
        return _attach_reliability(
            _invalid_result("zero-variable-load"),
            panel,
            config,
            "S3",
            solve_s3_panel,
            ["zero-variable-load"],
        )

    section = build_section_properties(panel)
    stiffener_section, effective_width = build_effective_stiffener_section(panel, config)
    modes = build_ritz_modes(panel, section, config)
    amplitudes = [0.0 for _ in modes]
    amplitudes, pressure_converged, pressure_iterations = solve_equilibrium_amplitudes(
        panel,
        modes,
        0.0,
        amplitudes,
        config,
    )
    if not pressure_converged:
        return _attach_reliability(
            _invalid_result(
                "non-convergence",
                {
                    "stage": "pressure-preload",
                    "iterations": pressure_iterations,
                },
            ),
            panel,
            config,
            "S3",
            solve_s3_panel,
            validation_reasons,
        )

    pressure_yield = yield_utilization(panel, section, stiffener_section, modes, amplitudes, 0.0, config)
    pressure_preload_response = {
        "iterations": pressure_iterations,
        "amplitudes": _mode_amplitude_summary(modes, amplitudes),
        "yield_utilization": pressure_yield,
        "controlling_yield_branch": max(
            ("plate", "stiffener_induced", "plate_induced"),
            key=lambda branch: float(pressure_yield[branch]),
        ),
    }
    if pressure_yield["max"] >= config.pressure_yield_limit:
        return _attach_reliability(
            _invalid_result(
                "pressure",
                {
                    "stage": "pressure-preload",
                    "yield_utilization": pressure_yield,
                    "pressure_preload_response": pressure_preload_response,
                    "pressure_iterations": pressure_iterations,
                },
            ),
            panel,
            config,
            "S3",
            solve_s3_panel,
            validation_reasons,
        )

    buckling = elastic_buckling_factors(panel, section, modes, config, stiffener_section)
    buckling_factor = buckling["critical_factor"]
    elastic_buckling_usage = None if buckling_factor is None else 1.0 / max(buckling_factor, EPS)
    column_factor = buckling["stiffener_column_factor"]
    global_family = buckling["modeled_failure_families"].get("global-stiffened-strip", {})
    global_elastic_cutoff_factor = _optional_float(global_family.get("critical_factor"))

    previous_load = 0.0
    previous_yield = pressure_yield["max"]
    yield_capacity_factor: float | None = None
    max_iterations = pressure_iterations
    collapse_state = "first-yield"
    final_yield = pressure_yield
    local_global_coupling = local_global_stiffness_scale(panel, modes, amplitudes, 0.0, config)

    accepted_steps = 0
    rejected_steps = 0
    cutbacks = 0
    min_accepted_step: float | None = None
    max_accepted_step: float | None = None
    current_step = min(
        max(config.initial_load_step, config.min_load_step, EPS),
        max(config.max_load_step, config.min_load_step, EPS),
    )
    cutback_factor = min(max(config.load_step_cutback, EPS), 0.95)

    while previous_load < config.max_load_factor - EPS:
        load_factor = min(previous_load + current_step, config.max_load_factor)
        attempted_step = load_factor - previous_load
        local_global_coupling = local_global_stiffness_scale(panel, modes, amplitudes, load_factor, config)
        if local_global_coupling["scale"] < 1.0:
            modes = build_ritz_modes(
                panel,
                section,
                config,
                global_stiffness_scale=local_global_coupling["scale"],
            )
        trial_amplitudes, converged, iterations = solve_equilibrium_amplitudes(
            panel,
            modes,
            load_factor,
            amplitudes,
            config,
        )
        max_iterations = max(max_iterations, iterations)
        if not converged:
            rejected_steps += 1
            next_step = current_step * cutback_factor
            if (
                next_step < max(config.min_load_step, EPS)
                or cutbacks >= config.max_load_step_cutbacks
            ):
                continuation = _continuation_summary(
                    accepted_steps,
                    rejected_steps,
                    cutbacks,
                    previous_load,
                    min_accepted_step,
                    max_accepted_step,
                    current_step,
                    config,
                )
                continuation["attempted_load_factor"] = load_factor
                continuation["attempted_step"] = attempted_step
                continuation["next_cutback_step"] = next_step
                continuation["newton_iterations"] = iterations
                continuation["cutback_exhausted"] = True
                return _attach_reliability(
                    _invalid_result(
                        "non-convergence",
                        {
                            "stage": "in-plane-continuation",
                            "load_factor": load_factor,
                            "iterations": iterations,
                            "buckling": buckling,
                            "continuation": continuation,
                        },
                    ),
                    panel,
                    config,
                    "S3",
                    solve_s3_panel,
                    validation_reasons,
                )
            current_step = next_step
            cutbacks += 1
            continue

        amplitudes = trial_amplitudes
        accepted_steps += 1
        min_accepted_step = (
            attempted_step
            if min_accepted_step is None
            else min(min_accepted_step, attempted_step)
        )
        max_accepted_step = (
            attempted_step
            if max_accepted_step is None
            else max(max_accepted_step, attempted_step)
        )
        ultimate_yield_global_factor = (
            global_elastic_cutoff_factor
            if config.include_lateral_deformation_in_ultimate_yield
            else None
        )
        final_yield = yield_utilization(
            panel,
            section,
            stiffener_section,
            modes,
            amplitudes,
            load_factor,
            config,
            ultimate_yield_global_factor,
        )
        if final_yield["max"] >= config.yield_utilization_limit:
            yield_capacity_factor = _interpolate_capacity(
                previous_load,
                previous_yield,
                load_factor,
                final_yield["max"],
                config.yield_utilization_limit,
            )
            previous_load = load_factor
            break
        previous_load = load_factor
        previous_yield = final_yield["max"]
        if previous_load >= 2.0:
            current_step = min(current_step * config.load_step_growth, config.max_load_step)

    continuation = _continuation_summary(
        accepted_steps,
        rejected_steps,
        cutbacks,
        previous_load,
        min_accepted_step,
        max_accepted_step,
        current_step,
        config,
    )

    ultimate_capacity_factor = yield_capacity_factor
    if ultimate_capacity_factor is None:
        if global_elastic_cutoff_factor is not None:
            ultimate_capacity_factor = global_elastic_cutoff_factor
            collapse_state = "global-elastic-cutoff"
        if column_factor is not None and (
            ultimate_capacity_factor is None or column_factor < ultimate_capacity_factor
        ):
            ultimate_capacity_factor = column_factor
            collapse_state = "stiffener-column-cutoff"

    if ultimate_capacity_factor is None or ultimate_capacity_factor <= EPS:
        return _attach_reliability(
            _invalid_result(
                "no-collapse-within-load-range",
                {
                    "max_load_factor": config.max_load_factor,
                    "buckling": buckling,
                    "yield_utilization": final_yield,
                },
            ),
            panel,
            config,
            "S3",
            solve_s3_panel,
            validation_reasons,
        )

    pressure_dominated_yield = _pressure_dominated_yield_limit(
        panel,
        pressure_yield,
        final_yield,
        config,
    )
    include_ultimate_in_buckling_strength = (
        config.include_pressure_dominated_yield_in_buckling_strength
        or not pressure_dominated_yield
    )
    buckling_strength_limits = {}
    excluded_buckling_strength_limits = {}
    if include_ultimate_in_buckling_strength:
        buckling_strength_limits["ultimate_capacity"] = ultimate_capacity_factor
    else:
        excluded_buckling_strength_limits[
            "ultimate_capacity"
        ] = "pressure-dominated-fixed-preload-yield"
    if buckling_factor is not None:
        buckling_strength_limits["elastic_buckling_envelope"] = buckling_factor
    if not buckling_strength_limits:
        buckling_strength_limits["ultimate_capacity"] = ultimate_capacity_factor
    buckling_strength_control, buckling_strength_capacity_factor = min(
        buckling_strength_limits.items(),
        key=lambda item: float(item[1]),
    )
    raw_ultimate_capacity_factor = ultimate_capacity_factor
    reported_ultimate_capacity_factor = max(
        raw_ultimate_capacity_factor,
        buckling_strength_capacity_factor,
    )
    ultimate_lifted_to_buckling_strength = (
        reported_ultimate_capacity_factor > raw_ultimate_capacity_factor + EPS
    )
    buckling_strength = {
        "capacity_factor": buckling_strength_capacity_factor,
        "usage_factor": 1.0 / max(buckling_strength_capacity_factor, EPS),
        "controlling_limit": buckling_strength_control,
        "component_capacity_factors": buckling_strength_limits,
        "excluded_component_capacity_factors": excluded_buckling_strength_limits,
        "elastic_usage_factor": elastic_buckling_usage,
        "ultimate_usage_factor": 1.0 / max(reported_ultimate_capacity_factor, EPS),
        "raw_ultimate_usage_factor": 1.0 / max(raw_ultimate_capacity_factor, EPS),
        "raw_ultimate_capacity_factor": raw_ultimate_capacity_factor,
        "reported_ultimate_capacity_factor": reported_ultimate_capacity_factor,
        "ultimate_lifted_to_buckling_strength": ultimate_lifted_to_buckling_strength,
        "pressure_dominated_yield_limit": pressure_dominated_yield,
        "ultimate_included": include_ultimate_in_buckling_strength,
    }
    diagnostics = {
        "collapse_state": collapse_state,
        "capacity_factor": reported_ultimate_capacity_factor,
        "raw_capacity_factor": raw_ultimate_capacity_factor,
        "yield_capacity_factor": yield_capacity_factor,
        "ultimate_yield_includes_lateral_deformation": (
            config.include_lateral_deformation_in_ultimate_yield
        ),
        "global_elastic_cutoff_factor": global_elastic_cutoff_factor,
        "buckling": buckling,
        "pressure_preload_yield_utilization": pressure_yield,
        "pressure_preload_response": pressure_preload_response,
        "final_yield_utilization": final_yield,
        "max_newton_iterations": max_iterations,
        "mode_count": len(modes),
        "section": asdict(section),
        "stiffener_section": asdict(stiffener_section),
        "effective_stiffener_plate_width": effective_width,
        "load_components": normalized_load_components(panel),
        "support_model": SUPPORTED_IN_PLANE_SUPPORTS[panel.in_plane_support],
        "local_global_coupling": local_global_coupling,
        "continuation_geometric_coupling": _ritz_geometric_matrix(panel, modes)[1],
        "continuation": continuation,
        "buckling_strength": buckling_strength,
    }
    return _attach_reliability(
        S3Result(
            buckling_usage_factor=buckling_strength["usage_factor"],
            ultimate_usage_factor=buckling_strength["ultimate_usage_factor"],
            elastic_buckling_usage_factor=elastic_buckling_usage,
            valid=True,
            invalid_reason=None,
            diagnostics=diagnostics,
            covered_domain_notes=_notes(),
        ),
        panel,
        config,
        "S3",
        solve_s3_panel,
        validation_reasons,
    )


def solve_u3_panel(panel: U3PanelInput, config: S3SolverConfig | None = None) -> S3Result:
    """Solve the reduced U3 load path and return usage-factor diagnostics."""

    config = config or S3SolverConfig()
    validation_reasons = collect_u3_validation_reasons(panel, config)
    validation_error = validation_reasons[0] if validation_reasons else None
    if validation_error is not None:
        return _attach_reliability(
            _invalid_u3_result(validation_error),
            panel,
            config,
            "U3",
            solve_u3_panel,
            validation_reasons,
        )

    if all(
        abs(value) <= EPS
        for value in (
            panel.axial_stress,
            panel.mean_transverse_stress,
            panel.shear_stress,
        )
    ):
        return _attach_reliability(
            _invalid_u3_result("zero-variable-load"),
            panel,
            config,
            "U3",
            solve_u3_panel,
            ["zero-variable-load"],
        )

    modes = build_u3_ritz_modes(panel, config)
    amplitudes = [0.0 for _ in modes]
    amplitudes, pressure_converged, pressure_iterations = solve_equilibrium_amplitudes(
        panel,
        modes,
        0.0,
        amplitudes,
        config,
    )
    if not pressure_converged:
        return _attach_reliability(
            _invalid_u3_result(
                "non-convergence",
                {
                    "stage": "pressure-preload",
                    "iterations": pressure_iterations,
                },
            ),
            panel,
            config,
            "U3",
            solve_u3_panel,
            validation_reasons,
        )

    pressure_yield = u3_yield_utilization(panel, modes, amplitudes, 0.0, config)
    pressure_preload_response = {
        "iterations": pressure_iterations,
        "amplitudes": _mode_amplitude_summary(modes, amplitudes),
        "yield_utilization": pressure_yield,
        "controlling_yield_branch": "plate",
    }
    if pressure_yield["max"] >= config.pressure_yield_limit:
        return _attach_reliability(
            _invalid_u3_result(
                "pressure",
                {
                    "stage": "pressure-preload",
                    "yield_utilization": pressure_yield,
                    "pressure_preload_response": pressure_preload_response,
                    "pressure_iterations": pressure_iterations,
                },
            ),
            panel,
            config,
            "U3",
            solve_u3_panel,
            validation_reasons,
        )

    buckling = elastic_u3_buckling_factors(panel, modes, config)
    buckling_factor = buckling["critical_factor"]
    elastic_buckling_usage = None if buckling_factor is None else 1.0 / max(buckling_factor, EPS)

    previous_load = 0.0
    previous_yield = pressure_yield["max"]
    yield_capacity_factor: float | None = None
    max_iterations = pressure_iterations
    collapse_state = "first-yield"
    final_yield = pressure_yield

    accepted_steps = 0
    rejected_steps = 0
    cutbacks = 0
    min_accepted_step: float | None = None
    max_accepted_step: float | None = None
    current_step = min(
        max(config.initial_load_step, config.min_load_step, EPS),
        max(config.max_load_step, config.min_load_step, EPS),
    )
    cutback_factor = min(max(config.load_step_cutback, EPS), 0.95)

    while previous_load < config.max_load_factor - EPS:
        load_factor = min(previous_load + current_step, config.max_load_factor)
        attempted_step = load_factor - previous_load
        trial_amplitudes, converged, iterations = solve_equilibrium_amplitudes(
            panel,
            modes,
            load_factor,
            amplitudes,
            config,
        )
        max_iterations = max(max_iterations, iterations)
        if not converged:
            rejected_steps += 1
            next_step = current_step * cutback_factor
            if (
                next_step < max(config.min_load_step, EPS)
                or cutbacks >= config.max_load_step_cutbacks
            ):
                continuation = _continuation_summary(
                    accepted_steps,
                    rejected_steps,
                    cutbacks,
                    previous_load,
                    min_accepted_step,
                    max_accepted_step,
                    current_step,
                    config,
                )
                continuation["attempted_load_factor"] = load_factor
                continuation["attempted_step"] = attempted_step
                continuation["next_cutback_step"] = next_step
                continuation["newton_iterations"] = iterations
                continuation["cutback_exhausted"] = True
                return _attach_reliability(
                    _invalid_u3_result(
                        "non-convergence",
                        {
                            "stage": "in-plane-continuation",
                            "load_factor": load_factor,
                            "iterations": iterations,
                            "buckling": buckling,
                            "continuation": continuation,
                        },
                    ),
                    panel,
                    config,
                    "U3",
                    solve_u3_panel,
                    validation_reasons,
                )
            current_step = next_step
            cutbacks += 1
            continue

        amplitudes = trial_amplitudes
        accepted_steps += 1
        min_accepted_step = (
            attempted_step
            if min_accepted_step is None
            else min(min_accepted_step, attempted_step)
        )
        max_accepted_step = (
            attempted_step
            if max_accepted_step is None
            else max(max_accepted_step, attempted_step)
        )
        final_yield = u3_yield_utilization(panel, modes, amplitudes, load_factor, config)
        if final_yield["max"] >= config.yield_utilization_limit:
            yield_capacity_factor = _interpolate_capacity(
                previous_load,
                previous_yield,
                load_factor,
                final_yield["max"],
                config.yield_utilization_limit,
            )
            previous_load = load_factor
            break
        previous_load = load_factor
        previous_yield = final_yield["max"]
        if previous_load >= 2.0:
            current_step = min(current_step * config.load_step_growth, config.max_load_step)

    continuation = _continuation_summary(
        accepted_steps,
        rejected_steps,
        cutbacks,
        previous_load,
        min_accepted_step,
        max_accepted_step,
        current_step,
        config,
    )

    if yield_capacity_factor is None or yield_capacity_factor <= EPS:
        return _attach_reliability(
            _invalid_u3_result(
                "no-collapse-within-load-range",
                {
                    "max_load_factor": config.max_load_factor,
                    "buckling": buckling,
                    "yield_utilization": final_yield,
                },
            ),
            panel,
            config,
            "U3",
            solve_u3_panel,
            validation_reasons,
        )

    buckling_strength_limits = {"ultimate_capacity": yield_capacity_factor}
    if buckling_factor is not None:
        buckling_strength_limits["elastic_buckling_envelope"] = buckling_factor
    buckling_strength_control, buckling_strength_capacity_factor = min(
        buckling_strength_limits.items(),
        key=lambda item: float(item[1]),
    )
    raw_ultimate_capacity_factor = yield_capacity_factor
    reported_ultimate_capacity_factor = max(
        raw_ultimate_capacity_factor,
        buckling_strength_capacity_factor,
    )
    ultimate_lifted_to_buckling_strength = (
        reported_ultimate_capacity_factor > raw_ultimate_capacity_factor + EPS
    )
    buckling_strength = {
        "capacity_factor": buckling_strength_capacity_factor,
        "usage_factor": 1.0 / max(buckling_strength_capacity_factor, EPS),
        "controlling_limit": buckling_strength_control,
        "component_capacity_factors": buckling_strength_limits,
        "excluded_component_capacity_factors": {},
        "elastic_usage_factor": elastic_buckling_usage,
        "ultimate_usage_factor": 1.0 / max(reported_ultimate_capacity_factor, EPS),
        "raw_ultimate_usage_factor": 1.0 / max(raw_ultimate_capacity_factor, EPS),
        "raw_ultimate_capacity_factor": raw_ultimate_capacity_factor,
        "reported_ultimate_capacity_factor": reported_ultimate_capacity_factor,
        "ultimate_lifted_to_buckling_strength": ultimate_lifted_to_buckling_strength,
        "pressure_dominated_yield_limit": False,
        "ultimate_included": True,
    }
    diagnostics = {
        "panel_family": "U3",
        "collapse_state": collapse_state,
        "capacity_factor": reported_ultimate_capacity_factor,
        "raw_capacity_factor": raw_ultimate_capacity_factor,
        "yield_capacity_factor": yield_capacity_factor,
        "buckling": buckling,
        "pressure_preload_yield_utilization": pressure_yield,
        "pressure_preload_response": pressure_preload_response,
        "final_yield_utilization": final_yield,
        "max_newton_iterations": max_iterations,
        "mode_count": len(modes),
        "load_components": normalized_load_components(panel),
        "support_model": SUPPORTED_IN_PLANE_SUPPORTS[panel.in_plane_support],
        "rotational_support": {
            "x_edges": panel.rotational_support_1,
            "y_edges": panel.rotational_support_2,
        },
        "continuation_geometric_coupling": _ritz_geometric_matrix(panel, modes)[1],
        "continuation": continuation,
        "buckling_strength": buckling_strength,
    }
    return _attach_reliability(
        S3Result(
            buckling_usage_factor=buckling_strength["usage_factor"],
            ultimate_usage_factor=buckling_strength["ultimate_usage_factor"],
            elastic_buckling_usage_factor=elastic_buckling_usage,
            valid=True,
            invalid_reason=None,
            diagnostics=diagnostics,
            covered_domain_notes=_u3_notes(),
        ),
        panel,
        config,
        "U3",
        solve_u3_panel,
        validation_reasons,
    )
