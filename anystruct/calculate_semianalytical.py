"""Reduced semi-analytical PULS S3/U3 panel calculations.

This module is a first physics milestone for regular stiffened S3 panels and
unstiffened U3 panels.  It keeps the PULS CSV files as benchmark data and does
not fit corrections from them.  The implementation follows the direct
semi-analytical shape described in DNV-CG-0128 Sec.4:

* panel deflections are represented with Rayleigh-Ritz sine modes,
* a nonlinear equilibrium residual is traced along a proportional in-plane
  load path while lateral pressure remains fixed,
* elastic mode factors and a major-yield collapse check are reported as
  usage factors.

The production PULS code uses a richer element library and element validity
manual than the public guideline.  The result diagnostics therefore name the
covered assumptions instead of presenting this reduced model as PULS parity.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func=None, **kwargs):
        if func is not None:
            return func
        return lambda wrapped: wrapped


EPS = 1.0e-12
SUPPORTED_IN_PLANE_SUPPORTS = {
    "Integrated": "method-a",
    "Girder - long": "method-b-long",
    "Girder - trans": "method-b-trans",
}
SUPPORTED_STIFFENER_TYPES = {"T-bar", "L-bulb", "Angle", "Flatbar"}
SUPPORTED_STIFFENER_BOUNDARIES = {"Cont", "Sniped"}
SUPPORTED_ROTATIONAL_SUPPORTS = {"SS", "CL", "FS", ""}
CSV_SAMPLE_METHOD_FIRST_N = "first-n"
CSV_SAMPLE_METHOD_STRATIFIED = "stratified-shuffle"
CSV_SAMPLE_METHOD_FULL = "full"
DEFAULT_CSV_SAMPLE_SIZE = 1200
DEFAULT_CSV_FIXTURE_SAMPLE_SIZE = 64
DEFAULT_CSV_SAMPLE_SEED = 128
DEFAULT_CSV_MODE_CONVERGENCE_SAMPLE_SIZE = 200
CSV_STRATIFICATION_KEYS = (
    "target_validity",
    "stiffener_type",
    "support",
    "stiffener_boundary",
    "pressure_state",
    "pressure_band",
    "usage_region",
)
DEFAULT_SHIP_SECTION_INPUTS = (
    r"C:\Github\ANYstructure\anystruct\ship_section_example.txt",
    r"C:\Users\AudunArnesenNyhus\OneDrive - Cefront\Documents\OKEA side section.txt",
    r"C:\Users\AudunArnesenNyhus\OneDrive - Cefront\Documents\OKEA mid section.txt",
    r"C:\Users\AudunArnesenNyhus\OneDrive - Cefront\Documents\OKEA transversal section.txt",
)
DEFAULT_RELIABILITY_BASELINE = Path("reports/puls_reliability_baseline.json")
DEFAULT_CSR_SP_TRAINING_CSV = Path("Processed CSV 20211019-113209_SP_inc_CSR.csv")
DEFAULT_CSR_UP_TRAINING_CSV = Path("Processed CSV 20211019-133434_UP_inc_CSR.csv")
PULS_MANUAL_S3_LIMITS = {
    "source": r"C:\Program Files\DNV\NauticusHull 20.36.2508\Manuals\UserManual PULS.pdf",
    "manual_file_date": "2025-08-05",
    "plate_slenderness_max": 200.0,
    "aspect_ratio_min": 0.17,
    "aspect_ratio_max": 20.0,
    "flatbar_web_slenderness_max": 35.0,
    "open_profile_web_slenderness_max": 90.0,
    "free_flange_slenderness_max": 15.0,
    "min_flange_width_to_web_height": 0.22,
}
PULS_MANUAL_U3_LIMITS = {
    "source": PULS_MANUAL_S3_LIMITS["source"],
    "manual_file_date": PULS_MANUAL_S3_LIMITS["manual_file_date"],
    "plate_slenderness_max": 200.0,
    "long_to_short_aspect_ratio_max": 20.0,
}
CSR_RULE_REFERENCE = {
    "source": (
        r"C:\Users\AudunArnesenNyhus\OneDrive - Cefront\Desktop\Rules and standards"
        r"\Common Structural Rules.pdf"
    ),
    "edition": "Rules for the Classification of Steel Ships 2014, Pt 12 Common Structural Rules",
    "proportions_clause": "Sec 10/2.2.1 and Table 10.2.1",
    "advanced_buckling_clause": "Sec 10/4 and Appendix D",
}
CSR_PLATE_SLENDERNESS_COEFFICIENTS = {
    "hull_envelope_or_tank_boundary": 100.0,
    "other_structure": 125.0,
}
CSR_STIFFENER_WEB_COEFFICIENTS = {
    "T-bar": 75.0,
    "Angle": 75.0,
    "L-bulb": 41.0,
    "Flatbar": 22.0,
}
CSR_FLANGE_OUTSTAND_COEFFICIENT = 12.0
CSR_MIN_TOTAL_FLANGE_WIDTH_TO_WEB_HEIGHT = 0.25
DEFAULT_ANYSTRUCTURE_CSR_CORROSION_ADDITION_MM = 2.0


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
    plate_corrosion_addition: float = 0.0
    web_corrosion_addition: float = 0.0
    flange_corrosion_addition: float = 0.0

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
    plate_corrosion_addition: float = 0.0

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
    # Production-tolerance model imperfections per DNV-CG-0128 Sec.6 (based on
    # IACS Rec.47): plate imperfection b/200 of the plate breadth and stiffener
    # imperfection L/1000 of the stiffener length, applied to the critical mode
    # shape of each Ritz family.
    plate_imperfection_breadth_fraction: float = 1.0 / 200.0
    stiffener_imperfection_length_fraction: float = 1.0 / 1000.0
    nonlinear_membrane_factor: float = 0.75
    global_stiffened_strip_capacity_factor: float | None = None
    web_shear_interaction_exponent: float = 1.0
    local_plate_web_interaction_exponent: float = 1.20
    flanged_local_plate_restraint_factor: float = 1.10
    s3_shear_buckling_capacity_factor: float = 0.75
    use_effective_stiffener_width: bool = False
    sniped_eccentricity_factor: float = 1.2
    web_local_sniped_eccentricity_factor: float = 1.0
    torsional_imperfection_scale: float = 1.0
    torsional_restraint_factor: float = 0.65
    pressure_local_share: float = 0.0
    pressure_global_share: float = 1.0
    use_separate_s3_pressure_modes: bool = True
    s3_pressure_mode_stiffness_factor: float = 5.0
    include_pressure_dominated_yield_in_buckling_strength: bool = False
    # The Perry-type stiffener lateral-deformation amplification (assumed bow
    # L/1000 per DNV-CG-0128 Sec.6) is part of the PULS stiffener limit
    # states, so it participates in the ultimate yield path by default.
    include_lateral_deformation_in_ultimate_yield: bool = True
    include_global_curvature_in_plate_yield: bool = False
    pressure_dominated_yield_preload_ratio: float = 0.05
    s3_major_yield_reserve_factor: float = 1.0
    yield_utilization_limit: float = 1.0
    pressure_yield_limit: float = 1.0
    max_load_factor: float = 100.0
    initial_load_step: float = 0.05
    load_step_growth: float = 1.08
    max_load_step: float = 1.5
    min_load_step: float = 1.0e-4
    load_step_cutback: float = 0.5
    max_load_step_cutbacks: int = 12
    use_accelerated_s3_continuation: bool = False
    continuation_refinement_tolerance: float = 1.0e-5
    continuation_refinement_max_iterations: int = 10
    newton_max_iterations: int = 40
    newton_tolerance: float = 1.0e-7
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 12.0
    max_plate_slenderness: float = 250.0
    max_web_slenderness: float = 180.0
    max_flange_slenderness: float = 45.0
    max_web_to_flange_ratio: float = 5.0
    hot_spot_grid: tuple[float, ...] = (0.125, 0.25, 0.5, 0.75, 0.875)
    membrane_hot_spot_fractions: tuple[float, ...] = (
        0.0,
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        1.0,
    )
    check_mode_convergence: bool = False
    medium_longitudinal_modes: tuple[int, ...] = (1, 2, 3, 4)
    medium_transverse_modes: tuple[int, ...] = (1, 2, 3)
    high_longitudinal_modes: tuple[int, ...] = (1, 2, 3, 4, 5)
    high_transverse_modes: tuple[int, ...] = (1, 2, 3, 4)
    high_confidence_drift_limit: float = 0.05
    medium_confidence_drift_limit: float = 0.12
    include_solver_diagnostics: bool = True


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


@dataclass(frozen=True)
class MembraneField:
    """Second-order von Karman/Marguerre membrane stress field for a sine basis.

    For an added deflection ``w = sum_i q_i sin(kx_i x) sin(ky_i y)`` and an
    initial deflection ``w0`` in the same basis, the Marguerre compatibility
    equation gives the Airy stress function as a finite cosine-harmonic series

        nabla^4 F = E [ L(w + w0, w + w0) - L(w0, w0) ]

    with ``L(u, v) = u_xy v_xy - (u_xx v_yy + u_yy v_xx) / 2``.  Each harmonic
    ``h`` with wave numbers ``P = p pi / a`` and ``Q = q pi / b`` carries the
    quadratic amplitude

        A_h = (q + q0)^T G_h (q + q0) - q0^T G_h q0

    and the exact particular solution
    ``F_h = E A_h cos(P x) cos(Q y) / (P^2 + Q^2)^2``.  Both the redistributed
    membrane stresses and the membrane strain energy follow analytically from
    this solution with no empirical coefficients.  Stresses are stored in the
    compression-positive convention used by the panel inputs.
    """

    elastic_modulus: float
    thickness: float
    coupling: np.ndarray
    imperfection_offset: np.ndarray
    energy_matrix: np.ndarray
    energy_coupling: np.ndarray
    sigma_x_grid: np.ndarray
    sigma_y_grid: np.ndarray
    tau_grid: np.ndarray
    grid_x_fractions: np.ndarray
    grid_y_fractions: np.ndarray
    edge_mean_axial_factors: np.ndarray


@dataclass(frozen=True)
class CurvaturePoint:
    x_fraction: float
    y_fraction: float
    d2x: np.ndarray
    d2y: np.ndarray
    dxy: np.ndarray


@dataclass(frozen=True)
class RitzRuntime:
    modes: Sequence[RitzMode]
    linear: np.ndarray
    geometric: np.ndarray
    nonlinear: np.ndarray
    pressure: np.ndarray
    imperfection: np.ndarray
    max_delta: np.ndarray
    geometric_coupling: dict[str, Any]
    plate_curvature_points: tuple[CurvaturePoint, ...]
    global_centerline_curvature_points: tuple[CurvaturePoint, ...]
    plate_x_fractions: np.ndarray
    plate_y_fractions: np.ndarray
    plate_d2x: np.ndarray
    plate_d2y: np.ndarray
    plate_dxy: np.ndarray
    global_centerline_d2x: np.ndarray
    membrane: MembraneField | None = None


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


def _ship_section_optional_value(
    section: Mapping[str, Any],
    *keys: str,
    default: Any = "",
) -> Any:
    for key in keys:
        if key in section:
            return _ship_section_value(section, key)
    return default


def ship_section_record_to_csv_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one ANYstructure ship-section PULS result record to CSV-like S3 fields."""

    plate = record["Plate geometry"]
    stiffener = record["Primary stiffeners"]
    material = record["Material"]
    loads = record["Applied loads"]
    support = record.get("Bound cond.", {})
    buckling = record.get("Buckling strength", {})
    ultimate = record.get("Ultimate capacity", {})
    global_elastic = record.get("Global elastic buckling", {})
    local_elastic = record.get("Local elastic buckling", {})
    failure_modes = record.get("Failure modes", {})
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
        "PULS global axial stress": _ship_section_optional_value(
            global_elastic,
            "Axial stress",
        ),
        "PULS global trans stress": _ship_section_optional_value(
            global_elastic,
            "Trans. Stress",
            "Trans. stress",
        ),
        "PULS global shear stress": _ship_section_optional_value(
            global_elastic,
            "Shear stress",
        ),
        "PULS local axial stress": _ship_section_optional_value(
            local_elastic,
            "Axial stress",
        ),
        "PULS local trans stress": _ship_section_optional_value(
            local_elastic,
            "Trans. Stress",
            "Trans. stress",
        ),
        "PULS local shear stress": _ship_section_optional_value(
            local_elastic,
            "Shear stress",
        ),
        "PULS failure plate buckling percent": _ship_section_optional_value(
            failure_modes,
            "Plate buckling",
        ),
        "PULS failure global stiffener buckling percent": _ship_section_optional_value(
            failure_modes,
            "Global stiffener buckling",
        ),
        "PULS failure torsional stiffener buckling percent": _ship_section_optional_value(
            failure_modes,
            "Torsional stiffener buckling",
        ),
        "PULS failure web stiffener buckling percent": _ship_section_optional_value(
            failure_modes,
            "Web stiffener buckling",
        ),
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
    text = str(value or "").strip()
    normalized = text.lower().replace("_", " ").replace("  ", " ")
    return {
        "Int": "Integrated",
        "int": "Integrated",
        "Integrated": "Integrated",
        "integrated": "Integrated",
        "GL": "Girder - long",
        "gl": "Girder - long",
        "Girder - long": "Girder - long",
        "girder - long": "Girder - long",
        "girder long": "Girder - long",
        "GT": "Girder - trans",
        "gt": "Girder - trans",
        "Girder - trans": "Girder - trans",
        "girder - trans": "Girder - trans",
        "girder trans": "Girder - trans",
    }.get(text, {
        "int": "Integrated",
        "integrated": "Integrated",
        "gl": "Girder - long",
        "girder - long": "Girder - long",
        "girder long": "Girder - long",
        "gt": "Girder - trans",
        "girder - trans": "Girder - trans",
        "girder trans": "Girder - trans",
    }.get(normalized, text))


def _anystructure_up_boundary_edges(value: Any) -> tuple[str, str, str, str]:
    """Return UP rotational support codes for left, right, upper, lower edges."""

    text = str(value or "SSSS").strip().upper().replace("-", "").replace(" ", "")
    if text in {"CCCC", "CLCL", "CC", "CL"}:
        return "CL", "CL", "CL", "CL"
    if text in {"SSSS", "SS"}:
        return "SS", "SS", "SS", "SS"
    if text in {"FSFS", "FFFF", "FS"}:
        return "FS", "FS", "FS", "FS"
    if len(text) != 4:
        text = "SSSS"

    edges: list[str] = []
    for letter in text:
        if letter == "C":
            edges.append("CL")
        elif letter == "F":
            edges.append("FS")
        else:
            edges.append("SS")
    return edges[0], edges[1], edges[2], edges[3]


def _anystructure_rotational_supports(value: Any) -> tuple[str, str]:
    left, right, upper, lower = _anystructure_up_boundary_edges(value)

    def paired(first: str, second: str) -> str:
        if first == second and first in {"CL", "FS"}:
            return first
        return "SS"

    return paired(left, right), paired(upper, lower)


def _anystructure_selected_method(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"1", "buckling"}:
        return "buckling"
    if text in {"2", "ultimate"}:
        return "ultimate"
    return text


def _selected_anystructure_method(calc_object: Any, selected_method: Any = None) -> str:
    """Return the PULS result branch requested by ANYstructure optimization."""

    if selected_method is not None:
        return _anystructure_selected_method(selected_method)
    all_structure = calc_object[0] if isinstance(calc_object, (list, tuple)) else calc_object
    try:
        return _anystructure_selected_method(all_structure.Plate.get_puls_method())
    except Exception:
        return ""


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


def _main_dict_value(candidate: Any, names: Sequence[str]) -> float | None:
    try:
        main_dict = getattr(candidate, "_main_dict")
    except Exception:
        main_dict = None
    if not isinstance(main_dict, Mapping):
        return None
    normalized = {str(key).strip().lower(): value for key, value in main_dict.items()}
    for name in names:
        value = normalized.get(name.strip().lower())
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        parsed = _optional_float(value)
        if parsed is not None:
            return parsed
    return None


def _anystructure_corrosion_addition_mm(
    *candidates: Any,
    default_mm: float = DEFAULT_ANYSTRUCTURE_CSR_CORROSION_ADDITION_MM,
) -> float:
    """Return a full local CSR corrosion addition in mm.

    ANYstructure currently stores plate/stiffener geometry as gross scantlings.
    If a project object exposes an explicit corrosion addition, use it.  Values
    less than 0.05 are treated as metres; larger values are treated as mm.
    """

    names = (
        "corrosion_addition_mm",
        "corrosion_addition",
        "tcorr",
        "t_corr",
        "corr_add",
        "cor_add",
        "tk",
        "plate corrosion addition",
        "corrosion addition",
    )
    for candidate in candidates:
        if candidate is None:
            continue
        for name in names:
            try:
                parsed = _optional_float(getattr(candidate, name))
            except Exception:
                parsed = None
            if parsed is not None:
                if parsed <= EPS:
                    return 0.0
                return parsed * 1000.0 if parsed < 0.05 else parsed
        parsed = _main_dict_value(candidate, names)
        if parsed is not None:
            if parsed <= EPS:
                return 0.0
            return parsed * 1000.0 if parsed < 0.05 else parsed
    return max(float(default_mm), 0.0)


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
    plate_corrosion = _anystructure_corrosion_addition_mm(plate, all_structure)

    if sp_or_up == "SP" and stiffener is not None:
        puls_boundary = stiffener.get_puls_boundary()
        in_plane_support = _anystructure_in_plane_support(puls_boundary)
        sigxd = _anystructure_axial_design_stress(stiffener.sigma_x1, stiffener.sigma_x2)
        stiffener_corrosion = _anystructure_corrosion_addition_mm(stiffener, all_structure)
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
            axial_stress=0.0 if in_plane_support == "Girder - trans" else sigxd,
            transverse_stress_1=0.0 if in_plane_support == "Girder - long" else stiffener.sigma_y1,
            transverse_stress_2=0.0 if in_plane_support == "Girder - long" else stiffener.sigma_y2,
            shear_stress=stiffener.tau_xy,
            pressure=pressure_mpa,
            in_plane_support=in_plane_support,
            elastic_modulus=elastic_modulus,
            poisson_ratio=poisson_ratio,
            plate_corrosion_addition=plate_corrosion,
            web_corrosion_addition=stiffener_corrosion,
            flange_corrosion_addition=stiffener_corrosion,
        )

    up_boundary = plate.get_puls_up_boundary() if hasattr(plate, "get_puls_up_boundary") else "SSSS"
    rotational_1, rotational_2 = _anystructure_rotational_supports(up_boundary)
    puls_boundary = plate.get_puls_boundary() if hasattr(plate, "get_puls_boundary") else "GL"
    in_plane_support = _anystructure_in_plane_support(puls_boundary)
    return U3PanelInput(
        length=plate.span * 1000.0,
        width=plate.spacing,
        plate_thickness=plate.t,
        yield_stress_plate=plate.mat_yield / 1.0e6,
        axial_stress_1=0.0 if in_plane_support == "Girder - trans" else plate.sigma_x1,
        axial_stress_2=0.0 if in_plane_support == "Girder - trans" else plate.sigma_x2,
        transverse_stress_1=0.0 if in_plane_support == "Girder - long" else plate.sigma_y1,
        transverse_stress_2=0.0 if in_plane_support == "Girder - long" else plate.sigma_y2,
        shear_stress=plate.tau_xy,
        pressure=pressure_mpa,
        in_plane_support=in_plane_support,
        rotational_support_1=rotational_1,
        rotational_support_2=rotational_2,
        elastic_modulus=elastic_modulus,
        poisson_ratio=poisson_ratio,
        plate_corrosion_addition=plate_corrosion,
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
    csr_requirement = calculate_csr_requirement(panel)
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
        "csr_requirement": csr_requirement,
        "csr_vector": csr_requirement["csr_vector"],
        "csr_color": "green" if csr_requirement["within_csr_proportions"] else "red",
        "diagnostics": solved.diagnostics,
        "result": solved.to_dict(),
    }


def predict_anystructure_csr_requirement(
    calc_object: Any,
    lat_press: float = 0.0,
) -> tuple[list[float | int], str, dict[str, Any]]:
    """Return equation-based CSR flags for ANYstructure without running ML."""

    panel = anystructure_panel_input(calc_object, lat_press)
    if panel is None:
        diagnostics = {
            "within_csr_proportions": False,
            "csr_vector": [0, 0, 0, 0],
            "failed": ["unsupported-anystructure-input"],
            "unknown": [],
        }
        return [0, 0, 0, 0], "red", diagnostics
    diagnostics = calculate_csr_requirement(panel)
    color = "green" if diagnostics["within_csr_proportions"] else "red"
    return list(diagnostics["csr_vector"]), color, diagnostics


def _anystructure_fast_solver_config(config: S3SolverConfig | None) -> S3SolverConfig:
    base = config or S3SolverConfig()
    return replace(
        base,
        check_mode_convergence=False,
        include_solver_diagnostics=False,
    )


def _predict_anystructure_uf_core(
    calc_object: Any,
    lat_press: float,
    config: S3SolverConfig | None,
    selected_method: Any = None,
    default_acceptance: float = 0.87,
    cache: MutableMapping[Any, np.ndarray] | None = None,
) -> tuple[float, float, float, float | None]:
    all_structure = calc_object[0] if isinstance(calc_object, (list, tuple)) else calc_object
    material_factor = _anystructure_material_factor(all_structure)
    fallback_acceptance = _optional_float(default_acceptance)
    if fallback_acceptance is None:
        fallback_acceptance = 0.87
    acceptance_limit = fallback_acceptance if material_factor is None else 1.0 / material_factor
    panel = anystructure_panel_input(calc_object, lat_press)
    if panel is None:
        return float("inf"), float("inf"), 0.0, acceptance_limit

    fast_config = _anystructure_fast_solver_config(config)
    method = _selected_anystructure_method(calc_object, selected_method)
    cache_key = None
    if cache is not None:
        cache_key = _anystructure_optimization_cache_key(
            panel,
            method,
            acceptance_limit,
            fast_config,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return float(cached[0]), float(cached[1]), float(cached[2]), float(cached[3])

    if method == "buckling" and isinstance(panel, S3PanelInput):
        prefilter = _s3_buckling_early_reject_vector(
            panel,
            fast_config,
            acceptance_limit,
        )
        if prefilter is not None:
            if cache is not None and cache_key is not None:
                cache[cache_key] = prefilter.copy()
            return (
                float(prefilter[0]),
                float(prefilter[1]),
                float(prefilter[2]),
                float(prefilter[3]),
            )

    solved = (
        solve_s3_panel(panel, fast_config)
        if isinstance(panel, S3PanelInput)
        else solve_u3_panel(panel, fast_config)
    )
    if (
        solved.valid
        and solved.buckling_usage_factor is not None
        and solved.ultimate_usage_factor is not None
    ):
        vector = np.array(
            [
                float(solved.buckling_usage_factor),
                float(solved.ultimate_usage_factor),
                1.0,
                float(acceptance_limit),
            ],
            dtype=float,
        )
        if cache is not None and cache_key is not None:
            cache[cache_key] = vector.copy()
        return (
            float(solved.buckling_usage_factor),
            float(solved.ultimate_usage_factor),
            1.0,
            acceptance_limit,
        )
    vector = np.array([float("inf"), float("inf"), 0.0, float(acceptance_limit)], dtype=float)
    if cache is not None and cache_key is not None:
        cache[cache_key] = vector.copy()
    return float("inf"), float("inf"), 0.0, acceptance_limit


def predict_anystructure_uf(
    calc_object: Any,
    lat_press: float = 0.0,
    config: S3SolverConfig | None = None,
    selected_method: Any = None,
    cache: MutableMapping[Any, np.ndarray] | None = None,
) -> np.ndarray:
    """Return legacy ANYstructure vector [buckling UF, ultimate UF, valid].

    The factors are intentionally returned un-factored.  ANYstructure compares
    them against a separate PULS acceptance limit, typically 1 / material factor.
    """

    buckling, ultimate, valid, _ = _predict_anystructure_uf_core(
        calc_object,
        lat_press,
        config,
        selected_method=selected_method,
        cache=cache,
    )
    return np.array([buckling, ultimate, valid], dtype=float)


def predict_anystructure_uf_with_acceptance(
    calc_object: Any,
    lat_press: float = 0.0,
    config: S3SolverConfig | None = None,
    default_acceptance: float = 0.87,
    selected_method: Any = None,
    cache: MutableMapping[Any, np.ndarray] | None = None,
) -> np.ndarray:
    """Return [buckling UF, ultimate UF, valid, acceptance limit] for ANYstructure."""

    buckling, ultimate, valid, acceptance = _predict_anystructure_uf_core(
        calc_object,
        lat_press,
        config,
        selected_method=selected_method,
        default_acceptance=default_acceptance,
        cache=cache,
    )
    vector = np.array([float("inf"), float("inf"), 0.0, acceptance], dtype=float)
    if valid:
        vector[0] = buckling
        vector[1] = ultimate
        vector[2] = valid
    return vector


def predict_anystructure_uf_batch(
    items: Iterable[Any],
    config: S3SolverConfig | None = None,
    default_acceptance: float = 0.87,
    selected_method: Any = None,
    cache: MutableMapping[Any, np.ndarray] | None = None,
) -> np.ndarray:
    """Return optimization vectors for multiple ANYstructure candidates.

    Each item may be `(calc_object, lat_press)` or the optimization tuple
    `(calc_object, x, lat_press)`.  The returned columns are
    `[buckling_uf, ultimate_uf, valid_prediction, acceptance]`.
    """

    rows = list(items)
    result = np.full((len(rows), 4), float("inf"), dtype=float)
    fallback_acceptance = _optional_float(default_acceptance)
    if fallback_acceptance is None:
        fallback_acceptance = 0.87
    result[:, 2] = 0.0
    result[:, 3] = float(fallback_acceptance)
    local_cache: MutableMapping[Any, np.ndarray] = {} if cache is None else cache
    predictor = predict_anystructure_uf_with_acceptance
    for index, item in enumerate(rows):
        try:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                calc_object = item[0]
                lat_press = item[2]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                calc_object = item[0]
                lat_press = item[1]
            else:
                calc_object = item
                lat_press = 0.0
            result[index] = predictor(
                calc_object,
                lat_press,
                config=config,
                default_acceptance=fallback_acceptance,
                selected_method=selected_method,
                cache=local_cache,
            )
        except Exception:
            result[index, 2] = 0.0
            result[index, 3] = float(fallback_acceptance)
    return result


def _hashable_float(value: Any, digits: int = 9) -> float | str:
    parsed = _optional_float(value)
    if parsed is None:
        return str(value)
    if not math.isfinite(parsed):
        return str(parsed)
    return round(parsed, digits)


def _anystructure_optimization_cache_key(
    panel: S3PanelInput | U3PanelInput,
    selected_method: str,
    acceptance_limit: float,
    config: S3SolverConfig,
) -> tuple[Any, ...]:
    return (
        type(panel).__name__,
        tuple(
            (name, _hashable_float(value) if isinstance(value, (int, float)) else str(value))
            for name, value in panel.__dict__.items()
        ),
        selected_method,
        _hashable_float(acceptance_limit),
        tuple(config.longitudinal_modes),
        tuple(config.transverse_modes),
        tuple(config.web_longitudinal_modes),
        tuple(config.web_depth_modes),
        _hashable_float(config.global_stiffened_strip_capacity_factor),
        _hashable_float(config.s3_shear_buckling_capacity_factor),
    )


def _s3_buckling_early_reject_vector(
    panel: S3PanelInput,
    config: S3SolverConfig,
    acceptance_limit: float,
) -> np.ndarray | None:
    """Return an optimization rejection vector when elastic buckling already fails.

    This is intentionally a one-way filter.  It never accepts a candidate and it
    never changes public solver results; it only skips continuation for
    buckling-method optimization candidates whose elastic buckling envelope is
    already above the PULS acceptance limit.
    """

    validation_reasons = collect_s3_validation_reasons(panel, config)
    if validation_reasons:
        return np.array([float("inf"), float("inf"), 0.0, float(acceptance_limit)], dtype=float)
    if all(
        abs(value) <= EPS
        for value in (
            panel.axial_stress,
            panel.mean_transverse_stress,
            panel.shear_stress,
        )
    ):
        return np.array([float("inf"), float("inf"), 0.0, float(acceptance_limit)], dtype=float)

    section = build_section_properties(panel)
    stiffener_section, _ = build_effective_stiffener_section(panel, config)
    modes = build_ritz_modes(panel, section, config)
    runtime = _build_ritz_runtime(panel, modes, config)
    amplitudes, pressure_converged, _ = solve_equilibrium_amplitudes(
        panel,
        modes,
        0.0,
        [0.0 for _ in modes],
        config,
        runtime,
    )
    if not pressure_converged:
        return np.array([float("inf"), float("inf"), 0.0, float(acceptance_limit)], dtype=float)

    pressure_yield = yield_utilization(
        panel,
        section,
        stiffener_section,
        modes,
        amplitudes,
        0.0,
        config,
        runtime=runtime,
    )
    if (
        panel.pressure > s3_pressure_capacity_limits(panel, section)["minimum"]
        or pressure_yield["max"] >= config.pressure_yield_limit
    ):
        return np.array([float("inf"), float("inf"), 0.0, float(acceptance_limit)], dtype=float)

    buckling = elastic_buckling_factors(panel, section, modes, config, stiffener_section)
    buckling_factor = buckling["critical_factor"]
    if buckling_factor is None or buckling_factor <= EPS:
        return None
    elastic_usage = 1.0 / max(float(buckling_factor), EPS)
    if elastic_usage >= float(acceptance_limit):
        return np.array([elastic_usage, float("inf"), 1.0, float(acceptance_limit)], dtype=float)
    return None


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


def _limit_check(value: float | None, *, maximum: float | None = None, minimum: float | None = None) -> bool | None:
    if value is None:
        return None
    if maximum is not None and value > maximum:
        return False
    if minimum is not None and value < minimum:
        return False
    return True


def _csr_strength_scale(yield_stress: float) -> float:
    """Return the CSR slenderness scaling sqrt(235 / sigma_yd)."""

    if yield_stress <= EPS or not math.isfinite(yield_stress):
        return float("nan")
    return math.sqrt(235.0 / yield_stress)


def _csr_check_from_ratio(
    ratio: float | None,
    coefficient: float | None,
    yield_stress: float,
) -> tuple[bool | None, float | None]:
    if ratio is None or coefficient is None:
        return None, None
    scale = _csr_strength_scale(yield_stress)
    if not math.isfinite(scale):
        return None, None
    limit = coefficient * scale
    return ratio <= limit, limit


def _net_thickness(gross_thickness: float, corrosion_addition: float) -> float | None:
    if gross_thickness <= EPS:
        return None
    net = gross_thickness - max(corrosion_addition, 0.0)
    return net if net > EPS else None


def _csr_flange_outstand(panel: S3PanelInput) -> float | None:
    """Return the flange outstand breadth used by CSR Table 10.2.1.

    The S3 input surface stores one flange breadth.  T-bars are treated as
    symmetric, while angles are treated conservatively as one-sided outstands.
    Bulb and flat-bar profiles do not use the angle/T flange outstand check.
    """

    if panel.stiffener_type == "T-bar":
        return max(0.5 * (panel.flange_width - panel.web_thickness), 0.0)
    if panel.stiffener_type == "Angle":
        return max(panel.flange_width - 0.5 * panel.web_thickness, 0.0)
    return None


def calculate_csr_requirement(
    panel: S3PanelInput | U3PanelInput,
    *,
    plate_location: str = "hull_envelope_or_tank_boundary",
) -> dict[str, Any]:
    """Return equation-based CSR proportion diagnostics for SemiAnalytical use.

    This deliberately evaluates the prescriptive CSR proportion checks outside
    the PULS/SemiAnalytical usage-factor solver.  Lengths, spacing, stiffener
    depth, and flange breadth are gross/scantling dimensions.  Thickness checks
    use net thickness after the panel corrosion-addition fields are deducted.
    """

    plate_coefficient = CSR_PLATE_SLENDERNESS_COEFFICIENTS.get(plate_location)
    width = min(panel.length, panel.width) if isinstance(panel, U3PanelInput) else panel.width
    plate_net_thickness = _net_thickness(panel.plate_thickness, panel.plate_corrosion_addition)
    plate_slenderness = width / plate_net_thickness if plate_net_thickness is not None else None
    plate_ok, plate_limit = _csr_check_from_ratio(
        plate_slenderness,
        plate_coefficient,
        panel.yield_stress_plate,
    )
    checks: dict[str, bool | None] = {"plate_slenderness": plate_ok}
    values: dict[str, float | None] = {
        "plate_slenderness": plate_slenderness,
        "plate_slenderness_limit": plate_limit,
        "plate_gross_thickness": panel.plate_thickness,
        "plate_corrosion_addition": panel.plate_corrosion_addition,
        "plate_net_thickness": plate_net_thickness,
    }
    limits: dict[str, float | None] = {
        "plate_coefficient": plate_coefficient,
        "plate_yield_scale": (
            _csr_strength_scale(panel.yield_stress_plate)
            if panel.yield_stress_plate > EPS
            else None
        ),
    }
    csr_vector: list[float | int] = [
        1 if plate_ok is True else 0,
        float("inf"),
        float("inf"),
        float("inf"),
    ]
    notes = [
        "CSR Sec 10 proportion checks are diagnostic only and do not alter SemiAnalytical UF results",
        "gross breadth/depth dimensions are used with net thickness after deducting the corrosion addition",
    ]

    if isinstance(panel, S3PanelInput):
        web_coefficient = CSR_STIFFENER_WEB_COEFFICIENTS.get(panel.stiffener_type)
        web_net_thickness = _net_thickness(panel.web_thickness, panel.web_corrosion_addition)
        web_slenderness = (
            panel.stiffener_height / web_net_thickness
            if web_net_thickness is not None
            else None
        )
        web_ok, web_limit = _csr_check_from_ratio(
            web_slenderness,
            web_coefficient,
            panel.yield_stress_stiffener,
        )

        flange_outstand = _csr_flange_outstand(panel)
        flange_net_thickness = _net_thickness(
            panel.flange_thickness,
            panel.flange_corrosion_addition,
        )
        flange_slenderness = (
            flange_outstand / flange_net_thickness
            if flange_outstand is not None and flange_net_thickness is not None
            else None
        )
        flange_coefficient = (
            CSR_FLANGE_OUTSTAND_COEFFICIENT
            if panel.stiffener_type in {"T-bar", "Angle"}
            else None
        )
        flange_ok, flange_limit = _csr_check_from_ratio(
            flange_slenderness,
            flange_coefficient,
            panel.yield_stress_stiffener,
        )

        total_flange_ratio = None
        web_flange_ok = True
        if panel.stiffener_type in {"T-bar", "Angle"}:
            total_flange_ratio = panel.flange_width / max(panel.stiffener_height, EPS)
            web_flange_ok = total_flange_ratio >= CSR_MIN_TOTAL_FLANGE_WIDTH_TO_WEB_HEIGHT
        elif panel.stiffener_type in {"Flatbar", "L-bulb"}:
            flange_ok = True
        else:
            web_flange_ok = None

        checks.update(
            {
                "web_slenderness": web_ok,
                "web_flange_ratio": web_flange_ok,
                "flange_outstand_slenderness": flange_ok,
            }
        )
        values.update(
            {
                "web_slenderness": web_slenderness,
                "web_slenderness_limit": web_limit,
                "web_gross_thickness": panel.web_thickness,
                "web_corrosion_addition": panel.web_corrosion_addition,
                "web_net_thickness": web_net_thickness,
                "flange_outstand": flange_outstand,
                "flange_outstand_slenderness": flange_slenderness,
                "flange_outstand_slenderness_limit": flange_limit,
                "flange_gross_thickness": panel.flange_thickness,
                "flange_corrosion_addition": panel.flange_corrosion_addition,
                "flange_net_thickness": flange_net_thickness,
                "total_flange_width_to_web_height": total_flange_ratio,
                "min_total_flange_width_to_web_height": (
                    CSR_MIN_TOTAL_FLANGE_WIDTH_TO_WEB_HEIGHT
                    if panel.stiffener_type in {"T-bar", "Angle"}
                    else None
                ),
            }
        )
        limits.update(
            {
                "web_coefficient": web_coefficient,
                "flange_outstand_coefficient": flange_coefficient,
                "stiffener_yield_scale": (
                    _csr_strength_scale(panel.yield_stress_stiffener)
                    if panel.yield_stress_stiffener > EPS
                    else None
                ),
            }
        )
        csr_vector = [
            1 if plate_ok is True else 0,
            1 if web_ok is True else 0,
            1 if web_flange_ok is True else 0,
            1 if flange_ok is True else 0,
        ]
        if panel.stiffener_type == "L-bulb":
            notes.append("L-bulb flange/bulb geometry is represented by the CSR bulb web coefficient; flange checks are not applied")
        if panel.stiffener_type == "Flatbar":
            notes.append("Flatbar has no separate flange outstand or web/flange ratio check")

    failed = [name for name, passed in checks.items() if passed is False]
    unknown = [name for name, passed in checks.items() if passed is None]
    return {
        "source": CSR_RULE_REFERENCE["source"],
        "edition": CSR_RULE_REFERENCE["edition"],
        "panel_family": "S3" if isinstance(panel, S3PanelInput) else "U3",
        "basis": {
            "proportions": CSR_RULE_REFERENCE["proportions_clause"],
            "advanced_buckling": CSR_RULE_REFERENCE["advanced_buckling_clause"],
        },
        "plate_location": plate_location,
        "limits": limits,
        "values": values,
        "checks": checks,
        "failed": failed,
        "unknown": unknown,
        "within_csr_proportions": not failed and not unknown,
        "csr_vector": csr_vector,
        "notes": notes,
    }


def _puls_manual_reference_domain(panel: S3PanelInput | U3PanelInput) -> dict[str, Any]:
    """Return source-backed PULS manual limit diagnostics without gating solves."""

    if isinstance(panel, S3PanelInput):
        limits = PULS_MANUAL_S3_LIMITS
        aspect_ratio = panel.length / panel.width if panel.width > EPS else None
        plate_slenderness = panel.width / panel.plate_thickness if panel.plate_thickness > EPS else None
        web_slenderness = (
            panel.stiffener_height / panel.web_thickness
            if panel.web_thickness > EPS
            else None
        )
        web_limit = (
            limits["flatbar_web_slenderness_max"]
            if panel.stiffener_type == "Flatbar"
            else limits["open_profile_web_slenderness_max"]
        )
        flange_slenderness = None
        flange_width_to_web_height = None
        if panel.stiffener_type != "Flatbar" and panel.flange_width > EPS and panel.flange_thickness > EPS:
            flange_slenderness = panel.flange_width / panel.flange_thickness
            flange_width_to_web_height = panel.flange_width / max(panel.stiffener_height, EPS)
        checks = {
            "plate_slenderness": _limit_check(plate_slenderness, maximum=limits["plate_slenderness_max"]),
            "aspect_ratio": _limit_check(
                aspect_ratio,
                minimum=limits["aspect_ratio_min"],
                maximum=limits["aspect_ratio_max"],
            ),
            "web_slenderness": _limit_check(web_slenderness, maximum=web_limit),
            "free_flange_slenderness": (
                True
                if panel.stiffener_type == "Flatbar"
                else _limit_check(flange_slenderness, maximum=limits["free_flange_slenderness_max"])
            ),
            "flange_width_to_web_height": (
                True
                if panel.stiffener_type == "Flatbar"
                else _limit_check(
                    flange_width_to_web_height,
                    minimum=limits["min_flange_width_to_web_height"],
                )
            ),
        }
        failed = [name for name, passed in checks.items() if passed is False]
        unknown = [name for name, passed in checks.items() if passed is None]
        return {
            "source": limits["source"],
            "manual_file_date": limits["manual_file_date"],
            "panel_family": "S3",
            "note": "manual reference limits are diagnostic only; solver covered-domain gates are unchanged",
            "limits": {key: value for key, value in limits.items() if key not in {"source", "manual_file_date"}},
            "values": {
                "aspect_ratio": aspect_ratio,
                "plate_slenderness": plate_slenderness,
                "web_slenderness": web_slenderness,
                "web_slenderness_limit": web_limit,
                "free_flange_slenderness": flange_slenderness,
                "flange_width_to_web_height": flange_width_to_web_height,
            },
            "checks": checks,
            "failed": failed,
            "unknown": unknown,
            "within_manual_limits": not failed and not unknown,
        }

    limits = PULS_MANUAL_U3_LIMITS
    short_side = min(panel.length, panel.width)
    long_side = max(panel.length, panel.width)
    long_to_short = long_side / max(short_side, EPS)
    plate_slenderness = short_side / panel.plate_thickness if panel.plate_thickness > EPS else None
    checks = {
        "plate_slenderness": _limit_check(plate_slenderness, maximum=limits["plate_slenderness_max"]),
        "long_to_short_aspect_ratio": _limit_check(
            long_to_short,
            maximum=limits["long_to_short_aspect_ratio_max"],
        ),
    }
    failed = [name for name, passed in checks.items() if passed is False]
    unknown = [name for name, passed in checks.items() if passed is None]
    return {
        "source": limits["source"],
        "manual_file_date": limits["manual_file_date"],
        "panel_family": "U3",
        "note": "manual reference limits are diagnostic only; solver covered-domain gates are unchanged",
        "limits": {key: value for key, value in limits.items() if key not in {"source", "manual_file_date"}},
        "values": {
            "long_to_short_aspect_ratio": long_to_short,
            "plate_slenderness": plate_slenderness,
        },
        "checks": checks,
        "failed": failed,
        "unknown": unknown,
        "within_manual_limits": not failed and not unknown,
    }


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
        "puls_manual_reference": _puls_manual_reference_domain(panel),
        "csr_equation_reference": calculate_csr_requirement(panel),
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
) -> tuple[float, float, float, float, float]:
    kx = m * math.pi / panel.length
    ky = n * math.pi / panel.width
    area_factor = panel.length * panel.width / 4.0
    bending = (
        stiffness.d11 * kx**4
        + 2.0 * (stiffness.d12 + 2.0 * stiffness.d66) * kx * kx * ky * ky
        + stiffness.d22 * ky**4
    )

    loads = normalized_load_components(panel)
    # The axial stress acts on the full effective section of the family field:
    # plate thickness for local/isotropic plate modes and the smeared section
    # area per unit width for the global stiffened-strip modes, so the
    # stiffener axial force is destabilizing for the global mode.  Transverse
    # and shear loads are carried by the plating alone.
    axial_resultant = (
        max(loads["signed_axial_stress"], 0.0) * stiffness.membrane_thickness
    )
    geometric = (
        axial_resultant * kx * kx
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


_PARITY_COS_COS = 0
_PARITY_COS_SIN = 1


def _half_range_cos_sin_integral(cos_count: int, sin_count: int, span: float) -> float:
    """Return ``int_0^L cos(nc pi y / L) sin(ns pi y / L) dy`` exactly."""

    if sin_count == 0 or cos_count == sin_count or (cos_count + sin_count) % 2 == 0:
        return 0.0
    return (
        2.0
        * span
        * sin_count
        / (math.pi * (sin_count * sin_count - cos_count * cos_count))
    )


def _membrane_energy_matrix(
    keys: Sequence[tuple[int, int, int]],
    length: float,
    width: float,
    thickness: float,
    elastic_modulus: float,
    poisson_ratio: float,
) -> np.ndarray:
    """Return the exact membrane-energy Gram matrix over the Airy harmonics.

    Each harmonic carries the unit-amplitude stress field of its particular
    solution.  Same-parity harmonics are orthogonal unless identical, while
    cos*cos and cos*sin harmonics couple in the energy whenever they share the
    x harmonic and their y harmonic counts have an odd sum, through the exact
    half-range integrals.  ``U = A^T M A / 2`` then replaces the diagonal
    coefficient form without approximation.
    """

    count = len(keys)
    matrix = np.zeros((count, count))
    p_waves = [key[0] * math.pi / length for key in keys]
    q_waves = [key[1] * math.pi / width for key in keys]
    inverse_biharmonic = [
        1.0 / max((p * p + q * q) ** 2, EPS)
        for p, q in zip(p_waves, q_waves)
    ]
    sigma_x_amp = [
        elastic_modulus * q * q * inv for q, inv in zip(q_waves, inverse_biharmonic)
    ]
    sigma_y_amp = [
        elastic_modulus * p * p * inv for p, inv in zip(p_waves, inverse_biharmonic)
    ]
    tau_amp = [
        elastic_modulus
        * p
        * q
        * inv
        * (1.0 if key[2] == _PARITY_COS_COS else -1.0)
        for key, p, q, inv in zip(keys, p_waves, q_waves, inverse_biharmonic)
    ]

    for h in range(count):
        p_h, q_h, parity_h = keys[h]
        for k in range(h, count):
            p_k, q_k, parity_k = keys[k]
            if p_h != p_k:
                continue
            # x integrals over [0, length]
            ix_cos = length if p_h == 0 else 0.5 * length
            ix_sin = 0.0 if p_h == 0 else 0.5 * length
            # y integrals over [0, width] for the sigma (Y1) and tau (Y2)
            # shape pairs.
            if parity_h == parity_k:
                if q_h != q_k:
                    continue
                if parity_h == _PARITY_COS_COS:
                    iy_sigma = width if q_h == 0 else 0.5 * width
                    iy_tau = 0.0 if q_h == 0 else 0.5 * width
                else:
                    iy_sigma = 0.5 * width
                    iy_tau = 0.5 * width
            else:
                if parity_h == _PARITY_COS_COS:
                    cos_q, sin_q = q_h, q_k
                else:
                    cos_q, sin_q = q_k, q_h
                # sigma shapes: cos(cos_q) * sin(sin_q); tau shapes:
                # sin(cos-parity q) * cos(sin-parity q).
                iy_sigma = _half_range_cos_sin_integral(cos_q, sin_q, width)
                iy_tau = _half_range_cos_sin_integral(sin_q, cos_q, width)
                if iy_sigma == 0.0 and iy_tau == 0.0:
                    continue
            normal_part = (
                sigma_x_amp[h] * sigma_x_amp[k]
                + sigma_y_amp[h] * sigma_y_amp[k]
                - poisson_ratio
                * (sigma_x_amp[h] * sigma_y_amp[k] + sigma_y_amp[h] * sigma_x_amp[k])
            )
            shear_part = 2.0 * (1.0 + poisson_ratio) * tau_amp[h] * tau_amp[k]
            value = (thickness / elastic_modulus) * (
                normal_part * ix_cos * iy_sigma + shear_part * ix_sin * iy_tau
            )
            matrix[h, k] = value
            matrix[k, h] = value
    return matrix


def build_membrane_field(
    modes: Sequence[RitzMode],
    length: float,
    width: float,
    thickness: float,
    elastic_modulus: float,
    imperfection: Sequence[float],
    grid_fractions: Sequence[float],
    poisson_ratio: float = 0.3,
) -> MembraneField | None:
    """Build the exact second-order membrane field for the Ritz basis.

    Sine-sine modes (``n >= 1``) expand pairwise into the four cos*cos
    harmonics ``(|m_i - m_j|, |n_i - n_j|)`` ... ``(m_i + m_j, n_i + n_j)``
    with closed-form coefficients; the constant ``(0, 0)`` harmonic always
    cancels, which is the analytic statement that the classical particular
    solution keeps straight panel edges with no induced mean stress.

    Cylindrical modes (``n == 0``, y-uniform ``sin(kx x)`` shapes) have
    identically zero self-coupling (a developable surface produces no
    second-order membrane stress), while their mixed pairs with sine-sine
    modes expand into cos*sin harmonics

        L(cyl_k, ss_i) = -kx_k^2 ky_i^2 / 4 *
            [cos(|m_k - m_i| pi x / a) - cos((m_k + m_i) pi x / a)] *
            sin(n_i pi y / b)

    which is the analytic carrier of the local-buckling to global-deflection
    membrane interaction.
    """

    count = len(modes)
    if count == 0:
        return None
    harmonics: dict[tuple[int, int, int], np.ndarray] = {}

    def accumulate(p: int, q: int, parity: int, i: int, j: int, coefficient: float) -> None:
        if (p == 0 and q == 0) or abs(coefficient) <= EPS:
            return
        matrix = harmonics.setdefault((p, q, parity), np.zeros((count, count)))
        matrix[i, j] += coefficient
        if i != j:
            matrix[j, i] += coefficient

    for i, first in enumerate(modes):
        for j in range(i, count):
            second = modes[j]
            if first.n == 0 and second.n == 0:
                continue
            if first.n > 0 and second.n > 0:
                cross = first.kx * first.ky * second.kx * second.ky
                normal = 0.5 * (
                    first.kx**2 * second.ky**2 + second.kx**2 * first.ky**2
                )
                m_diff, m_sum = abs(first.m - second.m), first.m + second.m
                n_diff, n_sum = abs(first.n - second.n), first.n + second.n
                accumulate(m_diff, n_diff, _PARITY_COS_COS, i, j, 0.25 * (cross - normal))
                accumulate(m_diff, n_sum, _PARITY_COS_COS, i, j, 0.25 * (cross + normal))
                accumulate(m_sum, n_diff, _PARITY_COS_COS, i, j, 0.25 * (cross + normal))
                accumulate(m_sum, n_sum, _PARITY_COS_COS, i, j, 0.25 * (cross - normal))
                continue
            cylinder, sine = (first, second) if first.n == 0 else (second, first)
            mixed = 0.25 * cylinder.kx**2 * sine.ky**2
            accumulate(abs(cylinder.m - sine.m), sine.n, _PARITY_COS_SIN, i, j, -mixed)
            accumulate(cylinder.m + sine.m, sine.n, _PARITY_COS_SIN, i, j, mixed)
    if not harmonics:
        return None

    keys = sorted(harmonics)
    coupling = np.stack([harmonics[key] for key in keys])
    p_waves = np.asarray([key[0] * math.pi / length for key in keys])
    q_waves = np.asarray([key[1] * math.pi / width for key in keys])
    wave_sq = p_waves**2 + q_waves**2
    inverse_biharmonic = 1.0 / np.maximum(wave_sq**2, EPS)
    is_cos_cos = np.asarray([key[2] == _PARITY_COS_COS for key in keys])
    energy_matrix = _membrane_energy_matrix(
        keys,
        length,
        width,
        thickness,
        elastic_modulus,
        poisson_ratio,
    )
    energy_coupling = np.einsum("hl,lij->hij", energy_matrix, coupling)

    imperfection_array = np.asarray(imperfection, dtype=float)
    if imperfection_array.shape != (count,):
        imperfection_array = np.zeros(count, dtype=float)
    imperfection_offset = np.einsum(
        "hij,i,j->h", coupling, imperfection_array, imperfection_array
    )

    fractions = np.asarray(grid_fractions, dtype=float)
    grid_x = np.repeat(fractions, len(fractions))
    grid_y = np.tile(fractions, len(fractions))
    p_counts = np.asarray([key[0] for key in keys], dtype=float)
    q_counts = np.asarray([key[1] for key in keys], dtype=float)
    cos_px = np.cos(math.pi * np.outer(grid_x, p_counts))
    sin_px = np.sin(math.pi * np.outer(grid_x, p_counts))
    cos_qy = np.cos(math.pi * np.outer(grid_y, q_counts))
    sin_qy = np.sin(math.pi * np.outer(grid_y, q_counts))
    sigma_shape = cos_px * np.where(is_cos_cos, cos_qy, sin_qy)
    tau_shape = sin_px * np.where(is_cos_cos, sin_qy, -cos_qy)
    # Compression-positive stress factors: the tension-convention solution
    # sigma_x = F_yy concentrates compression at the supported edges; flipping
    # all three components preserves the von Mises stress while matching the
    # panel input sign convention.  The cos*sin tau picks up the opposite sign
    # from the y derivative of its shape, absorbed into tau_shape above.
    sigma_x_grid = sigma_shape * (elastic_modulus * q_waves**2 * inverse_biharmonic)
    sigma_y_grid = sigma_shape * (elastic_modulus * p_waves**2 * inverse_biharmonic)
    tau_grid = tau_shape * (elastic_modulus * p_waves * q_waves * inverse_biharmonic)
    # Averaging sigma_x2 along a junction line y = const keeps only the p = 0
    # cos*cos harmonics; the cos*sin harmonics vanish identically at both
    # junction lines (sin(0) = sin(q pi) = 0).  Rows are the two long edges
    # y = 0 and y = b, where the redistribution sheds plate load into the
    # stiffeners.
    p_zero = (p_counts == 0.0) & is_cos_cos
    edge_factor = elastic_modulus * q_waves**2 * inverse_biharmonic
    edge_mean_axial_factors = np.vstack(
        [
            np.where(p_zero, edge_factor, 0.0),
            np.where(p_zero, edge_factor * np.cos(math.pi * q_counts), 0.0),
        ]
    )
    return MembraneField(
        elastic_modulus=elastic_modulus,
        thickness=thickness,
        coupling=coupling,
        imperfection_offset=imperfection_offset,
        energy_matrix=energy_matrix,
        energy_coupling=energy_coupling,
        sigma_x_grid=sigma_x_grid,
        sigma_y_grid=sigma_y_grid,
        tau_grid=tau_grid,
        grid_x_fractions=grid_x,
        grid_y_fractions=grid_y,
        edge_mean_axial_factors=edge_mean_axial_factors,
    )


@njit(cache=True)
def _membrane_amplitudes_kernel(
    coupling: np.ndarray,
    imperfection_offset: np.ndarray,
    total_deflection: np.ndarray,
) -> np.ndarray:
    harmonics = coupling.shape[0]
    modes = total_deflection.shape[0]
    result = np.empty(harmonics, dtype=np.float64)
    for harmonic in range(harmonics):
        value = 0.0
        for i in range(modes):
            qi = total_deflection[i]
            for j in range(modes):
                value += coupling[harmonic, i, j] * qi * total_deflection[j]
        result[harmonic] = value - imperfection_offset[harmonic]
    return result


@njit(cache=True)
def _membrane_force_kernel(
    energy_coupling: np.ndarray,
    total_deflection: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    harmonics = energy_coupling.shape[0]
    modes = total_deflection.shape[0]
    result = np.zeros(modes, dtype=np.float64)
    for i in range(modes):
        value = 0.0
        for harmonic in range(harmonics):
            weighted = amplitudes[harmonic]
            for j in range(modes):
                value += weighted * energy_coupling[harmonic, i, j] * total_deflection[j]
        result[i] = 2.0 * value
    return result


@njit(cache=True)
def _membrane_tangent_kernel(
    coupling: np.ndarray,
    imperfection_offset: np.ndarray,
    energy_matrix: np.ndarray,
    energy_coupling: np.ndarray,
    total_deflection: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    harmonics = coupling.shape[0]
    modes = total_deflection.shape[0]
    gradients = np.zeros((harmonics, modes), dtype=np.float64)
    for harmonic in range(harmonics):
        for i in range(modes):
            value = 0.0
            for j in range(modes):
                value += coupling[harmonic, i, j] * total_deflection[j]
            gradients[harmonic, i] = value

    weighted_gradients = np.zeros((harmonics, modes), dtype=np.float64)
    for harmonic in range(harmonics):
        for i in range(modes):
            value = 0.0
            for other in range(harmonics):
                value += energy_matrix[harmonic, other] * gradients[other, i]
            weighted_gradients[harmonic, i] = value

    tangent = np.zeros((modes, modes), dtype=np.float64)
    for i in range(modes):
        for j in range(modes):
            value = 0.0
            for harmonic in range(harmonics):
                value += gradients[harmonic, i] * weighted_gradients[harmonic, j]
            tangent[i, j] = 4.0 * value

    for i in range(modes):
        for j in range(modes):
            value = 0.0
            for harmonic in range(harmonics):
                value += amplitudes[harmonic] * energy_coupling[harmonic, i, j]
            tangent[i, j] += 2.0 * value
    return tangent


@njit(cache=True)
def _membrane_stress_components_kernel(
    sigma_x_grid: np.ndarray,
    sigma_y_grid: np.ndarray,
    tau_grid: np.ndarray,
    amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = sigma_x_grid.shape[0]
    harmonics = amplitudes.shape[0]
    sigma_x = np.zeros(points, dtype=np.float64)
    sigma_y = np.zeros(points, dtype=np.float64)
    tau = np.zeros(points, dtype=np.float64)
    for point in range(points):
        sx = 0.0
        sy = 0.0
        txy = 0.0
        for harmonic in range(harmonics):
            amplitude = amplitudes[harmonic]
            sx += sigma_x_grid[point, harmonic] * amplitude
            sy += sigma_y_grid[point, harmonic] * amplitude
            txy += tau_grid[point, harmonic] * amplitude
        sigma_x[point] = sx
        sigma_y[point] = sy
        tau[point] = txy
    return sigma_x, sigma_y, tau


@njit(cache=True)
def _max_von_mises_ratio_kernel(
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    tau: np.ndarray,
    yield_stress: float,
) -> float:
    max_vm = 0.0
    for index in range(sigma_x.shape[0]):
        vm_sq = (
            sigma_x[index] * sigma_x[index]
            - sigma_x[index] * sigma_y[index]
            + sigma_y[index] * sigma_y[index]
            + 3.0 * tau[index] * tau[index]
        )
        if vm_sq < 0.0:
            vm_sq = 0.0
        vm = math.sqrt(vm_sq)
        if vm > max_vm:
            max_vm = vm
    return max_vm / max(yield_stress, EPS)


def _membrane_amplitudes(field: MembraneField, total_deflection: np.ndarray) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        return (
            np.einsum("hij,i,j->h", field.coupling, total_deflection, total_deflection)
            - field.imperfection_offset
        )
    return _membrane_amplitudes_kernel(
        field.coupling,
        field.imperfection_offset,
        total_deflection,
    )


def _membrane_force(
    field: MembraneField,
    total_deflection: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        weighted = field.energy_matrix @ amplitudes
        return 2.0 * np.einsum("h,hij,j->i", weighted, field.coupling, total_deflection)
    return _membrane_force_kernel(field.energy_coupling, total_deflection, amplitudes)


def _membrane_tangent(
    field: MembraneField,
    total_deflection: np.ndarray,
    amplitudes: np.ndarray,
) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        gradients = np.einsum("hij,j->hi", field.coupling, total_deflection)
        weighted = field.energy_matrix @ amplitudes
        return 4.0 * gradients.T @ (field.energy_matrix @ gradients) + 2.0 * np.einsum(
            "h,hij->ij", weighted, field.coupling
        )
    return _membrane_tangent_kernel(
        field.coupling,
        field.imperfection_offset,
        field.energy_matrix,
        field.energy_coupling,
        total_deflection,
        amplitudes,
    )


def _membrane_stress_components(
    field: MembraneField,
    amplitudes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return compression-positive second-order stresses at the field grid."""

    if not NUMBA_AVAILABLE:
        return (
            field.sigma_x_grid @ amplitudes,
            field.sigma_y_grid @ amplitudes,
            field.tau_grid @ amplitudes,
        )
    return _membrane_stress_components_kernel(
        field.sigma_x_grid,
        field.sigma_y_grid,
        field.tau_grid,
        amplitudes,
    )


def _max_von_mises_ratio(
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    tau: np.ndarray,
    yield_stress: float,
) -> float:
    if not NUMBA_AVAILABLE:
        von_mises = np.sqrt(
            np.maximum(sigma_x**2 - sigma_x * sigma_y + sigma_y**2 + 3.0 * tau**2, 0.0)
        )
        return float(np.max(von_mises)) / max(yield_stress, EPS)
    return _max_von_mises_ratio_kernel(sigma_x, sigma_y, tau, yield_stress)


def _assign_family_imperfections(
    modes: list[RitzMode],
    family_amplitudes: Mapping[str, float],
) -> list[RitzMode]:
    """Assign production-tolerance imperfections to each family's critical mode.

    DNV-CG-0128 Sec.6 prescribes model imperfections harmonizing with the
    critical eigenmode shape: the minimum elastic-factor mode of each Ritz
    family carries the full tolerance amplitude.  Families without an elastic
    buckling driver fall back to their softest (lowest bending stiffness)
    mode, which is the fundamental deflection shape.
    """

    selected: dict[str, int] = {}
    selected_factor: dict[str, float] = {}
    fallback: dict[str, int] = {}
    for index, mode in enumerate(modes):
        if mode.family not in family_amplitudes:
            continue
        if mode.geometric_stiffness > EPS:
            factor = mode.linear_stiffness / mode.geometric_stiffness
            if mode.family not in selected or factor < selected_factor[mode.family]:
                selected[mode.family] = index
                selected_factor[mode.family] = factor
        current = fallback.get(mode.family)
        if current is None or mode.linear_stiffness < modes[current].linear_stiffness:
            fallback[mode.family] = index
    updated = list(modes)
    for family, amplitude in family_amplitudes.items():
        index = selected.get(family, fallback.get(family))
        if index is None or amplitude <= EPS:
            continue
        updated[index] = replace(updated[index], imperfection=amplitude)
    return updated


def _local_pressure_imperfection(panel: S3PanelInput) -> float:
    """Return the pressure-induced extra local plate imperfection.

    DNV-CG-0128 Sec.4.3 on the S3 element: "The lateral pressure amplifies the
    local plate deflections between primary stiffeners and it is modelled as
    an extra imperfection on top [of the] production model imperfection."
    The amplitude is the linear clamped plate-strip deflection
    ``p s^4 / (384 D)`` that follows from multi-bay continuity across the
    stiffener lines.
    """

    if panel.pressure <= 0.0:
        return 0.0
    return panel.pressure * panel.width**4 / (
        384.0 * _plate_bending_rigidity(panel)
    )


def build_ritz_modes(
    panel: S3PanelInput,
    section: S3SectionProperties,
    config: S3SolverConfig,
    global_stiffness_scale: float = 1.0,
) -> list[RitzMode]:
    modes: list[RitzMode] = []
    global_pressure_share = (
        0.0
        if config.use_separate_s3_pressure_modes
        else config.pressure_global_share
    )
    for family, pressure_share in (
        ("local", config.pressure_local_share),
        ("global", global_pressure_share),
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
                        imperfection=0.0,
                        nonlinear_stiffness=max(nonlinear, EPS),
                    )
                )
    if config.use_separate_s3_pressure_modes and config.pressure_global_share > EPS:
        stiffness = build_orthotropic_stiffness(panel, section, "global", config)
        stiffness = _with_longitudinal_stiffness_scale(stiffness, global_stiffness_scale)
        stiffness = OrthotropicStiffness(
            d11=stiffness.d11 * max(config.s3_pressure_mode_stiffness_factor, EPS),
            d12=stiffness.d12 * max(config.s3_pressure_mode_stiffness_factor, EPS),
            d22=stiffness.d22 * max(config.s3_pressure_mode_stiffness_factor, EPS),
            d66=stiffness.d66 * max(config.s3_pressure_mode_stiffness_factor, EPS),
            membrane_thickness=stiffness.membrane_thickness,
        )
        m = config.longitudinal_modes[0] if config.longitudinal_modes else 1
        n = config.transverse_modes[0] if config.transverse_modes else 1
        kx, ky, linear, _, nonlinear = _mode_linear_terms(panel, stiffness, config, m, n)
        modes.append(
            RitzMode(
                family="global-pressure",
                m=m,
                n=n,
                kx=kx,
                ky=ky,
                linear_stiffness=linear,
                geometric_stiffness=0.0,
                pressure_force=_pressure_generalized_force(
                    panel,
                    m,
                    n,
                    config.pressure_global_share,
                ),
                imperfection=0.0,
                nonlinear_stiffness=max(nonlinear, EPS),
            )
        )
    # Wide-panel cylindrical global modes: y-uniform sin(kx x) shapes of the
    # smeared stiffener/plate strip.  Their bending and geometric terms follow
    # from the same energy integrals with the y-uniform shape (area factor
    # a b / 2); the axial resultant acts on the full smeared section.  They
    # carry no second-order self-stiffening (developable surface), no shear or
    # transverse geometric drive, and no direct pressure share (the symmetric
    # clamped pressure response keeps its dedicated mode); their interaction
    # with the buckled plating enters exactly through the mixed cos*sin
    # membrane harmonics.
    cylindrical_stiffness = _with_longitudinal_stiffness_scale(
        build_orthotropic_stiffness(panel, section, "global", config),
        global_stiffness_scale,
    )
    cylinder_area_factor = panel.length * panel.width / 2.0
    cylinder_axial_resultant = (
        max(panel.axial_stress, 0.0) * cylindrical_stiffness.membrane_thickness
    )
    for m in config.longitudinal_modes:
        kx = m * math.pi / panel.length
        modes.append(
            RitzMode(
                family="global-cylindrical",
                m=m,
                n=0,
                kx=kx,
                ky=0.0,
                linear_stiffness=cylinder_area_factor
                * cylindrical_stiffness.d11
                * kx**4,
                geometric_stiffness=cylinder_area_factor
                * cylinder_axial_resultant
                * kx
                * kx,
                pressure_force=0.0,
                imperfection=0.0,
                nonlinear_stiffness=EPS,
            )
        )
    # The stiffener production imperfection is a bow of the stiffener/plate
    # unit (L/1000 per DNV-CG-0128 Sec.6), which is the cylindrical shape.
    modes = _assign_family_imperfections(
        modes,
        {
            "local": config.plate_imperfection_breadth_fraction * panel.width,
            "global-cylindrical": config.stiffener_imperfection_length_fraction
            * panel.length,
        },
    )
    # The pressure-induced extra local deflection keeps its own symmetric
    # long-wave shape: it adds to the fundamental (1,1) local mode rather than
    # to whichever local mode is elastically critical.
    pressure_imperfection = _local_pressure_imperfection(panel)
    if pressure_imperfection > EPS:
        for index, mode in enumerate(modes):
            if mode.family == "local" and mode.m == 1 and mode.n == 1:
                modes[index] = replace(
                    mode,
                    imperfection=mode.imperfection + pressure_imperfection,
                )
                break
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
                    imperfection=0.0,
                    nonlinear_stiffness=max(nonlinear, EPS),
                )
            )
    return _assign_family_imperfections(
        modes,
        {
            "plate": config.plate_imperfection_breadth_fraction
            * min(panel.length, panel.width),
        },
    )


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
    panel: S3PanelInput | U3PanelInput,
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


def _curvature_point(
    modes: Sequence[RitzMode],
    x: float,
    y: float,
    x_fraction: float,
    y_fraction: float,
    family: str | Sequence[str] | None = None,
) -> CurvaturePoint:
    families = (
        None
        if family is None
        else {family}
        if isinstance(family, str)
        else set(family)
    )
    d2x = []
    d2y = []
    dxy = []
    for mode in modes:
        if families is not None and mode.family not in families:
            d2x.append(0.0)
            d2y.append(0.0)
            dxy.append(0.0)
            continue
        sin_x = math.sin(mode.kx * x)
        # Cylindrical modes (n == 0) are uniform across the width.
        sin_y = math.sin(mode.ky * y) if mode.n > 0 else 1.0
        cos_x = math.cos(mode.kx * x)
        cos_y = math.cos(mode.ky * y)
        d2x.append(-mode.kx * mode.kx * sin_x * sin_y)
        d2y.append(-mode.ky * mode.ky * sin_x * sin_y)
        dxy.append(mode.kx * mode.ky * cos_x * cos_y)
    return CurvaturePoint(
        x_fraction=x_fraction,
        y_fraction=y_fraction,
        d2x=np.asarray(d2x, dtype=float),
        d2y=np.asarray(d2y, dtype=float),
        dxy=np.asarray(dxy, dtype=float),
    )


def _build_ritz_runtime(
    panel: S3PanelInput | U3PanelInput,
    modes: Sequence[RitzMode],
    config: S3SolverConfig,
) -> RitzRuntime:
    geometric, coupling = _ritz_geometric_matrix(panel, modes)
    max_delta = np.asarray(
        [
            max(
                1.0,
                0.10 / max(mode.kx, mode.ky, EPS),
            )
            for mode in modes
        ],
        dtype=float,
    )
    curvature_family = (
        None
        if isinstance(panel, U3PanelInput) or config.include_global_curvature_in_plate_yield
        else "local"
    )
    plate_points = tuple(
        _curvature_point(
            modes,
            panel.length * x_fraction,
            panel.width * y_fraction,
            x_fraction,
            y_fraction,
            family=curvature_family,
        )
        for x_fraction in config.hot_spot_grid
        for y_fraction in config.hot_spot_grid
    )
    global_centerline_points = tuple(
        _curvature_point(
            modes,
            panel.length * x_fraction,
            0.5 * panel.width,
            x_fraction,
            0.5,
            family=("global", "global-cylindrical"),
        )
        for x_fraction in config.hot_spot_grid
    )
    plate_x_fractions = np.asarray([point.x_fraction for point in plate_points], dtype=float)
    plate_y_fractions = np.asarray([point.y_fraction for point in plate_points], dtype=float)
    membrane = build_membrane_field(
        modes,
        panel.length,
        panel.width,
        panel.plate_thickness,
        panel.elastic_modulus,
        [mode.imperfection for mode in modes],
        config.membrane_hot_spot_fractions,
        poisson_ratio=panel.poisson_ratio,
    )
    return RitzRuntime(
        modes=modes,
        linear=np.diag([mode.linear_stiffness for mode in modes]),
        geometric=geometric,
        nonlinear=np.asarray([mode.nonlinear_stiffness for mode in modes], dtype=float),
        pressure=np.asarray([mode.pressure_force for mode in modes], dtype=float),
        imperfection=np.asarray([mode.imperfection for mode in modes], dtype=float),
        max_delta=max_delta,
        geometric_coupling=coupling,
        plate_curvature_points=plate_points,
        global_centerline_curvature_points=global_centerline_points,
        plate_x_fractions=plate_x_fractions,
        plate_y_fractions=plate_y_fractions,
        plate_d2x=np.vstack([point.d2x for point in plate_points]) if plate_points else np.zeros((0, len(modes))),
        plate_d2y=np.vstack([point.d2y for point in plate_points]) if plate_points else np.zeros((0, len(modes))),
        plate_dxy=np.vstack([point.dxy for point in plate_points]) if plate_points else np.zeros((0, len(modes))),
        global_centerline_d2x=(
            np.vstack([point.d2x for point in global_centerline_points])
            if global_centerline_points
            else np.zeros((0, len(modes)))
        ),
        membrane=membrane,
    )


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
    amplitude_utilization = amplitude_ratio / max(
        config.plate_imperfection_breadth_fraction, EPS
    )
    interaction_driver = local_elastic_utilization
    # Secant in-plane stiffness of the locally buckled plating.  The exact
    # single-mode von Karman solution with straight edges gives the membrane
    # strain epsilon = (2 sigma - sigma_cr) / E beyond the local critical
    # stress, i.e. a secant stiffness ratio sigma / (2 sigma - sigma_cr).
    # PULS computes its orthotropic macro coefficients "for an averaged state
    # (secant)" (User Manual Sec.3.8.3); this is that law for the reduced
    # scalar longitudinal degradation.
    if interaction_driver > 1.0:
        scale = interaction_driver / (2.0 * interaction_driver - 1.0)
    else:
        scale = 1.0
    return {
        "scale": scale,
        "local_elastic_utilization": local_elastic_utilization,
        "local_amplitude_ratio": amplitude_ratio,
        "amplitude_utilization": amplitude_utilization,
        "interaction_driver": interaction_driver,
    }


def _stiffener_column_factor(
    panel: S3PanelInput,
    section: S3SectionProperties,
    config: S3SolverConfig | None = None,
) -> float | None:
    """Return the wide-panel orthotropic global buckling candidate.

    The S3 panel breadth (number of stiffeners) is not part of the reduced
    input surface, so the global mode is taken in the wide-panel limit where
    the transverse wave number ky is a free continuous parameter.  For each
    axial half-wave count m the orthotropic eigenvalue

        lambda(m, ky) = (d11 kx^4 + 2 H kx^2 ky^2 + d22 ky^4)
                        / (Nx kx^2 + Ny ky^2)

    is minimized in closed form over ky^2 >= 0, with the axial resultant on
    the full smeared section (Nx = sigma_x A / s) and the transverse
    resultant on the plating (Ny = sigma_y t).  The pure-axial case reduces
    exactly to the Euler column of the stiffener/plate unit over the simply
    supported span (PULS User Manual Fig.11 asymmetric SS global modes), and
    the pure-transverse case reduces to the classical wide orthotropic plate
    formula 2 (sqrt(d11 d22) + H) kx^2 / Ny.
    """

    config = config or S3SolverConfig()
    stiffness = build_orthotropic_stiffness(panel, section, "global", config)
    axial_resultant = max(panel.axial_stress, 0.0) * stiffness.membrane_thickness
    transverse_resultant = max(panel.mean_transverse_stress, 0.0) * panel.plate_thickness
    if axial_resultant <= EPS and transverse_resultant <= EPS:
        return None

    cross_rigidity = stiffness.d12 + 2.0 * stiffness.d66
    best: float | None = None
    half_wave_counts = config.longitudinal_modes or (1,)
    for m in half_wave_counts:
        kx_sq = (m * math.pi / panel.length) ** 2
        bending_0 = stiffness.d11 * kx_sq * kx_sq
        bending_1 = 2.0 * cross_rigidity * kx_sq
        bending_2 = stiffness.d22
        load_0 = axial_resultant * kx_sq
        load_1 = transverse_resultant

        candidates = []
        if load_0 > EPS:
            candidates.append(bending_0 / load_0)
        if load_1 > EPS:
            # Interior stationary point of the Rayleigh quotient in u = ky^2:
            # c2 e u^2 + 2 c2 d u + (c1 d - e c0) = 0.
            a_term = bending_2 * load_1
            b_term = 2.0 * bending_2 * load_0
            c_term = bending_1 * load_0 - load_1 * bending_0
            discriminant = b_term * b_term - 4.0 * a_term * c_term
            if a_term > EPS and discriminant >= 0.0:
                root = (-b_term + math.sqrt(discriminant)) / (2.0 * a_term)
                if root > EPS:
                    candidates.append(
                        (bending_0 + bending_1 * root + bending_2 * root * root)
                        / (load_0 + load_1 * root)
                    )
        for value in candidates:
            if math.isfinite(value) and value > EPS and (best is None or value < best):
                best = value
    return best


def _plate_strip_shear_buckling(
    panel: S3PanelInput | U3PanelInput,
    length: float,
    width: float,
    thickness: float,
    shear_stress: float,
    capacity_factor: float = 1.0,
) -> dict[str, float] | None:
    """Return the elastic shear buckling factor for a plate strip.

    Notes
    -----
    This uses the classical simply-supported plate shear buckling coefficient

        k_tau = 5.34 + 4 / alpha^2

    where alpha = long_side / short_side >= 1.

    The classical elastic shear buckling stress is calculated directly as

        tau_cr = k_tau * pi^2 * E / (12 * (1 - nu^2)) * (t / short_side)^2

    The returned critical stress is then multiplied by the explicit
    `capacity_factor`.  S3 uses this as a visible reduced shear-interaction
    control; U3 keeps the default factor of 1.0.
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

    applied_capacity_factor = max(float(capacity_factor), EPS)
    classical_critical_stress = shear_coefficient * elastic_reference
    critical_stress = applied_capacity_factor * classical_critical_stress

    return {
        "factor": critical_stress / shear,
        "critical_stress": critical_stress,
        "classical_critical_stress": classical_critical_stress,
        "coefficient": shear_coefficient,
        "capacity_factor": applied_capacity_factor,
        "aspect_ratio": alpha,
        "short_side": short_side,
        "long_side": long_side,
    }


def _local_plate_shear_buckling(
    panel: S3PanelInput,
    config: S3SolverConfig,
) -> dict[str, float] | None:
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
        config.s3_shear_buckling_capacity_factor,
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

    The panel axial stress is the base proportional compression driver.  The
    SI/PI yield branches keep the public sniped-stiffener axial eccentricity
    moment.  The reduced web branch carries that contribution as an explicit
    sensitivity factor so the assumption remains visible in diagnostics and
    can be disabled for comparison without changing the SI/PI yield path.
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
    raw_absolute_moment = abs(float(sniped["absolute"]))
    web_local_sniped_factor = max(float(config.web_local_sniped_eccentricity_factor), 0.0)
    absolute_moment = raw_absolute_moment * web_local_sniped_factor
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
        "source": "stiffener-section-axial-plus-configured-sniped-web-edge-envelope",
        "controlling_edge": controlling_edge,
        "edge_compression": edge_compression,
        "edge_bending_stress": edge_bending,
        "effective_axial_stress": effective_axial,
        "sniped_eccentricity_moment": sniped,
        "web_local_sniped_eccentricity_factor": web_local_sniped_factor,
        "applied_sniped_eccentricity_moment": absolute_moment,
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
        config.s3_shear_buckling_capacity_factor,
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
    not yet implemented.  The default exponent is intentionally below the
    quadratic von-Mises-like value because the reduced candidates are already
    post-processed local capacities rather than independent stress components.
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


def _stiffener_torsional_buckling(
    panel: S3PanelInput,
    config: S3SolverConfig | None = None,
) -> dict[str, float] | None:
    """Return a reduced open-profile tripping/torsional stress candidate.

    DNV-CG-0128 uses a torsional reference stress based on St. Venant torsion,
    polar inertia, sectorial inertia, the stiffener span, and attachment
    restraint.  The reduced S3 solver uses the same ingredients in a
    gross-section estimate about the web root with an explicit restraint
    factor.  It is exposed as an approximate candidate in diagnostics.
    """

    config = config or S3SolverConfig()
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
    critical_stress = config.torsional_restraint_factor * (
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
        "restraint_factor": config.torsional_restraint_factor,
    }


def _global_stiffened_strip_capacity_adjustment(
    panel: S3PanelInput,
    section: S3SectionProperties,
    raw_factor: float,
    local_reference_factor: float | None,
    config: S3SolverConfig,
) -> dict[str, Any]:
    """Return the reduced global-strip capacity adjustment and diagnostics."""

    plate_reference_inertia = panel.width * panel.plate_thickness**3 / 12.0
    section_inertia_ratio = section.inertia_x / max(plate_reference_inertia, EPS)
    section_area_ratio = section.area / max(panel.width * panel.plate_thickness, EPS)
    aspect_ratio = panel.length / max(panel.width, EPS)
    plate_slenderness = panel.width / max(panel.plate_thickness, EPS)
    local_interaction_ratio = (
        raw_factor / max(local_reference_factor, EPS)
        if local_reference_factor is not None and local_reference_factor > EPS
        else None
    )
    load_family = _load_family(panel)

    if config.global_stiffened_strip_capacity_factor is not None:
        fixed_factor = max(float(config.global_stiffened_strip_capacity_factor), EPS)
        return {
            "raw_factor": raw_factor,
            "local_reference_factor": local_reference_factor,
            "local_interaction_ratio": local_interaction_ratio,
            "capacity_factor": fixed_factor,
            "mode": "fixed-override",
            "section": {
                "inertia_ratio": section_inertia_ratio,
                "area_ratio": section_area_ratio,
                "plate_slenderness": plate_slenderness,
                "aspect_ratio": aspect_ratio,
            },
            "modifiers": {
                "base": fixed_factor,
                "support": 1.0,
                "pressure": 1.0,
                "load": 1.0,
                "aspect": 1.0,
                "slenderness": 1.0,
                "section": 1.0,
                "local_interaction": 1.0,
            },
            "notes": ["fixed global-stiffened-strip capacity override"],
        }

    # PULS reports the GEB as the orthotropic eigenvalue itself; the local
    # buckling interaction enters through reduced (secant) stiffness
    # coefficients, which the caller applies via the closed-form coupling.
    # No empirical capacity knockdown is applied.
    return {
        "raw_factor": raw_factor,
        "local_reference_factor": local_reference_factor,
        "local_interaction_ratio": local_interaction_ratio,
        "capacity_factor": 1.0,
        "mode": "raw-orthotropic-eigenvalue",
        "load_family": load_family,
        "section": {
            "inertia_ratio": section_inertia_ratio,
            "area_ratio": section_area_ratio,
            "plate_slenderness": plate_slenderness,
            "aspect_ratio": aspect_ratio,
        },
        "notes": [
            "raw orthotropic GEB eigenvalue per the PULS definition; "
            "local-buckling interaction applied through the secant von Karman "
            "stiffness reduction"
        ],
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
    factor_rows = []
    for mode in modes:
        if mode.geometric_stiffness <= EPS:
            continue
        raw_factor = mode.linear_stiffness / mode.geometric_stiffness
        if mode.family == "global":
            capacity_factor = 1.0
            factor = raw_factor
            failure_family = "global-stiffened-strip"
        elif mode.family == "global-cylindrical":
            capacity_factor = 1.0
            factor = raw_factor
            failure_family = "global-stiffener-cutoff"
        else:
            capacity_factor = 1.0
            factor = raw_factor
            failure_family = "plate"
        factor_rows.append(
            {
                "factor": factor,
                "raw_factor": raw_factor,
                "capacity_factor": capacity_factor,
                "label": mode.label,
                "family": mode.family,
                "failure_family": failure_family,
            }
        )
    column_factor = _stiffener_column_factor(panel, section, config)
    if column_factor is not None:
        factor_rows.append(
            {
                "factor": column_factor,
                "label": "stiffener-column",
                "family": "stiffener-column",
                "failure_family": "global-stiffener-cutoff",
            }
        )
    shear_factor = _local_plate_shear_buckling(panel, config)
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
                "raw_factor": factor["factor"],
                "capacity_factor": 1.0,
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
    torsional_factor = _stiffener_torsional_buckling(panel, config)
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
            if row["failure_family"] not in (
                "global-stiffened-strip",
                "global-stiffener-cutoff",
            ):
                continue
            raw_factor = float(row.get("raw_factor", row["factor"]))
            adjustment = _global_stiffened_strip_capacity_adjustment(
                panel,
                section,
                raw_factor,
                local_reference_factor,
                config,
            )
            capacity_factor = float(adjustment["capacity_factor"])
            uncoupled_factor = raw_factor * capacity_factor
            interaction_driver = uncoupled_factor / max(local_reference_factor, EPS)
            # Self-consistent reduced GEB with the secant von Karman membrane
            # law: with the longitudinal stiffness reduced by u / (2u - 1) at
            # utilization u = lambda / lambda_local, the eigenvalue condition
            # lambda = raw * scale(lambda) has the closed-form solution
            # lambda = (raw + lambda_local) / 2 once the local mode buckles
            # first.
            if interaction_driver > 1.0:
                coupled_factor = 0.5 * (uncoupled_factor + local_reference_factor)
            else:
                coupled_factor = uncoupled_factor
            scale = coupled_factor / max(uncoupled_factor, EPS)
            row["raw_factor"] = raw_factor
            row["capacity_factor"] = capacity_factor
            row["global_capacity_adjustment"] = adjustment
            row["uncoupled_factor"] = uncoupled_factor
            row["elastic_coupling_scale"] = scale
            row["factor"] = coupled_factor
            elastic_global_coupling_rows.append(
                {
                    "mode": str(row["label"]),
                    "raw_factor": raw_factor,
                    "capacity_factor": capacity_factor,
                    "capacity_adjustment": adjustment,
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
        if "global_capacity_adjustment" in row:
            family_summary["raw_factor"] = row["raw_factor"]
            family_summary["capacity_factor"] = row["capacity_factor"]
            family_summary["global_capacity_adjustment"] = row["global_capacity_adjustment"]
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
    runtime: RitzRuntime | None = None,
) -> np.ndarray:
    limited = delta.copy()
    if runtime is not None:
        max_delta_values = np.maximum(runtime.max_delta, 0.5 * np.abs(amplitudes))
        mask = np.abs(limited) > max_delta_values
        if np.any(mask):
            limited[mask] = np.sign(limited[mask]) * max_delta_values[mask]
        return limited
    for index, mode in enumerate(modes):
        max_delta = max(
            1.0,
            0.5 * abs(float(amplitudes[index])),
            0.10 / max(mode.kx, mode.ky, EPS),
        )
        if abs(float(limited[index])) > max_delta:
            limited[index] = math.copysign(max_delta, float(limited[index]))
    return limited


@njit(cache=True)
def _solve_equilibrium_membrane_kernel(
    linear: np.ndarray,
    geometric: np.ndarray,
    coupling: np.ndarray,
    imperfection_offset: np.ndarray,
    energy_matrix: np.ndarray,
    energy_coupling: np.ndarray,
    pressure: np.ndarray,
    imperfection: np.ndarray,
    max_delta_base: np.ndarray,
    previous_amplitudes: np.ndarray,
    load_factor: float,
    newton_max_iterations: int,
    newton_tolerance: float,
) -> tuple[np.ndarray, bool, int]:
    mode_count = previous_amplitudes.shape[0]
    harmonic_count = coupling.shape[0]
    q = previous_amplitudes.copy()
    force = pressure + load_factor * (geometric @ imperfection)
    tangent_linear = linear - load_factor * geometric
    if np.max(np.abs(force)) <= EPS and np.max(np.abs(q)) <= EPS:
        return np.zeros(mode_count, dtype=np.float64), True, 0

    for iteration in range(1, newton_max_iterations + 1):
        total_deflection = q + imperfection

        membrane_amplitudes = np.empty(harmonic_count, dtype=np.float64)
        for harmonic in range(harmonic_count):
            value = 0.0
            for i in range(mode_count):
                qi = total_deflection[i]
                for j in range(mode_count):
                    value += coupling[harmonic, i, j] * qi * total_deflection[j]
            membrane_amplitudes[harmonic] = value - imperfection_offset[harmonic]

        nonlinear_response = np.zeros(mode_count, dtype=np.float64)
        for i in range(mode_count):
            value = 0.0
            for harmonic in range(harmonic_count):
                weighted = membrane_amplitudes[harmonic]
                for j in range(mode_count):
                    value += weighted * energy_coupling[harmonic, i, j] * total_deflection[j]
            nonlinear_response[i] = 2.0 * value

        linear_response = tangent_linear @ q
        residual = linear_response + nonlinear_response - force
        residual_norm = np.max(np.abs(residual))
        scale = max(
            np.max(np.abs(force)),
            np.max(np.abs(linear_response)),
            np.max(np.abs(nonlinear_response)),
            1.0,
        )
        if residual_norm <= newton_tolerance * scale:
            return q, True, iteration

        gradients = np.zeros((harmonic_count, mode_count), dtype=np.float64)
        for harmonic in range(harmonic_count):
            for i in range(mode_count):
                value = 0.0
                for j in range(mode_count):
                    value += coupling[harmonic, i, j] * total_deflection[j]
                gradients[harmonic, i] = value

        weighted_gradients = np.zeros((harmonic_count, mode_count), dtype=np.float64)
        for harmonic in range(harmonic_count):
            for i in range(mode_count):
                value = 0.0
                for other in range(harmonic_count):
                    value += energy_matrix[harmonic, other] * gradients[other, i]
                weighted_gradients[harmonic, i] = value

        tangent = tangent_linear.copy()
        for i in range(mode_count):
            for j in range(mode_count):
                value = 0.0
                for harmonic in range(harmonic_count):
                    value += gradients[harmonic, i] * weighted_gradients[harmonic, j]
                tangent[i, j] += 4.0 * value

        for i in range(mode_count):
            for j in range(mode_count):
                value = 0.0
                for harmonic in range(harmonic_count):
                    value += membrane_amplitudes[harmonic] * energy_coupling[harmonic, i, j]
                tangent[i, j] += 2.0 * value

        delta = np.linalg.solve(tangent, -residual)
        for i in range(mode_count):
            max_delta = max(max_delta_base[i], 0.5 * abs(q[i]))
            if abs(delta[i]) > max_delta:
                delta[i] = math.copysign(max_delta, delta[i])
            q[i] += delta[i]
            if not math.isfinite(q[i]):
                return previous_amplitudes.copy(), False, iteration

    return q, False, newton_max_iterations


def _solve_equilibrium_amplitudes_python(
    panel: S3PanelInput | U3PanelInput,
    modes: Sequence[RitzMode],
    load_factor: float,
    previous_amplitudes: Sequence[float],
    config: S3SolverConfig,
    runtime: RitzRuntime,
) -> tuple[list[float], bool, int]:
    q = np.asarray(previous_amplitudes, dtype=float)
    if q.shape != (len(modes),):
        q = np.zeros(len(modes), dtype=float)

    linear = runtime.linear
    geometric = runtime.geometric
    membrane = runtime.membrane
    nonlinear = runtime.nonlinear
    pressure = runtime.pressure
    imperfection = runtime.imperfection
    force = pressure + load_factor * (geometric @ imperfection)
    tangent_linear = linear - load_factor * geometric
    tangent_linear_diagonal = np.diagonal(tangent_linear).copy()
    diagonal_stride = len(modes) + 1
    if np.max(np.abs(force)) <= EPS and np.max(np.abs(q)) <= EPS:
        return [0.0 for _ in modes], True, 0

    for iteration in range(1, config.newton_max_iterations + 1):
        if membrane is not None:
            total_deflection = q + imperfection
            membrane_amplitudes = _membrane_amplitudes(membrane, total_deflection)
            nonlinear_response = _membrane_force(
                membrane, total_deflection, membrane_amplitudes
            )
        else:
            nonlinear_response = nonlinear * q**3
        linear_response = tangent_linear @ q
        residual = linear_response + nonlinear_response - force
        scale = max(
            float(np.max(np.abs(force))),
            float(np.max(np.abs(linear_response))),
            float(np.max(np.abs(nonlinear_response))),
            1.0,
        )
        if float(np.max(np.abs(residual))) <= config.newton_tolerance * scale:
            return q.tolist(), True, iteration

        if membrane is not None:
            tangent = tangent_linear + _membrane_tangent(
                membrane, total_deflection, membrane_amplitudes
            )
        else:
            tangent = tangent_linear.copy()
            tangent.flat[::diagonal_stride] = (
                tangent_linear_diagonal + 3.0 * nonlinear * q * q
            )
        try:
            delta = np.linalg.solve(tangent, -residual)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(tangent, -residual, rcond=None)[0]
        delta = _limit_newton_delta(modes, q, delta, runtime)
        q += delta
        if not np.all(np.isfinite(q)):
            return list(previous_amplitudes), False, iteration

    return q.tolist(), False, config.newton_max_iterations


def solve_equilibrium_amplitudes(
    panel: S3PanelInput | U3PanelInput,
    modes: Sequence[RitzMode],
    load_factor: float,
    previous_amplitudes: Sequence[float],
    config: S3SolverConfig,
    runtime: RitzRuntime | None = None,
) -> tuple[list[float], bool, int]:
    """Solve the coupled reduced Ritz continuation equilibrium.

    Normal resultants keep diagonal geometric terms in this basis.  Panel shear
    contributes off-diagonal geometric coupling between opposite-parity modes,
    so the load-path residual and Newton tangent are assembled as vectors and
    matrices instead of solving each amplitude independently.
    """

    if not modes:
        return [], True, 0

    runtime = runtime or _build_ritz_runtime(panel, modes, config)
    if NUMBA_AVAILABLE and runtime.membrane is not None:
        q = np.asarray(previous_amplitudes, dtype=float)
        if q.shape != (len(modes),):
            q = np.zeros(len(modes), dtype=float)
        try:
            solved, converged, iterations = _solve_equilibrium_membrane_kernel(
                runtime.linear,
                runtime.geometric,
                runtime.membrane.coupling,
                runtime.membrane.imperfection_offset,
                runtime.membrane.energy_matrix,
                runtime.membrane.energy_coupling,
                runtime.pressure,
                runtime.imperfection,
                runtime.max_delta,
                q,
                load_factor,
                config.newton_max_iterations,
                config.newton_tolerance,
            )
            return solved.tolist(), bool(converged), int(iterations)
        except Exception:
            pass
    return _solve_equilibrium_amplitudes_python(
        panel,
        modes,
        load_factor,
        previous_amplitudes,
        config,
        runtime,
    )


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
    family: str | Sequence[str] | None = None,
) -> tuple[float, float, float]:
    families = (
        None
        if family is None
        else {family}
        if isinstance(family, str)
        else set(family)
    )
    d2x = 0.0
    d2y = 0.0
    dxy = 0.0
    for mode, amplitude in zip(modes, amplitudes):
        if families is not None and mode.family not in families:
            continue
        sin_x = math.sin(mode.kx * x)
        sin_y = math.sin(mode.ky * y) if mode.n > 0 else 1.0
        cos_x = math.cos(mode.kx * x)
        cos_y = math.cos(mode.ky * y)
        d2x -= amplitude * mode.kx * mode.kx * sin_x * sin_y
        d2y -= amplitude * mode.ky * mode.ky * sin_x * sin_y
        dxy += amplitude * mode.kx * mode.ky * cos_x * cos_y
    return d2x, d2y, dxy


def _runtime_curvatures(point: CurvaturePoint, amplitudes: np.ndarray) -> tuple[float, float, float]:
    return (
        float(point.d2x @ amplitudes),
        float(point.d2y @ amplitudes),
        float(point.dxy @ amplitudes),
    )


def _pressure_stiffener_bending_moment(panel: S3PanelInput) -> float:
    """Return the controlling pressure bending moment for the SI/PI branches.

    The clamped continuous beam carries ``p s L^2 / 12`` at the frame support
    and ``p s L^2 / 24`` at midspan.  The PULS stiffener limit states are not
    always evaluated at midspan ("maximum curvature ... could be closer to
    the ends", User Manual Sec.3.10.1) and the support is additionally
    governed by the combined axial/bending limit state i = 6, so the reduced
    SI/PI branches keep the larger support moment as the envelope of both
    locations.  Sniped stiffeners are simply supported with the midspan
    moment ``p s L^2 / 8``.
    """

    if panel.pressure <= 0.0:
        return 0.0
    span_factor = 12.0 if panel.stiffener_boundary == "Cont" else 8.0
    return panel.pressure * panel.width * panel.length**2 / span_factor


def s3_pressure_capacity_limits(
    panel: S3PanelInput,
    section: S3SectionProperties,
) -> dict[str, float]:
    """Return the PULS S3 lateral pressure limits from linear beam/strip theory.

    PULS User Manual Sec.3.13 defines three maximum pressure criteria:

    * ``stiffener_bending``: first bending stress yield at the support of the
      stiffener/plate unit, ``p_Fs = 12 sigma_F W_min / (s L^2)`` with the
      section modulus taken at the stiffener flange mid-plane.  The clamped
      span factor 12 applies to continuous stiffeners; simply supported beam
      theory gives 8 for sniped stiffeners.
    * ``web_shear``: first pure shear yield in the stiffener web for the
      clamped stiffener, ``p_s = 2 V_s / (s L)`` with
      ``V_s = sigma_F t_w I / (sqrt(3) S_p)`` and the first moment of area
      ``S_p = s t_p z_g + t_w (z_g - t_p / 2)^2 / 2`` at the neutral axis.
    * ``plate_bending``: first surface yield from pure local bending of the
      clamped plate strip between stiffeners, ``p_F = 2 (t / s)^2 sigma_F``.

    The manual enforces the two stiffener limits ("Two different pressure
    limits are specified ...") while the plate strip value is reported as
    being "of practical interest" only, so ``minimum`` covers the enforced
    pair and ``plate_bending`` stays informational.
    """

    span_factor = 12.0 if panel.stiffener_boundary == "Cont" else 8.0
    spacing = max(panel.width, EPS)
    span = max(panel.length, EPS)
    centroid = section.centroid_from_plate_midplane
    flange_mid_distance = (
        0.5 * panel.plate_thickness
        + panel.stiffener_height
        + 0.5 * (0.0 if panel.stiffener_type == "Flatbar" else panel.flange_thickness)
        - centroid
    )
    minimum_section_modulus = section.inertia_x / max(abs(flange_mid_distance), EPS)
    stiffener_bending = (
        span_factor
        * panel.yield_stress_stiffener
        * minimum_section_modulus
        / (spacing * span**2)
    )

    first_moment = spacing * panel.plate_thickness * abs(centroid) + 0.5 * max(
        panel.web_thickness, EPS
    ) * max(abs(centroid) - 0.5 * panel.plate_thickness, 0.0) ** 2
    shear_yield_force = (
        panel.yield_stress_stiffener
        * panel.web_thickness
        * section.inertia_x
        / (math.sqrt(3.0) * max(first_moment, EPS))
    )
    web_shear = 2.0 * shear_yield_force / (spacing * span)

    plate_bending = (
        2.0
        * (panel.plate_thickness / spacing) ** 2
        * panel.yield_stress_plate
    )
    return {
        "stiffener_bending": stiffener_bending,
        "web_shear": web_shear,
        "plate_bending": plate_bending,
        "minimum": min(stiffener_bending, web_shear),
        "span_factor": span_factor,
        "minimum_section_modulus": minimum_section_modulus,
        "first_moment_of_area": first_moment,
    }


def u3_pressure_capacity_limit(panel: U3PanelInput) -> dict[str, float]:
    """Return the PULS U3 lateral pressure limit from linear strip theory.

    PULS User Manual Sec.2.10: ``p_f = 2 (t / s)^2 sigma_F`` corresponds to
    first material yielding in the extreme fibre along the long edges of a
    clamped plate unit strip, with ``s`` the shortest plate dimension.
    """

    short_side = max(min(panel.length, panel.width), EPS)
    plate_bending = (
        2.0 * (panel.plate_thickness / short_side) ** 2 * panel.yield_stress_plate
    )
    return {
        "plate_bending": plate_bending,
        "minimum": plate_bending,
        "short_side": short_side,
    }


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
    torsional = _stiffener_torsional_buckling(panel, config)
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


def _plate_membrane_field(
    panel: S3PanelInput | U3PanelInput,
    modes: Sequence[RitzMode],
    config: S3SolverConfig,
    runtime: RitzRuntime | None,
) -> MembraneField | None:
    if runtime is not None and runtime.membrane is not None:
        return runtime.membrane
    return build_membrane_field(
        modes,
        panel.length,
        panel.width,
        panel.plate_thickness,
        panel.elastic_modulus,
        [mode.imperfection for mode in modes],
        config.membrane_hot_spot_fractions,
        poisson_ratio=panel.poisson_ratio,
    )


def _plate_yield_ratio(
    panel: S3PanelInput,
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
    load_factor: float,
    config: S3SolverConfig,
    runtime: RitzRuntime | None = None,
) -> float:
    """Return the redistributed membrane stress control ratio for the plating.

    PULS evaluates its plate limit states on the redistributed membrane
    stresses (mid-plane stresses of each component plate); bending stresses
    across the plate thickness are explicitly excluded from the yield criteria
    (PULS User Manual Sec.3.1 and Sec.3.10.1).  The second-order membrane
    stresses follow from the exact Airy solution of the Marguerre
    compatibility equation for the Ritz deflection field, evaluated on a grid
    that includes the supported edges where the redistribution concentrates
    compression.
    """

    field = _plate_membrane_field(panel, modes, config, runtime)
    if field is None:
        vm = _stress_von_mises(
            load_factor * panel.axial_stress,
            load_factor * panel.mean_transverse_stress,
            load_factor * panel.shear_stress,
        )
        return vm / max(panel.yield_stress_plate, EPS)

    total_deflection = np.asarray(amplitudes, dtype=float) + np.asarray(
        [mode.imperfection for mode in modes], dtype=float
    )
    membrane_amplitudes = _membrane_amplitudes(field, total_deflection)
    sigma_x2, sigma_y2, tau2 = _membrane_stress_components(field, membrane_amplitudes)
    sigma_x = load_factor * panel.axial_stress + sigma_x2
    sigma_y = (
        load_factor
        * (
            panel.transverse_stress_1
            + (panel.transverse_stress_2 - panel.transverse_stress_1)
            * field.grid_y_fractions
        )
        + sigma_y2
    )
    tau = load_factor * panel.shear_stress + tau2
    return _max_von_mises_ratio(sigma_x, sigma_y, tau, panel.yield_stress_plate)


def _u3_plate_yield_ratio(
    panel: U3PanelInput,
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
    load_factor: float,
    config: S3SolverConfig,
    runtime: RitzRuntime | None = None,
) -> float:
    """Return the redistributed membrane stress control ratio for the plate.

    Same membrane-stress limit-state form as the S3 plate criterion, with the
    linearly varying axial stress interpolated along the plate length.
    """

    field = _plate_membrane_field(panel, modes, config, runtime)
    if field is None:
        vm = _stress_von_mises(
            load_factor * panel.axial_stress,
            load_factor * panel.mean_transverse_stress,
            load_factor * panel.shear_stress,
        )
        return vm / max(panel.yield_stress_plate, EPS)

    total_deflection = np.asarray(amplitudes, dtype=float) + np.asarray(
        [mode.imperfection for mode in modes], dtype=float
    )
    membrane_amplitudes = _membrane_amplitudes(field, total_deflection)
    sigma_x2, sigma_y2, tau2 = _membrane_stress_components(field, membrane_amplitudes)
    sigma_x = (
        load_factor
        * (
            panel.axial_stress_1
            + (panel.axial_stress_2 - panel.axial_stress_1) * field.grid_x_fractions
        )
        + sigma_x2
    )
    sigma_y = (
        load_factor
        * (
            panel.transverse_stress_1
            + (panel.transverse_stress_2 - panel.transverse_stress_1)
            * field.grid_y_fractions
        )
        + sigma_y2
    )
    tau = load_factor * panel.shear_stress + tau2
    return _max_von_mises_ratio(sigma_x, sigma_y, tau, panel.yield_stress_plate)


def u3_yield_utilization(
    panel: U3PanelInput,
    modes: Sequence[RitzMode],
    amplitudes: Sequence[float],
    load_factor: float,
    config: S3SolverConfig,
    runtime: RitzRuntime | None = None,
) -> dict[str, Any]:
    plate_ratio = _u3_plate_yield_ratio(panel, modes, amplitudes, load_factor, config, runtime)
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
    runtime: RitzRuntime | None = None,
    amplitudes: Sequence[float] | None = None,
) -> dict[str, float]:
    """Return attached-plating effective axial stress for stiffener checks.

    When the nonlinear response is available the redistributed membrane edge
    stress along the plate/stiffener junction lines is added.  Averaging the
    second-order Airy field along a junction keeps only its ``p = 0``
    harmonics, which is the analytic form of the load shed from the buckled
    plating into the stiffeners (compatibility of junction strain).
    """

    nominal_stress = load_factor * panel.axial_stress
    area_factor = gross_section.area / max(stiffener_section.area, EPS)
    shed_stress = 0.0
    if (
        runtime is not None
        and runtime.membrane is not None
        and amplitudes is not None
        and len(amplitudes) == len(runtime.imperfection)
    ):
        field = runtime.membrane
        total_deflection = np.asarray(amplitudes, dtype=float) + runtime.imperfection
        membrane_amplitudes = _membrane_amplitudes(field, total_deflection)
        edge_means = field.edge_mean_axial_factors @ membrane_amplitudes
        shed_stress = float(np.max(edge_means))
    return {
        "stress": nominal_stress * area_factor + shed_stress,
        "nominal_stress": nominal_stress,
        "area_factor": area_factor,
        "membrane_shed_stress": shed_stress,
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
    runtime: RitzRuntime | None = None,
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
        runtime,
        amplitudes,
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
    if runtime is not None:
        amplitude_array = np.asarray(amplitudes, dtype=float)
        if runtime.global_centerline_d2x.size:
            max_curvature = float(np.max(np.abs(runtime.global_centerline_d2x @ amplitude_array)))
    else:
        for x_fraction in config.hot_spot_grid:
            d2x, _, _ = _mode_curvatures(
                modes,
                amplitudes,
                panel.length * x_fraction,
                0.5 * panel.width,
                family=("global", "global-cylindrical"),
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
    # The Ritz response deflection stresses stay diagnostic: the guide-style
    # Perry lateral-deformation moment (with its slenderness reduction)
    # carries the global bending in the SI/PI branches, while the cylindrical
    # response enters the limit states through the membrane field.
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
    runtime: RitzRuntime | None = None,
) -> dict[str, Any]:
    plate_ratio = _plate_yield_ratio(panel, modes, amplitudes, load_factor, config, runtime)
    stiffener_ratios = _stiffener_yield_ratios(
        panel,
        section,
        stiffener_section,
        modes,
        amplitudes,
        load_factor,
        config,
        global_elastic_factor,
        runtime,
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
        "accelerated": config.use_accelerated_s3_continuation,
        "refinement_tolerance": config.continuation_refinement_tolerance,
        "refinement_max_iterations": config.continuation_refinement_max_iterations,
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
        "plate limit state uses redistributed membrane stresses from the exact Marguerre/Airy solution of the reduced deflection field; through-thickness plate bending is excluded per the PULS limit-state definition",
        "nonlinear membrane stiffness on the continuation path is the energy-consistent von Karman term from the same Airy solution; lateral pressure validity follows the PULS manual linear beam/strip limits",
        "buckling usage is the reduced buckling-strength envelope over ultimate capacity and elastic local/global buckling limits; fixed-pressure material preload contribution is reported in ultimate diagnostics but excluded from buckling-strength control by default",
        "shear-normal Ritz coupling is truncated in elastic and continuation checks; classical local plate shear remains a fallback candidate",
        "web-local compression-shear uses a reduced stiffener-section web-edge compression envelope and a reduced local plate-web interaction; torsional stiffener remains a reduced gross-section estimate",
        "stiffener yield exposes SI/PI section branches, lateral-deformation and sniped bending, SI-only torsional stress, and effective attached plate width",
        "global longitudinal strip stiffness degrades from local elastic utilization on the nonlinear load path",
        "PULS user manual S3 limits are reported as diagnostics; covered-domain gates remain controlled by solver config",
    ]


def _u3_notes() -> list[str]:
    return [
        "regular U3 unstiffened rectangular panels only; T1, K3, corrugation and FRP are outside this milestone",
        "positive PULS export normal stress is compression; signed stresses scale while lateral pressure remains fixed",
        "Rayleigh-Ritz sine modes use an isotropic plate basis with simply-supported trigonometric shapes",
        "buckling usage is the reduced buckling-strength envelope over first-yield capacity and elastic plate/shear buckling limits",
        "plate limit state uses redistributed membrane stresses from the exact Marguerre/Airy solution; through-thickness plate bending is excluded per the PULS limit-state definition, while the manual clamped-strip pressure limit gates lateral pressure",
        "U3 end stresses are interpolated in yield checks and averaged in elastic buckling checks",
        "PULS user manual U3 limits are reported as diagnostics; covered-domain gates remain controlled by solver config",
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
    preload_share = preload_max / max(_s3_major_yield_utilization_limit(config), EPS)
    return preload_share >= config.pressure_dominated_yield_preload_ratio


def _s3_major_yield_utilization_limit(config: S3SolverConfig) -> float:
    """Return the S3 utilization target for major yield/collapse detection."""

    reserve = max(config.s3_major_yield_reserve_factor, EPS)
    return max(config.yield_utilization_limit, EPS) * reserve


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
    if not config.include_solver_diagnostics and not config.check_mode_convergence:
        confidence = "medium" if result.valid else "low"
        confidence_reasons = [] if result.valid else list(validation_reasons or ())
        diagnostics = dict(result.diagnostics)
        diagnostics.setdefault("mode_convergence", {"enabled": False})
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
    runtime = _build_ritz_runtime(panel, modes, config)
    amplitudes = [0.0 for _ in modes]
    amplitudes, pressure_converged, pressure_iterations = solve_equilibrium_amplitudes(
        panel,
        modes,
        0.0,
        amplitudes,
        config,
        runtime,
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

    pressure_yield = yield_utilization(
        panel,
        section,
        stiffener_section,
        modes,
        amplitudes,
        0.0,
        config,
        runtime=runtime,
    )
    pressure_preload_response = {
        "iterations": pressure_iterations,
        "amplitudes": _mode_amplitude_summary(modes, amplitudes),
        "yield_utilization": pressure_yield,
        "controlling_yield_branch": max(
            ("plate", "stiffener_induced", "plate_induced"),
            key=lambda branch: float(pressure_yield[branch]),
        ),
        "pressure_mode_model": {
            "separate_symmetric_pressure_modes": config.use_separate_s3_pressure_modes,
            "pressure_family": (
                "global-pressure"
                if config.use_separate_s3_pressure_modes
                else "global"
            ),
            "pressure_mode_stiffness_factor": (
                config.s3_pressure_mode_stiffness_factor
                if config.use_separate_s3_pressure_modes
                else 1.0
            ),
            "manual_basis": (
                "S3 user manual separates symmetric/clamped pressure modes from "
                "asymmetric simply supported buckling modes"
            ),
        },
    }
    pressure_limits = s3_pressure_capacity_limits(panel, section)
    pressure_preload_response["pressure_capacity_limits"] = pressure_limits
    if (
        panel.pressure > pressure_limits["minimum"]
        or pressure_yield["max"] >= config.pressure_yield_limit
    ):
        return _attach_reliability(
            _invalid_result(
                "pressure",
                {
                    "stage": "pressure-preload",
                    "yield_utilization": pressure_yield,
                    "pressure_capacity_limits": pressure_limits,
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
    column_family = buckling["modeled_failure_families"].get("global-stiffener-cutoff", {})
    column_factor = _optional_float(column_family.get("critical_factor"))
    if column_factor is None:
        column_factor = buckling["stiffener_column_factor"]
    global_family = buckling["modeled_failure_families"].get("global-stiffened-strip", {})
    global_elastic_cutoff_factor = _optional_float(global_family.get("critical_factor"))
    # The Perry lateral-deformation amplification uses the governing global
    # eigenvalue: the minimum of the orthotropic strip mode and the wide-panel
    # (column-limit) candidate.
    governing_global_factor = min(
        (
            value
            for value in (global_elastic_cutoff_factor, column_factor)
            if value is not None
        ),
        default=None,
    )
    s3_major_yield_limit = _s3_major_yield_utilization_limit(config)

    previous_load = 0.0
    previous_yield = pressure_yield["max"]
    yield_capacity_factor: float | None = None
    max_iterations = pressure_iterations
    collapse_state = "major-yield"
    final_yield = pressure_yield
    local_global_coupling = local_global_stiffness_scale(panel, modes, amplitudes, 0.0, config)

    accepted_steps = 0
    rejected_steps = 0
    cutbacks = 0
    refinement_steps = 0
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
            runtime = _build_ritz_runtime(panel, modes, config)
        trial_amplitudes, converged, iterations = solve_equilibrium_amplitudes(
            panel,
            modes,
            load_factor,
            amplitudes,
            config,
            runtime,
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
                continuation["refinement_steps"] = refinement_steps
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

        lower_load = previous_load
        lower_yield = previous_yield
        lower_amplitudes = amplitudes
        lower_modes = modes
        lower_runtime = runtime
        lower_coupling = local_global_coupling
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
            governing_global_factor
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
            runtime,
        )
        if final_yield["max"] >= s3_major_yield_limit:
            high_load = load_factor
            high_yield = final_yield
            high_amplitudes = amplitudes
            high_modes = modes
            high_runtime = runtime
            high_coupling = local_global_coupling
            if config.use_accelerated_s3_continuation:
                tolerance = max(config.continuation_refinement_tolerance, EPS)
                for _ in range(max(config.continuation_refinement_max_iterations, 0)):
                    bracket = max(high_load - lower_load, 0.0)
                    if bracket <= tolerance * max(1.0, high_load):
                        break
                    denominator = high_yield["max"] - lower_yield
                    if abs(denominator) > EPS:
                        refined_load = lower_load + (
                            (s3_major_yield_limit - lower_yield)
                            * (high_load - lower_load)
                            / denominator
                        )
                        lower_bound = lower_load + 0.20 * (high_load - lower_load)
                        upper_bound = lower_load + 0.80 * (high_load - lower_load)
                        refined_load = min(max(refined_load, lower_bound), upper_bound)
                    else:
                        refined_load = 0.5 * (lower_load + high_load)
                    refined_coupling = local_global_stiffness_scale(
                        panel,
                        lower_modes,
                        lower_amplitudes,
                        refined_load,
                        config,
                    )
                    refined_modes = lower_modes
                    refined_runtime = lower_runtime
                    if refined_coupling["scale"] < 1.0:
                        refined_modes = build_ritz_modes(
                            panel,
                            section,
                            config,
                            global_stiffness_scale=refined_coupling["scale"],
                        )
                        refined_runtime = _build_ritz_runtime(panel, refined_modes, config)
                    refined_amplitudes, refined_converged, refined_iterations = (
                        solve_equilibrium_amplitudes(
                            panel,
                            refined_modes,
                            refined_load,
                            lower_amplitudes,
                            config,
                            refined_runtime,
                        )
                    )
                    max_iterations = max(max_iterations, refined_iterations)
                    if not refined_converged:
                        rejected_steps += 1
                        break
                    refinement_steps += 1
                    refined_yield = yield_utilization(
                        panel,
                        section,
                        stiffener_section,
                        refined_modes,
                        refined_amplitudes,
                        refined_load,
                        config,
                        ultimate_yield_global_factor,
                        refined_runtime,
                    )
                    if refined_yield["max"] >= s3_major_yield_limit:
                        high_load = refined_load
                        high_yield = refined_yield
                        high_amplitudes = refined_amplitudes
                        high_modes = refined_modes
                        high_runtime = refined_runtime
                        high_coupling = refined_coupling
                    else:
                        lower_load = refined_load
                        lower_yield = refined_yield["max"]
                        lower_amplitudes = refined_amplitudes
                        lower_modes = refined_modes
                        lower_runtime = refined_runtime
                        lower_coupling = refined_coupling
                amplitudes = high_amplitudes
                modes = high_modes
                runtime = high_runtime
                final_yield = high_yield
                local_global_coupling = high_coupling
                previous_load = high_load
            else:
                previous_load = load_factor
            yield_capacity_factor = _interpolate_capacity(
                lower_load,
                lower_yield,
                previous_load,
                final_yield["max"],
                s3_major_yield_limit,
            )
            break
        previous_load = load_factor
        previous_yield = final_yield["max"]
        if config.use_accelerated_s3_continuation or previous_load >= 2.0:
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
    continuation["refinement_steps"] = refinement_steps

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
        "s3_major_yield_utilization_limit": s3_major_yield_limit,
        "s3_major_yield_reserve_factor": config.s3_major_yield_reserve_factor,
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
        "continuation_geometric_coupling": runtime.geometric_coupling,
        "continuation": continuation,
        "buckling_strength": buckling_strength,
    } if config.include_solver_diagnostics else {
        "collapse_state": collapse_state,
        "capacity_factor": reported_ultimate_capacity_factor,
        "raw_capacity_factor": raw_ultimate_capacity_factor,
        "yield_capacity_factor": yield_capacity_factor,
        "global_elastic_cutoff_factor": global_elastic_cutoff_factor,
        "max_newton_iterations": max_iterations,
        "mode_count": len(modes),
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
    runtime = _build_ritz_runtime(panel, modes, config)
    amplitudes = [0.0 for _ in modes]
    amplitudes, pressure_converged, pressure_iterations = solve_equilibrium_amplitudes(
        panel,
        modes,
        0.0,
        amplitudes,
        config,
        runtime,
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

    pressure_yield = u3_yield_utilization(panel, modes, amplitudes, 0.0, config, runtime)
    pressure_preload_response = {
        "iterations": pressure_iterations,
        "amplitudes": _mode_amplitude_summary(modes, amplitudes),
        "yield_utilization": pressure_yield,
        "controlling_yield_branch": "plate",
    }
    pressure_limits = u3_pressure_capacity_limit(panel)
    pressure_preload_response["pressure_capacity_limits"] = pressure_limits
    if (
        panel.pressure > pressure_limits["minimum"]
        or pressure_yield["max"] >= config.pressure_yield_limit
    ):
        return _attach_reliability(
            _invalid_u3_result(
                "pressure",
                {
                    "stage": "pressure-preload",
                    "yield_utilization": pressure_yield,
                    "pressure_capacity_limits": pressure_limits,
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
            runtime,
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
        final_yield = u3_yield_utilization(panel, modes, amplitudes, load_factor, config, runtime)
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
        "continuation_geometric_coupling": runtime.geometric_coupling,
        "continuation": continuation,
        "buckling_strength": buckling_strength,
    } if config.include_solver_diagnostics else {
        "panel_family": "U3",
        "collapse_state": collapse_state,
        "capacity_factor": reported_ultimate_capacity_factor,
        "raw_capacity_factor": raw_ultimate_capacity_factor,
        "yield_capacity_factor": yield_capacity_factor,
        "max_newton_iterations": max_iterations,
        "mode_count": len(modes),
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


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _absolute_distribution_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_absolute_error": None,
            "median_absolute_error": None,
            "p95_absolute_error": None,
            "max_absolute_error": None,
        }
    absolute_values = [abs(value) for value in values]
    return {
        "count": len(absolute_values),
        "mean_absolute_error": statistics.fmean(absolute_values),
        "median_absolute_error": statistics.median(absolute_values),
        "p95_absolute_error": _percentile(absolute_values, 0.95),
        "max_absolute_error": max(absolute_values),
    }


def _error_summary(pairs: Sequence[tuple[float, float]]) -> dict[str, float | int | None]:
    if not pairs:
        return {
            "count": 0,
            "mean_absolute_error": None,
            "median_absolute_error": None,
            "p95_absolute_error": None,
            "mean_relative_error": None,
            "max_absolute_error": None,
        }
    absolute_errors = [abs(predicted - expected) for expected, predicted in pairs]
    relative_errors = [
        abs(predicted - expected) / max(abs(expected), EPS)
        for expected, predicted in pairs
    ]
    return {
        "count": len(pairs),
        "mean_absolute_error": statistics.fmean(absolute_errors),
        "median_absolute_error": statistics.median(absolute_errors),
        "p95_absolute_error": _percentile(absolute_errors, 0.95),
        "mean_relative_error": statistics.fmean(relative_errors),
        "max_absolute_error": max(absolute_errors),
    }


def _directional_error_summaries(
    pairs: Sequence[tuple[float, float]],
) -> tuple[dict[str, float | int | None], dict[str, float | int | None]]:
    """Return conservative and nonconservative usage-factor error summaries."""

    signed_errors = [predicted - expected for expected, predicted in pairs]
    conservative = [error for error in signed_errors if error >= 0.0]
    nonconservative = [error for error in signed_errors if error < 0.0]
    conservative_summary = _absolute_distribution_summary(conservative)
    nonconservative_summary = _absolute_distribution_summary(nonconservative)
    nonconservative_summary["p95_nonconservative_error"] = nonconservative_summary[
        "p95_absolute_error"
    ]
    return conservative_summary, nonconservative_summary


def _usage_region(usage_factor: float | None) -> str:
    if usage_factor is None:
        return "invalid-target"
    if usage_factor <= 0.87:
        return "<=0.87"
    if usage_factor <= 0.91:
        return "0.87..0.91"
    if usage_factor < 1.0:
        return "0.91..1.0"
    return ">=1.0"


def _dominant_failure_family_share_key(result: S3Result) -> str:
    if not result.valid:
        return f"invalid:{result.invalid_reason or 'unknown'}"
    modeled_families = result.diagnostics.get("buckling", {}).get("modeled_failure_families", {})
    if not modeled_families:
        return "none"
    family, summary = max(
        modeled_families.items(),
        key=lambda item: float(item[1].get("usage_share_percent", 0.0)),
    )
    share = float(summary.get("usage_share_percent", 0.0))
    if share >= 75.0:
        band = ">=75%"
    elif share >= 50.0:
        band = "50..75%"
    else:
        band = "<50%"
    return f"{family}:{band}"


def _web_local_coverage_key(result: S3Result) -> str:
    if not result.valid:
        return f"invalid:{result.invalid_reason or 'unknown'}"
    buckling = result.diagnostics.get("buckling", {})
    if buckling.get("critical_failure_family") != "web-local":
        return "not-critical"
    web_factor = buckling.get("stiffener_web_local") or {}
    return str(web_factor.get("coverage") or "unknown")


def _local_interaction_coverage_key(result: S3Result) -> str:
    if not result.valid:
        return f"invalid:{result.invalid_reason or 'unknown'}"
    buckling = result.diagnostics.get("buckling", {})
    if buckling.get("critical_failure_family") != "plate-web-local-interaction":
        return "not-critical"
    interaction = buckling.get("local_plate_web_interaction") or {}
    return str(interaction.get("coverage") or "unknown")


def _puls_manual_limit_key(result: S3Result) -> str:
    if not result.valid:
        return f"invalid:{result.invalid_reason or 'unknown'}"
    manual = (
        result.diagnostics
        .get("validation_domain", {})
        .get("puls_manual_reference", {})
    )
    failed = manual.get("failed") or []
    if not failed:
        return "within-manual-reference"
    return "+".join(str(item) for item in failed)


def _buckling_strength_control_key(result: S3Result) -> str:
    if not result.valid:
        return f"invalid:{result.invalid_reason or 'unknown'}"
    buckling_strength = result.diagnostics.get("buckling_strength") or {}
    return str(buckling_strength.get("controlling_limit") or "unknown")


@dataclass
class _Slice:
    rows: int = 0
    target_valid: int = 0
    predicted_valid: int = 0
    buckling_pairs: list[tuple[float, float]] = field(default_factory=list)
    ultimate_pairs: list[tuple[float, float]] = field(default_factory=list)

    def add(
        self,
        target_valid: bool,
        predicted_valid: bool,
        buckling_pair: tuple[float, float] | None,
        ultimate_pair: tuple[float, float] | None,
    ) -> None:
        self.rows += 1
        self.target_valid += int(target_valid)
        self.predicted_valid += int(predicted_valid)
        if buckling_pair is not None:
            self.buckling_pairs.append(buckling_pair)
        if ultimate_pair is not None:
            self.ultimate_pairs.append(ultimate_pair)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "target_valid": self.target_valid,
            "predicted_valid": self.predicted_valid,
            "buckling": _error_summary(self.buckling_pairs),
            "ultimate": _error_summary(self.ultimate_pairs),
        }


@dataclass
class _ResidualCluster:
    rows: int = 0
    target_valid: int = 0
    predicted_valid: int = 0
    buckling_signed_errors: list[float] = field(default_factory=list)
    ultimate_signed_errors: list[float] = field(default_factory=list)
    max_buckling_abs_error: float = 0.0
    max_buckling_row: int | None = None

    def add(
        self,
        target_valid: bool,
        predicted_valid: bool,
        row_index: int,
        buckling_pair: tuple[float, float] | None,
        ultimate_pair: tuple[float, float] | None,
    ) -> None:
        self.rows += 1
        self.target_valid += int(target_valid)
        self.predicted_valid += int(predicted_valid)
        if buckling_pair is not None:
            signed_error = buckling_pair[1] - buckling_pair[0]
            self.buckling_signed_errors.append(signed_error)
            absolute_error = abs(signed_error)
            if absolute_error >= self.max_buckling_abs_error:
                self.max_buckling_abs_error = absolute_error
                self.max_buckling_row = row_index
        if ultimate_pair is not None:
            self.ultimate_signed_errors.append(ultimate_pair[1] - ultimate_pair[0])

    def to_dict(self) -> dict[str, Any]:
        buckling_abs = [abs(value) for value in self.buckling_signed_errors]
        ultimate_abs = [abs(value) for value in self.ultimate_signed_errors]
        return {
            "rows": self.rows,
            "target_valid": self.target_valid,
            "predicted_valid": self.predicted_valid,
            "buckling_mae": statistics.fmean(buckling_abs) if buckling_abs else None,
            "buckling_bias": statistics.fmean(self.buckling_signed_errors)
            if self.buckling_signed_errors
            else None,
            "ultimate_mae": statistics.fmean(ultimate_abs) if ultimate_abs else None,
            "ultimate_bias": statistics.fmean(self.ultimate_signed_errors)
            if self.ultimate_signed_errors
            else None,
            "max_buckling_abs_error": self.max_buckling_abs_error if buckling_abs else None,
            "max_buckling_row": self.max_buckling_row,
        }


@dataclass
class BenchmarkReport:
    rows: int
    target_numeric_rows: int
    predicted_valid_rows: int
    validity_confusion: dict[str, int]
    invalid_reason_counts: dict[str, int]
    target_invalid_reason_counts: dict[str, int]
    buckling: dict[str, float | int | None]
    elastic_buckling: dict[str, float | int | None]
    ultimate: dict[str, float | int | None]
    slices: dict[str, dict[str, dict[str, Any]]]
    worst_buckling_rows: list[dict[str, Any]]
    title: str = "Reduced S3 benchmark report"
    passed: bool | None = None
    acceptance: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    uf_order_violations: int = 0
    confidence_counts: dict[str, int] = field(default_factory=dict)
    mode_convergence_summary: dict[str, Any] = field(default_factory=dict)
    sample_manifest: dict[str, Any] = field(default_factory=dict)
    conservative_error_summary: dict[str, Any] = field(default_factory=dict)
    nonconservative_error_summary: dict[str, Any] = field(default_factory=dict)
    residual_clusters: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_text(self) -> str:
        lines = [
            self.title,
            f"Rows: {self.rows}",
            f"Target numeric rows: {self.target_numeric_rows}",
            f"Predicted valid rows: {self.predicted_valid_rows}",
            f"Validity confusion: {self.validity_confusion}",
            f"Buckling Strength UF errors: {self.buckling}",
            f"Elastic buckling UF diagnostic errors: {self.elastic_buckling}",
            f"Ultimate UF errors: {self.ultimate}",
            f"Predicted invalid reasons: {self.invalid_reason_counts}",
            f"Target invalid reasons: {self.target_invalid_reason_counts}",
            f"UF order violations: {self.uf_order_violations}",
            f"Confidence counts: {self.confidence_counts}",
            f"Mode convergence summary: {self.mode_convergence_summary}",
            f"Conservative errors: {self.conservative_error_summary}",
            f"Nonconservative errors: {self.nonconservative_error_summary}",
        ]
        if self.passed is not None:
            lines.append(f"Acceptance passed: {self.passed}")
            lines.append(f"Acceptance: {self.acceptance}")
        for family, family_slices in self.slices.items():
            lines.append(f"{family} slices:")
            for key, summary in sorted(family_slices.items()):
                buckling = summary["buckling"]
                lines.append(
                    f"  {key}: rows={summary['rows']} target_valid={summary['target_valid']} "
                    f"predicted_valid={summary['predicted_valid']} "
                    f"buckling_mae={buckling['mean_absolute_error']}"
                )
        if self.worst_buckling_rows:
            lines.append("Worst buckling UF rows:")
            for row in self.worst_buckling_rows:
                lines.append(f"  {row}")
        return "\n".join(lines)


def _benchmark_rows(
    rows: Iterable[tuple[int, Mapping[str, Any]]],
    config: S3SolverConfig,
    input_mapper: Any = row_to_s3_input,
    solver: Any = solve_s3_panel,
    title: str = "Reduced S3 benchmark report",
    sample_manifest: Mapping[str, Any] | None = None,
    mode_convergence_sample_size: int | None = None,
) -> BenchmarkReport:
    buckling_pairs: list[tuple[float, float]] = []
    elastic_buckling_pairs: list[tuple[float, float]] = []
    ultimate_pairs: list[tuple[float, float]] = []
    worst_rows: list[dict[str, Any]] = []
    validity = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    invalid_reasons: dict[str, int] = {}
    target_invalid_reasons: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    mode_convergence_drifts: list[float] = []
    mode_convergence_enabled = 0
    mode_convergence_low = 0
    uf_order_violations = 0
    residual_cluster_accumulators: dict[str, dict[str, _ResidualCluster]] = {
        "failure_family": {},
        "stiffener_type": {},
        "support": {},
        "pressure": {},
        "load_family": {},
        "puls_manual_reference": {},
        "combined": {},
    }
    slices: dict[str, dict[str, _Slice]] = {
        "support": {},
        "stiffener_type": {},
        "pressure": {},
        "usage_region": {},
        "predicted_failure_family": {},
        "dominant_failure_family_share": {},
        "web_local_coverage": {},
        "local_interaction_coverage": {},
        "puls_manual_reference": {},
        "buckling_strength_control": {},
    }
    row_count = 0
    target_numeric_rows = 0
    predicted_valid_rows = 0

    for row_index, row in rows:
        row_count += 1
        expected_buckling = _optional_float(row.get("Buckling Actual usage Factor inc NaN"))
        expected_ultimate = _optional_float(row.get("Ultimate Actual usage Factor inc NaN"))
        target_valid = expected_buckling is not None and expected_ultimate is not None
        target_numeric_rows += int(target_valid)

        row_config = config
        if (
            mode_convergence_sample_size is not None
            and config.check_mode_convergence
            and row_count > mode_convergence_sample_size
        ):
            row_config = replace(config, check_mode_convergence=False)

        try:
            result = solver(input_mapper(row), row_config)
        except (TypeError, ValueError) as exc:
            result = _invalid_result("csv-row-error", {"error": str(exc)})

        predicted_valid = (
            result.valid
            and result.buckling_usage_factor is not None
            and result.ultimate_usage_factor is not None
        )
        predicted_valid_rows += int(predicted_valid)
        confidence_counts[result.confidence] = confidence_counts.get(result.confidence, 0) + 1
        mode_convergence = result.diagnostics.get("mode_convergence", {})
        if mode_convergence.get("enabled"):
            mode_convergence_enabled += 1
            drift = _optional_float(mode_convergence.get("max_relative_drift"))
            if drift is not None:
                mode_convergence_drifts.append(drift)
                if drift > config.medium_confidence_drift_limit:
                    mode_convergence_low += 1
        if target_valid and predicted_valid:
            validity["true_positive"] += 1
        elif target_valid and not predicted_valid:
            validity["false_negative"] += 1
        elif not target_valid and predicted_valid:
            validity["false_positive"] += 1
        else:
            validity["true_negative"] += 1

        if not predicted_valid:
            reason = result.invalid_reason or "invalid-without-reason"
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        if not target_valid:
            target_reason = str(row.get("output cl str buc", "") or "invalid-without-reason").strip()
            target_invalid_reasons[target_reason] = target_invalid_reasons.get(target_reason, 0) + 1

        buckling_pair = None
        ultimate_pair = None
        if target_valid and predicted_valid:
            buckling_pair = (expected_buckling, float(result.buckling_usage_factor))
            ultimate_pair = (expected_ultimate, float(result.ultimate_usage_factor))
            if ultimate_pair[1] > buckling_pair[1] + EPS:
                uf_order_violations += 1
            buckling_pairs.append(buckling_pair)
            if result.elastic_buckling_usage_factor is not None:
                elastic_buckling_pairs.append(
                    (expected_buckling, float(result.elastic_buckling_usage_factor))
                )
            ultimate_pairs.append(ultimate_pair)
            absolute_error = abs(buckling_pair[1] - buckling_pair[0])
            modeled_families = result.diagnostics["buckling"]["modeled_failure_families"]
            family_shares = {
                family: summary["usage_share_percent"]
                for family, summary in modeled_families.items()
            }
            worst_rows.append(
                {
                    "row_index": row_index,
                    "source": row.get("_source_file", ""),
                    "ship_line": row.get("_ship_line", ""),
                    "expected_buckling": buckling_pair[0],
                    "predicted_buckling": buckling_pair[1],
                    "absolute_error": absolute_error,
                    "support": row.get("In-plane support", ""),
                    "stiffener_type": row.get("Stiffener type", row.get("Panel family", "")),
                    "pressure": row.get("Pressure (fixed)", ""),
                    "predicted_failure_family": result.diagnostics["buckling"]["critical_failure_family"],
                    "failure_family_shares": family_shares,
                }
            )
            worst_rows.sort(key=lambda item: item["absolute_error"], reverse=True)
            del worst_rows[10:]

        failure_family = (
            result.diagnostics["buckling"]["critical_failure_family"]
            if predicted_valid
            else f"invalid:{result.invalid_reason or 'unknown'}"
        )
        pressure_key = "nonzero" if (_optional_float(row.get("Pressure (fixed)")) or 0.0) > 0.0 else "zero"
        load_family = (
            result.diagnostics.get("validation_domain", {}).get("load_family", "unknown")
            if result.valid
            else f"invalid:{result.invalid_reason or 'unknown'}"
        )
        cluster_keys = {
            "failure_family": failure_family,
            "stiffener_type": str(row.get("Stiffener type", row.get("Panel family", "")) or "missing"),
            "support": str(row.get("In-plane support", "") or "missing"),
            "pressure": pressure_key,
            "load_family": str(load_family),
            "puls_manual_reference": _puls_manual_limit_key(result),
            "combined": "|".join(
                (
                    failure_family,
                    str(row.get("Stiffener type", row.get("Panel family", "")) or "missing"),
                    str(row.get("In-plane support", "") or "missing"),
                    pressure_key,
                    str(load_family),
                )
            ),
        }
        for cluster_family, cluster_key in cluster_keys.items():
            residual_cluster_accumulators[cluster_family].setdefault(
                cluster_key,
                _ResidualCluster(),
            ).add(target_valid, predicted_valid, row_index, buckling_pair, ultimate_pair)

        keys = {
            "support": str(row.get("In-plane support", "") or "missing"),
            "stiffener_type": str(row.get("Stiffener type", row.get("Panel family", "")) or "missing"),
            "pressure": pressure_key,
            "usage_region": _usage_region(expected_buckling),
            "predicted_failure_family": (
                failure_family
            ),
            "dominant_failure_family_share": _dominant_failure_family_share_key(result),
            "web_local_coverage": _web_local_coverage_key(result),
            "local_interaction_coverage": _local_interaction_coverage_key(result),
            "puls_manual_reference": _puls_manual_limit_key(result),
            "buckling_strength_control": _buckling_strength_control_key(result),
        }
        for family, key in keys.items():
            slice_accumulator = slices[family].setdefault(key, _Slice())
            slice_accumulator.add(target_valid, predicted_valid, buckling_pair, ultimate_pair)

    conservative_buckling, nonconservative_buckling = _directional_error_summaries(buckling_pairs)
    conservative_ultimate, nonconservative_ultimate = _directional_error_summaries(ultimate_pairs)
    residual_clusters = {}
    for cluster_family, cluster_values in residual_cluster_accumulators.items():
        ordered = sorted(
            ((key, value.to_dict()) for key, value in cluster_values.items()),
            key=lambda item: (
                item[1]["buckling_mae"] is None,
                -(item[1]["buckling_mae"] or 0.0),
                item[0],
            ),
        )
        residual_clusters[cluster_family] = dict(ordered[:50])

    return BenchmarkReport(
        rows=row_count,
        target_numeric_rows=target_numeric_rows,
        predicted_valid_rows=predicted_valid_rows,
        validity_confusion=validity,
        invalid_reason_counts=dict(sorted(invalid_reasons.items())),
        target_invalid_reason_counts=dict(sorted(target_invalid_reasons.items())),
        buckling=_error_summary(buckling_pairs),
        elastic_buckling=_error_summary(elastic_buckling_pairs),
        ultimate=_error_summary(ultimate_pairs),
        slices={
            family: {key: value.to_dict() for key, value in family_slices.items()}
            for family, family_slices in slices.items()
        },
        worst_buckling_rows=worst_rows,
        title=title,
        uf_order_violations=uf_order_violations,
        confidence_counts=dict(sorted(confidence_counts.items())),
        mode_convergence_summary={
            "enabled_rows": mode_convergence_enabled,
            "low_convergence_rows": mode_convergence_low,
            "max_relative_drift": max(mode_convergence_drifts) if mode_convergence_drifts else None,
            "median_relative_drift": statistics.median(mode_convergence_drifts) if mode_convergence_drifts else None,
            "p95_relative_drift": _percentile(mode_convergence_drifts, 0.95),
        },
        sample_manifest=dict(sample_manifest or {}),
        conservative_error_summary={
            "buckling": conservative_buckling,
            "ultimate": conservative_ultimate,
        },
        nonconservative_error_summary={
            "buckling": nonconservative_buckling,
            "ultimate": nonconservative_ultimate,
        },
        residual_clusters=residual_clusters,
    )


def _summary_metric(report: BenchmarkReport, metric_path: str) -> float | None:
    current: Any = report.to_dict()
    for part in metric_path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return _optional_float(current)


def _baseline_metric(baseline: Mapping[str, Any] | None, benchmark_name: str, metric_path: str) -> float | None:
    if not baseline:
        return None
    current: Any = baseline.get(benchmark_name, baseline)
    for part in metric_path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return _optional_float(current)


def apply_acceptance_gates(
    report: BenchmarkReport,
    benchmark_name: str,
    thresholds: Mapping[str, Any],
    baseline: Mapping[str, Any] | None = None,
) -> BenchmarkReport:
    """Attach pass/fail acceptance diagnostics to a benchmark report."""

    checks: dict[str, dict[str, Any]] = {}

    def add_check(name: str, actual: Any, limit: Any, passed: bool) -> None:
        checks[name] = {
            "actual": actual,
            "limit": limit,
            "passed": bool(passed),
        }

    if "max_buckling_mae" in thresholds:
        actual = _summary_metric(report, "buckling.mean_absolute_error")
        limit = float(thresholds["max_buckling_mae"])
        add_check("max_buckling_mae", actual, limit, actual is not None and actual <= limit)
    if "max_ultimate_mae" in thresholds:
        actual = _summary_metric(report, "ultimate.mean_absolute_error")
        limit = float(thresholds["max_ultimate_mae"])
        add_check("max_ultimate_mae", actual, limit, actual is not None and actual <= limit)
    if "max_validity_mismatches" in thresholds:
        actual = report.validity_confusion.get("false_positive", 0) + report.validity_confusion.get(
            "false_negative",
            0,
        )
        limit = int(thresholds["max_validity_mismatches"])
        add_check("max_validity_mismatches", actual, limit, actual <= limit)
    if "max_uf_order_violations" in thresholds:
        actual = report.uf_order_violations
        limit = int(thresholds["max_uf_order_violations"])
        add_check("max_uf_order_violations", actual, limit, actual <= limit)

    if "baseline_buckling_mae_delta" in thresholds:
        actual = _summary_metric(report, "buckling.mean_absolute_error")
        baseline_value = _baseline_metric(baseline, benchmark_name, "buckling.mean_absolute_error")
        delta_limit = float(thresholds["baseline_buckling_mae_delta"])
        limit = None if baseline_value is None else baseline_value + delta_limit
        add_check(
            "baseline_buckling_mae_delta",
            actual,
            {
                "baseline": baseline_value,
                "allowed_delta": delta_limit,
                "limit": limit,
            },
            actual is not None and limit is not None and actual <= limit,
        )
    if "baseline_ultimate_mae_delta" in thresholds:
        actual = _summary_metric(report, "ultimate.mean_absolute_error")
        baseline_value = _baseline_metric(baseline, benchmark_name, "ultimate.mean_absolute_error")
        delta_limit = float(thresholds["baseline_ultimate_mae_delta"])
        limit = None if baseline_value is None else baseline_value + delta_limit
        add_check(
            "baseline_ultimate_mae_delta",
            actual,
            {
                "baseline": baseline_value,
                "allowed_delta": delta_limit,
                "limit": limit,
            },
            actual is not None and limit is not None and actual <= limit,
        )
    if "baseline_validity_mismatch_delta" in thresholds:
        actual = report.validity_confusion.get("false_positive", 0) + report.validity_confusion.get(
            "false_negative",
            0,
        )
        baseline_false_positive = _baseline_metric(
            baseline,
            benchmark_name,
            "validity_confusion.false_positive",
        )
        baseline_false_negative = _baseline_metric(
            baseline,
            benchmark_name,
            "validity_confusion.false_negative",
        )
        if baseline_false_positive is None or baseline_false_negative is None:
            baseline_mismatches = None
            limit = None
        else:
            baseline_mismatches = int(baseline_false_positive + baseline_false_negative)
            limit = baseline_mismatches + int(thresholds["baseline_validity_mismatch_delta"])
        add_check(
            "baseline_validity_mismatch_delta",
            actual,
            {
                "baseline": baseline_mismatches,
                "allowed_delta": int(thresholds["baseline_validity_mismatch_delta"]),
                "limit": limit,
            },
            limit is not None and actual <= limit,
        )
    if "baseline_buckling_relative_mae_factor" in thresholds:
        actual = _summary_metric(report, "buckling.mean_relative_error")
        baseline_value = _baseline_metric(baseline, benchmark_name, "buckling.mean_relative_error")
        factor = float(thresholds["baseline_buckling_relative_mae_factor"])
        limit = None if baseline_value is None else baseline_value * factor
        add_check(
            "baseline_buckling_relative_mae_factor",
            actual,
            {
                "baseline": baseline_value,
                "factor": factor,
                "limit": limit,
            },
            actual is not None and limit is not None and actual <= limit,
        )

    report.acceptance = {"benchmark": benchmark_name, "checks": checks}
    report.baseline = dict(baseline.get(benchmark_name, {})) if isinstance(baseline, Mapping) else {}
    report.passed = all(check["passed"] for check in checks.values()) if checks else True
    return report


@dataclass(frozen=True)
class CsvSampleSelection:
    row_indices: list[int] | None
    manifest: dict[str, Any]


def _csv_sampling_filter_active(
    target_valid_only: bool,
    target_uf_min: float | None,
    target_uf_max: float | None,
) -> bool:
    return target_valid_only or target_uf_min is not None or target_uf_max is not None


def _csv_sampling_filter_manifest(
    target_valid_only: bool,
    target_uf_min: float | None,
    target_uf_max: float | None,
    select_filtered_out: bool,
) -> dict[str, Any]:
    return {
        "target_valid_only": target_valid_only,
        "target_buckling_uf_min": target_uf_min,
        "target_buckling_uf_max": target_uf_max,
        "filter_mode": "filtered-out-monitor" if select_filtered_out else "included",
    }


def _csv_sampling_filter_reason(
    row: Mapping[str, Any],
    *,
    target_valid_only: bool,
    target_uf_min: float | None,
    target_uf_max: float | None,
) -> str:
    expected_buckling = _optional_float(row.get("Buckling Actual usage Factor inc NaN"))
    expected_ultimate = _optional_float(row.get("Ultimate Actual usage Factor inc NaN"))
    target_valid = expected_buckling is not None and expected_ultimate is not None
    if target_valid_only and not target_valid:
        return "invalid-target"
    if target_uf_min is not None or target_uf_max is not None:
        if not target_valid:
            return "invalid-target"
        assert expected_buckling is not None
        if target_uf_min is not None and expected_buckling < target_uf_min:
            return "below-uf-window"
        if target_uf_max is not None and expected_buckling > target_uf_max:
            return "above-uf-window"
    return "included"


def _csv_row_selected_by_filter(
    row: Mapping[str, Any],
    *,
    target_valid_only: bool,
    target_uf_min: float | None,
    target_uf_max: float | None,
    select_filtered_out: bool,
) -> tuple[bool, str]:
    reason = _csv_sampling_filter_reason(
        row,
        target_valid_only=target_valid_only,
        target_uf_min=target_uf_min,
        target_uf_max=target_uf_max,
    )
    if select_filtered_out:
        return reason != "included", reason
    return reason == "included", reason


def _csv_pressure_band(pressure: float | None) -> str:
    if pressure is None or pressure <= EPS:
        return "zero"
    if pressure <= 0.05:
        return "0..0.05"
    if pressure <= 0.15:
        return "0.05..0.15"
    if pressure <= 0.30:
        return "0.15..0.30"
    return ">0.30"


def _csv_stratification_fields(row: Mapping[str, Any]) -> dict[str, str]:
    pressure = _optional_float(row.get("Pressure (fixed)"))
    expected_buckling = _optional_float(row.get("Buckling Actual usage Factor inc NaN"))
    expected_ultimate = _optional_float(row.get("Ultimate Actual usage Factor inc NaN"))
    target_valid = expected_buckling is not None and expected_ultimate is not None
    pressure_value = pressure or 0.0
    return {
        "target_validity": "numeric" if target_valid else "invalid",
        "stiffener_type": str(row.get("Stiffener type", "") or "missing"),
        "support": str(row.get("In-plane support", "") or "missing"),
        "stiffener_boundary": str(row.get("Stiffener boundary", "") or "missing"),
        "pressure_state": "nonzero" if pressure_value > EPS else "zero",
        "pressure_band": _csv_pressure_band(pressure),
        "usage_region": _usage_region(expected_buckling),
    }


def _csv_stratum_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    fields = _csv_stratification_fields(row)
    return tuple(fields[key] for key in CSV_STRATIFICATION_KEYS)


def _sample_manifest(
    *,
    method: str,
    seed: int,
    requested_sample_size: int | None,
    row_count: int,
    selected_indices: Sequence[int] | None,
    eligible_row_count: int | None = None,
    filter_counts: Mapping[str, int] | None = None,
    filter_manifest: Mapping[str, Any] | None = None,
    stratum_total_counts: Mapping[tuple[str, ...], int] | None = None,
) -> dict[str, Any]:
    stratum_total_counts = stratum_total_counts or {}
    selected_set = set(selected_indices or [])
    selected_stratum_counts: dict[tuple[str, ...], int] = {}
    if selected_set and stratum_total_counts:
        # Filled by select_csv_sample_indices for stratified samples.
        selected_stratum_counts = {}
    return {
        "method": method,
        "seed": seed,
        "requested_sample_size": requested_sample_size,
        "row_count": row_count,
        "eligible_row_count": row_count if eligible_row_count is None else eligible_row_count,
        "selected_count": row_count if selected_indices is None else len(selected_indices),
        "selected_row_indices": list(selected_indices or []) if selected_indices is not None else None,
        "filter": dict(filter_manifest or {}),
        "filter_counts": dict(sorted((filter_counts or {}).items())),
        "stratification_keys": list(CSV_STRATIFICATION_KEYS),
        "stratum_counts": {
            "|".join(key): count for key, count in sorted(stratum_total_counts.items())
        },
        "selected_stratum_counts": {
            "|".join(key): count for key, count in sorted(selected_stratum_counts.items())
        },
    }


def select_csv_sample_indices(
    csv_path: str | Path,
    sample_method: str = CSV_SAMPLE_METHOD_STRATIFIED,
    sample_size: int | None = DEFAULT_CSV_SAMPLE_SIZE,
    seed: int = DEFAULT_CSV_SAMPLE_SEED,
    target_valid_only: bool = False,
    target_uf_min: float | None = None,
    target_uf_max: float | None = None,
    select_filtered_out: bool = False,
) -> CsvSampleSelection:
    """Return deterministic CSV row indices and a manifest for benchmark sampling."""

    if sample_method not in {
        CSV_SAMPLE_METHOD_FIRST_N,
        CSV_SAMPLE_METHOD_STRATIFIED,
        CSV_SAMPLE_METHOD_FULL,
    }:
        raise ValueError(f"Unsupported CSV sample method: {sample_method}")

    path = Path(csv_path)
    filter_active = _csv_sampling_filter_active(target_valid_only, target_uf_min, target_uf_max)
    filter_manifest = _csv_sampling_filter_manifest(
        target_valid_only,
        target_uf_min,
        target_uf_max,
        select_filtered_out,
    )
    if sample_method == CSV_SAMPLE_METHOD_FULL:
        if not filter_active and not select_filtered_out:
            row_count = sum(1 for _ in iter_csv_rows(path))
            return CsvSampleSelection(
                None,
                _sample_manifest(
                    method=sample_method,
                    seed=seed,
                    requested_sample_size=None,
                    row_count=row_count,
                    selected_indices=None,
                    filter_manifest=filter_manifest,
                ),
            )
        indices: list[int] = []
        filter_counts: dict[str, int] = {}
        row_count = 0
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                row_count += 1
                selected, reason = _csv_row_selected_by_filter(
                    row,
                    target_valid_only=target_valid_only,
                    target_uf_min=target_uf_min,
                    target_uf_max=target_uf_max,
                    select_filtered_out=select_filtered_out,
                )
                filter_counts[reason] = filter_counts.get(reason, 0) + 1
                if selected:
                    indices.append(row_index)
        return CsvSampleSelection(
            indices,
            _sample_manifest(
                method=sample_method,
                seed=seed,
                requested_sample_size=None,
                row_count=row_count,
                selected_indices=indices,
                eligible_row_count=len(indices),
                filter_counts=filter_counts,
                filter_manifest=filter_manifest,
            ),
        )

    if sample_size is None or sample_size <= 0:
        raise ValueError("CSV sample size must be a positive integer")

    if sample_method == CSV_SAMPLE_METHOD_FIRST_N:
        indices: list[int] = []
        filter_counts: dict[str, int] = {}
        row_count = 0
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                row_count += 1
                selected, reason = _csv_row_selected_by_filter(
                    row,
                    target_valid_only=target_valid_only,
                    target_uf_min=target_uf_min,
                    target_uf_max=target_uf_max,
                    select_filtered_out=select_filtered_out,
                )
                filter_counts[reason] = filter_counts.get(reason, 0) + 1
                if selected and len(indices) < sample_size:
                    indices.append(row_index)
        return CsvSampleSelection(
            indices,
            _sample_manifest(
                method=sample_method,
                seed=seed,
                requested_sample_size=sample_size,
                row_count=row_count,
                selected_indices=indices,
                eligible_row_count=sum(
                    count
                    for reason, count in filter_counts.items()
                    if (reason != "included") == select_filtered_out
                    or (reason == "included" and not select_filtered_out)
                ),
                filter_counts=filter_counts,
                filter_manifest=filter_manifest,
            ),
        )

    rng = random.Random(seed)
    groups: dict[tuple[str, ...], list[int]] = {}
    dimension_values: dict[str, set[str]] = {key: set() for key in CSV_STRATIFICATION_KEYS}
    filter_counts: dict[str, int] = {}
    row_count = 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            row_count += 1
            selected, reason = _csv_row_selected_by_filter(
                row,
                target_valid_only=target_valid_only,
                target_uf_min=target_uf_min,
                target_uf_max=target_uf_max,
                select_filtered_out=select_filtered_out,
            )
            filter_counts[reason] = filter_counts.get(reason, 0) + 1
            if not selected:
                continue
            fields = _csv_stratification_fields(row)
            for key, value in fields.items():
                dimension_values[key].add(value)
            groups.setdefault(tuple(fields[key] for key in CSV_STRATIFICATION_KEYS), []).append(row_index)

    eligible_row_count = sum(len(indices) for indices in groups.values())
    if eligible_row_count == 0:
        return CsvSampleSelection(
            [],
            _sample_manifest(
                method=sample_method,
                seed=seed,
                requested_sample_size=sample_size,
                row_count=row_count,
                selected_indices=[],
                eligible_row_count=0,
                filter_counts=filter_counts,
                filter_manifest=filter_manifest,
            ),
        )

    for indices in groups.values():
        rng.shuffle(indices)

    selected: list[int] = []
    selected_set: set[int] = set()

    def add_index(index: int) -> None:
        if len(selected) >= min(sample_size, eligible_row_count) or index in selected_set:
            return
        selected.append(index)
        selected_set.add(index)

    # Guarantee broad coverage for each stratification dimension before the
    # proportional allocation fills the rest of the sample.
    for key_position, key_name in enumerate(CSV_STRATIFICATION_KEYS):
        for value in sorted(dimension_values[key_name]):
            candidate_groups = [
                (len(indices), stratum_key, indices)
                for stratum_key, indices in groups.items()
                if stratum_key[key_position] == value
            ]
            rng.shuffle(candidate_groups)
            candidate_groups.sort(key=lambda item: item[0], reverse=True)
            for _, _, indices in candidate_groups:
                candidate = next((index for index in indices if index not in selected_set), None)
                if candidate is not None:
                    add_index(candidate)
                    break

    remaining_capacity = min(sample_size, eligible_row_count) - len(selected)
    stratum_total_counts = {key: len(indices) for key, indices in groups.items()}
    if remaining_capacity > 0:
        quotas: list[tuple[float, tuple[str, ...], int]] = []
        allocated: dict[tuple[str, ...], int] = {}
        for stratum_key, indices in groups.items():
            available = len([index for index in indices if index not in selected_set])
            if available <= 0:
                continue
            exact = remaining_capacity * len(indices) / eligible_row_count
            base = min(available, int(math.floor(exact)))
            allocated[stratum_key] = base
            quotas.append((exact - base, stratum_key, available))

        for stratum_key, count in allocated.items():
            for index in groups[stratum_key]:
                if count <= 0:
                    break
                if index in selected_set:
                    continue
                add_index(index)
                count -= 1

        while len(selected) < min(sample_size, eligible_row_count):
            progressed = False
            rng.shuffle(quotas)
            quotas.sort(key=lambda item: item[0], reverse=True)
            for _, stratum_key, _ in quotas:
                candidate = next(
                    (index for index in groups[stratum_key] if index not in selected_set),
                    None,
                )
                if candidate is None:
                    continue
                add_index(candidate)
                progressed = True
                if len(selected) >= min(sample_size, eligible_row_count):
                    break
            if not progressed:
                break

    rng.shuffle(selected)
    selected_stratum_counts: dict[tuple[str, ...], int] = {}
    index_to_stratum: dict[int, tuple[str, ...]] = {}
    for stratum_key, indices in groups.items():
        for index in indices:
            index_to_stratum[index] = stratum_key
    for index in selected:
        stratum_key = index_to_stratum[index]
        selected_stratum_counts[stratum_key] = selected_stratum_counts.get(stratum_key, 0) + 1
    manifest = {
        "method": sample_method,
        "seed": seed,
        "requested_sample_size": sample_size,
        "row_count": row_count,
        "eligible_row_count": eligible_row_count,
        "selected_count": len(selected),
        "selected_row_indices": selected,
        "filter": filter_manifest,
        "filter_counts": dict(sorted(filter_counts.items())),
        "stratification_keys": list(CSV_STRATIFICATION_KEYS),
        "dimension_counts": {
            key: {value: 0 for value in sorted(values)}
            for key, values in dimension_values.items()
        },
        "stratum_counts": {
            "|".join(key): count for key, count in sorted(stratum_total_counts.items())
        },
        "selected_stratum_counts": {
            "|".join(key): count for key, count in sorted(selected_stratum_counts.items())
        },
    }
    for stratum_key, count in stratum_total_counts.items():
        for key_name, value in zip(CSV_STRATIFICATION_KEYS, stratum_key):
            manifest["dimension_counts"][key_name][value] += count
    return CsvSampleSelection(selected, manifest)


def iter_csv_rows(
    csv_path: str | Path,
    limit: int | None = None,
    fixture: bool = False,
    sample_indices: Sequence[int] | None = None,
) -> Iterator[tuple[int, Mapping[str, str]]]:
    """Yield indexed CSV rows deterministically without loading the full file."""

    if fixture:
        selection = select_csv_sample_indices(
            csv_path,
            sample_method=CSV_SAMPLE_METHOD_STRATIFIED,
            sample_size=DEFAULT_CSV_FIXTURE_SAMPLE_SIZE,
            seed=DEFAULT_CSV_SAMPLE_SEED,
        )
        sample_indices = selection.row_indices
    if sample_indices is not None:
        selected_positions = {row_index: position for position, row_index in enumerate(sample_indices)}
        selected_rows: dict[int, tuple[int, Mapping[str, str]]] = {}
        with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
            for row_index, row in enumerate(csv.DictReader(handle)):
                position = selected_positions.get(row_index)
                if position is not None:
                    selected_rows[position] = (row_index, row)
                    if len(selected_rows) >= len(selected_positions):
                        break
        for position in range(len(sample_indices)):
            if position in selected_rows:
                yield selected_rows[position]
        return

    emitted = 0
    with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            yield row_index, row
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def iter_ship_section_rows(
    path: str | Path,
    limit: int | None = None,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield stiffened S3-like records from an ANYstructure ship-section export."""

    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    puls_results = payload.get("PULS results", {})
    emitted = 0
    for row_index, record in enumerate(puls_results.values()):
        if not isinstance(record, Mapping):
            continue
        if "Plate geometry" not in record or "Primary stiffeners" not in record:
            continue
        row = ship_section_record_to_csv_row(record)
        row["_source_file"] = source_path.name
        yield row_index, row
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def iter_ship_section_u3_rows(
    path: str | Path,
    limit: int | None = None,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    """Yield unstiffened U3-like records from an ANYstructure ship-section export."""

    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    puls_results = payload.get("PULS results", {})
    emitted = 0
    for row_index, record in enumerate(puls_results.values()):
        if not isinstance(record, Mapping):
            continue
        if "Geometry" not in record or "Primary stiffeners" in record:
            continue
        row = ship_section_record_to_u3_row(record)
        row["_source_file"] = source_path.name
        yield row_index, row
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def iter_ship_section_file_rows(
    paths: Sequence[str | Path],
    limit: int | None = None,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    emitted = 0
    for path in paths:
        for _, row in iter_ship_section_rows(path):
            yield emitted, row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def iter_ship_section_u3_file_rows(
    paths: Sequence[str | Path],
    limit: int | None = None,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    emitted = 0
    for path in paths:
        for _, row in iter_ship_section_u3_rows(path):
            yield emitted, row
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def benchmark_csv(
    csv_path: str | Path,
    config: S3SolverConfig | None = None,
    limit: int | None = None,
    fixture: bool = False,
    sample_method: str | None = None,
    sample_size: int | None = None,
    sample_seed: int = DEFAULT_CSV_SAMPLE_SEED,
    mode_convergence_sample_size: int | None = None,
    title: str = "Reduced S3 benchmark report",
    target_valid_only: bool = False,
    target_uf_min: float | None = None,
    target_uf_max: float | None = None,
) -> BenchmarkReport:
    config = config or S3SolverConfig()
    if fixture:
        sample_method = CSV_SAMPLE_METHOD_STRATIFIED
        sample_size = DEFAULT_CSV_FIXTURE_SAMPLE_SIZE if sample_size is None else sample_size
    if sample_method is None:
        if limit is not None:
            sample_method = CSV_SAMPLE_METHOD_FIRST_N
            sample_size = limit
        else:
            sample_method = CSV_SAMPLE_METHOD_FULL
    if sample_method == CSV_SAMPLE_METHOD_FULL:
        selection = select_csv_sample_indices(
            csv_path,
            sample_method=CSV_SAMPLE_METHOD_FULL,
            seed=sample_seed,
            target_valid_only=target_valid_only,
            target_uf_min=target_uf_min,
            target_uf_max=target_uf_max,
        )
        row_iterator = (
            iter_csv_rows(csv_path)
            if selection.row_indices is None
            else iter_csv_rows(csv_path, sample_indices=selection.row_indices)
        )
    else:
        selection = select_csv_sample_indices(
            csv_path,
            sample_method=sample_method,
            sample_size=sample_size,
            seed=sample_seed,
            target_valid_only=target_valid_only,
            target_uf_min=target_uf_min,
            target_uf_max=target_uf_max,
        )
        row_iterator = iter_csv_rows(csv_path, sample_indices=selection.row_indices)
    return _benchmark_rows(
        row_iterator,
        config,
        title=title,
        sample_manifest=selection.manifest,
        mode_convergence_sample_size=mode_convergence_sample_size,
    )


def benchmark_ship_section(
    path: str | Path,
    config: S3SolverConfig | None = None,
    limit: int | None = None,
) -> BenchmarkReport:
    return _benchmark_rows(iter_ship_section_rows(path, limit=limit), config or S3SolverConfig())


def benchmark_ship_sections(
    paths: Sequence[str | Path],
    config: S3SolverConfig | None = None,
    limit: int | None = None,
) -> BenchmarkReport:
    return _benchmark_rows(iter_ship_section_file_rows(paths, limit=limit), config or S3SolverConfig())


def benchmark_u3_ship_section(
    path: str | Path,
    config: S3SolverConfig | None = None,
    limit: int | None = None,
) -> BenchmarkReport:
    return _benchmark_rows(
        iter_ship_section_u3_rows(path, limit=limit),
        config or S3SolverConfig(),
        input_mapper=row_to_u3_input,
        solver=solve_u3_panel,
        title="Reduced U3 benchmark report",
    )


def benchmark_u3_ship_sections(
    paths: Sequence[str | Path],
    config: S3SolverConfig | None = None,
    limit: int | None = None,
) -> BenchmarkReport:
    return _benchmark_rows(
        iter_ship_section_u3_file_rows(paths, limit=limit),
        config or S3SolverConfig(),
        input_mapper=row_to_u3_input,
        solver=solve_u3_panel,
        title="Reduced U3 benchmark report",
    )


def _capacity_stress_from_factor(applied_stress: float, factor: Any) -> float | None:
    parsed_factor = _optional_float(factor)
    if parsed_factor is None:
        return None
    if abs(applied_stress) <= EPS:
        return None
    return abs(applied_stress) * parsed_factor


def _comparison_value(value: float | None) -> float | str:
    return "" if value is None else value


def _comparison_abs_error(solver_value: float | None, puls_value: Any) -> float | str:
    parsed_puls = _optional_float(puls_value)
    if solver_value is None or parsed_puls is None:
        return ""
    return abs(solver_value - parsed_puls)


def _s3_ship_section_intermediate_comparison(
    row: Mapping[str, Any],
    panel: S3PanelInput | None,
    result: S3Result,
) -> dict[str, Any]:
    """Return solver/PULS intermediate elastic-stress comparison columns.

    ANYstructure ship-section exports often include the PULS local/global
    elastic buckling stresses and failure-mode percentages.  The reduced
    solver stores factors, so stresses are reconstructed from the proportional
    applied load components.  Global stresses use the uncoupled global-strip
    elastic factor when available because that is closest to the exported
    PULS "Global elastic buckling" block.
    """

    puls_targets = {
        "global_axial": row.get("PULS global axial stress", ""),
        "global_trans": row.get("PULS global trans stress", ""),
        "global_shear": row.get("PULS global shear stress", ""),
        "local_axial": row.get("PULS local axial stress", ""),
        "local_trans": row.get("PULS local trans stress", ""),
        "local_shear": row.get("PULS local shear stress", ""),
    }
    puls_failures = {
        "plate_buckling": row.get("PULS failure plate buckling percent", ""),
        "global_stiffener_buckling": row.get("PULS failure global stiffener buckling percent", ""),
        "torsional_stiffener_buckling": row.get("PULS failure torsional stiffener buckling percent", ""),
        "web_stiffener_buckling": row.get("PULS failure web stiffener buckling percent", ""),
    }
    output: dict[str, Any] = {
        "puls_global_axial_stress": puls_targets["global_axial"],
        "solver_global_axial_stress": "",
        "global_axial_stress_abs_error": "",
        "puls_global_trans_stress": puls_targets["global_trans"],
        "solver_global_trans_stress": "",
        "global_trans_stress_abs_error": "",
        "puls_global_shear_stress": puls_targets["global_shear"],
        "solver_global_shear_stress": "",
        "global_shear_stress_abs_error": "",
        "puls_local_axial_stress": puls_targets["local_axial"],
        "solver_local_axial_stress": "",
        "local_axial_stress_abs_error": "",
        "puls_local_trans_stress": puls_targets["local_trans"],
        "solver_local_trans_stress": "",
        "local_trans_stress_abs_error": "",
        "puls_local_shear_stress": puls_targets["local_shear"],
        "solver_local_shear_stress": "",
        "local_shear_stress_abs_error": "",
        "solver_global_factor": "",
        "solver_global_uncoupled_factor": "",
        "solver_local_plate_factor": "",
        "solver_local_shear_factor": "",
        "puls_failure_plate_buckling_percent": puls_failures["plate_buckling"],
        "solver_failure_plate_buckling_percent": "",
        "failure_plate_buckling_percent_abs_error": "",
        "puls_failure_global_stiffener_buckling_percent": puls_failures["global_stiffener_buckling"],
        "solver_failure_global_stiffener_buckling_percent": "",
        "failure_global_stiffener_buckling_percent_abs_error": "",
        "puls_failure_torsional_stiffener_buckling_percent": puls_failures["torsional_stiffener_buckling"],
        "solver_failure_torsional_stiffener_buckling_percent": "",
        "failure_torsional_stiffener_buckling_percent_abs_error": "",
        "puls_failure_web_stiffener_buckling_percent": puls_failures["web_stiffener_buckling"],
        "solver_failure_web_stiffener_buckling_percent": "",
        "failure_web_stiffener_buckling_percent_abs_error": "",
    }
    if panel is None or not result.valid:
        return output

    buckling = result.diagnostics.get("buckling", {})
    modeled = buckling.get("modeled_failure_families", {})
    global_family = modeled.get("global-stiffened-strip", {})
    column_family = modeled.get("global-stiffener-cutoff", {})
    plate_family = modeled.get("plate", {})
    plate_shear_family = modeled.get("plate-shear", {})
    torsional_family = modeled.get("torsional-stiffener", {})
    web_family = modeled.get("web-local", {})

    def _family_uncoupled(family: Mapping[str, Any]) -> float | None:
        uncoupled = _optional_float(family.get("uncoupled_factor"))
        if uncoupled is not None:
            return uncoupled
        return _optional_float(family.get("critical_factor"))

    # PULS reports one GEB; the reduced model carries two global candidates
    # (orthotropic strip mode and the wide-panel cylindrical column limit), so
    # the comparison channel takes the governing minimum.
    global_candidates = [
        value
        for value in (
            _optional_float(global_family.get("critical_factor")),
            _optional_float(column_family.get("critical_factor")),
        )
        if value is not None
    ]
    global_factor = min(global_candidates) if global_candidates else None
    uncoupled_candidates = [
        value
        for value in (
            _family_uncoupled(global_family),
            _family_uncoupled(column_family),
        )
        if value is not None
    ]
    global_uncoupled_factor = min(uncoupled_candidates) if uncoupled_candidates else None
    if global_uncoupled_factor is None:
        global_uncoupled_factor = global_factor
    local_plate_factor = _optional_float(plate_family.get("critical_factor"))
    local_shear_factor = _optional_float(plate_shear_family.get("critical_factor"))

    output["solver_global_factor"] = _comparison_value(global_factor)
    output["solver_global_uncoupled_factor"] = _comparison_value(global_uncoupled_factor)
    output["solver_local_plate_factor"] = _comparison_value(local_plate_factor)
    output["solver_local_shear_factor"] = _comparison_value(local_shear_factor)

    # PULS reports GEB with the local-buckling reduced (secant) stiffness
    # coefficients applied, so the comparison channel uses the coupled factor.
    stress_pairs = {
        "global_axial": _capacity_stress_from_factor(panel.axial_stress, global_factor),
        "global_trans": _capacity_stress_from_factor(panel.mean_transverse_stress, global_factor),
        "global_shear": _capacity_stress_from_factor(panel.shear_stress, global_factor),
        "local_axial": _capacity_stress_from_factor(panel.axial_stress, local_plate_factor),
        "local_trans": _capacity_stress_from_factor(panel.mean_transverse_stress, local_plate_factor),
        "local_shear": _capacity_stress_from_factor(panel.shear_stress, local_shear_factor),
    }
    for name, solver_value in stress_pairs.items():
        output[f"solver_{name}_stress"] = _comparison_value(solver_value)
        output[f"{name}_stress_abs_error"] = _comparison_abs_error(
            solver_value,
            puls_targets[name],
        )

    family_to_failure_output = {
        "plate_buckling": plate_family,
        "global_stiffener_buckling": global_family,
        "torsional_stiffener_buckling": torsional_family,
        "web_stiffener_buckling": web_family,
    }
    for name, family in family_to_failure_output.items():
        solver_percent = _optional_float(family.get("usage_share_percent"))
        output[f"solver_failure_{name}_percent"] = _comparison_value(solver_percent)
        output[f"failure_{name}_percent_abs_error"] = _comparison_abs_error(
            solver_percent,
            puls_failures[name],
        )
    return output


def _summary_percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(math.ceil((percentile / 100.0) * len(ordered))) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return ordered[index]


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _summary_percentile(values, 95.0),
        "min": min(values),
        "max": max(values),
    }


def _s3_ship_intermediate_channel_summary(
    rows: Sequence[Mapping[str, str]],
    channel: str,
    group_key: str | None = None,
) -> dict[str, Any]:
    signed_errors: list[float] = []
    absolute_errors: list[float] = []
    ratios: list[float] = []
    grouped_rows: dict[str, list[Mapping[str, str]]] = {}
    puls_key = f"puls_{channel}"
    solver_key = f"solver_{channel}"
    error_key = f"{channel}_abs_error"
    for row in rows:
        puls = _optional_float(row.get(puls_key))
        solver = _optional_float(row.get(solver_key))
        absolute_error = _optional_float(row.get(error_key))
        if puls is None or solver is None:
            continue
        signed_errors.append(solver - puls)
        if absolute_error is None:
            absolute_error = abs(solver - puls)
        absolute_errors.append(absolute_error)
        if abs(puls) > EPS:
            ratios.append(solver / puls)
        if group_key is not None:
            grouped_rows.setdefault(row.get(group_key, "") or "missing", []).append(row)

    summary = {
        "count": len(signed_errors),
        "signed_error": _numeric_summary(signed_errors),
        "absolute_error": _numeric_summary(absolute_errors),
        "solver_to_puls_ratio": _numeric_summary(ratios),
    }
    if group_key is not None:
        summary["by_" + group_key] = {
            group: _s3_ship_intermediate_channel_summary(group_rows, channel)
            for group, group_rows in sorted(grouped_rows.items())
        }
    return summary


def summarize_s3_ship_intermediate_report(report_path: str | Path) -> dict[str, Any]:
    """Summarize PULS intermediate stress and failure-mode residuals.

    This keeps the real-world ship-section intermediate blocks visible in the
    machine-readable verification report without changing solver acceptance
    gates.  The row-level CSV remains the source for detailed investigation.
    """

    path = Path(report_path)
    if not path.exists():
        return {"available": False, "reason": "missing-report"}
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stress_channels = [
        "global_axial_stress",
        "global_trans_stress",
        "global_shear_stress",
        "local_axial_stress",
        "local_trans_stress",
        "local_shear_stress",
    ]
    failure_channels = [
        "failure_plate_buckling_percent",
        "failure_global_stiffener_buckling_percent",
        "failure_torsional_stiffener_buckling_percent",
        "failure_web_stiffener_buckling_percent",
    ]
    return {
        "available": True,
        "row_count": len(rows),
        "elastic_stress": {
            channel: _s3_ship_intermediate_channel_summary(rows, channel, group_key="support")
            for channel in stress_channels
        },
        "failure_mode_percent": {
            channel: _s3_ship_intermediate_channel_summary(rows, channel, group_key="support")
            for channel in failure_channels
        },
    }


def write_ship_section_comparison_csv(
    paths: Sequence[str | Path],
    output_path: str | Path,
    config: S3SolverConfig | None = None,
    limit: int | None = None,
) -> Path:
    """Write row-level S3/PULS comparison for ANYstructure section exports."""

    config = config or S3SolverConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    intermediate_fieldnames = [
        "puls_global_axial_stress",
        "solver_global_axial_stress",
        "global_axial_stress_abs_error",
        "puls_global_trans_stress",
        "solver_global_trans_stress",
        "global_trans_stress_abs_error",
        "puls_global_shear_stress",
        "solver_global_shear_stress",
        "global_shear_stress_abs_error",
        "puls_local_axial_stress",
        "solver_local_axial_stress",
        "local_axial_stress_abs_error",
        "puls_local_trans_stress",
        "solver_local_trans_stress",
        "local_trans_stress_abs_error",
        "puls_local_shear_stress",
        "solver_local_shear_stress",
        "local_shear_stress_abs_error",
        "solver_global_factor",
        "solver_global_uncoupled_factor",
        "solver_local_plate_factor",
        "solver_local_shear_factor",
        "puls_failure_plate_buckling_percent",
        "solver_failure_plate_buckling_percent",
        "failure_plate_buckling_percent_abs_error",
        "puls_failure_global_stiffener_buckling_percent",
        "solver_failure_global_stiffener_buckling_percent",
        "failure_global_stiffener_buckling_percent_abs_error",
        "puls_failure_torsional_stiffener_buckling_percent",
        "solver_failure_torsional_stiffener_buckling_percent",
        "failure_torsional_stiffener_buckling_percent_abs_error",
        "puls_failure_web_stiffener_buckling_percent",
        "solver_failure_web_stiffener_buckling_percent",
        "failure_web_stiffener_buckling_percent_abs_error",
    ]
    fieldnames = [
        "row_index",
        "source",
        "ship_line",
        "stiffener_type",
        "support",
        "pressure",
        "axial_stress",
        "transverse_stress_1",
        "transverse_stress_2",
        "shear_stress",
        "length",
        "spacing",
        "plate_thickness",
        "puls_buckling_uf",
        "solver_buckling_uf",
        "solver_elastic_buckling_uf",
        "buckling_abs_error",
        "puls_ultimate_uf",
        "solver_ultimate_uf",
        "ultimate_abs_error",
        "raw_first_yield_ultimate_uf",
        "ultimate_lifted_to_buckling_strength",
        "valid",
        "invalid_reason",
        "buckling_strength_control",
        "pressure_dominated_yield_limit",
        "ultimate_included_in_buckling_strength",
        "pressure_preload_yield_max",
        "final_yield_max",
        "collapse_state",
        "elastic_failure_family",
        "confidence",
        "confidence_reasons",
        "mode_convergence_max_relative_drift",
        "load_family",
    ] + intermediate_fieldnames
    rows: list[dict[str, Any]] = []
    for row_index, row in iter_ship_section_file_rows(paths, limit=limit):
        try:
            panel = row_to_s3_input(row)
            result = solve_s3_panel(panel, config)
        except (TypeError, ValueError) as exc:
            result = _invalid_result("ship-section-row-error", {"error": str(exc)})
        expected_buckling = _optional_float(row.get("Buckling Actual usage Factor inc NaN"))
        expected_ultimate = _optional_float(row.get("Ultimate Actual usage Factor inc NaN"))
        buckling_strength = result.diagnostics.get("buckling_strength", {}) if result.valid else {}
        buckling = result.diagnostics.get("buckling", {}) if result.valid else {}
        pressure_yield = (
            result.diagnostics.get("pressure_preload_yield_utilization", {})
            if result.valid
            else result.diagnostics.get("pressure_preload_response", {}).get("yield_utilization", {})
        )
        final_yield = result.diagnostics.get("final_yield_utilization", {}) if result.valid else {}
        solver_buckling = result.buckling_usage_factor
        solver_ultimate = result.ultimate_usage_factor
        intermediate_comparison = _s3_ship_section_intermediate_comparison(row, panel, result)
        rows.append(
            {
                "row_index": row_index,
                "source": row.get("_source_file", ""),
                "ship_line": row.get("_ship_line", ""),
                "stiffener_type": row.get("Stiffener type", ""),
                "support": row.get("In-plane support", ""),
                "pressure": row.get("Pressure (fixed)", ""),
                "axial_stress": row.get("Axial stress", ""),
                "transverse_stress_1": row.get("Trans. stress 1", ""),
                "transverse_stress_2": row.get("Trans. stress 2", ""),
                "shear_stress": row.get("Shear stress", ""),
                "length": row.get("Length of panel", ""),
                "spacing": row.get("Stiffener spacing", ""),
                "plate_thickness": row.get("Plate thick.", ""),
                "puls_buckling_uf": "" if expected_buckling is None else expected_buckling,
                "solver_buckling_uf": "" if solver_buckling is None else solver_buckling,
                "solver_elastic_buckling_uf": (
                    "" if result.elastic_buckling_usage_factor is None else result.elastic_buckling_usage_factor
                ),
                "buckling_abs_error": (
                    ""
                    if expected_buckling is None or solver_buckling is None
                    else abs(solver_buckling - expected_buckling)
                ),
                "puls_ultimate_uf": "" if expected_ultimate is None else expected_ultimate,
                "solver_ultimate_uf": "" if solver_ultimate is None else solver_ultimate,
                "ultimate_abs_error": (
                    ""
                    if expected_ultimate is None or solver_ultimate is None
                    else abs(solver_ultimate - expected_ultimate)
                ),
                "raw_first_yield_ultimate_uf": buckling_strength.get("raw_ultimate_usage_factor", ""),
                "ultimate_lifted_to_buckling_strength": buckling_strength.get(
                    "ultimate_lifted_to_buckling_strength",
                    "",
                ),
                "valid": result.valid,
                "invalid_reason": result.invalid_reason or "",
                "buckling_strength_control": buckling_strength.get("controlling_limit", ""),
                "pressure_dominated_yield_limit": buckling_strength.get("pressure_dominated_yield_limit", ""),
                "ultimate_included_in_buckling_strength": buckling_strength.get("ultimate_included", ""),
                "pressure_preload_yield_max": pressure_yield.get("max", ""),
                "final_yield_max": final_yield.get("max", ""),
                "collapse_state": result.diagnostics.get("collapse_state", "") if result.valid else "",
                "elastic_failure_family": buckling.get("critical_failure_family", ""),
                "confidence": result.confidence,
                "confidence_reasons": ";".join(result.confidence_reasons),
                "mode_convergence_max_relative_drift": result.diagnostics.get(
                    "mode_convergence",
                    {},
                ).get("max_relative_drift", ""),
                "load_family": result.diagnostics.get("validation_domain", {}).get("load_family", ""),
                **intermediate_comparison,
            }
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_u3_ship_section_comparison_csv(
    paths: Sequence[str | Path],
    output_path: str | Path,
    config: S3SolverConfig | None = None,
    limit: int | None = None,
) -> Path:
    """Write row-level U3/PULS comparison for ANYstructure section exports."""

    config = config or S3SolverConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "source",
        "ship_line",
        "support",
        "rotational_support",
        "rotational_support_2",
        "pressure",
        "length",
        "width",
        "plate_thickness",
        "axial_stress",
        "axial_stress_2",
        "transverse_stress_1",
        "transverse_stress_2",
        "shear_stress",
        "puls_buckling_uf",
        "solver_buckling_uf",
        "solver_elastic_buckling_uf",
        "buckling_abs_error",
        "puls_ultimate_uf",
        "solver_ultimate_uf",
        "ultimate_abs_error",
        "raw_first_yield_ultimate_uf",
        "ultimate_lifted_to_buckling_strength",
        "valid",
        "invalid_reason",
        "buckling_strength_control",
        "pressure_preload_yield_max",
        "final_yield_max",
        "collapse_state",
        "elastic_failure_family",
        "confidence",
        "confidence_reasons",
        "mode_convergence_max_relative_drift",
        "load_family",
    ]
    rows: list[dict[str, Any]] = []
    for row_index, row in iter_ship_section_u3_file_rows(paths, limit=limit):
        try:
            panel = row_to_u3_input(row)
            result = solve_u3_panel(panel, config)
        except (TypeError, ValueError) as exc:
            result = _invalid_u3_result("ship-section-row-error", {"error": str(exc)})
        expected_buckling = _optional_float(row.get("Buckling Actual usage Factor inc NaN"))
        expected_ultimate = _optional_float(row.get("Ultimate Actual usage Factor inc NaN"))
        buckling_strength = result.diagnostics.get("buckling_strength", {}) if result.valid else {}
        buckling = result.diagnostics.get("buckling", {}) if result.valid else {}
        pressure_yield = (
            result.diagnostics.get("pressure_preload_yield_utilization", {})
            if result.valid
            else result.diagnostics.get("pressure_preload_response", {}).get("yield_utilization", {})
        )
        final_yield = result.diagnostics.get("final_yield_utilization", {}) if result.valid else {}
        solver_buckling = result.buckling_usage_factor
        solver_ultimate = result.ultimate_usage_factor
        rows.append(
            {
                "row_index": row_index,
                "source": row.get("_source_file", ""),
                "ship_line": row.get("_ship_line", ""),
                "support": row.get("In-plane support", ""),
                "rotational_support": row.get("Rotational support", ""),
                "rotational_support_2": row.get("Rotational support 2", ""),
                "pressure": row.get("Pressure (fixed)", ""),
                "length": row.get("Plate length", ""),
                "width": row.get("Plate width", ""),
                "plate_thickness": row.get("Plate thick.", ""),
                "axial_stress": row.get("Axial stress", ""),
                "axial_stress_2": row.get("Axial stress 2", ""),
                "transverse_stress_1": row.get("Trans. stress 1", ""),
                "transverse_stress_2": row.get("Trans. stress 2", ""),
                "shear_stress": row.get("Shear stress", ""),
                "puls_buckling_uf": "" if expected_buckling is None else expected_buckling,
                "solver_buckling_uf": "" if solver_buckling is None else solver_buckling,
                "solver_elastic_buckling_uf": (
                    "" if result.elastic_buckling_usage_factor is None else result.elastic_buckling_usage_factor
                ),
                "buckling_abs_error": (
                    ""
                    if expected_buckling is None or solver_buckling is None
                    else abs(solver_buckling - expected_buckling)
                ),
                "puls_ultimate_uf": "" if expected_ultimate is None else expected_ultimate,
                "solver_ultimate_uf": "" if solver_ultimate is None else solver_ultimate,
                "ultimate_abs_error": (
                    ""
                    if expected_ultimate is None or solver_ultimate is None
                    else abs(solver_ultimate - expected_ultimate)
                ),
                "raw_first_yield_ultimate_uf": buckling_strength.get("raw_ultimate_usage_factor", ""),
                "ultimate_lifted_to_buckling_strength": buckling_strength.get(
                    "ultimate_lifted_to_buckling_strength",
                    "",
                ),
                "valid": result.valid,
                "invalid_reason": result.invalid_reason or "",
                "buckling_strength_control": buckling_strength.get("controlling_limit", ""),
                "pressure_preload_yield_max": pressure_yield.get("max", ""),
                "final_yield_max": final_yield.get("max", ""),
                "collapse_state": result.diagnostics.get("collapse_state", "") if result.valid else "",
                "elastic_failure_family": buckling.get("critical_failure_family", ""),
                "confidence": result.confidence,
                "confidence_reasons": ";".join(result.confidence_reasons),
                "mode_convergence_max_relative_drift": result.diagnostics.get(
                    "mode_convergence",
                    {},
                ).get("max_relative_drift", ""),
                "load_family": result.diagnostics.get("validation_domain", {}).get("load_family", ""),
            }
        )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_row_comparison_csv(
    rows: Iterable[tuple[int, Mapping[str, Any]]],
    output_path: str | Path,
    config: S3SolverConfig,
    input_mapper: Any = row_to_s3_input,
    solver: Any = solve_s3_panel,
    mode_convergence_sample_size: int | None = None,
) -> Path:
    """Write a compact generic row-level PULS/solver comparison CSV."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_index",
        "source",
        "ship_line",
        "panel_family",
        "support",
        "stiffener_type",
        "pressure",
        "puls_buckling_uf",
        "solver_buckling_uf",
        "solver_elastic_buckling_uf",
        "buckling_signed_error",
        "buckling_abs_error",
        "puls_ultimate_uf",
        "solver_ultimate_uf",
        "ultimate_signed_error",
        "ultimate_abs_error",
        "valid",
        "invalid_reason",
        "buckling_strength_control",
        "elastic_failure_family",
        "confidence",
        "confidence_reasons",
        "mode_convergence_max_relative_drift",
        "load_family",
        "puls_manual_reference",
        "residual_cluster",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for emitted_index, (row_index, row) in enumerate(rows, start=1):
            row_config = config
            if (
                mode_convergence_sample_size is not None
                and config.check_mode_convergence
                and emitted_index > mode_convergence_sample_size
            ):
                row_config = replace(config, check_mode_convergence=False)
            try:
                result = solver(input_mapper(row), row_config)
            except (TypeError, ValueError) as exc:
                result = _invalid_result("row-error", {"error": str(exc)})
            expected_buckling = _optional_float(row.get("Buckling Actual usage Factor inc NaN"))
            expected_ultimate = _optional_float(row.get("Ultimate Actual usage Factor inc NaN"))
            buckling = result.diagnostics.get("buckling", {}) if result.valid else {}
            buckling_strength = result.diagnostics.get("buckling_strength", {}) if result.valid else {}
            solver_buckling = result.buckling_usage_factor
            solver_ultimate = result.ultimate_usage_factor
            buckling_signed_error = (
                None
                if expected_buckling is None or solver_buckling is None
                else solver_buckling - expected_buckling
            )
            ultimate_signed_error = (
                None
                if expected_ultimate is None or solver_ultimate is None
                else solver_ultimate - expected_ultimate
            )
            failure_family = buckling.get("critical_failure_family", "")
            pressure_key = "nonzero" if (_optional_float(row.get("Pressure (fixed)")) or 0.0) > 0.0 else "zero"
            load_family = result.diagnostics.get("validation_domain", {}).get("load_family", "")
            manual_key = _puls_manual_limit_key(result)
            writer.writerow(
                {
                "row_index": row_index,
                "source": row.get("_source_file", ""),
                "ship_line": row.get("_ship_line", ""),
                "panel_family": row.get("Panel family", "S3"),
                "support": row.get("In-plane support", ""),
                "stiffener_type": row.get("Stiffener type", row.get("Panel family", "")),
                "pressure": row.get("Pressure (fixed)", ""),
                "puls_buckling_uf": "" if expected_buckling is None else expected_buckling,
                "solver_buckling_uf": "" if solver_buckling is None else solver_buckling,
                "solver_elastic_buckling_uf": (
                    "" if result.elastic_buckling_usage_factor is None else result.elastic_buckling_usage_factor
                ),
                "buckling_signed_error": "" if buckling_signed_error is None else buckling_signed_error,
                "buckling_abs_error": (
                    ""
                    if buckling_signed_error is None
                    else abs(buckling_signed_error)
                ),
                "puls_ultimate_uf": "" if expected_ultimate is None else expected_ultimate,
                "solver_ultimate_uf": "" if solver_ultimate is None else solver_ultimate,
                "ultimate_signed_error": "" if ultimate_signed_error is None else ultimate_signed_error,
                "ultimate_abs_error": (
                    ""
                    if ultimate_signed_error is None
                    else abs(ultimate_signed_error)
                ),
                "valid": result.valid,
                "invalid_reason": result.invalid_reason or "",
                "buckling_strength_control": buckling_strength.get("controlling_limit", ""),
                "elastic_failure_family": failure_family,
                "confidence": result.confidence,
                "confidence_reasons": ";".join(result.confidence_reasons),
                "mode_convergence_max_relative_drift": result.diagnostics.get(
                    "mode_convergence",
                    {},
                ).get("max_relative_drift", ""),
                "load_family": load_family,
                "puls_manual_reference": manual_key,
                "residual_cluster": "|".join(
                    (
                        str(failure_family),
                        str(row.get("Stiffener type", row.get("Panel family", "")) or "missing"),
                        str(row.get("In-plane support", "") or "missing"),
                        pressure_key,
                        str(load_family),
                    )
                ),
                }
            )
    return output


def _load_baseline(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    baseline_path = Path(path)
    if not baseline_path.exists():
        return {}
    with baseline_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _baseline_from_reports(reports: Mapping[str, BenchmarkReport]) -> dict[str, Any]:
    return {
        name: {
            "buckling": report.buckling,
            "ultimate": report.ultimate,
            "elastic_buckling": report.elastic_buckling,
            "validity_confusion": report.validity_confusion,
            "uf_order_violations": report.uf_order_violations,
            "confidence_counts": report.confidence_counts,
            "mode_convergence_summary": report.mode_convergence_summary,
            "sample_manifest": report.sample_manifest,
            "conservative_error_summary": report.conservative_error_summary,
            "nonconservative_error_summary": report.nonconservative_error_summary,
            "residual_clusters": report.residual_clusters,
        }
        for name, report in reports.items()
    }


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output


def _csr_label(value: Any) -> int | None:
    parsed = _optional_float(value)
    if parsed is not None:
        return 1 if parsed == 1.0 else 0
    text = str(value or "").strip().lower()
    if text == "ok":
        return 1
    if text == "not ok":
        return 0
    return None


def _csr_update_confusion(confusion: dict[str, int], predicted: int | None, target: int | None) -> None:
    if target is None:
        confusion["unknown_target"] += 1
    elif predicted is None:
        confusion["unknown_predicted"] += 1
    elif predicted == 1 and target == 1:
        confusion["true_positive"] += 1
    elif predicted == 1 and target == 0:
        confusion["false_positive"] += 1
    elif predicted == 0 and target == 1:
        confusion["false_negative"] += 1
    else:
        confusion["true_negative"] += 1


def _csr_confusion_summary(confusion: Mapping[str, int]) -> dict[str, Any]:
    tp = int(confusion.get("true_positive", 0))
    tn = int(confusion.get("true_negative", 0))
    fp = int(confusion.get("false_positive", 0))
    fn = int(confusion.get("false_negative", 0))
    n = tp + tn + fp + fn
    return {
        **{key: int(value) for key, value in confusion.items()},
        "rows_with_known_labels": n,
        "accuracy": (tp + tn) / n if n else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
    }


def _u3_training_row_to_input(row: Mapping[str, Any]) -> U3PanelInput:
    return U3PanelInput(
        length=_float_from_row(row, "Length of panel"),
        width=_float_from_row(row, "Stiffener spacing"),
        plate_thickness=_float_from_row(row, "Plate thick."),
        yield_stress_plate=_float_from_row(row, "Yield stress plate"),
        axial_stress_1=_float_from_row(row, "Axial stress"),
        axial_stress_2=_float_from_row(row, "Axial stress"),
        transverse_stress_1=_float_from_row(row, "Trans. stress 1"),
        transverse_stress_2=_float_from_row(row, "Trans. stress 2"),
        shear_stress=_float_from_row(row, "Shear stress"),
        pressure=_float_from_row(row, "Pressure (fixed)"),
        in_plane_support=str(row.get("In-plane support", "")).strip(),
    )


def _csr_mismatch_example(
    row: Mapping[str, Any],
    component: str,
    predicted: int | None,
    target: int | None,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    values = diagnostics.get("values", {})
    return {
        "row_index": row.get("", ""),
        "component": component,
        "stiffener_type": row.get("Stiffener type", "U3"),
        "target": target,
        "predicted": predicted,
        "length": row.get("Length of panel", ""),
        "width_or_spacing": row.get("Stiffener spacing", ""),
        "plate_thickness": row.get("Plate thick.", ""),
        "yield_stress_plate": row.get("Yield stress plate", ""),
        "stiffener_height": row.get("Stiff. Height", ""),
        "web_thickness": row.get("Web thick.", ""),
        "flange_width": row.get("Flange width", ""),
        "flange_thickness": row.get("Flange thick.", ""),
        "values": values,
    }


def compare_csr_training_sets(
    sp_csv: str | Path = DEFAULT_CSR_SP_TRAINING_CSV,
    up_csv: str | Path = DEFAULT_CSR_UP_TRAINING_CSV,
    output_dir: str | Path | None = "reports",
    max_examples_per_component: int = 25,
) -> dict[str, Any]:
    """Compare equation-based CSR diagnostics with recorded PULS CSR training labels."""

    components = {
        "plate": (0, "CSR plate cl"),
        "web": (1, "CSR web cl"),
        "web_flange": (2, "CSR web flange cl"),
        "flange": (3, "CSR flange cl"),
    }
    summary: dict[str, Any] = {
        "sp_csv": str(sp_csv),
        "up_csv": str(up_csv),
        "basis": {
            "source": CSR_RULE_REFERENCE["source"],
            "note": "training labels are recorded PULS CSR-Tank requirement outputs; equations are not ML fitted",
        },
        "components": {},
        "mismatch_examples": [],
    }
    raw_confusions = {
        name: {
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
            "unknown_target": 0,
            "unknown_predicted": 0,
        }
        for name in ("sp_plate", "sp_web", "sp_web_flange", "sp_flange", "up_plate")
    }
    examples_seen: dict[str, int] = {name: 0 for name in raw_confusions}

    with Path(sp_csv).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            panel = row_to_s3_input(row)
            diagnostics = calculate_csr_requirement(panel)
            vector = diagnostics["csr_vector"]
            for component, (index, target_column) in components.items():
                key = f"sp_{component}"
                value = vector[index]
                predicted = None if value == float("inf") else int(value)
                target = _csr_label(row.get(target_column))
                _csr_update_confusion(raw_confusions[key], predicted, target)
                if (
                    predicted is not None
                    and target is not None
                    and predicted != target
                    and examples_seen[key] < max_examples_per_component
                ):
                    summary["mismatch_examples"].append(
                        _csr_mismatch_example(row, key, predicted, target, diagnostics)
                    )
                    examples_seen[key] += 1

    with Path(up_csv).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            panel = _u3_training_row_to_input(row)
            diagnostics = calculate_csr_requirement(panel)
            predicted = int(diagnostics["csr_vector"][0])
            target = _csr_label(row.get("CSR-Tank req cl"))
            _csr_update_confusion(raw_confusions["up_plate"], predicted, target)
            if (
                predicted is not None
                and target is not None
                and predicted != target
                and examples_seen["up_plate"] < max_examples_per_component
            ):
                summary["mismatch_examples"].append(
                    _csr_mismatch_example(row, "up_plate", predicted, target, diagnostics)
                )
                examples_seen["up_plate"] += 1

    summary["components"] = {
        name: _csr_confusion_summary(confusion)
        for name, confusion in raw_confusions.items()
    }
    known_total = sum(component["rows_with_known_labels"] for component in summary["components"].values())
    error_total = sum(
        component["false_positive"] + component["false_negative"]
        for component in summary["components"].values()
    )
    summary["overall"] = {
        "known_component_labels": known_total,
        "component_errors": error_total,
        "component_accuracy": 1.0 - error_total / known_total if known_total else None,
    }

    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / "csr_training_comparison_summary.json", summary)
        examples_path = output / "csr_training_mismatch_examples.csv"
        fieldnames = [
            "row_index",
            "component",
            "stiffener_type",
            "target",
            "predicted",
            "length",
            "width_or_spacing",
            "plate_thickness",
            "yield_stress_plate",
            "stiffener_height",
            "web_thickness",
            "flange_width",
            "flange_thickness",
            "values",
        ]
        with examples_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for example in summary["mismatch_examples"]:
                writer.writerow({**example, "values": json.dumps(example["values"], sort_keys=True)})

    return summary


def run_verification_suite(
    output_dir: str | Path = "reports",
    baseline_path: str | Path | None = DEFAULT_RELIABILITY_BASELINE,
    update_baseline: bool = False,
    check_mode_convergence: bool = True,
    csv_path: str | Path = "PULSforChatGPT.csv",
    ship_section_paths: Sequence[str | Path] = DEFAULT_SHIP_SECTION_INPUTS,
    csv_sample_method: str = CSV_SAMPLE_METHOD_STRATIFIED,
    csv_sample_size: int = DEFAULT_CSV_SAMPLE_SIZE,
    csv_sample_seed: int = DEFAULT_CSV_SAMPLE_SEED,
    csv_full: bool = False,
    csv_mode_convergence_sample_size: int = DEFAULT_CSV_MODE_CONVERGENCE_SAMPLE_SIZE,
    csv_target_valid_only: bool = False,
    csv_target_uf_min: float | None = None,
    csv_target_uf_max: float | None = None,
    csv_outlier_sample_size: int = 200,
) -> dict[str, Any]:
    """Run the combined S3/U3 verification suite and write reports."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    baseline = _load_baseline(baseline_path)
    config = S3SolverConfig(check_mode_convergence=check_mode_convergence)
    reports: dict[str, BenchmarkReport] = {}

    reports["s3_ship_sections"] = benchmark_ship_sections(ship_section_paths, config=config)
    s3_ship_report_path = output / "verify_all_s3_ship_sections.csv"
    write_ship_section_comparison_csv(
        ship_section_paths,
        s3_ship_report_path,
        config=config,
    )

    reports["u3_ship_sections"] = benchmark_u3_ship_sections(ship_section_paths, config=config)
    write_u3_ship_section_comparison_csv(
        ship_section_paths,
        output / "verify_all_u3_ship_sections.csv",
        config=config,
    )

    selected_csv_method = CSV_SAMPLE_METHOD_FULL if csv_full else csv_sample_method
    csv_selection = select_csv_sample_indices(
        csv_path,
        sample_method=selected_csv_method,
        sample_size=None if csv_full else csv_sample_size,
        seed=csv_sample_seed,
        target_valid_only=csv_target_valid_only,
        target_uf_min=csv_target_uf_min,
        target_uf_max=csv_target_uf_max,
    )
    csv_rows = (
        iter_csv_rows(csv_path)
        if csv_selection.row_indices is None
        else iter_csv_rows(csv_path, sample_indices=csv_selection.row_indices)
    )
    reports["s3_csv_stratified"] = _benchmark_rows(
        csv_rows,
        config,
        title="Reduced S3 stratified CSV benchmark report",
        sample_manifest=csv_selection.manifest,
        mode_convergence_sample_size=csv_mode_convergence_sample_size,
    )
    write_row_comparison_csv(
        iter_csv_rows(csv_path)
        if csv_selection.row_indices is None
        else iter_csv_rows(csv_path, sample_indices=csv_selection.row_indices),
        output / "verify_all_s3_csv_stratified.csv",
        config=config,
        mode_convergence_sample_size=csv_mode_convergence_sample_size,
    )

    csv_filter_active = _csv_sampling_filter_active(
        csv_target_valid_only,
        csv_target_uf_min,
        csv_target_uf_max,
    )
    if csv_filter_active and csv_outlier_sample_size > 0 and not csv_full:
        outlier_selection = select_csv_sample_indices(
            csv_path,
            sample_method=CSV_SAMPLE_METHOD_STRATIFIED,
            sample_size=csv_outlier_sample_size,
            seed=csv_sample_seed,
            target_valid_only=csv_target_valid_only,
            target_uf_min=csv_target_uf_min,
            target_uf_max=csv_target_uf_max,
            select_filtered_out=True,
        )
        reports["s3_csv_outliers"] = _benchmark_rows(
            iter_csv_rows(csv_path, sample_indices=outlier_selection.row_indices),
            config,
            title="Reduced S3 stratified CSV outlier monitor report",
            sample_manifest=outlier_selection.manifest,
            mode_convergence_sample_size=min(
                csv_mode_convergence_sample_size,
                csv_outlier_sample_size,
            ),
        )
        write_row_comparison_csv(
            iter_csv_rows(csv_path, sample_indices=outlier_selection.row_indices),
            output / "verify_all_s3_csv_outliers.csv",
            config=config,
            mode_convergence_sample_size=min(
                csv_mode_convergence_sample_size,
                csv_outlier_sample_size,
            ),
        )

    fixture_selection = select_csv_sample_indices(
        csv_path,
        sample_method=CSV_SAMPLE_METHOD_STRATIFIED,
        sample_size=DEFAULT_CSV_FIXTURE_SAMPLE_SIZE,
        seed=csv_sample_seed,
    )
    reports["s3_csv_fixture"] = _benchmark_rows(
        iter_csv_rows(csv_path, sample_indices=fixture_selection.row_indices),
        config,
        title="Reduced S3 stratified fixture CSV benchmark report",
        sample_manifest=fixture_selection.manifest,
        mode_convergence_sample_size=min(
            csv_mode_convergence_sample_size,
            DEFAULT_CSV_FIXTURE_SAMPLE_SIZE,
        ),
    )
    write_row_comparison_csv(
        iter_csv_rows(csv_path, sample_indices=fixture_selection.row_indices),
        output / "verify_all_s3_csv_fixture.csv",
        config=config,
        mode_convergence_sample_size=min(
            csv_mode_convergence_sample_size,
            DEFAULT_CSV_FIXTURE_SAMPLE_SIZE,
        ),
    )

    baseline_available = bool(baseline)
    acceptance_thresholds: dict[str, dict[str, Any]] = {
        "u3_ship_sections": {
            "max_buckling_mae": 0.03,
            "max_ultimate_mae": 0.04,
            "max_validity_mismatches": 0,
            "max_uf_order_violations": 0,
        },
        "s3_ship_sections": {
            "max_uf_order_violations": 0,
        },
        "s3_csv_stratified": {
            "max_uf_order_violations": 0,
        },
        "s3_csv_fixture": {
            "max_uf_order_violations": 0,
        },
    }
    if "s3_csv_outliers" in reports:
        acceptance_thresholds["s3_csv_outliers"] = {
            "max_uf_order_violations": 0,
        }
    if baseline_available:
        if "u3_ship_sections" in baseline:
            acceptance_thresholds["u3_ship_sections"].update(
                {
                    "baseline_buckling_mae_delta": 1.0e-9,
                    "baseline_ultimate_mae_delta": 1.0e-9,
                }
            )
        if "s3_ship_sections" in baseline:
            acceptance_thresholds["s3_ship_sections"].update(
                {
                    "baseline_buckling_mae_delta": 0.005,
                    "baseline_ultimate_mae_delta": 0.005,
                    "baseline_validity_mismatch_delta": 0,
                }
            )
        if "s3_csv_stratified" in baseline:
            acceptance_thresholds["s3_csv_stratified"].update(
                {
                    "baseline_buckling_relative_mae_factor": 1.05,
                }
            )

    for name, report in reports.items():
        apply_acceptance_gates(
            report,
            name,
            acceptance_thresholds.get(name, {}),
            baseline,
        )

    summary = {
        "passed": all(report.passed for report in reports.values()),
        "baseline_path": str(baseline_path) if baseline_path is not None else None,
        "baseline_available": baseline_available,
        "mode_convergence_enabled": check_mode_convergence,
        "csv_sample_method": selected_csv_method,
        "csv_sample_size": None if csv_full else csv_sample_size,
        "csv_sample_seed": csv_sample_seed,
        "csv_mode_convergence_sample_size": csv_mode_convergence_sample_size,
        "csv_target_valid_only": csv_target_valid_only,
        "csv_target_buckling_uf_min": csv_target_uf_min,
        "csv_target_buckling_uf_max": csv_target_uf_max,
        "csv_outlier_sample_size": csv_outlier_sample_size,
        "s3_ship_intermediate_comparison": summarize_s3_ship_intermediate_report(s3_ship_report_path),
        "reports": {name: report.to_dict() for name, report in reports.items()},
    }
    _write_json(output / "verify_all_summary.json", summary)
    if update_baseline and baseline_path is not None:
        _write_json(baseline_path, _baseline_from_reports(reports))
    return summary


def run_full_csv_verification(
    csv_path: str | Path = "PULSforChatGPT.csv",
    output_dir: str | Path = "reports",
    baseline_path: str | Path | None = Path("reports/puls_full_csv_baseline.json"),
    update_baseline: bool = True,
) -> dict[str, Any]:
    """Run the full S3 CSV benchmark with mode convergence disabled."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = S3SolverConfig(check_mode_convergence=False)
    selection = select_csv_sample_indices(csv_path, sample_method=CSV_SAMPLE_METHOD_FULL)
    report = _benchmark_rows(
        iter_csv_rows(csv_path),
        config,
        title="Reduced S3 full CSV benchmark report",
        sample_manifest=selection.manifest,
        mode_convergence_sample_size=0,
    )
    apply_acceptance_gates(report, "s3_csv_full", {"max_uf_order_violations": 0}, {})
    write_row_comparison_csv(
        iter_csv_rows(csv_path),
        output / "verify_csv_full_rows.csv",
        config=config,
        mode_convergence_sample_size=0,
    )
    summary = {
        "passed": report.passed,
        "csv_path": str(csv_path),
        "mode_convergence_enabled": False,
        "reports": {"s3_csv_full": report.to_dict()},
    }
    _write_json(output / "verify_csv_full_summary.json", summary)
    if update_baseline and baseline_path is not None:
        _write_json(baseline_path, {"s3_csv_full": report.to_dict()})
    return summary


def run_speed_benchmark(
    csv_path: str | Path = "PULSforChatGPT.csv",
    iterations: int = 100,
) -> dict[str, Any]:
    """Return small local timing diagnostics for the runtime solver."""

    clock = __import__("time").perf_counter
    s3_panel = None
    for _, row in iter_csv_rows(csv_path, limit=500):
        try:
            candidate = row_to_s3_input(row)
            if solve_s3_panel(candidate, S3SolverConfig(check_mode_convergence=False)).valid:
                s3_panel = candidate
                break
        except Exception:
            continue
    if s3_panel is None:
        raise RuntimeError("no valid S3 panel found for speed benchmark")

    u3_panel = U3PanelInput(
        length=3000.0,
        width=750.0,
        plate_thickness=12.0,
        yield_stress_plate=355.0,
        axial_stress_1=100.0,
        axial_stress_2=100.0,
        transverse_stress_1=80.0,
        transverse_stress_2=80.0,
        shear_stress=20.0,
        pressure=0.1,
        in_plane_support="Integrated",
        rotational_support_1="SS",
        rotational_support_2="SS",
    )
    configs = {
        "full": S3SolverConfig(check_mode_convergence=False),
        "minimal": S3SolverConfig(
            check_mode_convergence=False,
            include_solver_diagnostics=False,
        ),
    }

    class _SpeedAnyStructurePart:
        def __init__(self, axial_stress: float = 60.0, method: str = "buckling") -> None:
            self.span = s3_panel.length / 1000.0
            self.spacing = s3_panel.stiffener_spacing
            self.t = s3_panel.plate_thickness
            self.hw = s3_panel.stiffener_height
            self.tw = s3_panel.web_thickness
            self.b = s3_panel.flange_width
            self.tf = s3_panel.flange_thickness
            self.mat_yield = s3_panel.yield_stress_plate * 1.0e6
            self.sigma_x1 = axial_stress
            self.sigma_x2 = axial_stress
            self.sigma_y1 = s3_panel.transverse_stress_1
            self.sigma_y2 = s3_panel.transverse_stress_2
            self.tau_xy = s3_panel.shear_stress
            self.mat_factor = 1.15
            self.corrosion_addition_mm = 0.0
            self._method = method

        def get_puls_sp_or_up(self) -> str:
            return "SP"

        def get_puls_boundary(self) -> str:
            return {
                "Integrated": "Int",
                "Girder - long": "GL",
                "Girder - trans": "GT",
            }.get(s3_panel.in_plane_support, "Int")

        def get_stiffener_type(self) -> str:
            return {
                "T-bar": "T",
                "Angle": "L",
                "Flatbar": "FB",
            }.get(s3_panel.stiffener_type, s3_panel.stiffener_type)

        def get_puls_stf_end(self) -> str:
            return "C" if s3_panel.stiffener_boundary == "Cont" else "S"

        def get_puls_method(self) -> str:
            return self._method

    class _SpeedAnyStructure:
        def __init__(self, axial_stress: float = 60.0, method: str = "buckling") -> None:
            self.Plate = _SpeedAnyStructurePart(axial_stress=axial_stress, method=method)
            self.Stiffener = _SpeedAnyStructurePart(axial_stress=axial_stress, method=method)
            self.E = s3_panel.elastic_modulus * 1.0e6
            self.v = s3_panel.poisson_ratio

    anystructure_full = _SpeedAnyStructure(axial_stress=max(s3_panel.axial_stress, 60.0), method="ultimate")
    anystructure_reject = _SpeedAnyStructure(axial_stress=900.0, method="buckling")
    anystructure_batch = [
        (anystructure_reject, None, 0.0),
        (anystructure_reject, None, 0.0),
        (anystructure_full, None, 0.0),
        (anystructure_full, None, 0.0),
    ] * 5

    def measure(label: str, fn: Any) -> dict[str, float | int | str]:
        for _ in range(5):
            fn()
        samples = []
        for _ in range(iterations):
            start = clock()
            fn()
            samples.append(clock() - start)
        return {
            "label": label,
            "iterations": iterations,
            "mean_seconds": statistics.mean(samples),
            "median_seconds": statistics.median(samples),
            "p95_seconds": _percentile(samples, 0.95),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
        }

    return {
        "s3_full": measure("s3_full", lambda: solve_s3_panel(s3_panel, configs["full"])),
        "s3_minimal": measure("s3_minimal", lambda: solve_s3_panel(s3_panel, configs["minimal"])),
        "u3_full": measure("u3_full", lambda: solve_u3_panel(u3_panel, configs["full"])),
        "u3_minimal": measure("u3_minimal", lambda: solve_u3_panel(u3_panel, configs["minimal"])),
        "anystructure_s3_full_vector": measure(
            "anystructure_s3_full_vector",
            lambda: predict_anystructure_uf_with_acceptance(
                anystructure_full,
                selected_method="ultimate",
            ),
        ),
        "anystructure_s3_buckling_early_reject": measure(
            "anystructure_s3_buckling_early_reject",
            lambda: predict_anystructure_uf_with_acceptance(
                anystructure_reject,
                selected_method="buckling",
            ),
        ),
        "anystructure_s3_batch_mixed_20": measure(
            "anystructure_s3_batch_mixed_20",
            lambda: predict_anystructure_uf_batch(anystructure_batch, cache={}),
        ),
    }


def _benchmark_command(args: argparse.Namespace) -> int:
    config = S3SolverConfig(
        max_load_factor=args.max_load_factor,
        use_effective_stiffener_width=args.effective_stiffener_width,
    )
    report = benchmark_csv(
        args.csv,
        config=config,
        limit=args.limit,
        fixture=args.fixture,
        sample_method=args.csv_sample_method,
        sample_size=args.csv_sample_size,
        sample_seed=args.csv_sample_seed,
        target_valid_only=args.csv_target_valid_only,
        target_uf_min=args.csv_target_uf_min,
        target_uf_max=args.csv_target_uf_max,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.format_text())
    return 0


def _ship_section_command(args: argparse.Namespace) -> int:
    config = S3SolverConfig(
        max_load_factor=args.max_load_factor,
        use_effective_stiffener_width=args.effective_stiffener_width,
    )
    report = benchmark_ship_sections(args.input, config=config, limit=args.limit)
    if args.output_csv:
        write_ship_section_comparison_csv(
            args.input,
            args.output_csv,
            config=config,
            limit=args.limit,
        )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.format_text())
    return 0


def _u3_ship_section_command(args: argparse.Namespace) -> int:
    config = S3SolverConfig(
        max_load_factor=args.max_load_factor,
    )
    report = benchmark_u3_ship_sections(args.input, config=config, limit=args.limit)
    if args.output_csv:
        write_u3_ship_section_comparison_csv(
            args.input,
            args.output_csv,
            config=config,
            limit=args.limit,
        )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.format_text())
    return 0


def _verify_all_command(args: argparse.Namespace) -> int:
    summary = run_verification_suite(
        output_dir=args.output_dir,
        baseline_path=args.baseline,
        update_baseline=args.update_baseline,
        check_mode_convergence=not args.no_mode_convergence,
        csv_path=args.csv,
        ship_section_paths=args.input,
        csv_sample_method=args.csv_sample_method,
        csv_sample_size=args.csv_sample_size,
        csv_sample_seed=args.csv_sample_seed,
        csv_full=args.csv_full,
        csv_mode_convergence_sample_size=args.csv_mode_convergence_sample_size,
        csv_target_valid_only=args.csv_target_valid_only,
        csv_target_uf_min=args.csv_target_uf_min,
        csv_target_uf_max=args.csv_target_uf_max,
        csv_outlier_sample_size=args.csv_outlier_sample_size,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Verification passed: {summary['passed']}")
        print(f"Summary JSON: {Path(args.output_dir) / 'verify_all_summary.json'}")
        for name, report in summary["reports"].items():
            print(
                f"{name}: passed={report['passed']} "
                f"buckling_mae={report['buckling']['mean_absolute_error']} "
                f"ultimate_mae={report['ultimate']['mean_absolute_error']} "
                f"confidence={report['confidence_counts']}"
            )
    return 0 if summary["passed"] else 1


def _verify_csv_full_command(args: argparse.Namespace) -> int:
    summary = run_full_csv_verification(
        csv_path=args.csv,
        output_dir=args.output_dir,
        baseline_path=args.baseline,
        update_baseline=not args.no_update_baseline,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        report = summary["reports"]["s3_csv_full"]
        print(f"Full CSV verification passed: {summary['passed']}")
        print(f"Summary JSON: {Path(args.output_dir) / 'verify_csv_full_summary.json'}")
        print(
            "s3_csv_full: "
            f"buckling_mae={report['buckling']['mean_absolute_error']} "
            f"ultimate_mae={report['ultimate']['mean_absolute_error']} "
            f"rows={report['rows']}"
        )
    return 0 if summary["passed"] else 1


def _speed_benchmark_command(args: argparse.Namespace) -> int:
    result = run_speed_benchmark(args.csv, iterations=args.iterations)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _verify_csr_training_command(args: argparse.Namespace) -> int:
    summary = compare_csr_training_sets(
        sp_csv=args.sp_csv,
        up_csv=args.up_csv,
        output_dir=args.output_dir,
        max_examples_per_component=args.max_examples,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("CSR training comparison")
        print(f"Summary JSON: {Path(args.output_dir) / 'csr_training_comparison_summary.json'}")
        print(f"Mismatch examples CSV: {Path(args.output_dir) / 'csr_training_mismatch_examples.csv'}")
        print(
            "Overall: "
            f"accuracy={summary['overall']['component_accuracy']:.6f} "
            f"errors={summary['overall']['component_errors']} "
            f"labels={summary['overall']['known_component_labels']}"
        )
        for name, component in summary["components"].items():
            print(
                f"{name}: accuracy={component['accuracy']:.6f} "
                f"fp={component['false_positive']} fn={component['false_negative']} "
                f"known={component['rows_with_known_labels']}"
            )
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduced semi-analytical PULS S3/U3 calculations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="Compare S3 results with a PULS CSV export")
    benchmark.add_argument("--csv", default="PULSforChatGPT.csv", help="PULS CSV path")
    benchmark.add_argument("--limit", type=int, default=None, help="Read only the first N selected CSV rows")
    benchmark.add_argument("--fixture", action="store_true", help="Use the deterministic stratified fixture sample")
    benchmark.add_argument(
        "--csv-sample-method",
        choices=(CSV_SAMPLE_METHOD_FIRST_N, CSV_SAMPLE_METHOD_STRATIFIED, CSV_SAMPLE_METHOD_FULL),
        default=None,
        help="CSV row selection method",
    )
    benchmark.add_argument(
        "--csv-sample-size",
        type=int,
        default=DEFAULT_CSV_SAMPLE_SIZE,
        help="Number of rows for stratified CSV sampling",
    )
    benchmark.add_argument(
        "--csv-sample-seed",
        type=int,
        default=DEFAULT_CSV_SAMPLE_SEED,
        help="Seed for deterministic shuffled CSV sampling",
    )
    benchmark.add_argument(
        "--csv-target-valid-only",
        action="store_true",
        help="Sample only CSV rows with numeric PULS buckling and ultimate UF targets",
    )
    benchmark.add_argument(
        "--csv-target-uf-min",
        type=float,
        default=None,
        help="Minimum target PULS buckling UF for sampled CSV rows",
    )
    benchmark.add_argument(
        "--csv-target-uf-max",
        type=float,
        default=None,
        help="Maximum target PULS buckling UF for sampled CSV rows",
    )
    benchmark.add_argument("--json", action="store_true", help="Print JSON instead of the text report")
    benchmark.add_argument(
        "--effective-stiffener-width",
        action="store_true",
        help="Apply length-based effective attached-plate width in stiffener yield and column checks",
    )
    benchmark.add_argument(
        "--max-load-factor",
        type=float,
        default=S3SolverConfig.max_load_factor,
        help="Maximum continuation factor for the benchmark solve",
    )
    benchmark.set_defaults(run=_benchmark_command)

    ship_section = subparsers.add_parser(
        "ship-section",
        help="Compare S3 results with an ANYstructure ship-section dictionary export",
    )
    ship_section.add_argument(
        "--input",
        nargs="+",
        default=[r"C:\Github\ANYstructure\anystruct\ship_section_example.txt"],
        help="ANYstructure ship-section export path(s)",
    )
    ship_section.add_argument("--limit", type=int, default=None, help="Read only the first N S3 rows")
    ship_section.add_argument("--json", action="store_true", help="Print JSON instead of the text report")
    ship_section.add_argument("--output-csv", default=None, help="Write row-level comparison CSV")
    ship_section.add_argument(
        "--effective-stiffener-width",
        action="store_true",
        help="Apply length-based effective attached-plate width in stiffener yield and column checks",
    )
    ship_section.add_argument(
        "--max-load-factor",
        type=float,
        default=S3SolverConfig.max_load_factor,
        help="Maximum continuation factor for the benchmark solve",
    )
    ship_section.set_defaults(run=_ship_section_command)

    u3_ship_section = subparsers.add_parser(
        "u3-ship-section",
        help="Compare U3 results with an ANYstructure ship-section dictionary export",
    )
    u3_ship_section.add_argument(
        "--input",
        nargs="+",
        default=[r"C:\Github\ANYstructure\anystruct\ship_section_example.txt"],
        help="ANYstructure ship-section export path(s)",
    )
    u3_ship_section.add_argument("--limit", type=int, default=None, help="Read only the first N U3 rows")
    u3_ship_section.add_argument("--json", action="store_true", help="Print JSON instead of the text report")
    u3_ship_section.add_argument("--output-csv", default=None, help="Write row-level comparison CSV")
    u3_ship_section.add_argument(
        "--max-load-factor",
        type=float,
        default=S3SolverConfig.max_load_factor,
        help="Maximum continuation factor for the benchmark solve",
    )
    u3_ship_section.set_defaults(run=_u3_ship_section_command)

    verify_all = subparsers.add_parser(
        "verify-all",
        help="Run S3/U3 verification, write row CSVs, and apply reliability acceptance gates",
    )
    verify_all.add_argument("--csv", default="PULSforChatGPT.csv", help="PULS CSV path")
    verify_all.add_argument(
        "--input",
        nargs="+",
        default=list(DEFAULT_SHIP_SECTION_INPUTS),
        help="ANYstructure ship-section export path(s)",
    )
    verify_all.add_argument("--output-dir", default="reports", help="Directory for verification reports")
    verify_all.add_argument(
        "--baseline",
        default=str(DEFAULT_RELIABILITY_BASELINE),
        help="Baseline JSON used for regression gates",
    )
    verify_all.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current verification metrics to the baseline JSON path",
    )
    verify_all.add_argument(
        "--no-mode-convergence",
        action="store_true",
        help="Disable refined-basis mode convergence checks during verification",
    )
    verify_all.add_argument(
        "--csv-sample-method",
        choices=(CSV_SAMPLE_METHOD_STRATIFIED, CSV_SAMPLE_METHOD_FULL),
        default=CSV_SAMPLE_METHOD_STRATIFIED,
        help="CSV row selection method for the S3 CSV verification gate",
    )
    verify_all.add_argument(
        "--csv-sample-size",
        type=int,
        default=DEFAULT_CSV_SAMPLE_SIZE,
        help="Number of CSV rows for deterministic stratified sampling",
    )
    verify_all.add_argument(
        "--csv-sample-seed",
        type=int,
        default=DEFAULT_CSV_SAMPLE_SEED,
        help="Seed for deterministic shuffled CSV sampling",
    )
    verify_all.add_argument(
        "--csv-full",
        action="store_true",
        help="Use all CSV rows in verify-all instead of the stratified sample",
    )
    verify_all.add_argument(
        "--csv-mode-convergence-sample-size",
        type=int,
        default=DEFAULT_CSV_MODE_CONVERGENCE_SAMPLE_SIZE,
        help="Number of sampled CSV rows that run refined-basis mode convergence checks",
    )
    verify_all.add_argument(
        "--csv-target-valid-only",
        action="store_true",
        help="Sample only CSV rows with numeric PULS buckling and ultimate UF targets",
    )
    verify_all.add_argument(
        "--csv-target-uf-min",
        type=float,
        default=None,
        help="Minimum target PULS buckling UF for sampled CSV rows",
    )
    verify_all.add_argument(
        "--csv-target-uf-max",
        type=float,
        default=None,
        help="Maximum target PULS buckling UF for sampled CSV rows",
    )
    verify_all.add_argument(
        "--csv-outlier-sample-size",
        type=int,
        default=200,
        help="Additional stratified monitor sample from rows excluded by the CSV target UF filter",
    )
    verify_all.add_argument("--json", action="store_true", help="Print JSON instead of text summary")
    verify_all.set_defaults(run=_verify_all_command)

    verify_csv_full = subparsers.add_parser(
        "verify-csv-full",
        help="Run the full S3 CSV benchmark with mode convergence disabled",
    )
    verify_csv_full.add_argument("--csv", default="PULSforChatGPT.csv", help="PULS CSV path")
    verify_csv_full.add_argument("--output-dir", default="reports", help="Directory for verification reports")
    verify_csv_full.add_argument(
        "--baseline",
        default="reports/puls_full_csv_baseline.json",
        help="Full CSV baseline JSON path",
    )
    verify_csv_full.add_argument(
        "--no-update-baseline",
        action="store_true",
        help="Do not write the full CSV baseline JSON",
    )
    verify_csv_full.add_argument("--json", action="store_true", help="Print JSON instead of text summary")
    verify_csv_full.set_defaults(run=_verify_csv_full_command)

    speed = subparsers.add_parser(
        "speed-benchmark",
        help="Run a small local S3/U3 runtime speed benchmark",
    )
    speed.add_argument("--csv", default="PULSforChatGPT.csv", help="PULS CSV path for an S3 sample")
    speed.add_argument("--iterations", type=int, default=100, help="Timing iterations per case")
    speed.set_defaults(run=_speed_benchmark_command)

    csr_training = subparsers.add_parser(
        "verify-csr-training",
        help="Compare equation-based CSR checks with recorded PULS CSR training labels",
    )
    csr_training.add_argument(
        "--sp-csv",
        default=str(DEFAULT_CSR_SP_TRAINING_CSV),
        help="CSR-labeled stiffened-panel training CSV",
    )
    csr_training.add_argument(
        "--up-csv",
        default=str(DEFAULT_CSR_UP_TRAINING_CSV),
        help="CSR-labeled unstiffened-panel training CSV",
    )
    csr_training.add_argument("--output-dir", default="reports", help="Directory for CSR comparison reports")
    csr_training.add_argument(
        "--max-examples",
        type=int,
        default=25,
        help="Maximum mismatch examples to store per CSR component",
    )
    csr_training.add_argument("--json", action="store_true", help="Print JSON instead of the text summary")
    csr_training.set_defaults(run=_verify_csr_training_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
