"""Nonlinear material hardening curves.

Implements the DNV-RP-C208 (September 2019, amended October 2022) section
4.6.6 recommendation: the flow stress is a function of true plastic strain
built from a stepwise linear part with a yield plateau and a power-law part:

    Part 1:  sigma_prop  -> sigma_yield    over  0        .. eps_p_y1
    Part 2:  sigma_yield -> sigma_yield_2  over  eps_p_y1 .. eps_p_y2
    Part 3:  sigma = K * (eps_p + (sigma_yield_2 / K)**(1/n) - eps_p_y2)**n

All evaluations are vectorized over numpy arrays of equivalent plastic
strain, because the return mapping calls them for every integration point and
thickness layer at once.

The curve is expressed in true stress / true plastic strain exactly as
tabulated by the RP; the solver consumes it directly as the J2 flow stress.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np


_MPA = 1.0e6


@dataclass(frozen=True)
class FiberSectionPlasticityConfig:
    """Opt-in beam fiber plasticity configuration.

    The beam fiber model uses an equivalent rectangular fiber grid whose
    coordinates are scaled to match the supplied section ``A``, ``Iy`` and
    ``Iz``.  It is intended for monotonic capacity checks of beam/stiffener
    idealizations; shear and torsion remain elastic in v1.
    """

    num_y: int = 5
    num_z: int = 5
    material_curve: Optional["DNVC208MaterialCurve"] = None

    def __post_init__(self) -> None:
        if self.num_y <= 0 or self.num_z <= 0:
            raise ValueError("num_y and num_z must be positive")


@dataclass(frozen=True)
class DNVC208MaterialCurve:
    """DNV-RP-C208 stepwise-linear plus power-law flow curve."""

    sigma_prop: float
    sigma_yield: float
    sigma_yield_2: float
    eps_p_y1: float
    eps_p_y2: float
    K: float
    n: float

    def __post_init__(self) -> None:
        if self.sigma_prop <= 0.0:
            raise ValueError("sigma_prop must be positive")
        if self.sigma_yield < self.sigma_prop:
            raise ValueError("sigma_yield must be >= sigma_prop")
        if self.sigma_yield_2 < self.sigma_yield:
            raise ValueError("sigma_yield_2 must be >= sigma_yield")
        if not (0.0 < self.eps_p_y1 < self.eps_p_y2):
            raise ValueError("require 0 < eps_p_y1 < eps_p_y2")
        if self.K <= 0.0 or not (0.0 < self.n < 1.0):
            raise ValueError("require K > 0 and 0 < n < 1")

    @property
    def _power_offset(self) -> float:
        """Plastic-strain offset making Part 3 continuous at eps_p_y2."""
        return (self.sigma_yield_2 / self.K) ** (1.0 / self.n) - self.eps_p_y2

    def flow_stress(self, eps_p: np.ndarray) -> np.ndarray:
        """Flow (yield) stress as a function of equivalent plastic strain."""
        eps_p = np.maximum(np.asarray(eps_p, dtype=float), 0.0)
        slope_1 = (self.sigma_yield - self.sigma_prop) / self.eps_p_y1
        slope_2 = (self.sigma_yield_2 - self.sigma_yield) / (self.eps_p_y2 - self.eps_p_y1)
        part_1 = self.sigma_prop + slope_1 * eps_p
        part_2 = self.sigma_yield + slope_2 * (eps_p - self.eps_p_y1)
        part_3 = self.K * np.power(np.maximum(eps_p + self._power_offset, 1.0e-12), self.n)
        return np.where(
            eps_p <= self.eps_p_y1,
            part_1,
            np.where(eps_p <= self.eps_p_y2, part_2, part_3),
        )

    def hardening_modulus(self, eps_p: np.ndarray) -> np.ndarray:
        """d(flow stress)/d(equivalent plastic strain)."""
        eps_p = np.maximum(np.asarray(eps_p, dtype=float), 0.0)
        slope_1 = (self.sigma_yield - self.sigma_prop) / self.eps_p_y1
        slope_2 = (self.sigma_yield_2 - self.sigma_yield) / (self.eps_p_y2 - self.eps_p_y1)
        base = np.maximum(eps_p + self._power_offset, 1.0e-12)
        slope_3 = self.K * self.n * np.power(base, self.n - 1.0)
        return np.where(
            eps_p <= self.eps_p_y1,
            slope_1,
            np.where(eps_p <= self.eps_p_y2, slope_2, slope_3),
        )


def curve_from_properties(properties: dict) -> DNVC208MaterialCurve:
    """Build a curve from an RP-C208 table row (e.g. Table 4-2 .. 4-6)."""
    return DNVC208MaterialCurve(
        sigma_prop=float(properties["sigma_prop"]),
        sigma_yield=float(properties["sigma_yield"]),
        sigma_yield_2=float(properties["sigma_yield_2"]),
        eps_p_y1=float(properties.get("eps_p_y1", 0.004)),
        eps_p_y2=float(properties["eps_p_y2"]),
        K=float(properties["K"]),
        n=float(properties["n"]),
    )


_DNV_C208_LOW_FRACTILE_STEEL_MPA: Dict[str, tuple[tuple[float, float, Mapping[str, float]], ...]] = {
    "S235": (
        (0.0, 16.0, {"sigma_prop": 211.7, "sigma_yield": 236.2, "sigma_yield_2": 243.4, "eps_p_y2": 0.020, "K": 520.0, "n": 0.166}),
        (16.0, 40.0, {"sigma_prop": 202.7, "sigma_yield": 226.1, "sigma_yield_2": 233.2, "eps_p_y2": 0.020, "K": 520.0, "n": 0.166}),
        (40.0, 63.0, {"sigma_prop": 193.7, "sigma_yield": 216.1, "sigma_yield_2": 223.0, "eps_p_y2": 0.020, "K": 520.0, "n": 0.166}),
        (63.0, 100.0, {"sigma_prop": 193.7, "sigma_yield": 216.1, "sigma_yield_2": 223.0, "eps_p_y2": 0.020, "K": 520.0, "n": 0.166}),
    ),
    "S275": (
        (0.0, 16.0, {"sigma_prop": 247.8, "sigma_yield": 276.5, "sigma_yield_2": 282.8, "eps_p_y2": 0.017, "K": 620.0, "n": 0.166}),
        (16.0, 40.0, {"sigma_prop": 238.8, "sigma_yield": 266.4, "sigma_yield_2": 272.6, "eps_p_y2": 0.017, "K": 620.0, "n": 0.166}),
        (40.0, 63.0, {"sigma_prop": 229.8, "sigma_yield": 256.3, "sigma_yield_2": 262.4, "eps_p_y2": 0.017, "K": 620.0, "n": 0.166}),
    ),
    "S355": (
        (0.0, 16.0, {"sigma_prop": 320.0, "sigma_yield": 357.0, "sigma_yield_2": 363.3, "eps_p_y2": 0.015, "K": 740.0, "n": 0.166}),
        (16.0, 40.0, {"sigma_prop": 311.0, "sigma_yield": 346.9, "sigma_yield_2": 353.1, "eps_p_y2": 0.015, "K": 740.0, "n": 0.166}),
        (40.0, 63.0, {"sigma_prop": 301.9, "sigma_yield": 336.9, "sigma_yield_2": 342.9, "eps_p_y2": 0.015, "K": 725.0, "n": 0.166}),
        (63.0, 100.0, {"sigma_prop": 283.9, "sigma_yield": 316.7, "sigma_yield_2": 322.5, "eps_p_y2": 0.015, "K": 725.0, "n": 0.166}),
    ),
    "S420": (
        (0.0, 16.0, {"sigma_prop": 378.7, "sigma_yield": 422.5, "sigma_yield_2": 427.6, "eps_p_y2": 0.012, "K": 738.0, "n": 0.140}),
        (16.0, 40.0, {"sigma_prop": 360.6, "sigma_yield": 402.4, "sigma_yield_2": 407.3, "eps_p_y2": 0.012, "K": 703.0, "n": 0.140}),
        (40.0, 63.0, {"sigma_prop": 351.6, "sigma_yield": 392.3, "sigma_yield_2": 397.1, "eps_p_y2": 0.012, "K": 686.0, "n": 0.140}),
    ),
    "S460": (
        (0.0, 16.0, {"sigma_prop": 414.8, "sigma_yield": 462.8, "sigma_yield_2": 466.9, "eps_p_y2": 0.010, "K": 772.0, "n": 0.120}),
        (16.0, 40.0, {"sigma_prop": 396.7, "sigma_yield": 442.7, "sigma_yield_2": 446.6, "eps_p_y2": 0.010, "K": 745.0, "n": 0.120}),
        (40.0, 63.0, {"sigma_prop": 374.2, "sigma_yield": 417.5, "sigma_yield_2": 421.2, "eps_p_y2": 0.010, "K": 703.0, "n": 0.120}),
    ),
}


def dnv_c208_steel_curve(grade: str, thickness: float, fractile: str = "low") -> DNVC208MaterialCurve:
    """Return an RP-C208 steel curve for a grade and plate thickness.

    ``thickness`` is in metres, matching solver SI units.  The built-in table
    covers the RP-C208 low-fractile curves from section 4.6.6.  Mean curves are
    intentionally not guessed; pass explicit properties through
    ``curve_from_properties`` if mean data is required.
    """
    if fractile.lower() not in {"low", "low_fractile", "5%", "5_percent"}:
        raise NotImplementedError("Built-in RP-C208 mean curves are not available; supply explicit curve properties")
    grade_key = grade.upper().replace(" ", "")
    if grade_key not in _DNV_C208_LOW_FRACTILE_STEEL_MPA:
        raise ValueError(f"Unsupported RP-C208 steel grade {grade!r}; use one of {sorted(_DNV_C208_LOW_FRACTILE_STEEL_MPA)}")
    thickness_mm = float(thickness) * 1000.0
    if thickness_mm <= 0.0:
        raise ValueError("thickness must be positive")
    rows = _DNV_C208_LOW_FRACTILE_STEEL_MPA[grade_key]
    selected = rows[-1][2]
    for lower, upper, properties_mpa in rows:
        if thickness_mm <= upper and thickness_mm > lower:
            selected = properties_mpa
            break
    properties = {
        "sigma_prop": float(selected["sigma_prop"]) * _MPA,
        "sigma_yield": float(selected["sigma_yield"]) * _MPA,
        "sigma_yield_2": float(selected["sigma_yield_2"]) * _MPA,
        "eps_p_y1": float(selected.get("eps_p_y1", 0.004)),
        "eps_p_y2": float(selected["eps_p_y2"]),
        "K": float(selected["K"]) * _MPA,
        "n": float(selected["n"]),
    }
    return curve_from_properties(properties)
