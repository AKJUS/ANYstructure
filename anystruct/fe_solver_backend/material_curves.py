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

import numpy as np


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
