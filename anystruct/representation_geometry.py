"""Shared geometry helpers for visual/export/FE structure representations.

The helpers in this module do not normalise or change calculation inputs.
They only place repeated representation members so any remainder is split
symmetrically about the model mid-plane.
"""

from __future__ import annotations

import math
from typing import Iterable


EPS = 1.0e-9


def positive_spacing(value: object, tolerance: float = EPS) -> float:
    """Return a positive spacing value, or zero when unavailable."""

    try:
        spacing = float(value)
    except (TypeError, ValueError):
        return 0.0
    return spacing if spacing > tolerance else 0.0


def centered_member_positions(
        total_length: float,
        spacing: float,
        *,
        fallback_midpoint: bool = True,
        max_count: int | None = 1000,
) -> tuple[float, ...]:
    """Return repeated member/support stations with symmetric end compensation.

    ``spacing`` is preserved as the nominal distance between adjacent stations.
    If ``total_length`` is not an exact multiple of ``spacing``, the leftover
    length is split equally at both ends.
    """

    total_length = max(float(total_length), EPS)
    spacing = positive_spacing(spacing)
    tol = max(total_length * EPS, EPS)
    if spacing <= 0.0:
        return (0.5 * total_length,) if fallback_midpoint else ()

    full_count = int(math.floor(total_length / spacing))
    if full_count <= 0:
        return (0.5 * total_length,) if fallback_midpoint else ()

    offset = 0.5 * (total_length - full_count * spacing)
    if offset <= tol:
        positions = [spacing * idx for idx in range(1, full_count)]
    else:
        positions = [offset + spacing * idx for idx in range(full_count + 1)]

    positions = [
        position
        for position in positions
        if tol < position < total_length - tol
    ]
    if not positions and fallback_midpoint:
        positions = [0.5 * total_length]
    if max_count is not None and max_count > 0 and len(positions) > max_count:
        positions = _symmetrically_sample_positions(positions, max_count)
    return tuple(float(position) for position in positions)


def centered_bay_breaks(
        total_length: float,
        spacing: float,
        *,
        max_count: int | None = 1000,
) -> tuple[float, ...]:
    """Return axis breakpoints for bays bounded by centered stations."""

    total_length = max(float(total_length), EPS)
    stations = centered_member_positions(
        total_length,
        spacing,
        fallback_midpoint=False,
        max_count=max_count,
    )
    return cleaned_axis_values((0.0, *stations, total_length), total_length)


def cleaned_axis_values(values: Iterable[float], total_length: float, tolerance: float | None = None) -> tuple[float, ...]:
    """Return sorted unique values clipped to ``[0, total_length]``."""

    total_length = max(float(total_length), EPS)
    tol = max(total_length * EPS, EPS) if tolerance is None else max(float(tolerance), 0.0)
    clean: list[float] = []
    for value in sorted(float(item) for item in values):
        value = min(max(value, 0.0), total_length)
        if not clean or abs(value - clean[-1]) > tol:
            clean.append(value)
    if not clean:
        clean = [0.0, total_length]
    clean[0] = 0.0 if abs(clean[0]) <= tol else clean[0]
    clean[-1] = total_length if abs(clean[-1] - total_length) <= tol else clean[-1]
    return tuple(clean)


def bay_ranges_from_support_positions(
        total_length: float,
        supports: Iterable[float],
        support_gap: float = 0.0,
) -> tuple[tuple[float, float], ...]:
    """Return member segment ranges split by internal support/girder lines."""

    total_length = max(float(total_length), 0.0)
    support_gap = max(float(support_gap), 0.0)
    if total_length <= EPS:
        return ()

    tol = max(total_length * EPS, EPS)
    internal_supports = sorted(
        position
        for position in (float(value) for value in supports)
        if tol < position < total_length - tol
    )
    breakpoints = (0.0, *internal_supports, total_length)
    ranges: list[tuple[float, float]] = []
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        left_gap = support_gap / 2.0 if any(abs(start - support) <= tol for support in internal_supports) else 0.0
        right_gap = support_gap / 2.0 if any(abs(end - support) <= tol for support in internal_supports) else 0.0
        segment_start = max(start + left_gap, 0.0)
        segment_end = min(end - right_gap, total_length)
        if segment_end > segment_start:
            ranges.append((float(segment_start), float(segment_end)))
    return tuple(ranges)


def closed_loop_member_count(total_length: float, spacing: float) -> int:
    """Return an equal cyclic member count for a closed circumference."""

    try:
        total_length = float(total_length)
        spacing = float(spacing)
    except (TypeError, ValueError):
        return 0
    if total_length <= 0.0 or spacing <= EPS:
        return 0
    return max(int(round(total_length / spacing)), 1)


def _symmetrically_sample_positions(positions: list[float], max_count: int) -> list[float]:
    """Sample a dense station list while preserving first/last symmetry."""

    if max_count <= 0 or len(positions) <= max_count:
        return positions
    if max_count == 1:
        return [positions[len(positions) // 2]]
    last = len(positions) - 1
    indexes = sorted({round(idx * last / (max_count - 1)) for idx in range(max_count)})
    return [positions[int(index)] for index in indexes]
