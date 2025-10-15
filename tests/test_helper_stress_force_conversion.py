import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anystruct.helper as hlp


class _DummyCylinder:
    """Minimal stub mimicking ``CylinderAndCurvedPlate`` ``get_Itot``."""

    def __init__(self, itot):
        self._itot = itot

    def get_Itot(self, **_kwargs):
        return self._itot


def _round_trip_forces(forces, **kwargs):
    stresses = hlp.helper_cylinder_stress_to_force_to_stress(forces=forces, **kwargs)
    back = hlp.helper_cylinder_stress_to_force_to_stress(stresses=stresses, **kwargs)
    return stresses, back


def test_round_trip_unstiffened_shell():
    dummy = _DummyCylinder(8.9)
    kwargs = dict(
        geometry=1,
        shell_t=0.02,
        shell_radius=6.5,
        shell_spacing=1.0,
        CylinderAndCurvedPlate=dummy,
    )
    forces = (1.2e6, 5.5e7, 2.4e5, 3.1e5)

    _, converted = _round_trip_forces(forces, **kwargs)

    for original, recovered in zip(forces, converted[:4]):
        assert math.isclose(original, recovered, rel_tol=1e-9)


def test_round_trip_longitudinal_stiffened_shell():
    dummy = _DummyCylinder(12.1)
    kwargs = dict(
        geometry=3,
        shell_t=0.025,
        shell_radius=4.5,
        shell_spacing=0.8,
        hw=0.35,
        tw=0.018,
        b=0.22,
        tf=0.02,
        CylinderAndCurvedPlate=dummy,
    )
    forces = (2.5e6, 7.2e7, 4.3e5, 5.6e5)

    _, converted = _round_trip_forces(forces, **kwargs)

    for original, recovered in zip(forces, converted[:4]):
        assert math.isclose(original, recovered, rel_tol=1e-9)
