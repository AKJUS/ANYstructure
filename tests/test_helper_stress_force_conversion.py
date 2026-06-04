import math
import sys
from pathlib import Path

import pytest

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


def test_conical_force_conversion_uses_section_4_stress_components():
    dummy = _DummyCylinder(12.1)
    alpha = math.degrees(math.atan((6.5 - 4.0) / 5.0))
    kwargs = dict(
        geometry=9,
        shell_t=0.02,
        shell_radius=4.0,
        shell_spacing=5.0,
        CylinderAndCurvedPlate=dummy,
        conical=True,
        psd=-0.1,
        cone_r1=4.0,
        cone_r2=6.5,
        cone_alpha=alpha,
        shell_lenght_l=5.0,
    )
    forces = (-1000, 2000, 1000, 500, 200, 100)

    sasd, smsd, tTsd, tQsd, shsd = hlp.helper_cylinder_stress_to_force_to_stress(
        forces=forces,
        **kwargs,
    )

    te = 0.02 * math.cos(math.radians(alpha))
    assert sasd == pytest.approx(-0.1e6 * 4.0 / (2 * te) - 1000 * 1000 / (2 * math.pi * 4.0 * te))
    assert smsd == pytest.approx(math.sqrt(2000 ** 2 + 1000 ** 2) * 1000 / (math.pi * 4.0 ** 2 * te))
    assert tTsd == pytest.approx(500 * 1000 / (2 * math.pi * 4.0 ** 2 * te))
    assert tQsd == pytest.approx(math.sqrt(200 ** 2 + 100 ** 2) * 1000 / (math.pi * 4.0 * te))
    assert shsd == pytest.approx(-0.1e6 * 4.0 / te)


def test_conical_stress_input_converts_to_equivalent_axis_one_forces():
    dummy = _DummyCylinder(12.1)
    alpha = math.degrees(math.atan((6.5 - 4.0) / 5.0))
    kwargs = dict(
        geometry=9,
        shell_t=0.02,
        shell_radius=4.0,
        shell_spacing=5.0,
        CylinderAndCurvedPlate=dummy,
        conical=True,
        psd=-0.1,
        cone_r1=4.0,
        cone_r2=6.5,
        cone_alpha=alpha,
        shell_lenght_l=5.0,
    )
    stresses = (-60e6, 20e6, 3e6, 2e6, -11e6)

    forces = hlp.helper_cylinder_stress_to_force_to_stress(stresses=stresses, **kwargs)
    converted = hlp.helper_cylinder_stress_to_force_to_stress(forces=forces[:6], **kwargs)

    assert forces[2] == 0
    assert forces[5] == 0
    for expected, actual in zip(stresses[:4], converted[:4]):
        assert actual == pytest.approx(expected)
    assert converted[4] == pytest.approx(-0.1e6 * 4.0 / (0.02 * math.cos(math.radians(alpha))))
