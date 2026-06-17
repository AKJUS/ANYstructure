"""Beam/member geometric validity evidence helpers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .elements import BeamElement
from .fe_core import FEModel


DEFAULT_BEAM_VALIDITY_PATH = Path("reports/beam_validity/beam_validity_report.json")


def _git_sha() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    except Exception:
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _single_beam(corotational: bool) -> tuple[FEModel, BeamElement]:
    model = FEModel("beam_validity")
    model.add_material("steel", 210.0e9, 0.3)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 2.0, 0.0, 0.0)
    section = {"area": 0.01, "Iy": 2.0e-6, "Iz": 1.0e-6, "J": 1.0e-6, "orientation": (0.0, 0.0, 1.0)}
    if corotational:
        section["geometric_nonlinearity"] = "corotational"
    element = BeamElement(1, [1, 2], "steel", section)
    model.add_element(1, element)
    return model, element


def corotational_rigid_rotation_metric(angle_degrees: float = 35.0) -> Dict[str, Any]:
    """Compare default and corotational internal force for a finite rigid rotation."""
    theta = float(np.deg2rad(angle_degrees))
    length = 2.0
    u = np.zeros(12)
    u[6] = length * np.cos(theta) - length
    u[7] = length * np.sin(theta)
    u[5] = theta
    u[11] = theta

    default_model, default_element = _single_beam(corotational=False)
    corot_model, corot_element = _single_beam(corotational=True)
    f_default, _k_default, _state_default = default_element.compute_nonlinear_response(default_model.mesh, default_model.get_material("steel"), u)
    f_corot, _k_corot, state_corot = corot_element.compute_nonlinear_response(corot_model.mesh, corot_model.get_material("steel"), u)
    return {
        "angle_degrees": float(angle_degrees),
        "default_force_norm": float(np.linalg.norm(f_default)),
        "corotational_force_norm": float(np.linalg.norm(f_corot)),
        "force_norm_ratio_corot_to_default": float(np.linalg.norm(f_corot) / max(np.linalg.norm(f_default), 1.0e-30)),
        "corotational_basic_deformation_norm": float(state_corot.get("basic_deformation_norm", 0.0)),
        "corotational_axial_extension": float(state_corot.get("axial_extension", 0.0)),
    }


def corotational_axial_extension_metric(extension: float = 0.002) -> Dict[str, Any]:
    """Check axial extension response for the corotational beam."""
    model, element = _single_beam(corotational=True)
    u = np.zeros(12)
    u[6] = float(extension)
    force, _k, state = element.compute_nonlinear_response(model.mesh, model.get_material("steel"), u)
    expected = 210.0e9 * 0.01 / 2.0 * float(extension)
    return {
        "extension": float(extension),
        "expected_end_force": float(expected),
        "computed_end_force": float(force[6]),
        "relative_error": float(abs(force[6] - expected) / max(abs(expected), 1.0)),
        "current_length": float(state.get("current_length", 0.0)),
    }


def generate_beam_validity_report() -> Dict[str, Any]:
    """Generate local beam/member geometric validity report."""
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git_sha(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "corotational_v1": {
            "rigid_rotation": corotational_rigid_rotation_metric(),
            "axial_extension": corotational_axial_extension_metric(),
        },
        "known_limitations": [
            "Corotational v1 is opt-in for 2-node elastic beam geometric response via cross_section['geometric_nonlinearity']='corotational'.",
            "Fiber-section plasticity keeps the existing beam-column path in this batch.",
        ],
    }


def write_beam_validity_report(path: Path | str = DEFAULT_BEAM_VALIDITY_PATH) -> Dict[str, Any]:
    report = generate_beam_validity_report()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
