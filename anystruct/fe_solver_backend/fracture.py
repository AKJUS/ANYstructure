"""Simplified nonlinear-static fracture and element erosion helpers.

This module implements a deliberately narrow damage model: elements are
softened after a converged nonlinear-static increment when a scalar material
state exceeds a threshold.  It is not crack propagation or fracture mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class FractureConfig:
    """Configuration for strain-triggered element erosion.

    The v1 trigger is the maximum equivalent plastic strain stored in an
    element state.  Deleted elements keep a small residual stiffness by default
    to avoid creating abrupt mechanisms in the next increment.
    """

    threshold: float
    residual_stiffness_fraction: float = 1.0e-6
    max_deleted_fraction: float = 0.25
    min_load_factor: float = 0.0
    element_scope: Tuple[str, ...] = ("shell", "beam")
    delete_after_converged_increment: bool = True
    record_history: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("FractureConfig.threshold must be positive")
        if (
            not np.isfinite(self.residual_stiffness_fraction)
            or self.residual_stiffness_fraction < 0.0
            or self.residual_stiffness_fraction > 1.0
        ):
            raise ValueError("FractureConfig.residual_stiffness_fraction must be in [0, 1]")
        if not np.isfinite(self.max_deleted_fraction) or not (0.0 < self.max_deleted_fraction <= 1.0):
            raise ValueError("FractureConfig.max_deleted_fraction must be in (0, 1]")
        if not np.isfinite(self.min_load_factor) or self.min_load_factor < 0.0:
            raise ValueError("FractureConfig.min_load_factor must be non-negative")
        scope = tuple(str(item).lower() for item in self.element_scope)
        invalid = sorted(set(scope) - {"shell", "beam"})
        if invalid:
            raise ValueError(f"Unsupported fracture element_scope entries: {invalid}")
        object.__setattr__(self, "element_scope", scope)
        if not self.delete_after_converged_increment:
            raise ValueError("Fracture v1 only supports delete_after_converged_increment=True")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "residual_stiffness_fraction": float(self.residual_stiffness_fraction),
            "max_deleted_fraction": float(self.max_deleted_fraction),
            "min_load_factor": float(self.min_load_factor),
            "element_scope": list(self.element_scope),
            "delete_after_converged_increment": bool(self.delete_after_converged_increment),
            "record_history": bool(self.record_history),
        }


ElementDeletionConfig = FractureConfig


@dataclass(frozen=True)
class ImpactFractureConfig:
    """Contact-triggered erosion for rigid-sphere impact analyses.

    This is intentionally separate from nonlinear-static plastic strain
    fracture.  Linear impact runs do not carry plastic integration-point state,
    so v1 impact erosion uses contact observables after a converged substep.
    """

    threshold: float
    trigger: str = "contact_force"
    residual_stiffness_fraction: float = 1.0e-6
    max_deleted_fraction: float = 0.25
    min_time: float = 0.0
    contact_area_radius_fraction: float = 0.25
    record_history: bool = True

    def __post_init__(self) -> None:
        trigger = str(self.trigger).lower()
        if trigger not in {"contact_force", "penetration_ratio", "contact_pressure"}:
            raise ValueError("ImpactFractureConfig.trigger must be 'contact_force', 'penetration_ratio', or 'contact_pressure'")
        if not np.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("ImpactFractureConfig.threshold must be positive")
        if (
            not np.isfinite(self.residual_stiffness_fraction)
            or self.residual_stiffness_fraction < 0.0
            or self.residual_stiffness_fraction > 1.0
        ):
            raise ValueError("ImpactFractureConfig.residual_stiffness_fraction must be in [0, 1]")
        if not np.isfinite(self.max_deleted_fraction) or not (0.0 < self.max_deleted_fraction <= 1.0):
            raise ValueError("ImpactFractureConfig.max_deleted_fraction must be in (0, 1]")
        if not np.isfinite(self.min_time) or self.min_time < 0.0:
            raise ValueError("ImpactFractureConfig.min_time must be non-negative")
        if not np.isfinite(self.contact_area_radius_fraction) or self.contact_area_radius_fraction <= 0.0:
            raise ValueError("ImpactFractureConfig.contact_area_radius_fraction must be positive")
        object.__setattr__(self, "trigger", trigger)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "trigger": self.trigger,
            "residual_stiffness_fraction": float(self.residual_stiffness_fraction),
            "max_deleted_fraction": float(self.max_deleted_fraction),
            "min_time": float(self.min_time),
            "contact_area_radius_fraction": float(self.contact_area_radius_fraction),
            "record_history": bool(self.record_history),
        }


@dataclass(frozen=True)
class ImpactDamageConfig:
    """Capacity-based engineering damage for rigid-sphere shell impact.

    This layer estimates contact-driven shell demand in an otherwise linear
    transient impact solve.  It is an erosion/screening model, not fracture
    mechanics, crack growth, or material nonlinear impact.
    """

    mode: str = "accumulated_damage"
    capacity_basis: str = "yield"
    damage_threshold: float = 1.0
    softening_start: float = 0.6
    delete_at: float = 1.0
    min_contact_area: float = 1.0e-6
    neighbor_smoothing: bool = False
    residual_stiffness_fraction: float = 1.0e-6
    max_deleted_fraction: float = 0.25
    user_capacity: Optional[float] = None
    contact_area_radius_fraction: float = 0.25
    impulse_reference_time: float = 0.01
    plastic_strain_capacity: float = 0.01
    strain_scale: float = 1.0
    record_history: bool = True

    def __post_init__(self) -> None:
        mode = str(self.mode).lower()
        if mode not in {"accumulated_damage", "instant_threshold"}:
            raise ValueError("ImpactDamageConfig.mode must be 'accumulated_damage' or 'instant_threshold'")
        capacity_basis = str(self.capacity_basis).lower()
        if capacity_basis not in {"yield", "ultimate_proxy", "user"}:
            raise ValueError("ImpactDamageConfig.capacity_basis must be 'yield', 'ultimate_proxy', or 'user'")
        if capacity_basis == "user" and (
            self.user_capacity is None or not np.isfinite(self.user_capacity) or self.user_capacity <= 0.0
        ):
            raise ValueError("ImpactDamageConfig.user_capacity must be positive when capacity_basis='user'")
        if not np.isfinite(self.damage_threshold) or self.damage_threshold <= 0.0:
            raise ValueError("ImpactDamageConfig.damage_threshold must be positive")
        if not np.isfinite(self.softening_start) or self.softening_start < 0.0:
            raise ValueError("ImpactDamageConfig.softening_start must be non-negative")
        if not np.isfinite(self.delete_at) or self.delete_at <= self.softening_start:
            raise ValueError("ImpactDamageConfig.delete_at must be greater than softening_start")
        if not np.isfinite(self.min_contact_area) or self.min_contact_area <= 0.0:
            raise ValueError("ImpactDamageConfig.min_contact_area must be positive")
        if (
            not np.isfinite(self.residual_stiffness_fraction)
            or self.residual_stiffness_fraction < 0.0
            or self.residual_stiffness_fraction > 1.0
        ):
            raise ValueError("ImpactDamageConfig.residual_stiffness_fraction must be in [0, 1]")
        if not np.isfinite(self.max_deleted_fraction) or not (0.0 < self.max_deleted_fraction <= 1.0):
            raise ValueError("ImpactDamageConfig.max_deleted_fraction must be in (0, 1]")
        if not np.isfinite(self.contact_area_radius_fraction) or self.contact_area_radius_fraction <= 0.0:
            raise ValueError("ImpactDamageConfig.contact_area_radius_fraction must be positive")
        if not np.isfinite(self.impulse_reference_time) or self.impulse_reference_time <= 0.0:
            raise ValueError("ImpactDamageConfig.impulse_reference_time must be positive")
        if not np.isfinite(self.plastic_strain_capacity) or self.plastic_strain_capacity <= 0.0:
            raise ValueError("ImpactDamageConfig.plastic_strain_capacity must be positive")
        if not np.isfinite(self.strain_scale) or self.strain_scale <= 0.0:
            raise ValueError("ImpactDamageConfig.strain_scale must be positive")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "capacity_basis", capacity_basis)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "capacity_basis": self.capacity_basis,
            "damage_threshold": float(self.damage_threshold),
            "softening_start": float(self.softening_start),
            "delete_at": float(self.delete_at),
            "min_contact_area": float(self.min_contact_area),
            "neighbor_smoothing": bool(self.neighbor_smoothing),
            "residual_stiffness_fraction": float(self.residual_stiffness_fraction),
            "max_deleted_fraction": float(self.max_deleted_fraction),
            "user_capacity": None if self.user_capacity is None else float(self.user_capacity),
            "contact_area_radius_fraction": float(self.contact_area_radius_fraction),
            "impulse_reference_time": float(self.impulse_reference_time),
            "plastic_strain_capacity": float(self.plastic_strain_capacity),
            "strain_scale": float(self.strain_scale),
            "record_history": bool(self.record_history),
        }


@dataclass(frozen=True)
class PlasticImpactDamageConfig:
    """Plastic-strain-driven erosion for nonlinear impact solves.

    Damage is evaluated from committed element plastic state after a converged
    nonlinear transient substep.  It is still engineering erosion, not crack
    propagation or cohesive fracture.
    """

    threshold: float = 0.01
    criterion: str = "fixed"
    softening_start: float = 0.6
    delete_at: float = 1.0
    residual_stiffness_fraction: float = 1.0e-6
    max_deleted_fraction: float = 0.25
    element_scope: Sequence[str] = ("shell", "beam")
    record_history: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or self.threshold <= 0.0:
            raise ValueError("PlasticImpactDamageConfig.threshold must be positive")
        if self.criterion not in {"fixed", "mesh_scaled_gl", "rtcl"}:
            raise ValueError("PlasticImpactDamageConfig.criterion must be 'fixed', 'mesh_scaled_gl' or 'rtcl'")
        if not np.isfinite(self.softening_start) or not (0.0 <= self.softening_start < 1.0):
            raise ValueError("PlasticImpactDamageConfig.softening_start must be in [0, 1)")
        if not np.isfinite(self.delete_at) or self.delete_at <= self.softening_start:
            raise ValueError("PlasticImpactDamageConfig.delete_at must be greater than softening_start")
        if (
            not np.isfinite(self.residual_stiffness_fraction)
            or self.residual_stiffness_fraction < 0.0
            or self.residual_stiffness_fraction > 1.0
        ):
            raise ValueError("PlasticImpactDamageConfig.residual_stiffness_fraction must be in [0, 1]")
        if not np.isfinite(self.max_deleted_fraction) or not (0.0 < self.max_deleted_fraction <= 1.0):
            raise ValueError("PlasticImpactDamageConfig.max_deleted_fraction must be in (0, 1]")
        scope = tuple(str(item).lower() for item in self.element_scope)
        unsupported = [item for item in scope if item not in {"shell", "beam"}]
        if unsupported:
            raise ValueError("PlasticImpactDamageConfig.element_scope supports only 'shell' and 'beam'")
        object.__setattr__(self, "element_scope", scope)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger": "equivalent_plastic_strain",
            "threshold": float(self.threshold),
            "criterion": str(self.criterion),
            "softening_start": float(self.softening_start),
            "delete_at": float(self.delete_at),
            "residual_stiffness_fraction": float(self.residual_stiffness_fraction),
            "max_deleted_fraction": float(self.max_deleted_fraction),
            "element_scope": list(self.element_scope),
            "record_history": bool(self.record_history),
        }


@dataclass(frozen=True)
class DeletedElementRecord:
    """One element erosion event."""

    element_id: int
    element_type: str
    step_index: int
    load_factor: float
    trigger_name: str
    trigger_value: float
    threshold: float
    location: str
    measure: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": int(self.element_id),
            "element_type": self.element_type,
            "step_index": int(self.step_index),
            "load_factor": float(self.load_factor),
            "trigger_name": self.trigger_name,
            "trigger_value": float(self.trigger_value),
            "threshold": float(self.threshold),
            "location": self.location,
            "measure": float(self.measure),
        }


def element_fracture_category(element: Any) -> Optional[str]:
    from .elements import BeamElement, ShellElement

    if isinstance(element, ShellElement):
        return "shell"
    if isinstance(element, BeamElement):
        return "beam"
    return None


def element_measure(mesh: Any, element: Any) -> float:
    """Return shell area or beam length for diagnostics."""
    category = element_fracture_category(element)
    if category == "shell":
        try:
            cache = element._nonlinear_geometry(mesh)
            return float(np.sum(np.asarray(cache.get("detw_all", []), dtype=float)))
        except Exception:
            coords = np.asarray(element.get_node_coordinates(mesh), dtype=float)
            if coords.shape[0] < 3:
                return 0.0
            area = 0.0
            p0 = coords[0]
            for idx in range(1, coords.shape[0] - 1):
                area += 0.5 * float(np.linalg.norm(np.cross(coords[idx] - p0, coords[idx + 1] - p0)))
            return area
    if category == "beam":
        try:
            coords = np.asarray(element.get_node_coordinates(mesh), dtype=float)
            if coords.shape[0] >= 2:
                return float(np.linalg.norm(coords[-1] - coords[0]))
        except Exception:
            return 0.0
    return 0.0


def rtcl_triaxiality_weight(triaxiality: np.ndarray) -> np.ndarray:
    """RTCL damage weight versus stress triaxiality (Tornqvist 2003).

    Combines Cockcroft-Latham for shear-dominated states with Rice-Tracey
    void growth for tension-dominated states, normalised so uniaxial tension
    (eta = 1/3) weighs 1:

        eta <= -1/3           : 0 (no ductile damage in compression)
        -1/3 < eta < 1/3      : (2 + 2 eta sqrt(12 - 27 eta^2))
                                / (3 eta + sqrt(12 - 27 eta^2))
        eta >= 1/3            : exp(1.5 eta - 0.5)

    Pure shear (eta = 0) weighs ~0.577 and equibiaxial tension (eta = 2/3,
    the plane-stress maximum) ~1.65, matching the published RTCL curve used
    in ship-collision studies.
    """
    eta = np.asarray(triaxiality, dtype=float)
    eta_mid = np.clip(eta, -1.0 / 3.0, 1.0 / 3.0)
    root = np.sqrt(np.maximum(12.0 - 27.0 * eta_mid**2, 0.0))
    cockcroft_latham = np.where(
        3.0 * eta_mid + root > 1.0e-12,
        (2.0 + 2.0 * eta_mid * root) / np.maximum(3.0 * eta_mid + root, 1.0e-12),
        0.0,
    )
    rice_tracey = np.exp(1.5 * eta - 0.5)
    return np.where(eta <= -1.0 / 3.0, 0.0, np.where(eta < 1.0 / 3.0, cockcroft_latham, rice_tracey))


def state_rtcl_increment(
    state: Any,
    previous_alpha: Optional[np.ndarray],
    elastic_modulus: float,
    poisson_ratio: float,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """RTCL-weighted plastic strain increments for one committed element state.

    Returns ``(alpha, weighted_increment)`` per integration point, where the
    increment is ``rtcl_weight(eta) * max(alpha - previous_alpha, 0)`` with the
    triaxiality ``eta`` evaluated from the return-mapped plane-stress state
    (``sigma = C_el (eps - eps_p)``).  Beam fiber states use the uniaxial
    limits: weight 1 for fibers in tension, 0 in compression.  Returns ``None``
    when the state carries no usable plastic data.
    """
    if not isinstance(state, Mapping):
        return None
    alpha = np.asarray(state.get("alpha", ()), dtype=float).reshape(-1)
    if alpha.size == 0:
        return None
    if previous_alpha is None or np.asarray(previous_alpha).size != alpha.size:
        previous = np.zeros_like(alpha)
    else:
        previous = np.asarray(previous_alpha, dtype=float).reshape(-1)
    delta = np.maximum(alpha - previous, 0.0)

    fiber_stress = np.asarray(state.get("fiber_stress", ()), dtype=float).reshape(-1)
    if fiber_stress.size == alpha.size:
        # Uniaxial fiber section: eta = +-1/3 exactly.
        weight = np.where(fiber_stress > 0.0, 1.0, 0.0)
        return alpha, weight * delta

    layer = np.asarray(state.get("layer_strain", ()), dtype=float)
    plastic = np.asarray(state.get("plastic_strain", ()), dtype=float)
    if layer.size == 0 or layer.shape != plastic.shape:
        return None
    layer = layer.reshape(-1, layer.shape[-1]) if layer.ndim > 1 else layer.reshape(-1, 1)
    if layer.shape[-1] != 3 or layer.shape[0] != alpha.size:
        return None
    elastic = layer - plastic.reshape(layer.shape)
    factor = float(elastic_modulus) / (1.0 - float(poisson_ratio) ** 2)
    sxx = factor * (elastic[:, 0] + float(poisson_ratio) * elastic[:, 1])
    syy = factor * (elastic[:, 1] + float(poisson_ratio) * elastic[:, 0])
    sxy = factor * (1.0 - float(poisson_ratio)) / 2.0 * elastic[:, 2]
    von_mises = np.sqrt(np.maximum(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
    mean_stress = (sxx + syy) / 3.0
    triaxiality = np.where(von_mises > 1.0e-9 * max(float(elastic_modulus), 1.0), mean_stress / np.maximum(von_mises, 1.0e-30), 0.0)
    return alpha, rtcl_triaxiality_weight(triaxiality) * delta


def state_equivalent_plastic_strain(state: Any) -> Tuple[float, str]:
    """Return max equivalent plastic strain and a compact location label."""
    if not isinstance(state, Mapping):
        return 0.0, "no_state"
    alpha = np.asarray(state.get("alpha", []), dtype=float).reshape(-1)
    if alpha.size == 0:
        return 0.0, "no_alpha"
    index = int(np.argmax(alpha))
    return float(alpha[index]), f"alpha[{index}]"


def detect_new_deletions(
    model: Any,
    states: Mapping[int, Any],
    config: FractureConfig,
    deleted_element_ids: Iterable[int],
    *,
    step_index: int,
    load_factor: float,
) -> Tuple[Tuple[DeletedElementRecord, ...], float]:
    """Find newly failed elements from committed nonlinear states."""
    if float(load_factor) + 1.0e-12 < config.min_load_factor:
        return (), 0.0

    deleted: Set[int] = {int(element_id) for element_id in deleted_element_ids}
    records = []
    max_utilization = 0.0
    for element_id, state in states.items():
        if not isinstance(element_id, int) or element_id in deleted:
            continue
        element = model.mesh.get_element(int(element_id))
        if element is None:
            continue
        category = element_fracture_category(element)
        if category is None or category not in config.element_scope:
            continue
        trigger_value, location = state_equivalent_plastic_strain(state)
        utilization = trigger_value / config.threshold if config.threshold > 0.0 else 0.0
        max_utilization = max(max_utilization, float(utilization))
        if trigger_value >= config.threshold:
            records.append(
                DeletedElementRecord(
                    element_id=int(element_id),
                    element_type=category,
                    step_index=int(step_index),
                    load_factor=float(load_factor),
                    trigger_name="max_equivalent_plastic_strain",
                    trigger_value=float(trigger_value),
                    threshold=float(config.threshold),
                    location=location,
                    measure=element_measure(model.mesh, element),
                )
            )
    return tuple(records), max_utilization


def deleted_pressure_load_resultant(model: Any, load_case: Optional[Any], deleted_element_ids: Iterable[int]) -> np.ndarray:
    """Assemble the resultant force removed from deleted pressure elements."""
    deleted = {int(element_id) for element_id in deleted_element_ids}
    result = np.zeros(3, dtype=float)
    if load_case is None or not deleted:
        return result
    for element_id, pressure in getattr(load_case, "pressure_loads", {}).items():
        if int(element_id) not in deleted:
            continue
        element = model.mesh.get_element(int(element_id))
        if element is None:
            continue
        f_elem = load_case._consistent_pressure_load(element, model.mesh, float(pressure))
        for idx in range(0, len(f_elem), 6):
            result += f_elem[idx:idx + 3]
    return result


def filtered_load_case_for_deleted_elements(load_case: Optional[Any], deleted_element_ids: Iterable[int]) -> Optional[Any]:
    """Return a shallow load-case copy with deleted pressure elements removed."""
    if load_case is None:
        return None
    deleted = {int(element_id) for element_id in deleted_element_ids}
    if not deleted:
        return load_case
    import copy

    filtered = copy.copy(load_case)
    filtered.pressure_loads = {
        int(element_id): float(pressure)
        for element_id, pressure in getattr(load_case, "pressure_loads", {}).items()
        if int(element_id) not in deleted
    }
    return filtered


def fracture_summary(
    model: Any,
    config: Optional[FractureConfig],
    records: Sequence[DeletedElementRecord],
    deleted_element_ids: Iterable[int],
    *,
    max_utilization: float = 0.0,
    warnings: Sequence[str] = (),
) -> Dict[str, Any]:
    deleted = {int(element_id) for element_id in deleted_element_ids}
    shell_area = 0.0
    beam_length = 0.0
    for element_id in deleted:
        element = model.mesh.get_element(element_id)
        if element is None:
            continue
        category = element_fracture_category(element)
        if category == "shell":
            shell_area += element_measure(model.mesh, element)
        elif category == "beam":
            beam_length += element_measure(model.mesh, element)
    first = min((record.load_factor for record in records), default=None)
    peak_trigger = max((record.trigger_value for record in records), default=0.0)
    return {
        "enabled": config is not None,
        "config": None if config is None else config.to_dict(),
        "deleted_count": len(deleted),
        "deleted_element_ids": sorted(deleted),
        "deleted_shell_area": float(shell_area),
        "deleted_beam_length": float(beam_length),
        "first_deletion_load_factor": None if first is None else float(first),
        "max_trigger_value": float(peak_trigger),
        "max_fracture_utilization": float(max_utilization),
        "records": [record.to_dict() for record in records],
        "warnings": list(warnings),
    }


def mpc_warning_for_deleted_shells(model: Any, deleted_element_ids: Iterable[int]) -> Optional[str]:
    """Warn when eroded shell nodes still participate in MPC topology."""
    shell_dofs = set()
    for element_id in deleted_element_ids:
        element = model.mesh.get_element(int(element_id))
        if element is None or element_fracture_category(element) != "shell":
            continue
        for node_id in getattr(element, "node_ids", []):
            node = model.mesh.get_node(int(node_id))
            if node is not None:
                shell_dofs.update(int(dof) for dof in node.dofs)
    if not shell_dofs:
        return None
    for element in model.mesh.elements.values():
        if not hasattr(element, "get_mpc_constraints"):
            continue
        for constraint in element.get_mpc_constraints(model.mesh):
            masters = constraint.get("masters", {}) or {}
            if shell_dofs.intersection(int(dof) for dof in masters):
                return (
                    "Fracture v1 softens/deactivates elements only; nodes, DOFs, "
                    "MPCs and beam-shell couplings remain intact after shell erosion."
                )
    return None
