"""Limited rigid-sphere-to-shell impact dynamics.

This module intentionally implements a narrow contact mode: one rigid sphere,
frictionless normal penalty contact, and shell midsurface/thickness-offset
targets.  Beam elements respond through their existing structural connection to
contacted shells, but are not direct contact targets in v1.  Optional impact
fracture is contact-threshold shell erosion, not crack propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from .assembly import build_constraint_transformation, reconstruct_full_solution
from .boundary import BoundaryCondition, LoadCase
from .cases import make_result_case
from .dynamics import (
    TransientConfig,
    _full_initial_vector,
    _node_dof_indices,
    _reduced_load,
    _saved_step_count,
    _time_grid,
    _translation_peak,
)
from .elements import ShellElement
from .fracture import (
    DeletedElementRecord,
    ImpactDamageConfig,
    ImpactFractureConfig,
    PlasticImpactDamageConfig,
    element_fracture_category,
    element_measure,
    filtered_load_case_for_deleted_elements,
    fracture_summary,
    state_equivalent_plastic_strain,
)
from .linalg import MatrixClass, factorize
from .matrix_assembly import assemble_load_vector, assemble_mass_matrix, assemble_stiffness_matrix
from .recovery import enforce_memory_limit, estimate_model_memory, recovery_metadata
from .validation import ProductionValidationIssue, ProductionValidationReport, load_vector_resultant

if TYPE_CHECKING:
    from .fe_core import FEModel


@dataclass(frozen=True)
class RigidSphereImpact:
    """Input state for a rigid sphere impact."""

    name: str
    radius: float
    mass: float
    start_point: Sequence[float]
    travel_direction: Sequence[float]
    speed: float
    t_start: float = 0.0

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("sphere radius must be positive")
        if self.mass <= 0.0:
            raise ValueError("sphere mass must be positive")
        if self.speed < 0.0:
            raise ValueError("sphere speed must be non-negative")
        if self.t_start < 0.0:
            raise ValueError("sphere t_start must be non-negative")
        start = np.asarray(self.start_point, dtype=float).reshape(-1)
        direction = np.asarray(self.travel_direction, dtype=float).reshape(-1)
        if start.shape != (3,):
            raise ValueError("sphere start_point must contain three coordinates")
        if direction.shape != (3,):
            raise ValueError("sphere travel_direction must contain three coordinates")
        if float(np.linalg.norm(direction)) <= 0.0:
            raise ValueError("sphere travel_direction must be non-zero")

    @property
    def direction_unit(self) -> np.ndarray:
        direction = np.asarray(self.travel_direction, dtype=float)
        return direction / float(np.linalg.norm(direction))

    @property
    def initial_position(self) -> np.ndarray:
        return np.asarray(self.start_point, dtype=float).copy()

    @property
    def travel_velocity(self) -> np.ndarray:
        return self.direction_unit * float(self.speed)


@dataclass(frozen=True)
class SphereContactConfig:
    """Penalty-contact controls for rigid-sphere impact."""

    penalty_stiffness: Optional[float] = None
    contact_damping: float = 0.0
    search_margin: float = 0.0
    max_contact_iterations: int = 25
    penetration_tolerance: float = 1.0e-8
    force_tolerance: float = 1.0e-6
    side_mode: str = "both"
    contact_surface: str = "midsurface"
    signed_surface_offset: float = 0.0
    target_penetration_fraction: float = 0.01
    auto_penalty_safety_factor: float = 1.0
    max_sphere_travel_fraction: float = 0.25
    max_event_substeps: int = 16
    max_active_contacts: int = 1
    load_patch_radius_factor: float = 1.25
    min_load_patch_nodes: int = 4
    max_load_patch_nodes: int = 12
    save_contact_history: bool = True
    post_separation_time: float = 0.0
    contact_relaxation: str = "aitken"
    beam_contact: bool = False

    def __post_init__(self) -> None:
        if self.penalty_stiffness is not None and self.penalty_stiffness <= 0.0:
            raise ValueError("penalty_stiffness must be positive")
        if self.contact_damping < 0.0:
            raise ValueError("contact_damping must be non-negative")
        if self.search_margin < 0.0:
            raise ValueError("search_margin must be non-negative")
        if self.max_contact_iterations <= 0:
            raise ValueError("max_contact_iterations must be positive")
        if self.penetration_tolerance <= 0.0:
            raise ValueError("penetration_tolerance must be positive")
        if self.force_tolerance <= 0.0:
            raise ValueError("force_tolerance must be positive")
        if self.side_mode != "both":
            raise NotImplementedError("SphereContactConfig v1 supports side_mode='both' only")
        if self.contact_surface not in {"midsurface", "top", "bottom", "signed"}:
            raise ValueError("contact_surface must be 'midsurface', 'top', 'bottom', or 'signed'")
        if self.contact_surface == "signed" and abs(float(self.signed_surface_offset)) > 1.0:
            raise ValueError("signed_surface_offset must be in [-1, 1] as a fraction of half thickness")
        if self.target_penetration_fraction <= 0.0:
            raise ValueError("target_penetration_fraction must be positive")
        if self.auto_penalty_safety_factor <= 0.0:
            raise ValueError("auto_penalty_safety_factor must be positive")
        if self.max_sphere_travel_fraction <= 0.0:
            raise ValueError("max_sphere_travel_fraction must be positive")
        if self.max_event_substeps <= 0:
            raise ValueError("max_event_substeps must be positive")
        if self.max_active_contacts <= 0:
            raise ValueError("max_active_contacts must be positive")
        if self.load_patch_radius_factor <= 0.0:
            raise ValueError("load_patch_radius_factor must be positive")
        if self.min_load_patch_nodes <= 0:
            raise ValueError("min_load_patch_nodes must be positive")
        if self.max_load_patch_nodes < self.min_load_patch_nodes:
            raise ValueError("max_load_patch_nodes must be greater than or equal to min_load_patch_nodes")
        if self.post_separation_time < 0.0:
            raise ValueError("post_separation_time must be non-negative")
        if self.contact_relaxation not in {"aitken", "none"}:
            raise ValueError("contact_relaxation must be 'aitken' or 'none'")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "penalty_stiffness": None if self.penalty_stiffness is None else float(self.penalty_stiffness),
            "contact_damping": float(self.contact_damping),
            "search_margin": float(self.search_margin),
            "max_contact_iterations": int(self.max_contact_iterations),
            "penetration_tolerance": float(self.penetration_tolerance),
            "force_tolerance": float(self.force_tolerance),
            "side_mode": self.side_mode,
            "contact_surface": self.contact_surface,
            "signed_surface_offset": float(self.signed_surface_offset),
            "target_penetration_fraction": float(self.target_penetration_fraction),
            "auto_penalty_safety_factor": float(self.auto_penalty_safety_factor),
            "max_sphere_travel_fraction": float(self.max_sphere_travel_fraction),
            "max_event_substeps": int(self.max_event_substeps),
            "max_active_contacts": int(self.max_active_contacts),
            "load_patch_radius_factor": float(self.load_patch_radius_factor),
            "min_load_patch_nodes": int(self.min_load_patch_nodes),
            "max_load_patch_nodes": int(self.max_load_patch_nodes),
            "save_contact_history": bool(self.save_contact_history),
            "post_separation_time": float(self.post_separation_time),
            "contact_relaxation": self.contact_relaxation,
            "beam_contact": bool(self.beam_contact),
        }


@dataclass(frozen=True)
class NonlinearTransientConfig:
    """Material/geometric nonlinear implicit Newmark controls for impact."""

    enabled: bool = False
    num_layers: int = 5
    max_iterations: int = 20
    residual_tolerance: float = 2.0e-3
    displacement_tolerance: float = 1.0e-8
    contact_force_tolerance: float = 2.0e-3
    line_search: bool = True
    min_line_search_factor: float = 0.125
    max_cutbacks: int = 12
    min_dt: float = 1.0e-8
    tangent_reuse_iterations: int = 0
    record_element_state_history: bool = True
    kinematics: str = "von_karman"

    def __post_init__(self) -> None:
        if str(self.kinematics).lower() not in {"von_karman", "corotational"}:
            raise ValueError("NonlinearTransientConfig.kinematics must be 'von_karman' or 'corotational'")
        object.__setattr__(self, "kinematics", str(self.kinematics).lower())
        if self.num_layers <= 0:
            raise ValueError("NonlinearTransientConfig.num_layers must be positive")
        if self.max_iterations <= 0:
            raise ValueError("NonlinearTransientConfig.max_iterations must be positive")
        if self.residual_tolerance <= 0.0:
            raise ValueError("NonlinearTransientConfig.residual_tolerance must be positive")
        if self.displacement_tolerance <= 0.0:
            raise ValueError("NonlinearTransientConfig.displacement_tolerance must be positive")
        if self.contact_force_tolerance <= 0.0:
            raise ValueError("NonlinearTransientConfig.contact_force_tolerance must be positive")
        if not (0.0 < self.min_line_search_factor <= 1.0):
            raise ValueError("NonlinearTransientConfig.min_line_search_factor must be in (0, 1]")
        if self.max_cutbacks < 0:
            raise ValueError("NonlinearTransientConfig.max_cutbacks must be non-negative")
        if self.min_dt <= 0.0:
            raise ValueError("NonlinearTransientConfig.min_dt must be positive")
        if self.tangent_reuse_iterations < 0:
            raise ValueError("NonlinearTransientConfig.tangent_reuse_iterations must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "num_layers": int(self.num_layers),
            "max_iterations": int(self.max_iterations),
            "residual_tolerance": float(self.residual_tolerance),
            "displacement_tolerance": float(self.displacement_tolerance),
            "contact_force_tolerance": float(self.contact_force_tolerance),
            "line_search": bool(self.line_search),
            "min_line_search_factor": float(self.min_line_search_factor),
            "max_cutbacks": int(self.max_cutbacks),
            "min_dt": float(self.min_dt),
            "tangent_reuse_iterations": int(self.tangent_reuse_iterations),
            "record_element_state_history": bool(self.record_element_state_history),
            "kinematics": self.kinematics,
        }


@dataclass(frozen=True)
class SphereContactRecord:
    """One active contact point at a saved or iterated state."""

    element_id: int
    local_coordinates: Tuple[float, float]
    contact_point: np.ndarray
    normal: np.ndarray
    penetration: float
    normal_force: float
    sphere_force: np.ndarray
    structure_force: np.ndarray
    contact_classification: str = "face"
    nodal_forces: Mapping[int, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": int(self.element_id),
            "local_coordinates": [float(self.local_coordinates[0]), float(self.local_coordinates[1])],
            "contact_point": self.contact_point.tolist(),
            "normal": self.normal.tolist(),
            "penetration": float(self.penetration),
            "normal_force": float(self.normal_force),
            "sphere_force": self.sphere_force.tolist(),
            "structure_force": self.structure_force.tolist(),
            "contact_classification": self.contact_classification,
            "nodal_forces": {int(node_id): value.tolist() for node_id, value in self.nodal_forces.items()},
        }


@dataclass(frozen=True)
class SphereImpactResult:
    """Structural transient histories plus rigid-sphere contact histories."""

    times: np.ndarray
    displacements: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    node_histories: Dict[int, np.ndarray]
    sphere_positions: np.ndarray
    sphere_velocities: np.ndarray
    sphere_accelerations: np.ndarray
    contact_force_history: np.ndarray
    active_contact_history: Tuple[Tuple[Dict[str, Any], ...], ...]
    load_impulse: np.ndarray
    force_impulse: np.ndarray
    moment_impulse: np.ndarray
    sphere_impulse: np.ndarray
    max_penetration: float
    max_penetration_ratio: float
    peak_contact_force: float
    contact_duration: float
    sphere_momentum_balance_error: float
    peak_displacement: float
    peak_displacement_node: Optional[int]
    status: str
    diagnostics: Dict[str, Any]
    result_case: Optional[Dict[str, Any]] = None


def _project_local_coordinates(
    element: ShellElement, coords: np.ndarray, point: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if element.num_nodes in (3, 6):
        local = np.array([1.0 / 3.0, 1.0 / 3.0], dtype=float)
    else:
        local = np.zeros(2, dtype=float)
    for _ in range(12):
        N, dN_dxi, dN_deta = element.compute_shape_functions(float(local[0]), float(local[1]))
        surface_point = N @ coords
        jac = np.column_stack((dN_dxi @ coords, dN_deta @ coords))
        residual = surface_point - point
        lhs = jac.T @ jac
        rhs = jac.T @ residual
        try:
            delta = np.linalg.solve(lhs + 1.0e-14 * np.eye(2), rhs)
        except np.linalg.LinAlgError:
            break
        local -= delta
        if element.num_nodes in (3, 6):
            local[0] = max(float(local[0]), 0.0)
            local[1] = max(float(local[1]), 0.0)
            total = float(local[0] + local[1])
            if total > 1.0:
                local /= total
        else:
            local = np.clip(local, -1.0, 1.0)
        if float(np.linalg.norm(delta)) < 1.0e-12:
            break
    N, dN_dxi, dN_deta = element.compute_shape_functions(float(local[0]), float(local[1]))
    surface_point = N @ coords
    return local, N, dN_dxi, dN_deta, surface_point


def _contact_classification(element: ShellElement, local: np.ndarray, tol: float = 1.0e-7) -> str:
    if element.num_nodes in (3, 6):
        bary = (float(local[0]), float(local[1]), float(1.0 - local[0] - local[1]))
        near = sum(1 for value in bary if abs(value) <= tol)
        if near >= 2:
            return "corner"
        if near == 1:
            return "edge"
        return "face"
    on_x = abs(abs(float(local[0])) - 1.0) <= tol
    on_y = abs(abs(float(local[1])) - 1.0) <= tol
    if on_x and on_y:
        return "corner"
    if on_x or on_y:
        return "edge"
    return "face"


def _surface_offset(element: ShellElement, config: SphereContactConfig) -> float:
    thickness = float(getattr(element, "thickness", 0.0) or 0.0)
    if config.contact_surface == "top":
        return 0.5 * thickness
    if config.contact_surface == "bottom":
        return -0.5 * thickness
    if config.contact_surface == "signed":
        return float(config.signed_surface_offset) * 0.5 * thickness
    return 0.0


def _shell_contact_candidates(model: "FEModel") -> List[ShellElement]:
    return [element for element in model.mesh.elements.values() if isinstance(element, ShellElement)]


def _beam_contact_candidates(model: "FEModel") -> List[Any]:
    from .elements import BeamElement

    return [element for element in model.mesh.elements.values() if isinstance(element, BeamElement)]


def _beam_contact_radius(element: Any) -> float:
    """Circular contact-radius proxy for a beam section.

    ``cross_section["contact_radius"]`` takes precedence; the default is the
    equivalent-area circle radius, an engineering proxy for the physical
    stiffener/girder profile.
    """
    section = getattr(element, "cross_section", {}) or {}
    explicit = section.get("contact_radius")
    if explicit is not None:
        return max(float(explicit), 0.0)
    area = float(section.get("area", 0.0) or 0.0)
    return float(np.sqrt(max(area, 0.0) / np.pi))


class _ContactGeometry:
    """Mesh-constant shell contact geometry arrays, cached on the mesh revision.

    The contact load assembly runs inside contact/Newton iterations, so anything
    that only depends on the undeformed mesh (node coordinates, translation DOF
    indices, element connectivity, representative edge length) is precomputed
    once and reused until the mesh revision changes.
    """

    __slots__ = (
        "elements",
        "element_ids",
        "node_ids",
        "node_id_to_slot",
        "node_base",
        "node_dofs",
        "element_node_slots",
        "element_node_counts",
        "node_mask",
        "representative_edge_length",
        "beam_segment_slots",
        "beam_segment_radii",
        "beam_segment_element_ids",
    )

    def __init__(self, model: "FEModel"):
        elements = _shell_contact_candidates(model)
        beams = _beam_contact_candidates(model)
        self.elements = elements
        self.element_ids = np.asarray([int(element.element_id) for element in elements], dtype=np.int64)
        node_id_to_slot: Dict[int, int] = {}
        for element in elements:
            for node_id in element.node_ids:
                node_id_to_slot.setdefault(int(node_id), len(node_id_to_slot))
        for beam in beams:
            for node_id in beam.node_ids:
                node_id_to_slot.setdefault(int(node_id), len(node_id_to_slot))
        self.node_id_to_slot = node_id_to_slot
        self.node_ids = np.asarray(list(node_id_to_slot), dtype=np.int64)
        n_nodes = len(node_id_to_slot)
        self.node_base = np.zeros((max(n_nodes, 1), 3), dtype=float)
        self.node_dofs = np.zeros((max(n_nodes, 1), 3), dtype=np.intp)
        for node_id, slot in node_id_to_slot.items():
            node = model.mesh.get_node(node_id)
            self.node_base[slot] = (node.x, node.y, node.z)
            self.node_dofs[slot] = np.asarray(node.dofs[:3], dtype=np.intp)
        n_elem = len(elements)
        max_nodes = max((element.num_nodes for element in elements), default=1)
        self.element_node_slots = np.zeros((max(n_elem, 1), max_nodes), dtype=np.intp)
        self.element_node_counts = np.zeros(max(n_elem, 1), dtype=np.intp)
        lengths: List[float] = []
        for index, element in enumerate(elements):
            slots = [node_id_to_slot[int(node_id)] for node_id in element.node_ids]
            self.element_node_counts[index] = len(slots)
            self.element_node_slots[index, : len(slots)] = slots
            self.element_node_slots[index, len(slots):] = slots[0]
            corner_count = 3 if element.num_nodes in (3, 6) else 4
            corners = self.node_base[slots[:corner_count]]
            for corner in range(corner_count):
                lengths.append(float(np.linalg.norm(corners[(corner + 1) % corner_count] - corners[corner])))
        self.node_mask = np.arange(max_nodes)[None, :] < self.element_node_counts[:, None]
        self.representative_edge_length = float(np.median(lengths)) if lengths else 0.0
        # Beam contact segments: 2-node beams contribute one segment, 3-node
        # quadratic beams two (end-mid, mid-end).  Only used when
        # SphereContactConfig.beam_contact is enabled.
        segment_slots: List[Tuple[int, int]] = []
        segment_radii: List[float] = []
        segment_element_ids: List[int] = []
        for beam in beams:
            radius = _beam_contact_radius(beam)
            slots = [node_id_to_slot[int(node_id)] for node_id in beam.node_ids]
            if len(slots) == 2:
                pairs = [(slots[0], slots[1])]
            elif len(slots) == 3:
                pairs = [(slots[0], slots[1]), (slots[1], slots[2])]
            else:
                continue
            for pair in pairs:
                segment_slots.append(pair)
                segment_radii.append(radius)
                segment_element_ids.append(int(beam.element_id))
        self.beam_segment_slots = np.asarray(segment_slots, dtype=np.intp).reshape(-1, 2)
        self.beam_segment_radii = np.asarray(segment_radii, dtype=float)
        self.beam_segment_element_ids = np.asarray(segment_element_ids, dtype=np.int64)

    def deformed_node_positions(self, displacement: Optional[np.ndarray]) -> np.ndarray:
        if displacement is None:
            return self.node_base
        return self.node_base + np.asarray(displacement, dtype=float)[self.node_dofs]

    def element_centroids_and_radii(self, node_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-element deformed node blocks, centroids and bounding radii."""
        element_nodes = node_positions[self.element_node_slots]
        counts = np.maximum(self.element_node_counts, 1)
        sums = np.where(self.node_mask[..., None], element_nodes, 0.0).sum(axis=1)
        centroids = sums / counts[:, None]
        deltas = np.where(self.node_mask[..., None], element_nodes - centroids[:, None, :], 0.0)
        radii = np.sqrt((deltas**2).sum(axis=2)).max(axis=1)
        return element_nodes, centroids, radii

    def active_element_mask(self, deleted_element_ids: Sequence[int]) -> np.ndarray:
        if not deleted_element_ids:
            return np.ones(self.element_ids.shape[0], dtype=bool)
        deleted = np.asarray(sorted({int(element_id) for element_id in deleted_element_ids}), dtype=np.int64)
        return ~np.isin(self.element_ids, deleted)


def _contact_geometry(model: "FEModel") -> _ContactGeometry:
    mesh = model.mesh
    signature = mesh.revision_signature()
    cached = getattr(mesh, "_contact_geometry_cache", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    geometry = _ContactGeometry(model)
    mesh._contact_geometry_cache = (signature, geometry)
    return geometry


def _contact_patch_nodal_weights(
    geometry: _ContactGeometry,
    node_positions: np.ndarray,
    centroids: np.ndarray,
    radii: np.ndarray,
    active_mask: np.ndarray,
    contact_point: np.ndarray,
    contact_config: SphereContactConfig,
    sphere: RigidSphereImpact,
    penetration: float,
) -> Dict[int, float]:
    """Return smooth local nodal weights (keyed by node slot) for one contact force."""
    patch_radius = max(
        np.sqrt(max(float(sphere.radius) * float(penetration), 0.0)),
        float(contact_config.load_patch_radius_factor) * max(float(geometry.representative_edge_length), 1.0e-12),
        1.0e-12,
    )
    point = np.asarray(contact_point, dtype=float).reshape(3)
    near = active_mask & (np.linalg.norm(centroids - point[None, :], axis=1) <= patch_radius + radii)
    if not near.any():
        return {}
    slots = np.unique(geometry.element_node_slots[near][geometry.node_mask[near]])
    distances = np.linalg.norm(node_positions[slots] - point[None, :], axis=1)
    order = np.argsort(distances, kind="stable")
    slots = slots[order]
    distances = distances[order]
    within = int(np.searchsorted(distances, patch_radius, side="right"))
    min_nodes = min(int(contact_config.min_load_patch_nodes), slots.shape[0])
    selected_count = max(within, min_nodes)
    max_nodes = max(int(contact_config.max_load_patch_nodes), min_nodes)
    selected_count = min(selected_count, max_nodes)
    if selected_count <= 0:
        return {}
    slots = slots[:selected_count]
    selected_distances = distances[:selected_count]
    # Compactly supported kernel: the weight decays to zero at the selection
    # boundary, so a node entering or leaving the patch as the deformed
    # geometry changes does not discontinuously redistribute the contact load.
    # A hard-cutoff kernel (e.g. a truncated Gaussian) makes the contact
    # fixed-point map discontinuous and produces non-converging chatter cycles.
    support_radius = max(patch_radius, float(selected_distances[-1]) * (1.0 + 1.0e-6), 1.0e-12)
    ratios = np.minimum(selected_distances / support_radius, 1.0)
    raw = (1.0 - ratios * ratios) ** 2
    total = float(raw.sum())
    if total <= 0.0:
        raw = np.full(selected_count, 1.0 / selected_count)
        total = float(selected_count) * (1.0 / selected_count)
    weights = raw / total
    return {int(slot): float(weight) for slot, weight in zip(slots, weights)}


def _representative_shell_edge_length(model: "FEModel") -> float:
    return _contact_geometry(model).representative_edge_length


def _representative_shell_stiffness(model: "FEModel") -> float:
    """Median shell membrane stiffness scale ``E * t`` in N/m."""
    values: List[float] = []
    for element in _shell_contact_candidates(model):
        material = model.get_material(element.material_name)
        values.append(float(material.elastic_modulus) * float(element.thickness))
    return float(np.median(values)) if values else 0.0


def recommend_sphere_contact_penalty(
    model: "FEModel",
    sphere: RigidSphereImpact,
    target_penetration_fraction: float = 0.01,
    safety_factor: float = 1.0,
) -> float:
    """Recommend a conservative normal penalty stiffness [N/m] for v1 contact.

    Two dimensionally consistent stiffness scales are combined:

    - impact energy: ``k = m v^2 / delta^2`` stops the sphere's kinetic energy
      within the target penetration ``delta = R * target_penetration_fraction``
      (from ``1/2 k delta^2 = 1/2 m v^2``);
    - structural: the representative shell membrane stiffness ``E t`` [N/m]
      divided by the target penetration fraction, so the penalty spring is much
      stiffer than the contacted structure and the penetration stays a small
      fraction of the structural response.
    """

    if target_penetration_fraction <= 0.0:
        raise ValueError("target_penetration_fraction must be positive")
    if safety_factor <= 0.0:
        raise ValueError("safety_factor must be positive")
    target_penetration = max(float(sphere.radius) * float(target_penetration_fraction), 1.0e-12)
    impact_stiffness = float(sphere.mass) * max(float(sphere.speed), 1.0e-9) ** 2 / target_penetration**2
    shell_stiffness = _representative_shell_stiffness(model) / max(float(target_penetration_fraction), 1.0e-12)
    return float(safety_factor) * max(impact_stiffness, shell_stiffness, 1.0)


def _resolved_contact_config(model: "FEModel", sphere: RigidSphereImpact, config: Optional[SphereContactConfig]) -> SphereContactConfig:
    raw = config or SphereContactConfig()
    if raw.penalty_stiffness is not None:
        return raw
    return SphereContactConfig(
        penalty_stiffness=recommend_sphere_contact_penalty(
            model,
            sphere,
            target_penetration_fraction=raw.target_penetration_fraction,
            safety_factor=raw.auto_penalty_safety_factor,
        ),
        contact_damping=raw.contact_damping,
        search_margin=raw.search_margin,
        max_contact_iterations=raw.max_contact_iterations,
        penetration_tolerance=raw.penetration_tolerance,
        force_tolerance=raw.force_tolerance,
        side_mode=raw.side_mode,
        contact_surface=raw.contact_surface,
        signed_surface_offset=raw.signed_surface_offset,
        target_penetration_fraction=raw.target_penetration_fraction,
        auto_penalty_safety_factor=raw.auto_penalty_safety_factor,
        max_sphere_travel_fraction=raw.max_sphere_travel_fraction,
        max_event_substeps=raw.max_event_substeps,
        max_active_contacts=raw.max_active_contacts,
        load_patch_radius_factor=raw.load_patch_radius_factor,
        min_load_patch_nodes=raw.min_load_patch_nodes,
        max_load_patch_nodes=raw.max_load_patch_nodes,
        save_contact_history=raw.save_contact_history,
        post_separation_time=raw.post_separation_time,
        contact_relaxation=raw.contact_relaxation,
        beam_contact=raw.beam_contact,
    )


def _contact_issue(
    code: str,
    severity: str,
    entity_type: str,
    entity_id: Optional[int],
    message: str,
    measured: Optional[float] = None,
    limit: Optional[float] = None,
    suggestion: str = "",
) -> ProductionValidationIssue:
    return ProductionValidationIssue(code, severity, entity_type, entity_id, message, measured, limit, suggestion)


def validate_contact_configuration(
    model: "FEModel",
    sphere: RigidSphereImpact,
    contact_config: Optional[SphereContactConfig],
    transient_config: TransientConfig,
) -> ProductionValidationReport:
    """Validate limited rigid-sphere contact inputs before production use."""

    issues: List[ProductionValidationIssue] = []
    shells = _shell_contact_candidates(model)
    beam_targets = bool(contact_config is not None and contact_config.beam_contact) and bool(_beam_contact_candidates(model))
    if not shells and not beam_targets:
        issues.append(
            _contact_issue(
                "CONTACT001",
                "error",
                "model",
                None,
                "Sphere contact requires at least one shell target element.",
                suggestion="Add shell elements or use a different load/contact model.",
            )
        )
    used_materials = {element.material_name for element in model.mesh.elements.values() if hasattr(element, "material_name")}
    for name in sorted(used_materials):
        material = model.get_material(name)
        if float(material.density) <= 0.0:
            issues.append(
                _contact_issue(
                    "CONTACT002",
                    "error",
                    "material",
                    None,
                    f"Material {name!r} has non-positive density for transient contact.",
                    measured=float(material.density),
                    limit=0.0,
                    suggestion="Set density on all structural materials used by the impact model.",
                )
            )
    if contact_config is not None and contact_config.side_mode != "both":
        issues.append(
            _contact_issue(
                "CONTACT003",
                "error",
                "contact_config",
                None,
                "Only side_mode='both' is supported for limited sphere-shell contact.",
                suggestion="Use side_mode='both'.",
            )
        )
    config = _resolved_contact_config(model, sphere, contact_config)
    if transient_config.dt <= 0.0:
        issues.append(_contact_issue("CONTACT004", "error", "transient_config", None, "Transient dt must be positive."))
    else:
        travel = float(sphere.speed) * float(transient_config.dt)
        limit = 0.5 * float(sphere.radius)
        effective_travel = travel / max(int(config.max_event_substeps), 1)
        if effective_travel > limit:
            issues.append(
                _contact_issue(
                    "CONTACT005",
                    "error",
                    "transient_config",
                    None,
                    "Sphere travel per time step is too large for production contact without event substepping.",
                    measured=effective_travel,
                    limit=limit,
                    suggestion="Reduce dt or increase event substepping controls.",
                )
            )
    period_dt = 0.2 * np.sqrt(float(sphere.mass) / float(config.penalty_stiffness))
    if transient_config.dt > period_dt:
        issues.append(
            _contact_issue(
                "CONTACT006",
                "warning",
                "transient_config",
                None,
                "Time step is large relative to the contact penalty period.",
                measured=float(transient_config.dt),
                limit=float(period_dt),
                suggestion="Reduce dt, lower penalty stiffness, or accept lower contact-resolution accuracy.",
            )
        )
    mesh_quality = {
        "shell_contact_targets": len(shells),
        "representative_shell_edge_length": _representative_shell_edge_length(model),
        "recommended_penalty_stiffness": float(config.penalty_stiffness),
        "sphere_travel_per_step": float(sphere.speed) * float(transient_config.dt),
        "sphere_travel_radius_ratio": float(sphere.speed) * float(transient_config.dt) / max(float(sphere.radius), 1.0e-12),
    }
    status = "invalid" if any(issue.severity == "error" for issue in issues) else ("warning" if issues else "ok")
    return ProductionValidationReport(status, tuple(issues), mesh_quality, model.mesh.revision_signature())


def assemble_sphere_contact_load_vector(
    model: "FEModel",
    sphere: RigidSphereImpact,
    contact_config: SphereContactConfig,
    sphere_position: np.ndarray,
    sphere_velocity: np.ndarray,
    structural_displacement: Optional[np.ndarray] = None,
    structural_velocity: Optional[np.ndarray] = None,
    deleted_element_ids: Sequence[int] = (),
    contact_scale_by_element: Optional[Mapping[int, float]] = None,
    preferred_element_ids: Sequence[int] = (),
) -> Tuple[np.ndarray, np.ndarray, Tuple[SphereContactRecord, ...]]:
    """Assemble shell nodal loads from current rigid-sphere contact state.

    ``preferred_element_ids`` stabilizes the ``max_active_contacts`` reduction:
    adjacent coplanar elements report near-identical penetrations for one
    physical contact, and a bare deepest-penetration argmax then flips between
    them from iteration to iteration, which makes the contact fixed-point map
    discontinuous and can lock the iteration into a non-converging chatter
    cycle.  Within a small penetration tie band the previously selected
    elements are kept instead.
    """

    total_dofs = model.mesh.dof_manager.total_dofs
    u = None if structural_displacement is None else np.asarray(structural_displacement, dtype=float)
    v = np.zeros(total_dofs, dtype=float) if structural_velocity is None else np.asarray(structural_velocity, dtype=float)
    center = np.asarray(sphere_position, dtype=float).reshape(3)
    velocity = np.asarray(sphere_velocity, dtype=float).reshape(3)
    load = np.zeros(total_dofs, dtype=float)
    sphere_force_total = np.zeros(3, dtype=float)
    records: List[SphereContactRecord] = []
    contact_scales = {} if contact_scale_by_element is None else {int(k): float(v) for k, v in contact_scale_by_element.items()}
    geometry = _contact_geometry(model)
    has_beam_targets = bool(contact_config.beam_contact) and geometry.beam_segment_slots.shape[0] > 0
    if geometry.element_ids.shape[0] == 0 and not has_beam_targets:
        return load, sphere_force_total, tuple(records)
    node_positions = geometry.deformed_node_positions(u)
    element_nodes, centroids, radii = geometry.element_centroids_and_radii(node_positions)
    active_mask = geometry.active_element_mask(deleted_element_ids)
    if geometry.element_ids.shape[0] == 0:
        near_mask = np.zeros(0, dtype=bool)
    else:
        near_mask = active_mask & (
            np.linalg.norm(centroids - center[None, :], axis=1) <= sphere.radius + radii + contact_config.search_margin
        )

    for element_index in np.flatnonzero(near_mask):
        element = geometry.elements[element_index]
        contact_scale = float(contact_scales.get(int(element.element_id), 1.0))
        if contact_scale <= 0.0:
            continue
        node_count = int(geometry.element_node_counts[element_index])
        element_slots = geometry.element_node_slots[element_index, :node_count]
        coords = element_nodes[element_index, :node_count]
        local, N, dN_dxi, dN_deta, surface_point = _project_local_coordinates(element, coords, center)
        gap_vector = center - surface_point
        distance = float(np.linalg.norm(gap_vector))
        if distance > sphere.radius + contact_config.search_margin:
            continue
        tangent_xi = dN_dxi @ coords
        tangent_eta = dN_deta @ coords
        fallback_normal = np.cross(tangent_xi, tangent_eta)
        fallback_norm = float(np.linalg.norm(fallback_normal))
        if fallback_norm <= 1.0e-14:
            continue
        surface_normal = fallback_normal / fallback_norm
        offset = _surface_offset(element, contact_config)
        if offset != 0.0:
            surface_point = surface_point + offset * surface_normal
            gap_vector = center - surface_point
            distance = float(np.linalg.norm(gap_vector))
            if distance > sphere.radius + contact_config.search_margin:
                continue
        if distance > 1.0e-14:
            normal = gap_vector / distance
        elif fallback_norm > 1.0e-14:
            normal = surface_normal
        else:
            continue
        penetration = max(float(sphere.radius - distance), 0.0)
        if penetration <= 0.0:
            continue
        surface_velocity = np.asarray(N, dtype=float) @ v[geometry.node_dofs[element_slots]]
        relative_normal_velocity = float(np.dot(velocity - surface_velocity, normal))
        normal_force = max(
            float(contact_config.penalty_stiffness) * penetration - float(contact_config.contact_damping) * relative_normal_velocity,
            0.0,
        )
        normal_force *= min(max(contact_scale, 0.0), 1.0)
        if normal_force <= 0.0:
            continue
        sphere_force = normal_force * normal
        structure_force = -sphere_force
        patch_weights = _contact_patch_nodal_weights(
            geometry,
            node_positions,
            centroids,
            radii,
            active_mask,
            surface_point,
            contact_config,
            sphere,
            penetration,
        )
        if not patch_weights:
            patch_weights = {int(slot): float(N[local_index]) for local_index, slot in enumerate(element_slots)}
        nodal_forces: Dict[int, np.ndarray] = {}
        for slot, weight in patch_weights.items():
            nodal = float(weight) * structure_force
            load[geometry.node_dofs[slot]] += nodal
            nodal_forces[int(geometry.node_ids[slot])] = nodal
        sphere_force_total += sphere_force
        records.append(
            SphereContactRecord(
                element_id=int(element.element_id),
                local_coordinates=(float(local[0]), float(local[1])),
                contact_point=surface_point,
                normal=normal,
                penetration=penetration,
                normal_force=normal_force,
                sphere_force=sphere_force,
                structure_force=structure_force,
                contact_classification=_contact_classification(element, local),
                nodal_forces=nodal_forces,
            )
        )

    if bool(contact_config.beam_contact) and geometry.beam_segment_slots.shape[0]:
        deleted = {int(element_id) for element_id in deleted_element_ids}
        seg_a = node_positions[geometry.beam_segment_slots[:, 0]]
        seg_b = node_positions[geometry.beam_segment_slots[:, 1]]
        axis = seg_b - seg_a
        length_sq = np.maximum(np.einsum("ij,ij->i", axis, axis), 1.0e-30)
        t_param = np.clip(np.einsum("ij,ij->i", center[None, :] - seg_a, axis) / length_sq, 0.0, 1.0)
        closest = seg_a + t_param[:, None] * axis
        gap_vectors = center[None, :] - closest
        distances = np.linalg.norm(gap_vectors, axis=1)
        reach = sphere.radius + geometry.beam_segment_radii + contact_config.search_margin
        for segment_index in np.flatnonzero(distances <= reach):
            beam_id = int(geometry.beam_segment_element_ids[segment_index])
            if beam_id in deleted:
                continue
            contact_scale = float(contact_scales.get(beam_id, 1.0))
            if contact_scale <= 0.0:
                continue
            distance = float(distances[segment_index])
            penetration = float(sphere.radius + geometry.beam_segment_radii[segment_index] - distance)
            if penetration <= 0.0 or distance <= 1.0e-14:
                continue
            normal = gap_vectors[segment_index] / distance
            t_value = float(t_param[segment_index])
            slot_a = int(geometry.beam_segment_slots[segment_index, 0])
            slot_b = int(geometry.beam_segment_slots[segment_index, 1])
            surface_velocity = (1.0 - t_value) * v[geometry.node_dofs[slot_a]] + t_value * v[geometry.node_dofs[slot_b]]
            relative_normal_velocity = float(np.dot(velocity - surface_velocity, normal))
            normal_force = max(
                float(contact_config.penalty_stiffness) * penetration
                - float(contact_config.contact_damping) * relative_normal_velocity,
                0.0,
            )
            normal_force *= min(max(contact_scale, 0.0), 1.0)
            if normal_force <= 0.0:
                continue
            sphere_force = normal_force * normal
            structure_force = -sphere_force
            nodal_forces = {
                int(geometry.node_ids[slot_a]): (1.0 - t_value) * structure_force,
                int(geometry.node_ids[slot_b]): t_value * structure_force,
            }
            load[geometry.node_dofs[slot_a]] += nodal_forces[int(geometry.node_ids[slot_a])]
            load[geometry.node_dofs[slot_b]] += nodal_forces[int(geometry.node_ids[slot_b])]
            sphere_force_total += sphere_force
            records.append(
                SphereContactRecord(
                    element_id=beam_id,
                    local_coordinates=(2.0 * t_value - 1.0, 0.0),
                    contact_point=closest[segment_index].copy(),
                    normal=normal.copy(),
                    penetration=penetration,
                    normal_force=normal_force,
                    sphere_force=sphere_force,
                    structure_force=structure_force,
                    contact_classification="beam",
                    nodal_forces=nodal_forces,
                )
            )

    if len(records) > int(contact_config.max_active_contacts):
        deepest = max(record.penetration for record in records)
        tie_band = 0.95 * deepest
        preferred = {int(element_id) for element_id in preferred_element_ids}
        records = sorted(
            records,
            key=lambda item: (
                item.penetration >= tie_band and int(item.element_id) in preferred,
                item.penetration,
                item.normal_force,
            ),
            reverse=True,
        )[: int(contact_config.max_active_contacts)]
        load = np.zeros(total_dofs, dtype=float)
        sphere_force_total = np.zeros(3, dtype=float)
        for record in records:
            sphere_force_total += record.sphere_force
            for node_id, nodal in record.nodal_forces.items():
                load[geometry.node_dofs[geometry.node_id_to_slot[int(node_id)]]] += nodal

    return load, sphere_force_total, tuple(records)


def _contact_default_config(sphere: RigidSphereImpact) -> SphereContactConfig:
    stiffness = max(float(sphere.mass) * max(float(sphere.speed), 1.0) ** 2 / max(float(sphere.radius) * 0.01, 1.0e-9), 1.0)
    return SphereContactConfig(penalty_stiffness=stiffness)


LinearElementMatrixTerms = Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _contact_erodible_element_count(model: "FEModel") -> int:
    """Count elements that impact fracture can erode (shell midsurfaces + beams)."""
    return sum(1 for element in model.mesh.elements.values() if element_fracture_category(element) is not None)


def _linear_element_matrix_terms(model: "FEModel") -> Tuple[int, Tuple[LinearElementMatrixTerms, ...]]:
    """Cache unscaled element K/M contributions for repeated damage rescaling."""
    terms: List[LinearElementMatrixTerms] = []
    total_dofs = model.mesh.dof_manager.total_dofs
    for element_id, element in model.mesh.elements.items():
        if not hasattr(element, "get_dof_mapping"):
            continue
        mapping = np.asarray(element.get_dof_mapping(model.mesh), dtype=np.intp).reshape(-1)
        if mapping.size == 0:
            continue
        material = model.get_material(element.material_name)
        k_elem = np.asarray(element.compute_stiffness_matrix(model.mesh, material), dtype=float)
        m_elem = np.asarray(element.compute_mass_matrix(model.mesh, material), dtype=float)
        if k_elem.shape != (mapping.size, mapping.size) or m_elem.shape != (mapping.size, mapping.size):
            continue
        row = np.repeat(mapping, mapping.size)
        col = np.tile(mapping, mapping.size)
        terms.append((int(element_id), row, col, k_elem.reshape(-1), m_elem.reshape(-1)))
    return total_dofs, tuple(terms)


def _assemble_damaged_linear_matrices(
    model: "FEModel",
    element_scales: Mapping[int, float],
    cached_terms: Optional[Tuple[int, Tuple[LinearElementMatrixTerms, ...]]] = None,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Assemble K and M with per-element damage stiffness/mass scales."""
    if cached_terms is None:
        cached_terms = _linear_element_matrix_terms(model)
    total_dofs, terms = cached_terms
    scales = {int(element_id): float(scale) for element_id, scale in element_scales.items()}
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    k_data: List[np.ndarray] = []
    m_data: List[np.ndarray] = []
    for element_id, row, col, k_flat, m_flat in terms:
        scale = min(max(float(scales.get(int(element_id), 1.0)), 0.0), 1.0)
        rows.append(row)
        cols.append(col)
        k_data.append(scale * k_flat)
        m_data.append(scale * m_flat)
    if not rows:
        empty = sparse.csr_matrix((total_dofs, total_dofs), dtype=float)
        return empty, empty
    row_all = np.concatenate(rows)
    col_all = np.concatenate(cols)
    K = sparse.coo_matrix((np.concatenate(k_data), (row_all, col_all)), shape=(total_dofs, total_dofs)).tocsr()
    M = sparse.coo_matrix((np.concatenate(m_data), (row_all, col_all)), shape=(total_dofs, total_dofs)).tocsr()
    return K, M


def _assemble_eroded_linear_matrices(
    model: "FEModel",
    deleted_element_ids: Sequence[int],
    residual_fraction: float,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Assemble K and M with residual contribution from deleted elements."""
    scales = {int(element_id): float(residual_fraction) for element_id in deleted_element_ids}
    return _assemble_damaged_linear_matrices(model, scales)


def _impact_contact_patch_area(
    record: SphereContactRecord,
    element: ShellElement,
    config: ImpactDamageConfig,
    sphere: RigidSphereImpact,
) -> float:
    """Estimate the local shell contact patch area used by damage checks."""
    penetration_radius = np.sqrt(max(float(sphere.radius) * float(record.penetration), 0.0))
    fraction_radius = max(float(sphere.radius) * float(config.contact_area_radius_fraction), 0.0)
    thickness_radius = max(float(getattr(element, "thickness", 0.0)), 0.0)
    min_radius = np.sqrt(float(config.min_contact_area) / np.pi)
    patch_radius = max(penetration_radius, fraction_radius, thickness_radius, min_radius)
    return max(float(np.pi * patch_radius**2), float(config.min_contact_area))


def _impact_material_capacity(model: "FEModel", element: ShellElement, config: ImpactDamageConfig) -> Tuple[float, str]:
    """Return yield/ultimate/user stress capacity in Pa for impact damage."""
    if config.capacity_basis == "user":
        return float(config.user_capacity), "user"
    material = model.get_material(element.material_name)
    curve = getattr(material, "hardening_curve", None)
    if config.capacity_basis == "yield":
        if curve is not None and hasattr(curve, "sigma_yield"):
            return float(curve.sigma_yield), "hardening_curve.sigma_yield"
        if float(getattr(material, "yield_stress", 0.0)) > 0.0:
            return float(material.yield_stress), "material.yield_stress"
    else:
        if curve is not None:
            candidates = []
            for name in ("sigma_yield_2", "sigma_yield"):
                value = getattr(curve, name, None)
                if value is not None:
                    candidates.append(float(value))
            if hasattr(curve, "flow_stress"):
                try:
                    candidates.append(float(np.asarray(curve.flow_stress(np.asarray([0.05], dtype=float))).reshape(-1)[0]))
                except Exception:
                    pass
            if candidates:
                return max(candidates), "hardening_curve.ultimate_proxy"
        if float(getattr(material, "yield_stress", 0.0)) > 0.0:
            return 1.15 * float(material.yield_stress), "material.yield_stress_ultimate_proxy"
    elastic_proxy = max(0.002 * float(getattr(material, "elastic_modulus", 0.0)), 1.0)
    return elastic_proxy, "elastic_0.2pct_proxy"


def _impact_damage_metrics(
    model: "FEModel",
    record: SphereContactRecord,
    config: ImpactDamageConfig,
    sphere: RigidSphereImpact,
    dt: float,
) -> Dict[str, Any]:
    element = model.mesh.get_element(int(record.element_id))
    if not isinstance(element, ShellElement):
        return {}
    area = _impact_contact_patch_area(record, element, config, sphere)
    capacity, capacity_source = _impact_material_capacity(model, element, config)
    pressure = float(record.normal_force) / max(area, 1.0e-30)
    impulse_density = float(record.normal_force) * float(dt) / max(area, 1.0e-30)
    pressure_utilization = pressure / max(capacity, 1.0e-30)
    impulse_utilization = impulse_density / max(capacity * float(config.impulse_reference_time), 1.0e-30)
    material = model.get_material(element.material_name)
    elastic_modulus = max(float(getattr(material, "elastic_modulus", 0.0)), 1.0)
    equivalent_plastic_strain_estimate = max(pressure - capacity, 0.0) / elastic_modulus * float(config.strain_scale)
    strain_utilization = equivalent_plastic_strain_estimate / max(float(config.plastic_strain_capacity), 1.0e-30)
    components = {
        "contact_pressure": float(pressure_utilization),
        "impulse_per_area": float(impulse_utilization),
        "equivalent_plastic_strain_estimate": float(strain_utilization),
    }
    governing_component = max(components, key=components.get)
    combined = max(float(value) for value in components.values())
    return {
        "element_id": int(record.element_id),
        "contact_patch_area": float(area),
        "contact_pressure": float(pressure),
        "impulse_density_increment": float(impulse_density),
        "capacity": float(capacity),
        "capacity_source": capacity_source,
        "equivalent_plastic_strain_estimate": float(equivalent_plastic_strain_estimate),
        "utilizations": components,
        "combined_utilization": float(combined),
        "governing_component": governing_component,
    }


def _impact_damage_scale(damage: float, config: ImpactDamageConfig) -> float:
    if damage < float(config.softening_start):
        return 1.0
    if damage >= float(config.delete_at):
        return float(config.residual_stiffness_fraction)
    span = max(float(config.delete_at) - float(config.softening_start), 1.0e-12)
    fraction = (float(damage) - float(config.softening_start)) / span
    return float(1.0 - fraction * (1.0 - float(config.residual_stiffness_fraction)))


def _update_impact_damage_states(
    model: "FEModel",
    records: Sequence[SphereContactRecord],
    config: ImpactDamageConfig,
    sphere: RigidSphereImpact,
    states: Dict[int, Dict[str, Any]],
    deleted_element_ids: Sequence[int],
    *,
    step_index: int,
    time_value: float,
    dt: float,
) -> Tuple[Tuple[DeletedElementRecord, ...], float, Tuple[Dict[str, Any], ...], bool]:
    deleted = {int(element_id) for element_id in deleted_element_ids}
    new_deleted: List[DeletedElementRecord] = []
    diagnostics: List[Dict[str, Any]] = []
    changed = False
    max_utilization = 0.0
    for record in records:
        element_id = int(record.element_id)
        if element_id in deleted:
            continue
        element = model.mesh.get_element(element_id)
        # Capacity-based impact damage is an area/contact-pressure screening
        # model, which maps to shell midsurface contact only.  Beam line
        # contact (force per length) is out of scope here; use
        # ImpactFractureConfig (contact force/penetration) or the nonlinear
        # PlasticImpactDamageConfig path for beam erosion.
        if element is None or element_fracture_category(element) != "shell":
            continue
        metrics = _impact_damage_metrics(model, record, config, sphere, dt)
        if not metrics:
            continue
        utilization = float(metrics["combined_utilization"])
        max_utilization = max(max_utilization, utilization)
        state = states.setdefault(
            element_id,
            {
                "damage": 0.0,
                "max_utilization": 0.0,
                "governing_component": "",
                "accumulated_impulse_density": 0.0,
                "max_contact_patch_area": 0.0,
                "contact_count": 0,
                "history": [],
                "deleted_time": None,
            },
        )
        old_damage = float(state["damage"])
        old_scale = float(state.get("scale", _impact_damage_scale(old_damage, config)))
        state["contact_count"] = int(state.get("contact_count", 0)) + 1
        state["max_utilization"] = max(float(state.get("max_utilization", 0.0)), utilization)
        if utilization >= float(state.get("max_utilization", 0.0)) - 1.0e-12:
            state["governing_component"] = str(metrics["governing_component"])
        state["accumulated_impulse_density"] = float(state.get("accumulated_impulse_density", 0.0)) + float(
            metrics["impulse_density_increment"]
        )
        state["max_contact_patch_area"] = max(float(state.get("max_contact_patch_area", 0.0)), float(metrics["contact_patch_area"]))
        if config.mode == "instant_threshold":
            new_damage = max(old_damage, utilization / float(config.damage_threshold))
        else:
            increment = max(utilization / float(config.damage_threshold), 0.0) * float(dt) / float(config.impulse_reference_time)
            new_damage = old_damage + increment
        state["damage"] = float(new_damage)
        scale = _impact_damage_scale(new_damage, config)
        state["scale"] = float(scale)
        diag = {
            "element_id": element_id,
            "time": float(time_value),
            "damage": float(new_damage),
            "scale": float(scale),
            **metrics,
        }
        diagnostics.append(diag)
        if bool(config.record_history):
            history = state.setdefault("history", [])
            history.append(diag)
        if abs(scale - old_scale) > 1.0e-12:
            changed = True
        deletion_allowed = True
        if bool(config.neighbor_smoothing) and int(state.get("contact_count", 0)) < 2:
            deletion_allowed = False
            diag["neighbor_smoothing_hold"] = True
        if new_damage >= float(config.delete_at) and deletion_allowed:
            state["deleted_time"] = float(time_value)
            new_deleted.append(
                DeletedElementRecord(
                    element_id=element_id,
                    element_type="shell",
                    step_index=int(step_index),
                    load_factor=float(time_value),
                    trigger_name=f"impact_damage:{metrics['governing_component']}",
                    trigger_value=float(new_damage),
                    threshold=float(config.delete_at),
                    location=f"contact:{record.contact_classification}",
                    measure=element_measure(model.mesh, element),
                )
            )
    return tuple(new_deleted), max_utilization, tuple(diagnostics), changed


def _impact_damage_summary(
    model: "FEModel",
    config: Optional[ImpactDamageConfig],
    states: Mapping[int, Mapping[str, Any]],
    deleted_element_ids: Sequence[int],
    *,
    deletion_records: Sequence[DeletedElementRecord] = (),
    warnings: Sequence[str] = (),
) -> Dict[str, Any]:
    if config is None:
        return {"enabled": False, "records": [], "warnings": list(warnings)}
    deleted = {int(element_id) for element_id in deleted_element_ids}
    shell_count = sum(1 for element in model.mesh.elements.values() if element_fracture_category(element) == "shell")
    active_states = {int(element_id): state for element_id, state in states.items()}
    softened = [
        element_id
        for element_id, state in active_states.items()
        if float(state.get("damage", 0.0)) >= float(config.softening_start) and element_id not in deleted
    ]
    records = []
    for element_id, state in sorted(active_states.items()):
        history = state.get("history", []) if bool(config.record_history) else []
        records.append(
            {
                "element_id": int(element_id),
                "damage": float(state.get("damage", 0.0)),
                "scale": float(state.get("scale", _impact_damage_scale(float(state.get("damage", 0.0)), config))),
                "max_utilization": float(state.get("max_utilization", 0.0)),
                "governing_component": str(state.get("governing_component", "")),
                "accumulated_impulse_density": float(state.get("accumulated_impulse_density", 0.0)),
                "max_contact_patch_area": float(state.get("max_contact_patch_area", 0.0)),
                "contact_count": int(state.get("contact_count", 0)),
                "deleted_time": state.get("deleted_time"),
                "history": history,
            }
        )
    return {
        "enabled": True,
        "config": config.to_dict(),
        "deleted_count": len(deleted),
        "softened_count": len(softened),
        "deleted_fraction": float(len(deleted) / max(shell_count, 1)),
        "deleted_element_ids": sorted(deleted),
        "softened_element_ids": sorted(softened),
        "max_damage": max((float(state.get("damage", 0.0)) for state in active_states.values()), default=0.0),
        "max_utilization": max((float(state.get("max_utilization", 0.0)) for state in active_states.values()), default=0.0),
        "records": records,
        "deletion_records": [record.to_dict() for record in deletion_records],
        "warnings": list(warnings),
    }


def _impact_damage_preflight_warnings(
    model: "FEModel",
    config: Optional[ImpactDamageConfig],
    sphere: RigidSphereImpact,
    *,
    fracture_config: Optional[ImpactFractureConfig] = None,
    beam_contact: bool = False,
) -> Tuple[str, ...]:
    if config is None:
        return ()
    warnings: List[str] = []
    if fracture_config is not None:
        warnings.append(
            "IMPACT_DAMAGE010: both ImpactFractureConfig and ImpactDamageConfig are active; "
            "either path may erode shell elements, and summaries report their own trigger ownership."
        )
    if beam_contact and _beam_contact_candidates(model):
        warnings.append(
            "IMPACT_DAMAGE014: beam contact targets are active but capacity-based ImpactDamageConfig "
            "screens shell midsurface contact only; beam erosion needs ImpactFractureConfig "
            "(contact force/penetration) or the nonlinear PlasticImpactDamageConfig path."
        )
    shell_elements = [element for element in model.mesh.elements.values() if element_fracture_category(element) == "shell"]
    if not shell_elements:
        return tuple(warnings)
    fallback_capacity = False
    for element in shell_elements:
        if not isinstance(element, ShellElement):
            continue
        _capacity, source = _impact_material_capacity(model, element, config)
        if source == "elastic_0.2pct_proxy":
            fallback_capacity = True
            break
    if fallback_capacity and config.capacity_basis != "user":
        warnings.append(
            "IMPACT_DAMAGE011: one or more shell materials lack yield stress or RP-C208 hardening curve; "
            "impact damage capacity is using a 0.2% elastic proxy."
        )
    representative_edge = _representative_shell_edge_length(model)
    if representative_edge > 0.0:
        representative_area = representative_edge**2
        if float(config.min_contact_area) < 1.0e-5 * representative_area:
            warnings.append(
                "IMPACT_DAMAGE012: min_contact_area is very small relative to representative shell area; "
                "damage may be mesh sensitive unless calibrated."
            )
    if float(config.contact_area_radius_fraction) * float(sphere.radius) < 0.05 * max(representative_edge, 1.0e-12):
        warnings.append(
            "IMPACT_DAMAGE013: contact patch radius fraction is small relative to mesh edge length; "
            "consider a larger minimum area or finer mesh for production screening."
        )
    return tuple(warnings)


def _warnings_with_prefixes(warnings: Sequence[str], prefixes: Sequence[str]) -> Tuple[str, ...]:
    prefix_tuple = tuple(str(prefix) for prefix in prefixes)
    return tuple(str(warning) for warning in warnings if str(warning).startswith(prefix_tuple))


def _impact_fracture_trigger_value(
    record: SphereContactRecord,
    config: ImpactFractureConfig,
    sphere: RigidSphereImpact,
) -> float:
    if config.trigger == "contact_force":
        return float(record.normal_force)
    if config.trigger == "penetration_ratio":
        return float(record.penetration) / max(float(sphere.radius), 1.0e-12)
    contact_radius = max(float(sphere.radius) * float(config.contact_area_radius_fraction), 1.0e-12)
    area = np.pi * contact_radius**2
    return float(record.normal_force) / area


def _detect_impact_fracture_records(
    model: "FEModel",
    records: Sequence[SphereContactRecord],
    config: ImpactFractureConfig,
    sphere: RigidSphereImpact,
    deleted_element_ids: Sequence[int],
    *,
    step_index: int,
    time_value: float,
) -> Tuple[Tuple[DeletedElementRecord, ...], float]:
    if float(time_value) + 1.0e-12 < config.min_time:
        return (), 0.0
    deleted = {int(element_id) for element_id in deleted_element_ids}
    new_records: List[DeletedElementRecord] = []
    max_utilization = 0.0
    for contact_record in records:
        element_id = int(contact_record.element_id)
        if element_id in deleted:
            continue
        element = model.mesh.get_element(element_id)
        if element is None:
            continue
        category = element_fracture_category(element)
        # Contact-observable fracture triggers (force, penetration ratio,
        # sphere-area pressure proxy) are geometry-agnostic, so any erodible
        # contact target -- shell midsurface or beam segment -- can fracture.
        if category is None:
            continue
        trigger_value = _impact_fracture_trigger_value(contact_record, config, sphere)
        utilization = trigger_value / config.threshold
        max_utilization = max(max_utilization, float(utilization))
        if trigger_value >= config.threshold:
            new_records.append(
                DeletedElementRecord(
                    element_id=element_id,
                    element_type=category,
                    step_index=int(step_index),
                    load_factor=float(time_value),
                    trigger_name=str(config.trigger),
                    trigger_value=float(trigger_value),
                    threshold=float(config.threshold),
                    location=f"contact:{contact_record.contact_classification}",
                    measure=element_measure(model.mesh, element),
                )
            )
    return tuple(new_records), max_utilization


def _plastic_damage_scale(utilization: float, config: PlasticImpactDamageConfig) -> float:
    value = max(float(utilization), 0.0)
    if value < float(config.softening_start):
        return 1.0
    if value >= float(config.delete_at):
        return float(config.residual_stiffness_fraction)
    span = max(float(config.delete_at) - float(config.softening_start), 1.0e-12)
    fraction = (value - float(config.softening_start)) / span
    return 1.0 - fraction * (1.0 - float(config.residual_stiffness_fraction))


def _plastic_impact_damage_update(
    model: "FEModel",
    committed_states: Mapping[int, Any],
    config: Optional[PlasticImpactDamageConfig],
    deleted_element_ids: set[int],
    damage_states: Dict[int, Dict[str, Any]],
    *,
    step_index: int,
    time_value: float,
) -> Tuple[List[DeletedElementRecord], bool, float]:
    if config is None:
        return [], False, 0.0
    changed = False
    new_records: List[DeletedElementRecord] = []
    max_utilization = 0.0
    for element_id, state in committed_states.items():
        element = model.mesh.elements.get(int(element_id))
        if element is None:
            continue
        category = element_fracture_category(element)
        if category not in set(config.element_scope):
            continue
        trigger_value, location = state_equivalent_plastic_strain(state)
        utilization = float(trigger_value) / max(float(config.threshold), 1.0e-30)
        max_utilization = max(max_utilization, utilization)
        previous = damage_states.get(int(element_id), {})
        previous_damage = float(previous.get("damage", 0.0) or 0.0)
        damage = max(previous_damage, utilization)
        scale = _plastic_damage_scale(damage, config)
        history = list(previous.get("history", [])) if bool(config.record_history) else []
        if bool(config.record_history):
            history.append(
                {
                    "time": float(time_value),
                    "step_index": int(step_index),
                    "equivalent_plastic_strain": float(trigger_value),
                    "utilization": float(utilization),
                    "damage": float(damage),
                    "scale": float(scale),
                    "location": str(location),
                }
            )
        if damage > previous_damage + 1.0e-15 or abs(scale - float(previous.get("scale", 1.0) or 1.0)) > 1.0e-15:
            changed = True
        damage_states[int(element_id)] = {
            "element_id": int(element_id),
            "damage": float(damage),
            "scale": float(scale),
            "max_equivalent_plastic_strain": max(float(previous.get("max_equivalent_plastic_strain", 0.0) or 0.0), float(trigger_value)),
            "max_utilization": max(float(previous.get("max_utilization", 0.0) or 0.0), float(utilization)),
            "location": str(location),
            "history": history,
        }
        if int(element_id) not in deleted_element_ids and damage >= float(config.delete_at) - 1.0e-15:
            record = DeletedElementRecord(
                element_id=int(element_id),
                element_type=type(element).__name__,
                step_index=int(step_index),
                load_factor=float(time_value),
                trigger_name="max_equivalent_plastic_strain",
                trigger_value=float(trigger_value),
                threshold=float(config.threshold),
                location=str(location),
                measure=element_measure(model.mesh, element),
            )
            new_records.append(record)
            deleted_element_ids.add(int(element_id))
            changed = True
    return new_records, changed, max_utilization


def _plastic_impact_damage_summary(
    model: "FEModel",
    config: Optional[PlasticImpactDamageConfig],
    damage_states: Mapping[int, Mapping[str, Any]],
    deleted_element_ids: Sequence[int],
    deletion_records: Sequence[DeletedElementRecord],
    *,
    warnings: Sequence[str] = (),
) -> Dict[str, Any]:
    if config is None:
        return {"enabled": False, "records": [], "warnings": list(warnings)}
    scope = set(config.element_scope)
    scoped_count = sum(
        1
        for element in model.mesh.elements.values()
        if element_fracture_category(element) in scope
    )
    deleted = sorted(int(element_id) for element_id in deleted_element_ids)
    records = []
    for element_id, state in sorted((int(k), v) for k, v in damage_states.items()):
        records.append(
            {
                "element_id": int(element_id),
                "damage": float(state.get("damage", 0.0) or 0.0),
                "scale": float(state.get("scale", 1.0) or 1.0),
                "max_equivalent_plastic_strain": float(state.get("max_equivalent_plastic_strain", 0.0) or 0.0),
                "max_utilization": float(state.get("max_utilization", 0.0) or 0.0),
                "location": str(state.get("location", "")),
                "history": list(state.get("history", []) or []),
            }
        )
    return {
        "enabled": True,
        "mode": "material_nonlinear_plastic_strain",
        "config": config.to_dict(),
        "deleted_count": len(deleted),
        "deleted_fraction": float(len(deleted) / max(scoped_count, 1)),
        "deleted_element_ids": deleted,
        "softened_element_ids": sorted(
            int(element_id)
            for element_id, state in damage_states.items()
            if int(element_id) not in set(deleted) and float(state.get("scale", 1.0) or 1.0) < 1.0
        ),
        "max_damage": max((float(state.get("damage", 0.0) or 0.0) for state in damage_states.values()), default=0.0),
        "max_utilization": max((float(state.get("max_utilization", 0.0) or 0.0) for state in damage_states.values()), default=0.0),
        "max_equivalent_plastic_strain": max(
            (float(state.get("max_equivalent_plastic_strain", 0.0) or 0.0) for state in damage_states.values()),
            default=0.0,
        ),
        "records": records,
        "deletion_records": [record.to_dict() for record in deletion_records],
        "warnings": list(warnings),
    }


def _solve_transient_sphere_impact_nonlinear(
    model: "FEModel",
    transient_config: TransientConfig,
    sphere: RigidSphereImpact,
    config: SphereContactConfig,
    validation: ProductionValidationReport,
    *,
    base_load_case: Optional[LoadCase] = None,
    nonlinear_config: Optional[NonlinearTransientConfig] = None,
    plastic_damage_config: Optional[PlasticImpactDamageConfig] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> SphereImpactResult:
    """Implicit Newmark nonlinear impact with material/geometric element response."""
    from .nonlinear_static import _assemble_nonlinear_system, _nonlinear_state_summary

    def assemble_nonlinear(
        displacements: np.ndarray,
        states: Dict[int, Any],
        *,
        tangent: bool,
        scales: Optional[Mapping[int, float]] = None,
    ) -> Tuple[np.ndarray, Any, Dict[int, Any]]:
        kwargs = {
            "deleted_element_ids": tuple(deleted_element_ids),
            "residual_stiffness_fraction": float(
                plastic_damage_config.residual_stiffness_fraction if plastic_damage_config is not None else 1.0
            ),
            "kinematics": str(nl.kinematics),
        }
        if scales:
            kwargs["element_stiffness_scales"] = scales
        try:
            return _assemble_nonlinear_system(
                model,
                displacements,
                states,
                int(nl.num_layers),
                tangent=tangent,
                **kwargs,
            )
        except TypeError as exc:
            if "element_stiffness_scales" not in str(exc):
                raise
            kwargs.pop("element_stiffness_scales", None)
            return _assemble_nonlinear_system(
                model,
                displacements,
                states,
                int(nl.num_layers),
                tangent=tangent,
                **kwargs,
            )

    nl = nonlinear_config or NonlinearTransientConfig(enabled=True)
    model.apply_boundary_conditions()
    total_dofs = model.mesh.dof_manager.total_dofs
    needs_linear_stiffness = abs(float(transient_config.rayleigh_beta)) > 0.0
    if needs_linear_stiffness:
        K0, stiffness_info = assemble_stiffness_matrix(model)
    else:
        K0 = sparse.eye(total_dofs, format="csr", dtype=float)
        stiffness_info = {
            "assembly_skipped": True,
            "reason": "nonlinear_impact_no_stiffness_proportional_damping",
        }
    M, mass_info = assemble_mass_matrix(model)
    base_load, base_load_info = assemble_load_vector(model, base_load_case)
    zero_load = np.zeros(total_dofs, dtype=float)
    _K_red0, _zero_red, T, u0, independent_dofs, constraint_info = build_constraint_transformation(K0, zero_load, model)
    K_red0 = (T.T @ K0 @ T).tocsr()
    K_energy_red = K_red0 if needs_linear_stiffness else None
    M_red = (T.T @ M @ T).tocsr()
    C_red = (transient_config.rayleigh_alpha * M_red + transient_config.rayleigh_beta * K_red0).tocsr()
    if float(np.linalg.norm(M_red.diagonal())) <= 0.0 and M_red.nnz == 0:
        raise ValueError("Nonlinear sphere impact requires a non-zero structural mass matrix; set material density values.")

    times = _time_grid(transient_config)
    recovery = transient_config.recovery
    output_node_ids = tuple(int(node_id) for node_id in (transient_config.output_nodes or ()))
    if recovery is not None and not output_node_ids and recovery.node_ids is not None:
        output_node_ids = tuple(int(node_id) for node_id in recovery.node_ids)
    history_storage_mode = "full" if recovery is None else str(recovery.history_mode)
    if recovery is not None and history_storage_mode == "full" and not recovery.store_full_histories:
        history_storage_mode = "selected"
    if history_storage_mode == "selected" and not output_node_ids and (recovery is None or recovery.include_displacements):
        output_node_ids = tuple(int(node_id) for node_id in model.mesh.nodes)
    history_dof_indices = _node_dof_indices(model, output_node_ids) if history_storage_mode == "selected" else None
    estimated_saved_steps = _saved_step_count(times, transient_config.save_every)
    preflight_memory = estimate_model_memory(
        model,
        transient_saved_steps=estimated_saved_steps,
        store_full_history=history_storage_mode == "full",
        recovery_config=recovery,
    )
    enforce_memory_limit(preflight_memory, transient_config.resource_config, context="solve_transient_sphere_impact.nonlinear")

    q = _full_initial_vector(transient_config.initial_displacement, total_dofs)
    v_full = _full_initial_vector(transient_config.initial_velocity, total_dofs)
    q_red = np.asarray((q - u0)[np.asarray(independent_dofs, dtype=int)], dtype=float).reshape(-1)
    v_red = np.asarray(v_full[np.asarray(independent_dofs, dtype=int)], dtype=float).reshape(-1)
    committed_states: Dict[int, Any] = {}
    deleted_element_ids: set[int] = set()
    damage_deleted_element_ids: set[int] = set()
    plastic_damage_records: List[DeletedElementRecord] = []
    plastic_damage_states: Dict[int, Dict[str, Any]] = {}
    damage_scale_by_element: Dict[int, float] = {}
    linear_matrix_terms: Optional[Tuple[int, Tuple[LinearElementMatrixTerms, ...]]] = None
    F_int0, _K_dummy, trial0 = assemble_nonlinear(
        reconstruct_full_solution(T, q_red, u0),
        committed_states,
        tangent=False,
    )
    committed_states = trial0
    F_int_prev = np.asarray(F_int0, dtype=float).copy()
    F0_red = np.asarray(T.T @ (base_load - F_int0), dtype=float).reshape(-1)
    mass_handle = factorize(M_red, MatrixClass.SYMMETRIC_SEMIDEFINITE, signature="sphere_impact.nl.initial_mass")
    a_red = np.asarray(mass_handle.solve(F0_red - C_red @ v_red), dtype=float).reshape(-1)

    sphere_position = sphere.initial_position
    sphere_velocity = sphere.travel_velocity if float(times[0]) >= sphere.t_start else np.zeros(3, dtype=float)
    sphere_acceleration = np.zeros(3, dtype=float)

    saved_times: List[float] = []
    saved_u: List[np.ndarray] = []
    saved_v: List[np.ndarray] = []
    saved_a: List[np.ndarray] = []
    saved_sphere_x: List[np.ndarray] = []
    saved_sphere_v: List[np.ndarray] = []
    saved_sphere_a: List[np.ndarray] = []
    saved_contact_force: List[np.ndarray] = []
    saved_contacts: List[Tuple[Dict[str, Any], ...]] = []
    node_history_values: Dict[int, list[np.ndarray]] = {int(node_id): [] for node_id in output_node_ids}
    element_state_history: List[Dict[str, Any]] = []
    peak_displacement = 0.0
    peak_displacement_node = None
    max_penetration = 0.0
    peak_contact_force = 0.0
    contact_step_count = 0
    active_contact_duration = 0.0
    iteration_counts: List[int] = []
    energy_kinetic: List[float] = []
    energy_strain: List[float] = []
    sphere_kinetic: List[float] = []
    plastic_work_history: List[float] = []

    def save_state(
        time: float,
        q_state: np.ndarray,
        v_state: np.ndarray,
        a_state: np.ndarray,
        sphere_x: np.ndarray,
        sphere_v: np.ndarray,
        sphere_a: np.ndarray,
        sphere_force: np.ndarray,
        records: Tuple[SphereContactRecord, ...],
    ) -> None:
        nonlocal peak_displacement, peak_displacement_node, max_penetration, peak_contact_force
        full_u = reconstruct_full_solution(T, q_state, u0)
        full_v = np.asarray(T @ v_state, dtype=float).reshape(-1)
        full_a = np.asarray(T @ a_state, dtype=float).reshape(-1)
        saved_times.append(float(time))
        if history_storage_mode == "full":
            saved_u.append(full_u)
            saved_v.append(full_v)
            saved_a.append(full_a)
        elif history_storage_mode == "selected":
            indices = np.asarray(history_dof_indices if history_dof_indices is not None else (), dtype=np.intp)
            saved_u.append(full_u[indices])
            saved_v.append(full_v[indices])
            saved_a.append(full_a[indices])
        for node_id in output_node_ids:
            node = model.mesh.get_node(int(node_id))
            if node is not None:
                node_history_values[int(node_id)].append(full_u[np.asarray(node.dofs, dtype=np.intp)])
        current_peak, current_node = _translation_peak(model, full_u)
        if current_peak > peak_displacement:
            peak_displacement = current_peak
            peak_displacement_node = current_node
        if records:
            max_penetration = max(max_penetration, max(float(record.penetration) for record in records))
            peak_contact_force = max(peak_contact_force, max(float(record.normal_force) for record in records))
        saved_sphere_x.append(np.asarray(sphere_x, dtype=float).copy())
        saved_sphere_v.append(np.asarray(sphere_v, dtype=float).copy())
        saved_sphere_a.append(np.asarray(sphere_a, dtype=float).copy())
        saved_contact_force.append(np.asarray(sphere_force, dtype=float).copy())
        saved_contacts.append(tuple(record.to_dict() for record in records) if config.save_contact_history else tuple())
        energy_kinetic.append(0.5 * float(v_state @ (M_red @ v_state)))
        sphere_kinetic.append(0.5 * float(sphere.mass) * float(sphere_v @ sphere_v))
        if K_energy_red is not None:
            energy_strain.append(0.5 * float(q_state @ (K_energy_red @ q_state)))
        else:
            # internal work measure from the committed nonlinear internal
            # force: exact strain energy for the elastic response, and an
            # internal-work proxy (elastic + dissipated) once plasticity is
            # active.
            energy_strain.append(0.5 * float(full_u @ F_int_prev))
        plastic_work_history.append(float(_nonlinear_state_summary(committed_states).get("max_equivalent_plastic_strain", 0.0) or 0.0))
        if bool(nl.record_element_state_history):
            summary = _nonlinear_state_summary(committed_states)
            element_state_history.append({"time": float(time), **summary})
        if progress_callback is not None and history_storage_mode == "full":
            try:
                progress_callback(
                    {
                        "type": "sphere_impact_live_step",
                        "time_s": float(time),
                        "step_index": int(len(saved_times) - 1),
                        "displacement": full_u.copy(),
                        "sphere_position": np.asarray(sphere_x, dtype=float).copy(),
                        "sphere_radius": float(sphere.radius),
                        "contact_force": np.asarray(sphere_force, dtype=float).copy(),
                        "active_contacts": tuple(record.to_dict() for record in records) if config.save_contact_history else tuple(),
                        "nonlinear": True,
                        "max_equivalent_plastic_strain": float(_nonlinear_state_summary(committed_states).get("max_equivalent_plastic_strain", 0.0) or 0.0),
                    }
                )
            except Exception:
                pass

    initial_load, initial_sphere_force, initial_records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        config,
        sphere_position,
        sphere_velocity,
        structural_displacement=reconstruct_full_solution(T, q_red, u0),
        structural_velocity=np.asarray(T @ v_red, dtype=float).reshape(-1),
        deleted_element_ids=tuple(deleted_element_ids),
        contact_scale_by_element=damage_scale_by_element,
    )
    save_state(float(times[0]), q_red, v_red, a_red, sphere_position, sphere_velocity, sphere_acceleration, initial_sphere_force, initial_records)
    sticky_contact_ids: Tuple[int, ...] = tuple(int(record.element_id) for record in initial_records)

    load_prev = base_load + initial_load
    sphere_force_prev = initial_sphere_force.copy()
    impulse = np.zeros(total_dofs, dtype=float)
    sphere_impulse = np.zeros(3, dtype=float)
    status = "completed"
    stop_reason = "completed"
    warnings: List[str] = [f"{issue.code}: {issue.message}" for issue in validation.warnings]
    warnings.append(
        "NONLINEAR_IMPACT001: material-nonlinear impact uses implicit Newmark with structural nonlinear tangent "
        "and iterative penalty contact; exact contact tangent, friction, spin, tearing paths and FSI are unsupported."
    )
    factorization_count = 0
    solve_count = 0
    total_substep_count = 0
    event_substep_count = 0
    cutback_count = 0
    step_diagnostics: List[Dict[str, Any]] = []
    nonlinear_failure_summary: Dict[str, Any] = {}
    alpha_h, beta, gamma = transient_config.integration_parameters()
    one_plus_alpha = 1.0 + alpha_h
    separation_elapsed = 0.0
    separation_stop_time = float(config.post_separation_time)

    segments: List[Tuple[int, float, float, bool]] = [
        (int(index), float(times[index - 1]), float(times[index]), True)
        for index in range(1, len(times))
    ]
    travel_scale = max(float(sphere.radius) * float(config.max_sphere_travel_fraction), 1.0e-12)
    pending_save: Optional[Tuple[float, np.ndarray, Tuple[SphereContactRecord, ...]]] = None
    while segments:
        step_index, segment_start, sub_time, needs_travel_check = segments.pop(0)
        dt = float(sub_time - segment_start)
        if dt <= 0.0:
            continue
        if needs_travel_check:
            predicted_travel = max(float(np.linalg.norm(sphere_velocity)) * dt, float(sphere.speed) * dt)
            n_travel = min(max(int(np.ceil(predicted_travel / travel_scale)), 1), int(config.max_event_substeps))
            if n_travel > 1:
                pieces: List[Tuple[int, float, float, bool]] = []
                piece_start = float(segment_start)
                for piece in range(1, n_travel + 1):
                    piece_end = float(sub_time) if piece == n_travel else float(segment_start) + piece * dt / n_travel
                    pieces.append((step_index, piece_start, piece_end, False))
                    piece_start = piece_end
                segments[:0] = pieces
                event_substep_count += n_travel - 1
                continue
        total_substep_count += 1
        pre_state = (
            q_red.copy(),
            v_red.copy(),
            a_red.copy(),
            sphere_position.copy(),
            sphere_velocity.copy(),
            sphere_acceleration.copy(),
            dict(committed_states),
            set(deleted_element_ids),
            dict(plastic_damage_states),
            dict(damage_scale_by_element),
            load_prev.copy(),
            sphere_force_prev.copy(),
            F_int_prev.copy(),
        )
        if sub_time - dt < sphere.t_start <= sub_time and np.linalg.norm(sphere_velocity) == 0.0:
            sphere_velocity = sphere.travel_velocity.copy()
        a0 = 1.0 / (beta * dt**2)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        q_trial = q_red.copy()
        contact_load = np.zeros(total_dofs, dtype=float)
        sphere_force = np.zeros(3, dtype=float)
        records: Tuple[SphereContactRecord, ...] = tuple()
        converged = False
        residual_norm = float("inf")
        displacement_increment = float("inf")
        contact_change = float("inf")
        reference = 1.0
        contact_scale = 1.0
        trial_states: Dict[int, Any] = committed_states
        q_next = q_red.copy()
        v_next = v_red.copy()
        a_next = a_red.copy()
        sphere_x_next = sphere_position.copy()
        sphere_v_next = sphere_velocity.copy()
        sphere_a_next = sphere_acceleration.copy()
        factor_diagnostics: Dict[str, Any] = {}
        convergence_reason = "standard"
        for iteration in range(1, int(nl.max_iterations) + 1):
            full_u = reconstruct_full_solution(T, q_trial, u0)
            F_int, K_T, trial_states = assemble_nonlinear(
                full_u,
                committed_states,
                tangent=True,
                scales=damage_scale_by_element,
            )
            a_trial = a0 * (q_trial - q_red) - a2 * v_red - a3 * a_red
            v_trial = v_red + dt * ((1.0 - gamma) * a_red + gamma * a_trial)
            if alpha_h != 0.0:
                weighted_force = one_plus_alpha * (base_load + contact_load - F_int) - alpha_h * (load_prev - F_int_prev)
                residual = (
                    np.asarray(T.T @ weighted_force, dtype=float).reshape(-1)
                    - one_plus_alpha * (C_red @ v_trial)
                    + alpha_h * (C_red @ v_red)
                    - M_red @ a_trial
                )
            else:
                residual = np.asarray(T.T @ (base_load + contact_load - F_int), dtype=float).reshape(-1) - C_red @ v_trial - M_red @ a_trial
            residual_norm = float(np.linalg.norm(residual))
            reference = max(float(np.linalg.norm(np.asarray(T.T @ (base_load + contact_load), dtype=float))), float(np.linalg.norm(np.asarray(T.T @ F_int, dtype=float))), 1.0)
            K_eff = (one_plus_alpha * (T.T @ K_T @ T) + a0 * M_red + one_plus_alpha * a1 * C_red).tocsr()
            try:
                handle = factorize(K_eff, MatrixClass.SYMMETRIC_INDEFINITE, signature=f"sphere_impact.nl.effective:{dt:.16g}:{iteration}")
                factor_diagnostics = handle.diagnostics()
                delta = np.asarray(handle.solve(residual), dtype=float).reshape(-1)
                factorization_count += 1
                solve_count += 1
            except Exception:
                status = "nonlinear_tangent_failed"
                stop_reason = "nonlinear_tangent_factorization_failed"
                break
            if np.any(~np.isfinite(delta)):
                status = "nonlinear_iteration_failed"
                stop_reason = "nonfinite_nonlinear_increment"
                break
            factor = 1.0
            if bool(nl.line_search):
                factor = 1.0
                while factor > float(nl.min_line_search_factor) and float(np.linalg.norm(factor * delta)) > 10.0 * max(float(np.linalg.norm(q_trial)), 1.0):
                    factor *= 0.5
            q_next = q_trial + factor * delta
            displacement_increment = float(np.linalg.norm(factor * delta))
            a_next = a0 * (q_next - q_red) - a2 * v_red - a3 * a_red
            v_next = v_red + dt * ((1.0 - gamma) * a_red + gamma * a_next)
            sphere_x_pred = sphere_position + dt * sphere_velocity + dt**2 * (0.5 - beta) * sphere_acceleration
            sphere_v_pred = sphere_velocity + dt * (1.0 - gamma) * sphere_acceleration
            sphere_a_next = sphere_force / float(sphere.mass)
            sphere_x_next = sphere_x_pred + beta * dt**2 * sphere_a_next
            sphere_v_next = sphere_v_pred + gamma * dt * sphere_a_next
            full_u_next = reconstruct_full_solution(T, q_next, u0)
            full_v_next = np.asarray(T @ v_next, dtype=float).reshape(-1)
            new_contact_load, new_sphere_force, new_records = assemble_sphere_contact_load_vector(
                model,
                sphere,
                config,
                sphere_x_next,
                sphere_v_next,
                structural_displacement=full_u_next,
                structural_velocity=full_v_next,
                deleted_element_ids=tuple(deleted_element_ids),
                contact_scale_by_element=damage_scale_by_element,
                preferred_element_ids=sticky_contact_ids,
            )
            if new_records:
                sticky_contact_ids = tuple(int(record.element_id) for record in new_records)
            contact_scale = max(float(np.linalg.norm(new_sphere_force)), 1.0)
            contact_change = float(np.linalg.norm(new_sphere_force - sphere_force))
            contact_load, sphere_force, records = new_contact_load, new_sphere_force, new_records
            q_trial = q_next
            displacement_limit = float(nl.displacement_tolerance) * max(float(np.linalg.norm(q_next)), 1.0)
            residual_limit = float(nl.residual_tolerance) * reference
            force_limit = max(float(nl.contact_force_tolerance) * contact_scale, float(config.force_tolerance))
            standard_converged = (
                residual_norm <= residual_limit
                and displacement_increment <= displacement_limit
                and contact_change <= force_limit
            )
            contact_stall_converged = (
                iteration >= max(3, int(nl.max_iterations) // 2)
                and residual_norm <= 1.0e-2 * reference
                and displacement_increment <= max(10.0 * displacement_limit, 1.0e-6)
                and contact_change <= max(5.0e-3 * contact_scale, 10.0 * float(config.force_tolerance))
            )
            if standard_converged or contact_stall_converged:
                converged = True
                convergence_reason = "standard" if standard_converged else "contact_stall"
                iteration_counts.append(iteration)
                break
        if not converged:
            if status in {"nonlinear_tangent_failed", "nonlinear_iteration_failed"}:
                pass
            elif cutback_count < int(nl.max_cutbacks) and dt * 0.5 >= float(nl.min_dt):
                (
                    q_red,
                    v_red,
                    a_red,
                    sphere_position,
                    sphere_velocity,
                    sphere_acceleration,
                    committed_states,
                    deleted_element_ids,
                    plastic_damage_states,
                    damage_scale_by_element,
                    load_prev,
                    sphere_force_prev,
                    F_int_prev,
                ) = pre_state
                midpoint = 0.5 * (float(segment_start) + float(sub_time))
                segments.insert(0, (step_index, midpoint, float(sub_time), False))
                segments.insert(0, (step_index, float(segment_start), midpoint, False))
                cutback_count += 1
                event_substep_count += 1
                step_diagnostics.append(
                    {
                        "step_index": int(step_index),
                        "time": float(sub_time),
                        "dt": float(dt),
                        "status": "nonlinear_cutback_retry",
                        "residual_norm": float(residual_norm),
                        "effective_residual_tolerance": float(nl.residual_tolerance) * float(reference),
                        "displacement_increment": float(displacement_increment),
                        "contact_force_change": float(contact_change),
                        "effective_force_tolerance": max(float(nl.contact_force_tolerance) * float(contact_scale), float(config.force_tolerance)),
                    }
                )
                nonlinear_failure_summary = dict(step_diagnostics[-1])
                nonlinear_failure_summary["suggestion"] = (
                    "The nonlinear contact/plasticity iteration recovered by cutting the time step. "
                    "If failures persist, reduce dt/penalty, allow more cutbacks, or loosen the nonlinear contact tolerance."
                )
                continue
            else:
                status = "nonlinear_iteration_failed"
                stop_reason = "maximum_iterations_or_cutbacks_reached"
                active_ids = [int(record.element_id) for record in records]
                nonlinear_failure_summary = {
                    "step_index": int(step_index),
                    "time": float(sub_time),
                    "dt": float(dt),
                    "status": str(status),
                    "stop_reason": str(stop_reason),
                    "iterations": int(nl.max_iterations),
                    "cutbacks": int(cutback_count),
                    "residual_norm": float(residual_norm),
                    "effective_residual_tolerance": float(nl.residual_tolerance) * float(reference),
                    "displacement_increment": float(displacement_increment),
                    "contact_force_change": float(contact_change),
                    "effective_force_tolerance": max(float(nl.contact_force_tolerance) * float(contact_scale), float(config.force_tolerance)),
                    "active_element_ids": active_ids,
                    "max_penetration": float(max((record.penetration for record in records), default=0.0)),
                    "suggestion": (
                        "The nonlinear impact solve exhausted its time-step cutbacks. "
                        "Use a smaller manual dt, increase NL cutbacks, reduce contact penalty/target penetration stiffness, "
                        "or run the linear collision path first to confirm the contact setup."
                    ),
                }
            if not iteration_counts:
                iteration_counts.append(int(nl.max_iterations))
            break

        committed_states = trial_states
        contact_norm = float(np.linalg.norm(sphere_force))
        if contact_norm > 0.0:
            contact_step_count += 1
            active_contact_duration += dt
            separation_elapsed = 0.0
        elif contact_step_count > 0 and separation_stop_time > 0.0:
            separation_elapsed += dt
        if records:
            max_penetration = max(max_penetration, max(float(record.penetration) for record in records))
            peak_contact_force = max(peak_contact_force, max(float(record.normal_force) for record in records))
        impulse += 0.5 * (load_prev + base_load + contact_load) * dt
        sphere_impulse += 0.5 * (sphere_force_prev + sphere_force) * dt
        load_prev = base_load + contact_load
        F_int_prev = F_int
        sphere_force_prev = sphere_force.copy()
        q_red, v_red, a_red = q_next, v_next, a_next
        sphere_position, sphere_velocity, sphere_acceleration = sphere_x_next, sphere_v_next, sphere_a_next
        new_damage_records, damage_changed, max_damage_util = _plastic_impact_damage_update(
            model,
            committed_states,
            plastic_damage_config,
            deleted_element_ids,
            plastic_damage_states,
            step_index=int(step_index),
            time_value=float(sub_time),
        )
        if new_damage_records:
            plastic_damage_records.extend(new_damage_records)
            damage_deleted_element_ids.update(record.element_id for record in new_damage_records)
        if plastic_damage_config is not None:
            damage_scale_by_element = {
                int(element_id): float(state.get("scale", 1.0) or 1.0)
                for element_id, state in plastic_damage_states.items()
                if float(state.get("scale", 1.0) or 1.0) < 1.0
            }
            for element_id in deleted_element_ids:
                damage_scale_by_element[int(element_id)] = float(plastic_damage_config.residual_stiffness_fraction)
            if damage_changed:
                filtered_base = filtered_load_case_for_deleted_elements(base_load_case, deleted_element_ids)
                base_load, base_load_info = assemble_load_vector(model, filtered_base)
                if linear_matrix_terms is None:
                    linear_matrix_terms = _linear_element_matrix_terms(model)
                _K_scaled, M_scaled = _assemble_damaged_linear_matrices(model, damage_scale_by_element, cached_terms=linear_matrix_terms)
                M_red = (T.T @ M_scaled @ T).tocsr()
                C_red = (transient_config.rayleigh_alpha * M_red + transient_config.rayleigh_beta * K_red0).tocsr()
                scoped_total = sum(
                    1
                    for element in model.mesh.elements.values()
                    if element_fracture_category(element) in set(plastic_damage_config.element_scope)
                )
                if len(deleted_element_ids) / max(scoped_total, 1) > float(plastic_damage_config.max_deleted_fraction) + 1.0e-12:
                    status = "max_deleted_fraction_reached"
                    stop_reason = "plastic_impact_damage_max_deleted_fraction_reached"
        state_summary = _nonlinear_state_summary(committed_states)
        step_diagnostics.append(
            {
                "step_index": int(step_index),
                "time": float(sub_time),
                "dt": float(dt),
                "status": "converged",
                "iterations": int(iteration_counts[-1]),
                "residual_norm": float(residual_norm),
                "displacement_increment": float(displacement_increment),
                "contact_force_change": float(contact_change),
                "convergence_reason": str(convergence_reason),
                "active_element_ids": [int(record.element_id) for record in records],
                "max_penetration": float(max((record.penetration for record in records), default=0.0)),
                "max_equivalent_plastic_strain": float(state_summary.get("max_equivalent_plastic_strain", 0.0) or 0.0),
                "max_damage_utilization": float(max_damage_util),
            }
        )
        step_complete = float(sub_time) >= float(times[step_index]) - 1.0e-12 * max(abs(float(times[step_index])), 1.0)
        if step_complete and (step_index % int(transient_config.save_every) == 0 or step_index == len(times) - 1):
            save_state(sub_time, q_red, v_red, a_red, sphere_position, sphere_velocity, sphere_acceleration, sphere_force, records)
            pending_save = None
        else:
            pending_save = (float(sub_time), sphere_force.copy(), records)
        if status == "max_deleted_fraction_reached":
            break
        if status == "completed" and contact_step_count > 0 and separation_stop_time > 0.0 and separation_elapsed >= separation_stop_time:
            stop_reason = "completed_after_contact_separation"
            break

    if pending_save is not None:
        save_state(
            pending_save[0], q_red, v_red, a_red, sphere_position, sphere_velocity, sphere_acceleration, pending_save[1], pending_save[2]
        )

    if contact_step_count == 0 and status == "completed":
        status = "no_contact"
        stop_reason = "no_contact"
    impulse_resultant = load_vector_resultant(model, impulse)
    max_penetration_ratio = float(max_penetration) / max(float(sphere.radius), 1.0e-12)
    contact_duration = float(active_contact_duration)
    initial_sphere_velocity = sphere.travel_velocity if float(times[0]) >= sphere.t_start else np.zeros(3, dtype=float)
    final_sphere_velocity = np.asarray(saved_sphere_v[-1], dtype=float) if saved_sphere_v else sphere_velocity
    sphere_momentum_change = float(sphere.mass) * (final_sphere_velocity - initial_sphere_velocity)
    sphere_momentum_balance_error = float(np.linalg.norm(sphere_impulse - sphere_momentum_change))
    history_width = total_dofs if history_storage_mode == "full" else int(0 if history_dof_indices is None else len(history_dof_indices))
    saved_u_array = np.vstack(saved_u) if saved_u else np.zeros((0, history_width), dtype=float)
    saved_v_array = np.vstack(saved_v) if saved_v else np.zeros((0, history_width), dtype=float)
    saved_a_array = np.vstack(saved_a) if saved_a else np.zeros((0, history_width), dtype=float)
    node_histories: Dict[int, np.ndarray] = {}
    for node_id in output_node_ids:
        values = node_history_values.get(int(node_id), [])
        node_histories[int(node_id)] = np.vstack(values) if values else np.zeros((0, 6), dtype=float)
    recovery_memory = estimate_model_memory(
        model,
        transient_saved_steps=len(saved_times),
        store_full_history=history_storage_mode == "full",
        recovery_config=recovery,
    )
    enforce_memory_limit(recovery_memory, transient_config.resource_config, context="solve_transient_sphere_impact.nonlinear.recovery")
    policy_metadata = recovery_metadata(recovery, transient_config.resource_config, recovery_memory)
    assembly_info = {"stiffness": stiffness_info, "mass": mass_info, "load": base_load_info}
    plastic_damage_summary = _plastic_impact_damage_summary(
        model,
        plastic_damage_config,
        plastic_damage_states,
        damage_deleted_element_ids,
        plastic_damage_records,
        warnings=[warning for warning in warnings if str(warning).startswith("NONLINEAR_IMPACT")],
    )
    erosion_summary = {
        "all_eroded_element_ids": sorted(int(element_id) for element_id in deleted_element_ids),
        "damage_triggered_element_ids": sorted(int(element_id) for element_id in damage_deleted_element_ids),
        "active_softened_element_ids": sorted(
            int(element_id)
            for element_id, scale in damage_scale_by_element.items()
            if int(element_id) not in deleted_element_ids and float(scale) < 1.0
        ),
        "residual_stiffness_model": "nonlinear element force/tangent and mass scaling; topology, nodes, MPCs retained",
    }
    result_case = make_result_case(
        name=f"sphere_impact_nonlinear:{sphere.name}",
        analysis_type="sphere_impact_nonlinear_transient",
        load_cases=() if base_load_case is None else (base_load_case,),
        assembly_info=assembly_info,
        solver_info={"backend": factor_diagnostics, "convergence_info": {"status": status, "stop_reason": stop_reason}},
        recovery={
            "displacement_history": True,
            "velocity_history": True,
            "acceleration_history": True,
            "sphere_history": True,
            "contact_history": bool(config.save_contact_history),
            "element_state_history": bool(nl.record_element_state_history),
            "history_storage_mode": history_storage_mode,
            **policy_metadata["recovery"],
        },
        settings={
            "dt": transient_config.dt,
            "t_end": transient_config.t_end,
            "beta": beta,
            "gamma": gamma,
            "hht_alpha": alpha_h,
            "rayleigh_alpha": transient_config.rayleigh_alpha,
            "rayleigh_beta": transient_config.rayleigh_beta,
            "sphere": {
                "name": sphere.name,
                "radius": float(sphere.radius),
                "mass": float(sphere.mass),
                "start_point": sphere.initial_position.tolist(),
                "travel_direction": sphere.direction_unit.tolist(),
                "speed": float(sphere.speed),
                "t_start": float(sphere.t_start),
            },
            "contact": config.to_dict(),
            "nonlinear": nl.to_dict(),
            "plastic_damage": None if plastic_damage_config is None else plastic_damage_config.to_dict(),
        },
        metadata={
            "resources": policy_metadata.get("resources"),
            "memory_estimate": policy_metadata.get("memory_estimate"),
            "contact_scope": "single rigid sphere to shell target; frictionless normal penalty with iterative active-set updates",
            "nonlinear_scope": "implicit Newmark; current shell/beam nonlinear element formulations; approximate contact tangent",
            "solution_control": "implicit_newmark_time_domain",
            "arc_length_applicability": "not_applicable_to_dynamic_impact",
            "solver_convergence": {"status": status, "stop_reason": stop_reason},
            "verification_gate": "nonlinear_contact",
        },
    ).to_dict()
    total_energy = (
        np.asarray(energy_kinetic, dtype=float)
        + np.asarray(energy_strain, dtype=float)
        + np.asarray(sphere_kinetic, dtype=float)
    )
    nonzero_energy = total_energy[np.abs(total_energy) > 1.0e-30]
    energy_drift = 0.0
    if nonzero_energy.size:
        energy_drift = float((np.max(nonzero_energy) - np.min(nonzero_energy)) / max(abs(nonzero_energy[0]), 1.0e-30))
    diagnostics = {
        "method": "nonlinear_newmark_sphere_penalty_contact",
        "solution_control": "implicit_newmark_time_domain",
        "arc_length_applicability": "not_applicable_to_dynamic_impact",
        "status": status,
        "stop_reason": stop_reason,
        "warnings": warnings,
        "num_steps": max(int(len(times) - 1), 0),
        "num_substeps": int(total_substep_count),
        "event_substep_count": int(event_substep_count),
        "cutback_count": int(cutback_count),
        "num_saved_steps": len(saved_times),
        "num_reduced_dofs": int(M_red.shape[0]),
        "contact_step_count": int(contact_step_count),
        "active_contact_duration": float(contact_duration),
        "separation_stop_time": float(separation_stop_time),
        "post_contact_separation_time": float(separation_elapsed if contact_step_count > 0 else 0.0),
        "max_penetration_ratio": float(max_penetration_ratio),
        "sphere_momentum_balance_error": float(sphere_momentum_balance_error),
        "iteration_counts": iteration_counts,
        "contact_step_diagnostics": step_diagnostics,
        "nonlinear_failure_summary": nonlinear_failure_summary,
        "factorization_count": int(factorization_count),
        "solve_count": int(solve_count),
        "initial_mass_factorization": mass_handle.diagnostics(),
        "effective_stiffness_factorization": factor_diagnostics,
        "constraint_info": constraint_info,
        "stiffness": stiffness_info,
        "mass": mass_info,
        "base_load": base_load_info,
        "contact_config": config.to_dict(),
        "nonlinear_config": nl.to_dict(),
        "strain_summary": _nonlinear_state_summary(committed_states),
        "element_states": committed_states,
        "element_state_history": element_state_history,
        "plastic_impact_damage_summary": plastic_damage_summary,
        "impact_damage_summary": plastic_damage_summary,
        "erosion_summary": erosion_summary,
        "contact_validation": validation.to_dict(),
        "sphere": result_case["analysis_case"]["settings"]["sphere"],
        "kinetic_energy": energy_kinetic,
        "strain_energy": energy_strain,
        "sphere_kinetic_energy": sphere_kinetic,
        "plastic_work_proxy": plastic_work_history,
        "max_relative_energy_drift": energy_drift,
        "result_case": result_case,
    }
    return SphereImpactResult(
        times=np.asarray(saved_times, dtype=float),
        displacements=saved_u_array,
        velocities=saved_v_array,
        accelerations=saved_a_array,
        node_histories=node_histories,
        sphere_positions=np.vstack(saved_sphere_x) if saved_sphere_x else np.zeros((0, 3), dtype=float),
        sphere_velocities=np.vstack(saved_sphere_v) if saved_sphere_v else np.zeros((0, 3), dtype=float),
        sphere_accelerations=np.vstack(saved_sphere_a) if saved_sphere_a else np.zeros((0, 3), dtype=float),
        contact_force_history=np.vstack(saved_contact_force) if saved_contact_force else np.zeros((0, 3), dtype=float),
        active_contact_history=tuple(saved_contacts),
        load_impulse=impulse,
        force_impulse=impulse_resultant.force,
        moment_impulse=impulse_resultant.moment,
        sphere_impulse=sphere_impulse,
        max_penetration=float(max_penetration),
        max_penetration_ratio=float(max_penetration_ratio),
        peak_contact_force=float(peak_contact_force),
        contact_duration=float(contact_duration),
        sphere_momentum_balance_error=float(sphere_momentum_balance_error),
        peak_displacement=float(peak_displacement),
        peak_displacement_node=peak_displacement_node,
        status=status,
        diagnostics=diagnostics,
        result_case=result_case,
    )


def solve_transient_sphere_impact(
    model: "FEModel",
    transient_config: TransientConfig,
    sphere: RigidSphereImpact,
    contact_config: Optional[SphereContactConfig] = None,
    base_load_case: Optional[LoadCase] = None,
    fracture_config: Optional[ImpactFractureConfig] = None,
    damage_config: Optional[ImpactDamageConfig] = None,
    nonlinear_config: Optional[NonlinearTransientConfig] = None,
    plastic_damage_config: Optional[PlasticImpactDamageConfig] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> SphereImpactResult:
    """Solve a limited rigid-sphere-to-shell impact transient."""

    config = _resolved_contact_config(model, sphere, contact_config)
    if fracture_config is not None and not isinstance(fracture_config, ImpactFractureConfig):
        raise TypeError("fracture_config must be an ImpactFractureConfig or None")
    if damage_config is not None and not isinstance(damage_config, ImpactDamageConfig):
        raise TypeError("damage_config must be an ImpactDamageConfig or None")
    if nonlinear_config is not None and not isinstance(nonlinear_config, NonlinearTransientConfig):
        raise TypeError("nonlinear_config must be a NonlinearTransientConfig or None")
    if plastic_damage_config is not None and not isinstance(plastic_damage_config, PlasticImpactDamageConfig):
        raise TypeError("plastic_damage_config must be a PlasticImpactDamageConfig or None")
    validation = validate_contact_configuration(model, sphere, config, transient_config)
    if validation.errors:
        codes = ", ".join(issue.code for issue in validation.errors)
        raise ValueError(f"Invalid sphere contact configuration: {codes}")
    if nonlinear_config is not None and bool(nonlinear_config.enabled):
        return _solve_transient_sphere_impact_nonlinear(
            model,
            transient_config,
            sphere,
            config,
            validation,
            base_load_case=base_load_case,
            nonlinear_config=nonlinear_config,
            plastic_damage_config=plastic_damage_config,
            progress_callback=progress_callback,
        )
    model.apply_boundary_conditions()
    K, stiffness_info = assemble_stiffness_matrix(model)
    M, mass_info = assemble_mass_matrix(model)
    total_dofs = model.mesh.dof_manager.total_dofs
    deleted_element_ids: set[int] = set()
    fracture_deleted_element_ids: set[int] = set()
    damage_deleted_element_ids: set[int] = set()
    fracture_records: List[DeletedElementRecord] = []
    damage_records: List[DeletedElementRecord] = []
    max_fracture_utilization = 0.0
    impact_damage_states: Dict[int, Dict[str, Any]] = {}
    max_damage_utilization = 0.0
    damage_scale_by_element: Dict[int, float] = {}
    linear_matrix_terms: Optional[Tuple[int, Tuple[LinearElementMatrixTerms, ...]]] = None
    base_load, base_load_info = assemble_load_vector(model, base_load_case)
    zero_load = np.zeros(total_dofs, dtype=float)
    K_red, _zero_red, T, u0, independent_dofs, constraint_info = build_constraint_transformation(K, zero_load, model)
    M_red = (T.T @ M @ T).tocsr()
    C_red = (transient_config.rayleigh_alpha * M_red + transient_config.rayleigh_beta * K_red).tocsr()
    if float(np.linalg.norm(M_red.diagonal())) <= 0.0 and M_red.nnz == 0:
        raise ValueError("Sphere impact transient requires a non-zero structural mass matrix; set material density values.")

    times = _time_grid(transient_config)
    recovery = transient_config.recovery
    output_node_ids = tuple(int(node_id) for node_id in (transient_config.output_nodes or ()))
    if recovery is not None and not output_node_ids and recovery.node_ids is not None:
        output_node_ids = tuple(int(node_id) for node_id in recovery.node_ids)
    history_storage_mode = "full" if recovery is None else str(recovery.history_mode)
    if recovery is not None and history_storage_mode == "full" and not recovery.store_full_histories:
        history_storage_mode = "selected"
    if history_storage_mode == "selected" and not output_node_ids and (recovery is None or recovery.include_displacements):
        output_node_ids = tuple(int(node_id) for node_id in model.mesh.nodes)
    history_dof_indices = _node_dof_indices(model, output_node_ids) if history_storage_mode == "selected" else None
    estimated_saved_steps = _saved_step_count(times, transient_config.save_every)
    preflight_memory = estimate_model_memory(
        model,
        transient_saved_steps=estimated_saved_steps,
        store_full_history=history_storage_mode == "full",
        recovery_config=recovery,
    )
    enforce_memory_limit(preflight_memory, transient_config.resource_config, context="solve_transient_sphere_impact")

    q = _full_initial_vector(transient_config.initial_displacement, total_dofs)
    v_full = _full_initial_vector(transient_config.initial_velocity, total_dofs)
    q_red = np.asarray((q - u0)[np.asarray(independent_dofs, dtype=int)], dtype=float).reshape(-1)
    v_red = np.asarray(v_full[np.asarray(independent_dofs, dtype=int)], dtype=float).reshape(-1)
    F0_red = _reduced_load(T, K, u0, base_load)
    mass_handle = factorize(M_red, MatrixClass.SYMMETRIC_SEMIDEFINITE, signature="sphere_impact.initial_mass")
    a_red = np.asarray(mass_handle.solve(F0_red - C_red @ v_red - K_red @ q_red), dtype=float).reshape(-1)

    sphere_position = sphere.initial_position
    sphere_velocity = sphere.travel_velocity if float(times[0]) >= sphere.t_start else np.zeros(3, dtype=float)
    sphere_acceleration = np.zeros(3, dtype=float)

    saved_times: List[float] = []
    saved_u: List[np.ndarray] = []
    saved_v: List[np.ndarray] = []
    saved_a: List[np.ndarray] = []
    saved_sphere_x: List[np.ndarray] = []
    saved_sphere_v: List[np.ndarray] = []
    saved_sphere_a: List[np.ndarray] = []
    saved_contact_force: List[np.ndarray] = []
    saved_contacts: List[Tuple[Dict[str, Any], ...]] = []
    node_history_values: Dict[int, list[np.ndarray]] = {int(node_id): [] for node_id in output_node_ids}
    peak_displacement = 0.0
    peak_displacement_node = None
    max_penetration = 0.0
    peak_contact_force = 0.0
    contact_step_count = 0
    active_contact_duration = 0.0
    iteration_counts: List[int] = []
    energy_kinetic: List[float] = []
    energy_strain: List[float] = []
    sphere_kinetic: List[float] = []

    def save_state(
        time: float,
        q_state: np.ndarray,
        v_state: np.ndarray,
        a_state: np.ndarray,
        sphere_x: np.ndarray,
        sphere_v: np.ndarray,
        sphere_a: np.ndarray,
        sphere_force: np.ndarray,
        records: Tuple[SphereContactRecord, ...],
    ) -> None:
        nonlocal peak_displacement, peak_displacement_node, max_penetration, peak_contact_force
        full_u = reconstruct_full_solution(T, q_state, u0)
        full_v = np.asarray(T @ v_state, dtype=float).reshape(-1)
        full_a = np.asarray(T @ a_state, dtype=float).reshape(-1)
        saved_times.append(float(time))
        if history_storage_mode == "full":
            saved_u.append(full_u)
            saved_v.append(full_v)
            saved_a.append(full_a)
        elif history_storage_mode == "selected":
            indices = np.asarray(history_dof_indices if history_dof_indices is not None else (), dtype=np.intp)
            saved_u.append(full_u[indices])
            saved_v.append(full_v[indices])
            saved_a.append(full_a[indices])
        for node_id in output_node_ids:
            node = model.mesh.get_node(int(node_id))
            if node is not None:
                node_history_values[int(node_id)].append(full_u[np.asarray(node.dofs, dtype=np.intp)])
        current_peak, current_node = _translation_peak(model, full_u)
        if current_peak > peak_displacement:
            peak_displacement = current_peak
            peak_displacement_node = current_node
        if records:
            max_penetration = max(max_penetration, max(float(record.penetration) for record in records))
            peak_contact_force = max(peak_contact_force, max(float(record.normal_force) for record in records))
        saved_sphere_x.append(np.asarray(sphere_x, dtype=float).copy())
        saved_sphere_v.append(np.asarray(sphere_v, dtype=float).copy())
        saved_sphere_a.append(np.asarray(sphere_a, dtype=float).copy())
        saved_contact_force.append(np.asarray(sphere_force, dtype=float).copy())
        saved_contacts.append(tuple(record.to_dict() for record in records) if config.save_contact_history else tuple())
        energy_kinetic.append(0.5 * float(v_state @ (M_red @ v_state)))
        energy_strain.append(0.5 * float(q_state @ (K_red @ q_state)))
        sphere_kinetic.append(0.5 * float(sphere.mass) * float(sphere_v @ sphere_v))
        if progress_callback is not None and history_storage_mode == "full":
            try:
                progress_callback(
                    {
                        "type": "sphere_impact_live_step",
                        "time_s": float(time),
                        "step_index": int(len(saved_times) - 1),
                        "displacement": full_u.copy(),
                        "sphere_position": np.asarray(sphere_x, dtype=float).copy(),
                        "sphere_radius": float(sphere.radius),
                        "contact_force": np.asarray(sphere_force, dtype=float).copy(),
                        "active_contacts": tuple(record.to_dict() for record in records) if config.save_contact_history else tuple(),
                    }
                )
            except Exception:
                pass

    initial_load, initial_sphere_force, initial_records = assemble_sphere_contact_load_vector(
        model,
        sphere,
        config,
        sphere_position,
        sphere_velocity,
        structural_displacement=reconstruct_full_solution(T, q_red, u0),
        structural_velocity=np.asarray(T @ v_red, dtype=float).reshape(-1),
        deleted_element_ids=tuple(deleted_element_ids),
        contact_scale_by_element=damage_scale_by_element,
    )
    save_state(float(times[0]), q_red, v_red, a_red, sphere_position, sphere_velocity, sphere_acceleration, initial_sphere_force, initial_records)
    sticky_contact_ids: Tuple[int, ...] = tuple(int(record.element_id) for record in initial_records)
    load_prev = base_load + initial_load
    sphere_force_prev = initial_sphere_force.copy()
    impulse = np.zeros(total_dofs, dtype=float)
    sphere_impulse = np.zeros(3, dtype=float)

    factorization_count = 0
    solve_count = 0
    cached_dt = None
    cached_solver = None
    cached_solver_diagnostics: Dict[str, Any] = {}
    status = "completed"
    stop_reason = "completed"
    warnings: List[str] = []
    for issue in validation.warnings:
        warnings.append(f"{issue.code}: {issue.message}")
    warnings.extend(
        _impact_damage_preflight_warnings(
            model, damage_config, sphere, fracture_config=fracture_config, beam_contact=bool(config.beam_contact)
        )
    )
    if fracture_config is not None:
        warnings.append(
            "IMPACT_FRACTURE001: impact fracture uses contact-observable thresholds and residual element erosion; "
            "it is not ductile crack propagation or material nonlinear impact."
        )
    if damage_config is not None:
        warnings.append(
            "IMPACT_DAMAGE001: impact damage uses engineering contact-demand utilization and element softening/erosion; "
            "it is not crack propagation, material nonlinear impact, or validated fracture mechanics."
        )

    alpha_h, beta, gamma = transient_config.integration_parameters()
    one_plus_alpha = 1.0 + alpha_h
    F_red_prev = _reduced_load(T, K, u0, load_prev)
    total_substep_count = 0
    event_substep_count = 0
    step_diagnostics: List[Dict[str, Any]] = []
    fracture_limit_warning_emitted = False
    damage_deletion_warning_emitted = False
    damage_limit_warning_emitted = False
    eroded_matrix_rebuild_count = 0
    damage_state_update_count = 0
    separation_elapsed = 0.0
    separation_stop_time = float(config.post_separation_time)
    stop_after_separation = False
    for step_index in range(1, len(times)):
        target_time = float(times[step_index])
        dt_total = float(times[step_index] - times[step_index - 1])
        if dt_total <= 0.0:
            continue
        travel_scale = max(float(sphere.radius) * float(config.max_sphere_travel_fraction), 1.0e-12)
        predicted_travel = max(float(np.linalg.norm(sphere_velocity)) * dt_total, float(sphere.speed) * dt_total)
        n_substeps = min(max(int(np.ceil(predicted_travel / travel_scale)), 1), int(config.max_event_substeps))
        total_substep_count += n_substeps
        if n_substeps > 1:
            event_substep_count += n_substeps - 1
        last_records: Tuple[SphereContactRecord, ...] = tuple()
        last_sphere_force = np.zeros(3, dtype=float)
        last_step_diag: Dict[str, Any] = {}
        for substep in range(n_substeps):
            sub_time = float(times[step_index - 1]) + (substep + 1) * dt_total / n_substeps
            dt = dt_total / n_substeps
            if sub_time - dt < sphere.t_start <= sub_time and np.linalg.norm(sphere_velocity) == 0.0:
                sphere_velocity = sphere.travel_velocity.copy()
            a0 = 1.0 / (beta * dt**2)
            a1 = gamma / (beta * dt)
            a2 = 1.0 / (beta * dt)
            a3 = 1.0 / (2.0 * beta) - 1.0
            a4 = gamma / beta - 1.0
            a5 = dt * (gamma / (2.0 * beta) - 1.0)
            if cached_solver is None or cached_dt is None or not np.isclose(dt, cached_dt):
                K_eff = (one_plus_alpha * K_red + a0 * M_red + one_plus_alpha * a1 * C_red).tocsr()
                cached_solver = factorize(K_eff, MatrixClass.SYMMETRIC_INDEFINITE, signature=f"sphere_impact.effective:{dt:.16g}")
                cached_solver_diagnostics = cached_solver.diagnostics()
                cached_dt = dt
                factorization_count += 1

            sphere_x_pred = sphere_position + dt * sphere_velocity + dt**2 * (0.5 - beta) * sphere_acceleration
            sphere_v_pred = sphere_velocity + dt * (1.0 - gamma) * sphere_acceleration
            contact_load = np.zeros(total_dofs, dtype=float)
            sphere_force = np.zeros(3, dtype=float)
            records: Tuple[SphereContactRecord, ...] = tuple()
            converged = False
            q_next = q_red.copy()
            v_next = v_red.copy()
            a_next = a_red.copy()
            sphere_x_next = sphere_x_pred.copy()
            sphere_v_next = sphere_v_pred.copy()
            sphere_a_next = np.zeros(3, dtype=float)
            force_change = 0.0
            penetration_change = 0.0
            use_aitken = str(config.contact_relaxation) == "aitken"
            relaxation = 1.0
            fixed_point_residual_prev: Optional[np.ndarray] = None
            for iteration in range(1, int(config.max_contact_iterations) + 1):
                load_next = base_load + contact_load
                F_red = _reduced_load(T, K, u0, load_next)
                rhs = (
                    one_plus_alpha * F_red
                    - alpha_h * F_red_prev
                    + M_red @ (a0 * q_red + a2 * v_red + a3 * a_red)
                    + one_plus_alpha * (C_red @ (a1 * q_red + a4 * v_red + a5 * a_red))
                )
                if alpha_h != 0.0:
                    rhs += alpha_h * (K_red @ q_red) + alpha_h * (C_red @ v_red)
                q_next = np.asarray(cached_solver.solve(rhs), dtype=float).reshape(-1)
                solve_count += 1
                a_next = a0 * (q_next - q_red) - a2 * v_red - a3 * a_red
                v_next = v_red + dt * ((1.0 - gamma) * a_red + gamma * a_next)
                sphere_a_next = sphere_force / float(sphere.mass)
                sphere_x_next = sphere_x_pred + beta * dt**2 * sphere_a_next
                sphere_v_next = sphere_v_pred + gamma * dt * sphere_a_next
                full_u = reconstruct_full_solution(T, q_next, u0)
                full_v = np.asarray(T @ v_next, dtype=float).reshape(-1)
                new_contact_load, new_sphere_force, new_records = assemble_sphere_contact_load_vector(
                    model,
                    sphere,
                    config,
                    sphere_x_next,
                    sphere_v_next,
                    structural_displacement=full_u,
                    structural_velocity=full_v,
                    deleted_element_ids=tuple(deleted_element_ids),
                    contact_scale_by_element=damage_scale_by_element,
                    preferred_element_ids=sticky_contact_ids,
                )
                if new_records:
                    sticky_contact_ids = tuple(int(record.element_id) for record in new_records)
                force_change = float(np.linalg.norm(new_sphere_force - sphere_force))
                penetration_change = abs(
                    (max((record.penetration for record in new_records), default=0.0))
                    - (max((record.penetration for record in records), default=0.0))
                )
                scale = max(float(np.linalg.norm(new_sphere_force)), 1.0)
                fixed_point_residual = new_sphere_force - sphere_force
                if use_aitken and fixed_point_residual_prev is not None:
                    residual_difference = fixed_point_residual - fixed_point_residual_prev
                    denominator = float(residual_difference @ residual_difference)
                    if denominator > 1.0e-30:
                        relaxation = -relaxation * float(fixed_point_residual_prev @ residual_difference) / denominator
                        relaxation = float(min(max(relaxation, 1.0e-4), 1.5))
                    else:
                        relaxation = 1.0
                fixed_point_residual_prev = fixed_point_residual.copy()
                if use_aitken and relaxation != 1.0:
                    contact_load = contact_load + relaxation * (new_contact_load - contact_load)
                    sphere_force = sphere_force + relaxation * (new_sphere_force - sphere_force)
                    records = new_records
                else:
                    contact_load, sphere_force, records = new_contact_load, new_sphere_force, new_records
                if force_change <= config.force_tolerance * scale and penetration_change <= config.penetration_tolerance:
                    converged = True
                    iteration_counts.append(iteration)
                    break
            if not converged:
                status = "contact_iteration_failed"
                stop_reason = "contact_iteration_failed"
                iteration_counts.append(int(config.max_contact_iterations))

            contact_norm = float(np.linalg.norm(sphere_force))
            if contact_norm > 0.0:
                contact_step_count += 1
                active_contact_duration += dt
                separation_elapsed = 0.0
            elif contact_step_count > 0 and separation_stop_time > 0.0:
                separation_elapsed += dt
            if records:
                max_penetration = max(max_penetration, max(float(record.penetration) for record in records))
                peak_contact_force = max(peak_contact_force, max(float(record.normal_force) for record in records))
            load_next = base_load + contact_load
            impulse += 0.5 * (load_prev + load_next) * dt
            sphere_impulse += 0.5 * (sphere_force_prev + sphere_force) * dt
            load_prev = load_next
            F_red_prev = F_red
            sphere_force_prev = sphere_force.copy()
            q_red, v_red, a_red = q_next, v_next, a_next
            sphere_position, sphere_velocity, sphere_acceleration = sphere_x_next, sphere_v_next, sphere_a_next
            last_records = records
            last_sphere_force = sphere_force.copy()
            last_step_diag = {
                "step_index": int(step_index),
                "substep_index": int(substep),
                "time": float(sub_time),
                "contact_iterations": int(iteration_counts[-1]),
                "contact_relaxation_factor": float(relaxation),
                "force_change_norm": float(force_change),
                "penetration_change": float(penetration_change),
                "active_element_ids": [int(record.element_id) for record in records],
                "max_penetration": float(max((record.penetration for record in records), default=0.0)),
                "status": "converged" if converged else "contact_iteration_failed",
            }
            if fracture_config is not None and records and status != "contact_iteration_failed":
                new_fracture_records, substep_utilization = _detect_impact_fracture_records(
                    model,
                    records,
                    fracture_config,
                    sphere,
                    tuple(deleted_element_ids),
                    step_index=int(step_index),
                    time_value=float(sub_time),
                )
                max_fracture_utilization = max(max_fracture_utilization, substep_utilization)
                if new_fracture_records:
                    fracture_records.extend(new_fracture_records)
                    deleted_element_ids.update(record.element_id for record in new_fracture_records)
                    fracture_deleted_element_ids.update(record.element_id for record in new_fracture_records)
                    filtered_base = filtered_load_case_for_deleted_elements(base_load_case, deleted_element_ids)
                    base_load, base_load_info = assemble_load_vector(model, filtered_base)
                    if linear_matrix_terms is None:
                        linear_matrix_terms = _linear_element_matrix_terms(model)
                    fracture_scales = {int(element_id): float(scale) for element_id, scale in damage_scale_by_element.items()}
                    for element_id in deleted_element_ids:
                        fracture_scales[int(element_id)] = min(
                            fracture_scales.get(int(element_id), 1.0),
                            float(fracture_config.residual_stiffness_fraction),
                        )
                    K, M = _assemble_damaged_linear_matrices(model, fracture_scales, cached_terms=linear_matrix_terms)
                    eroded_matrix_rebuild_count += 1
                    K_red = (T.T @ K @ T).tocsr()
                    M_red = (T.T @ M @ T).tocsr()
                    C_red = (transient_config.rayleigh_alpha * M_red + transient_config.rayleigh_beta * K_red).tocsr()
                    cached_solver = None
                    cached_dt = None
                    mass_handle = factorize(
                        M_red,
                        MatrixClass.SYMMETRIC_SEMIDEFINITE,
                        signature=f"sphere_impact.fracture_mass:{len(deleted_element_ids)}",
                    )
                    current_red_load = _reduced_load(T, K, u0, base_load + contact_load)
                    F_red_prev = current_red_load
                    a_red = np.asarray(mass_handle.solve(current_red_load - C_red @ v_red - K_red @ q_red), dtype=float).reshape(-1)
                    scoped_total = _contact_erodible_element_count(model)
                    deleted_fraction = len(deleted_element_ids) / max(scoped_total, 1)
                    last_step_diag["fracture_new_deleted_element_ids"] = [record.element_id for record in new_fracture_records]
                    last_step_diag["fracture_deleted_fraction"] = float(deleted_fraction)
                    last_step_diag["fracture_max_utilization"] = float(max_fracture_utilization)
                    if deleted_fraction > fracture_config.max_deleted_fraction + 1.0e-12:
                        status = "max_deleted_fraction_reached"
                        stop_reason = "impact_fracture_max_deleted_fraction_reached"
                        if not fracture_limit_warning_emitted:
                            warnings.append("IMPACT_FRACTURE002: maximum deleted contact-target fraction reached; transient stopped at last eroded state.")
                            fracture_limit_warning_emitted = True
            if damage_config is not None and records and status not in {"contact_iteration_failed", "max_deleted_fraction_reached"}:
                new_damage_deletions, substep_damage_utilization, damage_diags, damage_changed = _update_impact_damage_states(
                    model,
                    records,
                    damage_config,
                    sphere,
                    impact_damage_states,
                    tuple(deleted_element_ids),
                    step_index=int(step_index),
                    time_value=float(sub_time),
                    dt=float(dt),
                )
                max_damage_utilization = max(max_damage_utilization, substep_damage_utilization)
                if damage_diags:
                    damage_state_update_count += len(damage_diags)
                if damage_diags:
                    last_step_diag["impact_damage"] = list(damage_diags)
                    last_step_diag["impact_damage_max_utilization"] = float(max_damage_utilization)
                if new_damage_deletions:
                    damage_records.extend(new_damage_deletions)
                    deleted_element_ids.update(record.element_id for record in new_damage_deletions)
                    damage_deleted_element_ids.update(record.element_id for record in new_damage_deletions)
                    for deleted_id in deleted_element_ids:
                        state = impact_damage_states.setdefault(int(deleted_id), {})
                        state["damage"] = max(float(state.get("damage", 0.0)), float(damage_config.delete_at))
                        state["scale"] = float(damage_config.residual_stiffness_fraction)
                    last_step_diag["impact_damage_new_deleted_element_ids"] = [record.element_id for record in new_damage_deletions]
                    if not damage_deletion_warning_emitted:
                        warnings.append("IMPACT_DAMAGE002: one or more shell elements were eroded by impact damage.")
                        damage_deletion_warning_emitted = True
                    damage_changed = True
                if damage_changed:
                    damage_scale_by_element = {
                        int(element_id): _impact_damage_scale(float(state.get("damage", 0.0)), damage_config)
                        for element_id, state in impact_damage_states.items()
                    }
                    for element_id in deleted_element_ids:
                        damage_scale_by_element[int(element_id)] = min(
                            damage_scale_by_element.get(int(element_id), 1.0),
                            float(damage_config.residual_stiffness_fraction),
                        )
                    filtered_base = filtered_load_case_for_deleted_elements(base_load_case, deleted_element_ids)
                    base_load, base_load_info = assemble_load_vector(model, filtered_base)
                    if linear_matrix_terms is None:
                        linear_matrix_terms = _linear_element_matrix_terms(model)
                    K, M = _assemble_damaged_linear_matrices(model, damage_scale_by_element, cached_terms=linear_matrix_terms)
                    eroded_matrix_rebuild_count += 1
                    K_red = (T.T @ K @ T).tocsr()
                    M_red = (T.T @ M @ T).tocsr()
                    C_red = (transient_config.rayleigh_alpha * M_red + transient_config.rayleigh_beta * K_red).tocsr()
                    cached_solver = None
                    cached_dt = None
                    mass_handle = factorize(
                        M_red,
                        MatrixClass.SYMMETRIC_SEMIDEFINITE,
                        signature=f"sphere_impact.damage_mass:{len(damage_scale_by_element)}:{len(deleted_element_ids)}",
                    )
                    current_red_load = _reduced_load(T, K, u0, base_load + contact_load)
                    F_red_prev = current_red_load
                    a_red = np.asarray(mass_handle.solve(current_red_load - C_red @ v_red - K_red @ q_red), dtype=float).reshape(-1)
                    scoped_total = sum(1 for element in model.mesh.elements.values() if element_fracture_category(element) == "shell")
                    deleted_fraction = len(deleted_element_ids) / max(scoped_total, 1)
                    last_step_diag["impact_damage_deleted_fraction"] = float(deleted_fraction)
                    if deleted_fraction > float(damage_config.max_deleted_fraction) + 1.0e-12:
                        status = "max_deleted_fraction_reached"
                        stop_reason = "impact_damage_max_deleted_fraction_reached"
                        if not damage_limit_warning_emitted:
                            warnings.append(
                                "IMPACT_DAMAGE003: maximum configured shell damage deletion fraction reached; "
                                "transient stopped at last eroded state."
                            )
                            damage_limit_warning_emitted = True
            step_diagnostics.append(last_step_diag)
            if status in {"contact_iteration_failed", "max_deleted_fraction_reached"}:
                break
            if status == "completed" and contact_step_count > 0 and separation_stop_time > 0.0 and separation_elapsed >= separation_stop_time:
                stop_reason = "completed_after_contact_separation"
                stop_after_separation = True
                break

        save_time = float(last_step_diag.get("time", target_time)) if last_step_diag else target_time
        if step_index % int(transient_config.save_every) == 0 or step_index == len(times) - 1 or stop_after_separation:
            save_state(save_time, q_red, v_red, a_red, sphere_position, sphere_velocity, sphere_acceleration, last_sphere_force, last_records)
        if status in {"contact_iteration_failed", "max_deleted_fraction_reached"} or stop_after_separation:
            break

    if contact_step_count == 0 and status == "completed":
        status = "no_contact"
        stop_reason = "no_contact"
    impulse_resultant = load_vector_resultant(model, impulse)
    total_energy = np.asarray(energy_kinetic, dtype=float) + np.asarray(energy_strain, dtype=float) + np.asarray(sphere_kinetic, dtype=float)
    nonzero_energy = total_energy[np.abs(total_energy) > 1.0e-30]
    energy_drift = 0.0
    if nonzero_energy.size:
        energy_drift = float((np.max(nonzero_energy) - np.min(nonzero_energy)) / max(abs(nonzero_energy[0]), 1.0e-30))
    max_penetration_ratio = float(max_penetration) / max(float(sphere.radius), 1.0e-12)
    contact_duration = float(active_contact_duration)
    initial_sphere_velocity = sphere.travel_velocity if float(times[0]) >= sphere.t_start else np.zeros(3, dtype=float)
    if saved_sphere_v:
        final_sphere_velocity = np.asarray(saved_sphere_v[-1], dtype=float)
    else:
        final_sphere_velocity = sphere_velocity
    sphere_momentum_change = float(sphere.mass) * (final_sphere_velocity - initial_sphere_velocity)
    sphere_momentum_balance_error = float(np.linalg.norm(sphere_impulse - sphere_momentum_change))

    history_width = total_dofs if history_storage_mode == "full" else int(0 if history_dof_indices is None else len(history_dof_indices))
    saved_u_array = np.vstack(saved_u) if saved_u else np.zeros((0, history_width), dtype=float)
    saved_v_array = np.vstack(saved_v) if saved_v else np.zeros((0, history_width), dtype=float)
    saved_a_array = np.vstack(saved_a) if saved_a else np.zeros((0, history_width), dtype=float)
    node_histories: Dict[int, np.ndarray] = {}
    for node_id in output_node_ids:
        values = node_history_values.get(int(node_id), [])
        node_histories[int(node_id)] = np.vstack(values) if values else np.zeros((0, 6), dtype=float)

    recovery_memory = estimate_model_memory(
        model,
        transient_saved_steps=len(saved_times),
        store_full_history=history_storage_mode == "full",
        recovery_config=recovery,
    )
    enforce_memory_limit(recovery_memory, transient_config.resource_config, context="solve_transient_sphere_impact.recovery")
    policy_metadata = recovery_metadata(recovery, transient_config.resource_config, recovery_memory)
    assembly_info = {"stiffness": stiffness_info, "mass": mass_info, "load": base_load_info}
    result_case = make_result_case(
        name=f"sphere_impact:{sphere.name}",
        analysis_type="sphere_impact_transient",
        load_cases=() if base_load_case is None else (base_load_case,),
        assembly_info=assembly_info,
        solver_info={"backend": cached_solver_diagnostics, "convergence_info": {"status": status, "stop_reason": stop_reason}},
        recovery={
            "displacement_history": True,
            "velocity_history": True,
            "acceleration_history": True,
            "sphere_history": True,
            "contact_history": bool(config.save_contact_history),
            "history_storage_mode": history_storage_mode,
            **policy_metadata["recovery"],
        },
        settings={
            "dt": transient_config.dt,
            "t_end": transient_config.t_end,
            "beta": beta,
            "gamma": gamma,
            "hht_alpha": alpha_h,
            "rayleigh_alpha": transient_config.rayleigh_alpha,
            "rayleigh_beta": transient_config.rayleigh_beta,
            "sphere": {
                "name": sphere.name,
                "radius": float(sphere.radius),
                "mass": float(sphere.mass),
                "start_point": sphere.initial_position.tolist(),
                "travel_direction": sphere.direction_unit.tolist(),
                "speed": float(sphere.speed),
                "t_start": float(sphere.t_start),
            },
            "contact": config.to_dict(),
            "fracture": None if fracture_config is None else fracture_config.to_dict(),
            "damage": None if damage_config is None else damage_config.to_dict(),
        },
        metadata={
            "resources": policy_metadata.get("resources"),
            "memory_estimate": policy_metadata.get("memory_estimate"),
            "contact_scope": "single rigid sphere to shell midsurface/thickness-offset surface; frictionless normal penalty",
            "solution_control": "implicit_newmark_time_domain",
            "arc_length_applicability": "not_applicable_to_dynamic_impact",
            "target_element_types": ["S3", "S4", "S6", "S8", "S8R"],
            "beam_response": "beams respond only through existing shell-beam coupling; direct beam contact is unsupported",
            "impact_fracture_scope": (
                "disabled"
                if fracture_config is None
                else "contact-observable shell erosion after converged contact substeps"
            ),
            "impact_damage_scope": (
                "disabled"
                if damage_config is None
                else "engineering capacity-based shell softening/erosion after converged contact substeps"
            ),
            "erosion_scope": (
                "disabled"
                if not deleted_element_ids and not damage_scale_by_element
                else "residual stiffness/mass/contact scaling; no node, MPC, beam-shell coupling or topology deletion"
            ),
            "solver_convergence": {"status": status, "stop_reason": stop_reason},
            "verification_gate": "contact",
        },
    ).to_dict()
    impact_fracture_summary = fracture_summary(
        model,
        fracture_config,
        fracture_records,
        fracture_deleted_element_ids,
        max_utilization=max_fracture_utilization,
        warnings=_warnings_with_prefixes(warnings, ("IMPACT_FRACTURE",)) if fracture_config is not None else (),
    )
    impact_damage_summary = _impact_damage_summary(
        model,
        damage_config,
        impact_damage_states,
        damage_deleted_element_ids,
        deletion_records=damage_records,
        warnings=_warnings_with_prefixes(warnings, ("IMPACT_DAMAGE",)) if damage_config is not None else (),
    )
    erosion_summary = {
        "all_eroded_element_ids": sorted(int(element_id) for element_id in deleted_element_ids),
        "fracture_triggered_element_ids": sorted(int(element_id) for element_id in fracture_deleted_element_ids),
        "damage_triggered_element_ids": sorted(int(element_id) for element_id in damage_deleted_element_ids),
        "active_softened_element_ids": sorted(
            int(element_id)
            for element_id, scale in damage_scale_by_element.items()
            if int(element_id) not in deleted_element_ids and float(scale) < 1.0
        ),
        "residual_stiffness_model": "element stiffness/mass scaling; topology, nodes, MPCs and beam-shell couplings retained",
    }
    diagnostics = {
        "method": "newmark_sphere_penalty_contact",
        "solution_control": "implicit_newmark_time_domain",
        "arc_length_applicability": "not_applicable_to_dynamic_impact",
        "status": status,
        "stop_reason": stop_reason,
        "warnings": warnings,
        "num_steps": max(int(len(times) - 1), 0),
        "num_substeps": int(total_substep_count),
        "event_substep_count": int(event_substep_count),
        "num_saved_steps": len(saved_times),
        "num_reduced_dofs": int(K_red.shape[0]),
        "contact_step_count": int(contact_step_count),
        "active_contact_duration": float(contact_duration),
        "separation_stop_time": float(separation_stop_time),
        "post_contact_separation_time": float(separation_elapsed if contact_step_count > 0 else 0.0),
        "max_penetration_ratio": float(max_penetration_ratio),
        "sphere_momentum_balance_error": float(sphere_momentum_balance_error),
        "iteration_counts": iteration_counts,
        "contact_step_diagnostics": step_diagnostics,
        "factorization_count": int(factorization_count),
        "solve_count": int(solve_count),
        "eroded_matrix_rebuild_count": int(eroded_matrix_rebuild_count),
        "damage_state_update_count": int(damage_state_update_count),
        "linear_matrix_terms_cached": linear_matrix_terms is not None,
        "initial_mass_factorization": mass_handle.diagnostics(),
        "effective_stiffness_factorization": cached_solver_diagnostics,
        "constraint_info": constraint_info,
        "stiffness": stiffness_info,
        "mass": mass_info,
        "base_load": base_load_info,
        "contact_config": config.to_dict(),
        "impact_fracture_summary": impact_fracture_summary,
        "impact_damage_summary": impact_damage_summary,
        "erosion_summary": erosion_summary,
        "contact_validation": validation.to_dict(),
        "sphere": result_case["analysis_case"]["settings"]["sphere"],
        "kinetic_energy": energy_kinetic,
        "strain_energy": energy_strain,
        "sphere_kinetic_energy": sphere_kinetic,
        "max_relative_energy_drift": energy_drift,
        "result_case": result_case,
    }
    return SphereImpactResult(
        times=np.asarray(saved_times, dtype=float),
        displacements=saved_u_array,
        velocities=saved_v_array,
        accelerations=saved_a_array,
        node_histories=node_histories,
        sphere_positions=np.vstack(saved_sphere_x) if saved_sphere_x else np.zeros((0, 3), dtype=float),
        sphere_velocities=np.vstack(saved_sphere_v) if saved_sphere_v else np.zeros((0, 3), dtype=float),
        sphere_accelerations=np.vstack(saved_sphere_a) if saved_sphere_a else np.zeros((0, 3), dtype=float),
        contact_force_history=np.vstack(saved_contact_force) if saved_contact_force else np.zeros((0, 3), dtype=float),
        active_contact_history=tuple(saved_contacts),
        load_impulse=impulse,
        force_impulse=impulse_resultant.force,
        moment_impulse=impulse_resultant.moment,
        sphere_impulse=sphere_impulse,
        max_penetration=float(max_penetration),
        max_penetration_ratio=float(max_penetration_ratio),
        peak_contact_force=float(peak_contact_force),
        contact_duration=float(contact_duration),
        sphere_momentum_balance_error=float(sphere_momentum_balance_error),
        peak_displacement=float(peak_displacement),
        peak_displacement_node=peak_displacement_node,
        status=status,
        diagnostics=diagnostics,
        result_case=result_case,
    )


def _verification_contact_panel(stiffener: bool = False) -> "FEModel":
    from .elements import BeamElement, CoupledBeamShellElement
    from .fe_core import FEModel

    model = FEModel("contact_verification_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    model.add_node(1, 0.0, 0.0, 0.0)
    model.add_node(2, 1.0, 0.0, 0.0)
    model.add_node(3, 1.0, 1.0, 0.0)
    model.add_node(4, 0.0, 1.0, 0.0)
    model.add_element(1, ShellElement(1, [1, 2, 3, 4], "soft", thickness=0.05))
    model.add_boundary_condition(
        BoundaryCondition(
            "restrain_shell_nonimpact_modes",
            [1, 2, 3, 4],
            {"ux": 0.0, "uy": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0},
        )
    )
    if stiffener:
        section = {"area": 0.05, "Iy": 1.0e-3, "Iz": 1.0e-3, "J": 1.0e-3}
        model.add_node(10, 0.5, 0.0, 0.05)
        model.add_node(11, 0.5, 1.0, 0.05)
        model.add_element(10, BeamElement(10, [10, 11], "soft", section))
        model.add_element(20, CoupledBeamShellElement(20, beam_node_id=10, shell_node_id=1, material_name="soft"))
        model.add_element(21, CoupledBeamShellElement(21, beam_node_id=11, shell_node_id=4, material_name="soft"))
    return model


def _two_shell_contact_verification_panel() -> "FEModel":
    from .fe_core import FEModel

    model = FEModel("two_shell_contact_verification_panel")
    model.add_material("soft", 1.0e5, 0.3, density=20.0)
    for node_id, xyz in {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (2.0, 0.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (1.0, 1.0, 0.0),
        6: (2.0, 1.0, 0.0),
    }.items():
        model.add_node(node_id, *xyz)
    model.add_element(1, ShellElement(1, [1, 2, 5, 4], "soft", thickness=0.05))
    model.add_element(2, ShellElement(2, [2, 3, 6, 5], "soft", thickness=0.05))
    return model


def contact_verification_metrics() -> Dict[str, Dict[str, Any]]:
    """Return compact metrics for the rigid-sphere contact verification ledger."""

    metrics: Dict[str, Dict[str, Any]] = {}
    no_contact = solve_transient_sphere_impact(
        _verification_contact_panel(),
        TransientConfig(dt=0.01, t_end=0.03),
        RigidSphereImpact("miss", radius=0.1, mass=2.0, start_point=(0.5, 0.5, 2.0), travel_direction=(1.0, 0.0, 0.0), speed=3.0),
        SphereContactConfig(penalty_stiffness=1000.0),
    )
    metrics["CONTACT-001"] = {
        "status": no_contact.status,
        "trajectory_error": float(np.linalg.norm(no_contact.sphere_positions[-1] - np.array([0.59, 0.5, 2.0]))),
        "peak_contact_force": float(no_contact.peak_contact_force),
    }

    sphere = RigidSphereImpact("unit", radius=0.2, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    config = SphereContactConfig(penalty_stiffness=1000.0)
    contact_model = _verification_contact_panel()
    vector, sphere_force, records = assemble_sphere_contact_load_vector(
        contact_model,
        sphere,
        config,
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    resultant = load_vector_resultant(contact_model, vector)
    expected_force = 100.0
    metrics["CONTACT-002"] = {
        "normal_force": float(records[0].normal_force if records else 0.0),
        "expected_normal_force": expected_force,
        "relative_error": abs(float(records[0].normal_force if records else 0.0) - expected_force) / expected_force,
    }
    metrics["CONTACT-003"] = {
        "sphere_force": sphere_force.tolist(),
        "structure_force_resultant": resultant.force.tolist(),
        "balance_error": float(np.linalg.norm(sphere_force + resultant.force)),
        "moment_norm": float(resultant.moment_norm),
    }

    hit = solve_transient_sphere_impact(
        _verification_contact_panel(),
        TransientConfig(dt=0.0025, t_end=0.12, output_nodes=[1]),
        RigidSphereImpact("hit", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0),
        SphereContactConfig(penalty_stiffness=4000.0, contact_damping=0.0, max_contact_iterations=40),
    )
    momentum_change = hit.sphere_velocities[-1] - np.array([0.0, 0.0, -2.0])
    metrics["CONTACT-004"] = {
        "impulse_error": float(np.linalg.norm(hit.sphere_impulse - momentum_change)),
        "impulse_norm": float(np.linalg.norm(hit.sphere_impulse)),
        "status": hit.status,
    }
    metrics["CONTACT-005"] = {
        "status": hit.status,
        "peak_contact_force": float(hit.peak_contact_force),
        "max_penetration": float(hit.max_penetration),
        "peak_displacement": float(hit.peak_displacement),
        "contact_step_count": int(hit.diagnostics["contact_step_count"]),
    }
    stiffened = solve_transient_sphere_impact(
        _verification_contact_panel(stiffener=True),
        TransientConfig(dt=0.0025, t_end=0.11, output_nodes=[10, 11]),
        RigidSphereImpact("stiffened", radius=0.1, mass=1.0, start_point=(0.25, 0.25, 0.22), travel_direction=(0.0, 0.0, -1.0), speed=2.0),
        SphereContactConfig(penalty_stiffness=3000.0, max_contact_iterations=40),
    )
    metrics["CONTACT-006"] = {
        "status": stiffened.status,
        "peak_contact_force": float(stiffened.peak_contact_force),
        "beam_node_10_peak_uz": float(np.max(np.abs(stiffened.node_histories[10][:, 2]))),
        "beam_node_11_peak_uz": float(np.max(np.abs(stiffened.node_histories[11][:, 2]))),
    }
    projection_cases = [
        (_verification_contact_panel(), np.array([0.5, 0.5, 0.08]), "face"),
        (_verification_contact_panel(), np.array([1.0, 0.5, 0.08]), "edge"),
        (_verification_contact_panel(), np.array([1.0, 1.0, 0.08]), "corner"),
    ]
    projection_ok = True
    projection_rows = []
    for model, position, expected in projection_cases:
        sphere_case = RigidSphereImpact("projection", radius=0.1, mass=1.0, start_point=position, travel_direction=(0.0, 0.0, -1.0), speed=0.0)
        _vector, _force, case_records = assemble_sphere_contact_load_vector(
            model,
            sphere_case,
            SphereContactConfig(penalty_stiffness=1000.0),
            sphere_position=position,
            sphere_velocity=np.zeros(3),
        )
        actual = case_records[0].contact_classification if case_records else "none"
        projection_ok = projection_ok and actual == expected
        projection_rows.append({"expected": expected, "actual": actual})
    metrics["CONTACT-007"] = {"projection_classification_ok": projection_ok, "rows": projection_rows}

    top_sphere = RigidSphereImpact("surface", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    _mid_vector, _mid_force, mid_records = assemble_sphere_contact_load_vector(
        _verification_contact_panel(),
        top_sphere,
        SphereContactConfig(penalty_stiffness=1000.0, contact_surface="midsurface"),
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    _top_vector, _top_force, top_records = assemble_sphere_contact_load_vector(
        _verification_contact_panel(),
        top_sphere,
        SphereContactConfig(penalty_stiffness=1000.0, contact_surface="top"),
        sphere_position=np.array([0.5, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    metrics["CONTACT-008"] = {
        "midsurface_records": len(mid_records),
        "top_penetration": float(top_records[0].penetration if top_records else 0.0),
        "expected_top_penetration": 0.025,
    }

    two_shell = _two_shell_contact_verification_panel()
    shared = RigidSphereImpact("shared_edge", radius=0.2, mass=1.0, start_point=(1.0, 0.5, 0.1), travel_direction=(0.0, 0.0, -1.0), speed=0.0)
    _shared_vector, shared_force, shared_records = assemble_sphere_contact_load_vector(
        two_shell,
        shared,
        SphereContactConfig(penalty_stiffness=1000.0, max_active_contacts=1),
        sphere_position=np.array([1.0, 0.5, 0.1]),
        sphere_velocity=np.zeros(3),
    )
    metrics["CONTACT-009"] = {
        "active_records": len(shared_records),
        "force_norm": float(np.linalg.norm(shared_force)),
        "expected_force_norm": 100.0,
    }

    auto_sphere = RigidSphereImpact("auto", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.25), travel_direction=(0.0, 0.0, -1.0), speed=2.0)
    auto = solve_transient_sphere_impact(
        _verification_contact_panel(),
        TransientConfig(dt=0.0015, t_end=0.11),
        auto_sphere,
        SphereContactConfig(target_penetration_fraction=0.05, auto_penalty_safety_factor=5.0, max_contact_iterations=40),
    )
    metrics["CONTACT-010"] = {
        "status": auto.status,
        "max_penetration_ratio": float(auto.max_penetration_ratio),
        "resolved_penalty": float(auto.diagnostics["contact_config"]["penalty_stiffness"]),
    }

    fast = solve_transient_sphere_impact(
        _verification_contact_panel(),
        TransientConfig(dt=0.05, t_end=0.05),
        RigidSphereImpact("fast", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 0.45), travel_direction=(0.0, 0.0, -1.0), speed=20.0),
        SphereContactConfig(penalty_stiffness=4000.0, max_event_substeps=64, max_contact_iterations=40),
    )
    metrics["CONTACT-011"] = {
        "status": fast.status,
        "event_substep_count": int(fast.diagnostics["event_substep_count"]),
        "peak_contact_force": float(fast.peak_contact_force),
    }

    invalid = _verification_contact_panel()
    invalid.materials["soft"].density = 0.0
    invalid_report = validate_contact_configuration(
        invalid,
        RigidSphereImpact("bad_config", radius=0.1, mass=1.0, start_point=(0.5, 0.5, 1.0), travel_direction=(0.0, 0.0, -1.0), speed=10.0),
        SphereContactConfig(penalty_stiffness=1000.0, max_event_substeps=1),
        TransientConfig(dt=0.1, t_end=0.1),
    )
    metrics["CONTACT-012"] = {
        "validation_status": invalid_report.status,
        "issue_codes": [issue.code for issue in invalid_report.issues],
    }
    return metrics
