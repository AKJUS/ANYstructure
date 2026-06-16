"""Rule-aware geometric imperfection helpers for nonlinear capacity checks.

Imperfections are represented as stress-free reference-geometry offsets.  In
other words, applying an imperfection modifies nodal coordinates before the
nonlinear solve; zero displacement in the imperfect model has zero strain and
zero internal force.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .buckling import BucklingResult
    from .fe_core import FEModel
    from .nonlinear_static import NonlinearStaticResult


def _unit(vector: Sequence[float], fallback: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(-1)
    if value.size < 3:
        padded = np.zeros(3, dtype=float)
        padded[: value.size] = value
        value = padded
    norm = float(np.linalg.norm(value[:3]))
    if norm <= 1.0e-14:
        return _unit(fallback)
    return value[:3] / norm


def _node_coords(model: "FEModel", node_ids: Iterable[int]) -> Dict[int, np.ndarray]:
    coords: Dict[int, np.ndarray] = {}
    for node_id in node_ids:
        node = model.mesh.get_node(int(node_id))
        if node is None:
            raise ValueError(f"Node {node_id} not found")
        coords[int(node_id)] = node.coords()
    return coords


def _invalidate_element_caches(model: "FEModel") -> None:
    for element in model.mesh.elements.values():
        for name in ("_stiffness_matrix", "_mass_matrix", "_internal_forces", "_nl_cache"):
            if hasattr(element, name):
                setattr(element, name, None)
    if hasattr(model.mesh, "_sparsity_cache"):
        model.mesh._sparsity_cache = {}


@dataclass(frozen=True)
class ImperfectionField:
    """Nodal reference-coordinate offsets in metres."""

    offsets: Mapping[int, Sequence[float]]
    name: str = "imperfection"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_arrays(self) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        for node_id, offset in self.offsets.items():
            vector = np.asarray(offset, dtype=float).reshape(-1)
            if vector.size < 3:
                padded = np.zeros(3, dtype=float)
                padded[: vector.size] = vector
                vector = padded
            result[int(node_id)] = vector[:3].copy()
        return result

    @property
    def max_offset(self) -> float:
        return max((float(np.linalg.norm(offset)) for offset in self.as_arrays().values()), default=0.0)

    def combine(self, *others: "ImperfectionField", name: Optional[str] = None) -> "ImperfectionField":
        offsets = self.as_arrays()
        metadata: Dict[str, Any] = {"components": [self.name]}
        for other in others:
            metadata["components"].append(other.name)
            for node_id, offset in other.as_arrays().items():
                offsets[node_id] = offsets.get(node_id, np.zeros(3, dtype=float)) + offset
        return ImperfectionField(offsets, name=name or "+".join(metadata["components"]), metadata=metadata)


@dataclass(frozen=True)
class CompositeImperfection:
    """Combination of local/global imperfection fields."""

    components: Sequence[Any]
    name: str = "composite_imperfection"

    def to_field(self, model: "FEModel") -> ImperfectionField:
        fields = [to_imperfection_field(model, component) for component in self.components]
        if not fields:
            return ImperfectionField({}, name=self.name)
        return fields[0].combine(*fields[1:], name=self.name)


@dataclass(frozen=True)
class EigenmodeImperfection:
    """Imperfection scaled from a linear buckling mode."""

    buckling_result: "BucklingResult"
    mode_number: int = 1
    amplitude: float = 0.0
    dof_filter: str = "translations"
    name: str = "eigenmode_imperfection"

    def to_field(self, model: "FEModel") -> ImperfectionField:
        return imperfection_from_buckling_mode(
            model,
            self.buckling_result,
            self.mode_number,
            self.amplitude,
            dof_filter=self.dof_filter,
            name=self.name,
        )


@dataclass(frozen=True)
class StandardImperfection:
    """Deterministic DNV-style imperfection pattern."""

    kind: str
    node_ids: Sequence[int]
    amplitude: Optional[float] = None
    direction: Sequence[float] = (0.0, 0.0, 1.0)
    axes: Sequence[int] = (0, 1)
    waves: Tuple[int, int] = (1, 1)
    name: str = "standard_imperfection"

    def to_field(self, model: "FEModel") -> ImperfectionField:
        kind = self.kind.lower()
        if kind in {"member_bow", "bow"}:
            return standard_member_bow(model, self.node_ids, self.amplitude, self.direction, name=self.name)
        if kind in {"plate_mode", "plate_half_wave", "plate"}:
            return standard_plate_mode(model, self.node_ids, self.amplitude, self.direction, self.axes, self.waves, name=self.name)
        if kind in {"flange_twist", "twist"}:
            return standard_flange_twist(model, self.node_ids, self.amplitude if self.amplitude is not None else 0.02, self.direction, name=self.name)
        raise ValueError(f"Unknown standard imperfection kind {self.kind!r}")


@dataclass(frozen=True)
class ImperfectionCalibrationResult:
    """Result from binary-search equivalent imperfection calibration."""

    amplitude: float
    capacity: float
    iterations: int
    converged: bool
    history: Tuple[Dict[str, float], ...]
    result: Optional["NonlinearStaticResult"] = None


def to_imperfection_field(model: "FEModel", imperfection: Any) -> ImperfectionField:
    if imperfection is None:
        return ImperfectionField({})
    if isinstance(imperfection, ImperfectionField):
        return imperfection
    converter = getattr(imperfection, "to_field", None)
    if converter is not None:
        return converter(model)
    if isinstance(imperfection, Mapping):
        return ImperfectionField(imperfection)
    raise TypeError(f"Cannot convert {type(imperfection).__name__} to ImperfectionField")


def apply_imperfection(model: "FEModel", imperfection: Any, copy_model: bool = True) -> "FEModel":
    """Apply a stress-free geometric imperfection to a model."""
    target = copy.deepcopy(model) if copy_model else model
    field = to_imperfection_field(target, imperfection)
    for node_id, offset in field.as_arrays().items():
        node = target.mesh.get_node(node_id)
        if node is None:
            raise ValueError(f"Imperfection references missing node {node_id}")
        node.x += float(offset[0])
        node.y += float(offset[1])
        node.z += float(offset[2])
    _invalidate_element_caches(target)
    if not hasattr(target, "imperfection_metadata"):
        target.imperfection_metadata = []
    target.imperfection_metadata.append(
        {"name": field.name, "max_offset": field.max_offset, "metadata": dict(field.metadata)}
    )
    return target


def imperfection_from_buckling_mode(
    model: "FEModel",
    buckling_result: "BucklingResult",
    mode_number: int,
    amplitude: float,
    dof_filter: str = "translations",
    name: str = "eigenmode_imperfection",
) -> ImperfectionField:
    """Scale a buckling mode so the maximum nodal offset equals amplitude."""
    mode = next((item for item in buckling_result.modes if int(item.mode_number) == int(mode_number)), None)
    if mode is None:
        raise ValueError(f"Buckling mode {mode_number} not available")
    shape = np.asarray(mode.mode_shape, dtype=float).reshape(-1)
    offsets: Dict[int, np.ndarray] = {}
    filter_name = dof_filter.lower()
    for node_id, node in model.mesh.nodes.items():
        if filter_name in {"translations", "translation", "xyz"}:
            vector = shape[node.dofs[:3]]
        elif filter_name in {"z", "uz", "out_of_plane"}:
            vector = np.array([0.0, 0.0, shape[node.dofs[2]]], dtype=float)
        else:
            raise ValueError("dof_filter must be 'translations' or 'out_of_plane'")
        offsets[int(node_id)] = np.asarray(vector, dtype=float)
    max_norm = max((float(np.linalg.norm(value)) for value in offsets.values()), default=0.0)
    if max_norm <= 0.0:
        raise ValueError("Selected buckling mode has zero translational amplitude")
    scale = float(amplitude) / max_norm
    return ImperfectionField(
        {node_id: scale * value for node_id, value in offsets.items()},
        name=name,
        metadata={"source": "buckling_mode", "mode_number": int(mode_number), "amplitude": float(amplitude)},
    )


def standard_member_bow(
    model: "FEModel",
    member_nodes: Sequence[int],
    amplitude: Optional[float] = None,
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    name: str = "member_bow",
) -> ImperfectionField:
    """Half-sine member bow, defaulting to DNV-style L/300 amplitude."""
    coords = _node_coords(model, member_nodes)
    ordered_ids = list(coords)
    start = coords[ordered_ids[0]]
    end = coords[ordered_ids[-1]]
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise ValueError("member_nodes must span a non-zero length")
    axis /= length
    direction_vector = _unit(direction)
    direction_vector = direction_vector - float(direction_vector @ axis) * axis
    direction_vector = _unit(direction_vector)
    amp = float(length / 300.0 if amplitude is None else amplitude)
    offsets: Dict[int, np.ndarray] = {}
    for node_id, coord in coords.items():
        s = float((coord - start) @ axis) / length
        offsets[node_id] = amp * np.sin(np.pi * np.clip(s, 0.0, 1.0)) * direction_vector
    return ImperfectionField(offsets, name=name, metadata={"kind": "member_bow", "amplitude": amp, "length": length})


def _node_ids_from_region(model: "FEModel", shell_region: Sequence[int]) -> Tuple[int, ...]:
    node_ids = set()
    for item_id in shell_region:
        element = model.mesh.get_element(int(item_id))
        if element is not None and hasattr(element, "node_ids"):
            node_ids.update(int(node_id) for node_id in element.node_ids)
        elif model.mesh.get_node(int(item_id)) is not None:
            node_ids.add(int(item_id))
        else:
            raise ValueError(f"Region id {item_id} is neither a node nor an element")
    return tuple(sorted(node_ids))


def standard_plate_mode(
    model: "FEModel",
    shell_region: Sequence[int],
    amplitude: Optional[float] = None,
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    axes: Sequence[int] = (0, 1),
    waves: Tuple[int, int] = (1, 1),
    name: str = "plate_mode",
) -> ImperfectionField:
    """Sinusoidal plate imperfection, defaulting to s/200 amplitude."""
    node_ids = _node_ids_from_region(model, shell_region)
    coords = _node_coords(model, node_ids)
    ax0, ax1 = int(axes[0]), int(axes[1])
    values = np.asarray([coord for coord in coords.values()], dtype=float)
    lo = values[:, [ax0, ax1]].min(axis=0)
    hi = values[:, [ax0, ax1]].max(axis=0)
    spans = np.maximum(hi - lo, 1.0e-14)
    amp = float(min(spans) / 200.0 if amplitude is None else amplitude)
    direction_vector = _unit(direction)
    wx, wy = int(waves[0]), int(waves[1])
    offsets: Dict[int, np.ndarray] = {}
    for node_id, coord in coords.items():
        sx = (coord[ax0] - lo[0]) / spans[0]
        sy = (coord[ax1] - lo[1]) / spans[1]
        shape = np.sin(wx * np.pi * sx) * np.sin(wy * np.pi * sy)
        offsets[node_id] = amp * shape * direction_vector
    return ImperfectionField(offsets, name=name, metadata={"kind": "plate_mode", "amplitude": amp, "waves": (wx, wy)})


def standard_flange_twist(
    model: "FEModel",
    node_ids: Sequence[int],
    twist_radians: float = 0.02,
    direction: Sequence[float] = (0.0, 0.0, 1.0),
    name: str = "flange_twist",
) -> ImperfectionField:
    """Simple linear outstand twist pattern with DNV table default 0.02 rad."""
    coords = _node_coords(model, node_ids)
    values = np.asarray(list(coords.values()), dtype=float)
    centroid = values.mean(axis=0)
    lever = np.asarray([coord - centroid for coord in coords.values()])
    if lever.size == 0:
        return ImperfectionField({}, name=name)
    lever_norm = np.linalg.norm(lever, axis=1)
    max_lever = max(float(np.max(lever_norm)), 1.0e-14)
    direction_vector = _unit(direction)
    offsets = {
        node_id: float(twist_radians) * float(np.linalg.norm(coord - centroid)) / max_lever * direction_vector
        for node_id, coord in coords.items()
    }
    return ImperfectionField(offsets, name=name, metadata={"kind": "flange_twist", "twist_radians": float(twist_radians)})


def calibrate_imperfection_amplitude(
    model_builder: Callable[[], "FEModel"],
    target_capacity: float,
    imperfection_builder: Callable[[float], Any],
    load_program: Any,
    bracket: Tuple[float, float],
    tolerance: float = 0.02,
    max_iterations: int = 20,
    solver_kwargs: Optional[Mapping[str, Any]] = None,
) -> ImperfectionCalibrationResult:
    """Binary-search an equivalent imperfection amplitude for target capacity."""
    from .nonlinear_static import solve_static_nonlinear

    lo, hi = float(bracket[0]), float(bracket[1])
    if lo < 0.0 or hi <= lo:
        raise ValueError("bracket must satisfy 0 <= low < high")
    kwargs = dict(solver_kwargs or {})
    history = []
    best_result = None
    best_amp = hi
    best_capacity = float("nan")
    converged = False
    for iteration in range(1, max_iterations + 1):
        amp = 0.5 * (lo + hi)
        model = apply_imperfection(model_builder(), imperfection_builder(amp), copy_model=False)
        if hasattr(load_program, "stages"):
            result = solve_static_nonlinear(model, load_program=load_program, **kwargs)
        else:
            result = solve_static_nonlinear(model, load_case=load_program, **kwargs)
        capacity = float(getattr(result, "peak_load_factor", result.capacity_estimate))
        history.append({"iteration": float(iteration), "amplitude": amp, "capacity": capacity})
        best_result = result
        best_amp = amp
        best_capacity = capacity
        rel_error = abs(capacity - float(target_capacity)) / max(abs(float(target_capacity)), 1.0)
        if rel_error <= tolerance:
            converged = True
            break
        # Larger imperfections normally reduce buckling capacity.
        if capacity > target_capacity:
            lo = amp
        else:
            hi = amp
    return ImperfectionCalibrationResult(
        amplitude=best_amp,
        capacity=best_capacity,
        iterations=len(history),
        converged=converged,
        history=tuple(history),
        result=best_result,
    )

