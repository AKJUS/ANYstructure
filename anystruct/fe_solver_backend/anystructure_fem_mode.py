"""ANYstructure generated-geometry FEM mode.

The functions in this module consume geometry that ANYstructure has already
generated.  They do not parse IFC or external FEM files.  The workflow is:

1. normalize generated geometry and idealize tagged member plates as beams;
2. convert generated nodes/shells/beams into ``FEModel``;
3. generate symmetric design loads from existing ANYstructure inputs;
4. solve the full static model;
5. recover prestress from the static result;
6. solve linear buckling from that recovered prestress.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .assembly import compute_stresses, solve_linear
from .boundary import BoundaryCondition, LoadCase
from .buckling import BucklingResult, solve_eigenvalue_buckling
from .elements import BeamElement, QuadraticBeamElement, ShellElement
from .fe_core import FEModel
from .mesh_gen import InterpolatedBeamShellMPCElement, RigidLidMPCElement
from .validation import LoadResultant, load_case_resultant


@dataclass(frozen=True)
class AnyStructureFEMConfig:
    """Configuration for the generated-geometry full FEM workflow."""

    geometry_scale: float = 1.0
    include_shells: bool = True
    include_beams: bool = True
    idealize_stiffeners_as_beams: bool = True
    idealize_girders_as_beams: bool = True
    auto_idealize_member_plates_as_beams: bool = True
    exclude_idealized_member_plates: bool = True
    require_idealized_member_beams: bool = True
    default_material: str = "steel"
    elastic_modulus: float = 210.0e9
    poisson_ratio: float = 0.3
    density: float = 7850.0
    yield_stress: float = 355.0e6
    pressure_pa: Optional[float] = None
    pressure_sign: float = 1.0
    load_scale: float = 1.0
    num_buckling_modes: int = 5
    solver_type: str = "direct"
    stress_percentile: float = 95.0
    add_inplane_edge_loads: bool = True


@dataclass
class AnyStructureFEMResult:
    """Result from the generated-geometry FEM mode."""

    valid: bool
    status: str
    invalid_reason: Optional[str] = None
    static_solver_status: str = "not_run"
    buckling_solver_status: str = "not_run"
    critical_load_factor: Optional[float] = None
    buckling_load_factors: List[float] = field(default_factory=list)
    max_displacement: float = 0.0
    max_translation: float = 0.0
    stress_max: float = 0.0
    stress_percentile: float = 0.0
    node_count: int = 0
    element_count: int = 0
    shell_element_count: int = 0
    beam_element_count: int = 0
    prestress_summary: Dict[str, Any] = field(default_factory=dict)
    load_resultant: Optional[LoadResultant] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    displacements: Optional[np.ndarray] = None
    buckling_result: Optional[BucklingResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "invalid_reason": self.invalid_reason,
            "static_solver_status": self.static_solver_status,
            "buckling_solver_status": self.buckling_solver_status,
            "critical_load_factor": self.critical_load_factor,
            "buckling_load_factors": self.buckling_load_factors,
            "max_displacement": self.max_displacement,
            "max_translation": self.max_translation,
            "stress_max": self.stress_max,
            "stress_percentile": self.stress_percentile,
            "node_count": self.node_count,
            "element_count": self.element_count,
            "shell_element_count": self.shell_element_count,
            "beam_element_count": self.beam_element_count,
            "prestress_summary": self.prestress_summary,
            "load_resultant": None if self.load_resultant is None else self.load_resultant.to_dict(),
            "diagnostics": self.diagnostics,
        }


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _collection(obj: Any, *names: str) -> List[Any]:
    value = _value(obj, *names, default=[])
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [{"id": key, **val} if isinstance(val, Mapping) else {"id": key, "value": val} for key, val in value.items()]
    return list(value)


def _collection_by_name(obj: Any, name: str) -> List[Any]:
    if obj is None:
        return []
    missing = object()
    if isinstance(obj, Mapping):
        value = obj.get(name, missing)
    else:
        value = getattr(obj, name, missing)
    if value is missing or value is None:
        return []
    if isinstance(value, Mapping):
        return [{"id": key, **val} if isinstance(val, Mapping) else {"id": key, "value": val} for key, val in value.items()]
    return list(value)


def _combined_collections(obj: Any, *names: str) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    for name in names:
        items.extend((name, item) for item in _collection_by_name(obj, name))
    return items


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coords_from_node(node: Any, scale: float) -> Tuple[int, float, float, float]:
    if isinstance(node, Mapping):
        node_id = int(_value(node, "id", "node_id"))
        if "value" in node:
            coords = np.asarray(node["value"], dtype=float).reshape(-1)
            if coords.size >= 3:
                return node_id, float(coords[0]) * scale, float(coords[1]) * scale, float(coords[2]) * scale
        if "coords" in node:
            coords = np.asarray(node["coords"], dtype=float).reshape(-1)
            return node_id, float(coords[0]) * scale, float(coords[1]) * scale, float(coords[2]) * scale
        return (
            node_id,
            _as_float(_value(node, "x")) * scale,
            _as_float(_value(node, "y")) * scale,
            _as_float(_value(node, "z")) * scale,
        )
    if isinstance(node, (list, tuple)) and len(node) >= 4:
        return int(node[0]), float(node[1]) * scale, float(node[2]) * scale, float(node[3]) * scale
    node_id = int(_value(node, "id", "node_id"))
    return node_id, _as_float(_value(node, "x")) * scale, _as_float(_value(node, "y")) * scale, _as_float(_value(node, "z")) * scale


def _node_ids(item: Any) -> List[int]:
    ids = _value(item, "node_ids", "nodes", "connectivity", default=[])
    return [int(node_id) for node_id in ids]


def _cross_section(item: Any) -> Dict[str, float]:
    section = _value(item, "cross_section", "section", default=None)
    if isinstance(section, Mapping):
        source = section
    else:
        source = item
    section = {
        "area": _as_float(_value(source, "area", "A"), 0.01),
        "Iy": _as_float(_value(source, "Iy", "iy"), 1.0e-8),
        "Iz": _as_float(_value(source, "Iz", "iz"), 1.0e-8),
        "J": _as_float(_value(source, "J", "torsion_constant"), 1.0e-8),
        "shear_factor_y": _as_float(_value(source, "shear_factor_y", "ky"), 5.0 / 6.0),
        "shear_factor_z": _as_float(_value(source, "shear_factor_z", "kz"), 5.0 / 6.0),
    }
    orientation = _value(source, "orientation", "section_orientation", "web_direction", default=None)
    if orientation is not None:
        vector = [float(component) for component in orientation]
        if len(vector) >= 3 and any(abs(component) > 0.0 for component in vector[:3]):
            section["orientation"] = tuple(vector[:3])
    for key, aliases in (
        ("c_y", ("c_y", "cy", "fiber_distance_y")),
        ("c_z", ("c_z", "cz", "fiber_distance_z")),
        ("torsion_modulus", ("torsion_modulus", "Wt", "torsional_section_modulus")),
    ):
        value = _value(source, *aliases, default=None)
        if value is not None and float(value) > 0.0:
            section[key] = float(value)
    return section


def _has_cross_section(item: Any) -> bool:
    section = _value(item, "cross_section", "section", default=None)
    if isinstance(section, Mapping):
        return True
    return any(_value(item, name, default=None) is not None for name in ("area", "A", "Iy", "iy", "Iz", "iz", "J", "torsion_constant"))


def _material_name(item: Any, config: AnyStructureFEMConfig) -> str:
    return str(_value(item, "material", "material_name", default=config.default_material))


def _structural_member_role(item: Any, source_name: str = "") -> Optional[str]:
    raw = " ".join(
        str(value)
        for value in (
            source_name,
            _value(item, "role", "structural_role", "member_role", "kind", "category", "type", default=""),
        )
        if value is not None
    ).lower()
    if any(token in raw for token in ("stiffener", "longitudinal", "stringer")):
        return "stiffener"
    if any(token in raw for token in ("girder", "frame", "transverse")):
        return "girder"
    return None


def _is_idealized_member_plate(item: Any, config: AnyStructureFEMConfig) -> Optional[str]:
    role = _structural_member_role(item)
    if role == "stiffener" and config.idealize_stiffeners_as_beams:
        return role
    if role == "girder" and config.idealize_girders_as_beams:
        return role
    return None


def _add_model_element(model: FEModel, elem_id: int, element: Any) -> None:
    if model.mesh.get_element(elem_id) is not None:
        raise ValueError(f"duplicate-element-id-{elem_id}")
    model.add_element(elem_id, element)


def _item_copy(item: Any) -> Any:
    if isinstance(item, Mapping):
        return dict(item)
    return item


def _beam_item_copy(source_name: str, item: Any) -> Any:
    copied = _item_copy(item)
    role = _structural_member_role(copied, source_name)
    if isinstance(copied, Mapping):
        copied.setdefault("role", role or "beam")
        copied.setdefault("geometry_source", source_name)
    return copied


def _polygon_area(coords: np.ndarray) -> float:
    if coords.shape[0] < 3:
        return 0.0
    centroid = np.mean(coords, axis=0)
    area = 0.0
    for i in range(coords.shape[0]):
        a = coords[i] - centroid
        b = coords[(i + 1) % coords.shape[0]] - centroid
        area += 0.5 * float(np.linalg.norm(np.cross(a, b)))
    return area


def _infer_member_centerline(plates: List[Any], node_coords: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    coords = []
    for plate in plates:
        for node_id in _node_ids(plate):
            if node_id in node_coords:
                coords.append(node_coords[node_id])
    if len(coords) < 2:
        raise ValueError("member-plate-has-insufficient-nodes")
    points = np.asarray(coords, dtype=float)
    center = np.mean(points, axis=0)
    _, singular_values, vh = np.linalg.svd(points - center, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 1.0e-12:
        raise ValueError("member-plate-centerline-has-zero-length")
    direction = vh[0]
    projections = (points - center) @ direction
    p0 = center + float(np.min(projections)) * direction
    p1 = center + float(np.max(projections)) * direction
    if np.linalg.norm(p1 - p0) <= 1.0e-12:
        raise ValueError("member-plate-centerline-has-zero-length")
    return p0, p1


def _section_from_member_plates(
    plates: List[Any],
    node_coords: Dict[int, np.ndarray],
    p0: np.ndarray,
    p1: np.ndarray,
    scale: float,
) -> Dict[str, float]:
    for plate in plates:
        if _has_cross_section(plate):
            return _cross_section(plate)

    length = float(np.linalg.norm(p1 - p0))
    if length <= 0.0:
        raise ValueError("member-plate-centerline-has-zero-length")

    area = 0.0
    iy = 0.0
    iz = 0.0
    max_section_width = 0.0
    max_thickness = 0.0
    for plate in plates:
        node_ids = _node_ids(plate)
        coords = np.asarray([node_coords[node_id] for node_id in node_ids if node_id in node_coords], dtype=float)
        if coords.shape[0] < 3:
            continue
        thickness = _as_float(_value(plate, "thickness", "t"), 0.0)
        if thickness <= 0.0:
            continue
        face_area = _polygon_area(coords[:4] if coords.shape[0] >= 4 else coords)
        section_width = face_area / length if length > 0.0 else 0.0
        if section_width <= 0.0:
            continue
        area += thickness * section_width
        iy += thickness * section_width**3 / 12.0
        iz += section_width * thickness**3 / 12.0
        max_section_width = max(max_section_width, section_width)
        max_thickness = max(max_thickness, thickness)

    if area <= 0.0:
        raise ValueError("member-plate-section-could-not-be-inferred")

    scaled_area = area * scale**2
    scaled_iy = max(iy * scale**4, 1.0e-12)
    scaled_iz = max(iz * scale**4, 1.0e-12)
    section = {
        "area": scaled_area,
        "Iy": scaled_iy,
        "Iz": scaled_iz,
        "J": max((iy + iz) * scale**4, 1.0e-12),
        "shear_factor_y": 5.0 / 6.0,
        "shear_factor_z": 5.0 / 6.0,
    }
    if max_section_width > 0.0 and max_thickness > 0.0:
        # Strip section: width (web) along local z, thickness along local y.
        section["c_z"] = 0.5 * max_section_width * scale
        section["c_y"] = 0.5 * max_thickness * scale
        section["torsion_modulus"] = section["J"] / (max_thickness * scale)

    # The inferred strip section has its width (web) along the in-plane
    # direction transverse to the centerline.  Recover that direction so the
    # beam local z axis matches the Iy/Iz meaning used above.
    points = []
    for plate in plates:
        for node_id in _node_ids(plate):
            if node_id in node_coords:
                points.append(node_coords[node_id])
    if len(points) >= 3:
        points_arr = np.asarray(points, dtype=float)
        axis = (p1 - p0) / length
        offsets = points_arr - np.mean(points_arr, axis=0)
        offsets -= np.outer(offsets @ axis, axis)
        _, singular_values, vh = np.linalg.svd(offsets, full_matrices=False)
        if singular_values.size and singular_values[0] > 1.0e-9 * length:
            section["orientation"] = tuple(float(component) for component in vh[0])
    return section


def _member_group_key(item: Any, role: str) -> Tuple[str, str]:
    raw_id = _value(item, "member_id", "parent_id", "stiffener_id", "girder_id", "name", "id", default="member")
    return role, str(raw_id)


def _next_available_id(used: set, start: int) -> int:
    value = int(start)
    while value in used:
        value += 1
    used.add(value)
    return value


def idealize_generated_geometry_members(
    generated_geometry: Any,
    config: Optional[AnyStructureFEMConfig] = None,
) -> Dict[str, Any]:
    """Return generated geometry with tagged stiffener/girder plates collapsed to beams.

    ANYstructure may produce member geometry as plate surfaces.  For the default
    beam-shell solver idealization, tagged stiffener and girder plates are
    removed from the shell set and represented by equivalent line beams.
    """
    config = config or AnyStructureFEMConfig()
    if generated_geometry is None:
        raise ValueError("missing-generated-geometry")

    nodes = [_item_copy(node) for node in _collection(generated_geometry, "nodes")]
    node_coords = {node_id: np.array([x, y, z], dtype=float) for node_id, x, y, z in (_coords_from_node(node, 1.0) for node in nodes)}
    used_node_ids = set(node_coords)

    output: Dict[str, Any] = {
        "name": str(_value(generated_geometry, "name", default="ANYstructureGeneratedGeometry")),
        "nodes": nodes,
        "shells": [],
        "beams": [_beam_item_copy(source, beam) for source, beam in _combined_collections(
            generated_geometry,
            "beams",
            "beam_members",
            "line_members",
            "members",
            "stiffeners",
            "girders",
            "frames",
            "longitudinals",
            "transverses",
        )],
        "couplings": [_item_copy(item) for _source, item in _combined_collections(generated_geometry, "couplings", "mpcs")],
        "rigid_lids": [_item_copy(item) for _source, item in _combined_collections(generated_geometry, "rigid_lids", "diaphragms")],
        "supports": [_item_copy(item) for _source, item in _combined_collections(generated_geometry, "supports", "boundary_conditions")],
        "materials": [_item_copy(item) for item in _collection(generated_geometry, "materials")],
        "idealization": {"auto_member_beams": [], "excluded_member_plates": []},
    }

    member_plate_groups: Dict[Tuple[str, str], List[Any]] = {}
    excluded_member_roles: List[str] = []
    for _source_name, shell in _combined_collections(generated_geometry, "shells", "shell_faces", "faces", "plates"):
        role = _is_idealized_member_plate(shell, config)
        if role and config.exclude_idealized_member_plates:
            excluded_member_roles.append(role)
            output["idealization"]["excluded_member_plates"].append({"id": _value(shell, "id", "element_id", default=None), "role": role})
            if config.auto_idealize_member_plates_as_beams:
                member_plate_groups.setdefault(_member_group_key(shell, role), []).append(shell)
            continue
        output["shells"].append(_item_copy(shell))

    existing_member_keys = {
        _member_group_key(beam, role)
        for beam in output["beams"]
        for role in [_structural_member_role(beam)]
        if role is not None
    }
    existing_roles = {role for role, _member_id in existing_member_keys}
    if excluded_member_roles and config.require_idealized_member_beams and not config.auto_idealize_member_plates_as_beams:
        missing_roles = sorted({role for role in excluded_member_roles if role not in existing_roles})
        if missing_roles:
            raise ValueError(f"idealized-member-plates-require-beam-members-{','.join(missing_roles)}")

    used_element_ids = {
        int(element_id)
        for element_id in (
            _value(item, "id", "element_id", default=None)
            for item in output["shells"] + output["beams"] + output["couplings"] + output["rigid_lids"]
        )
        if element_id is not None
    }

    for key, plates in member_plate_groups.items():
        role, member_id = key
        if key in existing_member_keys:
            continue
        p0, p1 = _infer_member_centerline(plates, node_coords)
        section = _section_from_member_plates(plates, node_coords, p0, p1, config.geometry_scale)
        n0 = _next_available_id(used_node_ids, 900_000)
        n1 = _next_available_id(used_node_ids, n0 + 1)
        output["nodes"].extend(
            [
                {"id": n0, "coords": p0.tolist()},
                {"id": n1, "coords": p1.tolist()},
            ]
        )
        node_coords[n0] = p0
        node_coords[n1] = p1
        element_id = _next_available_id(used_element_ids, 20_000)
        beam = {
            "id": element_id,
            "node_ids": [n0, n1],
            "cross_section": section,
            "role": role,
            "member_id": member_id,
            "material": _material_name(plates[0], config),
            "generated_from": "member_plates",
        }
        output["beams"].append(beam)
        output["idealization"]["auto_member_beams"].append(
            {"id": element_id, "role": role, "member_id": member_id, "source_plate_count": len(plates)}
        )

    return output


def _add_materials(model: FEModel, generated_geometry: Any, config: AnyStructureFEMConfig) -> None:
    model.add_material(
        config.default_material,
        elastic_modulus=config.elastic_modulus,
        poisson_ratio=config.poisson_ratio,
        density=config.density,
        yield_stress=config.yield_stress,
    )
    model.current_material = config.default_material
    for item in _collection(generated_geometry, "materials"):
        name = str(_value(item, "name", "id", default=config.default_material))
        model.add_material(
            name,
            elastic_modulus=_as_float(_value(item, "elastic_modulus", "E"), config.elastic_modulus),
            poisson_ratio=_as_float(_value(item, "poisson_ratio", "nu"), config.poisson_ratio),
            density=_as_float(_value(item, "density"), config.density),
            yield_stress=_as_float(_value(item, "yield_stress", "fy"), config.yield_stress),
        )


def build_fe_model_from_generated_geometry(
    generated_geometry: Any,
    config: Optional[AnyStructureFEMConfig] = None,
) -> FEModel:
    """Convert ANYstructure-generated geometry into an ``FEModel``."""
    config = config or AnyStructureFEMConfig()
    if generated_geometry is None:
        raise ValueError("missing-generated-geometry")
    generated_geometry = idealize_generated_geometry_members(generated_geometry, config)

    model = FEModel(str(_value(generated_geometry, "name", default="ANYstructureGeneratedGeometry")))
    _add_materials(model, generated_geometry, config)

    nodes = _collection(generated_geometry, "nodes")
    if not nodes:
        raise ValueError("generated-geometry-has-no-nodes")
    for node in nodes:
        node_id, x, y, z = _coords_from_node(node, config.geometry_scale)
        model.add_node(node_id, x, y, z)

    element_count = 0
    skipped_member_plate_roles: List[str] = []
    if config.include_shells:
        for _source_name, shell in _combined_collections(generated_geometry, "shells", "shell_faces", "faces", "plates"):
            member_plate_role = _is_idealized_member_plate(shell, config)
            if member_plate_role and config.exclude_idealized_member_plates:
                skipped_member_plate_roles.append(member_plate_role)
                continue
            node_ids = _node_ids(shell)
            if len(node_ids) not in {4, 8}:
                raise ValueError(f"unsupported-shell-topology-{len(node_ids)}")
            elem_id = int(_value(shell, "id", "element_id", default=element_count + 1))
            thickness = _as_float(_value(shell, "thickness", "t"), 0.0) * config.geometry_scale
            if thickness <= 0.0:
                raise ValueError("shell-thickness-must-be-positive")
            elem_type = str(_value(shell, "type", default="")).upper()
            _add_model_element(model, elem_id, ShellElement(elem_id, node_ids, _material_name(shell, config), thickness=thickness, reduced_integration=(elem_type == "S8R")))
            element_count += 1

    beam_roles: List[str] = []
    if config.include_beams:
        beam_collections = (
            "beams",
            "beam_members",
            "line_members",
            "members",
            "stiffeners",
            "girders",
            "frames",
            "longitudinals",
            "transverses",
        )
        for source_name, beam in _combined_collections(generated_geometry, *beam_collections):
            node_ids = _node_ids(beam)
            elem_id = int(_value(beam, "id", "element_id", default=20_000 + element_count + 1))
            section = _cross_section(beam)
            material_name = _material_name(beam, config)
            role = _structural_member_role(beam, source_name)
            if len(node_ids) == 2:
                element = BeamElement(elem_id, node_ids, material_name, section)
            elif len(node_ids) == 3:
                element = QuadraticBeamElement(elem_id, node_ids, material_name, section)
            else:
                raise ValueError(f"unsupported-beam-topology-{len(node_ids)}")
            element.structural_role = role or "beam"
            element.geometry_source = source_name
            _add_model_element(model, elem_id, element)
            if role:
                beam_roles.append(role)
            element_count += 1

    if skipped_member_plate_roles and config.require_idealized_member_beams:
        missing_roles = sorted({role for role in skipped_member_plate_roles if role not in beam_roles})
        if missing_roles:
            raise ValueError(f"idealized-member-plates-require-beam-members-{','.join(missing_roles)}")

    for coupling in _collection(generated_geometry, "couplings", "mpcs"):
        elem_id = int(_value(coupling, "id", "element_id", default=30_000 + element_count + 1))
        beam_node_id = int(_value(coupling, "beam_node_id", "slave_node"))
        shell_node_ids = [int(node_id) for node_id in _value(coupling, "shell_node_ids", "master_nodes", default=[])]
        shape_weights = np.asarray(_value(coupling, "shape_weights", "weights", default=[]), dtype=float)
        eccentricity = np.asarray(_value(coupling, "eccentricity", default=[0.0, 0.0, 0.0]), dtype=float)
        if len(shell_node_ids) == 0 or shape_weights.size != len(shell_node_ids):
            raise ValueError("invalid-beam-shell-coupling")
        _add_model_element(
            model,
            elem_id,
            InterpolatedBeamShellMPCElement(
                elem_id,
                beam_node_id,
                shell_node_ids,
                shape_weights,
                eccentricity,
                material_name=config.default_material,
            ),
        )
        element_count += 1

    for lid in _collection(generated_geometry, "rigid_lids", "diaphragms"):
        elem_id = int(_value(lid, "id", "element_id", default=40_000 + element_count + 1))
        center_node_id = int(_value(lid, "center_node_id", "reference_node", "master_node"))
        ring_node_ids = [int(node_id) for node_id in _value(lid, "ring_node_ids", "node_ids", "nodes", default=[])]
        if len(ring_node_ids) < 3:
            raise ValueError("invalid-rigid-lid-ring")
        _add_model_element(
            model,
            elem_id,
            RigidLidMPCElement(
                elem_id,
                center_node_id,
                ring_node_ids,
                material_name=config.default_material,
            ),
        )
        element_count += 1

    for support in _collection(generated_geometry, "supports", "boundary_conditions"):
        node_ids = [int(node_id) for node_id in _value(support, "node_ids", "nodes", default=[])]
        constraints = dict(_value(support, "dof_constraints", "constraints", default={}))
        if node_ids and constraints:
            model.add_boundary_condition(BoundaryCondition(str(_value(support, "name", default="generated_support")), node_ids, constraints))

    if model.mesh.num_elements == 0:
        raise ValueError("generated-geometry-has-no-supported-elements")
    return model


def _calc_part(calc_object: Any, name: str) -> Any:
    root = calc_object[0] if isinstance(calc_object, (list, tuple)) else calc_object
    return _value(root, name, default=None)


def _pressure_pa(calc_object: Any, lat_press: float, config: AnyStructureFEMConfig) -> float:
    if config.pressure_pa is not None:
        return float(config.pressure_pa) * float(config.load_scale)
    if lat_press:
        return float(lat_press) * 1000.0 * float(config.load_scale)
    for part_name in ("Plate", "Stiffener"):
        part = _calc_part(calc_object, part_name)
        pressure = _value(part, "pressure", "lat_press", default=None)
        if pressure is not None:
            return float(pressure) * 1.0e6 * float(config.load_scale)
    return 0.0


def _design_stress_pa(calc_object: Any, attr1: str, attr2: Optional[str] = None) -> float:
    part = _calc_part(calc_object, "Stiffener") or _calc_part(calc_object, "Plate")
    if part is None:
        return 0.0
    first = _as_float(_value(part, attr1), 0.0)
    second = _as_float(_value(part, attr2), first) if attr2 else first
    value = first if abs(first) >= abs(second) else second
    return value * 1.0e6


def _shell_elements(model: FEModel) -> List[ShellElement]:
    return [element for element in model.mesh.elements.values() if isinstance(element, ShellElement)]


def _edge_weights(nodes: List[Any], axis: int) -> Dict[int, float]:
    if not nodes:
        return {}
    coords = np.asarray([node.coords() for node in nodes], dtype=float)
    if len(nodes) == 1:
        return {int(nodes[0].id): 1.0}
    order = np.argsort(coords[:, axis])
    sorted_nodes = [nodes[int(i)] for i in order]
    sorted_coords = coords[order, axis]
    weights = {}
    for i, node in enumerate(sorted_nodes):
        if i == 0:
            length = abs(sorted_coords[1] - sorted_coords[0])
        elif i == len(sorted_nodes) - 1:
            length = abs(sorted_coords[-1] - sorted_coords[-2])
        else:
            length = 0.5 * abs(sorted_coords[i + 1] - sorted_coords[i - 1])
        weights[int(node.id)] = float(length)
    return weights


def _add_rectangular_edge_loads(load_case: LoadCase, calc_object: Any, model: FEModel, config: AnyStructureFEMConfig) -> Dict[str, float]:
    shell_elements = _shell_elements(model)
    if not shell_elements:
        return {"axial_stress_pa": 0.0, "transverse_stress_pa": 0.0, "shear_stress_pa": 0.0}

    shell_node_ids = sorted({node_id for element in shell_elements for node_id in element.node_ids})
    nodes = [model.mesh.get_node(node_id) for node_id in shell_node_ids]
    nodes = [node for node in nodes if node is not None]
    coords = np.asarray([node.coords() for node in nodes], dtype=float)
    span = np.ptp(coords, axis=0)
    if span[0] <= 0.0 or span[1] <= 0.0:
        return {"axial_stress_pa": 0.0, "transverse_stress_pa": 0.0, "shear_stress_pa": 0.0}

    thickness = float(np.mean([element.thickness for element in shell_elements]))
    sx = _design_stress_pa(calc_object, "sigma_x1", "sigma_x2") * config.load_scale
    sy = _design_stress_pa(calc_object, "sigma_y1", "sigma_y2") * config.load_scale
    txy = _design_stress_pa(calc_object, "tau_xy") * config.load_scale
    xmin, xmax = float(np.min(coords[:, 0])), float(np.max(coords[:, 0]))
    ymin, ymax = float(np.min(coords[:, 1])), float(np.max(coords[:, 1]))
    tol = 1.0e-8 * max(float(np.max(span)), 1.0)
    x0 = [node for node in nodes if abs(node.x - xmin) <= tol]
    x1 = [node for node in nodes if abs(node.x - xmax) <= tol]
    y0 = [node for node in nodes if abs(node.y - ymin) <= tol]
    y1 = [node for node in nodes if abs(node.y - ymax) <= tol]

    def add_edge(edge_nodes: List[Any], weights_axis: int, force_per_length: np.ndarray) -> None:
        weights = _edge_weights(edge_nodes, weights_axis)
        for node_id, length in weights.items():
            load_case.add_nodal_load(node_id, forces=force_per_length * length)

    # Existing ANYstructure stress inputs are treated as compression-positive.
    add_edge(x0, 1, np.array([sx * thickness, txy * thickness, 0.0]))
    add_edge(x1, 1, np.array([-sx * thickness, -txy * thickness, 0.0]))
    add_edge(y0, 0, np.array([txy * thickness, sy * thickness, 0.0]))
    add_edge(y1, 0, np.array([-txy * thickness, -sy * thickness, 0.0]))
    return {"axial_stress_pa": sx, "transverse_stress_pa": sy, "shear_stress_pa": txy}


def build_symmetric_load_case(
    calc_object: Any,
    model: FEModel,
    config: Optional[AnyStructureFEMConfig] = None,
    lat_press: float = 0.0,
) -> LoadCase:
    """Create symmetric pressure and in-plane loads from ANYstructure inputs."""
    config = config or AnyStructureFEMConfig()
    load_case = LoadCase("anystructure_symmetric_load")
    pressure = _pressure_pa(calc_object, lat_press, config)
    if pressure != 0.0:
        for element in _shell_elements(model):
            load_case.add_pressure_load(int(element.element_id), float(config.pressure_sign) * pressure)
    if config.add_inplane_edge_loads:
        _add_rectangular_edge_loads(load_case, calc_object, model, config)
    return load_case


def recover_prestress_from_static_result(model: FEModel, displacements: np.ndarray) -> Tuple[Dict[int, Dict[str, float]], Dict[str, Any]]:
    """Recover element prestress states for buckling from a static solution."""
    states: Dict[int, Dict[str, float]] = {}
    stresses = compute_stresses(model, displacements)
    shell_count = 0
    beam_count = 0
    shell_compression = []
    beam_compression = []
    for element_id, element in model.mesh.elements.items():
        stress = stresses.get(element_id)
        if isinstance(element, ShellElement) and stress:
            sx = float(np.mean(stress.get("membrane_xx", np.zeros(1))))
            sy = float(np.mean(stress.get("membrane_yy", np.zeros(1))))
            txy = float(np.mean(stress.get("membrane_xy", np.zeros(1))))
            states[int(element_id)] = {
                "membrane_force_x": sx * element.thickness,
                "membrane_force_y": sy * element.thickness,
                "membrane_force_xy": txy * element.thickness,
            }
            shell_compression.append(max(-sx * element.thickness, -sy * element.thickness, 0.0))
            shell_count += 1
        elif isinstance(element, (BeamElement, QuadraticBeamElement)) and stress:
            axial_force = float(stress.get("axial_stress", 0.0)) * float(getattr(element, "_A", 0.0))
            states[int(element_id)] = {"axial_force": axial_force}
            beam_compression.append(max(-axial_force, 0.0))
            beam_count += 1
    summary = {
        "shell_elements": shell_count,
        "beam_elements": beam_count,
        "state_count": len(states),
        "max_shell_compression_resultant": float(max(shell_compression) if shell_compression else 0.0),
        "max_beam_compression_force": float(max(beam_compression) if beam_compression else 0.0),
    }
    return states, summary


def _stress_statistics(model: FEModel, displacements: np.ndarray, percentile: float) -> Dict[str, float]:
    values = []
    for stress in compute_stresses(model, displacements).values():
        if "von_mises" not in stress:
            continue
        values.extend(np.asarray(stress["von_mises"], dtype=float).reshape(-1).tolist())
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "max": 0.0, "percentile": 0.0}
    return {
        "count": int(arr.size),
        "max": float(np.max(arr)),
        "percentile": float(np.percentile(arr, percentile)),
    }


def _max_translation(model: FEModel, displacements: np.ndarray) -> float:
    value = 0.0
    for node in model.mesh.nodes.values():
        value = max(value, float(np.linalg.norm(displacements[node.dofs[:3]])))
    return value


def _beam_role_counts(model: FEModel) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for element in model.mesh.elements.values():
        if not isinstance(element, (BeamElement, QuadraticBeamElement)):
            continue
        role = str(getattr(element, "structural_role", "beam") or "beam")
        counts[role] = counts.get(role, 0) + 1
    return counts


def _invalid_result(reason: str, diagnostics: Optional[Dict[str, Any]] = None) -> AnyStructureFEMResult:
    return AnyStructureFEMResult(valid=False, status="invalid", invalid_reason=reason, diagnostics=diagnostics or {})


def run_anystructure_fem_mode(
    calc_object: Any,
    generated_geometry: Any,
    config: Optional[AnyStructureFEMConfig] = None,
    lat_press: float = 0.0,
) -> AnyStructureFEMResult:
    """Run static full-geometry analysis and buckling from recovered prestress."""
    config = config or AnyStructureFEMConfig()
    try:
        model = build_fe_model_from_generated_geometry(generated_geometry, config)
        load_case = build_symmetric_load_case(calc_object, model, config, lat_press=lat_press)
    except Exception as exc:
        return _invalid_result(str(exc))

    resultant = load_case_resultant(model, load_case)
    displacements, solver_info = solve_linear(model, load_case, solver_type=config.solver_type, constraint_mode="auto")
    static_status = str((solver_info.get("convergence_info") or {}).get("status", "unknown"))
    if static_status != "converged":
        return AnyStructureFEMResult(
            valid=False,
            status="static_failed",
            invalid_reason=static_status,
            static_solver_status=static_status,
            node_count=model.mesh.num_nodes,
            element_count=model.mesh.num_elements,
            shell_element_count=len(_shell_elements(model)),
            beam_element_count=sum(isinstance(e, (BeamElement, QuadraticBeamElement)) for e in model.mesh.elements.values()),
            load_resultant=resultant,
            diagnostics={"solver_info": solver_info},
            displacements=displacements,
        )

    prestress_states, prestress_summary = recover_prestress_from_static_result(model, displacements)
    if not prestress_states:
        return AnyStructureFEMResult(
            valid=False,
            status="prestress_failed",
            invalid_reason="failed-prestress-recovery",
            static_solver_status=static_status,
            node_count=model.mesh.num_nodes,
            element_count=model.mesh.num_elements,
            load_resultant=resultant,
            diagnostics={"solver_info": solver_info},
            displacements=displacements,
        )

    buckling = solve_eigenvalue_buckling(model, prestress_states, num_modes=config.num_buckling_modes)
    stress_stats = _stress_statistics(model, displacements, config.stress_percentile)
    mode_factors = [float(mode.load_factor) for mode in buckling.modes]
    valid = buckling.solver_status == "ok" and bool(mode_factors)
    status = "ok" if valid else "buckling_failed"
    invalid_reason = None if valid else buckling.solver_status
    return AnyStructureFEMResult(
        valid=valid,
        status=status,
        invalid_reason=invalid_reason,
        static_solver_status=static_status,
        buckling_solver_status=buckling.solver_status,
        critical_load_factor=buckling.critical_load_factor,
        buckling_load_factors=mode_factors,
        max_displacement=float(np.max(np.abs(displacements))) if displacements.size else 0.0,
        max_translation=_max_translation(model, displacements),
        stress_max=stress_stats["max"],
        stress_percentile=stress_stats["percentile"],
        node_count=model.mesh.num_nodes,
        element_count=model.mesh.num_elements,
        shell_element_count=len(_shell_elements(model)),
        beam_element_count=sum(isinstance(e, (BeamElement, QuadraticBeamElement)) for e in model.mesh.elements.values()),
        prestress_summary=prestress_summary,
        load_resultant=resultant,
        diagnostics={
            "solver_info": solver_info,
            "stress_statistics": stress_stats,
            "buckling": buckling.to_dict(),
            "beam_role_counts": _beam_role_counts(model),
        },
        displacements=displacements,
        buckling_result=buckling,
    )
