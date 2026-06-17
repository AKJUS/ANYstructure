"""Model mass properties and rigid-body inertia diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

from .elements import BeamElement, QuadraticBeamElement, ShellElement
from .matrix_assembly import assemble_mass_matrix

if TYPE_CHECKING:
    from .fe_core import FEModel


@dataclass
class MassProperties:
    """Integrated and assembled mass diagnostics for an FE model."""

    total_mass: float
    center_of_mass: np.ndarray
    first_moment: np.ndarray
    inertia_tensor_origin: np.ndarray
    inertia_tensor_center_of_mass: np.ndarray
    rigid_body_mass_matrix: np.ndarray
    assembled_translation_masses: Dict[str, float]
    num_mass_points: int
    skipped_elements: List[int]
    assembly_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_mass": float(self.total_mass),
            "center_of_mass": self.center_of_mass.tolist(),
            "first_moment": self.first_moment.tolist(),
            "inertia_tensor_origin": self.inertia_tensor_origin.tolist(),
            "inertia_tensor_center_of_mass": self.inertia_tensor_center_of_mass.tolist(),
            "rigid_body_mass_matrix": self.rigid_body_mass_matrix.tolist(),
            "assembled_translation_masses": dict(self.assembled_translation_masses),
            "num_mass_points": int(self.num_mass_points),
            "skipped_elements": [int(eid) for eid in self.skipped_elements],
            "assembly_info": self.assembly_info,
        }


def _point_inertia(points: List[Tuple[float, np.ndarray]], reference: np.ndarray) -> np.ndarray:
    inertia = np.zeros((3, 3), dtype=float)
    eye = np.eye(3)
    for mass, coords in points:
        r = np.asarray(coords, dtype=float) - reference
        inertia += float(mass) * ((float(r @ r) * eye) - np.outer(r, r))
    return inertia


def _rigid_body_modes(model: "FEModel", reference: np.ndarray) -> np.ndarray:
    total_dofs = model.mesh.dof_manager.total_dofs
    modes = np.zeros((total_dofs, 6), dtype=float)
    for node in model.mesh.nodes.values():
        x, y, z = node.coords() - reference
        ux, uy, uz, rx, ry, rz = node.dofs[:6]
        modes[ux, 0] = 1.0
        modes[uy, 1] = 1.0
        modes[uz, 2] = 1.0

        modes[uy, 3] = -z
        modes[uz, 3] = y
        modes[rx, 3] = 1.0

        modes[ux, 4] = z
        modes[uz, 4] = -x
        modes[ry, 4] = 1.0

        modes[ux, 5] = -y
        modes[uy, 5] = x
        modes[rz, 5] = 1.0
    return modes


def _shell_mass_points(model: "FEModel", element: ShellElement) -> List[Tuple[float, np.ndarray]]:
    material = model.get_material(element.material_name)
    coords = element.get_node_coordinates(model.mesh)
    points: List[Tuple[float, np.ndarray]] = []
    for (xi, eta), weight in zip(element.gauss_points, element.gauss_weights):
        N, dN_dxi, dN_deta = element.compute_shape_functions(float(xi), float(eta))
        _, _, _, det_j = element._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
        mass = float(material.density) * float(element.thickness) * float(det_j) * float(weight)
        points.append((mass, np.asarray(N @ coords, dtype=float)))
    return points


def _beam_mass_points(model: "FEModel", element: BeamElement) -> List[Tuple[float, np.ndarray]]:
    material = model.get_material(element.material_name)
    coords = element.get_node_coordinates(model.mesh)
    if isinstance(element, QuadraticBeamElement):
        length = float(np.linalg.norm(coords[2] - coords[0]))
        points = []
        for xi, weight in zip(element.GAUSS_POINTS, element.GAUSS_WEIGHTS):
            N, _ = element.compute_shape_functions(float(xi))
            mass = float(material.density) * float(element._A) * length / 2.0 * float(weight)
            points.append((mass, np.asarray(N @ coords, dtype=float)))
        return points

    length = float(np.linalg.norm(coords[1] - coords[0]))
    xi_values = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))
    points = []
    for xi in xi_values:
        N = np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0], dtype=float)
        mass = float(material.density) * float(element._A) * length / 2.0
        points.append((mass, np.asarray(N @ coords, dtype=float)))
    return points


def element_mass_points(model: "FEModel") -> Tuple[List[Tuple[float, np.ndarray]], List[int]]:
    """Return quadrature mass points for elements with physical density."""
    points: List[Tuple[float, np.ndarray]] = []
    skipped: List[int] = []
    for elem_id, element in model.mesh.elements.items():
        try:
            if isinstance(element, ShellElement):
                points.extend(_shell_mass_points(model, element))
            elif isinstance(element, BeamElement):
                points.extend(_beam_mass_points(model, element))
            else:
                skipped.append(int(elem_id))
        except Exception:
            skipped.append(int(elem_id))
    return points, skipped


def calculate_mass_properties(model: "FEModel", reference_point: Any = None) -> MassProperties:
    """Calculate integrated mass properties and assembled rigid-body mass.

    The scalar mass, first moment and inertia tensors are obtained from
    element quadrature.  The 6x6 rigid-body mass matrix is additionally formed
    from the assembled mass matrix using rigid translation/rotation fields, so
    shell and beam rotary inertia terms are visible in diagnostics.
    """
    points, skipped = element_mass_points(model)
    total_mass = float(sum(mass for mass, _ in points))
    first_moment = np.zeros(3, dtype=float)
    for mass, coords in points:
        first_moment += float(mass) * np.asarray(coords, dtype=float)
    center = first_moment / total_mass if total_mass > 0.0 else np.zeros(3, dtype=float)
    reference = center if reference_point is None else np.asarray(reference_point, dtype=float).reshape(3)

    M, assembly_info = assemble_mass_matrix(model)
    modes = _rigid_body_modes(model, reference)
    rbm = np.asarray(modes.T @ (M @ modes), dtype=float)
    tx = modes[:, 0]
    ty = modes[:, 1]
    tz = modes[:, 2]
    assembled_translation_masses = {
        "x": float(tx @ (M @ tx)),
        "y": float(ty @ (M @ ty)),
        "z": float(tz @ (M @ tz)),
    }

    return MassProperties(
        total_mass=total_mass,
        center_of_mass=center,
        first_moment=first_moment,
        inertia_tensor_origin=_point_inertia(points, np.zeros(3, dtype=float)),
        inertia_tensor_center_of_mass=_point_inertia(points, center),
        rigid_body_mass_matrix=rbm,
        assembled_translation_masses=assembled_translation_masses,
        num_mass_points=len(points),
        skipped_elements=skipped,
        assembly_info=assembly_info,
    )
