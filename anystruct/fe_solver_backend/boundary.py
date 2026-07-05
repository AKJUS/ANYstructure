"""
Boundary Conditions and Load Cases

This module provides classes for defining boundary conditions and loads
for the FE model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .fe_core import DOFManager, FEMesh, Material, Node


_SMALL = 1.0e-12


class _GravityFallbackMaterial:
    """Minimal material used when loads are assembled without model context."""

    density = 7850.0
    elastic_modulus = 210.0e9
    poisson_ratio = 0.3

    @property
    def shear_modulus(self) -> float:
        return self.elastic_modulus / (2.0 * (1.0 + self.poisson_ratio))


@dataclass
class BoundaryCondition:
    """
    Base class for boundary conditions.

    Boundary conditions constrain specific DOFs of nodes.
    """

    name: str
    node_ids: List[int]
    dof_constraints: Dict[str, float]

    def __post_init__(self):
        dof_names = ["ux", "uy", "uz", "rx", "ry", "rz"]
        self._dof_indices = {}
        for dof_name, value in self.dof_constraints.items():
            if dof_name in dof_names:
                self._dof_indices[dof_name] = dof_names.index(dof_name)

    def apply(self, dof_manager: "DOFManager"):
        """Apply this boundary condition to the DOF manager."""
        for node_id in self.node_ids:
            node_dofs = dof_manager.get_node_dofs(node_id)
            if not node_dofs:
                continue
            for dof_name, value in self.dof_constraints.items():
                if dof_name in self._dof_indices:
                    local_idx = self._dof_indices[dof_name]
                    global_dof = node_dofs[local_idx]
                    dof_manager.constrain_dof(global_dof)

    def get_constrained_dofs(self, dof_manager: "DOFManager") -> List[Tuple[int, float]]:
        """Get list of (global_dof, prescribed_value) pairs."""
        constrained = []
        for node_id in self.node_ids:
            node_dofs = dof_manager.get_node_dofs(node_id)
            if not node_dofs:
                continue
            for dof_name, value in self.dof_constraints.items():
                if dof_name in self._dof_indices:
                    local_idx = self._dof_indices[dof_name]
                    global_dof = node_dofs[local_idx]
                    constrained.append((global_dof, value))
        return constrained


@dataclass
class FixedSupport(BoundaryCondition):
    """Fully fixed support: all DOFs constrained to zero."""

    def __init__(self, name: str, node_ids: List[int]):
        dof_constraints = {dof: 0.0 for dof in ["ux", "uy", "uz", "rx", "ry", "rz"]}
        super().__init__(name, node_ids, dof_constraints)


@dataclass
class PinnedSupport(BoundaryCondition):
    """Pinned support: translational DOFs fixed, rotations free."""

    def __init__(self, name: str, node_ids: List[int]):
        dof_constraints = {dof: 0.0 for dof in ["ux", "uy", "uz"]}
        super().__init__(name, node_ids, dof_constraints)


@dataclass
class RollerSupport(BoundaryCondition):
    """Roller support constraining selected translational DOFs."""

    def __init__(self, name: str, node_ids: List[int], constrained_directions: Optional[List[str]] = None):
        if constrained_directions is None:
            constrained_directions = ["uy", "uz"]
        dof_constraints = {dof: 0.0 for dof in constrained_directions}
        super().__init__(name, node_ids, dof_constraints)


@dataclass
class SymmetryBC(BoundaryCondition):
    """Symmetry boundary condition in a global coordinate plane."""

    def __init__(self, name: str, node_ids: List[int], symmetry_plane: str = "xy"):
        if symmetry_plane == "xy":
            dof_constraints = {"uz": 0.0, "rx": 0.0, "ry": 0.0}
        elif symmetry_plane == "xz":
            dof_constraints = {"uy": 0.0, "rx": 0.0, "rz": 0.0}
        elif symmetry_plane == "yz":
            dof_constraints = {"ux": 0.0, "ry": 0.0, "rz": 0.0}
        else:
            dof_constraints = {}
        super().__init__(name, node_ids, dof_constraints)


@dataclass
class LoadCase:
    """
    Load case for the FE model.

    Contains nodal loads, element loads, pressure loads and optional gravity.
    Pressure loads on shell elements are assembled as consistent nodal loads by
    Gauss integration over the element surface, instead of equal area shares.
    """

    name: str
    nodal_loads: Dict[int, np.ndarray] = field(default_factory=dict)
    element_loads: Dict[int, np.ndarray] = field(default_factory=dict)
    pressure_loads: Dict[int, float] = field(default_factory=dict)
    gravity: Optional[np.ndarray] = None
    added_node_masses: Dict[int, float] = field(default_factory=dict)

    def add_nodal_load(
        self,
        node_id: int,
        load_vector: Optional[np.ndarray] = None,
        forces: Optional[np.ndarray] = None,
        moments: Optional[np.ndarray] = None,
    ):
        """
        Add a nodal load.

        Args:
            node_id: Node ID to apply load to.
            load_vector: [Fx, Fy, Fz] or [Fx, Fy, Fz, Mx, My, Mz].
            forces: Alternative force vector [Fx, Fy, Fz].
            moments: Optional moment vector [Mx, My, Mz].
        """
        if load_vector is not None:
            load_vector = np.asarray(load_vector, dtype=float)
            if len(load_vector) == 6:
                load = load_vector.copy()
            else:
                if moments is None:
                    moments = np.zeros(3)
                load = np.concatenate([load_vector[:3], np.asarray(moments, dtype=float)[:3]])
        elif forces is not None:
            if moments is None:
                moments = np.zeros(3)
            load = np.concatenate([np.asarray(forces, dtype=float)[:3], np.asarray(moments, dtype=float)[:3]])
        elif moments is not None:
            load = np.concatenate([np.zeros(3), np.asarray(moments, dtype=float)[:3]])
        else:
            load = np.zeros(6)

        if node_id in self.nodal_loads:
            self.nodal_loads[node_id] += load
        else:
            self.nodal_loads[node_id] = load

    def add_pressure_load(self, element_id: int, pressure: float):
        """
        Add a pressure load to a shell element.

        Positive pressure follows the element normal as defined by the element
        node ordering and natural-coordinate surface Jacobian.
        """
        self.pressure_loads[element_id] = float(pressure)

    def set_gravity(self, gx: float = 0.0, gy: float = 0.0, gz: float = -9.81):
        """Set gravity acceleration."""
        self.gravity = np.array([gx, gy, gz], dtype=float)

    def set_acceleration(self, ax: float = 0.0, ay: float = 0.0, az: float = 0.0):
        """Set a body-load acceleration field in x/y/z.

        Produces the consistent inertial load ``M a`` over the structural mass
        (element mass matrices) plus ``m_i a`` for any added nodal masses.  This
        is the same mechanism as :meth:`set_gravity`; use it to describe design
        accelerations (e.g. ship motions) in an arbitrary direction.
        """
        self.gravity = np.array([ax, ay, az], dtype=float)

    def add_node_mass(self, node_id: int, mass: float):
        """Add a lumped translational mass at a node.

        The added mass contributes an inertial load ``mass * acceleration`` at
        the node's translational DOFs whenever an acceleration/gravity field is
        set.  Use the frontend edge/ring helpers to distribute a total mass
        along a plate edge or a cylinder top/bottom ring.
        """
        mass = float(mass)
        if mass == 0.0:
            return
        self.added_node_masses[int(node_id)] = self.added_node_masses.get(int(node_id), 0.0) + mass

    def add_distributed_edge_mass(self, node_ids: Sequence[int], total_mass: float):
        """Distribute ``total_mass`` equally over the given nodes."""
        node_ids = [int(node_id) for node_id in node_ids]
        if not node_ids or float(total_mass) == 0.0:
            return
        share = float(total_mass) / float(len(node_ids))
        for node_id in node_ids:
            self.add_node_mass(node_id, share)

    @staticmethod
    def _surface_jacobian_and_normal(coords: np.ndarray, dN_dxi: np.ndarray, dN_deta: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute surface Jacobian magnitude and unit normal from shape derivatives.
        """
        tangent_xi = coords.T @ dN_dxi
        tangent_eta = coords.T @ dN_deta
        normal_raw = np.array(
            [
                tangent_xi[1] * tangent_eta[2] - tangent_xi[2] * tangent_eta[1],
                tangent_xi[2] * tangent_eta[0] - tangent_xi[0] * tangent_eta[2],
                tangent_xi[0] * tangent_eta[1] - tangent_xi[1] * tangent_eta[0],
            ]
        )
        det_j = float(np.linalg.norm(normal_raw))
        if det_j < _SMALL:
            return 0.0, np.array([0.0, 0.0, 1.0])
        return det_j, normal_raw / det_j

    @staticmethod
    def _fallback_lumped_pressure_load(element, mesh: "FEMesh", pressure: float) -> np.ndarray:
        """
        Fallback for unsupported element topologies.

        This keeps old behaviour available for non-shell or future custom elements,
        but all 4/8-node shell elements should use the consistent path.
        """
        coords = element.get_node_coordinates(mesh)
        num_nodes = len(element.node_ids)
        f_elem = np.zeros(num_nodes * 6)
        if num_nodes < 3:
            return f_elem

        if num_nodes in (4, 8):
            tri1_area = 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
            tri2_area = 0.5 * np.linalg.norm(np.cross(coords[0] - coords[2], coords[3] - coords[2]))
            area = tri1_area + tri2_area
        else:
            area = 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))

        normal_raw = np.cross(coords[1] - coords[0], coords[2] - coords[0])
        normal_norm = np.linalg.norm(normal_raw)
        normal = normal_raw / normal_norm if normal_norm > _SMALL else np.array([0.0, 0.0, 1.0])
        nodal_force = pressure * area / max(num_nodes, 1) * normal
        for i in range(num_nodes):
            f_elem[i * 6:i * 6 + 3] += nodal_force
        return f_elem

    def _consistent_pressure_load(self, element, mesh: "FEMesh", pressure: float) -> np.ndarray:
        """
        Assemble a consistent element pressure vector.

        For a shell element with shape functions N_i, the translational nodal
        load is:

            f_i = integral_A N_i * p * n dA

        Rotational pressure follower effects are deliberately not included here;
        this is a linear static/eigen-prep load vector.
        """
        if not hasattr(element, "compute_shape_functions") or not hasattr(element, "gauss_points"):
            return self._fallback_lumped_pressure_load(element, mesh, pressure)

        coords = element.get_node_coordinates(mesh)
        num_nodes = len(element.node_ids)
        f_elem = np.zeros(num_nodes * 6)
        gauss_points = getattr(element, "gauss_points")
        gauss_weights = getattr(element, "gauss_weights")

        for (xi, eta), weight in zip(gauss_points, gauss_weights):
            N, dN_dxi, dN_deta = element.compute_shape_functions(float(xi), float(eta))
            det_j, normal = self._surface_jacobian_and_normal(coords, dN_dxi, dN_deta)
            if det_j < _SMALL:
                continue
            for i in range(num_nodes):
                f_elem[i * 6:i * 6 + 3] += N[i] * pressure * normal * det_j * float(weight)
        return f_elem

    def _consistent_gravity_load(
        self,
        element,
        mesh: "FEMesh",
        material: "Material",
    ) -> np.ndarray:
        """
        Assemble element body force from the element mass matrix.

        With translational acceleration a, the consistent nodal load is M a.
        Rotational acceleration components are zero for ordinary gravity.
        """
        f_elem = np.zeros(len(element.node_ids) * 6)
        if self.gravity is None:
            return f_elem

        mass_matrix = element.compute_mass_matrix(mesh, material)
        acceleration = np.zeros_like(f_elem)
        for i in range(len(element.node_ids)):
            acceleration[i * 6:i * 6 + 3] = self.gravity
        return np.asarray(mass_matrix @ acceleration, dtype=float).reshape(-1)

    def get_load_vector(
        self,
        mesh: "FEMesh",
        dof_manager: "DOFManager",
        material_getter: Optional[Callable[[str], "Material"]] = None,
    ) -> np.ndarray:
        """Assemble the global load vector."""
        total_dofs = dof_manager.total_dofs
        F = np.zeros(total_dofs)

        # Nodal loads.
        for node_id, load in self.nodal_loads.items():
            node = mesh.get_node(node_id)
            if node:
                for i, dof in enumerate(node.dofs):
                    if i < len(load):
                        F[dof] += load[i]

        # User-provided element load vectors.
        for element_id, load in self.element_loads.items():
            element = mesh.get_element(element_id)
            if element is None:
                continue
            dof_mapping = element.get_dof_mapping(mesh)
            load = np.asarray(load, dtype=float)
            for i, dof in enumerate(dof_mapping):
                if i < len(load):
                    F[dof] += load[i]

        # Consistent pressure loads for shell elements.
        for element_id, pressure in self.pressure_loads.items():
            element = mesh.get_element(element_id)
            if element is None or not hasattr(element, "node_ids"):
                continue
            f_elem = self._consistent_pressure_load(element, mesh, pressure)
            dof_mapping = element.get_dof_mapping(mesh)
            for i, dof in enumerate(dof_mapping):
                if i < len(f_elem):
                    F[dof] += f_elem[i]

        # Gravity loads from element mass matrices.
        if self.gravity is not None:
            for element in mesh.elements.values():
                if not hasattr(element, "node_ids"):
                    continue
                if material_getter is None:
                    material = _GravityFallbackMaterial()
                else:
                    material = material_getter(element.material_name)
                f_elem = self._consistent_gravity_load(element, mesh, material)
                dof_mapping = element.get_dof_mapping(mesh)
                for i, dof in enumerate(dof_mapping):
                    if i < len(f_elem):
                        F[dof] += f_elem[i]

        # Inertial load from added masses under the acceleration field: both
        # model-level point masses (which also enter the mass matrix) and any
        # load-case-only added masses.
        if self.gravity is not None:
            acceleration = np.asarray(self.gravity, dtype=float)
            combined_masses: Dict[int, float] = {}
            for node_id, mass in getattr(mesh, "point_masses", {}).items():
                combined_masses[int(node_id)] = combined_masses.get(int(node_id), 0.0) + float(mass)
            for node_id, mass in self.added_node_masses.items():
                combined_masses[int(node_id)] = combined_masses.get(int(node_id), 0.0) + float(mass)
            for node_id, mass in combined_masses.items():
                node = mesh.get_node(int(node_id))
                if node is None or mass == 0.0:
                    continue
                for axis in range(3):
                    F[node.dofs[axis]] += float(mass) * acceleration[axis]

        return F


@dataclass
class InPlaneLoad:
    """
    In-plane load for stiffened panels.

    Represents axial, transverse and shear stresses applied to the panel.
    """

    axial_stress: float = 0.0
    transverse_stress: float = 0.0
    shear_stress: float = 0.0
    pressure: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "axial_stress": self.axial_stress,
            "transverse_stress": self.transverse_stress,
            "shear_stress": self.shear_stress,
            "pressure": self.pressure,
        }


class LoadCombination:
    """Linear combination of load cases."""

    def __init__(self, name: str, factors: Dict[str, float]):
        self.name = name
        self.factors = factors

    def get_combined_load_vector(
        self,
        load_cases: List[LoadCase],
        mesh: "FEMesh",
        dof_manager: "DOFManager",
    ) -> np.ndarray:
        F_total = np.zeros(dof_manager.total_dofs)
        for load_case in load_cases:
            if load_case.name in self.factors:
                factor = self.factors[load_case.name]
                F_total += factor * load_case.get_load_vector(mesh, dof_manager)
        return F_total


# Common boundary condition factory functions

def create_fixed_support(name: str, node_ids: List[int]) -> FixedSupport:
    """Create a fixed support boundary condition."""
    return FixedSupport(name, node_ids)


def create_pinned_support(name: str, node_ids: List[int]) -> PinnedSupport:
    """Create a pinned support boundary condition."""
    return PinnedSupport(name, node_ids)


def create_roller_support(
    name: str,
    node_ids: List[int],
    constrained_directions: Optional[List[str]] = None,
) -> RollerSupport:
    """Create a roller support boundary condition."""
    return RollerSupport(name, node_ids, constrained_directions)


def create_symmetry_bc(name: str, node_ids: List[int], symmetry_plane: str = "xy") -> SymmetryBC:
    """Create a symmetry boundary condition."""
    return SymmetryBC(name, node_ids, symmetry_plane)
