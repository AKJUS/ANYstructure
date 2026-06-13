"""
Finite Element Implementations

This module contains element formulations for:
- ShellElement: 4/8-node quadrilateral Mindlin-Reissner shell element
- BeamElement: 2-node Timoshenko beam element with 6 DOF/node
- QuadraticBeamElement: 3-node quadratic Timoshenko beam element
- CoupledBeamShellElement: kinematic MPC element for eccentric beam-shell interaction

Shell convention
----------------
The shell element forms stiffness and stresses in an element-local orthonormal
basis at each integration point:

    local x = projected xi tangent direction
    local y = in-surface direction perpendicular to local x
    local z = shell normal

Global nodal translations and rotations are transformed to this local basis
before the membrane, bending and transverse shear B-matrices are evaluated.

Shell shear treatment
---------------------
Membrane and bending always use full integration for the element topology.
Transverse shear depends on the topology:

    * 4-node: MITC4 assumed natural shear (covariant shear sampled at the four
      edge midpoints and interpolated), integrated at the full 2x2 rule.  This
      avoids both shear locking and the spurious zero-energy w-hourglass mode
      of one-point reduced shear integration.
    * 8-node: reduced 2x2 shear integration (S8R style).

Beam section convention
-----------------------
Beam local axes are (x = member axis, y, z).  ``Iy`` is the second moment of
area about the local y axis and governs bending that deflects the beam in
local z; ``Iz`` governs deflection in local y.  ``shear_factor_y`` scales the
shear area for transverse shear force in local y, ``shear_factor_z`` for local
z.  The optional ``cross_section["orientation"]`` vector prescribes the local
z direction (e.g. the stiffener web direction).  Without it, a heuristic picks
local y close to global Y (or global Z for members nearly parallel to Y),
which leaves the section orientation unconstrained for asymmetric sections.

Beam-shell coupling
-------------------
Beam-shell coupling is represented as a linear multi-point constraint (MPC),
not as a large penalty spring.  For a beam node offset from a shell node by
vector r, the coupling relation is:

    u_beam = u_shell + theta_shell x r
    theta_beam = theta_shell

The assembly solver eliminates these slave beam DOFs through a transformation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from .plasticity import lobatto_layers, plane_stress_elastic_matrix, plane_stress_return_map

if TYPE_CHECKING:
    from .fe_core import FEMesh, Material


_SMALL = 1.0e-12


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors without np.cross dispatch overhead."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
        dtype=float,
    )


def _inv2(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """Inverse and determinant of a 2x2 matrix without LAPACK overhead."""
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(det) < _SMALL:
        raise np.linalg.LinAlgError("singular 2x2 matrix")
    inv = np.array(
        [[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]],
        dtype=float,
    ) / det
    return inv, float(det)


def _section_orientation(cross_section: Dict[str, Any]) -> Optional[np.ndarray]:
    """Return the prescribed local-z (web) direction from a cross-section dict."""
    value = cross_section.get("orientation", cross_section.get("web_direction"))
    if value is None:
        return None
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size < 3 or float(np.linalg.norm(vector[:3])) < _SMALL:
        return None
    return np.array(vector[:3], dtype=float)


def _beam_rotation_matrix(e1: np.ndarray, orientation: Optional[np.ndarray]) -> np.ndarray:
    """Build the beam local frame [e1 e2 e3] with optional prescribed local z.

    ``orientation`` is the requested local z direction (section web direction).
    It is projected perpendicular to the member axis; if it is (nearly)
    parallel to the axis the heuristic fallback is used instead.
    """
    if orientation is not None:
        candidate = orientation - np.dot(orientation, e1) * e1
        norm = float(np.linalg.norm(candidate))
        if norm > 1.0e-6 * float(np.linalg.norm(orientation)):
            e3 = candidate / norm
            e2 = _cross3(e3, e1)
            e2 /= np.linalg.norm(e2)
            return np.column_stack((e1, e2, e3))
    trial = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(e1, trial))) > 0.95:
        trial = np.array([0.0, 0.0, 1.0])
    e2 = trial - np.dot(trial, e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = _cross3(e1, e2)
    e3 /= np.linalg.norm(e3)
    return np.column_stack((e1, e2, e3))


class Element(ABC):
    """Abstract base class for all FE elements."""

    def __init__(self, element_id: int, node_ids: List[int], material_name: str = "default"):
        self.element_id = element_id
        self.node_ids = node_ids
        self.material_name = material_name
        self._stiffness_matrix: Optional[np.ndarray] = None
        self._mass_matrix: Optional[np.ndarray] = None
        self._internal_forces: Optional[np.ndarray] = None

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def dofs_per_node(self) -> int:
        raise NotImplementedError

    @property
    def total_dofs(self) -> int:
        return self.num_nodes * self.dofs_per_node

    @abstractmethod
    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        raise NotImplementedError

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        return np.zeros((self.total_dofs, self.total_dofs))

    def compute_geometric_stiffness_matrix(
        self,
        mesh: "FEMesh",
        material: "Material",
        state: Optional[Any] = None,
    ) -> np.ndarray:
        return np.zeros((self.total_dofs, self.total_dofs))

    def compute_internal_forces(
        self, mesh: "FEMesh", displacements: np.ndarray, material: "Material"
    ) -> np.ndarray:
        return np.zeros(self.total_dofs)

    def compute_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any] = None,
        num_layers: int = 5,
        tangent: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        """Internal force vector, tangent stiffness and trial state at u_elem.

        The default is linear elastic: F = K u with the constant stiffness as
        tangent.  Elements supporting geometric and/or material nonlinearity
        override this.  The returned state is a trial state; the incremental
        solver commits it only when the load step converges.  With
        ``tangent=False`` the stiffness entry may be None (used by the line
        search, which only needs residuals).
        """
        K = self._stiffness_matrix
        if K is None:
            K = self.compute_stiffness_matrix(mesh, material)
        return K @ u_elem, (K if tangent else None), state

    def compute_stresses(
        self, mesh: "FEMesh", displacements: np.ndarray, material: "Material"
    ) -> Dict[str, np.ndarray]:
        return {}

    def get_dof_mapping(self, mesh: "FEMesh") -> List[int]:
        dof_mapping: List[int] = []
        for node_id in self.node_ids:
            node = mesh.get_node(node_id)
            if node:
                dof_mapping.extend(node.dofs)
        return dof_mapping

    def _get_element_displacements(self, mesh: "FEMesh", displacements: np.ndarray) -> np.ndarray:
        u = np.asarray(displacements, dtype=float)
        if u.size == self.total_dofs:
            return u.copy()
        dof_mapping = self.get_dof_mapping(mesh)
        if not dof_mapping:
            return np.zeros(self.total_dofs)
        dof_indices = np.asarray(dof_mapping, dtype=np.intp)
        if int(dof_indices.max()) >= u.size:
            raise IndexError(
                f"Displacement vector has length {u.size}, but element {self.element_id} "
                f"requires global DOF {int(dof_indices.max())}."
            )
        return u[dof_indices]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "element_id": self.element_id,
            "type": self.__class__.__name__,
            "node_ids": self.node_ids,
            "material_name": self.material_name,
        }


class ShellElement(Element):
    """4/8-node quadrilateral Mindlin-Reissner shell element."""

    GAUSS_POINTS_1x1 = np.array([[0.0, 0.0]], dtype=float)
    GAUSS_WEIGHTS_1x1 = np.array([4.0], dtype=float)

    GAUSS_POINTS_2x2 = np.array(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]], dtype=float
    ) / np.sqrt(3.0)
    GAUSS_WEIGHTS_2x2 = np.ones(4, dtype=float)

    GAUSS_POINTS_3x3 = np.array(
        [
            [-np.sqrt(3.0 / 5.0), -np.sqrt(3.0 / 5.0)],
            [0.0, -np.sqrt(3.0 / 5.0)],
            [np.sqrt(3.0 / 5.0), -np.sqrt(3.0 / 5.0)],
            [-np.sqrt(3.0 / 5.0), 0.0],
            [0.0, 0.0],
            [np.sqrt(3.0 / 5.0), 0.0],
            [-np.sqrt(3.0 / 5.0), np.sqrt(3.0 / 5.0)],
            [0.0, np.sqrt(3.0 / 5.0)],
            [np.sqrt(3.0 / 5.0), np.sqrt(3.0 / 5.0)],
        ],
        dtype=float,
    )
    GAUSS_WEIGHTS_3x3 = np.array(
        [
            25.0 / 81.0,
            40.0 / 81.0,
            25.0 / 81.0,
            40.0 / 81.0,
            64.0 / 81.0,
            40.0 / 81.0,
            25.0 / 81.0,
            40.0 / 81.0,
            25.0 / 81.0,
        ],
        dtype=float,
    )

    def __init__(
        self,
        element_id: int,
        node_ids: List[int],
        material_name: str = "default",
        thickness: float = 0.01,
    ):
        super().__init__(element_id, node_ids, material_name)
        self.thickness = float(thickness)
        self._is_8node = len(node_ids) == 8
        self._is_4node = len(node_ids) == 4
        if not (self._is_4node or self._is_8node):
            raise ValueError(f"ShellElement requires 4 or 8 nodes, got {len(node_ids)}")

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def dofs_per_node(self) -> int:
        return 6

    @property
    def gauss_points(self) -> np.ndarray:
        return self.GAUSS_POINTS_3x3 if self._is_8node else self.GAUSS_POINTS_2x2

    @property
    def gauss_weights(self) -> np.ndarray:
        return self.GAUSS_WEIGHTS_3x3 if self._is_8node else self.GAUSS_WEIGHTS_2x2

    @property
    def shear_gauss_points(self) -> np.ndarray:
        return self.GAUSS_POINTS_2x2 if self._is_8node else self.GAUSS_POINTS_1x1

    @property
    def shear_gauss_weights(self) -> np.ndarray:
        return self.GAUSS_WEIGHTS_2x2 if self._is_8node else self.GAUSS_WEIGHTS_1x1

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        coords = np.zeros((self.num_nodes, 3), dtype=float)
        for i, node_id in enumerate(self.node_ids):
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"Shell element {self.element_id} references missing node {node_id}")
            coords[i] = node.coords()
        return coords

    def compute_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._is_4node:
            return self._compute_4node_shape_functions(xi, eta)
        return self._compute_8node_shape_functions(xi, eta)

    def _compute_4node_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = np.array(
            [
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta),
            ],
            dtype=float,
        )
        dN_dxi = np.array(
            [
                -0.25 * (1.0 - eta),
                0.25 * (1.0 - eta),
                0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta),
            ],
            dtype=float,
        )
        dN_deta = np.array(
            [
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                0.25 * (1.0 + xi),
                0.25 * (1.0 - xi),
            ],
            dtype=float,
        )
        return N, dN_dxi, dN_deta

    def _compute_8node_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = np.zeros(8, dtype=float)
        N[0] = -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta)
        N[1] = -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta)
        N[2] = -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta)
        N[3] = -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta)
        N[4] = 0.5 * (1.0 - xi**2) * (1.0 - eta)
        N[5] = 0.5 * (1.0 + xi) * (1.0 - eta**2)
        N[6] = 0.5 * (1.0 - xi**2) * (1.0 + eta)
        N[7] = 0.5 * (1.0 - xi) * (1.0 - eta**2)

        dN_dxi = np.zeros(8, dtype=float)
        dN_dxi[0] = 0.25 * (1.0 - eta) * (1.0 + xi + eta) - 0.25 * (1.0 - xi) * (1.0 - eta)
        dN_dxi[1] = -0.25 * (1.0 - eta) * (1.0 - xi + eta) + 0.25 * (1.0 + xi) * (1.0 - eta)
        dN_dxi[2] = -0.25 * (1.0 + eta) * (1.0 - xi - eta) + 0.25 * (1.0 + xi) * (1.0 + eta)
        dN_dxi[3] = 0.25 * (1.0 + eta) * (1.0 + xi - eta) - 0.25 * (1.0 - xi) * (1.0 + eta)
        dN_dxi[4] = -xi * (1.0 - eta)
        dN_dxi[5] = 0.5 * (1.0 - eta**2)
        dN_dxi[6] = -xi * (1.0 + eta)
        dN_dxi[7] = -0.5 * (1.0 - eta**2)

        dN_deta = np.zeros(8, dtype=float)
        dN_deta[0] = 0.25 * (1.0 - xi) * (1.0 + xi + eta) - 0.25 * (1.0 - xi) * (1.0 - eta)
        dN_deta[1] = 0.25 * (1.0 + xi) * (1.0 - xi + eta) - 0.25 * (1.0 + xi) * (1.0 - eta)
        dN_deta[2] = -0.25 * (1.0 + xi) * (1.0 - xi - eta) + 0.25 * (1.0 + xi) * (1.0 + eta)
        dN_deta[3] = -0.25 * (1.0 - xi) * (1.0 + xi - eta) + 0.25 * (1.0 - xi) * (1.0 + eta)
        dN_deta[4] = -0.5 * (1.0 - xi**2)
        dN_deta[5] = -eta * (1.0 + xi)
        dN_deta[6] = 0.5 * (1.0 - xi**2)
        dN_deta[7] = -eta * (1.0 - xi)
        return N, dN_dxi, dN_deta

    def compute_jacobian(self, coords: np.ndarray, dN_dxi: np.ndarray, dN_deta: np.ndarray) -> np.ndarray:
        return np.array([coords.T @ dN_dxi, coords.T @ dN_deta], dtype=float)

    @staticmethod
    def _normalize(vector: np.ndarray) -> Tuple[np.ndarray, float]:
        norm = float(np.sqrt(vector @ vector))
        if norm < _SMALL:
            return np.zeros(3, dtype=float), norm
        return vector / norm, norm

    def _fallback_edge_direction(self, coords: np.ndarray, normal: np.ndarray) -> np.ndarray:
        candidate_edges = []
        if coords.shape[0] >= 2:
            candidate_edges.append(coords[1] - coords[0])
        if coords.shape[0] >= 4:
            candidate_edges.append(coords[2] - coords[3])
            candidate_edges.append(coords[3] - coords[0])
            candidate_edges.append(coords[2] - coords[1])
        for edge in candidate_edges:
            projected = edge - np.dot(edge, normal) * normal
            e1, n = self._normalize(projected)
            if n > _SMALL:
                return e1
        raise ValueError(f"Shell element {self.element_id} has no valid in-plane direction")

    def _local_frame_and_derivatives(
        self,
        coords: np.ndarray,
        dN_dxi: np.ndarray,
        dN_deta: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        J = self.compute_jacobian(coords, dN_dxi, dN_deta)
        tangent_xi = J[0]
        tangent_eta = J[1]
        e3, det_j = self._normalize(_cross3(tangent_xi, tangent_eta))
        if det_j < _SMALL:
            raise ValueError(f"Shell element {self.element_id} has a near-zero surface Jacobian")

        e1_raw = tangent_xi - np.dot(tangent_xi, e3) * e3
        e1, e1_norm = self._normalize(e1_raw)
        if e1_norm < _SMALL:
            e1 = self._fallback_edge_direction(coords, e3)
        e2, e2_norm = self._normalize(_cross3(e3, e1))
        if e2_norm < _SMALL:
            raise ValueError(f"Shell element {self.element_id} has an invalid local y direction")
        e1, _ = self._normalize(_cross3(e2, e3))
        R = np.column_stack((e1, e2, e3))

        J_local = np.array(
            [
                [np.dot(tangent_xi, e1), np.dot(tangent_xi, e2)],
                [np.dot(tangent_eta, e1), np.dot(tangent_eta, e2)],
            ],
            dtype=float,
        )
        try:
            inv_j_local, _ = _inv2(J_local)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Shell element {self.element_id} has a singular local Jacobian") from exc

        dN_dx = inv_j_local[0, 0] * dN_dxi + inv_j_local[0, 1] * dN_deta
        dN_dy = inv_j_local[1, 0] * dN_dxi + inv_j_local[1, 1] * dN_deta
        return R, dN_dx, dN_dy, det_j

    def _local_dof_transform(self, R: np.ndarray) -> np.ndarray:
        n_blocks = 2 * self.num_nodes
        T = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        blocks = T.reshape(n_blocks, 3, n_blocks, 3)
        indices = np.arange(n_blocks)
        blocks[indices, :, indices, :] = R.T
        return T

    def _build_shell_b_matrices(
        self,
        N: np.ndarray,
        dN_dx: np.ndarray,
        dN_dy: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B_m = np.zeros((3, self.total_dofs), dtype=float)
        B_b = np.zeros((3, self.total_dofs), dtype=float)
        B_s = np.zeros((2, self.total_dofs), dtype=float)

        B_m[0, 0::6] = dN_dx
        B_m[1, 1::6] = dN_dy
        B_m[2, 0::6] = dN_dy
        B_m[2, 1::6] = dN_dx

        B_b[0, 4::6] = dN_dx
        B_b[1, 3::6] = -dN_dy
        B_b[2, 4::6] = dN_dy
        B_b[2, 3::6] = -dN_dx

        B_s[0, 2::6] = dN_dx
        B_s[0, 4::6] = N
        B_s[1, 2::6] = dN_dy
        B_s[1, 3::6] = -N
        return B_m, B_b, B_s

    def _build_drilling_b_matrix(
        self,
        N: np.ndarray,
        dN_dx: np.ndarray,
        dN_dy: np.ndarray,
    ) -> np.ndarray:
        """
        Build a small drilling stabilization strain.

        The stabilized quantity is theta_z - 0.5 * (dv/dx - du/dy), so a
        physical rigid rotation about the shell normal has exactly zero energy.
        """
        B_d = np.zeros((1, self.total_dofs), dtype=float)
        B_d[0, 0::6] = 0.5 * dN_dy
        B_d[0, 1::6] = -0.5 * dN_dx
        B_d[0, 5::6] = N
        return B_d

    # MITC4 assumed natural transverse shear (4-node elements only).
    #
    # Covariant shear strains are sampled where they are exact for pure
    # bending, at the edge midpoints A(0,-1), B(1,0), C(0,1), D(-1,0):
    #
    #     gamma_xi(xi, eta)  = (1-eta)/2 * gamma_xi|A  + (1+eta)/2 * gamma_xi|C
    #     gamma_eta(xi, eta) = (1-xi)/2  * gamma_eta|D + (1+xi)/2  * gamma_eta|B
    #
    # with gamma_xi = dw/dxi + x_,xi * theta_y - y_,xi * theta_x in a fixed
    # element-plane frame, then mapped to Cartesian shear through the inverse
    # in-plane Jacobian at the integration point.  The element geometry is
    # treated as a flat facet in the frame evaluated at the element centre.
    _MITC4_SAMPLE_POINTS = {"A": (0.0, -1.0), "B": (1.0, 0.0), "C": (0.0, 1.0), "D": (-1.0, 0.0)}

    def _center_frame(self, coords: np.ndarray) -> np.ndarray:
        _, dN_dxi, dN_deta = self.compute_shape_functions(0.0, 0.0)
        R, _, _, _ = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
        return R

    def _mitc4_shear_samples(self, coords: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """Return in-plane node coordinates and covariant shear rows at A-D."""
        planar = coords @ R[:, :2]
        samples: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, (xi, eta) in self._MITC4_SAMPLE_POINTS.items():
            N, dN_dxi, dN_deta = self.compute_shape_functions(xi, eta)
            x_xi = float(dN_dxi @ planar[:, 0])
            y_xi = float(dN_dxi @ planar[:, 1])
            x_eta = float(dN_deta @ planar[:, 0])
            y_eta = float(dN_deta @ planar[:, 1])
            row_xi = np.zeros(self.total_dofs, dtype=float)
            row_eta = np.zeros(self.total_dofs, dtype=float)
            row_xi[2::6] = dN_dxi
            row_xi[3::6] = -N * y_xi
            row_xi[4::6] = N * x_xi
            row_eta[2::6] = dN_deta
            row_eta[3::6] = -N * y_eta
            row_eta[4::6] = N * x_eta
            samples[name] = (row_xi, row_eta)
        return planar, samples

    def _mitc4_shear_b_matrix(
        self,
        planar: np.ndarray,
        samples: Dict[str, Tuple[np.ndarray, np.ndarray]],
        xi: float,
        eta: float,
    ) -> Tuple[np.ndarray, float]:
        """Assumed-shear B matrix (local Cartesian) and in-plane Jacobian det."""
        _, dN_dxi, dN_deta = self.compute_shape_functions(xi, eta)
        J2 = np.array(
            [
                [float(dN_dxi @ planar[:, 0]), float(dN_dxi @ planar[:, 1])],
                [float(dN_deta @ planar[:, 0]), float(dN_deta @ planar[:, 1])],
            ],
            dtype=float,
        )
        try:
            inv_j2, det_j = _inv2(J2)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Shell element {self.element_id} has a singular in-plane Jacobian") from exc
        B_covariant = np.vstack(
            [
                0.5 * (1.0 - eta) * samples["A"][0] + 0.5 * (1.0 + eta) * samples["C"][0],
                0.5 * (1.0 - xi) * samples["D"][1] + 0.5 * (1.0 + xi) * samples["B"][1],
            ]
        )
        return inv_j2 @ B_covariant, det_j

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        E = material.elastic_modulus
        nu = material.poisson_ratio
        G = material.shear_modulus
        h = self.thickness
        kappa = 5.0 / 6.0

        shell_plane = np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]],
            dtype=float,
        )
        D_membrane = E * h / (1.0 - nu**2) * shell_plane
        D_bending = E * h**3 / (12.0 * (1.0 - nu**2)) * shell_plane
        D_shear = G * kappa * h * np.eye(2, dtype=float)

        K = np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
            N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
            R, dN_dx, dN_dy, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
            T = self._local_dof_transform(R)
            B_m, B_b, _ = self._build_shell_b_matrices(N, dN_dx, dN_dy)
            B_d = self._build_drilling_b_matrix(N, dN_dx, dN_dy)
            K_local = (B_m.T @ D_membrane @ B_m + B_b.T @ D_bending @ B_b) * det_j * weight

            drilling_stiffness = D_membrane[0, 0] * 1.0e-6
            K_local += (B_d.T @ (drilling_stiffness * np.eye(1)) @ B_d) * det_j * weight
            K += T.T @ K_local @ T

        if self._is_4node:
            R = self._center_frame(coords)
            T = self._local_dof_transform(R)
            planar, samples = self._mitc4_shear_samples(coords, R)
            for (xi, eta), weight in zip(self.GAUSS_POINTS_2x2, self.GAUSS_WEIGHTS_2x2):
                B_s, det_j = self._mitc4_shear_b_matrix(planar, samples, float(xi), float(eta))
                K_local = (B_s.T @ D_shear @ B_s) * det_j * weight
                K += T.T @ K_local @ T
        else:
            for (xi, eta), weight in zip(self.shear_gauss_points, self.shear_gauss_weights):
                N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
                R, dN_dx, dN_dy, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
                T = self._local_dof_transform(R)
                _, _, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
                K_local = (B_s.T @ D_shear @ B_s) * det_j * weight
                K += T.T @ K_local @ T

        self._stiffness_matrix = K
        return K

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        rho = material.density
        h = self.thickness
        M = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
            N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
            R, _, _, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
            T = self._local_dof_transform(R)
            M_local = np.zeros_like(M)
            outer_n = np.outer(N, N) * det_j * weight
            translational = rho * h * outer_n
            rotational = rho * h**3 / 12.0 * outer_n
            for d in range(3):
                M_local[d::6, d::6] += translational
                M_local[3 + d::6, 3 + d::6] += rotational
            M += T.T @ M_local @ T
        self._mass_matrix = M
        return M

    @staticmethod
    def _membrane_compression_from_state(state: Optional[Any]) -> Tuple[float, float, float]:
        """Return local membrane resultants with compression-positive convention."""
        if state is None or not isinstance(state, dict):
            return 0.0, 0.0, 0.0

        if "membrane_compression" in state:
            values = np.asarray(state["membrane_compression"], dtype=float).reshape(-1)
            if values.size >= 3:
                return float(values[0]), float(values[1]), float(values[2])
        if "membrane_forces" in state:
            values = np.asarray(state["membrane_forces"], dtype=float).reshape(-1)
            if values.size >= 3:
                return -float(values[0]), -float(values[1]), -float(values[2])

        compression_x = state.get("membrane_compression_x", state.get("Nx_compression"))
        compression_y = state.get("membrane_compression_y", state.get("Ny_compression"))
        compression_xy = state.get("membrane_compression_xy", state.get("Nxy_compression"))
        if compression_x is not None or compression_y is not None or compression_xy is not None:
            return (
                float(compression_x or 0.0),
                float(compression_y or 0.0),
                float(compression_xy or 0.0),
            )

        force_x = state.get("membrane_force_x", state.get("Nx"))
        force_y = state.get("membrane_force_y", state.get("Ny"))
        force_xy = state.get("membrane_force_xy", state.get("Nxy"))
        return (
            -float(force_x or 0.0),
            -float(force_y or 0.0),
            -float(force_xy or 0.0),
        )

    def compute_geometric_stiffness_matrix(
        self,
        mesh: "FEMesh",
        material: "Material",
        state: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Shell stress-stiffness matrix from membrane resultants.

        The state can supply either tension-positive membrane forces
        (``membrane_force_x/y/xy`` or ``membrane_forces``) or
        compression-positive values (``membrane_compression_x/y/xy``).  The
        returned matrix follows the package convention
        ``K phi = lambda KG phi`` with compression destabilizing.
        """
        Nx, Ny, Nxy = self._membrane_compression_from_state(state)
        if Nx == 0.0 and Ny == 0.0 and Nxy == 0.0:
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        coords = self.get_node_coordinates(mesh)
        KG = np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
            N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
            R, dN_dx, dN_dy, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
            T = self._local_dof_transform(R)
            G = np.zeros((2, self.total_dofs), dtype=float)
            G[0, 2::6] = dN_dx
            G[1, 2::6] = dN_dy
            N_matrix = np.array([[Nx, Nxy], [Nxy, Ny]], dtype=float)
            KG_local = (G.T @ N_matrix @ G) * det_j * float(weight)
            KG += T.T @ KG_local @ T
        return KG

    # ------------------------------------------------------------------
    # Incremental nonlinear response: total-Lagrangian von Karman membrane
    # kinematics in the flat-facet centre frame, layered J2 plane-stress
    # plasticity through the thickness, elastic transverse shear (MITC4 for
    # 4-node, reduced for 8-node) and elastic drilling stabilization.
    # ------------------------------------------------------------------

    def _nonlinear_geometry(self, mesh: "FEMesh") -> Dict[str, Any]:
        """Reference-configuration data, computed once per element."""
        cache = getattr(self, "_nl_cache", None)
        if cache is not None:
            return cache
        coords = self.get_node_coordinates(mesh)
        R0 = self._center_frame(coords)
        T0 = self._local_dof_transform(R0)
        planar = coords @ R0[:, :2]

        gp_data = []
        for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
            N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
            J2 = np.array(
                [
                    [float(dN_dxi @ planar[:, 0]), float(dN_dxi @ planar[:, 1])],
                    [float(dN_deta @ planar[:, 0]), float(dN_deta @ planar[:, 1])],
                ],
                dtype=float,
            )
            inv_j2, det_j = _inv2(J2)
            dN_dx = inv_j2[0, 0] * dN_dxi + inv_j2[0, 1] * dN_deta
            dN_dy = inv_j2[1, 0] * dN_dxi + inv_j2[1, 1] * dN_deta
            B_m, B_b, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
            B_d = self._build_drilling_b_matrix(N, dN_dx, dN_dy)
            Gw = np.zeros((2, self.total_dofs), dtype=float)
            Gw[0, 2::6] = dN_dx
            Gw[1, 2::6] = dN_dy
            gp_data.append(
                {
                    "B_m": B_m,
                    "B_b": B_b,
                    "B_d": B_d,
                    "Gw": Gw,
                    "detw": abs(det_j) * float(weight),
                }
            )

        shear_data = []
        if self._is_4node:
            _, samples = self._mitc4_shear_samples(coords, R0)
            for (xi, eta), weight in zip(self.GAUSS_POINTS_2x2, self.GAUSS_WEIGHTS_2x2):
                B_s, det_j = self._mitc4_shear_b_matrix(planar, samples, float(xi), float(eta))
                shear_data.append({"B_s": B_s, "detw": abs(det_j) * float(weight)})
        else:
            for (xi, eta), weight in zip(self.shear_gauss_points, self.shear_gauss_weights):
                N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
                J2 = np.array(
                    [
                        [float(dN_dxi @ planar[:, 0]), float(dN_dxi @ planar[:, 1])],
                        [float(dN_deta @ planar[:, 0]), float(dN_deta @ planar[:, 1])],
                    ],
                    dtype=float,
                )
                inv_j2, det_j = _inv2(J2)
                dN_dx = inv_j2[0, 0] * dN_dxi + inv_j2[0, 1] * dN_deta
                dN_dy = inv_j2[1, 0] * dN_dxi + inv_j2[1, 1] * dN_deta
                _, _, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
                shear_data.append({"B_s": B_s, "detw": abs(det_j) * float(weight)})

        cache = {"T0": T0, "gp": gp_data, "shear": shear_data}
        self._nl_cache = cache
        return cache

    def init_nonlinear_state(self, num_layers: int) -> Dict[str, np.ndarray]:
        n_points = len(self.gauss_points) * num_layers
        return {
            "plastic_strain": np.zeros((n_points, 3), dtype=float),
            "alpha": np.zeros(n_points, dtype=float),
        }

    def compute_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any] = None,
        num_layers: int = 5,
        tangent: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        cache = self._nonlinear_geometry(mesh)
        T0 = cache["T0"]
        u_loc = T0 @ np.asarray(u_elem, dtype=float)

        E = material.elastic_modulus
        nu = material.poisson_ratio
        G_mod = material.shear_modulus
        h = self.thickness
        curve = getattr(material, "hardening_curve", None)
        C_el = plane_stress_elastic_matrix(E, nu)
        D_shear = G_mod * (5.0 / 6.0) * h * np.eye(2, dtype=float)
        drilling_stiffness = C_el[0, 0] * h * 1.0e-6

        n_gp = len(cache["gp"])
        z_layers, w_layers = lobatto_layers(num_layers, h)

        if state is None:
            state = self.init_nonlinear_state(num_layers)

        # Membrane and bending strains at every integration point.
        memb_strain = np.zeros((n_gp, 3), dtype=float)
        curvature = np.zeros((n_gp, 3), dtype=float)
        B_eff_list = []
        theta_list = []
        for g, gp in enumerate(cache["gp"]):
            theta = gp["Gw"] @ u_loc  # (2,) transverse deflection gradients
            B_nl = np.vstack(
                [
                    theta[0] * gp["Gw"][0],
                    theta[1] * gp["Gw"][1],
                    theta[0] * gp["Gw"][1] + theta[1] * gp["Gw"][0],
                ]
            )
            B_eff = gp["B_m"] + B_nl
            memb_strain[g] = gp["B_m"] @ u_loc + np.array(
                [0.5 * theta[0] ** 2, 0.5 * theta[1] ** 2, theta[0] * theta[1]]
            )
            curvature[g] = gp["B_b"] @ u_loc
            B_eff_list.append(B_eff)
            theta_list.append(theta)

        if curve is None:
            # Elastic shortcut: resultants and integrated moduli in closed form.
            trial_state = state
            N_res = memb_strain @ (h * C_el).T
            M_res = curvature @ (h**3 / 12.0 * C_el).T
            C0 = np.broadcast_to(h * C_el, (n_gp, 3, 3))
            C1 = np.zeros((n_gp, 3, 3), dtype=float)
            C2 = np.broadcast_to(h**3 / 12.0 * C_el, (n_gp, 3, 3))
        else:
            # Layer strains for all (gp, layer) points at once, then one
            # vectorized return-map call for the whole element.
            layer_strain = (
                memb_strain[:, None, :] + z_layers[None, :, None] * curvature[:, None, :]
            ).reshape(n_gp * num_layers, 3)
            sigma, C_tan, ep_new, alpha_new = plane_stress_return_map(
                layer_strain,
                state["plastic_strain"],
                state["alpha"],
                E,
                nu,
                curve,
            )
            trial_state = {"plastic_strain": ep_new, "alpha": alpha_new}

            sigma = sigma.reshape(n_gp, num_layers, 3)
            C_tan = C_tan.reshape(n_gp, num_layers, 3, 3)

            # Through-thickness resultants and integrated tangent moduli.
            N_res = np.einsum("l,gli->gi", w_layers, sigma)
            M_res = np.einsum("l,l,gli->gi", w_layers, z_layers, sigma)
            C0 = np.einsum("l,glij->gij", w_layers, C_tan)
            C1 = np.einsum("l,l,glij->gij", w_layers, z_layers, C_tan)
            C2 = np.einsum("l,l,l,glij->gij", w_layers, z_layers, z_layers, C_tan)

        n_dof = self.total_dofs
        F_loc = np.zeros(n_dof, dtype=float)
        K_loc = np.zeros((n_dof, n_dof), dtype=float) if tangent else None
        for g, gp in enumerate(cache["gp"]):
            detw = gp["detw"]
            B_eff = B_eff_list[g]
            B_b = gp["B_b"]
            B_d = gp["B_d"]
            F_loc += (B_eff.T @ N_res[g] + B_b.T @ M_res[g]) * detw
            F_loc += B_d.T @ (drilling_stiffness * (B_d @ u_loc)) * detw
            if not tangent:
                continue
            K_loc += (B_eff.T @ C0[g] @ B_eff + B_b.T @ C2[g] @ B_b) * detw
            if curve is not None:
                coupling = B_eff.T @ C1[g] @ B_b
                K_loc += (coupling + coupling.T) * detw
            # Geometric (initial stress) stiffness from current membrane
            # resultants; tension-positive, so tension stiffens.
            N_mat = np.array(
                [[N_res[g, 0], N_res[g, 2]], [N_res[g, 2], N_res[g, 1]]], dtype=float
            )
            K_loc += gp["Gw"].T @ N_mat @ gp["Gw"] * detw
            # Elastic drilling stabilization.
            K_loc += B_d.T @ (drilling_stiffness * B_d) * detw

        for sh in cache["shear"]:
            B_s = sh["B_s"]
            K_s = B_s.T @ D_shear @ B_s * sh["detw"]
            F_loc += K_s @ u_loc
            if tangent:
                K_loc += K_s

        if not tangent:
            return T0.T @ F_loc, None, trial_state
        return T0.T @ F_loc, T0.T @ K_loc @ T0, trial_state

    def compute_stresses(
        self, mesh: "FEMesh", displacements: np.ndarray, material: "Material"
    ) -> Dict[str, np.ndarray]:
        coords = self.get_node_coordinates(mesh)
        u_elem_global = self._get_element_displacements(mesh, displacements)
        E = material.elastic_modulus
        nu = material.poisson_ratio
        G = material.shear_modulus
        h = self.thickness
        kappa = 5.0 / 6.0

        shell_plane = np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]],
            dtype=float,
        )
        D_stress = E / (1.0 - nu**2) * shell_plane
        D_bending_moment = E * h**3 / (12.0 * (1.0 - nu**2)) * shell_plane

        num_ip = len(self.gauss_points)
        stresses: Dict[str, np.ndarray] = {
            "membrane_xx": np.zeros(num_ip),
            "membrane_yy": np.zeros(num_ip),
            "membrane_xy": np.zeros(num_ip),
            "bending_xx": np.zeros(num_ip),
            "bending_yy": np.zeros(num_ip),
            "bending_xy": np.zeros(num_ip),
            "shear_xz": np.zeros(num_ip),
            "shear_yz": np.zeros(num_ip),
            "von_mises": np.zeros(num_ip),
        }
        mitc_planar = None
        mitc_samples = None
        mitc_u_local = None
        if self._is_4node:
            R_center = self._center_frame(coords)
            mitc_planar, mitc_samples = self._mitc4_shear_samples(coords, R_center)
            mitc_u_local = self._local_dof_transform(R_center) @ u_elem_global

        for idx, (xi, eta) in enumerate(self.gauss_points):
            N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
            R, dN_dx, dN_dy, _ = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
            T = self._local_dof_transform(R)
            u_local = T @ u_elem_global
            B_m, B_b, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
            membrane_strain = B_m @ u_local
            curvature = B_b @ u_local
            if self._is_4node:
                B_s_mitc, _ = self._mitc4_shear_b_matrix(mitc_planar, mitc_samples, float(xi), float(eta))
                shear_strain = B_s_mitc @ mitc_u_local
            else:
                shear_strain = B_s @ u_local

            sigma_m = D_stress @ membrane_strain
            moments = D_bending_moment @ curvature
            sigma_b = 6.0 * moments / max(h**2, _SMALL)
            tau_s = G * kappa * shear_strain

            stresses["membrane_xx"][idx] = sigma_m[0]
            stresses["membrane_yy"][idx] = sigma_m[1]
            stresses["membrane_xy"][idx] = sigma_m[2]
            stresses["bending_xx"][idx] = sigma_b[0]
            stresses["bending_yy"][idx] = sigma_b[1]
            stresses["bending_xy"][idx] = sigma_b[2]
            stresses["shear_xz"][idx] = tau_s[0]
            stresses["shear_yz"][idx] = tau_s[1]

            sigma_x = sigma_m[0] + sigma_b[0]
            sigma_y = sigma_m[1] + sigma_b[1]
            tau_xy = sigma_m[2] + sigma_b[2]
            stresses["von_mises"][idx] = np.sqrt(
                sigma_x**2 + sigma_y**2 - sigma_x * sigma_y + 3.0 * (tau_xy**2 + tau_s[0] ** 2 + tau_s[1] ** 2)
            )
        return stresses


class BeamElement(Element):
    """2-node Timoshenko beam element with 6 DOF per node."""

    def __init__(
        self,
        element_id: int,
        node_ids: List[int],
        material_name: str = "default",
        cross_section: Optional[Dict[str, float]] = None,
    ):
        super().__init__(element_id, node_ids, material_name)
        if len(node_ids) != 2:
            raise ValueError(f"BeamElement requires 2 nodes, got {len(node_ids)}")
        self.cross_section = cross_section or {}
        self._A = self.cross_section.get("area", 0.01)
        self._Iy = self.cross_section.get("Iy", 1.0e-8)
        self._Iz = self.cross_section.get("Iz", 1.0e-8)
        self._J = self.cross_section.get("J", 1.0e-8)
        self._ky = self.cross_section.get("shear_factor_y", 5.0 / 6.0)
        self._kz = self.cross_section.get("shear_factor_z", 5.0 / 6.0)
        self._orientation = _section_orientation(self.cross_section)
        # Optional exact stress-recovery data; estimated from A and I if absent.
        self._c_y = self.cross_section.get("c_y")
        self._c_z = self.cross_section.get("c_z")
        self._torsion_modulus = self.cross_section.get("torsion_modulus")

    def _fiber_distances(self) -> Tuple[float, float]:
        """Extreme fiber distances (local y, local z), estimated when not given."""
        c_y = self._c_y
        c_z = self._c_z
        if c_y is None or c_y <= 0.0:
            c_y = np.sqrt(abs(self._Iz) / max(self._A, _SMALL)) * 2.0
        if c_z is None or c_z <= 0.0:
            c_z = np.sqrt(abs(self._Iy) / max(self._A, _SMALL)) * 2.0
        return max(float(c_y), _SMALL), max(float(c_z), _SMALL)

    def _torsion_section_modulus(self) -> float:
        """Torsional section modulus Wt with tau = T / Wt, estimated when not given."""
        wt = self._torsion_modulus
        if wt is None or wt <= 0.0:
            wt = 2.0 * self._A
        return max(float(wt), _SMALL)

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        coords = np.zeros((2, 3), dtype=float)
        for i, node_id in enumerate(self.node_ids):
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"Beam element {self.element_id} references missing node {node_id}")
            coords[i] = node.coords()
        return coords

    def _beam_frame_and_transform(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        length = float(np.linalg.norm(coords[1] - coords[0]))
        if length < _SMALL:
            raise ValueError(f"Beam element {self.element_id} has near-zero length")
        e1 = (coords[1] - coords[0]) / length
        R = _beam_rotation_matrix(e1, self._orientation)
        T = np.zeros((12, 12), dtype=float)
        Rt = R.T
        for i in range(2):
            b = i * 6
            T[b:b + 3, b:b + 3] = Rt
            T[b + 3:b + 6, b + 3:b + 6] = Rt
        return length, T

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((self.total_dofs, self.total_dofs))
        E = material.elastic_modulus
        G = material.shear_modulus
        EA = E * self._A
        EIy = E * self._Iy
        EIz = E * self._Iz
        GJ = G * self._J
        K = np.zeros((12, 12), dtype=float)
        K[0, 0] = K[6, 6] = EA / L
        K[0, 6] = K[6, 0] = -EA / L

        # Bending about local y: deflection w (local z), rotation ry, EIy,
        # Timoshenko shear parameter from the local-z shear area.
        phi_w = 12.0 * EIy / max(G * self._A * self._kz * L**2, _SMALL)
        K[2, 2] = K[8, 8] = 12.0 * EIy / (L**3 * (1.0 + phi_w))
        K[2, 8] = K[8, 2] = -K[2, 2]
        K[2, 4] = K[4, 2] = -6.0 * EIy / (L**2 * (1.0 + phi_w))
        K[2, 10] = K[10, 2] = K[2, 4]
        K[8, 4] = K[4, 8] = -K[2, 4]
        K[8, 10] = K[10, 8] = -K[2, 4]
        K[4, 4] = K[10, 10] = (4.0 + phi_w) * EIy / (L * (1.0 + phi_w))
        K[4, 10] = K[10, 4] = (2.0 - phi_w) * EIy / (L * (1.0 + phi_w))

        # Bending about local z: deflection v (local y), rotation rz, EIz,
        # Timoshenko shear parameter from the local-y shear area.
        phi_v = 12.0 * EIz / max(G * self._A * self._ky * L**2, _SMALL)
        K[1, 1] = K[7, 7] = 12.0 * EIz / (L**3 * (1.0 + phi_v))
        K[1, 7] = K[7, 1] = -K[1, 1]
        K[1, 5] = K[5, 1] = 6.0 * EIz / (L**2 * (1.0 + phi_v))
        K[1, 11] = K[11, 1] = 6.0 * EIz / (L**2 * (1.0 + phi_v))
        K[7, 5] = K[5, 7] = -6.0 * EIz / (L**2 * (1.0 + phi_v))
        K[7, 11] = K[11, 7] = -6.0 * EIz / (L**2 * (1.0 + phi_v))
        K[5, 5] = K[11, 11] = (4.0 + phi_v) * EIz / (L * (1.0 + phi_v))
        K[5, 11] = K[11, 5] = (2.0 - phi_v) * EIz / (L * (1.0 + phi_v))

        K[3, 3] = K[9, 9] = GJ / L
        K[3, 9] = K[9, 3] = -GJ / L
        K_global = T.T @ K @ T
        self._stiffness_matrix = K_global
        return K_global

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((self.total_dofs, self.total_dofs))
        rho = material.density
        M = np.zeros((12, 12), dtype=float)
        mass_per_node = rho * self._A * L / 2.0
        # Lumped rotary inertia: bending rotations carry the rigid-bar term
        # rho*A*L^3/24 plus the section rotatory term rho*I*L/2; the torsion
        # DOF carries the polar section inertia rho*(Iy+Iz)*L/2, not a
        # bar-length term, because spinning about the member axis does not
        # translate the distributed mass.
        rotary_bar = mass_per_node * L**2 / 12.0
        rotary = (
            rho * (self._Iy + self._Iz) * L / 2.0,  # rx (torsion)
            rotary_bar + rho * self._Iy * L / 2.0,  # ry
            rotary_bar + rho * self._Iz * L / 2.0,  # rz
        )
        for i in range(2):
            b = i * 6
            for d in range(3):
                M[b + d, b + d] = mass_per_node
                M[b + 3 + d, b + 3 + d] = rotary[d]
        M_global = T.T @ M @ T
        self._mass_matrix = M_global
        return M_global

    @staticmethod
    def _axial_compression_from_state(state: Optional[Any]) -> float:
        """Return positive compression from a user/reference element state."""
        if state is None:
            return 0.0
        if isinstance(state, (int, float, np.number)):
            return float(state)
        if not isinstance(state, dict):
            return 0.0

        for key in ("axial_compression", "compression", "N_compression"):
            if key in state:
                return float(state[key])

        for key in ("axial_force", "N", "reference_axial_force"):
            if key in state:
                # Stress recovery uses positive axial strain/stress in tension,
                # so a negative axial force is destabilizing compression.
                return -float(state[key])
        return 0.0

    def compute_geometric_stiffness_matrix(
        self,
        mesh: "FEMesh",
        material: "Material",
        state: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Beam-column geometric stiffness for an axial reference compression.

        The returned matrix follows ``K phi = lambda KG phi``. A positive
        ``axial_compression`` in ``state`` therefore produces a positive
        destabilizing ``KG``.
        """
        axial_compression = self._axial_compression_from_state(state)
        if axial_compression == 0.0:
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        factor = axial_compression / (30.0 * L)
        L2 = L * L
        g_standard = factor * np.array(
            [
                [36.0, 3.0 * L, -36.0, 3.0 * L],
                [3.0 * L, 4.0 * L2, -3.0 * L, -L2],
                [-36.0, -3.0 * L, 36.0, -3.0 * L],
                [3.0 * L, -L2, -3.0 * L, 4.0 * L2],
            ],
            dtype=float,
        )

        K_geo = np.zeros((12, 12), dtype=float)
        v_rz = [1, 5, 7, 11]
        w_ry = [2, 4, 8, 10]
        sign_w = np.diag([1.0, -1.0, 1.0, -1.0])
        g_w = sign_w @ g_standard @ sign_w

        for i, row in enumerate(v_rz):
            for j, col in enumerate(v_rz):
                K_geo[row, col] += g_standard[i, j]
        for i, row in enumerate(w_ry):
            for j, col in enumerate(w_ry):
                K_geo[row, col] += g_w[i, j]

        return T.T @ K_geo @ T

    def compute_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any] = None,
        num_layers: int = 5,
        tangent: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        """Elastic beam-column response with von Karman axial coupling.

        The axial strain includes the transverse-displacement rotation terms

            eps = (u2-u1)/L + ((v2-v1)^2 + (w2-w1)^2) / (2 L^2)

        which gives the P-delta string effect consistently (internal force
        and tangent from the same potential).  Bending, shear and torsion
        remain linear elastic; material plasticity for beams is not part of
        this formulation.
        """
        cache = getattr(self, "_nl_cache", None)
        if cache is None:
            coords = self.get_node_coordinates(mesh)
            L, T = self._beam_frame_and_transform(coords)
            K_global = self.compute_stiffness_matrix(mesh, material)
            K_loc = T @ K_global @ T.T
            # Remove the linear axial block; the von Karman axial response
            # replaces it entirely.
            for i in (0, 6):
                K_loc[i, :] = 0.0
                K_loc[:, i] = 0.0
            cache = {"L": L, "T": T, "K_noax": K_loc, "EA": material.elastic_modulus * self._A}
            self._nl_cache = cache

        L = cache["L"]
        T = cache["T"]
        EA = cache["EA"]
        u_loc = T @ np.asarray(u_elem, dtype=float)

        du = u_loc[6] - u_loc[0]
        dv = u_loc[7] - u_loc[1]
        dw = u_loc[8] - u_loc[2]
        eps = du / L + (dv**2 + dw**2) / (2.0 * L**2)
        N_force = EA * eps

        d_eps = np.zeros(12, dtype=float)
        d_eps[0], d_eps[6] = -1.0 / L, 1.0 / L
        d_eps[1], d_eps[7] = -dv / L**2, dv / L**2
        d_eps[2], d_eps[8] = -dw / L**2, dw / L**2

        F_loc = cache["K_noax"] @ u_loc + N_force * L * d_eps
        if not tangent:
            return T.T @ F_loc, None, state
        K_loc = cache["K_noax"] + EA * L * np.outer(d_eps, d_eps)
        string = N_force / L
        for a, b in ((1, 7), (2, 8)):
            K_loc[a, a] += string
            K_loc[b, b] += string
            K_loc[a, b] -= string
            K_loc[b, a] -= string

        return T.T @ F_loc, T.T @ K_loc @ T, state

    def _end_displacements(self, u_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Local DOF vectors of the two geometric end nodes."""
        return u_local[0:6], u_local[6:12]

    def compute_stresses(self, mesh: "FEMesh", displacements: np.ndarray, material: "Material") -> Dict[str, Any]:
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return {}
        u_local = T @ self._get_element_displacements(mesh, displacements)
        E = material.elastic_modulus
        G = material.shear_modulus
        end_a, end_b = self._end_displacements(u_local)
        u1, v1, w1, rx1, ry1, rz1 = end_a
        u2, v2, w2, rx2, ry2, rz2 = end_b
        sigma_axial = E * (u2 - u1) / L
        kappa_z = (rz2 - rz1) / L
        kappa_y = (ry2 - ry1) / L
        M_z = E * self._Iz * kappa_z
        M_y = E * self._Iy * kappa_y
        c_y, c_z = self._fiber_distances()
        sigma_bending_y = M_y * c_z / max(self._Iy, _SMALL)
        sigma_bending_z = M_z * c_y / max(self._Iz, _SMALL)
        gamma_xy = (v2 - v1) / L - 0.5 * (rz1 + rz2)
        gamma_xz = (w2 - w1) / L + 0.5 * (ry1 + ry2)
        tau_y = G * self._ky * gamma_xy
        tau_z = G * self._kz * gamma_xz
        tau_torsion = G * self._J * (rx2 - rx1) / L / self._torsion_section_modulus()
        sigma_x = sigma_axial + sigma_bending_y + sigma_bending_z
        von_mises = np.sqrt(sigma_x**2 + 3.0 * (tau_y**2 + tau_z**2 + tau_torsion**2))
        return {
            "axial_stress": sigma_axial,
            "bending_stress_y": sigma_bending_y,
            "bending_stress_z": sigma_bending_z,
            "shear_stress_y": tau_y,
            "shear_stress_z": tau_z,
            "torsional_stress": tau_torsion,
            "von_mises": von_mises,
        }


class QuadraticBeamElement(BeamElement):
    """3-node quadratic Timoshenko beam element with 6 DOF per node."""

    GAUSS_POINTS = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)], dtype=float)
    GAUSS_WEIGHTS = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)

    def __init__(
        self,
        element_id: int,
        node_ids: List[int],
        material_name: str = "default",
        cross_section: Optional[Dict[str, float]] = None,
        eccentricity: Optional[np.ndarray] = None,
    ):
        Element.__init__(self, element_id, node_ids, material_name)
        if len(node_ids) != 3:
            raise ValueError(f"QuadraticBeamElement requires 3 nodes, got {len(node_ids)}")
        self.cross_section = cross_section or {}
        self._A = self.cross_section.get("area", 0.01)
        self._Iy = self.cross_section.get("Iy", 1.0e-8)
        self._Iz = self.cross_section.get("Iz", 1.0e-8)
        self._J = self.cross_section.get("J", 1.0e-8)
        self._ky = self.cross_section.get("shear_factor_y", 5.0 / 6.0)
        self._kz = self.cross_section.get("shear_factor_z", 5.0 / 6.0)
        self._orientation = _section_orientation(self.cross_section)
        self._c_y = self.cross_section.get("c_y")
        self._c_z = self.cross_section.get("c_z")
        self._torsion_modulus = self.cross_section.get("torsion_modulus")
        self.eccentricity = np.zeros(3, dtype=float) if eccentricity is None else np.asarray(eccentricity, dtype=float)

    def _end_displacements(self, u_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Geometric end nodes are 0 and 2; node 1 is the midside node.  The
        # end-difference over the full length equals the quadratic shape
        # function derivative evaluated at the element centre.
        return u_local[0:6], u_local[12:18]

    def compute_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any] = None,
        num_layers: int = 5,
        tangent: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        # The 2-node beam-column formulation does not apply to the 3-node
        # topology; fall back to the linear elastic response.
        return Element.compute_nonlinear_response(
            self, mesh, material, u_elem, state, num_layers, tangent
        )

    @property
    def num_nodes(self) -> int:
        return 3

    @property
    def dofs_per_node(self) -> int:
        return 6

    @property
    def total_dofs(self) -> int:
        return 18

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        coords = np.zeros((3, 3), dtype=float)
        for i, node_id in enumerate(self.node_ids):
            node = mesh.get_node(node_id)
            if node is None:
                raise ValueError(f"Quadratic beam element {self.element_id} references missing node {node_id}")
            coords[i] = node.coords()
        return coords

    def compute_shape_functions(self, xi: float) -> Tuple[np.ndarray, np.ndarray]:
        N = np.array([xi * (xi - 1.0) / 2.0, 1.0 - xi**2, xi * (xi + 1.0) / 2.0], dtype=float)
        dN_dxi = np.array([xi - 0.5, -2.0 * xi, xi + 0.5], dtype=float)
        return N, dN_dxi

    def _beam_frame_and_transform(self, coords: np.ndarray) -> Tuple[float, np.ndarray]:
        length = float(np.linalg.norm(coords[2] - coords[0]))
        if length < _SMALL:
            raise ValueError(f"Quadratic beam element {self.element_id} has near-zero length")
        e1 = (coords[2] - coords[0]) / length
        R = _beam_rotation_matrix(e1, self._orientation)
        T = np.zeros((18, 18), dtype=float)
        Rt = R.T
        for i in range(3):
            b = i * 6
            T[b:b + 3, b:b + 3] = Rt
            T[b + 3:b + 6, b + 3:b + 6] = Rt
        return length, T

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((18, 18), dtype=float)
        E = material.elastic_modulus
        G = material.shear_modulus
        EA = E * self._A
        EIy = E * self._Iy
        EIz = E * self._Iz
        GJ = G * self._J
        GA_y = G * self._A * self._ky
        GA_z = G * self._A * self._kz
        K = np.zeros((18, 18), dtype=float)
        for xi, weight in zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS):
            N, dN_dxi = self.compute_shape_functions(float(xi))
            dN_dx = dN_dxi * 2.0 / L
            B_axial = np.zeros((1, 18), dtype=float)
            B_torsion = np.zeros((1, 18), dtype=float)
            B_shear_xz = np.zeros((1, 18), dtype=float)
            B_shear_xy = np.zeros((1, 18), dtype=float)
            B_bend_y = np.zeros((1, 18), dtype=float)
            B_bend_z = np.zeros((1, 18), dtype=float)
            for i in range(3):
                b = i * 6
                B_axial[0, b + 0] = dN_dx[i]
                B_torsion[0, b + 3] = dN_dx[i]
                B_shear_xz[0, b + 2] = dN_dx[i]
                B_shear_xz[0, b + 4] = N[i]
                B_shear_xy[0, b + 1] = dN_dx[i]
                B_shear_xy[0, b + 5] = -N[i]
                B_bend_y[0, b + 4] = dN_dx[i]
                B_bend_z[0, b + 5] = dN_dx[i]
            jac = L / 2.0 * weight
            K += B_axial.T @ (EA * np.eye(1)) @ B_axial * jac
            K += B_torsion.T @ (GJ * np.eye(1)) @ B_torsion * jac
            K += B_shear_xz.T @ (GA_z * np.eye(1)) @ B_shear_xz * jac
            K += B_shear_xy.T @ (GA_y * np.eye(1)) @ B_shear_xy * jac
            K += B_bend_y.T @ (EIy * np.eye(1)) @ B_bend_y * jac
            K += B_bend_z.T @ (EIz * np.eye(1)) @ B_bend_z * jac
        K_global = T.T @ K @ T
        self._stiffness_matrix = K_global
        return K_global

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((18, 18), dtype=float)
        M = np.zeros((18, 18), dtype=float)
        rho = material.density
        # Consistent rotary inertia per rotation axis: polar (Iy+Iz) for
        # torsion, the matching section inertia for each bending rotation.
        rotary_inertia = (rho * (self._Iy + self._Iz), rho * self._Iy, rho * self._Iz)
        for xi, weight in zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS):
            N, _ = self.compute_shape_functions(float(xi))
            for i in range(3):
                for j in range(3):
                    translational = N[i] * N[j] * rho * self._A * weight * L / 2.0
                    for d in range(3):
                        M[i * 6 + d, j * 6 + d] += translational
                        M[i * 6 + 3 + d, j * 6 + 3 + d] += N[i] * N[j] * rotary_inertia[d] * weight * L / 2.0
        M_global = T.T @ M @ T
        self._mass_matrix = M_global
        return M_global

    def compute_geometric_stiffness_matrix(
        self,
        mesh: "FEMesh",
        material: "Material",
        state: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Beam-column stress stiffness from the lateral displacement gradient:

            KG = N_compression * integral (dN/dx)^T (dN/dx) dx

        applied to both transverse deflections.  This is the same
        destabilizing-gradient theory as the shell membrane KG and follows the
        package convention ``K phi = lambda KG phi`` with compression positive.
        The higher-order rotation-gradient term is omitted, which makes the
        predicted critical loads slightly conservative on coarse meshes.
        """
        axial_compression = self._axial_compression_from_state(state)
        if axial_compression == 0.0:
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((self.total_dofs, self.total_dofs), dtype=float)

        KG = np.zeros((18, 18), dtype=float)
        for xi, weight in zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS):
            _, dN_dxi = self.compute_shape_functions(float(xi))
            dN_dx = dN_dxi * 2.0 / L
            G_matrix = np.zeros((2, 18), dtype=float)
            for i in range(3):
                b = i * 6
                G_matrix[0, b + 1] = dN_dx[i]
                G_matrix[1, b + 2] = dN_dx[i]
            KG += axial_compression * (G_matrix.T @ G_matrix) * (L / 2.0 * weight)
        return T.T @ KG @ T


class CoupledBeamShellElement(Element):
    """Kinematic MPC coupling between one eccentric beam node and one shell node."""

    def __init__(
        self,
        element_id: int,
        beam_node_id: int,
        shell_node_id: int,
        material_name: str = "default",
        coupling_stiffness: float = 0.0,
        eccentricity: Optional[np.ndarray] = None,
    ):
        super().__init__(element_id, [beam_node_id, shell_node_id], material_name)
        self.beam_node_id = beam_node_id
        self.shell_node_id = shell_node_id
        self.coupling_stiffness = coupling_stiffness
        self.eccentricity = None if eccentricity is None else np.asarray(eccentricity, dtype=float)

    @property
    def num_nodes(self) -> int:
        return 2

    @property
    def dofs_per_node(self) -> int:
        return 6

    def get_node_coordinates(self, mesh: "FEMesh") -> np.ndarray:
        beam_node = mesh.get_node(self.beam_node_id)
        shell_node = mesh.get_node(self.shell_node_id)
        if beam_node is None or shell_node is None:
            raise ValueError(f"Coupling element {self.element_id} references a missing node")
        return np.array([beam_node.coords(), shell_node.coords()], dtype=float)

    def _eccentricity_vector(self, mesh: "FEMesh") -> np.ndarray:
        if self.eccentricity is not None:
            return np.asarray(self.eccentricity, dtype=float)
        coords = self.get_node_coordinates(mesh)
        return coords[0] - coords[1]

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        # Coupling is enforced exactly by assembly.build_constraint_transformation().
        # Returning zero avoids penalty-stiffness pollution.
        K = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        self._stiffness_matrix = K
        return K

    def get_mpc_constraints(self, mesh: "FEMesh") -> List[Dict[str, Any]]:
        """
        Return slave/master constraints for the eccentric beam-shell relation.

        Beam node DOFs are slaves. Shell node DOFs are masters:
            u_b = u_s + theta_s x r
            theta_b = theta_s
        where r points from shell node to beam node.
        """
        beam_node = mesh.get_node(self.beam_node_id)
        shell_node = mesh.get_node(self.shell_node_id)
        if beam_node is None or shell_node is None:
            return []

        b = beam_node.dofs
        s = shell_node.dofs
        rx, ry, rz = self._eccentricity_vector(mesh)

        return [
            {
                "slave": b[0],
                "masters": {s[0]: 1.0, s[4]: rz, s[5]: -ry},
                "value": 0.0,
                "label": f"beam_shell_ecc_u_x_{self.element_id}",
            },
            {
                "slave": b[1],
                "masters": {s[1]: 1.0, s[3]: -rz, s[5]: rx},
                "value": 0.0,
                "label": f"beam_shell_ecc_u_y_{self.element_id}",
            },
            {
                "slave": b[2],
                "masters": {s[2]: 1.0, s[3]: ry, s[4]: -rx},
                "value": 0.0,
                "label": f"beam_shell_ecc_u_z_{self.element_id}",
            },
            {"slave": b[3], "masters": {s[3]: 1.0}, "value": 0.0, "label": f"beam_shell_ecc_rx_{self.element_id}"},
            {"slave": b[4], "masters": {s[4]: 1.0}, "value": 0.0, "label": f"beam_shell_ecc_ry_{self.element_id}"},
            {"slave": b[5], "masters": {s[5]: 1.0}, "value": 0.0, "label": f"beam_shell_ecc_rz_{self.element_id}"},
        ]

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        return np.zeros((self.total_dofs, self.total_dofs), dtype=float)


ELEMENT_TYPES = {
    "shell": ShellElement,
    "beam": BeamElement,
    "quadratic_beam": QuadraticBeamElement,
    "coupled": CoupledBeamShellElement,
}


def create_element(
    element_type: str,
    element_id: int,
    node_ids: List[int],
    material_name: str = "default",
    **kwargs: Any,
) -> Element:
    if element_type not in ELEMENT_TYPES:
        raise ValueError(f"Unknown element type: {element_type}")
    if element_type == "coupled":
        if len(node_ids) != 2:
            raise ValueError("CoupledBeamShellElement factory requires [beam_node_id, shell_node_id]")
        return CoupledBeamShellElement(element_id, node_ids[0], node_ids[1], material_name, **kwargs)
    return ELEMENT_TYPES[element_type](element_id, node_ids, material_name, **kwargs)
