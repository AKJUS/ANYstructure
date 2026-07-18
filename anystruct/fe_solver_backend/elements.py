"""
Finite Element Implementations

This module contains element formulations for:
- ShellElement: 3/6-node triangular and 4/8-node quadrilateral Mindlin-Reissner shell element
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
    * 3-node: centroidal edge-compatible assumed shear.  This is the constant
      shear part of the DSG/MITC3 family: transverse shear is evaluated from
      the element-average shear gap in the centroid frame, preserving rigid
      body motion and constant-shear patches without the locking-prone fully
      integrated linear Mindlin shear term.
    * 6-node: quadratic displacement interpolation with reduced three-point
      triangular shear integration.
    * 8-node: reduced 2x2 shear integration (S8R style).  When full element
      reduced integration is requested, a small nullspace-projection
      hourglass stiffness stabilizes modes outside the six rigid-body modes.

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

from .jit_compiler import njit
from .material_curves import FiberSectionPlasticityConfig
from .plasticity import lobatto_layers, plane_stress_elastic_matrix, plane_stress_return_map

if TYPE_CHECKING:
    from .fe_core import FEMesh, Material


_SMALL = 1.0e-12


@njit
def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors without np.cross dispatch overhead."""
    return np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ],
    )


@njit
def _inv2(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """Inverse and determinant of a 2x2 matrix without LAPACK overhead."""
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(det) < _SMALL:
        raise np.linalg.LinAlgError("singular 2x2 matrix")
    inv = np.array(
        [[matrix[1, 1], -matrix[0, 1]], [-matrix[1, 0], matrix[0, 0]]],
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


def _rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return the axis-angle rotation vector for a proper 3x3 rotation matrix."""
    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace_value = float(np.trace(R))
    cos_angle = max(min((trace_value - 1.0) / 2.0, 1.0), -1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1.0e-12:
        return 0.5 * np.array(
            [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
            dtype=float,
        )
    if abs(np.pi - angle) < 1.0e-6:
        axis = np.sqrt(np.maximum(np.diag(R) + 1.0, 0.0) / 2.0)
        if R[2, 1] - R[1, 2] < 0.0:
            axis[0] *= -1.0
        if R[0, 2] - R[2, 0] < 0.0:
            axis[1] *= -1.0
        if R[1, 0] - R[0, 1] < 0.0:
            axis[2] *= -1.0
        norm = float(np.linalg.norm(axis))
        if norm <= _SMALL:
            return np.zeros(3, dtype=float)
        return angle * axis / norm
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    ) / (2.0 * np.sin(angle))
    return angle * axis


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
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        material: "Material",
        return_global: bool = False,
    ) -> Dict[str, Any]:
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


@njit
def _jit_compute_4node_shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = np.array(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ],
    )
    dN_dxi = np.array(
        [
            -0.25 * (1.0 - eta),
            0.25 * (1.0 - eta),
            0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta),
        ],
    )
    dN_deta = np.array(
        [
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
            0.25 * (1.0 + xi),
            0.25 * (1.0 - xi),
        ],
    )
    return N, dN_dxi, dN_deta


@njit
def _jit_compute_8node_shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


@njit
def _jit_compute_3node_shape_functions(r: float, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = np.array([1.0 - r - s, r, s])
    dN_dr = np.array([-1.0, 1.0, 0.0])
    dN_ds = np.array([-1.0, 0.0, 1.0])
    return N, dN_dr, dN_ds


@njit
def _jit_compute_6node_shape_functions(r: float, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    l1 = 1.0 - r - s
    l2 = r
    l3 = s

    N = np.zeros(6, dtype=float)
    N[0] = l1 * (2.0 * l1 - 1.0)
    N[1] = l2 * (2.0 * l2 - 1.0)
    N[2] = l3 * (2.0 * l3 - 1.0)
    N[3] = 4.0 * l1 * l2
    N[4] = 4.0 * l2 * l3
    N[5] = 4.0 * l3 * l1

    dN_dr = np.zeros(6, dtype=float)
    dN_dr[0] = 1.0 - 4.0 * l1
    dN_dr[1] = 4.0 * l2 - 1.0
    dN_dr[2] = 0.0
    dN_dr[3] = 4.0 * (l1 - l2)
    dN_dr[4] = 4.0 * l3
    dN_dr[5] = -4.0 * l3

    dN_ds = np.zeros(6, dtype=float)
    dN_ds[0] = 1.0 - 4.0 * l1
    dN_ds[1] = 0.0
    dN_ds[2] = 4.0 * l3 - 1.0
    dN_ds[3] = -4.0 * l2
    dN_ds[4] = 4.0 * l2
    dN_ds[5] = 4.0 * (l1 - l3)
    return N, dN_dr, dN_ds


@njit
def _jit_integrate_nonlinear_response(
    u_loc: np.ndarray,
    N_res: np.ndarray,
    M_res: np.ndarray,
    C0: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    B_m_all: np.ndarray,
    B_b_all: np.ndarray,
    B_d_all: np.ndarray,
    Gw_all: np.ndarray,
    detw_all: np.ndarray,
    B_s_all: np.ndarray,
    detw_shear_all: np.ndarray,
    D_shear: np.ndarray,
    drilling_stiffness: float,
    tangent: bool,
    has_plasticity: bool,
    n_dof: int,
) -> Tuple[np.ndarray, np.ndarray]:
    F_loc = np.zeros(n_dof)
    K_loc = np.zeros((n_dof, n_dof))
    
    n_gp = len(detw_all)
    for g in range(n_gp):
        detw = detw_all[g]
        B_m = B_m_all[g]
        B_b = B_b_all[g]
        B_d = B_d_all[g]
        Gw = Gw_all[g]
        
        # Calculate theta = Gw @ u_loc
        theta_0 = 0.0
        theta_1 = 0.0
        for i in range(n_dof):
            theta_0 += Gw[0, i] * u_loc[i]
            theta_1 += Gw[1, i] * u_loc[i]
            
        # Calculate B_eff = B_m + B_nl
        B_eff = np.zeros((3, n_dof))
        for i in range(n_dof):
            B_eff[0, i] = B_m[0, i] + theta_0 * Gw[0, i]
            B_eff[1, i] = B_m[1, i] + theta_1 * Gw[1, i]
            B_eff[2, i] = B_m[2, i] + theta_0 * Gw[1, i] + theta_1 * Gw[0, i]
            
        # B_eff.T @ N_res[g] + B_b.T @ M_res[g]
        N_g = N_res[g]
        M_g = M_res[g]
        B_eff_T_N = np.zeros(n_dof)
        B_b_T_M = np.zeros(n_dof)
        for i in range(n_dof):
            B_eff_T_N[i] = B_eff[0, i] * N_g[0] + B_eff[1, i] * N_g[1] + B_eff[2, i] * N_g[2]
            B_b_T_M[i] = B_b[0, i] * M_g[0] + B_b[1, i] * M_g[1] + B_b[2, i] * M_g[2]
            
        # B_d @ u_loc
        Bd_u = 0.0
        for i in range(n_dof):
            Bd_u += B_d[0, i] * u_loc[i]
            
        for i in range(n_dof):
            F_loc[i] += (B_eff_T_N[i] + B_b_T_M[i]) * detw
            F_loc[i] += B_d[0, i] * (drilling_stiffness * Bd_u) * detw
            
        if not tangent:
            continue
            
        # K_loc += (B_eff.T @ C0[g] @ B_eff + B_b.T @ C2[g] @ B_b) * detw
        C0_g = C0[g]
        C2_g = C2[g]
        
        C0_B_eff = np.zeros((3, n_dof))
        C2_B_b = np.zeros((3, n_dof))
        for r in range(3):
            for c in range(n_dof):
                val0 = 0.0
                val2 = 0.0
                for k in range(3):
                    val0 += C0_g[r, k] * B_eff[k, c]
                    val2 += C2_g[r, k] * B_b[k, c]
                C0_B_eff[r, c] = val0
                C2_B_b[r, c] = val2
                
        for r in range(n_dof):
            for c in range(n_dof):
                val_eff = 0.0
                val_b = 0.0
                for k in range(3):
                    val_eff += B_eff[k, r] * C0_B_eff[k, c]
                    val_b += B_b[k, r] * C2_B_b[k, c]
                K_loc[r, c] += (val_eff + val_b) * detw
                
        if has_plasticity:
            # coupling = B_eff.T @ C1[g] @ B_b
            C1_g = C1[g]
            C1_B_b = np.zeros((3, n_dof))
            for r in range(3):
                for c in range(n_dof):
                    val1 = 0.0
                    for k in range(3):
                        val1 += C1_g[r, k] * B_b[k, c]
                    C1_B_b[r, c] = val1
            
            for r in range(n_dof):
                for c in range(n_dof):
                    val_c = 0.0
                    for k in range(3):
                        val_c += B_eff[k, r] * C1_B_b[k, c]
                    K_loc[r, c] += (val_c + val_c) * detw
                    
        # Geometric initial stress stiffness
        N00 = N_g[0]
        N11 = N_g[1]
        N01 = N_g[2]
        
        N_Gw_0 = N00 * Gw[0] + N01 * Gw[1]
        N_Gw_1 = N01 * Gw[0] + N11 * Gw[1]
        for r in range(n_dof):
            for c in range(n_dof):
                K_loc[r, c] += (Gw[0, r] * N_Gw_0[c] + Gw[1, r] * N_Gw_1[c]) * detw
                K_loc[r, c] += B_d[0, r] * (drilling_stiffness * B_d[0, c]) * detw

    # Shear stiffness and force contribution
    for g in range(len(detw_shear_all)):
        detw_s = detw_shear_all[g]
        B_s = B_s_all[g]
        
        # F_loc += K_s @ u_loc
        Bs_u_0 = 0.0
        Bs_u_1 = 0.0
        for i in range(n_dof):
            Bs_u_0 += B_s[0, i] * u_loc[i]
            Bs_u_1 += B_s[1, i] * u_loc[i]
            
        f_s_0 = D_shear[0, 0] * Bs_u_0 + D_shear[0, 1] * Bs_u_1
        f_s_1 = D_shear[1, 0] * Bs_u_0 + D_shear[1, 1] * Bs_u_1
        
        for i in range(n_dof):
            F_loc[i] += (B_s[0, i] * f_s_0 + B_s[1, i] * f_s_1) * detw_s
            
        if not tangent:
            continue
            
        # K_loc += K_s
        C_s = np.zeros((2, n_dof))
        for r in range(2):
            for c in range(n_dof):
                C_s[r, c] = D_shear[r, 0] * B_s[0, c] + D_shear[r, 1] * B_s[1, c]
                
        for r in range(n_dof):
            for c in range(n_dof):
                K_loc[r, c] += (B_s[0, r] * C_s[0, c] + B_s[1, r] * C_s[1, c]) * detw_s
                
    return F_loc, K_loc


class ShellElement(Element):
    """3/6-node triangular and 4/8-node quadrilateral Mindlin-Reissner shell element."""

    TRI_GAUSS_POINTS_1 = np.array([[1.0 / 3.0, 1.0 / 3.0]], dtype=float)
    TRI_GAUSS_WEIGHTS_1 = np.array([0.5], dtype=float)

    TRI_GAUSS_POINTS_3 = np.array(
        [[1.0 / 6.0, 1.0 / 6.0], [2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 2.0 / 3.0]],
        dtype=float,
    )
    TRI_GAUSS_WEIGHTS_3 = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=float)

    _DUNAVANT_A1 = 0.059715871789770
    _DUNAVANT_B1 = 0.470142064105115
    _DUNAVANT_A2 = 0.797426985353087
    _DUNAVANT_B2 = 0.101286507323456
    TRI_GAUSS_POINTS_7 = np.array(
        [
            [1.0 / 3.0, 1.0 / 3.0],
            [_DUNAVANT_B1, _DUNAVANT_B1],
            [_DUNAVANT_A1, _DUNAVANT_B1],
            [_DUNAVANT_B1, _DUNAVANT_A1],
            [_DUNAVANT_B2, _DUNAVANT_B2],
            [_DUNAVANT_A2, _DUNAVANT_B2],
            [_DUNAVANT_B2, _DUNAVANT_A2],
        ],
        dtype=float,
    )
    TRI_GAUSS_WEIGHTS_7 = np.array(
        [
            0.1125,
            0.066197076394253,
            0.066197076394253,
            0.066197076394253,
            0.062969590272414,
            0.062969590272414,
            0.062969590272414,
        ],
        dtype=float,
    )

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
        drilling_stabilization: float = 1.0e-3,
        reduced_integration: bool = False,
        hourglass_stabilization: float = 1.0e-8,
    ):
        super().__init__(element_id, node_ids, material_name)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError(f"Shell element {element_id} has repeated node ids")
        self.thickness = float(thickness)
        self.drilling_stabilization = float(drilling_stabilization)
        self.reduced_integration = reduced_integration
        self.hourglass_stabilization = float(hourglass_stabilization)
        self._is_3node = len(node_ids) == 3
        self._is_6node = len(node_ids) == 6
        self._is_8node = len(node_ids) == 8
        self._is_4node = len(node_ids) == 4
        self._is_triangular = self._is_3node or self._is_6node
        self._is_quadrilateral = self._is_4node or self._is_8node
        if not (self._is_triangular or self._is_quadrilateral):
            raise ValueError(f"ShellElement requires 3, 4, 6 or 8 nodes, got {len(node_ids)}")

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def dofs_per_node(self) -> int:
        return 6

    @property
    def gauss_points(self) -> np.ndarray:
        if self._is_3node:
            return self.TRI_GAUSS_POINTS_3
        if self._is_6node:
            return self.TRI_GAUSS_POINTS_7
        if self._is_8node and self.reduced_integration:
            return self.GAUSS_POINTS_2x2
        return self.GAUSS_POINTS_3x3 if self._is_8node else self.GAUSS_POINTS_2x2

    @property
    def gauss_weights(self) -> np.ndarray:
        if self._is_3node:
            return self.TRI_GAUSS_WEIGHTS_3
        if self._is_6node:
            return self.TRI_GAUSS_WEIGHTS_7
        if self._is_8node and self.reduced_integration:
            return self.GAUSS_WEIGHTS_2x2
        return self.GAUSS_WEIGHTS_3x3 if self._is_8node else self.GAUSS_WEIGHTS_2x2

    @property
    def shear_gauss_points(self) -> np.ndarray:
        if self._is_3node:
            return self.TRI_GAUSS_POINTS_1
        if self._is_6node:
            return self.TRI_GAUSS_POINTS_3
        return self.GAUSS_POINTS_2x2 if self._is_8node else self.GAUSS_POINTS_1x1

    @property
    def shear_gauss_weights(self) -> np.ndarray:
        if self._is_3node:
            return self.TRI_GAUSS_WEIGHTS_1
        if self._is_6node:
            return self.TRI_GAUSS_WEIGHTS_3
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
        if self._is_3node:
            return self._compute_3node_shape_functions(xi, eta)
        if self._is_6node:
            return self._compute_6node_shape_functions(xi, eta)
        if self._is_4node:
            return self._compute_4node_shape_functions(xi, eta)
        return self._compute_8node_shape_functions(xi, eta)

    def _compute_3node_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _jit_compute_3node_shape_functions(xi, eta)

    def _compute_4node_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _jit_compute_4node_shape_functions(xi, eta)

    def _compute_6node_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _jit_compute_6node_shape_functions(xi, eta)

    def _compute_8node_shape_functions(self, xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _jit_compute_8node_shape_functions(xi, eta)

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
        if coords.shape[0] >= 3:
            candidate_edges.append(coords[2] - coords[0])
            candidate_edges.append(coords[2] - coords[1])
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

    def _reference_center(self) -> Tuple[float, float]:
        if self._is_triangular:
            return 1.0 / 3.0, 1.0 / 3.0
        return 0.0, 0.0

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
        xi, eta = self._reference_center()
        _, dN_dxi, dN_deta = self.compute_shape_functions(xi, eta)
        R, _, _, _ = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
        return R

    def _tri3_assumed_shear_b_matrix(self, coords: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Constant assumed-shear field for the 3-node triangle.

        The row space is evaluated in the centroid frame and integrated once
        over the triangular area.  This is the constant shear-gap part used by
        DSG3/MITC3-style triangles: rigid rotations satisfy
        ``dw/dx + theta_y = 0`` and ``dw/dy - theta_x = 0`` exactly, while a
        constant transverse-shear patch is reproduced without the locking-prone
        fully integrated linear Mindlin shear interpolation.
        """
        if not self._is_3node:
            raise ValueError("_tri3_assumed_shear_b_matrix is only valid for 3-node shells")
        planar = coords @ R[:, :2]
        r, s = self._reference_center()
        N, dN_dr, dN_ds = self.compute_shape_functions(r, s)
        J2 = np.array(
            [
                [float(dN_dr @ planar[:, 0]), float(dN_dr @ planar[:, 1])],
                [float(dN_ds @ planar[:, 0]), float(dN_ds @ planar[:, 1])],
            ],
            dtype=float,
        )
        try:
            inv_j2, det_j = _inv2(J2)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"Shell element {self.element_id} has a singular triangular shear Jacobian") from exc
        dN_dx = inv_j2[0, 0] * dN_dr + inv_j2[0, 1] * dN_ds
        dN_dy = inv_j2[1, 0] * dN_dr + inv_j2[1, 1] * dN_ds
        _, _, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
        return B_s, abs(float(det_j))

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

    def _rigid_body_mode_matrix(self, coords: np.ndarray) -> np.ndarray:
        modes = np.zeros((self.total_dofs, 6), dtype=float)
        centroid = np.mean(coords, axis=0)
        axes = np.eye(3, dtype=float)

        for local, coord in enumerate(coords):
            base = local * 6
            r = coord - centroid
            modes[base + 0, 0] = 1.0
            modes[base + 1, 1] = 1.0
            modes[base + 2, 2] = 1.0
            for axis_index, axis in enumerate(axes):
                displacement = _cross3(axis, r)
                modes[base : base + 3, 3 + axis_index] = displacement
                modes[base + 3 : base + 6, 3 + axis_index] = axis

        q, rmat = np.linalg.qr(modes)
        diag = np.abs(np.diag(rmat))
        if diag.size == 0:
            return np.zeros((self.total_dofs, 0), dtype=float)
        keep = diag > max(float(np.max(diag)) * 1.0e-10, _SMALL)
        return q[:, keep]

    def _hourglass_stabilization_matrix(self, K_base: np.ndarray, coords: np.ndarray) -> np.ndarray:
        """Small stiffness on non-rigid zero modes outside rigid motion."""
        coefficient = float(getattr(self, "hourglass_stabilization", 0.0))
        stabilized_topology = (self._is_8node and self.reduced_integration) or self._is_triangular
        if not stabilized_topology or coefficient <= 0.0:
            return np.zeros_like(K_base)

        K_sym = 0.5 * (K_base + K_base.T)
        eigvals, eigvecs = np.linalg.eigh(K_sym)
        max_eig = max(float(np.max(np.abs(eigvals))), 1.0)
        zero_tol = 1.0e-9 * max_eig
        zero_mask = np.abs(eigvals) < zero_tol
        if int(np.sum(zero_mask)) <= 6:
            return np.zeros_like(K_base)

        zero_space = eigvecs[:, zero_mask]
        rigid = self._rigid_body_mode_matrix(coords)
        if rigid.size:
            zero_space = zero_space - rigid @ (rigid.T @ zero_space)

        u, singular_values, _ = np.linalg.svd(zero_space, full_matrices=False)
        if singular_values.size == 0:
            return np.zeros_like(K_base)
        keep = singular_values > 1.0e-8
        if not np.any(keep):
            return np.zeros_like(K_base)

        hourglass_modes = u[:, keep]
        stiffness_scale = coefficient * max_eig
        K_hg = stiffness_scale * (hourglass_modes @ hourglass_modes.T)
        return 0.5 * (K_hg + K_hg.T)

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

            drilling_stiffness = G * h * getattr(self, "drilling_stabilization", 1.0e-3)
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
        elif self._is_3node:
            R = self._center_frame(coords)
            T = self._local_dof_transform(R)
            B_s, det_j = self._tri3_assumed_shear_b_matrix(coords, R)
            K_local = (B_s.T @ D_shear @ B_s) * det_j * float(np.sum(self.shear_gauss_weights))
            K += T.T @ K_local @ T
        else:
            for (xi, eta), weight in zip(self.shear_gauss_points, self.shear_gauss_weights):
                N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
                R, dN_dx, dN_dy, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
                T = self._local_dof_transform(R)
                _, _, B_s = self._build_shell_b_matrices(N, dN_dx, dN_dy)
                K_local = (B_s.T @ D_shear @ B_s) * det_j * weight
                K += T.T @ K_local @ T

        self._hourglass_stiffness_matrix = self._hourglass_stabilization_matrix(K, coords)
        K += self._hourglass_stiffness_matrix

        self._stiffness_matrix = K
        return K

    def compute_mass_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        rho = material.density
        h = self.thickness
        M = np.zeros((self.total_dofs, self.total_dofs), dtype=float)
        if self._is_8node and self.reduced_integration:
            area = 0.0
            for (xi, eta), weight in zip(self.gauss_points, self.gauss_weights):
                _N, dN_dxi, dN_deta = self.compute_shape_functions(float(xi), float(eta))
                _R, _dN_dx, _dN_dy, det_j = self._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
                area += float(det_j) * float(weight)
            translational = float(rho) * float(h) * max(area, 0.0) / max(self.num_nodes, 1)
            rotational = float(rho) * float(h) ** 3 / 12.0 * max(area, 0.0) / max(self.num_nodes, 1)
            for i in range(self.num_nodes):
                base = 6 * i
                M[base + 0, base + 0] = translational
                M[base + 1, base + 1] = translational
                M[base + 2, base + 2] = translational
                M[base + 3, base + 3] = rotational
                M[base + 4, base + 4] = rotational
                M[base + 5, base + 5] = rotational
            self._mass_matrix = M
            return M
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
        elif self._is_3node:
            B_s, det_j = self._tri3_assumed_shear_b_matrix(coords, R0)
            shear_data.append({"B_s": B_s, "detw": det_j * float(np.sum(self.shear_gauss_weights))})
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

        n_gp = len(gp_data)
        B_m_all = np.zeros((n_gp, 3, self.total_dofs))
        B_b_all = np.zeros((n_gp, 3, self.total_dofs))
        B_d_all = np.zeros((n_gp, 1, self.total_dofs))
        Gw_all = np.zeros((n_gp, 2, self.total_dofs))
        detw_all = np.zeros(n_gp)
        for g, gp in enumerate(gp_data):
            B_m_all[g] = gp["B_m"]
            B_b_all[g] = gp["B_b"]
            B_d_all[g] = gp["B_d"]
            Gw_all[g] = gp["Gw"]
            detw_all[g] = gp["detw"]

        n_shear = len(shear_data)
        B_s_all = np.zeros((n_shear, 2, self.total_dofs))
        detw_shear_all = np.zeros(n_shear)
        for g, sh in enumerate(shear_data):
            B_s_all[g] = sh["B_s"]
            detw_shear_all[g] = sh["detw"]

        cache = {
            "T0": T0,
            "gp": gp_data,
            "shear": shear_data,
            "B_m_all": B_m_all,
            "B_b_all": B_b_all,
            "B_d_all": B_d_all,
            "Gw_all": Gw_all,
            "detw_all": detw_all,
            "B_s_all": B_s_all,
            "detw_shear_all": detw_shear_all,
        }
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
        drilling_stiffness = G_mod * h * getattr(self, "drilling_stabilization", 1.0e-3)

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
                compute_tangent=tangent,
            )
            trial_state = {"plastic_strain": ep_new, "alpha": alpha_new, "layer_strain": layer_strain.copy()}

            sigma = sigma.reshape(n_gp, num_layers, 3)
            C_tan = C_tan.reshape(n_gp, num_layers, 3, 3)
            trial_state["layer_stress"] = sigma.reshape(n_gp * num_layers, 3).copy()

            # Through-thickness resultants and integrated tangent moduli.
            N_res = np.einsum("l,gli->gi", w_layers, sigma)
            M_res = np.einsum("l,l,gli->gi", w_layers, z_layers, sigma)
            C0 = np.einsum("l,glij->gij", w_layers, C_tan)
            C1 = np.einsum("l,l,glij->gij", w_layers, z_layers, C_tan)
            C2 = np.einsum("l,l,l,glij->gij", w_layers, z_layers, z_layers, C_tan)

        n_dof = self.total_dofs
        F_loc, K_loc_temp = _jit_integrate_nonlinear_response(
            u_loc,
            N_res,
            M_res,
            C0,
            C1,
            C2,
            cache["B_m_all"],
            cache["B_b_all"],
            cache["B_d_all"],
            cache["Gw_all"],
            cache["detw_all"],
            cache["B_s_all"],
            cache["detw_shear_all"],
            D_shear,
            drilling_stiffness,
            tangent,
            curve is not None,
            n_dof,
        )
        K_loc = K_loc_temp if tangent else None

        K_hg = getattr(self, "_hourglass_stiffness_matrix", None)
        if self._is_8node and self.reduced_integration and float(getattr(self, "hourglass_stabilization", 0.0)) > 0.0:
            if K_hg is None:
                self.compute_stiffness_matrix(mesh, material)
                K_hg = getattr(self, "_hourglass_stiffness_matrix", None)

        F_global = T0.T @ F_loc
        if K_hg is not None:
            u_global = np.asarray(u_elem, dtype=float)
            F_global = F_global + K_hg @ u_global

        if not tangent:
            return F_global, None, trial_state
        K_global = T0.T @ K_loc @ T0
        if K_hg is not None:
            K_global = K_global + K_hg
        return F_global, K_global, trial_state

    def compute_stresses(
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        material: "Material",
        return_global: bool = False,
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
        if return_global:
            stresses.update({
                "local_xx_top": np.zeros(num_ip),
                "local_yy_top": np.zeros(num_ip),
                "local_zz_top": np.zeros(num_ip),
                "local_xy_top": np.zeros(num_ip),
                "local_yz_top": np.zeros(num_ip),
                "local_xz_top": np.zeros(num_ip),
                "local_xx_bot": np.zeros(num_ip),
                "local_yy_bot": np.zeros(num_ip),
                "local_zz_bot": np.zeros(num_ip),
                "local_xy_bot": np.zeros(num_ip),
                "local_yz_bot": np.zeros(num_ip),
                "local_xz_bot": np.zeros(num_ip),
                "global_xx_top": np.zeros(num_ip),
                "global_yy_top": np.zeros(num_ip),
                "global_zz_top": np.zeros(num_ip),
                "global_xy_top": np.zeros(num_ip),
                "global_yz_top": np.zeros(num_ip),
                "global_xz_top": np.zeros(num_ip),
                "global_xx_bot": np.zeros(num_ip),
                "global_yy_bot": np.zeros(num_ip),
                "global_zz_bot": np.zeros(num_ip),
                "global_xy_bot": np.zeros(num_ip),
                "global_yz_bot": np.zeros(num_ip),
                "global_xz_bot": np.zeros(num_ip),
            })

        mitc_planar = None
        mitc_samples = None
        mitc_u_local = None
        tri3_B_s = None
        tri3_u_local = None
        if self._is_4node:
            R_center = self._center_frame(coords)
            mitc_planar, mitc_samples = self._mitc4_shear_samples(coords, R_center)
            mitc_u_local = self._local_dof_transform(R_center) @ u_elem_global
        elif self._is_3node:
            R_center = self._center_frame(coords)
            tri3_B_s, _ = self._tri3_assumed_shear_b_matrix(coords, R_center)
            tri3_u_local = self._local_dof_transform(R_center) @ u_elem_global

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
            elif self._is_3node:
                shear_strain = tri3_B_s @ tri3_u_local
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

            # Top surface (z = +h/2)
            sigma_x_top = sigma_m[0] + sigma_b[0]
            sigma_y_top = sigma_m[1] + sigma_b[1]
            tau_xy_top = sigma_m[2] + sigma_b[2]
            vm_top = np.sqrt(
                sigma_x_top**2 + sigma_y_top**2 - sigma_x_top * sigma_y_top + 3.0 * (tau_xy_top**2 + tau_s[0]**2 + tau_s[1]**2)
            )

            # Bottom surface (z = -h/2)
            sigma_x_bot = sigma_m[0] - sigma_b[0]
            sigma_y_bot = sigma_m[1] - sigma_b[1]
            tau_xy_bot = sigma_m[2] - sigma_b[2]
            vm_bot = np.sqrt(
                sigma_x_bot**2 + sigma_y_bot**2 - sigma_x_bot * sigma_y_bot + 3.0 * (tau_xy_bot**2 + tau_s[0]**2 + tau_s[1]**2)
            )

            stresses["von_mises"][idx] = max(vm_top, vm_bot)

            if return_global:
                # Top local stress tensor
                sigma_loc_top = np.array([
                    [sigma_x_top, tau_xy_top, tau_s[0]],
                    [tau_xy_top, sigma_y_top, tau_s[1]],
                    [tau_s[0], tau_s[1], 0.0]
                ], dtype=float)
                sigma_glob_top = R @ sigma_loc_top @ R.T

                # Bottom local stress tensor
                sigma_loc_bot = np.array([
                    [sigma_x_bot, tau_xy_bot, tau_s[0]],
                    [tau_xy_bot, sigma_y_bot, tau_s[1]],
                    [tau_s[0], tau_s[1], 0.0]
                ], dtype=float)
                sigma_glob_bot = R @ sigma_loc_bot @ R.T

                # Store local components
                stresses["local_xx_top"][idx] = sigma_x_top
                stresses["local_yy_top"][idx] = sigma_y_top
                stresses["local_zz_top"][idx] = 0.0
                stresses["local_xy_top"][idx] = tau_xy_top
                stresses["local_xz_top"][idx] = tau_s[0]
                stresses["local_yz_top"][idx] = tau_s[1]

                stresses["local_xx_bot"][idx] = sigma_x_bot
                stresses["local_yy_bot"][idx] = sigma_y_bot
                stresses["local_zz_bot"][idx] = 0.0
                stresses["local_xy_bot"][idx] = tau_xy_bot
                stresses["local_xz_bot"][idx] = tau_s[0]
                stresses["local_yz_bot"][idx] = tau_s[1]

                # Store global components
                stresses["global_xx_top"][idx] = sigma_glob_top[0, 0]
                stresses["global_yy_top"][idx] = sigma_glob_top[1, 1]
                stresses["global_zz_top"][idx] = sigma_glob_top[2, 2]
                stresses["global_xy_top"][idx] = sigma_glob_top[0, 1]
                stresses["global_xz_top"][idx] = sigma_glob_top[0, 2]
                stresses["global_yz_top"][idx] = sigma_glob_top[1, 2]

                stresses["global_xx_bot"][idx] = sigma_glob_bot[0, 0]
                stresses["global_yy_bot"][idx] = sigma_glob_bot[1, 1]
                stresses["global_zz_bot"][idx] = sigma_glob_bot[2, 2]
                stresses["global_xy_bot"][idx] = sigma_glob_bot[0, 1]
                stresses["global_xz_bot"][idx] = sigma_glob_bot[0, 2]
                stresses["global_yz_bot"][idx] = sigma_glob_bot[1, 2]
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
        self._fiber_plasticity = self.cross_section.get("fiber_plasticity")
        self._geometric_nonlinearity = str(
            self.cross_section.get("geometric_nonlinearity", self.cross_section.get("geometry", "von_karman"))
        ).lower()
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

    def _local_linear_stiffness(self, length: float, material: "Material", include_axial: bool = True) -> np.ndarray:
        L = float(length)
        E = material.elastic_modulus
        G = material.shear_modulus
        EA = E * self._A
        EIy = E * self._Iy
        EIz = E * self._Iz
        GJ = G * self._J
        K = np.zeros((12, 12), dtype=float)
        if include_axial:
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
        return K

    def compute_stiffness_matrix(self, mesh: "FEMesh", material: "Material") -> np.ndarray:
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return np.zeros((self.total_dofs, self.total_dofs))
        K = self._local_linear_stiffness(L, material)
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
        if bool(self.cross_section.get("consistent_mass", False)):
            # Consistent mass on the element's own linear interpolation,
            # mirroring the 3-node quadratic beam: N_i N_j blocks for the
            # translations (rho A) and for each rotation with its matching
            # section inertia (polar Iy+Iz for torsion).  Rigid-body
            # translational and rotational inertia are exact, so no lumped
            # bar-length correction terms are needed.
            coupling = np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]], dtype=float)
            rotary_density = (
                rho * (self._Iy + self._Iz),  # rx (torsion)
                rho * self._Iy,  # ry
                rho * self._Iz,  # rz
            )
            for i in range(2):
                for j in range(2):
                    factor = coupling[i, j] * L
                    for d in range(3):
                        M[i * 6 + d, j * 6 + d] = rho * self._A * factor
                        M[i * 6 + 3 + d, j * 6 + 3 + d] = rotary_density[d] * factor
            M_global = T.T @ M @ T
            self._mass_matrix = M_global
            return M_global
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

        # Wagner term: axial compression destabilizes twist about the shear
        # center, enabling torsional and flexural-torsional column modes
        # (stiffener tripping under axial stress).  The shear center is taken
        # at the centroid and warping is neglected, consistent with the
        # element's St. Venant torsion treatment, so the torsional critical
        # load is G*J*A/Ip.
        polar_ratio = (self._Iy + self._Iz) / max(self._A, 1.0e-30)
        g_torsion = axial_compression * polar_ratio / L
        K_geo[3, 3] += g_torsion
        K_geo[3, 9] -= g_torsion
        K_geo[9, 3] -= g_torsion
        K_geo[9, 9] += g_torsion

        return T.T @ K_geo @ T

    def _fiber_plasticity_config(self, material: "Material") -> Optional[FiberSectionPlasticityConfig]:
        config = self._fiber_plasticity
        if config is None:
            return None
        if isinstance(config, dict):
            config = FiberSectionPlasticityConfig(**config)
        elif config is True:
            config = FiberSectionPlasticityConfig()
        elif not isinstance(config, FiberSectionPlasticityConfig):
            raise TypeError("cross_section['fiber_plasticity'] must be a FiberSectionPlasticityConfig, dict or True")
        curve = config.material_curve or getattr(material, "hardening_curve", None)
        if curve is None:
            return None
        if config.material_curve is curve:
            return config
        return FiberSectionPlasticityConfig(config.num_y, config.num_z, curve)

    def _fiber_section_grid(self, config: FiberSectionPlasticityConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cross = self.cross_section if isinstance(getattr(self, "cross_section", None), dict) else {}
        web_height = float(cross.get("web_height") or 0.0)
        web_thickness = float(cross.get("web_thickness") or 0.0)
        flange_width = float(cross.get("flange_width") or 0.0)
        flange_thickness = float(cross.get("flange_thickness") or 0.0)
        shaped = (
            web_height > 0.0
            and web_thickness > 0.0
            and flange_width > 0.0
            and flange_thickness > 0.0
        )
        key = (
            "fiber_grid",
            int(config.num_y),
            int(config.num_z),
            float(self._A),
            float(self._Iy),
            float(self._Iz),
            round(web_height, 12),
            round(web_thickness, 12),
            round(flange_width, 12),
            round(flange_thickness, 12),
        )
        cache = getattr(self, "_fiber_grid_cache", None)
        if cache is not None and cache.get("key") == key:
            return cache["y"], cache["z"], cache["weights"]

        if shaped:
            # Profile-true fiber layout: web and flange strips follow the real
            # T/L section geometry so the plastification order (flange tips vs
            # web) is captured.  The coordinates and weights are afterwards
            # recentred and rescaled to reproduce the section constants
            # A/Iy/Iz exactly, keeping the elastic response identical to the
            # section-property idealization.  Flanges are laid out symmetric
            # in y also for L profiles (the section constants carry no product
            # of inertia).
            n_web = max(int(config.num_z), 3)
            n_flange = max(int(config.num_y), 3)
            web_z = (np.arange(n_web, dtype=float) + 0.5) * (web_height / n_web)
            y_parts = [np.zeros(n_web)]
            z_parts = [web_z]
            weight_parts = [np.full(n_web, web_height * web_thickness / n_web)]
            flange_y = (np.arange(n_flange, dtype=float) + 0.5) * (flange_width / n_flange) - 0.5 * flange_width
            for z_level in (web_height + 0.25 * flange_thickness, web_height + 0.75 * flange_thickness):
                y_parts.append(flange_y.copy())
                z_parts.append(np.full(n_flange, z_level))
                weight_parts.append(np.full(n_flange, flange_width * flange_thickness / (2 * n_flange)))
            y = np.concatenate(y_parts)
            z = np.concatenate(z_parts)
            weights = np.concatenate(weight_parts)
            total = float(np.sum(weights))
            if total > _SMALL and self._A > 0.0:
                weights *= float(self._A) / total
            area = float(np.sum(weights))
            if area > _SMALL:
                y = y - float(np.sum(weights * y)) / area
                z = z - float(np.sum(weights * z)) / area
        else:
            raw_y = np.linspace(-1.0, 1.0, int(config.num_y)) if config.num_y > 1 else np.zeros(1)
            raw_z = np.linspace(-1.0, 1.0, int(config.num_z)) if config.num_z > 1 else np.zeros(1)
            yy, zz = np.meshgrid(raw_y, raw_z, indexing="ij")
            y = yy.reshape(-1)
            z = zz.reshape(-1)
            weights = np.full(y.size, float(self._A) / max(y.size, 1), dtype=float)

        denom_y = float(np.sum(weights * y * y))
        denom_z = float(np.sum(weights * z * z))
        if denom_y > _SMALL and self._Iz > 0.0:
            y *= np.sqrt(float(self._Iz) / denom_y)
        else:
            y *= 0.0
        if denom_z > _SMALL and self._Iy > 0.0:
            z *= np.sqrt(float(self._Iy) / denom_z)
        else:
            z *= 0.0

        self._fiber_grid_cache = {"key": key, "y": y, "z": z, "weights": weights}
        return y, z, weights

    @staticmethod
    def _uniaxial_return_map(
        strain: np.ndarray,
        state: Optional[Any],
        E: float,
        curve: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        strain = np.asarray(strain, dtype=float).reshape(-1)
        n = strain.size
        if isinstance(state, dict) and np.asarray(state.get("plastic_strain", [])).size == n:
            plastic_old = np.asarray(state["plastic_strain"], dtype=float).reshape(-1)
            alpha_old = np.asarray(state.get("alpha", np.zeros(n)), dtype=float).reshape(-1)
        else:
            plastic_old = np.zeros(n, dtype=float)
            alpha_old = np.zeros(n, dtype=float)

        trial = E * (strain - plastic_old)
        abs_trial = np.abs(trial)
        flow_old = curve.flow_stress(alpha_old)
        yielding = abs_trial > flow_old + 1.0e-9 * np.maximum(flow_old, 1.0)

        stress = trial.copy()
        tangent = np.full(n, E, dtype=float)
        plastic_new = plastic_old.copy()
        alpha_new = alpha_old.copy()
        if not np.any(yielding):
            return stress, tangent, plastic_new, alpha_new

        indices = np.where(yielding)[0]
        for idx in indices:
            sign = 1.0 if trial[idx] >= 0.0 else -1.0
            dgamma = 0.0
            H = float(curve.hardening_modulus(np.array([alpha_old[idx]]))[0])
            for _ in range(30):
                alpha_trial = alpha_old[idx] + dgamma
                sy = float(curve.flow_stress(np.array([alpha_trial]))[0])
                H = float(curve.hardening_modulus(np.array([alpha_trial]))[0])
                residual = abs_trial[idx] - E * dgamma - sy
                if abs(residual) <= 1.0e-8 * max(sy, 1.0):
                    break
                dgamma = max(0.0, dgamma + residual / max(E + H, _SMALL))
            stress[idx] = sign * max(abs_trial[idx] - E * dgamma, 0.0)
            plastic_new[idx] = plastic_old[idx] + sign * dgamma
            alpha_new[idx] = alpha_old[idx] + dgamma
            tangent[idx] = E * H / max(E + H, _SMALL)
        return stress, tangent, plastic_new, alpha_new

    def _compute_fiber_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any],
        config: FiberSectionPlasticityConfig,
        tangent: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        coords = self.get_node_coordinates(mesh)
        L, T = self._beam_frame_and_transform(coords)
        u_loc = T @ np.asarray(u_elem, dtype=float)
        E = material.elastic_modulus
        G = material.shear_modulus
        y, z, weights = self._fiber_section_grid(config)

        du = u_loc[6] - u_loc[0]
        dv = u_loc[7] - u_loc[1]
        dw = u_loc[8] - u_loc[2]
        eps0 = du / L + (dv**2 + dw**2) / (2.0 * L**2)
        kappa_y = (u_loc[10] - u_loc[4]) / L
        kappa_z = (u_loc[11] - u_loc[5]) / L
        fiber_strain = eps0 + z * kappa_y + y * kappa_z
        stress, Et, plastic_new, alpha_new = self._uniaxial_return_map(
            fiber_strain, state, E, config.material_curve
        )

        B = np.zeros((fiber_strain.size, 12), dtype=float)
        B[:, 0] = -1.0 / L
        B[:, 6] = 1.0 / L
        B[:, 1] = -dv / L**2
        B[:, 7] = dv / L**2
        B[:, 2] = -dw / L**2
        B[:, 8] = dw / L**2
        B[:, 4] = -z / L
        B[:, 10] = z / L
        B[:, 5] = -y / L
        B[:, 11] = y / L

        F_loc = L * np.einsum("i,i,ij->j", weights, stress, B)
        K_loc = None
        if tangent:
            K_loc = L * np.einsum("i,i,ij,ik->jk", weights, Et, B, B)

        N_force = float(np.sum(weights * stress))
        if tangent:
            string = N_force / L
            for a, b in ((1, 7), (2, 8)):
                K_loc[a, a] += string
                K_loc[b, b] += string
                K_loc[a, b] -= string
                K_loc[b, a] -= string

        B_shear_y = np.zeros(12, dtype=float)
        B_shear_y[1], B_shear_y[7] = -1.0 / L, 1.0 / L
        B_shear_y[5], B_shear_y[11] = -0.5, -0.5
        B_shear_z = np.zeros(12, dtype=float)
        B_shear_z[2], B_shear_z[8] = -1.0 / L, 1.0 / L
        B_shear_z[4], B_shear_z[10] = 0.5, 0.5
        B_torsion = np.zeros(12, dtype=float)
        B_torsion[3], B_torsion[9] = -1.0 / L, 1.0 / L
        K_aux = L * (
            G * self._A * self._ky * np.outer(B_shear_y, B_shear_y)
            + G * self._A * self._kz * np.outer(B_shear_z, B_shear_z)
            + G * self._J * np.outer(B_torsion, B_torsion)
        )
        F_loc += K_aux @ u_loc
        if tangent:
            K_loc += K_aux

        trial_state = {
            "plastic_strain": plastic_new,
            "alpha": alpha_new,
            "fiber_strain": fiber_strain.copy(),
            "fiber_stress": stress.copy(),
            "axial_force": N_force,
        }
        if not tangent:
            return T.T @ F_loc, None, trial_state
        return T.T @ F_loc, T.T @ K_loc @ T, trial_state

    def _current_beam_frame_and_transform(self, coords: np.ndarray, u_elem: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        current = coords.copy()
        current[0] += np.asarray(u_elem[0:3], dtype=float)
        current[1] += np.asarray(u_elem[6:9], dtype=float)
        length = float(np.linalg.norm(current[1] - current[0]))
        if length < _SMALL:
            raise ValueError(f"Beam element {self.element_id} has near-zero current length")
        e1 = (current[1] - current[0]) / length
        R = _beam_rotation_matrix(e1, self._orientation)
        T = np.zeros((12, 12), dtype=float)
        Rt = R.T
        for i in range(2):
            b = i * 6
            T[b:b + 3, b:b + 3] = Rt
            T[b + 3:b + 6, b + 3:b + 6] = Rt
        return length, T, R, current

    def _compute_corotational_nonlinear_response(
        self,
        mesh: "FEMesh",
        material: "Material",
        u_elem: np.ndarray,
        state: Optional[Any],
        tangent: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Any]]:
        """Elastic 2-node corotational beam response.

        The element frame follows the current chord.  Rigid-body chord rotation
        is subtracted from nodal rotations, so a finite rigid rotation with
        matching nodal rotation vectors produces near-zero internal force.
        """
        coords = self.get_node_coordinates(mesh)
        L0, _T0 = self._beam_frame_and_transform(coords)
        Lc, Tc, Rc, _current = self._current_beam_frame_and_transform(coords, np.asarray(u_elem, dtype=float))
        R0 = _beam_rotation_matrix((coords[1] - coords[0]) / L0, self._orientation)
        rigid_rotation = _rotation_vector_from_matrix(Rc @ R0.T)

        u_global = np.asarray(u_elem, dtype=float).reshape(12)
        q = np.zeros(12, dtype=float)
        q[6] = Lc - L0
        q[3:6] = Rc.T @ (u_global[3:6] - rigid_rotation)
        q[9:12] = Rc.T @ (u_global[9:12] - rigid_rotation)

        K_loc = self._local_linear_stiffness(L0, material)
        F_loc = K_loc @ q
        F_global = Tc.T @ F_loc
        trial_state = {
            "geometric_nonlinearity": "corotational",
            "initial_length": float(L0),
            "current_length": float(Lc),
            "axial_extension": float(Lc - L0),
            "rigid_rotation_vector": rigid_rotation.copy(),
            "basic_deformation_norm": float(np.linalg.norm(q)),
        }
        if not tangent:
            return F_global, None, trial_state
        K_global = Tc.T @ K_loc @ Tc
        return F_global, K_global, trial_state

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
        remain linear elastic unless ``cross_section["fiber_plasticity"]`` is
        provided, in which case axial/bending response is integrated over a
        uniaxial fiber section using the material hardening curve.
        """
        if self._geometric_nonlinearity in {"corotational", "co_rotational", "corot"}:
            return self._compute_corotational_nonlinear_response(
                mesh, material, u_elem, state, tangent
            )

        fiber_config = self._fiber_plasticity_config(material)
        if fiber_config is not None:
            return self._compute_fiber_nonlinear_response(
                mesh, material, u_elem, state, fiber_config, tangent
            )

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

    def compute_stresses(
        self,
        mesh: "FEMesh",
        displacements: np.ndarray,
        material: "Material",
        return_global: bool = False,
    ) -> Dict[str, Any]:
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
        self._fiber_plasticity = self.cross_section.get("fiber_plasticity")
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
        """Gauss-point von Karman beam-column response on the quadratic interpolation.

        At each integration point the axial strain includes the transverse
        gradient coupling

            eps = u' + (v'^2 + w'^2) / 2

        with all gradients from the quadratic shape functions, giving internal
        force and consistent tangent (including the string/geometric term) from
        the same potential.  With ``cross_section["fiber_plasticity"]`` the
        axial/bending response is integrated over the uniaxial fiber grid per
        Gauss point using the material hardening curve; shear and torsion stay
        linear elastic, matching the 2-node beam formulation.
        """
        coords = self.get_node_coordinates(mesh)
        try:
            L, T = self._beam_frame_and_transform(coords)
        except ValueError:
            return Element.compute_nonlinear_response(
                self, mesh, material, u_elem, state, num_layers, tangent
            )
        u_loc = T @ np.asarray(u_elem, dtype=float)
        E = material.elastic_modulus
        G = material.shear_modulus
        EA = E * self._A
        EIy = E * self._Iy
        EIz = E * self._Iz
        GJ = G * self._J
        GA_y = G * self._A * self._ky
        GA_z = G * self._A * self._kz

        fiber_config = self._fiber_plasticity_config(material)
        num_gp = len(self.GAUSS_POINTS)
        if fiber_config is not None:
            y_f, z_f, w_f = self._fiber_section_grid(fiber_config)
            n_fibers = y_f.size
            fiber_strain_all = np.zeros(num_gp * n_fibers, dtype=float)
            B_fiber_all = np.zeros((num_gp * n_fibers, 18), dtype=float)

        F_loc = np.zeros(18, dtype=float)
        K_loc = np.zeros((18, 18), dtype=float) if tangent else None
        gp_data = []
        axial_forces = []
        for gp_index, (xi, weight) in enumerate(zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS)):
            N, dN_dxi = self.compute_shape_functions(float(xi))
            dN_dx = dN_dxi * 2.0 / L
            jac = L / 2.0 * float(weight)

            B_ax = np.zeros(18, dtype=float)
            B_v = np.zeros(18, dtype=float)
            B_w = np.zeros(18, dtype=float)
            B_ry = np.zeros(18, dtype=float)
            B_rz = np.zeros(18, dtype=float)
            B_torsion = np.zeros(18, dtype=float)
            B_shear_xy = np.zeros(18, dtype=float)
            B_shear_xz = np.zeros(18, dtype=float)
            for i in range(3):
                b = i * 6
                B_ax[b + 0] = dN_dx[i]
                B_v[b + 1] = dN_dx[i]
                B_w[b + 2] = dN_dx[i]
                B_torsion[b + 3] = dN_dx[i]
                B_ry[b + 4] = dN_dx[i]
                B_rz[b + 5] = dN_dx[i]
                B_shear_xy[b + 1] = dN_dx[i]
                B_shear_xy[b + 5] = -N[i]
                B_shear_xz[b + 2] = dN_dx[i]
                B_shear_xz[b + 4] = N[i]

            v_grad = float(B_v @ u_loc)
            w_grad = float(B_w @ u_loc)
            eps0 = float(B_ax @ u_loc) + 0.5 * (v_grad**2 + w_grad**2)
            kappa_y = float(B_ry @ u_loc)
            kappa_z = float(B_rz @ u_loc)
            B_membrane = B_ax + v_grad * B_v + w_grad * B_w

            # linear shear and torsion at every integration point
            gamma_xy = float(B_shear_xy @ u_loc)
            gamma_xz = float(B_shear_xz @ u_loc)
            twist = float(B_torsion @ u_loc)
            F_loc += jac * (GA_y * gamma_xy * B_shear_xy + GA_z * gamma_xz * B_shear_xz + GJ * twist * B_torsion)
            if tangent:
                K_loc += jac * (
                    GA_y * np.outer(B_shear_xy, B_shear_xy)
                    + GA_z * np.outer(B_shear_xz, B_shear_xz)
                    + GJ * np.outer(B_torsion, B_torsion)
                )

            if fiber_config is not None:
                rows = slice(gp_index * n_fibers, (gp_index + 1) * n_fibers)
                fiber_strain_all[rows] = eps0 + z_f * kappa_y + y_f * kappa_z
                B_fiber_all[rows] = (
                    B_membrane[None, :] + z_f[:, None] * B_ry[None, :] + y_f[:, None] * B_rz[None, :]
                )
                gp_data.append((jac, B_v, B_w, rows))
            else:
                axial_force = EA * eps0
                axial_forces.append(axial_force)
                F_loc += jac * (axial_force * B_membrane + EIy * kappa_y * B_ry + EIz * kappa_z * B_rz)
                if tangent:
                    K_loc += jac * (
                        EA * np.outer(B_membrane, B_membrane)
                        + axial_force * (np.outer(B_v, B_v) + np.outer(B_w, B_w))
                        + EIy * np.outer(B_ry, B_ry)
                        + EIz * np.outer(B_rz, B_rz)
                    )

        trial_state: Optional[Any] = state
        if fiber_config is not None:
            stress, Et, plastic_new, alpha_new = self._uniaxial_return_map(
                fiber_strain_all, state, E, fiber_config.material_curve
            )
            for jac, B_v, B_w, rows in gp_data:
                weights_gp = w_f
                stress_gp = stress[rows]
                F_loc += jac * np.einsum("f,f,fj->j", weights_gp, stress_gp, B_fiber_all[rows])
                axial_force = float(np.sum(weights_gp * stress_gp))
                axial_forces.append(axial_force)
                if tangent:
                    K_loc += jac * np.einsum("f,f,fj,fk->jk", weights_gp, Et[rows], B_fiber_all[rows], B_fiber_all[rows])
                    K_loc += jac * axial_force * (np.outer(B_v, B_v) + np.outer(B_w, B_w))
            trial_state = {
                "plastic_strain": plastic_new,
                "alpha": alpha_new,
                "fiber_strain": fiber_strain_all.copy(),
                "fiber_stress": stress.copy(),
                "axial_force": float(np.mean(axial_forces)) if axial_forces else 0.0,
            }

        if not tangent:
            return T.T @ F_loc, None, trial_state
        return T.T @ F_loc, T.T @ K_loc @ T, trial_state

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
        # Wagner term factor: see BeamElement.compute_geometric_stiffness_matrix.
        polar_ratio = (self._Iy + self._Iz) / max(self._A, 1.0e-30)
        for xi, weight in zip(self.GAUSS_POINTS, self.GAUSS_WEIGHTS):
            _, dN_dxi = self.compute_shape_functions(float(xi))
            dN_dx = dN_dxi * 2.0 / L
            G_matrix = np.zeros((2, 18), dtype=float)
            twist_gradient = np.zeros((1, 18), dtype=float)
            for i in range(3):
                b = i * 6
                G_matrix[0, b + 1] = dN_dx[i]
                G_matrix[1, b + 2] = dN_dx[i]
                twist_gradient[0, b + 3] = dN_dx[i]
            KG += axial_compression * (G_matrix.T @ G_matrix) * (L / 2.0 * weight)
            KG += axial_compression * polar_ratio * (twist_gradient.T @ twist_gradient) * (L / 2.0 * weight)
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
    "shell3": ShellElement,
    "tri3": ShellElement,
    "tria3": ShellElement,
    "t3": ShellElement,
    "s3": ShellElement,
    "shell6": ShellElement,
    "tri6": ShellElement,
    "tria6": ShellElement,
    "t6": ShellElement,
    "s6": ShellElement,
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
    normalized_type = str(element_type).lower()
    if normalized_type not in ELEMENT_TYPES:
        raise ValueError(f"Unknown element type: {element_type}")
    if normalized_type == "coupled":
        if len(node_ids) != 2:
            raise ValueError("CoupledBeamShellElement factory requires [beam_node_id, shell_node_id]")
        return CoupledBeamShellElement(element_id, node_ids[0], node_ids[1], material_name, **kwargs)
    return ELEMENT_TYPES[normalized_type](element_id, node_ids, material_name, **kwargs)
