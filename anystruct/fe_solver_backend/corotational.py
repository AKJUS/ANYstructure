"""Element-independent corotational kinematics for large rigid rotations.

This module implements the corotational (CR) formulation for the nonlinear
static solver.  Each element's rigid-body rotation is extracted from the
current deformed geometry, the nodal displacements are pulled back to the
reference configuration (removing the rigid motion), the element's own
nonlinear local response (layered J2 shells, fiber beams, local von Karman
coupling) acts on the small deformational part, and the resulting forces are
rotated forward to the current configuration:

    u_d,i (translation) = R_rig^T (x_i - x_c) - (X_i - X_c)
    u_d,i (rotation)    = rotvec(R_rig^T exp(skew(theta_i)))
    f_global            = E f_local(u_d),  E = blockdiag(R_rig, R_rig, ...)
    K_tangent           = E k_local E^T for shells (empirically stable), plus
                          frame-sensitivity geometric terms for beams

Scope and validity:

- small strains and small *deformational* rotations per element; rigid
  rotations of any magnitude (verified by rigid-rotation invariance);
- shells extract the rigid rotation from the element midsurface frame at the
  element center; beams from axis alignment plus mean axial twist;
- the deformational displacements are routed through the elements' own
  nonlinear local responses, so layered shell J2 plasticity, beam fiber
  plasticity and the local von Karman coupling are active in the corotated
  frame (plastic state is objective under rigid rotation); fracture/erosion
  remains unsupported in corotational mode;
- the tangent omits the rotational geometric stiffness, so Newton convergence
  is linear rather than quadratic near strongly rotating states — use more,
  smaller increments;
- the pull-back subtracts order-one nodal coordinates, so the internal force
  carries an intrinsic roundoff floor of roughly ``eps * ||K_e|| * L`` per
  element (~1e-7 N for steel at metre scale).  Use residual tolerances of
  1e-5..1e-6 relative and realistic load magnitudes; demanding convergence
  below the floor stalls the increment adaptation;
- eccentric beam-shell couplings use linear MPC constraints whose eccentricity
  vectors do not rotate; coupled regions should not undergo large rotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .fe_core import FEModel

_SMALL = 1.0e-12


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def rotation_matrix_from_vector(vector: np.ndarray) -> np.ndarray:
    """Rodrigues exponential map: rotation vector -> rotation matrix."""
    vector = np.asarray(vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < _SMALL:
        return np.eye(3) + _skew(vector)
    axis = vector / angle
    K = _skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rotation_vector_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Logarithmic map: rotation matrix -> rotation vector (robust near 0 and pi)."""
    R = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(R))
    cos_angle = min(max(0.5 * (trace - 1.0), -1.0), 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1.0e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float) * 0.5
    if angle > np.pi - 1.0e-6:
        # near pi: extract axis from the symmetric part
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        # fix signs from off-diagonal terms using the largest component
        k = int(np.argmax(axis))
        if axis[k] > _SMALL:
            for j in range(3):
                if j != k:
                    axis[j] = A[j, k] / axis[k]
        norm = float(np.linalg.norm(axis))
        axis = axis / norm if norm > _SMALL else np.array([1.0, 0.0, 0.0])
        return angle * axis
    factor = angle / (2.0 * np.sin(angle))
    return factor * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=float)


def _minimal_rotation(from_direction: np.ndarray, to_direction: np.ndarray) -> np.ndarray:
    """Smallest rotation mapping one unit vector onto another."""
    a = np.asarray(from_direction, dtype=float).reshape(3)
    b = np.asarray(to_direction, dtype=float).reshape(3)
    cross = np.cross(a, b)
    dot = float(a @ b)
    if dot < -1.0 + 1.0e-12:
        # antiparallel: rotate pi about any axis orthogonal to a
        seed = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, seed)
        axis = axis / max(float(np.linalg.norm(axis)), _SMALL)
        return rotation_matrix_from_vector(np.pi * axis)
    K = _skew(cross)
    return np.eye(3) + K + (K @ K) / (1.0 + dot)


def _shell_center_frame(element: Any, coords: np.ndarray) -> np.ndarray:
    """Element midsurface frame (columns = local axes) at the element center."""
    if element.num_nodes in (3, 6):
        xi, eta = 1.0 / 3.0, 1.0 / 3.0
    else:
        xi, eta = 0.0, 0.0
    _N, dN_dxi, dN_deta = element.compute_shape_functions(xi, eta)
    R, _dN_dx, _dN_dy, _det_j = element._local_frame_and_derivatives(coords, dN_dxi, dN_deta)
    return np.asarray(R, dtype=float)


def _beam_rigid_rotation(reference: np.ndarray, deformed: np.ndarray, node_rotations: np.ndarray) -> np.ndarray:
    """Beam rigid rotation: axis alignment composed with mean axial twist."""
    t_ref = reference[-1] - reference[0]
    t_def = deformed[-1] - deformed[0]
    t_ref = t_ref / max(float(np.linalg.norm(t_ref)), _SMALL)
    t_def = t_def / max(float(np.linalg.norm(t_def)), _SMALL)
    R_align = _minimal_rotation(t_ref, t_def)
    twists = []
    for theta in node_rotations:
        R_node = rotation_matrix_from_vector(theta)
        residual = rotation_vector_from_matrix(R_align.T @ R_node)
        twists.append(float(residual @ t_ref))
    mean_twist = float(np.mean(twists)) if twists else 0.0
    return R_align @ rotation_matrix_from_vector(mean_twist * t_ref)


def _element_category(element: Any) -> Optional[str]:
    from .elements import BeamElement, ShellElement

    if isinstance(element, ShellElement):
        return "shell"
    if isinstance(element, BeamElement):
        return "beam"
    return None


class _CorotationalReference:
    """Reference geometry, frame and linear stiffness per element."""

    __slots__ = ("stiffness", "coordinates", "centroid", "frame", "category")

    def __init__(self, element: Any, model: "FEModel"):
        material = model.get_material(element.material_name)
        self.stiffness = np.asarray(element.compute_stiffness_matrix(model.mesh, material), dtype=float).copy()
        self.coordinates = np.asarray(element.get_node_coordinates(model.mesh), dtype=float).copy()
        self.centroid = self.coordinates.mean(axis=0)
        self.category = _element_category(element)
        if self.category == "shell":
            self.frame = _shell_center_frame(element, self.coordinates)
        else:
            self.frame = None


def _corotational_cache(model: "FEModel") -> Dict[int, _CorotationalReference]:
    mesh = model.mesh
    signature = mesh.revision_signature()
    cached = getattr(mesh, "_corotational_cache", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    cache: Dict[int, _CorotationalReference] = {}
    for element_id, element in mesh.elements.items():
        if _element_category(element) is not None:
            cache[int(element_id)] = _CorotationalReference(element, model)
    mesh._corotational_cache = (signature, cache)
    return cache


def corotational_element_response(
    model: "FEModel",
    element_id: int,
    element: Any,
    u_element: np.ndarray,
    tangent: bool,
    committed_state: Optional[Any] = None,
    num_layers: int = 5,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Any]]:
    """Corotational internal force, tangent and trial state for one element.

    The deformational displacement ``u_d`` is a small-deformation field on the
    reference geometry, so it is routed through the element's own nonlinear
    local response — layered J2 plasticity for shells and fiber sections for
    beams — and the resulting force/tangent are rotated forward with the rigid
    frame.  Plastic state therefore lives in the corotated frame and is
    objective under arbitrary rigid rotation.

    Returns ``(None, None, None)`` for element types outside the corotational
    scope so the caller can fall back to the element's own response.
    """
    reference = _corotational_cache(model).get(int(element_id))
    if reference is None or reference.category is None:
        return None, None, None
    num_nodes = int(element.num_nodes)
    u = np.asarray(u_element, dtype=float).reshape(num_nodes, 6)
    translations = u[:, :3]
    rotations = u[:, 3:]
    deformed = reference.coordinates + translations

    if reference.category == "shell":
        R_ref = reference.frame
    R_def = None
    if reference.category == "shell":
        R_def = _shell_center_frame(element, deformed)
        R_rig = R_def @ reference.frame.T
    else:
        R_rig = _beam_rigid_rotation(reference.coordinates, deformed, rotations)

    centroid_def = deformed.mean(axis=0)
    u_d = np.zeros((num_nodes, 6), dtype=float)
    for node in range(num_nodes):
        u_d[node, :3] = R_rig.T @ (deformed[node] - centroid_def) - (reference.coordinates[node] - reference.centroid)
        R_node = rotation_matrix_from_vector(rotations[node])
        u_d[node, 3:] = rotation_vector_from_matrix(R_rig.T @ R_node)

    material = model.get_material(element.material_name)
    f_ref, k_ref, trial_state = element.compute_nonlinear_response(
        model.mesh, material, u_d.reshape(-1), committed_state, int(num_layers), tangent
    )
    f_ref = np.asarray(f_ref, dtype=float).reshape(-1)
    E = np.zeros((num_nodes * 6, num_nodes * 6), dtype=float)
    for node in range(num_nodes):
        E[node * 6 : node * 6 + 3, node * 6 : node * 6 + 3] = R_rig
        E[node * 6 + 3 : node * 6 + 6, node * 6 + 3 : node * 6 + 6] = R_rig
    f_global = E @ f_ref
    if not tangent:
        return f_global, None, trial_state
    k_ref = np.asarray(k_ref, dtype=float)
    # The rotated local tangent is the empirically stable choice for both
    # element families: adding the frame-sensitivity geometric terms (see
    # _consistent_corotational_tangent, kept for future nonsymmetric-tangent
    # work) makes the symmetrized Newton map repulsive near equilibrium for
    # bending-dominated shells and for plastically softened beams, while
    # E k E^T converges in a handful of iterations without line search.
    k_global = E @ k_ref @ E.T
    return f_global, k_global, trial_state


def _rotation_right_jacobian(vector: np.ndarray) -> np.ndarray:
    """Right Jacobian of the exponential map: d exp(theta + d) ~ exp(theta) skew(Jr d)."""
    vector = np.asarray(vector, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    K = _skew(vector)
    if angle < 1.0e-6:
        return np.eye(3) - 0.5 * K + (K @ K) / 6.0
    return (
        np.eye(3)
        - (1.0 - np.cos(angle)) / angle**2 * K
        + (angle - np.sin(angle)) / angle**3 * (K @ K)
    )


def _rigid_rotation_for_state(
    reference: "_CorotationalReference", element: Any, deformed: np.ndarray, rotations: np.ndarray
) -> np.ndarray:
    if reference.category == "shell":
        return _shell_center_frame(element, deformed) @ reference.frame.T
    return _beam_rigid_rotation(reference.coordinates, deformed, rotations)


def _rotation_sensitivity(
    reference: "_CorotationalReference",
    element: Any,
    deformed: np.ndarray,
    rotations: np.ndarray,
    R_rig: np.ndarray,
) -> np.ndarray:
    """G = d(omega)/du (3 x 6n): rigid-rotation sensitivity by central differences.

    ``omega`` is the left-increment rotation vector of the extracted rigid
    rotation, ``R_rig(u + du) ~ exp(skew(G du)) R_rig(u)``.  Shell frames
    depend only on the nodal translations; beam frames additionally pick up
    twist from the nodal rotations.
    """
    num_nodes = deformed.shape[0]
    G = np.zeros((3, num_nodes * 6), dtype=float)
    spread = float(np.max(np.linalg.norm(reference.coordinates - reference.centroid, axis=1)))
    step_translation = 1.0e-6 * max(spread, 1.0e-3)
    step_rotation = 1.0e-6
    include_rotations = reference.category == "beam"
    Rt = R_rig.T
    for node in range(num_nodes):
        for axis in range(3):
            perturbed = deformed.copy()
            perturbed[node, axis] += step_translation
            R_plus = _rigid_rotation_for_state(reference, element, perturbed, rotations)
            perturbed[node, axis] -= 2.0 * step_translation
            R_minus = _rigid_rotation_for_state(reference, element, perturbed, rotations)
            delta = rotation_vector_from_matrix(R_plus @ Rt) - rotation_vector_from_matrix(R_minus @ Rt)
            G[:, node * 6 + axis] = delta / (2.0 * step_translation)
        if include_rotations:
            for axis in range(3):
                perturbed_rotations = rotations.copy()
                perturbed_rotations[node, axis] += step_rotation
                R_plus = _rigid_rotation_for_state(reference, element, deformed, perturbed_rotations)
                perturbed_rotations[node, axis] -= 2.0 * step_rotation
                R_minus = _rigid_rotation_for_state(reference, element, deformed, perturbed_rotations)
                delta = rotation_vector_from_matrix(R_plus @ Rt) - rotation_vector_from_matrix(R_minus @ Rt)
                G[:, node * 6 + 3 + axis] = delta / (2.0 * step_rotation)
    return G


def _consistent_corotational_tangent(
    reference: "_CorotationalReference",
    element: Any,
    deformed: np.ndarray,
    rotations: np.ndarray,
    R_rig: np.ndarray,
    E: np.ndarray,
    f_global: np.ndarray,
    k_ref: np.ndarray,
) -> np.ndarray:
    """Consistent (symmetrized) corotational tangent.

    Chain rule of ``f = E(omega) K_ref u_d(u, omega)`` through the three
    dependency paths:

    - ``D`` — pull-back derivative at fixed frame: centered translations and
      right-Jacobian-corrected nodal rotation increments;
    - ``U`` — pull-back sensitivity to the rigid rotation;
    - ``S`` — rotation of the internal force with the frame.

    ``K = E K_ref D + (S + E K_ref U) G`` with ``G`` the frame sensitivity
    from :func:`_rotation_sensitivity`.  The result is symmetrized for the
    solver's symmetric factorization; the skew part vanishes at equilibrium.
    """
    num_nodes = deformed.shape[0]
    n_dofs = num_nodes * 6
    Rt = R_rig.T
    centroid_def = deformed.mean(axis=0)

    D = np.zeros((n_dofs, n_dofs), dtype=float)
    average = 1.0 / num_nodes
    for i in range(num_nodes):
        for j in range(num_nodes):
            weight = (1.0 if i == j else 0.0) - average
            D[i * 6 : i * 6 + 3, j * 6 : j * 6 + 3] = weight * Rt
        D[i * 6 + 3 : i * 6 + 6, i * 6 + 3 : i * 6 + 6] = Rt @ _rotation_right_jacobian(rotations[i])

    U = np.zeros((n_dofs, 3), dtype=float)
    for i in range(num_nodes):
        U[i * 6 : i * 6 + 3, :] = Rt @ _skew(deformed[i] - centroid_def)
        U[i * 6 + 3 : i * 6 + 6, :] = -Rt

    S = np.zeros((n_dofs, 3), dtype=float)
    for i in range(num_nodes):
        S[i * 6 : i * 6 + 3, :] = -_skew(f_global[i * 6 : i * 6 + 3])
        S[i * 6 + 3 : i * 6 + 6, :] = -_skew(f_global[i * 6 + 3 : i * 6 + 6])

    G = _rotation_sensitivity(reference, element, deformed, rotations, R_rig)
    EK = E @ k_ref
    tangent = EK @ D + (S + EK @ U) @ G
    return 0.5 * (tangent + tangent.T)


def validate_corotational_scope(model: "FEModel") -> None:
    """Validate corotational applicability.

    Since Phase 4 the corotational path routes the deformational displacements
    through the elements' own nonlinear local responses, so layered shell J2
    plasticity and beam fiber plasticity are supported.  The function is kept
    as the hook for future scope checks.
    """
    return None
