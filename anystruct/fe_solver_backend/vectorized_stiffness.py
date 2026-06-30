"""Vectorized JIT-compiled shell stiffness matrix formulation.

This module provides JIT-compiled routines to compute the global stiffness matrices
of a batch of ShellElements of the same configuration.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from .jit_compiler import njit, prange

_SMALL = 1.0e-12

@njit(cache=True)
def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

@njit(cache=True)
def _normalize(vector: np.ndarray) -> Tuple[np.ndarray, float]:
    norm = float(np.sqrt(np.dot(vector, vector)))
    if norm < _SMALL:
        return np.zeros(3, dtype=float), norm
    return vector / norm, norm

@njit(cache=True)
def _inv2(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if abs(det) < _SMALL:
        raise ValueError("singular 2x2 matrix")
    inv = np.array([
        [matrix[1, 1], -matrix[0, 1]],
        [-matrix[1, 0], matrix[0, 0]]
    ]) / det
    return inv, float(det)

@njit(cache=True)
def _compute_4node_shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = np.array([
        0.25 * (1.0 - xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 - eta),
        0.25 * (1.0 + xi) * (1.0 + eta),
        0.25 * (1.0 - xi) * (1.0 + eta)
    ])
    dN_dxi = np.array([
        -0.25 * (1.0 - eta),
        0.25 * (1.0 - eta),
        0.25 * (1.0 + eta),
        -0.25 * (1.0 + eta)
    ])
    dN_deta = np.array([
        -0.25 * (1.0 - xi),
        -0.25 * (1.0 + xi),
        0.25 * (1.0 + xi),
        0.25 * (1.0 - xi)
    ])
    return N, dN_dxi, dN_deta

@njit(cache=True)
def _compute_8node_shape_functions(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = np.zeros(8)
    N[0] = -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta)
    N[1] = -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta)
    N[2] = -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta)
    N[3] = -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta)
    N[4] = 0.5 * (1.0 - xi**2) * (1.0 - eta)
    N[5] = 0.5 * (1.0 + xi) * (1.0 - eta**2)
    N[6] = 0.5 * (1.0 - xi**2) * (1.0 + eta)
    N[7] = 0.5 * (1.0 - xi) * (1.0 - eta**2)

    dN_dxi = np.zeros(8)
    dN_dxi[0] = 0.25 * (1.0 - eta) * (1.0 + xi + eta) - 0.25 * (1.0 - xi) * (1.0 - eta)
    dN_dxi[1] = -0.25 * (1.0 - eta) * (1.0 - xi + eta) + 0.25 * (1.0 + xi) * (1.0 - eta)
    dN_dxi[2] = -0.25 * (1.0 + eta) * (1.0 - xi - eta) + 0.25 * (1.0 + xi) * (1.0 + eta)
    dN_dxi[3] = 0.25 * (1.0 + eta) * (1.0 + xi - eta) - 0.25 * (1.0 - xi) * (1.0 + eta)
    dN_dxi[4] = -xi * (1.0 - eta)
    dN_dxi[5] = 0.5 * (1.0 - eta**2)
    dN_dxi[6] = -xi * (1.0 + eta)
    dN_dxi[7] = -0.5 * (1.0 - eta**2)

    dN_deta = np.zeros(8)
    dN_deta[0] = 0.25 * (1.0 - xi) * (1.0 + xi + eta) - 0.25 * (1.0 - xi) * (1.0 - eta)
    dN_deta[1] = 0.25 * (1.0 + xi) * (1.0 - xi + eta) - 0.25 * (1.0 + xi) * (1.0 - eta)
    dN_deta[2] = -0.25 * (1.0 + xi) * (1.0 - xi - eta) + 0.25 * (1.0 + xi) * (1.0 + eta)
    dN_deta[3] = -0.25 * (1.0 - xi) * (1.0 + xi - eta) + 0.25 * (1.0 - xi) * (1.0 + eta)
    dN_deta[4] = -0.5 * (1.0 - xi**2)
    dN_deta[5] = -eta * (1.0 + xi)
    dN_deta[6] = 0.5 * (1.0 - xi**2)
    dN_deta[7] = -eta * (1.0 - xi)
    return N, dN_dxi, dN_deta

@njit(cache=True)
def _fallback_edge_direction_jit(coords: np.ndarray, normal: np.ndarray) -> np.ndarray:
    num_nodes = coords.shape[0]
    if num_nodes >= 2:
        edge = coords[1] - coords[0]
        proj = edge - np.dot(edge, normal) * normal
        e1, n = _normalize(proj)
        if n > _SMALL:
            return e1
    if num_nodes >= 4:
        edge = coords[2] - coords[3]
        proj = edge - np.dot(edge, normal) * normal
        e1, n = _normalize(proj)
        if n > _SMALL:
            return e1
        edge = coords[3] - coords[0]
        proj = edge - np.dot(edge, normal) * normal
        e1, n = _normalize(proj)
        if n > _SMALL:
            return e1
        edge = coords[2] - coords[1]
        proj = edge - np.dot(edge, normal) * normal
        e1, n = _normalize(proj)
        if n > _SMALL:
            return e1
    raise ValueError("Shell element has no valid in-plane direction")

@njit(cache=True)
def _local_frame_and_derivatives_jit(
    coords: np.ndarray,
    dN_dxi: np.ndarray,
    dN_deta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    J0 = np.zeros(3)
    J1 = np.zeros(3)
    for i in range(coords.shape[0]):
        J0 += coords[i] * dN_dxi[i]
        J1 += coords[i] * dN_deta[i]
        
    e3, det_j = _normalize(_cross3(J0, J1))
    if det_j < _SMALL:
        raise ValueError("Shell element has a near-zero surface Jacobian")
        
    e1_raw = J0 - np.dot(J0, e3) * e3
    e1, e1_norm = _normalize(e1_raw)
    if e1_norm < _SMALL:
        e1 = _fallback_edge_direction_jit(coords, e3)
        
    e2, e2_norm = _normalize(_cross3(e3, e1))
    if e2_norm < _SMALL:
        raise ValueError("Shell element has an invalid local y direction")
        
    e1, _ = _normalize(_cross3(e2, e3))
    R = np.zeros((3, 3))
    R[:, 0] = e1
    R[:, 1] = e2
    R[:, 2] = e3
    
    J_local = np.array([
        [np.dot(J0, e1), np.dot(J0, e2)],
        [np.dot(J1, e1), np.dot(J1, e2)]
    ])
    inv_j_local, _ = _inv2(J_local)
    
    dN_dx = inv_j_local[0, 0] * dN_dxi + inv_j_local[0, 1] * dN_deta
    dN_dy = inv_j_local[1, 0] * dN_dxi + inv_j_local[1, 1] * dN_deta
    return R, dN_dx, dN_dy, det_j

@njit(cache=True)
def _build_shell_b_matrices_jit(
    N: np.ndarray,
    dN_dx: np.ndarray,
    dN_dy: np.ndarray,
    total_dofs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    B_m = np.zeros((3, total_dofs))
    B_b = np.zeros((3, total_dofs))
    B_s = np.zeros((2, total_dofs))

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

@njit(cache=True)
def _build_drilling_b_matrix_jit(
    N: np.ndarray,
    dN_dx: np.ndarray,
    dN_dy: np.ndarray,
    total_dofs: int,
) -> np.ndarray:
    B_d = np.zeros((1, total_dofs))
    B_d[0, 0::6] = 0.5 * dN_dy
    B_d[0, 1::6] = -0.5 * dN_dx
    B_d[0, 5::6] = N
    return B_d

@njit(cache=True)
def _mitc4_shear_samples_jit(
    coords: np.ndarray,
    R: np.ndarray,
    total_dofs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    planar = np.zeros((4, 2))
    for i in range(4):
        planar[i, 0] = coords[i, 0] * R[0, 0] + coords[i, 1] * R[1, 0] + coords[i, 2] * R[2, 0]
        planar[i, 1] = coords[i, 0] * R[0, 1] + coords[i, 1] * R[1, 1] + coords[i, 2] * R[2, 1]
        
    pts = np.array([
        [0.0, -1.0],  # A
        [1.0, 0.0],   # B
        [0.0, 1.0],   # C
        [-1.0, 0.0]   # D
    ])
    
    samples_arr = np.zeros((4, 2, total_dofs))
    
    for pt_idx in range(4):
        xi = pts[pt_idx, 0]
        eta = pts[pt_idx, 1]
        N, dN_dxi, dN_deta = _compute_4node_shape_functions(xi, eta)
        
        x_xi = 0.0
        y_xi = 0.0
        x_eta = 0.0
        y_eta = 0.0
        for i in range(4):
            x_xi += dN_dxi[i] * planar[i, 0]
            y_xi += dN_dxi[i] * planar[i, 1]
            x_eta += dN_deta[i] * planar[i, 0]
            y_eta += dN_deta[i] * planar[i, 1]
            
        row_xi = np.zeros(total_dofs)
        row_eta = np.zeros(total_dofs)
        
        row_xi[2::6] = dN_dxi
        row_xi[3::6] = -N * y_xi
        row_xi[4::6] = N * x_xi
        
        row_eta[2::6] = dN_deta
        row_eta[3::6] = -N * y_eta
        row_eta[4::6] = N * x_eta
        
        samples_arr[pt_idx, 0] = row_xi
        samples_arr[pt_idx, 1] = row_eta
        
    return planar, samples_arr

@njit(cache=True)
def _mitc4_shear_b_matrix_jit(
    planar: np.ndarray,
    samples_arr: np.ndarray,
    xi: float,
    eta: float,
    total_dofs: int,
) -> Tuple[np.ndarray, float]:
    _, dN_dxi, dN_deta = _compute_4node_shape_functions(xi, eta)
    J2 = np.zeros((2, 2))
    for i in range(4):
        J2[0, 0] += dN_dxi[i] * planar[i, 0]
        J2[0, 1] += dN_dxi[i] * planar[i, 1]
        J2[1, 0] += dN_deta[i] * planar[i, 0]
        J2[1, 1] += dN_deta[i] * planar[i, 1]
        
    inv_j2, det_j = _inv2(J2)
    
    B_covariant = np.zeros((2, total_dofs))
    for i in range(total_dofs):
        B_covariant[0, i] = 0.5 * (1.0 - eta) * samples_arr[0, 0, i] + 0.5 * (1.0 + eta) * samples_arr[2, 0, i]
        B_covariant[1, i] = 0.5 * (1.0 - xi) * samples_arr[3, 1, i] + 0.5 * (1.0 + xi) * samples_arr[1, 1, i]
        
    res = np.zeros((2, total_dofs))
    for r in range(2):
        for c in range(total_dofs):
            val = 0.0
            for k in range(2):
                val += inv_j2[r, k] * B_covariant[k, c]
            res[r, c] = val
            
    return res, det_j

@njit(cache=True, parallel=True)
def compute_shell_stiffness_matrices_jit(
    coords_all: np.ndarray,      # (N, num_nodes, 3)
    is_4node: bool,
    thickness: float,
    drilling_stabilization: float,
    E: float,
    nu: float,
    G: float,
    gauss_points: np.ndarray,    # (num_gp, 2)
    gauss_weights: np.ndarray,   # (num_gp,)
    shear_points: np.ndarray,    # (num_shear_gp, 2)
    shear_weights: np.ndarray,   # (num_shear_gp,)
) -> np.ndarray:                 # (N, total_dofs, total_dofs)
    N_elem = coords_all.shape[0]
    num_nodes = 4 if is_4node else 8
    total_dofs = num_nodes * 6
    n_blocks = 2 * num_nodes
    
    shell_plane = np.zeros((3, 3))
    shell_plane[0, 0] = 1.0
    shell_plane[0, 1] = nu
    shell_plane[1, 0] = nu
    shell_plane[1, 1] = 1.0
    shell_plane[2, 2] = (1.0 - nu) / 2.0
    
    h = thickness
    D_membrane = E * h / (1.0 - nu**2) * shell_plane
    D_bending = E * h**3 / (12.0 * (1.0 - nu**2)) * shell_plane
    
    D_shear = np.zeros((2, 2))
    D_shear[0, 0] = G * (5.0 / 6.0) * h
    D_shear[1, 1] = G * (5.0 / 6.0) * h
    
    drilling_stiffness = G * h * drilling_stabilization
    
    K_all = np.zeros((N_elem, total_dofs, total_dofs))
    
    for e in prange(N_elem):
        coords = coords_all[e]
        K = np.zeros((total_dofs, total_dofs))
        
        num_gp = gauss_points.shape[0]
        for gp_idx in range(num_gp):
            xi = gauss_points[gp_idx, 0]
            eta = gauss_points[gp_idx, 1]
            weight = gauss_weights[gp_idx]
            
            if is_4node:
                N, dN_dxi, dN_deta = _compute_4node_shape_functions(xi, eta)
            else:
                N, dN_dxi, dN_deta = _compute_8node_shape_functions(xi, eta)
                
            R, dN_dx, dN_dy, det_j = _local_frame_and_derivatives_jit(coords, dN_dxi, dN_deta)
            B_m, B_b, _ = _build_shell_b_matrices_jit(N, dN_dx, dN_dy, total_dofs)
            B_d = _build_drilling_b_matrix_jit(N, dN_dx, dN_dy, total_dofs)
            
            scale = det_j * weight
            
            D_B_m = np.zeros((3, total_dofs))
            D_B_b = np.zeros((3, total_dofs))
            for r in range(3):
                for c in range(total_dofs):
                    val_m = 0.0
                    val_b = 0.0
                    for k in range(3):
                        val_m += D_membrane[r, k] * B_m[k, c]
                        val_b += D_bending[r, k] * B_b[k, c]
                    D_B_m[r, c] = val_m
                    D_B_b[r, c] = val_b
                    
            K_local = np.zeros((total_dofs, total_dofs))
            for r in range(total_dofs):
                for c in range(total_dofs):
                    val = 0.0
                    for k in range(3):
                        val += B_m[k, r] * D_B_m[k, c] + B_b[k, r] * D_B_b[k, c]
                    K_local[r, c] = val * scale
                    K_local[r, c] += B_d[0, r] * drilling_stiffness * B_d[0, c] * scale
            
            # Global transformation block-by-block
            for I in range(n_blocks):
                for J in range(n_blocks):
                    block = np.zeros((3, 3))
                    for r in range(3):
                        for c in range(3):
                            block[r, c] = K_local[3*I + r, 3*J + c]
                            
                    R_block_temp = np.zeros((3, 3))
                    for r in range(3):
                        for c in range(3):
                            val = 0.0
                            for k in range(3):
                                val += R[r, k] * block[k, c]
                            R_block_temp[r, c] = val
                    R_block = np.zeros((3, 3))
                    for r in range(3):
                        for c in range(3):
                            val = 0.0
                            for k in range(3):
                                val += R_block_temp[r, k] * R[c, k]
                            R_block[r, c] = val
                            
                    for r in range(3):
                        for c in range(3):
                            K[3*I + r, 3*J + c] += R_block[r, c]
                            
        # Transverse shear
        if is_4node:
            N_c, dN_dxi_c, dN_deta_c = _compute_4node_shape_functions(0.0, 0.0)
            R_center, _, _, _ = _local_frame_and_derivatives_jit(coords, dN_dxi_c, dN_deta_c)
            
            planar, samples_arr = _mitc4_shear_samples_jit(coords, R_center, total_dofs)
            
            gps_shear = np.array([
                [-1.0, -1.0],
                [1.0, -1.0],
                [-1.0, 1.0],
                [1.0, 1.0]
            ]) / np.sqrt(3.0)
            w_shear = np.array([1.0, 1.0, 1.0, 1.0])
            
            for sh_idx in range(4):
                xi = gps_shear[sh_idx, 0]
                eta = gps_shear[sh_idx, 1]
                weight = w_shear[sh_idx]
                
                B_s, det_j = _mitc4_shear_b_matrix_jit(planar, samples_arr, xi, eta, total_dofs)
                scale = det_j * weight
                
                D_B_s = np.zeros((2, total_dofs))
                for r in range(2):
                    for c in range(total_dofs):
                        val = 0.0
                        for k in range(2):
                            val += D_shear[r, k] * B_s[k, c]
                        D_B_s[r, c] = val
                        
                K_local = np.zeros((total_dofs, total_dofs))
                for r in range(total_dofs):
                    for c in range(total_dofs):
                        val = 0.0
                        for k in range(2):
                            val += B_s[k, r] * D_B_s[k, c]
                        K_local[r, c] = val * scale
                        
                for I in range(n_blocks):
                    for J in range(n_blocks):
                        block = np.zeros((3, 3))
                        for r in range(3):
                            for c in range(3):
                                block[r, c] = K_local[3*I + r, 3*J + c]
                        R_block_temp = np.zeros((3, 3))
                        for r in range(3):
                            for c in range(3):
                                val = 0.0
                                for k in range(3):
                                    val += R_center[r, k] * block[k, c]
                                R_block_temp[r, c] = val
                        R_block = np.zeros((3, 3))
                        for r in range(3):
                            for c in range(3):
                                val = 0.0
                                for k in range(3):
                                    val += R_block_temp[r, k] * R_center[c, k]
                                R_block[r, c] = val
                        for r in range(3):
                            for c in range(3):
                                K[3*I + r, 3*J + c] += R_block[r, c]
                                
        else:
            num_shear_gp = shear_points.shape[0]
            for sh_idx in range(num_shear_gp):
                xi = shear_points[sh_idx, 0]
                eta = shear_points[sh_idx, 1]
                weight = shear_weights[sh_idx]
                
                N, dN_dxi, dN_deta = _compute_8node_shape_functions(xi, eta)
                R, dN_dx, dN_dy, det_j = _local_frame_and_derivatives_jit(coords, dN_dxi, dN_deta)
                
                _, _, B_s = _build_shell_b_matrices_jit(N, dN_dx, dN_dy, total_dofs)
                scale = det_j * weight
                
                D_B_s = np.zeros((2, total_dofs))
                for r in range(2):
                    for c in range(total_dofs):
                        val = 0.0
                        for k in range(2):
                            val += D_shear[r, k] * B_s[k, c]
                        D_B_s[r, c] = val
                        
                K_local = np.zeros((total_dofs, total_dofs))
                for r in range(total_dofs):
                    for c in range(total_dofs):
                        val = 0.0
                        for k in range(2):
                            val += B_s[k, r] * D_B_s[k, c]
                        K_local[r, c] = val * scale
                        
                for I in range(n_blocks):
                    for J in range(n_blocks):
                        block = np.zeros((3, 3))
                        for r in range(3):
                            for c in range(3):
                                block[r, c] = K_local[3*I + r, 3*J + c]
                        R_block_temp = np.zeros((3, 3))
                        for r in range(3):
                            for c in range(3):
                                val = 0.0
                                for k in range(3):
                                    val += R[r, k] * block[k, c]
                                R_block_temp[r, c] = val
                        R_block = np.zeros((3, 3))
                        for r in range(3):
                            for c in range(3):
                                val = 0.0
                                for k in range(3):
                                    val += R_block_temp[r, k] * R[c, k]
                                R_block[r, c] = val
                        for r in range(3):
                            for c in range(3):
                                K[3*I + r, 3*J + c] += R_block[r, c]
                                
        K_all[e] = K
        
    return K_all
