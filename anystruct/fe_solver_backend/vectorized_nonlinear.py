import numpy as np
from typing import Tuple, Optional, Any, Dict

from .jit_compiler import njit
from .plasticity import _jit_plane_stress_return_map
from .elements import lobatto_layers, plane_stress_elastic_matrix


@njit
def _jit_batch_integrate_nonlinear_response(
    u_loc_batch: np.ndarray,
    N_res_batch: np.ndarray,
    M_res_batch: np.ndarray,
    C0_batch: np.ndarray,
    C1_batch: np.ndarray,
    C2_batch: np.ndarray,
    B_m_all_batch: np.ndarray,
    B_b_all_batch: np.ndarray,
    B_d_all_batch: np.ndarray,
    Gw_all_batch: np.ndarray,
    detw_all_batch: np.ndarray,
    B_s_all_batch: np.ndarray,
    detw_shear_all_batch: np.ndarray,
    D_shear: np.ndarray,
    drilling_stiffness: float,
    tangent: bool,
    has_plasticity: bool,
    n_dof: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_elem = u_loc_batch.shape[0]
    F_loc_batch = np.zeros((n_elem, n_dof))
    K_loc_batch = np.zeros((n_elem, n_dof, n_dof))
    
    n_gp = detw_all_batch.shape[1]
    
    for e in range(n_elem):
        u_loc = u_loc_batch[e]
        detw_all = detw_all_batch[e]
        B_m_all = B_m_all_batch[e]
        B_b_all = B_b_all_batch[e]
        B_d_all = B_d_all_batch[e]
        Gw_all = Gw_all_batch[e]
        N_res = N_res_batch[e]
        M_res = M_res_batch[e]
        C0 = C0_batch[e]
        C1 = C1_batch[e]
        C2 = C2_batch[e]
        
        for g in range(n_gp):
            detw = detw_all[g]
            B_m = B_m_all[g]
            B_b = B_b_all[g]
            B_d = B_d_all[g]
            Gw = Gw_all[g]
            
            theta_0 = 0.0
            theta_1 = 0.0
            for i in range(n_dof):
                theta_0 += Gw[0, i] * u_loc[i]
                theta_1 += Gw[1, i] * u_loc[i]
                
            B_eff = np.zeros((3, n_dof))
            for i in range(n_dof):
                B_eff[0, i] = B_m[0, i] + theta_0 * Gw[0, i]
                B_eff[1, i] = B_m[1, i] + theta_1 * Gw[1, i]
                B_eff[2, i] = B_m[2, i] + theta_0 * Gw[1, i] + theta_1 * Gw[0, i]
                
            N_g = N_res[g]
            M_g = M_res[g]
            B_eff_T_N = np.zeros(n_dof)
            B_b_T_M = np.zeros(n_dof)
            for i in range(n_dof):
                B_eff_T_N[i] = B_eff[0, i] * N_g[0] + B_eff[1, i] * N_g[1] + B_eff[2, i] * N_g[2]
                B_b_T_M[i] = B_b[0, i] * M_g[0] + B_b[1, i] * M_g[1] + B_b[2, i] * M_g[2]
                
            Bd_u = 0.0
            for i in range(n_dof):
                Bd_u += B_d[0, i] * u_loc[i]
                
            for i in range(n_dof):
                F_loc_batch[e, i] += (B_eff_T_N[i] + B_b_T_M[i]) * detw
                F_loc_batch[e, i] += B_d[0, i] * (drilling_stiffness * Bd_u) * detw
                
            if not tangent:
                continue
                
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
                    K_loc_batch[e, r, c] += (val_eff + val_b) * detw
                    
            if has_plasticity:
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
                        K_loc_batch[e, r, c] += (val_c + val_c) * detw
                        
            N00 = N_g[0]
            N11 = N_g[1]
            N01 = N_g[2]
            
            N_Gw_0 = N00 * Gw[0] + N01 * Gw[1]
            N_Gw_1 = N01 * Gw[0] + N11 * Gw[1]
            for r in range(n_dof):
                for c in range(n_dof):
                    K_loc_batch[e, r, c] += (Gw[0, r] * N_Gw_0[c] + Gw[1, r] * N_Gw_1[c]) * detw
                    K_loc_batch[e, r, c] += B_d[0, r] * (drilling_stiffness * B_d[0, c]) * detw

        n_shear = detw_shear_all_batch.shape[1]
        detw_shear_all = detw_shear_all_batch[e]
        B_s_all = B_s_all_batch[e]
        
        for g in range(n_shear):
            detw_s = detw_shear_all[g]
            B_s = B_s_all[g]
            
            Bs_u_0 = 0.0
            Bs_u_1 = 0.0
            for i in range(n_dof):
                Bs_u_0 += B_s[0, i] * u_loc[i]
                Bs_u_1 += B_s[1, i] * u_loc[i]
                
            f_s_0 = D_shear[0, 0] * Bs_u_0 + D_shear[0, 1] * Bs_u_1
            f_s_1 = D_shear[1, 0] * Bs_u_0 + D_shear[1, 1] * Bs_u_1
            
            for i in range(n_dof):
                F_loc_batch[e, i] += (B_s[0, i] * f_s_0 + B_s[1, i] * f_s_1) * detw_s
                
            if not tangent:
                continue
                
            C_s = np.zeros((2, n_dof))
            for r in range(2):
                for c in range(n_dof):
                    C_s[r, c] = D_shear[r, 0] * B_s[0, c] + D_shear[r, 1] * B_s[1, c]
                    
            for r in range(n_dof):
                for c in range(n_dof):
                    K_loc_batch[e, r, c] += (B_s[0, r] * C_s[0, c] + B_s[1, r] * C_s[1, c]) * detw_s
                    
    return F_loc_batch, K_loc_batch


def batch_shell_nonlinear_response(
    u_elem_batch: np.ndarray,
    T0_batch: np.ndarray,
    B_m_all_batch: np.ndarray,
    B_b_all_batch: np.ndarray,
    B_d_all_batch: np.ndarray,
    Gw_all_batch: np.ndarray,
    detw_all_batch: np.ndarray,
    B_s_all_batch: np.ndarray,
    detw_shear_all_batch: np.ndarray,
    E: float,
    nu: float,
    G_mod: float,
    h: float,
    drilling_stabilization: float,
    tangent: bool,
    curve: Any,
    plastic_strain_batch: np.ndarray,
    alpha_batch: np.ndarray,
    num_layers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes nonlinear response for a batch of elements."""
    n_elem = u_elem_batch.shape[0]
    n_dof = u_elem_batch.shape[1]
    
    u_loc_batch = (T0_batch @ u_elem_batch[:, :, None]).squeeze(-1)
    
    C_el = plane_stress_elastic_matrix(E, nu)
    D_shear = G_mod * (5.0 / 6.0) * h * np.eye(2, dtype=float)
    drilling_stiffness = G_mod * h * drilling_stabilization
    
    n_gp = detw_all_batch.shape[1]
    
    theta_batch = (Gw_all_batch @ u_loc_batch[:, None, :, None]).squeeze(-1)
    memb_strain_batch = (B_m_all_batch @ u_loc_batch[:, None, :, None]).squeeze(-1)
    memb_strain_batch += np.stack([
        0.5 * theta_batch[..., 0]**2,
        0.5 * theta_batch[..., 1]**2,
        theta_batch[..., 0] * theta_batch[..., 1]
    ], axis=-1)
    
    curvature_batch = (B_b_all_batch @ u_loc_batch[:, None, :, None]).squeeze(-1)
    
    has_plasticity = curve is not None
    if not has_plasticity:
        N_res_batch = memb_strain_batch @ (h * C_el).T
        M_res_batch = curvature_batch @ (h**3 / 12.0 * C_el).T
        C0_batch = np.broadcast_to(h * C_el, (n_elem, n_gp, 3, 3))
        C1_batch = np.zeros((n_elem, n_gp, 3, 3), dtype=float)
        C2_batch = np.broadcast_to(h**3 / 12.0 * C_el, (n_elem, n_gp, 3, 3))
        ep_new = plastic_strain_batch
        alpha_new = alpha_batch
        layer_strain = np.zeros((n_elem * n_gp * num_layers, 3), dtype=float)
    else:
        z_layers, w_layers = lobatto_layers(num_layers, h)
        layer_strain = (
            memb_strain_batch[:, :, None, :] + z_layers[None, None, :, None] * curvature_batch[:, :, None, :]
        ).reshape(n_elem * n_gp * num_layers, 3)
        
        sigma, C_tan, ep_new, alpha_new = _jit_plane_stress_return_map(
            layer_strain,
            plastic_strain_batch.reshape(n_elem * n_gp * num_layers, 3),
            alpha_batch.reshape(n_elem * n_gp * num_layers),
            E,
            nu,
            float(curve.sigma_prop),
            float(curve.sigma_yield),
            float(curve.sigma_yield_2),
            float(curve.eps_p_y1),
            float(curve.eps_p_y2),
            float(curve.K),
            float(curve.n),
            float(curve._power_offset),
            30,
            1.0e-10,
            tangent,
        )
        
        ep_new = ep_new.reshape(n_elem, n_gp * num_layers, 3)
        alpha_new = alpha_new.reshape(n_elem, n_gp * num_layers)
        
        sigma = sigma.reshape(n_elem, n_gp, num_layers, 3)
        C_tan = C_tan.reshape(n_elem, n_gp, num_layers, 3, 3)
        
        N_res_batch, M_res_batch, C0_batch, C1_batch, C2_batch = _jit_integrate_layers(
            sigma, C_tan, w_layers, z_layers, tangent
        )

    F_loc_batch, K_loc_batch = _jit_batch_integrate_nonlinear_response(
        u_loc_batch,
        N_res_batch,
        M_res_batch,
        C0_batch,
        C1_batch,
        C2_batch,
        B_m_all_batch,
        B_b_all_batch,
        B_d_all_batch,
        Gw_all_batch,
        detw_all_batch,
        B_s_all_batch,
        detw_shear_all_batch,
        D_shear,
        drilling_stiffness,
        tangent,
        has_plasticity,
        n_dof,
    )
    
    F_int_batch = (T0_batch.transpose(0, 2, 1) @ F_loc_batch[:, :, None]).squeeze(-1)
    if tangent:
        K_T_batch = T0_batch.transpose(0, 2, 1) @ K_loc_batch @ T0_batch
    else:
        K_T_batch = np.zeros((n_elem, 0, 0))
        
    return F_int_batch, K_T_batch, ep_new, alpha_new, layer_strain


@njit
def _jit_integrate_layers(
    sigma: np.ndarray,
    C_tan: np.ndarray,
    w_layers: np.ndarray,
    z_layers: np.ndarray,
    tangent: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_elem, n_gp, num_layers, _ = sigma.shape
    N_res_batch = np.zeros((n_elem, n_gp, 3))
    M_res_batch = np.zeros((n_elem, n_gp, 3))
    C0_batch = np.zeros((n_elem, n_gp, 3, 3))
    C1_batch = np.zeros((n_elem, n_gp, 3, 3))
    C2_batch = np.zeros((n_elem, n_gp, 3, 3))
    
    for e in range(n_elem):
        for g in range(n_gp):
            for l in range(num_layers):
                w = w_layers[l]
                z = z_layers[l]
                wz = w * z
                wz2 = wz * z
                for i in range(3):
                    N_res_batch[e, g, i] += w * sigma[e, g, l, i]
                    M_res_batch[e, g, i] += wz * sigma[e, g, l, i]
                    
                if tangent:
                    for i in range(3):
                        for j in range(3):
                            val = C_tan[e, g, l, i, j]
                            C0_batch[e, g, i, j] += w * val
                            C1_batch[e, g, i, j] += wz * val
                            C2_batch[e, g, i, j] += wz2 * val
                            
    return N_res_batch, M_res_batch, C0_batch, C1_batch, C2_batch
