"""Vectorized J2 plane-stress plasticity with isotropic hardening.

The return mapping follows the classical plane-stress projected algorithm
(Simo & Hughes).  In the eigenbasis of C*P the update decouples into two
scalar modes, so the plastic multiplier is found by a scalar Newton iteration
that runs simultaneously for every yielding integration point / thickness
layer (numpy arrays, no Python-level point loops).

When a tangent is requested for a nonlinear global solve, the public wrapper
returns a consistent algorithmic tangent obtained by differentiating the same
discrete stress update with central finite differences.  This is deliberately
more expensive than the older continuum tangent, but it keeps the shell
layered-plastic tangent consistent while a closed-form analytical algorithmic
tangent is still pending.

Conventions
-----------
Stress/strain vectors are [xx, yy, xy] with engineering shear strain.
The yield function is f = 1/2 sigma^T P sigma - 1/3 sigma_y(alpha)^2 with

    P = 1/3 [[ 2, -1, 0],
             [-1,  2, 0],
             [ 0,  0, 6]]

and alpha the equivalent plastic strain with rate
alpha_dot = lambda_dot * sqrt(2/3 sigma^T P sigma).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np

from .jit_compiler import njit

if TYPE_CHECKING:
    from .material_curves import DNVC208MaterialCurve

_P_MATRIX = np.array(
    [[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 6.0]],
    dtype=float,
) / 3.0


@njit
def plane_stress_elastic_matrix(E: float, nu: float) -> np.ndarray:
    return E / (1.0 - nu**2) * np.array(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]],
    )


@njit
def _jit_flow_stress(
    eps_p: np.ndarray,
    sigma_prop: float,
    sigma_yield: float,
    sigma_yield_2: float,
    eps_p_y1: float,
    eps_p_y2: float,
    K: float,
    n: float,
    power_offset: float,
) -> np.ndarray:
    eps_p = np.maximum(eps_p, 0.0)
    slope_1 = (sigma_yield - sigma_prop) / eps_p_y1
    slope_2 = (sigma_yield_2 - sigma_yield) / (eps_p_y2 - eps_p_y1)
    part_1 = sigma_prop + slope_1 * eps_p
    part_2 = sigma_yield + slope_2 * (eps_p - eps_p_y1)
    
    res = np.zeros(eps_p.shape[0])
    for i in range(eps_p.shape[0]):
        val = eps_p[i]
        if val <= eps_p_y1:
            res[i] = part_1[i]
        elif val <= eps_p_y2:
            res[i] = part_2[i]
        else:
            res[i] = K * np.power(max(val + power_offset, 1.0e-12), n)
    return res


@njit
def _jit_hardening_modulus(
    eps_p: np.ndarray,
    sigma_prop: float,
    sigma_yield: float,
    sigma_yield_2: float,
    eps_p_y1: float,
    eps_p_y2: float,
    K: float,
    n: float,
    power_offset: float,
) -> np.ndarray:
    eps_p = np.maximum(eps_p, 0.0)
    slope_1 = (sigma_yield - sigma_prop) / eps_p_y1
    slope_2 = (sigma_yield_2 - sigma_yield) / (eps_p_y2 - eps_p_y1)
    
    res = np.zeros(eps_p.shape[0])
    for i in range(eps_p.shape[0]):
        val = eps_p[i]
        if val <= eps_p_y1:
            res[i] = slope_1
        elif val <= eps_p_y2:
            res[i] = slope_2
        else:
            base = max(val + power_offset, 1.0e-12)
            res[i] = K * n * np.power(base, n - 1.0)
    return res


@njit
def _jit_plane_stress_return_map(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    E: float,
    nu: float,
    sigma_prop: float,
    sigma_yield: float,
    sigma_yield_2: float,
    eps_p_y1: float,
    eps_p_y2: float,
    K: float,
    n: float,
    power_offset: float,
    max_iterations: int = 30,
    tolerance: float = 1.0e-10,
    compute_tangent: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_points = strain.shape[0]
    C = plane_stress_elastic_matrix(E, nu)
    C_inv = np.linalg.inv(C)
    G = E / (2.0 * (1.0 + nu))
    c_a = E / (3.0 * (1.0 - nu))

    sigma = (strain - plastic_strain) @ C.T
    sy_n = _jit_flow_stress(
        alpha, sigma_prop, sigma_yield, sigma_yield_2, eps_p_y1, eps_p_y2, K, n, power_offset
    )
    a1 = sigma[:, 0] + sigma[:, 1]
    a2 = sigma[:, 0] - sigma[:, 1]
    a3 = sigma[:, 2]
    phi2_trial = a1**2 / 12.0 + a2**2 / 4.0 + a3**2
    f_trial = phi2_trial - sy_n**2 / 3.0

    yielding = f_trial > tolerance * np.maximum(sy_n**2, 1.0)
    C_ep = np.zeros((n_points, 3, 3))
    if compute_tangent:
        for i in range(n_points):
            C_ep[i] = C
    new_plastic = plastic_strain.copy()
    new_alpha = alpha.copy()

    if not np.any(yielding):
        return sigma, C_ep, new_plastic, new_alpha

    # Identify yielding indices
    yielding_indices = np.where(yielding)[0]
    n_yielding = yielding_indices.shape[0]

    b1 = a1[yielding]
    b23 = a2[yielding] ** 2 / 4.0 + a3[yielding] ** 2
    alpha_y = alpha[yielding]
    dl = np.zeros(n_yielding)

    for _ in range(max_iterations):
        dA = 1.0 + c_a * dl
        dB = 1.0 + 2.0 * G * dl
        phi2 = b1**2 / (12.0 * dA**2) + b23 / dB**2
        phi = np.sqrt(np.maximum(phi2, 1.0e-30))
        g = 2.0 * np.sqrt(np.maximum(phi2 / 3.0, 1.0e-30))
        alpha_new = alpha_y + dl * g
        sy = _jit_flow_stress(
            alpha_new, sigma_prop, sigma_yield, sigma_yield_2, eps_p_y1, eps_p_y2, K, n, power_offset
        )
        H = _jit_hardening_modulus(
            alpha_new, sigma_prop, sigma_yield, sigma_yield_2, eps_p_y1, eps_p_y2, K, n, power_offset
        )
        f = phi2 - sy**2 / 3.0

        all_scaled = True
        for i in range(n_yielding):
            if np.abs(f[i]) > tolerance * max(sy[i]**2, 1.0):
                all_scaled = False
                break
        if all_scaled:
            break

        d_phi2 = -2.0 * (b1**2 * c_a / (12.0 * dA**3) + 2.0 * G * b23 / dB**3)
        d_g = d_phi2 / (3.0 * np.maximum(np.sqrt(phi2 / 3.0), 1.0e-30))
        d_alpha = g + dl * d_g
        d_f = d_phi2 - (2.0 / 3.0) * sy * H * d_alpha
        
        # Safe division step
        for i in range(n_yielding):
            df_val = d_f[i]
            if np.abs(df_val) <= 1.0e-30:
                df_val = -1.0e-30 if df_val < 0.0 else 1.0e-30
            step = f[i] / df_val
            dl[i] = max(dl[i] - step, 0.0)

    dA = 1.0 + c_a * dl
    dB = 1.0 + 2.0 * G * dl
    sig_a = b1 / dA
    sig_b = a2[yielding] / dB
    tau = a3[yielding] / dB
    
    sigma_y_pts = np.zeros((n_yielding, 3))
    sigma_y_pts[:, 0] = (sig_a + sig_b) / 2.0
    sigma_y_pts[:, 1] = (sig_a - sig_b) / 2.0
    sigma_y_pts[:, 2] = tau

    for idx, i in enumerate(yielding_indices):
        sigma[i] = sigma_y_pts[idx]

    phi2 = sig_a**2 / 12.0 + sig_b**2 / 4.0 + tau**2
    new_alpha[yielding] = alpha_y + dl * 2.0 * np.sqrt(np.maximum(phi2 / 3.0, 1.0e-30))
    
    p_strain_yielding = strain[yielding] - sigma_y_pts @ C_inv.T
    for idx, i in enumerate(yielding_indices):
        new_plastic[i] = p_strain_yielding[idx]

    if not compute_tangent:
        return sigma, C_ep, new_plastic, new_alpha

    # Continuum elastoplastic tangent: C_ep = C - (C m)(C m)^T / (m^T C m + 4/9 sy^2 H)
    _P_MATRIX_local = np.array(
        [[2.0, -1.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 6.0]],
    ) / 3.0
    m = sigma_y_pts @ _P_MATRIX_local.T
    Cm = m @ C.T
    sy_final = _jit_flow_stress(
        new_alpha[yielding], sigma_prop, sigma_yield, sigma_yield_2, eps_p_y1, eps_p_y2, K, n, power_offset
    )
    H_final = _jit_hardening_modulus(
        new_alpha[yielding], sigma_prop, sigma_yield, sigma_yield_2, eps_p_y1, eps_p_y2, K, n, power_offset
    )
    
    denom = np.zeros(n_yielding)
    for idx in range(n_yielding):
        denom[idx] = m[idx, 0] * Cm[idx, 0] + m[idx, 1] * Cm[idx, 1] + m[idx, 2] * Cm[idx, 2]
    denom += (4.0 / 9.0) * sy_final**2 * H_final

    for idx, i in enumerate(yielding_indices):
        denom_val = max(denom[idx], 1.0e-30)
        for r in range(3):
            for c in range(3):
                C_ep[i, r, c] = C[r, c] - Cm[idx, r] * Cm[idx, c] / denom_val

    return sigma, C_ep, new_plastic, new_alpha


def plane_stress_return_map(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    E: float,
    nu: float,
    curve: "DNVC208MaterialCurve",
    max_iterations: int = 30,
    tolerance: float = 1.0e-10,
    compute_tangent: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map total strains to stresses, tangents and updated plastic state."""
    strain = np.asarray(strain, dtype=float)
    plastic_strain = np.asarray(plastic_strain, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    if curve is None:
        C = plane_stress_elastic_matrix(E, nu)
        n_points = strain.shape[0]
        C_ep = np.broadcast_to(C, (n_points, 3, 3)).copy() if compute_tangent else np.zeros((n_points, 3, 3))
        sigma = (strain - plastic_strain) @ C.T
        return sigma, C_ep, plastic_strain.copy(), alpha.copy()

    sigma, _continuum_tangent, new_plastic, new_alpha = _jit_plane_stress_return_map(
        strain,
        plastic_strain,
        alpha,
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
        max_iterations,
        tolerance,
        False,
    )
    if not compute_tangent:
        return sigma, _continuum_tangent, new_plastic, new_alpha

    C_alg = _finite_difference_algorithmic_tangent(
        strain,
        plastic_strain,
        alpha,
        E,
        nu,
        curve,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return sigma, C_alg, new_plastic, new_alpha


def _finite_difference_algorithmic_tangent(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    alpha: np.ndarray,
    E: float,
    nu: float,
    curve: "DNVC208MaterialCurve",
    max_iterations: int = 30,
    tolerance: float = 1.0e-10,
    step: float = 1.0e-7,
) -> np.ndarray:
    """Central-difference tangent of the exact discrete stress update."""
    n_points = int(strain.shape[0])
    tangent = np.zeros((n_points, 3, 3), dtype=float)
    for col in range(3):
        perturb = np.zeros_like(strain)
        perturb[:, col] = step
        sigma_plus, _, _, _ = _jit_plane_stress_return_map(
            strain + perturb,
            plastic_strain,
            alpha,
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
            max_iterations,
            tolerance,
            False,
        )
        sigma_minus, _, _, _ = _jit_plane_stress_return_map(
            strain - perturb,
            plastic_strain,
            alpha,
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
            max_iterations,
            tolerance,
            False,
        )
        tangent[:, :, col] = (sigma_plus - sigma_minus) / (2.0 * step)
    return tangent


_LOBATTO_RULES = {
    3: (np.array([-1.0, 0.0, 1.0]), np.array([1.0, 4.0, 1.0]) / 3.0),
    5: (
        np.array([-1.0, -np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0]),
        np.array([1.0 / 10.0, 49.0 / 90.0, 32.0 / 45.0, 49.0 / 90.0, 1.0 / 10.0]),
    ),
    7: (
        np.array(
            [-1.0, -0.830223896278567, -0.468848793470714, 0.0,
             0.468848793470714, 0.830223896278567, 1.0]
        ),
        np.array(
            [0.047619047619048, 0.276826047361566, 0.431745381209863, 0.487619047619048,
             0.431745381209863, 0.276826047361566, 0.047619047619048]
        ),
    ),
    9: (
        np.array(
            [-1.0, -0.899757995411460, -0.677186279510738, -0.363117463826178, 0.0,
             0.363117463826178, 0.677186279510738, 0.899757995411460, 1.0]
        ),
        np.array(
            [0.027777777777778, 0.165495361560806, 0.274538712500162, 0.346428510973046,
             0.371519274376417, 0.346428510973046, 0.274538712500162, 0.165495361560806,
             0.027777777777778]
        ),
    ),
    11: (
        np.array(
            [-1.0, -0.934001430408059, -0.784483473663144, -0.565235326996205,
             -0.295758135586939, 0.0, 0.295758135586939, 0.565235326996205,
             0.784483473663144, 0.934001430408059, 1.0]
        ),
        np.array(
            [0.018181818181818, 0.109612273266995, 0.187169881780305, 0.248048104264028,
             0.286879124779008, 0.300217595455691, 0.286879124779008, 0.248048104264028,
             0.187169881780305, 0.109612273266995, 0.018181818181818]
        ),
    ),
}


def lobatto_layers(num_layers: int, thickness: float) -> Tuple[np.ndarray, np.ndarray]:
    """Through-thickness Gauss-Lobatto coordinates and weights.

    Lobatto rules include the surface points, where yielding starts first.
    Returns (z, w) with z in [-h/2, h/2] and sum(w) = h.
    """
    if num_layers not in _LOBATTO_RULES:
        raise ValueError(f"num_layers must be one of {sorted(_LOBATTO_RULES)}")
    points, weights = _LOBATTO_RULES[num_layers]
    return 0.5 * thickness * points, 0.5 * thickness * weights
