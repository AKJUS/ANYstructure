import numpy as np
from scipy.sparse import diags, bmat, csc_matrix, eye
from scipy.sparse.linalg import eigsh

# Simple beam buckling with 10 elements
N = 20
L = 30.0
dx = L / N
EI = 3.3e10
P = 1.0e8

# Beam element K and Kg
def beam_element(L_e):
    k = EI / L_e**3 * np.array([
        [12, 6*L_e, -12, 6*L_e],
        [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
        [-12, -6*L_e, 12, -6*L_e],
        [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2]
    ])
    kg = 1.0 / (30 * L_e) * np.array([
        [36, 3*L_e, -36, 3*L_e],
        [3*L_e, 4*L_e**2, -3*L_e, -L_e**2],
        [-36, -3*L_e, 36, -3*L_e],
        [3*L_e, -L_e**2, -3*L_e, 4*L_e**2]
    ])
    return k, kg

K = np.zeros((2*N+2, 2*N+2))
Kg = np.zeros((2*N+2, 2*N+2))

for i in range(N):
    k, kg = beam_element(dx)
    idx = [2*i, 2*i+1, 2*i+2, 2*i+3]
    for r in range(4):
        for c in range(4):
            K[idx[r], idx[c]] += k[r,c]
            Kg[idx[r], idx[c]] += kg[r,c]

# Rigid body modes
# 1. Translation: w = 1, theta = 0
r1 = np.zeros(2*N+2)
r1[0::2] = 1.0

# 2. Rotation: w = x - L/2, theta = 1
r2 = np.zeros(2*N+2)
x_vals = np.linspace(0, L, N+1)
r2[0::2] = x_vals - L/2
r2[1::2] = 1.0

R = np.column_stack((r1, r2))
R, _ = np.linalg.qr(R)

# Projection matrix P = I - R R^T
Proj = np.eye(2*N+2) - R @ R.T

# Solve (P K P) x = lambda (P Kg P) x
K_proj = Proj @ K @ Proj
Kg_proj = Proj @ Kg @ Proj

# Add shift to nullspace so matrix is invertible
K_inv = np.linalg.pinv(K_proj)

w, v = np.linalg.eig(K_inv @ Kg_proj)
w = 1.0 / w[w > 1e-10]
w = np.sort(w)
print("Projected buckling load factors:", w[:3] * P / 1e8)

# Now check pinned-pinned
K_pinned = K[1:-1, 1:-1]
Kg_pinned = Kg[1:-1, 1:-1]
K_pinned = np.delete(K_pinned, -2, axis=0)
K_pinned = np.delete(K_pinned, -2, axis=1)
Kg_pinned = np.delete(Kg_pinned, -2, axis=0)
Kg_pinned = np.delete(Kg_pinned, -2, axis=1)
w2, v2 = np.linalg.eig(np.linalg.inv(K_pinned) @ Kg_pinned)
w2 = 1.0 / w2[w2 > 1e-10]
w2 = np.sort(w2)
print("Pinned-pinned load factors:", w2[:3] * P / 1e8)
