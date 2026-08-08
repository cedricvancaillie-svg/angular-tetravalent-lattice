import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

def run_passive_scaling():
    print("--- Rigorous Passive Scaling Test ---")
    L = 40.0
    t_final = 2.0
    A = np.array([[0,0,0,1], [1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)
    I = np.eye(4)
    Gamma = np.diag([1, 0, -1, 0])
    for eps in [0.1, 0.05, 0.025, 0.0125]:
        n_sites = int(L / eps)
        x = (np.arange(n_sites) - n_sites // 2) * eps
        S_init = np.zeros((4, n_sites))
        S_init[0, :] = np.exp(-(x**2) / 2.0)
        S_init /= np.sum(S_init)
        S_disc = S_init.copy()
        for _ in range(int(t_final / eps)):
            S_tilde = (1 - eps) * S_disc + eps * (A @ S_disc)
            S_disc = np.copy(S_tilde)
            S_disc[0] = np.roll(S_tilde[0], 1)
            S_disc[2] = np.roll(S_tilde[2], -1)
        k_freq = np.fft.fftfreq(n_sites, d=eps) * 2 * np.pi
        S_fft = np.fft.fft(S_init, axis=1)
        S_pde_fft = np.zeros_like(S_fft, dtype=complex)
        for j in range(n_sites):
            M = (A - I) - 1j * k_freq[j] * Gamma
            S_pde_fft[:, j] = expm(M * t_final) @ S_fft[:, j]
        S_pde = np.real(np.fft.ifft(S_pde_fft, axis=1))
        err = np.sum(np.abs(S_disc - S_pde))
        print(f"eps = {eps:6.4f} | L1 Error = {err:7.5f}")

def run_active_scaling():
    print("\n--- Active Bulk Scaling Test ---")
    A = np.array([[0,0,0,1], [1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)
    print(f"{'eps':>7} | {'Support Speed (c=1)':>20} | {'Ridge Speed (c=0.5)':>20}")
    print("-" * 55)
    for eps in [0.2, 0.1, 0.05, 0.025]:
        n_sites = int(40 / eps)
        steps = int(10 / eps)
        S = np.zeros((4, n_sites, n_sites))
        center = n_sites // 2
        S[0, center, center] = 1.0
        S[1] = S[0].copy()
        S[2] = S[0].copy()
        S[3] = S[0].copy()
        for _ in range(steps):
            S_tilde = S + eps * np.tensordot(A, S, axes=([1], [0]))
            S_new = np.empty_like(S_tilde)
            S_new[0] = np.roll(S_tilde[0], shift=1, axis=1)
            S_new[2] = np.roll(S_tilde[2], shift=-1, axis=1)
            S_new[1] = np.roll(S_tilde[1], shift=1, axis=0)
            S_new[3] = np.roll(S_tilde[3], shift=-1, axis=0)
            S = S_new
        rho = np.sum(S, axis=0)

        # 1. Absolute Support Edge (Information Light-Cone)
        profile_raw = rho[center, center:]
        support_idx = np.max(np.nonzero(profile_raw > 1e-7)[0])
        speed_support = support_idx / steps

        # 2. Macroscopic Density Ridge (Weyl Projection)
        # Take a multi-row strip average to kill transverse parity oscillations
        strip = rho[center-2 : center+3, center:]
        profile_strip = np.mean(strip, axis=0)

        # Apply a mild convolution to smooth longitudinal grid artifacts
        window = np.ones(5) / 5.0
        profile_smooth = np.convolve(profile_strip, window, mode='same')

        # Relative outer-threshold detector
        search_start = int(support_idx * 0.3)
        outer = profile_smooth[search_start:support_idx]
        thresh = 0.08 * outer.max()  # 8% of the local maximum
        above = np.where(outer > thresh)[0]

        ridge_idx = search_start + above[-1] if len(above) > 0 else support_idx // 2
        speed_ridge = ridge_idx / steps
        print(f"{eps:7.3f} | {speed_support:20.3f} | {speed_ridge:20.3f}")

def run_2d_walk(n_sites=200, steps=150, eps=0.05, mode='bulk', disorder=0.0):
    S = np.zeros((4, n_sites, n_sites))
    center = n_sites // 2
    y, x = np.ogrid[:n_sites, :n_sites]
    if mode == 'bulk':
        S[0] = np.exp(-((x - center)**2 + (y - center)**2) / 4.0)
    else:
        S[0] = np.exp(-((x - center)**2 + (y - 20)**2) / 4.0)
    S[1] = S[0].copy()
    S[2] = S[0].copy()
    S[3] = S[0].copy()
    A = np.array([[0,0,0,1], [1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)
    mask = np.zeros((n_sites, n_sites))
    mask[:, center:] = 1.0
    np.random.seed(42)
    noise = np.random.rand(n_sites, n_sites) - 0.5
    eps_grid = eps * (1.0 + disorder * noise)
    for _ in range(steps):
        if mode == 'bulk':
            S_tilde = S + eps * np.tensordot(A, S, axes=([1], [0]))
        elif mode == 'edge_disordered':
            S_rot_R = np.tensordot(A, S, axes=([1], [0]))
            S_rot_L = np.tensordot(A.T, S, axes=([1], [0]))
            S_tilde = S + eps_grid * (S_rot_R * mask + S_rot_L * (1 - mask))
        S_new = np.empty_like(S_tilde)
        S_new[0] = np.roll(S_tilde[0], shift=1, axis=1)
        S_new[2] = np.roll(S_tilde[2], shift=-1, axis=1)
        S_new[1] = np.roll(S_tilde[1], shift=1, axis=0)
        S_new[3] = np.roll(S_tilde[3], shift=-1, axis=0)
        S = S_new
        S /= np.sum(S)
    rho = np.sum(S, axis=0)
    if mode == 'edge_disordered':
        y_profile = np.sum(rho, axis=1)
        x_profile = np.sum(rho, axis=0)
        com_y = np.average(np.arange(n_sites), weights=y_profile)
        x_dist_sq = (np.arange(n_sites) - center)**2
        std_x = np.sqrt(np.average(x_dist_sq, weights=x_profile))
        v_y = (com_y - 20) / steps
        print(f"Disorder: {disorder:.1f} | Edge Velocity v_y: {v_y:.3f} | Localization std_x: {std_x:.2f}")
    return rho

if __name__ == "__main__":
    run_passive_scaling()
    run_active_scaling()
    rho_bulk = run_2d_walk(mode='bulk')
    plt.figure(figsize=(6, 6))
    plt.imshow(rho_bulk, cmap='magma', origin='lower')
    plt.title("Unnormalized Bulk: Expanding Diamond Wavefront")
    plt.colorbar(label="Density (Normalized for Vis)")
    plt.savefig("wave_ring.png", dpi=300, bbox_inches='tight')
    print("\n--- Systematic Edge Mode Robustness Sweep ---")
    for delta in [0.0, 0.4, 0.8]:
        rho_edge = run_2d_walk(mode='edge_disordered', steps=150, disorder=delta)
    plt.figure(figsize=(6, 6))
    plt.imshow(rho_edge, cmap='magma', origin='lower')
    plt.title("Robust Chiral Edge Current (80% Spatial Disorder)")
    plt.axvline(x=100, color='white', linestyle='--', alpha=0.5)
    plt.savefig("edge_state_disordered.png", dpi=300, bbox_inches='tight')
