import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def run_massive_scaling():
    print("--- Emergence of Mass: Wave Speed vs Inertia ---")
    A = np.array([[0,0,0,1], [1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)

    # =========================================================
    # 1. HIGH-RESOLUTION RUN -> accurate table numbers
    # =========================================================
    eps = 0.025
    steps = 160
    n_sites = 400
    center = n_sites // 2
    kappas = np.linspace(0.0, 0.15, 7)
    measured_speeds = []

    print(f"{'Mass (kappa)':>12} | {'Support Speed':>15} | {'Ridge Speed':>15}")
    print("-" * 50)

    for kappa in kappas:
        S = np.zeros((4, n_sites, n_sites))
        S[0, center, center] = 1.0
        S[1, center, center] = 1.0
        S[2, center, center] = 1.0
        S[3, center, center] = 1.0

        for _ in range(steps):
            S_tilde = S + eps * np.tensordot(A, S, axes=([1], [0]))
            S_new = np.empty_like(S_tilde)
            S_new[0] = (1 - kappa) * np.roll(S_tilde[0], shift=1, axis=1) + kappa * S_tilde[0]
            S_new[2] = (1 - kappa) * np.roll(S_tilde[2], shift=-1, axis=1) + kappa * S_tilde[2]
            S_new[1] = (1 - kappa) * np.roll(S_tilde[1], shift=1, axis=0) + kappa * S_tilde[1]
            S_new[3] = (1 - kappa) * np.roll(S_tilde[3], shift=-1, axis=0) + kappa * S_tilde[3]
            S = S_new

        rho = np.sum(S, axis=0)

        # Support edge
        profile = rho[center, center:]
        support_idx = np.max(np.nonzero(profile > 1e-12)[0]) if np.any(profile > 1e-12) else 0
        speed_support = support_idx / steps

        # Cumulative-mass ridge (70 % of total mass)
        y, x = np.ogrid[:n_sites, :n_sites]
        r = np.sqrt((x - center)**2 + (y - center)**2)
        max_r = int(steps * 0.90)
        flat_r = r.ravel()
        flat_rho = rho.ravel()
        mask = flat_r <= max_r
        order = np.argsort(flat_r[mask])
        sorted_r = flat_r[mask][order]
        sorted_rho = flat_rho[mask][order]
        cum_mass = np.cumsum(sorted_rho)
        total_mass = cum_mass[-1] if len(cum_mass) > 0 else 1.0
        target = 0.70 * total_mass
        idx = np.searchsorted(cum_mass, target)
        if idx >= len(sorted_r):
            idx = len(sorted_r) - 1
        ridge_idx = sorted_r[idx]
        speed_ridge = ridge_idx / steps
        measured_speeds.append(speed_ridge)

        print(f"{kappa:12.3f} | {speed_support:15.3f} | {speed_ridge:15.3f}")

    slope, intercept, r_value, _, _ = linregress(kappas, measured_speeds)
    print("\n--- Linear Regression Analysis ---")
    print(f"Equation: v = {intercept:.3f} + ({slope:.3f}) * kappa")
    print(f"R-squared: {r_value**2:.3f}")

    # =========================================================
    # 2. LOW-RESOLUTION RUN -> clear visualisation
    # =========================================================
    eps_vis = 0.1
    steps_vis = 80
    n_sites_vis = 200
    center_vis = n_sites_vis // 2
    profiles_to_plot = {}

    for kappa in [0.0, 0.10]:
        S = np.zeros((4, n_sites_vis, n_sites_vis))
        S[0, center_vis, center_vis] = 1.0
        S[1, center_vis, center_vis] = 1.0
        S[2, center_vis, center_vis] = 1.0
        S[3, center_vis, center_vis] = 1.0

        for _ in range(steps_vis):
            S_tilde = S + eps_vis * np.tensordot(A, S, axes=([1], [0]))
            S_new = np.empty_like(S_tilde)
            S_new[0] = (1 - kappa) * np.roll(S_tilde[0], shift=1, axis=1) + kappa * S_tilde[0]
            S_new[2] = (1 - kappa) * np.roll(S_tilde[2], shift=-1, axis=1) + kappa * S_tilde[2]
            S_new[1] = (1 - kappa) * np.roll(S_tilde[1], shift=1, axis=0) + kappa * S_tilde[1]
            S_new[3] = (1 - kappa) * np.roll(S_tilde[3], shift=-1, axis=0) + kappa * S_tilde[3]
            S = S_new

        profiles_to_plot[kappa] = np.sum(S, axis=0)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    ax1 = axes[0]
    vmax0 = np.percentile(profiles_to_plot[0.0], 99.5)
    im1 = ax1.imshow(profiles_to_plot[0.0], cmap='magma', origin='lower', vmax=vmax0)
    ax1.set_title("Massless Weyl Wave (kappa = 0.0)\nExpands at c=1")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    vmax10 = np.percentile(profiles_to_plot[0.10], 99.5)
    im2 = ax2.imshow(profiles_to_plot[0.10], cmap='magma', origin='lower', vmax=vmax10)
    ax2.set_title("Wave packet slowed and concentrated\nby geometric inertia (kappa = 0.10)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("emergence_of_mass.png", dpi=300, bbox_inches='tight')
    print("\nSaved visual comparison to 'emergence_of_mass.png'")

if __name__ == "__main__":
    run_massive_scaling()
