#!/usr/bin/env python3
"""
ATLC Simulator v1.4
===================

Standalone implementation of the master state-transition equation
from the paper:

    "A Discrete Angular Tetravalent Lattice and a Candidate Pathway
     to the Schrödinger Equation"
    Cédric Van Caillie, July 2026

This script:
  - Defines the full ATLC_Simulator class
  - Runs a reproducible 15 000-step simulation (seed=42)
  - Prints the statistics reported in Section 5 / Appendix B
  - Generates the three figures used in the paper

Usage:
    python ATLC_Simulator_v1.4.py
"""

import numpy as np
import matplotlib.pyplot as plt


def normalize(S):
    """Project the state vector onto the probability simplex."""
    S = np.clip(S, 0.0, 1.0)
    total = np.sum(S)
    return S / total if total > 0 else np.array([0.25, 0.25, 0.25, 0.25])


class ATLC_Simulator:
    """Angular Tetravalent Lattice Cellular simulator."""

    def __init__(self, twin_mode=False):
        self.lambda_base = 0.32
        self.kappa_elastic = 0.021
        self.kappa_lattice = 0.018
        self.r_min = 0.8
        self.a_fermat = 0.5
        self.phi0 = 1.0
        self.alpha = 0.15
        self.omega = 0.05
        self.D_state = np.array([1.1, 0.9, 1.8, 1.3])
        self.A = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0]], dtype=float)
        self.L_pos = self.A.copy()
        self.L_neg = np.array([[0, 1, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0],
                               [1, 0, 0, 0]], dtype=float)
        self.S_Still = np.array([0.43, 0.32, 0.06, 0.19])
        self.S_Still /= np.sum(self.S_Still)
        self.theta = 0.0
        self.t = 0.0
        self.twin_theta = 0.0
        self.cos_psi = self.cos_xi = self.cos_omega = 1.0

        if not twin_mode:
            self.twin_sim = ATLC_Simulator(twin_mode=True)
            self.twin_sim.theta = np.pi
        else:
            self.twin_sim = None

    def get_lambda_adaptive(self, S):
        return self.lambda_base * (1.0 + 0.8 * S[1] - 0.6 * S[2])

    def compute_elastic_strain_probability(self, lambda_ad, dom_idx):
        D = self.D_state[dom_idx] if dom_idx < 4 else np.mean(self.D_state)
        return 100 * (1 - (lambda_ad + self.kappa_elastic * D) / 1.0)

    def get_coherence_bias(self):
        r = self.a_fermat * np.sqrt(np.abs(self.theta) + 1e-8)
        phi = (self.phi0
               * np.exp(-self.alpha * r)
               * np.cos(self.theta)
               * np.cos(self.omega * self.t))
        return phi * np.array([0.15, 0.35, 0.30, 0.20])

    def get_geometry_correction(self, S):
        r_k = np.abs(S) + 0.01
        elastic_factor = (np.sin(np.pi / 2 - np.abs(S)) ** 4) * self.D_state
        return self.kappa_lattice * np.sum((1 - self.r_min / r_k) * elastic_factor)

    def step(self, S, sigma, epsilon):
        coherence_bias = self.get_coherence_bias()
        geometry_term = self.get_geometry_correction(S)

        self.twin_theta += sigma * np.pi / 2 + 0.015 * np.sqrt(np.abs(self.twin_theta) + 1)
        twin_phase_factor = np.exp(1j * (sigma + 1) * np.pi / 2)
        S_twin = normalize(0.92 * S + 0.08 * np.real(np.dot(self.A, S) * twin_phase_factor))

        orange_dom = S[2]
        L_loop = np.dot(
            self.L_pos if np.random.rand() < 0.55 + 0.45 * orange_dom else self.L_neg,
            S
        )

        phase_factor = np.exp(1j * sigma * np.pi / 2)
        rot_complex = np.dot(self.A, S) * phase_factor
        rot = np.real(rot_complex)

        core = sigma * rot * (1 + epsilon) + 0.35 * L_loop
        r180 = 0.5 * S_twin
        lambda_ad = self.get_lambda_adaptive(S)
        still_term = -lambda_ad * self.S_Still

        r_fermat = self.a_fermat * np.sqrt(np.abs(self.theta) + 1e-8)
        elastic_factor = (np.sin(np.pi / 2 - np.abs(S)) ** 4) * self.D_state
        elastic_term = self.kappa_elastic * elastic_factor * (1 + 0.25 * r_fermat)
        lattice_term = (self.kappa_lattice
                        * (1 - self.r_min / (np.abs(S) + 0.01))
                        * elastic_factor * self.D_state)
        boundary_term = 0.15 * (S_twin - S)
        entropy_term = epsilon * 0.05 * np.ones(4)

        corrections = (still_term + elastic_term + lattice_term + boundary_term
                       + entropy_term + coherence_bias + geometry_term)

        S_new = S + core + r180 + corrections

        self.theta += sigma * np.pi / 2 + 0.015 * np.sqrt(np.abs(self.theta) + 1)
        self.t += 1
        self.cos_psi = np.cos(self.theta * 0.7)
        self.cos_xi = np.cos(self.theta * 1.1)
        self.cos_omega = np.cos(self.theta * 0.9)

        return normalize(S_new)

    def run_simulation(self, n_steps=500, initial_S=None, seed=None):
        if seed is not None:
            np.random.seed(seed)

        S = np.array(initial_S) if initial_S is not None else np.array([0.25] * 4)
        history = [S.copy()]
        risks = []
        thetas = [self.theta]

        for _ in range(n_steps):
            sigma = np.random.choice([-1, 0, 1], p=[0.28, 0.44, 0.28])
            epsilon = np.random.normal(0, 0.07)
            S = self.step(S, sigma, epsilon)
            history.append(S.copy())
            thetas.append(self.theta)
            dom_idx = np.argmax(S)
            lambda_ad = self.get_lambda_adaptive(S)
            risks.append(self.compute_elastic_strain_probability(lambda_ad, dom_idx))

        return np.array(history), np.array(risks), np.array(thetas)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running 15 000-step simulation with seed=42 ...")
    sim = ATLC_Simulator()
    history, risks, thetas = sim.run_simulation(n_steps=15000, seed=42)
    times = np.arange(len(history))

    # Statistics reported in the paper
    means = np.mean(history, axis=0)
    print("\n========== NUMBERS FOR THE PAPER ==========")
    print(f"Mean state dwells (S1/S2/S3/S4): "
          f"{means[0]:.3f} / {means[1]:.3f} / {means[2]:.3f} / {means[3]:.3f}")
    print(f"Mean elastic strain probability: {np.mean(risks):.2f} %")
    print(f"Time spent with S4 > 0.3: {100 * np.mean(history[:, 3] > 0.3):.2f} %")
    print("==========================================\n")

    # Colours matching the paper: Blue, Yellow, Orange, Red
    state_colors = ['#1f77b4', '#ffcc00', '#ff7f0e', '#d62728']

    # ----- Figure 1: State evolution -----
    plt.figure(figsize=(12, 6))
    plt.stackplot(times[::50],
                  history[::50, 0], history[::50, 1],
                  history[::50, 2], history[::50, 3],
                  labels=['S1 (Blue)', 'S2 (Yellow)', 'S3 (Orange)', 'S4 (Red)'],
                  colors=state_colors, alpha=0.85)
    plt.title('State Evolution over 15,000 steps')
    plt.xlabel('Time step')
    plt.ylabel('Proportion')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('state_evolution_15000.png', dpi=300)
    print("Saved: state_evolution_15000.png")
    plt.close()

    # ----- Figure 2: Elastic rupture risk -----
    plt.figure(figsize=(12, 4))
    plt.plot(times[1::50], risks[::50], color='#d62728', lw=1.5)
    plt.title('Elastic Rupture Risk % over 15,000 steps')
    plt.xlabel('Time step')
    plt.ylabel('Risk %')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('risk_evolution_15000.png', dpi=300)
    print("Saved: risk_evolution_15000.png")
    plt.close()

    # ----- Figure 3: Fermat-spiral phase growth -----
    phase = np.sqrt(np.abs(thetas))
    plt.figure(figsize=(12, 4))
    plt.plot(times[::50], phase[::50], color='#2ca02c', lw=1.5)
    plt.title(r'Fermat Spiral Phase Growth ($\sqrt{|\theta|}$)')
    plt.xlabel('Time step')
    plt.ylabel(r'$\sqrt{|\theta|}$')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase_growth_15000.png', dpi=300)
    print("Saved: phase_growth_15000.png")
    plt.close()

    print("\nAll figures generated successfully.")
