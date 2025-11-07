#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 11:50:05 2025

@author: seung-hwan.kim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 — Comparative Dynamics of OU–Branching vs. Markov/Brownian Models
Author: [Your Name]
Description:
    Simulates and compares Ornstein–Uhlenbeck (OU)–Branching, Brownian, 
    and Markov processes to illustrate pediatric (OU-constrained) 
    vs. adult (Markov/Brownian) tumor evolution.

Outputs:
    - Fig1_OU_Branching_simulation_py.csv
    - Fig1_Brownian_simulation_py.csv
    - Fig1_Markov_simulation_py.csv
    - Figure1_OU_vs_Brownian_vs_Markov_py.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# PARAMETERS
# ==========================================================
T = 10.0           # total simulation time
dt = 0.01          # time step
theta = 1.0        # OU selection strength
sigma = 0.25       # diffusion coefficient
lam = 0.3          # branching (or jump) rate
mu = 0.0           # equilibrium mean
x0 = 0.0           # initial state
n_lineages = 10    # max number of lineages (for OU–Branching)
rng = np.random.default_rng(42)

# ==========================================================
# OU–Branching Simulation
# ==========================================================
def simulate_ou_branching(T, dt, theta, mu, sigma, lam, x0, n_lineages, rng):
    """Hybrid OU–Branching process using Euler–Maruyama integration."""
    n_steps = int(T / dt)
    tgrid = np.linspace(0, T, n_steps + 1)
    lineages = [{"id": 0, "x": x0, "hist_t": [0.0], "hist_x": [x0]}]
    next_id = 1

    for k in range(1, n_steps + 1):
        t_now = tgrid[k]
        current_count = len(lineages)

        for i in range(current_count):
            L = lineages[i]
            dW = rng.normal(0.0, np.sqrt(dt))
            # OU drift + diffusion
            x_new = L["x"] + theta * (mu - L["x"]) * dt + sigma * dW
            L["x"] = x_new
            L["hist_t"].append(t_now)
            L["hist_x"].append(x_new)
            # Branching event
            if rng.random() < (1 - np.exp(-lam * dt)):
                if len(lineages) < n_lineages:
                    lineages.append({"id": next_id, "x": x_new, 
                                     "hist_t": [t_now], "hist_x": [x_new]})
                    next_id += 1

    records = [(tt, L["id"], xx)
               for L in lineages
               for tt, xx in zip(L["hist_t"], L["hist_x"])]
    return pd.DataFrame(records, columns=["t", "lineage", "x"])

# ==========================================================
# Brownian Motion Simulation
# ==========================================================
def simulate_brownian(T, dt, sigma, x0, rng):
    """Neutral Brownian diffusion with variance increasing linearly in time."""
    n_steps = int(T / dt)
    tgrid = np.linspace(0, T, n_steps + 1)
    x = np.zeros_like(tgrid)
    x[0] = x0
    for k in range(1, len(tgrid)):
        dW = rng.normal(0.0, np.sqrt(dt))
        x[k] = x[k-1] + sigma * dW
    return pd.DataFrame({"t": tgrid, "x": x})

# ==========================================================
# Markov Jump Process Simulation
# ==========================================================
def simulate_markov(T, dt, lam, x0, rng):
    """Discrete continuous-time Markov chain (birth–death style)."""
    n_steps = int(T / dt)
    tgrid = np.linspace(0, T, n_steps + 1)
    x = np.zeros_like(tgrid)
    x[0] = x0
    for k in range(1, len(tgrid)):
        if rng.random() < (1 - np.exp(-lam * dt)):
            x[k] = x[k-1] + rng.choice([-1, 1])
        else:
            x[k] = x[k-1]
    return pd.DataFrame({"t": tgrid, "x": x})

# ==========================================================
# RUN SIMULATIONS
# ==========================================================
df_ou = simulate_ou_branching(T, dt, theta, mu, sigma, lam, x0, n_lineages, rng)
df_brown = simulate_brownian(T, dt, sigma, x0, rng)
df_markov = simulate_markov(T, dt, lam, x0, rng)

# ==========================================================
# VARIANCE OVER TIME
# ==========================================================
var_ou = df_ou.groupby("t")["x"].var().fillna(0)
var_brown = df_brown.groupby("t")["x"].var().fillna(0)
var_markov = df_markov.groupby("t")["x"].var().fillna(0)

# ==========================================================
# SAVE DATA
# ==========================================================
df_ou.to_csv("Fig1_OU_Branching_simulation_py.csv", index=False)
df_brown.to_csv("Fig1_Brownian_simulation_py.csv", index=False)
df_markov.to_csv("Fig1_Markov_simulation_py.csv", index=False)

# ==========================================================
# PLOT COMPOSITE FIGURE
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

# (A) OU–Branching
for lid, sub in df_ou.groupby("lineage"):
    axes[0,0].plot(sub["t"], sub["x"], alpha=0.9, linewidth=1.0)
axes[0,0].axhline(mu, linestyle="--", linewidth=1.0)
axes[0,0].set_title("A. OU–Branching Trajectories")
axes[0,0].set_xlabel("Time")
axes[0,0].set_ylabel("Trait X(t)")

# (B) Brownian motion
axes[0,1].plot(df_brown["t"], df_brown["x"], color="gray", linewidth=1.2)
axes[0,1].set_title("B. Brownian Motion (Neutral Drift)")
axes[0,1].set_xlabel("Time")
axes[0,1].set_ylabel("Trait X(t)")

# (C) Markov jumps
axes[1,0].plot(df_markov["t"], df_markov["x"], drawstyle="steps-post", 
               linewidth=1.2, color="tab:red")
axes[1,0].set_title("C. Markov Jump Process")
axes[1,0].set_xlabel("Time")
axes[1,0].set_ylabel("Discrete State")

# (D) Variance comparison
axes[1,1].plot(var_ou.index, var_ou.values, label="OU–Branching", linewidth=1.5)
axes[1,1].plot(var_brown.index, var_brown.values, label="Brownian", linewidth=1.5)
axes[1,1].plot(var_markov.index, var_markov.values, label="Markov", linewidth=1.5)
axes[1,1].set_title("D. Variance over Time")
axes[1,1].set_xlabel("Time")
axes[1,1].set_ylabel("Variance of X(t)")
axes[1,1].legend(
    loc='upper left',
    frameon=True,
    #facecolor='lightgray',
    edgecolor='gray',
    framealpha=1,
    fontsize=9
)

plt.tight_layout()
plt.savefig("Figure1_OU_vs_Brownian_vs_Markov_py.png", bbox_inches="tight", dpi=300)
plt.show()

print("Simulation complete! Files saved:")
print("- Figure1_OU_vs_Brownian_vs_Markov_py.png")
print("- Fig1_OU_Branching_simulation_py.csv")
print("- Fig1_Brownian_simulation_py.csv")
print("- Fig1_Markov_simulation_py.csv")