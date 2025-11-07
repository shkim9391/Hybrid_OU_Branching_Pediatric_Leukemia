#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  1 20:35:22 2025

@author: seung-hwan.kim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3 – Patient-level OU–Branching simulation (P15-style)
Generates a 4-panel composite figure:
  A. Average trait vs time
  B. Clone count vs time
  C. Trait vs clone count
  D. Lineage diagram (colored lifespans + dotted parent connectors)

No external files required. Uses a reproducible RNG seed.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ----------------------------
# Configuration (P15-style)
# ----------------------------
SEED   = 13         # tuned to show mid-run branching + a few survivors
MU     = 0.0
THETA  = 1.0
SIGMA  = 0.4
LAMBDA = 0.8        # branching rate
DELTA  = 0.5        # death rate
ETA    = 0.05       # SD of trait perturbation on branching (N(0, ETA^2))
DT     = 0.005
T      = 2.3        # ~2.3 years like your reference
MAX_CLONES = 2000   # safety cap for plotting

# ----------------------------
# Simulation data structures
# ----------------------------
class Clone:
    __slots__ = ("id","parent","birth_t","death_t","x_hist_t","x_hist")
    def __init__(self, cid, parent, t0, x0):
        self.id = cid
        self.parent = parent
        self.birth_t = t0
        self.death_t = None
        self.x_hist_t = [t0]
        self.x_hist   = [x0]

def simulate_ou_branching_with_lineage(mu=MU, theta=THETA, sigma=SIGMA,
                                       lam=LAMBDA, delta=DELTA, eta=ETA,
                                       dt=DT, T=T, seed=SEED,
                                       max_clones=MAX_CLONES):
    """Run a hybrid OU–branching + birth–death simulation and return
    time series + lineage records suitable for plotting Figure 3."""
    rng = np.random.default_rng(seed)
    steps = int(T / dt)

    # init founder
    clones = []
    founder = Clone(0, None, 0.0, 0.0)
    clones.append(founder)
    active = {0: founder}
    next_id = 1

    times = np.linspace(0, T, steps + 1)
    avg_trait    = np.zeros(steps + 1)   # mean over extant clones only
    clone_count  = np.zeros(steps + 1)   # number of extant clones
    branch_times = []                    # for dotted markers in panel A

    for k in range(1, steps + 1):
        t = times[k]
        for cid in list(active.keys()):
            c = active[cid]

            # OU step: X_{t+dt} = X_t + θ(μ−X_t)dt + σ√dt * ξ
            dW    = rng.normal(0.0, np.sqrt(dt))
            x_new = c.x_hist[-1] + theta * (mu - c.x_hist[-1]) * dt + sigma * dW
            c.x_hist.append(x_new)
            c.x_hist_t.append(t)

            # Branching event (thinning of Poisson process)
            if len(clones) < max_clones and rng.random() < (1.0 - np.exp(-lam * dt)):
                child_x = x_new + rng.normal(0.0, eta)  # N(0, ETA^2)
                child   = Clone(next_id, cid, t, child_x)
                clones.append(child)
                active[next_id] = child
                next_id += 1
                branch_times.append(t)

            # Death event
            if rng.random() < (1.0 - np.exp(-delta * dt)):
                c.death_t = t
                active.pop(cid, None)

        # extant-only averaging
        if active:
            avg_trait[k] = np.mean([active[cid].x_hist[-1] for cid in active])
        else:
            avg_trait[k] = avg_trait[k - 1]

        clone_count[k] = len(active)

    # finalize lifespans for remaining clones
    for c in clones:
        if c.death_t is None:
            c.death_t = T

    return times, avg_trait, clone_count, branch_times, clones

# ----------------------------
# Plotting
# ----------------------------
def plot_figure3(times, avg_trait, clone_count, branch_times, clones,
                 outfile="Figure3_Combined_P15.png"):
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs  = GridSpec(2, 3, height_ratios=[1, 1.15], figure=fig)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, :])

    # --- A. Average trait vs time (extant-only)
    axA.plot(times, avg_trait, linewidth=1.6)
    for bt in branch_times:
        axA.axvline(bt, color='grey', linestyle='dotted', linewidth=0.6, alpha=0.7)
    axA.set_title("A. Average trait vs Time (P15)")
    axA.set_xlabel("Time (years)")
    axA.set_ylabel("Average trait")

    # --- B. Clone count vs time
    axB.plot(times, clone_count, linewidth=1.6)
    axB.set_title("B. Clone count vs Time (P15)")
    axB.set_xlabel("Time (years)")
    axB.set_ylabel("Number of clones")

    # --- C. Trait vs clone count (clone count on x-axis)
    valid = ~np.isnan(avg_trait)
    axC.scatter(np.array(clone_count)[valid], np.array(avg_trait)[valid], s=10, alpha=0.75)
    axC.set_title("C. Trait vs Clone count (P15)")
    axC.set_xlabel("Clone count")
    axC.set_ylabel("Average trait")

    # --- D. Lineage diagram (colored lifespans + dotted parent connectors)
    # Safe color mapping by lineage ID (IDs may not be 0..N-1)
    traits0     = [c.x_hist[0] for c in clones]
    min_t       = float(np.min(traits0))
    max_t       = float(np.max(traits0)) if float(np.max(traits0)) != min_t else (min_t + 1.0)
    norms       = [(t - min_t) / (max_t - min_t) for t in traits0]
    cmap        = plt.cm.viridis
    id_to_norm  = {c.id: norms[i] for i, c in enumerate(clones)}
    y_positions = {c.id: i for i, c in enumerate(clones)}

    for c in clones:
        y = y_positions[c.id]
        axD.hlines(y, c.birth_t, c.death_t, colors=[cmap(id_to_norm[c.id])], linewidth=2)
        if c.parent is not None:
            py = y_positions[c.parent]
            axD.vlines(c.birth_t, min(y, py), max(y, py),
                       colors='grey', linestyles='dotted', linewidth=1.0)

    axD.set_title("D. Lineage diagram (P15)")
    axD.set_xlabel("Time (years)")
    axD.set_ylabel("Lineage index")
    axD.set_xlim(0, times[-1])
    axD.set_ylim(-0.5, max(y_positions.values()) + 0.5)

    plt.tight_layout(h_pad=2.0)
    plt.savefig(outfile, bbox_inches="tight", dpi=300)
    print(f"Saved: {outfile}")

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    times, avg_trait, clone_count, branch_times, clones = simulate_ou_branching_with_lineage()
    plot_figure3(times, avg_trait, clone_count, branch_times, clones)