#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 — Comparative Dynamics of OU–Branching vs. Brownian vs. Markov Jump Models

Simulates and compares:
  1) Hybrid Ornstein–Uhlenbeck (OU) process with branching events (multiple lineages)
  2) Brownian motion (neutral diffusion)
  3) Markov jump process (discrete +/- 1 steps with Poisson-like event rate)

Outputs (written to --outdir):
  - Fig1_OU_Branching_simulation.csv
  - Fig1_Brownian_simulation.csv
  - Fig1_Markov_simulation.csv
  - Figure1_OU_vs_Brownian_vs_Markov.png

Notes on "variance over time":
  - A single trajectory has no across-sample variance at each timepoint.
  - To make panel D meaningful, set --n_reps > 1 to estimate variance across replicates
    at each timepoint for each process.

Usage:
  python figure1_simulation.py
  python figure1_simulation.py --outdir results --seed 42 --n_reps 200
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# CONFIG
# ==========================================================
@dataclass(frozen=True)
class Params:
    T: float = 10.0
    dt: float = 0.01
    theta: float = 1.0
    sigma: float = 0.25
    lam: float = 0.3
    mu: float = 0.0
    x0: float = 0.0
    n_lineages: int = 10


def make_time_grid(T: float, dt: float) -> np.ndarray:
    n_steps = int(round(T / dt))
    return np.linspace(0.0, T, n_steps + 1)


def branching_event_prob(lam: float, dt: float) -> float:
    # P(event in dt) for a Poisson process with rate lam
    return 1.0 - np.exp(-lam * dt)


# ==========================================================
# SIMULATORS
# ==========================================================
def simulate_ou_branching(
    tgrid: np.ndarray,
    theta: float,
    mu: float,
    sigma: float,
    lam: float,
    x0: float,
    n_lineages: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Hybrid OU–Branching process using Euler–Maruyama integration.
    Each lineage evolves continuously; branching spawns a new lineage inheriting
    the parent's current state.
    """
    dt = float(tgrid[1] - tgrid[0])
    p_branch = branching_event_prob(lam, dt)

    lineages: List[Dict] = [{"id": 0, "x": float(x0), "hist_t": [0.0], "hist_x": [float(x0)]}]
    next_id = 1

    for k in range(1, len(tgrid)):
        t_now = float(tgrid[k])
        current_count = len(lineages)

        # Update existing lineages only (new lineages added later in this time step)
        for i in range(current_count):
            L = lineages[i]
            dW = rng.normal(0.0, np.sqrt(dt))

            x_old = float(L["x"])
            x_new = x_old + theta * (mu - x_old) * dt + sigma * dW

            L["x"] = x_new
            L["hist_t"].append(t_now)
            L["hist_x"].append(x_new)

            # Branching event (cap at n_lineages)
            if (len(lineages) < n_lineages) and (rng.random() < p_branch):
                lineages.append(
                    {"id": next_id, "x": x_new, "hist_t": [t_now], "hist_x": [x_new]}
                )
                next_id += 1

    records: List[Tuple[float, int, float]] = []
    for L in lineages:
        lid = int(L["id"])
        for tt, xx in zip(L["hist_t"], L["hist_x"]):
            records.append((float(tt), lid, float(xx)))

    return pd.DataFrame(records, columns=["t", "lineage", "x"])


def simulate_brownian(
    tgrid: np.ndarray,
    sigma: float,
    x0: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Neutral Brownian diffusion (single trajectory)."""
    dt = float(tgrid[1] - tgrid[0])
    x = np.zeros_like(tgrid, dtype=float)
    x[0] = float(x0)
    for k in range(1, len(tgrid)):
        dW = rng.normal(0.0, np.sqrt(dt))
        x[k] = x[k - 1] + sigma * dW
    return pd.DataFrame({"t": tgrid, "x": x})


def simulate_markov_jump(
    tgrid: np.ndarray,
    lam: float,
    x0: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Discrete jump process: at each dt, jump +/- 1 with Poisson-like event probability."""
    dt = float(tgrid[1] - tgrid[0])
    p_jump = branching_event_prob(lam, dt)

    x = np.zeros_like(tgrid, dtype=float)
    x[0] = float(x0)
    for k in range(1, len(tgrid)):
        if rng.random() < p_jump:
            x[k] = x[k - 1] + float(rng.choice([-1, 1]))
        else:
            x[k] = x[k - 1]
    return pd.DataFrame({"t": tgrid, "x": x})


# ==========================================================
# VARIANCE ESTIMATION (ACROSS REPLICATES)
# ==========================================================
def variance_over_time_from_reps(paths: np.ndarray) -> np.ndarray:
    """
    paths: shape (n_reps, n_time)
    returns: variance at each timepoint across reps
    """
    if paths.ndim != 2:
        raise ValueError("paths must be 2D: (n_reps, n_time)")
    return np.var(paths, axis=0, ddof=1) if paths.shape[0] > 1 else np.zeros(paths.shape[1])


def run_replicates(
    params: Params,
    tgrid: np.ndarray,
    n_reps: int,
    seed: int,
) -> Dict[str, pd.Series]:
    """
    Returns variance-over-time series for each process.
    For OU–Branching, we compute variance across *all lineage values per replicate*
    at each timepoint (then across replicates). This provides a stable signal for panel D.
    """
    # Independent RNG streams to avoid coupling between processes
    base = np.random.SeedSequence(seed)
    ss_ou, ss_bm, ss_mk = base.spawn(3)

    n_time = len(tgrid)

    # Brownian + Markov: store 1 value per time per replicate
    bm_paths = np.zeros((n_reps, n_time), dtype=float)
    mk_paths = np.zeros((n_reps, n_time), dtype=float)

    # OU–Branching: we summarize each replicate at each time by taking variance across lineages
    # (since OU–Branching naturally generates an ensemble).
    ou_lineage_var = np.zeros((n_reps, n_time), dtype=float)

    for r in range(n_reps):
        rng_ou = np.random.default_rng(ss_ou.spawn(1)[0].generate_state(1, dtype=np.uint32)[0] + r)
        rng_bm = np.random.default_rng(ss_bm.spawn(1)[0].generate_state(1, dtype=np.uint32)[0] + r)
        rng_mk = np.random.default_rng(ss_mk.spawn(1)[0].generate_state(1, dtype=np.uint32)[0] + r)

        df_ou = simulate_ou_branching(
            tgrid=tgrid,
            theta=params.theta,
            mu=params.mu,
            sigma=params.sigma,
            lam=params.lam,
            x0=params.x0,
            n_lineages=params.n_lineages,
            rng=rng_ou,
        )

        # variance across lineages at each time in this replicate
        ou_var_t = df_ou.groupby("t")["x"].var().reindex(tgrid, fill_value=0.0).to_numpy()
        ou_lineage_var[r, :] = ou_var_t

        df_bm = simulate_brownian(tgrid, params.sigma, params.x0, rng_bm)
        bm_paths[r, :] = df_bm["x"].to_numpy()

        df_mk = simulate_markov_jump(tgrid, params.lam, params.x0, rng_mk)
        mk_paths[r, :] = df_mk["x"].to_numpy()

    # Panel D series
    var_ou = pd.Series(np.mean(ou_lineage_var, axis=0), index=tgrid, name="OU–Branching (across lineages)")
    var_bm = pd.Series(variance_over_time_from_reps(bm_paths), index=tgrid, name="Brownian (across reps)")
    var_mk = pd.Series(variance_over_time_from_reps(mk_paths), index=tgrid, name="Markov (across reps)")

    return {"ou": var_ou, "brownian": var_bm, "markov": var_mk}


# ==========================================================
# PLOTTING
# ==========================================================
def plot_figure(
    df_ou: pd.DataFrame,
    df_brown: pd.DataFrame,
    df_markov: pd.DataFrame,
    var_ou: pd.Series,
    var_brown: pd.Series,
    var_markov: pd.Series,
    mu: float,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    # (A) OU–Branching trajectories
    for _, sub in df_ou.groupby("lineage"):
        axes[0, 0].plot(sub["t"], sub["x"], alpha=0.9, linewidth=1.0)
    axes[0, 0].axhline(mu, linestyle="--", linewidth=1.0)
    axes[0, 0].set_title("A. OU–Branching Trajectories")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Trait X(t)")

    # (B) Brownian motion
    axes[0, 1].plot(df_brown["t"], df_brown["x"], linewidth=1.2)
    axes[0, 1].set_title("B. Brownian Motion (Neutral Drift)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Trait X(t)")

    # (C) Markov jumps
    axes[1, 0].plot(df_markov["t"], df_markov["x"], drawstyle="steps-post", linewidth=1.2)
    axes[1, 0].set_title("C. Markov Jump Process")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("Discrete State")

    # (D) Variance comparison
    axes[1, 1].plot(var_ou.index, var_ou.values, label=str(var_ou.name), linewidth=1.5)
    axes[1, 1].plot(var_brown.index, var_brown.values, label=str(var_brown.name), linewidth=1.5)
    axes[1, 1].plot(var_markov.index, var_markov.values, label=str(var_markov.name), linewidth=1.5)
    axes[1, 1].set_title("D. Variance over Time")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Variance")
    axes[1, 1].legend(loc="upper left", frameon=True, edgecolor="gray", framealpha=1, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.show()


# ==========================================================
# CLI / MAIN
# ==========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate OU–Branching vs Brownian vs Markov for Figure 1.")
    p.add_argument("--outdir", type=str, default=".", help="Output directory (default: current).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--n_reps", type=int, default=1, help="Replicates for variance estimation (default: 1).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    params = Params()
    tgrid = make_time_grid(params.T, params.dt)

    # Separate RNG streams for the primary trajectories plotted in panels A–C
    ss = np.random.SeedSequence(args.seed)
    ss_ou, ss_bm, ss_mk = ss.spawn(3)
    rng_ou = np.random.default_rng(ss_ou)
    rng_bm = np.random.default_rng(ss_bm)
    rng_mk = np.random.default_rng(ss_mk)

    # Run one set of trajectories for the figure panels
    df_ou = simulate_ou_branching(
        tgrid=tgrid,
        theta=params.theta,
        mu=params.mu,
        sigma=params.sigma,
        lam=params.lam,
        x0=params.x0,
        n_lineages=params.n_lineages,
        rng=rng_ou,
    )
    df_brown = simulate_brownian(tgrid, params.sigma, params.x0, rng_bm)
    df_markov = simulate_markov_jump(tgrid, params.lam, params.x0, rng_mk)

    # Save trajectory data
    df_ou.to_csv(outdir / "Fig1_OU_Branching_simulation.csv", index=False)
    df_brown.to_csv(outdir / "Fig1_Brownian_simulation.csv", index=False)
    df_markov.to_csv(outdir / "Fig1_Markov_simulation.csv", index=False)

    # Variance series (meaningful if n_reps > 1)
    var_series = run_replicates(params, tgrid, n_reps=int(args.n_reps), seed=int(args.seed))
    var_ou = var_series["ou"]
    var_brown = var_series["brownian"]
    var_markov = var_series["markov"]

    # Plot + save figure
    out_png = outdir / "Figure1_OU_vs_Brownian_vs_Markov.png"
    plot_figure(df_ou, df_brown, df_markov, var_ou, var_brown, var_markov, params.mu, out_png)

    print("Simulation complete! Files saved to:", outdir)
    print("-", out_png.name)
    print("- Fig1_OU_Branching_simulation.csv")
    print("- Fig1_Brownian_simulation.csv")
    print("- Fig1_Markov_simulation.csv")


if __name__ == "__main__":
    main()
