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
  - (optional) Figure1_OU_vs_Brownian_vs_Markov.pdf

Notes on "variance over time":
  - A single trajectory has no across-sample variance at each timepoint.
  - To make panel D meaningful, set --n_reps > 1 to estimate variance across replicates.

Usage:
  python figure1_simulation.py
  python figure1_simulation.py --outdir results --seed 42 --n_reps 200 --save_pdf
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# DEFAULTS / FILE NAMES
# ==========================================================
OU_CSV = "Fig1_OU_Branching_simulation.csv"
BM_CSV = "Fig1_Brownian_simulation.csv"
MK_CSV = "Fig1_Markov_simulation.csv"
FIG_PNG = "Figure1_OU_vs_Brownian_vs_Markov.png"
FIG_PDF = "Figure1_OU_vs_Brownian_vs_Markov.pdf"


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


def event_prob_poisson(rate: float, dt: float) -> float:
    """P(event in dt) for a Poisson process with constant rate."""
    return 1.0 - np.exp(-rate * dt)


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
    Hybrid OU–Branching using Euler–Maruyama integration.
    Each lineage evolves continuously; branching spawns a new lineage inheriting
    the parent's current state.
    """
    if len(tgrid) < 2:
        raise ValueError("tgrid must have at least two points.")
    dt = float(tgrid[1] - tgrid[0])
    p_branch = event_prob_poisson(lam, dt)

    lineages: List[Dict[str, Any]] = [
        {"id": 0, "x": float(x0), "hist_t": [0.0], "hist_x": [float(x0)]}
    ]
    next_id = 1

    for k in range(1, len(tgrid)):
        t_now = float(tgrid[k])
        current_count = len(lineages)

        # Update existing lineages only (new lineages appended after this point)
        for i in range(current_count):
            L = lineages[i]
            dW = rng.normal(0.0, np.sqrt(dt))

            x_old = float(L["x"])
            x_new = x_old + theta * (mu - x_old) * dt + sigma * dW

            L["x"] = x_new
            L["hist_t"].append(t_now)
            L["hist_x"].append(x_new)

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
    """Discrete jump process: with prob ~ Poisson(lam), jump +/-1, else stay."""
    dt = float(tgrid[1] - tgrid[0])
    p_jump = event_prob_poisson(lam, dt)

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
    returns: variance at each timepoint across replicates
    """
    if paths.ndim != 2:
        raise ValueError("paths must be 2D: (n_reps, n_time)")
    if paths.shape[0] <= 1:
        return np.zeros(paths.shape[1], dtype=float)
    return np.var(paths, axis=0, ddof=1)


def run_replicates(
    params: Params,
    tgrid: np.ndarray,
    n_reps: int,
    seed: int,
) -> Dict[str, pd.Series]:
    """
    Returns variance-over-time series for each process.

    - Brownian + Markov: variance is computed across replicate paths.
    - OU–Branching: for each replicate, compute variance across lineages at each time;
      then average that lineage-variance curve across replicates for a stable signal.
    """
    if n_reps < 1:
        raise ValueError("--n_reps must be >= 1")

    base = np.random.SeedSequence(seed)
    ss_ou, ss_bm, ss_mk = base.spawn(3)

    # Spawn per-replicate seeds cleanly (no integer hacks)
    ou_reps = ss_ou.spawn(n_reps)
    bm_reps = ss_bm.spawn(n_reps)
    mk_reps = ss_mk.spawn(n_reps)

    n_time = len(tgrid)
    bm_paths = np.zeros((n_reps, n_time), dtype=float)
    mk_paths = np.zeros((n_reps, n_time), dtype=float)
    ou_lineage_var = np.zeros((n_reps, n_time), dtype=float)

    for r in range(n_reps):
        rng_ou = np.random.default_rng(ou_reps[r])
        rng_bm = np.random.default_rng(bm_reps[r])
        rng_mk = np.random.default_rng(mk_reps[r])

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

        # Variance across lineages at each time (reindex to full tgrid)
        ou_var_t = (
            df_ou.groupby("t")["x"]
            .var()
            .reindex(tgrid, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        ou_lineage_var[r, :] = ou_var_t

        df_bm = simulate_brownian(tgrid, params.sigma, params.x0, rng_bm)
        bm_paths[r, :] = df_bm["x"].to_numpy(dtype=float)

        df_mk = simulate_markov_jump(tgrid, params.lam, params.x0, rng_mk)
        mk_paths[r, :] = df_mk["x"].to_numpy(dtype=float)

    var_ou = pd.Series(
        np.mean(ou_lineage_var, axis=0), index=tgrid, name="OU–Branching (across lineages)"
    )
    var_bm = pd.Series(
        variance_over_time_from_reps(bm_paths), index=tgrid, name="Brownian (across reps)"
    )
    var_mk = pd.Series(
        variance_over_time_from_reps(mk_paths), index=tgrid, name="Markov (across reps)"
    )

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
    dpi: int = 300,
    save_pdf: bool = False,
    out_pdf: Path | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=dpi)

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
    fig.savefig(out_png, bbox_inches="tight", dpi=dpi)

    if save_pdf:
        if out_pdf is None:
            out_pdf = out_png.with_suffix(".pdf")
        fig.savefig(out_pdf, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


# ==========================================================
# CLI / MAIN
# ==========================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Simulate OU–Branching vs Brownian vs Markov for Figure 1."
    )
    p.add_argument("--outdir", type=str, default=".", help="Output directory (default: current).")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    p.add_argument("--n_reps", type=int, default=1, help="Replicates for variance estimation (default: 1).")
    p.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300).")
    p.add_argument("--save_pdf", action="store_true", help="Also save a PDF version of the figure.")
    p.add_argument("--no_show", action="store_true", help="Do not display the figure window.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    params = Params()
    tgrid = make_time_grid(params.T, params.dt)

    # Independent RNG streams for the primary trajectories plotted in panels A–C
    ss = np.random.SeedSequence(args.seed)
    ss_ou, ss_bm, ss_mk = ss.spawn(3)
    rng_ou = np.random.default_rng(ss_ou)
    rng_bm = np.random.default_rng(ss_bm)
    rng_mk = np.random.default_rng(ss_mk)

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
    df_ou.to_csv(outdir / OU_CSV, index=False)
    df_brown.to_csv(outdir / BM_CSV, index=False)
    df_markov.to_csv(outdir / MK_CSV, index=False)

    # Variance series (meaningful if n_reps > 1)
    var_series = run_replicates(params, tgrid, n_reps=int(args.n_reps), seed=int(args.seed))
    var_ou = var_series["ou"]
    var_brown = var_series["brownian"]
    var_markov = var_series["markov"]

    # Plot + save figure
    out_png = outdir / FIG_PNG
    out_pdf = outdir / FIG_PDF
    plot_figure(
        df_ou=df_ou,
        df_brown=df_brown,
        df_markov=df_markov,
        var_ou=var_ou,
        var_brown=var_brown,
        var_markov=var_markov,
        mu=params.mu,
        out_png=out_png,
        dpi=int(args.dpi),
        save_pdf=bool(args.save_pdf),
        out_pdf=out_pdf if args.save_pdf else None,
        show=(not args.no_show),
    )

    print("Simulation complete! Files saved to:", outdir)
    print("-", FIG_PNG)
    if args.save_pdf:
        print("-", FIG_PDF)
    print("-", OU_CSV)
    print("-", BM_CSV)
    print("-", MK_CSV)


if __name__ == "__main__":
    main()
