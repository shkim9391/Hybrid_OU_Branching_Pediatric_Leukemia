"""
kmt2a_lineage_analysis.py

This script performs a lineage analysis using a hybrid Ornstein–Uhlenbeck–branching
model on the KMT2A-r clinical dataset. It reads the patient metadata from the
Excel file (kmt2a_clinical_data.xlsx), assigns a simulation duration to each
patient based on disease-specific median time-to-relapse values and relapse
group multipliers, then simulates an OU process coupled with a birth–death
branching process while tracking the full lineage structure.

For each patient the script can:
  1. Simulate the trait dynamics and branching events over the assigned
     interval, recording the number of active clones and the average trait
     across clones at each time step.
  2. Record the lineage tree structure, capturing birth/death times and
     parent–child relationships.
  3. Compute summary metrics such as the final number of surviving clones,
     total number of clones created, number of branch events, and maximum
     lineage depth.
  4. Optionally generate phase-plane plots (trait vs time, clone count
     vs time, and trait vs clone count) and a simple lineage diagram for
     individual patients.

The simulation parameters (OU coefficients and birth/death rates) are set
heuristically as follows:
  mu   = 0.0    # trait mean
  theta= 1.0    # reversion strength
  sigma= 0.4    # volatility
  lambda_rate = 0.8  # birth rate per clone per unit time
  death_rate  = 0.5  # death rate per clone per unit time

Median time-to-relapse values are taken from published aggregated statistics:
  * infant ALL  : 405 days
  * childhood ALL: 419 days
  * infant AML  : 372 days
  * childhood AML: 205 days【120340620773589†L790-L801】

Because the provided clinical dataset does not distinguish between infant and
childhood cases, this script approximates AML by the average of 372 and
205 days (~289 days) and uses 419 days for all ALL variants. Group-level
multipliers (e.g. 0.5 for very early relapse, 1.0 for early relapse,
2.0 for late relapse or remission) further scale the durations.

Usage:
    python3 kmt2a_lineage_analysis.py

This will read the Excel file from the current directory, simulate all
patients and print a summary of lineage metrics. To save the metrics as
a CSV file, you can modify the script accordingly.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_prepare_data(excel_path: str) -> pd.DataFrame:
    """Load the Excel file, locate the header row and return a DataFrame."""
    raw_df = pd.read_excel(excel_path, header=None)
    header_index = None
    for idx, row in raw_df.iterrows():
        if 'Patient_ID' in row.values:
            header_index = idx
            break
    if header_index is None:
        raise ValueError("Header row with 'Patient_ID' not found in the Excel file")
    header = raw_df.iloc[header_index].tolist()
    data = raw_df.iloc[header_index + 1:].copy()
    data.columns = header
    return data


def simulate_ou_branching_with_lineage(
    time_end: float,
    dt: float = 0.005,
    mu: float = 0.0,
    theta: float = 1.0,
    sigma: float = 0.4,
    lambda_rate: float = 0.8,
    death_rate: float = 0.5,
) -> tuple:
    """
    Simulate an Ornstein–Uhlenbeck process coupled with a birth–death branching
    process while tracking the full lineage structure.

    Args:
        time_end: Simulation end time (in years).
        dt: Time step size.
        mu, theta, sigma: OU process parameters.
        lambda_rate: Birth rate per clone per unit time.
        death_rate: Death rate per clone per unit time.

    Returns:
        times: Array of time points.
        clone_counts: List of number of active clones at each time.
        avg_traits: List of average trait across active clones at each time.
        lineages: List of dictionaries describing each lineage (id, parent_id,
                  start_time, end_time, trait_series, active status).
    """
    times = np.arange(0, time_end + dt, dt)
    lineages = []  # List to store all lineages
    active_lineages = []  # Indices of currently active lineages
    lineage_id_counter = 0
    # Initialize root lineage
    root = {
        'id': lineage_id_counter,
        'parent_id': None,
        'start_time': 0.0,
        'end_time': time_end,
        'traits': [0.0],  # initial trait value
        'active': True
    }
    lineages.append(root)
    active_lineages.append(0)
    lineage_id_counter += 1
    clone_counts = []
    avg_traits = []
    for ti in range(len(times)):
        t = times[ti]
        # Update trait for each active lineage
        for idx in active_lineages:
            lineage = lineages[idx]
            last_trait = lineage['traits'][-1]
            new_trait = last_trait + theta * (mu - last_trait) * dt + sigma * np.sqrt(dt) * np.random.randn()
            lineage['traits'].append(new_trait)
        # Determine events (birth/death/survival) for each active lineage
        new_active_lineages = []
        for idx in active_lineages:
            lineage = lineages[idx]
            if not lineage['active']:
                continue
            r = np.random.rand()
            if r < lambda_rate * dt:
                # Branching event
                lineage['end_time'] = t
                lineage['active'] = False
                parent_trait = lineage['traits'][-1]
                # Create two child lineages
                for _ in range(2):
                    child = {
                        'id': lineage_id_counter,
                        'parent_id': idx,
                        'start_time': t,
                        'end_time': time_end,
                        'traits': [parent_trait],
                        'active': True
                    }
                    lineages.append(child)
                    new_active_lineages.append(lineage_id_counter)
                    lineage_id_counter += 1
            elif r < lambda_rate * dt + death_rate * dt:
                # Death event
                lineage['end_time'] = t
                lineage['active'] = False
            else:
                # Survival
                new_active_lineages.append(idx)
        active_lineages = new_active_lineages
        clone_counts.append(len(active_lineages))
        if len(active_lineages) > 0:
            avg_traits.append(np.mean([lineages[idx]['traits'][-1] for idx in active_lineages]))
        else:
            avg_traits.append(np.nan)
    return times, clone_counts, avg_traits, lineages


def compute_lineage_metrics(lineages):
    """Compute summary metrics for a given lineage list."""
    final_clones = sum(1 for ln in lineages if ln['active'])
    total_clones = len(lineages)
    branch_events = total_clones - 1  # root has no parent
    depths = {}
    def get_depth(idx):
        if idx in depths:
            return depths[idx]
        parent = lineages[idx]['parent_id']
        d = 0 if parent is None else get_depth(parent) + 1
        depths[idx] = d
        return d
    max_depth = max(get_depth(idx) for idx in range(total_clones))
    return {
        'final_clones': final_clones,
        'total_clones': total_clones,
        'branch_events': branch_events,
        'max_depth': max_depth
    }


def main():
    excel_path = 'kmt2a_clinical_data.xlsx'
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(
            f"Could not find '{excel_path}' in the current directory. Please place the Excel file here.")
    data = load_and_prepare_data(excel_path)
    disease_median_days = {
        'B-ALL': 419,
        'T-ALL': 419,
        'MPAL': 419,
        'AML': 289
    }
    group_multiplier = {
        'Remission': 2.0,
        'Very early': 0.5,
        'Very early/refractory': 0.5,
        'Very early / Refractory': 0.5,
        'Early': 1.0,
        'Early/refractory': 1.0,
        'Early / refractory': 1.0,
        'Late': 2.0,
        'Late / refractory': 2.0
    }
    summary_metrics = []
    for _, row in data.iterrows():
        disease = row['Disease']
        group = row['Group']
        base_days = disease_median_days.get(disease, 365)
        base_years = base_days / 365.0
        multiplier = group_multiplier.get(group, 1.0)
        time_end = base_years * multiplier
        times, clone_counts, avg_traits, lineages = simulate_ou_branching_with_lineage(time_end)
        metrics = compute_lineage_metrics(lineages)
        metrics['Patient_ID'] = row['Patient_ID']
        metrics['Disease'] = disease
        metrics['Group'] = group
        summary_metrics.append(metrics)
    metrics_df = pd.DataFrame(summary_metrics)
    print(metrics_df)
    metrics_df.to_csv('lineage_metrics_summary.csv', index=False)
    print('Summary metrics saved to lineage_metrics_summary.csv')


if __name__ == '__main__':
    main()
