"""
kmt2a_phase_plane_lineage_plots.py

Generate phase‑plane plots and a lineage diagram for a given patient
from the KMT2A‑r clinical dataset using the hybrid OU–branching model.

This script imports the helper functions `load_and_prepare_data` and
`simulate_ou_branching_with_lineage` from the `kmt2a_lineage_analysis`
module.  It then selects a patient by `Patient_ID`, determines the
simulation duration based on disease and relapse group, runs the
simulation and creates:

  1. A set of three phase‑plane plots: average trait vs. time, clone
     count vs. time, and trait vs. clone count.
  2. A simple lineage diagram showing clone lifespans and branch points.

Adjust the `PATIENT_ID` constant below to analyse a different patient.

Usage:
    python3 kmt2a_phase_plane_lineage_plots.py

Ensure that `kmt2a_clinical_data.xlsx` and `kmt2a_lineage_analysis.py`
are in the same directory as this script.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from kmt2a_lineage_analysis import (
    load_and_prepare_data,
    simulate_ou_branching_with_lineage,
)


# Choose the patient ID to analyse
PATIENT_ID = 'P15'

# Median relapse times in days by disease.  AML uses ~289 days as the
# average of infant (372) and childhood (205) AML relapse times【120340620773589†L790-L801】.
disease_median_days = {
    'B-ALL': 419,
    'T-ALL': 419,
    'MPAL': 419,
    'AML': 289,
}

# Relapse group multipliers
group_multiplier = {
    'Remission': 2.0,
    'Very early': 0.5,
    'Very early/refractory': 0.5,
    'Very early / Refractory': 0.5,
    'Early': 1.0,
    'Early/refractory': 1.0,
    'Early / refractory': 1.0,
    'Late': 2.0,
    'Late / refractory': 2.0,
}


def main():
    # Verify data file exists
    excel_path = 'kmt2a_clinical_data.xlsx'
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(
            f"Missing input file: {excel_path}. Please place it in the current directory."
        )
    # Load clinical data
    data = load_and_prepare_data(excel_path)
    # Select the row for the chosen patient
    if PATIENT_ID not in data['Patient_ID'].values:
        raise ValueError(f"Patient ID '{PATIENT_ID}' not found in the dataset")
    patient_row = data[data['Patient_ID'] == PATIENT_ID].iloc[0]
    disease = patient_row['Disease']
    group = patient_row['Group']
    # Determine simulation duration (in years)
    base_years = disease_median_days.get(disease, 365) / 365.0
    duration = base_years * group_multiplier.get(group, 1.0)
    # Run the OU–branching simulation with lineage tracking
    times, clone_counts, avg_traits, lineages = simulate_ou_branching_with_lineage(duration)
    # Plot phase‑plane figures
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(times, avg_traits)
    axs[0].set_title(f"A. Average trait vs Time ({PATIENT_ID})")
    axs[0].set_xlabel('Time (years)')
    axs[0].set_ylabel('Average trait')
    axs[1].plot(times, clone_counts)
    axs[1].set_title(f"B. Clone count vs Time ({PATIENT_ID})")
    axs[1].set_xlabel('Time (years)')
    axs[1].set_ylabel('Number of clones')
    valid = ~np.isnan(avg_traits)
    axs[2].scatter(np.array(clone_counts)[valid], np.array(avg_traits)[valid], s=10)
    axs[2].set_title(f"C. Trait vs Clone count ({PATIENT_ID})")
    axs[2].set_xlabel('Clone count')
    axs[2].set_ylabel('Average trait')
    plt.tight_layout()
    plt.savefig(f'fig3abc_phase_plane_patient_{PATIENT_ID}.png')
    plt.show()
    # Plot lineage diagram
    # Assign y‑positions to each lineage for plotting
    y_positions = {ln['id']: i for i, ln in enumerate(lineages)}
    # Normalise the initial traits to colour lineages
    traits0 = [ln['traits'][0] for ln in lineages]
    # Use finite numbers to avoid division by zero
    min_t = np.min(traits0)
    max_t = np.max(traits0) if np.max(traits0) != min_t else min_t + 1
    norm_traits = [(t - min_t) / (max_t - min_t) for t in traits0]
    cmap = plt.cm.viridis
    fig2, ax2 = plt.subplots(figsize=(10, max(len(lineages) * 0.2, 4)))
    for ln in lineages:
        y = y_positions[ln['id']]
        start, end = ln['start_time'], ln['end_time']
        color = cmap(norm_traits[ln['id']])
        ax2.hlines(y, start, end, colors=[color], linewidth=2)
        if ln['parent_id'] is not None:
            parent_y = y_positions[ln['parent_id']]
            ax2.vlines(start, min(y, parent_y), max(y, parent_y), colors='grey', linestyles='dotted')
    ax2.set_title(f"D. Lineage diagram ({PATIENT_ID})")
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('Lineage index')
    plt.tight_layout()
    plt.savefig(f'fig3d_lineage_patient_{PATIENT_ID}.png')
    plt.show()


if __name__ == '__main__':
    main()
