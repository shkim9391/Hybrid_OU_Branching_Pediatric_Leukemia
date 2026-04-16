# KMT2A_OU_Branching_Leukemia

Code, processed cohort tables, and reproducible figure-generation workflows for cohort-level and simulation-based analysis of hybrid Ornstein-Uhlenbeck (OU) / OU-Branching dynamics in pediatric KMT2A-rearranged leukemia.

## Overview

This repository accompanies a bioRxiv preprint presenting a computational and cohort-level extension of the hybrid OU-Branching framework for pediatric leukemia. The project focuses on clinical modeling, patient-specific simulations, lineage-aware summaries, and comparison of alternative stochastic baselines, including OU, Brownian, and Markov-style formulations.

The repository includes:

- figure-generation scripts for the main manuscript figures,
- processed clinical and lineage-related input tables,
- mutation and clone-fraction summaries used in phase-space and branching analyses,
- supplementary tables packaged for manuscript submission.

This repository is organized for manuscript reproducibility rather than as a general-purpose software package.

## Preprint context

**Version 1 | Posted November 2025 | bioRxiv Preprint**

This preprint represents a computational and cohort-level extension of the hybrid OU-Branching framework introduced in the author's manuscript under review at *Frontiers in Oncology* ("A hybrid Ornstein-Uhlenbeck-branching framework unifies microbial and pediatric tumor evolution", manuscript ID 1727973).

The *Frontiers in Oncology* paper emphasizes experimental validation and cross-domain analogies between microbial long-term evolution experiments and pediatric tumor evolution, with primary emphasis on biological interpretation. In contrast, this repository and associated preprint focus on:

- clinical modeling,
- patient-specific simulations,
- lineage-aware summaries,
- phase-plane visualization,
- stochastic baseline comparison in pediatric KMT2A-rearranged leukemia.

No text, figures, or data are intended to duplicate the separate in-review manuscript; this repository is meant to support the distinct computational preprint and its reproducible analyses.

## Repository contents

The current repository includes the following main files.

### Main figure scripts

- `Figure1_OU_vs_Brownian_vs_Markov.py`  
  Generates Figure 1, comparing OU-based dynamics against Brownian and Markov-style baselines. This script is intended to illustrate the conceptual and empirical differences between constrained mean-reverting dynamics and alternative stochastic formulations.

- `Figure2_kmt2a_clinical_data_analysis_by_ou_branching.py`  
  Generates Figure 2 from the KMT2A clinical dataset using the OU-Branching framework. This script likely performs patient-level or cohort-level modeling on the processed clinical table and summarizes the resulting evolutionary patterns.

- `Figure3_Combined_P15.py`  
  Generates Figure 3 using the P15 processed lineage / clone summary inputs. This script appears to integrate multiple data summaries into a combined patient-level visualization.

- `Figure4_P15_OU_Branching_Combined.py`  
  Generates Figure 4, combining P15-specific outputs with the OU-Branching analysis pipeline. This figure likely synthesizes lineage, clone-fraction, and mutation-VAF information into a joint evolutionary interpretation.

### Additional analysis scripts

- `kmt2a_lineage_analysis.py`  
  Performs lineage-focused analysis from processed inputs. Typical uses may include summarizing clone transitions, computing lineage-derived metrics, or generating intermediate outputs used by later figure scripts.

- `kmt2a_phase_plane_lineage_plots.py`  
  Produces lineage-aware phase-plane plots from processed patient or clone-level summaries. These plots are useful for visualizing trajectory structure under the fitted or summarized evolutionary model.

### Input and processed data files

- `kmt2a_clinical_data.xlsx`  
  Main processed clinical dataset used for cohort-level analysis.

- `P15_OU_clone_fractions.csv`  
  Clone-fraction summaries for patient/sample P15 used in branching and lineage analyses.

- `P15_OU_mutation_VAFs.csv`  
  Variant allele frequency (VAF) summaries for patient/sample P15 used in phase-plane and OU-Branching analyses.

- `lineage_metrics_summary.csv`  
  Summary table of lineage-level metrics used for downstream plotting, interpretation, or manuscript reporting.

- `Supplementary_Tables.xlsx`  
  Supplementary manuscript tables packaged for submission and reference.

## Suggested repository layout

A simple organization for public release would be:

```text
repo_root/
  Figure1_OU_vs_Brownian_vs_Markov.py
  Figure2_kmt2a_clinical_data_analysis_by_ou_branching.py
  Figure3_Combined_P15.py
  Figure4_P15_OU_Branching_Combined.py
  kmt2a_lineage_analysis.py
  kmt2a_phase_plane_lineage_plots.py
  kmt2a_clinical_data.xlsx
  P15_OU_clone_fractions.csv
  P15_OU_mutation_VAFs.csv
  lineage_metrics_summary.csv
  Supplementary_Tables.xlsx
  README.md

## Scientific scope

This project studies pediatric KMT2A-rearranged leukemia using stochastic evolutionary models that represent disease progression as a combination of constrained drift and branching structure.

Conceptually:
	•	OU dynamics model continuous, mean-reverting evolution around latent disease states.
	•	Brownian models provide a diffusion-only comparison without stabilizing pull.
	•	Markov-style models provide a discrete-state comparator.
	•	OU-Branching models extend the OU framework to allow divergence across clone or lineage structure.

The overall goal is not only predictive modeling, but also biologically interpretable characterization of leukemia progression, lineage diversification, and patient-specific evolutionary constraints.

File descriptions in more detail

Figure1_OU_vs_Brownian_vs_Markov.py

This script compares competing stochastic views of disease evolution. It is likely intended to:
	•	simulate or summarize dynamics under OU, Brownian, and Markov models,
	•	show how constrained mean reversion differs from unconstrained diffusion,
	•	motivate why OU-based structure may better capture clinically meaningful leukemia dynamics.

Typical output:
	•	comparison panels,
	•	conceptual summary plots,
	•	benchmark-style trajectory visualizations.

Figure2_kmt2a_clinical_data_analysis_by_ou_branching.py

This script performs the main clinical data analysis using the KMT2A cohort table. It likely:
	•	reads the processed clinical spreadsheet,
	•	constructs patient/timepoint-level inputs,
	•	applies OU-Branching summaries or model-derived calculations,
	•	exports the main figure for cohort-level interpretation.

Typical output:
	•	patient-level or cohort-level trend summaries,
	•	grouped model-derived statistics,
	•	manuscript-ready Figure 2 panels.

Figure3_Combined_P15.py

This script focuses on a combined analysis of patient/sample P15. It likely integrates:
	•	clone fractions,
	•	mutation VAF summaries,
	•	lineage-derived metrics,
	•	OU-based or OU-Branching visualization layers.

Typical output:
	•	combined patient-specific panels,
	•	intermediate-to-late disease trajectory summaries,
	•	integrated visual representation of P15 evolution.

Figure4_P15_OU_Branching_Combined.py

This script appears to generate the final P15-specific OU-Branching synthesis figure. It likely combines the major processed P15 inputs into a single summary display.

Typical output:
	•	multi-panel patient-level figure,
	•	joint clone/VAF/evolution overlays,
	•	final manuscript-ready composite visualization.

kmt2a_lineage_analysis.py

This script likely computes lineage-level summaries used by the figure scripts. Depending on implementation, it may:
	•	derive lineage metrics,
	•	summarize transitions or branch membership,
	•	calculate clone dominance or diversity-related measures,
	•	export reusable tables for phase-plane or combined plots.

kmt2a_phase_plane_lineage_plots.py

This script likely visualizes the state space of lineage progression. Typical operations may include:
	•	plotting coordinates in an inferred phase plane,
	•	overlaying lineage identity or branch assignments,
	•	annotating directional progression or equilibrium structure,
	•	producing patient-specific or cohort-level trajectory plots.

Data files and their roles

kmt2a_clinical_data.xlsx

Primary processed cohort table. This is the main clinical input for the leukemia analysis workflow.

Typical contents may include:
	•	patient identifiers,
	•	disease phase or timepoint variables,
	•	longitudinal measurements,
	•	clinical annotations used in the model-based analyses.

P15_OU_clone_fractions.csv

Processed clone-fraction table for the P15 case.

Typical uses:
	•	clone abundance summaries,
	•	branch-aware fraction analysis,
	•	patient-specific visualization in Figures 3 and 4.

P15_OU_mutation_VAFs.csv

Processed mutation-VAF table for the P15 case.

Typical uses:
	•	mutation-level trajectory summaries,
	•	phase-plane analysis,
	•	clone/VAF integration in patient-specific figures.

lineage_metrics_summary.csv

Derived lineage-summary table, likely used as an intermediate analytical product.

Typical uses:
	•	branch/lineage metric reporting,
	•	linkage between lineage analysis and plotting scripts,
	•	supporting figure annotations.

Supplementary_Tables.xlsx

Supplementary manuscript tables corresponding to the preprint or journal-facing submission package.

Typical workflow

A typical order of use from the repository root may be:

python kmt2a_lineage_analysis.py
python kmt2a_phase_plane_lineage_plots.py

python Figure1_OU_vs_Brownian_vs_Markov.py
python Figure2_kmt2a_clinical_data_analysis_by_ou_branching.py
python Figure3_Combined_P15.py
python Figure4_P15_OU_Branching_Combined.py

A practical interpretation of this workflow is:
	1.	compute lineage-level summaries,
	2.	generate phase-plane lineage plots,
	3.	generate the conceptual comparison figure,
	4.	generate the cohort-level clinical OU-Branching figure,
	5.	generate the P15 combined figure,
	6.	generate the final P15 OU-Branching synthesis figure.

Typical Python loading examples

Load the processed clinical dataset

import pandas as pd

clinical = pd.read_excel("kmt2a_clinical_data.xlsx")
print(clinical.head())

Load P15 clone fractions

import pandas as pd

clone_frac = pd.read_csv("P15_OU_clone_fractions.csv")
print(clone_frac.head())

Load P15 mutation VAF summaries

import pandas as pd

vafs = pd.read_csv("P15_OU_mutation_VAFs.csv")
print(vafs.head())

Load lineage summary metrics

import pandas as pd

lineage_metrics = pd.read_csv("lineage_metrics_summary.csv")
print(lineage_metrics.head())

Software environment

The repository is intended to run in Python 3 with the standard scientific Python stack. A typical environment will require:
	•	numpy
	•	pandas
	•	matplotlib
	•	scipy
	•	openpyxl

Depending on the scripts, you may also need:
	•	seaborn
	•	scikit-learn

A simple installation example is:

pip install numpy pandas matplotlib scipy openpyxl seaborn scikit-learn

A conda-based setup is also reasonable:

conda create -n kmt2a-ou python=3.11
conda activate kmt2a-ou
pip install numpy pandas matplotlib scipy openpyxl seaborn scikit-learn

Reproducibility notes

This repository is designed for manuscript reproducibility using processed and analysis-ready inputs.

Important notes:
	•	The repository is manuscript-specific.
	•	Figure scripts may assume that upstream processed tables already exist.
	•	Some filenames and figure panel contents may reflect manuscript-version-specific organization.
	•	Results may vary slightly if plotting defaults, software versions, or random seeds differ.

For a strong archival submission, it is recommended to preserve:
	•	the exact repository commit used for the manuscript,
	•	the precise processed input tables used in the analyses,
	•	the full supplementary tables workbook,
	•	an environment specification such as requirements.txt or environment.yml.

Recommended additions

For a cleaner public release, you may want to add:
	•	requirements.txt
	•	environment.yml
	•	LICENSE
	•	.gitignore
	•	a results/ folder for exported figure files
	•	a short note describing expected outputs for each figure script

Suggested requirements.txt

numpy
pandas
matplotlib
scipy
openpyxl
seaborn
scikit-learn

Example output mapping

A useful addition for readers is to indicate which script generates which manuscript figure:
	•	Figure1_OU_vs_Brownian_vs_Markov.py → Main Figure 1
	•	Figure2_kmt2a_clinical_data_analysis_by_ou_branching.py → Main Figure 2
	•	Figure3_Combined_P15.py → Main Figure 3
	•	Figure4_P15_OU_Branching_Combined.py → Main Figure 4
	•	Supplementary_Tables.xlsx → Supplementary tables package

Citation

If you use this repository, please cite the associated preprint.

Seung-Hwan Kim. Computational and cohort-level extension of hybrid OU-Branching modeling in pediatric KMT2A-rearranged leukemia. bioRxiv preprint, Version 1, November 2025.

Contact

Author: Seung-Hwan Kim
