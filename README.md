# KMT2A_OU_Branching_Leukemia

Code, processed cohort tables, supplementary tables, and reproducible figure-generation workflows for cohort-level and simulation-based analysis of hybrid Ornstein--Uhlenbeck (OU) and OU--Branching dynamics in pediatric KMT2A-rearranged leukemia.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17555292.svg)](https://doi.org/10.5281/zenodo.17555292)

## Overview

This repository supports the revised iScience manuscript:

**Modeling constrained tumor evolution through hybrid Ornstein--Uhlenbeck and branching dynamics**  
Manuscript number: **ISCIENCE-D-25-20135**

The project evaluates a hybrid OU--Branching framework for pediatric leukemia evolution. The model links continuous mean-reverting tumor-state dynamics with stochastic lineage birth, death, and extinction. The revised analysis treats Brownian motion as the zero-attraction limiting case of OU dynamics and uses case-level likelihood-based model comparison to evaluate whether patient trajectories are better described by Brownian diffusion, OU diffusion, Markov-emission dynamics, branching-only directional drift, or an OU--Branching jump-diffusion proxy.

This repository is organized for manuscript reproducibility rather than as a general-purpose software package.

## Main revision features

The revised repository and supplementary files support the following manuscript additions:

- Explicit Brownian--OU relationship: Brownian motion is treated as the zero-attraction OU limit.
- Explicit birth, death, and extinction definitions for the OU--Branching framework.
- Case-level likelihood-based model comparison using longitudinal targeted-sequencing VAF trajectories.
- New Figure 5: patient-by-model heatmap of delta-AICc values.
- New Supplementary Table S8: case-level model-comparison results.
- Improved code-to-output mapping and supplementary table annotation.

## Scientific scope

The repository studies pediatric KMT2A-rearranged leukemia using stochastic evolutionary models that represent disease progression as a combination of constrained tumor-state dynamics and lineage turnover.

Conceptually:

- **Brownian diffusion** provides a null continuous-drift model without stabilizing attraction.
- **OU diffusion** models continuous mean-reverting dynamics around latent disease-state equilibria.
- **Markov-emission models** provide a memoryless state-switching comparator.
- **Branching-only directional-drift models** capture directional change without OU stabilization.
- **OU--Branching jump-diffusion models** provide a tractable proxy for coupled mean reversion and abrupt lineage-like reconfiguration in longitudinal VAF trajectories.

The goal is not to claim that one model universally explains all pediatric leukemia cases. Instead, the workflow assigns model support at the individual-patient level and then interprets cohort-level relapse and subtype patterns alongside those case-level results.

## Data source and use

The empirical analysis uses de-identified clinical metadata and published longitudinal targeted-sequencing variant-allele-frequency (VAF) measurements from:

Ahlgren et al. *Nature Communications* 16, 8964.

No raw sequencing reads, single-cell profiles, independently generated clinical specimens, controlled-access genomic data, or identifiable patient information are used in this repository.

## Repository structure

Recommended repository layout:

```text
repo_root/
  data/
    raw/
      Ahlgren_2025_Supplementary_Data_1-18.xlsx
    processed/
      kmt2a_clinical_data.xlsx
      P15_OU_clone_fractions.csv
      P15_OU_mutation_VAFs.csv
      lineage_metrics_summary.csv
      Supplementary_Data.xlsx

  scripts/
    Figure1_OU_vs_Brownian_vs_Markov.py
    Figure2_kmt2a_clinical_data_analysis_by_ou_branching_model.py
    Figure3_Combined_P15.py
    Figure4_P15_OU_Branching_Combined.py
    run_case_level_aicc_model_comparison_longform.py
    kmt2a_lineage_analysis.py
    kmt2a_phase_plane_lineage_plots.py

  results/
    figures/
      Figure1_OU_vs_Brownian_vs_Markov.png
      Figure2_kmt2a_clinical_summary.png
      Figure3_Combined_P15.png
      Figure4_P15_OU_Branching_Combined.png
      Figure5_case_level_model_comparison_AICc.png
      Figure5_case_level_model_comparison_AICc.pdf
    tables/
      iScience_case_level_AICc_model_comparison.csv
      iScience_case_level_model_fits_long.csv
      iScience_case_level_VAF_timeseries_collapsed.csv
      iScience_case_level_AICc_model_comparison.xlsx

  requirements.txt
  environment.yml
  README.md
```

The exact repository organization may differ slightly depending on manuscript-version-specific file placement. The script-to-output map below defines the intended reproducibility workflow.

## Script-to-output map

| Script | Main output | Manuscript item |
|---|---|---|
| `Figure1_OU_vs_Brownian_vs_Markov.py` | OU, Brownian, and Markov benchmark trajectories and variance profiles | Figure 1 |
| `Figure2_kmt2a_clinical_data_analysis_by_ou_branching_model.py` | Cohort-level relapse and disease-subtype summaries | Figure 2 |
| `Figure3_Combined_P15.py` | P15 trait, clone-count, phase-plane, and lineage panels | Figure 3 |
| `Figure4_P15_OU_Branching_Combined.py` | P15 clone-fraction and VAF-like trajectory synthesis | Figure 4 |
| `run_case_level_aicc_model_comparison_longform.py` | Case-level AICc workbook/table and delta-AICc heatmap | Figure 5; Supplementary Table S8 |
| `kmt2a_lineage_analysis.py` | Lineage-derived metrics | Supplementary lineage metrics |
| `kmt2a_phase_plane_lineage_plots.py` | Lineage-aware phase-plane plots | Supporting analyses |

## Supplementary tables workbook

The workbook `Supplementary_Tables.xlsx` contains Supplementary Tables S1--S8. The current workbook includes the following sheets:

| Supplementary table | Sheet name | Contents |
|---|---|---|
| Supplementary Table S1 | `Sim_Params` | Simulation parameters for Figure 1 model benchmarking, including OU--Branching, Brownian, and Markov settings. |
| Supplementary Table S2 | `Clinical_Data` | De-identified clinical metadata from the Ahlgren et al. cohort, including patient ID, disease subtype, relapse group, fusion annotation, and karyotype fields. |
| Supplementary Table S3 | `Group_Summary` | Group-level summaries of simulated final clone count and terminal trait values by relapse group. |
| Supplementary Table S4 | `Disease_Summary` | Disease-level summaries of simulated final clone count and terminal trait values by disease subtype. |
| Supplementary Table S5 | `Simulation_Summary` | Patient-level simulation summaries used for Figure 2, including duration, final clone count, and final trait. |
| Supplementary Table S6 | `Patient_Outputs` | Column definitions and patient-level simulation output annotations. |
| Supplementary Table S7 | `Lineage_Metrics_Summary` | Lineage-level metrics, including final clones, total clones, branch events, maximum depth, patient ID, disease, and relapse group. |
| Supplementary Table S8 | `Case_Level_Model_Comparison` | Case-level likelihood-based model-comparison results for evaluable patients, including time points, disease, group, best model, AICc, Akaike weight, and delta-AICc values for each candidate model. |

### Supplementary Table S8 summary

`Case_Level_Model_Comparison` contains the case-level analysis supporting Figure 5. Patients were included when they had at least eight distinct longitudinal targeted-sequencing time points. For each evaluable patient, corrected VAF values were collapsed by patient and sampling day, logit-transformed, and fitted to five candidate dynamical models:

1. Brownian neutral diffusion
2. Branching-only directional drift
3. OU diffusion
4. Markov-emission benchmark
5. OU--Branching jump-diffusion proxy

The key reported fields include:

- `Patient`
- `n_timepoints`
- `min_day`
- `max_day`
- `n_variants_median`
- `mean_vaf_diagnosis`
- `mean_vaf_final`
- `Disease`
- `Group`
- `FusionGeneatDiagnosis`
- `Infant/Child`
- `Survival`
- `best_model`
- `best_AICc`
- `best_weight`
- `delta_AICc_Brownian`
- `delta_AICc_Branching-only drift`
- `delta_AICc_OU`
- `delta_AICc_Markov-emission`
- `delta_AICc_OU-Branching jump`

In the revised analysis, 16 patients were evaluable for case-level AICc comparison. OU diffusion was preferred in six cases, the OU--Branching jump-diffusion proxy in six cases, and Brownian diffusion in four cases. Neither the Markov-emission benchmark nor the branching-only directional-drift proxy was selected as the best-supported model in any evaluable case.

## Typical workflow

From the repository root, run the scripts in the following order when regenerating the manuscript figures and tables:

```bash
# Optional lineage-derived intermediate analyses
python scripts/kmt2a_lineage_analysis.py
python scripts/kmt2a_phase_plane_lineage_plots.py

# Main manuscript figures
python scripts/Figure1_OU_vs_Brownian_vs_Markov.py
python scripts/Figure2_kmt2a_clinical_data_analysis_by_ou_branching_model.py
python scripts/Figure3_Combined_P15.py
python scripts/Figure4_P15_OU_Branching_Combined.py

# Case-level model comparison for Figure 5 and Supplementary Table S8
python scripts/run_case_level_aicc_model_comparison_longform.py
```

The Figure 5 workflow expects access to the Ahlgren et al. supplementary workbook or an equivalent processed VAF table. If the script uses hard-coded paths, update the input and output paths near the top of the script before running.

Expected Figure 5 outputs include:

```text
Figure5_case_level_model_comparison_AICc.png
Figure5_case_level_model_comparison_AICc.pdf
iScience_case_level_AICc_model_comparison.xlsx
iScience_case_level_AICc_model_comparison.csv
iScience_case_level_model_fits_long.csv
iScience_case_level_VAF_timeseries_collapsed.csv
```

## Example Python loading commands

Load the supplementary workbook:

```python
import pandas as pd

supp = pd.ExcelFile("data/processed/Supplementary_Data.xlsx")
print(supp.sheet_names)
```

Load the case-level model-comparison table:

```python
import pandas as pd

s8 = pd.read_excel(
    "data/processed/Supplementary_Data.xlsx",
    sheet_name="Case_Level_Model_Comparison",
    header=2,
)
print(s8[["Patient", "Disease", "Group", "best_model", "best_AICc", "best_weight"]].head())
```

Load the processed clinical dataset:

```python
import pandas as pd

clinical = pd.read_excel("data/processed/kmt2a_clinical_data.xlsx")
print(clinical.head())
```

Load P15 clone fractions:

```python
import pandas as pd

clone_frac = pd.read_csv("data/processed/P15_OU_clone_fractions.csv")
print(clone_frac.head())
```

Load P15 mutation VAF summaries:

```python
import pandas as pd

vafs = pd.read_csv("data/processed/P15_OU_mutation_VAFs.csv")
print(vafs.head())
```

Load lineage summary metrics:

```python
import pandas as pd

lineage_metrics = pd.read_csv("data/processed/lineage_metrics_summary.csv")
print(lineage_metrics.head())
```

## Software environment

The repository is intended to run in Python 3.11 with the standard scientific Python stack.

Required packages:

```text
numpy
pandas
matplotlib
scipy
openpyxl
```

Optional packages depending on local plotting or exploratory scripts:

```text
seaborn
scikit-learn
```

A simple installation example is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scipy openpyxl seaborn scikit-learn
```

A conda-based setup is:

```bash
conda create -n kmt2a-ou python=3.11
conda activate kmt2a-ou
pip install numpy pandas matplotlib scipy openpyxl seaborn scikit-learn
```

## Suggested `requirements.txt`

```text
numpy
pandas
matplotlib
scipy
openpyxl
seaborn
scikit-learn
```

## Reproducibility notes

This repository is designed for manuscript reproducibility using processed and analysis-ready inputs.

Important notes:

- The repository is manuscript-specific.
- Figure scripts may assume that upstream processed tables already exist.
- Some filenames and figure panel contents may reflect manuscript-version-specific organization.
- Results may vary slightly if plotting defaults, software versions, or random seeds differ.
- The case-level model-comparison workflow uses longitudinal targeted-sequencing VAF trajectories as tumor-burden proxies; these data do not fully identify latent single-lineage birth, death, and extinction events.
- The OU--Branching jump-diffusion model is therefore used as a tractable proxy for abrupt lineage-like reconfiguration in VAF time series.

For archival reproducibility, preserve:

- the exact repository commit used for manuscript submission;
- the precise processed input tables;
- `Supplementary_Tables.xlsx` with sheets S1--S8;
- the generated Figure 5 outputs and case-level model-comparison tables;
- `requirements.txt` or `environment.yml`;
- the Zenodo DOI associated with the release.

## Recommended public-release additions

For a cleaner public release, include:

- `requirements.txt`
- `environment.yml`
- `LICENSE`
- `.gitignore`
- `data/raw/` and `data/processed/` folders
- `results/figures/` and `results/tables/` folders
- a frozen Zenodo release corresponding to the exact manuscript revision

## Citation

If you use this repository, please cite the associated manuscript and archived repository release.

Seung-Hwan Kim. **Modeling constrained tumor evolution through hybrid Ornstein--Uhlenbeck and branching dynamics.** iScience revised manuscript, ISCIENCE-D-25-20135.

Repository archive DOI: [https://doi.org/10.5281/zenodo.17555292](https://doi.org/10.5281/zenodo.17555292)

## Contact

Seung-Hwan Kim, Ph.D.  
Department of Biology, Fisher College  
Email: seung-hwan.kim@fisher.edu
