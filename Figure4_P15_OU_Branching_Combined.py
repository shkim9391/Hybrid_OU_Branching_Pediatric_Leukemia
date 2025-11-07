#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 12:28:39 2025

@author: seung-hwan.kim
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 4: P15 Hybrid OU–Branching Results (Composite A + B)
Clone Fractions and Mutation VAFs plotted together.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load data
clone_df = pd.read_csv("P15_OU_clone_fractions.csv")
vaf_df   = pd.read_csv("P15_OU_mutation_VAFs.csv")

# --- Create composite figure with 2 columns ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))   # width × height in inches
(axA, axB) = axes

# --- Panel A: Clone fractions ---
axA.plot(clone_df['day'], clone_df['Core_clone'],         label='Core clone',         color='tab:blue')
axA.plot(clone_df['day'], clone_df['Intermediate_clone'], label='Intermediate clone', color='tab:orange')
axA.plot(clone_df['day'], clone_df['Aggressive_clone'],   label='Aggressive clone',   color='tab:red')
axA.set_xlabel("Days from diagnosis")
axA.set_ylabel("Clone fraction")
axA.set_title("A. P15 Hybrid OU–Branching: Clone Fractions Over Time")
axA.legend(fontsize=8, frameon=False)

# --- Panel B: Mutation VAFs ---
colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green', 'tab:purple']
genes  = ['KRAS_core','TP53_aggressive','PTPN11_aggressive','CREBBP_aggressive','KMT2A_fusion_marker']
for m, c in zip(genes, colors):
    axB.plot(vaf_df['day'], vaf_df[m], label=m, color=c)
axB.set_xlabel("Days from diagnosis")
axB.set_ylabel("Simulated VAF")
axB.set_title("B. P15 Hybrid OU–Branching: Mutation VAFs Over Time")
axB.legend(fontsize=8, frameon=False)

# --- Global formatting ---
for ax in (axA, axB):
    ax.tick_params(labelsize=9)
    ax.title.set_fontsize(11)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)
    axA.legend(frameon=True, framealpha=1, edgecolor='gray')
    axB.legend(frameon=True, framealpha=1, edgecolor='gray')

plt.tight_layout(w_pad=2.5)  # space between panels
fig.savefig("Figure4_P15_OU_Branching_Combined.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: Figure4_P15_OU_Branching_Combined.png")