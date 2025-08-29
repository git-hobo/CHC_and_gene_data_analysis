#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 20:59:38 2025

@author: fabian-sc
"""

import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from adjustText import adjust_text

# -------------------------
# User settings
excel_path = os.path.join("..", "data", "CHC_analysis_draft_reformed.xlsx")      # your Excel file
data_sheet = "Pivot 1"         # sheet with the abundance/fraction table
index_col = 0                 # column holding sample IDs (row names)
# Optional: if you have metadata in another sheet with a "Sample" column that matches the index:
meta_sheet = "Colors"             # e.g., "metadata" or None
meta_sample_col = "Sample"    # column in metadata that matches row names
meta_group_col = "Colors"      # column in metadata used to color/shape points (optional)

# Plotting options
point_size = 5
label_font_size = 0.5

# Transform choices
use_hellinger = True          # Hellinger transform (sqrt of row-wise relative abundances)
use_wisconsin = False         # Wisconsin double standardization (row relabund + divide by col maxima)
# NMDS options
n_components = 2
n_init = 16                   # multiple random starts (like metaMDS)
random_state = 42
max_iter = 1000
# %%


# -------------------------
# Load data
df = pd.read_excel(excel_path, sheet_name=data_sheet, index_col=0)

# Now: rows = substances, cols = individuals
# → transpose so that rows = individuals, cols = substances
df = df.T

# Keep only numeric columns
X = df.select_dtypes(include=[np.number]).copy()

# Replace inf/nan with 0
X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X[X < 0] = 0  # clip negatives if needed

# Normalize rows to proportions
row_sums = X.sum(axis=1)
nonzero = row_sums > 0
X.loc[nonzero] = X.loc[nonzero].div(row_sums[nonzero], axis=0)

# Optional transforms
if use_wisconsin:
    col_max = X.max(axis=0).replace(0, 1)
    X = X.div(col_max, axis=1)

if use_hellinger:
    X = np.sqrt(X)

# Distance matrix: Bray–Curtis
D = squareform(pdist(X.values, metric='braycurtis'))

# NMDS (non-metric MDS on a precomputed dissimilarity)
nmds = MDS(
    n_components=n_components,
    metric=False,
    dissimilarity='precomputed',
    n_init=n_init,
    max_iter=max_iter,
    random_state=random_state,
    eps=1e-9,
)
coords = nmds.fit_transform(D)
stress = nmds.stress_

# Optional metadata for coloring/grouping
groups = None
if meta_sheet:
    meta = pd.read_excel(excel_path, sheet_name=meta_sheet)
    meta = meta.set_index(meta_sample_col).reindex(X.index)
    if meta_group_col in meta.columns:
        groups = meta[meta_group_col].astype(str)

# %%
# Plot
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)

if groups is None:
    ax.scatter(coords[:, 0], coords[:, 1], s=point_size)
else:
    for g, sub in pd.Series(range(len(X)), index=X.index).groupby(groups):
        ix = sub.values
        ax.scatter(coords[ix, 0], coords[ix, 1], s=point_size, label=g)
    ax.legend(title=meta_group_col, fontsize=8)




# build text objects with a faint background (prevents visual clutter)
texts = []
for i, name in enumerate(X.index):
    texts.append(
        ax.text(
            coords[i,0], coords[i,1], str(name),
            fontsize=label_font_size, zorder=3, clip_on=False,
        )
    )

# nudge labels apart + draw leader lines
adjust_text(
    texts,
    x=coords[:,0], y=coords[:,1],
    only_move={'points':'', 'text':'xy'},
    force_text=0.1, force_points=0.1,
    expand_text=(1, 1), expand_points=(1, 1),
    arrowprops=dict(
        arrowstyle='-', lw=0.6, alpha=0.6,
        shrinkA=1, shrinkB=1,      # <- keeps line off the text/point
        mutation_scale=1
    ),
    ax=ax
)

ax.margins(0.08)  # a bit more breathing room

ax.set_xlabel("NMDS1")
ax.set_ylabel("NMDS2")
ax.set_title(f"NMDS (Bray–Curtis) — stress={stress:.2f}")
ax.axhline(0, lw=0.5, alpha=0.3)
ax.axvline(0, lw=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join("..", "data", "nmds.svg"))
plt.show()
