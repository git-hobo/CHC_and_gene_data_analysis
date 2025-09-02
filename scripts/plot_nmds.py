#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 20:59:38 2025

@author: fabian-sc
"""

import os
import re
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from adjustText import adjust_text
import colorsys
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# -------------------------
# User settings
excel_path = os.path.join("..", "data", "CHC_analysis_finaldraft_reformed.xlsx")      # your Excel file
data_sheet = "Pivot 1"         # sheet with the abundance/fraction table
index_col = 0                 # column holding sample IDs (row names)
# Optional: if you have metadata in another sheet with a "Sample" column that matches the index:
meta_sheet = None             # e.g., "metadata" or None
meta_sample_col = "Sample"    # column in metadata that matches row names
meta_group_col = "Colors"      # column in metadata used to color/shape points (optional)

# Plotting options
point_size = 10
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

# data filtering
# filterout = ["unknown", "_"]
# pattern = '|'.join(map(re.escape, filterout))
# neg_pattern = fr'^(?!.*(?:{pattern})).*$'
# df = df.filter(regex=neg_pattern)
# ==> currently unimplemented, because it requires rebalancing the thing

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

# ---- helpers
def adjust_lightness(color, factor=1.0):
    """factor>1 -> lighter, factor<1 -> darker."""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0, min(1, l * factor))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return (r2, g2, b2)

def pick_base_colors(n):
    # use tab20 first; if more species, fall back to evenly spaced HSV hues
    tab = plt.get_cmap("tab20").colors
    if n <= len(tab):
        return list(tab[:n])
    # more species than tab20 → spaced hues
    return [mcolors.hsv_to_rgb((i / n, 0.65, 0.85)) for i in range(n)]

# ---- parse species/sex from index like "mmM", "eaF"
codes = pd.Index(X.index.astype(str))
species = codes.str[:2]      # e.g., "mm", "ea"
sex = codes.str[2:3].str.upper()   # "W", "F", "M" (or whatever is present)

unique_species = species.unique().tolist()
base_colors = dict(zip(unique_species, pick_base_colors(len(unique_species))))

# sex → shade factors (tweak to taste)
sex_factor = {
    "W": 1.00,  # workers: base
    "F": 1.15,  # females: lighter
    "M": 0.85,  # males: darker
}
# default if other codes appear
default_factor = 1.0

# final color per point
point_colors = [
    adjust_lightness(base_colors[sp], sex_factor.get(sx, default_factor))
    for sp, sx in zip(species, sex)
]

# ---- plot (no text labels)
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
sc = ax.scatter(coords[:, 0], coords[:, 1], s=point_size, c=point_colors)

# species legend (one dot per species in base hue)
species_handles = [
    Line2D([0], [0], marker='o', linestyle='',
           color=base_colors[sp], label=sp, markersize=6)
    for sp in unique_species
]
leg1 = ax.legend(handles=species_handles, title="Species", fontsize=8, loc="upper left")
ax.add_artist(leg1)

# sex legend (shade examples using the first species hue)
example_hue = list(base_colors.values())[0]
sex_handles = []
for label, f in [("Worker", sex_factor.get("W", 1.0)),
                 ("Female", sex_factor.get("F", 1.1)),
                 ("Male",   sex_factor.get("M", 0.9))]:
    sex_handles.append(
        Line2D([0], [0], marker='o', linestyle='',
               color=adjust_lightness(example_hue, f), label=label, markersize=6)
    )
ax.legend(handles=sex_handles, title="Sex (shade)", fontsize=2, loc="upper right")

ax.margins(0.08)
ax.set_xlabel("NMDS1")
ax.set_ylabel("NMDS2")
ax.set_title(f"NMDS (Bray–Curtis) — stress={stress:.2f}")
ax.axhline(0, lw=0.5, alpha=0.3)
ax.axvline(0, lw=0.5, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join("..", "data", "nmds.svg"))
plt.show()
# ---- split plotting by sex
sex_categories = sex.unique().tolist()  # e.g., ["W", "F", "M"]

fig, axes = plt.subplots(1, len(sex_categories), figsize=(6*len(sex_categories), 5), dpi=150, sharex=True, sharey=True)

if len(sex_categories) == 1:
    axes = [axes]  # make iterable if only one sex present

for ax, sx in zip(axes, sex_categories):
    mask = sex == sx
    sc = ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        s=point_size,
        c=[base_colors[sp] for sp in species[mask]],
        label=None
    )

    # legend for species (same in all subplots)
    if ax is axes[0]:  # only add once
        species_handles = [
            Line2D([0], [0], marker='o', linestyle='',
                   color=base_colors[sp], label=sp, markersize=6)
            for sp in unique_species
        ]
        ax.legend(handles=species_handles, title="Species", fontsize=2, loc="upper right")

    ax.set_title(f"Sex: {sx}")
    ax.axhline(0, lw=0.5, alpha=0.3)
    ax.axvline(0, lw=0.5, alpha=0.3)

# shared labels
fig.text(0.5, 0.04, "NMDS1", ha="center")
fig.text(0.04, 0.5, "NMDS2", va="center", rotation="vertical")
fig.suptitle(f"NMDS (Bray–Curtis) — stress={stress:.2f}", y=0.98)

plt.tight_layout(rect=[0.04, 0.04, 1, 0.95])
plt.savefig(os.path.join("..", "data", "nmds_per_sex.svg"))
plt.show()


# consistent axis limits across all plots
pad = 0.05
xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
xr = xmax - xmin
yr = ymax - ymin
xlim = (xmin - xr*pad, xmax + xr*pad)
ylim = (ymin - yr*pad, ymax + yr*pad)

# friendly names (fallback to code if unseen)
sex_names = {"W": "Worker", "F": "Female", "M": "Male"}

# make + save one plot per sex present
for sx in sex.unique():
    mask = sex == sx

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        s=point_size,
        c=[base_colors[sp] for sp in species[mask]],
    )

    # species legend
    species_handles = [
        Line2D([0], [0], marker='o', linestyle='',
               color=base_colors[sp], label=sp, markersize=6)
        for sp in unique_species
    ]
    ax.legend(handles=species_handles, title="Species", fontsize=8, loc="upper right")

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("NMDS1"); ax.set_ylabel("NMDS2")
    title_sex = sex_names.get(sx, str(sx))
    ax.set_title(f"NMDS (Bray–Curtis) — {title_sex} — stress={stress:.2f}")
    ax.axhline(0, lw=0.5, alpha=0.3); ax.axvline(0, lw=0.5, alpha=0.3)
    plt.tight_layout()

    outname = f"nmds_{sx}.svg"   # e.g., nmds_W.svg, nmds_F.svg, nmds_M.svg
    plt.savefig(os.path.join("..", "data", outname))
    plt.close(fig)