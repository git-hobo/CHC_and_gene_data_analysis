#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 16:46:42 2025

@author: fabian-sc
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path
from typing import Dict, List

# ---- config you can tweak ----
TREAT_ORDER: List[str] = ["odb", "bf", "chc"]  # desired order within each family
SINGLE_TREAT = "single"                        # label used for singleton families (no suffix)
INDEX_COL = "species"
# ------------------------------

def load_table(tsv_path: str | os.PathLike) -> pd.DataFrame:
    """Load the wide TSV (species + <FAM>[_<treat>] columns)."""
    df = pd.read_csv(tsv_path, sep="\t")
    if INDEX_COL in df.columns:
        df[INDEX_COL] = df[INDEX_COL].astype(str)
    return df
def tidy_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide table to long with columns: species, family, treatment, value.
    Includes both <FAM>_<treat> and singleton <FAM> columns.
    """
    # everything except the index column is a candidate
    value_cols = [c for c in df.columns if c != INDEX_COL]

    long = (
        df.melt(id_vars=[INDEX_COL], value_vars=value_cols,
                var_name="family_treat", value_name="value")
          .dropna(subset=["value"])
    )

    # split at the last underscore if present; otherwise mark as singleton
    has_us = long["family_treat"].str.contains("_")
    fam_treat = long.loc[has_us, "family_treat"].str.rsplit("_", n=1, expand=True)
    long.loc[has_us, "family"] = fam_treat[0]
    long.loc[has_us, "treatment"] = fam_treat[1]

    # singleton columns: entire header is the family name, treatment = SINGLE_TREAT
    long.loc[~has_us, "family"] = long.loc[~has_us, "family_treat"]
    long.loc[~has_us, "treatment"] = SINGLE_TREAT

    long = long.drop(columns=["family_treat"])

    # treatment order: regular ones first (if present), then SINGLE_TREAT
    # Build categories dynamically so we don’t drop anything unseen
    present_treats = pd.Index(long["treatment"].unique().tolist())
    ordered_cats = [t for t in TREAT_ORDER if t in present_treats] + \
                   ([SINGLE_TREAT] if SINGLE_TREAT in present_treats else [])
    long["treatment"] = pd.Categorical(long["treatment"],
                                       categories=ordered_cats,
                                       ordered=True)
    return long

def prepare_per_family(long: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Return {family -> DataFrame} with index=species, columns=ordered treatments (incl. SINGLE_TREAT).
    """
    out: Dict[str, pd.DataFrame] = {}
    for fam, sub in long.groupby("family", sort=True):
        # respect the categorical order set in tidy_long
        cols_in_order = [t for t in sub["treatment"].cat.categories if t in sub["treatment"].unique()]
        pivot = (
            sub.pivot_table(index=INDEX_COL,
                            columns="treatment",
                            values="value",
                            aggfunc="first")
               .reindex(columns=cols_in_order)
               .sort_index()
        )
        out[fam] = pivot
    return out


def plot_all_families_one_axes(
    per_family,
    species_colors,
    treat_order=None,     # e.g. ["odb","bf","chc"]; if None, use each pivot's column order
    group_width=0.8,      # width allotted to each (family,treatment) group
    fam_gap=0.7,          # extra gap between families (in group-width units)
    figsize=(18, 6),
    ystep=None            # set to 10 to force 10-step gridlines; else auto
):
    """
    Single-axes plot for ALL families on a shared y-scale.
    X-axis is a sequence of (family, treatment) groups. Each group shows
    bars for all species (colored consistently).

    per_family: dict[family -> pivot], pivot.index = species, pivot.columns = treatments
    species_colors: dict[species -> color]
    """

    families = list(per_family.keys())
    # Build a stable species order for color/legend
    all_species_sorted = sorted({sp for pvt in per_family.values() for sp in pvt.index})

    fig, ax = plt.subplots(figsize=figsize)

    # Build a flat sequence of (family, treatment) groups and compute x positions
    group_centers = []
    group_labels = []  # for x-ticks (two-line label: family \n treatment)
    x_cursor = 0.0
    bar_handles_for_legend = {}

    # Determine max value for y ticks/grids
    ymax_candidates = []
    for fam, pvt in per_family.items():
        ymax_candidates.append(np.nanmax(pvt.values) if pvt.size else 0.0)
    ymax = float(np.nanmax(ymax_candidates)) if ymax_candidates else 0.0

    # Iterate families in order
    for f_idx, fam in enumerate(families):
        pvt = per_family[fam]

        # figure out treatments in desired order for this family
        if treat_order is None:
            treatments = list(pvt.columns)
        else:
            treatments = [t for t in treat_order if t in pvt.columns]
        if not treatments:
            continue

        n_species = len(pvt.index)
        bar_w = group_width / max(1, n_species)  # split group width among species

        # Add an extra gap before every family except the first
        if f_idx > 0:
            x_cursor += fam_gap

        for t in treatments:
            # center x for this (fam, treat) group
            xc = x_cursor
            group_centers.append(xc)
            group_labels.append((fam, t))

            # bars for each species
            for i, sp in enumerate(pvt.index.tolist()):
                offset = (i - (n_species - 1) / 2.0) * bar_w
                height = pvt.loc[sp, t]
                h = ax.bar(
                    xc + offset,
                    height,
                    width=bar_w,
                    color=species_colors.get(sp, None),
                    edgecolor='black',
                    linewidth=0.5,
                )
                # For legend (one handle per species)
                if sp not in bar_handles_for_legend:
                    bar_handles_for_legend[sp] = h

            # advance x_cursor by one group width to place the next treatment group
            x_cursor += group_width

    # X tick labels: "Family\nTreatment"

    if group_centers:
        labels = [f"{fam}\n{t}" for fam, t in group_labels]
        ax.set_xticks(group_centers)
        # italic font style
        italic_font = fm.FontProperties(style='italic')
        ax.set_xticklabels(labels, fontproperties=italic_font, rotation=0)
        ax.set_xlabel("Family / Treatment")
        ax.set_ylabel("Count")
        ax.set_title("All families on one plot (species-colored, shared y-scale)")

    # Grid lines
    if ystep is None:
        ystep = 10 if ymax <= 200 else 20
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.5, color='grey', alpha=0.3)
    ax.set_yticks(np.arange(0, ymax + ystep, ystep))

    # Legend (species)
    handles = [bar_handles_for_legend[sp] for sp in all_species_sorted if sp in bar_handles_for_legend]
    labels = [sp for sp in all_species_sorted if sp in bar_handles_for_legend]
    ax.legend(handles, labels, title="Species", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    fig.tight_layout(rect=(0, 0, 0.88, 1))
    return fig, ax


def make_species_colors(all_species):
    """
    Create a deterministic color map for species across all plots.
    Uses a qualitative tab20 colormap (cycled if needed).
    """
    cmap = plt.cm.get_cmap("tab20", max(20, len(all_species)))
    return {sp: cmap(i % cmap.N) for i, sp in enumerate(sorted(all_species))}

def plot_family_species_colored(pivot, family, species_colors, ax=None, group_width=0.8):
    """
    Plot grouped bars with species on x, treatments grouped,
    but bars for the same species share the same color.

    Parameters
    ----------
    pivot : DataFrame (index=species, columns=treatments in desired order)
    family : str
    species_colors : dict {species -> color}
    ax : matplotlib Axes or None
    group_width : float in (0,1], total width allotted to each species group
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    species = list(pivot.index)
    treatments = list(pivot.columns)
    n_treat = len(treatments)
    n_species = len(species)

    # spacing within each species group
    bar_w = group_width / max(1, n_treat)
    x = np.arange(n_species)  # species positions

    # We draw by species (so we can set a single color for all its bars)
    for i, sp in enumerate(species):
        color = species_colors.get(sp, None)
        # positions of this species' bars across treatments
        # offsets centered around x[i]
        offsets = x[i] + (np.arange(n_treat) - (n_treat - 1) / 2.0) * bar_w
        heights = pivot.loc[sp].values
        ax.bar(offsets, heights, width=bar_w, color=color, edgecolor='black', linewidth=0.5)

    ax.set_title(f"{family}: grouped by species (species-colored)")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    ax.set_xticks(x, species, rotation=45, ha="right")
    
    
    # Horizontal grid lines every 10 units
    ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.5, color='grey', alpha=0.3)
    ax.set_yticks(np.arange(0, pivot.max().max() + 10, 10))


    # One legend entry per species; place outside to avoid overlap
    handles, labels = ax.get_legend_handles_labels()
    # deduplicate (matplotlib can repeat labels)
    seen = set()
    uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend(*zip(*uniq), title="Species", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

    return ax


if __name__ == "__main__":
    # Example usage:
    # 1) Load
    df = load_table(os.path.join("..", "data", "gene_counts.tsv"))

    # 2) Tidy
    long = tidy_long(df)

    # 3) Build per-family pivots (ready for plotting later)
    plt.rcParams['font.family'] = 'Liberation Sans'
    per_family = prepare_per_family(long)
    all_species = set().union(*[set(pvt.index) for pvt in per_family.values()])
    species_colors = make_species_colors(all_species)
    # Now `per_family["DESAT"]`, `per_family["ELO"]`, `per_family["FAS"]` are
    # DataFrames with rows=species, cols=ordered treatments, values=counts.
    # You can inspect one:
    print(per_family["DESAT"].head())

    # When ready to plot, uncomment below:
    import matplotlib.pyplot as plt
    for fam, pivot in per_family.items():
        plt.figure(figsize=(12, 5))
        plot_family_species_colored(pivot, fam, species_colors)
        plt.tight_layout()
        plt.savefig(os.path.join("..", "data", f"{fam}_spec.svg"))
        plt.show()


    df = load_table(os.path.join("..", "data", "hedychrum.tsv"))
    
    # Consistent colors:
    long = tidy_long(df)
    per_family = prepare_per_family(long)
    
    # One axes, shared y-scale, families concatenated on x:
    fig, ax = plot_all_families_one_axes(
        per_family,
        species_colors,
        treat_order=None,  # or None to use the pivot's existing order
        group_width=0.8,
        fam_gap=0.8,       # tweak spacing between families
        figsize=(20, 7),
        ystep=10           # grid every 10
    )
    plt.savefig(os.path.join("..", "data", "Hedychrum.svg"))

