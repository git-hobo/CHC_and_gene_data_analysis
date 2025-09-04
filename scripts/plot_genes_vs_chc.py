#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 09:56:10 2025

@author: fabian-sc
"""

import pandas as pd
import os
from chcchem.chc import *

### paths
species_path = os.path.join("..", "aux_info", "code_to_spec.tsv")
metadat_path = os.path.join("..", "aux_info", "substance_metadata.tsv")
chc_dat_path = os.path.join("..", "data", "CHC_analysis_finaldraft_reformed.xlsx")
gene_counts_path = os.path.join("..", "data", "gene_counts.tsv")

### loading dataframes
species_dict = dict(pd.read_csv(species_path, sep="\s+").values)
substance_metadata = pd.read_csv(metadat_path, sep="\t")
chc_dat_all = load_chc(chc_dat_path, "Pivot 1")
chc_dat_chl = load_chc(chc_dat_path, "Pivot 3")
gene_counts_data = pd.read_csv(gene_counts_path, sep="\s+", index_col=0)

### filters
dtol_species = ["Bombus_lapidarius", "Bombus_terrestris", "Vespa_crabro", "Vespula_germanica", "Cerceris_rybyensis"]

### chainlength data
chc_dat_chl["species"] = parse_to_species(chc_dat_chl.index, species_dict)
# Identify chain length columns (all numeric column names)
chain_cols = [c for c in chc_dat_chl.columns if isinstance(c, (int, float))]
chain_lengths = chc_dat_chl[chain_cols].columns.astype(int)

# --- Step 1: per-individual weighted average chain length ---
chc_dat_chl["avg_chain_length"] = (chc_dat_chl[chain_cols] * chain_lengths).sum(axis=1)

# --- Step 2: summarize per species ---
species_summary = (
    chc_dat_chl.groupby("species")["avg_chain_length"]
      .agg(["mean", "std", "count"])
      .reset_index()
      .set_index("species")
)

gene_data_subsection = species_summary.loc[species_summary.index.intersection(gene_counts_data.index)]
gene_data_subsection = gene_data_subsection.join(gene_counts_data, how="inner")
gene_data_subsection.insert(0, "species", gene_data_subsection.index)
scatter_gene_counts(gene_data_subsection, 
                    "mean",
                    "ELO_chc",
                    labelcol="species",
                    highlight_species=dtol_species,
                    highlight_color="purple")
plt.savefig(os.path.join("..", "data", "ELO.svg"))
plt.show()


### chainlength data
chc_dat_all["species"] = parse_to_species(chc_dat_chl.index, species_dict)
mb_richness = class_richness(chc_dat_all, substance_metadata)
gene_data_subsection = gene_data_subsection.join(mb_richness, how="inner")
scatter_gene_counts(gene_data_subsection, "methylbranched_richness", "FAS_chc",  labelcol="species")
plt.savefig(os.path.join("..", "data", "FAS.svg"))
plt.show()

### unsaturation data
unsat_richness = backbone_richness(chc_dat_all, substance_metadata)
gene_data_subsection = gene_data_subsection.join(unsat_richness, how="inner")
scatter_gene_counts(gene_data_subsection, "unsaturated_richness","DESAT_chc", labelcol="species")
plt.savefig(os.path.join("..", "data", "DESAT.svg"))
plt.show()
