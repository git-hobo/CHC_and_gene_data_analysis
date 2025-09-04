#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:57:53 2025

@author: fabian-sc
"""
import pandas as pd
import os
from chcchem.chc import * 

set_style(size=16)

### data opening
species_path = os.path.join("..", "aux_info", "code_to_spec.tsv")
gene_counts_path = os.path.join("..", "data", "gene_counts.tsv")
assembly_path = os.path.join("..", "aux_info", "asm_inf_all_01Sep2025.xlsx")
species_dict = dict(pd.read_csv(species_path, sep="\s+").values)
gene_counts_data = pd.read_csv(gene_counts_path, sep="\s+", index_col=0)
assembly_data = pd.read_excel(assembly_path, sheet_name="selected")

### data writing
busco_out = os.path.join("..", "fig", "busco.svg")
genespc_out = os.path.join("..", "fig", "genespace_vs_busco.svg")

### processing
species_list = ["Polistes_aff_dominula", "Formica_rufibarbis", "Hedychrum_nobile"]
selected_data = filter_species(assembly_data, species_list, neg=True)

### plotting BUSCO
dtol_species = ["Bombus_lapidarius", "Bombus_terrestris", "Vespa_crabro", "Vespula_germanica", "Cerceris_rybyensis"]
fig, ax = plot_busco_stacked(selected_data, 
                             name_col="species", 
                             title="BUSCO Summary", 
                             highlight_species=dtol_species,
                             highlight_color="purple")

plt.savefig(busco_out)
plt.show()

### plotting BUSCO vs gene counts
selected_data = selected_data.set_index("species")
gene_counts_data = gene_counts_data.join(selected_data, how="inner")
fig, ax = plt.subplots(figsize=(16, 9))

count_columns = ["DESAT_chc", "ELO_chc", "FAS_chc"]
color_labels = ["red", "blue", "purple"]
for i, column in enumerate(count_columns):
    scatter_gene_counts(
        gene_counts_data, "Complete percentage", column,
        color=color_labels[i], ax=ax, labelcol="species",
        freeze_limits=False
    )

ax.set_autoscale_on(True)  # not strictly needed once the unconditional freeze is gone
ax.relim()
ax.autoscale_view()
ax.legend()
plt.savefig(genespc_out)
plt.show()