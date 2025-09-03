#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:57:53 2025

@author: fabian-sc
"""
import pandas as pd
import os
from chcchem.chc import * 



species_path = os.path.join("..", "aux_info", "code_to_spec.tsv")
gene_counts_path = os.path.join("..", "data", "gene_counts.tsv")
assembly_path = os.path.join("..", "aux_info", "asm_inf_all_01Sep2025.xlsx")

species_dict = dict(pd.read_csv(species_path, sep="\s+").values)
gene_counts_data = pd.read_csv(gene_counts_path, sep="\s+", index_col=0)
assembly_data = pd.read_excel(assembly_path, sheet_name="selected")


fig, ax = plot_busco_stacked(assembly_data, name_col="species", title="BUSCO Summary")
plt.show()