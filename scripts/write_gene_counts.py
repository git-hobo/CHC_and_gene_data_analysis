# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 10:40:35 2025

@author: Nutzer
"""

import pandas as pd
import re
import glob
import os
from chcchem.chc import *

in_files = glob.glob("/media/fabian-sc/ag_nh_wd/GEvol/gene_counts/*.txt")
out_file_internal = "/media/fabian-sc/ag_nh_wd/GEvol/gene_counts.xlsx"
out_file_external = "/home/fabian-sc/Documents/CHC_and_gene_data/data/gene_counts.tsv"

results_per_fam = [pd.read_csv(in_file, sep="\t") for in_file in in_files]
selected_families = ["ELO", "FAS", "DESAT"]
hedychrum_families = ["Acyl", "CRAL", "RED", "ketoacyl", "FAR", "short", "_cf_", "Cyt", "GR", "PS-DH", "p450"]
selected_genera = ["Hedychrum"]
unselected_species = ["Polistes_aff_dominula", "Formica_rufibarbis", "Megascolia_maculata"]

code_to_spec = dict()
with open(os.path.join("..", "aux_info", "genome_species.tsv")) as tsv_fh:
    for line in tsv_fh:
        code, species = line.rstrip('\n').split()
        code_to_spec[code] = species

results_df = pd.DataFrame()
for i, df in enumerate(results_per_fam):
    for code in code_to_spec.keys():
        if code in in_files[i]:
            species = code_to_spec[code]
            code_oi = code
        else:
            continue
    try:
        code_and_gf = os.path.basename(in_files[i]).rsplit("_", maxsplit=1)[0]
        gene_family = re.sub(code_oi + "_", "", code_and_gf)
        results_df.loc[species, gene_family] = int(df.loc[0, "Number of annotated genes identified"])
    except KeyError:
        try:
            results_df.loc[species, df["Gene/Gene Family"]] = int(df.loc[0, "Number of annotated genes identified"])
        except KeyError:
            continue

results_df = results_df.reindex(sorted(results_df.columns), axis=1)
results_df.index.name = 'species'
fam_pattern = '|'.join(map(re.escape, selected_families))
subselection_df = results_df.filter(regex=fam_pattern)
spec_pattern = '|'.join(map(re.escape, unselected_species))
neg_pattern = fr'^(?!.*(?:{spec_pattern})).*$'
subselection_df = subselection_df.filter(regex=neg_pattern, axis=0)

hedychrum_df = filter_species(results_df, selected_genera)
hedychrum_df = filter_gene_families(hedychrum_df, hedychrum_families, neg=True)

subselection_df = subselection_df.astype('int64')
subselection_df.index.name = "species"
subselection_df.to_excel(out_file_internal)
subselection_df.to_csv(out_file_external, sep="\t")
hedychrum_df = hedychrum_df.astype('int64')
hedychrum_df.index.name = "species"
hedychrum_df.to_csv("/home/fabian-sc/Documents/CHC_and_gene_data/data/hedychrum.tsv", sep="\t")