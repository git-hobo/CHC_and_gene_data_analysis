#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 14:32:55 2025

@author: fabian-sc
"""
import pandas as pd

d = dict(pd.read_csv("/home/fabian-sc/Desktop/Bash/codes/species.tsv", sep="\t").values)
shorthand = [key[0:2] for key in list(d.keys()) if len(key) == 7]
shorthand_dict = dict()

for code, species in d.items():
    if len(code) == 7:
        shorthand_dict[code[0:2]] = species
    else:
        continue
    
pd.Series(shorthand_dict).to_csv("/home/fabian-sc/Documents/CHC_and_gene_data/aux_info/code_to_spec.tsv", sep="\t", header=False)