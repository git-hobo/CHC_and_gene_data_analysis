#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 18:12:07 2025

@author: paul-cr
"""

import pandas as pd
import os
from chcchem.chc import CHC

chc_table_path = os.path.join("..", "data", "CHC_analysis_finaldraft_reformed.xlsx")
meta_data_table = os.path.join("..", "aux_info", "substance_metadata.tsv")
chc_data = pd.read_excel(chc_table_path, sheet_name="Pivot 1", index_col=0)
substances = list(chc_data.index)

substance_metadata = {substance: CHC(substance).as_dict() for substance in substances}
df_meta = pd.DataFrame.from_dict(substance_metadata, orient="index")
df_meta.to_csv(meta_data_table, index=False, sep="\t")
