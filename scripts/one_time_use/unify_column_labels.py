#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:55:56 2025

@author: fabian-sc
"""

import pandas as pd
import os

excel_in = r"C:\Users\Nutzer\Downloads\CHC_analysis_finaldraft.xlsx"
excel_out = os.path.join("..", "..", "data", "CHC_analysis_draft.xlsx")
translation_file_path = os.path.join("..", "..","aux_info", "chc_code_translation.tsv")

xlsx = pd.ExcelFile(excel_in)
all_sheets = {}
for name in xlsx.sheet_names:
    if name.lower() == "readme":
        all_sheets[name] = pd.read_excel(xlsx, sheet_name=name, header=None, index_col=0)  # no header row
    else:
        all_sheets[name] = pd.read_excel(xlsx, sheet_name=name, header=0, index_col=0)
with open(translation_file_path, "rt") as tr_fh:
    lines = tr_fh.readlines()
translation = {line.split()[0]: line.split()[-1] for line in lines}

all_sheets_renamed = {sheet_name: sheet.rename(columns=translation) for sheet_name, sheet in all_sheets.items()}
for name, df in all_sheets_renamed.items():
    # only sort if the columns are a normal Index (not MultiIndex, not all numeric)
    if df.columns.nlevels == 1 and df.columns.dtype == object:
        all_sheets_renamed[name] = df[sorted(df.columns)]
with pd.ExcelWriter(excel_out) as writer:
    for sheet_name, df in all_sheets_renamed.items():
        df.to_excel(writer, sheet_name=sheet_name)