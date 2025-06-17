import pandas as pd
import numpy as np
from functools import reduce

mri_data = pd.read_csv("Data/PDMRI_Clean_Merged_6_13_25.csv")
gene_data = pd.read_csv("Data/gene_expression_summary.csv")
nhy_latest = pd.read_csv("Data/clean_mds_updrs.csv")

mri_data_bl  = mri_data.query("EVENT_ID == 'BL'")
gene_data_bl  = gene_data.query("EVENT_ID == 'BL'")

gene_data_bl = gene_data_bl.rename(columns={"NHY": "NHY_BL"})

baseline_dfs = [mri_data_bl,gene_data_bl]

baseline_merged = reduce(
    lambda left, right: pd.merge(
        left, right,
        on=["PATNO", "EVENT_ID"],   # merge keys
        how="inner",                # keep only PATNOs present in *all* tables
        suffixes=("", "_dup")       # avoids crashing if any column names clash
    ),
    baseline_dfs
)

baseline_merged = baseline_merged.drop(columns=["EVENT_ID"])
baseline_merged = baseline_merged.loc[:,~baseline_merged.columns.str.endswith("_dup")]
baseline_merged = baseline_merged.sort_values("PATNO").reset_index(drop=True)
baseline_merged["PATNO"] = baseline_merged["PATNO"].astype("Int64")

baseline_merged = baseline_merged.merge(nhy_latest,
                      how = 'left',
                      on = "PATNO",
                      suffixes=("", "_dup"))

baseline_merged = baseline_merged[baseline_merged["NHY"].notna() & (baseline_merged["NHY"] != 101)]

baseline_merged.to_csv("merged_all.csv",index=False)
