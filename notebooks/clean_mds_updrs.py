import numpy as np
import pandas as pd

df = pd.read_csv("MDS-UPDRS_Part_III_PATNO_NHY_ORIGENTRY_02Jun2025.csv",
                 na_values=["", " ", "N/A"])

df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
df = df.dropna()
df = df[~(df == 101).any(axis=1)]

df["ORIG_ENTRY"] = pd.to_datetime(df["ORIG_ENTRY"], format="%m/%Y")
df = df.sort_values(["PATNO", "ORIG_ENTRY"])

result = (
    df.groupby("PATNO", group_keys=False)      # keep each PATNO’s rows together
      .apply(lambda g: g.iloc[[0, -1]])       # first (0) and last (-1) row
)

result.to_csv("clean_mds_updrs.csv", index=False)
print(result)