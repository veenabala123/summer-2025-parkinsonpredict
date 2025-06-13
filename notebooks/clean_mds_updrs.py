import numpy as np
import pandas as pd

df = pd.read_csv("MDS-UPDRS_Part_III_02Jun2025.csv",
                 na_values=["", " ", "N/A"])

columns_kept = ["PATNO","NHY","EVENT_ID"]
df = df.drop(columns=df.columns.difference(columns_kept))

df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
df = df.dropna()
result = df[~(df == 101).any(axis=1)]

result.to_csv("clean_mds_updrs.csv", index=False)
print(result)
