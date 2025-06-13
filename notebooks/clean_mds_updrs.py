import numpy as np
import pandas as pd

df = pd.read_csv("MDS-UPDRS_Part_III_PATNO_NHY_ORIGENTRY_02Jun2025.csv",
                 na_values=["", " ", "N/A"])

df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
df = df.dropna()
df = df[~(df == 101).any(axis=1)]

columns_kept = ["PATNO","NHY","Origin_of_entry","EVENT_ID"]
result = df.drop(columns=df.columns.difference(columns_kept))

result.to_csv("clean_mds_updrs.csv", index=False)
print(result)
