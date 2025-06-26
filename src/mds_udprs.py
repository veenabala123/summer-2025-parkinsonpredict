import numpy as np
import pandas as pd

input_path = '/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/Diagnosis_History_UPDRS_HYS/'
data = pd.read_csv(input_path+'MDS-UPDRS_Part_III_03Jun2025.csv')

columns = ["PATNO","NHY","EVENT_ID", "LAST_UPDATE"]
data_filter = data[columns]

data_filter = data_filter.replace(r"^\s*$", np.nan, regex=True)
data_filter = data_filter.dropna()
data_clean = data_filter[~(data_filter == 101).any(axis=1)]


# === Sort NHY visits and get first/last visit per patient ===
data_clean.loc[:, 'LAST_UPDATE'] = pd.to_datetime(data_clean['LAST_UPDATE'], errors='coerce')
HYS_sorted = data_clean.sort_values(by=['PATNO', 'LAST_UPDATE']).reset_index(drop=True)
HYS_filtered = HYS_sorted[HYS_sorted["NHY"] <= 5]
HYS_last = HYS_filtered.groupby('PATNO').tail(1).reset_index(drop=True)
HYS_first = HYS_filtered.groupby('PATNO').head(1).reset_index(drop=True)

HYS_last.to_csv("clean_mds_updrs.csv", index=False)
print(HYS_last)