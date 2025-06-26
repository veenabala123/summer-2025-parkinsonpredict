"""
gene_matrix.py

Making a gene matrix from the counts with 50000 genes
Retaining all the genes and selecting 100 individuals each from different NHY group

Author: Pushpita Das
Date: June 2025
"""

import os,sys
import math

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
import seaborn as sns
from tqdm import tqdm


# === Path setup ===
path = '/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/finalised_dataset_updated/'
HYS_data_path = '/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/Diagnosis_History_UPDRS_HYS'
input_datapath = '/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/'
output_datapath = '/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/fomatted_data/Updated/'

# === Load data ===
data_gene = pd.read_csv(input_datapath + '/Gene_expression/metaDataIR3.csv')
HYS_data = pd.read_csv(HYS_data_path+"/MDS-UPDRS_Part_III_03Jun2025.csv")
gene_expression_path = input_datapath + "/Gene_expression/quant/"

# === Step 1: Filter gene metadata for baseline and selected diagnoses ===
diagnosis_filter = ["PD", "Control", "Prodromal", "Genetic Cohort"]
data_gene_BL = data_gene[
    (data_gene["DIAGNOSIS"].isin(diagnosis_filter)) &
    (data_gene["CLINICAL_EVENT"] == 'BL')
].copy()

# === Step 2: Keep only PATNOs with valid salmon gene files ===
def extract_patno(filename):
    parts = filename.split('.')
    return int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else None

files = sorted([
    f for f in os.listdir(gene_expression_path)
    if (f.startswith('PPMI-Phase1-IR3.') or f.startswith('PPMI-Phase2-IR3.')) and f.endswith('salmon.genes.sf')
])
valid_patnos_from_files = {extract_patno(f) for f in files if extract_patno(f) is not None}
data_gene_BL = data_gene_BL[data_gene_BL["PATNO"].isin(valid_patnos_from_files)].copy()


# === Step 4: Final matched subset between gene and HYS ===
common_patnos = set(data_gene_BL["PATNO"]) & set(HYS_data["PATNO"])
gene_matched = data_gene_BL[data_gene_BL["PATNO"].isin(common_patnos)].copy()
HYS_matched = HYS_data[HYS_data["PATNO"].isin(common_patnos)].copy()

# === Reporting shapes ===
print(f"Original genomic metadata: {data_gene.shape}")
print(f"Filtered baseline gene data: {data_gene_BL.shape}")
print(f"Matched gene data: {gene_matched.shape}")
print(f"Matched HYS data: {HYS_matched.shape}")

# === Sort NHY visits and get first/last visit per patient ===
HYS_matched['LAST_UPDATE'] = pd.to_datetime(HYS_matched['LAST_UPDATE'], errors='coerce')
HYS_sorted = HYS_matched.sort_values(by=['PATNO', 'LAST_UPDATE']).reset_index(drop=True)
HYS_filtered = HYS_sorted[HYS_sorted["NHY"] <= 5]
HYS_last = HYS_filtered.groupby('PATNO').tail(1).reset_index(drop=True)
HYS_first = HYS_filtered.groupby('PATNO').head(1).reset_index(drop=True)


# === Select three classes of patients based on NHY severity ===
patno_nhy_large = HYS_last[HYS_last["NHY"] > 2]
selected_patnos1 = patno_nhy_large['PATNO'].sample(n=100, random_state=42)
patno_nhy_med = HYS_last[(HYS_last["NHY"] == 1) | (HYS_last["NHY"] == 2)]
selected_patnos2 = patno_nhy_med['PATNO'].sample(n=100, random_state=42)
patno_nhy_small = HYS_last[HYS_last["NHY"] == 0]
selected_patnos3 = patno_nhy_small['PATNO'].sample(n=100, random_state=42)

patno_set = pd.concat([selected_patnos1, selected_patnos2, selected_patnos3]).reset_index(drop=True)
valid_keys_patno = set(patno_set)


# === Use first file to get gene ensemble IDs (column names) ===
first_file = files[0]
data_genecounts_first = pd.read_csv(os.path.join(gene_expression_path, first_file), sep='\t')
gene_ensemble_IDs = data_genecounts_first["Name"].str.split('.').str[0]
all_column = ["PATNO", "EVENT_ID", "NHY"] + gene_ensemble_IDs.tolist()

# === Initialize final gene matrix ===
gene_matrix = pd.DataFrame(columns=all_column)

# === Main loop: Load TPM values for selected PATNOs ===
for filename in tqdm(files, desc="Processing files"):
    parts = filename.split('.')
    patno_str = parts[1].strip()
    if patno_str.isdigit():
        patno = int(patno_str)
    else:
        continue
    
    event_ids = parts[2].strip()
    if patno not in valid_keys_patno:
        continue
    
    nhy_row = HYS_last.loc[HYS_last["PATNO"]==patno, "NHY"]
    nhy = nhy_row.values[0] if not nhy_row.empty else -1
    
    my_row = dict.fromkeys(all_column, 0.0)
    my_row.update({"PATNO": patno, "EVENT_ID": event_ids, "NHY": nhy})
    
    fullname = os.path.join(gene_expression_path, filename)
    data_genecounts = pd.read_csv(fullname, sep='\t')
    data_genecounts["gene_base"] = data_genecounts["Name"].str.split('.').str[0]
    gene_TPM_map = dict(zip(data_genecounts["gene_base"], data_genecounts["TPM"]))

    # gene_TPM_map = dict(zip(data_genecounts["Name"].str.split('.').str[0], data_genecounts["TPM"]))

    # Fill gene values for this row
    for gene_id in gene_ensemble_IDs:
        my_row[gene_id] = gene_TPM_map.get(gene_id, 0.0)  # default to 0 if missing

    gene_matrix = pd.concat([gene_matrix, pd.DataFrame([my_row], columns=gene_matrix.columns)], ignore_index=True)
print("✓ Done")

# === Save matrix ===
output_path = os.path.join(output_datapath, "gene_matrix_genemeta_MDS-UPDRS.csv")
gene_matrix.to_csv(output_path, index=False)