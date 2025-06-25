import os,sys
import math

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mlt
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation
from functools import reduce

import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score, KFold, StratifiedKFold,cross_val_predict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

sys.path.append("/Users/pushpita/Documents/Erdos_bootcamp/our_project/code_repo/src")  # adjust path as needed
from read_Parkinsonpredict import ReadData, LoadData
from ml_models import MlModels

# --- Load data
mri_data = pd.read_csv("/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/finalised_dataset/PDMRI_Clean_Merged_6_13_25.csv")
gene_data = pd.read_csv("/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/finalised_dataset/gene_expression_summary.csv")
nhy_latest = pd.read_csv("/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/finalised_dataset/clean_mds_updrs.csv")

mri_data_bl  = mri_data.query("EVENT_ID == 'BL'")
gene_data_bl  = gene_data.query("EVENT_ID == 'BL'")
gene_data_bl = gene_data_bl[gene_data_bl["AGE"].notna()].copy()

gene_data_bl = gene_data_bl.rename(columns={"NHY": "NHY_BL"})

baseline_dfs = [mri_data_bl,gene_data_bl]

baseline_merged = reduce(
    lambda left, right: pd.merge(
        left, right,
        on=["PATNO", "EVENT_ID"],   
        how="inner",                
        suffixes=("", "_dup")       
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

mri_drop_f = ["lh_MeanThickness", "lh_WhiteSurfArea", "rhCerebralWhiteMatterVol", "lhCerebralWhiteMatterVol"]
mri_drop_list = mri_drop_f + ["EVENT_ID", "NHY_BL", "NHY", 'PATNO', 'MRIRSLT']
X_data = baseline_merged.drop(columns=mri_drop_list)

X_data['GENDER'] = X_data['GENDER'].map({"Female": 1, "Male": 0})
Y_data = baseline_merged["NHY"].copy()
Y_data[Y_data == 0] = 0
Y_data[(Y_data == 1) | (Y_data == 2)] = 1
Y_data[Y_data > 2] = 2

print(X_data.shape, Y_data.shape)

X_, X_test, Y_, Y_test = train_test_split(X_data, Y_data, test_size=0.2, stratify=Y_data, shuffle=True, random_state=42)
X_train, X_cv, Y_train, Y_cv = train_test_split(X_, Y_, test_size=0.15, stratify=Y_, shuffle=True, random_state=42)

# Scale training features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
Y_train_array = Y_train.values  # Ensure it's a NumPy array for XGBoost

# Define parameter grid
param_grid = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "n_estimators": [100, 300, 800],
    "subsample": [0.5, 0.8, 1.0],
    "reg_alpha": [0, 0.5, 1.0],   # L1
    "reg_lambda": [0.5, 1.0, 2.0]  # L2
}

# Set up XGBClassifier
xgb_clf = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Perform grid search
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Fit
grid_search.fit(X_train_scaled, Y_train_array)

# Print results
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)