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
from data_loader import LoadData
from model import MlModels
from evaluation import Evaluate

# --- Load data
mri_data_path = "/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/clean_data/PDMRI_Clean_Merged_6_13_25.csv"
gene_data_path = "/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/clean_data/gene_expression_summary.csv"
nhy_latest_path = "/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/clean_data/clean_mds_updrs.csv"

dataload_all_cls = LoadData(input_updrs=nhy_latest_path,
    input_gene_clinical=gene_data_path,
    input_mri=mri_data_path,
    mri_data=True,
    gene_data=True,
    group_NHY=True,
    common_dataset=True,
    stratify_splits=True)

X_data, Y_data = dataload_all_cls.merged_data()
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = dataload_all_cls.data_split(X_data, Y_data)

# --- Scale training features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
Y_train_array = Y_train.values

# --- Boosted class weights
boost_class_weights = {0: 1.0, 1: 1.0, 2: 2.0}
sample_weight_array = np.array([boost_class_weights[y] for y in Y_train_array])

# --- Parameter grid
param_grid = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.05, 0.06, 0.1, 0.2],
    "n_estimators": [100, 300, 800, 900],
    "subsample": [0.3, 0.4, 0.8, 1.0],
    "reg_alpha": [0, 0.5, 1.0, 1.5, 2.0],   # L1
    "reg_lambda": [0.5, 1.0, 2.0, 2.4, 2.8]  # L2
}

# --- XGBoost model
xgb_clf = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss'
)

# --- Grid search
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=3,
    n_jobs=-1
)

# --- Fit with sample weights
grid_search.fit(X_train_scaled, Y_train_array, sample_weight=sample_weight_array)

# --- Results
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# --- Rebuild best model with randomness fixed
best_params = grid_search.best_params_
best_params.update({
    "objective": "multi:softprob",
    "num_class": 3,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "random_state": 42
})

# --- Scale CV set
X_cv_scaled = scaler.transform(X_cv)
Y_cv_array = Y_cv.values

# --- Retrain
best_model = XGBClassifier(**best_params)
best_model.fit(X_train_scaled, Y_train_array)

# --- Evaluate
preds = best_model.predict(X_cv_scaled)
print("✅ Validation Accuracy:", accuracy_score(Y_cv_array, preds))
print("\n📋 Classification Report:")
print(classification_report(Y_cv_array, preds))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(Y_cv_array, preds))