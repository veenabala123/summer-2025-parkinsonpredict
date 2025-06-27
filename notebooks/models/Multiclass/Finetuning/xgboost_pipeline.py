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
from sklearn.pipeline import Pipeline

sys.path.append("/Users/pushpita/Documents/Erdos_bootcamp/our_project/code_repo/src")  # adjust path as needed
from data_loader import LoadData
from model import MlModels
from evaluation import Evaluate

# --- Paths
mri_data_path = "/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/clean_data/PDMRI_Clean_Merged_6_13_25.csv"
gene_data_path = "/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/clean_data/gene_expression_summary.csv"
nhy_latest_path = "/Users/pushpita/Documents/Erdos_bootcamp/our_project/Data/clean_data/clean_mds_updrs.csv"

# --- Load data
dataload_all_cls = LoadData(
    input_updrs=nhy_latest_path,
    input_gene_clinical=gene_data_path,
    input_mri=mri_data_path,
    mri_data=True,
    gene_data=True,
    group_NHY=True,
    common_dataset=True,
    stratify_splits=True
)

X_data, Y_data = dataload_all_cls.merged_data()
X_train, Y_train, X_cv, Y_cv, X_test, Y_test = dataload_all_cls.data_split(X_data, Y_data)

# --- Compute sample weights
boost_class_weights = {0: 1.0, 1: 1.0, 2: 2.0}
Y_train_array = Y_train.values
sample_weight_array = np.array([boost_class_weights[y] for y in Y_train_array])

# --- Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('xgb', XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss'))
])

# --- Grid for both PCA and XGBoost
param_grid = {
    'pca__n_components': [50, 60, 70],
    'xgb__max_depth': [3, 4, 5],
    'xgb__learning_rate': [0.05, 0.06, 0.1],
    'xgb__n_estimators': [300, 800],
    'xgb__subsample': [0.4, 0.8],
    'xgb__reg_alpha': [1.5, 2.0],
    'xgb__reg_lambda': [1.0, 2.0]
}

# --- Grid search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=3,
    n_jobs=-1
)

# --- Fit with sample weights
grid_search.fit(X_train, Y_train_array, xgb__sample_weight=sample_weight_array)

# --- Best results
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# --- Evaluate on CV set
X_cv_scaled = StandardScaler().fit(X_train).transform(X_cv)
Y_cv_array = Y_cv.values

best_model = grid_search.best_estimator_
preds = best_model.predict(X_cv)

print("✅ Validation Accuracy:", accuracy_score(Y_cv_array, preds))
print("\n📋 Classification Report:")
print(classification_report(Y_cv_array, preds))
print("\n🧩 Confusion Matrix:")
print(confusion_matrix(Y_cv_array, preds))