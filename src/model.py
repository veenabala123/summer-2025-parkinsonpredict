"""
model.py

Container for all the ML models

Author: Pushpita Das
Date: June 2025
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils.class_weight import compute_sample_weight

class MlModels:
    """
    A flexible machine learning model wrapper supporting logistic regression, XGBoost, and neural networks.
    
    Args:
        model_name (str): One of ['logistic', 'logistic_cv', 'xgboost', 'neural_networks'].
        data (dict): Dictionary with keys 'X_train' and 'Y_train'.
        params (dict): Model-specific hyperparameters.
        pca_flag (bool): Whether to apply PCA after scaling.
        pca_components (int): Number of PCA components.
        sample_weight (bool): Whether to use sample weighting.
        class_weights (dict or str): For logistic models.
        boost_class_weights (dict): Extra boost factor for specific labels in XGBoost.
    """
    def __init__(self, model_name, 
                    data, 
                    params, 
                    pca_flag=True, 
                    pca_components = 10,
                    sample_weight=False, 
                    class_weights=None,
                    boost_class_weights=None):

        self.model_name = model_name
        self.params = params or {}
        self.X_train = data['X_train']
        self.Y_train = data['Y_train'].values.ravel()   #the models expect (n_samples, ) not (n_samples, 1)
        self.model = None
        self.pca = None
        self.pca_components = pca_components
        self.pca_flag = pca_flag
        self.scaler = None
        self.sample_weight = sample_weight
        self.class_weights = class_weights
        self.boost_class_weights = boost_class_weights or {}

    def scaled_input_data(self):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X_train)
        if self.pca_flag:
            self.pca = PCA(n_components=self.pca_components)
            X_pca = self.pca.fit_transform(X_scaled)
            return X_pca, self.Y_train
        else:
            return X_scaled, self.Y_train
        
    def _custom_sample_weight(self):
        
        sample_weight = compute_sample_weight(class_weight='balanced', y=self.Y_train)
        
        if self.boost_class_weights:
            boost_array = np.array([
                self.boost_class_weights.get(label, 1.0) for 
                label in self.Y_train])
            
            sample_weight *= boost_array
        return sample_weight
    
    def build_model(self):

        #First apply the scaling
        self.X_train_scaled, self.Y_train = self.scaled_input_data()
        
        if self.model_name == 'logistic':
            model = LogisticRegression(solver='lbfgs', 
                                    max_iter=1000, 
                                    class_weight=self.class_weights,
                                       **self.params)
        elif self.model_name == 'logistic_cv':
            model = LogisticRegressionCV(multi_class='ovr', 
                                        solver='lbfgs', 
                                        max_iter=1000, 
                                        class_weight=self.class_weights,
                                        **self.params)
            
        elif self.model_name == 'xgboost':
            model = XGBClassifier(**self.params)

        elif self.model_name == 'neural_networks':
            try:
                model = neural_nets(**self.params)
            except NameError:
                raise NotImplementedError("Define `neural_nets` or remove 'neural_networks' option.")        
        else:
            raise ValueError("Unsupported model type")

        if self.model_name == 'xgboost' and self.sample_weight:
            sample_weight = self._custom_sample_weight()
            model.fit(self.X_train_scaled, self.Y_train, sample_weight=sample_weight)
        else:
            model.fit(self.X_train_scaled, self.Y_train)

        self.model = model
        return model        
    
    # def neural_nets(self):