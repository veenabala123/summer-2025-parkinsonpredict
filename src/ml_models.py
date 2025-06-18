import os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, KFold

class MlModels:
    def __init__(self, model_name, data, params, pca_flag=True, pca_components = 10):

        self.model_name = model_name
        self.params = params or {}
        self.model = None
        self.X_train = data['X_train']
        self.Y_train = data['Y_train'].values.ravel()   #the models expect (n_samples, ) not (n_samples, 1)
        self.pca = None
        self.pca_components = pca_components
        self.pca_flag = pca_flag
        self.scaler = None
    
    def build_model(self):
        
        if self.model_name == 'logistic':
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs', **self.params)
        elif self.model_name == 'logistic_cv':
            model = LogisticRegressionCV(multi_class='multinomial', solver='lbfgs', **self.params)
        elif self.model_name == 'xgboost':
            model = XGBClassifier(**self.params)
        else:
            raise ValueError("Unsupported model type")
        
        if self.pca_flag:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(self.X_train)
            self.pca = PCA(n_components=self.pca_components)
            X_pca = self.pca.fit_transform(X_scaled)
            model.fit(X_pca, self.Y_train)
        else:
            model.fit(self.X_train, self.Y_train)

        self.model = model
        return model
    
    def evaluate_model(self, cv_folds, scoring='accuracy'):
        
        if self.model is None:
            raise ValueError("Model is not instantiated or trained")
        
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, self.X_train, self.Y_train, cv=cv, scoring=scoring)
        print(f"Cross-validation {scoring} scores: {scores}")
        print(f"Mean {scoring}: {np.mean(scores)}")
        return scores
    
    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        if (self.pca_flag):
            X_test_scaled = self.scaler.transform(X_test)
            X_test_pca = self.pca.transform(X_test_scaled)
            return self.model.predict(X_test_pca)
        else:
            return self.model.predict(X_test)