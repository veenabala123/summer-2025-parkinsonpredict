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

class Feature_selection:
    def __init__(self, data, grouped_y,
                selection_method='kbest', 
                k = 100, alpha=0.05, 
                random_state=42,
                **kwargs):

        self.selection_method = selection_method
        self.k = k
        self.alpha = alpha
        self.random_state = random_state
        self.selected_genes = None
        self.selector = None
        self.X_data = data.drop(columns=["PATNO", "EVENT_ID", "NHY"])
        self.Y_data = data["NHY"]
        self.grouped_y = grouped_y
        
        if self.grouped_y:
            Y_data_grouped = self.Y_data.copy()
            Y_data_grouped[Y_data_grouped == 0] = 0
            Y_data_grouped[(Y_data_grouped == 1) | (Y_data_grouped == 2)] = 1
            Y_data_grouped[Y_data_grouped > 2] = 2
        
        # X_data is (n_samples, n_genes), entries are TPM counts
        threshold_count = 5
        threshold_fraction = 0.10

        gene_mask = (self.X_data > threshold_count).sum(axis=0) >= (threshold_fraction * self.X_data.shape[0])
        X_filtered = self.X_data.loc[:, gene_mask]
        
        # Step 2: Remove low-variance genes
        var_thresh = VarianceThreshold(threshold=1e-5)
        self.X_data_f = pd.DataFrame(
            var_thresh.fit_transform(X_filtered),
            columns=X_filtered.columns[var_thresh.get_support()],
            index=X_filtered.index
        )
    
    def select_features(self):
        if self.selection_method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=self.k)
            
        elif self.selection_method == 'lasso':
            model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0/self.alpha)
            selector = SelectFromModel(model)

        elif self.selection_method == 'rfe':
            model = LogisticRegression(max_iter=1000)
            selector = RFE(model, n_features_to_select=self.k)

        elif self.selection_method == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            model.fit(self.X_data_f, self.Y_data)
            importances = model.feature_importances_
            top_k_idx = importances.argsort()[::-1][:self.k]
            self.selected_genes = self.X_data_f.columns[top_k_idx]
            return self.X_data_f.iloc[:, top_k_idx]  # skip selector object

        else:
            raise ValueError(f"Unknown feature selection method: {self.selection_method}")
        
        # Fit and get selected gene names
        selector.fit(self.X_data_f, self.Y_data)
        self.selector = selector
        selected_mask = selector.get_support()
        self.selected_genes = self.X_data_f.columns[selected_mask]
        return self.selected_genes.tolist()

class MlModels:
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

    def build_model(self):

        #First apply the scaling
        self.X_train_scaled, self.Y_train = self.scaled_input_data()
        
        if self.model_name == 'logistic':
            model = LogisticRegression(solver='lbfgs', max_iter=1000, **self.params)
        elif self.model_name == 'logistic_cv':
            model = LogisticRegressionCV(multi_class='ovr', solver='lbfgs', max_iter=1000, **self.params)
        elif self.model_name == 'xgboost':
            model = XGBClassifier(**self.params)
        elif self.model_name == 'neural_networks':
            model = neural_nets(**self.params)
        else:
            raise ValueError("Unsupported model type")

        if self.model_name == 'xgboost' and self.sample_weight:
            # Step 1: Get balanced sample weights
            sample_weight = compute_sample_weight(class_weight='balanced', y=self.Y_train)

            # Step 2: Boost specified classes (e.g., class 2 by 2x)
            if self.boost_class_weights:
                boost_array = np.array([
                    self.boost_class_weights.get(label, 1.0) for label in self.Y_train
                ])
                sample_weight = sample_weight * boost_array

            model.fit(self.X_train_scaled, self.Y_train, sample_weight=sample_weight)
        else:
            model.fit(self.X_train_scaled, self.Y_train)
        self.model = model
        return model        
    
    # def neural_nets(self):
        
    
    def evaluate_model(self, cv_folds, scoring='accuracy'):
        
        if self.model is None:
            raise ValueError("Model is not instantiated or trained")
        
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, self.X_train_scaled, self.Y_train, cv=cv, scoring=scoring)
        print(f"Cross-validation {scoring} scores: {scores}")
        print(f"Mean {scoring}: {np.mean(scores)}")
        return scores
    
    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been built or trained yet.")
        X_test_scaled = self.scaler.transform(X_test)
        if (self.pca_flag):
            X_test_pca = self.pca.transform(X_test_scaled)
            return self.model.predict(X_test_pca)
        else:
            return self.model.predict(X_test_scaled)