"""
evaluate.py

Evaluation utilities for trained models.

Author: Pushpita Das
Date: June 2025
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from tabulate import tabulate

class Evaluate:
    """
    Wraps evaluation utilities for a trained model.
    
    Args:
        model_obj: An instance of MlModels with .model, .scaler, etc.
        X_test (pd.DataFrame or np.ndarray): Test feature data.
        Y_test (pd.Series or np.ndarray): Test target labels.
        cv_folds (int): Number of cross-validation folds.
        scoring (str): Scoring metric for cross_val_score.
    """
    def __init__(self, model_obj,
                X_test, Y_test,
                cv_folds=5, 
                scoring='accuracy',
                 *args, **kwargs):
        
        self.model_obj = model_obj
        self.my_model = model_obj.model
        self.X_train_scaled = model_obj.X_train_scaled
        self.Y_train = model_obj.Y_train
        self.pca_comp = model_obj.pca
        self.pca_flag = model_obj.pca_flag
        self.scaler = model_obj.scaler
        self.x_test = X_test
        self.y_test = Y_test
        self.cv_folds=cv_folds
        self.scoring = scoring
    
    def evaluate_model(self):
        
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(self.my_model, self.X_train_scaled, 
                                self.Y_train, cv=cv, scoring=self.scoring)
        
        print(f"Cross-validation {self.scoring} scores: {scores}")
        print(f"Mean {self.scoring}: {np.mean(scores)}")
        return scores

    def predict(self, X):
        """Scale and predict on new data."""
        
        if self.model_obj is None:
            raise ValueError("Model has not been built or trained yet.")
        
        X_scaled = self.scaler.transform(X)
        if self.pca_flag and self.pca_comp is not None:
            X_scaled = self.pca_comp.transform(X_scaled)
        
        return self.my_model.predict(X_scaled)
        
    def predict_proba(self, X):
        if self.model_obj is None:
            raise ValueError("Model has not been built or trained yet.")
        if not hasattr(self.my_model, "predict_proba"):
            raise NotImplementedError("Model does not support probability prediction.")
        X_scaled = self.scaler.transform(X)
        if self.pca_flag and self.pca_comp is not None:
            X_scaled = self.pca_comp.transform(X_scaled)
        return self.my_model.predict_proba(X_scaled)
        
        
    def report_validation_metrics(self):
        
        predictions = self.predict(self.x_test)
        Y_true = self.y_test.values.ravel()
        
        acc_score = accuracy_score(Y_true, predictions)
        class_report = classification_report(Y_true, predictions, zero_division=0)
        confusion_ma = confusion_matrix(Y_true, predictions)

        class_report_dict = classification_report(Y_true, predictions, zero_division=0, output_dict=True)
        class_report_df = pd.DataFrame(class_report_dict).T.round(3)
        confusion_df = pd.DataFrame(confusion_ma)
        
        print(f"\n✅ Validation Accuracy: {acc_score:.4f}\n")
        print("📋 Classification Report:")
        print(tabulate(class_report_df, headers='keys', tablefmt='fancy_grid'))
        

        print("\n🧩 Confusion Matrix:")
        confusion_df = pd.DataFrame(confusion_ma)
        print(tabulate(confusion_df, tablefmt='fancy_grid', showindex="always"))
        return acc_score, class_report, confusion_ma
    
    def get_classification_report_df(self):
        
        preds = self.predict(self.x_test)
        y_true = self.y_test.values.ravel()
        report_dict = classification_report(y_true, preds, zero_division=0, output_dict=True)
        return pd.DataFrame(report_dict).T
    
    def get_confusion_matrix(self):
        preds = self.predict(self.x_test)
        y_true = self.y_test.values.ravel()
        return confusion_matrix(y_true, preds)


    def get_auc_scores(self, class_list=None):
        y_true = self.y_test.values.ravel()

        if not hasattr(self.my_model, "predict_proba"):
            raise NotImplementedError("Model does not support predict_proba needed for AUC.")

        # Get prediction probabilities
        y_score = self.predict_proba(self.x_test)

        # If class_list is not provided, infer it
        if class_list is None:
            class_list = np.unique(y_true).tolist()
        elif len(class_list) == 0:
            raise ValueError("Class list is empty, pass something!")

        n_classes = len(class_list)

        fpr, tpr, roc_auc = {}, {}, {}

        if n_classes == 2:
            # Binary classification
            fpr[1], tpr[1], _ = roc_curve(y_true, y_score[:, 1])
            roc_auc[1] = auc(fpr[1], tpr[1])
            auc_score = roc_auc_score(y_true, y_score[:, 1])
        else:
            # Multiclass classification
            y_true_bin = label_binarize(y_true, classes=class_list)

            if y_score.shape[1] != y_true_bin.shape[1]:
                raise ValueError("Mismatch in number of classes between prediction and ground truth.")

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            auc_score = roc_auc_score(y_true, y_score, multi_class='ovr')

        return {'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 'auc_score': auc_score}
