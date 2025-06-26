import os, sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, chi2, SelectFromModel, RFE
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import matplotlib.pyplot as plt
import seaborn as sns

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
        
        sparsity = (self.X_data == 0).mean().mean()
        print(f"Sparsity: {sparsity:.2%}")
        
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
    
    def diagnostic_scores_vals(self):
        """
        Compute F-statistics and p-values for the selected genes using ANOVA F-test.

        Returns:
            pd.DataFrame: A DataFrame with columns ['gene', 'F_score', 'p_value'], 
                        sorted by F_score in descending order.
        """
        f_vals, p_vals = f_classif(self.X_data_f[self.selected_genes.tolist()], self.Y_data)
        f_scores_df = pd.DataFrame({
            "gene": self.selected_genes.tolist(),
            "F_score": f_vals,
            "p_value": p_vals
        })
        f_scores_df_sorted = f_scores_df.sort_values(by="F_score", ascending=False)
        return f_scores_df_sorted
    
    def plot_feature_importance(self, top_n=20, figsize=(10, 6)):
        """
        Plot top selected genes with their importance scores based on the selected feature selection method.

        Args:
            top_n (int): Number of top genes to display. Default is 20.
            figsize (tuple): Size of the plot.
        """
        if self.selected_genes is None:
            raise ValueError("Run select_features() before plotting.")

        plt.figure(figsize=figsize)

        if self.selection_method == 'rf':
            # For Random Forest, use feature importances directly
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            model.fit(self.X_data_f, self.Y_data)
            importances = model.feature_importances_
            top_k_idx = importances.argsort()[::-1][:top_n]
            gene_names = self.X_data_f.columns[top_k_idx]
            scores = importances[top_k_idx]

            plot_df = pd.DataFrame({"gene": gene_names, "importance": scores})
            sns.barplot(data=plot_df, x="importance", y="gene", palette="viridis")
            plt.title(f"Top {top_n} Genes by Random Forest Importance")

        elif self.selection_method == 'kbest':
            # For SelectKBest, compute F-statistics and use them
            f_vals, _ = f_classif(self.X_data_f[self.selected_genes], self.Y_data)
            plot_df = pd.DataFrame({
                "gene": self.selected_genes,
                "F_score": f_vals
            }).sort_values(by="F_score", ascending=False).head(top_n)

            sns.barplot(data=plot_df, x="F_score", y="gene", palette="Blues_r")
            plt.title(f"Top {top_n} Genes by F-statistic (SelectKBest)")

        elif self.selection_method in ['lasso', 'rfe']:
            if hasattr(self.selector, 'estimator_'):
                coefs = self.selector.estimator_.coef_[0]
            elif hasattr(self.selector, 'coef_'):
                coefs = self.selector.coef_[0]
            else:
                raise AttributeError("Lasso/RFE model coefficients not found.")
            
            gene_names = self.selected_genes
            plot_df = pd.DataFrame({
                "gene": gene_names,
                "coef": coefs
            }).sort_values(by="coef", key=abs, ascending=False).head(top_n)

            sns.barplot(data=plot_df, x="coef", y="gene", palette="coolwarm")
            plt.title(f"Top {top_n} Genes by Coefficient Magnitude ({self.selection_method})")

        else:
            raise NotImplementedError(f"Visualization not implemented for method: {self.selection_method}")

        plt.xlabel("Score")
        plt.ylabel("Gene")
        plt.tight_layout()
        plt.show()
        
    def plot_top_genes_lollipop(self, top_n=20):
        """
        Plot a ranked lollipop chart of top N genes selected by SelectKBest (F-test).

        Args:
            top_n (int): Number of top genes to display.
        """
        if self.selection_method != 'kbest':
            raise NotImplementedError("Lollipop plot only implemented for 'kbest' method.")

        f_vals, _ = f_classif(self.X_data_f[self.selected_genes], self.Y_data)
        plot_df = pd.DataFrame({
            "gene": self.selected_genes,
            "F_score": f_vals
        }).sort_values(by="F_score", ascending=False).head(top_n)

        plt.figure(figsize=(10, 6))
        plt.hlines(y=plot_df["gene"], xmin=0, xmax=plot_df["F_score"], color="skyblue")
        plt.plot(plot_df["F_score"], plot_df["gene"], "o", color="steelblue")
        plt.xlabel("F-statistic")
        plt.ylabel("Gene")
        plt.title(f"Top {top_n} Genes by F-statistic (SelectKBest)")
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()