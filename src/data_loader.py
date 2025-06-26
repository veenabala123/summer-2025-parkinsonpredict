"""
data_loader.py

Data processing and loading classes.

Author: Pushpita Das
        Also assimilated Veena's script to merge the datafiles
Date: June 2025
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import reduce

class GeneDataReader:
    
    def __init__(self, data_path, output_path, debugging=0, gene_list=None, write_csv = 0, phaseIandII=1, genetic_cohort=1,
                 clinical_data=1, match_event=True, *args, **kwargs):
        
        self.input_datapath = data_path
        self.output_datapath = output_path
        self.debug = debugging
        self.write_csv = write_csv
        self.phaseIandII = phaseIandII
        self.genetic_cohort = genetic_cohort
        self.clinical_data = clinical_data
        self.gene_list = gene_list or [
            "ENSG00000188906", "ENSG00000185345", "ENSG00000158828", "ENSG00000145335", "ENSG00000159082", 
            "ENSG00000116288", "ENSG00000116675", "ENSG00000177628", "ENSG00000100225", "ENSG00000184381"
        ]
        self.match_event = match_event
        
    def HYS_data_filtered(self):
        """Read the gene expression metadata and also the HYS metadata file

        Returns:
            Filtered gene_expression metadata file and also HYS_metadata file
        """
        #Step 1: Load the data
        HYS_data_path = self.input_datapath + 'Diagnosis_History_UPDRS_HYS/'
        HYS_data = pd.read_csv(HYS_data_path+"MDS-UPDRS_Part_III_03Jun2025.csv")
        gene_metadata = pd.read_csv(self.input_datapath+'/Gene_expression/metaDataIR3.csv')

        #Step 2: From the MDS file choose only the patients in the following groups
        diagnosis_filter = ["PD", "Control", "Prodromal"]
        if self.genetic_cohort:
            diagnosis_filter.append("Genetic Cohort")
            
        gene_data_diagnosis = gene_metadata[gene_metadata["DIAGNOSIS"].isin(diagnosis_filter)]
        
        if self.match_event:
            
            gene_metadata_renamed = gene_data_diagnosis.rename(columns={'CLINICAL_EVENT': 'EVENT_ID'})
            #Step 3: Load the common (PATIENT_ID, EVENT_ID) between the MDS file and the gene metadata file
            # Keep only common PATNO and EVENT_ID combinations
            common_keys = pd.merge(
                HYS_data[['PATNO', 'EVENT_ID']],
                gene_metadata_renamed[['PATNO', 'EVENT_ID']],
                on=['PATNO', 'EVENT_ID']
            )

            # Step 4: Now merge back to get the matched rows from each original dataframe
            gene_matched = gene_data_diagnosis.merge(common_keys, left_on=['PATNO', 'CLINICAL_EVENT'], right_on=['PATNO', 'EVENT_ID'])
            HYS_matched = HYS_data.merge(common_keys, on=['PATNO', 'EVENT_ID'])
        else:
            common_patnos = set(HYS_data['PATNO']) & set(gene_data_diagnosis['PATNO'])
            HYS_matched = HYS_data[HYS_data['PATNO'].isin(common_patnos)].copy()
            gene_matched = gene_data_diagnosis[gene_data_diagnosis['PATNO'].isin(common_patnos)].copy()
        
        if self.debug:
            print("HYS data shape:", HYS_data.shape)
            print("Gene metadata shape:", gene_data_diagnosis.shape)
            print("Common PATNOs found:", len(common_keys))
            print("First few matched PATNOs:", list(common_keys)[:5])
            print("Matched gene samples:", gene_matched.shape)
            print("Matched HYS samples:", HYS_matched.shape)
            return gene_matched, HYS_matched
        else:
            return gene_matched, HYS_matched
        
        
    def load_gene_expression(self):
        """Reads the gene expression data and reconstructs it in a 
        readable format. 
        currently loads only one datapoint per sample, loading the early visit.
        """
        gene_matched, HYS_matched = self.HYS_data_filtered()
        
        if self.match_event:
            valid_keys = set(zip(gene_matched["PATNO"], gene_matched["EVENT_ID"].str.strip()))
        else:
            valid_keys = set(gene_matched["PATNO"])
        
        #Step 6: Make a list of all the gene expression files
        gene_expression_path = self.input_datapath + "/Gene_expression/quant/"
        if (self.phaseIandII):
            files = sorted([f for f in os.listdir(gene_expression_path) if 
                            (f.startswith('PPMI-Phase1-IR3.') or f.startswith('PPMI-Phase2-IR3.')) and 
                            f.endswith('salmon.genes.sf')])
        else:
            files = sorted([f for f in os.listdir(gene_expression_path) if 
                            f.startswith('PPMI-Phase1-IR3.') and f.endswith('salmon.genes.sf')])
        #Step 7: Make a empty pandas dataframe with column names
        if self.clinical_data:
            data_age = pd.read_csv(self.input_datapath+'/Subject_Demographics/Age_at_visit_22Jun2025.csv')
            educ_years = pd.read_csv(self.input_datapath+'/Subject_Demographics/Socio-Economics_22Jun2025.csv')
            all_column = ["PATNO", "EVENT_ID", "GENDER", "AGE", "EDUC_YRS"] + self.gene_list + ["NHY"]
        else:
            all_column = ["PATNO", "EVENT_ID"] + self.gene_list + ["NHY"]

        gene_counts_df = pd.DataFrame(columns=all_column)
        print(len(files))
        matched_count = 0  # initialize a counter
        unmatched_keys = []
        
        #Step 8: Loop through all the files
        for filename in files:
            parts = filename.split('.')
            patno_str = parts[1].strip()
            if patno_str.isdigit():
                patno = int(patno_str)
            else:
                unmatched_keys.append(filename)
                continue
            
            event_ids = parts[2].strip()
            if self.match_event:
            #Step 9: Only load the files with the common keys
                if (patno, event_ids) not in valid_keys:
                    unmatched_keys.append((patno, event_ids))
                    continue
            else:
                if patno not in valid_keys:
                    unmatched_keys.append((patno, event_ids))
                    continue
            matched_count += 1  # increment the counter
                
            #Step 10: Load the gene expression data file
            fullname = os.path.join(gene_expression_path, filename)
            data_genecounts = pd.read_csv(fullname, sep='\t')
            
            #Step 11: Loading the gene expression, stripping suffix to get the general code
            data_genecounts["geneID_base"] = data_genecounts["Name"].str.split('.').str[0]
            
            #Step 12: Dataset with only the genes in the list that we provided above
            gene_name_set = data_genecounts[data_genecounts["geneID_base"].isin(self.gene_list)]
            
            #Step 13: First load the NHY data
            if self.match_event:
                nhy_row = HYS_matched.loc[(HYS_matched["PATNO"]==patno) & 
                                        (HYS_matched["EVENT_ID"].str.strip()==event_ids), "NHY"]
            else:
                nhy_row = HYS_matched.loc[HYS_matched["PATNO"]==patno, "NHY"]
            
            nhy = nhy_row.values[0] if not nhy_row.empty else -1
            # if not nhy_row.empty:
            #     nhy = nhy_row.values[0]
            # else:
            #     print(f"No NHY for PATNO={patno}, EVENT_ID = {event_ids}")
            #     nhy = -1
                
            my_row = {"PATNO": patno, "EVENT_ID": event_ids, "NHY":nhy}
            
            if self.clinical_data:
                if self.match_event:
                    gender_row = gene_matched.loc[(gene_matched["PATNO"]==patno) & 
                                (gene_matched["EVENT_ID"].str.strip()==event_ids), "GENDER"]
                else:
                    gender_row = gene_matched.loc[gene_matched["PATNO"]==patno, "GENDER"]
                
                age_row = data_age.loc[(data_age["PATNO"]==patno) & 
                                    (data_age["EVENT_ID"].str.strip()=="BL"), "AGE_AT_VISIT"]
                educ_row = educ_years.loc[(educ_years["PATNO"]==patno), "EDUCYRS"]

                my_row["GENDER"] = gender_row.values[0] if not gender_row.empty else None
                my_row["AGE"] = age_row.values[0] if not age_row.empty else None
                my_row["EDUC_YRS"] = educ_row.values[0] if not educ_row.empty else None
            
            #Step 14: Loop through all the genenames in the list and assign TPM count 
            # for each gene in the list. Default to 0 if its missing
            for gene in self.gene_list:
                names_match = gene_name_set[gene_name_set["geneID_base"] == gene]
                if not names_match.empty:
                    my_row[gene] = names_match["TPM"].values[0]
                else:
                    my_row[gene] = 0
            
            #Step 15: Append as new row for each file (patient)
            gene_counts_df = pd.concat([gene_counts_df, pd.DataFrame([my_row], columns=gene_counts_df.columns)], ignore_index=True)
        
        if self.write_csv:
            # output_path = os.path.join(self.output_datapath, "gene_expression_summary.csv")
            output_path = os.path.join(self.output_datapath, "gene_expression_summary_50.csv")
            gene_counts_df.to_csv(output_path, index=False)

        print(f"Total matched gene files with valid PATNO/EVENT_ID pairs: {matched_count}")
        return gene_counts_df,unmatched_keys
    

class UPDRSReader:
    #Converted Veena's script into a class
    def __init__(self, file_path, output_path=None):
        """
        Parameters:
        - file_path: str, path to the raw UPDRS CSV file
        - output_path: str or None, if given, the cleaned data will be saved here
        """
        self.file_path = file_path
        self.output_path = output_path
        self.columns_kept = ["PATNO", "NHY", "EVENT_ID"]
        
        def clean_udprs(self):
            df = pd.read_csv(self.file_path, na_values=["", " ", "N/A"])

            df["latest_data"] = pd.to_datetime(df["ORIG_ENTRY"], format="%m/%Y")

            df = (df.sort_values("latest_data").drop_duplicates("PATNO", keep="last").sort_values("PATNO").reset_index(drop=True))

            columns_kept = ["PATNO","NHY","EVENT_ID"]
            df = df.drop(columns=df.columns.difference(self.columns_kept))
            df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
            df = df.dropna()
            result = df[~(df == 101).any(axis=1)]
            if self.output_path:
                result.to_csv(os.path.join(self.output_path,"clean_mds_updrs.csv"), index=False)
            return result
##Add the class for generating MRI

class LoadData:
    def __init__(self,input_updrs, input_gene_clinical, 
                input_mri, mri_data,
                mri_drop_list=None, group_NHY=True,
                group_binary=False,
                gene_data=True, common_dataset=True,
                test_size=0.2, validation_size = 0.15,
                write_csv=False, output_path=None,
                stratify_splits=False,
                print_column_names=False,
                *args, **kwargs):

        self.input_updrs = input_updrs
        self.group_NHY = group_NHY
        self.group_binary = group_binary
        self.mri_data = mri_data
        self.gene_data = gene_data
        self.input_gene_clinical = input_gene_clinical
        self.common_dataset = common_dataset
        self.input_mri = input_mri
        self.mri_drop_list = mri_drop_list or ["MRIRSLT", "lh_MeanThickness", 
                                        "lh_WhiteSurfArea", "rhCerebralWhiteMatterVol", 
                                        "lhCerebralWhiteMatterVol"]
        self.test_size = test_size
        self.validation_size = validation_size
        self.stratify_splits = stratify_splits
        self.write_csv = write_csv
        self.output_path = output_path
        self.print_cols = print_column_names


    def group_nhy(self, y):

        if self.group_binary:
            y = y.copy()
            y[y == 0] = 0
            y[y > 1] = 1
        else:
            y = y.copy()
            y[y == 0] = 0
            y[(y == 1) | (y == 2)] = 1
            y[y > 2] = 2
        return y.to_frame(name='NHY')

    def merged_data(self):
        # --- Load data
        gene_data = pd.read_csv(self.input_gene_clinical)
        nhy_latest = pd.read_csv(self.input_updrs)
        
        use_demographic = not self.mri_data and not self.gene_data and not self.common_dataset
        use_gene_only = self.gene_data and not self.mri_data and not self.common_dataset
        use_mri_only = self.mri_data and not self.gene_data and not self.common_dataset
        
        use_demographic_commondata = self.common_dataset and not self.mri_data and not self.gene_data
        use_gene_only_commondata = self.common_dataset and self.gene_data and not self.mri_data
        use_mri_only_commondata = self.common_dataset and self.mri_data and not self.gene_data
        
        use_all = self.mri_data and self.gene_data
        
        if use_demographic:
            clinical_data = gene_data[["PATNO", "EVENT_ID", "GENDER", "AGE", "EDUC_YRS"]]
            clinical_data_bl = clinical_data[clinical_data["EVENT_ID"] == "BL"]
            clinical_data_clean = clinical_data_bl[clinical_data_bl['AGE'].notna()]
        
            data_clean = clinical_data_clean.merge(nhy_latest, how='inner', on=["PATNO"])
            data_clean = data_clean[
                data_clean["NHY"].notna() & (data_clean["NHY"] != 101)
            ]
            X_data = data_clean.drop(columns=[
                col for col in data_clean.columns
                if col.startswith("EVENT_ID") or col in ["PATNO", "NHY"]
            ])
            X_data['GENDER'] = X_data['GENDER'].map({"Female": 1, "Male": 0})

            Y_data = data_clean["NHY"].copy()
            if self.group_NHY:
                Y_data = self.group_nhy(Y_data)
            print(f"X shape: {X_data.shape}, Y shape: {Y_data.shape}")
            print("Y class distribution:", Y_data.value_counts().to_dict())
            return X_data, Y_data

        elif use_mri_only:

            mri_data = pd.read_csv(self.input_mri)

            clinical_data = gene_data[["PATNO", "EVENT_ID", "GENDER", "AGE", "EDUC_YRS"]]
            clinical_data_bl = clinical_data[clinical_data["EVENT_ID"] == "BL"]
            clinical_data_clean = clinical_data_bl[clinical_data_bl['AGE'].notna()]

            clinical_bl_clean = clinical_data_clean.merge(nhy_latest, how='inner', on=["PATNO"])
            mri_data_clean = mri_data.merge(clinical_bl_clean, how='inner', on=["PATNO"])
            mri_data_clean = mri_data_clean[
                mri_data_clean["NHY"].notna() & (mri_data_clean["NHY"] != 101)
            ]
            mri_drop_base = ['PATNO', "NHY"] + self.mri_drop_list
            X_data_mri = mri_data_clean.drop(columns=[
                col for col in mri_data_clean.columns
                if col.startswith("EVENT_ID") 
                or col in mri_drop_base])
            X_data_mri['GENDER'] = X_data_mri['GENDER'].map({"Female": 1, "Male": 0})
            X_data = X_data_mri.copy()
            Y_data = mri_data_clean["NHY"].copy()
            if self.group_NHY:
                Y_data = self.group_nhy(Y_data)
            print(f"X shape: {X_data.shape}, Y shape: {Y_data.shape}")
            print("Y class distribution:", Y_data.value_counts().to_dict())
            return X_data, Y_data
        
        elif self.common_dataset:

            mri_data = pd.read_csv(self.input_mri)
        
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

            mri_drop_base = ['PATNO', "NHY", "NHY_BL"] + self.mri_drop_list
            X_data_all = baseline_merged.drop(columns=[
                col for col in baseline_merged.columns
                if col.startswith("EVENT_ID") 
                or col in mri_drop_base])
            
            X_data_all['GENDER'] = X_data_all['GENDER'].map({"Female": 1, "Male": 0})
            Y_data = baseline_merged["NHY"].copy()
            if self.group_NHY:
                Y_data = self.group_nhy(Y_data)
            print(f"X shape: {X_data_all.shape}, Y shape: {Y_data.shape}")
            print("Y class distribution:", Y_data.value_counts().to_dict())

            if use_demographic_commondata:
                X_data = X_data_all[["GENDER", "AGE", "EDUC_YRS"]]
                Y_data = Y_data
                if self.print_cols:
                    X_data.columns.to_series().to_csv("debug_columns_demographic.csv", index=False)
                return X_data, Y_data

            elif use_mri_only_commondata:
                mri_cols = [col for col in mri_data.columns
                            if col not in self.mri_drop_list 
                            and col not in ['PATNO', 'EVENT_ID']]
                print(mri_cols)
                columns_load = ["GENDER", "AGE", "EDUC_YRS"] + mri_cols
                X_data = X_data_all[columns_load]
                if self.print_cols:
                    X_data.columns.to_series().to_csv("debug_columns_mri.csv", index=False)
                return X_data, Y_data
                        
            elif use_all:
                X_data = X_data_all
                if self.print_cols:
                    X_data_all.columns.to_series().to_csv("debug_columns_all.csv", index=False)
                return X_data, Y_data

        elif use_gene_only:
            raise ValueError("Data merging for gene expression + demographic data not implemented")

        else:
            raise ValueError("Wrong Flags, check the doc-string and fix it first")
        

    def data_split(self, X_data, Y_data):
        """
        Splits X_data and Y_data into training, test and validation sets.

        Returns:
            X_train, Y_train, X_cv, Y_cv, X_test, Y_test
        """
        # Ensure Y_data is a Series
        if isinstance(Y_data, pd.DataFrame):
            assert Y_data.shape[1] == 1, "Expected Y_data to have only one column"
            Y_data = Y_data.iloc[:, 0]  # Convert to Series
        
        if (self.stratify_splits):
            
            assert Y_data.nunique() > 1, "Y_data has only one class — cannot stratify"
            assert not Y_data.isnull().any(), "Y_data contains NaNs — cannot stratify"
            
            X_, X_test, Y_, Y_test = train_test_split(X_data, Y_data, test_size=self.test_size, stratify=Y_data, shuffle=True, random_state=42)
            X_train, X_cv, Y_train, Y_cv = train_test_split(X_, Y_, test_size=self.validation_size, stratify=Y_, shuffle=True, random_state=42)
        else:
            X_, X_test, Y_, Y_test = train_test_split(X_data, Y_data, test_size=self.test_size, shuffle=True, random_state=42)
            X_train, X_cv, Y_train, Y_cv = train_test_split(X_, Y_, test_size=self.validation_size, shuffle=True, random_state=42)

        if (self.write_csv and self.output_path):
            train_data = X_.copy()
            train_data['NHY'] = Y_
            train_data.to_csv(os.path.join(self.output_path, "trainval_set.csv"), index=False)

            test_data = X_test.copy()
            test_data['NHY'] = Y_test
            test_data.to_csv(os.path.join(self.output_path, "test_set.csv"), index=False)

        return X_train, Y_train, X_cv, Y_cv, X_test, Y_test
