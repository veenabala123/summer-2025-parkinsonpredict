import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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

            df = df.drop(columns=df.columns.difference(self.columns_kept))

            df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
            df = df.dropna()
            result = df[~(df == 101).any(axis=1)]
            if self.output_path:
                result.to_csv(os.path.join(self.output_path,"clean_mds_updrs.csv"), index=False)
            return result
    
class DataSplitter:
    def __init__(self,input_path_updrs, input_path_gene_clinical, input_clinical, test_size=0.2, write_csv=False, output_path=None, group_NHY=True, 
                demographic_data=True, clinical_data=True, gene_data=True):
        self.input_path_updrs = input_path_updrs
        self.input_path_gene_clinical = input_path_gene_clinical 
        self.write_csv = write_csv
        self.output_path = output_path
        self.test_size = test_size
        self.group_NHY = group_NHY
        self.demographic_data = demographic_data
        self.clinical_data = clinical_data
        self.gene_data = gene_data

    def merged_data(self):
        # --- Load data
        mri_data = pd.read_csv(self.input_path_gene_clinical + "PDMRI_Clean_Merged_6_13_25.csv")
        gene_data = pd.read_csv(self.input_path_gene_clinical + "gene_expression_summary.csv")
        nhy_latest = pd.read_csv(self.input_path_updrs + "clean_mds_updrs.csv")
        
        
        return X_data, Y_data
    
    def data_split(self):
        
        return X_train, Y_train, X_cv, Y_cv















    def merged_data(self):
        data_full = pd.read_csv(self.input_path)
        
        # Check if data is sorted by PATNO
        is_sorted = data_full['PATNO'].is_monotonic_increasing
        print("Is PATNO sorted?:", is_sorted)
        
        # Check for unique patients
        unique_patients = data_full['PATNO'].nunique() == len(data_full)
        print("Do we have all unique patients?:", "Yes" if unique_patients else "No")
        
        # Split features and target
        X = data_full.drop(columns=['NHY_BL', 'NHY']) # NHY is the targer and MRIRSLT is normal/abnormal and clinically significant
        Y_target = data_full['NHY']
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y_target, test_size=self.test_size, shuffle=False)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y_target, test_size=self.test_size, shuffle=True, random_state=42)
        print(f"The train and test data is split according to {(1-0.2)* 100} - {(0.2)* 100}%")
        
        if (self.write_csv and self.output_path):
            train_data = X_train.copy()
            train_data['NHY'] = Y_train
            train_data.to_csv(os.path.join(self.output_path, "train_set.csv"), index=False)

            test_data = X_test.copy()
            test_data['NHY'] = Y_test
            test_data.to_csv(os.path.join(self.output_path, "test_set.csv"), index=False)

        # Drop ID/meta columns from features before returning
        X_train_cleaned = X_train.drop(columns=['PATNO', 'EVENT_ID', 'MRIRSLT'])
        if (self.group_NHY):
            print('The data is further grouped in 0 (Control), 1 (NHY = 1 and 2) and 2 (NHY > 2)')
            Y_train_grouped = Y_train.copy()
            Y_train_grouped[Y_train_grouped == 0] = 0
            Y_train_grouped[(Y_train_grouped == 1) | (Y_train_grouped == 2)] = 1
            Y_train_grouped[Y_train_grouped > 2] = 2
            return X_train_cleaned, Y_train_grouped.to_frame(name='NHY')
        else:
            return X_train_cleaned, Y_train.to_frame(name='NHY')
        