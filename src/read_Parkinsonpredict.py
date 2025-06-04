import os, sys
import numpy as np
import pandas as pd


class read_data:
    
    def __init__(self, data_path, output_path, debugging=0, *args, **kwargs):
        
        self.input_datapath = data_path
        self.output_datapath = output_path
        self.debug = debugging
    
    def gene_expression(self):
        """Reads the gene expression data and reconstructs it in a 
        readable format. 
        currently loads only one datapoint per sample, loading the early visit.
        """
        HYS_data_path = self.input_datapath + 'Diagnosis_History_UPDRS_HYS/'
        HYS_data = pd.read_csv(HYS_data_path+"MDS-UPDRS_Part_III_03Jun2025.csv")
        
        gene_metadata = pd.read_csv(self.input_datapath+'/Gene_expression/metaDataIR3.csv')
        gene_data_diagnosis = gene_metadata[gene_metadata["DIAGNOSIS"].isin(["PD", "Control", "Prodromal"])]
        gene_metadata_renamed = gene_data_diagnosis.rename(columns={'CLINICAL_EVENT': 'EVENT_ID'})

        # Keep only common PATNO and EVENT_ID combinations
        common_keys = pd.merge(
            HYS_data[['PATNO', 'EVENT_ID']],
            gene_metadata_renamed[['PATNO', 'EVENT_ID']],
            on=['PATNO', 'EVENT_ID']
        )

        # Now merge back to get the matched rows from each original dataframe
        gene_matched = gene_data_diagnosis.merge(common_keys, left_on=['PATNO', 'CLINICAL_EVENT'], right_on=['PATNO', 'EVENT_ID'])
        HYS_matched = HYS_data.merge(common_keys, on=['PATNO', 'EVENT_ID'])
        
        valid_keys = set(zip(gene_matched["PATNO"].astype(str), gene_matched["EVENT_ID"]))
        
        gene_expression_path = self.input_datapath + "/Gene_expression/quant/"
        files = sorted([f for f in os.listdir(gene_expression_path) if f.startswith('PPMI-Phase1-IR3.') and f.endswith('.sf')])
        
        if self.debug:
            print("HYS data shape:", HYS_data.shape)
            print("Gene metadata shape:", gene_data_diagnosis.shape)
            print("Common PATNOs found:", len(common_keys))
            print("First few matched PATNOs:", list(common_keys)[:5])
            print("Example gene file:", files[0])
            print("Matched gene samples:", gene_matched.shape)
            print("Matched HYS samples:", HYS_matched.shape)
            return gene_matched, HYS_matched
        
        # for i in range(files):
        #     parts = i.split('.')
        #     patno = parts[1]
        #     event_ids = parts[2]
        #     if (patno, event_ids) in valid_keys:
                
        
        
        
        
        
        return gene_matched, HYS_matched