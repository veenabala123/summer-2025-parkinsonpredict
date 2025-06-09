import os, sys
import numpy as np
import pandas as pd


class read_data:
    
    def __init__(self, data_path, output_path, debugging=0, gene_list=None, *args, **kwargs):
        
        self.input_datapath = data_path
        self.output_datapath = output_path
        self.debug = debugging
        # If no gene list is provided, use the default top PD-related genes
        # Format: [("GeneSymbol", "EnsembleID", score), ...]
        # I am putting the following in the list:
        # LRRK2,ENSG00000188906, 125.94  
        # PRKN,ENSG00000185345, 106.89
        # PINK1,ENSG00000158828, 105.61
        # SNCA,ENSG00000145335, 99.47
        # SYNJ1, ENSG00000159082, 98.41
        # PARK7,ENSG00000116288, 86.53
        # DNAJC6,ENSG00000116675, 67.27
        # GBA1,ENSG00000177628, 65.3
        # FBXO7,ENSG00000100225, 61.16
        # PLA2G6,ENSG00000184381, 59.6
        self.gene_list = gene_list or [
            "ENSG00000188906", "ENSG00000185345", "ENSG00000158828", "ENSG00000145335", "ENSG00000159082", 
            "ENSG00000116288", "ENSG00000116675", "ENSG00000177628", "ENSG00000100225", "ENSG00000184381"
        ]
        
    def gene_expression(self):
        """Reads the gene expression data and reconstructs it in a 
        readable format. 
        currently loads only one datapoint per sample, loading the early visit.
        """
        HYS_data_path = self.input_datapath + 'Diagnosis_History_UPDRS_HYS/'
        HYS_data = pd.read_csv(HYS_data_path+"MDS-UPDRS_Part_III_03Jun2025.csv")
        
        gene_metadata = pd.read_csv(self.input_datapath+'/Gene_expression/metaDataIR3.csv')
        gene_data_diagnosis = gene_metadata[gene_metadata["DIAGNOSIS"].isin(["PD", "Control", "Prodromal"])]    #Filter the Data with PD, Control or Prodromal
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
        files = sorted([f for f in os.listdir(gene_expression_path) if f.startswith('PPMI-Phase1-IR3.') and f.endswith('salmon.genes.sf')])
        
        if self.debug:
            print("HYS data shape:", HYS_data.shape)
            print("Gene metadata shape:", gene_data_diagnosis.shape)
            print("Common PATNOs found:", len(common_keys))
            print("First few matched PATNOs:", list(common_keys)[:5])
            print("Example gene file:", files[0])
            print("Matched gene samples:", gene_matched.shape)
            print("Matched HYS samples:", HYS_matched.shape)
            return gene_matched, HYS_matched
        
        col1 = ["PATNO", "EVENT_ID"]
        all_column = col1 + self.gene_list
        gene_counts_df = pd.DataFrame(columns=all_column)
        print(len(files))
        matched_count = 0  # initialize a counter
        unmatched_keys = []

        for filename in files:
            parts = filename.split('.')
            patno = parts[1]
            event_ids = parts[2]
            if (patno, event_ids) in valid_keys:
                matched_count += 1  # increment the counter

                fullname = os.path.join(gene_expression_path, filename)
                
                data_genecounts = pd.read_csv(fullname, sep='\t')
                data_genecounts["geneID_base"] = data_genecounts["Name"].str.split('.').str[0]
                gene_name_set = data_genecounts[data_genecounts["geneID_base"].isin(self.gene_list)]

                my_row = {"PATNO": patno, "EVENT_ID": event_ids}
                for gene in self.gene_list:
                    names_match = gene_name_set[gene_name_set["geneID_base"] == gene]
                    if not names_match.empty:
                        my_row[gene] = names_match["TPM"].values[0]
                    else:
                        my_row[gene] = 0
                
                gene_counts_df = pd.concat([gene_counts_df, pd.DataFrame([my_row])], ignore_index=True)
            else:
                unmatched_keys.append((patno, event_ids))
                
        gene_counts_df.to_csv(os.path.join(self.output_datapath, "gene_expression_summary.csv"), index=False)
        print(f"Total matched gene files with valid PATNO/EVENT_ID pairs: {matched_count}")
        return gene_matched, HYS_matched, gene_counts_df,unmatched_keys
