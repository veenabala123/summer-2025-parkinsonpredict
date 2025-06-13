import os, sys
import numpy as np
import pandas as pd


class ReadData:
    
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
        gene_data_diagnosis = gene_metadata[gene_metadata["DIAGNOSIS"].isin(["PD", "Control", "Prodromal"])]    #Filter the Data with PD, Control or Prodromal
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
        if self.debug:
            print("HYS data shape:", HYS_data.shape)
            print("Gene metadata shape:", gene_data_diagnosis.shape)
            print("Common PATNOs found:", len(common_keys))
            print("First few matched PATNOs:", list(common_keys)[:5])
            print("Example gene file:", files[0])
            print("Matched gene samples:", gene_matched.shape)
            print("Matched HYS samples:", HYS_matched.shape)
            return gene_matched, HYS_matched
        else:
            return gene_matched, HYS_matched
        
    def gene_expression(self):
        """Reads the gene expression data and reconstructs it in a 
        readable format. 
        currently loads only one datapoint per sample, loading the early visit.
        """
        #Step 1: Load the data
        HYS_data_path = self.input_datapath + 'Diagnosis_History_UPDRS_HYS/'
        HYS_data = pd.read_csv(HYS_data_path+"MDS-UPDRS_Part_III_03Jun2025.csv")
        gene_metadata = pd.read_csv(self.input_datapath+'/Gene_expression/metaDataIR3.csv')

        #Step 2: From the MDS file choose only the patients in the following groups
        gene_data_diagnosis = gene_metadata[gene_metadata["DIAGNOSIS"].isin(["PD", "Control", "Prodromal"])]    #Filter the Data with PD, Control or Prodromal
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

        #Step 5: Make a dictionary for the common patno and event_id
        valid_keys = set(zip(gene_matched["PATNO"], gene_matched["EVENT_ID"].str.strip()))
        
        #Step 6: Make a list of all the gene expression files
        gene_expression_path = self.input_datapath + "/Gene_expression/quant/"
        files = sorted([f for f in os.listdir(gene_expression_path) if f.startswith('PPMI-Phase1-IR3.') and f.endswith('salmon.genes.sf')])
        
        #Step 7: Make a empty pandas dataframe with column names
        col1 = ["PATNO", "EVENT_ID"]
        col2 = ["NHY"]
        all_column = col1 + self.gene_list + col2
        gene_counts_df = pd.DataFrame(columns=all_column)
        print(len(files))
        matched_count = 0  # initialize a counter
        unmatched_keys = []
        
        #Step 8: Loop through all the files
        for filename in files:
            parts = filename.split('.')
            patno = int(parts[1].strip())
            event_ids = parts[2].strip()
            
            #Step 9: Only load the files with the common keys
            if (patno, event_ids) in valid_keys:
                matched_count += 1  # increment the counter
                
                #Step 10: First load the NHY data
                nhy_row = HYS_matched.loc[(HYS_matched["PATNO"]==patno) & (HYS_matched["EVENT_ID"].str.strip()==event_ids), "NHY"]
                if not nhy_row.empty:
                    nhy = nhy_row.values[0]
                else:
                    print(f"No NHY for PATNO={patno}, EVENT_ID = {event_ids}")
                    nhy = -1
                    
                #Step 11: Load the gene expression data file
                fullname = os.path.join(gene_expression_path, filename)
                data_genecounts = pd.read_csv(fullname, sep='\t')
                
                #Step 12: Loading the gene expression, stripping suffix to get the general code
                data_genecounts["geneID_base"] = data_genecounts["Name"].str.split('.').str[0]
                
                #Step 13: Dataset with only the genes in the list that we provided above
                gene_name_set = data_genecounts[data_genecounts["geneID_base"].isin(self.gene_list)]

                my_row = {"PATNO": patno, "EVENT_ID": event_ids, "NHY":nhy}

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
            else:
                unmatched_keys.append((patno, event_ids))
                
        gene_counts_df.to_csv(os.path.join(self.output_datapath, "gene_expression_summary.csv"), index=False)
        print(f"Total matched gene files with valid PATNO/EVENT_ID pairs: {matched_count}")
        return gene_counts_df,unmatched_keys
