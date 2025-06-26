import os

#========= BASE PATHs ===============================================
PROJECT_ROOT = '/Users/pushpita/Documents/Erdos_bootcamp/our_project/'
DATA_DIR = os.path.join(PROJECT_ROOT, "Data/") 
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "Outputs/")

#==== Raw data files ======================================================
GENE_EXPRESSION_CLINICAL_FILE = os.path.join(DATA_DIR, "gene_expression_summary.csv")
MRI_FILE = os.path.join(DATA_DIR, "PDMRI_Clean_Merged_6_13_25.csv")
NHY_FILE = os.path.join(DATA_DIR, "clean_mds_updrs.csv")

