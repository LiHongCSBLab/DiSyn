import os
from collections import defaultdict

# data path
# TCGA
gene_expressions_csv = '../data/TCGA/gex_feature_11203.csv'
tcga_response = '../data/TCGA/PMID27354694_DR_OMICS_ad.csv'
gdsctcga_mapping_file = '../data/TCGA/drug_mapping_gdsc_tcga.csv'

# PDX
gene_expressions_csv_PDX = '../data/PDX/PDXGDSC_gex_1363.csv'
pdx_response = '../data/PDX/PMID26479923GDSCV1_DR.csv'
gdscpdx_mapping_file = '../data/PDX/drug_mapping_PDX.csv'

# ISPY2
gene_expressions_csv_ISPY = '../data/ISPY2/ISPYGDSC_gex_1361.csv'
ispy_response_pac = '../data/ISPY2/PMID35623341_ISPY2_DR_Paclitaxel.csv'
ispy_response_mk226 = '../data/ISPY2/PMID35623341_ISPY2_DR_Paclitaxel_MK-2206.csv'
gdscispy_mapping_file = '../data/ISPY2/drug_mapping_ISPY.csv'

# cellline_response
cellline_response = '../data/GDSCrel82_V1_DR_LNIC50andAUC_OMICS_expseq_DR.csv'


# result path
result_path = '../results/Disyn/'
