import os
import utils
import argparse
import pandas as pd
from model import Regressors


############################### INPUT ARGUMENTS ############################### 
# Read command line and set args
parser = argparse.ArgumentParser(prog='TRN', description='Train a model to predict drug-response for list of genes associated with a disease')
parser.add_argument('-i', '--input_dir', help='Input path to the list of genes, each line is a gene symbol', required=True)
parser.add_argument('-m', '--model', help='Regressor model: RF, SVR, MLP', default="RF", required=True)
parser.add_argument('-o', '--output_dir', help='Output directory', default="../results", required=True)
parser.add_argument('-en', '--experiment_name', help='The name of the experiment...', default="my_experiment", required=True)

args = vars(parser.parse_args())
input_path = args['input_dir']
model = args['model']
output_dir = args['output_dir']
experiment_name = args['experiment_name']
############################### INPUT ARGUMENTS ############################### 


# lst_genes = ["PLA2G2A", "NRAS", "BUB1", "BUB1", "CTNNB1", "PIK3CA", "FGFR3", "TLR2", "APC", "MCC", "PTPN12", "BRAF", "DLC1", "RAD54B", "PTPRJ", "CCND1", "MLH3"]
print("### Reading the input file...")
lst_genes = utils.read_input_gene_list(input_path)

print("### Collecting tested compound/drug-target interactions (i.e., bioactivities) from ChEMBL database...")
df_activities = utils.get_chembl_activities_df(lst_genes)


print("### Dropping duplicate entries...")
df_activities = df_activities.groupby('molecule_chembl_id').first().reset_index()
lst_molchembl_ids = df_activities["molecule_chembl_id"][:100]

print("### Fetching smiles strings of compounds from ChEMBL database...")
molid2smiles_dict = utils.get_smiles_from_molecule_chembl_id_list(lst_molchembl_ids)

print("### Creating ECFP4 fingerprints for training...")
mol_id_list, feat_arr = utils.get_ecfp4_features(molid2smiles_dict)
# get rows with valid molecule ids
df_activities = df_activities[df_activities["molecule_chembl_id"].isin(mol_id_list)]


print("### Creating random train, validation, test splits...")
train_feats, train_labels, train_ids, val_feats, val_labels, val_ids, test_feats, test_labels, test_ids = utils.split_data(feat_arr, df_activities["pchembl_value"][:len(mol_id_list)], df_activities["molecule_chembl_id"][:len(mol_id_list)])

print("### Training models and performing hyperparameter search...")
regressor_obj = Regressors(os.path.join(output_dir, experiment_name))
regressor_obj.regress_baseline_model(model, train_feats, train_labels, train_ids, val_feats, val_labels, val_ids, test_feats, test_labels, test_ids)


# python main.py -i ../data/COLORECTAL_CANCER_genes.txt -m SVR -o ../results -en CRC_predict
