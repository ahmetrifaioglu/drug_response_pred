import utils
import pandas as pd
from model import Regressors

lst_genes = ["PLA2G2A", "NRAS", "BUB1", "BUB1", "CTNNB1", "PIK3CA", "FGFR3", "TLR2", "APC", "MCC", "PTPN12", "BRAF", "DLC1", "RAD54B", "PTPRJ", "CCND1", "MLH3"]
print("Collecting tested compound/drug-target interactions (i.e., bioactivities) from ChEMBL database...")

all_act = []
for gene in lst_genes:
    acts = utils.get_IC50_values_given_target(gene)
    if acts:
        all_act.extend(acts)

df = pd.DataFrame(all_act, columns=["target_chembl_id", "molecule_chembl_id", "pchembl_value"])

print("Dropping duplicate entries...")
df = df.groupby('molecule_chembl_id').first().reset_index()


lst_molchembl_ids = df["molecule_chembl_id"][:100]

# print(len(lst_molchembl_ids))
molid2smiles_dict = utils.get_smiles_from_molecule_chembl_id_list(lst_molchembl_ids)


# print(molid2smiles_dict)
mol_id_list, feat_arr = utils.get_ecfp4_features(molid2smiles_dict)


# print(mol_id_list)
# get rows with valid molecule ids
df = df[df["molecule_chembl_id"].isin(mol_id_list)]

"""print(feat_arr.shape)
print(len(mol_id_list))
print(df[:len(mol_id_list)])"""


train_feats, train_labels, train_ids, val_feats, val_labels, val_ids, test_feats, test_labels, test_ids = utils.split_data(feat_arr, df["pchembl_value"][:len(mol_id_list)], df["molecule_chembl_id"][:len(mol_id_list)])

regressor_obj = Regressors()
regressor_obj.regress_baseline_model("RF", train_feats, train_labels, train_ids, val_feats, val_labels, val_ids, test_feats, test_labels, test_ids)

# print(molid2feat_dict)
# for ind, row in df.iterrows():
    





"""print("Training features:\n", train_feats)
print("Training labels:\n", train_labels)
print("Training sample IDs:\n", train_ids)
print("Validation features:\n", val_feats)
print("Validation labels:\n", val_labels)
print("Validation sample IDs:\n", val_ids)
print("Test features:\n", test_feats)
print("Test labels:\n", test_labels)
print("Test sample IDs:\n", test_ids)"""



"""
The aim here is not to provide an accurate drug/compı specific to a disease. Here the aim is to show how to create a predictive model to infer the activate compounds against a disease.

To do this, we first get the gene list provided by the user. 
The basic assumption is that we have prioritised list of genes that were inferred against a disease. 


Subsequnetly, the known activities against the list of genes are retrived from the ChEMBL database which is a manually curated database of bioactive molecules with drug-like properties. . We do not use the all the activities available in ChEMBL database. First, we apply several filtering steps 
to get more reliable bioactivities (i.e. )
The model should also be aware of the genes so the features of the genes can also be used. 

"""