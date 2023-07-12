import os
import sys
import rdkit
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from chembl_webresource_client.new_client import new_client



def get_IC50_values_given_target(gene_id="BRD4"):
    """
    Retrieves IC50 values from the ChEMBL database for a given gene ID.

    Args:
        gene_id (str): The gene ID for which IC50 values are to be retrieved. Defaults to "hERG".

    Returns:
        QuerySet: A QuerySet object containing IC50 values matching the given gene ID.
    """
    print(f"Checking the activities of gene id:{gene_id} match in ChEMBL database...")
    target = new_client.target
    activity = new_client.activity
    target = target.filter(target_type='SINGLE PROTEIN')
    # print(target)

    # checking if exact gene name found
    # tar_chem_id = target.filter(pref_name__iexact=gene_id).only('target_chembl_id')[0]
    # target_synonym__icontains=gene_name
    tar_chem_id = target.filter(target_synonym__icontains=gene_id).only('target_chembl_id')[0]
    lst_tar_comp_bioact_val = []
    if tar_chem_id:
        activities = activity.filter(target_chembl_id=tar_chem_id['target_chembl_id'], assay_type='B', standard_relation='=', standard_units='nM', target_type="SINGLE PROTEIN", pchembl_value__isnull=False).filter(standard_type="IC50")
        
        for act in activities:
            lst_tar_comp_bioact_val.append([tar_chem_id["target_chembl_id"], act["molecule_chembl_id"], act["pchembl_value"]])
            
        return lst_tar_comp_bioact_val
    else:
        return None


def get_ecfp4_fp_from_smiles(smiles, radius=2, nBits=1024):
    """
    Calculates the ECFP4 fingerprint from a given SMILES string.

    Args:
        smiles (str): The SMILES string representing the chemical structure. Defaults to 'Cc1ccccc1'.
        radius (int): The radius parameter for the Morgan fingerprint generation. Defaults to 2.
        nBits (int): The number of bits in the generated fingerprint. Defaults to 1024.

    Returns:
        list: A list representing the ECFP4 fingerprint as a sequence of bits.
    """
    m1 = Chem.MolFromSmiles(smiles)
    fpgen = AllChem.GetMorganFingerprintAsBitVect(m1, radius, nBits)
    return list(fpgen)

def read_input_gene_list(input_fl_path):
    """
    Reads a gene list from the input file.

    Args:
        input_fl_path (str): The path to the input file.

    Returns:
        list: A list of genes read from the file.
    """
    gene_fl = open(input_fl_path, "r")
    lst_genes = gene_fl.read().split("\n")
    gene_fl.close()
    return lst_genes

def get_smiles_from_molecule_chembl_id_list(lst_mol_id):
    """
    Retrieves canonical SMILES representations for a list of molecule ChEMBL IDs.

    Args:
        lst_mol_id (list): A list of molecule ChEMBL IDs for which canonical SMILES are to be retrieved.

    Returns:
        dict: A dictionary mapping molecule ChEMBL IDs to their corresponding canonical SMILES representations.
    """
    mol2smiles_dict = dict()
    molecule = new_client.molecule
    for molid in lst_mol_id:
        m1 = molecule.filter(chembl_id=molid).only(['molecule_chembl_id', 'molecule_structures'])[0]
        if m1:
            mol2smiles_dict[molid] = m1['molecule_structures']['canonical_smiles']
    return mol2smiles_dict


def get_ecfp4_feature_dict(smiles_dict):
    """
    Generates ECFP4 fingerprint feature vectors for a dictionary of SMILES.

    Args:
        smiles_dict (dict): A dictionary containing SMILES as keys and corresponding molecule representations.

    Returns:
        dict: A dictionary mapping SMILES to their corresponding ECFP4 fingerprint feature vectors.
    """
    feature_dict = dict()
    for smiles in smiles_dict:
        # print(smiles)
        try:
            feat_arr = get_ecfp4_fp_from_smiles(smiles_dict[smiles])    
            feature_dict[smiles] = feat_arr
        except:
            # TODO: the failed ecfp4 construction should be logged
            pass
    
    return feature_dict


def get_ecfp4_features(smiles_dict):
    mol_id_list = []
    feat_list = []

    for smiles in smiles_dict:
        
        try:
            feat_arr = get_ecfp4_fp_from_smiles(smiles_dict[smiles])    
            mol_id_list.append(smiles)
            feat_list.append(feat_arr)
        except:
            # TODO: the failed ecfp4 construction should be logged
            pass
    
    return mol_id_list, np.array(feat_list)





def split_data(features, labels, sample_ids, train_ratio=0.8, val_ratio=0.10, test_ratio=0.10):
    """
    Splits the data into training, validation, and test sets while maintaining corresponding sample IDs.

    Args:
        features (numpy.ndarray): Array of input features.
        labels (numpy.ndarray): Array of corresponding labels.
        sample_ids (numpy.ndarray): Array of sample IDs.
        train_ratio (float): Ratio of data to allocate for training. Default: 0.8
        val_ratio (float): Ratio of data to allocate for validation. Default: 0.10
        test_ratio (float): Ratio of data to allocate for testing. Default: 0.10

    Returns:
        tuple: A tuple containing the following arrays in the specified order:
            - train_features: Training set features
            - train_labels: Training set labels
            - train_sample_ids: Training set sample IDs
            - val_features: Validation set features
            - val_labels: Validation set labels
            - val_sample_ids: Validation set sample IDs
            - test_features: Test set features
            - test_labels: Test set labels
            - test_sample_ids: Test set sample IDs
    """
    # Shuffle the data randomly while maintaining the corresponding sample IDs
    data = np.column_stack((features, labels, sample_ids))
    np.random.shuffle(data)

    # Calculate the split indices
    num_samples = len(data)
    train_split = int(num_samples * train_ratio)
    val_split = int(num_samples * (train_ratio + val_ratio))

    # Split the data into training, validation, and test sets
    train_data = data[:train_split, :]
    val_data = data[train_split:val_split, :]
    test_data = data[val_split:, :]

    # Split features, labels, and sample IDs
    train_features, train_labels, train_sample_ids = train_data[:, :-2], train_data[:, -2], train_data[:, -1]
    val_features, val_labels, val_sample_ids = val_data[:, :-2], val_data[:, -2], val_data[:, -1]
    test_features, test_labels, test_sample_ids = test_data[:, :-2], test_data[:, -2], test_data[:, -1]

    return train_features, train_labels, train_sample_ids, val_features, val_labels, val_sample_ids, test_features, test_labels, test_sample_ids



def get_chembl_activities_df(lst_genes):
    """
    Retrieves ChEMBL activities as a DataFrame for a list of genes.

    Args:
        lst_genes (list): A list of genes for which ChEMBL activities are to be retrieved.

    Returns:
        pd.DataFrame: A pandas DataFrame containing ChEMBL activities with columns ["target_chembl_id", "molecule_chembl_id", "pchembl_value"].
    """
    all_act = []
    for gene in lst_genes:
        acts = get_IC50_values_given_target(gene)
        if acts:
            all_act.extend(acts)

    df = pd.DataFrame(all_act, columns=["target_chembl_id", "molecule_chembl_id", "pchembl_value"])
    return df
