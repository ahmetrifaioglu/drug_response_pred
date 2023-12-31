# Analysis Protocol for Identifying Candidate Drugs for a Specific Disease

## Methodology Protocol Design

1. Introduction:

   In this proposal, I present a protocol for identifying candidate drugs to potentially be used to cure or alleviate a disease phenotype. The protocol utilizes prioritized gene lists as one of the primary data inputs. Publicly available open source databases and tools will be used to perform the analysis.
2. Data Sources:

   a. Primary Data Input: Prioritized Gene Lists

   - The prioritized gene lists can be obtained from relevant studies, databases, or based on the marker genes specific to the chosen disease. These genes assumed to be associated with the disease phenotype and the aim is to find drugs that targeting these genes.

   b. Alternative Data Input: Publicly Accessible Datasets

   - The drug-target interaction databases such as ChEMBL, DrugBank to create the dataset for training
3. Computational Analysis Steps:

   a. Data Integration:

   - Merge the prioritized gene lists with other relevant datasets, if available, to augment the analysis and provide additional context.

   b. Drug Target Prioritization:

   - Utilize open source tools such as DrugBank, ChEMBL, or other drug databases to identify known drugs or compounds targeting genes from the prioritized gene lists and create a training dataset by extracting the features of drugs using open-source framework RDKit.

   c. Virtual Screening and Drug Repurposing:

   - Train a virtual screening method to infer the binding affinity between candidate drugs and the selected drug targets.
   - Provide a list of bioactive drugs for the target genes associated with the selected disease.
4. Output:
   The final output of the computational analysis is a recommendation list of candidate compounds/drugs that have the potential to aid in alleviating the observed disease phenotype.

## Solution Implementation - Summary

1. As a case study of the challenge, I selected "COLORECTAL CANCER" (i.e. CRC) as the disease of interest and got the list of genes that are associated with CRC from  `Online Mendelian Inheritance in Man (OMIM)` database https://omim.org/entry/114500 . For initial training, a subset of associated genes were exported to a text file where each line must include a gene symbol. The created file will be used as input to the system. In our case the file is stored under `data/COLORECTAL_CANCER_genes.txt`.
2. Subsequnetly, the known activities against the list of genes are retrived from the ChEMBL database which is a manually curated database of bioactive molecules with drug-like properties. The aim here is to get high confidence drug-target gene interaction data to create our training dataset. I used `ChEMBL webresource client` for getting and filtering data.
3. Several filtering and preprocessing steps were used to get more reliable bioactivities. The data points were selectively filtered based on various criteria such as the "target type" (specifically single protein), "taxonomy", "assay type" (covering binding assays), and "standard type" (i.e. IC50) attributes. Finally, the bioactivity without a `pChEMBL` value were filtered, which is used to obtain comparable measures of half-maximal response on a negative logarithmic scale in ChEMBL. The bioactivity measurements with pChEMBL value represents more reliable interactions since they are manually curated. The filtering steps were similar to the defined filters determined in our previous study called DEEPScreen (_Rifaioglu, A. S., Nalbat, E., Atalay, V., Martin, M. J., Cetin-Atalay, R., & Doğan, T. (2020). DEEPScreen: high performance drug–target interaction prediction with convolutional neural networks using 2-D structural compound representations. Chemical Science, 11(9), 2531-2557._ https://pubs.rsc.org/en/content/articlelanding/2020/sc/c9sc03414e ).
4. Once we created the compound-target (or gene) and bioactivity values (IC50) dataset we fetched the SMILEs strings of compounds which are line notations for encoding molecular structures.
5. Each compound was represented by a circular fingerprint (extended connectiviry fingerprints)`which are one of the widely-used features to train virtual screening methods. I used`RDKit` framework to create feature vectors using the smiles (https://www.rdkit.org/docs/GettingStartedInPython.html). For this, we created mol objects for each compound using their SMILES representations and then ECFP4 fingerprints were created for each compound which represents their feature vectors.
6. Virtual screening problem can be considered as a classification or a regression task. In classification, the aim is to predict whether a compound is active or inactive based on a predefined concentration threshold (such as 100 nM, 1 µM etc.) applied on IC50 values. In regression, the aim is to predict the real IC50 values. Once the final models are obtained a threshold can be used to provide candidate compounds as output. Here, I considered the problem as a regression problem and train a model to predict bioactivity values (pChEMBL values) given the training dataset.
7. Training: Users can select different machine learning algorithms (Support Vector Regression, Random Forest, Multi Layer Perceptron) to train their models. The constructed training dataset randomly seperated into 3 parts (training, validation and test). The models trained based on different loss functions such as Mean Squared Error (MSE), HuberLoss and Mean Absolute Error (MAE) loss.
8. Finally the compounds above a specific threshold can be used as candidate compounds. Here I did not provide a specific threshold as the threshold may be highly different depending on the gene/protein family. Selecting the compounds with high bioactivities based on pChEMBL values is quite easy by checking the output file (for example, pCHEMBL value of 6.0 equals to 1 micro molar) and the compounds above the threshold of 6.0 can  be retrieved from `<experiment_name>_predictions.csv`.


## Summary of scripts

-- **_evaluation_metrics.py_**
The script defines several functions for calculating evaluation metrics for regression models. The metrics included are:

- R-squared score (coefficient of determination) calculated using the `r2_score` function from `sklearn.metrics`.
- Mean squared error (MSE) calculated using the `mean_squared_error` function from `sklearn.metrics`.
- Root mean squared error (RMSE) calculated by taking the square root of the MSE.
- Mean absolute error (MAE) calculated using the `mean_absolute_error` function from `sklearn.metrics`.

Each function takes two arguments, `y_true` and `y_pred`, which represent the true values and predicted values, respectively, for a regression task. The functions return the corresponding evaluation metric value.

-- **_hyperparam_config.py_**
This script includes the dictionaries for hyperparameter search.

-- **_model.py_**
This script defines a class called Regressors that implements several regression models and their hyperparameter tuning using cross-validation.

-- **_utils.py_**
Includes several utilities function for data generation and manipulation.

-- **_main.py_**
The main function that takes the inputs and perform training.

## How to reproduce the results

- Clone the Git Repository
- Create conda environment as shown below

```
conda env create -f environment.yml

conda activate drug-pred-chal-env
```

- Run main.py script:

```
python main.py -i ../data/COLORECTAL_CANCER_genes.txt -m SVR -o ../results -en CRC_predict
```
After running the above scripts 3 output files will be generated under specified output folder (i.e. ../results folder in this example)
 - `<experiment_name>.model` keeps the trained model to be used later for making new predictions.
 - `<experiment_name>_performance_results.txt` where prediction performance results are stored.
 - `<experiment_name>_predictions.csv` where the real and predicted bioactivity values are stored. The candidate molecules can be selected from this file based on the threshold of interest.

## Explanation of Arguments

* **--input_dir, -i**: Input path to the list of genes, each line is a gene symbol
* **--model, -m**: Training algorithm to be selected. possible values RF, SVR, MLP
* **--output_dir, -o**: output directory where results will be save
* **--experiment_name, -en**: the name of the experiment, experiment name is attached to the output files(default: my_experiment)

## Notes

- The aim of this work is not to provide an accurate drug/compound specific to a disease. Here, the aim is to show how to create a predictive model to infer the activate compounds against a disease, which resources can be used and how a basic design could be.
- There are several ways to improve the pipeline proposed here:

  - This method can be further improved by considering the target gene or by creating a pairwise input machine learning model so that the target gene features can also be used.
  - The binding affinity values reported in the literature mostly consist of drug-target interaction pairs with high binding affinities which may cause a bias in predictive models. Therefore, a well-defined sampling techniques should be applied before creating the final model based on the distribution of the data.
  - Functional Analysis:

    - Perform functional analysis on the prioritized gene lists using open source tools such as Gene Ontology (GO) enrichment analysis, pathway analysis, or network analysis.
    - Identify enriched biological processes, molecular functions, or pathways associated with the prioritized genes, providing insights into the disease mechanism.
  - Pharmacokinetic and Toxicity Analysis:

    - Evaluate the pharmacokinetic properties and potential toxicity of the identified candidate drugs using open source tools such as Open Babel, RDKit, or Tox21.
    - Assess factors such as drug metabolism, absorption, distribution, excretion, and toxicity profiles to identify drugs with favorable pharmacological properties.
  - The trained model can be used for testing completely new compounds. 
