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

5. Output:
   The final output of the computational analysis is a recommendation list of candidate compounds/drugs that have the potential to aid in alleviating the observed disease phenotype.

## Solution Implementation - Summary

1. As a case study of the challenge, I selected "COLORECTAL CANCER" (i.e. CRC) as the disease of interest and got the list of genes that are associated with CRC from  `Online Mendelian Inheritance in Man (OMIM)` database https://omim.org/entry/114500 . For initial training, a subset of associated genes were exported to a text file where each line must include a gene symbol. The created file will be used as input to the system. In our case the file is stored under `data/COLORECTAL_CANCER_genes.txt`. 

2. Subsequnetly, the known activities against the list of genes are retrived from the ChEMBL database which is a manually curated database of bioactive molecules with drug-like properties. The aim here is to get high confidence drug-target gene interaction data to create our training dataset. I used `ChEMBL webresource client` for getting and filtering data. 

3. Several filtering and preprocessing steps were used to get more reliable bioactivities. The data points were selectively filtered based on various criteria such as the "target type" (specifically single protein), "taxonomy", "assay type" (covering binding assays), and "standard type" (i.e. IC50) attributes. Finally, the bioactivity without a `pChEMBL` value were filtered, which is used to obtain comparable measures of half-maximal response on a negative logarithmic scale in ChEMBL. The bioactivity measurements with pChEMBL value represents more reliable interactions since they are manually curated. The filtering steps were similar to the defined filters determined in our previous study called DEEPScreen (_Rifaioglu, A. S., Nalbat, E., Atalay, V., Martin, M. J., Cetin-Atalay, R., & Doğan, T. (2020). DEEPScreen: high performance drug–target interaction prediction with convolutional neural networks using 2-D structural compound representations. Chemical Science, 11(9), 2531-2557._ https://pubs.rsc.org/en/content/articlelanding/2020/sc/c9sc03414e ).

4. Once we created the compound-target (or gene) and bioactivity values (IC50) dataset we fetched the SMILEs strings of compounds which are line notations for encoding molecular structures.
  
5. Each compound was represented by a circular fingerprint (extended connectiviry fingerprints)` which are one of the widely-used features to train virtual screening methods. I used `RDKit` framework to create feature vectors using the smiles (https://www.rdkit.org/docs/GettingStartedInPython.html). For this, we created mol objects for each compound using their SMILES representations and then ECFP4 fingerprints were created for each compound which represents their feature vectors.

6. Virtual screening problem can be considered as a classification or a regression task. In classification, the aim is to predict whether a compound is active or inactive based on a predefined concentration threshold (such as 100 nM, 1 µM etc.) applied on IC50 values. In regression, the aim is to predict the real IC50 values. Once the final models are obtained a threshold can be used to provide candidate compounds as output. Here, I considered the problem as a regression problem and train a model to predict bioactivity values (pChEMBL values) given the training dataset.

7. Training: Users can select different machine learning algorithms (Support Vector Regression, Random Forest, Multi Layer Perceptron) to train their models. The constructed training dataset randomly seperated into 3 parts (training, validation and test). The models trained based on different loss functions such as Mean Squared Error (MSE), HuberLoss and Mean Absolute Error (MAE) loss.

8. Finally the compounds above a specific threshold (here I used 1 micro molar) are provided as active compounds as output.

The implementation includes 5 main scripts:

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



## Note:
- The aim of this work is not to provide an accurate drug/compound specific to a disease. Here, the aim is to show how to create a predictive model to infer the activate compounds against a disease, which resources can be used and how a basic design could be.

- There are several ways to improve the pipeline proposed here:
   - This method can be further improved by considering the target gene or by creating a pairwise input machilne learning model so that the the target The model should also be aware of the genes so the features of the genes can also be used. 
information is also incorporated. 

   - Since in majority of cases only the hihgly bioactive ocmpounds are reported against target, there are biases in predictive models. Therefore, a well-defined sampling techniques should be applied before creating the final model based on the distributin of the data.
     
   - Functional Analysis:
     - Perform functional analysis on the prioritized gene lists using open source tools such as Gene Ontology (GO) enrichment analysis, pathway analysis, or network analysis.
      - Identify enriched biological processes, molecular functions, or pathways associated with the prioritized genes, providing insights into the disease mechanism.
   
   - Pharmacokinetic and Toxicity Analysis:
      - Evaluate the pharmacokinetic properties and potential toxicity of the identified candidate drugs using open source tools such as Open Babel, RDKit, or Tox21.
      - Assess factors such as drug metabolism, absorption, distribution, excretion, and toxicity profiles to identify drugs with favorable pharmacological properties.
      
