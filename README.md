# Analysis Protocol for Identifying Candidate Drugs for a Specific Disease

**A. Methodology Protocol Design**

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

   b. Functional Analysis:
      - Perform functional analysis on the prioritized gene lists using open source tools such as Gene Ontology (GO) enrichment analysis, pathway analysis, or network analysis.
      - Identify enriched biological processes, molecular functions, or pathways associated with the prioritized genes, providing insights into the disease mechanism.

   c. Drug Target Prioritization:
      - Utilize open source tools such as DrugBank, ChEMBL, or other drug databases to identify known drugs or compounds targeting genes from the prioritized gene lists and create a training dataset by extracting the features of drugs using open-source framework RDKit.
    
   d. Virtual Screening and Drug Repurposing:
      - Train a virtual screening method to infer the binding affinity between candidate drugs and the selected drug targets.
      - Provide a list of bioactive drugs for the target genes associated with the selected disease.

   e. Pharmacokinetic and Toxicity Analysis (Not implemented but necessary):
      - Evaluate the pharmacokinetic properties and potential toxicity of the identified candidate drugs using open source tools such as Open Babel, RDKit, or Tox21.
      - Assess factors such as drug metabolism, absorption, distribution, excretion, and toxicity profiles to identify drugs with favorable pharmacological properties.

4. Output:
   The final output of the computational analysis is a recommendation list of candidate compounds/drugs that have the potential to aid in alleviating the observed disease phenotype.

**B. Solution Implementation**
1. Summary of the implementation:

As a case study of the challenge, I selected "COLORECTAL CANCER" (i.e. CRC) as the disease of interest and got the list of genes that are associated with CRC from  Online Mendelian Inheritance in Man (OMIM) database https://omim.org/entry/114500 . First, ChEMBL A

2. A subset of associated genes were saved to the text file where each line must include a gene symbol. The created file will be used as input to the system. In our case the file is stored under `data/COLORECTAL_CANCER_genes.txt`. Subsequnetly, the known activities against the list of genes are retrived from the ChEMBL database which is a manually curated database of bioactive molecules with drug-like properties. The aim here is to get manually-cruted drug-target interaction data to create our trainign dataset. First, we apply several filtering and preprocessing steps to get more reliable bioactivities. Initially, the data points were selectively filtered based on various criteria such as the "target type" (specifically single protein), "taxonomy" (including human), "assay type" (covering binding assays), and "standard type" (i.e. IC50) attributes. I used ChEMBL webresource client for getting and filtering data. Once we created the compound-target (or gene) and bioactivity values (IC50) dataset we fetched the SMILEs strings of compounds whic are line notations for encoding molecular structures. Subsequenly, each compounds was represented by a circular fingerprint (extended connectiviry fingerprint) which are one of the widely-used features to train virtual screening methods. I used RDKit framework to create feature vectors using the smiles. For this, we created a mol object for each compound and ECFP4 fingerprints were created for each compound.
Virtual screening can be considered as a classification or a regression task. We can determine a threshold value to convert
In this solution where we try to predict the IC50 values of each compound.
I implemented 4 different regression methods.  m  
4.
5. compound-target 

6. 
7. The implementation includes 5 main scripts:
How to create environment:


```
conda env create -f environment.yml

conda activate drug-pred-chal-env
```
How to run the analysis:


```
python main.py -i ../data/COLORECTAL_CANCER_genes.txt -m SVR -o ../results -en CRC_predict
```
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



"""
The aim here is not to provide an accurate drug/compı specific to a disease. Here the aim is to show how to create a predictive model to infer the activate compounds against a disease.

To do this, we first get the gene list provided by the user. 
The basic assumption is that we have prioritised list of genes that were inferred against a disease. 


 . We do not use the all the activities available in ChEMBL database. First, we apply several filtering steps 
to get more reliable bioactivities (i.e. )
The model should also be aware of the genes so the features of the genes can also be used. 

"""

"""

The model uses all the drug-target interaction associated with the provided list of genes without taking into account the target informatiom. 

This method can be further improved by considering the target gene or by creating a pairwise input machilne learning model so that the the target 
information is also incorporated. 

This model, since in majority of cases only the hihgly bioactive ocmpounds are reported against target, there are biases in predictive models. Therefore, 
a well-defined sampling techniques should be applied before creating the final model based on the distributin of the data. 


Here

"""

