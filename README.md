# Analysis Protocol for Identifying Candidate Drugs for a Specific Disease

A. Methodology Protocol Design

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

B. Solution Implementation
Provide a repository that includes a script in the scripting language of your preference (R, bash, python) that implements some of the above tools as you see fit within a 3 day timeline which is the duration of the challenge. Provide a minimal README.md file alongside that describes the protocol (part A of the challenge) and 

