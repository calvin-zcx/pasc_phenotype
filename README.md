# RECOVER - PASC
This study is part of the NIH Researching COVID to Enhance Recovery (RECOVER) Initiative, which seeks to understand, treat, and prevent the post-acute sequelae of SARS-CoV-2 infection (PASC). For more information on RECOVER, visit https://recovercovid.org/. 


## Related Works
1. Understanding Post-Acute Sequelae of SARS-CoV-2 Infection through Data-Driven Analysis with Longitudinal Electronic Health Records: Findings from the RECOVER Initiative
2. Machine Learning for Identifying Data-Driven Subphenotypes of Incident Post-Acute SARS-CoV-2 Infection Conditions with Large Scale Electronic Health Records: Findings from the RECOVER Initiative

## Test Systems
Windows 10 PC, 16 GB memory, 500 GB hard disk,  6 GB NVIDIA GeForce GTX 1060 GPU
Linux Ubuntu 18.04.2 LTS server, 62 GB memory, 500 GB hard disk, 11 GB GeForce RTX 2080 Ti GPU, and 16 CPU cores. 
Python environment install and activation
Notes: recommend using tmux at the terminal to run all the following commands
git clone https://github.com/calvin-zcx/pasc_phenotype.git
cd pasc_phenotype/
conda env create -f environment.yml   
conda activate pasc

## Code Structure
1. preprocess- EHR preorpcessing  
3. iptw - High-throughput screening of PASC by machine learning-based propensity-score reweighting method
4. prediction - PASC prediction
5. misc - all the related functions for different entities in EHR system

shell commands are located in each package to run different functions
