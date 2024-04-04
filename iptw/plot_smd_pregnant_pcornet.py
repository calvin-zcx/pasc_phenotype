import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.ticker as tck
import re
import forestplot as fp

import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd
from misc.utils import check_and_mkdir, stringlist_2_str, stringlist_2_list
from scipy import stats
import re
import itertools
import functools
import random
import seaborn as sns
from textwrap import wrap

print = functools.partial(print, flush=True)
import zepid
from zepid.graphics import EffectMeasurePlot
import shlex

np.random.seed(0)
random.seed(0)
from misc import utils

print = functools.partial(print, flush=True)


def label_map(x):
    covs_map = {
        'criticalcare':'Critical care',
        'ventilation':'Ventilation',
        'Female': 'Sex:Female',
        'Male': 'Sex:Male',
        'Other/Missing': 'Sex:Other/Missing',
        'age@18-24': 'age:18-24',
        'age@25-34': 'age:25-34',
        'age@35-49': 'age:35-49',
        'age@50-64': 'age:50-64',  # 'age@65+', # # expand 65
        '65-<75 years': 'age:65-74',
        '75-<85 years': 'age:75-84',
        '85+ years': 'age:≥85',
        'RE:Asian Non-Hispanic': 'Asian Non-Hispanic',
        'RE:Black or African American Non-Hispanic': 'Black Non-Hispanic',
        'RE:Hispanic or Latino Any Race': 'Hispanic or Latino Any Race',
        'RE:White Non-Hispanic': 'White Non-Hispanic',
        'RE:Other Non-Hispanic': 'Other Non-Hispanic',
        'RE:Unknown': 'Unknown race/ethnicity',
        'ADI1-9': 'Area Deprivation Index 1-9',
        'ADI10-19': 'Area Deprivation Index 10-19',
        'ADI20-29': 'Area Deprivation Index 20-29',
        'ADI30-39': 'Area Deprivation Index 30-39',
        'ADI40-49': 'Area Deprivation Index 40-49',
        'ADI50-59': 'Area Deprivation Index 50-59',
        'ADI60-69': 'Area Deprivation Index 60-69',
        'ADI70-79': 'Area Deprivation Index 70-79',
        'ADI80-89': 'Area Deprivation Index 80-89',
        'ADI90-100': 'Area Deprivation Index 90-100',
        'ADIMissing': 'Area Deprivation Index Missing',
        '03/20-06/20': 'Infection time:03/20-06/20',
        '07/20-10/20': 'Infection time:07/20-10/20',
        '11/20-02/21': 'Infection time:11/20-02/21',
        '03/21-06/21': 'Infection time:03/21-06/21',
        '07/21-10/21': 'Infection time:07/21-10/21',
        '11/21-02/22': 'Infection time:11/21-02/22',
        '03/22-06/22': 'Infection time:03/22-06/22',
        '07/22-10/22': 'Infection time:07/22-10/22',
        '11/22-02/23': 'Infection time:11/22-02/23',
        'inpatient visits 0': 'History inpatient:0',
        'inpatient visits 1-2': 'History inpatient:1-2',
        'inpatient visits 3-4': 'History inpatient:3-4',
        'inpatient visits >=5': 'History inpatient:≥5',
        'outpatient visits 0': 'History outpatient:0',
        'outpatient visits 1-2': 'History outpatient:1-2',
        'outpatient visits 3-4': 'History outpatient:3-4',
        'outpatient visits >=5': 'History outpatient:≥5',
        'emergency visits 0': 'History emergency:0',
        'emergency visits 1-2': 'History emergency:1-2',
        'emergency visits 3-4': 'History emergency:3-4',
        'emergency visits >=5': 'History emergency:≥5',
        'BMI: <18.5 under weight': 'BMI:<18.5 under weight',
        'BMI: 18.5-<25 normal weight': 'BMI:18.5-<25 normal',
        'BMI: 25-<30 overweight ': 'BMI:25-<30 overweight',
        'BMI: >=30 obese ': 'BMI:≥30 obese',
        'BMI: missing': 'BMI:missing',
        'Smoker: never': 'Smoker:never',
        'Smoker: current': 'Smoker:current',
        'Smoker: former': 'Smoker:former',
        'Smoker: missing': 'Smoker:missing',
        'PaxRisk:Cancer': 'Cancer',
        'PaxRisk:Chronic kidney disease': 'Chronic kidney disease',
        'PaxRisk:Chronic liver disease': 'Chronic liver disease',
        'PaxRisk:Chronic lung disease': 'Chronic lung disease',
        'PaxRisk:Cystic fibrosis': 'Cystic fibrosis',
        'PaxRisk:Dementia or other neurological conditions': 'Dementia or neurological',
        'PaxRisk:Diabetes': 'Diabetes',
        'PaxRisk:Disabilities': 'Disabilities',
        'PaxRisk:Heart conditions': 'Heart conditions',
        'PaxRisk:Hypertension': 'Hypertension',
        'PaxRisk:HIV infection': 'HIV infection',
        'PaxRisk:Immunocompromised condition or weakened immune system': 'Immune Dysfunction',
        'PaxRisk:Mental health conditions': 'Mental health conditions',
        'PaxRisk:Overweight and obesity': 'Overweight and obesity',
        'PaxRisk:Pregnancy': 'Pregnancy',
        'PaxRisk:Sickle cell disease or thalassemia': 'Sickle cell or anemia',
        'PaxRisk:Smoking current': 'Smoking current or former',
        'PaxRisk:Stroke or cerebrovascular disease': 'Stroke or cerebrovascular',
        'PaxRisk:Substance use disorders': 'Substance use disorders',
        'PaxRisk:Tuberculosis': 'Tuberculosis',
        'Fully vaccinated - Pre-index': 'Fully vaccinated',
        'Partially vaccinated - Pre-index': 'Partially vaccinated',
        'No evidence - Pre-index': 'No vaccination evidence',
        "DX: Coagulopathy": 'Coagulopathy',
        "DX: Peripheral vascular disorders ": 'Peripheral vascular disorders',
        "DX: Seizure/Epilepsy": 'Seizure or epilepsy',
        "DX: Weight Loss": 'Weight Loss',
        'DX: Obstructive sleep apnea': 'Obstructive sleep apnea',
        'DX: Epstein-Barr and Infectious Mononucleosis (Mono)': 'EBV and Infectious Mono',
        'DX: Herpes Zoster': 'Herpes Zoster',
        'pregage:18-<25 years': 'Age:18-24',
        'pregage:25-<30 years': 'Age:25-29',
        'pregage:30-<35 years': 'Age:30-34',
        'pregage:35-<40 years': 'Age:35-39',
        'pregage:40-<45 years': 'Age:40-44',
        'pregage:45-50 years': 'Age:45-50',
        'age_linear': 'Age',
        'bmi_linear': 'BMI',
        'adi_linear': 'Area Deprivation Index',
        "DX: Alcohol Abuse":'Alcohol Abuse',
        "DX: Anemia":'Anemia',
        "DX: Arrythmia": 'Arrythmia',
        "DX: Asthma":'Asthma',
        "DX: Cancer":'Cancer',
        "DX: Chronic Kidney Disease":'Chronic Kidney Disease',
        "DX: Chronic Pulmonary Disorders":'Chronic Pulmonary Disorders',
        "DX: Cirrhosis":'Cirrhosis',
        "DX: Coagulopathy":'Coagulopathy',
        "DX: Congestive Heart Failure":'Congestive Heart Failure',
        "DX: COPD":'COPD',
        "DX: Coronary Artery Disease":'Coronary Artery Disease',
        "DX: Dementia":'Dementia',
        "DX: Diabetes Type 1": 'T1DM',
        "DX: Diabetes Type 2": 'T2DM',
        "DX: End Stage Renal Disease on Dialysis":'Severe Renal Disease',
        "DX: Hemiplegia":'Hemiplegia',
        "DX: HIV" : 'HIV',
        "DX: Hypertension":'Hypertension',
        "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis":'Hypertension and Diabetes',
        "DX: Inflammatory Bowel Disorder":'Inflammatory Bowel Disorder',
        "DX: Lupus or Systemic Lupus Erythematosus":'Lupus or SLE',
        "DX: Mental Health Disorders":'Mental',
        "DX: Multiple Sclerosis":'Multiple Sclerosis',
        "DX: Parkinson's Disease":'Parkinson',
        "DX: Peripheral vascular disorders ":'Peripheral vascular disorders',
        "DX: Pregnant":'Pregnant',
        "DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)":'Pulmonary Circulation',
        "DX: Rheumatoid Arthritis":'Rheumatoid Arthritis',
        "DX: Seizure/Epilepsy":'Seizure/Epilepsy',
        "DX: Severe Obesity  (BMI>=40 kg/m2)":'Severe Obesity',
        "DX: Weight Loss":'Weight Loss',
        "DX: Down's Syndrome":r"Down's",
        'DX: Other Substance Abuse':'Substance Abuse',
        'DX: Cystic Fibrosis':'Cystic Fibrosis',
        'DX: Autism':'Autism',
        'DX: Sickle Cell':'Sickle Cell',
        "MEDICATION: Corticosteroids":'Corticosteroids',
        "MEDICATION: Immunosuppressant drug":'Immunosuppressant drug'
    }
    return covs_map.get(x, x)


if __name__ == '__main__':
    # pass

    indir = r'../data/recover/output/results-20230825/DX-pospreg-posnonpreg-Rev2PSM1to1/'
    # indir = r'../data/recover/output/results-20230825/DX-pospreg-posnonpreg-Rev2RerunOri/'

    output_dir = indir + r'figure/'
    print('indir', indir)
    print('outdir', output_dir)

    df = pd.read_csv(indir + '1-any_pasc-evaluation_balance-revised.csv')
    df = df.iloc[::-1]

    fig, axes = plt.subplots(figsize=(20, 28))
    ind = np.arange(len(df))

    width = 0.3
    axes.axvline(x=0, color='black')
    axes.axvline(x=0.1, linestyle='dashed', color='grey')
    axes.axvline(x=-0.1, linestyle='dashed', color='grey')

    axes.scatter(df['SMD before re-weighting'], ind, s=220,
                 edgecolors='grey', marker='^', c='indigo', #'lightgray',
                 label='PSM')  # facecolors='none',
    axes.scatter(df['SMD after re-weighting'], ind, s=220,
                 edgecolors='purple', marker='o', c='red',
                 label='PSM+IPW')

    # axes.scatter(df['SMD before re-weighting'], ind, s=220,
    #              edgecolors='grey', marker='^', c='indigo',  # 'lightgray',
    #              label='Before reweighting')  # facecolors='none',
    # axes.scatter(df['SMD after re-weighting'], ind, s=220,
    #              edgecolors='purple', marker='o', c='red',
    #              label='After reweighting')


    y_labels = ['\n'.join(wrap(label_map(x), 45)) for x in df['Unnamed: 0']]
    axes.set(yticks=ind, yticklabels=y_labels, ylim=[2 * width - 1, len(df)])
    plt.yticks(fontsize=26)
    plt.xticks(fontsize=26)
    plt.xlim(-0.15, 0.15)
    plt.xlabel('Standardized Mean Differences (SMD)', fontsize=24)
    plt.ylabel('Covariates', fontsize=24)

    axes.yaxis.grid()
    axes.legend(loc="upper right", prop={"size": 24})

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(False)
    plt.tight_layout()

    check_and_mkdir(output_dir)
    plt.savefig(output_dir + '_anyPASC_SMD_love_plot.png', bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + '_anyPASC_SMD_love_plot-small.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir + '_anyPASC_SMD_love_plot.pdf', bbox_inches='tight', transparent=True)
    plt.show()

    plt.close()
    print('Done!')
