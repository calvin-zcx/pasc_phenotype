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
    covs_columns = [
        'Female', 'Male', 'Other/Missing',
        'age@18-24', 'age@25-34', 'age@35-49', 'age@50-64',  # 'age@65+', # # expand 65
        '65-<75 years', '75-<85 years', '85+ years',
        'RE:Asian Non-Hispanic',
        'RE:Black or African American Non-Hispanic',
        'RE:Hispanic or Latino Any Race', 'RE:White Non-Hispanic',
        'RE:Other Non-Hispanic', 'RE:Unknown',
        'ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
        'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100', 'ADIMissing',
        '03/22-06/22', '07/22-10/22', '11/22-02/23',
        '03/23-06/23', '07/23-10/23', '11/23-02/24',
        '03/24-06/24', '07/24-10/24',
        # 'quart:01/22-03/22', 'quart:04/22-06/22', 'quart:07/22-09/22', 'quart:10/22-1/23',
        'inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits 3-4',
        'inpatient visits >=5',
        'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits 3-4',
        'outpatient visits >=5',
        'emergency visits 0', 'emergency visits 1-2', 'emergency visits 3-4',
        'emergency visits >=5',
        'BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ',
        'BMI: >=30 obese ', 'BMI: missing',
        'Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing',
        'PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
        'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
        'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
        'PaxRisk:Heart conditions', 'PaxRisk:Hypertension',
        # 'PaxRisk:HIV infection',
        'PaxRisk:Immunocompromised condition or weakened immune system', 'PaxRisk:Mental health conditions',
        'PaxRisk:Overweight and obesity',
        # 'PaxRisk:Pregnancy',
        'PaxRisk:Sickle cell disease or thalassemia',
        'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
        'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis',
        'Fully vaccinated - Pre-index', 'Partially vaccinated - Pre-index', 'No evidence - Pre-index',
        "DX: Coagulopathy", "DX: Peripheral vascular disorders ", "DX: Seizure/Epilepsy", "DX: Weight Loss",
        'DX: Obstructive sleep apnea', 'DX: Epstein-Barr and Infectious Mononucleosis (Mono)', 'DX: Herpes Zoster',
        'mental-base@Schizophrenia Spectrum and Other Psychotic Disorders',
        'mental-base@Depressive Disorders',
        'mental-base@Bipolar and Related Disorders',
        'mental-base@Anxiety Disorders',
        'mental-base@Obsessive-Compulsive and Related Disorders',
        'mental-base@Post-traumatic stress disorder',
        'mental-base@Bulimia nervosa',
        'mental-base@Binge eating disorder',
        # 'mental-base@premature ejaculation',
        'mental-base@Autism spectrum disorder',
        'mental-base@Premenstrual dysphoric disorder',
        'mental-base@SMI',
        'mental-base@non-SMI',
        'dxcovCNSLDN-base@MECFS',
        'dxcovCNSLDN-base@Narcolepsy',
        'dxcovCNSLDN-base@Pain',
        # 'dxcovCNSLDN-base@ADHD',
        'dxcovCNSLDN-base@alcohol opioid other substance ',
        'dxcovCNSLDN-base@traumatic brain injury',
        'dxcovCNSLDN-base@TBI-associated Symptoms'

    ]

    covs_map = {
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
        'ADI1-9': 'Area Depr.Idx 1-9',
        'ADI10-19': 'Area Depr.Idx 10-19',
        'ADI20-29': 'Area Depr.Idx 20-29',
        'ADI30-39': 'Area Depr.Idx 30-39',
        'ADI40-49': 'Area Depr.Idx 40-49',
        'ADI50-59': 'Area Depr.Idx 50-59',
        'ADI60-69': 'Area Depr.Idx 60-69',
        'ADI70-79': 'Area Depr.Idx 70-79',
        'ADI80-89': 'Area Depr.Idx 80-89',
        'ADI90-100': 'Area Depr.Idx 90-100',
        'ADIMissing': 'Area Depr.Idx Missing',
        '03/22-06/22': 'Infect time:03/22-06/22',
        '07/22-10/22': 'Infect time:07/22-10/22',
        '11/22-02/23': 'Infect time:11/22-02/23',
        '03/23-06/23':'Infect time:03/23-06/23',
        '07/23-10/23':'Infect time:07/23-10/23',
        '11/23-02/24': 'Infect time:11/23-02/24',
        '03/24-06/24': 'Infect time:03/24-06/24',
        '07/24-10/24':'Infect time:07/24-10/24',
        'inpatient visits 0': 'Hist.IP:0',
        'inpatient visits 1-2': 'Hist.IP:1-2',
        'inpatient visits 3-4': 'Hist.IP:3-4',
        'inpatient visits >=5': 'Hist.IP:≥5',
        'outpatient visits 0': 'Hist.OP:0',
        'outpatient visits 1-2': 'Hist.OP:1-2',
        'outpatient visits 3-4': 'Hist.OP:3-4',
        'outpatient visits >=5': 'Hist.OP:≥5',
        'emergency visits 0': 'Hist.ED:0',
        'emergency visits 1-2': 'Hist.ED:1-2',
        'emergency visits 3-4': 'Hist.ED:3-4',
        'emergency visits >=5': 'Hist.ED:≥5',
        'BMI: <18.5 under weight': 'BMI<18.5 underweight',
        'BMI: 18.5-<25 normal weight': 'BMI:18.5-<25 normal',
        'BMI: 25-<30 overweight ': 'BMI:25-<30 overweight',
        'BMI: >=30 obese ': 'BMI≥30 obese',
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
        'DX: Epstein-Barr and Infectious Mononucleosis (Mono)': 'EBV and Mono',
        'DX: Herpes Zoster': 'Herpes Zoster',
        'mental-base@Schizophrenia Spectrum and Other Psychotic Disorders':'Schizophrenia',
        'mental-base@Depressive Disorders':'Depression',
        'mental-base@Bipolar and Related Disorders':'Bipolar',
        'mental-base@Anxiety Disorders':'Anxiety',
        'mental-base@Obsessive-Compulsive and Related Disorders':'Obsessive-Compulsive Disorder',
        'mental-base@Post-traumatic stress disorder' : 'PTSD',
        'mental-base@Bulimia nervosa':'Bulimia nervosa',
        'mental-base@Binge eating disorder':'Binge eating disorder',
        'mental-base@premature ejaculation':'Premature ejaculation',
        'mental-base@Autism spectrum disorder':'Autism',
        'mental-base@Premenstrual dysphoric disorder':'Premenstrual dysphoric disorder',
        'mental-base@SMI':'Severe Mental Illness',
        'mental-base@non-SMI':'Non-Severe Mental Illness',
        'other-treat--1095-0-flag':'Bupropion use history',
        'dxcovCNSLDN-base@MECFS':'ME/CFS',
        'dxcovCNSLDN-base@Narcolepsy':'Narcolepsy',
        'dxcovCNSLDN-base@Pain':'Pain',
        'dxcovCNSLDN-base@alcohol opioid other substance ':'Alcohol/opioid/other substance',
        'dxcovCNSLDN-base@traumatic brain injury':'traumatic brain injury (TBI)',
        'dxcovCNSLDN-base@TBI-associated Symptoms':'TBI-associated Symptoms',
    }
    return covs_map[x]


if __name__ == '__main__':
    # pass

    indir = r'../data/recover/output/results/CNS-baseADHD-all-adhdCNS-inci-0-30s5/'
    indir = r'../data/recover/output/results/CNS-baseADHD-all-adhdCNS-inci-0-30s5-adjustless/'

    output_dir = indir + r'figure/'
    print('indir', indir)
    print('outdir', output_dir)

    df = pd.read_csv(indir + '1-any_pasc-evaluation_balance.csv')
    df = df.iloc[::-1]

    fig, axes = plt.subplots(figsize=(20, 33))
    # fig, axes = plt.subplots(figsize=(15, 20))

    ind = np.arange(len(df))

    width = 0.3
    axes.axvline(x=0, color='black')
    axes.axvline(x=0.1, linestyle='dashed', color='grey')
    axes.axvline(x=-0.1, linestyle='dashed', color='grey')

    axes.scatter(df['SMD before re-weighting'], ind, s=220,
                 edgecolors='grey', marker='^', c='lightgray',
                 label='Before reweighting')  # facecolors='none',
    axes.scatter(df['SMD after re-weighting'], ind, s=220,
                 edgecolors='purple', marker='o', c='red',
                 label='After reweighting')

    y_labels = ['\n'.join(wrap(label_map(x), 45)) for x in df['Unnamed: 0']]
    axes.set(yticks=ind, yticklabels=y_labels, ylim=[2 * width - 1, len(df)])
    plt.yticks(fontsize=26)
    plt.xticks(fontsize=26)
    plt.xlabel('Standardized Mean Differences (SMD)', fontsize=24)
    plt.ylabel('Covariates', fontsize=24)
    plt.xlim(-0.35, 0.35)
    print('selecting xlim for best visualization')

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
