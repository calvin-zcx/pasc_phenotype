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
    }
    return covs_map[x]


if __name__ == '__main__':
    # pass
    indir = r'../data/recover/output/results/Paxlovid-atrisk-all-pcornet-V3/'
    output_dir = indir + r'figure/'

    df = pd.read_csv(indir + '1-any_pasc-evaluation_balance.csv')
    df = df.iloc[::-1]

    fig, axes = plt.subplots(figsize=(20, 28))
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
