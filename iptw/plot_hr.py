import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

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
print = functools.partial(print, flush=True)
import zepid
from zepid.graphics import EffectMeasurePlot
import shlex
np.random.seed(0)
random.seed(0)


def plot_forest_for_dx():
    df = pd.read_csv(r'../data/V15_COVID19/output/character/specificDX/causal_effects_specific-v2.csv')
    df_select = df.sort_values(by='hr-w', ascending=False)
    df_select = df_select.loc[df_select['hr-w-p']<=0.05, :]
    # df_select = df_select.loc[df_select['hr-w']>1, :]

    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    for key, row in df_select.iterrows():
        name = row['pasc']
        hr = row['hr-w']
        ci = stringlist_2_list(row['hr-w-CI'])
        p = row['hr-w-p']

        if len(name.split()) >= 6:
            name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])
        labs.append(name)
        measure.append(hr)
        lower.append(ci[0])
        upper.append(ci[1])
        pval.append(p)

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(scale='log')

    p.labels(effectmeasure='aHR')
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    # p.colors(pointshape="s", errorbarcolor=c,  pointcolor=c)
    ax = p.plot(figsize=(9, 10), t_adjuster=0.03, max_value=13, min_value=0.5, size=5, decimal=2)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/specificDX/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'dx_hr_forest.png', bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'dx_hr_forest.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    # plt.close()


def plot_forest_for_med():
    df = pd.read_csv(r'../data/V15_COVID19/output/character/specificMed/causal_effects_specific_med.csv')
    df_select = df.sort_values(by='hr-w', ascending=False)
    df_select = df_select.loc[df_select['hr-w-p']<=0.05, :]
    df_select = df_select.loc[df_select['no. pasc in +']>=10, :]
    # df_select = df_select.loc[df_select['hr-w']>1, :]

    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    for key, row in df_select.iterrows():
        name = row['pasc']
        name_label = row['pasc-med'].strip('][').split(',')
        name_label = name_label[2].strip().strip(r'\'').lower()
        hr = row['hr-w']
        ci = stringlist_2_list(row['hr-w-CI'])
        p = row['hr-w-p']

        if len(name_label.split()) >= 7:
            name_label = ' '.join(name_label.split()[:5]) + '\n' + ' '.join(name_label.split()[5:])

        labs.append(name + '-' + name_label)
        measure.append(hr)
        lower.append(ci[0])
        upper.append(ci[1])
        pval.append(p)

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(effectmeasure='aHR', scale='log')

    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    # p.colors(pointshape="s", errorbarcolor=c,  pointcolor=c)
    ax = p.plot(figsize=(10, 12.5), t_adjuster=0.015, max_value=5, min_value=0.4, size=5, decimal=2)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/specificMed/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'med_hr_forest.png', bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'med_hr_forest.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    # plt.close()


def combine_pasc_list():
    df_icd = pd.read_csv('../data/V15_COVID19/output/character/pcr_cohorts_ICD_cnts_followup-ALL.csv')
    print('df_icd.shape:', df_icd.shape)
    df_pasc_list = pd.read_excel(r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx',
                                 sheet_name=r'PASC Screening List',
                                 usecols="A:N")
    print('df_pasc_list.shape', df_pasc_list.shape)

    df_icd_combined = pd.merge(df_icd, df_pasc_list, left_on='ICD', right_on='ICD-10-CM Code', how='left')
    df_icd_combined.to_csv('../data/V15_COVID19/output/character/pcr_cohorts_ICD_cnts_followup-ALL-combined_PASC.csv')

    df_pasc_combined = pd.merge(df_pasc_list, df_icd,  left_on='ICD-10-CM Code', right_on='ICD', how='left')
    df_pasc_combined.to_csv('../data/V15_COVID19/output/character/PASC_Adult_Combined_List_20220127_v3_combined_RWD.csv')


if __name__ == '__main__':
    # plot_forest_for_dx()
    # plot_forest_for_med()
    combine_pasc_list()