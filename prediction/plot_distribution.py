import os
import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import argparse
import time
import random
import pickle
import ast
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from datetime import datetime
import functools
import seaborn as sns

print = functools.partial(print, flush=True)
from misc import utils
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test, proportional_hazard_test, logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import k_fold_cross_validation
from PRModels import ml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def combine_predictive_performance(database='INSIGHT', severity='all'):
    # df_pasc_info = pd.read_excel(
    #     r'C:/Users/zangc/Documents/Boston/workshop/2021-PASC/prediction/PASC_risk_factors_predictability.xlsx',
    #     sheet_name='person_counts_LR_res')
    df_pasc_info = pd.read_excel(
        r'output/PASC_risk_factors_predictability.xlsx',
        sheet_name='person_counts_LR_res')
    df_pasc_info = df_pasc_info.sort_values(by=['Organ Domain', 'c_index'], ascending=False)

    if database == 'OneFlorida':
        dir_path = 'output/factors/OneFlorida/elix/'
    elif database == 'INSIGHT':
        dir_path = 'output/factors/INSIGHT/elix/'
    elif database == 'Pooled':
        dir_path = 'output/factors/Pooled/elix/'

    organ_list = [
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism',
        'Diseases of the Circulatory System',
        'Endocrine, Nutritional and Metabolic Diseases',
        'Diseases of the Respiratory System',
        'Diseases of the Nervous System',
        'Mental, Behavioral and Neurodevelopmental Disorders',
        'Diseases of the Skin and Subcutaneous Tissue',
        'Diseases of the Digestive System',
        'Diseases of the Genitourinary System',
        'Diseases of the Musculoskeletal System and Connective Tissue',
        'General',
    ]

    column_set = set([])
    # df_hr = None
    # df_p = None
    df_row = None
    for organ in organ_list:
        pasc_name_simple = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'PASC Name Simple'].tolist()
        pasc_name_raw = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'pasc'].tolist()
        for pasc, pasc_raw in zip(pasc_name_simple, pasc_name_raw):
            fname = dir_path + 'every_pasc/' + 'PASC-' + pasc_raw.replace('/', '_') + \
                    '-modeSelection-' + database + '-positive-{}.csv'.format(severity)

            print(fname)
            df = pd.read_csv(fname)
            df['pasc'] = pasc

            row = df.head(1)[['paras', 'E[fit]', 'Std[fit]', 'CI0', 'CI1', 'std-boost', 'kfold-values', 'pasc']]

            if df_row is not None:
                df_row = df_row.append(row)
            else:
                df_row = row

    df_row.to_csv(dir_path + 'combined_predictive_performance-{}.csv'.format(severity))

    print('Done')
    return df_row


def plot_person_pasc_counts():
    dfnyc = pd.read_csv(r'output/dataset/INSIGHT/stats/person_pasc_counts_INSIGHT.csv')
    dfflor = pd.read_csv(r'output/dataset/OneFlorida/stats/person_pasc_counts_OneFlorida.csv')

    # dfnyc['count'].hist(bins=26, density=1)
    # dfflor['count'].hist(bins=26, density=1)
    # plt.show()

    # nnyc, binsnyc, patchesnyc = plt.hist(x=dfnyc['count'], bins=range(26),
    #                                      color="#F65453", alpha=0.5, density=1, label='NYC', rwidth=1)
    # nflor, binsflor, patchesflor = plt.hist(x=dfflor['count'], bins=range(26),
    #                                         color='#0504aa', alpha=0.5, density=1, label='Florida', rwidth=1)

    sns.histplot(data=dfnyc, x="count", stat='probability', discrete=True, color="red", label='NYC', shrink=.8, fill=False )
    sns.histplot(data=dfflor, x="count", stat='probability', discrete=True, color='#4895ef', label='Florida', shrink=.8, fill=False) # #0504aa

    # sns.ecdfplot(data=dfnyc, x="count", stat="proportion", complementary=True, color="red", label='NYC')
    # sns.ecdfplot(data=dfflor, x="count",  stat="proportion", complementary=True, color="#4895ef", label='NYC')

    # plt.yscale('log')
    # plt.ylim(top=1.1)
    # plt.grid(axis='y', alpha=0.75)
    plt.xlabel('No. of PASC Per Patient')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.show()


def plot_auc_bar(database='INSIGHT', severity='all'):

    if database == 'OneFlorida':
        dir_path = 'output/factors/OneFlorida/elix/'
    elif database == 'INSIGHT':
        dir_path = 'output/factors/INSIGHT/elix/'
    elif database == 'Pooled':
        dir_path = 'output/factors/Pooled/elix/'

    df = pd.read_csv(dir_path + 'combined_predictive_performance-{}.csv'.format(severity))
    print('df_row.shape:', df.shape)

    df = df.sort_values(by=['E[fit]'], ascending=False)

    pasc_list = df['pasc'].tolist()
    auc_list_1 = df['E[fit]'].tolist()
    error_list_1 = df['Std[fit]'].tolist()
    ci0 = (df['E[fit]'] - df['CI0']).tolist()
    ci1 = (df['CI1'] - df['E[fit]']).tolist()
    yerr = np.vstack((ci0, ci1))

    # auc_list_2 = data[metrics + '_mean_of'].tolist()
    # error_list_2 = data[metrics + '_std_of'].tolist()

    fig, ax = plt.subplots(figsize=(15, 8.5))
    idx = np.arange(len(pasc_list))
    new_idx = np.asarray([2 * i for i in idx])
    # new_idx = np.asarray([i for i in idx])

    ax.bar(new_idx + 0.5, auc_list_1, 1, yerr=yerr, color='#98c1d9', edgecolor='black', alpha=.8)
    # ax.bar(new_idx + .6, auc_list_2, .6, yerr=error_list_2, color='#98c1d9', edgecolor='black', alpha=.8)

    # for x in range(len(pasc_list)):
    #     ax.text(x=idx[x] - 0.5, y=auc_list_1[x] + error_list_1[x] + 0.02, s=str(auc_list_1[x]), ha='center',
    #             va="center", rotation=90, rotation_mode="anchor")
    #     ax.text(x=idx[x] + 0.5, y=auc_list_2[x] + error_list_2[x] + 0.02, s=str(auc_list_2[x]), ha='center',
    #             va="center", rotation=90, rotation_mode="anchor")

    ax.set_xticks(new_idx + .5)
    ax.set_xlim([-1, len(new_idx) * 2])
    ax.set_ylim([.5, 1])
    # x_show = []
    # for
    ax.set_xticklabels(pasc_list, rotation=45, fontsize=15, ha='right', rotation_mode="anchor")
    ax.yaxis.grid()  # color='#D3D3D3', linestyle='--', linewidth=0.7)
    plt.ylabel('C-Index', fontsize=16) #, weight='bold')
    plt.yticks(fontsize=15)
    # ax.set(title=learner + ' ' + metrics)
    # plt.subplots_adjust(bottom=.3)
    # handle_list = [mpatches.Patch(color='#98c1d9', label='INSIGHT'),  # '#e26d5c'
    #                mpatches.Patch(color='#98c1d9', label='OneFlorida+')
                   # ]
    # plt.legend(handles=handle_list, prop={'size': 15})
    plt.tight_layout()
    plt.savefig(dir_path + 'figure/' + 'c-index-bar-plot-{}-{}.pdf'.format(database, severity), dpi=600)
    plt.savefig(dir_path + 'figure/' + 'c-index-bar-plot-{}-{}.png'.format(database, severity), dpi=600)
    plt.show()
    plt.close()


if __name__ == '__main__':
    start_time = time.time()
    # plot_person_pasc_counts()
    df_row = combine_predictive_performance(database='INSIGHT', severity='all')
    df_row = combine_predictive_performance(database='INSIGHT', severity='inpatienticu')
    df_row = combine_predictive_performance(database='INSIGHT', severity='outpatient')

    df_row = combine_predictive_performance(database='OneFlorida', severity='all')
    df_row = combine_predictive_performance(database='OneFlorida', severity='inpatienticu')
    df_row = combine_predictive_performance(database='OneFlorida', severity='outpatient')

    plot_auc_bar(database='INSIGHT', severity='all')
    plot_auc_bar(database='INSIGHT', severity='inpatienticu')
    plot_auc_bar(database='INSIGHT', severity='outpatient')

    # plot_auc_bar(database='OneFlorida', severity='all')
    # plot_auc_bar(database='OneFlorida', severity='inpatienticu')
    # plot_auc_bar(database='OneFlorida', severity='outpatient')

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
