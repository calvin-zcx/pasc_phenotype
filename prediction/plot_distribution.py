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
            df['pasc raw'] = pasc_raw

            row = df.head(1)[['paras', 'E[fit]', 'Std[fit]', 'CI0', 'CI1', 'std-boost', 'kfold-values',
                              'pasc', 'pasc raw']]

            if df_row is not None:
                df_row = df_row.append(row)
            else:
                df_row = row

    df_row.to_csv(dir_path + 'combined_predictive_performance-{}.csv'.format(severity))

    print('Done')
    return df_row


def combine_predictive_performance_revision2(database='INSIGHT', severity='all'):
    # df_pasc_info = pd.read_excel(
    #     r'C:/Users/zangc/Documents/Boston/workshop/2021-PASC/prediction/PASC_risk_factors_predictability.xlsx',
    #     sheet_name='person_counts_LR_res')
    df_pasc_info = pd.read_excel(
        r'output/PASC_risk_factors_predictability.xlsx',
        sheet_name='person_counts_LR_res')
    # df_pasc_info = df_pasc_info.sort_values(by=['Organ Domain', 'c_index'], ascending=False)
    df_pasc_info = df_pasc_info.sort_values(by=['c_index'], ascending=False)

    if database == 'OneFlorida':
        dir_path = 'output/factors/OneFlorida/elix/'
    elif database == 'INSIGHT':
        dir_path = 'output/factors/INSIGHT/elix/'
    elif database == 'Pooled':
        dir_path = 'output/factors/Pooled/elix/'

    # new order
    organ_list = [
        'Any PASC',
        'Diseases of the Nervous System',
        'Mental, Behavioral and Neurodevelopmental Disorders',
        'Diseases of the Skin and Subcutaneous Tissue',
        'Diseases of the Respiratory System',
        'Diseases of the Circulatory System',
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism',
        'Endocrine, Nutritional and Metabolic Diseases',
        'Diseases of the Digestive System',
        'Diseases of the Genitourinary System',
        'Diseases of the Musculoskeletal System and Connective Tissue',
        # 'Certain Infectious and Parasitic Diseases',
        'General']

    column_set = set([])
    # df_hr = None
    # df_p = None
    df_row = None
    for organ in organ_list:
        if organ == 'Any PASC':
            pasc_name_simple = ['Any PASC', ]
            pasc_name_raw = ['Any PASC', ]
        else:
            pasc_name_simple = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'PASC Name Simple'].tolist()
            pasc_name_raw = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'pasc'].tolist()

        for pasc, pasc_raw in zip(pasc_name_simple, pasc_name_raw):
            if pasc == 'Any PASC':
                fname = dir_path + 'any_pasc/' + 'any-at-least-1-pasc' + \
                        '-modeSelection-' + database + '-positive-{}.csv'.format(severity)
            else:
                fname = dir_path + 'every_pasc/' + 'PASC-' + pasc_raw.replace('/', '_') + \
                        '-modeSelection-' + database + '-positive-{}.csv'.format(severity)

            print(fname)
            df = pd.read_csv(fname)
            df['pasc'] = pasc
            df['pasc raw'] = pasc_raw
            df['Organ Domain'] = organ

            row = df.head(1)[['paras', 'E[fit]', 'Std[fit]', 'CI0', 'CI1', 'std-boost', 'kfold-values',
                              'pasc', 'pasc raw', 'Organ Domain']]

            if df_row is not None:
                df_row = df_row.append(row)
            else:
                df_row = row

    df_row.to_csv(dir_path + 'combined_predictive_performance-{}-revision2.csv'.format(severity))

    print('Done')
    return df_row


def print_predictive_performance(database='INSIGHT', severity='all'):
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

    df = pd.read_csv(dir_path + 'combined_predictive_performance-{}.csv'.format(severity))

    df = df.sort_values(by=['E[fit]'], ascending=False)

    results = []
    for key, row in df.iterrows():
        auc = row['E[fit]']
        low = row['CI0']
        high = row['CI1']
        pasc = row['pasc']
        r = '{} ({:.2f} ({:.2f}, {:.2f}))'.format(pasc, auc, low, high)
        results.append(r)
        print(r)

    print(', '.join(results))
    print('Done')
    return results


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


def plot_auc_bar_revision2(database='INSIGHT', severity='all', drop_pasc=[]):

    if database == 'OneFlorida':
        dir_path = 'output/factors/OneFlorida/elix/'
    elif database == 'INSIGHT':
        dir_path = 'output/factors/INSIGHT/elix/'
    elif database == 'Pooled':
        dir_path = 'output/factors/Pooled/elix/'

    df = pd.read_csv(dir_path + 'combined_predictive_performance-{}-revision2.csv'.format(severity))
    print('df_row.shape:', df.shape)

    organ_color = {
        'Any PASC':'red',
        'Diseases of the Nervous System': '#ADE8F4',
        'Mental, Behavioral and Neurodevelopmental Disorders': '#ADE8F4',
        'Diseases of the Skin and Subcutaneous Tissue':'#F1C0E8',
        'Diseases of the Respiratory System':'#FFD6A5',
        'Diseases of the Circulatory System':'#E63946',
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism':'#E63946',
        'Endocrine, Nutritional and Metabolic Diseases':'#CAFFBF',
        'Diseases of the Digestive System':'#E07A5F',
        'Diseases of the Genitourinary System':'#B0DDFF',
        'Diseases of the Musculoskeletal System and Connective Tissue':'#E5E5E5',
        # 'Certain Infectious and Parasitic Diseases',
        'General':'#D3D3D3'}

    df = df.drop(df[df['pasc'].isin(drop_pasc)].index)
    # df = df.sort_values(by=['E[fit]'], ascending=False)

    df['rank'] = df['E[fit]'].rank(ascending=False,)

    pasc_list = df['pasc'].tolist()
    auc_list_1 = df['E[fit]'].tolist()
    error_list_1 = df['Std[fit]'].tolist()
    ci0 = (df['E[fit]'] - df['CI0']).tolist()
    ci1 = (df['CI1'] - df['E[fit]']).tolist()
    yerr = np.vstack((ci0, ci1))
    colors = [organ_color[x] for x in df['Organ Domain']]
    ranks =  df['rank'].tolist()
    patterns = []
    for x in df['E[fit]']:
        if 0.7 <= x < 0.8:
            patterns.append('\\')
        elif x >= 0.8:
            patterns.append("o")
        else:
            patterns.append('')

    # patterns[0] = '|'
    # auc_list_2 = data[metrics + '_mean_of'].tolist()
    # error_list_2 = data[metrics + '_std_of'].tolist()

    fig, ax = plt.subplots(figsize=(15, 8.5))
    idx = np.arange(len(pasc_list))
    new_idx = np.asarray([2 * i for i in idx])
    # new_idx = np.asarray([i for i in idx])
    #
    bar = ax.bar(new_idx + 0.5, auc_list_1, 1, yerr=yerr,  color=colors, edgecolor='black', alpha=.8, hatch=patterns)
    # ax.bar(new_idx + .6, auc_list_2, .6, yerr=error_list_2, color='#98c1d9', edgecolor='black', alpha=.8)

    for i, rect in enumerate(bar):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height+ ci1[i] + 0.01, f'{ranks[i]:.0f}', ha='center', va='bottom')


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
    plt.savefig(dir_path + 'figure/' + 'c-index-bar-plot-{}-{}_revision2.pdf'.format(database, severity), dpi=600)
    plt.savefig(dir_path + 'figure/' + 'c-index-bar-plot-{}-{}_revision2.png'.format(database, severity), dpi=600)
    plt.show()
    plt.close()


def plot_auc_bar_sensitivity(database='INSIGHT', encoding='EC'):

    if database == 'OneFlorida':
        dir_path = 'output/factors/OneFlorida/elix/'
    elif database == 'INSIGHT':
        dir_path = 'output/factors/INSIGHT/elix/'

    df = pd.read_csv(dir_path + 'combined_predictive_performance-all.csv')
    df['pasc raw'] = df['pasc raw'].apply(lambda x : x.replace('/', '_'))
    print('df_row.shape:', df.shape)

    for m in ['LR', 'LIGHTGBM', 'MLP']:
        df_2 = pd.read_csv(dir_path + 'sensitivity/' + '{}_{}_{}_CI_res.csv'.format(encoding, database, m))
        df = pd.merge(
                df,
                df_2,
                left_on='pasc raw',
                right_on='Pasc',
                how='left',
                suffixes=('', '_' + m))
        print('combine:', m)


    df = df.sort_values(by=['E[fit]'], ascending=False)

    results = []
    pasc_list = df['pasc'].tolist()
    auc_list_1 = df['E[fit]'].tolist()
    error_list_1 = df['Std[fit]'].tolist()
    ci0 = (df['E[fit]'] - df['CI0']).tolist()
    ci1 = (df['CI1'] - df['E[fit]']).tolist()
    yerr = np.vstack((ci0, ci1))

    results.append((auc_list_1, yerr))
    # auc_list_2 = data[metrics + '_mean_of'].tolist()
    # error_list_2 = data[metrics + '_std_of'].tolist()

    for m in ['', '_LIGHTGBM', '_MLP']:
        auc = df['AUC' + m]
        ci0 = df['AUC CI' + m].apply(lambda x : float(x.split()[0].strip("(),")))
        ci1 = df['AUC CI' + m].apply(lambda x: float(x.split()[1].strip("(),")))

        auc_list = auc.tolist()
        ci0 = (auc - ci0).tolist()
        ci1 = (ci1 - auc).tolist()
        yerr = np.vstack((ci0, ci1))
        results.append((auc_list, yerr))

    fig, ax = plt.subplots(figsize=(15, 8.5))
    idx = np.arange(len(pasc_list))
    # new_idx = np.asarray([2 * i for i in idx])
    new_idx = np.asarray([i for i in idx])

    w = 0.2
    colors = ['#98c1d9', '#98d9b0', '#d998c1', '#d9b098']
    colors = ['#98c1d9', '#FFCC00', '#99CC99', '#FF0000']
    ax.bar(new_idx - 1.5*w, results[0][0], w, yerr=results[0][1], color=colors[0])#, edgecolor=None, alpha=.8)
    ax.bar(new_idx - 0.5*w, results[1][0], w, yerr=results[1][1], color=colors[1])#, edgecolor=None, alpha=.8)
    ax.bar(new_idx + 0.5*w, results[2][0], w, yerr=results[2][1], color=colors[2])#, edgecolor=None, alpha=.8)
    ax.bar(new_idx + 1.5*w, results[3][0], w, yerr=results[3][1], color=colors[3])#, edgecolor=None, alpha=.8)

    # ax.bar(new_idx + .6, auc_list_2, .6, yerr=error_list_2, color='#98c1d9', edgecolor='black', alpha=.8)

    # for x in range(len(pasc_list)):
    #     ax.text(x=idx[x] - 0.5, y=auc_list_1[x] + error_list_1[x] + 0.02, s=str(auc_list_1[x]), ha='center',
    #             va="center", rotation=90, rotation_mode="anchor")
    #     ax.text(x=idx[x] + 0.5, y=auc_list_2[x] + error_list_2[x] + 0.02, s=str(auc_list_2[x]), ha='center',
    #             va="center", rotation=90, rotation_mode="anchor")

    ax.set_xticks(new_idx )
    ax.set_xlim([-1, len(new_idx)])
    ax.set_ylim([.5, 1])

    ax.set_xticklabels(pasc_list, rotation=45, fontsize=15, ha='right', rotation_mode="anchor")
    ax.yaxis.grid()  # color='#D3D3D3', linestyle='--', linewidth=0.7)
    plt.ylabel('AUROC', fontsize=16) #, weight='bold')
    plt.yticks(fontsize=15)
    # ax.set(title=learner + ' ' + metrics)
    # plt.subplots_adjust(bottom=.3)
    handle_list = [mpatches.Patch(color=colors[0], label='COX'),  # '#e26d5c'
                   mpatches.Patch(color=colors[1], label='LR'),
                   mpatches.Patch(color=colors[2], label='GBM'),  # '#e26d5c'
                   mpatches.Patch(color=colors[3], label='DNN')
                   ]
    plt.legend(handles=handle_list, prop={'size': 15})
    plt.tight_layout()
    plt.savefig(dir_path + 'figure/' + 'auc_multiple_bars-plot-{}-{}.pdf'.format(database, encoding), dpi=600)
    plt.savefig(dir_path + 'figure/' + 'auc_multiple_bars_plot-{}-{}.png'.format(database, encoding), dpi=600)
    plt.show()
    plt.close()


def plot_auc_bar_sensitivity_revision2(database='INSIGHT', encoding='EC', drop_pasc = []):

    if database == 'OneFlorida':
        dir_path = 'output/factors/OneFlorida/elix/'
    elif database == 'INSIGHT':
        dir_path = 'output/factors/INSIGHT/elix/'

    df = pd.read_csv(dir_path + 'combined_predictive_performance-all-revision2.csv')
    df['pasc raw'] = df['pasc raw'].apply(lambda x : x.replace('/', '_'))

    df = df.drop(df[df['pasc'].isin(drop_pasc)].index)

    print('df_row.shape:', df.shape)

    for m in ['LR', 'LIGHTGBM', 'MLP']:
        df_2 = pd.read_csv(dir_path + 'sensitivity/' + '{}_{}_{}_CI_res.csv'.format(encoding, database, m))
        df = pd.merge(
                df,
                df_2,
                left_on='pasc raw',
                right_on='Pasc',
                how='left',
                suffixes=('', '_' + m))
        print('combine:', m)


    # df = df.sort_values(by=['E[fit]'], ascending=False)

    results = []
    pasc_list = df['pasc'].tolist()
    auc_list_1 = df['E[fit]'].tolist()
    error_list_1 = df['Std[fit]'].tolist()
    ci0 = (df['E[fit]'] - df['CI0']).tolist()
    ci1 = (df['CI1'] - df['E[fit]']).tolist()
    yerr = np.vstack((ci0, ci1))

    results.append((auc_list_1, yerr))
    # auc_list_2 = data[metrics + '_mean_of'].tolist()
    # error_list_2 = data[metrics + '_std_of'].tolist()

    for m in ['', '_LIGHTGBM', '_MLP']:
        auc = df['AUC' + m]
        ci0 = df['AUC CI' + m].apply(lambda x : float(x.split()[0].strip("(),")) if isinstance(x, str) else x)
        ci1 = df['AUC CI' + m].apply(lambda x: float(x.split()[1].strip("(),")) if isinstance(x, str) else x)

        auc_list = auc.tolist()
        ci0 = (auc - ci0).tolist()
        ci1 = (ci1 - auc).tolist()
        yerr = np.vstack((ci0, ci1))
        results.append((auc_list, yerr))

    fig, ax = plt.subplots(figsize=(18, 8.5))
    idx = np.arange(len(pasc_list))
    # new_idx = np.asarray([2 * i for i in idx])
    new_idx = np.asarray([i for i in idx])

    w = 0.2
    colors = ['#98c1d9', '#98d9b0', '#d998c1', '#d9b098']
    colors = ['#98c1d9', '#FFCC00', '#99CC99', '#FF0000']
    colors = ['#98c1d9','#FFCC00', '#CAFFBF', '#E63946']
    ax.bar(new_idx - 1.5*w, results[0][0], w, yerr=results[0][1], color=colors[0])#, edgecolor=colors[0])#, edgecolor=None, alpha=.8)
    ax.bar(new_idx - 0.5*w, results[1][0], w, yerr=results[1][1], color=colors[1])#, edgecolor=colors[1])#, edgecolor=None, alpha=.8)
    ax.bar(new_idx + 0.5*w, results[2][0], w, yerr=results[2][1], color=colors[2])#, edgecolor=colors[2])#, edgecolor=None, alpha=.8)
    ax.bar(new_idx + 1.5*w, results[3][0], w, yerr=results[3][1], color=colors[3])#, edgecolor=colors[3])#, edgecolor=None, alpha=.8)

    # ax.bar(new_idx + .6, auc_list_2, .6, yerr=error_list_2, color='#98c1d9', edgecolor='black', alpha=.8)

    # for x in range(len(pasc_list)):
    #     ax.text(x=idx[x] - 0.5, y=auc_list_1[x] + error_list_1[x] + 0.02, s=str(auc_list_1[x]), ha='center',
    #             va="center", rotation=90, rotation_mode="anchor")
    #     ax.text(x=idx[x] + 0.5, y=auc_list_2[x] + error_list_2[x] + 0.02, s=str(auc_list_2[x]), ha='center',
    #             va="center", rotation=90, rotation_mode="anchor")

    ax.set_xticks(new_idx )
    ax.set_xlim([-1, len(new_idx)])
    ax.set_ylim([.5, 1])

    ax.set_xticklabels(pasc_list, rotation=45, fontsize=15, ha='right', rotation_mode="anchor")
    ax.yaxis.grid()  # color='#D3D3D3', linestyle='--', linewidth=0.7)
    plt.ylabel('AUROC', fontsize=16) #, weight='bold')
    plt.yticks(fontsize=15)
    # ax.set(title=learner + ' ' + metrics)
    # plt.subplots_adjust(bottom=.3)
    handle_list = [mpatches.Patch(color=colors[0], label='COX'),  # '#e26d5c'
                   mpatches.Patch(color=colors[1], label='LR'),
                   mpatches.Patch(color=colors[2], label='GBM'),  # '#e26d5c'
                   mpatches.Patch(color=colors[3], label='DNN')
                   ]
    plt.legend(handles=handle_list, prop={'size': 15})
    plt.tight_layout()
    plt.savefig(dir_path + 'figure/' + 'auc_multiple_bars-plot-{}-{}-revision2.pdf'.format(database, encoding), dpi=600)
    plt.savefig(dir_path + 'figure/' + 'auc_multiple_bars_plot-{}-{}-revision2.png'.format(database, encoding), dpi=600)
    plt.show()
    plt.close()


if __name__ == '__main__':
    start_time = time.time()
    # plot_person_pasc_counts()
    # print_predictive_performance(database='INSIGHT', severity='all')

    # df_row = combine_predictive_performance(database='INSIGHT', severity='all')
    # df_row = combine_predictive_performance(database='INSIGHT', severity='inpatienticu')
    # df_row = combine_predictive_performance(database='INSIGHT', severity='outpatient')
    #
    # df_row = combine_predictive_performance(database='OneFlorida', severity='all')
    # df_row = combine_predictive_performance(database='OneFlorida', severity='inpatienticu')
    # df_row = combine_predictive_performance(database='OneFlorida', severity='outpatient')

    # plot_auc_bar(database='INSIGHT', severity='all')
    # plot_auc_bar(database='INSIGHT', severity='inpatienticu')
    # plot_auc_bar(database='INSIGHT', severity='outpatient')

    # plot_auc_bar(database='OneFlorida', severity='all')
    # plot_auc_bar(database='OneFlorida', severity='inpatienticu')
    # plot_auc_bar(database='OneFlorida', severity='outpatient')

    # plot_auc_bar_sensitivity(database='INSIGHT', encoding='EC')
    # plot_auc_bar_sensitivity(database='INSIGHT', encoding='icd_med')


    # 2023-10-2 revision 2
    # df_row = combine_predictive_performance_revision2(database='INSIGHT', severity='all')
    plot_auc_bar_revision2(database='INSIGHT', severity='all', drop_pasc=['Pressure ulcers'])
    zz
    # df_row = combine_predictive_performance_revision2(database='OneFlorida', severity='all')
    # plot_auc_bar_revision2(database='OneFlorida', severity='all', drop_pasc=['Pressure ulcers'])

    plot_auc_bar_sensitivity_revision2(database='INSIGHT', encoding='EC', drop_pasc=['Pressure ulcers', 'Any PASC'])
    plot_auc_bar_sensitivity_revision2(database='INSIGHT', encoding='icd_med', drop_pasc=['Pressure ulcers', 'Any PASC'])

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
