import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib as mpl
import time

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
from misc import utils
from matplotlib.colors import LogNorm, Normalize
from scipy.stats import uniform
from scipy.stats import randint


def combine_risk_p_value(database='INSIGHT', star=False, pvalue=0.01):
    df_pasc_info = pd.read_excel(
        r'C:/Users/zangc/Documents/Boston/workshop/2021-PASC/prediction/PASC_risk_factors_predictability.xlsx',
        sheet_name='person_counts_LR_res')
    df_pasc_info = df_pasc_info.sort_values(by=['Organ Domain', 'c_index'], ascending=False)

    if database == 'OneFlorida':
        dir_path = 'output/factors/OneFlorida/elix/'
    else:
        dir_path = 'output/factors/INSIGHT/elix/'

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
    df_hr = None
    df_p = None
    df_row = None
    for organ in organ_list:
        pasc_name_simple = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'PASC Name Simple'].tolist()
        pasc_name_raw = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'pasc'].tolist()
        for pasc, pasc_raw in zip(pasc_name_simple, pasc_name_raw):
            fname = dir_path + 'every_pasc/' + 'PASC-' + pasc_raw.replace('/',
                                                                          '_') + '-riskFactor-' + database + '-positive-all.csv'
            print(fname)
            df = pd.read_csv(fname)

            df['pasc'] = pasc
            if df_row is not None:
                df_row = df_row.append(df[['covariate', 'pasc', 'HR', 'CI-95% lower-bound', 'CI-95% upper-bound', 'p-Value']])
            else:
                df_row = df[['covariate', 'pasc', 'HR', 'CI-95% lower-bound', 'CI-95% upper-bound', 'p-Value']]



            df_sort = df.sort_values(by=['Unnamed: 0'], ascending=True).set_index('covariate')
            if df_hr is not None:
                df_hr = pd.merge(df_hr, df_sort[['HR']].rename(columns={'HR': pasc}),
                                 left_on='covariate',
                                 right_on='covariate',
                                 how='outer',
                                 suffixes=('', ''))  # left_on='pasc', right_on='pasc'
            else:
                df_hr = df_sort[['HR']].rename(columns={'HR': pasc})

            if df_p is not None:
                df_p = pd.merge(df_p, df_sort[['p-Value']].rename(columns={'p-Value': pasc}),
                                left_on='covariate',
                                right_on='covariate',
                                how='outer',
                                suffixes=('', '')
                                )  # left_on='pasc', right_on='pasc'
            else:
                df_p = df_sort[['p-Value']].rename(columns={'p-Value':pasc})

    print('df_hr.shape', df_hr.shape)
    print('df_p.shape', df_p.shape)

    df_hr.to_csv(dir_path + 'combined_risk.csv')
    df_p.to_csv(dir_path + 'combined_p-val.csv')
    df_row.to_csv(dir_path + 'combined_row_format.csv')

    print('Done')


def plot_heatmap_for_risk_grouped_by_organ(database='INSIGHT', star=False, pvalue=0.01):
    df_pasc_info = pd.read_excel(
        r'C:/Users/zangc/Documents/Boston/workshop/2021-PASC/prediction/PASC_risk_factors_predictability.xlsx',
        sheet_name='person_counts_LR_res')
    df_pasc_info = df_pasc_info.sort_values(by=['Organ Domain', 'c_index'], ascending=False)

    if database == 'OneFlorida':
        dir_path = 'output/factors/OneFlorida/elix/'
    else:
        dir_path = 'output/factors/INSIGHT/elix/'

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
    for organ in organ_list:
        pasc_name_simple = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'PASC Name Simple'].tolist()
        pasc_name_raw = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'pasc'].tolist()
        for pasc, pasc_raw in zip(pasc_name_simple, pasc_name_raw):
            fname = dir_path + 'every_pasc/' + 'PASC-' + pasc_raw.replace('/',
                                                                          '_') + '-riskFactor-' + database + '-positive-all.csv'
            print(fname)
            df = pd.read_csv(fname)
            df_sig = df.loc[df['p-Value'] <= pvalue, :]
            # sig_list = list(df_sig[['Unnamed: 0', 'covariate']].to_records(index=False))
            sig_list = list(df_sig['covariate'])
            column_set.update(sig_list)

    column_list = sorted(column_set)
    print(column_list)

    column_list = [
        'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ', 'BMI: <18.5 under weight', 'BMI: >=30 obese ',
        'Black or African American', 'DX: Alcohol Abuse', 'DX: Anemia', 'DX: Arrythmia', 'DX: Cancer',
        'DX: Chronic Kidney Disease', 'DX: Cirrhosis', 'DX: Coagulopathy', 'DX: Congestive Heart Failure',
        'DX: Coronary Artery Disease', 'DX: Dementia', 'DX: Diabetes Type 1', 'DX: Diabetes Type 2',
        'DX: End Stage Renal Disease on Dialysis', 'DX: Hemiplegia', 'DX: Hypertension',
        'DX: Hypertension and Type 1 or 2 Diabetes Diagnosis', 'DX: Inflammatory Bowel Disorder',
        'DX: Peripheral vascular disorders ', 'DX: Pregnant', 'DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)',
        'DX: Sickle Cell', 'DX: Weight Loss', 'MEDICATION: Corticosteroids', 'MEDICATION: Immunosuppressant drug',
        'Other',
        'White', 'emergency visits >=3', 'not hospitalized', 'hospitalized', 'icu',
    ]

    results = []
    row_name_list = []
    for organ in organ_list:
        pasc_name_simple = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'PASC Name Simple'].tolist()
        pasc_name_raw = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'pasc'].tolist()
        for pasc, pasc_raw in zip(pasc_name_simple, pasc_name_raw):
            fname = dir_path + 'every_pasc/' + 'PASC-' + pasc_raw.replace('/',
                                                                          '_') + '-riskFactor-' + database + '-positive-all.csv'
            print(fname)
            df = pd.read_csv(fname)

            hr_list = []
            ci_low_list = []
            ci_upp_list = []
            p_value_list = []

            if df_pasc_info.loc[df_pasc_info['pasc'] == pasc_raw, 'c_index'].values[0] < 0.7:
                continue

            row_name_list.append(pasc)
            for c in column_list:
                hr = df.loc[df['covariate'] == c, 'HR'].values[0]
                ci_low = df.loc[df['covariate'] == c, 'CI-95% lower-bound'].values[0]
                ci_upp = df.loc[df['covariate'] == c, 'CI-95% upper-bound'].values[0]
                p_value = df.loc[df['covariate'] == c, 'p-Value'].values[0]
                hr_list.append(hr)
                ci_low_list.append(ci_low)
                ci_upp_list.append(ci_upp)
                p_value_list.append(p_value)

            results.append(hr_list)

    df_heat_value = pd.DataFrame(results, columns=column_list, index=row_name_list)

    data = df_heat_value
    asp = data.shape[0] / float(data.shape[1])
    figw = 12  # 8
    figh = figw * asp * 1.2 # * 2.5

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    # f, (ax, cbar_ax) = plt.subplots(2, figsize=(15, 15), gridspec_kw=grid_kws)

    cmap = sns.cm.icefire_r
    cmap = sns.cm.rocket_r
    # if type == 'cifdiff-pvalue':
    #     # norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
    #     norm = LogNorm(vmin=0.05 / 137, vmax=1)
    #     fmt = ".1g"
    #     figw = 15  # 8
    #     figh = figw * asp
    # elif type == 'cifdiff':
    #     norm = Normalize(vmin=0, vmax=100, clip=True)
    #     fmt = ".1f"
    # elif (type == 'cif') or (type == 'cifneg'):
    #     # 256.8 for Insight
    #     # ? for florida?
    #     norm = Normalize(vmin=data.min().min(), vmax=256.8)  # data.max().max()
    #     fmt = ".1f"
    # else:
    #     norm = Normalize(vmin=data.min().min(), vmax=data.max().max())
    #     fmt = ".1f"
    # # norm = Normalize(vmin=1, vmax=5, clip=True)

    fmt = ".1f"
    norm = Normalize(vmin=data.min().min(), vmax=data.max().max())
    # gridspec_kw = {"width_ratios": [1, 3, 3, 2, 2, 10]}
    gridspec_kw = {"width_ratios": [1, 2, 2, 2, 2, 8]}
    heatmapkws = dict(square=False, cbar=False, cmap=cmap, linewidths=0.3,
                      vmin=data.min().min(), vmax=data.max().max(), fmt=fmt, norm=norm)
    tickskw = dict(xticklabels=False, yticklabels=False)

    left = 0.22
    # right = 0.87
    bottom = 0.1
    top = 0.85
    fig, axes = plt.subplots(figsize=(figw, figh))  # ncols=6, nrows=1, , gridspec_kw=gridspec_kw
    # plt.subplots_adjust(left=left, bottom=bottom, top=top, wspace=0.08, hspace=0.08 * asp)
    sns.heatmap(data, ax=axes, yticklabels=row_name_list, annot=True, **heatmapkws)
    plt.setp(axes.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    plt.show()

    fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(figw, figh), gridspec_kw=gridspec_kw)
    plt.subplots_adjust(left=left, bottom=bottom, top=top, wspace=0.08, hspace=0.08 * asp)
    sns.heatmap(data.iloc[:, 0:1], ax=axes[0], yticklabels=labs, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 1:3], ax=axes[1], yticklabels=False, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 3:5], ax=axes[2], yticklabels=False, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 5:7], ax=axes[3], yticklabels=False, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 7:9], ax=axes[4], yticklabels=False, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 9:], ax=axes[5], yticklabels=False, annot=True, **heatmapkws)

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

    # axes[1, 0].set_yticklabels([9])
    # axes[1, 1].set_xticklabels([4, 5, 6, 7, 8])
    # axes[1, 2].set_xticklabels([9, 10, 11])

    cax = fig.add_axes([0.92, 0.12, 0.025, 0.7])  # [left, bottom, width, height]
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if type == 'cifdiff-pvalue':
        fig.colorbar(sm, cax=cax, format='%.e')
    else:
        fig.colorbar(sm, cax=cax)
    output_dir = r'../data/{}/output/character/outcome/figure/{}/'.format(database, type)
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-month{}{}-p{:.3f}-V2.png'.format(
        database, type, month, select_criteria, pvalue), bbox_inches='tight', dpi=700)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-month{}{}-p{:.3f}-V2.pdf'.format(
        database, type, month, select_criteria, pvalue), bbox_inches='tight', transparent=True)

    plt.show()
    print()


def manhattan_plt(database='INSIGHT', ):
    df_pasc_info = pd.read_excel(
        r'C:/Users/zangc/Documents/Boston/workshop/2021-PASC/prediction/PASC_risk_factors_predictability.xlsx',
        sheet_name='person_counts_LR_res')
    df_pasc_info = df_pasc_info.sort_values(by=['Organ Domain', 'c_index'], ascending=False)

    if database == 'OneFlorida':
        dir_path = 'output/factors/OneFlorida/elix/'
    else:
        dir_path = 'output/factors/INSIGHT/elix/'
    df_hr = pd.read_csv(dir_path + 'combined_risk.csv')
    df_p = pd.read_csv(dir_path + 'combined_p-val.csv')
    df_p = df_p.set_index('covariate')
    df_hr = df_hr.set_index('covariate')
    df_row = pd.read_csv(dir_path + 'combined_row_format.csv')


    df_p = -np.log10(df_p)

    # df = pd.DataFrame({'gene': ['gene-%i' % i for i in np.arange(10000)],
    #                 'pvalue': uniform.rvs(size=10000),
    #                 'chromosome': ['ch-%i' % i for i in randint.rvs(0, 12, size=10000)]})

    # -log_10(pvalue)
    df = df_row
    df['minuslog10pvalue'] = -np.log10(df['p-Value'])
    # df['pasc'] = df['pasc'].astype('category')
    # df.chromosome = df.chromosome.cat.set_categories(['ch-%i' % i for i in range(12)], ordered=True)
    # df = df.sort_values('pasc')

    # How to plot gene vs. -log10(pvalue) and colour it by chromosome?
    df['ind'] = range(len(df))
    df_grouped = df.groupby(('pasc'))
    # manhattan plot
    fig = plt.figure(figsize=(18, 8))  # Set the figure size
    ax = fig.add_subplot(111)
    colors = ['darkred', 'darkgreen', 'darkblue', 'gold']
    x_labels = []
    x_labels_pos = []
    for num, (name, group) in enumerate(df_grouped):
        group.plot(kind='scatter', x='ind', y='minuslog10pvalue', color=colors[num % len(colors)], ax=ax, s=group.HR * 30 )
        x_labels.append(name)
        x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0]) / 2))
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)

    # set axis limits
    ax.set_xlim([0, len(df)])
    ax.set_ylim([0, 3.5])

    # x axis label
    ax.set_xlabel('PASC')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor")
    # show the graph
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    manhattan_plt(database='INSIGHT', )
    # combine_risk_p_value(database='INSIGHT', star=False, pvalue=0.01)
    # plot_heatmap_for_risk_grouped_by_organ(database='INSIGHT', star=False, pvalue=0.05)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
