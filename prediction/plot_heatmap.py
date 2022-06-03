import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib as mpl
import time
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def collect_covariate_name():
    df = pd.read_excel('output/factors/INSIGHT/elix/cov_name_mapping.xlsx')
    print('df.shape:', df.shape)
    pasc_name = {}
    for key, row in df.iterrows():
        pasc_name[row['covariate']] = row['name']

    print('len(pasc_name):', len(pasc_name))

    df_pasc_info = pd.read_excel(
        r'C:/Users/zangc/Documents/Boston/workshop/2021-PASC/prediction/PASC_risk_factors_predictability.xlsx',
        sheet_name='person_counts_LR_res')
    # df_pasc_info = df_pasc_info.sort_values(by=['Organ Domain', 'c_index'], ascending=True)
    df_pasc_info = df_pasc_info.sort_values(by=['c_index'], ascending=False)
    pasc_name_list = list(df_pasc_info['PASC Name Simple'])
    return pasc_name, pasc_name_list


def combine_risk_p_value(database='INSIGHT'):
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
                df_row = df_row.append(
                    df[['covariate', 'pasc', 'HR', 'CI-95% lower-bound', 'CI-95% upper-bound', 'p-Value']])
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
                df_p = df_sort[['p-Value']].rename(columns={'p-Value': pasc})

    print('df_hr.shape', df_hr.shape)
    print('df_p.shape', df_p.shape)

    df_hr.to_csv(dir_path + 'combined_risk.csv')
    df_p.to_csv(dir_path + 'combined_p-val.csv')
    df_row.to_csv(dir_path + 'combined_row_format.csv')

    print('Done')
    return df_hr, df_p, df_row


def combine_risk_p_value_with_interaction(database='INSIGHT', severity='all'):
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
    # df_hr = None
    # df_p = None
    df_row = None
    for organ in organ_list:
        pasc_name_simple = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'PASC Name Simple'].tolist()
        pasc_name_raw = df_pasc_info.loc[df_pasc_info['Organ Domain'] == organ, 'pasc'].tolist()
        for pasc, pasc_raw in zip(pasc_name_simple, pasc_name_raw):
            fname = dir_path + 'every_pasc/' + 'PASC-' + pasc_raw.replace('/', '_') + \
                    '-riskFactor-' + database + '-positive-{}.csv'.format(severity)
            fname_inter = dir_path + 'every_pasc/' + 'PASC-' + pasc_raw.replace('/', '_') + \
                          '-riskFactor-' + database + '-all-{}-interaction.csv'.format(severity)
            print(fname)
            df = pd.read_csv(fname)
            df = df.sort_values(by=['Unnamed: 0'], ascending=True)
            df['pasc'] = pasc

            print(fname_inter)
            df_inter = pd.read_csv(fname_inter)
            df_inter = df_inter.sort_values(by=['Unnamed: 0'], ascending=True)
            df_inter_a = df_inter.loc[~df_inter['covariate'].str.startswith('Inter+'), :].copy()
            df_inter_b = df_inter.loc[df_inter['covariate'].str.startswith('Inter+'), :].copy()

            df_inter_b['covariate'] = df_inter_b['covariate'].apply(lambda x: x.replace('Inter+', ''))

            df_inter_selected = pd.merge(
                df_inter_a[['covariate', 'HR', 'CI-95% lower-bound', 'CI-95% upper-bound', 'p-Value']],
                df_inter_b[['covariate', 'HR', 'CI-95% lower-bound', 'CI-95% upper-bound', 'p-Value']],
                left_on='covariate',
                right_on='covariate',
                how='outer',
                suffixes=('_base', '_inter'))  # left_on='pasc', right_on='pasc'
            df_result = pd.merge(
                df[['covariate', 'pasc', 'HR', 'CI-95% lower-bound', 'CI-95% upper-bound', 'p-Value']],
                df_inter_selected,
                left_on='covariate',
                right_on='covariate',
                how='outer',
                suffixes=('', ''))
            if df_row is not None:
                df_row = df_row.append(df_result)
            else:
                df_row = df_result

    df_row.to_csv(dir_path + 'combined_row_format_with_interaction-{}.csv'.format(severity))

    print('Done')
    return df_row


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
    figh = figw * asp * 1.2  # * 2.5

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
        group.plot(kind='scatter', x='ind', y='minuslog10pvalue', color=colors[num % len(colors)], ax=ax,
                   s=group.HR * 30)
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


def build_heat_map_from_selected_rows(database='INSIGHT',
                                      p_val_threshold=0.05 / 89,
                                      selected_cols=False,
                                      interactionge1=False,
                                      severity='all'):
    if database == 'OneFlorida':
        dir_path = 'output/factors/OneFlorida/elix/'
    else:
        dir_path = 'output/factors/INSIGHT/elix/'

    df_row = pd.read_csv(dir_path + 'combined_row_format_with_interaction-{}.csv'.format(severity))
    print('df_row.shape:', df_row.shape)
    if not interactionge1:
        df = df_row.loc[(df_row['p-Value'] < p_val_threshold) & (df_row['HR'] > 1), :].copy()
    else:
        df = df_row.loc[(df_row['p-Value'] < p_val_threshold) & (df_row['HR'] > 1) & (df_row['HR_inter'] > 1), :].copy()
    df['count'] = df['covariate'].apply(lambda x: (df['covariate'] == x).sum())
    df = df.sort_values(by=['count'], ascending=False)

    print('df.shape:', df.shape)

    df = df.drop(df[df['covariate'] == 'outpatient visits 0'].index, axis=0)

    print('after drop df.shape:', df.shape)

    covs = list(df['covariate'].unique())
    pascs = list(df['pasc'].unique())

    cov_name, pasc_name_list = collect_covariate_name()

    pasc_new = []
    for key in pasc_name_list:
        if key in pascs:
            pasc_new.append(key)
    assert len(pasc_new) == len(pascs)
    pascs = pasc_new

    if selected_cols:
        covs = [
            'hospitalized',
            'icu',
            '20-<40 years',
            '65-<75 years',
            '75+ years',
            'Female',
            '03/20-06/20',
            '07/21-11/21',
            'num_Comorbidity>=5',
            'DX: Arrythmia',
            'DX: Cancer',
            'DX: Chronic Kidney Disease',
            'DX: Cirrhosis',
            'DX: Coagulopathy',
            'DX: Dementia',
            'DX: End Stage Renal Disease on Dialysis',
            'DX: Mental Health Disorders',
            'DX: Pregnant',
            'DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)',
            'DX: Weight Loss',
            'BMI: <18.5 under weight',
            'BMI: >=30 obese ',
        ]
    else:
        covs_new = []
        for key, value in cov_name.items():
            if key in covs:
                covs_new.append(key)
        assert len(covs_new) == len(covs)
        covs = covs_new

    n_cov = len(covs)
    n_pasc = len(pascs)
    print('n_cov:', n_cov, 'n_pasc:', n_pasc)

    cov_id = {c: i for i, c in zip(range(n_cov), covs)}
    pasc_id = {c: i for i, c in zip(range(n_pasc), pascs)}
    data = np.empty((n_pasc, n_cov,))
    data[:] = np.nan
    for key, row in df.iterrows():
        cov = row['covariate']
        pasc = row['pasc']
        hr = row['HR']
        hr_l = row['CI-95% lower-bound']
        hr_u = row['CI-95% upper-bound']
        pval = row['p-Value']
        i = pasc_id[pasc]
        if cov not in cov_id:
            print(cov, 'not in cov_id')
            continue
        j = cov_id[cov]
        data[i, j] = hr

    df_data = pd.DataFrame(data, index=pascs,
                           columns=covs)  # [re.sub("\(.*?\)", "", x.replace('DX: ', '')) for x in covs])

    df_data = df_data.rename(columns=cov_name)
    # {'icu': 'ICU', '75+ years': '≥ 75 years', 'hospitalized': 'Hospitalized',
    #                                   'End Stage Renal Disease on Dialysis': 'End Stage Renal Disease',
    #                                   'Pulmonary Circulation Disorder  ': 'Pulmonary Circulation Disorder',
    #                                   'num_Comorbidity>=5': '>= 5 Comorbidities',
    #                                   'BMI: <18.5 under weight': 'Under weight (BMI < 18.5)',
    #                                   'BMI: >=30 obese ': 'Obese (BMI ≥ 30)',
    #                                   '20-<40 years': '20-39 years', '65-<75 years': '65-74 years'})
    fig = plt.figure(figsize=(14, 15))  # Set the figure size
    # cmap = sns.diverging_palette(240, 10, as_cmap=True)
    fig.add_subplot(111)
    ax = sns.heatmap(df_data, cbar=1, linewidths=2, vmax=5, vmin=1,
                     square=True, annot=True, fmt=".1f", cmap='Blues',
                     linecolor='#D3D3D3', cbar_kws={"shrink": .9})
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False, labeltop=True)

    plt.setp(ax.get_xticklabels(), rotation=-35, ha="right",
             rotation_mode="anchor")
    if n_cov < 35:
        ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize=14)
        ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize=14)
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="10%", pad=0.1)
    # plt.colorbar(ax.get_children()[0], shrink=0.5, cax=cax)
    plt.tight_layout()
    plt.savefig(dir_path + 'figure/risk_heat_map_p{:.6f}-{}-{}.png'.format(p_val_threshold,
                                                                           '-interGe1' if interactionge1 else '',
                                                                           severity),
                bbox_inches='tight', dpi=600)
    plt.savefig(dir_path + 'figure/risk_heat_map_p{:.6f}-{}-{}.pdf'.format(p_val_threshold,
                                                                           '-interGe1' if interactionge1 else '',
                                                                           severity),
                bbox_inches='tight', transparent=True)
    plt.show()

    return df_row, df, df_data


if __name__ == '__main__':
    start_time = time.time()
    # manhattan_plt(database='INSIGHT', )
    # df_row = combine_risk_p_value_with_interaction(database='INSIGHT', severity='all')
    # df_row = combine_risk_p_value_with_interaction(database='INSIGHT', severity='inpatienticu')
    # df_row = combine_risk_p_value_with_interaction(database='INSIGHT', severity='outpatient')

    # df_hr, df_p, df_row = combine_risk_p_value(database='INSIGHT')
    # plot_heatmap_for_risk_grouped_by_organ(database='INSIGHT', star=False, pvalue=0.05)
    # df_row, df, df_data = build_heat_map_from_selected_rows(database='INSIGHT', p_val_threshold=0.01)
    severity = 'inpatienticu'
    df_row, df, df_data = build_heat_map_from_selected_rows(database='INSIGHT', p_val_threshold=0.05 / 89,
                                                            selected_cols=False, interactionge1=False,
                                                            severity=severity)
    df_row, df, df_data = build_heat_map_from_selected_rows(database='INSIGHT', p_val_threshold=0.05 / 89,
                                                            selected_cols=False, interactionge1=True, severity=severity)
    df_row, df, df_data = build_heat_map_from_selected_rows(database='INSIGHT', p_val_threshold=0.01,
                                                            selected_cols=False, interactionge1=False,
                                                            severity=severity)
    df_row, df, df_data = build_heat_map_from_selected_rows(database='INSIGHT', p_val_threshold=0.01,
                                                            selected_cols=False, interactionge1=True, severity=severity)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
