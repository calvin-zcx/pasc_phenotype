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


def plot_heatmap_for_dx_subgroup():
    df = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
        sheet_name='diagnosis').set_index('i')

    df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    pvalue = 0.01  # 0.05 / 137
    df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    # df_select = df_select.loc[df_select['no. pasc in +'] >= 100, :]
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Diseases of the Nervous System',
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
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    key_list = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple']
            hr = row['Hazard Ratio, Adjusted']
            ci = stringlist_2_list(row['Hazard Ratio, Adjusted, Confidence Interval'])
            p = row['Hazard Ratio, Adjusted, P-Value']
            domain = row['Organ Domain']
            if name == 'General PASC':
                pasc_row = [name, hr, ci, p, domain, key]
                continue
            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])
                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)
                key_list.append(key)

    # add pasc at last
    # organ_n[-1] += 1
    # labs.append(pasc_row[0])
    # measure.append(pasc_row[1])
    # lower.append(pasc_row[2][0])
    # upper.append(pasc_row[2][1])
    # pval.append(pasc_row[3])
    # key_list.append(pasc_row[5])

    # load other
    heat_value = {'all': measure, }
    for severity in ['outpatient', 'inpatienticu',  # 'inpatient', 'icu',
                     'less65', '65to75', '75above',
                     'female', 'male', 'white', 'black',
                     # 'Anemia',
                     'Arrythmia',
                     'CKD',
                     'CPD-COPD',
                     'CAD',
                     'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids',
                     'healthy'
                     ]:
        _df = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-{}/causal_effects_specific.csv'.format(
                severity)).set_index('i')
        _df_selected = _df.loc[key_list, :]
        _hr = _df_selected.loc[:, 'hr-w']
        heat_value[severity] = _hr.tolist()
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    # f, (ax, cbar_ax) = plt.subplots(2, figsize=(15, 15), gridspec_kw=grid_kws)
    f = plt.subplots(figsize=(15, 15))

    df_heat_value = pd.DataFrame(heat_value)
    df_heat_value = df_heat_value.rename(columns={'all': 'Overall',
                                                  'outpatient': 'Outpatient',
                                                  # 'inpatient': 'Inpatient',
                                                  # 'icu': 'ICU',
                                                  'inpatienticu': 'Inpatient',
                                                  'less65': '<65', '65to75': '65-<75', '75above': '>=75',
                                                  'female': 'Female', 'male': 'Male', 'white': 'White',
                                                  'black': 'Black',
                                                  # 'Anemia': 'Anemia',
                                                  'Arrythmia': 'Arrythmia', 'CKD': 'CKD',
                                                  'CPD-COPD': 'CPD',
                                                  'CAD': 'CAD',
                                                  'T2D-Obesity': 'T2D', 'Hypertension': 'Hypertension',
                                                  'Mental-substance': 'Mental',
                                                  'Corticosteroids': 'Steroids history', 'healthy': 'Healthy'})

    ax = sns.heatmap(df_heat_value,  # np.array(measure).reshape(-1,1),
                     # ax = ax,
                     # cbar_ax = cbar_ax,
                     annot=True,
                     fmt=".1f",
                     # norm=Normalize(vmin=0.7, vmax=5, clip=True),
                     yticklabels=labs,
                     # center=1,
                     # square=True,
                     linewidths=0.3,
                     linecolor='white',
                     cmap=sns.cm.rocket_r,
                     cbar_kws={"shrink": .7},
                     )
    # ax.axvline(1, color='white', lw=10)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    left = 0.25
    right = 0.87
    bottom = 0.1
    top = 0.9
    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    # plt.subplots_adjust(left=0.25)
    plt.show()
    plt.tight_layout()
    print()


def plot_heatmap_for_dx_subgroup_split(database='V15_COVID19', type='hr', month=6, pasc=False):
    month_id = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    monthid = month_id.get(month, -1)
    print('month:', month, 'id:', monthid)

    if database == 'oneflorida':
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-oneflorida-4plot.xlsx',
            sheet_name='diagnosis').set_index('i')
        pvalue = 0.01
    elif database == 'V15_COVID19':
        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
            sheet_name='diagnosis').set_index('i')
        pvalue = 0.01  # 0.05 / 137
    else:
        raise ValueError

    df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    df_select = df_select.loc[df_select['no. pasc in covid +'] >= 100, :]
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Diseases of the Nervous System',
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
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    key_list = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple']
            if type == 'hr':
                hr = row['Hazard Ratio, Adjusted']
                ci = stringlist_2_list(row['Hazard Ratio, Adjusted, Confidence Interval'])
                p = row['Hazard Ratio, Adjusted, P-Value']
            elif type == 'km':
                hr = stringlist_2_list(row['km-w-diff'])[monthid] * (-1000)
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]

            domain = row['Organ Domain']
            if name == 'General PASC':
                pasc_row = [name, hr, ci, p, domain, key]
                continue
            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                name = name.strip('*')
                if (p > 0.05 / 137) and (p <= 0.01):
                    name += '*'

                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)
                key_list.append(key)

    if pasc:
        # add pasc at last
        organ_n[-1] += 1
        labs.append(pasc_row[0])
        measure.append(pasc_row[1])
        lower.append(pasc_row[2][0])
        upper.append(pasc_row[2][1])
        pval.append(pasc_row[3])
        key_list.append(pasc_row[5])

    # load other
    heat_value = {'all': measure, }
    for severity in ['outpatient', 'inpatient', 'icu',
                     'less65', '65to75', '75above',
                     'female', 'male', 'white', 'black',
                     'CPD-COPD', 'Anemia', 'Arrythmia', 'CAD', 'Hypertension',
                     'T2D-Obesity', 'CKD', 'Mental-substance', 'Corticosteroids',
                     'healthy'
                     ]:
        _df = pd.read_csv(
            r'../data/{}/output/character/outcome/DX-{}/causal_effects_specific.csv'.format(database,
                                                                                            severity)).set_index('i')
        _df_selected = _df.loc[key_list, :]
        if type == 'hr':
            _hr = _df_selected.loc[:, 'hr-w']
        elif type == 'km':
            _hr = _df_selected.loc[:, 'km-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (-1000))

        heat_value[severity] = _hr.tolist()
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    # f, (ax, cbar_ax) = plt.subplots(2, figsize=(15, 15), gridspec_kw=grid_kws)

    df_heat_value = pd.DataFrame(heat_value)
    df_heat_value = df_heat_value.rename(columns={'all': 'Overall',
                                                  'outpatient': 'Outpatient',
                                                  'inpatient': 'Inpatient',
                                                  'icu': 'ICU',
                                                  'less65': '<65', '65to75': '65-<75', '75above': '75+',
                                                  'female': 'Female', 'male': 'Male', 'white': 'White',
                                                  'black': 'Black',
                                                  'Anemia': 'Anemia', 'Arrythmia': 'Arrythmia', 'CKD': 'CKD',
                                                  'CPD-COPD': 'CPD',
                                                  'CAD': 'CAD',
                                                  'T2D-Obesity': 'T2D', 'Hypertension': 'Hypertension',
                                                  'Mental-substance': 'Mental',
                                                  'Corticosteroids': 'Steroids history', 'healthy': 'Healthy'})

    data = df_heat_value
    asp = data.shape[0] / float(data.shape[1]) / 1.6
    figw = 14  # 8
    figh = figw * asp

    cmap = sns.cm.icefire_r
    # cmap = sns.cm.rocket_r

    norm = Normalize(vmin=data.min().min(), vmax=data.max().max())
    # norm = Normalize(vmin=1, vmax=5, clip=True)

    gridspec_kw = {"width_ratios": [1, 3, 3, 2, 2, 10]}
    heatmapkws = dict(square=False, cbar=False, cmap=cmap, linewidths=0.3, vmin=data.min().min(), vmax=data.max().max())
    tickskw = dict(xticklabels=False, yticklabels=False)

    left = 0.22
    # right = 0.87
    bottom = 0.1
    top = 0.85
    fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(figw, figh), gridspec_kw=gridspec_kw)
    plt.subplots_adjust(left=left, bottom=bottom, top=top, wspace=0.08, hspace=0.08 * asp)
    sns.heatmap(data.iloc[:, 0:1], ax=axes[0], yticklabels=labs, annot=True, fmt=".1f", **heatmapkws)
    sns.heatmap(data.iloc[:, 1:4], ax=axes[1], yticklabels=False, annot=True, fmt=".1f", **heatmapkws)
    sns.heatmap(data.iloc[:, 4:7], ax=axes[2], yticklabels=False, annot=True, fmt=".1f", **heatmapkws)
    sns.heatmap(data.iloc[:, 7:9], ax=axes[3], yticklabels=False, annot=True, fmt=".1f", **heatmapkws)
    sns.heatmap(data.iloc[:, 9:11], ax=axes[4], yticklabels=False, annot=True, fmt=".1f", **heatmapkws)
    sns.heatmap(data.iloc[:, 11:], ax=axes[5], yticklabels=False, annot=True, fmt=".1f", **heatmapkws)

    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
                 rotation_mode="anchor")

    # axes[1, 0].set_yticklabels([9])
    # axes[1, 1].set_xticklabels([4, 5, 6, 7, 8])
    # axes[1, 2].set_xticklabels([9, 10, 11])

    cax = fig.add_axes([0.92, 0.12, 0.025, 0.7])  # [left, bottom, width, height]
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    output_dir = r'../data/{}/output/character/outcome/figure/'.format(database)
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-{}.png'.format(database,
                                                                    type,
                                                                    month), bbox_inches='tight', dpi=700)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-{}.pdf'.format(database,
                                                                    type,
                                                                    month), bbox_inches='tight', transparent=True)

    plt.show()
    print()


def plot_heatmap_for_dx_subgroup_absCumIncidence_split_V2(database='V15_COVID19', type='cif', month=6, pasc=True,
                                                          star=False, select_criteria='', pvalue=0.05 / 137):
    month_id = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    monthid = month_id.get(month, -1)
    print('month:', month, 'id:', monthid)

    if database == 'oneflorida':
        # df_aux = pd.read_excel(
        #     r'../data/oneflorida/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-oneflorida-4plot.xlsx',
        #     sheet_name='diagnosis')
        # pvalue = 0.01
        # df = pd.read_csv(r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific.csv',)
        # df_aux.rename(columns=lambda x: x + '_aux', inplace=True)
        # df = pd.merge(df, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index(
        #     'i')  # left_on='State_abr', right_on='address_state',
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
    elif database == 'V15_COVID19':
        # df_aux = pd.read_excel(
        #     r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
        #     sheet_name='diagnosis')
        # pvalue = 0.01  # 0.05 / 137
        # df = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific.csv',)
        # df_aux.rename(columns=lambda x: x + '_aux', inplace=True)
        # df = pd.merge(df, df_aux, left_on='pasc', right_on='PASC_aux', how='left').set_index(
        #     'i')  # left_on='State_abr', right_on='address_state',
        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
    else:
        raise ValueError
    print(df.columns)
    # df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    # df_select = df_select.loc[df_select['no. pasc in covid +'] >= 100, :]
    if select_criteria == 'insight':
        print('select_critera:', select_criteria)
        _df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
        pvalue = 0.01  # 0.05 / 137
        _df_select = _df.sort_values(by='hr-w', ascending=False)
        _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
        _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
        _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]

        print('_df_select.shape:', _df_select.shape, _df_select['pasc'])

        df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
        df_select = df_select.sort_values(by='hr-w', ascending=False)

    else:
        print('select_critera:', select_criteria, 'default use')
        df_select = df.sort_values(by='hr-w', ascending=False)
        # df_select = df_select.loc[df_select['hr-w-p'] <= pvalue, :]  #
        # df_select = df_select.loc[df_select['hr-w'] > 1, :]
        # df_select = df_select.loc[df_select['no. pasc in +'] >= 100, :]
        df_select = df_select.loc[df_select['selected'] == 1, :]

    df_select['rank'] = df_select['cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid])
    df_select = df_select.sort_values(by='rank', ascending=False)

    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Diseases of the Nervous System',
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
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    key_list = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple']
            pasc = row['pasc']
            hrp = row['hr-w-p']
            name = name.strip('*')

            if star:
                if (hrp > 0.05 / 137) and (hrp <= 0.01):
                    name += '**'
                elif (hrp > 0.01) and (hrp <= 0.05):
                    name += '*'

            if type == 'hr':
                hr = row['hr-w']
                ci = stringlist_2_list(row['hr-w-CI'])
                p = row['hr-w-p']
            elif type == 'km':
                hr = stringlist_2_list(row['km-w-diff'])[monthid] * (-1000)
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
            elif type == 'cif':
                hr = stringlist_2_list(row['cif_1_w'])[monthid] * 1000
                ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid],
                      stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
                p = np.nan
                # has Confidence interval, but not test, no p-value
            elif type == 'cifneg':
                hr = stringlist_2_list(row['cif_0_w'])[monthid] * 1000
                ci = [stringlist_2_list(row['cif_0_w_CILower'])[monthid],
                      stringlist_2_list(row['cif_0_w_CIUpper'])[monthid]]
                p = np.nan
                # has Confidence interval, but not test, no p-value
            elif type == 'cifdiff' or type == 'cifdiff-pvalue':
                hr = stringlist_2_list(row['cif-w-diff'])[monthid] * 1000
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
                # don't have CI, but has p-value for difference.

            domain = row['Organ Domain']
            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain, key]
                continue
            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                # name = name.strip('*')
                # if (p > 0.05 / 137) and (p <= 0.01):
                #     name += '*'

                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)
                key_list.append(key)

    if pasc:
        # add pasc at last
        organ_n[-1] += 1
        labs.append(pasc_row[0])
        measure.append(pasc_row[1])
        lower.append(pasc_row[2][0])
        upper.append(pasc_row[2][1])
        pval.append(pasc_row[3])
        key_list.append(pasc_row[5])

    # load other
    if type == 'cifdiff-pvalue':
        heat_value = {'all': pval, }
    else:
        heat_value = {'all': measure, }

    for severity in ['outpatient',
                     # 'inpatient', 'icu',
                     'inpatienticu',
                     'less65',
                     # '65to75', '75above',
                     'above65',
                     'female', 'male', 'white', 'black',
                     'Arrythmia',
                     'CAD',
                     'CKD',
                     'CPD-COPD',
                     # 'Anemia',
                     'Hypertension',
                     'Mental-substance',
                     'T2D-Obesity',
                     # 'Corticosteroids',
                     'healthy'
                     ]:
        _df = pd.read_csv(
            r'../data/{}/output/character/outcome/DX-{}/causal_effects_specific.csv'.format(
                database, severity)).set_index('i')
        _df_selected = _df.loc[key_list, :]
        if type == 'hr':
            _hr = _df_selected.loc[:, 'hr-w']
        elif type == 'km':
            _hr = _df_selected.loc[:, 'km-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (-1000))
        elif type == 'cif':
            _hr = _df_selected.loc[:, 'cif_1_w'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
            # ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid],
            #       stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
            # p = np.nan
            # has Confidence interval, but not test, no p-value
        elif type == 'cifneg':
            _hr = _df_selected.loc[:, 'cif_0_w'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
        elif type == 'cifdiff':
            _hr = _df_selected.loc[:, 'cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
            # ci = [np.nan, np.nan]  #
            # p = stringlist_2_list(row['km-w-diff-p'])[monthid]
        elif type == 'cifdiff-pvalue':
            _hr = _df_selected.loc[:, 'km-w-diff-p'].apply(lambda x: stringlist_2_list(x)[monthid])

        heat_value[severity] = _hr.tolist()

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    # f, (ax, cbar_ax) = plt.subplots(2, figsize=(15, 15), gridspec_kw=grid_kws)

    df_heat_value = pd.DataFrame(heat_value)
    df_heat_value = df_heat_value.rename(columns={'all': 'Overall',
                                                  'outpatient': 'Outpatient',
                                                  # 'inpatient': 'Inpatient',
                                                  # 'icu': 'ICU',
                                                  'inpatienticu': 'Inpatient',
                                                  'less65': '<65',  # '65to75': '65-<75', '75above': '75+',
                                                  'above65': '>= 65',
                                                  'female': 'Female', 'male': 'Male', 'white': 'White',
                                                  'black': 'Black',
                                                  # 'Anemia': 'Anemia',
                                                  'Arrythmia': 'Arrythmia',
                                                  'CKD': 'CKD',
                                                  'CPD-COPD': 'CPD',
                                                  'CAD': 'CAD',
                                                  'T2D-Obesity': 'T2D', 'Hypertension': 'Hypertension',
                                                  'Mental-substance': 'Mental',
                                                  # 'Corticosteroids': 'Steroids history',
                                                  'healthy': 'Healthy'})

    data = df_heat_value
    asp = data.shape[0] / float(data.shape[1]) / 1.6
    figw = 12  # 8
    figh = figw * asp

    # cmap = sns.cm.icefire_r
    cmap = sns.cm.rocket_r
    if type == 'cifdiff-pvalue':
        # norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        norm = LogNorm(vmin=0.05 / 137, vmax=1)
        fmt = ".1g"
        figw = 15  # 8
        figh = figw * asp
    elif type == 'cifdiff':
        norm = Normalize(vmin=0, vmax=100, clip=True)
        fmt = ".1f"
    elif (type == 'cif') or (type == 'cifneg'):
        # 256.8 for Insight
        # ? for florida?
        norm = Normalize(vmin=data.min().min(), vmax=256.8)  # data.max().max()
        fmt = ".1f"
    else:
        norm = Normalize(vmin=data.min().min(), vmax=data.max().max())
        fmt = ".1f"
    # norm = Normalize(vmin=1, vmax=5, clip=True)

    # gridspec_kw = {"width_ratios": [1, 3, 3, 2, 2, 10]}
    gridspec_kw = {"width_ratios": [1, 2, 2, 2, 2, 8]}
    heatmapkws = dict(square=False, cbar=False, cmap=cmap, linewidths=0.3,
                      vmin=data.min().min(), vmax=data.max().max(), fmt=fmt, norm=norm)
    tickskw = dict(xticklabels=False, yticklabels=False)

    left = 0.22
    # right = 0.87
    bottom = 0.1
    top = 0.85
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


def plot_heatmap_for_dx_subgroup_absCumIncidence_split(database='V15_COVID19', type='cif', month=6, pasc=True,
                                                       star=False, select_criteria='', pvalue=0.05 / 137):
    month_id = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    monthid = month_id.get(month, -1)
    print('month:', month, 'id:', monthid)

    if database == 'oneflorida':
        # df_aux = pd.read_excel(
        #     r'../data/oneflorida/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-oneflorida-4plot.xlsx',
        #     sheet_name='diagnosis')
        # pvalue = 0.01
        # df = pd.read_csv(r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific.csv',)
        # df_aux.rename(columns=lambda x: x + '_aux', inplace=True)
        # df = pd.merge(df, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index(
        #     'i')  # left_on='State_abr', right_on='address_state',
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
    elif database == 'V15_COVID19':
        # df_aux = pd.read_excel(
        #     r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
        #     sheet_name='diagnosis')
        # pvalue = 0.01  # 0.05 / 137
        # df = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific.csv',)
        # df_aux.rename(columns=lambda x: x + '_aux', inplace=True)
        # df = pd.merge(df, df_aux, left_on='pasc', right_on='PASC_aux', how='left').set_index(
        #     'i')  # left_on='State_abr', right_on='address_state',
        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
    else:
        raise ValueError
    print(df.columns)
    # df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    # df_select = df_select.loc[df_select['no. pasc in covid +'] >= 100, :]
    if select_criteria == 'insight':
        print('select_critera:', select_criteria)
        _df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
        pvalue = 0.01  # 0.05 / 137
        _df_select = _df.sort_values(by='hr-w', ascending=False)
        _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
        _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
        _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]

        print('_df_select.shape:', _df_select.shape, _df_select['pasc'])

        df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
        df_select = df_select.sort_values(by='hr-w', ascending=False)

    else:
        print('select_critera:', select_criteria, 'default use')
        df_select = df.sort_values(by='hr-w', ascending=False)
        df_select = df_select.loc[df_select['hr-w-p'] <= pvalue, :]  #
        df_select = df_select.loc[df_select['hr-w'] > 1, :]
        df_select = df_select.loc[df_select['no. pasc in +'] >= 100, :]

    df_select['rank'] = df_select['cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid])
    df_select = df_select.sort_values(by='rank', ascending=False)

    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Diseases of the Nervous System',
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
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    key_list = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple']
            pasc = row['pasc']
            hrp = row['hr-w-p']
            name = name.strip('*')

            if star:
                if (hrp > 0.05 / 137) and (hrp <= 0.01):
                    name += '**'
                elif (hrp > 0.01) and (hrp <= 0.05):
                    name += '*'

            if type == 'hr':
                hr = row['hr-w']
                ci = stringlist_2_list(row['hr-w-CI'])
                p = row['hr-w-p']
            elif type == 'km':
                hr = stringlist_2_list(row['km-w-diff'])[monthid] * (-1000)
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
            elif type == 'cif':
                hr = stringlist_2_list(row['cif_1_w'])[monthid] * 1000
                ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid],
                      stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
                p = np.nan
                # has Confidence interval, but not test, no p-value
            elif type == 'cifneg':
                hr = stringlist_2_list(row['cif_0_w'])[monthid] * 1000
                ci = [stringlist_2_list(row['cif_0_w_CILower'])[monthid],
                      stringlist_2_list(row['cif_0_w_CIUpper'])[monthid]]
                p = np.nan
                # has Confidence interval, but not test, no p-value
            elif type == 'cifdiff' or type == 'cifdiff-pvalue':
                hr = stringlist_2_list(row['cif-w-diff'])[monthid] * 1000
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
                # don't have CI, but has p-value for difference.

            domain = row['Organ Domain']
            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain, key]
                continue
            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                # name = name.strip('*')
                # if (p > 0.05 / 137) and (p <= 0.01):
                #     name += '*'

                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)
                key_list.append(key)

    if pasc:
        # add pasc at last
        organ_n[-1] += 1
        labs.append(pasc_row[0])
        measure.append(pasc_row[1])
        lower.append(pasc_row[2][0])
        upper.append(pasc_row[2][1])
        pval.append(pasc_row[3])
        key_list.append(pasc_row[5])

    # load other
    if type == 'cifdiff-pvalue':
        heat_value = {'all': pval, }
    else:
        heat_value = {'all': measure, }

    for severity in ['outpatient',
                     # 'inpatient', 'icu',
                     'inpatienticu',
                     'less65',
                     # '65to75', '75above',
                     'above65',
                     'female', 'male', 'white', 'black',
                     'Arrythmia',
                     'CAD',
                     'CKD',
                     'CPD-COPD',
                     # 'Anemia',
                     'Hypertension',
                     'Mental-substance',
                     'T2D-Obesity',
                     # 'Corticosteroids',
                     'healthy'
                     ]:
        _df = pd.read_csv(
            r'../data/{}/output/character/outcome/DX-{}/causal_effects_specific.csv'.format(
                database, severity)).set_index('i')
        _df_selected = _df.loc[key_list, :]
        if type == 'hr':
            _hr = _df_selected.loc[:, 'hr-w']
        elif type == 'km':
            _hr = _df_selected.loc[:, 'km-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (-1000))
        elif type == 'cif':
            _hr = _df_selected.loc[:, 'cif_1_w'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
            # ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid],
            #       stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
            # p = np.nan
            # has Confidence interval, but not test, no p-value
        elif type == 'cifneg':
            _hr = _df_selected.loc[:, 'cif_0_w'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
        elif type == 'cifdiff':
            _hr = _df_selected.loc[:, 'cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
            # ci = [np.nan, np.nan]  #
            # p = stringlist_2_list(row['km-w-diff-p'])[monthid]
        elif type == 'cifdiff-pvalue':
            _hr = _df_selected.loc[:, 'km-w-diff-p'].apply(lambda x: stringlist_2_list(x)[monthid])

        heat_value[severity] = _hr.tolist()

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    # f, (ax, cbar_ax) = plt.subplots(2, figsize=(15, 15), gridspec_kw=grid_kws)

    df_heat_value = pd.DataFrame(heat_value)
    df_heat_value = df_heat_value.rename(columns={'all': 'Overall',
                                                  'outpatient': 'Outpatient',
                                                  # 'inpatient': 'Inpatient',
                                                  # 'icu': 'ICU',
                                                  'inpatienticu': 'Inpatient',
                                                  'less65': '<65',  # '65to75': '65-<75', '75above': '75+',
                                                  'above65': '>= 65',
                                                  'female': 'Female', 'male': 'Male', 'white': 'White',
                                                  'black': 'Black',
                                                  # 'Anemia': 'Anemia',
                                                  'Arrythmia': 'Arrythmia',
                                                  'CKD': 'CKD',
                                                  'CPD-COPD': 'CPD',
                                                  'CAD': 'CAD',
                                                  'T2D-Obesity': 'T2D', 'Hypertension': 'Hypertension',
                                                  'Mental-substance': 'Mental',
                                                  # 'Corticosteroids': 'Steroids history',
                                                  'healthy': 'Healthy'})

    data = df_heat_value
    asp = data.shape[0] / float(data.shape[1]) / 1.6
    figw = 12  # 8
    figh = figw * asp

    # cmap = sns.cm.icefire_r
    cmap = sns.cm.rocket_r
    if type == 'cifdiff-pvalue':
        # norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        norm = LogNorm(vmin=0.05 / 137, vmax=1)
        fmt = ".1g"
        figw = 15  # 8
        figh = figw * asp
    elif type == 'cifdiff':
        norm = Normalize(vmin=0, vmax=100, clip=True)
        fmt = ".1f"
    elif (type == 'cif') or (type == 'cifneg'):
        # 256.8 for Insight
        # ? for florida?
        norm = Normalize(vmin=data.min().min(), vmax=256.8)  # data.max().max()
        fmt = ".1f"
    else:
        norm = Normalize(vmin=data.min().min(), vmax=data.max().max())
        fmt = ".1f"
    # norm = Normalize(vmin=1, vmax=5, clip=True)

    # gridspec_kw = {"width_ratios": [1, 3, 3, 2, 2, 10]}
    gridspec_kw = {"width_ratios": [1, 2, 2, 2, 2, 8]}
    heatmapkws = dict(square=False, cbar=False, cmap=cmap, linewidths=0.3,
                      vmin=data.min().min(), vmax=data.max().max(), fmt=fmt, norm=norm)
    tickskw = dict(xticklabels=False, yticklabels=False)

    left = 0.22
    # right = 0.87
    bottom = 0.1
    top = 0.85
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
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-month{}{}-p{:.3f}.png'.format(
        database, type, month, select_criteria, pvalue), bbox_inches='tight', dpi=700)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-month{}{}-p{:.3f}.pdf'.format(
        database, type, month, select_criteria, pvalue), bbox_inches='tight', transparent=True)

    plt.show()
    print()


def plot_heatmap_for_dx_subgroup_absCumIncidence_split_timeperiod(database='V15_COVID19', type='cif', month=6,
                                                                  pasc=True,
                                                                  star=False, select_criteria='', pvalue=0.05 / 137):
    month_id = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    monthid = month_id.get(month, -1)
    print('month:', month, 'id:', monthid)

    if database == 'oneflorida':
        # df_aux = pd.read_excel(
        #     r'../data/oneflorida/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-oneflorida-4plot.xlsx',
        #     sheet_name='diagnosis')
        # pvalue = 0.01
        # df = pd.read_csv(r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific.csv',)
        # df_aux.rename(columns=lambda x: x + '_aux', inplace=True)
        # df = pd.merge(df, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index(
        #     'i')  # left_on='State_abr', right_on='address_state',
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
    elif database == 'V15_COVID19':
        # df_aux = pd.read_excel(
        #     r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
        #     sheet_name='diagnosis')
        # pvalue = 0.01  # 0.05 / 137
        # df = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific.csv',)
        # df_aux.rename(columns=lambda x: x + '_aux', inplace=True)
        # df = pd.merge(df, df_aux, left_on='pasc', right_on='PASC_aux', how='left').set_index(
        #     'i')  # left_on='State_abr', right_on='address_state',
        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
    else:
        raise ValueError
    print(df.columns)
    # df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    # df_select = df_select.loc[df_select['no. pasc in covid +'] >= 100, :]
    if select_criteria == 'insight':
        print('select_critera:', select_criteria)
        _df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
        pvalue = 0.01  # 0.05 / 137
        _df_select = _df.sort_values(by='hr-w', ascending=False)
        # _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
        # _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
        # _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]
        _df_select = _df_select.loc[_df_select['selected'] == 1, :]
        print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        _df_select['rank'] = _df_select['cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid])
        _df_select = _df_select.sort_values(by='rank', ascending=False)
        # df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
        # df_select = df_select.sort_values(by='hr-w', ascending=False)

        df_select = df.loc[_df_select.index, :]

    else:
        print('select_critera:', select_criteria, 'default use')
        df_select = df.sort_values(by='hr-w', ascending=False)
        # df_select = df_select.loc[df_select['hr-w-p'] <= pvalue, :]  #
        # df_select = df_select.loc[df_select['hr-w'] > 1, :]
        # df_select = df_select.loc[df_select['no. pasc in +'] >= 100, :]
        df_select = df_select.loc[df_select['selected'] == 1, :]

        df_select['rank'] = df_select['cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid])
        df_select = df_select.sort_values(by='rank', ascending=False)

    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Diseases of the Nervous System',
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
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    key_list = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple']
            pasc = row['pasc']
            hrp = row['hr-w-p']
            name = name.strip('*')

            if star:
                if (hrp > 0.05 / 137) and (hrp <= 0.01):
                    name += '**'
                elif (hrp > 0.01) and (hrp <= 0.05):
                    name += '*'

            if type == 'hr':
                hr = row['hr-w']
                ci = stringlist_2_list(row['hr-w-CI'])
                p = row['hr-w-p']
            elif type == 'km':
                hr = stringlist_2_list(row['km-w-diff'])[monthid] * (-1000)
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
            elif type == 'cif':
                hr = stringlist_2_list(row['cif_1_w'])[monthid] * 1000
                ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid],
                      stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
                p = np.nan
                # has Confidence interval, but not test, no p-value
            elif type == 'cifneg':
                hr = stringlist_2_list(row['cif_0_w'])[monthid] * 1000
                ci = [stringlist_2_list(row['cif_0_w_CILower'])[monthid],
                      stringlist_2_list(row['cif_0_w_CIUpper'])[monthid]]
                p = np.nan
                # has Confidence interval, but not test, no p-value
            elif type == 'cifdiff' or type == 'cifdiff-pvalue':
                hr = stringlist_2_list(row['cif-w-diff'])[monthid] * 1000
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
                # don't have CI, but has p-value for difference.

            domain = row['Organ Domain']
            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain, key]
                continue
            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                # name = name.strip('*')
                # if (p > 0.05 / 137) and (p <= 0.01):
                #     name += '*'

                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)
                key_list.append(key)

    if pasc:
        # add pasc at last
        organ_n[-1] += 1
        labs.append(pasc_row[0])
        measure.append(pasc_row[1])
        lower.append(pasc_row[2][0])
        upper.append(pasc_row[2][1])
        pval.append(pasc_row[3])
        key_list.append(pasc_row[5])

    # load other
    if type == 'cifdiff-pvalue':
        heat_value = {'all': pval, }
    else:
        heat_value = {'all': measure, }

    for severity in ['03-20-06-20', '07-20-10-20', '11-20-02-21', '03-21-06-21', '07-21-11-21'
                     ]:
        _df = pd.read_csv(
            r'../data/{}/output/character/outcome/DX-{}/causal_effects_specific.csv'.format(
                database, severity)).set_index('i')
        _df_selected = _df.loc[key_list, :]
        if type == 'hr':
            _hr = _df_selected.loc[:, 'hr-w']
        elif type == 'km':
            _hr = _df_selected.loc[:, 'km-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (-1000))
        elif type == 'cif':
            _hr = _df_selected.loc[:, 'cif_1_w'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
            # ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid],
            #       stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
            # p = np.nan
            # has Confidence interval, but not test, no p-value
        elif type == 'cifneg':
            _hr = _df_selected.loc[:, 'cif_0_w'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
        elif type == 'cifdiff':
            _hr = _df_selected.loc[:, 'cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
            # ci = [np.nan, np.nan]  #
            # p = stringlist_2_list(row['km-w-diff-p'])[monthid]
        elif type == 'cifdiff-pvalue':
            _hr = _df_selected.loc[:, 'km-w-diff-p'].apply(lambda x: stringlist_2_list(x)[monthid])

        heat_value[severity] = _hr.tolist()

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    # f, (ax, cbar_ax) = plt.subplots(2, figsize=(15, 15), gridspec_kw=grid_kws)

    df_heat_value = pd.DataFrame(heat_value)
    if database == 'V15_COVID19':
        df_heat_value = df_heat_value.rename(columns={'all': 'Overall (100%)',
                                                      '03-20-06-20': '03/20-06/20 (31.8%)',
                                                      '07-20-10-20': '07/20-10/20 (5.7%)',
                                                      '11-20-02-21': '11/20-02/21 (41.5%)',
                                                      '03-21-06-21': '03/21-06/21 (15.8%)',
                                                      '07-21-11-21': '07/21-11/21 (5.1%)'})
    elif database == 'oneflorida':
        df_heat_value = df_heat_value.rename(columns={'all': 'Overall (100%)',
                                                      '03-20-06-20': '03/20-06/20 (9.1%)',
                                                      '07-20-10-20': '07/20-10/20 (27.0%)',
                                                      '11-20-02-21': '11/20-02/21 (28.0%)',
                                                      '03-21-06-21': '03/21-06/21 (10.4%)',
                                                      '07-21-11-21': '07/21-11/21 (25.5%)'})
    else:
        df_heat_value = df_heat_value.rename(columns={'all': 'Overall',
                                                      '03-20-06-20': '03/20-06/20',
                                                      '07-20-10-20': '07/20-10/20',
                                                      '11-20-02-21': '11/20-02/21',
                                                      '03-21-06-21': '03/21-06/21',
                                                      '07-21-11-21': '07/21-11/21'})

    data = df_heat_value
    asp = data.shape[0] / float(data.shape[1]) / 1.6
    figw = 7  # 8
    figh = figw * asp

    # cmap = sns.cm.icefire_r
    cmap = sns.cm.rocket_r
    if type == 'cifdiff-pvalue':
        # norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        norm = LogNorm(vmin=0.05 / 137, vmax=1)
        fmt = ".1g"
        figw = 15  # 8
        figh = figw * asp
    elif type == 'cifdiff':
        norm = Normalize(vmin=0, vmax=100, clip=True)
        fmt = ".1f"
    elif (type == 'cif') or (type == 'cifneg'):
        # 256.8 for Insight
        # ? for florida?
        norm = Normalize(vmin=data.min().min(), vmax=256.8)  # data.max().max()
        fmt = ".1f"
    else:
        norm = Normalize(vmin=data.min().min(), vmax=data.max().max())
        fmt = ".1f"
    # norm = Normalize(vmin=1, vmax=5, clip=True)

    # gridspec_kw = {"width_ratios": [1, 3, 3, 2, 2, 10]}
    gridspec_kw = {"width_ratios": [1, 5]}
    heatmapkws = dict(square=False, cbar=False, cmap=cmap, linewidths=0.3,
                      vmin=data.min().min(), vmax=data.max().max(), fmt=fmt, norm=norm)
    tickskw = dict(xticklabels=False, yticklabels=False)

    left = 0.22
    # right = 0.87
    bottom = 0.1
    top = 0.85
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(figw, figh), gridspec_kw=gridspec_kw)
    plt.subplots_adjust(left=left, bottom=bottom, top=top, wspace=0.08, hspace=0.08 * asp)
    sns.heatmap(data.iloc[:, 0:1], ax=axes[0], yticklabels=labs, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 1:], ax=axes[1], yticklabels=False, annot=True, **heatmapkws)

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
    output_dir = r'../data/{}/output/character/outcome/figure/timeperiod/{}/'.format(database, type)
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-month{}{}-p{:.3f}-V2.png'.format(
        database, type, month, select_criteria, pvalue), bbox_inches='tight', dpi=700)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-month{}{}-p{:.3f}-V2.pdf'.format(
        database, type, month, select_criteria, pvalue), bbox_inches='tight', transparent=True)

    plt.show()
    print()


def plot_heatmap_for_dx_subgroup_absCumIncidence_split_timeperiod_Variant(database='V15_COVID19', type='cif', month=6,
                                                                          pasc=True,
                                                                          star=False, select_criteria='',
                                                                          pvalue=0.05 / 137):
    month_id = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    monthid = month_id.get(month, -1)
    print('month:', month, 'id:', monthid)

    if database == 'oneflorida':
        # df_aux = pd.read_excel(
        #     r'../data/oneflorida/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-oneflorida-4plot.xlsx',
        #     sheet_name='diagnosis')
        # pvalue = 0.01
        # df = pd.read_csv(r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific.csv',)
        # df_aux.rename(columns=lambda x: x + '_aux', inplace=True)
        # df = pd.merge(df, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index(
        #     'i')  # left_on='State_abr', right_on='address_state',
        # df = pd.read_excel(
        #     r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific_v3.xlsx',
        #     sheet_name='diagnosis').set_index('i')
        df = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific.csv',).set_index('i')
    elif database == 'V15_COVID19':
        # df_aux = pd.read_excel(
        #     r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
        #     sheet_name='diagnosis')
        # pvalue = 0.01  # 0.05 / 137
        # df = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific.csv',)
        # df_aux.rename(columns=lambda x: x + '_aux', inplace=True)
        # df = pd.merge(df, df_aux, left_on='pasc', right_on='PASC_aux', how='left').set_index(
        #     'i')  # left_on='State_abr', right_on='address_state',
        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
    elif database == 'pooled':
        df = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/pooled/DX-all/causal_effects_specific.csv',
        ).set_index('i')
    else:
        raise ValueError
    print(df.columns)
    # df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    # df_select = df_select.loc[df_select['no. pasc in covid +'] >= 100, :]
    if select_criteria == 'insight':
        print('select_critera:', select_criteria)
        _df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis').set_index('i')
        pvalue = 0.05 / 137
        _df_select = _df.sort_values(by='hr-w', ascending=False)
        _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
        # _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
        # _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]
        _df_select = _df_select.loc[_df_select['selected'] == 1, :]
        print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        _df_select['rank'] = _df_select['cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid])
        _df_select = _df_select.sort_values(by='rank', ascending=False)
        # df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
        # df_select = df_select.sort_values(by='hr-w', ascending=False)

        df_select = df.loc[_df_select.index, :]
        df_select = pd.merge(df_select, _df_select[['pasc', 'PASC Name Simple', 'Organ Domain']],
                             left_on='pasc', right_on='pasc', how='left', suffixes=['', '']).set_index('Unnamed: 0')
        df_select.index = df_select.index + 1

    else:
        print('select_critera:', select_criteria, 'default use')
        df_select = df.sort_values(by='hr-w', ascending=False)
        df_select = df_select.loc[df_select['hr-w-p'] <= pvalue, :]  #
        # df_select = df_select.loc[df_select['hr-w'] > 1, :]
        # df_select = df_select.loc[df_select['no. pasc in +'] >= 100, :]
        df_select = df_select.loc[df_select['selected'] == 1, :]

        df_select['rank'] = df_select['cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid])
        df_select = df_select.sort_values(by='rank', ascending=False)

    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Diseases of the Nervous System',
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
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    key_list = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple']
            pasc = row['pasc']
            hrp = row['hr-w-p']
            name = name.strip('*')

            if star:
                if (hrp > 0.05 / 137) and (hrp <= 0.01):
                    name += '**'
                elif (hrp > 0.01) and (hrp <= 0.05):
                    name += '*'

            if type == 'hr':
                hr = row['hr-w']
                ci = stringlist_2_list(row['hr-w-CI'])
                p = row['hr-w-p']
            elif type == 'km':
                hr = stringlist_2_list(row['km-w-diff'])[monthid] * (-1000)
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
            elif type == 'cif':
                hr = stringlist_2_list(row['cif_1_w'])[monthid] * 1000
                ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid],
                      stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
                p = np.nan
                # has Confidence interval, but not test, no p-value
            elif type == 'cifneg':
                hr = stringlist_2_list(row['cif_0_w'])[monthid] * 1000
                ci = [stringlist_2_list(row['cif_0_w_CILower'])[monthid],
                      stringlist_2_list(row['cif_0_w_CIUpper'])[monthid]]
                p = np.nan
                # has Confidence interval, but not test, no p-value
            elif type == 'cifdiff' or type == 'cifdiff-pvalue':
                hr = stringlist_2_list(row['cif-w-diff'])[monthid] * 1000
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
                # don't have CI, but has p-value for difference.

            domain = row['Organ Domain']
            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain, key]
                continue
            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                # name = name.strip('*')
                # if (p > 0.05 / 137) and (p <= 0.01):
                #     name += '*'

                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)
                key_list.append(key)

    if pasc:
        # add pasc at last
        organ_n[-1] += 1
        labs.append(pasc_row[0])
        measure.append(pasc_row[1])
        lower.append(pasc_row[2][0])
        upper.append(pasc_row[2][1])
        pval.append(pasc_row[3])
        key_list.append(pasc_row[5])

    # load other
    if type == 'cifdiff-pvalue':
        heat_value = {'all': pval, }
    else:
        heat_value = {'all': measure, }

    for severity in ['1stwave', 'delta']:
        if database in ['onflorida', 'V15_COVID19']:
            _df = pd.read_csv(
                r'../data/{}/output/character/outcome/DX-{}/causal_effects_specific.csv'.format(
                    database, severity)).set_index('i')
        elif database == 'pooled':
            _df = pd.read_csv(
                r'../data/V15_COVID19/output/character/outcome/pooled/DX-{}/causal_effects_specific.csv'.format(
                    severity)).set_index('i')
        else:
            raise ValueError

        _df_selected = _df.loc[key_list, :]
        if type == 'hr':
            _hr = _df_selected.loc[:, 'hr-w']
        elif type == 'km':
            _hr = _df_selected.loc[:, 'km-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (-1000))
        elif type == 'cif':
            _hr = _df_selected.loc[:, 'cif_1_w'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
            # ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid],
            #       stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
            # p = np.nan
            # has Confidence interval, but not test, no p-value
        elif type == 'cifneg':
            _hr = _df_selected.loc[:, 'cif_0_w'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
        elif type == 'cifdiff':
            _hr = _df_selected.loc[:, 'cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
            # ci = [np.nan, np.nan]  #
            # p = stringlist_2_list(row['km-w-diff-p'])[monthid]
        elif type == 'cifdiff-pvalue':
            _hr = _df_selected.loc[:, 'km-w-diff-p'].apply(lambda x: stringlist_2_list(x)[monthid])

        heat_value[severity] = _hr.tolist()

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    # f, (ax, cbar_ax) = plt.subplots(2, figsize=(15, 15), gridspec_kw=grid_kws)

    df_heat_value = pd.DataFrame(heat_value)
    # if database == 'V15_COVID19':
    #     df_heat_value = df_heat_value.rename(columns={'all': 'Overall (100%)',
    #                                                   '03-20-06-20': '03/20-06/20 (31.8%)',
    #                                                   '07-20-10-20': '07/20-10/20 (5.7%)',
    #                                                   '11-20-02-21': '11/20-02/21 (41.5%)',
    #                                                   '03-21-06-21': '03/21-06/21 (15.8%)',
    #                                                   '07-21-11-21': '07/21-11/21 (5.1%)'})
    # elif database == 'oneflorida':
    #     df_heat_value = df_heat_value.rename(columns={'all': 'Overall (100%)',
    #                                                   '03-20-06-20': '03/20-06/20 (9.1%)',
    #                                                   '07-20-10-20': '07/20-10/20 (27.0%)',
    #                                                   '11-20-02-21': '11/20-02/21 (28.0%)',
    #                                                   '03-21-06-21': '03/21-06/21 (10.4%)',
    #                                                   '07-21-11-21': '07/21-11/21 (25.5%)'})
    # else:
    #     df_heat_value = df_heat_value.rename(columns={'all': 'Overall',
    #                                                   '03-20-06-20': '03/20-06/20',
    #                                                   '07-20-10-20': '07/20-10/20',
    #                                                   '11-20-02-21': '11/20-02/21',
    #                                                   '03-21-06-21': '03/21-06/21',
    #                                                   '07-21-11-21': '07/21-11/21'})

    data = df_heat_value
    asp = data.shape[0] / float(data.shape[1]) / 1.6
    figw = 3 #7  # 8
    figh = 15 # figw * asp

    # cmap = sns.cm.icefire_r
    cmap = sns.cm.rocket_r
    if type == 'cifdiff-pvalue':
        # norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        norm = LogNorm(vmin=0.05 / 137, vmax=1)
        fmt = ".1g"
        figw = 15  # 8
        figh = figw * asp
    elif type == 'cifdiff':
        norm = Normalize(vmin=0, vmax=100, clip=True)
        fmt = ".1f"
    elif (type == 'cif') or (type == 'cifneg'):
        # 256.8 for Insight
        # ? for florida?
        norm = Normalize(vmin=data.min().min(), vmax=256.8)  # data.max().max()
        fmt = ".1f"
    else:
        norm = Normalize(vmin=data.min().min(), vmax=data.max().max())
        fmt = ".1f"
    # norm = Normalize(vmin=1, vmax=5, clip=True)

    # gridspec_kw = {"width_ratios": [1, 3, 3, 2, 2, 10]}
    gridspec_kw = {"width_ratios": [1, 2]}
    heatmapkws = dict(square=True, cbar=False, cmap=cmap, linewidths=0.3,
                      vmin=data.min().min(), vmax=data.max().max(), fmt=fmt, norm=norm)
    tickskw = dict(xticklabels=False, yticklabels=False)

    left = 0.22
    # right = 0.87
    bottom = 0.1
    top = 0.85
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(figw, figh), gridspec_kw=gridspec_kw)
    # plt.subplots_adjust(left=left, bottom=bottom, top=top, wspace=0.08, hspace=0.08 * asp)
    sns.heatmap(data.iloc[:, 0:1], ax=axes[0], yticklabels=labs, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 1:], ax=axes[1], yticklabels=False, annot=True, **heatmapkws)

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

    if database in ['onflorida', 'V15_COVID19']:
        output_dir = r'../data/{}/output/character/outcome/figure/timeperiod/{}/'.format(database, type)
    elif database == 'pooled':
        output_dir = r'../data/V15_COVID19/output/character/outcome/figure/pooled/timeperiod/{}/'.format(type)
    else:
        raise ValueError

    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-variant{}{}-p{:.3f}.png'.format(
        database, type, month, select_criteria, pvalue), bbox_inches='tight', dpi=700)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-variant{}{}-p{:.3f}.pdf'.format(
        database, type, month, select_criteria, pvalue), bbox_inches='tight', transparent=True)

    plt.show()
    print()


def plot_heatmap_for_dx_subgroup_compare2data(database='V15_COVID19', type='cif', month=6, pasc=False, star=False):
    month_id = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    monthid = month_id.get(month, -1)
    print('month:', month, 'id:', monthid)

    # select according to NYC
    _df = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v2.xlsx',
        sheet_name='diagnosis').set_index('i')
    pvalue = 0.01  # 0.05 / 137
    _df_select = _df.sort_values(by='hr-w', ascending=False)
    _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
    _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
    _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]

    print('_df_select.shape:', _df_select.shape, _df_select['pasc'])

    df_aux = _df.copy()
    df = pd.read_csv(
        r'../data/V15_COVID19/output/character/outcome/compare2/DX2Com-all-select/causal_effects_specific.csv', )
    df_aux.rename(columns=lambda x: x + '_aux', inplace=True)
    df = pd.merge(df, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index(
        'i')  # left_on='State_abr', right_on='address_state',
    print(df.columns)

    df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain_aux'].unique()
    print(organ_list)
    organ_list = [
        'Diseases of the Nervous System',
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
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    key_list = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple_aux']

            hrp = row['hr-w-p']
            name = name.strip('*')

            if star:
                if (hrp > 0.05 / 137) and (hrp <= 0.01):
                    name += '*'

            if type == 'hr':
                hr = row['hr-w']
                ci = stringlist_2_list(row['hr-w-CI'])
                p = row['hr-w-p']
            elif type == 'km':
                hr = stringlist_2_list(row['km-w-diff'])[monthid] * (-1000)
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
            elif type == 'cif':
                hr = stringlist_2_list(row['cif_1_w'])[monthid] * 1000
                ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid],
                      stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
                p = np.nan
                # has Confidence interval, but not test, no p-value
            elif type == 'cifdiff' or type == 'cifdiff-pvalue':
                hr = stringlist_2_list(row['cif-w-diff'])[monthid] * 1000
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
                # don't have CI, but has p-value for difference.

            domain = row['Organ Domain_aux']
            if name == 'General PASC':
                pasc_row = [name, hr, ci, p, domain, key]
                continue
            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                # name = name.strip('*')
                # if (p > 0.05 / 137) and (p <= 0.01):
                #     name += '*'

                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)
                key_list.append(key)

    if pasc:
        # add pasc at last
        organ_n[-1] += 1
        labs.append(pasc_row[0])
        measure.append(pasc_row[1])
        lower.append(pasc_row[2][0])
        upper.append(pasc_row[2][1])
        pval.append(pasc_row[3])
        key_list.append(pasc_row[5])

    # load other
    if type == 'cifdiff-pvalue':
        heat_value = {'all': pval, }
    else:
        heat_value = {'all': measure, }

    for severity in ['outpatient', 'inpatient', 'icu',
                     'less65', '65to75', '75above',
                     'female', 'male', 'white', 'black',
                     'CPD-COPD', 'Anemia', 'Arrythmia', 'CAD', 'Hypertension',
                     'T2D-Obesity', 'CKD', 'Mental-substance', 'Corticosteroids',
                     'healthy'
                     ]:
        _df = pd.read_csv(
            r'../data/{}/output/character/outcome/compare2/DX2Com-{}-select/causal_effects_specific.csv'.format(
                database, severity)).set_index('i')
        _df_selected = _df.loc[key_list, :]
        if type == 'hr':
            _hr = _df_selected.loc[:, 'hr-w']
        elif type == 'km':
            _hr = _df_selected.loc[:, 'km-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (-1000))
        elif type == 'cif':
            _hr = _df_selected.loc[:, 'cif_1_w'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
            # ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid],
            #       stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
            # p = np.nan
            # has Confidence interval, but not test, no p-value
        elif type == 'cifdiff':
            _hr = _df_selected.loc[:, 'cif-w-diff'].apply(lambda x: stringlist_2_list(x)[monthid] * (1000))
            # ci = [np.nan, np.nan]  #
            # p = stringlist_2_list(row['km-w-diff-p'])[monthid]
        elif type == 'cifdiff-pvalue':
            _hr = _df_selected.loc[:, 'km-w-diff-p'].apply(lambda x: stringlist_2_list(x)[monthid])

        heat_value[severity] = _hr.tolist()

    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

    # grid_kws = {"height_ratios": (.9, .05), "hspace": .3}
    # f, (ax, cbar_ax) = plt.subplots(2, figsize=(15, 15), gridspec_kw=grid_kws)

    df_heat_value = pd.DataFrame(heat_value)
    df_heat_value = df_heat_value.rename(columns={'all': 'Overall',
                                                  'outpatient': 'Outpatient',
                                                  'inpatient': 'Inpatient',
                                                  'icu': 'ICU',
                                                  'less65': '<65', '65to75': '65-<75', '75above': '75+',
                                                  'female': 'Female', 'male': 'Male', 'white': 'White',
                                                  'black': 'Black',
                                                  'Anemia': 'Anemia', 'Arrythmia': 'Arrythmia', 'CKD': 'CKD',
                                                  'CPD-COPD': 'CPD',
                                                  'CAD': 'CAD',
                                                  'T2D-Obesity': 'T2D', 'Hypertension': 'Hypertension',
                                                  'Mental-substance': 'Mental',
                                                  'Corticosteroids': 'Steroids history', 'healthy': 'Healthy'})

    data = df_heat_value
    asp = data.shape[0] / float(data.shape[1]) / 1.6
    figw = 15  # 8
    figh = figw * asp

    cmap = sns.cm.icefire_r
    # cmap = sns.cm.rocket_r
    if type == 'cifdiff-pvalue':
        norm = LogNorm(vmin=data.min().min(), vmax=data.max().max())
        norm = LogNorm(vmin=0.05 / 137, vmax=1)

        fmt = ".1g"
        figw = 15.5  # 8
        figh = figw * asp
    elif type == 'cifdiff':
        norm = Normalize(vmin=0, vmax=100, clip=True)
        norm = Normalize(vmin=data.min().min(), vmax=data.max().max(), clip=True)
        # norm = Normalize(vmin=-150, vmax=150, clip=True)

        fmt = ".1f"
    else:
        # 256.8 for Insight
        # ? for florida?
        norm = Normalize(vmin=data.min().min(), vmax=256.8)  # data.max().max()
        fmt = ".1f"
    # norm = Normalize(vmin=1, vmax=5, clip=True)

    gridspec_kw = {"width_ratios": [1, 3, 3, 2, 2, 10]}
    heatmapkws = dict(square=False, cbar=False, cmap=cmap, linewidths=0.3,
                      vmin=data.min().min(), vmax=data.max().max(), fmt=fmt, norm=norm)
    tickskw = dict(xticklabels=False, yticklabels=False)

    left = 0.22
    # right = 0.87
    bottom = 0.1
    top = 0.85
    fig, axes = plt.subplots(ncols=6, nrows=1, figsize=(figw, figh), gridspec_kw=gridspec_kw)
    plt.subplots_adjust(left=left, bottom=bottom, top=top, wspace=0.08, hspace=0.08 * asp)
    sns.heatmap(data.iloc[:, 0:1], ax=axes[0], yticklabels=labs, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 1:4], ax=axes[1], yticklabels=False, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 4:7], ax=axes[2], yticklabels=False, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 7:9], ax=axes[3], yticklabels=False, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 9:11], ax=axes[4], yticklabels=False, annot=True, **heatmapkws)
    sns.heatmap(data.iloc[:, 11:], ax=axes[5], yticklabels=False, annot=True, **heatmapkws)

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
    output_dir = r'../data/{}/output/character/outcome/figure2Compare/{}/'.format(database, type)
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-{}.png'.format(database,
                                                                    type,
                                                                    month), bbox_inches='tight', dpi=700)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-{}.pdf'.format(database,
                                                                    type,
                                                                    month), bbox_inches='tight', transparent=True)

    plt.show()
    print()


def add_smd_to_med_sheet(database='insight'):
    if database == 'oneflorida':
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/MED-all/causal_effects_specific_med.xlsx',
            sheet_name='med_selected')
        df_aux = pd.read_csv(r'../data/oneflorida/output/character/outcome/MED-all/causal_effects_specific_med.csv', )

    else:
        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='med')
        df_aux = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/MED/causal_effects_specific_med.csv', )

    df = pd.merge(df, df_aux[['i', 'pasc', 'max smd', 'max smd iptw']], left_on='i', right_on='i', how='left')

    df.to_csv('../debug/med-{}-with-smd.csv'.format(database))
    print('Done')


if __name__ == '__main__':
    start_time = time.time()
    # add_smd_to_med_sheet(database='V15_COVID19')
    # add_smd_to_med_sheet(database='oneflorida')

    # plot_forest_for_dx()
    # plot_forest_for_med()
    # combine_pasc_list()
    # pasc_domain = add_pasc_domain_to_causal_results()
    # df_med = add_drug_name()
    # plot_heatmap_for_dx_subgroup()

    # # types: hr   km   cif   cifdiff
    # dataset = 'oneflorida'  #  'V15_COVID19'   #
    # dataset = 'V15_COVID19'
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='cifdiff', month=6, pasc=True,
    #                                                    star=True, select_criteria='insight')
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='hr', month=6, pasc=True,
    #                                                    star=True, select_criteria='insight')

    # dataset = 'V15_COVID19'
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='cifdiff',
    #                                                    month=6, pasc=True, star=True, pvalue=0.05 / 137)
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='cifdiff',
    #                                                    month=6, pasc=True, star=True, pvalue=0.05)
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='cifdiff',
    #                                                    month=6, pasc=True, star=True, pvalue=0.01)

    # 2022-05-04
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split_V2(database='V15_COVID19', type='cifdiff', month=6, pasc=True,
    #                                                       star=True)
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split_V2(database='oneflorida', type='cifdiff', month=6, pasc=True,
    #                                                       star=True)
    # 2022-05-04
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split_timeperiod(database='V15_COVID19', type='cifdiff', month=6,
    #                                                               pasc=True,
    #                                                               star=True)
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split_timeperiod(database='oneflorida', type='cifdiff', month=6,
    #                                                               pasc=True,
    #                                                               star=True,
    #                                                               select_criteria='insight')

    # 2022-06-13
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split_timeperiod_Variant(database='V15_COVID19', type='cifdiff',
    #                                                                       month=6,
    #                                                                       pasc=True,
    #                                                                       star=True)
    #
    # plot_heatmap_for_dx_subgroup_absCumIncidence_split_timeperiod_Variant(database='oneflorida', type='cifdiff',
    #                                                                       month=6,
    #                                                                       pasc=True,
    #                                                                       star=True,
    #                                                                       select_criteria='insight')
    # 2022-07-12
    plot_heatmap_for_dx_subgroup_absCumIncidence_split_timeperiod_Variant(database='pooled', type='cifdiff',
                                                                          month=6,
                                                                          pasc=True,
                                                                          star=True,
                                                                          select_criteria='insight')

    # for dataset in ['V15_COVID19', 'oneflorida']:
    #     for month in [6, 2, 3, 4, 5]: #
    #         plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='cifdiff', month=month, pasc=True,
    #                                                            star=True)
    #         plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='cifdiff', month=month, pasc=True,
    #                                                            star=True, pvalue=0.01)
    #         plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='cifdiff', month=month, pasc=True,
    #                                                            star=True, pvalue=0.05)
    #         plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='cifdiff-pvalue', month=month,
    #                                                            pasc=True, star=True)
    #         plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='cif', month=month, pasc=True,
    #                                                            star=True)
    #         plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='hr', month=month, pasc=True,
    #                                                            star=True)
    #         plot_heatmap_for_dx_subgroup_absCumIncidence_split(database=dataset, type='cifneg', month=month, pasc=True,
    #                                                            star=True)
    # #

    # plot_heatmap_for_dx_subgroup_compare2data(database='V15_COVID19', type='cifdiff', month=6, pasc=True, star=True)
    # plot_heatmap_for_dx_subgroup_compare2data(database='V15_COVID19', type='cifdiff-pvalue', month=6, pasc=True,
    #                                           star=True)
    # plot_heatmap_for_dx_subgroup_compare2data(database='V15_COVID19', type='hr', month=6, pasc=True, star=True)

    # plot_heatmap_for_dx_subgroup_split(database='V15_COVID19', type='hr', month=6)
    # plot_heatmap_for_dx_subgroup_split(database='oneflorida', type='hr', month=6)
    # plot_heatmap_for_dx_subgroup_split(database='V15_COVID19', type='km', month=6, pasc=True)
    # plot_heatmap_for_dx_subgroup_split(database='oneflorida', type='km', month=6, pasc=True)
    # plot_heatmap_for_dx_subgroup_split(database='V15_COVID19', type='km', month=5, pasc=True)
    # plot_heatmap_for_dx_subgroup_split(database='oneflorida', type='km', month=5, pasc=True)
    # plot_heatmap_for_dx_subgroup_split(database='V15_COVID19', type='km', month=4, pasc=True)
    # plot_heatmap_for_dx_subgroup_split(database='oneflorida', type='km', month=4, pasc=True)
    # plot_heatmap_for_dx_subgroup_split(database='V15_COVID19', type='km', month=3, pasc=True)
    # plot_heatmap_for_dx_subgroup_split(database='oneflorida', type='km', month=3, pasc=True)
    # plot_heatmap_for_dx_subgroup_split(database='V15_COVID19', type='km', month=2, pasc=True)
    # plot_heatmap_for_dx_subgroup_split(database='oneflorida', type='km', month=2, pasc=True)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
