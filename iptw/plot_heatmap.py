import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib as mpl

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
    for severity in ['outpatient', 'inpatient', 'icu',
                     'less65', '65to75', '75above',
                     'female', 'male', 'white', 'black',
                     'Anemia', 'Arrythmia', 'CKD', 'CPD-COPD', 'CAD',
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
                                                  'inpatient': 'Inpatient',
                                                  'icu': 'ICU',
                                                  'less65': '<65', '65to75': '65-<75', '75above': '>=75',
                                                  'female': 'Female', 'male': 'Male', 'white': 'White',
                                                  'black': 'Black',
                                                  'Anemia': 'Anemia', 'Arrythmia': 'Arrythmia', 'CKD': 'CKD',
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


def plot_heatmap_for_dx_subgroup_absCumIncidence_split(database='V15_COVID19', type='cif', month=6, pasc=False):
    month_id = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4}
    monthid = month_id.get(month, -1)
    print('month:', month, 'id:', monthid)

    if database == 'oneflorida':
        df_aux = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-oneflorida-4plot.xlsx',
            sheet_name='diagnosis')
        pvalue = 0.01
        df = pd.read_csv(r'../data/oneflorida/output/character/outcome/select/DX-all-select/causal_effects_specific.csv',)
    elif database == 'V15_COVID19':
        df_aux = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
            sheet_name='diagnosis')
        pvalue = 0.01  # 0.05 / 137
        df = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/select/DX-all-select/causal_effects_specific.csv',)

    else:
        raise ValueError
    df_aux.rename(columns=lambda x: x + '_aux', inplace=True)
    df = pd.merge(df, df_aux,  left_on='pasc', right_on='PASC_aux', how='left').set_index('i') # left_on='State_abr', right_on='address_state',
    print(df.columns)
    # df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    # df_select = df_select.loc[df_select['no. pasc in covid +'] >= 100, :]

    df_select = df.sort_values(by='hr-w', ascending=False)
    df_select = df_select.loc[df_select['hr-w-p'] <= pvalue, :]  #
    df_select = df_select.loc[df_select['hr-w'] > 1, :]
    df_select = df_select.loc[df_select['no. pasc in +'] >= 100, :]

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
            if type == 'hr':
                hr = row['hr-w']
                ci = stringlist_2_list(row['hr-w-CI'])
                p = row['hr-w-p']
            elif type == 'km':
                hr = stringlist_2_list(row['km-w-diff'])[monthid] * (-1000)
                ci = [np.nan, np.nan]  #
                p = stringlist_2_list(row['km-w-diff-p'])[monthid]
            elif type =='cif':
                hr = stringlist_2_list(row['cif_1_w'])[monthid] * 1000
                ci = [stringlist_2_list(row['cif_1_w_CILower'])[monthid], stringlist_2_list(row['cif_1_w_CIUpper'])[monthid]]
                p = np.nan
                # has Confidence interval, but not test, no p-value
            elif type == 'cifdiff':
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
    heat_value = {'all': measure, }
    for severity in ['outpatient', 'inpatient', 'icu',
                     'less65', '65to75', '75above',
                     'female', 'male', 'white', 'black',
                     'CPD-COPD', 'Anemia', 'Arrythmia', 'CAD', 'Hypertension',
                     'T2D-Obesity', 'CKD', 'Mental-substance', 'Corticosteroids',
                     'healthy'
                     ]:
        _df = pd.read_csv(
            r'../data/{}/output/character/outcome/select/DX-{}-select/causal_effects_specific.csv'.format(
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
    output_dir = r'../data/{}/output/character/outcome/figure/{}/'.format(database, type)
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-{}.png'.format(database,
                                                                    type,
                                                                    month), bbox_inches='tight', dpi=700)
    plt.savefig(output_dir + 'subgroup_heatmap_{}-{}-{}.pdf'.format(database,
                                                                    type,
                                                                    month), bbox_inches='tight', transparent=True)

    plt.show()
    print()


if __name__ == '__main__':
    # plot_forest_for_dx()
    # plot_forest_for_med()
    # combine_pasc_list()
    # pasc_domain = add_pasc_domain_to_causal_results()
    # df_med = add_drug_name()
    # plot_heatmap_for_dx_subgroup()

    # types: hr   km   cif   cifdiff
    plot_heatmap_for_dx_subgroup_absCumIncidence_split(database='V15_COVID19', type='cif', month=6, pasc=True)

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
