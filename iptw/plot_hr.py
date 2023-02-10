import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import re

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


def plot_forest_for_dx_organ(pvalue=0.05 / 137, star=True):
    df = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
        sheet_name='diagnosis')

    df_select = df.sort_values(by='hr-w', ascending=False)
    # pvalue = 0.01  # 0.05 / 137
    df_select = df_select.loc[df_select['hr-w-p'] <= pvalue, :]  #
    df_select = df_select.loc[df_select['hr-w'] > 1, :]
    df_select = df_select.loc[df_select['no. pasc in +'] >= 100, :]
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
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            pasc = row['pasc']
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']

            if star:
                if (p > 0.05 / 137) and (p <= 0.01):
                    name += '**'
                elif (p > 0.01) and (p <= 0.05):
                    name += '*'

            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain]
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

    # add pasc at last
    organ_n[-1] += 1
    labs.append(pasc_row[0])
    measure.append(pasc_row[1])
    lower.append(pasc_row[2][0])
    upper.append(pasc_row[2][1])
    pval.append(pasc_row[3])

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='black'),   # , linecolor='#fcb2ab')
    ax = p.plot(figsize=(8, .38 * len(labs)), t_adjuster=0.0108, max_value=3.5, min_value=0.9, size=5, decimal=2)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')  # linewidth=1,

    ax.set_yticklabels(labs, fontsize=11.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/organ/dx/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'dx_hr_{}-p{:.3f}-hrGe1-nGe100.png'.format('all', pvalue), bbox_inches='tight', dpi=900)
    plt.savefig(output_dir + 'dx_hr_{}-p{:.3f}-hrGe1-nGe100.pdf'.format('all', pvalue), bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_V2(pvalue=0.05 / 137, star=True, pasc_dx=False, text_right=False):
    df = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
        sheet_name='diagnosis')

    df_select = df.sort_values(by='hr-w', ascending=False)
    # pvalue = 0.01  # 0.05 / 137
    df_select = df_select.loc[df_select['selected'] == 1, :]  #
    # df_select = df_select.loc[df_select['hr-w'] > 1, :]
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
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            pasc = row['pasc']
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']

            if star:
                if (p > 0.05 / 137) and (p <= 0.01):
                    name += '**'
                elif (p > 0.01) and (p <= 0.05):
                    name += '*'

            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain]
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

    # add pasc at last
    if pasc_dx:
        organ_n[-1] += 1
        labs.append(pasc_row[0])
        measure.append(pasc_row[1])
        lower.append(pasc_row[2][0])
        upper.append(pasc_row[2][1])
        pval.append(pasc_row[3])

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='black'),   # , linecolor='#fcb2ab')
    ax = p.plot(figsize=(8, .38 * len(labs)), t_adjuster=0.0108, max_value=3.5, min_value=0.9, size=5, decimal=2,
                text_right=text_right)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')  # linewidth=1,

    ax.set_yticklabels(labs, fontsize=11.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/organ/dx/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'dx_hr_{}-p{:.3f}-hrGe1-nGe100{}-V2{}.png'.format(
        'all', pvalue,
        '-withU099' if pasc_dx else '',
        'text_right' if text_right else ''), bbox_inches='tight', dpi=600)

    plt.savefig(output_dir + 'dx_hr_{}-p{:.3f}-hrGe1-nGe100{}-V2{}.pdf'.format(
        'all', pvalue,
        '-withU099' if pasc_dx else '',
        'text_right' if text_right else ''), bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_V3(star=True, pasc_dx=False, text_right=False):
    df = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3-multitest-withMultiPval-DXMEDALL.xlsx',
        sheet_name='Sheet1')

    df_select = df.sort_values(by='hr-w', ascending=False)
    df_select = df_select.loc[df_select['selected'] == 1, :]  #
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

    nabsv = []
    ncumv = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            pasc = row['pasc']
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']

            nabs = row['no. pasc in +']
            ncum = stringlist_2_list(row['cif_1_w'])[-1] * 1000
            ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]

            if star:
                if p <= 0.001:
                    name += '***'
                elif p <= 0.01:
                    name += '**'
                elif p <= 0.05:
                    name += '*'

            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain, nabs, ncum]
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

                nabsv.append(nabs)
                ncumv.append(ncum)

    # add pasc at last
    if pasc_dx:
        organ_n[-1] += 1
        labs.append(pasc_row[0])
        measure.append(pasc_row[1])
        lower.append(pasc_row[2][0])
        upper.append(pasc_row[2][1])
        pval.append(pasc_row[3])

        nabsv.append(nabs)
        ncumv.append(ncum)

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          nabs=nabsv, ncumIncidence=ncumv)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR', add_label1='CIF per\n1000', add_label2='No. of\nCases')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='black'),   # , linecolor='#fcb2ab')
    ax = p.plot(figsize=(8.6, .38 * len(labs)), t_adjuster=0.0108, max_value=3.5, min_value=0.9, size=5, decimal=2,
                text_right=text_right)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')  # linewidth=1,

    ax.set_yticklabels(labs, fontsize=11.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/organ/dx/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'New-dx_hr_{}-pMultiBY-{}-V3-{}.png'.format(
        'all',
        '-withU099' if pasc_dx else '',
        'text_right' if text_right else ''), bbox_inches='tight', dpi=600)

    plt.savefig(output_dir + 'New-dx_hr_{}-pMultiBY-{}-V3-{}.pdf'.format(
        'all',
        '-withU099' if pasc_dx else '',
        'text_right' if text_right else ''), bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_V4(database='V15_COVID19', star=True, pasc_dx=False, text_right=False):

    if database == 'V15_COVID19':
        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
    elif database == 'oneflorida':
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_oneflorida-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')

    df_select = df.sort_values(by='hr-w', ascending=False)
    df_select = df_select.loc[df_select['selected'] == 1, :]  #
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

    nabsv = []
    ncumv = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            pasc = row['pasc']
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']

            # nabs = row['no. pasc in +']
            ncum = stringlist_2_list(row['cif_1_w'])[-1] * 1000
            ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]

            # use nabs for ncum_ci_negative
            nabs = stringlist_2_list(row['cif_0_w'])[-1] * 1000

            if star:
                if p <= 0.001:
                    name += '***'
                elif p <= 0.01:
                    name += '**'
                elif p <= 0.05:
                    name += '*'

            if (database == 'V15_COVID19') and (row['selected'] == 1) and (row['selected oneflorida'] == 1):
                name += r'$^{‡}$'

            if (database == 'oneflorida') and (row['selected'] == 1) and (row['selected insight'] == 1):
                name += r'$^{‡}$'

            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain, nabs, ncum]
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

                nabsv.append(nabs)
                ncumv.append(ncum)

    # add pasc at last
    if pasc_dx:
        organ_n[-1] += 1
        labs.append(pasc_row[0])
        measure.append(pasc_row[1])
        lower.append(pasc_row[2][0])
        upper.append(pasc_row[2][1])
        pval.append(pasc_row[3])

        nabsv.append(nabs)
        ncumv.append(ncum)

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          nabs=nabsv, ncumIncidence=ncumv)
    p.labels(scale='log')

    # organ = 'ALL'
    # p.labels(effectmeasure='aHR', add_label1='CIF per\n1000', add_label2='No. of\nCases')  # aHR
    p.labels(effectmeasure='aHR', add_label1='CIF per\n1000\nin Pos', add_label2='CIF per\n1000\nin Neg')

    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='black'),   # , linecolor='#fcb2ab')
    ax = p.plot(figsize=(8.6, .38 * len(labs)), t_adjuster=0.0108, max_value=3.5, min_value=0.9, size=5, decimal=2,
                text_right=text_right)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')  # linewidth=1,

    # ax.set_yticklabels(labs, fontsize=11.5)
    ax.set_yticklabels(labs, fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/{}/output/character/outcome/figure/organ/dx/'.format(database)
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'new-trim-dx_hr_{}-pMultiBY-{}-V4-2CIF-{}.png'.format(
        'all',
        '-withU099' if pasc_dx else '',
        'text_right' if text_right else ''), bbox_inches='tight', dpi=600)

    plt.savefig(output_dir + 'new-trim-dx_hr_{}-pMultiBY-{}-V4-2CIF-{}.pdf'.format(
        'all',
        '-withU099' if pasc_dx else '',
        'text_right' if text_right else ''), bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_med_organ(pvalue=0.05 / 459, star=True, datasite='insight'):
    if datasite == 'oneflorida':
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/MED-all/causal_effects_specific_med.xlsx',
            sheet_name='med_selected')
        df = df.rename(columns={'hr-w-p': 'Hazard Ratio, Adjusted, P-Value',
                                'hr-w': 'Hazard Ratio, Adjusted',
                                'hr-w-CI': 'Hazard Ratio, Adjusted, Confidence Interval'
                                })
        n_threshold = 66
    else:

        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='med')
        n_threshold = 100

    df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    # pvalue = 0.05 / 459  # 0.05 / 137
    df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    df_select = df_select.loc[df_select['no. pasc in +'] >= n_threshold, :]
    # df_select = df
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Diseases of the Nervous System',
        # 'Diseases of the Eye and Adnexa',
        'Diseases of the Skin and Subcutaneous Tissue',
        'Diseases of the Respiratory System',
        'Diseases of the Circulatory System',
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism',
        'Endocrine, Nutritional and Metabolic Diseases',
        'Diseases of the Digestive System',
        'Diseases of the Genitourinary System',
        'Diseases of the Musculoskeletal System and Connective Tissue',
        'Certain Infectious and Parasitic Diseases',
    ]
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple']
            hr = row['Hazard Ratio, Adjusted']
            ci = stringlist_2_list(row['Hazard Ratio, Adjusted, Confidence Interval'])
            p = row['Hazard Ratio, Adjusted, P-Value']
            domain = row['Organ Domain']
            # if name == 'General PASC':
            #     continue
            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 4:
                    name = ' '.join(name.split()[:3]) + '\n' + ' '.join(name.split()[3:])
                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    if datasite == 'oneflorida':
        c = '#A986B5'  # '#A986B5'
        p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='#fcb2ab')
        ax = p.plot(figsize=(8, 0.42 * len(labs)), t_adjuster=0.026, max_value=3.5, min_value=0.9, size=5,
                    decimal=2)  # * 27 / 45 *
    else:
        c = '#5494DA'  # '#A986B5'
        p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='#fcb2ab')
        ax = p.plot(figsize=(8, 0.42 * 27 / 45 * len(labs)), t_adjuster=0.0108, max_value=3.5, min_value=0.9, size=5,
                    decimal=2)  #
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    if datasite == 'oneflorida':
        output_dir = r'../data/oneflorida/output/character/outcome/figure/organ/med/'
    else:
        output_dir = r'../data/V15_COVID19/output/character/outcome/figure/organ/med/'

    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'med_hr_{}-p{:.3f}-hrGe1-nGe100.png'.format('all', pvalue), bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'med_hr_{}-p{:.3f}-hrGe1-nGe100.pdf'.format('all', pvalue), bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_med_organ_V2(pvalue=0.05 / 459, star=True, datasite='insight'):
    if datasite == 'oneflorida':
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/MED-all/causal_effects_specific_med.xlsx',
            sheet_name='med_selected')
        df = df.rename(columns={'hr-w-p': 'Hazard Ratio, Adjusted, P-Value',
                                'hr-w': 'Hazard Ratio, Adjusted',
                                'hr-w-CI': 'Hazard Ratio, Adjusted, Confidence Interval'
                                })
        n_threshold = 66
    else:

        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='med')
        n_threshold = 100

    df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    # pvalue = 0.05 / 459  # 0.05 / 137
    df_select = df_select.loc[df_select['selected'] == 1, :]
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    # df_select = df_select.loc[df_select['no. pasc in +'] >= n_threshold, :]
    # df_select = df
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Diseases of the Nervous System',
        # 'Diseases of the Eye and Adnexa',
        'Diseases of the Skin and Subcutaneous Tissue',
        'Diseases of the Respiratory System',
        'Diseases of the Circulatory System',
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism',
        'Endocrine, Nutritional and Metabolic Diseases',
        'Diseases of the Digestive System',
        'Diseases of the Genitourinary System',
        'Diseases of the Musculoskeletal System and Connective Tissue',
        'Certain Infectious and Parasitic Diseases',
        'General'
    ]
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple']
            hr = row['Hazard Ratio, Adjusted']
            ci = stringlist_2_list(row['Hazard Ratio, Adjusted, Confidence Interval'])
            p = row['Hazard Ratio, Adjusted, P-Value']
            domain = row['Organ Domain']
            # if name == 'General PASC':
            #     continue
            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 4:
                    name = ' '.join(name.split()[:3]) + '\n' + ' '.join(name.split()[3:])
                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    if datasite == 'oneflorida':
        c = '#A986B5'  # '#A986B5'
        p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='#fcb2ab')
        ax = p.plot(figsize=(8, 0.42 * len(labs)), t_adjuster=0.026, max_value=3.5, min_value=0.9, size=5,
                    decimal=2)  # * 27 / 45 *
    else:
        c = '#5494DA'  # '#A986B5'
        p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='#fcb2ab')
        ax = p.plot(figsize=(8, 0.42 * 27 / 45 * len(labs)), t_adjuster=0.0108, max_value=3.5, min_value=0.9, size=5,
                    decimal=2)  #
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    if datasite == 'oneflorida':
        output_dir = r'../data/oneflorida/output/character/outcome/figure/organ/med/'
    else:
        output_dir = r'../data/V15_COVID19/output/character/outcome/figure/organ/med/'

    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'med_hr_{}-p{:.3f}-hrGe1-nGe100-V2.png'.format('all', pvalue), bbox_inches='tight',
                dpi=600)
    plt.savefig(output_dir + 'med_hr_{}-p{:.3f}-hrGe1-nGe100-V2.pdf'.format('all', pvalue), bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_med_organ_V3(database='V15_COVID19', pvalue=0.05 / 459, ):
    if database == 'oneflorida':
        df = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/MED-all-new-trim/causal_effects_specific_med_oneflorida-MultiPval-DXMEDALL.xlsx',
            sheet_name='med')
        # df = df.rename(columns={'hr-w-p': 'Hazard Ratio, Adjusted, P-Value',
        #                         'hr-w': 'Hazard Ratio, Adjusted',
        #                         'hr-w-CI': 'Hazard Ratio, Adjusted, Confidence Interval'
        #                         })
        # n_threshold = 66
    elif database == 'V15_COVID19':

        df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/MED-all-new-trim/causal_effects_specific_med_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='med')
        # n_threshold = 100

    df_select = df.sort_values(by='hr-w', ascending=False)
    # pvalue = 0.05 / 459  # 0.05 / 137
    df_select = df_select.loc[df_select['selected'] == 1, :]
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    # df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    # df_select = df_select.loc[df_select['no. pasc in +'] >= n_threshold, :]
    # df_select = df
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Diseases of the Nervous System',
        # 'Diseases of the Eye and Adnexa',
        'Diseases of the Skin and Subcutaneous Tissue',
        'Diseases of the Respiratory System',
        'Diseases of the Circulatory System',
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism',
        'Endocrine, Nutritional and Metabolic Diseases',
        'Diseases of the Digestive System',
        'Diseases of the Genitourinary System',
        'Diseases of the Musculoskeletal System and Connective Tissue',
        # 'Certain Infectious and Parasitic Diseases',
        'General'
    ]
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []

    nabsv = []
    ncumv = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']

            if (database == 'V15_COVID19') and (row['selected'] == 1) and (row['selected oneflorida'] == 1):
                name += r'$^{‡}$'

            if (database == 'oneflorida') and (row['selected'] == 1) and (row['selected insight'] == 1):
                name += r'$^{‡}$'

            # nabs = row['no. pasc in +']
            ncum = stringlist_2_list(row['cif_1_w'])[-1] * 1000
            ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]

            # reuse- nabs for ci in neg
            nabs = stringlist_2_list(row['cif_0_w'])[-1] * 1000

            if domain == organ:
                organ_n[i] += 1
                if len(name.split()) >= 4:
                    name = ' '.join(name.split()[:3]) + '\n' + ' '.join(name.split()[3:])
                labs.append(name)
                measure.append(hr)
                lower.append(ci[0])
                upper.append(ci[1])
                pval.append(p)

                nabsv.append(nabs)
                ncumv.append(ncum)


    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          nabs=nabsv, ncumIncidence=ncumv)
    p.labels(scale='log')

    # organ = 'ALL'
    # p.labels(effectmeasure='aHR', add_label1='CIF per\n1000', add_label2='No. of\nCases')  # aHR
    p.labels(effectmeasure='aHR', add_label1='CIF per\n1000\nin Pos', add_label2='CIF per\n1000\nin Neg')

    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']

    if database == 'oneflorida':
        c = '#A986B5'  # '#A986B5'
        p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='#fcb2ab')
        ax = p.plot(figsize=(9, 0.43 * len(labs)), t_adjuster=0.026, max_value=3.5, min_value=0.9, size=5,
                    decimal=2)  # * 27 / 45 *
    else:
        c = '#5494DA'  # '#A986B5'
        p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='#fcb2ab')
        ax = p.plot(figsize=(9, 0.43 * 0.5 * len(labs)), t_adjuster=0.01, max_value=3.5, min_value=0.9, size=5,
                    decimal=2)  #
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=14)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    if database == 'oneflorida':
        output_dir = r'../data/oneflorida/output/character/outcome/figure/organ/med/'
    else:
        output_dir = r'../data/V15_COVID19/output/character/outcome/figure/organ/med/'

    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'new-trim-med_hr_{}-p{:.3f}-hrGe1-nGe100-V3-2CIF.png'.format('all', pvalue), bbox_inches='tight',
                dpi=600)
    plt.savefig(output_dir + 'new-trim-med_hr_{}-p{:.3f}-hrGe1-nGe100-V3-2CIF.pdf'.format('all', pvalue), bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_med_organ_compare2data(add_name=False, severity="all", star=False, select_criteria='',
                                             pvalue=0.05 / 596, add_pasc=False):
    if severity == 'all':
        df1 = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/MED-all-new-trim/causal_effects_specific_med_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='med')
        df2 = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/MED-all-new-trim/causal_effects_specific_med_oneflorida-MultiPval-DXMEDALL.xlsx',
            sheet_name='med')
    else:
        raise ValueError

    # if add_name:
    #     df_name = pd.read_excel(
    #         r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
    #         sheet_name='dx')
    #     df1 = pd.merge(df1, df_name[["pasc", "PASC Name Simple", "Organ Domain", "Original CCSR Domain", ]],
    #                    left_on='pasc', right_on='pasc', how='left')
    #     df1 = df1.rename(columns={x + '_y': x for x in ["PASC Name Simple", "Organ Domain", "Original CCSR Domain"]})

    df_aux = df2.rename(columns=lambda x: x + '_aux')
    df = pd.merge(df1, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index('i')

    # pvalue = 0.05 / 137  # 0.01  #

    if select_criteria == 'insight':
        # print('select_critera:', select_criteria)
        # _df = pd.read_excel(
        #     r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
        #     sheet_name='dx').set_index('i')
        # # pvalue = 0.01  # 0.05 / 137
        # _df_select = _df.sort_values(by='hr-w', ascending=False)
        # # _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
        # # _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
        # # _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]
        #
        # _df_select = _df_select.loc[_df_select['selected'] == 1, :]
        #
        # print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        #
        # df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
        # df_select = df_select.sort_values(by='hr-w', ascending=False)
        pass
    else:
        # def select_criteria_func(_df):
        #     _df_select = _df.sort_values(by='hr-w', ascending=False)
        #     _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue) | (_df_select['pasc'] == 'Muscle disorders'), :]  #
        #     _df_select = _df_select.loc[(_df_select['hr-w'] > 1) | (_df_select['pasc'] == 'Muscle disorders'), :]
        #     _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100) | (_df_select['pasc'] == 'Muscle disorders'),
        #                  :]
        #     print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        #     return _df_select

        def select_criteria_func_V2(_df):
            _df_select = _df.sort_values(by='hr-w', ascending=False)
            # _df_select = _df_select.loc[_df_select['selected'] == 1, :]
            _df_select = _df_select.loc[_df_select['selected oneflorida'] == 1, :]

            print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
            return _df_select

        # df_select = select_criteria_func(df)
        df_select = select_criteria_func_V2(df)

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
        'General'
    ]
    # 'Certain Infectious and Parasitic Diseases',
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    pasc_row2 = []
    color_list = []

    nabsv = []
    ncumv = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']
            pasc = row['pasc']

            hr2 = row['hr-w_aux']
            ci2 = stringlist_2_list(row['hr-w-CI_aux'])
            p2 = row['hr-w-p_aux']

            nabs = row['no. pasc in +']
            ncum = stringlist_2_list(row['cif_1_w'])[-1] * 1000
            ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]

            ncum_neg = stringlist_2_list(row['cif_0_w'])[-1] * 1000


            nabs2 = row['no. pasc in +_aux']
            ncum2 = stringlist_2_list(row['cif_1_w_aux'])[-1] * 1000
            ncum_ci2 = [stringlist_2_list(row['cif_1_w_CILower_aux'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper_aux'])[-1] * 1000]

            ncum_neg2 = stringlist_2_list(row['cif_0_w_aux'])[-1] * 1000

            # if star:
            #     if (p > 0.05 / 137) and (p <= 0.01):
            #         name += '**'
            #     elif (p > 0.01) and (p <= 0.05):
            #         name += '*'

            if (row['selected'] == 1) and (row['selected_aux'] == 1):
                name += r'$^{‡}$'


            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain]
                pasc_row2 = [name, hr2, ci2, p2, domain]
                continue
            if domain == organ:
                organ_n[i] += 2
                # if len(name.split()) == 4:
                #     name = ' '.join(name.split()[:2]) + '\n' + ' '.join(name.split()[2:])
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                labs.append(name)
                labs.append('')
                measure.append(hr)
                measure.append(hr2)
                lower.append(ci[0])
                upper.append(ci[1])
                lower.append(ci2[0])
                upper.append(ci2[1])
                pval.append(p)
                pval.append(p2)
                color_list.append('#ed6766')
                color_list.append('#A986B5')  # '#A986B5')

                nabsv.append(ncum_neg)  # nabsv.append(nabs)
                ncumv.append(ncum)
                nabsv.append(ncum_neg2) # nabsv.append(nabs2)
                ncumv.append(ncum2)

        if len(measure) == 0:
            continue

    # add pasc at last
    if add_pasc:
        if pasc_row:
            organ_n[-1] += 2
            labs.append(pasc_row[0])
            measure.append(pasc_row[1])
            lower.append(pasc_row[2][0])
            upper.append(pasc_row[2][1])
            pval.append(pasc_row[3])
            labs.append('')
            measure.append(pasc_row2[1])
            lower.append(pasc_row2[2][0])
            upper.append(pasc_row2[2][1])
            pval.append(pasc_row2[3])
            color_list.append('#ed6766')
            color_list.append('#A986B5')  # '#A986B5')
        else:
            print('pasc general not found!!!')

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          nabs=nabsv, ncumIncidence=ncumv)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR', add_label1='CIF per\n1000\nin Pos', add_label2='CIF per\n1000\nin Neg') # 'No. of\nCases')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=color_list, pointcolor=color_list)  # , linecolor='#fcb2ab')
    width = 9.
    height = .35 * len(labs)
    if len(labs) == 2:
        height = .3 * (len(labs) + 1)
    ax = p.plot_with_incidence(figsize=(width, height), t_adjuster=0.02, max_value=3, min_value=0.7, size=5, decimal=2)  # 0.02
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.0, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/organ/med_figure2Compare/'
    check_and_mkdir(output_dir)
    organ = 'all'
    i = 0
    plt.savefig(output_dir + 'new-trim-med-{}_hr-p{:.3f}-{}-new-NoPASC-VCIF.png'.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                dpi=650)
    plt.savefig(output_dir + 'new-trim-med-{}_hr-p{:.3f}-{}-new-NoPASC-VCIF.pdf'.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()

def plot_forest_for_dx_organ_compare2data(add_name=False, severity="all", star=True, select_criteria='',
                                          pvalue=0.05 / 137):
    if severity == 'all':
        df1 = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis')
        df2 = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific_v3.xlsx',
            sheet_name='diagnosis')
    elif severity == 'above65':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-above65/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-above65/causal_effects_specific.csv')
    elif severity == 'less65':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-less65/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-less65/causal_effects_specific.csv')
    elif severity == '65to75':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-65to75/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-65to75/causal_effects_specific.csv')

    if add_name:
        df_name = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis')
        df1 = pd.merge(df1, df_name[["pasc", "PASC Name Simple", "Organ Domain", "Original CCSR Domain", ]],
                       left_on='pasc', right_on='pasc', how='left')
        df1 = df1.rename(columns={x + '_y': x for x in ["PASC Name Simple", "Organ Domain", "Original CCSR Domain"]})

    df_aux = df2.rename(columns=lambda x: x + '_aux')
    df = pd.merge(df1, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index('i')

    # pvalue = 0.05 / 137  # 0.01  #

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
        # def select_criteria_func(_df):
        #     _df_select = _df.sort_values(by='hr-w', ascending=False)
        #     _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue) | (_df_select['pasc'] == 'Muscle disorders'), :]  #
        #     _df_select = _df_select.loc[(_df_select['hr-w'] > 1) | (_df_select['pasc'] == 'Muscle disorders'), :]
        #     _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100) | (_df_select['pasc'] == 'Muscle disorders'),
        #                  :]
        #     print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        #     return _df_select

        def select_criteria_func(_df):
            _df_select = _df.sort_values(by='hr-w', ascending=False)
            _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue), :]  #
            _df_select = _df_select.loc[(_df_select['hr-w'] > 1), :]
            _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100),
                         :]
            print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
            return _df_select

        df_select = select_criteria_func(df)

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
        'General'
    ]
    # 'Certain Infectious and Parasitic Diseases',
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    pasc_row2 = []
    color_list = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']
            pasc = row['pasc']

            hr2 = row['hr-w_aux']
            ci2 = stringlist_2_list(row['hr-w-CI_aux'])
            p2 = row['hr-w-p_aux']

            if star:
                if (p > 0.05 / 137) and (p <= 0.01):
                    name += '**'
                elif (p > 0.01) and (p <= 0.05):
                    name += '*'

            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain]
                pasc_row2 = [name, hr2, ci2, p2, domain]
                continue
            if domain == organ:
                organ_n[i] += 2
                # if len(name.split()) == 4:
                #     name = ' '.join(name.split()[:2]) + '\n' + ' '.join(name.split()[2:])
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                labs.append(name)
                labs.append('')
                measure.append(hr)
                measure.append(hr2)
                lower.append(ci[0])
                upper.append(ci[1])
                lower.append(ci2[0])
                upper.append(ci2[1])
                pval.append(p)
                pval.append(p2)
                color_list.append('#ed6766')
                color_list.append('#A986B5')  # '#A986B5')
        if len(measure) == 0:
            continue

    # add pasc at last
    if pasc_row:
        organ_n[-1] += 2
        labs.append(pasc_row[0])
        measure.append(pasc_row[1])
        lower.append(pasc_row[2][0])
        upper.append(pasc_row[2][1])
        pval.append(pasc_row[3])
        labs.append('')
        measure.append(pasc_row2[1])
        lower.append(pasc_row2[2][0])
        upper.append(pasc_row2[2][1])
        pval.append(pasc_row2[3])
        color_list.append('#ed6766')
        color_list.append('#A986B5')  # '#A986B5')
    else:
        print('pasc general not found!!!')

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=color_list, pointcolor=color_list)  # , linecolor='#fcb2ab')
    width = 9.
    height = .28 * len(labs)
    if len(labs) == 2:
        height = .3 * (len(labs) + 1)
    ax = p.plot(figsize=(width, height), t_adjuster=0.010, max_value=3, min_value=0.7, size=5, decimal=2)  # 0.02
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.0, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=11.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/figure2Compare/{}/'.format(severity)
    check_and_mkdir(output_dir)
    organ = 'all'
    i = 0
    plt.savefig(output_dir + '{}_hr-p{:.3f}-{}-new-NoPASCV2.png'.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                dpi=650)
    plt.savefig(output_dir + '{}_hr-p{:.3f}-{}-new-NoPASCV2.pdf'.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_compare2data_V2(add_name=False, severity="all", star=True, select_criteria='',
                                             pvalue=0.05 / 137, add_pasc=False):
    if severity == 'all':
        df1 = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis')
        df2 = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all/causal_effects_specific_v3.xlsx',
            sheet_name='diagnosis')
    elif severity == 'above65':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-above65/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-above65/causal_effects_specific.csv')
    elif severity == 'less65':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-less65/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-less65/causal_effects_specific.csv')
    elif severity == '65to75':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-65to75/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-65to75/causal_effects_specific.csv')

    if add_name:
        df_name = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
            sheet_name='diagnosis')
        df1 = pd.merge(df1, df_name[["pasc", "PASC Name Simple", "Organ Domain", "Original CCSR Domain", ]],
                       left_on='pasc', right_on='pasc', how='left')
        df1 = df1.rename(columns={x + '_y': x for x in ["PASC Name Simple", "Organ Domain", "Original CCSR Domain"]})

    df_aux = df2.rename(columns=lambda x: x + '_aux')
    df = pd.merge(df1, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index('i')

    # pvalue = 0.05 / 137  # 0.01  #

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
        # def select_criteria_func(_df):
        #     _df_select = _df.sort_values(by='hr-w', ascending=False)
        #     _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue) | (_df_select['pasc'] == 'Muscle disorders'), :]  #
        #     _df_select = _df_select.loc[(_df_select['hr-w'] > 1) | (_df_select['pasc'] == 'Muscle disorders'), :]
        #     _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100) | (_df_select['pasc'] == 'Muscle disorders'),
        #                  :]
        #     print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        #     return _df_select

        def select_criteria_func(_df):
            _df_select = _df.sort_values(by='hr-w', ascending=False)
            _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue), :]  #
            _df_select = _df_select.loc[(_df_select['hr-w'] > 1), :]
            _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100),
                         :]
            print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
            return _df_select

        def select_criteria_func_V2(_df):
            _df_select = _df.sort_values(by='hr-w', ascending=False)
            _df_select = _df_select.loc[_df_select['selected'] == 1, :]

            print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
            return _df_select

        # df_select = select_criteria_func(df)
        df_select = select_criteria_func_V2(df)

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
        'General'
    ]
    # 'Certain Infectious and Parasitic Diseases',
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    pasc_row2 = []
    color_list = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']
            pasc = row['pasc']

            hr2 = row['hr-w_aux']
            ci2 = stringlist_2_list(row['hr-w-CI_aux'])
            p2 = row['hr-w-p_aux']

            if star:
                if (p > 0.05 / 137) and (p <= 0.01):
                    name += '**'
                elif (p > 0.01) and (p <= 0.05):
                    name += '*'

            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain]
                pasc_row2 = [name, hr2, ci2, p2, domain]
                continue
            if domain == organ:
                organ_n[i] += 2
                # if len(name.split()) == 4:
                #     name = ' '.join(name.split()[:2]) + '\n' + ' '.join(name.split()[2:])
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                labs.append(name)
                labs.append('')
                measure.append(hr)
                measure.append(hr2)
                lower.append(ci[0])
                upper.append(ci[1])
                lower.append(ci2[0])
                upper.append(ci2[1])
                pval.append(p)
                pval.append(p2)
                color_list.append('#ed6766')
                color_list.append('#A986B5')  # '#A986B5')
        if len(measure) == 0:
            continue

    # add pasc at last
    if add_pasc:
        if pasc_row:
            organ_n[-1] += 2
            labs.append(pasc_row[0])
            measure.append(pasc_row[1])
            lower.append(pasc_row[2][0])
            upper.append(pasc_row[2][1])
            pval.append(pasc_row[3])
            labs.append('')
            measure.append(pasc_row2[1])
            lower.append(pasc_row2[2][0])
            upper.append(pasc_row2[2][1])
            pval.append(pasc_row2[3])
            color_list.append('#ed6766')
            color_list.append('#A986B5')  # '#A986B5')
        else:
            print('pasc general not found!!!')

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=color_list, pointcolor=color_list)  # , linecolor='#fcb2ab')
    width = 9.
    height = .28 * len(labs)
    if len(labs) == 2:
        height = .3 * (len(labs) + 1)
    ax = p.plot(figsize=(width, height), t_adjuster=0.010, max_value=3, min_value=0.7, size=5, decimal=2)  # 0.02
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.0, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=11.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/figure2Compare/{}/'.format(severity)
    check_and_mkdir(output_dir)
    organ = 'all'
    i = 0
    plt.savefig(output_dir + '{}_hr-p{:.3f}-{}-new-NoPASC-V3.png'.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                dpi=650)
    plt.savefig(output_dir + '{}_hr-p{:.3f}-{}-new-NoPASC-V3.pdf'.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_compare3data_V2 (add_name=False, severity="all", star=True, select_criteria='',
                                              pvalue=0.05 / 137, add_pasc=False):

    df1 = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/pooled/DX-all/causal_effects_specific-addinfo.xlsx',
        sheet_name='diagnosis')
    df2 = pd.read_csv(
        r'../data/V15_COVID19/output/character/outcome/pooled/DX-1stwave/causal_effects_specific.csv')
    df3 = pd.read_csv(
        r'../data/V15_COVID19/output/character/outcome/pooled/DX-delta/causal_effects_specific.csv')

    df_aux2 = df2.rename(columns=lambda x: x + '_aux2')
    df_aux3 = df3.rename(columns=lambda x: x + '_aux3')

    df = pd.merge(df1, df_aux2, left_on='pasc', right_on='pasc_aux2', how='left')
    df = pd.merge(df, df_aux3, left_on='pasc', right_on='pasc_aux3', how='left').set_index('i')

    # pvalue = 0.05 / 137  # 0.01  #
    #
    # if select_criteria == 'insight':
    #     print('select_critera:', select_criteria)
    #     _df = pd.read_excel(
    #         r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
    #         sheet_name='diagnosis').set_index('i')
    #     pvalue = 0.01  # 0.05 / 137
    #     _df_select = _df.sort_values(by='hr-w', ascending=False)
    #     _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
    #     _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
    #     _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]
    #
    #     print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
    #
    #     df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
    #     df_select = df_select.sort_values(by='hr-w', ascending=False)
    # else:
    #     # def select_criteria_func(_df):
    #     #     _df_select = _df.sort_values(by='hr-w', ascending=False)
    #     #     _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue) | (_df_select['pasc'] == 'Muscle disorders'), :]  #
    #     #     _df_select = _df_select.loc[(_df_select['hr-w'] > 1) | (_df_select['pasc'] == 'Muscle disorders'), :]
    #     #     _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100) | (_df_select['pasc'] == 'Muscle disorders'),
    #     #                  :]
    #     #     print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
    #     #     return _df_select
    #
    #     def select_criteria_func(_df):
    #         _df_select = _df.sort_values(by='hr-w', ascending=False)
    #         _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue), :]  #
    #         _df_select = _df_select.loc[(_df_select['hr-w'] > 1), :]
    #         _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100),
    #                      :]
    #         print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
    #         return _df_select

    def select_criteria_func_V2(_df):
        _df_select = _df.sort_values(by='hr-w', ascending=False)
        _df_select = _df_select.loc[_df_select['selected'] == 1, :]

        print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        return _df_select

    # df_select = select_criteria_func(df)
    df_select = select_criteria_func_V2(df)

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
        'General'
    ]
    # 'Certain Infectious and Parasitic Diseases',
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    pasc_row2 = []
    pasc_row3 = []
    color_list = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']
            pasc = row['pasc']

            hr2 = row['hr-w_aux2']
            ci2 = stringlist_2_list(row['hr-w-CI_aux2'])
            p2 = row['hr-w-p_aux2']

            hr3 = row['hr-w_aux3']
            ci3 = stringlist_2_list(row['hr-w-CI_aux3'])
            p3 = row['hr-w-p_aux3']

            #
            # if star:
            #     if (p > 0.05 / 137) and (p <= 0.01):
            #         name += '**'
            #     elif (p > 0.01) and (p <= 0.05):
            #         name += '*'

            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain]
                pasc_row2 = [name, hr2, ci2, p2, domain]
                pasc_row3 = [name, hr3, ci3, p3, domain]
                continue

            if domain == organ:
                organ_n[i] += 3
                # if len(name.split()) == 4:
                #     name = ' '.join(name.split()[:2]) + '\n' + ' '.join(name.split()[2:])
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])
                # if len(re.split('[, ]', name)) >= 5:
                #     name = ' '.join(re.split(', ', name)[:4]) + '\n' + ' '.join(re.split(', ', name)[4:])

                if name == 'Pulmonary fibrosis, edema, inflammation':
                    name = 'Pulmonary fibrosis,\nedema, inflammation'

                labs.append(name)
                labs.append('')
                labs.append('')
                measure.append(hr)
                measure.append(hr2)
                measure.append(hr3)
                lower.append(ci[0])
                upper.append(ci[1])
                lower.append(ci2[0])
                upper.append(ci2[1])
                lower.append(ci3[0])
                upper.append(ci3[1])
                pval.append(p)
                pval.append(p2)
                pval.append(p3)
                color_list.append('#808080')
                color_list.append('#ed6766')
                color_list.append('#A986B5')  # '#A986B5')
        if len(measure) == 0:
            continue

    # add pasc at last
    # if add_pasc:
    #     if pasc_row:
    #         organ_n[-1] += 3
    #         labs.append(pasc_row[0])
    #         measure.append(pasc_row[1])
    #         lower.append(pasc_row[2][0])
    #         upper.append(pasc_row[2][1])
    #         pval.append(pasc_row[3])
    #         labs.append('')
    #         measure.append(pasc_row2[1])
    #         lower.append(pasc_row2[2][0])
    #         upper.append(pasc_row2[2][1])
    #         pval.append(pasc_row2[3])
    #         color_list.append('#ed6766')
    #         color_list.append('#A986B5')  # '#A986B5')
    #     else:
    #         print('pasc general not found!!!')

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="o", errorbarcolor=color_list, pointcolor=color_list)  # , linecolor='#fcb2ab')
    width = 8.
    height = .23 * len(labs)
    if len(labs) == 2:
        height = .3 * (len(labs) + 1)
    ax = p.plot(figsize=(width, height), t_adjuster=0.0050, max_value=3.5, min_value=0.7, size=5, decimal=2)  # 0.02
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.0, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/figure2Variant/'
    check_and_mkdir(output_dir)
    organ = 'all'
    i = 0
    plt.savefig(output_dir + 'poole-1st-delta-23.png',  #.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                dpi=600)
    plt.savefig(output_dir + 'poole-1st-delta-23.pdf',  # .format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_compare2data_V3(add_name=False, severity="all", star=False, select_criteria='',
                                             pvalue=0.05 / 596, add_pasc=False):
    if severity == 'all':
        df1 = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
        df2 = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_oneflorida-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
    elif severity == 'above65':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-above65/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-above65/causal_effects_specific.csv')
    elif severity == 'less65':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-less65/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-less65/causal_effects_specific.csv')
    elif severity == '65to75':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-65to75/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-65to75/causal_effects_specific.csv')

    if add_name:
        df_name = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
        df1 = pd.merge(df1, df_name[["pasc", "PASC Name Simple", "Organ Domain", "Original CCSR Domain", ]],
                       left_on='pasc', right_on='pasc', how='left')
        df1 = df1.rename(columns={x + '_y': x for x in ["PASC Name Simple", "Organ Domain", "Original CCSR Domain"]})

    df_aux = df2.rename(columns=lambda x: x + '_aux')
    df = pd.merge(df1, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index('i')

    # pvalue = 0.05 / 137  # 0.01  #

    if select_criteria == 'insight':
        print('select_critera:', select_criteria)
        _df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx').set_index('i')
        # pvalue = 0.01  # 0.05 / 137
        _df_select = _df.sort_values(by='hr-w', ascending=False)
        # _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
        # _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
        # _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]

        _df_select = _df_select.loc[_df_select['selected'] == 1, :]

        print('_df_select.shape:', _df_select.shape, _df_select['pasc'])

        df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
        df_select = df_select.sort_values(by='hr-w', ascending=False)
    else:
        # def select_criteria_func(_df):
        #     _df_select = _df.sort_values(by='hr-w', ascending=False)
        #     _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue) | (_df_select['pasc'] == 'Muscle disorders'), :]  #
        #     _df_select = _df_select.loc[(_df_select['hr-w'] > 1) | (_df_select['pasc'] == 'Muscle disorders'), :]
        #     _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100) | (_df_select['pasc'] == 'Muscle disorders'),
        #                  :]
        #     print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        #     return _df_select

        def select_criteria_func(_df):
            _df_select = _df.sort_values(by='hr-w', ascending=False)
            _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue), :]  #
            _df_select = _df_select.loc[(_df_select['hr-w'] > 1), :]
            _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100),
                         :]
            print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
            return _df_select

        def select_criteria_func_V2(_df):
            _df_select = _df.sort_values(by='hr-w', ascending=False)
            _df_select = _df_select.loc[_df_select['selected'] == 1, :]

            print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
            return _df_select

        # df_select = select_criteria_func(df)
        df_select = select_criteria_func_V2(df)

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
        'General'
    ]
    # 'Certain Infectious and Parasitic Diseases',
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    pasc_row2 = []
    color_list = []

    nabsv = []
    ncumv = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']
            pasc = row['pasc']

            hr2 = row['hr-w_aux']
            ci2 = stringlist_2_list(row['hr-w-CI_aux'])
            p2 = row['hr-w-p_aux']

            nabs = row['no. pasc in +']
            ncum = stringlist_2_list(row['cif_1_w'])[-1] * 1000
            ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]

            ncum_neg = stringlist_2_list(row['cif_0_w'])[-1] * 1000


            nabs2 = row['no. pasc in +_aux']
            ncum2 = stringlist_2_list(row['cif_1_w_aux'])[-1] * 1000
            ncum_ci2 = [stringlist_2_list(row['cif_1_w_CILower_aux'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper_aux'])[-1] * 1000]

            ncum_neg2 = stringlist_2_list(row['cif_0_w_aux'])[-1] * 1000

            # if star:
            #     if (p > 0.05 / 137) and (p <= 0.01):
            #         name += '**'
            #     elif (p > 0.01) and (p <= 0.05):
            #         name += '*'

            if (row['selected'] == 1) and (row['selected_aux'] == 1):
                name += r'$^{‡}$'


            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain]
                pasc_row2 = [name, hr2, ci2, p2, domain]
                continue
            if domain == organ:
                organ_n[i] += 2
                # if len(name.split()) == 4:
                #     name = ' '.join(name.split()[:2]) + '\n' + ' '.join(name.split()[2:])
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                labs.append(name)
                labs.append('')
                measure.append(hr)
                measure.append(hr2)
                lower.append(ci[0])
                upper.append(ci[1])
                lower.append(ci2[0])
                upper.append(ci2[1])
                pval.append(p)
                pval.append(p2)
                color_list.append('#ed6766')
                color_list.append('#A986B5')  # '#A986B5')

                nabsv.append(ncum_neg)  # nabsv.append(nabs)
                ncumv.append(ncum)
                nabsv.append(ncum_neg2) # nabsv.append(nabs2)
                ncumv.append(ncum2)

        if len(measure) == 0:
            continue

    # add pasc at last
    if add_pasc:
        if pasc_row:
            organ_n[-1] += 2
            labs.append(pasc_row[0])
            measure.append(pasc_row[1])
            lower.append(pasc_row[2][0])
            upper.append(pasc_row[2][1])
            pval.append(pasc_row[3])
            labs.append('')
            measure.append(pasc_row2[1])
            lower.append(pasc_row2[2][0])
            upper.append(pasc_row2[2][1])
            pval.append(pasc_row2[3])
            color_list.append('#ed6766')
            color_list.append('#A986B5')  # '#A986B5')
        else:
            print('pasc general not found!!!')

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          nabs=nabsv, ncumIncidence=ncumv)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR', add_label1='CIF per\n1000\nin Pos', add_label2='CIF per\n1000\nin Neg') # 'No. of\nCases')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=color_list, pointcolor=color_list)  # , linecolor='#fcb2ab')
    width = 9.
    height = .21 * len(labs)
    if len(labs) == 2:
        height = .3 * (len(labs) + 1)
    ax = p.plot_with_incidence(figsize=(width, height), t_adjuster=0.005, max_value=3, min_value=0.7, size=5, decimal=2)  # 0.02
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.0, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/figure2Compare/{}/'.format(severity)
    check_and_mkdir(output_dir)
    organ = 'all'
    i = 0
    plt.savefig(output_dir + 'new-trim-{}_hr-p{:.3f}-{}-new-NoPASC-VCIF.png'.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                dpi=650)
    plt.savefig(output_dir + 'new-trim-{}_hr-p{:.3f}-{}-new-NoPASC-VCIF.pdf'.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_compare2data_sensitivity(add_name=False, severity="all", star=False, select_criteria='',
                                                        pvalue=0.05 / 596, add_pasc=False):
    if severity == 'all':
        df1 = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
        df2 = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_oneflorida-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
    elif severity == 'above65':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-above65/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-above65/causal_effects_specific.csv')
    elif severity == 'less65':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-less65/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-less65/causal_effects_specific.csv')
    elif severity == '65to75':
        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/DX-65to75/causal_effects_specific.csv')
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-65to75/causal_effects_specific.csv')

    if add_name:
        df_name = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
        df1 = pd.merge(df1, df_name[["pasc", "PASC Name Simple", "Organ Domain", "Original CCSR Domain", ]],
                       left_on='pasc', right_on='pasc', how='left')
        df1 = df1.rename(columns={x + '_y': x for x in ["PASC Name Simple", "Organ Domain", "Original CCSR Domain"]})

    df_aux = df2.rename(columns=lambda x: x + '_aux')
    df = pd.merge(df1, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index('i')

    # pvalue = 0.05 / 137  # 0.01  #

    if select_criteria == 'insight':
        print('select_critera:', select_criteria)
        _df = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx').set_index('i')
        # pvalue = 0.01  # 0.05 / 137
        _df_select = _df.sort_values(by='hr-w', ascending=False)
        # _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
        # _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
        # _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]

        _df_select = _df_select.loc[_df_select['selected'] == 1, :]

        print('_df_select.shape:', _df_select.shape, _df_select['pasc'])

        df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
        df_select = df_select.sort_values(by='hr-w', ascending=False)
    else:
        # def select_criteria_func(_df):
        #     _df_select = _df.sort_values(by='hr-w', ascending=False)
        #     _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue) | (_df_select['pasc'] == 'Muscle disorders'), :]  #
        #     _df_select = _df_select.loc[(_df_select['hr-w'] > 1) | (_df_select['pasc'] == 'Muscle disorders'), :]
        #     _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100) | (_df_select['pasc'] == 'Muscle disorders'),
        #                  :]
        #     print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        #     return _df_select
        def select_criteria_func_V2(_df):
            _df_select = _df.sort_values(by='hr-w', ascending=False)
            # _df_select = _df_select.loc[_df_select['selected'] == 1, :]
            _df_select = _df_select.loc[(_df_select['sensitivity'] == 1) | (_df_select['sensitivity_aux'] == 1), :]


            print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
            return _df_select

        # df_select = select_criteria_func(df)
        df_select = select_criteria_func_V2(df)

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
        'General'
    ]
    # 'Certain Infectious and Parasitic Diseases',
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    pasc_row2 = []
    color_list = []

    nabsv = []
    ncumv = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']
            pasc = row['pasc']

            hr2 = row['hr-w_aux']
            ci2 = stringlist_2_list(row['hr-w-CI_aux'])
            p2 = row['hr-w-p_aux']

            nabs = row['no. pasc in +']
            ncum = stringlist_2_list(row['cif_1_w'])[-1] * 1000
            ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]

            ncum_neg = stringlist_2_list(row['cif_0_w'])[-1] * 1000


            nabs2 = row['no. pasc in +_aux']
            ncum2 = stringlist_2_list(row['cif_1_w_aux'])[-1] * 1000
            ncum_ci2 = [stringlist_2_list(row['cif_1_w_CILower_aux'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper_aux'])[-1] * 1000]

            ncum_neg2 = stringlist_2_list(row['cif_0_w_aux'])[-1] * 1000

            # if star:
            #     if (p > 0.05 / 137) and (p <= 0.01):
            #         name += '**'
            #     elif (p > 0.01) and (p <= 0.05):
            #         name += '*'

            if (row['selected'] == 1) and (row['selected_aux'] == 1):
                name += r'$^{‡}$'


            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain]
                pasc_row2 = [name, hr2, ci2, p2, domain]
                continue
            if domain == organ:
                organ_n[i] += 2
                # if len(name.split()) == 4:
                #     name = ' '.join(name.split()[:2]) + '\n' + ' '.join(name.split()[2:])
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                labs.append(name)
                labs.append('')
                measure.append(hr)
                measure.append(hr2)
                lower.append(ci[0])
                upper.append(ci[1])
                lower.append(ci2[0])
                upper.append(ci2[1])
                pval.append(p)
                pval.append(p2)
                color_list.append('#ed6766')
                color_list.append('#A986B5')  # '#A986B5')

                nabsv.append(ncum_neg)  # nabsv.append(nabs)
                ncumv.append(ncum)
                nabsv.append(ncum_neg2) # nabsv.append(nabs2)
                ncumv.append(ncum2)

        if len(measure) == 0:
            continue

    # add pasc at last
    if add_pasc:
        if pasc_row:
            organ_n[-1] += 2
            labs.append(pasc_row[0])
            measure.append(pasc_row[1])
            lower.append(pasc_row[2][0])
            upper.append(pasc_row[2][1])
            pval.append(pasc_row[3])
            labs.append('')
            measure.append(pasc_row2[1])
            lower.append(pasc_row2[2][0])
            upper.append(pasc_row2[2][1])
            pval.append(pasc_row2[3])
            color_list.append('#ed6766')
            color_list.append('#A986B5')  # '#A986B5')
        else:
            print('pasc general not found!!!')

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          nabs=nabsv, ncumIncidence=ncumv)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR', add_label1='CIF per\n1000\nin Pos', add_label2='CIF per\n1000\nin Neg') # 'No. of\nCases')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=color_list, pointcolor=color_list)  # , linecolor='#fcb2ab')
    width = 10.5
    height = .3 * len(labs)
    if len(labs) == 2:
        height = .3 * (len(labs) + 1)
    ax = p.plot_with_incidence(figsize=(width, height), t_adjuster=0.005, max_value=3, min_value=0.7, size=5, decimal=2)  # 0.02
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    # for i in range(len(organ_n) - 1):
    #     ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.0, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/figure2Compare-sensitivity/{}/'.format(severity)
    check_and_mkdir(output_dir)
    organ = 'all'
    i = 0
    plt.savefig(output_dir + 'new-trim-SENSITIVITY-{}_hr-p{:.3f}-{}-new-NoPASC-VCIF.png'.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                dpi=650)
    plt.savefig(output_dir + 'new-trim-SENSITIVITY-{}_hr-p{:.3f}-{}-new-NoPASC-VCIF.pdf'.format(severity, pvalue, select_criteria),
                bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_compare2data_V3_moresensitivity(
        expdir, add_name=False, severity="all", star=False, select_criteria='', pvalue=0.05 / 596, add_pasc=False):
    print('expdir', expdir)

    if severity == 'all':
        df1_main = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
        df2_main = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_oneflorida-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')

        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/{}/causal_effects_specific.csv'.format(expdir),
            )
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/{}/causal_effects_specific.csv'.format(expdir),
            )
    else:
        raise ValueError


    # add selected feature, name, hr in main analyses
    df1 = pd.merge(df1, df1_main[["pasc", "PASC Name Simple", "Organ Domain",  # "Original CCSR Domain",
                                  'selected', 'selected oneflorida', 'sensitivity',
                                  'hr-w', 'hr-w-CI', 'hr-w-p']],
                   left_on='pasc', right_on='pasc', how='left', suffixes=('', '_main'))
    # df1 = df1.rename(columns={x + '_y': x for x in ["PASC Name Simple", "Organ Domain", "Original CCSR Domain"]})

    df2 = pd.merge(df2, df2_main[["pasc", 'selected', 'hr-w', 'hr-w-CI', 'hr-w-p']],
                   left_on='pasc', right_on='pasc', how='left', suffixes=('', '_main'))
    df_aux = df2.rename(columns=lambda x: x + '_aux')
    df = pd.merge(df1, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index('i')

    # pvalue = 0.05 / 137  # 0.01  #

    if select_criteria == 'insight':
        # print('select_critera:', select_criteria)
        # _df = pd.read_excel(
        #     r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
        #     sheet_name='dx').set_index('i')
        # # pvalue = 0.01  # 0.05 / 137
        # _df_select = _df.sort_values(by='hr-w', ascending=False)
        # # _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
        # # _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
        # # _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]
        #
        # _df_select = _df_select.loc[_df_select['selected'] == 1, :]
        #
        # print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        #
        # df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
        # df_select = df_select.sort_values(by='hr-w', ascending=False)
        pass
    else:
        def select_criteria_func_V3(_df):
            _df_select = _df.sort_values(by='hr-w_main', ascending=False)
            _df_select = _df_select.loc[_df_select['selected'] == 1, :]

            print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
            return _df_select

        # df_select = select_criteria_func(df)
        df_select = select_criteria_func_V3(df)

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
        'General'
    ]
    # 'Certain Infectious and Parasitic Diseases',
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    pasc_row2 = []
    color_list = []

    # nabsv = []
    # ncumv = []

    addcol1 = []
    addcol2 = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']
            pasc = row['pasc']

            hr_main = row['hr-w_main']
            ci_main = stringlist_2_list(row['hr-w-CI_main'])

            hr2 = row['hr-w_aux']
            ci2 = stringlist_2_list(row['hr-w-CI_aux'])
            p2 = row['hr-w-p_aux']

            hr2_main = row['hr-w_main_aux']
            ci2_main = stringlist_2_list(row['hr-w-CI_main_aux'])

            nabs = row['no. pasc in +']
            ncum = stringlist_2_list(row['cif_1_w'])[-1] * 1000
            ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]

            ncum_neg = stringlist_2_list(row['cif_0_w'])[-1] * 1000


            nabs2 = row['no. pasc in +_aux']
            ncum2 = stringlist_2_list(row['cif_1_w_aux'])[-1] * 1000
            ncum_ci2 = [stringlist_2_list(row['cif_1_w_CILower_aux'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper_aux'])[-1] * 1000]

            ncum_neg2 = stringlist_2_list(row['cif_0_w_aux'])[-1] * 1000

            # if star:
            #     if (p > 0.05 / 137) and (p <= 0.01):
            #         name += '**'
            #     elif (p > 0.01) and (p <= 0.05):
            #         name += '*'

            if (row['selected'] == 1) and (row['selected_aux'] == 1):
                name += r'$^{‡}$'


            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain]
                pasc_row2 = [name, hr2, ci2, p2, domain]
                continue

            if domain == organ:
                organ_n[i] += 2
                # if len(name.split()) == 4:
                #     name = ' '.join(name.split()[:2]) + '\n' + ' '.join(name.split()[2:])
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                labs.append(name)
                labs.append('')
                measure.append(hr)
                measure.append(hr2)
                lower.append(ci[0])
                upper.append(ci[1])
                lower.append(ci2[0])
                upper.append(ci2[1])
                pval.append(p)
                pval.append(p2)
                color_list.append('#ed6766')
                color_list.append('#A986B5')  # '#A986B5')

                # nabsv.append(ncum_neg)  # nabsv.append(nabs)
                # ncumv.append(ncum)
                # nabsv.append(ncum_neg2) # nabsv.append(nabs2)
                # ncumv.append(ncum2)
                addcol1.append('{1:.{0}f}'.format(2, hr_main))
                addcol1.append('{1:.{0}f}'.format(2, hr2_main))
                addcol2.append('(' + '{1:.{0}f}'.format(2, ci_main[0]) + ', ' + '{1:.{0}f}'.format(2, ci_main[1]) + ')')
                addcol2.append('(' + '{1:.{0}f}'.format(2, ci2_main[0]) + ', ' + '{1:.{0}f}'.format(2, ci2_main[1]) + ')')

        if len(measure) == 0:
            continue

    # add pasc at last
    # if add_pasc:
    #     if pasc_row:
    #         organ_n[-1] += 2
    #         labs.append(pasc_row[0])
    #         measure.append(pasc_row[1])
    #         lower.append(pasc_row[2][0])
    #         upper.append(pasc_row[2][1])
    #         pval.append(pasc_row[3])
    #         labs.append('')
    #         measure.append(pasc_row2[1])
    #         lower.append(pasc_row2[2][0])
    #         upper.append(pasc_row2[2][1])
    #         pval.append(pasc_row2[3])
    #         color_list.append('#ed6766')
    #         color_list.append('#A986B5')  # '#A986B5')
    #     else:
    #         print('pasc general not found!!!')

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          addcol1=addcol1, addcol2=addcol2)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR', add_label1='aHR\nmain', add_label2='95% CI\nmain') # 'No. of\nCases')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=color_list, pointcolor=color_list)  # , linecolor='#fcb2ab')
    width = 9.
    height = .21 * len(labs)
    if len(labs) == 2:
        height = .3 * (len(labs) + 1)
    ax = p.plot_with_addcols(figsize=(width, height), t_adjuster=0.005, max_value=3, min_value=0.7, size=5, decimal=2)  # 0.02
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.0, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/figure2Compare/{}/'.format(expdir)
    check_and_mkdir(output_dir)
    organ = 'all'
    i = 0
    plt.savefig(output_dir + 'new-trim-{}_hr-p{:.3f}-{}-{}.png'.format(severity, pvalue, select_criteria, expdir),
                bbox_inches='tight',
                dpi=650)
    plt.savefig(output_dir + 'new-trim-{}_hr-p{:.3f}-{}-{}.pdf'.format(severity, pvalue, select_criteria, expdir),
                bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_compare2data_V3_moresensitivity_downsample(
        expdir, add_name=False, severity="all", star=False, select_criteria='', pvalue=0.05 / 596, add_pasc=False):
    print('expdir', expdir)
    # down sample from insight
    # oneflorida the same as the primary analysis
    if severity == 'all':
        df1_main = pd.read_excel(
            r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')
        df2_main = pd.read_excel(
            r'../data/oneflorida/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_oneflorida-MultiPval-DXMEDALL.xlsx',
            sheet_name='dx')

        df1 = pd.read_csv(
            r'../data/V15_COVID19/output/character/outcome/{}/causal_effects_specific.csv'.format(expdir),
            )
        df2 = pd.read_csv(
            r'../data/oneflorida/output/character/outcome/DX-all-new-trim/causal_effects_specific.csv',
            )
    else:
        raise ValueError


    # add selected feature, name, hr in main analyses
    df1 = pd.merge(df1, df1_main[["pasc", "PASC Name Simple", "Organ Domain",  # "Original CCSR Domain",
                                  'selected', 'selected oneflorida', 'sensitivity',
                                  'hr-w', 'hr-w-CI', 'hr-w-p']],
                   left_on='pasc', right_on='pasc', how='left', suffixes=('', '_main'))
    # df1 = df1.rename(columns={x + '_y': x for x in ["PASC Name Simple", "Organ Domain", "Original CCSR Domain"]})

    df2 = pd.merge(df2, df2_main[["pasc", 'selected', 'hr-w', 'hr-w-CI', 'hr-w-p']],
                   left_on='pasc', right_on='pasc', how='left', suffixes=('', '_main'))
    df_aux = df2.rename(columns=lambda x: x + '_aux')
    df = pd.merge(df1, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index('i')

    # pvalue = 0.05 / 137  # 0.01  #

    if select_criteria == 'insight':
        # print('select_critera:', select_criteria)
        # _df = pd.read_excel(
        #     r'../data/V15_COVID19/output/character/outcome/DX-all-new-trim/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
        #     sheet_name='dx').set_index('i')
        # # pvalue = 0.01  # 0.05 / 137
        # _df_select = _df.sort_values(by='hr-w', ascending=False)
        # # _df_select = _df_select.loc[_df_select['hr-w-p'] <= pvalue, :]  #
        # # _df_select = _df_select.loc[_df_select['hr-w'] > 1, :]
        # # _df_select = _df_select.loc[_df_select['no. pasc in +'] >= 100, :]
        #
        # _df_select = _df_select.loc[_df_select['selected'] == 1, :]
        #
        # print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        #
        # df_select = df.loc[df['pasc'].isin(_df_select['pasc']), :]
        # df_select = df_select.sort_values(by='hr-w', ascending=False)
        pass
    else:
        def select_criteria_func_V3(_df):
            _df_select = _df.sort_values(by='hr-w_main', ascending=False)
            _df_select = _df_select.loc[_df_select['selected'] == 1, :]

            print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
            return _df_select

        # df_select = select_criteria_func(df)
        df_select = select_criteria_func_V3(df)

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
        'General'
    ]
    # 'Certain Infectious and Parasitic Diseases',
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']
    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    pasc_row2 = []
    color_list = []

    # nabsv = []
    # ncumv = []

    addcol1 = []
    addcol2 = []

    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']
            pasc = row['pasc']

            hr_main = row['hr-w_main']
            ci_main = stringlist_2_list(row['hr-w-CI_main'])

            hr2 = row['hr-w_aux']
            ci2 = stringlist_2_list(row['hr-w-CI_aux'])
            p2 = row['hr-w-p_aux']

            hr2_main = row['hr-w_main_aux']
            ci2_main = stringlist_2_list(row['hr-w-CI_main_aux'])

            nabs = row['no. pasc in +']
            ncum = stringlist_2_list(row['cif_1_w'])[-1] * 1000
            ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]

            ncum_neg = stringlist_2_list(row['cif_0_w'])[-1] * 1000


            nabs2 = row['no. pasc in +_aux']
            ncum2 = stringlist_2_list(row['cif_1_w_aux'])[-1] * 1000
            ncum_ci2 = [stringlist_2_list(row['cif_1_w_CILower_aux'])[-1] * 1000,
                       stringlist_2_list(row['cif_1_w_CIUpper_aux'])[-1] * 1000]

            ncum_neg2 = stringlist_2_list(row['cif_0_w_aux'])[-1] * 1000

            # if star:
            #     if (p > 0.05 / 137) and (p <= 0.01):
            #         name += '**'
            #     elif (p > 0.01) and (p <= 0.05):
            #         name += '*'

            if (row['selected'] == 1) and (row['selected_aux'] == 1):
                name += r'$^{‡}$'


            if pasc == 'PASC-General':
                pasc_row = [name, hr, ci, p, domain]
                pasc_row2 = [name, hr2, ci2, p2, domain]
                continue

            if domain == organ:
                organ_n[i] += 2
                # if len(name.split()) == 4:
                #     name = ' '.join(name.split()[:2]) + '\n' + ' '.join(name.split()[2:])
                if len(name.split()) >= 5:
                    name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])

                labs.append(name)
                labs.append('')
                measure.append(hr)
                measure.append(hr2)
                lower.append(ci[0])
                upper.append(ci[1])
                lower.append(ci2[0])
                upper.append(ci2[1])
                pval.append(p)
                pval.append(p2)
                color_list.append('#ed6766')
                color_list.append('#A986B5')  # '#A986B5')

                # nabsv.append(ncum_neg)  # nabsv.append(nabs)
                # ncumv.append(ncum)
                # nabsv.append(ncum_neg2) # nabsv.append(nabs2)
                # ncumv.append(ncum2)
                addcol1.append('{1:.{0}f}'.format(2, hr_main))
                addcol1.append('{1:.{0}f}'.format(2, hr2_main))
                addcol2.append('(' + '{1:.{0}f}'.format(2, ci_main[0]) + ', ' + '{1:.{0}f}'.format(2, ci_main[1]) + ')')
                addcol2.append('(' + '{1:.{0}f}'.format(2, ci2_main[0]) + ', ' + '{1:.{0}f}'.format(2, ci2_main[1]) + ')')

        if len(measure) == 0:
            continue

    # add pasc at last
    # if add_pasc:
    #     if pasc_row:
    #         organ_n[-1] += 2
    #         labs.append(pasc_row[0])
    #         measure.append(pasc_row[1])
    #         lower.append(pasc_row[2][0])
    #         upper.append(pasc_row[2][1])
    #         pval.append(pasc_row[3])
    #         labs.append('')
    #         measure.append(pasc_row2[1])
    #         lower.append(pasc_row2[2][0])
    #         upper.append(pasc_row2[2][1])
    #         pval.append(pasc_row2[3])
    #         color_list.append('#ed6766')
    #         color_list.append('#A986B5')  # '#A986B5')
    #     else:
    #         print('pasc general not found!!!')

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          addcol1=addcol1, addcol2=addcol2)
    p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR', add_label1='aHR\nmain', add_label2='95% CI\nmain') # 'No. of\nCases')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="s", errorbarcolor=color_list, pointcolor=color_list)  # , linecolor='#fcb2ab')
    width = 9.
    height = .21 * len(labs)
    if len(labs) == 2:
        height = .3 * (len(labs) + 1)
    ax = p.plot_with_addcols(figsize=(width, height), t_adjuster=0.005, max_value=3, min_value=0.7, size=5, decimal=2)  # 0.02
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.0, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/figure/figure2Compare/{}/'.format(expdir)
    check_and_mkdir(output_dir)
    organ = 'all'
    i = 0
    plt.savefig(output_dir + 'new-trim-{}_hr-p{:.3f}-{}-{}.png'.format(severity, pvalue, select_criteria, expdir),
                bbox_inches='tight',
                dpi=650)
    plt.savefig(output_dir + 'new-trim-{}_hr-p{:.3f}-{}-{}.pdf'.format(severity, pvalue, select_criteria, expdir),
                bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()

def combine_pasc_list():
    df_icd = pd.read_csv('../data/V15_COVID19/output/character/pcr_cohorts_ICD_cnts_followup-ALL.csv')
    print('df_icd.shape:', df_icd.shape)
    df_pasc_list = pd.read_excel(r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx',
                                 sheet_name=r'PASC Screening List',
                                 usecols="A:N")
    print('df_pasc_list.shape', df_pasc_list.shape)

    df_icd_combined = pd.merge(df_icd, df_pasc_list, left_on='ICD', right_on='ICD-10-CM Code', how='left')
    df_icd_combined.to_csv('../data/V15_COVID19/output/character/pcr_cohorts_ICD_cnts_followup-ALL-combined_PASC.csv')

    df_pasc_combined = pd.merge(df_pasc_list, df_icd, left_on='ICD-10-CM Code', right_on='ICD', how='left')
    df_pasc_combined.to_csv(
        '../data/V15_COVID19/output/character/PASC_Adult_Combined_List_20220127_v3_combined_RWD.csv')


def add_pasc_domain_to_causal_results():
    df = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/DX/causal_effects_specific.csv')
    print('df.shape:', df.shape)
    df_pasc = pd.read_excel(r'../data/mapping/PASC_Adult_Combined_List_20220127_v3.xlsx',
                            sheet_name=r'PASC Screening List',
                            usecols="A:N")
    print('df_pasc.shape', df_pasc.shape)
    pasc_domain = defaultdict(set)
    for key, row in df_pasc.iterrows():
        hd_domain = row['HD Domain (Defined by Nature paper)']
        # ccsr_code = row['CCSR CATEGORY 1']
        pasc = row['CCSR CATEGORY 1 DESCRIPTION']
        # icd = row['ICD-10-CM Code']
        # icd_name = row['ICD-10-CM Code Description']
        if pd.notna(hd_domain):
            pasc_domain[pasc].add(hd_domain)

    pasc_domain['Anemia'].add(
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism')
    pasc_domain['PASC-General'].add('Certain Infectious and Parasitic Diseases')

    df['ccsr domain'] = df['pasc'].apply(lambda x: '$'.join(pasc_domain.get(x, [])))
    df.to_csv(r'../data/V15_COVID19/output/character/outcome/DX/causal_effects_specific_withDomain.csv')

    return pasc_domain


def add_drug_name():
    rxing_index = utils.load(r'../data/mapping/selected_rxnorm_index.pkl')

    df = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/MED/causal_effects_specific_med.csv')
    df_name = pd.read_excel(
        r'../data/V15_COVID19/output/character/info_medication_cohorts_covid_4manuNegNoCovid_ALL_enriched_round3.xlsx',
        dtype={'rxnorm': str})

    df_combined = pd.merge(df, df_name, left_on='pasc', right_on='rxnorm', how='left')
    df_combined.to_csv(r'../data/V15_COVID19/output/character/outcome/MED/causal_effects_specific_med-withName.csv')
    return df_combined


def add_drug_previous_label():
    df = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/MED/causal_effects_specific_Medication-withName-simplev2-multitest-withMultiPval-DXMEDALL.xlsx',
        sheet_name='Sheet1'
    )

    df2 = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/DX-all/causal_effects_specific_withMedication_v3.xlsx',
        sheet_name='med'
    )

    dfm = pd.merge(df, df2[['pasc', 'PASC Name Simple', 'selected', 'Organ Domain', 'Organ Domain-old']],
                   how='left', left_on='pasc', right_on='pasc')

    dfm.to_excel(
        r'../data/V15_COVID19/output/character/outcome/MED/causal_effects_specific_Medication-withName-simplev2-multitest-withMultiPval-DXMEDALL-aux.xlsx',
    )


if __name__ == '__main__':
    # plot_forest_for_dx()
    # plot_forest_for_med()
    # combine_pasc_list()
    # pasc_domain = add_pasc_domain_to_causal_results()
    # df_med = add_drug_name()
    # plot_forest_for_dx_organ_V2(pvalue=0.05 / 137, text_right=False)
    # plot_forest_for_dx_organ_V2(pvalue=0.05 / 137, text_right=True)
    # plot_forest_for_dx_organ_V2(pvalue=0.05 / 137, pasc_dx=True)
    # plot_forest_for_dx_organ(pvalue=0.05/137)
    # plot_forest_for_dx_organ(pvalue=0.05)
    # plot_forest_for_med_organ(pvalue=0.05/459)
    # plot_forest_for_med_organ(pvalue=0.05 / 459, star=True, datasite='oneflorida')
    # plot_forest_for_med_organ_V2(pvalue=0.05/459)

    # plot_forest_for_dx_organ_V4(star=False, pasc_dx=False, text_right=False)
    # plot_forest_for_med_organ_V3(database='V15_COVID19')

    # plot_forest_for_med_organ_V3(database='oneflorida')
    plot_forest_for_med_organ_compare2data(add_name=False, severity="all", star=False, select_criteria='',
                                           pvalue=0.05 / 596, add_pasc=False)
    zz

    plot_forest_for_dx_organ_compare2data_V3(add_name=False, severity='all')
    # plot_forest_for_dx_organ_compare2data_sensitivity(add_name=False, severity='all')

    # 2023-1-24 revision 2
    expdir = r'DX-all-new-trim-nonlinear'  # r'DX-all-new-trim-spline' # r'DX-all-new-trim-vaccine'
    plot_forest_for_dx_organ_compare2data_V3_moresensitivity(expdir, add_name=False, severity='all')

    # expdir = r'DX-all-new-trim-downsample'  # r'DX-all-new-trim-vaccine'
    # plot_forest_for_dx_organ_compare2data_V3_moresensitivity_downsample(expdir, add_name=False, severity='all')

    # add_drug_previous_label()

    #
    # plot_forest_for_dx_organ_compare2data_V2(add_name=False, severity='all')
    # plot_forest_for_dx_organ_compare2data(add_name=False, severity='all', pvalue=0.01)
    # plot_forest_for_dx_organ_compare2data(add_name=False, severity='all', pvalue=0.05)
    # plot_forest_for_dx_organ_compare2data(add_name=True, severity='less65', select_criteria='insight')
    # plot_forest_for_dx_organ_compare2data(add_name=True, severity='above65', select_criteria='insight')
    # plot_forest_for_dx_organ_compare2data(add_name=True, severity='65to75', select_criteria='insight')
    # plot_forest_for_dx_organ_compare2data(add_name=True, severity='less65')
    # plot_forest_for_dx_organ_compare2data(add_name=True, severity='above65')
    # plot_forest_for_dx_organ_compare2data(add_name=True, severity='65to75')

    # plot_forest_for_dx_organ_compare3data_V2()

    print('Done!')
