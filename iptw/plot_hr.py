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

    # plot_forest_for_dx_organ_V3(star=False, pasc_dx=False, text_right=False)
    add_drug_previous_label()

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
    print('Done!')
