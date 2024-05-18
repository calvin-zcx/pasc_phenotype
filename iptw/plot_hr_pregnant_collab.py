import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import re
import forestplot as fp

import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd
from misc.utils import check_and_mkdir, stringlist_2_str, stringlist_2_list, pformat_symbol
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


def load_pasc_info():
    print('load and preprocess PASC info')
    df_pasc_info = pd.read_excel(r'../prediction/output/causal_effects_specific_withMedication_v3.xlsx',
                                 sheet_name='diagnosis')
    addedPASC_encoding = utils.load(r'../data/mapping/addedPASC_index_mapping.pkl')
    addedPASC_list = list(addedPASC_encoding.keys())
    brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
    brainfog_list = list(brainfog_encoding.keys())

    CFR_encoding = utils.load(r'../data/mapping/cognitive-fatigue-respiratory_index_mapping.pkl')
    CFR_list = list(CFR_encoding.keys())

    pasc_simname = {}
    pasc_organ = {}
    for index, rows in df_pasc_info.iterrows():
        pasc_simname[rows['pasc']] = (rows['PASC Name Simple'], rows['Organ Domain'])
        pasc_organ[rows['pasc']] = rows['Organ Domain']

    for p in addedPASC_list:
        pasc_simname[p] = (p, 'General-add')
        pasc_organ[p] = 'General-add'

    for p in brainfog_list:
        pasc_simname[p] = (p, 'brainfog')
        pasc_organ[p] = 'brainfog'

    for p in CFR_list:
        pasc_simname[p] = (p, 'cognitive-fatigue-respiratory')
        pasc_organ[p] = 'cognitive-fatigue-respiratory'

    pasc_simname['any_pasc'] = ('Long COVID', 'Any PASC')
    pasc_simname['smell and taste'] = ('smell and taste', 'General')

    pasc_simname['death'] = ('Death Overall', 'Death')
    pasc_simname['death_acute'] = ('Acute Death', 'Death')
    pasc_simname['death_postacute'] = ('Post Acute Death', 'Death')

    pasc_simname['any_CFR'] = ('Any CFR', 'Any CFR')
    pasc_simname['hospitalization_postacute'] = ('Hospitalization', 'Hospitalization')

    # add n3c
    pasc_simname['CP PASC-N3C'] = ('Long COVID (N3C)', 'Any PASC')
    pasc_simname['U09/B94-N3C'] = ('U099/B948 (N3C)', 'General')
    pasc_simname['Cognitive-N3C'] = ('Cognitive (N3C)', 'cognitive-fatigue-respiratory')
    pasc_simname['Fatigue-N3C'] = ('Fatigue (N3C)', 'cognitive-fatigue-respiratory')
    pasc_simname['Respiratory-N3C'] = ('Respiratory (N3C)', 'cognitive-fatigue-respiratory')
    pasc_simname['Any_secondary-N3C'] = ('Any CFR (N3C)', 'Any CFR')

    pasc_simname['CP PASC-Trim1-N3C'] = ('Long COVID-Trim1 (N3C)', 'sensitivity')
    pasc_simname['CP PASC-Trim2-N3C'] = ('Long COVID-Trim2 (N3C)', 'sensitivity')
    pasc_simname['CP PASC-Trim3-N3C'] = ('Long COVID-Trim3 (N3C)', 'sensitivity')

    return pasc_simname


def plot_forest_for_dx_organ_preg(star=True, text_right=False):
    # indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k10/'
    # indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k5-followupanydx/'
    # indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-followupanydx/'
    # indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-trimester1/'
    # indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-trimester3/'
    # indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-delivery1week/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-delivery1week-anyfollow/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-trimester1-anyfollow/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-trimester2-anyfollow/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-trimester3-anyfollow/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-inpatienticu/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-outpatient/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-inpatienticu-anyfollow/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3-outpatient-anyfollow/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1-delivery1week/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1-trimester1/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1-trimester2/'
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1-trimester3/'

    output_dir = indir + r'figure/'

    df = pd.read_csv(indir + 'causal_effects_specific.csv')
    df.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )

    pasc_simname_organ = load_pasc_info()

    df.insert(df.columns.get_loc('pasc') + 1, 'Organ Domain', '')
    df.insert(df.columns.get_loc('pasc') + 1, 'PASC Name Simple', '')

    for key, row in df.iterrows():
        pasc = row['pasc']
        if pasc in pasc_simname_organ:
            df.loc[key, 'PASC Name Simple'] = pasc_simname_organ[pasc][0]
            df.loc[key, 'Organ Domain'] = pasc_simname_organ[pasc][1]

    df_select = df.sort_values(by='hr-w', ascending=True)
    # df_select = df_select.loc[df_select['selected'] == 1, :]  #
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Any PASC',
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
        'General',
        # 'General-add',
        # 'brainfog',
        'cognitive-fatigue-respiratory'
    ]
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
            print(name, pasc)
            if pasc.startswith('smm:'):
                continue
            hr = row['hr-w']
            if pd.notna(row['hr-w-CI']):
                ci = stringlist_2_list(row['hr-w-CI'])
            else:
                ci = [np.nan, np.nan]

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

            # if (database == 'V15_COVID19') and (row['selected'] == 1) and (row['selected oneflorida'] == 1):
            #     name += r'$^{‡}$'
            #
            # if (database == 'oneflorida') and (row['selected'] == 1) and (row['selected insight'] == 1):
            #     name += r'$^{‡}$'

            # if pasc == 'PASC-General':
            #     pasc_row = [name, hr, ci, p, domain, nabs, ncum]
            #     continue
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

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          nabs=nabsv, ncumIncidence=ncumv)
    p.labels(scale='log')

    # organ = 'ALL'
    # p.labels(effectmeasure='aHR', add_label1='CIF per\n1000', add_label2='No. of\nCases')  # aHR
    p.labels(effectmeasure='aHR', add_label1='CIF per\n1000\nin case', add_label2='CIF per\n1000\nin Ctrl')

    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']

    c = '#F65453'
    p.colors(pointshape="o", errorbarcolor=c, pointcolor=c)  # , linecolor='black'),   # , linecolor='#fcb2ab')
    ax = p.plot_with_incidence(figsize=(9, .47 * len(labs)), t_adjuster=0.0108, max_value=3.5, min_value=0.1,
                               size=5, decimal=2,
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

    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'hr_2CIF-V5.png', bbox_inches='tight', dpi=600)

    plt.savefig(output_dir + 'hr_2CIF-V5.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_preg_v2(star=True, text_right=False):
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1-V2/'

    output_dir = indir + r'figure/'

    df = pd.read_csv(indir + 'causal_effects_specific.csv')
    df.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )

    pasc_simname_organ = load_pasc_info()

    df.insert(df.columns.get_loc('pasc') + 1, 'Organ Domain', '')
    df.insert(df.columns.get_loc('pasc') + 1, 'PASC Name Simple', '')

    for key, row in df.iterrows():
        pasc = row['pasc']
        if pasc in pasc_simname_organ:
            df.loc[key, 'PASC Name Simple'] = pasc_simname_organ[pasc][0]
            df.loc[key, 'Organ Domain'] = pasc_simname_organ[pasc][1]

    df_select = df.sort_values(by='hr-w', ascending=True)
    # df_select = df_select.loc[df_select['selected'] == 1, :]  #
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Any PASC',
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
        'General',
        # 'General-add',
        # 'brainfog',
        'Any CFR',
        'cognitive-fatigue-respiratory',
        'Death'
    ]
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
            print(name, pasc)
            if pasc.startswith('smm:'):
                continue
            hr = row['hr-w']
            if pd.notna(row['hr-w-CI']):
                ci = stringlist_2_list(row['hr-w-CI'])
            else:
                ci = [np.nan, np.nan]

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

            # if (database == 'V15_COVID19') and (row['selected'] == 1) and (row['selected oneflorida'] == 1):
            #     name += r'$^{‡}$'
            #
            # if (database == 'oneflorida') and (row['selected'] == 1) and (row['selected insight'] == 1):
            #     name += r'$^{‡}$'

            # if pasc == 'PASC-General':
            #     pasc_row = [name, hr, ci, p, domain, nabs, ncum]
            #     continue
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

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
                          nabs=nabsv, ncumIncidence=ncumv)
    p.labels(scale='log')

    # organ = 'ALL'
    # p.labels(effectmeasure='aHR', add_label1='CIF per\n1000', add_label2='No. of\nCases')  # aHR
    p.labels(effectmeasure='aHR', add_label1='CIF per\n1000\nin case', add_label2='CIF per\n1000\nin Ctrl')

    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']

    c = '#F65453'
    p.colors(pointshape="o", errorbarcolor=c, pointcolor=c)  # , linecolor='black'),   # , linecolor='#fcb2ab')
    ax = p.plot_with_incidence(figsize=(10, .47 * len(labs)), t_adjuster=0.0108, max_value=3.5, min_value=0.1,
                               size=5, decimal=2,
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

    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'hr_2CIF-V5.png', bbox_inches='tight', dpi=600)

    plt.savefig(output_dir + 'hr_2CIF-V5.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_preg_lib2(show='full'):
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1-V3/'
    output_dir = indir + r'figure/'

    df = pd.read_csv(indir + 'causal_effects_specific.csv')
    df.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )

    pasc_simname_organ = load_pasc_info()

    df.insert(df.columns.get_loc('pasc') + 1, 'Organ Domain', '')
    df.insert(df.columns.get_loc('pasc') + 1, 'PASC Name Simple', '')

    for key, row in df.iterrows():
        pasc = row['pasc']
        if pasc in pasc_simname_organ:
            df.loc[key, 'PASC Name Simple'] = pasc_simname_organ[pasc][0]
            df.loc[key, 'Organ Domain'] = pasc_simname_organ[pasc][1]

    df_select = df.sort_values(by='hr-w', ascending=True)
    # df_select = df_select.loc[df_select['selected'] == 1, :]  #
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Any PASC',
        # 'Death',
        # 'Hospitalization',
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
        'General',
        # 'General-add',
        # 'brainfog',
        'Any CFR',
        'cognitive-fatigue-respiratory',
    ]
    organ_mapname = {
        'Any PASC': 'Overall',
        'Death': 'Overall',
        'Hospitalization': 'Overall',
        'Diseases of the Nervous System': 'Neurologic',
        'Diseases of the Skin and Subcutaneous Tissue': 'Skin',
        'Diseases of the Respiratory System': 'Pulmonary',
        'Diseases of the Circulatory System': 'Circulatory',
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism': 'Blood',
        'Endocrine, Nutritional and Metabolic Diseases': 'Metabolic',
        'Diseases of the Digestive System': 'Digestive',
        'Diseases of the Genitourinary System': 'Genitourinary',
        'Diseases of the Musculoskeletal System and Connective Tissue': 'Musculoskeletal',
        # 'Certain Infectious and Parasitic Diseases',
        'General': 'General',
        # 'General-add',
        # 'brainfog':'Brain Fog',
        'Any CFR': 'CFR Overall',
        'cognitive-fatigue-respiratory': 'CFR Individuals',
    }
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']

    organ_n = np.zeros(len(organ_list))
    results_list = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)
        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            # if len(name.split()) >= 5:
            #     name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])
            if name == 'Dyspnea':
                name = 'Shortness of breath'
            elif name == 'Death Overall':
                continue
            elif name == 'Abnormal heartbeat':
                name = 'Dysrhythmia'
            elif name == 'Diabetes mellitus':
                name = 'Diabetes'

            pasc = row['pasc']
            print(name, pasc)
            hr = row['hr-w']
            if (hr <= 0.001) or pd.isna(hr):
                print('HR, ', hr, 'for ', name, pasc)
                continue

            if pd.notna(row['hr-w-CI']):
                ci = stringlist_2_list(row['hr-w-CI'])
            else:
                ci = [np.nan, np.nan]
            p = row['hr-w-p']

            ahr_pformat, ahr_psym = pformat_symbol(p)

            domain = row['Organ Domain']
            cif1 = stringlist_2_list(row['cif_1_w'])[-1] * 100
            cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 100,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 100]

            # use nabs for ncum_ci_negative
            cif0 = stringlist_2_list(row['cif_0_w'])[-1] * 100
            cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1] * 100,
                       stringlist_2_list(row['cif_0_w_CIUpper'])[-1] * 100]

            cif_diff = stringlist_2_list(row['cif-w-diff-2'])[-1] * 100
            cif_diff_ci = [stringlist_2_list(row['cif-w-diff-CILower'])[-1] * 100,
                           stringlist_2_list(row['cif-w-diff-CIUpper'])[-1] * 100]
            cif_diff_p = stringlist_2_list(row['cif-w-diff-p'])[-1]
            cif_diff_pformat, cif_diff_psym = pformat_symbol(cif_diff_p)

            result = [name, pasc, organ_mapname[organ],
                      hr, '{:.2f} ({:.2f}, {:.2f})'.format(hr, ci[0], ci[1]), p,
                      '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
                      '({:.2f},{:.2f})'.format(ci[0], ci[1]),
                      cif1, cif1_ci[0], cif1_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif1, cif1_ci[0], cif1_ci[1]),
                      cif0, cif0_ci[0], cif0_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif0, cif0_ci[0], cif0_ci[1]),
                      ahr_pformat + ahr_psym, ahr_psym,
                      cif_diff, '{:.2f} ({:.2f}, {:.2f})'.format(cif_diff, cif_diff_ci[0], cif_diff_ci[1]), cif_diff_p,
                      '{:.2f}'.format(cif_diff_ci[0]), '{:.2f}'.format(cif_diff_ci[1]),
                      '({:.2f},{:.2f})'.format(cif_diff_ci[0], cif_diff_ci[1]),
                      cif_diff_pformat + cif_diff_psym, cif_diff_psym
                      ]

            if domain == organ:
                results_list.append(result)

    df_result = pd.DataFrame(results_list,
                             columns=['name', 'pasc', 'group',
                                      'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
                                      'aHR-CI-str',
                                      'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
                                      'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
                                      'p-val-sci', 'sigsym',
                                      'cif_diff', 'cif_diff-str', 'cif_diff-p',
                                      'cif_diff_cilower', 'cif_diff_ciupper', 'cif_diff-CI-str',
                                      'cif_diff-p-format', 'cif_diff-p-symbol'])
    # df_result['-aHR'] = -1 * df_result['aHR']

    df_result = df_result.loc[~df_result['aHR'].isna()]
    print(df_result)
    plt.rc('font', family='serif')
    if show == 'full':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       ]

        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100', 'P-Value',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in Pregnant', 'CIF/100 in Ctrl']

    elif show == 'short':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       ]
        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100 (95% CI)', 'P-Value',
                               ]
        leftannote = []
        left_annoteheaders = []
    elif show == 'full-nopval':
        rightannote = ["aHR-str",
                       'cif_diff-str',
                       ]
        right_annoteheaders = ["HR (95% CI)",
                               'DIFF/100',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in Pregnant', 'CIF/100 in Ctrl']

    # fig, ax = plt.subplots()
    axs = fp.forestplot(
        df_result,  # the dataframe with results data
        figsize=(5, 12),
        estimate="aHR",  # col containing estimated effect size
        ll='aHR-lb',
        hl='aHR-ub',  # lower & higher limits of conf. int.
        varlabel="name",  # column containing the varlabels to be printed on far left
        # capitalize="capitalize",  # Capitalize labels
        pval="p-val",  # column containing p-values to be formatted
        starpval=True,
        annote=leftannote,  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        annoteheaders=left_annoteheaders,  # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
        rightannote=rightannote,
        # p_format, columns to report on right of plot
        right_annoteheaders=right_annoteheaders,  # p_format, ^corresponding headers
        groupvar="group",  # column containing group labels
        group_order=df_result['group'].unique(),
        xlabel="Hazard Ratio",  # x-label title
        xticks=[0.1, 1, 2],  # x-ticks to be printed
        color_alt_rows=True,
        # flush=True,
        sort=True,  # sort estimates in ascending order
        # sortby='-aHR',
        # table=True,  # Format as a table
        # logscale=True,
        # Additional kwargs for customizations
        **{
            # 'fontfamily': 'sans-serif',  # 'sans-serif'
            "marker": "D",  # set maker symbol as diamond
            "markersize": 35,  # adjust marker size
            "xlinestyle": (0., (10, 5)),  # long dash for x-reference line
            "xlinecolor": ".1",  # gray color for x-reference line
            "xtick_size": 12,  # adjust x-ticker fontsize
            # 'fontfamily': 'sans-serif',  # 'sans-serif'
        },
    )
    axs.axvline(x=1, ymin=0, ymax=0.95, color='grey', linestyle='dashed')
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'hr_moretabs-{}.png'.format(show), bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_moretabs-{}.pdf'.format(show), bbox_inches='tight', transparent=True)

    print('Done')
    return df_result


def plot_forest_for_dx_organ_preg_lib2_with_N3C(show='full'):
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1-V3/'
    output_dir = indir + r'figure_2cohorts/'
    df_1 = pd.read_csv(indir + 'causal_effects_specific.csv')
    df_1.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )

    # load n3c results, should edit to our format
    df_2 = pd.read_csv(r'../data/recover/output/pregnancy_output/N3C/24_04_24 Combined Causal Inference.csv')

    df = pd.concat([df_1, df_2], ignore_index=True, sort=False)

    pasc_simname_organ = load_pasc_info()

    df.insert(df.columns.get_loc('pasc') + 1, 'Organ Domain', '')
    df.insert(df.columns.get_loc('pasc') + 1, 'PASC Name Simple', '')

    for key, row in df.iterrows():
        pasc = row['pasc']
        if pasc in pasc_simname_organ:
            df.loc[key, 'PASC Name Simple'] = pasc_simname_organ[pasc][0]
            df.loc[key, 'Organ Domain'] = pasc_simname_organ[pasc][1]

    # df_select = df.sort_values(by='hr-w', ascending=True)
    df_select = df.sort_values(by='PASC Name Simple', ascending=True)
    # df_select = df_select.loc[df_select['selected'] == 1, :]  #
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Any PASC',
        # 'Death',
        # 'Hospitalization',
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
        'General',
        # 'General-add',
        # 'brainfog',
        'Any CFR',
        'cognitive-fatigue-respiratory',
    ]
    organ_mapname = {
        'Any PASC': 'Overall',
        'Death': 'Overall',
        'Hospitalization': 'Overall',
        'Diseases of the Nervous System': 'Neurologic',
        'Diseases of the Skin and Subcutaneous Tissue': 'Skin',
        'Diseases of the Respiratory System': 'Pulmonary',
        'Diseases of the Circulatory System': 'Circulatory',
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism': 'Blood',
        'Endocrine, Nutritional and Metabolic Diseases': 'Metabolic',
        'Diseases of the Digestive System': 'Digestive',
        'Diseases of the Genitourinary System': 'Genitourinary',
        'Diseases of the Musculoskeletal System and Connective Tissue': 'Musculoskeletal',
        # 'Certain Infectious and Parasitic Diseases',
        'General': 'General',
        # 'General-add',
        # 'brainfog':'Brain Fog',
        'Any CFR': 'CFR Overall',
        'cognitive-fatigue-respiratory': 'CFR Individuals',
    }
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']

    organ_n = np.zeros(len(organ_list))
    results_list = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)
        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            # if len(name.split()) >= 5:
            #     name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])
            if name == 'Dyspnea':
                name = 'Shortness of breath'
            elif name == 'Death Overall':
                continue
            elif name == 'Abnormal heartbeat':
                name = 'Dysrhythmia'
            elif name == 'Diabetes mellitus':
                name = 'Diabetes'

            pasc = row['pasc']
            print(name, pasc)
            hr = row['hr-w']
            if (hr <= 0.001) or pd.isna(hr):
                print('HR, ', hr, 'for ', name, pasc)
                continue

            if pd.notna(row['hr-w-CI']):
                ci = stringlist_2_list(row['hr-w-CI'])
            else:
                ci = [np.nan, np.nan]
            p = row['hr-w-p']

            ahr_pformat, ahr_psym = pformat_symbol(p)

            domain = row['Organ Domain']
            cif1 = stringlist_2_list(row['cif_1_w'])[-1] * 100
            cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 100,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 100]

            # use nabs for ncum_ci_negative
            cif0 = stringlist_2_list(row['cif_0_w'])[-1] * 100
            cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1] * 100,
                       stringlist_2_list(row['cif_0_w_CIUpper'])[-1] * 100]

            cif_diff = stringlist_2_list(row['cif-w-diff-2'])[-1] * 100
            cif_diff_ci = [stringlist_2_list(row['cif-w-diff-CILower'])[-1] * 100,
                           stringlist_2_list(row['cif-w-diff-CIUpper'])[-1] * 100]
            cif_diff_p = stringlist_2_list(row['cif-w-diff-p'])[-1]
            cif_diff_pformat, cif_diff_psym = pformat_symbol(cif_diff_p)

            result = [name, pasc, organ_mapname[organ],
                      hr, '{:.2f} ({:.2f}, {:.2f})'.format(hr, ci[0], ci[1]), p,
                      '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
                      '({:.2f},{:.2f})'.format(ci[0], ci[1]),
                      cif1, cif1_ci[0], cif1_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif1, cif1_ci[0], cif1_ci[1]),
                      cif0, cif0_ci[0], cif0_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif0, cif0_ci[0], cif0_ci[1]),
                      ahr_pformat + ahr_psym, ahr_psym,
                      cif_diff, '{:.2f} ({:.2f}, {:.2f})'.format(cif_diff, cif_diff_ci[0], cif_diff_ci[1]), cif_diff_p,
                      '{:.2f}'.format(cif_diff_ci[0]), '{:.2f}'.format(cif_diff_ci[1]),
                      '({:.2f},{:.2f})'.format(cif_diff_ci[0], cif_diff_ci[1]),
                      cif_diff_pformat + cif_diff_psym, cif_diff_psym
                      ]

            if domain == organ:
                results_list.append(result)

    df_result = pd.DataFrame(results_list,
                             columns=['name', 'pasc', 'group',
                                      'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
                                      'aHR-CI-str',
                                      'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
                                      'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
                                      'p-val-sci', 'sigsym',
                                      'cif_diff', 'cif_diff-str', 'cif_diff-p',
                                      'cif_diff_cilower', 'cif_diff_ciupper', 'cif_diff-CI-str',
                                      'cif_diff-p-format', 'cif_diff-p-symbol'])
    # df_result['-aHR'] = -1 * df_result['aHR']

    df_result = df_result.loc[~df_result['aHR'].isna()]
    print(df_result)
    plt.rc('font', family='serif')
    if show == 'full':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       ]

        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100', 'P-Value',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in Pregnant', 'CIF/100 in Ctrl']

    elif show == 'short':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       ]
        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100 (95% CI)', 'P-Value',
                               ]
        leftannote = []
        left_annoteheaders = []
    elif show == 'full-nopval':
        rightannote = ["aHR-str",
                       'cif_diff-str',
                       ]
        right_annoteheaders = ["HR (95% CI)",
                               'DIFF/100',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in Pregnant', 'CIF/100 in Ctrl']

    # fig, ax = plt.subplots()
    axs = fp.forestplot(
        df_result,  # the dataframe with results data
        figsize=(5, 12),
        estimate="aHR",  # col containing estimated effect size
        ll='aHR-lb',
        hl='aHR-ub',  # lower & higher limits of conf. int.
        varlabel="name",  # column containing the varlabels to be printed on far left
        # capitalize="capitalize",  # Capitalize labels
        pval="p-val",  # column containing p-values to be formatted
        starpval=True,
        annote=leftannote,  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        annoteheaders=left_annoteheaders,  # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
        rightannote=rightannote,
        # p_format, columns to report on right of plot
        right_annoteheaders=right_annoteheaders,  # p_format, ^corresponding headers
        groupvar="group",  # column containing group labels
        group_order=df_result['group'].unique(),
        xlabel="Hazard Ratio",  # x-label title
        xticks=[0.1, 1, 2],  # x-ticks to be printed
        color_alt_rows=True,
        # flush=True,
        sort=False, #True,  # sort estimates in ascending order
        # sortby='-aHR',
        # table=True,  # Format as a table
        # logscale=True,
        # Additional kwargs for customizations
        **{
            # 'fontfamily': 'sans-serif',  # 'sans-serif'
            "marker": "D",  # set maker symbol as diamond
            "markersize": 35,  # adjust marker size
            "xlinestyle": (0., (10, 5)),  # long dash for x-reference line
            "xlinecolor": ".1",  # gray color for x-reference line
            "xtick_size": 12,  # adjust x-ticker fontsize
            # 'fontfamily': 'sans-serif',  # 'sans-serif'
        },
    )
    axs.axvline(x=1, ymin=0, ymax=0.95, color='grey', linestyle='dashed')
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'hr_moretabs-{}.png'.format(show), bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_moretabs-{}.pdf'.format(show), bbox_inches='tight', transparent=True)

    print('Done')
    return df_result


def plot_forest_for_dx_organ_preg_subgroup(show='full', subgroup='trimester1'):
    indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1-V3-{}/'.format(subgroup)
    output_dir = indir + r'figure/'

    df = pd.read_csv(indir + 'causal_effects_specific.csv')
    df.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )

    pasc_simname_organ = load_pasc_info()

    df.insert(df.columns.get_loc('pasc') + 1, 'Organ Domain', '')
    df.insert(df.columns.get_loc('pasc') + 1, 'PASC Name Simple', '')

    for key, row in df.iterrows():
        pasc = row['pasc']
        if pasc in pasc_simname_organ:
            df.loc[key, 'PASC Name Simple'] = pasc_simname_organ[pasc][0]
            df.loc[key, 'Organ Domain'] = pasc_simname_organ[pasc][1]

    df_select = df.sort_values(by='hr-w', ascending=True)
    # df_select = df_select.loc[df_select['selected'] == 1, :]  #
    print('df_select.shape:', df_select.shape)

    organ_list = df_select['Organ Domain'].unique()
    print(organ_list)
    organ_list = [
        'Any PASC',
        # 'Death',
        # 'Hospitalization',
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
        'General',
        # 'General-add',
        # 'brainfog',
        'Any CFR',
        'cognitive-fatigue-respiratory',
    ]
    organ_mapname = {
        'Any PASC': 'Overall',
        'Death': 'Overall',
        'Hospitalization': 'Overall',
        'Diseases of the Nervous System': 'Neurologic',
        'Diseases of the Skin and Subcutaneous Tissue': 'Skin',
        'Diseases of the Respiratory System': 'Pulmonary',
        'Diseases of the Circulatory System': 'Circulatory',
        'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism': 'Blood',
        'Endocrine, Nutritional and Metabolic Diseases': 'Metabolic',
        'Diseases of the Digestive System': 'Digestive',
        'Diseases of the Genitourinary System': 'Genitourinary',
        'Diseases of the Musculoskeletal System and Connective Tissue': 'Musculoskeletal',
        # 'Certain Infectious and Parasitic Diseases',
        'General': 'General',
        # 'General-add',
        # 'brainfog':'Brain Fog',
        'Any CFR': 'CFR Overall',
        'cognitive-fatigue-respiratory': 'CFR Individuals',
    }
    # 'Injury, Poisoning and Certain Other Consequences of External Causes']

    organ_n = np.zeros(len(organ_list))
    results_list = []
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)
        for key, row in df_select.iterrows():
            name = row['PASC Name Simple'].strip('*')
            # if len(name.split()) >= 5:
            #     name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])
            if name == 'Dyspnea':
                name = 'Shortness of breath'
            elif name == 'Death Overall':
                continue
            elif name == 'Abnormal heartbeat':
                name = 'Dysrhythmia'
            elif name == 'Diabetes mellitus':
                name = 'Diabetes'

            pasc = row['pasc']
            print(name, pasc)
            hr = row['hr-w']
            if (hr <= 0.001) or pd.isna(hr):
                print('HR, ', hr, 'for ', name, pasc)
                continue

            if pd.notna(row['hr-w-CI']):
                ci = stringlist_2_list(row['hr-w-CI'])
            else:
                ci = [np.nan, np.nan]
            p = row['hr-w-p']

            ahr_pformat, ahr_psym = pformat_symbol(p)

            domain = row['Organ Domain']
            cif1 = stringlist_2_list(row['cif_1_w'])[-1] * 100
            cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 100,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 100]

            # use nabs for ncum_ci_negative
            cif0 = stringlist_2_list(row['cif_0_w'])[-1] * 100
            cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1] * 100,
                       stringlist_2_list(row['cif_0_w_CIUpper'])[-1] * 100]

            cif_diff = stringlist_2_list(row['cif-w-diff-2'])[-1] * 100
            cif_diff_ci = [stringlist_2_list(row['cif-w-diff-CILower'])[-1] * 100,
                           stringlist_2_list(row['cif-w-diff-CIUpper'])[-1] * 100]
            cif_diff_p = stringlist_2_list(row['cif-w-diff-p'])[-1]
            cif_diff_pformat, cif_diff_psym = pformat_symbol(cif_diff_p)

            result = [name, pasc, organ_mapname[organ],
                      hr, '{:.2f} ({:.2f}, {:.2f})'.format(hr, ci[0], ci[1]), p,
                      '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
                      '({:.2f},{:.2f})'.format(ci[0], ci[1]),
                      cif1, cif1_ci[0], cif1_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif1, cif1_ci[0], cif1_ci[1]),
                      cif0, cif0_ci[0], cif0_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif0, cif0_ci[0], cif0_ci[1]),
                      ahr_pformat + ahr_psym, ahr_psym,
                      cif_diff, '{:.2f} ({:.2f}, {:.2f})'.format(cif_diff, cif_diff_ci[0], cif_diff_ci[1]), cif_diff_p,
                      '{:.2f}'.format(cif_diff_ci[0]), '{:.2f}'.format(cif_diff_ci[1]),
                      '({:.2f},{:.2f})'.format(cif_diff_ci[0], cif_diff_ci[1]),
                      cif_diff_pformat + cif_diff_psym, cif_diff_psym
                      ]

            if domain == organ:
                results_list.append(result)

    df_result = pd.DataFrame(results_list,
                             columns=['name', 'pasc', 'group',
                                      'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
                                      'aHR-CI-str',
                                      'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
                                      'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
                                      'p-val-sci', 'sigsym',
                                      'cif_diff', 'cif_diff-str', 'cif_diff-p',
                                      'cif_diff_cilower', 'cif_diff_ciupper', 'cif_diff-CI-str',
                                      'cif_diff-p-format', 'cif_diff-p-symbol'])
    # df_result['-aHR'] = -1 * df_result['aHR']

    df_result = df_result.loc[~df_result['aHR'].isna()]
    print(df_result)
    plt.rc('font', family='serif')
    if show == 'full':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       ]

        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100', 'P-Value',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in Pregnant', 'CIF/100 in Ctrl']

    elif show == 'short':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       ]
        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100 (95% CI)', 'P-Value',
                               ]
        leftannote = []
        left_annoteheaders = []
    elif show == 'full-nopval':
        rightannote = ["aHR-str",
                       'cif_diff-str',
                       ]
        right_annoteheaders = ["HR (95% CI)",
                               'DIFF/100',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in Pregnant', 'CIF/100 in Ctrl']

    # fig, ax = plt.subplots()
    axs = fp.forestplot(
        df_result,  # the dataframe with results data
        figsize=(5, 12),
        estimate="aHR",  # col containing estimated effect size
        ll='aHR-lb',
        hl='aHR-ub',  # lower & higher limits of conf. int.
        varlabel="name",  # column containing the varlabels to be printed on far left
        # capitalize="capitalize",  # Capitalize labels
        pval="p-val",  # column containing p-values to be formatted
        starpval=True,
        annote=leftannote,  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        annoteheaders=left_annoteheaders,  # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
        rightannote=rightannote,
        # p_format, columns to report on right of plot
        right_annoteheaders=right_annoteheaders,  # p_format, ^corresponding headers
        groupvar="group",  # column containing group labels
        group_order=df_result['group'].unique(),
        xlabel="Hazard Ratio",  # x-label title
        xticks=[0.1, 1, 3.5],  # x-ticks to be printed
        color_alt_rows=True,
        # flush=True,
        sort=True,  # sort estimates in ascending order
        # sortby='-aHR',
        # table=True,  # Format as a table
        # logscale=True,
        # Additional kwargs for customizations
        **{
            # 'fontfamily': 'sans-serif',  # 'sans-serif'
            "marker": "D",  # set maker symbol as diamond
            "markersize": 35,  # adjust marker size
            "xlinestyle": (0., (10, 5)),  # long dash for x-reference line
            "xlinecolor": ".1",  # gray color for x-reference line
            "xtick_size": 12,  # adjust x-ticker fontsize
            # 'fontfamily': 'sans-serif',  # 'sans-serif'
        },
    )
    axs.axvline(x=1, ymin=0, ymax=0.95, color='grey', linestyle='dashed')
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'hr_moretabs-{}.png'.format(show), bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_moretabs-{}.pdf'.format(show), bbox_inches='tight', transparent=True)

    print('Done')
    return df_result

if __name__ == '__main__':
    # plot_forest_for_dx_organ_preg()
    # plot_forest_for_dx_organ_preg_v2()

    # only pcori results
    # plot_forest_for_dx_organ_preg_lib2(show='full')
    # plot_forest_for_dx_organ_preg_lib2(show='full-nopval')
    # plot_forest_for_dx_organ_preg_lib2(show='short')

    # pcori and n3c plots together
    # plot_forest_for_dx_organ_preg_lib2_with_N3C(show='full')
    # plot_forest_for_dx_organ_preg_lib2_with_N3C(show='full-nopval')
    # plot_forest_for_dx_organ_preg_lib2_with_N3C(show='short')


    # subgroup individual
    subgroup = 'delivery1week' #'trimester3'
    plot_forest_for_dx_organ_preg_subgroup(show='full', subgroup=subgroup)
    plot_forest_for_dx_organ_preg_subgroup(show='full-nopval', subgroup=subgroup)
    plot_forest_for_dx_organ_preg_subgroup(show='short', subgroup=subgroup)


    print('Done!')
