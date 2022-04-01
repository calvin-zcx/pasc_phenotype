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


def plot_forest_for_dx():
    # df = pd.read_csv(r'../data/V15_COVID19/output/character/specificDX/causal_effects_specific-v2.csv')
    df = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/DX/causal_effects_specific.csv')
    df_select = df.sort_values(by='hr-w', ascending=False)
    pvalue = 0.05 / 137
    df_select = df_select.loc[df_select['hr-w-p'] < pvalue, :]  #
    df_select = df_select.loc[df_select['no. pasc in +'] >= 100, :]
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
    ax = p.plot(figsize=(11, 18), t_adjuster=0.0108, max_value=60, min_value=0.3, size=5, decimal=2)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/DX/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'dx_hr_forest-p{}.png'.format(pvalue), bbox_inches='tight', dpi=900)
    plt.savefig(output_dir + 'dx_hr_forest-p{}.pdf'.format(pvalue), bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    # plt.close()


def plot_forest_for_dx_organ():
    # df = pd.read_excel(r'../data/V15_COVID19/output/character/outcome/DX/causal_effects_specific_Diagnosis_withDomain-simple-4plot.xlsx')
    df = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
        sheet_name='diagnosis')

    df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    pvalue = 0.01  # 0.05 / 137
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
    for i, organ in enumerate(organ_list):
        print(i + 1, 'organ', organ)

        for key, row in df_select.iterrows():
            name = row['PASC Name Simple']
            hr = row['Hazard Ratio, Adjusted']
            ci = stringlist_2_list(row['Hazard Ratio, Adjusted, Confidence Interval'])
            p = row['Hazard Ratio, Adjusted, P-Value']
            domain = row['Organ Domain']
            if name == 'General PASC':
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
    p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='#fcb2ab')
    ax = p.plot(figsize=(8, .42 * len(labs)), t_adjuster=0.0108, max_value=3.5, min_value=0.9, size=5, decimal=2)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/DX-all/organ/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'dx_hr_{}-p{}-hrGe1-nGe100.png'.format('all', pvalue), bbox_inches='tight', dpi=900)
    plt.savefig(output_dir + 'dx_hr_{}-p{}-hrGe1-nGe100.pdf'.format('all', pvalue), bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_med_organ():
    # df = pd.read_excel(r'../data/V15_COVID19/output/character/outcome/DX/causal_effects_specific_Diagnosis_withDomain-simple-4plot.xlsx')
    df = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
        sheet_name='med')

    df_select = df.sort_values(by='Hazard Ratio, Adjusted', ascending=False)
    pvalue = 0.05 / 459  # 0.05 / 137
    df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= pvalue, :]  #
    df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    df_select = df_select.loc[df_select['no. pasc in +'] >= 100, :]
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
    c = '#5494DA'  # '#A986B5'
    p.colors(pointshape="s", errorbarcolor=c, pointcolor=c)  # , linecolor='#fcb2ab')
    ax = p.plot(figsize=(8, 0.4 * 37 / 49 * len(labs)), t_adjuster=0.0108, max_value=3.5, min_value=0.9, size=5,
                decimal=2)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    for i in range(len(organ_n) - 1):
        ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=13)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/DX-all/organ/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'med_hr_{}-p{}-hrGe1-nGe100.png'.format('all', pvalue), bbox_inches='tight', dpi=900)
    plt.savefig(output_dir + 'med_hr_{}-p{}-hrGe1-nGe100.pdf'.format('all', pvalue), bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_dx_organ_compare2data(add_name=True, severity="above65"):
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
        df1 = pd.merge(df1, df_name[["pasc", "PASC Name Simple", "Organ Domain", "Original CCSR Domain",]],
                       left_on='pasc', right_on='pasc', how='left')
        df1 = df1.rename(columns={x + '_y': x for x in [ "PASC Name Simple", "Organ Domain", "Original CCSR Domain"]})

    df_aux = df2.rename(columns=lambda x: x + '_aux')
    df = pd.merge(df1, df_aux, left_on='pasc', right_on='pasc_aux', how='left').set_index('i')

    pvalue = 0.05 / 137  # 0.01  #

    def select_criteria(_df):
        _df_select = _df.sort_values(by='hr-w', ascending=False)
        _df_select = _df_select.loc[(_df_select['hr-w-p'] <= pvalue) | (_df_select['pasc'] == 'Muscle disorders'), :]  #
        _df_select = _df_select.loc[(_df_select['hr-w'] > 1) | (_df_select['pasc'] == 'Muscle disorders'), :]
        _df_select = _df_select.loc[(_df_select['no. pasc in +'] >= 100) | (_df_select['pasc'] == 'Muscle disorders'),
                     :]
        print('_df_select.shape:', _df_select.shape, _df_select['pasc'])
        return _df_select

    df_select = select_criteria(df)

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
            name = row['PASC Name Simple']
            hr = row['hr-w']
            ci = stringlist_2_list(row['hr-w-CI'])
            p = row['hr-w-p']
            domain = row['Organ Domain']
            pasc = row['pasc']

            hr2 = row['hr-w_aux']
            ci2 = stringlist_2_list(row['hr-w-CI_aux'])
            p2 = row['hr-w-p_aux']

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
    width = 9
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
    plt.savefig(output_dir + '{}-{}_hr_{}-p{:3f}-hrGe1-nGe100-{}v2.png'.format(i, organ, 'all', pvalue, severity),
                bbox_inches='tight',
                dpi=900)
    plt.savefig(output_dir + '{}-{}_hr_{}-p{}-hrGe1-nGe100-{}v2.pdf'.format(i, organ, 'all', pvalue, severity),
                bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def plot_forest_for_med_atc():
    with open(r'../data/mapping/atcL3_index_mapping.pkl', 'rb') as f:
        atcl3_encoding = pickle.load(f)
        print('Load to ATC-Level-3 to encoding mapping done! len(atcl3_encoding):', len(atcl3_encoding))
        record_example = next(iter(atcl3_encoding.items()))
        print('e.g.:', record_example)

    df = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/MED/causal_effects_specific_med.csv')
    df_select = df.sort_values(by='hr-w', ascending=False)
    pvalue = 0.01  # / 100
    df_select = df_select.loc[df_select['hr-w-p'] < pvalue, :]  #
    df_select = df_select.loc[df_select['no. pasc in +'] >= 50, :]

    # df_select = df_select.loc[df_select['hr-w']>1, :]

    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    for key, row in df_select.iterrows():
        name = row['pasc']
        # name_label = row['pasc-med'].strip('][').split(',')
        name_label = atcl3_encoding.get(name, [])[2].strip().strip(r'\'').lower()
        name_label = name_label.replace('<n>', ' ').replace('</n>', ' ')
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
    ax = p.plot(figsize=(11, 20), t_adjuster=0.015, max_value=5, min_value=0.4, size=5, decimal=2)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/MED/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'med_hr_forest-p{}.png'.format(pvalue), bbox_inches='tight', dpi=900)
    plt.savefig(output_dir + 'med_hr_forest-p{}.pdf'.format(pvalue), bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    # plt.close()


def plot_forest_for_med():
    rxing_index = utils.load(r'../data/mapping/selected_rxnorm_index.pkl')

    df = pd.read_csv(r'../data/V15_COVID19/output/character/outcome/MED/causal_effects_specific_med-snapshot-100.csv')
    df_select = df.sort_values(by='hr-w', ascending=False)
    pvalue = 0.05 / 100
    df_select = df_select.loc[df_select['hr-w-p'] < pvalue, :]  #
    df_select = df_select.loc[df_select['no. pasc in +'] >= 300, :]

    # df_select = df_select.loc[df_select['hr-w']>1, :]

    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    for key, row in df_select.iterrows():
        name = row['pasc']
        # name_label = row['pasc-med'].strip('][').split(',')
        if name in rxing_index:
            name_label = rxing_index[name][6]
            if pd.isna(name_label):
                name_label = ''
            else:
                name_label = name_label.strip().strip(r'\'').lower()
        else:
            name_label = name
        name_label = name_label.replace('<n>', ' ').replace('</n>', ' ')
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
    ax = p.plot(figsize=(11, 20), t_adjuster=0.015, max_value=5, min_value=0.4, size=5, decimal=2)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/outcome/MED/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'med_hr_forest-p{}.png'.format(pvalue), bbox_inches='tight', dpi=900)
    plt.savefig(output_dir + 'med_hr_forest-p{}.pdf'.format(pvalue), bbox_inches='tight', transparent=True)
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


if __name__ == '__main__':
    # plot_forest_for_dx()
    # plot_forest_for_med()
    # combine_pasc_list()
    # pasc_domain = add_pasc_domain_to_causal_results()
    # df_med = add_drug_name()
    # plot_forest_for_dx_organ()
    # plot_forest_for_med_organ()
    #
    # plot_forest_for_dx_organ_compare2data(add_name=False, severity='all')
    # plot_forest_for_dx_organ_compare2data(add_name=True, severity='less65')
    # plot_forest_for_dx_organ_compare2data(add_name=True, severity='above65')
    plot_forest_for_dx_organ_compare2data(add_name=True, severity='65to75')
