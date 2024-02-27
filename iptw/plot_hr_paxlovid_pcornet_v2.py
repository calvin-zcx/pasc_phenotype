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

print = functools.partial(print, flush=True)


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

    pasc_simname['any_pasc'] = ('Any PASC', 'Any PASC')
    pasc_simname['smell and taste'] = ('Smell and taste', 'General')

    pasc_simname['death'] = ('Death Overall', 'Death')
    # pasc_simname['death_acute'] = ('Acute Death', 'Death')
    pasc_simname['death_postacute'] = ('Death', 'Death')

    # pasc_simname['death'] = ('Death Overall', 'Death')
    pasc_simname['hospitalization_postacute'] = ('Hospitalization', 'Hospitalization')
    pasc_simname['any_CFR'] = ('Any CFR', 'Any CFR')

    return pasc_simname


def plot_forest_for_dx_organ_pax(star=True, text_right=False):
    indir = r'../data/recover/output/results/Paxlovid-atrisknopreg-all-pcornet-V2/'

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
        'Death',
        'Hospitalization',
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
    ax = p.plot_with_incidence(figsize=(9, .47 * len(labs)), t_adjuster=0.0108, max_value=1.3, min_value=0.5,
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


def plot_forest_for_dx_organ_pax_lib2(star=True, text_right=False):
    indir = r'../data/recover/output/results/Paxlovid-atrisknopreg-all-pcornet-V2/'
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
        'Death',
        'Hospitalization',
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
        # 'Any CFR',
        # 'cognitive-fatigue-respiratory',
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
        # 'brainfog',
        'Any CFR': 'CFR',
        'cognitive-fatigue-respiratory': 'CFR',
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
            if name == 'Death Overall':
                continue
            if name == 'Abnormal heartbeat':
                name = 'Dysrhythmia'

            pasc = row['pasc']
            print(name, pasc)
            hr = row['hr-w']
            if pd.notna(row['hr-w-CI']):
                ci = stringlist_2_list(row['hr-w-CI'])
            else:
                ci = [np.nan, np.nan]
            p = row['hr-w-p']
            # p_format =
            if p <= 0.001:
                sigsym = '$^{***}$'
                p_format = '{:.1e}'.format(p)
            elif p <= 0.01:
                sigsym = '$^{**}$'
                p_format = '{:.3f}'.format(p)
            elif p <= 0.05:
                sigsym = '$^{*}$'
                p_format = '{:.3f}'.format(p)
            else:
                sigsym = '$^{ns}$'
                p_format = '{:.3f}'.format(p)
            p_format += sigsym
            domain = row['Organ Domain']
            cif1 = stringlist_2_list(row['cif_1_w'])[-1] * 100
            cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 100,
                       stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 100]

            # use nabs for ncum_ci_negative
            cif0 = stringlist_2_list(row['cif_0_w'])[-1] * 100
            cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1] * 100,
                       stringlist_2_list(row['cif_0_w_CIUpper'])[-1] * 100]
            result = [name, pasc, organ_mapname[organ], hr,
                      '{:.2f}'.format(hr), p, '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
                      '({:.2f},{:.2f})'.format(ci[0], ci[1]),
                      cif1, cif1_ci[0], cif1_ci[1], '{:.2f}'.format(cif1),
                      cif0, cif0_ci[0], cif0_ci[1], '{:.2f}'.format(cif0),
                      p_format, sigsym]

            if domain == organ:
                results_list.append(result)

    # df_result = pd.concat(results_list,  ignore_index=True)
    # df_result = df_result.rename()
    df_result = pd.DataFrame(results_list,
                             columns=['name', 'pasc', 'group', 'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
                                      'aHR-CI-str',
                                      'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
                                      'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
                                      'p-val-sci', 'sigsym'])
    df_result['-aHR'] = -1 * df_result['aHR']
    plt.rc('font', family='serif')
    # fig, ax = plt.subplots()
    axs = fp.forestplot(
        df_result,  # the dataframe with results data
        figsize=(4, 12),
        estimate="aHR",  # col containing estimated effect size
        ll='aHR-lb',
        hl='aHR-ub',  # lower & higher limits of conf. int.
        varlabel="name",  # column containing the varlabels to be printed on far left
        # capitalize="capitalize",  # Capitalize labels
        pval="p-val",  # column containing p-values to be formatted
        starpval=True,
        annote=[],  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
        annoteheaders=[],
        rightannote=["aHR-str", "aHR-CI-str", 'p-val-sci', 'CIF1-str', 'CIF0-str'],
        # p_format, columns to report on right of plot
        right_annoteheaders=["aHR", "95% CI", "P-value", 'CIF1', 'CIF0'],  # p_format, ^corresponding headers
        groupvar="group",  # column containing group labels
        group_order=df_result['group'].unique(),
        xlabel="adjusted Hazard Ratio",  # x-label title
        xticks=[0.4, 1, 1.2],  # x-ticks to be printed
        color_alt_rows=True,
        # flush=True,
        sort=True,  # sort estimates in ascending order
        # sortby='-aHR',
        # table=True,  # Format as a table
        # logscale=True,
        # Additional kwargs for customizations
        **{
            "marker": "D",  # set maker symbol as diamond
            "markersize": 35,  # adjust marker size
            "xlinestyle": (0., (10, 5)),  # long dash for x-reference line
            "xlinecolor": ".1",  # gray color for x-reference line
            "xtick_size": 12,  # adjust x-ticker fontsize

        },
    )
    axs.axvline(x=1, ymin=0, ymax=0.95, color='grey', linestyle='dashed')
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'hr_moretabs.png', bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_moretabs.pdf', bbox_inches='tight', transparent=True)

    print('Done')
    return df_result


def plot_forest_for_pax_subgroup_lib2(star=True, text_right=False):
    subgroup_list = [
                     'female', 'male',
                     'white', 'black',
                     'less65', 'above65',
                     'PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease',
                     'PaxRisk:Chronic liver disease',
                     'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
                     'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes',
                     'PaxRisk:Disabilities',
                     'PaxRisk:Heart conditions', 'PaxRisk:Hypertension',
                     'PaxRisk:HIV infection',
                     'PaxRisk:immune',
                     'PaxRisk:Mental health conditions',
                     'PaxRisk:Overweight and obesity',  # 'PaxRisk:Pregnancy',
                     'PaxRisk:Sickle cell disease or thalassemia',
                     'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
                     'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis',
        'norisk', 'pregnant',
                     'VA',
                     'CFR']

    subgroup_info_map = {
        'norisk': ['No Documented Risk Factors', 'Without Indication'],
        'pregnant': ['Pregnant', 'Without Indication'],
        'VA': ['VA-like cohort', 'Sensitivity Analysis'],
        'CFR':['Cognitive/Fatigue/Respiratory', 'Sensitivity Analysis'],
        'female': ['Female', 'Sex'],
        'male': ['Male', 'Sex'],
        'white': ['White', 'Race'],
        'black': ['Black', 'Race'],
        'less65': ['<65', 'Age'],
        'above65': ['≥65', 'Age'],
        'PaxRisk:Cancer': ['Cancer', 'With Risk Factor'],
        'PaxRisk:Chronic kidney disease': ['Chronic Kidney Disease', 'With Risk Factor'],
        'PaxRisk:Chronic liver disease': ['Chronic Liver Disease', 'With Risk Factor'],
        'PaxRisk:Chronic lung disease': ['Chronic Lung Disease', 'With Risk Factor'],
        'PaxRisk:Cystic fibrosis': ['Cystic Fibrosis', 'With Risk Factor'],
        'PaxRisk:Dementia or other neurological conditions': ['Dementia/Neurological', 'With Risk Factor'],
        'PaxRisk:Diabetes': ['Diabetes', 'With Risk Factor'],
        'PaxRisk:Disabilities': ['Disabilities', 'With Risk Factor'],
        'PaxRisk:Heart conditions': ['Heart Conditions', 'With Risk Factor'],
        'PaxRisk:Hypertension': ['Hypertension', 'With Risk Factor'],
        'PaxRisk:HIV infection': ['HIV', 'With Risk Factor'],
        'PaxRisk:immune': ['Immune Dysfunction', 'With Risk Factor'],
        'PaxRisk:Mental health conditions': ['Mental Health Conditions', 'With Risk Factor'],
        'PaxRisk:Overweight and obesity': ['Overweight and Obesity', 'With Risk Factor'],  # 'PaxRisk:Pregnancy',
        'PaxRisk:Sickle cell disease or thalassemia': ['Sickle Cell/Anemia', 'With Risk Factor'],
        'PaxRisk:Smoking current': ['Smoking current/former', 'With Risk Factor'],
        'PaxRisk:Stroke or cerebrovascular disease': ['Stroke/Cerebrovascular Disease', 'With Risk Factor'],
        'PaxRisk:Substance use disorders': ['Substance Use Disorders', 'With Risk Factor'],
        'PaxRisk:Tuberculosis': ['Tuberculosis', 'With Risk Factor'], }

    output_dir = r'../data/recover/output/results/figure_subgroup/'

    results_list = []
    for subgroup in subgroup_list:
        indir = r'../data/recover/output/results/Paxlovid-atrisknopreg-{}-pcornet-V2/'.format(
            subgroup.replace(':', '_').replace('/', '-').replace(' ', '_')
        )
        info = subgroup_info_map[subgroup]
        subgroupname = info[0]
        grouplabel = info[1]

        # subgroupname = subgroup.split(':')[-1]
        if subgroup == 'norisk':
            indir = r'../data/recover/output/results/Paxlovid-norisk-all-pcornet-V2/'
            # subgroupname = 'No documented risk factors'
        elif subgroup == 'pregnant':
            indir = r'../data/recover/output/results/Paxlovid-pregnant-all-pcornet-V2/'
            # subgroupname = 'Pregnant'
        elif subgroup == 'CFR':
            indir = r'../data/recover/output/results/Paxlovid-atrisknopreg-all-pcornet-V2/'

        print('read:', indir)
        # if subgroup == 'PaxRisk:Dementia or other neurological conditions':
        #     df = pd.read_csv(indir + 'causal_effects_specific-snapshot-18.csv')
        #     subgroupname = 'Dementia or other neurological'
        # else:
        #     df = pd.read_csv(indir + 'causal_effects_specific.csv')

        if subgroup == 'CFR':
            df = pd.read_csv(indir + 'causal_effects_specific.csv')
            row = df.loc[df['pasc'] == 'any_CFR', :].squeeze()
            name = 'Any CFR'
        else:
            df = pd.read_csv(indir + 'causal_effects_specific-snapshot-2.csv')
            row = df.loc[df['pasc'] == 'any_pasc', :].squeeze()
            name = 'Any PASC'


        pasc = row['pasc']
        if row['case+'] < 500:
            print(subgroup, 'is very small (<500 in exposed), skip', row['case+'], row['ctrl-'])
            continue
        # print(name, pasc)
        hr = row['hr-w']
        if pd.notna(row['hr-w-CI']):
            ci = stringlist_2_list(row['hr-w-CI'])
        else:
            ci = [np.nan, np.nan]
        p = row['hr-w-p']
        # p_format =
        if p <= 0.001:
            sigsym = '$^{***}$'
            p_format = '{:.1e}'.format(p)
        elif p <= 0.01:
            sigsym = '$^{**}$'
            p_format = '{:.3f}'.format(p)
        elif p <= 0.05:
            sigsym = '$^{*}$'
            p_format = '{:.3f}'.format(p)
        else:
            sigsym = '$^{ns}$'
            p_format = '{:.3f}'.format(p)
        p_format += sigsym
        # domain = row['Organ Domain']
        cif1 = stringlist_2_list(row['cif_1_w'])[-1] * 100
        cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 100,
                   stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 100]

        # use nabs for ncum_ci_negative
        cif0 = stringlist_2_list(row['cif_0_w'])[-1] * 100
        cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1] * 100,
                   stringlist_2_list(row['cif_0_w_CIUpper'])[-1] * 100]
        result = [subgroupname, grouplabel, name, pasc,
                  '{:.0f}'.format(row['case+']), '{:.0f}'.format(row['ctrl-']),
                  hr,
                  '{:.2f}'.format(hr), p, '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
                  '({:.2f},{:.2f})'.format(ci[0], ci[1]),
                  cif1, cif1_ci[0], cif1_ci[1], '{:.2f}'.format(cif1),
                  cif0, cif0_ci[0], cif0_ci[1], '{:.2f}'.format(cif0),
                  p_format, sigsym]

        results_list.append(result)

    df_result = pd.DataFrame(results_list,
                             columns=['subgroup', 'grouplabel', 'name', 'pasc', 'No. in 1', 'No. in 0',
                                      'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
                                      'aHR-CI-str',
                                      'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
                                      'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
                                      'p-val-sci', 'sigsym'])
    df_result['-aHR'] = -1 * df_result['aHR']
    plt.rc('font', family='serif')
    # fig, ax = plt.subplots()
    axs = fp.forestplot(
        df_result,  # the dataframe with results data
        figsize=(4.5, 13),
        estimate="aHR",  # col containing estimated effect size
        ll='aHR-lb',
        hl='aHR-ub',  # lower & higher limits of conf. int.
        varlabel="subgroup",  # column containing the varlabels to be printed on far left
        # capitalize="capitalize",  # Capitalize labels
        pval="p-val",  # column containing p-values to be formatted
        starpval=True,
        annote=['No. in 1', 'No. in 0', ],  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
        annoteheaders=['No.Paxlovid', 'No.Ctrl', ],
        rightannote=["aHR-str", "aHR-CI-str", 'p-val-sci', 'CIF1-str', 'CIF0-str'],
        # p_format, columns to report on right of plot
        right_annoteheaders=["aHR", "95% CI", "P-value", 'CIF1', 'CIF0'],  # p_format, ^corresponding headers
        groupvar="grouplabel",  # column containing group labels
        # group_order=df_result['group'].unique(),
        xlabel="adjusted Hazard Ratio",  # x-label title
        xticks=[0.6, 1, 1.5],  # x-ticks to be printed
        color_alt_rows=True,
        # flush=True,
        # sort=True,  # sort estimates in ascending order
        # sortby='-aHR',
        # table=True,  # Format as a table
        # logscale=True,
        # Additional kwargs for customizations
        **{
            "marker": "D",  # set maker symbol as diamond
            "markersize": 35,  # adjust marker size
            "xlinestyle": (0., (10, 5)),  # long dash for x-reference line
            "xlinecolor": ".1",  # gray color for x-reference line
            "xtick_size": 14,  # adjust x-ticker fontsize

        },
    )
    axs.axvline(x=1, ymin=0, ymax=0.95, color='grey', linestyle='dashed')
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'hr_subgroup.png', bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_subgroup.pdf', bbox_inches='tight', transparent=True)

    print('Done')
    return df_result


if __name__ == '__main__':
    # plot_forest_for_dx_organ_pax()

    # df_result = plot_forest_for_dx_organ_pax_lib2()
    df_result = plot_forest_for_pax_subgroup_lib2()

    # df = fp.load_data("sleep")  # companion example data
    # df.head(3)
    #
    # # fig = plt.figure(figsize=(3, 8))  # figsize=(12, 9)
    # # ax = fig.add_subplot(1, 1, 1)
    # fp.forestplot(
    #     df,  # the dataframe with results data
    #     estimate="r",  # col containing estimated effect size
    #     ll="ll",
    #     hl="hl",  # lower & higher limits of conf. int.
    #     varlabel="label",  # column containing the varlabels to be printed on far left
    #     capitalize="capitalize",  # Capitalize labels
    #     pval="p-val",  # column containing p-values to be formatted
    #     annote=["n", "power", "est_ci"],  # columns to report on left of plot
    #     annoteheaders=["N", "Power", "Est. (95% Conf. Int.)"],  # ^corresponding headers
    #     rightannote=["formatted_pval", "group"],  # columns to report on right of plot
    #     right_annoteheaders=["P-value", "Variable group"],  # ^corresponding headers
    #     groupvar="group",  # column containing group labels
    #     group_order=[
    #         "labor factors",
    #         "occupation",
    #         "age",
    #         "health factors",
    #         "family factors",
    #         "area of residence",
    #         "other factors",
    #     ],
    #     xlabel="Pearson correlation coefficient",  # x-label title
    #     xticks=[-0.4, -0.2, 0, 0.2],  # x-ticks to be printed
    #     sort=True,  # sort estimates in ascending order
    #     table=True,  # Format as a table
    #     # Additional kwargs for customizations
    #     **{
    #         "marker": "D",  # set maker symbol as diamond
    #         "markersize": 35,  # adjust marker size
    #         "xlinestyle": (0, (10, 5)),  # long dash for x-reference line
    #         "xlinecolor": ".1",  # gray color for x-reference line
    #         "xtick_size": 12,  # adjust x-ticker fontsize
    #     },
    # )
    #
    # # plt.tight_layout()
    # plt.savefig("hr_plot_test.png", dpi="figure", bbox_inches="tight")
    # plt.savefig("hr_plot_test.pdf", dpi="figure", bbox_inches="tight")
    # plt.show()
    print('Done!')
