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

print = functools.partial(print, flush=True)


def load_pasc_info():
    print('load and preprocess PASC info')
    df_pasc_info = pd.read_excel(r'../prediction/output/causal_effects_specific_withMedication_v3.xlsx',
                                 sheet_name='diagnosis')
    addedPASC_encoding = utils.load(r'../data/mapping/addedPASC_index_mapping.pkl')
    addedPASC_list = list(addedPASC_encoding.keys())
    brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
    brainfog_list = list(brainfog_encoding.keys())
    print('remove Headache from brainfog_list')
    brainfog_list = ['Neurodegenerative', 'Memory-Attention', #'Headache',
                     'Sleep Disorder', 'Psych', 'Dysautonomia-Orthostatic', 'Stroke']


    CFR_encoding = utils.load(r'../data/mapping/cognitive-fatigue-respiratory_index_mapping.pkl')
    CFR_list = list(CFR_encoding.keys())

    # mecfs_encoding = utils.load(r'../data/mapping/mecfs_index_mapping.pkl')
    # mecfs_list = list(mecfs_encoding.keys())

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

    # for p in mecfs_list:
    #     pasc_simname[p] = (p, 'General-add')
    #     pasc_organ[p] = 'General-add'

    pasc_simname['any_pasc'] = ('Long COVID', 'Any PASC')
    pasc_simname['smell and taste'] = ('Smell and taste', 'General')
    pasc_simname['ME/CFS'] = ('ME/CFS', 'General')

    pasc_simname['death'] = ('Death Overall', 'Death')
    # pasc_simname['death_acute'] = ('Acute Death', 'Death')
    pasc_simname['death_postacute'] = ('All-cause mortality', 'Death')
    pasc_simname['cvddeath_postacute'] = ('CVD mortality', 'Death')

    # pasc_simname['death'] = ('Death Overall', 'Death')
    pasc_simname['hospitalization_postacute'] = ('Hospitalization', 'Hospitalization')
    pasc_simname['any_CFR'] = ('Any CFR', 'Any CFR')
    pasc_simname['any_brainfog'] = ('Any Brain Fog', 'Any Brain Fog')

    return pasc_simname


# def plot_forest_for_dx_organ_pax(star=True, text_right=False):
#     indir = r'../data/recover/output/results/Paxlovid-atrisknopreg-all-pcornet-V2/'
#
#     output_dir = indir + r'figure/'
#
#     df = pd.read_csv(indir + 'causal_effects_specific.csv')
#     df.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )
#
#     pasc_simname_organ = load_pasc_info()
#
#     df.insert(df.columns.get_loc('pasc') + 1, 'Organ Domain', '')
#     df.insert(df.columns.get_loc('pasc') + 1, 'PASC Name Simple', '')
#
#     for key, row in df.iterrows():
#         pasc = row['pasc']
#         if pasc in pasc_simname_organ:
#             df.loc[key, 'PASC Name Simple'] = pasc_simname_organ[pasc][0]
#             df.loc[key, 'Organ Domain'] = pasc_simname_organ[pasc][1]
#
#     df_select = df.sort_values(by='hr-w', ascending=True)
#     # df_select = df_select.loc[df_select['selected'] == 1, :]  #
#     print('df_select.shape:', df_select.shape)
#
#     organ_list = df_select['Organ Domain'].unique()
#     print(organ_list)
#     organ_list = [
#         'Any PASC',
#         'Death',
#         'Hospitalization',
#         'Diseases of the Nervous System',
#         'Diseases of the Skin and Subcutaneous Tissue',
#         'Diseases of the Respiratory System',
#         'Diseases of the Circulatory System',
#         'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism',
#         'Endocrine, Nutritional and Metabolic Diseases',
#         'Diseases of the Digestive System',
#         'Diseases of the Genitourinary System',
#         'Diseases of the Musculoskeletal System and Connective Tissue',
#         # 'Certain Infectious and Parasitic Diseases',
#         'General',
#         # 'General-add',
#         # 'brainfog',
#         'Any CFR',
#         'cognitive-fatigue-respiratory'
#     ]
#     # 'Injury, Poisoning and Certain Other Consequences of External Causes']
#     organ_n = np.zeros(len(organ_list))
#     labs = []
#     measure = []
#     lower = []
#     upper = []
#     pval = []
#     pasc_row = []
#
#     nabsv = []
#     ncumv = []
#
#     for i, organ in enumerate(organ_list):
#         print(i + 1, 'organ', organ)
#
#         for key, row in df_select.iterrows():
#             name = row['PASC Name Simple'].strip('*')
#             pasc = row['pasc']
#             print(name, pasc)
#             if pasc.startswith('smm:'):
#                 continue
#             hr = row['hr-w']
#             if pd.notna(row['hr-w-CI']):
#                 ci = stringlist_2_list(row['hr-w-CI'])
#             else:
#                 ci = [np.nan, np.nan]
#
#             p = row['hr-w-p']
#             domain = row['Organ Domain']
#
#             # nabs = row['no. pasc in +']
#             ncum = stringlist_2_list(row['cif_1_w'])[-1] * 1000
#             ncum_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 1000,
#                        stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 1000]
#
#             # use nabs for ncum_ci_negative
#             nabs = stringlist_2_list(row['cif_0_w'])[-1] * 1000
#
#             if star:
#                 if p <= 0.001:
#                     name += '***'
#                 elif p <= 0.01:
#                     name += '**'
#                 elif p <= 0.05:
#                     name += '*'
#
#             # if (database == 'V15_COVID19') and (row['selected'] == 1) and (row['selected oneflorida'] == 1):
#             #     name += r'$^{‡}$'
#             #
#             # if (database == 'oneflorida') and (row['selected'] == 1) and (row['selected insight'] == 1):
#             #     name += r'$^{‡}$'
#
#             # if pasc == 'PASC-General':
#             #     pasc_row = [name, hr, ci, p, domain, nabs, ncum]
#             #     continue
#             if domain == organ:
#                 organ_n[i] += 1
#                 if len(name.split()) >= 5:
#                     name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])
#                 labs.append(name)
#                 measure.append(hr)
#                 lower.append(ci[0])
#                 upper.append(ci[1])
#                 pval.append(p)
#
#                 nabsv.append(nabs)
#                 ncumv.append(ncum)
#
#     p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper,
#                           nabs=nabsv, ncumIncidence=ncumv)
#     p.labels(scale='log')
#
#     # organ = 'ALL'
#     # p.labels(effectmeasure='aHR', add_label1='CIF per\n1000', add_label2='No. of\nCases')  # aHR
#     p.labels(effectmeasure='aHR', add_label1='CIF per\n1000\nin case', add_label2='CIF per\n1000\nin Ctrl')
#
#     # p.colors(pointcolor='r')
#     # '#F65453', '#82A2D3'
#     # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
#
#     c = '#F65453'
#     p.colors(pointshape="o", errorbarcolor=c, pointcolor=c)  # , linecolor='black'),   # , linecolor='#fcb2ab')
#     ax = p.plot_with_incidence(figsize=(9, .47 * len(labs)), t_adjuster=0.0108, max_value=1.3, min_value=0.5,
#                                size=5, decimal=2,
#                                text_right=text_right)
#
#     # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
#     # plt.title('pasc', loc="center", x=0, y=0)
#     # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
#     # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
#
#     organ_n_cumsum = np.cumsum(organ_n)
#     for i in range(len(organ_n) - 1):
#         ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.09, color=p.linec, zorder=1, linestyle='--')  # linewidth=1,
#
#     # ax.set_yticklabels(labs, fontsize=11.5)
#     ax.set_yticklabels(labs, fontsize=14)
#
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_visible(True)
#     ax.spines['left'].set_visible(False)
#     plt.tight_layout()
#
#     check_and_mkdir(output_dir)
#     plt.savefig(output_dir + 'hr_2CIF-V5.png', bbox_inches='tight', dpi=600)
#
#     plt.savefig(output_dir + 'hr_2CIF-V5.pdf', bbox_inches='tight', transparent=True)
#     plt.show()
#     print()
#     # plt.clf()
#     plt.close()


# def plot_forest_for_dx_organ_pax_lib2(star=True, text_right=False):
#     indir = r'../data/recover/output/results/Paxlovid-atrisknopreg-all-pcornet-V2/'
#     # indir = r'../data/recover/output/results/Paxlovid-pregnant-all-pcornet-V2/'
#
#     output_dir = indir + r'figure/'
#
#     df = pd.read_csv(indir + 'causal_effects_specific.csv')
#     df.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )
#
#     pasc_simname_organ = load_pasc_info()
#
#     df.insert(df.columns.get_loc('pasc') + 1, 'Organ Domain', '')
#     df.insert(df.columns.get_loc('pasc') + 1, 'PASC Name Simple', '')
#
#     for key, row in df.iterrows():
#         pasc = row['pasc']
#         if pasc in pasc_simname_organ:
#             df.loc[key, 'PASC Name Simple'] = pasc_simname_organ[pasc][0]
#             df.loc[key, 'Organ Domain'] = pasc_simname_organ[pasc][1]
#
#     df_select = df.sort_values(by='hr-w', ascending=True)
#     # df_select = df_select.loc[df_select['selected'] == 1, :]  #
#     print('df_select.shape:', df_select.shape)
#
#     organ_list = df_select['Organ Domain'].unique()
#     print(organ_list)
#     organ_list = [
#         'Any PASC',
#         'Death',
#         'Hospitalization',
#         'Diseases of the Nervous System',
#         'Diseases of the Skin and Subcutaneous Tissue',
#         'Diseases of the Respiratory System',
#         'Diseases of the Circulatory System',
#         'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism',
#         'Endocrine, Nutritional and Metabolic Diseases',
#         'Diseases of the Digestive System',
#         'Diseases of the Genitourinary System',
#         'Diseases of the Musculoskeletal System and Connective Tissue',
#         # 'Certain Infectious and Parasitic Diseases',
#         'General',
#         # 'General-add',
#         # 'brainfog',
#         # 'Any CFR',
#         # 'cognitive-fatigue-respiratory',
#     ]
#     organ_mapname = {
#         'Any PASC': 'Overall',
#         'Death': 'Overall',
#         'Hospitalization': 'Overall',
#         'Diseases of the Nervous System': 'Neurologic',
#         'Diseases of the Skin and Subcutaneous Tissue': 'Skin',
#         'Diseases of the Respiratory System': 'Pulmonary',
#         'Diseases of the Circulatory System': 'Circulatory',
#         'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism': 'Blood',
#         'Endocrine, Nutritional and Metabolic Diseases': 'Metabolic',
#         'Diseases of the Digestive System': 'Digestive',
#         'Diseases of the Genitourinary System': 'Genitourinary',
#         'Diseases of the Musculoskeletal System and Connective Tissue': 'Musculoskeletal',
#         # 'Certain Infectious and Parasitic Diseases',
#         'General': 'General',
#         # 'General-add',
#         # 'brainfog',
#         'Any CFR': 'CFR',
#         'cognitive-fatigue-respiratory': 'CFR',
#     }
#     # 'Injury, Poisoning and Certain Other Consequences of External Causes']
#
#     organ_n = np.zeros(len(organ_list))
#     results_list = []
#     for i, organ in enumerate(organ_list):
#         print(i + 1, 'organ', organ)
#         for key, row in df_select.iterrows():
#             name = row['PASC Name Simple'].strip('*')
#             # if len(name.split()) >= 5:
#             #     name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])
#             if name == 'Dyspnea':
#                 name = 'Shortness of breath'
#             if name == 'Death Overall':
#                 continue
#             if name == 'Abnormal heartbeat':
#                 name = 'Dysrhythmia'
#
#             pasc = row['pasc']
#             print(name, pasc)
#             hr = row['hr-w']
#             if pd.notna(row['hr-w-CI']):
#                 ci = stringlist_2_list(row['hr-w-CI'])
#             else:
#                 ci = [np.nan, np.nan]
#             p = row['hr-w-p']
#             # p_format =
#             if p <= 0.001:
#                 sigsym = '$^{***}$'
#                 p_format = '{:.1e}'.format(p)
#             elif p <= 0.01:
#                 sigsym = '$^{**}$'
#                 p_format = '{:.3f}'.format(p)
#             elif p <= 0.05:
#                 sigsym = '$^{*}$'
#                 p_format = '{:.3f}'.format(p)
#             else:
#                 sigsym = '$^{ns}$'
#                 p_format = '{:.3f}'.format(p)
#             p_format += sigsym
#             domain = row['Organ Domain']
#             cif1 = stringlist_2_list(row['cif_1_w'])[-1] * 100
#             cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 100,
#                        stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 100]
#
#             # use nabs for ncum_ci_negative
#             cif0 = stringlist_2_list(row['cif_0_w'])[-1] * 100
#             cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1] * 100,
#                        stringlist_2_list(row['cif_0_w_CIUpper'])[-1] * 100]
#             result = [name, pasc, organ_mapname[organ], hr,
#                       '{:.2f}'.format(hr), p, '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
#                       '({:.2f},{:.2f})'.format(ci[0], ci[1]),
#                       cif1, cif1_ci[0], cif1_ci[1], '{:.2f}'.format(cif1),
#                       cif0, cif0_ci[0], cif0_ci[1], '{:.2f}'.format(cif0),
#                       p_format, sigsym]
#
#             if domain == organ:
#                 results_list.append(result)
#
#     # df_result = pd.concat(results_list,  ignore_index=True)
#     # df_result = df_result.rename()
#     df_result = pd.DataFrame(results_list,
#                              columns=['name', 'pasc', 'group', 'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
#                                       'aHR-CI-str',
#                                       'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
#                                       'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
#                                       'p-val-sci', 'sigsym'])
#     # df_result['-aHR'] = -1 * df_result['aHR']
#     df_result = df_result.loc[~df_result['aHR'].isna()]
#     plt.rc('font', family='serif')
#     # fig, ax = plt.subplots()
#     axs = fp.forestplot(
#         df_result,  # the dataframe with results data
#         figsize=(4, 12),
#         estimate="aHR",  # col containing estimated effect size
#         ll='aHR-lb',
#         hl='aHR-ub',  # lower & higher limits of conf. int.
#         varlabel="name",  # column containing the varlabels to be printed on far left
#         # capitalize="capitalize",  # Capitalize labels
#         pval="p-val",  # column containing p-values to be formatted
#         starpval=True,
#         annote=[],  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
#         # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
#         annoteheaders=[],
#         rightannote=["aHR-str", "aHR-CI-str", 'p-val-sci', 'CIF1-str', 'CIF0-str'],
#         # p_format, columns to report on right of plot
#         right_annoteheaders=["aHR", "95% CI", "P-value", 'CIF1', 'CIF0'],  # p_format, ^corresponding headers
#         groupvar="group",  # column containing group labels
#         group_order=df_result['group'].unique(),
#         xlabel="adjusted Hazard Ratio",  # x-label title
#         xticks=[0.4, 1, 1.2],  # x-ticks to be printed
#         color_alt_rows=True,
#         # flush=True,
#         sort=True,  # sort estimates in ascending order
#         # sortby='-aHR',
#         # table=True,  # Format as a table
#         # logscale=True,
#         # Additional kwargs for customizations
#         **{
#             "marker": "D",  # set maker symbol as diamond
#             "markersize": 35,  # adjust marker size
#             "xlinestyle": (0., (10, 5)),  # long dash for x-reference line
#             "xlinecolor": ".1",  # gray color for x-reference line
#             "xtick_size": 12,  # adjust x-ticker fontsize
#
#         },
#     )
#     axs.axvline(x=1, ymin=0, ymax=0.95, color='grey', linestyle='dashed')
#     check_and_mkdir(output_dir)
#     plt.savefig(output_dir + 'hr_moretabs.png', bbox_inches='tight', dpi=600)
#     plt.savefig(output_dir + 'hr_moretabs.pdf', bbox_inches='tight', transparent=True)
#
#     print('Done')
#     return df_result


# def plot_forest_for_pax_subgroup_lib2(star=True, text_right=False):
#     subgroup_list = [
#         'female', 'male',
#         'white', 'black',
#         'less65', 'above65',
#         'PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease',
#         'PaxRisk:Chronic liver disease',
#         'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
#         'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes',
#         'PaxRisk:Disabilities',
#         'PaxRisk:Heart conditions', 'PaxRisk:Hypertension',
#         'PaxRisk:HIV infection',
#         'PaxRisk:immune',
#         'PaxRisk:Mental health conditions',
#         'PaxRisk:Overweight and obesity',  # 'PaxRisk:Pregnancy',
#         'PaxRisk:Sickle cell disease or thalassemia',
#         'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
#         'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis',
#         'norisk', 'pregnant',
#         'VA',
#         'CFR']
#
#     subgroup_info_map = {
#         'norisk': ['No Documented Risk Factors', 'Without Indication'],
#         'pregnant': ['Pregnant', 'Without Indication'],
#         'VA': ['VA-like cohort', 'Sensitivity Analysis'],
#         'CFR': ['Fatigue, Cognitive, Respiratory', 'Sensitivity Analysis'],
#         'female': ['Female', 'Sex'],
#         'male': ['Male', 'Sex'],
#         'white': ['White', 'Race'],
#         'black': ['Black', 'Race'],
#         'less65': ['<65', 'Age'],
#         'above65': ['≥65', 'Age'],
#         'PaxRisk:Cancer': ['Cancer', 'With Risk Factor'],
#         'PaxRisk:Chronic kidney disease': ['Chronic Kidney Disease', 'With Risk Factor'],
#         'PaxRisk:Chronic liver disease': ['Chronic Liver Disease', 'With Risk Factor'],
#         'PaxRisk:Chronic lung disease': ['Chronic Lung Disease', 'With Risk Factor'],
#         'PaxRisk:Cystic fibrosis': ['Cystic Fibrosis', 'With Risk Factor'],
#         'PaxRisk:Dementia or other neurological conditions': ['Dementia/Neurological', 'With Risk Factor'],
#         'PaxRisk:Diabetes': ['Diabetes', 'With Risk Factor'],
#         'PaxRisk:Disabilities': ['Disabilities', 'With Risk Factor'],
#         'PaxRisk:Heart conditions': ['Heart Conditions', 'With Risk Factor'],
#         'PaxRisk:Hypertension': ['Hypertension', 'With Risk Factor'],
#         'PaxRisk:HIV infection': ['HIV', 'With Risk Factor'],
#         'PaxRisk:immune': ['Immune Dysfunction', 'With Risk Factor'],
#         'PaxRisk:Mental health conditions': ['Mental Health Conditions', 'With Risk Factor'],
#         'PaxRisk:Overweight and obesity': ['Overweight and Obesity', 'With Risk Factor'],  # 'PaxRisk:Pregnancy',
#         'PaxRisk:Sickle cell disease or thalassemia': ['Sickle Cell/Anemia', 'With Risk Factor'],
#         'PaxRisk:Smoking current': ['Smoking current/former', 'With Risk Factor'],
#         'PaxRisk:Stroke or cerebrovascular disease': ['Stroke/Cerebrovascular Disease', 'With Risk Factor'],
#         'PaxRisk:Substance use disorders': ['Substance Use Disorders', 'With Risk Factor'],
#         'PaxRisk:Tuberculosis': ['Tuberculosis', 'With Risk Factor'], }
#
#     output_dir = r'../data/recover/output/results/figure_subgroup/'
#
#     results_list = []
#     for subgroup in subgroup_list:
#         indir = r'../data/recover/output/results/Paxlovid-atrisknopreg-{}-pcornet-V2/'.format(
#             subgroup.replace(':', '_').replace('/', '-').replace(' ', '_')
#         )
#         info = subgroup_info_map[subgroup]
#         subgroupname = info[0]
#         grouplabel = info[1]
#
#         # subgroupname = subgroup.split(':')[-1]
#         if subgroup == 'norisk':
#             indir = r'../data/recover/output/results/Paxlovid-norisk-all-pcornet-V2/'
#             # subgroupname = 'No documented risk factors'
#         elif subgroup == 'pregnant':
#             indir = r'../data/recover/output/results/Paxlovid-pregnant-all-pcornet-V2/'
#             # subgroupname = 'Pregnant'
#         elif subgroup == 'CFR':
#             indir = r'../data/recover/output/results/Paxlovid-atrisknopreg-all-pcornet-V2/'
#
#         print('read:', indir)
#         # if subgroup == 'PaxRisk:Dementia or other neurological conditions':
#         #     df = pd.read_csv(indir + 'causal_effects_specific-snapshot-18.csv')
#         #     subgroupname = 'Dementia or other neurological'
#         # else:
#         #     df = pd.read_csv(indir + 'causal_effects_specific.csv')
#
#         if subgroup == 'CFR':
#             df = pd.read_csv(indir + 'causal_effects_specific.csv')
#             row = df.loc[df['pasc'] == 'any_CFR', :].squeeze()
#             name = 'Any CFR'
#         else:
#             df = pd.read_csv(indir + 'causal_effects_specific-snapshot-2.csv')
#             row = df.loc[df['pasc'] == 'any_pasc', :].squeeze()
#             name = 'Any PASC'
#
#         pasc = row['pasc']
#         if row['case+'] < 500:
#             print(subgroup, 'is very small (<500 in exposed), skip', row['case+'], row['ctrl-'])
#             continue
#         # print(name, pasc)
#         hr = row['hr-w']
#         if pd.notna(row['hr-w-CI']):
#             ci = stringlist_2_list(row['hr-w-CI'])
#         else:
#             ci = [np.nan, np.nan]
#         p = row['hr-w-p']
#         # p_format =
#         if p <= 0.001:
#             sigsym = '$^{***}$'
#             p_format = '{:.1e}'.format(p)
#         elif p <= 0.01:
#             sigsym = '$^{**}$'
#             p_format = '{:.3f}'.format(p)
#         elif p <= 0.05:
#             sigsym = '$^{*}$'
#             p_format = '{:.3f}'.format(p)
#         else:
#             sigsym = '$^{ns}$'
#             p_format = '{:.3f}'.format(p)
#         p_format += sigsym
#         # domain = row['Organ Domain']
#         cif1 = stringlist_2_list(row['cif_1_w'])[-1] * 100
#         cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 100,
#                    stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 100]
#
#         # use nabs for ncum_ci_negative
#         cif0 = stringlist_2_list(row['cif_0_w'])[-1] * 100
#         cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1] * 100,
#                    stringlist_2_list(row['cif_0_w_CIUpper'])[-1] * 100]
#         result = [subgroupname, grouplabel, name, pasc,
#                   '{:.0f}'.format(row['case+']), '{:.0f}'.format(row['ctrl-']),
#                   hr,
#                   '{:.2f}'.format(hr), p, '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
#                   '({:.2f},{:.2f})'.format(ci[0], ci[1]),
#                   cif1, cif1_ci[0], cif1_ci[1], '{:.2f}'.format(cif1),
#                   cif0, cif0_ci[0], cif0_ci[1], '{:.2f}'.format(cif0),
#                   p_format, sigsym]
#
#         results_list.append(result)
#
#     df_result = pd.DataFrame(results_list,
#                              columns=['subgroup', 'grouplabel', 'name', 'pasc', 'No. in 1', 'No. in 0',
#                                       'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
#                                       'aHR-CI-str',
#                                       'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
#                                       'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
#                                       'p-val-sci', 'sigsym'])
#     df_result['-aHR'] = -1 * df_result['aHR']
#     plt.rc('font', family='serif')
#     # fig, ax = plt.subplots()
#     axs = fp.forestplot(
#         df_result,  # the dataframe with results data
#         figsize=(4.5, 13),
#         estimate="aHR",  # col containing estimated effect size
#         ll='aHR-lb',
#         hl='aHR-ub',  # lower & higher limits of conf. int.
#         varlabel="subgroup",  # column containing the varlabels to be printed on far left
#         # capitalize="capitalize",  # Capitalize labels
#         pval="p-val",  # column containing p-values to be formatted
#         starpval=True,
#         annote=['No. in 1', 'No. in 0', ],  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
#         # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
#         annoteheaders=['No.Paxlovid', 'No.Ctrl', ],
#         rightannote=["aHR-str", "aHR-CI-str", 'p-val-sci', 'CIF1-str', 'CIF0-str'],
#         # p_format, columns to report on right of plot
#         right_annoteheaders=["aHR", "95% CI", "P-value", 'CIF1', 'CIF0'],  # p_format, ^corresponding headers
#         groupvar="grouplabel",  # column containing group labels
#         # group_order=df_result['group'].unique(),
#         xlabel="adjusted Hazard Ratio",  # x-label title
#         xticks=[0.6, 1, 1.5],  # x-ticks to be printed
#         color_alt_rows=True,
#         # flush=True,
#         # sort=True,  # sort estimates in ascending order
#         # sortby='-aHR',
#         # table=True,  # Format as a table
#         # logscale=True,
#         # Additional kwargs for customizations
#         **{
#             "marker": "D",  # set maker symbol as diamond
#             "markersize": 35,  # adjust marker size
#             "xlinestyle": (0., (10, 5)),  # long dash for x-reference line
#             "xlinecolor": ".1",  # gray color for x-reference line
#             "xtick_size": 14,  # adjust x-ticker fontsize
#
#         },
#     )
#     axs.axvline(x=1, ymin=0, ymax=0.95, color='grey', linestyle='dashed')
#     check_and_mkdir(output_dir)
#     plt.savefig(output_dir + 'hr_subgroup.png', bbox_inches='tight', dpi=600)
#     plt.savefig(output_dir + 'hr_subgroup.pdf', bbox_inches='tight', transparent=True)
#
#     print('Done')
#     return df_result


# def plot_forest_for_dx_organ_pax_lib2_cifdiff(show='full'):
#     indir = r'../data/recover/output/results/Paxlovid-atrisk-all-pcornet-V3/'
#     # indir = r'../data/recover/output/results/Paxlovid-pregnant-all-pcornet-V3/'
#
#     output_dir = indir + r'figure/'
#
#     df = pd.read_csv(indir + 'causal_effects_specific.csv')
#     df.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )
#
#     pasc_simname_organ = load_pasc_info()
#
#     df.insert(df.columns.get_loc('pasc') + 1, 'Organ Domain', '')
#     df.insert(df.columns.get_loc('pasc') + 1, 'PASC Name Simple', '')
#
#     for key, row in df.iterrows():
#         pasc = row['pasc']
#         if pasc in pasc_simname_organ:
#             df.loc[key, 'PASC Name Simple'] = pasc_simname_organ[pasc][0]
#             df.loc[key, 'Organ Domain'] = pasc_simname_organ[pasc][1]
#
#     df_select = df.sort_values(by='hr-w', ascending=True)
#     # df_select = df_select.loc[df_select['selected'] == 1, :]  #
#     print('df_select.shape:', df_select.shape)
#
#     organ_list = df_select['Organ Domain'].unique()
#     print(organ_list)
#     organ_list = [
#         'Any PASC',
#         'Death',
#         'Hospitalization',
#         'Diseases of the Nervous System',
#         'Diseases of the Skin and Subcutaneous Tissue',
#         'Diseases of the Respiratory System',
#         'Diseases of the Circulatory System',
#         'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism',
#         'Endocrine, Nutritional and Metabolic Diseases',
#         'Diseases of the Digestive System',
#         'Diseases of the Genitourinary System',
#         'Diseases of the Musculoskeletal System and Connective Tissue',
#         # 'Certain Infectious and Parasitic Diseases',
#         'General',
#         # 'General-add',
#         # 'brainfog',
#         # 'Any CFR',
#         # 'cognitive-fatigue-respiratory',
#     ]
#     organ_mapname = {
#         'Any PASC': 'Overall',
#         'Death': 'Overall',
#         'Hospitalization': 'Overall',
#         'Diseases of the Nervous System': 'Neurologic',
#         'Diseases of the Skin and Subcutaneous Tissue': 'Skin',
#         'Diseases of the Respiratory System': 'Pulmonary',
#         'Diseases of the Circulatory System': 'Circulatory',
#         'Diseases of the Blood and Blood Forming Organs and Certain Disorders Involving the Immune Mechanism': 'Blood',
#         'Endocrine, Nutritional and Metabolic Diseases': 'Metabolic',
#         'Diseases of the Digestive System': 'Digestive',
#         'Diseases of the Genitourinary System': 'Genitourinary',
#         'Diseases of the Musculoskeletal System and Connective Tissue': 'Musculoskeletal',
#         # 'Certain Infectious and Parasitic Diseases',
#         'General': 'General',
#         # 'General-add',
#         # 'brainfog',
#         'Any CFR': 'CFR',
#         'cognitive-fatigue-respiratory': 'CFR',
#     }
#     # 'Injury, Poisoning and Certain Other Consequences of External Causes']
#
#     organ_n = np.zeros(len(organ_list))
#     results_list = []
#     for i, organ in enumerate(organ_list):
#         print(i + 1, 'organ', organ)
#         for key, row in df_select.iterrows():
#             name = row['PASC Name Simple'].strip('*')
#             # if len(name.split()) >= 5:
#             #     name = ' '.join(name.split()[:4]) + '\n' + ' '.join(name.split()[4:])
#             if name == 'Dyspnea':
#                 name = 'Shortness of breath'
#             elif name == 'Death Overall':
#                 continue
#             elif name == 'Abnormal heartbeat':
#                 name = 'Dysrhythmia'
#             elif name == 'Diabetes mellitus':
#                 name = 'Diabetes'
#
#             pasc = row['pasc']
#             print(name, pasc)
#             hr = row['hr-w']
#             if pd.notna(row['hr-w-CI']):
#                 ci = stringlist_2_list(row['hr-w-CI'])
#             else:
#                 ci = [np.nan, np.nan]
#             p = row['hr-w-p']
#             # # p_format =
#             # if p <= 0.001:
#             #     sigsym = '$^{***}$'
#             #     p_format = '{:.1e}'.format(p)
#             # elif p <= 0.01:
#             #     sigsym = '$^{**}$'
#             #     p_format = '{:.3f}'.format(p)
#             # elif p <= 0.05:
#             #     sigsym = '$^{*}$'
#             #     p_format = '{:.3f}'.format(p)
#             # else:
#             #     sigsym = '$^{ns}$'
#             #     p_format = '{:.3f}'.format(p)
#             # p_format += sigsym
#
#             ahr_pformat, ahr_psym = pformat_symbol(p)
#
#             domain = row['Organ Domain']
#             cif1 = stringlist_2_list(row['cif_1_w'])[-1] * 100
#             cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1] * 100,
#                        stringlist_2_list(row['cif_1_w_CIUpper'])[-1] * 100]
#
#             # use nabs for ncum_ci_negative
#             cif0 = stringlist_2_list(row['cif_0_w'])[-1] * 100
#             cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1] * 100,
#                        stringlist_2_list(row['cif_0_w_CIUpper'])[-1] * 100]
#
#             cif_diff = stringlist_2_list(row['cif-w-diff-2'])[-1] * 100
#             cif_diff_ci = [stringlist_2_list(row['cif-w-diff-CILower'])[-1] * 100,
#                            stringlist_2_list(row['cif-w-diff-CIUpper'])[-1] * 100]
#             cif_diff_p = stringlist_2_list(row['cif-w-diff-p'])[-1]
#             cif_diff_pformat, cif_diff_psym = pformat_symbol(cif_diff_p)
#
#             result = [name, pasc, organ_mapname[organ],
#                       hr, '{:.2f}'.format(hr), p, '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
#                       '({:.2f},{:.2f})'.format(ci[0], ci[1]),
#                       cif1, cif1_ci[0], cif1_ci[1], '{:.2f}'.format(cif1),
#                       cif0, cif0_ci[0], cif0_ci[1], '{:.2f}'.format(cif0),
#                       ahr_pformat + ahr_psym, ahr_psym,
#                       cif_diff, '{:.2f}'.format(cif_diff), cif_diff_p,
#                       '{:.2f}'.format(cif_diff_ci[0]), '{:.2f}'.format(cif_diff_ci[1]),
#                       '({:.2f},{:.2f})'.format(cif_diff_ci[0], cif_diff_ci[1]),
#                       cif_diff_pformat + cif_diff_psym, cif_diff_psym
#                       ]
#
#             if domain == organ:
#                 results_list.append(result)
#
#     df_result = pd.DataFrame(results_list,
#                              columns=['name', 'pasc', 'group',
#                                       'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
#                                       'aHR-CI-str',
#                                       'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
#                                       'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
#                                       'p-val-sci', 'sigsym',
#                                       'cif_diff', 'cif_diff-str', 'cif_diff-p',
#                                       'cif_diff_cilower', 'cif_diff_ciupper', 'cif_diff-CI-str',
#                                       'cif_diff-p-format', 'cif_diff-p-symbol'])
#     # df_result['-aHR'] = -1 * df_result['aHR']
#
#     df_result = df_result.loc[~df_result['aHR'].isna()]
#     plt.rc('font', family='serif')
#     if show == 'full':
#         rightannote = ["aHR-str", "aHR-CI-str", 'p-val-sci',
#                        'cif_diff-str', 'cif_diff-CI-str', 'cif_diff-p-format',
#                        'CIF1-str', 'CIF0-str']
#         right_annoteheaders = ["aHR", "95% CI", "P-value",
#                                'CIF-DIFF (%)', '95% CI', 'P-Value',
#                                'CIF1', 'CIF0']
#     else:
#         rightannote = ["aHR-str", "aHR-CI-str", 'p-val-sci',
#                        'cif_diff-str', 'cif_diff-CI-str', 'cif_diff-p-format',
#                        ]
#         right_annoteheaders = ["aHR", "95% CI", "P-value",
#                                'CIF-DIFF (%)', '95% CI', 'P-Value',
#                                ]
#
#     # fig, ax = plt.subplots()
#     axs = fp.forestplot(
#         df_result,  # the dataframe with results data
#         figsize=(4, 12),
#         estimate="aHR",  # col containing estimated effect size
#         ll='aHR-lb',
#         hl='aHR-ub',  # lower & higher limits of conf. int.
#         varlabel="name",  # column containing the varlabels to be printed on far left
#         # capitalize="capitalize",  # Capitalize labels
#         pval="p-val",  # column containing p-values to be formatted
#         starpval=True,
#         annote=[],  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
#         # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
#         annoteheaders=[],
#         rightannote=rightannote,
#         # p_format, columns to report on right of plot
#         right_annoteheaders=right_annoteheaders,  # p_format, ^corresponding headers
#         groupvar="group",  # column containing group labels
#         group_order=df_result['group'].unique(),
#         xlabel="adjusted Hazard Ratio",  # x-label title
#         xticks=[0.4, 1, 1.2],  # x-ticks to be printed
#         color_alt_rows=True,
#         # flush=True,
#         sort=True,  # sort estimates in ascending order
#         # sortby='-aHR',
#         # table=True,  # Format as a table
#         # logscale=True,
#         # Additional kwargs for customizations
#         **{
#             "marker": "D",  # set maker symbol as diamond
#             "markersize": 35,  # adjust marker size
#             "xlinestyle": (0., (10, 5)),  # long dash for x-reference line
#             "xlinecolor": ".1",  # gray color for x-reference line
#             "xtick_size": 12,  # adjust x-ticker fontsize
#
#         },
#     )
#
#     axs.axvline(x=1, ymin=0, ymax=0.95, color='grey', linestyle='dashed')
#     check_and_mkdir(output_dir)
#     plt.savefig(output_dir + 'hr_moretabs-{}.png'.format(show), bbox_inches='tight', dpi=600)
#     plt.savefig(output_dir + 'hr_moretabs-{}.pdf'.format(show), bbox_inches='tight', transparent=True)
#
#     print('Done')
#     return df_result


def plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full'):

    output_dir = indir + r'figure/'

    if 'SSRI-overall-anxiety-ssri-base180-acutevsnot-mentalcovV2' in indir:
        df = pd.read_csv(indir + 'causal_effects_specific-snapshot-50.csv')
    elif 'SSRI-overall-depression-ssri-base180-acutevsnot-mentalcovV2' in indir:
        df = pd.read_csv(indir + 'causal_effects_specific-snapshot-50.csv')
    else:
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

    # df_select = df.sort_values(by='hr-w', ascending=True)
    df_select = df
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
        'Any CFR',
        'cognitive-fatigue-respiratory',
        'brainfog',

    ]

    organ_mapname = {
        'Any PASC': 'Overall in Postacute',
        'Death': 'Overall in Postacute',
        'Hospitalization': 'Overall in Postacute',
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
        'Any CFR': 'CFR',
        'cognitive-fatigue-respiratory': 'CFR Individuals',
        'brainfog' :'Brain Fog',

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
            elif name == 'Dysautonomia-Orthostatic':
                name = 'POTS'

            pasc = row['pasc']
            domain = row['Organ Domain']
            print(name, pasc, domain)

            if pasc == 'Headache' and domain == 'brainfog':
                print('headache already captured in neurologic domain, skip here in brainfog')
                continue

            hr = row['hr-w']
            if pd.notna(row['hr-w-CI']):
                ci = stringlist_2_list(row['hr-w-CI'])
            else:
                ci = [np.nan, np.nan]
            p = row['hr-w-p']
            # # p_format =
            # if p <= 0.001:
            #     sigsym = '$^{***}$'
            #     p_format = '{:.1e}'.format(p)
            # elif p <= 0.01:
            #     sigsym = '$^{**}$'
            #     p_format = '{:.3f}'.format(p)
            # elif p <= 0.05:
            #     sigsym = '$^{*}$'
            #     p_format = '{:.3f}'.format(p)
            # else:
            #     sigsym = '$^{ns}$'
            #     p_format = '{:.3f}'.format(p)
            # p_format += sigsym

            ahr_pformat, ahr_psym = pformat_symbol(p)


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

    # df_result = df_result.loc[~df_result['aHR'].isna()]
    df_result = df_result.loc[(1e-5 < df_result['aHR']) & (df_result['aHR'] < 100)].reset_index()
    df_result.loc[df_result['pasc']==r'Headache; including migraine', r'name'] = r'Headache '

    plt.rc('font', family='serif')
    if show == 'full':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       ]

        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100', 'P-Value',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in SSRI', 'CIF/100 in Ctrl']

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
                               'DIFF/100 (95% CI)',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in SSRI', 'CIF/100 in Ctrl']

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
        annote=leftannote,  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        annoteheaders=left_annoteheaders,  # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
        rightannote=rightannote,
        # p_format, columns to report on right of plot
        right_annoteheaders=right_annoteheaders,  # p_format, ^corresponding headers
        groupvar="group",  # column containing group labels
        group_order=df_result['group'].unique(),
        xlabel="Hazard Ratio",  # x-label title
        xticks=[0.5, 1, 2.],  #[0.1, 1, 10], #[0.3, 1, 2.5], #[0.5, 1, 2.],  # x-ticks to be printed  [0.5, 1, 35],  #
        color_alt_rows=True,
        # flush=True,
        # sort=False, #True,  # sort estimates in ascending order
        # sortby='-aHR',
        # table=True,  # Format as a table
        logscale= False, #True, #False, #True, #False, #True, #False, #True,
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
    plt.savefig(output_dir + 'hr_moretabs-{}-nosort.png'.format(show), bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_moretabs-{}-nosort.pdf'.format(show), bbox_inches='tight', transparent=True)

    print('Done')
    return df_result

def plot_forest_for_dx_organ_ssri_lib2_cifdiff_primary_v3(indir, drugname='CNS-ADHD', ctrlname='Non-CNS-ADHD', show='full'):
    # 2025-02-21 with any brain fog, revise control name,
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

    # df_select = df.sort_values(by='hr-w', ascending=True)
    df_select = df
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
        'Any CFR',
        'cognitive-fatigue-respiratory',
        'Any Brain Fog',
        'brainfog',

    ]

    organ_mapname = {
        'Any PASC': 'Overall in Postacute',
        'Death': 'Overall in Postacute',
        'Hospitalization': 'Overall in Postacute',
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
        'Any CFR': 'CFR',
        'cognitive-fatigue-respiratory': 'CFR Individuals',
        'Any Brain Fog': 'Brain Fog',
        'brainfog' :'Brain Fog Individuals',

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
            elif name == 'Dysautonomia-Orthostatic':
                name = 'POTS'

            pasc = row['pasc']
            domain = row['Organ Domain']
            print(name, pasc, domain)

            if pasc == 'Headache' and domain == 'brainfog':
                print('headache already captured in neurologic domain, skip here in brainfog')
                continue

            hr = row['hr-w']
            if pd.notna(row['hr-w-CI']):
                ci = stringlist_2_list(row['hr-w-CI'])
            else:
                ci = [np.nan, np.nan]
            p = row['hr-w-p']
            # # p_format =
            # if p <= 0.001:
            #     sigsym = '$^{***}$'
            #     p_format = '{:.1e}'.format(p)
            # elif p <= 0.01:
            #     sigsym = '$^{**}$'
            #     p_format = '{:.3f}'.format(p)
            # elif p <= 0.05:
            #     sigsym = '$^{*}$'
            #     p_format = '{:.3f}'.format(p)
            # else:
            #     sigsym = '$^{ns}$'
            #     p_format = '{:.3f}'.format(p)
            # p_format += sigsym

            ahr_pformat, ahr_psym = pformat_symbol(p)


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

    # df_result = df_result.loc[~df_result['aHR'].isna()]
    df_result = df_result.loc[(1e-5 < df_result['aHR']) & (df_result['aHR'] < 100)].reset_index()
    df_result.loc[df_result['pasc']==r'Headache; including migraine', r'name'] = r'Headache '

    plt.rc('font', family='serif')
    if show == 'full':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       ]

        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100', 'P-Value',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in CNS', 'CIF/100 in Non-CNS']

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
                               'DIFF/100 (95% CI)',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        # left_annoteheaders = ['CIF/100 in SSRI', 'CIF/100 in Non-SSRI']
        left_annoteheaders = ['CIF/100 in {}'.format(drugname), 'CIF/100 in {}'.format(ctrlname)]

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
        annote=leftannote,  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        annoteheaders=left_annoteheaders,  # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
        rightannote=rightannote,
        # p_format, columns to report on right of plot
        right_annoteheaders=right_annoteheaders,  # p_format, ^corresponding headers
        groupvar="group",  # column containing group labels
        group_order=df_result['group'].unique(),
        xlabel="Hazard Ratio",  # x-label title
        xticks=[0.1, 1, 10.],  #[0.1, 1, 5],  #[0.1, 1, 2.5],  #[0.5, 1, 2.],  #[0.1, 1, 10], #[0.3, 1, 2.5], #[0.5, 1, 2.],  # x-ticks to be printed  [0.5, 1, 35],  #
        color_alt_rows=True,
        # flush=True,
        # sort=False, #True,  # sort estimates in ascending order
        # sortby='-aHR',
        # table=True,  # Format as a table
        logscale= True, #True, #False, #True, #False, #True, #False, #True,
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
    # axs.set_xlim([0.1, 5])

    axs.axvline(x=1, ymin=0, ymax=0.95, color='grey', linestyle='dashed')
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'hr_moretabs-{}-nosort.png'.format(show), bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_moretabs-{}-nosort.pdf'.format(show), bbox_inches='tight', transparent=True)

    print('Done')
    return df_result


def plot_forest_for_dx_organ_ssri_lib2_cifdiff_primary_v3_individualSSRI(indir, drugname, ctrlname='Non-SSRI', show='full'):
    # 2025-02-21 with any brain fog, revise control name,
    output_dir = r'../data/recover/output/results/SSRI_individual_fig/' #+ r'figure/'

    if 'SSRI-overall-anxiety-ssri-base180-acutevsnot-mentalcovV3' in indir:
        df = pd.read_csv(indir + 'causal_effects_specific-snapshot-50.csv')
    elif 'SSRI-overall-depression-ssri-base180-acutevsnot-mentalcovV3' in indir:
        df = pd.read_csv(indir + 'causal_effects_specific-snapshot-50.csv')
    else:
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

    # df_select = df.sort_values(by='hr-w', ascending=True)
    df_select = df
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
        'Any CFR',
        'cognitive-fatigue-respiratory',
        'Any Brain Fog',
        'brainfog',

    ]

    organ_mapname = {
        'Any PASC': 'Overall in Postacute',
        'Death': 'Overall in Postacute',
        'Hospitalization': 'Overall in Postacute',
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
        'Any CFR': 'CFR',
        'cognitive-fatigue-respiratory': 'CFR Individuals',
        'Any Brain Fog': 'Brain Fog',
        'brainfog' :'Brain Fog Individuals',

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
            elif name == 'Dysautonomia-Orthostatic':
                name = 'POTS'

            pasc = row['pasc']
            domain = row['Organ Domain']
            print(name, pasc, domain)

            if pasc == 'Headache' and domain == 'brainfog':
                print('headache already captured in neurologic domain, skip here in brainfog')
                continue

            hr = row['hr-w']
            if pd.notna(row['hr-w-CI']):
                ci = stringlist_2_list(row['hr-w-CI'])
            else:
                ci = [np.nan, np.nan]
            p = row['hr-w-p']
            # # p_format =
            # if p <= 0.001:
            #     sigsym = '$^{***}$'
            #     p_format = '{:.1e}'.format(p)
            # elif p <= 0.01:
            #     sigsym = '$^{**}$'
            #     p_format = '{:.3f}'.format(p)
            # elif p <= 0.05:
            #     sigsym = '$^{*}$'
            #     p_format = '{:.3f}'.format(p)
            # else:
            #     sigsym = '$^{ns}$'
            #     p_format = '{:.3f}'.format(p)
            # p_format += sigsym

            ahr_pformat, ahr_psym = pformat_symbol(p)


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

    # df_result = df_result.loc[~df_result['aHR'].isna()]
    df_result = df_result.loc[(1e-5 < df_result['aHR']) & (df_result['aHR'] < 100)].reset_index()
    df_result.loc[df_result['pasc']==r'Headache; including migraine', r'name'] = r'Headache '

    plt.rc('font', family='serif')
    if show == 'full':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       ]

        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100', 'P-Value',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in SSRI', 'CIF/100 in Non-SSRI']

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
                               'DIFF/100 (95% CI)',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in {}'.format(drugname), 'CIF/100 in {}'.format(ctrlname)]

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
        annote=leftannote,  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        annoteheaders=left_annoteheaders,  # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
        rightannote=rightannote,
        # p_format, columns to report on right of plot
        right_annoteheaders=right_annoteheaders,  # p_format, ^corresponding headers
        groupvar="group",  # column containing group labels
        group_order=df_result['group'].unique(),
        xlabel="Hazard Ratio",  # x-label title
        xticks=[0.1, 1, 2.5],  #[0.5, 1, 2.],  #[0.1, 1, 10], #[0.3, 1, 2.5], #[0.5, 1, 2.],  # x-ticks to be printed  [0.5, 1, 35],  #
        color_alt_rows=True,
        # flush=True,
        # sort=False, #True,  # sort estimates in ascending order
        # sortby='-aHR',
        # table=True,  # Format as a table
        logscale= True, #False, #True, #False, #True, #False, #True, #False, #True,
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
    plt.savefig(output_dir + '{}-hr_moretabs-{}-nosort-{}.png'.format(drugname, show, drugname), bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + '{}-hr_moretabs-{}-nosort-{}.pdf'.format(drugname, show, drugname), bbox_inches='tight', transparent=True)

    print('Done')
    return df_result



def plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='full', ):
    subgroup_list = [
        'all',
        'female', 'male',
        'white', 'black',
        'less50', 'above50',
        # 'beforeomicron',
        # 'omicronbroad',
        'anxiety',
        'depression',
        'SMI',
        'nonSMI',
        #
        # 'PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease',
        # 'PaxRisk:Chronic liver disease',
        # 'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
        # 'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes',
        # 'PaxRisk:Disabilities',
        # 'PaxRisk:Heart conditions', 'PaxRisk:Hypertension',
        # 'PaxRisk:HIV infection',
        # 'PaxRisk:immune',
        # 'PaxRisk:Mental health conditions',
        # 'PaxRisk:Overweight and obesity', 'pregnant',  # 'PaxRisk:Pregnancy',
        # 'PaxRisk:Sickle cell disease or thalassemia',
        # 'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
        # 'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis',
        # 'pax1stwave',
        # 'pax2ndwave',
        # 'RUCA1@1',
        # 'RUCA1@2',
        # 'RUCA1@3',
        # 'RUCA1@4',
        # 'RUCA1@5',
        # 'RUCA1@6',
        # 'RUCA1@7',
        # 'RUCA1@8',
        # 'RUCA1@9',
        # 'RUCA1@10',
        # 'norisk',  # 'pregnant',
        # 'VA',
        # 'CFR'
    ]

    subgroup_info_map = {
        'all': ['Overall', pasc_outcome], #['all', 'Overall'],
        # 'norisk': ['Not at Risk', 'Without Indication'],
        # # 'pregnant': ['Pregnant', 'Without Indication'],
        # 'pregnant': ['Pregnant', 'With Risk Factor'],
        # 'VA': ['VA-like cohort', 'Sensitivity Analysis'],
        # 'CFR': ['Fatigue, Cognitive, Respiratory', 'Sensitivity Analysis'],

        # 'female': ['Female', 'Sex'],
        # 'male': ['Male', 'Sex'],
        # 'white': ['White', 'Race'],
        # 'black': ['Black', 'Race'],
        # 'less50': ['<50', 'Age'],
        # 'above50': ['≥50', 'Age'],
        # 'beforeomicron': ['Before Omicron', 'Infection Time'],
        # 'omicronbroad': ['Omicron and after', 'Infection Time'],
        # 'anxiety': ['Anxiety', 'Co-existing conditions'],
        # 'depression': ['Depression', 'Co-existing conditions'],
        # 'SMI': ['SMI', 'Co-existing conditions'],
        # 'nonSMI': ['non-SMI', 'Co-existing conditions'],

        'female': ['Female', 'Subgroup'],
        'male': ['Male', 'Subgroup'],
        'white': ['White', 'Subgroup'],
        'black': ['Black', 'Subgroup'],
        'less50': ['<50', 'Subgroup'],
        'above50': ['≥50', 'Subgroup'],
        'beforeomicron': ['Before Omicron', 'Subgroup'],
        'omicronbroad': ['Omicron and after', 'Subgroup'],
        'anxiety': ['Anxiety', 'Subgroup'],
        'depression': ['Depression', 'Subgroup'],
        'SMI': ['SMI', 'Subgroup'],
        'nonSMI': ['non-SMI', 'Subgroup'],

        # 'PaxRisk:Cancer': ['Cancer', 'With Risk Factor'],
        # 'PaxRisk:Chronic kidney disease': ['Chronic Kidney Disease', 'With Risk Factor'],
        # 'PaxRisk:Chronic liver disease': ['Chronic Liver Disease', 'With Risk Factor'],
        # 'PaxRisk:Chronic lung disease': ['Chronic Lung Disease', 'With Risk Factor'],
        # 'PaxRisk:Cystic fibrosis': ['Cystic Fibrosis', 'With Risk Factor'],
        # 'PaxRisk:Dementia or other neurological conditions': ['Dementia or Neurological', 'With Risk Factor'],
        # 'PaxRisk:Diabetes': ['Diabetes', 'With Risk Factor'],
        # 'PaxRisk:Disabilities': ['Disabilities', 'With Risk Factor'],
        # 'PaxRisk:Heart conditions': ['Heart Conditions', 'With Risk Factor'],
        # 'PaxRisk:Hypertension': ['Hypertension', 'With Risk Factor'],
        # 'PaxRisk:HIV infection': ['HIV', 'With Risk Factor'],
        # 'PaxRisk:immune': ['Immune Dysfunction', 'With Risk Factor'],
        # 'PaxRisk:Mental health conditions': ['Mental Health', 'With Risk Factor'],
        # 'PaxRisk:Overweight and obesity': ['Overweight and Obesity', 'With Risk Factor'],  # 'PaxRisk:Pregnancy',
        # 'PaxRisk:Sickle cell disease or thalassemia': ['Sickle Cell or Anemia', 'With Risk Factor'],
        # 'PaxRisk:Smoking current': ['Smoker current or former', 'With Risk Factor'],
        # 'PaxRisk:Stroke or cerebrovascular disease': ['Stroke or Cerebrovascular', 'With Risk Factor'],
        # 'PaxRisk:Substance use disorders': ['Substance Use Disorders', 'With Risk Factor'],
        # 'PaxRisk:Tuberculosis': ['Tuberculosis', 'With Risk Factor'],
        # 'RUCA1@1': ['Metropolitan core', 'Rural-Urban Commuting'],
        # 'RUCA1@2': ['Metropolitan high commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@3': ['Metropolitan low commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@4': ['Micropolitan core', 'Rural-Urban Commuting'],
        # 'RUCA1@5': ['Micropolitan high commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@6': ['Micropolitan low commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@7': ['Small town core', 'Rural-Urban Commuting'],
        # 'RUCA1@8': ['Small town high commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@9': ['Small town low commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@10': ['Rural areas', 'Rural-Urban Commuting'],

    }
    if pasc_outcome == 'any_pasc':
        subgroup_info_map['all'] =  ['Overall', 'Any PASC']
    if pasc_outcome == 'death_postacute':
        subgroup_info_map['all'] =  ['Overall', 'All-cause mortality']
    elif pasc_outcome == 'cvddeath_postacute':
        subgroup_info_map['all'] = ['Overall', 'CVD mortality']
    elif pasc_outcome == 'PASC-General':
        subgroup_info_map['all'] = ['Overall', 'U099/B948']
    elif pasc_outcome == 'ME/CFS':
        subgroup_info_map['all'] = ['Overall', 'ME/CFS']
    elif pasc_outcome == 'hospitalization_postacute':
        subgroup_info_map['all'] = ['Overall', 'Hospitalization']
    elif pasc_outcome == 'any_CFR':
        subgroup_info_map['all'] = ['Overall', 'Any CFR']


    pasc_simname_organ = load_pasc_info()

    output_dir = r'../data/recover/output/results/SSRI_figure_subgroup_v2/' #+ pasc_outcome + '/'

    results_list = []
    for subgroup in subgroup_list:
        indir = r'../data/recover/output/results/SSRI-overall-{}-ssri-base180-acutevsnot-mentalcovV2/'.format(
            subgroup.replace(':', '_').replace('/', '-').replace(' ', '_')
        )
        info = subgroup_info_map[subgroup]
        subgroupname = info[0]
        grouplabel = info[1]


        # # subgroupname = subgroup.split(':')[-1]
        # if subgroup == 'norisk':
        #     indir = r'../data/recover/output/results/Paxlovid-norisk-all-pcornet-V3/'
        #     # subgroupname = 'No documented risk factors'
        # elif subgroup == 'pregnant':
        #     indir = r'../data/recover/output/results/Paxlovid-pregnant-all-pcornet-V3/'
        #     # subgroupname = 'Pregnant'
        # # elif subgroup == 'CFR':
        # #     indir = r'../data/recover/output/results/Paxlovid-atrisk-all-pcornet-V3/'
        # elif subgroup == 'all':
        #     indir = r'../data/recover/output/results/Paxlovid-atrisk-all-pcornet-V3/'

        print('read:', indir)
        # if subgroup == 'PaxRisk:Dementia or other neurological conditions':
        #     df = pd.read_csv(indir + 'causal_effects_specific-snapshot-18.csv')
        #     subgroupname = 'Dementia or other neurological'
        # else:
        #     df = pd.read_csv(indir + 'causal_effects_specific.csv')

        # if subgroup == 'CFR':
        #     df = pd.read_csv(indir + 'causal_effects_specific.csv')
        #     row = df.loc[df['pasc'] == 'any_CFR', :].squeeze()
        #     name = 'Any CFR'
        # else:
        #     df = pd.read_csv(indir + 'causal_effects_specific-snapshot-2.csv')
        #     row = df.loc[df['pasc'] == 'any_pasc', :].squeeze()
        #     name = 'Any PASC'

        try:
            df = pd.read_csv(indir + 'causal_effects_specific.csv')
        except:
            df = pd.read_csv(indir + 'causal_effects_specific-snapshot-20.csv')

        df.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )

        row = df.loc[df['pasc'] == pasc_outcome, :].squeeze()
        name = pasc_simname_organ[pasc_outcome][0]  # 'Any PASC'

        if subgroup == 'all':
            N1 = row['case+']
            N0 = row['ctrl-']
            print('N1:', N1, 'N0:', N0)

        pasc = row['pasc']
        if row['case+'] < 500:
            print(subgroup, 'is very small (<500 in exposed), skip', row['case+'], row['ctrl-'])
            # continue
        # print(name, pasc)
        hr = row['hr-w']
        if (hr <= 0.001) or pd.isna(hr):
            print('HR is very small <0.001, ', hr, 'in subgroup', subgroup)
            # continue

        if pd.notna(row['hr-w-CI']):
            ci = stringlist_2_list(row['hr-w-CI'])
        else:
            ci = [np.nan, np.nan]
        p = row['hr-w-p']
        # p_format =
        # if p <= 0.001:
        #     sigsym = '$^{***}$'
        #     p_format = '{:.1e}'.format(p)
        # elif p <= 0.01:
        #     sigsym = '$^{**}$'
        #     p_format = '{:.3f}'.format(p)
        # elif p <= 0.05:
        #     sigsym = '$^{*}$'
        #     p_format = '{:.3f}'.format(p)
        # else:
        #     sigsym = '$^{ns}$'
        #     p_format = '{:.3f}'.format(p)
        # p_format += sigsym

        ahr_pformat, ahr_psym = pformat_symbol(p)

        # domain = row['Organ Domain']
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

        result = [subgroupname, grouplabel, name, pasc,
                  '{:.0f} ({:.1f}%)'.format(row['case+'], row['case+']/N1*100),
                  '{:.0f} ({:.1f}%)'.format(row['ctrl-'], row['ctrl-']/N0*100),
                  hr, '{:.2f} ({:.2f}, {:.2f})'.format(hr, ci[0], ci[1]), p,
                  '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
                  '({:.2f},{:.2f})'.format(ci[0], ci[1]),
                  cif1, cif1_ci[0], cif1_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif1, cif1_ci[0], cif1_ci[1]),
                  cif0, cif0_ci[0], cif0_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif0, cif0_ci[0], cif0_ci[1]),
                  ahr_pformat + ahr_psym, ahr_psym,
                  cif_diff, '{:.2f} ({:.2f}, {:.2f})'.format(cif_diff, cif_diff_ci[0], cif_diff_ci[1]), cif_diff_p,
                  '{:.2f}'.format(cif_diff_ci[0]), '{:.2f}'.format(cif_diff_ci[1]),
                  '({:.2f},{:.2f})'.format(cif_diff_ci[0], cif_diff_ci[1]),
                  cif_diff_pformat + cif_diff_psym, cif_diff_psym]

        results_list.append(result)

    df_result = pd.DataFrame(results_list,
                             columns=['subgroup', 'grouplabel', 'name', 'pasc', 'No. in 1', 'No. in 0',
                                      'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
                                      'aHR-CI-str',
                                      'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
                                      'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
                                      'p-val-sci', 'sigsym',
                                      'cif_diff', 'cif_diff-str', 'cif_diff-p',
                                      'cif_diff_cilower', 'cif_diff_ciupper', 'cif_diff-CI-str',
                                      'cif_diff-p-format', 'cif_diff-p-symbol'])

    df_result['-aHR'] = -1 * df_result['aHR']
    df_result['name'] = df_result['name'] + '-' + df_result['subgroup']
    df_result['pasc'] = df_result['pasc'] + '-' + df_result['subgroup']

    plt.rc('font', family='serif')
    # fig, ax = plt.subplots()
    if show == 'full':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       'No. in 1', 'No. in 0',
                       ]

        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100', 'P-Value',
                               'SSRI N=', 'Ctrl N=',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in SSRI', 'CIF/100 in Non-SSRI']

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

        leftannote = ['No. in 1', 'No. in 0', 'CIF1-str', 'CIF0-str']
        left_annoteheaders = ['SSRI N=', 'Ctrl N=', 'CIF/100 in SSRI', 'CIF/100 in Non-SSRI']

    if pasc_outcome == 'death_postacute':
        xticks = [0.1, 1, 10]
        logscale = True
        subgroup_info_map['all'] =  ['Overall', 'All-cause mortality']

    elif pasc_outcome == 'cvddeath_postacute':
        xticks = [0.1, 1, 10]
        logscale = True
        subgroup_info_map['all'] = ['Overall', 'CVD mortality']
    elif pasc_outcome == 'PASC-General':
        xticks = [0.2, 1, 2.5]
        logscale = False
        subgroup_info_map['all'] = ['Overall', 'U099/B948']
    elif pasc_outcome == 'ME/CFS':
        xticks = [0.2, 1, 2.5]
        logscale = False
    elif pasc_outcome == 'hospitalization_postacute':
        xticks = [0.5, 1, 2.]
        logscale = False
        subgroup_info_map['all'] = ['Overall', 'Hospitalization']
    elif pasc_outcome == 'any_CFR':
        xticks = [0.5, 1, 2.]
        logscale = False
        subgroup_info_map['all'] = ['Overall', 'Any CFR']

    else:
        xticks = [0.5, 1, 2.]
        logscale = False

    # xticks = [0.5, 1, 2.]

    axs = fp.forestplot(
        df_result,  # the dataframe with results data
        figsize=(3.5, 10),  # (7, 12), #(6, 10), # (4.5, 13)
        estimate="aHR",  # col containing estimated effect size
        ll='aHR-lb',
        hl='aHR-ub',  # lower & higher limits of conf. int.
        varlabel="subgroup",  # column containing the varlabels to be printed on far left
        # capitalize="capitalize",  # Capitalize labels
        pval="p-val",  # column containing p-values to be formatted
        starpval=True,
        annote=leftannote,  # ['No. in 1', 'No. in 0', ],  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        annoteheaders=left_annoteheaders,  # ['No.Paxlovid', 'No.Ctrl', ],
        rightannote=rightannote,  # ["aHR-str", "aHR-CI-str", 'p-val-sci', 'CIF1-str', 'CIF0-str'],
        # p_format, columns to report on right of plot
        right_annoteheaders=right_annoteheaders,
        groupvar="grouplabel",  # column containing group labels
        # group_order=df_result['group'].unique(),
        xlabel="Hazard Ratio",  # x-label title
        xticks=xticks,  # [0.6, 1, 1.5],  # x-ticks to be printed
        color_alt_rows=True,
        # flush=True,
        # sort=True,  # sort estimates in ascending order
        # sortby='-aHR',
        # table=True,  # Format as a table
        logscale=logscale,
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
    def _clean_name(x):
        return x.replace(':', '_').replace('/', '-').replace(' ', '_')

    plt.savefig(output_dir + 'hr_subgroup-{}-{}-v2.png'.format(_clean_name(pasc_outcome), show), bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_subgroup-{}-{}-v2.pdf'.format(_clean_name(pasc_outcome), show), bbox_inches='tight', transparent=True)

    print('Done')
    return df_result

def plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcomeV2(pasc_outcome, show='full', ):
    # 2025-02-21 rerun, with brain fog, change v2 --> v3, change labels, add hispanic

    subgroup_list = [
        'all',
        'female', 'male',
        'white', 'black', 'hispanic', #'nonhispanic',
        'less50', 'above50',
        # 'beforeomicron',
        # 'omicronbroad',
        'anxiety',
        'depression',
        'SMI',
        # 'nonSMI',
        #
        # 'PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease',
        # 'PaxRisk:Chronic liver disease',
        # 'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
        # 'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes',
        # 'PaxRisk:Disabilities',
        # 'PaxRisk:Heart conditions', 'PaxRisk:Hypertension',
        # 'PaxRisk:HIV infection',
        # 'PaxRisk:immune',
        # 'PaxRisk:Mental health conditions',
        # 'PaxRisk:Overweight and obesity', 'pregnant',  # 'PaxRisk:Pregnancy',
        # 'PaxRisk:Sickle cell disease or thalassemia',
        # 'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
        # 'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis',
        # 'pax1stwave',
        # 'pax2ndwave',
        # 'RUCA1@1',
        # 'RUCA1@2',
        # 'RUCA1@3',
        # 'RUCA1@4',
        # 'RUCA1@5',
        # 'RUCA1@6',
        # 'RUCA1@7',
        # 'RUCA1@8',
        # 'RUCA1@9',
        # 'RUCA1@10',
        # 'norisk',  # 'pregnant',
        # 'VA',
        # 'CFR'
    ]

    subgroup_info_map = {
        'all': ['Overall', pasc_outcome], #['all', 'Overall'],
        # 'norisk': ['Not at Risk', 'Without Indication'],
        # # 'pregnant': ['Pregnant', 'Without Indication'],
        # 'pregnant': ['Pregnant', 'With Risk Factor'],
        # 'VA': ['VA-like cohort', 'Sensitivity Analysis'],
        # 'CFR': ['Fatigue, Cognitive, Respiratory', 'Sensitivity Analysis'],

        # 'female': ['Female', 'Sex'],
        # 'male': ['Male', 'Sex'],
        # 'white': ['White', 'Race'],
        # 'black': ['Black', 'Race'],
        # 'less50': ['<50', 'Age'],
        # 'above50': ['≥50', 'Age'],
        # 'beforeomicron': ['Before Omicron', 'Infection Time'],
        # 'omicronbroad': ['Omicron and after', 'Infection Time'],
        # 'anxiety': ['Anxiety', 'Co-existing conditions'],
        # 'depression': ['Depression', 'Co-existing conditions'],
        # 'SMI': ['SMI', 'Co-existing conditions'],
        # 'nonSMI': ['non-SMI', 'Co-existing conditions'],

        'female': ['Female', 'Subgroup'],
        'male': ['Male', 'Subgroup'],
        'white': ['White', 'Subgroup'],
        'black': ['Black', 'Subgroup'],
        'hispanic': ['Hispanic', 'Subgroup'],
        'nonhispanic': ['Nonhispanic', 'Subgroup'],
        'less50': ['<50', 'Subgroup'],
        'above50': ['≥50', 'Subgroup'],
        'beforeomicron': ['Before Omicron', 'Subgroup'],
        'omicronbroad': ['Omicron and after', 'Subgroup'],
        'anxiety': ['Anxiety', 'Subgroup'],
        'depression': ['Depression', 'Subgroup'],
        'SMI': ['SMI', 'Subgroup'],
        'nonSMI': ['Non-SMI', 'Subgroup'],

        # 'PaxRisk:Cancer': ['Cancer', 'With Risk Factor'],
        # 'PaxRisk:Chronic kidney disease': ['Chronic Kidney Disease', 'With Risk Factor'],
        # 'PaxRisk:Chronic liver disease': ['Chronic Liver Disease', 'With Risk Factor'],
        # 'PaxRisk:Chronic lung disease': ['Chronic Lung Disease', 'With Risk Factor'],
        # 'PaxRisk:Cystic fibrosis': ['Cystic Fibrosis', 'With Risk Factor'],
        # 'PaxRisk:Dementia or other neurological conditions': ['Dementia or Neurological', 'With Risk Factor'],
        # 'PaxRisk:Diabetes': ['Diabetes', 'With Risk Factor'],
        # 'PaxRisk:Disabilities': ['Disabilities', 'With Risk Factor'],
        # 'PaxRisk:Heart conditions': ['Heart Conditions', 'With Risk Factor'],
        # 'PaxRisk:Hypertension': ['Hypertension', 'With Risk Factor'],
        # 'PaxRisk:HIV infection': ['HIV', 'With Risk Factor'],
        # 'PaxRisk:immune': ['Immune Dysfunction', 'With Risk Factor'],
        # 'PaxRisk:Mental health conditions': ['Mental Health', 'With Risk Factor'],
        # 'PaxRisk:Overweight and obesity': ['Overweight and Obesity', 'With Risk Factor'],  # 'PaxRisk:Pregnancy',
        # 'PaxRisk:Sickle cell disease or thalassemia': ['Sickle Cell or Anemia', 'With Risk Factor'],
        # 'PaxRisk:Smoking current': ['Smoker current or former', 'With Risk Factor'],
        # 'PaxRisk:Stroke or cerebrovascular disease': ['Stroke or Cerebrovascular', 'With Risk Factor'],
        # 'PaxRisk:Substance use disorders': ['Substance Use Disorders', 'With Risk Factor'],
        # 'PaxRisk:Tuberculosis': ['Tuberculosis', 'With Risk Factor'],
        # 'RUCA1@1': ['Metropolitan core', 'Rural-Urban Commuting'],
        # 'RUCA1@2': ['Metropolitan high commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@3': ['Metropolitan low commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@4': ['Micropolitan core', 'Rural-Urban Commuting'],
        # 'RUCA1@5': ['Micropolitan high commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@6': ['Micropolitan low commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@7': ['Small town core', 'Rural-Urban Commuting'],
        # 'RUCA1@8': ['Small town high commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@9': ['Small town low commuting', 'Rural-Urban Commuting'],
        # 'RUCA1@10': ['Rural areas', 'Rural-Urban Commuting'],

    }
    if pasc_outcome == 'any_pasc':
        subgroup_info_map['all'] =  ['Overall', 'Any PASC']
    if pasc_outcome == 'death_postacute':
        subgroup_info_map['all'] =  ['Overall', 'All-cause mortality']
    elif pasc_outcome == 'cvddeath_postacute':
        subgroup_info_map['all'] = ['Overall', 'CVD mortality']
    elif pasc_outcome == 'PASC-General':
        subgroup_info_map['all'] = ['Overall', 'U099/B948']
    elif pasc_outcome == 'ME/CFS':
        subgroup_info_map['all'] = ['Overall', 'ME/CFS']
    elif pasc_outcome == 'hospitalization_postacute':
        subgroup_info_map['all'] = ['Overall', 'Hospitalization']
    elif pasc_outcome == 'any_CFR':
        subgroup_info_map['all'] = ['Overall', 'Any CFR']
    elif pasc_outcome == 'any_brainfog':
        subgroup_info_map['all'] = ['Overall', 'Any Brain Fog']


    pasc_simname_organ = load_pasc_info()

    # output_dir = r'../data/recover/output/results/SSRI_figure_subgroup_v2/' #+ pasc_outcome + '/'
    # 2025-02-21 rerun, with brain fog, change v2 --> v3
    output_dir = r'../data/recover/output/results/SSRI_figure_subgroup_v3/' #+ pasc_outcome + '/'

    results_list = []
    for subgroup in subgroup_list:
        indir = r'../data/recover/output/results/SSRI-overall-{}-ssri-base180-acutevsnot-mentalcovV3/'.format(
            subgroup.replace(':', '_').replace('/', '-').replace(' ', '_')
        )
        info = subgroup_info_map[subgroup]
        subgroupname = info[0]
        grouplabel = info[1]


        # # subgroupname = subgroup.split(':')[-1]
        # if subgroup == 'norisk':
        #     indir = r'../data/recover/output/results/Paxlovid-norisk-all-pcornet-V3/'
        #     # subgroupname = 'No documented risk factors'
        # elif subgroup == 'pregnant':
        #     indir = r'../data/recover/output/results/Paxlovid-pregnant-all-pcornet-V3/'
        #     # subgroupname = 'Pregnant'
        # # elif subgroup == 'CFR':
        # #     indir = r'../data/recover/output/results/Paxlovid-atrisk-all-pcornet-V3/'
        # elif subgroup == 'all':
        #     indir = r'../data/recover/output/results/Paxlovid-atrisk-all-pcornet-V3/'

        print('read:', indir)
        # if subgroup == 'PaxRisk:Dementia or other neurological conditions':
        #     df = pd.read_csv(indir + 'causal_effects_specific-snapshot-18.csv')
        #     subgroupname = 'Dementia or other neurological'
        # else:
        #     df = pd.read_csv(indir + 'causal_effects_specific.csv')

        # if subgroup == 'CFR':
        #     df = pd.read_csv(indir + 'causal_effects_specific.csv')
        #     row = df.loc[df['pasc'] == 'any_CFR', :].squeeze()
        #     name = 'Any CFR'
        # else:
        #     df = pd.read_csv(indir + 'causal_effects_specific-snapshot-2.csv')
        #     row = df.loc[df['pasc'] == 'any_pasc', :].squeeze()
        #     name = 'Any PASC'

        try:
            df = pd.read_csv(indir + 'causal_effects_specific.csv')
        except:
            df = pd.read_csv(indir + 'causal_effects_specific-snapshot-20.csv')

        df.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )

        row = df.loc[df['pasc'] == pasc_outcome, :].squeeze()
        name = pasc_simname_organ[pasc_outcome][0]  # 'Any PASC'

        if subgroup == 'all':
            N1 = row['case+']
            N0 = row['ctrl-']
            print('N1:', N1, 'N0:', N0)

        pasc = row['pasc']
        if row['case+'] < 500:
            print(subgroup, 'is very small (<500 in exposed), skip', row['case+'], row['ctrl-'])
            # continue
        # print(name, pasc)
        hr = row['hr-w']
        if (hr <= 0.001) or pd.isna(hr):
            print('HR is very small <0.001, ', hr, 'in subgroup', subgroup)
            # continue

        if pd.notna(row['hr-w-CI']):
            ci = stringlist_2_list(row['hr-w-CI'])
        else:
            ci = [np.nan, np.nan]
        p = row['hr-w-p']
        # p_format =
        # if p <= 0.001:
        #     sigsym = '$^{***}$'
        #     p_format = '{:.1e}'.format(p)
        # elif p <= 0.01:
        #     sigsym = '$^{**}$'
        #     p_format = '{:.3f}'.format(p)
        # elif p <= 0.05:
        #     sigsym = '$^{*}$'
        #     p_format = '{:.3f}'.format(p)
        # else:
        #     sigsym = '$^{ns}$'
        #     p_format = '{:.3f}'.format(p)
        # p_format += sigsym

        ahr_pformat, ahr_psym = pformat_symbol(p)

        # domain = row['Organ Domain']
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

        result = [subgroupname, grouplabel, name, pasc,
                  '{:.0f} ({:.1f}%)'.format(row['case+'], row['case+']/N1*100),
                  '{:.0f} ({:.1f}%)'.format(row['ctrl-'], row['ctrl-']/N0*100),
                  hr, '{:.2f} ({:.2f}, {:.2f})'.format(hr, ci[0], ci[1]), p,
                  '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
                  '({:.2f},{:.2f})'.format(ci[0], ci[1]),
                  cif1, cif1_ci[0], cif1_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif1, cif1_ci[0], cif1_ci[1]),
                  cif0, cif0_ci[0], cif0_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif0, cif0_ci[0], cif0_ci[1]),
                  ahr_pformat + ahr_psym, ahr_psym,
                  cif_diff, '{:.2f} ({:.2f}, {:.2f})'.format(cif_diff, cif_diff_ci[0], cif_diff_ci[1]), cif_diff_p,
                  '{:.2f}'.format(cif_diff_ci[0]), '{:.2f}'.format(cif_diff_ci[1]),
                  '({:.2f},{:.2f})'.format(cif_diff_ci[0], cif_diff_ci[1]),
                  cif_diff_pformat + cif_diff_psym, cif_diff_psym]

        results_list.append(result)

    df_result = pd.DataFrame(results_list,
                             columns=['subgroup', 'grouplabel', 'name', 'pasc', 'No. in 1', 'No. in 0',
                                      'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
                                      'aHR-CI-str',
                                      'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
                                      'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
                                      'p-val-sci', 'sigsym',
                                      'cif_diff', 'cif_diff-str', 'cif_diff-p',
                                      'cif_diff_cilower', 'cif_diff_ciupper', 'cif_diff-CI-str',
                                      'cif_diff-p-format', 'cif_diff-p-symbol'])

    # df_result['-aHR'] = -1 * df_result['aHR']
    # df_result['name'] = df_result['name'] + '-' + df_result['subgroup']
    # df_result['pasc'] = df_result['pasc'] + '-' + df_result['subgroup']

    plt.rc('font', family='serif')
    # fig, ax = plt.subplots()
    if show == 'full':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       'No. in 1', 'No. in 0',
                       ]

        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100', 'P-Value',
                               'SSRI N=', 'Ctrl N=',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in SSRI', 'CIF/100 in Non-SSRI']

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

        leftannote = ['No. in 1', 'No. in 0', 'CIF1-str', 'CIF0-str']
        left_annoteheaders = ['SSRI N=', 'Non-SSRI N=', 'CIF/100 in SSRI', 'CIF/100 in Non-SSRI']

    if pasc_outcome == 'death_postacute':
        xticks = [0.1, 1, 10]
        logscale = True
        # subgroup_info_map['all'] =  ['Overall', 'All-cause mortality']

    elif pasc_outcome == 'cvddeath_postacute':
        xticks = [0.1, 1, 10]
        logscale = True
        # subgroup_info_map['all'] = ['Overall', 'CVD mortality']
    elif pasc_outcome == 'PASC-General':
        xticks = [0.2, 1, 2.5]
        logscale = False
        # subgroup_info_map['all'] = ['Overall', 'U099/B948']
    elif pasc_outcome == 'ME/CFS':
        xticks = [0.2, 1, 2.5]
        logscale = False
    elif pasc_outcome == 'hospitalization_postacute':
        xticks = [0.5, 1, 2.]
        logscale = False
        # subgroup_info_map['all'] = ['Overall', 'Hospitalization']
    elif pasc_outcome == 'any_CFR':
        xticks = [0.5, 1, 2.]
        logscale = False
        # subgroup_info_map['all'] = ['Overall', 'Any CFR']
    elif pasc_outcome == 'any_pasc':
        xticks = [0.5, 1, 2.]
        logscale = False
        # subgroup_info_map['all'] = ['Overall', 'Any PASC']
    else:
        xticks = [0.5, 1, 2.]
        logscale = False

    # xticks = [0.5, 1, 2.]

    axs = fp.forestplot(
        df_result,  # the dataframe with results data
        figsize=(3.5, 10),  # (7, 12), #(6, 10), # (4.5, 13)
        estimate="aHR",  # col containing estimated effect size
        ll='aHR-lb',
        hl='aHR-ub',  # lower & higher limits of conf. int.
        varlabel="subgroup",  # column containing the varlabels to be printed on far left
        # capitalize="capitalize",  # Capitalize labels
        pval="p-val",  # column containing p-values to be formatted
        starpval=True,
        annote=leftannote,  # ['No. in 1', 'No. in 0', ],  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        annoteheaders=left_annoteheaders,  # ['No.Paxlovid', 'No.Ctrl', ],
        rightannote=rightannote,  # ["aHR-str", "aHR-CI-str", 'p-val-sci', 'CIF1-str', 'CIF0-str'],
        # p_format, columns to report on right of plot
        right_annoteheaders=right_annoteheaders,
        groupvar="grouplabel",  # column containing group labels
        group_order=df_result['grouplabel'].unique(),
        xlabel="Hazard Ratio",  # x-label title
        xticks=xticks,  # [0.6, 1, 1.5],  # x-ticks to be printed
        color_alt_rows=True,
        # flush=True,
        # sort=True,  # sort estimates in ascending order
        # sortby='-aHR',
        # table=True,  # Format as a table
        logscale=logscale,
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
    def _clean_name(x):
        return x.replace(':', '_').replace('/', '-').replace(' ', '_')

    plt.savefig(output_dir + 'hr_subgroup-{}-{}-v2.png'.format(_clean_name(pasc_outcome), show), bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_subgroup-{}-{}-v2.pdf'.format(_clean_name(pasc_outcome), show), bbox_inches='tight', transparent=True)

    print('Done')
    return df_result


def summarize_CI_from_primary_and_boostrap_cifdiff():
    indir = r'../data/recover/output/results/Paxlovid-atrisk-all-pcornet-V3/'
    indir2 = r'../data/recover/output/results/Paxlovid-atrisk-all-pcornet-boostrap/'

    output_dir = indir2 + r'compareboostrap/'

    df = pd.read_csv(indir + 'causal_effects_specific.csv')
    df.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )
    pasc_simname_organ = load_pasc_info()
    df.insert(df.columns.get_loc('pasc') + 1, 'Organ Domain', '')
    df.insert(df.columns.get_loc('pasc') + 1, 'PASC Name Simple', '')

    df2 = pd.read_csv(indir2 + 'causal_effects_specific-snapshot-44.csv')
    df2.drop_duplicates(subset=['pasc'], keep='last', inplace=True, )
    df2.insert(df2.columns.get_loc('pasc') + 1, 'Organ Domain', '')
    df2.insert(df2.columns.get_loc('pasc') + 1, 'PASC Name Simple', '')

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
            if pd.notna(row['hr-w-CI']):
                ci = stringlist_2_list(row['hr-w-CI'])
            else:
                ci = [np.nan, np.nan]
            p = row['hr-w-p']

            ahr_pformat, ahr_psym = pformat_symbol(p)

            domain = row['Organ Domain']
            if domain != organ:
                continue

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

            row2 = df2.loc[df2['pasc'] == pasc, :].squeeze()
            cif1_boost = (row2['cif_1_w']) * 100
            cif1_ci_boost = [(row2['cif_1_w_CILower']) * 100,
                             (row2['cif_1_w_CIUpper']) * 100]

            cif0_boost = (row2['cif_0_w']) * 100
            cif0_ci_boost = [(row2['cif_0_w_CILower']) * 100,
                             (row2['cif_0_w_CIUpper']) * 100]

            cif_diff_boost = (row2['cif-w-diff-2']) * 100
            cif_diff_ci_boost = [(row2['cif-w-diff-CILower'])* 100,
                                 (row2['cif-w-diff-CIUpper']) * 100]
            cif_diff_p_boost = (row2['cif-w-diff-p'])
            cif_diff_pformat_boost, cif_diff_psym_boost = pformat_symbol(cif_diff_p_boost)

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
                      cif_diff_pformat , cif_diff_psym,
                      cif1_boost, cif1_ci_boost[0], cif1_ci_boost[1],
                      '{:.2f} ({:.2f}, {:.2f})'.format(cif1_boost, cif1_ci_boost[0], cif1_ci_boost[1]),
                      cif0_boost, cif0_ci_boost[0], cif0_ci_boost[1],
                      '{:.2f} ({:.2f}, {:.2f})'.format(cif0_boost, cif0_ci_boost[0], cif0_ci_boost[1]),
                      cif_diff_boost, '{:.2f} ({:.2f}, {:.2f})'.format(cif_diff_boost, cif_diff_ci_boost[0], cif_diff_ci_boost[1]),
                      cif_diff_p_boost,
                      '{:.2f}'.format(cif_diff_ci_boost[0]), '{:.2f}'.format(cif_diff_ci_boost[1]),
                      '({:.2f},{:.2f})'.format(cif_diff_ci_boost[0], cif_diff_ci_boost[1]),
                      cif_diff_pformat_boost , cif_diff_psym_boost,
                      ]


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
                                      'cif_diff-p-format', 'cif_diff-p-symbol',
                                      'CIF1_boost', 'CIF1-lb_boost', 'CIF1-ub_boost', 'CIF1-str_boost',
                                      'CIF0_boost', 'CIF0-lb_boost', 'CIF0-ub_boost', 'CIF0-str_boost',
                                      'cif_diff_boost', 'cif_diff-str_boost', 'cif_diff-p_boost',
                                      'cif_diff_cilower_boost', 'cif_diff_ciupper_boost', 'cif_diff-CI-str_boost',
                                      'cif_diff-p-format_boost', 'cif_diff-p-symbol_boost'
                                      ])
    # df_result['-aHR'] = -1 * df_result['aHR']
    check_and_mkdir(output_dir)
    df_result.to_csv(output_dir + 'cif-boostrap_comparison.csv')
    # df_result = df_result.loc[~df_result['aHR'].isna()]
    # plt.rc('font', family='serif')
    # if show == 'full':
    #     rightannote = ["aHR-str", 'p-val-sci',
    #                    'cif_diff-str', 'cif_diff-p-format',
    #                    ]
    #
    #     right_annoteheaders = ["HR (95% CI)", "P-value",
    #                            'DIFF/100', 'P-Value',
    #                            ]
    #
    #     leftannote = ['CIF1-str', 'CIF0-str']
    #     left_annoteheaders = ['CIF/100 in Paxlovid', 'CIF/100 in Ctrl']
    #
    # elif show == 'short':
    #     rightannote = ["aHR-str", 'p-val-sci',
    #                    'cif_diff-str', 'cif_diff-p-format',
    #                    ]
    #     right_annoteheaders = ["HR (95% CI)", "P-value",
    #                            'DIFF/100 (95% CI)', 'P-Value',
    #                            ]
    #     leftannote = []
    #     left_annoteheaders = []
    # elif show == 'full-nopval':
    #     rightannote = ["aHR-str",
    #                    'cif_diff-str',
    #                    ]
    #     right_annoteheaders = ["HR (95% CI)",
    #                            'DIFF/100',
    #                            ]
    #
    #     leftannote = ['CIF1-str', 'CIF0-str']
    #     left_annoteheaders = ['CIF/100 in Paxlovid', 'CIF/100 in Ctrl']
    #
    # # fig, ax = plt.subplots()
    # axs = fp.forestplot(
    #     df_result,  # the dataframe with results data
    #     figsize=(4, 12),
    #     estimate="aHR",  # col containing estimated effect size
    #     ll='aHR-lb',
    #     hl='aHR-ub',  # lower & higher limits of conf. int.
    #     varlabel="name",  # column containing the varlabels to be printed on far left
    #     # capitalize="capitalize",  # Capitalize labels
    #     pval="p-val",  # column containing p-values to be formatted
    #     starpval=True,
    #     annote=leftannote,  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
    #     annoteheaders=left_annoteheaders,  # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
    #     rightannote=rightannote,
    #     # p_format, columns to report on right of plot
    #     right_annoteheaders=right_annoteheaders,  # p_format, ^corresponding headers
    #     groupvar="group",  # column containing group labels
    #     group_order=df_result['group'].unique(),
    #     xlabel="Hazard Ratio",  # x-label title
    #     xticks=[0.4, 1, 1.2],  # x-ticks to be printed
    #     color_alt_rows=True,
    #     # flush=True,
    #     sort=True,  # sort estimates in ascending order
    #     # sortby='-aHR',
    #     # table=True,  # Format as a table
    #     # logscale=True,
    #     # Additional kwargs for customizations
    #     **{
    #         # 'fontfamily': 'sans-serif',  # 'sans-serif'
    #         "marker": "D",  # set maker symbol as diamond
    #         "markersize": 35,  # adjust marker size
    #         "xlinestyle": (0., (10, 5)),  # long dash for x-reference line
    #         "xlinecolor": ".1",  # gray color for x-reference line
    #         "xtick_size": 12,  # adjust x-ticker fontsize
    #         # 'fontfamily': 'sans-serif',  # 'sans-serif'
    #     },
    # )
    #
    # axs.axvline(x=1, ymin=0, ymax=0.95, color='grey', linestyle='dashed')
    # check_and_mkdir(output_dir)
    # plt.savefig(output_dir + 'hr_moretabs-{}.png'.format(show), bbox_inches='tight', dpi=600)
    # plt.savefig(output_dir + 'hr_moretabs-{}.pdf'.format(show), bbox_inches='tight', transparent=True)

    print('Done')
    return df_result


def plot_forest_for_ssri_subgroup_lib2_cifdiff(show='full', outcome='any_pasc'):
    subgroup_list = [
        'all',
        # 'white', 'black',
        # 'less35', 'above35',
        '18to25', '25to35', '35to50', '50to65', 'less65', 'above65',
        # 'trimester1', 'trimester2', 'trimester3', 'delivery1week',
        # '1stwave', 'alpha', 'delta', 'omicron', 'omicronafter',
        # 'bminormal', 'bmioverweight', 'bmiobese',
        # 'pregwithcond', 'pregwithoutcond',
        # 'fullyvac', 'partialvac', 'anyvac', 'novacdata',
    ]

    outcome_map = {'any_pasc': 'Long COVID',
                   'PASC-General': 'U099/B948',
                   'any_CFR': 'Any CFR',
                   'death': 'death_postacute',
                   'hospitalization':'hospitalization_postacute'}
    subgroup_info_map = {
        # 'all': ['All', 'Overall'],
        'all': ['All', outcome_map[outcome] + ' in:'],
        'white': ['White', 'Race'],
        'black': ['Black', 'Race'],
        'less35': ['<35', 'Age'],
        'above35': ['≥35', 'Age'],
        '18to25':['18-24', 'Age'],
        '25to35':['25-34', 'Age'],
        '35to50':['35-49', 'Age'],
        '50to65':['50-64', 'Age'],
        'less65': ['<65', 'Age'],
        'above65': ['≥65', 'Age'],
        'trimester1': ['I Trimester', 'Trimesters'],
        'trimester2': ['II Trimester', 'Trimesters'],
        'trimester3': ['III Trimester', 'Trimesters'],
        'delivery1week': ['1 week', 'Trimesters'],
        '1stwave': ['1st wave', 'Infection Time'],
        'alpha': ['Alpha', 'Infection Time'],
        'delta': ['Delta', 'Infection Time'],
        'omicron': ['Omicron', 'Infection Time'],
        'omicronafter': ['Omicron after', 'Infection Time'],
        'bminormal': ['Normal', 'BMI'],
        'bmioverweight': ['Overweight', 'BMI'],
        'bmiobese': ['Obese', 'BMI'],
        'pregwithcond': ['with co-existing conditions', 'With Risk Factor'],
        'pregwithoutcond': ['w/o co-existing conditions', 'With Risk Factor'],
        'fullyvac': ['fullyvac', 'Vaccine'],
        'partialvac': ['partialvac', 'Vaccine'],
        'anyvac': ['anyvac', 'Vaccine'],
        'novacdata': ['novacdata', 'Vaccine'],
    }

    output_dir = r'../data/recover/output/results/ssri_figure_subgroup/'

    results_list = []
    for subgroup in subgroup_list:
        indir = r'../data/recover/output/results/SSRI-overall-{}-ssri-base180-acutevsnot-mentalcov/'.format(
            subgroup.replace(':', '_').replace('/', '-').replace(' ', '_')
        )
        info = subgroup_info_map[subgroup]
        subgroupname = info[0]
        grouplabel = info[1]

        # subgroupname = subgroup.split(':')[-1]
        # if subgroup == 'all':
        #     indir = r'../data/recover/output/results/POSpreg_vs_posnon-usedx1k3useacute1-V3/'

        print('read:', indir)
        # if subgroup == 'PaxRisk:Dementia or other neurological conditions':
        #     df = pd.read_csv(indir + 'causal_effects_specific-snapshot-18.csv')
        #     subgroupname = 'Dementia or other neurological'
        # else:
        #     df = pd.read_csv(indir + 'causal_effects_specific.csv')

        # if subgroup == 'CFR':
        #     df = pd.read_csv(indir + 'causal_effects_specific.csv')
        #     row = df.loc[df['pasc'] == 'any_CFR', :].squeeze()
        #     name = 'Any CFR'
        # else:
        #     df = pd.read_csv(indir + 'causal_effects_specific-snapshot-2.csv')
        #     row = df.loc[df['pasc'] == 'any_pasc', :].squeeze()
        #     name = 'Any PASC'

        # if outcome == 'any_pasc':
        #     df = pd.read_csv(indir + 'causal_effects_specific.csv')
        #     row = df.loc[df['pasc'] == 'any_pasc', :].squeeze()
        #     name = 'Any PASC'

        df = pd.read_csv(indir + 'causal_effects_specific.csv')
        df = df.drop_duplicates(subset=['pasc'], keep='first')
        row = df.loc[df['pasc'] == outcome, :].squeeze()
        name = outcome

        pasc = row['pasc']
        # if row['case+'] < 500:
        #     print(subgroup, 'is very small (<500 in exposed), skip', row['case+'], row['ctrl-'])
        #     continue

        print(name, pasc, outcome)

        if outcome == 'PASC-General' and subgroup == '1stwave':
            print('No u099 in 1st wave')
            continue

        hr = row['hr-w']
        if pd.notna(row['hr-w-CI']):
            ci = stringlist_2_list(row['hr-w-CI'])
        else:
            ci = [np.nan, np.nan]
        p = row['hr-w-p']
        # p_format =
        # if p <= 0.001:
        #     sigsym = '$^{***}$'
        #     p_format = '{:.1e}'.format(p)
        # elif p <= 0.01:
        #     sigsym = '$^{**}$'
        #     p_format = '{:.3f}'.format(p)
        # elif p <= 0.05:
        #     sigsym = '$^{*}$'
        #     p_format = '{:.3f}'.format(p)
        # else:
        #     sigsym = '$^{ns}$'
        #     p_format = '{:.3f}'.format(p)
        # p_format += sigsym

        ahr_pformat, ahr_psym = pformat_symbol(p)

        # domain = row['Organ Domain']
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

        result = [subgroupname, grouplabel, name, pasc,
                  '{:.0f}'.format(row['case+']), '{:.0f}'.format(row['ctrl-']),
                  hr, '{:.2f} ({:.2f}, {:.2f})'.format(hr, ci[0], ci[1]), p,
                  '{:.2f}'.format(ci[0]), '{:.2f}'.format(ci[1]),
                  '({:.2f},{:.2f})'.format(ci[0], ci[1]),
                  cif1, cif1_ci[0], cif1_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif1, cif1_ci[0], cif1_ci[1]),
                  cif0, cif0_ci[0], cif0_ci[1], '{:.2f} ({:.2f}, {:.2f})'.format(cif0, cif0_ci[0], cif0_ci[1]),
                  ahr_pformat + ahr_psym, ahr_psym,
                  cif_diff, '{:.2f} ({:.2f}, {:.2f})'.format(cif_diff, cif_diff_ci[0], cif_diff_ci[1]), cif_diff_p,
                  '{:.2f}'.format(cif_diff_ci[0]), '{:.2f}'.format(cif_diff_ci[1]),
                  '({:.2f},{:.2f})'.format(cif_diff_ci[0], cif_diff_ci[1]),
                  cif_diff_pformat + cif_diff_psym, cif_diff_psym]

        results_list.append(result)

    df_result = pd.DataFrame(results_list,
                             columns=['subgroup', 'grouplabel', 'name', 'pasc', 'No. in 1', 'No. in 0',
                                      'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
                                      'aHR-CI-str',
                                      'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
                                      'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
                                      'p-val-sci', 'sigsym',
                                      'cif_diff', 'cif_diff-str', 'cif_diff-p',
                                      'cif_diff_cilower', 'cif_diff_ciupper', 'cif_diff-CI-str',
                                      'cif_diff-p-format', 'cif_diff-p-symbol'])

    df_result['-aHR'] = -1 * df_result['aHR']
    plt.rc('font', family='serif')
    # fig, ax = plt.subplots()
    if show == 'full':
        rightannote = ["aHR-str", 'p-val-sci',
                       'cif_diff-str', 'cif_diff-p-format',
                       'No. in 1', 'No. in 0',
                       ]

        right_annoteheaders = ["HR (95% CI)", "P-value",
                               'DIFF/100', 'P-Value',
                               'SSRI N=', 'Ctrl N=',
                               ]

        leftannote = ['CIF1-str', 'CIF0-str']
        left_annoteheaders = ['CIF/100 in SSRI', 'CIF/100 in Ctrl']

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

        leftannote = ['No. in 1', 'No. in 0', 'CIF1-str', 'CIF0-str']
        left_annoteheaders = ['SSRI N=', 'Ctrl N=', 'CIF/100 in SSRI', 'CIF/100 in Ctrl']

    axs = fp.forestplot(
        df_result,  # the dataframe with results data
        figsize=(4, 8),  # (7, 12), #(6, 10), # (4.5, 13)
        estimate="aHR",  # col containing estimated effect size
        ll='aHR-lb',
        hl='aHR-ub',  # lower & higher limits of conf. int.
        varlabel="subgroup",  # column containing the varlabels to be printed on far left
        # capitalize="capitalize",  # Capitalize labels
        pval="p-val",  # column containing p-values to be formatted
        starpval=True,
        annote=leftannote,  # ['No. in 1', 'No. in 0', ],  # ["aHR", "aHR-CI-str"],  # columns to report on left of plot
        # annoteheaders=[ "aHR", "Est. (95% Conf. Int.)"],  # ^corresponding headers
        annoteheaders=left_annoteheaders,  # ['No.Pregnant', 'No.Ctrl', ],
        rightannote=rightannote,  # ["aHR-str", "aHR-CI-str", 'p-val-sci', 'CIF1-str', 'CIF0-str'],
        # p_format, columns to report on right of plot
        right_annoteheaders=right_annoteheaders,
        # ["aHR", "95% CI", "P-value", 'CIF1', 'CIF0'],  # p_format, ^corresponding headers
        groupvar="grouplabel",  # column containing group labels
        # group_order=df_result['group'].unique(),
        xlabel="Hazard Ratio",  # x-label title
        xticks=[0.1, 1, 3], #[0.01, 1, 1.5] if outcome == 'PASC-General' else [0.35, 1, 1.5],  # x-ticks to be printed
        color_alt_rows=True,
        # flush=True,
        # sort=True,  # sort estimates in ascending order
        # sortby='-aHR',
        table=True,  # Format as a table
        logscale=True,
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
    plt.savefig(output_dir + 'hr_subgroup-{}-{}.png'.format(show, outcome), bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_subgroup-{}-{}.pdf'.format(show, outcome), bbox_inches='tight', transparent=True)

    print('Done')
    return df_result



if __name__ == '__main__':


    ## 2025-02-21 Primary, with any brain fog
    # indir = r'../data/recover/output/results/CNS-baseADHD-all-adhdCNS-inci-0-30/'
    # indir = r'../data/recover/output/results/CNS-overall-all-adhdCNS-inci-0-30/'

    indir = r'../data/recover/output/results/CNS-baseADHD-all-adhdCNS-inci-0-30s5/'
    indir = r'../data/recover/output/results/CNS-baseADHD-all-adhdCNS-inci-0-30s10/'
    indir = r'../data/recover/output/results/CNS-overall-all-adhdCNS-inci-0-30s5/'

    df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_primary_v3(indir, show='full-nopval')

    zz
    #
    # 2025-2-21 sensitivity, but the same figure as primary
    # indir = r'../data/recover/output/results/SSRI-overall-all-ssri-base180-acutevsnot-nosnriother-mentalcovV3/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_primary_v3(indir, show='full-nopval')
    # indir = r'../data/recover/output/results/SSRI-overall-omicronbroad-ssri-base180-acutevsnot-mentalcovV3/'
    indir = r'../data/recover/output/results/SSRI-overall-all-ssri-base180-fluvoxamineacutevsnot-mentalcovV3/'
    indir = r'../data/recover/output/results/SSRI-overall-all-ssri-base180-sertralineacutevsnot-mentalcovV3/'

    # indir = r'../data/recover/output/results/SSRI-overall-all-ssri-base180-S1Racutevsnot-mentalcovV3/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_primary_v3(indir, show='full-nopval')
    #
    # indir = r'../data/recover/output/results/SSRI-overall-all-ssri-base180-S1RacutevsNonS1R-mentalcovV3/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_primary_v3(indir, drugname='S1R agonists', ctrlname='Non-S1R agonists', show='full-nopval')

    indir = r'../data/recover/output/results/SSRI-overall-all-ssri-base180-S1RacutevsNonS1RNoCita-mentalcovV3/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_primary_v3(indir, drugname='S1R agonists',
    #                                                                   ctrlname='Non-S1R agonists', show='full-nopval')
    indir = r'../data/recover/output/results/SSRI-overall-all-ssri-base180-acuteS1R2vsnoSSRI-mentalcovV3/'
    df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_primary_v3(indir, drugname='fluvoxamine/fluoxetine',
                                                                      ctrlname='Non-SSRI', show='full-nopval')

    # indir = r'../data/recover/output/results/SSRI-overall-all-ssri-base180-S1RNoEscacutevsNonS1R-mentalcovV3/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_primary_v3(indir, drugname='S1R agonists',
    #                                                                   ctrlname='Non-S1R agonists', show='full-nopval')

    zz
    for drug in ['fluvoxamine', 'fluoxetine', 'escitalopram',
                 'citalopram', 'sertraline', 'paroxetine', 'vilazodone']:
        indir = r'../data/recover/output/results/SSRI-overall-all-ssri-base180-{}acutevsnot-mentalcovV3/'.format(drug)
        print(indir)
        df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_primary_v3_individualSSRI(indir, drugname=drug, show='full-nopval')

    zz

    # # 2024-12-15 plot subgroup, then try to summarize them
    # 2025-1-10
    # <50, xticks=[0.1, 1, 10], log = True
    # indir = r'../data/recover/output/results/SSRI-overall-less50-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    # indir = r'../data/recover/output/results/SSRI-overall-above50-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    # indir = r'../data/recover/output/results/SSRI-overall-SMI-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    # indir = r'../data/recover/output/results/SSRI-overall-nonSMI-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    
    # the following errors in some brain fog conditions. debug later
    # indir = r'../data/recover/output/results/SSRI-overall-depression-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    # indir = r'../data/recover/output/results/SSRI-overall-anxiety-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    # indir = r'../data/recover/output/results/SSRI-overall-omicronbroad-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    # indir = r'../data/recover/output/results/SSRI-overall-beforeomicron-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    # zz

    # 2025-1-11
    # indir = r'../data/recover/output/results/SSRI-overall-male-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    # indir = r'../data/recover/output/results/SSRI-overall-female-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    # indir = r'../data/recover/output/results/SSRI-overall-white-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')
    # indir = r'../data/recover/output/results/SSRI-overall-black-ssri-base180-acutevsnot-mentalcovV2/'
    # df_result = plot_forest_for_dx_organ_ssri_lib2_cifdiff_parimary_v2(indir, show='full-nopval')

    # 2025-1-11
    # # #plot subgroup analysis for different pasc outcomes (secondary, death, etc)
    # pasc_outcome = 'death_postacute'
    # pasc_outcome = 'hospitalization_postacute'
    # pasc_outcome = 'PASC-General'
    # pasc_outcome = 'any_CFR'
    # pasc_outcome = 'any_pasc'
    # df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='full-nopval')
    # pasc_outcome = 'death_postacute'
    # df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='full-nopval')
    # pasc_outcome = 'cvddeath_postacute'
    # df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='full-nopval')
    # pasc_outcome = 'any_CFR'
    # df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='full-nopval')
    # pasc_outcome = 'hospitalization_postacute'
    # df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='full-nopval')
    # pasc_outcome = 'PASC-General'
    # df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='full-nopval')
    # pasc_outcome = 'ME/CFS'
    # df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='full-nopval')

    # 2025-2-21
    pasc_outcome = 'any_pasc'
    df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcomeV2(pasc_outcome, show='full-nopval')
    # zz
    pasc_outcome = 'death_postacute'
    df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcomeV2(pasc_outcome, show='full-nopval')
    pasc_outcome = 'cvddeath_postacute'
    df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcomeV2(pasc_outcome, show='full-nopval')
    pasc_outcome = 'any_CFR'
    df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcomeV2(pasc_outcome, show='full-nopval')
    pasc_outcome = 'any_brainfog'
    df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcomeV2(pasc_outcome, show='full-nopval')
    pasc_outcome = 'hospitalization_postacute'
    df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcomeV2(pasc_outcome, show='full-nopval')
    pasc_outcome = 'PASC-General'
    df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcomeV2(pasc_outcome, show='full-nopval')
    pasc_outcome = 'ME/CFS'
    df_result = plot_forest_for_SSRI_subgroup_lib2_cifdiff_pascoutcomeV2(pasc_outcome, show='full-nopval')

    # zz
    # # 2024-11-17
    # plot_forest_for_ssri_subgroup_lib2_cifdiff(show='full-nopval', outcome='death')
    # # plot_forest_for_ssri_subgroup_lib2_cifdiff(show='full-nopval', outcome='hospitalization')
    #
    # plot_forest_for_ssri_subgroup_lib2_cifdiff(show='full-nopval', outcome='any_pasc')
    # plot_forest_for_ssri_subgroup_lib2_cifdiff(show='full-nopval', outcome='PASC-General')
    # plot_forest_for_ssri_subgroup_lib2_cifdiff(show='full-nopval', outcome='any_CFR')
    #
    # # #plot subgroup analysis for primary pasc outcomes
    # # df_result = plot_forest_for_pax_subgroup_lib2_cifdiff(show='full')
    # # df_result = plot_forest_for_pax_subgroup_lib2_cifdiff(show='full-nopval')
    # # df_result = plot_forest_for_pax_subgroup_lib2_cifdiff(show='short')
    #
    # # #plot subgroup analysis for different pasc outcomes (secondary, death, etc)
    # # pasc_outcome = 'death_postacute'
    # # pasc_outcome = 'hospitalization_postacute'
    # # pasc_outcome = 'PASC-General'
    # # pasc_outcome = 'any_CFR'
    # #
    # # df_result = plot_forest_for_pax_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='full')
    # # df_result = plot_forest_for_pax_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='full-nopval')
    # # df_result = plot_forest_for_pax_subgroup_lib2_cifdiff_pascoutcome(pasc_outcome, show='short')
    #
    # # comparing pasc individual conditions CI with boostrap results
    # # df_result = summarize_CI_from_primary_and_boostrap_cifdiff()

    print('Done!')
