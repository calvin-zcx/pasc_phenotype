import os
import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import argparse
import time
import random
import pickle
import ast
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from datetime import datetime
import functools
import seaborn as sns

print = functools.partial(print, flush=True)
from misc import utils
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from misc.utils import check_and_mkdir, stringlist_2_str, stringlist_2_list, pformat_symbol

def plot_cif_bar(outcome='any_pasc', target_cohort='pcornet', exposure=1):
    subgroup_list = [
        'all',
        'white', 'black',
        'less35', 'above35',
        'trimester1', 'trimester2', 'trimester3',
        # 'delivery1week',
        '1stwave', 'alpha', 'delta', 'omicron', 'omicronafter',  # 'omicronbroad',
        'bminormal', 'bmioverweight', 'bmiobese',
        'pregwithcond', 'pregwithoutcond',
        'fullyvac',  # 'partialvac',
        'anyvac', 'novacdata',
    ]

    subgroup_mapn3c = {
        'all': 'full_data',
        'white': 'raceWhite',
        'black': 'raceBlack',
        'less35': 'under35',
        'above35': 'over35',
        'trimester1': 'trimester_1',
        'trimester2': 'trimester_2',
        'trimester3': 'trimester_3',
        'delivery1week': '',
        '1stwave': 'firstwave',
        'alpha': 'alpha',
        'delta': 'delta',
        'omicron': 'omicron',
        'omicronafter': 'omicron-post',
        'omicronbroad': 'omicron',
        'bminormal': 'normalBMI',
        'bmioverweight': 'overweightBMI',
        'bmiobese': 'obeseBMI',
        'pregwithcond': 'anyRisk',
        'pregwithoutcond': 'noRisk',
        'fullyvac': 'fullvacc',
        'partialvac': 'partialvacc',
        'anyvac': 'anyvacc',
        'novacdata': 'novacc',
    }

    outcome_map = {'any_pasc': 'PASC',  # 'Long COVID',
                   'PASC-General': 'U099/B948',
                   'any_CFR': 'Any CFR'}

    outcome_mapn3c = {
        'any_pasc': 'CP_PASC',
        'PASC-General': 'U09_B94',
        'any_CFR': 'Any_CFR'}

    subgroup_info_map = {
        # 'all': ['All', 'Overall'],
        'all': ['Overall', outcome_map[outcome] + ' in:'],
        'white': ['Race-White', 'Race'],
        'black': ['Race-Black', 'Race'],
        'less35': ['Age <35', 'Age'],
        'above35': ['Age ≥35', 'Age'],
        'trimester1': ['1st Trimester', 'Trimesters'],
        'trimester2': ['2nd Trimester', 'Trimesters'],
        'trimester3': ['3rd Trimester', 'Trimesters'],
        'delivery1week': ['1 week', 'Trimesters'],
        '1stwave': ['1st wave', 'Infection Time'],
        'alpha': ['Alpha', 'Infection Time'],
        'delta': ['Delta', 'Infection Time'],
        'omicron': ['Omicron-BA.1&BA.2', 'Infection Time'],
        'omicronafter': ['Omicron-other subs', 'Infection Time'],
        'omicronbroad': ['Omicron', 'Infection Time'],
        'bminormal': ['BMI-Normal', 'BMI'],
        'bmioverweight': ['BMI-Overweight', 'BMI'],
        'bmiobese': ['BMI-Obese', 'BMI'],
        'pregwithcond': ['with Risk Factor', 'Risk Factor'],
        'pregwithoutcond': ['w/o Risk Factor', 'Risk Factor'],
        'fullyvac': ['Fully Vaccinated', 'Vaccinated'],
        'partialvac': ['Partially Vaccinated', 'Vaccinated'],
        'anyvac': ['Any Vaccinated', 'Vaccinated'],
        'novacdata': ['No Vaccine Records', 'Vaccinated'],
    }

    output_dir = r'../data/recover/output/pregnancy_output/figure_subgroup/'

    df2 = pd.read_excel(r'../data/recover/output/pregnancy_output/N3C/24_06_19 Cleaned N3C Results.xlsx')
    df2 = df2.set_index('Unnamed: 0')
    df2 = df2.transpose()

    results_list = []
    results_list1 = []
    results_list2 = []
    for subgroup in subgroup_list:
        indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1-V3-{}/'.format(
            subgroup.replace(':', '_').replace('/', '-').replace(' ', '_')
        )
        info = subgroup_info_map[subgroup]
        subgroupname = info[0]
        grouplabel = info[1]

        # subgroupname = subgroup.split(':')[-1]
        if subgroup == 'all':
            indir = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx1k3useacute1-V3/'

        print('read:', indir)

        df = pd.read_csv(indir + 'causal_effects_specific.csv')
        df = df.drop_duplicates(subset=['pasc'], keep='first')

        for cohort in ['pcornet', 'N3C']:
            if cohort == 'pcornet':
                row = df.loc[df['pasc'] == outcome, :].squeeze()
                name = outcome
                pasc = row['pasc']
            elif cohort == 'N3C':
                row_index = subgroup_mapn3c[subgroup] + '_' + outcome_mapn3c[outcome]
                row = df2.loc[row_index, :].squeeze()
                name = outcome
                pasc = row['pasc']
                subgroupname = subgroupname + ' (N3C)'

            # if row['case+'] < 500:
            #     print(subgroup, 'is very small (<500 in exposed), skip', row['case+'], row['ctrl-'])
            #     continue

            print(name, pasc, outcome)

            if outcome == 'PASC-General' and ((subgroup == '1stwave') or (subgroup == subgroup_mapn3c['1stwave'])):
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
            if cohort == 'pcornet':
                results_list1.append(result)
            elif cohort == 'N3C':
                results_list2.append(result)

    df_result1 = pd.DataFrame(results_list1,
                             columns=['subgroup', 'grouplabel', 'name', 'pasc', 'No. in 1', 'No. in 0',
                                      'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
                                      'aHR-CI-str',
                                      'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
                                      'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
                                      'p-val-sci', 'sigsym',
                                      'cif_diff', 'cif_diff-str', 'cif_diff-p',
                                      'cif_diff_cilower', 'cif_diff_ciupper', 'cif_diff-CI-str',
                                      'cif_diff-p-format', 'cif_diff-p-symbol'])

    df_result2 = pd.DataFrame(results_list2,
                              columns=['subgroup', 'grouplabel', 'name', 'pasc', 'No. in 1', 'No. in 0',
                                       'aHR', 'aHR-str', 'p-val', 'aHR-lb', 'aHR-ub',
                                       'aHR-CI-str',
                                       'CIF1', 'CIF1-lb', 'CIF1-ub', 'CIF1-str',
                                       'CIF0', 'CIF0-lb', 'CIF0-ub', 'CIF0-str',
                                       'p-val-sci', 'sigsym',
                                       'cif_diff', 'cif_diff-str', 'cif_diff-p',
                                       'cif_diff_cilower', 'cif_diff_ciupper', 'cif_diff-CI-str',
                                       'cif_diff-p-format', 'cif_diff-p-symbol'])




    if target_cohort == 'pcornet':
        df = df_result1
    elif target_cohort == 'N3C':
        df = df_result2
    else:
        raise ValueError

    plt.rc('font', family='serif')

    label_list = df['subgroup'].tolist()
    if exposure == 1:
        result_list = df['CIF1'].tolist()
        # error_list_1 = df['Std[fit]'].tolist()
        ci0 = (df['CIF1'] - df['CIF1-lb']).tolist()
        ci1 = (df['CIF1-ub'] - df['CIF1']).tolist()
    elif exposure == 0:
        result_list = df['CIF0'].tolist()
        # error_list_1 = df['Std[fit]'].tolist()
        ci0 = (df['CIF0'] - df['CIF0-lb']).tolist()
        ci1 = (df['CIF0-ub'] - df['CIF0']).tolist()
    else:
        raise ValueError

    yerr = np.vstack((ci0, ci1))
    # colors = [organ_color[x] for x in df['Organ Domain']]
    # ranks =  df['rank'].tolist()
    # patterns = []
    # for x in df['E[fit]']:
    #     if 0.7 <= x < 0.8:
    #         patterns.append('\\')
    #     elif x >= 0.8:
    #         patterns.append("o")
    #     else:
    #         patterns.append('')

    # patterns[0] = '|'
    # auc_list_2 = data[metrics + '_mean_of'].tolist()
    # error_list_2 = data[metrics + '_std_of'].tolist()

    fig, ax = plt.subplots(figsize=(11, 7))
    idx = np.arange(len(label_list))
    new_idx = np.asarray([1.5 * i for i in idx])
    # new_idx = np.asarray([i for i in idx])
    #
    bar = ax.bar(new_idx + 0.5, result_list, 1, yerr=yerr,  edgecolor='black', alpha=.8, ) # color=colors, hatch=patterns
    # ax.bar(new_idx + .6, auc_list_2, .6, yerr=error_list_2, color='#98c1d9', edgecolor='black', alpha=.8)

    for i, rect in enumerate(bar):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height+ ci1[i] + 0.01, f'{result_list[i]:.2f}', ha='center', va='bottom')


    ax.set_xticks(new_idx + .5)
    # ax.set_xlim([-1, len(new_idx) * 2])
    # ax.set_ylim([.5, 1])
    # x_show = []
    # for
    ax.set_xticklabels(label_list, rotation=45, fontsize=15, ha='right', rotation_mode="anchor")
    ax.yaxis.grid()  # color='#D3D3D3', linestyle='--', linewidth=0.7)
    plt.ylabel('CIF/100 in {}'.format('Pregnant' if exposure else 'Non-pregnant'), fontsize=16) #, weight='bold')
    plt.yticks(fontsize=15)
    plt.xlabel(outcome_map[outcome], fontsize=16)
    # ax.set(title=learner + ' ' + metrics)
    # plt.subplots_adjust(bottom=.3)
    # handle_list = [mpatches.Patch(color='#98c1d9', label='INSIGHT'),  # '#e26d5c'
    #                mpatches.Patch(color='#98c1d9', label='OneFlorida+')
                   # ]
    # plt.legend(handles=handle_list, prop={'size': 15})
    plt.tight_layout()
    plt.savefig(output_dir + 'bar-plot-{}-{}-Exposure{}.pdf'.format(outcome, target_cohort, exposure), dpi=600)
    plt.savefig(output_dir + 'bar-plot-{}-{}-Exposure{}.png'.format(outcome, target_cohort, exposure), dpi=600)
    plt.show()
    plt.close()

    return df_result1, df_result2


if __name__ == '__main__':
    start_time = time.time()
    exposure = 1
    plot_cif_bar(outcome='any_pasc', target_cohort='pcornet', exposure=exposure)
    plot_cif_bar(outcome='any_pasc', target_cohort='N3C', exposure=exposure)
    plot_cif_bar(outcome='PASC-General', target_cohort='pcornet', exposure=exposure)
    plot_cif_bar(outcome='PASC-General', target_cohort='N3C', exposure=exposure)
    plot_cif_bar(outcome='any_CFR', target_cohort='pcornet', exposure=exposure)
    plot_cif_bar(outcome='any_CFR', target_cohort='N3C', exposure=exposure)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
