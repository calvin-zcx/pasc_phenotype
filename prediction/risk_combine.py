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
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test, proportional_hazard_test, logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import k_fold_cross_validation
from PRModels import ml
import matplotlib.pyplot as plt
# from brokenaxes import brokenaxes

# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori
import zepid
from zepid.graphics import EffectMeasurePlot
from misc import utils


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['OneFlorida', 'INSIGHT', 'combined'], default='OneFlorida',
                        help='data bases')
    parser.add_argument('--encode', choices=['elix', 'icd_med'], default='elix',
                        help='data encoding')
    parser.add_argument('--population', choices=['positive', 'negative', 'all'], default='positive')
    parser.add_argument('--severity', choices=['all', 'outpatient', "inpatienticu",
                                               'inpatient', 'icu', 'ventilation', ],
                        default='all')
    parser.add_argument('--goal', choices=['anypasc', 'allpasc', 'anyorgan', 'allorgan',
                                           'anypascsevere', 'anypascmoderate'],
                        default='anypascsevere')
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()

    args.data_dir = r'output/dataset/{}/{}/'.format(args.dataset, args.encode)
    args.out_dir = r'output/factors/{}/{}/'.format(args.dataset, args.encode)
    args.fig_out_dir = r'output/figures/{}/{}/'.format(args.dataset, args.encode)

    # args.processed_data_file = r'output/dataset/{}/df_cohorts_covid_4manuNegNoCovidV2_bool_all-PosOnly-{}.csv'.format(
    #     args.dataset, args.encode)

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.out_dir)
    return args


def risk_table_2_datasets():
    args.out_dir = r''.format(args.dataset, args.encode)
    f1 = 'output/factors/INSIGHT/elix/any_pasc/any-at-least-1-pasc-riskFactor-INSIGHT-positive-all.csv'
    f2 = 'output/factors/INSIGHT/elix/any_pasc_severe/any-at-least-1-severe-pasc-riskFactor-INSIGHT-positive-all.csv'
    f3 = 'output/factors/INSIGHT/elix/any_pasc_moderate/any-at-least-1-moderate-pasc-riskFactor-INSIGHT-positive-all.csv'

    f1 = 'output/factors/INSIGHT/elix/any_pasc_severe/any-at-least-1-severe-pasc-riskFactor-INSIGHT-positive-all.csv'
    f2 = 'output/factors/INSIGHT/elix/any_pasc_severe/any-at-least-1-severe-pasc-riskFactor-INSIGHT-positive-outpatient.csv'
    f3 = 'output/factors/INSIGHT/elix/any_pasc_severe/any-at-least-1-severe-pasc-riskFactor-INSIGHT-positive-inpatienticu.csv'

    # f2 = 'output/factors/OneFlorida/elix/any_pasc_severe/any-at-least-2-severe-pasc-riskFactor-OneFlorida-positive-all.csv'
    # f3 = 'output/factors/INSIGHT/elix/any_pasc_moderate/any-at-least-2-moderate-pasc-riskFactor-INSIGHT-positive-all.csv'
    # f4 = 'output/factors/OneFlorida/elix/any_pasc_moderate/any-at-least-2-moderate-pasc-riskFactor-OneFlorida-positive-all.csv'

    df_vec = []

    def _hr_str(hr, hr_lower, hr_upper):
        return '{:.2f} ({:.2f}-{:.2f})'.format(hr, hr_lower, hr_upper)

    def _p_str(p):
        if p >= 0.001:
            return '{:.3f}'.format(p)
        else:
            return '{:.1e}'.format(p)

    for f in [f1, f2, f3]:
        df = pd.read_csv(f)
        print(f)
        df_format = df[['Unnamed: 0', 'covariate', ]].copy()
        df_format['Age & severity adjusted HR'] = np.nan
        df_format['Age & severity adjusted HR p-Value'] = np.nan

        df_format['Fully adjusted HR'] = np.nan
        df_format['Fully adjusted HR p-Value'] = np.nan
        df_format['selected'] = 0

        for key, rows in df.iterrows():
            hr = rows['HR']
            hr_lower = rows['CI-95% lower-bound']
            hr_upper = rows['CI-95% upper-bound']
            p = rows["p-Value"]
            if p < 0.01:
                df_format.loc[key, 'selected'] = 2
            elif p < 0.05:
                df_format.loc[key, 'selected'] = 1

            df_format.loc[key, 'Fully adjusted HR'] = _hr_str(hr, hr_lower, hr_upper)
            df_format.loc[key, 'Fully adjusted HR p-Value'] = _p_str(p)

            if 'ageAcute-HR' in rows.index:
                ageAcute_hr = rows['ageAcute-HR']
                ageAcute_hr_lower = rows['ageAcute-CI-95% lower-bound']
                ageAcute_hr_upper = rows['ageAcute-CI-95% upper-bound']
                ageAcute_p = rows["ageAcute-p-Value"]
                df_format.loc[key, 'Age & severity adjusted HR'] = _hr_str(ageAcute_hr, ageAcute_hr_lower,
                                                                           ageAcute_hr_upper)
                df_format.loc[key, 'Age & severity adjusted HR p-Value'] = _p_str(ageAcute_p)
            # else:
                # print('Not found ageAcute-HR info in', f)

        df_vec.append(df_format)

    df_out = df_vec[0]
    for i in range(1, len(df_vec)):
        df_out = pd.merge(df_out, df_vec[i], left_on='covariate', right_on='covariate', how='outer')

    # df_out.to_excel('output/factors/INSIGHT/elix/risk_table/any-at-least-1-pool-severe-moderate-pasc-riskFactor.xlsx',
    #                 index=False)
    df_out.to_excel('output/factors/INSIGHT/elix/risk_table/any-at-least-1-pool-severe-outpatient-inpatient-pasc-riskFactor.xlsx',
                    index=False)
    return df_out


def risk_stratified_by_severity(args):
    f1 = args.out_dir + 'any_pasc/' + 'any-at-least-1-pasc-riskFactor-{}-positive-all.csv'.format(args.dataset)
    f2 = args.out_dir + 'any_pasc_severe/' + 'any-at-least-1-severe-pasc-riskFactor-{}-positive-all.csv'.format(
        args.dataset)
    f3 = args.out_dir + 'any_pasc_moderate/' + 'any-at-least-1-moderate-pasc-riskFactor-{}-positive-all.csv'.format(
        args.dataset)
    # f4 = args.out_dir + 'any_pasc/' + 'any-at-least-1-pasc-riskFactor-{}-positive-icu.csv'.format(args.dataset)

    df = pd.read_csv(f1)
    for filename, label in zip([f2, f3], ['severe', 'moderate']):
        df2 = pd.read_csv(filename)
        df = pd.merge(df, df2, left_on='covariate', right_on='covariate', how='outer', suffixes=('', '_' + label))
        print(filename, label)

    df.to_csv(
        args.out_dir + 'risk_table/' + 'any-at-least-1-pasc[pooled-severe-moderate]-riskFactor-{}-positive_combined.csv'.format(
            args.dataset),
        index=False)
    return df


def plot_forest_for_risk_stratified_by_severity(args, star=True):
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

    f1 = args.out_dir + 'any_pasc_severe/' + 'any-at-least-1-severe-pasc-riskFactor-{}-positive-all.csv'.format(
        args.dataset)
    f2 = args.out_dir + 'any_pasc_severe/' + 'any-at-least-1-severe-pasc-riskFactor-{}-positive-outpatient.csv'.format(
        args.dataset)
    f3 = args.out_dir + 'any_pasc_severe/' + 'any-at-least-1-severe-pasc-riskFactor-{}-positive-inpatient.csv'.format(
        args.dataset)
    f4 = args.out_dir + 'any_pasc_severe/' + 'any-at-least-1-severe-pasc-riskFactor-{}-positive-icu.csv'.format(
        args.dataset)

    df = pd.read_csv(f1)
    df_select = df.loc[df['p-Value'] < 0.05, :]

    df_vec = []
    for filename, label in zip([f2, f3, f4], ['outpatient', 'inpatient', 'icu']):
        print(filename, label)
        df2 = pd.read_csv(filename)
        df_vec.append(df2)

    df_vec = df_vec[:0]

    organ_n = np.zeros(len(organ_list))
    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []
    pasc_row2 = []
    color_list = []

    for key, row in df_select.iterrows():
        cov_name = row['covariate']
        name = row['covariate'].replace('DX: ', '')
        hr = row['HR']
        ci = row['CI-95% lower-bound'], row['CI-95% upper-bound']
        p = row['p-Value']

        labs.append(name)
        measure.append(hr)
        lower.append(ci[0])
        upper.append(ci[1])
        color_list.append('#ed6766')

        for _df in df_vec:
            if (_df['covariate'] == cov_name).sum() > 0:
                _row = _df.loc[_df['covariate'] == cov_name, :].squeeze()
                _name = _row['covariate'].replace('DX: ', '')
                _hr = _row['HR']
                _ci = _row['CI-95% lower-bound'], _row['CI-95% upper-bound']
                _p = _row['p-Value']
            else:
                _hr = np.nan
                _ci = np.nan, np.nan
                _p = np.nan

            labs.append('')
            measure.append(_hr)
            lower.append(_ci[0])
            upper.append(_ci[1])
            color_list.append('#A986B5')

        if star:
            if p <= 0.01:
                name += '**'
            elif (p > 0.01) and (p <= 0.05):
                name += '*'

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    # p.labels(scale='log')

    # organ = 'ALL'
    p.labels(effectmeasure='aHR')  # aHR
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    c = '#F65453'
    p.colors(pointshape="o", errorbarcolor=color_list, pointcolor=color_list)  # , linecolor='#fcb2ab')
    width = 9.
    height = .28 * len(labs)
    if len(labs) == 2:
        height = .3 * (len(labs) + 1)
    ax = p.plot(figsize=(width, height), t_adjuster=0.010, max_value=3, min_value=0.5, size=5, decimal=2)  # 0.02
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)

    organ_n_cumsum = np.cumsum(organ_n)
    # for i in range(len(organ_n) - 1):
    #     ax.axhline(y=organ_n_cumsum[i] - .5, xmin=0.0, color=p.linec, zorder=1, linestyle='--')

    ax.set_yticklabels(labs, fontsize=11.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()

    output_dir = args.out_dir + 'figs/'
    utils.check_and_mkdir(output_dir)

    plt.savefig(output_dir + 'hr_severity_stratified_{}.png'.format(args.dataset),
                bbox_inches='tight',
                dpi=600)
    plt.savefig(output_dir + 'hr_severity_stratified_{}.pdf'.format(args.dataset),
                bbox_inches='tight',
                transparent=True)
    plt.show()
    print()
    # plt.clf()
    plt.close()


def get_model_c_index(args, severe=True, total=9):
    vcindex = []
    vstd = []
    for pasc_threshold in range(1, total):
        fname = args.out_dir + 'any_pasc_{}/any-at-least-{}-{}-pasc-modeSelection-{}-{}-{}.csv'.format(
            'severe' if severe else 'moderate',
            pasc_threshold, 'severe' if severe else 'moderate',
            args.dataset, args.population, args.severity)
        df = pd.read_csv(fname)
        for key, row in df.iterrows():
            cindex = row['E[fit]']
            st = row['Std[fit]']
            vcindex.append(cindex)
            vstd.append(st)
            break

    result = pd.DataFrame({'cindex': vcindex, 'std': vstd})
    return result


def get_c_index_of_all_pasc(args):
    df_pasc_info = pd.read_excel('output/causal_effects_specific_withMedication_v3.xlsx', sheet_name='diagnosis')
    selected_pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    print('len(selected_pasc_list)', len(selected_pasc_list))  # 44 pasc
    print(selected_pasc_list)
    # selected_organ_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'Organ Domain'].unique()
    # print('len(selected_organ_list)', len(selected_organ_list))

    pasc_name_map = {}
    for index, row in df_pasc_info.iterrows():
        pasc_name_map[row['pasc']] = row['PASC Name Simple']

    results = []
    for pasc in selected_pasc_list:
        df = pd.read_csv(args.out_dir + 'every_pasc/PASC-{}-modeSelection-{}-{}-{}.csv'.format(
            pasc.replace('/', '_'), args.dataset, args.population, args.severity))
        row1 =df.iloc[0]
        cindex = row1['E[fit]']
        cstd = row1['Std[fit]']
        results.append((pasc, pasc_name_map[pasc], cindex, cstd))

    df_result = pd.DataFrame(results, columns=['pasc', 'pasc name simple', 'cindex', 'cstd'])
    print('Dump done', df_result.shape)
    df_result.to_csv(args.out_dir + 'every_pasc/selected_pasc_cindex.csv')

    return df_result


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    df_result = get_c_index_of_all_pasc(args)

    # risk_stratified_by_severity(args)

    # 1.
    # plot_forest_for_risk_stratified_by_severity(args, star=True)

    # 2.
    # df = risk_table_2_datasets()

    # result1 = get_model_c_index(args, severe=True, total=9)
    # result2 = get_model_c_index(args, severe=False, total=9)
    # result = pd.concat([result1, result2], axis=1)
    # result.index = result.index+1
    # result.to_csv('output/factors/{}/elix/c_index_of_pasc_severity_over_threshold-v2.csv'.format(args.dataset))

    print('Done')
