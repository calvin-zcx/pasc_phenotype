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
# from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
# from lifelines.statistics import survival_difference_at_fixed_point_in_time_test, proportional_hazard_test, logrank_test
# from lifelines.plotting import add_at_risk_counts
# from lifelines.utils import k_fold_cross_validation
# # from PRModels import ml
# import matplotlib.pyplot as plt
# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori
#

def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['OneFlorida', 'INSIGHT'], default='INSIGHT',
                        help='data bases')
    parser.add_argument('--encode', choices=['elix', 'icd_med'], default='elix',
                        help='data encoding')
    parser.add_argument('--severity', choices=['all',
                                               'outpatient', 'inpatient', 'icu', 'ventilation', "inpatienticu"
                                               ],
                        default='all')
    parser.add_argument('--goal', choices=['anypasc', 'allpasc', 'anyorgan', 'allorgan'], default='allpasc')
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()

    # More args
    if args.dataset == 'INSIGHT':
        args.data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL.csv'
    elif args.dataset == 'OneFlorida':
        args.data_file = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all.csv'
    else:
        raise ValueError

    args.out_dir = r'../data/{}/output/character/query/'.format(args.dataset, args.encode)
    args.processed_data_file = args.out_dir + r'matrix_cohorts_covid_4manuNegNoCovidV2_bool_all-ANYPASC.csv'

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.out_dir)
    return args


def read_all_and_dump_covid_positive(data_file):
    # r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL.csv'
    # r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all.csv'
    print('Load data  file:', data_file)
    # a_debug = pd.DataFrame({'0':df.columns, '1':df.dtypes})
    df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
    df = df.loc[(df['covid'] == 1), :]
    print('df.loc[(df[covid] == 1), :].shape:', df.shape)

    df.to_csv(data_file.replace('.csv', '-PosOnly.csv'), index=False)
    print('Dump posOnly file done!:', data_file.replace('.csv', '-PosOnly.csv'))
    return df


def build_incident_pasc_from_all(args, dump=True):
    start_time = time.time()
    print('In build_incident_pasc_from_all')
    print('Step1: Load Covid positive data  file:', args.data_file)
    df = pd.read_csv(args.data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])

    print('Total df.shape:', df.shape)
    print('All Covid Positives:', (df['covid'] == 1).sum(), (df['covid'] == 1).mean())
    print('All Covid Negatives:', (df['covid'] == 0).sum(), (df['covid'] == 0).mean())

    # add number of comorbidity as features
    n_comor = df[[x for x in df.columns if (x.startswith('DX:') or x.startswith('MEDICATION:'))]].sum(axis=1)
    n_comor_cols = ['num_Comorbidity=0', 'num_Comorbidity=1', 'num_Comorbidity=2',
                    'num_Comorbidity=3', 'num_Comorbidity=4', 'num_Comorbidity>=5']
    print('len(n_comor > 0)', (n_comor > 0).sum())
    for i in [0, 1, 2, 3, 4, 5]:
        col = n_comor_cols[i]
        print(i, col)
        df[col] = 0
        if i < 5:
            df.loc[n_comor == i, col] = 1
        else:
            df.loc[n_comor >= 5, col] = 1
    print('After add number of comorbidities df.shape:', df.shape)

    # add selected incident PASC flag
    print('Step2: add selected incident PASC flag and time 2 event')
    df_pasc_info = pd.read_excel('../prediction/output/causal_effects_specific_withMedication_v3.xlsx',
                                 sheet_name='diagnosis')
    selected_pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    print('len(selected_pasc_list)', len(selected_pasc_list))
    print(selected_pasc_list)

    selected_organ_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'Organ Domain'].unique()
    print('len(selected_organ_list)', len(selected_organ_list))
    print(selected_organ_list)
    organ_pasc = {}
    for i, organ in enumerate(selected_organ_list):
        pascs = df_pasc_info.loc[
            (df_pasc_info['selected'] == 1) & (df_pasc_info['Organ Domain'] == organ), 'pasc'].tolist()
        organ_pasc[organ] = pascs
        print(i, organ, '-->', len(pascs), ':', pascs)

    print('selected PASC and organ domain done!')

    exclude_DX_list = {
        'Neurocognitive disorders': ['DX: Dementia'],
        'Diabetes mellitus with complication': ['DX: Diabetes Type 2'],
        'Chronic obstructive pulmonary disease and bronchiectasis': ['DX: Chronic Pulmonary Disorders', 'DX: COPD'],
        'Circulatory signs and symptoms': ['DX: Arrythmia'],
        'Anemia': ['DX: Anemia'],
        'Heart failure': ["DX: Congestive Heart Failure"]
    }

    print('Labeling INCIDENT pasc in {0,1}')
    # flag@pascname  for incidence label, dx-t2e@pascname for original shared t2e
    for pasc in selected_pasc_list:
        flag = df['dx-out@' + pasc] - df['dx-base@' + pasc]
        if pasc in exclude_DX_list:
            ex_DX_list = exclude_DX_list[pasc]
            print(pasc, 'further exclude', ex_DX_list)
            for ex_DX in ex_DX_list:
                flag -= df[ex_DX]

        df['flag@' + pasc] = (flag > 0).astype('int')

    def _debug_person(pid):
        _person = pd.DataFrame(data={'dx-base': df.loc[pid, ['dx-base@' + x for x in selected_pasc_list]].tolist(),
                                     'dx-out': df.loc[pid, ['dx-out@' + x for x in selected_pasc_list]].tolist(),
                                     'dx-t2e': df.loc[pid, ['dx-t2e@' + x for x in selected_pasc_list]].tolist()},
                               index=selected_pasc_list)
        return _person

    # build flag, t2e for any pasc
    # flag@pascname  for incidence label, dx-t2e@pascname for t2e which is shared from original data
    print('Any PASC: build flag, t2e for any pasc')
    specific_pasc_col = [x for x in df.columns if x.startswith('flag@')]
    n_pasc_series = df[specific_pasc_col].sum(axis=1)
    df['pasc-count'] = n_pasc_series  # number of incident pascs of this person
    df['pasc-flag'] = (n_pasc_series > 0).astype('int')  # indicator of any incident pasc of this person
    df['pasc-min-t2e'] = 180

    for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
        npasc = rows['pasc-count']
        if npasc >= 1:
            # if there are any incident pasc, t2e of any pasc is the earliest time of incident pasc
            pasc_flag_cols = list(rows[specific_pasc_col][rows[specific_pasc_col] > 0].index)
            pasc_t2e_cols = [x.replace('flag@', 'dx-t2e@') for x in pasc_flag_cols]
            t2e = rows.loc[pasc_t2e_cols].min()
        else:
            # if no incident pasc, t2e of any pasc: event, death, censoring, 180 days followup, whichever came first.
            # no event, only consider death, censoring, 180 days,
            # 1. approximated by the maximum-t2e of any selected pasc .
            #   unless all selected pasc happened, but not incident, this not happened in our data.
            # 2. directly follow the definition. Because I also stored max-followup information
            # t2e = rows.loc[['dx-t2e@' + x for x in selected_pasc_list]].max()
            t2e = max(30, np.min([rows['death t2e'], rows['maxfollowup'], 180]))

        df.loc[index, 'pasc-min-t2e'] = t2e

    # build flag, t2e for each organ
    print('Organ category: build flag, t2e for Organ category with a list of pascs')
    for organ in selected_organ_list:
        pascs = organ_pasc[organ]
        pascs_col = ['flag@' + x for x in pascs]
        organ_series = df[pascs_col].sum(axis=1)
        df['organ-count@' + organ] = organ_series
        df['organ-flag@' + organ] = (organ_series > 0).astype('int')
        df['organ-t2e@' + organ] = 180

    for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
        for organ in selected_organ_list:
            npasc = rows['organ-count@' + organ]
            pascs_col = ['flag@' + x for x in organ_pasc[organ]]
            if npasc >= 1:
                pasc_flag_cols = list(rows[pascs_col][rows[pascs_col] > 0].index)
                pasc_t2e_cols = [x.replace('flag@', 'dx-t2e@') for x in pasc_flag_cols]
                t2e = rows.loc[pasc_t2e_cols].min()
            else:
                # t2e = rows.loc[['dx-t2e@' + x for x in organ_pasc[organ]]].max()
                t2e = max(30, np.min([rows['death t2e'], rows['maxfollowup'], 180]))

            df.loc[index, 'organ-t2e@' + organ] = t2e

    print('Any Organ: build flag, t2e for any organ')
    specific_organ_col = [x for x in df.columns if x.startswith('organ-flag@')]
    n_organ_series = df[specific_organ_col].sum(axis=1)
    df['organ-count'] = n_organ_series
    df['organ-flag'] = (n_organ_series > 0).astype('int')
    df['organ-min-t2e'] = 180
    # b_debug = df[['pasc-count', 'pasc-flag', 'pasc-min-t2e', 'organ-count', 'organ-flag', 'organ-min-t2e']]
    for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
        norgan = rows['organ-count']
        if norgan >= 1:
            organ_flag_cols = list(rows[specific_organ_col][rows[specific_organ_col] > 0].index)
            organ_t2e_cols = [x.replace('organ-flag@', 'organ-t2e@') for x in organ_flag_cols]
            t2e = rows.loc[organ_t2e_cols].min()
        else:
            # t2e: event, death, censoring , 180, whichever came first.
            # no t2e, only consider death and censoring, which were considered by the maximum-t2e of any selected pasc
            # t2e = rows.loc[['organ-t2e@' + x for x in selected_organ_list]].max()
            t2e = max(30, np.min([rows['death t2e'], rows['maxfollowup'], 180]))

        df.loc[index, 'organ-min-t2e'] = t2e

    print('Add selected incident PASC, any pasc, organ system, flag done!')

    covs_columns = list(df.columns)[0:df.columns.get_loc('PX: Convalescent Plasma')] + \
                   ['pasc-count', 'pasc-flag', 'pasc-min-t2e'] + \
                   [x for x in df.columns if x.startswith('organ')]
    if dump:
        utils.check_and_mkdir(args.processed_data_file)
        df.loc[:, covs_columns].to_csv(args.processed_data_file, index=False)
        print('Dump to:', args.processed_data_file)

    print('build_incident_pasc_from_all Done! Total Time used:',
          time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print(len(df.loc[(df['covid'] == 0) & (df['pasc-flag'] == 0), :]),
          len(df.loc[(df['covid'] == 0) & (df['pasc-flag'] == 1), :]),
          len(df.loc[(df['covid'] == 0), :]),
          len(df.loc[(df['covid'] == 0) & (df['pasc-flag'] == 1), :]) / len(df.loc[(df['covid'] == 0), :]) * 10000
          )

    print(len(df.loc[(df['covid'] == 1) & (df['pasc-flag'] == 0), :]),
          len(df.loc[(df['covid'] == 1) & (df['pasc-flag'] == 1), :]),
          len(df.loc[(df['covid'] == 1), :]),
          len(df.loc[(df['covid'] == 1) & (df['pasc-flag'] == 1), :]) / len(df.loc[(df['covid'] == 0), :]) * 10000
          )

    print(len(df.loc[(df['pasc-flag'] == 0), :]),
          len(df.loc[(df['pasc-flag'] == 1), :]),
          len(df),
          len(df.loc[(df['pasc-flag'] == 1), :]) / len(df) * 10000
          )

    return df, df_pasc_info


if __name__ == '__main__':
    # python query_any_pasc.py --dataset INSIGHT 2>&1 | tee  log/query_any_pasc-insight.txt
    # python query_any_pasc.py --dataset OneFlorida 2>&1 | tee  log/query_any_pasc-oneflorida.txt

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    # -Pre step1: select Covid Positive data and dump
    # read_all_and_dump_covid_positive(r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL.csv')
    # read_all_and_dump_covid_positive(r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all.csv')

    # -Pre step2: build Covid Positive data and dump for future use
    df, df_pasc_info = build_incident_pasc_from_all(args)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
