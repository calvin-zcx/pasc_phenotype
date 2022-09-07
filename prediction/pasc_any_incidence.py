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
from tqdm import tqdm
from datetime import datetime

import functools
import seaborn as sns

print = functools.partial(print, flush=True)
from misc import utils

# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori
KFOLD = 5
MIN_PERCENTAGE = 0.005
N_SHUFFLE = 5


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['OneFlorida', 'INSIGHT', 'Pooled'], default='Pooled',
                        help='data bases')
    parser.add_argument('--encode', choices=['elix', 'icd_med'], default='elix',
                        help='data encoding')
    parser.add_argument('--population', choices=['positive', 'negative', 'all'], default='positive')
    parser.add_argument('--severity', choices=['all', 'outpatient', "inpatienticu",
                                               'inpatient', 'icu', 'ventilation', '1stwave', 'delta'], default='delta')
    parser.add_argument('--goal', choices=['anypasc', 'allpasc', 'anyorgan', 'allorgan',
                                           'anypascsevere', 'anypascmoderate'],
                        default='anypasc')
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()

    # More args
    if args.dataset == 'INSIGHT':
        # args.data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL-PosOnly.csv'
        # args.processed_data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL-PosOnly-anyPASC.csv'
        args.data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL.csv'
        args.processed_data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL-anyPASC.csv'
    elif args.dataset == 'OneFlorida':
        # args.data_file = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all-PosOnly.csv'
        # args.processed_data_file = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all-PosOnly-anyPASC.csv'
        args.data_file = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all.csv'
        args.processed_data_file = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all-anyPASC.csv'
    elif args.dataset == 'Pooled':
        # args.processed_data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL-PosOnly-anyPASC.csv'
        # args.processed_data_file2 = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all-PosOnly-anyPASC.csv'
        args.data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL.csv'
        args.data_file2 = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all.csv'
        args.processed_data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL-anyPASC.csv'
        args.processed_data_file2 = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all-anyPASC.csv'

    else:
        raise ValueError

    args.data_dir = r'output/dataset/{}/{}/'.format(args.dataset, args.encode)
    args.out_dir = r'output/factors/{}/{}/'.format(args.dataset, args.encode)

    # args.processed_data_file = r'output/dataset/{}/df_cohorts_covid_4manuNegNoCovidV2_bool_all-PosOnly-{}.csv'.format(
    #     args.dataset, args.encode)

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.out_dir)
    return args


def build_incident_pasc_from_all_positive(data_file, dump=False):
    start_time = time.time()
    print('In build_data_from_all_positive')
    print('Step1: Load Covid positive data  file:', data_file)
    df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])

    # df = df.drop(columns=['Unnamed: 0.1'])
    # df = df.loc[(df['covid'] == 1), :]
    # df.to_csv(args.data_file.replace('.csv', '-PosOnly.csv'))

    print('df.shape:', df.shape)
    print('Covid Positives:', (df['covid'] == 1).sum(), (df['covid'] == 1).mean())
    print('Covid Negative:', (df['covid'] == 0).sum(), (df['covid'] == 0).mean())

    # add selected incident PASC flag
    print('Step2: add selected incident PASC flag and time 2 event')
    df_pasc_info = pd.read_excel('output/causal_effects_specific_withMedication_v3.xlsx', sheet_name='diagnosis')
    selected_pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    print('len(selected_pasc_list)', len(selected_pasc_list))
    print(selected_pasc_list)

    selected_pasc_list_narrow = df_pasc_info.loc[df_pasc_info['selected_narrow'] == 1, 'pasc']
    print('len(selected_pasc_list_narrow)', len(selected_pasc_list_narrow))

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

    # 2022-05-25
    # build flag, t2e for any pasc with NARROW List
    # flag@pascname  for incidence label, dx-t2e@pascname for t2e which is shared from original data
    print('Any PASC: build flag, t2e for any pasc from Narrow List')
    specific_pasc_col_narrow = ['flag@' + x for x in selected_pasc_list_narrow]
    n_pasc_series_narrow = df[specific_pasc_col_narrow].sum(axis=1)
    df['pasc-narrow-count'] = n_pasc_series_narrow  # number of incident pascs of this person
    df['pasc-narrow-flag'] = (n_pasc_series_narrow > 0).astype('int')  # indicator of any incident pasc of this person
    # df['pasc-narrow-min-t2e'] = 180
    #
    # for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
    #     npasc = rows['pasc-narrow-count']
    #     if npasc >= 1:
    #         # if there are any incident pasc, t2e of any pasc is the earliest time of incident pasc
    #         pasc_flag_cols = list(rows[specific_pasc_col_narrow][rows[specific_pasc_col_narrow] > 0].index)
    #         pasc_t2e_cols = [x.replace('flag@', 'dx-t2e@') for x in pasc_flag_cols]
    #         t2e = rows.loc[pasc_t2e_cols].min()
    #     else:
    #         # if no incident pasc, t2e of any pasc: event, death, censoring, 180 days followup, whichever came first.
    #         # no event, only consider death, censoring, 180 days,
    #         # 1. approximated by the maximum-t2e of any selected pasc .
    #         #   unless all selected pasc happened, but not incident, this not happened in our data.
    #         # 2. directly follow the definition. Because I also stored max-followup information
    #         # t2e = rows.loc[['dx-t2e@' + x for x in selected_pasc_list]].max()
    #         t2e = max(30, np.min([rows['death t2e'], rows['maxfollowup'], 180]))
    #
    #     df.loc[index, 'pasc-narrow-min-t2e'] = t2e

    # before 2022-05-20
    # build flag, t2e for any pasc from broader list, all the follows are from broader list
    # flag@pascname  for incidence label, dx-t2e@pascname for t2e which is shared from original data
    print('Any PASC: build flag, t2e for any pasc')
    # specific_pasc_col = [x for x in df.columns if x.startswith('flag@')]
    specific_pasc_col = ['flag@' + x for x in selected_pasc_list]
    n_pasc_series = df[specific_pasc_col].sum(axis=1)
    df['pasc-count'] = n_pasc_series  # number of incident pascs of this person
    df['pasc-flag'] = (n_pasc_series > 0).astype('int')  # indicator of any incident pasc of this person
    # df['pasc-min-t2e'] = 180
    # for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
    #     npasc = rows['pasc-count']
    #     if npasc >= 1:
    #         # if there are any incident pasc, t2e of any pasc is the earliest time of incident pasc
    #         pasc_flag_cols = list(rows[specific_pasc_col][rows[specific_pasc_col] > 0].index)
    #         pasc_t2e_cols = [x.replace('flag@', 'dx-t2e@') for x in pasc_flag_cols]
    #         t2e = rows.loc[pasc_t2e_cols].min()
    #     else:
    #         # if no incident pasc, t2e of any pasc: event, death, censoring, 180 days followup, whichever came first.
    #         # no event, only consider death, censoring, 180 days,
    #         # 1. approximated by the maximum-t2e of any selected pasc .
    #         #   unless all selected pasc happened, but not incident, this not happened in our data.
    #         # 2. directly follow the definition. Because I also stored max-followup information
    #         # t2e = rows.loc[['dx-t2e@' + x for x in selected_pasc_list]].max()
    #         t2e = max(30, np.min([rows['death t2e'], rows['maxfollowup'], 180]))
    #
    #     df.loc[index, 'pasc-min-t2e'] = t2e

    # build ANY Severe v.s. non-severe PASC part, 2022-May-11

    specific_pasc_col_prevalence = ['dx-out@' + x for x in selected_pasc_list]
    n_pasc_series_pre = df[specific_pasc_col_prevalence].sum(axis=1)
    df['pasc-prevalence-count'] = n_pasc_series_pre  # number of incident pascs of this person
    df['pasc-prevalence-flag'] = (n_pasc_series_pre > 0).astype('int')  # indicator of any incident pasc of this person

    # if dump:
    #     utils.check_and_mkdir(args.processed_data_file)
    #     df.to_csv(args.processed_data_file, index=False)
    #     print('Dump to:', args.processed_data_file)

    print('build_data_from_all_positive Done! Total Time used:',
          time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return df, df_pasc_info


def print_incidence_table(cneg, cpos, flag_col, per=100000):
    neg_total = len(cneg)
    neg_yes = len(cneg.loc[cneg[flag_col] > 0, :])
    neg_no = len(cneg.loc[cneg[flag_col] == 0, :])

    pos_total = len(cpos)
    pos_yes = len(cpos.loc[cpos[flag_col] > 0, :])
    pos_no = len(cpos.loc[cpos[flag_col] == 0, :])

    print("\tNo\t\tYes\t\tTotal\t\t Per{}".format(per))
    print("Neg\t{}\t{}\t{}\t{:.2f}".format(neg_no, neg_yes, neg_total, neg_yes / neg_total * per))
    print("Pos\t{}\t{}\t{}\t{:.2f}".format(pos_no, pos_yes, pos_total, pos_yes / pos_total * per))
    print("Tot\t{}\t{}\t{}\t{:.2f}".format(pos_no + neg_no,
                                           pos_yes + neg_yes,
                                           pos_total + neg_total,
                                           (pos_yes + neg_yes) / (pos_total + neg_total) * per))


if __name__ == '__main__':
    # python screen_risk_factors.py --dataset INSIGHT --encode elix 2>&1 | tee  log/screen_anyPASC-risk_factors-insight-elix.txt
    # python screen_risk_factors.py --dataset OneFlorida --encode elix 2>&1 | tee  log/screen_anyPASC-risk_factors-OneFlorida-elix.txt

    start_time = time.time()
    args = parse_args()

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    # -Pre step2: build Covid Positive data and dump for future use
    df, df_pasc_info = build_incident_pasc_from_all_positive(args.data_file)

    print('Process done, df.shape:', df.shape)
    print('Covid Positives:', (df['covid'] == 1).sum(), (df['covid'] == 1).mean())
    print('Covid Negative:', (df['covid'] == 0).sum(), (df['covid'] == 0).mean())

    if args.dataset == 'Pooled':
        print('Load Second data file:', args.data_file2)
        df2, _ = build_incident_pasc_from_all_positive(args.data_file2)
        print('Load done, df2.shape:', df2.shape)
        df = df.append(df2, ignore_index=True)
        print('df.append(df2, ignore_index=True) done, df.shape:', df.shape)
        print('Covid Positives:', (df['covid'] == 1).sum(), (df['covid'] == 1).mean())
        print('Covid Negative:', (df['covid'] == 0).sum(), (df['covid'] == 0).mean())

    if args.severity == '1stwave':
        print('Considering patients in 1st wave, Mar-1-2020 to Sep.-30-2020')
        df = df.loc[(df['index date'] >= datetime(2020, 3, 1, 0, 0)) & (df['index date'] < datetime(2020, 10, 1, 0, 0)), :].copy()
    elif args.severity == 'delta':
        print('Considering patients in Delta wave, June-1-2021 to Nov.-30-2021')
        df = df.loc[(df['index date'] >= datetime(2021, 6, 1, 0, 0)) & (df['index date'] < datetime(2021, 12, 1, 0, 0)), :].copy()
    else:
        print('Considering ALL cohorts')


    # Incidence table
    print('Incidence Table')
    print_incidence_table(df.loc[(df['covid'] == 0), :],
                          df.loc[(df['covid'] == 1), :], 'pasc-flag', per=1000)
    # Prevalence table
    print('Prevalence Table')
    print_incidence_table(df.loc[(df['covid'] == 0), :],
                          df.loc[(df['covid'] == 1), :], 'pasc-prevalence-flag', per=1000)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    print('Done')
