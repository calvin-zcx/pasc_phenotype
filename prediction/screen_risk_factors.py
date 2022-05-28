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

# from mlxtend.preprocessing import TransactionEncoder
# from mlxtend.frequent_patterns import apriori
KFOLD = 5
MIN_PERCENTAGE = 0.005
N_SHUFFLE = 5


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['OneFlorida', 'INSIGHT', 'Pooled'], default='OneFlorida',
                        help='data bases')
    parser.add_argument('--encode', choices=['elix', 'icd_med'], default='elix',
                        help='data encoding')
    parser.add_argument('--population', choices=['positive', 'negative', 'all'], default='positive')
    parser.add_argument('--severity', choices=['all', 'outpatient', "inpatienticu",
                                               'inpatient', 'icu', 'ventilation', ], default='all')
    parser.add_argument('--goal', choices=['anypasc', 'allpasc', 'anyorgan', 'allorgan',
                                           'anypascsevere', 'anypascmoderate'],
                        default='anypascsevere')
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


def build_incident_pasc_from_all_positive(args, dump=True):
    start_time = time.time()
    print('In build_data_from_all_positive')
    print('Step1: Load Covid positive data  file:', args.data_file)
    df = pd.read_csv(args.data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
    # df = df.drop(columns=['Unnamed: 0.1'])
    # df = df.loc[(df['covid'] == 1), :]
    # df.to_csv(args.data_file.replace('.csv', '-PosOnly.csv'))
    print('df.shape:', df.shape)
    print('Covid Positives:', (df['covid'] == 1).sum(), (df['covid'] == 1).mean())
    print('Covid Negative:', (df['covid'] == 0).sum(), (df['covid'] == 0).mean())

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
    df_pasc_info = pd.read_excel('output/causal_effects_specific_withMedication_v3.xlsx', sheet_name='diagnosis')
    selected_pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    print('len(selected_pasc_list)', len(selected_pasc_list))
    print(selected_pasc_list)

    selected_pasc_list_narrow = df_pasc_info.loc[df_pasc_info['selected_narrow'] == 1, 'pasc']
    print('len(selected_pasc_list_narrow)', len(selected_pasc_list_narrow))

    pasc_severe_list = df_pasc_info.loc[(df_pasc_info['selected'] == 1) & (df_pasc_info['severity'] == 1), 'pasc']
    pasc_moderate_list = df_pasc_info.loc[(df_pasc_info['selected'] == 1) & (df_pasc_info['severity'] == 0), 'pasc']
    print('len(pasc_severe_list)', len(pasc_severe_list))
    print('len(pasc_moderate_list)', len(pasc_moderate_list))

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

    # 2022-05-25
    # build flag, t2e for any pasc with NARROW List
    # flag@pascname  for incidence label, dx-t2e@pascname for t2e which is shared from original data
    print('Any PASC: build flag, t2e for any pasc from Narrow List')
    specific_pasc_col_narrow = ['flag@' + x for x in selected_pasc_list_narrow]
    n_pasc_series_narrow = df[specific_pasc_col_narrow].sum(axis=1)
    df['pasc-narrow-count'] = n_pasc_series_narrow  # number of incident pascs of this person
    df['pasc-narrow-flag'] = (n_pasc_series_narrow > 0).astype('int')  # indicator of any incident pasc of this person
    df['pasc-narrow-min-t2e'] = 180

    for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
        npasc = rows['pasc-narrow-count']
        if npasc >= 1:
            # if there are any incident pasc, t2e of any pasc is the earliest time of incident pasc
            pasc_flag_cols = list(rows[specific_pasc_col_narrow][rows[specific_pasc_col_narrow] > 0].index)
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

        df.loc[index, 'pasc-narrow-min-t2e'] = t2e

    # before 2022-05-20
    # build flag, t2e for any pasc from broader list, all the follows are from broader list
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

    # build ANY Severe v.s. non-severe PASC part, 2022-May-11
    print('Any Sever or Moderate PASC: build flag, t2e for any Sever or Moderate pasc')
    severe_pasc_col = ['flag@' + x for x in pasc_severe_list]
    moderate_pasc_col = ['flag@' + x for x in pasc_moderate_list]

    n_severe_pasc_series = df[severe_pasc_col].sum(axis=1)
    df['pasc-severe-count'] = n_severe_pasc_series  # number of incident severe pascs of this person
    df['pasc-severe-flag'] = (n_severe_pasc_series > 0).astype(
        'int')  # indicator of any incident severe pasc of this person
    df['pasc-severe-min-t2e'] = 180

    n_moderate_pasc_series = df[moderate_pasc_col].sum(axis=1)
    df['pasc-moderate-count'] = n_moderate_pasc_series  # number of incident moderate pascs of this person
    df['pasc-moderate-flag'] = (n_moderate_pasc_series > 0).astype(
        'int')  # indicator of any incident moderate pasc of this person
    df['pasc-moderate-min-t2e'] = 180

    for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
        npasc = rows['pasc-severe-count']
        if npasc >= 1:
            # if there are any incident pasc, t2e of any pasc is the earliest time of incident pasc
            pasc_flag_cols = list(rows[severe_pasc_col][rows[severe_pasc_col] > 0].index)
            pasc_t2e_cols = [x.replace('flag@', 'dx-t2e@') for x in pasc_flag_cols]
            t2e = rows.loc[pasc_t2e_cols].min()
        else:
            t2e = max(30, np.min([rows['death t2e'], rows['maxfollowup'], 180]))

        df.loc[index, 'pasc-severe-min-t2e'] = t2e

        npasc = rows['pasc-moderate-count']
        if npasc >= 1:
            pasc_flag_cols = list(rows[moderate_pasc_col][rows[moderate_pasc_col] > 0].index)
            pasc_t2e_cols = [x.replace('flag@', 'dx-t2e@') for x in pasc_flag_cols]
            t2e = rows.loc[pasc_t2e_cols].min()
        else:
            t2e = max(30, np.min([rows['death t2e'], rows['maxfollowup'], 180]))

        df.loc[index, 'pasc-moderate-min-t2e'] = t2e

    # there can be 3 cases: only severe, both severe and moderate, and only moderate
    # the above 2 type columns account for: has severe (only severe + both), has moderate (both + only moderate)
    #
    df['pasc-moderateonly-flag'] = ((df['pasc-moderate-flag'] - df['pasc-severe-flag']) > 0).astype('int')
    # df['pasc-moderateonly-count'] = df['pasc-moderate-count']
    # df['pasc-moderateonly-min-t2e'] = df['pasc-moderate-min-t2e']

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

    if dump:
        utils.check_and_mkdir(args.processed_data_file)
        df.to_csv(args.processed_data_file, index=False)
        print('Dump to:', args.processed_data_file)

    print('build_data_from_all_positive Done! Total Time used:',
          time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return df, df_pasc_info


def distribution_statistics(args, df, df_pasc_info):
    print("df.shape", df.shape)
    specific_pasc_col = [x for x in df.columns if x.startswith('flag@')]
    # pasc_person_counts = df[specific_pasc_col].sum().reset_index().rename(columns={'index': "pasc", 0: "count"})
    pasc_person_counts = pd.DataFrame({'count': df[specific_pasc_col].sum(),
                                       'mean': df[specific_pasc_col].mean(),
                                       'per1k': df[specific_pasc_col].mean() * 1000}).reset_index().rename(
        columns={'index': "pasc"})

    pasc_person_counts['pasc'] = pasc_person_counts['pasc'].apply(lambda x: x.split("@")[-1])
    df_selected_pasc = df_pasc_info.loc[df_pasc_info['selected'] == 1, :]
    df_pasc_person_counts = pd.merge(pasc_person_counts,
                                     df_selected_pasc[['i', 'pasc', 'PASC Name Simple', 'Notes',
                                                       'selected', 'Organ Domain', 'Original CCSR Domain']],
                                     left_on='pasc', right_on='pasc', how='left')

    out_dir = r'output/dataset/{}/stats/'.format(args.dataset)
    utils.check_and_mkdir(out_dir)
    df_pasc_person_counts.to_csv(out_dir + 'pasc_person_counts_{}.csv'.format(args.dataset))
    df_person_pasc_counts = df['pasc-count'].rename("count")
    df_person_pasc_counts.to_csv(out_dir + 'person_pasc_counts_{}.csv'.format(args.dataset))
    return df_pasc_person_counts, df_person_pasc_counts


def collect_feature_columns_4_risk_analysis(args, df):
    col_names = []
    # if args.severity == 'all':
    # col_names += ['hospitalized', 'ventilation', 'criticalcare']
    col_names += ['not hospitalized', 'hospitalized', 'icu']

    # col_names += ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
    col_names += ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years']

    # col_names += ['Female', 'Male', 'Other/Missing']
    col_names += ['Female', 'Male']

    # col_names += ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    col_names += ['Asian', 'Black or African American', 'White', 'Other']

    # col_names += ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    col_names += ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']

    # col_names += ['inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits 3-4', 'inpatient visits >=5',
    #               'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits 3-4', 'outpatient visits >=5',
    #               'emergency visits 0', 'emergency visits 1-2', 'emergency visits 3-4', 'emergency visits >=5']
    # col_names += ['inpatient visits 0', 'inpatient visits 1-4', 'inpatient visits >=5',
    #               'emergency visits 0', 'emergency visits 1-4', 'emergency visits >=5']
    col_names += ['inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits >=3',
                  'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits >=3',
                  'emergency visits 0', 'emergency visits 1-2', 'emergency visits >=3']

    # col_names += ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49', 'ADI50-59', 'ADI60-69', 'ADI70-79',
    #               'ADI80-89', 'ADI90-100']
    col_names += ['ADI1-19', 'ADI20-39', 'ADI40-59', 'ADI60-79', 'ADI80-100']

    col_names += ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ',
                  'BMI: >=30 obese ', 'BMI: missing']

    col_names += ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']

    col_names += ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21', '07/21-11/21']

    col_names += ['num_Comorbidity=0', 'num_Comorbidity=1', 'num_Comorbidity=2', 'num_Comorbidity=3',
                  'num_Comorbidity=4', 'num_Comorbidity>=5']

    if args.encode == 'icd_med':
        col_names += list(df.columns)[df.columns.get_loc('death t2e') + 1:df.columns.get_loc('label')]
    else:
        col_names += ["DX: Alcohol Abuse", "DX: Anemia", "DX: Arrythmia", "DX: Asthma", "DX: Cancer",
                      "DX: Chronic Kidney Disease", "DX: Chronic Pulmonary Disorders", "DX: Cirrhosis",
                      "DX: Coagulopathy", "DX: Congestive Heart Failure",
                      "DX: COPD", "DX: Coronary Artery Disease", "DX: Dementia", "DX: Diabetes Type 1",
                      "DX: Diabetes Type 2", "DX: End Stage Renal Disease on Dialysis", "DX: Hemiplegia",
                      "DX: HIV", "DX: Hypertension", "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis",
                      "DX: Inflammatory Bowel Disorder", "DX: Lupus or Systemic Lupus Erythematosus",
                      "DX: Mental Health Disorders", "DX: Multiple Sclerosis", "DX: Parkinson's Disease",
                      "DX: Peripheral vascular disorders ", "DX: Pregnant",
                      "DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)",
                      "DX: Rheumatoid Arthritis", "DX: Seizure/Epilepsy",
                      "DX: Severe Obesity  (BMI>=40 kg/m2)", "DX: Weight Loss",
                      "DX: Down's Syndrome", 'DX: Other Substance Abuse', 'DX: Cystic Fibrosis',
                      'DX: Autism', 'DX: Sickle Cell',
                      'DX: Obstructive sleep apnea',  # added 2022-05-25
                      'DX: Epstein-Barr and Infectious Mononucleosis (Mono)',  # added 2022-05-25
                      'DX: Herpes Zoster',  # added 2022-05-25
                      ]

        col_names += ["MEDICATION: Corticosteroids", "MEDICATION: Immunosuppressant drug"]

    # add at 2022-05-25
    col_names += ['Fully vaccinated - Pre-index', 'Partially vaccinated - Pre-index', 'No evidence - Pre-index',]

    # col_names += [
    #         'Anti-platelet Therapy', 'Aspirin', 'Baricitinib', 'Bamlanivimab Monoclonal Antibody Treatment',
    #         'Bamlanivimab and Etesevimab Monoclonal Antibody Treatment',
    #         'Casirivimab and Imdevimab Monoclonal Antibody Treatment',
    #         'Any Monoclonal Antibody Treatment (Bamlanivimab, Bamlanivimab and Etesevimab, Casirivimab and Imdevimab, '
    #         'Sotrovimab, and unspecified monoclonal antibodies)',
    #         'Colchicine', 'Corticosteroids', 'Dexamethasone', 'Factor Xa Inhibitors', 'Fluvoxamine', 'Heparin',
    #         'Inhaled Steroids', 'Ivermectin', 'Low Molecular Weight Heparin', 'Molnupiravir', 'Nirmatrelvir',
    #         'Paxlovid', 'Remdesivir', 'Ritonavir', 'Sotrovimab Monoclonal Antibody Treatment',
    #         'Thrombin Inhibitors', 'Tocilizumab (Actemra)', 'PX: Convalescent Plasma']

    print('encoding:', args.encode, 'len(col_names):', len(col_names))
    print(col_names)
    return col_names


def pre_transform_feature(df):
    # col_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49', 'ADI50-59', 'ADI60-69', 'ADI70-79',
    #              'ADI80-89', 'ADI90-100']
    df['ADI1-19'] = (df['ADI1-9'] + df['ADI10-19'] >= 1).astype('int')
    df['ADI20-39'] = (df['ADI20-29'] + df['ADI30-39'] >= 1).astype('int')
    df['ADI40-59'] = (df['ADI40-49'] + df['ADI50-59'] >= 1).astype('int')
    df['ADI60-79'] = (df['ADI60-69'] + df['ADI70-79'] >= 1).astype('int')
    df['ADI80-100'] = (df['ADI80-89'] + df['ADI90-100'] >= 1).astype('int')

    df['75+ years'] = (df['75-<85 years'] + df['85+ years'] >= 1).astype('int')

    # df['inpatient visits 1-4'] = (df['inpatient visits 1-2'] + df['inpatient visits 3-4'] >= 1).astype('int')
    # df['outpatient visits 1-4'] = (df['outpatient visits 1-2'] + df['outpatient visits 3-4'] >= 1).astype('int')
    # df['emergency visits 1-4'] = (df['emergency visits 1-2'] + df['emergency visits 3-4'] >= 1).astype('int')
    df['inpatient visits >=3'] = (df['inpatient visits >=5'] + df['inpatient visits 3-4'] >= 1).astype('int')
    df['outpatient visits >=3'] = (df['outpatient visits >=5'] + df['outpatient visits 3-4'] >= 1).astype('int')
    df['emergency visits >=3'] = (df['emergency visits >=5'] + df['emergency visits 3-4'] >= 1).astype('int')

    df['not hospitalized'] = 1 - df['hospitalized']
    df['icu'] = ((df['ventilation'] + df['criticalcare']) >= 1).astype('int')

    return df


#
# def risk_factor_of_pasc(args, pasc_name, dump=True):
#     infile = args.data_dir + pasc_name + '_{}'.format(
#         'dx_med_' if args.encode == 'icd_med' else '') + args.dataset + '.csv'
#     print('In risk_factor_of_pasc:')
#     pasc = pasc_name.replace('_', '/')
#     print('PASC:', pasc, 'Infile:', infile)
#
#     df = pd.read_csv(infile)
#     print('df.shape:', df.shape)
#     df = pre_transform_feature(df)
#     print('df.shape after pre_transform_feature:', df.shape)
#     # df_label = df['covid']
#
#     covs_columns = collect_feature_columns_4_risk_analysis(args, df)
#
#     if pasc == 'Neurocognitive disorders':
#         covs_columns = [x for x in covs_columns if x != "DX: Dementia"]
#     elif pasc == 'Diabetes mellitus with complication':
#         covs_columns = [x for x in covs_columns if x != 'DX: Diabetes Type 2']
#
#     pasc_flag = df['dx-out@' + pasc]
#     pasc_t2e = df['dx-t2e@' + pasc]  # .astype('float')
#     pasc_baseline = df['dx-base@' + pasc]
#
#     death_flag = df['death']
#     death_t2e = df['death t2e']
#     # pasc_flag.loc[(death_t2e == pasc_t2e)] = 2
#     print('#death:', (death_t2e == pasc_t2e).sum(),
#           '#death:', (death_t2e == pasc_t2e).sum(),
#           'ratio of death:', (death_t2e == pasc_t2e).mean())
#
#     cox_data = df.loc[:, covs_columns]
#     print('cox_data.shape before number filter:', cox_data.shape)
#     cox_data = cox_data.loc[:, cox_data.columns[(cox_data.mean() >= 0.001) & (cox_data.mean() < 1)]]
#     print('cox_data.shape after number filter:', cox_data.shape)
#
#     model = ml.CoxPrediction(random_seed=args.random_seed).cross_validation_fit(
#         cox_data, pasc_t2e, pasc_flag, kfold=5, scoring_method="concordance_index")
#
#     model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[], pre='uni-')
#     model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
#         '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
#         'Female', 'Male'], pre='ageSex-')
#     model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
#         '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
#         'Female', 'Male', 'hospitalized', 'ventilation', 'criticalcare'], pre='ageSexAcute-')
#     if dump:
#         utils.check_and_mkdir(args.out_dir)
#         model.risk_results.reset_index().sort_values(by=['HR'], ascending=False).to_csv(
#             args.out_dir + pasc_name + '-riskFactor.csv')
#         model.results.sort_values(by=['E[fit]'], ascending=False).to_csv(
#             args.out_dir + pasc_name + '-modeSelection.csv')
#
#     return model


def risk_factor_of_any_pasc(args, df, df_pasc_info, pasc_threshold=1, dump=True):
    print('in risk_factor_of_any_pasc, PASC is defined by >=', pasc_threshold)
    # print('df.shape:', df.shape)
    # df = pre_transform_feature(df)
    # print('df.shape after pre_transform_feature:', df.shape)
    covs_columns = collect_feature_columns_4_risk_analysis(args, df)

    pasc_flag = (df['pasc-count'] >= pasc_threshold).astype('int')
    pasc_t2e = df['pasc-min-t2e']  # this time 2 event can only be used for >= 1 pasc. If >= 2, how to define t2e?
    print('pos:{} ({:.3%})'.format(pasc_flag.sum(), pasc_flag.mean()),
          'neg:{} ({:.3%})'.format((1 - pasc_flag).sum(), (1 - pasc_flag).mean()))
    # 1 pasc --> the earliest; 2 pasc --> 2nd earliest, et.c
    if pasc_threshold >= 2:
        print('pasc thereshold >=', pasc_threshold, 't2e is defined as the ', pasc_threshold, 'th earliest events time')
        df['pasc-min-t2e'] = 180
        specific_pasc_col = [x for x in df.columns if x.startswith('flag@')]
        for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
            npasc = rows['pasc-count']
            if npasc >= pasc_threshold:
                # if at least pasc_threshold pasc occur, t2e is the pasc_threshold^th earlist time
                pasc_flag_cols = list(rows[specific_pasc_col][rows[specific_pasc_col] > 0].index)
                pasc_t2e_cols = [x.replace('flag@', 'dx-t2e@') for x in pasc_flag_cols]
                # t2e = rows[pasc_t2e_cols].min()
                time_vec = sorted(rows[pasc_t2e_cols])
                t2e = time_vec[min(len(time_vec) - 1, pasc_threshold - 1)]
            else:
                # if events number < pasc_threshold occur, e.g. pasc_threshold=2, but only 1 event happened,
                # then 2-event pasc did not happen
                # t2e is the event, death, censoring, 180 days, whichever came first.
                # t2e = rows.loc[[x.replace('flag@', 'dx-t2e@') for x in specific_pasc_col]].max()
                t2e = max(30, np.min([rows['death t2e'], rows['maxfollowup'], 180]))

            df.loc[index, 'pasc-min-t2e'] = t2e

    # support >= 2,3,4... by updating pasc-min-t2e definition.
    # pasc_name = 'Heart failure'
    # pasc_flag = df['flag@'+pasc_name]
    # pasc_t2e = df['dx-t2e@'+pasc_name]

    cox_data = df.loc[:, covs_columns]
    print('cox_data.shape before number filter:', cox_data.shape)
    cox_data = cox_data.loc[:, cox_data.columns[(cox_data.mean() >= MIN_PERCENTAGE) & (cox_data.mean() < 1)]]
    print('cox_data.shape after number filter:', cox_data.shape)

    model = ml.CoxPrediction(random_seed=args.random_seed, ).cross_validation_fit(
        cox_data, pasc_t2e, pasc_flag, kfold=KFOLD, n_shuffle=N_SHUFFLE, scoring_method="concordance_index")
    # paras_grid={'l1_ratio': [0], 'penalizer': [0.1]}

    model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[], pre='uni-')
    model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
        '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years'], pre='age-')
    if args.severity == 'all':
        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
            '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
            'not hospitalized', 'hospitalized', 'icu'], pre='ageAcute-')
        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
            '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
            'not hospitalized', 'hospitalized', 'icu', 'Female', 'Male', ], pre='ageAcuteSex-')

    if dump:
        utils.check_and_mkdir(args.out_dir + 'any_pasc/')
        model.risk_results.reset_index().sort_values(by=['HR'], ascending=False).to_csv(
            args.out_dir + 'any_pasc/any-at-least-{}-pasc-riskFactor-{}-{}-{}.csv'.format(
                pasc_threshold, args.dataset, args.population, args.severity))
        model.results.sort_values(by=['E[fit]'], ascending=False).to_csv(
            args.out_dir + 'any_pasc/any-at-least-{}-pasc-modeSelection-{}-{}-{}.csv'.format(
                pasc_threshold, args.dataset, args.population, args.severity))

    return model


def risk_factor_of_any_organ(args, df, df_pasc_info, organ_threshold=1, dump=True):
    print('Organ is defined by >=', organ_threshold)

    # print('df.shape:', df.shape)
    # df = pre_transform_feature(df)
    # print('df.shape after pre_transform_feature:', df.shape)
    covs_columns = collect_feature_columns_4_risk_analysis(args, df)

    pasc_flag = (df['organ-count'] >= organ_threshold).astype('int')
    print('pos:{} ({:.3%})'.format(pasc_flag.sum(), pasc_flag.mean()),
          'neg:{} ({:.3%})'.format((1 - pasc_flag).sum(), (1 - pasc_flag).mean()))

    pasc_t2e = df['organ-min-t2e']
    # 1 pasc --> the earliest; 2 pasc --> 2nd earliest, et.c
    if organ_threshold >= 2:
        print('organ_threshold >=', organ_threshold, 't2e is defined as the ', organ_threshold,
              'th earliest events time')
        df['organ-min-t2e'] = 180
        specific_organ_col = [x for x in df.columns if x.startswith('organ-flag@')]
        for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
            norgan = rows['organ-count']
            if norgan >= organ_threshold:
                organ_flag_cols = list(rows[specific_organ_col][rows[specific_organ_col] > 0].index)
                organ_t2e_cols = [x.replace('organ-flag@', 'organ-t2e@') for x in organ_flag_cols]
                # t2e = rows[organ_t2e_cols].min()
                time_vec = sorted(rows[organ_t2e_cols])
                t2e = time_vec[min(len(time_vec) - 1, organ_threshold - 1)]
            else:
                t2e = max(30, np.min([rows['death t2e'], rows['maxfollowup'], 180]))

            df.loc[index, 'organ-min-t2e'] = t2e

    # support >= 2,3,4... by updating pasc-min-t2e definition.
    cox_data = df.loc[:, covs_columns]
    print('cox_data.shape before number filter:', cox_data.shape)
    cox_data = cox_data.loc[:, cox_data.columns[(cox_data.mean() >= MIN_PERCENTAGE) & (cox_data.mean() < 1)]]
    print('cox_data.shape after number filter:', cox_data.shape)

    model = ml.CoxPrediction(random_seed=args.random_seed, ).cross_validation_fit(
        cox_data, pasc_t2e, pasc_flag, kfold=KFOLD, n_shuffle=N_SHUFFLE, scoring_method="concordance_index")

    model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[], pre='uni-')
    model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
        '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years'], pre='age-')
    model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
        '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
        'not hospitalized', 'hospitalized', 'icu'], pre='ageAcute-')
    model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
        '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
        'not hospitalized', 'hospitalized', 'icu', 'Female', 'Male', ], pre='ageAcuteSex-')
    if dump:
        utils.check_and_mkdir(args.out_dir + 'any_organ/')
        model.risk_results.reset_index().sort_values(by=['HR'], ascending=False).to_csv(
            args.out_dir + 'any_organ/any-at-least-{}-ORGAN-riskFactor-{}-{}-{}.csv'.format(organ_threshold,
                                                                                            args.dataset,
                                                                                            args.population,
                                                                                            args.severity))
        model.results.sort_values(by=['E[fit]'], ascending=False).to_csv(
            args.out_dir + 'any_organ/any-at-least-{}-ORGAN-modeSelection-{}-{}-{}.csv'.format(organ_threshold,
                                                                                               args.dataset,
                                                                                               args.population,
                                                                                               args.severity))

    return model


def risk_factor_of_any_pasc_severity(args, df, df_pasc_info, severe=True, pasc_threshold=1, dump=True):
    print('in risk_factor_of_any_pasc severity, severe or moderate PASC is defined by >=', pasc_threshold)

    # print('df.shape:', df.shape)
    # df = pre_transform_feature(df)
    # print('df.shape after pre_transform_feature:', df.shape)
    covs_columns = collect_feature_columns_4_risk_analysis(args, df)

    if severe:
        print('Focusing on Severe PASC')
        pasc_flag = (df['pasc-severe-count'] >= pasc_threshold).astype('int')
        pasc_t2e = df['pasc-severe-min-t2e']
        t2ecolname = 'pasc-severe-min-t2e'
        pasccntname = 'pasc-severe-count'
        # this time 2 event can only be used for >= 1 pasc. If >= 2, how to define t2e?
        specific_pasc_col = df_pasc_info.loc[(df_pasc_info['selected'] == 1) & (df_pasc_info['severity'] == 1), 'pasc']
        print('len(pasc_severe_list)', len(specific_pasc_col))

    else:
        print('Focusing on Moderate PASC')
        # moderate flag: only moderate, no severe
        pasc_flag = (df['pasc-moderate-count'] >= pasc_threshold).astype('int')
        pasc_flag = ((pasc_flag - df['pasc-severe-flag']) >= 1).astype('int')
        pasc_t2e = df['pasc-moderate-min-t2e']
        t2ecolname = 'pasc-moderate-min-t2e'
        pasccntname = 'pasc-moderate-count'
        specific_pasc_col = df_pasc_info.loc[(df_pasc_info['selected'] == 1) & (df_pasc_info['severity'] == 0), 'pasc']
        print('len(pasc_moderate_list)', len(specific_pasc_col))

    specific_pasc_col = ['flag@' + x for x in specific_pasc_col]
    print('pos:{} ({:.3%})'.format(pasc_flag.sum(), pasc_flag.mean()),
          'neg:{} ({:.3%})'.format((1 - pasc_flag).sum(), (1 - pasc_flag).mean()))

    # 1 pasc --> the earliest; 2 pasc --> 2nd earliest, et.c
    if pasc_threshold >= 2:
        print('pasc thereshold >=', pasc_threshold, 't2e is defined as the ', pasc_threshold, 'th earliest events time')
        df[t2ecolname] = 180
        # replace this with severity-related PASC
        # specific_pasc_col = [x for x in df.columns if x.startswith('flag@')]
        for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
            npasc = rows[pasccntname]
            if npasc >= pasc_threshold:
                # if at least pasc_threshold pasc occur, t2e is the pasc_threshold^th earlist time
                pasc_flag_cols = list(rows[specific_pasc_col][rows[specific_pasc_col] > 0].index)
                pasc_t2e_cols = [x.replace('flag@', 'dx-t2e@') for x in pasc_flag_cols]
                # t2e = rows[pasc_t2e_cols].min()
                time_vec = sorted(rows[pasc_t2e_cols])
                t2e = time_vec[min(len(time_vec) - 1, pasc_threshold - 1)]
            else:
                # if events number < pasc_threshold occur, e.g. pasc_threshold=2, but only 1 event happened,
                # then 2-event pasc did not happen
                # t2e is the event, death, censoring, 180 days, whichever came first.
                # t2e = rows.loc[[x.replace('flag@', 'dx-t2e@') for x in specific_pasc_col]].max()
                t2e = max(30, np.min([rows['death t2e'], rows['maxfollowup'], 180]))

            df.loc[index, t2ecolname] = t2e

    cox_data = df.loc[:, covs_columns]
    print('cox_data.shape before number filter:', cox_data.shape)
    cox_data = cox_data.loc[:, cox_data.columns[(cox_data.mean() >= MIN_PERCENTAGE) & (cox_data.mean() < 1)]]
    print('cox_data.shape after number filter:', cox_data.shape)

    model = ml.CoxPrediction(random_seed=args.random_seed, ).cross_validation_fit(
        cox_data, pasc_t2e, pasc_flag, kfold=KFOLD, n_shuffle=N_SHUFFLE, scoring_method="concordance_index")
    # paras_grid={'l1_ratio': [0], 'penalizer': [0.1]}

    model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[], pre='uni-')
    model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
        '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years'], pre='age-')
    if args.severity == 'all':
        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
            '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
            'not hospitalized', 'hospitalized', 'icu'], pre='ageAcute-')
        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
            '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
            'not hospitalized', 'hospitalized', 'icu', 'Female', 'Male', ], pre='ageAcuteSex-')

    if dump:
        utils.check_and_mkdir(args.out_dir + 'any_pasc_{}/'.format('severe' if severe else 'moderate'))
        model.risk_results.reset_index().sort_values(by=['HR'], ascending=False).to_csv(
            args.out_dir + 'any_pasc_{}/any-at-least-{}-{}-pasc-riskFactor-{}-{}-{}.csv'.format(
                'severe' if severe else 'moderate',
                pasc_threshold, 'severe' if severe else 'moderate',
                args.dataset, args.population, args.severity))
        model.results.sort_values(by=['E[fit]'], ascending=False).to_csv(
            args.out_dir + 'any_pasc_{}/any-at-least-{}-{}-pasc-modeSelection-{}-{}-{}.csv'.format(
                'severe' if severe else 'moderate',
                pasc_threshold, 'severe' if severe else 'moderate',
                args.dataset, args.population, args.severity))

    return model


def screen_any_pasc(args, df, df_pasc_info):
    print('In screen_any_pasc, args: ', args)
    print('random_seed: ', args.random_seed)
    model_dict = {}
    for i in range(1, 9):
        print('screen_any_pasc, In threshold:', i)
        model = risk_factor_of_any_pasc(args, df, df_pasc_info, pasc_threshold=i, dump=True)
        model_dict[i] = model

    return model_dict


def screen_any_pasc_severity(args, df, df_pasc_info, severe):
    print('In screen_any_pasc_severity, args: ', args)
    print('screen_any_pasc_severity, severe:', severe)
    print('random_seed: ', args.random_seed)
    model_dict = {}
    for i in range(1, 9):
        print('screen_any_pasc_severity, In threshold:', i)
        model = risk_factor_of_any_pasc_severity(args, df, df_pasc_info, severe, pasc_threshold=i, dump=True)
        model_dict[i] = model

    return model_dict


def screen_any_organ(args, df, df_pasc_info):
    print('In screen_any_organ, args: ', args)
    print('random_seed: ', args.random_seed)
    model_dict = {}
    for i in range(1, 9):
        print('screen_any_pasc, In threshold:', i)
        model = risk_factor_of_any_organ(args, df, df_pasc_info, organ_threshold=i, dump=True)
        model_dict[i] = model

    return model_dict


def screen_all_organ(args, df, df_pasc_info, selected_organ_list, dump=True):
    print('In screen_all_organ, args: ', args)
    print('random_seed: ', args.random_seed)
    # print('df.shape:', df.shape)
    # df = pre_transform_feature(df)
    # print('df.shape after pre_transform_feature:', df.shape)
    covs_columns = collect_feature_columns_4_risk_analysis(args, df)

    # build flag, t2e for each organ
    print('Screening All Organ category')
    i = 0
    model_dict = {}
    for organ in tqdm(selected_organ_list, total=len(selected_organ_list)):
        i += 1
        print(i, 'screening:', organ)
        pasc_flag = df['organ-flag@' + organ]
        pasc_t2e = df['organ-t2e@' + organ]
        print('pos:{} ({:.3%})'.format(pasc_flag.sum(), pasc_flag.mean()),
              'neg:{} ({:.3%})'.format((1 - pasc_flag).sum(), (1 - pasc_flag).mean()))

        cox_data = df.loc[:, covs_columns]
        print('cox_data.shape before number filter:', cox_data.shape)
        cox_data = cox_data.loc[:, cox_data.columns[(cox_data.mean() >= MIN_PERCENTAGE) & (cox_data.mean() < 1)]]
        print('cox_data.shape after number filter:', cox_data.shape)

        model = ml.CoxPrediction(random_seed=args.random_seed, ).cross_validation_fit(
            cox_data, pasc_t2e, pasc_flag, kfold=KFOLD, n_shuffle=N_SHUFFLE, scoring_method="concordance_index")

        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[], pre='uni-')
        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
            '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years'], pre='age-')
        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
            '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
            'not hospitalized', 'hospitalized', 'icu'], pre='ageAcute-')
        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
            '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
            'not hospitalized', 'hospitalized', 'icu', 'Female', 'Male', ], pre='ageAcuteSex-')
        if dump:
            utils.check_and_mkdir(args.out_dir + 'every_organ/')
            model.risk_results.reset_index().sort_values(by=['HR'], ascending=False).to_csv(
                args.out_dir + 'every_organ/ORGAN-{}-riskFactor-{}-{}-{}.csv'.format(
                    organ, args.dataset, args.population, args.severity))
            model.results.sort_values(by=['E[fit]'], ascending=False).to_csv(
                args.out_dir + 'every_organ/ORGAN-{}-modeSelection-{}-{}-{}.csv'.format(
                    organ, args.dataset, args.population, args.severity))
            print('Dump done', organ)

        model_dict[organ] = model

    return model_dict


def screen_all_pasc(args, df, df_pasc_info, selected_pasc_list, dump=True):
    print('In screen_all_pasc, args: ', args)
    print('random_seed: ', args.random_seed)
    # print('df.shape:', df.shape)
    # df = pre_transform_feature(df)
    # print('df.shape after pre_transform_feature:', df.shape)
    covs_columns = collect_feature_columns_4_risk_analysis(args, df)

    # build flag, t2e for each organ
    print('Screening All PASC category')
    i = 0
    model_dict = {}
    for pasc in tqdm(selected_pasc_list, total=len(selected_pasc_list)):
        i += 1
        print(i, 'screening:', pasc)
        pasc_flag = df['flag@' + pasc]
        pasc_t2e = df['dx-t2e@' + pasc]

        print('pos:{} ({:.3%})'.format(pasc_flag.sum(), pasc_flag.mean()),
              'neg:{} ({:.3%})'.format((1 - pasc_flag).sum(), (1 - pasc_flag).mean()))

        cox_data = df.loc[:, covs_columns]
        print('cox_data.shape before number filter:', cox_data.shape)
        cox_data = cox_data.loc[:, cox_data.columns[(cox_data.mean() >= MIN_PERCENTAGE) & (cox_data.mean() < 1)]]
        print('cox_data.shape after number filter:', cox_data.shape)

        model = ml.CoxPrediction(random_seed=args.random_seed, ).cross_validation_fit(
            cox_data, pasc_t2e, pasc_flag, kfold=KFOLD, n_shuffle=N_SHUFFLE, scoring_method="concordance_index")

        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[], pre='uni-')
        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
            '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years'], pre='age-')
        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
            '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
            'not hospitalized', 'hospitalized', 'icu'], pre='ageAcute-')
        model.uni_variate_risk(cox_data, pasc_t2e, pasc_flag, adjusted_col=[
            '20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years',
            'not hospitalized', 'hospitalized', 'icu', 'Female', 'Male', ], pre='ageAcuteSex-')
        if dump:
            utils.check_and_mkdir(args.out_dir + 'every_pasc/')
            model.risk_results.reset_index().sort_values(by=['HR'], ascending=False).to_csv(
                args.out_dir + 'every_pasc/PASC-{}-riskFactor-{}-{}-{}.csv'.format(
                    pasc.replace('/', '_'), args.dataset, args.population, args.severity))
            model.results.sort_values(by=['E[fit]'], ascending=False).to_csv(
                args.out_dir + 'every_pasc/PASC-{}-modeSelection-{}-{}-{}.csv'.format(
                    pasc.replace('/', '_'), args.dataset, args.population, args.severity))
            print('Dump done', pasc)

        model_dict[pasc] = model

    return model_dict


def combination_of_pasc():
    # specific_pasc_col = [x for x in df.columns if x.startswith('flag@')]
    # pasc_name = {}
    # for index, row in df_pasc_info.iterrows():
    #     pasc_name['flag@' + row['pasc']] = row['PASC Name Simple']
    #
    # pasc_data = df.loc[(df['covid'] == 1) & (df['pasc-count'] >= 1), specific_pasc_col].rename(columns=pasc_name)

    # # te = TransactionEncoder()
    # # te_ary = te.fit(pasc_data).transform(pasc_data)
    # freitem = apriori(pasc_data, min_support=0.001, use_colnames=True, low_memory=True)
    # freitem['length'] = freitem['itemsets'].apply(lambda x: len(x))
    # freitem['itemsets'] = freitem['itemsets'].apply(lambda x: '; '.join(list(x)))
    # freitem['Occurrence'] = freitem['support'] * len(pasc_data)
    # freitem['Crude Incidence'] = freitem['support'] * len(pasc_data) / len(df.loc[df['covid'] == 1, :])
    # freitem.to_csv(args.out_dir + 'frequent_pasc-covid-positive.csv')

    # pasc_data = df.loc[(df['covid'] == 0) & (df['pasc-count'] >= 1), specific_pasc_col].rename(columns=pasc_name)
    # # te = TransactionEncoder()
    # # te_ary = te.fit(pasc_data).transform(pasc_data)
    # freitem2 = apriori(pasc_data, min_support=0.0001, use_colnames=True, low_memory=True)
    # freitem2['length'] = freitem2['itemsets'].apply(lambda x: len(x))
    # freitem2['itemsets'] = freitem2['itemsets'].apply(lambda x: '; '.join(list(x)))
    # freitem2['Occurrence'] = freitem2['support'] * len(pasc_data)
    # freitem2['Crude Incidence'] = freitem2['support'] * len(pasc_data) / len(df.loc[df['covid'] == 0, :])
    # freitem2.to_csv(args.out_dir + 'frequent_pasc-covid-negative.csv')
    #
    # freitem_combined = pd.merge(freitem, freitem2, left_on='itemsets', right_on='itemsets', how='left')
    # freitem_combined.to_csv(args.out_dir + 'frequent_pasc-combined.csv')
    #
    # print("done!")
    # print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # sys.exit(0)
    pass


if __name__ == '__main__':
    # python screen_risk_factors.py --dataset INSIGHT --encode elix 2>&1 | tee  log/screen_anyPASC-risk_factors-insight-elix.txt
    # python screen_risk_factors.py --dataset OneFlorida --encode elix 2>&1 | tee  log/screen_anyPASC-risk_factors-OneFlorida-elix.txt

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    # -Pre step1: select Covid Positive data and dump
    # read_all_and_dump_covid_positive(r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL.csv')
    # read_all_and_dump_covid_positive(r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all.csv')
    #
    # -Pre step2: build Covid Positive data and dump for future use
    # df, df_pasc_info = build_incident_pasc_from_all_positive(args)

    # sys.exit(0)

    # Step 1: Load pre-processed data for screening. May dynamically fine tune feature
    print('Load data file:', args.processed_data_file)
    df = pd.read_csv(args.processed_data_file, dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date'])  # , nrows=100
    print('Load done, df.shape:', df.shape)
    print('Covid Positives:', (df['covid'] == 1).sum(), (df['covid'] == 1).mean())
    print('Covid Negative:', (df['covid'] == 0).sum(), (df['covid'] == 0).mean())
    if args.dataset == 'Pooled':
        print('Load Second data file:', args.processed_data_file2)
        df2 = pd.read_csv(args.processed_data_file2, dtype={'patid': str, 'site': str, 'zip': str},
                          parse_dates=['index date'])  # , nrows=100
        print('Load done, df2.shape:', df2.shape)
        df = df.append(df2, ignore_index=True)
        print('df.append(df2, ignore_index=True) done, df.shape:', df.shape)
        print('Covid Positives:', (df['covid'] == 1).sum(), (df['covid'] == 1).mean())
        print('Covid Negative:', (df['covid'] == 0).sum(), (df['covid'] == 0).mean())

    # Step 2: Load pasc meta information
    df_pasc_info = pd.read_excel('output/causal_effects_specific_withMedication_v3.xlsx', sheet_name='diagnosis')
    selected_pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    print('len(selected_pasc_list)', len(selected_pasc_list))  # 44 pasc
    print(selected_pasc_list)
    selected_organ_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'Organ Domain'].unique()
    print('len(selected_organ_list)', len(selected_organ_list))

    specific_pasc_col = [x for x in df.columns if x.startswith('flag@')]
    pasc_name = {}
    for index, row in df_pasc_info.iterrows():
        pasc_name['flag@' + row['pasc']] = row['PASC Name Simple']

    # Step 3: set Covid pos, neg, or all population
    if args.population == 'positive':
        print('Using Covid positive  cohorts')
        df = df.loc[(df['covid'] == 1), :].copy()
    elif args.population == 'negative':
        print('Using Covid negative  cohorts')
        df = df.loc[(df['covid'] == 0), :].copy()
    else:
        print('Using Both Covid Positive and Negative  cohorts')

    print('Select population:', args.population, 'df.shape:', df.shape)

    # Step 4:
    print('Do feature transformation before severity stratification, df.shape:', df.shape)
    df = pre_transform_feature(df)
    print('df.shape after pre_transform_feature:', df.shape)

    # Step 5: set sub-population
    # focusing on: all, outpatient, inpatienticu (namely inpatient in a broad sense)
    # 'all', 'outpatient', 'inpatient', 'critical', 'ventilation'   can add more later, just these 4 for brevity
    if args.severity == 'outpatient':
        print('Considering outpatient cohorts')
        df = df.loc[(df['hospitalized'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif (args.severity == 'inpatienticu') or (args.severity == 'nonoutpatient'):
        print('Considering inpatient/hospitalized including icu cohorts, namely non-outpatient')
        df = df.loc[(df['hospitalized'] == 1) | (df['criticalcare'] == 1), :].copy()
    elif args.severity == 'inpatient':
        print('Considering inpatient/hospitalized cohorts but not ICU')
        df = df.loc[(df['hospitalized'] == 1) & (df['ventilation'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif args.severity == 'icu':
        KFOLD = 1
        print('using KFOLD=1 in the ICU case due to sample size issue')
        print('Considering ICU (hospitalized ventilation or critical care) cohorts')
        df = df.loc[(((df['hospitalized'] == 1) & (df['ventilation'] == 1)) | (df['criticalcare'] == 1)), :].copy()
    elif args.severity == 'ventilation':
        print('Considering (hospitalized) ventilation cohorts')
        df = df.loc[(df['hospitalized'] == 1) & (df['ventilation'] == 1), :].copy()
    else:
        print('Considering ALL cohorts')

    print('Select sub- cohorts:', args.severity, 'df.shape:', df.shape)

    # Step 6: Basic statistics
    pasc_flag = df['pasc-flag']  # (df['pasc-count'] >= 1).astype('int')
    pasc_t2e = df.loc[
        pasc_flag == 1, 'pasc-min-t2e']  # this time 2 event can only be used for >= 1 pasc. If >= 2, how to define t2e?
    print('PASC pos:{} ({:.3%})'.format(pasc_flag.sum(), pasc_flag.mean()),
          'PASC neg:{} ({:.3%})'.format((1 - pasc_flag).sum(), (1 - pasc_flag).mean()))
    print('PASC t2e quantile:', np.quantile(pasc_t2e, [0.5, 0.25, 0.75]))

    severe_pasc_flag = df['pasc-severe-flag']
    severe_pasc_t2e = df.loc[df['pasc-severe-flag'] == 1, 'pasc-severe-min-t2e']
    print('Severe PASC pos:{} ({:.3%})'.format(severe_pasc_flag.sum(), severe_pasc_flag.sum() / pasc_flag.sum()))
    print('Severe PASC t2e quantile:', np.quantile(severe_pasc_t2e, [0.5, 0.25, 0.75]))

    moderate_pasc_flag = df['pasc-moderateonly-flag']
    moderate_pasc_t2e = df.loc[df['pasc-moderateonly-flag'] == 1, 'pasc-moderate-min-t2e']
    print('Moderate PASC pos:{} ({:.3%})'.format(moderate_pasc_flag.sum(), moderate_pasc_flag.sum() / pasc_flag.sum()))
    print('Moderate PASC t2e quantile:', np.quantile(moderate_pasc_t2e, [0.5, 0.25, 0.75]))

    df_pasc_distribution = pd.DataFrame(
        {'any_pasc_sum': df.loc[pasc_flag == 1, specific_pasc_col].sum(),
         'any_pasc-mean': df.loc[pasc_flag == 1, specific_pasc_col].mean(),
         'any_severe_pasc_sum': df.loc[severe_pasc_flag == 1, specific_pasc_col].sum(),
         'any_severe_pasc-mean': df.loc[severe_pasc_flag == 1, specific_pasc_col].mean(),
         'any_moderateonly_pasc_sum': df.loc[moderate_pasc_flag == 1, specific_pasc_col].sum(),
         'any_moderateonly_pasc-mean': df.loc[moderate_pasc_flag == 1, specific_pasc_col].mean()
         })
    df_pasc_distribution['pasc'] = df_pasc_distribution.index
    df_pasc_distribution.index = df_pasc_distribution['pasc'].apply(lambda x: pasc_name[x])
    utils.check_and_mkdir(args.out_dir)
    df_pasc_distribution.to_csv(
        args.out_dir + 'pasc_ditribution-{}-{}-{}.csv'.format(args.dataset, args.population, args.severity))

    # Step 7: final computing risk factors
    # ['anypasc', 'allpasc', 'anyorgan', 'allorgan']
    print(args.goal)
    if args.goal == 'anypasc':
        # screening risk factor of >= k PASC
        screen_any_pasc(args, df, df_pasc_info)
    elif args.goal == 'anyorgan':
        screen_any_organ(args, df, df_pasc_info)
    elif args.goal == 'allpasc':
        screen_all_pasc(args, df, df_pasc_info, selected_pasc_list, dump=True)
    elif args.goal == 'allorgan':
        screen_all_organ(args, df, df_pasc_info, selected_organ_list, dump=True)
    elif args.goal == 'anypascsevere':
        screen_any_pasc_severity(args, df, df_pasc_info, severe=True)
    elif args.goal == 'anypascmoderate':
        screen_any_pasc_severity(args, df, df_pasc_info, severe=False)

    # df, df_pasc_info = build_data_from_all_positive(args)
    # df_pasc_person_counts, df_person_pasc_counts = distribution_statistics(args, df, df_pasc_info)

    # df, df_pasc_info, model_dict = screen_any_organ(args)

    # # pasc_name = 'Neurocognitive disorders'  # 'Diabetes mellitus with complication' # 'Anemia' #
    # # model = risk_factor_of_pasc(args, pasc_name, dump=False)
    #
    # causal_res = pd.read_excel('output/causal_effects_specific_withMedication_v3.xlsx',
    #                            sheet_name='diagnosis')
    # filtered_data = causal_res.loc[(causal_res['hr-w'] > 1) & (causal_res['hr-w-p'] < 0.01), :]
    # filtered_data = filtered_data.reset_index(drop=True)
    # pasc_list = list(filtered_data['pasc'])
    #
    # idx = 0
    # for selected_PASC in pasc_list:  # ['Neurocognitive disorders']:  # , 'Diabetes mellitus with complication']:  # tqdm(pasc_list):
    #     pasc_name = selected_PASC.replace('/', '_')
    #     model = risk_factor_of_pasc(args, pasc_name)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
