import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from misc import utils
import itertools
import functools
from tqdm import tqdm
import datetime
import seaborn as sns
from sklearn.preprocessing import SplineTransformer
import itertools

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input

    parser.add_argument('--site', default='all',  # choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL', 'all'],
                        help='one particular site or all')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--negative_ratio', type=int, default=10)  # 5
    parser.add_argument('--selectpasc', action='store_true')

    args = parser.parse_args()

    # More args

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.save_model_filename)
    return args


def summary_covariate(df, label, weights, smd, smd_weighted, before, after):
    # (covariates_treated_mu, covariates_treated_var, covariates_controlled_mu, covariates_controlled_var), \
    # (covariates_treated_w_mu, covariates_treated_w_var, covariates_controlled_w_mu, covariates_controlled_w_var)

    columns = df.columns
    df_pos = df.loc[label == 1, :]
    df_neg = df.loc[label == 0, :]
    df_pos_mean = df_pos.mean()
    df_neg_mean = df_neg.mean()
    df_pos_sum = df_pos.sum()
    df_neg_sum = df_neg.sum()
    df_summary = pd.DataFrame(index=df.columns, data={
        'Positive Total Patients': df_pos.sum(),
        'Negative Total Patients': df_neg.sum(),
        'Positive Percentage/mean': df_pos.mean(),
        'Positive std': before[1],
        'Negative Percentage/mean': df_neg.mean(),
        'Negative std': before[3],
        'Positive mean after re-weighting': after[0],
        'Negative mean after re-weighting': after[2],
        'Positive std after re-weighting': before[1],
        'Negative std after re-weighting': before[3],
        'SMD before re-weighting': smd,
        'SMD after re-weighting': smd_weighted,
    })
    # df_summary.to_csv('../data/V15_COVID19/output/character/outcome-dx-evaluation_encoding_balancing.csv')
    return df_summary


def select_subpopulation(df, severity):
    if severity == 'inpatient':
        print('Considering inpatient/hospitalized cohorts but not ICU')
        df = df.loc[(df['hospitalized'] == 1) & (df['ventilation'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif severity == 'icu':
        print('Considering ICU (hospitalized ventilation or critical care) cohorts')
        df = df.loc[(((df['hospitalized'] == 1) & (df['ventilation'] == 1)) | (df['criticalcare'] == 1)), :].copy()
    if severity == 'inpatienticu':
        print('Considering inpatient/hospitalized including icu cohorts')
        df = df.loc[(df['hospitalized'] == 1) | (df['criticalcare'] == 1), :].copy()
    elif severity == 'outpatient':
        print('Considering outpatient cohorts')
        df = df.loc[(df['hospitalized'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif severity == 'female':
        print('Considering female cohorts')
        df = df.loc[(df['Female'] == 1), :].copy()
    elif severity == 'male':
        print('Considering male cohorts')
        df = df.loc[(df['Male'] == 1), :].copy()
    elif severity == 'white':
        print('Considering white cohorts')
        df = df.loc[(df['White'] == 1), :].copy()
    elif severity == 'black':
        print('Considering black cohorts')
        df = df.loc[(df['Black or African American'] == 1), :].copy()
    elif severity == '20to40':
        print('Considering 20to40 cohorts')
        df = df.loc[(df['20-<40 years'] == 1), :].copy()
    elif severity == '40to55':
        print('Considering 40to55 cohorts')
        df = df.loc[(df['40-<55 years'] == 1), :].copy()
    elif severity == '55to65':
        print('Considering 55to65 cohorts')
        df = df.loc[(df['55-<65 years'] == 1), :].copy()
    elif severity == 'less65':
        print('Considering less65 cohorts')
        df = df.loc[(df['20-<40 years'] == 1) | (df['40-<55 years'] == 1) | (df['55-<65 years'] == 1), :].copy()
    elif severity == '65to75':
        print('Considering 65to75 cohorts')
        df = df.loc[(df['65-<75 years'] == 1), :].copy()
    elif severity == '75above':
        print('Considering 75above cohorts')
        df = df.loc[(df['75-<85 years'] == 1) | (df['85+ years'] == 1), :].copy()
    elif severity == 'above65':
        print('Considering above65 cohorts')
        df = df.loc[(df['65-<75 years'] == 1) | (df['75-<85 years'] == 1) | (df['85+ years'] == 1), :].copy()
    elif severity == 'Anemia':
        print('Considering Anemia cohorts')
        df = df.loc[(df["DX: Anemia"] == 1), :].copy()
    elif severity == 'Arrythmia':
        print('Considering Arrythmia cohorts')
        df = df.loc[(df["DX: Arrythmia"] == 1), :].copy()
    elif severity == 'CKD':
        print('Considering CKD cohorts')
        df = df.loc[(df["DX: Chronic Kidney Disease"] == 1), :].copy()
    elif severity == 'CPD-COPD':
        print('Considering CPD-COPD cohorts')
        df = df.loc[(df["DX: Chronic Pulmonary Disorders"] == 1) | (df["DX: COPD"] == 1), :].copy()
    elif severity == 'CAD':
        print('Considering CAD cohorts')
        df = df.loc[(df["DX: Coronary Artery Disease"] == 1), :].copy()
    elif severity == 'T2D-Obesity':
        print('Considering T2D-Obesity cohorts')
        df = df.loc[(df["DX: Diabetes Type 2"] == 1) | (df["DX: Severe Obesity  (BMI>=40 kg/m2)"] == 1), :].copy()
    elif severity == 'Hypertension':
        print('Considering Hypertension cohorts')
        df = df.loc[(df["DX: Hypertension"] == 1), :].copy()
    elif severity == 'Mental-substance':
        print('Considering Mental-substance cohorts')
        df = df.loc[(df["DX: Mental Health Disorders"] == 1) | (df['DX: Other Substance Abuse'] == 1), :].copy()
    elif severity == 'Corticosteroids':
        print('Considering Corticosteroids cohorts')
        df = df.loc[(df["MEDICATION: Corticosteroids"] == 1), :].copy()
    elif severity == 'healthy':
        # no comorbidity and no PASC?
        print('Considering baseline totally healthy cohorts')
        selected_cols = [x for x in df.columns if
                         (x.startswith('dx-base@')
                          or x.startswith('DX:')
                          or x.startswith('MEDICATION:'))]
        flag = df[selected_cols].sum(axis=1)
        df = df.loc[(flag == 0), :].copy()
    elif severity == '03-20-06-20':
        print('Considering patients in 03/20-06/20')
        df = df.loc[(df['03/20-06/20'] == 1), :].copy()
    elif severity == '07-20-10-20':
        print('Considering patients in 07/20-10/20')
        df = df.loc[(df['07/20-10/20'] == 1), :].copy()
    elif severity == '11-20-02-21':
        print('Considering patients in 11/20-02/21')
        df = df.loc[(df['11/20-02/21'] == 1), :].copy()
    elif severity == '03-21-06-21':
        print('Considering patients in 03/21-06/21')
        df = df.loc[(df['03/21-06/21'] == 1), :].copy()
    elif severity == '07-21-11-21':
        print('Considering patients in 07/21-11/21')
        df = df.loc[(df['07/21-11/21'] == 1), :].copy()
    elif severity == '1stwave':
        print('Considering patients in 1st wave, Mar-1-2020 to Sep.-30-2020')
        df = df.loc[(df['index date'] >= datetime.datetime(2020, 3, 1, 0, 0)) & (
                df['index date'] < datetime.datetime(2020, 10, 1, 0, 0)), :].copy()
    elif severity == 'delta':
        print('Considering patients in Delta wave, June-1-2021 to Nov.-30-2021')
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 6, 1, 0, 0)) & (
                df['index date'] < datetime.datetime(2021, 12, 1, 0, 0)), :].copy()
    elif severity == 'alpha':
        print('Considering patients in Alpha + others wave, Oct.-1-2020 to May-31-2021')
        df = df.loc[(df['index date'] >= datetime.datetime(2020, 10, 1, 0, 0)) & (
                df['index date'] < datetime.datetime(2021, 6, 1, 0, 0)), :].copy()
    else:
        print('Considering ALL cohorts')

    if severity == 'pospreg-posnonpreg':
        # select index date
        print('Before selecting index date < 2022-6-1, df.shape', df.shape)
        df = df.loc[(df['index date'] < datetime.datetime(2022, 6, 1, 0, 0)), :]  # .copy()
        print('After selecting index date < 2022-6-1, df.shape', df.shape)

        # select age
        print('Before selecting age <= 50, df.shape', df.shape)
        df = df.loc[df['age'] <= 50, :]  # .copy()
        print('After selecting age <= 50, df.shape', df.shape)

        # select female
        print('Before selecting female, df.shape', df.shape)
        df = df.loc[df['Female'] == 1, :]  # .copy()
        print('After selecting female, df.shape', df.shape)

        # covid positive patients only
        print('Before selecting covid+, df.shape', df.shape)
        df = df.loc[df['covid'] == 1, :]  # .copy()
        print('After selecting covid+, df.shape', df.shape)

        # # pregnant patients only
        # print('Before selecting pregnant, df.shape', df.shape)
        # df = df.loc[df['flag_pregnancy'] == 1, :]#.copy()
        # print('After selecting pregnant, df.shape', df.shape)
        #
        # # infection during pregnancy period
        # print('Before selecting infection in gestational period, df.shape', df.shape)
        # df = df.loc[(df['index date'] >= df['flag_pregnancy_start_date']) & (
        #         df['index date'] <= df['flag_delivery_date'] + datetime.timedelta(days=7)), :].copy()
        # print('After selecting infection in gestational period, df.shape', df.shape)

    return df


def add_columns_simple(df):
    # df['<45'] = (df['age'] < 45).astype('int')
    # df['45-<54'] = ((df['age'] >= 45) & (df['age'] <= 54)).astype('int')
    # df['55-<65'] = ((df['age'] >= 55) & (df['age'] <= 64)).astype('int')
    # df['65-74'] = ((df['age'] >= 65) & (df['age'] <= 74)).astype('int')
    # df['75-84'] = ((df['age'] >= 75) & (df['age'] <= 84)).astype('int')
    # df['>=75'] = (df['age'] >= 85).astype('int')

    df['<35'] = (df['age'] < 35).astype('int')
    df['>=35'] = (df['age'] >= 35).astype('int')

    df['RE:White Non-Hispanic'] = 0
    df['RE:Black or African American Non-Hispanic'] = 0
    df['RE:Hispanic or Latino Any Race'] = 0
    # df['RE:Asian Non-Hispanic'] = 0
    df['RE:Other Non-Hispanic'] = 0
    df['RE:Unknown'] = 0

    for c in ['nihrace:American Indian or Alaska Native',
              'nihrace:Asian',
              'nihrace:Native Hawaiian or Other Pacific Islander',
              'nihrace:Black or African American',
              'nihrace:White',
              'nihrace:Multiple Race',
              'nihrace:Missing']:
        df[c] = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 'index date', 'flag_delivery_date', 'flag_pregnancy_start_date', 'flag_pregnancy_end_date'
        # index_date = row['index date']
        # age = row['age']
        race = row['race']
        if (row['White'] == 1) and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:White Non-Hispanic'] = 1
        elif (row['Black or African American'] == 1) and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:Black or African American Non-Hispanic'] = 1
        elif (row['Hispanic: Yes'] == 1):
            df.loc[index, 'RE:Hispanic or Latino Any Race'] = 1
        elif ((row['Other'] == 1) or (row['Asian'] == 1)) and (
                (row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:Other Non-Hispanic'] = 1
        else:
            df.loc[index, 'RE:Unknown'] = 1

        if race == '01':
            df.loc[index, 'nihrace:American Indian or Alaska Native'] = 1  # 'American Indian or Alaska Native',
        elif race == '02':
            df.loc[index, 'nihrace:Asian'] = 1  # 'Asian',
        elif race == '04':
            df.loc[
                index, 'nihrace:Native Hawaiian or Other Pacific Islander'] = 1  # 'Native Hawaiian or Other Pacific Islander,',
        elif race == '03':
            df.loc[index, 'nihrace:Black or African American'] = 1  # 'Black or African American',
        elif race == '05':
            df.loc[index, 'nihrace:White'] = 1  # 'White',
        elif race == '06':
            df.loc[index, 'nihrace:Multiple Race'] = 1  # 'Multiple Race',
        else:
            df.loc[index, 'nihrace:Missing'] = 1  # 'Missing'

    pregtreat_names = ['SSRI', 'SNRI', 'GLP-1', 'SGLT2', 'Metformin', 'Insulin',
                       'paxlovid', 'remdesivir', 'covidvaccine']
    # pregtreat_flag = np.zeros((n, 9), dtype='int16')
    # pregtreat_t2e = np.zeros((n, 9), dtype='int16')  # date of earliest prescriptions
    # pregtreat_t2eall = []
    # pregtreat_column_names = (
    #         ['pregtreat-flag@' + x for x in pregtreat_names] +
    #         ['pregtreat-t2e@' + x for x in pregtreat_names] +
    #         ['pregtreat-t2eall@' + x for x in pregtreat_names])

    selected_cols = [x for x in df.columns if
                     (x.startswith('pregtreat-flag@')  # or
                      )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')
    return df


def _add_col(df):
    df['Type 1 or 2 Diabetes Diagnosis'] = (
            ((df["DX: Diabetes Type 1"] >= 1).astype('int') + (df["DX: Diabetes Type 2"] >= 1).astype(
                'int')) >= 1).astype('int')

    selected_cols = [x for x in df.columns if (
            x.startswith('DX:') or
            x.startswith('MEDICATION:') or
            x.startswith('CCI:') or
            x.startswith('obc:')
    )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')
    df.loc[:, r"DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"] = \
        (df.loc[:, r'DX: Hypertension'] & (
                df.loc[:, r'DX: Diabetes Type 1'] | df.loc[:, r'DX: Diabetes Type 2'])).astype('int')

    # baseline part have been binarized already
    selected_cols = [x for x in df.columns if
                     (x.startswith('dx-out@') or
                      x.startswith('dxadd-out@') or
                      x.startswith('dxbrainfog-out@') or
                      x.startswith('covidmed-out@') or
                      x.startswith('smm-out@') or
                      x.startswith('dxdxCFR-out@')
                      )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    df.loc[df['death t2e'] < 0, 'death t2e'] = 9999
    df.loc[df['death t2e'] < 0, 'death'] = 0

    df['cci_quan:0'] = 0
    df['cci_quan:1-2'] = 0
    df['cci_quan:3-4'] = 0
    df['cci_quan:5-10'] = 0
    df['cci_quan:11+'] = 0

    df['age18-24'] = 0
    df['age15-34'] = 0
    df['age35-49'] = 0
    df['age50-64'] = 0
    df['age65+'] = 0

    df['RE:Asian Non-Hispanic'] = 0
    df['RE:Black or African American Non-Hispanic'] = 0
    df['RE:Hispanic or Latino Any Race'] = 0
    df['RE:White Non-Hispanic'] = 0
    df['RE:Other Non-Hispanic'] = 0
    df['RE:Unknown'] = 0

    df['No. of Visits:0'] = 0
    df['No. of Visits:1-3'] = 0
    df['No. of Visits:4-9'] = 0
    df['No. of Visits:10-19'] = 0
    df['No. of Visits:>=20'] = 0

    df['No. of hospitalizations:0'] = 0
    df['No. of hospitalizations:1'] = 0
    df['No. of hospitalizations:>=1'] = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 'index date', 'flag_delivery_date', 'flag_pregnancy_start_date', 'flag_pregnancy_end_date'
        index_date = row['index date']
        age = row['age']
        if pd.notna(age):
            if age < 25:
                df.loc[index, 'age18-24'] = 1
            elif age < 35:
                df.loc[index, 'age15-34'] = 1
            elif age < 50:
                df.loc[index, 'age35-49'] = 1
            elif age < 65:
                df.loc[index, 'age50-64'] = 1
            elif age >= 65:
                df.loc[index, 'age65+'] = 1

        if row['score_cci_quan'] <= 0:
            df.loc[index, 'cci_quan:0'] = 1
        elif row['score_cci_quan'] <= 2:
            df.loc[index, 'cci_quan:1-2'] = 1
        elif row['score_cci_quan'] <= 4:
            df.loc[index, 'cci_quan:3-4'] = 1
        elif row['score_cci_quan'] <= 10:
            df.loc[index, 'cci_quan:5-10'] = 1
        elif row['score_cci_quan'] >= 11:
            df.loc[index, 'cci_quan:11+'] = 1

        if row['Asian'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:Asian Non-Hispanic'] = 1
        elif row['Black or African American'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:Black or African American Non-Hispanic'] = 1
        elif row['Hispanic: Yes']:
            df.loc[index, 'RE:Hispanic or Latino Any Race'] = 1
        elif row['White'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:White Non-Hispanic'] = 1
        elif row['Other'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:Other Non-Hispanic'] = 1
        else:
            df.loc[index, 'RE:Unknown'] = 1

        visits = row['inpatient no.'] + row['outpatient no.'] + row['emergency visits no.'] + row['other visits no.']
        if visits == 0:
            df.loc[index, 'No. of Visits:0'] = 1
        elif visits <= 3:
            df.loc[index, 'No. of Visits:1-3'] = 1
        elif visits <= 9:
            df.loc[index, 'No. of Visits:4-9'] = 1
        elif visits <= 19:
            df.loc[index, 'No. of Visits:10-19'] = 1
        else:
            df.loc[index, 'No. of Visits:>=20'] = 1

        if row['inpatient no.'] == 0:
            df.loc[index, 'No. of hospitalizations:0'] = 1
        elif row['inpatient no.'] == 1:
            df.loc[index, 'No. of hospitalizations:1'] = 1
        else:
            df.loc[index, 'No. of hospitalizations:>=1'] = 1

        # any PASC

        # adi use mine later, not transform here, add missing

        # monthly changes, add later. Already there

    return df


# def sum_all_cols():
#     start_time = time.time()
#     args = parse_args()
#
#     np.random.seed(args.random_seed)
#     random.seed(args.random_seed)
#
#     print('args: ', args)
#     print('random_seed: ', args.random_seed)
#
#     # %% 1. Load  Data
#     print('In cohorts_characterization_build_data...')
#     if args.site == 'all':
#         sites = ['wcm', 'nyu', 'montefiore', 'mshs', 'columbia']
#         print('len(sites), sites:', len(sites), sites)
#     else:
#         sites = [args.site, ]
#
#     df_list = []
#     for ith, site in tqdm(enumerate(sites)):
#         print('Loading: ', ith, site)
#         data_file = r'../data/recover/output_hf/{}/matrix_cohorts_{}_nout-withAllDays_{}-few.csv'.format(
#             site, 'hf', site)
#
#         print('Load data covariates file:', data_file)
#         df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
#                          parse_dates=['index date', 'dob'])
#         print('df.shape:', df.shape)
#         df_list.append(df)
#
#     # combine all sites and select subcohorts
#     df = pd.concat(df_list, ignore_index=True)
#
#     # add_columns(df)
#
#     # Only binarize baseline flag --> stringent manner. The outcome keeps count
#     selected_cols = [x for x in df.columns if
#                      (x.startswith('ADDX:') or
#                       x.startswith('ADMED:') or
#                       x.startswith('HFMED:')
#                       )]
#     df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')
#
#     # selected_cols.extend(['death', 'age', 'Female', 'Male', 'Other/Missing'])
#     # selected_cols.extend(['Asian', 'Black or African American', 'White', 'Other', 'Missing'])
#     dfdes = df.describe()
#     dfdes.transpose().to_csv('dfdes_v2.csv')
#
#     # b = df['index date'].groupby([df['index date'].dt.year, df['index date'].dt.month]).agg('count')
#     # b = df['index date'].groupby([df['index date'].dt.year, ]).agg('count')
#     # b.to_csv('time.csv')
#
#     print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

def _n_str(n):
    return '{:,}'.format(n)


def _quantile_str(x):
    v = x.quantile([0.25, 0.5, 0.75]).to_list()
    return '{:.0f} ({:.0f}—{:.0f})'.format(v[1], v[0], v[2])


def _percentage_str(x):
    n = x.sum()
    per = x.mean()
    return '{:,} ({:.1f})'.format(n, per * 100)


def _smd(x1, x2):
    m1 = x1.mean()
    m2 = x2.mean()
    v1 = x1.var()
    v2 = x2.var()

    VAR = np.sqrt((v1 + v2) / 2)
    smd = np.divide(
        m1 - m2,
        VAR, out=np.zeros_like(m1), where=VAR != 0)
    return smd


def detailed_race_eth_table(df):
    # race_column_names = ['American Indian or Alaska Native', 'Asian',
    #                      'Native Hawaiian or Other Pacific Islander,', 'Black or African American',
    #                      'White', 'Multiple Race',
    #                      'Missing']
    race_column_names = ['nihrace:American Indian or Alaska Native',
                         'nihrace:Asian',
                         'nihrace:Native Hawaiian or Other Pacific Islander',
                         'nihrace:Black or African American',
                         'nihrace:White',
                         'nihrace:Multiple Race',
                         'nihrace:Missing']
    hispanic_column_names = ['Hispanic: No', 'Hispanic: Yes', 'Hispanic: Other/Missing']
    gender_column_names = ['Female', 'Male', 'Other/Missing']

    results = []
    for rc in race_column_names:
        df_sub = df.loc[df[rc] == 1, :]
        print(rc, len(df_sub))
        rrow = []
        for eth in hispanic_column_names:
            df_sub = df.loc[(df[rc] == 1) & (df[eth] == 1), :]
            tri = df_sub[gender_column_names].sum(axis=0)
            rrow.extend(list(tri))
        results.append(rrow)

    results_df = pd.DataFrame(results, index=race_column_names,
                              columns=list(itertools.product(hispanic_column_names, gender_column_names)))
    return results_df


def detailed_race_eth_table_pcori(df):
    # race_column_names = ['American Indian or Alaska Native', 'Asian',
    #                      'Native Hawaiian or Other Pacific Islander,', 'Black or African American',
    #                      'White', 'Multiple Race',
    #                      'Missing']
    race_column_names = ['pcori:American Indian/Alaska Native',
                         'pcori:Asian',
                         'pcori:Black/African American',
                         'pcori:Hawaiian/Pacific Islander',
                         'pcori:White',
                         'pcori:Multirace',
                         'pcori:other',
                         'Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing'
                         ]
    # hispanic_column_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    gender_column_names = ['Male', 'Female', 'Other/Missing']

    results = []
    for rc in race_column_names:
        df_sub = df.loc[df[rc] == 1, :]
        n_sub = len(df_sub)
        print(rc, len(df_sub))
        rrow = []
        for eth in gender_column_names:
            df_sub = df.loc[(df[rc] == 1) & (df[eth] == 1), :]
            # tri = df_sub[gender_column_names].sum(axis=0)
            # rrow.extend(list(tri))
            rrow.append(len(df_sub))
        rrow.append(n_sub)
        results.append(rrow)

    results_df = pd.DataFrame(results, index=race_column_names,
                              columns=gender_column_names + ['total',])
    return results_df


def add_col(df):
    selected_cols = [x for x in df.columns if (
            x.startswith('DX:') or
            x.startswith('MEDICATION:') or
            x.startswith('CCI:') or
            x.startswith('obc:')
    )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    # baseline part have been binarized already
    selected_cols = [x for x in df.columns if
                     (x.startswith('dx-out@') or
                      x.startswith('dxadd-out@') or
                      x.startswith('dxbrainfog-out@') or
                      x.startswith('covidmed-out@') or
                      x.startswith('smm-out@') or
                      x.startswith('dxdxCFR-out@') or
                      x.startswith('mental-base@') or
                      x.startswith('dxbrainfog-base@') or
                      x.startswith('cnsldn-flag@') or
                      x.startswith('treat-flag@')

                      )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    # pasc_flag = (df['dxbrainfog-out@' + pasc].copy() >= 1).astype('int')
    # pasc_t2e = df['dxbrainfog-t2e@' + pasc].astype('float')
    # pasc_baseline = df['dxbrainfog-base@' + pasc]

    brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
    brainfog_list = list(brainfog_encoding.keys())
    # ['Neurodegenerative', 'Memory-Attention', 'Headache', 'Sleep Disorder', 'Psych', 'Dysautonomia-Orthostatic', 'Stroke']


    df['brain_fog-cnt'] = df[['dxbrainfog-out@' + x for x in brainfog_list]].sum(axis=1)
    df['brain_fog-flag'] = (df['brain_fog-cnt'] > 0).astype('int')

    df.loc[df['death t2e'] < 0, 'death'] = np.nan
    df.loc[df['death t2e'] < 0, 'death t2e'] = 9999
    df.loc[df['death t2e'] == 9999, 'death t2e'] = np.nan

    df['death in acute'] = (df['death t2e'] <= 30).astype('int')
    df['death post acute'] = (df['death t2e'] > 30).astype('int')

    df['RE:White Non-Hispanic'] = 0
    df['RE:Black or African American Non-Hispanic'] = 0
    df['RE:Hispanic or Latino Any Race'] = 0
    # df['RE:Asian Non-Hispanic'] = 0
    df['RE:Other Non-Hispanic'] = 0
    df['RE:Unknown'] = 0


    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 'index date', 'flag_delivery_date', 'flag_pregnancy_start_date', 'flag_pregnancy_end_date'
        # index_date = row['index date']
        # age = row['age']
        if (row['White'] == 1) and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:White Non-Hispanic'] = 1
        elif (row['Black or African American'] == 1) and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:Black or African American Non-Hispanic'] = 1
        elif (row['Hispanic: Yes'] == 1):
            df.loc[index, 'RE:Hispanic or Latino Any Race'] = 1
        elif ((row['Other'] == 1) or (row['Asian'] == 1)) and (
                (row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:Other Non-Hispanic'] = 1
        else:
            df.loc[index, 'RE:Unknown'] = 1

    ssri_names = ['fluvoxamine', 'fluoxetine', 'escitalopram', 'citalopram', 'sertraline', 'paroxetine',
                  'vilazodone']
    snri_names = ['desvenlafaxine', 'duloxetine', 'levomilnacipran', 'milnacipran', 'venlafaxine']
    other_names = ['bupropion', ]  # ['wellbutrin']

    cnsldn_names = [
          'adderall_combo', 'lisdexamfetamine', 'methylphenidate', 'guanfacine']

    # pregtreat_names = ['SSRI', 'SNRI', 'GLP-1', 'SGLT2', 'Metformin', 'Insulin',
    #                    'paxlovid', 'remdesivir', 'covidvaccine']

    df['SSRI-cnt'] = df[['treat-flag@' + x for x in ssri_names]].sum(axis=1)
    df['SSRI'] = (df['SSRI-cnt'] > 0).astype('int')

    df['SNRI-cnt'] = df[['treat-flag@' + x for x in snri_names]].sum(axis=1)
    df['SNRI'] = (df['SNRI-cnt'] > 0).astype('int')

    df['bupropion'] = (df['treat-flag@bupropion'] > 0).astype('int')

    df['ADHDdrug-cnt'] = df[['cnsldn-flag@' + x for x in cnsldn_names]].sum(axis=1)
    df['ADHDdrug'] = (df['ADHDdrug-cnt'] > 0).astype('int')

    df['paxlovid'] = (df['treat-flag@paxlovid'] > 0).astype('int')
    df['remdesivir'] = (df['treat-flag@remdesivir'] > 0).astype('int')


    # pregtreat_flag = np.zeros((n, 9), dtype='int16')
    # pregtreat_t2e = np.zeros((n, 9), dtype='int16')  # date of earliest prescriptions
    # pregtreat_t2eall = []
    # pregtreat_column_names = (
    #         ['pregtreat-flag@' + x for x in pregtreat_names] +
    #         ['pregtreat-t2e@' + x for x in pregtreat_names] +
    #         ['pregtreat-t2eall@' + x for x in pregtreat_names])

    # selected_cols = [x for x in df.columns if
    #                  (x.startswith('pregtreat-flag@')  # or
    #                   )]
    # df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    return df


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
    brainfog_list = list(brainfog_encoding.keys())

    # %% 1. Load  Data
    print('In cohorts_characterization_build_data...')
    site = 'oneflorida'
    site = 'insight'

    data_file = 'FedCauseTable1-{}.csv'.format(site)
    print('Load data covariates file:', data_file)
    df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'dob'])
    print('df.shape:', df.shape)

    ## df = add_columns_simple(df)

    # results_df = detailed_race_eth_table_pcori(df)
    # results_df.to_csv('pcori_race_the_sex-{}.csv'.format(site))

    df = add_col(df)
    out_file = r'Table-pcori_race_the_sex-{}.xlsx'.format(site)
    output_columns = ['All', ]
    row_names = []
    records = []

    # N
    row_names.append('N')
    records.append([
        _n_str(len(df)),
        # _n_str(len(df_pos)),
        # _n_str(len(df_neg)),
        # np.nan
    ])

    # Sex
    row_names.append('Sex — no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    # sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]), ]
         for c in sex_col])

    # age
    row_names.append('Median age (IQR) — yr')
    records.append([
        _quantile_str(df['age']),
    ])

    row_names.append('Age group — no. (%)')
    records.append([])

    # age_col = ['<35', '>=35']
    # row_names.extend(age_col)
    # records.extend(
    #     [[_percentage_str(df[c]), ]
    #      for c in age_col])

    # Race
    row_names.append('Race — no. (%)')
    records.append([])
    # col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    col_names = ['RE:White Non-Hispanic', 'RE:Black or African American Non-Hispanic', 'RE:Hispanic or Latino Any Race',
                 'RE:Other Non-Hispanic', 'RE:Unknown']
    _col_names = ['table1-NHW', 'table1-NHB', 'table1-Hispanics', 'table1-Other', 'table1-Unknown']
    row_names.extend(_col_names)
    records.extend(
        [[_percentage_str(df[c]), ]
         for c in col_names])

    # # Ethnic group
    # row_names.append('Ethnic group — no. (%)')
    # records.append([])
    # col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # pregtreat_names = ['SSRI', 'SNRI', 'GLP-1', 'SGLT2', 'Metformin', 'Insulin',
    #                    'paxlovid', 'remdesivir', 'covidvaccine']
    # col_names = ['pregtreat-flag@' + x for x in pregtreat_names]

    drug_names = ['SSRI', 'SNRI', 'bupropion', 'ADHDdrug',
                       'paxlovid', 'remdesivir', ]
    col_names = [x for x in drug_names]

    row_names.extend(drug_names)
    records.extend(
        [[_percentage_str(df[c]), ]
         for c in col_names])

    df_out = pd.DataFrame(records, columns=output_columns, index=row_names)
    # df_out['SMD'] = df_out['SMD'].astype(float)
    df_out.to_excel(out_file)
    print('Dump done ', df_out)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
