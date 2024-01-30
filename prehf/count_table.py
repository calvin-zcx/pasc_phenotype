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


def add_columns(df):
    df['<45'] = (df['age'] < 45).astype('int')
    df['45-<54'] = ((df['age'] >= 45) & (df['age'] <= 54)).astype('int')
    df['55-<65'] = ((df['age'] >= 55) & (df['age'] <= 64)).astype('int')
    df['65-74'] = ((df['age'] >= 65) & (df['age'] <= 74)).astype('int')
    df['75-84'] = ((df['age'] >= 75) & (df['age'] <= 84)).astype('int')
    df['>=75'] = (df['age'] >= 85).astype('int')


def sum_all_cols():
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    # %% 1. Load  Data
    print('In cohorts_characterization_build_data...')
    if args.site == 'all':
        sites = ['wcm', 'nyu', 'montefiore', 'mshs', 'columbia']
        print('len(sites), sites:', len(sites), sites)
    else:
        sites = [args.site, ]

    df_list = []
    for ith, site in tqdm(enumerate(sites)):
        print('Loading: ', ith, site)
        data_file = r'../data/recover/output_hf/{}/matrix_cohorts_{}_nout-withAllDays_{}-few.csv'.format(
            site, 'hf', site)

        print('Load data covariates file:', data_file)
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
                         parse_dates=['index date', 'dob'])
        print('df.shape:', df.shape)
        df_list.append(df)

    # combine all sites and select subcohorts
    df = pd.concat(df_list, ignore_index=True)

    add_columns(df)

    # Only binarize baseline flag --> stringent manner. The outcome keeps count
    selected_cols = [x for x in df.columns if
                     (x.startswith('ADDX:') or
                      x.startswith('ADMED:') or
                      x.startswith('HFMED:')
                      )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    # selected_cols.extend(['death', 'age', 'Female', 'Male', 'Other/Missing'])
    # selected_cols.extend(['Asian', 'Black or African American', 'White', 'Other', 'Missing'])
    dfdes = df.describe()
    dfdes.transpose().to_csv('dfdes_v2.csv')

    # b = df['index date'].groupby([df['index date'].dt.year, df['index date'].dt.month]).agg('count')
    # b = df['index date'].groupby([df['index date'].dt.year, ]).agg('count')
    # b.to_csv('time.csv')

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    # %% 1. Load  Data
    print('In cohorts_characterization_build_data...')
    if args.site == 'all':
        sites = ['wcm', 'nyu', 'montefiore', 'mshs', 'columbia']
        print('len(sites), sites:', len(sites), sites)
    else:
        sites = [args.site, ]

    df_list = []
    for ith, site in tqdm(enumerate(sites)):
        print('Loading: ', ith, site)
        data_file = r'../data/recover/output_hf/{}/matrix_cohorts_{}_nout-withAllDays_{}-basic-demo.csv'.format(
            site, 'hf', site)

        print('Load data covariates file:', data_file)
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
                         parse_dates=['index date', 'dob'])
        print('df.shape:', df.shape)
        df_list.append(df)

    # combine all sites and select subcohorts
    df = pd.concat(df_list, ignore_index=True)
    print('len(df)', len(df))

    race_column_names = ['American Indian or Alaska Native', 'Asian',
                         'Native Hawaiian or Other Pacific Islander,', 'Black or African American',
                         'White', 'Multiple Race',
                         'Missing']
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

    results_df.to_csv('race_the_sex.csv')

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

