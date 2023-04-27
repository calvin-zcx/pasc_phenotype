import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
from evaluation import *
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from PSModels import ml
from misc import utils
import itertools
import functools
from tqdm import tqdm
import datetime
import seaborn as sns
from sklearn.preprocessing import SplineTransformer
from collections import defaultdict

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    # parser.add_argument('--dataset', choices=['oneflorida', 'V15_COVID19'], default='V15_COVID19',
    #                     help='data bases')
    parser.add_argument('--site', default='all',  # choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL', 'all'],
                        help='one particular site or all')
    parser.add_argument('--severity', choices=['all',
                                               'outpatient', 'inpatient', 'icu', 'inpatienticu',
                                               'female', 'male',
                                               'white', 'black',
                                               'less65', '65to75', '75above', '20to40', '40to55', '55to65', 'above65',
                                               'Anemia', 'Arrythmia', 'CKD', 'CPD-COPD', 'CAD',
                                               'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids',
                                               'healthy',
                                               '03-20-06-20', '07-20-10-20', '11-20-02-21',
                                               '03-21-06-21', '07-21-11-21',
                                               '1stwave', 'delta', 'alpha', 'deltaAndBefore', 'omicron',
                                               'deltaAndBeforeoutpatient', 'deltaAndBeforeinpatienticu',
                                               'omicronoutpatient', 'omicroninpatienticu'],
                        default='all')

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--negative_ratio', type=float, default=3)  # 5
    parser.add_argument('--downsample_ratio', type=float, default=1.0)  # 5

    parser.add_argument('--selectpasc', action='store_true')

    args = parser.parse_args()

    # More args

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.save_model_filename)
    return args


def _evaluation_helper(X, T, PS_logits, loss):
    y_pred_prob = logits_to_probability(PS_logits, normalized=False)
    auc = roc_auc_score(T, y_pred_prob)
    max_smd, smd, max_smd_weighted, smd_w, before, after = cal_deviation(X, T, PS_logits, normalized=False,
                                                                         verbose=False)
    n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
    n_unbalanced_feature_weighted = len(np.where(smd_w > SMD_THRESHOLD)[0])
    result = (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
    return result


def _loss_helper(v_loss, v_weights):
    return np.dot(v_loss, v_weights) / np.sum(v_weights)


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


def stringlist_2_list(s):
    r = s.strip('][').replace(' ', '').split(';')
    # r = list(map(float, r))
    r = [float(x) for x in r if x != '']
    return r


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
    elif severity == 'deltaAndBefore':
        print('Considering patients in Delta wave and before, start to Nov.-30-2021')
        df = df.loc[(df['index date'] < datetime.datetime(2021, 12, 1, 0, 0)), :].copy()
    elif severity == 'omicron':
        print('Considering patients in Omicon and after wave, Dec 1, 2021 to Now')
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 12, 1, 0, 0)), :].copy()
    elif severity == 'deltaAndBeforeoutpatient':
        print('Considering patients in Delta wave and before, start to Nov.-30-2021, and outpatient patients')
        df = df.loc[(df['index date'] < datetime.datetime(2021, 12, 1, 0, 0)), :]
        df = df.loc[(df['hospitalized'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif severity == 'deltaAndBeforeinpatienticu':
        print('Considering patients in Delta wave and before, start to Nov.-30-2021, and inpatienticu')
        df = df.loc[(df['index date'] < datetime.datetime(2021, 12, 1, 0, 0)), :]
        df = df.loc[(df['hospitalized'] == 1) | (df['criticalcare'] == 1), :].copy()
    elif severity == 'omicronoutpatient':
        print('Considering patients in Omicon and after wave, Dec 1, 2021 to Now, and outpatient patients')
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 12, 1, 0, 0)), :]
        df = df.loc[(df['hospitalized'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif severity == 'omicroninpatienticu':
        print('Considering patients in Omicon and after wave, Dec 1, 2021 to Now, and inpatienticu')
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 12, 1, 0, 0)), :]
        df = df.loc[(df['hospitalized'] == 1) | (df['criticalcare'] == 1), :].copy()
    else:
        print('Considering ALL cohorts')

    return df


if __name__ == "__main__":
    # python summarize_any_pasc_stratified.py --site all --severity all 2>&1 | tee  log_recover/summarize_any_pasc_stratified_all.txt

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)

    # %% 1. Load  Data
    df_pasc = pd.read_excel(r'any_pasc_recover_summary_cxzang-20230323.xlsx', sheet_name='dx')
    preOmiOutlist = df_pasc.loc[df_pasc['risk_by deltaAndBefore outpatient'] == 1, 'pasc'].to_list()
    preOmiInlist = df_pasc.loc[df_pasc['risk_by deltaAndBefore inpatienticu'] == 1, 'pasc'].to_list()
    omiOutlist = df_pasc.loc[df_pasc['risk_by omicron outpatient'] == 1, 'pasc'].to_list()
    omiInlist = df_pasc.loc[df_pasc['risk_by omicron inpatienticu'] == 1, 'pasc'].to_list()

    pasc_simname = {}
    pasc_organ = {}
    for index, rows in df_pasc.iterrows():
        pasc_simname[rows['pasc']] = (rows['PASC Name Simple'], rows['Organ Domain'])
        pasc_organ[rows['pasc']] = rows['Organ Domain']

    # organ_pasc_preOmiOutlist = {}
    # organ_pasc_preOmiInlist = {}
    # organ_pasc_omiOutlist = {}
    # organ_pasc_omiInlist = {}
    #
    # organ_list = list(df_pasc['Organ Domain'].unique())
    # for org in organ_list:
    #     pasc_under_organ = df_pasc.loc[df_pasc['Organ Domain'] == org, 'pasc']
    #     organ_pasc_preOmiOutlist[org] = [y for y in pasc_under_organ if y in preOmiOutlist]
    #     organ_pasc_preOmiInlist[org] = [y for y in pasc_under_organ if y in preOmiInlist]
    #     organ_pasc_omiOutlist[org] = [y for y in pasc_under_organ if y in omiOutlist]
    #     organ_pasc_omiInlist[org] = [y for y in pasc_under_organ if y in omiInlist]

    print('In cohorts_characterization_build_data...')
    if args.site == 'all':
        sites = ['mcw', 'nebraska', 'utah', 'utsw',
                 'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
                 'ufh', 'usf', 'nch', 'miami',  # 'emory',
                 'pitt', 'psu', 'temple', 'michigan',
                 'ochsner', 'ucsf', 'lsu',
                 'vumc']

        # for debug purpose, comment out when running
        # sites = ['wcm', 'montefiore']  # , 'mshs',

        print('len(sites), sites:', len(sites), sites)
    else:
        sites = [args.site, ]

    df_info_list = []

    for ith, site in tqdm(enumerate(sites)):
        print('Loading: ', site)
        data_file = r'../data/recover/output/{}/matrix_cohorts_covid_4manuNegNoCovidV2age18_boolbase-nout-withAllDays-withPreg_{}.csv'.format(
            site,
            site)
        # Load Covariates Data
        print('Load data covariates file:', data_file)
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
        # because a patid id may occur in multiple sites. patid were site specific
        print('df.shape:', df.shape)
        if df.shape[0] == 0:
            print('0 selected patients in', site, args.severity, 'skip and continue')
            continue

        df['any_pasc_flag'] = 0
        df['any_pasc_type'] = np.nan
        df['any_pasc_t2e'] = np.nan
        df['any_pasc_txt'] = ''

        df['any_2dx_pasc_flag'] = 0
        df['any_2dx_pasc_t2e'] = np.nan
        df['any_2dx30day_pasc_flag'] = 0
        df['any_2dx30day_pasc_t2e'] = np.nan

        for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
            # pasc_list = []
            if (rows['index date'] < datetime.datetime(2021, 12, 1, 0, 0)) and (
                    (rows['hospitalized'] == 0) and (rows['criticalcare'] == 0)):
                # 'deltaAndBeforeoutpatient':
                # print('Considering patients in Delta wave and before, start to Nov.-30-2021, and outpatient patients')
                pasc_list = preOmiOutlist
                # organ_pasc = organ_pasc_preOmiOutlist
                df.loc[index, 'any_pasc_type'] = 'preOmiOut'
            elif (rows['index date'] < datetime.datetime(2021, 12, 1, 0, 0)) and (
                    (rows['hospitalized'] == 1) or (rows['criticalcare'] == 1)):
                # severity == 'deltaAndBeforeinpatienticu':
                # print('Considering patients in Delta wave and before, start to Nov.-30-2021, and inpatienticu')
                pasc_list = preOmiInlist
                # organ_pasc = organ_pasc_preOmiInlist
                df.loc[index, 'any_pasc_type'] = 'preOmiIn'
            elif (rows['index date'] >= datetime.datetime(2021, 12, 1, 0, 0)) and (
                    (rows['hospitalized'] == 0) and (rows['criticalcare'] == 0)):
                # severity == 'omicronoutpatient':
                # print('Considering patients in Omicon and after wave, Dec 1, 2021 to Now, and outpatient patients')
                pasc_list = omiOutlist
                # organ_pasc = organ_pasc_omiOutlist
                df.loc[index, 'any_pasc_type'] = 'omiOut'
            elif (rows['index date'] >= datetime.datetime(2021, 12, 1, 0, 0)) and (
                    (rows['hospitalized'] == 1) or (rows['criticalcare'] == 1)):
                # severity == 'omicroninpatienticu':
                # print('Considering patients in Omicon and after wave, Dec 1, 2021 to Now, and inpatienticu')
                pasc_list = omiInlist
                # organ_pasc = organ_pasc_omiInlist
                df.loc[index, 'any_pasc_type'] = 'omiIn'
            else:
                print('Cannot stratified patients', ith, site, index)
                continue

            # for any 1 pasc
            t2e_list = []
            pasc_1_list = []
            pasc_1_name = []
            pasc_1_text = ''

            # for any 2 pasc with different time intervals
            sameorgan_pasc_2_list = defaultdict(list)

            for p in pasc_list:
                if (rows['dx-out@' + p] > 0) and (rows['dx-base@' + p] == 0):
                    t2e_list.append(rows['dx-t2e@' + p])
                    pasc_1_list.append(p)
                    pasc_1_name.append(pasc_simname[p])
                    pasc_1_text += (pasc_simname[p][0] + ';')

                    _t2eall = stringlist_2_list(rows['dx-t2eall@' + p])
                    for _t2e in _t2eall:
                        sameorgan_pasc_2_list[pasc_organ[p]].append((p, _t2e))

            if len(t2e_list) > 0:
                df.loc[index, 'any_pasc_flag'] = 1
                df.loc[index, 'any_pasc_t2e'] = np.min(t2e_list)
                df.loc[index, 'any_pasc_txt'] = pasc_1_text

            for _k, _v in sameorgan_pasc_2_list.items():
                _pasc_2_list = [_a[0] for _a in _v]
                _pasc_2_t2e_list = [_a[1] for _a in _v]
                all_time = np.array(_pasc_2_t2e_list)
                t2e_min = all_time.min()
                t2e_max = all_time.max()
                if t2e_max - t2e_min >= 1:
                    df.loc[index, 'any_2dx_pasc_flag'] = 1
                    if np.isnan(rows['any_2dx_pasc_t2e']) or (rows['any_2dx_pasc_t2e'] < t2e_min):
                        df.loc[index, 'any_2dx_pasc_t2e'] = t2e_min

                if t2e_max - t2e_min >= 30:
                    df.loc[index, 'any_2dx30day_pasc_flag'] = 1
                    if np.isnan(rows['any_2dx30day_pasc_t2e']) or (rows['any_2dx30day_pasc_t2e'] < t2e_min):
                        df.loc[index, 'any_2dx30day_pasc_t2e'] = t2e_min

        col_names = pd.Series(df.columns)
        select_cols = col_names[:163].to_list() + ['any_pasc_flag', 'any_pasc_type', 'any_pasc_t2e', 'any_pasc_txt',
                                                   'any_2dx_pasc_flag', 'any_2dx_pasc_t2e',
                                                   'any_2dx30day_pasc_flag', 'any_2dx30day_pasc_t2e']

        df_info = df[select_cols]  # 'Unnamed: 0',
        df_info_list.append(df_info)
        print(ith, site, 'df.shape:', df.shape, 'df_info.shape:', df_info.shape)

    df_info = pd.concat(df_info_list, ignore_index=True)

    print('all df_info.shape:', df_info.shape)

    print('Done load data! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    out_file_balance = r'../data/recover/output/results/anyPASC2DX/anyPASC_2DX_stratified_period_severity_nsites-{}.csv'.format(
        len(sites))
    utils.check_and_mkdir(out_file_balance)

    df_info.to_csv(out_file_balance)
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
