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
                                               '1stwave', 'delta', 'alpha', 'preg-pos-neg',
                                               'pospreg-posnonpreg', 'anyfollowupdx',
                                               'PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease',
                                               'PaxRisk:Chronic liver disease',
                                               'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
                                               'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes',
                                               'PaxRisk:Disabilities',
                                               'PaxRisk:Heart conditions', 'PaxRisk:Hypertension',
                                               'PaxRisk:HIV infection',
                                               'PaxRisk:Immunocompromised condition or weakened immune system',
                                               'PaxRisk:immune',
                                               'PaxRisk:Mental health conditions',
                                               'PaxRisk:Overweight and obesity', 'PaxRisk:Pregnancy',
                                               'PaxRisk:Sickle cell disease or thalassemia',
                                               'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
                                               'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis',
                                               'VA', '2022-04', '2022-03', 'pax1stwave', 'pax2ndwave',
                                               'RUCA1@1', 'RUCA1@2', 'RUCA1@3', 'RUCA1@4', 'RUCA1@5',
                                               'RUCA1@6', 'RUCA1@7', 'RUCA1@8', 'RUCA1@9', 'RUCA1@10',
                                               'RUCA1@99', 'ZIPMissing',
                                               ],
                        default='all')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--negative_ratio', type=int, default=5)  # 5
    parser.add_argument('--selectpasc', action='store_true')
    parser.add_argument('--build_data', action='store_true')

    parser.add_argument('--exptype',
                        choices=['ssri-base-180-0',
                                 'ssri-base-120-0',
                                 'ssri-acute0-15',
                                 'ssri-acute0-7',
                                 'ssri-base-180-0-clean',
                                 'ssri-acute0-15-clean',

                                 'snri-base-180-0',
                                 'snri-base-120-0',
                                 'snri-acute0-15',
                                 'snri-acute0-7',
                                 'snri-base-180-0-clean',
                                 'snri-acute0-15-clean',

                                 'ssriVSsnri-base-180-0',
                                 'ssriVSsnri-base-120-0',
                                 'ssriVSsnri-acute0-15',
                                 'ssriVSsnri-acute0-7',
                                 'ssriVSsnri-base-180-0-clean',
                                 'ssriVSsnri-acute0-15-clean',

                                 'bupropion-base-180-0',
                                 'bupropion-acute0-15'
                                 'ssriVSbupropion-base-180-0',
                                 'ssriVSbupropion-acute0-15',

                                 'bupropion-base-180-0-clean',
                                 'bupropion-acute0-15-clean'
                                 'ssriVSbupropion-base-180-0-clean',
                                 'ssriVSbupropion-acute0-15-clean',

                                 'ssri-base-180-0-cleanv2',

                                 ], default='base180-0')

    # parser.add_argument('--cohorttype',
    #                     choices=['atrisk', 'norisk', 'atrisklabdx', 'norisklabdx'],
    #                     default='atrisk')
    parser.add_argument('--cohorttype',
                        choices=['atrisknopreg', 'norisk', 'pregnant',
                                 'atrisk',
                                 'atrisknopreglabdx', 'norisklabdx', 'pregnantlabdx',
                                 'overall', ],
                        default='overall')
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


def feature_process_pregnancy(df):
    print('feature_process_additional, df.shape', df.shape)
    start_time = time.time()

    df['gestational age at delivery'] = np.nan
    df['gestational age of infection'] = np.nan
    df['preterm birth'] = np.nan

    df['infection at trimester1'] = 0
    df['infection at trimester2'] = 0
    df['infection at trimester3'] = 0

    # ['flag_delivery_type_Spontaneous', 'flag_delivery_type_Cesarean',
    # 'flag_delivery_type_Operative', 'flag_delivery_type_Vaginal', 'flag_delivery_type_other-unsepc',]
    df['flag_delivery_type_other-unsepc'] = (
            (df['flag_delivery_type_Other'] + df['flag_delivery_type_Unspecified']) >= 1).astype('int')
    df['flag_delivery_type_Vaginal-Spontaneous'] = (
            (df['flag_delivery_type_Spontaneous'] + df['flag_delivery_type_Vaginal']) >= 1).astype('int')
    df['flag_delivery_type_Cesarean-Operative'] = (
            (df['flag_delivery_type_Cesarean'] + df['flag_delivery_type_Operative']) >= 1).astype('int')

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 'index date', 'flag_delivery_date', 'flag_pregnancy_start_date', 'flag_pregnancy_end_date'
        index_date = row['index date']
        del_date = row['flag_delivery_date']
        preg_date = row['flag_pregnancy_start_date']
        if pd.notna(del_date) and pd.notna(preg_date):
            gesage = (del_date - preg_date).days / 7
            df.loc[index, 'gestational age at delivery'] = gesage
            df.loc[index, 'preterm birth'] = int(gesage < 37)

        if pd.notna(index_date) and pd.notna(preg_date):
            infectage = (index_date - preg_date).days / 7
            df.loc[index, 'gestational age of infection'] = infectage
            if infectage <= 13:
                df.loc[index, 'infection at trimester1'] = 1
            elif infectage <= 27:
                df.loc[index, 'infection at trimester2'] = 1
            elif infectage > 27:
                df.loc[index, 'infection at trimester3'] = 1

    print('feature_process_additional Done! Time used:',
          time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df


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
    elif severity == 'anyfollowupdx':
        print('Considering patients with anyfollowupdx')
        print('before followupanydx', len(df))
        df = df.loc[(df['followupanydx'] == 1), :].copy()
        print('after followupanydx', len(df))
    elif severity in ['PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
                      'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
                      'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
                      'PaxRisk:Heart conditions', 'PaxRisk:Hypertension', 'PaxRisk:HIV infection',
                      # 'PaxRisk:Immunocompromised condition or weakened immune system',
                      'PaxRisk:Mental health conditions',
                      'PaxRisk:Overweight and obesity', 'PaxRisk:Pregnancy',
                      'PaxRisk:Sickle cell disease or thalassemia',
                      'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
                      'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis', ]:
        print('Considering {} cohorts'.format(severity))
        print('before selection', len(df))
        df = df.loc[(df[severity] == 1), :].copy()
        print('after followupanydx', len(df))
    elif severity == 'PaxRisk:immune':
        print('PaxRisk:immune for PaxRisk:Immunocompromised condition or weakened immune system')
        print('before selection', len(df))
        df = df.loc[(df['PaxRisk:Immunocompromised condition or weakened immune system'] == 1), :].copy()
        print('after followupanydx', len(df))
    elif severity == 'VA':
        print('Considering VA-like cohorts')
        print('initial cohort before selection:', df.shape)
        df = df.loc[(df['age'] >= 60) | (df['PaxRisk-Count'] > 0), :]
        df_male = df.loc[(df['Male'] == 1), :]
        df_female = df.loc[(df['Female'] == 1), :]
        n_male = len(df_male)
        n_female = len(df_female)
        r = 0.8783
        delta = (1 - 1 / r) * n_male + n_female
        print('n_male+n_female', n_male + n_female,
              'n_male', n_male,
              'n_female', n_female,
              'r', r
              )
        print('n_male/(n_male + n_female - delta):', n_male / (n_male + n_female - delta))
        print('sample n_female-delta female', n_female - delta)
        df_female_sub = df_female.sample(n=int(n_female - delta), replace=False, random_state=args.random_seed)
        print('df_male.shape', df_male.shape, 'df_female_sub.shape', df_female_sub.shape, )
        df = pd.concat([df_male, df_female_sub], ignore_index=True)
        df = df.copy()
        print('after build, final VA cohort df', len(df), df.shape)
    elif severity == '2022-04':
        print('Considering patients after 2022-4-1')
        print('before build df', len(df), df.shape)
        df = df.loc[(df['index date'] >= datetime.datetime(2022, 4, 1, 0, 0)), :].copy()
        print('after build, final cohort df', len(df), df.shape)
    elif severity == '2022-03':
        print('Considering patients after 2022-3-1')
        print('before build df', len(df), df.shape)
        df = df.loc[(df['index date'] >= datetime.datetime(2022, 3, 1, 0, 0)), :].copy()
        print('after build, final cohort df', len(df), df.shape)
    elif severity == 'pax1stwave':
        print('Considering patients in pax1stwave, 22-3-1 to 22-10-1')
        df = df.loc[(df['index date'] >= datetime.datetime(2022, 3, 1, 0, 0)) & (
                df['index date'] < datetime.datetime(2022, 10, 1, 0, 0)), :].copy()
    elif severity == 'pax2ndwave':
        print('Considering patients in pax2ndwave, 22-10-1 to 23-2-1 to ')
        df = df.loc[(df['index date'] >= datetime.datetime(2022, 10, 1, 0, 0)) & (
                df['index date'] <= datetime.datetime(2023, 2, 1, 0, 0)), :].copy()
    elif severity in ['RUCA1@1', 'RUCA1@2', 'RUCA1@3', 'RUCA1@4', 'RUCA1@5',
                      'RUCA1@6', 'RUCA1@7', 'RUCA1@8', 'RUCA1@9', 'RUCA1@10',
                      'RUCA1@99', 'ZIPMissing']:
        print('Considering RUCA codes, ', severity)
        print('before selection', len(df))
        df = df.loc[(df[severity] == 1), :].copy()
        print('after selecting RUCA', severity, len(df))
    else:
        print('Considering ALL cohorts')

    return df


def exact_match_on(df_case, df_ctrl, kmatch, cols_to_match, random_seed=0):
    print('len(case)', len(df_case), 'len(ctrl)', len(df_ctrl))
    ctrl_list = []
    n_no_match = 0
    for index, rows in tqdm(df_case.iterrows(), total=df_case.shape[0]):
        if index == 275:
            print(275, 'debug')
        boolidx = df_ctrl[cols_to_match[0]] == rows[cols_to_match[0]]
        for c in cols_to_match[1:]:
            boolidx &= df_ctrl[c] == rows[c]
        sub_df = df_ctrl.loc[boolidx, :]
        if len(sub_df) >= kmatch:
            _add_index = sub_df.sample(n=kmatch, replace=False, random_state=random_seed).index
        else:
            _add_index = []
            n_no_match += 1
            print('No match for', index)
        ctrl_list.extend(_add_index)
        # df_ctrl.drop(_add_index, inplace=True)
        if len(_add_index) > 0:
            df_ctrl = df_ctrl[~df_ctrl.index.isin(_add_index)]
        if len(df_ctrl) == 0:
            break

    print('Done, {}/{} no match'.format(n_no_match, len(df_case)))
    return ctrl_list


def _clean_name_(s, maxlen=50):
    s = s.replace(':', '-').replace('/', '-').replace('@', '-')
    s_trunc = (s[:maxlen] + '..') if len(s) > maxlen else s
    return s_trunc


if __name__ == "__main__":
    # python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisk --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisk-all-V2.txt
    # python screen_paxlovid_iptw_pcornet.py  --cohorttype atrisk --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-atrisk-anyfollowupdx-V2.txt

    # python screen_paxlovid_iptw_pcornet.py  --cohorttype norisk --severity all 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisk-all-V2.txt
    # python screen_paxlovid_iptw_pcornet.py  --cohorttype norisk --severity anyfollowupdx 2>&1 | tee  log_recover/screen_paxlovid_iptw_pcornet-norisk-anyfollowupdx-V2.txt

    # python screen_ssri_iptw_pcornet.py  --exptype ssri-base-180-0  2>&1 | tee  log_ssri/screen_ssri_iptw_pcornet-ssri-base-180-0.txt

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    # add matched cohorts later
    # in_file = 'recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNet-SSRI-v3-addPaxFeats-addGeneralEC-withexposure.csv'
    in_file = 'recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNet-SSRI-v5-withmental-addPaxFeats-addGeneralEC-withexposure.csv'
    df = pd.read_csv(in_file,
                     dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'dob',
                                  'flag_delivery_date',
                                  'flag_pregnancy_start_date',
                                  'flag_pregnancy_end_date'
                                  ])
    print('df.shape:', df.shape)

    # define treated and untreated here

    print('exposre strategy, args.exptype:', args.exptype)
    print('check exposure strategy, n1, negative ratio, n0, if match on mental health, no-user definition')

    # 2024-09-07 replace 'PaxRisk:Mental health conditions' with 'SSRI-Indication-dsmAndExlix-flag'
    # control group, not using snri criteria? --> add -clean group
    if args.exptype == 'ssri-base-180-0':
        df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
                df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        case_label = 'SSRI-180-0'
        ctrl_label = 'Nouser'
    elif args.exptype == 'ssri-base-180-0-clean':
        df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['snri-treat--180-180-flag'] == 0)
                     & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0)
                     & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        case_label = 'SSRI-180-0-clean'
        ctrl_label = 'Nouser-clean'

    elif args.exptype == 'ssri-base-180-0-cleanv2':
        df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['snri-treat--180-0-flag'] == 0)
                     & (df['other-treat--180-0-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-0-flag'] == 0) & (df['snri-treat--180-0-flag'] == 0)
                     & (df['other-treat--180-0-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        case_label = 'SSRI-180-0-clean'
        ctrl_label = 'Nouser-clean'

    # elif args.exptype == 'ssri-base-120-0':
    #     df1 = df.loc[(df['ssri-treat--120-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
    #     df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
    #             df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
    #     case_label = 'SSRI-120-0'
    #     ctrl_label = 'Nouser'
    # elif args.exptype == 'ssri-acute0-7':
    #     df1 = df.loc[(df['ssri-treat-0-7-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
    #     df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
    #             df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
    #     case_label = 'SSRI-0-7'
    #     ctrl_label = 'Nouser'
    elif args.exptype == 'ssri-acute0-15':
        df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
                df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        case_label = 'SSRI-0-15'
        ctrl_label = 'Nouser'

    elif args.exptype == 'ssri-acute0-15-clean':
        df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['snri-treat--180-180-flag'] == 0)
                     & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0)
                     & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]

        case_label = 'SSRI-0-15-clean'
        ctrl_label = 'Nouser-clean'

    elif args.exptype == 'snri-base-180-0':
        df1 = df.loc[(df['snri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
                df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        case_label = 'SNRI-180-0'
        ctrl_label = 'Nouser'

    elif args.exptype == 'snri-base-180-0-clean':
        df1 = df.loc[(df['snri-treat--180-0-flag'] >= 1) & (df['ssri-treat--180-180-flag'] == 0)
                     & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0)
                     & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        case_label = 'SNRI-180-0-clean'
        ctrl_label = 'Nouser-clean'
    # elif args.exptype == 'snri-base-120-0':
    #     df1 = df.loc[(df['snri-treat--120-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
    #     df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
    #             df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
    #     case_label = 'SNRI-120-0'
    #     ctrl_label = 'Nouser'
    # elif args.exptype == 'snri-acute0-7':
    #     df1 = df.loc[(df['snri-treat-0-7-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
    #     df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
    #             df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
    #     case_label = 'SNRI-0-7'
    #     ctrl_label = 'Nouser'
    elif args.exptype == 'snri-acute0-15':
        df1 = df.loc[(df['snri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
                df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        case_label = 'SNRI-0-15'
        ctrl_label = 'Nouser'

    elif args.exptype == 'snri-acute0-15-clean':
        df1 = df.loc[(df['snri-treat-0-15-flag'] >= 1) & (df['ssri-treat--180-180-flag'] == 0)
                     & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0)
                     & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        case_label = 'SNRI-180-0-clean'
        ctrl_label = 'Nouser-clean'

    elif args.exptype == 'ssriVSsnri-base-180-0':
        df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['snri-treat--180-180-flag'] == 0), :]
        df0 = df.loc[(df['snri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['ssri-treat--180-180-flag'] == 0), :]
        case_label = 'SSRI-180-0'
        ctrl_label = 'SNRI-180-0'

    elif args.exptype == 'ssriVSsnri-base-180-0-clean':

        df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['snri-treat--180-180-flag'] == 0) & (df['other-treat--180-180-flag'] == 0), :]
        df0 = df.loc[(df['snri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['ssri-treat--180-180-flag'] == 0) & (df['other-treat--180-180-flag'] == 0), :]
        case_label = 'SSRI-180-0-clean'
        ctrl_label = 'SNRI-180-0-clean'

    # elif args.exptype == 'ssriVSsnri-base-120-0':
    #     df1 = df.loc[(df['ssri-treat--120-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
    #                 df['snri-treat--180-180-flag'] == 0), :]
    #     df0 = df.loc[(df['snri-treat--120-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
    #                 df['ssri-treat--180-180-flag'] == 0), :]
    #     case_label = 'SSRI-120-0'
    #     ctrl_label = 'SNRI-120-0'
    # elif args.exptype == 'ssriVSsnri-acute0-7':
    #     df1 = df.loc[(df['ssri-treat-0-7-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
    #                 df['snri-treat--180-180-flag'] == 0), :]
    #     df0 = df.loc[(df['snri-treat-0-7-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
    #                 df['ssri-treat--180-180-flag'] == 0), :]
    #     case_label = 'SSRI-0-7'
    #     ctrl_label = 'SNRI-0-7'
    elif args.exptype == 'ssriVSsnri-acute0-15':
        df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['snri-treat--180-180-flag'] == 0), :]
        df0 = df.loc[(df['snri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['ssri-treat--180-180-flag'] == 0), :]
        case_label = 'SSRI-0-15'
        ctrl_label = 'SNRI-0-15'

    elif args.exptype == 'ssriVSsnri-acute0-15-clean':
        df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['snri-treat--180-180-flag'] == 0) & (df['other-treat--180-180-flag'] == 0), :]
        df0 = df.loc[(df['snri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['ssri-treat--180-180-flag'] == 0) & (df['other-treat--180-180-flag'] == 0), :]
        case_label = 'SSRI-0-15-clean'
        ctrl_label = 'SNRI-0-15-clean'

    elif args.exptype == 'ssriVSbupropion-base-180-0':
        df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['other-treat--180-180@bupropion'] == 0), :]
        df0 = df.loc[(df['other-treat--180-0@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['ssri-treat--180-180-flag'] == 0), :]
        case_label = 'SSRI-180-0'
        ctrl_label = 'bupropion-180-0'

    elif args.exptype == 'ssriVSbupropion-base-180-0-clean':
        df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['other-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
        df0 = df.loc[(df['other-treat--180-0@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
        case_label = 'SSRI-180-0-clean'
        ctrl_label = 'bupropion-180-0-clean'

    elif args.exptype == 'ssriVSbupropion-acute0-15':
        df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['other-treat--180-180@bupropion'] == 0), :]
        df0 = df.loc[(df['other-treat-0-15@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['ssri-treat--180-180-flag'] == 0), :]
        case_label = 'SSRI-0-15'
        ctrl_label = 'bupropion-0-15'

    elif args.exptype == 'ssriVSbupropion-acute0-15-clean':
        df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['other-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
        df0 = df.loc[(df['other-treat-0-15@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
                df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
        case_label = 'SSRI-0-15-clean'
        ctrl_label = 'bupropion-0-15-clean'

    elif args.exptype == 'bupropion-base-180-0':
        df1 = df.loc[(df['other-treat--180-0@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
                df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (df['other-treat--180-180@bupropion'] == 0), :]
        case_label = 'bupropion-180-0'
        ctrl_label = 'Nouser'

    elif args.exptype == 'bupropion-base-180-0-clean':
        df1 = df.loc[(df['other-treat--180-0@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) &
                     (df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
                df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (df['other-treat--180-180-flag'] == 0), :]
        case_label = 'bupropion-180-0-clean'
        ctrl_label = 'Nouser-clean'


    elif args.exptype == 'bupropion-acute0-15':
        df1 = df.loc[(df['other-treat-0-15@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
                df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (df['other-treat--180-180@bupropion'] == 0), :]
        case_label = 'bupropion-0-15'
        ctrl_label = 'Nouser'

    elif args.exptype == 'bupropion-acute0-15-clean':
        df1 = df.loc[(df['other-treat-0-15@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) &
                     (df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
        df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
                df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (df['other-treat--180-180-flag'] == 0), :]
        case_label = 'bupropion-0-15-clean'
        ctrl_label = 'Nouser-clean'

    n1 = len(df1)
    print('n1', n1, 'n0', len(df0))
    df0 = df0.sample(n=min(len(df0), int(args.negative_ratio * n1)), replace=False, random_state=args.random_seed)
    print('after sample, n0', len(df0), 'with ratio:', args.negative_ratio, args.negative_ratio * n1)
    df1['treated'] = 1
    df1['SSRI'] = 1
    df0['treated'] = 0
    df0['SSRI'] = 0

    df = pd.concat([df1, df0], ignore_index=True)

    print('Before select_subpopulation, len(df)', len(df))
    df = select_subpopulation(df, args.severity)
    print('After select_subpopulation, len(df)', len(df))

    # pre-process data a little bit
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
                      x.startswith('dxCFR-out@') or
                      x.startswith('mental-base@')
                      )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    # data clean for <0 error death records, and add censoring to the death time to event columns

    df.loc[df['death t2e'] < 0, 'death'] = 0
    df.loc[df['death t2e'] < 0, 'death t2e'] = 9999

    # death in [0, 180). 1: evnt, 0: censored, censored at 180. death at 180, not counted, thus use <
    df['death all'] = ((df['death'] == 1) & (df['death t2e'] >= 0) & (df['death t2e'] < 180)).astype('int')
    df['death t2e all'] = df['death t2e'].clip(lower=0, upper=180)
    df.loc[df['death all'] == 0, 'death t2e all'] = df['maxfollowup'].clip(lower=0, upper=180)

    # death in [0, 30). 1: evnt, 0: censored, censored at 30. death at 30, not counted, thus use <
    df['death acute'] = ((df['death'] == 1) & (df['death t2e'] <= 30)).astype('int')
    df['death t2e acute'] = df['death t2e all'].clip(upper=31)

    # death in [30, 180).  1:event, 0: censored. censored at 180 or < 30, say death at 20, flag is 0, time is 20
    df['death postacute'] = ((df['death'] == 1) & (df['death t2e'] >= 31) & (df['death t2e'] < 180)).astype('int')
    df['death t2e postacute'] = df['death t2e all']

    #
    df['hospitalization-acute-flag'] = (df['hospitalization-acute-flag'] >= 1).astype('int')
    df['hospitalization-acute-t2e'] = df['hospitalization-acute-t2e'].clip(upper=31)
    df['hospitalization-postacute-flag'] = (df['hospitalization-postacute-flag'] >= 1).astype('int')

    #
    # pre-process PASC info
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

    # pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    pasc_list_raw = df_pasc_info.loc[df_pasc_info['selected_narrow'] == 1, 'pasc'].to_list()
    _exclude_list = ['Pressure ulcer of skin', 'Fluid and electrolyte disorders']
    pasc_list = [x for x in pasc_list_raw if x not in _exclude_list]

    pasc_add = ['smell and taste', ]
    print('len(pasc_list)', len(pasc_list), 'len(pasc_add)', len(pasc_add))
    print('pasc_list:', pasc_list)
    print('pasc_add', pasc_add)

    for p in pasc_list:
        df[p + '_pasc_flag'] = 0
    for p in pasc_add:
        df[p + '_pasc_flag'] = 0

    for p in CFR_list:
        df[p + '_CFR_flag'] = 0

    df['any_pasc_flag'] = 0
    df['any_pasc_type'] = np.nan
    df['any_pasc_t2e'] = 180  # np.nan
    df['any_pasc_txt'] = ''
    df['any_pasc_baseline'] = 0  # placeholder for screening, no special meaning, null column

    df['any_CFR_flag'] = 0
    # df['any_CFR_type'] = np.nan
    df['any_CFR_t2e'] = 180  # np.nan
    df['any_CFR_txt'] = ''
    df['any_CFR_baseline'] = 0  # placeholder for screening, no special meaning, null column

    for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
        # for any 1 pasc
        t2e_list = []
        pasc_1_list = []
        pasc_1_name = []
        pasc_1_text = ''
        for p in pasc_list:
            if (rows['dx-out@' + p] > 0) and (rows['dx-base@' + p] == 0):
                t2e_list.append(rows['dx-t2e@' + p])
                pasc_1_list.append(p)
                pasc_1_name.append(pasc_simname[p])
                pasc_1_text += (pasc_simname[p][0] + ';')

                df.loc[index, p + '_pasc_flag'] = 1

        for p in pasc_add:
            if (rows['dxadd-out@' + p] > 0) and (rows['dxadd-base@' + p] == 0):
                t2e_list.append(rows['dxadd-t2e@' + p])
                pasc_1_list.append(p)
                pasc_1_name.append(pasc_simname[p])
                pasc_1_text += (pasc_simname[p][0] + ';')

                df.loc[index, p + '_pasc_flag'] = 1

        if len(t2e_list) > 0:
            df.loc[index, 'any_pasc_flag'] = 1
            df.loc[index, 'any_pasc_t2e'] = np.min(t2e_list)
            df.loc[index, 'any_pasc_txt'] = pasc_1_text
        else:
            df.loc[index, 'any_pasc_flag'] = 0
            df.loc[index, 'any_pasc_t2e'] = rows[['dx-t2e@' + p for p in pasc_list]].max()  # censoring time

        # for CFR pasc
        CFR_t2e_list = []
        CFR_1_list = []
        CFR_1_name = []
        CFR_1_text = ''
        for p in CFR_list:
            if (rows['dxCFR-out@' + p] > 0) and (rows['dxCFR-base@' + p] == 0):
                CFR_t2e_list.append(rows['dxCFR-t2e@' + p])
                CFR_1_list.append(p)
                CFR_1_name.append(pasc_simname[p])
                CFR_1_text += (pasc_simname[p][0] + ';')

                df.loc[index, p + '_CFR_flag'] = 1

        if len(CFR_t2e_list) > 0:
            df.loc[index, 'any_CFR_flag'] = 1
            df.loc[index, 'any_CFR_t2e'] = np.min(CFR_t2e_list)
            df.loc[index, 'any_CFR_txt'] = CFR_1_text
        else:
            df.loc[index, 'any_CFR_flag'] = 0
            df.loc[index, 'any_CFR_t2e'] = rows[['dxCFR-t2e@' + p for p in CFR_list]].max()  # censoring time

    # pd.Series(df.columns).to_csv('recover_covid_pos-with-pax-V3-column-name.csv')

    print('Severity cohorts:', args.severity,
          # 'df1.shape:', df1.shape,
          # 'df2.shape:', df2.shape,
          'df.shape:', df.shape,
          )

    col_names = pd.Series(df.columns)
    df_info = df[['patid', 'site', 'index date', 'treated',
                  'hospitalized',
                  'ventilation', 'criticalcare', 'maxfollowup', 'death', 'death t2e',
                  '03/20-06/20', '07/20-10/20', '11/20-02/21',
                  '03/21-06/21', '07/21-10/21', '11/21-02/22',
                  '03/22-06/22', '07/22-10/22', '11/22-02/23',
                  '03/23-06/23', '07/23-10/23', '11/23-02/24',
                  ]]  # 'Unnamed: 0',

    df_label = (df['treated'] >= 1).astype('int')

    # how to deal with death?
    df_outcome_cols = ['death', 'death t2e'] + [x for x in
                                                list(df.columns)
                                                if x.startswith('dx') or
                                                x.startswith('smm') or
                                                x.startswith('any_pasc') or
                                                x.startswith('dxadd') or
                                                x.startswith('dxbrainfog') or
                                                x.startswith('dxCFR')
                                                ]

    df_outcome = df.loc[:, df_outcome_cols]  # .astype('float')

    # if args.cohorttype in ['overall']:
    covs_columns = [
        'Female', 'Male', 'Other/Missing',
        'age@18-24', 'age@25-34', 'age@35-49', 'age@50-64',  # 'age@65+', # # expand 65
        '65-<75 years', '75-<85 years', '85+ years',
        'RE:Asian Non-Hispanic',
        'RE:Black or African American Non-Hispanic',
        'RE:Hispanic or Latino Any Race', 'RE:White Non-Hispanic',
        'RE:Other Non-Hispanic', 'RE:Unknown',
        'ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
        'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100', 'ADIMissing',
        '03/22-06/22', '07/22-10/22', '11/22-02/23',
        # 'quart:01/22-03/22', 'quart:04/22-06/22', 'quart:07/22-09/22', 'quart:10/22-1/23',
        'inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits 3-4',
        'inpatient visits >=5',
        'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits 3-4',
        'outpatient visits >=5',
        'emergency visits 0', 'emergency visits 1-2', 'emergency visits 3-4',
        'emergency visits >=5',
        'BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ',
        'BMI: >=30 obese ', 'BMI: missing',
        'Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing',
        'PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
        'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
        'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
        'PaxRisk:Heart conditions', 'PaxRisk:Hypertension', 'PaxRisk:HIV infection',
        'PaxRisk:Immunocompromised condition or weakened immune system', 'PaxRisk:Mental health conditions',
        'PaxRisk:Overweight and obesity', 'PaxRisk:Pregnancy', 'PaxRisk:Sickle cell disease or thalassemia',
        'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
        'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis',
        'Fully vaccinated - Pre-index', 'Partially vaccinated - Pre-index', 'No evidence - Pre-index',
        "DX: Coagulopathy", "DX: Peripheral vascular disorders ", "DX: Seizure/Epilepsy", "DX: Weight Loss",
        'DX: Obstructive sleep apnea', 'DX: Epstein-Barr and Infectious Mononucleosis (Mono)', 'DX: Herpes Zoster',
        'mental-base@Schizophrenia Spectrum and Other Psychotic Disorders',
        'mental-base@Depressive Disorders',
        'mental-base@Bipolar and Related Disorders',
        'mental-base@Anxiety Disorders',
        'mental-base@Obsessive-Compulsive and Related Disorders',
        'mental-base@Post-traumatic stress disorder',
        'mental-base@Bulimia nervosa',
        'mental-base@Binge eating disorder',
        'mental-base@premature ejaculation',
        'mental-base@Autism spectrum disorder',
        'mental-base@Premenstrual dysphoric disorder',
        'mental-base@SMI',
        'mental-base@non-SMI',

    ]
    if 'bupropion' not in args.exptype:
        covs_columns += ['other-treat--1095-0-flag', ]

    print('cohorttype:', args.cohorttype)
    print('len(covs_columns):', len(covs_columns), covs_columns)

    df_covs = df.loc[:, covs_columns].astype('float')
    print('df.shape:', df.shape, 'df_covs.shape:', df_covs.shape)

    print('all',
          'df.shape', df.shape,
          'df_info.shape:', df_info.shape,
          'df_label.shape:', df_label.shape,
          'df_covs.shape:', df_covs.shape)
    print('Done load data! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # Load index information
    with open(r'../data/mapping/icd_pasc_mapping.pkl', 'rb') as f:
        icd_pasc = pickle.load(f)
        print('Load ICD-10 to PASC mapping done! len(icd_pasc):', len(icd_pasc))
        record_example = next(iter(icd_pasc.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/pasc_index_mapping.pkl', 'rb') as f:
        pasc_encoding = pickle.load(f)
        print('Load PASC to encoding mapping done! len(pasc_encoding):', len(pasc_encoding))
        record_example = next(iter(pasc_encoding.items()))
        print('e.g.:', record_example)

    selected_screen_list = (['any_pasc', 'PASC-General',
                             'death', 'death_acute', 'death_postacute',
                             'any_CFR',
                             'hospitalization_acute', 'hospitalization_postacute'] +
                            CFR_list +
                            pasc_list +
                            addedPASC_list +
                            brainfog_list)

    causal_results = []
    results_columns_name = []
    for i, pasc in tqdm(enumerate(selected_screen_list, start=1), total=len(selected_screen_list)):

        print('\n In screening:', i, pasc)
        # if i < 17:
        #     continue
        if pasc == 'any_pasc':
            pasc_flag = df['any_pasc_flag'].astype('int')
            pasc_t2e = df['any_pasc_t2e'].astype('float')
            pasc_baseline = df['any_pasc_baseline']
        elif pasc == 'death':
            pasc_flag = df['death'].astype('int')
            pasc_t2e = df['death t2e all'].astype('float')
            pasc_baseline = (df['death'].isna()).astype('int')  # all 0, no death in baseline
        elif pasc == 'death_acute':
            pasc_flag = df['death acute'].astype('int')
            pasc_t2e = df['death t2e acute'].astype('float')
            pasc_baseline = (df['death'].isna()).astype('int')  # all 0, no death in baseline
        elif pasc == 'death_postacute':
            pasc_flag = df['death postacute'].astype('int')
            pasc_t2e = df['death t2e postacute'].astype('float')
            pasc_baseline = (df['death'].isna()).astype('int')  # all 0, no death in baseline
        elif pasc == 'hospitalization_acute':
            pasc_flag = df['hospitalization-acute-flag'].astype('int')
            pasc_t2e = df['hospitalization-acute-t2e'].astype('float')
            pasc_baseline = (df['hospitalization-base'].isna()).astype('int')  # all 0, not acounting incident inpatient
        elif pasc == 'hospitalization_postacute':
            pasc_flag = df['hospitalization-postacute-flag'].astype('int')
            pasc_t2e = df['hospitalization-postacute-t2e'].astype('float')
            pasc_baseline = (df['hospitalization-base'].isna()).astype('int')  # all 0, not acounting incident inpatient
        elif pasc in pasc_list:
            pasc_flag = (df['dx-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['dx-t2e@' + pasc].astype('float')
            pasc_baseline = df['dx-base@' + pasc]
        elif pasc in addedPASC_list:
            pasc_flag = (df['dxadd-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['dxadd-t2e@' + pasc].astype('float')
            pasc_baseline = df['dxadd-base@' + pasc]
        elif pasc in brainfog_list:
            pasc_flag = (df['dxbrainfog-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['dxbrainfog-t2e@' + pasc].astype('float')
            pasc_baseline = df['dxbrainfog-base@' + pasc]
        elif pasc == 'any_CFR':
            pasc_flag = df['any_CFR_flag'].astype('int')
            pasc_t2e = df['any_CFR_t2e'].astype('float')
            pasc_baseline = df['any_CFR_baseline']
        elif pasc in CFR_list:
            pasc_flag = (df['dxCFR-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['dxCFR-t2e@' + pasc].astype('float')
            pasc_baseline = df['dxCFR-base@' + pasc]

        # considering competing risks of death, and (all, acute, post-acute) death as outcomes
        if pasc == 'death':
            print('considering pasc death over all time, not set competing risk')
        elif pasc == 'death_acute':
            print('considering pasc death in acute phase, not set competing risk')
        elif pasc == 'death_postacute':
            print('considering pasc death in POST acute phase, set acute death as censored')
            pasc_flag.loc[df['death acute'] == 1] = 0
        else:
            # general conditions
            print('considering general pasc in POST acute phase, set any death as competing risk')
            death_flag = df['death']
            death_t2e = df['death t2e']
            # if there is a death event, pasc from 30-180, post acute death censored event time
            # acute death < 30, censored at 30 days
            pasc_flag.loc[(death_t2e <= pasc_t2e)] = 2

            print('#death:', (death_t2e == pasc_t2e).sum(), ' #death in covid+:',
                  df_label[(death_t2e == pasc_t2e)].sum(),
                  'ratio of death in covid+:', df_label[(death_t2e == pasc_t2e)].mean())

        # Select population free of outcome at baseline
        idx = (pasc_baseline < 1)
        covid_label = df_label[idx]  # actually current is the treatment label
        n_covid_pos = covid_label.sum()
        n_covid_neg = (covid_label == 0).sum()
        print('n case:', n_covid_pos, 'n control:', n_covid_neg, )

        # Sample all negative
        # sampled_neg_index = covid_label[(covid_label == 0)].sample(n=int(args.negative_ratio * n_covid_pos),
        #                                                                replace=False,
        #                                                                random_state=args.random_seed).index
        sampled_neg_index = covid_label[(covid_label == 0)].sample(frac=1,
                                                                   replace=False,
                                                                   random_state=args.random_seed).index

        pos_neg_selected = pd.Series(False, index=pasc_baseline.index)
        pos_neg_selected[sampled_neg_index] = True
        pos_neg_selected[covid_label[covid_label == 1].index] = True
        #
        # pat_info = df_info.loc[pos_neg_selected, :]
        covid_label = df_label[pos_neg_selected]
        covs_array = df_covs.loc[pos_neg_selected, :]
        pasc_flag = pasc_flag[pos_neg_selected]
        pasc_t2e = pasc_t2e[pos_neg_selected]
        print('pasc_t2e.describe():', pasc_t2e.describe())

        # all the post-acute events and t2e are from 30/31 - 180, this is just for data clean
        # pasc_t2e[pasc_t2e <= 30] = 30

        print('pasc_flag.value_counts():\n', pasc_flag.value_counts())
        print(i, pasc, '-- Selected cohorts {}/{} ({:.2f}%), Paxlovid pos:neg = {}:{} sample ratio -/+={}, '
                       'Overall pasc events pos:neg:death '
                       '= {}:{}:{}'.format(
            pos_neg_selected.sum(), len(df), pos_neg_selected.sum() / len(df) * 100,
            covid_label.sum(), (covid_label == 0).sum(), args.negative_ratio,
            (pasc_flag == 1).sum(), (pasc_flag == 0).sum(), (pasc_flag == 2).sum()))

        # model = ml.PropensityEstimator(learner='LR', random_seed=args.random_seed).cross_validation_fit(covs_array,
        #                                                                                                 covid_label,
        #                                                                                                 verbose=0)
        # , paras_grid = {
        #     'penalty': 'l2',
        #     'C': 0.03162277660168379,
        #     'max_iter': 200,
        #     'random_state': 0}

        model = ml.PropensityEstimator(learner='LR', paras_grid={
            'penalty': ['l2'],  # 'l1',
            'C': 10 ** np.arange(-1.5, 1., 0.25),  # 10 ** np.arange(-2, 1.5, 0.5),
            'max_iter': [150],  # [100, 200, 500],
            'random_state': [args.random_seed], }, add_none_penalty=False).cross_validation_fit(
            covs_array, covid_label, verbose=0)

        ps = model.predict_ps(covs_array)
        model.report_stats()
        iptw = model.predict_inverse_weight(covs_array, covid_label, stabilized=True, clip=True)
        smd, smd_weighted, before, after = model.predict_smd(covs_array, covid_label, abs=False, verbose=True)
        # plt.scatter(range(len(smd)), smd)
        # plt.scatter(range(len(smd)), smd_weighted)
        # plt.show()
        print('n unbalanced covariates before:after = {}:{}'.format(
            (np.abs(smd) > SMD_THRESHOLD).sum(),
            (np.abs(smd_weighted) > SMD_THRESHOLD).sum())
        )
        out_file_balance = r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcov/{}-{}-results.csv'.format(
            args.cohorttype,
            args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
            args.exptype,  # '-select' if args.selectpasc else '',
            i, _clean_name_(pasc))

        utils.check_and_mkdir(out_file_balance)
        model.results.to_csv(out_file_balance)  # args.save_model_filename +

        df_summary = summary_covariate(covs_array, covid_label, iptw, smd, smd_weighted, before, after)
        df_summary.to_csv(
            '../data/recover/output/results/SSRI-{}-{}-{}-mentalcov/{}-{}-evaluation_balance.csv'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                i, _clean_name_(pasc)))

        dfps = pd.DataFrame({'ps': ps, 'iptw': iptw, 'Exposure': covid_label})

        dfps.to_csv(
            '../data/recover/output/results/SSRI-{}-{}-{}-mentalcov/{}-{}-evaluation_ps-iptw.csv'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                i, _clean_name_(pasc)))
        try:
            figout = r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcov/{}-{}-PS.png'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                i, _clean_name_(pasc))
            print('Dump ', figout)

            ax = plt.subplot(111)
            sns.histplot(
                dfps, x="ps", hue="SSRI", element="step",
                stat="percent", common_norm=False, bins=25,
            )
            plt.tight_layout()
            # plt.show()
            plt.title(pasc, fontsize=12)
            plt.savefig(figout)
            plt.close()
        except Exception as e:
            print('Dump Error', figout)
            print(str(e))
            plt.close()

        km, km_w, cox, cox_w, cif, cif_w = weighted_KM_HR(
            covid_label, iptw, pasc_flag, pasc_t2e,
            fig_outfile=r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcov/{}-{}-km.png'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                i, _clean_name_(pasc)),
            title=pasc,
            legends={'case': case_label, 'control': ctrl_label})

        try:
            # change 2022-03-20 considering competing risk 2
            # change 2024-02-29 add CI for CIF difference and KM difference
            _results = [i, pasc,
                        covid_label.sum(), (covid_label == 0).sum(),
                        (pasc_flag[covid_label == 1] == 1).sum(), (pasc_flag[covid_label == 0] == 1).sum(),
                        (pasc_flag[covid_label == 1] == 1).mean(), (pasc_flag[covid_label == 0] == 1).mean(),
                        (pasc_flag[covid_label == 1] == 2).sum(), (pasc_flag[covid_label == 0] == 2).sum(),
                        (pasc_flag[covid_label == 1] == 2).mean(), (pasc_flag[covid_label == 0] == 2).mean(),
                        (np.abs(smd) > SMD_THRESHOLD).sum(), (np.abs(smd_weighted) > SMD_THRESHOLD).sum(),
                        np.abs(smd).max(), np.abs(smd_weighted).max(),
                        km[2], km[3], km[6].p_value,
                        list(km[6].diff_of_mean), list(km[6].diff_of_mean_lower), list(km[6].diff_of_mean_upper),
                        cif[2], cif[4], cif[5], cif[6], cif[7], cif[8], cif[9],
                        list(cif[10].diff_of_mean), list(cif[10].diff_of_mean_lower), list(cif[10].diff_of_mean_upper),
                        cif[10].p_value,
                        km_w[2], km_w[3], km_w[6].p_value,
                        list(km_w[6].diff_of_mean), list(km_w[6].diff_of_mean_lower), list(km_w[6].diff_of_mean_upper),
                        cif_w[2], cif_w[4], cif_w[5], cif_w[6], cif_w[7], cif_w[8], cif_w[9],
                        list(cif_w[10].diff_of_mean), list(cif_w[10].diff_of_mean_lower),
                        list(cif_w[10].diff_of_mean_upper),
                        cif_w[10].p_value,
                        cox[0], cox[1], cox[3].summary.p.treatment if pd.notna(cox[3]) else np.nan, cox[2], cox[4],
                        cox_w[0], cox_w[1], cox_w[3].summary.p.treatment if pd.notna(cox_w[3]) else np.nan, cox_w[2],
                        cox_w[4], model.best_hyper_paras]
            causal_results.append(_results)
            results_columns_name = [
                'i', 'pasc', 'case+', 'ctrl-',
                'no. pasc in +', 'no. pasc in -', 'mean pasc in +', 'mean pasc in -',
                'no. death in +', 'no. death in -', 'mean death in +', 'mean death in -',
                'no. unbalance', 'no. unbalance iptw', 'max smd', 'max smd iptw',
                'km-diff', 'km-diff-time', 'km-diff-p',
                'km-diff-2', 'km-diff-CILower', 'km-diff-CIUpper',
                'cif-diff', "cif_1", "cif_0", "cif_1_CILower", "cif_1_CIUpper", "cif_0_CILower", "cif_0_CIUpper",
                'cif-diff-2', 'cif-diff-CILower', 'cif-diff-CIUpper', 'cif-diff-p',
                'km-w-diff', 'km-w-diff-time', 'km-w-diff-p',
                'km-w-diff-2', 'km-w-diff-CILower', 'km-w-diff-CIUpper',
                'cif-w-diff', "cif_1_w", "cif_0_w", "cif_1_w_CILower", "cif_1_w_CIUpper", "cif_0_w_CILower",
                "cif_0_w_CIUpper", 'cif-w-diff-2', 'cif-w-diff-CILower', 'cif-w-diff-CIUpper', 'cif-w-diff-p',
                'hr', 'hr-CI', 'hr-p', 'hr-logrank-p', 'hr_different_time',
                'hr-w', 'hr-w-CI', 'hr-w-p', 'hr-w-logrank-p', "hr-w_different_time", 'best_hyper_paras']
            print('causal result:\n', causal_results[-1])

            if i % 2 == 0:
                pd.DataFrame(causal_results, columns=results_columns_name). \
                    to_csv(
                    r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcov/causal_effects_specific-snapshot-{}.csv'.format(
                        args.cohorttype,
                        args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                        args.exptype,  # '-select' if args.selectpasc else '',
                        i))
        except:
            print('Error in ', i, pasc)
            df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

            df_causal.to_csv(
                r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcov/causal_effects_specific-ERRORSAVE.csv'.format(
                    args.cohorttype,
                    args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                    args.exptype,  # '-select' if args.selectpasc else '',
                ))

        print('done one pasc, time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

    df_causal.to_csv(
        r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcov/causal_effects_specific.csv'.format(
            args.cohorttype,
            args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
            args.exptype,  # '-select' if args.selectpasc else '',
        ))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
