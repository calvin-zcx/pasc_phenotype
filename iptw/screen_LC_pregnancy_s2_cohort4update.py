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
import ast

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
                                               'white', 'black', 'hispanic', 'nonhispanic',
                                               '18to25', '25to35', '35to50', '50to65',
                                               'less65', 'above65',
                                               'less65omicronbroad', 'above65omicronbroad',
                                               'less50', 'above50',
                                               '65to75', '75above', '20to40', '40to55', '55to65',
                                               'Anemia', 'Arrythmia', 'CKD', 'CPD-COPD', 'CAD',
                                               'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids',
                                               'healthy',
                                               '03-20-06-20', '07-20-10-20', '11-20-02-21',
                                               '03-21-06-21', '07-21-11-21',
                                               '1stwave', 'delta', 'alpha',
                                               'omicron', 'omicronafter', 'omicronbroad', 'beforeomicron',
                                               'preg-pos-neg',
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
                                               'depression', 'anxiety', 'SMI', 'nonSMI',
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
                                 'ssri-acute0-15-cleanv2',
                                 'ssri-acute0-15-incident',
                                 'ssri-acute0-15-incident_nobasemental',
                                 'ssri-acute0-15-incident_norequiremental',
                                 'ssri-acute0-15-incident-pax05',
                                 'ssri-acute0-15-incident-pax15',
                                 'ssri-acute0-15-incident-continue',

                                 # 'fluvoxamine-acute0-15-incident-continue',
                                 'fluvoxamine-base180withmental-acutevsnot',
                                 'fluvoxamine-base180-acutevsnot',
                                 'fluvoxamine-base180withmental-acutevsnot-continue',

                                 'ssri-post30',
                                 'ssri-post30-basemental',
                                 'ssri-post30-nobasemental',

                                 'ssri-base180-acutevsnot',
                                 'ssri-base180-acutevsnot-nosnriother',  # sensitivity 2025-2-21
                                 'ssri-base180withmental-acutevsnot',

                                 'ssri-base180-acuteS1R2vsnoSSRI',
                                 # individual
                                 'ssri-base180-S1Racutevsnot',
                                 'ssri-base180-S1RacutevsNonS1R',
                                 'ssri-base180-S1RacutevsNonS1RNoCita',
                                 'ssri-base180-S1RNoEscacutevsNonS1R',

                                 'ssri-base180-fluvoxamineacutevsnot',
                                 'ssri-base180-fluoxetineacutevsnot',
                                 'ssri-base180-escitalopramacutevsnot',
                                 'ssri-base180-citalopramacutevsnot',
                                 'ssri-base180-sertralineacutevsnot',
                                 'ssri-base180-paroxetineacutevsnot',
                                 'ssri-base180-vilazodoneacutevsnot',

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
                                 'bupropion-acute0-15',
                                 'ssriVSbupropion-base-180-0',
                                 'ssriVSbupropion-acute0-15',

                                 'bupropion-base-180-0-clean',
                                 'bupropion-acute0-15-clean',
                                 'ssriVSbupropion-base-180-0-clean',
                                 'ssriVSbupropion-acute0-15-clean',
                                 'ssri-base-180-0-cleanv2',

                                 ], default='ssri-base180-acutevsnot')  # 'base180-0'

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


def _decode_list_date_code(lstr):
    result = ast.literal_eval(lstr)

    return result


def feature_process_pregnancy(df):
    print('feature_process_additional, df.shape', df.shape)
    start_time = time.time()

    code_pregoutcomecat = utils.load(r'../data/mapping/pregnancy_code_to_outcome_categories_mapping.pkl')

    df['gestational age at delivery'] = np.nan  # week, used for determine other outcome
    df['gestational age of infection'] = np.nan  # week

    df['preterm birth<37'] = 0
    df['preterm birth<37Andlivebirth'] = 0
    df['preterm birth<34'] = 0
    df['preterm birth<34Andlivebirth'] = 0

    df['infection at trimester1'] = 0
    df['infection at trimester2'] = 0
    df['infection at trimester3'] = 0

    # 2025-7-18 will revise later per ob comments
    df['preg_outcome-livebirth'] = 0
    df['preg_outcome-stillbirth'] = 0
    df['preg_outcome-miscarriage'] = 0
    df['preg_outcome-abortion'] = 0
    df['preg_outcome-other'] = 0
    df['preg_outcome-miscarriage<20week'] = 0
    df['preg_outcome-abortion<20week'] = 0

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

        flag_maternal_age = row['flag_maternal_age']  # age at pre or delivery, there is another age at infection

        delivery_code = row['flag_delivery_code']
        exclude_dx = row['flag_exclusion_dx_detail']
        exclude_px = row['flag_exclusion_px_detail']

        if pd.notna(delivery_code):
            if isinstance(delivery_code, float):
                print(delivery_code)
                print(row.iloc[:5])

            delivery_code = delivery_code.strip().upper().replace('.', '')
            if delivery_code in code_pregoutcomecat:
                out_info = code_pregoutcomecat[delivery_code]
                col = 'preg_outcome-' + out_info[0]
                df.loc[index, col] = 1

        if pd.notna(exclude_dx):
            exclude_dx = _decode_list_date_code(exclude_dx)
            for _tup in exclude_dx:
                _day, _code = _tup
                _code = _code.strip().upper().replace('.', '')
                if _code in code_pregoutcomecat:
                    out_info = code_pregoutcomecat[_code]
                    col = 'preg_outcome-' + out_info[0]
                    df.loc[index, col] = 1

        if pd.notna(exclude_px):
            exclude_px = _decode_list_date_code(exclude_px)
            for _tup in exclude_px:
                _day, _code = _tup
                _code = _code.strip().upper().replace('.', '')
                if _code in code_pregoutcomecat:
                    out_info = code_pregoutcomecat[_code]
                    col = 'preg_outcome-' + out_info[0]
                    df.loc[index, col] = 1

        if pd.notna(del_date) and pd.notna(preg_date):
            gesage = (del_date - preg_date).days / 7
            df.loc[index, 'gestational age at delivery'] = gesage
            df.loc[index, 'preterm birth<37'] = int(gesage < 37)
            df.loc[index, 'preterm birth<34'] = int(gesage < 34)

            if df.loc[index, 'preg_outcome-livebirth'] == 1:
                df.loc[index, 'preterm birth<37Andlivebirth'] = int(gesage < 37)
                df.loc[index, 'preterm birth<34Andlivebirth'] = int(gesage < 34)

            if gesage < 20:
                if df.loc[index, 'preg_outcome-miscarriage'] == 1:
                    df.loc[index, 'preg_outcome-miscarriage<20week'] = 1

                if df.loc[index, 'preg_outcome-abortion'] == 1:
                    df.loc[index, 'preg_outcome-abortion<20week'] = 1

        if pd.notna(index_date) and pd.notna(preg_date):
            infectage = (index_date - preg_date).days / 7
            df.loc[index, 'gestational age of infection'] = infectage
            if preg_date <= index_date <= del_date:
                # add this to more accurate capture following, though with infection during pregnancy EC, results are same
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
    elif severity == 'hispanic':
        print('Considering hispanic cohorts')
        df = df.loc[(df['Hispanic: Yes'] == 1), :].copy()
    elif severity == 'nonhispanic':
        print('Considering nonhispanic cohorts')
        df = df.loc[(df['Hispanic: No'] == 1), :].copy()
    elif severity == '18to25':
        print('Considering age@18-24 cohorts')
        df = df.loc[(df['age@18-24'] == 1), :].copy()
    elif severity == '25to35':
        print('Considering age@25-34 cohorts')
        df = df.loc[(df['age@25-34'] == 1), :].copy()
    elif severity == '35to50':
        print('Considering age@35-49 cohorts')
        df = df.loc[(df['age@35-49'] == 1), :].copy()
    elif severity == '50to65':
        print('Considering age@50-64 cohorts')
        df = df.loc[(df['age@50-64'] == 1), :].copy()
    elif severity == 'less65':
        print('Considering less65 cohorts')
        # df = df.loc[(df['20-<40 years'] == 1) | (df['40-<55 years'] == 1) | (df['55-<65 years'] == 1), :].copy()
        df = df.loc[(df['age@18-24'] == 1) | (df['age@25-34'] == 1) | (df['age@35-49'] == 1) | (df['age@50-64'] == 1),
             :].copy()
    elif severity == 'above65':
        print('Considering above65 cohorts')
        df = df.loc[(df['65-<75 years'] == 1) | (df['75-<85 years'] == 1) | (df['85+ years'] == 1), :].copy()
    elif severity == 'less50':
        print('Considering less50 cohorts')
        # df = df.loc[(df['20-<40 years'] == 1) | (df['40-<55 years'] == 1) | (df['55-<65 years'] == 1), :].copy()
        df = df.loc[(df['age@18-24'] == 1) | (df['age@25-34'] == 1) | (df['age@35-49'] == 1), :].copy()
    elif severity == 'above50':
        print('Considering above50 cohorts')
        df = df.loc[
             (df['age@50-64'] == 1) | (df['65-<75 years'] == 1) | (df['75-<85 years'] == 1) | (df['85+ years'] == 1),
             :].copy()
    elif severity == 'less65omicronbroad':
        print('Considering less65omicronbroad cohorts')
        # df = df.loc[(df['20-<40 years'] == 1) | (df['40-<55 years'] == 1) | (df['55-<65 years'] == 1), :].copy()
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 12, 1, 0, 0)) &
                    ((df['age@18-24'] == 1) | (df['age@25-34'] == 1) | (df['age@35-49'] == 1) | (df['age@50-64'] == 1)),
             :].copy()
    elif severity == 'above65omicronbroad':
        print('Considering above65omicronbroad cohorts')
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 12, 1, 0, 0)) &
                    ((df['65-<75 years'] == 1) | (df['75-<85 years'] == 1) | (df['85+ years'] == 1)), :].copy()

    elif severity == '20to40':
        print('Considering 20to40 cohorts')
        df = df.loc[(df['20-<40 years'] == 1), :].copy()
    elif severity == '40to55':
        print('Considering 40to55 cohorts')
        df = df.loc[(df['40-<55 years'] == 1), :].copy()
    elif severity == '55to65':
        print('Considering 55to65 cohorts')
        df = df.loc[(df['55-<65 years'] == 1), :].copy()
    elif severity == '65to75':
        print('Considering 65to75 cohorts')
        df = df.loc[(df['65-<75 years'] == 1), :].copy()
    elif severity == '75above':
        print('Considering 75above cohorts')
        df = df.loc[(df['75-<85 years'] == 1) | (df['85+ years'] == 1), :].copy()
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
    elif severity == 'alpha':
        print('Considering patients in Alpha + others wave, Oct.-1-2020 to May-31-2021')
        df = df.loc[(df['index date'] >= datetime.datetime(2020, 10, 1, 0, 0)) & (
                df['index date'] < datetime.datetime(2021, 6, 1, 0, 0)), :].copy()
    elif severity == 'delta':
        print('Considering patients in Delta wave, June-1-2021 to Nov.-30-2021')
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 6, 1, 0, 0)) & (
                df['index date'] < datetime.datetime(2021, 12, 1, 0, 0)), :].copy()
    elif severity == 'omicron':
        print('Considering patients in omicron wave, December-1-2021 to March.-30-2022')
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 12, 1, 0, 0)) & (
                df['index date'] < datetime.datetime(2022, 4, 1, 0, 0)), :].copy()
    elif severity == 'omicronafter':
        print('Considering patients in omicronafter wave, April-1-2022 to end of inclusion windows')
        df = df.loc[(df['index date'] >= datetime.datetime(2022, 4, 1, 0, 0)), :].copy()
    elif severity == 'omicronbroad':
        print('Considering patients in omicron wave, >= December-1-2021 ')
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 12, 1, 0, 0)), :].copy()
    elif severity == 'beforeomicron':
        print('Considering patients in omicron wave, < December-1-2021 ')
        df = df.loc[(df['index date'] < datetime.datetime(2021, 12, 1, 0, 0)), :].copy()
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

    elif severity == 'depression':
        print('Considering base depression cohorts')
        df = df.loc[(df['mental-base@Depressive Disorders'] >= 1), :].copy()
    elif severity == 'anxiety':
        print('Considering base anxiety cohorts')
        df = df.loc[(df['mental-base@Anxiety Disorders'] >= 1), :].copy()
    elif severity == 'SMI':
        print('Considering base SMI cohorts')
        df = df.loc[(df['mental-base@SMI'] >= 1), :].copy()
    elif severity == 'nonSMI':
        print('Considering base nonSMI cohorts')
        df = df.loc[(df['mental-base@non-SMI'] >= 1), :].copy()
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

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    # # load data here
    print('Step 1: load data')
    infile = r'../data/recover/output/pregnancy_output_y4/pregnant_yr4.csv'
    # infile = r'../data/recover/output/pregnancy_CX_0501_2025/pregnancy_{}.csv'.format("wcm_pcornet_all")
    print('Loading:', infile)
    df = pd.read_csv(infile,
                     dtype={'patid': str, 'site': str, 'zip': str, 'flag_delivery_code':str},
                     parse_dates=['index date', 'dob',
                                  'flag_delivery_date',
                                  'flag_pregnancy_start_date',
                                  'flag_pregnancy_end_date'
                                  ],
                     # nrows=200000, # for debug
                     )
    print(df.shape)
    N = len(df)

    print('Loading:', infile,  'Done! Time used:',
          time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print('Before selecting pregnant, len(df)\n', len(df))
    n = len(df)
    df = df.loc[((df['flag_pregnancy'] == 1) | (df['flag_exclusion'] == 1)), :]
    print('After selecting pregnant, len(df),\n',
          '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))
    # zz
    df = feature_process_pregnancy(df)

    # df.to_csv(r'../data/recover/output/pregnancy_CX_0501_2025/pregnancy_{}-addFeat-test.csv'.format("wcm_pcornet_all"))
    # zz

    print('Before selecting pregnant after +30 days, len(df)\n', len(df))
    n = len(df)
    time_order_flag = (df['index date'] + datetime.timedelta(days=30) <= df['flag_pregnancy_start_date'])
    df = df.loc[time_order_flag, :]
    print('After selecting pregnant >= +30 days, len(df),\n',
          '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))


    df.to_csv(r'../data/recover/output/pregnancy_output_y4/pregnant_yr4_pergnantOnsetGEinfect30days.csv') # pregnancy_output_y4
    print('Cohort build Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print('Before selecting pregnant after +180 days, len(df)\n', len(df))
    n = len(df)
    time_order_flag = (df['index date'] + datetime.timedelta(days=180) <= df['flag_pregnancy_start_date'])
    df = df.loc[time_order_flag, :]
    print('After selecting pregnant after +180 days, len(df),\n',
          '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))

    pasc_flag = df['any_pasc_flag'].astype('int')
    pasc_t2e_label = 'any_pasc_t2e'

    # pasc_flag = df['any_CFR_flag'].astype('int')
    # pasc_t2e_label = 'any_CFR_t2e'
    # # # # #
    # pasc_flag = df['any_brainfog_flag'].astype('int')
    # pasc_t2e_label = 'any_brainfog_t2e'
    # # # #
    # pasc_flag = (df['dxMECFS-out@ME/CFS'].copy() >= 1).astype('int')
    # pasc_t2e = 'dxMECFS-t2e@ME/CFS'
    # # #
    # pasc_flag = (df['dx-out@' + 'PASC-General'].copy() >= 1).astype('int')
    # pasc_t2e = 'dx-t2e@' + 'PASC-General'

    contrl_label = (df['any_pasc_flag'] == 0) & (df['any_CFR_flag'] == 0) & (df['any_brainfog_flag'] == 0) & \
                   (df['dxMECFS-out@ME/CFS'] == 0) & (df['dx-out@' + 'PASC-General'] == 0)

    df1 = df.loc[(pasc_flag > 0), :]
    print("Exposed group: ((pasc_flag > 0) ) len(df1)", len(df1))
    df0 = df.loc[(pasc_flag == 0), :]
    print("Contrl group:  ((pasc_flag == 0)  len(df0)", len(df0))
    df0 = df.loc[contrl_label, :]
    print("V2-Contrl group, no all LC:  contrl_label  len(df0)", len(df0))

    pasc_time = df1[pasc_t2e_label].apply(lambda x: datetime.timedelta(x))
    # delivery_time = df1['flag_delivery_date'] - df1['index date']
    # delivery_time = df1['flag_pregnancy_start_date'] - df1['index date']
    # time_order_flag_1 = pasc_time < delivery_time
    # time_order_flag_1 = (df1['index date'] + datetime.timedelta(days=180) <= df1['flag_pregnancy_start_date'])
    # df1 = df1.loc[time_order_flag_1, :]

    print("Exposed group: ((pasc_flag > 0) len(df1)", len(df1), 'VS',
          "Contrl group: ((pasc_flag == 0)  len(df0)", len(df0))

    print('***Preterm birth rate:')
    for col in ['preterm birth<37', 'preterm birth<37Andlivebirth',
                'preterm birth<34', 'preterm birth<34Andlivebirth', ]:
        print(col, 'among all pregnant:', '\t',
              'Exposed:', (df1[col] == 1).sum(), '{:.2f}%'.format((df1[col] == 1).mean() * 100), '\t',
              'Control:', (df0[col] == 1).sum(), '{:.2f}%'.format((df0[col] == 1).mean() * 100), )

        print(col, 'among live birth:', '\t',
              'Exposed:', (df1[col] == 1).sum(),
              '{:.2f}%'.format((df1[col] == 1).sum() / (df1['preg_outcome-livebirth'] == 1).sum() * 100), '\t',
              'Control:', (df0[col] == 1).sum(),
              '{:.2f}%'.format((df0[col] == 1).sum() / (df0['preg_outcome-livebirth'] == 1).sum() * 100)
              )

    print('***Other Outcomes among all pregnant:')
    for col in ['preg_outcome-livebirth', 'preg_outcome-stillbirth',
                'preg_outcome-miscarriage', 'preg_outcome-miscarriage<20week',
                'preg_outcome-abortion', 'preg_outcome-abortion<20week',
                'preg_outcome-other', ]:
        print(col, '\t',
              'Exposed:', (df1[col] == 1).sum(), '{:.2f}%'.format((df1[col] == 1).mean() * 100), '\t',
              'Control:', (df0[col] == 1).sum(), '{:.2f}%'.format((df0[col] == 1).mean() * 100),
              )

    # df0 = df.loc[(pasc_flag == 0), :]
    # print("Contrl group: ((pasc_flag == 0)  len(df0)", len(df0))
    # print('***Preterm birth rate:')
    # for col in ['preterm birth<37', 'preterm birth<37Andlivebirth',
    #             'preterm birth<34', 'preterm birth<34Andlivebirth', ]:
    #     print(col, 'among all pregnant:',  (df0[col] == 1).sum(), '{:.2f}%'.format((df0[col] == 1).mean() * 100))
    #     print(col, 'among live birth:', (df0[col] == 1).sum(),
    #           '{:.2f}%'.format((df0[col] == 1).sum() / (df0['preg_outcome-livebirth'] == 1).sum() * 100)
    #           )
    #
    # print('***Other Outcomes among all pregnant:')
    # for col in ['preg_outcome-livebirth', 'preg_outcome-stillbirth',
    #             'preg_outcome-miscarriage', 'preg_outcome-miscarriage<20week',
    #             'preg_outcome-abortion', 'preg_outcome-abortion<20week',
    #             'preg_outcome-other', ]:
    #     print(col, (df0[col] == 1).sum(), '{:.2f}%'.format((df0[col] == 1).mean() * 100))



    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
