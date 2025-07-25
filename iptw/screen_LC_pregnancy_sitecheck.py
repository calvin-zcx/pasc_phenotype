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
    # print('Step 1: load data')
    # # in_file = r'../data/recover/output/pregnancy_output/pregnant_yr4.csv'
    # # in_file = r'recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNet-SSRI-v6-withmentalCFSCVD.csv'
    # in_file = r'recover25Q2_covid_pos_addPaxFeats-withexposure.csv'
    # in_file = r'recover25Q2_covid_pos_addPaxFeats-addADHDctrl-withexposure.csv'
    #
    # # df = pd.read_csv(in_file,
    # #                  dtype={'patid': str, 'site': str, 'zip': str},
    # #                  parse_dates=['index date', 'dob',
    # #                               'flag_delivery_date',
    # #                               'flag_pregnancy_start_date',
    # #                               'flag_pregnancy_end_date'
    # #                               ])
    # df = pd.read_csv(in_file, dtype={'patid': str, 'site': str, 'zip': str},
    #                  parse_dates=['index date', 'dob',
    #                               'flag_delivery_date',
    #                               'flag_pregnancy_start_date',
    #                               'flag_pregnancy_end_date'
    #                               ])
    # print('df.shape:', df.shape)
    #
    # print('Step 2: select covid+, female, age <=50')
    # start_time = time.time()
    # print('*' * 100)
    # print('Applying more specific/flexible eligibility criteria for cohort selection')
    # N = len(df)
    # # covid positive patients only
    # print('Before selecting covid+, len(df)\n', len(df))
    # n = len(df)
    # df = df.loc[df['covid'] == 1, :]
    # print('After selecting covid+, len(df),\n',
    #       '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))
    #
    # # # at least 6 month follow-up, up to 2023-4-30
    # # n = len(df)
    # # df = df.loc[(df['index date'] <= datetime.datetime(2022, 10, 31, 0, 0)), :]  # .copy()
    # # print('After selecting index date <= 2022-10-31, len(df)\n',
    # #       '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))
    #
    # # select female
    # print('Before selecting female=1 len(df)\n', len(df))
    # n = len(df)
    # df = df.loc[df['Female'] == 1, :]  # .copy()
    # print('After selecting female, len(df)\n',
    #       '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))
    #
    # # select age 18-50
    # n = len(df)
    # df = df.loc[df['age'] <= 50, :]
    # print('After selecting age <= 50, len(df)\n',
    #       '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))
    #
    # print('Before select_subpopulation, len(df)', len(df))
    # df = select_subpopulation(df, args.severity)
    # print('After select_subpopulation, len(df)', len(df))
    #
    # print('stablize data by df.copy()')
    # df = df.copy()
    #
    # print('Step 3: add Long COVID label')
    # # pre-process data a little bit
    # selected_cols = [x for x in df.columns if (
    #         x.startswith('DX:') or
    #         x.startswith('MEDICATION:') or
    #         x.startswith('CCI:') or
    #         x.startswith('obc:')
    # )]
    # df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')
    # df.loc[:, r"DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"] = \
    #     (df.loc[:, r'DX: Hypertension'] & (
    #             df.loc[:, r'DX: Diabetes Type 1'] | df.loc[:, r'DX: Diabetes Type 2'])).astype('int')
    #
    # # baseline part have been binarized already
    # selected_cols = [x for x in df.columns if
    #                  (x.startswith('dx-out@') or
    #                   x.startswith('dxadd-out@') or
    #                   x.startswith('dxbrainfog-out@') or
    #                   x.startswith('covidmed-out@') or
    #                   x.startswith('smm-out@') or
    #                   x.startswith('dxCFR-out@') or
    #                   x.startswith('mental-base@') or
    #                   x.startswith('dxMECFS-base@') or
    #                   x.startswith('dxCVDdeath-base@')
    #                   )]
    # df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')
    #
    # # data clean for <0 error death records, and add censoring to the death time to event columns
    #
    # df.loc[df['death t2e'] < 0, 'death'] = 0
    # df.loc[df['death t2e'] < 0, 'death t2e'] = 9999
    #
    # # death in [0, 180). 1: evnt, 0: censored, censored at 180. death at 180, not counted, thus use <
    # df['death all'] = ((df['death'] == 1) & (df['death t2e'] >= 0) & (df['death t2e'] < 180)).astype('int')
    # df['death t2e all'] = df['death t2e'].clip(lower=0, upper=180)
    # df.loc[df['death all'] == 0, 'death t2e all'] = df['maxfollowup'].clip(lower=0, upper=180)
    #
    # # death in [0, 30). 1: evnt, 0: censored, censored at 30. death at 30, not counted, thus use <
    # df['death acute'] = ((df['death'] == 1) & (df['death t2e'] <= 30)).astype('int')
    # df['death t2e acute'] = df['death t2e all'].clip(upper=31)
    #
    # # death in [30, 180).  1:event, 0: censored. censored at 180 or < 30, say death at 20, flag is 0, time is 20
    # df['death postacute'] = ((df['death'] == 1) & (df['death t2e'] >= 31) & (df['death t2e'] < 180)).astype('int')
    # df['death t2e postacute'] = df['death t2e all']
    #
    # df['cvd death postacute'] = ((df['dxCVDdeath-out@death_cardiovascular'] >= 1) & (df['death'] == 1)
    #                              & (df['death t2e'] >= 31) & (df['death t2e'] < 180)).astype('int')
    # df['cvd death t2e postacute'] = df['death t2e all']
    #
    # df['hospitalization-acute-flag'] = (df['hospitalization-acute-flag'] >= 1).astype('int')
    # df['hospitalization-acute-t2e'] = df['hospitalization-acute-t2e'].clip(upper=31)
    # df['hospitalization-postacute-flag'] = (df['hospitalization-postacute-flag'] >= 1).astype('int')
    #
    # #
    # # pre-process PASC info
    # df_pasc_info = pd.read_excel(r'../prediction/output/causal_effects_specific_withMedication_v3.xlsx',
    #                              sheet_name='diagnosis')
    # addedPASC_encoding = utils.load(r'../data/mapping/addedPASC_index_mapping.pkl')
    # addedPASC_list = list(addedPASC_encoding.keys())
    # brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
    # brainfog_list = list(brainfog_encoding.keys())
    #
    # CFR_encoding = utils.load(r'../data/mapping/cognitive-fatigue-respiratory_index_mapping.pkl')
    # CFR_list = list(CFR_encoding.keys())
    #
    # mecfs_encoding = utils.load(r'../data/mapping/mecfs_index_mapping.pkl')
    # mecfs_list = list(mecfs_encoding.keys())
    #
    # pasc_simname = {}
    # pasc_organ = {}
    # for index, rows in df_pasc_info.iterrows():
    #     pasc_simname[rows['pasc']] = (rows['PASC Name Simple'], rows['Organ Domain'])
    #     pasc_organ[rows['pasc']] = rows['Organ Domain']
    #
    # for p in addedPASC_list:
    #     pasc_simname[p] = (p, 'General-add')
    #     pasc_organ[p] = 'General-add'
    #
    # for p in brainfog_list:
    #     pasc_simname[p] = (p, 'brainfog')
    #     pasc_organ[p] = 'brainfog'
    #
    # for p in CFR_list:
    #     pasc_simname[p] = (p, 'cognitive-fatigue-respiratory')
    #     pasc_organ[p] = 'cognitive-fatigue-respiratory'
    #
    # for p in mecfs_list:
    #     pasc_simname[p] = (p, 'General-add')
    #     pasc_organ[p] = 'General-add'
    #
    # # for p in mecfs_list:
    # #     pasc_simname[p] = (p, 'ME/CFS')
    # #     pasc_organ[p] = 'ME/CFS'
    #
    # # pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    # pasc_list_raw = df_pasc_info.loc[df_pasc_info['selected_narrow'] == 1, 'pasc'].to_list()
    # _exclude_list = ['Pressure ulcer of skin', 'Fluid and electrolyte disorders']
    # pasc_list = [x for x in pasc_list_raw if x not in _exclude_list]
    #
    # pasc_add = ['smell and taste', ]
    # pasc_add_mecfs = ['ME/CFS', ]
    # print('len(pasc_list)', len(pasc_list), 'len(pasc_add)', len(pasc_add))
    # print('pasc_list:', pasc_list)
    # print('pasc_add', pasc_add)
    # print('pasc_add_mecfs', pasc_add_mecfs)
    #
    # for p in pasc_list:
    #     df[p + '_pasc_flag'] = 0
    # for p in pasc_add:
    #     df[p + '_pasc_flag'] = 0
    # for p in pasc_add_mecfs:
    #     df[p + '_pasc_flag'] = 0
    # for p in CFR_list:
    #     df[p + '_CFR_flag'] = 0
    #
    # # move brainfog_list_any and '_brainfog_flag'  below
    #
    # df['any_pasc_flag'] = 0
    # df['any_pasc_type'] = np.nan
    # df['any_pasc_t2e'] = 180  # np.nan
    # df['any_pasc_txt'] = ''
    # df['any_pasc_baseline'] = 0  # placeholder for screening, no special meaning, null column
    #
    # df['any_CFR_flag'] = 0
    # # df['any_CFR_type'] = np.nan
    # df['any_CFR_t2e'] = 180  # np.nan
    # df['any_CFR_txt'] = ''
    # df['any_CFR_baseline'] = 0  # placeholder for screening, no special meaning, null column
    #
    # # 2025-2-20, original list 7, current any brain fog excludes headache because already in individual any pasc
    # # ['Neurodegenerative', 'Memory-Attention', 'Headache',
    # # 'Sleep Disorder', 'Psych', 'Dysautonomia-Orthostatic', 'Stroke'])
    # df['any_brainfog_flag'] = 0
    # # df['any_brainfog_type'] = np.nan
    # df['any_brainfog_t2e'] = 180  # np.nan
    # df['any_brainfog_txt'] = ''
    # df['any_brainfog_baseline'] = 0  # placeholder for screening, no special meaning, null column
    # brainfog_list_any = ['Neurodegenerative', 'Memory-Attention',  # 'Headache',
    #                      'Sleep Disorder', 'Psych', 'Dysautonomia-Orthostatic', 'Stroke']
    # for p in brainfog_list_any:
    #     df[p + '_brainfog_flag'] = 0
    #
    # print('brainfog_list_any:', brainfog_list_any)
    # print('len(brainfog_list_any):', len(brainfog_list_any), 'len(brainfog_list)', len(brainfog_list))
    #
    # for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
    #     # for any 1 pasc
    #     t2e_list = []
    #     pasc_1_list = []
    #     pasc_1_name = []
    #     pasc_1_text = ''
    #     for p in pasc_list:
    #         if (rows['dx-out@' + p] > 0) and (rows['dx-base@' + p] == 0):
    #             t2e_list.append(rows['dx-t2e@' + p])
    #             pasc_1_list.append(p)
    #             pasc_1_name.append(pasc_simname[p])
    #             pasc_1_text += (pasc_simname[p][0] + ';')
    #
    #             df.loc[index, p + '_pasc_flag'] = 1
    #
    #     for p in pasc_add:
    #         if (rows['dxadd-out@' + p] > 0) and (rows['dxadd-base@' + p] == 0):
    #             t2e_list.append(rows['dxadd-t2e@' + p])
    #             pasc_1_list.append(p)
    #             pasc_1_name.append(pasc_simname[p])
    #             pasc_1_text += (pasc_simname[p][0] + ';')
    #
    #             df.loc[index, p + '_pasc_flag'] = 1
    #
    #     for p in pasc_add_mecfs:
    #         # dxMECFS-base@ME/CFS
    #         if (rows['dxMECFS-out@' + p] > 0) and (rows['dxMECFS-base@' + p] == 0):
    #             t2e_list.append(rows['dxMECFS-t2e@' + p])
    #             pasc_1_list.append(p)
    #             pasc_1_name.append(pasc_simname[p])
    #             pasc_1_text += (pasc_simname[p][0] + ';')
    #
    #             df.loc[index, p + '_pasc_flag'] = 1
    #
    #     if len(t2e_list) > 0:
    #         df.loc[index, 'any_pasc_flag'] = 1
    #         df.loc[index, 'any_pasc_t2e'] = np.min(t2e_list)
    #         df.loc[index, 'any_pasc_txt'] = pasc_1_text
    #     else:
    #         df.loc[index, 'any_pasc_flag'] = 0
    #         df.loc[index, 'any_pasc_t2e'] = rows[['dx-t2e@' + p for p in pasc_list]].max()  # censoring time
    #
    #     # for CFR pasc
    #     CFR_t2e_list = []
    #     CFR_1_list = []
    #     CFR_1_name = []
    #     CFR_1_text = ''
    #     for p in CFR_list:
    #         if (rows['dxCFR-out@' + p] > 0) and (rows['dxCFR-base@' + p] == 0):
    #             CFR_t2e_list.append(rows['dxCFR-t2e@' + p])
    #             CFR_1_list.append(p)
    #             CFR_1_name.append(pasc_simname[p])
    #             CFR_1_text += (pasc_simname[p][0] + ';')
    #
    #             df.loc[index, p + '_CFR_flag'] = 1
    #
    #     if len(CFR_t2e_list) > 0:
    #         df.loc[index, 'any_CFR_flag'] = 1
    #         df.loc[index, 'any_CFR_t2e'] = np.min(CFR_t2e_list)
    #         df.loc[index, 'any_CFR_txt'] = CFR_1_text
    #     else:
    #         df.loc[index, 'any_CFR_flag'] = 0
    #         df.loc[index, 'any_CFR_t2e'] = rows[['dxCFR-t2e@' + p for p in CFR_list]].max()  # censoring time
    #
    #     # for brain fog pasc
    #     brainfog_t2e_list = []
    #     brainfog_1_list = []
    #     brainfog_1_name = []
    #     brainfog_1_text = ''
    #     for p in brainfog_list_any:
    #         if (rows['dxbrainfog-out@' + p] > 0) and (rows['dxbrainfog-base@' + p] == 0):
    #             brainfog_t2e_list.append(rows['dxbrainfog-t2e@' + p])
    #             brainfog_1_list.append(p)
    #             brainfog_1_name.append(pasc_simname[p])
    #             brainfog_1_text += (pasc_simname[p][0] + ';')
    #
    #             df.loc[index, p + '_brainfog_flag'] = 1
    #
    #     if len(brainfog_t2e_list) > 0:
    #         df.loc[index, 'any_brainfog_flag'] = 1
    #         df.loc[index, 'any_brainfog_t2e'] = np.min(brainfog_t2e_list)
    #         df.loc[index, 'any_brainfog_txt'] = brainfog_1_text
    #     else:
    #         df.loc[index, 'any_brainfog_flag'] = 0
    #         df.loc[index, 'any_brainfog_t2e'] = rows[
    #             ['dxbrainfog-t2e@' + p for p in brainfog_list_any]].max()  # censoring time
    #
    # # End of defining ANY *** conditions
    # # Load index information
    # with open(r'../data/mapping/icd_pasc_mapping.pkl', 'rb') as f:
    #     icd_pasc = pickle.load(f)
    #     print('Load ICD-10 to PASC mapping done! len(icd_pasc):', len(icd_pasc))
    #     record_example = next(iter(icd_pasc.items()))
    #     print('e.g.:', record_example)
    #
    # with open(r'../data/mapping/pasc_index_mapping.pkl', 'rb') as f:
    #     pasc_encoding = pickle.load(f)
    #     print('Load PASC to encoding mapping done! len(pasc_encoding):', len(pasc_encoding))
    #     record_example = next(iter(pasc_encoding.items()))
    #     print('e.g.:', record_example)
    #
    # selected_screen_list = (['any_pasc', 'PASC-General', 'ME/CFS',
    #                          'death', 'death_acute', 'death_postacute', 'cvddeath_postacute',
    #                          'any_CFR', 'any_brainfog',
    #                          'hospitalization_acute', 'hospitalization_postacute'] +
    #                         CFR_list +
    #                         pasc_list +
    #                         addedPASC_list +
    #                         brainfog_list)
    #
    # # pd.Series(df.columns).to_csv('recover_covid_pos-with-pax-V3-column-name.csv')
    # pasc_flag = df['any_pasc_flag'].astype('int')
    # pasc_t2e = df['any_pasc_t2e'].astype('float')
    #
    # print('Severity cohorts:', args.severity,
    #       # 'df1.shape:', df1.shape,
    #       # 'df2.shape:', df2.shape,
    #       'df.shape:', df.shape,
    #       )
    #
    # # df.to_csv(r'../data/recover/output/pregnancy_output/pregnant_yr4.csv') # pregnancy_output_y4
    # df.to_csv(r'../data/recover/output/pregnancy_output_y4/pregnant_yr4.csv') # pregnancy_output_y4
    #
    # print('Cohort build Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    #
    # zz

    # df = pd.read_csv(r'../data/recover/output/pregnancy_output/pregnant_yr4.csv',
    #                  dtype={'patid': str, 'site': str, 'zip': str},
    #                  parse_dates=['index date', 'dob',
    #                               'flag_delivery_date',
    #                               'flag_pregnancy_start_date',
    #                               'flag_pregnancy_end_date'
    #                               ])
    infile = r'../data/recover/output/pregnancy_output_y4/pregnant_yr4.csv'
    # infile = r'../data/recover/output/pregnancy_CX_0501_2025/pregnancy_{}.csv'.format("wcm_pcornet_all")
    print('Loading:', infile)
    df = pd.read_csv(infile,
                     dtype={'patid': str, 'site': str, 'zip': str, 'flag_delivery_code': str},
                     parse_dates=['index date', 'dob',
                                  'flag_delivery_date',
                                  'flag_pregnancy_start_date',
                                  'flag_pregnancy_end_date'
                                  ],
                     # nrows=200000,  # for debug
                     )
    print(df.shape)
    N = len(df)

    print('Loading:', infile, 'Done! Time used:',
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

    print('Before selecting pregnant after +180 days, len(df)\n', len(df))
    n = len(df)
    time_order_flag = (df['index date'] + datetime.timedelta(days=180) <= df['flag_pregnancy_start_date'])
    df_ec = df.loc[time_order_flag, :].copy()
    print('After selecting pregnant after +180 days, len(df),\n',
          '{}\t{:.2f}%\t{:.2f}%'.format(len(df_ec), len(df_ec) / n * 100, len(df_ec) / N * 100))

    pasc_flag = df_ec['any_pasc_flag'].astype('int')
    pasc_t2e_label = 'any_pasc_t2e'

    contrl_label = (df_ec['any_pasc_flag'] == 0) & (df_ec['any_CFR_flag'] == 0) & (df_ec['any_brainfog_flag'] == 0) & \
                   (df_ec['dxMECFS-out@ME/CFS'] == 0) & (df_ec['dx-out@' + 'PASC-General'] == 0)

    # pasc_flag = df_ec['any_CFR_flag'].astype('int')
    # pasc_t2e_label = 'any_CFR_t2e'
    # # # # #
    # pasc_flag = df_ec['any_brainfog_flag'].astype('int')
    # pasc_t2e_label = 'any_brainfog_t2e'
    # # # #
    # pasc_flag = (df_ec['dxMECFS-out@ME/CFS'].copy() >= 1).astype('int')
    # pasc_t2e = 'dxMECFS-t2e@ME/CFS'
    # # #
    # pasc_flag = (df_ec['dx-out@' + 'PASC-General'].copy() >= 1).astype('int')
    # pasc_t2e = 'dx-t2e@' + 'PASC-General'

    sites = [
        'ochin_pcornet_all', 'northwestern_pcornet_all', 'intermountain_pcornet_all', 'mcw_pcornet_all',
        'iowa_pcornet_all',
        'missouri_pcornet_all', 'nebraska_pcornet_all', 'utah_pcornet_all', 'utsw_pcornet_all', 'wcm_pcornet_all',
        'monte_pcornet_all', 'mshs_pcornet_all', 'columbia_pcornet_all', 'nyu_pcornet_all', 'ufh_pcornet_all',
        'emory_pcornet_all', 'nch_pcornet_all', 'pitt_pcornet_all', 'osu_pcornet_all', 'psu_pcornet_all',
        'temple_pcornet_all', 'michigan_pcornet_all', 'stanford_pcornet_all', 'ochsner_pcornet_all',
        'ucsf_pcornet_all', 'lsu_pcornet_all', 'vumc_pcornet_all', 'duke_pcornet_all', 'wakeforest_pcornet_all']

    for site in sites:
        df = df_ec.loc[df_ec['site'] == site]

        df1 = df.loc[(pasc_flag > 0), :]
        # print("Exposed group: ((pasc_flag > 0) ) len(df1)", len(df1))
        df0 = df.loc[contrl_label, :]
        # print("Contrl group:  ((pasc_flag == 0)  len(df0)", len(df0))

        pasc_time = df1[pasc_t2e_label].apply(lambda x: datetime.timedelta(x))
        # delivery_time = df1['flag_delivery_date'] - df1['index date']
        # delivery_time = df1['flag_pregnancy_start_date'] - df1['index date']
        # time_order_flag_1 = pasc_time < delivery_time
        # time_order_flag_1 = (df1['index date'] + datetime.timedelta(days=180) <= df1['flag_pregnancy_start_date'])
        # df1 = df1.loc[time_order_flag_1, :]

        print(site, "\nExposed group: ((pasc_flag > 0) len(df1)", len(df1), 'VS',
              "Contrl group: ((pasc_flag == 0)  len(df0)", len(df0))

        # print('***Preterm birth rate:')
        for col in ['preterm birth<37Andlivebirth',
                    'preterm birth<34Andlivebirth', ]:
            print(col, 'among live birth:', '\t',
                  'Exposed:', (df1[col] == 1).sum(),
                  '{:.2f}%'.format((df1[col] == 1).sum() / (df1['preg_outcome-livebirth'] == 1).sum() * 100), '\t',
                  'Control:', (df0[col] == 1).sum(),
                  '{:.2f}%'.format((df0[col] == 1).sum() / (df0['preg_outcome-livebirth'] == 1).sum() * 100)
                  )

        # print('***Other Outcomes among all pregnant:')
        # for col in ['preg_outcome-livebirth', 'preg_outcome-stillbirth',
        #             'preg_outcome-miscarriage', 'preg_outcome-miscarriage<20week',
        #             'preg_outcome-abortion', 'preg_outcome-abortion<20week',
        #             'preg_outcome-other', ]:
        #     print(col, '\t',
        #           'Exposed:', (df1[col] == 1).sum(), '{:.2f}%'.format((df1[col] == 1).mean() * 100), '\t',
        #           'Control:', (df0[col] == 1).sum(), '{:.2f}%'.format((df0[col] == 1).mean() * 100),
        #           )

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

    zz

    df1['exposed'] = 1
    df0['exposed'] = 0
    df10 = pd.concat([df1, df0], ignore_index=True)
    df10.to_csv(
        r'../data/recover/output/pregnancy_output_y4/pregnant_yr4-{}.csv'.format(pasc_t2e_label))  # pregnancy_output_y4

    zz
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
                                                x.startswith('dxCFR') or
                                                x.startswith('dxMECFS')
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
        elif pasc == 'cvddeath_postacute':
            pasc_flag = df['cvd death postacute'].astype('int')
            pasc_t2e = df['cvd death t2e postacute'].astype('float')
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
        elif pasc == 'ME/CFS':
            pasc_flag = (df['dxMECFS-out@ME/CFS'].copy() >= 1).astype('int')
            pasc_t2e = df['dxMECFS-t2e@ME/CFS'].astype('float')
            pasc_baseline = df['dxMECFS-base@ME/CFS']

        elif pasc in brainfog_list:
            pasc_flag = (df['dxbrainfog-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['dxbrainfog-t2e@' + pasc].astype('float')
            pasc_baseline = df['dxbrainfog-base@' + pasc]
        elif pasc == 'any_brainfog':
            pasc_flag = df['any_brainfog_flag'].astype('int')
            pasc_t2e = df['any_brainfog_t2e'].astype('float')
            pasc_baseline = df['any_brainfog_baseline']
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
        elif pasc == 'cvddeath_postacute':
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
        out_file_balance = r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcovV3/{}-{}-results.csv'.format(
            args.cohorttype,
            args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
            args.exptype,  # '-select' if args.selectpasc else '',
            i, _clean_name_(pasc))

        utils.check_and_mkdir(out_file_balance)
        model.results.to_csv(out_file_balance)  # args.save_model_filename +

        df_summary = summary_covariate(covs_array, covid_label, iptw, smd, smd_weighted, before, after)
        df_summary.to_csv(
            '../data/recover/output/results/SSRI-{}-{}-{}-mentalcovV3/{}-{}-evaluation_balance.csv'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                i, _clean_name_(pasc)))

        dfps = pd.DataFrame({'ps': ps, 'iptw': iptw, 'Exposure': covid_label})

        dfps.to_csv(
            '../data/recover/output/results/SSRI-{}-{}-{}-mentalcovV3/{}-{}-evaluation_ps-iptw.csv'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                i, _clean_name_(pasc)))
        try:
            figout = r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcovV3/{}-{}-PS.png'.format(
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
            fig_outfile=r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcovV3/{}-{}-km.png'.format(
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
                    r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcovV3/causal_effects_specific-snapshot-{}.csv'.format(
                        args.cohorttype,
                        args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                        args.exptype,  # '-select' if args.selectpasc else '',
                        i))
        except:
            print('Error in ', i, pasc)
            df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

            df_causal.to_csv(
                r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcovV3/causal_effects_specific-ERRORSAVE.csv'.format(
                    args.cohorttype,
                    args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                    args.exptype,  # '-select' if args.selectpasc else '',
                ))

        print('done one pasc, time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

    df_causal.to_csv(
        r'../data/recover/output/results/SSRI-{}-{}-{}-mentalcovV3/causal_effects_specific.csv'.format(
            args.cohorttype,
            args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
            args.exptype,  # '-select' if args.selectpasc else '',
        ))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
