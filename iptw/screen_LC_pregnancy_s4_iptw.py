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
from scipy.stats.contingency import odds_ratio
import statsmodels.api as sm

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
    parser.add_argument('--negative_ratio', type=int, default=99)  # 5
    parser.add_argument('--selectpasc', action='store_true')
    parser.add_argument('--build_data', action='store_true')
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--adjustless', action='store_true')

    parser.add_argument('--cohorttype',  #
                        choices=['pregafter30', 'pregafter180',
                                 ],
                        default='pregafter180')
    parser.add_argument('--exptype',
                        choices=['anypasc', 'anyCFR', 'mecfs', 'brainfog', 'U099',
                                 ], default='anypasc')  # 'base180-0'

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


def add_col(df):
    # re-organize Paxlovid risk factors
    # see https://www.cdc.gov/coronavirus/2019-ncov/need-extra-precautions/people-with-medical-conditions.html
    # and https://www.paxlovid.com/who-can-take
    print('in add_col, df.shape', df.shape)

    print('Build covs for outpatient cohorts w/o ICU or ventilation')
    df['inpatient'] = ((df['hospitalized'] == 1) & (df['ventilation'] == 0) & (df['criticalcare'] == 0)).astype('int')
    df['icu'] = (((df['hospitalized'] == 1) & (df['ventilation'] == 1)) | (df['criticalcare'] == 1)).astype('int')
    df['inpatienticu'] = ((df['hospitalized'] == 1) | (df['criticalcare'] == 1) | (df['ventilation'] == 1)).astype(
        'int')
    df['outpatient'] = ((df['hospitalized'] == 0) & (df['criticalcare'] == 0) & (df['ventilation'] == 0)).astype('int')

    df["Type 1 or 2 Diabetes Diagnosis"] = (
            ((df["DX: Diabetes Type 1"] >= 1).astype('int') + (df["DX: Diabetes Type 2"] >= 1).astype('int')) >= 1
    ).astype('int')

    df['PaxRisk:Cancer'] = (
            ((df["DX: Cancer"] >= 1).astype('int') +
             (df['CCI:Cancer'] >= 1).astype('int') +
             (df['CCI:Metastatic Carcinoma'] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Chronic kidney disease'] = (df["DX: Chronic Kidney Disease"] >= 1).astype('int')

    df['PaxRisk:Chronic liver disease'] = (
            ((df["DX: Cirrhosis"] >= 1).astype('int') +
             (df['CCI:Mild Liver Disease'] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Chronic lung disease'] = (
            ((df['CCI:Chronic Pulmonary Disease'] >= 1).astype('int') +
             (df["DX: Asthma"] >= 1).astype('int') +
             (df["DX: Chronic Pulmonary Disorders"] >= 1).astype('int') +
             (df["DX: COPD"] >= 1).astype('int') +
             (df["DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)"] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Cystic fibrosis'] = (df['DX: Cystic Fibrosis'] >= 1).astype('int')

    df['PaxRisk:Dementia or other neurological conditions'] = (
            ((df['DX: Dementia'] >= 1).astype('int') +
             (df['CCI:Dementia'] >= 1).astype('int') +
             (df["DX: Parkinson's Disease"] >= 1).astype('int') +
             (df["DX: Multiple Sclerosis"] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Diabetes'] = (
            ((df["DX: Diabetes Type 1"] >= 1).astype('int') +
             (df["DX: Diabetes Type 2"] >= 1).astype('int') +
             (df['CCI:Diabetes without complications'] >= 1).astype('int') +
             (df['CCI:Diabetes with complications'] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Disabilities'] = (
            ((df['CCI:Paraplegia and Hemiplegia'] >= 1).astype('int') +
             (df["DX: Down's Syndrome"] >= 1).astype('int') +
             (df["DX: Hemiplegia"] >= 1).astype('int') +
             (df["DX: Autism"] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Heart conditions'] = (
            ((df["DX: Congestive Heart Failure"] >= 1).astype('int') +
             (df["DX: Coronary Artery Disease"] >= 1).astype('int') +
             (df["DX: Arrythmia"] >= 1).astype('int') +
             (df['CCI:Myocardial Infarction'] >= 1).astype('int') +
             (df['CCI:Congestive Heart Failure'] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Hypertension'] = (df["DX: Hypertension"] >= 1).astype('int')

    df['PaxRisk:HIV infection'] = (
            ((df["DX: HIV"] >= 1).astype('int') +
             (df['CCI:AIDS/HIV'] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Immunocompromised condition or weakened immune system'] = (
            ((df['DX: Inflammatory Bowel Disorder'] >= 1).astype('int') +
             (df['DX: Lupus or Systemic Lupus Erythematosus'] >= 1).astype('int') +
             (df['DX: Rheumatoid Arthritis'] >= 1).astype('int') +
             (df["MEDICATION: Corticosteroids"] >= 1).astype('int') +
             (df["MEDICATION: Immunosuppressant drug"] >= 1).astype('int') +
             (df["CCI:Connective Tissue Disease-Rheumatic Disease"] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Mental health conditions'] = (df["DX: Mental Health Disorders"] >= 1).astype('int')

    df["PaxRisk:Overweight and obesity"] = (
            (df["DX: Severe Obesity  (BMI>=40 kg/m2)"] >= 1) | (df['bmi'] >= 25) | (df['addPaxRisk:Obesity'] >= 1)
    ).astype('int')

    df["PaxRisk:Obesity"] = (
            (df["DX: Severe Obesity  (BMI>=40 kg/m2)"] >= 1) | (df['bmi'] >= 30) | (df['addPaxRisk:Obesity'] >= 1)
    ).astype('int')

    # physical activity
    # not captured

    # pregnancy, use infection during pregnant, label from pregnant cohorts
    print('comments out pregnancy paxrisk, add back later!!!')
    df["PaxRisk:Pregnancy"] = ((df['flag_pregnancy'] == 1) &
                               (df['index date'] >= df['flag_pregnancy_start_date'] - datetime.timedelta(days=7)) &
                               (df['index date'] <= df['flag_delivery_date'] + datetime.timedelta(days=7))).astype(
        'int')

    df['PaxRisk:Sickle cell disease or thalassemia'] = (
            ((df['DX: Sickle Cell'] >= 1).astype('int') +
             (df["DX: Anemia"] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Smoking current'] = ((df['Smoker: current'] >= 1) | (df['Smoker: former'] >= 1)).astype('int')
    df['PaxRisk:Smoking-Tobacco'] = ((df['Smoker: current'] >= 1) | (df['Smoker: former'] >= 1)).astype('int')
    df['PaxRisk:Smoking-Tobacco-Disorder'] = ((df['Smoker: current'] >= 1) | (df['Smoker: former'] >= 1) | (
                df['dx-base@Tobacco-related disorders'] >= 1)).astype('int')

    # cell transplant, --> autoimmu category?

    df['PaxRisk:Stroke or cerebrovascular disease'] = (df['CCI:Cerebrovascular Disease'] >= 1).astype('int')

    df['PaxRisk:Substance use disorders'] = (
            ((df["DX: Alcohol Abuse"] >= 1).astype('int') +
             (df['DX: Other Substance Abuse'] >= 1).astype('int') +
             (df['addPaxRisk:Drug Abuse'] >= 1).astype('int')
             ) >= 1
    ).astype('int')

    df['PaxRisk:Tuberculosis'] = (df['addPaxRisk:tuberculosis'] >= 1).astype('int')

    # PAXLOVID is not recommended for people with severe kidney disease
    # PAXLOVID is not recommended for people with severe liver disease
    df['PaxExclude:liver'] = (df['CCI:Moderate or Severe Liver Disease'] >= 1).astype('int')
    df['PaxExclude:end-stage kidney disease'] = (df["DX: End Stage Renal Disease on Dialysis"] >= 1).astype('int')

    pax_risk_cols = [x for x in df.columns if x.startswith('PaxRisk:')]
    print('pax_risk_cols:', len(pax_risk_cols), pax_risk_cols)
    df['PaxRisk-Count'] = df[pax_risk_cols].sum(axis=1)

    pax_exclude_cols = [x for x in df.columns if x.startswith('PaxExclude:')]
    print('pax_exclude_cols:', len(pax_exclude_cols), pax_exclude_cols)
    df['PaxExclude-Count'] = df[pax_exclude_cols].sum(axis=1)
    # Tuberculosis, not captured, need to add columns?

    # 2024-09-07 add ssri indication covs
    mental_cov = ['mental-base@Schizophrenia Spectrum and Other Psychotic Disorders',
                  'mental-base@Depressive Disorders',
                  'mental-base@Bipolar and Related Disorders',
                  'mental-base@Anxiety Disorders',
                  'mental-base@Obsessive-Compulsive and Related Disorders',
                  'mental-base@Post-traumatic stress disorder',
                  'mental-base@Bulimia nervosa',
                  'mental-base@Binge eating disorder',
                  'mental-base@premature ejaculation',
                  'mental-base@Autism spectrum disorder',
                  'mental-base@Premenstrual dysphoric disorder', ]

    df['SSRI-Indication-dsm-cnt'] = df[mental_cov].sum(axis=1)
    df['SSRI-Indication-dsm-flag'] = (df['SSRI-Indication-dsm-cnt'] > 0).astype('int')

    df['SSRI-Indication-dsmAndExlix-cnt'] = df[mental_cov + ['mental-base@SMI', 'mental-base@non-SMI', ]].sum(axis=1)
    df['SSRI-Indication-dsmAndExlix-flag'] = (df['SSRI-Indication-dsmAndExlix-cnt'] > 0).astype('int')

    # ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    df['cci_quan:0'] = 0
    df['cci_quan:1-2'] = 0
    df['cci_quan:3-4'] = 0
    df['cci_quan:5-10'] = 0
    df['cci_quan:11+'] = 0

    df['cci_quan:3+'] = 0

    # ['age18-24', 'age25-34', 'age35-49', 'age50-64', 'age65+']
    # df['age@18-<25'] = 0
    # df['age@25-<30'] = 0
    # df['age@30-<35'] = 0
    # df['age@35-<40'] = 0
    # df['age@40-<45'] = 0
    # df['age@45-<50'] = 0
    # df['age@50-<55'] = 0
    # df['age@55-<60'] = 0
    # df['age@60-<65'] = 0
    # df['age@65-<60'] = 0

    df['age@18-24'] = 0
    df['age@25-34'] = 0
    df['age@35-49'] = 0
    df['age@50-64'] = 0
    df['age@65+'] = 0

    # ['RE:Asian Non-Hispanic', 'RE:Black or African American Non-Hispanic', 'RE:Hispanic or Latino Any Race',
    # 'RE:White Non-Hispanic', 'RE:Other Non-Hispanic', 'RE:Unknown']
    df['RE:Asian Non-Hispanic'] = 0
    df['RE:Black or African American Non-Hispanic'] = 0
    df['RE:Hispanic or Latino Any Race'] = 0
    df['RE:White Non-Hispanic'] = 0
    df['RE:Other Non-Hispanic'] = 0
    df['RE:Unknown'] = 0

    # ['No. of Visits:0', 'No. of Visits:1-3', 'No. of Visits:4-9', 'No. of Visits:10-19', 'No. of Visits:>=20',
    # 'No. of hospitalizations:0', 'No. of hospitalizations:1', 'No. of hospitalizations:>=1']
    df['No. of Visits:0'] = 0
    df['No. of Visits:1-3'] = 0
    df['No. of Visits:4-9'] = 0
    df['No. of Visits:10-19'] = 0
    df['No. of Visits:>=20'] = 0

    df['No. of hospitalizations:0'] = 0
    df['No. of hospitalizations:1'] = 0
    df['No. of hospitalizations:>=1'] = 0

    df['quart:01/22-03/22'] = 0
    df['quart:04/22-06/22'] = 0
    df['quart:07/22-09/22'] = 0
    df['quart:10/22-1/23'] = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 'index date', 'flag_delivery_date', 'flag_pregnancy_start_date', 'flag_pregnancy_end_date'
        index_date = pd.to_datetime(row['index date'])
        if index_date < datetime.datetime(2022, 4, 1, 0, 0):
            df.loc[index, 'quart:01/22-03/22'] = 1
        elif index_date < datetime.datetime(2022, 7, 1, 0, 0):
            df.loc[index, 'quart:04/22-06/22'] = 1
        elif index_date < datetime.datetime(2022, 10, 1, 0, 0):
            df.loc[index, 'quart:07/22-09/22'] = 1
        elif index_date <= datetime.datetime(2023, 2, 1, 0, 0):
            df.loc[index, 'quart:10/22-1/23'] = 1

        age = row['age']
        if pd.notna(age):
            if age < 25:
                df.loc[index, 'age@18-24'] = 1
            elif age < 35:
                df.loc[index, 'age@25-34'] = 1
            elif age < 50:
                df.loc[index, 'age@35-49'] = 1
            elif age < 65:
                df.loc[index, 'age@50-64'] = 1
            elif age >= 65:
                df.loc[index, 'age@65+'] = 1

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

        if row['score_cci_quan'] >= 3:
            df.loc[index, 'cci_quan:3+'] = 1

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

    # pre-process data a little bit
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
                      x.startswith('pregnancyout2nd-out@') or
                      x.startswith('dxCFR-out@') or
                      x.startswith('mental-base@') or
                      x.startswith('dxMECFS-base@') or
                      x.startswith('dxCVDdeath-base@') or
                      x.startswith('dxcovCNSLDN-base') or
                      x.startswith('dxcovNaltrexone-basedrugonset@') or
                      x.startswith('dxcovNaltrexone-base@') or
                      x.startswith('covNaltrexone_med-basedrugonset@') or
                      x.startswith('covNaltrexone_med-base@')
                      )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')
    print('Finish add_col, df.shape', df.shape)
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
    s = s.replace(':', '-').replace('/', '-').replace('@', '-').replace('<', 'Less')
    s_trunc = (s[:maxlen] + '..') if len(s) > maxlen else s
    return s_trunc


def more_ec_for_cohort_selection_new_order(df, cohorttype):
    print('in more_ec_for_cohort_selection, df.shape', df.shape)
    print('Applying more specific/flexible eligibility criteria for cohort selection')

    # select index date
    # print('Before selecting index date from 2022-3-1 to 2023-2-1, len(df)', len(df))
    # df = df.loc[
    #      (df['index date'] >= datetime.datetime(2020, 3, 1, 0, 0)) &
    #      (df['index date'] <= datetime.datetime(2023, 2, 1, 0, 0)), :]
    # # print('After selecting index date from 2022-1-1 to 2023-2-1, len(df)', len(df))
    # print('After selecting index date from 2022-3-1 to 2023-2-1, len(df)', len(df))

    df_ec_start = df.copy()
    print('Build df_ec_start for calculating EC proportion, len(df_ec_start)', len(df_ec_start))

    # Exclusion, no hospitalized
    # print('Before selecting no hospitalized, len(df)', len(df))
    df = df.loc[(df['outpatient'] == 1), :]
    print('After selecting no hospitalized, len(df)', len(df),
          'exclude not outpatient in df_ec_start', (df_ec_start['outpatient'] != 1).sum())
    print('treated:', (df['treated'] == 1).sum(), 'untreated:', (df['treated'] == 0).sum())

    def ec_no_U099_baseline(_df):
        print('before ec_no_U099_baseline, _df.shape', _df.shape)
        print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

        n0 = len(_df)
        _df = _df.loc[(_df['dx-base@PASC-General'] == 0)]
        n1 = len(_df)
        print('after ec_no_U099_baseline, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

        # print('exclude baseline U099 in df_ec_start', (df_ec_start['dx-base@PASC-General'] > 0).sum())
        return _df

    # ADHD_before_drug_onset
    def ec_ADHD_baseline(_df):
        print('before ec_ADHD_baseline, _df.shape', _df.shape)
        print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

        n0 = len(_df)
        _df = _df.loc[(_df['ADHD_before_drug_onset'] == 1)]
        n1 = len(_df)
        print('after ec_ADHD_baseline, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

        # print('include baseline ADHD in df_ec_start', (df_ec_start['ADHD_before_drug_onset'] == 0).sum())
        return _df

    def ec_no_pregnant_baseline(_df):
        print('before ec_no_pregnant_baseline, _df.shape', _df.shape)
        print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

        n0 = len(_df)
        _df = _df.loc[(_df['PaxRisk:Pregnancy'] == 0)]
        n1 = len(_df)
        print('after ec_no_pregnant_baseline, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

        # print('include baseline ADHD in df_ec_start', (df_ec_start['ADHD_before_drug_onset'] == 0).sum())
        return _df

    def ec_no_HIV_baseline(_df):
        print('before ec_no_HIV_baseline, _df.shape', _df.shape)
        print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

        n0 = len(_df)
        _df = _df.loc[(_df['PaxRisk:HIV infection'] == 0)]
        n1 = len(_df)
        print('after ec_no_HIV_baseline, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

        # print('include baseline ADHD in df_ec_start', (df_ec_start['ADHD_before_drug_onset'] == 0).sum())
        return _df

    def ec_no_other_covid_treatment(_df):
        print('before ec_no_other_covid_treatment, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[(~(_df['treat-t2e@remdesivir'] <= 14)) &
                      (_df['Remdesivir'] == 0) &
                      (_df['Molnupiravir'] == 0) &
                      (_df[
                           'Any Monoclonal Antibody Treatment (Bamlanivimab, Bamlanivimab and Etesevimab, Casirivimab and Imdevimab, Sotrovimab, and unspecified monoclonal antibodies)'] == 0) &
                      (_df['PX: Convalescent Plasma'] == 0) &
                      (_df['pax_contra'] == 0)]
        n1 = len(_df)
        print('after ec_no_other_covid_treatment, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))

        print('exclude acute covid-19 treatment in df_ec_start -14 - +14', (
                (df_ec_start['treat-t2e@remdesivir'] <= 14) |
                (df_ec_start['Remdesivir'] > 0) |
                (df_ec_start['Molnupiravir'] > 0) |
                (df_ec_start[
                     'Any Monoclonal Antibody Treatment (Bamlanivimab, Bamlanivimab and Etesevimab, Casirivimab and Imdevimab, Sotrovimab, and unspecified monoclonal antibodies)'] > 0) |
                (df_ec_start['PX: Convalescent Plasma'] > 0)).sum())

        # print('exclude contraindication drugs in df_ec_start -14 - +14', (df_ec_start['pax_contra'] > 0).sum())
        return _df

    def ec_no_severe_conditions_4_pax(_df):
        print('before ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
        print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

        n0 = len(_df)
        _df = _df.loc[(_df['PaxExclude-Count'] == 0)]
        print('after ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
        n1 = len(_df)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        print('treated:', (_df['treated'] == 1).sum(), 'untreated:', (_df['treated'] == 0).sum())

        # print('exclude severe conditions in df_ec_start', (df_ec_start['PaxExclude-Count'] > 0).sum())
        return _df

    def ec_at_least_one_risk_4_pax(_df):
        print('before ec_at_least_one_risk_4_pax, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[(_df['age'] >= 50) | (_df['PaxRisk-Count'] > 0)]
        n1 = len(_df)
        print('after ec_at_least_one_risk_4_pax, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        return _df

    def ec_not_at_risk_4_pax(_df):
        print('before ec_not_at_risk_4_pax, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[~((_df['age'] >= 50) | (_df['PaxRisk-Count'] > 0))]
        n1 = len(_df)
        print('after ec_not_at_risk_4_pax, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        return _df

    df = ec_no_U099_baseline(df)
    df = ec_no_severe_conditions_4_pax(df)
    df = ec_no_pregnant_baseline(df)
    df = ec_no_HIV_baseline(df)

    if cohorttype == 'baseADHD':
        print('default: require baseline ADHD diagnosis:')
        df = ec_ADHD_baseline(df)

    ## df = ec_no_other_covid_treatment(df)
    ## df_risk = ec_at_least_one_risk_4_pax(df)
    ## df_norisk = ec_not_at_risk_4_pax(df)

    return df


if __name__ == "__main__":
    # python screen_naltrexone_iptw_pcornet.py  --cohorttype matchK10replace  2>&1 | tee  log_recover/screen_naltrexone_iptw_pcornet-matchK10replace.txt
    # python screen_naltrexone_iptw_pcornet.py  --cohorttype matchK5replace  2>&1 | tee  log_recover/screen_naltrexone_iptw_pcornet-matchK5replace.txt
    # python screen_LC_pregnancy_s4_iptw.py 2>&1 | tee  preg_output/screen_LC_pregnancy_s4_iptw-pregafter180-anypasc.txt
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    # **************
    # in_file_infectt0 = r'./cns_output/Matrix-cns-adhd-CNS-ADHD-acuteIncident-0-30-25Q2-v3.csv'
    in_file = r'../data/recover/output/pregnancy_output_y4/pregnant_yr4_pergnantOnsetGEinfect30days-updateAtPregOnset.csv'
    in_file_og = r'../data/recover/output/pregnancy_output_y4/pregnant_yr4_pergnantOnsetGEinfect30days.csv'

    print('infile update at pregnant onset:', in_file)
    print('infile at covid infection:', in_file_og)
    print('cohorttype:', )

    df = pd.read_csv(in_file,
                     dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'dob',
                                  'index_date_pregnant_onset',
                                  'index_date_delivery',
                                  'index_date_pregnant_end'
                                  ])

    df_og = pd.read_csv(in_file_og,
                        dtype={'patid': str, 'site': str, 'zip': str},
                        parse_dates=['index date', 'dob',
                                     'flag_delivery_date',
                                     'flag_pregnancy_start_date',
                                     'flag_pregnancy_end_date'
                                     ])

    print('df.shape:', df.shape)
    print('df_og.shape:', df_og.shape)

    df = pd.merge(df, df_og, how='left', left_on=['patid', 'site'], right_on=['patid', 'site'], suffixes=('', '_og'), )
    print('After left merge, merged df.shape:', df.shape)

    print('Additional feature preprocessing for adjust and outcomes...')
    print('...Part1:  add_col: ing')
    df = add_col(df)
    print('... add_col done!')

    # deal with pregnancy related cov and outcomes
    print('...Part2:  feature_process_pregnancy: ing')
    df = feature_process_pregnancy(df)
    print('... feature_process_pregnancy done!')

    ## to do
    ## 1. cov, outcome, and expsoure columns
    ## 2. double check, simple table,
    ## 3. adjust analysis and print results
    ## 4. print final table on cov, outcome, etc.

    # df.to_csv(r'../data/recover/output/pregnancy_output_y4/pregnant_yr4_pergnantOnsetGEinfect30days-updateAtPregOnset-mergeOG.csv')

    print('Before selecting cohortype, ', args.cohorttype, len(df))
    N = len(df)

    ## Step-1: Build exposed groups
    # pregnancy onset after covid infection selection, primary 180, use this for other sensitivity analysis
    if args.cohorttype == 'pregafter30':
        pass
    elif args.cohorttype == 'pregafter180':
        n = len(df)
        time_order_flag = (df['index date'] + datetime.timedelta(days=180) <= df['flag_pregnancy_start_date'])
        df = df.loc[time_order_flag, :]
        print('After selecting pregnant >= +180 days, len(df),\n',
              '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))
    else:
        raise ValueError

    # label target exposure of interest
    if args.exptype == 'anypasc':
        pasc_flag = df['any_pasc_flag'].astype('int')
        pasc_t2e_label = 'any_pasc_t2e'
    else:
        raise ValueError

    contrl_label = (df['any_pasc_flag'] == 0) & (df['any_CFR_flag'] == 0) & (df['any_brainfog_flag'] == 0) & \
                   (df['dxMECFS-out@ME/CFS'] == 0) & (df['dx-out@' + 'PASC-General'] == 0)

    df_pos = df.loc[(pasc_flag > 0), :]
    df_pos['exposed'] = 1
    print("Exposed group: ((pasc_flag > 0) ) len(df_pos)", len(df_pos))

    df_neg = df.loc[contrl_label, :]
    df_neg['exposed'] = 0
    print("V2-Contrl group, no all LC:  contrl_label  len(df_neg)", len(df_neg))

    df = pd.concat([df_pos.reset_index(), df_neg.reset_index()], ignore_index=True)
    print('After selected exposed groul, len(df)', len(df),
          'exposed group:', len(df_pos),
          'contrl group:', len(df_neg))

    print('After selecting cohortype, ', len(df))
    case_label = 'Exposed'
    ctrl_label = 'Ctrl'

    print('Before select_subpopulation, len(df)', args.cohorttype, len(df))
    df = select_subpopulation(df, args.severity)
    print('After select_subpopulation, len(df)', len(df))

    print('After building exposure groups:\n',
          'args.cohorttype:', args.cohorttype, 'args.exptype:', args.exptype, 'args.severity', args.severity,
          'len(df)', len(df), 'df.shape', df.shape,
          'exposed group:', (df['exposed'] == 1).sum(), 'contrl group:', (df['exposed'] == 0).sum())
    print('%: {}/{}\t{:.2f}%'.format(len(df), N, len(df) / N * 100))

    out_file_for_table = r'../data/recover/output/pregnancy_output_y4/pregnant_yr4_exposureBuilt-{}-{}-{}.csv'.format(
        args.cohorttype, args.severity, args.exptype)
    df.to_csv(out_file_for_table, index=False)
    print('Dump {} done!'.format(out_file_for_table))
    print('Done load data! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    # **************

    # print('Read dump file for debug:')
    # df = pd.read_csv(r'../data/recover/output/pregnancy_output_y4/pregnant_yr4_exposureBuilt-pregafter180-all-anypasc.csv',
    #                  dtype={'patid': str, 'site': str, 'zip': str},
    #                  parse_dates=['index date', 'dob',
    #                               'flag_delivery_date',
    #                               'flag_pregnancy_start_date',
    #                               'flag_pregnancy_end_date'
    #                               ],
    #                  )
    print('df.shape:', df.shape)

    # Step-2: Select Cov columns
    # zz
    ### 2025-09-11
    ### remove death and pasc label portion here, already labeled in step 1. refer to code in other iptw file
    ### data clean for <0 error death records, and add censoring to the death time to event columns
    ### End of defining ANY *** conditions
    ### pd.Series(df.columns).to_csv('recover_covid_pos-with-pax-V3-column-name.csv')

    col_names = pd.Series(df.columns)
    df_info = df[['patid', 'site', 'index date', 'exposed',
                  'hospitalized',
                  'ventilation', 'criticalcare', 'maxfollowup', 'death', 'death t2e',
                  '03/20-06/20', '07/20-10/20', '11/20-02/21',
                  '03/21-06/21', '07/21-10/21', '11/21-02/22',
                  '03/22-06/22', '07/22-10/22', '11/22-02/23',
                  '03/23-06/23', '07/23-10/23', '11/23-02/24',
                  '03/24-06/24', '07/24-10/24',
                  ]]
    df_label = (df['exposed'] >= 1).astype('int')

    # # how to deal with death?
    # df_outcome_cols = ['death', 'death t2e'] + [x for x in
    #                                             list(df.columns)
    #                                             if x.startswith('dx') or
    #                                             x.startswith('smm') or
    #                                             x.startswith('any_pasc') or
    #                                             x.startswith('dxadd') or
    #                                             x.startswith('dxbrainfog') or
    #                                             x.startswith('dxCFR') or
    #                                             x.startswith('dxMECFS') or
    #                                             x.startswith('PaxRisk:') or
    #                                             x.startswith('dxcovCNSLDN-base@')
    #                                             ]
    #
    # df_outcome = df.loc[:, df_outcome_cols]  # .astype('float')

    covs_columns = (['pregage:18-<25 years', 'pregage:25-<30 years', 'pregage:30-<35 years',
                     'pregage:35-<40 years', 'pregage:40-<45 years', 'pregage:45-50 years',
                     'RE:Asian Non-Hispanic',
                     'RE:Black or African American Non-Hispanic',
                     'RE:Hispanic or Latino Any Race', 'RE:White Non-Hispanic',
                     'RE:Other Non-Hispanic', 'RE:Unknown',
                     'BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ',
                     'BMI: >=30 obese ', 'BMI: missing',
                     ] +
                    [
                        'PaxRisk:Cancer',
                        'PaxRisk:Chronic kidney disease',
                        'PaxRisk:Chronic liver disease',
                        'PaxRisk:Chronic lung disease',
                        # 'PaxRisk:Cystic fibrosis',
                        # 'PaxRisk:Dementia or other neurological conditions',
                        'PaxRisk:Diabetes',
                        # 'PaxRisk:Disabilities',
                        'PaxRisk:Heart conditions',
                        'PaxRisk:Hypertension',
                        'PaxRisk:HIV infection',
                        'PaxRisk:Immunocompromised condition or weakened immune system',
                        # 'PaxRisk:Mental health conditions', # use obc mental part
                        # 'PaxRisk:Overweight and obesity',
                        'PaxRisk:Obesity',
                        # 'PaxRisk:Pregnancy',
                        'PaxRisk:Sickle cell disease or thalassemia',
                        'PaxRisk:Smoking current',
                        'PaxRisk:Stroke or cerebrovascular disease',
                        'PaxRisk:Substance use disorders',  # use this one, more than obc: substance one
                        'PaxRisk:Tuberculosis'] +
                    [
                        # 'obc:Placenta accreta spectrum',
                        'obc:Pulmonary hypertension',
                        'obc:Chronic renal disease',
                        'obc:Cardiac disease, preexisting',
                        # 'obc:HIV/AIDS',
                        # 'obc:Preeclampsia with severe features',
                        # 'obc:Placental abruption',
                        'obc:Bleeding disorder, preexisting',
                        'obc:Anemia, preexisting',
                        # 'obc:Twin/multiple pregnancy',
                        'obc:Preterm birth (< 37 weeks)',
                        # 'obc:Placenta previa, complete or partial',
                        'obc:Neuromuscular disease',
                        'obc:Asthma, acute or moderate/severe',
                        'obc:Preeclampsia without severe features or gestational hypertension',
                        # 'obc:Connective tissue or autoimmune disease',
                        'obc:Uterine fibroids',
                        # 'obc:Substance use disorder',
                        'obc:Gastrointestinal disease',
                        'obc:Chronic hypertension',
                        'obc:Major mental health disorder',
                        # 'obc:Preexisting diabetes mellitus',
                        'obc:Thyrotoxicosis',
                        'obc:Previous cesarean birth',
                        # 'obc:Gestational diabetes mellitus',
                        # 'obc:Delivery BMI\xa0>\xa040'
                    ])

    if args.adjustless:
        covs_columns = [
            'pregage:18-<25 years', 'pregage:25-<30 years', 'pregage:30-<35 years',
            'pregage:35-<40 years', 'pregage:40-<45 years', 'pregage:45-50 years',
            'RE:Asian Non-Hispanic',
            'RE:Black or African American Non-Hispanic',
            'RE:Hispanic or Latino Any Race', 'RE:White Non-Hispanic',
            'RE:Other Non-Hispanic', 'RE:Unknown',
            'BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ',
            'BMI: >=30 obese ', 'BMI: missing',
            'PaxRisk:Diabetes',
            'PaxRisk:Obesity',
            'PaxRisk:Chronic kidney disease',
            'PaxRisk:Hypertension',
            'PaxRisk:Immunocompromised condition or weakened immune system',
            'PaxRisk:Smoking current',
            'PaxRisk:Substance use disorders',
            # diabites
            #
        ]

    print('len(covs_columns):', len(covs_columns), covs_columns)
    print('Still lack time between infect and preg onset, setup later')
    #
    # if args.cohorttype == 'overall':
    #     covs_columns += ['ADHD_before_drug_onset', ]

    print('len(covs_columns):', len(covs_columns), covs_columns)
    df_covs = df.loc[:, covs_columns].astype('float')
    print('df.shape:', df.shape, 'df_covs.shape:', df_covs.shape)

    print('all',
          'df.shape', df.shape,
          'df_info.shape:', df_info.shape,
          'df_label.shape:', df_label.shape,
          'df_covs.shape:', df_covs.shape)

    # Step-3, Setup outcomes of interest
    # Load index information
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

    # selected_screen_list = (['any_pasc', 'PASC-General', 'ME/CFS',
    #                          'death', 'death_acute', 'death_postacute', 'cvddeath_postacute',
    #                          'any_CFR', 'any_brainfog',
    #                          'hospitalization_acute', 'hospitalization_postacute'] +
    #                         CFR_list +
    #                         pasc_list +
    #                         addedPASC_list +
    #                         brainfog_list)

    # icd_OBC = utils.load(r'../data/mapping/icd_OBComorbidity_mapping.pkl')
    # OBC_encoding = utils.load(r'../data/mapping/OBComorbidity_index_mapping.pkl')
    # OBC_outcome_list = list(OBC_encoding.keys())

    icd_SMMpasc = utils.load(r'../data/mapping/icd_SMMpasc_mapping.pkl')
    SMMpasc_encoding = utils.load(r'../data/mapping/SMMpasc_index_mapping.pkl')

    icd_pregnancyout2nd = utils.load(r'../data/mapping/icd_pregnancyout2nd_mapping.pkl')
    pregnancyout2nd_encoding = utils.load(r'../data/mapping/pregnancyout2nd_index_mapping.pkl')

    # outcome_smm_column_names = ['smm-out@' + x for x in SMMpasc_encoding.keys()] + \
    #                            ['smm-t2e@' + x for x in SMMpasc_encoding.keys()] + \
    #                            ['smm-base@' + x for x in SMMpasc_encoding.keys()] + \
    #                            ['smm-t2eall@' + x for x in SMMpasc_encoding.keys()]
    # ['smm-out@smm:Acute myocardial infarction', 'smm-out@smm:Aneurysm', 'smm-out@smm:Acute renal failure',
    # 'smm-out@smm:Adult respiratory distress syndrome', 'smm-out@smm:Amniotic fluid embolism',
    # 'smm-out@smm:Cardiac arrest/ventricular fibrillation', 'smm-out@smm:Conversion of cardiac rhythm',
    # 'smm-out@smm:Disseminated intravascular coagulation', 'smm-out@smm:Eclampsia',
    # 'smm-out@smm:Heart failure/arrest during surgery or procedure',
    # 'smm-out@smm:Puerperal cerebrovascular disorders', 'smm-out@smm:Pulmonary edema / Acute heart failure',
    # 'smm-out@smm:Severe anesthesia complications', 'smm-out@smm:Sepsis', 'smm-out@smm:Shock',
    # 'smm-out@smm:Sickle cell disease with crisis', 'smm-out@smm:Air and thrombotic embolism',
    # 'smm-out@smm:Blood products transfusion', 'smm-out@smm:Hysterectomy', 'smm-out@smm:Temporary tracheostomy',
    # 'smm-out@smm:Ventilation']
    SMM_outcome_list = ['smm-out@' + x for x in SMMpasc_encoding.keys()]
    df['SMM-overall-Any'] = (df[SMM_outcome_list].sum(axis=1) > 0).astype('int')

    pregnancyout2nd_list = ['pregnancyout2nd-out@' + x for x in pregnancyout2nd_encoding.keys()]

    selected_screen_list = ['preterm birth<37Andlivebirth', 'preterm birth<34Andlivebirth',
                            'preg_outcome-livebirth', 'preg_outcome-stillbirth',
                            'preg_outcome-miscarriage', 'preg_outcome-miscarriage<20week',
                            'preg_outcome-abortion', 'preg_outcome-abortion<20week',
                            'preg_outcome-other', 'SMM-overall-Any'
                            ]  + pregnancyout2nd_list + SMM_outcome_list

    # Step-4, Adjusted analysis and dump results
    causal_results = []
    results_columns_name = []

    # save the original df for later use
    df_og = df.copy()

    for i, outcome_of_interest in tqdm(enumerate(selected_screen_list, start=1), total=len(selected_screen_list)):
        """
        # outcome setup for pregnant outcome: exposure, cov, and outcome
        # Binary during 1 year after pregnancy onset
        # perterm birth is defined among live birth
        # others can be evaluated among all
        """
        print('\n In screening:', i, outcome_of_interest)

        if ((outcome_of_interest == 'preterm birth<37Andlivebirth') or
                (outcome_of_interest == 'preterm birth<34Andlivebirth')):
            # select live birth cohort only for calculating
            # select outcome flag
            df = df_og.loc[df_og['preg_outcome-livebirth'] == 1]
        else:
            df = df_og

        outcome_of_interest_flag = (df[outcome_of_interest] >= 1).astype('int')  # outcome label, binary
        exposure_label = (df['exposed'] >= 1).astype('int')  # exposure label, binary
        covs_array = df.loc[:, covs_columns].astype('float')

        print('Overall patients: {} ({:.2f}%)'.format(len(df), 1. * 100),
              'Outcome event: {} ({:.2f}%)'.format(outcome_of_interest_flag.sum(),
                                                   outcome_of_interest_flag.sum() / len(df) * 100))
        print('Exposed patients: {} ({:.2f}%)'.format((exposure_label == 1).sum(), (exposure_label == 1).mean() * 100),
              'Outcome event: {} ({:.2f}%)'.format(outcome_of_interest_flag[(exposure_label == 1)].sum(),
                                                   outcome_of_interest_flag[(exposure_label == 1)].sum() / (
                                                           exposure_label == 1).sum() * 100))
        print('Control patients: {} ({:.2f}%)'.format((exposure_label == 0).sum(), (exposure_label == 0).mean() * 100),
              'Outcome event: {} ({:.2f}%)'.format(outcome_of_interest_flag[(exposure_label == 0)].sum(),
                                                   outcome_of_interest_flag[(exposure_label == 0)].sum() / (
                                                           exposure_label == 0).sum() * 100))

        # zz
        # model = ml.PropensityEstimator(learner='LR', random_seed=args.random_seed).cross_validation_fit(covs_array,
        #                                                                                                 exposure_label,
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
            covs_array, exposure_label, verbose=0)

        ps = model.predict_ps(covs_array)
        model.report_stats()
        iptw = model.predict_inverse_weight(covs_array, exposure_label, stabilized=True, clip=True)
        smd, smd_weighted, before, after = model.predict_smd(covs_array, exposure_label, abs=False, verbose=True)
        # plt.scatter(range(len(smd)), smd)
        # plt.scatter(range(len(smd)), smd_weighted)
        # plt.show()

        print('n unbalanced covariates before:after = {}:{}'.format(
            (np.abs(smd) > SMD_THRESHOLD).sum(),
            (np.abs(smd_weighted) > SMD_THRESHOLD).sum())
        )
        out_file_balance = r'../data/recover/output/results/LCPregOut-{}-{}-{}s{}{}/{}-{}-results.csv'.format(
            args.cohorttype,
            args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
            args.exptype,  # '-select' if args.selectpasc else '',
            args.negative_ratio, '-adjustless' if args.adjustless else '',
            i, _clean_name_(outcome_of_interest))

        utils.check_and_mkdir(out_file_balance)
        model.results.to_csv(out_file_balance)  # args.save_model_filename +

        df_summary = summary_covariate(covs_array, exposure_label, iptw, smd, smd_weighted, before, after)
        df_summary.to_csv(
            '../data/recover/output/results/LCPregOut-{}-{}-{}s{}{}/{}-{}-evaluation_balance.csv'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                args.negative_ratio, '-adjustless' if args.adjustless else '',
                i, _clean_name_(outcome_of_interest)))

        dfps = pd.DataFrame({'ps': ps, 'iptw': iptw, 'Exposure': exposure_label})

        dfps.to_csv(
            '../data/recover/output/results/LCPregOut-{}-{}-{}s{}{}/{}-{}-evaluation_ps-iptw.csv'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectoutcome_of_interest else '',
                args.negative_ratio, '-adjustless' if args.adjustless else '',
                i, _clean_name_(outcome_of_interest)))
        try:
            figout = r'../data/recover/output/results/LCPregOut-{}-{}-{}s{}{}/{}-{}-PS.png'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectoutcome_of_interest else '',
                args.negative_ratio, '-adjustless' if args.adjustless else '',
                i, _clean_name_(outcome_of_interest))
            print('Dump ', figout)

            ax = plt.subplot(111)
            sns.histplot(
                dfps, x="ps", hue="SSRI", element="step",
                stat="percent", common_norm=False, bins=25,
            )
            plt.tight_layout()
            # plt.show()
            plt.title(outcome_of_interest, fontsize=12)
            plt.savefig(figout)
            plt.close()
        except Exception as e:
            print('Dump Error', figout)
            print(str(e))
            plt.close()

        # 1. calculate odds ratio by original contigency table
        #             exposed    unexposed
        # cases          a           b
        # noncases       c           d
        n_exposed = (exposure_label == 1).sum()
        n_unexposed = (exposure_label == 0).sum()
        cont_table_a = outcome_of_interest_flag[(exposure_label == 1)].sum()
        cont_table_b = outcome_of_interest_flag[(exposure_label == 0)].sum()
        cont_table_c = (outcome_of_interest_flag[(exposure_label == 1)] == 0).sum()
        cont_table_d = (outcome_of_interest_flag[(exposure_label == 0)] == 0).sum()

        cont_table = [[cont_table_a, cont_table_b], [cont_table_c, cont_table_d]]
        res_crude = odds_ratio(cont_table)
        or_crude = res_crude.statistic
        ci_crude = res_crude.confidence_interval(confidence_level=0.95)
        res_crude = [or_crude, ci_crude.low, ci_crude.high]

        # 2. calculate odds ratio by re-weighted contigency table
        n_exposed_iptw = iptw[exposure_label == 1].sum()
        n_unexposed_iptw = iptw[exposure_label == 0].sum()
        cont_table_a_iptw = np.inner(iptw[exposure_label == 1], outcome_of_interest_flag[(exposure_label == 1)])
        cont_table_b_iptw = np.inner(iptw[exposure_label == 0], outcome_of_interest_flag[(exposure_label == 0)])
        cont_table_c_iptw = n_exposed_iptw - cont_table_a_iptw
        cont_table_d_iptw = n_unexposed_iptw - cont_table_b_iptw

        # just store here, not for odds_ratio, where A 2x2 contingency table.  Elements must be non-negative integers.
        cont_table_iptw = [[cont_table_a_iptw, cont_table_b_iptw], [cont_table_c_iptw, cont_table_d_iptw]]
        # res_crude_iptw = odds_ratio(cont_table_iptw)
        # or_crude_iptw= res_crude_iptw.statistic
        # ci_crude_iptw = res_crude_iptw.confidence_interval(confidence_level=0.95)
        # or_ci_crude_iptw = [or_crude_iptw, ci_crude_iptw.low, ci_crude_iptw.high]

        # 3. logistic regression  iptw as another cov
        # covs_array['iptw'] = iptw
        X = pd.DataFrame({'exposed':exposure_label, 'iptw':iptw})
        X = sm.add_constant(X)
        Y = outcome_of_interest_flag
        # Fit the logistic regression model
        try:
            model_iptwonly = sm.Logit(Y, X).fit()
            # Print the model summary to get p-values
            print(model_iptwonly.summary())
            # Calculate odds ratios by exponentiating the coefficients
            odds_ratios_iptwonly = np.exp(model_iptwonly.params)
            odds_ratios_ci_iptwonly = np.exp(model_iptwonly.conf_int())
            print("\nOdds Ratios:")
            print(odds_ratios_iptwonly)
            # Extract p-values directly from the model summary
            p_values_iptwonly = model_iptwonly.pvalues
            print("\nP-values:")
            print(p_values_iptwonly)
            # [ods ration, ci low, ci upper, p]
            res_iptwonly=[odds_ratios_iptwonly.exposed,
                          odds_ratios_ci_iptwonly.loc['exposed', 0],
                          odds_ratios_ci_iptwonly.loc['exposed', 1],
                          p_values_iptwonly.exposed,
                          odds_ratios_iptwonly, odds_ratios_ci_iptwonly, p_values_iptwonly]
        except:
            print('Error in Fit the logistic regression model with iptw only', i, outcome_of_interest,
                        exposure_label.sum(), (exposure_label == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 1).sum(),
                        (outcome_of_interest_flag[exposure_label == 0] == 1).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 1).mean(),
                        (outcome_of_interest_flag[exposure_label == 0] == 1).mean(),
                        (outcome_of_interest_flag[exposure_label == 1] == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 0] == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 0).mean(),
                        (outcome_of_interest_flag[exposure_label == 0] == 0).mean(),)
            res_iptwonly = [None, ] * 7

        # 4 IPTW to re-weight
        X = pd.DataFrame({'exposed': exposure_label, 'iptw': iptw})
        X = sm.add_constant(X)
        Y = outcome_of_interest_flag

        # Fit the weighted GLM model with a Binomial family and logit link
        # The weights argument is used for freq_weights
        try:
            model_reweight = sm.GLM(Y, X[['const', 'exposed']], family=sm.families.Binomial(), freq_weights=X['iptw']).fit()
            print(model_reweight.summary())
            odds_ratios_reweight  = np.exp(model_reweight.params)
            odds_ratios_ci_reweight = np.exp(model_reweight.conf_int())
            print("\nOdds Ratios:")
            print(odds_ratios_reweight)
            # Extract p-values directly from the model summary
            p_values_reweight = model_reweight.pvalues
            print("\nP-values:")
            print(p_values_reweight)
            # [ods ration, ci low, ci upper, p]
            res_reweight = [odds_ratios_reweight.exposed,
                            odds_ratios_ci_reweight.loc['exposed', 0],
                            odds_ratios_ci_reweight.loc['exposed', 1],
                            p_values_reweight.exposed,
                            odds_ratios_reweight, odds_ratios_ci_reweight, p_values_reweight]
        except:
            print('Error in Fit the logistic regression model re-weighted by IPTW', i, outcome_of_interest,
                        exposure_label.sum(), (exposure_label == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 1).sum(),
                        (outcome_of_interest_flag[exposure_label == 0] == 1).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 1).mean(),
                        (outcome_of_interest_flag[exposure_label == 0] == 1).mean(),
                        (outcome_of_interest_flag[exposure_label == 1] == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 0] == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 0).mean(),
                        (outcome_of_interest_flag[exposure_label == 0] == 0).mean(),)
            res_reweight = [None, ] * 7

        # 5. logistic regression  all the other cov
        covs_columns = [
            'age_pregonset',
            'days_between_covid_pregnant_onset',
            # 'RE:Asian Non-Hispanic',
            # 'RE:Black or African American Non-Hispanic',
            # 'RE:Hispanic or Latino Any Race', 'RE:White Non-Hispanic',
            # 'RE:Other Non-Hispanic', 'RE:Unknown',
            # 'BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ',
            # 'BMI: >=30 obese ', 'BMI: missing',
            'PaxRisk:Diabetes',
            'PaxRisk:Obesity',
            'PaxRisk:Chronic kidney disease',
            'PaxRisk:Hypertension',
            'PaxRisk:Immunocompromised condition or weakened immune system',
            'PaxRisk:Smoking current',
            'PaxRisk:Substance use disorders'
        ]
        data_dict = {'exposed': exposure_label} #, 'iptw':iptw}
        for col in covs_columns:
            data_dict[col] = df[col]
        X = pd.DataFrame(data_dict)
        X = sm.add_constant(X)
        Y = outcome_of_interest_flag
        # Fit the logistic regression model
        try:
            model_regress = sm.Logit(Y, X).fit()
            # Print the model summary to get p-values
            print(model_regress.summary())
            # Calculate odds ratios by exponentiating the coefficients
            odds_ratios_regress = np.exp(model_regress.params)
            odds_ratios_ci_regress = np.exp(model_regress.conf_int())
            print("\nOdds Ratios:")
            print(odds_ratios_regress)
            # Extract p-values directly from the model summary
            p_values_regress = model_regress.pvalues
            print("\nP-values:")
            print(p_values_regress)
            res_regress = [ odds_ratios_regress.exposed,
                            odds_ratios_ci_regress.loc['exposed', 0],
                            odds_ratios_ci_regress.loc['exposed', 1],
                            p_values_regress.exposed,
                            odds_ratios_regress, odds_ratios_ci_regress, p_values_regress
                         ]
        except:
            print('Error in Fit the logistic regression model with all covs', i, outcome_of_interest,
                        exposure_label.sum(), (exposure_label == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 1).sum(),
                        (outcome_of_interest_flag[exposure_label == 0] == 1).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 1).mean(),
                        (outcome_of_interest_flag[exposure_label == 0] == 1).mean(),
                        (outcome_of_interest_flag[exposure_label == 1] == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 0] == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 0).mean(),
                        (outcome_of_interest_flag[exposure_label == 0] == 0).mean(),)
            res_regress = [None, ] * 7

        # 6. logistic regression iptw and all the other cov
        #

        # km, km_w, cox, cox_w, cif, cif_w = weighted_KM_HR(
        #     exposure_label, iptw, outcome_of_interest_flag, outcome_of_interest_t2e,
        #     fig_outfile=r'../data/recover/output/results/LCPregOut-{}-{}-{}s{}{}/{}-{}-km.png'.format(
        #         args.cohorttype,
        #         args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
        #         args.exptype,  # '-select' if args.selectoutcome_of_interest else '',
        #         args.negative_ratio, '-adjustless' if args.adjustless else '',
        #         i, _clean_name_(outcome_of_interest)),
        #     title=outcome_of_interest,
        #     legends={'case': case_label, 'control': ctrl_label})

        try:
            # change 2022-03-20 considering competing risk 2
            # change 2024-02-29 add CI for CIF difference and KM difference
            _results = [i, outcome_of_interest,
                        exposure_label.sum(), (exposure_label == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 1).sum(),
                        (outcome_of_interest_flag[exposure_label == 0] == 1).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 1).mean(),
                        (outcome_of_interest_flag[exposure_label == 0] == 1).mean(),
                        (outcome_of_interest_flag[exposure_label == 1] == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 0] == 0).sum(),
                        (outcome_of_interest_flag[exposure_label == 1] == 0).mean(),
                        (outcome_of_interest_flag[exposure_label == 0] == 0).mean(),
                        (np.abs(smd) > SMD_THRESHOLD).sum(), (np.abs(smd_weighted) > SMD_THRESHOLD).sum(),
                        np.abs(smd).max(), np.abs(smd_weighted).max(),
                        # km[2], km[3], km[6].p_value,
                        # list(km[6].diff_of_mean), list(km[6].diff_of_mean_lower), list(km[6].diff_of_mean_upper),
                        # cif[2], cif[4], cif[5], cif[6], cif[7], cif[8], cif[9],
                        # list(cif[10].diff_of_mean), list(cif[10].diff_of_mean_lower), list(cif[10].diff_of_mean_upper),
                        # cif[10].p_value,
                        # km_w[2], km_w[3], km_w[6].p_value,
                        # list(km_w[6].diff_of_mean), list(km_w[6].diff_of_mean_lower), list(km_w[6].diff_of_mean_upper),
                        # cif_w[2], cif_w[4], cif_w[5], cif_w[6], cif_w[7], cif_w[8], cif_w[9],
                        # list(cif_w[10].diff_of_mean), list(cif_w[10].diff_of_mean_lower),
                        # list(cif_w[10].diff_of_mean_upper),
                        # cif_w[10].p_value,
                        # cox[0], cox[1], cox[3].summary.p.treatment if pd.notna(cox[3]) else np.nan, cox[2], cox[4],
                        # cox_w[0], cox_w[1], cox_w[3].summary.p.treatment if pd.notna(cox_w[3]) else np.nan, cox_w[2],
                        # cox_w[4],
                        n_exposed, n_unexposed,  cont_table_a, cont_table_b, cont_table_c, cont_table_d,
                        res_crude[0], res_crude[1], res_crude[2],
                        n_exposed_iptw, n_unexposed_iptw,
                        cont_table_a_iptw, cont_table_b_iptw, cont_table_c_iptw, cont_table_d_iptw,
                        res_iptwonly[0], res_iptwonly[1], res_iptwonly[2], res_iptwonly[3],
                        res_iptwonly[4], res_iptwonly[5], res_iptwonly[6],
                        res_reweight[0], res_reweight[1], res_reweight[2], res_reweight[3],
                        res_reweight[4], res_reweight[5], res_reweight[6],
                        res_regress[0], res_regress[1], res_regress[2], res_regress[3],
                        res_regress[4], res_regress[5], res_regress[6],

                        model.best_hyper_paras]
            causal_results.append(_results)
            results_columns_name = [
                'i', 'outcome_of_interest', 'case+', 'ctrl-',
                'no. outcome_of_interest in +', 'no. outcome_of_interest in -', 'mean outcome_of_interest in +',
                'mean outcome_of_interest in -',
                'no. death in +', 'no. death in -', 'mean death in +', 'mean death in -',
                'no. unbalance', 'no. unbalance iptw', 'max smd', 'max smd iptw',
                #
                # 'km-diff', 'km-diff-time', 'km-diff-p',
                # 'km-diff-2', 'km-diff-CILower', 'km-diff-CIUpper',
                # 'cif-diff', "cif_1", "cif_0", "cif_1_CILower", "cif_1_CIUpper", "cif_0_CILower", "cif_0_CIUpper",
                # 'cif-diff-2', 'cif-diff-CILower', 'cif-diff-CIUpper', 'cif-diff-p',
                # 'km-w-diff', 'km-w-diff-time', 'km-w-diff-p',
                # 'km-w-diff-2', 'km-w-diff-CILower', 'km-w-diff-CIUpper',
                # 'cif-w-diff', "cif_1_w", "cif_0_w", "cif_1_w_CILower", "cif_1_w_CIUpper", "cif_0_w_CILower",
                # "cif_0_w_CIUpper", 'cif-w-diff-2', 'cif-w-diff-CILower', 'cif-w-diff-CIUpper', 'cif-w-diff-p',
                # 'hr', 'hr-CI', 'hr-p', 'hr-logrank-p', 'hr_different_time',
                # 'hr-w', 'hr-w-CI', 'hr-w-p', 'hr-w-logrank-p', "hr-w_different_time",
                "n_exposed", "n_unexposed", "cont_table_a", "cont_table_b", "cont_table_c", "cont_table_d",
                "odds_ratios_crude", "odds_ratios ci_crude.low", "odds_ratios ci_crude.high",

                "n_exposed_iptw", "n_unexposed_iptw",
                "cont_table_a_iptw", "cont_table_b_iptw", "cont_table_c_iptw", "cont_table_d_iptw",

                "odds_ratios_iptwonly.exposed", "odds_ratios_ci_iptwonly lower", "odds_ratios_ci_iptwonly upper",
                "p_values_iptwonly.exposed",
                "odds_ratios_iptwonly all", "odds_ratios_ci_iptwonly all", "p_values_iptwonly all",

                "odds_ratios_iptwreweight.exposed", "odds_ratios_ci_iptwreweight lower", "odds_ratios_ci_iptwreweight upper",
                "p_values_iptwreweight.exposed",
                "odds_ratios_iptwreweight all", "odds_ratios_ci_iptwreweight all", "p_values_iptwreweight all",

                "odds_ratios_regress.exposed", "odds_ratios_ci_regress lower", "odds_ratios_ci_regress upper",
                "p_values_regress.exposed",
                "odds_ratios_regress all", "odds_ratios_ci_regress all", "p_values_regress all",
                'best_hyper_paras']
            print('causal result:\n', causal_results[-1])

            if i % 2 == 0:
                pd.DataFrame(causal_results, columns=results_columns_name). \
                    to_csv(
                    r'../data/recover/output/results/LCPregOut-{}-{}-{}s{}{}/causal_effects_specific-snapshot-{}.csv'.format(
                        args.cohorttype,
                        args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                        args.exptype,  # '-select' if args.selectpasc else '',
                        args.negative_ratio, '-adjustless' if args.adjustless else '',
                        i))
        except:
            print('Error in ', i, outcome_of_interest)
            df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

            df_causal.to_csv(
                r'../data/recover/output/results/LCPregOut-{}-{}-{}s{}{}/causal_effects_specific-ERRORSAVE.csv'.format(
                    args.cohorttype,
                    args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                    args.exptype,  # '-select' if args.selectoutcome_of_interest else '',
                    args.negative_ratio, '-adjustless' if args.adjustless else '',
                ))

        print('done one outcome_of_interest, time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

    df_causal.to_csv(
        r'../data/recover/output/results/LCPregOut-{}-{}-{}s{}{}/causal_effects_specific.csv'.format(
            args.cohorttype,
            args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
            args.exptype,  # '-select' if args.selectoutcome_of_interest else '',
            args.negative_ratio, '-adjustless' if args.adjustless else '',
        ))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
