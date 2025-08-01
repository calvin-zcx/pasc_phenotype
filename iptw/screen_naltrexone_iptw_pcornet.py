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
    parser.add_argument('--negative_ratio', type=int, default=100)  # 5
    parser.add_argument('--selectpasc', action='store_true')
    parser.add_argument('--build_data', action='store_true')
    parser.add_argument('--dump', action='store_true')
    parser.add_argument('--adjustless', action='store_true')

    parser.add_argument('--exptype',
                        choices=['nal-inc0-30',
                                 ], default='nal-inc0-30')  # 'base180-0'

    parser.add_argument('--cohorttype', #
                        choices=['matchK10replace', 'matchK5replace', 'matchK15replace',
                                   ],
                        default='matchK10replace')
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
    s = s.replace(':', '-').replace('/', '-').replace('@', '-')
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

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)

    # in_file_infectt0 = r'./cns_output/Matrix-cns-adhd-CNS-ADHD-acuteIncident-0-30-25Q2-v3.csv'
    if args.cohorttype == 'matchK10replace':
        in_file = r'./naltrexone_output/Matrix-naltrexone-naltrexone-acuteIncident-0-30-25Q3-naltrexCovAtDrugOnset-applyEC-PainIncludeOnly-kmatch-10-v3-replace.csv'
        print('cohorttype: matchK10replace:', args.cohorttype, in_file)
    elif args.cohorttype == 'matchK5replace':
        in_file = r'./naltrexone_output/Matrix-naltrexone-naltrexone-acuteIncident-0-30-25Q3-naltrexCovAtDrugOnset-applyEC-PainIncludeOnly-kmatch-5-v3-replace.csv'
        print('cohorttype: matchK10replace:', args.cohorttype, in_file)
    else:
        raise ValueError

    print('infile:', in_file)
    df = pd.read_csv(in_file,
                     dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'dob',
                                  'flag_delivery_date',
                                  'flag_pregnancy_start_date',
                                  'flag_pregnancy_end_date'
                                  ])

    print('df.shape:', df.shape)

    case_label = 'Treated'
    ctrl_label = 'Nontreated'

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

    # data clean for <0 error death records, and add censoring to the death time to event columns

    df['death'].fillna(0, inplace=True,)

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

    df['cvd death postacute'] = ((df['dxCVDdeath-out@death_cardiovascular'] >= 1) & (df['death'] == 1)
                                 & (df['death t2e'] >= 31) & (df['death t2e'] < 180)).astype('int')
    df['cvd death t2e postacute'] = df['death t2e all']

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

    mecfs_encoding = utils.load(r'../data/mapping/mecfs_index_mapping.pkl')
    mecfs_list = list(mecfs_encoding.keys())

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

    for p in mecfs_list:
        pasc_simname[p] = (p, 'General-add')
        pasc_organ[p] = 'General-add'

    # for p in mecfs_list:
    #     pasc_simname[p] = (p, 'ME/CFS')
    #     pasc_organ[p] = 'ME/CFS'

    # pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    pasc_list_raw = df_pasc_info.loc[df_pasc_info['selected_narrow'] == 1, 'pasc'].to_list()
    _exclude_list = ['Pressure ulcer of skin', 'Fluid and electrolyte disorders']
    pasc_list = [x for x in pasc_list_raw if x not in _exclude_list]

    pasc_add = ['smell and taste', ]
    pasc_add_mecfs = ['ME/CFS', ]
    print('len(pasc_list)', len(pasc_list), 'len(pasc_add)', len(pasc_add))
    print('pasc_list:', pasc_list)
    print('pasc_add', pasc_add)
    print('pasc_add_mecfs', pasc_add_mecfs)

    for p in pasc_list:
        df[p + '_pasc_flag'] = 0
    for p in pasc_add:
        df[p + '_pasc_flag'] = 0
    for p in pasc_add_mecfs:
        df[p + '_pasc_flag'] = 0
    for p in CFR_list:
        df[p + '_CFR_flag'] = 0

    # move brainfog_list_any and '_brainfog_flag'  below

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

    # 2025-2-20, original list 7, current any brain fog excludes headache because already in individual any pasc
    # ['Neurodegenerative', 'Memory-Attention', 'Headache',
    # 'Sleep Disorder', 'Psych', 'Dysautonomia-Orthostatic', 'Stroke'])
    df['any_brainfog_flag'] = 0
    # df['any_brainfog_type'] = np.nan
    df['any_brainfog_t2e'] = 180  # np.nan
    df['any_brainfog_txt'] = ''
    df['any_brainfog_baseline'] = 0  # placeholder for screening, no special meaning, null column
    brainfog_list_any = ['Neurodegenerative', 'Memory-Attention',  # 'Headache',
                         'Sleep Disorder', 'Psych', 'Dysautonomia-Orthostatic', 'Stroke']
    for p in brainfog_list_any:
        df[p + '_brainfog_flag'] = 0

    print('brainfog_list_any:', brainfog_list_any)
    print('len(brainfog_list_any):', len(brainfog_list_any), 'len(brainfog_list)', len(brainfog_list))

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

        for p in pasc_add_mecfs:
            # dxMECFS-base@ME/CFS
            if (rows['dxMECFS-out@' + p] > 0) and (rows['dxMECFS-base@' + p] == 0):
                t2e_list.append(rows['dxMECFS-t2e@' + p])
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

        # for brain fog pasc
        brainfog_t2e_list = []
        brainfog_1_list = []
        brainfog_1_name = []
        brainfog_1_text = ''
        for p in brainfog_list_any:
            if (rows['dxbrainfog-out@' + p] > 0) and (rows['dxbrainfog-base@' + p] == 0):
                brainfog_t2e_list.append(rows['dxbrainfog-t2e@' + p])
                brainfog_1_list.append(p)
                brainfog_1_name.append(pasc_simname[p])
                brainfog_1_text += (pasc_simname[p][0] + ';')

                df.loc[index, p + '_brainfog_flag'] = 1

        if len(brainfog_t2e_list) > 0:
            df.loc[index, 'any_brainfog_flag'] = 1
            df.loc[index, 'any_brainfog_t2e'] = np.min(brainfog_t2e_list)
            df.loc[index, 'any_brainfog_txt'] = brainfog_1_text
        else:
            df.loc[index, 'any_brainfog_flag'] = 0
            df.loc[index, 'any_brainfog_t2e'] = rows[
                ['dxbrainfog-t2e@' + p for p in brainfog_list_any]].max()  # censoring time

    # End of defining ANY *** conditions
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
                  '03/24-06/24', '07/24-10/24',

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
                                                x.startswith('dxMECFS') or
                                                x.startswith('PaxRisk:') or
                                                x.startswith('dxcovCNSLDN-base@')
                                                ]

    df_outcome = df.loc[:, df_outcome_cols]  # .astype('float')

    # if args.cohorttype in ['overall']:
    covs_columns = [
        'Female', 'Male', 'Other/Missing',
        'age@18-24', 'age@25-34', 'age@35-49', 'age@50-64', 'age@65+',
        'RE:Asian Non-Hispanic',
        'RE:Black or African American Non-Hispanic',
        'RE:Hispanic or Latino Any Race', 'RE:White Non-Hispanic',
        'RE:Other Non-Hispanic', 'RE:Unknown',
        'outpatient', 'inpatienticu',
        'ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
        'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100', 'ADIMissing',
        '03/22-06/22', '07/22-10/22', '11/22-02/23',
        '03/23-06/23', '07/23-10/23', '11/23-02/24',
        '03/24-06/24', '07/24-10/24',
        # 'quart:01/22-03/22', 'quart:04/22-06/22', 'quart:07/22-09/22', 'quart:10/22-1/23',
        'inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits 3-4',
        'inpatient visits >=5',
        'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits 3-4',
        'outpatient visits >=5',
        'emergency visits 0', 'emergency visits 1-2', 'emergency visits 3-4',
        'emergency visits >=5',
        'BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ',
        'BMI: >=30 obese ', 'BMI: missing',
        # 'PaxRisk:Cancer',
        'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
        'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
        'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
        'PaxRisk:Heart conditions', 'PaxRisk:Hypertension',  # 'PaxRisk:HIV infection',
        'PaxRisk:Immunocompromised condition or weakened immune system', 'PaxRisk:Mental health conditions',
        #'PaxRisk:Overweight and obesity',  # 'PaxRisk:Pregnancy',
        'PaxRisk:Sickle cell disease or thalassemia',
        'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
        #'PaxRisk:Substance use disorders',
        'PaxRisk:Tuberculosis',
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
        # 'mental-base@premature ejaculation',
        'mental-base@Autism spectrum disorder',
        'mental-base@Premenstrual dysphoric disorder',
        'mental-base@SMI',
        #'mental-base@non-SMI',
        'dxcovNaltrexone-basedrugonset@MECFS',
        # 'dxcovNaltrexone-basedrugonset@Pain',
        'dxcovNaltrexone-basedrugonset@substance use disorder ',
        'dxcovNaltrexone-basedrugonset@opioid use disorder',
        'dxcovNaltrexone-basedrugonset@Opioid induced constipation',
        # 'dxcovNaltrexone-basedrugonset@Obesity',
        'PaxRisk:Obesity',
        'dxcovNaltrexone-basedrugonset@Crohn-Inflamm_Bowel',
        'dxcovNaltrexone-basedrugonset@fibromyalgia',
        'dxcovNaltrexone-basedrugonset@multiple sclerosis',
        'dxcovNaltrexone-basedrugonset@POTS'
    ]

    if args.adjustless:
        covs_columns = [
            'Female', 'Male', 'Other/Missing',
            'age@18-24', 'age@25-34', 'age@35-49', 'age@50-64',  # 'age@65+', # # expand 65
            '65-<75 years', '75-<85 years', '85+ years', ]
    #
    # if args.cohorttype == 'overall':
    #     covs_columns += ['ADHD_before_drug_onset', ]

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

    selected_screen_list = (['any_pasc', 'PASC-General', 'ME/CFS',
                             'death', 'death_acute', 'death_postacute', 'cvddeath_postacute',
                             'any_CFR', 'any_brainfog',
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
        out_file_balance = r'../data/recover/output/results/naltrexone-{}-{}-{}s{}{}/{}-{}-results.csv'.format(
            args.cohorttype,
            args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
            args.exptype,  # '-select' if args.selectpasc else '',
            args.negative_ratio, '-adjustless' if args.adjustless else '',
            i, _clean_name_(pasc))

        utils.check_and_mkdir(out_file_balance)
        model.results.to_csv(out_file_balance)  # args.save_model_filename +

        df_summary = summary_covariate(covs_array, covid_label, iptw, smd, smd_weighted, before, after)
        df_summary.to_csv(
            '../data/recover/output/results/naltrexone-{}-{}-{}s{}{}/{}-{}-evaluation_balance.csv'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                args.negative_ratio, '-adjustless' if args.adjustless else '',
                i, _clean_name_(pasc)))

        dfps = pd.DataFrame({'ps': ps, 'iptw': iptw, 'Exposure': covid_label})

        dfps.to_csv(
            '../data/recover/output/results/naltrexone-{}-{}-{}s{}{}/{}-{}-evaluation_ps-iptw.csv'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                args.negative_ratio, '-adjustless' if args.adjustless else '',
                i, _clean_name_(pasc)))
        try:
            figout = r'../data/recover/output/results/naltrexone-{}-{}-{}s{}{}/{}-{}-PS.png'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                args.negative_ratio, '-adjustless' if args.adjustless else '',
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
            fig_outfile=r'../data/recover/output/results/naltrexone-{}-{}-{}s{}{}/{}-{}-km.png'.format(
                args.cohorttype,
                args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                args.exptype,  # '-select' if args.selectpasc else '',
                args.negative_ratio, '-adjustless' if args.adjustless else '',
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
                    r'../data/recover/output/results/naltrexone-{}-{}-{}s{}{}/causal_effects_specific-snapshot-{}.csv'.format(
                        args.cohorttype,
                        args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                        args.exptype,  # '-select' if args.selectpasc else '',
                        args.negative_ratio, '-adjustless' if args.adjustless else '',
                        i))
        except:
            print('Error in ', i, pasc)
            df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

            df_causal.to_csv(
                r'../data/recover/output/results/naltrexone-{}-{}-{}s{}{}/causal_effects_specific-ERRORSAVE.csv'.format(
                    args.cohorttype,
                    args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
                    args.exptype,  # '-select' if args.selectpasc else '',
                    args.negative_ratio, '-adjustless' if args.adjustless else '',
                ))

        print('done one pasc, time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

    df_causal.to_csv(
        r'../data/recover/output/results/naltrexone-{}-{}-{}s{}{}/causal_effects_specific.csv'.format(
            args.cohorttype,
            args.severity.replace(':', '_').replace('/', '-').replace(' ', '_'),
            args.exptype,  # '-select' if args.selectpasc else '',
            args.negative_ratio, '-adjustless' if args.adjustless else '',
        ))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
