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
    parser.add_argument('--site', default='oneflorida',
                        choices=['insight', 'oneflorida', 'COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL', 'all'],
                        help='one particular site or all')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--negative_ratio', type=int, default=10)  # 5
    parser.add_argument('--selectpasc', action='store_true')
    parser.add_argument('--build_data', action='store_true')
    parser.add_argument('--cohorttype', choices=['lab-dx-med', 'lab-dx', 'lab'], default='lab-dx-med')

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


def add_col(df):
    # re-organize Paxlovid risk factors
    # see https://www.cdc.gov/coronavirus/2019-ncov/need-extra-precautions/people-with-medical-conditions.html
    # and https://www.paxlovid.com/who-can-take
    print('in add_col, df.shape', df.shape)

    # print('Build covs for outpatient cohorts w/o ICU or ventilation')
    # df['inpatient'] = ((df['hospitalized'] == 1) & (df['ventilation'] == 0) & (df['criticalcare'] == 0)).astype('int')
    # df['icu'] = (((df['hospitalized'] == 1) & (df['ventilation'] == 1)) | (df['criticalcare'] == 1)).astype('int')
    # df['inpatienticu'] = ((df['hospitalized'] == 1) | (df['criticalcare'] == 1) | (df['ventilation'] == 1)).astype(
    #     'int')
    # df['outpatient'] = ((df['hospitalized'] == 0) & (df['criticalcare'] == 0) & (df['ventilation'] == 0)).astype('int')
    #
    # df['PaxRisk:Cancer'] = (
    #         ((df["DX: Cancer"] >= 1).astype('int') +
    #          (df['CCI:Cancer'] >= 1).astype('int') +
    #          (df['CCI:Metastatic Carcinoma'] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Chronic kidney disease'] = (df["DX: Chronic Kidney Disease"] >= 1).astype('int')
    #
    # df['PaxRisk:Chronic liver disease'] = (
    #         ((df["DX: Cirrhosis"] >= 1).astype('int') +
    #          (df['CCI:Mild Liver Disease'] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Chronic lung disease'] = (
    #         ((df['CCI:Chronic Pulmonary Disease'] >= 1).astype('int') +
    #          (df["DX: Asthma"] >= 1).astype('int') +
    #          (df["DX: Chronic Pulmonary Disorders"] >= 1).astype('int') +
    #          (df["DX: COPD"] >= 1).astype('int') +
    #          (df["DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)"] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Cystic fibrosis'] = (df['DX: Cystic Fibrosis'] >= 1).astype('int')
    #
    # df['PaxRisk:Dementia or other neurological conditions'] = (
    #         ((df['DX: Dementia'] >= 1).astype('int') +
    #          (df['CCI:Dementia'] >= 1).astype('int') +
    #          (df["DX: Parkinson's Disease"] >= 1).astype('int') +
    #          (df["DX: Multiple Sclerosis"] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Diabetes'] = (
    #         ((df["DX: Diabetes Type 1"] >= 1).astype('int') +
    #          (df["DX: Diabetes Type 2"] >= 1).astype('int') +
    #          (df['CCI:Diabetes without complications'] >= 1).astype('int') +
    #          (df['CCI:Diabetes with complications'] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Disabilities'] = (
    #         ((df['CCI:Paraplegia and Hemiplegia'] >= 1).astype('int') +
    #          (df["DX: Down's Syndrome"] >= 1).astype('int') +
    #          (df["DX: Hemiplegia"] >= 1).astype('int') +
    #          (df["DX: Autism"] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Heart conditions'] = (
    #         ((df["DX: Congestive Heart Failure"] >= 1).astype('int') +
    #          (df["DX: Coronary Artery Disease"] >= 1).astype('int') +
    #          (df["DX: Arrythmia"] >= 1).astype('int') +
    #          (df['CCI:Myocardial Infarction'] >= 1).astype('int') +
    #          (df['CCI:Congestive Heart Failure'] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Hypertension'] = (df["DX: Hypertension"] >= 1).astype('int')
    #
    # df['PaxRisk:HIV infection'] = (
    #         ((df["DX: HIV"] >= 1).astype('int') +
    #          (df['CCI:AIDS/HIV'] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Immunocompromised condition or weakened immune system'] = (
    #         ((df['DX: Inflammatory Bowel Disorder'] >= 1).astype('int') +
    #          (df['DX: Lupus or Systemic Lupus Erythematosus'] >= 1).astype('int') +
    #          (df['DX: Rheumatoid Arthritis'] >= 1).astype('int') +
    #          (df["MEDICATION: Corticosteroids"] >= 1).astype('int') +
    #          (df["MEDICATION: Immunosuppressant drug"] >= 1).astype('int') +
    #          (df["CCI:Connective Tissue Disease-Rheumatic Disease"] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Mental health conditions'] = (df["DX: Mental Health Disorders"] >= 1).astype('int')
    #
    # df["PaxRisk:Overweight and obesity"] = (
    #         (df["DX: Severe Obesity  (BMI>=40 kg/m2)"] >= 1) | (df['bmi'] >= 25) | (df['addPaxRisk:Obesity'] >= 1)
    # ).astype('int')
    #
    # # physical activity
    # # not captured
    #
    # # pregnancy, use infection during pregnant, label from pregnant cohorts
    # df["PaxRisk:Pregnancy"] = ((df['flag_pregnancy'] == 1) &
    #                            (df['index date'] >= df['flag_pregnancy_start_date'] - datetime.timedelta(days=7)) &
    #                            (df['index date'] <= df['flag_delivery_date'] + datetime.timedelta(days=7))).astype(
    #     'int')
    #
    # df['PaxRisk:Sickle cell disease or thalassemia'] = (
    #         ((df['DX: Sickle Cell'] >= 1).astype('int') +
    #          (df["DX: Anemia"] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Smoking current'] = ((df['Smoker: current'] >= 1) | (df['Smoker: former'] >= 1)).astype('int')
    #
    # # cell transplant, --> autoimmu category?
    #
    # df['PaxRisk:Stroke or cerebrovascular disease'] = (df['CCI:Cerebrovascular Disease'] >= 1).astype('int')
    #
    # df['PaxRisk:Substance use disorders'] = (
    #         ((df["DX: Alcohol Abuse"] >= 1).astype('int') +
    #          (df['DX: Other Substance Abuse'] >= 1).astype('int') +
    #          (df['addPaxRisk:Drug Abuse'] >= 1).astype('int')
    #          ) >= 1
    # ).astype('int')
    #
    # df['PaxRisk:Tuberculosis'] = (df['addPaxRisk:tuberculosis'] >= 1).astype('int')
    #
    # # PAXLOVID is not recommended for people with severe kidney disease
    # # PAXLOVID is not recommended for people with severe liver disease
    # df['PaxExclude:liver'] = (df['CCI:Moderate or Severe Liver Disease'] >= 1).astype('int')
    # df['PaxExclude:end-stage kidney disease'] = (df["DX: End Stage Renal Disease on Dialysis"] >= 1).astype('int')
    #
    # pax_risk_cols = [x for x in df.columns if x.startswith('PaxRisk:')]
    # print('pax_risk_cols:', len(pax_risk_cols), pax_risk_cols)
    # df['PaxRisk-Count'] = df[pax_risk_cols].sum(axis=1)
    #
    # pax_exclude_cols = [x for x in df.columns if x.startswith('PaxExclude:')]
    # print('pax_exclude_cols:', len(pax_exclude_cols), pax_exclude_cols)
    # df['PaxExclude-Count'] = df[pax_exclude_cols].sum(axis=1)
    # # Tuberculosis, not captured, need to add columns?
    #
    # # 2024-09-07 add ssri indication covs
    # mental_cov = ['mental-base@Schizophrenia Spectrum and Other Psychotic Disorders',
    #               'mental-base@Depressive Disorders',
    #               'mental-base@Bipolar and Related Disorders',
    #               'mental-base@Anxiety Disorders',
    #               'mental-base@Obsessive-Compulsive and Related Disorders',
    #               'mental-base@Post-traumatic stress disorder',
    #               'mental-base@Bulimia nervosa',
    #               'mental-base@Binge eating disorder',
    #               'mental-base@premature ejaculation',
    #               'mental-base@Autism spectrum disorder',
    #               'mental-base@Premenstrual dysphoric disorder', ]
    #
    # df['SSRI-Indication-dsm-cnt'] = df[mental_cov].sum(axis=1)
    # df['SSRI-Indication-dsm-flag'] = (df['SSRI-Indication-dsm-cnt'] > 0).astype('int')
    #
    # df['SSRI-Indication-dsmAndExlix-cnt'] = df[mental_cov + ['mental-base@SMI', 'mental-base@non-SMI',]].sum(axis=1)
    # df['SSRI-Indication-dsmAndExlix-flag'] = (df['SSRI-Indication-dsmAndExlix-cnt'] > 0).astype('int')
    #
    # # ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    # df['cci_quan:0'] = 0
    # df['cci_quan:1-2'] = 0
    # df['cci_quan:3-4'] = 0
    # df['cci_quan:5-10'] = 0
    # df['cci_quan:11+'] = 0
    #
    # # ['age18-24', 'age25-34', 'age35-49', 'age50-64', 'age65+']
    # # df['age@18-<25'] = 0
    # # df['age@25-<30'] = 0
    # # df['age@30-<35'] = 0
    # # df['age@35-<40'] = 0
    # # df['age@40-<45'] = 0
    # # df['age@45-<50'] = 0
    # # df['age@50-<55'] = 0
    # # df['age@55-<60'] = 0
    # # df['age@60-<65'] = 0
    # # df['age@65-<60'] = 0

    df['age@18-24'] = 0
    df['age@25-34'] = 0
    df['age@35-49'] = 0
    df['age@50-64'] = 0
    df['age@65-74'] = 0
    df['age@75+'] = 0

    df['table1-NHW'] = 0
    df['table1-NHB'] = 0
    df['table1-Hispanics'] = 0
    df['table1-Other'] = 0
    df['table1-Unknown'] = 0


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
            elif age < 75:
                df.loc[index, 'age@65-74'] = 1
            elif age >= 75:
                df.loc[index, 'age@75+'] = 1

        # if row['score_cci_quan'] <= 0:
        #     df.loc[index, 'cci_quan:0'] = 1
        # elif row['score_cci_quan'] <= 2:
        #     df.loc[index, 'cci_quan:1-2'] = 1
        # elif row['score_cci_quan'] <= 4:
        #     df.loc[index, 'cci_quan:3-4'] = 1
        # elif row['score_cci_quan'] <= 10:
        #     df.loc[index, 'cci_quan:5-10'] = 1
        # elif row['score_cci_quan'] >= 11:
        #     df.loc[index, 'cci_quan:11+'] = 1
        #
        # if row['Asian'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
        #     df.loc[index, 'RE:Asian Non-Hispanic'] = 1
        # elif row['Black or African American'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
        #     df.loc[index, 'RE:Black or African American Non-Hispanic'] = 1
        # elif row['Hispanic: Yes']:
        #     df.loc[index, 'RE:Hispanic or Latino Any Race'] = 1
        # elif row['White'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
        #     df.loc[index, 'RE:White Non-Hispanic'] = 1
        # elif row['Other'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
        #     df.loc[index, 'RE:Other Non-Hispanic'] = 1
        # else:
        #     df.loc[index, 'RE:Unknown'] = 1

        if row['White'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'table1-NHW'] = 1
        elif row['Black or African American'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'table1-NHB'] = 1
        elif row['Hispanic: Yes']:
            df.loc[index, 'table1-Hispanics'] = 1
        elif (row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1):
            df.loc[index, 'table1-Other'] = 1
        else:
            df.loc[index, 'table1-Unknown'] = 1


        # visits = row['inpatient no.'] + row['outpatient no.'] + row['emergency visits no.'] + row['other visits no.']
        # if visits == 0:
        #     df.loc[index, 'No. of Visits:0'] = 1
        # elif visits <= 3:
        #     df.loc[index, 'No. of Visits:1-3'] = 1
        # elif visits <= 9:
        #     df.loc[index, 'No. of Visits:4-9'] = 1
        # elif visits <= 19:
        #     df.loc[index, 'No. of Visits:10-19'] = 1
        # else:
        #     df.loc[index, 'No. of Visits:>=20'] = 1
        #
        # if row['inpatient no.'] == 0:
        #     df.loc[index, 'No. of hospitalizations:0'] = 1
        # elif row['inpatient no.'] == 1:
        #     df.loc[index, 'No. of hospitalizations:1'] = 1
        # else:
        #     df.loc[index, 'No. of hospitalizations:>=1'] = 1

        # any PASC

        # adi use mine later, not transform here, add missing

        # monthly changes, add later. Already there
    print('Finish add_col, df.shape', df.shape)
    return df


def more_ec_for_cohort_selection(df):
    print('in more_ec_for_cohort_selection, df.shape', df.shape)
    print('Applying more specific/flexible eligibility criteria for cohort selection')

    # select index date
    # print('Before selecting index date from 2022-4-1 to 2023-2-28, len(df)', len(df))
    df = df.loc[
         (df['index date'] >= datetime.datetime(2022, 3, 1, 0, 0)) &
         (df['index date'] <= datetime.datetime(2023, 2, 28, 0, 0)), :]
    # print('After selecting index date from 2022-1-1 to 2023-2-28, len(df)', len(df))
    print('After selecting index date from 2022-3-1 to 2023-2-28, len(df)', len(df))

    # Exclusion, no hospitalized
    # print('Before selecting no hospitalized, len(df)', len(df))
    df = df.loc[(df['outpatient'] == 1), :]
    print('After selecting no hospitalized, len(df)', len(df))

    def ec_no_U099_baseline(_df):
        print('before ec_no_U099_baseline, _df.shape', _df.shape)
        _df = _df.loc[(_df['dx-base@PASC-General'] == 0)]
        print('after ec_no_U099_baseline, _df.shape', _df.shape)
        return _df

    df = ec_no_U099_baseline(df)

    def ec_no_other_covid_treatment(_df):
        print('before ec_no_other_covid_treatment, _df.shape', _df.shape)
        _df = _df.loc[(~(_df['treat-t2e@remdesivir'] <= 14)) &
                      (_df['Remdesivir'] == 0) &
                      (_df['Molnupiravir'] == 0) &
                      (_df[
                           'Any Monoclonal Antibody Treatment (Bamlanivimab, Bamlanivimab and Etesevimab, Casirivimab and Imdevimab, Sotrovimab, and unspecified monoclonal antibodies)'] == 0) &
                      (_df['PX: Convalescent Plasma'] == 0) &
                      (_df['pax_contra'] == 0)]
        print('after ec_no_other_covid_treatment, _df.shape', _df.shape)
        return _df

    def ec_at_least_one_risk_4_pax(_df):
        print('before ec_at_least_one_risk_4_pax, _df.shape', _df.shape)
        _df = _df.loc[(_df['age'] >= 50) | (_df['PaxRisk-Count'] > 0)]
        print('after ec_at_least_one_risk_4_pax, _df.shape', _df.shape)
        return _df

    def ec_not_at_risk_4_pax(_df):
        print('before ec_not_at_risk_4_pax, _df.shape', _df.shape)
        _df = _df.loc[~((_df['age'] >= 50) | (_df['PaxRisk-Count'] > 0))]
        print('after ec_not_at_risk_4_pax, _df.shape', _df.shape)
        return _df

    def ec_no_severe_conditions_4_pax(_df):
        print('before ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
        _df = _df.loc[(_df['PaxExclude-Count'] == 0)]
        print('after ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
        return _df

    print('**************build treated patients')
    # drug initiation within 5 days
    df_pos = df.loc[(df['treat-flag@paxlovid'] > 0), :]
    print('After selecting pax prescription, len(df_pos)', len(df_pos))
    df_pos = df_pos.loc[(df_pos['treat-t2e@paxlovid'] <= 5), :]
    print('After selecting pax prescription within 5 days, len(df_pos)', len(df_pos))
    df_pos = ec_no_other_covid_treatment(df_pos)

    print('**************build treated patients**AT risk patients')
    df_pos_risk = ec_at_least_one_risk_4_pax(df_pos)
    df_pos_risk = ec_no_severe_conditions_4_pax(df_pos_risk)

    print('**************build treated patients**NO risk patients')
    df_pos_norisk = ec_not_at_risk_4_pax(df_pos)
    df_pos_norisk = ec_no_severe_conditions_4_pax(df_pos_norisk)

    print('**************build control patients')
    # non initiation group, no paxlovid
    df_ctrl = df.loc[(df['treat-flag@paxlovid'] == 0), :]
    print('After selecting NO pax prescription, len(df_ctrl)', len(df_ctrl))
    df_ctrl = ec_no_other_covid_treatment(df_ctrl)

    print('**************build control patients**AT risk patients')
    df_ctrl_risk = ec_at_least_one_risk_4_pax(df_ctrl)
    df_ctrl_risk = ec_no_severe_conditions_4_pax(df_ctrl_risk)

    print('**************build control patients**NO risk patients')
    df_ctrl_norisk = ec_not_at_risk_4_pax(df_ctrl)
    df_ctrl_norisk = ec_no_severe_conditions_4_pax(df_ctrl_norisk)

    # # select age and risk
    # # print('Before selecting age >= 50 or at least on risk, len(df)', len(df))
    # df = df.loc[(df['age'] >= 50) | (df['pax_risk'] > 0), :]  # .copy()
    # print('After selecting age >= 50 or at least on risk, len(df)', len(df))
    #
    # # Exclusion, no contra
    # # print('Before selecting pax drug contraindication, len(df)', len(df))
    # df = df.loc[(df['pax_contra'] == 0), :]
    # print('After selecting pax drug contraindication, len(df)', len(df))

    return df_pos_risk, df_pos_norisk, df_ctrl_risk, df_ctrl_norisk


def more_ec_for_cohort_selection_new_order(df):
    print('in more_ec_for_cohort_selection, df.shape', df.shape)
    print('Applying more specific/flexible eligibility criteria for cohort selection')

    # select index date
    # print('Before selecting index date from 2022-4-1 to 2023-2-1, len(df)', len(df))
    print('Before selecting index date from 2022-3-1 to 2023-2-1, len(df)', len(df))
    df = df.loc[
         (df['index date'] >= datetime.datetime(2022, 3, 1, 0, 0)) &
         (df['index date'] <= datetime.datetime(2023, 2, 1, 0, 0)), :]
    # print('After selecting index date from 2022-1-1 to 2023-2-1, len(df)', len(df))
    print('After selecting index date from 2022-3-1 to 2023-2-1, len(df)', len(df))

    df_ec_start = df.copy()
    print('Build df_ec_start for calculating EC proportion, len(df_ec_start)', len(df_ec_start))

    # Exclusion, no hospitalized
    # print('Before selecting no hospitalized, len(df)', len(df))
    df = df.loc[(df['outpatient'] == 1), :]
    print('After selecting no hospitalized, len(df)', len(df),
          'exclude not outpatient in df_ec_start', (df_ec_start['outpatient'] != 1).sum())

    def ec_no_U099_baseline(_df):
        print('before ec_no_U099_baseline, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[(_df['dx-base@PASC-General'] == 0)]
        n1 = len(_df)
        print('after ec_no_U099_baseline, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))

        print('exclude baseline U099 in df_ec_start', (df_ec_start['dx-base@PASC-General'] > 0).sum())
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

        print('exclude contraindication drugs in df_ec_start -14 - +14', (df_ec_start['pax_contra'] > 0).sum())
        return _df

    def ec_no_severe_conditions_4_pax(_df):
        print('before ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[(_df['PaxExclude-Count'] == 0)]
        print('after ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
        n1 = len(_df)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        print('exclude severe conditions in df_ec_start', (df_ec_start['PaxExclude-Count'] > 0).sum())
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
    df = ec_no_other_covid_treatment(df)

    df_risk = ec_at_least_one_risk_4_pax(df)
    df_norisk = ec_not_at_risk_4_pax(df)

    print('**************build treated patients for AT risk patients')
    # drug initiation within 5 days
    df_risk_pos = df_risk.loc[(df_risk['treat-flag@paxlovid'] > 0), :]
    print('After selecting pax prescription, len(df_risk_pos)', len(df_risk_pos))
    df_risk_pos = df_risk_pos.loc[(df_risk_pos['treat-t2e@paxlovid'] <= 5), :]
    print('After selecting pax prescription within 5 days, len(df_risk_pos)', len(df_risk_pos))
    print('**************build control patients for AT risk patients')
    # non initiation group, no paxlovid
    df_risk_ctrl = df_risk.loc[(df_risk['treat-flag@paxlovid'] == 0), :]
    print('After selecting NO pax prescription, len(df_risk_ctrl)', len(df_risk_ctrl))

    print('**************build treated patients for No risk patients')
    # drug initiation within 5 days
    df_norisk_pos = df_norisk.loc[(df_norisk['treat-flag@paxlovid'] > 0), :]
    print('After selecting pax prescription, len(df_norisk_pos)', len(df_norisk_pos))
    df_norisk_pos = df_norisk_pos.loc[(df_norisk_pos['treat-t2e@paxlovid'] <= 5), :]
    print('After selecting pax prescription within 5 days, len(df_norisk_pos)', len(df_norisk_pos))
    print('**************build control patients for No risk patients')
    # non initiation group, no paxlovid
    df_norisk_ctrl = df_norisk.loc[(df_norisk['treat-flag@paxlovid'] == 0), :]
    print('After selecting NO pax prescription, len(df_norisk_ctrl)', len(df_norisk_ctrl))

    return df_risk_pos, df_norisk_pos, df_risk_ctrl, df_norisk_ctrl


def more_ec_for_cohort_selection_4_ssri(df):
    print('in more_ec_for_cohort_selection, df.shape', df.shape)
    print('Applying more specific/flexible eligibility criteria for cohort selection')

    # select index date
    # print('Before selecting index date from 2022-4-1 to 2023-2-1, len(df)', len(df))
    # print('Before selecting index date from 2022-3-1 to 2023-2-1, len(df)', len(df))
    print('Before selecting index date from *** to 2023-2-1, len(df)', len(df))

    df = df.loc[
         # (df['index date'] >= datetime.datetime(2022, 3, 1, 0, 0)) &
         (df['index date'] <= datetime.datetime(2023, 2, 1, 0, 0)), :]
    # print('After selecting index date from 2022-1-1 to 2023-2-1, len(df)', len(df))
    print('After selecting index date from *** to 2023-2-1, len(df)', len(df))

    df_ec_start = df.copy()
    print('Build df_ec_start for calculating EC proportion, len(df_ec_start)', len(df_ec_start))

    # Exclusion, no hospitalized
    # print('Before selecting no hospitalized, len(df)', len(df))
    df = df.loc[(df['outpatient'] == 1), :]
    print('After selecting no hospitalized, len(df)', len(df),
          'exclude not outpatient in df_ec_start', (df_ec_start['outpatient'] != 1).sum())

    def ec_no_U099_baseline(_df):
        print('before ec_no_U099_baseline, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[(_df['dx-base@PASC-General'] == 0)]
        n1 = len(_df)
        print('after ec_no_U099_baseline, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))

        print('exclude baseline U099 in df_ec_start', (df_ec_start['dx-base@PASC-General'] > 0).sum())
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

        print('exclude contraindication drugs in df_ec_start -14 - +14', (df_ec_start['pax_contra'] > 0).sum())
        return _df

    def ec_no_severe_conditions_4_pax(_df):
        print('before ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[(_df['PaxExclude-Count'] == 0)]
        print('after ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
        n1 = len(_df)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        print('exclude severe conditions in df_ec_start', (df_ec_start['PaxExclude-Count'] > 0).sum())
        return _df

    # def ec_at_least_one_risk_4_pax(_df):
    #     print('before ec_at_least_one_risk_4_pax, _df.shape', _df.shape)
    #     n0 = len(_df)
    #     _df = _df.loc[(_df['age'] >= 50) | (_df['PaxRisk-Count'] > 0)]
    #     n1 = len(_df)
    #     print('after ec_at_least_one_risk_4_pax, _df.shape', _df.shape)
    #     print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    #     return _df

    def ec_at_least_one_risk_4_pax_nopregnant(_df):
        print('before ec_at_least_one_risk_4_pax_nopregnant, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[((_df['age'] >= 50) | (_df['PaxRisk-Count'] > 0)) & (_df["PaxRisk:Pregnancy"] == 0)]
        n1 = len(_df)
        print('after ec_at_least_one_risk_4_pax_nopregnant, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        return _df

    def ec_pregnant_risk_4_pax(_df):
        print('before ec_pregnant_risk_4_pax, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[(_df["PaxRisk:Pregnancy"] == 1)]
        n1 = len(_df)
        print('after ec_pregnant_risk_4_pax, _df.shape', _df.shape)
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
    df = ec_no_other_covid_treatment(df)

    # print('**************build treated patients')
    # # drug initiation within 5 days
    # df_pos = df.loc[(df['treat-flag@paxlovid'] > 0), :]
    # print('After selecting pax prescription, len(df_pos)', len(df_pos))
    # df_pos = df_pos.loc[(df_pos['treat-t2e@paxlovid'] <= 5), :]
    # print('After selecting pax prescription within 5 days, len(df_pos)', len(df_pos))
    #
    # df_pos_risk = ec_at_least_one_risk_4_pax_nopregnant(df_pos.copy())
    # df_pos_norisk = ec_not_at_risk_4_pax(df_pos.copy())
    # df_pos_preg = ec_pregnant_risk_4_pax(df_pos.copy())
    #
    # print('**************build control patients')
    # # non initiation group, no paxlovid
    # df_ctrl = df.loc[(df['treat-flag@paxlovid'] == 0), :]
    # print('After selecting NO pax prescription, len(df_ctrl)', len(df_ctrl))
    #
    # df_ctrl_risk = ec_at_least_one_risk_4_pax_nopregnant(df_ctrl.copy())
    # df_ctrl_norisk = ec_not_at_risk_4_pax(df_ctrl.copy())
    # df_ctrl_preg = ec_pregnant_risk_4_pax(df_ctrl).copy()

    # return df_pos_risk, df_pos_norisk, df_pos_preg, df_ctrl_risk, df_ctrl_norisk, df_ctrl_preg
    return df


def more_ec_for_cohort_selection_4_ssri_part2(df):
    print('in more_ec_for_cohort_selection_risk_norisk_pregnant_part2, df.shape', df.shape)
    df_ec_start = df.copy()
    print('Build df_ec_start for calculating EC proportion, len(df_ec_start)', len(df_ec_start))

    def ec_no_U099_baseline(_df):
        print('before ec_no_U099_baseline, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[(_df['dx-base@PASC-General'] == 0)]
        n1 = len(_df)
        print('after ec_no_U099_baseline, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))

        print('exclude baseline U099 in df_ec_start', (df_ec_start['dx-base@PASC-General'] > 0).sum())
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

        print('exclude contraindication drugs in df_ec_start -14 - +14', (df_ec_start['pax_contra'] > 0).sum())
        return _df

    def ec_no_severe_conditions_4_pax(_df):
        print('before ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[(_df['PaxExclude-Count'] == 0)]
        print('after ec_no_severe_conditions_4_pax, _df.shape', _df.shape)
        n1 = len(_df)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        print('exclude severe conditions in df_ec_start', (df_ec_start['PaxExclude-Count'] > 0).sum())
        return _df

    # def ec_at_least_one_risk_4_pax(_df):
    #     print('before ec_at_least_one_risk_4_pax, _df.shape', _df.shape)
    #     n0 = len(_df)
    #     _df = _df.loc[(_df['age'] >= 50) | (_df['PaxRisk-Count'] > 0)]
    #     n1 = len(_df)
    #     print('after ec_at_least_one_risk_4_pax, _df.shape', _df.shape)
    #     print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
    #     return _df

    def ec_at_least_one_risk_4_pax_nopregnant(_df):
        print('before ec_at_least_one_risk_4_pax_nopregnant, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[((_df['age'] >= 50) | (_df['PaxRisk-Count'] > 0)) & (_df["PaxRisk:Pregnancy"] == 0)]
        n1 = len(_df)
        print('after ec_at_least_one_risk_4_pax_nopregnant, _df.shape', _df.shape)
        print('n0:{}, n1:{}, n1-n0 change:{}'.format(n0, n1, n1 - n0))
        return _df

    def ec_pregnant_risk_4_pax(_df):
        print('before ec_pregnant_risk_4_pax, _df.shape', _df.shape)
        n0 = len(_df)
        _df = _df.loc[(_df["PaxRisk:Pregnancy"] == 1)]
        n1 = len(_df)
        print('after ec_pregnant_risk_4_pax, _df.shape', _df.shape)
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

    # df = ec_no_U099_baseline(df)
    # df = ec_no_severe_conditions_4_pax(df)
    # df = ec_no_other_covid_treatment(df)

    # print('**************build treated patients')
    # # drug initiation within 5 days
    # df_pos = df.loc[(df['treat-flag@paxlovid'] > 0), :]
    # print('After selecting pax prescription, len(df_pos)', len(df_pos))
    # df_pos = df_pos.loc[(df_pos['treat-t2e@paxlovid'] <= 5), :]
    # print('After selecting pax prescription within 5 days, len(df_pos)', len(df_pos))
    #
    # df_pos_risk = ec_at_least_one_risk_4_pax_nopregnant(df_pos.copy())
    # df_pos_norisk = ec_not_at_risk_4_pax(df_pos.copy())
    # df_pos_preg = ec_pregnant_risk_4_pax(df_pos.copy())
    #
    # print('**************build control patients')
    # # non initiation group, no paxlovid
    # df_ctrl = df.loc[(df['treat-flag@paxlovid'] == 0), :]
    # print('After selecting NO pax prescription, len(df_ctrl)', len(df_ctrl))
    #
    # df_ctrl_risk = ec_at_least_one_risk_4_pax_nopregnant(df_ctrl.copy())
    # df_ctrl_norisk = ec_not_at_risk_4_pax(df_ctrl.copy())
    # df_ctrl_preg = ec_pregnant_risk_4_pax(df_ctrl).copy()

    # return df_pos_risk, df_pos_norisk, df_pos_preg, df_ctrl_risk, df_ctrl_norisk, df_ctrl_preg
    return df


def exact_match_on(df_case, df_ctrl, kmatch, cols_to_match, random_seed=0):
    print('len(case)', len(df_case), 'len(ctrl)', len(df_ctrl))
    ctrl_list = []
    n_no_match = 0
    n_fewer_match = 0
    for index, rows in tqdm(df_case.iterrows(), total=df_case.shape[0]):
        # if index == 275 or index == 1796:
        #     print(275, 'debug')
        #     print(1796, 'debug')
        # else:
        #     continue
        boolidx = df_ctrl[cols_to_match[0]] == rows[cols_to_match[0]]
        for c in cols_to_match[1:]:
            boolidx &= df_ctrl[c] == rows[c]
        sub_df = df_ctrl.loc[boolidx, :]
        if len(sub_df) >= kmatch:
            _add_index = sub_df.sample(n=kmatch, replace=False, random_state=random_seed).index
        elif len(sub_df) > 0:
            n_fewer_match += 1
            _add_index = sub_df.sample(frac=1, replace=False, random_state=random_seed).index
            print(len(sub_df), ' match for', index)
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

    print('Done, total {}:{} no match, {} fewer match'.format(len(df_case), n_no_match, n_fewer_match))
    return ctrl_list


if __name__ == "__main__":
    # python screen_dx_recover_pregnancy_cohort2.py --site all --severity pospreg-posnonpreg 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort2_all_pospreg-posnonpreg.txt
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)
    # zz
    # %% Step 1. Build or Load  Data
    print('In screen_ssri_build_pcornet ...')
    print('in build_data, aggregate each site and additional columns...')
    if args.site == 'all':
        sites = ['ochin',
                 'intermountain', 'mcw', 'iowa', 'missouri', 'nebraska', 'utah', 'utsw',
                 'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
                 'ufh', 'emory', 'usf', 'nch', 'miami',
                 'pitt', 'osu', 'psu', 'temple', 'michigan',
                 'ochsner', 'ucsf', 'lsu',
                 'vumc', 'duke', 'musc']

        # for debug
        ## sites = ['wcm', 'mshs', 'columbia']  # , 'montefiore', 'nyu', ]
        # sites = ['wcm', 'mshs', 'columbia', 'montefiore', 'nyu', ]
        # sites = ['wcm']
    elif args.site == 'insight':
        sites = ['wcm', 'montefiore', 'mshs', 'columbia', 'nyu', ]

    elif args.site == 'oneflorida':
        sites = ['ufh', 'emory', 'usf', 'nch', 'miami',]

    else:
        sites = [args.site, ]

    print('len(sites), sites:', len(sites), sites)

    brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
    brainfog_list = list(brainfog_encoding.keys())

    # pasc_flag = (df['dxbrainfog-out@' + pasc].copy() >= 1).astype('int')
    # pasc_t2e = df['dxbrainfog-t2e@' + pasc].astype('float')
    # pasc_baseline = df['dxbrainfog-base@' + pasc]

    df_info_list = []
    df_label_list = []
    df_covs_list = []
    df_outcome_list = []

    df_list = []
    df_ssri_list = []
    for ith, site in tqdm(enumerate(sites)):
        print('Loading: ', ith, site)
        data_file = r'../data/recover/output/{}/matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_{}.csv'.format(
            site, site)
        print('read file from:', data_file)
        # Load Covariates Data
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
                         parse_dates=['index date', 'dob'])
        print('df.shape:', df.shape)

        # # add new columns 2
        # data_file_add = r'../data/recover/output/pregnancy_data/pregnancy_{}.csv'.format(site)
        # print('add columns from:', data_file_add)
        # df_add = pd.read_csv(data_file_add, dtype={'patid': str, 'site': str, 'zip': str},
        #                      parse_dates=['index date', 'dob',
        #                                   'flag_delivery_date',
        #                                   'flag_pregnancy_start_date',
        #                                   'flag_pregnancy_end_date'])
        # print('df_add.shape:', df_add.shape)
        # select_cols = ['patid', 'site', 'covid', 'index date',
        #                'flag_Tier1_ori', 'flag_Tier2_ori', 'flag_Tier3_ori', 'flag_Tier4_ori',
        #                'flag_Tier1', 'flag_Tier2', 'flag_Tier3', 'flag_Tier4', 'flag_inclusion',
        #                'flag_delivery_code', 'flag_exclusion', 'flag_exclusion_dx',
        #                'flag_exclusion_dx_detail', 'flag_exclusion_px', 'flag_exclusion_px_detail',
        #                'flag_exclusion_drg', 'flag_exclusion_drg_detail', 'flag_Z3A_current',
        #                'flag_Z3A_current_dx', 'flag_Z3A_previous', 'flag_Z3A_previous_dx', 'flag_pregnancy',
        #                'flag_delivery_date', 'flag_delivery_type_Spontaneous', 'flag_delivery_type_Cesarean',
        #                'flag_delivery_type_Operative', 'flag_delivery_type_Vaginal',
        #                'flag_delivery_type_Unspecified', 'flag_delivery_type_Other',
        #                'flag_pregnancy_start_date', 'flag_pregnancy_gestational_age',
        #                'flag_pregnancy_end_date', 'flag_maternal_age']
        # df = pd.merge(df, df_add[select_cols], how='left', left_on='patid', right_on='patid', suffixes=('', '_z'), )
        # print('After left merge, merged df.shape:', df.shape)
        #
        # # add new columns
        # # data_file_add = r'../data/recover/output/{}/matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_{}-addCFR-PaxRisk-U099-Hospital-SSRI.csv'.format(
        # #     site, site)
        # # 2024-7-12 add missing 1 ssri
        # # data_file_add = r'../data/recover/output/{}/matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_{}-addCFR-PaxRisk-U099-Hospital-SSRI-v2.csv'.format(
        # #     site, site)
        # # 2024-7-29 add weillbutrin
        # # data_file_add = r'../data/recover/output/{}/matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_{}-addCFR-PaxRisk-U099-Hospital-SSRI-v3.csv'.format(
        # #     site, site)
        #
        # # 2024-9-6 add mental covs, weillbutrin --> bupropion, v5 with all time over all period, base+acute+fup
        # data_file_add = r'../data/recover/output/{}/matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_{}-addCFR-PaxRisk-U099-Hospital-SSRI-v5-withmental.csv'.format(
        #     site, site)
        # print('add columns from:', data_file_add)
        # df_add = pd.read_csv(data_file_add, dtype={'patid': str, 'site': str})
        # df_ssri_list.append(df_add)
        # print('df_add.shape:', df_add.shape)
        # df = pd.merge(df, df_add, how='left', left_on='patid', right_on='patid', suffixes=('', '_y'), )
        # print('After left merge, merged df.shape:', df.shape)


        df_list.append(df)
        print('Done', ith, site)

    # combine all sites and select subcohorts
    print('To concat len(df_list)', len(df_list))
    df = pd.concat(df_list, ignore_index=True)
    # df_ssri = pd.concat(df_ssri_list, ignore_index=True)

    print(r"df['site'].value_counts(sort=False)", df['site'].value_counts(sort=False))
    print(r"df['site'].value_counts()", df['site'].value_counts())

    df = df.loc[df['covid'] == 1, :].copy()

    # add
    df = add_col(df)
    print('add cols, then df.shape:', df.shape)
    out_data_file = 'recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNet-nihtable1-{}.csv'.format(args.site)
    print('dump to', out_data_file)
    df.to_csv(out_data_file)

    print('dump done!')
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
