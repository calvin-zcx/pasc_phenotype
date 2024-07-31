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
                                               '1stwave', 'alpha', 'delta', 'omicron', 'omicronafter', 'preg-pos-neg',
                                               'pospreg-posnonpreg', 'omicronbroad',
                                               'followupanydx',
                                               'trimester1', 'trimester2', 'trimester3', 'delivery1week',
                                               'trimester1-anyfollow', 'trimester2-anyfollow', 'trimester3-anyfollow',
                                               'delivery1week-anyfollow',
                                               'inpatienticu-anyfollow', 'outpatient-anyfollow',
                                               'controlInpatient',
                                               'less35', 'above35',
                                               'bminormal', 'bmioverweight', 'bmiobese',
                                               'pregwithcond',
                                               'pregwithoutcond',
                                               'fullyvac', 'partialvac', 'anyvac', 'novacdata',

                                               ],
                        default='all')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--negative_ratio', type=int, default=10)  # 5
    parser.add_argument('--selectpasc', action='store_true')

    parser.add_argument("--kmatch", type=int, default=3)
    parser.add_argument("--usedx", type=int, default=1)
    parser.add_argument("--useacute", type=int, default=1)
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


def select_subpopulation(df, severity, args):
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
    elif severity == 'followupanydx':
        print('Considering patients with any follow up dx')
        df = df.loc[(df['followupanydx'] == 1), :].copy()
    elif severity == 'less35':
        print('Considering less35 pregnant cohorts')
        df = df.loc[
             (df['pregage:18-<25 years'] == 1) | (df['pregage:25-<30 years'] == 1) | (df['pregage:30-<35 years'] == 1),
             :].copy()
    elif severity == 'above35':
        print('Considering above35 pregnant cohorts')
        df = df.loc[
             (df['pregage:35-<40 years'] == 1) | (df['pregage:40-<45 years'] == 1) | (df['pregage:45-50 years'] == 1),
             :].copy()
    elif severity == 'fullyvac':
        print('Considering fullyvac pregnant cohorts')
        df = df.loc[(df['Fully vaccinated - Pre-index'] == 1), :].copy()
    elif severity == 'partialvac':
        print('Considering partialvac pregnant cohorts')
        df = df.loc[(df['Partially vaccinated - Pre-index'] == 1), :].copy()
    elif severity == 'anyvac':
        print('Considering both fullyvac and partialvac pregnant cohorts')
        df = df.loc[(df['Fully vaccinated - Pre-index'] == 1) | (df['Partially vaccinated - Pre-index'] == 1), :].copy()
    elif severity == 'novacdata':
        print('Considering novacdata pregnant cohorts')
        df = df.loc[(df['No evidence - Pre-index'] == 1), :].copy()

    elif severity == 'bminormal':
        print('Considering bminormal  pregnant cohorts')
        df = df.loc[(df['BMI: 18.5-<25 normal weight'] == 1), :].copy()
    elif severity == 'bmioverweight':
        print('Considering bmioverweight  pregnant cohorts')
        df = df.loc[(df['BMI: 25-<30 overweight '] == 1), :].copy()
    elif severity == 'bmiobese':
        print('Considering bmiobese  pregnant cohorts')
        df = df.loc[(df['BMI: >=30 obese '] == 1), :].copy()

    elif severity == 'pregwithcond':
        print('Considering pregnancy with pre-existing conditions cohorts')
        df = df.loc[(df["DX: Diabetes Type 1"] >= 1) |
                    (df["DX: Diabetes Type 2"] >= 1) |
                    (df["DX: Hypertension"] >= 1) |
                    (df["DX: Asthma"] >= 1) |
                    (df["DX: Severe Obesity  (BMI>=40 kg/m2)"] == 1) |
                    (df['bmi'] >= 40), :].copy()

    elif severity == 'pregwithoutcond':
        print('Considering pregnancy withOUT pre-existing conditions cohorts')
        df = df.loc[~((df["DX: Diabetes Type 1"] >= 1) |
                    (df["DX: Diabetes Type 2"] >= 1) |
                    (df["DX: Hypertension"] >= 1) |
                    (df["DX: Asthma"] >= 1) |
                    (df["DX: Severe Obesity  (BMI>=40 kg/m2)"] == 1) |
                    (df['bmi'] >= 40)), :].copy()

    elif severity in ['trimester1', 'trimester2', 'trimester3', 'delivery1week',
                      'trimester1-anyfollow', 'trimester2-anyfollow', 'trimester3-anyfollow',
                      'delivery1week-anyfollow']:
        print('len(df)', len(df))
        df_case = df.loc[df['flag_pregnancy'] == 1, :]
        df_ctrl = df.loc[df['flag_pregnancy'] == 0, :]
        print('len(df_case)', len(df_case), 'len(df_ctrl)', len(df_ctrl), 'case+ctrl', len(df_case) + len(df_ctrl))
        # https://www.nichd.nih.gov/health/topics/factsheets/pregnancy

        if severity in ['trimester1', 'trimester1-anyfollow']:
            df_case_sub = df_case.loc[df_case['gestational age of infection'] <= 12, :]
        elif severity in ['trimester2', 'trimester2-anyfollow']:
            df_case_sub = df_case.loc[(df_case['gestational age of infection'] <= 28) & (
                    df_case['gestational age of infection'] > 13), :]
        elif severity in ['trimester3', 'trimester3-anyfollow']:
            df_case_sub = df_case.loc[df_case['gestational age of infection'] >= 29, :]
        elif severity in ['delivery1week', 'delivery1week-anyfollow']:
            # _ind_ = np.timedelta64(-7, 'D') <= (df_case['flag_delivery_date'] - df_case['index date']) <= np.timedelta64(7, 'D')
            ind_1 = np.timedelta64(-7, 'D') <= (df_case['flag_delivery_date'] - df_case['index date'])
            ind_2 = (df_case['flag_delivery_date'] - df_case['index date']) <= np.timedelta64(7, 'D')
            ind_and = ind_1 & ind_2
            df_case_sub = df_case.loc[ind_and, :]

        print('len(df_case_sub)', len(df_case_sub), 'len(df_ctrl)', len(df_ctrl), 'df_case_sub+ctrl',
              len(df_case_sub) + len(df_ctrl))
        # features only apply to case/pregant, not for control. For other feature, no need to do this
        if len(df_case_sub) * args.kmatch <= len(df_ctrl):
            df_ctrl_sub = df_ctrl.sample(n=len(df_case_sub) * args.kmatch, replace=False, random_state=args.random_seed)
        else:
            df_ctrl_sub = df_ctrl

        df = pd.concat([df_case_sub, df_ctrl_sub], ignore_index=True)
        df = df.copy()
        print('len(df_case_sub)', len(df_case_sub), 'len(df_ctrl_sub)', len(df_ctrl_sub),
              'df', len(df))

        if severity in ['trimester1-anyfollow', 'trimester2-anyfollow', 'trimester3-anyfollow',
                        'delivery1week-anyfollow']:
            print('Further considering patients with any follow up dx')
            df = df.loc[(df['followupanydx'] == 1), :].copy()
            print('further anyfollow up, df', len(df))
    elif severity == 'inpatienticu-anyfollow':
        print('Considering inpatient/hospitalized including icu cohorts')
        df = df.loc[(df['hospitalized'] == 1) | (df['criticalcare'] == 1), :]
        print('Further considering patients with any follow up dx')
        df = df.loc[(df['followupanydx'] == 1), :].copy()
        print('further anyfollow up, df', len(df))
    elif severity == 'outpatient-anyfollow':
        print('Considering outpatient cohorts')
        df = df.loc[(df['hospitalized'] == 0) & (df['criticalcare'] == 0), :]
        print('Further considering patients with any follow up dx')
        df = df.loc[(df['followupanydx'] == 1), :].copy()
        print('further anyfollow up, df', len(df))
    elif severity == 'controlInpatient':
        print('controlInpatient: len(df)', len(df))
        df_case = df.loc[df['flag_pregnancy'] == 1, :]
        df_ctrl = df.loc[df['flag_pregnancy'] == 0, :]
        print('len(df_case)', len(df_case), 'len(df_ctrl)', len(df_ctrl), 'case+ctrl', len(df_case) + len(df_ctrl))

        df_ctrl = df_ctrl.loc[
                  ((df_ctrl['hospitalized'] == 1) | (df_ctrl['ventilation'] == 1) | (df_ctrl['criticalcare'] == 1)),
                  :].copy()
        print('After selection: len(df_case)', len(df_case), 'len(df_ctrl)', len(df_ctrl), 'case+ctrl',
              len(df_case) + len(df_ctrl))
        # After selection: len(df_case) 29975 len(df_ctrl) 2578 case+ctrl 32553
        # with problem, because control, namely inpatient cohort are too small
        df = pd.concat([df_case, df_ctrl], ignore_index=True)
        df = df.copy()
        print('df', len(df))
    else:
        print('Considering ALL cohorts, no selection')

    # if severity == 'pospreg-posnonpreg':
    #     # select index date
    #     print('Before selecting index date < 2022-6-1, df.shape', df.shape)
    #     df = df.loc[(df['index date'] < datetime.datetime(2022, 6, 1, 0, 0)), :]  # .copy()
    #     print('After selecting index date < 2022-6-1, df.shape', df.shape)
    #
    #     # select age
    #     print('Before selecting age <= 50, df.shape', df.shape)
    #     df = df.loc[df['age'] <= 50, :]  # .copy()
    #     print('After selecting age <= 50, df.shape', df.shape)
    #
    #     # select female
    #     print('Before selecting female, df.shape', df.shape)
    #     df = df.loc[df['Female'] == 1, :]  # .copy()
    #     print('After selecting female, df.shape', df.shape)
    #
    #     # covid positive patients only
    #     print('Before selecting covid+, df.shape', df.shape)
    #     df = df.loc[df['covid'] == 1, :]  # .copy()
    #     print('After selecting covid+, df.shape', df.shape)
    #
    #     # # pregnant patients only
    #     # print('Before selecting pregnant, df.shape', df.shape)
    #     # df = df.loc[df['flag_pregnancy'] == 1, :]#.copy()
    #     # print('After selecting pregnant, df.shape', df.shape)
    #     #
    #     # # infection during pregnancy period
    #     # print('Before selecting infection in gestational period, df.shape', df.shape)
    #     # df = df.loc[(df['index date'] >= df['flag_pregnancy_start_date']) & (
    #     #         df['index date'] <= df['flag_delivery_date'] + datetime.timedelta(days=7)), :].copy()
    #     # print('After selecting infection in gestational period, df.shape', df.shape)

    return df


def more_ec_for_cohort_selection(df):
    print('more_ec_for_cohort_selection, df.shape', df.shape)
    start_time = time.time()
    print('*' * 100)
    print('in more_ec_for_cohort_selection, len(df)', len(df))
    print('Applying more specific/flexible eligibility criteria for cohort selection')
    N = len(df)
    # covid positive patients only
    print('Before selecting covid+, len(df)\n', len(df))
    n = len(df)
    df = df.loc[df['covid'] == 1, :]
    print('After selecting covid+, len(df),\n',
          '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))

    # at least 6 month follow-up, up to 2023-4-30
    n = len(df)
    df = df.loc[(df['index date'] <= datetime.datetime(2022, 10, 31, 0, 0)), :]  # .copy()
    print('After selecting index date <= 2022-10-31, len(df)\n',
          '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))

    # covid+, inclusion windows
    df_general = df.copy()

    # select female
    n = len(df)
    df = df.loc[df['Female'] == 1, :]  # .copy()
    print('After selecting female, len(df)\n',
          '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))

    # select age 18-50
    n = len(df)
    df = df.loc[df['age'] <= 50, :]
    print('After selecting age <= 50, len(df)\n',
          '{}\t{:.2f}%\t{:.2f}%'.format(len(df), len(df) / n * 100, len(df) / N * 100))

    print('Branching building two groups here:')
    # group1: pregnant and covid+ in pregnancy
    # pregnant patients only
    print('Branch-1, building eligible covid infection during pregnancy')
    print('Before selecting, len(df)', len(df))
    df1 = df.loc[df['flag_pregnancy'] == 1, :]
    print('After selecting pregnant females, len(df1)', len(df1))

    # infection during pregnancy period
    df1 = df1.loc[(df1['index date'] >= df1['flag_pregnancy_start_date'] - datetime.timedelta(days=7)) & (
            df1['index date'] <= df1['flag_delivery_date'] + datetime.timedelta(days=7)), :].copy()
    print('After selecting infection in gestational period, len(df1)', len(df1))

    # group2: non-pregnant group
    print('Branch-2, building eligible covid infected Non-pregnant female')
    print('Before selecting, len(df)', len(df))
    df2 = df.loc[(df['flag_pregnancy'] == 0) & (df['flag_exclusion'] == 0), :].copy()
    print('After selecting non-pregnant female, len(df2)', len(df2))

    # group3: pregnant but with ectopic/abortions/etc not expected delivery outcome, should be combined with pregnant group?
    print(
        'Branch-3, building pregnant but with ectopic/abortions/etc not expected delivery outcome, might be combined with group1')
    print('Before selecting, len(df)', len(df))
    df3 = df.loc[(df['flag_exclusion'] == 1), :].copy()
    print('After selecting error-pregnant female, len(df3)', len(df3))  # 391
    ## currently not implement flag_pregnancy_start_date for Pregnancy with abortive outcome
    # df3 = df3.loc[(df3['index date'] >= df3['flag_pregnancy_start_date'] - datetime.timedelta(days=7)) & (
    #         df3['index date'] <= df3['flag_delivery_date'] + datetime.timedelta(days=7)), :].copy()
    # print('After selecting infection in gestational period, len(df3)', len(df3))

    print('*' * 100)
    print('more_ec_for_cohort_selection Done! Time used:',
          time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df_general, df1, df2


def exact_match_on(df_case, df_ctrl, kmatch, cols_to_match, random_seed=0):
    print('Matched on columns:', cols_to_match)
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
            # df_ctrl = df_ctrl.loc[~df_ctrl.index.isin(_add_index)]
            df_ctrl.drop(index=_add_index, inplace=True)
        if len(df_ctrl) == 0:
            break

    print('Done, total {}:{} no match, {} fewer match'.format(len(df_case), n_no_match, n_fewer_match))
    return ctrl_list


# def build_matched_control(df_case, df_contrl, kmatche=1, usedx=True):
#     age_col = ['pregage:18-<25 years',
#                'pregage:25-<30 years',
#                'pregage:30-<35 years',
#                'pregage:35-<40 years',
#                'pregage:40-<45 years',
#                'pregage:45-50 years', ]
#     period_col = ['03/20-06/20', '07/20-10/20', '11/20-02/21',
#                   '03/21-06/21', '07/21-10/21', '11/21-02/22',
#                   '03/22-06/22', '07/22-10/22', ]
#     acute_col = ['outpatient', 'hospitalized', 'icu']
#     # race_col = ['Asian', 'Black or African American', 'White', 'Other']  # , 'Missing'
#     # eth_col = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
#
#     # adi_col = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
#     #            'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100', 'ADIMissing']
#     # dx_col = ["DX: Asthma", "DX: Cancer", "DX: Chronic Kidney Disease",
#     #           "DX: Congestive Heart Failure", "DX: End Stage Renal Disease on Dialysis",
#     #           "DX: Hypertension", "DX: Pregnant",
#     #           ]
#     # dx_col = ["DX: Anemia",
#     #           "DX: Cancer",
#     #           "DX: Chronic Kidney Disease",
#     #           "DX: Diabetes Type 2",
#     #           "DX: Hypertension",
#     #           'DX: Obstructive sleep apnea',
#     #           "MEDICATION: Corticosteroids",
#     #           "MEDICATION: Immunosuppressant drug",
#     #           ]
#     dx_col = ["DX: Anemia",
#               "Type 1 or 2 Diabetes Diagnosis",
#               "DX: Hypertension",
#               "autoimmune/immune suppression",
#               "DX: Mental Health Disorders",
#               "Severe Obesity",
#               "DX: Asthma"
#               ]
#
#     cci_score = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
#     # cols_to_match = ['site', ] + age_col + period_col + acute_col + race_col + eth_col
#     cols_to_match = ['pcornet', ] + age_col + period_col + acute_col
#     if usedx:
#         cols_to_match += dx_col
#
#     ctrl_list = exact_match_on(df_case.copy(), df_contrl.copy(), kmatche, cols_to_match, )
#
#     print('len(ctrl_list)', len(ctrl_list))
#     neg_selected = pd.Series(False, index=df_contrl.index)
#     neg_selected[ctrl_list] = True
#     df_ctrl_matched = df_contrl.loc[neg_selected, :]
#     print('len(df_case):', len(df_case),
#           'len(df_contrl):', len(df_contrl),
#           'len(df_ctrl_matched):', len(df_ctrl_matched), )
#     return df_ctrl_matched.copy()


def add_any_pasc(df, exclude_list=[]):
    # pre-process PASC info
    print('in add_any_pasc, exlcude_list:', len(exclude_list), exclude_list)

    df_pasc_info = pd.read_excel(r'../prediction/output/causal_effects_specific_withMedication_v3.xlsx',
                                 sheet_name='diagnosis')
    pasc_simname = {}
    pasc_organ = {}
    for index, rows in df_pasc_info.iterrows():
        pasc_simname[rows['pasc']] = (rows['PASC Name Simple'], rows['Organ Domain'])
        pasc_organ[rows['pasc']] = rows['Organ Domain']

    # pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    pasc_list_raw = df_pasc_info.loc[df_pasc_info['selected_narrow'] == 1, 'pasc'].to_list()
    pasc_list = []
    for x in pasc_list_raw:
        if x in exclude_list:
            print('Exclude condition:', x)
        else:
            pasc_list.append(x)

    print('len(pasc_list_raw)', len(pasc_list_raw), 'len(pasc_list)', len(pasc_list))
    for p in pasc_list:
        df[p + '_pasc_flag'] = 0
    df['any_pasc_flag'] = 0
    df['any_pasc_type'] = np.nan
    df['any_pasc_t2e'] = 180  # np.nan
    df['any_pasc_txt'] = ''
    df['any_pasc_baseline'] = 0  # placeholder for screening, no special meaning, null column
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

        if len(t2e_list) > 0:
            df.loc[index, 'any_pasc_flag'] = 1
            df.loc[index, 'any_pasc_t2e'] = np.min(t2e_list)
            df.loc[index, 'any_pasc_txt'] = pasc_1_text
        else:
            df.loc[index, 'any_pasc_flag'] = 0
            df.loc[index, 'any_pasc_t2e'] = rows[['dx-t2e@' + p for p in pasc_list]].max()
    return df


def feature_process_additional(df):
    print('feature_process_additional, df.shape', df.shape)
    start_time = time.time()
    df['inpatient'] = ((df['hospitalized'] == 1) & (df['ventilation'] == 0) & (df['criticalcare'] == 0)).astype('int')
    df['icu'] = (((df['hospitalized'] == 1) & (df['ventilation'] == 1)) | (df['criticalcare'] == 1)).astype('int')
    df['inpatienticu'] = ((df['hospitalized'] == 1) | (df['criticalcare'] == 1)).astype('int')
    df['outpatient'] = ((df['hospitalized'] == 0) & (df['criticalcare'] == 0)).astype('int')
    #
    df["Type 1 or 2 Diabetes Diagnosis"] = (
            ((df["DX: Diabetes Type 1"] >= 1).astype('int') + (df["DX: Diabetes Type 2"] >= 1).astype('int')) >= 1
    ).astype('int')

    df["autoimmune/immune suppression"] = (
            (df['DX: Inflammatory Bowel Disorder'] >= 1) | (df['DX: Lupus or Systemic Lupus Erythematosus'] >= 1) |
            (df['DX: Rheumatoid Arthritis'] >= 1) |
            (df["MEDICATION: Corticosteroids"] >= 1) | (df["MEDICATION: Immunosuppressant drug"] >= 1)
    ).astype('int')

    df["Severe Obesity"] = ((df["DX: Severe Obesity  (BMI>=40 kg/m2)"] >= 1) | (df['bmi'] >= 40)).astype('int')

    df.loc[:, r"DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"] = \
        ((df.loc[:, r'DX: Hypertension'] >= 1) & (
                (df.loc[:, r'DX: Diabetes Type 1'] >= 1) | (df.loc[:, r'DX: Diabetes Type 2'] >= 1))).astype('int')

    # Baseline covs
    selected_cols = [x for x in df.columns if (
            x.startswith('DX:') or
            x.startswith('MEDICATION:') or
            x.startswith('CCI:') or
            x.startswith('obc:')
    )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    # Incident outcome part - baseline part have been binarized already
    selected_cols = [x for x in df.columns if
                     (x.startswith('dx-out@') or
                      x.startswith('dxadd-out@') or
                      x.startswith('dxbrainfog-out@') or
                      x.startswith('covidmed-out@') or
                      x.startswith('smm-out@')
                      )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    df.loc[df['death t2e'] < 0, 'death'] = 0
    df.loc[df['death t2e'] < 0, 'death t2e'] = 9999

    df['gestational age at delivery'] = np.nan
    df['gestational age of infection'] = np.nan
    df['preterm birth'] = np.nan

    # ['flag_delivery_type_Spontaneous', 'flag_delivery_type_Cesarean',
    # 'flag_delivery_type_Operative', 'flag_delivery_type_Vaginal', 'flag_delivery_type_other-unsepc',]
    df['flag_delivery_type_other-unsepc'] = (
            (df['flag_delivery_type_Other'] + df['flag_delivery_type_Unspecified']) >= 1).astype('int')
    df['flag_delivery_type_Vaginal-Spontaneous'] = (
            (df['flag_delivery_type_Spontaneous'] + df['flag_delivery_type_Vaginal']) >= 1).astype('int')
    df['flag_delivery_type_Cesarean-Operative'] = (
            (df['flag_delivery_type_Cesarean'] + df['flag_delivery_type_Operative']) >= 1).astype('int')

    df['cci_quan:0'] = 0
    df['cci_quan:1-2'] = 0
    df['cci_quan:3-4'] = 0
    df['cci_quan:5-10'] = 0
    df['cci_quan:11+'] = 0

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

    print('feature_process_additional Done! Time used:',
          time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df


def feature_process_4_subgroup(df):
    print('feature_process_4_subgroup, df.shape', df.shape)
    start_time = time.time()

    print('feature_process_4_subgroup Done! Time used:',
          time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df


if __name__ == "__main__":
    # python screen_dx_recover_pregnancy_cohort3_iptw.py --site all --severity all --kmatch 1 --usedx 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_iptw_kmatch1-useSelectdx1.txt
    # python screen_dx_recover_pregnancy_cohort3_iptw.py --site all --severity all --kmatch 3 --usedx 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_iptw_kmatch3-useSelectdx1-V2.txt
    # python screen_dx_recover_pregnancy_cohort3_iptw.py --site all --severity all --kmatch 5 --usedx 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_iptw_kmatch5-useSelectdx1.txt
    # python screen_dx_recover_pregnancy_cohort3_iptw.py --site all --severity all --kmatch 10 --usedx 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_iptw_kmatch10-useSelectdx1.txt

    # python screen_dx_recover_pregnancy_cohort3_iptw.py --site all --severity all --kmatch 1 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_iptw_kmatch1-usedx0.txt
    # python screen_dx_recover_pregnancy_cohort3_iptw.py --site all --severity all --kmatch 3 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_iptw_kmatch3-usedx0.txt
    # python screen_dx_recover_pregnancy_cohort3_iptw.py --site all --severity all --kmatch 5 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_iptw_kmatch5-usedx0.txt
    # python screen_dx_recover_pregnancy_cohort3_iptw.py --site all --severity all --kmatch 10 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_iptw_kmatch10-usedx0.txt

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)
    # step 0: preprocess and dump data, in another file

    # step 1: load data
    print('Step 1: load data')
    df1, df2 = utils.load(r'../data/recover/output/pregnancy_output/_selected_preg_cohort_1-2.pkl')
    print('Select matched cohort, kmatch:', args.kmatch, 'usedx:', args.usedx)

    # if args.usedx == 0:
    #     match_file_name = r'../data/recover/output/pregnancy_output/_selected_preg_cohort2-matched-k{}-usedx0.pkl'.format(
    #         args.kmatch)
    # elif args.usedx > 0:
    #     match_file_name = r'../data/recover/output/pregnancy_output/_selected_preg_cohort2-matched-k{}-useSelectdx{}.pkl'.format(
    #         args.kmatch, args.usedx)
    # else:
    #     raise ValueError
    if args.useacute == 0:
        match_file_name = r'../data/recover/output/pregnancy_output/_selected_preg_cohort2-matched-k{}-useSelectdx1-useacute0.pkl'.format(
            args.kmatch)
    elif args.useacute > 0:
        match_file_name = r'../data/recover/output/pregnancy_output/_selected_preg_cohort2-matched-k{}-useSelectdx1-useacute1V2.pkl'.format(
            args.kmatch)
    else:
        raise ValueError

    print('load matched file:', match_file_name)
    df2_matched = utils.load(match_file_name)
    print('len(df1)', len(df1),
          'len(df2)', len(df2),
          'len(df2_matched)', len(df2_matched))

    print('df1.shape', df1.shape,
          'df2.shape', df2.shape,
          'df2_matched.shape', df2_matched.shape)

    if args.usedx == 0:
        col1_add = "autoimmune/immune suppression"
        df_col1_add = (
                (df2_matched['DX: Inflammatory Bowel Disorder'] >= 1) | (
                df2_matched['DX: Lupus or Systemic Lupus Erythematosus'] >= 1) |
                (df2_matched['DX: Rheumatoid Arthritis'] >= 1) |
                (df2_matched["MEDICATION: Corticosteroids"] >= 1) | (
                        df2_matched["MEDICATION: Immunosuppressant drug"] >= 1)
        ).astype('int')

        col2_add = "Severe Obesity"
        df_col2_add = ((df2_matched["DX: Severe Obesity  (BMI>=40 kg/m2)"] >= 1) | (df2_matched['bmi'] >= 40)).astype(
            'int')

        if col1_add not in df2_matched.columns:
            df2_matched.insert(df2_matched.columns.get_loc('gestational age at delivery'),
                               col1_add, df_col1_add)
        if col2_add not in df2_matched.columns:
            df2_matched.insert(df2_matched.columns.get_loc('gestational age at delivery'),
                               col2_add, df_col2_add)

        print('After adding columns',
              'df1.shape', df1.shape,
              'df2.shape', df2.shape,
              'df2_matched.shape', df2_matched.shape)

    # ## step 1.5 add CFR columns, also need boolean operations!
    # df_add = pd.read_csv('recover29Nov27_covid_pos_addCFR_only_4_pregnancy.csv',
    #                      dtype={'patid': str, 'site': str})
    df_add = pd.read_csv('recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNetPax.csv',
                         dtype={'patid': str, 'site': str})
    print('df1.shape:', df1.shape)
    df1 = pd.merge(df1, df_add, how='left', left_on=['site', 'patid'], right_on=['site', 'patid'],
                   suffixes=('', '_y'), )
    print('after merge CFR columns, df1.shape:', df1.shape)

    # print('df2.shape:', df2.shape)
    # df2 = pd.merge(df2, df_add, how='left', left_on=['site', 'patid'], right_on=['site', 'patid'],
    #                suffixes=('', '_y'), )
    # print('after merge CFR columns, df2.shape:', df2.shape)

    print('df2_matched.shape:', df2_matched.shape)
    df2_matched = pd.merge(df2_matched, df_add, how='left', left_on=['site', 'patid'], right_on=['site', 'patid'],
                           suffixes=('', '_y'), )
    print('after merge CFR columns, df2_matched.shape:', df2_matched.shape)

    # _col_check = pd.DataFrame(df1.columns).to_csv('preg_col_check.csv')

    # combine df1 and df2 into df
    df = pd.concat([df1, df2_matched], ignore_index=True)

    print('Before select_subpopulation, len(df)', len(df))
    df = select_subpopulation(df, args.severity, args)
    print('After select_subpopulation, len(df)', len(df))

    # some additional feature processing
    selected_cols = [x for x in df.columns if x.startswith('dxCFR-out@')]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    # data clean for <0 error death records, and add censoring to the death time to event columns

    df.loc[df['death t2e'] < 0, 'death'] = 0
    df.loc[df['death t2e'] < 0, 'death t2e'] = 9999

    # death in [0, 180). 1: evnt, 0: censored, censored at 180. death at 180, not counted, thus use <
    df['death all'] = ((df['death'] == 1) & (df['death t2e'] >= 0) & (df['death t2e'] < 180)).astype('int')
    df['death t2e all'] = df['death t2e'].clip(lower=0, upper=180)
    df.loc[df['death all'] == 0, 'death t2e all'] = df['maxfollowup'].clip(lower=0, upper=180)

    # death in [0, 30). 1: evnt, 0: censored, censored at 30. death at 30, not counted, thus use <
    df['death acute'] = ((df['death'] == 1) & (df['death t2e'] < 30)).astype('int')
    df['death t2e acute'] = df['death t2e all'].clip(upper=30)

    # death in [30, 180).  1:event, 0: censored. censored at 180 or < 30, say death at 20, flag is 0, time is 20
    df['death postacute'] = ((df['death'] == 1) & (df['death t2e'] >= 30) & (df['death t2e'] < 180)).astype('int')
    df['death t2e postacute'] = df['death t2e all']

    #
    df['hospitalization-acute-flag'] = (df['hospitalization-acute-flag'] >= 1).astype('int')
    df['hospitalization-acute-t2e'] = df['hospitalization-acute-t2e'].clip(upper=31)
    df['hospitalization-postacute-flag'] = (df['hospitalization-postacute-flag'] >= 1).astype('int')

    #

    # step 2: load and preprocess PASC info
    print('Step 2: load and preprocess PASC info')
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
    _exclude_list = ['Pressure ulcer of skin', 'Anemia',
                     'Other specified and unspecified skin disorders', 'Circulatory signs and symptoms',
                     'Other specified and unspecified gastrointestinal disorders',
                     'Abdominal pain and other digestive/abdomen signs and symptoms',
                     'Fluid and electrolyte disorders',
                     'Acute phlebitis; thrombophlebitis and thromboembolism',
                     'Acute pulmonary embolism',
                     'Headache; including migraine']
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

    # df = add_any_pasc(df, exclude_list=['Anemia',
    #                                     'Acute phlebitis; thrombophlebitis and thromboembolism',
    #                                     'Acute pulmonary embolism'])

    # reuse pasc select command
    # also add or revised some covs, e.g. pre-di cov is wrong if follow dx
    # df = add_any_pasc(df, exclude_list=['Anemia', ])

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

    print('Severity cohorts:', args.severity, 'df.shape:', df.shape, )

    print(r"df.loc[df['flag_pregnancy']==1, 'any_pasc_flag'].sum():",
          df.loc[df['flag_pregnancy'] == 1, 'any_pasc_flag'].sum(),
          r"df.loc[df['flag_pregnancy']==1, 'any_pasc_flag'].mean():",
          df.loc[df['flag_pregnancy'] == 1, 'any_pasc_flag'].mean())

    print(r"df.loc[df['flag_pregnancy']==0, 'any_pasc_flag'].sum():",
          df.loc[df['flag_pregnancy'] == 0, 'any_pasc_flag'].sum(),
          r"df.loc[df['flag_pregnancy']==0, 'any_pasc_flag'].mean():",
          df.loc[df['flag_pregnancy'] == 0, 'any_pasc_flag'].mean())

    print('Cohort build Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # step 3: select covs, labels, outcomes for analysis
    print('Step 3: select covs, labels, outcomes for analysis')
    col_names = pd.Series(df.columns)
    df_info = df[['patid', 'site', 'index date', 'hospitalized',
                  'ventilation', 'criticalcare', 'maxfollowup', 'death', 'death t2e',
                  'flag_pregnancy', 'flag_delivery_date', 'flag_pregnancy_start_date',
                  'flag_pregnancy_gestational_age', 'flag_pregnancy_end_date', 'flag_maternal_age',
                  '03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21',
                  '07/21-10/21', '11/21-02/22', '03/22-06/22', '07/22-10/22', '11/22-02/23',
                  '03/23-06/23', '07/23-10/23', '11/23-02/24',
                  ]]  # 'Unnamed: 0',

    df_label = df['flag_pregnancy']

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

    covs_columns = [  #'inpatient', 'icu', 'outpatient',
                       'ventilation', 'criticalcare',
                       "Type 1 or 2 Diabetes Diagnosis", "autoimmune/immune suppression", "Severe Obesity", ] + \
                   [x for x in
                    list(df.columns)[
                    df.columns.get_loc('pregage:18-<25 years'):(df.columns.get_loc('No evidence - Post-index'))]
                    if (not x.startswith('YM:')) and not x.startswith('RUCA1') and (
                            x not in
                            ['ZIPMissing', '11/22-02/23', '03/23-06/23', '07/23-10/23', '11/23-02/24',
                             'Female', 'Male', 'Other/Missing', 'DX: Pregnant',
                             'DX: Hypertension and Type 1 or 2 Diabetes Diagnosis',
                             'No evidence - Post-index', 'Fully vaccinated - Post-index',
                             'Partially vaccinated - Post-index',
                             'outpatient visits 0', 'outpatient visits 1-2',
                             'outpatient visits 3-4', 'outpatient visits >=5',
                             ])
                    ]

    # days = (df['index date'] - datetime.datetime(2020, 3, 1, 0, 0)).apply(lambda x: x.days)
    # days = np.array(days).reshape((-1, 1))
    # # days_norm = (days - days.min())/(days.max() - days.min())
    # # spline = SplineTransformer(degree=3, n_knots=7)
    # spline = SplineTransformer(degree=3, n_knots=5)
    #
    # days_sp = spline.fit_transform(np.array(days))  # identical
    # # days_norm_sp = spline.fit_transform(days_norm) # identical
    #
    # print('len(covs_columns):', len(covs_columns))
    #
    # # delet old date feature and use spline
    # covs_columns = [x for x in covs_columns if x not in
    #                 ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21',
    #                  '07/21-10/21', '11/21-02/22', '03/22-06/22', '07/22-10/22', '11/22-02/23']]
    # print('after delete 8 days len(covs_columns):', len(covs_columns))

    #
    # new_day_cols = ['days_splie_{}'.format(i) for i in range(days_sp.shape[1])]
    # covs_columns += new_day_cols
    # print('after adding {} days len(covs_columns):'.format(days_sp.shape[1]), len(covs_columns))
    # for i in range(days_sp.shape[1]):
    #     print('add', i, new_day_cols[i])
    #     df_covs[new_day_cols[i]] = days_sp[:, i]

    # # days between pregnancy and infection,  only make sense when comparison groups are also pregnant
    # days_since_preg = (df['index date'] - df['flag_pregnancy_start_date']).apply(lambda x: x.days)
    # days_since_preg = np.array(days_since_preg).reshape((-1, 1))
    # spline = SplineTransformer(degree=3, n_knots=5)
    # days_since_preg_sp = spline.fit_transform(np.array(days_since_preg))  # identical
    #
    # new_days_since_preg_cols = ['days_since_preg_splie_{}'.format(i) for i in range(days_since_preg_sp.shape[1])]
    # covs_columns += new_days_since_preg_cols
    # print('after adding {} days len(covs_columns):'.format(days_since_preg_sp.shape[1]), len(covs_columns))
    # for i in range(days_since_preg_sp.shape[1]):
    #     print('add', i, new_days_since_preg_cols[i])
    #     df_covs[new_days_since_preg_cols[i]] = days_since_preg_sp[:, i]

    # df_covs_list.append(df_covs)

    print('len(covs_columns):', len(covs_columns), covs_columns)
    df_covs = df.loc[:, covs_columns].astype('float')
    print('df.shape:', df.shape, 'df_covs.shape:', df_covs.shape)

    print('all',
          'df.shape', df.shape,
          'df_info.shape:', df_info.shape,
          'df_label.shape:', df_label.shape,
          'df_covs.shape:', df_covs.shape)
    print('Done load data! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # df = pd.concat(df_outcome_list, ignore_index=True)
    # df_info = pd.concat(df_info_list, ignore_index=True)
    # df_label = pd.concat(df_label_list, ignore_index=True)
    # df_covs = pd.concat(df_covs_list, ignore_index=True)
    # df = df_outcome

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

    SMMpasc_encoding = utils.load(r'../data/mapping/SMMpasc_index_mapping.pkl')
    SMMpasc_list = list(SMMpasc_encoding.keys())

    # smm outcomes only make sense when comparison groups are also pregnant
    selected_screen_list = (['any_pasc', 'PASC-General',
                             'death', 'death_acute', 'death_postacute',
                             'any_CFR',
                             'hospitalization_acute', 'hospitalization_postacute'] +
                            CFR_list + pasc_list + addedPASC_list + brainfog_list)  # + SMMpasc_list
    causal_results = []
    results_columns_name = []
    # for i, pasc in tqdm(enumerate(pasc_encoding.keys(), start=1), total=len(pasc_encoding)):
    for i, pasc in tqdm(enumerate(selected_screen_list, start=1), total=len(selected_screen_list)):
        print('\n In c:', i, pasc)
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
        elif pasc in CFR_list:
            pasc_flag = (df['dxCFR-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['dxCFR-t2e@' + pasc].astype('float')
            pasc_baseline = df['dxCFR-base@' + pasc]
        elif pasc == 'any_CFR':
            pasc_flag = df['any_CFR_flag'].astype('int')
            pasc_t2e = df['any_CFR_t2e'].astype('float')
            pasc_baseline = df['any_CFR_baseline']
        elif pasc.startswith('smm'):
            pasc_flag = (df['smm-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['smm-t2e@' + pasc].astype('float')
            pasc_baseline = df['smm-base@' + pasc]

        # # considering competing risks
        # death_flag = df['death']
        # death_t2e = df['death t2e']
        # pasc_flag.loc[(death_t2e == pasc_t2e)] = 2
        # print('#death:', (death_t2e == pasc_t2e).sum(), ' #death in covid+:', df_label[(death_t2e == pasc_t2e)].sum(),
        #       'ratio of death in covid+:', df_label[(death_t2e == pasc_t2e)].mean())

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
        # Select negative: pos : neg = 1:2 for IPTW

        covid_label = df_label[idx]  # actually current is the pregnant label

        n_covid_pos = covid_label.sum()
        n_covid_neg = (covid_label == 0).sum()
        # print('n_covid_pos:', n_covid_pos, 'n_covid_neg:', n_covid_neg, )
        print('n pregnant:', n_covid_pos, 'n not pregnant:', n_covid_neg, )

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
        # pasc_t2e[pasc_t2e <= 30] = 30

        print('pasc_flag.value_counts():\n', pasc_flag.value_counts())
        print(i, pasc, '-- Selected cohorts {}/{} ({:.2f}%), covid pos:neg = {}:{} sample ratio -/+={}, '
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
            'C': 10 ** np.arange(-2, 1.5, 0.5),
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
        out_file_balance = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx{}k{}useacute{}-V3-{}/{}-{}-results.csv'.format(
            args.usedx,
            args.kmatch,
            args.useacute,
            args.severity,
            i,
            pasc.replace(':', '-').replace('/', '-'))
        utils.check_and_mkdir(out_file_balance)
        model.results.to_csv(out_file_balance)  # args.save_model_filename +

        df_summary = summary_covariate(covs_array, covid_label, iptw, smd, smd_weighted, before, after)
        df_summary.to_csv(
            '../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx{}k{}useacute{}-V3-{}/{}-{}-evaluation_balance.csv'.format(
                args.usedx,
                args.kmatch,
                args.useacute,
                args.severity,
                i, pasc.replace(':', '-').replace('/', '-')))

        dfps = pd.DataFrame({'ps': ps, 'iptw': iptw, 'pregnancy': covid_label})

        dfps.to_csv(
            '../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx{}k{}useacute{}-V3-{}/{}-{}-evaluation_ps-iptw.csv'.format(
                args.usedx,
                args.kmatch,
                args.useacute,
                args.severity,
                i, pasc.replace(':', '-').replace('/', '-')))
        try:
            figout = r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx{}k{}useacute{}-V3-{}/{}-{}-PS.png'.format(
                args.usedx,
                args.kmatch,
                args.useacute,
                args.severity,
                i, pasc.replace(':', '-').replace('/', '-'))
            print('Dump ', figout)

            ax = plt.subplot(111)
            sns.histplot(
                dfps, x="ps", hue="pregnancy", element="step",
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
            fig_outfile=r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx{}k{}useacute{}-V3-{}/{}-{}-km.png'.format(
                args.usedx,
                args.kmatch,
                args.useacute,
                args.severity,
                i, pasc.replace(':', '-').replace('/', '-')),
            title=pasc,
            legends={'case': 'Covid Pos Pregnant', 'control': 'Covid Pos Non-pregnant'})

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

            if i % 5 == 0:
                pd.DataFrame(causal_results, columns=results_columns_name). \
                    to_csv(
                    r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx{}k{}useacute{}-V3-{}/causal_effects_specific-snapshot-{}.csv'.format(
                        args.usedx,
                        args.kmatch,
                        args.useacute,
                        args.severity,
                        i))
        except:
            print('Error in ', i, pasc)
            df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

            df_causal.to_csv(
                r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx{}k{}useacute{}-V3-{}/causal_effects_specific-ERRORSAVE.csv'.format(
                    args.usedx,
                    args.kmatch, args.useacute, args.severity, ))

        print('done one pasc, time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

    df_causal.to_csv(
        r'../data/recover/output/pregnancy_output/POSpreg_vs_posnon-usedx{}k{}useacute{}-V3-{}/causal_effects_specific.csv'.format(
            args.usedx,
            args.kmatch, args.useacute, args.severity, ))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
