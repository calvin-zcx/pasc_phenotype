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
                                               'pospreg-posnonpreg'],
                        default='all')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--negative_ratio', type=int, default=10)  # 5
    parser.add_argument('--selectpasc', action='store_true')

    parser.add_argument("--kmatch", type=int, default=3)
    parser.add_argument("--usedx", type=int, default=1)  # useacute
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
    print('Matched on columns, len(cols_to_match):', len(cols_to_match), cols_to_match)
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


def build_matched_control(df_case, df_contrl, kmatche=1, usedx=True, useacute=True):
    age_col = ['pregage:18-<25 years',
               'pregage:25-<30 years',
               'pregage:30-<35 years',
               'pregage:35-<40 years',
               'pregage:40-<45 years',
               'pregage:45-50 years', ]
    period_col = ['03/20-06/20', '07/20-10/20', '11/20-02/21',
                  '03/21-06/21', '07/21-10/21', '11/21-02/22',
                  '03/22-06/22', '07/22-10/22', ]
    # acute_col = ['outpatient', 'hospitalized', 'icu']
    acute_col = ['ventilation', 'criticalcare', ]
    # race_col = ['Asian', 'Black or African American', 'White', 'Other']  # , 'Missing'
    # eth_col = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']

    # adi_col = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
    #            'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100', 'ADIMissing']
    # dx_col = ["DX: Asthma", "DX: Cancer", "DX: Chronic Kidney Disease",
    #           "DX: Congestive Heart Failure", "DX: End Stage Renal Disease on Dialysis",
    #           "DX: Hypertension", "DX: Pregnant",
    #           ]
    # dx_col = ["DX: Anemia",
    #           "DX: Cancer",
    #           "DX: Chronic Kidney Disease",
    #           "DX: Diabetes Type 2",
    #           "DX: Hypertension",
    #           'DX: Obstructive sleep apnea',
    #           "MEDICATION: Corticosteroids",
    #           "MEDICATION: Immunosuppressant drug",
    #           ]
    dx_col = [  # "DX: Anemia",
        "Type 1 or 2 Diabetes Diagnosis",
        "DX: Hypertension",
        "autoimmune/immune suppression",
        "DX: Mental Health Disorders",
        "Severe Obesity",
        "DX: Asthma"
    ]

    cci_score = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    # cols_to_match = ['site', ] + age_col + period_col + acute_col + race_col + eth_col
    cols_to_match = ['pcornet', ] + age_col + period_col  # + acute_col

    if useacute:
        cols_to_match += acute_col

    if usedx:
        cols_to_match += dx_col

    ctrl_list = exact_match_on(df_case.copy(), df_contrl.copy(), kmatche, cols_to_match, )

    print('len(ctrl_list)', len(ctrl_list))
    neg_selected = pd.Series(False, index=df_contrl.index)
    neg_selected[ctrl_list] = True
    df_ctrl_matched = df_contrl.loc[neg_selected, :]
    print('len(df_case):', len(df_case),
          'len(df_contrl):', len(df_contrl),
          'len(df_ctrl_matched):', len(df_ctrl_matched), )
    return df_ctrl_matched.copy()


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

    df.loc[df['death t2e'] < 0, 'death t2e'] = 9999
    df.loc[df['death t2e'] < 0, 'death'] = 0

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


if __name__ == "__main__":
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 1 --usedx 1 --useacute 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch1-useSelectdx1-useacute1.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 3 --usedx 1 --useacute 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch3-useSelectdx1-useacute1.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 5 --usedx 1 --useacute 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch5-useSelectdx1-useacute1.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 10 --usedx 1 --useacute 1 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch10-useSelectdx1-useacute1.txt

    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 1 --usedx 1 --useacute 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch1-useSelectdx1-useacute0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 3 --usedx 1 --useacute 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch3-useSelectdx1-useacute0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 5 --usedx 1 --useacute 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch5-useSelectdx1-useacute0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 10 --usedx 1 --useacute 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch10-useSelectdx1-useacute0.txt

    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 1 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch1-usedx0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 3 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch3-usedx0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 5 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch5-usedx0.txt
    # python screen_dx_recover_pregnancy_cohort3_buildcohort.py --site all --severity all --kmatch 10 --usedx 0 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort3_buildcohort_kmatch10-usedx0.txt

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)

    # %% 1. Load  Data
    print('In cohorts_characterization_build_data...')

    # if args.site == 'all':
    #     sites = ['ochin',
    #              'intermountain', 'mcw', 'iowa', 'missouri', 'nebraska', 'utah', 'utsw',
    #              'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
    #              'ufh', 'emory', 'usf', 'nch', 'miami',
    #              'pitt', 'osu', 'psu', 'temple', 'michigan',
    #              'ochsner', 'ucsf', 'lsu',
    #              'vumc', 'duke', 'musc']
    #
    #     df_site = pd.read_excel(r'../prerecover/RECOVER Adult Site schemas_edit.xlsx')
    #     _site_network = df_site[['Schema name', 'pcornet']].values.tolist()
    #     site_network = {x[0].strip(): x[1].strip() for x in _site_network}
    #     # sites = ['wcm', 'montefiore', 'mshs', ]
    #     # sites = ['wcm', ]
    #     # sites = ['pitt', ]
    #     print('len(sites), sites:', len(sites), sites)
    # else:
    #     sites = [args.site, ]
    #
    # df_info_list = []
    # df_label_list = []
    # df_covs_list = []
    # df_outcome_list = []
    #
    # df_list = []
    # for ith, site in tqdm(enumerate(sites), total=len(sites)):
    #     print('Loading: ', ith, site)
    #     # matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_mshs + pregnancy tag afterwards
    #     data_file = r'../data/recover/output/pregnancy_data/pregnancy_{}.csv'.format(site)
    #     # Load Covariates Data
    #     print('Load data covariates file:', data_file)
    #     df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
    #                      parse_dates=['index date', 'dob',
    #                                   'flag_delivery_date',
    #                                   'flag_pregnancy_start_date',
    #                                   'flag_pregnancy_end_date'])
    #     # because a patid id may occur in multiple sites. patid were site specific
    #     df['pcornet'] = site_network[site]
    #     print('df.shape:', df.shape)
    #     df_list.append(df)
    #
    # # combine all sites and select subcohorts
    # df = pd.concat(df_list, ignore_index=True)
    #
    # # print(r"df['site'].value_counts(sort=False)", df['site'].value_counts(sort=False))
    # print(r"df['site'].value_counts()", df['site'].value_counts())
    # print('over all: df.shape:', df.shape)
    # print('Pregnant in all:',
    #       len(df),
    #       df['flag_pregnancy'].sum(),
    #       df['flag_pregnancy'].mean())
    # print('Pregnant in pos:',
    #       len(df.loc[df['covid'] == 1, :]),
    #       df.loc[df['covid'] == 1, 'flag_pregnancy'].sum(),
    #       df.loc[df['covid'] == 1, 'flag_pregnancy'].mean())
    # print('Pregnant in neg:',
    #       len(df.loc[df['covid'] == 0, :]),
    #       df.loc[df['covid'] == 0, 'flag_pregnancy'].sum(),
    #       df.loc[df['covid'] == 0, 'flag_pregnancy'].mean())
    # print('Pregnant excluded special cases in all:',
    #       len(df),
    #       df['flag_exclusion'].sum(),
    #       df['flag_exclusion'].mean())
    #
    # # %% 2. Cohort building
    # df = select_subpopulation(df, args.severity)
    # df_general, df1, df2 = more_ec_for_cohort_selection(df)
    # zz
    # df1 = feature_process_additional(df1)
    # df2 = feature_process_additional(df2)
    #
    # utils.check_and_mkdir(r'../data/recover/output/pregnancy_output/')
    # df1.to_csv(r'../data/recover/output/pregnancy_output/covidpos_eligible_pregnant.csv')
    # df2.to_csv(r'../data/recover/output/pregnancy_output/covidpos_eligible_Non-pregnant.csv')
    # utils.dump((df1, df2), r'../data/recover/output/pregnancy_output/_selected_preg_cohort_1-2.pkl')
    #
    # print('Severity cohorts:', args.severity,
    #       'df.shape:', df.shape,
    #       'df_general.shape:', df_general.shape,
    #       'df1.shape:', df1.shape,
    #       'df2.shape:', df2.shape,
    #       )
    # print('Cohort build Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    #
    # zz

    df1, df2 = utils.load(r'../data/recover/output/pregnancy_output/_selected_preg_cohort_1-2.pkl')
    print(r"df1['site'].value_counts()", df1['site'].value_counts())
    print('len(df1)', len(df1), 'len(df2)', len(df2))

    print('Build matched cohort, kmatch:', args.kmatch, 'usedx:', args.usedx, 'useacute:', args.useacute)
    df2_matched = build_matched_control(df1, df2, kmatche=args.kmatch, usedx=args.usedx, useacute=args.useacute)
    utils.dump(df2_matched,
               r'../data/recover/output/pregnancy_output/_selected_preg_cohort2-matched-k{}-useSelectdx{}-useacute{}V2.pkl'.format(
                   args.kmatch, args.usedx, args.useacute))
    print('Cohort build Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
