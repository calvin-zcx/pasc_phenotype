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
                        default='pospreg-posnonpreg')
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


def add_any_pasc(df, exclude_list = []):
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


if __name__ == "__main__":
    # python screen_dx_recover_pregnancy_cohort2.py --site all --severity pospreg-posnonpreg 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_cohort2_all_pospreg-posnonpreg.txt
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)

    # %% 1. Load  Data
    print('In cohorts_characterization_build_data...')
    if args.site == 'all':
        sites = ['mcw', 'nebraska', 'utah', 'utsw',
                 'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
                 'ufh', 'usf', 'miami',  # 'emory', 'nch',
                 'pitt', 'psu', 'temple', 'michigan',
                 'ochsner', 'ucsf',  # 'lsu',
                 'vumc']

        # sites = ['wcm', 'montefiore', 'mshs',]
        # sites = ['wcm', ]
        # sites = ['pitt', ]
        print('len(sites), sites:', len(sites), sites)
    else:
        sites = [args.site, ]

    df_info_list = []
    df_label_list = []
    df_covs_list = []
    df_outcome_list = []

    df_list = []
    for ith, site in tqdm(enumerate(sites)):
        print('Loading: ', ith, site)
        data_file = r'../data/recover/output/results_before_20230903/pregnancy_data/pregnancy_{}.csv'.format(site)
        # Load Covariates Data
        print('Load data covariates file:', data_file)
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
                         parse_dates=['index date', 'flag_delivery_date', 'flag_pregnancy_start_date',
                                      'flag_pregnancy_end_date'])
        # because a patid id may occur in multiple sites. patid were site specific
        print('df.shape:', df.shape)
        df_list.append(df)

    # combine all sites and select subcohorts
    df = pd.concat(df_list, ignore_index=True)
    df = select_subpopulation(df, args.severity)
    # --> 18-50 years old, female, covid+
    print('Branching building two groups here:')
    # group1: pregnant and covid+ in pregnancy
    # pregnant patients only
    print('Before selecting pregnant, df.shape', df.shape)
    df1 = df.loc[df['flag_pregnancy'] == 1, :]
    print('After selecting pregnant, df1.shape', df1.shape)

    # infection during pregnancy period
    print('Before selecting infection in gestational period, df1.shape', df1.shape)
    df1 = df1.loc[(df1['index date'] >= df1['flag_pregnancy_start_date']) & (
            df1['index date'] <= df1['flag_delivery_date'] + datetime.timedelta(days=7)), :].copy()
    print('After selecting infection in gestational period, df1.shape', df1.shape)

    # group2: non-pregnant group
    print('Before selecting non-pregnant, df.shape', df.shape)
    # df2 = df.loc[df['flag_pregnancy'] == 0, :].copy()
    df2 = df.loc[(df['flag_pregnancy'] == 0) & (df['flag_exclusion'] == 0), :].copy()

    print('After selecting non-pregnant, df2.shape', df2.shape)

    # combine df1 and df2 into df
    df = pd.concat([df1, df2], ignore_index=True)
    df['11/22-02/23'] = ((df["YM: November 2022"] + df["YM: December 2022"] +
                          df["YM: January 2023"] + df["YM: February 2023"]) >= 1).astype('int')
    # df = add_any_pasc(df, exclude_list=['Anemia',
    #                                     'Acute phlebitis; thrombophlebitis and thromboembolism',
    #                                     'Acute pulmonary embolism'])

    # reuse pasc select command
    df = add_any_pasc(df, exclude_list=['Anemia',
                                        ])

    # df.to_csv('pos_preg_femalenot_pitt.csv')
    # df.to_csv('pos_preg_femalenot.csv')
    # zz
    # 'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids'
    print('Severity cohorts:', args.severity,
          'df1.shape:', df1.shape,
          'df2.shape:', df2.shape,
          'df.shape:', df.shape,
          )

    col_names = pd.Series(df.columns)
    df_info = df[['patid', 'site', 'index date', 'hospitalized',
                  'ventilation', 'criticalcare', 'maxfollowup', 'death', 'death t2e',
                  'flag_pregnancy', 'flag_delivery_date', 'flag_pregnancy_start_date',
                  'flag_pregnancy_gestational_age', 'flag_pregnancy_end_date', 'flag_maternal_age',
                  '03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21',
                  '07/21-10/21', '11/21-02/22', '03/22-06/22', '07/22-10/22', '11/22-02/23'
                  ]]  # 'Unnamed: 0',
    # df_info_list.append(df_info)
    # df_label = df['covid']
    df_label = df['flag_pregnancy']
    # df_label_list.append(df_label)

    df_outcome_cols = ['death', 'death t2e'] + [x for x in
                                                list(df.columns)
                                                if x.startswith('dx') or x.startswith('smm') or x.startswith('any_pasc')
                                                ]
    df_outcome = df.loc[:, df_outcome_cols]  # .astype('float')
    # df_outcome_list.append(df_outcome)
    # 'hospitalized',
    covs_columns = ['ventilation', 'criticalcare', ] + \
                   [x for x in
                    list(df.columns)[
                    df.columns.get_loc('pregage:18-<25 years'):(df.columns.get_loc('obc:Pulmonary hypertension'))]
                    if (not x.startswith('YM:')) and (
                            x not in ['Female', 'Male', 'hospitalized', 'Other/Missing', 'DX: Pregnant',
                                      'No evidence - Post-index', 'Fully vaccinated - Post-index',
                                      'Partially vaccinated - Post-index',
                                      'outpatient visits 0', 'outpatient visits 1-2',
                                      'outpatient visits 3-4', 'outpatient visits >=5',
                                      'obc:Preterm birth (< 37 weeks)', 'obc:Gestational diabetes mellitus',
                                      'obc:Delivery BMI\xa0>\xa040', 'obc:Previous cesarean birth',
                                      'obc:Preeclampsia without severe features or gestational hypertension',
                                      'obc:Preeclampsia with severe features',
                                      'obc:Placenta accreta spectrum', 'obc:Placental abruption',
                                      'obc:Twin/multiple pregnancy',
                                      'obc:Placenta previa, complete or partial'])
                    ]

    days = (df['index date'] - datetime.datetime(2020, 3, 1, 0, 0)).apply(lambda x: x.days)
    days = np.array(days).reshape((-1, 1))
    # days_norm = (days - days.min())/(days.max() - days.min())
    # spline = SplineTransformer(degree=3, n_knots=7)
    spline = SplineTransformer(degree=3, n_knots=5)

    days_sp = spline.fit_transform(np.array(days))  # identical
    # days_norm_sp = spline.fit_transform(days_norm) # identical

    print('len(covs_columns):', len(covs_columns))

    # delet old date feature and use spline
    covs_columns = [x for x in covs_columns if x not in
                    ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21',
                     '07/21-10/21', '11/21-02/22', '03/22-06/22', '07/22-10/22', '11/22-02/23']]
    print('after delete 8 days len(covs_columns):', len(covs_columns))
    df_covs = df.loc[:, covs_columns].astype('float')

    new_day_cols = ['days_splie_{}'.format(i) for i in range(days_sp.shape[1])]
    covs_columns += new_day_cols
    print('after adding {} days len(covs_columns):'.format(days_sp.shape[1]), len(covs_columns))
    for i in range(days_sp.shape[1]):
        print('add', i, new_day_cols[i])
        df_covs[new_day_cols[i]] = days_sp[:, i]

    # # days between pregnancy and infection
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
    print(ith, 'df.shape:', df.shape, 'df_covs.shape:', df_covs.shape)

    # df = pd.concat(df_outcome_list, ignore_index=True)
    # df_info = pd.concat(df_info_list, ignore_index=True)
    # df_label = pd.concat(df_label_list, ignore_index=True)
    # df_covs = pd.concat(df_covs_list, ignore_index=True)
    df = df_outcome

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

    pasc_encoding = {}
    pasc_encoding['any_pasc'] = [np.nan, np.nan]
    with open(r'../data/mapping/pasc_index_mapping.pkl', 'rb') as f:
        _pasc_encoding = pickle.load(f)
        print('Load PASC to encoding mapping done! len(pasc_encoding):', len(_pasc_encoding))
        record_example = next(iter(_pasc_encoding.items()))
        print('e.g.:', record_example)

    pasc_encoding.update(_pasc_encoding)

    SMMpasc_encoding = utils.load(r'../data/mapping/SMMpasc_index_mapping.pkl')
    pasc_encoding.update(SMMpasc_encoding)

    # %% 2. PASC specific cohorts for causal inference
    # if args.selectpasc:
    #     df_select = pd.read_excel(
    #         r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
    #         sheet_name='diagnosis').set_index('i')
    #     df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= 0.05, :]  #
    #     df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    #     selected_list = df_select.index.tolist()
    #     print('Selected: len(selected_list):', len(selected_list))
    #     print(df_select['PASC Name Simple'])



    causal_results = []
    results_columns_name = []
    for i, pasc in tqdm(enumerate(pasc_encoding.keys(), start=1), total=len(pasc_encoding)):
        # bulid specific cohorts:
        # if args.selectpasc:
        #     if i not in selected_list:
        #         print('Skip:', i, pasc, 'because args.selectpasc, p<=0.05, hr > 1 in Insight')
        #         continue
        print('\n In c:', i, pasc)

        if pasc.startswith('smm'):
            pasc_flag = (df['smm-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['smm-t2e@' + pasc].astype('float')
            pasc_baseline = df['smm-base@' + pasc]
        elif pasc == 'any_pasc':
            pasc_flag = df['any_pasc_flag'].astype('int')
            pasc_t2e = df['any_pasc_t2e'].astype('float')
            pasc_baseline = df['any_pasc_baseline']
        else:
            pasc_flag = (df['dx-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['dx-t2e@' + pasc].astype('float')
            pasc_baseline = df['dx-base@' + pasc]

        # considering competing risks
        death_flag = df['death']
        death_t2e = df['death t2e']
        pasc_flag.loc[(death_t2e == pasc_t2e)] = 2
        print('#death:', (death_t2e == pasc_t2e).sum(), ' #death in covid+:', df_label[(death_t2e == pasc_t2e)].sum(),
              'ratio of death in covid+:', df_label[(death_t2e == pasc_t2e)].mean())

        # Select population free of outcome at baseline
        idx = (pasc_baseline < 1)
        # Select negative: pos : neg = 1:2 for IPTW

        covid_label = df_label[idx]  # actually current is the pregnant label

        n_covid_pos = covid_label.sum()
        n_covid_neg = (covid_label == 0).sum()

        # print('n_covid_pos:', n_covid_pos, 'n_covid_neg:', n_covid_neg, )
        print('n pregnant:', n_covid_pos, 'n not pregnant:', n_covid_neg, )

        print('# stratum match/stratified sampling by age group')
        # stratum match/stratified sampling
        match_cols = ['pregage:18-<25 years',
                      'pregage:25-<30 years',
                      'pregage:30-<35 years',
                      'pregage:35-<40 years',
                      'pregage:40-<45 years',
                      'pregage:45-50 years', ]

        match_cols_prop = np.array(df_covs.loc[(df_label == 1) & idx, match_cols].mean().to_list())
        match_cols_n = match_cols_prop * n_covid_pos * args.negative_ratio
        match_cols_in_0 = np.array(df_covs.loc[(df_label == 0) & idx, match_cols].sum().to_list())

        match_index_list = []
        for mc, mn1, mn0 in zip(match_cols, match_cols_n, match_cols_in_0):
            mn_ = min(mn1, mn0)
            mn_ = int(mn_)
            match_sampled_neg_index = df_label[(df_label == 0) & idx & (df_covs[mc] == 1)].sample(
                n=mn_,
                replace=False,
                random_state=args.random_seed).index
            match_index_list.append(match_sampled_neg_index)

        sampled_neg_index = match_index_list[0]
        for _iii in range(1, len(match_index_list)):
            sampled_neg_index = sampled_neg_index.append(match_index_list[_iii])

        print('len(sampled_neg_index):', len(sampled_neg_index))

        print('# stratum match/stratified sampling by index date')
        match2_cols = ['03/20-06/20', '07/20-10/20',
                       '11/20-02/21', '03/21-06/21',
                       '07/21-10/21', '11/21-02/22', '03/22-06/22', ]

        match2_cols_prop = np.array(df_info.loc[(df_label == 1) & idx, match2_cols].mean().to_list())
        match2_cols_n = match2_cols_prop * n_covid_pos * args.negative_ratio
        match2_cols_in_0 = np.array(df_info.loc[(df_label == 0) & idx, match2_cols].sum().to_list())

        match2_index_list = []
        for mc, mn1, mn0 in zip(match2_cols, match2_cols_n, match2_cols_in_0):
            mn_ = min(mn1, mn0)
            mn_ = int(mn_)
            match2_sampled_neg_index = df_label[(df_label == 0) & idx & (df_info[mc] == 1)].sample(
                n=mn_,
                replace=False,
                random_state=args.random_seed).index
            match2_index_list.append(match2_sampled_neg_index)

        sampled2_neg_index = match2_index_list[0]
        for _iii in range(1, len(match2_index_list)):
            sampled2_neg_index = sampled2_neg_index.append(match2_index_list[_iii])

        print('len(sampled2_neg_index):', len(sampled2_neg_index))

        print('# intersection of stratum match/stratified sampling by age and index date:')
        sampled_neg_index = sampled_neg_index.intersection(sampled2_neg_index)
        print('after intersection, len(sampled_neg_index):', len(sampled_neg_index))

        print('Sampled with stratified, * folds, min.--args.negative_ratio * n_covid_pos:--',
              args.negative_ratio * n_covid_pos, '\n',
              'n_covid_pos:', n_covid_pos,
              'len(sampled_neg_index):', len(sampled_neg_index))

        # if args.negative_ratio * n_covid_pos < n_covid_neg:
        #     print('replace=False, args.negative_ratio * n_covid_pos:', args.negative_ratio * n_covid_pos,
        #           'n_covid_neg:', n_covid_neg)
        #     sampled_neg_index = covid_label[(covid_label == 0)].sample(n=int(args.negative_ratio * n_covid_pos),
        #                                                                replace=False,
        #                                                                random_state=args.random_seed).index
        # else:
        #     print('replace=True')
        #     # print('Use negative patients with replacement, args.negative_ratio * n_covid_pos:',
        #     #       args.negative_ratio * n_covid_pos,
        #     #       'n_covid_neg:', n_covid_neg)
        #     # sampled_neg_index = covid_label[(covid_label == 0)].sample(n=args.negative_ratio * n_covid_pos,
        #     #                                                            replace=True,
        #     #                                                            random_state=args.random_seed).index
        #     print('Not using sample with replacement. Use all negative patients, args.negative_ratio * n_covid_pos:',
        #           args.negative_ratio * n_covid_pos,
        #           'n_covid_pos:', n_covid_pos,
        #           'n_covid_neg:', n_covid_neg)
        #     sampled_neg_index = covid_label[(covid_label == 0)].index

        pos_neg_selected = pd.Series(False, index=pasc_baseline.index)
        pos_neg_selected[sampled_neg_index] = True
        pos_neg_selected[covid_label[covid_label == 1].index] = True
        #
        pat_info = df_info.loc[pos_neg_selected, :]
        covid_label = df_label[pos_neg_selected]
        covs_array = df_covs.loc[pos_neg_selected, :]
        pasc_flag = pasc_flag[pos_neg_selected]
        pasc_t2e = pasc_t2e[pos_neg_selected]
        print('pasc_t2e.describe():', pasc_t2e.describe())
        pasc_t2e[pasc_t2e <= 30] = 30

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
        out_file_balance = r'../data/recover/output/results-20230825/DX-{}{}-Rev2RerunOri/{}-{}-results.csv'.format(
            args.severity,
            '-select' if args.selectpasc else '',
            i,
            pasc.replace(':', '-').replace('/', '-'))
        utils.check_and_mkdir(out_file_balance)
        model.results.to_csv(out_file_balance)  # args.save_model_filename +

        df_summary = summary_covariate(covs_array, covid_label, iptw, smd, smd_weighted, before, after)
        df_summary.to_csv(
            '../data/recover/output/results-20230825/DX-{}{}-Rev2RerunOri/{}-{}-evaluation_balance.csv'.format(
                args.severity,
                '-select' if args.selectpasc else '',
                i, pasc.replace(':', '-').replace('/', '-')))

        dfps = pd.DataFrame({'ps': ps, 'iptw': iptw, 'covid': covid_label})

        dfps.to_csv(
            '../data/recover/output/results-20230825/DX-{}{}-Rev2RerunOri/{}-{}-evaluation_ps-iptw.csv'.format(
                args.severity,
                '-select' if args.selectpasc else '',
                i, pasc.replace(':', '-').replace('/', '-')))
        try:
            figout = r'../data/recover/output/results-20230825/DX-{}{}-Rev2RerunOri/{}-{}-PS.png'.format(
                args.severity,
                '-select' if args.selectpasc else '',
                i, pasc.replace(':', '-').replace('/', '-'))
            print('Dump ', figout)

            ax = plt.subplot(111)
            sns.histplot(
                dfps, x="ps", hue="covid", element="step",
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
            fig_outfile=r'../data/recover/output/results-20230825/DX-{}{}-Rev2RerunOri/{}-{}-km.png'.format(
                args.severity,
                '-select' if args.selectpasc else '',
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

            if i % 2 == 0:
                pd.DataFrame(causal_results, columns=results_columns_name). \
                    to_csv(r'../data/recover/output/results-20230825/DX-{}{}-Rev2RerunOri/causal_effects_specific-snapshot-{}.csv'.format(
                    args.severity, '-select' if args.selectpasc else '', i))
        except:
            print('Error in ', i, pasc)
            df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

            df_causal.to_csv(
                r'../data/recover/output/results-20230825/DX-{}{}-Rev2RerunOri/causal_effects_specific-ERRORSAVE.csv'.format(
                    args.severity,
                    '-select' if args.selectpasc else '', ))

        print('done one pasc, time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

    df_causal.to_csv(
        r'../data/recover/output/results-20230825/DX-{}{}-Rev2RerunOri/causal_effects_specific.csv'.format(
            args.severity,
            '-select' if args.selectpasc else ''))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
