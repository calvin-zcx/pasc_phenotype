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
    parser.add_argument('--build_data', action='store_true')

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


def more_ec_for_cohort_selection(df):
    print('in more_ec_for_cohort_selection, df.shape', df.shape)
    print('Applying more specific/flexible eligibility criteria for cohort selection')

    # select index date
    # print('Before selecting index date from 2022-4-1 to 2023-2-28, len(df)', len(df))
    df = df.loc[(df['index date'] <= datetime.datetime(2023, 2, 28, 0, 0)) &
                (df['index date'] >= datetime.datetime(2022, 4, 1, 0, 0)), :]  # .copy()
    print('After selecting index date from 2022-4-1 to 2023-2-28, len(df)', len(df))

    # select age and risk
    # print('Before selecting age >= 50 or at least on risk, len(df)', len(df))
    df = df.loc[(df['age'] >= 50) | (df['pax_risk'] > 0), :]  # .copy()
    print('After selecting age >= 50 or at least on risk, len(df)', len(df))

    # Exclusion, no hospitalized
    # print('Before selecting no hospitalized, len(df)', len(df))
    df = df.loc[(df['inpatienticu'] == 0), :]
    print('After selecting no hospitalized, len(df)', len(df))

    # Exclusion, no contra
    # print('Before selecting pax drug contraindication, len(df)', len(df))
    df = df.loc[(df['pax_contra'] == 0), :]
    print('After selecting pax drug contraindication, len(df)', len(df))

    # drug initiation within 5 days
    df_pos = df.loc[(df['treat-flag@paxlovid'] > 0), :]
    print('After selecting pax prescription, len(df_pos)', len(df_pos))
    df_pos = df_pos.loc[(df_pos['treat-t2e@paxlovid'] <= 5), :]
    print('After selecting pax prescription within 5 days, len(df_pos)', len(df_pos))

    # non initiation group, no paxlovid
    df_control = df.loc[(df['treat-flag@paxlovid'] == 0), :]
    print('After selecting NO pax prescription, len(df_control)', len(df_control))

    return df_pos, df_control


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

    # %% Step 1. Build or Load  Data
    print('In cohorts_characterization_build_data...')
    if args.build_data:
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
            data_file = r'../data/recover/output/{}/matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_{}.csv'.format(
                site, site)

            # Load Covariates Data
            df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
                             parse_dates=['index date', 'dob'])
            print('df.shape:', df.shape)
            df_list.append(df)

        # combine all sites and select subcohorts
        df = pd.concat(df_list, ignore_index=True)
        print(r"df['site'].value_counts(sort=False)", df['site'].value_counts(sort=False))
        print(r"df['site'].value_counts()", df['site'].value_counts())
        df['Paxlovid'] = (df['Paxlovid'] > 0).astype('int')
        print('over all: df.shape:', df.shape)
        print('pax in all:', len(df), df['Paxlovid'].sum(), df['Paxlovid'].mean())
        print('pax in pos:', len(df.loc[df['covid'] == 1, :]), df.loc[df['covid'] == 1, 'Paxlovid'].sum(),
              df.loc[df['covid'] == 1, 'Paxlovid'].mean())
        print('pax in neg:', len(df.loc[df['covid'] == 0, :]), df.loc[df['covid'] == 0, 'Paxlovid'].sum(),
              df.loc[df['covid'] == 0, 'Paxlovid'].mean())

        df['treat-flag-bool@paxlovid'] = (df['treat-flag@paxlovid'] > 0).astype('int')
        print('updated paxlovid codes in all:', len(df), df['treat-flag-bool@paxlovid'].sum(),
              df['treat-flag-bool@paxlovid'].mean())
        print('pax in pos:', len(df.loc[df['covid'] == 1, :]),
              df.loc[df['covid'] == 1, 'treat-flag-bool@paxlovid'].sum(),
              df.loc[df['covid'] == 1, 'treat-flag-bool@paxlovid'].mean())
        print('pax in neg:', len(df.loc[df['covid'] == 0, :]),
              df.loc[df['covid'] == 0, 'treat-flag-bool@paxlovid'].sum(),
              df.loc[df['covid'] == 0, 'treat-flag-bool@paxlovid'].mean())

        df = df.loc[df['covid'] == 1, :].copy()

        print('covid+: df.shape:', df.shape)
        # df.to_csv('recoverINSIGHT5Nov27_covid_pos.csv')
        df.to_csv('recover29Nov27_covid_pos.csv')
        print('dump done!')
    else:
        data_file = 'recover29Nov27_covid_pos.csv'
        data_file = 'recoverINSIGHT5Nov27_covid_pos.csv'
        print('Load data covariates file:', data_file)
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date', 'dob'])
        # pd.DataFrame(df.columns).to_csv('recover_covid_pos-columns-names.csv')
        print('df.shape:', df.shape)

        des = df.describe()
        des.transpose().to_csv(data_file + 'describe.csv')

    # pre-process data a little bit
    print('Considering inpatient/hospitalized cohorts but not ICU')
    df['inpatient'] = ((df['hospitalized'] == 1) & (df['ventilation'] == 0) & (df['criticalcare'] == 0)).astype('int')
    print('Considering ICU (hospitalized ventilation or critical care) cohorts')
    df['icu'] = (((df['hospitalized'] == 1) & (df['ventilation'] == 1)) | (df['criticalcare'] == 1)).astype('int')
    print('Considering inpatient/hospitalized including icu cohorts')
    df['inpatienticu'] = ((df['hospitalized'] == 1) | (df['criticalcare'] == 1)).astype('int')
    print('Considering outpatient cohorts')
    df['outpatient'] = ((df['hospitalized'] == 0) & (df['criticalcare'] == 0)).astype('int')

    df_treat, df_control = more_ec_for_cohort_selection(df)
    df_treat['treated'] = 1
    df_control['treated'] = 0
    print('len(df_treat)', len(df_treat), 'len(df_control)', len(df_control))
    df_treat.to_csv(data_file.replace('.csv', '-ECselectedTreated.csv'))
    df_control.to_csv(data_file.replace('.csv', '-ECselectedControl.csv'))

    # should build two cohorts:
    # 1 trial emulation -- ec
    # 2 RW patients -- matched
    # the following ones help the matched

    """
    selected_cols = [x for x in df.columns if x.startswith('DX:')]
    df['n_baseline_condition'] = df[selected_cols].sum(axis=1)
    df['any_baseline_condition'] = (df['n_baseline_condition'] >= 1).astype('int')

    df_pos = df.loc[df["Paxlovid"] >= 1, :].copy()
    df_neg = df.loc[df["Paxlovid"] == 0, :].copy()

    acute_col = ['outpatient', 'hospitalized', 'icu', ]  # 'criticalcare', 'ventilation']
    age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
    sex_col = ['Female', 'Male']  # , 'Other/Missing'
    # race_col = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    race_col = ['Asian', 'Black or African American', 'White', 'Other']

    eth_col = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    # period_col = [
    #     # "YM: March 2020", "YM: April 2020", "YM: May 2020", "YM: June 2020", "YM: July 2020",
    #     # "YM: August 2020", "YM: September 2020", "YM: October 2020", "YM: November 2020", "YM: December 2020",
    #     # "YM: January 2021", "YM: February 2021", "YM: March 2021", "YM: April 2021", "YM: May 2021",
    #     # "YM: June 2021", "YM: July 2021", "YM: August 2021", "YM: September 2021", "YM: October 2021",
    #     # "YM: November 2021",
    #     "YM: December 2021", "YM: January 2022",
    #     "YM: February 2022", "YM: March 2022", "YM: April 2022", "YM: May 2022",
    #     "YM: June 2022", "YM: July 2022", "YM: August 2022", "YM: September 2022",
    #     "YM: October 2022", "YM: November 2022", "YM: December 2022", "YM: January 2023",
    #     "YM: February 2023",
    # ]
    period_col = ['11/21-02/22', '03/22-06/22', '07/22-10/22', '11/22-02/23']
    adi_col = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
               'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100']
    dx_col = ["DX: Asthma", "DX: Cancer", "DX: Chronic Kidney Disease",
              "DX: Congestive Heart Failure", "DX: End Stage Renal Disease on Dialysis",
              "DX: Hypertension", "DX: Pregnant",
              ]
    # cols_to_match = ['site',] + acute_col + age_col + sex_col + race_col + eth_col + period_col + adi_col + dx_col
    cols_to_match = ['site', ] + acute_col + age_col + sex_col + race_col + eth_col + period_col + [
        'any_baseline_condition', ]
    cols_to_match = ['site', ] + acute_col + age_col + race_col + period_col + ['any_baseline_condition', ]
    cols_to_match = acute_col + age_col + race_col + period_col + ['any_baseline_condition', ]

    ctrl_list = exact_match_on(df_pos, df_neg, 10, cols_to_match, )

    print('len(ctrl_list)', len(ctrl_list))
    neg_selected = pd.Series(False, index=df_neg.index)
    neg_selected[ctrl_list] = True
    df_ctrl = df_neg.loc[neg_selected, :]
    print('len(df_pos):', len(df_pos),
          'len(df_neg):', len(df_neg),
          'len(df_ctrl):', len(df_ctrl), )

    df_pos.to_csv('recover_covid_pos-with-pax-V6.csv')
    df_ctrl.to_csv('recover_covid_pos-without-pax-matched-V6.csv')
    """
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
