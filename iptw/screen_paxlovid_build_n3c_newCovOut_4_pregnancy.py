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


def add_col(df):
    df['Type 1 or 2 Diabetes Diagnosis'] = (
            ((df["DX: Diabetes Type 1"] >= 1).astype('int') + (df["DX: Diabetes Type 2"] >= 1).astype(
                'int')) >= 1).astype('int')

    # ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    df['cci_quan:0'] = 0
    df['cci_quan:1-2'] = 0
    df['cci_quan:3-4'] = 0
    df['cci_quan:5-10'] = 0
    df['cci_quan:11+'] = 0

    # ['age18-24', 'age15-34', 'age35-49', 'age50-64', 'age65+']
    df['age18-24'] = 0
    df['age15-34'] = 0
    df['age35-49'] = 0
    df['age50-64'] = 0
    df['age65+'] = 0

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

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 'index date', 'flag_delivery_date', 'flag_pregnancy_start_date', 'flag_pregnancy_end_date'
        index_date = row['index date']
        age = row['age']
        if pd.notna(age):
            if age < 25:
                df.loc[index, 'age18-24'] = 1
            elif age < 35:
                df.loc[index, 'age15-34'] = 1
            elif age < 50:
                df.loc[index, 'age35-49'] = 1
            elif age < 65:
                df.loc[index, 'age50-64'] = 1
            elif age >= 65:
                df.loc[index, 'age65+'] = 1

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
    print('In screen_paxlovid_build_n3c_newCovOuty ...')

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
        # data_file = r'../data/recover/output/{}/matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_{}.csv'.format(
        #     site, site)
        #
        # # Load Covariates Data
        # df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
        #                  parse_dates=['index date', 'dob'])
        # print('df.shape:', df.shape)
        # add new columns
        data_file_add = r'../data/recover/output/{}/matrix_cohorts_covid_posOnly18base-nbaseout-alldays-preg_{}-addCFR.csv'.format(
            site, site)

        # Load Covariates Data
        df_add = pd.read_csv(data_file_add, dtype={'patid': str, 'site': str})
        print('df_add.shape:', df_add.shape)
        # df = pd.merge(df, df_add, how='left', left_on='patid', right_on='patid', suffixes=('', '_y'), )

        # print('After left merge, merged df.shape:', df.shape)
        # df_list.append(df)
        df_list.append(df_add)

    # combine all sites and select subcohorts
    df = pd.concat(df_list, ignore_index=True)
    print(r"df['site'].value_counts(sort=False)", df['site'].value_counts(sort=False))
    print(r"df['site'].value_counts()", df['site'].value_counts())

    print('len(df):', len(df))
    df = df.loc[df['covid'] == 1, :].copy()
    print('after selecting covid pos, len(df):', len(df))

    print('covid+: df.shape:', df.shape)
    out_data_file = 'recover29Nov27_covid_pos_addCFR_only_4_pregnancy.csv'
    # df.to_csv('recoverINSIGHT5Nov27_covid_pos.csv')
    df.to_csv(out_data_file)
    print('dump done!')

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
