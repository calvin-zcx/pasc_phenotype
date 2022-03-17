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

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['oneflorida', 'V15_COVID19'], default='V15_COVID19',
                        help='data bases')
    parser.add_argument('--site', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL', 'all'], default='ALL',
                        help='site dataset')
    parser.add_argument('--severity', choices=['all',
                                               'outpatient', 'inpatient', 'icu',
                                               'female', 'male',
                                               'white', 'black',
                                               'less65', '65to75', '75above',
                                               'Anemia', 'Arrythmia', 'CKD', 'CPD-COPD', 'CAD',
                                               'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids'],
                        default='all')

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--negative_ratio', type=int, default=5)
    args = parser.parse_args()

    # More args
    args.data_file = r'../data/{}/output/character/matrix_cohorts_covid_4manuNegNoCovid_bool_{}.csv'.format(
        args.dataset,
        args.site)

    args.output_file = r'../data/{}/output/character/matrix_cohorts_covid_4manuNegNoCovid_bool_{}_withPS.csv'.format(
        args.dataset,
        args.site)

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


if __name__ == "__main__":
    # python data_add_ps.py --dataset V15_COVID19 --site ALL --severity all 2>&1 | tee  log/data_add_ps_insight_ALL_all.txt
    # python data_add_ps.py --dataset oneflorida --site all --severity all 2>&1 | tee  log/data_add_ps_oneflorida_all_all.txt

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)

    # %% 1. Load  Data
    # Load Covariates Data
    print('Load data covariates file:', args.data_file)
    df = pd.read_csv(args.data_file, dtype={'patid': str}, parse_dates=['index date'])
    # because a patid id may occur in multiple sites. patid were site specific
    print('df.shape:', df.shape)
    if args.severity == 'inpatient':
        print('Considering inpatient/hospitalized cohorts but not ICU')
        df = df.loc[(df['hospitalized'] == 1) & (df['ventilation']==0) & (df['criticalcare']==0), :].copy()
    elif args.severity == 'icu':
        print('Considering ICU (hospitalized ventilation or critical care) cohorts')
        df = df.loc[(((df['hospitalized'] == 1) & (df['ventilation']==1)) | (df['criticalcare']==1)), :].copy()
    elif args.severity == 'outpatient':
        print('Considering outpatient cohorts')
        df = df.loc[(df['hospitalized'] == 0) & (df['criticalcare']==0), :].copy()
    elif args.severity == 'female':
        print('Considering female cohorts')
        df = df.loc[(df['Female'] == 1), :].copy()
    elif args.severity == 'male':
        print('Considering male cohorts')
        df = df.loc[(df['Male'] == 1), :].copy()
    elif args.severity == 'white':
        print('Considering white cohorts')
        df = df.loc[(df['White'] == 1), :].copy()
    elif args.severity == 'black':
        print('Considering black cohorts')
        df = df.loc[(df['Black or African American'] == 1), :].copy()
    elif args.severity == 'less65':
        print('Considering less65 cohorts')
        df = df.loc[(df['20-<40 years'] == 1) | (df['40-<55 years'] == 1) | (df['55-<65 years'] == 1), :].copy()
    elif args.severity == '65to75':
        print('Considering 65to75 cohorts')
        df = df.loc[(df['65-<75 years'] == 1), :].copy()
    elif args.severity == '75above':
        print('Considering 75above cohorts')
        df = df.loc[(df['75-<85 years'] == 1) | (df['85+ years'] == 1), :].copy()
    elif args.severity == 'Anemia':
        print('Considering Anemia cohorts')
        df = df.loc[(df["DX: Anemia"] == 1), :].copy()
    elif args.severity == 'Arrythmia':
        print('Considering Arrythmia cohorts')
        df = df.loc[(df["DX: Arrythmia"] == 1), :].copy()
    elif args.severity == 'CKD':
        print('Considering CKD cohorts')
        df = df.loc[(df["DX: Chronic Kidney Disease"] == 1), :].copy()
    elif args.severity == 'CPD-COPD':
        print('Considering CPD-COPD cohorts')
        df = df.loc[(df["DX: Chronic Pulmonary Disorders"] == 1) | (df["DX: COPD"] == 1), :].copy()
    elif args.severity == 'CAD':
        print('Considering CAD cohorts')
        df = df.loc[(df["DX: Coronary Artery Disease"] == 1), :].copy()
    elif args.severity == 'T2D-Obesity':
        print('Considering T2D-Obesity cohorts')
        df = df.loc[(df["DX: Diabetes Type 2"] == 1) | (df["DX: Severe Obesity  (BMI>=40 kg/m2)"] == 1), :].copy()
    elif args.severity == 'Hypertension':
        print('Considering Hypertension cohorts')
        df = df.loc[(df["DX: Hypertension"] == 1), :].copy()
    elif args.severity == 'Mental-substance':
        print('Considering Mental-substance cohorts')
        df = df.loc[(df["DX: Mental Health Disorders"] == 1) | (df['DX: Other Substance Abuse'] == 1), :].copy()
    elif args.severity == 'Corticosteroids':
        print('Considering Corticosteroids cohorts')
        df = df.loc[(df["MEDICATION: Corticosteroids"] == 1), :].copy()
    else:
        print('Considering ALL cohorts')
    # 'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids'
    print('Severity cohorts:', args.severity, 'df.shape:', df.shape)
    df_info = df[['Unnamed: 0', 'patid', 'site', 'index date', 'hospitalized',
                  'ventilation', 'criticalcare', 'maxfollowup', 'death', 'death t2e']]
    df_label = df['covid']
    covs_columns = [x for x in
                    list(df.columns)[
                    df.columns.get_loc('20-<40 years'):(df.columns.get_loc('MEDICATION: Immunosuppressant drug') + 1)]
                    if not x.startswith('YM:')
                    ]
    print('len(covs_columns):', len(covs_columns))
    df_covs = df.loc[:, covs_columns].astype('float')

    print('df.shape:', df.shape)
    print('df_covs.shape:', df_covs.shape)

    model = ml.PropensityEstimator(learner='LR', random_seed=args.random_seed, paras_grid = {
        'penalty': 'l2',
        'C': 0.03162277660168379,
        'max_iter': 200,
        'random_state': 0}).cross_validation_fit(df_covs, df_label, verbose=0)
    #
    # , paras_grid = {
    #     'penalty': 'l2',
    #     'C': 0.03162277660168379,
    #     'max_iter': 200,
    #     'random_state': 0}

    ps = model.predict_ps(df_covs)
    model.report_stats()
    iptw = model.predict_inverse_weight(df_covs, df_label, stabilized=True, clip=False)
    smd, smd_weighted, before, after = model.predict_smd(df_covs, df_label, abs=False, verbose=True)
    plt.scatter(range(len(smd)), smd)
    plt.scatter(range(len(smd)), smd_weighted)
    plt.show()
    print('n unbalanced covariates before:after = {}:{}'.format(
        (smd > SMD_THRESHOLD).sum(),
        (smd_weighted > SMD_THRESHOLD).sum())
    )
    df['ps'] = ps
    df['iptw'] = iptw

    utils.check_and_mkdir(args.output_file)
    df.to_csv(args.output_file, index=False)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
