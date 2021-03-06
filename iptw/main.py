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

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL'], default='COL',
                        help='site dataset')
    parser.add_argument("--random_seed", type=int, default=0)
    # parser.add_argument('--run_model', choices=['LSTM', 'LR', 'MLP', 'XGBOOST', 'LIGHTGBM'], default='MLP')
    args = parser.parse_args()

    # More args
    # args.data_file = r'../data/V15_COVID19/output/character/pcr_cohorts_covariate_elixh_encoding_{}.csv'.format(
    #     args.dataset) # matrix_cohorts_covid_4manuscript_bool_COL.csv
    args.data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuscript_bool_{}.csv'.format(
        args.dataset)

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
    df_summary.to_csv('../data/V15_COVID19/output/character/evaluation_covariates_balancing.csv')
    return df_summary


if __name__ == "__main__":
    # python main.py --dataset ALL 2>&1 | tee  log/main_covariates_elixhauser.txt
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)

    # %% 1. Load Data
    print('Load data file:', args.data_file)
    df = pd.read_csv(args.data_file, dtype={'patid': str}, parse_dates=['index date'])
    # .set_index('patid')
    # because a patid id may occur in multiple sites. patid wers site specific

    df_info = df[['Unnamed: 0', 'patid', 'site', 'index date', 'hospitalized',
                  'ventilation', 'criticalcare', 'maxfollowup', 'death', 'death t2e','YM: March 2020',
                  'YM: April 2020', 'YM: May 2020', 'YM: June 2020', 'YM: July 2020', 'YM: August 2020',
                  'YM: September 2020', 'YM: October 2020', 'YM: November 2020', 'YM: December 2020',
                  'YM: January 2021', 'YM: February 2021', 'YM: March 2021', 'YM: April 2021', 'YM: May 2021',
                  'YM: June 2021', 'YM: July 2021', 'YM: August 2021', 'YM: September 2021', 'YM: October 2021',
                  'YM: November 2021', 'YM: December 2021', 'YM: January 2022',]]
    df_label = df['covid']
    covs_columns = [x for x in
                    list(df.columns)[df.columns.get_loc('20-<40 years'):(df.columns.get_loc('MEDICATION: Immunosuppressant drug')+1)]
                    if not x.startswith('YM:')
                    ]
    print('len(covs_columns):', len(covs_columns))
    df_covs = df.loc[:, covs_columns].astype('float')
    print('df.shape:', df.shape)
    print('df_covs.shape:', df_covs.shape)
    # df_covs changed to 0 or 1 for the count columns
    # to do
    #
    # covs_count = df_covs.sum()
    # covs_columns_exclude = covs_count[covs_count < 10].index
    # print('len(covs_columns_exclude):', len(covs_columns_exclude))
    #
    # df_covs_include = df_covs.drop(covs_columns_exclude, axis=1)
    # print('df_covs_include.shape:', df_covs_include.shape)

    # df_covs[['inpatient visits', 'outpatient visits', 'emergency visits', 'other visits']].hist(bins=50)
    # plt.show()

    # utilization, dx, medication were stored as counts.
    # For simplicity here, just binaryize by threshold 0
    # Can cast type to float for numerical computation

    df_covs_array = (df_covs_include > 0).astype('float')

    model = ml.PropensityEstimator(learner='LR', random_seed=args.random_seed, paras_grid={
        'penalty': 'l2',
        'C': 0.03162277660168379,
        'max_iter': 200,
        'random_state': 0}).cross_validation_fit(df_covs_array, df_label)

    ps = model.predict_ps(df_covs_array)
    iptw = model.predict_inverse_weight(df_covs_array, df_label, stabilized=True, clip=False)
    smd, smd_weighted, before, after = model.predict_smd(df_covs_array, df_label, abs=False, verbose=True)
    plt.scatter(range(len(smd)), smd)
    plt.scatter(range(len(smd)), smd_weighted)
    plt.show()

    df_summary = summary_covariate(df_covs_array, df_label, iptw, smd, smd_weighted, before, after)

    model.results.to_csv(
        '../data/V15_COVID19/output/character/evaluation_elixhauser_encoding_model_selection.csv')  # args.save_model_filename +

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
