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
    parser.add_argument('--dataset', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL'], default='ALL', help='site dataset')
    parser.add_argument("--random_seed", type=int, default=0)
    # parser.add_argument('--run_model', choices=['LSTM', 'LR', 'MLP', 'XGBOOST', 'LIGHTGBM'], default='MLP')
    args = parser.parse_args()

    # More args
    args.data_file = r'../data/V15_COVID19/output/character/pcr_cohorts_covariate_elixh_encoding_{}.csv'.format(args.dataset)

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.save_model_filename)
    return args


def _evaluation_helper(X, T, PS_logits, loss):
    y_pred_prob = logits_to_probability(PS_logits, normalized=False)
    auc = roc_auc_score(T, y_pred_prob)
    max_smd, smd, max_smd_weighted, smd_w = cal_deviation(X, T, PS_logits, normalized=False, verbose=False)
    n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
    n_unbalanced_feature_weighted = len(np.where(smd_w > SMD_THRESHOLD)[0])
    result = (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
    return result


def _loss_helper(v_loss, v_weights):
    return np.dot(v_loss, v_weights) / np.sum(v_weights)


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
    df = pd.read_csv(args.data_file,
                     dtype={'patid': str, 'covid': int})
    # .set_index('patid')
    # because a patid id may occur in multiple sites. patid wers site specific

    df_info = df[['Unnamed: 0', 'patid', 'site']]
    df_label = df['covid']
    df_covs = df.iloc[:, df.columns.get_loc('age20-39'):]
    print('df.shape:', df.shape)
    print('df_covs.shape:', df_covs.shape)
    # df_covs changed to 0 or 1 for the count columns
    # to do
    #
    covs_count = df_covs.sum()
    covs_columns_exclude = covs_count[covs_count < 10].index
    print('len(covs_columns_exclude):', len(covs_columns_exclude))

    df_covs_include = df_covs.drop(covs_columns_exclude, axis=1)
    print('df_covs_include.shape:', df_covs_include.shape)

    # df_covs[['inpatient visits', 'outpatient visits', 'emergency visits', 'other visits']].hist(bins=50)
    # plt.show()

    # utilization, dx, medication were stored as counts.
    # For simplicity here, just binaryize by threshold 0
    # Can cast type to float for numerical computation

    df_covs_array = (df_covs_include > 0).astype('float')

    model = ml.PropensityEstimator(learner='LR').cross_validation_fit(df_covs_array, df_label)
    # , paras_grid = {'penalty': 'l2',
    #                 'C': 0.03162277660168379,
    #                 'max_iter': 200,
    #                 'random_state': 0}
    ps = model.predict_ps(df_covs_array)
    iptw = model.predict_inverse_weight(df_covs_array, df_label, stabilized=True, clip=False)
    smd, smd_weighted = model.predict_smd(df_covs_array, df_label, abs=False, verbose=True)
    plt.plot(smd)
    plt.plot(smd_weighted)
    plt.show()

    model.results.to_csv('../data/V15_COVID19/output/character/evaluation_elixhauser_encoding_model_selection.csv')  # args.save_model_filename +

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
