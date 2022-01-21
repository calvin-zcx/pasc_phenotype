import sys
# for linux env.
sys.path.insert(0, '..')
import time
from dataset import *
import pickle
import argparse
from evaluation import *
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from PSModels import mlp, lstm, ml
from misc import check_and_mkdir
import itertools
import functools
print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='COL', help='input dataset')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--run_model', choices=['LSTM', 'LR', 'MLP', 'XGBOOST', 'LIGHTGBM'], default='MLP')
    parser.add_argument('--stats', action='store_true')
    # Output
    parser.add_argument('--output_dir', type=str, default='output/')
    args = parser.parse_args()

    # More args
    if args.dataset == 'COL':
        args.data_file = r'../data/V15_COVID19/output/data_cohorts_COL.pkl'
    elif args.dataset == 'WCM':
        args.data_file = r'../data/V15_COVID19/output/data_cohorts_WCM.pkl'

    if args.random_seed >= 0:
        rseed = args.random_seed
    else:
        from datetime import datetime
        rseed = datetime.now()
    args.random_seed = rseed
    args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    check_and_mkdir(args.save_model_filename)
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
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)

    # %% 1. Load Data
    print('Load data file:', args.data_file)
    data_raw = pickle.load(open(args.data_file, 'rb'))
    print('len(data_raw):', len(data_raw))

    # load diagnoses, medication encoding mapping

    with open(_fname, 'rb') as f:
        # change later, move this file to pickles also
        gpiing_names_cnt = pickle.load(f)
        drug_name = {}
        for key, val in gpiing_names_cnt.items():
            drug_name[key] = '/'.join(val[0])
    print('Using GPI vocabulary, len(drug_name) :', len(drug_name))


    # 1-C: build pytorch dataset
    print("Constructed Dataset, choose med_code_topk:", args.med_code_topk)
    my_dataset = Dataset(treated, controlled_sample,
                         med_code_topk=args.med_code_topk,
                         diag_name=dx_name,
                         med_name=drug_name)  # int(len(treated)/5)) #150)

    n_feature = my_dataset.DIM_OF_CONFOUNDERS  # my_dataset.med_vocab_length + my_dataset.diag_vocab_length + 3
    feature_name = my_dataset.FEATURE_NAME
    print('n_feature: ', n_feature, ':')
    # print(feature_name)

    train_ratio = 0.7  # 0.5
    val_ratio = 0.1
    print('train_ratio: ', train_ratio,
          'val_ratio: ', val_ratio,
          'test_ratio: ', 1 - (train_ratio + val_ratio))

    dataset_size = len(my_dataset)
    indices = list(range(dataset_size))
    train_index = int(np.floor(train_ratio * dataset_size))
    val_index = int(np.floor(val_ratio * dataset_size))

    np.random.shuffle(indices)

    train_indices, val_indices, test_indices = indices[:train_index], \
                                               indices[train_index:train_index + val_index], \
                                               indices[train_index + val_index:]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                             sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                              sampler=test_sampler)
    data_loader = torch.utils.data.DataLoader(my_dataset, batch_size=args.batch_size,
                                              sampler=SubsetRandomSampler(indices))

    # %% Logistic regression PS PSModels
    if args.run_model in ['LR', 'XGBOOST', 'LIGHTGBM']:
        print("**************************************************")
        print("**************************************************")
        print(args.run_model, ' PS model learning:')

        print('Train data:')
        train_x, train_t, train_y = flatten_data(my_dataset, train_indices)
        print('Validation data:')
        val_x, val_t, val_y = flatten_data(my_dataset, val_indices)
        print('Test data:')
        test_x, test_t, test_y = flatten_data(my_dataset, test_indices)
        print('All data:')
        x, t, y = flatten_data(my_dataset, indices)  # all the data

        # put fixed parameters also into a list e.g. 'objective' : ['binary',]
        if args.run_model == 'LR':
            paras_grid = {
                'penalty': ['l1', 'l2'],
                'C': 10 ** np.arange(-3, 3, 0.2),  # 'C': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20],
                'max_iter': [200],  # [100, 200, 500],
                'random_state': [args.random_seed],
            }
        elif args.run_model == 'XGBOOST':
            paras_grid = {
                'max_depth': [3, 4],
                'min_child_weight': np.linspace(0, 1, 5),
                'learning_rate': np.arange(0.01, 1, 0.1),
                'colsample_bytree': np.linspace(0.05, 1, 5),
                'random_state': [args.random_seed],
            }
        elif args.run_model == 'LIGHTGBM':
            paras_grid = {
                'max_depth': [3, 4, 5],
                'learning_rate': np.arange(0.01, 1, 0.1),
                'num_leaves': np.arange(5, 50, 10),
                'min_child_samples': [200, 250, 300],
                'random_state': [args.random_seed],
            }
        else:
            paras_grid = {}

        # ----2. Learning IPW using PropensityEstimator
        # model = ml.PropensityEstimator(args.run_model, paras_grid).fit(train_x, train_t, val_x, val_t)
        model = ml.PropensityEstimator(args.run_model, paras_grid).fit_and_test(train_x, train_t, val_x, val_t, test_x,
                                                                                test_t)

        with open(args.save_model_filename, 'wb') as f:
            pickle.dump(model, f)

        model.results.to_csv(args.save_model_filename + '_ALL-model-select.csv')
        # ----3. Evaluation learned PropensityEstimator
        results_all_list, results_all_df = final_eval_ml(model, args, train_x, train_t, train_y, val_x, val_t, val_y,
                                                         test_x, test_t, test_y, x, t, y,
                                                         drug_name, feature_name, n_feature, dump_ori=False)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


