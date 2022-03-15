import sys

# for linux env.
sys.path.insert(0, '..')
sys.path.insert(0, '../..')
import numpy as np
import itertools
import time
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm, tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, log_loss
# import xgboost as xgb
import lightgbm as lgb
from iptw.evaluation import cal_deviation, SMD_THRESHOLD, cal_weights


class PropensityEstimator:
    def __init__(self, learner='LR', paras_grid=None, random_seed=0):
        self.learner = learner
        assert self.learner in ('LR', 'LIGHTGBM')
        self.random_seed = random_seed
        if (paras_grid is None) or (not paras_grid) or (not isinstance(paras_grid, dict)):
            if self.learner == 'LR':
                self.paras_grid = {
                    'penalty': ['l2'],  # 'l1',
                    'C': 10 ** np.arange(-2, 2.5, 0.5),
                    'max_iter': [200],  # [100, 200, 500],
                    'random_state': [random_seed],
                    'n_jobs': [4],
                }
            elif self.learner == 'LIGHTGBM':
                self.paras_grid = {
                    'max_depth': [3, 4, 5],
                    'learning_rate': np.arange(0.01, 1, 0.1),
                    'num_leaves': np.arange(5, 50, 10),
                    'min_child_samples': [200, 250, 300],
                    'random_state': [random_seed],
                }

            else:
                self.paras_grid = {}
        else:
            self.paras_grid = {k: v for k, v in paras_grid.items()}
            for k, v in self.paras_grid.items():
                if isinstance(v, str) or not isinstance(v, (list, set)):
                    print(k, v, 'is a fixed parameter')
                    self.paras_grid[k] = [v, ]

        if self.paras_grid:
            paras_names, paras_v = zip(*self.paras_grid.items())
            paras_list = list(itertools.product(*paras_v))
            self.paras_names = paras_names
            self.paras_list = [{self.paras_names[i]: para[i] for i in range(len(para))} for para in paras_list]
            if self.learner == 'LR':
                no_penalty_case = {'penalty': 'none', 'max_iter': 200, 'random_state': random_seed, 'n_jobs': 4}
                if (no_penalty_case not in self.paras_list) and (len(self.paras_list) > 1):
                    self.paras_list.append(no_penalty_case)
                    print('Add no penalty case to logistic regression model:', no_penalty_case)

        else:
            self.paras_names = []
            self.paras_list = [{}]

        self.best_hyper_paras = None
        self.best_model = None
        self.best_balance = [float('inf'), np.nan]  # (mean, std), mean, the smaller, the better. number of unbalanced features
        self.best_fit = [float('-inf'), np.nan]  # (mean, std), mean, the larger, the better, AUC
        self.best_balance_k_folds_detail = []  # k #(SMD>threshold)
        self.best_fit_k_folds_detail = []  # k AUC

        # self.global_best_val = float('-inf')
        # self.global_best_balance = float('inf')
        name = ['loss', 'auc', 'max_smd', 'n_unbalanced_feat', 'max_smd_iptw', 'n_unbalanced_feat_iptw']
        self.results_col_name = ['model-i', 'fold-k', 'paras'] + [pre + x for pre in ['test_', 'all_'] for x in name]
        self.results = []

    @staticmethod
    def _evaluation_helper(X, T, T_pre):
        loss = log_loss(T, T_pre)
        auc = roc_auc_score(T, T_pre)
        max_smd, smd, max_smd_weighted, smd_w, before, after = cal_deviation(X, T, T_pre, normalized=True, verbose=False, abs=True)
        n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
        n_unbalanced_feature_weighted = len(np.where(smd_w > SMD_THRESHOLD)[0])
        result = (loss, auc, max_smd, n_unbalanced_feature, max_smd_weighted, n_unbalanced_feature_weighted)
        return result

    def _model_estimation(self, para_d, X_train, T_train):
        # model estimation on training data
        if self.learner == 'LR':
            if para_d.get('penalty', '') == 'l1':
                para_d['solver'] = 'liblinear'
            else:
                para_d['solver'] = 'lbfgs'
            model = LogisticRegression(**para_d).fit(X_train, T_train)
        elif self.learner == 'LIGHTGBM':
            model = lgb.LGBMClassifier(**para_d).fit(X_train, T_train)
        # elif learner == 'SVM':
        #   model = svm.SVC().fit(confounder, treatment)
        # elif learner == 'CART':
        #   model = tree.DecisionTreeClassifier(max_depth=6).fit(confounder, treatment)
        else:
            raise ValueError

        return model

    def cross_validation_fit(self, X, T, kfold=10, verbose=1, shuffle=True):
        start_time = time.time()
        kf = KFold(n_splits=kfold, random_state=self.random_seed, shuffle=shuffle)
        if verbose:
            print('Model {} Searching Space N={} by '
                  '{}-k-fold cross validation: '.format(self.learner,
                                                        len(self.paras_list),
                                                        kf.get_n_splits()), self.paras_grid)
        # For each model in model space, do cross-validation training and testing,
        # performance of a model is average (std) over K cross-validated datasets
        # select best model with the best average K-cross-validated performance
        X = np.asarray(X)
        T = np.asarray(T)
        for i, para_d in tqdm(enumerate(self.paras_list, 1), total=len(self.paras_list)):
            i_model_balance_over_kfold = []
            i_model_fit_over_kfold = []
            for k, (train_index, test_index) in enumerate(kf.split(X), 1):
                print('Training {}th (/{}) model over the {}th-fold data'.format(i, len(self.paras_list), k))
                # training and testing datasets:
                X_train = X[train_index, :]
                T_train = T[train_index]
                X_test = X[test_index, :]
                T_test = T[test_index]

                # model estimation on training data
                model = self._model_estimation(para_d, X_train, T_train)

                # propensity scores on training and testing datasets
                T_train_pre = model.predict_proba(X_train)[:, 1]
                T_test_pre = model.predict_proba(X_test)[:, 1]

                # evaluating goodness-of-balance and goodness-of-fit
                # result_train = self._evaluation_helper(X_train, T_train, T_train_pre)
                result_test = self._evaluation_helper(X_test, T_test, T_test_pre)
                result_all = self._evaluation_helper(
                    np.concatenate((X_train, X_test)),
                    np.concatenate((T_train, T_test)),
                    np.concatenate((T_train_pre, T_test_pre))
                )
                i_model_balance_over_kfold.append(result_all[5])
                i_model_fit_over_kfold.append(result_test[1])

                self.results.append((i, k, para_d) + result_test + result_all)
                # end of one fold

            i_model_balance = [np.mean(i_model_balance_over_kfold), np.std(i_model_balance_over_kfold)]
            i_model_fit = [np.mean(i_model_fit_over_kfold), np.std(i_model_fit_over_kfold)]

            if (i_model_balance[0] < self.best_balance[0]) or \
                    ((i_model_balance[0] == self.best_balance[0]) and (i_model_fit[0] > self.best_fit[0])):
                # model with current best configuration re-trained on the whole dataset.
                self.best_model = self._model_estimation(para_d, X, T)
                self.best_hyper_paras = para_d
                self.best_balance = i_model_balance
                self.best_fit = i_model_fit
                self.best_balance_k_folds_detail = i_model_balance_over_kfold
                self.best_fit_k_folds_detail = i_model_fit_over_kfold

            if verbose:
                self.report_stats()
        # end of training
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        self.results = pd.DataFrame(self.results, columns=self.results_col_name)
        if verbose:
            self.report_stats()
        return self

    def report_stats(self):
        pd.set_option('display.max_columns', None)
        describe = pd.DataFrame(self.results, columns=self.results_col_name).describe()
        # describe = self.results.describe()
        print('All training and evaluation details:\n', describe)
        print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        print('Best model: ', self.best_model)
        print('Best configuration: ', self.best_hyper_paras)
        print('Best WHOLE DATA balance mean(std): ', self.best_balance, 'k-fold details:', self.best_balance_k_folds_detail)
        print('Best TEST DATA fit mean(std) ', self.best_fit, 'k-fold details:', self.best_fit_k_folds_detail)
        return describe

    def predict_ps(self, X):
        pred_ps = self.best_model.predict_proba(X)[:, 1]
        return pred_ps

    def predict_inverse_weight(self, X,  T, stabilized=True, clip=False):
        T_pre = self.predict_ps(X)
        treated_w, controlled_w = cal_weights(T, T_pre, normalized=True, stabilized=stabilized, clip=clip)
        weight = np.zeros((len(T)))
        weight[T == 1] = treated_w.squeeze()
        weight[T == 0] = controlled_w.squeeze()
        return weight

    def predict_smd(self, X,  T, abs=False, verbose=False):
        T_pre = self.predict_ps(X)
        max_smd, smd, max_smd_weighted, smd_weighted, before, after = cal_deviation(X, T, T_pre, normalized=True,
                                                                                    verbose=verbose, abs=abs)
        return smd, smd_weighted, before, after

    def predict_loss(self, X, T):
        T_pre = self.predict_ps(X)
        return log_loss(T, T_pre)


