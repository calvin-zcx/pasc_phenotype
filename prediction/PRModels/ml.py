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
# from iptw.evaluation import cal_deviation, SMD_THRESHOLD, cal_weights
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test, proportional_hazard_test, logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import k_fold_cross_validation


class CoxPrediction:
    def __init__(self, learner='COX', paras_grid=None, random_seed=0):
        self.learner = learner
        assert self.learner in ('COX',)
        self.random_seed = random_seed
        if (paras_grid is None) or (not paras_grid) or (not isinstance(paras_grid, dict)):
            if self.learner == 'COX':
                self.paras_grid = {
                    'l1_ratio': [0],  # 'l2',
                    # 'penalizer': 10 ** np.arange(-3, 3, 0.5),
                    'penalizer': 10 ** np.arange(-3, 0.5, 0.5),

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
            # if self.learner == 'LR':
            #     no_penalty_case = {'penalty': 'none', 'max_iter': 200, 'random_state': random_seed}
            #     if (no_penalty_case not in self.paras_list) and (len(self.paras_list) > 1):
            #         self.paras_list.append(no_penalty_case)
            #         print('Add no penalty case to logistic regression model:', no_penalty_case)

        else:
            self.paras_names = []
            self.paras_list = [{}]

        self.best_hyper_paras = None
        self.best_model = None

        self.best_fit = [float('-inf'), np.nan]  # (mean, std), mean, the larger, the better, AUC
        self.best_fit_k_folds_detail = []  # k AUC

        self.results_colname = ['model-i', 'fold-k', 'paras', 'E[fit]', 'Std[fit]', 'kfold-values']
        self.results = []
        self.risk_results = np.nan

    def cross_validation_fit(self, cov_df_ori, T, E, kfold=5, verbose=1, shuffle=True, scoring_method="concordance_index"):
        start_time = time.time()
        cov_df = cov_df_ori.copy()
        cov_df['T'] = T
        cov_df['E'] = E

        if verbose:
            print('Model {} Searching Space N={} by {}-k-fold cross validation: '.format(
                self.learner, len(self.paras_list), kfold)
            )
        # For each model in model space, do cross-validation training and testing,
        # performance of a model is average (std) over K cross-validated datasets
        # select best model with the best average K-cross-validated performance
        for i, para_d in tqdm(enumerate(self.paras_list, 1), total=len(self.paras_list)):
            print('\nTraining {}th (/{}) model over the {}-fold data, metric:{}'.format(
                i, len(self.paras_list), kfold, scoring_method))

            if self.learner == 'COX':
                model = CoxPHFitter(**para_d)  # .fit(cov_df, 'T', 'E')
            else:
                raise ValueError

            if kfold > 1:
                i_model_fit_over_kfold = k_fold_cross_validation(model,
                                                                 cov_df, 'T', event_col='E',
                                                                 k=kfold, scoring_method=scoring_method)
            else:
                model.fit(cov_df, 'T', 'E')
                i_model_fit_over_kfold = [model.concordance_index_, ]


            i_model_fit = [np.mean(i_model_fit_over_kfold), np.std(i_model_fit_over_kfold)]
            self.results.append([i, kfold, para_d] + i_model_fit + [i_model_fit_over_kfold, ])

            if i_model_fit[0] > self.best_fit[0]:
                # model with current best configuration re-trained on the whole dataset.
                self.best_model = model.fit(cov_df, 'T', 'E')
                self.best_hyper_paras = para_d
                self.best_fit = i_model_fit
                self.best_fit_k_folds_detail = i_model_fit_over_kfold

            if verbose:
                self.report_stats()

        # end of training
        print('Fit Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        self.results = pd.DataFrame(self.results, columns=self.results_colname)
        try:
            if self.learner == 'COX':
                HR = self.best_model.hazard_ratios_
                CI = self.best_model.confidence_intervals_.apply(np.exp)
                PVal = self.best_model.summary.p
            else:
                HR = CI = PVal = np.nan
        except:
            HR = CI = PVal = np.nan

        try:
            self.risk_results = pd.DataFrame({'HR': HR,
                                              'CI-95% lower-bound': CI.iloc[:, 0],
                                              'CI-95% upper-bound': CI.iloc[:, 1],
                                              'p-Value': PVal,
                                              'sum': cov_df_ori.sum().loc[HR.index],
                                              'mean': cov_df_ori.mean().loc[HR.index]})
        except:
            self.risk_results = pd.DataFrame()


        if verbose:
            self.report_stats()
        return self

    def report_stats(self):
        pd.set_option('display.max_columns', None)
        describe = pd.DataFrame(self.results, columns=self.results_colname).describe()
        # describe = self.results.describe()
        # print('All training and evaluation details:\n', describe)
        print('Model {} Searching Space N={}: '.format(self.learner, len(self.paras_list)), self.paras_grid)
        print('Best model: ', self.best_model)
        print('Best configuration: ', self.best_hyper_paras)
        print('Best TEST DATA fit mean(std) ', self.best_fit, 'k-fold details:', self.best_fit_k_folds_detail)
        return describe

    def uni_variate_risk(self, cov_df_ori, T, E, adjusted_col=[], pre=''):
        start_time = time.time()
        if pre == '':
            if len(adjusted_col) == 0:
                pre = 'uni-'
            else:
                pre = 'uni-adjust-'
        add_col = [pre + x for x in ['HR', 'CI-95% lower-bound', 'CI-95% upper-bound', 'p-Value']]

        for col in add_col:
            self.risk_results[col] = 0

        cov_df = cov_df_ori.copy()
        cov_df['T'] = T
        cov_df['E'] = E

        for index, row in self.risk_results.iterrows():

            if len(adjusted_col) == 0:
                cox_data = cov_df[['T', 'E', index]]
                try:
                    model = CoxPHFitter().fit(cox_data, 'T', 'E')
                except:
                    model = CoxPHFitter(**self.best_hyper_paras).fit(cox_data, 'T', 'E')
            else:
                cox_data = cov_df[['T', 'E', index] + [x for x in adjusted_col if ((x != index) and (x in cov_df.columns))]]
                model = CoxPHFitter(**self.best_hyper_paras).fit(cox_data, 'T', 'E')

            HR = model.hazard_ratios_[index]
            CI = np.exp(model.confidence_intervals_.values.reshape(-1))
            PVal = model.summary.p[index]
            self.risk_results.loc[index, add_col] = [HR, CI[0], CI[1], PVal]

        print('Done uni_varate_risk! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # def predict_ps(self, X):
    #     pred_ps = self.best_model.predict_proba(X)[:, 1]
    #     return pred_ps




