import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test, proportional_hazard_test, logrank_test
import pandas as pd
from misc.utils import check_and_mkdir
import pickle
import os
from lifelines.plotting import add_at_risk_counts

# Define unbalanced threshold, where SMD > SMD_THRESHOLD are defined as unbalanced features
SMD_THRESHOLD = 0.1


def model_eval_common(X, T, Y, PS_logits, loss=None, normalized=False, verbose=1, figsave='', report=5):
    y_pred_prob = logits_to_probability(PS_logits, normalized)
    # 1. IPTW sample weights
    treated_w, controlled_w = cal_weights(T, PS_logits, normalized=normalized, stabilized=True)
    treated_PS, control_PS = y_pred_prob[T == 1], y_pred_prob[T == 0]
    n_treat, n_control = (T == 1).sum(), (T == 0).sum()

    if verbose:
        print('loss: {}'.format(loss))
        print('treated_weights:',
              pd.Series(treated_w.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(treated_w))
        print('controlled_weights:', pd.Series(controlled_w.flatten()).describe().to_string().replace('\n',
                                                                                                      ';'))  # stats.describe(controlled_w))
        print('treated_PS:',
              pd.Series(treated_PS.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(treated_PS))
        print('controlled_PS:',
              pd.Series(control_PS.flatten()).describe().to_string().replace('\n', ';'))  # stats.describe(control_PS))

    IPTW_ALL = (loss, treated_w, controlled_w,
                pd.Series(treated_w.flatten()).describe().to_string().replace('\n', ';'),
                pd.Series(controlled_w.flatten()).describe().to_string().replace('\n', ';'),
                # stats.describe(treated_w), stats.describe(controlled_w),
                pd.Series(treated_PS.flatten()).describe().to_string().replace('\n', ';'),
                pd.Series(control_PS.flatten()).describe().to_string().replace('\n', ';'),
                # stats.describe(treated_PS), stats.describe(control_PS),
                n_treat, n_control)

    # 2. AUC score
    AUC = roc_auc_score(T, y_pred_prob)
    AUC_weighted = roc_auc_IPTW(T, PS_logits, normalized)  # newly added, not sure
    AUC_expected = roc_auc_expected(PS_logits, normalized)  # newly added, not sure
    AUC_diff = np.absolute(AUC_expected - AUC)
    AUC_ALL = (AUC, AUC_weighted, AUC_expected, AUC_diff)
    if verbose:
        print('AUC:{}\tAUC_weighted:{}\tAUC_expected:{}\tAUC_diff:{}'.
              format(AUC, AUC_weighted, AUC_expected, AUC_diff))

    # 3. SMD score
    if report >= 3:
        max_smd, smd, max_smd_weighted, smd_w, _, _ = cal_deviation(X, T, PS_logits, normalized)
        n_unbalanced_feature = len(np.where(smd > SMD_THRESHOLD)[0])
        n_unbalanced_feature_w = len(np.where(smd_w > SMD_THRESHOLD)[0])
        SMD_ALL = (max_smd, smd, n_unbalanced_feature, max_smd_weighted, smd_w, n_unbalanced_feature_w)
        if verbose:
            print('max_smd_original:{}\tmax_smd_IPTW:{}\tn_unbalanced_feature_origin:{}\tn_unbalanced_feature_IPTW:{}'.
                  format(max_smd, max_smd_weighted, n_unbalanced_feature, n_unbalanced_feature_w))
    else:
        SMD_ALL = []

    # 4. ATE score
    if report >= 4:
        ATE, ATE_w = cal_ATE(T, PS_logits, Y, normalized)
        UncorrectedEstimator_EY1, UncorrectedEstimator_EY0, ATE_original = ATE
        IPWEstimator_EY1, IPWEstimator_EY0, ATE_weighted = ATE_w
        ATE_ALL = (UncorrectedEstimator_EY1, UncorrectedEstimator_EY0, ATE_original,
                   IPWEstimator_EY1, IPWEstimator_EY0, ATE_weighted)
        if verbose:
            print('E[Y1]:{}\tE[Y0]:{}\tATE:{}\tIPTW-E[Y1]:{}\tIPTW-E[Y0]:{}\tIPTW-ATE:{}'.
                  format(UncorrectedEstimator_EY1, UncorrectedEstimator_EY0, ATE_original,
                         IPWEstimator_EY1, IPWEstimator_EY0, ATE_weighted))
    else:
        ATE_ALL = []

    # 5. Survival
    if report >= 5:
        survival, survival_w, cox_HR_ori, cox_HR = cal_survival_KM(T, PS_logits, Y, normalized)
        kmf1, kmf0, ate, survival_1, survival_0, results = survival
        kmf1_w, kmf0_w, ate_w, survival_1_w, survival_0_w, results_w = survival_w
        KM_ALL = (survival, survival_w, cox_HR_ori, cox_HR)
        if verbose:
            print('KM_treated at [180, 365, 540, 730]:', survival_1,
                  'KM_control at [180, 365, 540, 730]:', survival_0)
            # results.print_summary()
            print('KM_treated_IPTW at [180, 365, 540, 730]:', survival_1_w,
                  'KM_control_IPTW at [180, 365, 540, 730]:', survival_0_w)
            print('KM_treated - KM_control:', ate)
            print('KM_treated_IPTW - KM_control_IPTW:', ate_w)
            print('Cox Hazard ratio ori {} (CI: {})'.format(cox_HR_ori[0], cox_HR_ori[1]))
            print('Cox Hazard ratio iptw {} (CI: {})'.format(cox_HR[0], cox_HR[1]))
            # results_w.print_summary()
            ax = plt.subplot(111)
            ax.set_title(os.path.basename(figsave))
            kmf1.plot_survival_function(ax=ax)
            kmf0.plot_survival_function(ax=ax)
            kmf1_w.plot_survival_function(ax=ax)
            kmf0_w.plot_survival_function(ax=ax)
            if figsave:
                plt.savefig(figsave + '_km.png')
            plt.clf()
            # plt.show()
    else:
        KM_ALL = []

    return IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL


def final_eval_ml(model, args, train_x, train_t, train_y, val_x, val_t, val_y, test_x, test_t, test_y,
                  x, t, y, drug_name, feature_name, n_feature, dump_ori=True):
    # ----. Model Evaluation & Final ATE results
    # model_eval_common(X, T, Y, PS_logits, loss=None, normalized=False, verbose=1, figsave='', report=5)
    print("*****" * 5, 'Evaluation on Train data:')
    train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL = model_eval_common(
        train_x, train_t, train_y, model.predict_ps(train_x), loss=model.predict_loss(train_x, train_t),
        normalized=True, figsave=args.save_model_filename + '_train')

    print("*****" * 5, 'Evaluation on Validation data:')
    val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL = model_eval_common(
        val_x, val_t, val_y, model.predict_ps(val_x), loss=model.predict_loss(val_x, val_t),
        normalized=True, figsave=args.save_model_filename + '_val')

    print("*****" * 5, 'Evaluation on Test data:')
    test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL = model_eval_common(
        test_x, test_t, test_y, model.predict_ps(test_x), loss=model.predict_loss(test_x, test_t),
        normalized=True, figsave=args.save_model_filename + '_test')

    print("*****" * 5, 'Evaluation on ALL data:')
    IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL = model_eval_common(
        x, t, y, model.predict_ps(x), loss=model.predict_loss(x, t),
        normalized=True, figsave=args.save_model_filename + '_all')

    results_train_val_test_all = []
    results_all_list = [
        (train_IPTW_ALL, train_AUC_ALL, train_SMD_ALL, train_ATE_ALL, train_KM_ALL),
        (val_IPTW_ALL, val_AUC_ALL, val_SMD_ALL, val_ATE_ALL, val_KM_ALL),
        (test_IPTW_ALL, test_AUC_ALL, test_SMD_ALL, test_ATE_ALL, test_KM_ALL),
        (IPTW_ALL, AUC_ALL, SMD_ALL, ATE_ALL, KM_ALL)
    ]
    for tw, tauc, tsmd, tate, tkm in results_all_list:
        results_train_val_test_all.append(
            [args.treated_drug, drug_name[args.treated_drug], tw[7], tw[8],
             tw[0], tauc[0], tauc[1], tauc[2], tauc[3],
             tsmd[0], tsmd[3], tsmd[2], tsmd[5],
             n_feature, (tsmd[1] >= 0).sum(),
             *tate,
             [180, 365, 540, 730],
             tkm[0][3], tkm[0][4], tkm[0][2],
             tkm[1][3], tkm[1][4], tkm[1][2],
             tkm[2][0], tkm[2][1], tkm[3][0], tkm[3][1],
             tw[1].sum(), tw[2].sum(),
             tw[3], tw[4], tw[5], tw[6],
             ';'.join(feature_name[np.where(tsmd[1] > SMD_THRESHOLD)[0]]),
             ';'.join(feature_name[np.where(tsmd[4] > SMD_THRESHOLD)[0]])
             ])
    results_train_val_test_all = pd.DataFrame(results_train_val_test_all,
                                              columns=['drug', 'name', 'n_treat', 'n_ctrl',
                                                       'loss', 'AUC', 'AUC_IPTW', 'AUC_expected', 'AUC_diff',
                                                       'max_unbalanced_original', 'max_unbalanced_weighted',
                                                       'n_unbalanced_feature', 'n_unbalanced_feature_IPTW',
                                                       'n_feature', 'n_feature_nonzero',
                                                       'EY1_original', 'EY0_original', 'ATE_original',
                                                       'EY1_IPTW', 'EY0_IPTW', 'ATE_IPTW',
                                                       'KM_time_points',
                                                       'KM1_original', 'KM0_original', 'KM1-0_original',
                                                       'KM1_IPTW', 'KM0_IPTW', 'KM1-0_IPTW', 'HR_ori', 'HR_ori_CI',
                                                       'HR_IPTW', 'HR_IPTW_CI',
                                                       'n_treat_IPTW', 'n_ctrl_IPTW',
                                                       'treat_IPTW_stats', 'ctrl_IPTW_stats',
                                                       'treat_PS_stats', 'ctrl_PS_stats',
                                                       'unbalance_feature', 'unbalance_feature_IPTW'],
                                              index=['train', 'val', 'test', 'all'])

    # SMD_ALL = (max_smd, smd, n_unbalanced_feature, max_smd_weighted, smd_w, n_unbalanced_feature_w)
    print('Unbalanced features SMD:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          np.where(SMD_ALL[1] > SMD_THRESHOLD)[0])
    print('Unbalanced features SMD value:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          SMD_ALL[1][SMD_ALL[1] > SMD_THRESHOLD])
    print('Unbalanced features SMD names:\n', SMD_ALL[2], '{:2f}%'.format(SMD_ALL[2] * 100. / len(SMD_ALL[1])),
          feature_name[np.where(SMD_ALL[1] > SMD_THRESHOLD)[0]])
    print('Unbalanced features IPTW-SMD:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          np.where(SMD_ALL[4] > SMD_THRESHOLD)[0])
    print('Unbalanced features IPTW-SMD value:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          SMD_ALL[4][SMD_ALL[4] > SMD_THRESHOLD])
    print('Unbalanced features IPTW-SMD names:\n', SMD_ALL[5], '{:2f}%'.format(SMD_ALL[5] * 100. / len(SMD_ALL[4])),
          feature_name[np.where(SMD_ALL[4] > SMD_THRESHOLD)[0]])

    results_train_val_test_all.to_csv(args.save_model_filename + '_results.csv')
    print('Dump to ', args.save_model_filename + '_results.csv')
    if dump_ori:
        with open(args.save_model_filename + '_results.pt', 'wb') as f:
            pickle.dump(results_all_list + [feature_name, ], f)  #
            print('Dump to ', args.save_model_filename + '_results.pt')

    return results_all_list, results_train_val_test_all


# %%  Aux-functions
def logits_to_probability(logits, normalized):
    if normalized:
        if len(logits.shape) == 1:
            return logits
        elif len(logits.shape) == 2:
            return logits[:, 1]
        else:
            raise ValueError
    else:
        if len(logits.shape) == 1:
            return 1 / (1 + np.exp(-logits))
        elif len(logits.shape) == 2:
            prop = softmax(logits, axis=1)
            return prop[:, 1]
        else:
            raise ValueError


def roc_auc_IPTW(y_true, logits_treatment, normalized):
    treated_w, controlled_w = cal_weights(y_true, logits_treatment, normalized=normalized)
    y_pred_prob = logits_to_probability(logits_treatment, normalized)
    weight = np.zeros((len(logits_treatment), 1))
    weight[y_true == 1] = treated_w
    weight[y_true == 0] = controlled_w
    AUC = roc_auc_score(y_true, y_pred_prob, sample_weight=weight)
    return AUC


def roc_auc_expected(logits_treatment, normalized):
    y_pred_prob = logits_to_probability(logits_treatment, normalized)
    weights = np.concatenate([y_pred_prob, 1 - y_pred_prob])
    t = np.concatenate([np.ones_like(y_pred_prob), np.zeros_like(y_pred_prob)])
    p_hat = np.concatenate([y_pred_prob, y_pred_prob])
    AUC = roc_auc_score(t, p_hat, sample_weight=weights)
    return AUC


def cal_weights(golds_treatment, logits_treatment, normalized, stabilized=True, clip=False):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    logits_treatment = logits_to_probability(logits_treatment, normalized)
    p_T = len(ones_idx[0]) / (len(ones_idx[0]) + len(zeros_idx[0]))
    # comment out p_T scaled IPTW
    if stabilized:
        # stabilized weights:   treated_w.sum() + controlled_w.sum() ~ N
        treated_w, controlled_w = p_T / logits_treatment[ones_idx], (1 - p_T) / (
                1. - logits_treatment[zeros_idx])  # why *p_T here?
    else:
        # standard IPTW:  treated_w.sum() + controlled_w.sum() > N
        treated_w, controlled_w = 1. / logits_treatment[ones_idx], 1. / (
                1. - logits_treatment[zeros_idx])  # why *p_T here? my added test

    if clip:
        treated_w = np.clip(treated_w, a_min=1e-06, a_max=100)
        controlled_w = np.clip(controlled_w, a_min=1e-06, a_max=100)
        # pred_clip_propensity = np.clip(pred_propensity, a_min=np.quantile(pred_propensity, 0.1), a_max=np.quantile(pred_propensity, 0.9))

    treated_w, controlled_w = np.reshape(treated_w, (len(treated_w), 1)), np.reshape(controlled_w,
                                                                                     (len(controlled_w), 1))
    return treated_w, controlled_w


def weighted_mean(x, w):
    # input: x: n * d, w: n * 1
    # output: d
    x_w = np.multiply(x, w)
    n_w = w.sum()
    m_w = np.sum(x_w, axis=0) / n_w
    return m_w


def weighted_var(x, w):
    # x: n * d, w: n * 1
    m_w = weighted_mean(x, w)  # d
    nw, nsw = w.sum(), (w ** 2).sum()
    var = np.multiply((x - m_w) ** 2, w)  # n*d
    var = np.sum(var, axis=0) * (nw / (nw ** 2 - nsw))
    return var


def cal_deviation(covariates, golds_treatment, logits_treatment, normalized, verbose=1, abs=True):
    # covariates, and IPTW
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized=normalized)
    if verbose:
        print('In cal_deviation: n_treated:{}, n_treated_w:{} |'
              'n_controlled:{}, n_controlled_w:{} |'
              'n:{}, n_w:{}'.format(len(treated_w), treated_w.sum(), len(controlled_w), controlled_w.sum(),
                                    len(golds_treatment), treated_w.sum() + controlled_w.sum()))
    covariates = np.asarray(covariates)  # original covariates, to be weighted
    covariates_treated, covariates_controlled = covariates[ones_idx], covariates[zeros_idx]

    # Original SMD
    covariates_treated_mu, covariates_treated_var = np.mean(covariates_treated, axis=0), np.var(covariates_treated,
                                                                                                axis=0, ddof=1)
    covariates_controlled_mu, covariates_controlled_var = np.mean(covariates_controlled, axis=0), np.var(
        covariates_controlled, axis=0,
        ddof=1)
    VAR = np.sqrt((covariates_treated_var + covariates_controlled_var) / 2)
    # covariates_deviation = np.abs(covariates_treated_mu - covariates_controlled_mu) / VAR
    # covariates_deviation[np.isnan(covariates_deviation)] = 0  # -1  # 0  # float('-inf') represent VAR is 0
    covariates_deviation = np.divide(
        covariates_treated_mu - covariates_controlled_mu,
        VAR, out=np.zeros_like(covariates_treated_mu), where=VAR != 0)
    if abs:
        covariates_deviation = np.abs(covariates_deviation)

    max_unbalanced_original = np.max(np.abs(covariates_deviation))

    # Weighted SMD
    covariates_treated_w_mu, covariates_treated_w_var = weighted_mean(covariates_treated, treated_w), weighted_var(
        covariates_treated,
        treated_w)
    covariates_controlled_w_mu, covariates_controlled_w_var = weighted_mean(covariates_controlled,
                                                                            controlled_w), weighted_var(
        covariates_controlled, controlled_w)
    VAR_w = np.sqrt((covariates_treated_w_var + covariates_controlled_w_var) / 2)
    # covariates_deviation_w = np.abs(covariates_treated_w_mu - covariates_controlled_w_mu) / VAR_w
    # covariates_deviation_w[np.isnan(covariates_deviation_w)] = 0  # -1  # 0
    covariates_deviation_w = np.divide(
        covariates_treated_w_mu - covariates_controlled_w_mu,
        VAR_w, out=np.zeros_like(covariates_treated_w_mu), where=VAR_w != 0)
    if abs:
        covariates_deviation_w = np.abs(covariates_deviation_w)

    max_unbalanced_weighted = np.max(np.abs(covariates_deviation_w))

    return max_unbalanced_original, covariates_deviation, max_unbalanced_weighted, covariates_deviation_w, \
           (covariates_treated_mu, covariates_treated_var, covariates_controlled_mu, covariates_controlled_var), \
           (covariates_treated_w_mu, covariates_treated_w_var, covariates_controlled_w_mu, covariates_controlled_w_var)


def cal_ATE(golds_treatment, logits_treatment, golds_outcome, normalized):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized)
    if len(golds_outcome.shape) == 1:
        treated_outcome, controlled_outcome = golds_outcome[ones_idx], golds_outcome[zeros_idx]
    elif len(golds_outcome.shape) == 2:
        treated_outcome, controlled_outcome = golds_outcome[ones_idx, 0], golds_outcome[zeros_idx, 0]
        treated_outcome[treated_outcome == -1] = 0  # censor as 0
        controlled_outcome[controlled_outcome == -1] = 0
    else:
        raise ValueError

    treated_outcome_w = np.multiply(treated_outcome, treated_w.squeeze())
    controlled_outcome_w = np.multiply(controlled_outcome, controlled_w.squeeze())

    # ATE original
    UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val = np.mean(treated_outcome), np.mean(controlled_outcome)
    ATE = UncorrectedEstimator_EY1_val - UncorrectedEstimator_EY0_val

    # ATE weighted
    IPWEstimator_EY1_val, IPWEstimator_EY0_val = treated_outcome_w.sum() / treated_w.sum(), controlled_outcome_w.sum() / controlled_w.sum()
    ATE_w = IPWEstimator_EY1_val - IPWEstimator_EY0_val

    # NMI code for bias reference
    IPWEstimator_EY1_val_old, IPWEstimator_EY0_val_old = np.mean(treated_outcome_w), np.mean(controlled_outcome_w)
    ATE_w_old = IPWEstimator_EY1_val_old - IPWEstimator_EY0_val_old

    return (UncorrectedEstimator_EY1_val, UncorrectedEstimator_EY0_val, ATE), (
        IPWEstimator_EY1_val, IPWEstimator_EY0_val, ATE_w)


def cal_survival_KM(golds_treatment, logits_treatment, golds_outcome, normalized):
    ones_idx, zeros_idx = np.where(golds_treatment == 1), np.where(golds_treatment == 0)
    treated_w, controlled_w = cal_weights(golds_treatment, logits_treatment, normalized)
    if len(golds_outcome.shape) == 2:
        treated_outcome, controlled_outcome = golds_outcome[ones_idx, 0], golds_outcome[zeros_idx, 0]
        treated_outcome[treated_outcome == -1] = 0
        controlled_outcome[controlled_outcome == -1] = 0
    else:
        raise ValueError

    # kmf = KaplanMeierFitter()
    T = golds_outcome[:, 1]
    kmf1 = KaplanMeierFitter(label='Treated').fit(T[ones_idx], event_observed=treated_outcome, label="Treated")
    kmf0 = KaplanMeierFitter(label='Control').fit(T[zeros_idx], event_observed=controlled_outcome, label="Control")

    point_in_time = [180, 365, 540, 730]
    results = survival_difference_at_fixed_point_in_time_test(point_in_time, kmf1, kmf0)
    # results.print_summary()
    survival_1 = kmf1.predict(point_in_time).to_numpy()
    survival_0 = kmf0.predict(point_in_time).to_numpy()
    ate = survival_1 - survival_0

    kmf1_w = KaplanMeierFitter(label='Treated_IPTW').fit(T[ones_idx], event_observed=treated_outcome,
                                                         label="Treated_IPTW", weights=treated_w)
    kmf0_w = KaplanMeierFitter(label='Control_IPTW').fit(T[zeros_idx], event_observed=controlled_outcome,
                                                         label="Control_IPTW", weights=controlled_w)
    results_w = survival_difference_at_fixed_point_in_time_test(point_in_time, kmf1_w, kmf0_w)
    # results_w.print_summary()
    survival_1_w = kmf1_w.predict(point_in_time).to_numpy()
    survival_0_w = kmf0_w.predict(point_in_time).to_numpy()
    ate_w = survival_1_w - survival_0_w

    # ax = plt.subplot(111)
    # kmf1.plot_survival_function(ax=ax)
    # kmf0.plot_survival_function(ax=ax)
    # kmf1_w.plot_survival_function(ax=ax)
    # kmf0_w.plot_survival_function(ax=ax)
    # plt.show()

    # cox for hazard ratio
    cph = CoxPHFitter()
    event = golds_outcome[:, 0]
    event[event == -1] = 0
    weight = np.zeros(len(golds_treatment))
    weight[ones_idx] = treated_w.squeeze()
    weight[zeros_idx] = controlled_w.squeeze()
    cox_data = pd.DataFrame({'T': T, 'event': event, 'treatment': golds_treatment, 'weights': weight})
    try:
        cph.fit(cox_data, 'T', 'event', weights_col='weights', robust=True)
        HR = cph.hazard_ratios_['treatment']
        CI = np.exp(cph.confidence_intervals_.values.reshape(-1))

        cph_ori = CoxPHFitter()
        cox_data_ori = pd.DataFrame({'T': T, 'event': event, 'treatment': golds_treatment})
        cph_ori.fit(cox_data_ori, 'T', 'event')
        HR_ori = cph_ori.hazard_ratios_['treatment']
        CI_ori = np.exp(cph_ori.confidence_intervals_.values.reshape(-1))
    except:
        cph = HR = CI = None
        cph_ori = HR_ori = CI_ori = None

    return (kmf1, kmf0, ate, survival_1, survival_0, results), \
           (kmf1_w, kmf0_w, ate_w, survival_1_w, survival_0_w, results_w), \
           (HR_ori, CI_ori, cph_ori), \
           (HR, CI, cph)


def flag_2binary(label):
    if isinstance(label, pd.core.series.Series):
        return label.where(label == 1, 0)
    elif isinstance(label, (list, np.ndarray)):
        return np.where(label == 1, 1, 0)
    else:
        raise ValueError


def weighted_KM_HR(golds_treatment, weights, events_flag, events_t2e, fig_outfile='', title=''):
    # considering competing risk in this version, 2022-03-20
    #
    ones_idx, zeros_idx = golds_treatment == 1, golds_treatment == 0
    treated_w, controlled_w = weights[ones_idx], weights[zeros_idx]
    treated_flag, controlled_flag = events_flag[ones_idx], events_flag[zeros_idx]
    treated_t2e, controlled_t2e = events_t2e[ones_idx], events_t2e[zeros_idx]

    # Part-1. https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html
    kmf1 = KaplanMeierFitter(label='COVID+').fit(treated_t2e,
                                                 event_observed=flag_2binary(treated_flag), label="COVID+")
    kmf0 = KaplanMeierFitter(label='Control').fit(controlled_t2e,
                                                  event_observed=flag_2binary(controlled_flag), label="Control")

    point_in_time = [60, 90, 120, 150, 180]
    results = survival_difference_at_fixed_point_in_time_test(point_in_time, kmf1, kmf0)
    # results.print_summary()
    survival_1 = kmf1.predict(point_in_time).to_numpy()
    survival_0 = kmf0.predict(point_in_time).to_numpy()
    ate = survival_1 - survival_0

    kmf1_w = KaplanMeierFitter(label='COVID+ Adjusted').fit(treated_t2e, event_observed=flag_2binary(treated_flag),
                                                            label="COVID+ Adjusted", weights=treated_w)
    kmf0_w = KaplanMeierFitter(label='Control Adjusted').fit(controlled_t2e, event_observed=flag_2binary(controlled_flag),
                                                             label="Control Adjusted", weights=controlled_w)
    results_w = survival_difference_at_fixed_point_in_time_test(point_in_time, kmf1_w, kmf0_w)
    # results_w.print_summary()
    survival_1_w = kmf1_w.predict(point_in_time).to_numpy()
    survival_0_w = kmf0_w.predict(point_in_time).to_numpy()
    ate_w = survival_1_w - survival_0_w

    if fig_outfile:
        ax = plt.subplot(111)
        kmf1.plot_survival_function(ax=ax)
        kmf1_w.plot_survival_function(ax=ax)
        kmf0.plot_survival_function(ax=ax)
        kmf0_w.plot_survival_function(ax=ax)

        plt.title(title, fontsize=12)
        plt.savefig(fig_outfile)
        plt.close()

    # Part-2, cumulative incidence for competing risks
    # 0: censoring, 1: event of interest, 2: competing risk, e.g. death
    # https://lifelines.readthedocs.io/en/latest/fitters/univariate/AalenJohansenFitter.html
    ajf1 = AalenJohansenFitter(calculate_variance=True).fit(treated_t2e, treated_flag,
                                                            event_of_interest=1,
                                                            label="COVID+")
    ajf0 = AalenJohansenFitter(calculate_variance=True).fit(controlled_t2e, controlled_flag,
                                                            event_of_interest=1,
                                                            label="Control")
    cif_1 = ajf1.predict(point_in_time).to_numpy()
    cif_0 = ajf0.predict(point_in_time).to_numpy()
    cifdiff = cif_1 - cif_0
    ajf1_CI = ajf1.confidence_interval_cumulative_density_
    ajf0_CI = ajf0.confidence_interval_cumulative_density_
    cif_1_CILower = ajf1_CI[ajf1_CI.columns[0]].asof(point_in_time).to_numpy()
    cif_1_CIUpper = ajf1_CI[ajf1_CI.columns[1]].asof(point_in_time).to_numpy()
    cif_0_CILower = ajf0_CI[ajf0_CI.columns[0]].asof(point_in_time).to_numpy()
    cif_0_CIUpper = ajf0_CI[ajf0_CI.columns[1]].asof(point_in_time).to_numpy()

    ajf1w = AalenJohansenFitter(calculate_variance=True).fit(treated_t2e, treated_flag,
                                                             event_of_interest=1,
                                                             label="Covid-19 Positive", weights=treated_w)
    ajf0w = AalenJohansenFitter(calculate_variance=True).fit(controlled_t2e, controlled_flag,
                                                             event_of_interest=1,
                                                             label="Controls", weights=controlled_w)
    cif_1_w = ajf1w.predict(point_in_time).to_numpy()
    cif_0_w = ajf0w.predict(point_in_time).to_numpy()
    cifdiff_w = cif_1_w - cif_0_w

    ajf1w_CI = ajf1w.confidence_interval_cumulative_density_
    ajf0w_CI = ajf0w.confidence_interval_cumulative_density_
    cif_1_w_CILower = ajf1w_CI[ajf1w_CI.columns[0]].asof(point_in_time).to_numpy()
    cif_1_w_CIUpper = ajf1w_CI[ajf1w_CI.columns[1]].asof(point_in_time).to_numpy()
    cif_0_w_CILower = ajf0w_CI[ajf0w_CI.columns[0]].asof(point_in_time).to_numpy()
    cif_0_w_CIUpper = ajf0w_CI[ajf0w_CI.columns[1]].asof(point_in_time).to_numpy()
    # cif_1_w_CILower = ajf1w_CI.loc[point_in_time, ajf1w_CI.columns[0]].to_numpy()
    # cif_1_w_CIUpper = ajf1w_CI.loc[point_in_time, ajf1w_CI.columns[1]].to_numpy()
    # cif_0_w_CILower = ajf0w_CI.loc[point_in_time, ajf0w_CI.columns[0]].to_numpy()
    # cif_0_w_CIUpper = ajf0w_CI.loc[point_in_time, ajf0w_CI.columns[1]].to_numpy()

    if fig_outfile:
        ax = plt.subplot(111)
        # ajf1.plot(ax=ax)
        ajf1w.plot(ax=ax, loc=slice(0., controlled_t2e.max()))  # 0, 180
        # ajf0.plot(ax=ax)
        ajf0w.plot(ax=ax, loc=slice(0., controlled_t2e.max()))
        add_at_risk_counts(ajf1w, ajf0w, ax=ax)
        plt.xlim([0, 180])
        plt.tight_layout()

        # plt.ylim([0, ajf0w.cumulative_density_.loc[180][0] * 3])

        plt.title(title, fontsize=12)
        plt.savefig(fig_outfile.replace('-km.png', '-cumIncidence.png'))
        plt.close()

    # Part-3: Cox for hazard ratio
    # https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html
    # Competing risk sceneriao: 0 for censoring, 1 for target event, 2 for competing risk death
    # --> competing risk 2, death, as censoring in cox model. Only caring event 1
    # https://github.com/CamDavidsonPilon/lifelines/issues/619
    cph = CoxPHFitter()
    cox_data = pd.DataFrame(
        {'T': events_t2e, 'event': flag_2binary(events_flag), 'treatment': golds_treatment, 'weights': weights})
    try:
        cph.fit(cox_data, 'T', 'event', weights_col='weights', robust=True)
        HR = cph.hazard_ratios_['treatment']
        CI = np.exp(cph.confidence_intervals_.values.reshape(-1))
        test_results = logrank_test(treated_t2e, controlled_t2e, event_observed_A=flag_2binary(treated_flag),
                                    event_observed_B=flag_2binary(controlled_flag), weights_A=treated_w, weights_B=controlled_w, )
        test_p = test_results.p_value
        hr_different_time = cph.compute_followup_hazard_ratios(cox_data, point_in_time)
        hr_different_time = hr_different_time['treatment'].to_numpy()

        cph_ori = CoxPHFitter()
        cox_data_ori = pd.DataFrame({'T': events_t2e, 'event': flag_2binary(events_flag), 'treatment': golds_treatment})
        cph_ori.fit(cox_data_ori, 'T', 'event')
        HR_ori = cph_ori.hazard_ratios_['treatment']
        CI_ori = np.exp(cph_ori.confidence_intervals_.values.reshape(-1))
        test_results_ori = logrank_test(treated_t2e, controlled_t2e, event_observed_A=flag_2binary(treated_flag),
                                        event_observed_B=flag_2binary(controlled_flag))
        test_p_ori = test_results_ori.p_value
        hr_different_time_ori = cph_ori.compute_followup_hazard_ratios(cox_data, point_in_time)
        hr_different_time_ori = hr_different_time_ori['treatment'].to_numpy()

    except:
        cph = HR = CI = test_p_ori = None
        cph_ori = HR_ori = CI_ori = test_p = None
        hr_different_time = hr_different_time_ori = [np.nan]*len(point_in_time)

    return (kmf1, kmf0, ate, point_in_time, survival_1, survival_0, results), \
           (kmf1_w, kmf0_w, ate_w, point_in_time, survival_1_w, survival_0_w, results_w), \
           (HR_ori, CI_ori, test_p_ori, cph_ori, hr_different_time_ori), \
           (HR, CI, test_p, cph, hr_different_time), \
           (ajf1, ajf0, cifdiff, point_in_time, cif_1, cif_0, cif_1_CILower, cif_1_CIUpper, cif_0_CILower, cif_0_CIUpper), \
           (ajf1w, ajf0w, cifdiff_w, point_in_time, cif_1_w, cif_0_w, cif_1_w_CILower, cif_1_w_CIUpper, cif_0_w_CILower, cif_0_w_CIUpper)


def weighted_KM_HR_pooled(golds_treatment, weights, events_flag, events_t2e, database_flag, fig_outfile='', title=''):
    # considering competing risk in this version, 2022-03-20
    #
    ones_idx, zeros_idx = golds_treatment == 1, golds_treatment == 0
    treated_w, controlled_w = weights[ones_idx], weights[zeros_idx]
    treated_flag, controlled_flag = events_flag[ones_idx], events_flag[zeros_idx]
    treated_t2e, controlled_t2e = events_t2e[ones_idx], events_t2e[zeros_idx]

    treatd_database_flag, controlled_database_flag = database_flag[ones_idx], events_flag[zeros_idx]

    # Part-1. https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html
    kmf1 = KaplanMeierFitter(label='COVID+').fit(treated_t2e,
                                                 event_observed=flag_2binary(treated_flag), label="COVID+")
    kmf0 = KaplanMeierFitter(label='Control').fit(controlled_t2e,
                                                  event_observed=flag_2binary(controlled_flag), label="Control")

    point_in_time = [60, 90, 120, 150, 180]
    results = survival_difference_at_fixed_point_in_time_test(point_in_time, kmf1, kmf0)
    # results.print_summary()
    survival_1 = kmf1.predict(point_in_time).to_numpy()
    survival_0 = kmf0.predict(point_in_time).to_numpy()
    ate = survival_1 - survival_0

    kmf1_w = KaplanMeierFitter(label='COVID+ Adjusted').fit(treated_t2e, event_observed=flag_2binary(treated_flag),
                                                            label="COVID+ Adjusted", weights=treated_w)
    kmf0_w = KaplanMeierFitter(label='Control Adjusted').fit(controlled_t2e, event_observed=flag_2binary(controlled_flag),
                                                             label="Control Adjusted", weights=controlled_w)
    results_w = survival_difference_at_fixed_point_in_time_test(point_in_time, kmf1_w, kmf0_w)
    # results_w.print_summary()
    survival_1_w = kmf1_w.predict(point_in_time).to_numpy()
    survival_0_w = kmf0_w.predict(point_in_time).to_numpy()
    ate_w = survival_1_w - survival_0_w

    if fig_outfile:
        ax = plt.subplot(111)
        kmf1.plot_survival_function(ax=ax)
        kmf1_w.plot_survival_function(ax=ax)
        kmf0.plot_survival_function(ax=ax)
        kmf0_w.plot_survival_function(ax=ax)

        plt.title(title, fontsize=12)
        plt.savefig(fig_outfile)
        plt.close()

    # Part-2, cumulative incidence for competing risks
    # 0: censoring, 1: event of interest, 2: competing risk, e.g. death
    # https://lifelines.readthedocs.io/en/latest/fitters/univariate/AalenJohansenFitter.html
    ajf1 = AalenJohansenFitter(calculate_variance=True).fit(treated_t2e, treated_flag,
                                                            event_of_interest=1,
                                                            label="COVID+")
    ajf0 = AalenJohansenFitter(calculate_variance=True).fit(controlled_t2e, controlled_flag,
                                                            event_of_interest=1,
                                                            label="Control")
    cif_1 = ajf1.predict(point_in_time).to_numpy()
    cif_0 = ajf0.predict(point_in_time).to_numpy()
    cifdiff = cif_1 - cif_0
    ajf1_CI = ajf1.confidence_interval_cumulative_density_
    ajf0_CI = ajf0.confidence_interval_cumulative_density_
    cif_1_CILower = ajf1_CI[ajf1_CI.columns[0]].asof(point_in_time).to_numpy()
    cif_1_CIUpper = ajf1_CI[ajf1_CI.columns[1]].asof(point_in_time).to_numpy()
    cif_0_CILower = ajf0_CI[ajf0_CI.columns[0]].asof(point_in_time).to_numpy()
    cif_0_CIUpper = ajf0_CI[ajf0_CI.columns[1]].asof(point_in_time).to_numpy()

    ajf1w = AalenJohansenFitter(calculate_variance=True).fit(treated_t2e, treated_flag,
                                                             event_of_interest=1,
                                                             label="COVID+ Adjusted", weights=treated_w)
    ajf0w = AalenJohansenFitter(calculate_variance=True).fit(controlled_t2e, controlled_flag,
                                                             event_of_interest=1,
                                                             label="Control Adjusted", weights=controlled_w)
    cif_1_w = ajf1w.predict(point_in_time).to_numpy()
    cif_0_w = ajf0w.predict(point_in_time).to_numpy()
    cifdiff_w = cif_1_w - cif_0_w

    ajf1w_CI = ajf1w.confidence_interval_cumulative_density_
    ajf0w_CI = ajf0w.confidence_interval_cumulative_density_
    cif_1_w_CILower = ajf1w_CI[ajf1w_CI.columns[0]].asof(point_in_time).to_numpy()
    cif_1_w_CIUpper = ajf1w_CI[ajf1w_CI.columns[1]].asof(point_in_time).to_numpy()
    cif_0_w_CILower = ajf0w_CI[ajf0w_CI.columns[0]].asof(point_in_time).to_numpy()
    cif_0_w_CIUpper = ajf0w_CI[ajf0w_CI.columns[1]].asof(point_in_time).to_numpy()
    # cif_1_w_CILower = ajf1w_CI.loc[point_in_time, ajf1w_CI.columns[0]].to_numpy()
    # cif_1_w_CIUpper = ajf1w_CI.loc[point_in_time, ajf1w_CI.columns[1]].to_numpy()
    # cif_0_w_CILower = ajf0w_CI.loc[point_in_time, ajf0w_CI.columns[0]].to_numpy()
    # cif_0_w_CIUpper = ajf0w_CI.loc[point_in_time, ajf0w_CI.columns[1]].to_numpy()

    if fig_outfile:
        ax = plt.subplot(111)
        # ajf1.plot(ax=ax)
        ajf1w.plot(ax=ax, loc=slice(0., controlled_t2e.max()))  # 0, 180
        # ajf0.plot(ax=ax)
        ajf0w.plot(ax=ax, loc=slice(0., controlled_t2e.max()))
        add_at_risk_counts(ajf1w, ajf0w, ax=ax)
        plt.tight_layout()

        # plt.ylim([0, ajf0w.cumulative_density_.loc[180][0] * 3])

        plt.title(title, fontsize=12)
        plt.savefig(fig_outfile.replace('-km.png', '-cumIncidence.png'))
        plt.close()

    # Part-3: Cox for hazard ratio
    # https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html
    # Competing risk sceneriao: 0 for censoring, 1 for target event, 2 for competing risk death
    # --> competing risk 2, death, as censoring in cox model. Only caring event 1
    # https://github.com/CamDavidsonPilon/lifelines/issues/619
    cph = CoxPHFitter()
    cox_data = pd.DataFrame(
        {'T': events_t2e, 'event': flag_2binary(events_flag), 'treatment': golds_treatment, 'weights': weights})
    try:
        cph.fit(cox_data, 'T', 'event', weights_col='weights', robust=True)
        HR = cph.hazard_ratios_['treatment']
        CI = np.exp(cph.confidence_intervals_.values.reshape(-1))
        test_results = logrank_test(treated_t2e, controlled_t2e, event_observed_A=flag_2binary(treated_flag),
                                    event_observed_B=flag_2binary(controlled_flag), weights_A=treated_w, weights_B=controlled_w, )
        test_p = test_results.p_value
        hr_different_time = cph.compute_followup_hazard_ratios(cox_data, point_in_time)
        hr_different_time = hr_different_time['treatment'].to_numpy()

        cph_ori = CoxPHFitter()
        cox_data_ori = pd.DataFrame({'T': events_t2e, 'event': flag_2binary(events_flag), 'treatment': golds_treatment})
        cph_ori.fit(cox_data_ori, 'T', 'event')
        HR_ori = cph_ori.hazard_ratios_['treatment']
        CI_ori = np.exp(cph_ori.confidence_intervals_.values.reshape(-1))
        test_results_ori = logrank_test(treated_t2e, controlled_t2e, event_observed_A=flag_2binary(treated_flag),
                                        event_observed_B=flag_2binary(controlled_flag))
        test_p_ori = test_results_ori.p_value
        hr_different_time_ori = cph_ori.compute_followup_hazard_ratios(cox_data, point_in_time)
        hr_different_time_ori = hr_different_time_ori['treatment'].to_numpy()

        cph_inter = CoxPHFitter()
        cox_data_inter = pd.DataFrame(
            {'T': events_t2e, 'event': flag_2binary(events_flag), 'treatment': golds_treatment, 'weights': weights,
             'database': database_flag, 'inter': database_flag * golds_treatment})
        cph_inter.fit(cox_data_inter, 'T', 'event', weights_col='weights', robust=True)
        HR_inter_treat = cph_inter.hazard_ratios_['treatment']
        CI_inter_treat = np.exp(cph_inter.confidence_intervals_.loc["treatment", :].values.reshape(-1))
        HR_inter = cph_inter.hazard_ratios_['inter']
        CI_inter = np.exp(cph_inter.confidence_intervals_.loc["inter", :].values.reshape(-1))
        HR_inter_database = cph_inter.hazard_ratios_['database']
        CI_inter_database = np.exp(cph_inter.confidence_intervals_.loc["database", :].values.reshape(-1))
        # test_results = logrank_test(treated_t2e, controlled_t2e, event_observed_A=flag_2binary(treated_flag),
        #                             event_observed_B=flag_2binary(controlled_flag), weights_A=treated_w,
        #                             weights_B=controlled_w, )
        # test_p = test_results.p_value
        # hr_different_time = cph.compute_followup_hazard_ratios(cox_data, point_in_time)
        # hr_different_time = hr_different_time['treatment'].to_numpy()

    except:
        cph = HR = CI = test_p_ori = None
        cph_ori = HR_ori = CI_ori = test_p = None
        hr_different_time = hr_different_time_ori = [np.nan]*len(point_in_time)
        HR_inter_treat = CI_inter_treat = HR_inter = CI_inter = HR_inter_database = CI_inter_database = cph_inter = None

    return (kmf1, kmf0, ate, point_in_time, survival_1, survival_0, results), \
           (kmf1_w, kmf0_w, ate_w, point_in_time, survival_1_w, survival_0_w, results_w), \
           (HR_ori, CI_ori, test_p_ori, cph_ori, hr_different_time_ori), \
           (HR, CI, test_p, cph, hr_different_time), \
           (HR_inter_treat, CI_inter_treat, HR_inter, CI_inter, HR_inter_database, CI_inter_database, cph_inter), \
           (ajf1, ajf0, cifdiff, point_in_time, cif_1, cif_0, cif_1_CILower, cif_1_CIUpper, cif_0_CILower, cif_0_CIUpper), \
           (ajf1w, ajf0w, cifdiff_w, point_in_time, cif_1_w, cif_0_w, cif_1_w_CILower, cif_1_w_CIUpper, cif_0_w_CILower, cif_0_w_CIUpper)
