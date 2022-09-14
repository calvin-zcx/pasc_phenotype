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
import seaborn as sns

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['oneflorida', 'V15_COVID19'], default='oneflorida',
                        help='data bases')
    parser.add_argument('--site', choices=['COL', 'MSHS', 'MONTE', 'NYU', 'WCM', 'ALL', 'all'], default='all',
                        help='site dataset')
    parser.add_argument('--severity', choices=['all', 'outpatient', 'inpatient', 'icu'], default='all')

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--negative_ratio', type=int, default=5)
    # parser.add_argument('--run_model', choices=['LSTM', 'LR', 'MLP', 'XGBOOST', 'LIGHTGBM'], default='MLP')
    args = parser.parse_args()

    # More args
    args.data_file = r'../data/{}/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_{}.csv'.format(
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
    # python screen_med_rxnorm_v2.py --dataset V15_COVID19 --site ALL --severity icu 2>&1 | tee  log/screen_med_rxnorm_insight_ALL_icu.txt
    # python screen_med_rxnorm_v2.py --dataset V15_COVID19 --site ALL --severity inpatient 2>&1 | tee  log/screen_med_rxnorm_insight_ALL_inpatient.txt
    # python screen_med_rxnorm_v2.py --dataset V15_COVID19 --site ALL --severity outpatient 2>&1 | tee  log/screen_med_rxnorm_insight_ALL_outpatient.txt
    # python screen_med_rxnorm_v2.py --dataset oneflorida --site all --severity all 2>&1 | tee  log/screen_med_rxnorm_oneflorida_all_all.txt
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
    df = pd.read_csv(args.data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
    # because a patid id may occur in multiple sites. patid were site specific
    print('df.shape:', df.shape)
    if args.severity == 'inpatient':
        print('Considering inpatient/hospitalized cohorts but not ICU')
        df = df.loc[(df['hospitalized'] == 1) & (df['ventilation'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif args.severity == 'icu':
        print('Considering ICU (hospitalized ventilation or critical care) cohorts')
        df = df.loc[(((df['hospitalized'] == 1) & (df['ventilation'] == 1)) | (df['criticalcare'] == 1)), :].copy()
    elif args.severity == 'outpatient':
        print('Considering outpatient cohorts')
        df = df.loc[(df['hospitalized'] == 0) & (df['criticalcare'] == 0), :].copy()
    else:
        print('Considering ALL cohorts')

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

    # Load index information
    with open(r'../data/mapping/rxnorm_ingredient_mapping_combined_moietyfirst.pkl', 'rb') as f:
        rxnorm_ing = pickle.load(f)
        print('Load rxRNOM_CUI to ingredient mapping done! len(rxnorm_atc):', len(rxnorm_ing))
        record_example = next(iter(rxnorm_ing.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/rxnorm_atc_mapping.pkl', 'rb') as f:
        rxnorm_atc = pickle.load(f)
        print('Load rxRNOM_CUI to ATC mapping done! len(rxnorm_atc):', len(rxnorm_atc))
        record_example = next(iter(rxnorm_atc.items()))
        print('e.g.:', record_example)

    rxing_index = utils.load(r'../data/mapping/selected_rxnorm_index.pkl')
    covidmed_codes = utils.load(r'../data/mapping/query3_medication_codes.pkl')
    covidmed_column_names = [
        'Anti-platelet Therapy', 'Aspirin', 'Baricitinib', 'Bamlanivimab Monoclonal Antibody Treatment',
        'Bamlanivimab and Etesevimab Monoclonal Antibody Treatment',
        'Casirivimab and Imdevimab Monoclonal Antibody Treatment',
        'Any Monoclonal Antibody Treatment (Bamlanivimab, Bamlanivimab and Etesevimab, Casirivimab and Imdevimab, Sotrovimab, and unspecified monoclonal antibodies)',
        'Colchicine', 'Corticosteroids', 'Dexamethasone', 'Factor Xa Inhibitors', 'Fluvoxamine', 'Heparin',
        'Inhaled Steroids', 'Ivermectin', 'Low Molecular Weight Heparin', 'Molnupiravir', 'Nirmatrelvir',
        'Paxlovid', 'Remdesivir', 'Ritonavir', 'Sotrovimab Monoclonal Antibody Treatment',
        'Thrombin Inhibitors', 'Tocilizumab (Actemra)', 'PX: Convalescent Plasma']

    # %% 2. PASC specific cohorts for causal inference
    drug_name_list = covidmed_column_names + list(rxing_index.keys())
    causal_results = []
    for i, pasc in tqdm(enumerate(drug_name_list, start=1), total=len(drug_name_list)):
        # if i < 31:
        #     continue
        # bulid specific cohorts:

        # if pasc != 'Factor Xa Inhibitors':
        #     continue

        if pasc.isnumeric():
            drugname = rxing_index[pasc][6]
            if pd.isna(drugname):
                drugname = pasc

            pasc_flag = df['med-out@' + pasc]
            pasc_t2e = df['med-t2e@' + pasc]  # .astype('float')
            pasc_baseline = df['med-base@' + pasc]
        else:
            drugname = pasc
            pasc_flag = df['covidmed-out@' + pasc]
            pasc_t2e = df['covidmed-t2e@' + pasc]  # .astype('float')
            pasc_baseline = df['covidmed-base@' + pasc]

        print('\nIn screening:', i, pasc, drugname)

        # Select population free of outcome at baseline
        idx = (pasc_baseline < 1)
        # Select negative: pos : neg = 1:2 for IPTW
        covid_label = df_label[idx]
        n_covid_pos = covid_label.sum()
        n_covid_neg = (covid_label == 0).sum()
        sampled_neg_index = covid_label[(covid_label == 0)].sample(n=args.negative_ratio * n_covid_pos,
                                                                   replace=False,
                                                                   random_state=args.random_seed).index
        pos_neg_selected = pd.Series(False, index=pasc_baseline.index)
        pos_neg_selected[sampled_neg_index] = True
        pos_neg_selected[covid_label[covid_label == 1].index] = True
        #
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
            'C': 10 ** np.arange(-2, 0.5, 0.5),
            'max_iter': [200],  # [100, 200, 500],
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
            (np.abs(smd) > SMD_THRESHOLD).sum(),  # should using abs, because abs set False above, 2022-09-07
            (np.abs(smd_weighted) > SMD_THRESHOLD).sum())
        )
        out_file_balance = r'../data/{}/output/character/outcome/MED-{}-new-trim/{}-{}-results.csv'.format(args.dataset,
                                                                                                      args.severity, i,
                                                                                                      pasc)

        utils.check_and_mkdir(out_file_balance)
        model.results.to_csv(out_file_balance)  # args.save_model_filename +

        df_summary = summary_covariate(covs_array, covid_label, iptw, smd, smd_weighted, before, after)
        df_summary.to_csv(
            r'../data/{}/output/character/outcome/MED-{}-new-trim/{}-{}-evaluation_balance.csv'.format(args.dataset,
                                                                                                  args.severity, i,
                                                                                                  pasc))
        dfps = pd.DataFrame({'ps': ps, 'iptw': iptw, 'covid': covid_label})

        dfps.to_csv(
            r'../data/{}/output/character/outcome/MED-{}-new-trim/{}-{}-evaluation_ps-iptw.csv'.format(args.dataset,
                                                                                                  args.severity, i,
                                                                                                  pasc))
        try:
            figout = r'../data/{}/output/character/outcome/MED-{}-new-trim/{}-{}-{}-PS.png'.format(
                args.dataset, args.severity,
                i, pasc, drugname)
            print('Dump ', figout)
            ax = plt.subplot(111)
            sns.histplot(
                dfps, x="ps", hue="covid", element="step",
                stat="percent", common_norm=False, bins=25,
            )
            plt.tight_layout()
            # plt.show()
            plt.title(pasc + '-' + drugname, fontsize=12)
            plt.savefig(figout)
            plt.close()
        except Exception as e:
            print('Dump Error', figout)
            print(str(e))
            plt.close()

        try:
            km, km_w, cox, cox_w, cif, cif_w = weighted_KM_HR(
                covid_label, iptw, pasc_flag, pasc_t2e,
                fig_outfile=r'../data/{}/output/character/outcome/MED-{}-new-trim/{}-{}-{}-km.png'.format(
                    args.dataset,
                    args.severity, i,
                    pasc, drugname),
                title=pasc + '-' + drugname)
        except:
            # there are fig file name errors due to drugname
            plt.close()
            km, km_w, cox, cox_w, cif, cif_w = weighted_KM_HR(
                covid_label, iptw, pasc_flag, pasc_t2e,
                fig_outfile=r'../data/{}/output/character/outcome/MED-{}-new-trim/{}-{}-km.png'.format(
                    args.dataset, args.severity, i, pasc),
                title=pasc + '-' + drugname)

        try:
            # change 2022-03-20 considering competing risk 2
            _results = [i, pasc,
                        covid_label.sum(), (covid_label == 0).sum(),
                        (pasc_flag[covid_label == 1] == 1).sum(), (pasc_flag[covid_label == 0] == 1).sum(),
                        (pasc_flag[covid_label == 1] == 1).mean(), (pasc_flag[covid_label == 0] == 1).mean(),
                        (pasc_flag[covid_label == 1] == 2).sum(), (pasc_flag[covid_label == 0] == 2).sum(),
                        (pasc_flag[covid_label == 1] == 2).mean(), (pasc_flag[covid_label == 0] == 2).mean(),
                        (np.abs(smd) > SMD_THRESHOLD).sum(), (np.abs(smd_weighted) > SMD_THRESHOLD).sum(),
                        np.abs(smd).max(), np.abs(smd_weighted).max(),
                        km[2], km[3], km[6].p_value,
                        cif[2], cif[4], cif[5], cif[6], cif[7], cif[8], cif[9],
                        km_w[2], km_w[3], km_w[6].p_value,
                        cif_w[2], cif_w[4], cif_w[5], cif_w[6], cif_w[7], cif_w[8], cif_w[9],
                        cox[0], cox[1], cox[3].summary.p.treatment if pd.notna(cox[3]) else np.nan, cox[2], cox[4],
                        cox_w[0], cox_w[1], cox_w[3].summary.p.treatment if pd.notna(cox_w[3]) else np.nan, cox_w[2],
                        cox_w[4], model.best_hyper_paras]
            causal_results.append(_results)
            results_columns_name = [
                'i', 'pasc', 'covid+', 'covid-',
                'no. pasc in +', 'no. pasc in -', 'mean pasc in +', 'mean pasc in -',
                'no. death in +', 'no. death in -', 'mean death in +', 'mean death in -',
                'no. unbalance', 'no. unbalance iptw', 'max smd', 'max smd iptw',
                'km-diff', 'km-diff-time', 'km-diff-p',
                'cif-diff', "cif_1", "cif_0", "cif_1_CILower", "cif_1_CIUpper", "cif_0_CILower", "cif_0_CIUpper",
                'km-w-diff', 'km-w-diff-time', 'km-w-diff-p',
                'cif-w-diff', "cif_1_w", "cif_0_w", "cif_1_w_CILower", "cif_1_w_CIUpper", "cif_0_w_CILower",
                "cif_0_w_CIUpper",
                'hr', 'hr-CI', 'hr-p', 'hr-logrank-p', 'hr_different_time',
                'hr-w', 'hr-w-CI', 'hr-w-p', 'hr-w-logrank-p', "hr-w_different_time", 'best_hyper_paras']
            print('causal result:\n', causal_results[-1])

            if i % 50 == 0:
                pd.DataFrame(causal_results, columns=results_columns_name). \
                    to_csv(
                    r'../data/{}/output/character/outcome/MED-{}-new-trim/causal_effects_specific_med-snapshot-{}.csv'.format(
                        args.dataset, args.severity, i))
        except:
            print('Error in ', i, pasc)
            df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

            df_causal.to_csv(
                r'../data/{}/output/character/outcome/MED-{}-new-trim/causal_effects_specific_med-ERRORSAVE.csv'.format(
                    args.dataset, args.severity))

        print('done one pasc, time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

    df_causal.to_csv(r'../data/{}/output/character/outcome/MED-{}-new-trim/causal_effects_specific_med.csv'.format(
        args.dataset, args.severity))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
