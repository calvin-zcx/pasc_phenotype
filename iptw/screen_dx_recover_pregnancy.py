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
                                               '1stwave', 'delta', 'alpha', 'preg-pos-neg'],
                        default='preg-pos-neg')
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument('--negative_ratio', type=int, default=10)  # 5
    parser.add_argument('--selectpasc', action='store_true')

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
        df = df.loc[(df['index date'] >= datetime.datetime(2020, 3, 1, 0, 0)) & (df['index date'] < datetime.datetime(2020, 10, 1, 0, 0)), :].copy()
    elif severity == 'delta':
        print('Considering patients in Delta wave, June-1-2021 to Nov.-30-2021')
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 6, 1, 0, 0)) & (df['index date'] < datetime.datetime(2021, 12, 1, 0, 0)), :].copy()
    elif severity == 'alpha':
        print('Considering patients in Alpha + others wave, Oct.-1-2020 to May-31-2021')
        df = df.loc[(df['index date'] >= datetime.datetime(2020, 10, 1, 0, 0)) & (
                    df['index date'] < datetime.datetime(2021, 6, 1, 0, 0)), :].copy()
    else:
        print('Considering ALL cohorts')

    if severity == 'preg-pos-neg':
        # select female
        print('Before selecting female, df.shape', df.shape)
        df = df.loc[df['Female'] == 1, :] #.copy()
        print('After selecting female, df.shape', df.shape)

        # select age
        print('Before selecting age <= 50, df.shape', df.shape)
        df = df.loc[df['age'] <= 50, :] #.copy()
        print('After selecting age <= 50, df.shape', df.shape)

        # select index date
        print('Before selecting index date < 2022-6-1, df.shape', df.shape)
        df = df.loc[(df['index date'] < datetime.datetime(2022, 6, 1, 0, 0)), :]#.copy()
        print('After selecting index date < 2022-6-1, df.shape', df.shape)

        # pregnant patients only
        print('Before selecting pregnant, df.shape', df.shape)
        df = df.loc[df['flag_pregnancy'] == 1, :]#.copy()
        print('After selecting pregnant, df.shape', df.shape)

        # infection during pregnancy period
        print('Before selecting infection in gestational period, df.shape', df.shape)
        df = df.loc[(df['index date'] >= df['flag_pregnancy_start_date']) & (
                df['index date'] < df['flag_delivery_date']), :].copy()
        print('After selecting infection in gestational period, df.shape', df.shape)

    return df


if __name__ == "__main__":
    # python screen_dx_recover_pregnancy.py --site all --severity preg-pos-neg 2>&1 | tee  log_recover/screen_dx_recover_pregnancy_all_preg-pos-neg.txt
    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # print('save_model_filename', args.save_model_filename)

    # %% 1. Load  Data
    print('In cohorts_characterization_build_data...')
    if args.site == 'all':
        sites = ['mcw', 'nebraska', 'utah', 'utsw',
                 'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
                 'ufh',  'usf',  'miami',  # 'emory', 'nch',
                 'pitt', 'psu', 'temple', 'michigan',
                 'ochsner', 'ucsf', #'lsu',
                 'vumc']

        # sites = ['wcm', 'montefiore', 'mshs',]

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
        data_file = r'../data/recover/output/pregnancy_data/pregnancy_{}.csv'.format(site)
        # Load Covariates Data
        print('Load data covariates file:', data_file)
        df = pd.read_csv(data_file,  dtype={'patid': str, 'site': str, 'zip': str},
                         parse_dates=['index date', 'flag_delivery_date', 'flag_pregnancy_start_date',
                                      'flag_pregnancy_end_date'])
        # because a patid id may occur in multiple sites. patid were site specific
        print('df.shape:', df.shape)
        df_list.append(df)

    # combine all sites and select subcohorts
    df = pd.concat(df_list, ignore_index=True)
    df = select_subpopulation(df, args.severity)

    # 'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids'
    print('Severity cohorts:', args.severity, 'df.shape:', df.shape)
    col_names = pd.Series(df.columns)
    df_info = df[['patid', 'site', 'index date', 'hospitalized',
                  'ventilation', 'criticalcare', 'maxfollowup', 'death', 'death t2e',
                  'flag_pregnancy', 'flag_delivery_date', 'flag_pregnancy_start_date',
                  'flag_pregnancy_gestational_age', 'flag_pregnancy_end_date', 'flag_maternal_age']]  # 'Unnamed: 0',
    # df_info_list.append(df_info)
    df_label = df['covid']
    # df_label_list.append(df_label)

    df_outcome_cols = ['death', 'death t2e'] + [x for x in
                    list(df.columns)
                    if x.startswith('dx') or x.startswith('smm')
                    ]
    df_outcome = df.loc[:, df_outcome_cols]  #.astype('float')
    # df_outcome_list.append(df_outcome)

    covs_columns = ['hospitalized', 'ventilation', 'criticalcare',] + \
                   [x for x in
                    list(df.columns)[
                    df.columns.get_loc('pregage:18-<25 years'):(df.columns.get_loc('obc:Delivery BMI\xa0>\xa040') + 1)]
                    if (not x.startswith('YM:')) and (x not in ['Female', 'Male', 'Other/Missing',
                                                               'outpatient visits 0', 'outpatient visits 1-2',
                                                               'outpatient visits 3-4', 'outpatient visits >=5', ])
                    ]

    days = (df['index date'] - datetime.datetime(2020, 3, 1, 0, 0)).apply(lambda x:x.days)
    days = np.array(days).reshape((-1,1))
    # days_norm = (days - days.min())/(days.max() - days.min())
    spline = SplineTransformer(degree=3, n_knots=7)
    days_sp = spline.fit_transform(np.array(days))  # identical
    # days_norm_sp = spline.fit_transform(days_norm) # identical

    print('len(covs_columns):', len(covs_columns))

    # delet old date feature and use spline
    covs_columns = [x for x in covs_columns if x not in
                    ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21',
                     '07/21-10/21', '11/21-02/22', '03/22-06/22', '07/22-10/22'] ]
    print('after delete 8 days len(covs_columns):', len(covs_columns))
    df_covs = df.loc[:, covs_columns].astype('float')

    new_day_cols = ['days_splie_{}'.format(i) for i in range(days_sp.shape[1])]
    covs_columns += new_day_cols
    print('after adding {} days len(covs_columns):'.format(days_sp.shape[1]), len(covs_columns))
    for i in range(days_sp.shape[1]):
        print('add', i, new_day_cols[i])
        df_covs[new_day_cols[i]] = days_sp[:, i]

    # days between pregnancy and infection
    days_since_preg = (df['index date'] - df['flag_pregnancy_start_date']).apply(lambda x: x.days)
    days_since_preg = np.array(days_since_preg).reshape((-1, 1))
    spline = SplineTransformer(degree=3, n_knots=5)
    days_since_preg_sp = spline.fit_transform(np.array(days_since_preg))  # identical

    new_days_since_preg_cols = ['days_since_preg_splie_{}'.format(i) for i in range(days_since_preg_sp.shape[1])]
    covs_columns += new_days_since_preg_cols
    print('after adding {} days len(covs_columns):'.format(days_since_preg_sp.shape[1]), len(covs_columns))
    for i in range(days_since_preg_sp.shape[1]):
        print('add', i, new_days_since_preg_cols[i])
        df_covs[new_days_since_preg_cols[i]] = days_since_preg_sp[:, i]

    # df_covs_list.append(df_covs)
    print(ith, 'df.shape:', df.shape, 'df_covs.shape:', df_covs.shape)

    # df = pd.concat(df_outcome_list, ignore_index=True)
    # df_info = pd.concat(df_info_list, ignore_index=True)
    # df_label = pd.concat(df_label_list, ignore_index=True)
    # df_covs = pd.concat(df_covs_list, ignore_index=True)
    df = df_outcome

    print('all',
          'df.shape', df.shape,
          'df_info.shape:', df_info.shape,
          'df_label.shape:', df_label.shape,
          'df_covs.shape:', df_covs.shape)
    print('Done load data! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


    # Load index information
    with open(r'../data/mapping/icd_pasc_mapping.pkl', 'rb') as f:
        icd_pasc = pickle.load(f)
        print('Load ICD-10 to PASC mapping done! len(icd_pasc):', len(icd_pasc))
        record_example = next(iter(icd_pasc.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/pasc_index_mapping.pkl', 'rb') as f:
        pasc_encoding = pickle.load(f)
        print('Load PASC to encoding mapping done! len(pasc_encoding):', len(pasc_encoding))
        record_example = next(iter(pasc_encoding.items()))
        print('e.g.:', record_example)

    SMMpasc_encoding = utils.load(r'../data/mapping/SMMpasc_index_mapping.pkl')

    # %% 2. PASC specific cohorts for causal inference
    # if args.selectpasc:
    #     df_select = pd.read_excel(
    #         r'../data/V15_COVID19/output/character/outcome/DX-all/Diagnosis_Medication_refine_Organ_Domain-V2-4plot.xlsx',
    #         sheet_name='diagnosis').set_index('i')
    #     df_select = df_select.loc[df_select['Hazard Ratio, Adjusted, P-Value'] <= 0.05, :]  #
    #     df_select = df_select.loc[df_select['Hazard Ratio, Adjusted'] > 1, :]
    #     selected_list = df_select.index.tolist()
    #     print('Selected: len(selected_list):', len(selected_list))
    #     print(df_select['PASC Name Simple'])
    pasc_encoding.update(SMMpasc_encoding)

    causal_results = []
    results_columns_name = []
    for i, pasc in tqdm(enumerate(pasc_encoding.keys(), start=1), total=len(pasc_encoding)):
        # bulid specific cohorts:
        # if args.selectpasc:
        #     if i not in selected_list:
        #         print('Skip:', i, pasc, 'because args.selectpasc, p<=0.05, hr > 1 in Insight')
        #         continue
        print('\n In screening:', i, pasc)
        if not pasc.startswith('smm:'):
            pasc_flag = (df['dx-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['dx-t2e@' + pasc].astype('float')
            pasc_baseline = df['dx-base@' + pasc]
        else:
            pasc_flag = (df['smm-out@' + pasc].copy() >= 1).astype('int')
            pasc_t2e = df['smm-t2e@' + pasc].astype('float')
            pasc_baseline = df['smm-base@' + pasc]


        # considering competing risks
        death_flag = df['death']
        death_t2e = df['death t2e']
        pasc_flag.loc[(death_t2e == pasc_t2e)] = 2
        print('#death:', (death_t2e == pasc_t2e).sum(), ' #death in covid+:', df_label[(death_t2e == pasc_t2e)].sum(),
              'ratio of death in covid+:', df_label[(death_t2e == pasc_t2e)].mean())

        # Select population free of outcome at baseline
        idx = (pasc_baseline < 1)
        # Select negative: pos : neg = 1:2 for IPTW
        covid_label = df_label[idx]
        n_covid_pos = covid_label.sum()
        n_covid_neg = (covid_label == 0).sum()

        print('n_covid_pos:', n_covid_pos, 'n_covid_neg:', n_covid_neg, )

        if args.negative_ratio * n_covid_pos < n_covid_neg:
            print('replace=False, args.negative_ratio * n_covid_pos:', args.negative_ratio * n_covid_pos,
                  'n_covid_neg:', n_covid_neg)
            sampled_neg_index = covid_label[(covid_label == 0)].sample(n=int(args.negative_ratio * n_covid_pos),
                                                                       replace=False,
                                                                       random_state=args.random_seed).index
        else:
            print('replace=True')
            # print('Use negative patients with replacement, args.negative_ratio * n_covid_pos:',
            #       args.negative_ratio * n_covid_pos,
            #       'n_covid_neg:', n_covid_neg)
            # sampled_neg_index = covid_label[(covid_label == 0)].sample(n=args.negative_ratio * n_covid_pos,
            #                                                            replace=True,
            #                                                            random_state=args.random_seed).index
            print('Not using sample with replacement. Use all negative patients, args.negative_ratio * n_covid_pos:',
                  args.negative_ratio * n_covid_pos,
                  'n_covid_pos:', n_covid_pos,
                  'n_covid_neg:', n_covid_neg)
            sampled_neg_index = covid_label[(covid_label == 0)].index

        pos_neg_selected = pd.Series(False, index=pasc_baseline.index)
        pos_neg_selected[sampled_neg_index] = True
        pos_neg_selected[covid_label[covid_label == 1].index] = True
        #
        pat_info = df_info.loc[pos_neg_selected, :]
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
            'C': 10 ** np.arange(-2, 1.5, 0.5),
            'max_iter': [150],  # [100, 200, 500],
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
            (np.abs(smd) > SMD_THRESHOLD).sum(),
            (np.abs(smd_weighted) > SMD_THRESHOLD).sum())
        )
        out_file_balance = r'../data/recover/output/results/DX-{}{}/{}-{}-results.csv'.format(
            args.severity,
            '-select' if args.selectpasc else '',
            i,
            pasc.replace(':', '-').replace('/', '-'))
        utils.check_and_mkdir(out_file_balance)
        model.results.to_csv(out_file_balance)  # args.save_model_filename +

        df_summary = summary_covariate(covs_array, covid_label, iptw, smd, smd_weighted, before, after)
        df_summary.to_csv(
            '../data/recover/output/results/DX-{}{}/{}-{}-evaluation_balance.csv'.format(
                args.severity,
                '-select' if args.selectpasc else '',
                i, pasc.replace(':', '-').replace('/', '-')))

        dfps = pd.DataFrame({'ps': ps, 'iptw': iptw, 'covid': covid_label})

        dfps.to_csv(
            '../data/recover/output/results/DX-{}{}/{}-{}-evaluation_ps-iptw.csv'.format(
                args.severity,
                '-select' if args.selectpasc else '',
                i, pasc.replace(':', '-').replace('/', '-')))
        try:
            figout = r'../data/recover/output/results/DX-{}{}/{}-{}-PS.png'.format(
                args.severity,
                '-select' if args.selectpasc else '',
                i, pasc.replace(':', '-').replace('/', '-'))
            print('Dump ', figout)

            ax = plt.subplot(111)
            sns.histplot(
                dfps, x="ps", hue="covid", element="step",
                stat="percent", common_norm=False, bins=25,
            )
            plt.tight_layout()
            # plt.show()
            plt.title(pasc, fontsize=12)
            plt.savefig(figout)
            plt.close()
        except Exception as e:
            print('Dump Error', figout)
            print(str(e))
            plt.close()

        km, km_w, cox, cox_w, cif, cif_w = weighted_KM_HR(
            covid_label, iptw, pasc_flag, pasc_t2e,
            fig_outfile=r'../data/recover/output/results/DX-{}{}/{}-{}-km.png'.format(
                args.severity,
                '-select' if args.selectpasc else '',
                i, pasc.replace(':', '-').replace('/', '-')),
            title=pasc)

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
                'cif-w-diff', "cif_1_w", "cif_0_w", "cif_1_w_CILower", "cif_1_w_CIUpper", "cif_0_w_CILower", "cif_0_w_CIUpper",
                'hr', 'hr-CI', 'hr-p', 'hr-logrank-p', 'hr_different_time',
                'hr-w', 'hr-w-CI', 'hr-w-p', 'hr-w-logrank-p', "hr-w_different_time", 'best_hyper_paras']
            print('causal result:\n', causal_results[-1])

            if i % 50 == 0:
                pd.DataFrame(causal_results, columns=results_columns_name). \
                    to_csv(r'../data/recover/output/results/DX-{}{}/causal_effects_specific-snapshot-{}.csv'.format(
                    args.severity, '-select' if args.selectpasc else '', i))
        except:
            print('Error in ', i, pasc)
            df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

            df_causal.to_csv(
                r'../data/recover/output/results/DX-{}{}/causal_effects_specific-ERRORSAVE.csv'.format(
                    args.severity,
                    '-select' if args.selectpasc else '',))

        print('done one pasc, time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    df_causal = pd.DataFrame(causal_results, columns=results_columns_name)

    df_causal.to_csv(
        r'../data/recover/output/results/DX-{}{}/causal_effects_specific.csv'.format(
            args.severity,
            '-select' if args.selectpasc else ''))
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
