import os
import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import argparse
import time
import random
import pickle
import ast
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
import tqdm as tqdm
from datetime import datetime
import functools

print = functools.partial(print, flush=True)
from misc import utils
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test, proportional_hazard_test, logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import k_fold_cross_validation
from PRModels import ml


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['OneFlorida', 'INSIGHT'], default='INSIGHT',
                        help='data bases')
    parser.add_argument('--encode', choices=['elix', 'icd_med'], default='elix',
                        help='data encoding')
    # parser.add_argument('--severity', choices=['all',
    #                                            'outpatient', 'inpatient', 'icu', 'inpatienticu',
    #                                            'female', 'male',
    #                                            'white', 'black',
    #                                            'less65', '65to75', '75above', '20to40', '40to55', '55to65', 'above65',
    #                                            'Anemia', 'Arrythmia', 'CKD', 'CPD-COPD', 'CAD',
    #                                            'T2D-Obesity', 'Hypertension', 'Mental-substance', 'Corticosteroids',
    #                                            'healthy'],
    #                     default='all')

    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()

    # More args
    # args.data_file = r'../data/{}/output/character/matrix_cohorts_covid_4manuNegNoCovid_bool_{}.csv'.format(
    #     args.dataset,
    #     args.site)
    args.data_dir = r'output/dataset/{}/{}/'.format(args.dataset, args.encode)
    args.out_dir = r'output/factors/{}/{}/'.format(args.dataset, args.encode)

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.out_dir)
    return args


def collect_feature_columns(args, df):
    col_names = []
    col_names += ['hospitalized', 'ventilation', 'criticalcare']

    # col_names += ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
    col_names += ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years']

    # col_names += ['Female', 'Male', 'Other/Missing']
    col_names += ['Female', 'Male']

    # col_names += ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    col_names += ['Asian', 'Black or African American', 'White', 'Other']

    col_names += ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']

    # col_names += ['inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits 3-4', 'inpatient visits >=5',
    #               'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits 3-4', 'outpatient visits >=5',
    #               'emergency visits 0', 'emergency visits 1-2', 'emergency visits 3-4', 'emergency visits >=5']
    col_names += ['inpatient visits 0', 'inpatient visits 1-4', 'inpatient visits >=5',
                  'outpatient visits 0', 'outpatient visits 1-4', 'outpatient visits >=5',
                  'emergency visits 0', 'emergency visits 1-4', 'emergency visits >=5']

    # col_names += ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49', 'ADI50-59', 'ADI60-69', 'ADI70-79',
    #               'ADI80-89', 'ADI90-100']
    col_names += ['ADI1-19', 'ADI20-39', 'ADI40-59', 'ADI60-79', 'ADI80-100']

    col_names += ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight', 'BMI: 25-<30 overweight ',
                  'BMI: >=30 obese ', 'BMI: missing']

    col_names += ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']

    col_names += ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21', '07/21-11/21']

    col_names += ['num_Comorbidity=0', 'num_Comorbidity=1', 'num_Comorbidity=2', 'num_Comorbidity=3',
                  'num_Comorbidity=4', 'num_Comorbidity>=5']

    if args.encode == 'icd_med':
        col_names += list(df.columns)[df.columns.get_loc('death t2e') + 1:df.columns.get_loc('label')]
    else:
        col_names += ["DX: Alcohol Abuse", "DX: Anemia", "DX: Arrythmia", "DX: Asthma", "DX: Cancer",
                      "DX: Chronic Kidney Disease", "DX: Chronic Pulmonary Disorders", "DX: Cirrhosis",
                      "DX: Coagulopathy", "DX: Congestive Heart Failure",
                      "DX: COPD", "DX: Coronary Artery Disease", "DX: Dementia", "DX: Diabetes Type 1",
                      "DX: Diabetes Type 2", "DX: End Stage Renal Disease on Dialysis", "DX: Hemiplegia",
                      "DX: HIV", "DX: Hypertension", "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis",
                      "DX: Inflammatory Bowel Disorder", "DX: Lupus or Systemic Lupus Erythematosus",
                      "DX: Mental Health Disorders", "DX: Multiple Sclerosis", "DX: Parkinson's Disease",
                      "DX: Peripheral vascular disorders ", "DX: Pregnant",
                      "DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)",
                      "DX: Rheumatoid Arthritis", "DX: Seizure/Epilepsy",
                      "DX: Severe Obesity  (BMI>=40 kg/m2)", "DX: Weight Loss",
                      "DX: Down's Syndrome", 'DX: Other Substance Abuse', 'DX: Cystic Fibrosis',
                      'DX: Autism', 'DX: Sickle Cell'
                      ]

        col_names += ["MEDICATION: Corticosteroids", "MEDICATION: Immunosuppressant drug"]

    print('encoding:', args.encode, 'len(col_names):', len(col_names))
    print(col_names)
    return col_names


def pre_transform_feature(df):
    # col_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49', 'ADI50-59', 'ADI60-69', 'ADI70-79',
    #              'ADI80-89', 'ADI90-100']
    df['ADI1-19'] = (df['ADI1-9'] + df['ADI10-19'] >= 1).astype('int')
    df['ADI20-39'] = (df['ADI20-29'] + df['ADI30-39'] >= 1).astype('int')
    df['ADI40-59'] = (df['ADI40-49'] + df['ADI50-59'] >= 1).astype('int')
    df['ADI60-79'] = (df['ADI60-69'] + df['ADI70-79'] >= 1).astype('int')
    df['ADI80-100'] = (df['ADI80-89'] + df['ADI90-100'] >= 1).astype('int')

    df['75+ years'] = (df['75-<85 years'] + df['85+ years'] >= 1).astype('int')

    df['inpatient visits 1-4'] = (df['inpatient visits 1-2'] + df['inpatient visits 3-4'] >= 1).astype('int')
    df['outpatient visits 1-4'] = (df['outpatient visits 1-2'] + df['outpatient visits 3-4'] >= 1).astype('int')
    df['emergency visits 1-4'] = (df['emergency visits 1-2'] + df['emergency visits 3-4'] >= 1).astype('int')

    return df


def risk_factor_of_pasc(args, pasc_name, dump=True):
    infile = args.data_dir + pasc_name + '_{}'.format(
        'dx_med_' if args.encode == 'icd_med' else '') + args.dataset + '.csv'
    print('In risk_factor_of_pasc:')
    pasc = pasc_name.replace('_', '/')
    print('PASC:', pasc, 'Infile:', infile)

    df = pd.read_csv(infile)
    print('df.shape:', df.shape)
    df = pre_transform_feature(df)
    print('df.shape after pre_transform_feature:', df.shape)
    # df_label = df['covid']

    covs_columns = collect_feature_columns(args, df)

    if pasc == 'Neurocognitive disorders':
        covs_columns = [x for x in covs_columns if x != "DX: Dementia"]
    elif pasc == 'Diabetes mellitus with complication':
        covs_columns = [x for x in covs_columns if x != 'DX: Diabetes Type 2']

    pasc_flag = df['dx-out@' + pasc]
    pasc_t2e = df['dx-t2e@' + pasc]  # .astype('float')
    pasc_baseline = df['dx-base@' + pasc]

    death_flag = df['death']
    death_t2e = df['death t2e']
    # pasc_flag.loc[(death_t2e == pasc_t2e)] = 2
    print('#death:', (death_t2e == pasc_t2e).sum(),
          '#death:', (death_t2e == pasc_t2e).sum(),
          'ratio of death:', (death_t2e == pasc_t2e).mean())

    cox_data = df.loc[:, covs_columns]
    print('cox_data.shape before number filter:', cox_data.shape)
    cox_data = cox_data.loc[:, cox_data.columns[cox_data.mean() >= 0.001]]
    print('cox_data.shape after number filter:', cox_data.shape)

    model = ml.CoxPrediction(random_seed=args.random_seed).cross_validation_fit(
        cox_data, pasc_t2e, pasc_flag, kfold=5, scoring_method="concordance_index")

    if dump:
        utils.check_and_mkdir(args.out_dir)
        model.risk_results.reset_index().sort_values(by=['HR'], ascending=False).to_csv(
            args.out_dir + pasc_name + '-riskFactor.csv')
        model.results.sort_values(by=['E[fit]'], ascending=False).to_csv(
            args.out_dir + pasc_name + '-modeSelection.csv')

    return model


if __name__ == '__main__':
    # python screen_risk_factors.py --dataset INSIGHT --encode elix 2>&1 | tee  log/screen_risk_factors-insight-elix.txt
    # python screen_risk_factors.py --dataset OneFlorida --encode elix 2>&1 | tee  log/screen_risk_factors-OneFlorida-elix.txt

    start_time = time.time()
    args = parse_args()

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    print('args: ', args)
    print('random_seed: ', args.random_seed)
    # pasc_name = 'Neurocognitive disorders'  # 'Diabetes mellitus with complication' # 'Anemia' #
    # model = risk_factor_of_pasc(args, pasc_name, dump=False)

    causal_res = pd.read_excel('output/causal_effects_specific_withMedication_v3.xlsx',
                               sheet_name='diagnosis')
    filtered_data = causal_res.loc[(causal_res['hr-w'] > 1) & (causal_res['hr-w-p'] < 0.01), :]
    filtered_data = filtered_data.reset_index(drop=True)
    pasc_list = list(filtered_data['pasc'])

    idx = 0
    for selected_PASC in pasc_list:  # ['Neurocognitive disorders']:  # , 'Diabetes mellitus with complication']:  # tqdm(pasc_list):
        pasc_name = selected_PASC.replace('/', '_')
        model = risk_factor_of_pasc(args, pasc_name)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
