import fnmatch
import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import datetime
from misc import utils
import functools
import fnmatch
from lifelines import KaplanMeierFitter, CoxPHFitter

print = functools.partial(print, flush=True)

# from iptw.PSModels import ml
# from iptw.evaluation import *


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    # Input
    parser.add_argument('--dataset', choices=['OneFlorida', 'INSIGHT', 'combined'], default='INSIGHT',
                        help='data bases')
    parser.add_argument('--encode', choices=['elix', 'icd_med'], default='elix',
                        help='data encoding')
    parser.add_argument('--population', choices=['positive', 'negative', 'all'], default='positive')
    parser.add_argument('--severity', choices=['all', 'outpatient', "inpatienticu",
                                               'inpatient', 'icu', 'ventilation', ], default='all')
    parser.add_argument('--goal', choices=['anypasc', 'allpasc', 'anyorgan', 'allorgan',
                                           'anypascsevere', 'anypascmoderate'],
                        default='anypascsevere')
    parser.add_argument("--random_seed", type=int, default=0)

    args = parser.parse_args()

    # More args
    if args.dataset == 'INSIGHT':
        args.data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL-PosOnly.csv'
        args.processed_data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL-PosOnly-anyPASC.csv'
        # args.data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL.csv'
        # args.processed_data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL-anyPASC.csv'

    elif args.dataset == 'OneFlorida':
        args.data_file = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all-PosOnly.csv'
        args.processed_data_file = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all-PosOnly-anyPASC.csv'
        # args.data_file = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all.csv'
        # args.processed_data_file = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all-anyPASC.csv'
    else:
        raise ValueError

    args.data_dir = r'output/dataset/{}/{}/'.format(args.dataset, args.encode)
    args.out_dir = r'output/factors/{}/{}/'.format(args.dataset, args.encode)

    # args.processed_data_file = r'output/dataset/{}/df_cohorts_covid_4manuNegNoCovidV2_bool_all-PosOnly-{}.csv'.format(
    #     args.dataset, args.encode)

    if args.random_seed < 0:
        from datetime import datetime
        args.random_seed = int(datetime.now())

    # args.save_model_filename = os.path.join(args.output_dir, '_S{}{}'.format(args.random_seed, args.run_model))
    # utils.check_and_mkdir(args.out_dir)
    return args


def pre_transform_feature(df):
    # col_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49', 'ADI50-59', 'ADI60-69', 'ADI70-79',
    #              'ADI80-89', 'ADI90-100']
    df['ADI1-19'] = (df['ADI1-9'] + df['ADI10-19'] >= 1).astype('int')
    df['ADI20-39'] = (df['ADI20-29'] + df['ADI30-39'] >= 1).astype('int')
    df['ADI40-59'] = (df['ADI40-49'] + df['ADI50-59'] >= 1).astype('int')
    df['ADI60-79'] = (df['ADI60-69'] + df['ADI70-79'] >= 1).astype('int')
    df['ADI80-100'] = (df['ADI80-89'] + df['ADI90-100'] >= 1).astype('int')

    df['75+ years'] = (df['75-<85 years'] + df['85+ years'] >= 1).astype('int')

    # df['inpatient visits 1-4'] = (df['inpatient visits 1-2'] + df['inpatient visits 3-4'] >= 1).astype('int')
    # df['outpatient visits 1-4'] = (df['outpatient visits 1-2'] + df['outpatient visits 3-4'] >= 1).astype('int')
    # df['emergency visits 1-4'] = (df['emergency visits 1-2'] + df['emergency visits 3-4'] >= 1).astype('int')
    df['inpatient visits >=3'] = (df['inpatient visits >=5'] + df['inpatient visits 3-4'] >= 1).astype('int')
    df['outpatient visits >=3'] = (df['outpatient visits >=5'] + df['outpatient visits 3-4'] >= 1).astype('int')
    df['emergency visits >=3'] = (df['emergency visits >=5'] + df['emergency visits 3-4'] >= 1).astype('int')

    df['not hospitalized'] = 1 - df['hospitalized']
    df['icu'] = ((df['ventilation'] + df['criticalcare']) >= 1).astype('int')

    df['not hospitalized'] = (((df['hospitalized'] == 0) & (df['criticalcare'] == 0)) >= 1).astype('int')
    df['hospitalized w/o icu'] = (
            ((df['hospitalized'] == 1) & (df['ventilation'] == 0) & (df['criticalcare'] == 0)) >= 1).astype('int')
    df['icu'] = ((((df['hospitalized'] == 1) & (df['ventilation'] == 1)) | (df['criticalcare'] == 1)) >= 1).astype(
        'int')

    return df


def table1_cohorts_characterization_analyse(args):
    # severity in 'hospitalized', 'ventilation', None
    in_file = args.processed_data_file
    out_file = args.out_dir + r'/table1_of_{}_table1-V4.xlsx'.format(args.dataset)

    print('Try to load:', in_file)
    df = pd.read_csv(in_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])

    print('Load done, df.shape:', df.shape)
    print('Covid Positives:', (df['covid'] == 1).sum(), (df['covid'] == 1).mean())
    print('Covid Negative:', (df['covid'] == 0).sum(), (df['covid'] == 0).mean())

    # Step 2: Load pasc meta information
    df_pasc_info = pd.read_excel('output/causal_effects_specific_withMedication_v3.xlsx', sheet_name='diagnosis')
    selected_pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    print('len(selected_pasc_list)', len(selected_pasc_list))  # 44 pasc
    print(selected_pasc_list)
    selected_organ_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'Organ Domain'].unique()
    print('len(selected_organ_list)', len(selected_organ_list))

    specific_pasc_col = [x for x in df.columns if x.startswith('flag@')]
    pasc_name = {}
    for index, row in df_pasc_info.iterrows():
        pasc_name['flag@' + row['pasc']] = row['PASC Name Simple']

    # Step 3: set Covid pos, neg, or all population
    if args.population == 'positive':
        print('Using Covid positive  cohorts')
        df = df.loc[(df['covid'] == 1), :].copy()
    elif args.population == 'negative':
        print('Using Covid negative  cohorts')
        df = df.loc[(df['covid'] == 0), :].copy()
    else:
        print('Using Both Covid Positive and Negative  cohorts')

    print('Select population:', args.population, 'df.shape:', df.shape)

    # Step 4:
    print('Do feature transformation before severity stratification, df.shape:', df.shape)
    df = pre_transform_feature(df)
    print('df.shape after pre_transform_feature:', df.shape)

    # Step 5: set sub-population
    # focusing on: all, outpatient, inpatienticu (namely inpatient in a broad sense)
    # 'all', 'outpatient', 'inpatient', 'critical', 'ventilation'   can add more later, just these 4 for brevity
    if args.severity == 'outpatient':
        print('Considering outpatient cohorts')
        df = df.loc[(df['hospitalized'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif (args.severity == 'inpatienticu') or (args.severity == 'nonoutpatient'):
        print('Considering inpatient/hospitalized including icu cohorts, namely non-outpatient')
        df = df.loc[(df['hospitalized'] == 1) | (df['criticalcare'] == 1), :].copy()
    elif args.severity == 'inpatient':
        print('Considering inpatient/hospitalized cohorts but not ICU')
        df = df.loc[(df['hospitalized'] == 1) & (df['ventilation'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif args.severity == 'icu':
        KFOLD = 1
        print('using KFOLD=1 in the ICU case due to sample size issue')
        print('Considering ICU (hospitalized ventilation or critical care) cohorts')
        df = df.loc[(((df['hospitalized'] == 1) & (df['ventilation'] == 1)) | (df['criticalcare'] == 1)), :].copy()
    elif args.severity == 'ventilation':
        print('Considering (hospitalized) ventilation cohorts')
        df = df.loc[(df['hospitalized'] == 1) & (df['ventilation'] == 1), :].copy()
    else:
        print('Considering ALL cohorts')

    print('Select sub- cohorts:', args.severity, 'df.shape:', df.shape)

    # Step 6: Basic statistics
    pasc_flag = df['pasc-flag']  # (df['pasc-count'] >= 1).astype('int')
    pasc_t2e = df.loc[
        pasc_flag == 1, 'pasc-min-t2e']  # this time 2 event can only be used for >= 1 pasc. If >= 2, how to define t2e?
    print('PASC pos:{} ({:.3%})'.format(pasc_flag.sum(), pasc_flag.mean()),
          'PASC neg:{} ({:.3%})'.format((1 - pasc_flag).sum(), (1 - pasc_flag).mean()))
    print('PASC t2e quantile:', np.quantile(pasc_t2e, [0.5, 0.25, 0.75]))

    severe_pasc_flag = df['pasc-severe-flag']
    severe_pasc_t2e = df.loc[df['pasc-severe-flag'] == 1, 'pasc-severe-min-t2e']
    print('Severe PASC pos:{} ({:.3%})'.format(severe_pasc_flag.sum(), severe_pasc_flag.sum() / pasc_flag.sum()))
    print('Severe PASC t2e quantile:', np.quantile(severe_pasc_t2e, [0.5, 0.25, 0.75]))

    moderate_pasc_flag = df['pasc-moderateonly-flag']
    moderate_pasc_t2e = df.loc[df['pasc-moderateonly-flag'] == 1, 'pasc-moderate-min-t2e']
    print('Moderate PASC pos:{} ({:.3%})'.format(moderate_pasc_flag.sum(), moderate_pasc_flag.sum() / pasc_flag.sum()))
    print('Moderate PASC t2e quantile:', np.quantile(moderate_pasc_t2e, [0.5, 0.25, 0.75]))

    # df_pos = df.loc[df['pasc-severe-flag'] == 1, :]
    # df_neg = df.loc[df['pasc-moderateonly-flag'] == 1, :]

    def _n_str(n):
        return '{:,}'.format(n)

    def _quantile_str(x):
        v = x.quantile([0.25, 0.5, 0.75]).to_list()
        return '{:.0f} ({:.0f}—{:.0f})'.format(v[1], v[0], v[2])

    def _percentage_str(x):
        n = x.sum()
        per = x.mean()
        return '{:,} ({:.1f})'.format(n, per * 100)

    def _smd(x1, x2):
        m1 = x1.mean()
        m2 = x2.mean()
        v1 = x1.var()
        v2 = x2.var()

        VAR = np.sqrt((v1 + v2) / 2)
        smd = np.divide(
            m1 - m2,
            VAR, out=np.zeros_like(m1), where=VAR != 0)
        return smd

    row_names = []
    records = []

    # N
    row_names.append('Total')
    records.append([
        _n_str(len(df)),
        _percentage_str(df['pasc-flag']),
        _percentage_str(df['pasc-severe-flag']),
        _percentage_str(df['pasc-moderateonly-flag'])
    ])

    # age
    # row_names.append('Median age (IQR) — yr')
    # records.append([
    #     _quantile_str(df['age']),
    #     np.nan,
    #     np.nan
    # ])

    row_names.append('Severity of Acute Infection — no. (%)')
    records.append([])
    col_name = ['not hospitalized', 'hospitalized w/o icu', 'icu']
    row_names.extend(col_name)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])
          ] for c in col_name])

    row_names.append('Age group — no. (%)')
    records.append([])
    age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
    age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years']
    df['75+ years'] = df['75-<85 years'] + df['85+ years']

    row_names.extend(age_col)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])
          ] for c in age_col])

    # Sex
    row_names.append('Sex — no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])
          ] for c in sex_col])

    # Race
    row_names.append('Race — no. (%)')
    records.append([])
    col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    row_names.extend(['Asian', 'Black', 'White', 'Other', 'Missing'])
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])] for c in col_names])

    # Ethnic group
    row_names.append('Ethnic group — no. (%)')
    records.append([])
    col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    row_names.extend(['Hispanic', 'Not Hispanic', 'Other/Missing'])
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])] for c in col_names])

    # ADI
    row_names.append('Median area deprivation index (IQR) — rank')
    records.append([])
    col_names = ['ADI1-19', 'ADI20-39', 'ADI40-59', 'ADI60-79', 'ADI80-100']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])] for c in col_names])

    # utilization
    row_names.append('No. of hospital visits in the past 3 yr — no. (%)')
    records.append([])
    # part 1
    col_names = ['inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits >=3',
                 'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits >=3',
                 'emergency visits 0', 'emergency visits 1-2', 'emergency visits >=3']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])] for c in col_names])

    # BMI
    row_names.append('Body Mass Index')
    records.append([])
    col_names = ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight',
                 'BMI: 25-<30 overweight ', 'BMI: >=30 obese ', 'BMI: missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])] for c in col_names])

    # Smoking:
    row_names.append('Smoking')
    records.append([])
    col_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])] for c in col_names])

    # time of index period
    row_names.append('Index periods of patients — no. (%)')
    records.append([])

    # part 1
    col_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21', '07/21-11/21']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])] for c in col_names])

    # # part 2
    # col_names = ['YM: March 2020',
    #              'YM: April 2020', 'YM: May 2020', 'YM: June 2020', 'YM: July 2020',
    #              'YM: August 2020', 'YM: September 2020', 'YM: October 2020',
    #              'YM: November 2020', 'YM: December 2020', 'YM: January 2021',
    #              'YM: February 2021', 'YM: March 2021', 'YM: April 2021', 'YM: May 2021',
    #              'YM: June 2021', 'YM: July 2021', 'YM: August 2021',
    #              'YM: September 2021', 'YM: October 2021', 'YM: November 2021',
    #              'YM: December 2021', 'YM: January 2022', ]
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    # df = pd.DataFrame(records, columns=['Covid+', 'Covid-', 'SMD'], index=row_names)

    # Coexisting coditions
    row_names.append('Pre-existing conditions — no. (%)')
    records.append([])
    col_names = ['num_Comorbidity=0', 'num_Comorbidity=1', 'num_Comorbidity=2', 'num_Comorbidity=3',
                 'num_Comorbidity=4', 'num_Comorbidity>=5',
                 "DX: Alcohol Abuse", "DX: Anemia", "DX: Arrythmia", "DX: Asthma", "DX: Cancer",
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
                 'DX: Autism', 'DX: Sickle Cell',
                 'DX: Obstructive sleep apnea',  # added 2022-05-25
                 'DX: Epstein-Barr and Infectious Mononucleosis (Mono)',  # added 2022-05-25
                 'DX: Herpes Zoster',  # added 2022-05-25
                 "MEDICATION: Corticosteroids", "MEDICATION: Immunosuppressant drug",
                 'Fully vaccinated - Pre-index', 'Partially vaccinated - Pre-index', 'No evidence - Pre-index',
                 ]
    col_names_out = [
        'No comorbidity', '1 comorbidity', '2 comorbidities', '3 comorbidities',
        '4 comorbidities', '>=5 comorbidities',
        "Alcohol Abuse", "Anemia", "Arrythmia", "Asthma", "Cancer",
        "Chronic Kidney Disease", "Chronic Pulmonary Disorders", "Cirrhosis",
        "Coagulopathy", "Congestive Heart Failure",
        "COPD", "Coronary Artery Disease", "Dementia", "Diabetes Type 1",
        "Diabetes Type 2", "End Stage Renal Disease on Dialysis", "Hemiplegia",
        "HIV", "Hypertension", "Hypertension and Type 1 or 2 Diabetes Diagnosis",
        "Inflammatory Bowel Disorder", "Lupus or Systemic Lupus Erythematosus",
        "Mental Health Disorders", "Multiple Sclerosis", "Parkinson's Disease",
        "Peripheral vascular disorders ", "Pregnant",
        "Pulmonary Circulation Disorder",
        "Rheumatoid Arthritis", "Seizure/Epilepsy",
        "Severe Obesity  (BMI>=40 kg/m2)", "Weight Loss",
        "Down's Syndrome", 'Other Substance Abuse', 'Cystic Fibrosis',
        'Autism', 'Sickle Cell',
        'Obstructive sleep apnea',  # added 2022-05-25
        'Epstein-Barr and Infectious Mononucleosis (Mono)',  # added 2022-05-25
        'Herpes Zoster',  # added 2022-05-25
        "Prescription of Corticosteroids", "Prescription of Immunosuppressant drug",
        'Fully vaccinated - Pre-index', 'Partially vaccinated - Pre-index', 'No evidence - Pre-index',
    ]
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df[c] == 1, 'pasc-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-severe-flag']),
          _percentage_str(df.loc[df[c] == 1, 'pasc-moderateonly-flag'])] for c in col_names])

    df_out = pd.DataFrame(records, columns=['COVID Positive', 'Any PASC', 'Predictable PASC', 'Unpredictable PASC'],
                      index=row_names)
    # df['SMD'] = df['SMD'].astype(float)
    df_out.to_excel(out_file)
    print('Dump done ', df)
    return df

def table1_cohorts_characterization_analyse_revised(args):
    """
    2023-6-13, change from % within stratum to % within column
    :param args:
    :return:
    """
    # severity in 'hospitalized', 'ventilation', None
    in_file = args.processed_data_file
    out_file = args.out_dir + r'/table1_of_{}_table1-V5-revised-percent-per-column.xlsx'.format(args.dataset)

    print('Try to load:', in_file)
    df = pd.read_csv(in_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])

    print('Load done, df.shape:', df.shape)
    print('Covid Positives:', (df['covid'] == 1).sum(), (df['covid'] == 1).mean())
    print('Covid Negative:', (df['covid'] == 0).sum(), (df['covid'] == 0).mean())

    # Step 2: Load pasc meta information
    df_pasc_info = pd.read_excel('output/causal_effects_specific_withMedication_v3.xlsx', sheet_name='diagnosis')
    selected_pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    print('len(selected_pasc_list)', len(selected_pasc_list))  # 44 pasc
    print(selected_pasc_list)
    selected_organ_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'Organ Domain'].unique()
    print('len(selected_organ_list)', len(selected_organ_list))

    specific_pasc_col = [x for x in df.columns if x.startswith('flag@')]
    pasc_name = {}
    for index, row in df_pasc_info.iterrows():
        pasc_name['flag@' + row['pasc']] = row['PASC Name Simple']

    # Step 3: set Covid pos, neg, or all population
    if args.population == 'positive':
        print('Using Covid positive  cohorts')
        df = df.loc[(df['covid'] == 1), :].copy()
    elif args.population == 'negative':
        print('Using Covid negative  cohorts')
        df = df.loc[(df['covid'] == 0), :].copy()
    else:
        print('Using Both Covid Positive and Negative  cohorts')

    print('Select population:', args.population, 'df.shape:', df.shape)

    # Step 4:
    print('Do feature transformation before severity stratification, df.shape:', df.shape)
    df = pre_transform_feature(df)
    print('df.shape after pre_transform_feature:', df.shape)

    # Step 5: set sub-population
    # focusing on: all, outpatient, inpatienticu (namely inpatient in a broad sense)
    # 'all', 'outpatient', 'inpatient', 'critical', 'ventilation'   can add more later, just these 4 for brevity
    if args.severity == 'outpatient':
        print('Considering outpatient cohorts')
        df = df.loc[(df['hospitalized'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif (args.severity == 'inpatienticu') or (args.severity == 'nonoutpatient'):
        print('Considering inpatient/hospitalized including icu cohorts, namely non-outpatient')
        df = df.loc[(df['hospitalized'] == 1) | (df['criticalcare'] == 1), :].copy()
    elif args.severity == 'inpatient':
        print('Considering inpatient/hospitalized cohorts but not ICU')
        df = df.loc[(df['hospitalized'] == 1) & (df['ventilation'] == 0) & (df['criticalcare'] == 0), :].copy()
    elif args.severity == 'icu':
        KFOLD = 1
        print('using KFOLD=1 in the ICU case due to sample size issue')
        print('Considering ICU (hospitalized ventilation or critical care) cohorts')
        df = df.loc[(((df['hospitalized'] == 1) & (df['ventilation'] == 1)) | (df['criticalcare'] == 1)), :].copy()
    elif args.severity == 'ventilation':
        print('Considering (hospitalized) ventilation cohorts')
        df = df.loc[(df['hospitalized'] == 1) & (df['ventilation'] == 1), :].copy()
    else:
        print('Considering ALL cohorts')

    print('Select sub- cohorts:', args.severity, 'df.shape:', df.shape)

    # Step 6: Basic statistics
    pasc_flag = df['pasc-flag']  # (df['pasc-count'] >= 1).astype('int')
    pasc_t2e = df.loc[
        pasc_flag == 1, 'pasc-min-t2e']  # this time 2 event can only be used for >= 1 pasc. If >= 2, how to define t2e?
    print('PASC pos:{} ({:.3%})'.format(pasc_flag.sum(), pasc_flag.mean()),
          'PASC neg:{} ({:.3%})'.format((1 - pasc_flag).sum(), (1 - pasc_flag).mean()))
    print('PASC t2e quantile:', np.quantile(pasc_t2e, [0.5, 0.25, 0.75]))

    severe_pasc_flag = df['pasc-severe-flag']
    severe_pasc_t2e = df.loc[df['pasc-severe-flag'] == 1, 'pasc-severe-min-t2e']
    print('Severe PASC pos:{} ({:.3%})'.format(severe_pasc_flag.sum(), severe_pasc_flag.sum() / pasc_flag.sum()))
    print('Severe PASC t2e quantile:', np.quantile(severe_pasc_t2e, [0.5, 0.25, 0.75]))

    moderate_pasc_flag = df['pasc-moderateonly-flag']
    moderate_pasc_t2e = df.loc[df['pasc-moderateonly-flag'] == 1, 'pasc-moderate-min-t2e']
    print('Moderate PASC pos:{} ({:.3%})'.format(moderate_pasc_flag.sum(), moderate_pasc_flag.sum() / pasc_flag.sum()))
    print('Moderate PASC t2e quantile:', np.quantile(moderate_pasc_t2e, [0.5, 0.25, 0.75]))

    # df_pos = df.loc[df['pasc-severe-flag'] == 1, :]
    # df_neg = df.loc[df['pasc-moderateonly-flag'] == 1, :]

    def _n_str(n):
        return '{:,}'.format(n)

    def _quantile_str(x):
        v = x.quantile([0.25, 0.5, 0.75]).to_list()
        return '{:.0f} ({:.0f}—{:.0f})'.format(v[1], v[0], v[2])

    def _percentage_str(x):
        n = x.sum()
        per = x.mean()
        return '{:,} ({:.1f})'.format(n, per * 100)

    def _smd(x1, x2):
        m1 = x1.mean()
        m2 = x2.mean()
        v1 = x1.var()
        v2 = x2.var()

        VAR = np.sqrt((v1 + v2) / 2)
        smd = np.divide(
            m1 - m2,
            VAR, out=np.zeros_like(m1), where=VAR != 0)
        return smd

    row_names = []
    records = []

    # N
    row_names.append('Total')
    records.append([
        _n_str(len(df)),
        _percentage_str(df['pasc-flag']),
        _percentage_str(df['pasc-severe-flag']),
        _percentage_str(df['pasc-moderateonly-flag'])
    ])

    # age
    # row_names.append('Median age (IQR) — yr')
    # records.append([
    #     _quantile_str(df['age']),
    #     np.nan,
    #     np.nan
    # ])

    row_names.append('Severity of Acute Infection — no. (%)')
    records.append([])
    col_name = ['not hospitalized', 'hospitalized w/o icu', 'icu']
    row_names.extend(col_name)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])
          ] for c in col_name])

    row_names.append('Age group — no. (%)')
    records.append([])
    age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
    age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years']
    df['75+ years'] = df['75-<85 years'] + df['85+ years']

    row_names.extend(age_col)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])
          ] for c in age_col])

    # Sex
    row_names.append('Sex — no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])
          ] for c in sex_col])

    # Race
    row_names.append('Race — no. (%)')
    records.append([])
    col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    row_names.extend(['Asian', 'Black', 'White', 'Other', 'Missing'])
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])] for c in col_names])

    # Ethnic group
    row_names.append('Ethnic group — no. (%)')
    records.append([])
    col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    row_names.extend(['Hispanic', 'Not Hispanic', 'Other/Missing'])
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])] for c in col_names])

    # ADI
    row_names.append('Median area deprivation index (IQR) — rank')
    records.append([])
    col_names = ['ADI1-19', 'ADI20-39', 'ADI40-59', 'ADI60-79', 'ADI80-100']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])] for c in col_names])

    # utilization
    row_names.append('No. of hospital visits in the past 3 yr — no. (%)')
    records.append([])
    # part 1
    col_names = ['inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits >=3',
                 'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits >=3',
                 'emergency visits 0', 'emergency visits 1-2', 'emergency visits >=3']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])] for c in col_names])

    # BMI
    row_names.append('Body Mass Index')
    records.append([])
    col_names = ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight',
                 'BMI: 25-<30 overweight ', 'BMI: >=30 obese ', 'BMI: missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])] for c in col_names])

    # Smoking:
    row_names.append('Smoking')
    records.append([])
    col_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])] for c in col_names])

    # time of index period
    row_names.append('Index periods of patients — no. (%)')
    records.append([])

    # part 1
    col_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21', '07/21-11/21']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])] for c in col_names])

    # # part 2
    # col_names = ['YM: March 2020',
    #              'YM: April 2020', 'YM: May 2020', 'YM: June 2020', 'YM: July 2020',
    #              'YM: August 2020', 'YM: September 2020', 'YM: October 2020',
    #              'YM: November 2020', 'YM: December 2020', 'YM: January 2021',
    #              'YM: February 2021', 'YM: March 2021', 'YM: April 2021', 'YM: May 2021',
    #              'YM: June 2021', 'YM: July 2021', 'YM: August 2021',
    #              'YM: September 2021', 'YM: October 2021', 'YM: November 2021',
    #              'YM: December 2021', 'YM: January 2022', ]
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    # df = pd.DataFrame(records, columns=['Covid+', 'Covid-', 'SMD'], index=row_names)

    # Coexisting coditions
    row_names.append('Pre-existing conditions — no. (%)')
    records.append([])
    col_names = ['num_Comorbidity=0', 'num_Comorbidity=1', 'num_Comorbidity=2', 'num_Comorbidity=3',
                 'num_Comorbidity=4', 'num_Comorbidity>=5',
                 "DX: Alcohol Abuse", "DX: Anemia", "DX: Arrythmia", "DX: Asthma", "DX: Cancer",
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
                 'DX: Autism', 'DX: Sickle Cell',
                 'DX: Obstructive sleep apnea',  # added 2022-05-25
                 'DX: Epstein-Barr and Infectious Mononucleosis (Mono)',  # added 2022-05-25
                 'DX: Herpes Zoster',  # added 2022-05-25
                 "MEDICATION: Corticosteroids", "MEDICATION: Immunosuppressant drug",
                 'Fully vaccinated - Pre-index', 'Partially vaccinated - Pre-index', 'No evidence - Pre-index',
                 ]
    col_names_out = [
        'No comorbidity', '1 comorbidity', '2 comorbidities', '3 comorbidities',
        '4 comorbidities', '>=5 comorbidities',
        "Alcohol Abuse", "Anemia", "Arrythmia", "Asthma", "Cancer",
        "Chronic Kidney Disease", "Chronic Pulmonary Disorders", "Cirrhosis",
        "Coagulopathy", "Congestive Heart Failure",
        "COPD", "Coronary Artery Disease", "Dementia", "Diabetes Type 1",
        "Diabetes Type 2", "End Stage Renal Disease on Dialysis", "Hemiplegia",
        "HIV", "Hypertension", "Hypertension and Type 1 or 2 Diabetes Diagnosis",
        "Inflammatory Bowel Disorder", "Lupus or Systemic Lupus Erythematosus",
        "Mental Health Disorders", "Multiple Sclerosis", "Parkinson's Disease",
        "Peripheral vascular disorders ", "Pregnant",
        "Pulmonary Circulation Disorder",
        "Rheumatoid Arthritis", "Seizure/Epilepsy",
        "Severe Obesity  (BMI>=40 kg/m2)", "Weight Loss",
        "Down's Syndrome", 'Other Substance Abuse', 'Cystic Fibrosis',
        'Autism', 'Sickle Cell',
        'Obstructive sleep apnea',  # added 2022-05-25
        'Epstein-Barr and Infectious Mononucleosis (Mono)',  # added 2022-05-25
        'Herpes Zoster',  # added 2022-05-25
        "Prescription of Corticosteroids", "Prescription of Immunosuppressant drug",
        'Fully vaccinated - Pre-index', 'Partially vaccinated - Pre-index', 'No evidence - Pre-index',
    ]
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]),
          _percentage_str(df.loc[df['pasc-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-severe-flag'] == 1, c]),
          _percentage_str(df.loc[df['pasc-moderateonly-flag'] == 1, c])] for c in col_names])

    df_out = pd.DataFrame(records, columns=['COVID Positive', 'Any PASC', 'Predictable PASC', 'Unpredictable PASC'],
                      index=row_names)
    # df['SMD'] = df['SMD'].astype(float)
    df_out.to_excel(out_file)
    print('Dump done ', df)
    return df


def generate_table_2_hr(infile, outfile):
    print('In generate_table_2_hr', infile, outfile)

    def _hr_ci_str(hr, l, u):
        return '{:.2f} ({:.2f}-{:.2f})'.format(hr, l, u)

    def _p_log(p):
        logp = -np.log10(p)
        return '{:.1f}'.format(logp)

    df = pd.read_csv(infile)
    df = df.sort_values(by=['Unnamed: 0'], ascending=True)

    df['Univariate HR'] = np.nan
    df['Univariate HR, -log10 P-Value'] = np.nan
    df['Fully adjusted HR'] = np.nan
    df['Fully adjusted HR, -log10 P-Value'] = np.nan
    df['Age, sex and severity adjusted HR'] = np.nan
    df['Age, sex and severity adjusted HR, -log10 P-Value'] = np.nan

    for key, row in df.iterrows():
        print(key)
        df.loc[key, 'Univariate HR'] = _hr_ci_str(row['uni-HR'],
                                                  row['uni-CI-95% lower-bound'],
                                                  row['uni-CI-95% upper-bound'])

        df.loc[key, 'Univariate HR, -log10 P-Value'] = _p_log(row['uni-p-Value'])

        df.loc[key, 'Fully adjusted HR'] = _hr_ci_str(row['HR'],
                                                      row['CI-95% lower-bound'],
                                                      row['CI-95% upper-bound'])

        df.loc[key, 'Fully adjusted HR, -log10 P-Value'] = _p_log(row['p-Value'])

        df.loc[key, 'Age, sex and severity adjusted HR'] = _hr_ci_str(row['ageAcuteSex-HR'],
                                                                      row['ageAcuteSex-CI-95% lower-bound'],
                                                                      row['ageAcuteSex-CI-95% upper-bound'])

        df.loc[key, 'Age, sex and severity adjusted HR, -log10 P-Value'] = _p_log(row['ageAcuteSex-p-Value'])

    df.to_excel(outfile)

    return df


if __name__ == '__main__':
    # python pre_data_manuscript_table1.py --dataset ALL --cohorts covid_4manuNegNoCovid 2>&1 | tee  log/pre_data_manuscript_table1_covid_4manuNegNoCovid.txt
    start_time = time.time()
    args = parse_args()
    df_table1 = table1_cohorts_characterization_analyse(args)
    df_table2 = table1_cohorts_characterization_analyse_revised(args)

    # df = generate_table_2_hr(
    #     'output/factors/INSIGHT/elix/any_pasc/any-at-least-1-pasc-riskFactor-INSIGHT-positive-all.csv',
    #     'output/factors/INSIGHT/elix/Table2-any-at-least-1-pasc-riskFactor-INSIGHT-positive-all.xlsx',
    # )
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
