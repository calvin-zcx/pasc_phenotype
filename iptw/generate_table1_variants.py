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
# import eligibility_setting as ecs
import functools
import fnmatch
from lifelines import KaplanMeierFitter, CoxPHFitter

print = functools.partial(print, flush=True)

# from iptw.PSModels import ml
# from iptw.evaluation import *


def table1_cohorts_characterization_analyse_pooled(severity='all'):
    # severity in 'hospitalized', 'ventilation', None
    data_file = r'../data/V15_COVID19/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL.csv'
    data_file2 = r'../data/oneflorida/output/character/matrix_cohorts_covid_4manuNegNoCovidV2_bool_all.csv'

    print('Load POOLED data covariates file:', data_file, data_file2)
    df1 = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
    print('df1.shape:', df1.shape)
    df2 = pd.read_csv(data_file2, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
    print('df2.shape:', df2.shape)

    df1['database'] = 0
    df2['database'] = 1
    df = pd.concat([df1, df2], ignore_index=True, sort=False)
    print('df.shape:', df.shape)
    del df1, df2

    if severity == '1stwave':
        print('Considering patients in 1st wave, Mar-1-2020 to Sep.-30-2020')
        df = df.loc[(df['index date'] >= datetime.datetime(2020, 3, 1, 0, 0)) & (
                    df['index date'] < datetime.datetime(2020, 10, 1, 0, 0)), :].copy()
    elif severity == 'delta':
        print('Considering patients in Delta wave, June-1-2021 to Nov.-30-2021')
        df = df.loc[(df['index date'] >= datetime.datetime(2021, 6, 1, 0, 0)) & (
                    df['index date'] < datetime.datetime(2021, 12, 1, 0, 0)), :].copy()
    else:
        print('Considering ALL cohorts')

    print('df.shape:', df.shape)
    out_file = r'../data/V15_COVID19/output/character/outcome/table1_of_matrix_cohorts_CompareVariants-{}.xlsx'.format(severity)
    df_pos = df.loc[df["covid"] == 1, :]
    df_neg = df.loc[df["covid"] == 0, :]

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
    row_names.append('N')
    records.append([
        _n_str(len(df)),
        _n_str(len(df_pos)),
        _n_str(len(df_neg)),
        np.nan
    ])

    # age
    row_names.append('Median age (IQR) — yr')
    records.append([
        _quantile_str(df['age']),
        _quantile_str(df_pos['age']),
        _quantile_str(df_neg['age']),
        _smd(df_pos['age'], df_neg['age'])
    ])

    row_names.append('Age group — no. (%)')
    records.append([])
    age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
    age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years']
    df['75+ years'] = df['75-<85 years'] + df['85+ years']
    df_pos['75+ years'] = df_pos['75-<85 years'] + df_pos['85+ years']
    df_neg['75+ years'] = df_neg['75-<85 years'] + df_neg['85+ years']

    row_names.extend(age_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in age_col])

    # Sex
    row_names.append('Sex — no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in sex_col])

    # Race
    row_names.append('Race — no. (%)')
    records.append([])
    col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    # Ethnic group
    row_names.append('Ethnic group — no. (%)')
    records.append([])
    col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    # ADI
    row_names.append('Median area deprivation index (IQR) — rank')
    records.append([
        _quantile_str(df['adi']),
        _quantile_str(df_pos['adi']),
        _quantile_str(df_neg['adi']),
        _smd(df_pos['adi'], df_neg['adi'])
    ])

    # follow-up
    row_names.append('Follow-up days (IQR)')
    records.append([
        _quantile_str(df['maxfollowup']),
        _quantile_str(df_pos['maxfollowup']),
        _quantile_str(df_neg['maxfollowup']),
        _smd(df_pos['maxfollowup'], df_neg['maxfollowup'])
    ])

    # utilization
    row_names.append('No. of hospital visits in the past 3 yr — no. (%)')
    records.append([])
    # part 1
    col_names = ['inpatient no.', 'outpatient no.', 'emergency visits no.', 'other visits no.']
    col_names_out = ['No. of Inpatient Visits', 'No. of Outpatient Visits',
                     'No. of Emergency Visits', 'No. of Other Visits']

    row_names.extend(col_names_out)
    records.extend(
        [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    # part2
    col_names = ['inpatient visits 0', 'inpatient visits 1-2', 'inpatient visits 3-4',
                 'inpatient visits >=5',
                 'outpatient visits 0', 'outpatient visits 1-2', 'outpatient visits 3-4',
                 'outpatient visits >=5',
                 'emergency visits 0', 'emergency visits 1-2', 'emergency visits 3-4',
                 'emergency visits >=5']
    col_names_out = ['Inpatient 0', 'Inpatient 1-2', 'Inpatient 3-4', 'Inpatient >=5',
                     'Outpatient 0', 'Outpatient 1-2', 'Outpatient 3-4', 'Outpatient >=5',
                     'Emergency 0', 'Emergency 1-2', 'Emergency 3-4', 'Emergency >=5']

    df_pos['Inpatient >=3'] = df_pos['inpatient visits 3-4'] + df_pos['inpatient visits >=5']
    df_neg['Inpatient >=3'] = df_neg['inpatient visits 3-4'] + df_neg['inpatient visits >=5']
    df_pos['Outpatient >=3'] = df_pos['outpatient visits 3-4'] + df_pos['outpatient visits >=5']
    df_neg['Outpatient >=3'] = df_neg['outpatient visits 3-4'] + df_neg['outpatient visits >=5']
    df_pos['Emergency >=3'] = df_pos['emergency visits 3-4'] + df_pos['emergency visits >=5']
    df_neg['Emergency >=3'] = df_neg['emergency visits 3-4'] + df_neg['emergency visits >=5']

    df['Inpatient >=3'] = df['inpatient visits 3-4'] + df['inpatient visits >=5']
    df['Outpatient >=3'] = df['outpatient visits 3-4'] + df['outpatient visits >=5']
    df['Emergency >=3'] = df['emergency visits 3-4'] + df['emergency visits >=5']

    col_names = ['inpatient visits 0', 'inpatient visits 1-2', 'Inpatient >=3',
                 'outpatient visits 0', 'outpatient visits 1-2', 'Outpatient >=3',
                 'emergency visits 0', 'emergency visits 1-2', 'Emergency >=3']
    col_names_out = ['Inpatient 0', 'Inpatient 1-2',  'Inpatient >=3',
                     'Outpatient 0', 'Outpatient 1-2',  'Outpatient >=3',
                     'Emergency 0', 'Emergency 1-2', 'Emergency >=3']
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    # BMI
    row_names.append('BMI (IQR)')
    records.append([
        _quantile_str(df['bmi']),
        _quantile_str(df_pos['bmi']),
        _quantile_str(df_neg['bmi']),
        _smd(df_pos['bmi'], df_neg['bmi'])
    ])

    col_names = ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight',
                 'BMI: 25-<30 overweight ', 'BMI: >=30 obese ', 'BMI: missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    # Smoking:
    col_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    # time of index period
    row_names.append('Index time period of patients — no. (%)')
    records.append([])

    # part 1
    col_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21', '03/21-06/21', '07/21-11/21']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    # part 2
    col_names = ['YM: March 2020',
                 'YM: April 2020', 'YM: May 2020', 'YM: June 2020', 'YM: July 2020',
                 'YM: August 2020', 'YM: September 2020', 'YM: October 2020',
                 'YM: November 2020', 'YM: December 2020', 'YM: January 2021',
                 'YM: February 2021', 'YM: March 2021', 'YM: April 2021', 'YM: May 2021',
                 'YM: June 2021', 'YM: July 2021', 'YM: August 2021',
                 'YM: September 2021', 'YM: October 2021', 'YM: November 2021',
                 'YM: December 2021', 'YM: January 2022', ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    # df = pd.DataFrame(records, columns=['Covid+', 'Covid-', 'SMD'], index=row_names)

    # Coexisting coditions
    row_names.append('Coexisting conditions — no. (%)')
    records.append([])
    col_names = ["DX: Alcohol Abuse", "DX: Anemia", "DX: Arrythmia", "DX: Asthma", "DX: Cancer",
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
                 "MEDICATION: Corticosteroids", "MEDICATION: Immunosuppressant drug"
                 ]
    col_names_out = ["Alcohol Abuse", "Anemia", "Arrythmia", "Asthma", "Cancer",
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
                     "Prescription of Corticosteroids", "Prescription of Immunosuppressant drug"
                     ]
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in col_names])

    df = pd.DataFrame(records, columns=['All', 'COVID Positive', 'COVID Negative', 'SMD'], index=row_names)
    df['SMD'] = df['SMD'].astype(float)
    df.to_excel(out_file)
    print('Dump done ', df)
    return df, df


if __name__ == '__main__':
    # python pre_data_manuscript_table1.py --dataset ALL --cohorts covid_4manuNegNoCovid 2>&1 | tee  log/pre_data_manuscript_table1_covid_4manuNegNoCovid.txt
    start_time = time.time()

    df_table1, df = table1_cohorts_characterization_analyse_pooled(severity='all')
    df_table1, df = table1_cohorts_characterization_analyse_pooled(severity='1stwave')
    df_table1, df = table1_cohorts_characterization_analyse_pooled(severity='delta')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


