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


def table1_cohorts_characterization_analyse(pivot='covid'):
    # severity in 'hospitalized', 'ventilation', None

    data_file2 = r'pos_preg_femalenot.csv'

    if pivot == 'covid':
        data_file = r'preg_pos_neg.csv'
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
        pcol = 'covid'
        df_pos = df.loc[df["covid"] == 1, :]
        df_neg = df.loc[df["covid"] == 0, :]
        out_file = r'preg_pos_neg_covaraite_summary.xlsx'
        output_columns = ['All', 'Pregnant COVID Positive', 'Pregnant COVID Negative', 'SMD']
    elif pivot == 'pregnancy':
        data_file = r'pos_preg_femalenot.csv'
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
        pcol = 'flag_pregnancy'
        df_pos = df.loc[df["flag_pregnancy"] == 1, :]
        df_neg = df.loc[df["flag_pregnancy"] == 0, :]
        out_file = r'pos_preg_femalenot_covaraite_summary.xlsx'
        output_columns = ['All', 'Pregnant COVID Positive', 'Non-Pregnant COVID Positive', 'SMD']

    else:
        raise ValueError

    print('Load data covariates file:', data_file, df.shape, pcol)

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
    # age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75-<85 years', '85+ years']
    # age_col = ['20-<40 years', '40-<55 years', '55-<65 years', '65-<75 years', '75+ years']

    age_col = ['pregage:18-<25 years', 'pregage:25-<30 years', 'pregage:30-<35 years',
               'pregage:35-<40 years', 'pregage:40-<45 years', 'pregage:45-50 years']

    # df['75+ years'] = df['75-<85 years'] + df['85+ years']
    # df_pos['75+ years'] = df_pos['75-<85 years'] + df_pos['85+ years']
    # df_neg['75+ years'] = df_neg['75-<85 years'] + df_neg['85+ years']

    row_names.extend(age_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in age_col])

    # Sex
    row_names.append('Sex — no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in sex_col])

    # Race
    row_names.append('Race — no. (%)')
    records.append([])
    col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Ethnic group
    row_names.append('Ethnic group — no. (%)')
    records.append([])
    col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

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
        [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in
         col_names])

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
    col_names_out = ['Inpatient 0', 'Inpatient 1-2', 'Inpatient >=3',
                     'Outpatient 0', 'Outpatient 1-2', 'Outpatient >=3',
                     'Emergency 0', 'Emergency 1-2', 'Emergency >=3']
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

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
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Smoking:
    col_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Vaccine:
    col_names = ['Fully vaccinated - Pre-index',
                 'Fully vaccinated - Post-index',
                 'Partially vaccinated - Pre-index',
                 'Partially vaccinated - Post-index',
                 'No evidence - Pre-index',
                 'No evidence - Post-index',
                 ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])


    # time of index period
    row_names.append('Index time period of patients — no. (%)')
    records.append([])

    # part 1
    col_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21',
                 '03/21-06/21', '07/21-10/21', '11/21-02/22',
                 '03/22-06/22', '07/22-10/22']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # part 2
    col_names = ['YM: March 2020',
                 'YM: April 2020', 'YM: May 2020', 'YM: June 2020', 'YM: July 2020',
                 'YM: August 2020', 'YM: September 2020', 'YM: October 2020',
                 'YM: November 2020', 'YM: December 2020', 'YM: January 2021',
                 'YM: February 2021', 'YM: March 2021', 'YM: April 2021', 'YM: May 2021',
                 'YM: June 2021', 'YM: July 2021', 'YM: August 2021',
                 'YM: September 2021', 'YM: October 2021', 'YM: November 2021',
                 'YM: December 2021', 'YM: January 2022', "YM: February 2022", "YM: March 2022", "YM: April 2022",
                 "YM: May 2022",
                 "YM: June 2022", "YM: July 2022", "YM: August 2022", "YM: September 2022",
                 "YM: October 2022", "YM: November 2022", "YM: December 2022", "YM: January 2023",
                 "YM: February 2023", ]

    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

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
                 'DX: Obstructive sleep apnea',  # added 2022-05-25
                 'DX: Epstein-Barr and Infectious Mononucleosis (Mono)',  # added 2022-05-25
                 'DX: Herpes Zoster',  # added 2022-05-25
                 "MEDICATION: Corticosteroids", "MEDICATION: Immunosuppressant drug"
                 ] + ['obc:Placenta accreta spectrum', 'obc:Pulmonary hypertension', 'obc:Chronic renal disease',
                      'obc:Cardiac disease, preexisting', 'obc:HIV/AIDS', 'obc:Preeclampsia with severe features',
                      'obc:Placental abruption', 'obc:Bleeding disorder, preexisting', 'obc:Anemia, preexisting',
                      'obc:Twin/multiple pregnancy', 'obc:Preterm birth (< 37 weeks)',
                      'obc:Placenta previa, complete or partial',
                      'obc:Neuromuscular disease', 'obc:Asthma, acute or moderate/severe',
                      'obc:Preeclampsia without severe features or gestational hypertension',
                      'obc:Connective tissue or autoimmune disease', 'obc:Uterine fibroids',
                      'obc:Substance use disorder',
                      'obc:Gastrointestinal disease', 'obc:Chronic hypertension', 'obc:Major mental health disorder',
                      'obc:Preexisting diabetes mellitus', 'obc:Thyrotoxicosis', 'obc:Previous cesarean birth',
                      'obc:Gestational diabetes mellitus', 'obc:Delivery BMI\xa0>\xa040']

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
                     'Obstructive sleep apnea',  # added 2022-05-25
                     'Epstein-Barr and Infectious Mononucleosis (Mono)',  # added 2022-05-25
                     'Herpes Zoster',  # added 2022-05-25
                     "Prescription of Corticosteroids", "Prescription of Immunosuppressant drug"
                     ] + ['Placenta accreta spectrum', 'Pulmonary hypertension', 'Chronic renal disease',
                          'Cardiac disease, preexisting', 'HIV/AIDS', 'Preeclampsia with severe features',
                          'Placental abruption', 'Bleeding disorder, preexisting', 'Anemia, preexisting',
                          'Twin/multiple pregnancy', 'Preterm birth (< 37 weeks)',
                          'Placenta previa, complete or partial',
                          'Neuromuscular disease', 'Asthma, acute or moderate/severe',
                          'Preeclampsia without severe features or gestational hypertension',
                          'Connective tissue or autoimmune disease', 'Uterine fibroids',
                          'Substance use disorder',
                          'Gastrointestinal disease', 'Chronic hypertension', 'Major mental health disorder',
                          'Preexisting diabetes mellitus', 'Thyrotoxicosis', 'Previous cesarean birth',
                          'Gestational diabetes mellitus', r'Delivery BMI>40']
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    df_out = pd.DataFrame(records, columns=output_columns, index=row_names)
    df_out['SMD'] = df_out['SMD'].astype(float)
    df_out.to_excel(out_file)
    print('Dump done ', df_out)
    return df, df_out


if __name__ == '__main__':
    # python pre_data_manuscript_table1.py --dataset ALL --cohorts covid_4manuNegNoCovid 2>&1 | tee  log/pre_data_manuscript_table1_covid_4manuNegNoCovid.txt
    start_time = time.time()

    df, df_tab = table1_cohorts_characterization_analyse(pivot='covid')
    df, df_tab = table1_cohorts_characterization_analyse(pivot='pregnancy')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))