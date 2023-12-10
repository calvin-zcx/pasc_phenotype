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

def add_col(df):
    df['Type 1 or 2 Diabetes Diagnosis'] = (
            ((df["DX: Diabetes Type 1"] >= 1).astype('int') + (df["DX: Diabetes Type 2"] >= 1).astype(
                'int')) >= 1).astype('int')

    selected_cols = [x for x in df.columns if (
            x.startswith('DX:') or
            x.startswith('MEDICATION:') or
            x.startswith('CCI:') or
            x.startswith('obc:')
    )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')
    df.loc[:, r"DX: Hypertension and Type 1 or 2 Diabetes Diagnosis"] = \
        (df.loc[:, r'DX: Hypertension'] & (
                df.loc[:, r'DX: Diabetes Type 1'] | df.loc[:, r'DX: Diabetes Type 2'])).astype('int')

    # baseline part have been binarized already
    selected_cols = [x for x in df.columns if
                     (x.startswith('dx-out@') or
                      x.startswith('dxadd-out@') or
                      x.startswith('dxbrainfog-out@') or
                      x.startswith('covidmed-out@') or
                      x.startswith('smm-out@') or
                      x.startswith('dxdxCFR-out@')
                      )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    df.loc[df['death t2e'] < 0, 'death t2e'] = 9999
    df.loc[df['death t2e'] < 0, 'death'] = 0

    df['cci_quan:0'] = 0
    df['cci_quan:1-2'] = 0
    df['cci_quan:3-4'] = 0
    df['cci_quan:5-10'] = 0
    df['cci_quan:11+'] = 0

    df['age18-24'] = 0
    df['age15-34'] = 0
    df['age35-49'] = 0
    df['age50-64'] = 0
    df['age65+'] = 0

    df['RE:Asian Non-Hispanic'] = 0
    df['RE:Black or African American Non-Hispanic'] = 0
    df['RE:Hispanic or Latino Any Race'] = 0
    df['RE:White Non-Hispanic'] = 0
    df['RE:Other Non-Hispanic'] = 0
    df['RE:Unknown'] = 0

    df['No. of Visits:0'] = 0
    df['No. of Visits:1-3'] = 0
    df['No. of Visits:4-9'] = 0
    df['No. of Visits:10-19'] = 0
    df['No. of Visits:>=20'] = 0

    df['No. of hospitalizations:0'] = 0
    df['No. of hospitalizations:1'] = 0
    df['No. of hospitalizations:>=1'] = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # 'index date', 'flag_delivery_date', 'flag_pregnancy_start_date', 'flag_pregnancy_end_date'
        index_date = row['index date']
        age = row['age']
        if pd.notna(age):
            if age < 25:
                df.loc[index, 'age18-24'] = 1
            elif age < 35:
                df.loc[index, 'age15-34'] = 1
            elif age < 50:
                df.loc[index, 'age35-49'] = 1
            elif age < 65:
                df.loc[index, 'age50-64'] = 1
            elif age >= 65:
                df.loc[index, 'age65+'] = 1

        if row['score_cci_quan'] <= 0:
            df.loc[index, 'cci_quan:0'] = 1
        elif row['score_cci_quan'] <= 2:
            df.loc[index, 'cci_quan:1-2'] = 1
        elif row['score_cci_quan'] <= 4:
            df.loc[index, 'cci_quan:3-4'] = 1
        elif row['score_cci_quan'] <= 10:
            df.loc[index, 'cci_quan:5-10'] = 1
        elif row['score_cci_quan'] >= 11:
            df.loc[index, 'cci_quan:11+'] = 1

        if row['Asian'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:Asian Non-Hispanic'] = 1
        elif row['Black or African American'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:Black or African American Non-Hispanic'] = 1
        elif row['Hispanic: Yes']:
            df.loc[index, 'RE:Hispanic or Latino Any Race'] = 1
        elif row['White'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:White Non-Hispanic'] = 1
        elif row['Other'] and ((row['Hispanic: Yes'] == 0) or (row['Hispanic: No'] == 1)):
            df.loc[index, 'RE:Other Non-Hispanic'] = 1
        else:
            df.loc[index, 'RE:Unknown'] = 1

        visits = row['inpatient no.'] + row['outpatient no.'] + row['emergency visits no.'] + row['other visits no.']
        if visits == 0:
            df.loc[index, 'No. of Visits:0'] = 1
        elif visits <= 3:
            df.loc[index, 'No. of Visits:1-3'] = 1
        elif visits <= 9:
            df.loc[index, 'No. of Visits:4-9'] = 1
        elif visits <= 19:
            df.loc[index, 'No. of Visits:10-19'] = 1
        else:
            df.loc[index, 'No. of Visits:>=20'] = 1

        if row['inpatient no.'] == 0:
            df.loc[index, 'No. of hospitalizations:0'] = 1
        elif row['inpatient no.'] == 1:
            df.loc[index, 'No. of hospitalizations:1'] = 1
        else:
            df.loc[index, 'No. of hospitalizations:>=1'] = 1

        # any PASC

        # adi use mine later, not transform here, add missing

        # monthly changes, add later. Already there

    return df


def add_any_pasc(df):
    df_pasc_info = pd.read_excel(r'../prediction/output/causal_effects_specific_withMedication_v3.xlsx',
                                 sheet_name='diagnosis')
    addedPASC_encoding = utils.load(r'../data/mapping/addedPASC_index_mapping.pkl')
    addedPASC_list = list(addedPASC_encoding.keys())
    brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
    brainfog_list = list(brainfog_encoding.keys())

    pasc_simname = {}
    pasc_organ = {}
    for index, rows in df_pasc_info.iterrows():
        pasc_simname[rows['pasc']] = (rows['PASC Name Simple'], rows['Organ Domain'])
        pasc_organ[rows['pasc']] = rows['Organ Domain']

    for p in addedPASC_list:
        pasc_simname[p] = (p, 'General-add')
        pasc_organ[p] = 'General-add'

    for p in brainfog_list:
        pasc_simname[p] = (p, 'brainfog')
        pasc_organ[p] = 'brainfog'

    # pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
    pasc_list_raw = df_pasc_info.loc[df_pasc_info['selected_narrow'] == 1, 'pasc'].to_list()
    _exclude_list = ['Pressure ulcer of skin', ]
    pasc_list = [x for x in pasc_list_raw if x not in _exclude_list]

    pasc_add = ['smell and taste', ]
    print('len(pasc_list)', len(pasc_list), 'len(pasc_add)', len(pasc_add))

    for p in pasc_list:
        df[p + '_pasc_flag'] = 0
    for p in pasc_add:
        df[p + '_pasc_flag'] = 0

    df['any_pasc_flag'] = 0
    df['any_pasc_type'] = np.nan
    df['any_pasc_t2e'] = 180  # np.nan
    df['any_pasc_txt'] = ''
    df['any_pasc_baseline'] = 0  # placeholder for screening, no special meaning, null column
    for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
        # for any 1 pasc
        t2e_list = []
        pasc_1_list = []
        pasc_1_name = []
        pasc_1_text = ''
        for p in pasc_list:
            if (rows['dx-out@' + p] > 0) and (rows['dx-base@' + p] == 0):
                t2e_list.append(rows['dx-t2e@' + p])
                pasc_1_list.append(p)
                pasc_1_name.append(pasc_simname[p])
                pasc_1_text += (pasc_simname[p][0] + ';')

                df.loc[index, p + '_pasc_flag'] = 1

        for p in pasc_add:
            if (rows['dxadd-out@' + p] > 0) and (rows['dxadd-base@' + p] == 0):
                t2e_list.append(rows['dxadd-t2e@' + p])
                pasc_1_list.append(p)
                pasc_1_name.append(pasc_simname[p])
                pasc_1_text += (pasc_simname[p][0] + ';')

                df.loc[index, p + '_pasc_flag'] = 1

        if len(t2e_list) > 0:
            df.loc[index, 'any_pasc_flag'] = 1
            df.loc[index, 'any_pasc_t2e'] = np.min(t2e_list)
            df.loc[index, 'any_pasc_txt'] = pasc_1_text
        else:
            df.loc[index, 'any_pasc_flag'] = 0
            df.loc[index, 'any_pasc_t2e'] = rows[['dx-t2e@' + p for p in pasc_list]].max()  # censoring time

    return df

def table1_cohorts_characterization_analyse():
    # %% Step 1. Load  Data
    df1 = pd.read_csv(r'../iptw/recover29Nov27_covid_pos-ECselectedTreated.csv',
                      dtype={'patid': str, 'site': str, 'zip': str},
                      parse_dates=['index date', 'dob'])
    df2 = pd.read_csv('../iptw/recover29Nov27_covid_pos-ECselectedControl.csv',
                      dtype={'patid': str, 'site': str, 'zip': str},
                      parse_dates=['index date', 'dob'])
    print('treated df1.shape', df1.shape,
          'control df2.shape', df2.shape,
          )

    df_pos = add_col(df1)
    df_neg = add_col(df2)
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    df = add_any_pasc(df)

    print('treated df_pos.shape', df_pos.shape,
          'control df_neg.shape', df_neg.shape,
          'combined df.shape', df.shape, )

    out_file = r'Paxlovid_4N3C_summary_PCORnet29Dec9.xlsx'
    output_columns = ['All', 'COVID Positive Paxlovid', 'COVID Positive w/o Paxlovid', 'SMD']

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

    row_names.append('PASC — no. (%)')
    records.append([])
    col_names = ['any_pasc_flag',]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Sex
    row_names.append('Sex — no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    # sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in sex_col])

    row_names.append('Acute severity — no. (%)')
    records.append([])
    col_names = ['outpatient', 'inpatient', 'icu', 'inpatienticu']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

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

    age_col = ['age18-24', 'age15-34', 'age35-49', 'age50-64' 'age65+']
    row_names.extend(age_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in age_col])

    # Race
    row_names.append('Race — no. (%)')
    records.append([])
    # col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    col_names = ['RE:Asian Non-Hispanic', 'RE:Black or African American Non-Hispanic', 'RE:Hispanic or Latino Any Race',
                 'RE:White Non-Hispanic', 'RE:Other Non-Hispanic', 'RE:Unknown']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # # Ethnic group
    # row_names.append('Ethnic group — no. (%)')
    # records.append([])
    # col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # # follow-up
    # row_names.append('Follow-up days (IQR)')
    # records.append([
    #     _quantile_str(df['maxfollowup']),
    #     _quantile_str(df_pos['maxfollowup']),
    #     _quantile_str(df_neg['maxfollowup']),
    #     _smd(df_pos['maxfollowup'], df_neg['maxfollowup'])
    # ])

    # utilization
    row_names.append('No. of hospital visits in the past 3 yr — no. (%)')
    records.append([])
    # part 1
    col_names = ['No. of Visits:0', 'No. of Visits:1-3', 'No. of Visits:4-9', 'No. of Visits:10-19',
                 'No. of Visits:>=20',
                 'No. of hospitalizations:0', 'No. of hospitalizations:1', 'No. of hospitalizations:>=1']
    col_names_out = col_names
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in
         col_names])

    # ADI
    row_names.append('Median area deprivation index (IQR) — rank')
    records.append([
        _quantile_str(df['adi']),
        _quantile_str(df_pos['adi']),
        _quantile_str(df_neg['adi']),
        _smd(df_pos['adi'], df_neg['adi'])
    ])

    col_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
                 'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100',
                 'ADIMissing']
    row_names.extend(col_names)
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

    # # Smoking:
    # col_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])
    #
    # # Vaccine:
    # col_names = ['Fully vaccinated - Pre-index',
    #              'Fully vaccinated - Post-index',
    #              'Partially vaccinated - Pre-index',
    #              'Partially vaccinated - Post-index',
    #              'No evidence - Pre-index',
    #              'No evidence - Post-index',
    #              ]
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # time of index period
    row_names.append('Index time period of patients — no. (%)')
    records.append([])

    # part 1
    col_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21',
                 '03/21-06/21', '07/21-10/21', '11/21-02/22',
                 '03/22-06/22', '07/22-10/22', '11/22-02/23',
                 '03/23-06/23', '07/23-10/23', '11/23-02/24', ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # part 2
    col_names = ["YM: January 2022", "YM: February 2022", "YM: March 2022", "YM: April 2022", "YM: May 2022",
                 "YM: June 2022", "YM: July 2022", "YM: August 2022", "YM: September 2022",
                 "YM: October 2022", "YM: November 2022", "YM: December 2022",
                 "YM: January 2023", "YM: February 2023", ]

    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # df = pd.DataFrame(records, columns=['Covid+', 'Covid-', 'SMD'], index=row_names)

    # # Coexisting coditions
    # row_names.append('Coexisting conditions — no. (%)')
    #
    # records.append([])
    # col_names = ["DX: Alcohol Abuse", "DX: Anemia", "DX: Arrythmia", "DX: Asthma", "DX: Cancer",
    #              "DX: Chronic Kidney Disease", "DX: Chronic Pulmonary Disorders", "DX: Cirrhosis",
    #              "DX: Coagulopathy", "DX: Congestive Heart Failure",
    #              "DX: COPD", "DX: Coronary Artery Disease", "DX: Dementia", "DX: Diabetes Type 1",
    #              "DX: Diabetes Type 2", 'Type 1 or 2 Diabetes Diagnosis',
    #              "DX: End Stage Renal Disease on Dialysis", "DX: Hemiplegia",
    #              "DX: HIV", "DX: Hypertension", "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis",
    #              "DX: Inflammatory Bowel Disorder", "DX: Lupus or Systemic Lupus Erythematosus",
    #              "DX: Mental Health Disorders", "DX: Multiple Sclerosis", "DX: Parkinson's Disease",
    #              "DX: Peripheral vascular disorders ", "DX: Pregnant",
    #              "DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)",
    #              "DX: Rheumatoid Arthritis", "DX: Seizure/Epilepsy",
    #              "DX: Severe Obesity  (BMI>=40 kg/m2)", "DX: Weight Loss",
    #              "DX: Down's Syndrome", 'DX: Other Substance Abuse', 'DX: Cystic Fibrosis',
    #              'DX: Autism', 'DX: Sickle Cell',
    #              'DX: Obstructive sleep apnea',  # added 2022-05-25
    #              'DX: Epstein-Barr and Infectious Mononucleosis (Mono)',  # added 2022-05-25
    #              'DX: Herpes Zoster',  # added 2022-05-25
    #              "MEDICATION: Corticosteroids", "MEDICATION: Immunosuppressant drug"
    #              ] + ['obc:Placenta accreta spectrum', 'obc:Pulmonary hypertension', 'obc:Chronic renal disease',
    #                   'obc:Cardiac disease, preexisting', 'obc:HIV/AIDS', 'obc:Preeclampsia with severe features',
    #                   'obc:Placental abruption', 'obc:Bleeding disorder, preexisting', 'obc:Anemia, preexisting',
    #                   'obc:Twin/multiple pregnancy', 'obc:Preterm birth (< 37 weeks)',
    #                   'obc:Placenta previa, complete or partial',
    #                   'obc:Neuromuscular disease', 'obc:Asthma, acute or moderate/severe',
    #                   'obc:Preeclampsia without severe features or gestational hypertension',
    #                   'obc:Connective tissue or autoimmune disease', 'obc:Uterine fibroids',
    #                   'obc:Substance use disorder',
    #                   'obc:Gastrointestinal disease', 'obc:Chronic hypertension', 'obc:Major mental health disorder',
    #                   'obc:Preexisting diabetes mellitus', 'obc:Thyrotoxicosis', 'obc:Previous cesarean birth',
    #                   'obc:Gestational diabetes mellitus', 'obc:Delivery BMI\xa0>\xa040'] + [
    #                 'CCI:Myocardial Infarction', 'CCI:Congestive Heart Failure', 'CCI:Periphral Vascular Disease',
    #                 'CCI:Cerebrovascular Disease', 'CCI:Dementia', 'CCI:Chronic Pulmonary Disease',
    #                 'CCI:Connective Tissue Disease-Rheumatic Disease', 'CCI:Peptic Ulcer Disease',
    #                 'CCI:Mild Liver Disease',
    #                 'CCI:Diabetes without complications', 'CCI:Diabetes with complications',
    #                 'CCI:Paraplegia and Hemiplegia',
    #                 'CCI:Renal Disease', 'CCI:Cancer', 'CCI:Moderate or Severe Liver Disease',
    #                 'CCI:Metastatic Carcinoma',
    #                 'CCI:AIDS/HIV',
    #             ] + [
    #                 "autoimmune/immune suppression",
    #                 "Severe Obesity",
    #             ]
    #
    # col_names_out = ["Alcohol Abuse", "Anemia", "Arrythmia", "Asthma", "Cancer",
    #                  "Chronic Kidney Disease", "Chronic Pulmonary Disorders", "Cirrhosis",
    #                  "Coagulopathy", "Congestive Heart Failure",
    #                  "COPD", "Coronary Artery Disease", "Dementia", "Diabetes Type 1",
    #                  "Diabetes Type 2", 'Type 1 or 2 Diabetes Diagnosis',
    #                  "End Stage Renal Disease on Dialysis", "Hemiplegia",
    #                  "HIV", "Hypertension", "Hypertension and Type 1 or 2 Diabetes Diagnosis",
    #                  "Inflammatory Bowel Disorder", "Lupus or Systemic Lupus Erythematosus",
    #                  "Mental Health Disorders", "Multiple Sclerosis", "Parkinson's Disease",
    #                  "Peripheral vascular disorders ", "Pregnant",
    #                  "Pulmonary Circulation Disorder",
    #                  "Rheumatoid Arthritis", "Seizure/Epilepsy",
    #                  "Severe Obesity  (BMI>=40 kg/m2)", "Weight Loss",
    #                  "Down's Syndrome", 'Other Substance Abuse', 'Cystic Fibrosis',
    #                  'Autism', 'Sickle Cell',
    #                  'Obstructive sleep apnea',  # added 2022-05-25
    #                  'Epstein-Barr and Infectious Mononucleosis (Mono)',  # added 2022-05-25
    #                  'Herpes Zoster',  # added 2022-05-25
    #                  "Prescription of Corticosteroids", "Prescription of Immunosuppressant drug"
    #                  ] + ['Placenta accreta spectrum', 'Pulmonary hypertension', 'Chronic renal disease',
    #                       'Cardiac disease, preexisting', 'HIV/AIDS', 'Preeclampsia with severe features',
    #                       'Placental abruption', 'Bleeding disorder, preexisting', 'Anemia, preexisting',
    #                       'Twin/multiple pregnancy', 'Preterm birth (< 37 weeks)',
    #                       'Placenta previa, complete or partial',
    #                       'Neuromuscular disease', 'Asthma, acute or moderate/severe',
    #                       'Preeclampsia without severe features or gestational hypertension',
    #                       'Connective tissue or autoimmune disease', 'Uterine fibroids',
    #                       'Substance use disorder',
    #                       'Gastrointestinal disease', 'Chronic hypertension', 'Major mental health disorder',
    #                       'Preexisting diabetes mellitus', 'Thyrotoxicosis', 'Previous cesarean birth',
    #                       'Gestational diabetes mellitus', r'Delivery BMI>40'] + [
    #                     'CCI:Myocardial Infarction', 'CCI:Congestive Heart Failure', 'CCI:Periphral Vascular Disease',
    #                     'CCI:Cerebrovascular Disease', 'CCI:Dementia', 'CCI:Chronic Pulmonary Disease',
    #                     'CCI:Connective Tissue Disease-Rheumatic Disease', 'CCI:Peptic Ulcer Disease',
    #                     'CCI:Mild Liver Disease',
    #                     'CCI:Diabetes without complications', 'CCI:Diabetes with complications',
    #                     'CCI:Paraplegia and Hemiplegia',
    #                     'CCI:Renal Disease', 'CCI:Cancer', 'CCI:Moderate or Severe Liver Disease',
    #                     'CCI:Metastatic Carcinoma',
    #                     'CCI:AIDS/HIV',
    #                 ] + [
    #                     "autoimmune/immune suppression",
    #                     "Severe Obesity",
    #                 ]
    #
    # row_names.extend(col_names_out)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])
    #
    # col_names = ['score_cci_charlson', 'score_cci_quan']
    # col_names_out = ['score_cci_charlson', 'score_cci_quan']
    # row_names.extend(col_names_out)
    # records.extend(
    #     [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])
    row_names.append('CCI Score — no. (%)')
    records.append([])

    col_names = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    col_names_out = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
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

    # %% Step 1. Load  Data
    df1 = pd.read_csv(r'../iptw/recover29Nov27_covid_pos-ECselectedTreated.csv',
                      dtype={'patid': str, 'site': str, 'zip': str},
                      parse_dates=['index date', 'dob'])
    df2 = pd.read_csv('../iptw/recover29Nov27_covid_pos-ECselectedControl.csv',
                      dtype={'patid': str, 'site': str, 'zip': str},
                      parse_dates=['index date', 'dob'])
    print('treated df1.shape', df1.shape,
          'control df2.shape', df2.shape,
          )

    df_pos = add_col(df1)
    df_neg = add_col(df2)

    df_pos = add_any_pasc(df_pos)
    df_neg = add_any_pasc(df_neg)

    df = pd.concat([df_pos, df_neg], ignore_index=True)

    print('treated df_pos.shape', df_pos.shape,
          'control df_neg.shape', df_neg.shape,
          'combined df.shape', df.shape, )

    out_file = r'Paxlovid_4N3C_summary_PCORnet29Dec9.xlsx'
    output_columns = ['All', 'COVID Positive Paxlovid', 'COVID Positive w/o Paxlovid', 'SMD']


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

    row_names.append('PASC — no. (%)')
    records.append([])
    col_names = ['any_pasc_flag', ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Sex
    row_names.append('Sex — no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    # sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in sex_col])

    row_names.append('Acute severity — no. (%)')
    records.append([])
    col_names = ['outpatient', 'inpatient', 'icu', 'inpatienticu']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

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

    age_col = ['age18-24', 'age15-34', 'age35-49', 'age50-64', 'age65+']
    row_names.extend(age_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in age_col])

    # Race
    row_names.append('Race and Ethnicity — no. (%)')
    records.append([])
    # col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
    col_names = ['RE:Asian Non-Hispanic', 'RE:Black or African American Non-Hispanic', 'RE:Hispanic or Latino Any Race',
                 'RE:White Non-Hispanic', 'RE:Other Non-Hispanic', 'RE:Unknown']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # # Ethnic group
    # row_names.append('Ethnic group — no. (%)')
    # records.append([])
    # col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # # follow-up
    # row_names.append('Follow-up days (IQR)')
    # records.append([
    #     _quantile_str(df['maxfollowup']),
    #     _quantile_str(df_pos['maxfollowup']),
    #     _quantile_str(df_neg['maxfollowup']),
    #     _smd(df_pos['maxfollowup'], df_neg['maxfollowup'])
    # ])

    col_names = ['score_cci_charlson', 'score_cci_quan']
    col_names_out = ['score_cci_charlson', 'score_cci_quan']
    row_names.extend(col_names_out)
    records.extend(
        [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])
    row_names.append('CCI Score — no. (%)')
    records.append([])

    col_names = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    col_names_out = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # utilization
    row_names.append('No. of hospital visits in the past 3 yr — no. (%)')
    records.append([])
    # part 1
    col_names = ['No. of Visits:0', 'No. of Visits:1-3', 'No. of Visits:4-9', 'No. of Visits:10-19',
                 'No. of Visits:>=20',
                 'No. of hospitalizations:0', 'No. of hospitalizations:1', 'No. of hospitalizations:>=1']
    col_names_out = col_names
    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in
         col_names])

    # ADI
    row_names.append('Median area deprivation index (IQR) — rank')
    records.append([
        _quantile_str(df['adi']),
        _quantile_str(df_pos['adi']),
        _quantile_str(df_neg['adi']),
        _smd(df_pos['adi'], df_neg['adi'])
    ])

    col_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
                 'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100',
                 'ADIMissing']
    row_names.extend(col_names)
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

    # # Smoking:
    # col_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])
    #
    # # Vaccine:
    # col_names = ['Fully vaccinated - Pre-index',
    #              'Fully vaccinated - Post-index',
    #              'Partially vaccinated - Pre-index',
    #              'Partially vaccinated - Post-index',
    #              'No evidence - Pre-index',
    #              'No evidence - Post-index',
    #              ]
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # time of index period
    row_names.append('Index time period of patients — no. (%)')
    records.append([])

    # part 1
    col_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21',
                 '03/21-06/21', '07/21-10/21', '11/21-02/22',
                 '03/22-06/22', '07/22-10/22', '11/22-02/23',
                 '03/23-06/23', '07/23-10/23', '11/23-02/24', ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # part 2
    col_names = ["YM: January 2022", "YM: February 2022", "YM: March 2022", "YM: April 2022", "YM: May 2022",
                 "YM: June 2022", "YM: July 2022", "YM: August 2022", "YM: September 2022",
                 "YM: October 2022", "YM: November 2022", "YM: December 2022",
                 "YM: January 2023", "YM: February 2023", ]

    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])


    df_out = pd.DataFrame(records, columns=output_columns, index=row_names)
    df_out['SMD'] = df_out['SMD'].astype(float)
    df_out.to_excel(out_file)
    print('Dump done ', df_out)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
