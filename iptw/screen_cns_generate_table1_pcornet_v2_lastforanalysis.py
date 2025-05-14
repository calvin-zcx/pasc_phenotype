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
import random

print = functools.partial(print, flush=True)


# from iptw.PSModels import ml
# from iptw.evaluation import *
def _t2eall_to_int_list_dedup(t2eall):
    t2eall = t2eall.strip(';').split(';')
    t2eall = set(map(int, t2eall))
    t2eall = sorted(t2eall)

    return t2eall


# def add_col(df):
#     cnsldn_names = [
#         'adderall_combo', 'lisdexamfetamine', 'methylphenidate', 'dexmethylphenidate',
#     ]
#
#     # step 1: add index date
#     # step 2: add ADHD
#     df['index_date_drug'] = np.nan
#     df['index_date_drug_days'] = np.nan
#     df['ADHD_before_drug_onset'] = 0
#
#     for index, row in tqdm(df.iterrows(), total=len(df)):
#         index_date = pd.to_datetime(row['index date'])
#
#         # step 1: drug onset day
#         drug_onsetday_list = []
#         for x in cnsldn_names:
#             t2eall = row['cnsldn-t2eall@' + x]
#             if pd.notna(t2eall):
#                 t2eall = _t2eall_to_int_list_dedup(t2eall)
#                 for t2e in t2eall:
#                     if 0 <= t2e < 30:
#                         drug_onsetday_list.append(t2e)
#
#         drug_onsetday_list = sorted(drug_onsetday_list)
#         if drug_onsetday_list:
#             if row['treated'] == 1:
#                 index_date_drug = index_date + datetime.timedelta(days=drug_onsetday_list[0])
#                 df.loc[index, 'index_date_drug'] = index_date_drug
#                 df.loc[index, 'index_date_drug_days'] = drug_onsetday_list[0]
#             else:
#                 index_date_drug = index_date  # or matching on index_date_drug_days
#
#
#         # step 2: ADHD at drug
#         t2eallADHD = row['dxcovCNSLDN-t2eall@ADHD']
#         if pd.notna(t2eallADHD):
#             t2eallADHD = _t2eall_to_int_list_dedup(t2eallADHD)
#             for t in t2eallADHD:
#                 if (row['treated'] == 1) & (len(drug_onsetday_list) > 0):
#                     if t <= drug_onsetday_list[0]:
#                         df.loc[index, 'ADHD_before_drug_onset'] = 1
#                 else:
#                     if t <= 0:
#                         df.loc[index, 'ADHD_before_drug_onset'] = 1
#
#     selected_cols = [x for x in df.columns if (
#             x.startswith('DX:') or
#             x.startswith('MEDICATION:') or
#             x.startswith('CCI:') or
#             x.startswith('obc:')
#     )]
#     df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')
#
#     # baseline part have been binarized already
#     selected_cols = [x for x in df.columns if
#                      (x.startswith('dx-out@') or
#                       x.startswith('dxadd-out@') or
#                       x.startswith('dxbrainfog-out@') or
#                       x.startswith('covidmed-out@') or
#                       x.startswith('smm-out@') or
#                       x.startswith('dxdxCFR-out@') or
#                       x.startswith('mental-base@') or
#                       x.startswith('dxcovCNSLDN-base@')
#                       )]
#     df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')
#
#     df.loc[df['death t2e'] < 0, 'death'] = np.nan
#     df.loc[df['death t2e'] < 0, 'death t2e'] = 9999
#     df.loc[df['death t2e'] == 9999, 'death t2e'] = np.nan
#
#     df['death in acute'] = (df['death t2e'] <= 30).astype('int')
#     df['death post acute'] = (df['death t2e'] > 30).astype('int')
#
#     return df


def table1_less_4_print(exptype='all'):
    # %% Step 1. Load  Data
    start_time = time.time()
    np.random.seed(0)
    random.seed(0)

    in_file = './cns_output/cohort-CNS-baseADHD-all-adhdCNS-inci-0-30s5.csv'

    print('in read: ', in_file)
    df = pd.read_csv(in_file,
                     dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'dob',
                                  'flag_delivery_date',
                                  'flag_pregnancy_start_date',
                                  'flag_pregnancy_end_date'
                                  ],
                     )
    print('df.shape:', df.shape)
    print('Read Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    df1 = df.loc[(df['treated'] == 1), :]
    df0 = df.loc[(df['treated'] == 0), :]
    case_label = 'Treated'
    ctrl_label = 'Nontreated'

    n1 = len(df1)
    n0 = len(df0)
    print('n1', n1, 'n0', n0)
    df1['treated'] = 1
    df0['treated'] = 0

    df = pd.concat([df1, df0], ignore_index=True)
    selected_cols = [x for x in df.columns if (
                    x.startswith('DX:') or
                    x.startswith('MEDICATION:') or
                    x.startswith('CCI:') or
                    x.startswith('obc:') or
                    x.startswith('mental-base@') or
                    x.startswith('dxcovCNSLDN-base@') or
                    x.startswith('dxMECFS-base@') or
                    x.startswith('dxCVDdeath-base@') or
                    x.startswith('dxcovCNSLDN-base@') or
                    x.startswith('PaxRisk')
            )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    # df = add_col(df)

    df_pos = df.loc[df['treated'] == 1, :]
    df_neg = df.loc[df['treated'] == 0, :]

    out_file = in_file.replace('.csv', '-Table.xlsx')
    output_columns = ['All', case_label, ctrl_label, 'SMD']

    print('treated df_pos.shape', df_pos.shape,
          'control df_neg.shape', df_neg.shape,
          'combined df.shape', df.shape, )
    # return df, df
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

    # row_names.append('PASC — no. (%)')
    # records.append([])
    # col_names = ['any_pasc_flag', ]
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    row_names.append('Drugs — no. (%)')
    records.append([])

    drug_col = ['cnsldn-treat-0-30@adderall_combo', 'cnsldn-treat-0-30@lisdexamfetamine',
               'cnsldn-treat-0-30@methylphenidate', 'cnsldn-treat-0-30@dexmethylphenidate']
    drug_col_output = ['adderall', 'lisdexamfetamine', 'methylphenidate', 'dexmethylphenidate']
    row_names.extend(drug_col_output)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in drug_col])

    # Sex
    row_names.append('Sex — no. (%)')
    records.append([])
    sex_col = ['Female', 'Male', 'Other/Missing']
    # sex_col = ['Female', 'Male']

    row_names.extend(sex_col)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in sex_col])

    # time
    row_names.append('Median drug onset time (IQR) — days')
    records.append([
        _quantile_str(df['index_date_drug_days']),
        _quantile_str(df_pos['index_date_drug_days']),
        _quantile_str(df_neg['index_date_drug_days']),
        _smd(df_pos['index_date_drug_days'], df_neg['index_date_drug_days'])
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

    age_col = ['age@18-24', 'age@25-34', 'age@35-49', 'age@50-64', 'age@65+']
    age_col_output = ['18-24', '25-34', '35-49', '50-64', '65+']
    row_names.extend(age_col_output)
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
    #
    # row_names.append('T2 Death days (IQR)')
    # records.append([
    #     _quantile_str(df['death t2e']),
    #     _quantile_str(df_pos['death t2e']),
    #     _quantile_str(df_neg['death t2e']),
    #     _smd(df_pos['death t2e'], df_neg['death t2e'])
    # ])
    # col_names = ['death', 'death in acute', 'death post acute']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in
    #      col_names])

    # # utilization
    # row_names.append('No. of hospital visits in the past 3 yr — no. (%)')
    # records.append([])
    # # part 1
    # col_names = ['No. of Visits:0', 'No. of Visits:1-3', 'No. of Visits:4-9', 'No. of Visits:10-19',
    #              'No. of Visits:>=20',
    #              'No. of hospitalizations:0', 'No. of hospitalizations:1', 'No. of hospitalizations:>=1']
    # col_names_out = col_names
    # row_names.extend(col_names_out)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in
    #      col_names])

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

    # ADI
    row_names.append('Median area deprivation index (IQR) — rank')
    records.append([
        _quantile_str(df['adi']),
        _quantile_str(df_pos['adi']),
        _quantile_str(df_neg['adi']),
        _smd(df_pos['adi'], df_neg['adi'])
    ])

    # col_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
    #              'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100',
    #              'ADIMissing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # BMI
    row_names.append('BMI (IQR)')
    records.append([
        _quantile_str(df['bmi']),
        _quantile_str(df_pos['bmi']),
        _quantile_str(df_neg['bmi']),
        _smd(df_pos['bmi'], df_neg['bmi'])
    ])

    # col_names = ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight',
    #              'BMI: 25-<30 overweight ', 'BMI: >=30 obese ', 'BMI: missing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # Smoking:
    # col_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']
    # row_names.extend(col_names)
    # records.extend(
    #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])

    # Vaccine:
    col_names = ['Fully vaccinated - Pre-index',
                 # 'Fully vaccinated - Post-index',
                 'Partially vaccinated - Pre-index',
                 # 'Partially vaccinated - Post-index',
                 'No evidence - Pre-index',
                 # 'No evidence - Post-index',
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
                 '03/22-06/22', '07/22-10/22', '11/22-02/23',
                 '03/23-06/23', '07/23-10/23', '11/23-02/24',
                 '03/24-06/24', '07/24-10/24', '11/24-02/25',
                 ]
    # col_names = [
    #              '03/22-06/22', '07/22-10/22', '11/22-02/23',
    #              ]
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # Coexisting coditions
    row_names.append('Coexisting conditions — no. (%)')

    records.append([])
    col_names = (
            ['dxcovCNSLDN-base@ADHD', 'ADHD_before_drug_onset',
             'dxcovCNSLDN-base@Narcolepsy', 'dxcovCNSLDN-base@MECFS', 'dxcovCNSLDN-base@Pain',
             'dxcovCNSLDN-base@alcohol opioid other substance ', 'dxcovCNSLDN-base@traumatic brain injury',
             'dxcovCNSLDN-base@TBI-associated Symptoms'] +
            ['PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
             'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
             'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
             'PaxRisk:Heart conditions', 'PaxRisk:Hypertension', 'PaxRisk:HIV infection',
             'PaxRisk:Immunocompromised condition or weakened immune system', 'PaxRisk:Mental health conditions',
             'PaxRisk:Overweight and obesity', 'PaxRisk:Pregnancy',
             'PaxRisk:Sickle cell disease or thalassemia',
             'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
             'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis'] +
            ["DX: Coagulopathy", "DX: Peripheral vascular disorders ", "DX: Seizure/Epilepsy", "DX: Weight Loss",
             'DX: Obstructive sleep apnea', 'DX: Epstein-Barr and Infectious Mononucleosis (Mono)', 'DX: Herpes Zoster',
             'mental-base@Schizophrenia Spectrum and Other Psychotic Disorders',
             'mental-base@Depressive Disorders',
             'mental-base@Bipolar and Related Disorders',
             'mental-base@Anxiety Disorders',
             'mental-base@Obsessive-Compulsive and Related Disorders',
             'mental-base@Post-traumatic stress disorder',
             'mental-base@Bulimia nervosa',
             'mental-base@Binge eating disorder',
             'mental-base@premature ejaculation',
             'mental-base@Autism spectrum disorder',
             'mental-base@Premenstrual dysphoric disorder',
             'mental-base@SMI',
             'mental-base@non-SMI',

             ]
        # + ['cnsldn-treat--1095-0@' + x for x in [
        # 'naltrexone', 'LDN_name', 'adderall_combo', 'lisdexamfetamine', 'methylphenidate',
        # 'amphetamine', 'amphetamine_nocombo', 'dextroamphetamine', 'dextroamphetamine_nocombo', 'modafinil',
        # 'pitolisant', 'solriamfetol', 'armodafinil', 'atomoxetine', 'benzphetamine',
        # 'azstarys_combo', 'dexmethylphenidate', 'dexmethylphenidate_nocombo', 'diethylpropion', 'methamphetamine',
        # 'phendimetrazine', 'phentermine', 'caffeine', 'fenfluramine_delet', 'oxybate_delet',
        # 'doxapram_delet', 'guanfacine']]
    )

    col_names_out = (['ADHD', 'ADHD_before_drug_onset', 'Narcolepsy', 'MECFS', 'Pain',
                      'alcohol opioid other substance', 'traumatic brain injury',
                    'TBI-associated Symptoms'] + ['Cancer', 'Chronic kidney disease', 'Chronic liver disease',
                                                           'Chronic lung disease', 'Cystic fibrosis',
                                                           'Dementia or other neurological conditions', 'Diabetes',
                                                           'Disabilities',
                                                           'Heart conditions', 'Hypertension', 'HIV infection',
                                                           'Immunocompromised condition or weakened immune system',
                                                           'Mental health conditions',
                                                           'Overweight and obesity', 'Pregnancy',
                                                           'Sickle cell disease or thalassemia',
                                                           'Smoking current or former',
                                                           'Stroke or cerebrovascular disease',
                                                           'Substance use disorders', 'Tuberculosis', ] +
                     ["Coagulopathy", "Peripheral vascular disorders ", "Seizure/Epilepsy", "Weight Loss",
                      'Obstructive sleep apnea', 'Epstein-Barr and Infectious Mononucleosis (Mono)', 'Herpes Zoster',
                      'Schizophrenia Spectrum and Other Psychotic Disorders',
                      'Depressive Disorders',
                      'Bipolar and Related Disorders',
                      'Anxiety Disorders',
                      'Obsessive-Compulsive and Related Disorders',
                      'Post-traumatic stress disorder',
                      'Bulimia nervosa',
                      'Binge eating disorder',
                      'premature ejaculation',
                      'Autism spectrum disorder',
                      'Premenstrual dysphoric disorder',
                      'SMI',
                      'non-SMI',
                    ]
                     # + [
                     #     'naltrexone', 'LDN_name', 'adderall_combo', 'lisdexamfetamine', 'methylphenidate',
                     #     'amphetamine', 'amphetamine_nocombo', 'dextroamphetamine', 'dextroamphetamine_nocombo',
                     #     'modafinil',
                     #     'pitolisant', 'solriamfetol', 'armodafinil', 'atomoxetine', 'benzphetamine',
                     #     'azstarys_combo', 'dexmethylphenidate', 'dexmethylphenidate_nocombo', 'diethylpropion',
                     #     'methamphetamine',
                     #     'phendimetrazine', 'phentermine', 'caffeine', 'fenfluramine_delet', 'oxybate_delet',
                     #     'doxapram_delet', 'guanfacine']
                     )

    row_names.extend(col_names_out)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

    # col_names = ['score_cci_charlson', 'score_cci_quan']
    # col_names_out = ['score_cci_charlson', 'score_cci_quan']
    # row_names.extend(col_names_out)
    # records.extend(
    #     [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
    #      for c in col_names])
    # row_names.append('CCI Score — no. (%)')
    # records.append([])

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

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return df, df_out


if __name__ == '__main__':
    start_time = time.time()


    df, df_out = table1_less_4_print(exptype='CNS-ADHD-acuteIncident-0-30')  # 'ssri-base180-acutevsnot'


    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
