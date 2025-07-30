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


def add_col(df):
    print('add and revise some covs per the needs. in addition to prior screen_build_naltrexone file')
    drug_names = [
        'naltrexone',
    ]

    # step 1: add index date
    # step 2: add ADHD
    df['index_date_drug'] = np.nan
    df['index_date_drug_days'] = np.nan
    df['ADHD_before_drug_onset'] = 0

    covNaltrexone_med_names = ['NSAIDs_combined', 'opioid drug']
    for x in covNaltrexone_med_names:
        df['covNaltrexone_med-base@' + x] = 0  # -3yrs to 0
        df['covNaltrexone_med-basedrugonset@' + x] = 0  # target drug, acute Naltrexone time, no left limit

    covNaltrexone_names = ['MECFS', 'Pain', 'pain include', 'pain tbd', 'pain exclude',
                           'substance use disorder ', 'opioid use disorder',
                           'Opioid induced constipation', 'Obesity',
                           'Crohn-Inflamm_Bowel', 'fibromyalgia', 'multiple sclerosis', 'POTS']
    for x in covNaltrexone_names:
        df['dxcovNaltrexone-basedrugonset@' + x] = 0  # target drug, acute Naltrexone time, no left limit

    for index, row in tqdm(df.iterrows(), total=len(df)):
        index_date = pd.to_datetime(row['index date'])

        # step 1: drug onset day
        drug_onsetday_list = []
        for x in drug_names:
            t2eall = row['cnsldn-t2eall@' + x]
            if pd.notna(t2eall):
                t2eall = _t2eall_to_int_list_dedup(t2eall)
                for t2e in t2eall:
                    if 0 <= t2e < 30:
                        drug_onsetday_list.append(t2e)

        drug_onsetday_list = sorted(drug_onsetday_list)
        if drug_onsetday_list:
            if row['treated'] == 1:
                index_date_drug = index_date + datetime.timedelta(days=drug_onsetday_list[0])
                df.loc[index, 'index_date_drug'] = index_date_drug
                df.loc[index, 'index_date_drug_days'] = drug_onsetday_list[0]
            else:
                index_date_drug = index_date  # or matching on index_date_drug_days

        # step 2: ADHD at drug
        t2eallADHD = row['dxcovCNSLDN-t2eall@ADHD']
        if pd.notna(t2eallADHD):
            t2eallADHD = _t2eall_to_int_list_dedup(t2eallADHD)
            for t in t2eallADHD:
                if (row['treated'] == 1) & (len(drug_onsetday_list) > 0):
                    if t <= drug_onsetday_list[0]:
                        df.loc[index, 'ADHD_before_drug_onset'] = 1
                else:
                    if t <= 0:
                        df.loc[index, 'ADHD_before_drug_onset'] = 1

        # step 3: ['NSAIDs_combined', 'opioid drug'], at covid t0 and target drug index t0
        for x in covNaltrexone_med_names:
            t2eall = row['covNaltrexone_med-t2eall@' + x]
            if pd.notna(t2eall):
                t2eall = _t2eall_to_int_list_dedup(t2eall)

                for t2e in t2eall:
                    if -1095 <= t2e <= -7:
                        df.loc[index, 'covNaltrexone_med-base@' + x] = 1

                    # just before drug index for treated, covid 0 for untreated, no left bound
                    if (row['treated'] == 1) & (len(drug_onsetday_list) > 0):
                        if t2e <= drug_onsetday_list[0]:
                            df.loc[index, 'covNaltrexone_med-basedrugonset@' + x] = 1
                    else:
                        if t2e <= 0:
                            df.loc[index, 'covNaltrexone_med-basedrugonset@' + x] = 1

        # step 4: ['MECFS', 'Pain', 'substance use disorder ', 'opioid use disorder',
        #  'Opioid induced constipation', 'Obesity', 'Crohn-Inflamm_Bowel', 'fibromyalgia',
        #  'multiple sclerosis', 'POTS'], at target drug index t0
        for x in covNaltrexone_names:
            t2eall = row['dxcovNaltrexone-t2eall@' + x]
            if pd.notna(t2eall):
                t2eall = _t2eall_to_int_list_dedup(t2eall)

                for t2e in t2eall:
                    # just before drug index for treated, covid 0 for untreated, no left bound
                    if (row['treated'] == 1) & (len(drug_onsetday_list) > 0):
                        if t2e <= drug_onsetday_list[0]:
                            df.loc[index, 'dxcovNaltrexone-basedrugonset@' + x] = 1
                    else:
                        if t2e <= 0:
                            df.loc[index, 'dxcovNaltrexone-basedrugonset@' + x] = 1

        # end
    selected_cols = [x for x in df.columns if (
            x.startswith('DX:') or
            x.startswith('MEDICATION:') or
            x.startswith('CCI:') or
            x.startswith('obc:')
    )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    # baseline part have been binarized already
    selected_cols = [x for x in df.columns if
                     (x.startswith('dx-out@') or
                      x.startswith('dxadd-out@') or
                      x.startswith('dxbrainfog-out@') or
                      x.startswith('covidmed-out@') or
                      x.startswith('smm-out@') or
                      x.startswith('dxdxCFR-out@') or
                      x.startswith('mental-base@') or
                      x.startswith('dxcovCNSLDN-base@') or
                      x.startswith('dxcovNaltrexone-base@')  # or
                      # x.startswith('dxcovNaltrexone-base@')
                      )]
    df.loc[:, selected_cols] = (df.loc[:, selected_cols].astype('int') >= 1).astype('int')

    df.loc[df['death t2e'] < 0, 'death'] = np.nan
    df.loc[df['death t2e'] < 0, 'death t2e'] = 9999
    df.loc[df['death t2e'] == 9999, 'death t2e'] = np.nan

    df['death in acute'] = (df['death t2e'] <= 30).astype('int')
    df['death post acute'] = (df['death t2e'] > 30).astype('int')

    return df


# def add_any_pasc(df):
#     df_pasc_info = pd.read_excel(r'../prediction/output/causal_effects_specific_withMedication_v3.xlsx',
#                                  sheet_name='diagnosis')
#     addedPASC_encoding = utils.load(r'../data/mapping/addedPASC_index_mapping.pkl')
#     addedPASC_list = list(addedPASC_encoding.keys())
#     brainfog_encoding = utils.load(r'../data/mapping/brainfog_index_mapping.pkl')
#     brainfog_list = list(brainfog_encoding.keys())
#
#     pasc_simname = {}
#     pasc_organ = {}
#     for index, rows in df_pasc_info.iterrows():
#         pasc_simname[rows['pasc']] = (rows['PASC Name Simple'], rows['Organ Domain'])
#         pasc_organ[rows['pasc']] = rows['Organ Domain']
#
#     for p in addedPASC_list:
#         pasc_simname[p] = (p, 'General-add')
#         pasc_organ[p] = 'General-add'
#
#     for p in brainfog_list:
#         pasc_simname[p] = (p, 'brainfog')
#         pasc_organ[p] = 'brainfog'
#
#     # pasc_list = df_pasc_info.loc[df_pasc_info['selected'] == 1, 'pasc']
#     pasc_list_raw = df_pasc_info.loc[df_pasc_info['selected_narrow'] == 1, 'pasc'].to_list()
#     _exclude_list = ['Pressure ulcer of skin', 'Fluid and electrolyte disorders']
#     pasc_list = [x for x in pasc_list_raw if x not in _exclude_list]
#
#     pasc_add = ['smell and taste', ]
#     print('len(pasc_list)', len(pasc_list), 'len(pasc_add)', len(pasc_add))
#
#     for p in pasc_list:
#         df[p + '_pasc_flag'] = 0
#     for p in pasc_add:
#         df[p + '_pasc_flag'] = 0
#
#     df['any_pasc_flag'] = 0
#     df['any_pasc_type'] = np.nan
#     df['any_pasc_t2e'] = 180  # np.nan
#     df['any_pasc_txt'] = ''
#     df['any_pasc_baseline'] = 0  # placeholder for screening, no special meaning, null column
#     for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
#         # for any 1 pasc
#         t2e_list = []
#         pasc_1_list = []
#         pasc_1_name = []
#         pasc_1_text = ''
#         for p in pasc_list:
#             if (rows['dx-out@' + p] > 0) and (rows['dx-base@' + p] == 0):
#                 t2e_list.append(rows['dx-t2e@' + p])
#                 pasc_1_list.append(p)
#                 pasc_1_name.append(pasc_simname[p])
#                 pasc_1_text += (pasc_simname[p][0] + ';')
#
#                 df.loc[index, p + '_pasc_flag'] = 1
#
#         for p in pasc_add:
#             if (rows['dxadd-out@' + p] > 0) and (rows['dxadd-base@' + p] == 0):
#                 t2e_list.append(rows['dxadd-t2e@' + p])
#                 pasc_1_list.append(p)
#                 pasc_1_name.append(pasc_simname[p])
#                 pasc_1_text += (pasc_simname[p][0] + ';')
#
#                 df.loc[index, p + '_pasc_flag'] = 1
#
#         if len(t2e_list) > 0:
#             df.loc[index, 'any_pasc_flag'] = 1
#             df.loc[index, 'any_pasc_t2e'] = np.min(t2e_list)
#             df.loc[index, 'any_pasc_txt'] = pasc_1_text
#         else:
#             df.loc[index, 'any_pasc_flag'] = 0
#             df.loc[index, 'any_pasc_t2e'] = rows[['dx-t2e@' + p for p in pasc_list]].max()  # censoring time
#
#     return df


# def table1_more_4_analyse(exptype, cohort='all', subgroup='all'):
#     # %% Step 1. Load  Data
#     start_time = time.time()
#
#     np.random.seed(0)
#     random.seed(0)
#     negative_ratio = 5
#
#     # add matched cohorts later
#     in_file = '../iptw/recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNet-SSRI-v2-addPaxFeats-addGeneralEC-withexposure.csv'
#     in_file = '../iptw/recover29Nov27_covid_pos_addCFR-PaxRisk-U099-Hospital-Preg_4PCORNet-SSRI-v5-withmental-addPaxFeats-addGeneralEC-withexposure.csv'
#
#     print('in read: ', in_file)
#     df = pd.read_csv(in_file,
#                      dtype={'patid': str, 'site': str, 'zip': str},
#                      parse_dates=['index date', 'dob',
#                                   'flag_delivery_date',
#                                   'flag_pregnancy_start_date',
#                                   'flag_pregnancy_end_date'
#                                   ])
#     print('df.shape:', df.shape)
#
#     # define treated and untreated here
#     # 2024-09-07 replace 'PaxRisk:Mental health conditions' with 'SSRI-Indication-dsmAndExlix-flag'
#     # control group, not using snri criteria? --> add -clean group
#     if exptype == 'ssri-base-180-0':
#         df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#                 df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         case_label = 'SSRI-180-0'
#         ctrl_label = 'Nouser'
#     elif exptype == 'ssri-base-180-0-clean':
#         df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['snri-treat--180-180-flag'] == 0)
#                      & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0)
#                      & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         case_label = 'SSRI-180-0-clean'
#         ctrl_label = 'Nouser-clean'
#
#     elif exptype == 'ssri-base-180-0-cleanv2':
#         df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['snri-treat--180-0-flag'] == 0)
#                      & (df['other-treat--180-0-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-0-flag'] == 0) & (df['snri-treat--180-0-flag'] == 0)
#                      & (df['other-treat--180-0-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         case_label = 'SSRI-180-0-clean'
#         ctrl_label = 'Nouser-clean'
#
#     elif exptype == 'ssri-base180-acutevsnot':
#         df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['ssri-treat-0-15-flag'] >= 1), :]
#         df0 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['ssri-treat-0-15-flag'] == 0), :]
#         case_label = 'SSRI-180-with-acute15'
#         ctrl_label = 'SSRI-180-no-acute15'
#
#     elif exptype == 'ssri-base180withmental-acutevsnot':
#         df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['ssri-treat-0-15-flag'] >= 1) & (
#                 df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['ssri-treat-0-15-flag'] == 0) & (
#                 df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         case_label = 'SSRI-180withmental-with-acute15'
#         ctrl_label = 'SSRI-180withmental-no-acute15'
#
#     # elif exptype == 'ssri-base-120-0':
#     #     df1 = df.loc[(df['ssri-treat--120-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#     #     df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#     #             df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#     #     case_label = 'SSRI-120-0'
#     #     ctrl_label = 'Nouser'
#     # elif exptype == 'ssri-acute0-7':
#     #     df1 = df.loc[(df['ssri-treat-0-7-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#     #     df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#     #             df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#     #     case_label = 'SSRI-0-7'
#     #     ctrl_label = 'Nouser'
#     elif exptype == 'ssri-acute0-15':
#         df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#                 df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         case_label = 'SSRI-0-15'
#         ctrl_label = 'Nouser'
#
#     elif exptype == 'ssri-acute0-15-clean':
#         df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['snri-treat--180-180-flag'] == 0)
#                      & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0)
#                      & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#
#         case_label = 'SSRI-0-15-clean'
#         ctrl_label = 'Nouser-clean'
#
#     elif exptype == 'snri-base-180-0':
#         df1 = df.loc[(df['snri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#                 df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         case_label = 'SNRI-180-0'
#         ctrl_label = 'Nouser'
#
#     elif exptype == 'snri-base-180-0-clean':
#         df1 = df.loc[(df['snri-treat--180-0-flag'] >= 1) & (df['ssri-treat--180-180-flag'] == 0)
#                      & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0)
#                      & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         case_label = 'SNRI-180-0-clean'
#         ctrl_label = 'Nouser-clean'
#     # elif exptype == 'snri-base-120-0':
#     #     df1 = df.loc[(df['snri-treat--120-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#     #     df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#     #             df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#     #     case_label = 'SNRI-120-0'
#     #     ctrl_label = 'Nouser'
#     # elif exptype == 'snri-acute0-7':
#     #     df1 = df.loc[(df['snri-treat-0-7-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#     #     df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#     #             df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#     #     case_label = 'SNRI-0-7'
#     #     ctrl_label = 'Nouser'
#     elif exptype == 'snri-acute0-15':
#         df1 = df.loc[(df['snri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#                 df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         case_label = 'SNRI-0-15'
#         ctrl_label = 'Nouser'
#
#     elif exptype == 'snri-acute0-15-clean':
#         df1 = df.loc[(df['snri-treat-0-15-flag'] >= 1) & (df['ssri-treat--180-180-flag'] == 0)
#                      & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0)
#                      & (df['other-treat--180-180-flag'] == 0) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         case_label = 'SNRI-180-0-clean'
#         ctrl_label = 'Nouser-clean'
#
#     elif exptype == 'ssriVSsnri-base-180-0':
#         df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['snri-treat--180-180-flag'] == 0), :]
#         df0 = df.loc[(df['snri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['ssri-treat--180-180-flag'] == 0), :]
#         case_label = 'SSRI-180-0'
#         ctrl_label = 'SNRI-180-0'
#
#     elif exptype == 'ssriVSsnri-base-180-0-clean':
#
#         df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['snri-treat--180-180-flag'] == 0) & (df['other-treat--180-180-flag'] == 0), :]
#         df0 = df.loc[(df['snri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['ssri-treat--180-180-flag'] == 0) & (df['other-treat--180-180-flag'] == 0), :]
#         case_label = 'SSRI-180-0-clean'
#         ctrl_label = 'SNRI-180-0-clean'
#
#     # elif exptype == 'ssriVSsnri-base-120-0':
#     #     df1 = df.loc[(df['ssri-treat--120-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#     #                 df['snri-treat--180-180-flag'] == 0), :]
#     #     df0 = df.loc[(df['snri-treat--120-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#     #                 df['ssri-treat--180-180-flag'] == 0), :]
#     #     case_label = 'SSRI-120-0'
#     #     ctrl_label = 'SNRI-120-0'
#     # elif exptype == 'ssriVSsnri-acute0-7':
#     #     df1 = df.loc[(df['ssri-treat-0-7-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#     #                 df['snri-treat--180-180-flag'] == 0), :]
#     #     df0 = df.loc[(df['snri-treat-0-7-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#     #                 df['ssri-treat--180-180-flag'] == 0), :]
#     #     case_label = 'SSRI-0-7'
#     #     ctrl_label = 'SNRI-0-7'
#     elif exptype == 'ssriVSsnri-acute0-15':
#         df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['snri-treat--180-180-flag'] == 0), :]
#         df0 = df.loc[(df['snri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['ssri-treat--180-180-flag'] == 0), :]
#         case_label = 'SSRI-0-15'
#         ctrl_label = 'SNRI-0-15'
#
#     elif exptype == 'ssriVSsnri-acute0-15-clean':
#         df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['snri-treat--180-180-flag'] == 0) & (df['other-treat--180-180-flag'] == 0), :]
#         df0 = df.loc[(df['snri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['ssri-treat--180-180-flag'] == 0) & (df['other-treat--180-180-flag'] == 0), :]
#         case_label = 'SSRI-0-15-clean'
#         ctrl_label = 'SNRI-0-15-clean'
#
#     elif exptype == 'ssriVSbupropion-base-180-0':
#         df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['other-treat--180-180@bupropion'] == 0), :]
#         df0 = df.loc[(df['other-treat--180-0@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['ssri-treat--180-180-flag'] == 0), :]
#         case_label = 'SSRI-180-0'
#         ctrl_label = 'bupropion-180-0'
#
#     elif exptype == 'ssriVSbupropion-base-180-0-clean':
#         df1 = df.loc[(df['ssri-treat--180-0-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['other-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
#         df0 = df.loc[(df['other-treat--180-0@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
#         case_label = 'SSRI-180-0-clean'
#         ctrl_label = 'bupropion-180-0-clean'
#
#     elif exptype == 'ssriVSbupropion-acute0-15':
#         df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['other-treat--180-180@bupropion'] == 0), :]
#         df0 = df.loc[(df['other-treat-0-15@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['ssri-treat--180-180-flag'] == 0), :]
#         case_label = 'SSRI-0-15'
#         ctrl_label = 'bupropion-0-15'
#
#     elif exptype == 'ssriVSbupropion-acute0-15-clean':
#         df1 = df.loc[(df['ssri-treat-0-15-flag'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['other-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
#         df0 = df.loc[(df['other-treat-0-15@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (
#                 df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
#         case_label = 'SSRI-0-15-clean'
#         ctrl_label = 'bupropion-0-15-clean'
#
#     elif exptype == 'bupropion-base-180-0':
#         df1 = df.loc[(df['other-treat--180-0@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#                 df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (df['other-treat--180-180@bupropion'] == 0), :]
#         case_label = 'bupropion-180-0'
#         ctrl_label = 'Nouser'
#
#     elif exptype == 'bupropion-base-180-0-clean':
#         df1 = df.loc[(df['other-treat--180-0@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) &
#                      (df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#                 df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (df['other-treat--180-180-flag'] == 0), :]
#         case_label = 'bupropion-180-0-clean'
#         ctrl_label = 'Nouser-clean'
#
#
#     elif exptype == 'bupropion-acute0-15':
#         df1 = df.loc[(df['other-treat-0-15@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#                 df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (df['other-treat--180-180@bupropion'] == 0), :]
#         case_label = 'bupropion-0-15'
#         ctrl_label = 'Nouser'
#
#     elif exptype == 'bupropion-acute0-15-clean':
#         df1 = df.loc[(df['other-treat-0-15@bupropion'] >= 1) & (df['SSRI-Indication-dsmAndExlix-flag'] > 0) &
#                      (df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0), :]
#         df0 = df.loc[(df['ssri-treat--180-180-flag'] == 0) & (df['snri-treat--180-180-flag'] == 0) & (
#                 df['SSRI-Indication-dsmAndExlix-flag'] > 0) & (df['other-treat--180-180-flag'] == 0), :]
#         case_label = 'bupropion-0-15-clean'
#         ctrl_label = 'Nouser-clean'
#
#     print('exposre strategy, exptype:', exptype)
#     print('check exposure strategy, n1, negative ratio, n0, if match on mental health, no-user definition')
#
#     n1 = len(df1)
#     n0 = len(df0)
#     print('n1', n1, 'n0', len(df0))
#     # df0 = df0.sample(n=min(len(df0), int(negative_ratio * n1)), replace=False, random_state=0)
#     # print('after sample, n0', len(df0), 'with ratio:', negative_ratio, negative_ratio * n1)
#     df1['treated'] = 1
#     df0['treated'] = 0
#
#     df = pd.concat([df1, df0], ignore_index=True)
#     df = add_col(df)
#
#     df_pos = df.loc[df['treated'] == 1, :]
#     df_neg = df.loc[df['treated'] == 0, :]
#
#     out_file = r'./ssri_output/Table-recover29Nov27_covid_pos_{}-mentalCov.xlsx'.format(exptype)
#     output_columns = ['All', case_label, ctrl_label, 'SMD']
#
#     print('treated df_pos.shape', df_pos.shape,
#           'control df_neg.shape', df_neg.shape,
#           'combined df.shape', df.shape, )
#
#     def _n_str(n):
#         return '{:,}'.format(n)
#
#     def _quantile_str(x):
#         v = x.quantile([0.25, 0.5, 0.75]).to_list()
#         return '{:.0f} ({:.0f}—{:.0f})'.format(v[1], v[0], v[2])
#
#     def _percentage_str(x):
#         n = x.sum()
#         per = x.mean()
#         return '{:,} ({:.1f})'.format(n, per * 100)
#
#     def _smd(x1, x2):
#         m1 = x1.mean()
#         m2 = x2.mean()
#         v1 = x1.var()
#         v2 = x2.var()
#
#         VAR = np.sqrt((v1 + v2) / 2)
#         smd = np.divide(
#             m1 - m2,
#             VAR, out=np.zeros_like(m1), where=VAR != 0)
#         return smd
#
#     row_names = []
#     records = []
#
#     # N
#     row_names.append('N')
#     records.append([
#         _n_str(len(df)),
#         _n_str(len(df_pos)),
#         _n_str(len(df_neg)),
#         np.nan
#     ])
#
#     # row_names.append('PASC — no. (%)')
#     # records.append([])
#     # col_names = ['any_pasc_flag', ]
#     # row_names.extend(col_names)
#     # records.extend(
#     #     [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#     #      for c in col_names])
#
#     # Sex
#     row_names.append('Sex — no. (%)')
#     records.append([])
#     sex_col = ['Female', 'Male', 'Other/Missing']
#     # sex_col = ['Female', 'Male']
#
#     row_names.extend(sex_col)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in sex_col])
#
#     row_names.append('Acute severity — no. (%)')
#     records.append([])
#     # col_names = ['outpatient', 'inpatient', 'icu', 'inpatienticu',
#     #              'hospitalized', 'ventilation', 'criticalcare', ]
#     col_names = ['outpatient', ]
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # age
#     row_names.append('Median age (IQR) — yr')
#     records.append([
#         _quantile_str(df['age']),
#         _quantile_str(df_pos['age']),
#         _quantile_str(df_neg['age']),
#         _smd(df_pos['age'], df_neg['age'])
#     ])
#
#     row_names.append('Age group — no. (%)')
#     records.append([])
#
#     age_col = ['age@18-24', 'age@25-34', 'age@35-49', 'age@50-64',
#                '65-<75 years', '75-<85 years', '85+ years']  # 'age@65+'
#     row_names.extend(age_col)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in age_col])
#
#     # Race
#     row_names.append('Race — no. (%)')
#     records.append([])
#     # col_names = ['Asian', 'Black or African American', 'White', 'Other', 'Missing']
#     col_names = ['RE:Asian Non-Hispanic', 'RE:Black or African American Non-Hispanic', 'RE:Hispanic or Latino Any Race',
#                  'RE:White Non-Hispanic', 'RE:Other Non-Hispanic', 'RE:Unknown']
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # Ethnic group
#     row_names.append('Ethnic group — no. (%)')
#     records.append([])
#     col_names = ['Hispanic: Yes', 'Hispanic: No', 'Hispanic: Other/Missing']
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # follow-up
#     row_names.append('Follow-up days (IQR)')
#     records.append([
#         _quantile_str(df['maxfollowup']),
#         _quantile_str(df_pos['maxfollowup']),
#         _quantile_str(df_neg['maxfollowup']),
#         _smd(df_pos['maxfollowup'], df_neg['maxfollowup'])
#     ])
#
#     row_names.append('T2 Death days (IQR)')
#     records.append([
#         _quantile_str(df['death t2e']),
#         _quantile_str(df_pos['death t2e']),
#         _quantile_str(df_neg['death t2e']),
#         _smd(df_pos['death t2e'], df_neg['death t2e'])
#     ])
#     col_names = ['death', 'death in acute', 'death post acute']
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in
#          col_names])
#
#     # utilization
#     row_names.append('No. of hospital visits in the past 3 yr — no. (%)')
#     records.append([])
#     # part 1
#     col_names = ['No. of Visits:0', 'No. of Visits:1-3', 'No. of Visits:4-9', 'No. of Visits:10-19',
#                  'No. of Visits:>=20',
#                  'No. of hospitalizations:0', 'No. of hospitalizations:1', 'No. of hospitalizations:>=1']
#     col_names_out = col_names
#     row_names.extend(col_names_out)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in
#          col_names])
#
#     # utilization
#     row_names.append('No. of hospital visits in the past 3 yr — no. (%)')
#     records.append([])
#     # part 1
#     col_names = ['inpatient no.', 'outpatient no.', 'emergency visits no.', 'other visits no.']
#     col_names_out = ['No. of Inpatient Visits', 'No. of Outpatient Visits',
#                      'No. of Emergency Visits', 'No. of Other Visits']
#
#     row_names.extend(col_names_out)
#     records.extend(
#         [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])] for c in
#          col_names])
#
#     # part2
#     df_pos['Inpatient >=3'] = df_pos['inpatient visits 3-4'] + df_pos['inpatient visits >=5']
#     df_neg['Inpatient >=3'] = df_neg['inpatient visits 3-4'] + df_neg['inpatient visits >=5']
#     df_pos['Outpatient >=3'] = df_pos['outpatient visits 3-4'] + df_pos['outpatient visits >=5']
#     df_neg['Outpatient >=3'] = df_neg['outpatient visits 3-4'] + df_neg['outpatient visits >=5']
#     df_pos['Emergency >=3'] = df_pos['emergency visits 3-4'] + df_pos['emergency visits >=5']
#     df_neg['Emergency >=3'] = df_neg['emergency visits 3-4'] + df_neg['emergency visits >=5']
#
#     df['Inpatient >=3'] = df['inpatient visits 3-4'] + df['inpatient visits >=5']
#     df['Outpatient >=3'] = df['outpatient visits 3-4'] + df['outpatient visits >=5']
#     df['Emergency >=3'] = df['emergency visits 3-4'] + df['emergency visits >=5']
#
#     col_names = ['inpatient visits 0', 'inpatient visits 1-2', 'Inpatient >=3',
#                  'outpatient visits 0', 'outpatient visits 1-2', 'Outpatient >=3',
#                  'emergency visits 0', 'emergency visits 1-2', 'Emergency >=3']
#     col_names_out = ['Inpatient 0', 'Inpatient 1-2', 'Inpatient >=3',
#                      'Outpatient 0', 'Outpatient 1-2', 'Outpatient >=3',
#                      'Emergency 0', 'Emergency 1-2', 'Emergency >=3']
#     row_names.extend(col_names_out)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # ADI
#     row_names.append('Median area deprivation index (IQR) — rank')
#     records.append([
#         _quantile_str(df['adi']),
#         _quantile_str(df_pos['adi']),
#         _quantile_str(df_neg['adi']),
#         _smd(df_pos['adi'], df_neg['adi'])
#     ])
#
#     col_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
#                  'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100',
#                  'ADIMissing']
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # BMI
#     row_names.append('BMI (IQR)')
#     records.append([
#         _quantile_str(df['bmi']),
#         _quantile_str(df_pos['bmi']),
#         _quantile_str(df_neg['bmi']),
#         _smd(df_pos['bmi'], df_neg['bmi'])
#     ])
#
#     col_names = ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight',
#                  'BMI: 25-<30 overweight ', 'BMI: >=30 obese ', 'BMI: missing']
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # Smoking:
#     col_names = ['Smoker: never', 'Smoker: current', 'Smoker: former', 'Smoker: missing']
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # Vaccine:
#     col_names = ['Fully vaccinated - Pre-index',
#                  'Fully vaccinated - Post-index',
#                  'Partially vaccinated - Pre-index',
#                  'Partially vaccinated - Post-index',
#                  'No evidence - Pre-index',
#                  'No evidence - Post-index',
#                  ]
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # time of index period
#     row_names.append('Index time period of patients — no. (%)')
#     records.append([])
#
#     # part 1
#     col_names = ['03/20-06/20', '07/20-10/20', '11/20-02/21',
#                  '03/21-06/21', '07/21-10/21', '11/21-02/22',
#                  '03/22-06/22', '07/22-10/22', '11/22-02/23',
#                  '03/23-06/23', '07/23-10/23', '11/23-02/24', ]
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # part 2
#     col_names = ["YM: March 2020", "YM: April 2020", "YM: May 2020", "YM: June 2020", "YM: July 2020",
#                  "YM: August 2020", "YM: September 2020", "YM: October 2020", "YM: November 2020", "YM: December 2020",
#                  "YM: January 2021", "YM: February 2021", "YM: March 2021", "YM: April 2021", "YM: May 2021",
#                  "YM: June 2021", "YM: July 2021", "YM: August 2021", "YM: September 2021", "YM: October 2021",
#                  "YM: November 2021", "YM: December 2021",
#                  "YM: January 2022", "YM: February 2022",
#                  "YM: March 2022", "YM: April 2022", "YM: May 2022",
#                  "YM: June 2022", "YM: July 2022", "YM: August 2022",
#                  "YM: September 2022",
#                  "YM: October 2022", "YM: November 2022",
#                  "YM: December 2022",
#                  "YM: January 2023", "YM: February 2023",
#                  "YM: March 2023", "YM: April 2023", "YM: May 2023",
#                  "YM: June 2023", "YM: July 2023", "YM: August 2023",
#                  "YM: September 2023", "YM: October 2023",
#                  "YM: November 2023", "YM: December 2023", ]
#
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # part 3
#     col_names = ['quart:01/22-03/22', 'quart:04/22-06/22', 'quart:07/22-09/22', 'quart:10/22-1/23', ]
#     row_names.extend(col_names)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     # Coexisting coditions
#     row_names.append('Coexisting conditions — no. (%)')
#
#     records.append([])
#     col_names = (['PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
#                   'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
#                   'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
#                   'PaxRisk:Heart conditions', 'PaxRisk:Hypertension', 'PaxRisk:HIV infection',
#                   'PaxRisk:Immunocompromised condition or weakened immune system', 'PaxRisk:Mental health conditions',
#                   'PaxRisk:Overweight and obesity', 'PaxRisk:Pregnancy', 'PaxRisk:Sickle cell disease or thalassemia',
#                   'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
#                   'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis', 'PaxExclude:liver',
#                   'PaxExclude:end-stage kidney disease'] +
#                  ["DX: Alcohol Abuse", "DX: Anemia",
#                   "DX: Arrythmia", "DX: Asthma",
#                   "DX: Cancer",
#                   "DX: Chronic Kidney Disease",
#                   "DX: Chronic Pulmonary Disorders",
#                   "DX: Cirrhosis",
#                   "DX: Coagulopathy",
#                   "DX: Congestive Heart Failure",
#                   "DX: COPD",
#                   "DX: Coronary Artery Disease",
#                   "DX: Dementia", "DX: Diabetes Type 1",
#                   "DX: Diabetes Type 2",
#                   # 'Type 1 or 2 Diabetes Diagnosis',
#                   "DX: End Stage Renal Disease on Dialysis",
#                   "DX: Hemiplegia",
#                   "DX: HIV", "DX: Hypertension",
#                   "DX: Hypertension and Type 1 or 2 Diabetes Diagnosis",
#                   "DX: Inflammatory Bowel Disorder",
#                   "DX: Lupus or Systemic Lupus Erythematosus",
#                   "DX: Mental Health Disorders",
#                   "DX: Multiple Sclerosis",
#                   "DX: Parkinson's Disease",
#                   "DX: Peripheral vascular disorders ",
#                   "DX: Pregnant",
#                   "DX: Pulmonary Circulation Disorder  (PULMCR_ELIX)",
#                   "DX: Rheumatoid Arthritis",
#                   "DX: Seizure/Epilepsy",
#                   "DX: Severe Obesity  (BMI>=40 kg/m2)",
#                   "DX: Weight Loss",
#                   "DX: Down's Syndrome",
#                   'DX: Other Substance Abuse',
#                   'DX: Cystic Fibrosis',
#                   'DX: Autism', 'DX: Sickle Cell',
#                   'DX: Obstructive sleep apnea',
#                   # added 2022-05-25
#                   'DX: Epstein-Barr and Infectious Mononucleosis (Mono)',
#                   # added 2022-05-25
#                   'DX: Herpes Zoster',  # added 2022-05-25
#                   "MEDICATION: Corticosteroids",
#                   "MEDICATION: Immunosuppressant drug"
#                   ] +
#                  ['obc:Placenta accreta spectrum',
#                   'obc:Pulmonary hypertension',
#                   'obc:Chronic renal disease',
#                   'obc:Cardiac disease, preexisting',
#                   'obc:HIV/AIDS',
#                   'obc:Preeclampsia with severe features',
#                   'obc:Placental abruption',
#                   'obc:Bleeding disorder, preexisting',
#                   'obc:Anemia, preexisting',
#                   'obc:Twin/multiple pregnancy',
#                   'obc:Preterm birth (< 37 weeks)',
#                   'obc:Placenta previa, complete or partial',
#                   'obc:Neuromuscular disease',
#                   'obc:Asthma, acute or moderate/severe',
#                   'obc:Preeclampsia without severe features or gestational hypertension',
#                   'obc:Connective tissue or autoimmune disease',
#                   'obc:Uterine fibroids',
#                   'obc:Substance use disorder',
#                   'obc:Gastrointestinal disease',
#                   'obc:Chronic hypertension',
#                   'obc:Major mental health disorder',
#                   'obc:Preexisting diabetes mellitus',
#                   'obc:Thyrotoxicosis',
#                   'obc:Previous cesarean birth',
#                   'obc:Gestational diabetes mellitus',
#                   'obc:Delivery BMI\xa0>\xa040'] + [
#                      'CCI:Myocardial Infarction', 'CCI:Congestive Heart Failure', 'CCI:Periphral Vascular Disease',
#                      'CCI:Cerebrovascular Disease', 'CCI:Dementia', 'CCI:Chronic Pulmonary Disease',
#                      'CCI:Connective Tissue Disease-Rheumatic Disease', 'CCI:Peptic Ulcer Disease',
#                      'CCI:Mild Liver Disease',
#                      'CCI:Diabetes without complications', 'CCI:Diabetes with complications',
#                      'CCI:Paraplegia and Hemiplegia',
#                      'CCI:Renal Disease', 'CCI:Cancer', 'CCI:Moderate or Severe Liver Disease',
#                      'CCI:Metastatic Carcinoma',
#                      'CCI:AIDS/HIV',
#                  ] + [
#                      'addPaxRisk:Drug Abuse', 'addPaxRisk:Obesity', 'addPaxRisk:tuberculosis',
#                  ] + [
#                      'mental-base@Schizophrenia Spectrum and Other Psychotic Disorders',
#                      'mental-base@Depressive Disorders',
#                      'mental-base@Bipolar and Related Disorders',
#                      'mental-base@Anxiety Disorders',
#                      'mental-base@Obsessive-Compulsive and Related Disorders',
#                      'mental-base@Post-traumatic stress disorder',
#                      'mental-base@Bulimia nervosa',
#                      'mental-base@Binge eating disorder',
#                      'mental-base@premature ejaculation',
#                      'mental-base@Autism spectrum disorder',
#                      'mental-base@Premenstrual dysphoric disorder',
#                      'mental-base@SMI',
#                      'mental-base@non-SMI',
#                      'other-treat--1095-0-flag',
#                      'SSRI-Indication-dsm-flag',
#                      'SSRI-Indication-dsmAndExlix-flag'
#                  ])
#
#     col_names_out = (['PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
#                       'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
#                       'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
#                       'PaxRisk:Heart conditions', 'PaxRisk:Hypertension', 'PaxRisk:HIV infection',
#                       'PaxRisk:Immunocompromised condition or weakened immune system',
#                       'PaxRisk:Mental health conditions',
#                       'PaxRisk:Overweight and obesity', 'PaxRisk:Pregnancy',
#                       'PaxRisk:Sickle cell disease or thalassemia',
#                       'PaxRisk:Smoking current', 'PaxRisk:Stroke or cerebrovascular disease',
#                       'PaxRisk:Substance use disorders', 'PaxRisk:Tuberculosis', 'PaxExclude:liver',
#                       'PaxExclude:end-stage kidney disease'] +
#                      ["Alcohol Abuse", "Anemia", "Arrythmia", "Asthma",
#                       "Cancer",
#                       "Chronic Kidney Disease", "Chronic Pulmonary Disorders",
#                       "Cirrhosis",
#                       "Coagulopathy", "Congestive Heart Failure",
#                       "COPD", "Coronary Artery Disease", "Dementia",
#                       "Diabetes Type 1",
#                       "Diabetes Type 2",  # 'Type 1 or 2 Diabetes Diagnosis',
#                       "End Stage Renal Disease on Dialysis", "Hemiplegia",
#                       "HIV", "Hypertension",
#                       "Hypertension and Type 1 or 2 Diabetes Diagnosis",
#                       "Inflammatory Bowel Disorder",
#                       "Lupus or Systemic Lupus Erythematosus",
#                       "Mental Health Disorders", "Multiple Sclerosis",
#                       "Parkinson's Disease",
#                       "Peripheral vascular disorders ", "Pregnant",
#                       "Pulmonary Circulation Disorder",
#                       "Rheumatoid Arthritis", "Seizure/Epilepsy",
#                       "Severe Obesity  (BMI>=40 kg/m2)", "Weight Loss",
#                       "Down's Syndrome", 'Other Substance Abuse',
#                       'Cystic Fibrosis',
#                       'Autism', 'Sickle Cell',
#                       'Obstructive sleep apnea',  # added 2022-05-25
#                       'Epstein-Barr and Infectious Mononucleosis (Mono)',
#                       # added 2022-05-25
#                       'Herpes Zoster',  # added 2022-05-25
#                       "Prescription of Corticosteroids",
#                       "Prescription of Immunosuppressant drug"
#                       ] +
#                      ['obc:Placenta accreta spectrum',
#                       'obc:Pulmonary hypertension',
#                       'obc:Chronic renal disease',
#                       'obc:Cardiac disease, preexisting',
#                       'obc:HIV/AIDS',
#                       'obc:Preeclampsia with severe features',
#                       'obc:Placental abruption',
#                       'obc:Bleeding disorder, preexisting',
#                       'obc:Anemia, preexisting',
#                       'obc:Twin/multiple pregnancy',
#                       'obc:Preterm birth (< 37 weeks)',
#                       'obc:Placenta previa, complete or partial',
#                       'obc:Neuromuscular disease',
#                       'obc:Asthma, acute or moderate/severe',
#                       'obc:Preeclampsia without severe features or gestational hypertension',
#                       'obc:Connective tissue or autoimmune disease',
#                       'obc:Uterine fibroids',
#                       'obc:Substance use disorder',
#                       'obc:Gastrointestinal disease',
#                       'obc:Chronic hypertension',
#                       'obc:Major mental health disorder',
#                       'obc:Preexisting diabetes mellitus',
#                       'obc:Thyrotoxicosis',
#                       'obc:Previous cesarean birth',
#                       'obc:Gestational diabetes mellitus',
#                       'obc:Delivery BMI>40'] +
#                      [
#                          'CCI:Myocardial Infarction', 'CCI:Congestive Heart Failure', 'CCI:Periphral Vascular Disease',
#                          'CCI:Cerebrovascular Disease', 'CCI:Dementia', 'CCI:Chronic Pulmonary Disease',
#                          'CCI:Connective Tissue Disease-Rheumatic Disease', 'CCI:Peptic Ulcer Disease',
#                          'CCI:Mild Liver Disease',
#                          'CCI:Diabetes without complications', 'CCI:Diabetes with complications',
#                          'CCI:Paraplegia and Hemiplegia',
#                          'CCI:Renal Disease', 'CCI:Cancer', 'CCI:Moderate or Severe Liver Disease',
#                          'CCI:Metastatic Carcinoma',
#                          'CCI:AIDS/HIV',
#                      ] + [
#                          'addPaxRisk:Drug Abuse', 'addPaxRisk:Obesity', 'addPaxRisk:tuberculosis',
#                      ] +
#                      [
#                          'Schizophrenia Spectrum and Other Psychotic Disorders',
#                          'Depressive Disorders',
#                          'Bipolar and Related Disorders',
#                          'Anxiety Disorders',
#                          'Obsessive-Compulsive and Related Disorders',
#                          'Post-traumatic stress disorder',
#                          'Bulimia nervosa',
#                          'Binge eating disorder',
#                          'premature ejaculation',
#                          'Autism spectrum disorder',
#                          'Premenstrual dysphoric disorder',
#                          'SMI',
#                          'non-SMI',
#                          'bupropion--1095-0',
#                          'SSRI-Indication-dsm-flag',
#                          'SSRI-Indication-dsmAndExlix-flag'
#                      ]
#                      )
#
#     row_names.extend(col_names_out)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#     #
#     # col_names = ['score_cci_charlson', 'score_cci_quan']
#     # col_names_out = ['score_cci_charlson', 'score_cci_quan']
#     # row_names.extend(col_names_out)
#     # records.extend(
#     #     [[_quantile_str(df[c]), _quantile_str(df_pos[c]), _quantile_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#     #      for c in col_names])
#     row_names.append('CCI Score — no. (%)')
#     records.append([])
#
#     col_names = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
#     col_names_out = ['cci_quan:0', 'cci_quan:1-2', 'cci_quan:3-4', 'cci_quan:5-10', 'cci_quan:11+']
#     row_names.extend(col_names_out)
#     records.extend(
#         [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
#          for c in col_names])
#
#     df_out = pd.DataFrame(records, columns=output_columns, index=row_names)
#     df_out['SMD'] = df_out['SMD'].astype(float)
#     df_out.to_excel(out_file)
#     print('Dump done ', df_out)
#     return df, df_out


def build_exposure_group_and_table1_less_4_print(exptype='all', debug=False):
    # %% Step 1. Load  Data
    start_time = time.time()
    np.random.seed(0)
    random.seed(0)

    # in_file = 'recover25Q3_covid_pos_naltrexone-withexposure.csv'
    in_file = 'recover25Q3_covid_pos_naltrexone_Painsub-withexposure.csv'

    print('in read: ', in_file)
    if debug:
        nrows = 200000
    else:
        nrows = None
    df = pd.read_csv(in_file,
                     dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'dob',
                                  'flag_delivery_date',
                                  'flag_pregnancy_start_date',
                                  'flag_pregnancy_end_date'
                                  ],
                     nrows=nrows,
                     )
    print('df.shape:', df.shape)
    print('Read Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    """
    Drug captures
        cnsldn_names = [
            'naltrexone', 'LDN_name', 
            'adderall_combo', 'lisdexamfetamine', 'methylphenidate',
            'amphetamine', 'amphetamine_nocombo', 'dextroamphetamine', 'dextroamphetamine_nocombo', 'modafinil',
            'pitolisant', 'solriamfetol', 'armodafinil', 'atomoxetine', 'benzphetamine',
            'azstarys_combo', 'dexmethylphenidate', 'dexmethylphenidate_nocombo', 'diethylpropion', 'methamphetamine',
            'phendimetrazine', 'phentermine', 'caffeine', 'fenfluramine_delet', 'oxybate_delet',
            'doxapram_delet', 'guanfacine'
            ]
    Drug in different time windows
        for x in cnsldn_names:
            print('cnsldn-treat-0-30@'+x, df['cnsldn-treat-0-30@' + x].sum())
            print('cnsldn-treat--30-30@'+x, df['cnsldn-treat--30-30@'+x].sum())
            print('cnsldn-treat-0-15@'+x, df['cnsldn-treat-0-15@'+x].sum())
            print('cnsldn-treat-0-5@'+x, df['cnsldn-treat-0-5@'+x].sum())
            print('cnsldn-treat-0-7@'+x, df['cnsldn-treat-0-7@'+x].sum())
            print('cnsldn-treat--15-15@'+x, df['cnsldn-treat--15-15@'+x].sum())

            print('cnsldn-treat--30-0@'+x, df['cnsldn-treat--30-0@'+x].sum())
            print('cnsldn-treat--60-0@'+x, df['cnsldn-treat--60-0@'+x].sum())
            print('cnsldn-treat--90-0@'+x, df['cnsldn-treat--90-0@'+x].sum())
            print('cnsldn-treat--120-0@'+x, df['cnsldn-treat--120-0@'+x].sum())
            print('cnsldn-treat--180-0@'+x, df['cnsldn-treat--180-0@'+x].sum())

            print('cnsldn-treat--120-120@'+x, df['cnsldn-treat--120-120@'+x].sum())
            print('cnsldn-treat--180-180@'+x, df['cnsldn-treat--180-180@'+x].sum())

            print('cnsldn-treat--365-0@'+x, df['cnsldn-treat--365-0@'+x].sum())
            print('cnsldn-treat--1095-0@'+x, df['cnsldn-treat--1095-0@'+x].sum())
            print('cnsldn-treat-30-180@'+x, df['cnsldn-treat-30-180@'+x].sum())
            print('cnsldn-treat--1095-30@'+x, df['cnsldn-treat--1095-30@'+x].sum())

    """
    # to define exposure strategies below:
    if exptype == 'naltrexone-acuteIncident-0-30':
        df1 = df.loc[(df['cnsldn-treat-0-30@naltrexone'] >= 1) &
                     (df['cnsldn-treat--1095-0@naltrexone'] == 0), :]

        df0 = df.loc[(df['cnsldn-treat--1095-30@naltrexone'] == 0), :]

        case_label = 'naltrexone 0 to 30 Incident'
        ctrl_label = 'no naltrexone'

    else:
        raise ValueError

    print('exposre strategy, exptype:', exptype)

    n1 = len(df1)
    n0 = len(df0)
    print('n1', n1, 'n0', n0)
    df1['treated'] = 1
    df0['treated'] = 0

    df = pd.concat([df1, df0], ignore_index=True)
    df = add_col(df)

    df_pos = df.loc[df['treated'] == 1, :]
    df_neg = df.loc[df['treated'] == 0, :]

    out_file_df = r'./naltrexone_output/Matrix-naltrexone-{}-25Q3-naltrexCovAtDrugOnset-Painsub.csv'.format(exptype)
    out_file = r'./naltrexone_output/Table-naltrexone-{}-25Q3-naltrexCovAtDrugOnset-Painsub.xlsx'.format(exptype)

    output_columns = ['All', case_label, ctrl_label, 'SMD']

    print('treated df_pos.shape', df_pos.shape,
          'control df_neg.shape', df_neg.shape,
          'combined df.shape', df.shape, )

    print('Dump file for updating covs at drug index ', out_file_df)
    df.to_csv(out_file_df, index=False)
    print('Dump file done ', out_file_df)

    # step 2: generate initial table!
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
    drug_col_output = ['adderall_combo', 'lisdexamfetamine', 'methylphenidate', 'dexmethylphenidate']
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

    row_names.append('Acute severity — no. (%)')
    records.append([])
    col_names = ['outpatient', 'inpatient', 'icu', 'inpatienticu']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

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

    col_names = ['BMI: <18.5 under weight', 'BMI: 18.5-<25 normal weight',
                 'BMI: 25-<30 overweight ', 'BMI: >=30 obese ', 'BMI: missing']
    row_names.extend(col_names)
    records.extend(
        [[_percentage_str(df[c]), _percentage_str(df_pos[c]), _percentage_str(df_neg[c]), _smd(df_pos[c], df_neg[c])]
         for c in col_names])

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
            ['dxcovNaltrexone-base@MECFS', 'dxcovNaltrexone-base@Pain',
             'dxcovNaltrexone-base@pain include', 'dxcovNaltrexone-base@pain tbd', 'dxcovNaltrexone-base@pain exclude',
             'dxcovNaltrexone-base@substance use disorder ',
             'dxcovNaltrexone-base@opioid use disorder', 'dxcovNaltrexone-base@Opioid induced constipation',
             'dxcovNaltrexone-base@Obesity', 'dxcovNaltrexone-base@Crohn-Inflamm_Bowel',
             'dxcovNaltrexone-base@fibromyalgia', 'dxcovNaltrexone-base@multiple sclerosis',
             'dxcovNaltrexone-base@POTS',
             'covNaltrexone_med-base@NSAIDs_combined', 'covNaltrexone_med-base@opioid drug',
             'covNaltrexone_med-basedrugonset@NSAIDs_combined',
             'covNaltrexone_med-basedrugonset@opioid drug', ] +
            ['dxcovNaltrexone-basedrugonset@MECFS', 'dxcovNaltrexone-basedrugonset@Pain',
             'dxcovNaltrexone-basedrugonset@substance use disorder ',
             'dxcovNaltrexone-basedrugonset@opioid use disorder',
             'dxcovNaltrexone-basedrugonset@Opioid induced constipation',
             'dxcovNaltrexone-basedrugonset@Obesity',
             'dxcovNaltrexone-basedrugonset@Crohn-Inflamm_Bowel',
             'dxcovNaltrexone-basedrugonset@fibromyalgia',
             'dxcovNaltrexone-basedrugonset@multiple sclerosis',
             'dxcovNaltrexone-basedrugonset@POTS'] +
            ['dxcovCNSLDN-base@ADHD', 'ADHD_before_drug_onset', 'dxcovCNSLDN-base@Narcolepsy', 'dxcovCNSLDN-base@MECFS',
             'dxcovCNSLDN-base@Pain',
             'dxcovCNSLDN-base@alcohol opioid other substance '] +
            ['PaxRisk:Cancer', 'PaxRisk:Chronic kidney disease', 'PaxRisk:Chronic liver disease',
             'PaxRisk:Chronic lung disease', 'PaxRisk:Cystic fibrosis',
             'PaxRisk:Dementia or other neurological conditions', 'PaxRisk:Diabetes', 'PaxRisk:Disabilities',
             'PaxRisk:Heart conditions', 'PaxRisk:Hypertension', 'PaxRisk:HIV infection',
             'PaxRisk:Immunocompromised condition or weakened immune system', 'PaxRisk:Mental health conditions',
             'PaxRisk:Overweight and obesity', 'PaxRisk:Obesity', 'PaxRisk:Pregnancy',
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
             'mental-base@non-SMI', ] + ['cnsldn-treat--1095-0@' + x for x in [
        'naltrexone', 'LDN_name', 'adderall_combo', 'lisdexamfetamine', 'methylphenidate',
        'amphetamine', 'amphetamine_nocombo', 'dextroamphetamine', 'dextroamphetamine_nocombo', 'modafinil',
        'pitolisant', 'solriamfetol', 'armodafinil', 'atomoxetine', 'benzphetamine',
        'azstarys_combo', 'dexmethylphenidate', 'dexmethylphenidate_nocombo', 'diethylpropion', 'methamphetamine',
        'phendimetrazine', 'phentermine', 'caffeine', 'fenfluramine_delet', 'oxybate_delet',
        'doxapram_delet', 'guanfacine']]
    )

    col_names_out = (['MECFS', 'Pain', 'pain include', 'pain tbd', 'pain exclude', 'substance use disorder ', 'opioid use disorder', 'Opioid induced constipation',
                      'Obesity:just code', 'Crohn-Inflamm_Bowel', 'fibromyalgia', 'multiple sclerosis', 'POTS',
                      'NSAIDs_combined@base', 'opioid drug@base',
                      'NSAIDs_combined@basedrugonset',
                      'opioid drug@basedrugonset'
                      ] + ['MECFS@drugonset', 'Pain@drugonset', 'substance use disorder @drugonset',
                           'opioid use disorder@drugonset', 'Opioid induced constipation@drugonset',
                           'Obesity@drugonset', 'Crohn-Inflamm_Bowel@drugonset', 'fibromyalgia@drugonset',
                           'multiple sclerosis@drugonset', 'POTS@drugonset']
                     + ['ADHD', 'ADHD_before_drug_onset', 'Narcolepsy', 'MECFS', 'Pain',
                        'alcohol opioid other substance'] + ['Cancer', 'Chronic kidney disease',
                                                             'Chronic liver disease',
                                                             'Chronic lung disease', 'Cystic fibrosis',
                                                             'Dementia or other neurological conditions', 'Diabetes',
                                                             'Disabilities',
                                                             'Heart conditions', 'Hypertension', 'HIV infection',
                                                             'Immunocompromised condition or weakened immune system',
                                                             'Mental health conditions',
                                                             'Overweight and obesity', 'Obesity', 'Pregnancy',
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
                      'non-SMI', ] + [
                         'naltrexone', 'LDN_name', 'adderall_combo', 'lisdexamfetamine', 'methylphenidate',
                         'amphetamine', 'amphetamine_nocombo', 'dextroamphetamine', 'dextroamphetamine_nocombo',
                         'modafinil',
                         'pitolisant', 'solriamfetol', 'armodafinil', 'atomoxetine', 'benzphetamine',
                         'azstarys_combo', 'dexmethylphenidate', 'dexmethylphenidate_nocombo', 'diethylpropion',
                         'methamphetamine',
                         'phendimetrazine', 'phentermine', 'caffeine', 'fenfluramine_delet', 'oxybate_delet',
                         'doxapram_delet', 'guanfacine']
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

    # 2025-07-15
    df, df_out = build_exposure_group_and_table1_less_4_print(exptype='naltrexone-acuteIncident-0-30',
                                                              debug=False)  # 'ssri-base180-acutevsnot'

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
