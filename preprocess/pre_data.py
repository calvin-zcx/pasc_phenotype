import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import Counter
from collections import defaultdict
import utils
import functools

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='WCM', help='input dataset')
    args = parser.parse_args()
    if args.dataset == 'COL':
        args.input_file = r'../data/V15_COVID19/output/data_cohorts_COL.pkl'
        args.output_file_covariates = r'../data/V15_COVID19/output/cohorts_covariates_df_COL.csv'
        args.output_file_outcomes = r'../data/V15_COVID19/output/cohorts_outcomes_df_COL.csv'
    elif args.dataset == 'WCM':
        args.input_file = r'../data/V15_COVID19/output/data_cohorts_WCM.pkl'
        args.output_file_covariates = r'../data/V15_COVID19/output/cohorts_covariates_df_WCM.csv'
        args.output_file_outcomes = r'../data/V15_COVID19/output/cohorts_outcomes_df_WCM.csv'

    print('args:', args)
    return args


def _load_mapping():
    print('... loading encoding mapping dictionaries:')
    with open(r'../data/mapping/icd_ccsr_mapping.pkl', 'rb') as f:
        icd_ccsr = pickle.load(f)
        print('Load ICD-10 to CCSR mapping done! len(icd_ccsr):', len(icd_ccsr))
        record_example = next(iter(icd_ccsr.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/ccsr_index_mapping.pkl', 'rb') as f:
        ccsr_encoding = pickle.load(f)
        print('Load CCSR to encoding mapping done! len(ccsr_encoding):', len(ccsr_encoding))
        record_example = next(iter(ccsr_encoding.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/rxnorm_ingredient_mapping_combined.pkl', 'rb') as f:
        rxnorm_ing = pickle.load(f)
        print('Load rxRNOM_CUI to ingredient mapping done! len(rxnorm_atc):', len(rxnorm_ing))
        record_example = next(iter(rxnorm_ing.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/rxnorm_atc_mapping.pkl', 'rb') as f:
        rxnorm_atc = pickle.load(f)
        print('Load rxRNOM_CUI to ATC mapping done! len(rxnorm_atc):', len(rxnorm_atc))
        record_example = next(iter(rxnorm_atc.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/atcL3_index_mapping.pkl', 'rb') as f:
        atcl3_encoding = pickle.load(f)
        print('Load to ATC-Level-3 to encoding mapping done! len(atcl3_encoding):', len(atcl3_encoding))
        record_example = next(iter(atcl3_encoding.items()))
        print('e.g.:', record_example)

    # with open(r'../data/mapping/atc_rxnorm_mapping.pkl', 'rb') as f:
    #     atc_rxnorm = pickle.load(f)
    #     print('Load ATC to rxRNOM_CUI mapping done! len(atc_rxnorm):', len(atc_rxnorm))
    #     record_example = next(iter(atc_rxnorm.items()))
    #     print('e.g.:', record_example)

    return icd_ccsr, ccsr_encoding, rxnorm_ing, rxnorm_atc, atcl3_encoding


def _is_in_baseline(event_time, index_time):
    # baseline: -18 month to -1 month prior to the index date
    return -540 <= (event_time - index_time).days <= -30


def _is_in_followup(event_time, index_time):
    # baseline: 1 month to 5 month after the index date
    return 30 <= (event_time - index_time).days <= 150


def _encoding_age(age):
    # ['age20-29', 'age30-39', 'age40-49', 'age50-59', 'age60-69', 'age70-79', 'age>=80']
    encoding = np.zeros((1, 7), dtype='float')
    if age <= 29:
        encoding[0, 0] = 1
    elif age <= 39:
        encoding[0, 1] = 1
    elif age <= 49:
        encoding[0, 2] = 1
    elif age <= 59:
        encoding[0, 3] = 1
    elif age <= 69:
        encoding[0, 4] = 1
    elif age <= 79:
        encoding[0, 5] = 1
    else:
        encoding[0, 6] = 1
    return encoding


def _encoding_gender(gender):
    gender = gender.upper()
    if gender == 'F':
        return 1
    else:
        return 0


def _encoding_race(race):
    # ['white', 'black', 'asian', 'other', 'unknown']
    encoding = np.zeros((1, 5), dtype='float')
    if race == '05':
        encoding[0, 0] = 1
    elif race == '03':
        encoding[0, 1] = 1
    elif race == '02':
        encoding[0, 2] = 1
    elif race == 'NI' or race == '07' or race == 'UN':
        # NI=No information, 07=Refuse to answer, UN=Unknown
        encoding[0, 3] = 1
    else:
        encoding[0, 4] = 1
    return encoding


def _encoding_hispanic(hispanic):
    # ['not hispanic', 'hispanic', 'unknown']
    encoding = np.zeros((1, 3), dtype='float')
    if hispanic == 'N':
        encoding[0, 0] = 1
    elif hispanic == 'Y':
        encoding[0, 1] = 1
    else:
        encoding[0, 2] = 1
    return encoding


def _encoding_social(nation_adi, impute_value):
    # ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
    #  'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100']
    encoding = np.zeros((1, 10), dtype='float')
    if pd.isna(nation_adi):
        nation_adi = impute_value
    if nation_adi >= 100:
        nation_adi = 99
    if nation_adi < 1:
        nation_adi = 1
    pos = int(nation_adi) // 10
    encoding[0, pos] = 1
    return encoding


def _encoding_utilization(enc_list, index_date):
    # encoding uitlization in the baseline
    # ['inpatient visits', 'outpatient visits', 'emergency visits', 'other visits']
    encoding = np.zeros((1, 4), dtype='float')
    for records in enc_list:
        enc_date, type = records
        if _is_in_baseline(enc_date, index_date):
            if type == 'EI' or type == 'IP':
                encoding[0, 0] += 1
            elif type == 'AV' or type == 'OA' or type == 'TH':
                encoding[0, 1] += 1
            elif type == 'ED':
                encoding[0, 2] += 1
            else:
                encoding[0, 3] += 1

    return encoding


def _encoding_dx(dx_list, icd_ccsr, ccsr_encoding, index_date):
    # encoding 544 ccsr diagnoses codes in the baseline
    encoding = np.zeros((1, len(ccsr_encoding)), dtype='float')
    for records in dx_list:
        dx_date, icd = records[:2]
        if _is_in_baseline(dx_date, index_date):
            icd = icd.replace('.', '')
            if icd in icd_ccsr:
                ccsr = icd_ccsr[icd][0]  # to check out of index value
                pos = ccsr_encoding[ccsr][0]
                encoding[0, pos] += 1
            else:
                print('ERROR:', icd, 'not in icd to ccsr dictionary!')
    return encoding


def _encoding_med(med_list, rxnorm_ing, rxnorm_atc, atcl3_encoding, index_date, atc_level=3):
    # encoding 269 atc level 3 diagnoses codes in the baseline
    # mapping rxnorm_cui to its ingredient(s)
    # for each ingredient, mapping to atc and thus atc[:5] is level three
    # summarize all unique atcL3 codes
    atclevel_chars = {1: 1, 2: 3, 3: 4, 4: 5, 5: 7}
    atc_n_chars = atclevel_chars.get(atc_level, 4)  # default level 3, using first 4 chars
    encoding = np.zeros((1, len(atcl3_encoding)), dtype='float')
    for records in med_list:
        med_date, rxnorm, supply_days = records
        if _is_in_baseline(med_date, index_date):
            if rxnorm in rxnorm_atc:
                atcl3_set = set([x[0][:atc_n_chars] for x in rxnorm_atc[rxnorm]])
                pos_list = [atcl3_encoding[x][0] for x in atcl3_set]
                for pos in pos_list:
                    encoding[0, pos] += 1
            elif rxnorm in rxnorm_ing:
                ing_list = rxnorm_ing[rxnorm]
                atcl3_set = set([])
                for ing in ing_list:
                    if ing in rxnorm_atc:
                        atcl3_set.update([x[0][:atc_n_chars] for x in rxnorm_atc[ing]])

                pos_list = [atcl3_encoding[x][0] for x in atcl3_set]
                for pos in pos_list:
                    encoding[0, pos] += 1
            else:
                print('ERROR:', rxnorm, 'not in rxnorm to atc dictionary or rxnorm-to-ing-to-atc!')
    return encoding


def build_baseline_covariates(args):
    start_time = time.time()
    print('In build_baseline_covariates...')
    # step 1: load encoding dictionary
    icd_ccsr, ccsr_encoding, rxnorm_ing, rxnorm_atc, atcl3_encoding = _load_mapping()

    # step 2: load cohorts pickle data
    print('Load cohorts pickle data file:', args.input_file)
    with open(args.input_file, 'rb') as f:
        id_data = pickle.load(f)
        print('Load covid patients pickle data done! len(id_data):', len(id_data))

    # step 3: encoding cohorts into matrix
    df_records_aux = []  # for double check, and get basic information
    df_records_aux_column = []

    n = len(id_data)
    pid_list = []
    column_names = []

    age_array = np.zeros((n, 7), dtype='float')
    age_column_names = ['age20-29', 'age30-39', 'age40-49', 'age50-59', 'age60-69', 'age70-79', 'age>=80']

    gender_array = np.zeros((n, 1), dtype='float')
    gender_column_names = ['gender-female', ]

    race_array = np.zeros((n, 5), dtype='float')
    race_column_names = ['white', 'black', 'asian', 'other', 'unknown']

    hispanic_array = np.zeros((n, 3), dtype='float')
    hispanic_column_names = ['not hispanic', 'hispanic', 'unknown']

    social_array = np.zeros((n, 10), dtype='float')
    social_column_names = ['ADI1-9', 'ADI10-19', 'ADI20-29', 'ADI30-39', 'ADI40-49',
                           'ADI50-59', 'ADI60-69', 'ADI70-79', 'ADI80-89', 'ADI90-100']

    utilization_array = np.zeros((n, 4), dtype='float')
    utilization_column_names = ['inpatient visits', 'outpatient visits', 'emergency visits', 'other visits']

    dx_array = np.zeros((n, 544), dtype='float')  # ccsr 6-char category
    dx_column_names = list(ccsr_encoding.keys())

    med_array = np.zeros((n, 269), dtype='float')  # atc level 3 category
    med_column_names = list(atcl3_encoding.keys())

    # impute adi value by median of all:
    adi_value_list = [v[1][7] for key, v in id_data.items()]
    adi_value_default = np.nanmedian(adi_value_list)
    # pd.DataFrame(adi_value_list).describe()
    # pd.DataFrame(adi_value_list).hist(bins=20)
    for i, (pid, item) in enumerate(id_data.items()):
        pid_list.append(pid)

        index_info, demo, dx, med, covid_lab, enc = item
        flag, index_date, covid_loinc, flag_name, index_age_year = index_info
        birth_date, gender, race, hispanic, zipcode, state, city, nation_adi, state_adi = demo
        records_aux = []
        records_aux.extend(index_info + demo)

        age_array[i, :] = _encoding_age(index_age_year)
        gender_array[i] = _encoding_gender(gender)
        race_array[i, :] = _encoding_race(race)
        hispanic_array[i, :] = _encoding_hispanic(hispanic)
        social_array[i, :] = _encoding_social(nation_adi, adi_value_default)
        # Only count following covariates in baseline
        utilization_array[i, :] = _encoding_utilization(enc, index_date)
        dx_array[i, :] = _encoding_dx(dx, icd_ccsr, ccsr_encoding, index_date)
        med_array[i, :] = _encoding_med(med, rxnorm_ing, rxnorm_atc, atcl3_encoding, index_date)

        df_records_aux.append(records_aux)

    #   step 4: build pandas, column, and dump
    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    pass


def build_outcomes_in_followup(args):
    pass


if __name__ == '__main__':
    # python pre_cohort.py --dataset COL 2>&1 | tee  log/pre_cohort_combine_and_EC_COL.txt
    # python pre_cohort.py --dataset WCM 2>&1 | tee  log/pre_cohort_combine_and_EC_WCM.txt
    start_time = time.time()
    args = parse_args()
    baseline_covariates_df = build_baseline_covariates(args)
    outcomes_df = build_baseline_covariates(args)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
