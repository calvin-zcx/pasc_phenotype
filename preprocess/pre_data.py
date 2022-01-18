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
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='COL', help='input dataset')
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


def encode_data(id_data):
    # select info in baseline windows for baseline covariates
    # select info in follow-up windows for outcomes
    # build encoding dimension
    # fill encoding information for baseline and follow-up
    # selection or others may be later

    print('In encode_date, len(id_data):', len(id_data))

    pid_list = []
    demo_list = []

    dx_baseline_list = []
    med_baseline_list = []
    lab_baseline_list = []
    encounter_baseline_list = []

    dx_followup_list = []
    med_followup_list = []
    lab_followup_list = []
    encounter_followup_list = []

    column_names = []

    for pid, item in id_data.items():
        pid_list.append(pid)
        index_info, demo, dx, med, lab, enc = item
        demo_list.append(index_info + demo)
        dx_list.append(dx)
        med_list.append(med)
        lab_list.append(lab)
        encounter_list.append(enc)


def is_in_baseline(event_time, index_time):


def build_baseline_covariates(args):
    start_time = time.time()
    print('In build_data_frame...')
    # %% step 1: load encoding dictionary
    with open(r'../data/mapping/icd_ccsr_mapping.pkl', 'rb') as f:
        icd_ccsr = pickle.load(f)
        print('Load ICD-10 to CCSR mapping done! len(icd_ccsr):', len(icd_ccsr))
        record_example = next(iter(icd_ccsr.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/rxnorm_atc_mapping.pkl', 'rb') as f:
        rxnorm_atc = pickle.load(f)
        print('Load rxRNOM_CUI to ATC mapping done! len(rxnorm_atc):', len(rxnorm_atc))
        record_example = next(iter(rxnorm_atc.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/atc_rxnorm_mapping.pkl', 'rb') as f:
        atc_rxnorm = pickle.load(f)
        print('Load ATC to rxRNOM_CUI mapping done! len(atc_rxnorm):', len(atc_rxnorm))
        record_example = next(iter(atc_rxnorm.items()))
        print('e.g.:', record_example)

    # %%  step 2: load cohorts pickle data
    print('Load cohorts pickle data file:', args.input_file)
    with open(args.input_file, 'rb') as f:
        id_data = pickle.load(f)
        print('Load covid patients pickle data done! len(id_data):', len(id_data))

    # %%  step 3: encoding cohorts into matrix
    df_data = encode_data(id_data)
    # %%  step 4: build pandas, column, and dump
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
