import sys
# for linux env.
sys.path.insert(0,'..')
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm
import time
import pickle
import argparse
import csv
import utils
import numpy as np
import functools
from collections import Counter
print = functools.partial(print, flush=True)
from collections import defaultdict

# 1. aggregate preprocessed data [covid lab, demo, diagnosis, medication] into cohorts data structure
# 2. applying EC [eligibility criteria] to include and exclude patients for each cohorts
# 3. summarize basic statistics when applying each EC criterion


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess cohorts')
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='COL', help='input dataset')
    args = parser.parse_args()
    if args.dataset == 'COL':
        args.covid_lab_file = r'../data/V15_COVID19/output/patient_covid_lab_COL.pkl'
        args.demo_file = r'../data/V15_COVID19/output/patient_demo_COL.pkl'
        args.dx_file = r'../data/V15_COVID19/output/diagnosis_COL.pkl'
        args.med_file = r'../data/V15_COVID19/output/medication_COL.pkl'
        args.output_file = r'../data/V15_COVID19/output/covid_cohorts_COL.pkl'
    elif args.dataset == 'WCM':
        args.covid_lab_file = r'../data/V15_COVID19/output/patient_covid_lab_WCM.pkl'
        args.demo_file = r'../data/V15_COVID19/output/patient_demo_WCM.pkl'
        args.dx_file = r'../data/V15_COVID19/output/diagnosis_WCM.pkl'
        args.med_file = r'../data/V15_COVID19/output/medication_WCM.pkl'
        args.output_file = r'../data/V15_COVID19/output/covid_cohorts_WCM.pkl'

    print('args:', args)
    return args


def read_preprocessed_data(args):
    """
    load pre-processed data, [make potential transformation]
    :param args contains all data file names
    :return: a list of imported pickle files
    :Notice:
        1.COL data: e.g:
        df.shape: (16666999, 19)

        2. WCM data: e.g.
        df.shape: (47319049, 19)

    """
    start_time = time.time()
    # 1. load covid patients lab list
    with open(args.covid_lab_file, 'rb') as f:
        id_lab = pickle.load(f)
        print('Load covid patients lab list done! len(id_lab):', len(id_lab))

    # 2. load demographics file
    with open(args.demo_file, 'rb') as f:
        id_demo = pickle.load(f)
        print('load demographics file done! len(id_demo):', len(id_demo))

    # 3. load diagnosis file
    with open(args.dx_file, 'rb') as f:
        id_dx = pickle.load(f)
        print('load diagnosis file done! len(id_dx):', len(id_dx))

    # # 4. load medication file
    # with open(args.med_file, 'rb') as f:
    #     id_med = pickle.load(f)
    #     print('load medication file done! len(id_med):', len(id_med))

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_lab, id_demo, id_dx


def integrate_preprocessed_data(args):
    data = read_preprocessed_data(args)
    print('len(data):', len(data))
    pass


if __name__ == '__main__':
    # python pre_cohort.py --dataset COL 2>&1 | tee  log/pre_cohort_combine_and_EC.txt
    # python pre_cohort.py --dataset WCM 2>&1 | tee  log/pre_cohort_combine_and_EC.txt
    start_time = time.time()
    args = parse_args()

    integrate_preprocessed_data(args)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
