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
import seaborn as sns
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='COL', help='input dataset')
    args = parser.parse_args()
    if args.dataset == 'COL':
        args.input_file = r'../data/V15_COVID19/output/covid_lab_COL.xlsx'
        args.output_file = r'../data/V15_COVID19/output/patient_covid_lab_COL.pkl'
    elif args.dataset == 'WCM':
        args.input_file = r'../data/V15_COVID19/output/covid_lab_WCM.xlsx'
        args.output_file = r'../data/V15_COVID19/output/patient_covid_lab_WCM.pkl'

    print('args:', args)
    return args


def read_covid_lab_and_generate_label(input_file, output_file=''):
    """
    :param data_file: input demographics file with std format
    :param out_file: output id_covidlab[patid] = [(time, code, label), ...] sorted by time,  pickle
    :return: id_covidlab[patid] = [(time, code, label), ...] sorted by time
    :Notice:
        1.COL data: e.g:
        df.shape: (372633, 36)

        2. WCM data: e.g.
        df.shape:

    """
    start_time = time.time()
    df = pd.read_excel(input_file, sheet_name='Sheet1')
    df.rename(columns=lambda x: x.upper(), inplace=True)
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    # df_sub = df[['PATID', 'RESULT_DATE', 'LAB_LOINC', 'RESULT_QUAL']]  # use specimen_date?
    # records_list = df_sub.values.tolist()
    id_lab = defaultdict(list)  # {x[0]: x[1:] for x in records_list}
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0

    for index, row in df.iterrows():
        patid = row['PATID']
        lab_date = row["RESULT_DATE"]  # dx_date may be null. no imputation. If there is no date, not recording
        lab_code = row['LAB_LOINC']
        result_label = row['RESULT_QUAL']

        if pd.isna(lab_code):
            n_no_dx += 1
        if pd.isna(lab_date):
            n_no_date += 1

        if pd.isna(lab_code) or pd.isna(lab_date):
            n_discard_row += 1
        else:
            id_lab[patid].append((lab_date, lab_code, result_label))
            n_recorded_row += 1

    print('n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row)
    # sort
    print('sort dx list in id_dx by time')
    for patid, lab_list in id_lab.items():
        lab_list_sorted = sorted(lab_list, key=lambda x: x[0])
        id_lab[patid] = lab_list_sorted
    print('len(id_lab):', len(id_lab))

    # add more information to df
    df = df.sort_values(by=['PATID', 'RESULT_DATE'])
    df['n_test'] = df['PATID'].apply(lambda x: len(id_lab[x]))
    df['covid_positive'] = df['PATID'].apply(lambda x: {'POSITIVE', 'positive'}.issubset([a[-1] for a in id_lab[x]]))

    if output_file:
        utils.check_and_mkdir(output_file)
        pickle.dump(id_lab, open(output_file, 'wb'))
        df.to_excel(output_file.replace('.pkl', '') + '.xlsx')
        print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_lab, df


def data_analyisi(id_lab):
    cnt = []
    for k, v in id_lab.items():
        cnt.append(len(v))

    c = Counter(cnt)
    df = pd.DataFrame(c.most_common(), columns=['covid_test_count', 'num_of_people'])
    df.to_excel(r'../data/V15_COVID19/output/test_frequency_COL.xlsx')


if __name__ == '__main__':
    # python pre_demo.py --dataset COL 2>&1 | tee  log/pre_demo_COL.txt
    # python pre_demo.py --dataset WCM 2>&1 | tee  log/pre_demo_WCM.txt
    start_time = time.time()
    args = parse_args()
    id_lab, df = read_covid_lab_and_generate_label(args.input_file, args.output_file)
    # patient_dates = build_patient_dates(args.demo_file, args.dx_file, r'output/patient_dates.pkl')
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
