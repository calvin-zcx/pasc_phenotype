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


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess diagnosis')
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='WCM', help='input dataset')
    args = parser.parse_args()
    if args.dataset == 'COL':
        args.input_file = r'../data/V15_COVID19/COL/diagnosis.sas7bdat'
        args.output_file = r'../data/V15_COVID19/output/diagnosis_COL.pkl'
    elif args.dataset == 'WCM':
        args.input_file = r'../data/V15_COVID19/WCM/diagnosis.sas7bdat'
        args.output_file = r'../data/V15_COVID19/output/diagnosis_WCM.pkl'

    print('args:', args)
    return args


def read_diagnosis(input_file, output_file=''):
    """
    :param data_file: input demographics file with std format
    :param out_file: output id_code-list[patid] = [(time, ICD), ...] pickle sorted by time
    :return: id_code-list[patid] = [(time, ICD), ...]  sorted by time
    :Notice:
        discard rows with NULL admit_date or dx

        1.COL data: e.g:
        df.shape: (16666999, 19)

        2. WCM data: e.g.
        df.shape: (47319049, 19)

    """
    start_time = time.time()
    chunksize = 100000
    sasds = pd.read_sas(input_file,
                        encoding='WINDOWS-1252',
                        chunksize=chunksize,
                        iterator=True)
    id_dx = defaultdict(list)
    i = 0
    n_rows = 0
    dfs = []
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0
    for chunk in sasds:  # , meta
        i += 1
        if chunk.empty:
            print("ERROR: Empty chunk! break!")
            break

        n_rows += len(chunk)
        chunk.rename(columns=lambda x: x.upper(), inplace=True)
        if i == 1:
            print('chunk.shape', chunk.shape)
            print('chunk.columns', chunk.columns)

        for index, row in chunk.iterrows():
            patid = row['PATID']
            enc_id = row['ENCOUNTERID']
            enc_type = row['ENC_TYPE']
            dx = row['DX']
            dx_type = row["DX_TYPE"]
            dx_date = row["ADMIT_DATE"]  # dx_date may be null. no imputation. If there is no date, not recording

            if pd.isna(dx):
                n_no_dx += 1
            if pd.isna(dx_date):
                n_no_date += 1

            if pd.isna(dx) or pd.isna(dx_date):
                n_discard_row += 1
            else:
                id_dx[patid].append((dx_date, dx, dx_type, enc_type))
                n_recorded_row += 1

        dfs.append(chunk[['PATID', 'ENCOUNTERID', 'ENC_TYPE', "ADMIT_DATE", 'DX', "DX_TYPE"]])

        if i % 10 == 0:
            print('chunk:', i, 'len(dfs):', len(dfs),
                  'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('n_rows:', n_rows, 'n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row)

    print('n_rows:', n_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row)

    print('len(id_dx):', len(id_dx))
    dfs = pd.concat(dfs)
    print('dfs.shape', dfs.shape)
    # sort
    print('sort dx list in id_dx by time')
    for patid, dx_list in id_dx.items():
        dx_list_sorted = sorted(dx_list, key = lambda x: x[0])
        id_dx[patid] = dx_list_sorted

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if output_file:
        utils.check_and_mkdir(output_file)
        pickle.dump(id_dx, open(output_file, 'wb'))
        dfs.to_csv(output_file.replace('.pkl', '') + '.csv')
        print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_dx, dfs


if __name__ == '__main__':
    # python pre_diagnosis.py --dataset COL 2>&1 | tee  log/pre_diagnosis_COL.txt
    # python pre_diagnosis.py --dataset WCM 2>&1 | tee  log/pre_diagnosis_WCM.txt
    start_time = time.time()
    args = parse_args()
    id_dx, df = read_diagnosis(args.input_file, args.output_file)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
