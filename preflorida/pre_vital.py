import sys
# for linux env.
sys.path.insert(0,'..')
import pandas as pd
import time
import pickle
import joblib
import argparse
from misc import utils
import functools
from tqdm import tqdm
import datetime
import numpy as np

print = functools.partial(print, flush=True)
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess vital')
    parser.add_argument('--dataset', choices=['all'], default='all', help='combined dataset')
    args = parser.parse_args()

    args.input_file = r'../data/oneflorida/{}/VITAL.csv'.format(args.dataset)
    args.patient_list_file = r'../data/oneflorida/output/{}/patient_covid_lab_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file = r'../data/oneflorida/output/{}/vital_{}.pkl'.format(args.dataset, args.dataset)
    print('args:', args)
    return args


def _replace_nan_to_nullstring(alist):
    return tuple([x if pd.notna(x) else '' for x in alist])


def _replace_nullstring_to_nan(alist):
    return [np.nan if x == '' else x for x in alist]


def _to_float(num):
    try:
        return float(num)
    except ValueError:
        return np.nan


def read_vital(input_file, output_file='', selected_patients={}):
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
    sasds = pd.read_csv(input_file,
                        dtype=str,
                        parse_dates=['MEASURE_DATE'],
                        chunksize=chunksize,
                        iterator=True)  # encoding='utf-8',

    if selected_patients:
        print('using selected_patients, len(selected_patients):', len(selected_patients))

    id_vital = defaultdict(list)
    i = 0
    n_rows = 0
    dfs = []
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0
    n_not_in_list_row = 0
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
            dx_date = row["MEASURE_DATE"]
            ht = _to_float(row['HT'])
            wt = _to_float(row['WT'])
            ori_bmi = _to_float(row['ORIGINAL_BMI'])
            smoking = row['SMOKING']
            tobacco = row['TOBACCO']

            if pd.isna(dx_date):
                n_no_date += 1
            if isinstance(dx_date, str):
                dx_date = utils.str_to_datetime(dx_date)

            if pd.isna(dx_date) or (pd.isnull([ht, wt, ori_bmi, smoking, tobacco]).all()):
                n_discard_row += 1
            else:
                if not selected_patients:
                    id_vital[patid].append(_replace_nan_to_nullstring([dx_date, ht, wt, ori_bmi, smoking, tobacco]))
                    n_recorded_row += 1
                else:
                    if patid in selected_patients:
                        id_vital[patid].append(_replace_nan_to_nullstring([dx_date, ht, wt, ori_bmi, smoking, tobacco]))
                        n_recorded_row += 1
                    else:
                        n_not_in_list_row += 1

        # # monte case, too large, error. other sites ok
        # dfs.append(chunk[['PATID', 'ENCOUNTERID', 'ENC_TYPE', "ADMIT_DATE", 'DX', "DX_TYPE"]])
        dfs.append(chunk[["MEASURE_DATE"]])

        if i % 10 == 0:
            print('chunk:', i, 'len(dfs):', len(dfs),
                  'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('n_rows:', n_rows, 'n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
                  'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('n_rows:', n_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('len(id_vital):', len(id_vital))
    dfs = pd.concat(dfs)
    print('dfs.shape', dfs.shape)
    print('Time range of diagnosis table of selected patients:', dfs["MEASURE_DATE"].describe(datetime_is_numeric=True))

    # sort and de-duplicates
    print('sort dx list in id_vital by time')
    for patid, dx_list in id_vital.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        dx_list_sorted_ = sorted(set(dx_list), key=lambda x: x[0])
        dx_list_sorted = [_replace_nullstring_to_nan(x) for x in dx_list_sorted_]
        id_vital[patid] = dx_list_sorted
    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if output_file:
        print('Dump id_vital to {}'.format(output_file))
        utils.check_and_mkdir(output_file)
        utils.dump(id_vital, output_file, chunk=4)

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_vital, dfs


if __name__ == '__main__':
    # python pre_diagnosis.py --dataset all 2>&1 | tee  log/pre_diagnosis_all.txt

    start_time = time.time()
    args = parse_args()
    print('Selected site:', args.dataset)
    with open(args.patient_list_file, 'rb') as f:
        selected_patients = pickle.load(f)
        print('len(selected_patients):', len(selected_patients))

    id_vital, df = read_vital(args.input_file, args.output_file, selected_patients)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
