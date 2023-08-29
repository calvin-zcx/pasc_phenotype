import sys
# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import time
import pickle
import joblib
import argparse
from misc import utils
import functools
from tqdm import tqdm
from collections import Counter

print = functools.partial(print, flush=True)
from collections import defaultdict
from misc.utilsql import *


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess diagnosis')
    parser.add_argument('--dataset', default='wcm', help='site dataset')
    args = parser.parse_args()

    args.input_file = r'../data/recover/output_hf/{}/diagnosis_anyHF_{}.csv'.format(args.dataset, args.dataset)
    args.demo_file = r'../data/recover/output_hf/{}/patient_demo_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file = r'../data/recover/output_hf/{}/patient_hf_list_{}.pkl'.format(args.dataset, args.dataset)

    print('args:', args)
    return args


def read_hr_diagnosis_build_patlist(args, id_demo):
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
    print('Choose dataset:', args.dataset, )

    # print('in heart_failure_dx.xlsx')
    # # step: load covid lab test codes, may be updated by:
    # print('Step 1: load heart failure (comprehensive) dx codes')
    # df_hf = pd.read_excel(r'../data/mapping/heart_failure_dx.xlsx', sheet_name='dx', dtype=str)
    #
    # print('df_hf.shape:', df_hf.shape)
    # code_set = set(df_hf['dx code'].to_list())
    # print('Selected all heart failure related codes: ', code_set)
    # print('len(code_set):', len(code_set))

    # step: read selected HR dx to build patient list and basic index date
    print('Step 2, read HF dx data, and select patients who had any HR diagnoses!')
    df = pd.read_csv(args.input_file, dtype=str, parse_dates=['ADMIT_DATE', "DX_DATE"])
    print('df.shape', df.shape, 'df.columns', df.columns)
    print('No rows of dx', df.shape[0], 'Unique HF patid:', len(df['PATID'].unique()))
    print('Time range of All HF patients:', df["ADMIT_DATE"].describe(datetime_is_numeric=True))

    id_dx = defaultdict(list)
    i = 0
    n_rows = 0
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0
    n_no_dob_row = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        i += 1
        patid = row['PATID']
        enc_id = row['ENCOUNTERID']
        enc_type = row['ENC_TYPE']
        dx = row['DX']
        dx_type = row["DX_TYPE"]
        # dx_date may be null. no imputation. If there is no date, not recording
        dx_date = row["ADMIT_DATE"]

        if pd.isna(dx):
            n_no_dx += 1
        if pd.isna(dx_date):
            n_no_date += 1

        if pd.isna(dx) or pd.isna(dx_date):
            n_discard_row += 1
        else:
            if patid in id_demo:
                # updated 2022-10-26.
                if pd.notna(id_demo[patid][0]):
                    # (lab_date - id_demo[patid][0]).days // 365
                    age = (dx_date - pd.to_datetime(id_demo[patid][0])).days / 365
                else:
                    age = np.nan
            else:
                print('No age information for:', patid, )
                age = np.nan
                n_no_dob_row += 1

            id_dx[patid].append((dx_date, dx, dx_type, enc_type, age, enc_id))
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_no_dob_row:', n_no_dob_row, 'len(id_dx):', len(id_dx))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # sort and de-duplicates
    print('sort dx list in id_dx by time')
    for patid, dx_list in id_dx.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        dx_list_sorted = sorted(set(dx_list), key=lambda x: x[0])
        id_dx[patid] = dx_list_sorted
    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if args.output_file:
        print('Dump file:', args.output_file)
        utils.check_and_mkdir(args.output_file)
        pickle.dump(id_dx, open(args.output_file, 'wb'))
        print('dump done to {}'.format(args.output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_dx, df


if __name__ == '__main__':
    # python pre_hf_dx.py --dataset wcm 2>&1 | tee  log/pre_hf_dx_wcm.txt

    start_time = time.time()
    args = parse_args()
    print('Selected site:', args.dataset)
    print('args:', args)

    with open(args.demo_file, 'rb') as f:
        # to add age information for each covid tests
        id_demo = pickle.load(f)
        print('load', args.demo_file, 'demo information: len(id_demo):', len(id_demo))

    id_dx, df = read_hr_diagnosis_build_patlist(args, id_demo)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
