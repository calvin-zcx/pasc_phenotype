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

print = functools.partial(print, flush=True)
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess diagnosis')
    parser.add_argument('--dataset', choices=['all'], default='all', help='combined dataset')
    args = parser.parse_args()

    args.input_file = r'../data/oneflorida/{}/IMMUNIZATION.csv'.format(args.dataset)
    args.patient_list_file = r'../data/oneflorida/output/{}/patient_covid_lab_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file = r'../data/oneflorida/output/{}/immunization_{}.pkl'.format(args.dataset, args.dataset)

    print('args:', args)
    return args


def read_immunization(input_file, output_file='', selected_patients={}):
    """
    :param data_file: input immunization file with std format
    :param out_file: output id_code-list[patid] = [(time, code, codetype,  encid), ...] pickle sorted by time
    :return: id_code-list[patid] = [(time, code, codetype, encid), ...]  sorted by time
    :Notice:
        discard rows with NULL admit_date or dx

        1.COL data: e.g:
        df.shape: (211434, 15)

        2. WCM data: e.g.
        df.shape: None
    """
    start_time = time.time()
    chunksize = 100000
    sasds = pd.read_csv(input_file,
                        dtype=str,
                        parse_dates=['VX_RECORD_DATE', 'VX_ADMIN_DATE', 'VX_EXP_DATE'],
                        encoding='WINDOWS-1252',
                        chunksize=chunksize,
                        iterator=True)
    if selected_patients:
        print('using selected_patients, len(selected_patients):', len(selected_patients))

    id_px = defaultdict(list)
    i = 0
    n_rows = 0
    dfs = []
    n_no_px = 0
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
            px = row['VX_CODE']
            px_type = row["VX_CODE_TYPE"]
            px_date = row["VX_RECORD_DATE"]

            if pd.isna(px):
                n_no_px += 1

            if pd.isna(px_date):
                n_no_date += 1

            if pd.isna(px) or pd.isna(px_date):
                n_discard_row += 1
            else:
                if not selected_patients:
                    id_px[patid].append((px_date, px, px_type, enc_id))
                    n_recorded_row += 1
                else:
                    if patid in selected_patients:
                        id_px[patid].append((px_date, px, px_type, enc_id))
                        n_recorded_row += 1
                    else:
                        n_not_in_list_row += 1

        # # monte case, too large, error. other sites ok
        # dfs.append(chunk[['PATID', 'ENCOUNTERID', 'ENC_TYPE', "ADMIT_DATE", 'DX', "DX_TYPE"]])
        dfs.append(chunk[["VX_RECORD_DATE"]])

        if i % 10 == 0:
            print('chunk:', i, 'len(dfs):', len(dfs),
                  'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('n_rows:', n_rows, 'n_no_dx:', n_no_px, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
                  'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('n_rows:', n_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('n_no_dx:', n_no_px, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('len(id_px):', len(id_px))

    # sort and de-duplicates
    print('sort px list in id_px by time')
    for patid, px_list in id_px.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        px_list_sorted = sorted(set(px_list), key=lambda x: x[0])
        id_px[patid] = px_list_sorted
    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if output_file:
        print('Dump id_px to {}'.format(output_file))
        utils.check_and_mkdir(output_file)
        utils.dump(id_px, output_file, chunk=4)

    if dfs:
        dfs = pd.concat(dfs)
        print('dfs.shape', dfs.shape)
        print('Time range of diagnosis table of selected patients:',
              dfs["VX_RECORD_DATE"].describe(datetime_is_numeric=True))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_px, dfs


if __name__ == '__main__':
    # python pre_immun.py --dataset all 2>&1 | tee  log/pre_immun_all.txt

    start_time = time.time()
    args = parse_args()
    print('Selected site:', args.dataset)
    with open(args.patient_list_file, 'rb') as f:
        selected_patients = pickle.load(f)
        print('len(selected_patients):', len(selected_patients))

    id_px, df = read_immunization(args.input_file, args.output_file, selected_patients)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
