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

# 1. preprocess prescribing file, support supply day calculation and imputation
# 2. preprocess med_admin file if necessary, support supply day calculation and imputation
# 3. combine results from both files if necessary (for COL case)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess cohorts')
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='COL', help='input dataset')
    args = parser.parse_args()
    if args.dataset == 'COL':
        args.med_admin_file = r'../data/V15_COVID19/COL/med_admin.sas7bdat'
        args.prescribe_file = r'../data/V15_COVID19/COL/prescribing.sas7bdat'
        args.med_admin_output = r'../data/V15_COVID19/output/med_admin_COL.pkl'
        args.prescribe_output = r'../data/V15_COVID19/output/prescribing_COL.pkl'
        args.output_file = r'../data/V15_COVID19/output/medication_COL.pkl'

        args.patient_list_file = r'../data/V15_COVID19/output/patient_covid_lab_COL.pkl'
    elif args.dataset == 'WCM':
        args.prescribe_file = r'../data/V15_COVID19/WCM/prescribing.sas7bdat'
        args.output_file = r'../data/V15_COVID19/output/medication_WCM.pkl'

        args.patient_list_file = r'../data/V15_COVID19/output/patient_covid_lab_WCM.pkl'

    print('args:', args)
    return args


def read_prescribing(input_file, output_file='', selected_patients={}):
    """
    :param input_file: input prescribing file with std format
    :param out_file: output id_code-list[patid] = [(start_time, rxnorm, days), ...]  sorted by start_time pickle
    :return: id_code-list[patid] = [(start_time, rxnorm, days), ...]  sorted by start_time pickle
    :Notice:
        discard rows with NULL start_date or rxnorm
        1.COL data: e.g:
        df.shape: (3119792, 32)

        2. WCM data: e.g.
        df.shape: (48987841, 32)
    """
    start_time = time.time()
    print('In read_prescribing, input_file:', input_file, 'output_file:', output_file)
    if selected_patients:
        print('using selected_patients, len(selected_patients):', len(selected_patients))

    chunksize = 100000
    sasds = pd.read_sas(input_file,
                        encoding='WINDOWS-1252',
                        chunksize=chunksize,
                        iterator=True)

    id_med = defaultdict(set)
    i = 0
    n_rows = 0
    # dfs = []

    n_no_rxnorm = 0
    n_no_date = 0
    n_no_days_supply = 0

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
            rx_order_date = row['RX_ORDER_DATE']
            rx_start_date = row['RX_START_DATE']
            rx_end_date = row['RX_END_DATE']
            rxnorm = row['RXNORM_CUI']
            rx_days = row['RX_DAYS_SUPPLY']
            raw_rxnorm = row['RAW_RXNORM_CUI']

            # start_date
            if pd.notna(rx_start_date):
                start_date = rx_start_date
            elif pd.notna(rx_order_date):
                start_date = rx_order_date
            else:
                start_date = np.nan
                n_no_date += 1

            # rxrnom
            if pd.isna(rxnorm):
                if pd.notna(raw_rxnorm):
                    rxnorm = raw_rxnorm
                else:
                    n_no_rxnorm += 1
                    rxnorm = np.nan

            # days supply
            if pd.notna(rx_days):
                days = int(float(rx_days))
            elif pd.notna(start_date) and pd.notna(rx_end_date):
                days = (rx_end_date - start_date).days + 1
            else:
                days = -1
                n_no_days_supply += 1

            if pd.isna(rxnorm) or pd.isna(start_date):
                n_discard_row += 1
            else:
                if not selected_patients:
                    id_med[patid].add((start_date, rxnorm, days))
                    n_recorded_row += 1
                else:
                    if patid in selected_patients:
                        id_med[patid].add((start_date, rxnorm, days))
                        n_recorded_row += 1
                    else:
                        n_not_in_list_row += 1

        if i % 10 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('n_rows:', n_rows,
                  'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_no_days_supply:', n_no_days_supply,
                  'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('n_rows:', n_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('n_rows:', n_rows,
          'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_no_days_supply:', n_no_days_supply,
          'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('len(id_med):', len(id_med))
    # dfs = pd.concat(dfs)
    # print('dfs.shape', dfs.shape)

    # sort
    print('sort dx list in id_dx by time')
    for patid, med_list in id_med.items():
        med_list_sorted = sorted(med_list, key=lambda x: x[0])
        id_med[patid] = med_list_sorted

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if output_file:
        utils.check_and_mkdir(output_file)
        pickle.dump(id_med, open(output_file, 'wb'))
        print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_med


def read_med_admin(input_file, output_file='', selected_patients={}):
    """
    :param input_file: input med_admin file with std format
    :param out_file: output id_code-list[patid] = [(start_time, rxnorm, days), ...]  sorted by start_time pickle
    :return: id_code-list[patid] = [(start_time, rxnorm, days), ...]  sorted by start_time pickle
    :Notice:
        discard rows with NULL start_date or rxnorm
        Only applied to COL data, no WCM
        1.COL data: e.g:
        df.shape: (6890724, 32)

        2. WCM data: e.g.
        df.shape: (0, 21)
    """
    start_time = time.time()
    print('In read_med_admin, input_file:', input_file, 'output_file:', output_file)
    if selected_patients:
        print('using selected_patients, len(selected_patients):', len(selected_patients))

    chunksize = 100000
    sasds = pd.read_sas(input_file,
                        encoding='WINDOWS-1252',
                        chunksize=chunksize,
                        iterator=True)
    id_med = defaultdict(set)
    i = 0
    n_rows = 0
    # dfs = []

    n_no_rxnorm = 0
    n_no_date = 0
    n_no_days_supply = 0

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
            rx_start_date = row['MEDADMIN_START_DATE']
            rx_end_date = row['MEDADMIN_STOP_DATE']
            med_type = row['MEDADMIN_TYPE']
            rxnorm = row['MEDADMIN_CODE']
            names = row['RAW_MEDADMIN_MED_NAME']

            # start_date
            if pd.notna(rx_start_date):
                start_date = rx_start_date
            else:
                start_date = np.nan
                n_no_date += 1

            # rxrnom
            if (med_type != 'RX') or pd.isna(rxnorm):
                n_no_rxnorm += 1
                rxnorm = np.nan

            # days supply
            if pd.notna(start_date) and pd.notna(rx_end_date):
                days = (rx_end_date - start_date).days + 1
            else:
                days = -1
                n_no_days_supply += 1

            if pd.isna(rxnorm) or pd.isna(start_date):
                n_discard_row += 1
            else:
                if not selected_patients:
                    id_med[patid].add((start_date, rxnorm, days))
                    n_recorded_row += 1
                else:
                    if patid in selected_patients:
                        id_med[patid].add((start_date, rxnorm, days))
                        n_recorded_row += 1
                    else:
                        n_not_in_list_row += 1

        if i % 10 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('n_rows:', n_rows,
                  'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_no_days_supply:', n_no_days_supply,
                  'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('n_rows:', n_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('n_rows:', n_rows,
          'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_no_days_supply:', n_no_days_supply,
          'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('len(id_med):', len(id_med))

    # dfs = pd.concat(dfs)
    # print('dfs.shape', dfs.shape)
    # sort
    print('sort dx list in id_dx by time')
    for patid, med_list in id_med.items():
        med_list_sorted = sorted(med_list, key=lambda x: x[0])
        id_med[patid] = med_list_sorted

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if output_file:
        utils.check_and_mkdir(output_file)
        pickle.dump(id_med, open(output_file, 'wb'))
        print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_med


def combine_2_id_med(id_med1, id_med2, output_file=''):
    start_time = time.time()
    print('In combine_2_id_med:',
          'len(id_med1)', len(id_med1), 'len(id_med2)', len(id_med2),
          'output_file:', output_file)
    id_med = defaultdict(set)
    for patid, med_list in id_med1.items():
        id_med[patid] = set(med_list)

    for patid, records in id_med2.items():
        if patid not in id_med:
            id_med[patid] = set(records)
        else:
            id_med[patid].update(records)
    print('Combined len(id_med):', len(id_med))

    # sort
    print('sort combined id_med by time')
    for patid, med_list in id_med.items():
        med_list_sorted = sorted(med_list, key=lambda x: x[0])
        id_med[patid] = med_list_sorted

    if output_file:
        utils.check_and_mkdir(output_file)
        pickle.dump(id_med, open(output_file, 'wb'))
        print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_med


if __name__ == '__main__':
    # python pre_medication.py --dataset COL 2>&1 | tee  log/pre_medication_COL.txt
    # python pre_medication.py --dataset WCM 2>&1 | tee  log/pre_medication_WCM.txt
    start_time = time.time()
    args = parse_args()
    with open(args.patient_list_file, 'rb') as f:
        selected_patients = pickle.load(f)
        print('len(selected_patients):', len(selected_patients))

    print('args.dataset:', args.dataset)
    if args.dataset == 'COL':
        print("step 1. prescribe")
        id_med1 = read_prescribing(args.prescribe_file, args.prescribe_output, selected_patients)
        print("step 2. med_admin")
        id_med2 = read_med_admin(args.med_admin_file, args.med_admin_output, selected_patients)
        print("step 3. combine")
        id_med = combine_2_id_med(id_med1, id_med2, args.output_file)
    else:
        print("Here for {}, only using prescribing file".format(args.dataset))
        id_med = read_prescribing(args.prescribe_file, args.output_file, selected_patients)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
