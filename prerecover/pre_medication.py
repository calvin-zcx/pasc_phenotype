import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import time
import pickle
import argparse
from misc import utils
from misc.utils import clean_date_str
import numpy as np
import functools
from misc.utilsql import *

print = functools.partial(print, flush=True)
from collections import defaultdict
from datetime import datetime, date

# 1. preprocess prescribing file, support supply day calculation and imputation
# 2. preprocess med_admin file if necessary, support supply day calculation and imputation
# 3. combine results from both files if necessary (for COL case)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess medication table')
    parser.add_argument('--dataset', default='ochsner', help='site dataset')
    args = parser.parse_args()

    args.patient_list_file = r'../data/recover/output/{}/patient_covid_lab_{}.pkl'.format(args.dataset, args.dataset)
    args.med_admin_file = r'{}.med_admin'.format(args.dataset)
    args.prescribe_file = r'{}.prescribing'.format(args.dataset)

    args.med_admin_output = r'../data/recover/output/{}/_med_admin_{}.pkl'.format(args.dataset, args.dataset)
    args.prescribe_output = r'../data/recover/output/{}/_prescribing_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file = r'../data/recover/output/{}/medication_{}.pkl'.format(args.dataset, args.dataset)

    print('args:', args)
    return args



def read_prescribing(input_file, output_file='', selected_patients={}):
    """
    :param selected_patients:
    :param output_file:
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
    # sasds = pd.read_sas(input_file,
    #                     encoding='WINDOWS-1252',
    #                     chunksize=chunksize,
    #                     iterator=True)

    connect_string, cred_dict = load_sql_credential()
    table_name = input_file
    table_size = get_table_size(connect_string, table_name)
    table_rows = get_table_rows(connect_string, table_name)
    print('Read sql table:', table_name, '| Table size:', table_size, '| No. of rows:', table_rows)
    n_chunk = int(np.ceil(table_rows / chunksize))
    sql_query = """select * from {};
                        """.format(table_name)

    print('read:', sql_query)
    engine = create_engine(connect_string)
    connection = engine.connect().execution_options(
        stream_results=True, max_row_buffer=chunksize
    )

    id_med = defaultdict(set)
    i = 0
    n_rows = 0
    dfs = []

    n_no_rxnorm = 0
    n_no_date = 0
    n_no_days_supply = 0

    n_discard_row = 0
    n_recorded_row = 0
    n_not_in_list_row = 0

    for chunk in tqdm(pd.read_sql(sql_query, connection, chunksize=chunksize), total=n_chunk):
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
            rx_order_date = clean_date_str(row['RX_ORDER_DATE'])
            rx_start_date = clean_date_str(row['RX_START_DATE'])
            rx_end_date = clean_date_str(row['RX_END_DATE'])
            rxnorm = row['RXNORM_CUI']
            rx_days = row['RX_DAYS_SUPPLY']
            if 'RAW_RXNORM_CUI' in row.index:
                raw_rxnorm = row['RAW_RXNORM_CUI']
            else:
                raw_rxnorm = np.nan

            encid = row['ENCOUNTERID']  # 2022-10-23 ADD encounter id to drug structure

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
            if pd.notna(rx_days) and rx_days:
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
                    id_med[patid].add(
                        (start_date, rxnorm, days, encid, 'pr'))  # 2022-10-24 add encounter id and prescribeing table id
                    n_recorded_row += 1
                else:
                    if patid in selected_patients:
                        id_med[patid].add(
                            (start_date, rxnorm, days, encid, 'pr'))  # 2022-10-24 add encounter id and prescribeing table id
                        n_recorded_row += 1
                    else:
                        n_not_in_list_row += 1

        dfs.append(chunk[['RX_START_DATE']])
        if i % 25 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('n_rows:', n_rows,
                  'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_no_days_supply:', n_no_days_supply,
                  'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:',
                  n_not_in_list_row)

    print('n_rows:', n_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('n_rows:', n_rows,
          'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_no_days_supply:', n_no_days_supply,
          'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('len(id_med):', len(id_med))
    dfs = pd.concat(dfs)
    print('dfs.shape', dfs.shape)
    print('Time range of prescribing table  of selected patients [RX_START_DATE]:',
          pd.to_datetime(dfs["RX_START_DATE"]).describe(datetime_is_numeric=True))

    # sort
    print('Sort med_list in id_med by time')
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
    :param selected_patients:
    :param output_file:
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
    # sasds = pd.read_sas(input_file,
    #                     encoding='WINDOWS-1252',
    #                     chunksize=chunksize,
    #                     iterator=True)

    connect_string, cred_dict = load_sql_credential()
    table_name = input_file
    table_size = get_table_size(connect_string, table_name)
    table_rows = get_table_rows(connect_string, table_name)
    print('Read sql table:', table_name, '| Table size:', table_size, '| No. of rows:', table_rows)
    n_chunk = int(np.ceil(table_rows / chunksize))
    sql_query = """select * from {};
                       """.format(table_name)

    print('read:', sql_query)
    engine = create_engine(connect_string)
    connection = engine.connect().execution_options(
        stream_results=True, max_row_buffer=chunksize
    )

    id_med = defaultdict(set)
    i = 0
    n_rows = 0
    dfs = []

    n_no_rxnorm = 0
    n_no_date = 0
    n_no_days_supply = 0

    n_discard_row = 0
    n_recorded_row = 0
    n_not_in_list_row = 0

    for chunk in tqdm(pd.read_sql(sql_query, connection, chunksize=chunksize), total=n_chunk):
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
            rx_start_date = clean_date_str(row['MEDADMIN_START_DATE'])
            rx_end_date = clean_date_str(row['MEDADMIN_STOP_DATE'])
            med_type = row['MEDADMIN_TYPE']
            rxnorm = row['MEDADMIN_CODE']
            if 'RAW_MEDADMIN_MED_NAME' in row.index:
                names = row['RAW_MEDADMIN_MED_NAME']

            if 'RAW_MEDADMIN_CODE' in row.index:
                raw_rxnorm = row['RAW_MEDADMIN_CODE']
            else:
                raw_rxnorm = np.nan

            encid = row['ENCOUNTERID']  # 2022-10-23 ADD encounter id to drug structure

            # start_date
            if pd.notna(rx_start_date):
                start_date = rx_start_date
            else:
                start_date = np.nan
                n_no_date += 1

            # rxrnom
            # this code might wrong due to NDC codes 2023-9-4
            # if (med_type != 'RX') or pd.isna(rxnorm):
            #     n_no_rxnorm += 1
            #     rxnorm = np.nan
            # change to the following
            if pd.isna(rxnorm):
                if pd.notna(raw_rxnorm):
                    rxnorm = raw_rxnorm
                else:
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
                    id_med[patid].add(
                        (start_date, rxnorm, days, encid, 'ma'))  # 2022-10-24 add encounter id and med_admi table id
                    n_recorded_row += 1
                else:
                    if patid in selected_patients:
                        id_med[patid].add((start_date, rxnorm, days, encid, 'ma'))
                        n_recorded_row += 1
                    else:
                        n_not_in_list_row += 1

        dfs.append(chunk[['MEDADMIN_START_DATE']])
        if i % 25 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('n_rows:', n_rows,
                  'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_no_days_supply:', n_no_days_supply,
                  'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:',
                  n_not_in_list_row)

    print('n_rows:', n_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('n_rows:', n_rows,
          'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_no_days_supply:', n_no_days_supply,
          'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('len(id_med):', len(id_med))
    try:
        dfs = pd.concat(dfs)
        print('dfs.shape', dfs.shape)
        print('Time range of med_admin table  of selected patients [MEDADMIN_START_DATE]:',
              pd.to_datetime(dfs["MEDADMIN_START_DATE"]).describe(datetime_is_numeric=True))
    except Exception as e:
        # empty file, empty list, nothing to concatenate
        #  raise ValueError("No objects to concatenate")
        # ValueError: No objects to concatenate
        print(e)

    # sort
    print('sort med_list in id_med by time')
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
    # python pre_medication.py --dataset NYU 2>&1 | tee  log/pre_medication_NYU.txt
    # python pre_medication.py --dataset MONTE 2>&1 | tee  log/pre_medication_MONTE.txt
    # python pre_medication.py --dataset MSHS 2>&1 | tee  log/pre_medication_MSHS.txt

    start_time = time.time()
    args = parse_args()
    with open(args.patient_list_file, 'rb') as f:
        selected_patients = pickle.load(f)
        print('len(selected_patients):', len(selected_patients))

    print('args.dataset:', args.dataset)
    print("step 1. prescribe")
    id_med1 = read_prescribing(args.prescribe_file, args.prescribe_output, selected_patients)
    print('read_prescribing done, len(id_med1):', len(id_med1))

    print("step 2. med_admin")
    id_med2 = read_med_admin(args.med_admin_file, args.med_admin_output, selected_patients)
    print('read_med_admin done, len(id_med2):', len(id_med2))

    print("step 3. combine both")
    id_med = combine_2_id_med(id_med1, id_med2, args.output_file)
    print('combine_2_id_med done, len(id_med):', len(id_med))

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
