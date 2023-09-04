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
from collections import Counter

print = functools.partial(print, flush=True)
from collections import defaultdict
from datetime import datetime, date


# 1. preprocess prescribing file, support supply day calculation and imputation
# 2. preprocess med_admin file if necessary, support supply day calculation and imputation
# 3. combine results from both files if necessary (for COL case)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess medication table')
    parser.add_argument('--dataset', default='nebraska', help='site dataset')
    args = parser.parse_args()

    args.med_admin_file = r'{}.med_admin'.format(args.dataset)
    args.prescribe_file = r'{}.prescribing'.format(args.dataset)
    args.dispensing_file = r'{}.dispensing'.format(args.dataset)

    args.med_admin_output = r'../data/recover/output/{}/covid_med_admin_{}.csv'.format(args.dataset, args.dataset)
    args.prescribe_output = r'../data/recover/output/{}/covid_prescribing_{}.csv'.format(args.dataset, args.dataset)
    args.dispensing_output = r'../data/recover/output/{}/covid_dispensing_{}.csv'.format(args.dataset, args.dataset)

    print('args:', args)
    return args


def read_prescribing_4_covid(input_file, output_file, code_set, chunksize=100000):
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
    print('In read_prescribing_4_covid, input_file:', input_file, 'output_file:', output_file)
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

    # id_med = defaultdict(set)
    i = 0
    n_rows = 0
    dfs = []
    dfs_covid = []
    n_covid_rows = 0
    patid_set = set([])
    patid_covid_set = set([])
    # cnt = Counter([])
    cnt_code = Counter([])

    for chunk in tqdm(pd.read_sql(sql_query, connection, chunksize=chunksize), total=n_chunk, mininterval=3):
        i += 1
        if chunk.empty:
            print("ERROR: Empty chunk! break!")
            break
        n_rows += len(chunk)
        chunk.rename(columns=lambda x: x.upper(), inplace=True)
        if i == 1:
            print('chunk.shape', chunk.shape)
            print('chunk.columns', chunk.columns)

        select_id = chunk['RXNORM_CUI'].isin(code_set)
        if 'RAW_RXNORM_CUI' in chunk.columns:
            select_id = select_id | chunk['RAW_RXNORM_CUI'].isin(code_set)

        chunk_covid_records = chunk.loc[select_id, :].copy()
        dfs_covid.append(chunk_covid_records)

        patid_set.update(chunk['PATID'])
        patid_covid_set.update(chunk_covid_records['PATID'])

        n_rows += len(chunk)
        n_covid_rows += len(chunk_covid_records)

        cnt_code.update(chunk_covid_records['RXNORM_CUI'])
        dfs.append(chunk[["RX_START_DATE"]])

        if i % 25 == 0:
            print('chunk:', i, 'len(dfs):', len(dfs), 'len(dfs_covid):', len(dfs_covid),
                  'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

            print('n_rows:', n_rows, 'n_covid_rows:', n_covid_rows, )
            print('len(patid_set):', len(patid_set))
            print('len(patid_covid_set):', len(patid_covid_set))

    print('n_rows:', n_rows, 'n_covid_rows:', n_covid_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('len(patid_set):', len(patid_set))
    print('len(patid_covid_set):', len(patid_covid_set))
    print('covid DX Counter:', cnt_code)

    dfs = pd.concat(dfs)
    print('dfs.shape', dfs.shape)
    print('Time range of diagnosis table of all patients:',
          pd.to_datetime(dfs["RX_START_DATE"]).describe(datetime_is_numeric=True))

    dfs_covid_all = pd.concat(dfs_covid)
    print('dfs_covid_all.shape', dfs_covid_all.shape)
    print('dfs_covid_all.columns', dfs_covid_all.columns)
    dfs_covid_all.rename(columns=lambda x: x.upper(), inplace=True)
    print('dfs_covid_all.columns', dfs_covid_all.columns)
    print('Time range of diagnosis table of selected covid patients:',
          pd.to_datetime(dfs_covid_all["RX_START_DATE"]).describe(datetime_is_numeric=True))

    print('Output file:', output_file)
    utils.check_and_mkdir(output_file)
    dfs_covid_all.to_csv(output_file, index=False)

    connection.close()
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return dfs_covid_all


def read_med_admin_4_covid(input_file, output_file, code_set, chunksize=100000):
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

    # id_med = defaultdict(set)
    i = 0
    n_rows = 0
    dfs = []
    dfs_covid = []
    n_covid_rows = 0
    patid_set = set([])
    patid_covid_set = set([])
    # cnt = Counter([])
    cnt_code = Counter([])

    for chunk in tqdm(pd.read_sql(sql_query, connection, chunksize=chunksize), total=n_chunk, mininterval=3):
        i += 1
        if chunk.empty:
            print("ERROR: Empty chunk! break!")
            break
        n_rows += len(chunk)
        chunk.rename(columns=lambda x: x.upper(), inplace=True)
        if i == 1:
            print('chunk.shape', chunk.shape)
            print('chunk.columns', chunk.columns)

        select_id = chunk['MEDADMIN_CODE'].isin(code_set)
        if 'RAW_MEDADMIN_CODE' in chunk.columns:
            # can be ndc code, type ND, not ndc11, say with NDC: prefix, or with - -
            select_id = select_id | chunk['RAW_MEDADMIN_CODE'].isin(code_set)

        chunk_covid_records = chunk.loc[select_id, :].copy()
        dfs_covid.append(chunk_covid_records)

        patid_set.update(chunk['PATID'])
        patid_covid_set.update(chunk_covid_records['PATID'])

        n_rows += len(chunk)
        n_covid_rows += len(chunk_covid_records)

        cnt_code.update(chunk_covid_records['MEDADMIN_CODE'])
        dfs.append(chunk[["MEDADMIN_START_DATE"]])
        if i % 25 == 0:
            print('chunk:', i, 'len(dfs):', len(dfs), 'len(dfs_covid):', len(dfs_covid),
                  'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

            print('n_rows:', n_rows, 'n_covid_rows:', n_covid_rows, )
            print('len(patid_set):', len(patid_set))
            print('len(patid_covid_set):', len(patid_covid_set))

    print('n_rows:', n_rows, 'n_covid_rows:', n_covid_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('len(patid_set):', len(patid_set))
    print('len(patid_covid_set):', len(patid_covid_set))
    print('covid DX Counter:', cnt_code)

    dfs = pd.concat(dfs)
    print('dfs.shape', dfs.shape)
    print('Time range of diagnosis table of all patients:',
          pd.to_datetime(dfs["MEDADMIN_START_DATE"]).describe(datetime_is_numeric=True))

    dfs_covid_all = pd.concat(dfs_covid)
    print('dfs_covid_all.shape', dfs_covid_all.shape)
    print('dfs_covid_all.columns', dfs_covid_all.columns)
    dfs_covid_all.rename(columns=lambda x: x.upper(), inplace=True)
    print('dfs_covid_all.columns', dfs_covid_all.columns)
    print('Time range of diagnosis table of selected covid patients:',
          pd.to_datetime(dfs_covid_all["MEDADMIN_START_DATE"]).describe(datetime_is_numeric=True))

    print('Output file:', output_file)
    utils.check_and_mkdir(output_file)
    dfs_covid_all.to_csv(output_file, index=False)

    connection.close()
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return dfs_covid_all


def read_dispensing_4_covid(input_file, output_file, code_set, chunksize=100000):
    """
    :param selected_patients:
    :param output_file:
    :param input_file: input dispensing file with std format
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
    print('In dispensing, input_file:', input_file, 'output_file:', output_file)
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

    # id_med = defaultdict(set)
    i = 0
    n_rows = 0
    dfs = []
    dfs_covid = []
    n_covid_rows = 0
    patid_set = set([])
    patid_covid_set = set([])
    # cnt = Counter([])
    cnt_code = Counter([])

    for chunk in tqdm(pd.read_sql(sql_query, connection, chunksize=chunksize), total=n_chunk, mininterval=3):
        i += 1
        if chunk.empty:
            print("ERROR: Empty chunk! break!")
            break
        n_rows += len(chunk)
        chunk.rename(columns=lambda x: x.upper(), inplace=True)
        if i == 1:
            print('chunk.shape', chunk.shape)
            print('chunk.columns', chunk.columns)

        select_id = chunk['NDC'].isin(code_set)
        if 'RAW_NDC' in chunk.columns:
            # raw NDC might be not NDC11, say with NDC: prefix, or with --, but not complicate here
            select_id = select_id | chunk['RAW_NDC'].isin(code_set)

        chunk_covid_records = chunk.loc[select_id, :].copy()
        dfs_covid.append(chunk_covid_records)

        patid_set.update(chunk['PATID'])
        patid_covid_set.update(chunk_covid_records['PATID'])

        n_rows += len(chunk)
        n_covid_rows += len(chunk_covid_records)

        cnt_code.update(chunk_covid_records['NDC'])
        dfs.append(chunk[["DISPENSE_DATE"]])
        if i % 25 == 0:
            print('chunk:', i, 'len(dfs):', len(dfs), 'len(dfs_covid):', len(dfs_covid),
                  'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

            print('n_rows:', n_rows, 'n_covid_rows:', n_covid_rows, )
            print('len(patid_set):', len(patid_set))
            print('len(patid_covid_set):', len(patid_covid_set))

    print('n_rows:', n_rows, 'n_covid_rows:', n_covid_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('len(patid_set):', len(patid_set))
    print('len(patid_covid_set):', len(patid_covid_set))
    print('covid DX Counter:', cnt_code)

    # dfs and dfs_convid can be empty, then error raise. but it is OK.
    try:
        dfs = pd.concat(dfs)
        print('dfs.shape', dfs.shape)
        print('Time range of diagnosis table of all patients:',
              pd.to_datetime(dfs["DISPENSE_DATE"]).describe(datetime_is_numeric=True))
    except Exception as e:
        # empty file, empty list, nothing to concatenate
        #  raise ValueError("No objects to concatenate")
        # ValueError: No objects to concatenate
        print(e, 'in dfs = pd.concat(dfs)')

    try:
        dfs_covid_all = pd.concat(dfs_covid)
        print('Time range of diagnosis table of selected covid patients:',
              pd.to_datetime(dfs_covid_all["DISPENSE_DATE"]).describe(datetime_is_numeric=True))

    except Exception as e:
        # empty file, empty list, nothing to concatenate
        #  raise ValueError("No objects to concatenate")
        # ValueError: No objects to concatenate
        print(e, 'in dfs_covid_all = pd.concat(dfs_covid)')
        dfs_covid_all = pd.DataFrame(columns=chunk.columns)
        print('dfs_covid_all.shape', dfs_covid_all.shape)
        print('dfs_covid_all.columns', dfs_covid_all.columns)

    dfs_covid_all.rename(columns=lambda x: x.upper(), inplace=True)
    print('dfs_covid_all.columns', dfs_covid_all.columns)

    print('Output file:', output_file)
    utils.check_and_mkdir(output_file)
    dfs_covid_all.to_csv(output_file, index=False)

    connection.close()
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return dfs_covid_all


if __name__ == '__main__':
    # python pre_med_4covid.py --dataset wcm 2>&1 | tee  log/pre_med_4covid_wcm.txt

    start_time = time.time()
    args = parse_args()

    # step 1: load covid related drugs
    df_pax = pd.read_excel(r'../data/mapping/covid_drug.xlsx', sheet_name='paxlovid', dtype=str)
    df_rem = pd.read_excel(r'../data/mapping/covid_drug.xlsx', sheet_name='remdesivir', dtype=str)
    df_drug = pd.concat([df_pax, df_rem, ], ignore_index=True, sort=False)
    code_set = set(df_drug['code1'].to_list())
    print('Selected all Covid related drug codes: ', df_drug, code_set)
    print('len(code_set):', len(code_set))
    print('args.dataset:', args.dataset)

    print("step 1. extract covid drug from prescribe")
    df_med1 = read_prescribing_4_covid(args.prescribe_file, args.prescribe_output, code_set)
    print('read_prescribing done, len(df_med1):', df_med1.shape)

    print("step 2. extract covid drug from med_admin")
    df_med2 = read_med_admin_4_covid(args.med_admin_file, args.med_admin_output, code_set)
    print('read_med_admin done, len(df_med2):', df_med2.shape)

    print("step 3. extract covid drug from dispensing")
    df_med3 = read_dispensing_4_covid(args.dispensing_file, args.dispensing_output, code_set)
    print('read_med_admin done, len(df_med3):', df_med3.shape)

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
