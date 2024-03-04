import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import time
import argparse
import functools
from misc import utils

print = functools.partial(print, flush=True)
from collections import Counter
import psycopg2
import json
from sqlalchemy import create_engine
from tqdm import tqdm
from misc.utilsql import *


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess encounter for pregnant cohort')
    parser.add_argument('--dataset', default='wcm_pcornet_all',
                        help='all recover sites')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    args.input_file = r'{}.encounter'.format(args.dataset)
    args.output_file = r'../data/recover/output/{}/pregnant_encounter_{}.csv'.format(args.dataset, args.dataset)
    args.output_file_xlsx = r'../data/recover/output/{}/pregnant_encounter_{}.xlsx'.format(args.dataset, args.dataset)

    print('args:', args)

    return args


def read_encounter_4_pregnant(args, code_set, chunksize=100000, debug=False):
    start_time = time.time()
    print('In read_encounter_4_pregnant')
    print('Choose dataset:', args.dataset, 'chunksize:', chunksize, 'debug:', debug)

    # step 1: load procedures codes for pregnant/delivery:
    print('Step 1: load procedures codes for pregnant/delivery')
    print('len(code_set):', len(code_set))
    print('code_set: ', code_set)

    # step 2: read procedures results by chunk, due to large file size
    print('Step 2, read procedures data, and select patients with any pregnant/delivery procedures')
    connect_string, cred_dict = load_sql_credential()
    table_name = args.input_file
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

    dfs = []  # holds data chunks
    dfs_pregnant = []
    cnt = Counter([])
    cnt_code = Counter([])
    i = 0
    n_rows = 0
    n_pregnant_rows = 0
    patid_set = set([])
    patid_pregnant_set = set([])

    for chunk in tqdm(pd.read_sql(sql_query, connection, chunksize=chunksize), total=n_chunk):
        i += 1
        if chunk.empty:
            print("ERROR: Empty chunk! break!")
            break

        chunk.rename(columns=lambda x: x.upper(), inplace=True)
        if i == 1:
            print('chunk.shape', chunk.shape)
            print('chunk.columns', chunk.columns)

        if debug:
            dfs.append(chunk)

        chunk_pregnant_records = chunk.loc[chunk['DRG'].isin(code_set), :]
        dfs_pregnant.append(chunk_pregnant_records)

        patid_set.update(chunk['PATID'])
        patid_pregnant_set.update(chunk_pregnant_records['PATID'])

        n_rows += len(chunk)
        n_pregnant_rows += len(chunk_pregnant_records)

        cnt.update(chunk_pregnant_records['DRG_TYPE'])
        cnt_code.update(chunk_pregnant_records['DRG'])

        if i % 50 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('len(patid_set):', len(patid_set))
            print('len(patid_pregnant_set):', len(patid_pregnant_set))
            print('cnt, DRG type:', cnt)
            print('cnt_code, DRG code:', cnt_code)

            if debug:
                print('IN DEBUG MODE, BREAK, AND DUMP!')
                break

    print('n_rows:', n_rows, 'n_pregnant_rows:', n_pregnant_rows)
    print('len(patid_set):', len(patid_set))
    print('len(patid_pregnant_set):', len(patid_pregnant_set))
    print('#chunk: ', i, 'chunk size:', chunksize)
    print('Counter:', cnt)
    print('Loinc Counter:', cnt_code)

    dfs_pregnant_all = pd.concat(dfs_pregnant)
    print('dfs_pregnant_all.shape', dfs_pregnant_all.shape)
    print('dfs_pregnant_all.columns', dfs_pregnant_all.columns)
    dfs_pregnant_all.rename(columns=lambda x: x.upper(), inplace=True)
    print('dfs_pregnant_all.columns', dfs_pregnant_all.columns)

    print('Output file:', args.output_file)
    utils.check_and_mkdir(args.output_file)
    dfs_pregnant_all.to_csv(args.output_file, index=False)

    if debug:
        try:
            # dump xlsx for debugging
            print('Output file:', args.output_file_xlsx)
            utils.check_and_mkdir(args.output_file_xlsx)
            dfs_pregnant_all.to_excel(args.output_file_xlsx)
        except Exception as e:
            # in write raise ValueError(ValueError: This sheet is too large!
            # Your sheet size is: 1592362, 35 Max sheet size is: 1048576, 16384
            print(e)

        dfs_all = pd.concat(dfs)
        print('dfs_all.shape', dfs_all.shape)
        print('dfs_all.columns', dfs_all.columns)
        dfs_all.rename(columns=lambda x: x.upper(), inplace=True)
        print('dfs_all.columns', dfs_all.columns)
        dfs_all.to_csv("{}_lab_all.csv".format(args.dataset))

    connection.close()
    print('Total Time used after dump files:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return dfs_pregnant_all  # dfs_all, dfs_pregnant_all, meta


if __name__ == '__main__':
    # python pre_lab_4pregnant.py --dataset pregnant_database 2>&1 | tee  log/pre_lab_COL.txt
    # python pre_lab_4pregnant.py --dataset main_database 2>&1 | tee  log/pre_lab_WCM.txt
    # python pre_lab_4pregnant.py --dataset wcm 2>&1 | tee  log\pre_lab_WCM.txt
    start_time = time.time()
    args = parse_args()

    # step 1: load pregnancy/delivery related codes
    df_include = pd.read_excel(r'../data/mapping/RECOVER Preg CP_Technical Details_v3_11.8.22.xlsx',
                               sheet_name='Inclusion',
                               dtype=str)
    df_exclude = pd.read_excel(r'../data/mapping/RECOVER Preg CP_Technical Details_v3_11.8.22.xlsx',
                               sheet_name='Exclusion',
                               dtype=str)

    df_preg = pd.concat([df_include, df_exclude, ], ignore_index=True, sort=False)
    print(df_preg['CodeType'].value_counts())

    df_preg_pro = df_preg.loc[(df_preg['CodeType'] == 'MS-DRG'), 'Code']
    code_set = set(df_preg_pro.to_list())
    print('Selected all pregnant/delivery related MS-DRG codes, both inclusion and exclusion: ')
    print('len(code_set):', len(code_set))
    print('code_set:', code_set)

    print(args)
    dfs_pregnant_all = read_encounter_4_pregnant(args, code_set, debug=args.debug)
    print('Total Time used after dump files:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
