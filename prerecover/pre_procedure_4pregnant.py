import sys
# for linux env.
sys.path.insert(0,'..')
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
    parser = argparse.ArgumentParser(description='preprocess procedure for pregnant cohort')
    parser.add_argument('--dataset', default='temple',
                        help='all recover sites')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    args.input_file = r'{}.lab_result_cm'.format(args.dataset)
    args.output_file = r'../data/recover/output/{}/covid_lab_{}.csv'.format(args.dataset, args.dataset)
    args.output_file_xlsx = r'../data/recover/output/{}/covid_lab_{}.xlsx'.format(args.dataset, args.dataset)

    print('args:', args)

    return args


def read_lab_and_count_covid(args, chunksize=100000, debug=False):
    start_time = time.time()
    print('in read_lab_and_count_covid')
    print('Choose dataset:', args.dataset, 'chunksize:', chunksize, 'debug:', debug)
    # step 1: load covid lab test codes, may be updated by:
    print('Step 1: load and selected covid related lab code')
    df_covid = pd.read_csv(r'../data/V15_COVID19/covid_phenotype/COVID_LOINC_all.csv')
    # No. We can choose according our list,
    # and then narrow down to the PCR test in pre_covid_lab.py
    # !!! chosen PCR tests in pre_covid_lab.py
    # Keep all covid related codes here

    print('df_covid.shape:', df_covid.shape)
    code_set = set(df_covid['loinc_num'].to_list())
    print('Selected all Covid related codes: ', code_set)
    print('len(code_set):', len(code_set))

    # step 2: read lab results by chunk, due to large file size
    print('Step 2, read lab data, and select patients who took COVID PCR test, with their covid lab records')

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
    dfs_covid = []
    cnt = Counter([])
    cnt_code = Counter([])
    i = 0
    n_rows = 0
    n_covid_rows = 0
    patid_set = set([])
    patid_covid_set = set([])

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

        chunk_covid_records = chunk.loc[chunk['LAB_LOINC'].isin(code_set), :]
        dfs_covid.append(chunk_covid_records)
        # only keep covid test records.
        # other records of patients who took covid test need to scan again

        patid_set.update(chunk['PATID'])
        patid_covid_set.update(chunk_covid_records['PATID'])

        n_rows += len(chunk)
        n_covid_rows += len(chunk_covid_records)

        cnt.update(chunk_covid_records['RESULT_QUAL'])
        cnt_code.update(chunk_covid_records['LAB_LOINC'])

        if i % 50 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('len(patid_set):', len(patid_set))
            print('len(patid_covid_set):', len(patid_covid_set))

            if debug:
                print('IN DEBUG MODE, BREAK, AND DUMP!')
                break

    print('n_rows:', n_rows, 'n_covid_rows:', n_covid_rows)
    print('len(patid_set):', len(patid_set))
    print('len(patid_covid_set):', len(patid_covid_set))
    print('#chunk: ', i, 'chunk size:', chunksize)
    print('Counter:', cnt)
    print('Loinc Counter:', cnt_code)

    dfs_covid_all = pd.concat(dfs_covid)
    print('dfs_covid_all.shape', dfs_covid_all.shape)
    print('dfs_covid_all.columns', dfs_covid_all.columns)
    dfs_covid_all.rename(columns=lambda x: x.upper(), inplace=True)
    print('dfs_covid_all.columns', dfs_covid_all.columns)

    print('Output file:', args.output_file)
    utils.check_and_mkdir(args.output_file)
    dfs_covid_all.to_csv(args.output_file, index=False)

    if debug:
        try:
            # dump xlsx for debugging
            print('Output file:', args.output_file_xlsx)
            utils.check_and_mkdir(args.output_file_xlsx)
            dfs_covid_all.to_excel(args.output_file_xlsx)
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

    return dfs_covid_all  # dfs_all, dfs_covid_all, meta


if __name__ == '__main__':
    # python pre_lab_4covid.py --dataset covid_database 2>&1 | tee  log/pre_lab_COL.txt
    # python pre_lab_4covid.py --dataset main_database 2>&1 | tee  log/pre_lab_WCM.txt
    # python pre_lab_4covid.py --dataset wcm 2>&1 | tee  log\pre_lab_WCM.txt
    start_time = time.time()
    args = parse_args()

    # step 1: load pregnancy/delivery related diagnosis
    df_include = pd.read_excel(r'../data/mapping/RECOVER Preg CP_Technical Details_v3_11.8.22.xlsx',
                               sheet_name='Inclusion',
                               dtype=str)
    df_exclude = pd.read_excel(r'../data/mapping/RECOVER Preg CP_Technical Details_v3_11.8.22.xlsx',
                               sheet_name='Exclusion',
                               dtype=str)

    df_preg = pd.concat([df_include, df_exclude, ], ignore_index=True, sort=False)
    df_preg_pro = df_preg.loc[(df_preg['CodeType'] == 'ICD-10-PCS')|(df_preg['CodeType'] == 'CPT'), 'Code']
    code_set = set(df_preg_pro.to_list())
    print('Selected all pregnant/delivery related ICD-10-PCS and CPT procedure codes, both inclusion and exclusion: ')
    print('len(code_set):', len(code_set))
    print('code_set:', code_set)

    print(args)
    dfs_covid_all = read_lab_and_count_covid(args, debug=args.debug)
    print('Total Time used after dump files:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
