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
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess-count pos and negative from lab file in all recover sites')
    parser.add_argument('--dataset', default='wcm', help='all recover sites')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    args.patient_list_file = r'../data/recover/output/{}/patient_covid_lab-dx-med_{}.pkl'.format(args.dataset, args.dataset)
    args.input_file = r'{}.lab_result_cm'.format(args.dataset)
    args.output_file = r'../data/recover/output/{}/ckd_lab_{}.csv'.format(args.dataset, args.dataset)

    print('args:', args)

    return args


def read_ckd_lab(args, chunksize=100000, selected_patients={}):
    start_time = time.time()
    print('in read_ckd_lab')
    print('Choose dataset:', args.dataset, 'chunksize:', chunksize,)
    # step 1: load covid lab test codes, may be updated by:
    print('Step 1: load and selected covid related lab code')
    if selected_patients:
        print('using selected_patients, len(selected_patients):', len(selected_patients))

    df_ckd1 = pd.read_excel(r'../data/mapping/ckd_codes_revised.xlsx', sheet_name=r'serum creatinine')
    df_ckd2 = pd.read_excel(r'../data/mapping/ckd_codes_revised.xlsx', sheet_name=r'eGFR')

    print('serum creatinine df_ckd1.shape:', df_ckd1.shape, 'serum creatinine df_ckd2.shape:', df_ckd2.shape)
    code_set1 = set(df_ckd1['code'].to_list())
    code_set2 = set(df_ckd2['code'].to_list())
    code_set = code_set1.union(code_set2)
    print('Selected all ckd related codes: ', len(code_set),
          'where serum creatinine: ', len(code_set1),
          'where eGFR: ', len(code_set2))

    # step 2: read lab results by chunk, due to large file size
    print('Step 2, read lab data from selected patients who took COVID PCR test, and  their CKD lab records')

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
    dfs_covid_ckd = []
    cnt = Counter([])
    cnt_code = Counter([])
    i = 0
    n_rows = 0
    n_covid_ckd_rows = 0
    patid_set = set([])
    patid_covid_ckd_set = set([])

    for chunk in tqdm(pd.read_sql(sql_query, connection, chunksize=chunksize), total=n_chunk):
        i += 1
        if chunk.empty:
            print("ERROR: Empty chunk! break!")
            break

        chunk.rename(columns=lambda x: x.upper(), inplace=True)
        if i == 1:
            print('chunk.shape', chunk.shape)
            print('chunk.columns', chunk.columns)

        if selected_patients:
            chunk_covid_ckd_records = chunk.loc[chunk['LAB_LOINC'].isin(code_set) & chunk['PATID'].isin(selected_patients), :]
        else:
            chunk_covid_ckd_records = chunk.loc[chunk['LAB_LOINC'].isin(code_set), :]

        dfs_covid_ckd.append(chunk_covid_ckd_records)

        patid_set.update(chunk['PATID'])
        patid_covid_ckd_set.update(chunk_covid_ckd_records['PATID'])

        n_rows += len(chunk)
        n_covid_ckd_rows += len(chunk_covid_ckd_records)

        cnt.update(chunk_covid_ckd_records['RESULT_QUAL'])
        cnt_code.update(chunk_covid_ckd_records['LAB_LOINC'])

        if i % 50 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('len(patid_set):', len(patid_set))
            print('len(chunk_covid_ckd_records):', len(chunk_covid_ckd_records))


    print('n_rows:', n_rows, 'n_covid_rows:', n_covid_ckd_rows)
    print('len(patid_set):', len(patid_set))
    print('len(patid_covid_ckd_set):', len(patid_covid_ckd_set))
    print('#chunk: ', i, 'chunk size:', chunksize)
    print('Counter:', cnt)
    print('Loinc Counter:', cnt_code)

    dfs_covid_ckd_all = pd.concat(dfs_covid_ckd)
    print('dfs_covid_ckd_all.shape', dfs_covid_ckd_all.shape)
    print('dfs_covid_ckd_all.columns', dfs_covid_ckd_all.columns)

    print('Output file:', args.output_file)
    utils.check_and_mkdir(args.output_file)
    dfs_covid_ckd_all.to_csv(args.output_file, index=False)

    connection.close()
    print('Total Time used after dump files:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return dfs_covid_ckd_all


if __name__ == '__main__':
    # python pre_lab_4covid.py --dataset covid_database 2>&1 | tee  log/pre_lab_COL.txt
    # python pre_lab_4covid.py --dataset main_database 2>&1 | tee  log/pre_lab_WCM.txt
    # python pre_lab_4covid.py --dataset wcm 2>&1 | tee  log\pre_lab_WCM.txt
    start_time = time.time()
    args = parse_args()
    print('args.dataset:', args.dataset)
    with open(args.patient_list_file, 'rb') as f:
        selected_patients = pickle.load(f)
        print('len(selected_patients):', len(selected_patients))

    print(args)
    dfs_covid_ckd_all = read_ckd_lab(args, selected_patients=selected_patients)
    print('Total Time used after dump files:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
