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

    args.input_file = r'{}.diagnosis'.format(args.dataset)
    args.output_file = r'../data/recover/output_/{}/covid_diagnosis_{}.csv'.format(args.dataset, args.dataset)
    print('args:', args)
    return args


def read_diagnosis_4_covid(args, chunksize=100000, ):
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
    print('Choose dataset:', args.dataset, 'chunksize:', chunksize, )

    # step 1: load covid lab test codes, may be updated by:
    print('Step 1: use dx U071 to find covid patients')
    code_set = {'U071', 'U07.1'}
    print('COVID-19 dx codes: ', code_set)
    print('len(code_set):', len(code_set))

    # step 2: read dx results by chunk, due to large file size
    print('Step 2, read dx data, and select patients who had any COVID-19 diagnoses!')
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

    i = 0
    n_rows = 0
    dfs = []
    dfs_covid = []
    n_covid_rows = 0
    patid_set = set([])
    patid_covid_set = set([])
    # cnt = Counter([])
    cnt_code = Counter([])

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

        chunk_dx = chunk['DX'].apply(lambda x: x.strip().replace('.', '').upper() if isinstance(x, str) else x)
        chunk_covid_records = chunk.loc[chunk_dx.isin(code_set), :]
        chunk_covid_records.rename(columns=lambda x: x.upper(), inplace=True)
        dfs_covid.append(chunk_covid_records)
        # only select cohorts with covid dx.

        patid_set.update(chunk['PATID'])
        patid_covid_set.update(chunk_covid_records['PATID'])

        n_rows += len(chunk)
        n_covid_rows += len(chunk_covid_records)

        cnt_code.update(chunk_covid_records['DX'])

        # # monte case, too large, error. other sites ok
        # dfs.append(chunk[['PATID', 'ENCOUNTERID', 'ENC_TYPE', "ADMIT_DATE", 'DX', "DX_TYPE"]])
        dfs.append(chunk[["ADMIT_DATE"]])

        if i % 50 == 0:
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
          pd.to_datetime(dfs["ADMIT_DATE"]).describe(datetime_is_numeric=True))

    dfs_covid_all = pd.concat(dfs_covid)
    print('dfs_covid_all.shape', dfs_covid_all.shape)
    print('dfs_covid_all.columns', dfs_covid_all.columns)
    dfs_covid_all.rename(columns=lambda x: x.upper(), inplace=True)
    print('dfs_covid_all.columns', dfs_covid_all.columns)
    print('Time range of diagnosis table of selected covid patients:',
          pd.to_datetime(dfs_covid_all["ADMIT_DATE"]).describe(datetime_is_numeric=True))

    print('Output file:', args.output_file)
    utils.check_and_mkdir(args.output_file)
    dfs_covid_all.to_csv(args.output_file, index=False)

    connection.close()
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return dfs_covid_all


if __name__ == '__main__':
    # python pre_dx_4covid.py --dataset wcm 2>&1 | tee  log/pre_dx_4covid
    # _wcm.txt

    start_time = time.time()
    args = parse_args()
    print('Selected site:', args.dataset)
    print('args:', args)
    df = read_diagnosis_4_covid(args, )

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
