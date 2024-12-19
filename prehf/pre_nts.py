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
from misc.utilsql import *
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess diagnosis')
    parser.add_argument('--dataset', default='wcm', help='site dataset')
    args = parser.parse_args()

    args.input_file = 'notes_s2.merged' #r'{}.diagnosis'.format(args.dataset)
    args.patient_list_file = r'../data/recover/output_hf/{}/patient_hf_list_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file = r'../data/recover/output_hf/{}/nts_{}.csv'.format(args.dataset, args.dataset)
    print('args:', args)
    return args


def read_nts(dataset, output_file, selected_patients={}):
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

    # sasds = pd.read_sas(input_file,
    #                     encoding='WINDOWS-1252',
    #                     chunksize=chunksize,
    #                     iterator=True)
    connect_string, cred_dict = load_sql_credential()
    table_name = 'notes_s2.merged'
    if dataset == 'montefiore':
        dataset = 'monte'
    table_size = get_table_size(connect_string, table_name)
    table_rows = get_table_rows(connect_string, 'notes_s2.merged where site=\'{}\''.format(dataset))
    print('Read sql table:', table_name, '| Table size:', table_size, '| No. of rows in :', dataset, table_rows)
    n_chunk = int(np.ceil(table_rows / chunksize))
    sql_query = """select * from notes_s2.merged where site=\'{}\';
                        """.format(dataset)

    print('read:', sql_query)
    engine = create_engine(connect_string)
    connection = engine.connect().execution_options(
        stream_results=True, max_row_buffer=chunksize
    )

    if selected_patients:
        print('using selected_patients, len(selected_patients):', len(selected_patients))

    i = 0
    n_rows = 0
    dfs = []
    dfs_hf = []
    n_hf_rows = 0
    patid_set = set([])
    patid_hf_set = set([])
    # cnt = Counter([])
    cnt_code = Counter([])

    for chunk in tqdm(pd.read_sql(sql_query, connection, chunksize=chunksize), total=n_chunk):
        i += 1
        if chunk.empty:
            print("ERROR: Empty chunk! break!")
            break

        n_rows += len(chunk)
        # chunk.rename(columns=lambda x: x.upper(), inplace=True)
        if i == 1:
            print('chunk.shape', chunk.shape)
            print('chunk.columns', chunk.columns)

        chunk_pid = chunk['site_person_id']
        chunk_hf_records = chunk.loc[chunk_pid.isin(selected_patients), :]
        # chunk_hf_records.rename(columns=lambda x: x.upper(), inplace=True)
        dfs_hf.append(chunk_hf_records)
        # only select cohorts with HF dx.

        patid_set.update(chunk['site_person_id'])
        patid_hf_set.update(chunk_hf_records['site_person_id'])

        n_rows += len(chunk)
        n_hf_rows += len(chunk_hf_records)

        cnt_code.update(chunk_hf_records['note_type_source_value'])

        # # monte case, too large, error. other sites ok
        # dfs.append(chunk[['PATID', 'ENCOUNTERID', 'ENC_TYPE', "ADMIT_DATE", 'DX', "DX_TYPE"]])
        dfs.append(chunk[["note_date"]])

        if i % 50 == 0:
            print('chunk:', i, 'len(dfs):', len(dfs), 'len(dfs_hf):', len(dfs_hf),
                  'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

            print('n_rows:', n_rows, 'n_hf_rows:', n_hf_rows, )
            print('len(patid_set):', len(patid_set))
            print('len(patid_hf_set):', len(patid_hf_set))

    print('n_rows:', n_rows, 'n_hf_rows:', n_hf_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('len(patid_set):', len(patid_set))
    print('len(patid_hf_set):', len(patid_hf_set))
    print('HF DX Counter:', cnt_code)

    dfs = pd.concat(dfs)
    print('dfs.shape', dfs.shape)
    print('Time range of diagnosis table of all patients:',
          pd.to_datetime(dfs["note_date"]).describe(datetime_is_numeric=True))

    dfs_hf_all = pd.concat(dfs_hf)
    print('dfs_hf_all.shape', dfs_hf_all.shape)
    print('dfs_hf_all.columns', dfs_hf_all.columns)
    # dfs_hf_all.rename(columns=lambda x: x.upper(), inplace=True)
    print('dfs_hf_all.columns', dfs_hf_all.columns)
    print('Time range of diagnosis table of selected HF patients:',
          pd.to_datetime(dfs_hf_all["note_date"]).describe(datetime_is_numeric=True))

    print('Output file:', output_file)
    utils.check_and_mkdir(output_file)
    dfs_hf_all.to_csv(output_file, index=False)

    connection.close()
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return dfs_hf_all


if __name__ == '__main__':
    # python pre_diagnosis.py --dataset WCM 2>&1 | tee  log/pre_diagnosis_WCM.txt

    start_time = time.time()
    args = parse_args()
    print('Selected site:', args.dataset)
    with open(args.patient_list_file, 'rb') as f:
        selected_patients = pickle.load(f)
        print('len(selected_patients):', len(selected_patients))

    df = read_nts(args.dataset, args.output_file, selected_patients)
    print('len(selected_patients)', len(selected_patients), 'len(df)', len(df))
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
