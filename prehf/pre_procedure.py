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
from misc.utilsql import *

print = functools.partial(print, flush=True)
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess diagnosis')
    parser.add_argument('--dataset', default='ochsner', help='site dataset')
    args = parser.parse_args()

    args.input_file = r'{}.procedures'.format(args.dataset)
    args.patient_list_file = r'../data/recover/output_hf/{}/patient_hf_list_{}.pkl'.format(args.dataset, args.dataset)

    # args.patient_list_file = r'../data/recover/output/{}/patient_covid_lab_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file = r'../data/recover/output_hf/{}/procedures_{}.pkl'.format(args.dataset, args.dataset)

    args.input_file2 = r'{}.obs_gen'.format(args.dataset)
    args.output_file2 = r'../data/recover/output_hf/{}/obs_gen_{}.pkl'.format(args.dataset, args.dataset)

    print('args:', args)
    return args


def read_procedure(input_file, output_file='', selected_patients={}):
    """
    :param data_file: input procedure file with std format
    :param out_file: output id_code-list[patid] = [(time, code, codetype, enctype, encid), ...] pickle sorted by time
    :return: id_code-list[patid] = [(time, code, codetype, enctype, encid), ...]  sorted by time
    :Notice:
        discard rows with NULL admit_date or dx

        1.COL data: e.g:
        df.shape: (6570861, 15)

        2. WCM data: e.g.
        df.shape: (69604478, 15)

    """
    start_time = time.time()
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
            enc_id = row['ENCOUNTERID']
            enc_type = row['ENC_TYPE']
            px = row['PX']
            px_type = row["PX_TYPE"]
            px_date = utils.clean_date_str(row["PX_DATE"])
            admit_date = utils.clean_date_str(row['ADMIT_DATE'])

            if pd.isna(px) or (px == ''):
                n_no_px += 1

            if pd.isna(px_date):
                if pd.notna(admit_date):
                    px_date = admit_date
                else:
                    n_no_date += 1

            if pd.isna(px) or pd.isna(px_date) or (px == ''):
                n_discard_row += 1
            else:
                if not selected_patients:
                    id_px[patid].append((px_date, px, px_type, enc_type, enc_id))
                    n_recorded_row += 1
                else:
                    if patid in selected_patients:
                        id_px[patid].append((px_date, px, px_type, enc_type, enc_id))
                        n_recorded_row += 1
                    else:
                        n_not_in_list_row += 1

        # # monte case, too large, error. other sites ok
        # dfs.append(chunk[['PATID', 'ENCOUNTERID', 'ENC_TYPE', "ADMIT_DATE", 'DX', "DX_TYPE"]])
        dfs.append(chunk[["ADMIT_DATE"]])

        if i % 25 == 0:
            print('chunk:', i, 'len(dfs):', len(dfs),
                  'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('n_rows:', n_rows, 'n_no_dx:', n_no_px, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
                  'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('n_rows:', n_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('n_no_dx:', n_no_px, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('len(id_px):', len(id_px))
    dfs = pd.concat(dfs)
    print('dfs.shape', dfs.shape)
    print('Time range of diagnosis table of selected patients:',
          pd.to_datetime(dfs["ADMIT_DATE"]).describe(datetime_is_numeric=True))

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
        utils.dump(id_px, output_file)

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_px, dfs


def read_obs_gen(input_file, output_file='', selected_patients={}):
    """
    :param input_file: input obs_gen file with std format
    :param output_file: output file
    :param selected_patients: scan information of selected patients
    :return: id_code-list[patid] = [(time, code, codetype, source, text, encid), ...]  sorted by time
    :Notice:
        discard rows with NULL admit_date or dx

        1.COL data: e.g:
        df.shape: None

        2. WCM data: e.g.
        df.shape: (1560368, 15)

    """
    start_time = time.time()
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
            enc_id = row['ENCOUNTERID']
            px = row['OBSGEN_CODE']
            px_type = row["OBSGEN_TYPE"]
            px_date = utils.clean_date_str(row["OBSGEN_START_DATE"])
            result_text = row['OBSGEN_RESULT_TEXT']
            source = row['OBSGEN_SOURCE']

            if pd.isna(px):
                n_no_px += 1

            if pd.isna(px_date):
                n_no_date += 1

            if pd.isna(px) or pd.isna(px_date):
                n_discard_row += 1
            else:
                if not selected_patients:
                    id_px[patid].append((px_date, px, px_type, result_text, source, enc_id))
                    n_recorded_row += 1
                else:
                    if patid in selected_patients:
                        id_px[patid].append((px_date, px, px_type, result_text, source, enc_id))
                        n_recorded_row += 1
                    else:
                        n_not_in_list_row += 1

        # # monte case, too large, error. other sites ok
        # dfs.append(chunk[['PATID', 'ENCOUNTERID', 'ENC_TYPE', "ADMIT_DATE", 'DX', "DX_TYPE"]])
        dfs.append(chunk[["OBSGEN_START_DATE"]])

        if i % 25 == 0:
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
        utils.dump(id_px, output_file)

    if dfs:
        dfs = pd.concat(dfs)
        print('dfs.shape', dfs.shape)
        print('Time range of diagnosis table of selected patients:',
              pd.to_datetime(dfs["OBSGEN_START_DATE"]).describe(datetime_is_numeric=True))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_px, dfs


if __name__ == '__main__':
    # python pre_procedure.py --dataset COL 2>&1 | tee  log/pre_procedure_COL.txt
    # python pre_procedure.py --dataset WCM 2>&1 | tee  log/pre_procedure_WCM.txt
    # python pre_procedure.py --dataset NYU 2>&1 | tee  log/pre_procedure_NYU.txt
    # python pre_procedure.py --dataset MONTE 2>&1 | tee  log/pre_procedure_MONTE.txt
    # python pre_procedure.py --dataset MSHS 2>&1 | tee  log/pre_procedure_MSHS.txt
    start_time = time.time()
    args = parse_args()
    print('Selected site:', args.dataset)
    with open(args.patient_list_file, 'rb') as f:
        selected_patients = pickle.load(f)
        print('len(selected_patients):', len(selected_patients))

    id_px, df = read_procedure(args.input_file, args.output_file, selected_patients)
    id_obs, df_obs = read_obs_gen(args.input_file2, args.output_file2, selected_patients)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
