import sys
# for linux env.
sys.path.insert(0,'..')
import pandas as pd
import time
import pickle
import argparse
from misc import utils
import functools
from misc.utilsql import *

print = functools.partial(print, flush=True)
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess diagnosis')
    parser.add_argument('--dataset', default='pitt', help='site dataset')
    args = parser.parse_args()

    # use all selected patients by lab-dx-med-preg, selected covid-CP specific cohorts later in pre_cohort_**.py
    args.patient_list_file = r'../data/recover/output/{}/patient_covid_lab-dx-med-preg_{}.pkl'.format(args.dataset,
                                                                                                 args.dataset)
    # args.patient_list_file = r'../data/recover/output/{}/patient_covid_lab_{}.pkl'.format(args.dataset, args.dataset)

    args.input_file = r'{}.encounter'.format(args.dataset)
    args.output_file = r'../data/recover/output/{}/encounter_{}.pkl'.format(args.dataset, args.dataset)
    print('args:', args)
    return args


def read_encounter(input_file, output_file='', selected_patients={}):
    """
    :param data_file: input encounter file with std format
    :param out_file: output id_code-list[patid] = [(time, enc_type), ...] pickle sorted by time, and de-duplicates
    :return: id_code-list[patid] = [(time, enc_type), ...]  sorted by time, and de-duplicates
    :Notice:
        1.COL data: e.g:
        df.shape: (4002975, 31)

        2. WCM data: e.g.
        df.shape: (66362203, 31)

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

    id_enc = defaultdict(list)
    i = 0
    n_rows = 0
    dfs = []
    n_no_date = 0
    n_no_type = 0
    n_no_source = 0
    n_discard_row = 0
    n_recorded_row = 0
    n_not_in_list_row = 0
    for chunk in tqdm(pd.read_sql(sql_query, connection, chunksize=chunksize), total=n_chunk, mininterval=5):
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
            admit_date = row["ADMIT_DATE"]
            admit_source = row['ADMITTING_SOURCE']
            # didn't store discharge date. Currently only use encounter type for major focus
            dis_date = row["DISCHARGE_DATE"]
            drg = row['DRG']
            drg_type = row['DRG_TYPE']

            if pd.isna(admit_date):
                n_no_date += 1
            if pd.isna(enc_type):
                n_no_type += 1
            if pd.isna(admit_source):
                n_no_source += 1

            if pd.isna(admit_date):
                n_discard_row += 1
            else:
                if not selected_patients:
                    id_enc[patid].append((admit_date, enc_type, enc_id, drg, drg_type))
                    n_recorded_row += 1
                else:
                    if patid in selected_patients:
                        id_enc[patid].append((admit_date, enc_type, enc_id, drg, drg_type))
                        n_recorded_row += 1
                    else:
                        n_not_in_list_row += 1

        dfs.append(chunk[["ADMIT_DATE"]])

        if i % 25 == 0:
            print('chunk:', i, 'len(dfs):', len(dfs),
                  'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('n_rows:', n_rows, 'n_no_date:', n_no_date, 'n_no_type:', n_no_type, 'n_no_source:', n_no_source,
                  'n_discard_row:', n_discard_row,
                  'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('n_rows:', n_rows, '#chunk: ', i, 'chunk size:', chunksize)
    print('n_no_date:', n_no_date, 'n_no_type:', n_no_type, 'n_no_source:', n_no_source,
          'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:', n_not_in_list_row)

    print('len(id_enc):', len(id_enc))
    dfs = pd.concat(dfs)
    print('dfs.shape', dfs.shape)
    print('Time range of encounter table  of selected patients [ADMIT_DATE]:',
          pd.to_datetime(dfs["ADMIT_DATE"]).describe(datetime_is_numeric=True))

    # sort
    print('sort encounter list in id_enc by time')
    for patid, records in id_enc.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        records_sorted_list = sorted(set(records), key=lambda x: x[0])
        id_enc[patid] = records_sorted_list

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if output_file:
        utils.check_and_mkdir(output_file)
        # pickle.dump(id_enc, open(output_file, 'wb'))
        utils.dump_compressed(id_enc, output_file)  # use compreseed dump 2025 Q2, file name, not file handle
        # dfs.to_csv(output_file.replace('.pkl', '') + '.csv')
        print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_enc


if __name__ == '__main__':
    # python pre_encounter.py --dataset COL 2>&1 | tee  log/pre_encounter_COL.txt
    # python pre_encounter.py --dataset WCM 2>&1 | tee  log/pre_encounter_WCM.txt
    # python pre_encounter.py --dataset NYU 2>&1 | tee  log/pre_encounter_NYU.txt
    # python pre_encounter.py --dataset MONTE 2>&1 | tee  log/pre_encounter_MONTE.txt
    # python pre_encounter.py --dataset MSHS 2>&1 | tee  log/pre_encounter_MSHS.txt

    start_time = time.time()
    args = parse_args()
    with open(args.patient_list_file, 'rb') as f:
        selected_patients = pickle.load(f)
        print('len(selected_patients):', len(selected_patients))

    id_enc = read_encounter(args.input_file, args.output_file, selected_patients)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
