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
from collections import defaultdict
from misc.utilsql import *
import pickle


# trying to replace pre_ckd_lab.py, dump csv and id_lab pickle
# the dumped csv files are the same as pre_ckd_lab.py results if use the same loinc codelist

def parse_args():
    parser = argparse.ArgumentParser(description='extract selected lab results')
    parser.add_argument('--dataset', default='wcm', help='all recover sites')
    # parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    args.patient_list_file = r'../data/recover/output/{}/patient_covid_lab-dx-med-preg_{}.pkl.gz'.format(args.dataset,
                                                                                                 args.dataset)
    args.input_file = r'{}.lab_result_cm'.format(args.dataset)
    args.output_csv_file = r'../data/recover/output/{}/lab_result_select_{}.csv'.format(args.dataset, args.dataset)
    args.output_pkl_file = r'../data/recover/output/{}/lab_result_select_{}.pkl.gz'.format(args.dataset, args.dataset)

    print('args:', args)

    return args


def read_lab_result(args, code_set, selected_patients={}):
    start_time = time.time()
    chunksize = 100000

    print('in read_lab_result')
    print('Choose dataset:', args.dataset, 'chunksize:', chunksize, )
    if selected_patients:
        print('using selected_patients, len(selected_patients):', len(selected_patients))

    # read lab results by chunk, due to large file size
    print('Read lab data from selected patients who selected by COVID CP and selected lab loinc records')
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

    # dfs = []  # holds data chunks
    dfs_covid_ckd = []
    cnt = Counter([])
    cnt_code = Counter([])
    n_covid_ckd_rows = 0
    patid_set = set([])
    patid_covid_ckd_set = set([])

    id_lab = defaultdict(list)
    i = 0
    n_rows = 0
    n_no_loinc = 0
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

        # part 1: selected all records as chunk, and dumped as csv for furture analyses
        if selected_patients:
            chunk_covid_ckd_records = chunk.loc[
                                      chunk['LAB_LOINC'].isin(code_set) & chunk['PATID'].isin(selected_patients), :]
        else:
            chunk_covid_ckd_records = chunk.loc[chunk['LAB_LOINC'].isin(code_set), :]

        dfs_covid_ckd.append(chunk_covid_ckd_records)

        patid_set.update(chunk['PATID'])
        patid_covid_ckd_set.update(chunk_covid_ckd_records['PATID'])
        n_covid_ckd_rows += len(chunk_covid_ckd_records)

        cnt.update(chunk_covid_ckd_records['RESULT_QUAL'])
        cnt_code.update(chunk_covid_ckd_records['LAB_LOINC'])

        # part 2: selected per row as a data structure
        # to use already selectec chunck chunk_covid_ckd_records to speed up time
        for index, row in chunk_covid_ckd_records.iterrows():
            patid = row['PATID']
            enc_id = row['ENCOUNTERID']
            loinc = row['LAB_LOINC']

            order_date = row["LAB_ORDER_DATE"]  #
            spe_date = row["SPECIMEN_DATE"]
            lab_date = row["RESULT_DATE"]

            result_qual = row['RESULT_QUAL']
            result_num = row['RESULT_NUM']
            result_unit = row['RESULT_UNIT']

            if pd.isna(loinc):
                n_no_loinc += 1

            if pd.isna(lab_date):
                if pd.notna(spe_date):
                    lab_date = spe_date
                elif pd.notna(order_date):
                    lab_date = order_date
                else:
                    n_no_date += 1

            if pd.isna(loinc) or pd.isna(lab_date):
                n_discard_row += 1
            else:
                if not selected_patients:
                    id_lab[patid].append((lab_date, loinc, result_qual, result_num, result_unit, enc_id))
                    n_recorded_row += 1
                else:
                    if patid in selected_patients:
                        id_lab[patid].append((lab_date, loinc, result_qual, result_num, result_unit, enc_id))
                        n_recorded_row += 1
                    else:
                        n_not_in_list_row += 1

        if i % 50 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('n_rows:', n_rows,
                  'n_no_loinc:', n_no_loinc, 'n_no_date:', n_no_date,
                  'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:',
                  n_not_in_list_row)
            print('len(patid_set):', len(patid_set))
            print('len(chunk_covid_ckd_records):', len(chunk_covid_ckd_records))

    print('read database done!')
    print('n_rows:', n_rows, 'len(id_lab):', len(id_lab),
          'n_no_loinc:', n_no_loinc, 'n_no_date:', n_no_date,
          'n_discard_row:', n_discard_row, 'n_recorded_row:', n_recorded_row, 'n_not_in_list_row:',
          n_not_in_list_row)

    print('n_covid_ckd_rows:', n_covid_ckd_rows)
    print('len(patid_set):', len(patid_set))
    print('len(patid_covid_ckd_set):', len(patid_covid_ckd_set))
    print('#chunk: ', i, 'chunk size:', chunksize)
    print('Counter:', cnt)
    print('Loinc Counter:', cnt_code)

    # sort and de-duplicates
    print('sort lab list in id_lab by time')
    for patid, lab_list in id_lab.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        lab_list_sorted = sorted(set(lab_list), key=lambda x: x[0])
        id_lab[patid] = lab_list_sorted
    print('sort lab list done, total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print('Dump selected lab results to csv:', args.output_csv_file)
    # dump all selected lab result as csv, all columns as it is
    dfs_covid_ckd_all = pd.concat(dfs_covid_ckd)
    print('dfs_covid_ckd_all.shape', dfs_covid_ckd_all.shape)
    print('dfs_covid_ckd_all.columns', dfs_covid_ckd_all.columns)

    print('Output file:', args.output_csv_file)
    utils.check_and_mkdir(args.output_csv_file)
    dfs_covid_ckd_all.to_csv(args.output_csv_file, index=False)

    print('Dump id_lab to {}'.format(args.output_pkl_file))
    utils.check_and_mkdir(args.output_pkl_file)
    utils.dump_compressed(id_lab, args.output_pkl_file)

    connection.close()
    print('Total Time used after dump files:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_lab, dfs_covid_ckd_all


if __name__ == '__main__':
    # python pre_lab_4covid.py --dataset covid_database 2>&1 | tee  log/pre_lab_COL.txt
    # python pre_lab_4covid.py --dataset main_database 2>&1 | tee  log/pre_lab_WCM.txt
    # python pre_lab_4covid.py --dataset wcm 2>&1 | tee  log\pre_lab_WCM.txt
    start_time = time.time()
    args = parse_args()
    print(args)
    print('args.dataset:', args.dataset)
    # step 1: select targeted lab (loinc) code list
    # here only use ckd related lab result. might add or change to others, say hba1c etc
    df_ckd1 = pd.read_excel(r'../data/mapping/ckd_codes_revised.xlsx', sheet_name=r'serum creatinine')
    df_ckd2 = pd.read_excel(r'../data/mapping/ckd_codes_revised.xlsx', sheet_name=r'eGFR')

    print('serum creatinine df_ckd1.shape:', df_ckd1.shape, 'serum creatinine df_ckd2.shape:', df_ckd2.shape)
    code_set1 = set(df_ckd1['code'].to_list())
    code_set2 = set(df_ckd2['code'].to_list())
    code_set = code_set1.union(code_set2)
    print('Selected all ckd related codes: ', len(code_set), 'where serum creatinine: ', len(code_set1),
          'where eGFR: ', len(code_set2))

    # step 2: select targeted patient list
    # with open(args.patient_list_file, 'rb') as f:
    #     selected_patients = pickle.load(f)
    #     print('len(selected_patients):', len(selected_patients))

    selected_patients = utils.load(args.patient_list_file)
    print('len(selected_patients):', len(selected_patients))

    # step 3: extract lab result
    id_lab, dfs_covid_ckd_all = read_lab_result(args, code_set, selected_patients=selected_patients)
    print('Total Time used after dump files:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
