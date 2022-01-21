import sys
# for linux env.
sys.path.insert(0,'..')
import pandas as pd
import time
import pickle
import argparse
from misc import utils
import numpy as np
import functools
from collections import Counter
print = functools.partial(print, flush=True)
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='COL', help='input dataset')
    args = parser.parse_args()
    if args.dataset == 'COL':
        args.input_file = r'../data/V15_COVID19/output/covid_lab_COL.csv' # xlsx
        args.output_file = r'../data/V15_COVID19/output/patient_covid_lab_COL.pkl'
        args.demo_file = r'../data/V15_COVID19/output/patient_demo_COL.pkl'
    elif args.dataset == 'WCM':
        args.input_file = r'../data/V15_COVID19/output/covid_lab_WCM.csv'  # xlsx
        args.output_file = r'../data/V15_COVID19/output/patient_covid_lab_WCM.pkl'
        args.demo_file = r'../data/V15_COVID19/output/patient_demo_WCM.pkl'

    print('args:', args)
    return args


def read_covid_lab_and_generate_label(input_file, output_file='', id_demo={}):
    """
    :param data_file: input demographics file with std format
    :param out_file: output id_covidlab[patid] = [(time, code, label, age), ...] sorted by time,  pickle
    :return: id_covidlab[patid] = [(time, code, label, age), ...] sorted by time
    :Notice:
        1.COL data: e.g:
        df.shape: (372633, 36)

        2. WCM data: e.g.
        df.shape:

    """
    start_time = time.time()
    # df = pd.read_excel(input_file, sheet_name='Sheet1',
    #                    dtype=str,
    #                    parse_dates=['SPECIMEN_DATE', "RESULT_DATE"])
    df = pd.read_csv(input_file, dtype=str, parse_dates=['SPECIMEN_DATE', "RESULT_DATE"])
    df.rename(columns=lambda x: x.upper(), inplace=True)
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    # df_sub = df[['PATID', 'RESULT_DATE', 'LAB_LOINC', 'RESULT_QUAL']]  # use specimen_date?
    # records_list = df_sub.values.tolist()
    id_lab = defaultdict(list)  # {x[0]: x[1:] for x in records_list}
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0
    i = 0
    for index, row in df.iterrows():
        i+=1
        patid = row['PATID']
        lab_date = row["RESULT_DATE"]  # dx_date may be null. no imputation. If there is no date, not recording
        lab_code = row['LAB_LOINC']
        result_label = row['RESULT_QUAL'].upper()

        if pd.isna(lab_code):
            n_no_dx += 1
        if pd.isna(lab_date):
            n_no_date += 1

        if pd.isna(lab_code) or pd.isna(lab_date):
            n_discard_row += 1
        else:
            if patid in id_demo:
                age = (lab_date - id_demo[patid][0]).days // 365
            else:
                print('No age information for:', patid)
                age = np.nan

            id_lab[patid].append((lab_date, lab_code, result_label, age))
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row)
    # sort
    print('sort lab list in id_lab by time')
    for patid, lab_list in id_lab.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        lab_list_sorted = sorted(set(lab_list), key=lambda x: x[0])
        id_lab[patid] = lab_list_sorted
    print('len(id_lab):', len(id_lab))

    # add more information to df
    print('add more information to excel')
    df = df.sort_values(by=['PATID', 'RESULT_DATE'])
    df['n_test'] = df['PATID'].apply(lambda x: len(id_lab[x]))
    df['covid_positive'] = df['PATID'].apply(lambda x: 'POSITIVE' in [a[2].upper() for a in id_lab[x]])
    # Do we need a better definition considering NI?
    if id_demo:
        df['age'] = np.nan
        for index, row in df.iterrows():
            patid = row['PATID']
            lab_date = row["RESULT_DATE"]  # dx_date may be null. no imputation. If there is no date, not recording
            if patid in id_demo:
                df.loc[index, "age"] = (lab_date - id_demo[patid][0]).days // 365
                # lab_date.year - id_demo[patid][0].year

    # df['covid_positive'] = df['PATID'].apply(lambda x: any(b in [a[-1] for a in id_lab[x]] for b in ['POSITIVE',
    # 'positive'])) any(b in [a[-1] for a in id_lab[x]] for b in ['POSITIVE', 'positive']) {'POSITIVE',
    # 'positive'}.issubset([a[-1] for a in id_lab[x]])
    if output_file:
        utils.check_and_mkdir(output_file)
        pickle.dump(id_lab, open(output_file, 'wb'))
        df.to_excel(output_file.replace('.pkl', '') + '.xlsx')
        print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_lab, df


def data_analysis(id_lab):
    cnt = []
    for k, v in id_lab.items():
        cnt.append(len(v))
    pd.DataFrame(cnt).describe()
    c = Counter(cnt)
    df = pd.DataFrame(c.most_common(), columns=['covid_test_count', 'num_of_people'])
    df.to_excel(r'../data/V15_COVID19/output/test_frequency_COL.xlsx')


if __name__ == '__main__':
    # python pre_covid_lab.py --dataset COL 2>&1 | tee  log/pre_covid_lab_COL.txt
    # python pre_covid_lab.py --dataset WCM 2>&1 | tee  log/pre_covid_lab_WCM.txt
    start_time = time.time()
    args = parse_args()
    with open(args.demo_file, 'rb') as f:
        # to add age information for each covid tests
        id_demo = pickle.load(f)
        print('load demo information: len(id_demo):', len(id_demo))

    id_lab, df = read_covid_lab_and_generate_label(args.input_file, args.output_file, id_demo)
    # patient_dates = build_patient_dates(args.demo_file, args.dx_file, r'output/patient_dates.pkl')
    print('#positive:', len(df.loc[df['covid_positive'], 'PATID'].unique()))
    print('#negative:', len(df.loc[~df['covid_positive'], 'PATID'].unique()))
    print('total:', len(df.loc[:, 'PATID'].unique()))
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
