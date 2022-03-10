import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import time
import pickle
import argparse
from misc import utils
import numpy as np
import functools
from collections import Counter
from tqdm import tqdm

print = functools.partial(print, flush=True)
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess covid lab')
    parser.add_argument('--dataset', choices=['all'],  # 'covid_database', 'main_database',
                        default='all', help='combined datasets')
    args = parser.parse_args()

    args.input_file1 = r'../data/oneflorida/output/covid_database/covid_lab_covid_database.csv'
    args.input_file2 = r'../data/oneflorida/output/main_database/covid_lab_main_database.csv'

    args.demo_file = r'../data/oneflorida/output/{}/patient_demo_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file = r'../data/oneflorida/output/{}/patient_covid_lab_{}.pkl'.format(args.dataset, args.dataset)

    print('args:', args)
    return args


def read_covid_lab_and_generate_label(args, output_file='', id_demo={}):
    """
    1. scan date range of all covid-related tests
    2. selected only PCR-tested records
    3. build id --> index information
    [4.] save encounter id for check covid test encounter type later in the cohort build
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
    print('Step 1: load selected and existed covid PCR/Antigen lab code')
    df_covid_list = pd.read_csv(r'../data/V15_COVID19/covid_phenotype/COVID_LOINC_all.csv')
    """
        loinc_num	component	type	COL Frequency	WCM Frequency	MONTE Frequency	NYU Frequency	MSHS Frequency	Total Orders
        94309-2	SARS coronavirus 2 RNA	Molecular			385168	762104	21518	1168790
        94500-6	SARS coronavirus 2 RNA	Molecular	336953	508395		40423	88057	973828
        94306-8	SARS coronavirus 2 RNA panel	Molecular					569015	569015
        94746-5	SARS coronavirus 2 RNA	Molecular				559367		559367
        (very few, ignore currently) 94558-4	SARS coronavirus 2 Ag	Antigen	56	120		5828		6004
        94759-8	SARS coronavirus 2 RNA	Molecular		8			448	456
        """
    # updated 2022-02-02, also include antigen, to be consistent with CDC cohorts
    # even in RWD, they are small
    df_covid_pcr_list = df_covid_list.loc[
                        (df_covid_list['type'].isin(['Molecular', "Antigen"])),
                        :]  # & (df_covid_list['Total Orders'] > 0)
    pd.set_option('display.max_columns', None)
    print(df_covid_pcr_list)
    code_set = set(df_covid_pcr_list['loinc_num'].to_list())
    print('Selected and existed Covid PCR/Antigen codes:', code_set)
    # df = pd.read_excel(input_file, sheet_name='Sheet1',
    #                    dtype=str,
    #                    parse_dates=['SPECIMEN_DATE', "RESULT_DATE"])

    df1 = pd.read_csv(args.input_file1, dtype=str, parse_dates=['SPECIMEN_DATE', "RESULT_DATE", 'LAB_ORDER_DATE'])
    df1.rename(columns=lambda x: x.upper(), inplace=True)
    print('df1.shape', df1.shape)
    print('df1.columns', df1.columns)
    print('Unique patid:', len(df1['PATID'].unique()))

    df2 = pd.read_csv(args.input_file2, dtype=str, parse_dates=['SPECIMEN_DATE', "RESULT_DATE", 'LAB_ORDER_DATE'])
    df2.rename(columns=lambda x: x.upper(), inplace=True)
    print('df2.shape', df2.shape)
    print('df2.columns', df2.columns)
    print('Unique patid:', len(df2['PATID'].unique()))

    df = pd.concat([df1, df2])
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    print('Unique patid:', len(df['PATID'].unique()))

    print('Time range of All Covid Test:', df["RESULT_DATE"].describe(datetime_is_numeric=True))
    print('PCR/Antigen/Antibody LAB_LOINC Count:', df["LAB_LOINC"].describe())
    print('PCR/Antigen/Antibody RESULT_QUAL Count:', df["RESULT_QUAL"].describe())

    print('.........Select PCR/Antigen patients:...........')
    df_covid = df.loc[df['LAB_LOINC'].isin(code_set), :]
    print('PCR/Antigen tested df_covid.shape', df_covid.shape)
    print('Unique patid of PCR/Antigen test:', len(df_covid['PATID'].unique()))
    print('Time range of PCR/Antigen Covid Test:', df_covid["RESULT_DATE"].describe(datetime_is_numeric=True))
    print('PCR/Antigen LAB_LOINC Count:', df_covid["LAB_LOINC"].describe())
    print('PCR/Antigen RESULT_QUAL Count:', df_covid["RESULT_QUAL"].describe())

    id_lab = defaultdict(list)
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0
    i = 0
    for index, row in tqdm(df_covid.iterrows(), total=len(df_covid)):
        i += 1
        patid = row['PATID']
        lab_date = row["RESULT_DATE"]
        specimen_date = row['SPECIMEN_DATE']
        lab_code = row['LAB_LOINC']
        result_label = row['RESULT_QUAL'].upper()
        enc_id = row['ENCOUNTERID']

        if pd.isna(lab_code):
            n_no_dx += 1

        # If there is no lab_date, using specimen_date, if also no specimen date, not recording
        if pd.isna(lab_date):
            if pd.isna(specimen_date):
                n_no_date += 1
            else:
                lab_date = specimen_date

        if pd.isna(lab_code) or pd.isna(lab_date):
            n_discard_row += 1
        else:
            if patid in id_demo:
                age = (lab_date - id_demo[patid][0]).days // 365
            else:
                print('No age information for:', patid)
                age = np.nan

            id_lab[patid].append((lab_date, lab_code, result_label, age, enc_id))  # newly added enc_id for future use
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'len(id_lab):', len(id_lab))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # sort
    print('Sort lab lists in id_lab by time')
    for patid, lab_list in id_lab.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        lab_list_sorted = sorted(set(lab_list), key=lambda x: x[0])
        id_lab[patid] = lab_list_sorted
    print('len(id_lab):', len(id_lab))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # Add more information to df, just for debugging.
    # Formal positive and negative definition in pre_cohort.py
    print('Add more information to csv for debugging purpose')
    df_covid = df_covid.sort_values(by=['PATID', 'RESULT_DATE'])
    df_covid['n_test'] = df_covid['PATID'].apply(lambda x: len(id_lab[x]))
    df_covid['covid_positive'] = df_covid['PATID'].apply(lambda x: 'POSITIVE' in [a[2].upper() for a in id_lab[x]])
    # Do we need a better definition considering NI?
    # yes. we should exclude NI from negative cases.
    if id_demo:
        df_covid['age'] = np.nan
        age_list = []
        n_no_age = 0
        for index, row in tqdm(df_covid.iterrows(), total=len(df_covid)):
            patid = row['PATID']
            lab_date = row["RESULT_DATE"]  # dx_date may be null. no imputation. If there is no date, not recording
            if patid in id_demo:
                # df_covid.loc[index, "age"] = (lab_date - id_demo[patid][0]).days // 365
                age = (lab_date - id_demo[patid][0]).days // 365
            else:
                age = np.nan
                n_no_age += 1
            age_list.append(age)
        df_covid['age'] = age_list
        print('n_no_age:', n_no_age, r"len(df_covid['age'])", len(df_covid['age']))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    if output_file:
        print('Dump file:', output_file)
        utils.check_and_mkdir(output_file)
        pickle.dump(id_lab, open(output_file, 'wb'))
        df_covid.to_csv(output_file.replace('.pkl', '') + '.csv')
        print('dump done to {}'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_lab, df_covid, df


def data_analysis(id_lab):
    cnt = []
    for k, v in id_lab.items():
        cnt.append(len(v))
    pd.DataFrame(cnt).describe()
    c = Counter(cnt)
    df = pd.DataFrame(c.most_common(), columns=['covid_test_count', 'num_of_people'])
    df.to_excel(r'../data/V15_COVID19/output/test_frequency_COL.xlsx')


if __name__ == '__main__':
    # python pre_covid_lab.py --dataset all 2>&1 | tee  log/pre_covid_lab_allcombined.txt

    start_time = time.time()
    args = parse_args()
    with open(args.demo_file, 'rb') as f:
        # to add age information for each covid tests
        id_demo = pickle.load(f)
        print('load', args.demo_file, 'demo information: len(id_demo):', len(id_demo))

    id_lab, df_pcr, df = read_covid_lab_and_generate_label(args, args.output_file, id_demo)
    print('PCR+Antigen+Antibody-test #total:', len(df.loc[:, 'PATID'].unique()))
    print('PCR/Antigen-test #total:', len(df_pcr.loc[:, 'PATID'].unique()))
    print('PCR/Antigen-test #positive:', len(df_pcr.loc[df_pcr['covid_positive'], 'PATID'].unique()))
    print('PCR/Antigen-test #negative:', len(df_pcr.loc[~df_pcr['covid_positive'], 'PATID'].unique()))
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
