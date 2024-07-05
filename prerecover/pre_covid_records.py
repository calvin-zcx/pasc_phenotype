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
from misc.utils import clean_date_str


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--dataset', default='utah_pcornet_all', help='site dataset')
    args = parser.parse_args()

    args.input_lab = r'../data/recover/output/{}/covid_lab_{}.csv'.format(args.dataset, args.dataset)
    args.input_diagnosis = r'../data/recover/output/{}/covid_diagnosis_{}.csv'.format(args.dataset, args.dataset)
    args.input_prescribing = r'../data/recover/output/{}/covid_prescribing_{}.csv'.format(args.dataset, args.dataset)
    args.input_med_admin = r'../data/recover/output/{}/covid_med_admin_{}.csv'.format(args.dataset, args.dataset)
    args.input_dispensing = r'../data/recover/output/{}/covid_dispensing_{}.csv'.format(args.dataset, args.dataset)

    args.preg_diagnosis = r'../data/recover/output/{}/pregnant_diagnosis_{}.csv'.format(args.dataset, args.dataset)
    args.preg_procedure = r'../data/recover/output/{}/pregnant_procedures_{}.csv'.format(args.dataset, args.dataset)
    args.preg_encounter = r'../data/recover/output/{}/pregnant_encounter_{}.csv'.format(args.dataset, args.dataset)

    args.demo_file = r'../data/recover/output/{}/patient_demo_{}.pkl'.format(args.dataset, args.dataset)

    args.output_file_lab = r'../data/recover/output/{}/patient_covid_lab_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file_labdx = r'../data/recover/output/{}/patient_covid_lab-dx_{}.pkl'.format(args.dataset, args.dataset)
    args.output_file_labdxmed = r'../data/recover/output/{}/patient_covid_lab-dx-med_{}.pkl'.format(
        args.dataset, args.dataset)
    args.output_file_labdxmedpreg = r'../data/recover/output/{}/patient_covid_lab-dx-med-preg_{}.pkl'.format(
        args.dataset, args.dataset)

    print('args:', args)
    return args


def read_covid_lab_and_generate_label(input_file, output_file='', id_demo={}):
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
    print('In read_covid_lab_and_generate_label, input_file:', input_file)

    print('Step 1: load selected and existed covid PCR/Antigen lab code')
    df_covid_list = pd.read_csv(r'../data/V15_COVID19/covid_phenotype/COVID_LOINC_all.csv', dtype=str)
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
    # df_covid_pcr_list = df_covid_list.loc[
    #                     (df_covid_list['type'].isin(['Molecular', "Antigen"])) & (df_covid_list['Total Orders'] > 0),
    #                     :]
    # Changed 2022-10-20, no need for count >0, current data were beyond INSIGHT
    df_covid_pcr_list = df_covid_list.loc[(df_covid_list['type'].isin(['Molecular', "Antigen"])), :]
    pd.set_option('display.max_columns', None)
    print(df_covid_pcr_list)
    code_set = set(df_covid_pcr_list['loinc_num'].to_list())
    print('Selected and existed Covid PCR/Antigen codes, len(code_set):', len(code_set), code_set)
    # df = pd.read_excel(input_file, sheet_name='Sheet1',
    #                    dtype=str,
    #                    parse_dates=['SPECIMEN_DATE', "RESULT_DATE"])

    df = pd.read_csv(input_file, dtype=str, parse_dates=['SPECIMEN_DATE', "RESULT_DATE"])
    df.rename(columns=lambda x: x.upper(), inplace=True)
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    print('Unique patid:', len(df['PATID'].unique()))
    print('Time range of All Covid Test:', df["RESULT_DATE"].describe(datetime_is_numeric=True))

    print('.........Select PCR/Antigen patients:...........')
    df_covid = df.loc[df['LAB_LOINC'].isin(code_set), :]
    print('PCR/Antigen tested df_covid.shape', df_covid.shape)
    print('Unique patid of PCR/Antigen test:', len(df_covid['PATID'].unique()), 'in all covid-realted test patients',
          len(df['PATID'].unique()))
    print('Time range of PCR/Antigen Covid Test:', df_covid["RESULT_DATE"].describe(datetime_is_numeric=True))

    # check covid test result value:
    print('\nRESULT_QUAL:')
    print(df_covid['RESULT_QUAL'].value_counts(dropna=False))

    if 'RAW_RESULT' in df_covid.columns:
        print('\nRAW_RESULT:')
        print(df_covid['RAW_RESULT'].value_counts(dropna=False))

    if 'RESULT_TEXT' in df_covid.columns:
        print('\nRESULT_TEXT:')
        print(df_covid['RESULT_TEXT'].value_counts(dropna=False))

    id_lab = defaultdict(list)
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0
    i = 0
    n_no_dob_row = 0
    for index, row in tqdm(df_covid.iterrows(), total=len(df_covid), mininterval=5):
        i += 1
        patid = row['PATID']
        lab_date = row["RESULT_DATE"]
        specimen_date = row['SPECIMEN_DATE']
        lab_code = row['LAB_LOINC']
        result_label = row['RESULT_QUAL']  # .upper()  # need to impute this, e.g., vumc
        enc_id = row['ENCOUNTERID']

        if pd.isna(lab_code):
            n_no_dx += 1

        # 2022-10-25, case beyond insight, e.g. vumc
        # 2022-11-17 mshs also showed inconsistency
        #            just compare 'NI', just for insight, try not too broad as the following function
        # def _result_value_need_impute(_x):
        #     if isinstance(_x, str):
        #         _x = _x.strip().upper()
        #     return not _x.startswith(
        #         ('NOT DETECTED', 'NEG', 'NOT', 'NEGATIVE', 'UNDETECTED', 'DETECTED', 'POSITIVE', 'POS',
        #          'PRESUMPTIVE', 'INVALID', 'INCONCLUSIVE')
        #     )

        if pd.isna(result_label) or (result_label == 'NI'):
            if 'RAW_RESULT' in row.index:
                result_label = row['RAW_RESULT']

        if pd.isna(result_label) or (result_label == 'NI'):
            if 'RESULT_TEXT' in row.index:
                result_label = row['RESULT_TEXT']

        if pd.isna(result_label):
            result_label = 'NI'

        if isinstance(result_label, str):
            result_label = result_label.strip().upper()

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
                # updated 2022-10-26.
                if pd.notna(id_demo[patid][0]):
                    age = (lab_date - pd.to_datetime(id_demo[patid][0])).days / 365
                else:
                    age = np.nan
            else:
                print('No age information for:', patid, lab_date, specimen_date, lab_code, result_label)
                age = np.nan
                n_no_dob_row += 1

            # newly added enc_id and '' encounter type for future use
            # (date, code, result_label,  age,  enc_id, encounter_type, CP-method )
            id_lab[patid].append((lab_date, lab_code, result_label, age, enc_id, '', 'lab'))
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_no_dob_row:', n_no_dob_row, 'len(id_lab):', len(id_lab))
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
        for index, row in tqdm(df_covid.iterrows(), total=len(df_covid), mininterval=10):
            patid = row['PATID']
            lab_date = row["RESULT_DATE"]  # dx_date may be null. no imputation. If there is no date, not recording
            if patid in id_demo:
                # df_covid.loc[index, "age"] = (lab_date - id_demo[patid][0]).days // 365
                # age = (lab_date - pd.to_datetime(id_demo[patid][0])).days / 365  # (lab_date - id_demo[patid][0]).days // 365
                if pd.notna(id_demo[patid][0]):
                    age = (lab_date - pd.to_datetime(
                        id_demo[patid][0])).days / 365  # (lab_date - id_demo[patid][0]).days // 365
                else:
                    age = np.nan
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


def read_covid_diagnosis(input_file, id_demo):
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
    print('In read_covid_diagnosis, input_file:', input_file)
    df = pd.read_csv(input_file, dtype=str, parse_dates=['ADMIT_DATE', "DX_DATE"])
    df.rename(columns=lambda x: x.upper(), inplace=True)
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    print('Unique patid:', len(df['PATID'].unique()))
    print('Time range of All Covid dx:', df["ADMIT_DATE"].describe(datetime_is_numeric=True))

    id_dx = defaultdict(list)
    i = 0
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0
    n_no_dob_row = 0

    for index, row in tqdm(df.iterrows(), total=len(df), mininterval=5):
        i += 1
        patid = row['PATID']
        enc_id = row['ENCOUNTERID']
        enc_type = row['ENC_TYPE']
        dx = row['DX']
        dx_type = row["DX_TYPE"]
        dx_date = row["ADMIT_DATE"]  # dx_date may be null. no imputation. If there is no date, not recording

        if pd.isna(dx):
            n_no_dx += 1
        if pd.isna(dx_date):
            n_no_date += 1

        if pd.isna(dx) or pd.isna(dx_date):
            n_discard_row += 1
        else:
            if patid in id_demo:
                # updated 2022-10-26.
                if pd.notna(id_demo[patid][0]):
                    age = (dx_date - pd.to_datetime(
                        id_demo[patid][0])).days / 365  # (lab_date - id_demo[patid][0]).days // 365
                else:
                    age = np.nan
            else:
                age = np.nan
                n_no_dob_row += 1
                print(n_no_dob_row, 'No age information for:', index, i, patid, dx_date, )

            # (date, code, result_label,  age,  enc_id, encounter_type, CP-method )
            id_dx[patid].append((dx_date, dx, 'Positive', age, enc_id, enc_type, 'dx'))
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_no_dob_row:', n_no_dob_row, 'len(id_dx):', len(id_dx))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # sort and de-duplicates
    print('sort dx list in id_dx by time')
    for patid, dx_list in id_dx.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        dx_list_sorted = sorted(set(dx_list), key=lambda x: x[0])
        id_dx[patid] = dx_list_sorted
    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_dx, df


def read_covid_prescribing(input_file, id_demo):
    """
    :param selected_patients:
    :param output_file:
    :param input_file: input prescribing file with std format
    :param out_file: output id_code-list[patid] = [(start_time, rxnorm, days), ...]  sorted by start_time pickle
    :return: id_code-list[patid] = [(start_time, rxnorm, days), ...]  sorted by start_time pickle
    :Notice:
        discard rows with NULL start_date or rxnorm
        1.COL data: e.g:
        df.shape: (3119792, 32)

        2. WCM data: e.g.
        df.shape: (48987841, 32)
    """
    start_time = time.time()
    print('In read_covid_prescribing, input_file:', input_file)
    try:
        df = pd.read_csv(input_file, dtype=str, parse_dates=['RX_ORDER_DATE', 'RX_START_DATE', 'RX_END_DATE'])
    except Exception as e:
        print(e)
        return defaultdict(list), None

    # df.rename(columns=lambda x: x.upper(), inplace=True)
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    print('Unique patid:', len(df['PATID'].unique()))
    print('Time range of All Covid dx:', df["RX_START_DATE"].describe(datetime_is_numeric=True))

    id_med = defaultdict(list)
    i = 0
    # n_rows = 0
    # dfs = []

    n_no_rxnorm = 0
    n_no_date = 0
    n_no_days_supply = 0

    n_discard_row = 0
    n_recorded_row = 0
    # n_not_in_list_row = 0
    n_no_dob_row = 0

    for index, row in tqdm(df.iterrows(), total=len(df), mininterval=5):
        i += 1
        patid = row['PATID']
        rx_order_date = row['RX_ORDER_DATE']
        rx_start_date = row['RX_START_DATE']
        rx_end_date = row['RX_END_DATE']
        rxnorm = row['RXNORM_CUI']
        rx_days = row['RX_DAYS_SUPPLY']

        if 'RAW_RXNORM_CUI' in row.index:
            raw_rxnorm = row['RAW_RXNORM_CUI']
        else:
            raw_rxnorm = np.nan

        encid = row['ENCOUNTERID']  # 2022-10-23 ADD encounter id to drug structure

        # start_date
        if pd.notna(rx_start_date):
            start_date = rx_start_date
        elif pd.notna(rx_order_date):
            start_date = rx_order_date
        else:
            start_date = np.nan
            n_no_date += 1

        # rxrnom
        if pd.isna(rxnorm):
            if pd.notna(raw_rxnorm):
                rxnorm = raw_rxnorm
            else:
                n_no_rxnorm += 1
                rxnorm = np.nan

        # days supply
        if pd.notna(rx_days) and rx_days:
            days = int(float(rx_days))
        elif pd.notna(start_date) and pd.notna(rx_end_date):
            days = (rx_end_date - start_date).days + 1
        else:
            days = -1
            n_no_days_supply += 1

        if pd.isna(rxnorm) or pd.isna(start_date):
            n_discard_row += 1
        else:
            if patid in id_demo:
                if pd.notna(id_demo[patid][0]):
                    age = (start_date - pd.to_datetime(id_demo[patid][0])).days / 365
                else:
                    age = np.nan
            else:
                print('No age information for:', index, i, patid, start_date, )
                age = np.nan
                n_no_dob_row += 1

            # (date, code, result_label,  age,  enc_id, encounter_type, CP-method )
            # e.g., nebraska, can records, same day, same rxnorm, but without encid
            id_med[patid].append((start_date, rxnorm, 'Positive', age, encid, '', 'med-pr'))
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_no_dob_row:', n_no_dob_row, 'n_no_days_supply', n_no_days_supply,
          'len(id_med):', len(id_med))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    print('len(id_med):', len(id_med))

    # sort
    print('Sort med list in id_med by time')
    for patid, med_list in id_med.items():
        med_list_sorted = sorted(set(med_list), key=lambda x: x[0])
        id_med[patid] = med_list_sorted

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_med, df


def read_covid_med_admin(input_file, id_demo):
    """
    :param selected_patients:
    :param output_file:
    :param input_file: input med_admin file with std format
    :param out_file: output id_code-list[patid] = [(start_time, rxnorm, days), ...]  sorted by start_time pickle
    :return: id_code-list[patid] = [(start_time, rxnorm, days), ...]  sorted by start_time pickle
    :Notice:
        discard rows with NULL start_date or rxnorm
        Only applied to COL data, no WCM
        1.COL data: e.g:
        df.shape: (6890724, 32)

        2. WCM data: e.g.
        df.shape: (0, 21)
    """
    start_time = time.time()
    print('In read_covid_med_admin, input_file:', input_file)

    try:
        df = pd.read_csv(input_file, dtype=str, parse_dates=['MEDADMIN_START_DATE', 'MEDADMIN_STOP_DATE', ])
    except Exception as e:
        print(e)
        return defaultdict(list), None

    # df.rename(columns=lambda x: x.upper(), inplace=True)
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    print('Unique patid:', len(df['PATID'].unique()))
    print('Time range of All Covid dx:', df["MEDADMIN_START_DATE"].describe(datetime_is_numeric=True))

    id_med = defaultdict(list)
    i = 0
    # n_rows = 0
    # dfs = []

    n_no_rxnorm = 0
    n_no_date = 0
    n_no_days_supply = 0

    n_discard_row = 0
    n_recorded_row = 0
    # n_not_in_list_row = 0
    n_no_dob_row = 0

    for index, row in tqdm(df.iterrows(), total=len(df), mininterval=3):
        i += 1
        patid = row['PATID']
        rx_start_date = row['MEDADMIN_START_DATE']
        rx_end_date = row['MEDADMIN_STOP_DATE']
        med_type = row['MEDADMIN_TYPE']
        rxnorm = row['MEDADMIN_CODE']

        if 'RAW_MEDADMIN_MED_NAME' in row.index:
            names = row['RAW_MEDADMIN_MED_NAME']

        if 'RAW_MEDADMIN_CODE' in row.index:
            raw_rxnorm = row['RAW_MEDADMIN_CODE']
        else:
            raw_rxnorm = np.nan

        encid = row['ENCOUNTERID']  # 2022-10-23 ADD encounter id to drug structure

        # start_date
        if pd.notna(rx_start_date):
            start_date = rx_start_date
        else:
            start_date = np.nan
            n_no_date += 1

        # rxrnom
        # if (med_type != 'RX') or pd.isna(rxnorm):
        #     n_no_rxnorm += 1
        #     rxnorm = np.nan

        if pd.isna(rxnorm):
            if pd.notna(raw_rxnorm):
                rxnorm = raw_rxnorm
            else:
                n_no_rxnorm += 1
                rxnorm = np.nan

        # days supply
        if pd.notna(start_date) and pd.notna(rx_end_date):
            days = (rx_end_date - start_date).days + 1
        else:
            days = -1
            n_no_days_supply += 1

        if pd.isna(rxnorm) or pd.isna(start_date):
            n_discard_row += 1
        else:
            if patid in id_demo:
                if pd.notna(id_demo[patid][0]):
                    age = (start_date - pd.to_datetime(id_demo[patid][0])).days / 365
                else:
                    age = np.nan
            else:
                print('No age information for:', index, i, patid, start_date, )
                age = np.nan
                n_no_dob_row += 1

            id_med[patid].append((start_date, rxnorm, 'Positive', age, encid, '', 'med-medadmin'))
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_no_dob_row:', n_no_dob_row, 'n_no_days_supply', n_no_days_supply,
          'len(id_med):', len(id_med))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    print('len(id_med):', len(id_med))

    # sort
    print('sort med_list in id_med by time')
    for patid, med_list in id_med.items():
        med_list_sorted = sorted(set(med_list), key=lambda x: x[0])
        id_med[patid] = med_list_sorted

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_med, df


def read_covid_dispensing(input_file, id_demo):
    """
    :param selected_patients:
    :param output_file:
    :param input_file: input dispensing file with std format
    :param out_file: output id_code-list[patid] = [(start_time, rxnorm, days), ...]  sorted by start_time pickle
    :return: id_code-list[patid] = [(start_time, rxnorm, days), ...]  sorted by start_time pickle
    :Notice:
        discard rows with NULL start_date or rxnorm
        1.COL data: e.g:
        df.shape: (3119792, 32)

        2. WCM data: e.g.
        df.shape: (48987841, 32)
    """
    start_time = time.time()
    print('In read_covid_dispensing, input_file:', input_file)
    try:
        df = pd.read_csv(input_file, dtype=str, parse_dates=['DISPENSE_DATE', ])
    except Exception as e:
        print(e)
        return defaultdict(list), None

    df.rename(columns=lambda x: x.upper(), inplace=True)
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    print('Unique patid:', len(df['PATID'].unique()))
    print('Time range of All Covid dx:', df["DISPENSE_DATE"].describe(datetime_is_numeric=True))

    id_med = defaultdict(list)
    i = 0
    # n_rows = 0
    # dfs = []

    n_no_rxnorm = 0
    n_no_date = 0
    n_no_days_supply = 0

    n_discard_row = 0
    n_recorded_row = 0
    # n_not_in_list_row = 0
    n_no_dob_row = 0

    for index, row in tqdm(df.iterrows(), total=len(df), mininterval=3):
        i += 1
        patid = row['PATID']
        rx_order_date = row['DISPENSE_DATE']
        # rx_start_date = row['RX_START_DATE']
        # rx_end_date = row['RX_END_DATE']
        rxnorm = row['NDC']
        rx_days = row['DISPENSE_SUP']

        if 'RAW_NDC' in row.index:
            raw_rxnorm = row['RAW_NDC']
        else:
            raw_rxnorm = np.nan

        encid = ''  # row['ENCOUNTERID']  # no enc id in dispense table
        dispense_source = row['DISPENSE_SOURCE']

        # start_date
        if pd.notna(rx_order_date):
            start_date = rx_order_date
        else:
            start_date = np.nan
            n_no_date += 1

        # rxrnom
        if pd.isna(rxnorm):
            if pd.notna(raw_rxnorm):
                rxnorm = raw_rxnorm
            else:
                n_no_rxnorm += 1
                rxnorm = np.nan

        # days supply
        if pd.notna(rx_days) and rx_days:
            days = int(float(rx_days))
        else:
            days = -1
            n_no_days_supply += 1

        if pd.isna(rxnorm) or pd.isna(start_date):
            n_discard_row += 1
        else:
            if patid in id_demo:
                if pd.notna(id_demo[patid][0]):
                    age = (start_date - pd.to_datetime(id_demo[patid][0])).days / 365
                else:
                    age = np.nan
            else:
                print('No age information for:', index, i, patid, start_date, )
                age = np.nan
                n_no_dob_row += 1

            # (date, code, result_label,  age,  enc_id, encounter_type, CP-method )
            # e.g., nebraska, can records, same day, same rxnorm, but without encid
            # no encounter id, thus '', using dispense_source for encounter type
            """ dispense_source: 
            OD=Order/EHR BI=Billing CL=Claim 
            PM=Pharmacy Benefit Manager DR=Derived NI=No information UN=Unknown OT=Other """
            id_med[patid].append((start_date, rxnorm, 'Positive', age, encid, dispense_source, 'med-dispense'))
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_rxnorm:', n_no_rxnorm, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_no_dob_row:', n_no_dob_row, 'n_no_days_supply', n_no_days_supply,
          'len(id_med):', len(id_med))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    print('len(id_med):', len(id_med))

    # sort
    print('Sort med list in id_med by time')
    for patid, med_list in id_med.items():
        med_list_sorted = sorted(set(med_list), key=lambda x: x[0])
        id_med[patid] = med_list_sorted

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_med, df


def read_pregnant_diagnosis(input_file, id_demo):
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
    print('In read_pregnant_diagnosis, input_file:', input_file)
    df = pd.read_csv(input_file, dtype=str, parse_dates=['ADMIT_DATE', "DX_DATE"])
    df.rename(columns=lambda x: x.upper(), inplace=True)
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    print('Unique patid:', len(df['PATID'].unique()))
    print('Time range of All Covid dx:', df["ADMIT_DATE"].describe(datetime_is_numeric=True))

    id_dx = defaultdict(list)
    i = 0
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0
    n_no_dob_row = 0

    for index, row in tqdm(df.iterrows(), total=len(df), mininterval=5):
        i += 1
        patid = row['PATID']
        enc_id = row['ENCOUNTERID']
        enc_type = row['ENC_TYPE']
        dx = row['DX']
        dx_type = row["DX_TYPE"]
        dx_date = row["ADMIT_DATE"]  # dx_date may be null. no imputation. If there is no date, not recording

        if pd.isna(dx):
            n_no_dx += 1
        if pd.isna(dx_date):
            n_no_date += 1

        if pd.isna(dx) or pd.isna(dx_date):
            n_discard_row += 1
        else:
            if patid in id_demo:
                # updated 2022-10-26.
                if pd.notna(id_demo[patid][0]):
                    age = (dx_date - pd.to_datetime(
                        id_demo[patid][0])).days / 365  # (lab_date - id_demo[patid][0]).days // 365
                else:
                    age = np.nan
            else:
                age = np.nan
                n_no_dob_row += 1
                print(n_no_dob_row, 'No age information for:', index, i, patid, dx_date, )

            # (date, code, result_label,  age,  enc_id, encounter_type, CP-method )
            id_dx[patid].append((dx_date, dx, 'Pregnant', age, enc_id, enc_type, 'preg-dx'))
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_no_dob_row:', n_no_dob_row, 'len(id_dx):', len(id_dx))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # sort and de-duplicates
    print('sort dx list in id_dx by time')
    for patid, dx_list in id_dx.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        dx_list_sorted = sorted(set(dx_list), key=lambda x: x[0])
        id_dx[patid] = dx_list_sorted
    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_dx, df


def read_pregnant_procedure(input_file, id_demo):
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
    print('In read_pregnant_procedure, input_file:', input_file)
    df = pd.read_csv(input_file, dtype=str, parse_dates=['ADMIT_DATE', "PX_DATE"])
    df.rename(columns=lambda x: x.upper(), inplace=True)
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    print('Unique patid:', len(df['PATID'].unique()))
    print('Time range of All Covid dx:', df["ADMIT_DATE"].describe(datetime_is_numeric=True))

    id_dx = defaultdict(list)
    i = 0
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0
    n_no_dob_row = 0

    for index, row in tqdm(df.iterrows(), total=len(df), mininterval=5):
        i += 1
        patid = row['PATID']
        enc_id = row['ENCOUNTERID']
        enc_type = row['ENC_TYPE']
        dx = row['PX']
        dx_type = row["PX_TYPE"]
        dx_date = row["PX_DATE"]  #
        dx_date2 = row["ADMIT_DATE"]  #

        if pd.isna(dx) or (dx == ''):
            n_no_dx += 1

        if pd.isna(dx_date):
            if pd.notna(dx_date2):
                dx_date = dx_date2
            else:
                n_no_date += 1

        if pd.isna(dx) or pd.isna(dx_date) or (dx == ''):
            n_discard_row += 1
        else:
            if patid in id_demo:
                # updated 2022-10-26.
                if pd.notna(id_demo[patid][0]):
                    age = (dx_date - pd.to_datetime(
                        id_demo[patid][0])).days / 365  # (lab_date - id_demo[patid][0]).days // 365
                else:
                    age = np.nan
            else:
                age = np.nan
                n_no_dob_row += 1
                print(n_no_dob_row, 'No age information for:', index, i, patid, dx_date, )

            # (date, code, result_label,  age,  enc_id, encounter_type, CP-method )
            id_dx[patid].append((dx_date, dx, 'Pregnant', age, enc_id, enc_type, 'preg-px'))
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_no_dob_row:', n_no_dob_row, 'len(id_dx):', len(id_dx))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # sort and de-duplicates
    print('sort px list in id_px by time')
    for patid, dx_list in id_dx.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        dx_list_sorted = sorted(set(dx_list), key=lambda x: x[0])
        id_dx[patid] = dx_list_sorted
    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_dx, df


def read_pregnant_encounter(input_file, id_demo):
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
    print('In read_pregnant_encounter, input_file:', input_file)
    df = pd.read_csv(input_file, dtype=str, parse_dates=['ADMIT_DATE', "DISCHARGE_DATE"])
    df.rename(columns=lambda x: x.upper(), inplace=True)
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    print('Unique patid:', len(df['PATID'].unique()))
    print('Time range of All Covid dx:', df["ADMIT_DATE"].describe(datetime_is_numeric=True))

    id_dx = defaultdict(list)
    i = 0
    n_no_dx = 0
    n_no_date = 0
    n_discard_row = 0
    n_recorded_row = 0
    n_no_dob_row = 0

    for index, row in tqdm(df.iterrows(), total=len(df), mininterval=5):
        i += 1
        patid = row['PATID']
        enc_id = row['ENCOUNTERID']
        enc_type = row['ENC_TYPE']
        dx = row['DRG'] # drg code from encounter table for pregnant cp
        dx_type = row["DRG_TYPE"]
        dx_date = row["ADMIT_DATE"]  #
        dx_date2 = row["DISCHARGE_DATE"]  #

        if pd.isna(dx):
            n_no_dx += 1
        if pd.isna(dx_date):
            n_no_date += 1

        if pd.isna(dx) or pd.isna(dx_date):
            n_discard_row += 1
        else:
            if patid in id_demo:
                # updated 2022-10-26.
                if pd.notna(id_demo[patid][0]):
                    age = (dx_date - pd.to_datetime(
                        id_demo[patid][0])).days / 365  # (lab_date - id_demo[patid][0]).days // 365
                else:
                    age = np.nan
            else:
                age = np.nan
                n_no_dob_row += 1
                print(n_no_dob_row, 'No age information for:', index, i, patid, dx_date, )

            # (date, code, result_label,  age,  enc_id, encounter_type, CP-method )
            id_dx[patid].append((dx_date, dx, 'Pregnant', age, enc_id, enc_type, 'preg-encdrg'))
            n_recorded_row += 1

    print('Readlines:', i, 'n_no_dx:', n_no_dx, 'n_no_date:', n_no_date, 'n_discard_row:', n_discard_row,
          'n_recorded_row:', n_recorded_row, 'n_no_dob_row:', n_no_dob_row, 'len(id_dx):', len(id_dx))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # sort and de-duplicates
    print('sort encounter list in id_dx by time')
    for patid, dx_list in id_dx.items():
        # add a set operation to reduce duplicates
        # sorted returns a sorted list
        dx_list_sorted = sorted(set(dx_list), key=lambda x: x[0])
        id_dx[patid] = dx_list_sorted
    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_dx, df


def data_analysis(id_lab):
    cnt = []
    for k, v in id_lab.items():
        cnt.append(len(v))
    pd.DataFrame(cnt).describe()
    c = Counter(cnt)
    df = pd.DataFrame(c.most_common(), columns=['covid_test_count', 'num_of_people'])
    df.to_excel(r'../data/V15_COVID19/output/test_frequency_COL.xlsx')


def combine_2_id_records(id_rec1, id_rec2, output_file=''):
    start_time = time.time()
    print('In combine_2_id_records:',
          'len(id_rec1)', len(id_rec1), 'len(id_rec2)', len(id_rec2))

    id_data = defaultdict(list)
    for patid, rec_list in id_rec1.items():
        id_data[patid] = list(rec_list)

    n_new_add = 0
    n_exist_update = 0
    for patid, records in id_rec2.items():
        if patid not in id_data:
            id_data[patid] = list(records)
            n_new_add += 1
        else:
            id_data[patid].extend(records)
            n_exist_update += 1
    print('Combined len(id_data):', len(id_data), 'n_new_add:', n_new_add, 'n_exist_update:', n_exist_update)

    # sort
    print('sort combined id_data by time')
    for patid, rec_list in id_data.items():
        rec_list_sorted = sorted(set(rec_list), key=lambda x: x[0])
        id_data[patid] = rec_list_sorted

    if output_file:
        utils.check_and_mkdir(output_file)
        pickle.dump(id_data, open(output_file, 'wb'))
        print('Dump done to {}!'.format(output_file))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return id_data


if __name__ == '__main__':
    # python pre_covid_lab.py --dataset WCM 2>&1 | tee  log/pre_covid_lab_WCM.txt
    start_time = time.time()
    args = parse_args()
    with open(args.demo_file, 'rb') as f:
        # to add age information for each covid tests
        # 'PATID' -->[ 'BIRTH_DATE', 'SEX', 'RACE', 'HISPANIC', zipcode, state, city, adi_nation, adi_state]
        id_demo = pickle.load(f)
        print('load', args.demo_file, 'demo information: len(id_demo):', len(id_demo))

    # covid CP from lab
    id_lab, df_pcr, df = read_covid_lab_and_generate_label(args.input_lab, args.output_file_lab, id_demo)

    # covid CP from dx
    id_dx, df_dx = read_covid_diagnosis(args.input_diagnosis, id_demo)
    print('Combine lab and dx')
    id_labdx = combine_2_id_records(id_lab, id_dx, output_file=args.output_file_labdx)

    # covid CP from med
    id_medpre, df_medpre = read_covid_prescribing(args.input_prescribing, id_demo)
    id_medmedadmin, df_medmedadmin = read_covid_med_admin(args.input_med_admin, id_demo)
    id_meddispensing, df_meddispensing = read_covid_dispensing(args.input_dispensing, id_demo)

    print('Combine prescripting and med_admi')
    id_med1 = combine_2_id_records(id_medpre, id_medmedadmin)
    print('Combine prescripting and med_admi and dispensing')
    id_med = combine_2_id_records(id_med1, id_meddispensing)

    # 2024-7-5 add pregnant portions to include more
    # pregnant related records/labels/potential new patients
    id_pregdx, df_pregdx = read_pregnant_diagnosis(args.preg_diagnosis, id_demo)
    id_pregpx, df_pregpx = read_pregnant_procedure(args.preg_procedure, id_demo)
    id_pregenc, df_pregenc = read_pregnant_encounter(args.preg_encounter, id_demo)

    print('combine preg from dx, px and enc-drg')
    id_preg1 = combine_2_id_records(id_pregdx, id_pregpx)
    id_preg = combine_2_id_records(id_preg1, id_pregenc)

    print('Combine labdx and med, and dump to:', args.output_file_labdxmed)
    id_labdxmed = combine_2_id_records(id_labdx, id_med, output_file=args.output_file_labdxmed)
    print('Combine labdxmed and preg, and dump to:', args.output_file_labdxmedpreg)
    id_labdxmedpreg = combine_2_id_records(id_labdxmed, id_preg, output_file=args.output_file_labdxmedpreg)
    print('len(id_lab)', len(id_lab), 'len(id_labdx)', len(id_labdx), 'len(id_labdxmed)', len(id_labdxmed))
    print('len(id_pregdx)', len(id_pregdx), 'len(id_pregpx)', len(id_pregpx),
          'len(id_pregenc)', len(id_pregenc), 'len(id_preg)', len(id_preg),
          'len(id_labdxmedpreg)', len(id_labdxmedpreg))

    # print('PCR+Antigen+Antibody-test #total:', len(df.loc[:, 'PATID'].unique()))
    # print('PCR/Antigen-test #total:', len(df_pcr.loc[:, 'PATID'].unique()))
    # print('PCR/Antigen-test #positive:', len(df_pcr.loc[df_pcr['covid_positive'], 'PATID'].unique()))
    # print('PCR/Antigen-test #negative:', len(df_pcr.loc[~df_pcr['covid_positive'], 'PATID'].unique()))
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
