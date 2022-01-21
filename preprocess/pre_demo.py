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

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='WCM', help='input dataset')
    args = parser.parse_args()
    if args.dataset == 'COL':
        args.input_file = r'../data/V15_COVID19/COL/demographic.sas7bdat'
        args.address_file = r'../data/V15_COVID19/COL/lds_address_history.sas7bdat'

        args.output_file = r'../data/V15_COVID19/output/patient_demo_COL.pkl'
    elif args.dataset == 'WCM':
        args.input_file = r'../data/V15_COVID19/WCM/demographic.sas7bdat'
        args.address_file = r'../data/V15_COVID19/WCM/lds_address_history.sas7bdat'

        args.output_file = r'../data/V15_COVID19/output/patient_demo_WCM.pkl'

    print('args:', args)
    return args


def read_address(input_file):
    """
        :param data_file: input demographics file with std format
        :return: id_demo[patid] = zip5/9
        :Notice:
            1.COL data: e.g:
            df.shape: (549115, 12)
            n_no_zip: 0 n_has_zip9: 300686 n_has_zip5: 248429

            2. WCM data: e.g.
            df.shape: (685598, 12)
            n_no_zip: 111323 n_has_zip9: 1967 n_has_zip5: 572308

        """
    start_time = time.time()
    # 1. load address zip file
    print('In read_address, input_file:', input_file)
    df = utils.read_sas_2_df(input_file)
    print('df.shape', df.shape, 'df.columns:', df.columns)

    # 2. load zip adi dictionary
    with open(r'../data/mapping/zip9or5_adi_mapping.pkl', 'rb') as f:
        zip_adi = pickle.load(f)
        print('load zip9or5_adi_mapping.pkl file done! len(zip_adi):', len(zip_adi))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # 3. build patid --> address dictionary
    id_zip = {}
    n_no_zip = 0
    n_has_zip5 = n_has_zip9 = 0
    n_has_adi = 0
    for index, row in df.iterrows():
        patid = row['PATID']
        city = row['ADDRESS_CITY']
        state = row['ADDRESS_STATE']
        zip5 = row['ADDRESS_ZIP5']
        zip9 = row['ADDRESS_ZIP9']
        if pd.notna(zip9):
            zipcode = zip9
            n_has_zip9 += 1
        elif pd.notna(zip5):
            zipcode = zip5
            n_has_zip5 += 1
        else:
            zipcode = np.nan
            n_no_zip += 1

        if pd.notna(zipcode) and (zipcode in zip_adi):
            adi = zip_adi[zipcode]
            n_has_adi += 1
        else:
            adi = [np.nan, np.nan]

        id_zip[patid] = [zipcode, state, city] + adi

    print('len(id_zip):', len(id_zip), 'n_no_zip:', n_no_zip,
          'n_has_zip9:', n_has_zip9, 'n_has_zip5:', n_has_zip5, 'n_has_adi:', n_has_adi)

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_zip, df


def read_demo(input_file, id_zip, output_file=''):
    """
    :param data_file: input demographics file with std format, id and zipcode mapping
    :param out_file: output id_demo[patid] = (sex, bdate, race, zipcode) pickle
    :return: id_demo[patid] = (sex, bdate, race, zipcode)
    :Notice:
        1.COL data: e.g:
        df.shape: (665952, 17)
        df['SEX'].value_counts():
            F     383186
            M     282449
            UN       222
            OT        95
        df_demo['RACE'].value_counts():
            05    252437 05=White
            07    181736 07=Refuse to answer
            06    130641 06=Multiple race
            03     77446 03=Black or African American
            02     17360 02=Asian
            OT      6332 OT=Other
        HISPANIC
            N           284645
            UN          199929
            Y           173171
            NI            8207
        2. WCM data: e.g.
        df.shape: (1534329, 17)
        df['SEX'].value_counts():
            F     889980
            M     642363
            UN      1986
        df_demo['RACE'].value_counts():
            NI    712225 NI=No information
            05    535746 05=White
            03    177187 03=Black or African American
            02    108054 02=Asian
            UN      1101 UN=Unknown
            04         8 04=Native Hawaiian or Other Pacific Islander
            01         8 01=American Indian or Alaska Native
        HISPANIC
        N     1267042
        Y      265346
        NI       1941
    """
    start_time = time.time()
    print('In read_demo, input_file:', input_file, 'len(id_zip):', len(id_zip), 'output_file', output_file)
    df = utils.read_sas_2_df(input_file)
    print('df.shape', df.shape, 'df.columns:', df.columns)
    df_sub = df[['PATID', 'BIRTH_DATE', 'SEX', 'RACE', 'HISPANIC']]
    records_list = df_sub.values.tolist()
    id_demo = {x[0]: x[1:] + list(id_zip.get(x[0], [np.nan, np.nan, np.nan, np.nan, np.nan])) for x in records_list}

    print('df.shape {}, len(id_demo) {}'.format(df.shape, len(id_demo)))
    if output_file:
        utils.check_and_mkdir(output_file)
        pickle.dump(id_demo, open(output_file, 'wb'))
        # df_sub.to_csv(output_file.replace('.pkl', '') + '.csv')

        print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_demo, df_sub


if __name__ == '__main__':
    # python pre_demo.py --dataset COL 2>&1 | tee  log/pre_demo_COL.txt
    # python pre_demo.py --dataset WCM 2>&1 | tee  log/pre_demo_WCM.txt
    start_time = time.time()
    args = parse_args()
    id_zip, df_addr = read_address(args.address_file)
    id_demo, df_sub = read_demo(args.input_file, id_zip, args.output_file)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
