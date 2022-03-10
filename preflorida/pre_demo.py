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
    parser.add_argument('--dataset', choices=['all'], default='all', help='combined dataset')
    args = parser.parse_args()

    args.demo_file1 = r'../data/oneflorida/covid_database/DEMOGRAPHIC.csv'
    args.demo_file2 = r'../data/oneflorida/main_database/DEMOGRAPHIC.csv'

    # args.address_file1 = r'../data/oneflorida/covid_database/LDS_ADDRESS_HISTORY.csv'
    # args.address_file2 = r'../data/oneflorida/main_database/LDS_ADDRESS_HISTORY.csv'

    args.output_file = r'../data/oneflorida/output/{}/patient_demo_{}.pkl'.format(args.dataset, args.dataset)

    print('args:', args)
    return args


def read_demo(args, output_file=''):
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
    print('In read_demo, output_file', output_file)

    df1 = pd.read_csv(args.demo_file1, dtype=str, parse_dates=['BIRTH_DATE'])
    df1.rename(columns=lambda x: x.upper(), inplace=True)
    print('df1.shape', df1.shape)
    print('df1.columns', df1.columns)
    print('Unique patid:', len(df1['PATID'].unique()))
    print('nan zip:', df1['ZIP_CODE'].isna().sum())

    df2 = pd.read_csv(args.demo_file2, dtype=str, parse_dates=['BIRTH_DATE'])
    df2.rename(columns=lambda x: x.upper(), inplace=True)
    print('df2.shape', df2.shape)
    print('df2.columns', df2.columns)
    print('Unique patid:', len(df2['PATID'].unique()))
    print('nan zip:', df2['ZIP_CODE'].isna().sum())

    df = pd.concat([df1, df2])
    print('df.shape', df.shape)
    print('df.columns', df.columns)
    print('Unique patid:', len(df['PATID'].unique()))
    print('nan zip:', df['ZIP_CODE'].isna().sum())

    df = df.drop_duplicates(subset=['PATID'])
    print('drop_duplicates df.shape', df.shape)
    print('drop_duplicates df.columns', df.columns)
    print('drop_duplicates Unique patid:', len(df['PATID'].unique()))
    df.loc[:, 'ZIP_CODE'] = df['ZIP_CODE'].apply(lambda x: x.replace('-', '') if isinstance(x, str) else x)
    print('nan zip:', df['ZIP_CODE'].isna().sum())

    print('SEX:', df['SEX'].value_counts(dropna=False))
    print('RACE:', df['RACE'].value_counts(dropna=False))
    print('HISPANIC:', df['HISPANIC'].value_counts(dropna=False))

    print('df.shape', df.shape, 'df.columns:', df.columns)
    df_sub = df[['PATID', 'BIRTH_DATE', 'SEX', 'RACE', 'HISPANIC', 'ZIP_CODE']]

    with open(r'../data/mapping/zip9or5_adi_mapping.pkl', 'rb') as f:
        zip_adi = pickle.load(f)
        print('load zip9or5_adi_mapping.pkl file done! len(zip_adi):', len(zip_adi))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    records_list = df_sub.values.tolist()
    # 'PATID' -->[ 'BIRTH_DATE', 'SEX', 'RACE', 'HISPANIC',  zipcode, state, city, adi_nation, adi_state]
    # id_demo = {x[0]: x[1:] + list(id_zip.get(x[0], [np.nan, np.nan, np.nan, np.nan, np.nan])) for x in records_list}
    id_demo = {}
    n_has_adi = 0
    for x in records_list:
        pid = x[0]
        if '--' in pid:
            print(pid, x)
            continue

        zipcode = x[-1]
        if pd.notna(zipcode) and (zipcode in zip_adi):
            adi = zip_adi[zipcode]
            n_has_adi += 1
        else:
            adi = [np.nan, np.nan]
        # not using address file, just use zipcode from demographics
        id_demo[pid] = x[1:] + [np.nan, np.nan] + adi

    print('df.shape {}, len(id_demo) {}, n_has_adi {}'.format(df.shape, len(id_demo), n_has_adi))

    if output_file:
        utils.check_and_mkdir(output_file)
        pickle.dump(id_demo, open(output_file, 'wb'))
        # df_sub.to_csv(output_file.replace('.pkl', '') + '.csv')

        print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_demo, df_sub


if __name__ == '__main__':
    # python pre_demo.py --dataset all 2>&1 | tee  log/pre_demo_allcombined.txt

    start_time = time.time()
    args = parse_args()
    id_demo, df_sub = read_demo(args, args.output_file)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
