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
from misc.utilsql import *

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--dataset', default='intermountain', help='site dataset')
    args = parser.parse_args()

    args.input_file = r'{}.demographic'.format(args.dataset)
    args.address_file = r'{}.lds_address_history'.format(args.dataset)
    args.geocoded_file = r'{}.geocoded_2020'.format(args.dataset)
    args.output_file = r'../data/recover/output/{}/patient_demo_{}.pkl'.format(args.dataset, args.dataset)

    if args.dataset == 'utsw_pcornet_all':
        args.geocoded_file = r'{}.geocoded_all'.format(args.dataset)
        print(args.dataset, args.geocoded_file)

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

            3.NYU:
            df.shape: (653395, 12) len(id_zip): 653395
            n_no_zip: 0 n_has_zip9: 0 n_has_zip5: 653395 n_has_adi: 653340

            4.MONTE:
            df.shape: (272874, 12) len(id_zip):
            272874 n_no_zip: 313 n_has_zip9: 52894 n_has_zip5: 219667 n_has_adi: 270779

            5. MSHS
            df.shape: (1723311, 12) len(id_zip):
            1723311 n_no_zip: 96687 n_has_zip9: 0 n_has_zip5: 1626624 n_has_adi: 1626414
        """
    start_time = time.time()

    # 1. load address zip file
    table_name = input_file  # '{}.lds_address_history'.format(args.dataset, )
    print('Read sql table:', table_name)
    df = load_whole_table_from_sql(table_name)
    df = df.sort_values(by=['PATID', 'ADDRESS_PERIOD_START'])  # tring to use the latest address
    print('df.shape', df.shape, 'df.columns:', df.columns)

    # 2. load zip adi dictionary, using updated aid from 2020 version
    # still use 2020 version rather than 2021 due to large missingness in 2021 version
    with open(r'../data/mapping/zip9or5_adi_mapping_2020.pkl', 'rb') as f:
        # with open(r'../data/mapping/zip9or5_adi_mapping_2021.pkl', 'rb') as f:
        # with open(r'../data/mapping/zip9or5_adi_mapping.pkl', 'rb') as f:
        zip_adi = pickle.load(f)
        print('load zip9or5_adi_mapping.pkl file done! len(zip_adi):', len(zip_adi))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # 3. build patid --> address dictionary
    id_zip = {}
    n_no_zip = 0
    n_has_zip5 = n_has_zip9 = 0
    n_has_adi = 0
    for index, row in tqdm(df.iterrows(), total=len(df)):
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

        if isinstance(zipcode, str) and len(zipcode) >= 5:
            zipcode5 = zipcode[:5]
        else:
            zipcode5 = np.nan

        # if zip9 no adi, try to use zip5 to impute
        # if zip5, and no adi, just rewrite nan to nan
        if pd.notna(zipcode) and (zipcode in zip_adi):
            adi = zip_adi[zipcode]
            n_has_adi += 1
            if pd.isna(adi[0]):
                if pd.notna(zipcode5) and (zipcode5 in zip_adi):
                    adi[0] = zip_adi[zipcode5][0]

            if pd.isna(adi[1]):
                if pd.notna(zipcode5) and (zipcode5 in zip_adi):
                    adi[1] = zip_adi[zipcode5][1]
        else:
            adi = [np.nan, np.nan]

        # 2022-10-26 only update notna zipcode
        if pd.notna(zipcode):
            id_zip[patid] = [zipcode, state, city] + adi

    print('df.shape:', df.shape, 'len(id_zip):', len(id_zip), 'n_no_zip:', n_no_zip,
          'n_has_zip9:', n_has_zip9, 'n_has_zip5:', n_has_zip5, 'n_has_adi:', n_has_adi)

    # 2022-10-26 enrich id zip by demographic table in some sites
    table_name = r'{}.demographic'.format(args.dataset)
    print('Read sql table:', table_name)
    df_dem = load_whole_table_from_sql(table_name)
    print('df_dem.shape', df_dem.shape, 'df_dem.columns:', df_dem.columns)
    if 'ZIPCODE' in df_dem.columns:
        zcol = 'ZIPCODE'
    elif 'ZIP_CODE' in df_dem.columns:
        zcol = 'ZIP_CODE'
    else:
        zcol = ''

    n_add_zip_from_demo = 0
    if zcol:
        print(zcol, ' in df_dem.columns, begin enrich')
        for index, row in tqdm(df_dem.iterrows(), total=len(df_dem)):
            patid = row['PATID']
            zipcode = row[zcol]
            if isinstance(zipcode, str):
                zipcode = zipcode.strip().replace('-', '')
                if isinstance(zipcode, str) and len(zipcode) >= 5:
                    zipcode5 = zipcode[:5]
                else:
                    zipcode5 = np.nan

                if pd.notna(zipcode) and (zipcode in zip_adi):
                    adi = zip_adi[zipcode]

                    if pd.isna(adi[0]):
                        if pd.notna(zipcode5) and (zipcode5 in zip_adi):
                            adi[0] = zip_adi[zipcode5][0]

                    if pd.isna(adi[1]):
                        if pd.notna(zipcode5) and (zipcode5 in zip_adi):
                            adi[1] = zip_adi[zipcode5][1]
                else:
                    adi = [np.nan, np.nan]

                if patid not in id_zip:
                    id_zip[patid] = [zipcode, np.nan, np.nan] + adi
                    n_add_zip_from_demo += 1
                else:
                    rec = id_zip[patid]
                    if pd.isna(rec[0]):
                        id_zip[patid] = [zipcode, rec[1], rec[2]] + adi
                        n_add_zip_from_demo += 1

    print('n_add_zip_from_demo:', n_add_zip_from_demo)
    print('len(id_zip):', len(id_zip))

    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_zip, df


def read_address_and_geocoded(input_file_address, input_file_geo):
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

            3.NYU:
            df.shape: (653395, 12) len(id_zip): 653395
            n_no_zip: 0 n_has_zip9: 0 n_has_zip5: 653395 n_has_adi: 653340

            4.MONTE:
            df.shape: (272874, 12) len(id_zip):
            272874 n_no_zip: 313 n_has_zip9: 52894 n_has_zip5: 219667 n_has_adi: 270779

            5. MSHS
            df.shape: (1723311, 12) len(id_zip):
            1723311 n_no_zip: 96687 n_has_zip9: 0 n_has_zip5: 1626624 n_has_adi: 1626414
        """
    start_time = time.time()

    # 1. load address zip file
    table_name = input_file_address  # '{}.lds_address_history'.format(args.dataset, )
    print('Read sql table:', table_name)
    df = load_whole_table_from_sql(table_name)
    df = df.sort_values(by=['PATID', 'ADDRESS_PERIOD_START'])  # tring to use the latest address
    print('df.shape', df.shape, 'df.columns:', df.columns)

    table_name2 = input_file_geo  # '{}.lds_address_history'.format(args.dataset, )
    print('Read sql table:', table_name2)
    df_geo = load_whole_table_from_sql(table_name2)
    print('df_geo.shape', df_geo.shape, 'df_geo.columns:', df_geo.columns)

    # 2. load zip adi dictionary, using updated aid from 2020 version
    # still use 2020 version rather than 2021 due to large missingness in 2021 version
    with open(r'../data/mapping/zip9or5_adi_mapping_2020.pkl', 'rb') as f:
        # with open(r'../data/mapping/zip9or5_adi_mapping_2021.pkl', 'rb') as f:
        # with open(r'../data/mapping/zip9or5_adi_mapping.pkl', 'rb') as f:
        zip_adi = pickle.load(f)
        print('load zip9or5_adi_mapping.pkl file done! len(zip_adi):', len(zip_adi))

    with open(r'../data/mapping/fips_adi_mapping_2020.pkl', 'rb') as f:
        fips_adi = pickle.load(f)
        print('load fips_adi_mapping_2020.pkl file done! len(fips_adi):', len(fips_adi))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    # 3. build patid --> address dictionary
    print('step 1, build adi from address table')
    id_zip = {}
    n_no_zip = 0
    n_has_zip5 = n_has_zip9 = 0
    n_has_adi = 0
    for index, row in tqdm(df.iterrows(), total=len(df), mininterval=10):
        patid = row['PATID']
        city = row['ADDRESS_CITY']
        state = row['ADDRESS_STATE']
        zip5 = row['ADDRESS_ZIP5']
        zip9 = row['ADDRESS_ZIP9']
        if pd.notna(zip9) and zip9:  # zip9 not '' or None
            zipcode = zip9
            n_has_zip9 += 1
        elif pd.notna(zip5) and zip5:
            zipcode = zip5
            n_has_zip5 += 1
        else:
            zipcode = np.nan
            n_no_zip += 1

        if isinstance(zipcode, str) and len(zipcode) >= 5:
            zipcode5 = zipcode[:5]
        else:
            zipcode5 = np.nan

        # if zip9 no adi, try to use zip5 to impute
        # if zip5, and no adi, just rewrite nan to nan
        if pd.notna(zipcode) and (zipcode in zip_adi):
            adi = zip_adi[zipcode]
            n_has_adi += 1
            if pd.isna(adi[0]):
                if pd.notna(zipcode5) and (zipcode5 in zip_adi):
                    adi[0] = zip_adi[zipcode5][0]

            if pd.isna(adi[1]):
                if pd.notna(zipcode5) and (zipcode5 in zip_adi):
                    adi[1] = zip_adi[zipcode5][1]
        else:
            adi = [np.nan, np.nan]

        # 2022-10-26 only update notna zipcode
        if pd.notna(zipcode):
            id_zip[patid] = [zipcode, state, city] + adi

    print('df.shape:', df.shape, 'len(id_zip):', len(id_zip), 'n_no_zip:', n_no_zip,
          'n_has_zip9:', n_has_zip9, 'n_has_zip5:', n_has_zip5, 'n_has_adi:', n_has_adi)

    # 2022-10-26 enrich id zip by demographic table in some sites
    print('step 2, enrich adi from demographic table')
    table_name = r'{}.demographic'.format(args.dataset)
    print('Read sql table:', table_name)
    df_dem = load_whole_table_from_sql(table_name)
    print('df_dem.shape', df_dem.shape, 'df_dem.columns:', df_dem.columns)
    if 'ZIPCODE' in df_dem.columns:
        zcol = 'ZIPCODE'
    elif 'ZIP_CODE' in df_dem.columns:
        zcol = 'ZIP_CODE'
    else:
        zcol = ''

    n_add_zip_from_demo = 0
    if zcol:
        print(zcol, ' in df_dem.columns, begin enrich')
        for index, row in tqdm(df_dem.iterrows(), total=len(df_dem)):
            patid = row['PATID']
            zipcode = row[zcol]
            if isinstance(zipcode, str):
                zipcode = zipcode.strip().replace('-', '')
                if isinstance(zipcode, str) and len(zipcode) >= 5:
                    zipcode5 = zipcode[:5]
                else:
                    zipcode5 = np.nan

                if pd.notna(zipcode) and (zipcode in zip_adi):
                    adi = zip_adi[zipcode]

                    if pd.isna(adi[0]):
                        if pd.notna(zipcode5) and (zipcode5 in zip_adi):
                            adi[0] = zip_adi[zipcode5][0]

                    if pd.isna(adi[1]):
                        if pd.notna(zipcode5) and (zipcode5 in zip_adi):
                            adi[1] = zip_adi[zipcode5][1]
                else:
                    adi = [np.nan, np.nan]

                if patid not in id_zip:
                    id_zip[patid] = [zipcode, np.nan, np.nan] + adi
                    n_add_zip_from_demo += 1
                else:
                    rec = id_zip[patid]
                    if pd.isna(rec[0]):
                        id_zip[patid] = [zipcode, rec[1], rec[2]] + adi
                        n_add_zip_from_demo += 1

    print('n_add_zip_from_demo:', n_add_zip_from_demo)
    print('len(id_zip):', len(id_zip))

    # step 3
    print('step 3, update adi from geocoded table')
    print('df_geo.shape', df_geo.shape, 'df_geo.columns:', df_geo.columns)

    if 'GEOCODING_BLOCKGROUP_ID2020' in df_geo.columns:
        zcol = 'GEOCODING_BLOCKGROUP_ID2020'
    elif 'fips_block_group_id_2020'.upper() in df_geo.columns:
        zcol = 'fips_block_group_id_2020'.upper()
    elif 'census_block_group_id_2020'.upper() in df_geo.columns:
        zcol = 'census_block_group_id_2020'.upper()
    elif 'census_block_group_2020'.upper() in df_geo.columns:
        zcol = 'census_block_group_2020'.upper()
    elif 'census_block_id'.upper() in df_geo.columns:
        zcol = 'census_block_id'.upper()
    else:
        zcol = ''

    n_add_zip_from_geocoded = 0
    if zcol:
        print(zcol, ' in df_geo.columns, begin enrich')
        for index, row in tqdm(df_geo.iterrows(), total=len(df_geo), mininterval=10):
            patid = row['PATID']
            fipscode = row[zcol]
            if pd.notna(fipscode) and (fipscode in fips_adi):
                adi = fips_adi[fipscode]
            else:
                adi = [np.nan, np.nan]

            if patid not in id_zip:
                id_zip[patid] = [fipscode, np.nan, np.nan] + adi
                n_add_zip_from_geocoded += 1
            else:
                rec = id_zip[patid]
                if pd.isna(rec[0]):
                    id_zip[patid] = [fipscode, rec[1], rec[2]] + adi
                    n_add_zip_from_geocoded += 1

    print('n_add_zip_from_geocoded:', n_add_zip_from_geocoded)
    print('len(id_zip):', len(id_zip))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_zip, df, df_dem, df_geo


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
    # df = utils.read_sas_2_df(input_file)
    table_name = input_file  # '{}.demographic'.format(args.dataset, )
    print('Read sql table:', table_name)
    df = load_whole_table_from_sql(table_name)
    print('df.shape', df.shape, 'df.columns:', df.columns)

    print('SEX:', df['SEX'].value_counts(dropna=False))
    print('RACE:', df['RACE'].value_counts(dropna=False))
    print('HISPANIC:', df['HISPANIC'].value_counts(dropna=False))

    print('df.shape', df.shape, 'df.columns:', df.columns)
    df_sub = df[['PATID', 'BIRTH_DATE', 'SEX', 'RACE', 'HISPANIC']]
    # or add imputation logics, say raw_gender, raw_sex etc.

    records_list = df_sub.values.tolist()
    # 'PATID' -->[ 'BIRTH_DATE', 'SEX', 'RACE', 'HISPANIC', zipcode, state, city, adi_nation, adi_state]
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
    # python pre_demo.py --dataset NYU 2>&1 | tee  log/pre_demo_NYU.txt
    # python pre_demo.py --dataset MONTE 2>&1 | tee  log/pre_demo_MONTE.txt
    # python pre_demo.py --dataset MSHS 2>&1 | tee  log/pre_demo_MSHS.txt
    start_time = time.time()
    args = parse_args()
    # id_zip, df_addr = read_address(args.address_file)
    id_zip, df_addr, df_dem, df_geo = read_address_and_geocoded(args.address_file, args.geocoded_file)
    id_demo, df_sub = read_demo(args.input_file, id_zip, args.output_file)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
