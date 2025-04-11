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
from misc.utilsql import *

print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess demographics')
    parser.add_argument('--dataset', default='musc', help='site dataset')
    args = parser.parse_args()

    args.input_file = r'{}.death'.format(args.dataset)
    args.output_file = r'../data/recover/output/{}/death_{}.pkl.gz'.format(args.dataset, args.dataset)

    print('args:', args)
    return args


def read_death(input_file, output_file=''):
    """
    :param data_file: input demographics file with std format, id and zipcode mapping
    :param out_file: output id_death[patid] = ['DEATH_DATE', 'DEATH_DATE_IMPUTE', 'DEATH_SOURCE', 'DEATH_MATCH_CONFIDENCE', 'CDRN_FACILITYID'] pickle
    :return: id_death[patid] = ['DEATH_DATE', 'DEATH_DATE_IMPUTE', 'DEATH_SOURCE', 'DEATH_MATCH_CONFIDENCE', 'CDRN_FACILITYID']
    :Notice:
        1.COL data: e.g:
        df.shape: (3753, 6)
        df['SEX'].value_counts():
    """
    start_time = time.time()
    print('In read_death, input_file:', input_file, 'output_file', output_file)
    # df = utils.read_sas_2_df(input_file)
    table_name = input_file  # '{}.lds_address_history'.format(args.dataset, )
    print('Read sql table:', table_name)
    df = load_whole_table_from_sql(table_name)

    if 'DEATH_DATE_IMPUTE' in df.columns:
        print('DEATH_DATE_IMPUTE:', df['DEATH_DATE_IMPUTE'].value_counts(dropna=False))
    else:
        df['DEATH_DATE_IMPUTE'] = np.nan

    print('DEATH_SOURCE:', df['DEATH_SOURCE'].value_counts(dropna=False))
    print('DEATH_MATCH_CONFIDENCE:', df['DEATH_MATCH_CONFIDENCE'].value_counts(dropna=False))
    if 'CDRN_FACILITYID' in df.columns:
        print('CDRN_FACILITYID:', df['CDRN_FACILITYID'].value_counts(dropna=False))
    else:
        df['CDRN_FACILITYID'] = np.nan

    print('df.shape', df.shape, 'df.columns:', df.columns)
    # musc ocurred error when not using , errors='coerce'
    print('Time range of death table [DEATH_DATE]:',
          pd.to_datetime(df["DEATH_DATE"], errors='coerce').describe(datetime_is_numeric=True))

    # df_sub = df
    df_sub = df[['PATID', 'DEATH_DATE', 'DEATH_DATE_IMPUTE', 'DEATH_SOURCE', 'DEATH_MATCH_CONFIDENCE', 'CDRN_FACILITYID']]

    records_list = df_sub.values.tolist()
    # 'PATID' -->['DEATH_DATE', 'DEATH_DATE_IMPUTE', 'DEATH_SOURCE', 'DEATH_MATCH_CONFIDENCE', 'CDRN_FACILITYID']
    id_death = {x[0]: x[1:] for x in records_list}

    print('len(id_demo) {}'.format(len(id_death)))
    if output_file:
        utils.check_and_mkdir(output_file)
        utils.dump(id_death, output_file)

        print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_death


if __name__ == '__main__':
    # python pre_death.py --dataset COL 2>&1 | tee  log/pre_death_COL.txt
    # python pre_death.py --dataset WCM 2>&1 | tee  log/pre_death_WCM.txt
    # python pre_death.py --dataset NYU 2>&1 | tee  log/pre_death_NYU.txt
    # python pre_death.py --dataset MONTE 2>&1 | tee  log/pre_death_MONTE.txt
    # python pre_death.py --dataset MSHS 2>&1 | tee  log/pre_death_MSHS.txt

    start_time = time.time()
    args = parse_args()
    id_death = read_death(args.input_file, args.output_file)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
