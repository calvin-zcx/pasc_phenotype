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

    args.input_file = r'../data/oneflorida/{}/DEATH.csv'.format(args.dataset)
    args.output_file = r'../data/oneflorida/output/{}/death_{}.pkl'.format(args.dataset, args.dataset)

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

    df = pd.read_csv(input_file, dtype=str, parse_dates=['DEATH_DATE'])

    print('DEATH_DATE_IMPUTE:', df['DEATH_DATE_IMPUTE'].value_counts(dropna=False))
    print('DEATH_SOURCE:', df['DEATH_SOURCE'].value_counts(dropna=False))
    print('DEATH_MATCH_CONFIDENCE:', df['DEATH_MATCH_CONFIDENCE'].value_counts(dropna=False))
    # print('CDRN_FACILITYID:', df['CDRN_FACILITYID'].value_counts(dropna=False))

    print('df.shape', df.shape, 'df.columns:', df.columns)
    print('Time range of death table [DEATH_DATE]:',
          df["DEATH_DATE"].describe(datetime_is_numeric=True))

    df_sub = df
    records_list = df_sub.values.tolist()
    # 'PATID' -->['DEATH_DATE', 'DEATH_DATE_IMPUTE', 'DEATH_SOURCE', 'DEATH_MATCH_CONFIDENCE', 'SOURCE_MASKED']
    id_death = {x[0]: x[1:] for x in records_list}

    print('len(id_demo) {}'.format(len(id_death)))
    if output_file:
        utils.check_and_mkdir(output_file)
        utils.dump(id_death, output_file, chunk=4)

        print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_death


if __name__ == '__main__':
    # python pre_death.py --dataset all 2>&1 | tee  log/pre_death_all.txt

    start_time = time.time()
    args = parse_args()
    id_death = read_death(args.input_file, args.output_file)
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
