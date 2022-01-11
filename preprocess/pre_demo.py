import sys
# for linux env.
sys.path.insert(0,'..')
from datetime import datetime
import os
import pandas as pd
from tqdm import tqdm
import time
import pickle
import argparse
import csv
import utils
import numpy as np
import functools
from collections import Counter
print = functools.partial(print, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='process parameters')
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='WCM', help='input dataset')
    args = parser.parse_args()
    if args.dataset == 'COL':
        args.input_file = r'../data/V15_COVID19/COL/demographic.sas7bdat'
        args.output_file = r'../data/V15_COVID19/output/patient_demo_COL.pkl'
    elif args.dataset == 'WCM':
        args.input_file = r'../data/V15_COVID19/WCM/demographic.sas7bdat'
        args.output_file = r'../data/V15_COVID19/output/patient_demo_WCM.pkl'

    print('args:', args)
    return args


def read_demo(input_file, output_file=''):
    """
    :param data_file: input demographics file with std format
    :param out_file: output id_demo[patid] = (sex, bdate, race) pickle
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
    """
    start_time = time.time()
    df = utils.read_sas_2_df(input_file)
    df_sub = df[['PATID', 'BIRTH_DATE', 'SEX', 'RACE', ]]
    records_list = df_sub.values.tolist()
    id_demo = {x[0]: x[1:] for x in records_list}

    print('df.shape {}, len(id_demo) {}'.format(df.shape, len(id_demo)))
    if output_file:
        utils.check_and_mkdir(output_file)
        pickle.dump(id_demo, open(output_file, 'wb'))
        # df_sub.to_csv(output_file.replace('.pkl', '') + '.csv')

        print('dump done to {}'.format(output_file))
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return id_demo, df_sub


if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    id_demo, df_sub = read_demo(args.input_file, args.output_file)
    # patient_dates = build_patient_dates(args.demo_file, args.dx_file, r'output/patient_dates.pkl')
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
