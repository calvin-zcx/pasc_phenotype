import sys

# for linux env.
sys.path.insert(0, '..')
import os
import pickle
import numpy as np
from collections import defaultdict, OrderedDict
import pandas as pd
import requests
import functools
from misc import utils
import re
from tqdm import tqdm

print = functools.partial(print, flush=True)
import time
import warnings
import datetime
import zipfile

if __name__ == '__main__':
    start_time = time.time()
    cohort_df = pd.read_excel(
        r'../data/V15_COVID19/output/character/cp_dm/diabetes_incidence-Sep2.xlsx',
        dtype={'patid': str, 'site': str, 'zip': str},
        parse_dates=['index date'])

    sites = ['wcm', 'mshs', 'columbia', 'montefiore', 'nyu', ]
    df_list = []
    site_map = {'NYU': 'nyu',
                'COL': 'columbia',
                'WCM': 'wcm',
                'MONTE': 'montefiore',
                'MSHS': 'mshs',
                }
    for ith, site in tqdm(enumerate(sites)):
        print('Loading: ', ith, site)
        data_file = r'../data/recover/output/{}/matrix_cohorts_covid_posneg18base-nbaseout-alldays-preg_{}.csv'.format(
            site, site)
        print('read file from:', data_file)
        # Load Covariates Data
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
                         parse_dates=['index date'])
        print('df.shape:', df.shape)

        df_list.append(df)
        print('Done', ith, site)

    # combine all sites and select subcohorts
    print('To concat len(df_list)', len(df_list))
    df = pd.concat(df_list, ignore_index=True)

    n_notfound = 0
    n_consist = 0
    n_notconsist = 0
    n_notconsist0to1 = 0
    n_notconsist1to0 = 0
    print('cohort_df.shape', cohort_df.shape, 'df.shape', df.shape, )
    pid_notconsist_list = []

    ith = 0
    for index, rows in tqdm(cohort_df.iterrows(), total=cohort_df.shape[0]):
        ith += 1
        patid = rows['patid']
        site = rows['site']
        covid = rows['covid']

        # if patid in df_sub['patid']:
        covid2_ser = df.loc[(df['site'] == site_map[site]) & (df['patid'] == patid), 'covid']
        if len(covid2_ser) == 0:
            n_notfound += 1
        else:
            covid2 = covid2_ser.values[0]
            if covid == covid2:
                n_consist += 1
            else:
                n_notconsist += 1
                if (covid == 1) and (covid2 == 0):
                    n_notconsist1to0 += 1
                elif (covid == 0) and (covid2 == 1):
                    n_notconsist0to1 += 1
                pid_notconsist_list.append(rows)

        if ith % 500 == 0:
            print(ith, 'n_notfound', n_notfound, 'n_consist', n_consist, 'n_notconsist', n_notconsist,
                  'n_notconsist0to1', n_notconsist0to1, 'n_notconsist1to0', n_notconsist1to0)

    df_pid_notconsist = pd.concat([x.to_frame().T for x in pid_notconsist_list])
    # df_pid_notconsist = pd.concat(pid_notconsist_list)
    print('n_notfound', n_notfound, 'n_consist', n_consist, 'n_notconsist', n_notconsist)
    print(ith, 'n_notfound', n_notfound, 'n_consist', n_consist, 'n_notconsist', n_notconsist,
          'n_notconsist0to1', n_notconsist0to1, 'n_notconsist1to0', n_notconsist1to0)
    df_pid_notconsist.to_csv('df_pid_notconsist.csv')
