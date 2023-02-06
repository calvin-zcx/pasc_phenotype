import os
import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import argparse
import time
import random
import pickle
import ast
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import functools
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce

print = functools.partial(print, flush=True)
from misc import utils

if __name__ == '__main__':
    start_time = time.time()

    df_site = pd.read_excel('RECOVER Adult Site schemas_edit.xlsx')

    site_list = df_site.loc[df_site['selected'] == 1, 'Schema name']
    vdfec = []
    vdf = []
    for i, site in tqdm(enumerate(site_list)):
        print(i, site)
        try:
            df_ec = pd.read_csv(r'../data/recover/output/{}/cohorts_covid_4manuNegNoCovidV2_{}_info.csv'.format(site, site))
            df_ec['n_site'] = 1
            vdfec.append(df_ec)
            data_file = r'../data/recover/output/{}/matrix_cohorts_covid_4manuNegNoCovidV2_boolbase-nout-withAllDays_{}.csv'.format(
                site,
                site)
            df_site = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
            df_site_sub = df_site.iloc[:, :175]
            dsum = df_site_sub.sum()
            dmean = df_site_sub.mean()
            print('')
            vdf.append(df_site_sub)
        except Exception as e:
            print('[ERROR:]', e, file=sys.stderr)
            continue
    dfec_sum = reduce(lambda x, y: x.add(y, fill_value=0), vdfec)
    print(dfec_sum)
    dfec_sum.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2_all_info.csv')
    df = pd.concat(vdf, axis=0, ignore_index=True)
    df.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2_all_covariates.csv')

    dsum = df.sum()
    dsum.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2_all_covariates_sum.csv')

    dmean = df.mean()
    dmean.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2_all_covariates_mean.csv')

    dsmdf = pd.DataFrame({'sum': dsum[2:], 'mean':dmean})
    dsmdf.to_csv(r'output/cohorts_covid_4manuNegNoCovidV2_all_covariates_summary.csv')

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

