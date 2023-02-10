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

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    start_time = time.time()

    cohort_df = pd.read_csv(
        r'..\data\V15_COVID19\output\character\cp_pregnancy\matrix_cohorts_covid_4manuNegNoCovidV2_bool_ALL-anyPASC_pregnancy.csv',
        dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date', 'flag_delivery_date', 'dob'])
    print('1 cohort_df:', cohort_df.shape)

    df = cohort_df.loc[(cohort_df['index date'] < datetime.datetime(2022, 3, 1, 0, 0)), :].copy()
    print('2 df:', df.shape)

    df = df.loc[df['age'] < 50, :].copy()
    print('3 df:', df.shape)

    df = df.loc[df['Female']==1, :].copy()
    print('4 df:', df.shape)



    print(list(df.columns))
    df.head()
    df_label = df['covid']
    print(df_label.shape, df_label.sum(), )

    dfpos = df.loc[df_label==1, :]
    dfneg = df.loc[df_label == 0, :]

    print((df_label==1).sum(), (df_label==1).mean())
    print((df_label==0).sum(), (df_label==0).mean())

    idposcase = (dfpos['flag_pregnancy']==1)
    idnegcase = (dfneg['flag_pregnancy']==1)

    print(idposcase.sum(), idposcase.mean())
    print(idnegcase.sum(), idnegcase.mean())

    dfpreg = df.loc[df['flag_pregnancy']==1, :]

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

