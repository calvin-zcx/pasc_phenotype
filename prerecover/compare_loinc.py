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


if __name__ == '__main__':
    # python pre_codemapping.py 2>&1 | tee  log/pre_codemapping_zip_adi.txt
    start_time = time.time()
    df = pd.read_excel(r'../data/mapping/covid_lab_2023/Covid LOINC Comparison_12.13.22jb.xlsx',
                           sheet_name='LOINC Comparison',
                           dtype=str)  # read all sheets

    df_info = pd.read_excel(r'../data/mapping/covid_lab_2023/Loinc_Sarscov2_Export_20230216020415.xlsx',
                           sheet_name='Loinc_Sarscov2_Export_202302160',
                           dtype=str)  # read all sheets

    dfcom = pd.merge(df, df_info, left_on='LOINC_NUM', right_on='LOINC_NUM', how='left')

    ls = set(df['LOINC_NUM'])
    rs = set(df_info['LOINC_NUM'])

    dfcom.to_csv(r'../data/mapping/covid_lab_2023/Covid LOINC Comparison_12.13.22jb_aux.csv')
    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

