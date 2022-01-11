import sys
# for linux env.
sys.path.insert(0,'..')
import scipy
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import math
import itertools
import os
import pyreadstat
from sas7bdat import SAS7BDAT
import argparse
import csv
import functools
print = functools.partial(print, flush=True)
import utils
from collections import Counter

if __name__ == '__main__':
    start_time = time.time()
    df_covid = pd.read_csv(r'../data/V15_COVID19/covid_lab_test_names.csv')
    code_set = set(df_covid['concept_code'].to_list())
    sasds = pd.read_sas('../data/V15_COVID19/COL/lab_result_cm.sas7bdat', encoding='windows-1252', chunksize=100000,
                        iterator=True)  # 'iso-8859-1' (LATIN1) and Windows cp1252 (WLATIN1)
    dfs = []  # holds data chunks
    dfs_covid = []
    cnt = Counter([])
    i = 0
    for chunk in sasds:
        dfs.append(chunk)
        chunk_covid_records = chunk.loc[chunk['LAB_LOINC'].isin(code_set), :]

        dfs_covid.append(chunk_covid_records)
        cnt.update(chunk_covid_records['RESULT_QUAL'])
        i += 1
        if i % 10 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        if i == 10:
            break

    dfs_covid_all = pd.concat(dfs_covid)
    print('#chunk: ', i)
    print('Counter:', cnt)
    dfs_all = pd.concat(dfs)
    dfs_covid_all.to_excel("col_covid_lab_sample.xlsx")
    dfs_all.to_excel("col_lab_sample.xlsx")
    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))