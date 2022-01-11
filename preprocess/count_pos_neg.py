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


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess-count pos and negative from lab file')
    parser.add_argument('--dataset', choices=['COL', 'WCM'], default='COL', help='input dataset directory')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    return args


def read_lab_and_count_covid(dataset='COL', chunksize=100000, debug=False):
    start_time = time.time()
    print('Choose dataset:', dataset, 'chunksize:', chunksize, 'debug:', debug)
    # step 1: load covid lab test codes, may be updated by:
    # https://github.com/National-COVID-Cohort-Collaborative/Phenotype_Data_Acquisition/wiki/Latest-Phenotype
    df_covid = pd.read_csv(r'../data/V15_COVID19/covid_lab_test_names.csv')
    code_set = set(df_covid['concept_code'].to_list())

    # step 2: read lab results by chunk, due to large file size
    sasds = pd.read_sas('../data/V15_COVID19/{}/lab_result_cm.sas7bdat'.format(dataset),
                        encoding='windows-1252',
                        chunksize=chunksize,
                        iterator=True)  # 'iso-8859-1' (LATIN1) and Windows cp1252 (WLATIN1)
    dfs = []  # holds data chunks
    dfs_covid = []
    cnt = Counter([])
    i = 0
    n_rows = 0
    n_covid_rows = 0
    for chunk in sasds:
        n_rows += len(chunk)
        dfs.append(chunk)
        if dataset == 'COL':
            chunk_covid_records = chunk.loc[chunk['LAB_LOINC'].isin(code_set), :]
        elif dataset == 'WCM':
            chunk_covid_records = chunk.loc[chunk['lab_loinc'].isin(code_set), :]

        n_covid_rows += len(chunk_covid_records)
        if dataset == 'COL':
            cnt.update(chunk_covid_records['RESULT_QUAL'])
        elif dataset == 'WCM':
            cnt.update(chunk_covid_records['result_qual'])

        dfs_covid.append(chunk_covid_records)

        i += 1
        if i == 1:
            print('chunk.shape', chunk.shape)
            print('chunk.columns', chunk.columns)

        if i % 10 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            if debug:
                print('IN DEBUG MODE, BREAK, AND DUMP!')
                break

    dfs_covid_all = pd.concat(dfs_covid)
    print('n_rows:', n_rows, 'n_covid_rows:', n_covid_rows)
    print('#chunk: ', i, 'chunk size:', chunksize)
    print('Counter:', cnt)
    dfs_all = pd.concat(dfs)
    dfs_covid_all.to_excel("{}_covid_lab_sample.xlsx".format(dataset))
    dfs_all.to_excel("{}_lab_sample.xlsx".format(dataset))
    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


if __name__ == '__main__':
    args = parse_args()
    print(args)
    read_lab_and_count_covid(dataset=args.dataset, debug=args.debug)
