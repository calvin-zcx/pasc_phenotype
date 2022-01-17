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


def covid_lab_phenotyping():
    df_wcm = pd.read_csv(r'../data/V15_COVID19/covid_phenotype/covid_lab_test_names.csv', dtype=str)
    print('df_wcm.shape:', df_wcm.shape)
    df_n3c = pd.read_csv(r'../data/V15_COVID19/covid_phenotype/NC3_Latest_Covid_Phenotype_Lab.csv', dtype=str)
    print('df_n3c.shape:', df_n3c.shape)

    union_codes = set(df_wcm['LAB_LOINC'].to_list()) | set(df_n3c['LAB_LOINC'].to_list())
    print('Union: ', len(union_codes))

    df_combined = pd.merge(df_wcm, df_n3c, on='LAB_LOINC', how='outer')
    print('df_combined.shape:', df_combined.shape)

    df_COL_fre = pd.read_excel(r'../data/V15_COVID19/covid_phenotype/lab_loinc_COL_frequency.xlsx',
                               sheet_name='lab_loinc_COL_frequency')
    print('df_COL_fre.shape:', df_COL_fre.shape)
    df_WCM_fre = pd.read_excel(r'../data/V15_COVID19/covid_phenotype/lab_loinc_WCM_frequency.xlsx',
                               sheet_name='lab_loinc_WCM_frequency')
    print('df_WCM_fre.shape:', df_WCM_fre.shape)

    def inner_and_count(code, fre):
        df = pd.merge(code, fre, on='LAB_LOINC', how='inner')
        print('Total occur:', df['Frequency'].sum(), ', found', df.shape[0], 'covid codes in RWD')
        cols = []
        for x in ['LAB_LOINC', 'concept_name', 'type', 'Name', 'Frequency']:
            if x in df.columns:
                cols.append(x)
        print(df[cols])
        return df

    print('WCM codes in COL')
    inner_and_count(df_wcm, df_COL_fre)
    print('WCM codes in WCM')
    inner_and_count(df_wcm, df_WCM_fre)

    print('N3C codes in COL')
    inner_and_count(df_n3c, df_COL_fre)
    print('N3C codes in WCM')
    inner_and_count(df_n3c, df_WCM_fre)

    print('Combined codes in COL')
    fre_col = inner_and_count(df_combined, df_COL_fre)
    print('Combined codes in WCM')
    fre_wcm = inner_and_count(df_combined, df_WCM_fre)

    df_combined1 = pd.merge(df_combined, fre_col[['LAB_LOINC', 'Frequency']], on='LAB_LOINC', how='left')
    df_combined2 = pd.merge(df_combined1, fre_wcm[['LAB_LOINC', 'Frequency']], on='LAB_LOINC', how='left')
    df_combined2.to_csv(r'../data/V15_COVID19/covid_phenotype/covid_lab_test_names_enriched.csv')

    return df_combined


def PASC_lab_frequency():
    df_pasc_lab = pd.read_excel(r'../data/V15_COVID19/covid_phenotype/PASC_Lab_LOINC.xlsx',
                               sheet_name='LOINC')
    print('df_pasc_lab.shape:', df_pasc_lab.shape)

    df_COL_fre = pd.read_excel(r'../data/V15_COVID19/covid_phenotype/lab_loinc_COL_frequency.xlsx',
                               sheet_name='lab_loinc_COL_frequency')
    print('df_COL_fre.shape:', df_COL_fre.shape)
    df_WCM_fre = pd.read_excel(r'../data/V15_COVID19/covid_phenotype/lab_loinc_WCM_frequency.xlsx',
                               sheet_name='lab_loinc_WCM_frequency')
    print('df_WCM_fre.shape:', df_WCM_fre.shape)

    df_combined1 = pd.merge(df_pasc_lab, df_COL_fre[['LAB_LOINC', 'Frequency']], left_on='LOINC', right_on='LAB_LOINC', how='left')
    df_combined2 = pd.merge(df_combined1, df_WCM_fre[['LAB_LOINC', 'Frequency']], left_on='LOINC', right_on='LAB_LOINC', how='left')
    df_combined2.to_csv(r'../data/V15_COVID19/covid_phenotype/PASC_Lab_LOINC_frquency_in_data.csv')

    return df_combined2


def read_lab_and_count_covid(dataset='COL', chunksize=100000, debug=False):
    start_time = time.time()
    print('Choose dataset:', dataset, 'chunksize:', chunksize, 'debug:', debug)
    # step 1: load covid lab test codes, may be updated by:
    # https://github.com/National-COVID-Cohort-Collaborative/Phenotype_Data_Acquisition/wiki/Latest-Phenotype
    df_covid = pd.read_csv(r'../data/V15_COVID19/covid_phenotype/covid_lab_test_names.csv')
    code_set = set(df_covid['LAB_LOINC'].to_list())

    # step 2: read lab results by chunk, due to large file size
    sasds = pd.read_sas('../data/V15_COVID19/{}/lab_result_cm.sas7bdat'.format(dataset),
                        encoding='WINDOWS-1252',
                        chunksize=chunksize,
                        iterator=True)  # 'iso-8859-1' (LATIN1) and Windows cp1252 (WLATIN1)
    # sasds = pyreadstat.read_file_in_chunks(pyreadstat.read_sas7bdat,
    #                                        '../data/V15_COVID19/{}/lab_result_cm.sas7bdat'.format(dataset),
    #                                        chunksize=chunksize)  #, multiprocess=True, num_processes=4)

    dfs = []  # holds data chunks
    dfs_covid = []
    cnt = Counter([])
    i = 0
    n_rows = 0
    n_covid_rows = 0
    patid_set = set([])
    patid_covid_set = set([])
    for chunk in sasds:  # , meta
        i += 1
        if chunk.empty:
            print("ERROR: Empty chunk! break!")
            break

        n_rows += len(chunk)
        if debug:
            dfs.append(chunk)

        if dataset == 'COL':
            chunk_covid_records = chunk.loc[chunk['LAB_LOINC'].isin(code_set), :]
            patid_set.update(chunk['PATID'])
        elif dataset == 'WCM':
            chunk_covid_records = chunk.loc[chunk['lab_loinc'].isin(code_set), :]
            patid_set.update(chunk['patid'])

        n_covid_rows += len(chunk_covid_records)
        if dataset == 'COL':
            cnt.update(chunk_covid_records['RESULT_QUAL'])
            patid_covid_set.update(chunk_covid_records['PATID'])
        elif dataset == 'WCM':
            cnt.update(chunk_covid_records['result_qual'])
            patid_covid_set.update(chunk_covid_records['patid'])

        dfs_covid.append(chunk_covid_records)
        if i == 1:
            print('chunk.shape', chunk.shape)
            print('chunk.columns', chunk.columns)

        if i % 10 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            print('len(patid_set):', len(patid_set))
            print('len(patid_covid_set):', len(patid_covid_set))

            if debug:
                print('IN DEBUG MODE, BREAK, AND DUMP!')
                break

    print('n_rows:', n_rows, 'n_covid_rows:', n_covid_rows)
    print('len(patid_set):', len(patid_set))
    print('len(patid_covid_set):', len(patid_covid_set))
    print('#chunk: ', i, 'chunk size:', chunksize)
    print('Counter:', cnt)
    dfs_covid_all = pd.concat(dfs_covid)
    print('dfs_covid_all.shape', dfs_covid_all.shape)
    print('dfs_covid_all.columns', dfs_covid_all.columns)
    dfs_covid_all.rename(columns=lambda x: x.upper(), inplace=True)
    print('dfs_covid_all.columns', dfs_covid_all.columns)
    dfs_covid_all.to_excel("{}_covid_lab_sample.xlsx".format(dataset))
    dfs_covid_all.to_csv("{}_covid_lab_sample.csv".format(dataset))

    if debug:
        dfs_all = pd.concat(dfs)
        print('dfs_all.shape', dfs_all.shape)
        print('dfs_all.columns', dfs_all.columns)
        dfs_all.rename(columns=lambda x: x.upper(), inplace=True)
        print('dfs_all.columns', dfs_all.columns)
        # dfs_all.to_excel("{}_lab_sample.xlsx".format(dataset))
        dfs_all.to_csv("{}_lab_sample.csv".format(dataset))
    print('Total Time used after dump files:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    return dfs_covid_all  # dfs_all, dfs_covid_all, meta


if __name__ == '__main__':
    # later: rename: pre_lab,   rename upper to simplify codes
    # python pre_lab.py --dataset COL 2>&1 | tee  log/count_pos_neg_COL.txt
    # python pre_lab.py --dataset WCM 2>&1 | tee  log/count_pos_neg_WCM.txt
    args = parse_args()
    print(args)
    # covid_lab_phenotyping()
    df = PASC_lab_frequency()
    # dfs_covid_all = read_lab_and_count_covid(dataset=args.dataset, debug=args.debug)
