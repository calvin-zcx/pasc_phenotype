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


def covid_lab_frequency_in_data():
    df_pasc_lab = pd.read_excel(r'../data/V15_COVID19/covid_phenotype/COVID_LOINC_all.xlsx', sheet_name='Clean')
    print('df_pasc_lab.shape:', df_pasc_lab.shape)

    df_fre = pd.read_csv(r'../data/V15_COVID19/covid_phenotype/all_lab_LOINC_merge_result.csv')
    print('df_fre.shape:', df_fre.shape)
    df_combined = pd.merge(df_pasc_lab, df_fre, left_on='loinc_num', right_on='LOINC', how='left')
    df_combined.to_csv(r'../data/V15_COVID19/covid_phenotype/COVID_LOINC_all_with_frequency.csv')

    return df_combined


def covid_lab_add_comments():
    df_pasc_lab = pd.read_excel(r'../data/V15_COVID19/covid_phenotype/COVID_LOINC_all.xlsx', sheet_name='Clean')
    print('df_pasc_lab.shape:', df_pasc_lab.shape)

    df_fre = pd.read_csv(r'../data/V15_COVID19/covid_phenotype/Codes in RECOVER but not CDC list by Kristin.csv')
    print('df_fre.shape:', df_fre.shape)
    df_combined = pd.merge(df_pasc_lab, df_fre, left_on='loinc_num', right_on='LOINC', how='left')
    df_combined.to_csv(r'../data/V15_COVID19/covid_phenotype/_COVID_LOINC_all_with_comments.csv')

    return df_combined


if __name__ == '__main__':
    # build covid phenotyping list
    args = parse_args()
    print(args)
    # covid_lab_phenotyping()
    # df = PASC_lab_frequency()
    # df = covid_lab_frequency_in_data()
    df = covid_lab_add_comments()
