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


def add_rwd_to_neurologic_code_list():
    df_rwd = pd.read_excel(r'../data/mapping/PASC_Adult_Combined_List_submit_withRWEV3.xlsx',
                           sheet_name=r'PASC Screening List')

    df_neuro = pd.read_excel(r'../data/mapping/Neurologic PASC Code List_3_21-Chengxi.xlsx',
                             sheet_name=r'Neurologic Codes')

    df_combined = pd.merge(df_neuro, df_rwd[["ICD-10-CM Code",
                                             "dx code Covid Cohort",
                                             "total Covid Cohort",
                                             "no. in positive group Covid Cohort",
                                             "no. in negative group Covid Cohort",
                                             "ratio Covid Cohort",
                                             "dx code Covid Cohort NegNoCovid",
                                             "total Covid Cohort NegNoCovid",
                                             "no. in positive group Covid Cohort NegNoCovid",
                                             "no. in negative group Covid Cohort NegNoCovid",
                                             "ratio Covid Cohort NegNoCovid",
                                             "dx code Oneflorida Covid Cohort NegNoCovid",
                                             "total Oneflorida Covid Cohort NegNoCovid",
                                             "no. in positive group Oneflorida Covid Cohort NegNoCovid",
                                             "no. in negative group Oneflorida Covid Cohort NegNoCovid",
                                             "ratio Oneflorida Covid Cohort NegNoCovid"]],
                           left_on='ICD-10-CM Code',
                           right_on="ICD-10-CM Code", how='left')
    df_combined.to_csv(r'../data/mapping/Neurologic PASC Code List_3_21-Chengxi_helper.csv')

    return df_combined


if __name__ == '__main__':
    start_time = time.time()

    add_rwd_to_neurologic_code_list()

    print('Done! Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
