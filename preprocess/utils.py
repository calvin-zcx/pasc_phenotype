import sys
# for linux env.
sys.path.insert(0,'..')
import scipy
import numpy as np
import pandas as pd
import time
from collections import defaultdict
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


def check_and_mkdir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('make dir:', dirname)
    else:
        print(dirname, 'exists, no change made')


def sas_2_csv(infile, outfile):
    start_time = time.time()
    # r'../COL/lab_result_cm.sas7bdat'
    with SAS7BDAT(infile, skip_header=False) as reader:
        #     df3 = reader.to_data_frame()
        print('read:', infile)
        check_and_mkdir(outfile)
        print('dump as:', outfile)
        reader.convert_file(outfile, delimiter=',')

    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def read_sas_2_df(infile=r'../data/V15_COVID19/COL/encounter.sas7bdat'):
    # Windows-1252
    start_time = time.time()
    sasds = pd.read_sas(infile, encoding='windows-1252', chunksize=100000, iterator=True) # 'iso-8859-1' (LATIN1) and Windows cp1252 (WLATIN1)
    dfs = []  # holds data chunks
    i = 0
    for chunk in sasds:
        dfs.append(chunk)
        i += 1
        if i % 10 == 0:
            print('chunk:',i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    #     if i == 10:
    #         break
    print('#chunk: ', i)
    dfs_all = pd.concat(dfs)
    print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return dfs_all


if __name__ == '__main__':
    start_time = time.time()
    df = read_sas_2_df(infile=r'../data/V15_COVID19/COL/encounter.sas7bdat')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))