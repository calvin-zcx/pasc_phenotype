import sys
# for linux env.
sys.path.insert(0, '..')
import scipy
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import pickle
import math
import itertools
import os
import pyreadstat
from sas7bdat import SAS7BDAT
import argparse
import csv
import functools
print = functools.partial(print, flush=True)
# import joblib


def check_and_mkdir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('make dir:', dirname)
    else:
        print(dirname, 'exists, no change made')


def dump(data, filename):
    print('Try to dump data to', filename)
    check_and_mkdir(filename)
    try:
        # MemoryError for pickle.dump for a large or complex file
        with open(filename, 'wb') as fo:
            pickle.dump(data, fo)
        print('Dump Done by pickle.dump! Saved as:', filename)
    except Exception as e:
        print(e)
        # print('Try to use joblib.dump(data, filename) and loading by joblib.load(filename)')
        # joblib.dump(data, filename + '.joblib')
        # print('Dump done by joblib.dump! Saved as:', filename + '.joblib')
        print('Try to split file into two and dump')
        data1 = dict(list(data.items())[:len(data) // 2])
        data2 = dict(list(data.items())[len(data) // 2:])
        with open(filename+'-part1', 'wb') as fo:
            pickle.dump(data1, fo)
            print('Dump Done by pickle.dump! Saved as:', filename+'-part1')
        with open(filename+'-part2', 'wb') as fo:
            pickle.dump(data2, fo)
            print('Dump Done by pickle.dump! Saved as:', filename+'-part2')


def load(filename):
    start_time = time.time()
    print('Try to load data file', filename)
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print('Load done by pickle.load! len(data):', len(data),
              'Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return data
    except Exception as e:
        print(e)
        print('Try to load two parts:')
        with open(filename + '-part1', 'rb') as f:
            data = pickle.load(f)
            print('load {}-part1 done, len:{}'.format(filename, len(data)))
        with open(filename + '-part2', 'rb') as f:
            data2 = pickle.load(f)
            print('load {}-part2 done, len:{}'.format(filename, len(data2)))
        data.update(data2)
        print('Load and combine data done, len(data):', len(data),
              'Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return data


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


def read_sas_2_df(infile, chunksize=100000, encoding='WINDOWS-1252', column_name='upper'):
    # Windows-1252
    print(r"Warning: only use [read_sas_2_df(infile, chunksize, encoding, column_name)] for small files, e.g. < 1 GB")
    print('read: ', infile, 'chunksize:', chunksize, 'encoding:', encoding, 'column_name:', column_name)
    start_time = time.time()
    sasds = pd.read_sas(infile,
                        encoding=encoding,
                        chunksize=chunksize,
                        iterator=True)  # 'iso-8859-1' (LATIN1) and Windows cp1252 (WLATIN1)
    df = []  # holds data chunks
    i = 0
    n_rows = 0
    for chunk in sasds:
        i += 1
        if chunk.empty:
            print("ERROR: Empty chunk! break!")
            break
        n_rows += len(chunk)
        df.append(chunk)
        if i % 10 == 0:
            print('chunk:', i, 'time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    print('#chunk: ', i, 'chunk size:', chunksize, 'n_rows:', n_rows)
    df = pd.concat(df)
    print('df.shape:', df.shape)
    print('df.columns:', df.columns)
    if column_name == 'upper':
        df.rename(columns=lambda x : x.upper(), inplace=True)
        print('df.columns:', df.columns)
    elif column_name == 'lower':
        df.rename(columns=lambda x : x.lower(), inplace=True)
        print('df.columns:', df.columns)
    else:
        print('keep original column name')

    print('read_sas_2_df Done! Total Time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    return df


def stringlist_2_list(s):
    r = s.strip('][').replace(',', ' ').split()
    r = list(map(float, r))
    return r


def stringlist_2_str(s, percent=False, digit=-1):
    r = s.strip('][').replace(',', ' ').split()
    r = list(map(float, r))
    if percent:
        r = [x * 100 for x in r]

    if digit == 0:
        rr = ','.join(['{:.0f}'.format(x) for x in r])
    elif digit == 1:
        rr = ','.join(['{:.1f}'.format(x) for x in r])
    elif digit == 2:
        rr = ','.join(['{:.2f}'.format(x) for x in r])
    elif digit == 3:
        rr = ','.join(['{:.1f}'.format(x) for x in r])
    else:
        rr = ','.join(['{}'.format(x) for x in r])
    return rr

if __name__ == '__main__':
    start_time = time.time()
    df = read_sas_2_df(infile=r'../data/V15_COVID19/COL/encounter.sas7bdat')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))