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
# import pyreadstat
# from sas7bdat import SAS7BDAT
import argparse
import csv
import functools
import requests
import json
import pgzip

print = functools.partial(print, flush=True)
# import joblib
import re
# from datetime import datetime
from collections import Counter
from scipy import stats
from datetime import datetime, date


def pformat_symbol(p):
    if p <= 0.001:
        sigsym = '$^{***}$'
        p_format = '{:.1e}'.format(p)
    elif p <= 0.01:
        sigsym = '$^{**}$'
        p_format = '{:.3f}'.format(p)
    elif p <= 0.05:
        sigsym = '$^{*}$'
        p_format = '{:.3f}'.format(p)
    else:
        sigsym = '$^{ns}$'
        p_format = '{:.3f}'.format(p)
    return p_format, sigsym


def boot_matrix(z, B):
    """Bootstrap sample
    Returns all bootstrap samples in a matrix"""
    z = np.array(z).flatten()
    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]


def bootstrap_mean_ci(x, B=1000, alpha=0.05):
    n = len(x)
    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    sampling_distribution = xboot.mean(axis=1)
    quantile_confidence_interval = np.percentile(sampling_distribution, q=(100 * alpha / 2, 100 * (1 - alpha / 2)))
    std = sampling_distribution.std()
    # if plot:
    #     plt.hist(sampling_distribution, bins="fd")
    return quantile_confidence_interval, std


def bootstrap_mean_pvalue(x, expected_mean=0., B=1000):
    """
    Ref:
    1. https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#cite_note-:0-1
    2. https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    3. https://github.com/mayer79/Bootstrap-p-values/blob/master/Bootstrap%20p%20values.ipynb
    4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html?highlight=one%20sample%20ttest
    Bootstrap p values for one-sample t test
    Returns boostrap p value, test statistics and parametric p value"""
    n = len(x)
    orig = stats.ttest_1samp(x, expected_mean)
    # Generate boostrap distribution of sample mean
    x_boots = boot_matrix(x - x.mean() + expected_mean, B=B)
    x_boots_mean = x_boots.mean(axis=1)
    t_boots = (x_boots_mean - expected_mean) / (x_boots.std(axis=1, ddof=1) / np.sqrt(n))
    p = np.mean(t_boots >= orig[0])
    p_final = 2 * min(p, 1 - p)
    # Plot bootstrap distribution
    # if plot:
    #     plt.figure()
    #     plt.hist(x_boots_mean, bins="fd")
    return p_final, orig


def bootstrap_mean_pvalue_2samples(x, y, equal_var=False, B=1000):
    """
    Bootstrap hypothesis testing for comparing the means of two independent samples
    Ref:
    1. https://en.wikipedia.org/wiki/Bootstrapping_(statistics)#cite_note-:0-1
    2. https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    3. https://github.com/mayer79/Bootstrap-p-values/blob/master/Bootstrap%20p%20values.ipynb
    4. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html?highlight=one%20sample%20ttest
    Bootstrap p values for one-sample t test
    Returns boostrap p value, test statistics and parametric p value"""
    n = len(x)
    orig = stats.ttest_ind(x, y, equal_var=equal_var)
    pooled_mean = np.concatenate((x, y), axis=None).mean()

    xboot = boot_matrix(x - x.mean() + pooled_mean,
                        B=B)  # important centering step to get sampling distribution under the null
    yboot = boot_matrix(y - y.mean() + pooled_mean, B=B)
    sampling_distribution = stats.ttest_ind(xboot, yboot, axis=1, equal_var=equal_var)[0]

    if np.isnan(orig[1]):
        p_final = np.nan
    else:
        # Calculate proportion of bootstrap samples with at least as strong evidence against null
        p = np.mean(sampling_distribution >= orig[0])
        # RESULTS
        # print("p value for null hypothesis of equal population means:")
        # print("Parametric:", orig[1])
        # print("Bootstrap:", 2 * min(p, 1 - p))
        p_final = 2 * min(p, 1 - p)

    return p_final, orig


def ndc_normalization(x):
    # https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html#s6_0
    # "You can download the algorithm used for normalizing NDC codes (RTF document)."
    # https://www.nlm.nih.gov/research/umls/rxnorm/docs/techdoc.html#sat
    if len(x) - len(x.replace('-', '')) == 2:  # If NDC string contains 2 dashes
        a, b, c = x.split('-')
        na, nb, nc = [len(t) for t in x.split('-')]  # a-b-c format
        if (na == 6) and (nb == 4) and (nc == 2):
            ndc = a[1:] + b + c
        elif (na == 6) and (nb == 4) and (nc == 1):
            ndc = a[1:] + b + '0' + c
        elif (na == 6) and (nb == 3) and (nc == 2):
            ndc = a[1:] + '0' + b + c
        elif (na == 6) and (nb == 3) and (nc == 1):
            ndc = a[1:] + '0' + b + '0' + c
        elif (na == 5) and (nb == 4) and (nc == 2):  # MMX, GS, CVX
            ndc = a + b + c
        elif (na == 5) and (nb == 4) and (nc == 1):
            ndc = a + b + '0' + c
        elif (na == 5) and (nb == 3) and (nc == 2):  # MTHSPL
            ndc = a + '0' + b + c
        elif (na == 4) and (nb == 4) and (nc == 2):
            ndc = '0' + a + b + c
        else:
            ndc = ''
    elif (len(x) == 11) and ('-' not in x):  # rxnorm, NDDF, MMSL
        ndc = x
    elif (len(x) == 12) and ('-' not in x):  # and (x[0] == '0'): #VANDF, some initial with 9
        ndc = x[1:]
    else:
        ndc = ''

    ndc = ndc.replace('*', '0')

    if re.sub(r"[0123456789]", '', ndc):
        ndc = ''

    return ndc


def _clean_path_name_(s, maxlen=50):
    s = s.replace(':', '-').replace('/', '-')
    s_trunc = (s[:maxlen] + '..') if len(s) > maxlen else s
    return s_trunc


def check_and_mkdir(path):
    dirname = os.path.dirname(path)
    if dirname == '':
        dirname = '.'

    if not os.path.exists(dirname):
        os.makedirs(dirname)
        print('make dir:', dirname)
    else:
        print(dirname, 'exists, no change made')


def split_dict_data_and_dump(infile, chunk=4):
    start_time = time.time()
    with open(infile, 'rb') as f:
        data = pickle.load(f)
    print('Load done by pickle.load! len(data):', len(data),
          'Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
    print('Try to split file into chunk={} and dump'.format(chunk))
    step = len(data) // chunk
    for i in range(chunk):
        left = i * step
        right = (i + 1) * step
        if i == (chunk - 1):
            right = len(data)
        data_part = dict(list(data.items())[left:right])
        with open(infile + '-part{}'.format(i + 1), 'wb') as fo:
            pickle.dump(data_part, fo)
            print('Dump Done by pickle.dump! Saved as:', infile + '-part{}'.format(i + 1))
            del data_part

    print('Split dict into pieces done!')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


def dump_compressed(data, filename, num_cpu=8, blocksize=2 * 10 ** 8):
    start_time = time.time()
    print('Try to dump data to {}'.format(filename), ', compressed by gzip')
    check_and_mkdir(filename)
    if not filename.lower().endswith(('.gz', '.gzip')):
        filename += '.gz'

    try:
        with pgzip.open(filename, "wb", thread=num_cpu, blocksize=blocksize) as fw:
            pickle.dump(data, fw)

        print('Dump Done by dump_compressed! Saved as:', filename)
        print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

    except Exception as e:
        print(e)


def dump(data, filename, chunk=2, chunk_must=False):
    print('Try to dump data to', filename)
    check_and_mkdir(filename)
    try:
        if not chunk_must:
            # MemoryError for pickle.dump for a large or complex file
            with open(filename, 'wb') as fo:
                pickle.dump(data, fo)
            print('Dump Done by pickle.dump! Saved as:', filename)
        else:
            print('Try to split file into chunk={} and dump'.format(chunk))
            step = len(data) // chunk
            for i in range(chunk):
                left = i * step
                right = (i + 1) * step
                if i == (chunk - 1):
                    right = len(data)
                data_part = dict(list(data.items())[left:right])
                with open(filename + '-part{}'.format(i + 1), 'wb') as fo:
                    pickle.dump(data_part, fo)
                    print('Dump Done by pickle.dump! Saved as:', filename + '-part{}'.format(i + 1))
    except Exception as e:
        print(e)
        # print('Try to use joblib.dump(data, filename) and loading by joblib.load(filename)')
        # joblib.dump(data, filename + '.joblib')
        # print('Dump done by joblib.dump! Saved as:', filename + '.joblib')
        print('Try to split file into chunk={} and dump'.format(chunk))
        step = len(data) // chunk
        for i in range(chunk):
            left = i * step
            right = (i + 1) * step
            if i == (chunk - 1):
                right = len(data)
            data_part = dict(list(data.items())[left:right])
            with open(filename + '-part{}'.format(i + 1), 'wb') as fo:
                pickle.dump(data_part, fo)
                print('Dump Done by pickle.dump! Saved as:', filename + '-part{}'.format(i + 1))
        print('dump {} done!'.format(filename + '-part[1-{}]'.format(chunk)))
        if os.path.isfile(filename):
            os.remove(filename)
        else:
            # If it fails, inform the user.
            print("Error: %s file not found" % filename)
        print('remove {} done!'.format(filename))
        # data1 = dict(list(data.items())[:len(data) // 2])
        # data2 = dict(list(data.items())[len(data) // 2:])
        # with open(filename+'-part1', 'wb') as fo:
        #     pickle.dump(data1, fo)
        #     print('Dump Done by pickle.dump! Saved as:', filename+'-part1')
        # with open(filename+'-part2', 'wb') as fo:
        #     pickle.dump(data2, fo)
        #     print('Dump Done by pickle.dump! Saved as:', filename+'-part2')


def load(filename, chunk=2, num_cpu=8):
    start_time = time.time()
    print('Try to load data file', filename)

    # 2024-7-25 add compressed pickle support, using pgzip
    if filename.lower().endswith(('.gz', '.gzip')):
        print('Detect .gz file, using gzip to depress and load...')

        with pgzip.open(filename, "rb", thread=num_cpu) as fr:
            data = pickle.load(fr)
        print('Load done by pgzip.open >> pickle.load! len(data):', len(data),
              'Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
        return data
    else:
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print('Load done by pickle.load! len(data):', len(data),
                  'Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            return data
        except Exception as e:
            print(e)
            print('Try to load {} chunks:'.format(chunk))
            with open(filename + '-part1', 'rb') as f:
                data = pickle.load(f)
                print('load {}-part1 done, len:{}'.format(filename, len(data)))
            for i in range(1, chunk):
                with open(filename + '-part{}'.format(i + 1), 'rb') as f:
                    data_part = pickle.load(f)
                    print('load {}-part{} done, len:{}'.format(filename, i + 1, len(data_part)))
                    data.update(data_part)
            print('Load and combine data done, len(data):', len(data),
                  'Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
            return data


# def sas_2_csv(infile, outfile):
#     start_time = time.time()
#     # r'../COL/lab_result_cm.sas7bdat'
#     with SAS7BDAT(infile, skip_header=False) as reader:
#         #     df3 = reader.to_data_frame()
#         print('read:', infile)
#         check_and_mkdir(outfile)
#         print('dump as:', outfile)
#         reader.convert_file(outfile, delimiter=',')
#
#     print('Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))


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
        df.rename(columns=lambda x: x.upper(), inplace=True)
        print('df.columns:', df.columns)
    elif column_name == 'lower':
        df.rename(columns=lambda x: x.lower(), inplace=True)
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


def str_to_datetime(s):
    # input: e.g. '2016-10-14'
    # output: datetime.datetime(2016, 10, 14, 0, 0)
    # Other choices:
    #       pd.to_datetime('2016-10-14')  # very very slow
    #       datetime.strptime('2016-10-14', '%Y-%m-%d')   #  slow
    # ymd = list(map(int, s.split('-')))
    ymd = list(map(int, re.split(r'[-\/:.]', s)))
    assert (len(ymd) == 3) or (len(ymd) == 1)
    if len(ymd) == 3:
        assert 1 <= ymd[1] <= 12
        assert 1 <= ymd[2] <= 31
    elif len(ymd) == 1:
        ymd = ymd + [1, 1]  # If only year, set to Year-Jan.-1st
    return datetime(*ymd)


def _parse_ndc_rxnorm_api(ndc):
    # Notice: https://rxnav.nlm.nih.gov/REST/ndcstatus.json?ndc=00071015723
    # change at 2022-02-28
    r = requests.get('https://rxnav.nlm.nih.gov/REST/ndcstatus.json?ndc={}'.format(ndc))
    data = r.json()
    rx = name = ''
    if ('ndcStatus' in data) and ('rxcui' in data['ndcStatus']) and ('conceptName' in data['ndcStatus']):
        rx = data['ndcStatus']['rxcui']
        name = data['ndcStatus']['conceptName']
    return rx, name


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def tofloat(num):
    try:
        num = float(num)
        return num
    except ValueError:
        return np.nan


def clean_date_str(x):
    if isinstance(x, str):
        x = pd.to_datetime(x, errors='coerce').date()
    elif isinstance(x, pd.Timestamp):
        x = x.date()
    elif isinstance(x, date):
        x = x
    else:
        x = np.nan
    return x


def split_shell_file(fname, divide=2, skip_first=1):
    f = open(fname, 'r')
    content_list = f.readlines()
    n = len(content_list)
    n_d = np.ceil((n - skip_first) / divide)
    seg = [0, ] + [int(i * n_d + skip_first) for i in range(1, divide)] + [n]
    for i in range(divide):
        fout_name = fname.split('.')
        fout_name = ''.join(fout_name[:-1]) + '-' + str(i) + '.' + fout_name[-1]
        fout = open(fout_name, 'w')
        for l in content_list[seg[i]:seg[i + 1]]:
            fout.write(l)
        fout.close()
    print('dump done')


def split_shell_file_bydelta(fname, delta, skip_first=1):
    f = open(fname, 'r')
    content_list = f.readlines()
    n = len(content_list)
    divide = int(np.ceil((n - skip_first) / delta))

    seg = [0, ] + [int(i * delta + skip_first) for i in range(1, divide)] + [n]
    for i in range(divide):
        fout_name = fname.split('.')
        fout_name = ''.join(fout_name[:-1]) + '-' + str(i) + '.' + fout_name[-1]
        fout = open(fout_name, 'w')
        for l in content_list[seg[i]:seg[i + 1]]:
            fout.write(l)
        fout.close()
    print('dump done')


def is_mci(code):
    """
        ICD9
            331.83 Mild cognitive impairment, so stated
            294.9	 Unspecified persistent mental disorders due to conditions classified elsewhere
        ICD10
            G31.84 Mild cognitive impairment, so stated
            F09	Unspecified mental disorder due to known physiological condition

     :param code: code string to test
     :return:  true or false
     """
    assert isinstance(code, str)
    code_set = ('331.83', '294.9', 'G31.84', 'F09', '33183', '2949', 'G3184')
    return code.startswith(code_set)


def is_AD(code):
    """
    ICD9
        331.0		Alzheimer's disease
    ICD10
        G30     Alzheimer's disease
        G30.0	Alzheimer's disease with early onset
        G30.1 	Alzheimer's disease with late onset
        G30.8	Other Alzheimer's disease
        G30.9	Alzheimer's disease, unspecified

    :param code: code string to test
    :return:  true or false
    """
    assert isinstance(code, str)
    code_set = ('331.0', '3310', 'G30')
    return code.startswith(code_set)


def is_dementia(code):
    """
    ICD9
        294.10, 294.11 , 294.20, 294.21
        290.-   (all codes and variants in 290.- tree)
    ICD10
        F01.- All codes and their variants in this trees
        F02.- All codes and their variants in this trees
        F03.- All codes and their variants in this trees
    :param code: code string to test
    :return:  true or false
    """
    assert isinstance(code, str)
    code_set = ('294.10', '294.11', '294.20', '294.21', '2941', '29411', '2942', '29421')
    code_set += ('290',)
    code_set += ('F01', 'F02', 'F03')
    return code.startswith(code_set)


def load_icd_to_ccw(path):
    """ e.g. there exisit E93.50 v.s. E935.0.
            thus 5 more keys in the dot-ICD than nodot-ICD keys
            [('E9350', 2),
             ('E9351', 2),
             ('E8500', 2),
             ('E8502', 2),
             ('E8501', 2),
             ('31222', 1),
             ('31200', 1),
             ('3124', 1),
             ('F919', 1),
             ('31281', 1)]
    :param path: e.g. 'mapping/CCW_to_use_enriched.json'
    :return:
    """
    with open(path) as f:
        data = json.load(f)
        print('len(ccw_codes):', len(data))
        name_id = {x: str(i) for i, x in enumerate(data.keys())}
        id_name = {v: k for k, v in name_id.items()}
        n_dx = 0
        icd_ccwid = {}
        icd_ccwname = {}
        icddot_ccwid = {}
        for name, dx in data.items():
            n_dx += len(dx)
            for icd in dx:
                icd_no_dot = icd.strip().replace('.', '')
                if icd_no_dot:  # != ''
                    icd_ccwid[icd_no_dot] = name_id.get(name)
                    icd_ccwname[icd_no_dot] = name
                    icddot_ccwid[icd] = name_id.get(name)
        data_info = (name_id, id_name, data)
        return icd_ccwid, icd_ccwname, data_info


if __name__ == '__main__':
    start_time = time.time()
    # df = read_sas_2_df(infile=r'../data/V15_COVID19/COL/encounter.sas7bdat')
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
