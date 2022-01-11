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


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess-transform sas 2 csv')
    parser.add_argument('--input', help='input sas file')  # default=r'../data/V15_COVID19/COL/demographic.csv',
    parser.add_argument('--output', help='output csv')
    args = parser.parse_args()
    return args


# --input ../data/V15_COVID19/COL/encounter.sas7bdat --output ../data/V15_COVID19/COL_csv/encounter.csv
if __name__ == '__main__':
    start_time = time.time()
    args = parse_args()
    print(args)
    utils.sas_2_csv(args.input, args.output)
    print('Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))