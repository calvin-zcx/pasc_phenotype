import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
# from evaluation import *
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
# from PSModels import ml
from misc import utils
import itertools
import functools
from tqdm import tqdm
import datetime
import seaborn as sns
from sklearn.preprocessing import SplineTransformer

print = functools.partial(print, flush=True)


if __name__ == "__main__":
    start_time = time.time()
    data_file = r'../data/recover/output/pregnant_female_18-50.csv'
    print('Load data covariates file:', data_file)
    df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'flag_delivery_date', 'flag_pregnancy_start_date',
                                  'flag_pregnancy_end_date'])
    print('df.shape', df.shape)

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
