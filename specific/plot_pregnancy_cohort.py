import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import re

import numpy as np
import csv
from collections import Counter, defaultdict
import pandas as pd
from misc.utils import check_and_mkdir, stringlist_2_str, stringlist_2_list
from scipy import stats
import re
import itertools
import functools
import random
import seaborn as sns
import time
from tqdm import tqdm
from misc import utils

if __name__ == '__main__':
    start_time = time.time()

    df = pd.read_csv('preg_pos_neg.csv', dtype={'patid': str, 'site': str, 'zip': str},
                     parse_dates=['index date', 'flag_delivery_date', 'flag_pregnancy_start_date',
                                  'flag_pregnancy_end_date'])
    print('all df.shape:', df.shape)
    zz

    df = df.loc[df['covid']==0, :]
    print('covid positive df.shape:', df.shape)

    days_since_preg = (df['index date'] - df['flag_pregnancy_start_date']).apply(lambda x: x.days)
    sns.displot(days_since_preg)
    plt.show()



    days_between_deliv = (df['flag_delivery_date'] - df['index date']).apply(lambda x: x.days)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.displot(days_between_deliv, kde=True)
    plt.xlabel('Delivery date - Infection date (Days)', fontsize=10)
    plt.tight_layout()
    plt.show()

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
