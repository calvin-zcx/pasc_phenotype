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

print = functools.partial(print, flush=True)
import zepid
from zepid.graphics import EffectMeasurePlot
import shlex

np.random.seed(0)
random.seed(0)
from misc import utils


def extract_n3c_results():
    infile = r'../data/recover/output/results/Paxlovid-n3c-all-narrow/causal_effects_specific-snapshot-4n3c.xlsx'
    infile = r'../data/recover/output/results/Paxlovid-pcornet-all-narrow/causal_effects_specific-snapshot-4n3c.xlsx'

    df = pd.read_excel(
        infile,
        sheet_name='Sheet1')

    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    pasc_row = []

    nabsv = []
    ncumv = []

    for key, row in df.iterrows():
        pasc = row['pasc']
        hr = row['hr-w']
        ci = stringlist_2_list(row['hr-w-CI'])
        p = row['hr-w-p']

        cif1 = stringlist_2_list(row['cif_1_w'])[-1]
        cif1_ci = [stringlist_2_list(row['cif_1_w_CILower'])[-1],
                   stringlist_2_list(row['cif_1_w_CIUpper'])[-1]]

        cif0 = stringlist_2_list(row['cif_0_w'])[-1]
        cif0_ci = [stringlist_2_list(row['cif_0_w_CILower'])[-1],
                   stringlist_2_list(row['cif_0_w_CIUpper'])[-1]]

        # print('{}\t{:.3f} ({:.3f}, {:.3f})\t{:.3f} ({:.3f}, {:.3f})'.format(pasc,
        #                                                                     cif1, cif1_ci[0], cif1_ci[1],
        #                                                                     cif0, cif0_ci[0], cif0_ci[1],
        #                                                                     ))
        # print('aHR')
        print('{}\t{:.3f} ({:.3f}, {:.3f})\t{:.3e}'.format(pasc, hr,ci[0], ci[1], p))


if __name__ == '__main__':
    extract_n3c_results()
