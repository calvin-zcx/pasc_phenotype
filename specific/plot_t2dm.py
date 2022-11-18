import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib as mpl
import time

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
from collections import defaultdict
from collections import OrderedDict

print = functools.partial(print, flush=True)


def stringlist_2_flist(s):
    r = s.strip('[').replace(']', ',').replace(' ', '').split(',')
    # r = list(map(float, r))
    r = [float(x) for x in r if x != '']
    return r


def stringlist_2_datelist(s):
    r = s.strip('[').replace(']', ',').replace(' ', '').split(',')
    # r = list(map(float, r))
    r = [pd.to_datetime(x) for x in r if x != '']
    return r


if __name__ == '__main__':
    df = pd.read_csv(r'../data/V15_COVID19/output/character/cp_dm/HbA1C_df.csv')
    windows = 30
    right = 720
    left = -right

    for covid in [0, 1]:

        dflab = df.loc[(df['covid'] == covid) & ((df['flag_diabetes_lab'] == 1) | (df['flag_diabetes_dx_lab'] == 1) | (
                    df['flag_diabetes_med_lab'] == 1)), :]
        print('covid', covid, 'len(df_select)', len(dflab))
        day_vlist = defaultdict(list)
        for key, row in dflab.iterrows():
            index_date = pd.to_datetime(row['index date'])
            if pd.isna(row['HbA1C_value_sequence_all']):
                continue
            va1c = stringlist_2_flist(row['HbA1C_value_sequence_all'])
            vdate = stringlist_2_datelist(row['HbA1C_value_time_all'])
            for a1c, d in zip(va1c, vdate):
                delta = (d - index_date).days
                # day_vlist[delta].append(a1c)
                if pd.notna(a1c):
                    day_vlist[(delta // windows + 1) * windows].append(a1c)

        day_vlist = OrderedDict(sorted(day_vlist.items()))
        results = []
        for delta, val in day_vlist.items():
            results.append([delta, len(val), np.median(val), np.quantile(val, 0.25), np.quantile(val, 0.75), np.mean(val),
                            np.min(val), np.max(val), np.std(val)])

        df_result = pd.DataFrame(results, columns=['day', 'n', 'median', 'lower', 'upper', 'mean', 'min', 'max', 'std'])
        df_result = df_result.sort_values(by=['day'])

        df_period = df_result.loc[(df_result["day"] >= left) & (df_result["day"] <= right)]
        ax1 = sns.lineplot(data=df_period, x="day", y="median",
                           label=('COVID Pos+' if covid else 'COVID Neg-') + ' ({})'.format(len(dflab)))
        # ax2 = sns.lineplot(data=df_period, x="day", y="lower", color='grey', linestyle='--')
        # ax3 = sns.lineplot(data=df_period, x="day", y="upper", color='grey', linestyle='--')

        plt.fill_between(df_period['day'], df_period['lower'], df_period['upper'],
                         alpha=0.2, interpolate=True)

        # plt.xlim(left, right)
        plt.axhline(y=6.5, color='r', linestyle='--')
        plt.axvline(x=0, color='r', linestyle='--')

    plt.ylim(5, 8.5)
    plt.xlabel('Days since infection')
    plt.ylabel('Median A1C (1st Q, 3rd Q)')
    plt.title('A1C trajectory around COVID-19 infection', fontsize=12)
    plt.savefig(r'../data/V15_COVID19/output/character/cp_dm/a1c_trajectory/lab+_only_{}.png'.format(right))
    # plt.close()
    plt.show()
