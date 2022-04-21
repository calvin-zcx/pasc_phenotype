import os
import sys

# for linux env.
sys.path.insert(0, '..')
import pandas as pd
import numpy as np
import argparse
import time
import random
import pickle
import ast
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
from datetime import datetime
import functools
import seaborn as sns

print = functools.partial(print, flush=True)
from misc import utils
from lifelines import KaplanMeierFitter, CoxPHFitter, AalenJohansenFitter
from lifelines.statistics import survival_difference_at_fixed_point_in_time_test, proportional_hazard_test, logrank_test
from lifelines.plotting import add_at_risk_counts
from lifelines.utils import k_fold_cross_validation
from PRModels import ml
import matplotlib.pyplot as plt


def plot_person_pasc_counts():
    dfnyc = pd.read_csv(r'output/dataset/INSIGHT/stats/person_pasc_counts_INSIGHT.csv')
    dfflor = pd.read_csv(r'output/dataset/OneFlorida/stats/person_pasc_counts_OneFlorida.csv')

    # dfnyc['count'].hist(bins=26, density=1)
    # dfflor['count'].hist(bins=26, density=1)
    # plt.show()

    # nnyc, binsnyc, patchesnyc = plt.hist(x=dfnyc['count'], bins=range(26),
    #                                      color="#F65453", alpha=0.5, density=1, label='NYC', rwidth=1)
    # nflor, binsflor, patchesflor = plt.hist(x=dfflor['count'], bins=range(26),
    #                                         color='#0504aa', alpha=0.5, density=1, label='Florida', rwidth=1)

    sns.histplot(data=dfnyc, x="count", stat='probability', discrete=True, color="red", label='NYC', shrink=.8, fill=False )
    sns.histplot(data=dfflor, x="count", stat='probability', discrete=True, color='#4895ef', label='Florida', shrink=.8, fill=False) # #0504aa

    # sns.ecdfplot(data=dfnyc, x="count", stat="proportion", complementary=True, color="red", label='NYC')
    # sns.ecdfplot(data=dfflor, x="count",  stat="proportion", complementary=True, color="#4895ef", label='NYC')

    # plt.yscale('log')
    plt.ylim(top=1.1)
    # plt.grid(axis='y', alpha=0.75)
    plt.xlabel('No. of PASC Per Patient')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    plot_person_pasc_counts()

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
