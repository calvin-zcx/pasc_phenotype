import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
from evaluation import *
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from PSModels import ml
from misc import utils
import itertools
import functools
from tqdm import tqdm
import datetime
import seaborn as sns

print = functools.partial(print, flush=True)


if __name__ == "__main__":
    # Load index information
    with open(r'../data/mapping/icd_pasc_mapping.pkl', 'rb') as f:
        icd_pasc = pickle.load(f)
        print('Load ICD-10 to PASC mapping done! len(icd_pasc):', len(icd_pasc))
        record_example = next(iter(icd_pasc.items()))
        print('e.g.:', record_example)

    with open(r'../data/mapping/pasc_index_mapping.pkl', 'rb') as f:
        pasc_encoding = pickle.load(f)
        print('Load PASC to encoding mapping done! len(pasc_encoding):', len(pasc_encoding))
        record_example = next(iter(pasc_encoding.items()))
        print('e.g.:', record_example)

    rlist = []
    for i, pasc in tqdm(enumerate(pasc_encoding.keys(), start=1), total=len(pasc_encoding)):
        # bulid specific cohorts:
        infname = '../data/{}/output/character/outcome/DX-{}-new/{}-{}-evaluation_ps-iptw.csv'.format(
                'V15_COVID19', 'all', i, pasc)
        df = pd.read_csv(infname)
        rlist.append(df)

    dfps = pd.concat(rlist, ignore_index=True)
    # dfps.reset_index(drop=True)
    print(dfps.describe())

    ax = plt.subplot(111)
    sns.histplot(
        dfps.sample(frac=0.1), x="ps", hue="covid", #element="step",
        stat="percent", common_norm=False, bins=50, kde=True
    )
    plt.xlim(right=1)
    # ax.set_yscale('log')
    # # plt.tight_layout()
    plt.show()

    ax2 = plt.subplot(111)
    sns.histplot(
        dfps.sample(frac=0.1), x="iptw", hue="covid",  # element="step",
        stat="percent", common_norm=False, bins=50, kde=True
    )
    # plt.xlim(right=1)
    ax2.set_yscale('log')
    # # plt.tight_layout()
    plt.show()

    # plt.title(pasc, fontsize=12)
    # plt.savefig(figout)
    # plt.close()
    print()