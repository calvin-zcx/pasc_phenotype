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

    df = pd.read_excel(
        r'../data/V15_COVID19/output/character/outcome/DX-all-new/causal_effects_specific_dx_insight-MultiPval-DXMEDALL.xlsx',
        sheet_name='dx')
    df_select = df.sort_values(by='hr-w', ascending=False)
    df_select = df_select.loc[df_select['selected'] == 1, :]

    for i, pasc in tqdm(enumerate(pasc_encoding.keys(), start=1), total=len(pasc_encoding)):
        # bulid specific cohorts:
        infname = '../data/{}/output/character/outcome/DX-{}-new/{}-{}-evaluation_ps-iptw.csv'.format(
                'V15_COVID19', 'all', i, pasc)
        df = pd.read_csv(infname)
        rlist.append(df)

    dfps = pd.concat(rlist, ignore_index=True)
    # dfps.reset_index(drop=True)
    print(dfps.describe())
    print(dfps.loc[dfps['covid']==1, ['ps', 'iptw']].describe())
    print(dfps.loc[dfps['covid']==0, ['ps', 'iptw']].describe())

    dfps_sample = dfps.sample(frac=0.01)

    dfps_sample['ps'] = np.clip(dfps_sample['ps'],
                                  a_min=np.quantile(dfps_sample['ps'], 0.01),
                                  a_max=np.quantile(dfps_sample['ps'], 0.99))

    ax = plt.subplot(111)
    sns.histplot(
        dfps_sample, x="ps", hue="covid", #element="step",
        stat="proportion", common_norm=False, bins=50, kde=True
    )
    # plt.xlim(right=1)
    # ax.set_yscale('log')
    # # plt.tight_layout()
    plt.show()

    dfps_sample['iptw'] = np.clip(dfps_sample['iptw'],
                                  a_min=np.quantile(dfps_sample['iptw'], 0.01),
                                  a_max=np.quantile(dfps_sample['iptw'], 0.99))
    ax2 = plt.subplot(111)
    sns.histplot(
        dfps_sample, x="iptw", hue="covid",  # element="step",
        stat="proportion", common_norm=False, bins=50, kde=True
    )
    # plt.xlim(right=1)
    # ax2.set_yscale('log')
    # # plt.tight_layout()
    plt.show()

    # plt.title(pasc, fontsize=12)
    # plt.savefig(figout)
    # plt.close()
    print()