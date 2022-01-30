import os
import shutil
import zipfile

import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

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

np.random.seed(0)
random.seed(0)


def plot_forest_for_dx():
    df = pd.read_csv(r'../data/V15_COVID19/output/character/specific/causal_effects_specific.csv')
    df_select = df.sort_values(by='hr-w', ascending=False)
    df_select = df_select.loc[df_select['hr-w-p']<=0.05, :]
    # df_select = df_select.loc[df_select['hr-w']>1, :]

    labs = []
    measure = []
    lower = []
    upper = []
    pval = []
    for key, row in df_select.iterrows():
        name = row['pasc']
        hr = row['hr-w']
        ci = stringlist_2_list(row['hr-w-CI'])
        p = row['hr-w-p']

        labs.append(name[:40])
        measure.append(hr)
        lower.append(ci[0])
        upper.append(ci[1])
        pval.append(p)

    p = EffectMeasurePlot(label=labs, effect_measure=measure, lcl=lower, ucl=upper)
    p.labels(scale='log')

    p.labels(effectmeasure='aHR')
    # p.colors(pointcolor='r')
    # '#F65453', '#82A2D3'
    # c = ['#870001', '#F65453', '#fcb2ab', '#003396', '#5494DA','#86CEFA']
    # p.colors(pointshape="s", errorbarcolor=c,  pointcolor=c)
    ax = p.plot(figsize=(9, 10), t_adjuster=0.03, max_value=13, min_value=0.5, size=5, decimal=2)
    # plt.title(drug_name, loc="right", x=.7, y=1.045) #"Random Effect Model(Risk Ratio)"
    # plt.title('pasc', loc="center", x=0, y=0)
    # plt.suptitle("Missing Data Imputation Method", x=-0.1, y=0.98)
    # ax.set_xlabel("Favours Control      Favours Haloperidol       ", fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()
    output_dir = r'../data/V15_COVID19/output/character/specific/'
    check_and_mkdir(output_dir)
    plt.savefig(output_dir + 'hr_forest.png', bbox_inches='tight', dpi=600)
    plt.savefig(output_dir + 'hr_forest.pdf', bbox_inches='tight', transparent=True)
    plt.show()
    print()
    # plt.clf()
    # plt.close()


if __name__ == '__main__':
    plot_forest_for_dx()
