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

    sites = ['mcw', 'nebraska', 'utah', 'utsw',
             'wcm', 'montefiore', 'mshs', 'columbia', 'nyu',
             'ufh', 'usf', 'nch', 'miami',  # 'emory',
             'pitt', 'psu', 'temple', 'michigan',
             'ochsner', 'ucsf', 'lsu',
             'vumc']

    site_month = {}
    for ith, site in tqdm(enumerate(sites)):
        print('Loading: ', site)
        data_file = r'../data/recover/output/{}/matrix_cohorts_covid_4manuNegNoCovidV2age18_boolbase-nout-withAllDays-withPreg_{}.csv'.format(
            site,
            site)
        # Load Covariates Data
        print('Load data covariates file:', data_file)
        df = pd.read_csv(data_file, dtype={'patid': str, 'site': str, 'zip': str}, parse_dates=['index date'])
        # because a patid id may occur in multiple sites. patid were site specific
        print('all df.shape:', df.shape)
        df_label = df['covid']
        df = df.loc[df_label==1, :]
        print('positive df.shape:', df.shape)

        if df.shape[0] == 0:
            print('0 selected patients in', site, 'skip and continue')
            continue

        yearmonth_column_names = [
            "YM: March 2020", "YM: April 2020", "YM: May 2020", "YM: June 2020", "YM: July 2020",
            "YM: August 2020", "YM: September 2020", "YM: October 2020", "YM: November 2020", "YM: December 2020",
            "YM: January 2021", "YM: February 2021", "YM: March 2021", "YM: April 2021", "YM: May 2021",
            "YM: June 2021", "YM: July 2021", "YM: August 2021", "YM: September 2021", "YM: October 2021",
            "YM: November 2021", "YM: December 2021", "YM: January 2022",
            "YM: February 2022", "YM: March 2022", "YM: April 2022", "YM: May 2022",
            "YM: June 2022", "YM: July 2022", "YM: August 2022", "YM: September 2022",
            "YM: October 2022", "YM: November 2022", "YM: December 2022", "YM: January 2023",
            "YM: February 2023",
        ]

        ym = df[yearmonth_column_names].sum()
        site_month[site] = ym

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.plot(ym, marker='o', linestyle='-')
        ax.set_ylabel('Covid case per month')
        ax.set_title(site)
        ax.set_xticklabels([x[4:] for x in yearmonth_column_names], rotation = 45)
        figout = r'output/figure/dynamics/{}-{}.jpeg'.format(ith, site)
        utils.check_and_mkdir(figout)

        plt.savefig(figout)
        plt.close()

    print('Done! Total Time used:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
