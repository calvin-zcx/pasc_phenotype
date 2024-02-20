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
from sklearn.preprocessing import SplineTransformer

print = functools.partial(print, flush=True)

if __name__ == "__main__":
    sites = ['wcm']
    site = 'wcm'
    f1 = r'../data/recover/output/{}/cohorts_covid_posOnly18base_{}.pkl'.format(site, site)
    f2 = r'../data/recover/output/{}/cohorts_covid_posOnly18base_{}_lab-dx.pkl'.format(site, site)

    id_data1 = utils.load(f1)
    id_data2 = utils.load(f2)

    neg_pats = {k: id_data1[k][4] for k in set(id_data1) - set(id_data2)}
    miss_pats = {k: [id_data2[k][0], id_data2[k][4]] for k in set(id_data2) - set(id_data1)}
