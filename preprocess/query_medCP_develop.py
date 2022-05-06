import sys

# for linux env.
sys.path.insert(0, '..')
import time
import pickle
import argparse
import os
import random
import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
from misc import utils
import itertools
import functools
from tqdm import tqdm

print = functools.partial(print, flush=True)


if __name__ == "__main__":
    dir_name = r'C:/Users/zangc/Documents/Boston/workshop/2021-PASC/query_specific/'
    # med_file = 'Neurologic_Medications_ASN_v0.xlsx'
    med_file = 'Pulmonary_Medications_es_v0.xlsx'

    df_enc = pd.read_csv(dir_name + 'med_encounter_zilong.csv')
    df_med = pd.read_excel(dir_name + med_file)
    df = pd.merge(df_med, df_enc, left_on='rxnorm', right_on='rxnorm', how='left')

    df.to_csv(dir_name + med_file + '-aux.csv', index=False)